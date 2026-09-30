"""Optional on-screen text (OCR) from sampled video frames.

Off by default. When on, the video is downloaded (not just the audio), frames
are sampled — at the video's cuts when ffmpeg is installed, otherwise at
``fps`` per second — near duplicate frames are dropped, and the rest are read
by one of:

* ``local``      RapidOCR (PaddleOCR models on onnxruntime). Free, offline.
* ``openai``     gpt-5-mini vision (32x32 patches, about 690 tokens per frame).
* ``anthropic``  Claude vision (about width*height/750 tokens per frame).

Costs are estimated from the published per-token prices below before any
paid call is made, so the user sees the number first.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from shortform_notes.config import Settings

logger = logging.getLogger(__name__)

# USD per 1M input tokens, from the vendors' pricing pages (checked 2026-08-24).
OPENAI_PRICE_PER_M = {"gpt-5-mini": 0.25, "gpt-5": 1.25, "gpt-5-nano": 0.05}
# Sonnet 5's $2 is an introductory rate the pricing page runs through 2026-08-31; it lists at $3
# after that, so re-check this table if the estimates start reading low.
ANTHROPIC_PRICE_PER_M = {"claude-sonnet-5": 2.0, "claude-opus-5": 5.0, "claude-haiku-4-5": 1.0}

# The two vendors count image tokens by completely different arithmetic, so each gets its own
# function rather than a single fudged per-frame constant.
#
# OpenAI's GPT-5 family bills 32x32 patches: ceil(w/32) * ceil(h/32), capped at a patch budget,
# times a per-model multiplier. This replaces the base+tile formula gpt-4o-mini used (2,833 base /
# 5,667 per tile) — those numbers do not apply to gpt-5-mini at all and are gone.
OPENAI_PATCH_SIDE = 32
OPENAI_PATCH_BUDGET = 1536  # gpt-5-mini/nano at detail=high; over budget, the image is rescaled to fit
OPENAI_PATCH_MULTIPLIER = 1.2  # gpt-5-mini and gpt-5.x; gpt-5-nano is 1.5
# Anthropic publishes width*height/750 for every current model.
ANTHROPIC_PIXELS_PER_TOKEN = 750

FRAME_MAX_SIDE = 1024  # frames are downscaled to this before OCR; enough for overlay text
DEDUPE_PIXEL_DELTA = 40  # a thumbnail pixel counts as changed when its gray value moves by at least this much
DEDUPE_CHANGED_FRACTION = 0.004  # a frame is "new" when at least this fraction of thumbnail pixels changed
FRAMES_PER_REQUEST = 8

# Contact sheets: frames are tiled into grids before they go to a vision model, which costs
# one image's tokens per GRID_COLS*GRID_ROWS frames and shows the model their order spatially.
GRID_COLS = 4
GRID_ROWS = 4
FRAMES_PER_GRID = GRID_COLS * GRID_ROWS
GRID_CELL_MAX_SIDE = 512  # per cell, so a 4x4 sheet of portrait frames lands near 1152x2048
PORTRAIT_ASPECT = 9 / 16  # short-form video is portrait; the cost estimates assume that shape

OCR_PROMPT = (
    "These are frames from a short video, in order. For each frame, transcribe every piece of "
    "on-screen text exactly as written (captions, overlays, labels, lists). Return JSON: "
    '{"frames": [{"index": 0, "text": "..."}, ...]}. Use an empty string when a frame has no text. '
    "Do not describe the image; only transcribe text."
)


@dataclass(frozen=True)
class Frame:
    seconds: float
    png: bytes  # PNG-encoded, downscaled


@dataclass(frozen=True)
class Grid:
    """A contact sheet: several frames tiled into one PNG, each cell labelled with its timestamp."""

    png: bytes
    seconds: tuple[float, ...]  # cell timestamps, row-major

    def describe(self) -> str:
        span = f"{timestamp(self.seconds[0])} to {timestamp(self.seconds[-1])}"
        return f"{len(self.seconds)} frames, {span}"


@dataclass(frozen=True)
class OcrEstimate:
    frames: int
    provider: str
    usd: float  # 0.0 for local

    def describe(self) -> str:
        if self.provider == "local":
            return f"about {self.frames} frames, free (local OCR)"
        return f"about {self.frames} frames, about ${self.usd:.3f} via {self.provider}"


# ── estimates ──────────────────────────────────────────────────────────


def frame_count(duration_seconds: float, fps: float, native_fps: float = 30.0) -> int:
    """Frames that will be sampled before de-duplication. fps=0 means every frame."""
    rate = native_fps if fps <= 0 else fps
    return max(1, math.ceil(duration_seconds * rate))


def openai_image_tokens(width: int, height: int) -> int:
    """Image tokens on the GPT-5 patch pricing.

    An image over the budget is rescaled to fit it, so clamping the patch count is an upper
    bound on the rescaled one — a few percent high on a sheet, which is the safe direction
    for a number shown to the user before they spend anything.
    """
    patches = math.ceil(width / OPENAI_PATCH_SIDE) * math.ceil(height / OPENAI_PATCH_SIDE)
    return round(min(patches, OPENAI_PATCH_BUDGET) * OPENAI_PATCH_MULTIPLIER)


def anthropic_image_tokens(width: int, height: int) -> int:
    return round(width * height / ANTHROPIC_PIXELS_PER_TOKEN)


def _fit(width: int, height: int, max_side: int) -> tuple[int, int]:
    """The image as it is actually sent: scaled down until its long side is max_side."""
    scale = min(1.0, max_side / max(width, height))
    return round(width * scale), round(height * scale)


def _image_usd(provider: str, model: str, width: int, height: int) -> float:
    if provider == "openai":
        return openai_image_tokens(width, height) * OPENAI_PRICE_PER_M.get(model, 0.25) / 1_000_000
    if provider == "anthropic":
        return anthropic_image_tokens(width, height) * ANTHROPIC_PRICE_PER_M.get(model, 2.0) / 1_000_000
    return 0.0  # local OCR is free and the CLI backends bill to a subscription


def sheet_dims() -> tuple[int, int]:
    """A full contact sheet of portrait cells: the shape the vision estimate assumes."""
    return GRID_COLS * round(GRID_CELL_MAX_SIDE * PORTRAIT_ASPECT), GRID_ROWS * GRID_CELL_MAX_SIDE


def per_frame_usd(provider: str, model: str, width: int = 720, height: int = 1280) -> float:
    return _image_usd(provider, model, *_fit(width, height, FRAME_MAX_SIDE))


def per_sheet_usd(provider: str, model: str) -> float:
    """One contact sheet at the detail vision sends it with."""
    return _image_usd(provider, model, *sheet_dims())


def estimate(duration_seconds: float, settings: Settings) -> OcrEstimate:
    frames = frame_count(duration_seconds, settings.ocr_fps)
    model = settings.ocr_openai_model if settings.ocr_provider == "openai" else settings.ocr_anthropic_model
    return OcrEstimate(frames, settings.ocr_provider, round(frames * per_frame_usd(settings.ocr_provider, model), 4))


# ── frame extraction (OpenCV; no ffmpeg binary needed) ─────────────────


class _Deduper:
    """Drops a frame that looks like the one kept before it, on a 64x64 grayscale comparison."""

    def __init__(self) -> None:
        self._last = None

    def is_new(self, bgr) -> bool:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype("float32")
        changed = 1.0 if self._last is None else float((np.abs(small - self._last) >= DEDUPE_PIXEL_DELTA).mean())
        if changed < DEDUPE_CHANGED_FRACTION:
            return False
        self._last = small
        return True


def _encode(bgr) -> bytes | None:
    """Downscale to FRAME_MAX_SIDE and PNG-encode; None when encoding fails."""
    import cv2

    h, w = bgr.shape[:2]
    scale = min(1.0, FRAME_MAX_SIDE / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if ok else None


def _extract_sync(video_path: str, fps: float) -> list[Frame]:
    import cv2  # lazy: optional dependency (opencv-python-headless)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {video_path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 1 if fps <= 0 else max(1, round(native / fps))
    frames: list[Frame] = []
    deduper = _Deduper()
    index = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if index % step == 0 and deduper.is_new(bgr):
            png = _encode(bgr)
            if png:
                frames.append(Frame(seconds=index / native, png=png))
        index += 1
    cap.release()
    return frames


def _frames_from_images_sync(images: list[bytes]) -> list[Frame]:
    import cv2
    import numpy as np

    frames: list[Frame] = []
    for index, data in enumerate(images):
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        png = _encode(bgr)
        if png:
            # A slideshow has no clock; slide N is labelled 00:0N so the existing mm:ss
            # plumbing (contact-sheet cells, scene times, agentic frame filenames) still works.
            frames.append(Frame(seconds=float(index), png=png))
    return frames


async def frames_from_images(images: list[bytes]) -> list[Frame]:
    """Carousel slides (encoded JPEG/PNG bytes, in order) as Frames, one per slide, no de-duplication."""
    return await asyncio.to_thread(_frames_from_images_sync, images)


async def extract_frames(video_path: str, fps: float) -> list[Frame]:
    return await asyncio.to_thread(_extract_sync, video_path, fps)


# ── cut-aware sampling (ffmpeg keyframes, optional) ────────────────────
#
# Fixed-rate sampling lands wherever the clock says, which on a fast-cut reel means several
# frames of one shot and none of the next. A video's keyframes sit at its cuts, so decoding
# only those follows the edit instead of the clock. ffmpeg is optional: without it, or when
# a rate was asked for explicitly, the OpenCV sampler above still runs.

FFMPEG_TIMEOUT_SECONDS = 120
MIN_KEYFRAMES = 4  # below this the keyframes are too sparse to describe the video; use the clock instead
MAX_KEYFRAMES = 600  # an all-intra video makes every frame a keyframe; decode an even spread of them


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def ffmpeg_argv(video_path: str, out_pattern: str) -> list[str]:
    """Decode only keyframes, write them as PNGs, and report each one's timestamp on stderr."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-skip_frame",
        "nokey",  # decode keyframes only: fast, and they fall on the cuts
        "-i",
        video_path,
        "-an",
        "-fps_mode",
        "passthrough",  # one output image per decoded frame, no rate conversion
        "-vf",
        "showinfo",  # prints pts_time for each frame, in output order
        "-f",
        "image2",
        out_pattern,
    ]


def parse_showinfo(stderr: str) -> list[float]:
    """Timestamps from ffmpeg's showinfo filter, in the order the frames were written."""
    return [float(m) for m in re.findall(r"pts_time:\s*([0-9]+\.?[0-9]*)", stderr)]


async def _run_ffmpeg(argv: list[str]) -> tuple[int, str]:
    """Run ffmpeg and return (returncode, stderr). The subprocess boundary; patched in tests."""
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout=FFMPEG_TIMEOUT_SECONDS)
    except TimeoutError:
        proc.kill()
        raise RuntimeError(f"ffmpeg timed out after {FFMPEG_TIMEOUT_SECONDS}s") from None
    return proc.returncode or 0, err.decode(errors="replace")


def _load_keyframes_sync(tmpdir: str, times: list[float]) -> list[Frame]:
    """Decode the PNGs ffmpeg wrote, then run the same de-duplication the fps sampler uses."""
    import cv2

    paths = sorted(str(p) for p in Path(tmpdir).glob("kf-*.png"))
    shots = [(path, times[i] if i < len(times) else float(i)) for i, path in enumerate(paths)]
    if len(shots) > MAX_KEYFRAMES:
        step = len(shots) / MAX_KEYFRAMES
        shots = [shots[int(i * step)] for i in range(MAX_KEYFRAMES)]
    frames: list[Frame] = []
    deduper = _Deduper()
    for path, seconds in shots:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None or not deduper.is_new(bgr):
            continue
        png = _encode(bgr)
        if png:
            frames.append(Frame(seconds=seconds, png=png))
    return frames


async def keyframes(video_path: str) -> list[Frame]:
    """Frames at the video's cuts, de-duplicated. Empty when ffmpeg finds nothing."""
    with tempfile.TemporaryDirectory(prefix="shortform-notes-kf-") as tmpdir:
        code, stderr = await _run_ffmpeg(ffmpeg_argv(video_path, f"{tmpdir}/kf-%05d.png"))
        # showinfo prints one pts_time per written frame, in the order they were written.
        frames = await asyncio.to_thread(_load_keyframes_sync, tmpdir, parse_showinfo(stderr))
        if code != 0 and not frames:
            raise RuntimeError(f"ffmpeg exited {code}: {stderr.strip().splitlines()[-1][:200] if stderr else ''}")
        return frames


async def sample_frames(video_path: str, settings: Settings) -> list[Frame]:
    """The frames to work from: cut-aware when ffmpeg allows it, otherwise fixed-rate.

    An explicit ``--ocr-fps`` means the user asked for a rate, so the rate is what they get.
    """
    if not settings.fps_explicit and has_ffmpeg():
        try:
            frames = await keyframes(video_path)
        except Exception as exc:  # noqa: BLE001 (ffmpeg is optional; the OpenCV sampler still works)
            logger.warning("ffmpeg keyframe sampling failed, falling back to %sfps: %s", settings.ocr_fps, exc)
        else:
            if len(frames) >= MIN_KEYFRAMES:
                logger.info("sampled %d frames at the video's cuts (ffmpeg keyframes)", len(frames))
                return frames
            logger.info("only %d keyframes; falling back to %s fps sampling", len(frames), settings.ocr_fps)
    return await extract_frames(video_path, settings.ocr_fps)


def timestamp(seconds: float) -> str:
    mm, ss = divmod(int(seconds), 60)
    return f"{mm:02d}:{ss:02d}"


def parse_timestamp(label: str) -> float | None:
    """``"01:07"`` (or ``"1:07"``, ``"00:01:07"``, ``"67"``) to seconds; None when it is not a time."""
    parts = str(label or "").strip().strip("[]").split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if not values or len(values) > 3:
        return None
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


def nearest_frame(frames: Sequence[Frame], label: str) -> Frame | None:
    """The sampled frame a scene's ``mm:ss`` label came from.

    The label is the contact-sheet cell's, which ``timestamp`` truncated to whole seconds, so the
    frame that produced it is the one whose truncated time matches; failing an exact match (a
    model rounding, or answering in seconds) the nearest one wins, ties to the earlier frame.
    """
    target = parse_timestamp(label)
    if target is None or not frames:
        return None
    exact = [f for f in frames if int(f.seconds) == int(target)]
    if exact:
        return exact[0]
    return min(frames, key=lambda f: (abs(f.seconds - target), f.seconds))


# ── contact sheets ─────────────────────────────────────────────────────


def _label(cell, text: str) -> None:
    """Burn a timestamp into the cell's top-left corner, on a filled box so it reads on any frame."""
    import cv2

    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(cell, (0, 0), (tw + 8, th + 8), (0, 0, 0), -1)
    cv2.putText(cell, text, (4, th + 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _tile_sync(frames: list[Frame], cols: int, rows: int, cell_max_side: int) -> list[Grid]:
    import cv2
    import numpy as np

    per_sheet = cols * rows
    grids: list[Grid] = []
    for start in range(0, len(frames), per_sheet):
        batch = frames[start : start + per_sheet]
        cells = []
        for frame in batch:
            img = cv2.imdecode(np.frombuffer(frame.png, dtype="uint8"), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = min(1.0, cell_max_side / max(h, w))
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            _label(img, timestamp(frame.seconds))
            cells.append(img)
        if not cells:
            continue
        # One cell size for the whole sheet (every frame shares the video's aspect ratio); pad the last row.
        ch, cw = cells[0].shape[:2]
        cells = [c if c.shape[:2] == (ch, cw) else cv2.resize(c, (cw, ch)) for c in cells]
        blank = np.zeros((ch, cw, 3), dtype="uint8")
        padded = cells + [blank] * (-len(cells) % cols)
        sheet = np.vstack([np.hstack(padded[i : i + cols]) for i in range(0, len(padded), cols)])
        ok, buf = cv2.imencode(".png", sheet)
        if ok:
            grids.append(Grid(png=buf.tobytes(), seconds=tuple(f.seconds for f in batch[: len(cells)])))
    return grids


async def tile_frames(
    frames: list[Frame], cols: int = GRID_COLS, rows: int = GRID_ROWS, cell_max_side: int = GRID_CELL_MAX_SIDE
) -> list[Grid]:
    """Tile frames into chronological contact sheets, ``cols*rows`` frames per sheet."""
    if not frames:
        return []
    return await asyncio.to_thread(_tile_sync, frames, cols, rows, cell_max_side)


# ── OCR backends ───────────────────────────────────────────────────────


def _ocr_local_sync(frames: list[Frame]) -> list[str]:
    import cv2
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR  # lazy: optional dependency

    engine = RapidOCR()
    texts: list[str] = []
    for frame in frames:
        img = cv2.imdecode(np.frombuffer(frame.png, dtype="uint8"), cv2.IMREAD_COLOR)
        result, _ = engine(img)
        texts.append(" ".join(line[1] for line in (result or []) if line[1]).strip())
    return texts


async def _ocr_local(frames: list[Frame], settings: Settings) -> list[str]:
    return await asyncio.to_thread(_ocr_local_sync, frames)


def _parse_batch(text: str, count: int) -> list[str]:
    try:
        data = json.loads(text or "{}")
    except json.JSONDecodeError:
        return [""] * count
    by_index = {int(f.get("index", i)): str(f.get("text") or "") for i, f in enumerate(data.get("frames") or [])}
    return [by_index.get(i, "") for i in range(count)]


async def _ocr_openai(frames: list[Frame], settings: Settings) -> list[str]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    out: list[str] = []
    for start in range(0, len(frames), FRAMES_PER_REQUEST):
        batch = frames[start : start + FRAMES_PER_REQUEST]
        content = [{"type": "text", "text": OCR_PROMPT}]
        for i, frame in enumerate(batch):
            b64 = base64.b64encode(frame.png).decode()
            content.append({"type": "text", "text": f"frame {i}"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}})
        response = await client.chat.completions.create(
            model=settings.ocr_openai_model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_completion_tokens=6000,  # `max_tokens` is rejected by gpt-5; this also covers reasoning tokens
        )
        out.extend(_parse_batch(response.choices[0].message.content or "", len(batch)))
    return out


async def _ocr_anthropic(frames: list[Frame], settings: Settings) -> list[str]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    out: list[str] = []
    for start in range(0, len(frames), FRAMES_PER_REQUEST):
        batch = frames[start : start + FRAMES_PER_REQUEST]
        content: list[dict] = []
        for i, frame in enumerate(batch):
            content.append({"type": "text", "text": f"frame {i}"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(frame.png).decode(),
                    },
                }
            )
        content.append({"type": "text", "text": OCR_PROMPT})
        response = await client.messages.create(
            model=settings.ocr_anthropic_model,
            max_tokens=4000,
            messages=[{"role": "user", "content": content}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "frames": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"index": {"type": "integer"}, "text": {"type": "string"}},
                                    "required": ["index", "text"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["frames"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined to read these frames")
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        out.extend(_parse_batch(text, len(batch)))
    return out


_BACKENDS = {"local": "_ocr_local", "openai": "_ocr_openai", "anthropic": "_ocr_anthropic"}


def merge_text(frames: list[Frame], texts: list[str]) -> str | None:
    """Timestamped lines, consecutive duplicates collapsed. None when nothing was read."""
    lines: list[str] = []
    previous = ""
    for frame, text in zip(frames, texts, strict=False):
        clean = " ".join(text.split())
        if clean and clean != previous:
            lines.append(f"[{timestamp(frame.seconds)}] {clean}")
        if clean:
            previous = clean
    return "\n".join(lines) or None


async def read_screen_text(
    video_path: str, settings: Settings, frames: list[Frame] | None = None
) -> tuple[str | None, int]:
    """Sample, de-duplicate and OCR frames. Returns (text, frames_read).

    ``frames`` lets a caller that already sampled the video (the vision path)
    reuse them instead of decoding it a second time.
    """
    frames = await sample_frames(video_path, settings) if frames is None else frames
    if not frames:
        return None, 0
    backend = globals()[_BACKENDS[settings.ocr_provider]]
    texts = await backend(frames, settings)
    return merge_text(frames, texts), len(frames)


def to_png_bytes(image) -> bytes:  # small helper for tests that build frames with Pillow
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
