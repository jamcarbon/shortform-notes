"""One LLM call over caption and transcript, returning title, summary and takeaways.

Backends (all produce the same JSON shape):

* ``openai``      Chat Completions with a strict JSON schema.
* ``anthropic``   Messages API with ``output_config`` JSON schema.
* ``claude-code`` shells out to ``claude -p`` (Claude subscription, no API key).
* ``codex``       shells out to ``codex exec`` (ChatGPT subscription, no API key).

The CLI backends run one-shot with all tools disabled and read-only sandboxes;
they receive only the prompt. Under ``--vision agentic`` they instead get the
sampled frames on disk and read-only access to that one directory, so the agent
can open an original when a contact-sheet cell is too small to settle a
question. Summaries are best-effort: on any failure the note is still written
with the verbatim caption and transcript.

With ``--vision`` every backend also sees the video. Frames sampled by ``ocr``
are tiled into timestamped contact sheets and sent as images: the APIs take
image blocks, ``claude -p`` takes them as an API-style user message on its
``--input-format stream-json`` stdin, and ``codex exec`` takes them as ``-i``
files. Only ``none``, which makes no call at all, cannot see them. The same
call then also returns ``scenes``, a timestamped breakdown of what is on
screen, which becomes the note's "Video breakdown" section, and marks each scene
``keep_screenshot`` true or false: whether that frame shows something the narration
does not already say. Only kept scenes become screenshots, so selectivity costs no
extra call. Whenever a summary runs, the call also picks one ``category`` for the
whole video from the configured taxonomy (``Settings.categories``).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from shortform_notes import ocr
from shortform_notes.config import Settings
from shortform_notes.note import Scene

logger = logging.getLogger(__name__)

CLI_TIMEOUT_SECONDS = 180
# Hard cap on frames per summary call: a long video would otherwise blow up one request.
# Tiled 16 to a contact sheet, so this is at most 3 images however long the video is.
MAX_VISION_FRAMES = 48

_TEXT_PROPERTIES = {
    "title": {"type": "string", "description": "Short descriptive title, max 8 words, no hashtags, no emoji"},
    "summary": {"type": "string", "description": "2-4 sentences: what the video is about and its main point"},
    "takeaways": {
        "type": "array",
        "items": {"type": "string"},
        "description": "3-6 concrete, specific takeaways: exact quantities, steps, names, numbers when present",
    },
}
# Only asked for when the model can see the frames; without them it would have to invent the scenes.
_SCENES_PROPERTY = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "time": {"type": "string", "description": "Timestamp as printed on the frame, mm:ss"},
            "description": {
                "type": "string",
                "description": (
                    "What is on screen at that moment, concretely: who or what, the action, the setting, "
                    "shot changes, and any visible text quoted exactly. Several sentences when the frame "
                    "warrants it. Only what is visible."
                ),
            },
            "keep_screenshot": {
                "type": "boolean",
                "description": (
                    "True only if this frame shows information the transcript and caption do not already "
                    "convey. Default to false."
                ),
            },
            "reason": {"type": "string", "description": "One short line: why this frame is or is not worth keeping"},
        },
        "required": ["time", "description", "keep_screenshot", "reason"],
        "additionalProperties": False,
    },
    "description": "Scene-by-scene breakdown, one entry per distinct moment, in chronological order",
}


def _category_property(categories: Sequence[str]) -> dict:
    return {
        "type": "string",
        "enum": list(categories),
        "description": "The single best-fitting category for the whole video, from the list",
    }


def summary_schema(with_frames: bool = False, categories: Sequence[str] = ()) -> dict:
    """The JSON contract every backend answers with.

    ``scenes`` is added only under vision, ``category`` whenever there is a taxonomy to choose from.
    """
    properties = {**_TEXT_PROPERTIES, "scenes": _SCENES_PROPERTY} if with_frames else dict(_TEXT_PROPERTIES)
    if categories:
        properties["category"] = _category_property(categories)
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),  # strict schemas require every declared property
        "additionalProperties": False,
    }


class SummaryError(Exception):
    """A backend failed; the caller degrades to a caption-derived title."""


@dataclass(frozen=True)
class Summary:
    title: str
    summary: str
    takeaways: tuple[str, ...]
    scenes: tuple[Scene, ...] = ()  # empty unless the model saw the frames
    category: str | None = None  # one of Settings.categories; None when no summary ran


def build_prompt(audience: str, with_frames: bool = False, categories: Sequence[str] = ()) -> str:
    prompt = (
        f"You are turning a short social-media video into a personal note for {audience}. "
        "You are given the video's caption and/or spoken transcript. Return JSON with keys "
        "title (max 8 words, no hashtags, no emoji), summary (2-4 sentences: what the video is "
        "about and its main point) and takeaways (3-6 concrete, specific bullets: exact "
        "quantities, steps, names, numbers when present).\n"
        "Rules: never invent details that are not in the caption, transcript, or on-screen text. If the video is "
        "a recipe, the takeaways must list every ingredient with amounts and the steps in order. "
        "Plain text only."
    )
    if with_frames:
        prompt += (
            f"\nYou are also given contact sheets of frames sampled from the video. Each cell is one frame, "
            f"laid out {ocr.GRID_COLS} per row in chronological order, left to right then top to bottom, with "
            "its timestamp printed in the top-left corner of the cell. Read them as a filmstrip: use them for "
            "what happens on screen, the actions, results and quantities that are shown rather than said. The "
            "frames are a source like the others, so report what they show and invent nothing.\n"
            "Also return scenes: a chronological breakdown of the video, one entry per distinct moment, "
            "timestamped with the label printed on the frame it came from. Describe each moment concretely "
            "and in detail — who or what is on screen, what they are doing, the setting and objects around "
            "them, how the shot is framed and when it cuts or the camera moves, and any text, numbers or "
            "labels visible in the frame, quoted exactly. Write several sentences for a moment that has that "
            "much in it, and one for a moment that does not. Prefer the specific to the general: 'a hand "
            "grates a yellow onion into a glass bowl' beats 'someone prepares an ingredient'. Merge cells "
            "that show the same moment into one entry. Everything you write must be visible in the frames: "
            "detail means looking harder, never guessing, and never narrating what a video like this "
            "usually does next.\n"
            "For every scene also decide keep_screenshot, with a one-line reason. The note keeps a "
            "screenshot only of scenes marked true, and a note with three well-chosen screenshots beats one "
            "with fifteen redundant ones, so the default is false. Mark true only when the frame carries "
            "information a reader of the transcript and caption would otherwise miss: a diagram, chart, "
            "table or slide; a written list, recipe card, code, settings screen or app UI; the finished "
            "result (the plated dish, the completed build, the before/after); or an object, place or "
            "technique whose look matters and is not described in words. Mark false for a person talking "
            "to camera, reaction shots, b-roll that only illustrates what is being said, transitions, "
            "blurry or mid-cut frames, on-screen text that just repeats the spoken words or the "
            "caption, and sponsor or self-promotion segments, logos and end cards. When several "
            "scenes show the same information, keep only the clearest one. Most "
            "videos need zero to four screenshots; a talking-head video usually needs none."
        )
    if categories:
        prompt += (
            "\nAlso return category: the one category from this list that best fits the whole video, "
            f"spelled exactly as listed: {'; '.join(categories)}. Choose by the subject matter a reader "
            "would file it under, not by the format of the video."
        )
        if any(c.lower() == "other" for c in categories):
            prompt += " Use Other only when nothing else fits."
    return prompt


def _user_message(caption: str | None, transcript: str | None, screen_text: str | None = None) -> str:
    msg = f"Caption:\n{caption or '(none)'}\n\nTranscript:\n{transcript or '(none)'}"
    if screen_text:
        msg += f"\n\nOn-screen text (OCR, with timestamps):\n{screen_text}"
    return msg


# Vision: frames sampled by ocr.extract_frames, tiled into contact sheets, attached to the summary call.


def select_frames(frames: Sequence[ocr.Frame], limit: int = MAX_VISION_FRAMES) -> list[ocr.Frame]:
    """At most ``limit`` frames, evenly spaced across the video when there are more."""
    if len(frames) <= limit:
        return list(frames)
    step = (len(frames) - 1) / (limit - 1)
    return [frames[round(i * step)] for i in range(limit)]


def _image_block(grid: ocr.Grid) -> dict:
    """Anthropic-shaped image block; also what ``claude -p`` accepts on its stream-json stdin."""
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(grid.png).decode()},
    }


@dataclass(frozen=True)
class VisionEstimate:
    frames: int
    sheets: int  # images actually sent, after tiling
    usd: float  # 0.0 on the subscription backends

    def describe(self) -> str:
        sheets = f"{self.sheets} contact sheet{'s' if self.sheets != 1 else ''}"
        price = "included in your subscription" if not self.usd else f"about ${self.usd:.3f}"
        return f"up to {self.frames} frames as {sheets}, {price}"


def vision_estimate(duration_seconds: float, settings: Settings) -> VisionEstimate:
    """What one video costs, before de-duplication drops near-identical frames.

    Priced per contact sheet, not per frame: tiling is what makes vision affordable.
    ``claude-code`` and ``codex`` run on a subscription, so their price is 0.
    """
    frames = min(ocr.frame_count(duration_seconds, settings.ocr_fps), MAX_VISION_FRAMES)
    sheets = math.ceil(frames / ocr.FRAMES_PER_GRID)
    model = settings.openai_summary_model if settings.summary_provider == "openai" else settings.anthropic_summary_model
    return VisionEstimate(frames, sheets, round(sheets * ocr.per_sheet_usd(settings.summary_provider, model), 4))


def _cli_prompt(
    caption: str | None,
    transcript: str | None,
    settings: Settings,
    screen_text: str | None = None,
    with_frames: bool = False,
) -> str:
    """Single-string prompt for agent CLIs: instructions, schema and content, JSON-only reply."""
    return (
        f"{build_prompt(settings.audience, with_frames, settings.categories)}\n\n"
        f"Reply with ONLY a JSON object matching this schema, no prose, no code fences:\n"
        f"{json.dumps(summary_schema(with_frames, settings.categories))}\n\n"
        f"{_user_message(caption, transcript, screen_text)}"
    )


# ── agentic vision ─────────────────────────────────────────────────────
#
# One-shot vision hands an agent backend a frozen view: the contact sheets, and nothing else it
# can act on. Agentic vision instead writes the frames it sampled to a directory and tells the
# agent where they are, so it can open the ones the sheets left ambiguous. The sheets still go
# in the request — they are the cheap overview — the directory is only for a second look.


def frame_filename(seconds: float) -> str:
    """``00-03.png``: the cell's timestamp, with the colon swapped for a filesystem-safe dash."""
    return f"{ocr.timestamp(seconds).replace(':', '-')}.png"


def write_frames(frames: Sequence[ocr.Frame], parent: Path) -> Path:
    """The sampled frames as timestamp-named PNGs, at the resolution they were sampled at."""
    directory = parent / "frames"
    directory.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        (directory / frame_filename(frame.seconds)).write_bytes(frame.png)
    return directory


def agentic_instructions(directory: Path, frames: Sequence[ocr.Frame]) -> str:
    """Tell the agent the directory and what is in it. Without the inventory it guesses filenames."""
    inventory = ", ".join(frame_filename(frame.seconds) for frame in frames)
    return (
        f"\n\nThe contact sheets give you the overview. The full-resolution originals are in "
        f"{directory}, named by timestamp: {inventory}. Open any frame you need to read text, "
        f"identify people or objects, or examine a moment more closely before writing your scenes. "
        f"You can only read those files — everything else is off, and you still reply with the "
        f"JSON object and nothing else."
    )


def extract_json(text: str) -> dict:
    """Lenient parse for CLI output: strips code fences, then takes the outermost ``{...}``."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise SummaryError(f"no JSON object in model output: {text[:120]!r}")
    return json.loads(cleaned[start : end + 1])


def _scene_time(raw: object) -> str:
    """mm:ss. Models echo the label printed on the cell ("00:03"), but some answer in seconds."""
    text = str(raw if raw is not None else "").strip().strip("[]")
    try:
        return ocr.timestamp(float(text))
    except ValueError:
        return text


def _scenes(raw: object) -> tuple[Scene, ...]:
    """Whatever the model called a scene, as Scenes. Anything unusable is dropped, not raised."""
    if not isinstance(raw, list):
        return ()
    scenes = []
    for item in raw:
        keep, reason = False, ""
        if isinstance(item, dict):
            time, description = item.get("time"), str(item.get("description") or "").strip()
            keep = _truthy(item.get("keep_screenshot"))
            reason = str(item.get("reason") or "").strip()
        elif isinstance(item, str):  # a backend that ignored the schema answers with plain lines
            time, description = None, item.strip()
        else:
            continue
        if description:
            scenes.append(Scene(time=_scene_time(time), description=description, keep=keep, reason=reason))
    return tuple(scenes)


def _truthy(raw: object) -> bool:
    """CLI backends are not schema-enforced and sometimes answer "true" or "yes" as strings."""
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "yes", "1")
    return raw is True


def match_category(raw: object, categories: Sequence[str]) -> str | None:
    """The taxonomy's own spelling of the model's answer; None when there is no taxonomy or no answer.

    The API backends enforce the enum, but the CLI ones only read it, so a case or ``&``/``and``
    slip is mapped back rather than creating a near-duplicate Notion option. An answer that is not
    on the list at all becomes "Other" when the taxonomy has one.
    """
    text = str(raw or "").strip()
    if not text or not categories:
        return None

    def norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower().replace("&", "and"))

    by_norm = {norm(c): c for c in categories}
    if norm(text) in by_norm:
        return by_norm[norm(text)]
    return next((c for c in categories if c.lower() == "other"), None)


def _coerce(data: dict, fallback_title: str, categories: Sequence[str] = ()) -> Summary:
    takeaways = tuple(str(t).strip() for t in data.get("takeaways") or [] if str(t).strip())
    return Summary(
        title=(data.get("title") or fallback_title).strip(),
        summary=(data.get("summary") or "").strip(),
        takeaways=takeaways,
        scenes=_scenes(data.get("scenes")),
        category=match_category(data.get("category"), categories),
    )


# API backends


async def _summarize_openai(
    caption: str | None,
    transcript: str | None,
    settings: Settings,
    screen_text: str | None = None,
    grids: Sequence[ocr.Grid] = (),
    frames: Sequence[ocr.Frame] = (),  # unused: an API call gets one frozen look at the sheets
) -> dict:
    from openai import AsyncOpenAI

    user: str | list[dict] = _user_message(caption, transcript, screen_text)
    if grids:
        user = [{"type": "text", "text": user}]
        for i, grid in enumerate(grids, 1):
            b64 = base64.b64encode(grid.png).decode()
            user.append({"type": "text", "text": f"contact sheet {i} of {len(grids)}: {grid.describe()}"})
            # detail=high, not low: low gives the sheet a far smaller pixel budget, and the model
            # then invents cell text rather than reading it (measured on gpt-4o-mini: 2/16 cells
            # correct at low, 16/16 at high). The difference between seeing the video and guessing.
            user.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}})
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=settings.openai_summary_model,
        messages=[
            {"role": "system", "content": build_prompt(settings.audience, bool(grids), settings.categories)},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "reel_summary",
                "strict": True,
                "schema": summary_schema(bool(grids), settings.categories),
            },
        },
        # The GPT-5 family rejects `max_tokens` (400, "use max_completion_tokens") and any
        # temperature but the default, so neither appears here. The budget covers reasoning
        # tokens as well as the reply — too small and a reasoning model spends it all thinking
        # and returns an empty message. The schema keeps the output tight without a temperature.
        max_completion_tokens=4000,
    )
    return json.loads(response.choices[0].message.content or "{}")


async def _summarize_anthropic(
    caption: str | None,
    transcript: str | None,
    settings: Settings,
    screen_text: str | None = None,
    grids: Sequence[ocr.Grid] = (),
    frames: Sequence[ocr.Frame] = (),  # unused: an API call gets one frozen look at the sheets
) -> dict:
    from anthropic import AsyncAnthropic

    content: list[dict] = []
    for i, grid in enumerate(grids, 1):
        content.append({"type": "text", "text": f"contact sheet {i} of {len(grids)}: {grid.describe()}"})
        content.append(_image_block(grid))
    content.append({"type": "text", "text": _user_message(caption, transcript, screen_text)})
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    response = await client.messages.create(
        model=settings.anthropic_summary_model,
        max_tokens=4000,  # scenes with a reason each outgrow 2000 tokens on a long, fast-cut video
        system=build_prompt(settings.audience, bool(grids), settings.categories),
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": summary_schema(bool(grids), settings.categories)}},
    )
    if response.stop_reason == "refusal":
        raise SummaryError("model declined to summarize this video")
    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    return json.loads(text or "{}")


# Coding-agent CLI backends


async def _run_cli(argv: list[str], stdin_text: str) -> str:
    """Run a CLI with the prompt on stdin; return stdout. Raises SummaryError on failure."""
    executable = shutil.which(argv[0])
    if executable is None:
        raise SummaryError(f"`{argv[0]}` is not on PATH")
    # The resolved path, not the bare name: on Windows npm installs `claude` and `codex` as
    # `.cmd` shims, which CreateProcess only finds by full path (a bare name is WinError 2).
    proc = await asyncio.create_subprocess_exec(
        executable,
        *argv[1:],
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin_text.encode()), timeout=CLI_TIMEOUT_SECONDS)
    except TimeoutError:
        proc.kill()
        raise SummaryError(f"`{argv[0]}` timed out after {CLI_TIMEOUT_SECONDS}s") from None
    if proc.returncode != 0:
        raise SummaryError(f"`{argv[0]}` exited {proc.returncode}: {err.decode(errors='replace')[:300]}")
    return out.decode(errors="replace")


def claude_code_argv(settings: Settings, images: bool = False, frames_dir: str | None = None) -> list[str]:
    """``claude -p`` one-shot: JSON envelope, nothing persisted, tools off unless frames are offered.

    Images can only be sent as API-style user messages on stdin, which needs
    ``--input-format stream-json``; the CLI then requires a matching
    ``--output-format stream-json`` and ``--verbose`` (both verified against the CLI).

    ``frames_dir`` turns on agentic vision. ``--tools`` sets the *available* tool set, so naming
    only ``Read`` leaves the agent no way to write, run a command or reach the network; the
    matching ``--allowed-tools`` keeps a non-interactive run from stalling on a permission
    prompt it has no way to answer, and ``--add-dir`` is what makes the directory readable at all.
    """
    argv = ["claude", "-p", "--output-format", "stream-json" if images else "json"]
    if images:
        argv += ["--input-format", "stream-json", "--verbose"]
    if frames_dir:
        argv += ["--tools", "Read", "--allowed-tools", "Read", "--add-dir", frames_dir]
    else:
        argv += ["--tools", ""]
    argv += ["--disable-slash-commands", "--no-session-persistence"]
    if settings.claude_code_model:
        argv += ["--model", settings.claude_code_model]
    return argv


def claude_code_stdin(prompt: str, grids: Sequence[ocr.Grid]) -> str:
    """One stream-json user message: the prompt, then a labelled image block per contact sheet."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for i, grid in enumerate(grids, 1):
        content.append({"type": "text", "text": f"contact sheet {i} of {len(grids)}: {grid.describe()}"})
        content.append(_image_block(grid))
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"


def _stream_json_result(raw: str) -> dict:
    """The final ``result`` event of a stream-json run; same shape as the plain JSON envelope."""
    for line in reversed(raw.strip().splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            return event
    raise SummaryError(f"no result event in claude stream-json output: {raw[:120]!r}")


async def _summarize_claude_code(
    caption: str | None,
    transcript: str | None,
    settings: Settings,
    screen_text: str | None = None,
    grids: Sequence[ocr.Grid] = (),
    frames: Sequence[ocr.Frame] = (),
) -> dict:
    prompt = _cli_prompt(caption, transcript, settings, screen_text, with_frames=bool(grids))
    # The directory lives exactly as long as the call, like the media the pipeline downloads.
    with tempfile.TemporaryDirectory(prefix="shortform-notes-vision-") as tmpdir:
        frames_dir = None
        if frames and settings.vision_is_agentic:
            frames_dir = str(write_frames(frames, Path(tmpdir)))
            prompt += agentic_instructions(Path(frames_dir), frames)
        raw = await _run_cli(
            claude_code_argv(settings, images=bool(grids), frames_dir=frames_dir),
            claude_code_stdin(prompt, grids) if grids else prompt,
        )
    envelope = _stream_json_result(raw) if grids else json.loads(raw)
    if envelope.get("is_error"):
        raise SummaryError(f"claude -p reported an error: {str(envelope.get('result'))[:200]}")
    return extract_json(str(envelope.get("result") or ""))


def codex_argv(settings: Settings, last_message_path: str, image_paths: Sequence[str] = ()) -> list[str]:
    """``codex exec`` one-shot: read-only sandbox, no approval prompts, final message to a file.

    ``-i/--image`` attaches files to the initial prompt, so contact sheets go through as temp PNGs.
    """
    # No --ask-for-approval: codex 0.147 removed it from `exec`, which never prompts anyway.
    # --skip-git-repo-check because a reel is imported from wherever the user happens to be,
    # and codex otherwise refuses to start outside a trusted git repo.
    argv = [
        "codex",
        "exec",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--output-last-message",
        last_message_path,
    ]
    for path in image_paths:
        argv += ["-i", path]
    if settings.codex_model:
        argv += ["--model", settings.codex_model]
    return argv + ["-"]  # "-" = read the prompt from stdin


async def _summarize_codex(
    caption: str | None,
    transcript: str | None,
    settings: Settings,
    screen_text: str | None = None,
    grids: Sequence[ocr.Grid] = (),
    frames: Sequence[ocr.Frame] = (),
) -> dict:
    # The temp directory holds the reply file, any contact sheets and, under agentic vision, the
    # frames themselves; it is removed on the way out.
    with tempfile.TemporaryDirectory(prefix="shortform-notes-codex-") as tmpdir:
        last_message = Path(tmpdir) / "last.txt"
        image_paths = []
        for i, grid in enumerate(grids, 1):
            path = Path(tmpdir) / f"sheet-{i}.png"
            path.write_bytes(grid.png)
            image_paths.append(str(path))
        prompt = _cli_prompt(caption, transcript, settings, screen_text, with_frames=bool(grids))
        if grids:
            sheets = "; ".join(f"sheet {i} is {g.describe()}" for i, g in enumerate(grids, 1))
            prompt += f"\n\nThe attached images are the contact sheets, in order: {sheets}."
        if frames and settings.vision_is_agentic:
            # `codex exec` already runs --sandbox read-only, so the path in the prompt is enough.
            prompt += agentic_instructions(write_frames(frames, Path(tmpdir)), frames)
        stdout = await _run_cli(codex_argv(settings, str(last_message), image_paths), prompt)
        text = last_message.read_text() if last_message.exists() else stdout
    return extract_json(text)


# Names, not function objects, so the lookup is late-bound (tests patch these).
_BACKENDS = {
    "openai": "_summarize_openai",
    "anthropic": "_summarize_anthropic",
    "claude-code": "_summarize_claude_code",
    "codex": "_summarize_codex",
}


async def summarize(
    caption: str | None,
    transcript: str | None,
    settings: Settings,
    title_hint: str | None = None,
    screen_text: str | None = None,
    frames: Sequence[ocr.Frame] = (),
) -> Summary:
    """Return a Summary; degrades to a caption-derived title when the LLM step fails or is off."""
    fallback_title = (title_hint or caption or transcript or "Reel").split("\n")[0][:60]
    backend_name = _BACKENDS.get(settings.summary_provider)
    if backend_name is None:
        return Summary(fallback_title, "", ())
    backend = globals()[backend_name]
    grids: list[ocr.Grid] = []
    selected: list[ocr.Frame] = []
    if frames and settings.can_see_video:
        selected = select_frames(frames)
        grids = await ocr.tile_frames(selected)
        mode = "agentic" if settings.vision_is_agentic else "one-shot"
        logger.info("vision (%s): %d frames as %d contact sheet(s)", mode, len(selected), len(grids))
    try:
        # The same frames the sheets were built from, so an agentic backend can open the originals.
        data = await backend(caption, transcript, settings, screen_text, grids, selected)
    except Exception as exc:  # noqa: BLE001 (summary is best-effort; the verbatim note is still written)
        logger.warning("summary failed (%s): %s", settings.summary_provider, exc)
        return Summary(fallback_title, "", ())
    summary = _coerce(data, fallback_title, settings.categories)
    kept = sum(scene.keep for scene in summary.scenes)
    logger.info("summary: category=%s, %d of %d scenes kept", summary.category, kept, len(summary.scenes))
    return summary
