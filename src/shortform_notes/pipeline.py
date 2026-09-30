"""Orchestrator: URL to caption and transcript, to summary, to a Markdown file on disk.

Each source is independent and the note records which ones succeeded:

  1. caption    Instagram: captioned-embed payload (no key). TikTok/YouTube:
                yt-dlp ``description``. Cheap and often the whole recipe.
  2. transcript yt-dlp ``bestaudio``, then OpenAI transcription or local faster-whisper.
  3. summary    one LLM call: OpenAI / Anthropic API, or the ``claude`` / ``codex``
                CLI using an existing subscription. Best-effort. With ``vision``
                on, the sampled video frames ride along in that same call, and
                the model marks which scenes deserve a screenshot and picks a
                category for the video.

Then, each step best-effort and degrading to a warning in the note: the kept
scenes' frames are saved as PNGs under ``<output_dir>/assets/<note>/``, a Notion
page is created when NOTION_TOKEN and NOTION_DATABASE_ID are set, the Markdown
note is written, and the import is recorded in the local library index. The
video itself is only ever in a temporary directory.

A link already in the library (any spelling of it, see ``urls.reel_key``) is not
imported twice: the earlier note is returned and nothing runs, unless ``force``
asks for a re-import, which replaces the earlier note, its screenshots and its
Notion page. A note imported before Notion was set up is re-imported so it gets one.

Typical cost with everything on: about $0.004 per minute-long reel.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import httpx

from shortform_notes import instagram, library, media, notion_writer, ocr, urls
from shortform_notes.config import AGENTIC_VISION_PROVIDERS, Settings, load_settings
from shortform_notes.note import ReelContent, Scene, build_note, note_filename
from shortform_notes.summarize import summarize, vision_estimate
from shortform_notes.transcribe import transcribe

logger = logging.getLogger(__name__)


class ReelImportError(Exception):
    """Nothing usable could be fetched from the URL."""


SLIDE_FETCH_TIMEOUT = 20.0


async def fetch_slides(image_urls: tuple[str, ...]) -> tuple[list[bytes], list[str]]:
    """Download carousel slides from Instagram's CDN, in order. A failed slide is a warning, not a failure."""
    images: list[bytes] = []
    warnings: list[str] = []
    async with httpx.AsyncClient(headers=instagram.HEADERS, timeout=SLIDE_FETCH_TIMEOUT, follow_redirects=True) as http:
        for index, image_url in enumerate(image_urls, start=1):
            try:
                resp = await http.get(image_url)
                resp.raise_for_status()
                images.append(resp.content)
            except httpx.HTTPError as exc:
                warnings.append(f"slide {index} could not be fetched: {exc}")
    return images, warnings


@dataclass(frozen=True)
class ReelImportResult:
    path: Path
    title: str
    summary: str
    takeaways: tuple[str, ...]
    sources: tuple[str, ...]
    warnings: tuple[str, ...]
    scenes: tuple[Scene, ...] = ()  # only under --vision; see summarize.summary_schema
    category: str | None = None
    notion_url: str | None = None
    screenshot_count: int = 0
    duplicate: bool = False  # already in the library: nothing ran, ``path`` is the earlier note

    def to_dict(self) -> dict:
        data = {
            "path": str(self.path),
            "title": self.title,
            "summary": self.summary,
            "takeaways": list(self.takeaways),
            "sources": list(self.sources),
            "warnings": list(self.warnings),
            "category": self.category,
            "notion_url": self.notion_url,
            "screenshot_count": self.screenshot_count,
        }
        if self.scenes:  # absent, not empty, so a run without vision looks exactly as it did
            data["scenes"] = [scene.to_dict() for scene in self.scenes]
        if self.duplicate:
            data["duplicate"] = True
        return data


async def gather_content(url: str, tmpdir: str, settings: Settings) -> tuple[ReelContent, list[ocr.Frame]]:
    """Collect caption, transcript and (for OCR or vision) sampled video frames from every source that works.

    The frames come back alongside the note content because they are an input to
    the summary call, not something the note itself renders.
    """
    platform = urls.platform_for(url)
    clean_url = urls.strip_tracking(url)
    embed: instagram.InstagramEmbed | None = None
    downloaded: media.DownloadedMedia | None = None
    warnings: list[str] = []
    want_frames = settings.ocr or settings.can_see_video
    if settings.vision and not settings.can_see_video:
        warnings.append("Vision skipped: no summary backend is configured, so nothing would see the frames")
    if settings.vision_agentic and settings.can_see_video and not settings.vision_is_agentic:
        warnings.append(
            f"Agentic vision needs an agent backend that can open the frames "
            f"({', '.join(AGENTIC_VISION_PROVIDERS)}); {settings.summary_provider} ran one-shot instead"
        )

    if platform == "instagram":
        shortcode = await instagram.resolve_shortcode(url)
        if shortcode:
            clean_url = f"https://www.instagram.com/reel/{shortcode}/"
            embed = await instagram.fetch_embed(shortcode)
            if embed is None:
                warnings.append("Instagram embed endpoint returned no payload (blocked, private or deleted)")
        else:
            warnings.append("Could not resolve an Instagram shortcode from the link")

    try:
        downloaded = await media.download_media(
            clean_url, tmpdir, download=settings.can_transcribe or want_frames, video=want_frames
        )
        warnings.extend(downloaded.warnings)
    except media.MediaFetchError as exc:
        # A photo/carousel post has no video for yt-dlp to find; that is not worth a warning.
        if not (embed and embed.image_urls and "No video formats found" in str(exc)):
            warnings.append(f"yt-dlp could not fetch media: {exc}")

    caption = (embed.caption if embed else None) or (downloaded.caption if downloaded else None)
    transcript = None
    if downloaded and downloaded.audio_path and settings.can_transcribe:
        try:
            transcript = await transcribe(downloaded.audio_path, settings)
        except Exception as exc:  # noqa: BLE001 (surface as a warning, keep the caption)
            warnings.append(f"transcription failed: {exc}")
        if not transcript and "transcription failed" not in " ".join(warnings):
            warnings.append("Audio downloaded but transcription returned nothing")
    elif not settings.can_transcribe:
        warnings.append(
            'Transcription skipped (set OPENAI_API_KEY, or pip install "shortform-notes[local]" for offline Whisper)'
        )

    duration = (embed.duration if embed else None) or (downloaded.duration if downloaded else None) or 0

    # Sampled once and shared: the summary call sees the frames, OCR reads the same ones.
    frames: list[ocr.Frame] = []
    slides = embed.image_urls if embed and not (downloaded and downloaded.video_path) else ()
    # A still post is its text: slides are always OCR'd (local RapidOCR is free), whatever --ocr says.
    run_ocr = settings.ocr or bool(slides)
    if slides and (settings.can_see_video or run_ocr):
        # A carousel / photo post: yt-dlp has no video to give, so the slides themselves are the frames.
        images, slide_warnings = await fetch_slides(slides)
        warnings.extend(slide_warnings)
        try:
            frames = await ocr.frames_from_images(images)
        except Exception as exc:  # noqa: BLE001 (surface as a warning; a text-only summary still runs)
            warnings.append(f"Slides could not be decoded: {exc}")
        if not frames:
            warnings.append("Slides skipped: none of the carousel images could be fetched")
    elif settings.can_see_video and downloaded and downloaded.video_path:
        logger.info("vision (%s): %s", settings.summary_provider, vision_estimate(duration, settings).describe())
        try:
            frames = await ocr.sample_frames(downloaded.video_path, settings)
        except Exception as exc:  # noqa: BLE001 (surface as a warning; a text-only summary still runs)
            warnings.append(f"Vision failed: frames could not be sampled: {exc}")
    elif settings.can_see_video:
        warnings.append("Vision skipped: no video could be downloaded")

    screen_text = None
    if run_ocr and (frames or (downloaded and downloaded.video_path)):
        if frames:
            logger.info("OCR: %d %s", len(frames), "slides" if slides else "sampled frames")
        else:
            logger.info("OCR: %s", ocr.estimate(duration, settings).describe())
        try:
            screen_text, frames_read = await ocr.read_screen_text(
                downloaded.video_path if downloaded else "", settings, frames or None
            )
            if not screen_text:
                warnings.append(f"OCR read {frames_read} frames and found no on-screen text")
        except Exception as exc:  # noqa: BLE001 (surface as a warning, keep the rest of the note)
            warnings.append(f"OCR failed: {exc}")
    elif settings.ocr:
        warnings.append("OCR skipped: no video could be downloaded")

    # Slides only feed the summary model as images when vision is on; OCR alone still reads them.
    vision_frames = frames if settings.can_see_video else []

    sources = tuple(
        s
        for s, present in (
            ("caption", caption),
            ("transcript", transcript),
            ("screen_text", screen_text),
            ("slides" if slides else "video", vision_frames),
        )
        if present
    )
    if not sources:
        raise ReelImportError("; ".join(warnings) or "no caption or audio available")

    posted = (
        datetime.fromtimestamp(downloaded.timestamp, tz=timezone.utc) if downloaded and downloaded.timestamp else None
    )
    content = ReelContent(
        url=clean_url,
        platform=platform,
        caption=caption,
        transcript=transcript,
        screen_text=screen_text,
        title=downloaded.title if downloaded else None,
        creator_handle=(embed.username if embed else None) or (downloaded.creator_handle if downloaded else None),
        creator_name=downloaded.creator_name if downloaded else None,
        posted=posted,
        duration=(embed.duration if embed else None) or (downloaded.duration if downloaded else None),
        thumbnail=(embed.thumbnail if embed else None) or (downloaded.thumbnail if downloaded else None),
        sources=sources,
        warnings=tuple(warnings),
    )
    return content, vision_frames


ASSETS_DIR = "assets"


def _save_screenshots_sync(
    scenes: Sequence[Scene], frames: Sequence[ocr.Frame], output_dir: Path, note_slug: str
) -> tuple[tuple[Scene, ...], list[str]]:
    """Write the frame behind every kept scene to ``assets/<note_slug>/<mm-ss>.png``.

    ``frames`` must be the frames the model was shown, in cell-number order (``Summary.frames``):
    a scene names its frame by that number, and falls back to its timestamp when it gave none.
    Returns the scenes with ``image_path`` set (relative to ``output_dir``, where the note lives)
    and any warnings. Two kept scenes that resolve to the same frame share it: only the first
    gets the image, so the note never shows one picture twice.
    """
    warnings: list[str] = []
    used: set[int] = set()
    names: set[str] = set()
    out: list[Scene] = []
    for scene in scenes:
        frame = None
        if scene.keep:
            frame = ocr.frame_by_number(frames, scene.frame) or ocr.nearest_frame(frames, scene.time)
        if frame is None or id(frame) in used:
            if scene.keep and frame is None:
                warnings.append(f"Screenshot for scene {scene.time or '(no time)'} skipped: no matching frame")
            out.append(scene)
            continue
        name = ocr.timestamp(frame.seconds).replace(":", "-")
        if name in names:  # two frames inside one second
            name = f"{name}-{sum(n.startswith(name) for n in names) + 1}"
        relative = f"{ASSETS_DIR}/{note_slug}/{name}.png"
        try:
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(frame.png)
        except OSError as exc:
            warnings.append(f"Screenshot {scene.time} could not be saved: {exc}")
            out.append(scene)
            continue
        used.add(id(frame))
        names.add(name)
        out.append(scene.with_image(relative))
    return tuple(out), warnings


async def save_screenshots(
    scenes: Sequence[Scene], frames: Sequence[ocr.Frame], output_dir: Path, note_slug: str
) -> tuple[tuple[Scene, ...], list[str]]:
    if not frames or not any(scene.keep for scene in scenes):
        return tuple(scenes), []
    return await asyncio.to_thread(_save_screenshots_sync, scenes, frames, output_dir, note_slug)


def _already_imported(output_dir: Path, entry: library.LibraryEntry) -> ReelImportResult:
    """The earlier import, as a result: nothing is fetched, summarized or written."""
    logger.info("already imported on %s: %s", entry.imported_at[:10], entry.note_path)
    return ReelImportResult(
        output_dir / entry.note_path,
        entry.title,
        entry.summary,
        (),
        (),
        (f"already imported on {entry.imported_at[:10]}; nothing was re-run",),
        category=entry.category,
        notion_url=entry.notion_url,
        screenshot_count=entry.screenshot_count,
        duplicate=True,
    )


def _inside(output_dir: Path, relative: str) -> Path | None:
    """``output_dir / relative`` when it stays inside ``output_dir`` (the index is hand-editable)."""
    root = output_dir.resolve()
    target = (root / relative).resolve()
    return target if target != root and target.is_relative_to(root) else None


def _unique_path(directory: Path, filename: str, now: datetime) -> Path:
    path = directory / filename
    if not path.exists():
        return path
    return directory / f"{filename[:-3]}-{now.strftime('%H%M%S')}.md"


async def import_reel(
    url: str, settings: Settings | None = None, now: datetime | None = None, force: bool = False
) -> ReelImportResult:
    """Fetch, transcribe, summarize and write ``<output_dir>/<date>-<creator>-<slug>.md``.

    A video already in the library comes back as that earlier import (``duplicate``) without
    running anything; ``force`` re-imports it and replaces the earlier note and Notion page.
    """
    settings = settings or load_settings()
    now = now or datetime.now(timezone.utc)
    clean = urls.detect_reel_url(url)
    if not clean:
        raise ReelImportError(f"not a supported Instagram / TikTok / YouTube Shorts link: {url}")
    earlier = library.find(settings.output_dir, clean)
    if earlier and not force and (earlier.notion_url or not settings.can_write_notion):
        return _already_imported(settings.output_dir, earlier)

    with tempfile.TemporaryDirectory(prefix="shortform-notes-") as tmpdir:
        content, frames = await gather_content(clean, tmpdir, settings)
    result = await summarize(
        content.caption,
        content.transcript,
        settings,
        title_hint=content.title,
        screen_text=content.screen_text,
        frames=frames,
    )

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    filename = note_filename(content.posted or now, content.creator_handle, result.title)
    if earlier and earlier.note_path == filename:
        path = settings.output_dir / filename  # a re-import with the same title overwrites the earlier note
    else:
        path = _unique_path(settings.output_dir, filename, now)
    warnings = list(content.warnings)
    earlier_assets = _inside(settings.output_dir, f"{ASSETS_DIR}/{Path(earlier.note_path).stem}") if earlier else None
    if earlier_assets and earlier_assets.is_dir():
        shutil.rmtree(earlier_assets, ignore_errors=True)  # before saving: the new note may reuse the folder

    try:
        scenes, shot_warnings = await save_screenshots(result.scenes, result.frames, settings.output_dir, path.stem)
        warnings += shot_warnings
    except Exception as exc:  # noqa: BLE001 (screenshots are extra; the note is still written)
        scenes = result.scenes
        warnings.append(f"Screenshots could not be saved: {exc}")
    screenshot_count = sum(1 for scene in scenes if scene.image_path)

    notion_url = None
    if settings.can_write_notion:
        try:
            page = await notion_writer.create_reel_page(
                settings,
                content,
                result.title,
                result.summary,
                result.takeaways,
                scenes,
                result.category,
                base_dir=settings.output_dir,
                replace_existing=force,
            )
            notion_url = page.url or None
            warnings += page.warnings
            if page.existing:
                warnings.append("Notion already had a page for this video; linked it instead of creating another")
        except Exception as exc:  # noqa: BLE001 (Notion is a copy; the local note is the record)
            logger.warning("Notion page not created: %s", exc)
            warnings.append(f"Notion page not created: {exc}")
    elif settings.notion and bool(settings.notion_token) != bool(settings.notion_database_id):
        missing = "NOTION_DATABASE_ID" if settings.notion_token else "NOTION_TOKEN"
        warnings.append(f"Notion skipped: {missing} is not set")

    content = replace(content, warnings=tuple(warnings))
    note = build_note(content, result.title, result.summary, result.takeaways, now, scenes, result.category, notion_url)
    path.write_text(note, encoding="utf-8")
    earlier_note = _inside(settings.output_dir, earlier.note_path) if earlier else None
    if earlier_note and earlier_note != path.resolve():
        earlier_note.unlink(missing_ok=True)
    logger.info(
        "reel imported: %s sources=%s scenes=%d screenshots=%d category=%s",
        path,
        content.sources,
        len(scenes),
        screenshot_count,
        result.category,
    )

    first_shot = next((scene.image_path for scene in scenes if scene.image_path), None)
    try:
        library.record(
            settings.output_dir,
            library.LibraryEntry(
                imported_at=now.isoformat(),
                title=result.title,
                url=content.url,
                platform=content.platform,
                category=result.category,
                note_path=path.name,
                notion_url=notion_url,
                creator=f"@{content.creator_handle}" if content.creator_handle else content.creator_name,
                summary=result.summary,
                screenshot_count=screenshot_count,
                thumbnail=first_shot or content.thumbnail,
            ),
        )
    except Exception as exc:  # noqa: BLE001 (the index is a convenience; the note is already on disk)
        logger.warning("library index not updated: %s", exc)
        warnings.append(f"Library index not updated: {exc}")

    return ReelImportResult(
        path,
        result.title,
        result.summary,
        result.takeaways,
        content.sources,
        tuple(warnings),
        scenes,
        category=result.category,
        notion_url=notion_url,
        screenshot_count=screenshot_count,
    )
