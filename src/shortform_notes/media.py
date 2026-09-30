"""Audio + metadata download via yt-dlp.

``bestaudio/best`` and no ffmpeg: Instagram and YouTube expose a standalone
audio stream, TikTok falls through to the muxed mp4, and transcription APIs
accept both. When the picture is wanted too, a muxed mp4 is used where one
exists and, with ffmpeg on PATH, video and audio streams are merged where not.
Never set a custom User-Agent; yt-dlp pairs its UA with the rest of the browser
fingerprint and Instagram rejects mismatches.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# OpenAI's transcription upload cap; yt-dlp aborts the download past this.
MAX_AUDIO_BYTES = 25 * 1024 * 1024


class MediaFetchError(Exception):
    """yt-dlp could not fetch the video; the caller should surface, not retry."""


@dataclass(frozen=True)
class DownloadedMedia:
    audio_path: str | None
    video_path: str | None
    caption: str | None
    creator_handle: str | None
    creator_name: str | None
    title: str | None
    timestamp: int | None
    duration: float | None
    thumbnail: str | None
    webpage_url: str | None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _video_format() -> str:
    """The yt-dlp format for a run that needs the picture.

    YouTube has stopped serving a muxed mp4 for many Shorts (only DASH video-only and audio-only
    streams), so with ffmpeg on PATH the best H.264 stream up to 1080p is merged with the audio;
    H.264 because every decoder here (ffmpeg keyframes, OpenCV) reads it, which AV1 is not
    guaranteed to be. Without ffmpeg nothing can merge, so only a muxed file will do.
    """
    muxed = "best[ext=mp4][height<=1080]/best"
    if shutil.which("ffmpeg") is None:
        return muxed
    return f"bv*[height<=1080][vcodec^=avc1]+ba/bv*[height<=1080]+ba/{muxed}/bv*+ba"


def _ytdlp_sync(url: str, tmpdir: str, download: bool, video: bool = False) -> DownloadedMedia:
    import yt_dlp  # lazy: heavy import, and tests stub this function

    warnings: list[str] = []
    opts = {
        "format": _video_format() if video else "bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": f"{tmpdir}/%(id)s.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "max_filesize": MAX_AUDIO_BYTES,
        "socket_timeout": 20,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=download)
    downloads = info.get("requested_downloads") or []
    path = downloads[0].get("filepath") if downloads else None
    if download and path is None:
        warnings.append("yt-dlp returned metadata but no downloadable stream")
    audio_path, video_path = (path, path) if video else (path, None)
    title = info.get("title") or ""
    return DownloadedMedia(
        audio_path=audio_path,
        video_path=video_path,
        caption=(info.get("description") or "").strip() or None,
        creator_handle=info.get("channel") or info.get("uploader_id"),
        creator_name=info.get("uploader") or info.get("channel"),
        # yt-dlp synthesises "Video by <user>" for Instagram, not real content.
        title=None if title.startswith("Video by ") else (title or None),
        timestamp=info.get("timestamp"),
        duration=info.get("duration"),
        thumbnail=info.get("thumbnail"),
        webpage_url=info.get("webpage_url") or url,
        warnings=tuple(warnings),
    )


async def download_media(url: str, tmpdir: str, download: bool = True, video: bool = False) -> DownloadedMedia:
    """Run yt-dlp off the event loop. ``download=False`` fetches metadata only; ``video`` keeps the picture."""
    try:
        return await asyncio.to_thread(_ytdlp_sync, url, tmpdir, download, video)
    except Exception as exc:  # yt-dlp raises DownloadError and friends
        message = str(exc).splitlines()[0][:300]
        logger.warning("yt-dlp failed for %s: %s", url, message)
        raise MediaFetchError(message) from exc
