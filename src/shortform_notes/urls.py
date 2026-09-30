"""URL detection and normalisation for supported platforms."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

# One regex, shared by the CLI, the MCP tool and the detector, so they never disagree.
REEL_URL_RE = re.compile(
    r"https?://(?:www\.|m\.|web\.)?(?:"
    r"instagram\.com/(?:reels?|p|tv|share/(?:reel|p|reels))/[A-Za-z0-9_\-]+"
    r"|(?:vm\.|vt\.)?tiktok\.com/[^\s<>\"']+"
    r"|youtube\.com/shorts/[A-Za-z0-9_\-]+"
    r"|youtu\.be/[A-Za-z0-9_\-]+"
    # Facebook: /reel/<id>, the /share/r/ and /share/v/ links the app copies, /watch?v=, page videos, fb.watch
    r"|facebook\.com/(?:reels?/[0-9]+|share/[rv]/[A-Za-z0-9_\-]+|watch/?\?v=[0-9]+|[^/\s]+/videos/[0-9]+)"
    r"|fb\.watch/[A-Za-z0-9_\-]+"
    r")[^\s<>\"']*",
    re.I,
)


def detect_reel_url(text: str) -> str | None:
    """Return the first supported short-video URL in ``text``, else None."""
    match = REEL_URL_RE.search(text or "")
    return match.group(0).rstrip(".,;)") if match else None


def platform_for(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if "instagram.com" in host:
        return "instagram"
    if "tiktok.com" in host:
        return "tiktok"
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if "facebook.com" in host or host == "fb.watch":
        return "facebook"
    return "unknown"


def strip_tracking(url: str) -> str:
    """Drop query/fragment (``?igsh=``, ``?si=``, ``?mibextid=``) so the same video dedups.

    Facebook's ``/watch?v=<id>`` is the one link whose query *is* the video, so ``v`` survives.
    """
    parsed = urlparse(url)
    query = ""
    if "facebook.com" in (parsed.hostname or "") and parsed.path.rstrip("/") == "/watch":
        video = parse_qs(parsed.query).get("v")
        query = f"v={video[0]}" if video else ""
    return parsed._replace(query=query, fragment="").geturl()
