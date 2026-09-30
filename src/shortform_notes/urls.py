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


_ID_PATTERNS = {
    # /reel/X, /reels/X, /p/X and /tv/X are one post: Instagram serves each id under all of them.
    "instagram": re.compile(r"^/(?:reels?|p|tv)/([A-Za-z0-9_\-]+)"),
    "youtube": re.compile(r"^/(?:shorts/)?([A-Za-z0-9_\-]+)$"),  # youtube.com/shorts/X and youtu.be/X
    "tiktok": re.compile(r"/video/([0-9]+)"),
    "facebook": re.compile(r"^/(?:reels?/|[^/]+/videos/)([0-9]+)"),
}


def reel_key(url: str) -> str:
    """One identity per video, however its link was spelled: ``instagram:<id>``, ``youtube:<id>``...

    What de-duplication compares. A link whose id cannot be read offline (a share link, a
    short tiktok or fb.watch redirect) falls back to ``url:<host><path>``, minus ``www.``/``m.``,
    so the same spelling still matches itself.
    """
    parsed = urlparse(strip_tracking(url.strip()))
    platform = platform_for(url)
    path = parsed.path.rstrip("/")
    pattern = _ID_PATTERNS.get(platform)
    match = pattern.search(path) if pattern else None
    if match:
        return f"{platform}:{match.group(1)}"
    if platform == "facebook" and parsed.query.startswith("v="):
        return f"facebook:{parsed.query[2:]}"
    host = re.sub(r"^(?:www|m|web)\.", "", (parsed.hostname or "").lower())
    return f"url:{host}{path}"


def reel_search_term(key: str) -> str:
    """A substring every spelling of the key's link contains; what a Notion ``url contains`` filter needs."""
    return key.split(":", 1)[1]
