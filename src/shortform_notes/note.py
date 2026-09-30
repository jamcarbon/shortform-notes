"""Render the Markdown note (Obsidian-friendly YAML frontmatter + sections)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

_SLUG_MAX = 48


@dataclass(frozen=True)
class Scene:
    """One moment the summary model saw in the video frames, with the timestamp it was labelled with.

    Produced only under ``--vision``: without frames the model has nothing to
    describe, so a note written from caption and transcript alone has no scenes.
    """

    time: str  # mm:ss, as printed on the contact-sheet cell; "" when the model gave none
    description: str
    # The model's call on whether this moment is worth a screenshot: true only when the frame shows
    # something the narration does not already say (a diagram, a result, an on-screen list...).
    keep: bool = False
    reason: str = ""  # one line on why it was kept or skipped, for tuning the prompt
    image_path: str | None = None  # the saved screenshot, relative to the note's folder; set by the pipeline

    def with_image(self, path: str) -> Scene:
        return replace(self, image_path=path)

    def to_dict(self) -> dict:
        data = {"time": self.time, "description": self.description, "keep": self.keep, "reason": self.reason}
        if self.image_path:
            data["image_path"] = self.image_path
        return data

    def line(self) -> str:
        return f"- [{self.time}] {self.description}" if self.time else f"- {self.description}"


@dataclass(frozen=True)
class ReelContent:
    url: str
    platform: str
    caption: str | None
    transcript: str | None
    screen_text: str | None  # timestamped on-screen text from OCR, when enabled
    title: str | None  # platform-provided title (YouTube/TikTok); Instagram has none
    creator_handle: str | None
    creator_name: str | None
    posted: datetime | None
    duration: float | None
    thumbnail: str | None
    sources: tuple[str, ...]
    warnings: tuple[str, ...]


def slugify(text: str, max_len: int = _SLUG_MAX) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:max_len].rstrip("-") or "reel"


def note_filename(date: datetime, creator_handle: str | None, title: str) -> str:
    handle = slugify(creator_handle or "", 24) if creator_handle else ""
    parts = [date.strftime("%Y-%m-%d"), handle, slugify(title)]
    return "-".join(p for p in parts if p) + ".md"


def _yaml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def tag_for(category: str) -> str:
    """An Obsidian-safe tag: no spaces or ampersands."""
    return slugify(category.replace("&", "and"), 40)


def _blockquote(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.strip().splitlines())


def build_note(
    content: ReelContent,
    title: str,
    summary: str,
    takeaways: tuple[str, ...],
    imported_at: datetime,
    scenes: Sequence[Scene] = (),
    category: str | None = None,
    notion_url: str | None = None,
) -> str:
    """Frontmatter, what the model made of the video, then the verbatim caption and transcript.

    ``sources`` records which inputs actually produced the note (caption, transcript)
    so a wrong summary is attributable to its input rather than a mystery.
    """
    front = [
        "---",
        "type: reel",
        f"platform: {content.platform}",
        f"source: {content.url}",
        f"creator: {_yaml_str('@' + content.creator_handle) if content.creator_handle else 'null'}",
        f"creator_name: {_yaml_str(content.creator_name) if content.creator_name else 'null'}",
        f"posted: {content.posted.strftime('%Y-%m-%d') if content.posted else 'null'}",
        f"imported: {imported_at.strftime('%Y-%m-%d')}",
        f"duration_seconds: {round(content.duration) if content.duration else 'null'}",
        f"sources: [{', '.join(content.sources)}]",
        f"category: {_yaml_str(category) if category else 'null'}",
        f"notion: {_yaml_str(notion_url) if notion_url else 'null'}",
        f"tags: [reel{', ' + _yaml_str(tag_for(category)) if category else ''}]",
        "---",
    ]
    creator = f"@{content.creator_handle}" if content.creator_handle else (content.creator_name or "unknown creator")
    body = [f"# {title}", "", f"**Source:** [{content.platform}, {creator}]({content.url})"]
    if category:
        body += ["", f"**Category:** `{category}`"]
    if notion_url:
        body += ["", f"**Notion:** [{notion_url}]({notion_url})"]
    shots = [scene for scene in scenes if scene.image_path]
    # The platform thumbnail is only a fallback: the chosen screenshots say more, and a CDN link expires.
    if content.thumbnail and not shots:
        body += ["", f"![thumbnail]({content.thumbnail})"]
    body += ["", "## Summary", "", summary or "_No summary generated._"]
    if takeaways:
        body += ["", "## Key takeaways", ""] + [f"- {t}" for t in takeaways]
    if shots:
        body += ["", "## Screenshots", ""]
        for scene in shots:
            label = f"[{scene.time}] " if scene.time else ""
            # Angle brackets let a path with spaces through CommonMark and Obsidian alike.
            body += [f"![{scene.time or 'screenshot'}](<{scene.image_path}>)", f"*{label}{scene.description}*", ""]
        body.pop()
    if scenes:
        body += ["", "## Video breakdown", ""] + [scene.line() for scene in scenes]
    body += ["", "## Caption", ""]
    body += [_blockquote(content.caption)] if content.caption else ["_No caption._"]
    body += ["", "## Transcript", ""]
    body += [content.transcript] if content.transcript else ["_No transcript._"]
    if content.screen_text:
        body += ["", "## On-screen text", "", content.screen_text]
    if content.warnings:
        body += ["", "## Import warnings", ""] + [f"- {w}" for w in content.warnings]
    body += ["", f"*Imported by shortform-notes on {imported_at.strftime('%Y-%m-%d %H:%M UTC')}*", ""]
    return "\n".join(front + [""] + body)
