"""A tiny local index of every import, for browsing by category.

One JSON file, ``<output_dir>/.library.json``, appended to after each import
whether or not Notion is configured, so local-only use still gets a library.
Paths inside it are relative to the output directory, so the folder can move
(or sync to another machine) without breaking the index.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from shortform_notes.urls import reel_key

logger = logging.getLogger(__name__)

LIBRARY_FILE = ".library.json"
_LOCK = threading.Lock()  # the web server imports on several threads


@dataclass(frozen=True)
class LibraryEntry:
    imported_at: str  # ISO 8601, UTC
    title: str
    url: str
    platform: str
    category: str | None
    note_path: str  # relative to the output directory
    notion_url: str | None = None
    creator: str | None = None
    summary: str = ""
    screenshot_count: int = 0
    thumbnail: str | None = None  # a kept screenshot, relative to the output directory; else the platform's URL

    @classmethod
    def from_dict(cls, data: dict) -> LibraryEntry:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def library_path(output_dir: Path) -> Path:
    return Path(output_dir) / LIBRARY_FILE


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("library index %s is unreadable (%s); starting a new one", path, exc)
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _same_reel(item: dict, key: str) -> bool:
    return bool(item.get("url")) and reel_key(str(item["url"])) == key


def record(output_dir: Path, entry: LibraryEntry) -> Path:
    """Add ``entry`` and write the index atomically.

    An earlier entry for the same note, or for the same video under any spelling of its link, is
    replaced: a re-import supersedes the import before it.
    """
    path = library_path(output_dir)
    key = reel_key(entry.url)
    with _LOCK:
        items = [item for item in _read(path) if item.get("note_path") != entry.note_path and not _same_reel(item, key)]
        items.append(asdict(entry))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    return path


def load(output_dir: Path) -> list[LibraryEntry]:
    """Every recorded import, newest first. Entries whose note was deleted are still listed."""
    entries = []
    for item in _read(library_path(output_dir)):
        try:
            entries.append(LibraryEntry.from_dict(item))
        except TypeError:  # a hand-edited entry missing a required field
            continue
    return sorted(entries, key=lambda e: e.imported_at, reverse=True)


def find(output_dir: Path, url: str) -> LibraryEntry | None:
    """The newest import of this video (any spelling of its link) whose note is still on disk.

    An entry whose note was deleted does not count: deleting the note is how a user asks for a fresh import.
    """
    key = reel_key(url)
    for entry in load(output_dir):
        if entry.url and reel_key(entry.url) == key and (Path(output_dir) / entry.note_path).is_file():
            return entry
    return None
