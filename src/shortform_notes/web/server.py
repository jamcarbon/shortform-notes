"""Local setup and import UI: ``shortform-notes web``.

No extra dependencies: stdlib ``http.server`` bound to 127.0.0.1 only, serving
one HTML page and a few JSON endpoints. It writes the same config file the CLI,
MCP server and Claude Code skill read, so choices made here apply everywhere.

The Library view reads the local index (``library.py``) and serves notes and
screenshots from the output folder through ``/files/``, never outside it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import unquote, urlparse

from shortform_notes import __version__, config, library
from shortform_notes.pipeline import ReelImportError, import_reel

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8765
HOST = "127.0.0.1"
SAVABLE_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "SHORTFORM_NOTES_SUMMARY_PROVIDER",
    "SHORTFORM_NOTES_TRANSCRIBE_PROVIDER",
    "SHORTFORM_NOTES_DIR",
    "SHORTFORM_NOTES_AUDIENCE",
    "SHORTFORM_NOTES_CLAUDE_CODE_MODEL",
    "SHORTFORM_NOTES_CODEX_MODEL",
    "SHORTFORM_NOTES_WHISPER_MODEL",
    "SHORTFORM_NOTES_OCR",
    "SHORTFORM_NOTES_OCR_PROVIDER",
    "SHORTFORM_NOTES_OCR_FPS",
)
# What /files/ will serve from the output folder; anything else is a 404.
SERVABLE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".md": "text/plain; charset=utf-8",  # shown as text; "Open" hands it to the desktop app
}


# state (no machine scanning: only our own config file and package facts)


def describe() -> dict:
    current = config.read_config_file()
    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "config_path": str(config.CONFIG_PATH),
        "config_exists": config.CONFIG_PATH.exists(),
        "home": str(Path.home()),
        "repo_dir": str(Path(__file__).resolve().parents[3]),
        "faster_whisper": config._has_module("faster_whisper"),
        "current": {k: current.get(k, "") for k in SAVABLE_KEYS if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")},
        "has_openai_key": bool(current.get("OPENAI_API_KEY")),
        "has_anthropic_key": bool(current.get("ANTHROPIC_API_KEY")),
    }


def output_dir() -> Path:
    try:
        return config.load_settings().output_dir
    except ValueError:  # a bad provider name in the config file; the folder is still readable
        return Path(config.read_config_file().get("SHORTFORM_NOTES_DIR") or config.DEFAULT_OUTPUT_DIR).expanduser()


def library_payload() -> dict:
    """Every import, newest first, plus the categories present (for the filter) in taxonomy order."""
    root = output_dir()
    entries = []
    for entry in library.load(root):
        item = asdict(entry)
        item["note_exists"] = (root / entry.note_path).exists()
        entries.append(item)
    try:
        taxonomy = config.load_settings().categories
    except ValueError:
        taxonomy = config.DEFAULT_CATEGORIES
    present = {e["category"] for e in entries if e["category"]}
    ordered = [c for c in taxonomy if c in present] + sorted(present - set(taxonomy))
    return {"output_dir": str(root), "entries": entries, "categories": ordered}


def resolve_served(relative: str) -> Path | None:
    """A file under the output folder with a servable type, or None. Blocks ``..`` and absolute paths."""
    root = output_dir().resolve()
    try:
        target = (root / unquote(relative)).resolve()
    except (OSError, ValueError):
        return None
    if not target.is_relative_to(root) or target.suffix.lower() not in SERVABLE_TYPES or not target.is_file():
        return None
    return target


def cli_warning(provider: str) -> str | None:
    """Only after the user picked a CLI backend: a PATH lookup for that one binary."""
    binary = {"claude-code": "claude", "codex": "codex"}.get(provider)
    if binary and not shutil.which(binary):
        return f"`{binary}` is not on your PATH. Install it before importing."
    return None


# HTTP handler


def _load_index() -> bytes:
    return files("shortform_notes.web").joinpath("index.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    server_version = f"shortform-notes/{__version__}"

    def log_message(self, fmt: str, *args) -> None:  # quiet by default; -v shows them
        logger.info("%s " + fmt, self.address_string(), *args)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        path = urlparse(self.path).path
        if path.startswith("/files/"):
            self._file(path.removeprefix("/files/"))
        elif path == "/api/library":
            self._json(200, library_payload())
        elif self.path in ("/", "/index.html"):
            body = _load_index()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self._json(200, describe())
        else:
            self._json(404, {"error": "not found"})

    def _file(self, relative: str) -> None:
        target = resolve_served(relative)
        if target is None:
            self._json(404, {"error": "not found"})
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", SERVABLE_TYPES[target.suffix.lower()])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/config":
            self._save_config()
        elif self.path == "/api/import":
            self._import()
        elif self.path == "/api/open":
            self._open_folder()
        else:
            self._json(404, {"error": "not found"})

    def _save_config(self) -> None:
        data = self._read_json()
        existing = config.read_config_file()
        merged = {**existing, **{k: str(data.get(k, existing.get(k, ""))).strip() for k in SAVABLE_KEYS}}
        out_dir = merged.get("SHORTFORM_NOTES_DIR")
        if out_dir:
            try:
                Path(out_dir).expanduser().mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self._json(400, {"error": f"cannot create folder {out_dir}: {exc}"})
                return
        path = config.write_config_file(merged)
        self._json(
            200,
            {
                "saved": str(path),
                "detect": describe(),
                "warning": cli_warning(merged.get("SHORTFORM_NOTES_SUMMARY_PROVIDER", "")),
            },
        )

    def _import(self) -> None:
        url = str(self._read_json().get("url", "")).strip()
        if not url:
            self._json(400, {"error": "missing url"})
            return
        try:
            result = asyncio.run(import_reel(url, config.load_settings()))
        except ReelImportError as exc:
            self._json(422, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 (surface to the page, keep the server alive)
            logger.exception("import crashed for %s", url)
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        note = result.path.read_text(encoding="utf-8")
        self._json(200, {**result.to_dict(), "note": note})

    def _open_folder(self) -> None:
        raw = str(self._read_json().get("path", ""))
        # A library entry sends its note path relative to the output folder.
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = resolve_served(raw) or output_dir() / "__missing__"
        if not target.exists():
            self._json(404, {"error": "no such path"})
            return
        opener = {"darwin": ["open"], "win32": ["explorer"]}.get(sys.platform, ["xdg-open"])
        try:
            subprocess.Popen([*opener, str(target)])
        except OSError as exc:
            self._json(500, {"error": str(exc)})
            return
        self._json(200, {"opened": str(target)})


# entry point


def _bind(preferred: int, attempts: int = 20) -> ThreadingHTTPServer:
    """Bind the preferred port, else try the next ones, so a busy port does not block a first run."""
    for port in range(preferred, preferred + attempts):
        try:
            return ThreadingHTTPServer((HOST, port), Handler)
        except OSError as exc:
            if exc.errno not in (48, 98, 10048):  # EADDRINUSE on macOS / Linux / Windows
                raise
    raise OSError(f"no free port in {preferred}-{preferred + attempts - 1}")


def serve(port: int | None = None, open_browser: bool = True) -> None:
    server = _bind(port or int(os.environ.get("SHORTFORM_NOTES_PORT") or DEFAULT_PORT))
    port = server.server_address[1]
    url = f"http://{HOST}:{port}/"
    print(f"shortform-notes {__version__}: setup and import UI at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
