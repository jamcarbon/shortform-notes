"""Runtime settings, all from environment variables (CLI flags override them).

Everything is optional. shortform-notes picks the best available backend for each
LLM step, so it works with:

* an API key           OPENAI_API_KEY and/or ANTHROPIC_API_KEY
* a coding-agent CLI   ``claude`` (Claude Code) or ``codex`` on PATH; summaries
                       run on that CLI's subscription, no key needed
* nothing at all       caption and metadata note; add ``[local]`` for an offline
                       Whisper transcript
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT_DIR = "reels"
# Written by the setup UI (`shortform-notes web`); real environment variables always win over it.
CONFIG_PATH = Path(os.environ.get("SHORTFORM_NOTES_CONFIG") or "~/.config/shortform-notes/config.env").expanduser()
# Matched mid-tier defaults on both vendors: neither bargain-bin quality nor silent flagship
# spend. Every one is overridable with the SHORTFORM_NOTES_*_MODEL variables below.
DEFAULT_OPENAI_TRANSCRIBE_MODEL = "gpt-transcribe"
DEFAULT_OPENAI_SUMMARY_MODEL = "gpt-5-mini"
DEFAULT_ANTHROPIC_SUMMARY_MODEL = "claude-sonnet-5"
DEFAULT_WHISPER_MODEL = "base"

SUMMARY_PROVIDERS = ("openai", "anthropic", "claude-code", "codex", "none")
# Every summary backend takes images: the APIs as image blocks, `claude -p` via its
# stream-json stdin, `codex exec` via `-i`. Only "none", which makes no call at all, cannot.
VISION_SUMMARY_PROVIDERS = ("openai", "anthropic", "claude-code", "codex")
# Agentic vision needs a backend that can open files on its own. The API backends get one frozen
# look at whatever we put in the request, so they can only ever run the one-shot mode.
AGENTIC_VISION_PROVIDERS = ("claude-code", "codex")
OCR_PROVIDERS = ("local", "openai", "anthropic")
DEFAULT_OCR_FPS = 1.0  # one frame per second; 0 means every frame
DEFAULT_OCR_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_OCR_ANTHROPIC_MODEL = "claude-sonnet-5"
TRANSCRIBE_PROVIDERS = ("openai", "local", "none")
WHISPER_DEVICES = ("auto", "cuda", "cpu")
_FALSE = {"0", "false", "no", "off"}

# Notion. Only the token and the target database are required; the version is pinned so a Notion
# release cannot silently change request shapes under us (checked current on 2026-09-30).
DEFAULT_NOTION_VERSION = "2026-03-11"
# The whole video gets exactly one of these. Short and broad on purpose: a Notion select with
# forty near-synonyms is not browsable. Override with SHORTFORM_NOTES_CATEGORIES or --categories.
DEFAULT_CATEGORIES = (
    "Cooking & Recipes",
    "Fitness & Health",
    "Tech & Software",
    "AI",
    "Business & Money",
    "Productivity & Learning",
    "Science & Education",
    "DIY & Home",
    "Travel",
    "Style & Beauty",
    "Entertainment & Humor",
    "News & Commentary",
    "Other",
)


def parse_categories(raw: str | None) -> tuple[str, ...]:
    """Comma-separated taxonomy. Notion select options may not contain commas, so that is the separator."""
    items = [c.strip() for c in (raw or "").split(",")]
    seen: dict[str, None] = {}
    for item in items:
        if item and item.lower() not in {k.lower() for k in seen}:
            seen[item] = None
    return tuple(seen) or DEFAULT_CATEGORIES


@dataclass(frozen=True)
class Settings:
    output_dir: Path
    openai_api_key: str | None
    anthropic_api_key: str | None
    summary_provider: str  # one of SUMMARY_PROVIDERS
    transcribe_provider: str  # one of TRANSCRIBE_PROVIDERS
    openai_transcribe_model: str
    openai_summary_model: str
    anthropic_summary_model: str
    claude_code_model: str | None  # None means the CLI's own default
    codex_model: str | None
    whisper_model: str
    audience: str
    ocr: bool  # read on-screen text from sampled video frames (costs more time, and money on API backends)
    ocr_provider: str  # one of OCR_PROVIDERS
    ocr_fps: float  # frames sampled per second, for OCR and vision alike; 0 = every frame
    ocr_openai_model: str
    ocr_anthropic_model: str  # who the note is for; shapes the summary prompt
    vision: bool  # attach the sampled frames to the summary call so the model sees the video
    vision_agentic: bool  # let an agent backend open the full-resolution frames itself
    fps_explicit: bool  # a rate was asked for, so use it instead of ffmpeg's cut-aware sampling
    # Defaults below keep Settings(...) constructible the way it was before these fields existed.
    whisper_device: str = "auto"  # one of WHISPER_DEVICES; auto picks CUDA when ctranslate2 sees a GPU
    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    notion_token: str | None = None
    notion_database_id: str | None = None
    notion_version: str = DEFAULT_NOTION_VERSION
    notion: bool = True  # False turns the Notion write off even when it is configured (--no-notion)

    @property
    def can_write_notion(self) -> bool:
        return self.notion and bool(self.notion_token and self.notion_database_id)

    @property
    def can_transcribe(self) -> bool:
        return self.transcribe_provider != "none"

    @property
    def can_see_video(self) -> bool:
        """Vision was asked for *and* a summary backend that can read images is selected."""
        return self.vision and self.summary_provider in VISION_SUMMARY_PROVIDERS

    @property
    def vision_is_agentic(self) -> bool:
        """Agentic vision was asked for *and* the backend is an agent that can go and look."""
        return self.can_see_video and self.vision_agentic and self.summary_provider in AGENTIC_VISION_PROVIDERS

    @property
    def vision_is_metered(self) -> bool:
        """True when frames cost money per call; the CLI backends bill to a subscription."""
        return self.summary_provider in ("openai", "anthropic")

    @property
    def can_summarize(self) -> bool:
        return self.summary_provider != "none"


def _has_module(name: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detect_summary_provider(openai_key: str | None, anthropic_key: str | None) -> str:
    """Free-to-run first: a flat-rate CLI the user already pays for beats spending per token.

    "No API key required" is the tool's whole pitch, so auto-detection must never
    quietly bill an API when ``claude`` or ``codex`` is sitting on PATH. A key is
    still used when it is the only thing available, and ``--summary`` overrides all of it.
    """
    if shutil.which("claude"):
        return "claude-code"
    if shutil.which("codex"):
        return "codex"
    if openai_key and _has_module("openai"):
        return "openai"
    if anthropic_key and _has_module("anthropic"):
        return "anthropic"
    return "none"


def detect_transcribe_provider(openai_key: str | None) -> str:
    if openai_key and _has_module("openai"):
        return "openai"
    if _has_module("faster_whisper"):
        return "local"
    return "none"


def read_config_file(path: Path | None = None) -> dict[str, str]:
    """Parse a KEY=VALUE file (comments and blank lines ignored). A missing file yields {}."""
    path = path or CONFIG_PATH  # resolved at call time so tests (and SHORTFORM_NOTES_CONFIG) can redirect it
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def write_config_file(values: dict[str, str], path: Path | None = None) -> Path:
    """Write KEY=VALUE pairs (empty values dropped), owner-readable only since it may hold API keys."""
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# shortform-notes configuration, written by shortform-notes setup. Edit freely.", ""]
    lines += [f"{k}={v}" for k, v in sorted(values.items()) if v]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _env() -> dict[str, str]:
    """Config file values overlaid by the real environment (env wins)."""
    return {**read_config_file(), **os.environ}


def _validate(value: str, allowed: tuple[str, ...], what: str) -> str:
    value = value.lower().strip()
    if value not in allowed:
        raise ValueError(f"unknown {what} {value!r}; use one of: {', '.join(allowed)}")
    return value


def load_settings(
    output_dir: str | None = None,
    summary_provider: str | None = None,
    transcribe_provider: str | None = None,
    ocr: bool | None = None,
    ocr_provider: str | None = None,
    ocr_fps: float | None = None,
    vision: bool | None = None,
    vision_agentic: bool | None = None,
    notion: bool | None = None,
    categories: str | None = None,
    whisper_device: str | None = None,
) -> Settings:
    """Build settings from env, with optional explicit overrides (CLI flags win)."""
    env = _env()
    openai_key = env.get("OPENAI_API_KEY") or None
    anthropic_key = env.get("ANTHROPIC_API_KEY") or None

    summary = summary_provider or env.get("SHORTFORM_NOTES_SUMMARY_PROVIDER") or "auto"
    summary = detect_summary_provider(openai_key, anthropic_key) if summary == "auto" else summary
    transcribe = transcribe_provider or env.get("SHORTFORM_NOTES_TRANSCRIBE_PROVIDER") or "auto"
    if env.get("SHORTFORM_NOTES_TRANSCRIBE", "1").lower() in _FALSE:  # legacy off switch
        transcribe = "none"
    transcribe = detect_transcribe_provider(openai_key) if transcribe == "auto" else transcribe

    ocr_on = (env.get("SHORTFORM_NOTES_OCR", "0").lower() not in _FALSE) if ocr is None else ocr
    ocr_backend = ocr_provider or env.get("SHORTFORM_NOTES_OCR_PROVIDER") or "auto"
    if ocr_backend == "auto":
        ocr_backend = "openai" if openai_key and _has_module("openai") else "local"
    fps_raw = env.get("SHORTFORM_NOTES_OCR_FPS", "")
    fps = float(fps_raw) if ocr_fps is None and fps_raw else (DEFAULT_OCR_FPS if ocr_fps is None else ocr_fps)
    # SHORTFORM_NOTES_VISION takes 0/1 as before, and "agentic" to turn on the mode as well.
    notion_token = env.get("NOTION_TOKEN") or None
    notion_db = normalize_notion_id(env.get("NOTION_DATABASE_ID") or "") or None
    notion_on = notion if notion is not None else True
    notion_ready = notion_on and bool(notion_token and notion_db)
    # Unset, vision follows Notion: the screenshots a Notion page shows are chosen by the model
    # looking at the frames, so writing to Notion without vision would give pages with no images.
    vision_raw = env.get("SHORTFORM_NOTES_VISION", "").lower() or ("1" if notion_ready else "0")
    vision_on = (vision_raw not in _FALSE) if vision is None else vision
    agentic_on = (vision_raw == "agentic") if vision_agentic is None else vision_agentic
    return Settings(
        output_dir=Path(output_dir or env.get("SHORTFORM_NOTES_DIR") or DEFAULT_OUTPUT_DIR).expanduser(),
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        summary_provider=_validate(summary, SUMMARY_PROVIDERS, "summary provider"),
        transcribe_provider=_validate(transcribe, TRANSCRIBE_PROVIDERS, "transcribe provider"),
        openai_transcribe_model=env.get("SHORTFORM_NOTES_TRANSCRIBE_MODEL", DEFAULT_OPENAI_TRANSCRIBE_MODEL),
        openai_summary_model=env.get("SHORTFORM_NOTES_OPENAI_MODEL", DEFAULT_OPENAI_SUMMARY_MODEL),
        anthropic_summary_model=env.get("SHORTFORM_NOTES_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_SUMMARY_MODEL),
        claude_code_model=env.get("SHORTFORM_NOTES_CLAUDE_CODE_MODEL") or None,
        codex_model=env.get("SHORTFORM_NOTES_CODEX_MODEL") or None,
        whisper_model=env.get("SHORTFORM_NOTES_WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
        audience=env.get("SHORTFORM_NOTES_AUDIENCE", "the reader"),
        ocr=ocr_on,
        ocr_provider=_validate(ocr_backend, OCR_PROVIDERS, "OCR provider"),
        ocr_fps=max(0.0, fps),
        ocr_openai_model=env.get("SHORTFORM_NOTES_OCR_OPENAI_MODEL", DEFAULT_OCR_OPENAI_MODEL),
        ocr_anthropic_model=env.get("SHORTFORM_NOTES_OCR_ANTHROPIC_MODEL", DEFAULT_OCR_ANTHROPIC_MODEL),
        vision=vision_on,
        vision_agentic=agentic_on,
        fps_explicit=ocr_fps is not None or bool(fps_raw),
        whisper_device=_validate(
            whisper_device or env.get("SHORTFORM_NOTES_WHISPER_DEVICE") or "auto", WHISPER_DEVICES, "whisper device"
        ),
        categories=parse_categories(categories or env.get("SHORTFORM_NOTES_CATEGORIES")),
        notion_token=notion_token,
        notion_database_id=notion_db,
        notion_version=env.get("NOTION_VERSION") or DEFAULT_NOTION_VERSION,
        notion=notion_on,
    )


def normalize_notion_id(raw: str) -> str:
    """Accept a bare id, a dashed UUID, or the database's full share URL; return the 32-hex id."""
    text = raw.strip()
    # A share URL ends ".../<title>-<32 hex>?v=<view id>"; the view id is also 32 hex, so drop the query first.
    text = text.split("?", 1)[0].split("#", 1)[0]
    matches = re.findall(r"[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}", text)
    return matches[-1].replace("-", "").lower() if matches else text
