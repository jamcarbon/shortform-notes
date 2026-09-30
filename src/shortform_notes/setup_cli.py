"""Terminal setup wizard: ``shortform-notes setup``.

Asks the same questions as the web page (summary backend, transcription,
folder, audience) and writes the same config file, so either path leaves the
tool in an identical state. It then checks the GPU for local Whisper and,
optionally, connects a Notion database.
"""

from __future__ import annotations

import asyncio
import getpass
import sys
from dataclasses import replace
from pathlib import Path

from shortform_notes import config, notion_writer

NOTION_INTEGRATIONS_URL = "https://www.notion.so/profile/integrations"

SUMMARY_CHOICES = [
    ("claude-code", "Claude Code", "uses your Claude subscription through the claude CLI, no API key"),
    ("codex", "Codex CLI", "uses your ChatGPT subscription through the codex CLI, no API key"),
    ("openai", "OpenAI API key", "pay per use, about $0.001 per reel; also enables the best transcription"),
    ("anthropic", "Anthropic API key", "pay per use with Claude via the API"),
    ("none", "No summary", "save the caption, transcript and metadata only"),
]
OCR_CHOICES = [
    ("off", "Off", "caption and transcript only (default)"),
    ("local", "On, local OCR", "free, runs on this computer; slower per reel"),
    ("openai", "On, OpenAI vision", "about $0.005 per 30 s video at 1 frame per second"),
    ("anthropic", "On, Claude vision", "about $0.05 per 30 s video at 1 frame per second with claude-sonnet-5"),
]
TRANSCRIBE_CHOICES = [
    ("openai", "OpenAI", "about $0.0045 per minute of video; needs an OpenAI API key"),
    ("local", "Offline on this computer", "free and private; the first run downloads a 75 MB model"),
    ("none", "Skip transcripts", "caption and metadata only"),
]


def _say(text: str = "") -> None:
    print(text, flush=True)


def _pick(question: str, choices: list[tuple[str, str, str]], default: str, ask=input) -> str:
    _say(question)
    for i, (_, title, desc) in enumerate(choices, 1):
        _say(f"  {i}) {title}: {desc}")
    ids = [c[0] for c in choices]
    default_index = ids.index(default) + 1
    while True:
        raw = ask(f"Choose 1-{len(choices)} [{default_index}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return ids[int(raw) - 1]
        _say("Enter a number from the list.")


def _ask_secret(label: str, existing: str, ask_secret=getpass.getpass) -> str:
    if existing:
        value = ask_secret(f"{label} (press Enter to keep the saved one): ").strip()
        return value or existing
    while True:
        value = ask_secret(f"{label}: ").strip()
        if value:
            return value
        _say("A key is required for this option.")


def report_gpu(requested: str = "auto") -> str:
    """Say which device local Whisper will use, and why. Returns the device."""
    from shortform_notes import transcribe

    if not config._has_module("faster_whisper"):
        _say('   Local Whisper is not installed yet: pip install "shortform-notes[local]"')
        return "cpu"
    device, compute = transcribe.resolve_device(requested)
    if device == "cuda":
        _say(f"   GPU found: local transcription will run on cuda ({compute}).")
        _say('   If the first import logs "retrying on the CPU", install the CUDA libraries:')
        _say('   pip install "shortform-notes[cuda]"')
    else:
        _say(f"   No CUDA GPU visible to ctranslate2: local transcription will run on the CPU ({compute}).")
    return device


def ask_notion(current: dict[str, str], ask=input, ask_secret=getpass.getpass) -> tuple[str, str]:
    """Optional Notion connection. Returns (token, database id); empty strings when skipped."""
    token, database = current.get("NOTION_TOKEN", ""), current.get("NOTION_DATABASE_ID", "")
    _say("5/5  Also save each reel as a page in a Notion database? (optional)")
    _say(f"   1. Create an internal integration at {NOTION_INTEGRATIONS_URL} and copy its")
    _say("      Internal Integration Secret (starts with ntn_ or secret_).")
    _say("   2. Open (or create) the database in Notion, click ••• → Connections, and add the integration.")
    _say("   3. Copy the database link (••• → Copy link); the id is taken from it.")
    _say("   Missing columns (Source URL, Platform, Category, Tags...) are added automatically.")
    default = "y" if token and database else "n"
    answer = ask(f"   Connect Notion? y/n [{default}]: ").strip().lower() or default
    if not answer.startswith("y"):
        return "", ""
    token = _ask_secret("   Notion integration secret", token, ask_secret)
    while True:
        raw = ask(f"   Database link or id [{database or 'required'}]: ").strip() or database
        database = config.normalize_notion_id(raw)
        if database:
            break
    settings = config.load_settings(transcribe_provider="none", summary_provider="none")
    try:
        check = notion_writer.check_connection(replace(settings, notion_token=token, notion_database_id=database))
        _say("   " + asyncio.run(check))
    except Exception as exc:  # noqa: BLE001 (shown to the user; they can fix it and re-run setup)
        _say(f"   Could not reach the database yet: {exc}")
        _say("   Saved anyway; imports will warn until this is fixed.")
    return token, database


def run_setup(ask=input, ask_secret=getpass.getpass) -> Path:
    """Interactive wizard. ``ask``/``ask_secret`` are injectable for tests."""
    current = config.read_config_file()
    _say("shortform-notes setup")
    _say("Answers are saved to " + str(config.CONFIG_PATH) + ". Press Enter to accept a default.")
    _say()

    summary = _pick(
        "1/5  Where should the summary run?",
        SUMMARY_CHOICES,
        current.get("SHORTFORM_NOTES_SUMMARY_PROVIDER") or "claude-code",
        ask,
    )
    openai_key = current.get("OPENAI_API_KEY", "")
    anthropic_key = current.get("ANTHROPIC_API_KEY", "")
    if summary == "openai":
        openai_key = _ask_secret("OpenAI API key", openai_key, ask_secret)
    elif summary == "anthropic":
        anthropic_key = _ask_secret("Anthropic API key", anthropic_key, ask_secret)
    _say()

    transcribe_default = current.get("SHORTFORM_NOTES_TRANSCRIBE_PROVIDER") or ("openai" if openai_key else "local")
    transcribe = _pick("2/5  How should audio be transcribed?", TRANSCRIBE_CHOICES, transcribe_default, ask)
    if transcribe == "openai" and not openai_key:
        openai_key = _ask_secret("OpenAI API key", "", ask_secret)
    whisper_device = current.get("SHORTFORM_NOTES_WHISPER_DEVICE", "")
    if transcribe == "local":
        report_gpu(whisper_device or "auto")
    _say()

    ocr_default = (
        current.get("SHORTFORM_NOTES_OCR_PROVIDER", "off") if current.get("SHORTFORM_NOTES_OCR") == "1" else "off"
    )
    _say("Optional: read on-screen text from video frames. This downloads the full video and reads")
    _say("one frame per cut, or one per second if ffmpeg is not installed. Set SHORTFORM_NOTES_OCR_FPS")
    _say("to read on a clock instead; 0 means every frame, which costs about 30 times more on a paid backend.")
    ocr_choice = _pick("Read on-screen text?", OCR_CHOICES, ocr_default, ask)
    if ocr_choice == "openai" and not openai_key:
        openai_key = _ask_secret("OpenAI API key", "", ask_secret)
    if ocr_choice == "anthropic" and not anthropic_key:
        anthropic_key = _ask_secret("Anthropic API key", "", ask_secret)
    _say()

    default_dir = current.get("SHORTFORM_NOTES_DIR") or str(Path.home() / "shortform-notes")
    folder = ask(f"3/5  Folder for notes [{default_dir}]: ").strip() or default_dir
    Path(folder).expanduser().mkdir(parents=True, exist_ok=True)
    _say()

    default_audience = current.get("SHORTFORM_NOTES_AUDIENCE", "")
    audience = ask(
        f"4/5  Who are the notes for? Optional, shapes the summary [{default_audience or 'the reader'}]: "
    ).strip()
    audience = audience or default_audience
    _say()

    notion_token, notion_db = ask_notion(current, ask, ask_secret)
    _say()

    path = config.write_config_file(
        {
            **current,
            "SHORTFORM_NOTES_SUMMARY_PROVIDER": summary,
            "SHORTFORM_NOTES_TRANSCRIBE_PROVIDER": transcribe,
            "SHORTFORM_NOTES_OCR": "1" if ocr_choice != "off" else "",
            "SHORTFORM_NOTES_OCR_PROVIDER": ocr_choice if ocr_choice != "off" else "",
            "SHORTFORM_NOTES_DIR": folder,
            "SHORTFORM_NOTES_AUDIENCE": audience,
            "OPENAI_API_KEY": openai_key,
            "ANTHROPIC_API_KEY": anthropic_key,
            "NOTION_TOKEN": notion_token,
            "NOTION_DATABASE_ID": notion_db,
            "SHORTFORM_NOTES_WHISPER_DEVICE": whisper_device,
        }
    )
    _say(f"Saved {path}")
    _say()
    _say("Next: import a link with")
    _say("  shortform-notes https://www.instagram.com/p/DS3DPehEnpA/")
    if summary in ("claude-code", "codex"):
        tool = "Claude Code" if summary == "claude-code" else "Codex"
        _say(f"To use it inside {tool}, run: shortform-notes web  and copy the prompt on the last page.")
    return path


def main() -> int:
    if not sys.stdin.isatty():
        _say("shortform-notes setup needs an interactive terminal. Run: shortform-notes web")
        return 2
    try:
        run_setup()
    except (KeyboardInterrupt, EOFError):
        _say("\nSetup cancelled. Nothing was saved.")
        return 1
    return 0
