"""Terminal wizard writes the same config the web page does."""

from unittest.mock import AsyncMock, patch

from shortform_notes import config, notion_writer
from shortform_notes.setup_cli import run_setup


def test_wizard_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.env")
    answers = iter(["", "", "", str(tmp_path / "notes"), "", ""])  # last: Notion, default no
    path = run_setup(ask=lambda _prompt: next(answers), ask_secret=lambda _p: "unused")
    saved = config.read_config_file(path)
    assert saved["SHORTFORM_NOTES_SUMMARY_PROVIDER"] == "claude-code"
    assert saved["SHORTFORM_NOTES_TRANSCRIBE_PROVIDER"] == "local"
    assert saved["SHORTFORM_NOTES_DIR"] == str(tmp_path / "notes")
    assert (tmp_path / "notes").is_dir()
    assert "OPENAI_API_KEY" not in saved
    assert "NOTION_TOKEN" not in saved


def test_wizard_openai_key_and_reprompt(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.env")
    answers = iter(["9", "3", "1", "2", str(tmp_path / "n"), "a home cook", "n"])
    secrets = iter(["", "sk-test"])  # first empty answer is rejected
    path = run_setup(ask=lambda _p: next(answers), ask_secret=lambda _p: next(secrets))
    saved = config.read_config_file(path)
    assert saved["SHORTFORM_NOTES_SUMMARY_PROVIDER"] == "openai"
    assert saved["OPENAI_API_KEY"] == "sk-test"
    assert saved["SHORTFORM_NOTES_TRANSCRIBE_PROVIDER"] == "openai"
    assert saved["SHORTFORM_NOTES_AUDIENCE"] == "a home cook"
    assert saved["SHORTFORM_NOTES_OCR"] == "1" and saved["SHORTFORM_NOTES_OCR_PROVIDER"] == "local"


def test_wizard_connects_notion_from_a_share_link(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.env")
    db = "0123456789abcdef0123456789abcdef"
    link = f"https://www.notion.so/me/Reels-{db}?v={'9' * 32}"
    answers = iter(["", "3", "", str(tmp_path / "n"), "", "y", link])
    check = AsyncMock(return_value="connected: data source ds-1 with 1 columns")
    with patch.object(notion_writer, "check_connection", check):
        path = run_setup(ask=lambda _p: next(answers), ask_secret=lambda _p: "ntn_secret")
    saved = config.read_config_file(path)
    assert saved["NOTION_TOKEN"] == "ntn_secret" and saved["NOTION_DATABASE_ID"] == db
    assert check.call_args.args[0].notion_database_id == db
