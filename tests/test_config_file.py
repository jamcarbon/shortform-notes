"""Persistent config file: written by the web UI, read by everything else; real env wins."""

import os
from pathlib import Path

from shortform_notes import config


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "config.env"
    config.write_config_file({"SHORTFORM_NOTES_DIR": "/notes/reels", "OPENAI_API_KEY": "sk-x", "EMPTY": ""}, path)
    assert config.read_config_file(path) == {"SHORTFORM_NOTES_DIR": "/notes/reels", "OPENAI_API_KEY": "sk-x"}
    if os.name != "nt":  # Windows has no POSIX permission bits; chmod(0o600) only toggles read-only
        assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_read_ignores_comments_and_quotes(tmp_path):
    path = tmp_path / "config.env"
    path.write_text("# comment\n\nSHORTFORM_NOTES_AUDIENCE=\"a home cook\"\nBAD LINE\nX='y'\n")
    assert config.read_config_file(path) == {"SHORTFORM_NOTES_AUDIENCE": "a home cook", "X": "y"}


def test_missing_file_is_empty(tmp_path):
    assert config.read_config_file(tmp_path / "nope.env") == {}


def test_env_overrides_config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.env"
    config.write_config_file({"SHORTFORM_NOTES_SUMMARY_PROVIDER": "codex", "SHORTFORM_NOTES_DIR": "/from/file"}, path)
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("SHORTFORM_NOTES_DIR", "/from/env")
    s = config.load_settings(transcribe_provider="none")
    assert s.summary_provider == "codex"
    assert s.output_dir == Path("/from/env")
