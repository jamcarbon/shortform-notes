"""The local setup server: state, config save, and CLI warning (no network, no scanning)."""

import json
import threading
from http.client import HTTPConnection

import pytest

from shortform_notes import config
from shortform_notes.web import server as web


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.env")
    srv = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    srv.shutdown()


def _json(conn, method, path, body=None):
    conn.request(method, path, body=json.dumps(body) if body else None, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())


def test_index_and_state(client):
    client.request("GET", "/")
    resp = client.getresponse()
    assert resp.status == 200 and b"Where should the summary run?" in resp.read()
    status, state = _json(client, "GET", "/api/state")
    assert status == 200
    assert state["config_exists"] is False
    assert "vaults" not in state  # no machine scanning


def test_save_config_writes_file_and_creates_dir(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web.shutil, "which", lambda name: None)
    out = tmp_path / "notes" / "reels"
    status, data = _json(
        client,
        "POST",
        "/api/config",
        {
            "SHORTFORM_NOTES_SUMMARY_PROVIDER": "claude-code",
            "SHORTFORM_NOTES_DIR": str(out),
            "OPENAI_API_KEY": "sk-secret",
        },
    )
    assert status == 200
    assert out.is_dir()
    saved = config.read_config_file(config.CONFIG_PATH)
    assert saved["SHORTFORM_NOTES_SUMMARY_PROVIDER"] == "claude-code" and saved["OPENAI_API_KEY"] == "sk-secret"
    assert data["detect"]["has_openai_key"] is True and "OPENAI_API_KEY" not in json.dumps(data["detect"])
    assert "claude" in data["warning"]  # CLI chosen but not on PATH: warning, not a hard error


def test_import_rejects_bad_url(client):
    status, data = _json(client, "POST", "/api/import", {"url": "https://example.com/x"})
    assert status == 422 and "not a supported" in data["error"]


def test_library_lists_imports_and_serves_only_output_files(client, tmp_path, monkeypatch):
    from shortform_notes import library

    out = tmp_path / "notes"
    (out / "assets" / "n").mkdir(parents=True)
    (out / "n.md").write_text("# note", encoding="utf-8")
    (out / "assets" / "n" / "00-03.png").write_bytes(b"\x89PNG")
    (out / "secret.env").write_text("TOKEN=x", encoding="utf-8")
    (tmp_path / "outside.png").write_bytes(b"\x89PNG")
    monkeypatch.setenv("SHORTFORM_NOTES_DIR", str(out))
    base = dict(url="u", platform="youtube", note_path="n.md", thumbnail="assets/n/00-03.png")
    library.record(out, library.LibraryEntry("2026-09-01T00:00:00", "Old", category="AI", **base))
    library.record(
        out,
        library.LibraryEntry(
            "2026-09-02T00:00:00", "New", category="Cooking & Recipes", **{**base, "note_path": "gone.md"}
        ),
    )
    status, data = _json(client, "GET", "/api/library")
    assert status == 200
    assert [e["title"] for e in data["entries"]] == ["New", "Old"]  # newest first
    assert [e["note_exists"] for e in data["entries"]] == [False, True]
    assert data["categories"] == ["Cooking & Recipes", "AI"]  # taxonomy order, not alphabetical

    for path, expected in [
        ("/files/n.md", 200),
        ("/files/assets/n/00-03.png", 200),
        ("/files/secret.env", 404),  # not a servable type
        ("/files/../outside.png", 404),  # outside the output folder
        ("/files/..%2Foutside.png", 404),
    ]:
        client.request("GET", path)
        resp = client.getresponse()
        resp.read()
        assert resp.status == expected, path
