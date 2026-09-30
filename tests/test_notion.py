"""Notion writer, screenshot selection, categories and the library index. No network: httpx.MockTransport."""

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from shortform_notes import config, library, notion_writer, ocr, pipeline, urls
from shortform_notes.note import ReelContent, Scene, build_note
from shortform_notes.summarize import Summary, _coerce, match_category, summary_schema
from tests.test_pipeline import NOW, settings

DB = "0123456789abcdef0123456789abcdef"
DS = "ds-1"


def content(**overrides) -> ReelContent:
    base = ReelContent(
        url="https://www.youtube.com/shorts/abc",
        platform="youtube",
        caption="Brownies #baking #Dessert, #baking",
        transcript="Add espresso. " * 300,  # > 2000 chars, so it must be chunked
        screen_text=None,
        title=None,
        creator_handle="cook",
        creator_name="A Cook",
        posted=datetime(2026, 8, 11, tzinfo=timezone.utc),
        duration=39.4,
        thumbnail="https://cdn.example/t.jpg",
        sources=("transcript", "video"),
        warnings=(),
    )
    return replace(base, **overrides)


def notion_settings(tmp_path: Path, **overrides):
    return replace(settings(tmp_path), notion_token="secret_x", notion_database_id=DB, **overrides)


class FakeNotion:
    """Just enough of the Notion API to record what the writer sends."""

    def __init__(
        self, properties: dict | None = None, fail_patch_schema: bool = False, existing: list[dict] | None = None
    ) -> None:
        self.properties = properties if properties is not None else {"Title": {"type": "title"}}
        self.fail_patch_schema = fail_patch_schema
        self.existing = existing or []  # pages the data source query returns
        self.calls: list[tuple[str, str, object]] = []
        self.uploads = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path.removeprefix("/v1"), request.method
        body: object = None
        if request.headers.get("content-type", "").startswith("application/json"):
            body = json.loads(request.content)
        self.calls.append((method, path, body))
        assert request.headers["Notion-Version"] == config.DEFAULT_NOTION_VERSION
        assert request.headers["Authorization"] == "Bearer secret_x"
        if method == "GET" and path == f"/databases/{DB}":
            return httpx.Response(200, json={"id": DB, "data_sources": [{"id": DS, "name": "Reels"}]})
        if method == "GET" and path == f"/data_sources/{DS}":
            return httpx.Response(200, json={"id": DS, "properties": self.properties})
        if method == "PATCH" and path == f"/data_sources/{DS}":
            if self.fail_patch_schema:
                return httpx.Response(403, json={"code": "restricted_resource", "message": "no"})
            return httpx.Response(200, json={"id": DS})
        if method == "POST" and path == f"/data_sources/{DS}/query":
            return httpx.Response(200, json={"object": "list", "results": self.existing, "has_more": False})
        if method == "POST" and path == "/file_uploads":
            self.uploads += 1
            return httpx.Response(200, json={"id": f"up-{self.uploads}", "status": "pending"})
        if method == "POST" and path.endswith("/send"):
            assert b'name="file"' in request.content  # multipart, field "file"
            return httpx.Response(200, json={"id": path.split("/")[2], "status": "uploaded"})
        if method == "POST" and path == "/pages":
            return httpx.Response(200, json={"id": "page-1", "url": "https://www.notion.so/page-1"})
        if method == "PATCH" and path.startswith("/blocks/"):
            return httpx.Response(200, json={})
        if method == "PATCH" and path.startswith("/pages/"):
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"code": "object_not_found", "message": path})

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def find(self, method: str, path: str) -> list:
        return [body for m, p, body in self.calls if m == method and p == path]


def shots(tmp_path: Path) -> list[Scene]:
    image = tmp_path / "assets" / "n" / "00-03.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\x89PNG fake")
    return [
        Scene("00:01", "talking head", keep=False, reason="face"),
        Scene("00:03", "the finished tray", keep=True, reason="result").with_image("assets/n/00-03.png"),
    ]


async def test_page_goes_to_the_data_source_with_properties_images_and_chunked_text(tmp_path):
    fake = FakeNotion()
    async with fake.client() as http:
        page = await notion_writer.create_reel_page(
            notion_settings(tmp_path),
            content(),
            "Better Brownies",
            "Three hacks.",
            ("espresso",),
            shots(tmp_path),
            "Cooking & Recipes",
            base_dir=tmp_path,
            http=http,
        )
    assert page.url == "https://www.notion.so/page-1"
    # Missing columns were added to the data source, typed.
    added = fake.find("PATCH", f"/data_sources/{DS}")[0]["properties"]
    assert added["Category"] == {"select": {}} and added["Tags"] == {"multi_select": {}}
    created = fake.find("POST", "/pages")[0]
    assert created["parent"] == {"type": "data_source_id", "data_source_id": DS}
    props = created["properties"]
    assert props["Title"]["title"][0]["text"]["content"] == "Better Brownies"  # title column found by type
    assert props["Category"] == {"select": {"name": "Cooking & Recipes"}}
    assert props["Platform"] == {"select": {"name": "youtube"}}
    assert props["Duration (s)"] == {"number": 39}
    assert props["Posted"] == {"date": {"start": "2026-08-11"}}
    assert [t["name"] for t in props["Tags"]["multi_select"]] == ["baking", "dessert"]  # deduped, no commas
    images = [b for b in created["children"] if b["type"] == "image"]
    assert len(images) == 1  # only the kept scene
    assert images[0]["image"]["file_upload"] == {"id": "up-1"}
    assert images[0]["image"]["caption"][0]["text"]["content"].startswith("[00:03] the finished tray")
    texts = [rt["text"]["content"] for b in created["children"] for rt in b.get(b["type"], {}).get("rich_text", [])]
    assert all(len(t) <= notion_writer.RICH_TEXT_MAX for t in texts)
    # The cover is a second upload of the first screenshot.
    assert fake.find("PATCH", "/pages/page-1")[0]["cover"] == {"type": "file_upload", "file_upload": {"id": "up-2"}}


async def test_a_column_with_the_wrong_type_is_skipped_with_a_warning(tmp_path):
    fake = FakeNotion({"Name": {"type": "title"}, "Category": {"type": "rich_text"}, **_all_but_category()})
    async with fake.client() as http:
        page = await notion_writer.create_reel_page(
            notion_settings(tmp_path), content(), "t", "", (), (), "AI", base_dir=tmp_path, http=http
        )
    props = fake.find("POST", "/pages")[0]["properties"]
    assert "Category" not in props and "Name" in props
    assert any("'Category' is a rich_text" in w for w in page.warnings)
    assert not fake.find("PATCH", f"/data_sources/{DS}")  # nothing missing, nothing added


def _all_but_category() -> dict:
    return {n: {"type": k} for n, k in notion_writer.SCHEMA.items() if n != "Category"}


async def test_long_pages_are_appended_in_batches_of_100(tmp_path):
    fake = FakeNotion()
    many = [Scene(f"00:{i:02d}", f"scene {i}") for i in range(60)]
    long = content(transcript="\n\n".join(f"para {i}" for i in range(150)))
    async with fake.client() as http:
        await notion_writer.create_reel_page(
            notion_settings(tmp_path), long, "t", "", (), many, None, base_dir=tmp_path, http=http
        )
    assert len(fake.find("POST", "/pages")[0]["children"]) == 100
    appended = fake.find("PATCH", "/blocks/page-1/children")
    assert appended and all(len(b["children"]) <= 100 for b in appended)


async def test_unshared_database_explains_itself(tmp_path):
    fake = FakeNotion()
    s = replace(notion_settings(tmp_path), notion_database_id="f" * 32)
    async with fake.client() as http:
        with pytest.raises(notion_writer.NotionError, match="Connections"):
            await notion_writer.create_reel_page(s, content(), "t", "", (), (), None, base_dir=tmp_path, http=http)


def test_rich_text_chunks_on_boundaries():
    text = ("word " * 900).strip()
    pieces = notion_writer._chunks(text)
    assert all(len(p) <= 2000 for p in pieces) and " ".join(pieces) == text


# ── selection, categories, frames ──────────────────────────────────────


def test_nearest_frame_matches_the_cell_label():
    frames = [ocr.Frame(0.0, b"a"), ocr.Frame(3.7, b"b"), ocr.Frame(9.2, b"c")]
    assert ocr.nearest_frame(frames, "00:03").png == b"b"  # label is the truncated second
    assert ocr.nearest_frame(frames, "00:08").png == b"c"  # no exact match: nearest
    assert ocr.nearest_frame(frames, "not a time") is None
    assert ocr.parse_timestamp("1:07") == 67


def test_category_is_mapped_to_the_taxonomy_spelling():
    cats = ("Cooking & Recipes", "Tech & Software", "Other")
    assert match_category("cooking and recipes", cats) == "Cooking & Recipes"
    assert match_category("Knitting", cats) == "Other"
    assert match_category("Knitting", ("A", "B")) is None
    assert match_category("", cats) is None


def test_schema_carries_keep_flags_and_a_category_enum():
    schema = summary_schema(True, ("A", "B"))
    assert schema["properties"]["category"]["enum"] == ["A", "B"]
    assert "category" in schema["required"]
    item = schema["properties"]["scenes"]["items"]
    assert item["properties"]["keep_screenshot"]["type"] == "boolean"


def test_string_booleans_from_cli_backends_are_understood():
    data = {"title": "t", "scenes": [{"time": "00:01", "description": "d", "keep_screenshot": "true", "reason": "r"}]}
    summary = _coerce(data, "fb", ("X",))
    assert summary.scenes[0].keep is True and summary.scenes[0].reason == "r"


def test_only_kept_scenes_are_saved_and_each_frame_once(tmp_path):
    frames = [ocr.Frame(1.2, b"one"), ocr.Frame(5.0, b"five")]
    scenes = [
        Scene("00:01", "a", keep=True),
        Scene("00:01", "same frame again", keep=True),
        Scene("00:05", "b", keep=False),
    ]
    out, warnings = pipeline._save_screenshots_sync(scenes, frames, tmp_path, "note")
    assert [s.image_path for s in out] == ["assets/note/00-01.png", None, None]
    assert (tmp_path / "assets/note/00-01.png").read_bytes() == b"one"
    assert warnings == []


def test_the_numbered_frame_is_saved_not_the_first_one_in_that_second(tmp_path):
    # A fast-cut reel puts several frames inside one second; the scene's time names the second,
    # its frame number names the cell the model actually described. That cell is the screenshot.
    frames = [ocr.Frame(15.0, b"transition"), ocr.Frame(15.6, b"close-up"), ocr.Frame(33.9, b"zoomed-out")]
    scenes = [
        Scene("00:15", "the close-up", keep=True, frame=2),
        Scene("00:15", "the transition too", keep=True, frame=1),
        Scene("00:33", "no number: falls back to the time", keep=True),
        Scene("00:40", "number out of range, time matches nothing close", keep=True, frame=9),
    ]
    out, warnings = pipeline._save_screenshots_sync(scenes, frames, tmp_path, "note")
    assert [s.image_path for s in out] == [
        "assets/note/00-15.png",
        "assets/note/00-15-2.png",  # same second, different frame: its own file, not an overwrite
        "assets/note/00-33.png",
        None,  # the time fallback resolves to 00:33's frame, which is already used
    ]
    assert (tmp_path / "assets/note/00-15.png").read_bytes() == b"close-up"
    assert (tmp_path / "assets/note/00-15-2.png").read_bytes() == b"transition"
    assert warnings == []


def test_frame_numbers_from_cli_backends_are_understood():
    scenes = [
        {"time": "00:01", "frame": n, "description": "d", "keep_screenshot": True, "reason": "r"}
        for n in (3, "4", "#5", "", None, True, "five")
    ]
    summary = _coerce({"title": "t", "scenes": scenes}, "fb")
    assert [s.frame for s in summary.scenes] == [3, 4, 5, None, None, None, None]
    assert ocr.frame_by_number([ocr.Frame(0.0, b"a")], 1).png == b"a"
    assert ocr.frame_by_number([ocr.Frame(0.0, b"a")], 0) is None


def test_note_embeds_only_kept_screenshots_and_shows_the_category():
    scenes = [Scene("00:01", "face"), Scene("00:03", "tray", keep=True).with_image("assets/n/00-03.png")]
    md = build_note(content(), "T", "S", (), NOW, scenes, "Cooking & Recipes", "https://notion.so/x")
    assert md.count("![") == 1 and "(<assets/n/00-03.png>)" in md
    assert 'category: "Cooking & Recipes"' in md and '"cooking-and-recipes"' in md
    assert "**Notion:** [https://notion.so/x]" in md
    assert "cdn.example/t.jpg" not in md  # a screenshot replaces the platform thumbnail


# ── config ─────────────────────────────────────────────────────────────


def test_notion_settings_and_vision_follow_the_env(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "none.env")
    for key in ("SHORTFORM_NOTES_VISION", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NOTION_TOKEN", "secret")
    monkeypatch.setenv("NOTION_DATABASE_ID", f"https://www.notion.so/me/Reels-{DB}?v={'9' * 32}")
    monkeypatch.setenv("SHORTFORM_NOTES_CATEGORIES", "Food, Tech ,food,")
    s = config.load_settings(transcribe_provider="none", summary_provider="none")
    assert s.can_write_notion and s.notion_database_id == DB  # parsed out of the share URL, not the view id
    assert s.vision  # auto-on with Notion
    assert s.categories == ("Food", "Tech")
    off = config.load_settings(transcribe_provider="none", summary_provider="none", notion=False)
    assert not off.can_write_notion and not off.vision
    assert config.load_settings(transcribe_provider="none", categories="A,B").categories == ("A", "B")
    with pytest.raises(ValueError):
        config.load_settings(transcribe_provider="none", whisper_device="tpu")


def test_facebook_links_are_detected():
    for link, clean in [
        ("https://www.facebook.com/reel/123456?mibextid=x", "https://www.facebook.com/reel/123456"),
        ("https://www.facebook.com/share/r/1AbC/", "https://www.facebook.com/share/r/1AbC/"),
        ("https://fb.watch/xyz/", "https://fb.watch/xyz/"),
        ("https://m.facebook.com/watch/?v=42&ref=s", "https://m.facebook.com/watch/?v=42"),
    ]:
        found = urls.detect_reel_url(f"see {link}")
        assert found and urls.strip_tracking(found) == clean and urls.platform_for(found) == "facebook"
    assert urls.detect_reel_url("https://www.facebook.com/groups/x/") is None


# ── pipeline wiring ────────────────────────────────────────────────────


async def test_import_writes_screenshots_notion_and_library(tmp_path):
    s = notion_settings(tmp_path)
    frames = [ocr.Frame(3.0, b"png-bytes")]
    summary = Summary(
        "Title",
        "Sum",
        ("t",),
        (Scene("00:03", "tray", keep=True, reason="result", frame=1), Scene("00:04", "face")),
        "AI",
        frames=tuple(frames),
    )
    fake_page = notion_writer.NotionPage("p", "https://notion.so/p", ("a screenshot warning",))
    with (
        patch.object(pipeline, "gather_content", AsyncMock(return_value=(content(), frames))),
        patch.object(pipeline, "summarize", AsyncMock(return_value=summary)),
        patch.object(notion_writer, "create_reel_page", AsyncMock(return_value=fake_page)) as create,
    ):
        result = await pipeline.import_reel("https://www.youtube.com/shorts/abc", s, now=NOW)
    assert result.category == "AI" and result.notion_url == "https://notion.so/p" and result.screenshot_count == 1
    passed_scenes = create.call_args.args[5]
    assert passed_scenes[0].image_path and not passed_scenes[1].image_path
    assert (s.output_dir / passed_scenes[0].image_path).read_bytes() == b"png-bytes"
    assert "a screenshot warning" in result.warnings
    note = result.path.read_text(encoding="utf-8")
    assert "https://notion.so/p" in note and "a screenshot warning" in note
    [entry] = library.load(s.output_dir)
    assert entry.category == "AI" and entry.notion_url == "https://notion.so/p" and entry.thumbnail.endswith(".png")


async def test_a_notion_failure_still_writes_the_note_and_the_library(tmp_path):
    s = notion_settings(tmp_path)
    with (
        patch.object(pipeline, "gather_content", AsyncMock(return_value=(content(), []))),
        patch.object(pipeline, "summarize", AsyncMock(return_value=Summary("Title", "", (), (), "AI"))),
        patch.object(
            notion_writer, "create_reel_page", AsyncMock(side_effect=notion_writer.NotionError("token rejected"))
        ),
    ):
        result = await pipeline.import_reel("https://www.youtube.com/shorts/abc", s, now=NOW)
    assert result.path.exists() and result.notion_url is None
    assert any("Notion page not created: token rejected" in w for w in result.warnings)
    assert library.load(s.output_dir)[0].notion_url is None


async def test_without_notion_nothing_is_sent(tmp_path):
    with (
        patch.object(pipeline, "gather_content", AsyncMock(return_value=(content(), []))),
        patch.object(pipeline, "summarize", AsyncMock(return_value=Summary("Title", "", ()))),
        patch.object(notion_writer, "create_reel_page", AsyncMock()) as create,
    ):
        result = await pipeline.import_reel("https://www.youtube.com/shorts/abc", settings(tmp_path), now=NOW)
    create.assert_not_called()
    assert library.load(settings(tmp_path).output_dir)[0].note_path == result.path.name


def test_library_replaces_an_entry_for_the_same_note_and_survives_corruption(tmp_path):
    entry = library.LibraryEntry("2026-01-01T00:00:00", "a", "u", "youtube", None, "a.md")
    library.record(tmp_path, entry)
    library.record(tmp_path, replace(entry, title="b"))
    assert [e.title for e in library.load(tmp_path)] == ["b"]
    library.library_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert library.load(tmp_path) == []


# ── de-duplication ─────────────────────────────────────────────────────


def test_every_spelling_of_a_link_is_one_video():
    same = [
        ("https://www.instagram.com/reel/Ddh7h8rB1gh/?stkn=x", "https://instagram.com/p/Ddh7h8rB1gh"),
        ("https://www.youtube.com/shorts/iLe2PLFdIgY?si=q", "https://youtu.be/iLe2PLFdIgY"),
        ("https://www.facebook.com/watch/?v=42&ref=s", "https://m.facebook.com/watch?v=42"),
        ("https://www.facebook.com/reel/42/", "https://www.facebook.com/somepage/videos/42"),
        ("https://www.tiktok.com/@a/video/7300?lang=en", "https://m.tiktok.com/@a/video/7300"),
        ("https://fb.watch/xyz/", "https://fb.watch/xyz"),
    ]
    for a, b in same:
        assert urls.reel_key(a) == urls.reel_key(b), (a, b)
    assert urls.reel_key("https://www.instagram.com/reel/A1/") != urls.reel_key("https://www.instagram.com/reel/B2/")
    assert urls.reel_key("https://www.instagram.com/p/Ddh7h8rB1gh/") == "instagram:Ddh7h8rB1gh"
    # The search term is a substring of every spelling, so Notion's "contains" filter finds them all.
    for a, b in same:
        term = urls.reel_search_term(urls.reel_key(a))
        assert term in a and term in b


def test_library_finds_a_video_under_another_spelling_only_while_its_note_exists(tmp_path):
    (tmp_path / "a.md").write_text("note", encoding="utf-8")
    url = "https://www.instagram.com/reel/X1/"
    entry = library.LibraryEntry("2026-01-01T00:00:00", "a", url, "instagram", None, "a.md")
    library.record(tmp_path, entry)
    assert library.find(tmp_path, "https://instagram.com/p/X1?igsh=q") == entry
    assert library.find(tmp_path, "https://www.instagram.com/reel/Y2/") is None
    (tmp_path / "a.md").unlink()
    assert library.find(tmp_path, url) is None  # deleting the note asks for a fresh import
    # A re-import under another spelling and another note replaces the entry rather than adding one.
    library.record(tmp_path, replace(entry, url="https://instagram.com/p/X1", note_path="b.md", title="b"))
    assert [e.title for e in library.load(tmp_path)] == ["b"]


async def test_a_link_already_in_the_library_runs_nothing(tmp_path):
    s = settings(tmp_path)
    with (
        patch.object(pipeline, "gather_content", AsyncMock(return_value=(content(), []))),
        patch.object(pipeline, "summarize", AsyncMock(return_value=Summary("Title", "Sum", ()))),
    ):
        first = await pipeline.import_reel("https://www.youtube.com/shorts/abc", s, now=NOW)
    gather = AsyncMock()
    with patch.object(pipeline, "gather_content", gather):
        again = await pipeline.import_reel("https://youtu.be/abc?si=x", s, now=NOW)
    gather.assert_not_called()
    assert again.duplicate and again.path == first.path and again.title == "Title"
    assert again.to_dict()["duplicate"] is True and "duplicate" not in first.to_dict()


async def test_force_replaces_the_note_its_screenshots_and_the_library_entry(tmp_path):
    s = settings(tmp_path)
    frames = [ocr.Frame(3.0, b"old-png")]
    kept = (Scene("00:03", "tray", keep=True, frame=1),)
    old_summary = Summary("Old", "", (), kept, frames=tuple(frames))
    with (
        patch.object(pipeline, "gather_content", AsyncMock(return_value=(content(), frames))),
        patch.object(pipeline, "summarize", AsyncMock(return_value=old_summary)),
    ):
        first = await pipeline.import_reel("https://www.youtube.com/shorts/abc", s, now=NOW)
    old_assets = s.output_dir / "assets" / first.path.stem
    assert (old_assets / "00-03.png").exists()
    with (
        patch.object(pipeline, "gather_content", AsyncMock(return_value=(content(), []))),
        patch.object(pipeline, "summarize", AsyncMock(return_value=Summary("New title", "", ()))),
    ):
        second = await pipeline.import_reel("https://www.youtube.com/shorts/abc", s, now=NOW, force=True)
    assert not second.duplicate and second.path != first.path
    assert second.path.exists() and not first.path.exists() and not old_assets.exists()
    assert [e.note_path for e in library.load(s.output_dir)] == [second.path.name]


async def test_a_note_imported_before_notion_was_set_up_is_reimported_to_reach_notion(tmp_path):
    with (
        patch.object(pipeline, "gather_content", AsyncMock(return_value=(content(), []))),
        patch.object(pipeline, "summarize", AsyncMock(return_value=Summary("Title", "", ()))),
    ):
        first = await pipeline.import_reel("https://www.youtube.com/shorts/abc", settings(tmp_path), now=NOW)
    fake_page = notion_writer.NotionPage("p", "https://notion.so/p")
    s = notion_settings(tmp_path)
    with (
        patch.object(pipeline, "gather_content", AsyncMock(return_value=(content(), []))),
        patch.object(pipeline, "summarize", AsyncMock(return_value=Summary("Title", "", ()))),
        patch.object(notion_writer, "create_reel_page", AsyncMock(return_value=fake_page)) as create,
    ):
        second = await pipeline.import_reel("https://www.youtube.com/shorts/abc", s, now=NOW)
    assert create.call_args.kwargs["replace_existing"] is False
    assert second.path == first.path and second.notion_url == "https://notion.so/p"  # same title: overwritten
    [entry] = library.load(s.output_dir)
    assert entry.notion_url == "https://notion.so/p"


def _existing_page(url: str, page_id: str = "old-page") -> dict:
    return {
        "object": "page",
        "id": page_id,
        "url": f"https://www.notion.so/{page_id}",
        "properties": {notion_writer.PROP_SOURCE: {"type": "url", "url": url}},
    }


async def test_notion_links_an_existing_page_for_the_same_video_instead_of_creating_one(tmp_path):
    # "abc" also matches another video's link as a substring; reel_key tells them apart.
    fake = FakeNotion(
        existing=[_existing_page("https://youtu.be/abcdef", "other-video"), _existing_page("https://youtu.be/abc")]
    )
    async with fake.client() as http:
        page = await notion_writer.create_reel_page(
            notion_settings(tmp_path), content(), "T", "S", (), shots(tmp_path), None, base_dir=tmp_path, http=http
        )
    assert page.existing and page.id == "old-page" and page.url == "https://www.notion.so/old-page"
    query = fake.find("POST", f"/data_sources/{DS}/query")[0]
    assert query["filter"] == {"property": "Source URL", "url": {"contains": "abc"}}
    assert not fake.find("POST", "/pages") and fake.uploads == 0  # nothing written, nothing uploaded


async def test_notion_replace_trashes_the_earlier_page_then_creates_a_new_one(tmp_path):
    fake = FakeNotion(existing=[_existing_page("https://www.youtube.com/shorts/abc")])
    async with fake.client() as http:
        page = await notion_writer.create_reel_page(
            notion_settings(tmp_path),
            content(),
            "T",
            "S",
            (),
            shots(tmp_path),
            None,
            base_dir=tmp_path,
            http=http,
            replace_existing=True,
        )
    assert not page.existing and page.id == "page-1"
    assert fake.find("PATCH", "/pages/old-page") == [{"in_trash": True}]
    methods = [(m, p) for m, p, _ in fake.calls]
    assert methods.index(("PATCH", "/pages/old-page")) < methods.index(("POST", "/pages"))
