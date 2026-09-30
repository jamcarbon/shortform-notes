"""Vision: contact-sheet tiling, the four backend payloads, the frame cap, and pipeline wiring.

No network and no subprocesses: ``openai`` / ``anthropic`` are replaced by stub modules whose
clients record their payload, and the CLI backends are patched at ``summarize._run_cli``.
"""

import json
import re
import sys
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from shortform_notes import config, media, ocr, summarize
from shortform_notes.media import DownloadedMedia
from shortform_notes.note import ReelContent, Scene, build_note
from shortform_notes.pipeline import ReelImportResult, import_reel
from shortform_notes.summarize import MAX_VISION_FRAMES, select_frames, vision_estimate

NOW = datetime(2026, 8, 23, 20, 30, tzinfo=timezone.utc)

cv2 = pytest.importorskip("cv2")  # vision samples and tiles frames with OpenCV
np = pytest.importorskip("numpy")

REPLY = {"title": "T", "summary": "S", "takeaways": ["a"]}
REPLY_JSON = json.dumps(REPLY)


def settings(tmp_path, **over) -> config.Settings:
    base = dict(
        output_dir=tmp_path / "reels",
        openai_api_key="sk-test",
        anthropic_api_key="sk-ant-test",
        summary_provider="openai",
        transcribe_provider="none",
        openai_transcribe_model="x",
        openai_summary_model="gpt-5-mini",
        anthropic_summary_model="claude-sonnet-5",
        claude_code_model=None,
        codex_model=None,
        whisper_model="base",
        audience="the reader",
        ocr=False,
        ocr_provider="local",
        ocr_fps=1.0,
        ocr_openai_model="gpt-5-mini",
        ocr_anthropic_model="claude-sonnet-5",
        vision=True,
        vision_agentic=False,
        fps_explicit=False,
    )
    return config.Settings(**{**base, **over})


def frames(count: int, width: int = 180, height: int = 320) -> list[ocr.Frame]:
    """Real PNGs, portrait like a reel, each a different shade so tiling is visible."""
    out = []
    for i in range(count):
        img = np.full((height, width, 3), (i * 7) % 256, dtype=np.uint8)
        out.append(ocr.Frame(seconds=float(i), png=cv2.imencode(".png", img)[1].tobytes()))
    return out


def video(**over) -> DownloadedMedia:
    base = dict(
        audio_path=None,
        video_path="/tmp/fake.mp4",
        caption="cap",
        creator_handle="c",
        creator_name="C",
        title=None,
        timestamp=None,
        duration=12.0,
        thumbnail=None,
        webpage_url="u",
    )
    return DownloadedMedia(**{**base, **over})


@contextmanager
def stub_openai(recorder: dict):
    """Replace the ``openai`` module; the client records the create() kwargs."""

    class Completions:
        async def create(self, **kwargs):
            recorder.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=REPLY_JSON))])

    module = types.ModuleType("openai")
    module.AsyncOpenAI = lambda **_: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    with patch.dict(sys.modules, {"openai": module}):
        yield


@contextmanager
def stub_anthropic(recorder: dict):
    class Messages:
        async def create(self, **kwargs):
            recorder.update(kwargs)
            block = SimpleNamespace(type="text", text=REPLY_JSON)
            return SimpleNamespace(stop_reason="end_turn", content=[block])

    module = types.ModuleType("anthropic")
    module.AsyncAnthropic = lambda **_: SimpleNamespace(messages=Messages())
    with patch.dict(sys.modules, {"anthropic": module}):
        yield


def stream_json_output(result: str = REPLY_JSON, is_error: bool = False) -> str:
    """What `claude -p --output-format stream-json` writes: events, then a final result event."""
    return "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": result}]}}),
            json.dumps({"type": "result", "subtype": "success", "is_error": is_error, "result": result}),
        ]
    )


# ── tiling ─────────────────────────────────────────────────────────────


async def test_tile_frames_builds_chronological_sheets():
    sheets = await ocr.tile_frames(frames(20))
    assert len(sheets) == 2  # 16 to a sheet, then the remainder
    assert sheets[0].seconds == tuple(float(i) for i in range(16))
    assert sheets[1].seconds == (16.0, 17.0, 18.0, 19.0)
    first = cv2.imdecode(np.frombuffer(sheets[0].png, dtype="uint8"), cv2.IMREAD_COLOR)
    assert first.shape[:2] == (320 * 4, 180 * 4)  # 4x4 of cells already at the 320 max side
    partial = cv2.imdecode(np.frombuffer(sheets[1].png, dtype="uint8"), cv2.IMREAD_COLOR)
    assert partial.shape[:2] == (320, 180 * 4)  # one padded row, not a ragged image
    assert sheets[1].numbers == (17, 18, 19, 20)  # numbered across sheets, as the model names them
    assert sheets[1].describe() == "4 frames, #17 to #20, 00:16 to 00:19"
    assert ocr.cell_label(17, 16.4) == "#17 00:16"


async def test_tile_frames_downscales_and_labels_cells():
    sheets = await ocr.tile_frames(frames(1, width=720, height=1280))
    sheet = cv2.imdecode(np.frombuffer(sheets[0].png, dtype="uint8"), cv2.IMREAD_COLOR)
    # A 720x1280 frame is capped to a 288x512 cell, then padded out to a full row of four.
    assert sheet.shape[:2] == (ocr.GRID_CELL_MAX_SIDE, 288 * 4)
    corner = sheet[0:20, 0:60]
    assert corner.min() == 0 and corner.max() == 255  # the timestamp is burnt in, white on black
    assert await ocr.tile_frames([]) == []


# ── API backends ───────────────────────────────────────────────────────


async def test_sheets_reach_the_openai_payload(tmp_path):
    sent = {}
    with stub_openai(sent):
        result = await summarize.summarize("cap", "spoken", settings(tmp_path), frames=frames(20))
    assert result.title == "T"
    content = sent["messages"][1]["content"]
    images = [part for part in content if part["type"] == "image_url"]
    assert len(images) == 2  # two sheets, not twenty frames
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert images[0]["image_url"]["detail"] == "high"  # low resizes the sheet to a thumbnail
    assert content[0]["text"].startswith("Caption:\ncap")
    labels = [part.get("text") or "" for part in content]
    assert any(label.startswith("contact sheet 1 of 2: 16 frames, #1 to #16, 00:00 to 00:15") for label in labels)
    assert "contact sheets of frames sampled" in sent["messages"][0]["content"]


async def test_openai_request_uses_gpt5_compatible_parameters(tmp_path):
    """gpt-5-mini 400s on `max_tokens` and on any temperature but the default."""
    sent = {}
    with stub_openai(sent):
        await summarize.summarize("cap", "spoken", settings(tmp_path), frames=frames(4))
    assert "max_tokens" not in sent and "temperature" not in sent
    assert sent["max_completion_tokens"] >= 2000  # reasoning tokens come out of this budget too


async def test_sheets_reach_the_anthropic_payload(tmp_path):
    sent = {}
    with stub_anthropic(sent):
        result = await summarize.summarize(
            "cap", None, settings(tmp_path, summary_provider="anthropic"), frames=frames(3)
        )
    assert result.title == "T"
    content = sent["messages"][0]["content"]
    images = [part for part in content if part["type"] == "image"]
    assert len(images) == 1
    assert images[0]["source"]["media_type"] == "image/png"
    assert content[-1]["text"].startswith("Caption:\ncap")  # the text prompt comes after the sheets
    assert "contact sheets of frames sampled" in sent["system"]


async def test_no_frames_keeps_the_plain_text_payload(tmp_path):
    sent = {}
    with stub_openai(sent):
        await summarize.summarize("cap", None, settings(tmp_path, vision=False), frames=frames(4))
    assert sent["messages"][1]["content"].startswith("Caption:\ncap")
    assert "contact sheets" not in sent["messages"][0]["content"]


# ── claude-code backend ────────────────────────────────────────────────


async def test_claude_code_uses_stream_json_with_image_blocks(tmp_path):
    with patch.object(summarize, "_run_cli", AsyncMock(return_value=stream_json_output())) as run:
        result = await summarize.summarize(
            "cap", None, settings(tmp_path, summary_provider="claude-code"), frames=frames(20)
        )
    assert result.title == "T"
    argv, stdin = run.await_args.args
    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv  # the CLI rejects stream-json output without it
    assert argv[argv.index("--tools") + 1] == ""  # hardening is unchanged
    assert "--no-session-persistence" in argv and "--disable-slash-commands" in argv

    message = json.loads(stdin.strip())
    assert message["type"] == "user" and message["message"]["role"] == "user"
    content = message["message"]["content"]
    assert content[0]["text"].startswith("You are turning a short social-media video")
    images = [part for part in content if part["type"] == "image"]
    assert len(images) == 2
    assert images[0]["source"]["type"] == "base64" and images[0]["source"]["media_type"] == "image/png"


async def test_claude_code_without_vision_keeps_the_plain_stdin_path(tmp_path):
    envelope = json.dumps({"is_error": False, "result": REPLY_JSON})
    with patch.object(summarize, "_run_cli", AsyncMock(return_value=envelope)) as run:
        result = await summarize.summarize(
            "cap", None, settings(tmp_path, summary_provider="claude-code", vision=False)
        )
    assert result.title == "T"
    argv, stdin = run.await_args.args
    assert argv[argv.index("--output-format") + 1] == "json"
    assert "--input-format" not in argv and "--verbose" not in argv
    assert stdin.startswith("You are turning a short social-media video")


async def test_claude_code_stream_json_error_degrades(tmp_path):
    raw = stream_json_output(result="Not logged in", is_error=True)
    with patch.object(summarize, "_run_cli", AsyncMock(return_value=raw)):
        result = await summarize.summarize(
            "First line of caption", None, settings(tmp_path, summary_provider="claude-code"), frames=frames(2)
        )
    assert result.title == "First line of caption" and result.summary == ""


def test_stream_json_result_needs_a_result_event():
    assert summarize._stream_json_result(stream_json_output())["result"] == REPLY_JSON
    with pytest.raises(summarize.SummaryError, match="no result event"):
        summarize._stream_json_result('{"type": "assistant"}\nnot json at all')


# ── codex backend ──────────────────────────────────────────────────────


async def test_codex_attaches_sheets_as_image_files(tmp_path):
    seen = {}

    async def fake_run(argv, prompt):
        images = [argv[i + 1] for i, a in enumerate(argv) if a == "-i"]
        seen["images"] = images
        seen["exists"] = [Path(p).exists() for p in images]
        seen["png"] = [Path(p).read_bytes()[:8] for p in images]
        seen["prompt"] = prompt
        Path(argv[argv.index("--output-last-message") + 1]).write_text(REPLY_JSON)
        return "ignored stdout"

    with patch.object(summarize, "_run_cli", fake_run):
        result = await summarize.summarize("cap", None, settings(tmp_path, summary_provider="codex"), frames=frames(20))
    assert result.title == "T"
    assert len(seen["images"]) == 2  # one -i per contact sheet
    assert all(seen["exists"])
    assert all(header == b"\x89PNG\r\n\x1a\n" for header in seen["png"])
    assert "contact sheets, in order" in seen["prompt"]
    assert not any(Path(p).exists() for p in seen["images"])  # temp files cleaned up


async def test_codex_without_vision_passes_no_images(tmp_path):
    async def fake_run(argv, prompt):
        assert "-i" not in argv
        Path(argv[argv.index("--output-last-message") + 1]).write_text(REPLY_JSON)
        return ""

    with patch.object(summarize, "_run_cli", fake_run):
        result = await summarize.summarize("cap", None, settings(tmp_path, summary_provider="codex", vision=False))
    assert result.title == "T"


# ── agentic vision ─────────────────────────────────────────────────────


def test_frame_filenames_are_the_cell_labels():
    assert summarize.frame_filename(1, 0) == "01_00-00.png"
    assert summarize.frame_filename(7, 3.2) == "07_00-03.png"
    assert summarize.frame_filename(8, 3.7) == "08_00-03.png"  # same second, its own file
    assert summarize.frame_filename(48, 75) == "48_01-15.png"


def test_oneshot_argv_still_disables_every_tool(tmp_path):
    argv = summarize.claude_code_argv(settings(tmp_path, summary_provider="claude-code"), images=True)
    assert argv[argv.index("--tools") + 1] == ""
    assert "--add-dir" not in argv and "--allowed-tools" not in argv


def test_agentic_argv_allows_only_read_in_only_the_frames_dir(tmp_path):
    argv = summarize.claude_code_argv(
        settings(tmp_path, summary_provider="claude-code"), images=True, frames_dir="/tmp/f/frames"
    )
    assert argv[argv.index("--tools") + 1] == "Read"  # the available set, not just the permitted one
    assert argv[argv.index("--allowed-tools") + 1] == "Read"  # so -p never stalls on a prompt
    assert argv[argv.index("--add-dir") + 1] == "/tmp/f/frames"
    assert "--no-session-persistence" in argv and "--disable-slash-commands" in argv
    joined = " ".join(argv)
    for forbidden in ("Bash", "Write", "Edit", "WebFetch", "WebSearch", "--dangerously-skip-permissions"):
        assert forbidden not in joined


async def test_agentic_claude_code_writes_frames_and_names_them_in_the_prompt(tmp_path):
    seen = {}

    async def fake_run(argv, stdin):
        directory = Path(argv[argv.index("--add-dir") + 1])
        seen["dir"] = directory
        seen["files"] = sorted(p.name for p in directory.iterdir())
        seen["png"] = (directory / "01_00-00.png").read_bytes()[:8]
        seen["prompt"] = json.loads(stdin)["message"]["content"][0]["text"]
        return stream_json_output()

    with patch.object(summarize, "_run_cli", fake_run):
        result = await summarize.summarize(
            "cap", None, settings(tmp_path, summary_provider="claude-code", vision_agentic=True), frames=frames(3)
        )
    assert result.title == "T"
    assert seen["files"] == ["01_00-00.png", "02_00-01.png", "03_00-02.png"]
    assert seen["png"] == b"\x89PNG\r\n\x1a\n"  # the real frame, not a placeholder
    assert str(seen["dir"]) in seen["prompt"] and "02_00-01.png" in seen["prompt"]
    assert "Open any frame you need" in seen["prompt"]
    assert not seen["dir"].exists()  # torn down with the call, like the downloaded media


async def test_oneshot_claude_code_writes_no_frames_dir(tmp_path):
    with patch.object(summarize, "_run_cli", AsyncMock(return_value=stream_json_output())) as run:
        await summarize.summarize("cap", None, settings(tmp_path, summary_provider="claude-code"), frames=frames(3))
    argv, stdin = run.await_args.args
    assert "--add-dir" not in argv
    assert "full-resolution originals" not in json.loads(stdin)["message"]["content"][0]["text"]


async def test_agentic_codex_puts_the_frames_dir_in_the_prompt(tmp_path):
    seen = {}

    async def fake_run(argv, prompt):
        seen["prompt"] = prompt
        directory = Path(re.search(r"originals are in (\S+),", prompt).group(1))
        seen["dir"] = directory
        seen["files"] = sorted(p.name for p in directory.iterdir())
        assert "--sandbox" in argv and argv[argv.index("--sandbox") + 1] == "read-only"
        Path(argv[argv.index("--output-last-message") + 1]).write_text(REPLY_JSON)
        return ""

    with patch.object(summarize, "_run_cli", fake_run):
        result = await summarize.summarize(
            "cap", None, settings(tmp_path, summary_provider="codex", vision_agentic=True), frames=frames(3)
        )
    assert result.title == "T"
    assert seen["files"] == ["01_00-00.png", "02_00-01.png", "03_00-02.png"]
    assert not seen["dir"].exists()


def test_agentic_needs_an_agent_backend(tmp_path):
    for provider in ("claude-code", "codex"):
        assert settings(tmp_path, summary_provider=provider, vision_agentic=True).vision_is_agentic is True
    for provider in ("openai", "anthropic", "none"):
        assert settings(tmp_path, summary_provider=provider, vision_agentic=True).vision_is_agentic is False
    # asking for agentic never turns vision on by itself
    assert settings(tmp_path, vision=False, vision_agentic=True).vision_is_agentic is False


async def test_agentic_on_an_api_backend_runs_oneshot_and_warns(tmp_path):
    """The frames still reach the model as sheets; only the second look is unavailable."""
    sent = {}
    with stub_openai(sent):
        result = await summarize.summarize(
            "cap", None, settings(tmp_path, summary_provider="openai", vision_agentic=True), frames=frames(3)
        )
    assert result.title == "T"
    assert any(part["type"] == "image_url" for part in sent["messages"][1]["content"])


# ── the cap ────────────────────────────────────────────────────────────


def test_select_frames_caps_and_spreads():
    assert select_frames(frames(5)) == frames(5)
    picked = select_frames(frames(100))
    assert len(picked) == MAX_VISION_FRAMES
    assert picked[0].seconds == 0 and picked[-1].seconds == 99
    assert [f.seconds for f in picked] == sorted({f.seconds for f in picked})
    assert len(select_frames(frames(9), limit=3)) == 3


async def test_summarize_never_sends_more_than_the_cap(tmp_path):
    sent = {}
    with stub_openai(sent):
        result = await summarize.summarize("cap", None, settings(tmp_path), frames=frames(200))
    images = [part for part in sent["messages"][1]["content"] if part["type"] == "image_url"]
    assert len(images) == MAX_VISION_FRAMES // ocr.FRAMES_PER_GRID == 3
    # The summary carries the frames it showed, so a scene's frame number resolves against those, not all 200.
    assert list(result.frames) == select_frames(frames(200))


def test_vision_estimate_prices_sheets_not_frames(tmp_path):
    est = vision_estimate(60, settings(tmp_path))
    assert est.frames == 48 and est.sheets == 3
    sheet_tokens = ocr.openai_image_tokens(*ocr.sheet_dims())
    assert est.usd == pytest.approx(3 * sheet_tokens * 0.25 / 1e6, abs=1e-4)  # 3 images, not 48
    short = vision_estimate(10, settings(tmp_path))
    assert short.frames == 10 and short.sheets == 1
    assert "1 contact sheet," in short.describe()
    assert vision_estimate(60, settings(tmp_path, summary_provider="anthropic")).usd > est.usd


def test_vision_estimate_is_free_on_subscription_backends(tmp_path):
    for provider in ("claude-code", "codex"):
        est = vision_estimate(60, settings(tmp_path, summary_provider=provider))
        assert est.sheets == 3 and est.usd == 0.0
        assert "included in your subscription" in est.describe()


# ── video breakdown ────────────────────────────────────────────────────


def test_scenes_are_only_asked_for_when_the_model_can_see():
    assert "scenes" not in summarize.summary_schema()["properties"]
    with_frames = summarize.summary_schema(with_frames=True)
    assert with_frames["required"] == ["title", "summary", "takeaways", "scenes"]
    cell = with_frames["properties"]["scenes"]["items"]
    assert cell["required"] == ["time", "frame", "description", "keep_screenshot", "reason"]
    assert "scenes" not in summarize.build_prompt("me")
    assert "scenes" in summarize.build_prompt("me", with_frames=True)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"time": "00:03", "description": "cracks an egg"}, ("00:03", "cracks an egg")),
        ({"time": "[00:03]", "description": "cracks an egg"}, ("00:03", "cracks an egg")),
        ({"time": 3, "description": "cracks an egg"}, ("00:03", "cracks an egg")),  # answered in seconds
        ({"time": 75.4, "description": "cracks an egg"}, ("01:15", "cracks an egg")),
        ({"description": "cracks an egg"}, ("", "cracks an egg")),  # no time: still usable
        ("cracks an egg", ("", "cracks an egg")),  # a backend that ignored the schema
    ],
)
def test_scene_times_are_normalized(raw, expected):
    scenes = summarize._scenes([raw])
    assert (scenes[0].time, scenes[0].description) == expected


def test_unusable_scenes_are_dropped_not_raised():
    assert summarize._scenes([{"time": "00:01", "description": ""}, None, 7, {}]) == ()
    assert summarize._scenes("not a list") == ()
    assert summarize._scenes(None) == ()


async def test_scenes_survive_the_round_trip_from_a_backend(tmp_path):
    reply = {**REPLY, "scenes": [{"time": "00:00", "description": "a hand cracks an egg"}]}
    with patch.object(summarize, "_run_cli", AsyncMock(return_value=stream_json_output(json.dumps(reply)))):
        result = await summarize.summarize(
            "cap", None, settings(tmp_path, summary_provider="claude-code"), frames=frames(3)
        )
    assert [(s.time, s.description) for s in result.scenes] == [("00:00", "a hand cracks an egg")]


def test_note_renders_a_video_breakdown_section(tmp_path):
    content = ReelContent(
        url="u",
        platform="tiktok",
        caption="cap",
        transcript=None,
        screen_text=None,
        title=None,
        creator_handle="c",
        creator_name=None,
        posted=None,
        duration=8.0,
        thumbnail=None,
        sources=("caption", "video"),
        warnings=(),
    )
    scenes = (Scene("00:00", "a hand cracks an egg"), Scene("", "the cake comes out"))
    note = build_note(content, "T", "S", ("a",), NOW, scenes)
    assert "## Video breakdown" in note
    assert "- [00:00] a hand cracks an egg" in note
    assert "- the cake comes out" in note  # no timestamp, no empty brackets
    assert note.index("## Video breakdown") < note.index("## Caption")
    assert "## Video breakdown" not in build_note(content, "T", "S", ("a",), NOW)


def test_json_output_carries_scenes_only_when_there_are_some(tmp_path):
    plain = ReelImportResult(tmp_path / "n.md", "T", "S", ("a",), ("caption",), ())
    assert "scenes" not in plain.to_dict()  # a run without vision looks exactly as it did
    seen = ReelImportResult(tmp_path / "n.md", "T", "S", ("a",), ("caption", "video"), (), (Scene("00:03", "d"),))
    assert seen.to_dict()["scenes"] == [{"time": "00:03", "description": "d", "keep": False, "reason": ""}]


async def test_the_breakdown_reaches_the_saved_note(tmp_path):
    reply = {**REPLY, "scenes": [{"time": "00:02", "description": "zest goes in"}]}
    with (
        patch.object(media, "download_media", AsyncMock(return_value=video())),
        patch.object(ocr, "sample_frames", AsyncMock(return_value=frames(3))),
        patch("shortform_notes.summarize._summarize_openai", AsyncMock(return_value=reply)),
    ):
        result = await import_reel("https://www.tiktok.com/@c/video/1", settings(tmp_path))
    assert "## Video breakdown\n\n- [00:02] zest goes in" in result.path.read_text()
    assert result.to_dict()["scenes"] == [{"time": "00:02", "description": "zest goes in", "keep": False, "reason": ""}]


def test_cli_points_at_the_breakdown_without_reprinting_it(tmp_path, capsys):
    from shortform_notes.cli import _print_result

    scenes = (Scene("00:03", "zest goes in"), Scene("00:05", "into the oven"))
    _print_result(ReelImportResult(tmp_path / "n.md", "T", "S", ("a",), ("caption", "video"), (), scenes), False)
    out = capsys.readouterr().out
    assert "(2 scenes under 'Video breakdown' in the note, 0 kept as screenshots)" in out
    assert "zest goes in" not in out  # the note holds the detail, the terminal stays short


async def test_a_run_without_vision_never_asks_for_scenes(tmp_path):
    sent = {}
    with stub_openai(sent):
        await summarize.summarize("cap", None, settings(tmp_path, vision=False), frames=frames(4))
    schema = sent["response_format"]["json_schema"]["schema"]
    assert "scenes" not in schema["properties"]
    with stub_openai(sent):
        await summarize.summarize("cap", None, settings(tmp_path), frames=frames(4))
    assert "scenes" in sent["response_format"]["json_schema"]["schema"]["properties"]


# ── pipeline wiring ────────────────────────────────────────────────────


async def test_pipeline_sends_frames_and_records_the_video_source(tmp_path):
    with (
        patch.object(media, "download_media", AsyncMock(return_value=video())) as download,
        patch.object(ocr, "sample_frames", AsyncMock(return_value=frames(3))),
        patch("shortform_notes.summarize._summarize_openai", AsyncMock(return_value=REPLY)) as backend,
    ):
        result = await import_reel("https://www.tiktok.com/@c/video/1", settings(tmp_path))
    assert download.await_args.kwargs["video"] is True
    assert result.sources == ("caption", "video")
    assert len(backend.await_args.args[4]) == 1  # three frames arrive as one sheet
    assert "sources: [caption, video]" in result.path.read_text()


async def test_vision_reuses_the_frames_ocr_would_have_sampled(tmp_path):
    with (
        patch.object(media, "download_media", AsyncMock(return_value=video())),
        patch.object(ocr, "sample_frames", AsyncMock(return_value=frames(3))) as extract,
        patch.object(ocr, "read_screen_text", AsyncMock(return_value=("[00:01] 2 cups flour", 3))) as read,
        patch("shortform_notes.summarize._summarize_openai", AsyncMock(return_value=REPLY)),
    ):
        result = await import_reel("https://www.tiktok.com/@c/video/1", settings(tmp_path, ocr=True))
    assert extract.await_count == 1  # sampled once, not once per consumer
    assert read.await_args.args[2] == frames(3)
    assert result.sources == ("caption", "screen_text", "video")


async def test_vision_on_every_backend_downloads_the_video(tmp_path):
    for provider in ("openai", "anthropic", "claude-code", "codex"):
        with (
            patch.object(media, "download_media", AsyncMock(return_value=video())) as download,
            patch.object(ocr, "sample_frames", AsyncMock(return_value=frames(2))),
            patch(f"shortform_notes.summarize._summarize_{provider.replace('-', '_')}", AsyncMock(return_value=REPLY)),
        ):
            result = await import_reel(
                "https://www.tiktok.com/@c/video/1", settings(tmp_path, summary_provider=provider)
            )
        assert download.await_args.kwargs["video"] is True, provider
        assert result.sources == ("caption", "video"), provider


async def test_no_summary_backend_warns_and_skips_the_download(tmp_path):
    with patch.object(media, "download_media", AsyncMock(return_value=video())) as download:
        result = await import_reel("https://www.tiktok.com/@c/video/1", settings(tmp_path, summary_provider="none"))
    assert result.sources == ("caption",)
    assert any("Vision skipped: no summary backend" in w for w in result.warnings)
    assert download.await_args.kwargs["video"] is False  # no point paying for the whole mp4


async def test_frame_sampling_failure_is_a_warning(tmp_path):
    with (
        patch.object(media, "download_media", AsyncMock(return_value=video())),
        patch.object(ocr, "sample_frames", AsyncMock(side_effect=RuntimeError("no codec"))),
        patch("shortform_notes.summarize._summarize_openai", AsyncMock(return_value=REPLY)),
    ):
        result = await import_reel("https://www.tiktok.com/@c/video/1", settings(tmp_path))
    assert result.sources == ("caption",)
    assert any("Vision failed" in w for w in result.warnings)


async def test_vision_skipped_when_no_video_downloaded(tmp_path):
    with (
        patch.object(media, "download_media", AsyncMock(return_value=video(video_path=None))),
        patch("shortform_notes.summarize._summarize_openai", AsyncMock(return_value=REPLY)),
    ):
        result = await import_reel("https://www.tiktok.com/@c/video/1", settings(tmp_path))
    assert any("Vision skipped: no video" in w for w in result.warnings)


async def test_agentic_on_an_api_backend_warns_in_the_note(tmp_path):
    with (
        patch.object(media, "download_media", AsyncMock(return_value=video())),
        patch.object(ocr, "sample_frames", AsyncMock(return_value=frames(3))),
        patch("shortform_notes.summarize._summarize_openai", AsyncMock(return_value=REPLY)),
    ):
        result = await import_reel(
            "https://www.tiktok.com/@c/video/1", settings(tmp_path, summary_provider="openai", vision_agentic=True)
        )
    assert any("Agentic vision needs an agent backend" in w for w in result.warnings)
    assert "video" in result.sources  # it still saw the sheets, so vision itself was not skipped


# ── flag, env and config precedence ────────────────────────────────────


def _load(**over) -> config.Settings:
    return config.load_settings(summary_provider="openai", transcribe_provider="none", **over)


def test_vision_defaults_off(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "none.env")
    monkeypatch.delenv("SHORTFORM_NOTES_VISION", raising=False)
    assert _load().vision is False


def test_vision_flag_beats_env_beats_config_file(monkeypatch, tmp_path):
    path = tmp_path / "config.env"
    config.write_config_file({"SHORTFORM_NOTES_VISION": "1"}, path)
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    monkeypatch.delenv("SHORTFORM_NOTES_VISION", raising=False)
    assert _load().vision is True
    monkeypatch.setenv("SHORTFORM_NOTES_VISION", "off")
    assert _load().vision is False
    assert _load(vision=True).vision is True
    monkeypatch.setenv("SHORTFORM_NOTES_VISION", "1")
    assert _load(vision=False).vision is False


def test_can_see_video_covers_every_backend_but_none(tmp_path):
    for provider in ("openai", "anthropic", "claude-code", "codex"):
        assert settings(tmp_path, summary_provider=provider).can_see_video is True, provider
    assert settings(tmp_path, summary_provider="none").can_see_video is False
    assert settings(tmp_path, vision=False).can_see_video is False
    assert settings(tmp_path, summary_provider="codex").vision_is_metered is False
    assert settings(tmp_path, summary_provider="openai").vision_is_metered is True


def test_cli_flags_reach_settings(tmp_path, monkeypatch):
    from shortform_notes.cli import main
    from shortform_notes.pipeline import ReelImportResult

    monkeypatch.delenv("SHORTFORM_NOTES_VISION", raising=False)
    result = ReelImportResult(tmp_path / "n.md", "T", "S", (), ("caption",), ())
    with patch("shortform_notes.cli.import_reel", AsyncMock(return_value=result)) as run:
        main(["https://youtu.be/x", "--vision", "--summary", "openai", "-o", str(tmp_path)])
        assert run.await_args.args[1].vision is True
        main(["https://youtu.be/x", "--no-vision", "--summary", "openai", "-o", str(tmp_path)])
        assert run.await_args.args[1].vision is False
        # `--vision agentic` turns on both; plain `--vision` leaves the mode off.
        main(["https://youtu.be/x", "--vision", "agentic", "--summary", "codex", "-o", str(tmp_path)])
        settings_used = run.await_args.args[1]
        assert settings_used.vision is True and settings_used.vision_agentic is True
        main(["https://youtu.be/x", "--vision", "--summary", "codex", "-o", str(tmp_path)])
        assert run.await_args.args[1].vision_agentic is False


def test_vision_env_takes_agentic_as_well_as_a_boolean(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "none.env")
    monkeypatch.setenv("SHORTFORM_NOTES_VISION", "agentic")
    loaded = _load()
    assert loaded.vision is True and loaded.vision_agentic is True
    monkeypatch.setenv("SHORTFORM_NOTES_VISION", "1")
    assert _load().vision is True and _load().vision_agentic is False
    monkeypatch.setenv("SHORTFORM_NOTES_VISION", "0")
    assert _load().vision is False and _load().vision_agentic is False
