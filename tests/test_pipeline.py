"""URL detection, Instagram embed parsing, note rendering, and pipeline orchestration."""

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import ANY, AsyncMock, patch

import pytest

from shortform_notes import instagram, media, ocr, pipeline, urls
from shortform_notes.config import Settings
from shortform_notes.media import DownloadedMedia, MediaFetchError
from shortform_notes.note import ReelContent, build_note, note_filename, slugify
from shortform_notes.pipeline import ReelImportError, import_reel
from shortform_notes.summarize import Summary

NOW = datetime(2026, 8, 23, 20, 30, tzinfo=timezone.utc)


def settings(tmp_path: Path, openai: bool = False, anthropic: bool = False, provider: str | None = None) -> Settings:
    return Settings(
        output_dir=tmp_path / "reels",
        openai_api_key="sk-test" if openai else None,
        anthropic_api_key="sk-ant-test" if anthropic else None,
        summary_provider=provider or ("openai" if openai else "anthropic" if anthropic else "none"),
        transcribe_provider="openai" if openai else "none",
        openai_transcribe_model="gpt-transcribe",
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
        vision=False,
        vision_agentic=False,
        fps_explicit=False,
    )


def embed_html(caption: str = "Symmetry featuring @karanaujla out now", **overrides) -> str:
    """Build the captioned-embed HTML shape Instagram serves (JSON-in-a-JSON-string)."""
    media_ = {
        "shortcode": "DQCkNLtgqEe",
        "is_video": True,
        "display_url": "https://cdn.example/display.jpg",
        "thumbnail_src": "https://cdn.example/thumb.jpg",
        "video_duration": 26.192,
        "video_url": "https://cdn.example/video.mp4",
        "edge_media_to_caption": {"edges": [{"node": {"text": caption}}] if caption else []},
        "owner": {"username": "teddysphotos"},
        **overrides,
    }
    context = json.dumps({"context": {"type": "GraphVideo"}, "gql_data": {"shortcode_media": media_}})
    return '<html><script>{"contextJSON":' + json.dumps(context) + "}</script></html>"


def downloaded(**overrides) -> DownloadedMedia:
    base = dict(
        audio_path=None,
        video_path=None,
        caption="yt-dlp description",
        creator_handle="chef",
        creator_name="Chef Name",
        title=None,
        timestamp=1755000000,
        duration=30.0,
        thumbnail="https://cdn.example/yt.jpg",
        webpage_url="https://www.tiktok.com/@chef/video/1",
    )
    return DownloadedMedia(**{**base, **overrides})


# ── URL detection ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "check this https://www.instagram.com/reel/DQCkNLtgqEe/?igsh=abc",
            "https://www.instagram.com/reel/DQCkNLtgqEe/?igsh=abc",
        ),
        ("https://instagram.com/p/DQCkNLtgqEe/", "https://instagram.com/p/DQCkNLtgqEe/"),
        ("https://www.instagram.com/share/reel/_69O6RoGd", "https://www.instagram.com/share/reel/_69O6RoGd"),
        (
            "https://www.tiktok.com/@chef/video/7301234567890123456",
            "https://www.tiktok.com/@chef/video/7301234567890123456",
        ),
        ("https://vm.tiktok.com/ZMabc123/", "https://vm.tiktok.com/ZMabc123/"),
        ("https://youtube.com/shorts/dQw4w9WgXcQ?si=xyz", "https://youtube.com/shorts/dQw4w9WgXcQ?si=xyz"),
        ("look: https://youtu.be/dQw4w9WgXcQ.", "https://youtu.be/dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", None),
        ("https://example.com/recipe", None),
        ("no link here", None),
    ],
)
def test_detect_reel_url(text, expected):
    assert urls.detect_reel_url(text) == expected


def test_platform_and_tracking_strip():
    assert urls.platform_for("https://www.instagram.com/reel/X/") == "instagram"
    assert urls.platform_for("https://vm.tiktok.com/Z/") == "tiktok"
    assert urls.platform_for("https://youtu.be/Z") == "youtube"
    assert urls.strip_tracking("https://www.instagram.com/reel/X/?igsh=abc#frag") == "https://www.instagram.com/reel/X/"


# ── Instagram embed parsing ────────────────────────────────────────────


def test_parse_embed_context_json():
    embed = instagram.parse_embed(embed_html(), "DQCkNLtgqEe")
    assert embed.caption == "Symmetry featuring @karanaujla out now"
    assert embed.username == "teddysphotos"
    assert embed.video_url == "https://cdn.example/video.mp4"
    assert embed.duration == pytest.approx(26.192)
    assert embed.is_video is True


def test_parse_embed_carousel_image_urls():
    html = embed_html(
        is_video=False,
        video_url=None,
        edge_sidecar_to_children={
            "edges": [
                {"node": {"display_url": "https://cdn.example/slide1.jpg", "is_video": False}},
                {"node": {"display_url": "https://cdn.example/clip.jpg", "is_video": True}},
                {"node": {"display_url": "https://cdn.example/slide3.jpg", "is_video": False}},
            ]
        },
    )
    embed = instagram.parse_embed(html, "DQCkNLtgqEe")
    assert embed.is_video is False
    assert embed.image_urls == ("https://cdn.example/slide1.jpg", "https://cdn.example/slide3.jpg")


def test_parse_embed_single_photo_image_url():
    embed = instagram.parse_embed(embed_html(is_video=False, video_url=None), "DQCkNLtgqEe")
    assert embed.image_urls == ("https://cdn.example/display.jpg",)


def test_parse_embed_video_has_no_image_urls():
    assert instagram.parse_embed(embed_html(), "DQCkNLtgqEe").image_urls == ()


def test_parse_embed_no_caption():
    assert instagram.parse_embed(embed_html(caption=""), "X").caption is None


def test_parse_embed_caption_div_fallback():
    html = '<div class="Caption"><a>user</a> Line one<br>Line &amp; two</div></div>'
    embed = instagram.parse_embed(html, "ABC")
    assert embed.caption == "user Line one\nLine & two"
    assert embed.shortcode == "ABC"


def test_parse_embed_shell_only_is_none():
    assert instagram.parse_embed("<html><script>window.__d = 1</script></html>" * 100, "X") is None


async def test_resolve_shortcode_direct():
    assert await instagram.resolve_shortcode("https://www.instagram.com/reel/DQCkNLtgqEe/?igsh=1") == "DQCkNLtgqEe"
    assert await instagram.resolve_shortcode("https://www.tiktok.com/@x/video/1") is None


# ── Note rendering ─────────────────────────────────────────────────────


def test_slugify_and_filename():
    assert slugify("Symmetry ft. Karan Aujla — Out NOW!") == "symmetry-ft-karan-aujla-out-now"
    assert slugify("") == "reel"
    assert note_filename(NOW, "teddysphotos", "Symmetry out now") == "2026-08-23-teddysphotos-symmetry-out-now.md"
    assert note_filename(NOW, None, "T") == "2026-08-23-t.md"


def test_build_note_sections():
    content = ReelContent(
        url="https://www.instagram.com/reel/X/",
        platform="instagram",
        caption="Cap line 1\n\nCap line 2",
        transcript="spoken words",
        screen_text="[00:02] 2 cups flour",
        title=None,
        creator_handle="chef",
        creator_name="Chef",
        posted=NOW,
        duration=26.2,
        thumbnail="https://cdn.example/t.jpg",
        sources=("caption", "transcript"),
        warnings=("w1",),
    )
    note = build_note(content, "Title", "A summary.", ("one", "two"), NOW)
    assert note.startswith("---\ntype: reel\nplatform: instagram\n")
    assert 'creator: "@chef"' in note
    assert "sources: [caption, transcript]" in note
    assert "duration_seconds: 26" in note
    assert "## Summary\n\nA summary." in note
    assert "- one\n- two" in note
    assert "> Cap line 1\n>\n> Cap line 2" in note
    assert "## Transcript\n\nspoken words" in note
    assert "## On-screen text\n\n[00:02] 2 cups flour" in note
    assert "- w1" in note


# ── Pipeline ───────────────────────────────────────────────────────────


async def test_import_reel_no_keys_caption_only(tmp_path):
    """Zero-config path: caption + metadata note, no transcript, no summary, no network to LLMs."""
    with (
        patch.object(
            instagram, "fetch_embed", AsyncMock(return_value=instagram.parse_embed(embed_html(), "DQCkNLtgqEe"))
        ),
        patch.object(media, "download_media", AsyncMock(return_value=downloaded(audio_path=None))),
    ):
        result = await import_reel("https://www.instagram.com/reel/DQCkNLtgqEe/?igsh=abc", settings(tmp_path), NOW)
    assert result.sources == ("caption",)
    assert result.path.name == "2025-08-12-teddysphotos-symmetry-featuring-karanaujla-out-now.md"
    text = result.path.read_text()
    assert "Symmetry featuring @karanaujla out now" in text
    assert "_No transcript._" in text
    assert any("Transcription skipped" in w for w in result.warnings)


async def test_import_reel_full_path(tmp_path):
    with (
        patch.object(
            instagram, "fetch_embed", AsyncMock(return_value=instagram.parse_embed(embed_html(), "DQCkNLtgqEe"))
        ),
        patch.object(media, "download_media", AsyncMock(return_value=downloaded(audio_path=str(tmp_path / "a.m4a")))),
        patch.object(pipeline, "transcribe", AsyncMock(return_value="spoken words")),
        patch(
            "shortform_notes.summarize._summarize_openai",
            AsyncMock(return_value={"title": "Symmetry", "summary": "S.", "takeaways": ["t1"]}),
        ),
    ):
        result = await import_reel("https://www.instagram.com/reel/DQCkNLtgqEe/", settings(tmp_path, openai=True), NOW)
    assert result.sources == ("caption", "transcript")
    assert result.title == "Symmetry"
    assert result.takeaways == ("t1",)
    assert result.path.name == "2025-08-12-teddysphotos-symmetry.md"
    assert "spoken words" in result.path.read_text()


async def test_import_reel_anthropic_summary(tmp_path):
    with (
        patch.object(media, "download_media", AsyncMock(return_value=downloaded(audio_path=None))),
        patch(
            "shortform_notes.summarize._summarize_anthropic",
            AsyncMock(return_value={"title": "T", "summary": "S", "takeaways": []}),
        ) as claude,
    ):
        result = await import_reel("https://www.tiktok.com/@chef/video/1", settings(tmp_path, anthropic=True), NOW)
    assert claude.await_count == 1
    assert result.title == "T"
    assert result.sources == ("caption",)


async def test_import_reel_summary_failure_degrades(tmp_path):
    with (
        patch.object(
            media, "download_media", AsyncMock(return_value=downloaded(caption="First line\nmore", audio_path=None))
        ),
        patch("shortform_notes.summarize._summarize_openai", AsyncMock(side_effect=RuntimeError("boom"))),
    ):
        result = await import_reel("https://youtube.com/shorts/dQw4w9WgXcQ", settings(tmp_path, openai=True), NOW)
    assert result.title == "First line"
    assert result.summary == ""
    assert result.path.exists()


async def test_import_reel_nothing_fetched_raises(tmp_path):
    with (
        patch.object(instagram, "fetch_embed", AsyncMock(return_value=None)),
        patch.object(media, "download_media", AsyncMock(side_effect=MediaFetchError("login required"))),
        pytest.raises(ReelImportError, match="login required"),
    ):
        await import_reel("https://www.instagram.com/reel/DQCkNLtgqEe/", settings(tmp_path), NOW)


async def test_import_reel_rejects_unsupported_url(tmp_path):
    with pytest.raises(ReelImportError, match="not a supported"):
        await import_reel("https://example.com/video", settings(tmp_path), NOW)


async def test_duplicate_filename_gets_suffix(tmp_path):
    s = settings(tmp_path)
    with patch.object(media, "download_media", AsyncMock(return_value=downloaded(audio_path=None))):
        first = await import_reel("https://www.tiktok.com/@chef/video/1", s, NOW)
        second = await import_reel("https://www.tiktok.com/@chef/video/2", s, NOW)  # another video, same title
    assert first.path != second.path
    assert second.path.name.endswith("-203000.md")


async def test_import_reel_carousel_slides_feed_vision(tmp_path):
    """No video from yt-dlp, but the embed lists slides: they become the frames the summary model sees."""
    html = embed_html(
        is_video=False,
        video_url=None,
        edge_sidecar_to_children={
            "edges": [
                {"node": {"display_url": "https://cdn.example/slide1.jpg", "is_video": False}},
                {"node": {"display_url": "https://cdn.example/slide2.jpg", "is_video": False}},
            ]
        },
    )
    fake_frames = [ocr.Frame(seconds=0.0, png=b"png1"), ocr.Frame(seconds=1.0, png=b"png2")]
    summarize_mock = AsyncMock(return_value=Summary("Slides", "S", (), ()))
    with (
        patch.object(instagram, "fetch_embed", AsyncMock(return_value=instagram.parse_embed(html, "DQCkNLtgqEe"))),
        patch.object(media, "download_media", AsyncMock(side_effect=media.MediaFetchError("No video formats found!"))),
        patch.object(pipeline, "fetch_slides", AsyncMock(return_value=([b"jpg1", b"jpg2"], []))) as slides,
        patch.object(ocr, "frames_from_images", AsyncMock(return_value=fake_frames)) as decode,
        patch.object(pipeline, "summarize", summarize_mock),
    ):
        result = await import_reel(
            "https://www.instagram.com/p/DQCkNLtgqEe/", replace(settings(tmp_path, openai=True), vision=True), NOW
        )
    slides.assert_awaited_once_with(("https://cdn.example/slide1.jpg", "https://cdn.example/slide2.jpg"))
    decode.assert_awaited_once_with([b"jpg1", b"jpg2"])
    assert result.sources == ("caption", "slides")
    assert summarize_mock.await_args.kwargs["frames"] == fake_frames
    assert not any("yt-dlp could not fetch" in w for w in result.warnings)


async def test_import_reel_still_post_is_ocrd_without_ocr_flag(tmp_path):
    """A still post is its text: slides are OCR'd even when --ocr is off and vision is off."""
    html = embed_html(is_video=False, video_url=None)
    fake_frames = [ocr.Frame(seconds=0.0, png=b"png1")]
    with (
        patch.object(instagram, "fetch_embed", AsyncMock(return_value=instagram.parse_embed(html, "DQCkNLtgqEe"))),
        patch.object(media, "download_media", AsyncMock(side_effect=MediaFetchError("No video formats found!"))),
        patch.object(pipeline, "fetch_slides", AsyncMock(return_value=([b"jpg1"], []))),
        patch.object(ocr, "frames_from_images", AsyncMock(return_value=fake_frames)),
        patch.object(ocr, "read_screen_text", AsyncMock(return_value=("[00:00] TOP 100 FILMS", 1))) as read,
        patch.object(pipeline, "summarize", AsyncMock(return_value=Summary("Slides", "S", (), ()))) as summ,
    ):
        result = await import_reel("https://www.instagram.com/p/DQCkNLtgqEe/", settings(tmp_path, openai=True), NOW)
    read.assert_awaited_once_with("", ANY, fake_frames)
    assert result.sources == ("caption", "screen_text")
    assert summ.await_args.kwargs["frames"] == []  # vision off: the model gets the OCR text, not the images
    assert "## On-screen text" in result.path.read_text()
    assert "TOP 100 FILMS" in result.path.read_text()
