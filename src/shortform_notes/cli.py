"""``shortform-notes <url> [<url> ...]``: the command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from shortform_notes import __version__
from shortform_notes.config import (
    OCR_PROVIDERS,
    SUMMARY_PROVIDERS,
    TRANSCRIBE_PROVIDERS,
    WHISPER_DEVICES,
    load_settings,
)
from shortform_notes.ocr import FRAMES_PER_GRID
from shortform_notes.pipeline import ReelImportError, import_reel
from shortform_notes.summarize import MAX_VISION_FRAMES, vision_estimate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shortform-notes",
        description=(
            "Turn Instagram / Facebook Reels, TikToks and YouTube Shorts into Markdown notes, and optionally "
            "Notion pages: transcript, a few well-chosen screenshots, category and summary."
        ),
        epilog=(
            "Without an API key, summaries run through `claude` (Claude Code) or `codex` if either is on your PATH. "
            "Env: OPENAI_API_KEY, ANTHROPIC_API_KEY, SHORTFORM_NOTES_DIR, SHORTFORM_NOTES_SUMMARY_PROVIDER, "
            "SHORTFORM_NOTES_TRANSCRIBE_PROVIDER, SHORTFORM_NOTES_AUDIENCE, NOTION_TOKEN, NOTION_DATABASE_ID, "
            "SHORTFORM_NOTES_CATEGORIES, SHORTFORM_NOTES_WHISPER_DEVICE. See .env.example."
        ),
    )
    parser.add_argument(
        "urls",
        nargs="*",
        help=(
            "one or more reel / short-video links; or 'web' (browser setup), "
            "'setup' (terminal setup), 'mcp' (MCP server)"
        ),
    )
    parser.add_argument("-o", "--out", help="output directory (default: ./reels or $SHORTFORM_NOTES_DIR)")
    parser.add_argument(
        "--summary",
        choices=[*SUMMARY_PROVIDERS, "auto"],
        help="summary backend: openai, anthropic, claude-code, codex, or none (default: auto)",
    )
    parser.add_argument(
        "--transcribe",
        choices=[*TRANSCRIBE_PROVIDERS, "auto"],
        help="openai (API key), local (offline faster-whisper), or none (default: auto)",
    )
    parser.add_argument("--no-transcript", action="store_true", help="alias for --transcribe none")
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="also read on-screen text from video frames (slower; costs money on API backends, free with local)",
    )
    parser.add_argument("--no-ocr", action="store_true", help="turn OCR off even if the config enables it")
    parser.add_argument("--ocr-provider", choices=list(OCR_PROVIDERS), help="local (free), openai, or anthropic")
    parser.add_argument(
        "--ocr-fps",
        type=float,
        help=(
            "sample frames on a clock instead of at the video's cuts, for both OCR and --vision; "
            "1 is one per second, 0 is every frame"
        ),
    )
    parser.add_argument(
        "--vision",
        nargs="?",
        const="on",
        choices=("on", "agentic"),
        help=(
            "show the summary model the video: sampled frames are tiled into timestamped contact sheets "
            f"and sent with the summary call (any backend but 'none'; at most {MAX_VISION_FRAMES} frames "
            f"per video, {FRAMES_PER_GRID} to a sheet). `--vision agentic` additionally lets the "
            "claude-code and codex backends open the full-resolution frames themselves; slower, but it "
            "can read fine print and check a fast cut the sheets blur"
        ),
    )
    parser.add_argument("--no-vision", action="store_true", help="turn vision off even if the config enables it")
    parser.add_argument(
        "--notion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "also write each reel as a page in a Notion database (default: on when NOTION_TOKEN and "
            "NOTION_DATABASE_ID are set). Turns --vision on unless --no-vision is given, since the "
            "screenshots are chosen by the model looking at the frames"
        ),
    )
    parser.add_argument(
        "--categories",
        metavar="LIST",
        help=(
            "comma-separated category taxonomy for this run, e.g. 'Recipes, Fitness, Tech, Other' "
            "(default: $SHORTFORM_NOTES_CATEGORIES or a built-in list); each video gets exactly one"
        ),
    )
    parser.add_argument(
        "--whisper-device",
        choices=list(WHISPER_DEVICES),
        help="local transcription device: auto (GPU when ctranslate2 sees one), cuda, or cpu",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    parser.add_argument("-v", "--verbose", action="store_true", help="show fetch/debug logs")
    parser.add_argument("--version", action="version", version=f"shortform-notes {__version__}")
    return parser


def _print_result(result, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    print(f"saved {result.path}  (sources: {', '.join(result.sources)})")
    print(f"  {result.title}" + (f"  [{result.category}]" if result.category else ""))
    if result.notion_url:
        print(f"  notion: {result.notion_url}")
    if result.summary:
        print(f"  {result.summary}")
    for t in result.takeaways:
        print(f"  - {t}")
    if result.scenes:
        print(
            f"  ({len(result.scenes)} scenes under 'Video breakdown' in the note, "
            f"{result.screenshot_count} kept as screenshots)"
        )
    for w in result.warnings:
        print(f"  warning: {w}")


async def _run(urls: list[str], args: argparse.Namespace) -> int:
    settings = load_settings(
        output_dir=args.out,
        summary_provider=args.summary,
        transcribe_provider="none" if args.no_transcript else args.transcribe,
        ocr=False if args.no_ocr else (True if args.ocr else None),
        ocr_provider=args.ocr_provider,
        ocr_fps=args.ocr_fps,
        vision=False if args.no_vision else (True if args.vision else None),
        vision_agentic=False if args.no_vision else (True if args.vision == "agentic" else None),
        notion=args.notion,
        categories=args.categories,
        whisper_device=args.whisper_device,
    )
    if args.notion and not settings.can_write_notion and not args.json:
        print(
            "--notion: NOTION_TOKEN and NOTION_DATABASE_ID must both be set; writing local notes only", file=sys.stderr
        )
    rate = f"{settings.ocr_fps:g} frames/s" if settings.ocr_fps else "every frame"
    if settings.ocr and settings.ocr_provider != "local" and not args.json:
        from shortform_notes.ocr import estimate

        for secs in (30, 60):
            note = estimate(secs, settings).describe()
            print(f"OCR on ({settings.ocr_provider}, {rate}): a {secs}s video is {note}", file=sys.stderr)
    if settings.can_see_video and not args.json:
        # One line, not two: past the frame cap every duration sends the same number of sheets.
        note = vision_estimate(60, settings).describe()
        print(f"vision on ({settings.summary_provider}, {rate}): a 60s video sends {note}", file=sys.stderr)
    failures = 0
    for url in urls:
        try:
            _print_result(await import_reel(url, settings), args.json)
        except ReelImportError as exc:
            failures += 1
            if args.json:
                print(json.dumps({"url": url, "error": str(exc)}))
            else:
                print(f"failed {url}\n  {exc}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    if args.urls == ["setup"]:  # terminal wizard; writes the same config file as the web page
        from shortform_notes.setup_cli import main as setup_main

        return setup_main()
    if args.urls and args.urls[0] == "web":  # `shortform-notes web [port]`: local setup and import UI
        from shortform_notes.web.server import serve

        serve(port=int(args.urls[1]) if len(args.urls) > 1 else None, open_browser=not args.json)
        return 0
    if args.urls == ["mcp"]:  # `shortform-notes mcp`: stdio MCP server for Claude Code / Claude Desktop
        from shortform_notes.mcp_server import run_server

        run_server()
        return 0
    if not args.urls:
        parser.print_help()
        return 2
    return asyncio.run(_run(args.urls, args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
