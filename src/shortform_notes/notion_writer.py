"""Write an imported reel into a Notion database as one page.

Only ``httpx`` (already a core dependency) is needed. The flow, against Notion API
version ``Settings.notion_version`` (2026-03-11 when this was written):

1. Resolve the database to its data source. Since 2025-09-03 a database is a
   container of data sources; pages are created under a *data source*, and the
   property schema lives there too. A data source id is accepted directly.
2. Make sure the properties below exist, adding any that are missing (so an empty
   database works) and skipping, with a warning, any that exist with another type.
   Then look for a page with the same video in ``Source URL`` (any spelling of its
   link, see ``urls.reel_key``). One found is returned instead of creating a second;
   under ``replace_existing`` it is moved to the trash and a new page made.
3. Upload every kept screenshot: ``POST /v1/file_uploads`` then
   ``POST /v1/file_uploads/{id}/send`` with the bytes as multipart ``file``.
4. Create the page with the properties and the first 100 blocks, append the rest
   100 at a time, then set the first screenshot as the page cover (for gallery views).

New select / multi-select options (a category the database has not seen yet) are
created by Notion itself when a page uses them.

Rename the columns by editing the ``PROP_*`` constants. The title column is found
by type, so it can be called anything.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from shortform_notes.config import Settings
from shortform_notes.note import ReelContent, Scene
from shortform_notes.urls import reel_key, reel_search_term

logger = logging.getLogger(__name__)

NOTION_API = "https://api.notion.com/v1"
TIMEOUT = 60.0
MAX_RETRIES = 4

PROP_TITLE = "Name"  # used only when creating a title column is somehow needed; the existing one is found by type
PROP_SOURCE = "Source URL"
PROP_PLATFORM = "Platform"
PROP_CREATOR = "Creator"
PROP_POSTED = "Posted"
PROP_DURATION = "Duration (s)"
PROP_CATEGORY = "Category"
PROP_TAGS = "Tags"

# name -> Notion property type. The title property is handled separately.
SCHEMA = {
    PROP_SOURCE: "url",
    PROP_PLATFORM: "select",
    PROP_CREATOR: "rich_text",
    PROP_POSTED: "date",
    PROP_DURATION: "number",
    PROP_CATEGORY: "select",
    PROP_TAGS: "multi_select",
}

# Notion's request limits (https://developers.notion.com/reference/request-limits).
RICH_TEXT_MAX = 2000  # characters per text object
BLOCKS_PER_REQUEST = 100
OPTION_NAME_MAX = 100
MAX_TAGS = 10


class NotionError(Exception):
    """A Notion request failed; the message says what to fix when that is knowable."""


@dataclass(frozen=True)
class NotionPage:
    id: str
    url: str
    warnings: tuple[str, ...] = ()
    existing: bool = False  # an earlier page for the same video, linked instead of creating another


# ── HTTP ───────────────────────────────────────────────────────────────


class _Notion:
    def __init__(self, http: httpx.AsyncClient, settings: Settings) -> None:
        self.http = http
        self.headers = {
            "Authorization": f"Bearer {settings.notion_token}",
            "Notion-Version": settings.notion_version,
        }

    async def request(self, method: str, path: str, **kwargs) -> dict:
        """One API call with retries on 429 / 5xx (honouring Retry-After); NotionError on anything else."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = await self.http.request(method, f"{NOTION_API}{path}", headers=self.headers, **kwargs)
            except httpx.TransportError as exc:
                if attempt == MAX_RETRIES:
                    raise NotionError(f"{method} {path}: {exc}") from exc
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == MAX_RETRIES:
                    break
                await asyncio.sleep(float(resp.headers.get("Retry-After") or 2**attempt))
                continue
            if resp.is_success:
                return resp.json()
            break
        raise NotionError(_explain(resp, method, path))


def _explain(resp: httpx.Response, method: str, path: str) -> str:
    try:
        body = resp.json()
    except ValueError:
        body = {}
    code, message = body.get("code", ""), body.get("message", resp.text[:200])
    if resp.status_code == 401:
        return "Notion rejected the token (401). NOTION_TOKEN is the integration's Internal Integration Secret."
    if resp.status_code == 404 and ("/databases/" in path or "/data_sources/" in path):
        return (
            "Notion could not find the database (404). Share it with your integration: open the database, "
            "click ••• → Connections → add your integration. Then check NOTION_DATABASE_ID."
        )
    return f"Notion {method} {path} failed ({resp.status_code} {code}): {message}"


# ── schema ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Schema:
    data_source_id: str
    title_property: str
    usable: frozenset[str]  # names from SCHEMA that exist with the right type


async def resolve_data_source(notion: _Notion, database_id: str) -> tuple[str, dict]:
    """(data source id, its properties) for a database id — or for a data source id given directly."""
    try:
        database = await notion.request("GET", f"/databases/{database_id}")
    except NotionError as exc:
        try:
            source = await notion.request("GET", f"/data_sources/{database_id}")
        except NotionError:
            raise exc from None
        return source["id"], source.get("properties") or {}
    sources = database.get("data_sources") or []
    if not sources:
        raise NotionError("the Notion database has no data source; is it a linked view of another database?")
    if len(sources) > 1:
        logger.info(
            "Notion database has %d data sources; writing to the first, %r", len(sources), sources[0].get("name")
        )
    source = await notion.request("GET", f"/data_sources/{sources[0]['id']}")
    return source["id"], source.get("properties") or {}


async def ensure_schema(notion: _Notion, database_id: str, warnings: list[str]) -> _Schema:
    source_id, properties = await resolve_data_source(notion, database_id)
    title = next((name for name, prop in properties.items() if prop.get("type") == "title"), PROP_TITLE)
    missing = {name: kind for name, kind in SCHEMA.items() if name not in properties}
    wrong = {name for name, kind in SCHEMA.items() if name in properties and properties[name].get("type") != kind}
    for name in sorted(wrong):
        warnings.append(
            f"Notion column {name!r} is a {properties[name].get('type')}, expected {SCHEMA[name]}; left empty"
        )
    if missing:
        logger.info("adding Notion columns: %s", ", ".join(missing))
        try:
            await notion.request(
                "PATCH", f"/data_sources/{source_id}", json={"properties": {n: {k: {}} for n, k in missing.items()}}
            )
        except NotionError as exc:
            warnings.append(f"Could not add Notion columns {', '.join(missing)}: {exc}")
            usable = frozenset(n for n in SCHEMA if n in properties and n not in wrong)
            return _Schema(source_id, title, usable)
    usable = frozenset(n for n in SCHEMA if n not in wrong)
    return _Schema(source_id, title, usable)


# ── content ────────────────────────────────────────────────────────────


def _chunks(text: str, size: int = RICH_TEXT_MAX) -> list[str]:
    """Split on paragraph, then sentence, then word boundaries so no piece exceeds ``size``."""
    text = text.strip()
    pieces: list[str] = []
    while len(text) > size:
        cut = max(text.rfind("\n", 0, size), text.rfind(". ", 0, size) + 1, text.rfind(" ", 0, size))
        cut = cut if cut > size // 2 else size
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def rich_text(text: str, link: str | None = None) -> list[dict]:
    items = []
    for piece in _chunks(text):
        item: dict = {"type": "text", "text": {"content": piece}}
        if link:
            item["text"]["link"] = {"url": link}
        items.append(item)
    return items


def _block(kind: str, text: str = "", **extra) -> dict:
    return {"object": "block", "type": kind, kind: {"rich_text": rich_text(text), **extra}}


def _paragraphs(text: str, kind: str = "paragraph") -> list[dict]:
    """A long text as several blocks, one per ~2000 characters, keeping its own paragraph breaks."""
    blocks = []
    for para in re.split(r"\n\s*\n", text.strip()):
        blocks += [_block(kind, piece) for piece in _chunks(para)]
    return blocks


def _option(name: str) -> str:
    """Select option names may not contain commas and are capped at 100 characters."""
    return name.replace(",", " ").strip()[:OPTION_NAME_MAX]


def hashtags(caption: str | None, limit: int = MAX_TAGS) -> list[str]:
    seen: dict[str, None] = {}
    for tag in re.findall(r"#(\w[\w.]*)", caption or ""):
        seen.setdefault(tag.lower().rstrip("."), None)
    return list(seen)[:limit]


def build_properties(content: ReelContent, title: str, category: str | None, schema: _Schema) -> dict:
    props: dict = {schema.title_property: {"title": rich_text(title[:RICH_TEXT_MAX]) or rich_text("Reel")}}
    creator = f"@{content.creator_handle}" if content.creator_handle else content.creator_name
    values = {
        PROP_SOURCE: {"url": content.url},
        PROP_PLATFORM: {"select": {"name": _option(content.platform)}},
        PROP_CREATOR: {"rich_text": rich_text(creator)} if creator else None,
        PROP_POSTED: {"date": {"start": content.posted.strftime("%Y-%m-%d")}} if content.posted else None,
        PROP_DURATION: {"number": round(content.duration)} if content.duration else None,
        PROP_CATEGORY: {"select": {"name": _option(category)}} if category else None,
        PROP_TAGS: {"multi_select": [{"name": _option(t)} for t in hashtags(content.caption)]},
    }
    props.update({k: v for k, v in values.items() if v is not None and k in schema.usable})
    return props


def build_blocks(
    content: ReelContent,
    summary: str,
    takeaways: Sequence[str],
    scenes: Sequence[Scene],
    uploads: dict[str, str],  # scene image_path -> file upload id
) -> list[dict]:
    blocks: list[dict] = []
    creator = f"@{content.creator_handle}" if content.creator_handle else (content.creator_name or "unknown creator")
    blocks.append(
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": rich_text(f"{content.platform}, {creator}: ") + rich_text(content.url, link=content.url)
            },
        }
    )
    if summary:
        blocks += [_block("heading_2", "Summary"), *_paragraphs(summary)]
    if takeaways:
        blocks.append(_block("heading_2", "Key takeaways"))
        blocks += [_block("bulleted_list_item", t) for t in takeaways]
    shots = [s for s in scenes if s.image_path and s.image_path in uploads]
    if shots:
        blocks.append(_block("heading_2", "Screenshots"))
        for scene in shots:
            caption = f"[{scene.time}] {scene.description}" if scene.time else scene.description
            blocks.append(
                {
                    "object": "block",
                    "type": "image",
                    "image": {
                        "type": "file_upload",
                        "file_upload": {"id": uploads[scene.image_path]},
                        "caption": rich_text(caption)[:1],  # one text object is plenty for a caption
                    },
                }
            )
    if scenes:
        # A toggle keeps a long breakdown out of the way; its children count against the same 100 limit.
        children = [
            _block("bulleted_list_item", f"[{s.time}] {s.description}" if s.time else s.description)
            for s in scenes[:BLOCKS_PER_REQUEST]
        ]
        blocks.append(_block("toggle", f"Video breakdown ({len(scenes)} scenes)", children=children))
    blocks.append(_block("heading_2", "Transcript"))
    blocks += _paragraphs(content.transcript) if content.transcript else [_block("paragraph", "No transcript.")]
    if content.caption:
        blocks += [_block("heading_2", "Caption"), *_paragraphs(content.caption, kind="quote")]
    if content.screen_text:
        children = _paragraphs(content.screen_text)[:BLOCKS_PER_REQUEST]
        blocks.append(_block("toggle", "On-screen text", children=children))
    if content.warnings:
        children = [_block("bulleted_list_item", w) for w in content.warnings][:BLOCKS_PER_REQUEST]
        blocks.append(_block("toggle", "Import warnings", children=children))
    return blocks


# ── uploads ────────────────────────────────────────────────────────────


async def upload_image(notion: _Notion, path: Path) -> str:
    """Single-part file upload (screenshots are well under the 20 MiB cap). Returns the upload id."""
    data = await asyncio.to_thread(path.read_bytes)
    created = await notion.request(
        "POST", "/file_uploads", json={"mode": "single_part", "filename": path.name, "content_type": "image/png"}
    )
    sent = await notion.request(
        "POST", f"/file_uploads/{created['id']}/send", files={"file": (path.name, data, "image/png")}
    )
    if sent.get("status") != "uploaded":
        raise NotionError(f"upload of {path.name} ended as {sent.get('status')!r}")
    return created["id"]


# ── existing pages ─────────────────────────────────────────────────────


async def find_pages(notion: _Notion, schema: _Schema, url: str) -> list[dict]:
    """Pages in the data source whose ``Source URL`` is this video, newest first.

    Notion can only filter on a substring, so the filter narrows and ``reel_key`` decides.
    """
    key = reel_key(url)
    found = await notion.request(
        "POST",
        f"/data_sources/{schema.data_source_id}/query",
        json={
            "filter": {"property": PROP_SOURCE, "url": {"contains": reel_search_term(key)}},
            "sorts": [{"timestamp": "created_time", "direction": "descending"}],
            "page_size": 20,
        },
    )
    pages = []
    for page in found.get("results") or []:
        source = ((page.get("properties") or {}).get(PROP_SOURCE) or {}).get("url")
        if source and reel_key(source) == key and not page.get("in_trash"):
            pages.append(page)
    return pages


# ── entry point ────────────────────────────────────────────────────────


async def create_reel_page(
    settings: Settings,
    content: ReelContent,
    title: str,
    summary: str,
    takeaways: Sequence[str],
    scenes: Sequence[Scene],
    category: str | None,
    base_dir: Path,
    http: httpx.AsyncClient | None = None,
    replace_existing: bool = False,
) -> NotionPage:
    """Create the page and return its URL. Raises NotionError when the page itself cannot be made;
    a screenshot or cover that fails is a warning on the returned page instead.

    When the database already has a page for this video, that page is returned (``existing``)
    and nothing is written; with ``replace_existing`` it is trashed and a new one created.
    ``scenes[*].image_path`` is relative to ``base_dir`` (the note's folder).
    """
    if not settings.can_write_notion:
        raise NotionError("Notion is not configured (NOTION_TOKEN and NOTION_DATABASE_ID)")
    warnings: list[str] = []
    owns_client = http is None
    http = http or httpx.AsyncClient(timeout=TIMEOUT)
    try:
        notion = _Notion(http, settings)
        schema = await ensure_schema(notion, settings.notion_database_id or "", warnings)

        if PROP_SOURCE in schema.usable:
            try:
                earlier = await find_pages(notion, schema, content.url)
            except NotionError as exc:
                earlier = []
                warnings.append(f"Could not check Notion for an earlier page of this video: {exc}")
            if earlier and not replace_existing:
                page = earlier[0]
                logger.info("Notion already has a page for this video: %s", page.get("url"))
                return NotionPage(id=page["id"], url=page.get("url") or "", warnings=tuple(warnings), existing=True)
            for page in earlier:
                try:
                    await notion.request("PATCH", f"/pages/{page['id']}", json={"in_trash": True})
                    logger.info("moved the earlier Notion page to the trash: %s", page.get("url"))
                except NotionError as exc:
                    warnings.append(f"The earlier Notion page {page.get('url')} was not moved to the trash: {exc}")

        uploads: dict[str, str] = {}
        for scene in scenes:
            if not scene.image_path:
                continue
            try:
                uploads[scene.image_path] = await upload_image(notion, base_dir / scene.image_path)
            except (NotionError, OSError) as exc:
                warnings.append(f"Screenshot {scene.time or scene.image_path} was not uploaded to Notion: {exc}")

        blocks = build_blocks(content, summary, takeaways, scenes, uploads)
        page = await notion.request(
            "POST",
            "/pages",
            json={
                "parent": {"type": "data_source_id", "data_source_id": schema.data_source_id},
                "properties": build_properties(content, title, category, schema),
                "children": blocks[:BLOCKS_PER_REQUEST],
            },
        )
        for start in range(BLOCKS_PER_REQUEST, len(blocks), BLOCKS_PER_REQUEST):
            try:
                await notion.request(
                    "PATCH",
                    f"/blocks/{page['id']}/children",
                    json={"children": blocks[start : start + BLOCKS_PER_REQUEST]},
                )
            except NotionError as exc:
                warnings.append(f"Part of the Notion page body was not written: {exc}")
                break

        first_shot = next((s for s in scenes if s.image_path and s.image_path in uploads), None)
        if first_shot:
            # A file upload attaches once, so the cover gets its own copy of the image.
            try:
                cover_id = await upload_image(notion, base_dir / (first_shot.image_path or ""))
                await notion.request(
                    "PATCH",
                    f"/pages/{page['id']}",
                    json={"cover": {"type": "file_upload", "file_upload": {"id": cover_id}}},
                )
            except (NotionError, OSError) as exc:
                logger.info("Notion page cover not set: %s", exc)
        logger.info("Notion page created: %s (%d screenshots)", page.get("url"), len(uploads))
        return NotionPage(id=page["id"], url=page.get("url") or "", warnings=tuple(warnings))
    finally:
        if owns_client:
            await http.aclose()


async def check_connection(settings: Settings, http: httpx.AsyncClient | None = None) -> str:
    """For setup: resolve the database and report what will be written to. Raises NotionError."""
    owns_client = http is None
    http = http or httpx.AsyncClient(timeout=TIMEOUT)
    try:
        notion = _Notion(http, settings)
        source_id, properties = await resolve_data_source(notion, settings.notion_database_id or "")
        missing = [n for n in SCHEMA if n not in properties]
        note = f"; will add columns: {', '.join(missing)}" if missing else ""
        return f"connected: data source {source_id} with {len(properties)} columns{note}"
    finally:
        if owns_client:
            await http.aclose()
