---
name: reel
description: Import an Instagram Reel, TikTok or YouTube Short into a Markdown note via shortform-notes. Use when the user pastes such a link or invokes /reel <url>.
---

# /reel <url>

Run the shortform-notes pipeline on the link and report back.

1. Run: `shortform-notes "<url>" --vision --json` (add `-o <dir>` if the user named a folder). `--vision` shows the summary model the video's frames, so the note describes what happens on screen rather than guessing from the caption.
2. Parse the JSON. On `error`, tell the user what failed (Instagram private/deleted, yt-dlp blocked, etc.) and stop.
3. Reply with the note path, the title, the summary, and the takeaways as bullets. Mention any `warnings`.
   If the JSON has `"duplicate": true`, the link was already imported and nothing ran: say so, give the existing note path (and `notion_url`), and offer to re-import with `--force` only if the user wants a fresh version.
4. If the user asks follow-up questions about the video, read the note file. It has the verbatim caption and transcript, and — when vision ran — a `## Video breakdown` section of timestamped lines describing what is on screen moment by moment. Use that section for "what happened at X", "what did they do after Y", and anything visual; the same entries are in the JSON under `scenes`.

Never paste the raw transcript unless asked; the note has it.
