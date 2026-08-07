#!/usr/bin/env python3
"""Export a Claude Code session transcript (.jsonl) to Markdown and/or HTML.

Usage:
    python3 tools/export_chat.py SESSION.jsonl -o export/chat
    python3 tools/export_chat.py SESSION.jsonl -o export/chat --thinking
    python3 tools/export_chat.py --latest -o export/chat

Produces chat.md and chat.html. Convert the HTML to PDF with:
    libreoffice --headless --convert-to pdf export/chat.html
"""

import argparse
import html as htmllib
import json
import re
from datetime import datetime
from pathlib import Path

SKIP_TYPES = {
    "queue-operation", "attachment", "file-history-snapshot",
    "file-history-delta", "last-prompt", "ai-title", "mode", "system",
}

SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
IDE_SELECTION_RE = re.compile(r"<ide_selection>.*?</ide_selection>", re.S)

# Records that are automated notifications rather than human turns.
NOTIFICATION_MARKERS = (
    "[SYSTEM NOTIFICATION - NOT USER INPUT]",
    "<task-notification>",
    "<local-command-stdout>",
)

# Which input field best summarises each tool call.
TOOL_KEY_FIELD = {
    "Bash": "command",
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "Glob": "pattern",
    "Grep": "pattern",
    "WebFetch": "url",
    "WebSearch": "query",
    "Skill": "skill",
    "Agent": "prompt",
}


def clean(text):
    """Strip injected system blocks from message text."""
    text = SYSTEM_REMINDER_RE.sub("", text)
    text = IDE_SELECTION_RE.sub("", text)
    return text.strip()


def is_notification(text):
    return any(marker in text for marker in NOTIFICATION_MARKERS)


def block_text(content):
    """Flatten a tool_result content field into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, dict) and b.get("type") == "image":
                parts.append("[image]")
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts)
    return ""


def truncate(text, limit):
    if limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n... [{omitted:,} more characters truncated]"


def parse(path, include_thinking=False, tool_limit=1500):
    """Walk the transcript and yield a flat list of render events."""
    results = {}   # tool_use_id -> result text
    records = []

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # First pass: collect tool results so we can attach them to their calls.
    for d in records:
        msg = d.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                results[b.get("tool_use_id")] = block_text(b.get("content"))

    events = []
    for d in records:
        rtype = d.get("type")
        if rtype in SKIP_TYPES:
            continue
        msg = d.get("message")
        if not isinstance(msg, dict):
            continue

        ts = d.get("timestamp", "")
        content = msg.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            continue

        for b in content:
            if not isinstance(b, dict):
                continue
            btype = b.get("type")

            if btype == "text":
                text = clean(b.get("text", ""))
                if not text or is_notification(text):
                    continue
                events.append({"kind": rtype, "text": text, "ts": ts})

            elif btype == "thinking" and include_thinking:
                text = (b.get("thinking") or "").strip()
                if text:
                    events.append({"kind": "thinking", "text": text, "ts": ts})

            elif btype == "tool_use":
                name = b.get("name", "tool")
                tin = b.get("input") or {}
                field = TOOL_KEY_FIELD.get(name)
                summary = tin.get(field) if field else None
                if not isinstance(summary, str):
                    summary = json.dumps(tin, indent=2)[:400]
                events.append({
                    "kind": "tool",
                    "name": name,
                    "desc": tin.get("description", ""),
                    "input": summary.strip(),
                    "output": truncate(results.get(b.get("id"), "").strip(), tool_limit),
                    "ts": ts,
                })

            elif btype == "image":
                events.append({"kind": "image", "ts": ts})

    return events


def fmt_time(ts):
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ts[:16]


def to_markdown(events, title):
    out = [f"# {title}", ""]
    for e in events:
        kind = e["kind"]
        if kind == "user":
            out += [f"## 🧑 User · {fmt_time(e['ts'])}", "", e["text"], ""]
        elif kind == "assistant":
            out += [f"### 🤖 Claude", "", e["text"], ""]
        elif kind == "thinking":
            out += ["<details><summary>💭 Thinking</summary>", "", e["text"], "", "</details>", ""]
        elif kind == "image":
            out += ["*[image attached]*", ""]
        elif kind == "tool":
            label = f"🔧 **{e['name']}**" + (f" — {e['desc']}" if e["desc"] else "")
            out += [label, "", "```", e["input"], "```", ""]
            if e["output"]:
                out += ["<details><summary>output</summary>", "", "```", e["output"], "```", "", "</details>", ""]
    return "\n".join(out)


CSS = """
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.5; color: #1a1a1a; max-width: 46em; margin: 2em auto; }
h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: .3em; }
h2 { font-size: 13pt; color: #0b5394; margin-top: 1.8em;
     border-top: 1px solid #ccc; padding-top: .8em; }
h3 { font-size: 11.5pt; color: #38761d; margin-top: 1.4em; }
pre { background: #f4f4f4; border: 1px solid #ddd; padding: .6em;
      font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt;
      white-space: pre-wrap; word-wrap: break-word; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt;
       background: #f4f4f4; padding: .1em .3em; }
.tool { color: #7f6000; font-weight: bold; margin-top: 1em; }
.out { color: #555; font-size: 9pt; margin-left: .4em; }
"""


def md_inline(text):
    """Minimal inline markdown -> HTML (escaped first)."""
    text = htmllib.escape(text)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)  # drop link targets
    return text


def body_to_html(text):
    """Render a message body, preserving fenced code blocks."""
    parts = re.split(r"```[a-zA-Z0-9_-]*\n(.*?)```", text, flags=re.S)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(f"<pre>{htmllib.escape(part)}</pre>")
        else:
            for para in part.split("\n\n"):
                para = para.strip()
                if para:
                    out.append(f"<p>{md_inline(para).replace(chr(10), '<br>')}</p>")
    return "\n".join(out)


def to_html(events, title):
    out = [f"<html><head><meta charset='utf-8'><title>{htmllib.escape(title)}</title>",
           f"<style>{CSS}</style></head><body>", f"<h1>{htmllib.escape(title)}</h1>"]
    for e in events:
        kind = e["kind"]
        if kind == "user":
            out.append(f"<h2>User &middot; {fmt_time(e['ts'])}</h2>")
            out.append(body_to_html(e["text"]))
        elif kind == "assistant":
            out.append("<h3>Claude</h3>")
            out.append(body_to_html(e["text"]))
        elif kind == "thinking":
            out.append("<h3>Thinking</h3>")
            out.append(body_to_html(e["text"]))
        elif kind == "image":
            out.append("<p><em>[image attached]</em></p>")
        elif kind == "tool":
            desc = f" &mdash; {htmllib.escape(e['desc'])}" if e["desc"] else ""
            out.append(f"<p class='tool'>{htmllib.escape(e['name'])}{desc}</p>")
            out.append(f"<pre>{htmllib.escape(e['input'])}</pre>")
            if e["output"]:
                out.append(f"<pre class='out'>{htmllib.escape(e['output'])}</pre>")
    out.append("</body></html>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", help="path to the .jsonl transcript")
    ap.add_argument("--latest", action="store_true",
                    help="use the most recently modified transcript for this project")
    ap.add_argument("-o", "--out", default="export/chat", help="output path without extension")
    ap.add_argument("--thinking", action="store_true", help="include Claude's reasoning blocks")
    ap.add_argument("--tool-limit", type=int, default=1500,
                    help="max characters of each tool output (0 = unlimited)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.latest or not args.session:
        proj = Path.home() / ".claude/projects"
        cand = sorted(proj.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not cand:
            raise SystemExit("no transcripts found under ~/.claude/projects")
        session = cand[-1]
    else:
        session = Path(args.session)

    events = parse(session, args.thinking, args.tool_limit)
    title = args.title or f"Claude Code session — {session.stem[:8]}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(to_markdown(events, title))
    out.with_suffix(".html").write_text(to_html(events, title))

    kinds = {}
    for e in events:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"source : {session}")
    print(f"events : {kinds}")
    print(f"wrote  : {out.with_suffix('.md')}")
    print(f"wrote  : {out.with_suffix('.html')}")


if __name__ == "__main__":
    main()
