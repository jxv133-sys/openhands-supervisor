"""Ideas: a plain chat with the local model for brainstorming what to build.

Deliberately small. No sandbox and no code execution: it runs on the
supervisor host. A few read-only tools make it useful:

- list_projects / read_project / read_file: look at what you already have
- fetch_url: read a public web page (never LAN/loopback addresses)
- propose_project: hand a settled idea to the "New project" goal assistant

Chats are saved as JSON in <projects>/.ideas/. The model is called through
Ollama's /api/chat with streaming and native tool calling; `chat_turn` yields
small events (text, thinking, tool calls) that the web server streams to the
page as NDJSON.
"""

from __future__ import annotations

import html
import ipaddress
import json
import re
import secrets
import socket
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

from .loopguard import RepetitionDetector

IDEAS_DIR = ".ideas"
MAX_TOOL_ROUNDS = 6
TOOL_RESULT_CHARS = 8000
FETCH_BYTES = 400_000

SYSTEM_PROMPT = """You are a brainstorming partner for a developer who builds Rust and Python projects with a
local coding agent (OpenHands, kept on track by a supervisor that checks every milestone with
real tests). Help them find and sharpen ideas worth building: suggest concrete projects, weigh
scope (a local model does best with small, testable steps), point out risks, and build on what
they already have.

Be concise and concrete. Use short paragraphs or lists; no filler.

Tools: look at their existing projects when it helps (to build on them or avoid duplicates);
fetch a web page when you need facts like a crate's or library's docs. When the user settles on
an idea, call propose_project with a short folder name and a 2-5 sentence description of what
to build (language, how it's used, main features) so they can start it with one click."""

TOOLS = [
    {"type": "function", "function": {
        "name": "list_projects",
        "description": "List the user's existing projects with their goal's first line and progress.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_project",
        "description": "Read one project's goal, milestones (with status and check command) and final review result.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from a project (e.g. src/main.rs or README.md). Use list_files first if unsure.",
        "parameters": {"type": "object", "properties": {"project": {"type": "string"}, "path": {"type": "string"}},
                       "required": ["project", "path"]}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List the files of a project.",
        "parameters": {"type": "object", "properties": {"project": {"type": "string"}}, "required": ["project"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch a public web page (http/https) and return its text, e.g. a library's documentation.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "propose_project",
        "description": "Offer the user a one-click way to start a project from the idea you settled on together.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "short folder name, lowercase with hyphens, e.g. todo-rs"},
            "idea": {"type": "string", "description": "2-5 sentences: what to build, language, how it's used, main features"}},
            "required": ["name", "idea"]}}},
]


# --- storage ----------------------------------------------------------------

_ID_RE = re.compile(r"^[a-z0-9]{6,32}$")


class IdeaStore:
    def __init__(self, projects_root: Path):
        self.dir = Path(projects_root) / IDEAS_DIR

    def _path(self, chat_id: str) -> Path:
        if not _ID_RE.match(chat_id or ""):
            raise KeyError(chat_id)
        return self.dir / f"{chat_id}.json"

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("*.json")) if self.dir.is_dir() else []:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"id": data["id"], "title": data.get("title", ""), "updated": data.get("updated", 0)})
        return sorted(out, key=lambda c: c["updated"], reverse=True)

    def get(self, chat_id: str) -> dict:
        path = self._path(chat_id)
        if not path.is_file():
            raise KeyError(chat_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def new(self) -> dict:
        now = time.time()
        return {"id": secrets.token_hex(6), "title": "", "created": now, "updated": now, "messages": []}

    def save(self, chat: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        chat["updated"] = time.time()
        path = self._path(chat["id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(chat, indent=1), encoding="utf-8")
        tmp.replace(path)

    def delete(self, chat_id: str) -> None:
        path = self._path(chat_id)
        if not path.is_file():
            raise KeyError(chat_id)
        path.unlink()


# --- tools ------------------------------------------------------------------------

def _is_public(host: str) -> bool:
    """True only if every address the host resolves to is a public one."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            return False
    return bool(infos)


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr|pre|section|article)>", "\n", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def fetch_url(url: str, allow_private: bool = False, timeout: float = 15) -> str:
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return "error: only http(s) URLs can be fetched"
    if not allow_private and not _is_public(parsed.hostname):
        return "error: that address is on a private network or can't be resolved; only public web pages can be fetched"
    req = urllib.request.Request(url, headers={"User-Agent": "openhands-supervisor-ideas/1.0", "Accept": "text/html,text/plain,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # A redirect could point somewhere private: check where we ended up too.
            final = urllib.parse.urlparse(resp.geturl())
            if not allow_private and final.hostname and not _is_public(final.hostname):
                return "error: the page redirected to a private address"
            ctype = resp.headers.get("Content-Type", "")
            data = resp.read(FETCH_BYTES)
    except Exception as exc:
        return f"error: could not fetch the page: {exc}"
    text = data.decode("utf-8", errors="replace")
    if "html" in ctype or "<html" in text[:500].lower():
        text = html_to_text(text)
    return text[:TOOL_RESULT_CHARS] + (f"\n...(cut, {len(text)} chars in total)" if len(text) > TOOL_RESULT_CHARS else "")


class Tools:
    """Runs the model's tool calls against the supervisor's projects."""

    def __init__(self, app, allow_web: bool = True, allow_private: bool = False):
        self.app = app
        self.allow_web = allow_web
        self.allow_private = allow_private

    def run(self, name: str, args: dict) -> str:
        try:
            if name == "list_projects":
                rows = [f"- {p['name']}: {(p.get('goal') or '').splitlines()[0][:120] if p.get('goal') else '(no goal)'} "
                        f"({p.get('done', 0)}/{p.get('total', 0)} milestones done)" for p in self.app.state()["projects"]]
                return "\n".join(rows) or "(no projects yet)"
            if name == "read_project":
                data = self.app.project(str(args.get("name", "")))
                c = data["contract"]
                lines = [f"Goal:\n{c['goal']}", "Milestones:"]
                lines += [f"{i}. [{m['status']}] {m['title']} (check: {m['check']})" for i, m in enumerate(c["milestones"], 1)]
                review = data.get("review")
                if review:
                    reqs = review.get("requirements") or []
                    lines.append(f"Final review: {review.get('verdict')}, "
                                 f"{sum(r.get('passed') is True for r in reqs)}/{len(reqs)} requirements pass")
                return "\n".join(lines)
            if name == "list_files":
                files = self.app.files(str(args.get("project", "")))["files"]
                return "\n".join(f["path"] for f in files[:300]) or "(no files)"
            if name == "read_file":
                data = self.app.file(str(args.get("project", "")), str(args.get("path", "")))
                if data.get("binary"):
                    return "(binary file)"
                text = data["text"]
                return text[:TOOL_RESULT_CHARS] + ("\n...(cut)" if len(text) > TOOL_RESULT_CHARS else "")
            if name == "fetch_url":
                if not self.allow_web:
                    return "error: web access is turned off (chat_web = false)"
                return fetch_url(str(args.get("url", "")), allow_private=self.allow_private)
            if name == "propose_project":
                return "Shown to the user as a 'Start this project' button."
        except Exception as exc:  # unknown project/file etc.: tell the model, don't crash the chat
            return f"error: {exc}"
        return f"error: unknown tool {name!r}"


def tool_summary(name: str, args: dict) -> str:
    if name == "read_project":
        return f"read project {args.get('name', '')}"
    if name == "list_files":
        return f"list files of {args.get('project', '')}"
    if name == "read_file":
        return f"read {args.get('project', '')}/{args.get('path', '')}"
    if name == "fetch_url":
        return f"fetch {args.get('url', '')}"
    if name == "propose_project":
        return f"propose project {args.get('name', '')}"
    return name.replace("_", " ")


# --- model ------------------------------------------------------------------------

class ThinkSplitter:
    """Separates <think>...</think> from the answer when a model puts its
    reasoning in the content stream (older Ollama / no `think` option)."""

    def __init__(self):
        self.inside = False
        self.buf = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        self.buf += text
        out: list[tuple[str, str]] = []
        while self.buf:
            tag = "</think>" if self.inside else "<think>"
            i = self.buf.find(tag)
            if i >= 0:
                if i:
                    out.append(("thinking" if self.inside else "text", self.buf[:i]))
                self.buf = self.buf[i + len(tag):]
                self.inside = not self.inside
                continue
            # keep a possible partial tag at the end for the next chunk
            keep = next((k for k in range(min(len(tag) - 1, len(self.buf)), 0, -1) if tag.startswith(self.buf[-k:])), 0)
            emit, self.buf = (self.buf[:-keep], self.buf[-keep:]) if keep else (self.buf, "")
            if emit:
                out.append(("thinking" if self.inside else "text", emit))
            break
        return out


def ollama_stream(url: str, payload: dict, timeout: float) -> Iterator[dict]:
    """POST /api/chat with stream=true; yields each NDJSON chunk. Closing the
    generator closes the connection, which makes Ollama stop generating."""
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            line = line.strip()
            if line:
                yield json.loads(line)


def _for_model(messages: list[dict], budget_chars: int) -> list[dict]:
    """The stored history in Ollama's format, oldest messages dropped to fit."""
    clean = []
    for m in messages:
        msg = {"role": m["role"], "content": m.get("content", "")}
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        if m["role"] == "tool":
            msg["tool_name"] = m.get("tool_name", "")
        clean.append(msg)
    total, keep = 0, []
    for msg in reversed(clean):
        total += len(msg["content"]) + len(json.dumps(msg.get("tool_calls", "")))
        if total > budget_chars and keep:
            break
        keep.append(msg)
    keep.reverse()
    while keep and keep[0]["role"] == "tool":  # never start with an orphaned tool result
        keep.pop(0)
    return keep


def chat_turn(chat: dict, user_text: str, settings, tools: Tools,
              stream: Callable[[str, dict, float], Iterator[dict]] = ollama_stream,
              busy: bool = False) -> Iterator[dict]:
    """Add the user's message, run the model (with tool rounds), append the
    replies to chat["messages"], and yield events for the page:
      {"type": "thinking"|"text", "text"}           streamed pieces
      {"type": "tool", "name", "args", "summary"}   a tool call starting
      {"type": "tool_result", "name", "text"}       its (shortened) result
      {"type": "propose", "name", "idea"}           a settled idea
      {"type": "notice", "text"} / {"type": "error", "text"}
    The caller saves the chat afterwards, also when the page disconnects."""
    chat["messages"].append({"role": "user", "content": user_text, "ts": time.time()})
    if not chat.get("title"):
        chat["title"] = re.sub(r"\s+", " ", user_text).strip()[:70]
    if busy:
        yield {"type": "notice", "text": "A project run is using the model too, so replies may be slower (one GPU)."}
    options = {"temperature": 0.7}
    for key in ("top_p", "top_k"):
        if getattr(settings, key, None) is not None:
            options[key] = getattr(settings, key)
    if getattr(settings, "presence_penalty", 0):
        options["presence_penalty"] = settings.presence_penalty
    budget = int(settings.max_input_tokens * 3 * 0.8)

    for _round in range(MAX_TOOL_ROUNDS + 1):
        payload = {
            "model": settings.model, "stream": True, "options": options,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + _for_model(chat["messages"], budget),
            "tools": TOOLS if _round < MAX_TOOL_ROUNDS else [],  # last round: answer, no more tools
        }
        if getattr(settings, "chat_think", None) is not None:
            payload["think"] = settings.chat_think
        text, thinking, calls = "", "", []
        splitter, loop = ThinkSplitter(), RepetitionDetector()
        started = time.monotonic()
        try:
            gen = stream(settings.ollama_url, payload, settings.llm_timeout)
            try:
                for chunk in gen:
                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"])
                    msg = chunk.get("message") or {}
                    pieces = []
                    if msg.get("thinking"):
                        pieces.append(("thinking", msg["thinking"]))
                    if msg.get("content"):
                        pieces += splitter.feed(msg["content"])
                    for kind, piece in pieces:
                        if kind == "thinking":
                            thinking += piece
                        else:
                            text += piece
                        loop.feed(piece)
                        yield {"type": kind, "text": piece}
                    calls += msg.get("tool_calls") or []
                    if len(text) + len(thinking) > 1200 and loop.check():
                        yield {"type": "notice", "text": "The model started repeating itself, so the reply was cut off."}
                        break
            finally:
                gen.close()  # stops Ollama generating if we left early
        except GeneratorExit:  # the page pressed Stop / went away: keep what was written
            if text or thinking:
                chat["messages"].append({"role": "assistant", "content": text, "thinking": thinking,
                                         "stopped": True, "ts": time.time()})
            raise
        except Exception as exc:
            yield {"type": "error", "text": f"The model call failed: {exc}"}
            if text or thinking:
                chat["messages"].append({"role": "assistant", "content": text, "thinking": thinking, "ts": time.time()})
            return
        reply = {"role": "assistant", "content": text, "ts": time.time(), "seconds": round(time.monotonic() - started, 1)}
        if thinking:
            reply["thinking"] = thinking
        if calls:
            reply["tool_calls"] = calls
        chat["messages"].append(reply)
        if not calls:
            return
        for call in calls:
            fn = call.get("function") or {}
            name, args = fn.get("name", ""), fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            yield {"type": "tool", "name": name, "args": args, "summary": tool_summary(name, args)}
            result = tools.run(name, args)
            if name == "propose_project" and not result.startswith("error"):
                yield {"type": "propose", "name": str(args.get("name", "")), "idea": str(args.get("idea", ""))}
            chat["messages"].append({"role": "tool", "tool_name": name, "content": result, "ts": time.time()})
            yield {"type": "tool_result", "name": name, "text": result[:1500]}
    yield {"type": "notice", "text": "Stopped after several tool calls in a row."}
