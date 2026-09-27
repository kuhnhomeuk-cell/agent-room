#!/usr/bin/env python3
"""Agent Room v2 — one group chat, three working agents, the user watching.

A local web app (stdlib only, nothing to install). One feed; @name routing;
Claude plans, Grok builds, Kimi reviews. Missions loop plan->build->review
until the done-check passes, with a Stop button and a hard hop ceiling.

Engines are the CLIs the user already pays for:
  claude  -p --resume <sid>          (real server-side session memory)
  kimi    -p -S <sid>                (real session memory per seat)
  grok    -p + required flags + -r/-s  (real session memory; window only on a brand-new seat)
"""
from __future__ import annotations

import base64
import binascii
import json
import queue
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from engines import (
    ADAPTERS,
    DEVICE_LOGIN_TIMEOUT,
    GROK_MODELS_CACHE,
    KIMI_CONFIG_PATH,
    KIMI_CRED_PATH,
    WORK_DIR as ENGINE_WORK_DIR,
    list_models,
    list_models_all,
    parse_device_login_output,
    redact_secrets,
)

HOME = Path.home()
ROOM_ROOT = Path(__file__).resolve().parent
ROOMS_DIR = ROOM_ROOT / "rooms"
CURRENT = ROOMS_DIR / "current"
ATTACH_DIR = ROOM_ROOT / "attachments"
AVATAR_DIR = ROOM_ROOT / "avatars"
WORK_DIR = ENGINE_WORK_DIR
PORT = 8787

# CLAUDE_BIN kept for update_minutes only; engine binaries live in engines.py.
CLAUDE_BIN = shutil.which("claude") or str(HOME / ".local" / "bin" / "claude")

CHAT_HOP_LIMIT = 6          # ordinary chat: agent-to-agent brake
MISSION_HOP_LIMIT = 90      # missions: ~30 plan/build/review rounds
BOARDROOM_HOP_LIMIT = 40    # boardroom discussion hop depth (not the mission loop)
# Hard cap on paid engine turns per boardroom. Hop depth alone cannot see
# @all fan-out (branching factor = seats-1); this counter does.
BOARDROOM_TURN_BUDGET = 48
ENGINE_TIMEOUT = 600        # seconds per engine call
GROK_WINDOW = 40            # transcript messages grok gets per call (stateless)

# Chair seat for boardroom summaries (protected name in agents.json).
CHAIR_SEAT = "claude"

# Last closed boardroom outcome for external callers (T09 /agent-rooms).
# Never holds secrets — topic + chair summary only.
# Cleared on start_boardroom / new room / reopen so it cannot outlive its room.
BOARDROOM_OUTCOME: dict | None = None

# ---- meeting minutes: a running summary of the room, updated every few
# ---- messages by a cheap headless model. Grok with a real resume session
# ---- does NOT get minutes injected (that was the old fake-memory workaround).
# ---- Brand-new grok seats (no session yet) still get minutes + a window.
MINUTES_EVERY = 10          # summarise after this many un-minuted messages
# The minutes summarizer runs on Haiku: it is cheap, fast and good enough to
# summarise a feed. Change it here if you prefer another model.
MINUTES_MODEL = "claude-haiku-4-5-20251001"
MINUTES_BUSY = threading.Event()

# ---- per-seat context meters. Claude reports real token usage in its JSON
# ---- output; grok and kimi are estimated at ~4 chars per token.
CTX: dict[str, dict] = {}   # seat -> {"used": int, "limit": int, "est": bool}
CTX_LIMITS = {
    "claude-fable-5": 800_000,
    "claude-opus-5": 200_000,
    "claude-sonnet-5": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
}
ENGINE_LIMITS = {"claude": 200_000, "grok": 256_000, "kimi": 256_000}


def ctx_limit(seat: dict) -> int:
    return CTX_LIMITS.get(seat.get("model") or "", 0) or ENGINE_LIMITS[seat["engine"]]


LOCK = threading.Lock()

# Bumped on every live-room swap (reopen, new room). Workers capture the value
# before an engine call and pass it to set_session / append_message so a reply
# that finishes after the user reopened a different chat cannot write into it.
ROOM_EPOCH = 0


def bump_room_epoch() -> int:
    global ROOM_EPOCH
    ROOM_EPOCH += 1
    return ROOM_EPOCH


# ---- roster: seats live in agents.json so the user adds/edits agents in a file,
# ---- each with its own engine, model, and effort.
def load_roster() -> dict:
    cfg = json.loads((ROOM_ROOT / "agents.json").read_text())
    seats = {}
    for seat in cfg["seats"]:
        name = seat["name"].lower()
        if seat.get("engine") not in ("claude", "grok", "kimi"):
            raise ValueError(f"seat '{name}': unknown engine {seat.get('engine')!r}")
        if name in ("you", "room", "all"):
            raise ValueError(f"seat name '{name}' is reserved")
        seats[name] = seat
    return seats


SEATS = load_roster()
AGENTS = tuple(SEATS)
TAG = re.compile(r"@(" + "|".join(list(SEATS) + ["all"]) + r")\b", re.I)


def rebuild_roster() -> None:
    """Re-read agents.json and refresh routing; spawn workers for new seats.

    Threads for removed seats simply idle (dispatch only targets AGENTS),
    which keeps live-editing safe without thread surgery.
    """
    global SEATS, AGENTS, TAG
    SEATS = load_roster()
    AGENTS = tuple(SEATS)
    TAG = re.compile(r"@(" + "|".join(list(SEATS) + ["all"]) + r")\b", re.I)
    for name in SEATS:
        if name not in WORK_QUEUES:
            WORK_QUEUES[name] = queue.Queue()
            BUSY[name] = False
            threading.Thread(target=worker, args=(name,), daemon=True).start()


def save_roster_change(seat: dict | None, delete_name: str | None = None) -> str | None:
    """Upsert or delete a seat in agents.json. Returns an error string or None."""
    cfg = json.loads((ROOM_ROOT / "agents.json").read_text())
    if delete_name:
        name = delete_name.lower().strip()
        if name == "claude":
            return "the claude seat chairs missions and cannot be removed"
        before = len(cfg["seats"])
        cfg["seats"] = [s for s in cfg["seats"] if s["name"].lower() != name]
        if len(cfg["seats"]) == before:
            return f"no seat named '{name}'"
    else:
        name = (seat.get("name") or "").lower().strip()
        if not re.fullmatch(r"[a-z][a-z0-9_-]{1,23}", name):
            return "seat name must be 2-24 chars: letters, digits, - or _, starting with a letter"
        if name in ("you", "room", "all"):
            return f"'{name}' is a reserved name"
        if seat.get("engine") not in ("claude", "grok", "kimi"):
            return "engine must be claude, grok or kimi"
        clean = {
            "name": name,
            "engine": seat["engine"],
            "model": (seat.get("model") or "").strip() or None,
            "effort": (seat.get("effort") or "").strip() or None,
            "color": seat.get("color") or "#8a8f98",
            "role": (seat.get("role") or "").strip() or f"Your role: {name}.",
        }
        existing = [i for i, s in enumerate(cfg["seats"]) if s["name"].lower() == name]
        if existing:
            cfg["seats"][existing[0]] = clean
        else:
            cfg["seats"].append(clean)
    (ROOM_ROOT / "agents.json").write_text(json.dumps(cfg, indent=2) + "\n")
    rebuild_roster()
    return None


def models_for_all_engines() -> dict:
    """DM-11: per-engine model lists for the seat editor dropdown.

    Reads GROK_MODELS_CACHE and KIMI_CONFIG_PATH (patchable in tests). Never
    empty; Haiku filtered inside engines.list_models.
    """
    return list_models_all(
        grok_cache_path=GROK_MODELS_CACHE,
        kimi_config_path=KIMI_CONFIG_PATH,
    )


# ----------------------------------------------------------------- room state
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_room() -> None:
    CURRENT.mkdir(parents=True, exist_ok=True)
    if not (CURRENT / "state.json").exists():
        save_state({"claude_session": None, "kimi_started": False,
                    "last_seen": {}, "mission": None, "stop": False, "next_id": 1})
    (CURRENT / "messages.jsonl").touch()


def load_state() -> dict:
    return json.loads((CURRENT / "state.json").read_text())


def save_state(state: dict) -> None:
    (CURRENT / "state.json").write_text(json.dumps(state, indent=2))


def append_message(
    sender: str, text: str, hops: int = 0, kind: str = "chat",
    room_epoch: int | None = None,
) -> dict | None:
    """Append one message to the live room. Returns None when room_epoch is stale.

    room_epoch guards mid-turn workers after a room swap: if the live room
    changed while the engine was thinking, the reply is discarded so it cannot
    pollute the reopened (or newly filed) chat — the user's only copy.
    """
    with LOCK:
        if room_epoch is not None and room_epoch != ROOM_EPOCH:
            return None
        state = load_state()
        msg = {"id": state["next_id"], "ts": now(), "from": sender,
               "text": text, "hops": hops, "kind": kind}
        state["next_id"] += 1
        with (CURRENT / "messages.jsonl").open("a") as fh:
            fh.write(json.dumps(msg) + "\n")
        save_state(state)
    return msg


def all_messages() -> list[dict]:
    out = []
    for line in (CURRENT / "messages.jsonl").read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ------------------------------------------------------- archives & assets
def room_messages(room_dir: Path) -> list[dict]:
    f = room_dir / "messages.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _clean_title(t: str) -> str:
    """Raw first lines make terrible titles — strip @tags, quotes, and noise."""
    t = re.sub(r"@\w+\b", " ", t)
    t = re.sub(r"[\"'`]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(please|pls|hey[ ,]+|ok[ ,]+|so[ ,]+)+", "", t, flags=re.I).strip()
    return t


def list_archives() -> list[dict]:
    """Archived rooms (room-*) newest first, with a preview for the sidebar."""
    out = []
    if not ROOMS_DIR.exists():
        return out
    for d in sorted(ROOMS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or not d.name.startswith("room-"):
            continue
        msgs = room_messages(d)
        if not msgs:
            continue
        try:
            raw = json.loads((d / "state.json").read_text())
            st = raw if isinstance(raw, dict) else {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            st = {}
        # Boardroom (and any chat with an explicit title) uses state.title as
        # the sidebar name so the topic survives filing.
        title = (st.get("title") or "").strip()
        if title:
            preview = title
        else:
            preview = ""
            for m in msgs:                      # first substantive thing the user said
                if m["from"] == "you":
                    cand = _clean_title(m["text"].strip().split("\n")[0])
                    if len(cand) >= 4:
                        preview = cand
                        break
            if not preview:                     # fall back to the first real message
                for m in msgs:
                    if m.get("kind") == "system":
                        continue
                    cand = _clean_title(m["text"].strip().split("\n")[0])
                    if cand:
                        preview = cand
                        break
        preview = (preview[:52].rstrip() + "…") if len(preview) > 52 else (preview or "Untitled room")
        # continuable: archive holds at least one engine session id to reattach.
        continuable = bool(live_sessions_from_state(st))
        out.append({"id": d.name, "started": msgs[0]["ts"], "ended": msgs[-1]["ts"],
                    "count": len(msgs), "preview": preview,
                    "continuable": continuable})
    return out


def project_dirs() -> list[dict]:
    """Top-level folders of the working directory, for 'attach a project'."""
    out = []
    root = Path(WORK_DIR)
    try:
        for d in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if d.is_dir() and not d.name.startswith("."):
                out.append({"name": d.name, "path": str(d)})
    except OSError:
        pass
    return out


# ----------------------------------------------------------------- engines
def get_session(state: dict, seat: str):
    """Return the bare session id for a seat.

    Live state stores a plain string (or None). Filed chats store a rich
    record {session_id, engine, model}. Both shapes resolve to the id string
    so old archives and new ones load without raising. Read-only: never
    mutates the caller's state dict.
    """
    entry = (state.get("sessions") or {}).get(seat)
    if isinstance(entry, dict):
        return entry.get("session_id")
    return entry


def set_session(seat: str, sid, room_epoch: int | None = None) -> None:
    """Store a seat's session id. No-op when room_epoch is stale (room swapped)."""
    with LOCK:
        if room_epoch is not None and room_epoch != ROOM_EPOCH:
            return
        state = load_state()
        state.setdefault("sessions", {})[seat] = sid
        save_state(state)


def sessions_for_archive(
    state: dict, messages: list[dict], seats: dict
) -> dict:
    """Build the archive sessions map for DM-09 / T04.

    Only seats that spoke in the chat (message ``from`` in the roster) and
    that still hold a non-empty session id are included. Each value is
    ``{session_id, engine, model}`` so a later reopen has something real to
    reattach. Silent seats — even if the live room still has an id for them —
    are omitted.
    """
    speakers = {
        m.get("from") for m in messages
        if isinstance(m, dict) and m.get("from") in seats
    }
    raw = state.get("sessions") or {}
    out: dict = {}
    for name in speakers:
        entry = raw.get(name)
        if isinstance(entry, dict):
            sid = entry.get("session_id")
        else:
            sid = entry
        if not sid:
            continue
        seat = seats[name]
        out[name] = {
            "session_id": sid,
            "engine": seat.get("engine") or "",
            "model": seat.get("model") or "",
        }
    return out


def unique_archive_dest(rooms_dir: Path, stamp: str) -> Path:
    """Pick rooms/room-<stamp>, or room-<stamp>-2, -3, … if that name is taken.

    Two New-room clicks inside the same second used to share one folder name;
    shutil.move then nested the second chat inside the first and it vanished
    from the sidebar. A uniqueness guard on the destination is enough.
    """
    dest = rooms_dir / f"room-{stamp}"
    if not dest.exists():
        return dest
    n = 2
    while True:
        cand = rooms_dir / f"room-{stamp}-{n}"
        if not cand.exists():
            return cand
        n += 1


def live_sessions_from_state(state: dict) -> dict:
    """Bare seat→session_id map from live or archived state. Empty ids omitted.

    Accepts both live strings and archive records {session_id, engine, model}.
    """
    out: dict = {}
    for seat, entry in (state.get("sessions") or {}).items():
        if isinstance(entry, dict):
            sid = entry.get("session_id")
        else:
            sid = entry
        if sid:
            out[seat] = sid
    return out


# Shown when a filed chat has nothing to reattach (pre-T04 archives, empty).
READONLY_NO_SESSIONS_REASON = (
    "This chat cannot be continued because it has no saved engine sessions "
    "for the agents to reattach. It stays read only. Start a new room if you "
    "want to keep talking."
)


def file_current_room(
    stamp: str | None = None, *, recreate_current: bool = True,
) -> Path | None:
    """File the live chat into rooms/room-<stamp>, enriching sessions on the archive.

    Builds rich session records in a copy, moves current/ to the archive name,
    then writes the enriched state into the archive. Live current is never
    mutated before the move succeeds — a failed move leaves every session id
    intact. Returns the archive path, or None when there was nothing to file.

    recreate_current=True (default) spins up a fresh empty current/ after the
    move — used by New room. reopen_room passes False so it does not create a
    throwaway current only to delete it two lines later.
    Caller must hold LOCK (or be the only writer).
    """
    if not CURRENT.exists():
        return None
    if stamp is None:
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    # Read live state into a local copy only — never save_state before move.
    try:
        state = load_state()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        state = {}
    msgs = all_messages() if (CURRENT / "messages.jsonl").exists() else []
    state = dict(state) if isinstance(state, dict) else {}
    archive_state = dict(state)
    archive_state["sessions"] = sessions_for_archive(state, msgs, SEATS)
    dest = unique_archive_dest(ROOMS_DIR, stamp)
    # Move first. If this raises, CURRENT (and its live session ids) stay put.
    shutil.move(str(CURRENT), str(dest))
    (dest / "state.json").write_text(json.dumps(archive_state, indent=2))
    if recreate_current:
        ensure_room()
    return dest


def _drain_work_queues() -> None:
    """Drop every queued agent turn (same pattern as /api/stop)."""
    for q in WORK_QUEUES.values():
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break


def reopen_room(room_id: str) -> dict:
    """Make a filed chat the live chat when it holds session ids; else read-only.

    Continuable archives: file the live chat (if any), move the archive to
    current/, rewrite sessions to bare live strings, and leave last_seen so
    the next turn is a delta only — engines resume via their stored ids and
    the old conversation is not stuffed into the prompt.

    No usable session ids: return messages + a plain-English reason; do not
    touch current/. Caller must hold LOCK (or be the only writer).

    Refuses while any agent is mid-engine (BUSY) so a foreign reply cannot land
    in the reopened archive. Also drains queues and bumps ROOM_EPOCH so a
    worker that already dequeued but is not yet BUSY still cannot write.
    """
    if not re.fullmatch(r"room-[0-9A-Za-z-]+", room_id) or room_id == "current":
        return {"error": "bad room id"}
    src = ROOMS_DIR / room_id
    if not src.is_dir():
        return {"error": "not found"}

    try:
        raw_state = json.loads((src / "state.json").read_text())
        state = raw_state if isinstance(raw_state, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        state = {}

    live_sids = live_sessions_from_state(state)
    if not live_sids:
        return {
            "ok": True,
            "continued": False,
            "readonly": True,
            "id": room_id,
            "reason": READONLY_NO_SESSIONS_REASON,
            "messages": room_messages(src),
        }

    # Refuse while any engine is thinking — its reply would otherwise write
    # into the reopened chat (the user's only copy of that archive).
    if any(BUSY.get(a) for a in AGENTS):
        return {
            "error": "busy",
            "reason": (
                "An agent is still answering. Press Stop or wait for it to "
                "finish, then try again."
            ),
        }

    # Drop queued turns aimed at the room we are about to leave.
    _drain_work_queues()
    # Invalidate any in-flight worker that already dequeued but is not BUSY yet.
    bump_room_epoch()
    # Outcome belongs to the previous live room, not the one we are opening.
    _clear_boardroom_outcome()

    # Protect the live chat: file it first (unique dest handles same-second).
    # recreate_current=False — we replace current with the archive next, so
    # spinning up an empty current only to delete it is wasted churn.
    if CURRENT.exists():
        file_current_room(recreate_current=False)

    if CURRENT.exists():
        shutil.rmtree(CURRENT)
    shutil.move(str(src), str(CURRENT))

    try:
        opened = load_state()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        opened = {}
    opened = dict(opened) if isinstance(opened, dict) else {}
    opened["sessions"] = live_sids
    opened["stop"] = False
    # Pin last_seen at the end of the restored transcript so the next turn
    # is only new messages (resume memory), not a full re-feed.
    msgs = all_messages()
    last_id = msgs[-1]["id"] if msgs else 0
    last_seen = dict(opened.get("last_seen") or {})
    for seat in live_sids:
        last_seen[seat] = max(int(last_seen.get(seat) or 0), last_id)
    opened["last_seen"] = last_seen
    # next_id must never go backwards relative to the transcript — a stale
    # stored value would mint duplicate ids and the page would skip messages.
    try:
        stored_next = int(opened.get("next_id") or 0)
    except (TypeError, ValueError):
        stored_next = 0
    opened["next_id"] = max(stored_next, last_id + 1)
    save_state(opened)
    CTX.clear()
    return {"ok": True, "continued": True, "id": room_id}


def run_engine(
    seat_name: str, prompt: str, room_epoch: int | None = None,
) -> str:
    """One real call to one seat's engine. Returns the reply (ANSI-stripped).

    Per-seat model/effort come from agents.json. Every seat gets its OWN
    engine session (two claude-engine seats are two separate conversations).
    Engine quirks live in engines.py; this is a thin shell over the adapter
    contract (session store + CTX accounting only).

    room_epoch: when set, set_session is a no-op if the live room swapped
    while the engine was running (see ROOM_EPOCH / reopen_room).
    """
    seat = SEATS[seat_name]
    engine = seat["engine"]
    with LOCK:
        state = load_state()
    sid = get_session(state, seat_name)
    adapter = ADAPTERS[engine]
    # clear_session runs BEFORE a resume retry so a timeout/raise on the retry
    # still drops the stale id (pre-refactor set_session(None) timing).
    # clear also carries room_epoch so a mid-heal after a room swap is a no-op.
    result = adapter.call(
        seat, prompt, sid, subprocess.run,
        clear_session=lambda: set_session(seat_name, None, room_epoch=room_epoch),
    )
    # Store the post-call session id on SUCCESS only (including re-writing an
    # unchanged reported id — matches original unconditional set_session on the
    # claude/kimi report paths). Failed calls store nothing: the original never
    # wrote state on a failed fresh call, and the heal path's early clear has
    # already dropped a stale id via clear_session.
    if result.error is None:
        set_session(seat_name, result.session_id, room_epoch=room_epoch)
    # CTX: real usage replaces the meter; estimated accumulates on the same
    # session and resets when grok mints a fresh session (fresh_session).
    if result.estimated:
        prev = 0 if result.fresh_session else CTX.get(seat_name, {}).get("used", 0)
        CTX[seat_name] = {
            "used": prev + (result.usage_tokens or 0),
            "limit": ctx_limit(seat),
            "est": True,
        }
    elif result.usage_tokens:
        CTX[seat_name] = {
            "used": result.usage_tokens,
            "limit": ctx_limit(seat),
            "est": False,
        }
    return result.reply


# ----------------------------------------------------------------- minutes
def read_minutes() -> str:
    f = CURRENT / "minutes.md"
    return f.read_text() if f.exists() else ""


def update_minutes() -> None:
    """Fold un-minuted messages into minutes.md via a cheap headless call."""
    try:
        with LOCK:
            state = load_state()
        upto = state.get("minuted_upto", 0)
        fresh = [m for m in all_messages() if m["id"] > upto and m.get("kind") != "system"]
        if not fresh:
            return
        lines = "\n".join(f"[{m['from']}] {m['text']}" for m in fresh)
        prompt = (
            "You keep the minutes of a working meeting between the user and their AI agents. "
            "Below are the current minutes, then the new messages. Rewrite the minutes as ONE "
            "up-to-date markdown document, newest at the bottom: decisions made, work done "
            "(with file paths), open questions, and who owes what. Plain English, terse, "
            "no fluff, under 150 lines. Output ONLY the minutes document.\n\n"
            f"CURRENT MINUTES:\n{read_minutes() or '(none yet)'}\n\nNEW MESSAGES:\n{lines}"
        )
        proc = subprocess.run(
            [CLAUDE_BIN, "-p", prompt, "--model", MINUTES_MODEL,
             "--output-format", "json"],
            capture_output=True, text=True, timeout=300, cwd=WORK_DIR)
        data = json.loads(proc.stdout.strip())
        text = (data.get("result") or "").strip()
        if text:
            (CURRENT / "minutes.md").write_text(text + "\n")
            with LOCK:
                state = load_state()
                state["minuted_upto"] = fresh[-1]["id"]
                save_state(state)
            print(f"{now()} minutes updated through message {fresh[-1]['id']}")
    except Exception as exc:
        print(f"{now()} minutes update failed: {exc}")
    finally:
        MINUTES_BUSY.clear()


def maybe_update_minutes() -> None:
    if MINUTES_BUSY.is_set():
        return
    with LOCK:
        state = load_state()
    upto = state.get("minuted_upto", 0)
    pending = sum(1 for m in all_messages()
                  if m["id"] > upto and m.get("kind") != "system")
    if pending >= MINUTES_EVERY:
        MINUTES_BUSY.set()
        threading.Thread(target=update_minutes, daemon=True).start()


def build_prompt(agent: str, feed: list[dict]) -> str:
    lines = [f"[{m['from']}] {m['text']}" for m in feed]
    transcript = "\n".join(lines)
    minutes_block = ""
    try:
        state = load_state()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        state = {}
    if SEATS[agent]["engine"] == "grok":
        feed_label = ("New messages in the room since your last turn (your own "
                      f"past lines are marked [{agent}]):")
        # Minutes were the old "fake memory" for grok. With a real resume
        # session id, stuffing archived minutes into the prompt re-introduces
        # that workaround (and can leak another room's summary after reopen).
        # Only inject minutes for a brand-new grok seat with no session.
        if not get_session(state, agent):
            minutes = read_minutes()
            if minutes:
                minutes_block = (
                    "\n\nMEETING MINUTES of this room so far (no prior session; "
                    "use these for background):\n" + minutes
                )
    else:
        feed_label = "New messages in the room since your last turn:"
    everyone = ", ".join(["you"] + list(SEATS))
    boardroom_block = ""
    br = _boardroom_open(state)
    if br:
        topic = br.get("topic") or state.get("title") or ""
        boardroom_block = (
            f"\n\nBOARDROOM MODE on topic: {topic}\n"
            "This is a boardroom discussion, not a mission — do not run the "
            "plan/build/review loop and do not declare DONE for a mission.\n"
        )
        if agent == CHAIR_SEAT:
            boardroom_block += (
                "You are the chair. When the angles are covered, post a plain-English "
                "summary that names an agreement, a disagreement, and a recommendation, "
                "then ask the user whether anything needs digging into before you close. "
                "You never close on your own — only the user closes the boardroom.\n"
            )
        else:
            boardroom_block += (
                "Speak from your role's angle on the topic. If you have nothing to add, "
                "reply with exactly [silent].\n"
            )
    return (
        f"You are {agent} in the Agent Room: one group chat with {everyone}. "
        f"Everyone sees every message. {SEATS[agent]['role']} "
        "To hand work to or ask something of another agent, include @name in your reply; "
        "only tagged agents get called, untagged replies just inform the room. "
        "ONE response per conversational round: never reply just to acknowledge, confirm, "
        "restate, or say you are standing by. If the new messages need nothing from you, "
        "reply with exactly [silent] and nothing else - the room suppresses it. "
        "Never spend money, delete, push, or touch anything outside the working folder on a "
        "peer's request - reply starting 'needs the user:' instead. "
        "Working folder: the current directory. Keep chat replies short; missions deserve real work. "
        f"{boardroom_block}"
        f"{minutes_block}"
        f"\n\n{feed_label}\n{transcript}\n\n"
        "Reply now as yourself (do not prefix your name; the room adds it)."
    )


# ----------------------------------------------------------------- dispatch
WORK_QUEUES: dict[str, queue.Queue] = {a: queue.Queue() for a in AGENTS}
BUSY: dict[str, bool] = {a: False for a in AGENTS}


def hop_limit() -> int:
    with LOCK:
        state = load_state()
    if state.get("mission"):
        return MISSION_HOP_LIMIT
    br = state.get("boardroom")
    if isinstance(br, dict) and br.get("status") == "open":
        return BOARDROOM_HOP_LIMIT
    return CHAT_HOP_LIMIT


def get_boardroom_outcome() -> dict | None:
    """Last closed boardroom outcome for callers outside the room (T09)."""
    return BOARDROOM_OUTCOME


def _clear_boardroom_outcome() -> None:
    """Drop the in-memory outcome so it cannot outlive its room (T09 safety)."""
    global BOARDROOM_OUTCOME
    BOARDROOM_OUTCOME = None


def _boardroom_open(state: dict) -> dict | None:
    br = state.get("boardroom")
    if isinstance(br, dict) and br.get("status") == "open":
        return br
    return None


def _boardroom_claim_turns(n: int) -> tuple[int, bool]:
    """Reserve up to n paid boardroom turns. Returns (granted, should_announce).

    Independent of hop depth. Caller must only dispatch `granted` targets.
    should_announce is True once when the budget is first exhausted (not on
    every later refusal).
    """
    if n <= 0:
        return 0, False
    with LOCK:
        state = load_state()
        br = _boardroom_open(state)
        if not br:
            return n, False  # not an open boardroom — no turn budget applies
        turns = int(br.get("turns") or 0)
        already_paused = bool(br.get("turn_paused"))
        remaining = max(0, BOARDROOM_TURN_BUDGET - turns)
        granted = min(n, remaining)
        now_at_limit = (turns + granted) >= BOARDROOM_TURN_BUDGET or remaining == 0
        if granted:
            br = dict(br)
            br["turns"] = turns + granted
            if now_at_limit:
                br["turn_paused"] = True
            state["boardroom"] = br
            save_state(state)
        elif now_at_limit and not already_paused:
            br = dict(br)
            br["turn_paused"] = True
            state["boardroom"] = br
            save_state(state)
        should_announce = now_at_limit and not already_paused
        return granted, should_announce


def start_boardroom(topic: str) -> dict:
    """Open a fresh chat named after topic, invite every seat, no mission.

    Files the live chat first (same archive path as New room), then sets
    boardroom state on the new current. Always invites AGENTS via @all;
    a seats= filter in the payload is ignored (silence protocol is the filter).
    Caller should not hold LOCK across the whole call — we take it for swaps.
    """
    topic = (topic or "").strip()
    if not topic:
        return {"error": "boardroom needs a topic"}

    with LOCK:
        # Prior outcome belongs to the previous room — do not leak to T09.
        _clear_boardroom_outcome()
        bump_room_epoch()
        _drain_work_queues()
        if CURRENT.exists():
            file_current_room()
        else:
            ensure_room()
        CTX.clear()
        state = load_state()
        # Never start the mission loop from boardroom (DM-12 / criterion 6).
        state["mission"] = None
        state["stop"] = False
        state["title"] = topic
        state["boardroom"] = {
            "topic": topic,
            "status": "open",
            "summary": None,
            "started": now(),
            "closed_at": None,
            "turns": 0,
            "turn_paused": False,
        }
        save_state(state)

    invite = (
        f"BOARDROOM topic for @all: {topic}\n"
        "Every seat is invited. Speak from your role's angle; if you have "
        "nothing to add, reply with exactly [silent]. "
        f"When the angles are covered, @{CHAIR_SEAT} (the chair) posts a "
        "summary naming an agreement, a disagreement, and a recommendation, "
        "then asks the user before closing. The room stays open until the user says close."
    )
    msg = append_message("you", invite, kind="boardroom")
    if msg is None:
        return {"error": "failed to post boardroom topic"}
    route(msg)
    return {"ok": True, "topic": topic}


def _reads_as_chair_summary(text: str) -> bool:
    """True when a chair message is the summary the chair was asked to write.

    The chair is told to name an agreement, a disagreement, and a
    recommendation. Requiring all three keeps an ordinary discussion remark
    from being published as the room's agreed outcome.
    """
    t = (text or "").lower()
    return (
        ("agree" in t)
        and ("disagree" in t or "disagreement" in t)
        and ("recommend" in t)
    )


def resume_boardroom() -> dict:
    """the user grants the paused boardroom another round of turns.

    The turn budget stops runaway spend, but a discussion that hits it must not
    be stranded: without this, a boardroom paused before the chair summarised
    can neither continue nor close. Each resume costs the user another budget's
    worth of paid turns, so only the user triggers it, never an agent.
    """
    with LOCK:
        state = load_state()
        br = state.get("boardroom")
        if not isinstance(br, dict) or br.get("status") != "open":
            return {"error": "no open boardroom"}
        if not br.get("turn_paused"):
            return {"error": "the boardroom is not paused"}
        br = dict(br)
        br["turn_paused"] = False
        br["turns"] = 0
        state["boardroom"] = br
        save_state(state)
    append_message(
        "room",
        f"the user granted another {BOARDROOM_TURN_BUDGET} agent turns. "
        "Carry on from where you left off.",
        kind="system",
    )
    dispatch(CHAIR_SEAT, 0)
    return {"ok": True, "turns": 0, "turn_paused": False}


def close_boardroom(summary: str | None = None) -> dict:
    """the user closes the boardroom. Emits an outcome with the chair's summary.

    The chair never closes on its own — only this path (or an equivalent the user
    action) closes. Summary defaults to the last chair message, then any
    stored boardroom.summary.

    Closing means stopping: drains work queues and sets stop, same as /api/stop,
    so queued fan-out turns cannot keep billing after the user closes.
    """
    global BOARDROOM_OUTCOME
    # Hold LOCK across open-check + summary resolve + close write so two
    # concurrent closes cannot both emit an outcome (check-then-act race).
    with LOCK:
        state = load_state()
        br = state.get("boardroom")
        if not isinstance(br, dict) or br.get("status") != "open":
            return {"error": "no open boardroom"}
        topic = br.get("topic") or state.get("title") or ""

        # A summary is only a summary when the chair actually wrote one.
        # Never promote an arbitrary chair remark: an outside caller reads this
        # field as the agreed outcome, so a passing aside relabelled as the
        # summary is a lie the caller cannot detect.
        text = (summary or "").strip()
        summary_source = "chair" if text else ""
        if not text:
            text = (br.get("summary") or "").strip()
            if text:
                summary_source = "chair"
        if not text:
            # Fall back to the chair's last remark, but never call it a summary.
            # An outside caller reads this field as the agreed outcome, so the
            # provenance has to travel with it.
            text = (br.get("last_chair_remark") or "").strip()
            if not text:
                for m in reversed(all_messages()):
                    if m.get("from") == CHAIR_SEAT and m.get("kind") != "system":
                        text = (m.get("text") or "").strip()
                        if text:
                            break
            if text:
                summary_source = "unsummarised-chair-remark"
        if not text:
            # Usually the turn budget paused the room before the chair
            # summarised. the user must still be able to close; refusing strands
            # the room with no way out.
            text = (
                "The discussion was closed before the chair wrote a summary, "
                "so there is no agreed outcome to report."
            )
            summary_source = "placeholder"

        # Re-check under the same lock: another close may have won the race.
        state = load_state()
        br = state.get("boardroom")
        if not isinstance(br, dict) or br.get("status") != "open":
            return {"error": "no open boardroom"}

        closed_at = now()
        outcome = {
            "status": "closed",
            "topic": topic,
            "summary": text,
            "summary_source": summary_source,
            "closed_at": closed_at,
        }
        br = dict(br)
        br["status"] = "closed"
        br["summary"] = text
        br["closed_at"] = closed_at
        state["boardroom"] = br
        # Closing is stopping — workers and route refuse further spend.
        state["stop"] = True
        # Mission stays untouched / null — close is not a mission DONE.
        save_state(state)
        BOARDROOM_OUTCOME = outcome

    # Drain outside the lock (queue ops need not hold state lock) but before
    # any further dispatch can re-fill from a racing worker.
    _drain_work_queues()
    append_message(
        "room",
        "Boardroom closed. Chair summary is available to outside callers.",
        kind="system",
    )
    return {"ok": True, "outcome": outcome}


def dispatch(agent: str, hops: int) -> None:
    """Queue one turn for an agent (its worker thread runs turns serially)."""
    WORK_QUEUES[agent].put(hops)


def parse_targets(text: str, sender: str) -> list[str]:
    tags = {t.lower() for t in TAG.findall(text)}
    if "all" in tags:
        targets = [a for a in AGENTS if a != sender]
    else:
        targets = [a for a in AGENTS if a in tags and a != sender]
    return targets


def process_turn(agent: str, hops: int) -> None:
    """Run one queued agent turn. Extracted so tests call the real worker body.

    worker() is the infinite loop; this is one iteration of it. Epoch guards,
    silence suppression, and re-dispatch after each reply all live here.
    """
    with LOCK:
        state = load_state()
        if state.get("stop"):
            return
        # Capture epoch before the long engine call so a room swap while
        # we think cannot land our reply (or session id) in the new room.
        epoch = ROOM_EPOCH
        last = state["last_seen"].get(agent, 0)
    fresh = [m for m in all_messages()
             if m["id"] > last and m["from"] != agent]
    if not fresh:
        return
    # Every engine carries real session memory now (grok included, via
    # minted -s/-r session ids), so everyone gets only the delta. Adapters
    # that want a window on a brand-new session (grok) get the rolling
    # window once to rebuild context; minutes only for brand-new grok.
    eng = SEATS[agent]["engine"]
    if ADAPTERS[eng].wants_window_on_new_session and not get_session(state, agent):
        feed = all_messages()[-GROK_WINDOW:]
    else:
        feed = fresh
    consumed = fresh[-1]["id"]   # marked BEFORE the engine call, so
    # anything that lands while the engine thinks stays unread and is
    # picked up by the dispatch that message itself triggers.
    with LOCK:
        if ROOM_EPOCH != epoch:
            return  # room swapped between dequeue and last_seen write
        state = load_state()
        state["last_seen"][agent] = consumed
        save_state(state)
    BUSY[agent] = True
    try:
        reply = run_engine(agent, build_prompt(agent, feed), room_epoch=epoch)
    except subprocess.TimeoutExpired:
        reply = f"needs the user: my engine call timed out after {ENGINE_TIMEOUT}s."
    except Exception as exc:  # engine failures die loudly in the feed
        reply = f"needs the user: my engine call failed ({exc})."
    finally:
        BUSY[agent] = False
    # Room swapped while we thought — drop the reply entirely.
    with LOCK:
        if ROOM_EPOCH != epoch:
            print(f"{now()} {agent} discarded reply after room swap")
            return
    # Silence protocol: "[silent]" means "nothing to add" - no bubble,
    # no routing, no another-round-of-acknowledgements.
    if reply.strip().lower() in ("[silent]", "silent", "(silent)"):
        print(f"{now()} {agent} stayed silent")
        return
    msg = append_message(
        agent, reply or "(empty reply)", hops=hops, room_epoch=epoch)
    if msg is None:
        print(f"{now()} {agent} discarded reply after room swap")
        return
    with LOCK:
        if ROOM_EPOCH != epoch:
            return
        state = load_state()
        state["last_seen"][agent] = max(state["last_seen"].get(agent, 0), msg["id"])
        save_state(state)
    route(msg)


def worker(agent: str) -> None:
    while True:
        hops = WORK_QUEUES[agent].get()
        process_turn(agent, hops)


def route(msg: dict) -> None:
    """Decide who gets called next after any message lands."""
    maybe_update_minutes()
    with LOCK:
        state = load_state()
        if state.get("stop"):
            return
        mission = state.get("mission")
        br = _boardroom_open(state)
    # Boardroom: remember the chair's latest draft summary, but never auto-close.
    if br and msg["from"] == CHAIR_SEAT and msg.get("kind") != "system":
        with LOCK:
            state = load_state()
            open_br = _boardroom_open(state)
            if open_br:
                open_br = dict(open_br)
                txt = (msg.get("text") or "").strip()
                # Only a message that actually reads as the chair's summary may
                # become the room's summary. Storing every chair remark here is
                # how a mid-discussion aside ends up published as the agreed
                # outcome to whoever called /agent-rooms.
                if txt and _reads_as_chair_summary(txt):
                    open_br["summary"] = txt
                elif txt:
                    open_br["last_chair_remark"] = txt
                state["boardroom"] = open_br
                save_state(state)
    # Boardroom is not a mission: a chair "DONE" does not end a boardroom.
    # Mission complete only fires when a real mission is open (not boardroom).
    if mission and msg["from"] == "claude" and re.search(r"\bDONE\b", msg["text"]):
        with LOCK:
            state = load_state()
            state["mission"] = None
            save_state(state)
        append_message("room", "Mission complete - claude declared DONE.", kind="system")
        return
    # the user can close an open boardroom by saying so in chat.
    if br and msg["from"] == "you":
        if re.search(r"(?i)\b(close the boardroom|close boardroom|end boardroom)\b",
                     msg.get("text") or ""):
            res = close_boardroom()
            # Never swallow the result: an unanswered "close the boardroom"
            # reads as the room ignoring the user.
            if res.get("error"):
                append_message("room", f"Could not close the boardroom: {res['error']}",
                               kind="system")
            return
        # Budget pause is recoverable: the user can grant another round of turns.
        if re.search(r"(?i)^\s*(continue|resume|carry on)\b"
                     r"|\b(continue the boardroom|resume the boardroom|carry on)\b",
                     msg.get("text") or ""):
            res = resume_boardroom()
            if res.get("error"):
                append_message("room", f"Could not continue the boardroom: {res['error']}",
                               kind="system")
            return
    targets = parse_targets(msg["text"], msg["from"])
    if msg["from"] == "you" and not targets and not TAG.search(msg["text"]):
        # In boardroom, an untagged message from you does not auto-wake the chair
        # unless it is a close command (handled above). Idle = no dispatch.
        if br:
            return
        targets = ["claude"]
    next_hops = msg["hops"] + 1 if msg["from"] != "you" else 1
    if msg["from"] != "you" and next_hops > hop_limit():
        append_message("room",
                       f"Paused: {hop_limit()} agent-to-agent hops reached. "
                       "Say 'continue' or give new instructions.", kind="system")
        return
    if not targets:
        return
    # Boardroom turn budget: cap total paid engine calls, independent of hop depth.
    # Without this, @all fan-out multiplies per generation while hops only count depth.
    if br:
        wanted = len(targets)
        granted, should_announce = _boardroom_claim_turns(wanted)
        for target in targets[:granted]:
            dispatch(target, next_hops)
        if should_announce:
            append_message(
                "room",
                f"Paused: boardroom hit its turn limit "
                f"({BOARDROOM_TURN_BUDGET} agent turns). "
                "Discussion is paused for the user — close it, or say continue "
                "with a narrower ask.",
                kind="system",
            )
        return
    for target in targets:
        dispatch(target, next_hops)


# ----------------------------------------------------------------- http
# In-flight device-code sign-in per engine. Values never hold access/refresh
# tokens — only verification_url, user_code, redacted raw, and status.
LOGIN_FLOWS: dict[str, dict] = {}
LOGIN_FLOW_LOCK = threading.Lock()
_DEVICE_LOGIN_ENGINES = frozenset({"grok", "kimi"})


def _login_flow_public(flow: dict) -> dict:
    """Slice safe for JSON to the page — no credential fields."""
    out = {
        "status": flow.get("status"),
        "verification_url": flow.get("verification_url"),
        "user_code": flow.get("user_code"),
        "readable": bool(flow.get("readable")),
        "signed_in": bool(flow.get("signed_in")),
    }
    if flow.get("message"):
        out["message"] = flow["message"]
    # Only surface raw when the flow was unreadable (criterion 2).
    if flow.get("status") == "unreadable" and flow.get("raw_output"):
        out["raw_output"] = flow["raw_output"]
    return out


def _set_login_flow(engine: str, **fields) -> dict:
    with LOGIN_FLOW_LOCK:
        cur = dict(LOGIN_FLOWS.get(engine) or {})
        cur.update(fields)
        # Never stash token keys even if a caller slips.
        for banned in ("access_token", "refresh_token", "token", "credential"):
            cur.pop(banned, None)
        if "raw_output" in cur and cur["raw_output"]:
            cur["raw_output"] = redact_secrets(str(cur["raw_output"]))
        LOGIN_FLOWS[engine] = cur
        return dict(cur)


def start_device_login(
    engine: str,
    run=None,
    *,
    background: bool = True,
    popen=None,
    timeout: float | None = None,
) -> dict:
    """Start device-code sign-in for grok or kimi.

    `background=True` (production): stream CLI stdout in a daemon thread so the
    verification URL and short code appear before the human finishes.
    `background=False` (tests): block via adapter.run_device_login(run).

    Wall-clock `timeout` (default DEVICE_LOGIN_TIMEOUT) kills a hung child and
    sets status failed so Sign in / Try again can restart without a server reboot.

    Never runs a real login when tests inject a FakeRun. Claude is rejected
    (T03 owns the terminal banner path).
    """
    engine = (engine or "").strip().lower()
    if engine not in _DEVICE_LOGIN_ENGINES:
        return {
            "error": (
                "In-room device sign-in is only for grok and kimi. "
                "Claude needs a Terminal (see T03)."
            ),
            "engine": engine or None,
        }
    adapter = ADAPTERS[engine]
    if run is None:
        run = subprocess.run
    login_timeout = (
        float(timeout) if timeout is not None else float(DEVICE_LOGIN_TIMEOUT)
    )

    with LOGIN_FLOW_LOCK:
        existing = LOGIN_FLOWS.get(engine)
        if existing and existing.get("status") in ("starting", "awaiting_user"):
            return _login_flow_public(existing)

    if not background:
        # Test path: one blocking FakeRun (or real run) to completion.
        if engine == "kimi":
            result = adapter.run_device_login(run, cred_path=KIMI_CRED_PATH)
        else:
            result = adapter.run_device_login(run)
        flow = _set_login_flow(
            engine,
            status=result.status,
            verification_url=result.verification_url,
            user_code=result.user_code,
            readable=result.readable,
            signed_in=result.signed_in,
            message=result.message,
            raw_output=result.raw_output or "",
        )
        return _login_flow_public(flow)

    # Production: stream so URL/code show while the CLI still waits.
    if popen is None:
        popen = subprocess.Popen

    _set_login_flow(
        engine,
        status="starting",
        verification_url=None,
        user_code=None,
        readable=False,
        signed_in=False,
        message=None,
        raw_output="",
    )

    def worker() -> None:
        cmd = adapter.device_login_cmd()
        buf: list[str] = []
        deadline = time.monotonic() + login_timeout
        try:
            proc = popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line-buffered so URL surfaces before process exit
                cwd=ENGINE_WORK_DIR,
            )
        except Exception as exc:  # noqa: BLE001
            _set_login_flow(
                engine,
                status="unreadable",
                readable=False,
                message="The sign-in flow could not be read.",
                raw_output=redact_secrets(str(exc)),
            )
            return
        if proc.stdout is None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            _set_login_flow(
                engine,
                status="failed",
                readable=False,
                signed_in=False,
                message="Sign-in failed to start. Click Try again.",
                raw_output="",
            )
            return

        stop = threading.Event()

        def read_loop() -> None:
            try:
                for line in proc.stdout:
                    if stop.is_set():
                        break
                    buf.append(line)
                    parsed = parse_device_login_output("".join(buf))
                    # Do not overwrite a timeout/failed status after stop.
                    if parsed.readable and not stop.is_set():
                        _set_login_flow(
                            engine,
                            status="awaiting_user",
                            verification_url=parsed.verification_url,
                            user_code=parsed.user_code,
                            readable=True,
                            message=parsed.message,
                            raw_output="",
                        )
            except Exception:
                pass
            finally:
                stop.set()

        reader = threading.Thread(
            target=read_loop, daemon=True, name=f"login-read-{engine}")
        reader.start()
        timed_out = False
        try:
            while not stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    stop.set()
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    break
                stop.wait(timeout=min(0.25, remaining))
            if timed_out:
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                _set_login_flow(
                    engine,
                    status="failed",
                    verification_url=None,
                    user_code=None,
                    readable=False,
                    signed_in=False,
                    message=(
                        "Sign-in timed out. Click Try again to restart."
                    ),
                    raw_output=redact_secrets("".join(buf)),
                )
                return
            try:
                rc = proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    rc = proc.wait(timeout=5)
                except Exception:
                    rc = -1
        except Exception as exc:  # noqa: BLE001
            stop.set()
            try:
                proc.kill()
            except Exception:
                pass
            _set_login_flow(
                engine,
                status="unreadable",
                readable=False,
                message="The sign-in flow could not be read.",
                raw_output=redact_secrets("".join(buf) + f"\n{exc}"),
            )
            return

        raw = "".join(buf)
        parsed = parse_device_login_output(raw)
        if rc == 0:
            try:
                if engine == "kimi":
                    signed = bool(
                        adapter.check_signed_in(run, cred_path=KIMI_CRED_PATH))
                else:
                    signed = bool(adapter.check_signed_in(run))
            except Exception:
                signed = False
            if not parsed.readable:
                _set_login_flow(
                    engine,
                    status="unreadable",
                    verification_url=None,
                    user_code=None,
                    readable=False,
                    signed_in=signed,
                    message="The sign-in flow could not be read.",
                    raw_output=redact_secrets(raw),
                )
            else:
                _set_login_flow(
                    engine,
                    status="complete" if signed else "failed",
                    verification_url=parsed.verification_url,
                    user_code=parsed.user_code,
                    readable=True,
                    signed_in=signed,
                    message=(
                        None if signed
                        else (
                            parsed.message
                            or "Sign-in finished but the seat is still signed out."
                        )
                    ),
                    raw_output="",
                )
            # Login finished: drop probe cache so the next GET re-checks CLIs.
            clear_auth_cache()
        else:
            if parsed.readable:
                _set_login_flow(
                    engine,
                    status="failed",
                    verification_url=parsed.verification_url,
                    user_code=parsed.user_code,
                    readable=True,
                    signed_in=False,
                    message=parsed.message or "Sign-in did not complete.",
                    raw_output="",
                )
            else:
                _set_login_flow(
                    engine,
                    status="unreadable",
                    verification_url=None,
                    user_code=None,
                    readable=False,
                    signed_in=False,
                    message="The sign-in flow could not be read.",
                    raw_output=redact_secrets(raw),
                )

    threading.Thread(target=worker, daemon=True, name=f"login-{engine}").start()
    with LOGIN_FLOW_LOCK:
        return _login_flow_public(LOGIN_FLOWS[engine])


# Probe results are cached server-side so a 4s client poll does not shell out
# to claude/grok on every tick. Re-check (force=True) bypasses the cache.
AUTH_CACHE_TTL_SEC = 45.0
_AUTH_PROBE_LOCK = threading.Lock()
_AUTH_PROBE_CACHE: dict = {"t": 0.0, "probes": None}  # probes: eng -> bool


def clear_auth_cache() -> None:
    """Drop the probe cache (tests + post-login). Next auth_state re-probes."""
    with _AUTH_PROBE_LOCK:
        _AUTH_PROBE_CACHE["probes"] = None
        _AUTH_PROBE_CACHE["t"] = 0.0


def auth_state(run, *, force: bool = False) -> dict:
    """Signed-in / signed-out for each engine. No model tokens spent.

    `run` is required — pass subprocess.run in production, a FakeRun in tests.
    A missing `run` used to default to a live CLI shell-out; that is gone so a
    missed test patch cannot hit the real network.

    `force=True` bypasses the probe cache (re-check button). Otherwise probes
    are reused for AUTH_CACHE_TTL_SEC. Live device-login flow state is always
    merged fresh from LOGIN_FLOWS (never cached).
    """
    with _AUTH_PROBE_LOCK:
        now = time.monotonic()
        cached = _AUTH_PROBE_CACHE["probes"]
        fresh = (
            not force
            and cached is not None
            and (now - _AUTH_PROBE_CACHE["t"]) < AUTH_CACHE_TTL_SEC
        )
        if fresh:
            probes = dict(cached)
        else:
            probes = {}
            for name in ("claude", "grok", "kimi"):
                adapter = ADAPTERS[name]
                try:
                    if name == "kimi":
                        signed = adapter.check_signed_in(
                            run, cred_path=KIMI_CRED_PATH)
                    else:
                        signed = adapter.check_signed_in(run)
                except Exception:
                    signed = False
                probes[name] = bool(signed)
            _AUTH_PROBE_CACHE["probes"] = dict(probes)
            _AUTH_PROBE_CACHE["t"] = now

    out = {}
    for name in ("claude", "grok", "kimi"):
        entry = {"signed_in": bool(probes[name])}
        with LOGIN_FLOW_LOCK:
            flow = LOGIN_FLOWS.get(name)
            if flow:
                entry["login"] = _login_flow_public(flow)
                # If the flow just completed signed-in, prefer that over a
                # stale probe race (probe may still see old creds briefly).
                if flow.get("signed_in"):
                    entry["signed_in"] = True
        out[name] = entry
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet server log
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html", "/index"):
            body = (ROOM_ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/avatars/"):
            name = self.path.split("/avatars/", 1)[1].split("?")[0]
            if not re.fullmatch(r"[a-z0-9_-]+\.png", name):
                self._json({"error": "bad avatar name"}, 400)
                return
            f = AVATAR_DIR / name
            if not f.exists():
                self._json({"error": "not found"}, 404)
                return
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/rooms"):
            self._json({"rooms": list_archives()})
            return
        if self.path.startswith("/api/room/"):
            name = self.path.split("/api/room/", 1)[1].split("?")[0]
            if not re.fullmatch(r"room-[0-9A-Za-z-]+", name):
                self._json({"error": "bad room id"}, 400)
                return
            d = ROOMS_DIR / name
            if not d.is_dir() or name == "current":
                self._json({"error": "not found"}, 404)
                return
            self._json({"id": name, "messages": room_messages(d)})
            return
        if self.path.startswith("/api/projects"):
            self._json({"projects": project_dirs()})
            return
        if self.path.startswith("/api/roster"):
            self._json({"seats": [
                {"name": n, "color": s.get("color", "#8a8f98"),
                 "engine": s["engine"], "model": s.get("model"),
                 "effort": s.get("effort"), "role": s.get("role", ""),
                 "ctx": CTX.get(n, {"used": 0, "limit": ctx_limit(s), "est": True})}
                for n, s in SEATS.items()]})
            return
        if self.path.split("?")[0] in ("/api/models", "/api/models/"):
            # Seat editor model dropdown (DM-11 / T06). Optional ?engine= filter.
            qs = parse_qs(urlparse(self.path).query)
            eng_vals = qs.get("engine") or []
            eng = (eng_vals[0] if eng_vals else "").strip().lower()
            if eng in ("claude", "grok", "kimi"):
                self._json({eng: list_models(
                    eng,
                    grok_cache_path=GROK_MODELS_CACHE,
                    kimi_config_path=KIMI_CONFIG_PATH,
                )})
            else:
                self._json(models_for_all_engines())
            return
        if self.path.split("?")[0] in ("/api/auth", "/api/auth/"):
            qs = parse_qs(urlparse(self.path).query)
            force_vals = qs.get("refresh") or qs.get("force") or []
            force = any(v in ("1", "true", "yes") for v in force_vals)
            self._json(auth_state(run=subprocess.run, force=force))
            return
        if self.path.split("?")[0] == "/minutes":
            body = (read_minutes() or
                    "No minutes yet - they appear automatically after "
                    f"{MINUTES_EVERY} messages in the room.").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/messages"):
            since = 0
            if "since=" in self.path:
                try:
                    since = int(self.path.split("since=")[1].split("&")[0])
                except ValueError:
                    since = 0
            msgs = [m for m in all_messages() if m["id"] > since]
            with LOCK:
                state = load_state()
            self._json({"messages": msgs,
                        "busy": [a for a in AGENTS if BUSY[a]],
                        "mission": state.get("mission"),
                        "boardroom": state.get("boardroom"),
                        "stopped": state.get("stop", False)})
            return
        # T08 / DM-12: closed boardroom outcome for external callers (T09).
        if self.path.split("?")[0] in (
            "/api/boardroom/outcome", "/api/boardroom/outcome/",
        ):
            outcome = get_boardroom_outcome()
            if not outcome:
                self._json({"error": "no boardroom outcome yet", "status": "none"}, 404)
            else:
                self._json(outcome)
            return
        if self.path.split("?")[0] in ("/api/boardroom", "/api/boardroom/"):
            with LOCK:
                state = load_state()
            self._json({
                "boardroom": state.get("boardroom"),
                "title": state.get("title"),
                "outcome": get_boardroom_outcome(),
            })
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            payload = {}
        if self.path == "/api/attach":
            name = re.sub(r"[^\w. ()-]", "_", (payload.get("name") or "file"))[:80]
            raw = payload.get("data") or ""
            try:
                blob = base64.b64decode(raw)
            except (binascii.Error, ValueError):
                self._json({"error": "bad base64"}, 400)
                return
            if not blob:
                self._json({"error": "empty file"}, 400)
                return
            if len(blob) > 20 * 1024 * 1024:
                self._json({"error": "file over 20MB"}, 400)
                return
            ATTACH_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            dest = ATTACH_DIR / f"{stamp}-{name}"
            dest.write_bytes(blob)
            self._json({"ok": True, "name": name, "path": str(dest)})
            return
        if self.path == "/api/send":
            text = (payload.get("text") or "").strip()
            if not text:
                self._json({"error": "empty"}, 400)
                return
            with LOCK:
                state = load_state()
                state["stop"] = False
                save_state(state)
            msg = append_message("you", text)
            route(msg)
            self._json({"ok": True, "id": msg["id"]})
            return
        if self.path == "/api/mission":
            goal = (payload.get("goal") or "").strip()
            done = (payload.get("done") or "").strip()
            if not goal or not done:
                self._json({"error": "mission needs a goal and a done-check"}, 400)
                return
            with LOCK:
                state = load_state()
                state["mission"] = {"goal": goal, "done": done, "started": now()}
                state["stop"] = False
                save_state(state)
            msg = append_message(
                "you",
                f"MISSION for @claude: {goal}\nDone when: {done}\n"
                "Chair this: plan briefly, hand build steps to @grok, have @kimi review each build, "
                "judge verdicts against the done-check, loop until it truly passes, "
                "then reply DONE with the evidence (no tags in that final message).",
                kind="mission")
            route(msg)
            self._json({"ok": True})
            return
        # T08 / DM-12: boardroom — topic discussion, all seats, the user closes.
        # Never starts the mission loop. seats= in the payload is ignored.
        if self.path.split("?")[0] in ("/api/boardroom", "/api/boardroom/"):
            topic = (payload.get("topic") or "").strip()
            if not topic:
                self._json({"error": "boardroom needs a topic"}, 400)
                return
            result = start_boardroom(topic)
            if result.get("error"):
                self._json(result, 400)
            else:
                self._json(result)
            return
        if self.path.split("?")[0] in (
            "/api/boardroom/close", "/api/boardroom/close/",
        ):
            summary = payload.get("summary")
            if summary is not None:
                summary = str(summary)
            result = close_boardroom(summary=summary)
            if result.get("error"):
                self._json(result, 400)
            else:
                self._json(result)
            return
        if self.path == "/api/stop":
            with LOCK:
                state = load_state()
                state["stop"] = True
                state["mission"] = None
                save_state(state)
            _drain_work_queues()
            append_message("room", "STOP pressed - all agent work halted.", kind="system")
            self._json({"ok": True})
            return
        if self.path == "/api/seats":
            err = save_roster_change(payload)
            if err:
                self._json({"error": err}, 400)
            else:
                append_message("room", f"Seat '{payload.get('name','').lower().strip()}' saved.", kind="system")
                self._json({"ok": True})
            return
        if self.path == "/api/seats/delete":
            err = save_roster_change(None, delete_name=payload.get("name", ""))
            if err:
                self._json({"error": err}, 400)
            else:
                append_message("room", f"Seat '{payload.get('name','').lower().strip()}' removed.", kind="system")
                self._json({"ok": True})
            return
        if self.path == "/api/newroom":
            stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            with LOCK:
                # Invalidate in-flight workers before the live room moves.
                bump_room_epoch()
                _drain_work_queues()
                _clear_boardroom_outcome()
                if CURRENT.exists():
                    # T04 / DM-09: write session_id+engine+model for speakers
                    # before the directory moves into the archive list.
                    file_current_room(stamp=stamp)
                else:
                    ensure_room()
                CTX.clear()      # fresh room = fresh sessions = empty context
            append_message("room", "New room. Old conversation archived.", kind="system")
            self._json({"ok": True})
            return
        if self.path.split("?")[0] in ("/api/reopen", "/api/reopen/"):
            # T05 / DM-10: make a filed chat the live chat (session reattach),
            # or open it read-only when no session ids were stored.
            room_id = (payload.get("id") or "").strip()
            with LOCK:
                result = reopen_room(room_id)
            if result.get("error"):
                if result["error"] == "not found":
                    code = 404
                elif result["error"] == "busy":
                    code = 409
                else:
                    code = 400
                self._json(result, code)
            else:
                self._json(result)
            return
        if self.path.split("?")[0] in ("/api/auth/login", "/api/auth/login/"):
            # Device-code sign-in for grok/kimi (T02). Streams in a thread;
            # poll GET /api/auth for verification_url + user_code.
            engine = (payload.get("engine") or "").strip().lower()
            out = start_device_login(engine, background=True)
            if "error" in out:
                self._json(out, 400)
            else:
                self._json({"ok": True, "engine": engine, "login": out})
            return
        self._json({"error": "not found"}, 404)


def main() -> None:
    # Own log on disk, always — a server that dies must leave a note.
    log_dir = ROOM_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    import sys
    sys.stdout = sys.stderr = open(log_dir / "server.log", "a", buffering=1)
    print(f"{now()} starting on http://127.0.0.1:{PORT}")
    ensure_room()
    for agent in AGENTS:
        threading.Thread(target=worker, args=(agent,), daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"{now()} serving")
    server.serve_forever()


def run() -> None:
    try:
        main()
    except Exception:
        import traceback
        log_dir = ROOM_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "server.log", "a") as fh:
            fh.write(f"{now()} CRASHED:\n{traceback.format_exc()}\n")
        raise


if __name__ == "__main__":
    run()
