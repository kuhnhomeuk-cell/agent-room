#!/usr/bin/env python3
"""/agent-rooms command (T09 / DM-13).

Start the Agent Room server if it is not running, attach if it is, open the
browser on a fresh boardroom chat named after the topic, wait for the user to
close, and print the chair's outcome back to the invoking session.

Never presents an unsummarised chair remark or a placeholder as an agreed
decision. Waiting is a slow poll with no model calls.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

DEFAULT_BASE_URL = "http://127.0.0.1:8787"
DEFAULT_PORT = 8787
# Slow poll: the user may take an hour. Never spin the CPU or burn tokens.
DEFAULT_POLL_INTERVAL_S = 5.0
MIN_POLL_INTERVAL_S = 2.0
# Dead-server guard: consecutive unreachable polls before we exit.
DEFAULT_MAX_CONSECUTIVE_UNREACHABLE = 6
# Hard cap so an abandoned wait cannot sit forever (the user may take an hour;
# four hours is the ceiling).
DEFAULT_MAX_WAIT_S = 4 * 3600.0
ROOM_ROOT = Path(__file__).resolve().parent
SERVER_PY = ROOM_ROOT / "server.py"
HEALTH_PATH = "/api/messages"
BOARDROOM_PATH = "/api/boardroom"
OUTCOME_PATH = "/api/boardroom/outcome"
HTTP_TIMEOUT_S = 5


@dataclass
class Deps:
    """Injectable I/O so tests drive the real run() path with a fake server."""

    base_url: str = DEFAULT_BASE_URL
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
    server_py: Path = SERVER_PY
    max_consecutive_unreachable: int = DEFAULT_MAX_CONSECUTIVE_UNREACHABLE
    max_wait_s: float = DEFAULT_MAX_WAIT_S
    outcome_file: str | None = None
    is_up: Callable[[], bool] | None = None
    start_server: Callable[[], None] | None = None
    open_browser: Callable[[str], None] | None = None
    http_json: Callable[[str, str, dict | None], tuple[int, Any]] | None = None
    sleep: Callable[[float], None] | None = None

    def __post_init__(self) -> None:
        if self.poll_interval_s < MIN_POLL_INTERVAL_S:
            self.poll_interval_s = MIN_POLL_INTERVAL_S
        if self.max_consecutive_unreachable < 1:
            self.max_consecutive_unreachable = 1
        if self.max_wait_s < self.poll_interval_s:
            self.max_wait_s = self.poll_interval_s
        if self.is_up is None:
            self.is_up = self._default_is_up
        if self.start_server is None:
            self.start_server = self._default_start_server
        if self.open_browser is None:
            self.open_browser = self._default_open_browser
        if self.http_json is None:
            self.http_json = self._default_http_json
        if self.sleep is None:
            self.sleep = time.sleep

    def _default_is_up(self) -> bool:
        try:
            code, _ = self._default_http_json("GET", HEALTH_PATH, None)
            return code == 200
        except Exception:
            return False

    def _default_start_server(self) -> None:
        # Detached process, same pattern as start.sh.
        subprocess.Popen(
            [sys.executable, str(self.server_py)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _default_open_browser(self, url: str) -> None:
        # Any OS; ignore failures so a headless host still gets the outcome.
        try:
            webbrowser.open(url)
        except Exception:
            pass

    def _default_http_json(
        self, method: str, path: str, body: dict | None
    ) -> tuple[int, Any]:
        url = self.base_url.rstrip("/") + path
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                raw = resp.read().decode() or "{}"
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"raw": raw}
                return resp.status, parsed
        except urllib.error.HTTPError as e:
            raw = e.read().decode() if e.fp else ""
            try:
                parsed = json.loads(raw) if raw else {"error": str(e)}
            except json.JSONDecodeError:
                parsed = {"error": raw or str(e)}
            return e.code, parsed
        except urllib.error.URLError:
            return 0, None


def default_deps() -> Deps:
    return Deps()


def ensure_server(deps: Deps) -> str:
    """Attach if the room is already up; otherwise start it once.

    Returns "attached" or "started". Never starts a second process when the
    health endpoint already answers.
    """
    if deps.is_up():
        return "attached"
    deps.start_server()
    # Ready immediately (common with a fast bind, and with test fakes).
    if deps.is_up():
        return "started"
    # Otherwise wait until the new process answers, or give up.
    for _ in range(24):
        deps.sleep(0.5)
        if deps.is_up():
            return "started"
    raise RuntimeError(
        "Agent Room server did not become ready after start. "
        f"Check {ROOM_ROOT / 'logs' / 'server.log'}."
    )


def post_boardroom(topic: str, deps: Deps) -> dict:
    """POST /api/boardroom — opens a fresh chat named after the topic."""
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("boardroom needs a topic")
    code, body = deps.http_json("POST", BOARDROOM_PATH, {"topic": topic})
    if code != 200 or not isinstance(body, dict) or body.get("error"):
        err = (body or {}).get("error") if isinstance(body, dict) else body
        raise RuntimeError(f"POST /api/boardroom failed ({code}): {err}")
    return body


def wait_for_outcome(deps: Deps, stdout: TextIO) -> dict:
    """Slow-poll GET /api/boardroom/outcome until the user closes.

    No model tokens: HTTP only. Plain English waiting line so the session is
    not mistaken for hung. Exits if the server dies or the max wait is hit.
    """
    stdout.write(
        "Waiting on the room — the user closes the discussion when they are done. "
        "This session is idle (no model tokens) until then.\n"
    )
    stdout.flush()
    consecutive_unreachable = 0
    waited_s = 0.0
    while True:
        code, body = deps.http_json("GET", OUTCOME_PATH, None)

        if code == 0:
            consecutive_unreachable += 1
            if consecutive_unreachable >= deps.max_consecutive_unreachable:
                raise RuntimeError(
                    "The Agent Room server stopped answering while waiting "
                    f"({consecutive_unreachable} consecutive unreachable "
                    "polls). It is not hung forever — the server is down. "
                    "Restart the room (./start.sh or "
                    "re-run /agent-rooms) and try again."
                )
        else:
            consecutive_unreachable = 0

        if (
            code == 200
            and isinstance(body, dict)
            and body.get("status") == "closed"
        ):
            return body

        if waited_s >= deps.max_wait_s:
            raise RuntimeError(
                f"Timed out waiting on the room after {int(deps.max_wait_s)}s "
                "(maximum wait). The discussion is still open in the room — "
                "close it there, or re-run /agent-rooms later."
            )

        deps.sleep(deps.poll_interval_s)
        waited_s += deps.poll_interval_s


def format_outcome(outcome: dict) -> str:
    """Render the closed boardroom outcome for the invoking session.

    summary_source decides the wording:
      chair — genuine agreed outcome
      unsummarised-chair-remark — a passing remark, not a decision
      placeholder — chair never spoke; no agreed outcome
      anything else — unknown provenance; never upgraded to a decision
    """
    topic = outcome.get("topic") or "(untitled)"
    summary = (outcome.get("summary") or "").strip()
    source = (outcome.get("summary_source") or "").strip()
    closed_at = outcome.get("closed_at") or ""

    lines = [f"Boardroom closed.", f"Topic: {topic}"]
    if closed_at:
        lines.append(f"Closed at: {closed_at}")

    if source == "chair":
        lines.append("Chair summary (agreed outcome):")
        lines.append(summary)
    elif source == "unsummarised-chair-remark":
        lines.append(
            "The chair never wrote a summary. What follows is an unsummarised "
            "chair remark — not an agreed outcome or decision."
        )
        lines.append(f"Last chair remark: {summary}")
        lines.append(f"summary_source: {source}")
    elif source == "placeholder":
        lines.append(
            "No agreed outcome: the discussion closed before the chair "
            "summarised (placeholder)."
        )
        if summary:
            lines.append(summary)
        lines.append(f"summary_source: {source}")
    else:
        # Unknown provenance — refuse to upgrade it to a decision.
        lines.append(
            "Outcome received with unknown summary_source — treat as "
            "not an agreed decision until checked."
        )
        lines.append(f"summary_source: {source or '(missing)'}")
        if summary:
            lines.append(summary)

    return "\n".join(lines) + "\n"


def _write_outcome_file(path: str | None, text: str) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def run(
    topic: str,
    *,
    deps: Deps | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Full command path: ensure server → browser → post → wait → report."""
    deps = deps or default_deps()
    stdout = stdout or sys.stdout
    topic = (topic or "").strip()
    if not topic:
        stdout.write("Usage: agent_rooms.py <topic>\n")
        return 2

    try:
        status = ensure_server(deps)
    except RuntimeError as e:
        stdout.write(f"Error: {e}\n")
        return 1
    stdout.write(
        f"Room server: {status} at {deps.base_url}\n"
    )
    stdout.flush()

    deps.open_browser(deps.base_url.rstrip("/") + "/")

    try:
        post_boardroom(topic, deps)
    except (RuntimeError, ValueError) as e:
        stdout.write(f"Error: {e}\n")
        return 1
    stdout.write(
        f"Boardroom opened on topic: {topic}\n"
        "Browser is open — follow the discussion there.\n"
    )
    stdout.flush()

    try:
        outcome = wait_for_outcome(deps, stdout)
    except RuntimeError as e:
        stdout.write(f"Error: {e}\n")
        return 1

    text = format_outcome(outcome)
    stdout.write(text)
    stdout.flush()
    _write_outcome_file(deps.outcome_file, text)
    return 0


def main(argv: list[str] | None = None, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-rooms",
        description=(
            "Convene the Agent Room on a topic and return the chair's "
            "summary after the user closes."
        ),
    )
    parser.add_argument(
        "topic",
        nargs="+",
        help="Boardroom topic (fresh chat is named after this)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Room base URL (default {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_S,
        help=f"Seconds between outcome polls (default {DEFAULT_POLL_INTERVAL_S})",
    )
    parser.add_argument(
        "--outcome-file",
        default=None,
        help="Write the formatted outcome here when the user closes (for background waits)",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=DEFAULT_MAX_WAIT_S,
        help=f"Give up after this many seconds (default {int(DEFAULT_MAX_WAIT_S)})",
    )
    args = parser.parse_args(argv)
    topic = " ".join(args.topic)
    # Preserve any injectables from a patched default_deps() (tests);
    # only override the CLI knobs.
    deps = default_deps()
    deps.base_url = args.base_url
    deps.poll_interval_s = max(MIN_POLL_INTERVAL_S, float(args.poll_interval))
    deps.outcome_file = args.outcome_file
    deps.max_wait_s = float(args.max_wait)
    return run(topic, deps=deps, stdout=stdout or sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
