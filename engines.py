"""Common agent-session contract for Agent Room engines.

Each CLI engine (claude, grok, kimi) is a small adapter behind one result
shape. Engine quirks stay here; server.py only stores the session and updates
context meters from EngineResult.

stdlib only. `run` is injected (subprocess.run signature-in-use) so tests can
fake the CLI without patching globals. `clear_session` is an optional injected
callable the room provides so resume self-heal can drop a stale id BEFORE the
retry (matching pre-refactor set_session(None) timing).
"""
from __future__ import annotations

import json
import os
import shutil
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover — older interpreters
    tomllib = None  # type: ignore

HOME = Path.home()
CLAUDE_BIN = shutil.which("claude") or str(HOME / ".local" / "bin" / "claude")
GROK_BIN = shutil.which("grok") or str(HOME / ".grok" / "bin" / "grok")
KIMI_BIN = shutil.which("kimi") or str(HOME / ".kimi-code" / "bin" / "kimi")

# Match server.py defaults so adapters keep the same timeouts/cwd when the
# injected run is the real subprocess.run (adapters own these kwargs).
ENGINE_TIMEOUT = 600
# Auth probes are cheap status checks — never a model generation.
AUTH_TIMEOUT = 30
# Device-code login waits for the human; allow several minutes.
DEVICE_LOGIN_TIMEOUT = 600
# The folder the agents work in. Set AGENT_ROOM_WORKDIR, or it defaults to
# the folder the server was started from.
WORK_DIR = os.environ.get("AGENT_ROOM_WORKDIR") or str(Path.cwd())

# Kimi has no status subcommand; signed-in is read from this credential file.
KIMI_CRED_PATH = HOME / ".kimi-code" / "credentials" / "kimi-code.json"

# Model catalogs (DM-11 / T06). Read-only paths; never written by the room.
GROK_MODELS_CACHE = HOME / ".grok" / "models_cache.json"
KIMI_CONFIG_PATH = HOME / ".kimi-code" / "config.toml"

# One place: raw model id → human display name (design contract rule 1).
# Fable and Opus are distinct — never map claude-fable-5 to "Opus 5".
_DISPLAY_NAMES: dict[str, str] = {
    "fable": "Fable 5",
    "opus": "Opus 5",
    "sonnet": "Sonnet 5",
    "claude-fable-5": "Fable 5",
    "claude-opus-5": "Opus 5",
    "claude-sonnet-5": "Sonnet 5",
    "grok-4.5": "Grok 4.5",
    "kimi-code/k3": "K3",
    "kimi-code/k3-256k": "K3-256k",
    "kimi-code/kimi-for-coding": "K2.7 Coding",
    "kimi-code/kimi-for-coding-highspeed": "K2.7 Coding Highspeed",
    "k3": "K3",
    "k3-256k": "K3-256k",
    "kimi-for-coding": "K2.7 Coding",
    "kimi-for-coding-highspeed": "K2.7 Coding Highspeed",
}

# Claude has no discovery API — three aliases that track the latest model.
CLAUDE_MODELS: list[dict[str, str]] = [
    {"id": "fable", "name": "Fable 5"},
    {"id": "opus", "name": "Opus 5"},
    {"id": "sonnet", "name": "Sonnet 5"},
]

# Used when a source file is missing, unreadable, or empty after filtering.
FALLBACK_MODELS: dict[str, list[dict[str, str]]] = {
    "claude": list(CLAUDE_MODELS),
    "grok": [{"id": "grok-4.5", "name": "Grok 4.5"}],
    "kimi": [
        {"id": "kimi-code/k3", "name": "K3"},
        {"id": "kimi-code/k3-256k", "name": "K3-256k"},
        {"id": "kimi-code/kimi-for-coding", "name": "K2.7 Coding"},
        {"id": "kimi-code/kimi-for-coding-highspeed", "name": "K2.7 Coding Highspeed"},
    ],
}


def model_display_name(model_id: str | None) -> str:
    """Map a model id to its on-screen name. Never returns a raw id when known."""
    if not model_id:
        return ""
    mid = str(model_id).strip()
    if mid in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[mid]
    # Unknown id: title-case the last path segment as a last resort, still no
    # Haiku offering (filter happens at list time).
    tail = mid.split("/")[-1]
    return tail.replace("-", " ").replace("_", " ").strip() or mid


def _is_haiku(model_id: str, name: str = "") -> bool:
    blob = f"{model_id} {name}".lower()
    return "haiku" in blob


def _clean_models(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop Haiku and empty ids; keep first occurrence of each id."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for e in entries:
        mid = (e.get("id") or "").strip()
        name = (e.get("name") or "").strip() or model_display_name(mid)
        if not mid or _is_haiku(mid, name):
            continue
        if mid in seen:
            continue
        seen.add(mid)
        out.append({"id": mid, "name": name})
    return out


def _default_log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _read_grok_models(path: Path, log: Callable[[str], None]) -> list[dict[str, str]] | None:
    """Parse ~/.grok/models_cache.json. None means fall back."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log(f"model list: grok cache unreadable ({path}): {exc}; using fallback")
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        log(f"model list: grok cache invalid JSON ({path}): {exc}; using fallback")
        return None
    models_obj = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models_obj, dict) or not models_obj:
        log(f"model list: grok cache has no models ({path}); using fallback")
        return None
    entries: list[dict[str, str]] = []
    for key, val in models_obj.items():
        mid = str(key)
        name = ""
        if isinstance(val, dict):
            info = val.get("info") if isinstance(val.get("info"), dict) else val
            if isinstance(info, dict):
                mid = str(info.get("id") or info.get("model") or key)
                name = str(info.get("name") or info.get("system_prompt_label") or "")
        if not name:
            name = model_display_name(mid)
        entries.append({"id": mid, "name": name})
    cleaned = _clean_models(entries)
    if not cleaned:
        log(f"model list: grok cache yielded no usable models ({path}); using fallback")
        return None
    return cleaned


def _read_kimi_models(path: Path, log: Callable[[str], None]) -> list[dict[str, str]] | None:
    """Parse ~/.kimi-code/config.toml [models."…"] tables. None means fall back.

    One path only: tomllib (stdlib on Python 3.11+).
    No regex dual-parser — both paths were independently deletable under green.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log(f"model list: kimi config unreadable ({path}): {exc}; using fallback")
        return None
    if tomllib is None:  # pragma: no cover — Python < 3.11
        log(f"model list: tomllib unavailable; using fallback")
        return None
    try:
        data = tomllib.loads(raw)
    except Exception as exc:  # tomllib.TOMLDecodeError and peers
        log(f"model list: kimi config invalid TOML ({path}): {exc}; using fallback")
        return None
    entries: list[dict[str, str]] = []
    if isinstance(data, dict):
        models_tbl = data.get("models")
        if isinstance(models_tbl, dict):
            for mid, body in models_tbl.items():
                name = ""
                if isinstance(body, dict):
                    dn = body.get("display_name")
                    if isinstance(dn, str) and dn.strip():
                        name = dn.strip()
                if not name:
                    name = model_display_name(str(mid))
                entries.append({"id": str(mid), "name": name})
    cleaned = _clean_models(entries)
    if not cleaned:
        log(f"model list: kimi config yielded no usable models ({path}); using fallback")
        return None
    return cleaned


def list_models(
    engine: str,
    *,
    grok_cache_path: Path | str | None = None,
    kimi_config_path: Path | str | None = None,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, str]]:
    """Return [{id, name}, …] for a seat engine. Never empty; never includes Haiku.

    claude: fixed aliases. grok: models_cache.json. kimi: config.toml models tables.
    Missing or unreadable sources fall back to FALLBACK_MODELS and log why.
    """
    log_fn = log or _default_log
    eng = (engine or "").strip().lower()
    if eng == "claude":
        return _clean_models(list(CLAUDE_MODELS)) or list(FALLBACK_MODELS["claude"])

    if eng == "grok":
        path = Path(grok_cache_path) if grok_cache_path is not None else GROK_MODELS_CACHE
        if not path.is_file():
            log_fn(f"model list: grok cache missing ({path}); using fallback")
            return list(FALLBACK_MODELS["grok"])
        got = _read_grok_models(path, log_fn)
        return got if got else list(FALLBACK_MODELS["grok"])

    if eng == "kimi":
        path = Path(kimi_config_path) if kimi_config_path is not None else KIMI_CONFIG_PATH
        if not path.is_file():
            log_fn(f"model list: kimi config missing ({path}); using fallback")
            return list(FALLBACK_MODELS["kimi"])
        got = _read_kimi_models(path, log_fn)
        return got if got else list(FALLBACK_MODELS["kimi"])

    # Unknown engine — still never empty.
    log_fn(f"model list: unknown engine {engine!r}; using claude fallback")
    return list(FALLBACK_MODELS["claude"])


def list_models_all(
    *,
    grok_cache_path: Path | str | None = None,
    kimi_config_path: Path | str | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """All three engines' model lists (for GET /api/models)."""
    return {
        eng: list_models(
            eng,
            grok_cache_path=grok_cache_path,
            kimi_config_path=kimi_config_path,
            log=log,
        )
        for eng in ("claude", "grok", "kimi")
    }


ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")
# https URL in device-flow text (strip trailing punctuation).
_DEVICE_URL_RE = re.compile(r"https://[^\s<>\"']+")
# Prefer device/verification-looking URLs over docs/support links.
_DEVICE_URL_HINTS = (
    "device", "verify", "verification", "oauth", "activate",
    "user-code", "usercode", "user_code", "login/device",
)
# Label is case-insensitive; code characters stay case-sensitive (A-Z0-9 only).
# Prevents "error code: timeout" from becoming a fake user_code.
_DEVICE_CODE_LABELED_RE = re.compile(
    r"(?i:(?:enter(?:\s+the)?\s+)?code\s*[:=]?\s*)"
    r"([A-Z0-9]{4,8}(?:-[A-Z0-9]{4,8})?)"
)
_DEVICE_CODE_HYPHEN_RE = re.compile(r"\b([A-Z0-9]{4,8}-[A-Z0-9]{4,8})\b")
# Labeled credential values (optional quotes around key/value).
_SECRET_LABELED_RE = re.compile(
    r"(?i)([\"']?)(access_token|refresh_token|api[_-]?key|id_token|"
    r"client_secret|session[_-]?key|token)\1(\s*[=:]\s*)([\"']?)"
    r"([^\s\"',}\]]+)\4?"
)
# Authorization: Bearer <value> / bearer <value>
_SECRET_BEARER_RE = re.compile(
    r"(?i)\b(Authorization\s*:\s*Bearer|Bearer)(\s+)(\S+)"
)
# JWT-shaped (three base64url segments starting with eyJ).
_SECRET_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"
)
# Long high-entropy runs (unlabeled secrets); skip URLs and redaction markers.
_SECRET_ENTROPY_RE = re.compile(r"\b[A-Za-z0-9_+\/=-]{40,}\b")

# run(cmd, **kwargs) -> CompletedProcess-like (returncode, stdout, stderr)
RunFn = Callable[..., object]
# clear_session() drops the stored seat session id immediately (before retry).
ClearSessionFn = Callable[[], None]


@dataclass
class EngineResult:
    """Uniform result of one engine call."""

    reply: str                          # ANSI-stripped, engine noise removed
    session_id: Optional[str]           # id to store for this seat after the call
    requested_model: str
    actual_model: str                   # same as requested unless engine reports otherwise
    usage_tokens: Optional[int]         # real count when reported; else estimated/None
    estimated: bool
    error: Optional[str] = None         # set when the call failed even after self-heal
    # True when this call minted a fresh engine session (grok -s). Room resets
    # the estimated CTX accumulator on that seat (same as pre-refactor semantics).
    fresh_session: bool = False


@dataclass
class DeviceLoginResult:
    """Parsed device-code flow: URL + short code only. Never holds tokens.

    Intentionally no returncode field — exit status is process-local and is
    never consumed by the server or the page.
    """

    verification_url: Optional[str] = None
    user_code: Optional[str] = None
    raw_output: str = ""
    readable: bool = False              # True when URL or code was found
    message: Optional[str] = None       # plain English when unreadable / failed
    signed_in: bool = False
    status: str = "unknown"             # awaiting_user|complete|failed|unreadable


class EngineAdapter:
    """Duck-typed base: subclasses implement call() and the window flag."""

    wants_window_on_new_session: bool = False
    # Engines that support in-room device-code sign-in (not claude).
    supports_device_login: bool = False

    def call(
        self,
        seat: dict,
        prompt: str,
        session_id: Optional[str],
        run: RunFn,
        clear_session: Optional[ClearSessionFn] = None,
    ) -> EngineResult:
        raise NotImplementedError

    def check_signed_in(
        self,
        run: RunFn,
        cred_path: Optional[Path] = None,
    ) -> bool:
        """Return True when the engine is signed in. Never spends model tokens.

        `run` is the injected process runner (same shape as call()). `cred_path`
        is only used by file-based probes (kimi); others ignore it.
        """
        raise NotImplementedError

    def device_login_cmd(self) -> list:
        """CLI argv for this engine's device-code login. Raise if unsupported."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support device-code login")

    def run_device_login(
        self,
        run: RunFn,
        cred_path: Optional[Path] = None,
    ) -> DeviceLoginResult:
        """Run device-code login via injected runner; parse URL/code; re-probe.

        Blocking: waits for the CLI to exit (tests use a FakeRun). Production
        server may stream the same command via Popen and call
        parse_device_login_output on partial output, then re-probe itself.
        Never returns access/refresh tokens — only URL, short code, and
        redacted raw text when unreadable.
        """
        if not self.supports_device_login:
            return DeviceLoginResult(
                readable=False,
                message="This engine does not support in-room sign-in.",
                status="failed",
            )
        cmd = self.device_login_cmd()
        try:
            proc = run(cmd, capture_output=True, text=True,
                       timeout=DEVICE_LOGIN_TIMEOUT, cwd=WORK_DIR)
        except Exception as exc:  # noqa: BLE001 — surface as unreadable failure
            return DeviceLoginResult(
                raw_output=redact_secrets(str(exc)),
                readable=False,
                message="The sign-in flow could not be read.",
                status="unreadable",
            )
        raw = _strip_ansi(_raw_out(proc))
        result = parse_device_login_output(raw)
        # Exit status stays local — never stored on DeviceLoginResult.
        rc = getattr(proc, "returncode", 1)
        if rc == 0:
            try:
                result.signed_in = bool(
                    self.check_signed_in(run, cred_path=cred_path))
            except Exception:
                result.signed_in = False
            result.status = "complete" if result.signed_in else "failed"
            if not result.signed_in and result.message is None:
                result.message = "Sign-in finished but the seat is still signed out."
            # On success with a readable parse, do not carry raw CLI text
            # (may contain noise); URL + code are enough.
            if result.readable:
                result.raw_output = ""
            else:
                # Exit 0 but unreadable — still surface redacted raw.
                result.raw_output = redact_secrets(raw)
                result.message = "The sign-in flow could not be read."
                result.status = "unreadable"
        else:
            if result.readable:
                result.status = "failed"
                result.message = result.message or "Sign-in did not complete."
                result.raw_output = ""
            else:
                result.status = "unreadable"
                result.raw_output = redact_secrets(raw)
                result.message = "The sign-in flow could not be read."
            result.signed_in = False
        return result


def _raw_out(proc) -> str:
    return (proc.stdout or "").strip() or (proc.stderr or "").strip()


def _estimate_tokens(prompt: str, reply: str) -> int:
    return (len(prompt) + len(reply)) // 4  # ~4 chars per token


def _strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def redact_secrets(text: str) -> str:
    """Strip credential values from text before any UI or log path sees it.

    Label deny-list first, then shape-based fallbacks (Bearer, JWT, long
    high-entropy runs) so unlabeled tokens cannot reach the browser.
    """
    if not text:
        return ""
    out = _SECRET_LABELED_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(1)}{m.group(3)}"
                  f"{m.group(4)}[REDACTED]{m.group(4)}",
        text,
    )
    out = _SECRET_BEARER_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", out)
    out = _SECRET_JWT_RE.sub("[REDACTED_JWT]", out)

    def _entropy(m: re.Match) -> str:
        s = m.group(0)
        if "REDACTED" in s:
            return s
        # Skip URL path/query tails already handled; require mixed classes so
        # plain words/ids are not blanked.
        has_alpha = any(c.isalpha() for c in s)
        has_digit = any(c.isdigit() for c in s)
        if not (has_alpha and has_digit):
            return s
        return "[REDACTED]"

    out = _SECRET_ENTROPY_RE.sub(_entropy, out)
    return out


def _pick_device_url(text: str) -> Optional[str]:
    """Prefer a device/verification URL over a general docs/support link."""
    found = _DEVICE_URL_RE.findall(text or "")
    if not found:
        return None
    cleaned = [u.rstrip(".,);]'\"") for u in found]
    for u in cleaned:
        low = u.lower()
        if any(h in low for h in _DEVICE_URL_HINTS):
            return u
    return cleaned[0]


def parse_device_login_output(text: str) -> DeviceLoginResult:
    """Defensive parse of device-code CLI text.

    Prefers a device/verification-looking https URL over a general one, and
    any short code-shaped token. When neither is found, readable is False and
    raw_output (redacted) is kept so the room can show it instead of silence.
    Address without a code still surfaces the URL plus a plain message.
    """
    cleaned = _strip_ansi(text or "")
    url = _pick_device_url(cleaned)
    code = None
    m_code = _DEVICE_CODE_LABELED_RE.search(cleaned)
    if m_code:
        code = m_code.group(1)
    else:
        m_hy = _DEVICE_CODE_HYPHEN_RE.search(cleaned)
        if m_hy:
            code = m_hy.group(1)
    readable = bool(url or code)
    raw = "" if readable else redact_secrets(cleaned)
    if not readable:
        message = "The sign-in flow could not be read."
    elif url and not code:
        message = "The code could not be read. Open the link below."
    else:
        message = None
    return DeviceLoginResult(
        verification_url=url,
        user_code=code,
        raw_output=raw,
        readable=readable,
        message=message,
        status="awaiting_user" if readable else "unreadable",
    )


class ClaudeAdapter(EngineAdapter):
    wants_window_on_new_session = False

    def check_signed_in(
        self,
        run: RunFn,
        cred_path: Optional[Path] = None,
    ) -> bool:
        # Official probe: `claude auth status --json` → {"loggedIn": bool}.
        _ = cred_path
        cmd = [CLAUDE_BIN, "auth", "status", "--json"]
        try:
            proc = run(cmd, capture_output=True, text=True,
                       timeout=AUTH_TIMEOUT, cwd=WORK_DIR)
        except Exception:
            return False
        if getattr(proc, "returncode", 1) != 0:
            return False
        raw = (getattr(proc, "stdout", None) or "") or (
            getattr(proc, "stderr", None) or "")
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return False
        return bool(isinstance(data, dict) and data.get("loggedIn") is True)

    def call(
        self,
        seat: dict,
        prompt: str,
        session_id: Optional[str],
        run: RunFn,
        clear_session: Optional[ClearSessionFn] = None,
    ) -> EngineResult:
        requested = seat.get("model") or ""
        cmd = [CLAUDE_BIN, "-p", prompt, "--output-format", "json",
               "--permission-mode", "acceptEdits"]
        if seat.get("model"):
            cmd += ["--model", seat["model"]]
        if seat.get("effort"):
            cmd += ["--effort", seat["effort"]]
        if session_id:
            cmd += ["--resume", session_id]

        proc = run(cmd, capture_output=True, text=True,
                   timeout=ENGINE_TIMEOUT, cwd=WORK_DIR)
        raw = _raw_out(proc)
        effective_sid = session_id

        # A stored session can go stale. Resume failure -> drop it BEFORE the
        # retry so a timeout/raise on the retry still leaves the stale id gone
        # (original set_session(seat, None) ran first).
        if proc.returncode != 0 and session_id:
            if clear_session is not None:
                clear_session()
            effective_sid = None
            cmd = [c for c in cmd if c not in ("--resume", session_id)]
            proc = run(cmd, capture_output=True, text=True,
                       timeout=ENGINE_TIMEOUT, cwd=WORK_DIR)
            raw = _raw_out(proc)

        err = None if proc.returncode == 0 else (raw or f"exit {proc.returncode}")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Preserve pre-refactor: no .strip() after ANSI.sub on this path.
            return EngineResult(
                reply=_strip_ansi(raw),
                session_id=effective_sid,
                requested_model=requested,
                actual_model=requested,
                usage_tokens=None,
                estimated=False,
                error=err,
            )

        # Capture session id immediately after JSON parse (original set_session
        # ran before usage/result handling). Later payload faults must not drop it.
        if data.get("session_id"):
            effective_sid = data["session_id"]

        # Model + reply before usage math — usage fields can be non-numeric.
        reported = data.get("model")
        actual = reported if isinstance(reported, str) and reported else requested
        try:
            # EngineResult.reply contract: ANSI-stripped even on successful JSON.
            reply = _strip_ansi(str(data.get("result") or "")).strip()
        except Exception:
            reply = _strip_ansi(raw)

        used: Optional[int] = None
        try:
            u = data.get("usage") or {}
            if not isinstance(u, dict):
                u = {}
            total = sum(u.get(k, 0) for k in
                        ("input_tokens", "cache_read_input_tokens",
                         "cache_creation_input_tokens", "output_tokens"))
            if isinstance(total, int) and total:
                used = total
        except (TypeError, ValueError, AttributeError):
            used = None

        return EngineResult(
            reply=reply,
            session_id=effective_sid,
            requested_model=requested,
            actual_model=actual,
            usage_tokens=used,
            estimated=False,
            error=err,
        )


class GrokAdapter(EngineAdapter):
    wants_window_on_new_session = True
    supports_device_login = True

    def device_login_cmd(self) -> list:
        # Headless device-code flow; alias --device-code also works.
        return [GROK_BIN, "login", "--device-auth"]

    def check_signed_in(
        self,
        run: RunFn,
        cred_path: Optional[Path] = None,
    ) -> bool:
        # `grok models` prints "You are logged in with grok.com." on success.
        # Spends no model tokens. Near-expiry is fine — the CLI refreshes itself.
        _ = cred_path
        cmd = [GROK_BIN, "models"]
        try:
            proc = run(cmd, capture_output=True, text=True,
                       timeout=AUTH_TIMEOUT, cwd=WORK_DIR)
        except Exception:
            return False
        if getattr(proc, "returncode", 1) != 0:
            return False
        out = _strip_ansi(
            (getattr(proc, "stdout", None) or "")
            or (getattr(proc, "stderr", None) or "")
        )
        # Skip blank/banner lines; match the first non-empty line (not index 0).
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            return line.startswith("You are logged in")
        return False

    def call(
        self,
        seat: dict,
        prompt: str,
        session_id: Optional[str],
        run: RunFn,
        clear_session: Optional[ClearSessionFn] = None,
    ) -> EngineResult:
        # Required flags: without --max-turns grok stops after ~2 turns; effort is always high.
        # Model via -m/--model when the seat has one (CLI gained the flag).
        # Session memory: mint UUID with -s, resume with -r.
        requested = seat.get("model") or ""
        grok_new_sid = None
        cmd = [GROK_BIN, "-p", prompt, "--max-turns", "150",
               "--always-approve", "--effort", "high"]
        if seat.get("model"):
            cmd += ["-m", seat["model"]]
        if session_id:
            cmd += ["-r", session_id]
        else:
            grok_new_sid = str(uuid.uuid4())
            cmd += ["-s", grok_new_sid]

        proc = run(cmd, capture_output=True, text=True,
                   timeout=ENGINE_TIMEOUT, cwd=WORK_DIR)
        raw = _raw_out(proc)
        effective_sid = session_id

        # Session rot self-heal: drop the stale id BEFORE the fresh retry so a
        # timeout/raise on the retry still leaves the stale id gone.
        if proc.returncode != 0 and session_id:
            if clear_session is not None:
                clear_session()
            effective_sid = None
            grok_new_sid = str(uuid.uuid4())
            cmd = [GROK_BIN, "-p", prompt, "--max-turns", "150",
                   "--always-approve", "--effort", "high"]
            if seat.get("model"):
                cmd += ["-m", seat["model"]]
            cmd += ["-s", grok_new_sid]
            proc = run(cmd, capture_output=True, text=True,
                       timeout=ENGINE_TIMEOUT, cwd=WORK_DIR)
            raw = _raw_out(proc)

        if proc.returncode == 0 and grok_new_sid:
            effective_sid = grok_new_sid

        reply = _strip_ansi(raw).strip()
        turn = _estimate_tokens(prompt, reply)
        err = None if proc.returncode == 0 else (raw or f"exit {proc.returncode}")
        return EngineResult(
            reply=reply,
            session_id=effective_sid,
            requested_model=requested,
            actual_model=requested,
            usage_tokens=turn,
            estimated=True,
            error=err,
            fresh_session=bool(grok_new_sid),
        )


class KimiAdapter(EngineAdapter):
    wants_window_on_new_session = False
    supports_device_login = True

    def device_login_cmd(self) -> list:
        # Device-code flow; no extra flags needed.
        return [KIMI_BIN, "login"]

    def check_signed_in(
        self,
        run: RunFn,
        cred_path: Optional[Path] = None,
    ) -> bool:
        # No status subcommand. Read the credential file only — never log
        # token values. Near-expiry with a refresh_token counts as signed in
        # (the CLI refreshes in the background). Missing file → signed out.
        _ = run  # file probe; process runner unused
        path = Path(cred_path) if cred_path is not None else KIMI_CRED_PATH
        try:
            raw = path.read_text()
        except (OSError, UnicodeDecodeError):
            # FileNotFoundError is an OSError subclass. UnicodeDecodeError is
            # raised by read_text() on binary/non-UTF-8 files — still signed out.
            return False
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(data, dict):
            return False
        # Refresh token present ⇒ CLI can renew; treat as signed in even if
        # access_token is near or past expiry.
        refresh = data.get("refresh_token")
        if isinstance(refresh, str) and refresh.strip():
            return True
        # No refresh: only signed in while access_token is present and unexpired.
        access = data.get("access_token")
        if not (isinstance(access, str) and access.strip()):
            return False
        expires_at = data.get("expires_at")
        if expires_at is None:
            return True  # token present, no expiry field
        try:
            return float(expires_at) > time.time()
        except (TypeError, ValueError):
            return False

    def call(
        self,
        seat: dict,
        prompt: str,
        session_id: Optional[str],
        run: RunFn,
        clear_session: Optional[ClearSessionFn] = None,
    ) -> EngineResult:
        # -p is non-interactive; memory rides on -S <session_id>, printed
        # after every reply as "To resume this session: kimi -r <id>".
        # clear_session is accepted for contract parity (kimi has no self-heal).
        _ = clear_session
        requested = seat.get("model") or ""
        if session_id:
            cmd = [KIMI_BIN, "-S", session_id, "-p", prompt]
            if seat.get("model"):
                cmd += ["-m", seat["model"]]
        else:
            cmd = [KIMI_BIN, "-p", prompt]
            if seat.get("model"):
                cmd += ["-m", seat["model"]]

        proc = run(cmd, capture_output=True, text=True,
                   timeout=ENGINE_TIMEOUT, cwd=WORK_DIR)
        raw = _raw_out(proc)
        effective_sid = session_id
        m = re.search(r"kimi -r (session_\S+)", raw)
        if m:
            effective_sid = m.group(1)
        raw = re.sub(r"To resume this session:.*", "", raw)
        raw = re.sub(r"^\s*•\s*", "", raw, flags=re.M)
        reply = _strip_ansi(raw).strip()
        turn = _estimate_tokens(prompt, reply)
        err = None if proc.returncode == 0 else (raw or f"exit {proc.returncode}")
        return EngineResult(
            reply=reply,
            session_id=effective_sid,
            requested_model=requested,
            actual_model=requested,
            usage_tokens=turn,
            estimated=True,
            error=err,
        )


ADAPTERS: dict[str, EngineAdapter] = {
    "claude": ClaudeAdapter(),
    "grok": GrokAdapter(),
    "kimi": KimiAdapter(),
}
