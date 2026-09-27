"""Unit tests for engines.py (Agent Room common session contract).

stdlib only. Fake `run` callables per case — no real CLI, no network.
"""
from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOM = Path(__file__).resolve().parent
if str(ROOM) not in sys.path:
    sys.path.insert(0, str(ROOM))

from engines import (  # noqa: E402
    ADAPTERS,
    ENGINE_TIMEOUT,
    WORK_DIR,
    ClaudeAdapter,
    EngineResult,
    GrokAdapter,
    KimiAdapter,
    _estimate_tokens,
    parse_device_login_output,
    redact_secrets,
)


def _proc(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class FakeRun:
    """Callable that returns scripted CompletedProcess-like objects in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []  # list of (cmd, kwargs)

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), kwargs))
        if not self.responses:
            raise AssertionError(f"unexpected run() call: {cmd!r}")
        return self.responses.pop(0)


def _assert_runner_timeout_cwd(test: unittest.TestCase, run: FakeRun) -> None:
    """Adapters own timeout/cwd; every real subprocess.run call must pass both."""
    test.assertTrue(run.calls, "expected at least one run() call")
    for _cmd, kwargs in run.calls:
        test.assertEqual(kwargs.get("timeout"), ENGINE_TIMEOUT)
        test.assertEqual(kwargs.get("cwd"), WORK_DIR)
        test.assertTrue(kwargs.get("capture_output"))
        test.assertTrue(kwargs.get("text"))


class TestClaudeAdapter(unittest.TestCase):
    def test_happy_path_json_session_and_real_usage(self):
        # Requested model differs from reported model — actual_model must follow report.
        payload = {
            "result": "hello from claude",
            "session_id": "sess-abc",
            "model": "claude-opus-5",
            "usage": {
                "input_tokens": 100,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 5,
                "output_tokens": 50,
            },
        }
        run = FakeRun(_proc(stdout=json.dumps(payload)))
        seat = {"engine": "claude", "model": "claude-sonnet-5", "effort": "high"}
        result = ClaudeAdapter().call(seat, "hi", None, run)

        self.assertEqual(result.reply, "hello from claude")
        self.assertEqual(result.session_id, "sess-abc")
        self.assertEqual(result.requested_model, "claude-sonnet-5")
        self.assertEqual(result.actual_model, "claude-opus-5")  # report wins
        self.assertEqual(result.usage_tokens, 175)  # 100+20+5+50
        self.assertFalse(result.estimated)
        self.assertIsNone(result.error)
        # cmd shape
        cmd = run.calls[0][0]
        self.assertIn("--output-format", cmd)
        self.assertIn("json", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("claude-sonnet-5", cmd)
        self.assertIn("--effort", cmd)
        self.assertNotIn("--resume", cmd)
        _assert_runner_timeout_cwd(self, run)

    def test_json_reply_is_ansi_stripped(self):
        payload = {
            "result": "\x1b[31mred\x1b[0m ok",
            "session_id": "s1",
            "model": "m-reported",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        run = FakeRun(_proc(stdout=json.dumps(payload)))
        result = ClaudeAdapter().call(
            {"engine": "claude", "model": "m-req"}, "p", None, run)
        self.assertEqual(result.reply, "red ok")
        self.assertNotIn("\x1b", result.reply)
        self.assertEqual(result.actual_model, "m-reported")

    def test_stale_resume_self_heal(self):
        good = {
            "result": "recovered",
            "session_id": "sess-new",
            "model": "claude-sonnet-5",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        run = FakeRun(
            _proc(stdout="resume failed", stderr="no such session", returncode=1),
            _proc(stdout=json.dumps(good)),
        )
        cleared = []
        seat = {"engine": "claude", "model": "claude-sonnet-5"}
        result = ClaudeAdapter().call(
            seat, "prompt", "stale-sid", run,
            clear_session=lambda: cleared.append("cleared"),
        )

        self.assertEqual(len(run.calls), 2)
        first_cmd, second_cmd = run.calls[0][0], run.calls[1][0]
        self.assertIn("--resume", first_cmd)
        self.assertIn("stale-sid", first_cmd)
        self.assertNotIn("--resume", second_cmd)
        self.assertNotIn("stale-sid", second_cmd)
        self.assertEqual(result.reply, "recovered")
        self.assertEqual(result.session_id, "sess-new")
        self.assertEqual(result.usage_tokens, 15)
        self.assertFalse(result.estimated)
        self.assertEqual(cleared, ["cleared"])
        _assert_runner_timeout_cwd(self, run)

    def test_early_clear_before_retry_timeout(self):
        """Stale id must be dropped BEFORE the retry — even if retry times out."""
        order = []

        def clear():
            order.append("clear")

        def run(cmd, **kwargs):
            order.append(("run", list(cmd), kwargs.get("timeout"), kwargs.get("cwd")))
            if len([x for x in order if isinstance(x, tuple)]) == 1:
                return _proc(stdout="", stderr="stale", returncode=1)
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

        with self.assertRaises(subprocess.TimeoutExpired):
            ClaudeAdapter().call(
                {"engine": "claude", "model": "m"}, "p", "stale-sid", run,
                clear_session=clear,
            )
        self.assertEqual(order[0][0], "run")  # first attempt
        self.assertEqual(order[1], "clear")   # clear BEFORE retry
        self.assertEqual(order[2][0], "run")  # retry that times out
        self.assertNotIn("--resume", order[2][1])

    def test_session_captured_despite_malformed_usage(self):
        # usage values that make sum() raise TypeError — session + model + reply
        # still returned (original set_session ran before usage handling).
        payload = {
            "result": "still useful",
            "session_id": "sess-kept",
            "model": "reported-model",
            "usage": {"input_tokens": "not-a-number", "output_tokens": 3},
        }
        run = FakeRun(_proc(stdout=json.dumps(payload)))
        result = ClaudeAdapter().call(
            {"engine": "claude", "model": "req-model"}, "p", None, run)
        self.assertEqual(result.session_id, "sess-kept")
        self.assertEqual(result.reply, "still useful")
        self.assertEqual(result.actual_model, "reported-model")
        self.assertIsNone(result.usage_tokens)
        self.assertEqual(result.requested_model, "req-model")


class TestGrokAdapter(unittest.TestCase):
    def test_fresh_session_mints_uuid(self):
        run = FakeRun(_proc(stdout="grok says hi"))
        seat = {"engine": "grok", "model": "grok-4.5"}
        result = GrokAdapter().call(seat, "build it", None, run)

        cmd = run.calls[0][0]
        self.assertIn("-s", cmd)
        self.assertNotIn("-r", cmd)
        sid_idx = cmd.index("-s") + 1
        minted = cmd[sid_idx]
        # valid uuid
        uuid.UUID(minted)
        self.assertEqual(result.session_id, minted)
        self.assertTrue(result.fresh_session)
        self.assertTrue(result.estimated)
        self.assertEqual(result.reply, "grok says hi")
        # required flags
        self.assertIn("--max-turns", cmd)
        self.assertIn("150", cmd)
        self.assertIn("--always-approve", cmd)
        self.assertIn("--effort", cmd)
        self.assertIn("high", cmd)
        _assert_runner_timeout_cwd(self, run)

    def test_failed_resume_heals_to_fresh_uuid(self):
        run = FakeRun(
            _proc(stdout="", stderr="session gone", returncode=1),
            _proc(stdout="healed reply"),
        )
        cleared = []
        seat = {"engine": "grok"}
        result = GrokAdapter().call(
            seat, "again", "old-grok-sid", run,
            clear_session=lambda: cleared.append(1),
        )

        self.assertEqual(len(run.calls), 2)
        first, second = run.calls[0][0], run.calls[1][0]
        self.assertIn("-r", first)
        self.assertIn("old-grok-sid", first)
        self.assertIn("-s", second)
        self.assertNotIn("-r", second)
        new_sid = second[second.index("-s") + 1]
        uuid.UUID(new_sid)
        self.assertEqual(result.session_id, new_sid)
        self.assertTrue(result.fresh_session)
        self.assertEqual(result.reply, "healed reply")
        self.assertEqual(cleared, [1])
        _assert_runner_timeout_cwd(self, run)

    def test_early_clear_before_retry_timeout(self):
        order = []

        def clear():
            order.append("clear")

        def run(cmd, **kwargs):
            order.append("run")
            if order.count("run") == 1:
                return _proc(stdout="", stderr="gone", returncode=1)
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

        with self.assertRaises(subprocess.TimeoutExpired):
            GrokAdapter().call(
                {"engine": "grok"}, "p", "old-g", run, clear_session=clear)
        self.assertEqual(order, ["run", "clear", "run"])


class TestKimiAdapter(unittest.TestCase):
    def test_session_id_extracted_and_resume_line_stripped(self):
        raw = (
            "• real answer here\n"
            "To resume this session: kimi -r session_abc123xyz\n"
        )
        run = FakeRun(_proc(stdout=raw))
        seat = {"engine": "kimi", "model": "kimi-k3"}
        result = KimiAdapter().call(seat, "review this", None, run)

        self.assertEqual(result.session_id, "session_abc123xyz")
        self.assertNotIn("To resume", result.reply)
        self.assertNotIn("session_abc123xyz", result.reply)
        self.assertNotIn("•", result.reply)
        self.assertIn("real answer here", result.reply)
        self.assertTrue(result.estimated)
        cmd = run.calls[0][0]
        self.assertIn("-p", cmd)
        self.assertIn("-m", cmd)
        self.assertIn("kimi-k3", cmd)
        _assert_runner_timeout_cwd(self, run)

    def test_resume_uses_dash_S_and_model_flag(self):
        """Resume path: -S before -p, and -m model when seat has a model."""
        run = FakeRun(_proc(stdout="continued\nTo resume this session: kimi -r session_old\n"))
        seat = {"engine": "kimi", "model": "kimi-k3"}
        result = KimiAdapter().call(seat, "more", "session_old", run)
        cmd = run.calls[0][0]
        self.assertIn("-S", cmd)
        self.assertIn("session_old", cmd)
        self.assertIn("-p", cmd)
        self.assertIn("-m", cmd)
        self.assertIn("kimi-k3", cmd)
        # Flag order: binary, -S, sid, -p, prompt, -m, model
        self.assertEqual(cmd[1], "-S")
        self.assertEqual(cmd[2], "session_old")
        self.assertEqual(cmd[3], "-p")
        self.assertEqual(cmd[4], "more")
        m_idx = cmd.index("-m")
        self.assertEqual(cmd[m_idx + 1], "kimi-k3")
        self.assertGreater(m_idx, cmd.index("-p"))
        self.assertEqual(result.session_id, "session_old")
        _assert_runner_timeout_cwd(self, run)


class TestAuthProbes(unittest.TestCase):
    """DM-06: signed-in / signed-out per engine, no model generation."""

    def test_claude_signed_in_from_status_json(self):
        run = FakeRun(_proc(stdout=json.dumps({"loggedIn": True})))
        self.assertTrue(ClaudeAdapter().check_signed_in(run))
        cmd = run.calls[0][0]
        self.assertEqual(cmd[1:4], ["auth", "status", "--json"])
        # Status probe only — never a generation prompt.
        self.assertNotIn("-p", cmd)

    def test_claude_signed_out_when_loggedIn_false(self):
        run = FakeRun(_proc(stdout=json.dumps({"loggedIn": False})))
        self.assertFalse(ClaudeAdapter().check_signed_in(run))

    def test_claude_signed_out_on_failed_probe(self):
        run = FakeRun(_proc(stdout="", stderr="not logged in", returncode=1))
        self.assertFalse(ClaudeAdapter().check_signed_in(run))

    def test_grok_signed_in_from_models_login_line(self):
        run = FakeRun(_proc(stdout="You are logged in with grok.com.\n\nDefault model: grok-4.5\n"))
        self.assertTrue(GrokAdapter().check_signed_in(run))
        cmd = run.calls[0][0]
        self.assertEqual(cmd[1], "models")
        self.assertNotIn("-p", cmd)

    def test_grok_signed_in_despite_ansi_and_leading_blank(self):
        # Colour codes + a blank banner line before the login phrase must still
        # count as signed in (every other engines.py path strips ANSI first).
        out = (
            "\n"
            "\x1b[32mYou are logged in with grok.com.\x1b[0m\n"
            "\n"
            "Default model: grok-4.5\n"
        )
        run = FakeRun(_proc(stdout=out))
        self.assertTrue(GrokAdapter().check_signed_in(run))

    def test_grok_signed_out_when_models_fails(self):
        run = FakeRun(_proc(stdout="", stderr="not authenticated", returncode=1))
        self.assertFalse(GrokAdapter().check_signed_in(run))

    def test_kimi_signed_in_when_refresh_token_present_near_expiry(self):
        # expires_at is already past; refresh_token still present → signed in.
        import tempfile
        import time
        payload = {
            "access_token": "tok-redacted",
            "refresh_token": "ref-redacted",
            "expires_at": int(time.time()) - 60,  # already expired
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kimi-code.json"
            path.write_text(json.dumps(payload))
            run = FakeRun()  # must not be called for a file-only probe
            self.assertTrue(KimiAdapter().check_signed_in(run, cred_path=path))
            self.assertEqual(run.calls, [])

    def test_kimi_signed_out_when_cred_file_missing(self):
        missing = Path("/tmp/agent-room-no-such-kimi-creds.json")
        if missing.exists():
            missing.unlink()
        run = FakeRun()
        # Missing file → signed out, never raises.
        self.assertFalse(
            KimiAdapter().check_signed_in(run, cred_path=missing))
        self.assertEqual(run.calls, [])

    def test_kimi_signed_out_when_expired_and_no_refresh(self):
        import tempfile
        import time
        payload = {
            "access_token": "tok-redacted",
            "expires_at": int(time.time()) - 60,
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kimi-code.json"
            path.write_text(json.dumps(payload))
            self.assertFalse(
                KimiAdapter().check_signed_in(FakeRun(), cred_path=path))

    def test_kimi_signed_out_on_non_utf8_cred_file(self):
        # Binary / non-UTF-8 credentials are malformed, not a crash.
        # check_signed_in must return False and must not raise.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kimi-code.json"
            path.write_bytes(b"\xff\xfe\x00\x01 binary junk \x80\x81")
            run = FakeRun()
            try:
                result = KimiAdapter().check_signed_in(run, cred_path=path)
            except Exception as exc:  # noqa: BLE001 — assert no raise
                self.fail(f"check_signed_in raised on binary creds: {exc!r}")
            self.assertFalse(result)
            self.assertEqual(run.calls, [])

    def _load_server_module(self, name: str):
        import importlib.util
        if name in sys.modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, ROOM / "server.py")
        server = importlib.util.module_from_spec(spec)
        with mock.patch("http.server.ThreadingHTTPServer"):
            spec.loader.exec_module(server)
        return server

    def test_server_auth_endpoint_returns_all_three(self):
        """Asking the room for auth state answers claude, grok, and kimi."""
        server = self._load_server_module("agent_room_server_auth_probe_test")
        server.clear_auth_cache()

        def fake_check(self, run, cred_path=None):
            return True

        with mock.patch.object(ClaudeAdapter, "check_signed_in", fake_check), \
             mock.patch.object(GrokAdapter, "check_signed_in", fake_check), \
             mock.patch.object(KimiAdapter, "check_signed_in", fake_check):
            state = server.auth_state(run=FakeRun())
        self.assertEqual(set(state.keys()), {"claude", "grok", "kimi"})
        for eng in ("claude", "grok", "kimi"):
            self.assertIn("signed_in", state[eng])
            self.assertIsInstance(state[eng]["signed_in"], bool)
            # Must be True when the probe returns True. A vacuous bool check
            # hides broken wiring (wrong kwarg, wrong ADAPTERS key, raise).
            self.assertIs(state[eng]["signed_in"], True)

    def test_server_auth_probe_raise_isolates_engines(self):
        """One engine probe raising → that engine signed_out; others intact."""
        server = self._load_server_module(
            "agent_room_server_auth_probe_raise_test")
        server.clear_auth_cache()

        def ok_check(self, run, cred_path=None):
            return True

        def boom_check(self, run, cred_path=None):
            raise RuntimeError("probe exploded")

        with mock.patch.object(ClaudeAdapter, "check_signed_in", boom_check), \
             mock.patch.object(GrokAdapter, "check_signed_in", ok_check), \
             mock.patch.object(KimiAdapter, "check_signed_in", ok_check):
            state = server.auth_state(run=FakeRun())
        self.assertIs(state["claude"]["signed_in"], False)
        self.assertIs(state["grok"]["signed_in"], True)
        self.assertIs(state["kimi"]["signed_in"], True)

    def test_auth_state_requires_run(self):
        """Finding 8: run= is required — no silent default to live subprocess."""
        server = self._load_server_module("agent_room_server_auth_run_required")
        sig = inspect.signature(server.auth_state)
        param = sig.parameters["run"]
        self.assertIs(param.default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            server.auth_state()  # noqa: missing run

    def test_auth_cache_skips_second_probe_within_ttl(self):
        """Finding 3: two GETs inside the cache window spawn no second probe."""
        server = self._load_server_module("agent_room_server_auth_cache")
        server.clear_auth_cache()
        calls: list[str] = []

        def fake_check(self, run, cred_path=None):
            calls.append(type(self).__name__)
            return True

        with mock.patch.object(ClaudeAdapter, "check_signed_in", fake_check), \
             mock.patch.object(GrokAdapter, "check_signed_in", fake_check), \
             mock.patch.object(KimiAdapter, "check_signed_in", fake_check):
            a = server.auth_state(run=FakeRun())
            b = server.auth_state(run=FakeRun())
        self.assertEqual(len(calls), 3, f"second call re-probed: {calls}")
        self.assertEqual(a, b)
        self.assertGreaterEqual(server.AUTH_CACHE_TTL_SEC, 30)
        self.assertLessEqual(server.AUTH_CACHE_TTL_SEC, 60)

    def test_auth_force_bypasses_cache(self):
        """Finding 3: re-check (force=True) re-probes even inside the TTL."""
        server = self._load_server_module("agent_room_server_auth_force")
        server.clear_auth_cache()
        calls: list[str] = []

        def fake_check(self, run, cred_path=None):
            calls.append(type(self).__name__)
            return True

        with mock.patch.object(ClaudeAdapter, "check_signed_in", fake_check), \
             mock.patch.object(GrokAdapter, "check_signed_in", fake_check), \
             mock.patch.object(KimiAdapter, "check_signed_in", fake_check):
            server.auth_state(run=FakeRun())
            self.assertEqual(len(calls), 3)
            server.auth_state(run=FakeRun(), force=True)
            self.assertEqual(len(calls), 6, "force=True must re-probe all three")


class TestDeviceLogin(unittest.TestCase):
    """DM-07 / T02: grok and kimi device-code sign-in from the room."""

    GROK_SAMPLE = (
        "To sign in, open https://auth.x.ai/device in your browser\n"
        "and enter code: WXYZ-1234\n"
        "Waiting for authorization...\n"
    )
    KIMI_SAMPLE = (
        "Visit https://www.kimi.com/code/device\n"
        "Enter the code AB12-CD34 to continue.\n"
    )

    def test_parse_surfaces_url_and_code(self):
        """Criterion 1: first https URL + short code from device-flow text."""
        info = parse_device_login_output(self.GROK_SAMPLE)
        self.assertEqual(info.verification_url, "https://auth.x.ai/device")
        self.assertEqual(info.user_code, "WXYZ-1234")
        self.assertTrue(info.readable)

    def test_grok_run_device_login_surfaces_url_and_code(self):
        """Criterion 1 via adapter seam: login cmd + parsed URL and code."""
        run = FakeRun(_proc(stdout=self.GROK_SAMPLE, returncode=0),
                      _proc(stdout="You are logged in with grok.com.\n"))
        result = GrokAdapter().run_device_login(run)
        self.assertEqual(result.verification_url, "https://auth.x.ai/device")
        self.assertEqual(result.user_code, "WXYZ-1234")
        cmd = run.calls[0][0]
        self.assertEqual(cmd[1:3], ["login", "--device-auth"])
        # Never a model generation prompt on the login call.
        self.assertNotIn("-p", cmd)

    def test_kimi_run_device_login_surfaces_url_and_code(self):
        """Criterion 1 for kimi: `kimi login` + parsed URL and code."""
        # Success re-probe is file-based; unused FakeRun responses after login.
        run = FakeRun(_proc(stdout=self.KIMI_SAMPLE, returncode=0))
        result = KimiAdapter().run_device_login(run)
        self.assertEqual(result.verification_url,
                         "https://www.kimi.com/code/device")
        self.assertEqual(result.user_code, "AB12-CD34")
        cmd = run.calls[0][0]
        self.assertEqual(cmd[1], "login")
        self.assertNotIn("-p", cmd)

    def test_unreadable_output_exposes_raw_not_silent(self):
        """Criterion 2: no URL/code → raw text + plain message, not silence."""
        raw = "something went wrong: connection refused (no device UI)"
        run = FakeRun(_proc(stdout=raw, returncode=1))
        result = GrokAdapter().run_device_login(run)
        self.assertFalse(result.readable)
        self.assertIsNone(result.verification_url)
        self.assertIsNone(result.user_code)
        self.assertIn("connection refused", result.raw_output)
        self.assertIsNotNone(result.message)
        self.assertIn("could not be read", result.message.lower())

    def test_success_reprobes_and_reports_signed_in(self):
        """Criterion 3: exit 0 → re-probe → signed_in True."""
        run = FakeRun(
            _proc(stdout=self.GROK_SAMPLE, returncode=0),
            _proc(stdout="You are logged in with grok.com.\n"),
        )
        result = GrokAdapter().run_device_login(run)
        self.assertTrue(result.signed_in)
        self.assertEqual(result.status, "complete")
        # Login then models probe — two calls, both non-generation.
        self.assertEqual(len(run.calls), 2)
        self.assertEqual(run.calls[1][0][1], "models")

    def test_no_token_fields_on_device_login_result(self):
        """Criterion 4: result surface has no credential/token values."""
        # Even if CLI dumps a token-looking line, we never put it on the result
        # as a credential field; only URL + short code (and redacted raw).
        nasty = (
            self.GROK_SAMPLE
            + "access_token=sekrit-access-value\n"
            + "refresh_token=sekrit-refresh-value\n"
        )
        run = FakeRun(
            _proc(stdout=nasty, returncode=0),
            _proc(stdout="You are logged in with grok.com.\n"),
        )
        result = GrokAdapter().run_device_login(run)
        self.assertEqual(result.verification_url, "https://auth.x.ai/device")
        self.assertEqual(result.user_code, "WXYZ-1234")
        # No attribute that holds a raw token value.
        for name in ("access_token", "refresh_token", "token", "credential"):
            self.assertFalse(hasattr(result, name))
        blob = json.dumps({
            "verification_url": result.verification_url,
            "user_code": result.user_code,
            "raw_output": result.raw_output,
            "message": result.message,
            "signed_in": result.signed_in,
            "status": result.status,
        })
        self.assertNotIn("sekrit-access-value", blob)
        self.assertNotIn("sekrit-refresh-value", blob)

    def test_unreadable_path_redacts_token_keeps_surroundings(self):
        """Finding 3: redaction must run on the unreadable path (raw populated).

        Readable path blanks raw_output, so a leak test there is vacuous.
        Nonzero exit, no URL, no code, token lines present — token gone,
        surrounding diagnostic text still present.
        """
        raw = (
            "device flow broken: connection refused\n"
            "access_token=sekrit-access-xyz99\n"
            "please retry later\n"
        )
        run = FakeRun(_proc(stdout=raw, returncode=1))
        result = GrokAdapter().run_device_login(run)
        self.assertFalse(result.readable)
        self.assertTrue(result.raw_output)
        self.assertIn("connection refused", result.raw_output)
        self.assertIn("please retry later", result.raw_output)
        self.assertNotIn("sekrit-access-xyz99", result.raw_output)
        self.assertIn("[REDACTED]", result.raw_output)

    def test_error_code_timeout_not_parsed_as_user_code(self):
        """Finding 5: label case-fold only; 'error code: timeout' is unreadable."""
        info = parse_device_login_output("error code: timeout\n")
        self.assertFalse(info.readable)
        self.assertIsNone(info.user_code)
        self.assertIsNone(info.verification_url)

    def test_prefers_device_url_over_docs_link(self):
        """Finding 6: multi-URL — prefer device/verification over docs."""
        text = (
            "See docs at https://docs.x.ai/help/login\n"
            "Then open https://auth.x.ai/device\n"
            "code: ABCD-EF12\n"
        )
        info = parse_device_login_output(text)
        self.assertEqual(info.verification_url, "https://auth.x.ai/device")
        self.assertEqual(info.user_code, "ABCD-EF12")
        self.assertTrue(info.readable)

    def test_url_without_code_says_code_unreadable(self):
        """Finding 10: address alone still warns that the code could not be read."""
        text = "Open https://auth.x.ai/device to continue\n"
        info = parse_device_login_output(text)
        self.assertTrue(info.readable)
        self.assertEqual(info.verification_url, "https://auth.x.ai/device")
        self.assertIsNone(info.user_code)
        self.assertIsNotNone(info.message)
        self.assertIn("code could not be read", info.message.lower())

    def _load_server_module(self, name: str):
        import importlib.util
        if name in sys.modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, ROOM / "server.py")
        server = importlib.util.module_from_spec(spec)
        with mock.patch("http.server.ThreadingHTTPServer"):
            spec.loader.exec_module(server)
        return server

    def test_server_start_device_login_surfaces_url_and_code(self):
        """Criterion 1 at the server auth handler seam (blocking path)."""
        server = self._load_server_module(
            "agent_room_server_device_login_test")
        run = FakeRun(
            _proc(stdout=self.GROK_SAMPLE, returncode=0),
            _proc(stdout="You are logged in with grok.com.\n"),
        )
        # Blocking helper for tests: drive adapter to completion with FakeRun.
        out = server.start_device_login("grok", run=run, background=False)
        self.assertEqual(out.get("verification_url"),
                         "https://auth.x.ai/device")
        self.assertEqual(out.get("user_code"), "WXYZ-1234")
        self.assertTrue(out.get("signed_in"))

    def test_server_rejects_claude_device_login(self):
        """Claude sign-in is T03 — server must not attempt a device flow."""
        server = self._load_server_module(
            "agent_room_server_device_login_claude_test")
        out = server.start_device_login("claude", run=FakeRun(),
                                       background=False)
        self.assertIn("error", out)
        # Must not have invoked any process runner.
        # (FakeRun with no responses would raise if called.)

    def test_hung_login_times_out_and_is_restartable(self):
        """Finding 1: hung CLI → failed status; Sign in / Try again works again."""
        import time

        server = self._load_server_module(
            "agent_room_server_hung_login_test")
        server.LOGIN_FLOWS.clear()

        class HungPopen:
            def __init__(self, *args, **kwargs):
                self.stdout = self
                self.returncode = None
                self.killed = False
                self._release = False

            def __iter__(self):
                return self

            def __next__(self):
                # Block until kill() — simulates a CLI that never exits or prints.
                while not self._release:
                    time.sleep(0.02)
                raise StopIteration

            def kill(self):
                self.killed = True
                self._release = True
                self.returncode = -9

            def wait(self, timeout=None):
                self._release = True
                if self.returncode is None:
                    self.returncode = -9
                return self.returncode

            def poll(self):
                return self.returncode

        pops = []

        def fake_popen(*args, **kwargs):
            p = HungPopen(*args, **kwargs)
            pops.append(p)
            return p

        out = server.start_device_login(
            "grok",
            run=FakeRun(),
            background=True,
            popen=fake_popen,
            timeout=0.35,
        )
        self.assertEqual(out.get("status"), "starting")

        status = None
        deadline = time.time() + 3.0
        while time.time() < deadline:
            with server.LOGIN_FLOW_LOCK:
                flow = server.LOGIN_FLOWS.get("grok") or {}
                status = flow.get("status")
                msg = flow.get("message") or ""
            if status == "failed":
                break
            time.sleep(0.05)
        self.assertEqual(status, "failed", "hung login must expire to failed")
        self.assertIn("timed out", msg.lower())
        self.assertTrue(pops, "worker must have started the CLI")
        self.assertTrue(pops[0].killed, "hung child must be killed")

        # Recoverable from the page: a new start must not be blocked.
        run = FakeRun(
            _proc(stdout=self.GROK_SAMPLE, returncode=0),
            _proc(stdout="You are logged in with grok.com.\n"),
        )
        out2 = server.start_device_login("grok", run=run, background=False)
        self.assertEqual(out2.get("user_code"), "WXYZ-1234")
        self.assertTrue(out2.get("signed_in"))

    def test_background_worker_streams_url_before_exit(self):
        """Finding 7: production streaming path surfaces URL while CLI still open."""
        import time

        server = self._load_server_module(
            "agent_room_server_stream_login_test")
        server.LOGIN_FLOWS.clear()
        popen_kwargs = []

        class StreamingPopen:
            def __init__(self, *args, **kwargs):
                popen_kwargs.append(kwargs)
                self.returncode = None
                self._lines = [
                    "Open https://auth.x.ai/device\n",
                    "enter code: STRM-9999\n",
                ]
                self._i = 0
                self.stdout = self
                self._closed = False

            def __iter__(self):
                return self

            def __next__(self):
                if self._i < len(self._lines):
                    line = self._lines[self._i]
                    self._i += 1
                    return line
                # Hang after URL/code so worker must surface them pre-exit.
                while self.returncode is None:
                    time.sleep(0.02)
                raise StopIteration

            def kill(self):
                self.returncode = -9

            def wait(self, timeout=None):
                if self.returncode is None:
                    self.returncode = 0
                return self.returncode

            def poll(self):
                return self.returncode

        server.start_device_login(
            "grok",
            run=FakeRun(
                _proc(stdout="You are logged in with grok.com.\n"),
            ),
            background=True,
            popen=StreamingPopen,
            timeout=2.0,
        )

        got = None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with server.LOGIN_FLOW_LOCK:
                flow = dict(server.LOGIN_FLOWS.get("grok") or {})
            if flow.get("status") == "awaiting_user" and flow.get("user_code"):
                got = flow
                break
            time.sleep(0.03)
        self.assertIsNotNone(got, "URL/code must surface before process exit")
        self.assertEqual(got.get("verification_url"), "https://auth.x.ai/device")
        self.assertEqual(got.get("user_code"), "STRM-9999")
        self.assertTrue(popen_kwargs)
        self.assertEqual(popen_kwargs[0].get("bufsize"), 1)

        # Unblock the fake so the worker can finish cleanly.
        with server.LOGIN_FLOW_LOCK:
            # Find live process via closing the hang: wait path sets returncode.
            pass
        # Kill via a second start is blocked while awaiting_user; expire instead
        # by waiting for the short timeout or force-clear for isolation.
        server.LOGIN_FLOWS.clear()

    def test_popen_without_stdout_fails_cleanly(self):
        """Finding 8: no assert; missing stdout → failed, restartable."""
        import time

        server = self._load_server_module(
            "agent_room_server_no_stdout_login_test")
        server.LOGIN_FLOWS.clear()

        class NoStdoutPopen:
            def __init__(self, *args, **kwargs):
                self.stdout = None
                self.returncode = None
                self.killed = False

            def kill(self):
                self.killed = True
                self.returncode = -9

            def wait(self, timeout=None):
                if self.returncode is None:
                    self.returncode = -9
                return self.returncode

            def poll(self):
                return self.returncode

        server.start_device_login(
            "grok", run=FakeRun(), background=True, popen=NoStdoutPopen,
            timeout=1.0,
        )
        status = None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with server.LOGIN_FLOW_LOCK:
                status = (server.LOGIN_FLOWS.get("grok") or {}).get("status")
            if status == "failed":
                break
            time.sleep(0.03)
        self.assertEqual(status, "failed")

    def test_html_device_login_still_grok_kimi_only(self):
        """T02 device-code cards stay grok/kimi only (unchanged by T03)."""
        html = (ROOM / "index.html").read_text()
        # renderAuthFlows still iterates only device-login engines.
        self.assertIn("for(const eng of ['grok','kimi'])", html)
        # Sign-in button still posts to the device-login endpoint.
        self.assertIn("/api/auth/login", html)
        self.assertIn("startDeviceLogin", html)


class TestClaudeBanner(unittest.TestCase):
    """DM-08 / T03: claude signed-out banner + seat signed-out for all engines.

    HTML is the seam. Behavioural checks run the real functions under node so
    a mutation that removes the feature fails — string presence alone is not
    enough (Standards round 2).
    """

    def _html(self) -> str:
        return (ROOM / "index.html").read_text()

    def _extract_js_function(self, html: str, name: str) -> str:
        marker = f"function {name}"
        idx = html.find(marker)
        self.assertGreater(idx, -1, f"{name} not found in index.html")
        brace = html.find("{", idx)
        depth = 0
        i = brace
        while i < len(html):
            if html[i] == "{":
                depth += 1
            elif html[i] == "}":
                depth -= 1
                if depth == 0:
                    return html[idx : i + 1]
            i += 1
        self.fail(f"unbalanced braces extracting {name}")

    def _run_node(self, script: str) -> str:
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            self.fail(
                f"node failed ({proc.returncode}):\n"
                f"stdout={proc.stdout}\nstderr={proc.stderr}"
            )
        return proc.stdout

    def test_html_claude_banner_names_command_and_terminal(self):
        """Criterion 1: while claude is signed out, banner names the command
        and says it needs a Terminal window."""
        html = self._html()
        # Dedicated host element the renderer fills/clears.
        self.assertIn('id="claudeAuthBanner"', html)
        self.assertIn("function renderClaudeBanner", html)
        # Exact command the user must type — not a paraphrase that omits it.
        self.assertIn("claude auth login", html)
        # Plain English: needs a Terminal (not "run it here").
        self.assertRegex(html, r"[Tt]erminal")
        # Gated on claude reporting signed out — not always-on chrome.
        # Break this gate and the banner would show while signed in.
        self.assertRegex(
            html,
            r"authByEngine\.claude[\s\S]{0,200}signed_in\s*===\s*false"
            r"|signed_in\s*===\s*false[\s\S]{0,200}authByEngine\.claude"
            r"|st\.signed_in\s*===\s*false[\s\S]{0,120}claude"
            r"|claude[\s\S]{0,80}signed_in\s*===\s*false",
        )

    def test_html_recheck_reruns_probe_not_login(self):
        """Criterion 2: re-check control re-runs the auth probe on demand."""
        html = self._html()
        # Re-check is labeled in plain English for the user.
        self.assertRegex(html, r"Check again|Re-?check|I(?:'|’)ve signed in")
        # It must call refreshAuth (GET /api/auth), never start a login.
        self.assertIn("function renderClaudeBanner", html)
        idx = html.find("function renderClaudeBanner")
        self.assertGreater(idx, 0)
        rest = html[idx:idx + 2500]
        self.assertIn("refreshAuth", rest)
        # Re-check must force-bypass the server cache (otherwise button is
        # meaningless next to the 4s poll).
        self.assertRegex(
            rest,
            r"refreshAuth\s*\(\s*\{\s*force\s*:\s*true\s*\}\s*\)",
        )
        self.assertNotIn("startDeviceLogin", rest)
        self.assertNotIn("/api/auth/login", rest)
        # refreshAuth itself must request the force query when force is set.
        raf = self._extract_js_function(html, "refreshAuth")
        self.assertIn("/api/auth?refresh=1", raf)
        self.assertIn("authReqSeq", html)
        self.assertIn("seq!==authReqSeq", raf.replace(" ", ""))

    def test_html_banner_cleared_when_signed_in(self):
        """Criterion 3: once probe reports signed in, banner is gone.

        Runs the real renderClaudeBanner under node. Mutation that removes the
        signed-in clear (early-return without tearing down the host) leaves
        STALE content and fails this test. A second unconditional clear on the
        build path must not make this green.
        """
        html = self._html()
        fn = self._extract_js_function(html, "renderClaudeBanner")
        # Structural: signed-in early-return branch must itself clear the host.
        m = re.search(
            r"if\s*\(\s*!\s*\(\s*st\s*&&\s*st\.signed_in\s*===\s*false\s*\)\s*\)"
            r"\s*\{([^}]*)\}",
            fn,
        )
        self.assertIsNotNone(
            m,
            "signed-in early-return branch missing from renderClaudeBanner",
        )
        branch = m.group(1)
        self.assertTrue(
            "replaceChildren()" in branch
            or re.search(r"innerHTML\s*=\s*['\"]['\"]", branch)
            or "textContent=''" in branch
            or 'textContent=""' in branch,
            f"signed-in branch must clear host; got: {branch!r}",
        )
        self.assertIn("return", branch)

        # Behavioural: start with STALE children, signed-in → host empty.
        script = f"""
let authByEngine = {{ claude: {{ signed_in: true }} }};
const host = {{
  kids: ['STALE_BANNER'],
  replaceChildren(...nodes) {{ this.kids = nodes.slice(); }},
  get innerHTML() {{ return this.kids.join(''); }},
  set innerHTML(v) {{ this.kids = v ? [v] : []; }},
}};
const document = {{
  getElementById(id) {{ return id === 'claudeAuthBanner' ? host : null; }},
  createElement(tag) {{
    return {{
      tag, className: '', textContent: '', type: '', onclick: null,
      children: [],
      appendChild(c) {{ this.children.push(c); return c; }},
    }};
  }},
}};
{fn}
renderClaudeBanner();
if (host.kids.length !== 0) {{
  console.error('FAIL kids=' + JSON.stringify(host.kids));
  process.exit(1);
}}
console.log('cleared');
"""
        out = self._run_node(script)
        self.assertIn("cleared", out)

        # And signed-out still builds the banner (control that fn works).
        script2 = f"""
let authByEngine = {{ claude: {{ signed_in: false }} }};
const host = {{
  kids: [],
  replaceChildren(...nodes) {{ this.kids = nodes.slice(); }},
}};
const document = {{
  getElementById(id) {{ return id === 'claudeAuthBanner' ? host : null; }},
  createElement(tag) {{
    return {{
      tag, className: '', textContent: '', type: '', onclick: null,
      children: [],
      appendChild(c) {{ this.children.push(c); return c; }},
    }};
  }},
}};
function refreshAuth() {{}}
{fn}
renderClaudeBanner();
const box = host.kids[0];
if (!box || box.className !== 'claude-banner') {{
  console.error('FAIL no banner box');
  process.exit(1);
}}
const texts = [];
function walk(n) {{
  if (!n) return;
  if (n.textContent) texts.push(n.textContent);
  (n.children || []).forEach(walk);
}}
walk(box);
const joined = texts.join('|');
if (!joined.includes('Claude is signed out') || !joined.includes('claude auth login')) {{
  console.error('FAIL texts=' + joined);
  process.exit(1);
}}
console.log('built');
"""
        out2 = self._run_node(script2)
        self.assertIn("built", out2)

    def test_html_never_attempts_claude_login(self):
        """Criterion 4: room never tries to complete the claude flow itself."""
        html = self._html()
        # No shell-out to `claude auth login` from the page.
        self.assertNotIn("claude auth login --", html)
        # Device-login starter must not be called with claude.
        self.assertNotIn("startDeviceLogin('claude')", html)
        self.assertNotIn('startDeviceLogin("claude")', html)
        self.assertNotIn("startDeviceLogin(`claude`)", html)
        # Server still rejects claude device login (belt + suspenders).
        server = self._load_server_module(
            "agent_room_server_claude_banner_no_login")
        out = server.start_device_login(
            "claude", run=FakeRun(), background=False)
        self.assertIn("error", out)
        self.assertNotIn("verification_url", out)

    def test_apply_auth_to_seats_covers_all_three(self):
        """Seat-level signed-out class for claude, grok, and kimi.

        Exercises the real applyAuthToSeats under node for each engine.
        Excluding claude via `if(eng==='claude')continue` (or any eng filter)
        makes this fail — unlike the old string check for eng!=='grok' only.
        """
        html = self._html()
        fn = self._extract_js_function(html, "applyAuthToSeats")
        # No engine-name filter that would skip a seat before the toggle.
        self.assertIsNone(
            re.search(
                r"if\s*\(\s*eng\s*(===|!==|==|!=)\s*['\"][^'\"]+['\"]\s*\)\s*continue",
                fn,
            ),
            "applyAuthToSeats must not early-continue on engine name",
        )
        self.assertIn(".seat.signed-out", html)
        self.assertIn(".seat.signed-out .s-name::after", html)

        # Behavioural: each of the three engines, alone signed-out, marks
        # only its seat; then all three signed-out marks all three.
        script = f"""
const roster = {{
  seat_claude: {{ engine: 'claude' }},
  seat_grok: {{ engine: 'grok' }},
  seat_kimi: {{ engine: 'kimi' }},
}};
let authByEngine = {{}};
const marked = {{}};
const document = {{
  getElementById(id) {{
    if (!id.startsWith('seat-')) return null;
    const name = id.slice(5);
    if (!(name in roster)) return null;
    return {{
      classList: {{
        toggle(cls, on) {{
          if (cls !== 'signed-out') return;
          if (on) marked[name] = true;
          else delete marked[name];
        }},
      }},
    }};
  }},
}};
{fn}
function reset() {{
  for (const k of Object.keys(marked)) delete marked[k];
}}
function setAuth(map) {{
  authByEngine = map;
  reset();
  applyAuthToSeats();
}}
// Each engine alone signed-out.
for (const eng of ['claude', 'grok', 'kimi']) {{
  const map = {{
    claude: {{ signed_in: true }},
    grok: {{ signed_in: true }},
    kimi: {{ signed_in: true }},
  }};
  map[eng] = {{ signed_in: false }};
  setAuth(map);
  const seat = 'seat_' + eng;
  if (!marked[seat]) {{
    console.error('FAIL missing signed-out for ' + eng + ' got ' + JSON.stringify(marked));
    process.exit(1);
  }}
  for (const other of ['claude', 'grok', 'kimi']) {{
    if (other === eng) continue;
    if (marked['seat_' + other]) {{
      console.error('FAIL spillover onto ' + other + ' when only ' + eng + ' out');
      process.exit(1);
    }}
  }}
}}
// All three out.
setAuth({{
  claude: {{ signed_in: false }},
  grok: {{ signed_in: false }},
  kimi: {{ signed_in: false }},
}});
for (const eng of ['claude', 'grok', 'kimi']) {{
  if (!marked['seat_' + eng]) {{
    console.error('FAIL all-three missing ' + eng);
    process.exit(1);
  }}
}}
console.log('seats-ok');
"""
        out = self._run_node(script)
        self.assertIn("seats-ok", out)

    def test_no_duplicate_startup_refresh_auth(self):
        """Finding 5: bare refreshAuth() at page load is gone; loadRoster owns it."""
        html = self._html()
        # loadRoster still kicks the first probe.
        self.assertIn("refreshAuth()", self._extract_js_function(html, "loadRoster"))
        # Trailing startup block must not fire a second bare refreshAuth().
        # Strip // comments so a note about the removed call cannot false-match.
        tail = html[html.rfind("loadRoster().then") :]
        tail_nocomment = re.sub(r"//[^\n]*", "", tail)
        self.assertNotRegex(
            tail_nocomment,
            r"(?<![\w.])refreshAuth\s*\(\s*\)\s*;",
            "duplicate bare refreshAuth() after loadRoster().then",
        )

    def test_device_login_result_has_no_returncode_field(self):
        """Finding 7: returncode is not a public DeviceLoginResult field."""
        from engines import DeviceLoginResult

        fields = {f.name for f in DeviceLoginResult.__dataclass_fields__.values()}
        self.assertNotIn("returncode", fields)
        result = parse_device_login_output(
            "Visit https://auth.x.ai/device\ncode: ABCD-1234\n"
        )
        self.assertFalse(hasattr(result, "returncode") and
                         "returncode" in getattr(result, "__dataclass_fields__", {}))

    def _load_server_module(self, name: str):
        import importlib.util
        if name in sys.modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, ROOM / "server.py")
        server = importlib.util.module_from_spec(spec)
        with mock.patch("http.server.ThreadingHTTPServer"):
            spec.loader.exec_module(server)
        return server


class TestRedactSecrets(unittest.TestCase):
    """Findings 2 + 4: redactor is tested directly; shape-based fallbacks."""

    JWT = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )

    def test_redact_secrets_labeled_values(self):
        """Finding 2: direct unit coverage of redact_secrets."""
        text = "access_token=sekrit1 refresh_token:sekrit2 api_key=sekrit3"
        out = redact_secrets(text)
        self.assertNotIn("sekrit1", out)
        self.assertNotIn("sekrit2", out)
        self.assertNotIn("sekrit3", out)
        self.assertIn("[REDACTED]", out)

    def test_redact_token_label(self):
        text = f"token: {self.JWT}"
        out = redact_secrets(text)
        self.assertNotIn(self.JWT, out)
        self.assertIn("[REDACTED]", out)

    def test_redact_authorization_bearer(self):
        text = f"Authorization: Bearer {self.JWT}"
        out = redact_secrets(text)
        self.assertNotIn(self.JWT, out)

    def test_redact_bearer_lowercase(self):
        text = f"bearer {self.JWT}"
        out = redact_secrets(text)
        self.assertNotIn(self.JWT, out)

    def test_redact_session_key(self):
        text = "session_key=super-sekrit-session-value-abc"
        out = redact_secrets(text)
        self.assertNotIn("super-sekrit-session-value-abc", out)
        self.assertIn("[REDACTED]", out)

    def test_redact_bare_jwt_on_own_line(self):
        text = f"oops\n{self.JWT}\nend\n"
        out = redact_secrets(text)
        self.assertNotIn(self.JWT, out)
        self.assertIn("oops", out)
        self.assertIn("end", out)

    def test_redact_json_credential_dump(self):
        text = (
            '{\n'
            '  "access_token": "sekrit-json-access",\n'
            '  "refresh_token": "sekrit-json-refresh"\n'
            '}\n'
        )
        out = redact_secrets(text)
        self.assertNotIn("sekrit-json-access", out)
        self.assertNotIn("sekrit-json-refresh", out)

    def test_unlabeled_jwt_absent_from_unreadable_surface(self):
        """Acceptance: unlabeled JWT in unreadable output never reaches UI blob."""
        raw = (
            "login failed hard\n"
            f"{self.JWT}\n"
            "try again later\n"
        )
        run = FakeRun(_proc(stdout=raw, returncode=1))
        result = GrokAdapter().run_device_login(run)
        self.assertFalse(result.readable)
        self.assertIn("login failed hard", result.raw_output)
        self.assertIn("try again later", result.raw_output)
        self.assertNotIn(self.JWT, result.raw_output)
        # Public server surface shape
        blob = json.dumps({
            "raw_output": result.raw_output,
            "message": result.message,
            "status": result.status,
        })
        self.assertNotIn(self.JWT, blob)

    def test_high_entropy_run_redacted(self):
        # Built from halves so secret scanners do not flag this fake fixture.
        fixture = "".join(["aB3dE5fG7hI9", "jK1lM2nO4pQ6", "rS8tU0vW1xY2", "zA3bC4d"])
        secret = fixture
        self.assertGreaterEqual(len(secret), 40)
        out = redact_secrets(f"noise {secret} tail")
        self.assertNotIn(secret, out)
        self.assertIn("noise", out)
        self.assertIn("tail", out)


class TestShared(unittest.TestCase):
    def test_ansi_stripping(self):
        colored = "\x1b[31mred\x1b[0m plain"
        run = FakeRun(_proc(stdout=colored))
        result = GrokAdapter().call({"engine": "grok"}, "x", None, run)
        self.assertEqual(result.reply, "red plain")
        self.assertNotIn("\x1b", result.reply)

    def test_estimated_token_math(self):
        prompt = "abcd" * 10   # 40 chars
        reply = "efgh" * 5     # 20 chars
        self.assertEqual(_estimate_tokens(prompt, reply), (40 + 20) // 4)
        run = FakeRun(_proc(stdout=reply))
        result = GrokAdapter().call({"engine": "grok"}, prompt, None, run)
        self.assertEqual(result.usage_tokens, (len(prompt) + len(reply)) // 4)
        self.assertTrue(result.estimated)

    def test_adapter_registry_completeness(self):
        self.assertEqual(set(ADAPTERS.keys()), {"claude", "grok", "kimi"})
        self.assertIsInstance(ADAPTERS["claude"], ClaudeAdapter)
        self.assertIsInstance(ADAPTERS["grok"], GrokAdapter)
        self.assertIsInstance(ADAPTERS["kimi"], KimiAdapter)
        self.assertFalse(ADAPTERS["claude"].wants_window_on_new_session)
        self.assertTrue(ADAPTERS["grok"].wants_window_on_new_session)
        self.assertFalse(ADAPTERS["kimi"].wants_window_on_new_session)

    def test_engine_result_fields(self):
        r = EngineResult(
            reply="ok",
            session_id="s",
            requested_model="m",
            actual_model="m",
            usage_tokens=1,
            estimated=False,
        )
        self.assertIsNone(r.error)
        self.assertFalse(r.fresh_session)


class TestServerImport(unittest.TestCase):
    def test_import_server_no_start(self):
        """server.py must import cleanly and must not start a server on import."""
        # Guard: only __main__ calls run(); importing is side-effect free
        # of serve_forever (verify by reading source, then import).
        src = (ROOM / "server.py").read_text()
        self.assertIn('if __name__ == "__main__":', src)
        self.assertIn("run()", src)
        # The HTTP server is only constructed inside main(), not at module level.
        self.assertNotRegex(
            src.split("def main")[0],
            r"ThreadingHTTPServer\(",
        )
        # Dead engine bins must not remain in server.py (engines.py owns them).
        self.assertNotIn("GROK_BIN", src)
        self.assertNotIn("KIMI_BIN", src)
        # CLAUDE_BIN still used by update_minutes.
        self.assertIn("CLAUDE_BIN", src)
        # Import under a unique name so re-runs don't fight sys.modules.
        name = "agent_room_server_e4_test"
        if name in sys.modules:
            del sys.modules[name]
        # engines is already importable from ROOM on path
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, ROOM / "server.py")
        mod = importlib.util.module_from_spec(spec)
        # Importing server loads agents.json and builds SEATS — real, local, fine.
        # It must not bind a port.
        with mock.patch("http.server.ThreadingHTTPServer") as http_cls:
            spec.loader.exec_module(mod)
            http_cls.assert_not_called()
        self.assertTrue(hasattr(mod, "run_engine"))
        self.assertTrue(hasattr(mod, "ADAPTERS"))


class TestServerRunEngine(unittest.TestCase):
    """Drive server.run_engine with fakes: CTX accounting + session persistence."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        name = "agent_room_server_e4b_run_engine"
        if name in sys.modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, ROOM / "server.py")
        cls.server = importlib.util.module_from_spec(spec)
        with mock.patch("http.server.ThreadingHTTPServer"):
            spec.loader.exec_module(cls.server)

    def setUp(self):
        # Isolate CTX and sessions; pick real seat names from roster when possible.
        self.server.CTX.clear()
        seats = self.server.SEATS
        # Prefer known engines; fall back to first seat of each engine.
        self.claude_seat = next(
            (n for n, s in seats.items() if s["engine"] == "claude"), None)
        self.grok_seat = next(
            (n for n, s in seats.items() if s["engine"] == "grok"), None)
        self.kimi_seat = next(
            (n for n, s in seats.items() if s["engine"] == "kimi"), None)
        self.assertIsNotNone(self.claude_seat, "roster needs a claude seat")
        self.assertIsNotNone(self.grok_seat, "roster needs a grok seat")
        self.session_writes = []

    def _patch_session(self, initial=None):
        """Patch get_session / set_session to an in-memory store."""
        store = dict(initial or {})

        def get_session(state, seat):
            return store.get(seat)

        def set_session(seat, sid, room_epoch=None):
            # room_epoch accepted for API parity with server.set_session; the
            # in-memory store has no epoch to check.
            self.session_writes.append((seat, sid))
            store[seat] = sid

        return store, get_session, set_session

    def test_ctx_real_usage_from_claude(self):
        payload = {
            "result": "answer",
            "session_id": "c-sess",
            "model": "reported-m",
            "usage": {
                "input_tokens": 40,
                "output_tokens": 10,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        }
        run = FakeRun(_proc(stdout=json.dumps(payload)))
        store, get_s, set_s = self._patch_session()
        with mock.patch.object(self.server, "get_session", get_s), \
             mock.patch.object(self.server, "set_session", set_s), \
             mock.patch.object(self.server, "load_state", return_value={"sessions": {}}), \
             mock.patch.object(self.server.ADAPTERS["claude"], "call",
                               wraps=ClaudeAdapter().call) as wrapped:
            # Inject FakeRun via wrapping: replace subprocess.run path by
            # patching the adapter call to feed our FakeRun.
            def call_with_fake(seat, prompt, session_id, run_fn, clear_session=None):
                return ClaudeAdapter().call(
                    seat, prompt, session_id, run, clear_session=clear_session)
            wrapped.side_effect = call_with_fake
            reply = self.server.run_engine(self.claude_seat, "hi")
        self.assertEqual(reply, "answer")
        self.assertEqual(self.server.CTX[self.claude_seat]["used"], 50)
        self.assertFalse(self.server.CTX[self.claude_seat]["est"])
        # Unconditional store of reported session id (fix 3).
        self.assertIn((self.claude_seat, "c-sess"), self.session_writes)

    def test_ctx_estimated_accumulates_then_resets_on_fresh_grok(self):
        # First call: fresh grok session, CTX starts from 0.
        run1 = FakeRun(_proc(stdout="aaaa"))  # reply 4 chars
        store, get_s, set_s = self._patch_session()

        def call1(seat, prompt, session_id, run_fn, clear_session=None):
            return GrokAdapter().call(
                seat, prompt, session_id, run1, clear_session=clear_session)

        with mock.patch.object(self.server, "get_session", get_s), \
             mock.patch.object(self.server, "set_session", set_s), \
             mock.patch.object(self.server, "load_state", return_value={"sessions": {}}), \
             mock.patch.object(self.server.ADAPTERS["grok"], "call", side_effect=call1):
            self.server.run_engine(self.grok_seat, "bb")  # prompt 2 chars
        used1 = self.server.CTX[self.grok_seat]["used"]
        self.assertTrue(self.server.CTX[self.grok_seat]["est"])
        # (2+4)//4 = 1
        self.assertEqual(used1, (len("bb") + len("aaaa")) // 4)
        stored_sid = store.get(self.grok_seat)
        self.assertIsNotNone(stored_sid)

        # Second call: same session (resume) — estimated accumulates.
        run2 = FakeRun(_proc(stdout="cccccccc"))  # 8 chars
        def call2(seat, prompt, session_id, run_fn, clear_session=None):
            self.assertEqual(session_id, stored_sid)
            return GrokAdapter().call(
                seat, prompt, session_id, run2, clear_session=clear_session)

        with mock.patch.object(self.server, "get_session", get_s), \
             mock.patch.object(self.server, "set_session", set_s), \
             mock.patch.object(self.server, "load_state",
                               return_value={"sessions": dict(store)}), \
             mock.patch.object(self.server.ADAPTERS["grok"], "call", side_effect=call2):
            self.server.run_engine(self.grok_seat, "dddd")  # 4 chars
        used2 = self.server.CTX[self.grok_seat]["used"]
        turn2 = (len("dddd") + len("cccccccc")) // 4
        self.assertEqual(used2, used1 + turn2)

        # Third call: failed resume heals to fresh session — CTX resets.
        run3 = FakeRun(
            _proc(stdout="", stderr="rot", returncode=1),
            _proc(stdout="ee"),  # 2 chars
        )
        cleared_early = []

        def call3(seat, prompt, session_id, run_fn, clear_session=None):
            def tracking_clear():
                cleared_early.append(True)
                if clear_session:
                    clear_session()
            return GrokAdapter().call(
                seat, prompt, session_id, run3, clear_session=tracking_clear)

        self.session_writes.clear()
        with mock.patch.object(self.server, "get_session", get_s), \
             mock.patch.object(self.server, "set_session", set_s), \
             mock.patch.object(self.server, "load_state",
                               return_value={"sessions": dict(store)}), \
             mock.patch.object(self.server.ADAPTERS["grok"], "call", side_effect=call3):
            self.server.run_engine(self.grok_seat, "f")  # 1 char
        self.assertTrue(cleared_early, "early clear must fire on stale resume")
        # Early clear wrote None, then final set wrote new uuid.
        self.assertEqual(self.session_writes[0], (self.grok_seat, None))
        final_sid = self.session_writes[-1][1]
        self.assertIsNotNone(final_sid)
        self.assertNotEqual(final_sid, stored_sid)
        used3 = self.server.CTX[self.grok_seat]["used"]
        turn3 = (len("f") + len("ee")) // 4
        self.assertEqual(used3, turn3)  # reset, not used2 + turn3

    def test_unconditional_set_session_when_unchanged(self):
        """Even when engine reports the same sid, set_session must still run."""
        same = "sess-same"
        payload = {
            "result": "ok",
            "session_id": same,
            "model": "m",
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
        run = FakeRun(_proc(stdout=json.dumps(payload)))
        store, get_s, set_s = self._patch_session({self.claude_seat: same})

        def call(seat, prompt, session_id, run_fn, clear_session=None):
            return ClaudeAdapter().call(
                seat, prompt, session_id, run, clear_session=clear_session)

        with mock.patch.object(self.server, "get_session", get_s), \
             mock.patch.object(self.server, "set_session", set_s), \
             mock.patch.object(self.server, "load_state",
                               return_value={"sessions": {self.claude_seat: same}}), \
             mock.patch.object(self.server.ADAPTERS["claude"], "call", side_effect=call):
            self.server.run_engine(self.claude_seat, "again")
        # Fix 3: rewrite even when unchanged (would have been skipped by !=).
        self.assertIn((self.claude_seat, same), self.session_writes)

    def test_early_clear_on_run_engine_when_retry_times_out(self):
        """run_engine must clear stale session before a retry that times out."""
        store, get_s, set_s = self._patch_session({self.claude_seat: "stale"})
        order = []

        def call(seat, prompt, session_id, run_fn, clear_session=None):
            def tracking_run(cmd, **kwargs):
                order.append("run")
                if order.count("run") == 1:
                    return _proc(stdout="", stderr="stale", returncode=1)
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

            def tracking_clear():
                order.append("clear")
                if clear_session:
                    clear_session()

            return ClaudeAdapter().call(
                seat, prompt, session_id, tracking_run,
                clear_session=tracking_clear,
            )

        with mock.patch.object(self.server, "get_session", get_s), \
             mock.patch.object(self.server, "set_session", set_s), \
             mock.patch.object(self.server, "load_state",
                               return_value={"sessions": {self.claude_seat: "stale"}}), \
             mock.patch.object(self.server.ADAPTERS["claude"], "call", side_effect=call):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.server.run_engine(self.claude_seat, "p")
        self.assertEqual(order, ["run", "clear", "run"])
        # Early clear left None in the store (even though final set_session
        # never ran because the exception escaped run_engine).
        self.assertIsNone(store.get(self.claude_seat))
        self.assertIn((self.claude_seat, None), self.session_writes)

    def test_no_session_store_on_failed_call(self):
        """A failed engine call (result.error set) must write no session state
        — the original never stored a sid from a failed fresh call."""
        store, get_s, set_s = self._patch_session({})

        def call(seat, prompt, session_id, run_fn, clear_session=None):
            run = FakeRun(_proc(stdout="", stderr="boom", returncode=1))
            return ClaudeAdapter().call(
                seat, prompt, session_id, run, clear_session=clear_session)

        with mock.patch.object(self.server, "get_session", get_s), \
             mock.patch.object(self.server, "set_session", set_s), \
             mock.patch.object(self.server, "load_state",
                               return_value={"sessions": {}}), \
             mock.patch.object(self.server.ADAPTERS["claude"], "call", side_effect=call):
            self.server.run_engine(self.claude_seat, "p")
        self.assertEqual(self.session_writes, [])


class TestArchiveSessions(unittest.TestCase):
    """DM-09 / T04: filing a chat records per-seat session id, engine, model."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        name = "agent_room_server_t04_archive"
        if name in sys.modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, ROOM / "server.py")
        cls.server = importlib.util.module_from_spec(spec)
        with mock.patch("http.server.ThreadingHTTPServer"):
            spec.loader.exec_module(cls.server)

    def setUp(self):
        seats = self.server.SEATS
        self.claude_seat = next(
            n for n, s in seats.items() if s["engine"] == "claude")
        self.grok_seat = next(
            n for n, s in seats.items() if s["engine"] == "grok")
        self.kimi_seat = next(
            (n for n, s in seats.items() if s["engine"] == "kimi"), None)
        # A third seat that will stay silent in the fixture.
        silent = next(
            (n for n in seats if n not in (self.claude_seat, self.grok_seat)),
            None,
        )
        self.silent_seat = silent
        self.claude_model = seats[self.claude_seat].get("model") or ""
        self.grok_model = seats[self.grok_seat].get("model") or ""

    def test_filing_writes_session_engine_model_for_speakers_only(self):
        """Criterion 1: archive sessions hold exact sid/engine/model per speaker;
        seats that did not speak are absent. Not a vacuous 'has sessions key'."""
        claude_sid = "claude-sess-exact-001"
        grok_sid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        state = {
            "next_id": 5,
            "sessions": {
                self.claude_seat: claude_sid,
                self.grok_seat: grok_sid,
                # Silent seat has a leftover live id — must NOT be filed.
                **({self.silent_seat: "silent-should-not-appear"}
                   if self.silent_seat else {}),
            },
        }
        messages = [
            {"id": 1, "from": "you", "text": f"@{self.claude_seat} hi", "kind": "chat"},
            {"id": 2, "from": self.claude_seat, "text": "hello", "kind": "chat"},
            {"id": 3, "from": "you", "text": f"@{self.grok_seat} build", "kind": "chat"},
            {"id": 4, "from": self.grok_seat, "text": "built", "kind": "chat"},
        ]
        self.assertTrue(
            hasattr(self.server, "sessions_for_archive"),
            "server must expose sessions_for_archive(state, messages, seats)",
        )
        out = self.server.sessions_for_archive(
            state, messages, self.server.SEATS)

        # Exact values for seats that spoke — not merely "key present".
        self.assertEqual(
            out[self.claude_seat],
            {
                "session_id": claude_sid,
                "engine": "claude",
                "model": self.claude_model,
            },
        )
        self.assertEqual(
            out[self.grok_seat],
            {
                "session_id": grok_sid,
                "engine": "grok",
                "model": self.grok_model,
            },
        )
        if self.silent_seat:
            self.assertNotIn(self.silent_seat, out)
        # the user is never a seat.
        self.assertNotIn("you", out)

    def _seed_current(self, rooms, claude_sid, grok_sid, *,
                      silent_id="silent-left-over"):
        """Write a live current/ with two speakers and optional silent seat."""
        current = rooms / "current"
        current.mkdir()
        sessions = {
            self.claude_seat: claude_sid,
            self.grok_seat: grok_sid,
        }
        if self.silent_seat:
            sessions[self.silent_seat] = silent_id
        state = {
            "claude_session": None,
            "kimi_started": False,
            "last_seen": {},
            "mission": None,
            "stop": False,
            "next_id": 5,
            "sessions": sessions,
        }
        (current / "state.json").write_text(json.dumps(state, indent=2))
        msgs = [
            {"id": 1, "ts": "t1", "from": "you",
             "text": f"@{self.claude_seat} plan", "hops": 0, "kind": "chat"},
            {"id": 2, "ts": "t2", "from": self.claude_seat,
             "text": "plan ready", "hops": 1, "kind": "chat"},
            {"id": 3, "ts": "t3", "from": "you",
             "text": f"@{self.grok_seat} go", "hops": 0, "kind": "chat"},
            {"id": 4, "ts": "t4", "from": self.grok_seat,
             "text": "done", "hops": 1, "kind": "chat"},
        ]
        with (current / "messages.jsonl").open("w") as fh:
            for m in msgs:
                fh.write(json.dumps(m) + "\n")
        return current, state

    def test_file_current_room_writes_archive_sessions(self):
        """Criterion 1 via the save/archive seam: filing current rewrites
        state.json sessions to rich records; directory name is room-<stamp>."""
        import tempfile

        claude_sid = "filed-claude-sid-xyz"
        grok_sid = "b2c3d4e5-f6a7-8901-bcde-f12345678901"
        stamp = "2026-08-03-test"
        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            current, _ = self._seed_current(rooms, claude_sid, grok_sid)

            self.assertTrue(
                hasattr(self.server, "file_current_room"),
                "server must expose file_current_room() for the archive seam",
            )
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", current):
                archived = self.server.file_current_room(stamp=stamp)

            self.assertIsNotNone(archived)
            self.assertTrue(archived.is_dir())
            # Finding 9: stamp → directory name is what the sidebar lists.
            self.assertEqual(archived.name, f"room-{stamp}")
            self.assertEqual(archived, rooms / f"room-{stamp}")
            filed_state = json.loads((archived / "state.json").read_text())
            sess = filed_state["sessions"]
            self.assertEqual(sess[self.claude_seat]["session_id"], claude_sid)
            self.assertEqual(sess[self.claude_seat]["engine"], "claude")
            self.assertEqual(sess[self.claude_seat]["model"], self.claude_model)
            self.assertEqual(sess[self.grok_seat]["session_id"], grok_sid)
            self.assertEqual(sess[self.grok_seat]["engine"], "grok")
            self.assertEqual(sess[self.grok_seat]["model"], self.grok_model)
            if self.silent_seat:
                self.assertNotIn(self.silent_seat, sess)
            # Fresh current exists for the next chat.
            self.assertTrue((rooms / "current" / "state.json").exists())

    def test_post_newroom_archives_rich_session_records(self):
        """Finding 1: drive POST /api/newroom itself — not just the helper.

        Replacing file_current_room with a plain shutil.move must make this
        test go red (archived sessions stay bare strings, not rich records).
        """
        import tempfile
        from io import BytesIO

        claude_sid = "api-claude-sid-001"
        grok_sid = "c3d4e5f6-a7b8-9012-cdef-123456789012"
        stamp = "2026-08-03-apiroom"
        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            current, live_before = self._seed_current(
                rooms, claude_sid, grok_sid)

            body = b"{}"
            h = self.server.Handler.__new__(self.server.Handler)
            h.path = "/api/newroom"
            h.headers = {"Content-Length": str(len(body))}
            h.rfile = BytesIO(body)
            h.wfile = BytesIO()
            h.request_version = "HTTP/1.1"
            h.command = "POST"
            h.client_address = ("127.0.0.1", 9)
            h.close_connection = False
            h.log_message = lambda *a, **k: None
            status = {}

            def send_response(code, message=None):
                status["code"] = code

            h.send_response = send_response
            h.send_header = lambda *a, **k: None
            h.end_headers = lambda: None

            fixed_dt = mock.Mock()
            fixed_dt.now.return_value.strftime.return_value = stamp
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", current), \
                 mock.patch.object(self.server, "datetime", fixed_dt):
                h.do_POST()

            self.assertEqual(status.get("code"), 200)
            archived = rooms / f"room-{stamp}"
            self.assertTrue(
                archived.is_dir(),
                f"POST /api/newroom must create {archived.name}",
            )
            filed = json.loads((archived / "state.json").read_text())
            sess = filed["sessions"]
            # Exact rich records — bare-string move would leave these as str.
            self.assertEqual(
                sess[self.claude_seat],
                {
                    "session_id": claude_sid,
                    "engine": "claude",
                    "model": self.claude_model,
                },
            )
            self.assertEqual(
                sess[self.grok_seat],
                {
                    "session_id": grok_sid,
                    "engine": "grok",
                    "model": self.grok_model,
                },
            )
            if self.silent_seat:
                self.assertNotIn(self.silent_seat, sess)
            # Live sids that spoke must not survive as bare strings only.
            self.assertIsInstance(sess[self.claude_seat], dict)
            self.assertIsInstance(sess[self.grok_seat], dict)
            # Handler also left a usable fresh current.
            self.assertTrue((rooms / "current" / "state.json").exists())
            # Guard: live_before still describes the pre-archive shape we set.
            self.assertEqual(
                live_before["sessions"][self.claude_seat], claude_sid)

    def test_failed_move_leaves_live_session_ids_intact(self):
        """Finding 2: if shutil.move raises, live session ids must survive.

        A save_state-before-move design drops every non-speaker (and rewrites
        speakers) on the live current even when the archive never lands.
        """
        import tempfile

        claude_sid = "live-claude-keep-me"
        grok_sid = "live-grok-keep-me"
        silent_id = "live-silent-keep-me"
        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            current, _ = self._seed_current(
                rooms, claude_sid, grok_sid, silent_id=silent_id)
            before = (current / "state.json").read_text()

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", current), \
                 mock.patch.object(
                     self.server.shutil, "move",
                     side_effect=OSError("disk full"),
                 ):
                with self.assertRaises(OSError):
                    self.server.file_current_room(stamp="fail-move")

            # current/ still here — move never happened.
            self.assertTrue(current.is_dir())
            self.assertFalse((rooms / "room-fail-move").exists())
            after = json.loads((current / "state.json").read_text())
            # Every live session id intact, including silent leftover.
            self.assertEqual(after["sessions"][self.claude_seat], claude_sid)
            self.assertEqual(after["sessions"][self.grok_seat], grok_sid)
            if self.silent_seat:
                self.assertEqual(
                    after["sessions"][self.silent_seat], silent_id)
            # Live file content must not have been rewritten to rich form.
            self.assertEqual(
                after["sessions"][self.claude_seat], claude_sid,
            )
            self.assertIsInstance(after["sessions"][self.claude_seat], str)
            # Byte-identical state is the strongest proof nothing was saved.
            self.assertEqual((current / "state.json").read_text(), before)

    def test_empty_session_id_skipped_in_archive(self):
        """Finding 5: speakers with empty/None session ids are omitted."""
        state = {
            "sessions": {
                self.claude_seat: "",
                self.grok_seat: None,
            },
        }
        messages = [
            {"from": self.claude_seat, "text": "hi"},
            {"from": self.grok_seat, "text": "yo"},
        ]
        out = self.server.sessions_for_archive(
            state, messages, self.server.SEATS)
        self.assertNotIn(self.claude_seat, out)
        self.assertNotIn(self.grok_seat, out)
        self.assertEqual(out, {})

    def test_get_session_does_not_mutate_state(self):
        """Finding 7: get_session is a pure read — no setdefault side effect."""
        state = {"next_id": 1}  # no sessions key at all
        sid = self.server.get_session(state, self.claude_seat)
        self.assertIsNone(sid)
        self.assertNotIn("sessions", state)

        empty = {"sessions": {}}
        self.server.get_session(empty, self.claude_seat)
        self.assertEqual(empty, {"sessions": {}})

    def test_grok_mints_session_id_when_chat_begins(self):
        """Criterion 2: first grok call mints a real UUID via -s/--session-id,
        stores it, and that same id is what filing would archive."""
        run = FakeRun(_proc(stdout="grok first reply"))
        result = GrokAdapter().call(
            {"engine": "grok", "model": self.grok_model}, "start", None, run)
        cmd = run.calls[0][0]
        # Mint path uses -s / --session-id, not resume.
        self.assertTrue("-s" in cmd or "--session-id" in cmd)
        flag = "-s" if "-s" in cmd else "--session-id"
        minted = cmd[cmd.index(flag) + 1]
        uuid.UUID(minted)  # must be a real UUID
        self.assertEqual(result.session_id, minted)
        self.assertTrue(result.fresh_session)
        self.assertIsNone(result.error)

        # That minted id is what sessions_for_archive records for a speaker.
        state = {"sessions": {self.grok_seat: minted}}
        messages = [
            {"id": 1, "from": "you", "text": f"@{self.grok_seat} hi"},
            {"id": 2, "from": self.grok_seat, "text": "grok first reply"},
        ]
        out = self.server.sessions_for_archive(
            state, messages, self.server.SEATS)
        self.assertEqual(out[self.grok_seat]["session_id"], minted)
        self.assertEqual(out[self.grok_seat]["engine"], "grok")
        self.assertEqual(out[self.grok_seat]["model"], self.grok_model)

    def test_pre_change_archive_loads_without_raising(self):
        """Criterion 3: a chat filed before T04 (no sessions / string sessions)
        still loads; list_archives and get_session do not raise."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            old = rooms / "room-2026-08-01-legacy"
            old.mkdir()
            # Pre-T04 state shape: no sessions key at all.
            (old / "state.json").write_text(json.dumps({
                "claude_session": "legacy-claude-sid",
                "kimi_started": False,
                "last_seen": {"claude": 2},
                "mission": None,
                "stop": False,
                "next_id": 3,
            }, indent=2))
            with (old / "messages.jsonl").open("w") as fh:
                fh.write(json.dumps({
                    "id": 1, "ts": "2026-08-01T10:00:00", "from": "you",
                    "text": "hello", "hops": 0, "kind": "chat",
                }) + "\n")
                fh.write(json.dumps({
                    "id": 2, "ts": "2026-08-01T10:00:01", "from": "claude",
                    "text": "hi", "hops": 1, "kind": "chat",
                }) + "\n")

            with mock.patch.object(self.server, "ROOMS_DIR", rooms):
                listed = self.server.list_archives()
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["id"], "room-2026-08-01-legacy")

            # get_session on pre-T04 shape (no sessions) returns None, no raise.
            legacy_state = json.loads((old / "state.json").read_text())
            sid = self.server.get_session(legacy_state, "claude")
            self.assertIsNone(sid)

            # String-form live sessions still resolve to the bare id.
            string_state = {"sessions": {self.claude_seat: "plain-string-sid"}}
            self.assertEqual(
                self.server.get_session(string_state, self.claude_seat),
                "plain-string-sid",
            )
            # Rich archive form also resolves (session_id only — no id fallback).
            rich_state = {
                "sessions": {
                    self.claude_seat: {
                        "session_id": "rich-sid",
                        "engine": "claude",
                        "model": "m",
                    }
                }
            }
            self.assertEqual(
                self.server.get_session(rich_state, self.claude_seat),
                "rich-sid",
            )
            # Speculative id-only records are not a supported shape.
            id_only = {
                "sessions": {self.claude_seat: {"id": "should-not-count"}}
            }
            self.assertIsNone(
                self.server.get_session(id_only, self.claude_seat))

    def test_filing_tolerates_corrupt_and_missing_state(self):
        """Finding 4: corrupt JSON, non-dict top level, missing state file —
        file_current_room never raises; archive still lands."""
        import tempfile

        cases = [
            ("corrupt", "NOT-JSON{{{"),
            ("non_dict", json.dumps(["a", "list"])),
            ("missing", None),  # no state.json at all
        ]
        for label, payload in cases:
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as td:
                    rooms = Path(td)
                    current = rooms / "current"
                    current.mkdir()
                    if payload is not None:
                        (current / "state.json").write_text(payload)
                    with (current / "messages.jsonl").open("w") as fh:
                        fh.write(json.dumps({
                            "id": 1, "ts": "t", "from": "you",
                            "text": "hi", "hops": 0, "kind": "chat",
                        }) + "\n")
                    stamp = f"edge-{label}"
                    with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                         mock.patch.object(self.server, "CURRENT", current):
                        archived = self.server.file_current_room(stamp=stamp)
                    self.assertIsNotNone(archived)
                    self.assertTrue(archived.is_dir())
                    self.assertEqual(archived.name, f"room-{stamp}")
                    # Archive state is a dict with sessions map (possibly empty).
                    filed = json.loads((archived / "state.json").read_text())
                    self.assertIsInstance(filed, dict)
                    self.assertIsInstance(filed.get("sessions"), dict)
                    # Fresh current recreated.
                    self.assertTrue(
                        (rooms / "current" / "state.json").exists())

    def test_load_room_state_is_not_kept_as_dead_code(self):
        """Finding 3: load_room_state has no production caller — do not keep it."""
        self.assertFalse(
            hasattr(self.server, "load_room_state"),
            "load_room_state is dead code; remove it (T05 will add read-back)",
        )


class TestReopenRoom(unittest.TestCase):
    """DM-10 / T05: reopen a filed chat and reattach engine sessions."""

    ARCHIVE_MARKER = "ARCHIVE_SECRET_MARKER_T05_xyz_do_not_prompt"

    @classmethod
    def setUpClass(cls):
        import importlib.util
        name = "agent_room_server_t05_reopen"
        if name in sys.modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, ROOM / "server.py")
        cls.server = importlib.util.module_from_spec(spec)
        with mock.patch("http.server.ThreadingHTTPServer"):
            spec.loader.exec_module(cls.server)

    def setUp(self):
        seats = self.server.SEATS
        self.claude_seat = next(
            n for n, s in seats.items() if s["engine"] == "claude")
        self.grok_seat = next(
            n for n, s in seats.items() if s["engine"] == "grok")
        self.kimi_seat = next(
            (n for n, s in seats.items() if s["engine"] == "kimi"), None)
        self.claude_model = seats[self.claude_seat].get("model") or ""
        self.grok_model = seats[self.grok_seat].get("model") or ""
        self.kimi_model = (
            seats[self.kimi_seat].get("model") or "" if self.kimi_seat else "")
        self.server.CTX.clear()

    def _write_archive(
        self, rooms, room_id, *, sessions, messages, last_seen=None, next_id=None,
    ):
        """Write rooms/<room_id>/ with state.json + messages.jsonl."""
        d = rooms / room_id
        d.mkdir(parents=True)
        max_id = max((m.get("id", 0) for m in messages), default=0)
        state = {
            "claude_session": None,
            "kimi_started": False,
            "last_seen": last_seen if last_seen is not None else {
                self.claude_seat: max_id,
                self.grok_seat: max_id,
                **({self.kimi_seat: max_id} if self.kimi_seat else {}),
            },
            "mission": None,
            "stop": False,
            "next_id": next_id if next_id is not None else max_id + 1,
            "sessions": sessions,
        }
        (d / "state.json").write_text(json.dumps(state, indent=2))
        with (d / "messages.jsonl").open("w") as fh:
            for m in messages:
                fh.write(json.dumps(m) + "\n")
        return d

    def _seed_empty_current(self, rooms):
        current = rooms / "current"
        current.mkdir()
        state = {
            "claude_session": None,
            "kimi_started": False,
            "last_seen": {},
            "mission": None,
            "stop": False,
            "next_id": 1,
            "sessions": {},
        }
        (current / "state.json").write_text(json.dumps(state, indent=2))
        (current / "messages.jsonl").write_text("")
        return current

    def _rich_sessions(self, claude_sid, grok_sid, kimi_sid=None):
        out = {
            self.claude_seat: {
                "session_id": claude_sid,
                "engine": "claude",
                "model": self.claude_model,
            },
            self.grok_seat: {
                "session_id": grok_sid,
                "engine": "grok",
                "model": self.grok_model,
            },
        }
        if self.kimi_seat and kimi_sid:
            out[self.kimi_seat] = {
                "session_id": kimi_sid,
                "engine": "kimi",
                "model": self.kimi_model,
            }
        return out

    def _archive_msgs(self):
        """Two-turn archive whose marker must never re-enter a resume prompt."""
        return [
            {"id": 1, "ts": "t1", "from": "you",
             "text": f"remember {self.ARCHIVE_MARKER}", "hops": 0, "kind": "chat"},
            {"id": 2, "ts": "t2", "from": self.claude_seat,
             "text": f"got it: {self.ARCHIVE_MARKER}", "hops": 1, "kind": "chat"},
            {"id": 3, "ts": "t3", "from": "you",
             "text": f"@{self.grok_seat} also note {self.ARCHIVE_MARKER}",
             "hops": 0, "kind": "chat"},
            {"id": 4, "ts": "t4", "from": self.grok_seat,
             "text": "noted", "hops": 1, "kind": "chat"},
        ]

    def test_reopen_reattaches_resume_flags_without_archive_in_prompt(self):
        """Criteria 1+2: each engine gets its stored sid on the resume flag,
        and the archived conversation is absent from the prompt."""
        import tempfile

        self.assertTrue(
            hasattr(self.server, "reopen_room"),
            "server must expose reopen_room() for the resume seam",
        )
        claude_sid = "resume-claude-sid-T05-001"
        grok_sid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        kimi_sid = "kimi-resume-sid-T05-abc"
        room_id = "room-2026-08-03-resume1"
        msgs = self._archive_msgs()
        if self.kimi_seat:
            msgs = list(msgs) + [
                {"id": 5, "ts": "t5", "from": "you",
                 "text": f"@{self.kimi_seat} note {self.ARCHIVE_MARKER}",
                 "hops": 0, "kind": "chat"},
                {"id": 6, "ts": "t6", "from": self.kimi_seat,
                 "text": "kimi noted", "hops": 1, "kind": "chat"},
            ]
        max_id = msgs[-1]["id"]

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_empty_current(rooms)
            self._write_archive(
                rooms, room_id,
                sessions=self._rich_sessions(claude_sid, grok_sid, kimi_sid),
                messages=msgs,
                last_seen={
                    self.claude_seat: max_id,
                    self.grok_seat: max_id,
                    **({self.kimi_seat: max_id} if self.kimi_seat else {}),
                },
                next_id=max_id + 1,
            )

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                result = self.server.reopen_room(room_id)

            self.assertTrue(result.get("ok"), f"reopen failed: {result}")
            self.assertTrue(result.get("continued"), f"expected continued: {result}")
            # Archive is now live current.
            self.assertTrue((rooms / "current" / "state.json").exists())
            self.assertFalse((rooms / room_id).exists())
            live = json.loads((rooms / "current" / "state.json").read_text())
            # Live shape: bare session strings, exact stored ids.
            self.assertEqual(live["sessions"][self.claude_seat], claude_sid)
            self.assertEqual(live["sessions"][self.grok_seat], grok_sid)
            if self.kimi_seat:
                self.assertEqual(live["sessions"][self.kimi_seat], kimi_sid)

            # Next turn: append a fresh user message, drive run_engine for each
            # seat. Capture the argv actually handed to the runner + the prompt.
            new_text = "@all what was the secret? (fresh only)"
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                self.server.append_message("you", new_text)

            captures = {}  # engine -> (cmd, prompt)

            def make_call(engine_name, adapter_cls, fake_run):
                def call(seat, prompt, session_id, run_fn, clear_session=None):
                    result = adapter_cls().call(
                        seat, prompt, session_id, fake_run,
                        clear_session=clear_session)
                    captures[engine_name] = (fake_run.calls[0][0], prompt, session_id)
                    return result
                return call

            claude_run = FakeRun(_proc(stdout=json.dumps({
                "result": "from memory",
                "session_id": claude_sid,
                "model": self.claude_model,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })))
            grok_run = FakeRun(_proc(stdout="from grok memory"))
            kimi_run = FakeRun(
                _proc(stdout="from kimi memory\nTo resume this session: "
                      f"kimi -r {kimi_sid}\n"))

            def _drive_next_turns():
                # Build the same prompt the worker would: only messages after
                # last_seen, never the archive window when a session is attached.
                state = self.server.load_state()
                last_c = state["last_seen"].get(self.claude_seat, 0)
                fresh_c = [m for m in self.server.all_messages()
                           if m["id"] > last_c and m["from"] != self.claude_seat]
                self.assertIsNotNone(
                    self.server.get_session(state, self.claude_seat))
                self.assertIsNotNone(
                    self.server.get_session(state, self.grok_seat))
                # Criterion 3: with a session, feed is delta-only (not the window).
                all_msgs = self.server.all_messages()
                self.assertLess(
                    len(fresh_c), len(all_msgs),
                    "resume feed must be a delta, not the full transcript",
                )
                prompt_c = self.server.build_prompt(self.claude_seat, fresh_c)
                self.server.run_engine(self.claude_seat, prompt_c)

                last_g = state["last_seen"].get(self.grok_seat, 0)
                fresh_g = [m for m in self.server.all_messages()
                           if m["id"] > last_g and m["from"] != self.grok_seat]
                # Explicit worker rule: session present ⇒ no window.
                eng = self.server.SEATS[self.grok_seat]["engine"]
                if (self.server.ADAPTERS[eng].wants_window_on_new_session
                        and not self.server.get_session(state, self.grok_seat)):
                    feed_g = self.server.all_messages()[-self.server.GROK_WINDOW:]
                else:
                    feed_g = fresh_g
                prompt_g = self.server.build_prompt(self.grok_seat, feed_g)
                self.server.run_engine(self.grok_seat, prompt_g)

                if self.kimi_seat:
                    last_k = state["last_seen"].get(self.kimi_seat, 0)
                    fresh_k = [m for m in self.server.all_messages()
                               if m["id"] > last_k and m["from"] != self.kimi_seat]
                    prompt_k = self.server.build_prompt(self.kimi_seat, fresh_k)
                    self.server.run_engine(self.kimi_seat, prompt_k)

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(
                     self.server.ADAPTERS["claude"], "call",
                     side_effect=make_call("claude", ClaudeAdapter, claude_run)), \
                 mock.patch.object(
                     self.server.ADAPTERS["grok"], "call",
                     side_effect=make_call("grok", GrokAdapter, grok_run)), \
                 mock.patch.object(
                     self.server.ADAPTERS["kimi"], "call",
                     side_effect=make_call("kimi", KimiAdapter, kimi_run)):
                _drive_next_turns()

            # --- Claude: --resume <stored id>, marker absent from prompt ---
            cmd_c, prompt_c_cap, sid_c = captures["claude"]
            self.assertEqual(sid_c, claude_sid)
            self.assertIn("--resume", cmd_c)
            self.assertEqual(cmd_c[cmd_c.index("--resume") + 1], claude_sid)
            self.assertNotIn(self.ARCHIVE_MARKER, prompt_c_cap)
            self.assertIn("fresh only", prompt_c_cap)

            # --- Grok: -r <stored id>, no window smuggle ---
            cmd_g, prompt_g_cap, sid_g = captures["grok"]
            self.assertEqual(sid_g, grok_sid)
            self.assertIn("-r", cmd_g)
            self.assertEqual(cmd_g[cmd_g.index("-r") + 1], grok_sid)
            self.assertNotIn(self.ARCHIVE_MARKER, prompt_g_cap)
            self.assertIn("fresh only", prompt_g_cap)

            if self.kimi_seat:
                cmd_k, prompt_k_cap, sid_k = captures["kimi"]
                self.assertEqual(sid_k, kimi_sid)
                self.assertIn("-S", cmd_k)
                self.assertEqual(cmd_k[cmd_k.index("-S") + 1], kimi_sid)
                self.assertNotIn(self.ARCHIVE_MARKER, prompt_k_cap)
                self.assertIn("fresh only", prompt_k_cap)

    def test_archive_without_sessions_opens_readonly_with_reason(self):
        """Criterion 4: no stored session ids → read-only + plain-English reason."""
        import tempfile

        room_id = "room-2026-08-01-legacy-ro"
        msgs = [
            {"id": 1, "ts": "t1", "from": "you",
             "text": "old chat no sessions", "hops": 0, "kind": "chat"},
            {"id": 2, "ts": "t2", "from": "claude",
             "text": "hi", "hops": 1, "kind": "chat"},
        ]
        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            current = self._seed_empty_current(rooms)
            # Pre-T04 shape: no sessions key (or empty).
            self._write_archive(rooms, room_id, sessions={}, messages=msgs)

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", current):
                result = self.server.reopen_room(room_id)

            self.assertTrue(result.get("ok"), f"unexpected: {result}")
            self.assertTrue(result.get("readonly"), "must open read-only")
            self.assertFalse(result.get("continued", False))
            reason = result.get("reason") or ""
            self.assertTrue(reason.strip(), "must explain why in plain English")
            # Plain English: no jargon-only, must mention cannot continue / sessions.
            low = reason.lower()
            self.assertTrue(
                any(w in low for w in (
                    "cannot be continued", "can't be continued",
                    "cannot continue", "read-only", "read only",
                    "no saved", "no session",
                )),
                f"reason should explain why it cannot continue: {reason!r}",
            )
            self.assertTrue(result.get("messages"), "must return messages to view")
            # Live current must be untouched; archive still in place.
            self.assertTrue((rooms / room_id).is_dir())
            self.assertTrue(current.is_dir())
            live = json.loads((current / "state.json").read_text())
            self.assertEqual(live.get("sessions") or {}, {})

    def test_two_chats_filed_same_second_both_listed(self):
        """Criterion 6: same-second stamps get unique dest names; both listed."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            stamp = "2026-08-03-120000"

            def seed_and_file(label):
                current = rooms / "current"
                if current.exists():
                    import shutil
                    shutil.rmtree(current)
                current.mkdir()
                (current / "state.json").write_text(json.dumps({
                    "claude_session": None,
                    "kimi_started": False,
                    "last_seen": {},
                    "mission": None,
                    "stop": False,
                    "next_id": 2,
                    "sessions": {self.claude_seat: f"sid-{label}"},
                }, indent=2))
                with (current / "messages.jsonl").open("w") as fh:
                    fh.write(json.dumps({
                        "id": 1, "ts": "t", "from": "you",
                        "text": f"chat {label} unique body words",
                        "hops": 0, "kind": "chat",
                    }) + "\n")
                    fh.write(json.dumps({
                        "id": 2, "ts": "t", "from": self.claude_seat,
                        "text": f"reply {label}", "hops": 1, "kind": "chat",
                    }) + "\n")
                with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                     mock.patch.object(self.server, "CURRENT", current):
                    return self.server.file_current_room(stamp=stamp)

            a1 = seed_and_file("alpha")
            a2 = seed_and_file("beta")
            self.assertIsNotNone(a1)
            self.assertIsNotNone(a2)
            self.assertNotEqual(a1.name, a2.name,
                                "same-second filings must not share a folder name")
            self.assertTrue(a1.is_dir())
            self.assertTrue(a2.is_dir())
            # Neither archive is nested inside the other.
            self.assertEqual(a1.parent, rooms)
            self.assertEqual(a2.parent, rooms)

            with mock.patch.object(self.server, "ROOMS_DIR", rooms):
                listed = self.server.list_archives()
            ids = {r["id"] for r in listed}
            self.assertIn(a1.name, ids)
            self.assertIn(a2.name, ids)
            self.assertEqual(len(listed), 2)

    def test_post_reopen_handler_continues_when_sessions_present(self):
        """Drive POST /api/reopen itself — not just the helper (T04 lesson)."""
        import tempfile
        from io import BytesIO

        claude_sid = "api-reopen-claude-99"
        grok_sid = "d4e5f6a7-b8c9-0123-def0-123456789abc"
        room_id = "room-2026-08-03-apireopen"
        msgs = self._archive_msgs()
        max_id = msgs[-1]["id"]

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_empty_current(rooms)
            self._write_archive(
                rooms, room_id,
                sessions=self._rich_sessions(claude_sid, grok_sid),
                messages=msgs,
                last_seen={self.claude_seat: max_id, self.grok_seat: max_id},
                next_id=max_id + 1,
            )

            body = json.dumps({"id": room_id}).encode()
            h = self.server.Handler.__new__(self.server.Handler)
            h.path = "/api/reopen"
            h.headers = {"Content-Length": str(len(body))}
            h.rfile = BytesIO(body)
            h.wfile = BytesIO()
            h.request_version = "HTTP/1.1"
            h.command = "POST"
            h.client_address = ("127.0.0.1", 9)
            h.close_connection = False
            h.log_message = lambda *a, **k: None
            status = {}
            payload_out = {}

            def send_response(code, message=None):
                status["code"] = code

            def write(data):
                h.wfile.write(data)
                try:
                    payload_out.update(json.loads(data.decode()))
                except Exception:
                    pass

            h.send_response = send_response
            h.send_header = lambda *a, **k: None
            h.end_headers = lambda: None
            orig_write = h.wfile.write

            # Capture JSON body via _json path: Handler._json writes to wfile.
            def capturing_json(obj, code=200):
                status["code"] = code
                payload_out.clear()
                payload_out.update(obj)
                body_b = json.dumps(obj).encode()
                h.wfile.write(body_b)

            h._json = capturing_json

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                h.do_POST()

            self.assertEqual(status.get("code"), 200)
            self.assertTrue(payload_out.get("ok"))
            self.assertTrue(payload_out.get("continued"))
            live = json.loads((rooms / "current" / "state.json").read_text())
            self.assertEqual(live["sessions"][self.claude_seat], claude_sid)
            self.assertEqual(live["sessions"][self.grok_seat], grok_sid)
            self.assertFalse((rooms / room_id).exists())

    def test_filing_live_then_new_room_still_works(self):
        """Criterion 5: New room still files current and starts clean."""
        import tempfile
        from io import BytesIO

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            current = rooms / "current"
            current.mkdir()
            (current / "state.json").write_text(json.dumps({
                "claude_session": None,
                "kimi_started": False,
                "last_seen": {},
                "mission": None,
                "stop": False,
                "next_id": 3,
                "sessions": {self.claude_seat: "keep-me-in-archive"},
            }, indent=2))
            with (current / "messages.jsonl").open("w") as fh:
                fh.write(json.dumps({
                    "id": 1, "ts": "t", "from": "you",
                    "text": "live chat before newroom", "hops": 0, "kind": "chat",
                }) + "\n")
                fh.write(json.dumps({
                    "id": 2, "ts": "t", "from": self.claude_seat,
                    "text": "ok", "hops": 1, "kind": "chat",
                }) + "\n")

            body = b"{}"
            h = self.server.Handler.__new__(self.server.Handler)
            h.path = "/api/newroom"
            h.headers = {"Content-Length": "2"}
            h.rfile = BytesIO(body)
            h.wfile = BytesIO()
            h.request_version = "HTTP/1.1"
            h.command = "POST"
            h.client_address = ("127.0.0.1", 9)
            h.close_connection = False
            h.log_message = lambda *a, **k: None
            status = {}
            h.send_response = lambda code, message=None: status.update(code=code)
            h.send_header = lambda *a, **k: None
            h.end_headers = lambda: None
            stamp = "2026-08-03-newroom5"
            fixed_dt = mock.Mock()
            fixed_dt.now.return_value.strftime.return_value = stamp
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", current), \
                 mock.patch.object(self.server, "datetime", fixed_dt):
                h.do_POST()
            self.assertEqual(status.get("code"), 200)
            archived = rooms / f"room-{stamp}"
            self.assertTrue(archived.is_dir())
            filed = json.loads((archived / "state.json").read_text())
            self.assertEqual(
                filed["sessions"][self.claude_seat]["session_id"],
                "keep-me-in-archive",
            )
            # Fresh current: no leftover sessions from the filed chat.
            fresh = json.loads((rooms / "current" / "state.json").read_text())
            self.assertFalse(fresh.get("sessions"))

    # ------------------------------------------------------------------
    # Round-2 review findings (T05 re-review): seven pins.
    # ------------------------------------------------------------------

    def test_grok_resumed_prompt_excludes_archived_minutes(self):
        """Finding 1: codeword in archive minutes must not enter grok's prompt
        after reopen, while resume still carries the stored session id."""
        import tempfile

        codeword = "BANANAPHONE"
        grok_sid = "b0a1n2a3-n4a5-6789-phon-e01234567890"
        claude_sid = "resume-claude-minutes-T05"
        room_id = "room-2026-08-03-minutes-ban"
        msgs = self._archive_msgs()
        max_id = msgs[-1]["id"]

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_empty_current(rooms)
            arch = self._write_archive(
                rooms, room_id,
                sessions=self._rich_sessions(claude_sid, grok_sid),
                messages=msgs,
                last_seen={self.claude_seat: max_id, self.grok_seat: max_id},
                next_id=max_id + 1,
            )
            (arch / "minutes.md").write_text(
                f"# Minutes\n- secret codeword {codeword} was decided\n"
            )

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                result = self.server.reopen_room(room_id)
                self.assertTrue(result.get("continued"), f"expected continue: {result}")
                state = self.server.load_state()
                self.assertEqual(
                    self.server.get_session(state, self.grok_seat), grok_sid,
                )
                # Same delta feed the worker would build after reopen.
                last_g = state["last_seen"].get(self.grok_seat, 0)
                fresh_g = [m for m in self.server.all_messages()
                           if m["id"] > last_g and m["from"] != self.grok_seat]
                # Force a non-empty feed so build_prompt is exercised.
                if not fresh_g:
                    self.server.append_message("you", f"@{self.grok_seat} resume check")
                    state = self.server.load_state()
                    last_g = state["last_seen"].get(self.grok_seat, 0)
                    fresh_g = [m for m in self.server.all_messages()
                               if m["id"] > last_g and m["from"] != self.grok_seat]
                prompt_g = self.server.build_prompt(self.grok_seat, fresh_g)

            self.assertNotIn(
                codeword, prompt_g,
                "archived minutes must not be injected into grok's resume prompt",
            )
            # Session still attached (resume path, not a fake minutes workaround).
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                live = self.server.load_state()
            self.assertEqual(self.server.get_session(live, self.grok_seat), grok_sid)

    def test_reopen_preserves_live_messages_in_rooms(self):
        """Finding 2: reopening must file the live chat first. Without
        file_current_room before rmtree, this test fails (live chat gone)."""
        import tempfile

        live_marker = "LIVE_CHAT_MUST_SURVIVE_T05_xyz"
        claude_sid = "live-claude-sid"
        grok_sid = "c1d2e3f4-a5b6-7890-cdef-1234567890ab"
        arch_sid_c = "arch-claude-sid"
        arch_sid_g = "d2e3f4a5-b6c7-8901-def0-234567890abc"
        room_id = "room-2026-08-03-preserve-live"
        arch_msgs = self._archive_msgs()
        max_id = arch_msgs[-1]["id"]

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            current = rooms / "current"
            current.mkdir()
            (current / "state.json").write_text(json.dumps({
                "claude_session": None,
                "kimi_started": False,
                "last_seen": {self.claude_seat: 2, self.grok_seat: 2},
                "mission": None,
                "stop": False,
                "next_id": 3,
                "sessions": {
                    self.claude_seat: claude_sid,
                    self.grok_seat: grok_sid,
                },
            }, indent=2))
            with (current / "messages.jsonl").open("w") as fh:
                fh.write(json.dumps({
                    "id": 1, "ts": "t1", "from": "you",
                    "text": live_marker, "hops": 0, "kind": "chat",
                }) + "\n")
                fh.write(json.dumps({
                    "id": 2, "ts": "t2", "from": self.claude_seat,
                    "text": "reply to live", "hops": 1, "kind": "chat",
                }) + "\n")

            self._write_archive(
                rooms, room_id,
                sessions=self._rich_sessions(arch_sid_c, arch_sid_g),
                messages=arch_msgs,
                last_seen={self.claude_seat: max_id, self.grok_seat: max_id},
                next_id=max_id + 1,
            )

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                result = self.server.reopen_room(room_id)

            self.assertTrue(result.get("continued"), f"reopen failed: {result}")
            # Live messages must still be findable somewhere under rooms/.
            found = []
            for d in rooms.iterdir():
                if not d.is_dir():
                    continue
                mf = d / "messages.jsonl"
                if not mf.exists():
                    continue
                body = mf.read_text()
                if live_marker in body:
                    found.append(d.name)
            self.assertTrue(
                found,
                "live chat messages vanished from rooms/ after reopen "
                "(file_current_room before delete is missing)",
            )
            # And they must not only live in the reopened current (filed aside).
            self.assertTrue(
                any(name != "current" for name in found),
                "live chat must be filed as an archive, not only remain as current",
            )

    def test_mid_turn_reopen_cannot_append_foreign_reply(self):
        """Finding 3: agent mid-turn must not land its reply in the reopened chat."""
        import tempfile

        foreign = "FOREIGN_REPLY_FROM_OLD_ROOM_T05"
        arch_sid_c = "arch-c-mid"
        arch_sid_g = "e3f4a5b6-c7d8-9012-ef01-34567890abcd"
        room_id = "room-2026-08-03-midturn"
        arch_msgs = self._archive_msgs()
        max_id = arch_msgs[-1]["id"]

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            current = rooms / "current"
            current.mkdir()
            (current / "state.json").write_text(json.dumps({
                "claude_session": None,
                "kimi_started": False,
                "last_seen": {self.claude_seat: 1},
                "mission": None,
                "stop": False,
                "next_id": 2,
                "sessions": {self.claude_seat: "old-live-sid"},
            }, indent=2))
            with (current / "messages.jsonl").open("w") as fh:
                fh.write(json.dumps({
                    "id": 1, "ts": "t1", "from": "you",
                    "text": f"@{self.claude_seat} work please",
                    "hops": 0, "kind": "chat",
                }) + "\n")

            self._write_archive(
                rooms, room_id,
                sessions=self._rich_sessions(arch_sid_c, arch_sid_g),
                messages=arch_msgs,
                last_seen={self.claude_seat: max_id, self.grok_seat: max_id},
                next_id=max_id + 1,
            )

            # Reset busy flags from any prior test noise.
            for a in list(self.server.BUSY):
                self.server.BUSY[a] = False

            # --- Path A: refuse while busy ---
            self.server.BUSY[self.claude_seat] = True
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                refused = self.server.reopen_room(room_id)
            self.assertEqual(
                refused.get("error"), "busy",
                f"must refuse reopen while agent busy: {refused}",
            )
            # Archive and live current must be untouched.
            self.assertTrue((rooms / room_id).is_dir())
            self.assertTrue((rooms / "current" / "messages.jsonl").exists())
            live_body = (rooms / "current" / "messages.jsonl").read_text()
            self.assertNotIn(foreign, live_body)
            self.server.BUSY[self.claude_seat] = False

            # --- Path B: in-flight worker past the busy check (epoch guard) ---
            # Capture epoch, bump via reopen, then try set_session + append with
            # the stale epoch — both must be no-ops on the reopened room.
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                epoch_before = getattr(self.server, "ROOM_EPOCH", 0)
                result = self.server.reopen_room(room_id)
                self.assertTrue(
                    result.get("continued"),
                    f"reopen after busy clear failed: {result}",
                )
                epoch_after = self.server.ROOM_EPOCH
                self.assertNotEqual(
                    epoch_before, epoch_after,
                    "reopen must bump ROOM_EPOCH so in-flight workers cannot write",
                )
                # Stale write attempts from the old room's worker.
                self.server.set_session(
                    self.claude_seat, "poison-sid", room_epoch=epoch_before)
                stale_msg = self.server.append_message(
                    self.claude_seat, foreign, hops=1, room_epoch=epoch_before)
                self.assertIsNone(
                    stale_msg,
                    "append_message with stale room_epoch must discard the write",
                )
                live = self.server.load_state()
                # Session must still be the reattached archive id, not poison.
                self.assertEqual(
                    self.server.get_session(live, self.claude_seat), arch_sid_c,
                )
                reopened_body = (rooms / "current" / "messages.jsonl").read_text()
                self.assertNotIn(
                    foreign, reopened_body,
                    "foreign mid-turn reply must not land in reopened chat",
                )

    def test_bad_room_ids_rejected(self):
        """Finding 5: room-id guard is the security boundary — pin it."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_empty_current(rooms)
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                for bad in ("current", "..", "room-../x"):
                    result = self.server.reopen_room(bad)
                    self.assertEqual(
                        result.get("error"), "bad room id",
                        f"id {bad!r} must be rejected as bad room id, got {result}",
                    )
            # current/ must still exist after the bad attempts
            self.assertTrue((rooms / "current").is_dir())

    def test_next_id_reconciled_with_transcript(self):
        """Finding 6: next_id must be max(stored, last_id+1) after reopen."""
        import tempfile

        room_id = "room-2026-08-03-nextid"
        # Transcript ends at id 10, but stored next_id is stale (3).
        msgs = [
            {"id": 1, "ts": "t1", "from": "you", "text": "hi",
             "hops": 0, "kind": "chat"},
            {"id": 10, "ts": "t10", "from": self.claude_seat, "text": "bye",
             "hops": 1, "kind": "chat"},
        ]
        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_empty_current(rooms)
            self._write_archive(
                rooms, room_id,
                sessions=self._rich_sessions("c-next", "f4a5b6c7-d8e9-0123-f012-4567890abcde"),
                messages=msgs,
                last_seen={self.claude_seat: 10, self.grok_seat: 10},
                next_id=3,  # stale — at or below last id
            )
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                result = self.server.reopen_room(room_id)
                self.assertTrue(result.get("continued"), result)
                live = self.server.load_state()
                self.assertGreaterEqual(
                    int(live["next_id"]), 11,
                    "next_id must be at least last_id+1 after reopen",
                )
                msg = self.server.append_message("you", "after reopen")
                self.assertGreaterEqual(msg["id"], 11)
                # No duplicate of any prior id.
                ids = [m["id"] for m in self.server.all_messages()]
                self.assertEqual(len(ids), len(set(ids)), f"duplicate ids: {ids}")


class TestMinutesSummarizerHaikuException(unittest.TestCase):
    """DM-14 / T10: minutes summarizer stays on Haiku.

    Asserts the model value actually passed to the CLI at call time — not a
    source-text grep (which would stay green if the string only lived in a
    comment). Mutating MINUTES_MODEL must fail this test; deleting the
    exception comment must not.
    """

    EXPECTED_MODEL = "claude-haiku-4-5-20251001"

    @classmethod
    def setUpClass(cls):
        import importlib.util
        name = "agent_room_server_t10_minutes"
        if name in sys.modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, ROOM / "server.py")
        cls.server = importlib.util.module_from_spec(spec)
        with mock.patch("http.server.ThreadingHTTPServer"):
            spec.loader.exec_module(cls.server)

    def _model_from_cmd(self, cmd):
        """Pull the value after --model from a CLI argv list."""
        try:
            i = list(cmd).index("--model")
        except ValueError:
            self.fail(f"minutes CLI call missing --model: {cmd!r}")
        if i + 1 >= len(cmd):
            self.fail(f"--model has no value in cmd: {cmd!r}")
        return cmd[i + 1]

    def test_update_minutes_passes_haiku_model_at_call_time(self):
        """The summarizer hands Haiku to the runner — silent swap fails here."""
        import tempfile

        srv = self.server
        self.assertTrue(
            hasattr(srv, "MINUTES_MODEL"),
            "server must expose MINUTES_MODEL for the minutes summarizer",
        )
        # Constant itself must be Haiku (the pin).
        self.assertEqual(srv.MINUTES_MODEL, self.EXPECTED_MODEL)

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            current = rooms / "current"
            current.mkdir()
            state = {
                "claude_session": None,
                "kimi_started": False,
                "last_seen": {},
                "mission": None,
                "stop": False,
                "next_id": 3,
                "sessions": {},
                "minuted_upto": 0,
            }
            (current / "state.json").write_text(json.dumps(state, indent=2))
            with (current / "messages.jsonl").open("w") as fh:
                fh.write(json.dumps({
                    "id": 1, "ts": "t1", "from": "you",
                    "text": "hello room", "hops": 0, "kind": "chat",
                }) + "\n")
                fh.write(json.dumps({
                    "id": 2, "ts": "t2", "from": "claude",
                    "text": "hi", "hops": 1, "kind": "chat",
                }) + "\n")

            fake = FakeRun(_proc(stdout=json.dumps({"result": "# Minutes\n- said hello\n"})))
            # Clear busy so update_minutes can finish cleanly.
            srv.MINUTES_BUSY.clear()
            with mock.patch.object(srv, "ROOMS_DIR", rooms), \
                 mock.patch.object(srv, "CURRENT", current), \
                 mock.patch.object(srv.subprocess, "run", side_effect=fake):
                srv.update_minutes()

            self.assertTrue(fake.calls, "update_minutes must invoke the runner")
            cmd, _kwargs = fake.calls[0]
            model_used = self._model_from_cmd(cmd)
            # Call-time value is what matters — not a source grep.
            self.assertEqual(
                model_used,
                self.EXPECTED_MODEL,
                f"minutes summarizer must run on Haiku at call time; got {model_used!r}",
            )
            # And that value must come from the constant (not a hard-coded
            # string elsewhere that drifts from MINUTES_MODEL).
            self.assertEqual(model_used, srv.MINUTES_MODEL)


class TestModelPicker(unittest.TestCase):
    """DM-11 / T06: seat editor model dropdown lists real models per engine.

    Seams: engines.list_models (source files + fallbacks), server /api/models,
    save_roster + adapter.call (saved model used on next reply).
    Tests use fixtures only — never depend on the user's live ~/.grok or
    ~/.kimi-code files.
    """

    def setUp(self):
        self._tmp = Path(self._mk_tmp())
        self.grok_cache = self._tmp / "models_cache.json"
        self.kimi_cfg = self._tmp / "config.toml"
        self.logs: list[str] = []

    def _mk_tmp(self) -> str:
        import tempfile
        return tempfile.mkdtemp(prefix="ar-models-")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _log(self, msg: str) -> None:
        self.logs.append(str(msg))

    def _write_grok_cache(self, models: dict) -> None:
        self.grok_cache.write_text(json.dumps({
            "fetched_at": "2026-08-03T00:00:00Z",
            "models": models,
        }))

    def _write_kimi_cfg(self, body: str) -> None:
        self.kimi_cfg.write_text(body)

    def test_claude_list_is_three_fixed_aliases(self):
        """Criterion 1: claude offers exactly fable/opus/sonnet by display name."""
        from engines import list_models
        models = list_models(
            "claude",
            grok_cache_path=self.grok_cache,
            kimi_config_path=self.kimi_cfg,
            log=self._log,
        )
        ids = [m["id"] for m in models]
        names = [m["name"] for m in models]
        self.assertEqual(ids, ["fable", "opus", "sonnet"])
        self.assertEqual(names, ["Fable 5", "Opus 5", "Sonnet 5"])
        # Fable and Opus must not be conflated (design-contract defect).
        self.assertNotEqual(
            next(m["name"] for m in models if m["id"] == "fable"),
            next(m["name"] for m in models if m["id"] == "opus"),
        )

    def test_grok_list_from_cache_fixture(self):
        """Criterion 1: grok ids+names come from models_cache.json, not hardcode alone."""
        from engines import list_models
        self._write_grok_cache({
            "grok-4.5": {
                "info": {"id": "grok-4.5", "name": "Grok 4.5"},
            },
            "grok-test-only": {
                "info": {"id": "grok-test-only", "name": "Grok Test Only"},
            },
        })
        models = list_models(
            "grok",
            grok_cache_path=self.grok_cache,
            kimi_config_path=self.kimi_cfg,
            log=self._log,
        )
        by_id = {m["id"]: m["name"] for m in models}
        self.assertEqual(by_id["grok-4.5"], "Grok 4.5")
        self.assertEqual(by_id["grok-test-only"], "Grok Test Only")
        self.assertEqual(set(by_id), {"grok-4.5", "grok-test-only"})
        self.assertEqual(self.logs, [], "readable cache must not fall back")

    def test_kimi_list_from_config_fixture(self):
        """Criterion 1: kimi ids+names come from [models.\"…\"] tables.

        Fixture ids must be absent from _DISPLAY_NAMES so a mutation that
        ignores display_name from the file cannot hide behind the hard-coded
        map (same shape as grok-test-only).
        """
        from engines import list_models, _DISPLAY_NAMES
        unique_id = "kimi-code/kimi-test-only"
        unique_name = "Kimi Test Only From Config"
        self.assertNotIn(
            unique_id, _DISPLAY_NAMES,
            "fixture id must not be pre-mapped or this test is vacuous",
        )
        self._write_kimi_cfg(
            'default_model = "kimi-code/kimi-test-only"\n\n'
            f'[models."{unique_id}"]\n'
            'provider = "managed:kimi-code"\n'
            'model = "kimi-test-only"\n'
            f'display_name = "{unique_name}"\n\n'
            '[models."kimi-code/kimi-test-only-b"]\n'
            'provider = "managed:kimi-code"\n'
            'model = "kimi-test-only-b"\n'
            'display_name = "Kimi Test Only B From Config"\n'
        )
        models = list_models(
            "kimi",
            grok_cache_path=self.grok_cache,
            kimi_config_path=self.kimi_cfg,
            log=self._log,
        )
        by_id = {m["id"]: m["name"] for m in models}
        self.assertEqual(by_id[unique_id], unique_name)
        self.assertEqual(
            by_id["kimi-code/kimi-test-only-b"],
            "Kimi Test Only B From Config",
        )
        self.assertEqual(
            set(by_id),
            {unique_id, "kimi-code/kimi-test-only-b"},
        )
        self.assertEqual(self.logs, [])

    def test_haiku_absent_from_every_engine_and_fallback(self):
        """Criterion 2: Haiku never appears — live lists and all fallbacks."""
        from engines import list_models, FALLBACK_MODELS
        # Poison fixtures with haiku-like entries; they must be filtered out.
        self._write_grok_cache({
            "grok-4.5": {"info": {"id": "grok-4.5", "name": "Grok 4.5"}},
            "haiku-sneak": {"info": {"id": "haiku-sneak", "name": "Haiku Sneak"}},
            "x": {"info": {"id": "x", "name": "Something Haiku-ish"}},
        })
        self._write_kimi_cfg(
            '[models."kimi-code/k3"]\n'
            'display_name = "K3"\n\n'
            '[models."kimi-code/haiku-x"]\n'
            'display_name = "Haiku X"\n\n'
            '[models."kimi-code/ok"]\n'
            'display_name = "OK Model"\n'
        )
        for engine in ("claude", "grok", "kimi"):
            models = list_models(
                engine,
                grok_cache_path=self.grok_cache,
                kimi_config_path=self.kimi_cfg,
                log=self._log,
            )
            self.assertTrue(models, f"{engine} list must not be empty")
            for m in models:
                blob = f"{m.get('id', '')} {m.get('name', '')}".lower()
                self.assertNotIn("haiku", blob, f"{engine} leaked Haiku: {m}")
        # Fallbacks themselves must also be Haiku-free (used when files missing).
        for engine, models in FALLBACK_MODELS.items():
            self.assertTrue(models, f"fallback for {engine} must not be empty")
            for m in models:
                blob = f"{m.get('id', '')} {m.get('name', '')}".lower()
                self.assertNotIn(
                    "haiku", blob, f"fallback {engine} leaked Haiku: {m}")

    def test_missing_source_falls_back_and_logs_never_empty(self):
        """Criterion 3: missing/unreadable source → fallback list + log line."""
        from engines import list_models, FALLBACK_MODELS
        # Paths do not exist.
        missing_grok = self._tmp / "no-such-cache.json"
        missing_kimi = self._tmp / "no-such-config.toml"
        self.assertFalse(missing_grok.exists())
        self.assertFalse(missing_kimi.exists())

        grok_models = list_models(
            "grok",
            grok_cache_path=missing_grok,
            kimi_config_path=missing_kimi,
            log=self._log,
        )
        self.assertEqual(grok_models, FALLBACK_MODELS["grok"])
        self.assertTrue(grok_models, "dropdown must never be empty")
        self.assertTrue(
            any("grok" in line.lower() and (
                "fallback" in line.lower() or "missing" in line.lower()
                or "unreadable" in line.lower())
                for line in self.logs),
            f"expected fallback log for grok; got {self.logs!r}",
        )

        self.logs.clear()
        # Unreadable (directory, not a file).
        bad_kimi = self._tmp / "kimi-as-dir"
        bad_kimi.mkdir()
        kimi_models = list_models(
            "kimi",
            grok_cache_path=missing_grok,
            kimi_config_path=bad_kimi,
            log=self._log,
        )
        self.assertEqual(kimi_models, FALLBACK_MODELS["kimi"])
        self.assertTrue(kimi_models)
        self.assertTrue(
            any("kimi" in line.lower() for line in self.logs),
            f"expected fallback log for kimi; got {self.logs!r}",
        )

        # Corrupt grok JSON → fallback too.
        self.logs.clear()
        corrupt = self._tmp / "corrupt.json"
        corrupt.write_text("{not json")
        models = list_models(
            "grok",
            grok_cache_path=corrupt,
            kimi_config_path=missing_kimi,
            log=self._log,
        )
        self.assertEqual(models, FALLBACK_MODELS["grok"])
        self.assertTrue(self.logs)

    def test_api_models_handler_returns_exact_lists(self):
        """Handler seam: GET /api/models returns per-engine lists (not just helper)."""
        import importlib.util
        name = "agent_room_server_t06_models_api"
        if name in sys.modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, ROOM / "server.py")
        server = importlib.util.module_from_spec(spec)
        with mock.patch("http.server.ThreadingHTTPServer"):
            spec.loader.exec_module(server)

        self._write_grok_cache({
            "grok-4.5": {"info": {"id": "grok-4.5", "name": "Grok 4.5"}},
        })
        self._write_kimi_cfg(
            '[models."kimi-code/k3"]\n'
            'display_name = "K3"\n'
        )
        # Point the server at fixtures so the handler is exercised end-to-end.
        with mock.patch.object(server, "GROK_MODELS_CACHE", self.grok_cache), \
             mock.patch.object(server, "KIMI_CONFIG_PATH", self.kimi_cfg):
            # Drive the handler via do_GET path if present; else pure function.
            if hasattr(server, "models_for_all_engines"):
                payload = server.models_for_all_engines()
            else:
                # Fallback: call list_models via engines through server module.
                payload = {
                    eng: server.list_models(eng) for eng in ("claude", "grok", "kimi")
                }

        self.assertEqual(
            [m["id"] for m in payload["claude"]],
            ["fable", "opus", "sonnet"],
        )
        self.assertEqual(
            [m["name"] for m in payload["claude"]],
            ["Fable 5", "Opus 5", "Sonnet 5"],
        )
        self.assertEqual(
            [(m["id"], m["name"]) for m in payload["grok"]],
            [("grok-4.5", "Grok 4.5")],
        )
        self.assertEqual(
            [(m["id"], m["name"]) for m in payload["kimi"]],
            [("kimi-code/k3", "K3")],
        )
        for eng, models in payload.items():
            for m in models:
                self.assertNotIn("haiku", f"{m['id']} {m['name']}".lower())

        # Also hit the HTTP handler itself when available.
        if hasattr(server, "Handler"):
            from io import BytesIO
            handler = server.Handler.__new__(server.Handler)
            handler.path = "/api/models"
            captured = {}

            def _json(obj, status=200):
                captured["body"] = obj
                captured["status"] = status

            handler._json = _json
            with mock.patch.object(server, "GROK_MODELS_CACHE", self.grok_cache), \
                 mock.patch.object(server, "KIMI_CONFIG_PATH", self.kimi_cfg):
                handler.do_GET()
            self.assertEqual(captured.get("status"), 200)
            body = captured["body"]
            self.assertEqual(
                [m["id"] for m in body["claude"]],
                ["fable", "opus", "sonnet"],
            )
            self.assertEqual(body["grok"][0]["id"], "grok-4.5")
            self.assertEqual(body["kimi"][0]["name"], "K3")

    def test_saved_model_used_on_next_reply_after_reload(self):
        """Criterion 5: saved model survives roster reload and is passed to the CLI."""
        import importlib.util
        import tempfile
        name = "agent_room_server_t06_saved_model"
        if name in sys.modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, ROOM / "server.py")
        server = importlib.util.module_from_spec(spec)
        with mock.patch("http.server.ThreadingHTTPServer"):
            spec.loader.exec_module(server)

        with tempfile.TemporaryDirectory(prefix="ar-roster-") as td:
            root = Path(td)
            agents = {
                "_help": "test",
                "seats": [
                    {
                        "name": "claude",
                        "engine": "claude",
                        "model": "fable",
                        "effort": "high",
                        "color": "#111",
                        "role": "chair",
                    },
                    {
                        "name": "grok",
                        "engine": "grok",
                        "model": None,
                        "effort": None,
                        "color": "#222",
                        "role": "builder",
                    },
                    {
                        "name": "kimi",
                        "engine": "kimi",
                        "model": None,
                        "effort": None,
                        "color": "#333",
                        "role": "reviewer",
                    },
                ],
            }
            agents_path = root / "agents.json"
            agents_path.write_text(json.dumps(agents, indent=2) + "\n")
            with mock.patch.object(server, "ROOM_ROOT", root):
                # Simulate seat editor save: change grok model, persist, reload.
                err = server.save_roster_change({
                    "name": "grok",
                    "engine": "grok",
                    "model": "grok-4.5",
                    "effort": "",
                    "color": "#222",
                    "role": "builder",
                })
                self.assertIsNone(err, err)
                # "Server restart": re-read roster from disk.
                server.rebuild_roster()
                seat = server.SEATS["grok"]
                self.assertEqual(seat.get("model"), "grok-4.5")

                # Next reply: adapter must pass -m grok-4.5 (or --model).
                run = FakeRun(_proc(stdout="built"))
                result = GrokAdapter().call(seat, "build it", None, run)
                self.assertIsNone(result.error)
                cmd = run.calls[0][0]
                # Prefer -m (grok's flag); also accept --model for robustness.
                if "-m" in cmd:
                    self.assertEqual(cmd[cmd.index("-m") + 1], "grok-4.5")
                elif "--model" in cmd:
                    self.assertEqual(cmd[cmd.index("--model") + 1], "grok-4.5")
                else:
                    self.fail(f"saved model not on command line: {cmd}")

                # Claude path: saved alias reaches --model.
                err = server.save_roster_change({
                    "name": "claude",
                    "engine": "claude",
                    "model": "sonnet",
                    "effort": "low",
                    "color": "#111",
                    "role": "chair",
                })
                self.assertIsNone(err, err)
                server.rebuild_roster()
                cseat = server.SEATS["claude"]
                self.assertEqual(cseat.get("model"), "sonnet")
                payload = {
                    "result": "ok",
                    "session_id": "s1",
                    "model": "sonnet",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
                crun = FakeRun(_proc(stdout=json.dumps(payload)))
                ClaudeAdapter().call(cseat, "hi", None, crun)
                ccmd = crun.calls[0][0]
                self.assertIn("--model", ccmd)
                self.assertEqual(ccmd[ccmd.index("--model") + 1], "sonnet")

    def test_html_dropdown_no_haiku_and_loads_from_api(self):
        """Seat editor pulls model lists from the API; Haiku not hard-coded in UI."""
        html = (ROOM / "index.html").read_text()
        # Must talk to the models endpoint (not only a static MODELS map with Haiku).
        self.assertIn("/api/models", html)
        # Engine change must re-fill the model select without reload.
        self.assertIn("fillModelOptions", html)
        self.assertIn("dEngine", html)
        # Standing rule: no Haiku option in the seat editor catalog.
        # (Minutes summarizer is a separate exception, not in this dropdown.)
        # Strip the free-text custom path if present; flag any haiku option value.
        self.assertNotRegex(
            html,
            r"claude-haiku|haiku-4|['\"]haiku['\"]",
            "seat editor must not offer Haiku",
        )
        # Display names for the three claude aliases must be distinct.
        # Prefer live API data; static map may remain as a fallback cache.
        if "Fable 5" in html and "Opus 5" in html:
            # If both appear in a static map, they must not share a key incorrectly.
            # The known mock defect mapped fable → Opus; guard against that.
            self.assertNotRegex(
                html,
                r"claude-fable-5['\"]\s*[,:]\s*['\"]Opus 5",
            )
            self.assertNotRegex(
                html,
                r"['\"]fable['\"]\s*[,:)\]]\s*['\"]Opus 5",
            )

    def _extract_js_function(self, html: str, name: str) -> str:
        marker = f"function {name}"
        idx = html.find(marker)
        self.assertGreater(idx, -1, f"{name} not found in index.html")
        brace = html.find("{", idx)
        depth = 0
        i = brace
        while i < len(html):
            if html[i] == "{":
                depth += 1
            elif html[i] == "}":
                depth -= 1
                if depth == 0:
                    return html[idx : i + 1]
            i += 1
        self.fail(f"unbalanced braces extracting {name}")

    def _run_node(self, script: str) -> str:
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            self.fail(
                f"node failed ({proc.returncode}):\n"
                f"stdout={proc.stdout}\nstderr={proc.stderr}"
            )
        return proc.stdout

    def test_seat_editor_dropdown_behaviour(self):
        """Criterion 4: real fillModelOptions drives options by engine.

        Given a roster catalog: correct option list per engine, engine change
        swaps the list, chosen model round-trips (not HTML-source-only).
        """
        html = (ROOM / "index.html").read_text()
        normalize = self._extract_js_function(html, "normalizeModelId")
        models_for = self._extract_js_function(html, "modelsForEngine")
        fill = self._extract_js_function(html, "fillModelOptions")
        chosen = self._extract_js_function(html, "chosenModel")
        # MODEL_ID_ALIASES is a const the normalize function closes over.
        alias_m = re.search(
            r"const MODEL_ID_ALIASES\s*=\s*\{[^}]+\};",
            html,
        )
        self.assertIsNotNone(alias_m, "MODEL_ID_ALIASES const missing")
        script = f"""
{alias_m.group(0)}
let MODELS_BY_ENGINE = {{
  claude: [
    {{id:'fable', name:'Fable 5'}},
    {{id:'opus', name:'Opus 5'}},
    {{id:'sonnet', name:'Sonnet 5'}},
  ],
  grok: [
    {{id:'grok-4.5', name:'Grok 4.5'}},
    {{id:'grok-test-only', name:'Grok Test Only'}},
  ],
  kimi: [
    {{id:'kimi-code/kimi-test-only', name:'Kimi Test Only From Config'}},
    {{id:'kimi-code/k3', name:'K3'}},
  ],
}};
const MODELS_FALLBACK = MODELS_BY_ENGINE;
const selState = {{ opts: [], value: '', disabled: false }};
const txtState = {{ display: '', value: '' }};
let descText = '';
const document = {{
  getElementById(id) {{
    if (id === 'dModelSel') {{
      return {{
        get options() {{ return selState.opts; }},
        get value() {{ return selState.value; }},
        set value(v) {{ selState.value = v; }},
        set disabled(v) {{ selState.disabled = !!v; }},
        get disabled() {{ return selState.disabled; }},
        set innerHTML(html) {{
          selState.opts = [];
          const re = /<option value="([^"]*)">([^<]*)<\\/option>/g;
          let m;
          while ((m = re.exec(html))) {{
            selState.opts.push({{ value: m[1], text: m[2] }});
          }}
        }},
      }};
    }}
    if (id === 'dModel') {{
      return {{
        style: {{
          get display() {{ return txtState.display; }},
          set display(v) {{ txtState.display = v; }},
        }},
        get value() {{ return txtState.value; }},
        set value(v) {{ txtState.value = v; }},
      }};
    }}
    if (id === 'dModelDesc') {{
      return {{
        get textContent() {{ return descText; }},
        set textContent(v) {{ descText = v; }},
      }};
    }}
    return null;
  }},
}};
{normalize}
{models_for}
{fill}
{chosen}

function optionValues() {{
  return selState.opts.map(o => o.value);
}}
function optionLabels() {{
  return selState.opts.map(o => o.text);
}}

// 1) Claude roster → fable/opus/sonnet (+ engine default); current selected.
fillModelOptions('claude', 'opus');
const claudeVals = optionValues();
if (JSON.stringify(claudeVals) !== JSON.stringify(['','fable','opus','sonnet'])) {{
  console.error('FAIL claude options', claudeVals);
  process.exit(1);
}}
if (selState.value !== 'opus') {{
  console.error('FAIL claude current not selected', selState.value);
  process.exit(1);
}}
if (optionLabels()[1] !== 'Fable 5' || optionLabels()[2] !== 'Opus 5') {{
  console.error('FAIL claude labels', optionLabels());
  process.exit(1);
}}

// 2) Engine change swaps the list immediately (criterion 4).
fillModelOptions('grok', '');
const grokVals = optionValues();
if (JSON.stringify(grokVals) !== JSON.stringify(['','grok-4.5','grok-test-only'])) {{
  console.error('FAIL grok options after engine swap', grokVals);
  process.exit(1);
}}
if (selState.value !== '') {{
  console.error('FAIL engine swap should reset to default', selState.value);
  process.exit(1);
}}
// User picks a model.
selState.value = 'grok-test-only';
if (chosenModel() !== 'grok-test-only') {{
  console.error('FAIL chosenModel after pick', chosenModel());
  process.exit(1);
}}

// 3) Round-trip: open seat with saved kimi model → same id selected.
fillModelOptions('kimi', 'kimi-code/kimi-test-only');
const kimiVals = optionValues();
if (!kimiVals.includes('kimi-code/kimi-test-only')) {{
  console.error('FAIL kimi options missing saved id', kimiVals);
  process.exit(1);
}}
if (selState.value !== 'kimi-code/kimi-test-only') {{
  console.error('FAIL kimi round-trip select', selState.value);
  process.exit(1);
}}
if (chosenModel() !== 'kimi-code/kimi-test-only') {{
  console.error('FAIL kimi round-trip chosenModel', chosenModel());
  process.exit(1);
}}
// Alias normalize: old full claude id → alias option.
fillModelOptions('claude', 'claude-opus-5');
if (selState.value !== 'opus') {{
  console.error('FAIL alias normalize', selState.value);
  process.exit(1);
}}
console.log('dropdown-ok');
"""
        out = self._run_node(script)
        self.assertIn("dropdown-ok", out)

    def test_display_name_map_one_place(self):
        """Design rule: ids map to display names in one place; Fable ≠ Opus."""
        from engines import model_display_name
        self.assertEqual(model_display_name("fable"), "Fable 5")
        self.assertEqual(model_display_name("opus"), "Opus 5")
        self.assertEqual(model_display_name("sonnet"), "Sonnet 5")
        self.assertEqual(model_display_name("claude-fable-5"), "Fable 5")
        self.assertEqual(model_display_name("claude-opus-5"), "Opus 5")
        self.assertEqual(model_display_name("claude-sonnet-5"), "Sonnet 5")
        self.assertEqual(model_display_name("grok-4.5"), "Grok 4.5")
        self.assertNotEqual(
            model_display_name("claude-fable-5"),
            model_display_name("claude-opus-5"),
        )


class TestSidebarRedesign(unittest.TestCase):
    """DM-01..05 / T07: redesigned sidebar — collapse, row rhythm, filter,
    display names, memory words.

    Mutation bar: drive the real JS helpers under node (same pattern as T03
    and T06). Do not greps for CSS literals as proof of layout; where only
    the design contract tokens can be checked, assert definition + shared
    use of the token by seat, nav, and room rows together.
    """

    def _html(self) -> str:
        return (ROOM / "index.html").read_text()

    def _extract_js_function(self, html: str, name: str) -> str:
        marker = f"function {name}"
        idx = html.find(marker)
        self.assertGreater(idx, -1, f"{name} not found in index.html")
        brace = html.find("{", idx)
        depth = 0
        i = brace
        while i < len(html):
            if html[i] == "{":
                depth += 1
            elif html[i] == "}":
                depth -= 1
                if depth == 0:
                    return html[idx : i + 1]
            i += 1
        self.fail(f"unbalanced braces extracting {name}")

    def _extract_const_object(self, html: str, name: str) -> str:
        # const NAME = { ... };  (single-level braces only — maps are flat)
        m = re.search(rf"const {name}\s*=\s*\{{", html)
        self.assertIsNotNone(m, f"const {name} missing")
        start = m.start()
        brace = html.find("{", m.start())
        depth = 0
        i = brace
        while i < len(html):
            if html[i] == "{":
                depth += 1
            elif html[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    if end < len(html) and html[end] == ";":
                        end += 1
                    return html[start:end]
            i += 1
        self.fail(f"unbalanced braces extracting const {name}")

    def _run_node(self, script: str) -> str:
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            self.fail(
                f"node failed ({proc.returncode}):\n"
                f"stdout={proc.stdout}\nstderr={proc.stderr}"
            )
        return proc.stdout

    def _minimal_dom_harness(self) -> str:
        """Minimal DOM stubs so seatRow / applyCtx / filter can run under node."""
        return r"""
const byId = {};
function makeEl(tag) {
  const el = {
    tag, _id: '', className: '', textContent: '', title: '',
    type: '', onclick: null, dataset: {}, style: {},
    children: [], attrs: {},
    get id() { return this._id; },
    set id(v) {
      if (this._id && byId[this._id] === this) delete byId[this._id];
      this._id = v || '';
      if (this._id) byId[this._id] = this;
    },
    appendChild(c) { this.children.push(c); c.parent = this; return c; },
    querySelector(sel) {
      const walk = (n) => {
        if (!n) return null;
        if (sel.startsWith('.') && (n.className || '').split(/\s+/).includes(sel.slice(1))) return n;
        if (sel.startsWith('#') && n.id === sel.slice(1)) return n;
        for (const c of (n.children || [])) {
          const hit = walk(c);
          if (hit) return hit;
        }
        return null;
      };
      return walk(this);
    },
    querySelectorAll(sel) {
      const out = [];
      const walk = (n) => {
        if (!n) return;
        if (sel === 'i' && n.tag === 'i') out.push(n);
        if (sel.startsWith('.') && (n.className || '').split(/\s+/).includes(sel.slice(1))) out.push(n);
        if (sel.startsWith('[') && sel.endsWith(']')) {
          const key = sel.slice(1, -1);
          if (key.startsWith('data-') && n.dataset && key.slice(5) in n.dataset) out.push(n);
        }
        for (const c of (n.children || [])) walk(c);
      };
      // For querySelectorAll('i') on a gauge, only search descendants + self.
      if (sel === 'i') {
        const blocks = [];
        const w = (n) => {
          if (!n) return;
          if (n !== this && n.tag === 'i') blocks.push(n);
          // include self if self is not the root? for gauge.querySelectorAll('i') we want children
          for (const c of (n.children || [])) {
            if (c.tag === 'i') blocks.push(c);
            else w(c);
          }
        };
        for (const c of (this.children || [])) {
          if (c.tag === 'i') blocks.push(c);
          else w(c);
        }
        return blocks;
      }
      walk(this);
      return out;
    },
    classList: {
      _owner: null,
      _sync(owner) { this._owner = owner; },
      contains(c) { return (this._owner.className || '').split(/\s+/).includes(c); },
      add(c) {
        const parts = (this._owner.className || '').split(/\s+/).filter(Boolean);
        if (!parts.includes(c)) parts.push(c);
        this._owner.className = parts.join(' ');
      },
      remove(c) {
        this._owner.className = (this._owner.className || '').split(/\s+/).filter(x => x && x !== c).join(' ');
      },
      toggle(c, force) {
        const has = this.contains(c);
        if (force === true || (!has && force !== false)) { this.add(c); return true; }
        this.remove(c); return false;
      },
    },
    setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'data-tip') this.dataset.tip = String(v); },
    getAttribute(k) { return this.attrs[k]; },
    addEventListener(type, fn) {
      this._listeners = this._listeners || {};
      (this._listeners[type] = this._listeners[type] || []).push(fn);
    },
    click() {
      for (const fn of (this._listeners && this._listeners.click) || []) fn({ stopPropagation() {} });
      if (typeof this.onclick === 'function') this.onclick({ stopPropagation() {} });
    },
    get innerHTML() {
      return this.children.map(c => c.outerHTML || c.textContent || '').join('');
    },
    set innerHTML(v) {
      this.children = [];
      if (v) this._rawHTML = v;
      else this._rawHTML = '';
    },
  };
  el.classList._sync(el);
  return el;
}
const document = {
  createElement(tag) { return makeEl(tag); },
  getElementById(id) { return byId[id] || null; },
};
// avatarImg stub used by seatRow
function avatarImg(name, cls) {
  const img = makeEl('img');
  img.className = cls || 'ava';
  img.alt = name;
  return img;
}
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
"""

    def test_seat_at_rest_shows_only_avatar_name_status(self):
        """Criterion 1: at rest = avatar + name + status only; expand reveals rest.

        Builds a real seatRow under node. A regression that re-attaches model,
        effort, or the old thin bar to the collapsed row fails the rest scan.
        """
        html = self._html()
        seat_row = self._extract_js_function(html, "seatRow")
        # Helpers seatRow closes over.
        helpers = []
        for name in (
            "modelDisplayName",
            "memoryWords",
            "seatMeta",
            "applyCtx",
            "fmtTok",
            "effortLabel",
        ):
            if f"function {name}" in html:
                helpers.append(self._extract_js_function(html, name))
        display_map = ""
        if "const MODEL_DISPLAY_NAMES" in html:
            display_map = self._extract_const_object(html, "MODEL_DISPLAY_NAMES")
        elif "const _DISPLAY_NAMES" in html:
            display_map = self._extract_const_object(html, "_DISPLAY_NAMES")
        # seatRow closes over MEM_TIP and catalog constants.
        mem_tip = "const MEM_TIP = 'working memory tip';"
        m = re.search(r"const MEM_TIP\s*=\s*[^;]+;", html)
        if m:
            mem_tip = m.group(0)
        catalog = (
            "const MODELS_BY_ENGINE = null;\n"
            "const MODELS_FALLBACK = "
            "{claude:[{id:'fable',name:'Fable 5'}],"
            "grok:[{id:'grok-4.5',name:'Grok 4.5'}],"
            "kimi:[{id:'kimi-code/k3',name:'K3'}]};\n"
        )

        script = f"""
{self._minimal_dom_harness()}
{display_map}
{mem_tip}
{catalog}
{chr(10).join(helpers)}
{seat_row}

const seat = {{
  name: 'claude',
  engine: 'claude',
  model: 'claude-fable-5',
  effort: 'medium',
  color: '#d98d75',
  role: 'Planner and chair — breaks goals into steps.',
  ctx: {{ used: 116000, limit: 200000, est: true }},
}};
const root = seatRow(seat);
// Live path calls applyCtx after append; drive the same here.
if (typeof applyCtx === 'function') applyCtx(seat);
// seatRow may return a wrap (seat + details) or the seat itself.
let seatEl = root;
let details = null;
if (root.id === 'seat-claude') {{
  seatEl = root;
}} else {{
  seatEl = root.querySelector('#seat-claude') || root.children.find(c => c.id === 'seat-claude') || root;
  details = root.children.find(c => (c.className || '').includes('details')) || null;
}}
if (!seatEl || seatEl.id !== 'seat-claude') {{
  console.error('FAIL no seat element with id seat-claude');
  process.exit(1);
}}
// Collect all text visible in the collapsed seat row (not details panel).
function texts(n, skipDetails) {{
  const out = [];
  const walk = (el) => {{
    if (!el) return;
    if (skipDetails && (el.className || '').includes('details')) return;
    if (el.textContent) out.push(el.textContent);
    for (const c of (el.children || [])) walk(c);
  }};
  walk(n);
  return out.join(' | ');
}}
const restText = texts(seatEl, true);
// At rest: name present.
if (!restText.includes('claude')) {{
  console.error('FAIL name missing at rest', restText);
  process.exit(1);
}}
// At rest: raw model id and effort must not appear on the seat row.
if (restText.includes('claude-fable-5') || restText.includes('fable')) {{
  console.error('FAIL model leaked into at-rest row', restText);
  process.exit(1);
}}
if (/medium/i.test(restText) && !/signed out/i.test(restText)) {{
  // effort word on the collapsed row
  console.error('FAIL effort leaked into at-rest row', restText);
  process.exit(1);
}}
// Old thin bar / percentage must not exist on the seat row.
const hasPctClass = !!(seatEl.querySelector && seatEl.querySelector('.ctx-pct'));
const hasCtxBar = !!(seatEl.querySelector && seatEl.querySelector('.ctxrow'));
if (hasPctClass || hasCtxBar) {{
  console.error('FAIL old percentage bar still on seat row');
  process.exit(1);
}}
if (/%/.test(restText)) {{
  console.error('FAIL percentage in at-rest text', restText);
  process.exit(1);
}}
// Expanding must reveal role + friendly model + effort + memory label.
if (typeof seatEl.click === 'function') seatEl.click();
// If details is a sibling, the wrap should open it.
const wrap = root;
const det = details || wrap.children.find(c => (c.className || '').includes('details'))
  || (byId['details-claude'] || null);
if (!det) {{
  console.error('FAIL no details panel after build');
  process.exit(1);
}}
// Open if click toggle did not (some builds open via class on wrap).
det.classList.add('open');
const openText = texts(det, false);
if (!openText.includes('Planner') && !openText.includes('chair') && !openText.includes('breaks goals')) {{
  console.error('FAIL role missing from details', openText);
  process.exit(1);
}}
if (!openText.includes('Fable 5')) {{
  console.error('FAIL friendly model name missing from details', openText);
  process.exit(1);
}}
if (openText.includes('claude-fable-5')) {{
  console.error('FAIL raw model id in details', openText);
  process.exit(1);
}}
if (!/Medium effort/i.test(openText) && !/medium/i.test(openText)) {{
  console.error('FAIL effort missing from details', openText);
  process.exit(1);
}}
// Memory words (not a bare percent) in the open panel.
if (/%/.test(openText)) {{
  console.error('FAIL percentage in details', openText);
  process.exit(1);
}}
const memOk = /Half used|Getting full|Plenty of room|Fresh|Nearly full/i.test(openText);
if (!memOk) {{
  console.error('FAIL memory label missing from details', openText);
  process.exit(1);
}}
console.log('seat-rest-ok');
"""
        out = self._run_node(script)
        self.assertIn("seat-rest-ok", out)

    def test_memory_words_four_blocks_and_thresholds(self):
        """Criterion 5: four blocks + plain-English labels; thresholds hold.

        Exercises the real memoryWords helper across every band. A mutation
        that returns a percentage string, wrong block count, or conflates
        bands fails. Not a source grep.
        """
        html = self._html()
        fn = self._extract_js_function(html, "memoryWords")
        script = f"""
{fn}
const cases = [
  [0,  0, 'Fresh'],
  [10, 0, 'Fresh'],
  [14, 0, 'Fresh'],
  [15, 1, 'Plenty'],
  [39, 1, 'Plenty'],
  [40, 2, 'Half'],
  [64, 2, 'Half'],
  [65, 3, 'Getting'],
  [84, 3, 'Getting'],
  [85, 4, 'Nearly'],
  [100,4, 'Nearly'],
];
for (const [pct, blocks, needle] of cases) {{
  const m = memoryWords(pct);
  if (typeof m.label !== 'string' || !m.label) {{
    console.error('FAIL empty label at', pct); process.exit(1);
  }}
  if (m.label.includes('%')) {{
    console.error('FAIL percent in label', m.label); process.exit(1);
  }}
  if (m.blocks !== blocks) {{
    console.error('FAIL blocks at', pct, 'got', m.blocks, 'want', blocks); process.exit(1);
  }}
  if (!m.label.includes(needle) && !new RegExp(needle, 'i').test(m.label)) {{
    console.error('FAIL label at', pct, m.label, 'missing', needle); process.exit(1);
  }}
  if (blocks > 4 || blocks < 0) {{
    console.error('FAIL block count out of range', m.blocks); process.exit(1);
  }}
}}
// Tone ramp: warn from 65, hot from 85.
if (memoryWords(64).tone === 'warn' || memoryWords(64).tone === 'hot') {{
  console.error('FAIL 64 should not warn'); process.exit(1);
}}
if (memoryWords(65).tone !== 'warn') {{
  console.error('FAIL 65 should warn', memoryWords(65)); process.exit(1);
}}
if (memoryWords(85).tone !== 'hot') {{
  console.error('FAIL 85 should be hot', memoryWords(85)); process.exit(1);
}}
console.log('memory-ok');
"""
        out = self._run_node(script)
        self.assertIn("memory-ok", out)

    def test_sidebar_model_display_names_match_python(self):
        """Criterion 4: no raw id on screen; Fable and Opus are not conflated.

        HTML has its own fallback map (server unreachable). It must agree with
        engines.model_display_name for the known ids — especially fable ≠ opus.
        """
        html = self._html()
        from engines import model_display_name as py_name

        fn = self._extract_js_function(html, "modelDisplayName")
        display_map = ""
        if "const MODEL_DISPLAY_NAMES" in html:
            display_map = self._extract_const_object(html, "MODEL_DISPLAY_NAMES")
        elif "const _DISPLAY_NAMES" in html:
            display_map = self._extract_const_object(html, "_DISPLAY_NAMES")
        else:
            self.fail("HTML model display-name map const missing")

        ids = [
            "fable", "opus", "sonnet",
            "claude-fable-5", "claude-opus-5", "claude-sonnet-5",
            "grok-4.5",
            "kimi-code/k3",
        ]
        # Build expected from Python (single source of truth for the contract).
        expected = {i: py_name(i) for i in ids}
        script = f"""
{display_map}
{fn}
const expected = {json.dumps(expected)};
for (const [id, want] of Object.entries(expected)) {{
  const got = modelDisplayName(id);
  if (got !== want) {{
    console.error('FAIL', id, 'got', got, 'want', want);
    process.exit(1);
  }}
  // Raw id must never be the display string for known mapped ids.
  if (got === id) {{
    console.error('FAIL raw id returned for', id);
    process.exit(1);
  }}
}}
if (modelDisplayName('claude-fable-5') === modelDisplayName('claude-opus-5')) {{
  console.error('FAIL fable conflated with opus');
  process.exit(1);
}}
if (modelDisplayName('claude-fable-5') === 'Opus 5') {{
  console.error('FAIL mock defect reintroduced: fable → Opus 5');
  process.exit(1);
}}
console.log('display-ok');
"""
        out = self._run_node(script)
        self.assertIn("display-ok", out)

    def test_filter_narrows_agents_and_chats_together(self):
        """Criterion 3: one filter input narrows seats and past chats together.

        Runs the real applySidebarFilter (or equivalent) under node against a
        fixture list. Filtering to a seat name hides unrelated seats AND
        unrelated rooms; clearing restores both.
        """
        html = self._html()
        # Prefer a dedicated helper; fall back to whatever the page names it.
        fn_name = None
        for candidate in ("applySidebarFilter", "applyFilter", "filterSidebar"):
            if f"function {candidate}" in html:
                fn_name = candidate
                break
        self.assertIsNotNone(
            fn_name,
            "sidebar filter function missing (applySidebarFilter/applyFilter)",
        )
        fn = self._extract_js_function(html, fn_name)
        script = f"""
// Fixture DOM: two seats, two rooms, one filter input.
function makeNode(tag, attrs={{}}) {{
  const n = {{
    tag, className: attrs.className || '', id: attrs.id || '',
    dataset: Object.assign({{}}, attrs.dataset || {{}}),
    children: [], style: {{}},
    classList: {{
      _c: new Set((attrs.className || '').split(/\\s+/).filter(Boolean)),
      contains(c) {{ return this._c.has(c); }},
      add(c) {{ this._c.add(c); this._owner.className = [...this._c].join(' '); }},
      remove(c) {{ this._c.delete(c); this._owner.className = [...this._c].join(' '); }},
      toggle(c, force) {{
        if (force === true) this.add(c);
        else if (force === false) this.remove(c);
        else if (this.contains(c)) this.remove(c); else this.add(c);
      }},
    }},
  }};
  n.classList._owner = n;
  return n;
}}
const seatsHost = makeNode('div', {{id: 'seats'}});
const roomsHost = makeNode('div', {{id: 'roomsList'}});
const seatA = makeNode('div', {{dataset: {{filter: 'claude planner chair'}}}});
seatA.id = 'wrap-claude';
const seatB = makeNode('div', {{dataset: {{filter: 'grok builder'}}}});
seatB.id = 'wrap-grok';
seatsHost.children.push(seatA, seatB);
const roomA = makeNode('button', {{dataset: {{filter: 'sidebar redesign'}}}});
const roomB = makeNode('button', {{dataset: {{filter: 'paint consumption'}}}});
roomsHost.children.push(roomA, roomB);
const seatsEmpty = makeNode('div', {{id: 'seatsEmpty', className: 'list-empty'}});
const roomsEmpty = makeNode('div', {{id: 'roomsEmpty', className: 'list-empty'}});
const filterBox = makeNode('div', {{id: 'filter'}});
const filterInput = {{ value: '', id: 'filterInput' }};
const byId = {{
  seats: seatsHost, roomsList: roomsHost,
  seatsEmpty, roomsEmpty, filter: filterBox, filterInput,
  // legacy id some builds keep as alias
  roomsFilter: filterInput,
}};
// querySelectorAll on hosts
seatsHost.querySelectorAll = (sel) => {{
  if (sel.includes('data-filter') || sel === '[data-filter]') return seatsHost.children.slice();
  return [];
}};
roomsHost.querySelectorAll = (sel) => {{
  if (sel.includes('data-filter') || sel === '[data-filter]') return roomsHost.children.slice();
  return [];
}};
const document = {{
  getElementById(id) {{ return byId[id] || null; }},
}};
{fn}
function run(q) {{
  filterInput.value = q;
  {fn_name}();
}}
function hidden(n) {{
  return n.classList.contains('hidden-by-filter') || n.style.display === 'none';
}}
run('claude');
if (hidden(seatA)) {{ console.error('FAIL claude seat hidden'); process.exit(1); }}
if (!hidden(seatB)) {{ console.error('FAIL grok seat still visible'); process.exit(1); }}
// Room filter: 'claude' matches neither room title → both rooms hidden.
if (!hidden(roomA) || !hidden(roomB)) {{
  // Some implementations only filter rooms when query matches room text;
  // both should hide when query is a seat-only term.
  console.error('FAIL rooms not filtered with seat-only query',
    hidden(roomA), hidden(roomB));
  process.exit(1);
}}
run('redesign');
if (!hidden(seatA) || !hidden(seatB)) {{
  console.error('FAIL seats not filtered by room-only query'); process.exit(1);
}}
if (hidden(roomA)) {{ console.error('FAIL redesign room hidden'); process.exit(1); }}
if (!hidden(roomB)) {{ console.error('FAIL paint room still visible'); process.exit(1); }}
run('');
if (hidden(seatA) || hidden(seatB) || hidden(roomA) || hidden(roomB)) {{
  console.error('FAIL clear did not restore'); process.exit(1);
}}
console.log('filter-ok');
"""
        out = self._run_node(script)
        self.assertIn("filter-ok", out)

    def test_apply_ctx_uses_blocks_not_percentage(self):
        """Criterion 5 (render path): applyCtx fills four blocks + label, no %."""
        html = self._html()
        apply = self._extract_js_function(html, "applyCtx")
        mem = self._extract_js_function(html, "memoryWords")
        # fmtTok may still exist for tooltips elsewhere; pull if present.
        extras = []
        if "function fmtTok" in html:
            extras.append(self._extract_js_function(html, "fmtTok"))
        script = f"""
const byId = {{}};
function make(id, cls) {{
  const el = {{
    id, className: cls || '', textContent: '', title: '',
    children: [], style: {{}}, dataset: {{}},
    setAttribute(k, v) {{ this.dataset[k.replace(/^data-/, '')] = String(v); }},
    classList: {{
      _c: new Set((cls || '').split(/\\s+/).filter(Boolean)),
      add(c) {{ this._c.add(c); el.className = [...this._c].join(' '); }},
      remove(...cs) {{ for (const c of cs) this._c.delete(c); el.className = [...this._c].join(' '); }},
      contains(c) {{ return this._c.has(c); }},
      toggle(c, force) {{
        if (force === false) this.remove(c);
        else if (force === true || !this.contains(c)) this.add(c);
        else this.remove(c);
      }},
    }},
    querySelectorAll(sel) {{
      if (sel === 'i') return this.children.filter(c => c.tag === 'i');
      return [];
    }},
  }};
  byId[id] = el;
  return el;
}}
const gauge = make('mem-claude', 'mem-gauge');
// four block children
for (let i = 0; i < 4; i++) {{
  gauge.children.push({{ tag: 'i', className: '', classList: {{
    _c: new Set(),
    add(c) {{ this._c.add(c); }},
    remove(c) {{ this._c.delete(c); }},
    contains(c) {{ return this._c.has(c); }},
    toggle(c, on) {{ if (on) this.add(c); else this.remove(c); }},
  }}}});
}}
const label = make('memlabel-claude', 'mem-label');
// Old percentage element must NOT be required; if present it must stay empty / unused.
const document = {{ getElementById(id) {{ return byId[id] || null; }} }};
const MEM_TIP = 'working memory tip';
{chr(10).join(extras)}
{mem}
{apply}
applyCtx({{
  name: 'claude',
  ctx: {{ used: 130000, limit: 200000, est: true }},  // 65%
}});
// Label is words.
if (!label.textContent || label.textContent.includes('%')) {{
  console.error('FAIL label', label.textContent);
  process.exit(1);
}}
if (!/Getting full/i.test(label.textContent)) {{
  console.error('FAIL expected Getting full, got', label.textContent);
  process.exit(1);
}}
// Three blocks on at 65%.
const onCount = gauge.children.filter(c => c.classList.contains('on')).length;
if (onCount !== 3) {{
  console.error('FAIL onCount', onCount);
  process.exit(1);
}}
// Gauge carries warn tone.
if (!gauge.classList.contains('warn') && !(gauge.className || '').includes('warn')) {{
  console.error('FAIL warn tone missing', gauge.className);
  process.exit(1);
}}
// No thin-bar width paint.
if (gauge.style && gauge.style.width && String(gauge.style.width).includes('%')) {{
  console.error('FAIL percentage width on gauge');
  process.exit(1);
}}
console.log('apply-ctx-ok');
"""
        out = self._run_node(script)
        self.assertIn("apply-ctx-ok", out)

    def test_seat_row_real_path_four_blocks_no_percent(self):
        """Criterion 5: real seatRow path builds four blocks + word labels, never %.

        Drives the production seatRow + applyCtx path under node with real seat
        data. A fixture that hand-builds four gauge children does not count —
        this asserts the count seatRow itself produces, and that the label it
        wires into holds plain English (not a percentage) after applyCtx.
        """
        html = self._html()
        seat_row = self._extract_js_function(html, "seatRow")
        helpers = []
        for name in (
            "modelDisplayName",
            "memoryWords",
            "applyCtx",
            "effortLabel",
        ):
            if f"function {name}" in html:
                helpers.append(self._extract_js_function(html, name))
        display_map = ""
        if "const MODEL_DISPLAY_NAMES" in html:
            display_map = self._extract_const_object(html, "MODEL_DISPLAY_NAMES")
        elif "const _DISPLAY_NAMES" in html:
            display_map = self._extract_const_object(html, "_DISPLAY_NAMES")
        mem_tip = "const MEM_TIP = 'working memory tip';"
        m = re.search(r"const MEM_TIP\s*=\s*[^;]+;", html)
        if m:
            mem_tip = m.group(0)
        catalog = (
            "const MODELS_BY_ENGINE = null;\n"
            "const MODELS_FALLBACK = "
            "{claude:[{id:'fable',name:'Fable 5'}],"
            "grok:[{id:'grok-4.5',name:'Grok 4.5'}],"
            "kimi:[{id:'kimi-code/k3',name:'K3'}]};\n"
        )
        script = f"""
{self._minimal_dom_harness()}
{display_map}
{mem_tip}
{catalog}
{chr(10).join(helpers)}
{seat_row}

const seats = [
  {{
    name: 'claude', engine: 'claude', model: 'claude-fable-5',
    effort: 'medium', color: '#d98d75',
    role: 'Your role: PLANNER and chair.',
    // 65% → Getting full, 3 blocks on
    ctx: {{ used: 130000, limit: 200000, est: true }},
  }},
  {{
    name: 'grok', engine: 'grok', model: null, effort: 'high',
    color: '#9aa0a8', role: 'Your role: BUILDER.',
    // 7% → Fresh, 0 blocks on
    ctx: {{ used: 14000, limit: 200000, est: true }},
  }},
  {{
    name: 'kimi', engine: 'kimi', model: 'kimi-code/k3', effort: 'medium',
    color: '#7a8b9a', role: 'Your role: REVIEWER.',
    // 90% → Nearly full, 4 blocks on
    ctx: {{ used: 180000, limit: 200000, est: true }},
  }},
];

function collectText(n) {{
  const out = [];
  const walk = (el) => {{
    if (!el) return;
    if (el.textContent) out.push(String(el.textContent));
    for (const c of (el.children || [])) walk(c);
  }};
  walk(n);
  return out.join(' | ');
}}

const wordsOk = /Fresh|Plenty of room|Half used|Getting full|Nearly full/i;
for (const seat of seats) {{
  const root = seatRow(seat);
  // Production loadRoster appends then applyCtx — same order.
  applyCtx(seat);
  const gauge = byId['mem-' + seat.name];
  const label = byId['memlabel-' + seat.name];
  if (!gauge || !label) {{
    console.error('FAIL missing mem nodes for', seat.name);
    process.exit(1);
  }}
  // Real row builder must produce exactly four block children.
  const blocks = (gauge.children || []).filter(c => c.tag === 'i');
  // Also try querySelectorAll if harness wired it.
  const qBlocks = typeof gauge.querySelectorAll === 'function'
    ? gauge.querySelectorAll('i') : blocks;
  const blockCount = Math.max(blocks.length, (qBlocks && qBlocks.length) || 0);
  if (blockCount !== 4) {{
    console.error('FAIL block count for', seat.name, 'got', blockCount, 'want 4');
    process.exit(1);
  }}
  if (!label.textContent || label.textContent.includes('%')) {{
    console.error('FAIL percent (or empty) on real path label', seat.name, label.textContent);
    process.exit(1);
  }}
  if (!wordsOk.test(label.textContent)) {{
    console.error('FAIL label not plain English', seat.name, label.textContent);
    process.exit(1);
  }}
  // Tip must be set on the real label element.
  const tip = label.attrs && label.attrs['data-tip']
    || label.dataset && label.dataset.tip
    || label.getAttribute && label.getAttribute('data-tip');
  if (!tip || !/working memory|conversation|fresh room/i.test(String(tip))) {{
    console.error('FAIL memory tip missing on label', seat.name, tip);
    process.exit(1);
  }}
  // Whole seat markup must not carry a bare percentage readout.
  const markup = collectText(root);
  if (/%/.test(markup)) {{
    console.error('FAIL percentage in real seat markup', seat.name, markup);
    process.exit(1);
  }}
}}
// Known band check on the 65% seat.
const claudeLabel = byId['memlabel-claude'];
if (!/Getting full/i.test(claudeLabel.textContent)) {{
  console.error('FAIL 65% seat label', claudeLabel.textContent);
  process.exit(1);
}}
const claudeGauge = byId['mem-claude'];
const onCount = (claudeGauge.children || []).filter(
  c => c.tag === 'i' && c.classList && c.classList.contains('on')
).length;
if (onCount !== 3) {{
  console.error('FAIL onCount at 65%', onCount);
  process.exit(1);
}}
console.log('seat-row-memory-ok');
"""
        out = self._run_node(script)
        self.assertIn("seat-row-memory-ok", out)

    def test_row_rhythm_shared_token(self):
        """Criterion 2: one row height token used by seats, nav, and room rows.

        Computed style needs a browser; without one we assert the design
        contract the only way unit tests can: --row-h:44px and --row-gap:2px
        are defined once, and .seat / .nav-item|/.row / .room-row all consume
        var(--row-h) (not a hard-coded 42px/32px). A mutation that reverts
        seats to 42px while leaving the token defined fails.
        """
        html = self._html()
        # Token definitions
        self.assertRegex(
            html,
            r"--row-h\s*:\s*44px",
            "design token --row-h:44px missing",
        )
        self.assertRegex(
            html,
            r"--row-gap\s*:\s*2px",
            "design token --row-gap:2px missing",
        )
        # Seat consumes the token for height (not a literal 42px).
        seat_block = re.search(
            r"\.seat\s*\{[^}]+\}",
            html,
        )
        self.assertIsNotNone(seat_block, ".seat rule missing")
        self.assertIn(
            "var(--row-h)",
            seat_block.group(0),
            ".seat must use var(--row-h)",
        )
        self.assertNotRegex(
            seat_block.group(0),
            r"min-height\s*:\s*42px",
            ".seat must not hard-code the old 42px height",
        )
        # Nav items (class may be .nav-item or .row per mock).
        nav_ok = bool(
            re.search(
                r"\.(?:nav-item|row)\s*\{[^}]*var\(--row-h\)",
                html,
            )
        )
        self.assertTrue(nav_ok, "nav row class must use var(--row-h)")
        room_ok = bool(
            re.search(
                r"\.room-row\s*\{[^}]*var\(--row-h\)",
                html,
            )
        )
        self.assertTrue(room_ok, ".room-row must use var(--row-h)")
        # Shared gap on the nav column / sub-list.
        self.assertRegex(
            html,
            r"gap\s*:\s*var\(--row-gap\)",
            "row gap must use var(--row-gap)",
        )

    def test_no_sidebar_percentage_or_thin_bar_css(self):
        """Criterion 5: percentage readout and 3px thin bar stay out of sidebar.

        Asserts the old .ctx-pct / 3px .ctx bar rules are gone (not merely
        unused), so a future edit cannot light them back up by id.
        """
        html = self._html()
        # Old percentage class rule must not remain as a live style.
        self.assertIsNone(
            re.search(r"\.seat\s+\.ctx-pct\s*\{", html),
            "old .ctx-pct rule still in CSS",
        )
        self.assertIsNone(
            re.search(r"\.seat\s+\.ctx\s*\{[^}]*height\s*:\s*3px", html),
            "old 3px thin context bar still in CSS",
        )
        # Sidebar markup must not include a percentage meter host.
        self.assertNotIn('class="ctx-pct"', html)
        # Any remaining "ctx-pct" string in the file (comment, dead id) is a
        # regression — the class must be fully gone, not merely unused.
        self.assertNotIn("ctx-pct", html)


class TestBoardroom(unittest.TestCase):
    """DM-12 / T08: boardroom mode — topic chat, all seats, chair summary, idle.

    Mutation bar: drive the real POST handlers; assert exact topic/summary
    values (not mere existence); prove mission stays None; prove idle does
    not re-invoke engines or close.
    """

    TOPIC = "Should we adopt widget X for the Agent Room board?"
    CHAIR_SUMMARY = (
        "Agreement: widget X is ready to trial.\n"
        "Disagreement: roll-out speed — now vs after T09.\n"
        "Recommendation: trial on one ticket first.\n"
        "Anything to dig into before I close?"
    )

    @classmethod
    def setUpClass(cls):
        import importlib.util
        name = "agent_room_server_t08_boardroom"
        if name in sys.modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, ROOM / "server.py")
        cls.server = importlib.util.module_from_spec(spec)
        with mock.patch("http.server.ThreadingHTTPServer"):
            spec.loader.exec_module(cls.server)

    def setUp(self):
        self.server.CTX.clear()
        if hasattr(self.server, "BOARDROOM_OUTCOME"):
            self.server.BOARDROOM_OUTCOME = None
        self.all_seats = list(self.server.AGENTS)
        self.assertIn("claude", self.all_seats, "chair seat 'claude' required")
        self.assertGreaterEqual(len(self.all_seats), 2)

    def _seed_prior_chat(self, rooms):
        """Live current/ with one prior user line so filing has something to keep."""
        current = rooms / "current"
        current.mkdir()
        state = {
            "claude_session": None,
            "kimi_started": False,
            "last_seen": {},
            "mission": None,
            "stop": False,
            "next_id": 2,
            "sessions": {},
        }
        (current / "state.json").write_text(json.dumps(state, indent=2))
        msg = {
            "id": 1, "ts": "prior", "from": "you",
            "text": "prior chat before boardroom", "hops": 0, "kind": "chat",
        }
        (current / "messages.jsonl").write_text(json.dumps(msg) + "\n")
        return current

    def _handler_post(self, path, payload: dict):
        from io import BytesIO
        body = json.dumps(payload).encode()
        h = self.server.Handler.__new__(self.server.Handler)
        h.path = path
        h.headers = {"Content-Length": str(len(body))}
        h.rfile = BytesIO(body)
        h.wfile = BytesIO()
        h.request_version = "HTTP/1.1"
        h.command = "POST"
        h.client_address = ("127.0.0.1", 9)
        h.close_connection = False
        h.log_message = lambda *a, **k: None
        status = {}
        response_body = {}

        def capture_json(obj, code=200):
            status["code"] = code
            response_body["json"] = obj
            raw = json.dumps(obj).encode()
            h.wfile.write(raw)

        h.send_response = lambda code, message=None: status.__setitem__("code", code)
        h.send_header = lambda *a, **k: None
        h.end_headers = lambda: None
        h._json = lambda obj, code=200: capture_json(obj, code)
        h.do_POST()
        return status.get("code"), response_body.get("json")

    def _handler_get(self, path):
        from io import BytesIO
        h = self.server.Handler.__new__(self.server.Handler)
        h.path = path
        h.headers = {}
        h.rfile = BytesIO(b"")
        h.wfile = BytesIO()
        h.request_version = "HTTP/1.1"
        h.command = "GET"
        h.client_address = ("127.0.0.1", 9)
        h.close_connection = False
        h.log_message = lambda *a, **k: None
        status = {}
        response_body = {}

        def capture_json(obj, code=200):
            status["code"] = code
            response_body["json"] = obj

        h.send_response = lambda code, message=None: status.__setitem__("code", code)
        h.send_header = lambda *a, **k: None
        h.end_headers = lambda: None
        h._json = lambda obj, code=200: capture_json(obj, code)
        h.do_GET()
        return status.get("code"), response_body.get("json")

    def test_post_boardroom_opens_named_chat_and_files_previous(self):
        """Criterion 1: POST /api/boardroom files prior chat, new chat titled topic."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            current = self._seed_prior_chat(rooms)
            prior_text = (current / "messages.jsonl").read_text()

            dispatched = []

            def capture_dispatch(agent, hops):
                dispatched.append(agent)

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch", side_effect=capture_dispatch):
                code, body = self._handler_post(
                    "/api/boardroom", {"topic": self.TOPIC})

            self.assertEqual(code, 200, f"boardroom POST failed: {body}")
            self.assertTrue(body.get("ok"), body)

            # Prior chat must be filed alongside other archives (not discarded).
            archives = [
                d for d in rooms.iterdir()
                if d.is_dir() and d.name.startswith("room-")
            ]
            self.assertTrue(
                archives,
                "starting a boardroom must file the previous live chat",
            )
            filed_msgs = (archives[0] / "messages.jsonl").read_text()
            self.assertIn("prior chat before boardroom", filed_msgs)
            self.assertEqual(filed_msgs, prior_text)

            # New live chat is named after the exact topic.
            live = json.loads((rooms / "current" / "state.json").read_text())
            self.assertEqual(
                live.get("title"), self.TOPIC,
                "live state.title must be the topic string (not a timestamp)",
            )
            br = live.get("boardroom")
            self.assertIsInstance(br, dict, "boardroom state must be a dict")
            self.assertEqual(br.get("topic"), self.TOPIC)
            self.assertEqual(br.get("status"), "open")
            self.assertIsNone(live.get("mission"))

            msgs = [
                json.loads(line)
                for line in (rooms / "current" / "messages.jsonl")
                .read_text().splitlines()
                if line.strip()
            ]
            your_msgs = [m for m in msgs if m.get("from") == "you"]
            self.assertTrue(your_msgs, "boardroom must post the topic as you")
            topic_msg = your_msgs[0]["text"]
            self.assertIn(self.TOPIC, topic_msg)
            self.assertRegex(topic_msg, r"(?i)@all\b")

            # Sidebar listing: title drives the preview when filed.
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                stamp = "2026-08-03-br-title"
                archived = self.server.file_current_room(stamp=stamp)
                listed = self.server.list_archives()
            match = next(
                (r for r in listed if r["id"] == archived.name), None)
            self.assertIsNotNone(match, f"filed boardroom missing from list: {listed}")
            self.assertEqual(
                match["preview"], self.TOPIC,
                "filed boardroom must show the topic as its name in the list",
            )

    def test_boardroom_invites_every_seat(self):
        """Criterion 2: every seat is dispatched; no pre-chosen panel."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_prior_chat(rooms)
            dispatched = []

            def capture_dispatch(agent, hops):
                dispatched.append(agent)

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch", side_effect=capture_dispatch):
                code, body = self._handler_post(
                    "/api/boardroom", {"topic": self.TOPIC})

            self.assertEqual(code, 200, body)
            self.assertEqual(
                sorted(dispatched),
                sorted(self.all_seats),
                f"boardroom must invite every seat once, got {dispatched}",
            )

            # seats= filter must be ignored (v1: silence protocol only).
            with tempfile.TemporaryDirectory() as td2:
                rooms2 = Path(td2)
                self._seed_prior_chat(rooms2)
                d2 = []

                def cap2(agent, hops):
                    d2.append(agent)

                with mock.patch.object(self.server, "ROOMS_DIR", rooms2), \
                     mock.patch.object(self.server, "CURRENT", rooms2 / "current"), \
                     mock.patch.object(self.server, "dispatch", side_effect=cap2):
                    code2, body2 = self._handler_post(
                        "/api/boardroom",
                        {"topic": self.TOPIC, "seats": ["claude"]},
                    )
                self.assertEqual(code2, 200, body2)
                self.assertEqual(
                    sorted(d2),
                    sorted(self.all_seats),
                    "seats filter must be ignored; silence protocol is the filter",
                )

    def test_chair_prompt_names_agreement_disagreement_recommendation(self):
        """Criterion 3: chair prompt requires the three-part summary + ask the user."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_prior_chat(rooms)
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch"):
                self._handler_post("/api/boardroom", {"topic": self.TOPIC})

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                feed = self.server.all_messages()
                prompt = self.server.build_prompt("claude", feed)

            lower = prompt.lower()
            self.assertIn("agreement", lower, prompt)
            self.assertIn("disagreement", lower, prompt)
            self.assertIn("recommendation", lower, prompt)
            self.assertTrue(
                "before" in lower and "close" in lower,
                f"chair must ask the user before closing: {prompt}",
            )
            self.assertIn(self.TOPIC, prompt)

            # Chair posting a summary must NOT close the boardroom.
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch"):
                msg = self.server.append_message(
                    "claude", self.CHAIR_SUMMARY, hops=1)
                self.server.route(msg)
                live = self.server.load_state()
            self.assertEqual(live["boardroom"]["status"], "open")
            self.assertIsNone(getattr(self.server, "BOARDROOM_OUTCOME", None))

    def test_idle_boardroom_no_engine_and_stays_open(self):
        """Criterion 4: idle costs nothing — no re-dispatch, no auto-close.

        Drives process_turn (the real worker body), not a hand-copied loop.
        """
        import tempfile

        engine_calls = []

        def fake_run_engine(agent, prompt, room_epoch=None):
            engine_calls.append(agent)
            return "[silent]"

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_prior_chat(rooms)

            for a in self.server.AGENTS:
                q = self.server.WORK_QUEUES.get(a)
                if q is None:
                    continue
                while not q.empty():
                    try:
                        q.get_nowait()
                    except Exception:
                        break

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(
                     self.server, "run_engine", side_effect=fake_run_engine), \
                 mock.patch.object(
                     self.server, "maybe_update_minutes", return_value=None):
                code, body = self._handler_post(
                    "/api/boardroom", {"topic": self.TOPIC})
                self.assertEqual(code, 200, body)

                for agent in list(self.all_seats):
                    q = self.server.WORK_QUEUES[agent]
                    while not q.empty():
                        hops = q.get_nowait()
                        self.server.process_turn(agent, hops)

                after_invite = list(engine_calls)
                self.assertEqual(
                    sorted(after_invite),
                    sorted(self.all_seats),
                    "each seat should have been invoked once on invite",
                )

                before = len(engine_calls)
                # Idle: no new messages, no timer wakes — queues stay empty.
                for agent in list(self.all_seats):
                    self.assertTrue(self.server.WORK_QUEUES[agent].empty())
                self.assertEqual(
                    len(engine_calls), before,
                    "idle boardroom must not re-invoke engines",
                )

                live = self.server.load_state()
                self.assertEqual(live["boardroom"]["status"], "open")
                self.assertIsNone(getattr(self.server, "BOARDROOM_OUTCOME", None))

    def test_close_emits_outcome_with_chair_summary(self):
        """Criterion 5: the user close emits outcome carrying the chair summary."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_prior_chat(rooms)
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch"):
                self._handler_post("/api/boardroom", {"topic": self.TOPIC})
                self.server.append_message(
                    "claude", self.CHAIR_SUMMARY, hops=1)

                code, body = self._handler_post("/api/boardroom/close", {})
            self.assertEqual(code, 200, body)
            self.assertTrue(body.get("ok"), body)

            self.assertTrue(
                hasattr(self.server, "get_boardroom_outcome"),
                "server must expose get_boardroom_outcome() for external callers",
            )
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                outcome = self.server.get_boardroom_outcome()
            self.assertIsInstance(outcome, dict)
            self.assertEqual(outcome.get("topic"), self.TOPIC)
            self.assertEqual(
                outcome.get("summary"), self.CHAIR_SUMMARY,
                "outcome.summary must be the chair's exact summary text",
            )
            self.assertEqual(outcome.get("status"), "closed")

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                gcode, gbody = self._handler_get("/api/boardroom/outcome")
            self.assertEqual(gcode, 200, gbody)
            self.assertEqual(gbody.get("summary"), self.CHAIR_SUMMARY)
            self.assertEqual(gbody.get("topic"), self.TOPIC)
            self.assertEqual(gbody.get("status"), "closed")

            live = json.loads((rooms / "current" / "state.json").read_text())
            self.assertEqual(live["boardroom"]["status"], "closed")

    def test_close_works_when_chair_never_summarised(self):
        """Budget can pause a room before the chair summarises. the user must still
        be able to close it: refusing strands the room with no way out."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td) / "rooms"
            rooms.mkdir()
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch", lambda *a, **k: None):
                self.server.ensure_room()
                self.server.start_boardroom("a topic with no summary")
                st = self.server.load_state()
                st["boardroom"]["turn_paused"] = True
                self.server.save_state(st)
                # No chair message exists at all.
                self.assertFalse(
                    [m for m in self.server.all_messages()
                     if m.get("from") == self.server.CHAIR_SEAT
                     and m.get("kind") != "system"],
                    "precondition: chair must not have spoken",
                )
                res = self.server.close_boardroom()
            self.assertNotIn("error", res, f"close must not strand the room: {res}")
            self.assertTrue(res.get("ok"))
            self.assertIn(
                "before the chair wrote a summary",
                ((res.get("outcome") or {}).get("summary") or ""),
                "the outcome must say plainly that there was no agreed outcome",
            )

    def test_resume_clears_pause_and_restores_budget(self):
        """A paused boardroom is recoverable: the user grants another round."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td) / "rooms"
            rooms.mkdir()
            calls = []
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch",
                                   side_effect=lambda a, h: calls.append(a)):
                self.server.ensure_room()
                self.server.start_boardroom("a topic that runs long")
                st = self.server.load_state()
                st["boardroom"]["turn_paused"] = True
                st["boardroom"]["turns"] = self.server.BOARDROOM_TURN_BUDGET
                self.server.save_state(st)
                # Paused: no turns may be claimed.
                granted, _ = self.server._boardroom_claim_turns(1)
                self.assertEqual(granted, 0, "paused room must grant no turns")

                res = self.server.resume_boardroom()
                self.assertNotIn("error", res, f"resume failed: {res}")
                st = self.server.load_state()
                self.assertFalse(st["boardroom"]["turn_paused"])
                self.assertEqual(st["boardroom"]["turns"], 0)
                # Budget is genuinely restored, not just the flag flipped.
                granted, _ = self.server._boardroom_claim_turns(3)
                self.assertEqual(granted, 3, "resumed room must grant turns again")
            self.assertIn(self.server.CHAIR_SEAT, calls,
                          "resume must wake the chair so the room carries on")

    def test_resume_refuses_when_not_paused(self):
        """Resume costs another budget of paid turns, so it only applies to a
        genuinely paused room. An agent must never be able to top itself up."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td) / "rooms"
            rooms.mkdir()
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch", lambda *a, **k: None):
                self.server.ensure_room()
                self.server.start_boardroom("a topic still running")
                res = self.server.resume_boardroom()
            self.assertIn("error", res)
            self.assertIn("not paused", res["error"])

    def test_outcome_labels_where_its_summary_came_from(self):
        """An outside caller reads outcome.summary as the agreed outcome. When
        no real summary exists, the provenance must say so, or a stray chair
        remark gets presented to T09 as an agreement."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td) / "rooms"
            rooms.mkdir()
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch", lambda *a, **k: None):
                self.server.ensure_room()
                self.server.start_boardroom("a topic the chair never summarises")
                # Chair speaks mid-discussion, but never summarises.
                self.server.append_message(
                    self.server.CHAIR_SEAT, "Good point, though I'd push back on the timing.")
                res = self.server.close_boardroom()
            outcome = res.get("outcome") or {}
            self.assertEqual(
                outcome.get("summary_source"), "unsummarised-chair-remark",
                "a passing chair remark must never be labelled as the chair's summary",
            )

            rooms2 = Path(td) / "rooms2"
            rooms2.mkdir()
            with mock.patch.object(self.server, "ROOMS_DIR", rooms2), \
                 mock.patch.object(self.server, "CURRENT", rooms2 / "current"), \
                 mock.patch.object(self.server, "dispatch", lambda *a, **k: None):
                self.server.ensure_room()
                self.server.start_boardroom("a topic closed with a real summary")
                res2 = self.server.close_boardroom(summary="Agreed X. Disagreed Y. Recommend Z.")
            self.assertEqual((res2.get("outcome") or {}).get("summary_source"), "chair")

    def test_route_does_not_promote_a_chair_aside_to_the_summary(self):
        """Drives the REAL route() path, not close_boardroom directly. A chair
        remark mid-discussion must not become the room's agreed outcome."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td) / "rooms"
            rooms.mkdir()
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch", lambda *a, **k: None):
                self.server.ensure_room()
                self.server.start_boardroom("a topic with chatter")
                aside = self.server.append_message(
                    self.server.CHAIR_SEAT,
                    "Good point, though I would push back on the timing.")
                self.server.route(aside)
                st = self.server.load_state()
                self.assertNotEqual(
                    (st["boardroom"].get("summary") or ""), aside["text"],
                    "an ordinary chair remark must not be stored as the summary",
                )
                res = self.server.close_boardroom()
            outcome = res.get("outcome") or {}
            self.assertEqual(outcome.get("summary_source"), "unsummarised-chair-remark")

            rooms2 = Path(td) / "rooms2"
            rooms2.mkdir()
            with mock.patch.object(self.server, "ROOMS_DIR", rooms2), \
                 mock.patch.object(self.server, "CURRENT", rooms2 / "current"), \
                 mock.patch.object(self.server, "dispatch", lambda *a, **k: None):
                self.server.ensure_room()
                self.server.start_boardroom("a topic the chair summarises")
                real = self.server.append_message(
                    self.server.CHAIR_SEAT,
                    "We agree on the shape. Kimi and grok disagree on timing. "
                    "I recommend shipping the smaller version first. "
                    "Anything to dig into before I close?")
                self.server.route(real)
                res2 = self.server.close_boardroom()
            self.assertEqual((res2.get("outcome") or {}).get("summary_source"), "chair",
                             "a genuine chair summary must be labelled as one")

    def test_boardroom_never_starts_mission(self):
        """Criterion 6: boardroom mode never sets or runs the mission loop."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_prior_chat(rooms)
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch"):
                code, body = self._handler_post(
                    "/api/boardroom",
                    {
                        "topic": self.TOPIC,
                        "goal": "sneak mission",
                        "done": "when sneaked",
                    },
                )
            self.assertEqual(code, 200, body)
            live = json.loads((rooms / "current" / "state.json").read_text())
            self.assertIsNone(
                live.get("mission"),
                "boardroom must leave mission=None even if goal/done are posted",
            )
            self.assertIsInstance(live.get("boardroom"), dict)
            msgs = (rooms / "current" / "messages.jsonl").read_text()
            self.assertNotIn('"kind": "mission"', msgs)
            self.assertNotIn('"kind":"mission"', msgs)

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                self.assertIsNone(self.server.load_state().get("mission"))

    # ---- Round-2 standards fixes (turn budget, close-stops-billing, real guards)

    def test_boardroom_turn_budget_caps_all_fanout(self):
        """Finding 1: three seats tagging @all stay under an explicit turn budget.

        Hop depth alone cannot see fan-out. Drive the real process_turn path
        so every reply re-routes through route() the way production does.
        """
        import tempfile
        import time

        trio = ("claude", "grok", "kimi")
        engine_calls = []

        def fake_run_engine(agent, prompt, room_epoch=None):
            engine_calls.append(agent)
            # Each seat tags every other seat — the fan-out shape that explodes.
            others = " ".join(f"@{a}" for a in trio if a != agent)
            return f"{others} more thoughts from {agent}"

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_prior_chat(rooms)

            # Drain any leftover queue items from other tests.
            for a in self.server.AGENTS:
                q = self.server.WORK_QUEUES.get(a)
                if q is None:
                    continue
                while not q.empty():
                    try:
                        q.get_nowait()
                    except Exception:
                        break

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "AGENTS", trio), \
                 mock.patch.object(
                     self.server, "run_engine", side_effect=fake_run_engine), \
                 mock.patch.object(
                     self.server, "maybe_update_minutes", return_value=None):
                # Real dispatch → real queues → real process_turn.
                code, body = self._handler_post(
                    "/api/boardroom", {"topic": self.TOPIC})
                self.assertEqual(code, 200, body)

                budget = getattr(self.server, "BOARDROOM_TURN_BUDGET", None)
                self.assertIsInstance(
                    budget, int,
                    "server must expose BOARDROOM_TURN_BUDGET as an int",
                )
                self.assertGreater(budget, 0)

                # Process turns until queues empty or we clearly over-ran.
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    progress = False
                    for agent in trio:
                        q = self.server.WORK_QUEUES[agent]
                        if q.empty():
                            continue
                        try:
                            hops = q.get_nowait()
                        except Exception:
                            continue
                        self.server.process_turn(agent, hops)
                        progress = True
                    if not progress:
                        break

                self.assertLessEqual(
                    len(engine_calls), budget,
                    f"@all fan-out produced {len(engine_calls)} engine calls; "
                    f"must stay ≤ BOARDROOM_TURN_BUDGET={budget}",
                )
                self.assertGreater(
                    len(engine_calls), 0,
                    "invite must still run at least one engine turn",
                )
                # Budget must have been hit (otherwise fan-out would explode).
                live = self.server.load_state()
                br = live.get("boardroom") or {}
                self.assertGreaterEqual(
                    int(br.get("turns") or 0), 1,
                    "boardroom must track a turn counter",
                )
                feed = (rooms / "current" / "messages.jsonl").read_text().lower()
                if len(engine_calls) >= budget:
                    self.assertTrue(
                        "turn" in feed and ("limit" in feed or "paused" in feed),
                        "hitting the turn budget must post a plain-English pause",
                    )

    def test_close_boardroom_drains_queues_and_stops_billing(self):
        """Finding 2: closing drains queued work; nothing keeps billing."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_prior_chat(rooms)

            for a in self.server.AGENTS:
                q = self.server.WORK_QUEUES.get(a)
                if q is None:
                    continue
                while not q.empty():
                    try:
                        q.get_nowait()
                    except Exception:
                        break

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch") as disp:
                self._handler_post("/api/boardroom", {"topic": self.TOPIC})
                # Simulate a fan-out backlog that would keep billing after close.
                for agent in self.all_seats:
                    self.server.WORK_QUEUES[agent].put(3)
                    self.server.WORK_QUEUES[agent].put(4)
                self.server.append_message(
                    "claude", self.CHAIR_SUMMARY, hops=1)

                code, body = self._handler_post("/api/boardroom/close", {})
            self.assertEqual(code, 200, body)

            for agent in self.all_seats:
                self.assertTrue(
                    self.server.WORK_QUEUES[agent].empty(),
                    f"close must drain WORK_QUEUES[{agent!r}]",
                )
            live = json.loads((rooms / "current" / "state.json").read_text())
            self.assertTrue(
                live.get("stop"),
                "close must set stop so workers and route refuse further spend",
            )

            # A late agent reply must not re-dispatch after close.
            disp.reset_mock()
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch") as disp2:
                msg = self.server.append_message(
                    "grok", "@claude more after close", hops=2)
                self.server.route(msg)
            disp2.assert_not_called()

    def test_boardroom_path_has_no_idle_wake(self):
        """Finding 3: no timer/interval/scheduled wake on the boardroom path."""
        src = (ROOM / "server.py").read_text()
        self.assertNotIn(
            "boardroom_on_idle", src,
            "boardroom_on_idle was a no-op stub; delete it, do not keep it",
        )
        # Boardroom helpers must not schedule a wake (Timer / sleep loops).
        # Bound the search to the boardroom block of server.py.
        start = src.find("def start_boardroom")
        end = src.find("\ndef dispatch")
        self.assertGreater(start, 0, "start_boardroom must exist")
        block = src[start:end if end > start else start + 8000]
        for needle in (
            "threading.Timer",
            "Timer(",
            "time.sleep",
            "sched.",
            "call_later",
            "setInterval",
        ):
            self.assertNotIn(
                needle, block,
                f"boardroom path must not schedule wakes via {needle}",
            )
        # route/close must also lack boardroom timers (global scan for boardroom+Timer).
        self.assertIsNone(
            re.search(r"boardroom.{0,80}Timer|Timer.{0,80}boardroom", src, re.I | re.S),
            "no Timer may be wired to boardroom",
        )

    def test_idle_uses_process_turn_not_hand_rolled_worker(self):
        """Finding 4: idle criterion drives process_turn (the real worker body)."""
        import tempfile

        engine_calls = []

        def fake_run_engine(agent, prompt, room_epoch=None):
            engine_calls.append(agent)
            return "[silent]"

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_prior_chat(rooms)

            for a in self.server.AGENTS:
                q = self.server.WORK_QUEUES.get(a)
                if q is None:
                    continue
                while not q.empty():
                    try:
                        q.get_nowait()
                    except Exception:
                        break

            self.assertTrue(
                hasattr(self.server, "process_turn"),
                "worker body must be process_turn() so tests call the real path",
            )

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(
                     self.server, "run_engine", side_effect=fake_run_engine), \
                 mock.patch.object(
                     self.server, "maybe_update_minutes", return_value=None):
                code, body = self._handler_post(
                    "/api/boardroom", {"topic": self.TOPIC})
                self.assertEqual(code, 200, body)

                # Drain invite via the real turn processor (not a hand copy).
                for agent in list(self.all_seats):
                    q = self.server.WORK_QUEUES[agent]
                    while not q.empty():
                        hops = q.get_nowait()
                        self.server.process_turn(agent, hops)

                self.assertEqual(
                    sorted(engine_calls),
                    sorted(self.all_seats),
                    "invite should invoke each seat once via process_turn",
                )
                before = len(engine_calls)
                # No further messages → no further turns, room stays open.
                for agent in list(self.all_seats):
                    self.assertTrue(
                        self.server.WORK_QUEUES[agent].empty(),
                        f"idle must leave {agent} queue empty after silent replies",
                    )
                self.assertEqual(len(engine_calls), before)
                live = self.server.load_state()
                self.assertEqual(live["boardroom"]["status"], "open")
                self.assertIsNone(getattr(self.server, "BOARDROOM_OUTCOME", "x"))

    def test_chair_instruction_not_invite_text(self):
        """Finding 5: chair role block is tested, not invite-embedded words."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_prior_chat(rooms)
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch"):
                self._handler_post("/api/boardroom", {"topic": self.TOPIC})

            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"):
                feed = self.server.all_messages()
                chair = self.server.build_prompt("claude", feed)
                other = self.server.build_prompt("grok", feed)

            # Load-bearing chair line — must live in the chair block, not only the invite.
            self.assertIn(
                "You never close on your own", chair,
                "chair prompt must carry the no-self-close instruction",
            )
            self.assertIn("only the user closes", chair)
            # Non-chair seats must not get the chair close instruction.
            self.assertNotIn("You never close on your own", other)
            self.assertIn("Speak from your role", other)

    def test_chair_done_does_not_close_boardroom(self):
        """Finding 6: chair 'DONE' must not end a boardroom or fake mission-complete."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_prior_chat(rooms)
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch"):
                self._handler_post("/api/boardroom", {"topic": self.TOPIC})
                msg = self.server.append_message(
                    "claude",
                    "DONE\nAgreement: ship it.\nDisagreement: timing.\n"
                    "Recommendation: trial first.",
                    hops=2,
                )
                self.server.route(msg)
                live = self.server.load_state()
                msgs = self.server.all_messages()

            self.assertEqual(live["boardroom"]["status"], "open")
            self.assertIsNone(getattr(self.server, "BOARDROOM_OUTCOME", None))
            self.assertIsNone(live.get("mission"))
            joined = " ".join(m.get("text") or "" for m in msgs)
            self.assertNotIn(
                "Mission complete", joined,
                "chair DONE in a boardroom must not emit mission-complete",
            )

    def test_untagged_you_does_not_wake_chair_in_boardroom(self):
        """Finding 7: stray untagged the user lines must not cost a chair call."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_prior_chat(rooms)
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch") as disp:
                self._handler_post("/api/boardroom", {"topic": self.TOPIC})
                disp.reset_mock()
                msg = self.server.append_message(
                    "you", "just thinking out loud, no tags")
                self.server.route(msg)
            disp.assert_not_called()

    def test_outcome_cleared_when_boardroom_starts_or_room_swaps(self):
        """Finding 8: BOARDROOM_OUTCOME must not outlive its room."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rooms = Path(td)
            self._seed_prior_chat(rooms)
            with mock.patch.object(self.server, "ROOMS_DIR", rooms), \
                 mock.patch.object(self.server, "CURRENT", rooms / "current"), \
                 mock.patch.object(self.server, "dispatch"):
                self._handler_post("/api/boardroom", {"topic": self.TOPIC})
                self.server.append_message(
                    "claude", self.CHAIR_SUMMARY, hops=1)
                self._handler_post("/api/boardroom/close", {})
                self.assertIsNotNone(self.server.get_boardroom_outcome())

                # New boardroom must clear the prior outcome for T09 callers.
                self._handler_post(
                    "/api/boardroom", {"topic": "A different topic entirely"})
                self.assertIsNone(
                    self.server.get_boardroom_outcome(),
                    "starting a boardroom must clear the previous outcome",
                )

    def test_html_boardroom_start_and_close_controls(self):
        """Finding 10: the user can start and close a boardroom from the page."""
        html = (ROOM / "index.html").read_text()
        self.assertGreater(
            len(re.findall(r"(?i)boardroom", html)), 0,
            "index.html must mention boardroom (UI is currently missing)",
        )
        self.assertIn("/api/boardroom", html)
        self.assertIn("/api/boardroom/close", html)
        # Visible controls, not only a comment.
        self.assertRegex(
            html,
            r'id=["\']boardroomBtn["\']|id=["\']boardroomForm["\']',
            "page needs a visible boardroom start control",
        )
        self.assertRegex(
            html,
            r'id=["\']closeBoardroomBtn["\']|close boardroom|Close boardroom',
            "page needs a visible boardroom close control",
        )


if __name__ == "__main__":
    unittest.main()
