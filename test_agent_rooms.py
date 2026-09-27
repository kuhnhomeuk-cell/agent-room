"""T09 / DM-13: /agent-rooms command — start/attach, post, wait, report.

Seams: command start/post/wait steps driven with a fake server (no real
server, no real browser, no real rooms/, no model engines).

Mutation bar: drive run() (the real command path), not only helpers.
Assert exact start counts, exact sleep intervals, exact outcome wording.
Round-2 standards: each honesty branch, unknown provenance, closed-status
gate, poll-interval floor, readiness retry, dead-server exit, I/O timeout.
"""
from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOM = Path(__file__).resolve().parent


class FakeServer:
    """In-memory stand-in for the room HTTP API used by agent_rooms.run."""

    def __init__(self):
        self.up = False
        self.start_count = 0
        self.boardroom_posts = []
        self.outcome = None  # None → 404; dict → 200 closed (after pending)
        self.http_log = []
        self.model_calls = 0
        self.sleep_log = []
        self.browser_opens = []
        self.polls_until_outcome = 0
        self._polls = 0
        # 200 + non-closed status this many times before real outcome (gate test).
        self.pending_200_polls = 0
        self.pending_200_status = "none"
        # After start: stay down for this many is_up checks (readiness retry).
        self.ready_after_is_up_checks = 0
        self._is_up_checks = 0
        # Force outcome endpoint unreachable (code 0).
        self.outcome_unreachable = False
        # After this many outcome polls, flip to unreachable (dead mid-wait).
        self.die_after_outcome_polls = 0

    def is_up(self) -> bool:
        self._is_up_checks += 1
        if not self.up:
            return False
        if self.ready_after_is_up_checks:
            # Count only checks while "started" flag is set.
            return self._is_up_checks >= self.ready_after_is_up_checks
        return True

    def start_server(self) -> None:
        self.start_count += 1
        self.up = True
        # Reset is_up counter so ready_after counts from the start call onward
        # only when delayed readiness is requested.
        if self.ready_after_is_up_checks:
            self._is_up_checks = 0

    def open_browser(self, url: str) -> None:
        self.browser_opens.append(url)

    def sleep(self, seconds: float) -> None:
        self.sleep_log.append(seconds)

    def http_json(self, method: str, path: str, body=None):
        """Fake HTTP. No real network. Tracks model-ish misuse as model_calls."""
        self.http_log.append((method, path, body))
        # The wait path must never look like an engine/model call.
        if path.startswith("/api/engine") or "dispatch" in path:
            self.model_calls += 1
        if method == "POST" and path.rstrip("/") == "/api/boardroom":
            topic = (body or {}).get("topic", "")
            self.boardroom_posts.append(topic)
            return 200, {"ok": True, "topic": topic}
        if method == "GET" and path.rstrip("/") == "/api/boardroom/outcome":
            self._polls += 1
            if self.outcome_unreachable:
                return 0, None
            if (
                self.die_after_outcome_polls
                and self._polls > self.die_after_outcome_polls
            ):
                return 0, None
            if self.pending_200_polls and self._polls <= self.pending_200_polls:
                return 200, {"status": self.pending_200_status,
                             "error": "boardroom still open"}
            if self.outcome is None:
                return 404, {"error": "no boardroom outcome yet", "status": "none"}
            if self.polls_until_outcome and self._polls <= self.polls_until_outcome:
                return 404, {"error": "no boardroom outcome yet", "status": "none"}
            return 200, dict(self.outcome)
        if method == "GET" and path.rstrip("/") == "/api/messages":
            return (200, {"messages": []}) if self.up else (0, None)
        return 404, {"error": "not found"}


def _deps_from_fake(
    fake: FakeServer,
    poll_interval: float = 5.0,
    **extra,
):
    import agent_rooms

    return agent_rooms.Deps(
        base_url="http://127.0.0.1:8787",
        poll_interval_s=poll_interval,
        is_up=fake.is_up,
        start_server=fake.start_server,
        open_browser=fake.open_browser,
        http_json=fake.http_json,
        sleep=fake.sleep,
        **extra,
    )


def _closed_chair(topic: str, summary: str = "Agreement: A. Disagreement: B. Recommend: C."):
    return {
        "status": "closed",
        "topic": topic,
        "summary": summary,
        "summary_source": "chair",
        "closed_at": "2026-08-03T12:00:00",
    }


class TestAgentRoomsCommand(unittest.TestCase):
    """DM-13 / T09: convene the room from another session."""

    TOPIC = "Should we ship the boardroom skill this week?"

    def test_second_invocation_attaches_not_starts(self):
        """Criterion 1: second ensure attaches; start_server runs once only."""
        import agent_rooms

        fake = FakeServer()
        deps = _deps_from_fake(fake)

        status1 = agent_rooms.ensure_server(deps)
        self.assertEqual(status1, "started")
        self.assertEqual(fake.start_count, 1)

        status2 = agent_rooms.ensure_server(deps)
        self.assertEqual(
            status2, "attached",
            "second call must attach to the running server, not start another",
        )
        self.assertEqual(
            fake.start_count, 1,
            "a second invocation must not start a duplicate server",
        )

        # Drive the real run() path twice with the same fake: still one start.
        fake2 = FakeServer()
        fake2.outcome = _closed_chair(self.TOPIC)
        out = io.StringIO()
        deps2 = _deps_from_fake(fake2)
        rc1 = agent_rooms.run(self.TOPIC, deps=deps2, stdout=out)
        self.assertEqual(rc1, 0)
        rc2 = agent_rooms.run(self.TOPIC, deps=deps2, stdout=out)
        self.assertEqual(rc2, 0)
        self.assertEqual(
            fake2.start_count, 1,
            "two full run() invocations against one live server start once only",
        )

    def test_run_opens_browser_and_posts_boardroom_topic(self):
        """Criteria 2: browser open + POST /api/boardroom with the exact topic."""
        import agent_rooms

        fake = FakeServer()
        fake.outcome = _closed_chair(self.TOPIC)
        out = io.StringIO()
        deps = _deps_from_fake(fake)
        rc = agent_rooms.run(self.TOPIC, deps=deps, stdout=out)
        self.assertEqual(rc, 0)
        self.assertEqual(fake.boardroom_posts, [self.TOPIC])
        self.assertTrue(fake.browser_opens, "must open the room in a browser")
        self.assertTrue(
            any("8787" in u for u in fake.browser_opens),
            f"browser URL must point at the room: {fake.browser_opens}",
        )
        posts = [c for c in fake.http_log if c[0] == "POST"]
        self.assertTrue(posts, "run() must POST to the server")
        self.assertEqual(posts[0][1].rstrip("/"), "/api/boardroom")
        self.assertEqual(posts[0][2].get("topic"), self.TOPIC)

    def test_wait_loop_is_slow_and_makes_no_model_call(self):
        """Criterion 4: slow poll, plain waiting message, zero model calls."""
        import agent_rooms

        fake = FakeServer()
        fake.polls_until_outcome = 3
        fake.outcome = _closed_chair(self.TOPIC)
        poll = 5.0
        out = io.StringIO()
        deps = _deps_from_fake(fake, poll_interval=poll)
        rc = agent_rooms.run(self.TOPIC, deps=deps, stdout=out)
        self.assertEqual(rc, 0)
        text = out.getvalue().lower()
        self.assertRegex(
            text, r"wait(ing)? on the room",
            "must say plainly that the session is waiting on the room",
        )
        self.assertGreaterEqual(
            len(fake.sleep_log), 3,
            "must sleep between polls while outcome is pending",
        )
        for s in fake.sleep_log:
            self.assertGreaterEqual(
                s, agent_rooms.MIN_POLL_INTERVAL_S,
                f"poll sleep {s} is tight spin; min is {agent_rooms.MIN_POLL_INTERVAL_S}",
            )
            self.assertEqual(s, poll, f"must use configured poll interval, got {s}")
        self.assertEqual(
            fake.model_calls, 0,
            "waiting must not invoke engines or model paths",
        )
        # No tight busy-loop: every outcome miss is paired with a sleep.
        outcome_gets = [
            c for c in fake.http_log
            if c[0] == "GET" and c[1].rstrip("/") == "/api/boardroom/outcome"
        ]
        # polls_until_outcome misses + one success
        self.assertGreaterEqual(len(outcome_gets), 4)
        self.assertGreaterEqual(len(fake.sleep_log), len(outcome_gets) - 1)

    def test_unsummarised_chair_remark_reported_honestly(self):
        """Acceptance: unsummarised-chair-remark is not an agreed decision.

        Mutation bar: deleting this branch (fall through to unknown) must fail.
        Unknown also says "not an agreed decision" and echoes summary_source,
        so we assert branch-unique wording that the unknown path cannot emit.
        """
        import agent_rooms

        fake = FakeServer()
        remark = "I think we should maybe try widget X next sprint."
        fake.outcome = {
            "status": "closed",
            "topic": self.TOPIC,
            "summary": remark,
            "summary_source": "unsummarised-chair-remark",
            "closed_at": "2026-08-03T12:00:00",
        }
        out = io.StringIO()
        rc = agent_rooms.run(self.TOPIC, deps=_deps_from_fake(fake), stdout=out)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn(remark, text)
        lower = text.lower()
        # Branch-unique phrases (unknown fallthrough lacks these exact labels).
        self.assertIn(
            "last chair remark:",
            lower,
            "unsummarised branch must label the remark; unknown fallthrough does not",
        )
        self.assertIn(
            "the chair never wrote a summary",
            lower,
            "unsummarised branch must explain the missing summary",
        )
        self.assertNotIn(
            "unknown summary_source",
            lower,
            "known unsummarised source must not use the unknown-provenance wording",
        )
        self.assertRegex(
            lower,
            r"not an agreed (outcome|decision)|not (an )?agreed outcome",
            "must state this is not an agreed outcome/decision",
        )
        self.assertNotRegex(
            lower,
            r"chair summary \(agreed|agreed outcome:",
            "must not present an unsummarised remark as the agreed outcome",
        )
        self.assertIn("unsummarised", lower)

    def test_placeholder_outcome_reported_honestly(self):
        """Acceptance: placeholder is not an agreed decision.

        Mutation bar: deleting this branch (fall through to unknown) must fail.
        Unknown echoes `summary_source: placeholder` so a bare `placeholder`
        assert is not enough — require the dedicated closed-before wording.
        """
        import agent_rooms

        fake = FakeServer()
        placeholder = (
            "The discussion was closed before the chair wrote a summary, "
            "so there is no agreed outcome to report."
        )
        fake.outcome = {
            "status": "closed",
            "topic": self.TOPIC,
            "summary": placeholder,
            "summary_source": "placeholder",
            "closed_at": "2026-08-03T12:00:00",
        }
        out = io.StringIO()
        rc = agent_rooms.run(self.TOPIC, deps=_deps_from_fake(fake), stdout=out)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        lower = text.lower()
        self.assertIn(
            "closed before the chair",
            lower,
            "placeholder branch must say the chair never summarised",
        )
        self.assertIn(
            "summarised (placeholder)",
            lower,
            "placeholder branch must name placeholder in its own sentence",
        )
        self.assertNotIn(
            "unknown summary_source",
            lower,
            "known placeholder must not use the unknown-provenance wording",
        )
        self.assertRegex(
            lower,
            r"no agreed outcome|not an agreed",
            "placeholder must not be sold as a decision",
        )
        self.assertNotIn("chair summary (agreed outcome)", lower)
        self.assertNotRegex(
            lower,
            r"(?<![nN]o )agreed outcome:|the (room )?decided",
        )

    def test_chair_summary_presented_as_agreed_outcome(self):
        """Positive control: a real chair summary is presented as the outcome."""
        import agent_rooms

        summary = (
            "Agreement: ship T09 today.\n"
            "Disagreement: poll interval 5s vs 10s.\n"
            "Recommendation: use 5s."
        )
        fake = FakeServer()
        fake.outcome = _closed_chair(self.TOPIC, summary)
        out = io.StringIO()
        rc = agent_rooms.run(self.TOPIC, deps=_deps_from_fake(fake), stdout=out)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn(summary, text)
        self.assertIn(self.TOPIC, text)
        lower = text.lower()
        self.assertRegex(lower, r"chair summary|agreed outcome")
        self.assertIn("chair summary (agreed outcome):", lower)
        self.assertNotIn("unsummarised", lower)
        self.assertNotIn("placeholder", lower)
        self.assertNotIn("unknown summary_source", lower)

    def test_unknown_summary_source_never_sold_as_agreed_decision(self):
        """Finding 2: malformed/future summary_source is not a decision.

        Mutation bar: replacing the unknown warning with
        'Chair summary (agreed outcome):' must fail this test.
        """
        import agent_rooms

        fake = FakeServer()
        fake.outcome = {
            "status": "closed",
            "topic": self.TOPIC,
            "summary": "Looks like we agreed to ship.",
            "summary_source": "future-new-kind",
            "closed_at": "2026-08-03T12:00:00",
        }
        out = io.StringIO()
        rc = agent_rooms.run(self.TOPIC, deps=_deps_from_fake(fake), stdout=out)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        lower = text.lower()
        self.assertIn(
            "unknown summary_source",
            lower,
            "unknown provenance must be labeled as unknown",
        )
        self.assertIn("not an agreed decision", lower)
        self.assertNotIn(
            "chair summary (agreed outcome)",
            lower,
            "must never upgrade unknown provenance to an agreed decision label",
        )
        self.assertIn("future-new-kind", text)
        # format_outcome unit check: same property without run() wrapper.
        formatted = agent_rooms.format_outcome(fake.outcome)
        self.assertIn("unknown summary_source", formatted.lower())
        self.assertNotIn("Chair summary (agreed outcome):", formatted)

    def test_wait_requires_closed_status_not_mere_http_200(self):
        """Finding 3: 200 with status != closed must keep waiting.

        Mutation bar: accepting status 'none' as done must fail this test.
        The old fake only returned 404 while pending, so a 200-with-pending
        path was never exercised.
        """
        import agent_rooms

        fake = FakeServer()
        fake.pending_200_polls = 3
        fake.outcome = _closed_chair(self.TOPIC)
        out = io.StringIO()
        deps = _deps_from_fake(fake, poll_interval=5.0)
        rc = agent_rooms.run(self.TOPIC, deps=deps, stdout=out)
        self.assertEqual(rc, 0)
        # Must have slept through the non-closed 200s.
        self.assertGreaterEqual(
            len(fake.sleep_log), 3,
            "must keep polling when HTTP 200 arrives with status != closed",
        )
        text = out.getvalue()
        self.assertIn("Chair summary (agreed outcome):", text)
        # If the gate accepted status none, format_outcome would get the
        # pending body and would not print the chair agreed-outcome label.
        self.assertIn(_closed_chair(self.TOPIC)["summary"], text)

    def test_only_closed_status_ends_the_wait(self):
        """The gate must be strict equality to 'closed'. Widening it to accept
        any other status value must fail a test, whichever value is chosen."""
        import agent_rooms

        # "closing" catches a prefix match; the arbitrary value catches a
        # mutant that simply adds another accepted status to the list.
        for pending_status in ("none", "pending", "open", "running", "",
                               "closing", "zzz-unknown-42"):
            with self.subTest(status=pending_status):
                fake = FakeServer()
                fake.pending_200_polls = 2
                fake.pending_200_status = pending_status
                fake.outcome = _closed_chair(self.TOPIC)
                out = io.StringIO()
                deps = _deps_from_fake(fake, poll_interval=5.0)
                rc = agent_rooms.run(self.TOPIC, deps=deps, stdout=out)
                self.assertEqual(rc, 0)
                self.assertGreaterEqual(
                    len(fake.sleep_log), 2,
                    f"status {pending_status!r} must not end the wait",
                )
                self.assertIn(_closed_chair(self.TOPIC)["summary"], out.getvalue())

    def test_unreachable_threshold_is_clamped_to_at_least_one(self):
        """A threshold of 0 or less would exit on the first transient blip, or
        loop forever, depending on the comparison. Clamp it."""
        import agent_rooms

        for bad in (0, -1, -99):
            with self.subTest(value=bad):
                deps = _deps_from_fake(FakeServer(), poll_interval=5.0)
                deps.max_consecutive_unreachable = bad
                deps.__post_init__()
                self.assertGreaterEqual(
                    deps.max_consecutive_unreachable, 1,
                    "an unreachable threshold below 1 is never valid",
                )

    def test_max_wait_is_never_shorter_than_one_poll(self):
        """A max wait shorter than the poll interval times out before the first
        poll, so the room could never be reported even if it closed at once."""
        import agent_rooms

        deps = _deps_from_fake(FakeServer(), poll_interval=5.0)
        deps.max_wait_s = 1.0
        deps.__post_init__()
        self.assertGreaterEqual(
            deps.max_wait_s, deps.poll_interval_s,
            "max wait must allow at least one poll",
        )

    def test_poll_interval_zero_is_clamped_on_deps_and_main(self):
        """Finding 4: --poll-interval 0 must not spin.

        Mutation bar: deleting either clamp (Deps.__post_init__ or main)
        must fail. Asserts real sleep values after injecting 0, not the
        test's own injected floor.
        """
        import agent_rooms

        # Clamp on Deps construction.
        deps0 = agent_rooms.Deps(
            poll_interval_s=0,
            is_up=lambda: True,
            start_server=lambda: None,
            open_browser=lambda u: None,
            http_json=lambda *a: (404, {"status": "none"}),
            sleep=lambda s: None,
        )
        self.assertGreaterEqual(
            deps0.poll_interval_s,
            agent_rooms.MIN_POLL_INTERVAL_S,
            "Deps must clamp poll_interval_s below the minimum",
        )

        # Full run path with poll_interval=0: sleeps must be >= floor.
        fake = FakeServer()
        fake.polls_until_outcome = 2
        fake.outcome = _closed_chair(self.TOPIC)
        out = io.StringIO()
        deps = _deps_from_fake(fake, poll_interval=0)
        rc = agent_rooms.run(self.TOPIC, deps=deps, stdout=out)
        self.assertEqual(rc, 0)
        self.assertTrue(fake.sleep_log, "wait must sleep between polls")
        for s in fake.sleep_log:
            self.assertGreaterEqual(
                s, agent_rooms.MIN_POLL_INTERVAL_S,
                f"--poll-interval 0 must clamp; slept {s}",
            )

        # CLI main path: --poll-interval 0.
        fake2 = FakeServer()
        fake2.polls_until_outcome = 2
        fake2.outcome = _closed_chair(self.TOPIC)
        deps_main = _deps_from_fake(fake2, poll_interval=99)  # overwritten by main
        out2 = io.StringIO()
        with mock.patch.object(agent_rooms, "default_deps", return_value=deps_main):
            rc2 = agent_rooms.main(
                [self.TOPIC, "--poll-interval", "0"],
                stdout=out2,
            )
        self.assertEqual(rc2, 0)
        self.assertGreaterEqual(
            deps_main.poll_interval_s,
            agent_rooms.MIN_POLL_INTERVAL_S,
            "main must clamp --poll-interval 0",
        )
        for s in fake2.sleep_log:
            self.assertGreaterEqual(s, agent_rooms.MIN_POLL_INTERVAL_S)

    def test_ensure_server_retries_until_ready(self):
        """Finding 5: readiness retry loop after start must be exercised.

        Mutation bar: deleting the for-loop retry keeps a sync-up fake green.
        This fake stays down on the first post-start check, then comes up.
        """
        import agent_rooms

        fake = FakeServer()
        # After start, first is_up is the immediate check (False until count 2).
        # ensure_server: start → immediate is_up → loop sleep+is_up.
        fake.ready_after_is_up_checks = 2
        deps = _deps_from_fake(fake)
        status = agent_rooms.ensure_server(deps)
        self.assertEqual(status, "started")
        self.assertEqual(fake.start_count, 1)
        self.assertTrue(
            fake.sleep_log,
            "must sleep while waiting for the newly started server to listen",
        )
        for s in fake.sleep_log:
            self.assertEqual(s, 0.5, f"readiness poll sleep must be 0.5s, got {s}")

    def test_ensure_server_raises_when_never_ready(self):
        """Readiness loop must give up rather than hang."""
        import agent_rooms

        fake = FakeServer()
        fake.ready_after_is_up_checks = 10_000  # never ready
        deps = _deps_from_fake(fake)
        with self.assertRaises(RuntimeError) as ctx:
            agent_rooms.ensure_server(deps)
        self.assertIn("did not become ready", str(ctx.exception).lower())

    def test_wait_exits_when_server_dies_mid_wait(self):
        """Finding 6: consecutive unreachable must stop the wait.

        Mutation bar: bare while-True with no exit on code 0 hangs forever.
        """
        import agent_rooms

        fake = FakeServer()
        fake.die_after_outcome_polls = 1  # one pending, then dead
        # Never set a closed outcome — server dies instead.
        out = io.StringIO()
        deps = _deps_from_fake(
            fake,
            poll_interval=2.0,
            max_consecutive_unreachable=3,
            max_wait_s=3600,
        )
        rc = agent_rooms.run(self.TOPIC, deps=deps, stdout=out)
        self.assertNotEqual(rc, 0, "dead server mid-wait must exit non-zero")
        text = out.getvalue().lower()
        self.assertTrue(
            "unreachable" in text or "stopped answering" in text or "server" in text,
            f"must explain the dead server in plain English: {out.getvalue()!r}",
        )
        # Must not have spun without bound: sleeps bounded by threshold + slack.
        self.assertLessEqual(len(fake.sleep_log), 10)

    def test_wait_exits_when_max_wait_exceeded(self):
        """Finding 6: maximum wait must stop an endless open room."""
        import agent_rooms

        fake = FakeServer()
        # Stay pending forever (404).
        fake.outcome = None
        out = io.StringIO()
        deps = _deps_from_fake(
            fake,
            poll_interval=2.0,
            max_consecutive_unreachable=100,
            max_wait_s=5.0,  # 3 polls * 2s >= 5
        )
        rc = agent_rooms.run(self.TOPIC, deps=deps, stdout=out)
        self.assertNotEqual(rc, 0, "max wait must exit non-zero")
        text = out.getvalue().lower()
        self.assertTrue(
            "timed out" in text or "time limit" in text or "too long" in text
            or "maximum wait" in text or "gave up" in text,
            f"must explain the timeout in plain English: {out.getvalue()!r}",
        )
        self.assertTrue(fake.sleep_log)

    def test_main_argv_path(self):
        """Drive the CLI entry (main) so the skill's command path is covered."""
        import agent_rooms

        fake = FakeServer()
        fake.outcome = _closed_chair(self.TOPIC)
        deps = _deps_from_fake(fake)
        out = io.StringIO()
        with mock.patch.object(agent_rooms, "default_deps", return_value=deps):
            rc = agent_rooms.main([self.TOPIC], stdout=out)
        self.assertEqual(rc, 0)
        self.assertEqual(fake.boardroom_posts, [self.TOPIC])
        self.assertIn(self.TOPIC, out.getvalue())

    def test_outcome_file_written_when_requested(self):
        """Skill background path: command writes the outcome to a file."""
        import agent_rooms
        import tempfile

        fake = FakeServer()
        fake.outcome = _closed_chair(self.TOPIC, "Agreement: file path works.")
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "outcome.txt"
            deps = _deps_from_fake(fake, outcome_file=str(path))
            rc = agent_rooms.run(self.TOPIC, deps=deps, stdout=out)
            self.assertEqual(rc, 0)
            self.assertTrue(path.is_file(), "must write --outcome-file")
            body = path.read_text()
            self.assertIn("Agreement: file path works.", body)
            self.assertIn("Chair summary (agreed outcome):", body)

    def test_skill_is_user_invoked_only(self):
        """Criterion 5: skill runs only when the user types /agent-rooms."""
        skill = Path.home() / ".claude" / "skills" / "agent-rooms" / "SKILL.md"
        self.assertTrue(skill.is_file(), f"skill missing at {skill}")
        text = skill.read_text()
        # Frontmatter: disable-model-invocation must be true.
        self.assertRegex(
            text,
            r"(?m)^disable-model-invocation:\s*true\s*$",
            "skill must set disable-model-invocation: true so it never auto-fires",
        )
        # Name matches the command.
        self.assertRegex(text, r"(?m)^name:\s*agent-rooms\s*$")

    def test_skill_does_not_block_one_bash_for_an_hour(self):
        """Finding 7: skill must not rely on a foreground hour-long block.

        Bash tool max timeout is ~10 minutes. Skill must background the
        command and/or re-read an outcome file.
        """
        skill = Path.home() / ".claude" / "skills" / "agent-rooms" / "SKILL.md"
        text = skill.read_text()
        lower = text.lower()
        # Must not instruct a pure foreground block as the wait strategy.
        self.assertNotRegex(
            lower,
            r"let it block|shell out to the cli below and let it block",
            "skill must not tell the harness to block one Bash call for the wait",
        )
        self.assertTrue(
            "background" in lower
            or "outcome-file" in lower
            or "--outcome-file" in text
            or "outcome file" in lower,
            "skill must background the wait or re-read an outcome file",
        )
        self.assertTrue(
            "ten minute" in lower
            or "10 minute" in lower
            or "timeout" in lower
            or "do not block" in lower
            or "don't block" in lower
            or "must not block" in lower,
            "skill must name the timeout/block constraint so the agent obeys it",
        )


class TestDefaultIoLayer(unittest.TestCase):
    """Finding 8: real I/O helpers — timeout, decode fallbacks, unreachable."""

    def _deps_with_default_http(self):
        import agent_rooms

        return agent_rooms.Deps(
            is_up=lambda: False,
            start_server=lambda: None,
            open_browser=lambda u: None,
            sleep=lambda s: None,
            # http_json left None → __post_init__ binds _default_http_json
        )

    def test_http_json_passes_timeout_five(self):
        import agent_rooms

        deps = self._deps_with_default_http()
        mock_resp = mock.MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"ok": true}'
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp) as urlopen:
            code, body = deps.http_json("GET", "/api/messages", None)
        self.assertEqual(code, 200)
        self.assertEqual(body, {"ok": True})
        # timeout=5 must be on the call (keyword or positional).
        args, kwargs = urlopen.call_args
        timeout = kwargs.get("timeout")
        if timeout is None and len(args) >= 2:
            timeout = args[1]
        self.assertEqual(
            timeout, 5,
            "urlopen must use timeout=5 so a hung server cannot block forever",
        )

    def test_http_json_urlerror_returns_zero_none(self):
        """Unreachable convention finding 6 depends on."""
        deps = self._deps_with_default_http()
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            code, body = deps.http_json("GET", "/api/boardroom/outcome", None)
        self.assertEqual(code, 0)
        self.assertIsNone(body)

    def test_http_json_httperror_parses_json_body(self):
        deps = self._deps_with_default_http()
        fp = io.BytesIO(b'{"error":"no boardroom outcome yet","status":"none"}')
        err = urllib.error.HTTPError(
            "http://127.0.0.1:8787/api/boardroom/outcome",
            404,
            "Not Found",
            hdrs=None,
            fp=fp,
        )
        with mock.patch("urllib.request.urlopen", side_effect=err):
            code, body = deps.http_json("GET", "/api/boardroom/outcome", None)
        self.assertEqual(code, 404)
        self.assertEqual(body.get("status"), "none")
        self.assertIn("no boardroom", body.get("error", ""))

    def test_http_json_bad_json_falls_back_to_raw(self):
        deps = self._deps_with_default_http()
        mock_resp = mock.MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"not-json-at-all"
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            code, body = deps.http_json("GET", "/api/messages", None)
        self.assertEqual(code, 200)
        self.assertEqual(body, {"raw": "not-json-at-all"})

    def test_http_json_posts_json_body(self):
        deps = self._deps_with_default_http()
        mock_resp = mock.MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"ok":true}'
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=mock_resp) as urlopen:
            code, body = deps.http_json(
                "POST", "/api/boardroom", {"topic": "ship it"}
            )
        self.assertEqual(code, 200)
        req = urlopen.call_args.args[0]
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.data, json.dumps({"topic": "ship it"}).encode())

    def test_default_is_up_true_on_200(self):
        import agent_rooms

        deps = agent_rooms.Deps(
            start_server=lambda: None,
            open_browser=lambda u: None,
            sleep=lambda s: None,
        )
        mock_resp = mock.MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"messages":[]}'
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            self.assertTrue(deps.is_up())

    def test_default_is_up_false_on_urlerror(self):
        import agent_rooms

        deps = agent_rooms.Deps(
            start_server=lambda: None,
            open_browser=lambda u: None,
            sleep=lambda s: None,
        )
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("down"),
        ):
            self.assertFalse(deps.is_up())

    def test_default_start_server_spawns_detached(self):
        import agent_rooms

        deps = agent_rooms.Deps(
            is_up=lambda: False,
            open_browser=lambda u: None,
            http_json=lambda *a: (0, None),
            sleep=lambda s: None,
            server_py=Path("/tmp/fake-server.py"),
        )
        with mock.patch("subprocess.Popen") as popen:
            deps.start_server()
        popen.assert_called_once()
        args, kwargs = popen.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], sys.executable)
        self.assertTrue(str(cmd[1]).endswith("fake-server.py"))
        self.assertTrue(
            kwargs.get("start_new_session"),
            "start_server must detach (start_new_session=True)",
        )
        # stdout/stderr discarded so a detached room does not pin the parent.
        import subprocess as _sp

        self.assertIs(kwargs.get("stdout"), _sp.DEVNULL)
        self.assertIs(kwargs.get("stderr"), _sp.DEVNULL)

    def test_default_open_browser_invokes_open(self):
        import agent_rooms

        deps = agent_rooms.Deps(
            is_up=lambda: True,
            start_server=lambda: None,
            http_json=lambda *a: (0, None),
            sleep=lambda s: None,
        )
        with mock.patch("webbrowser.open") as opener:
            deps.open_browser("http://127.0.0.1:8787/")
        opener.assert_called_once_with("http://127.0.0.1:8787/")

    def test_default_open_browser_swallows_oserror(self):
        import agent_rooms

        deps = agent_rooms.Deps(
            is_up=lambda: True,
            start_server=lambda: None,
            http_json=lambda *a: (0, None),
            sleep=lambda s: None,
        )
        with mock.patch("subprocess.Popen", side_effect=OSError("no open")):
            deps.open_browser("http://127.0.0.1:8787/")  # must not raise


if __name__ == "__main__":
    unittest.main()
