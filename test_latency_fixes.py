"""Corrected latency fixes: size-aware first-byte deadline, attempt cap,
gate fast-break, hedge with SOFT first-byte accounting.

Measured on the live run (2026-10-19, ``req=8f51aeee``, b2aee4b): one request
spent 191s failing because a dead SOCKS exit stalled 129s (bounded, before
this, only by the 120s read timeout), every retry re-uploaded ~360KB of
context, ``_maybe_backoff`` slept on deterministic answers, and up to 6
attempts ran sequentially.

Corrected vs the reverted commit: the deadline scales with the uploaded body
(a flat 15s kill large contexts on slow-but-alive exits) and a deadline miss
is a SOFT pool signal — booking it hard immediately blacklisted exits for
120min and drained the pool to 100% failure. No network access here: the
client, the transport and the pool are stubbed.
"""
import asyncio
import sys

import httpx
import pytest

# server parses its own CLI args at import; keep pytest's argv out of that.
_argv = sys.argv
sys.argv = [sys.argv[0]]
import server as srv  # noqa: E402
from proxy_pool import (  # noqa: E402
    FIRST_BYTE_TIMEOUT_BASE,
    FIRST_BYTE_UPLOAD_BPS,
    MAX_UPSTREAM_ATTEMPTS,
)

sys.argv = _argv


class _Resp:
    def __init__(self, status):
        self.status_code = status
        self.closed = False

    async def aclose(self):
        self.closed = True


class _FakeRequest:
    """Enough of an httpx Request for _first_byte_timeout."""

    def __init__(self, body_len):
        self.content = b"x" * body_len

    @property
    def headers(self):
        return {}

    def __repr__(self):
        return f"<FakeRequest {len(self.content)}B>"


class _Send:
    """``send`` that answers after ``delay`` seconds (or never)."""

    def __init__(self, delay, status=200, request_obj=None):
        self.delay = delay
        self.status = status
        self.calls = 0

    async def send(self, request, stream=True):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return _Resp(self.status)


# ── first-byte deadline ────────────────────────────────────────────────────


def test_first_byte_deadline_scales_with_body_length():
    from proxy_pool import FIRST_BYTE_TIMEOUT_CAP

    small = srv._first_byte_timeout(_FakeRequest(10_000))
    big = srv._first_byte_timeout(_FakeRequest(300_000))
    assert small == pytest.approx(FIRST_BYTE_TIMEOUT_BASE, abs=1.0)
    assert big > small, "a large context must get extra upload time"
    assert big <= FIRST_BYTE_TIMEOUT_CAP, "never exceed the cap"


def test_first_byte_deadline_is_capped():
    # 10MB at the configured rate would exceed the cap; cap must win so one
    # huge context never pins an exit queue for minutes.
    from proxy_pool import FIRST_BYTE_TIMEOUT_CAP

    t = srv._first_byte_timeout(_FakeRequest(10_000_000))
    assert t == FIRST_BYTE_TIMEOUT_CAP


def test_send_bounded_times_out_a_stalled_exit(monkeypatch):
    monkeypatch.setattr(srv, "FIRST_BYTE_TIMEOUT_BASE", 0.05)
    client = _Send(5.0)
    with pytest.raises(httpx.ConnectTimeout) as exc:
        asyncio.run(
            srv._send_bounded(client, _FakeRequest(0))
        )
    assert "no response headers within" in str(exc.value)
    assert client.calls == 1


def test_send_bounded_passes_a_fast_response():
    resp = asyncio.run(srv._send_bounded(_Send(0.0), _FakeRequest(0)))
    assert resp.status_code == 200


def test_first_byte_timeout_is_classified_soft():
    # The deadline miss must down-classify pool accounting — the flat-15s
    # version hard-blacklisted every slow exit and drained the pool.
    dead_tunnel = httpx.ConnectError("SOCKS handshake failed")
    deadline_miss = httpx.ConnectTimeout("no response headers within 15s")
    assert srv._is_first_byte_timeout(deadline_miss) is True
    assert srv._is_first_byte_timeout(dead_tunnel) is False
    assert srv._is_first_byte_timeout(None) is False
    assert srv._is_first_byte_timeout(RuntimeError("no response headers within 15s")) is False


# ── attempt cap ────────────────────────────────────────────────────────────


def test_attempt_budget_clamps_to_the_ceiling():
    assert srv._attempt_budget(None) == MAX_UPSTREAM_ATTEMPTS - 1
    assert srv._attempt_budget(5) == MAX_UPSTREAM_ATTEMPTS - 1
    assert srv._attempt_budget(1) == 1
    assert srv._attempt_budget(0) == 0
    assert srv._attempt_budget("bogus") == MAX_UPSTREAM_ATTEMPTS - 1


# ── deterministic answers do not sleep ─────────────────────────────────────


@pytest.mark.parametrize("status", [403, 413, 504])
def test_maybe_backoff_fast_breaks_deterministic_status(monkeypatch, status):
    async def boom(*args, **kwargs):
        raise AssertionError("must not sleep on a deterministic answer")

    monkeypatch.setattr(srv, "_backoff", boom)
    asyncio.run(srv._maybe_backoff(2, status))


def test_maybe_backoff_still_sleeps_on_transient_status(monkeypatch):
    slept = []

    async def fake_backoff(attempt, base=1.0, max_delay=10.0):
        slept.append(attempt)

    monkeypatch.setattr(srv, "_backoff", fake_backoff)
    asyncio.run(srv._maybe_backoff(2, 429))
    assert slept == [2]


# ── gate hedge ─────────────────────────────────────────────────────────────


class _FakePool:
    def __init__(self):
        self.released = []
        self.rate_limited = []
        self.current = None

    def release(self, addr=None):
        self.released.append(addr)

    def report_ratelimit(self, addr=None):
        self.rate_limited.append(addr)

    def get_client(self, url, streaming=True):
        return _FakeClient(url)


class _FakeClient:
    def __init__(self, url):
        self.url = url

    def build_request(self, method, path, json=None, headers=None):
        return {"url": self.url, "method": method, "path": path}


def _hedge_env(monkeypatch, answers):
    """Wire a fake pool + transport; returns (pool, reported, pool_reports).

    ``answers`` maps exit address to ``(delay, status)``; the delay decides
    which exit answers first, so the hedge's race is deterministic.
    ``reported`` collects gate reports; ``pool_reports`` collects
    ``(addr, hard)`` records passed to ``_safe_pool_failure``.
    """
    pool = _FakePool()
    reported = []
    pool_reports = []
    queue = [{"address": a} for a in answers]

    async def fake_select(exclude=None):
        return queue.pop(0) if queue else None

    async def fake_send(client, request, *, stream=True):
        delay, status = answers[request["url"].rsplit("/", 1)[-1]]
        await asyncio.sleep(delay)
        return _Resp(status)

    monkeypatch.setattr(srv, "PROXY_POOL_ENABLED", True)
    monkeypatch.setattr(srv, "proxy_pool", pool)
    monkeypatch.setattr(srv, "_select_with_repoll", fake_select)
    monkeypatch.setattr(srv, "_send_bounded", fake_send)
    monkeypatch.setattr(srv, "_report_free_tier_block", lambda addr: reported.append(addr))
    monkeypatch.setattr(
        srv,
        "_safe_pool_failure",
        lambda addr, exc=None, hard=False: pool_reports.append((addr, hard)),
    )
    return pool, reported, pool_reports


def _run_hedge():
    return asyncio.run(
        srv._hedge_gate_retry(
            method="POST",
            path="/inference/v1/chat/completions",
            json_body={"messages": []},
            headers={},
            exclude={"z:9"},
        )
    )


def test_hedge_keeps_the_first_2xx_and_releases_the_loser(monkeypatch):
    pool, reported, _ = _hedge_env(
        monkeypatch, {"a:1": (0.0, 403), "b:2": (0.01, 200)}
    )
    result = _run_hedge()
    assert result is not None
    resp, client, addr, gated, failed = result
    assert addr == "b:2"
    assert resp.status_code == 200
    assert gated == ["a:1"]
    assert failed == []
    assert reported == ["a:1"], "the losing 403 still counts as a gate"
    assert pool.released == ["a:1"], "only the winner keeps its slot"
    assert pool.current == {"address": "b:2"}


def test_hedge_books_a_gate_that_lands_with_the_winner(monkeypatch):
    # Both answers land in the same wait batch; whichever is read second must
    # still be booked (winner picked first -> the post-pass handles the gate).
    pool, reported, _ = _hedge_env(
        monkeypatch, {"a:1": (0.0, 403), "b:2": (0.0, 200)}
    )
    result = _run_hedge()
    assert result is not None
    assert result[2] == "b:2"
    assert result[3] == ["a:1"]
    assert reported == ["a:1"]
    assert pool.released == ["a:1"]


def test_hedge_cancels_a_slow_loser(monkeypatch):
    # The winner answers first, so the stalled exit is cancelled and its slot
    # is released without waiting for its deadline.
    pool, reported, _ = _hedge_env(
        monkeypatch, {"a:1": (0.5, 403), "b:2": (0.0, 200)}
    )
    result = _run_hedge()
    assert result is not None
    assert result[2] == "b:2"
    assert result[3] == [], "a cancelled loser has no answer to book"
    assert reported == []
    assert pool.released == ["a:1"]


def test_hedge_returns_none_when_every_exit_gates(monkeypatch):
    pool, reported, _ = _hedge_env(
        monkeypatch, {"a:1": (0.0, 403), "b:2": (0.0, 403)}
    )
    assert _run_hedge() is None
    assert sorted(reported) == ["a:1", "b:2"]
    assert sorted(pool.released) == ["a:1", "b:2"]


def test_hedge_reports_a_429_loser_as_rate_limit(monkeypatch):
    pool, _, pool_reports = _hedge_env(
        monkeypatch, {"a:1": (0.0, 429), "b:2": (0.01, 200)}
    )
    result = _run_hedge()
    assert result is not None
    assert result[2] == "b:2"
    assert pool.rate_limited == ["a:1"]
    assert result[4] == ["a:1"], "429 losers are booked, not marked failed"
    assert pool_reports == []
    assert pool.released == ["a:1"]


def test_hedge_reports_a_transport_failure_as_soft(monkeypatch):
    # CORRECTED: a transport failure inside the hedge is a SOFT pool signal.
    # Booking it hard would blacklist slow-but-alive exits after deadline
    # misses and (measured 2026-10-19) drain the pool to 100% failure.
    pool, _, _ = _hedge_env(
        monkeypatch, {"a:1": (0.0, 403), "b:2": (0.0, 200)}
    )

    async def failing_send(client, request, *, stream=True):
        if request["url"].endswith("a:1"):
            raise httpx.ConnectTimeout(
                "no response headers within 15s", request=request
            )
        return _Resp(200)

    monkeypatch.setattr(srv, "_send_bounded", failing_send)
    recorded = []
    monkeypatch.setattr(
        srv,
        "_safe_pool_failure",
        lambda addr, exc=None, hard=False: recorded.append((addr, hard)),
    )
    result = _run_hedge()
    assert result is not None
    assert result[2] == "b:2"
    assert pool.released == ["a:1"]
    assert recorded == [("a:1", False)], "hedge transport failure stays soft"


def test_hedge_bails_when_only_a_gated_exit_is_left(monkeypatch):
    pool, reported, _ = _hedge_env(monkeypatch, {"z:9": (0.0, 403)})
    assert _run_hedge() is None
    assert pool.released == ["z:9"]
    assert reported == []


def test_hedge_is_a_noop_when_the_pool_is_disabled(monkeypatch):
    monkeypatch.setattr(srv, "PROXY_POOL_ENABLED", False)
    assert _run_hedge() is None


def test_hedge_wins_and_lower_status_booked_without_penalty(monkeypatch):
    # A 5xx loser is upstream/fault-of-that-exit: no gate booked, no hard
    # failure, but the slot is released and the loser listed as failed.
    pool, reported, pool_reports = _hedge_env(
        monkeypatch, {"a:1": (0.0, 500), "b:2": (0.01, 200)}
    )
    result = _run_hedge()
    assert result is not None
    assert result[2] == "b:2"
    assert reported == []
    assert pool_reports == []  # 5xx inside the hedge = no pool penalty
    assert result[4] == ["a:1"]
    assert pool.released == ["a:1"]


def test_safe_pool_failure_drops_the_grace_retry_on_a_deadline_miss(monkeypatch):
    # A first-byte deadline miss already spent the full attempt window, so it
    # must not also earn the same-exit grace retry; other soft failures keep it.
    calls = []
    monkeypatch.setattr(srv, "PROXY_POOL_ENABLED", True)
    monkeypatch.setattr(
        srv.proxy_pool,
        "report_failure",
        lambda addr, **kw: calls.append((addr, kw)),
    )
    srv._safe_pool_failure(
        "a:1", httpx.ConnectTimeout("no response headers within 45s"), hard=False
    )
    srv._safe_pool_failure("b:2", httpx.ReadError("torn tunnel"), hard=False)
    assert calls == [
        ("a:1", {"hard": False, "grace": False}),
        ("b:2", {"hard": False, "grace": True}),
    ]
