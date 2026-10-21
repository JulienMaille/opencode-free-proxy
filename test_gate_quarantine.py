"""FreeTier-gate exit quarantine + per-request exclusion (offline).

An exit IP that gates every request (burned exit) must stop being selected,
while single isolated gates stay penalty-free. No network access: all pool
state is crafted in-memory, hot buffer kept >= HOT_MIN so no refill fires.
"""
import asyncio
import time

from proxy_pool import ProxyPool


def _make_pool():
    pool = ProxyPool()
    # Suppress disk persistence during tests (stats save at most once/min).
    pool._last_stats_save = time.time()
    return pool


def _hot(n=6):
    now = time.time()
    return [{"address": f"10.9.0.{i}:4145", "verified_at": now} for i in range(1, n + 1)]


def test_gate_strikes_quarantine_exit():
    pool = _make_pool()
    addr = "9.9.9.9:4145"
    assert not pool._is_cooling(addr)
    pool.report_free_tier_block(addr)
    pool.report_free_tier_block(addr)
    assert not pool._is_cooling(addr), "2 strikes must stay penalty-free"
    pool.report_free_tier_block(addr)
    assert pool._is_cooling(addr), "3rd strike in window must quarantine"


def test_gate_success_resets_streak():
    pool = _make_pool()
    addr = "9.9.9.9:4145"
    pool.report_free_tier_block(addr)
    pool.report_free_tier_block(addr)
    pool.report_success(addr)
    assert not pool._is_cooling(addr)
    pool.report_free_tier_block(addr)
    pool.report_free_tier_block(addr)
    assert not pool._is_cooling(addr), "streak must have reset on success"


def test_select_skips_excluded_exit():
    async def go():
        pool = _make_pool()
        pool.hot = _hot()
        pool.current = {"address": "10.9.0.1:4145"}
        p = await pool.select(exclude={"10.9.0.1:4145"})
        assert p is not None
        assert p["address"] != "10.9.0.1:4145"
        pool.release(p["address"])
    asyncio.run(go())


def test_select_last_resort_when_all_excluded():
    async def go():
        pool = _make_pool()
        pool.hot = _hot()
        pool.current = None
        addrs = {p["address"] for p in pool.hot}
        p = await pool.select(exclude=set(addrs))
        assert p is not None, "exclusion must never strand the request"
        pool.release(p["address"])
    asyncio.run(go())


def test_select_verifies_fresh_candidate_when_hot_all_excluded():
    async def go():
        pool = _make_pool()
        pool.hot = _hot()
        pool.current = None
        fresh = {"address": "10.9.1.1:4145", "source": "test"}
        pool.candidates = [fresh]

        async def _ok(p):
            return p["address"] == fresh["address"]

        pool._verify_with_retry = _ok
        gated = {p["address"] for p in pool.hot}
        p = await pool.select(exclude=set(gated))
        assert p is not None
        assert p["address"] == fresh["address"], (
            "an untouched candidate must be probed before a gated exit is reused"
        )
        pool.release(p["address"])
    asyncio.run(go())


def test_select_last_resorts_after_failed_fresh_candidates():
    async def go():
        pool = _make_pool()
        pool.hot = _hot()
        pool.current = None
        pool.candidates = [{"address": "10.9.1.1:4145", "source": "test"}]

        async def _bad(p):
            return False

        pool._verify_with_retry = _bad
        gated = {p["address"] for p in pool.hot}
        p = await pool.select(exclude=set(gated))
        assert p is not None, "a failed fresh probe must still fall back to a hot exit"
        assert p["address"] in gated
        pool.release(p["address"])
    asyncio.run(go())


def test_deadline_failure_rotates_without_blacklist_or_grace_retry():
    # Live evidence: exit 199.58.185.9 answered "no response headers within
    # 45s" and the very next attempt went to that same exit, burning a second
    # full deadline. A deadline miss already consumed the attempt budget, so it
    # must rotate immediately (cooldown, no blacklist) instead of earning the
    # soft grace retry.
    async def go():
        pool = _make_pool()
        pool.hot = _hot()
        pool.current = pool.hot[0]
        addr = pool.current["address"]
        pool.report_failure(addr, hard=False, grace=False)
        assert pool.blacklist.get(addr, 0) <= time.time(), (
            "a slow-but-alive exit must not be blacklisted for a deadline miss"
        )
        assert pool.current is None, "sticky exit must be cleared to rotate"
        assert addr not in {p["address"] for p in pool.hot}
        p = await pool.select()
        assert p is not None and p["address"] != addr
        pool.release(p["address"])
    asyncio.run(go())


def test_soft_failure_still_gets_one_grace_retry_on_the_same_exit():
    async def go():
        pool = _make_pool()
        pool.hot = _hot()
        pool.current = pool.hot[0]
        addr = pool.current["address"]
        pool.report_failure(addr, hard=False)  # grace=True default
        assert pool.current is not None and pool.current["address"] == addr
        p = await pool.select()
        assert p is not None and p["address"] == addr, (
            "a fast soft failure keeps the documented grace retry"
        )
        pool.release(p["address"])
    asyncio.run(go())
