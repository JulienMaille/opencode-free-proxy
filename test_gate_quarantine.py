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
