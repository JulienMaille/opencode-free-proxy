"""Non-retriable 5xx from upstream: roll once, penalize nothing (relay-500).

Measured 2026-10-18 (10:07->10:18, deepseek-v4.1-flash-free): all 24 requests
answered the SAME empty-bodied 500, on every exit, and the old cooldown +
``_mark_fail`` per exit swept the pool while penalizing exits that were fine.
These tests pin the zero-penalty contract, mirroring relay-413's
``report_too_large``. No network access: pool state is crafted in memory.
"""
import asyncio
import time

from proxy_pool import ProxyPool


def _make_pool():
    pool = ProxyPool()
    # Suppress disk persistence during tests (stats save at most once/min).
    pool._last_stats_save = time.time()
    return pool


def test_upstream_500_rotates_with_zero_penalty():
    pool = _make_pool()
    addr = "9.9.9.9:4145"
    entry = pool._entry(addr)
    before = dict(pool._stat(addr))
    assert pool.report_upstream_internal_error(addr, 500) is True
    assert entry["cooling_until"] == 0.0, "no cooldown for a 5xx"
    assert not pool._is_cooling(addr), "not cooling"
    assert entry["consecutive_failures"] == 0, "no failure increment"
    assert entry["fail"] == 0, "no entry fail counter"
    assert addr not in pool.blacklist, "not blacklisted"
    assert addr not in pool.rate_limits, "not rate-limited"
    assert pool.stats[addr] == before, "no ok/fail counter change"


def test_upstream_500_clears_sticky_current():
    pool = _make_pool()
    addr = "9.9.9.9:4145"
    pool.current = {"address": addr}
    pool.report_upstream_internal_error(addr, 500)
    assert pool.current is None, "sticky rotates so the next select can move on"


def test_retriable_statuses_keep_their_penalty():
    """The 5xx change must not weaken 504/429 handling."""
    pool = _make_pool()
    addr = "9.9.9.9:4145"
    pool.report_http_status(addr, 504)
    assert pool._is_cooling(addr), "504 still cools"
    assert pool._entry(addr)["consecutive_failures"] == 1


def test_rolled_exit_is_excluded_so_the_roll_reaches_a_sibling():
    """report_upstream_internal_error only clears the sticky; the retry loops
    exclude the exit (gated_exits) so the single roll really lands on another
    exit rather than re-selecting the one that just returned the 5xx."""
    pool = _make_pool()
    pool._ready = True
    now = time.time()
    pool.hot = [{"address": f"10.9.0.{i}:4145", "verified_at": now} for i in range(1, 16)]
    used = pool.hot[0]["address"]
    pool.current = pool.hot[0]
    pool.report_upstream_internal_error(used, 500)
    nxt = asyncio.run(pool._select_unlocked(exclude={used}))
    assert nxt is not None
    assert nxt["address"] != used, "the excluded exit must not come back"
