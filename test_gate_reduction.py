#!/usr/bin/env python3
"""FreeTier 403 reduction invariants (offline, no network, no live probes).

Covers the three fixes:
  1. A gate is counted against the DISTINCT exits a request hit, and a pool
     last-resort repeat is flagged instead of counted again.
  2. A conversation that gated out recently gets its rotation capped to one
     exit (a cheap recovery probe), then the memory expires.
  3. Conversation -> session ids survive a restart, and every Zen request
     carries a stable prompt_cache_key so a known conversation is not treated
     as a fresh session upstream.

Run with `python -m pytest test_gate_reduction.py -q`.
"""
import sys
import time

import pytest

# server.py parses CLI args at import: stub argv first so pytest's own
# argv never leaks into argparse.
sys.argv = ["test_gate_reduction"]

import server as srv


@pytest.fixture(autouse=True)
def _isolate_session_state(tmp_path, monkeypatch):
    """Never touch the real sessions.json, and start each test with no memory."""
    monkeypatch.setattr(srv, "_SESSIONS_FILE", tmp_path / "sessions.json")
    srv._user_sessions.clear()
    srv._last_sessions_save = 0.0
    yield
    srv._user_sessions.clear()


# ── 1. distinct-exit gate accounting ───────────────────────────────

def test_note_gate_counts_distinct_exits_and_flags_pooled_repeat():
    gated = set()
    blocks = []
    prev_enabled = srv.PROXY_POOL_ENABLED
    prev_report = srv._report_free_tier_block
    srv.PROXY_POOL_ENABLED = True
    srv._report_free_tier_block = lambda addr: blocks.append(addr)
    try:
        count, exhausted = srv._note_free_tier_gate(0, gated, "1.1.1.1:4145")
        assert (count, exhausted) == (1, False)
        count, exhausted = srv._note_free_tier_gate(count, gated, "2.2.2.2:4145")
        assert (count, exhausted) == (2, False)
        # The pool handed back an exit this request already gated (its
        # last-resort path): NOT a third distinct exit.
        count, exhausted = srv._note_free_tier_gate(count, gated, "1.1.1.1:4145")
        assert (count, exhausted) == (2, True)
        assert blocks == ["1.1.1.1:4145", "2.2.2.2:4145"], "one strike per distinct exit"
    finally:
        srv.PROXY_POOL_ENABLED = prev_enabled
        srv._report_free_tier_block = prev_report


def test_note_gate_treats_direct_as_its_own_issuer():
    gated = set()
    prev_enabled = srv.PROXY_POOL_ENABLED
    prev_report = srv._report_free_tier_block
    srv.PROXY_POOL_ENABLED = True
    srv._report_free_tier_block = lambda addr: None
    try:
        count, exhausted = srv._note_free_tier_gate(0, gated, None)
        assert (count, exhausted) == (1, False)
        # A second direct gate cannot be helped by rotating: no fresh issuer.
        count, exhausted = srv._note_free_tier_gate(count, gated, None)
        assert (count, exhausted) == (1, True)
        # ...and a real exit is still a distinct issuer afterwards.
        count, exhausted = srv._note_free_tier_gate(count, gated, "9.9.9.9:4145")
        assert (count, exhausted) == (2, False)
    finally:
        srv.PROXY_POOL_ENABLED = prev_enabled
        srv._report_free_tier_block = prev_report


# ── 2. conversation gate memory ────────────────────────────────────

def test_gate_memory_caps_rotation_then_expires():
    srv._gate_rejected.clear()
    assert srv._gate_rotation_budget("ses_a", "m") == srv._FREE_TIER_FAIL_FAST_GATES

    srv._remember_gate_rejection("ses_a")
    assert srv._gate_rejection_is_fresh("ses_a")
    assert srv._gate_rotation_budget("ses_a", "m") == 1
    # Another conversation is untouched.
    assert srv._gate_rotation_budget("ses_b", "m") == srv._FREE_TIER_FAIL_FAST_GATES

    # Expiry drops the entry (a lifted gate is picked up on the next turn).
    srv._gate_rejected["ses_a"] = time.time() - 1
    assert not srv._gate_rejection_is_fresh("ses_a")
    assert "ses_a" not in srv._gate_rejected

    # A missing/blank session id is never remembered or capped.
    srv._remember_gate_rejection(None)
    srv._remember_gate_rejection("")
    assert srv._gate_rejected == {}
    assert srv._gate_rotation_budget(None, "m") == srv._FREE_TIER_FAIL_FAST_GATES


def test_gate_memory_is_bounded():
    srv._gate_rejected.clear()
    for i in range(srv._GATE_REJECTED_MAX + 10):
        srv._remember_gate_rejection(f"ses_{i}")
    assert len(srv._gate_rejected) <= srv._GATE_REJECTED_MAX
    # The newest entry survives the pruning.
    assert srv._gate_rejection_is_fresh(f"ses_{srv._GATE_REJECTED_MAX + 9}")


# ── 3. session persistence + stable prompt_cache_key ───────────────

def test_session_id_survives_restart():
    messages = [{"role": "user", "content": "bonjour"}]

    first = srv.get_session("user", messages)
    srv._save_user_sessions(force=True)

    # Simulate a restart: memory gone, disk reloaded.
    srv._user_sessions.clear()
    srv._load_user_sessions()
    assert srv.get_session("user", messages) == first

    # A growing conversation keeps its id (prefix match on the short history).
    grown = messages + [
        {"role": "assistant", "content": "salut"},
        {"role": "user", "content": "suite"},
    ]
    assert srv.get_session("user", grown) == first


def test_session_file_is_pruned_to_per_user_cap():
    for i in range(srv._MAX_SESSIONS_PER_USER + 5):
        srv.get_session("user", [{"role": "user", "content": f"m{i}"}])
    srv._save_user_sessions(force=True)
    srv._user_sessions.clear()
    srv._load_user_sessions()
    assert len(srv._user_sessions["user"]) <= srv._MAX_SESSIONS_PER_USER


def test_zen_request_sends_stable_prompt_cache_key():
    model = "deepseek-v4.1-flash-free"
    messages = [{"role": "user", "content": "hi"}]

    body, _headers = srv.zen_request(model, messages, True, [], None, "ses_stable123")
    assert body["prompt_cache_key"] == "ses_stable123"

    # A caller-supplied key always wins.
    body, _headers = srv.zen_request(
        model, messages, True, [], None, "ses_stable123",
        sampling={"prompt_cache_key": "caller-key"},
    )
    assert body["prompt_cache_key"] == "caller-key"

    # No session id -> nothing is invented.
    body, _headers = srv.zen_request(model, messages, True, [], None, None)
    assert "prompt_cache_key" not in body
