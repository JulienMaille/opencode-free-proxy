#!/usr/bin/env python3
"""Zen compat-invariant pins (offline, no live probes).

Covers:
  1. max_tokens never renamed to max_completion_tokens.
  2. reasoning_content + tool_calls survive cache-key ops.
  3. prompt_cache_retention stripped (chat + Responses bodies).
  4. keyless omission wins over injected Authorization.
  5. empty discovery never wipes snapshot.

Plus narrow coverage for the Zen-only deltas: dual-key models.dev lookup,
_humanize_name fallback, and the conservative text-only default modalities.

Run with `python -m pytest test_zen_compat.py -q`, or directly with
`python test_zen_compat.py` when pytest is unavailable. No network calls:
discovery paths use injected fake fetches.
"""
import asyncio
import hashlib
import sys
import time

# server.py parses CLI args at import: stub argv first so pytest's own
# argv never leaks into argparse.
sys.argv = ["test_zen_compat"]

import server as srv


# ── fakes (inject, never live) ─────────────────────────────────────

class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, zen_payload, md_payload, md_status=200):
        self._zen = zen_payload
        self._md = md_payload
        self._md_status = md_status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None, timeout=None):
        if "models.dev" in url:
            return _FakeResp(self._md_status, self._md)
        return _FakeResp(200, self._zen)


def _run_discovery(zen_payload, md_payload, md_status=200):
    """Run _fetch_free_models with an injected fake fetch (no network)."""
    orig = srv.httpx.AsyncClient
    srv.httpx.AsyncClient = lambda *a, **k: _FakeClient(zen_payload, md_payload, md_status)
    try:
        asyncio.run(srv._fetch_free_models())
    finally:
        srv.httpx.AsyncClient = orig


def _snapshot():
    return (
        list(srv._models_cache),
        {k: dict(v) for k, v in srv._models_meta.items()},
        srv._models_checked_at,
        set(srv._dead_models),
        set(srv.DEAD_IDS),
    )


def _restore(snap):
    cache, meta, checked_at, dead, dead_ids = snap
    srv._models_cache = cache
    srv._models_meta = meta
    srv._models_checked_at = checked_at
    srv._dead_models = dead
    srv.DEAD_IDS = dead_ids


# ── 1. max_tokens never renamed ────────────────────────────────────

def test_max_tokens_never_renamed():
    body, _ = srv.zen_request("m", [{"role": "user", "content": "hi"}], False, None, None, "s", max_tokens=100)
    assert body.get("max_tokens") == 100
    assert "max_completion_tokens" not in body

    body, _ = srv.zen_request("m", [{"role": "user", "content": "hi"}], False, None, None, "s",
                              max_completion_tokens=200)
    assert body.get("max_completion_tokens") == 200
    assert "max_tokens" not in body

    body, _ = srv.zen_request("m", [{"role": "user", "content": "hi"}], False, None, None, "s",
                              max_tokens=100, max_completion_tokens=200)
    assert body.get("max_tokens") == 100
    assert body.get("max_completion_tokens") == 200

    # Responses wire body maps the budget to max_output_tokens (protocol
    # translation), never to a synthesized max_completion_tokens key.
    # (Budgets >= the 512 floor so the tool-free minimum doesn't interfere.)
    rb = srv._zen_responses_body("m", [{"role": "user", "content": "hi"}], None,
                                 max_tokens=2000, max_completion_tokens=1500)
    assert rb.get("max_output_tokens") == 2000
    assert "max_completion_tokens" not in rb


# ── 2. reasoning_content + tool_calls survive cache-key ops ────────

def test_reasoning_tool_calls_survive_cache_key_ops():
    tc = [{"id": "bash:0", "type": "function",
           "function": {"name": "bash", "arguments": "{}"}}]
    msgs = [{"role": "assistant", "content": "hello",
             "reasoning_content": "think", "tool_calls": tc}]
    out = srv._prepare_upstream_messages("sess-rt", msgs)
    assert out[0].get("reasoning_content") == "think"
    assert out[0].get("tool_calls") == tc
    assert out[0].get("content") == "hello"

    # Cached-thinking re-injection must not clobber tool_calls.
    key = hashlib.sha256(b"hello").hexdigest()
    srv._reasoning_cache.setdefault("sess-rt2", {})[key] = {"text": "cached-think", "at": time.time()}
    try:
        msgs2 = [{"role": "assistant", "content": "hello", "tool_calls": tc}]
        out2 = srv._prepare_upstream_messages("sess-rt2", msgs2)
        assert out2[0].get("reasoning_content") == "cached-think"
        assert out2[0].get("tool_calls") == tc
    finally:
        srv._reasoning_cache.pop("sess-rt2", None)

    # prompt_cache_key rides along additively through sampling into the body.
    sampling = srv._sampling_from({"temperature": 0.5, "prompt_cache_key": "k1"})
    assert sampling.get("prompt_cache_key") == "k1"
    body, _ = srv.zen_request("m", msgs, False, None, None, "s", sampling=sampling)
    assert body.get("prompt_cache_key") == "k1"
    assert body["messages"][0].get("tool_calls") == tc
    assert body["messages"][0].get("reasoning_content") == "think"


# ── 3. prompt_cache_retention stripped ─────────────────────────────

def test_prompt_cache_retention_stripped():
    sampling = srv._sampling_from({"prompt_cache_key": "k1", "prompt_cache_retention": "keep"})
    assert "prompt_cache_retention" not in sampling
    assert sampling.get("prompt_cache_key") == "k1"

    body, _ = srv.zen_request("m", [{"role": "user", "content": "hi"}], False, None, None, "s",
                              sampling={"prompt_cache_key": "k1", "prompt_cache_retention": "keep"})
    assert "prompt_cache_retention" not in body
    assert body.get("prompt_cache_key") == "k1"

    rb = srv._zen_responses_body("m", [{"role": "user", "content": "hi"}], None,
                                 sampling={"prompt_cache_key": "k1", "prompt_cache_retention": "keep"})
    assert "prompt_cache_retention" not in rb
    assert rb.get("prompt_cache_key") == "k1"


# ── 4. keyless omission wins ───────────────────────────────────────

def test_keyless_omission_wins_over_injected_authorization():
    _, headers = srv.zen_request("m", [{"role": "user", "content": "hi"}], False, None, None, "s")
    assert not any(str(k).lower() == "authorization" for k in headers)

    # A pool/hook merge that injects Authorization still loses: the strip
    # runs after the merge (applied-last-wins).
    merged = dict(headers)
    merged["Authorization"] = "Bearer injected"
    merged["X-Relay-Foo"] = "bar"
    stripped = srv._strip_direct_headers(merged)
    assert not any(str(k).lower() == "authorization" for k in stripped)
    assert "X-Relay-Foo" not in stripped
    assert stripped.get("Content-Type") == "application/json"

    # Case-insensitive + non-dict hardening.
    stripped2 = srv._strip_direct_headers({"aUtHoRiZaTiOn": "Bearer x", "Ok": "1"})
    assert not any(str(k).lower() == "authorization" for k in stripped2)
    assert stripped2 == {"Ok": "1"}
    assert srv._strip_direct_headers(None) == {}


# ── 5. empty discovery never wipes snapshot ────────────────────────

def test_empty_discovery_never_wipes_snapshot():
    snap = _snapshot()
    srv._models_cache = ["opencode/keep-free"]
    srv._models_meta = {"opencode/keep-free": {
        "name": "Keep (Free)",
        "limit": dict(srv._DEFAULT_LIMIT),
        "modalities": {"input": ["text"], "output": ["text"]},
        "api": None,
    }}
    srv._models_checked_at = 1234.5
    try:
        # Empty Zen free list.
        _run_discovery({"data": [{"id": "some-nonfree-model"}]}, {"opencode": {"models": {}}})
        assert srv._models_cache == ["opencode/keep-free"]
        assert "opencode/keep-free" in srv._models_meta
        assert srv._models_checked_at == 1234.5

        # Failed Zen fetch (exception) also keeps the snapshot.
        orig = srv.httpx.AsyncClient

        class _BoomClient(_FakeClient):
            async def get(self, url, headers=None, timeout=None):
                raise RuntimeError("boom")

        srv.httpx.AsyncClient = lambda *a, **k: _BoomClient({}, {})
        try:
            asyncio.run(srv._fetch_free_models())
        finally:
            srv.httpx.AsyncClient = orig
        assert srv._models_cache == ["opencode/keep-free"]
        assert srv._models_checked_at == 1234.5
    finally:
        _restore(snap)


# ── Zen-only delta coverage ────────────────────────────────────────

def test_dual_key_models_dev_lookup():
    snap = _snapshot()
    srv._dead_models = set()
    srv.DEAD_IDS = set()
    try:
        zen = {"data": [{"id": "opencode/mimo-v2.5-free"}]}
        md = {"opencode": {"models": {
            # Bare id only (no opencode/ prefix, no -free suffix).
            "mimo-v2.5": {
                "name": "Mimo",
                "limit": {"context": 1000, "output": 100},
                "modalities": {"input": ["text"], "output": ["text"]},
                "provider": {"npm": "x"},
            },
        }}}
        _run_discovery(zen, md)
        assert "opencode/mimo-v2.5-free" in srv._models_cache
        assert srv._models_meta["opencode/mimo-v2.5-free"]["name"] == "Mimo"
    finally:
        _restore(snap)


def test_humanize_name_fallback():
    assert srv._humanize_name("opencode/mimo-v2.5-free") == "Mimo V2.5 (Free)"
    assert srv._humanize_name("big-pickle") == "Big Pickle (Free)"
    snap = _snapshot()
    srv._dead_models = set()
    srv.DEAD_IDS = set()
    try:
        zen = {"data": [{"id": "opencode/no-meta-free"}]}
        _run_discovery(zen, {"opencode": {"models": {}}})
        assert srv._models_meta["opencode/no-meta-free"]["name"] == "No Meta (Free)"
    finally:
        _restore(snap)


def test_conservative_image_gating_default():
    assert srv._DEFAULT_MODALITIES == {"input": ["text"], "output": ["text"]}
    snap = _snapshot()
    srv._dead_models = set()
    srv.DEAD_IDS = set()
    try:
        zen = {"data": [{"id": "opencode/no-meta-free"}]}
        _run_discovery(zen, {"opencode": {"models": {}}}, md_status=500)
        assert srv._models_meta["opencode/no-meta-free"]["modalities"] == {
            "input": ["text"], "output": ["text"]}
    finally:
        _restore(snap)


_TESTS = [
    test_max_tokens_never_renamed,
    test_reasoning_tool_calls_survive_cache_key_ops,
    test_prompt_cache_retention_stripped,
    test_keyless_omission_wins_over_injected_authorization,
    test_empty_discovery_never_wipes_snapshot,
    test_dual_key_models_dev_lookup,
    test_humanize_name_fallback,
    test_conservative_image_gating_default,
]


def main():
    failed = 0
    for fn in _TESTS:
        try:
            fn()
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
        else:
            print(f"PASS {fn.__name__}")
    print(f"{len(_TESTS) - failed}/{len(_TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
