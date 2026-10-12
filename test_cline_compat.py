#!/usr/bin/env python3
"""Cline compat pins (offline, no live probes, no keys).

Covers:
  1. Routing: only the `cline/` prefix routes; id passthrough never strips
     `cline-pass/`; retired IDs (glm-5.2, kimi-k2.7-code, kimi-k2.6,
     deepseek-v4-flash) have no aliases and pass through as-is.
  2. Cap-delay parser variants: "2h 15m", "45m", "20" (bare seconds),
     case-insensitive, plus no-delay -> None.
  3. Alias coverage: every canonical ID resolves from its bare suffix
     (7 free + 12 cline-pass + 5 billing = 24).
  4. Free-limit parser (pi-freeflow port): marker-gated, model-name
     trailing-punct stripped, bare N = hours.
  5. Slot pool (offline, temp pool file): active-first/capped-last
     ordering, stale detection, same-account duplicate refusal, JSON
     roundtrip. Static keys load as key-N slots.
  6. Cap-gate roll (offline, fake upstream): an all-capped pool fails fast
     with the reset hint and zero upstream calls, buffered AND streaming, and
     a capped slot really drops out of the roll (skip by slot name OR token).
  7. Live-429 roll: a slot that 429s gets exactly ONE upstream call per
     request, bookkeeping persists under the canonical model id, and an
     unparseable 429 retires the slot without inventing a reset.

Run with `python -m pytest test_cline_compat.py -q`, or directly with
`python test_cline_compat.py` when pytest is unavailable. No network calls.
"""
import asyncio
import json
import os
import sys
import tempfile
import time

# server.py parses CLI args at import: stub argv first so pytest's own
# argv never leaks into argparse.
sys.argv = ["test_cline_compat"]

import cline_pool as cp
import cline_proxy as cx

import server as srv


def _temp_pool_env():
    """Point the pool + keys files at temp paths (offline, no keys)."""
    d = tempfile.mkdtemp(prefix="cline-compat-")
    pool_path = os.path.join(d, "cline-pool.json")
    keys_path = os.path.join(d, "cline-api-keys.txt")
    open(keys_path, "w").close()
    os.environ["CLINE_POOL_FILE"] = pool_path
    cp.POOL_FILE = cp.resolve_pool_path()
    cp.POOL_BACKUP_FILE = cp.Path(str(cp.POOL_FILE) + ".bak")
    cp.KEYS_FILE = cp.Path(keys_path)
    cp.reset_pool_cache_for_test()
    # Mark the live free rotation FRESH so the chat-path refresh added in
    # server.py does not fire on every offline request (it would try to GET
    # through the fake upstream). Tests that exercise the refresh reset this.
    srv._CLINE_FREE_IDS = ["cline-free/test-only"]
    srv._CLINE_FREE_AT = time.time()
    srv._CLINE_FREE_FETCHING = False
    return d


# ── 1. routing / passthrough ─────────────────────────────────────

def test_routing_prefix_only():
    assert cx.is_cline_model("cline/cline-pass/glm-5.3")
    assert cx.is_cline_model("CLINE/kimi-k3")
    assert cx.is_cline_model("  cline/openai/gpt-4o  ")
    assert not cx.is_cline_model("amd/DeepSeek-V4-Flash")
    assert not cx.is_cline_model("cline-pass/glm-5.3")  # no prefix -> not routed
    assert not cx.is_cline_model(None)
    assert not cx.is_cline_model("")


def test_passthrough_never_strips_cline_pass():
    # Model IDs pass through UNCHANGED (only the routing prefix is stripped).
    assert cx.cline_model_id("cline/cline-pass/glm-5.3") == "cline-pass/glm-5.3"
    assert cx.cline_model_id("cline/cline-pass/kimi-k3") == "cline-pass/kimi-k3"
    assert cx.cline_model_id("cline/anthropic/claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"
    assert cx.cline_model_id("cline/cline-free/kimi-k3") == "cline-free/kimi-k3"


def test_retired_ids_have_no_aliases():
    # Retired: NO aliases — pass through so Cline errors on them itself.
    assert cx.cline_model_id("cline/glm-5.2") == "glm-5.2"
    assert cx.cline_model_id("cline/kimi-k2.7-code") == "kimi-k2.7-code"
    assert cx.cline_model_id("cline/kimi-k2.6") == "kimi-k2.6"
    assert cx.cline_model_id("cline/deepseek-v4-flash") == "deepseek-v4-flash"


def test_friendly_slugs_resolve():
    assert cx.cline_model_id("cline/cline-free/kimi-k3") == "cline-free/kimi-k3"
    assert cx.cline_model_id("cline/cline-free/deepseek-v4.1-flash") == "cline-free/deepseek-v4.1-flash"
    assert cx.cline_model_id("cline/cline-free/muse-spark-1.3-contributor") == "cline-free/muse-spark-1.3-contributor"
    assert cx.cline_model_id("cline/z-ai/glm-5.3-flash") == "z-ai/glm-5.3-flash"
    assert cx.cline_model_id("cline/cline-free/solar-pro4") == "cline-free/solar-pro4"
    assert cx.cline_model_id("cline/solar-pro4") == "cline-free/solar-pro4"
    assert cx.cline_model_id("cline/deepseek-v4.1-flash") == "cline-free/deepseek-v4.1-flash"
    # Bare suffixes shared by FREE + cline-pass serve FREE; cline-pass stays
    # reachable via its exact verbatim ID.
    assert cx.cline_model_id("cline/kimi-k3") == "cline-free/kimi-k3"
    assert cx.cline_model_id("cline/glm-5.3-flash") == "z-ai/glm-5.3-flash"
    assert cx.cline_model_id("cline/muse-spark-1.3-contributor") == "cline-free/muse-spark-1.3-contributor"
    assert cx.cline_model_id("cline/cline-pass/kimi-k3") == "cline-pass/kimi-k3"
    assert cx.cline_model_id("cline/cline-pass/glm-5.3-flash") == "cline-pass/glm-5.3-flash"
    assert cx.cline_model_id("cline/cline-pass/deepseek-v4.1-flash") == "cline-pass/deepseek-v4.1-flash"
    assert cx.cline_model_id("cline/glm-5.3") == "cline-pass/glm-5.3"
    assert cx.cline_model_id("cline/GLM-5.3-FLASH") == "z-ai/glm-5.3-flash"
    assert cx.cline_model_id("cline/deepseek-v4-pro") == "cline-pass/deepseek-v4-pro"
    assert cx.cline_model_id("cline/mimo-v2.5-pro") == "cline-pass/mimo-v2.5-pro"
    assert cx.cline_model_id("cline/minimax-m3") == "cline-pass/minimax-m3"
    assert cx.cline_model_id("cline/qwen3.8-max") == "cline-pass/qwen3.8-max"
    assert cx.cline_model_id("cline/qwen3.7-max") == "cline-pass/qwen3.7-max"
    assert cx.cline_model_id("cline/qwen3.7-plus") == "cline-pass/qwen3.7-plus"
    assert cx.cline_model_id("cline/gpt-4o") == "openai/gpt-4o"
    assert cx.cline_model_id("cline/deepseek-chat") == "deepseek/deepseek-chat"
    assert cx.cline_model_id("cline/gemini-2.5-pro") == "google/gemini-2.5-pro"
    assert cx.cline_model_id("cline/claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"
    assert cx.cline_model_id("cline/minimax-m2.5") == "minimax/minimax-m2.5"


# ── 2. cap-delay parser variants ─────────────────────────────────

def test_cap_delay_hours_minutes():
    assert cp.parse_cap_delay_seconds("free limit reached on model cline-pass/kimi-k3, try again in 2h 15m") == 2 * 3600 + 15 * 60


def test_cap_delay_minutes_only():
    assert cp.parse_cap_delay_seconds("Free limit reached, try again in 45m") == 45 * 60


def test_cap_delay_bare_number_is_seconds():
    assert cp.parse_cap_delay_seconds("try again in 20") == 20


def test_cap_delay_case_insensitive_and_words():
    assert cp.parse_cap_delay_seconds("TRY AGAIN IN 1H 5M") == 3600 + 300
    assert cp.parse_cap_delay_seconds("try again in 45 minutes") == 45 * 60
    assert cp.parse_cap_delay_seconds("try again in 30s") == 30


def test_cap_delay_absent_is_none():
    assert cp.parse_cap_delay_seconds("Rate limit exceeded") is None
    assert cp.parse_cap_delay_seconds("") is None
    assert cp.parse_cap_delay_seconds(None) is None


# ── 3. canonical list coverage ───────────────────────────────────

def test_canonical_list_has_17_ids():
    ids = cx.cline_models()
    assert len(ids) == 24
    for mid in (
        "cline-free/deepseek-v4.1-flash",
        "cline-free/muse-spark-1.3-contributor",
        "cline-free/mimo-v2.6-flash",
        "stealth/space-bunny-alpha",
        "z-ai/glm-5.3-flash",
        "cline-free/kimi-k3",
        "cline-free/solar-pro4",
        "cline-pass/glm-5.3",
        "cline-pass/glm-5.3-flash",
        "cline-pass/kimi-k3",
        "cline-pass/deepseek-v4-pro",
        "cline-pass/deepseek-v4.1-flash",
        "cline-pass/mimo-v2.5",
        "cline-pass/mimo-v2.5-pro",
        "cline-pass/minimax-m3",
        "cline-pass/muse-spark-1.3-contributor",
        "cline-pass/qwen3.8-max",
        "cline-pass/qwen3.7-max",
        "cline-pass/qwen3.7-plus",
        "anthropic/claude-sonnet-4-6",
        "google/gemini-2.5-pro",
        "deepseek/deepseek-chat",
        "openai/gpt-4o",
        "minimax/minimax-m2.5",
    ):
        assert mid in ids


# ── 4. free-limit parser (pi-freeflow port) ──────────────────────

def test_free_limit_marker_gated():
    body = ("Error 429: Daily free limit reached on model deepseek/deepseek-v4.1-flash. "
            "Try again in 20h 4m")
    out = cp.parse_cline_free_limit(body)
    assert out is not None
    assert out["modelId"] == "deepseek/deepseek-v4.1-flash"
    assert out["resetAt"] - int(time.time() * 1000) > 20 * 3600_000
    assert cp.parse_cline_free_limit("Rate limit exceeded") is None
    assert cp.parse_cline_free_limit("") is None
    assert cp.parse_cline_free_limit(None) is None


def test_free_limit_model_trailing_punct_stripped():
    out = cp.parse_cline_free_limit("free limit reached on model kimi-k3. Try again in 45m")
    assert out is not None
    assert out["modelId"] == "kimi-k3"
    out2 = cp.parse_cline_free_limit("free limit reached on model solar-pro4, try again in 45m")
    assert out2 is not None
    assert out2["modelId"] == "solar-pro4"


def test_free_limit_bare_number_is_hours():
    out = cp.parse_cline_free_limit("free limit reached on model kimi-k3, try again in 20")
    assert out is not None
    assert out["resetAt"] - int(time.time() * 1000) > 19 * 3600_000
    out2 = cp.parse_cline_free_limit("Free limit reached on model x, try again in 45m")
    assert out2 is not None
    assert out2["resetAt"] - int(time.time() * 1000) <= 46 * 60_000


# ── 5. slot pool (offline, temp pool file) ───────────────────────

def _seed_slots(*tokens):
    _temp_pool_env()
    pool = {"accounts": []}
    for i, tok in enumerate(tokens):
        pool["accounts"].append({
            "slot": f"s{i}", "token": tok,
            "addedAt": "2026-01-01T00:00:00Z",
        })
    if pool["accounts"]:
        pool["activeSlot"] = pool["accounts"][0]["slot"]
    cp.save_pool(pool)
    cp.reset_pool_cache_for_test()
    return pool


def test_slot_ordering_active_first_capped_last():
    _seed_slots("workos:AAA", "workos:BBB", "workos:CCC")
    assert cp.roll_order() == ["s0", "s1", "s2"]
    future = int(time.time() * 1000) + 3600_000
    cp.record_limit_on_disk("s0", "cline-free/kimi-k3", future)
    cp.reset_pool_cache_for_test()
    assert cp.roll_order("cline-free/kimi-k3") == ["s1", "s2", "s0"]
    # No model filter: file order again.
    assert cp.roll_order() == ["s0", "s1", "s2"]


def test_stale_detection():
    now_ms = int(time.time() * 1000)
    fresh = {"slot": "a", "token": "workos:X", "expiresAt": now_ms + 3600_000}
    stale = {"slot": "b", "token": "workos:Y", "expiresAt": now_ms - 1000}
    legacy = {"slot": "c", "token": "clp_zzz"}  # key slots never stale
    assert cp.is_token_stale(fresh) is False
    assert cp.is_token_stale(stale) is True
    assert cp.is_token_stale(legacy) is False
    # 60s skew: expiry 30s out counts as stale.
    assert cp.is_token_stale({"slot": "d", "token": "workos:Z",
                              "expiresAt": now_ms + 30_000}) is True


def test_same_account_duplicate_refusal():
    _temp_pool_env()
    cp.add_account("main", "workos:TOKEN1",
                   {"accountId": "user-1", "email": "a@x.com", "refreshToken": "r1"})
    try:
        cp.add_account("dup", "workos:TOKEN2",
                       {"accountId": "user-1", "email": "other@y.com"})
    except ValueError as e:
        assert "already saved as [main]" in str(e)
    else:
        raise AssertionError("duplicate accountId accepted")
    # Email fallback also refuses.
    try:
        cp.add_account("dup2", "workos:TOKEN3", {"email": "A@X.com"})
    except ValueError as e:
        assert "already saved as [main]" in str(e)
    else:
        raise AssertionError("duplicate email accepted")
    # Re-logging into the SAME slot is allowed (not a duplicate).
    cp.add_account("main", "workos:TOKEN9", {"accountId": "user-1"})
    assert cp.load_pool()["accounts"][0]["token"] == "workos:TOKEN9"


def test_pool_roundtrip():
    _temp_pool_env()
    assert cp.load_pool() == {"accounts": []}
    cp.add_account("default", "workos:AAA",
                   {"refreshToken": "r", "expiresAt": int(time.time() * 1000) + 99999,
                    "accountId": "u1", "email": "e@x.com"})
    cp.reset_pool_cache_for_test()
    pool = cp.load_pool()
    assert [a["slot"] for a in pool["accounts"]] == ["default"]
    assert pool["activeSlot"] == "default"
    assert pool["accounts"][0]["email"] == "e@x.com"
    # .bak recovery: corrupt main, backup holds... valid empty stays empty.
    cp.save_pool({"accounts": []})
    cp.reset_pool_cache_for_test()
    assert cp.load_pool() == {"accounts": []}
    # Static keys merge as key-N slots, deduped by token.
    with open(cp.KEYS_FILE, "w", encoding="utf-8") as f:
        f.write("# comment\n\nsk_live123\n")
    cp.reset_pool_cache_for_test()
    names = [a["slot"] for a in cp.all_accounts()]
    assert "key-1" in names
    # sk_ acceptance + clp_ + jwt + workos: forms all load.
    assert cp._accepts_token("sk_abc") and cp._accepts_token("clp_xyz")
    assert cp._accepts_token("eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjF9.sig")
    assert cp._accepts_token("workos:eyJhYmM.hmac.sig")
    assert not cp._accepts_token("") and not cp._accepts_token("random")


# ── 6. non-streaming bridge: upstream forced SSE + offline aggregation ─

_FAKE_SSE = (
    b'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1720000000,'
    b'"model":"cline-free/kimi-k3","choices":[{"index":0,"delta":{"role":"assistant",'
    b'"content":"Hello"},"finish_reason":null}]}\n\n'
    b'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1720000000,'
    b'"model":"cline-free/kimi-k3","choices":[{"index":0,"delta":{"content":" world"},'
    b'"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
    b'data: [DONE]\n\n'
)


def test_cline_upstream_forces_stream_true():
    body = {"model": "cline-free/kimi-k3", "messages": [], "stream": False}
    forced = srv._cline_upstream_stream_body(body)
    assert forced["stream"] is True
    assert body["stream"] is False  # caller body untouched
    assert srv._cline_upstream_stream_body({"stream": True})["stream"] is True


def test_cline_request_body_still_passes_client_stream():
    assert srv.cline_request_body("m", [], False, None, None)["stream"] is False
    assert srv.cline_request_body("m", [], True, None, None)["stream"] is True


def test_cline_sse_to_completion_aggregates():
    out = srv._cline_sse_to_completion(_FAKE_SSE, "cline-free/kimi-k3")
    assert out["object"] == "chat.completion"
    assert out["model"] == "cline-free/kimi-k3"
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "Hello world"}
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["prompt_tokens"] == 3
    assert out["usage"]["completion_tokens"] == 2


def test_cline_sse_to_completion_reasoning_and_tools():
    raw = (
        b'data: {"choices":[{"index":0,"delta":{"content":"Hi","reasoning_content":"think"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        b'"function":{"name":"get","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\n'
        b'data: [DONE]\n\n'
    )
    out = srv._cline_sse_to_completion(raw, "m")
    assert out["choices"][0]["message"]["content"] == "Hi"
    assert out["choices"][0]["message"]["reasoning_content"] == "think"
    assert out["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_cline_sse_to_completion_error_marker_and_empty():
    raw = b'data: {"error": {"message": "boom", "type": "upstream_error"}}\n\ndata: [DONE]\n\n'
    out = srv._cline_sse_to_completion(raw, "m")
    assert out == {"error": {"message": "boom", "type": "upstream_error"}}
    assert srv._cline_sse_to_completion(b"", "m") is None
    assert srv._cline_sse_to_completion(b"not sse at all", "m") is None


def test_upstream_error_preserves_string_message():
    # {"error": "<plain string>"} must surface the string, not "HTTP <code>".
    assert srv._upstream_error({"error": "boom"}) == {"message": "boom"}
    assert srv._upstream_error({"error": {"message": "m"}}) == {"message": "m"}
    assert srv._upstream_error({"error": None}) == {}
    assert srv._upstream_error({"error": ["x"]}) == {}
    assert srv._upstream_error({"error": "  "}) == {}
    assert srv._upstream_error("nope") == {}
    assert srv._upstream_error(None) == {}


def test_as_dict_coercion():
    assert srv._as_dict({"a": 1}) == {"a": 1}
    assert srv._as_dict("x") == {}
    assert srv._as_dict(["x"]) == {}
    assert srv._as_dict(None) == {}
    assert srv._as_dict(5) == {}


def test_cline_sse_to_completion_string_error_marker():
    # Mid-stream plain-string error must become an error marker, not vanish.
    raw = b'data: {"error": "plain"}\n\ndata: [DONE]\n\n'
    out = srv._cline_sse_to_completion(raw, "m")
    assert isinstance(out, dict) and "error" in out


def test_cline_sse_to_completion_nondict_chunks_ignored():
    # Non-dict SSE chunks must not crash the aggregator.
    raw = b'data: "hi"\n\ndata: [1]\n\ndata: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    out = srv._cline_sse_to_completion(raw, "m")
    assert out["choices"][0]["message"]["content"] == "ok"


def test_first_chunk_error_string_body():
    # Must not crash and must surface a sensible message.
    msg = srv._first_chunk_error('data: {"error": "plain"}')
    assert msg is not None and "plain" in msg[0]
    assert srv._first_chunk_error('data: {"choices":[]}') is None


def test_zen_stream_nondict_chunk_no_crash():
    # _zen_stream_to_completion path: non-dict chunks + string errors are skipped/surfaced, never crash.
    assert srv._upstream_error({"error": "x"}) == {"message": "x"}


# ── 7. recommended-models free rotation (offline) ─────────────────

def test_recommended_url_constant():
    assert cx.CLINE_RECOMMENDED_URL == "https://api.cline.bot/api/v1/ai/cline/recommended-models"


def test_recommended_free_extraction_verbatim():
    payload = {
        "recommended": [{"id": "cline-pass/glm-5.3"}],
        "free": [
            {"id": "cline-free/new-model-x"},
            {"id": "z-ai/glm-5.3-flash"},
            {"id": "  cline-free/spaced  "},
        ],
        "clinePass": [{"id": "cline-pass/kimi-k3"}],
    }
    ids = cx.parse_recommended_free(payload)
    # Verbatim IDs, whitespace trimmed, sorted distinct; recommended/
    # clinePass entries never leak in.
    assert ids == ["cline-free/new-model-x", "cline-free/spaced", "z-ai/glm-5.3-flash"]


def test_recommended_free_empty_malformed_fallback():
    assert cx.parse_recommended_free({}) == []
    assert cx.parse_recommended_free({"free": []}) == []
    assert cx.parse_recommended_free({"free": "nope"}) == []
    assert cx.parse_recommended_free({"free": [{"noid": 1}, {"id": ""}, {"id": None}, "str", None]}) == []
    assert cx.parse_recommended_free(None) == []
    assert cx.parse_recommended_free("str") == []
    assert cx.parse_recommended_free([]) == []
    assert cx.parse_recommended_free({"free": [{"id": "a"}, {"id": "a"}, {"id": "b"}]}) == ["a", "b"]


def test_dynamic_free_merge_never_drop():
    try:
        # Dynamic-wins: when a live rotation is cached, ONLY those IDs are
        # free — rotated-out/billed IDs drop out even if statically listed.
        cx.set_dynamic_free_ids(["cline-free/new-model-x", "z-ai/glm-5.3-flash"])
        merged = cx.cline_free_models()
        assert merged == ["cline-free/new-model-x", "z-ai/glm-5.3-flash"]
        assert "cline-free/new-model-x" in merged
        for mid in (
            "cline-free/deepseek-v4.1-flash",
            "cline-free/muse-spark-1.3-contributor",
            "cline-free/kimi-k3",
            "cline-free/solar-pro4",
        ):
            assert mid not in merged
        assert not cx.is_free_id("cline-free/kimi-k3")
        assert not cx.is_free_id("cline-free/solar-pro4")
        assert cx.is_free_id("cline-free/new-model-x")
        # Empty dynamic falls back to the static canonical 7.
        cx.set_dynamic_free_ids([])
        merged2 = cx.cline_free_models()
        for mid in (
            "cline-free/deepseek-v4.1-flash",
            "cline-free/muse-spark-1.3-contributor",
            "z-ai/glm-5.3-flash",
            "cline-free/kimi-k3",
            "cline-free/solar-pro4",
        ):
            assert mid in merged2
        # Retired IDs still route verbatim (passthrough), advertised-ness
        # is a listing concern — routing never rewrites.
        assert cx.cline_model_id("cline/cline-free/kimi-k3") == "cline-free/kimi-k3"
        assert cx.cline_model_id("cline/cline-free/solar-pro4") == "cline-free/solar-pro4"
    finally:
        cx.set_dynamic_free_ids([])


# ── 8. cap-gate roll: never spins, never calls upstream (offline) ──
#
# Regression pins for the 2026-10 incident: ONE slot named "default", capped on
# a single free model, made a streaming request log "Slot default capped ...
# skipping" 24k+ times. The cap-gate `continue` never advanced the loop bound
# AND stored a bearer token where the pool matches slot names, so the same slot
# was re-selected forever — on a path with no `await`, which pegged a core at
# 100%, grew RSS to 7.9 GB and stopped the whole event loop (/v1/models dead).

_CAPPED_MODEL = "cline-free/deepseek-v4.1-flash"

_OK_SSE_LINES = (
    'data: {"id":"chatcmpl-ok","object":"chat.completion.chunk","created":1720000000,'
    '"model":"cline-free/deepseek-v4.1-flash","choices":[{"index":0,'
    '"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-ok","object":"chat.completion.chunk","created":1720000000,'
    '"model":"cline-free/deepseek-v4.1-flash","choices":[{"index":0,'
    '"delta":{"content":" world"},"finish_reason":"stop"}]}',
    "data: [DONE]",
)
_OK_SSE_BODY = ("\n\n".join(_OK_SSE_LINES) + "\n\n").encode()


def _seed_named_slots(*names):
    """Seed slots under real names (the incident pool had exactly one: default)."""
    _temp_pool_env()
    accounts = [{"slot": name, "token": f"workos:{name.upper()}",
                 "addedAt": "2026-01-01T00:00:00Z"} for name in names]
    cp.save_pool({"accounts": accounts})
    cp.reset_pool_cache_for_test()
    return [a["token"] for a in accounts]


def _cap_slot(slot, model=_CAPPED_MODEL, hours=24):
    cp.record_limit_on_disk(slot, model, int(time.time() * 1000) + hours * 3600_000)
    cp.reset_pool_cache_for_test()


class _FakeResp:
    """httpx.Response stand-in (no sockets, no usage keys written)."""

    def __init__(self, status=200, body=b"", lines=()):
        self.status_code = status
        self._body = body
        self._lines = list(lines)

    async def aread(self):
        return self._body

    async def aclose(self):
        pass

    def aiter_lines(self):
        async def gen():
            for line in self._lines:
                yield line
        return gen()


class _FakeUpstream:
    """`_cline_client` stand-in that records every upstream attempt.

    ``calls`` = chat completions (path, Authorization), ``gets`` = the
    recommended-models rotation fetch. ``on_send`` runs just before the
    response is handed back (lets a test change pool state mid-request).
    """

    def __init__(self, body=b"", lines=(), status=200, recommended_body=b"", on_send=None):
        self.calls = []  # (path, Authorization header) per attempt
        self.gets = []   # rotation fetches
        self._body = body
        self._lines = lines
        self._status = status
        self._recommended_body = recommended_body
        self._on_send = on_send

    def build_request(self, method, path, json=None, headers=None):
        return (method, path, json, headers or {})

    def _fire(self):
        if self._on_send:
            self._on_send()

    async def post(self, path, json=None, headers=None):
        self.calls.append((path, (headers or {}).get("Authorization")))
        self._fire()
        return _FakeResp(self._status, self._body)

    async def send(self, request, stream=False):
        _method, path, _body, headers = request
        self.calls.append((path, headers.get("Authorization")))
        self._fire()
        return _FakeResp(self._status, self._body, self._lines)

    async def get(self, path, headers=None):
        self.gets.append(path)
        # Real I/O yields the loop; without this the fake would let every
        # coroutine run to completion without ever interleaving, so a
        # stampede could not happen even with no guard at all.
        await asyncio.sleep(0)
        self._fire()
        return _FakeResp(200, self._recommended_body)

    async def aclose(self):
        pass


class _patched:
    """Attribute monkeypatch (context manager, no pytest dependency)."""

    def __init__(self, target, attr, value):
        self._target, self._attr, self._value = target, attr, value
        self._saved = None

    def __enter__(self):
        self._saved = getattr(self._target, self._attr)
        setattr(self._target, self._attr, self._value)
        return self

    def __exit__(self, *exc):
        setattr(self._target, self._attr, self._saved)
        return False


class _roll_tripwire:
    """Bound the roll so a spin FAILS instead of hanging the suite.

    A wall-clock timeout is useless on the skip path: it never awaits, so it
    blocks the very event loop the timeout callback runs on. Counting
    ``select()`` calls (one per roll iteration) and raising past the ceiling
    unwinds the spin and surfaces it as a loud assertion. The cap-gate log
    lines are counted too: the gate must stay one line per slot, never one
    per iteration.
    """

    def __init__(self, max_selects=16, max_cap_lines=2):
        self.max_selects = max_selects
        self.max_cap_lines = max_cap_lines
        self.selects = 0
        self.cap_skips = []
        self._pool = srv.cline_keys
        self._had_own_select = False
        self._saved_select = None
        self._saved_log = None

    def __enter__(self):
        self._had_own_select = "select" in self._pool.__dict__
        self._saved_select = self._pool.__dict__.get("select")
        real_select = self._pool.select
        self._saved_log = srv._log

        def counting_select(model=None, skip=None):
            self.selects += 1
            if self.selects > self.max_selects:
                raise AssertionError(
                    f"roll called select() {self.selects}x (max {self.max_selects}): "
                    "the cap gate is re-selecting the same capped slot instead of skipping it")
            return real_select(model, skip)

        def counting_log(*a):
            text = str(a[0]) if a else ""
            if "capped on" in text and "skipping" in text:
                self.cap_skips.append(text)
                if len(self.cap_skips) > self.max_cap_lines:
                    raise AssertionError(
                        f"cap gate logged {len(self.cap_skips)}x "
                        f"(max {self.max_cap_lines}) — log spam")
            return self._saved_log(*a)

        self._pool.select = counting_select
        srv._log = counting_log
        return self

    def __exit__(self, *exc):
        if self._had_own_select:
            self._pool.select = self._saved_select
        else:
            del self._pool.select
        srv._log = self._saved_log
        return False


def test_select_skip_accepts_token_and_slot_name():
    # A capped slot must actually drop out of the roll. The skip set is built
    # from bearer tokens by the roll loops and from slot names by the routing
    # preference, so BOTH forms have to filter.
    toks = _seed_named_slots("default", "backup")
    _cap_slot("default")
    assert cp.pool.select(_CAPPED_MODEL) == toks[1]           # open slot first
    assert cp.pool.select(_CAPPED_MODEL, skip={toks[0]}) == toks[1]   # token form
    assert cp.pool.select(_CAPPED_MODEL, skip={"default"}) == toks[1]  # slot form
    # Sole slot capped: either form exhausts the roll (no infinite re-pick).
    _seed_named_slots("default")
    _cap_slot("default")
    assert cp.pool.select(_CAPPED_MODEL, skip={"default"}) is None
    assert cp.pool.select(_CAPPED_MODEL, skip={toks[0]}) is None
    assert cp.pool.is_capped_for("default", _CAPPED_MODEL) is True
    assert cp.pool.cap_reset_ms_for("default", _CAPPED_MODEL) > time.time() * 1000
    assert cp.pool.cap_reset_ms_for("default", "other/model") is None


def test_all_capped_pool_fails_fast_with_reset_hint():
    # Incident, buffered sibling: must return a prompt 429 carrying the reset
    # hint — never "all Cline logins exhausted", never an upstream call.
    _seed_named_slots("default")
    _cap_slot("default")
    up = _FakeUpstream(body=b"must never be requested")
    tripwire = _roll_tripwire()
    started = time.time()
    with _patched(srv, "_cline_client", lambda streaming=False: up), tripwire:
        resp = asyncio.run(asyncio.wait_for(
            srv.cline_request_with_retry({"model": _CAPPED_MODEL, "messages": []},
                                         "u", _CAPPED_MODEL),
            timeout=15))
    assert time.time() - started < 5.0, "all-capped roll did not fail fast"
    # One selection per slot + the final exhausted one, no spin.
    assert tripwire.selects <= 4, f"{tripwire.selects} selections for one slot"
    assert up.calls == [], f"capped pool made {len(up.calls)} upstream call(s)"
    assert resp.status_code == 429
    err = json.loads(resp.body)["error"]
    assert err["type"] == "rate_limit_error"
    assert err["code"] == "rate_limit_exceeded"
    assert "daily free cap" in err["message"]
    assert "try again in" in err["message"]      # nearest reset hint survived
    assert "exhausted" not in err["message"]     # not the generic fallback


def test_stream_all_capped_pool_does_not_spin():
    # The wedge itself: one capped slot used to loop ~24k times with no await.
    _seed_named_slots("default")
    _cap_slot("default")
    up = _FakeUpstream(body=b"must never be requested", lines=_OK_SSE_LINES)
    tripwire = _roll_tripwire()

    async def drive():
        chunks = []
        async for chunk in srv.cline_stream_with_retry(
                {"model": _CAPPED_MODEL, "messages": []}, "u", _CAPPED_MODEL):
            chunks.append(chunk)
            if len(chunks) > 4:
                raise AssertionError("stream kept yielding past its terminal error")
        return chunks

    started = time.time()
    with _patched(srv, "_cline_client", lambda streaming=False: up), tripwire:
        chunks = asyncio.run(asyncio.wait_for(drive(), timeout=15))
    assert time.time() - started < 5.0, "streaming roll did not fail fast"
    # Bounded iterations AND one cap-gate log line per slot, never per iteration.
    assert tripwire.selects <= 4, f"{tripwire.selects} selections for one slot"
    assert len(tripwire.cap_skips) <= 1, f"{len(tripwire.cap_skips)} cap-gate log lines"
    assert up.calls == [], f"capped pool made {len(up.calls)} upstream call(s)"
    body = "".join(chunks)
    assert '"type": "rate_limit_error"' in body
    assert '"code": "rate_limit_exceeded"' in body
    assert "daily free cap" in body and "try again in" in body
    assert '"choices"' not in body          # no fake assistant turn


def test_capped_plus_open_slot_serves_the_open_one():
    # Mixed pool: the capped login is skipped and the open one serves.
    toks = _seed_named_slots("capped-one", "open-one")
    _cap_slot("capped-one")
    up = _FakeUpstream(body=_OK_SSE_BODY)
    tripwire = _roll_tripwire()
    with _patched(srv, "_cline_client", lambda streaming=False: up), tripwire:
        data = asyncio.run(asyncio.wait_for(
            srv.cline_request_with_retry({"model": _CAPPED_MODEL, "messages": []},
                                         "u", _CAPPED_MODEL),
            timeout=15))
    assert up.calls == [("/chat/completions", f"Bearer {toks[1]}")]
    assert tripwire.selects <= 3, f"{tripwire.selects} selections for a 2-slot pool"
    assert data["choices"][0]["message"]["content"] == "Hello world"


def test_stream_capped_plus_open_slot_serves_the_open_one():
    toks = _seed_named_slots("capped-one", "open-one")
    _cap_slot("capped-one")
    up = _FakeUpstream(body=_OK_SSE_BODY, lines=_OK_SSE_LINES)
    tripwire = _roll_tripwire()

    async def drive():
        return [chunk async for chunk in srv.cline_stream_with_retry(
            {"model": _CAPPED_MODEL, "messages": []}, "u", _CAPPED_MODEL)]

    with _patched(srv, "_cline_client", lambda streaming=False: up), tripwire:
        chunks = asyncio.run(asyncio.wait_for(drive(), timeout=15))
    assert up.calls == [("/chat/completions", f"Bearer {toks[1]}")]
    assert tripwire.selects <= 3, f"{tripwire.selects} selections for a 2-slot pool"
    body = "".join(chunks)
    assert '"content":"Hello"' in body and '"content":" world"' in body
    assert "data: [DONE]" in body
    assert '"error"' not in body


# ── 9. live-429 roll: one upstream call per slot (offline) ──────────
#
# Second layer of the same defect: the pre-attempt gate fires once, then the
# slot 429'd LIVE and was handed straight back by select() — 6 calls / 40s
# buffered, 10+ calls / >90s streaming, every one on the same bearer.

_429_BODY = (
    b'{"error":{"message":"Error 429: Daily free limit reached on model '
    b'deepseek/deepseek-v4.1-flash. Try again in 22h 52m",'
    b'"type":"rate_limit_error"}}'
)


def _add_static_key(token="sk_live_keyslot"):
    """Add a static-key slot (key-N), the other half of the pool."""
    with open(cp.KEYS_FILE, "w", encoding="utf-8") as f:
        f.write(token + "\n")
    cp.reset_pool_cache_for_test()
    return token


def _calls_per_key(up):
    """{Authorization header: upstream attempts} — one entry per slot."""
    per = {}
    for _path, auth in up.calls:
        per[auth] = per.get(auth, 0) + 1
    return per


def _drive_buffered(model, timeout=15):
    return asyncio.run(asyncio.wait_for(
        srv.cline_request_with_retry({"model": model, "messages": []}, "u", model),
        timeout=timeout))


def _drive_stream(model, timeout=15):
    async def drive():
        return [chunk async for chunk in srv.cline_stream_with_retry(
            {"model": model, "messages": []}, "u", model)]
    return asyncio.run(asyncio.wait_for(drive(), timeout=timeout))


def test_report_capped_normalises_the_routing_prefix():
    # Persistence-boundary pin (cause 2, second layer): even a caller that hands
    # the pool the routing-prefixed id must not write a limits entry the cap
    # gate can never match.
    toks = _seed_named_slots("default")
    prefixed = "cline/" + _CAPPED_MODEL
    assert cp.cap_model_key(prefixed) == _CAPPED_MODEL
    cp.pool.report_capped(toks[0], prefixed, _429_BODY.decode())
    limits = cp.load_pool().get("limits") or {}
    assert list(limits.get("default") or {}) == [_CAPPED_MODEL], limits
    # The gate accepts either form and still finds the cap.
    assert cp.pool.cap_reset_ms_for("default", _CAPPED_MODEL) is not None
    assert cp.pool.cap_reset_ms_for("default", prefixed) is not None
    assert cp.pool.is_capped_for("default", prefixed) is True
    assert cp.pool.nearest_reset_hint(prefixed)
    assert cp.roll_order(prefixed)[-1] == "default"   # capped-last ordering
    # A success under either form clears it.
    cp.pool.report_slot_success(toks[0], prefixed)
    assert not (cp.load_pool().get("limits") or {})
    assert cp.pool.cap_reset_ms_for("default", _CAPPED_MODEL) is None


def test_429_slot_is_not_re_selected_within_request():
    # Live 429 on an open slot: retire it for this request instead of rolling
    # the same bearer again (was 2 calls/slot on the buffered path).
    toks = _seed_named_slots("alpha", "beta")
    up = _FakeUpstream(body=_429_BODY, status=429)
    tripwire = _roll_tripwire()
    started = time.time()
    with _patched(srv, "_cline_client", lambda streaming=False: up), tripwire:
        resp = _drive_buffered(_CAPPED_MODEL)
    assert time.time() - started < 5.0, f"roll took {time.time() - started:.1f}s"
    assert _calls_per_key(up) == {f"Bearer {t}": 1 for t in toks}
    assert tripwire.cap_skips == []     # nothing was pre-capped; this was live
    assert resp.status_code == 429
    err = json.loads(resp.body)["error"]
    assert err["type"] == "rate_limit_error"
    assert err["code"] == "rate_limit_exceeded"
    assert "daily free cap" in err["message"]
    assert "try again in" in err["message"]   # parsed resetAt -> real hint


def test_all_slots_429_including_key_slot_fails_fast():
    # Login slot + static key slot, both capped live: one call each, fast.
    toks = _seed_named_slots("alpha")
    key = _add_static_key()
    up = _FakeUpstream(body=_429_BODY, status=429)
    started = time.time()
    with _patched(srv, "_cline_client", lambda streaming=False: up), _roll_tripwire():
        resp = _drive_buffered(_CAPPED_MODEL)
    assert time.time() - started < 5.0, f"roll took {time.time() - started:.1f}s"
    assert _calls_per_key(up) == {f"Bearer {toks[0]}": 1, f"Bearer {key}": 1}
    err = json.loads(resp.body)["error"]
    assert "All 2 Cline login(s) on this model's daily free cap" in err["message"]
    assert "try again in" in err["message"]


def test_parsed_429_persists_under_verbatim_id():
    # Cause 2 pin: bookkeeping must land on the CANONICAL id, because that is
    # what the pre-attempt gate looks up. Caller hands in the routing-prefixed
    # id on purpose: a prefixed limits entry would be invisible to the gate and
    # the second request would 429 upstream again.
    _seed_named_slots("default")
    prefixed = "cline/" + _CAPPED_MODEL
    up1 = _FakeUpstream(body=_429_BODY, status=429)
    with _patched(srv, "_cline_client", lambda streaming=False: up1), _roll_tripwire():
        _drive_buffered(prefixed)
    assert cp.cap_model_key(prefixed) == _CAPPED_MODEL
    limits = cp.load_pool().get("limits") or {}
    assert list(limits.get("default") or {}) == [_CAPPED_MODEL], limits
    # Second request: short-circuits at the pre-attempt gate, zero upstream calls.
    up2 = _FakeUpstream(body=b"must never be requested")
    tripwire = _roll_tripwire()
    with _patched(srv, "_cline_client", lambda streaming=False: up2), tripwire:
        resp = _drive_buffered(prefixed)
    assert up2.calls == [], f"{len(up2.calls)} upstream call(s) despite a persisted cap"
    assert len(tripwire.cap_skips) == 1
    assert "try again in" in json.loads(resp.body)["error"]["message"]


def test_stream_429_slot_is_not_re_selected_within_request():
    # The 2-minute streaming roll: 10 upstream calls on one bearer.
    toks = _seed_named_slots("alpha", "beta")
    up = _FakeUpstream(body=_429_BODY, status=429)
    started = time.time()
    with _patched(srv, "_cline_client", lambda streaming=False: up), _roll_tripwire():
        chunks = _drive_stream(_CAPPED_MODEL)
    assert time.time() - started < 5.0, f"stream roll took {time.time() - started:.1f}s"
    assert _calls_per_key(up) == {f"Bearer {t}": 1 for t in toks}
    body = "".join(chunks)
    assert '"type": "rate_limit_error"' in body
    assert '"code": "rate_limit_exceeded"' in body
    assert "daily free cap" in body and "try again in" in body
    assert '"choices"' not in body


def test_stream_parsed_429_persists_under_verbatim_id():
    _seed_named_slots("default")
    prefixed = "cline/" + _CAPPED_MODEL
    up1 = _FakeUpstream(body=_429_BODY, status=429)
    with _patched(srv, "_cline_client", lambda streaming=False: up1), _roll_tripwire():
        chunks = _drive_stream(prefixed)
    assert "daily free cap" in "".join(chunks)
    limits = cp.load_pool().get("limits") or {}
    assert list(limits.get("default") or {}) == [_CAPPED_MODEL], limits
    # Next request never reaches upstream again.
    up2 = _FakeUpstream(body=b"must never be requested", lines=_OK_SSE_LINES)
    tripwire = _roll_tripwire()
    with _patched(srv, "_cline_client", lambda streaming=False: up2), tripwire:
        chunks2 = _drive_stream(prefixed)
    assert up2.calls == [], f"{len(up2.calls)} upstream call(s) despite a persisted cap"
    assert len(tripwire.cap_skips) == 1
    assert "daily free cap" in "".join(chunks2)


def test_unparseable_429_retires_slot_without_inventing_a_reset():
    # No free-limit marker -> nothing is persisted (no fake reset invented),
    # but the slot is still retired in-request and the answer is still a
    # rate-limit response with the generic "try again later" hint.
    toks = _seed_named_slots("alpha", "beta")
    up = _FakeUpstream(body=b'{"error":{"message":"Rate limit exceeded"}}', status=429)
    with _patched(srv, "_cline_client", lambda streaming=False: up), _roll_tripwire():
        resp = _drive_buffered(_CAPPED_MODEL)
    assert _calls_per_key(up) == {f"Bearer {t}": 1 for t in toks}
    assert not (cp.load_pool().get("limits") or {}), "invented a reset from an unparsed 429"
    err = json.loads(resp.body)["error"]
    assert err["type"] == "rate_limit_error"
    assert "daily free cap" in err["message"] and "try again later" in err["message"]

    toks2 = _seed_named_slots("gamma")
    up2 = _FakeUpstream(body=b'{"error":{"message":"Rate limit exceeded"}}', status=429)
    with _patched(srv, "_cline_client", lambda streaming=False: up2), _roll_tripwire():
        chunks = _drive_stream(_CAPPED_MODEL)
    assert _calls_per_key(up2) == {f"Bearer {toks2[0]}": 1}
    body = "".join(chunks)
    assert '"type": "rate_limit_error"' in body
    assert "daily free cap" in body and "try again later" in body


# ── 10. sweep paths: retiring vs. looping on one slot ───────────────

_404_BODY = b'{"error":{"message":"model not available for this account"}}'
_503_BODY = b'{"error":{"message":"upstream exploded"}}'


async def _no_backoff(attempt, base=1.0, max_delay=10.0):
    """Backoff stub: keep retry-shape tests instant and deterministic."""
    return None


def test_client_error_sweep_retires_each_slot():
    # A <500 per-slot failure (404 = model not authorized) must ADVANCE: the
    # old path `continue`d without retiring, so select() handed the same bearer
    # back for the whole ceiling.
    toks = _seed_named_slots("alpha", "beta")
    up = _FakeUpstream(body=_404_BODY, status=404)
    with _patched(srv, "_cline_client", lambda streaming=False: up), _roll_tripwire():
        resp = _drive_buffered(_CAPPED_MODEL)
    assert _calls_per_key(up) == {f"Bearer {t}": 1 for t in toks}
    assert resp.status_code == 404

    toks2 = _seed_named_slots("alpha", "beta")
    up2 = _FakeUpstream(body=_404_BODY, status=404)
    with _patched(srv, "_cline_client", lambda streaming=False: up2), \
            _patched(srv, "_backoff", _no_backoff), _roll_tripwire():
        chunks = _drive_stream(_CAPPED_MODEL)
    assert _calls_per_key(up2) == {f"Bearer {t}": 1 for t in toks2}
    body = "".join(chunks)
    # Not a rate limit: the real 404 reason must survive the sweep.
    assert '"type": "upstream_error"' in body
    assert '"code": "404"' in body
    assert "model not available for this account" in body
    assert "rate_limit" not in body


def test_stream_exhausted_tail_reports_the_real_upstream_failure():
    # 5xx retries the SAME slot (transient), so the iteration ceiling — not
    # select() exhaustion — ends the roll. The tail used to print
    # "failed after retries: None" (a literal None) and drop the real message.
    _seed_named_slots("alpha")
    up = _FakeUpstream(body=_503_BODY, status=503)
    tripwire = _roll_tripwire(max_selects=64)
    with _patched(srv, "_cline_client", lambda streaming=False: up), \
            _patched(srv, "_backoff", _no_backoff), tripwire:
        chunks = _drive_stream(_CAPPED_MODEL)
    body = "".join(chunks)
    assert "failed after retries: None" not in body, body
    assert "None" not in body
    assert "upstream exploded" in body, body
    assert '"code": "503"' in body
    assert tripwire.selects > 1        # really did retry the slot


def test_failfast_429_threshold_is_reachable():
    # The net must be sized to the pool: min(CLINE_POOL_429_FAILFAST, slots+1).
    # Simulate the regression it exists to contain — a select() that IGNORES the
    # skip set, so the roll keeps hammering the one capped slot. The body is an
    # unparseable 429 on purpose: nothing is persisted, so the pre-attempt gate
    # cannot mask the counting (a parsed 429 would retire at the gate instead).
    def _stubborn(upstream):
        real_select = cp.pool.select

        def select_ignoring_skip(model=None, skip=None):
            return real_select(model)   # deliberately drops `skip`
        return _patched(srv.cline_keys, "select", select_ignoring_skip)

    # Buffered: 1 slot -> effective ceiling 2, not 12.
    _seed_named_slots("default")
    up = _FakeUpstream(body=b'{"error":{"message":"Rate limit exceeded"}}', status=429)
    with _patched(srv, "_cline_client", lambda streaming=False: up), _stubborn(up):
        resp = _drive_buffered(_CAPPED_MODEL)
    msg = json.loads(resp.body)["error"]["message"]
    assert "pool rate-limited" in msg, msg
    assert "2 consecutive 429s" in msg, msg
    assert len(up.calls) == 2, f"{len(up.calls)} upstream calls before the net fired"

    # Streaming: same pool-sized ceiling on its own code path.
    _seed_named_slots("default")
    up2 = _FakeUpstream(body=b'{"error":{"message":"Rate limit exceeded"}}', status=429)
    with _patched(srv, "_cline_client", lambda streaming=False: up2), _stubborn(up2):
        chunks = _drive_stream(_CAPPED_MODEL)
    body = "".join(chunks)
    assert "pool rate-limited" in body, body
    assert "2 consecutive 429s" in body, body
    assert len(up2.calls) == 2, f"{len(up2.calls)} upstream calls before the net fired"
    assert '"type": "rate_limit_error"' in body


# ── 11. cline path: per-model correctness (offline) ─────────────────

_ROTATION_BODY = b'{"recommended":[],"free":[{"id":"cline-free/brand-new-model"}],"clinePass":[]}'


def test_cline_sampling_drops_reasoning_effort():
    # Forwarding reasoning_effort unconditionally means every client that sets
    # it (OpenCode: nearly every request) breaks the models that reject the
    # field. Omitting it is the only choice that cannot break what works today.
    body = {"temperature": 0.3, "top_p": 0.9, "stop": ["x"],
            "reasoning_effort": "high", "reasoning": {"effort": "high"}}
    out = srv._cline_sampling_from(body)
    assert out == {"temperature": 0.3, "top_p": 0.9, "stop": ["x"]}
    assert "reasoning_effort" not in out and "reasoning" not in out
    req = srv.cline_request_body(_CAPPED_MODEL, [], True, None, None,
                                 None, None, out)
    assert "reasoning_effort" not in req
    assert req["temperature"] == 0.3      # the safe keys still pass through
    assert "reasoning_effort" not in srv._CLINE_SAMPLING_KEYS


def test_alias_slug_index_is_derived_and_behaviour_preserving():
    # The slug lookup is now DERIVED from _CLINE_ALIASES instead of being a
    # second hand-maintained table, so a future alias can never be missed.
    # NOTE: the z-ai namespace was NOT actually broken — _slug() strips every
    # dash, so _slug("z-ai/glm-5.3-flash") == "zai/glm5.3flash", which the
    # hand-written row already covered. Deriving is robustness, not a repair;
    # these assertions pin that it changed nothing observable today.
    for key, canonical in cx._CLINE_ALIASES.items():
        assert cx.cline_model_id("cline/" + key) == canonical, key
    hand_rows = {k: v for k, v in cx._CLINE_ALIASES.items() if k == cx._slug(k)}
    assert hand_rows, "expected hand-written slug rows to exist"
    for slug_key, canonical in hand_rows.items():
        assert cx._CLINE_ALIAS_SLUGS.get(slug_key) == canonical, slug_key
    by_slug = {}
    for key, canonical in cx._CLINE_ALIASES.items():
        s = cx._slug(key)
        if by_slug.setdefault(s, canonical) != canonical:
            raise AssertionError(f"slug collision on {s!r}: {by_slug[s]} vs {canonical}")
    assert cx.cline_model_id("cline/z-ai/glm-5.3-flash") == "z-ai/glm-5.3-flash"
    assert cx.cline_model_id("cline/cline-free/mimo-v2.6-flash") == "cline-free/mimo-v2.6-flash"
    assert cx.cline_model_id("cline/mimo-v2.6-flash") == "cline-free/mimo-v2.6-flash"
    assert cx.cline_model_id("cline/mimov26flash") == "cline-free/mimo-v2.6-flash"
    assert cx.cline_model_id("cline/stealth/space-bunny-alpha") == "stealth/space-bunny-alpha"
    assert cx.cline_model_id("cline/space-bunny-alpha") == "stealth/space-bunny-alpha"
    assert cx.cline_model_id("cline/spacebunnyalpha") == "stealth/space-bunny-alpha"
    assert cx.cline_model_id("cline/Z-AI/GLM-5.3-FLASH") == "z-ai/glm-5.3-flash"
    assert cx.cline_model_id("cline/z_ai/glm5.3 flash") == "z-ai/glm-5.3-flash"
    # Retired ids still pass through untouched.
    assert cx.cline_model_id("cline/glm-5.2") == "glm-5.2"
    assert cx.cline_model_id("cline/deepseek-v4-flash") == "deepseek-v4-flash"


def test_cline_model_id_resolves_against_live_free_rotation():
    # A model rotated INTO free[] is unknown to the hand-maintained table; it
    # must still resolve, or is_free_id() is False and the request 402s.
    try:
        cx.set_dynamic_free_ids(["cline-free/brand-new-model", "cline-free/other-thing"])
        assert cx.cline_model_id("cline/brand-new-model") == "cline-free/brand-new-model"
        assert cx.cline_model_id("cline/BRAND_New_Model") == "cline-free/brand-new-model"
        assert cx.cline_model_id("cline/cline-free/brandnewmodel") == "cline-free/brand-new-model"
        # The whole point: it is recognised as free, so no paid gate.
        assert cx.is_free_id(cx.cline_model_id("cline/brand-new-model")) is True
        # Unknown names still pass through verbatim (Cline errors on them).
        assert cx.cline_model_id("cline/totally-unknown-xyz") == "totally-unknown-xyz"
        # Deterministic across repeated calls (cached index).
        for _ in range(3):
            assert cx.cline_model_id("cline/brand-new-model") == "cline-free/brand-new-model"
    finally:
        cx.set_dynamic_free_ids([])


def test_chat_path_refreshes_stale_free_rotation():
    # The chat path must refresh a stale rotation (it used to happen only at
    # startup and on /v1/models, so a rotated-in model stayed 402-gated).
    _seed_named_slots("default")
    srv._CLINE_FREE_IDS = []
    srv._CLINE_FREE_AT = 0.0
    up = _FakeUpstream(body=_OK_SSE_BODY, lines=_OK_SSE_LINES,
                       recommended_body=_ROTATION_BODY)
    try:
        with _patched(srv, "_cline_client", lambda streaming=False: up):
            _drive_buffered(_CAPPED_MODEL)
        assert len(up.gets) == 1, "chat request did not refresh the stale rotation"
        assert cx.is_free_id("cline-free/brand-new-model") is True
        # TTL: a second chat request must NOT refetch.
        with _patched(srv, "_cline_client", lambda streaming=False: up):
            _drive_buffered(_CAPPED_MODEL)
        assert len(up.gets) == 1, "rotation refetched inside the TTL"
        # Streaming path refreshes too.
        srv._CLINE_FREE_IDS = []
        srv._CLINE_FREE_AT = 0.0
        with _patched(srv, "_cline_client", lambda streaming=False: up):
            _drive_stream(_CAPPED_MODEL)
        assert len(up.gets) == 2
    finally:
        cx.set_dynamic_free_ids([])
        srv._CLINE_FREE_IDS = ["cline-free/test-only"]
        srv._CLINE_FREE_AT = time.time()


def test_rotation_refresh_cannot_stampede():
    # Concurrent chat requests must not all fetch: the TTL check is not atomic.
    _seed_named_slots("default")
    srv._CLINE_FREE_IDS = []
    srv._CLINE_FREE_AT = 0.0
    up = _FakeUpstream(recommended_body=_ROTATION_BODY)
    try:
        async def race():
            return await asyncio.gather(
                srv._cline_fetch_recommended_free(),
                srv._cline_fetch_recommended_free(),
                srv._cline_fetch_recommended_free(),
            )
        with _patched(srv, "_cline_client", lambda streaming=False: up):
            asyncio.run(asyncio.wait_for(race(), timeout=15))
        assert len(up.gets) == 1, f"{len(up.gets)} concurrent rotation fetches"
        assert srv._CLINE_FREE_FETCHING is False   # flag always released
    finally:
        cx.set_dynamic_free_ids([])
        srv._CLINE_FREE_IDS = ["cline-free/test-only"]
        srv._CLINE_FREE_AT = time.time()


def test_stream_context_limit_fails_fast_like_buffered():
    # A deterministic context-window 400 must not burn the retry budget on the
    # streaming path (~30s) nor be reported as a generic upstream_error.
    body = b'{"error":{"code":"context_length_exceeded","message":"maximum context length exceeded"}}'
    toks = _seed_named_slots("alpha", "beta")
    up = _FakeUpstream(body=body, status=400)
    with _patched(srv, "_cline_client", lambda streaming=False: up), \
            _patched(srv, "_backoff", _no_backoff), _roll_tripwire():
        chunks = _drive_stream(_CAPPED_MODEL)
    assert len(up.calls) == 1, f"{len(up.calls)} upstream calls for a deterministic 400"
    stream_body = "".join(chunks)
    assert '"type": "context_window_error"' in stream_body
    assert "maximum context length exceeded" in stream_body
    # Buffered parity (already had the check — pinned so it stays).
    up2 = _FakeUpstream(body=body, status=400)
    with _patched(srv, "_cline_client", lambda streaming=False: up2):
        resp = _drive_buffered(_CAPPED_MODEL)
    assert len(up2.calls) == 1
    assert json.loads(resp.body)["error"]["type"] == "context_window_error"


def test_stream_success_bookkeeping_only_after_done():
    # A 200 that never reaches [DONE] must NOT clear a real daily cap (nor bump
    # usage / mark the slot active). The cap is applied mid-request by
    # on_send, i.e. after the pre-attempt gate already let the slot through.
    def cap_now():
        cp.record_limit_on_disk("default", _CAPPED_MODEL,
                                int(time.time() * 1000) + 3600_000)
        cp.reset_pool_cache_for_test()

    # Truncated stream -> cap survives.
    _seed_named_slots("default")
    truncated = _FakeUpstream(
        lines=('data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}',),
        on_send=cap_now)
    with _patched(srv, "_cline_client", lambda streaming=False: truncated), \
            _patched(srv, "_backoff", _no_backoff):
        chunks = _drive_stream(_CAPPED_MODEL)
    assert "incomplete_stream" in "".join(chunks)
    assert cp.pool.cap_reset_ms_for("default", _CAPPED_MODEL) is not None, \
        "a truncated stream cleared a real daily cap"
    limits = cp.load_pool().get("limits") or {}
    assert _CAPPED_MODEL in (limits.get("default") or {}), limits

    # Completed stream -> bookkeeping still happens.
    _seed_named_slots("default")
    complete = _FakeUpstream(lines=_OK_SSE_LINES, on_send=cap_now)
    with _patched(srv, "_cline_client", lambda streaming=False: complete):
        chunks2 = _drive_stream(_CAPPED_MODEL)
    assert "data: [DONE]" in "".join(chunks2)
    assert cp.pool.cap_reset_ms_for("default", _CAPPED_MODEL) is None, \
        "a completed stream failed to clear the cap"


def test_zen_new_free_aliases_resolve():
    # pi-freeflow catalog sync (2026-10-04): fledge-alias pre-existed;
    # ling-3.1-flash-* rows are new. Bare slash-free forms resolve to the
    # canonical -free id; unknown ids still pass through untouched.
    assert srv.resolveCanonicalModelId("fledge-alpha") == "fledge-alpha-free"
    assert srv.resolveCanonicalModelId("ling-3.1-flash") == "ling-3.1-flash-free"
    assert srv.resolveCanonicalModelId("ling-3.1") == "ling-3.1-flash-free"
    assert srv.resolveCanonicalModelId("mimo-v2.6-flash") == "mimo-v2.6-flash-free"
    assert srv.resolveCanonicalModelId("space-bunny-alpha") == "stealth/space-bunny-alpha"
    assert srv._is_blocked_model("fledge-alpha-free") is False
    assert srv._is_blocked_model("ling-3.1-flash-free") is False


_TESTS = [
    test_passthrough_never_strips_cline_pass,
    test_retired_ids_have_no_aliases,
    test_friendly_slugs_resolve,
    test_cap_delay_hours_minutes,
    test_cap_delay_minutes_only,
    test_cap_delay_bare_number_is_seconds,
    test_cap_delay_case_insensitive_and_words,
    test_cap_delay_absent_is_none,
    test_canonical_list_has_17_ids,
    test_free_limit_marker_gated,
    test_free_limit_model_trailing_punct_stripped,
    test_free_limit_bare_number_is_hours,
    test_slot_ordering_active_first_capped_last,
    test_stale_detection,
    test_same_account_duplicate_refusal,
    test_pool_roundtrip,
    test_cline_upstream_forces_stream_true,
    test_cline_request_body_still_passes_client_stream,
    test_cline_sse_to_completion_aggregates,
    test_cline_sse_to_completion_reasoning_and_tools,
    test_cline_sse_to_completion_error_marker_and_empty,
    test_upstream_error_preserves_string_message,
    test_as_dict_coercion,
    test_cline_sse_to_completion_string_error_marker,
    test_cline_sse_to_completion_nondict_chunks_ignored,
    test_first_chunk_error_string_body,
    test_zen_stream_nondict_chunk_no_crash,
    test_recommended_url_constant,
    test_recommended_free_extraction_verbatim,
    test_recommended_free_empty_malformed_fallback,
    test_dynamic_free_merge_never_drop,
    test_select_skip_accepts_token_and_slot_name,
    test_all_capped_pool_fails_fast_with_reset_hint,
    test_stream_all_capped_pool_does_not_spin,
    test_capped_plus_open_slot_serves_the_open_one,
    test_stream_capped_plus_open_slot_serves_the_open_one,
    test_429_slot_is_not_re_selected_within_request,
    test_all_slots_429_including_key_slot_fails_fast,
    test_parsed_429_persists_under_verbatim_id,
    test_stream_429_slot_is_not_re_selected_within_request,
    test_stream_parsed_429_persists_under_verbatim_id,
    test_unparseable_429_retires_slot_without_inventing_a_reset,
    test_report_capped_normalises_the_routing_prefix,
    test_client_error_sweep_retires_each_slot,
    test_stream_exhausted_tail_reports_the_real_upstream_failure,
    test_failfast_429_threshold_is_reachable,
    test_cline_sampling_drops_reasoning_effort,
    test_alias_slug_index_is_derived_and_behaviour_preserving,
    test_cline_model_id_resolves_against_live_free_rotation,
    test_chat_path_refreshes_stale_free_rotation,
    test_rotation_refresh_cannot_stampede,
    test_stream_context_limit_fails_fast_like_buffered,
    test_stream_success_bookkeeping_only_after_done,
    test_zen_new_free_aliases_resolve,
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
