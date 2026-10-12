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
import json
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


# ── Issuer affinity + rejected-reasoning recovery ──────────────────

_AFFINITY_SNAP = None


def _affinity_snapshot():
    return (
        dict(srv._conv_affinity),
        {k: (set(v[0]), v[1]) for k, v in srv._rejected_reasoning.items()},
    )


def _affinity_restore(snap):
    aff, rej = snap
    srv._conv_affinity.clear()
    srv._conv_affinity.update(aff)
    srv._rejected_reasoning.clear()
    srv._rejected_reasoning.update(rej)


def _reasoning_item(blob, item_id="rs_1"):
    return {"type": "reasoning", "id": item_id, "encrypted_content": blob, "summary": []}


def _body_with_reasoning(blobs, prev_id="resp_prev", extra_user="hello"):
    return {
        "model": "m",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": extra_user}]},
            {"type": "reasoning", "id": "rs_x", "encrypted_content": blobs[0], "summary": []},
        ],
        "previous_response_id": prev_id,
    }


def test_conversation_key_derivation_stability():
    b1 = {"model": "m", "input": [{"role": "user", "content": [{"type": "input_text", "text": "first msg"}]}]}
    b2 = {"model": "m", "input": [{"role": "user", "content": [{"type": "input_text", "text": "first msg"}]}]}
    b3 = {"model": "m", "input": [{"role": "user", "content": [{"type": "input_text", "text": "other convo"}]}]}
    assert srv._responses_conversation_key(b1) == srv._responses_conversation_key(b2)
    assert srv._responses_conversation_key(b1) != srv._responses_conversation_key(b3)
    # sha256 of the first user message content.
    assert srv._responses_conversation_key(b1) == "sha256:" + hashlib.sha256(b"first msg").hexdigest()
    # prompt_cache_key wins when present.
    b4 = dict(b1, prompt_cache_key="conv-42")
    assert srv._responses_conversation_key(b4) == "pck:conv-42"
    # Only the FIRST user message matters: later differences don't change the key.
    b5 = {"model": "m", "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "first msg"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "second msg"}]},
    ]}
    assert srv._responses_conversation_key(b5) == srv._responses_conversation_key(b1)
    # String content form also works.
    b6 = {"model": "m", "input": [{"role": "user", "content": "first msg"}]}
    assert srv._responses_conversation_key(b6) == srv._responses_conversation_key(b1)
    # No user message → None.
    assert srv._responses_conversation_key({"model": "m", "input": [{"type": "reasoning"}]}) is None
    assert srv._responses_conversation_key({}) is None
    assert srv._responses_conversation_key(None) is None


def test_reasoning_caller_mismatch_regex():
    assert srv._is_reasoning_caller_mismatch(
        'invalid_request_error: reasoning encrypted_content "blob123" was not issued to this caller'
    )
    assert srv._is_reasoning_caller_mismatch(
        "Reasoning encrypted_content 'abc' was not issued to this caller."
    )
    assert not srv._is_reasoning_caller_mismatch("reasoning encrypted_content was issued to this caller")
    assert not srv._is_reasoning_caller_mismatch("some unrelated 400 error")
    assert not srv._is_reasoning_caller_mismatch("")


def test_expired_reasoning_reference_regex():
    assert srv._is_expired_reasoning_reference(
        "invalid_request_error: referenced reasoning item 'rs_abc123' was not found or has expired"
    )
    assert srv._is_expired_reasoning_reference(
        "Referenced reasoning item 'rs_x' was not found or has expired."
    )
    assert not srv._is_expired_reasoning_reference("referenced reasoning item 'rs_x' was found")
    assert not srv._is_expired_reasoning_reference("reasoning encrypted_content was not issued to this caller")
    assert not srv._is_expired_reasoning_reference("")


def test_strip_reasoning_encryption_removes_blobs_and_previous_id():
    body = _body_with_reasoning(["blobA", "blobB"])
    body["input"].append(_reasoning_item("blobB", "rs_2"))
    stripped = srv._strip_reasoning_encryption(body)
    # encrypted_content gone from every reasoning item.
    for item in stripped["input"]:
        if item.get("type") == "reasoning":
            assert "encrypted_content" not in item
    assert "previous_response_id" not in stripped
    # Other fields survive.
    assert stripped["model"] == "m"
    assert stripped["input"][0]["role"] == "user"
    assert [i.get("id") for i in stripped["input"] if i.get("type") == "reasoning"] == ["rs_x", "rs_2"]
    # Original body is not mutated (deep copy).
    assert body["input"][1]["encrypted_content"] == "blobA"
    assert body["previous_response_id"] == "resp_prev"
    # input_messages field is covered too.
    body2 = {"input_messages": [_reasoning_item("blobC")], "previous_response_id": "p"}
    stripped2 = srv._strip_reasoning_encryption(body2)
    assert "encrypted_content" not in stripped2["input_messages"][0]
    assert "previous_response_id" not in stripped2


def test_strip_all_reasoning_items_removes_whole_items():
    body = _body_with_reasoning(["blobA"])
    stripped = srv._strip_all_reasoning_items(body)
    assert all(i.get("type") != "reasoning" for i in stripped["input"])
    assert "previous_response_id" not in stripped
    assert stripped["input"][0]["role"] == "user"
    # Original untouched.
    assert body["input"][1]["encrypted_content"] == "blobA"


def test_rejected_reasoning_remember_and_strip():
    snap = _affinity_snapshot()
    try:
        # Upstream rejected the body carrying blobA: every blob in it was
        # issued by the old backend, so all of them are remembered.
        body = _body_with_reasoning(["blobA"])
        srv._remember_rejected_reasoning("conv1", body)
        assert "conv1" in srv._rejected_reasoning
        hashes, exp = srv._rejected_reasoning["conv1"]
        assert hashes == {hashlib.sha256(b"blobA").hexdigest()}
        assert exp > time.time()

        # A later turn replaying the rejected blob PLUS a fresh one: only the
        # rejected blob's item is dropped.
        body2 = _body_with_reasoning(["blobA"])
        body2["input"].append(_reasoning_item("blobB", "rs_2"))
        stripped = srv._strip_rejected_reasoning("conv1", body2)
        reasoning = [i for i in stripped["input"] if i.get("type") == "reasoning"]
        assert [i.get("id") for i in reasoning] == ["rs_2"]
        # Original body not mutated.
        assert len([i for i in body2["input"] if i.get("type") == "reasoning"]) == 2

        # Unknown key → same object back (no copy).
        assert srv._strip_rejected_reasoning("nope", body) is body
        # Empty rejected set → same object back.
        srv._rejected_reasoning["empty"] = (set(), time.time() + 3600)
        assert srv._strip_rejected_reasoning("empty", body) is body

        # Expired entry → treated as absent and swept.
        srv._rejected_reasoning["old"] = ({"x"}, time.time() - 1)
        assert srv._strip_rejected_reasoning("old", body) is body
        assert "old" not in srv._rejected_reasoning

        # Remembering with no blobs is a no-op.
        srv._remember_rejected_reasoning("conv2", {"input": [{"role": "user", "content": "hi"}]})
        assert "conv2" not in srv._rejected_reasoning
    finally:
        _affinity_restore(snap)


def test_affinity_remember_ttl_and_eviction():
    snap = _affinity_snapshot()
    try:
        srv._affinity_remember("k1", "relay-a")
        assert srv._conv_affinity["k1"][0] == "relay-a"
        assert srv._conv_affinity["k1"][1] > time.time()

        # Expired entries are swept on write and treated as absent on read.
        srv._conv_affinity["stale"] = ("relay-old", time.time() - 1)
        srv._affinity_remember("k2", "relay-b")
        assert "stale" not in srv._conv_affinity
        aff = srv._conv_affinity.get("stale")
        assert aff is None or aff[1] <= time.time()

        # Re-remembering refreshes the relay and expiry.
        srv._affinity_remember("k1", "relay-c")
        assert srv._conv_affinity["k1"][0] == "relay-c"

        # Cap: oldest entry is evicted once the dict is full.
        srv._conv_affinity.clear()
        for i in range(srv._CONV_AFFINITY_MAX):
            srv._conv_affinity[f"fill{i}"] = (f"r{i}", time.time() + 3600 + i)
        srv._affinity_remember("newest", "relay-new")
        assert len(srv._conv_affinity) == srv._CONV_AFFINITY_MAX
        assert "newest" in srv._conv_affinity
        assert "fill0" not in srv._conv_affinity  # oldest expiry evicted
    finally:
        _affinity_restore(snap)


def test_rejected_reasoning_cap():
    snap = _affinity_snapshot()
    try:
        srv._rejected_reasoning.clear()
        body = _body_with_reasoning(["b"])
        for i in range(srv._REJECTED_REASONING_MAX_KEYS + 5):
            srv._remember_rejected_reasoning(f"c{i}", body)
        assert len(srv._rejected_reasoning) == srv._REJECTED_REASONING_MAX_KEYS
        # Per-key blob cap.
        many = _body_with_reasoning(["b0"])
        for i in range(srv._REJECTED_REASONING_MAX_BLOBS + 10):
            many["input"].append(_reasoning_item(f"blob{i}", f"rs_{i}"))
        srv._remember_rejected_reasoning("big", many)
        assert len(srv._rejected_reasoning["big"][0]) == srv._REJECTED_REASONING_MAX_BLOBS
    finally:
        _affinity_restore(snap)


# ── pi-freeflow 880b103 / 1.33.3: parallel tool calls kept apart ────
def test_parallel_tool_calls_split_by_call_id():
    """A model labeling every parallel call index 0 must not merge them."""
    slots, by_id = {}, {}
    for tc in [
        {"index": 0, "id": "call_A", "function": {"name": "bash", "arguments": '{"a"'}},
        {"index": 0, "id": "call_B", "function": {"name": "read", "arguments": '{"b"'}},
        {"index": 0, "function": {"arguments": ':1}'}},   # id-less -> call_A continues
    ]:
        key = srv._tool_call_slot_key(tc, slots, by_id)
        slot = slots.setdefault(key, {"id": "", "function": {"name": "", "arguments": ""}})
        if tc.get("id"):
            slot["id"] = tc["id"]
        fn = tc.get("function") or {}
        slot["function"]["name"] += fn.get("name") or ""
        slot["function"]["arguments"] += fn.get("arguments") or ""
    assert len(slots) == 2, slots
    assert slots[by_id["call_A"]]["function"]["arguments"] == '{"a":1}'
    assert slots[by_id["call_B"]]["function"]["arguments"] == '{"b"'


def test_tool_call_same_id_stays_one_slot():
    slots, by_id = {}, {}
    for tc in [
        {"index": 0, "id": "call_Q", "function": {"name": "bash"}},
        {"index": 0, "id": "call_Q", "function": {"arguments": '{"x":1}'}},
    ]:
        key = srv._tool_call_slot_key(tc, slots, by_id)
        slot = slots.setdefault(key, {"id": "", "function": {"name": "", "arguments": ""}})
        slot["function"]["name"] += (tc.get("function") or {}).get("name") or ""
        slot["function"]["arguments"] += (tc.get("function") or {}).get("arguments") or ""
    assert len(slots) == 1
    assert slots[0]["function"]["arguments"] == '{"x":1}'


# ── pi-freeflow 1.30.0: injected placeholders never reach a tool-less host ──
def test_injected_placeholder_calls_stripped_only_without_caller_tools():
    calls = [
        {"id": "c1", "function": {"name": "bash", "arguments": "{}"}},
        {"id": "c2", "function": {"name": "todo_write", "arguments": "{}"}},
    ]
    kept = srv._strip_injected_tool_calls(calls, caller_declared_tools=False)
    assert [c["function"]["name"] for c in kept] == ["todo_write"]
    # A host that DID declare tools keeps everything, including a real "bash".
    kept2 = srv._strip_injected_tool_calls(calls, caller_declared_tools=True)
    assert len(kept2) == 2


def test_injected_placeholder_deltas_stripped_in_place():
    piece = {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"name": "read", "arguments": "{}"}},
        {"index": 1, "function": {"name": "todo_write", "arguments": "{}"}},
    ]}}]}
    srv._strip_injected_tool_deltas(piece, caller_declared_tools=False)
    names = [tc["function"]["name"] for tc in piece["choices"][0]["delta"]["tool_calls"]]
    assert names == ["todo_write"], names


# ── pi-freeflow 98a12c5 / 1.33.0: per-model effort clamp ───────────
def test_effort_clamped_to_model_ceiling():
    # muse-spark tops out at xhigh; max/ultra clamp down.
    assert srv._clamp_effort_for_model("muse-spark-1.3-contributor-free", "max") == "xhigh"
    assert srv._clamp_effort_for_model("muse-spark-1.3-contributor-free", "high") == "high"
    # longcat/space-bunny legitimately offer max: it must survive.
    assert srv._clamp_effort_for_model("longcat-2.5-preview-free", "max") == "max"
    assert srv._clamp_effort_for_model("space-bunny-free", "max") == "max"
    assert srv._clamp_effort_for_model("longcat-2.5-preview-free", "minimal") == "minimal"
    # Unrecognised levels are dropped, not guessed.
    assert srv._clamp_effort_for_model("space-bunny-free", "wat") is None


# ── pi-freeflow 37c2cb6 / 1.33.2: tool-argument schema repair ──────
_TODO_SCHEMA = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["init"]},
        "list": {"type": "array", "items": {"type": "object",
                                            "properties": {"phase": {"type": "string"},
                                                           "items": {"type": "array", "items": {"type": "string"}}}}},
        "items": {"type": "array", "items": {"type": "string"}},
        "task": {"type": "string"},
        "phase": {"type": "string"},
    },
    "required": ["op"],
}


def test_repair_root_field_swapped_for_declared_type():
    """Measured live: muse-spark-1.3 emits {"items":[...]} for a "list" field."""
    raw = json.dumps({"op": "init",
                      "items": [{"phase": "Research", "items": ["Map repo structure"]}]})
    fixed = srv._repair_tool_arguments(raw, _TODO_SCHEMA)
    assert fixed is not None
    assert json.loads(fixed)["list"][0]["phase"] == "Research"
    assert "items" not in json.loads(fixed)


def test_repair_nested_unknown_key_renamed():
    """Measured live: step-5 emits "tasks" inside a phase instead of "items"."""
    raw = json.dumps({"op": "init", "list": [{"phase": "Research", "tasks": ["Map repo structure"]}]})
    fixed = srv._repair_tool_arguments(raw, _TODO_SCHEMA)
    assert fixed is not None
    assert json.loads(fixed)["list"][0]["items"] == ["Map repo structure"]


def test_repair_refuses_when_it_must():
    canonical = json.dumps({"op": "init", "list": [{"phase": "A", "items": ["x"]}]})
    assert srv._repair_tool_arguments(canonical, _TODO_SCHEMA) is None
    # Truncated JSON is never reconstructed.
    assert srv._repair_tool_arguments('{"op":"init","list":[{"phase":"A"', _TODO_SCHEMA) is None
    assert srv._repair_tool_arguments("not json", _TODO_SCHEMA) is None
    assert srv._repair_tool_arguments('["a"]', _TODO_SCHEMA) is None
    # No declared properties -> nothing to check.
    assert srv._repair_tool_arguments(json.dumps({"op": "init"}),
                                      {"type": "object", "properties": {}}) is None
    # Ambiguous: two declared keys absent, one undeclared present -> refuse.
    amb = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
    assert srv._repair_tool_arguments(json.dumps({"zzz": "v"}), amb) is None


# ── pi-freeflow f637c58 / 1.34.0: stale prefix off a call's front ───
def test_repair_salvages_stale_previous_call_prefix():
    """A single call can arrive with the previous call's tail glued in front."""
    bash_schema = {"type": "object",
                   "properties": {"i": {"type": "string"}, "command": {"type": "string"},
                                  "timeout": {"type": "number"}}}
    todo_with_i = {"type": "object", "properties": {
        "i": {"type": "string"},
        "op": {"type": "string"},
        "list": {"type": "array", "items": {"type": "object"}},
    }}
    cases = [
        (',"timeout":400}{"i":"p1","command":"ls"}', bash_schema),
        (']}]}{"i":"Planning research phases","op":"init"}', todo_with_i),
    ]
    for raw, schema in cases:
        fixed = srv._repair_tool_arguments(raw, schema)
        assert fixed is not None, raw
        assert json.loads(fixed) is not None, fixed


def test_salvage_refused_when_junk_hides_no_real_payload():
    bash_schema = {"type": "object",
                   "properties": {"i": {"type": "string"}, "command": {"type": "string"}}}
    # A brace appears, but the object after it does not match the tool.
    assert srv._repair_tool_arguments('["nonsense", true]', bash_schema) is None
    assert srv._repair_tool_arguments('{"nope": 1}', _TODO_SCHEMA) is None
    # Truncated AFTER a brace still yields nothing.
    assert srv._repair_tool_arguments(']["op":"init","list":[{"phase"', _TODO_SCHEMA) is None


def test_salvage_leaves_wellformed_alone():
    good = json.dumps({"op": "init", "list": [{"phase": "A", "items": ["x"]}]})
    assert srv._repair_tool_arguments(good, _TODO_SCHEMA) is None


def _sse(delta, finish=None):
    return f"data: {json.dumps({'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n"


async def _collect(gen):
    return [c async for c in gen]


def _reassemble(chunks):
    name, args = None, ""
    for unit in chunks:
        body = unit.strip()
        if not body.startswith("data:") or "[DONE]" in body:
            continue
        piece = json.loads(body[5:].strip())
        for tc in (piece["choices"][0].get("delta") or {}).get("tool_calls") or []:
            if tc.get("function", {}).get("name"):
                name = tc["function"]["name"]
            args += tc["function"].get("arguments") or ""
    return name, args


def test_stream_repair_withholds_and_flushes_repaired_args():
    async def _run():
        seq = [
            _sse({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                  "function": {"name": "todo", "arguments": '{"op":"init","items":['}}]}),
            _sse({"tool_calls": [{"index": 0, "function": {"arguments": '{"phase":"A","items":["x"]}'}}]}),
            _sse({"tool_calls": [{"index": 0, "function": {"arguments": "]}"}}]}),
            _sse({}, finish="tool_calls"),
            "data: [DONE]\n\n",
        ]

        async def _agen(items):
            for i in items:
                yield i

        return await _collect(srv._repair_stream_tool_args(_agen(seq), {"todo": _TODO_SCHEMA}, "m"))

    name, args = _reassemble(asyncio.run(_run()))
    assert name == "todo", name
    assert json.loads(args)["list"][0]["items"] == ["x"], args


def test_stream_passthrough_when_no_schemas():
    seq = [
        _sse({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "todo", "arguments": '{"op":"init",'}}]}),
        _sse({"tool_calls": [{"index": 0, "function": {"arguments": '"list":[]}'}}]}),
        _sse({}, finish="tool_calls"),
        "data: [DONE]\n\n",
    ]

    async def _run():
        async def _agen(items):
            for i in items:
                yield i

        return await _collect(srv._repair_stream_tool_args(_agen(seq), {}, "m"))

    assert "".join(asyncio.run(_run())) == "".join(seq)


_TESTS = [
    test_max_tokens_never_renamed,
    test_reasoning_tool_calls_survive_cache_key_ops,
    test_prompt_cache_retention_stripped,
    test_keyless_omission_wins_over_injected_authorization,
    test_empty_discovery_never_wipes_snapshot,
    test_dual_key_models_dev_lookup,
    test_humanize_name_fallback,
    test_conservative_image_gating_default,
    test_conversation_key_derivation_stability,
    test_reasoning_caller_mismatch_regex,
    test_expired_reasoning_reference_regex,
    test_strip_reasoning_encryption_removes_blobs_and_previous_id,
    test_strip_all_reasoning_items_removes_whole_items,
    test_rejected_reasoning_remember_and_strip,
    test_affinity_remember_ttl_and_eviction,
    test_rejected_reasoning_cap,
    test_parallel_tool_calls_split_by_call_id,
    test_tool_call_same_id_stays_one_slot,
    test_injected_placeholder_calls_stripped_only_without_caller_tools,
    test_injected_placeholder_deltas_stripped_in_place,
    test_effort_clamped_to_model_ceiling,
    test_repair_root_field_swapped_for_declared_type,
    test_repair_nested_unknown_key_renamed,
    test_repair_refuses_when_it_must,
    test_repair_salvages_stale_previous_call_prefix,
    test_salvage_refused_when_junk_hides_no_real_payload,
    test_salvage_leaves_wellformed_alone,
    test_stream_repair_withholds_and_flushes_repaired_args,
    test_stream_passthrough_when_no_schemas,
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
