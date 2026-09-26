#!/usr/bin/env python3
"""Cline compat pins (offline, no live probes, no keys).

Covers:
  1. Routing: only the `cline/` prefix routes; id passthrough never strips
     `cline-pass/`; retired IDs (glm-5.2, kimi-k2.7-code, kimi-k2.6,
     deepseek-v4-flash) have no aliases and pass through as-is.
  2. Cap-delay parser variants: "2h 15m", "45m", "20" (bare seconds),
     case-insensitive, plus no-delay -> None.
  3. Alias coverage: every canonical ID resolves from its bare suffix.

Run with `python -m pytest test_cline_compat.py -q`, or directly with
`python test_cline_compat.py` when pytest is unavailable. No network calls.
"""
import sys

# server.py parses CLI args at import: stub argv first so pytest's own
# argv never leaks into argparse.
sys.argv = ["test_cline_compat"]

import cline_pool as cp
import cline_proxy as cx


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
    assert cx.cline_model_id("cline/cline-pass/deepseek-v4.1-flash") == "cline-pass/deepseek-v4.1-flash"
    assert cx.cline_model_id("cline/cline-pass/qwen3.7-plus") == "cline-pass/qwen3.7-plus"
    assert cx.cline_model_id("cline/anthropic/claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"


def test_retired_ids_have_no_aliases():
    # Retired: NO aliases — pass through so Cline errors on them itself.
    assert cx.cline_model_id("cline/glm-5.2") == "glm-5.2"
    assert cx.cline_model_id("cline/kimi-k2.7-code") == "kimi-k2.7-code"
    assert cx.cline_model_id("cline/kimi-k2.6") == "kimi-k2.6"
    assert cx.cline_model_id("cline/deepseek-v4-flash") == "deepseek-v4-flash"


def test_friendly_slugs_resolve():
    assert cx.cline_model_id("cline/glm-5.3") == "cline-pass/glm-5.3"
    assert cx.cline_model_id("cline/GLM-5.3-FLASH") == "cline-pass/glm-5.3-flash"
    assert cx.cline_model_id("cline/kimi-k3") == "cline-pass/kimi-k3"
    assert cx.cline_model_id("cline/deepseek-v4-pro") == "cline-pass/deepseek-v4-pro"
    assert cx.cline_model_id("cline/mimo-v2.5-pro") == "cline-pass/mimo-v2.5-pro"
    assert cx.cline_model_id("cline/minimax-m3") == "cline-pass/minimax-m3"
    assert cx.cline_model_id("cline/muse-spark-1.3-contributor") == "cline-pass/muse-spark-1.3-contributor"
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
    assert len(ids) == 17
    for mid in (
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


_TESTS = [
    test_routing_prefix_only,
    test_passthrough_never_strips_cline_pass,
    test_retired_ids_have_no_aliases,
    test_friendly_slugs_resolve,
    test_cap_delay_hours_minutes,
    test_cap_delay_minutes_only,
    test_cap_delay_bare_number_is_seconds,
    test_cap_delay_case_insensitive_and_words,
    test_cap_delay_absent_is_none,
    test_canonical_list_has_17_ids,
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
