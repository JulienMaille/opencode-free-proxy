"""Cline direct-model serving with key rotation.

Models addressed with a ``cline/`` prefix are routed straight to Cline's
OpenAI-compatible endpoint (``api.cline.bot/api/v1``) using the rotating key
pool from :mod:`cline_pool`. A key that hits a rate limit is rotated away
into cooldown with escalating backoff, mirroring the AMD provider.

Model IDs pass through UNCHANGED: the ``cline/`` routing prefix is stripped,
but the upstream ID (e.g. ``cline-pass/glm-5.3``) is never rewritten.
"""

_CLINE_PREFIXES = ("cline/",)

# Canonical Cline model IDs: 12 ClinePass IDs plus 5 usage-billing IDs.
# The live GET /models endpoint returns an OpenAI-shaped list WITHOUT the
# cline-pass/ IDs, so this hardcoded list is the mandatory static fallback.
# Do NOT prepend cline/ here.
_CLINE_CANONICAL = (
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
)

# Friendly slug (lowercased, no separators) + bare-suffix IDs -> canonical
# Cline ID. Keyed by the model id as the client sends it after the cline/
# prefix so e.g. "glm-5.3" or "cline-pass/GLM-5.3" resolves to
# "cline-pass/glm-5.3". Retired IDs (glm-5.2, kimi-k2.7-code, kimi-k2.6,
# deepseek-v4-flash) deliberately have NO entries: they pass through as-is
# so Cline errors on them itself.
_CLINE_ALIASES = {
    # cline-pass/glm-5.3
    "cline-pass/glm-5.3": "cline-pass/glm-5.3",
    "clinepass/glm5.3": "cline-pass/glm-5.3",
    "glm-5.3": "cline-pass/glm-5.3",
    "glm5.3": "cline-pass/glm-5.3",
    "glm53": "cline-pass/glm-5.3",
    # cline-pass/glm-5.3-flash
    "cline-pass/glm-5.3-flash": "cline-pass/glm-5.3-flash",
    "clinepass/glm5.3flash": "cline-pass/glm-5.3-flash",
    "glm-5.3-flash": "cline-pass/glm-5.3-flash",
    "glm5.3flash": "cline-pass/glm-5.3-flash",
    "glm53flash": "cline-pass/glm-5.3-flash",
    # cline-pass/kimi-k3
    "cline-pass/kimi-k3": "cline-pass/kimi-k3",
    "clinepass/kimik3": "cline-pass/kimi-k3",
    "kimi-k3": "cline-pass/kimi-k3",
    "kimik3": "cline-pass/kimi-k3",
    # cline-pass/deepseek-v4-pro
    "cline-pass/deepseek-v4-pro": "cline-pass/deepseek-v4-pro",
    "clinepass/deepseekv4pro": "cline-pass/deepseek-v4-pro",
    "deepseek-v4-pro": "cline-pass/deepseek-v4-pro",
    "deepseekv4pro": "cline-pass/deepseek-v4-pro",
    # cline-pass/deepseek-v4.1-flash (dot variants: _slug keeps ".", so list both)
    "cline-pass/deepseek-v4.1-flash": "cline-pass/deepseek-v4.1-flash",
    "clinepass/deepseekv4.1flash": "cline-pass/deepseek-v4.1-flash",
    "deepseek-v4.1-flash": "cline-pass/deepseek-v4.1-flash",
    "deepseekv4.1flash": "cline-pass/deepseek-v4.1-flash",
    "deepseekv41flash": "cline-pass/deepseek-v4.1-flash",
    # cline-pass/mimo-v2.5
    "cline-pass/mimo-v2.5": "cline-pass/mimo-v2.5",
    "clinepass/mimov2.5": "cline-pass/mimo-v2.5",
    "mimo-v2.5": "cline-pass/mimo-v2.5",
    "mimov2.5": "cline-pass/mimo-v2.5",
    "mimov25": "cline-pass/mimo-v2.5",
    # cline-pass/mimo-v2.5-pro
    "cline-pass/mimo-v2.5-pro": "cline-pass/mimo-v2.5-pro",
    "clinepass/mimov2.5pro": "cline-pass/mimo-v2.5-pro",
    "mimo-v2.5-pro": "cline-pass/mimo-v2.5-pro",
    "mimov2.5pro": "cline-pass/mimo-v2.5-pro",
    "mimov25pro": "cline-pass/mimo-v2.5-pro",
    # cline-pass/minimax-m3
    "cline-pass/minimax-m3": "cline-pass/minimax-m3",
    "clinepass/minimaxm3": "cline-pass/minimax-m3",
    "minimax-m3": "cline-pass/minimax-m3",
    "minimaxm3": "cline-pass/minimax-m3",
    # cline-pass/muse-spark-1.3-contributor
    "cline-pass/muse-spark-1.3-contributor": "cline-pass/muse-spark-1.3-contributor",
    "clinepass/musespark1.3contributor": "cline-pass/muse-spark-1.3-contributor",
    "muse-spark-1.3-contributor": "cline-pass/muse-spark-1.3-contributor",
    "musespark1.3contributor": "cline-pass/muse-spark-1.3-contributor",
    "musespark13contributor": "cline-pass/muse-spark-1.3-contributor",
    # cline-pass/qwen3.8-max
    "cline-pass/qwen3.8-max": "cline-pass/qwen3.8-max",
    "clinepass/qwen3.8max": "cline-pass/qwen3.8-max",
    "qwen3.8-max": "cline-pass/qwen3.8-max",
    "qwen3.8max": "cline-pass/qwen3.8-max",
    "qwen38max": "cline-pass/qwen3.8-max",
    # cline-pass/qwen3.7-max
    "cline-pass/qwen3.7-max": "cline-pass/qwen3.7-max",
    "clinepass/qwen3.7max": "cline-pass/qwen3.7-max",
    "qwen3.7-max": "cline-pass/qwen3.7-max",
    "qwen3.7max": "cline-pass/qwen3.7-max",
    "qwen37max": "cline-pass/qwen3.7-max",
    # cline-pass/qwen3.7-plus
    "cline-pass/qwen3.7-plus": "cline-pass/qwen3.7-plus",
    "clinepass/qwen3.7plus": "cline-pass/qwen3.7-plus",
    "qwen3.7-plus": "cline-pass/qwen3.7-plus",
    "qwen3.7plus": "cline-pass/qwen3.7-plus",
    "qwen37plus": "cline-pass/qwen3.7-plus",
    # anthropic/claude-sonnet-4-6 (usage billing)
    "anthropic/claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "anthropic/claudesonnet46": "anthropic/claude-sonnet-4-6",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "claudesonnet46": "anthropic/claude-sonnet-4-6",
    # google/gemini-2.5-pro (usage billing)
    "google/gemini-2.5-pro": "google/gemini-2.5-pro",
    "google/gemini2.5pro": "google/gemini-2.5-pro",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini2.5pro": "google/gemini-2.5-pro",
    "gemini25pro": "google/gemini-2.5-pro",
    # deepseek/deepseek-chat (usage billing)
    "deepseek/deepseek-chat": "deepseek/deepseek-chat",
    "deepseek/deepseekchat": "deepseek/deepseek-chat",
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseekchat": "deepseek/deepseek-chat",
    # openai/gpt-4o (usage billing)
    "openai/gpt-4o": "openai/gpt-4o",
    "openai/gpt4o": "openai/gpt-4o",
    "gpt-4o": "openai/gpt-4o",
    "gpt4o": "openai/gpt-4o",
    # minimax/minimax-m2.5 (usage billing)
    "minimax/minimax-m2.5": "minimax/minimax-m2.5",
    "minimax/minimaxm2.5": "minimax/minimax-m2.5",
    "minimax-m2.5": "minimax/minimax-m2.5",
    "minimaxm2.5": "minimax/minimax-m2.5",
    "minimaxm25": "minimax/minimax-m2.5",
}


def _slug(mid: str) -> str:
    return mid.lower().replace("_", "").replace(" ", "").replace("-", "")


def cline_models() -> list[str]:
    """Distinct canonical Cline IDs the proxy is configured to route to."""
    return sorted(set(_CLINE_CANONICAL))


def is_cline_model(raw_model: str | None) -> bool:
    """True if the client addressed the model with the Cline routing prefix."""
    if not raw_model:
        return False
    m = str(raw_model).strip().lower()
    return m.startswith(_CLINE_PREFIXES)


def cline_model_id(raw_model: str) -> str:
    """Resolve a client model id to the verbatim Cline model ID.

    ``cline/<name>`` maps through ``_CLINE_ALIASES``; exact matches win
    first, then slug (separator/case-insensitive) matches. The upstream
    ``cline-pass/`` segment is NEVER stripped. Unmapped names (including
    retired IDs) pass through as-is so Cline can error on unknown IDs
    itself.
    """
    raw = str(raw_model)
    base = raw.strip()
    lowered = base.lower()
    for pre in _CLINE_PREFIXES:
        if lowered.startswith(pre):
            base = base[len(pre):]
            break
    direct = _CLINE_ALIASES.get(base)
    if direct:
        return direct
    # Alias match even if separators/case differ in the friendly part
    friendly = _CLINE_ALIASES.get(_slug(base))
    if friendly:
        return friendly
    return base
