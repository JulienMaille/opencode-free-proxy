"""AMD Radeon TokenFactory direct-model serving with key rotation.

Models addressed with an ``amd/`` or ``radeon/`` prefix are routed straight
to AMD's OpenAI-compatible TokenFactory endpoint
(``developer.amd.com.cn/radeon/api/v1``) using the rotating key pool from
:mod:`amd_pool`. A key that hits a rate limit is rotated away into cooldown
with escalating backoff, mirroring the NVIDIA provider.
"""

_AMD_PREFIXES = ("amd/", "radeon/")

# Canonical TokenFactory model IDs, verbatim as returned by GET /models
# (source-of-truth is the live endpoint; this hardcoded list is the
# 2026-09-14 fallback from models.dev). Do NOT prepend amd/ here.
_AMD_CANONICAL = (
    "DeepSeek-V4-Flash",
    "DeepSeek-V4-Flash-Vision-Exp",
    "DeepSeek-V4.1-Flash",
    "MiniCPM5-2B",
    "Qwen3.8-27B",
    "Qwen3.8-Flash-Next",
)

# Friendly slug (lowercased, no separators) + legacy IDs -> canonical
# verbatim TokenFactory ID. Keyed by the model id as the client sends it
# after the amd/ prefix so e.g. "deepseek-v4-flash" or "DeepSeek-V4-Flash"
# resolves to "DeepSeek-V4-Flash".
_AMD_ALIASES = {
    # DeepSeek-V4-Flash (Cline defaultModelId)
    "deepseekv4flash": "DeepSeek-V4-Flash",
    "DeepSeek-V4-Flash": "DeepSeek-V4-Flash",
    # DeepSeek-V4-Flash-Vision-Exp
    "deepseekv4flashvisionexp": "DeepSeek-V4-Flash-Vision-Exp",
    "DeepSeek-V4-Flash-Vision-Exp": "DeepSeek-V4-Flash-Vision-Exp",
    # DeepSeek-V4.1-Flash (dot variants: _slug keeps ".", so list both)
    "deepseekv4.1flash": "DeepSeek-V4.1-Flash",
    "deepseekv41flash": "DeepSeek-V4.1-Flash",
    "DeepSeek-V4.1-Flash": "DeepSeek-V4.1-Flash",
    # MiniCPM5-2B
    "minicpm52b": "MiniCPM5-2B",
    "MiniCPM5-2B": "MiniCPM5-2B",
    # Qwen3.8-27B
    "qwen3.827b": "Qwen3.8-27B",
    "qwen3827b": "Qwen3.8-27B",
    "Qwen3.8-27B": "Qwen3.8-27B",
    # Qwen3.8-Flash-Next
    "qwen3.8flashnext": "Qwen3.8-Flash-Next",
    "qwen38flashnext": "Qwen3.8-Flash-Next",
    "Qwen3.8-Flash-Next": "Qwen3.8-Flash-Next",
    # Legacy / alias IDs seen in mirrors / hackathon
    "deepseekv4flash0731": "DeepSeek-V4-Flash",
    "DeepSeek-V4-Flash-0731": "DeepSeek-V4-Flash",
    "minicpmv46": "MiniCPM-V46",
    "MiniCPM-V46": "MiniCPM-V46",
    "minicpm51b": "MiniCPM5-1B",
    "MiniCPM5-1B": "MiniCPM5-1B",
    "qwen3.635ba3b": "Qwen3.6-35B-A3B",
    "qwen3635ba3b": "Qwen3.6-35B-A3B",
    "Qwen3.6-35B-A3B": "Qwen3.6-35B-A3B",
}


def _slug(mid: str) -> str:
    return mid.lower().replace("_", "").replace(" ", "").replace("-", "")


def amd_models() -> list[str]:
    """Distinct canonical TokenFactory IDs the proxy is configured to route to."""
    return sorted(set(_AMD_CANONICAL))


def is_amd_model(raw_model: str | None) -> bool:
    """True if the client addressed the model with an AMD routing prefix."""
    if not raw_model:
        return False
    m = str(raw_model).strip().lower()
    return m.startswith(_AMD_PREFIXES)


def amd_model_id(raw_model: str) -> str:
    """Resolve a client model id to the verbatim TokenFactory model ID.

    ``amd/<name>`` or ``radeon/<name>`` maps through ``_AMD_ALIASES``;
    exact matches win first, then slug (separator/case-insensitive) matches.
    Unmapped names are passed through as-is so AMD can error on unknown IDs
    itself.
    """
    raw = str(raw_model)
    base = raw.strip()
    lowered = base.lower()
    for pre in _AMD_PREFIXES:
        if lowered.startswith(pre):
            base = base[len(pre):]
            break
    direct = _AMD_ALIASES.get(base)
    if direct:
        return direct
    # Alias match even if separators/case differ in the friendly part
    friendly = _AMD_ALIASES.get(_slug(base))
    if friendly:
        return friendly
    return base
