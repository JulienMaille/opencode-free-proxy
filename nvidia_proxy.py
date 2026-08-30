"""NVIDIA NIM direct-model serving with key rotation.

Models addressed with an ``nvidia/`` or ``nvimin/`` prefix are routed straight
to NVIDIA's OpenAI-compatible NIM endpoint (``integrate.api.nvidia.com``)
using the rotating key pool from :mod:`nvidia_pool`. A key is rotated away and
cooled down when it hits a rate limit twice in a row.
"""
_NVIDIA_PREFIXES = ("nvidia/", "nvimin/")

# Friendly model alias -> NVIDIA NIM slug. Keyed by the model id as the client
# sends it after the nvidia/ prefix (lowercased, without separators) so we can
# resolve e.g. "deepseek-v4-pro-0813" or "deepseek-ai/deepseek-v4-pro-0813".
_NVIDIA_ALIASES = {
    "deepseekv4pro0813": "deepseek-ai/deepseek-v4-pro-0813",
    "deepseek-ai/deepseek-v4-pro-0813": "deepseek-ai/deepseek-v4-pro-0813",
    "deepseekv4flash0731": "deepseek-ai/deepseek-v4-flash-0731",
    "deepseek-ai/deepseek-v4-flash-0731": "deepseek-ai/deepseek-v4-flash-0731",
    "deepseekcoderv3": "deepseek-ai/deepseek-coder",
    "kimi-k3": "moonshotai/kimi-k3",
    "kimik3": "moonshotai/kimi-k3",
    "moonshotai/kimi-k3": "moonshotai/kimi-k3",
    "kimi-k2.6": "moonshotai/kimi-k2.6",
    "llama3132bitinstruct": "meta/llama-3.1-32b-instruct",
    "llama32vision": "meta/llama-3.2-11b-vision-instruct",
}


def _slug(mid: str) -> str:
    return mid.lower().replace("_", "").replace(" ", "").replace("-", "")


def nvidia_models() -> list[str]:
    """Distinct NVIDIA NIM slugs the proxy is configured to route to."""
    return sorted({v for v in _NVIDIA_ALIASES.values()})


def is_nvidia_model(raw_model: str | None) -> bool:
    """True if the client addressed the model with an NVIDIA routing prefix."""
    if not raw_model:
        return False
    m = str(raw_model).strip().lower()
    return m.startswith(_NVIDIA_PREFIXES)


def nvidia_model_id(raw_model: str) -> str:
    """Resolve a client model id to the NVIDIA NIM slug.

    ``nvidia/<name>`` or ``nvimin/<name>`` maps through ``_NVIDIA_ALIASES``;
    any string that already contains ``/`` is passed through unchanged (so a
    full ``deepseek-ai/deepseek-v4-pro-0813`` works too). Unmapped names are
    passed through as-is so NVIDIA can error on unknown slugs itself.
    """
    raw = str(raw_model)
    base = raw.strip()
    lowered = base.lower()
    for pre in _NVIDIA_PREFIXES:
        if lowered.startswith(pre):
            base = base[len(pre):]
            break
    direct = _NVIDIA_ALIASES.get(base)
    if direct:
        return direct
    # Alias match even if separators/case differ in the friendly part
    friendly = _NVIDIA_ALIASES.get(_slug(base))
    if friendly:
        return friendly
    return base
