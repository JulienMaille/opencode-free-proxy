"""Cline direct-model serving with per-user login-slot rotation.

Models addressed with a ``cline/`` prefix are routed straight to Cline's
OpenAI-compatible endpoint (``api.cline.bot/api/v1``) using the login-slot
pool from :mod:`cline_pool` (device-login slots in ``cline-pool.json`` plus
static key slots from ``cline-api-keys.txt``).

Model IDs pass through UNCHANGED: the ``cline/`` routing prefix is stripped,
but the upstream ID (e.g. ``cline-free/kimi-k3``) is never rewritten.
Requests send IDs verbatim.
"""

_CLINE_PREFIXES = ("cline/",)

# Canonical Cline model IDs (cold-start/static fallback ONLY):
#  - 4 live-free IDs (observed free[] 2026-10-04: deepseek-v4.1-flash,
#    muse-spark-1.3-contributor, mimo-v2.6-flash, space-bunny-alpha).
#  - ROTATED-OUT/BILLED (kept for cold-start listing only — NOT free on a
#    near-zero balance; login slots serve the live free[] rotation only):
#      "z-ai/glm-5.3-flash" bills now (cost 3.4e-05 -> 402 on empty balance;
#      key slots only), "cline-free/kimi-k3" + "cline-free/solar-pro4" 404
#      upstream (dead). The live GET /models endpoint returns an OpenAI-shaped
#      list WITHOUT the cline-pass/ IDs, so this hardcoded list is the
#      mandatory static fallback. The live free-tier rotation (GET
#      CLINE_RECOMMENDED_URL -> free[]) REPLACES it via set_dynamic_free_ids()
#      whenever the dynamic cache is non-empty; the canonical IDs apply only
#      as a cold-start fallback while no live rotation is cached.
# Do NOT prepend cline/ here.
_CLINE_FREE_CANONICAL = (
    "cline-free/deepseek-v4.1-flash",
    "cline-free/muse-spark-1.3-contributor",
    "cline-free/mimo-v2.6-flash",
    "stealth/space-bunny-alpha",
    "z-ai/glm-5.3-flash",
    "cline-free/kimi-k3",
    "cline-free/solar-pro4",
)

# Official Cline free-tier rotation endpoint. GET returns
# ``{recommended:[...], free:[{id...}], clinePass:[...]}``; only the
# ``free[]`` IDs are consumed, verbatim (never rewritten). The live GET
# /models endpoint (458 IDs) carries NO cline-free/* entries, so this is
# the authoritative free-tier source. _CLINE_FREE_CANONICAL above stays as
# the cold-start/failure fallback and is never dropped from the merge.
CLINE_RECOMMENDED_URL = "https://api.cline.bot/api/v1/ai/cline/recommended-models"

# Last-known-good live free IDs (verbatim upstream IDs — entries already
# carry their own namespace, no cline/ prefix is added). Replaced on each
# successful recommended-models fetch; cline_free_models() returns them
# INSTEAD of _CLINE_FREE_CANONICAL whenever non-empty (dynamic wins); the
# static list above is a cold-start fallback only.
_CLINE_FREE_DYNAMIC: list[str] = []


def parse_recommended_free(payload) -> list[str]:
    """Extract verbatim ``free[]`` model IDs from a recommended-models body.

    Returns sorted distinct ``id`` strings; ``[]`` for empty/malformed
    bodies (never raises — callers treat empty as a failed fetch and keep
    the cached rotation).
    """
    try:
        if not isinstance(payload, dict):
            return []
        free = payload.get("free")
        if not isinstance(free, list):
            return []
        ids = set()
        for entry in free:
            mid = entry.get("id") if isinstance(entry, dict) else None
            if isinstance(mid, str) and mid.strip():
                ids.add(mid.strip())
        return sorted(ids)
    except Exception:
        return []


def set_dynamic_free_ids(ids) -> list[str]:
    """Replace the live free-rotation cache; returns the effective free set.

    The live rotation REPLACES the static cold-start fallback whenever it is
    non-empty (rotated-out IDs such as cline-free/kimi-k3 stop counting as
    free); an empty rotation keeps the previous cache, and with no live
    cache at all cline_free_models() falls back to _CLINE_FREE_CANONICAL.
    """
    global _CLINE_FREE_DYNAMIC
    try:
        _CLINE_FREE_DYNAMIC = sorted({str(m).strip() for m in (ids or []) if str(m).strip()})
    except Exception:
        _CLINE_FREE_DYNAMIC = []
    return cline_free_models()


def cline_dynamic_free_ids() -> list[str]:
    """Last-known-good live free IDs (verbatim, before the static merge)."""
    return list(_CLINE_FREE_DYNAMIC)

_CLINE_CANONICAL = _CLINE_FREE_CANONICAL + (
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
    # cline-free/deepseek-v4.1-flash (FREE)
    "cline-free/deepseek-v4.1-flash": "cline-free/deepseek-v4.1-flash",
    "clinefree/deepseekv4.1flash": "cline-free/deepseek-v4.1-flash",
    "deepseekv4.1flash": "cline-free/deepseek-v4.1-flash",
    "deepseekv41flash": "cline-free/deepseek-v4.1-flash",
    # cline-free/muse-spark-1.3-contributor (FREE)
    "cline-free/muse-spark-1.3-contributor": "cline-free/muse-spark-1.3-contributor",
    "clinefree/musespark1.3contributor": "cline-free/muse-spark-1.3-contributor",
    "muse-spark-1.3-contributor": "cline-free/muse-spark-1.3-contributor",
    "musespark1.3contributor": "cline-free/muse-spark-1.3-contributor",
    "musespark13contributor": "cline-free/muse-spark-1.3-contributor",
    # z-ai/glm-5.3-flash (BILLED — kept for cold-start listing only)
    "z-ai/glm-5.3-flash": "z-ai/glm-5.3-flash",
    "zai/glm5.3flash": "z-ai/glm-5.3-flash",
    "glm-5.3-flash": "z-ai/glm-5.3-flash",
    "glm5.3flash": "z-ai/glm-5.3-flash",
    "glm53flash": "z-ai/glm-5.3-flash",
    # cline-free/kimi-k3 (BILLED/ROTATED-OUT — kept for cold-start listing)
    "cline-free/kimi-k3": "cline-free/kimi-k3",
    "clinefree/kimik3": "cline-free/kimi-k3",
    "kimi-k3": "cline-free/kimi-k3",
    "kimik3": "cline-free/kimi-k3",
    # cline-free/solar-pro4 (BILLED/ROTATED-OUT — kept for cold-start listing)
    "cline-free/solar-pro4": "cline-free/solar-pro4",
    "clinefree/solarpro4": "cline-free/solar-pro4",
    "solar-pro4": "cline-free/solar-pro4",
    "solarpro4": "cline-free/solar-pro4",
    # cline-free/mimo-v2.6-flash (FREE — live free[] 2026-10-04)
    "cline-free/mimo-v2.6-flash": "cline-free/mimo-v2.6-flash",
    "clinefree/mimov2.6flash": "cline-free/mimo-v2.6-flash",
    "mimo-v2.6-flash": "cline-free/mimo-v2.6-flash",
    "mimov2.6flash": "cline-free/mimo-v2.6-flash",
    "mimov26flash": "cline-free/mimo-v2.6-flash",
    # stealth/space-bunny-alpha (FREE — live free[] 2026-10-04)
    "stealth/space-bunny-alpha": "stealth/space-bunny-alpha",
    "stealth/spacebunnyalpha": "stealth/space-bunny-alpha",
    "space-bunny-alpha": "stealth/space-bunny-alpha",
    "spacebunnyalpha": "stealth/space-bunny-alpha",
    # cline-pass/glm-5.3
    "cline-pass/glm-5.3": "cline-pass/glm-5.3",
    "clinepass/glm5.3": "cline-pass/glm-5.3",
    "glm-5.3": "cline-pass/glm-5.3",
    "glm5.3": "cline-pass/glm-5.3",
    "glm53": "cline-pass/glm-5.3",
    # cline-pass/glm-5.3-flash (exact cline-pass/* only; bare suffix serves FREE)
    "cline-pass/glm-5.3-flash": "cline-pass/glm-5.3-flash",
    "clinepass/glm5.3flash": "cline-pass/glm-5.3-flash",
    # cline-pass/kimi-k3 (exact cline-pass/* only; bare suffix serves FREE)
    "cline-pass/kimi-k3": "cline-pass/kimi-k3",
    "clinepass/kimik3": "cline-pass/kimi-k3",
    # cline-pass/deepseek-v4-pro
    "cline-pass/deepseek-v4-pro": "cline-pass/deepseek-v4-pro",
    "clinepass/deepseekv4pro": "cline-pass/deepseek-v4-pro",
    "deepseek-v4-pro": "cline-pass/deepseek-v4-pro",
    "deepseekv4pro": "cline-pass/deepseek-v4-pro",
    # cline-pass/deepseek-v4.1-flash (exact cline-pass/* only; bare serves FREE)
    "cline-pass/deepseek-v4.1-flash": "cline-pass/deepseek-v4.1-flash",
    "clinepass/deepseekv4.1flash": "cline-pass/deepseek-v4.1-flash",
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
    # cline-pass/muse-spark-1.3-contributor (exact cline-pass/* only; bare serves FREE)
    "cline-pass/muse-spark-1.3-contributor": "cline-pass/muse-spark-1.3-contributor",
    "clinepass/musespark1.3contributor": "cline-pass/muse-spark-1.3-contributor",
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


# slug(client key) -> canonical id, DERIVED from _CLINE_ALIASES at import.
# Hand-maintaining this second table duplicated _slug's rules by hand and could
# silently drift from them, so it is derived instead. Note _slug() strips EVERY
# dash globally, so _slug("z-ai/glm-5.3-flash") == "zai/glm5.3flash" (dashes
# inside a namespace segment are stripped too) — the hand-written key was
# already correct, and this table reproduces all of them exactly. A collision
# here would be a table defect — pinned by a test.
_CLINE_ALIAS_SLUGS: dict = {_slug(k): v for k, v in _CLINE_ALIASES.items()}


def cline_models() -> list[str]:
    """Distinct canonical Cline IDs the proxy is configured to route to."""
    return sorted(set(_CLINE_CANONICAL))


def cline_free_models() -> list[str]:
    """Effective FREE IDs: live recommended-models rotation when cached.

    Dynamic wins over static: whenever a live rotation has been fetched
    (non-empty dynamic cache), ONLY those verbatim IDs count as free, so
    rotated-out IDs (cline-free/kimi-k3, cline-free/solar-pro4) and
    now-billed IDs (z-ai/glm-5.3-flash) are excluded. The static
    _CLINE_FREE_CANONICAL list applies only as a cold-start fallback while
    no live rotation is cached.
    """
    if _CLINE_FREE_DYNAMIC:
        return sorted(set(_CLINE_FREE_DYNAMIC))
    return sorted(set(_CLINE_FREE_CANONICAL))


def is_free_id(mid) -> bool:
    """True if ``mid`` is in the effective live free set (dynamic-wins)."""
    if not isinstance(mid, str) or not mid.strip():
        return False
    return mid.strip() in set(cline_free_models())


# slug -> verbatim free id, cached on the current free-id set.
_FREE_SLUG_CACHE: tuple = ((), None)


def _free_slug_index() -> dict:
    """slug(tail) -> verbatim free id, for the live-rotation model fallback.

    A hand-maintained alias table cannot know about a model Cline rotates into
    ``free[]``, so an unresolved friendly name must still be able to resolve.
    Keys are the slug of each id's LAST path segment (plus its full slug), so
    ``new-model-x`` finds ``cline-free/new-model-x``. Built from the sorted id
    list with first-wins semantics: resolution stays deterministic when two ids
    share a tail. Cached on the id tuple so the per-request lookup is cheap.
    """
    global _FREE_SLUG_CACHE
    ids = tuple(cline_free_models())  # sorted upstream
    cached_ids, index = _FREE_SLUG_CACHE
    if cached_ids == ids:
        return index
    index = {}
    for mid in reversed(ids):  # reversed + assignment = first-id wins
        index[_slug(mid)] = mid
        index.setdefault(_slug(mid.rsplit("/", 1)[-1]), mid)
    _FREE_SLUG_CACHE = (ids, index)
    return index


def is_cline_model(raw_model: str | None) -> bool:
    """True if the client addressed the model with the Cline routing prefix."""
    if not raw_model:
        return False
    m = str(raw_model).strip().lower()
    return m.startswith(_CLINE_PREFIXES)


def cline_model_id(raw_model: str) -> str:
    """Resolve a client model id to the verbatim Cline model ID.

    ``cline/<name>`` maps through ``_CLINE_ALIASES``; exact matches win
    first, then slug (separator/case-insensitive) matches, then the live free
    rotation by full/tail slug. The upstream ``cline-pass/`` segment is NEVER
    stripped. Unmapped names (including retired IDs) pass through as-is so
    Cline can error on unknown IDs itself.

    The live-free fallback is what keeps a rotated-in model usable: a name the
    hand-maintained table has never heard of still resolves to the verbatim
    free id, so ``is_free_id`` sees it and the paid gate does not 402 it.
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
    slug = _slug(base)
    # Alias match even if separators/case differ in the friendly part
    friendly = _CLINE_ALIAS_SLUGS.get(slug)
    if friendly:
        return friendly
    # Live free rotation: a rotated-in id the alias table has never seen.
    rotated = _free_slug_index().get(slug)
    if rotated:
        return rotated
    return base
