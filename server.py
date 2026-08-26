import argparse
import asyncio
import hashlib
import json
import os
import random
import secrets
import sys
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from proxy_pool import pool as proxy_pool
from proxy_pool import (
    ALLOWED_PROXY_PORTS,
    MAX_RETRIES,
    PROXY_PORT_FILTER_ENABLED,
    REQUEST_CONNECT_TIMEOUT,
    REQUEST_READ_TIMEOUT,
    STREAM_READ_TIMEOUT,
)

_BASE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent

# ── CLI args ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenCode Free Proxy")
    p.add_argument("--port", type=int, default=6446, help="Listen port (default: 6446)")
    p.add_argument("--host", default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    p.add_argument("--proxy", default=None, help="Static SOCKS5 proxy (socks5://host:port)")
    p.add_argument("--proxy-pool", action=argparse.BooleanOptionalAction, default=True, help="Enable SOCKS5 proxy pool with transport-failure and per-proxy 429 rotation (default: on, use --no-proxy-pool to disable)")
    p.add_argument("--api-key", default=None, help="API key for client auth")
    return p.parse_args()

args = parse_args()

# ── SOCKS5 Proxy / Pool ───────────────────────────────────────────

def normalize_proxy_url(raw: str | None) -> str | None:
    if not raw:
        return None
    if not raw.startswith("socks5://") and not raw.startswith("socks4://"):
        return "socks5://" + raw
    return raw

STATIC_PROXY = normalize_proxy_url(args.proxy or os.environ.get("SOCKS5_PROXY"))

# Proxy pool: default on, disable with --no-proxy-pool or OPENCODE_PROXY_POOL=false
_pp_env = os.environ.get("OPENCODE_PROXY_POOL", "").lower()
PROXY_POOL_ENABLED = args.proxy_pool
if _pp_env:
    PROXY_POOL_ENABLED = _pp_env not in ("0", "false", "no")

_default_proxy = None if PROXY_POOL_ENABLED else STATIC_PROXY

# Default client for direct/static-proxy mode
_default_client = httpx.AsyncClient(
    base_url="https://opencode.ai",
    timeout=httpx.Timeout(
        connect=REQUEST_CONNECT_TIMEOUT,
        read=REQUEST_READ_TIMEOUT,
        write=REQUEST_READ_TIMEOUT,
        pool=REQUEST_CONNECT_TIMEOUT,
    ),
    proxy=_default_proxy,
)
_stream_default_client = httpx.AsyncClient(
    base_url="https://opencode.ai",
    timeout=httpx.Timeout(
        connect=REQUEST_CONNECT_TIMEOUT,
        read=STREAM_READ_TIMEOUT,
        write=STREAM_READ_TIMEOUT,
        pool=REQUEST_CONNECT_TIMEOUT,
    ),
    proxy=_default_proxy,
)

# ── App ───────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: Starlette):
    # Start background model discovery
    asyncio.ensure_future(_periodic_model_refresh())
    if PROXY_POOL_ENABLED:
        _log("Proxy pool enabled, loading SOCKS5 proxies in background...")
        asyncio.ensure_future(proxy_pool.load())
        _log("  (pool will be ready once verification completes)")
    yield
    await proxy_pool.close()
    await _default_client.aclose()
    await _stream_default_client.aclose()

app = Starlette(lifespan=_lifespan)


def _json(fn):
    """Wrap an endpoint so dict returns become JSONResponse (Starlette doesn't auto-encode)."""
    async def wrapper(request: Request):
        result = await fn(request)
        if isinstance(result, dict):
            return JSONResponse(result)
        return result
    return wrapper

PORT = args.port or int(os.environ.get("PORT", "6446"))
HOST = args.host or os.environ.get("HOST", "0.0.0.0")
OC_VERSION = "1.15.0"
PROXY_VERSION = "18"

# ── API Keys ──────────────────────────────────────────────────────

API_KEY = args.api_key or os.environ.get("LOCAL_KEY") or os.environ.get("API_KEY")


def auth(request: Request) -> str | None:
    if not API_KEY:
        return "user"
    hdr = request.headers.get("authorization") or request.headers.get("x-api-key") or ""
    tok = hdr[7:] if hdr.startswith("Bearer ") else hdr
    if tok and secrets.compare_digest(tok, API_KEY):
        return "user"
    return None


# ── Helpers ───────────────────────────────────────────────────────

def _log(*a):
    msg = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
    if "[zen] OK" in msg or "buffered fallback" in msg:
        print(f"\x1b[32m{msg}\x1b[0m", flush=True)
    elif "Stream error 400" in msg or "400 diag" in msg:
        print(f"\x1b[33m{msg}\x1b[0m", flush=True)
    elif "[zen]" in msg:
        print(f"\x1b[31m{msg}\x1b[0m", flush=True)
    elif "[pool]" in msg:
        print(f"\x1b[33m{msg}\x1b[0m", flush=True)
    else:
        print(msg, flush=True)
    with open(_BASE_DIR / "proxy.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def oc_id(prefix: str) -> str:
    ts = format(int(time.time() * 1000), "x")
    rnd = secrets.token_urlsafe(12)[:16]
    return f"{prefix}_{ts}{rnd}"


_NO_CACHE = {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ── Token usage tracking (persisted across runs) ──────────────────
# tokens.json maps model id -> {input, output, cache_hit, cache_miss}.

_TOKENS_FILE = _BASE_DIR / "tokens.json"
_DEFAULT_TOKENS = {"input": 0, "output": 0, "cache_hit": 0, "cache_miss": 0}


def _load_tokens():
    """Load per-model token usage from disk.

    Migrates the legacy flat-file shape (a single bucket at the top level)
    into an "unknown" bucket, since those totals predate per-model tracking
    and their originating model is not recorded.
    """
    global _tokens
    try:
        with open(_TOKENS_FILE, encoding="utf-8") as f:
            raw = json.load(f) or {}
    except Exception:
        raw = {}

    if isinstance(raw, dict) and "input" in raw:
        # Legacy flat shape -> single "unknown" bucket.
        _tokens = {
            "unknown": {
                key: (raw.get(key) if isinstance(raw.get(key), int) else 0)
                for key in ("input", "output", "cache_hit", "cache_miss")
            }
        }
        _persist_tokens()
    elif isinstance(raw, dict):
        _tokens = {}
        for m, bucket in raw.items():
            if isinstance(bucket, dict):
                _tokens[m] = {
                    **_DEFAULT_TOKENS,
                    **{k: int(v or 0) for k, v in bucket.items() if k in _DEFAULT_TOKENS},
                }
    else:
        _tokens = {}


def _persist_tokens():
    try:
        _TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_TOKENS_FILE, "w", encoding="utf-8") as f:
            json.dump(_tokens, f, indent=2, sort_keys=True)
    except OSError:
        pass


def _add_tokens(model: str, inp: int = 0, out: int = 0, cache_hit: int = 0, cache_miss: int = 0):
    """Accumulate usage counters under the given model id."""
    model = _normalize_model(model) or "unknown"
    bucket = _tokens.setdefault(model, dict(_DEFAULT_TOKENS))
    bucket["input"] += max(0, inp or 0)
    bucket["output"] += max(0, out or 0)
    bucket["cache_hit"] += max(0, cache_hit or 0)
    bucket["cache_miss"] += max(0, cache_miss or 0)
    _persist_tokens()


_load_tokens()


# ── reasoning_content persistence ────────────────────────────────
# Some Zen upstream models run in a "thinking mode" that REQUIRES the
# assistant's `reasoning_content` to be passed back verbatim on every later
# turn of the same session:
#   "The `reasoning_content` in the thinking mode must be passed back to the API."
# Clients (e.g. Pi) often echo only the final `content` of an assistant turn and
# drop the thinking, which makes the upstream reject the whole request with a
# 400. We capture the `reasoning_content` the upstream emits per session, keyed
# by a hash of the final content, and restore it onto assistant messages that
# lack it before re-forwarding them.
_REASONING_CACHE_MAX = 200  # reasoning entries kept per session
_REASONING_CACHE_MAX_SESSIONS = 500  # bound total sessions on disk
_reasoning_file = _BASE_DIR / "reasoning_cache.json"
_reasoning_cache: dict[str, dict[str, str]] = {}


def _load_reasoning():
    global _reasoning_cache
    try:
        with open(_reasoning_file, encoding="utf-8") as f:
            raw = json.load(f) or {}
        _reasoning_cache = {k: dict(v) for k, v in raw.items() if isinstance(v, dict)}
    except Exception:
        _reasoning_cache = {}


def _save_reasoning():
    """Persist the reasoning cache without blocking the event loop.

    Snapshots the cache (so the writer thread never races live mutations) and
    runs the JSON dump in a thread-pool executor; the write is fire-and-forget.
    """
    body = {sid: dict(sack) for sid, sack in _reasoning_cache.items()}

    def _write():
        try:
            _reasoning_file.parent.mkdir(parents=True, exist_ok=True)
            with open(_reasoning_file, "w", encoding="utf-8") as f:
                json.dump(body, f, ensure_ascii=False)
        except OSError:
            pass

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _write()  # no running loop (startup path): write inline
        return
    loop.run_in_executor(None, _write)


def _remember_reasoning(session_id, content, reasoning):
    """Store reasoning_content emitted for a given assistant content hash."""
    if not session_id or not content or not reasoning:
        return
    key = hashlib.sha256(content.encode("utf-8")).hexdigest()
    sack = _reasoning_cache.get(session_id)
    if not isinstance(sack, dict):
        sack = {}
        _reasoning_cache[session_id] = sack
    sack[key] = reasoning
    if len(sack) > _REASONING_CACHE_MAX:
        for old in list(sack)[: len(sack) - _REASONING_CACHE_MAX]:
            sack.pop(old, None)
    # Bound total sessions (dict is insertion-ordered, so drop the oldest) to
    # keep the on-disk cache from growing without bound across many users.
    if len(_reasoning_cache) > _REASONING_CACHE_MAX_SESSIONS:
        overflow = len(_reasoning_cache) - _REASONING_CACHE_MAX_SESSIONS
        for sid in list(_reasoning_cache)[:overflow]:
            _reasoning_cache.pop(sid, None)
    _save_reasoning()


_load_reasoning()


def _exc_desc(e: Exception | None) -> str:
    """Human-readable exception for logs: type chain + messages + repr fallback."""
    if e is None:
        return "unknown"
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = e
    depth = 0
    while cur is not None and id(cur) not in seen and depth < 4:
        seen.add(id(cur))
        t = type(cur).__name__
        msg = str(cur).strip()
        # httpx often wraps an empty ConnectError around an OS-level error; str() may be "".
        # Fall back to args/repr so the log is not just "ConnectError (caused by ConnectError)".
        if not msg:
            if getattr(cur, "args", None):
                try:
                    msg = repr(cur.args[0]) if len(cur.args) == 1 else repr(cur.args)
                except Exception:
                    msg = ""
            if not msg or msg in ("()", "''", '""'):
                try:
                    msg = repr(cur)
                except Exception:
                    msg = ""
                # Trim verbose httpx repr to the first line
                if "\n" in msg:
                    msg = msg.splitlines()[0]
                if len(msg) > 400:
                    msg = msg[:400] + "…"
        seg = t if not msg else f"{t}: {msg}"
        # Attach request URL if httpx error carries it (ConnectError.request etc.)
        req = getattr(cur, "request", None)
        if req is not None:
            try:
                url = getattr(req, "url", None)
                if url:
                    seg += f" req={url}"
            except Exception:
                pass
        parts.append(seg)
        nxt = getattr(cur, "__cause__", None)
        if nxt is None:
            nxt = getattr(cur, "__context__", None)
        # Avoid infinite loop on self-referential context
        if nxt is cur:
            break
        cur = nxt
        depth += 1
    return " (caused by ".join(parts) + ")" * (len(parts) - 1) if len(parts) > 1 else parts[0]

def _first_chunk_error(raw_line: str) -> tuple[str, bool] | None:
    """Classify an SSE chunk that is an upstream error payload.

    Returns (message, is_rate_limit) or None for normal chunks. Only genuine
    rate-limit markers (FreeUsageLimitError / 429 / rate_limit|free_usage|
    usage_limit|quota type or code) count as rate limits — any other upstream
    error (e.g. a 503 queue-full body) is NOT the proxy's fault and must not
    flag it.
    """
    trimmed = raw_line.strip()
    if trimmed.startswith("data: "):
        trimmed = trimmed[6:].strip()
    if not trimmed or trimmed == "[DONE]" or not trimmed.startswith("{"):
        return None
    try:
        parsed = json.loads(trimmed)
    except json.JSONDecodeError:
        return None
    if "FreeUsageLimitError" in trimmed or parsed.get("error") or parsed.get("type") == "error":
        err = parsed.get("error") or {}
        if isinstance(err, dict):
            msg = err.get("message") or parsed.get("message") or "Upstream error"
            kind = f"{err.get('code', '')} {err.get('type', '')}"
        else:
            msg = parsed.get("message") or str(err) or "Upstream error"
            kind = ""
        # type/code may sit on the error object or on the payload root
        kind += f" {parsed.get('code', '')} {parsed.get('type', '')}"
        is_rate_limit = (
            "FreeUsageLimitError" in trimmed
            or "429" in trimmed
            or "rate limit" in msg.lower()
            or any(k in kind.lower() for k in ("rate_limit", "free_usage", "usage_limit", "quota"))
        )
        return msg, is_rate_limit
    return None


def _openai_stream_error(
    message: str,
    error_type: str = "upstream_error",
    code: str | None = None,
) -> str:
    """Return a terminal OpenAI SSE sequence that always carries finish_reason.

    The wire error object is emitted first for compatibility, then a normal
    assistant chunk carries ``message`` as content and closes with
    ``finish_reason: "stop"`` followed by ``[DONE]``. Clients that hard-fail
    when a stream ends without any finish_reason (observed as "OpenAI
    completions stream closed before a finish_reason was received") instead
    receive a completed turn whose text explains the failure.
    """
    error = {"message": message, "type": error_type}
    if code:
        error["code"] = code
    cid = oc_id("chatcmpl")
    now = int(time.time())
    content_chunk = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": now,
        "model": "",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": f"[upstream error] {message}"},
                "finish_reason": None,
            }
        ],
    }
    stop_chunk = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": now,
        "model": "",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return (
        f"data: {json.dumps({'error': error})}\n\n"
        f"data: {json.dumps(content_chunk)}\n\n"
        f"data: {json.dumps(stop_chunk)}\n\n"
        "data: [DONE]\n\n"
    )


def _stream_preview(value: str, limit: int = 160) -> str:
    """Return a bounded, escaped preview suitable for diagnostics."""
    return repr(value[:limit])


def _is_context_limit_error(data: dict | None = None, body: str = "") -> bool:
    """Recognize deterministic upstream context-size errors.

    Providers do not use one consistent error code: some return
    ``context_length_exceeded`` while others label the same 400 as
    ``invalid_request_error`` and put the useful signal in the message.
    These errors cannot be fixed by changing proxies, so retrying them only
    repeats the same request and needlessly delays the caller.
    """
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        error = {}
    text = " ".join(
        str(value)
        for value in (
            error.get("code"),
            error.get("type"),
            error.get("message"),
            body,
        )
        if value is not None
    ).lower()
    return (
        "context_length_exceeded" in text
        or "maximum context length" in text
        or "max context length" in text
        or "too many tokens" in text
        or ("requested" in text and "tokens" in text and "prompt" in text)
    )
def _is_region_error(data: dict | None = None, body: str = "") -> bool:
    """Recognize geo-restriction errors that depend on the proxy's exit country.

    Unlike context/rate errors, a ``RegionError`` is proxy-specific: the same
    request routed through a proxy in an allowed country succeeds. It must be
    treated as a proxy failure (blacklist + rotate), not a terminal upstream
    error that repeats identically on every proxy.
    """
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        error = {}
    etype = str(error.get("type") or "").lower()
    if "region" in etype:
        return True
    text = " ".join(
        str(v) for v in (error.get("message"), body) if v is not None
    ).lower()
    return "not available in your country" in text


def _is_promotion_ended_error(data: dict | None = None, body: str = "") -> bool:
    """Recognize entitlement errors for a model whose free promotion ended.

    Observed as ``{"type":"ModelError","message":"Free promotion has ended
    for ... Free"}`` (HTTP 401). This repeats identically on every proxy and
    every retry — it is account/entitlement level, not transport level — so
    retrying is pure waste. The model is dead until upstream re-lists it.
    """
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        error = {}
    etype = str(error.get("type") or "").strip().lower()
    text = " ".join(
        str(v) for v in (error.get("message"), body) if v is not None
    ).lower()
    return "promotion has ended" in text or etype == "modelerror"


def _blocks_text(content) -> str:
    """Extract plain text from Anthropic content blocks (str or list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content) if content is not None else ""


# Free models are always discovered live from the Zen API — no hardcoded
# fallback list. The cache starts empty and fills within moments of startup.
_models_cache: list[str] = []
_models_meta: dict[str, dict] = {}  # model_id -> {name, limit, modalities}
_dead_models: set[str] = set()  # ids rejected upstream (free promotion ended); excluded from rediscovery
_MODELS_REFRESH_SECS = 43200  # safety-net refresh every 12h; unknown models trigger on-demand


async def _fetch_free_models():
    """Query Zen API + models.dev, merge free model list with context limits."""
    global _models_cache, _models_meta
    try:
        async with httpx.AsyncClient() as c:
            # 1. Get available models from Zen API
            r = await c.get(
                "https://opencode.ai/zen/v1/models",
                headers={"User-Agent": f"opencode/{OC_VERSION}", "x-opencode-client": "cli"},
                timeout=10,
            )
            if r.status_code != 200:
                _log(f"[models] Zen API returned {r.status_code}, keeping cached models")
                return
            data = r.json()
            all_models = [m["id"] for m in data.get("data", []) if isinstance(m, dict)]
            free = [m for m in all_models if "free" in m.lower() and m not in _dead_models]
            if not free:
                _log("[models] No free models found in Zen API, keeping cached")
                return

            # 2. Fetch context limits from models.dev
            try:
                md = await c.get("https://models.dev/api.json", timeout=10)
                if md.status_code == 200:
                    md_data = md.json()
                    oc_models = md_data.get("opencode", {}).get("models", {})
                    meta = {}
                    for mid in free:
                        entry = oc_models.get(mid)
                        if entry:
                            meta[mid] = {
                                "name": entry.get("name"),
                                "limit": entry.get("limit"),
                                "modalities": entry.get("modalities"),
                            }
                    _models_meta = meta
                    _log(f"[models] Loaded metadata for {len(meta)} models from models.dev")
            except Exception as e:
                _log(f"[models] models.dev fetch failed: {e}, metadata may be missing")

            _models_cache = free
            _log(f"[models] Discovered {len(free)} free models: {', '.join(free)}")
    except Exception as e:
        _log(f"[models] Fetch failed: {e}, keeping cached models")






async def _periodic_model_refresh():
    """Refresh the free model list on a slow cadence as a safety net.

    Primary discovery happens at startup and on demand when a request names
    an unknown model; this timer only catches models that appear and are
    never requested.
    """
    await _fetch_free_models()
    while True:
        await asyncio.sleep(_MODELS_REFRESH_SECS)
        await _fetch_free_models()

async def _ensure_model_known(model: str) -> bool:
    """Return True once ``model`` is in the discovered free list.

    Unknown ids trigger an immediate discovery pass so a freshly published
    free model is usable on the very next request instead of waiting for the
    periodic refresh.
    """
    if model in _dead_models:
        return False
    if model in _models_cache:
        return True
    await _fetch_free_models()
    return model in _models_cache


def _mark_model_dead(model: str | None) -> bool:
    """Drop a model the upstream refuses with a promotion-ended ModelError.

    Removes it from the served list so clients get an immediate, clear
    "Unknown model" instead of per-request 401s, and keeps it out of future
    discovery passes. Returns True if this call retired it.
    """
    global _models_cache, _models_meta
    if not model or model in _dead_models:
        return False
    _dead_models.add(model)
    if model in _models_cache:
        _models_cache = [m for m in _models_cache if m != model]
        _models_meta.pop(model, None)
    _log(f"[models] Retired {model} (upstream: free promotion ended)")
    return True


def _normalize_model(model: str) -> str:
    """Normalize a client model id to the bare Zen upstream id.

    opencode sends ids like `opencode-local/muse-spark-1.2-contributor-free:high`:
    `opencode-local/` is the provider prefix and `:high` is the reasoning-effort
    suffix. Both are stripped, as is the legacy `ocf-` prefix.
    """
    if not model:
        return model
    if "/" in model:
        model = model.rsplit("/", 1)[-1]
    if ":" in model:
        model = model.split(":", 1)[0]
    if model.startswith("ocf-"):
        model = model[4:]
    return model


def _normalize_role(role: str) -> str:
    """Map newer OpenAI roles to variants accepted by the Zen upstream."""
    # `developer` is the newer OpenAI system-prompt role; Zen only accepts
    # `system`, `user`, `assistant`, `tool`, `latest_reminder`.
    return "system" if role == "developer" else role


def _normalize_messages(messages: list[dict]) -> list[dict] | None:
    """Return a shallow copy of messages with roles normalized for Zen."""
    if not messages:
        return messages
    out = []
    for m in messages:
        if isinstance(m, dict) and m.get("role"):
            m = dict(m)
            m["role"] = _normalize_role(m["role"])
        out.append(m)
    return out


def _assistant_message(m) -> dict:
    """Normalize one assistant message into the upstream thinking-mode shape:
    `reasoning_content` as a top-level string field plus `content` as text.
    Handles clients that send thinking as content-block(s) or a `reasoning` field.
    """
    m = dict(m)
    raw_content = m.get("content")
    reasoning = m.get("reasoning_content") or m.get("reasoning")
    new_content = raw_content

    if isinstance(raw_content, list):
        texts = []
        for b in raw_content:
            if not isinstance(b, dict):
                texts.append(str(b))
                continue
            btype = (b.get("type") or "").lower()
            if btype == "text":
                texts.append(b.get("text") or "")
            elif btype in ("reasoning", "reasoning_content", "thinking", "analysis_tokens", "analysis"):
                rt = b.get("text") or b.get("reasoning_content") or b.get("content") or ""
                if rt:
                    reasoning = (reasoning or "") + rt
            else:
                texts.append(b.get("text") or str(b))
        new_content = "\n".join(t for t in texts if t) if texts else None

    # Preserve every field (e.g. tool_calls) and only rewrite content/reasoning
    out = dict(m)
    out["role"] = "assistant"
    out["content"] = new_content
    out.pop("reasoning", None)
    if reasoning:
        out["reasoning_content"] = reasoning
    else:
        out.pop("reasoning_content", None)
    return out


def _prepare_upstream_messages(session_id, messages: list[dict]) -> list[dict]:
    """Shape the message array for the Zen thinking-mode upstream.

    - Collapse consecutive assistant messages into one message carrying both
      `content` and `reasoning_content` (some clients split a reasoning turn).
    - Re-inject the cached `reasoning_content` for any assistant message that
      only has `content`, so the upstream's thinking-mode validation passes.
    """
    if not messages:
        return messages
    out: list[dict] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if not isinstance(m, dict) or m.get("role") != "assistant":
            out.append(m)
            i += 1
            continue

        combined = _assistant_message(m)
        # Merge any immediately following assistant messages (reasoning + answer split)
        j = i + 1
        while j < n and isinstance(messages[j], dict) and messages[j].get("role") == "assistant":
            part = _assistant_message(messages[j])
            part_content = part.get("content")
            if part_content:
                combined["content"] = (combined.get("content") or "") + ("\n" if combined.get("content") else "") + part_content
            if part.get("reasoning_content") and not combined.get("reasoning_content"):
                combined["reasoning_content"] = part["reasoning_content"]
            if part.get("tool_calls") and not combined.get("tool_calls"):
                combined["tool_calls"] = part["tool_calls"]
            j += 1

        # Inject the cached thinking text for this turn if the client dropped it
        content = combined.get("content")
        if (
            not combined.get("reasoning_content")
            and isinstance(content, str)
            and content
            and session_id
        ):
            key = hashlib.sha256(content.encode("utf-8")).hexdigest()
            cached = _reasoning_cache.get(session_id, {}).get(key)
            if cached:
                combined["reasoning_content"] = cached

        # The thinking-mode upstream rejects ANY assistant message that lacks
        # the `reasoning_content` field (even if the client omitted the
        # thinking). A turn with no thinking legitimately carries an empty
        # string, so guarantee the key is present on every assistant message.
        if "reasoning_content" not in combined:
            combined["reasoning_content"] = ""

        out.append(combined)
        i = j
    return out


# Session per conversation (hash-based lookup)
_user_sessions: dict[str, dict[str, str]] = {}
_MAX_SESSIONS_PER_USER = 500


def _remember_session(sessions: dict[str, str], key: str, value: str):
    sessions[key] = value
    while len(sessions) > _MAX_SESSIONS_PER_USER:
        sessions.pop(next(iter(sessions)))


def _hash_messages(messages: list[dict]) -> str:
    parts = []
    for m in (messages or []):
        role = m.get("role", "")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        parts.append(f"{role}:{content}")
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:16]


def get_session(user: str, messages: list[dict]) -> str:
    if user not in _user_sessions:
        _user_sessions[user] = {}
    sessions = _user_sessions[user]

    for n in range(len(messages), 0, -1):
        h = _hash_messages(messages[:n])
        if h in sessions:
            full_h = _hash_messages(messages)
            _remember_session(sessions, full_h, sessions[h])
            return sessions[h]

    new_id = f"ses_{oc_id('ses')}"
    full_h = _hash_messages(messages)
    _remember_session(sessions, full_h, new_id)
    return new_id


def force_new_session(user: str, messages: list[dict]) -> str:
    new_id = f"ses_{oc_id('ses')}"
    if user not in _user_sessions:
        _user_sessions[user] = {}
    full_h = _hash_messages(messages)
    _remember_session(_user_sessions[user], full_h, new_id)
    return new_id


async def _backoff(attempt: int, base: float = 1.0, max_delay: float = 10.0):
    """Exponential backoff with jitter."""
    delay = min(base * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.25)
    await asyncio.sleep(delay + jitter)


async def _client_gone(request: Request) -> bool:
    """True if the requesting client has disconnected (so retries stop)."""
    try:
        return await request.is_disconnected()
    except Exception:
        return False


def _local_rate_limit_response(message: str) -> JSONResponse:
    """Return promptly so the caller can retry through another proxy."""
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": message,
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
            }
        },
    )


# ── Zen API transport ─────────────────────────────────────────────

# Mid-stream transport deaths (torn tunnel / incomplete chunked read) are
# resumed by re-issuing the request with the partially-streamed assistant
# output appended, so the model continues from the cutoff instead of
# replaying the turn. Bound resume attempts separately from MAX_RETRIES.
MAX_STREAM_CONTINUATIONS = 4  # torn-stream resumes; long reasoning models need headroom


def _needs_buffered_fallback(model: str | None, tools, stream) -> bool:
    """Buffered bridge for models whose streaming truncates.

    Muse Spark 1.2 via zen/go truncates streaming tool-calls (opencodex#2156
    / opencode#40888). x-preview-f-free now shows the same: reasoning=2-7k
    then streamed_any=True reason_len=3k+ content_len=0 and no [DONE].
    Same safe fix: model-scoped stream:false upstream, reframe as SSE,
    rather than loosening the global fail-closed guard.
    TODO: narrow x-preview gate to reasoning-heavy turns once upstream fixes
    stream:true without tools; currently buffered on every stream:true.
    """
    if not stream:
        return False
    m = (model or "").lower()
    # Gate: muse-spark only when tools are present (stream:true+tools matrix);
    # x-preview-f truncates even on plain reasoning turns, so buffer it whenever streaming.
    if "muse-spark" in m or "muse_spark" in m:
        return bool(tools)
    if "x-preview" in m or "x_preview" in m:
        return True
    return False

def _buffered_to_openai_sse(data: dict, model: str):
    """Reframe a buffered chat/completions object as OpenAI SSE chunks."""
    choice = (data.get("choices") or [{}])[0] or {}
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    raw_finish = choice.get("finish_reason")
    finish = raw_finish if raw_finish is not None else ("tool_calls" if tool_calls else "stop")
    usage = data.get("usage") or {}
    cid = data.get("id") or oc_id("chatcmpl")
    created = data.get("created") or int(time.time())
    # Tokens already counted by _zen_request_with_retry buffered path; do not double-count.
    # role chunk
    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
    if content:
        # Keep one delta; chunking is not required for correctness
        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"
    for idx, tc in enumerate(tool_calls):
        fn = tc.get("function") or {}
        args = fn.get("arguments") or ""
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        # tool_calls delta (OpenAI streaming shape)
        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': idx, 'id': tc.get('id') or oc_id('call'), 'type': 'function', 'function': {'name': fn.get('name') or '', 'arguments': args}}]}, 'finish_reason': None}]})}\n\n"
    # finish chunk
    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]})}\n\n"
    if usage:
        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [], 'usage': usage})}\n\n"
    yield "data: [DONE]\n\n"

def _buffered_to_anthropic_sse(data: dict, model: str, input_tokens: int):
    """Reframe a buffered chat/completions object as Anthropic Messages SSE."""
    choice = (data.get("choices") or [{}])[0] or {}
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    raw_finish = choice.get("finish_reason")
    finish = raw_finish if raw_finish is not None else ("tool_calls" if tool_calls else "stop")
    usage = data.get("usage") or {}
    # Tokens already counted by _zen_request_with_retry buffered path; do not double-count.
    msg_id = oc_id("msg")
    # message_start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'usage': {'input_tokens': input_tokens or 0, 'output_tokens': 0, **_NO_CACHE}}})}\n\n"
    idx = 0
    has_text = bool(content)
    if has_text:
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        idx = 1
    for tc in tool_calls:
        fn = tc.get("function") or {}
        args = fn.get("arguments") or ""
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': {'type': 'tool_use', 'id': tc.get('id') or oc_id('toolu'), 'name': fn.get('name') or ''}})}\n\n"
        if args:
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'input_json_delta', 'partial_json': args}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"
        idx += 1
    if not has_text and not tool_calls:
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    output_tokens = usage.get("completion_tokens") if usage.get("completion_tokens") is not None else (len(content) // 4 + sum(len((tc.get('function') or {}).get('arguments') or '') for tc in tool_calls) // 4) or 0
    stop_reason = "tool_use" if finish == "tool_calls" else ("max_tokens" if finish == "length" else "end_turn")
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason}, 'usage': {'output_tokens': output_tokens}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


def _continuation_body(req_body: dict, content: str, reasoning: str) -> dict:
    """Clone the request body with the partial assistant output appended."""
    body = dict(req_body)
    msgs = list(body.get("messages") or [])
    if content:
        msgs.append({
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning or "",
        })
    body["messages"] = msgs
    return body


_SAMPLING_KEYS = ("temperature", "top_p", "stop")


def _sampling_from(body: dict) -> dict:
    """Pick client sampling params the upstream accepts; absent = upstream default."""
    return {k: body[k] for k in _SAMPLING_KEYS if body.get(k) is not None}


def zen_request(model, messages, stream, tools, tool_choice, session_id, max_tokens=None, max_completion_tokens=None, sampling: dict | None = None):
    model = _normalize_model(model)
    req_body: dict = {"model": model, "messages": messages, "stream": bool(stream)}
    if tools:
        req_body["tools"] = tools
    if tool_choice:
        req_body["tool_choice"] = tool_choice
    if max_tokens is not None:
        req_body["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        req_body["max_completion_tokens"] = max_completion_tokens
    if sampling:
        req_body.update(sampling)

    request_id = oc_id("msg")
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer public",
        "User-Agent": f"opencode/{OC_VERSION} ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.13",
        "x-opencode-client": "cli",
        "x-opencode-project": "global",
        "x-opencode-request": request_id,
        "x-opencode-session": session_id,
    }
    return req_body, headers


# ── Proxy-aware Zen API calls ─────────────────────────────────────

def _needs_stream_bridge(model: str | None) -> bool:
    """True for models whose upstream hangs on fully-buffered generations.

    x-preview reasons for minutes before emitting its first byte; the
    non-streaming transport (short read timeout, no keep-alive traffic) dies
    with ReadTimeout long before the answer materializes, while the streaming
    transport survives the silent warm-up. Clients asking stream=false for
    such models get an internal bridge: stream upstream, aggregate, return
    one chat.completion object.
    """
    if not model:
        return False
    m = str(model)
    return any(alias in m for alias in ("x-preview", "hy3"))


async def _aggregate_upstream_completion(
    request: Request,
    req_body: dict,
    headers: dict,
    user: str,
    messages: list[dict],
    session_id: str = None,
    model: str = None,
    max_retries: int = None,
):
    """Stream the upstream response and assemble a buffered chat.completion.

    Consumes ``_zen_stream_with_retry`` (which already owns proxy-pool
    selection, retries, torn-stream resume and terminal-error framing) and
    folds the emitted deltas into the same JSON object the non-streaming
    transport would have returned.
    """
    stream_body = dict(req_body)
    stream_body["stream"] = True

    cid = oc_id("chatcmpl")
    created = int(time.time())
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason = None
    usage: dict | None = None
    saw_error = None

    gen = _zen_stream_with_retry(
        request, stream_body, headers, user, messages, session_id, model, max_retries
    )
    got_any_chunk = False
    async for raw in gen:
        for block in str(raw).split("\n\n"):
            block = block.strip()
            if not block.startswith("data:"):
                continue
            payload = block[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                piece = json.loads(payload)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            got_any_chunk = True
            if isinstance(piece.get("error"), dict):
                saw_error = piece["error"].get("message") or "Upstream stream error"
                continue
            choices = piece.get("choices")
            first_choice = choices[0] if isinstance(choices, list) and choices else {}
            if not isinstance(first_choice, dict):
                continue
            delta = first_choice.get("delta") or {}
            if not isinstance(delta, dict):
                delta = {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tool_calls.setdefault(
                    idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = slot["function"]["name"] + fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] = slot["function"]["arguments"] + fn["arguments"]
            if first_choice.get("finish_reason"):
                finish_reason = first_choice["finish_reason"]
            u = piece.get("usage")
            if isinstance(u, dict) and (u.get("prompt_tokens") or u.get("completion_tokens")):
                usage = u

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    has_tool_calls = bool(tool_calls)

    # Terminal failures: nothing usable was produced. Surface the upstream
    # error exactly like the buffered transport would (502 + error JSON).
    # _openai_stream_error frames errors as literal "[upstream error] …"
    # content; strip that framing so callers never see it as an answer.
    if saw_error and not has_tool_calls and content.startswith("[upstream error]"):
        msg = content[len("[upstream error] "):]
        return JSONResponse(
            status_code=502,
            content={"error": {"message": msg, "type": "upstream_error"}},
        )
    if not got_any_chunk:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "Upstream stream produced no data", "type": "upstream_error"}},
        )
    # Reasoning-only with no answer text: the model spent its whole output
    # budget thinking. Not a transport failure — return a well-formed
    # completion carrying finish_reason (usually "length") plus whatever
    # reasoning survived, so clients can raise max_tokens and retry.
    if not content and not has_tool_calls:
        _log(f"[zen] [{model}|stream-bridge] reasoning-only response, finish={finish_reason} ({len(reasoning)} chars of thinking)")
        usage = usage or {
            "prompt_tokens": len(json.dumps(messages)) // 4,
            "completion_tokens": len(reasoning) // 4,
            "completion_tokens_details": {"reasoning_tokens": len(reasoning) // 4},
        }
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model or "",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "", "reasoning_content": reasoning},
                    "finish_reason": finish_reason or "length",
                }
            ],
            "usage": usage,
        }

    if session_id and reasoning and content:
        _remember_reasoning(session_id, content, reasoning)

    message: dict = {"role": "assistant"}
    if content:
        message["content"] = content
    elif has_tool_calls:
        message["content"] = None
    if reasoning:
        message["reasoning_content"] = reasoning
    ordered_calls = [tool_calls[i] for i in sorted(tool_calls)]
    if ordered_calls:
        message["tool_calls"] = [
            {
                "id": c["id"] or oc_id("call"),
                "type": "function",
                "function": {
                    "name": c["function"]["name"],
                    "arguments": c["function"]["arguments"],
                },
            }
            for c in ordered_calls
        ]
        finish_reason = finish_reason or "tool_calls"

    if not usage:
        usage = {
            "prompt_tokens": len(json.dumps(messages)) // 4,
            "completion_tokens": (len(content) + len(reasoning)) // 4,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": len(reasoning) // 4},
        }

    _add_tokens(
        model or "unknown",
        usage.get("prompt_tokens") or 0,
        usage.get("completion_tokens") or 0,
        usage.get("prompt_cache_hit_tokens") or (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
        usage.get("prompt_cache_miss_tokens") or 0,
    )

    _log(f"[zen] OK [{model}|stream-bridge] aggregated {len(content)} chars, finish={finish_reason}")
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model or "",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or "stop",
            }
        ],
        "usage": usage,
    }


async def _zen_request_with_retry(
    request: Request,
    req_body: dict,
    headers: dict,
    user: str,
    messages: list[dict],
    session_id: str = None,
    model: str = None,
    max_retries: int = None,
):
    """Non-streaming Zen API call with proxy pool retry on 429."""
    if req_body.get("stream") is False and _needs_stream_bridge(model):
        # Buffered transport times out on this model's silent warm-up; stream
        # internally and hand the caller a normal completion object instead.
        return await _aggregate_upstream_completion(
            request, req_body, headers, user, messages, session_id, model, max_retries
        )
    last_error = None
    attempts = MAX_RETRIES if max_retries is None else max_retries

    for attempt in range(attempts + 1):
        if await _client_gone(request):
            _log("[zen] Client disconnected; aborting retries")
            return None
        if PROXY_POOL_ENABLED:
            # Load pool if needed
            if not proxy_pool.ready:
                await proxy_pool.load()

            if not await proxy_pool.select():
                _log(f"[pool] No proxy available ({proxy_pool.get_pool_state()}), "
                     f"forcing refresh")
                await proxy_pool.force_refresh()
                if not await proxy_pool.select():
                    _log("[pool] Still no proxy after refresh, falling back to direct")
                    client = _default_client
                    proxy_addr = None
                else:
                    p = proxy_pool.current
                    proxy_addr = p["address"]
                    client = proxy_pool.get_client(f"socks5://{proxy_addr}")
                    _log(f"[pool] Retry {attempt}: using proxy {proxy_addr}")
            else:
                p = proxy_pool.current
                proxy_addr = p["address"]
                client = proxy_pool.get_client(f"socks5://{proxy_addr}")

        else:
            client = _default_client
            proxy_addr = None

        try:
            resp = await client.post(
                "/zen/v1/chat/completions",
                json=req_body,
                headers=headers,
            )
        except Exception as e:
            _log(f"[zen] Request failed (attempt {attempt}): {_exc_desc(e)}")
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_failure(proxy_addr)
            last_error = e
            if attempt < attempts:
                await _backoff(attempt)
            continue

        try:
            body_bytes = await resp.aread()
        except Exception as e:
            _log(f"[zen] Response read failed (attempt {attempt}): {_exc_desc(e)}")
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_failure(proxy_addr)
            last_error = e
            if attempt < attempts:
                await _backoff(attempt)
                continue
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"Upstream response read failed: {_exc_desc(e)}", "type": "upstream_error"}},
            )
        if PROXY_POOL_ENABLED and proxy_addr:
            proxy_pool.report_success(proxy_addr)
        body_text = body_bytes.decode("utf-8", errors="replace")
        try:
            data = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {}

        is_429 = resp.status_code == 429
        is_rate_limit = is_429 or "FreeUsageLimitError" in body_text

        if is_rate_limit:
            err_msg = (data.get("error") or {}).get("message") or "Rate limit exceeded"
            _log(f"[zen] 429 (attempt {attempt}): {err_msg}")
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_ratelimit(proxy_addr)
            return _local_rate_limit_response(err_msg + " (free model rate limit)")

        if resp.status_code >= 400:
            err_msg = (data.get("error") or {}).get("message") or f"HTTP {resp.status_code}"
            is_context_exceeded = _is_context_limit_error(data, body_text)
            is_region_blocked = _is_region_error(data, body_text)
            _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Error {resp.status_code}: {err_msg} tools={'yes' if req_body.get('tools') else 'no'} stream={req_body.get('stream')} body={body_text[:300]!r}")
            if resp.status_code == 400:
                _log_reasoning_diag(req_body)
                _dump_400_body(req_body, body_text, proxy_addr)
            if is_region_blocked:
                # Geo-restriction is proxy-specific: blacklist this proxy and
                # rotate to a different one on the next attempt.
                if PROXY_POOL_ENABLED and proxy_addr:
                    proxy_pool.report_region_block(proxy_addr)
                _log(f"[zen] Region-blocked via {proxy_addr}; rotating proxy")
                if attempt < attempts:
                    await _backoff(attempt)
                    continue
                return JSONResponse(
                    status_code=resp.status_code,
                    content={"error": {"message": err_msg, "type": "upstream_error"}},
                )
            # Not a proxy failure in general — a 503 "Endpoint is
            # unavailable" from the Console is upstream capacity-shaped and
            # observed to clear on retry, sometimes only after several
            # attempts through the SAME exit (streaks up to 14). Keep the
            # current proxy (sticky selection) so retries reuse the healthy
            # path instead of hopping between exits; never blacklist.
            # Retry only 5xx (transient upstream). 4xx are request problems:
            # a rejected body ([1210] Invalid API parameter) never heals via
            # backoff or proxy rotation, so fail fast like context-limit.
            if resp.status_code >= 500 and not is_context_exceeded and attempt < attempts:
                await _backoff(attempt)
                continue
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": {"message": err_msg, "type": "upstream_error"}},
            )

        usage = (data.get("usage") or {})
        if isinstance(usage, dict) and ("prompt_tokens" in usage or "completion_tokens" in usage):
            _add_tokens(model,
                usage.get("prompt_tokens") or 0,
                usage.get("completion_tokens") or 0,
                usage.get("prompt_cache_hit_tokens") or (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                usage.get("prompt_cache_miss_tokens") or 0,
            )

        # Remember emitted reasoning for future turns of this session
        _msg = ((data.get("choices") or [{}])[0] or {}).get("message") or {}
        if session_id and _msg.get("content") and (_msg.get("reasoning_content") or _msg.get("reasoning")):
            _remember_reasoning(
                session_id,
                _msg["content"] if isinstance(_msg["content"], str) else "",
                _msg.get("reasoning_content") or _msg.get("reasoning") or "",
            )
        _log(f"[zen] OK [{model}|{proxy_addr or 'direct'}] response complete")
        return data

    if last_error:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Upstream error after {attempts + 1} attempts: {_exc_desc(last_error)}", "type": "upstream_error"}},
        )
    return JSONResponse(
        status_code=502,
        content={"error": {"message": "Upstream request failed", "type": "upstream_error"}},
    )


async def _zen_stream_with_retry(
    request: Request,
    req_body: dict,
    headers: dict,
    user: str,
    messages: list[dict],
    session_id: str = None,
    model: str = None,
    max_retries: int = None,
):
    """Streaming Zen API call with proxy pool retry on 429/403 and ReadError recovery."""
    last_error = None
    attempts = MAX_RETRIES if max_retries is None else max_retries
    finish_delivered = False  # True once a finish_reason chunk was forwarded
    tool_streamed = False  # True once a tool-call fragment was forwarded
    continuations = 0  # mid-stream resume attempts (see MAX_STREAM_CONTINUATIONS)

    attempt = 0
    while True:
        # Total work cap: plain retries (attempts) + mid-stream continuations
        # (MAX_STREAM_CONTINUATIONS) share one counter, so 4 torn-stream resumes
        # consume attempt slots and can starve a subsequent connect retry.
        # This is intentional — documents bounded total work (attempts+1 + 4 max).
        if attempt > attempts + MAX_STREAM_CONTINUATIONS:
            break
        if await _client_gone(request):
            _log("[zen] Client disconnected; aborting retries")
            return
        if PROXY_POOL_ENABLED:
            if not proxy_pool.ready:
                await proxy_pool.load()

            if not await proxy_pool.select():
                _log(f"[pool] No proxy available ({proxy_pool.get_pool_state()}), forcing refresh")
                await proxy_pool.force_refresh()
                if not await proxy_pool.select():
                    _log("[pool] Still no proxy after refresh, falling back to direct")
                    client = _stream_default_client
                    proxy_addr = None
                else:
                    p = proxy_pool.current
                    proxy_addr = p["address"]
                    client = proxy_pool.get_client(
                        f"socks5://{proxy_addr}", streaming=True
                    )
            else:
                p = proxy_pool.current
                proxy_addr = p["address"]
                client = proxy_pool.get_client(
                    f"socks5://{proxy_addr}", streaming=True
                )
        else:
            client = _stream_default_client
            proxy_addr = None

        if client is None:
            err = RuntimeError(
                f"no http client (model={model} proxy={proxy_addr or 'direct'} "
                f"pool={proxy_pool.get_pool_state() if PROXY_POOL_ENABLED else 'disabled'} "
                f"attempt={attempt})"
            )
            _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Stream client is None: {_exc_desc(err)}")
            err.__cause__ = last_error
            last_error = err
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_failure(proxy_addr, hard=True)
            # Force a fresh selection next loop; also try direct fallback immediately
            if PROXY_POOL_ENABLED:
                proxy_pool.current = None
            client = _stream_default_client
            proxy_addr = None
            if client is None:
                yield _openai_stream_error(f"Stream client unavailable: {_exc_desc(err)}", "upstream_error", "transport_error")
                return
            _log(f"[zen] Retrying with direct client after None client (attempt {attempt})")
        try:
            upstream_request = client.build_request(
                "POST", "/zen/v1/chat/completions", json=req_body, headers=headers
            )
            resp = await client.send(upstream_request, stream=True)
        except Exception as e:
            _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Stream request failed (attempt {attempt} pool={proxy_pool.get_pool_state() if PROXY_POOL_ENABLED else 'disabled'}): {_exc_desc(e)}")
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_failure(
                    proxy_addr,
                    hard=isinstance(
                        e, (httpx.ConnectTimeout, httpx.ConnectError, httpx.ProxyError)
                    ),
                )
            last_error = e
            if attempt < attempts:
                await _backoff(attempt)
                attempt += 1
                continue
            yield _openai_stream_error(
                f"Stream request failed: {_exc_desc(e)}",
                "upstream_error",
                "transport_error",
            )
            return

        if resp.status_code == 429:
            try:
                raw = await resp.aread()
                try:
                    data = json.loads(raw)
                    err_msg = (data.get("error") or {}).get("message") or "Rate limit exceeded"
                except Exception:
                    err_msg = "Rate limit exceeded"
            except Exception as e:
                _log(f"[zen] Stream 429 read failed (attempt {attempt}): {_exc_desc(e)}")
                if PROXY_POOL_ENABLED and proxy_addr:
                    proxy_pool.report_failure(proxy_addr)
                await resp.aclose()
                if attempt < attempts:
                    await _backoff(attempt)
                    attempt += 1
                    continue
                yield _openai_stream_error(f"Upstream error: {_exc_desc(e)}")
                return
            _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Stream 429 (attempt {attempt}): {err_msg}")
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_ratelimit(proxy_addr)
            await resp.aclose()
            yield _openai_stream_error(
                err_msg + " (free model rate limit)",
                "rate_limit_error",
                "rate_limit_exceeded",
            )
            return

        if resp.status_code >= 400:
            try:
                raw = await resp.aread()
            except Exception as e:
                _log(f"[zen] Stream error body read failed (attempt {attempt}): {_exc_desc(e)}")
                if PROXY_POOL_ENABLED and proxy_addr:
                    proxy_pool.report_failure(proxy_addr)
                await resp.aclose()
                if attempt < attempts:
                    await _backoff(attempt)
                    attempt += 1
                    continue
                yield _openai_stream_error(f"Upstream error: {_exc_desc(e)}")
                return
            body_text = raw.decode("utf-8", errors="replace")
            body_preview = raw[:2000] if raw else b"<empty body>"
            if not raw:
                _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Stream error {resp.status_code}: empty body (headers try to diagnose proxy vs upstream)")
            try:
                data = json.loads(body_text)
            except (json.JSONDecodeError, TypeError, ValueError):
                data = {}
            err_msg = (data.get("error") or {}).get("message") or f"HTTP {resp.status_code}" + ("" if body_text.strip() else " (empty body)")
            is_context_exceeded = _is_context_limit_error(data, body_text)
            is_region_blocked = _is_region_error(data, body_text)
            _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Stream error {resp.status_code}: {body_preview!r} len={len(raw)}")
            if resp.status_code == 400:
                _log_reasoning_diag(req_body)

            if is_region_blocked:
                # Geo-restriction is proxy-specific: this proxy's exit country is
                # blocked for the model. Blacklist it and rotate to a different
                # proxy instead of retrying through the same (or another) blocked
                # region. Never record it as a success.
                if PROXY_POOL_ENABLED and proxy_addr:
                    proxy_pool.report_region_block(proxy_addr)
                _log(f"[zen] Region-blocked via {proxy_addr}; rotating proxy")
                await resp.aclose()
                if attempt < attempts:
                    await _backoff(attempt)
                    attempt += 1
                    continue
                yield _openai_stream_error(err_msg, "upstream_error", "region_blocked")
                return

            # Not a proxy failure: 4xx/5xx are upstream or request errors
            # that repeat identically on every proxy, so never blacklist for
            # them. A 503 is upstream capacity: keep the current exit and let
            # the retry below ride it out (no report_success — it was not
            # healthy for this call).
            if _is_promotion_ended_error(data, body_text):
                # Entitlement error: identical on every proxy/retry. Retire
                # the model and terminate the stream immediately.
                await resp.aclose()
                _mark_model_dead(model)
                yield _openai_stream_error(f"{model} is no longer available: {err_msg}", "upstream_error", "model_retired")
                return
            if is_context_exceeded:
                _log("[zen] Context limit error; not retrying on another proxy")
            await resp.aclose()
            if resp.status_code >= 500 and not is_context_exceeded and attempt < attempts:
                await _backoff(attempt)
                attempt += 1
                continue
            yield _openai_stream_error(err_msg, "upstream_error", str(resp.status_code))
            return

        # Success — stream the response
        retry_stream = False
        streamed_any = False
        stream_completed = False
        _reason_buf = ""
        _content_buf = ""
        try:
            try:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        err = ValueError("upstream stream event is not an SSE data event")
                        _log(
                            f"[zen] Malformed upstream stream (attempt {attempt}): "
                            f"{err}; raw={_stream_preview(line)}"
                        )
                        if PROXY_POOL_ENABLED and proxy_addr:
                            proxy_pool.report_failure(proxy_addr)
                        last_error = err
                        if not streamed_any and attempt < attempts and not await _client_gone(request):
                            retry_stream = True
                            break
                        yield _openai_stream_error(str(err), "upstream_error", "malformed_stream")
                        return

                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        if PROXY_POOL_ENABLED and proxy_addr:
                            proxy_pool.report_success(proxy_addr)
                        stream_completed = True
                        _log(f"[zen] OK [{model}|{proxy_addr or 'direct'}] stream complete")
                        if session_id and _reason_buf and _content_buf:
                            _remember_reasoning(session_id, _content_buf, _reason_buf)
                        yield line + "\n\n"
                        streamed_any = True
                        break

                    try:
                        piece = json.loads(payload)
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        err = ValueError(f"malformed upstream SSE JSON: {e}")
                        _log(
                            f"[zen] Malformed upstream stream (attempt {attempt}): "
                            f"{err}; raw={_stream_preview(payload)}"
                        )
                        if PROXY_POOL_ENABLED and proxy_addr:
                            proxy_pool.report_failure(proxy_addr)
                        last_error = err
                        if not streamed_any and attempt < attempts and not await _client_gone(request):
                            retry_stream = True
                            break
                        yield _openai_stream_error(str(err), "upstream_error", "malformed_stream")
                        return
                    if not isinstance(piece, dict):
                        err = ValueError("malformed upstream SSE JSON: expected an object")
                        _log(
                            f"[zen] Malformed upstream stream (attempt {attempt}): "
                            f"{err}; raw={_stream_preview(payload)}"
                        )
                        if PROXY_POOL_ENABLED and proxy_addr:
                            proxy_pool.report_failure(proxy_addr)
                        last_error = err
                        if not streamed_any and attempt < attempts and not await _client_gone(request):
                            retry_stream = True
                            break
                        yield _openai_stream_error(str(err), "upstream_error", "malformed_stream")
                        return

                    # A valid fragment only proves that the stream is alive;
                    # it is not a successful request yet. Keep the transport
                    # failure counter until the upstream sends [DONE].
                    event_error = _first_chunk_error(line)
                    if event_error:
                        err_msg, is_rate_limit = event_error
                        _log(f"[zen] Stream error in body (attempt {attempt}): {err_msg}")
                        if PROXY_POOL_ENABLED and proxy_addr and is_rate_limit:
                            proxy_pool.report_ratelimit(proxy_addr)
                        if is_rate_limit:
                            yield _openai_stream_error(
                                err_msg + " (free model rate limit)",
                                "rate_limit_error",
                                "rate_limit_exceeded",
                            )
                            return
                        if _is_promotion_ended_error(piece, line):
                            _mark_model_dead(model)
                            yield _openai_stream_error(f"{model} is no longer available: {err_msg}", "upstream_error", "model_retired")
                            return
                        if not streamed_any and attempt < attempts and not await _client_gone(request):
                            retry_stream = True
                            break
                        yield _openai_stream_error(err_msg)
                        return

                    # Capture emitted reasoning_content for re-injection on later turns
                    if '"delta"' in line:
                        choices = piece.get("choices")
                        first_choice = choices[0] if isinstance(choices, list) and choices else {}
                        if not isinstance(first_choice, dict):
                            first_choice = {}
                        d = first_choice.get("delta") or {}
                        if not isinstance(d, dict):
                            d = {}
                        if d.get("reasoning_content"):
                            _reason_buf += d["reasoning_content"]
                        if d.get("content"):
                            _content_buf += d["content"]
                        if d.get("tool_calls"):
                            tool_streamed = True
                        fr = first_choice.get("finish_reason")
                        if fr:
                            if fr == "tool_calls":
                                tool_streamed = True
                            finish_delivered = True
                            if session_id and _reason_buf and _content_buf:
                                _remember_reasoning(session_id, _content_buf, _reason_buf)
                            _reason_buf = ""
                            _content_buf = ""
                    yield line + "\n\n"
                    streamed_any = True
                    if '"usage"' in line:
                        u = piece.get("usage") or {}
                        if isinstance(u, dict) and (u.get("prompt_tokens") or u.get("completion_tokens")):
                            _add_tokens(model,
                                u.get("prompt_tokens") or 0,
                                u.get("completion_tokens") or 0,
                                u.get("prompt_cache_hit_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                                u.get("prompt_cache_miss_tokens") or 0,
                            )
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.TransportError) as e:
                _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Stream interrupted: {_exc_desc(e)}")
                last_error = e
                # A torn-down tunnel cannot carry this stream further; rotate
                # away immediately rather than giving the same proxy another
                # chance.
                if PROXY_POOL_ENABLED and proxy_addr:
                    proxy_pool.report_stream_failure(proxy_addr)
                if finish_delivered:
                    # The client already saw finish_reason; the turn completed.
                    # Just terminate the stream cleanly instead of failing it.
                    yield "data: [DONE]\n\n"
                    return
                if (
                    (not streamed_any or (not _content_buf and not tool_streamed))
                    and attempt < attempts
                    and not await _client_gone(request)
                ):
                    # Nothing was delivered, or only transient thinking — no
                    # answer text or tool fragments reached the client:
                    # replaying the request is safe and the model restarts
                    # cleanly.
                    retry_stream = True
                elif (
                    continuations < MAX_STREAM_CONTINUATIONS
                    and not tool_streamed
                    and _content_buf
                    and not await _client_gone(request)
                ):
                    # Resume: replay the request with the partial assistant
                    # output appended so the model continues from the cutoff
                    # instead of replaying the whole turn. Requires real
                    # content: upstream rejects assistant messages where
                    # content and tool_calls are both unset.
                    continuations += 1
                    req_body = _continuation_body(req_body, _content_buf, _reason_buf)
                    _content_buf = ""
                    _reason_buf = ""
                    retry_stream = True
                    _log(
                        f"[zen] Continuing stream (attempt {attempt + 1}, "
                        f"continuation {continuations})"
                    )
                else:
                    yield _openai_stream_error(
                        f"Stream interrupted: {_exc_desc(e)}",
                        "upstream_error",
                        "transport_error",
                    )
                    return
        finally:
            try:
                await resp.aclose()
            except Exception:
                pass
        if retry_stream:
            await _backoff(attempt)
            attempt += 1
            continue
        if not stream_completed:
            preview = (_content_buf[:120] + "…") if len(_content_buf) > 120 else _content_buf
            preview = preview.replace("\n", " ")
            detail = (
                f"streamed_any={streamed_any} "
                f"content_len={len(_content_buf)} reason_len={len(_reason_buf)} "
                f"tool_streamed={tool_streamed} finish_delivered={finish_delivered} "
                f"stream_completed={stream_completed} preview={preview!r}"
            )
            err = ValueError(f"upstream stream ended before [DONE] ({detail})")
            _log(f"[zen] [{model}|{proxy_addr or 'direct'}] {_exc_desc(err)} (attempt {attempt} pool={proxy_pool.get_pool_state() if PROXY_POOL_ENABLED else 'disabled'})")
            # Graceful upstream close after bytes were delivered is not a
            # proxy/transport failure — the model truncated the turn (often
            # mid-tool-call). Blacklisting the exit burns good proxies.
            # Only hard-blacklist when nothing was delivered at all.
            if PROXY_POOL_ENABLED and proxy_addr and not streamed_any:
                proxy_pool.report_stream_failure(proxy_addr)
            elif PROXY_POOL_ENABLED and proxy_addr and streamed_any:
                # Rotate away from this connection without poisoning the pool
                try:
                    proxy_pool._evict_client(proxy_addr)
                except Exception:
                    pass
                if proxy_pool.current and proxy_pool.current.get("address") == proxy_addr:
                    proxy_pool.current = None
            last_error = err
            if finish_delivered:
                yield "data: [DONE]\n\n"
                return
            if (not streamed_any or (not _content_buf and not tool_streamed)) and attempt < attempts:
                await _backoff(attempt)
                attempt += 1
                continue
            if (
                continuations < MAX_STREAM_CONTINUATIONS
                and not tool_streamed
                and _content_buf
                and not await _client_gone(request)
            ):
                continuations += 1
                req_body = _continuation_body(req_body, _content_buf, _reason_buf)
                _content_buf = ""
                _reason_buf = ""
                await _backoff(attempt)
                attempt += 1
                continue
            yield _openai_stream_error(str(err), "upstream_error", "incomplete_stream")
        return

    # All retries exhausted
    yield _openai_stream_error(
        f"Stream failed after {attempts + 1} attempts: {_exc_desc(last_error)}",
        "upstream_error",
        "transport_error",
    )


async def _zen_stream_anthropic_with_retry(
    request: Request,
    req_body: dict,
    headers: dict,
    user: str,
    messages: list[dict],
    model: str,
    input_tokens: int,
    session_id: str = None,
    max_retries: int = None,
):
    """Anthropic-format streaming with proxy pool retry on 429 and ReadError recovery."""
    msg_id = oc_id("msg")
    content_idx = 0
    tool_idx = -1
    text_closed = False
    output_tokens = 0
    headers_sent = False
    streamed_any = False  # True once any SSE byte is sent to the client
    last_error = None
    _reason_buf = ""
    _content_buf = ""
    attempts = MAX_RETRIES if max_retries is None else max_retries
    continuations = 0  # mid-stream resume attempts (see MAX_STREAM_CONTINUATIONS)
    finish_delivered = False  # True once the terminal finish_reason chunk was emitted
    tool_streamed = False  # True once an input_json_delta / tool block was emitted

    def send_sse(event: str, data: dict) -> str:
        nonlocal streamed_any
        streamed_any = True
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def close_indices() -> list[int]:
        """Indices of all open content blocks needing content_block_stop."""
        idx = []
        if content_idx > 0 and not text_closed:
            idx.append(0)
        offset = 1 if content_idx > 0 else 0
        for i in range(tool_idx + 1):
            idx.append(i + offset)
        return idx

    attempt = 0
    while True:
        # Total work cap: plain retries (attempts) + continuations share one counter.
        # It's intentional — see _zen_stream_with_retry.
        if attempt > attempts + MAX_STREAM_CONTINUATIONS:
            break
        if await _client_gone(request):
            _log("[zen] Client disconnected; aborting retries")
            return
        if PROXY_POOL_ENABLED:
            if not proxy_pool.ready:
                await proxy_pool.load()

            if not await proxy_pool.select():
                _log(f"[pool] No proxy ({proxy_pool.get_pool_state()}), forcing refresh")
                await proxy_pool.force_refresh()
                if not await proxy_pool.select():
                    _log("[pool] Fallback to direct")
                    client = _stream_default_client
                    proxy_addr = None
                else:
                    p = proxy_pool.current
                    proxy_addr = p["address"]
                    client = proxy_pool.get_client(
                        f"socks5://{proxy_addr}", streaming=True
                    )
            else:
                p = proxy_pool.current
                proxy_addr = p["address"]
                client = proxy_pool.get_client(
                    f"socks5://{proxy_addr}", streaming=True
                )
        else:
            client = _stream_default_client
            proxy_addr = None

        try:
            async with client.stream("POST", "/zen/v1/chat/completions", json=req_body, headers=headers) as resp:
                if resp.status_code == 429:
                    try:
                        raw = await resp.aread()
                        parsed = json.loads(raw)
                        err_msg = (parsed.get("error") or {}).get("message") or "Rate limit"
                    except Exception:
                        err_msg = "Rate limit"
                    _log(f"[zen] Anthropic stream 429 (attempt {attempt}): {err_msg}")
                    if PROXY_POOL_ENABLED and proxy_addr:
                        proxy_pool.report_ratelimit(proxy_addr)
                    yield send_sse("error", {"type": "error", "error": {"type": "rate_limit_error", "message": err_msg + " (free model rate limit)"}})
                    return

                if resp.status_code >= 400:
                    raw = await resp.aread()
                    body_text = raw.decode("utf-8", errors="replace")
                    try:
                        data = json.loads(body_text)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        data = {}
                    err_msg = (data.get("error") or {}).get("message") or f"HTTP {resp.status_code}"
                    is_context_exceeded = _is_context_limit_error(data, body_text)
                    is_region_blocked = _is_region_error(data, body_text)
                    _log(f"[zen] Anthropic stream error {resp.status_code}: {raw[:300]}")
                    if is_region_blocked:
                        # Geo-restriction is proxy-specific: blacklist and rotate.
                        if PROXY_POOL_ENABLED and proxy_addr:
                            proxy_pool.report_region_block(proxy_addr)
                        _log(f"[zen] Region-blocked via {proxy_addr}; rotating proxy")
                        if attempt < attempts:
                            await _backoff(attempt)
                            attempt += 1
                            continue
                        yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": err_msg}})
                        return
                    # Not a proxy failure: 4xx/5xx are upstream or request
                    # errors that repeat identically on every proxy.
                    if _is_promotion_ended_error(data, body_text):
                        _mark_model_dead(model)
                        yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": f"{model} is no longer available: {err_msg}"}})
                        return
                    if is_context_exceeded:
                        _log("[zen] Context limit error; not retrying on another proxy")
                    if resp.status_code >= 500 and not is_context_exceeded and attempt < attempts:
                        await _backoff(attempt)
                        attempt += 1
                        continue
                    yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": err_msg}})
                    return

                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue

                    if not headers_sent:
                        first_err = _first_chunk_error(raw_line)
                        if first_err:
                            err_msg, is_rate_limit = first_err
                            _log(f"[zen] Anthropic error in body (attempt {attempt}): {err_msg}")
                            if PROXY_POOL_ENABLED and proxy_addr and is_rate_limit:
                                proxy_pool.report_ratelimit(proxy_addr)
                            if is_rate_limit:
                                yield send_sse("error", {"type": "error", "error": {"type": "rate_limit_error", "message": err_msg + " (free model rate limit)"}})
                                return
                            try:
                                _piece = json.loads(raw_line.strip()[6:] if raw_line.strip().startswith("data: ") else raw_line)
                            except (json.JSONDecodeError, TypeError, ValueError):
                                _piece = {}
                            if _is_promotion_ended_error(_piece, raw_line):
                                _mark_model_dead(model)
                                yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": f"{model} is no longer available: {err_msg}"}})
                                return
                            if attempt < attempts:
                                break
                            yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": err_msg}})
                            return

                    if raw_line.startswith("data: "):
                        payload = raw_line[6:].strip()
                        if payload == "[DONE]":
                            if not finish_delivered:
                                for i in close_indices():
                                    yield send_sse("content_block_stop", {"type": "content_block_stop", "index": i})
                                yield send_sse("message_delta", {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": "end_turn"},
                                    "usage": {"output_tokens": output_tokens},
                                })
                            yield send_sse("message_stop", {"type": "message_stop"})
                            return
                        try:
                            parsed = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        u = parsed.get("usage")
                        if isinstance(u, dict) and (u.get("prompt_tokens") or u.get("completion_tokens")):
                            _add_tokens(model,
                                u.get("prompt_tokens") or 0,
                                u.get("completion_tokens") or 0,
                                u.get("prompt_cache_hit_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                                u.get("prompt_cache_miss_tokens") or 0,
                            )

                        delta = (parsed.get("choices") or [{}])[0].get("delta") or {}
                        if not delta:
                            continue

                        # Capture upstream reasoning for re-injection in this session
                        if delta.get("reasoning_content"):
                            _reason_buf += delta["reasoning_content"]
                        if delta.get("content"):
                            _content_buf += delta["content"]
                        if (parsed.get("choices") or [{}])[0].get("finish_reason"):
                            if session_id and _reason_buf and _content_buf:
                                _remember_reasoning(session_id, _content_buf, _reason_buf)
                            _reason_buf = ""
                            _content_buf = ""

                        if not headers_sent:
                            headers_sent = True
                            yield send_sse("message_start", {
                                "type": "message_start",
                                "message": {
                                    "id": msg_id, "type": "message", "role": "assistant", "content": [],
                                    "model": model, "stop_reason": None,
                                    "usage": {"input_tokens": input_tokens or 0, "output_tokens": 0, **_NO_CACHE},
                                },
                            })

                        if delta.get("content"):
                            if content_idx == 0 and tool_idx == -1:
                                yield send_sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                                content_idx = 1
                            yield send_sse("content_block_delta", {
                                "type": "content_block_delta", "index": 0,
                                "delta": {"type": "text_delta", "text": delta["content"]},
                            })
                            output_tokens += -(-len(delta["content"]) // 4)

                        for tc in delta.get("tool_calls", []):
                            idx = tc.get("index", 0)
                            if idx > tool_idx:
                                tool_streamed = True
                                if tool_idx == -1 and content_idx > 0:
                                    yield send_sse("content_block_stop", {"type": "content_block_stop", "index": 0})
                                    text_closed = True
                                tool_idx = idx
                                block_idx = idx + 1 if content_idx > 0 else idx
                                yield send_sse("content_block_start", {
                                    "type": "content_block_start", "index": block_idx,
                                    "content_block": {"type": "tool_use", "id": tc.get("id") or oc_id("toolu"), "name": (tc.get("function") or {}).get("name") or ""},
                                })
                            func = tc.get("function") or {}
                            if func.get("arguments"):
                                block_idx = idx + 1 if content_idx > 0 else idx
                                yield send_sse("content_block_delta", {
                                    "type": "content_block_delta", "index": block_idx,
                                    "delta": {"type": "input_json_delta", "partial_json": func["arguments"]},
                                })
                                output_tokens += -(-len(func["arguments"]) // 4)

                        finish_reason = (parsed.get("choices") or [{}])[0].get("finish_reason")
                        if finish_reason:
                            if finish_reason == "tool_calls":
                                tool_streamed = True
                            finish_delivered = True
                            for i in close_indices():
                                yield send_sse("content_block_stop", {"type": "content_block_stop", "index": i})

                            stop_reason = "end_turn"
                            if finish_reason == "tool_calls":
                                stop_reason = "tool_use"
                            elif finish_reason == "length":
                                stop_reason = "max_tokens"

                            yield send_sse("message_delta", {
                                "type": "message_delta",
                                "delta": {"stop_reason": stop_reason},
                                "usage": {"output_tokens": output_tokens},
                            })
                # If we broke out before emitting headers (e.g. an in-body
                # upstream error), replay on a fresh attempt without touching
                # the pool: application errors are not proxy failures.
                if not headers_sent and attempt < attempts and not await _client_gone(request):
                    await _backoff(attempt)
                    attempt += 1
                    continue
                # The upstream stream ended without a finish_reason.
                preview = (_content_buf[:120] + "…") if len(_content_buf) > 120 else _content_buf
                preview = preview.replace("\n", " ")
                detail = (
                    f"headers_sent={headers_sent} streamed_any={streamed_any} "
                    f"content_len={len(_content_buf)} reason_len={len(_reason_buf)} "
                    f"tool_streamed={tool_streamed} finish_delivered={finish_delivered} preview={preview!r}"
                )
                _log(f"[zen] Anthropic stream incomplete (attempt {attempt} proxy={proxy_addr or 'direct'} pool={proxy_pool.get_pool_state() if PROXY_POOL_ENABLED else 'disabled'} {detail})")
                last_error = ValueError(f"upstream stream closed before finish_reason ({detail})")
                if PROXY_POOL_ENABLED and proxy_addr and not streamed_any:
                    proxy_pool.report_stream_failure(proxy_addr)
                elif PROXY_POOL_ENABLED and proxy_addr and streamed_any:
                    try:
                        proxy_pool._evict_client(proxy_addr)
                    except Exception:
                        pass
                    if proxy_pool.current and proxy_pool.current.get("address") == proxy_addr:
                        proxy_pool.current = None
                if finish_delivered:
                    # Terminal finish_reason was already delivered; nothing
                    # more to emit.
                    return
                if (
                    continuations < MAX_STREAM_CONTINUATIONS
                    and not tool_streamed
                    and _content_buf
                    and not await _client_gone(request)
                ):
                    continuations += 1
                    req_body = _continuation_body(req_body, _content_buf, _reason_buf)
                    _content_buf = ""
                    _reason_buf = ""
                    _log(
                        f"[zen] Continuing anthropic stream (attempt {attempt + 1}, "
                        f"continuation {continuations})"
                    )
                    await _backoff(attempt)
                    attempt += 1
                    continue
                if headers_sent:
                    # Message started but did not finish: close open blocks
                    # and emit a clean error event.
                    for i in close_indices():
                        yield send_sse("content_block_stop", {"type": "content_block_stop", "index": i})
                yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": "Stream interrupted: upstream closed the connection before completing the message"}})
                return

        except Exception as e:
            _log(f"[zen] Anthropic stream HTTP error (attempt {attempt}): {_exc_desc(e)}")
            # Only transport-level failures are proxy failures; a bug in our
            # translation code must not blacklist a healthy proxy.
            if PROXY_POOL_ENABLED and proxy_addr and isinstance(e, httpx.HTTPError):
                proxy_pool.report_stream_failure(proxy_addr)
            last_error = e
            if headers_sent and finish_delivered:
                # Terminal finish_reason already delivered; stop cleanly.
                return
            # Never replay once answer bytes reached the client; but if only
            # thinking was streamed (no content, no tool fragments), a fresh
            # request is safe.
            if (
                (not streamed_any or (not _content_buf and not tool_streamed))
                and attempt < attempts
                and not await _client_gone(request)
            ):
                await _backoff(attempt)
                attempt += 1
                continue
            if (
                headers_sent
                and continuations < MAX_STREAM_CONTINUATIONS
                and not tool_streamed
                and _content_buf
                and not await _client_gone(request)
            ):
                # Resume: replay with partial output appended so the model
                # continues instead of replaying the whole turn. Requires
                # real content: upstream rejects assistant messages where
                # content and tool_calls are both unset.
                continuations += 1
                req_body = _continuation_body(req_body, _content_buf, _reason_buf)
                _content_buf = ""
                _reason_buf = ""
                _log(
                    f"[zen] Continuing anthropic stream (attempt {attempt + 1}, "
                    f"continuation {continuations}); {_exc_desc(e)}"
                )
                await _backoff(attempt)
                attempt += 1
                continue
            if not streamed_any:
                yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": _exc_desc(e)}})
            else:
                # Message was already started: close any blocks left open by
                # the aborted attempt and terminate the message cleanly.
                for i in close_indices():
                    yield send_sse("content_block_stop", {"type": "content_block_stop", "index": i})
                yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": "Stream interrupted: upstream closed the connection before completing the message"}})
            return

    if not headers_sent:
        yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": f"Stream failed after {attempts + 1} attempts: {_exc_desc(last_error)}"}})


# ── Anthropic Messages → OpenAI conversion ────────────────────────

def _anthropic_thinking(blocks) -> str:
    """Extract reasoning text from Anthropic `thinking`/`redacted_thinking` blocks."""
    parts = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"):
            t = b.get("thinking") or b.get("text")
            if t:
                parts.append(t)
    return "\n".join(parts)


def _fmt_cont(cont) -> str:
    if cont is None:
        return "-"
    if isinstance(cont, str):
        return "str(" + repr(cont[:60]) + ")"
    if isinstance(cont, list):
        return "list[" + ",".join(b.get("type", "?") for b in cont if isinstance(b, dict)) + "]"
    return type(cont).__name__


def _prune_dangling_tools(messages: list[dict]) -> list[dict]:
    """Drop `tool` messages whose `tool_call_id` is not declared by a preceding
    assistant message in THIS body. A warm upstream session remembers the
    declaring turns, but a fresh session has no memory and rejects them. Used
    only when we are about to send the request to a brand-new session."""
    if not isinstance(messages, list):
        return messages
    out: list[dict] = []
    open_tc: set = set()
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role")
        if role == "assistant":
            open_tc = {tc.get("id") for tc in (m.get("tool_calls") or []) if isinstance(tc, dict)}
            out.append(m)
        elif role == "tool":
            tid = m.get("tool_call_id")
            if tid in open_tc:
                out.append(m)
                open_tc.discard(tid)
            else:
                _log("[zen] Pruned dangling tool message (id=%r) for fresh session" % (tid,))
        else:
            open_tc = set()
            out.append(m)
    return out


_400_DUMP_DIR = _BASE_DIR / "diag"
_400_DUMP_MAX_PER_MODEL = 3


def _dump_400_body(req_body: dict, upstream_body: str, proxy_addr: str | None):
    """Persist a rejected 400 request body for offline diagnosis.

    The [1210] "Invalid API parameter" rejection is deterministic per body
    (same session fails on every proxy while others succeed), so the payload
    itself must be inspected. Bounded: at most _400_DUMP_MAX_PER_MODEL files
    per model, then it stops writing — enough to capture one full episode.
    """
    try:
        model = str(req_body.get("model") or "unknown")
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        model_dir = _400_DUMP_DIR / safe
        model_dir.mkdir(parents=True, exist_ok=True)
        existing = list(model_dir.glob("*.json"))
        if len(existing) >= _400_DUMP_MAX_PER_MODEL:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        payload = {
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "proxy": proxy_addr,
            "upstream_error": upstream_body[:2000],
            "request": req_body,
        }
        path = model_dir / f"req_{stamp}_{secrets.token_hex(3)}.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        _log(f"[zen] 400 body dumped: {path.name} ({len(existing) + 1}/{_400_DUMP_MAX_PER_MODEL})")
    except Exception as e:
        _log(f"[zen] 400 body dump failed: {_exc_desc(e)}")


def _log_reasoning_diag(req_body: dict):
    """Dump the ordered message layout (roles + tool/reasoning detail) whenever the
    upstream rejects a 400, to diagnose context/ordering problems on session switch."""
    try:
        msgs = (req_body or {}).get("messages") or []
        parts = [f"model={req_body.get('model')}", f"n={len(msgs)}"]
        last_tc_ids: set[str] = set()
        ordering_ok = True
        for idx, m in enumerate(msgs):
            if not isinstance(m, dict):
                parts.append(f"[{idx}]?")
                continue
            role = m.get("role") or "?"
            if role == "assistant":
                tcs = m.get("tool_calls") or []
                last_tc_ids = {tc.get("id") for tc in tcs if isinstance(tc, dict)}
                rsn = m.get("reasoning_content") or m.get("reasoning")
                parts.append(
                    f"[{idx}]assistant content={_fmt_cont(m.get('content'))} tc={sorted(last_tc_ids)} "
                    + (f"rsn={len(str(rsn))}" if rsn else "NO_REASONING")
                )
            elif role == "tool":
                cid = m.get("tool_call_id")
                preceded = cid in last_tc_ids
                if not preceded:
                    ordering_ok = False
                parts.append(f"[{idx}]tool id={cid} preceded_tc={preceded}")
                last_tc_ids = set()
            elif role == "function":
                name = m.get("name")
                parts.append(f"[{idx}]function name={name}")
            else:
                parts.append(f"[{idx}]{role}")
        parts.append(f"ORDER_OK={ordering_ok}")
        _log(f"[zen] 400 diag: " + " ".join(parts))
    except Exception as e:
        _log(f"[zen] 400 diag failed: {_exc_desc(e)}")


def anthropic_to_openai(body: dict) -> tuple[list[dict], list[dict] | None]:
    messages = []

    if body.get("system"):
        sys_text = _blocks_text(body["system"])
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": msg["role"], "content": content})
        elif isinstance(content, list):
            text = _blocks_text(content)
            reasoning = _anthropic_thinking(content)
            tool_uses = [b for b in content if b.get("type") == "tool_use"]

            if tool_uses and msg.get("role") == "assistant":
                entry = {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": t["id"],
                            "type": "function",
                            "function": {
                                "name": t["name"],
                                "arguments": json.dumps(t.get("input") or {}),
                            },
                        }
                        for t in tool_uses
                    ],
                }
                if reasoning:
                    entry["reasoning_content"] = reasoning
                messages.append(entry)
            elif any(b.get("type") == "tool_result" for b in content):
                for b in content:
                    if b.get("type") == "tool_result":
                        messages.append({
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": _blocks_text(b.get("content")),
                        })
            else:
                entry = {"role": msg["role"], "content": text}
                if reasoning and msg.get("role") == "assistant":
                    entry["reasoning_content"] = reasoning
                messages.append(entry)
        else:
            messages.append({"role": msg["role"], "content": str(content)})

    tools_out = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description") or "",
                "parameters": t.get("input_schema") or {},
            },
        }
        for t in body.get("tools", [])
    ]

    return messages, tools_out or None


# ── OpenAI response → Anthropic Messages format ──────────────────

def openai_to_anthropic(oai_resp: dict, model: str, input_tokens: int) -> dict:
    choice = (oai_resp.get("choices") or [None])[0]
    if not choice:
        return {
            "id": oc_id("msg"),
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
            "model": model,
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": input_tokens or 0,
                "output_tokens": 0,
                **_NO_CACHE,
            },
        }

    content = []
    msg = choice.get("message") or {}
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls", []):
        try:
            inp = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            inp = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id") or oc_id("toolu"),
            "name": tc["function"]["name"],
            "input": inp,
        })
    if not content:
        content.append({"type": "text", "text": ""})

    stop_reason = "end_turn"
    fr = choice.get("finish_reason")
    if fr == "tool_calls":
        stop_reason = "tool_use"
    elif fr == "length":
        stop_reason = "max_tokens"

    return {
        "id": oc_id("msg"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": (oai_resp.get("usage") or {}).get("prompt_tokens") or input_tokens or 0,
            "output_tokens": (oai_resp.get("usage") or {}).get("completion_tokens") or 0,
            **_NO_CACHE,
        },
    }


# ── Routes: OpenAI format ─────────────────────────────────────────

async def list_models(request: Request):
    if not _models_cache:
        await _fetch_free_models()
    data = []
    for m in _models_cache:
        entry = {"id": m, "object": "model", "created": 1779000000, "owned_by": "opencode-free"}
        meta = _models_meta.get(m)
        if meta:
            if meta.get("limit"):
                entry["limits"] = meta["limit"]
            if meta.get("modalities"):
                entry["modalities"] = meta["modalities"]
        data.append(entry)
    return {"object": "list", "data": data}


async def chat_completions(request: Request):
    user = auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": {"message": "Invalid API key"}})

    body = await request.json()
    model = body.get("model")
    messages = body.get("messages")
    stream = body.get("stream")
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")

    model = _normalize_model(model)
    if not await _ensure_model_known(model):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Unknown model: {model}. Available: {', '.join(_models_cache)}"}},
        )

    session_id = get_session(user, messages)

    up_messages = _prepare_upstream_messages(session_id, _normalize_messages(messages))
    req_body, headers = zen_request(model, up_messages, stream, tools, tool_choice, session_id, body.get("max_tokens"), body.get("max_completion_tokens"), _sampling_from(body))

    if stream:
        if _needs_buffered_fallback(model, tools, stream):
            _log(f"[zen] buffered fallback: stream:true -> stream:false upstream (model={model})")
            buffered_body = dict(req_body)
            buffered_body["stream"] = False
            data = await _zen_request_with_retry(request, buffered_body, headers, user, messages or [], session_id, model)
            if isinstance(data, JSONResponse):
                return data
            if data is None:
                return JSONResponse(status_code=502, content={"error": {"message": "Client disconnected", "type": "upstream_error"}})
            async def _muse_spark_openai_sse():
                for chunk in _buffered_to_openai_sse(data, model):
                    yield chunk
            return StreamingResponse(
                _muse_spark_openai_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                },
            )
        return StreamingResponse(
            _zen_stream_with_retry(request, req_body, headers, user, messages or [], session_id, model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        data = await _zen_request_with_retry(request, req_body, headers, user, messages or [], session_id, model)
        if isinstance(data, JSONResponse):
            return data
        if data is None:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "Client disconnected", "type": "upstream_error"}},
            )
        if not data.get("choices"):
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "Invalid upstream response", "type": "upstream_error"}},
            )
        return data


# ── Routes: Anthropic Messages format ─────────────────────────────

async def messages(request: Request):
    user = auth(request)
    if not user:
        return JSONResponse(
            status_code=401,
            content={"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key"}},
        )

    body = await request.json()
    model = body.get("model")
    stream = body.get("stream")

    model = _normalize_model(model)

    if not await _ensure_model_known(model):
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": f"Unknown model: {model}. Available: {', '.join(_models_cache)}"}},
        )

    oai_messages, tools = anthropic_to_openai(body)
    session_id = get_session(user, oai_messages)
    input_tokens = len(json.dumps(oai_messages)) // 4

    up_messages = _prepare_upstream_messages(session_id, oai_messages)
    req_body, headers = zen_request(model, up_messages, stream, tools, None, session_id, body.get("max_tokens"), body.get("max_completion_tokens"), _sampling_from(body))

    if stream:
        if _needs_buffered_fallback(model, tools, stream):
            _log(f"[zen] muse-spark buffered fallback (anthropic): stream:true+tools -> stream:false upstream (model={model})")
            buffered_body = dict(req_body)
            buffered_body["stream"] = False
            data = await _zen_request_with_retry(request, buffered_body, headers, user, oai_messages, session_id, model)
            if isinstance(data, JSONResponse):
                return data
            if data is None:
                return JSONResponse(
                    status_code=502,
                    content={"type": "error", "error": {"type": "upstream_error", "message": "Client disconnected"}},
                )
            async def _muse_spark_anthropic_sse():
                for chunk in _buffered_to_anthropic_sse(data, model, input_tokens):
                    yield chunk
            return StreamingResponse(
                _muse_spark_anthropic_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                },
            )
        return StreamingResponse(
            _zen_stream_anthropic_with_retry(request, req_body, headers, user, oai_messages, model, input_tokens, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        data = await _zen_request_with_retry(request, req_body, headers, user, oai_messages, session_id, model)
        if isinstance(data, JSONResponse):
            return data
        if data is None:
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "upstream_error", "message": "Client disconnected"}},
            )
        if not data.get("choices"):
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "upstream_error", "message": "Invalid upstream response"}},
            )
        return openai_to_anthropic(data, model, input_tokens)


# ── /v1/responses (Responses API for Codex openai_base_url) ─────────────

def _responses_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _response_obj(resp_id: str, model: str, status: str, output: list, usage: dict | None = None) -> dict:
    obj = {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": status,
        "output": output,
    }
    if usage is not None:
        obj["usage"] = usage
    return obj


def _chat_usage_to_responses(usage: dict | None) -> dict:
    usage = usage or {}
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _chat_to_responses(data: dict, model: str):
    """Convert a chat/completions completion object into a Responses object."""
    choice = (data.get("choices") or [None])[0]
    msg = (choice or {}).get("message") or {}
    output = []
    content = msg.get("content")
    tcs = msg.get("tool_calls")
    if content:
        output.append({
            "type": "message", "id": oc_id("msg"), "status": "completed",
            "role": "assistant", "content": [{"type": "output_text", "text": content}],
        })
    if tcs:
        for tc in tcs:
            fn = tc.get("function") or {}
            output.append({
                "type": "function_call", "id": tc.get("id") or oc_id("call"),
                "status": "completed", "name": fn.get("name", ""),
                "arguments": fn.get("arguments", ""), "call_id": tc.get("id"),
            })
    return JSONResponse(_response_obj(
        data.get("id") or oc_id("resp"), model, "completed", output,
        _chat_usage_to_responses(data.get("usage")),
    ))


async def _stream_as_responses(upstream, model: str, session_id=None):
    """Translate an upstream OpenAI chat-completions SSE stream into the
    Responses API event sequence a codex_responses client expects.

    Zen only speaks chat/completions, so we reshape its ``chat.completion.chunk``
    stream into ``response.*`` SSE events (including a terminal
    ``response.completed`` carrying status). Without this the client sees an
    empty stream and reports 'stream closed before finish_reason'.
    """
    resp_id = oc_id("resp")
    msg_id = oc_id("msg")
    text_buf: list[str] = []
    tool_calls: dict[int, dict] = {}
    fc_order: list[int] = []
    fc_index: dict[int, int] = {}
    fc_open: dict[int, bool] = {}
    finish_reason = None
    usage = None
    msg_open = False
    msg_output_index = 0
    next_index = 0
    yield _responses_sse("response.created", {
        "type": "response.created",
        "response": _response_obj(resp_id, model, "in_progress", []),
    })
    yield _responses_sse("response.in_progress", {
        "type": "response.in_progress",
        "response": _response_obj(resp_id, model, "in_progress", []),
    })

    try:
        while True:
            try:
                raw = await asyncio.wait_for(anext(upstream), timeout=30.0)
            except StopAsyncIteration:
                break
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as e:
                _log(f"[zen] Responses upstream stalled; finalizing partial output: {_exc_desc(e)}")
                break
            payload = raw[len("data:"):].strip()
            if payload in ("[DONE]", ""):
                continue
            try:
                data = json.loads(payload)
            except Exception:
                continue
            choices = data.get("choices") or []
            if not choices:
                if data.get("usage"):
                    usage = data["usage"]
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            tcs = delta.get("tool_calls")
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            if data.get("usage"):
                usage = data["usage"]
            if content:
                if not msg_open:
                    msg_open = True
                    idx = next_index
                    next_index += 1
                    msg_output_index = idx
                    yield _responses_sse("response.output_item.added", {
                        "type": "response.output_item.added",
                        "output_index": idx,
                        "item": {"type": "message", "id": msg_id, "status": "in_progress",
                                 "role": "assistant", "content": []},
                    })
                    yield _responses_sse("response.content_part.added", {
                        "type": "response.content_part.added",
                        "item_id": msg_id, "output_index": idx, "content_index": 0,
                        "part": {"type": "output_text", "text": ""},
                    })
                text_buf.append(content)
                yield _responses_sse("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": msg_id, "output_index": msg_output_index, "content_index": 0,
                    "delta": content,
                })
            if tcs:
                for t in tcs:
                    idx = t.get("index", 0)
                    tc = tool_calls.get(idx)
                    if tc is None:
                        tc = {"id": t.get("id") or oc_id("call"), "name": "", "args": ""}
                        tool_calls[idx] = tc
                        fc_order.append(idx)
                    if t.get("id"):
                        tc["id"] = t["id"]
                    fn = t.get("function") or {}
                    if fn.get("name"):
                        tc["name"] = fn["name"]
                    frag = fn.get("arguments") or ""
                    if frag:
                        tc["args"] += frag
                    if not fc_open.get(idx):
                        fc_open[idx] = True
                        fc_index[idx] = next_index
                        next_index += 1
                        yield _responses_sse("response.output_item.added", {
                            "type": "response.output_item.added",
                            "output_index": fc_index[idx],
                            "item": {"type": "function_call", "id": tc["id"],
                                     "status": "in_progress", "name": tc["name"],
                                     "arguments": ""},
                        })
                    if frag:
                        yield _responses_sse("response.function_call_arguments.delta", {
                            "type": "response.function_call_arguments.delta",
                            "item_id": tc["id"], "delta": frag,
                        })
    except Exception as e:
        _log(f"[zen] Responses upstream stream error: {_exc_desc(e)}")

    output = []
    if msg_open and text_buf:
        full = "".join(text_buf)
        yield _responses_sse("response.output_text.done", {
            "type": "response.output_text.done",
            "item_id": msg_id, "output_index": msg_output_index, "content_index": 0, "text": full,
        })
        yield _responses_sse("response.content_part.done", {
            "type": "response.content_part.done",
            "item_id": msg_id, "output_index": msg_output_index, "content_index": 0,
            "part": {"type": "output_text", "text": full},
        })
        yield _responses_sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": msg_output_index,
            "item": {"type": "message", "id": msg_id, "status": "completed",
                     "role": "assistant", "content": [{"type": "output_text", "text": full}]},
        })
        output.append({"type": "message", "id": msg_id, "status": "completed",
                       "role": "assistant", "content": [{"type": "output_text", "text": full}]})

    for idx in fc_order:
        tc = tool_calls[idx]
        yield _responses_sse("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "item_id": tc["id"], "delta": tc["args"],
        })
        yield _responses_sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": fc_index.get(idx, len(output)),
            "item": {"type": "function_call", "id": tc["id"], "status": "completed",
                     "name": tc["name"], "arguments": tc["args"], "call_id": tc["id"]},
        })
        output.append({"type": "function_call", "id": tc["id"], "status": "completed",
                       "name": tc["name"], "arguments": tc["args"], "call_id": tc["id"]})

    status = "completed" if finish_reason in ("stop", "tool_calls", "function_call") else "incomplete"
    yield _responses_sse("response.completed", {
        "type": "response.completed",
        "response": _response_obj(resp_id, model, status, output, _chat_usage_to_responses(usage)),
    })


async def handle_responses(request: Request):
    try:
        body = await request.json()
        model = body.get("model", "")
        stream = body.get("stream", False)
        messages = _input_to_messages(body.get("input", ""))
        tools = _extract_tools(body)
        tool_choice = body.get("tool_choice")
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function" and "function" not in tool_choice:
            tool_choice = {"type": "function", "function": {"name": tool_choice.get("name", "")}}

        user = auth(request)
        if not user:
            return JSONResponse(status_code=401, content={"error": {"message": "Invalid API key"}})

        # Ensure discovered model list is warm before alias mapping (cold-start gap: alias -> _models_cache[0])
        if not _models_cache:
            await _fetch_free_models()
        zen_model = _map_model(model)
        # Validate against discovered free list like chat_completions/messages do
        if not await _ensure_model_known(zen_model):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"Unknown model: {zen_model}. Available: {', '.join(_models_cache)}"}},
            )
        messages = _normalize_messages(messages)
        session_id = get_session(user, messages)
        up_messages = _prepare_upstream_messages(session_id, messages)
        req_body, headers = zen_request(zen_model, up_messages, stream, tools, tool_choice, session_id, body.get("max_tokens"), body.get("max_completion_tokens"), _sampling_from(body))
        if stream:
            # Known-truncating models: stream:true+tools (muse-spark) or stream:true reasoning (x-preview)
            # closes without finish_reason/[DONE]. Same buffered bridge as chat/messages.
            if _needs_buffered_fallback(zen_model, tools, True):
                _log(f"[zen] responses buffered fallback: stream:true -> stream:false upstream (model={zen_model})")
                buffered_body = dict(req_body)
                buffered_body["stream"] = False
                data = await _zen_request_with_retry(request, buffered_body, headers, user, messages, session_id, zen_model)
                if data is None:
                    return JSONResponse(
                        status_code=502,
                        content={"error": {"message": "Client disconnected", "type": "upstream_error"}},
                    )
                if isinstance(data, JSONResponse):
                    return data
                # Reframe buffered chat/completions as Responses stream
                async def _buffered_responses():
                    async def _single_chunk():
                        # Buffered choices carry a full "message"; reshape to a
                        # streaming "delta" event so _stream_as_responses
                        # actually emits the text/tool-call output items.
                        ch = dict((data.get("choices") or [{}])[0] or {})
                        msg = ch.get("message") or {}
                        delta = {"role": "assistant", "content": msg.get("content")}
                        if msg.get("tool_calls"):
                            delta["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(msg["tool_calls"])]
                        ch["delta"] = delta
                        yield f"data: {json.dumps({'choices': [ch], 'usage': data.get('usage')})}\n\n"
                        yield "data: [DONE]\n\n"
                    async for evt in _stream_as_responses(_single_chunk(), model, session_id):
                        yield evt
                return StreamingResponse(
                    _buffered_responses(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache, no-transform",
                        "X-Accel-Buffering": "no",
                    },
                )
            upstream = _zen_stream_with_retry(request, req_body, headers, user, messages, session_id, zen_model)
            return StreamingResponse(
                _stream_as_responses(upstream, model, session_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                },
            )

        data = await _zen_request_with_retry(request, req_body, headers, user, messages, session_id, zen_model)
        if isinstance(data, JSONResponse):
            return data
        if data is None:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "Client disconnected", "type": "upstream_error"}},
            )
        return _chat_to_responses(data, model)
    except Exception as e:
        import traceback as _tb
        _log(f"[responses] ERROR: {_tb.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "proxy_error"}},
        )

def _input_to_messages(inp):
    """Convert Responses API 'input' to chat messages array."""
    if isinstance(inp, str):
        return [{"role": "user", "content": inp}]
    if isinstance(inp, list):
        msgs = []
        for item in inp:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                msgs.append({"role": role, "content": content})
        if not msgs:
            msgs.append({"role": "user", "content": ""})
        return msgs
    return [{"role": "user", "content": ""}]

def _extract_tools(body):
    tools = body.get("tools") or []
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if "function" in t:
            out.append(t)
        else:
            out.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters") or {},
                },
            })
    return out or None

def _map_model(model: str) -> str:
    """Resolve legacy role aliases against the DISCOVERED model list only.

    No model ids are hardcoded: aliases pick from whatever the Zen API
    currently exposes, falling back to the first discovered free model.
    """
    if not _models_cache:
        return _normalize_model(model)
    m = model.lower().replace("-", "").replace("_", "")
    if m == "opencodedefault":
        return _models_cache[0]
    if m == "opencodefast":
        return next((x for x in _models_cache if "flash" in x or "lightning" in x or "mini" in x), _models_cache[0])
    if m == "opencodesmart":
        return next((x for x in _models_cache if "ultra" in x or "pro" in x or "smart" in x), _models_cache[0])
    return _normalize_model(model)


async def health(request: Request):
    pool_state = None
    if PROXY_POOL_ENABLED:
        pool_state = proxy_pool.get_pool_state() if proxy_pool.ready else "loading"
    return {
        "status": "ok",
        "version": f"v{PROXY_VERSION}",
        "models": len(_models_cache),
        "socks5": STATIC_PROXY,
        "proxy_pool": PROXY_POOL_ENABLED,
        "proxy_port_filter": sorted(ALLOWED_PROXY_PORTS) if PROXY_PORT_FILTER_ENABLED else None,
        "pool_state": pool_state,
        "pool_size": len(proxy_pool.hot) if PROXY_POOL_ENABLED else None,
        "tokens": dict(_tokens),
        "endpoints": ["/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/models"],
    }


app.add_route("/v1/models", _json(list_models), methods=["GET"])
app.add_route("/v1/chat/completions", _json(chat_completions), methods=["POST"])
app.add_route("/v1/messages", _json(messages), methods=["POST"])
app.add_route("/v1/responses", _json(handle_responses), methods=["POST"])
app.add_route("/health", _json(health), methods=["GET"])


# ── Start ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"OpenCode Free Proxy v{PROXY_VERSION} on http://{HOST}:{PORT}")
    if PROXY_POOL_ENABLED:
        print("  Proxy pool: ENABLED (auto-discovers SOCKS5 proxies, rotates on rate-limit)")
    elif STATIC_PROXY:
        print(f"  SOCKS5 proxy: {STATIC_PROXY}")
    else:
        print("  No SOCKS5 proxy configured (use --proxy, --proxy-pool, SOCKS5_PROXY, or OPENCODE_PROXY_POOL=true)")
    print("  OpenAI:    POST /v1/chat/completions")
    print("  Anthropic: POST /v1/messages")
    print("  Models:    GET  /v1/models")
    print("  Health:    GET  /health")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
