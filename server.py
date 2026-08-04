import argparse
import asyncio
import hashlib
import json
import os
import random
import secrets
import sys
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
    MAX_RETRIES,
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
PROXY_VERSION = "14"

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
    if "[zen]" in msg:
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

_TOKENS_FILE = _BASE_DIR / "tokens.json"
_DEFAULT_TOKENS = {"input": 0, "output": 0, "cache_hit": 0, "cache_miss": 0}


def _load_tokens():
    global _tokens
    try:
        with open(_TOKENS_FILE, encoding="utf-8") as f:
            _tokens = {**_DEFAULT_TOKENS, **(json.load(f) or {})}
    except Exception:
        _tokens = dict(_DEFAULT_TOKENS)


def _add_tokens(inp: int = 0, out: int = 0, cache_hit: int = 0, cache_miss: int = 0):
    _tokens["input"] += max(0, inp or 0)
    _tokens["output"] += max(0, out or 0)
    _tokens["cache_hit"] += max(0, cache_hit or 0)
    _tokens["cache_miss"] += max(0, cache_miss or 0)
    try:
        _TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_TOKENS_FILE, "w", encoding="utf-8") as f:
            json.dump(_tokens, f, indent=2)
    except OSError:
        pass


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
    """Human-readable exception for logs: type name always, message when present."""
    if e is None:
        return "unknown"
    desc = type(e).__name__
    if str(e):
        desc += f": {e}"
    cause = getattr(e, "__cause__", None)
    if cause is not None and cause is not e:
        desc += f" (caused by {_exc_desc(cause)})"
    return desc


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
    """Return a terminal OpenAI SSE error event."""
    error = {"message": message, "type": error_type}
    if code:
        error["code"] = code
    return (
        f"data: {json.dumps({'error': error})}\n\n"
        "data: [DONE]\n\n"
    )


def _stream_preview(value: str, limit: int = 160) -> str:
    """Return a bounded, escaped preview suitable for diagnostics."""
    return repr(value[:limit])


def _blocks_text(content) -> str:
    """Extract plain text from Anthropic content blocks (str or list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content) if content is not None else ""


# Dynamically discovered free models from Zen API (fallback if fetch fails)
_MODELS_FALLBACK = [
    "deepseek-v4-flash-free",
    "mimo-v2.5-free",
    "ling-3.0-flash-free",
    "nemotron-3-ultra-free",
    "north-mini-code-free",
    "laguna-s-2.1-free",
]
_models_cache: list[str] = list(_MODELS_FALLBACK)
_models_meta: dict[str, dict] = {}  # model_id -> {name, limit, modalities}
_MODELS_REFRESH_SECS = 18000  # refresh every 5 hours


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
            free = [m for m in all_models if "free" in m.lower()]
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
    """Periodically refresh the free model list from Zen API."""
    await _fetch_free_models()
    while True:
        await asyncio.sleep(_MODELS_REFRESH_SECS)
        await _fetch_free_models()


def _normalize_model(model: str) -> str:
    """Strip ocf- prefix if present (backward compat)."""
    return model[4:] if model.startswith("ocf-") else model


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


def zen_request(model, messages, stream, tools, tool_choice, session_id):
    model = _normalize_model(model)
    req_body: dict = {"model": model, "messages": messages, "stream": bool(stream)}
    if tools:
        req_body["tools"] = tools
    if tool_choice:
        req_body["tool_choice"] = tool_choice

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

async def _zen_request_with_retry(
    req_body: dict,
    headers: dict,
    user: str,
    messages: list[dict],
    session_id: str = None,
    max_retries: int = None,
):
    """Non-streaming Zen API call with proxy pool retry on 429."""
    last_error = None
    attempts = MAX_RETRIES if max_retries is None else max_retries

    for attempt in range(attempts + 1):
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
            is_context_exceeded = "context_length_exceeded" in (data.get("error") or {}).get("code", "")
            _log(f"[zen] Error {resp.status_code}: {err_msg}")
            # Not a proxy failure: 4xx/5xx are upstream or request errors that
            # repeat identically on every proxy, so never blacklist for them.
            if not is_context_exceeded and attempt < attempts:
                await _backoff(attempt)
                continue
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": {"message": err_msg, "type": "upstream_error"}},
            )

        usage = (data.get("usage") or {})
        if isinstance(usage, dict) and ("prompt_tokens" in usage or "completion_tokens" in usage):
            _add_tokens(
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
    req_body: dict,
    headers: dict,
    user: str,
    messages: list[dict],
    session_id: str = None,
    max_retries: int = None,
):
    """Streaming Zen API call with proxy pool retry on 429/403 and ReadError recovery."""
    last_error = None
    attempts = MAX_RETRIES if max_retries is None else max_retries

    for attempt in range(attempts + 1):
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

        try:
            upstream_request = client.build_request(
                "POST", "/zen/v1/chat/completions", json=req_body, headers=headers
            )
            resp = await client.send(upstream_request, stream=True)
        except Exception as e:
            _log(f"[zen] Stream request failed (attempt {attempt}): {_exc_desc(e)}")
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_failure(proxy_addr)
            last_error = e
            if attempt < attempts:
                await _backoff(attempt)
            continue

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
                    continue
                yield _openai_stream_error(f"Upstream error: {_exc_desc(e)}")
                return
            _log(f"[zen] Stream 429 (attempt {attempt}): {err_msg}")
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
                    continue
                yield _openai_stream_error(f"Upstream error: {_exc_desc(e)}")
                return
            is_context_exceeded = b"context_length_exceeded" in raw
            _log(f"[zen] Stream error {resp.status_code}: {raw[:500]}")
            if resp.status_code == 400:
                _log_reasoning_diag(req_body)

            # Not a proxy failure: 4xx/5xx are upstream or request errors that
            # repeat identically on every proxy, so never blacklist for them.
            await resp.aclose()
            if not is_context_exceeded and attempt < attempts:
                await _backoff(attempt)
                continue
            yield _openai_stream_error(f"Upstream error {resp.status_code}")
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
                        if not streamed_any and attempt < attempts:
                            retry_stream = True
                            break
                        yield _openai_stream_error(str(err))
                        return

                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        stream_completed = True
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
                        if not streamed_any and attempt < attempts:
                            retry_stream = True
                            break
                        yield _openai_stream_error(str(err))
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
                        if not streamed_any and attempt < attempts:
                            retry_stream = True
                            break
                        yield _openai_stream_error(str(err))
                        return

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
                        if not streamed_any and attempt < attempts:
                            retry_stream = True
                            break
                        yield _openai_stream_error(err_msg)
                        return

                    # Capture emitted reasoning_content for re-injection on later turns
                    if session_id and '"delta"' in line:
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
                        fr = first_choice.get("finish_reason")
                        if fr:
                            if _reason_buf and _content_buf:
                                _remember_reasoning(session_id, _content_buf, _reason_buf)
                            _reason_buf = ""
                            _content_buf = ""
                    yield line + "\n\n"
                    streamed_any = True
                    if '"usage"' in line:
                        u = piece.get("usage") or {}
                        if isinstance(u, dict) and (u.get("prompt_tokens") or u.get("completion_tokens")):
                            _add_tokens(
                                u.get("prompt_tokens") or 0,
                                u.get("completion_tokens") or 0,
                                u.get("prompt_cache_hit_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                                u.get("prompt_cache_miss_tokens") or 0,
                            )
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.TransportError) as e:
                _log(f"[zen] Stream interrupted: {_exc_desc(e)}")
                last_error = e
                # A torn-down tunnel is a proxy failure: flag it so the next
                # request does not re-select this proxy and pay the timeout
                # again (report_failure also clears self.current).
                if PROXY_POOL_ENABLED and proxy_addr:
                    proxy_pool.report_failure(proxy_addr)
                # Never retry once bytes have already been sent to the client; a
                # fresh request would replay the whole stream and corrupt output.
                if not streamed_any and attempt < attempts:
                    retry_stream = True
                else:
                    yield _openai_stream_error(f"Stream interrupted: {_exc_desc(e)}")
                    return
        finally:
            try:
                await resp.aclose()
            except Exception:
                pass
        if retry_stream:
            await _backoff(attempt)
            continue
        if not stream_completed:
            err = ValueError("upstream stream ended before [DONE]")
            _log(f"[zen] {_exc_desc(err)} (attempt {attempt})")
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_failure(proxy_addr)
            last_error = err
            if not streamed_any and attempt < attempts:
                await _backoff(attempt)
                continue
            yield _openai_stream_error(str(err))
        return

    # All retries exhausted
    yield _openai_stream_error(
        f"Stream failed after {attempts + 1} attempts: {_exc_desc(last_error)}"
    )


async def _zen_stream_anthropic_with_retry(
    req_body: dict,
    headers: dict,
    user: str,
    messages: list[dict],
    model: str,
    input_tokens: int,
    session_id: str = None,
    max_retries: int = None,
):
    """Anthropic-format streaming with proxy pool retry on 429."""
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

    for attempt in range(attempts + 1):
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
                    is_context_exceeded = b"context_length_exceeded" in raw
                    _log(f"[zen] Anthropic stream error {resp.status_code}: {raw[:300]}")
                    # Not a proxy failure: 4xx/5xx are upstream or request
                    # errors that repeat identically on every proxy.
                    if not is_context_exceeded and attempt < attempts:
                        await _backoff(attempt)
                        continue
                    yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": f"HTTP {resp.status_code}"}})
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
                            if attempt < attempts:
                                break
                            yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": err_msg}})
                            return

                    if raw_line.startswith("data: "):
                        payload = raw_line[6:].strip()
                        if payload == "[DONE]":
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
                            _add_tokens(
                                u.get("prompt_tokens") or 0,
                                u.get("completion_tokens") or 0,
                                u.get("prompt_cache_hit_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                                u.get("prompt_cache_miss_tokens") or 0,
                            )

                        delta = (parsed.get("choices") or [{}])[0].get("delta") or {}
                        if not delta:
                            continue

                        # Capture upstream reasoning for re-injection in this session
                        if session_id and (
                            delta.get("reasoning_content") or ((parsed.get("choices") or [{}])[0].get("finish_reason"))
                        ):
                            if delta.get("reasoning_content"):
                                _reason_buf += delta["reasoning_content"]
                            if delta.get("content"):
                                _content_buf += delta["content"]
                            if (parsed.get("choices") or [{}])[0].get("finish_reason"):
                                if _reason_buf and _content_buf:
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
                            yield send_sse("message_stop", {"type": "message_stop"})
                            return

                # If we broke out before emitting headers, retry with another proxy.
                if not headers_sent and attempt < attempts:
                    await _backoff(attempt)
                    continue
                return

        except Exception as e:
            _log(f"[zen] Anthropic stream HTTP error (attempt {attempt}): {_exc_desc(e)}")
            # Only transport-level failures are proxy failures; a bug in our
            # translation code must not blacklist a healthy proxy.
            if PROXY_POOL_ENABLED and proxy_addr and isinstance(e, httpx.HTTPError):
                proxy_pool.report_failure(proxy_addr)
            last_error = e
            # Never retry once bytes have already been sent to the client; a
            # fresh request would replay the whole stream and corrupt output.
            if not streamed_any and attempt < attempts:
                await _backoff(attempt)
                continue
            if not streamed_any:
                yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": _exc_desc(e)}})
            else:
                # Message was already started: close any blocks left open by
                # the aborted attempt and terminate the message cleanly.
                for i in close_indices():
                    yield send_sse("content_block_stop", {"type": "content_block_stop", "index": i})
                yield send_sse("message_stop", {"type": "message_stop"})
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
    if model not in _models_cache:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Unknown model: {model}. Available: {', '.join(_models_cache)}"}},
        )

    session_id = get_session(user, messages)

    up_messages = _prepare_upstream_messages(session_id, _normalize_messages(messages))
    req_body, headers = zen_request(model, up_messages, stream, tools, tool_choice, session_id)

    if stream:
        return StreamingResponse(
            _zen_stream_with_retry(req_body, headers, user, messages or [], session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _zen_request_with_retry(req_body, headers, user, messages or [], session_id)


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

    if model not in _models_cache:
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": f"Unknown model: {model}. Available: {', '.join(_models_cache)}"}},
        )

    oai_messages, tools = anthropic_to_openai(body)
    session_id = get_session(user, oai_messages)
    input_tokens = len(json.dumps(oai_messages)) // 4

    up_messages = _prepare_upstream_messages(session_id, oai_messages)
    req_body, headers = zen_request(model, up_messages, stream, tools, None, session_id)

    if stream:
        return StreamingResponse(
            _zen_stream_anthropic_with_retry(req_body, headers, user, oai_messages, model, input_tokens, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        data = await _zen_request_with_retry(req_body, headers, user, oai_messages, session_id)
        if isinstance(data, JSONResponse):
            return data
        if not data.get("choices"):
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "upstream_error", "message": "Invalid upstream response"}},
            )
        return openai_to_anthropic(data, model, input_tokens)


# ── /v1/responses (Responses API for Codex openai_base_url) ─────────────

async def handle_responses(request: Request):
    body = await request.json()
    model = body.get("model", "")
    stream = body.get("stream", False)
    messages = _input_to_messages(body.get("input", ""))
    tools = _extract_tools(body)

    user = auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": {"message": "Invalid API key"}})

    zen_model = _map_model(model)
    messages = _normalize_messages(messages)
    session_id = get_session(user, messages)
    up_messages = _prepare_upstream_messages(session_id, messages)
    req_body, headers = zen_request(zen_model, up_messages, stream, tools, body.get("tool_choice"), session_id)

    if stream:
        return StreamingResponse(
            _zen_stream_with_retry(req_body, headers, user, messages, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _zen_request_with_retry(req_body, headers, user, messages, session_id)

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
    tools = body.get("tools", [])
    return [t for t in tools if isinstance(t, dict)] if tools else None

def _map_model(model: str) -> str:
    m = model.lower().replace("-", "").replace("_", "")
    if m in ("opencodedefault",):
        return "ling-3.0-flash-free"
    if m in ("opencodefast",):
        return "deepseek-v4-flash-free"
    if m in ("opencodesmart",):
        return "mimo-v2.5-free"
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
        "pool_state": pool_state,
        "pool_size": len(proxy_pool.hot) if PROXY_POOL_ENABLED else None,
        "tokens": dict(_tokens),
        "endpoints": ["/v1/chat/completions", "/v1/messages", "/v1/models"],
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
    print(f"  Models: {', '.join(_models_cache)}")
    if API_KEY:
        print(f"  API key:   {API_KEY[:8]}...")
    else:
        print("  API key:   (none - open access)")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
