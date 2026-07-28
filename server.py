import argparse
import asyncio
import hashlib
import json
import os
import random
import secrets
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from proxy_pool import pool as proxy_pool
from proxy_pool import MAX_RETRIES, REQUEST_CONNECT_TIMEOUT, REQUEST_READ_TIMEOUT

# ── CLI args ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenCode Free Proxy")
    p.add_argument("--port", type=int, default=None, help="Listen port (default: 6446)")
    p.add_argument("--host", default=None, help="Listen host (default: 0.0.0.0)")
    p.add_argument("--proxy", default=None, help="Static SOCKS5 proxy (socks5://host:port)")
    p.add_argument("--proxy-pool", action="store_true", default=None, help="Enable SOCKS5 proxy pool with auto-rotation on rate-limit")
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

# Proxy pool: enabled by --proxy-pool or OPENCODE_PROXY_POOL=true
_pp_env = os.environ.get("OPENCODE_PROXY_POOL", "").lower()
PROXY_POOL_ENABLED = (
    True if args.proxy_pool else
    False if args.proxy_pool is False else
    _pp_env in ("1", "true", "yes")
)

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

# ── App ───────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Start background model discovery
    asyncio.ensure_future(_periodic_model_refresh())
    if PROXY_POOL_ENABLED:
        _log("Proxy pool enabled, loading SOCKS5 proxies in background...")
        asyncio.ensure_future(proxy_pool.load())
        _log("  (pool will be ready once verification completes)")
    yield
    await proxy_pool.close()

app = FastAPI(lifespan=_lifespan)

PORT = args.port or int(os.environ.get("PORT", "6446"))
HOST = args.host or os.environ.get("HOST", "0.0.0.0")
OC_VERSION = "1.15.0"
PROXY_VERSION = "10"

# ── API Keys ──────────────────────────────────────────────────────

API_KEY = args.api_key or os.environ.get("LOCAL_KEY") or os.environ.get("API_KEY")


def auth(request: Request) -> str | None:
    if not API_KEY:
        return "user"
    hdr = request.headers.get("authorization") or request.headers.get("x-api-key") or ""
    tok = hdr[7:] if hdr.startswith("Bearer ") else hdr
    if tok == API_KEY:
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
    with open("proxy.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def oc_id(prefix: str) -> str:
    ts = format(int(time.time() * 1000), "x")
    rnd = secrets.token_urlsafe(12)[:16]
    return f"{prefix}_{ts}{rnd}"


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


# Session per conversation (hash-based lookup)
_user_sessions: dict[str, dict[str, str]] = {}


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
            sessions[full_h] = sessions[h]
            _log(f"SESSION match prefix={n}/{len(messages)} -> {sessions[h]}")
            return sessions[h]

    new_id = f"ses_{oc_id('ses')}"
    full_h = _hash_messages(messages)
    sessions[full_h] = new_id
    _log(f"SESSION new {new_id} (msgs={len(messages)})")
    return new_id


def force_new_session(user: str, messages: list[dict]) -> str:
    new_id = f"ses_{oc_id('ses')}"
    if user not in _user_sessions:
        _user_sessions[user] = {}
    full_h = _hash_messages(messages)
    _user_sessions[user][full_h] = new_id
    _log(f"SESSION forced new {new_id} (msgs={len(messages)})")
    return new_id


async def _backoff(attempt: int, base: float = 1.0, max_delay: float = 10.0):
    """Exponential backoff with jitter."""
    delay = min(base * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.25)
    await asyncio.sleep(delay + jitter)


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

            # Force new session when rotating proxies
            if attempt > 0:
                headers["x-opencode-session"] = force_new_session(user, messages)
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
            _log(f"[zen] Request failed (attempt {attempt}): {e}")
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_failure(proxy_addr)
            last_error = e
            if attempt < attempts:
                await _backoff(attempt)
            continue

        body_bytes = await resp.aread()
        try:
            data = json.loads(body_bytes)
        except json.JSONDecodeError:
            data = {}

        is_429 = resp.status_code == 429
        is_rate_limit = is_429 or "FreeUsageLimitError" in body_bytes.decode()

        if is_rate_limit:
            err_msg = (data.get("error") or {}).get("message") or "Rate limit exceeded"
            _log(f"[zen] 429 (attempt {attempt}): {err_msg}")
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_ratelimit(proxy_addr)
            if attempt < attempts:
                await _backoff(attempt)
                continue
            return JSONResponse(
                status_code=429,
                content={"error": {"message": err_msg + " (free model rate limit)", "type": "rate_limit_error", "code": "rate_limit_exceeded"}},
            )

        if resp.status_code >= 400:
            err_msg = (data.get("error") or {}).get("message") or f"HTTP {resp.status_code}"
            is_context_exceeded = "context_length_exceeded" in (data.get("error") or {}).get("code", "")
            _log(f"[zen] Error {resp.status_code}: {err_msg}")
            if PROXY_POOL_ENABLED and proxy_addr and not is_context_exceeded:
                proxy_pool.report_failure(proxy_addr)
            if not is_context_exceeded and attempt < attempts:
                await _backoff(attempt)
                continue
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": {"message": err_msg, "type": "upstream_error"}},
            )

        return data

    if last_error:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Upstream error after {attempts + 1} attempts: {last_error}", "type": "upstream_error"}},
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
                    client = _default_client
                    proxy_addr = None
                else:
                    p = proxy_pool.current
                    proxy_addr = p["address"]
                    client = proxy_pool.get_client(f"socks5://{proxy_addr}")
            else:
                p = proxy_pool.current
                proxy_addr = p["address"]
                client = proxy_pool.get_client(f"socks5://{proxy_addr}")

            if attempt > 0:
                headers["x-opencode-session"] = force_new_session(user, messages)
        else:
            client = _default_client
            proxy_addr = None

        try:
            upstream_request = client.build_request(
                "POST", "/zen/v1/chat/completions", json=req_body, headers=headers
            )
            resp = await client.send(upstream_request, stream=True)
        except Exception as e:
            _log(f"[zen] Stream request failed (attempt {attempt}): {e}")
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_failure(proxy_addr)
            last_error = e
            if attempt < attempts:
                await _backoff(attempt)
            continue

        if resp.status_code == 429:
            raw = await resp.aread()
            try:
                data = json.loads(raw)
                err_msg = (data.get("error") or {}).get("message") or "Rate limit exceeded"
            except Exception:
                err_msg = "Rate limit exceeded"
            _log(f"[zen] Stream 429 (attempt {attempt}): {err_msg}")
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_ratelimit(proxy_addr)
            await resp.aclose()
            if attempt < attempts:
                await _backoff(attempt)
                continue
            yield f'data: {json.dumps({"error": {"message": err_msg + " (free model rate limit)", "type": "rate_limit_error", "code": "rate_limit_exceeded"}})}\n\n'
            return

        if resp.status_code >= 400:
            raw = await resp.aread()
            is_context_exceeded = b"context_length_exceeded" in raw
            _log(f"[zen] Stream error {resp.status_code}: {raw[:500]}")
            if PROXY_POOL_ENABLED and proxy_addr and not is_context_exceeded:
                proxy_pool.report_failure(proxy_addr)
            await resp.aclose()
            if not is_context_exceeded and attempt < attempts:
                await _backoff(attempt)
                continue
            yield f'data: {json.dumps({"error": {"message": f"Upstream error {resp.status_code}", "type": "upstream_error"}})}\n\n'
            return

        # Success — stream the response
        chunk_count = 0
        try:
            try:
                async for line in resp.aiter_lines():
                    if line:
                        if line.startswith(":"):
                            continue
                        chunk_count += 1
                        yield line + "\n\n"
                        if line.strip() == "data: [DONE]":
                            break
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.TransportError) as e:
                _log(f"[zen] Stream interrupted: {e}")
                last_error = e
                if attempt < attempts:
                    await _backoff(attempt)
                    continue
                yield f'data: {json.dumps({"error": {"message": f"Stream interrupted: {e}", "type": "upstream_error"}})}\n\n'
                return
        finally:
            await resp.aclose()
        _log(f"STREAM done: {chunk_count} chunks")
        return

    # All retries exhausted
    yield f'data: {json.dumps({"error": {"message": f"Stream failed after {attempts + 1} attempts: {last_error}", "type": "upstream_error"}})}\n\n'


async def _zen_stream_anthropic_with_retry(
    req_body: dict,
    headers: dict,
    user: str,
    messages: list[dict],
    model: str,
    input_tokens: int,
    max_retries: int = None,
):
    """Anthropic-format streaming with proxy pool retry on 429."""
    msg_id = oc_id("msg")
    content_idx = 0
    tool_idx = -1
    text_closed = False
    output_tokens = 0
    headers_sent = False
    last_error = None
    attempts = MAX_RETRIES if max_retries is None else max_retries

    def send_sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    for attempt in range(attempts + 1):
        if PROXY_POOL_ENABLED:
            if not proxy_pool.ready:
                await proxy_pool.load()

            if not await proxy_pool.select():
                _log(f"[pool] No proxy ({proxy_pool.get_pool_state()}), forcing refresh")
                await proxy_pool.force_refresh()
                if not await proxy_pool.select():
                    _log("[pool] Fallback to direct")
                    client = _default_client
                    proxy_addr = None
                else:
                    p = proxy_pool.current
                    proxy_addr = p["address"]
                    client = proxy_pool.get_client(f"socks5://{proxy_addr}")
            else:
                p = proxy_pool.current
                proxy_addr = p["address"]
                client = proxy_pool.get_client(f"socks5://{proxy_addr}")

            if attempt > 0:
                headers["x-opencode-session"] = force_new_session(user, messages)
        else:
            client = _default_client
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
                    if attempt < attempts:
                        await _backoff(attempt)
                        continue
                    yield send_sse("error", {"type": "error", "error": {"type": "rate_limit_error", "message": err_msg + " (free model rate limit)"}})
                    return

                if resp.status_code >= 400:
                    raw = await resp.aread()
                    is_context_exceeded = b"context_length_exceeded" in raw
                    _log(f"[zen] Anthropic stream error {resp.status_code}: {raw[:300]}")
                    if PROXY_POOL_ENABLED and proxy_addr and not is_context_exceeded:
                        proxy_pool.report_failure(proxy_addr)
                    if not is_context_exceeded and attempt < attempts:
                        await _backoff(attempt)
                        continue
                    yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": f"HTTP {resp.status_code}"}})
                    return

                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue

                    if not headers_sent:
                        trimmed = raw_line.strip()
                        if trimmed.startswith("{") and ("FreeUsageLimitError" in trimmed or '"error"' in trimmed):
                            try:
                                parsed = json.loads(trimmed)
                                if parsed.get("error") or parsed.get("type") == "error":
                                    err_msg = (parsed.get("error") or {}).get("message") or parsed.get("message") or "Rate limit"
                                    _log(f"[zen] Anthropic error in body (attempt {attempt}): {err_msg}")
                                    if PROXY_POOL_ENABLED and proxy_addr:
                                        proxy_pool.report_ratelimit(proxy_addr)
                                    if attempt < attempts:
                                        break
                                    yield send_sse("error", {"type": "error", "error": {"type": "rate_limit_error", "message": err_msg + " (free model rate limit)"}})
                                    return
                            except json.JSONDecodeError:
                                pass

                    if raw_line.startswith("data: "):
                        payload = raw_line[6:].strip()
                        if payload == "[DONE]":
                            total_blocks = (1 if content_idx > 0 and not text_closed else 0) + (tool_idx + 1 if tool_idx >= 0 else 0)
                            for i in range(total_blocks):
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

                        delta = (parsed.get("choices") or [{}])[0].get("delta") or {}
                        if not delta:
                            continue

                        if not headers_sent:
                            headers_sent = True
                            yield send_sse("message_start", {
                                "type": "message_start",
                                "message": {
                                    "id": msg_id, "type": "message", "role": "assistant", "content": [],
                                    "model": model, "stop_reason": None,
                                    "usage": {"input_tokens": input_tokens or 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
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
                            total_blocks = (1 if content_idx > 0 and not text_closed else 0) + (tool_idx + 1 if tool_idx >= 0 else 0)
                            for i in range(total_blocks):
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
                    if last_error:
                        await _backoff(attempt)
                    continue
                return

        except Exception as e:
            _log(f"[zen] Anthropic stream HTTP error (attempt {attempt}): {e}")
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_failure(proxy_addr)
            last_error = e
            if attempt < attempts:
                await _backoff(attempt)
                continue
            if not headers_sent:
                yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": str(e)}})
            return

    if not headers_sent:
        yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": f"Stream failed after {attempts + 1} attempts: {last_error}"}})


# ── Anthropic Messages → OpenAI conversion ────────────────────────

def anthropic_to_openai(body: dict) -> tuple[list[dict], list[dict] | None]:
    messages = []

    if body.get("system"):
        sys_val = body["system"]
        if isinstance(sys_val, str):
            sys_text = sys_val
        elif isinstance(sys_val, list):
            sys_text = "\n".join(b.get("text", "") for b in sys_val)
        else:
            sys_text = ""
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": msg["role"], "content": content})
        elif isinstance(content, list):
            text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
            tool_uses = [b for b in content if b.get("type") == "tool_use"]

            if tool_uses and msg.get("role") == "assistant":
                messages.append({
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
                })
            elif any(b.get("type") == "tool_result" for b in content):
                for b in content:
                    if b.get("type") == "tool_result":
                        c = b.get("content")
                        if isinstance(c, str):
                            result_text = c
                        elif isinstance(c, list):
                            result_text = "\n".join(x.get("text", "") for x in c)
                        else:
                            result_text = ""
                        messages.append({
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": result_text,
                        })
            else:
                messages.append({"role": msg["role"], "content": text})
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
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
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
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


# ── Routes: OpenAI format ─────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
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


@app.post("/v1/chat/completions")
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

    _log(f"REQUEST model={model} stream={stream} msgs={len(messages or [])} tools={len(tools or [])}")

    model = _normalize_model(model)
    if model not in _models_cache:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Unknown model: {model}. Available: {', '.join(_models_cache)}"}},
        )

    session_id = get_session(user, messages)
    _log(f"session={session_id[:16]}... {user} {model} {'stream' if stream else 'sync'} msgs={len(messages or [])}")

    req_body, headers = zen_request(model, messages, stream, tools, tool_choice, session_id)

    if stream:
        return StreamingResponse(
            _zen_stream_with_retry(req_body, headers, user, messages or []),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _zen_request_with_retry(req_body, headers, user, messages or [])


# ── Routes: Anthropic Messages format ─────────────────────────────

@app.post("/v1/messages")
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

    _log(f"[ANT] {user} {model} {'stream' if stream else 'sync'} msgs={len(oai_messages)}")

    req_body, headers = zen_request(model, oai_messages, stream, tools, None, session_id)

    if stream:
        return StreamingResponse(
            _zen_stream_anthropic_with_retry(req_body, headers, user, oai_messages, model, input_tokens),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        data = await _zen_request_with_retry(req_body, headers, user, oai_messages)
        if isinstance(data, JSONResponse):
            return data
        if not data.get("choices"):
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "upstream_error", "message": "Invalid upstream response"}},
            )
        return openai_to_anthropic(data, model, input_tokens)


# ── /v1/responses (Responses API for Codex openai_base_url) ─────────────

@app.post("/v1/responses")
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
    req_body, headers = zen_request(zen_model, messages, stream, tools, body.get("tool_choice"), get_session(user, messages))

    if stream:
        return StreamingResponse(
            _zen_stream_with_retry(req_body, headers, user, messages),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _zen_request_with_retry(req_body, headers, user, messages)

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


@app.get("/health")
async def health():
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
        "endpoints": ["/v1/chat/completions", "/v1/messages", "/v1/models"],
    }


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
