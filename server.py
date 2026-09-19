import argparse
import asyncio
import contextvars
import copy
import hashlib
import json
import os
import random
import secrets
import sys
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse

from proxy_pool import pool as proxy_pool
from proxy_pool import (
    ALLOWED_PROXY_PORTS,
    MAX_RETRIES,
    PROXY_PORT_FILTER_ENABLED,
    REQUEST_CONNECT_TIMEOUT,
    REQUEST_READ_TIMEOUT,
    STREAM_READ_TIMEOUT,
)

from nvidia_pool import pool as nvidia_keys
from nvidia_proxy import is_nvidia_model, nvidia_model_id, nvidia_models

from amd_pool import pool as amd_keys
from amd_proxy import is_amd_model, amd_model_id, amd_models

_BASE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent

# ── CLI args ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenCode Free Proxy")
    p.add_argument("--port", type=int, default=6446, help="Listen port (default: 6446)")
    p.add_argument("--host", default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    p.add_argument("--proxy", default=None, help="Static SOCKS5 proxy (socks5://host:port)")
    p.add_argument("--proxy-pool", action=argparse.BooleanOptionalAction, default=True, help="Enable SOCKS5 proxy pool with transport-failure and per-proxy 429 rotation (default: on, use --no-proxy-pool to disable)")
    p.add_argument("--api-key", default=None, help="API key for client auth")
    p.add_argument("--allow-direct-fallback", action=argparse.BooleanOptionalAction, default=False, help="Allow one direct (no-proxy) fetch after proxy-pool exhaustion on retriable statuses (default: off; direct fetch exposes this server's own IP)")
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

# Exhaust-then-direct fallback: default OFF (direct fetch exposes this
# server's own IP). Opt in with --allow-direct-fallback or
# OPENCODE_ALLOW_DIRECT_FALLBACK=1/true/yes.
_df_env = os.environ.get("OPENCODE_ALLOW_DIRECT_FALLBACK", "").lower()
ALLOW_DIRECT_FALLBACK = bool(args.allow_direct_fallback)
if _df_env:
    ALLOW_DIRECT_FALLBACK = _df_env in ("1", "true", "yes", "on")

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
# Dedicated no-proxy clients for the exhaust-then-direct fallback attempt.
# These always bypass STATIC_PROXY / env proxies so the fallback is truly direct.
_direct_client = httpx.AsyncClient(
    base_url="https://opencode.ai",
    timeout=httpx.Timeout(
        connect=REQUEST_CONNECT_TIMEOUT,
        read=REQUEST_READ_TIMEOUT,
        write=REQUEST_READ_TIMEOUT,
        pool=REQUEST_CONNECT_TIMEOUT,
    ),
    proxy=None,
    trust_env=False,
)
_stream_direct_client = httpx.AsyncClient(
    base_url="https://opencode.ai",
    timeout=httpx.Timeout(
        connect=REQUEST_CONNECT_TIMEOUT,
        read=STREAM_READ_TIMEOUT,
        write=STREAM_READ_TIMEOUT,
        pool=REQUEST_CONNECT_TIMEOUT,
    ),
    proxy=None,
    trust_env=False,
)

# ── App ───────────────────────────────────────────────────────────

def _spawn_background(coro):
    """Spawn a background task with a clean req context (no inherited tag)."""
    _clear_req_ctx()
    return asyncio.ensure_future(coro)


@asynccontextmanager
async def _lifespan(app: Starlette):
    # Start background model discovery
    _spawn_background(_periodic_model_refresh())
    if PROXY_POOL_ENABLED:
        _log("Proxy pool enabled, loading SOCKS5 proxies in background...")
        _spawn_background(proxy_pool.load())
        _log("  (pool will be ready once verification completes)")
    yield
    try:
        await _flush_background_io()
    except Exception:
        pass
    try:
        _persist_tokens_sync()
    except Exception:
        pass
    try:
        _save_reasoning_sync()()
    except Exception:
        pass
    await proxy_pool.close()
    await _default_client.aclose()
    await _stream_default_client.aclose()
    await _direct_client.aclose()
    await _stream_direct_client.aclose()

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
OC_VERSION = "1.18.31"
PROXY_VERSION = "19"
# Native OpenCode project id: sha1 hex of "git-remote:<normalized remote>".
# The console/gate inspects x-opencode-project; the real CLI hashes the
# user's project remote (verified against upstream packages/core/project.ts).
_ZEN_PROJECT_ID = hashlib.sha1(b"git-remote:github.com/JulienMaille/opencode-free-proxy").hexdigest()

# ── API Keys ──────────────────────────────────────────────────────

API_KEY = args.api_key or os.environ.get("LOCAL_KEY") or os.environ.get("API_KEY")

_AUTH_OPEN_WARNED = False


def _is_loopback_host(host: str | None) -> bool:
    try:
        h = (host or "").strip().lower()
    except Exception:
        return False
    return h in ("127.0.0.1", "localhost", "::1")


def auth(request: Request) -> str | None:
    if not API_KEY:
        # Fail closed for non-loopback clients (default --host 0.0.0.0 listens
        # on all interfaces): an unset key must NOT become a LAN-open proxy.
        # Open mode is allowed only for loopback clients so local tools stay
        # compatible. Gate on the client address, not the bind HOST.
        try:
            client_host = request.client.host if request.client else None
        except Exception:
            client_host = None
        if not _is_loopback_host(client_host):
            return None
        global _AUTH_OPEN_WARNED
        if not _AUTH_OPEN_WARNED:
            _AUTH_OPEN_WARNED = True
            print(
                "[auth] WARNING: no API key configured (LOCAL_KEY/API_KEY/--api-key unset) — "
                "running OPEN for loopback clients only; requests from other hosts get 401. "
                "Set an API key before exposing this port to the network.",
                flush=True,
            )
        return "user"
    hdr = request.headers.get("authorization") or request.headers.get("x-api-key") or ""
    tok = hdr[7:] if hdr.startswith("Bearer ") else hdr
    if tok and secrets.compare_digest(tok, API_KEY):
        return "user"
    return None


# ── Helpers ───────────────────────────────────────────────────────

def _redact_for_log(text: str) -> str:
    """Redact Authorization / Bearer / nvapi- secrets so logs never leak keys."""
    try:
        s = str(text)
        s = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-~+/=]+", r"\1***", s)
        s = re.sub(r"nvapi-[A-Za-z0-9._\-]+", "nvapi-***", s)
        s = re.sub(r"(?i)(authorization['\"\s:=]+)([A-Za-z0-9._\-~+/=]+)", r"\1***", s)
        return s
    except Exception:
        return str(text)


_req_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("req_ctx", default="")


def new_request_id() -> str:
    """Generate a per-request correlation id (8 hex chars) and bind it."""
    rid = f"req={secrets.token_hex(4)}"
    _req_ctx.set(rid)
    return rid


def _req_tag() -> str:
    try:
        v = _req_ctx.get()
    except Exception:
        return ""
    return f" {v}" if v else ""


def _log(*a):
    msg = _redact_for_log(f"[{time.strftime('%H:%M:%S')}]" + _req_tag() + " " + " ".join(str(x) for x in a))
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
    _append_log(_BASE_DIR / "proxy.log", msg)


# ── Background disk-I/O tracking (never lose data on shutdown) ──
# _append_log() and _schedule_tokens_flush() offload sync disk writes via
# run_in_executor and register the futures here; _flush_background_io()
# awaits the outstanding ones (called from the lifespan shutdown path).

_BG_IO: set = set()
_BG_IO_LOCK = threading.Lock()


def _track_bg_io(fut):
    try:
        with _BG_IO_LOCK:
            _BG_IO.add(fut)
        fut.add_done_callback(_untrack_bg_io)
    except Exception:
        pass


def _untrack_bg_io(fut):
    try:
        with _BG_IO_LOCK:
            _BG_IO.discard(fut)
    except Exception:
        pass


async def _flush_background_io(timeout: float = 10.0):
    """Await outstanding background disk writes (log/tokens/reasoning)."""
    try:
        with _BG_IO_LOCK:
            pending = list(_BG_IO)
        if pending:
            await asyncio.wait(pending, timeout=timeout)
    except Exception:
        pass


_LOG_MAX_BYTES = 5 * 1024 * 1024  # rotate proxy.log past 5 MB
_LOG_KEEP_BYTES = 1 * 1024 * 1024  # ...keeping the last 1 MB
# Serialize check-then-act rotate+append so concurrent executor writers
# cannot interleave stat>cap -> tail-read -> truncate -> append.
_LOG_LOCK = threading.Lock()


def _append_log_sync(path, msg: str):
    """Append one line, rotating the file down to its tail past the cap."""
    try:
        with _LOG_LOCK:
            if path.exists() and path.stat().st_size > _LOG_MAX_BYTES:
                with open(path, "rb") as f:
                    f.seek(-_LOG_KEEP_BYTES, 2)
                    tail = f.read()
                nl = tail.find(b"\n")
                if nl != -1:
                    tail = tail[nl + 1:]
                with open(path, "wb") as f:
                    f.write(f"[... rotated, kept last {len(tail) // 1024} KB ...]\n".encode("utf-8"))
                    f.write(tail)
            with open(path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
    except OSError:
        pass


def _append_log(path, msg: str):
    """Non-blocking append: offload the sync rotate+write to a worker thread.

    Keeps proxy.log rotation behavior identical to _append_log_sync while
    keeping _log() off the request hot path. Fire-and-forget from the event
    loop; flushed implicitly since the executor write completes — and every
    shutdown path calls _flush_background_io() before exit (see lifespan).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _append_log_sync(path, msg)
        return
    try:
        fut = loop.run_in_executor(None, _append_log_sync, path, msg)
        _track_bg_io(fut)
    except RuntimeError:
        # Loop closing / shutting down: fall back to a best-effort sync write.
        try:
            _append_log_sync(path, msg)
        except OSError:
            pass


_ZEN_ID_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_zen_id_timestamp = 0
_zen_id_counter = 0


def oc_id(prefix: str, descending: bool = False) -> str:
    """Native OpenCode identifier (verbatim port of the CLI's identifier.ts,
    verified against anomalyco/opencode): 6-byte timestamp+counter hex + 14
    random base62 chars → `<prefix>_<26 chars>` (30 total with prefix).
    Sessions descend (~inverted timestamp: newest sorts first), requests
    ascend. A per-millisecond monotonic counter keeps ids unique under the
    same ms; submission can be strided to interleave other id chains."""
    global _zen_id_timestamp, _zen_id_counter
    ts = int(time.time() * 1000)
    if ts != _zen_id_timestamp:
        _zen_id_timestamp = ts
        _zen_id_counter = 0
    _zen_id_counter += 1
    current = (ts << 12) + _zen_id_counter
    value = ~current if descending else current
    time_part = "".join(
        format((value >> (40 - 8 * i)) & 0xFF, "02x") for i in range(6)
    )
    rand = "".join(_ZEN_ID_CHARS[b % 62] for b in secrets.token_bytes(14))
    return f"{prefix}_{time_part}{rand}"


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


def _persist_tokens_sync(snapshot=None):
    try:
        data = snapshot if snapshot is not None else _tokens
        _TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_TOKENS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except OSError:
        pass


# Debounced tokens flush: the hot path only marks dirty + schedules one
# background write per quiet window instead of mkdir/open/write per call.
_tokens_dirty = False
_tokens_flush_scheduled = False
_TOKENS_FLUSH_DELAY = 2.0


def _persist_tokens():
    _persist_tokens_sync()


def _schedule_tokens_flush():
    """Mark tokens dirty and schedule a single background flush.

    Coalesces bursts of _add_tokens() calls into one executor write per
    _TOKENS_FLUSH_DELAY window. Shutdown always flushes synchronously (see
    lifespan), so no data is lost.
    """
    global _tokens_dirty, _tokens_flush_scheduled
    _tokens_dirty = True
    if _tokens_flush_scheduled:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _persist_tokens_sync()
        _tokens_dirty = False
        return
    _tokens_flush_scheduled = True

    async def _delayed_flush():
        global _tokens_dirty, _tokens_flush_scheduled
        try:
            await asyncio.sleep(_TOKENS_FLUSH_DELAY)
            if _tokens_dirty:
                try:
                    # Snapshot on the loop thread so the executor serializes an
                    # immutable copy while _add_tokens() keeps mutating live.
                    try:
                        snapshot = copy.deepcopy(_tokens)
                    except Exception:
                        snapshot = {m: dict(b) for m, b in _tokens.items()}
                    fut = loop.run_in_executor(None, _persist_tokens_sync, snapshot)
                    _track_bg_io(fut)
                    await fut
                except Exception:
                    pass
                _tokens_dirty = False
        finally:
            _tokens_flush_scheduled = False

    try:
        fut = asyncio.ensure_future(_delayed_flush())
        _track_bg_io(fut)
    except RuntimeError:
        _tokens_flush_scheduled = False
        _persist_tokens_sync()
        _tokens_dirty = False


def _add_tokens(model: str, inp: int = 0, out: int = 0, cache_hit: int = 0, cache_miss: int = 0):
    """Accumulate usage counters under the given model id."""
    model = _normalize_model(model) or "unknown"
    bucket = _tokens.setdefault(model, dict(_DEFAULT_TOKENS))
    bucket["input"] += max(0, inp or 0)
    bucket["output"] += max(0, out or 0)
    bucket["cache_hit"] += max(0, cache_hit or 0)
    bucket["cache_miss"] += max(0, cache_miss or 0)
    _schedule_tokens_flush()


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
# Entries older than this are dropped on save: a thinking-mode session that
# has been idle for a full day is gone client-side anyway, so its cached
# thinking is dead weight. Keeps reasoning_cache.json from growing to tens
# of MB (each reasoning blob can be very large).
_REASONING_ENTRY_TTL_SECS = 24 * 3600
_reasoning_file = _BASE_DIR / "reasoning_cache.json"
# _reasoning_cache: session -> content-hash -> {"text", "at"}
_reasoning_cache: dict[str, dict[str, dict]] = {}


def _load_reasoning():
    global _reasoning_cache
    try:
        with open(_reasoning_file, encoding="utf-8") as f:
            raw = json.load(f) or {}
    except Exception:
        _reasoning_cache = {}
        return
    now = time.time()
    fresh: dict[str, dict[str, dict]] = {}
    kept_sessions = kept_entries = 0
    dropped_entries = 0
    for sid, sack in raw.items():
        if not isinstance(sack, dict):
            continue
        fsack: dict[str, dict] = {}
        for key, val in sack.items():
            # Accept the legacy flat shape {hash: text} and migrate it.
            if isinstance(val, str):
                val = {"text": val, "at": 0}
            if (
                isinstance(val, dict)
                and isinstance(val.get("text"), str)
                and now - float(val.get("at") or 0) <= _REASONING_ENTRY_TTL_SECS
            ):
                fsack[key] = val
                kept_entries += 1
            else:
                dropped_entries += 1
        if fsack:
            fresh[sid] = fsack
            kept_sessions += 1
    _reasoning_cache = dict(list(fresh.items())[-_REASONING_CACHE_MAX_SESSIONS:])
    if dropped_entries:
        print(f"[reasoning] load: kept {kept_entries} entries in {kept_sessions} sessions, expired {dropped_entries}", flush=True)


def _prune_reasoning(now: float | None = None) -> int:
    """Drop expired entries and keep the session cap. Returns dropped count."""
    now = time.time() if now is None else now
    dropped = 0
    for sack in _reasoning_cache.values():
        if not isinstance(sack, dict):
            continue
        for key in list(sack):
            val = sack.get(key)
            if not isinstance(val, dict) or now - float(val.get("at") or 0) > _REASONING_ENTRY_TTL_SECS:
                sack.pop(key, None)
                dropped += 1
    for sid in list(_reasoning_cache):
        if not _reasoning_cache[sid]:
            _reasoning_cache.pop(sid, None)
    if len(_reasoning_cache) > _REASONING_CACHE_MAX_SESSIONS:
        overflow = len(_reasoning_cache) - _REASONING_CACHE_MAX_SESSIONS
        for sid in list(_reasoning_cache)[:overflow]:
            _reasoning_cache.pop(sid, None)
            dropped += 1
    return dropped


# Serialize reasoning cache persistence: concurrent _save_reasoning()
# calls must not interleave their prune/snapshot/write steps (reorder/torn
# writes), and the executor writer must see an immutable snapshot.
_reasoning_lock = threading.Lock()


def _save_reasoning_sync():
    """Prune + snapshot + atomic-write the reasoning cache (blocking).

    Runs fully on the calling thread; the lifespan shutdown path calls the
    returned writer inline via ``_save_reasoning_sync()()`` so tmp+replace
    always executes.
    """
    with _reasoning_lock:
        _prune_reasoning()
        try:
            body = copy.deepcopy(_reasoning_cache)
        except Exception:
            body = {sid: dict(sack) for sid, sack in _reasoning_cache.items()}

        def _write():
            try:
                _reasoning_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = _reasoning_file.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(body, f, ensure_ascii=False)
                os.replace(tmp, _reasoning_file)
            except OSError:
                pass

        return _write


def _flush_reasoning_sync():
    """Prune + snapshot + atomic-write entirely in a worker thread."""
    with _reasoning_lock:
        _prune_reasoning()
        try:
            body = copy.deepcopy(_reasoning_cache)
        except Exception:
            body = {sid: dict(sack) for sid, sack in _reasoning_cache.items()}
        try:
            _reasoning_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = _reasoning_file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(body, f, ensure_ascii=False)
            os.replace(tmp, _reasoning_file)
        except OSError:
            pass


def _save_reasoning():
    """Persist the reasoning cache without blocking the event loop.

    Prune + snapshot + tmp+os.replace all run in a thread-pool executor so
    the loop never blocks on deepcopy/prune; the write is fire-and-forget
    but tracked so shutdown can flush it.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _save_reasoning_sync()()  # no running loop (startup path): write inline
        return
    try:
        fut = loop.run_in_executor(None, _flush_reasoning_sync)
        _track_bg_io(fut)
    except RuntimeError:
        _save_reasoning_sync()()


def _remember_reasoning(session_id, content, reasoning):
    """Store reasoning_content emitted for a given assistant content hash."""
    if not session_id or not content or not reasoning:
        return
    key = hashlib.sha256(content.encode("utf-8")).hexdigest()
    with _reasoning_lock:
        sack = _reasoning_cache.get(session_id)
        if not isinstance(sack, dict):
            sack = {}
            _reasoning_cache[session_id] = sack
        sack[key] = {"text": reasoning, "at": time.time()}
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
    """Return a terminal OpenAI SSE error sequence (no assistant content).

    Emits a single ``{"error": ...}`` event followed by ``[DONE]``. Never a
    fake ``stop`` turn: strict clients ingest content chunks as successful
    answers and stall silently on disguised errors.
    """
    error = {"message": message, "type": error_type}
    if code:
        error["code"] = code
    # Terminal failure: emit a bare error event, NO assistant content chunk.
    # The previous shape appended a "[upstream error] ..." content chunk with
    # finish_reason "stop", which strict clients (opencode AI SDK:
    # text-delta + finish-step stop) ingest as a *successful* turn whose text
    # happens to be an error message — the agent then stalls silently instead
    # of surfacing/ retrying a proper failure. A bare error event maps to the
    # SDK's `error` branch (Effect.fail -> halt -> finish "error"), which is
    # visible and retryable. Clients that need finish_reasoned turns only get
    # them on success paths now.
    return f"data: {json.dumps({'error': error})}\n\n" "data: [DONE]\n\n"


def _transport_error_message(exc: BaseException) -> str:
    """Build a mid-stream transport-failure message that harnesses (oh-my-pi)
    auto-retry on transient errors. Its classifier regex-matches the message
    text (e.g. ``connection.?error``), so prefix the raw cause with a standard
    transient phrase; otherwise a dropped proxy connection reads as terminal
    and the harness kills the session instead of retrying."""
    return f"connection error: proxy dropped the upstream stream: {_exc_desc(exc)}"


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
def _is_bare_internal_error(status_code: int, data: dict | None = None, body: str = "") -> bool:
    """Recognize the deterministic model-down 500: bare ``Internal server error``.

    Zen returns exactly ``{"type":"error","error":{"type":"error","message":
    "Internal server error"}}`` with no detail when the model deployment itself
    is down (observed for muse-spark-1.3: identical on direct connections and
    on every proxy exit, while other models succeed on the same proxies).
    Retrying it across proxies only burns every attempt with backoff and delays
    the caller, so it must fail fast with a clear hint instead.
    """
    if status_code != 500:
        return False
    error = data.get("error") if isinstance(data, dict) else None
    msg = (error.get("message") if isinstance(error, dict) else None) or ""
    if str(msg).strip().lower() != "internal server error":
        return False
    return len((body or "").strip()) < 200


def _is_degraded_error(data: dict | None = None, body: str = "") -> bool:
    """Recognize NVIDIA NIM 'DEGRADED function cannot be invoked' 400s.

    NVIDIA returns this when a model deployment is temporarily degraded (often
    a ``{"status":400,"title":"Bad Request","detail":"...DEGRADED..."}`` body).
    It is deployment-level, NOT key-dependent: every key in the pool returns
    the same 400, so retrying/rotating keys only burns the whole pool on
    backoff. It must fail fast and terminal instead.
    """
    error = data.get("error") if isinstance(data, dict) else None
    text = "".join(
        str(value)
        for value in (
            error.get("message") if isinstance(error, dict) else error,
            error.get("detail") if isinstance(error, dict) else None,
            data.get("detail") if isinstance(data, dict) else None,
            data.get("title") if isinstance(data, dict) else None,
            body,
        )
        if value is not None
    ).lower()
    return (
        "degraded function" in text
        or "cannot be invoked" in text
        or ("degraded" in text and "function" in text)
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


def _is_unavailable_error(data: dict | None = None, body: str = "") -> bool:
    """Recognize 'Model is unavailable' provider errors (dead upstream model).

    Observed as HTTP 400 `Error from provider (Console): Upstream request
    failed: Model is unavailable.` for e.g. deepseek-v4-flash-free. Like the
    promotion-ended error this repeats identically on every proxy/retry, so the
    model is retired on first sight instead of burning attempts per request.
    Unlike promotion-ended there is no entitlement phrase to key persistence
    off, so callers retire session-only (persistent=False): a restart
    re-discovers the id, and the blocklist below still hides it from clients.
    """
    if not isinstance(data, dict):
        data = {}
    error = data.get("error")
    if not isinstance(error, dict):
        error = {}
    text = " ".join(
        str(v) for v in (error.get("message"), body) if v is not None
    ).lower()
    return "model is unavailable" in text

def _is_promotion_ended_error(data: dict | None = None, body: str = "", status_code: int | None = None, model: str | None = None) -> bool:
    """Recognize entitlement errors for a model whose free promotion ended.

    Observed as ``{"type":"ModelError","message":"Free promotion has ended
    for ... Free"}`` (HTTP 401). This repeats identically on every proxy and
    every retry — it is account/entitlement level, not transport level — so
    retrying is pure waste. The model is dead until upstream re-lists it.

    A bare ModelError body (no promotion phrase, e.g. ``deepseek-*-free``
    401) retires via the same lazy path, but only when it is a 401 on a
    listed ``-free`` model — other ModelError shapes must not retire.
    """
    if not isinstance(data, dict):
        data = {}
    err = data.get("error")
    if not isinstance(err, dict):
        err = {}
    top_type = str(data.get("type") or "").strip().lower()
    err_type = str(err.get("type") or "").strip().lower()
    text = " ".join(
        str(v) for v in (data.get("message"), err.get("message"), body) if v is not None
    ).lower()
    if "promotion has ended" in text:
        return True
    if top_type != "modelerror" and err_type != "modelerror":
        return False
    try:
        code = int(status_code) if status_code is not None else None
    except Exception:
        code = None
    # In-body stream errors arrive without an HTTP status (stream already
    # 200): require the explicit phrase there — a bare ModelError mid-stream
    # (e.g. transient "overloaded") must NOT retire. A real HTTP status must
    # be 401 to qualify as the keyless-tier retire case.
    if code is None:
        return False
    if code != 401:
        return False
    m = _normalize_model(model) if model else ""
    return bool(m) and (bool(_FREE_MODEL_RE.match(m)) or m in _FREE_MODEL_EXTRA or m in _models_cache or m in _dead_models or m in DEAD_IDS)


def _is_free_tier_gate_error(data: dict | None = None, body: str = "") -> bool:
    """403 FreeTierError: exit rejected from Zen free tier ('can only be used
    from within OpenCode'). Exit-dependent — other exits keep working, so
    penalize the exit and rotate instead of surfacing terminal 403."""
    try:
        if body and ("FreeTierError" in body or "can only be used from within OpenCode" in body):
            return True
        if data:
            err = data.get("error") or {}
            msg = f"{err.get('type') or ''} {err.get('message') or ''}" if isinstance(err, dict) else str(err)
            return "FreeTierError" in msg or "can only be used from within OpenCode" in msg
    except Exception:
        return False
    return False


def _report_free_tier_block(proxy_addr: str | None):
    """Rotate off a FreeTier-gated exit with zero health penalty.

    The gate is request-shaped (same request 403s on every rotated exit while
    other requests OK on those same exits) — see
    ``proxy_pool.report_free_tier_block``. SYNC (owner: event-loop thread /
    request path). Falls back to a bare sticky reset if the pool hook is
    unavailable.
    """
    if not PROXY_POOL_ENABLED or not proxy_addr:
        return
    try:
        report = getattr(proxy_pool, "report_free_tier_block", None)
        if report is None:
            if proxy_pool.current and proxy_pool.current.get("address") == proxy_addr:
                proxy_pool.current = None
            return
        report(proxy_addr)
    except Exception:
        pass


# Fail fast on a request-shaped gate: the same request id 403s on every
# rotated exit (neither ``6471b37d`` nor ``592f7836`` ever succeeded), so
# sweeping all 6 attempts just burns latency + healthy exits. Surface after
# this many CONSECUTIVE gates; any non-gate outcome resets the count.
_FREE_TIER_FAIL_FAST_GATES = 3


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
# Free-tier ids whose name does not carry a "-free" suffix (models.dev cost == 0),
# kept alongside name-tagged models so discovery doesn't drop them.
_FREE_MODEL_EXTRA: set[str] = {"big-pickle"}
# Upstream lists these ids but they always fail: deepseek-v4-flash-free
# returns "Model is unavailable", north-mini-code-free 400s on multi-turn tool
# calls, ling-3.0-flash-free left the free tier (404, paid slug instead).
# Filtered from discovery AND from every listing/request path so clients never
# see them. Volatile upstream state lives here (not DEAD_IDS): a restart
# re-discovers when upstream heals, and session retire still applies below.
_MODELS_BLOCKED: set[str] = {"deepseek-v4-flash-free", "north-mini-code-free", "ling-3.0-flash-free"}
_MODELS_BLOCKED_LOWER: frozenset[str] = frozenset(m.lower() for m in _MODELS_BLOCKED)
# Precise free filter (port of pi-opencode-free FREE_REGEX): optional
# `opencode/` prefix + mandatory `-free` suffix, case-insensitive.
_FREE_MODEL_RE = re.compile(r"^(opencode/)?.*-free$", re.IGNORECASE)
_MODELS_REFRESH_SECS = 43200  # safety-net refresh every 12h; unknown models trigger on-demand
_DEFAULT_LIMIT = {"context": 128000, "output": 16384, "contextWindow": 128000, "maxTokens": 16384}
# Conservative image-gating default: unknown models advertise text-only input.
# Metadata-present path gates on models.dev modalities explicitly; only the
# unknown-model fallback uses this.
_DEFAULT_MODALITIES = {"input": ["text"], "output": ["text"]}
# Parallel discovery deadline (~3s combined for Zen + models.dev).
_DISCOVERY_DEADLINE_SECS = 3.5
# Timestamp (epoch secs) of the last successful discovery snapshot; None until
# the first success. Never cleared on empty/failed refreshes (staleness signal).
_models_checked_at: float | None = None


def _is_blocked_model(model: str | None) -> bool:
    """True if the id is blocklisted (listed upstream but always fails)."""
    return bool(model) and str(model).lower() in _MODELS_BLOCKED_LOWER


def _served_models() -> list[str]:
    """Discovered free ids minus the always-failing blocklist."""
    return [m for m in _models_cache if not _is_blocked_model(m)]


# ── Persistent dead-model denylist (pi-freeflow DEAD_MODEL_IDS pattern) ──
# File-backed so a retired -free id stays retired across restarts: stale
# disk state or a resurrected upstream listing cannot bring it back.
# Pattern only — no upstream id values are copied here.
_DEAD_IDS_FILE = _BASE_DIR / "data" / "dead-models.json"
DEAD_IDS: set[str] = set()  # persistent denylist; kept in sync with _dead_models


def _sanitize_models_cache() -> int:
    """Purge denylisted ids from the served snapshot. Returns purged count."""
    global _models_cache, _models_meta
    dead = _dead_models | DEAD_IDS
    if not dead:
        return 0
    purged = 0
    if _models_cache:
        kept = [m for m in _models_cache if m not in dead]
        purged = len(_models_cache) - len(kept)
        if purged:
            _models_cache = kept
    for m in list(_models_meta):
        if m in dead:
            _models_meta.pop(m, None)
    return purged


def _load_dead_ids() -> None:
    """Load the persistent denylist from disk (disk-cache read + sanitize)."""
    global DEAD_IDS
    try:
        with open(_DEAD_IDS_FILE, encoding="utf-8") as f:
            raw = json.load(f) or []
    except FileNotFoundError:
        raw = []
    except Exception:
        raw = []
    ids = {str(m).strip() for m in raw} if isinstance(raw, list) else set()
    ids = {m for m in ids if m}
    DEAD_IDS = ids
    _dead_models.update(ids)
    _sanitize_models_cache()


def _persist_dead_ids() -> None:
    """Persist the denylist (best-effort, never raises)."""
    try:
        _DEAD_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEAD_IDS_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(DEAD_IDS), f, indent=2, sort_keys=True)
    except OSError:
        pass


# ── Stealth/omen admission gate (stealth-models policy, pattern only) ──
# stealth/* and omen* ids enter the picker/aliases only with Zen free-list
# presence AND the free-regex (or explicit allowlist). Ghost client-side
# listings (e.g. Omen Alpha) fail the gate here instead of 404ing at
# request time. No new models are added by this gate.
_STEALTH_OMEN_RE = re.compile(r"^(opencode/)?(stealth[/\-_].*|omen.*)$", re.IGNORECASE)


def _is_stealth_or_omen(model_id: str | None) -> bool:
    try:
        return bool(model_id) and bool(_STEALTH_OMEN_RE.match(str(model_id).strip()))
    except Exception:
        return False


def _stealth_admitted(model_id: str, zen_free: set[str] | None = None) -> bool:
    """True unless a stealth/omen id fails the admission gate (logs rejection)."""
    if not _is_stealth_or_omen(model_id):
        return True
    mid = str(model_id).strip()
    if not (_FREE_MODEL_RE.match(mid) or mid in _FREE_MODEL_EXTRA):
        _log(f"[models] stealth gate: rejected {mid!r} (fails free filter, no allowlist entry)")
        return False
    if zen_free is not None and mid not in zen_free:
        _log(f"[models] stealth gate: rejected {mid!r} (absent from Zen free list)")
        return False
    return True


_load_dead_ids()


async def _fetch_free_models():
    """Query Zen API + models.dev, merge free model list with context limits.

    Both sources are fetched in parallel behind a combined ~3s deadline.
    On empty discovery the previous snapshot is kept (never wipe good cache).
    """
    global _models_cache, _models_meta, _models_checked_at
    try:
        async with httpx.AsyncClient() as c:
            zen_headers = {"User-Agent": f"opencode/{OC_VERSION}", "x-opencode-client": "cli"}

            async def _get_zen():
                return await c.get(
                    "https://opencode.ai/zen/v1/models",
                    headers=zen_headers,
                    timeout=10,
                )

            async def _get_models_dev():
                return await c.get("https://models.dev/api.json", timeout=10)

            try:
                zen_res, md_res = await asyncio.wait_for(
                    asyncio.gather(_get_zen(), _get_models_dev(), return_exceptions=True),
                    timeout=_DISCOVERY_DEADLINE_SECS,
                )
            except (asyncio.TimeoutError, TimeoutError):
                _log("[models] Discovery deadline exceeded, keeping cached models")
                return
            r = zen_res
            if isinstance(r, BaseException) or r is None:
                _log(f"[models] Zen API fetch failed ({r!r} if error), keeping cached models")
                return
            if getattr(r, "status_code", None) != 200:
                _log(f"[models] Zen API returned {getattr(r, 'status_code', '?')}, keeping cached models")
                return
            data = r.json()
            all_models = [m["id"] for m in data.get("data", []) if isinstance(m, dict)]
            zen_free = {m for m in all_models if (_FREE_MODEL_RE.match(m) or m in _FREE_MODEL_EXTRA)}
            free = [
                m for m in all_models
                if (_FREE_MODEL_RE.match(m) or m in _FREE_MODEL_EXTRA)
                and m not in _dead_models
                and m not in DEAD_IDS
                and not _is_blocked_model(m)
                and _stealth_admitted(m, zen_free)
            ]
            if not free:
                _log("[models] No free models found in Zen API, keeping cached")
                return

            # 2. Merge context limits from models.dev (parallel fetch above).
            meta: dict[str, dict] = {}
            try:
                md = md_res
                if isinstance(md, BaseException) or md is None:
                    raise RuntimeError(f"models.dev fetch failed: {md!r}")
                if md.status_code == 200:
                    md_data = md.json()
                    oc_models = md_data.get("opencode", {}).get("models", {})
                    for mid in free:
                        # Dual-key models.dev lookup: exact id, then bare
                        # (strip `opencode/` prefix), then base (also strip
                        # `-free` suffix) — models.dev keys bare ids.
                        entry = oc_models.get(mid) or oc_models.get(_bareModelId(mid)) or oc_models.get(baseModelId(mid))
                        if entry:
                            limit = entry.get("limit") or {}
                            if not isinstance(limit, dict):
                                limit = {}
                            # Conservative defaults when fields are missing.
                            ctx = limit.get("context") or limit.get("contextWindow") or _DEFAULT_LIMIT["context"]
                            out = limit.get("output") or limit.get("maxTokens") or _DEFAULT_LIMIT["maxTokens"]
                            mods = entry.get("modalities")
                            if not isinstance(mods, dict):
                                mods = {}
                            # Normalize modalities to text/image for /v1/models.
                            in_mods = [m for m in (mods.get("input") or []) if m in ("text", "image")] or ["text"]
                            out_mods = [m for m in (mods.get("output") or []) if m in ("text", "image")] or ["text"]
                            # Data-driven wire routing: models.dev provider.npm ==
                            # "@ai-sdk/openai" speaks the Responses wire protocol.
                            _prov = entry.get("provider") or {}
                            _npm = _prov.get("npm") if isinstance(_prov, dict) else None
                            meta[mid] = {
                                "name": entry.get("name") or _humanize_name(mid),
                                "limit": {"context": ctx, "output": out, "contextWindow": ctx, "maxTokens": out},
                                "modalities": {"input": in_mods, "output": out_mods},
                                "api": "openai-responses" if _npm == "@ai-sdk/openai" else "openai-completions",
                            }
                        else:
                            _prev_api = None
                            try:
                                _prev_api = (_models_meta.get(mid) or {}).get("api")
                            except Exception:
                                _prev_api = None
                            meta[mid] = {
                                "name": _humanize_name(mid),
                                "limit": dict(_DEFAULT_LIMIT),
                                "modalities": dict(_DEFAULT_MODALITIES),
                                "api": _prev_api if _prev_api in ("openai-responses", "openai-completions") else None,
                            }
                    _models_meta = meta
                    _log(f"[models] Loaded metadata for {len(meta)} models from models.dev")
                else:
                    raise RuntimeError(f"models.dev returned {md.status_code}")
            except Exception as e:
                _log(f"[models] models.dev fetch failed: {e}, using conservative defaults")
                _models_meta = {mid: {"name": _humanize_name(mid), "limit": dict(_DEFAULT_LIMIT), "modalities": dict(_DEFAULT_MODALITIES), "api": ((_models_meta.get(mid) or {}).get("api") if isinstance(_models_meta.get(mid), dict) else None)} for mid in free}

            _models_cache = free
            _models_checked_at = time.time()
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
    if _is_blocked_model(model) or model in _dead_models or model in DEAD_IDS:
        return False
    _sanitize_models_cache()
    if model in _models_cache:
        if _is_blocked_model(model):
            return False
        if not _stealth_admitted(model, set(_models_cache)):
            return False
        return True
    if _is_stealth_or_omen(model) and not (
        _FREE_MODEL_RE.match(model) or model in _FREE_MODEL_EXTRA
    ):
        _log(f"[models] stealth gate: rejected {model!r} (fails free filter, no allowlist entry)")
        return False
    await _fetch_free_models()
    return model in _models_cache and not _is_blocked_model(model)


def _has_promotion_phrase(data: dict | None = None, body: str = "") -> bool:
    """True when the error evidence carries the explicit promotion-ended phrase."""
    try:
        if not isinstance(data, dict):
            data = {}
        err = data.get("error")
        if not isinstance(err, dict):
            err = {}
        text = " ".join(
            str(v) for v in (data.get("message"), err.get("message"), body) if v is not None
        ).lower()
        return "promotion has ended" in text
    except Exception:
        return False


def _mark_model_dead(model: str | None, persistent: bool = True) -> bool:
    """Drop a model the upstream refuses with a promotion-ended ModelError.

    Removes it from the served list so clients get an immediate, clear
    "Unknown model" instead of per-request 401s, and keeps it out of future
    discovery passes. Returns True if this call retired it.

    persistent=True (explicit promotion phrase) also adds to file-backed
    DEAD_IDS; persistent=False (bare 401 ModelError, no phrase) is
    session-only so a transient 401 cannot brick the model across restarts.
    """
    global _models_cache, _models_meta
    if not model or model in _dead_models or model in DEAD_IDS:
        if model and model in DEAD_IDS and model not in _dead_models:
            _dead_models.add(model)
        return False
    _dead_models.add(model)
    if persistent:
        DEAD_IDS.add(model)
        _persist_dead_ids()
    if model in _models_cache:
        _models_cache = [m for m in _models_cache if m != model]
        _models_meta.pop(model, None)
    _log(f"[models] Retired {model} (upstream: free promotion ended){'' if persistent else ' [session-only]'}")
    return True


# ── Slash-free aliasing for OpenCode Zen models (no Kilo) ─────────
# Keys are slash-free / colon-free picker aliases; values are the bare
# canonical Zen ids actually served via Zen (no provider prefix, no :effort
# suffix) so the picker never sees slash/colon breakage.
MODEL_ALIASES: dict[str, str] = {
    "muse-spark": "muse-spark-1.3-contributor-free",
    "muse-spark-1.2": "muse-spark-1.2-contributor-free",
    "muse-spark-1.3": "muse-spark-1.3-contributor-free",
    "big-pickle": "big-pickle",
    "mimo": "mimo-v2.5-free",
    "mimo-v2.5": "mimo-v2.5-free",
    "nemotron-lightning": "nemotron-3.5-lightning-free",
    "nemotron-3.5-lightning": "nemotron-3.5-lightning-free",
    "nemotron-ultra": "nemotron-3-ultra-free",
    "nemotron-3-ultra": "nemotron-3-ultra-free",
    "laguna": "laguna-s-2.1-free",
    "laguna-s-2.1": "laguna-s-2.1-free",
    "longcat": "longcat-2.0-free",
    "longcat-2.0": "longcat-2.0-free",
    "hy3": "hy3-free",
    "ling-flash": "ling-3.0-flash-fin-free",
}


def resolveCanonicalModelId(alias: str | None) -> str | None:
    """Resolve a slash-free picker alias to its bare canonical Zen id.

    Handles `provider/prefix` stripping and `:effort` suffix stripping, so
    inputs like `opencode-local/muse-spark:high` resolve without slash/colon
    breakage. Unknown ids pass through as the bare normalized id.
    """
    if not alias:
        return alias
    s = str(alias)
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    if ":" in s:
        s = s.split(":", 1)[0]
    if s.startswith("ocf-"):
        s = s[4:]
    s = s.strip()
    if not s:
        return s
    hit = MODEL_ALIASES.get(s.lower())
    candidate = hit if hit else s
    # Stealth/omen admission: ghost client-side listings must not enter via
    # alias. A stealth/omen target requires the free-regex (or explicit
    # allowlist); Zen free-list presence is enforced downstream in
    # _ensure_model_known / _fetch_free_models. Rejected aliases fall back to
    # the raw id so the request fails closed as "Unknown model" instead of
    # 404ing upstream.
    if _is_stealth_or_omen(candidate) and not (
        _FREE_MODEL_RE.match(candidate) or candidate in _FREE_MODEL_EXTRA
    ):
        _log(f"[models] stealth gate: rejected alias {s!r} -> {candidate!r} (fails free filter, no allowlist entry)")
        return s
    if _is_stealth_or_omen(candidate) and _models_cache and candidate not in _models_cache:
        _log(f"[models] stealth gate: rejected alias {s!r} -> {candidate!r} (absent from Zen free list)")
        return s
    return candidate


def baseModelId(mid: str | None) -> str | None:
    """Strip provider prefix and `-free` suffix for models.dev fallback lookup.

    models.dev keys models under bare ids (e.g. `mimo-v2.5`) while Zen serves
    `opencode/`-prefixed `-free` ids, so exact-match alone misses metadata.
    """
    if not mid:
        return mid
    s = str(mid).strip()
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    if s.lower().endswith("-free"):
        s = s[: -len("-free")]
    return s


def _bareModelId(mid: str | None) -> str | None:
    """Strip only the provider prefix (keep `-free` suffix)."""
    if not mid:
        return mid
    s = str(mid).strip()
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    return s


def _humanize_name(mid: str | None) -> str:
    """Human-readable display name fallback: strip `-free`, Title-Case + " (Free)"."""
    try:
        s = str(mid or "").strip()
        if "/" in s:
            s = s.rsplit("/", 1)[-1]
        if s.lower().endswith("-free"):
            s = s[: -len("-free")]
        s = re.sub(r"[_-]+", " ", s).strip()
        words = [w[:1].upper() + w[1:] if w else w for w in s.split(" ") if w]
        # Collapse any double spaces from consecutive separators.
        label = " ".join(words)
        return f"{label} (Free)" if label else "Unknown (Free)"
    except Exception:
        return "Unknown (Free)"


def _normalize_model(model: str) -> str:
    """Normalize a client model id to the bare Zen upstream id.

    opencode sends ids like `opencode-local/muse-spark-1.2-contributor-free:high`:
    `opencode-local/` is the provider prefix and `:high` is the reasoning-effort
    suffix. Both are stripped, as is the legacy `ocf-` prefix. Slash-free
    picker aliases resolve via resolveCanonicalModelId().
    """
    if not model:
        return model
    return resolveCanonicalModelId(model)


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
            if part.get("tool_calls"):
                combined.setdefault("tool_calls", []).extend(part["tool_calls"])
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
            if isinstance(cached, dict):
                cached = cached.get("text")
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

    new_id = oc_id("ses", descending=True)
    full_h = _hash_messages(messages)
    _remember_session(sessions, full_h, new_id)
    return new_id


async def _backoff(attempt: int, base: float = 1.0, max_delay: float = 10.0):
    """Exponential backoff with jitter."""
    delay = min(base * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.25)
    await asyncio.sleep(delay + jitter)


# ── pi-freeflow: exhaust-then-direct fallback helpers ─────────────
# Retriable upstream statuses: try next proxy/key; 504 fast-breaks pool
# cycling (no backoff sleep, immediate rotate). After the pool is exhausted,
# one direct-fetch attempt goes out with no proxy and relay headers stripped
# — gated behind --allow-direct-fallback / OPENCODE_ALLOW_DIRECT_FALLBACK
# (default OFF: a direct fetch exposes this server's own IP upstream).
_RETRIABLE_STATUSES = frozenset({408, 429, 502, 503, 504, *range(520, 531)})


def _is_retriable_status(code) -> bool:
    try:
        return int(code) in _RETRIABLE_STATUSES
    except Exception:
        return False


# ── 413 = size, not health (pi-freeflow relay.ts:158-181) ──────────
# A 413 Payload Too Large means the request body exceeded an upstream limit —
# the proxy exit did nothing wrong. It is deliberately NOT in
# _RETRIABLE_STATUSES: every retry loop has a dedicated 413 branch that
# rotates to the next proxy (then direct-if-enabled) with ZERO pool-health
# accounting — no cooling_until, no blacklist, no consecutive_failures++, no
# report_failure / report_success / report_http_status.
def _is_too_large_status(code) -> bool:
    try:
        return int(code) == 413
    except Exception:
        return False


def _rotate_on_too_large(addr: str | None) -> bool:
    """Rotate off a 413 exit with zero health penalty. Returns True (retry).

    Delegates to proxy_pool.report_too_large (which only clears the sticky
    ``current`` pointer — no cooldown/blacklist/counters touched). Releases
    nothing: each retry loop owns its concurrency slot (``finally`` in the
    buffered loop, manual ``release()`` in the streaming loops). Callers must
    NOT record anything else for a 413.
    """
    if not PROXY_POOL_ENABLED or not addr:
        return True
    try:
        proxy_pool.report_too_large(addr)
    except Exception:
        # Fall back to a bare sticky reset if the pool helper is unavailable.
        try:
            if proxy_pool.current and proxy_pool.current.get("address") == addr:
                proxy_pool.current = None
        except Exception:
            pass
    return True


# ── Watchdog matrix (pi-freeflow client.ts:shouldRecoverOnHealth) ───
# Pure gate: True = force a pool refresh / key rotation (recover), False =
# hold the current selection (prevent flapping). Matrix:
#   None / gone            → True
#   version mismatch       → True
#   sseDegraded True       → True
#   sseDegraded undefined  → False (hold)
def should_rotate_on_health(health) -> bool:
    """Pure predicate: should an empty selection force recovery (True)?"""
    if health is None:
        return True
    if not isinstance(health, dict):
        return True
    if health.get("gone"):
        return True
    if health.get("versionMismatch") is True:
        return True
    ver = health.get("version")
    exp = health.get("expectedVersion", health.get("expected_version"))
    if ver is not None and exp is not None and str(ver) != str(exp):
        return True
    if health.get("sseDegraded") is True:
        return True
    healthy = health.get("healthy")
    cooling = health.get("cooling")
    # Pool snapshot shape (proxy_pool.health_snapshot): zero healthy entries
    # means the pool is gone/degraded → recover. (bool is an int subclass,
    # so exclude it explicitly — flags are handled above.)
    if isinstance(healthy, int) and not isinstance(healthy, bool):
        if isinstance(cooling, int) and not isinstance(cooling, bool):
            return healthy <= 0
        return healthy <= 0
    return False


def _pool_health_for_gate() -> dict | None:
    """Best-effort pool snapshot for the watchdog gate (no network).

    Returns None when the snapshot itself fails — the gate treats None as
    "recover", preserving the old always-refresh behavior on error.
    """
    try:
        return proxy_pool.health_snapshot()
    except Exception:
        return None


# ── Probe hardening (pi-freeflow 038321c) ──────────────────────────
# "Don't declare dead on first miss": a liveness/health miss sleeps 200ms
# and re-polls once before the caller treats the target as gone.
_PROBE_REPOLL_SECS = 0.2


async def _select_with_repoll():
    """proxy_pool.select() with probe hardening (no network beyond select).

    An empty selection races pool refill — sleep 200ms and re-poll once
    before the caller declares the pool empty and forces a refresh.
    """
    p = await proxy_pool.select()
    if p is not None:
        return p
    await asyncio.sleep(_PROBE_REPOLL_SECS)
    return await proxy_pool.select()


async def _maybe_force_refresh(reason: str) -> bool:
    """Force a pool refresh only when the watchdog gate says rotate.

    Returns True when a refresh was forced. When the gate says hold
    (healthy current / no degradation signal), the refresh is skipped to
    prevent flapping — the caller falls through to its normal fallback.
    """
    if not should_rotate_on_health(_pool_health_for_gate()):
        _log(f"[pool] {reason}: watchdog holds current selection (no refresh, no flapping)")
        return False
    await proxy_pool.force_refresh()
    return True


_ABORT_TYPE_NAMES = frozenset({
    "AbortError", "CancelledError", "Cancel", "ClientDisconnect", "Disconnect",
    "ClientDisconnected", "ConnectionAborted", "GeneratorExit",
})
_ABORT_TEXT_MARKERS = (
    "aborterror", "client disconnect", "client disconnected", "client abort", "aborted",
    "generator exit", "event loop is closed",
    "connection reset", "broken pipe", "client closed",
)
# Substrings that prove the error came from the UPSTREAM side (never a client abort),
# checked before _ABORT_TEXT_MARKERS. Bare "disconnect" used to live in the marker
# list and misclassified httpx RemoteProtocolError("Server disconnected without
# sending a response") as "Client aborted", terminating the stream instead of
# rotating the proxy exit and retrying. Mirrors proxy_pool/nvidia_pool is_cancelled.
_UPSTREAM_NOT_ABORT_MARKERS = (
    "server disconnect",
    "without sending a response",
    "peer closed connection",
)


def _is_client_abort(exc: BaseException | None) -> bool:
    """True for client-side aborts: never mark the pool failed for these.

    Mirrors proxy_pool.is_cancelled / nvidia_pool.is_cancelled so all three
    pools agree on GeneratorExit/Disconnect/AbortError variants.
    """
    if exc is None:
        return False
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return True
    name = type(exc).__name__
    if name in _ABORT_TYPE_NAMES:
        return True
    text = f"{name}: {exc}".lower()
    if any(m in text for m in _UPSTREAM_NOT_ABORT_MARKERS):
        return False
    return any(m in text for m in _ABORT_TEXT_MARKERS)


def _clear_req_ctx():
    """Reset the req tag so background tasks never inherit a request's id.

    ContextVars propagate into tasks spawned from a request handler
    (asyncio.ensure_future / create_task copy the current context), so
    background work like pool refills must clear the tag first — otherwise
    log lines from unrelated background work get misattributed to whatever
    request happened to spawn them.
    """
    try:
        _req_ctx.set("")
    except Exception:
        pass


def _strip_direct_headers(headers: dict | None) -> dict:
    """Strip relay/proxy hop headers for the direct-fetch fallback attempt.

    Keyless applied-last-wins audit (Zen-only): Zen free tier rejects ANY
    Authorization Bearer (401), so this strip MUST run after any pool/hook
    header merge and MUST also drop Authorization (case-insensitive). No layer
    in server.py / proxy_pool.py re-adds Authorization afterwards —
    proxy_pool.get_client() only sets base_url/timeout/proxy, never headers,
    and zen_request() builds Zen headers fresh with no Authorization key.
    NVIDIA NIM path (_nvidia_headers) is the sole legit Bearer user and never
    flows through this strip.
    """
    if not isinstance(headers, dict):
        return {}
    out = {}
    for k, v in headers.items():
        if v is None:
            continue
        kl = str(k).lower()
        if kl == "authorization":
            continue
        if kl.startswith("x-relay-") or kl.startswith("x-proxy"):
            continue
        out[k] = v
    return out


# Throttled upstream-429 hint (once per 10 min): shared free-tier IP quota.
_UPSTREAM_429_HINT = "Shared free-tier IP quota — rotating proxy/exit before surfacing 429"
_UPSTREAM_429_HINT_INTERVAL = 600.0
_last_429_hint_ts = 0.0


def _log_429_hint():
    global _last_429_hint_ts
    now = time.time()
    if now - _last_429_hint_ts < _UPSTREAM_429_HINT_INTERVAL:
        return
    _last_429_hint_ts = now
    _log(f"[zen] {_UPSTREAM_429_HINT}")


async def _maybe_backoff(attempt: int, status_code=None):
    """Backoff except fast-break on 504: rotate pool immediately, no sleep."""
    if status_code is not None:
        try:
            if int(status_code) == 504:
                return
        except Exception:
            pass
    await _backoff(attempt)


def _ensure_req_id():
    """Guarantee a req=<8hex> tag exists for this request context."""
    try:
        if not _req_ctx.get():
            new_request_id()
    except Exception:
        pass
    return _req_tag()


def _safe_pool_failure(addr: str | None, exc: BaseException | None = None, hard: bool = False) -> bool:
    """Report a transport failure unless it is a client-side abort.

    Client AbortError/CancelledError means the caller went away — the proxy
    exit did nothing wrong, so the pool must not be marked failed.
    Returns True when the failure was actually recorded.
    """
    if not PROXY_POOL_ENABLED or not addr:
        return False
    if _is_client_abort(exc):
        _log(f"[pool] client abort ({type(exc).__name__ if exc is not None else '?'}); not marking {addr} failed")
        return False
    proxy_pool.report_failure(addr, hard=hard)
    return True


def _safe_pool_stream_failure(addr: str | None, exc: BaseException | None = None) -> bool:
    """Abort-guarded report_stream_failure. Returns True when recorded."""
    if not PROXY_POOL_ENABLED or not addr:
        return False
    if _is_client_abort(exc):
        _log(f"[pool] client abort ({type(exc).__name__ if exc is not None else '?'}); not marking {addr} failed")
        return False
    proxy_pool.report_stream_failure(addr)
    return True


def _report_upstream_status(addr: str | None, status: int | None) -> bool:
    """Record an upstream HTTP status for escalating 5xx/504 cooldown.

    Delegates to proxy_pool.report_http_status (escalating entry cooldown;
    504 fast-breaks instead of cycling the pool). 429 is excluded — it keeps
    its dedicated report_ratelimit sites. Returns True if the caller should
    roll to the next proxy, False on 504 fast-break (stop cycling).
    """
    if not PROXY_POOL_ENABLED or not addr or status is None:
        return True
    try:
        code = int(status)
    except Exception:
        return True
    if code == 429:
        return True  # dedicated report_ratelimit path owns 429
    try:
        return proxy_pool.report_http_status(addr, code)
    except Exception:
        return True


def _report_nvidia_status(key: str | None, status: int | None) -> bool:
    """Record an upstream HTTP status for escalating 5xx/504 key cooldown.

    Delegates to nvidia_keys.report_http_status (429 keeps its dedicated
    report_rate_limit sites). Returns True if the caller should roll to the
    next key, False on 504 fast-break.
    """
    if not key or status is None:
        return True
    try:
        code = int(status)
    except Exception:
        return True
    if code == 429:
        return True  # dedicated report_rate_limit path owns 429
    try:
        return nvidia_keys.report_http_status(key, code)
    except Exception:
        return True


def _report_amd_status(key: str | None, status: int | None) -> bool:
    """Record an upstream HTTP status for escalating 5xx/504 key cooldown.

    Delegates to amd_keys.report_http_status (429 keeps its dedicated
    report_rate_limit sites). Returns True if the caller should roll to the
    next key, False on 504 fast-break.
    """
    if not key or status is None:
        return True
    try:
        code = int(status)
    except Exception:
        return True
    if code == 429:
        return True  # dedicated report_rate_limit path owns 429
    try:
        return amd_keys.report_http_status(key, code)
    except Exception:
        return True


def _log_direct_fallback(model, status):
    """Loud log line whenever the gated post-exhaustion direct fetch fires."""
    _log(f"[zen] DIRECT FALLBACK [{model}]: pool exhausted on retriable {status}; one no-proxy fetch (server IP exposed)")


async def _direct_buffered_post(path: str, req_body: dict, headers: dict, client=None):
    """One direct-fetch attempt: no proxy, relay headers stripped.

    Used only after the proxy pool is exhausted on a retriable status, and
    only when ALLOW_DIRECT_FALLBACK is on. Returns (status_code, data,
    body_text) or (None, None, "") on transport failure.
    """
    client = client or _direct_client
    try:
        resp = await client.post(path, json=req_body, headers=_strip_direct_headers(headers))
    except Exception as e:
        return None, None, f"direct-fetch transport failure: {_exc_desc(e)}"
    try:
        raw = await resp.aread()
    except Exception as e:
        return resp.status_code, {}, f"direct-fetch read failure: {_exc_desc(e)}"
    text = raw.decode("utf-8", errors="replace") if raw else ""
    try:
        data = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
        data = {}
    return resp.status_code, data, text


async def _client_gone(request: Request) -> bool:
    """True if the requesting client has disconnected (so retries stop).

    Probe hardening: don't declare dead on first miss — a positive first
    poll sleeps 200ms and re-polls once; only a confirmed miss aborts.
    """
    try:
        if not await request.is_disconnected():
            return False
    except Exception:
        return False
    try:
        await asyncio.sleep(_PROBE_REPOLL_SECS)
        return bool(await request.is_disconnected())
    except Exception:
        return True


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
    # muse-spark streams through /zen/v1/responses (see _is_muse_spark) and does
    # not use the chat/completions fallback at all.
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


# prompt_cache_key passes through additively on the Zen path; retention is
# never forwarded (stripped in zen_request). Kept out of _SAMPLING_KEYS so the
# NVIDIA path is untouched.
_ZEN_SAMPLING_KEYS = _SAMPLING_KEYS + ("prompt_cache_key",)


def _sampling_from(body: dict) -> dict:
    """Pick client sampling params the upstream accepts; absent = upstream default."""
    return {k: body[k] for k in _ZEN_SAMPLING_KEYS if body.get(k) is not None}


_NVIDIA_SAMPLING_KEYS = _SAMPLING_KEYS + ("reasoning_effort",)


def _nvidia_sampling_from(body: dict) -> dict:
    """NVIDIA NIM also accepts reasoning_effort (low/medium/high) on reasoning models."""
    return {k: body[k] for k in _NVIDIA_SAMPLING_KEYS if body.get(k) is not None}


_AMD_SAMPLING_KEYS = _SAMPLING_KEYS + ("reasoning_effort",)


def _amd_sampling_from(body: dict) -> dict:
    """AMD TokenFactory mirrors the NVIDIA sampling surface (reasoning_effort)."""
    return {k: body[k] for k in _AMD_SAMPLING_KEYS if body.get(k) is not None}


def zen_request(model, messages, stream, tools, tool_choice, session_id, max_tokens=None, max_completion_tokens=None, sampling: dict | None = None):
    model = _normalize_model(model)
    req_body: dict = {"model": model, "messages": messages, "stream": bool(stream)}
    if tools:
        req_body["tools"] = tools
    if tool_choice:
        # Zen thinking mode rejects forced tool_choice (e.g. {"type":"function",
        # "function":{"name":"..."}} or "none"). Downgrade to "auto" so the
        # request passes thinking-mode validation; the model will still call
        # the right tool based on context.
        if tool_choice == "auto" or tool_choice == "none":
            req_body["tool_choice"] = tool_choice
        else:
            req_body["tool_choice"] = "auto"
    # max_tokens hygiene: prefer max_tokens, never synthesize
    # max_completion_tokens from it. Both forward verbatim when supplied.
    if max_tokens is not None:
        req_body["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        req_body["max_completion_tokens"] = max_completion_tokens
    if sampling:
        req_body.update(sampling)
    # prompt_cache_retention is never sent upstream; prompt_cache_key (if any)
    # rides along additively via sampling.
    req_body.pop("prompt_cache_retention", None)
    req_body.pop("store", None)  # never forward store:true to Zen

    request_id = oc_id("msg")
    # Keyless: Zen free tier rejects ANY Authorization Bearer (401), so omit
    # the header entirely. Keep x-opencode-client/project + UA.
    # Applied-last-wins: this fresh dict is built AFTER any caller header
    # merge, and no code path in server.py / proxy_pool.py re-adds
    # Authorization afterwards (pool clients set transport only; direct-
    # fallback paths only strip via _strip_direct_headers, never inject).
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"opencode/{OC_VERSION}",
        "x-opencode-client": "cli",
        "x-opencode-project": _ZEN_PROJECT_ID,
        "x-opencode-request": request_id,
        "x-opencode-session": session_id,
    }
    if any(str(k).lower() == "authorization" for k in headers):
        raise RuntimeError("keyless: Authorization leaked into zen_request headers")
    return req_body, headers


# ── Muse Spark via /zen/v1/responses ─────────────────────────────
# Muse Spark models (muse-spark-1.2/1.3-contributor-free, future versions) are
# NOT served by /zen/v1/chat/completions — that route returns a deterministic
# bare 500 "Internal server error", which previously presented as a flood of
# retries. They are only served by the Responses endpoint. Requests addressed
# at these models are translated to the Responses shape (input items,
# max_output_tokens, reasoning:{effort,summary}) and streamed from
# /zen/v1/responses, then translated back into chat.completion.chunk SSE so
# every downstream client keeps speaking plain chat-completions.
# (Same fix as 9router ab044e6/acb5c34.)

_MUSE_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def _is_muse_spark(model: str | None) -> bool:
    m = (model or "").lower()
    return "muse-spark" in m or "muse_spark" in m


def _model_wire_api(model: str | None) -> str:
    """Cached wire protocol for ``model`` with substring fallback.

    Returns the persisted ``_models_meta`` ``api`` ("openai-responses" vs
    "openai-completions") when present, else falls back to the existing
    ``_is_muse_spark()`` substring check — so muse-spark behavior is
    identical when meta is missing.
    """
    try:
        api = (_models_meta.get(str(model)) or {}).get("api")
        if api in ("openai-responses", "openai-completions"):
            return api
    except Exception:
        pass
    return "openai-responses" if _is_muse_spark(model) else "openai-completions"


def _muse_effort(effort) -> str | None:
    if not effort:
        return None
    e = str(effort).lower().strip()
    if e in ("max", "ultra"):
        return "xhigh"  # muse tops out at xhigh
    return e if e in _MUSE_EFFORTS else None


def _model_suffix_effort(model: str | None) -> str | None:
    """Extract the reasoning-effort suffix from a client model id (…:high)."""
    if not model:
        return None
    m = str(model)
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    if ":" not in m:
        return None
    return _muse_effort(m.split(":", 1)[1])


def _resolve_muse_effort(model_id: str | None, body: dict | None = None) -> str | None:
    """Resolve the muse reasoning effort with the Responses xhigh clamp.

    Precedence: ``:suffix`` on the model id > ``reasoning.effort`` >
    ``reasoning_effort``. Without this, a Chat-style ``reasoning_effort``
    (e.g. "max") falls through the ``if eff:`` gate as None and the request
    goes out with no ``reasoning`` block at all (same normalization 9router
    does at its executor boundary).
    """
    eff = _model_suffix_effort(model_id)
    if eff:
        return eff
    if isinstance(body, dict):
        r = body.get("reasoning")
        if isinstance(r, dict):
            eff = _muse_effort(r.get("effort"))
            if eff:
                return eff
        eff = _muse_effort(body.get("reasoning_effort"))
        if eff:
            return eff
    return None


# Floor for tool-free Responses requests: below this the model can burn the
# whole budget on reasoning and emit zero events (deterministic empty-EOF).
_RESPONSES_MIN_OUTPUT_TOKENS = 512


_RESPONSES_SAMPLING_KEYS = ("temperature", "top_p")


def _responses_sampling_from(body: dict) -> dict:
    """Sampling params the Responses API accepts.

    ``stop`` is a Chat Completions field: blindly merging it into the
    Responses body risks a 400 or silent ignore, so it stays Chat-only.
    ``prompt_cache_key`` passes through additively; retention is stripped
    in _zen_responses_body.
    """
    return {k: body[k] for k in _RESPONSES_SAMPLING_KEYS + ("prompt_cache_key",) if body.get(k) is not None}


def _responses_content_parts(content, assistant: bool = False) -> list:
    """Chat content → Responses content parts (images supported for muse)."""
    text_type = "output_text" if assistant else "input_text"
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": text_type, "text": content}] if content else []
    parts = []
    for b in content if isinstance(content, list) else [content]:
        if not isinstance(b, dict):
            parts.append({"type": text_type, "text": str(b)})
            continue
        bt = (b.get("type") or "").lower()
        if bt == "text":
            t = b.get("text") or ""
            if t:
                parts.append({"type": text_type, "text": t})
        elif bt == "image_url":
            iu = b.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else iu
            if url:
                parts.append({"type": "input_image", "image_url": url})
        # reasoning/thinking blocks are dropped: the upstream derives its own
        # thinking from the conversation and does not accept prior reasoning.
    return parts


def _chat_to_responses_input(messages: list[dict]) -> list:
    """Convert chat.completions messages to Responses input items.

    Clients (opencode) reuse tool-call ids across turns (e.g. ``bash:0`` for
    every bash call in a session). The Responses API rejects a second
    ``function_call_output`` for the same ``call_id``, so repeats are
    renumbered in call order (``bash:0`` → ``bash:0__fc1`` …) and each output
    pairs with the oldest unmatched call of its id. Pairing state resets at
    every assistant message so interrupted turns can't mis-pair later ones.
    """
    items = []
    counters: dict[str, int] = {}  # orig call_id -> emitted function_call count
    pending: dict[str, list] = {}  # orig call_id -> queue of new ids awaiting an output
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("system", "user"):
            parts = _responses_content_parts(m.get("content"))
            if parts:
                items.append({"role": role, "content": parts})
        elif role == "assistant":
            pending = {}  # turn boundary: an unmatched call must not pair a later turn's output
            parts = _responses_content_parts(m.get("content"), assistant=True)
            if parts:
                items.append({"role": "assistant", "content": parts})
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args = fn.get("arguments") or ""
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                orig = tc.get("id") or "call"
                n = counters.get(orig, 0)
                counters[orig] = n + 1
                new_id = orig if n == 0 else f"{orig}__fc{n}"
                pending.setdefault(orig, []).append(new_id)
                items.append({
                    "type": "function_call",
                    "call_id": new_id,
                    "name": fn.get("name") or "",
                    "arguments": args,
                })
        elif role == "tool":
            out = m.get("content")
            if not isinstance(out, str):
                out = json.dumps(out, ensure_ascii=False) if out is not None else ""
            orig = m.get("tool_call_id") or "call"
            q = pending.get(orig)
            if q:
                call_id = q.pop(0)
            elif orig in counters:
                # The real pair was already emitted in-window; this is a
                # re-echoed duplicate — drop it rather than minting a second
                # output that would 400 or dangle unattached.
                continue
            else:
                # No call with this id anywhere in the window (history was
                # truncated mid-turn): keep the result under a unique id
                # rather than deleting evidence or duplicating an id.
                call_id = f"{orig}__orphan0"
            items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": out,
            })
    return items


def _responses_continuation_body(resp_body: dict, content: str) -> dict:
    """Clone a Responses body with the partial assistant output appended.

    A mid-stream rate-limit or torn tunnel cuts muse answers off mid-work;
    re-issuing with the partial output as an assistant item lets the model
    continue from the cutoff instead of discarding the whole turn.
    """
    body = dict(resp_body)
    items = list(body.get("input") or [])
    if content:
        items.append({
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        })
    body["input"] = items
    return body


def _zen_responses_body(model, messages, tools, effort=None, max_tokens=None,
                        max_completion_tokens=None, sampling: dict | None = None) -> dict:
    """Build the /zen/v1/responses request body from chat-style parameters."""
    body: dict = {
        "model": model,
        "input": _chat_to_responses_input(messages),
        "stream": True,
    }
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "name": (t.get("function") or {}).get("name") or "",
                "description": (t.get("function") or {}).get("description") or "",
                "parameters": (t.get("function") or {}).get("parameters") or {},
            }
            for t in tools
            if isinstance(t, dict)
        ]
        body["tool_choice"] = "auto"
    # max_tokens hygiene: prefer max_tokens (never rename it); fall back to
    # max_completion_tokens only when max_tokens is absent.
    mt = max_tokens if max_tokens is not None else max_completion_tokens
    # Muse's reasoning tokens count against the SAME budget as output
    # (usage showed completion_tokens=128 with zero text at max_out=128),
    # so a tool-free budget below the floor is raised to leave room for
    # thinking AND text. Unset stays unset (upstream default, unlimited).
    if mt is not None and not tools and mt < _RESPONSES_MIN_OUTPUT_TOKENS:
        mt = _RESPONSES_MIN_OUTPUT_TOKENS
    if mt is not None:
        body["max_output_tokens"] = mt
    eff = _muse_effort(effort)
    # No default effort cap: unset effort passes through with no reasoning
    # block (upstream default, full quality). Only explicit high/xhigh/max/
    # ultra on tool-free muse requests is downgraded to low — those levels
    # deterministically burn a small output budget thinking.
    if not tools and _is_muse_spark(model) and eff in ("high", "xhigh", "max", "ultra"):
        eff = "low"
        _log(f"[zen] [{model}|scan] downgrading reasoning {effort} to low (tool-free muse request)")
    if eff:
        body["reasoning"] = {"effort": eff, "summary": "auto"}
    if sampling:
        body.update(sampling)
    body.pop("prompt_cache_retention", None)
    body.pop("store", None)  # never forward store:true to Zen
    return body


def _responses_diag(resp_body: dict) -> str:
    """One-line digest of the Responses request body for empty-EOF correlation.

    Direct-mode runs prove the zero-event EOF comes from upstream, not the
    proxy — so the next question is *which* requests come back empty.
    """
    try:
        inp = resp_body.get("input") or []
        chars = sum(len(json.dumps(i, ensure_ascii=False)) for i in inp)
        tools = resp_body.get("tools") or []
        return (
            f"reasoning={resp_body.get('reasoning')!r} "
            f"tools={len(tools)} input_items={len(inp)} "
            f"input_chars~{chars} max_out={resp_body.get('max_output_tokens')!r}"
        )
    except Exception:
        return "?"


def _responses_usage_to_chat(usage: dict | None) -> dict | None:
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("input_tokens") or 0
    completion = usage.get("output_tokens") or 0
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": usage.get("total_tokens") or (prompt + completion),
        "prompt_cache_hit_tokens": cached,
    }


async def _zen_responses_stream_with_retry(
    request: Request,
    resp_body: dict,
    headers: dict,
    user: str,
    messages: list[dict],
    session_id: str = None,
    model: str = None,
    max_retries: int = None,
):
    """Stream /zen/v1/responses as OpenAI chat.completion.chunk SSE with proxy retry.

    Response events are folded back into chat chunks: output_text deltas →
    content, reasoning deltas → reasoning_content, function_call items →
    tool_calls, response.completed → finish chunk + usage + [DONE].
    """
    _ensure_req_id()
    last_error = None
    attempts = MAX_RETRIES if max_retries is None else max_retries
    direct_only = False  # set after pool exhaustion: next loop is the one direct-fetch attempt
    cid = oc_id("chatcmpl")
    created = int(time.time())
    continuations = 0  # mid-stream resumes (partial content appended as context)

    def _chunk(delta: dict, finish: str | None = None) -> str:
        return f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model or '', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n"

    tool_idx: dict[str, int] = {}
    tool_args: dict[int, str] = {}
    tool_count_holder = [0]

    def _fc_chunks(item: dict) -> list:
        """Translate a full Responses function_call item into chat tool_call chunk(s).

        Not every upstream announce carries the add→delta sequence: some
        servers only emit the finished item. Without this fallback the call
        was silently dropped — the classic "announces the action, executes
        nothing, stops" turn. Only the not-yet-emitted suffix of arguments
        is sent, so a done event repeating deltas never duplicates them.
        """
        nonlocal tool_streamed, role_sent
        chunks: list = []
        if not role_sent:
            role_sent = True
            chunks.append(_chunk({"role": "assistant", "content": ""}))
        iid = item.get("id") or item.get("call_id") or ""
        idx = tool_idx.get(iid)
        head = None
        if idx is None:
            idx = tool_count_holder[0]
            tool_count_holder[0] += 1
            tool_idx[iid] = idx
            tool_args[idx] = ""
            head = {
                "index": idx,
                "id": item.get("call_id") or item.get("id") or oc_id("call"),
                "type": "function",
                "function": {"name": item.get("name") or "", "arguments": ""},
            }
        full = item.get("arguments") or ""
        emitted = tool_args.get(idx, "")
        suffix = full[len(emitted):] if isinstance(full, str) else ""
        if head is not None:
            head["function"]["arguments"] = suffix
            tool_args[idx] = emitted + suffix
            chunks.append(_chunk({"tool_calls": [head]}))
            tool_streamed = True
        elif suffix:
            tool_args[idx] = emitted + suffix
            chunks.append(_chunk({"tool_calls": [{
                "index": idx,
                "function": {"arguments": suffix},
            }]}))
            tool_streamed = True
        return chunks

    attempt = 0
    free_tier_gates = 0  # consecutive FreeTier gates this request; fail fast at _FREE_TIER_FAIL_FAST_GATES
    while True:
        if attempt > attempts + MAX_STREAM_CONTINUATIONS:
            break
        if await _client_gone(request):
            _log("[zen] Client disconnected; aborting retries")
            return
        if PROXY_POOL_ENABLED and not direct_only:
            if not proxy_pool.ready:
                await proxy_pool.load()
            p = await _select_with_repoll()
            if p is None:
                _log(f"[pool] No proxy available ({proxy_pool.get_pool_state()}), forcing refresh")
                await _maybe_force_refresh("No proxy available")
                p = await proxy_pool.select()
                if p is None:
                    _log("[pool] Still no proxy after refresh, falling back to direct")
                    client = _stream_direct_client
                    proxy_addr = None
                else:
                    proxy_addr = p["address"]
                    client = proxy_pool.get_client(f"socks5://{proxy_addr}", streaming=True)
            else:
                proxy_addr = p["address"]
                client = proxy_pool.get_client(f"socks5://{proxy_addr}", streaming=True)
        else:
            if direct_only:
                headers = _strip_direct_headers(headers)
            client = _stream_direct_client if direct_only else _stream_default_client
            proxy_addr = None

        try:
            upstream_request = client.build_request(
                "POST", "/zen/v1/responses", json=resp_body, headers=headers
            )
            resp = await client.send(upstream_request, stream=True)
        except Exception as e:
            _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Responses request failed (attempt {attempt}): {_exc_desc(e)}")
            if _is_client_abort(e):
                if PROXY_POOL_ENABLED and proxy_addr:
                    proxy_pool.release(proxy_addr)
                yield _openai_stream_error(f"Client aborted: {_exc_desc(e)}", "upstream_error", "transport_error")
                return
            # Setup failure = the tunnel never established (SOCKS
            # handshake / connect / TLS died before any HTTP bytes).
            # Always hard: retrying the same exit just burns another
            # attempt on it. Note the allowlist approach misses
            # socksio's raw ProtocolError("Malformed reply") — it escapes
            # httpx unwrapped (no req= URL in the log), so it fell into
            # the soft 1/2 grace path and was retried on the dead exit.
            # (Abort already returned above, so _safe_pool_failure records.)
            _safe_pool_failure(proxy_addr, e, hard=True)
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.release(proxy_addr)
            last_error = e
            free_tier_gates = 0  # transport outcome breaks the gate streak
            if attempt < attempts:
                await _backoff(attempt)
                attempt += 1
                continue
            yield _openai_stream_error(f"Stream request failed: {_exc_desc(e)}", "upstream_error", "transport_error")
            return

        if resp.status_code == 429:
            try:
                raw = await resp.aread()
                data = json.loads(raw)
                err_msg = (data.get("error") or {}).get("message") or "Rate limit exceeded"
            except Exception:
                err_msg = "Rate limit exceeded"
            _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Responses 429 (attempt {attempt}): {err_msg}")
            _log_429_hint()
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_ratelimit(proxy_addr)
                proxy_pool.release(proxy_addr)
            await resp.aclose()
            free_tier_gates = 0  # 429 is exit-shaped quota, not the request gate
            if attempt < attempts:
                await _maybe_backoff(attempt, 429)
                attempt += 1
                continue
            if PROXY_POOL_ENABLED and not direct_only and ALLOW_DIRECT_FALLBACK:
                direct_only = True
                _log_direct_fallback(model, 429)
                headers = _strip_direct_headers(headers)
                await _maybe_backoff(attempt, 429)
                attempt += 1
                continue
            if PROXY_POOL_ENABLED and not ALLOW_DIRECT_FALLBACK:
                _log(f"[zen] pool exhausted on retriable 429; direct fallback disabled (opt in with --allow-direct-fallback)")
            yield _openai_stream_error(err_msg + " (free model rate limit)", "rate_limit_error", "rate_limit_exceeded")
            return

        if resp.status_code >= 400:
            try:
                raw = await resp.aread()
            except Exception:
                raw = b""
            body_text = raw.decode("utf-8", errors="replace")
            try:
                data = json.loads(body_text)
            except (json.JSONDecodeError, TypeError, ValueError):
                data = {}
            err_msg = (data.get("error") or {}).get("message") or f"HTTP {resp.status_code}"
            _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Responses error {resp.status_code}: {raw[:400]!r}")
            if _is_too_large_status(resp.status_code):
                # 413 = size, not health: rotate with ZERO health accounting,
                # then direct-if-enabled (keep last 413 body for salvage).
                _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Responses 413 too-large: {err_msg} — rotating (no penalty)")
                if PROXY_POOL_ENABLED and proxy_addr:
                    _rotate_on_too_large(proxy_addr)
                    proxy_pool.release(proxy_addr)
                await resp.aclose()
                if attempt < attempts and not await _client_gone(request):
                    await _maybe_backoff(attempt, resp.status_code)
                    attempt += 1
                    continue
                if PROXY_POOL_ENABLED and not direct_only and ALLOW_DIRECT_FALLBACK:
                    direct_only = True
                    _log_direct_fallback(model, resp.status_code)
                    headers = _strip_direct_headers(headers)
                    await _maybe_backoff(attempt, resp.status_code)
                    attempt += 1
                    continue
                if PROXY_POOL_ENABLED and not ALLOW_DIRECT_FALLBACK:
                    _log(f"[zen] pool exhausted on 413; direct fallback disabled (opt in with --allow-direct-fallback)")
                yield _openai_stream_error(err_msg, "upstream_error", "413")
            if resp.status_code == 403 and _is_free_tier_gate_error(data, body_text):
                free_tier_gates += 1
                _log(f"[zen] [{model}|{proxy_addr or 'direct'}] FreeTier gate 403 ({free_tier_gates}/{_FREE_TIER_FAIL_FAST_GATES}): {err_msg} — rotating (no penalty)")
                if PROXY_POOL_ENABLED and proxy_addr:
                    _report_free_tier_block(proxy_addr)
                    proxy_pool.release(proxy_addr)
                if free_tier_gates >= _FREE_TIER_FAIL_FAST_GATES:
                    _log(f"[zen] [{model}] FreeTier gate on {free_tier_gates} consecutive exits — request-shaped, failing fast")
                    await resp.aclose()
                    yield _openai_stream_error(err_msg, "upstream_error", "free_tier_gate")
                    return
                if attempt < attempts and not await _client_gone(request):
                    await resp.aclose()
                    await _maybe_backoff(attempt, resp.status_code)
                    attempt += 1
                    continue
                if PROXY_POOL_ENABLED and not direct_only and ALLOW_DIRECT_FALLBACK:
                    direct_only = True
                    _log_direct_fallback(model, resp.status_code)
                    headers = _strip_direct_headers(headers)
                    await resp.aclose()
                    await _maybe_backoff(attempt, resp.status_code)
                    attempt += 1
                    continue
                await resp.aclose()
                yield _openai_stream_error(err_msg, "upstream_error", "free_tier_gate")
                return
            if _is_retriable_status(resp.status_code) and PROXY_POOL_ENABLED and not direct_only and attempt >= attempts:
                # Escalating 5xx/504 cooldown for this exit; 504 fast-breaks
                # (False = stop cycling the pool, surface the error now).
                if not _report_upstream_status(proxy_addr, resp.status_code):
                    if PROXY_POOL_ENABLED and proxy_addr:
                        proxy_pool.release(proxy_addr)
                    await resp.aclose()
                    yield _openai_stream_error(err_msg, "upstream_error", str(resp.status_code))
                    return
                if not ALLOW_DIRECT_FALLBACK:
                    _log(f"[zen] pool exhausted on retriable {resp.status_code}; direct fallback disabled (opt in with --allow-direct-fallback)")
                    if PROXY_POOL_ENABLED and proxy_addr:
                        proxy_pool.release(proxy_addr)
                    await resp.aclose()
                    yield _openai_stream_error(err_msg, "upstream_error", str(resp.status_code))
                    return
                direct_only = True
                _log_direct_fallback(model, resp.status_code)
                headers = _strip_direct_headers(headers)
                if PROXY_POOL_ENABLED and proxy_addr:
                    proxy_pool.release(proxy_addr)
                await resp.aclose()
                await _maybe_backoff(attempt, resp.status_code)
                free_tier_gates = 0  # retriable outcome breaks the gate streak
                attempt += 1
                continue
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.release(proxy_addr)
            if _is_region_error(data, body_text):
                if PROXY_POOL_ENABLED and proxy_addr:
                    proxy_pool.report_region_block(proxy_addr)
                free_tier_gates = 0  # region outcome breaks the gate streak
                if attempt < attempts:
                    await _backoff(attempt)
                    attempt += 1
                    continue
                yield _openai_stream_error(err_msg, "upstream_error", "region_blocked")
                return
            # Entitlement error (401 ModelError incl. promotion-ended): retire
            # the model and terminate immediately, same lazy path as chat.
            if _is_unavailable_error(data, body_text) or _is_promotion_ended_error(data, body_text, resp.status_code, model):
                await resp.aclose()
                _mark_model_dead(model, persistent=_has_promotion_phrase(data, body_text))
                yield _openai_stream_error(f"{model} is no longer available: {err_msg}", "upstream_error", "model_retired")
                return
            await resp.aclose()
            if _is_retriable_status(resp.status_code) and attempt < attempts:
                # 504 "Upstream response was not valid JSON" is the provider
                # (Console) failing to parse the MODEL's raw output — the shape
                # of our request (reasoning level, budget) influences whether
                # the model emits parseable JSON. Log the request shape so the
                # next occurrence is correlatable, then retry as before.
                # 504 fast-breaks pool cycling (no backoff sleep).
                if resp.status_code == 504:
                    _log(f"[zen] [{model}|{proxy_addr or 'direct'}] 504 detail (attempt {attempt}); req={_responses_diag(resp_body)}")
                # Escalating 5xx/504 cooldown; 504 fast-break stops cycling.
                if not _report_upstream_status(proxy_addr, resp.status_code):
                    yield _openai_stream_error(err_msg, "upstream_error", str(resp.status_code))
                    return
                await _maybe_backoff(attempt, resp.status_code)
                attempt += 1
                continue
            yield _openai_stream_error(err_msg, "upstream_error", str(resp.status_code))
            return

        # Success — translate response.* events into chat chunks
        streamed_any = False
        stream_completed = False
        role_sent = False
        tool_streamed = False
        _reason_buf = ""
        _content_buf = ""
        finish_reason = None
        saw_error_event = False
        transport_reported = False
        empty_turn_retry = False
        tool_idx.clear()
        tool_args.clear()
        tool_count_holder[0] = 0
        tool_count = 0
        try:
            try:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if not line.startswith("data"):
                        continue
                    payload = line[4:].strip()
                    if payload.startswith(":"):
                        payload = payload[1:].strip()
                    if not payload or payload == "[DONE]":
                        if payload == "[DONE]":
                            stream_completed = True
                            yield "data: [DONE]\n\n"
                            break
                        continue
                    try:
                        piece = json.loads(payload)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    if not isinstance(piece, dict):
                        continue
                    ptype = piece.get("type") or ""

                    if ptype in ("error", "response.failed", "response.incomplete") and not piece.get("response"):
                        err = piece.get("error") or {}
                        err_msg = (err.get("message") if isinstance(err, dict) else None) or piece.get("message") or "Upstream error"
                        _log(f"[zen] Responses stream error (attempt {attempt}): {err_msg}")
                        last_error = ValueError(err_msg)
                        # Same lazy-retire path as chat on ModelError bodies.
                        if _is_unavailable_error(piece, json.dumps(piece) if isinstance(piece, dict) else "") or _is_promotion_ended_error(piece, json.dumps(piece) if isinstance(piece, dict) else "", None, model):
                            _mark_model_dead(model, persistent=_has_promotion_phrase(piece, json.dumps(piece) if isinstance(piece, dict) else ""))
                            yield _openai_stream_error(f"{model} is no longer available: {err_msg}", "upstream_error", "model_retired")
                            return
                        if not streamed_any and attempt < attempts and not await _client_gone(request):
                            saw_error_event = True
                            break  # retry below
                        if (
                            streamed_any and _content_buf and not tool_streamed
                            and continuations < MAX_STREAM_CONTINUATIONS
                            and not await _client_gone(request)
                        ):
                            # Mid-stream in-body error (e.g. provider rate limit):
                            # resume from the cutoff instead of truncating.
                            saw_error_event = True
                            continuations += 1
                            resp_body = _responses_continuation_body(resp_body, _content_buf)
                            _content_buf = ""
                            _reason_buf = ""
                            _log(f"[zen] Continuing responses stream after in-body error (continuation {continuations})")
                            break  # retry below (streams away from the rate-limited exit)
                        yield _openai_stream_error(err_msg)
                        return

                    if ptype == "response.output_text.delta":
                        d = piece.get("delta") or ""
                        if not role_sent:
                            role_sent = True
                            yield _chunk({"role": "assistant", "content": ""})
                        yield _chunk({"content": d})
                        _content_buf += d
                        streamed_any = True
                    elif ptype in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                        d = piece.get("delta") or ""
                        if not role_sent:
                            role_sent = True
                            yield _chunk({"role": "assistant", "content": ""})
                        yield _chunk({"reasoning_content": d})
                        _reason_buf += d
                        streamed_any = True
                    elif ptype == "response.output_item.added":
                        item = piece.get("item") or {}
                        if item.get("type") == "function_call":
                            for fc_chunk in _fc_chunks(item):
                                yield fc_chunk
                            tool_count = tool_count_holder[0]
                            streamed_any = True
                    elif ptype == "response.function_call_arguments.delta":
                        iid = piece.get("item_id") or ""
                        idx = tool_idx.get(iid, 0)
                        frag = piece.get("delta") or ""
                        yield _chunk({"tool_calls": [{
                            "index": idx,
                            "function": {"arguments": frag},
                        }]})
                        if idx not in tool_args:
                            tool_args[idx] = ""
                        tool_args[idx] += frag
                        streamed_any = True
                    elif ptype == "response.output_text.done":
                        txt = piece.get("text") or ""
                        if txt and not _content_buf:
                            if not role_sent:
                                role_sent = True
                                yield _chunk({"role": "assistant", "content": ""})
                            yield _chunk({"content": txt})
                            _content_buf += txt
                            streamed_any = True
                    elif ptype == "response.output_item.done":
                        item = piece.get("item") or {}
                        if isinstance(item, dict) and item.get("type") == "function_call":
                            # Upstream only announces the call once it is done
                            # (no add/delta sequence): recover it here.
                            for fc_chunk in _fc_chunks(item):
                                yield fc_chunk
                            tool_count = tool_count_holder[0]
                            streamed_any = True
                        elif isinstance(item, dict) and item.get("type") == "message":
                            txts = []
                            for part in item.get("content") or []:
                                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                                    if part.get("text"):
                                        txts.append(part["text"])
                            txt = "".join(txts)
                            if txt and not _content_buf:
                                if not role_sent:
                                    role_sent = True
                                    yield _chunk({"role": "assistant", "content": ""})
                                yield _chunk({"content": txt})
                                _content_buf += txt
                                streamed_any = True
                    elif ptype in ("response.completed", "response.incomplete", "response.failed") or (
                        ptype.startswith("response.")
                        and isinstance(piece.get("response"), dict)
                        and piece.get("response", {}).get("status") in ("completed", "incomplete", "failed")
                    ):
                        response_obj = piece.get("response") or {}
                        status = response_obj.get("status") or ""
                        if status == "failed":
                            ferr = response_obj.get("error") or {}
                            fmsg = (ferr.get("message") if isinstance(ferr, dict) else None) or "Upstream error"
                            _log(f"[zen] Responses {status} (attempt {attempt}): {fmsg}")
                            last_error = ValueError(fmsg)
                            if not streamed_any and attempt < attempts and not await _client_gone(request):
                                saw_error_event = True
                                break  # retry below
                            if (
                                streamed_any and _content_buf and not tool_streamed
                                and continuations < MAX_STREAM_CONTINUATIONS
                                and not await _client_gone(request)
                            ):
                                saw_error_event = True
                                continuations += 1
                                resp_body = _responses_continuation_body(resp_body, _content_buf)
                                _content_buf = ""
                                _reason_buf = ""
                                _log(f"[zen] Continuing responses stream after failed status (continuation {continuations})")
                                break  # retry below
                            yield _openai_stream_error(fmsg)
                            return
                        if not _content_buf:
                            try:
                                for oitem in response_obj.get("output") or []:
                                    if not isinstance(oitem, dict):
                                        continue
                                    if oitem.get("type") == "message":
                                        for part in oitem.get("content") or []:
                                            if isinstance(part, dict) and part.get("type") in ("output_text", "text") and part.get("text"):
                                                if not role_sent:
                                                    role_sent = True
                                                    yield _chunk({"role": "assistant", "content": ""})
                                                yield _chunk({"content": part["text"]})
                                                _content_buf += part["text"]
                                                streamed_any = True
                            except Exception:
                                pass
                            if _content_buf:
                                _log(f"[zen] [{model}|{proxy_addr or 'direct'}] recovered content from {status or 'completed'} snapshot ({len(_content_buf)} chars, no deltas)")
                        usage = _responses_usage_to_chat(response_obj.get("usage"))
                        for fitem in response_obj.get("output") or []:
                            # A completed envelope may be the ONLY carrier of
                            # function calls (no add/delta sequence upstream):
                            # recover each call so the turn is never empty of
                            # the tool use the model decided.
                            if isinstance(fitem, dict) and fitem.get("type") == "function_call":
                                for fc_chunk in _fc_chunks(fitem):
                                    yield fc_chunk
                                tool_count = tool_count_holder[0]
                                streamed_any = True
                        if status == "incomplete":
                            # Deterministic budget exhaustion (e.g. max_output_tokens
                            # hit on a micro-request): retrying the identical
                            # request just burns attempts. Translate to a
                            # length finish immediately, never retry.
                            finish_reason = "length"
                        else:
                            finish_reason = "tool_calls" if tool_count else "stop"
                        if finish_reason == "stop" and not _content_buf:
                            # Empty completed turn: the client sees finish=stop
                            # with no text and no tool call and ends the agent
                            # loop — the classic "stops silently, zero errors".
                            # Retry on another exit instead; only surface the
                            # empty turn when every attempt came back empty.
                            _log(f"[zen] [{model}|{proxy_addr or 'direct'}] empty completed turn (reason_len={len(_reason_buf)}), retrying another exit (attempt {attempt})")
                            saw_error_event = True
                            last_error = ValueError("upstream returned an empty completed turn")
                            if attempt < attempts and not await _client_gone(request):
                                empty_turn_retry = True
                                break  # retry below
                        yield _chunk({}, finish_reason)
                        if usage:
                            yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model or '', 'choices': [], 'usage': usage})}\n\n"
                            _add_tokens(
                                model or "unknown",
                                usage.get("prompt_tokens") or 0,
                                usage.get("completion_tokens") or 0,
                                usage.get("prompt_cache_hit_tokens") or 0,
                                0,
                            )
                        if session_id and _reason_buf and _content_buf:
                            _remember_reasoning(session_id, _content_buf, _reason_buf)
                        yield "data: [DONE]\n\n"
                        stream_completed = True
                        if PROXY_POOL_ENABLED and proxy_addr:
                            proxy_pool.report_success(proxy_addr)
                        _log(f"[zen] OK [{model}|{proxy_addr or 'direct'}] responses stream complete content={len(_content_buf)} reason={len(_reason_buf)} tools={tool_count} finish={finish_reason}")
                        break
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.TransportError) as e:
                _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Responses stream interrupted: {_exc_desc(e)}")
                last_error = e
                if _safe_pool_stream_failure(proxy_addr, e):
                    transport_reported = True
                elif _is_client_abort(e):
                    transport_reported = True
                if not streamed_any and attempt < attempts and not await _client_gone(request):
                    pass  # retry below
                elif (
                    _content_buf and not tool_streamed
                    and continuations < MAX_STREAM_CONTINUATIONS
                    and not await _client_gone(request)
                ):
                    # Torn stream after partial content: resume from the cutoff
                    # on a fresh exit instead of truncating the turn.
                    continuations += 1
                    resp_body = _responses_continuation_body(resp_body, _content_buf)
                    _content_buf = ""
                    _reason_buf = ""
                    _log(f"[zen] Continuing responses stream after transport cut (continuation {continuations})")
                else:
                    yield _openai_stream_error(_transport_error_message(e), "upstream_error", "transport_error")
                    return
        finally:
            try:
                await resp.aclose()
            except Exception:
                pass
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.release(proxy_addr)

        if stream_completed:
            return
        if empty_turn_retry and attempt < attempts and not await _client_gone(request):
            # Last attempt came back `completed` with zero content — not a
            # proxy or streamer fault, just a dud generation. A different exit
            # gets a genuinely fresh roll, so rotate without blacklisting.
            try:
                await proxy_pool.rotate_away(proxy_addr)
            except Exception:
                pass
            await _backoff(attempt)
            attempt += 1
            continue
        if not streamed_any and attempt < attempts and not await _client_gone(request):
            # Empty graceful EOF: the tunnel accepted the request but delivered
            # zero events. Same class as a torn stream — blacklist and rotate
            # so the retry (and the next request) does not stick to the same
            # dead exit. Mirrors the chat-completions empty-stream path.
            if (
                PROXY_POOL_ENABLED and proxy_addr
                and not saw_error_event and not transport_reported
            ):
                _safe_pool_stream_failure(proxy_addr)
            if not transport_reported and not saw_error_event:
                last_error = ValueError(f"responses stream ended before completion (streamed_any={streamed_any} content_len={len(_content_buf)})")
                _log(f"[zen] [{model}|{proxy_addr or 'direct'}] empty responses EOF (attempt {attempt}); req={_responses_diag(resp_body)}")
            await _backoff(attempt)
            attempt += 1
            continue
        if PROXY_POOL_ENABLED and proxy_addr:
            if not streamed_any and not saw_error_event and not transport_reported:
                _safe_pool_stream_failure(proxy_addr)
            elif streamed_any:
                # Partial then graceful EOF: rotate without poisoning the pool.
                try:
                    await proxy_pool.rotate_away(proxy_addr)
                except Exception:
                    pass
        if (
            streamed_any and _content_buf and not tool_streamed
            and continuations < MAX_STREAM_CONTINUATIONS
            and (attempt < attempts or continuations <= MAX_STREAM_CONTINUATIONS)
            and not await _client_gone(request)
        ):
            # Graceful EOF after partial content, no finish event: resume from
            # the cutoff instead of truncating the turn.
            continuations += 1
            resp_body = _responses_continuation_body(resp_body, _content_buf)
            _log(f"[zen] Continuing responses stream after empty EOF (continuation {continuations})")
            _content_buf = ""
            _reason_buf = ""
            await _backoff(attempt)
            attempt += 1
            continue
        err = ValueError(f"responses stream ended before completion (streamed_any={streamed_any} content_len={len(_content_buf)})")
        _log(f"[zen] [{model}|{proxy_addr or 'direct'}] {_exc_desc(err)}; req={_responses_diag(resp_body)}")
        yield _openai_stream_error(_transport_error_message(last_error or err), "upstream_error", "incomplete_stream")
        return

    yield _openai_stream_error(
        f"connection error: responses stream failed after {attempts + 1} attempts: {_exc_desc(last_error)}",
        "upstream_error",
        "transport_error",
    )


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
    gen=None,
):
    """Stream the upstream response and assemble a buffered chat.completion.

    Consumes ``_zen_stream_with_retry`` (which already owns proxy-pool
    selection, retries, torn-stream resume and terminal-error framing) and
    folds the emitted deltas into the same JSON object the non-streaming
    transport would have returned. ``gen`` overrides the stream source (used by
    the muse-spark Responses-API route, which emits the same chat chunks).
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

    if gen is None:
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
    # _openai_stream_error emits a bare {"error": ...} event with no content
    # chunks; saw_error carries its message. (Legacy "[upstream error] ..."
    # content framing retired: strict clients ingest content as answers.)
    if saw_error and not has_tool_calls and not content:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": saw_error, "type": "upstream_error"}},
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
    _ensure_req_id()
    if req_body.get("stream") is False and _needs_stream_bridge(model):
        # Buffered transport times out on this model's silent warm-up; stream
        # internally and hand the caller a normal completion object instead.
        return await _aggregate_upstream_completion(
            request, req_body, headers, user, messages, session_id, model, max_retries
        )
    last_error = None
    attempts = MAX_RETRIES if max_retries is None else max_retries
    last_status: int | None = None
    pool_exhausted_retriable = False
    free_tier_gates = 0  # consecutive FreeTier gates this request; fail fast at _FREE_TIER_FAIL_FAST_GATES

    for attempt in range(attempts + 1):
        if await _client_gone(request):
            _log("[zen] Client disconnected; aborting retries")
            return None
        if PROXY_POOL_ENABLED:
            # Load pool if needed
            if not proxy_pool.ready:
                await proxy_pool.load()

            p = await _select_with_repoll()
            if p is None:
                _log(f"[pool] No proxy available ({proxy_pool.get_pool_state()}), "
                     f"forcing refresh")
                await _maybe_force_refresh("No proxy available")
                p = await proxy_pool.select()
                if p is None:
                    _log("[pool] Still no proxy after refresh, falling back to direct")
                    client = _default_client
                    proxy_addr = None
                else:
                    proxy_addr = p["address"]
                    client = proxy_pool.get_client(f"socks5://{proxy_addr}")
                    _log(f"[pool] Retry {attempt}: using proxy {proxy_addr}")
            else:
                proxy_addr = p["address"]
                client = proxy_pool.get_client(f"socks5://{proxy_addr}")

        else:
            client = _default_client
            proxy_addr = None

        try:
            try:
                resp = await client.post(
                    "/zen/v1/chat/completions",
                    json=req_body,
                    headers=headers,
                )
            except Exception as e:
                _log(f"[zen] Request failed (attempt {attempt}): {_exc_desc(e)}")
                _safe_pool_failure(proxy_addr, e, hard=False)
                last_error = e
                free_tier_gates = 0  # transport outcome breaks the gate streak
                if _is_client_abort(e):
                    return JSONResponse(
                        status_code=502,
                        content={"error": {"message": f"Client aborted: {_exc_desc(e)}", "type": "upstream_error"}},
                    )
                if attempt < attempts:
                    await _backoff(attempt)
                continue

            try:
                body_bytes = await resp.aread()
            except Exception as e:
                _log(f"[zen] Response read failed (attempt {attempt}): {_exc_desc(e)}")
                _safe_pool_failure(proxy_addr, e, hard=False)
                last_error = e
                free_tier_gates = 0  # transport outcome breaks the gate streak
                if _is_client_abort(e):
                    return JSONResponse(
                        status_code=502,
                        content={"error": {"message": f"Client aborted: {_exc_desc(e)}", "type": "upstream_error"}},
                    )
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
                last_status = 429
                _log(f"[zen] 429 (attempt {attempt}): {err_msg}")
                _log_429_hint()
                if PROXY_POOL_ENABLED and proxy_addr:
                    proxy_pool.report_ratelimit(proxy_addr)
                free_tier_gates = 0  # 429 is exit-shaped quota, not the request gate
                # fresh quota. Rotate and retry instead of telling the client;
                # only surface the error once every attempt is exhausted.
                if attempt < attempts and not await _client_gone(request):
                    await _maybe_backoff(attempt, 429)
                    continue
                pool_exhausted_retriable = bool(PROXY_POOL_ENABLED)
                if pool_exhausted_retriable:
                    break
                return _local_rate_limit_response(err_msg + " (free model rate limit)")

            if resp.status_code >= 400:
                err_msg = (data.get("error") or {}).get("message") or f"HTTP {resp.status_code}"
                last_status = resp.status_code
                if _is_too_large_status(resp.status_code):
                    # 413 = size, not health: rotate with ZERO pool-health
                    # accounting (no cooldown/blacklist/counters), keep the
                    # body for direct-fallback salvage below. 413 is smaller
                    # than this body in every proxy but MAY succeed direct
                    # (different upstream limit) — so treat it like a
                    # retriable exhaust only when the flag allows direct.
                    _log(f"[zen] 413 too-large (attempt {attempt}): {err_msg} — rotating (no penalty)")
                    _rotate_on_too_large(proxy_addr)
                    if attempt < attempts and not await _client_gone(request):
                        await _maybe_backoff(attempt, resp.status_code)
                        continue
                    if PROXY_POOL_ENABLED and ALLOW_DIRECT_FALLBACK:
                        pool_exhausted_retriable = True
                        break
                    return JSONResponse(
                        status_code=413,
                        content={"error": {"message": err_msg, "type": "upstream_error", "code": "payload_too_large"}},
                    )
                is_context_exceeded = _is_context_limit_error(data, body_text)
                bare_500 = _is_bare_internal_error(resp.status_code, data, body_text)
                _log(f"[zen] Error {resp.status_code}: {err_msg}")
                # Entitlement error (401 ModelError incl. promotion-ended):
                # identical on every proxy/retry — retire the model and fail
                # fast instead of burning retries across the pool.
                if _is_unavailable_error(data, body_text) or _is_promotion_ended_error(data, body_text, resp.status_code, model):
                    _log(f"[zen] Model {model} retired upstream (ModelError); dropping from list")
                    _mark_model_dead(model, persistent=_has_promotion_phrase(data, body_text))
                    return JSONResponse(
                        status_code=resp.status_code,
                        content={"error": {"message": f"{model} is no longer available: {err_msg}", "type": "upstream_error", "code": "model_retired"}},
                    )
                if resp.status_code == 403 and _is_free_tier_gate_error(data, body_text):
                    free_tier_gates += 1
                    _log(f"[zen] [{model}|{proxy_addr or 'direct'}] FreeTier gate 403 ({free_tier_gates}/{_FREE_TIER_FAIL_FAST_GATES}): {err_msg} — rotating (no penalty)")
                    if PROXY_POOL_ENABLED and proxy_addr:
                        _report_free_tier_block(proxy_addr)
                    if free_tier_gates >= _FREE_TIER_FAIL_FAST_GATES:
                        _log(f"[zen] [{model}] FreeTier gate on {free_tier_gates} consecutive exits — request-shaped, failing fast")
                        return JSONResponse(
                            status_code=403,
                            content={"error": {"message": err_msg, "type": "upstream_error", "code": "free_tier_gate"}},
                        )
                    if attempt < attempts and not await _client_gone(request):
                        await _maybe_backoff(attempt, resp.status_code)
                        continue
                    return JSONResponse(
                        status_code=403,
                        content={"error": {"message": err_msg, "type": "upstream_error", "code": "free_tier_gate"}},
                    )
                # Not a proxy failure: 4xx/5xx are upstream or request errors that
                # repeat identically on every proxy, so never blacklist for them.
                # Fail fast on deterministic errors (all 4xx except retriable
                # 408/429, and the bare 500 "Internal server error" that means
                # the model is down upstream): retrying them across proxies only
                # multiplies one fast failure into ~6 slow ones with backoff.
                # Retriable (408/429/502/503/504/520-530) rotate to the next
                # proxy; 504 fast-breaks pool cycling (no backoff sleep).
                if is_context_exceeded:
                    _log("[zen] Context limit error; not retrying on another proxy")
                if bare_500:
                    _log(f"[zen] Model {model} appears down upstream (bare 500); failing fast — try another model")
                    err_msg = f"{err_msg} (model appears down upstream; try another free model)"
                elif _is_retriable_status(resp.status_code) and not is_context_exceeded and attempt < attempts:
                    # Escalating 5xx/504 cooldown; 504 fast-break stops cycling.
                    if not _report_upstream_status(proxy_addr, resp.status_code):
                        return JSONResponse(
                            status_code=resp.status_code,
                            content={"error": {"message": err_msg, "type": "upstream_error"}},
                        )
                    await _maybe_backoff(attempt, resp.status_code)
                    continue
                elif _is_retriable_status(resp.status_code) and not is_context_exceeded:
                    _report_upstream_status(proxy_addr, resp.status_code)
                    pool_exhausted_retriable = bool(PROXY_POOL_ENABLED)
                    if pool_exhausted_retriable:
                        break
                elif resp.status_code >= 500 and not is_context_exceeded and attempt < attempts:
                    if not _report_upstream_status(proxy_addr, resp.status_code):
                        return JSONResponse(
                            status_code=resp.status_code,
                            content={"error": {"message": err_msg, "type": "upstream_error"}},
                        )
                    await _maybe_backoff(attempt, resp.status_code)
                    continue
                return JSONResponse(
                    status_code=resp.status_code,
                    content={"error": {"message": err_msg, "type": "upstream_error"}},
                )

            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_success(proxy_addr)
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
            return data
        finally:
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.release(proxy_addr)

    # Exhaust-then-direct: pool cycled through all retries on a retriable
    # status — one direct-fetch attempt (no proxy, relay headers stripped).
    # Gated behind --allow-direct-fallback / OPENCODE_ALLOW_DIRECT_FALLBACK
    # (default off: a direct fetch exposes this server's own IP upstream).
    if pool_exhausted_retriable and not await _client_gone(request):
        if not ALLOW_DIRECT_FALLBACK:
            _log(f"[zen] pool exhausted on retriable {last_status}; direct fallback disabled (opt in with --allow-direct-fallback)")
        else:
            _log_direct_fallback(model, last_status)
            code, data, text = await _direct_buffered_post(
                "/zen/v1/chat/completions", req_body, headers
            )
            if code is not None and code < 400 and isinstance(data, dict) and data.get("choices"):
                usage = (data.get("usage") or {})
                if isinstance(usage, dict) and ("prompt_tokens" in usage or "completion_tokens" in usage):
                    _add_tokens(model,
                        usage.get("prompt_tokens") or 0,
                        usage.get("completion_tokens") or 0,
                        usage.get("prompt_cache_hit_tokens") or (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                        usage.get("prompt_cache_miss_tokens") or 0,
                    )
                _log(f"[zen] OK [{model}|direct] direct-fetch succeeded after pool exhaustion")
                return data
            if code is not None:
                _log(f"[zen] direct-fetch returned {code}: {text[:200]!r}")
                if isinstance(data, dict) and data.get("choices"):
                    return data
                return JSONResponse(
                    status_code=code if isinstance(code, int) and code >= 400 else 502,
                    content={"error": {"message": (data.get("error") or {}).get("message") if isinstance(data, dict) else text or "Direct fetch failed", "type": "upstream_error"}},
                )
            _log(f"[zen] direct-fetch failed: {text[:200]}")
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
    _ensure_req_id()
    last_error = None
    last_status: int | None = None
    pool_exhausted_retriable = False
    direct_only = False  # set after pool exhaustion: next loop is the one direct-fetch attempt
    attempts = MAX_RETRIES if max_retries is None else max_retries
    finish_delivered = False  # True once a finish_reason chunk was forwarded
    tool_streamed = False  # True once a tool-call fragment was forwarded
    continuations = 0  # mid-stream resume attempts (see MAX_STREAM_CONTINUATIONS)

    attempt = 0
    free_tier_gates = 0  # consecutive FreeTier gates this request; fail fast at _FREE_TIER_FAIL_FAST_GATES
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
        if PROXY_POOL_ENABLED and not direct_only:
            if not proxy_pool.ready:
                await proxy_pool.load()

            p = await _select_with_repoll()
            if p is None:
                _log(f"[pool] No proxy available ({proxy_pool.get_pool_state()}), forcing refresh")
                await _maybe_force_refresh("No proxy available")
                p = await proxy_pool.select()
                if p is None:
                    _log("[pool] Still no proxy after refresh, falling back to direct")
                    client = _stream_default_client
                    proxy_addr = None
                else:
                    proxy_addr = p["address"]
                    client = proxy_pool.get_client(
                        f"socks5://{proxy_addr}", streaming=True
                    )
            else:
                proxy_addr = p["address"]
                client = proxy_pool.get_client(
                    f"socks5://{proxy_addr}", streaming=True
                )
        else:
            # Mirror the Responses-stream branch: the post-exhaustion
            # direct-fetch attempt strips relay headers and uses the
            # dedicated no-proxy client instead of the default one.
            if direct_only:
                headers = _strip_direct_headers(headers)
            client = _stream_direct_client if direct_only else _stream_default_client
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
            _safe_pool_failure(proxy_addr, err, hard=True)
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
            if _is_client_abort(e):
                if PROXY_POOL_ENABLED and proxy_addr:
                    _log(f"[pool] client abort ({type(e).__name__}); not marking {proxy_addr} failed")
                    proxy_pool.release(proxy_addr)
                yield _openai_stream_error(
                    f"Client aborted: {_exc_desc(e)}",
                    "upstream_error",
                    "transport_error",
                )
                return
            # Same as the responses setup path: a setup failure means the
            # tunnel never established, so rotate immediately instead of
            # spending the soft-grace retry on the same dead exit.
            # (Abort already returned above, so _safe_pool_failure records.)
            _safe_pool_failure(proxy_addr, e, hard=True)
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.release(proxy_addr)
            last_error = e
            free_tier_gates = 0  # transport outcome breaks the gate streak
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
                if _is_client_abort(e):
                    if PROXY_POOL_ENABLED and proxy_addr:
                        proxy_pool.release(proxy_addr)
                    await resp.aclose()
                    yield _openai_stream_error(f"Client aborted: {_exc_desc(e)}")
                    return
                _safe_pool_failure(proxy_addr, e, hard=False)
                if PROXY_POOL_ENABLED and proxy_addr:
                    proxy_pool.release(proxy_addr)
                await resp.aclose()
                if attempt < attempts:
                    await _maybe_backoff(attempt, 429)
                    attempt += 1
                    continue
                yield _openai_stream_error(f"Upstream error: {_exc_desc(e)}")
                return
            _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Stream 429 (attempt {attempt}): {err_msg}")
            _log_429_hint()
            last_status = 429
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.report_ratelimit(proxy_addr)
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.release(proxy_addr)
            await resp.aclose()
            free_tier_gates = 0  # 429 is exit-shaped quota, not the request gate
            # Per-IP quota burn: the next proxy has fresh quota. Rotate and
            # retry before ever telling the client about the rate limit.
            if attempt < attempts and not await _client_gone(request):
                await _maybe_backoff(attempt, 429)
                attempt += 1
                continue
            if PROXY_POOL_ENABLED and not pool_exhausted_retriable and ALLOW_DIRECT_FALLBACK:
                # Exhaust-then-direct: one direct-fetch attempt (no proxy,
                # relay headers stripped) before surfacing the 429.
                pool_exhausted_retriable = True
                direct_only = True
                _log_direct_fallback(model, 429)
                headers = _strip_direct_headers(headers)
                await _maybe_backoff(attempt, 429)
                attempt += 1
                continue
            if PROXY_POOL_ENABLED and not ALLOW_DIRECT_FALLBACK:
                _log(f"[zen] pool exhausted on retriable 429; direct fallback disabled (opt in with --allow-direct-fallback)")
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
                if _is_client_abort(e):
                    if PROXY_POOL_ENABLED and proxy_addr:
                        proxy_pool.release(proxy_addr)
                    await resp.aclose()
                    yield _openai_stream_error(f"Client aborted: {_exc_desc(e)}")
                    return
                _safe_pool_failure(proxy_addr, e, hard=False)
                if PROXY_POOL_ENABLED and proxy_addr:
                    proxy_pool.release(proxy_addr)
                await resp.aclose()
                if attempt < attempts:
                    await _maybe_backoff(attempt, None)
                    attempt += 1
                    continue
                yield _openai_stream_error(f"Upstream error: {_exc_desc(e)}")
                return
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.release(proxy_addr)
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
                free_tier_gates = 0  # region outcome breaks the gate streak
                if attempt < attempts:
                    await _backoff(attempt)
                    attempt += 1
                    continue
                yield _openai_stream_error(err_msg, "upstream_error", "region_blocked")
                return

            if resp.status_code == 403 and _is_free_tier_gate_error(data, body_text):
                free_tier_gates += 1
                _log(f"[zen] [{model}|{proxy_addr or 'direct'}] FreeTier gate 403 ({free_tier_gates}/{_FREE_TIER_FAIL_FAST_GATES}): {err_msg} — rotating (no penalty)")
                if PROXY_POOL_ENABLED and proxy_addr:
                    _report_free_tier_block(proxy_addr)
                await resp.aclose()
                if free_tier_gates >= _FREE_TIER_FAIL_FAST_GATES:
                    _log(f"[zen] [{model}] FreeTier gate on {free_tier_gates} consecutive exits — request-shaped, failing fast")
                    yield _openai_stream_error(err_msg, "upstream_error", "free_tier_gate")
                    return
                if attempt < attempts and not await _client_gone(request):
                    await _maybe_backoff(attempt, resp.status_code)
                    attempt += 1
                    continue
                if PROXY_POOL_ENABLED and not direct_only and ALLOW_DIRECT_FALLBACK:
                    direct_only = True
                    _log_direct_fallback(model, resp.status_code)
                    headers = _strip_direct_headers(headers)
                    await _maybe_backoff(attempt, resp.status_code)
                    attempt += 1
                    continue
                yield _openai_stream_error(err_msg, "upstream_error", "free_tier_gate")
                return

            # Not a proxy failure: 4xx/5xx are upstream or request errors
            # that repeat identically on every proxy, so never blacklist for
            # them. A 503 is upstream capacity: keep the current exit and let
            # the retry below ride it out (no report_success — it was not
            # healthy for this call). 413 is a size signal (dedicated branch
            # above) — never cooled/blacklisted/counted.
            if _is_unavailable_error(data, body_text) or _is_promotion_ended_error(data, body_text, resp.status_code, model):
                # Entitlement error: identical on every proxy/retry. Retire
                # the model and terminate the stream immediately.
                await resp.aclose()
                _mark_model_dead(model, persistent=_has_promotion_phrase(data, body_text))
                yield _openai_stream_error(f"{model} is no longer available: {err_msg}", "upstream_error", "model_retired")
                return
            if _is_too_large_status(resp.status_code):
                _log(f"[zen] [{model}|{proxy_addr or 'direct'}] Stream 413 too-large: {err_msg} — rotating (no penalty)")
                _rotate_on_too_large(proxy_addr)
                await resp.aclose()
                last_status = 413
                if attempt < attempts and not await _client_gone(request):
                    await _maybe_backoff(attempt, resp.status_code)
                    attempt += 1
                    continue
                if PROXY_POOL_ENABLED and not pool_exhausted_retriable:
                    pool_exhausted_retriable = True
                    if ALLOW_DIRECT_FALLBACK:
                        direct_only = True
                        _log_direct_fallback(model, resp.status_code)
                        headers = _strip_direct_headers(headers)
                        await _maybe_backoff(attempt, resp.status_code)
                        attempt += 1
                        continue
                    _log(f"[zen] pool exhausted on 413; direct fallback disabled (opt in with --allow-direct-fallback)")
                yield _openai_stream_error(err_msg, "upstream_error", "413")
                return
            if is_context_exceeded:
                _log("[zen] Context limit error; not retrying on another proxy")
            await resp.aclose()
            last_status = resp.status_code
            if _is_bare_internal_error(resp.status_code, data, body_text):
                _log(f"[zen] Model {model} appears down upstream (bare 500); failing fast — try another model")
                err_msg = f"{err_msg} (model appears down upstream; try another free model)"
            elif _is_retriable_status(resp.status_code) and not is_context_exceeded and attempt < attempts:
                # Escalating 5xx/504 cooldown for this exit (504 fast-breaks —
                # False means stop cycling, surface the error on this pass).
                if not _report_upstream_status(proxy_addr, resp.status_code):
                    yield _openai_stream_error(err_msg, "upstream_error", str(resp.status_code))
                    return
                await _maybe_backoff(attempt, resp.status_code)
                attempt += 1
                continue
            elif _is_retriable_status(resp.status_code) and not is_context_exceeded and PROXY_POOL_ENABLED and not pool_exhausted_retriable:
                if not _report_upstream_status(proxy_addr, resp.status_code):
                    yield _openai_stream_error(err_msg, "upstream_error", str(resp.status_code))
                    return
                pool_exhausted_retriable = True
                if ALLOW_DIRECT_FALLBACK:
                    direct_only = True
                    _log_direct_fallback(model, resp.status_code)
                    headers = _strip_direct_headers(headers)
                    await _maybe_backoff(attempt, resp.status_code)
                    attempt += 1
                    continue
                _log(f"[zen] pool exhausted on retriable {resp.status_code}; direct fallback disabled (opt in with --allow-direct-fallback)")
            elif resp.status_code >= 500 and not is_context_exceeded and attempt < attempts:
                if not _report_upstream_status(proxy_addr, resp.status_code):
                    yield _openai_stream_error(err_msg, "upstream_error", str(resp.status_code))
                    return
                await _maybe_backoff(attempt, resp.status_code)
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
                        _safe_pool_failure(proxy_addr, err)
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
                        _safe_pool_failure(proxy_addr, err)
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
                        _safe_pool_failure(proxy_addr, err)
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
                        if _is_unavailable_error(piece, line) or _is_promotion_ended_error(piece, line, None, model):
                            _mark_model_dead(model, persistent=_has_promotion_phrase(piece, line))
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
                # chance. Client aborts never mark the pool failed.
                _safe_pool_stream_failure(proxy_addr, e)
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
                        _transport_error_message(e),
                        "upstream_error",
                        "transport_error",
                    )
                    return
        finally:
            try:
                await resp.aclose()
            except Exception:
                pass
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.release(proxy_addr)
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
                _safe_pool_stream_failure(proxy_addr)
            elif PROXY_POOL_ENABLED and proxy_addr and streamed_any:
                # Rotate away from this connection without poisoning the pool
                try:
                    await proxy_pool.rotate_away(proxy_addr)
                except Exception:
                    pass
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
            # Infer terminal finish when the upstream closed without [DONE]
            # but content/tool fragments were already delivered: tool_calls
            # pending -> "tool_calls", else content present -> "stop". Only
            # the truly-empty case falls through to the error path below.
            inferred = "tool_calls" if tool_streamed else ("stop" if _content_buf else None)
            if inferred and streamed_any:
                _log(f"[zen] [{model}|{proxy_addr or 'direct'}] inferring finish={inferred} on graceful close without [DONE] (content_len={len(_content_buf)})")
                if session_id and _reason_buf and _content_buf:
                    _remember_reasoning(session_id, _content_buf, _reason_buf)
                _term_id = oc_id("chatcmpl")
                _term_created = int(time.time())
                yield f"data: {json.dumps({'id': _term_id, 'object': 'chat.completion.chunk', 'created': _term_created, 'model': model or '', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': inferred}]})}\n\n"
                yield "data: [DONE]\n\n"
                return
            yield _openai_stream_error(
                _transport_error_message(err), "upstream_error", "incomplete_stream"
            )
        return

    # All retries exhausted
    yield _openai_stream_error(
        f"connection error: stream failed after {attempts + 1} attempts: {_exc_desc(last_error)}",
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
    _ensure_req_id()
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
    free_tier_gates = 0  # consecutive FreeTier gates this request; fail fast at _FREE_TIER_FAIL_FAST_GATES
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

            p = await _select_with_repoll()
            if p is None:
                _log(f"[pool] No proxy ({proxy_pool.get_pool_state()}), forcing refresh")
                await _maybe_force_refresh("No proxy")
                p = await proxy_pool.select()
                if p is None:
                    _log("[pool] Fallback to direct")
                    client = _stream_default_client
                    proxy_addr = None
                else:
                    proxy_addr = p["address"]
                    client = proxy_pool.get_client(
                        f"socks5://{proxy_addr}", streaming=True
                    )
            else:
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
                    _log_429_hint()
                    if PROXY_POOL_ENABLED and proxy_addr:
                        proxy_pool.report_ratelimit(proxy_addr)
                    free_tier_gates = 0  # 429 is exit-shaped quota, not the request gate
                    # Per-IP quota: rotate to a fresh exit before telling the
                    # client we are rate-limited.
                    if attempt < attempts and not await _client_gone(request):
                        await _maybe_backoff(attempt, 429)
                        attempt += 1
                        continue
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
                        free_tier_gates = 0  # region outcome breaks the gate streak
                        if attempt < attempts:
                            await _backoff(attempt)
                            attempt += 1
                            continue
                        yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": err_msg}})
                        return
                    if resp.status_code == 403 and _is_free_tier_gate_error(data, body_text):
                        free_tier_gates += 1
                        _log(f"[zen] [{model}|{proxy_addr or 'direct'}] FreeTier gate 403 ({free_tier_gates}/{_FREE_TIER_FAIL_FAST_GATES}): {err_msg} — rotating (no penalty)")
                        if PROXY_POOL_ENABLED and proxy_addr:
                            _report_free_tier_block(proxy_addr)
                        if free_tier_gates >= _FREE_TIER_FAIL_FAST_GATES:
                            _log(f"[zen] [{model}] FreeTier gate on {free_tier_gates} consecutive exits — request-shaped, failing fast")
                            yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": err_msg}})
                            return
                        if attempt < attempts and not await _client_gone(request):
                            await _maybe_backoff(attempt, resp.status_code)
                            attempt += 1
                            continue
                        yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": err_msg}})
                        return
                    # Not a proxy failure: 4xx/5xx are upstream or request
                    # errors that repeat identically on every proxy. 413 is a
                    # size signal — rotate with zero health accounting.
                    if _is_unavailable_error(data, body_text) or _is_promotion_ended_error(data, body_text, resp.status_code, model):
                        _mark_model_dead(model, persistent=_has_promotion_phrase(data, body_text))
                        yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": f"{model} is no longer available: {err_msg}"}})
                        return
                    if _is_too_large_status(resp.status_code):
                        _log(f"[zen] Anthropic stream 413 too-large (attempt {attempt}): {err_msg} — rotating (no penalty)")
                        _rotate_on_too_large(proxy_addr)
                        if attempt < attempts and not await _client_gone(request):
                            await _maybe_backoff(attempt, resp.status_code)
                            attempt += 1
                            continue
                        yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": err_msg, "code": "payload_too_large"}})
                        return
                    if is_context_exceeded:
                        _log("[zen] Context limit error; not retrying on another proxy")
                    if _is_bare_internal_error(resp.status_code, data, body_text):
                        _log(f"[zen] Model {model} appears down upstream (bare 500); failing fast — try another model")
                        err_msg = f"{err_msg} (model appears down upstream; try another free model)"
                    elif _is_retriable_status(resp.status_code) and not is_context_exceeded and attempt < attempts:
                        # Escalating 5xx/504 cooldown; 504 fast-break stops cycling.
                        if not _report_upstream_status(proxy_addr, resp.status_code):
                            yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": err_msg}})
                            return
                        await _maybe_backoff(attempt, resp.status_code)
                        attempt += 1
                        continue
                    elif resp.status_code >= 500 and not is_context_exceeded and attempt < attempts:
                        if not _report_upstream_status(proxy_addr, resp.status_code):
                            yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": err_msg}})
                            return
                        await _maybe_backoff(attempt, resp.status_code)
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
                            if _is_unavailable_error(_piece, raw_line) or _is_promotion_ended_error(_piece, raw_line, None, model):
                                _mark_model_dead(model, persistent=_has_promotion_phrase(_piece, raw_line))
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
                    _safe_pool_stream_failure(proxy_addr)
                elif PROXY_POOL_ENABLED and proxy_addr and streamed_any:
                    try:
                        await proxy_pool.rotate_away(proxy_addr)
                    except Exception:
                        pass
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
                    # Infer terminal stop when the upstream closed without a
                    # finish_reason but content/tool fragments were delivered:
                    # tool_calls pending -> "tool_use", else content -> "end_turn".
                    # Truly-empty (no content, no tools) keeps the error path.
                    _inferred = "tool_use" if tool_streamed else ("end_turn" if _content_buf else None)
                    if _inferred and (headers_sent or streamed_any):
                        _log(f"[zen] Anthropic inferring stop={_inferred} on close without finish_reason (content_len={len(_content_buf)})")
                        if session_id and _reason_buf and _content_buf:
                            _remember_reasoning(session_id, _content_buf, _reason_buf)
                        for i in close_indices():
                            yield send_sse("content_block_stop", {"type": "content_block_stop", "index": i})
                        yield send_sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": _inferred}, "usage": {"output_tokens": output_tokens}})
                        yield send_sse("message_stop", {"type": "message_stop"})
                        return
                    # Message started but did not finish: close open blocks
                    # and emit a clean error event.
                    for i in close_indices():
                        yield send_sse("content_block_stop", {"type": "content_block_stop", "index": i})
                yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": "Stream interrupted: upstream closed the connection before completing the message"}})
                return

        except Exception as e:
            _log(f"[zen] Anthropic stream HTTP error (attempt {attempt}): {_exc_desc(e)}")
            # Only transport-level failures are proxy failures; a bug in our
            # translation code must not blacklist a healthy proxy. Client
            # aborts never mark the pool failed.
            if isinstance(e, httpx.HTTPError):
                _safe_pool_stream_failure(proxy_addr, e)
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
        finally:
            if PROXY_POOL_ENABLED and proxy_addr:
                proxy_pool.release(proxy_addr)

    if not headers_sent:
        yield send_sse("error", {"type": "error", "error": {"type": "upstream_error", "message": f"connection error: stream failed after {attempts + 1} attempts: {_exc_desc(last_error)}"}})


# ── NVIDIA NIM direct transport (key rotation on 2x consecutive 429) ──

# Fail-fast: if this many *consecutive* 429s happen without a single key
# succeeding, the whole pool is saturated with rate limits. Returning a clear
# 429 to the client beats silently looping every key with backoff for minutes
# (which looks like "thinking" to the client). A "consecutive" run is broken by
# any success OR any non-429 (e.g. a socket error that still tries the pool).
NVIDIA_POOL_429_FAILFAST = 12

def nvidia_request_body(model, messages, stream, tools, tool_choice, max_tokens=None, max_completion_tokens=None, sampling=None):
    """Build a standard OpenAI-compatible body for the NVIDIA NIM endpoint.

    Unlike the Zen thinking-mode upstream, NIM accepts normal OpenAI messages;
    we only normalize roles and don't inject any reasoning_content.
    """
    body: dict = {"model": model, "messages": messages, "stream": bool(stream)}
    if tools:
        body["tools"] = tools
    if tool_choice:
        if tool_choice in ("auto", "none"):
            body["tool_choice"] = tool_choice
        else:
            body["tool_choice"] = "auto"
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        body["max_completion_tokens"] = max_completion_tokens
    if sampling:
        body.update(sampling)
    return body


def _nvidia_headers(key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }


def _nvidia_client(streaming: bool):
    from nvidia_pool import (
        NVIDIA_CONNECT_TIMEOUT,
        NVIDIA_READ_TIMEOUT,
        NVIDIA_STREAM_READ_TIMEOUT,
    )
    # trust_env=False: NVIDIA calls must go directly to integrate.api.nvidia.com
    # and must NOT be routed through the SOCKS proxy pool / system HTTP proxy,
    # otherwise heavy (long-thinking) models get mishandled (e.g. proxied 404s).
    return httpx.AsyncClient(
        base_url="https://integrate.api.nvidia.com",
        timeout=httpx.Timeout(
            connect=NVIDIA_CONNECT_TIMEOUT,
            read=NVIDIA_STREAM_READ_TIMEOUT if streaming else NVIDIA_READ_TIMEOUT,
            write=NVIDIA_STREAM_READ_TIMEOUT if streaming else NVIDIA_READ_TIMEOUT,
            pool=NVIDIA_CONNECT_TIMEOUT,
        ),
        proxy=None,
        trust_env=False,
    )


def _nvidia_rate_limit_response(err_msg: str) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": err_msg + " (nvidia key rate limit)",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
            }
        },
    )


async def nvidia_request_with_retry(
    req_body: dict,
    user: str,
    model: str,
    max_retries: int = None,
):
    """Buffered NVIDIA NIM call with key rotation on consecutive 429s.

    Keys rotate when one hits a rate limit twice in a row (nvidia_keys
    enforces the cooldown). Returns the parsed OpenAI completion dict, a
    JSONResponse error, or None if the client gave up.
    """
    attempts = max_retries if max_retries is not None else MAX_RETRIES
    last_error = None
    last_key = None
    last_status = None
    last_body = ""
    key_limit = len(nvidia_keys.keys) or 1
    # Consecutive pool-wide 429s (not broken by a success or non-429). When this
    # reaches NVIDIA_POOL_429_FAILFAST the whole pool is saturated → fail fast
    # instead of looping every key with backoff for minutes ("thinking").
    consecutive_429 = 0
    # Sweep the whole pool: each iteration picks the next usable key (the pool
    # advances internally), so a model that 404s for some keys (key lacks
    # access) but works for another eventually succeeds. Rate limits rotate and
    # cool down the offending key.
    for attempt in range(key_limit + attempts):
        key = nvidia_keys.select()
        if key is None:
            return _nvidia_rate_limit_response("all NVIDIA keys are in rate-limit cooldown")
        client = _nvidia_client(streaming=False)
        headers = _nvidia_headers(key)
        try:
            try:
                resp = await client.post(
                    "/v1/chat/completions", json=req_body, headers=headers
                )
            except Exception as e:
                last_error = e
                _log(f"[nvidia] Request failed (attempt {attempt}, key {key[-8:]}): {_exc_desc(e)}")
                if attempt < key_limit + attempts - 1:
                    await _backoff(attempt)
                    continue
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": f"NVIDIA request failed: {_exc_desc(e)}", "type": "upstream_error"}},
                )

            body_bytes = await resp.aread()
            body_text = body_bytes.decode("utf-8", errors="replace")
            last_key = key
            last_status = resp.status_code
            last_body = body_text
            try:
                data = json.loads(body_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {}

            is_429 = resp.status_code == 429
            is_rate_limit = is_429 or "FreeUsageLimitError" in body_text or (
                "erate" in body_text and "429" in body_text
            )
            if is_rate_limit:
                err_msg = (data.get("error") or {}).get("message") or "Rate limit exceeded"
                rotated = nvidia_keys.report_rate_limit(key)
                _log(f"[nvidia] 429 on key {key[-8:]} (attempt {attempt}): {err_msg} rotated={rotated}")
                consecutive_429 += 1
                # Whole pool saturated with rate limits → fail fast with a clear
                # 429 rather than silently looping every key for minutes.
                if consecutive_429 >= NVIDIA_POOL_429_FAILFAST:
                    _log(f"[nvidia] {consecutive_429} consecutive 429s → pool saturated; failing fast")
                    return _nvidia_rate_limit_response(
                        f"NVIDIA pool rate-limited ({consecutive_429} consecutive 429s); try again in a moment"
                    )
                # Two consecutive 429s rotate the key to cooldown; keep trying
                # other keys. Fewer than two → short backoff + retry same key.
                if attempt < key_limit + attempts - 1:
                    await _backoff(attempt)
                    continue
                return _nvidia_rate_limit_response(err_msg)

            if resp.status_code >= 400:
                consecutive_429 = 0  # a non-429 breaks the consecutive-429 run
                err_msg = (data.get("error") or {}).get("message") or f"HTTP {resp.status_code}"
                is_context = _is_context_limit_error(data, body_text)
                _log(f"[nvidia] Error {resp.status_code} (key {key[-8:]}): {err_msg} body={body_text[:200]!r}")
                if resp.status_code in (401, 403):
                    # Bad/revoked key — rotate away (not rate-limit but unusable).
                    nvidia_keys.report_rate_limit(key)
                if is_context:
                    # Context window exhaustion is per-model/request, not key.
                    return JSONResponse(
                        status_code=resp.status_code,
                        content={"error": {"message": err_msg, "type": "context_window_error"}},
                    )
                if _is_degraded_error(data, body_text):
                    # NVIDIA NIM deployment is DEGRADED: every key returns the same
                    # 400. Fail fast instead of sweeping the whole pool on backoff.
                    _log(f"[nvidia] Model deployment DEGRADED (key {key[-8:]}): {err_msg}")
                    return JSONResponse(
                        status_code=503,
                        content={"error": {
                            "message": f"NVIDIA model temporarily unavailable (degraded): {err_msg}",
                            "type": "upstream_error",
                        }},
                    )
                # 404 (model not authorized for this key) → advance to the next
                # key with no delay; a different key may have access.
                if resp.status_code < 500 and attempt < key_limit + attempts - 1:
                    continue
                # 5xx → escalating key cooldown (504 fast-breaks, no cycling);
                # then transient retry with backoff.
                if not _report_nvidia_status(key, resp.status_code):
                    return JSONResponse(
                        status_code=resp.status_code,
                        content={"error": {"message": err_msg, "type": "upstream_error"}},
                    )
                if attempt < key_limit + attempts - 1:
                    await _backoff(attempt)
                    continue
                return JSONResponse(
                    status_code=resp.status_code,
                    content={"error": {"message": err_msg, "type": "upstream_error"}},
                )

            nvidia_keys.report_success(key)
            consecutive_429 = 0
            usage = data.get("usage") or {}
            if isinstance(usage, dict) and ("prompt_tokens" in usage or "completion_tokens" in usage):
                _add_tokens(model,
                    usage.get("prompt_tokens") or 0,
                    usage.get("completion_tokens") or 0,
                    usage.get("prompt_cache_hit_tokens") or (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                    usage.get("prompt_cache_miss_tokens") or 0,
                )
            return data
        finally:
            try:
                await client.aclose()
            except Exception:
                pass

    last_err = f"NVIDIA error after retries (last key {last_key[-8:] if last_key else 'none'}: HTTP {last_status})"
    if last_error:
        last_err += f": {_exc_desc(last_error)}"
    return JSONResponse(
        status_code=last_status or 502,
        content={"error": {"message": last_err, "type": "upstream_error"}},
    )


async def nvidia_stream_with_retry(
    req_body: dict,
    user: str,
    model: str,
    max_retries: int = None,
):
    """Streaming NVIDIA NIM call with key rotation on consecutive 429s.

    Yields OpenAI SSE lines. A key that 429s twice consecutively rotates away
    (cooldown) and the request is retried with the next key.
    """
    attempts = max_retries if max_retries is not None else MAX_RETRIES
    last_error = None
    key_limit = len(nvidia_keys.keys) or 1
    used_keys_in_request = 0
    # Consecutive pool-wide 429s → fail fast when the whole pool is saturated.
    consecutive_429 = 0
    # Track stream throughput so we don't replay once bytes reached the client.
    streamed_any = False

    attempt = 0
    while attempt <= attempts + key_limit * 2:
        key = nvidia_keys.select()
        if key is None:
            yield _openai_stream_error(
                "all NVIDIA keys are in rate-limit cooldown",
                "rate_limit_error", "rate_limit_exceeded",
            )
            return
        used_keys_in_request += 1
        client = _nvidia_client(streaming=True)
        try:
            upstream_request = client.build_request(
                "POST", "/v1/chat/completions", json=req_body,
                headers=_nvidia_headers(key),
            )
            resp = await client.send(upstream_request, stream=True)
        except Exception as e:
            last_error = e
            _log(f"[nvidia] Stream request failed (attempt {attempt}, key {key[-8:]}): {_exc_desc(e)}")
            await client.aclose()
            if attempt < attempts + key_limit * 2:
                attempt += 1
                await _backoff(attempt)
                continue
            yield _openai_stream_error(f"Stream request failed: {_exc_desc(e)}", "upstream_error", "transport_error")
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
                err_msg = f"Rate limit ({_exc_desc(e)})"
            rotated = nvidia_keys.report_rate_limit(key)
            _log(f"[nvidia] Stream 429 on key {key[-8:]} (attempt {attempt}): {err_msg} rotated={rotated}")
            await resp.aclose()
            await client.aclose()
            consecutive_429 += 1
            # Whole pool saturated → fail fast with a clear 429 instead of
            # silently looping every key for minutes (looks like "thinking").
            if consecutive_429 >= NVIDIA_POOL_429_FAILFAST:
                _log(f"[nvidia] {consecutive_429} consecutive 429s → pool saturated; failing fast")
                yield _openai_stream_error(
                    f"NVIDIA pool rate-limited ({consecutive_429} consecutive 429s); try again in a moment",
                    "rate_limit_error", "rate_limit_exceeded",
                )
                return
            if rotated and used_keys_in_request < key_limit:
                attempt += 1
                await _backoff(attempt)
                continue
            if attempt < attempts + key_limit * 2:
                attempt += 1
                await _backoff(attempt)
                continue
            yield _openai_stream_error(err_msg, "rate_limit_error", "rate_limit_exceeded")
            return

        if resp.status_code >= 400:
            try:
                raw = await resp.aread()
            except Exception:
                raw = b""
            # Decode the CURRENT attempt's body first, then parse it: the
            # stale-body bug parsed a previous attempt's body_text here.
            body_text = raw.decode("utf-8", errors="replace")
            try:
                data = json.loads(body_text) if body_text else {}
            except Exception:
                _log(f"[nvidia] Stream error {resp.status_code} (key {key[-8:]}): unparsable body {raw[:400]!r}")
                data = {}
            if not isinstance(data, dict):
                _log(f"[nvidia] Stream error {resp.status_code} (key {key[-8:]}): non-object body {raw[:400]!r}")
                data = {}
            err_msg = (data.get("error") or {}).get("message") if isinstance(data.get("error"), dict) else None
            err_msg = err_msg or f"HTTP {resp.status_code}"
            consecutive_429 = 0  # a non-429 breaks the consecutive-429 run
            _log(f"[nvidia] Stream error {resp.status_code} (key {key[-8:]}): {err_msg}")
            if _is_degraded_error(data, body_text):
                # NVIDIA NIM deployment DEGRADED → every key returns the same
                # 400; fail fast rather than sweeping the whole pool on backoff.
                await resp.aclose()
                await client.aclose()
                _log(f"[nvidia] Model deployment DEGRADED (key {key[-8:]}): {err_msg}")
                yield _openai_stream_error(
                    f"NVIDIA model temporarily unavailable (degraded): {err_msg}",
                    "upstream_error", str(resp.status_code),
                )
                return
            if resp.status_code in (401, 403):
                nvidia_keys.report_rate_limit(key)
            await resp.aclose()
            await client.aclose()
            # No bytes have streamed yet -> safe to try another key. A 404 means
            # this key lacks the model; a 5xx may be transient. Sweep the pool.
            # 5xx records escalating key cooldown (504 fast-breaks, no cycling).
            if resp.status_code < 500 and attempt < attempts + key_limit:
                attempt += 1
                await _backoff(attempt)
                continue
            if resp.status_code >= 500:
                if not _report_nvidia_status(key, resp.status_code):
                    yield _openai_stream_error(err_msg, "upstream_error", str(resp.status_code))
                    return
                if attempt < attempts + key_limit * 2:
                    attempt += 1
                    await _backoff(attempt)
                    continue
            yield _openai_stream_error(err_msg, "upstream_error", str(resp.status_code))
            return

        # Success — stream it through
        nvidia_keys.report_success(key)
        consecutive_429 = 0
        stream_completed = False
        try:
            try:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        stream_completed = True
                        yield line + "\n\n"
                        streamed_any = True
                        break
                    try:
                        piece = json.loads(payload)
                    except Exception:
                        piece = {}
                    if not isinstance(piece, dict):
                        continue
                    # In-stream error (e.g. rate limit surfaced mid-body)
                    ev = _first_chunk_error(line)
                    if ev:
                        err_msg, is_rate_limit = ev
                        if is_rate_limit:
                            rotated = nvidia_keys.report_rate_limit(key)
                            _log(f"[nvidia] Stream body rate-limit on key {key[-8:]}: {err_msg} rotated={rotated}")
                            yield _openai_stream_error(err_msg, "rate_limit_error", "rate_limit_exceeded")
                        else:
                            yield _openai_stream_error(err_msg)
                        return
                    if '"usage"' in line:
                        u = piece.get("usage") or {}
                        if isinstance(u, dict) and (u.get("prompt_tokens") or u.get("completion_tokens")):
                            _add_tokens(model,
                                u.get("prompt_tokens") or 0,
                                u.get("completion_tokens") or 0,
                                u.get("prompt_cache_hit_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                                u.get("prompt_cache_miss_tokens") or 0,
                            )
                    yield line + "\n\n"
                    streamed_any = True
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.TransportError) as e:
                _log(f"[nvidia] Stream interrupted (key {key[-8:]}): {_exc_desc(e)}")
                last_error = e
                if not streamed_any and attempt < attempts + key_limit * 2:
                    attempt += 1
                    await _backoff(attempt)
                    continue
                yield _openai_stream_error(_transport_error_message(e), "upstream_error", "transport_error")
                return
        finally:
            try:
                await resp.aclose()
            except Exception:
                pass
            try:
                await client.aclose()
            except Exception:
                pass

        if not stream_completed:
            _log(f"[nvidia] Stream ended before [DONE] on key {key[-8:]}")
            if not streamed_any and attempt < attempts + key_limit * 2:
                attempt += 1
                await _backoff(attempt)
                continue
            yield _openai_stream_error("NVIDIA stream ended before completion", "upstream_error", "incomplete_stream")
            return
        return

    yield _openai_stream_error(
        f"connection error: NVIDIA stream failed after retries: {_exc_desc(last_error)}",
        "upstream_error", "transport_error",
    )


# ── AMD Radeon TokenFactory direct transport (mirrors NVIDIA) ──

# Fail-fast: if this many *consecutive* 429s happen without a single key
# succeeding, the whole pool is saturated with rate limits. Returning a clear
# 429 to the client beats silently looping every key with backoff for minutes
# (which looks like "thinking" to the client). A "consecutive" run is broken by
# any success OR any non-429 (e.g. a socket error that still tries the pool).
AMD_POOL_429_FAILFAST = 12

# Dynamic discovery cache for GET /models on the TokenFactory endpoint.
# _AMD_DYNAMIC_IDS holds verbatim upstream IDs (no amd/ prefix); a 0.0 stamp
# means "never fetched successfully". Never wiped on empty/failure — the
# hardcoded amd_models() fallback always applies.
_AMD_DYNAMIC_IDS: list[str] = []
_AMD_DYNAMIC_AT: float = 0.0
_AMD_DYNAMIC_TTL = 3600.0


def amd_request_body(model, messages, stream, tools, tool_choice, max_tokens=None, max_completion_tokens=None, sampling=None):
    """Build a standard OpenAI-compatible body for the AMD TokenFactory endpoint.

    TokenFactory is OpenAI-compatible for /chat/completions; like NIM we only
    normalize roles and don't inject any reasoning_content.
    """
    body: dict = {"model": model, "messages": messages, "stream": bool(stream)}
    if tools:
        body["tools"] = tools
    if tool_choice:
        if tool_choice in ("auto", "none"):
            body["tool_choice"] = tool_choice
        else:
            body["tool_choice"] = "auto"
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        body["max_completion_tokens"] = max_completion_tokens
    if sampling:
        body.update(sampling)
    return body


def _amd_headers(key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }


def _amd_client(streaming: bool):
    from amd_pool import (
        AMD_BASE_URL,
        AMD_CONNECT_TIMEOUT,
        AMD_READ_TIMEOUT,
        AMD_STREAM_READ_TIMEOUT,
    )
    # trust_env=False: AMD calls go directly to developer.amd.com.cn and must
    # NOT be routed through the SOCKS proxy pool / system HTTP proxy.
    return httpx.AsyncClient(
        base_url=AMD_BASE_URL,
        timeout=httpx.Timeout(
            connect=AMD_CONNECT_TIMEOUT,
            read=AMD_STREAM_READ_TIMEOUT if streaming else AMD_READ_TIMEOUT,
            write=AMD_STREAM_READ_TIMEOUT if streaming else AMD_READ_TIMEOUT,
            pool=AMD_CONNECT_TIMEOUT,
        ),
        proxy=None,
        trust_env=False,
    )


def _amd_rate_limit_response(err_msg: str) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": err_msg + " (amd key rate limit)",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
            }
        },
    )


async def _amd_fetch_dynamic_ids() -> list[str]:
    """GET the TokenFactory /models list with a pooled key (dynamic discovery).

    Returns verbatim upstream IDs on success; on any failure (or empty pool)
    returns [] and the caller falls back to amd_models(). Never wipes the
    cached dynamic list on failure.
    """
    global _AMD_DYNAMIC_IDS, _AMD_DYNAMIC_AT
    now = time.time()
    if _AMD_DYNAMIC_IDS and now - _AMD_DYNAMIC_AT < _AMD_DYNAMIC_TTL:
        return _AMD_DYNAMIC_IDS
    if not amd_keys.ready:
        return _AMD_DYNAMIC_IDS or amd_models()
    key = amd_keys.select()
    if key is None:
        return _AMD_DYNAMIC_IDS or amd_models()
    client = _amd_client(streaming=False)
    try:
        try:
            resp = await client.get("/models", headers=_amd_headers(key))
        except Exception as e:
            _log(f"[amd] Dynamic /models fetch failed: {_exc_desc(e)}")
            return _AMD_DYNAMIC_IDS or amd_models()
        try:
            raw = await resp.aread()
        except Exception as e:
            _log(f"[amd] Dynamic /models read failed: {_exc_desc(e)}")
            return _AMD_DYNAMIC_IDS or amd_models()
        if resp.status_code >= 400:
            _log(f"[amd] Dynamic /models HTTP {resp.status_code}")
            if resp.status_code in (401, 403):
                amd_keys.report_rate_limit(key)
            else:
                _report_amd_status(key, resp.status_code)
            return _AMD_DYNAMIC_IDS or amd_models()
        try:
            payload = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
            return _AMD_DYNAMIC_IDS or amd_models()
        ids: list[str] = []
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            for entry in data:
                mid = entry.get("id") if isinstance(entry, dict) else None
                if mid:
                    ids.append(str(mid))
        if ids:
            amd_keys.report_success(key)
            _AMD_DYNAMIC_IDS = sorted(set(ids))
            _AMD_DYNAMIC_AT = now
            return _AMD_DYNAMIC_IDS
        return _AMD_DYNAMIC_IDS or amd_models()
    finally:
        try:
            await client.aclose()
        except Exception:
            pass


async def amd_request_with_retry(
    req_body: dict,
    user: str,
    model: str,
    max_retries: int = None,
):
    """Buffered AMD TokenFactory call with key rotation on consecutive 429s.

    Mirrors nvidia_request_with_retry. Returns the parsed OpenAI completion
    dict, a JSONResponse error, or None if the client gave up.
    """
    attempts = max_retries if max_retries is not None else MAX_RETRIES
    last_error = None
    last_key = None
    last_status = None
    last_body = ""
    key_limit = len(amd_keys.keys) or 1
    # Consecutive pool-wide 429s (not broken by a success or non-429). When this
    # reaches AMD_POOL_429_FAILFAST the whole pool is saturated → fail fast
    # instead of looping every key with backoff for minutes ("thinking").
    consecutive_429 = 0
    # Sweep the whole pool: each iteration picks the next usable key (the pool
    # advances internally), so a model that 404s for some keys (key lacks
    # access) but works for another eventually succeeds. Rate limits rotate and
    # cool down the offending key.
    for attempt in range(key_limit + attempts):
        key = amd_keys.select()
        if key is None:
            return _amd_rate_limit_response("all AMD keys are in rate-limit cooldown")
        client = _amd_client(streaming=False)
        headers = _amd_headers(key)
        try:
            try:
                resp = await client.post(
                    "/chat/completions", json=req_body, headers=headers
                )
            except Exception as e:
                last_error = e
                _log(f"[amd] Request failed (attempt {attempt}, key {key[-8:]}): {_exc_desc(e)}")
                if attempt < key_limit + attempts - 1:
                    await _backoff(attempt)
                    continue
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": f"AMD request failed: {_exc_desc(e)}", "type": "upstream_error"}},
                )

            body_bytes = await resp.aread()
            body_text = body_bytes.decode("utf-8", errors="replace")
            last_key = key
            last_status = resp.status_code
            last_body = body_text
            try:
                data = json.loads(body_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {}

            is_429 = resp.status_code == 429
            is_rate_limit = is_429 or "FreeUsageLimitError" in body_text or (
                "erate" in body_text and "429" in body_text
            )
            if is_rate_limit:
                err_msg = (data.get("error") or {}).get("message") or "Rate limit exceeded"
                rotated = amd_keys.report_rate_limit(key)
                _log(f"[amd] 429 on key {key[-8:]} (attempt {attempt}): {err_msg} rotated={rotated}")
                consecutive_429 += 1
                # Whole pool saturated with rate limits → fail fast with a clear
                # 429 rather than silently looping every key for minutes.
                if consecutive_429 >= AMD_POOL_429_FAILFAST:
                    _log(f"[amd] {consecutive_429} consecutive 429s → pool saturated; failing fast")
                    return _amd_rate_limit_response(
                        f"AMD pool rate-limited ({consecutive_429} consecutive 429s); try again in a moment"
                    )
                if attempt < key_limit + attempts - 1:
                    await _backoff(attempt)
                    continue
                return _amd_rate_limit_response(err_msg)

            if resp.status_code >= 400:
                consecutive_429 = 0  # a non-429 breaks the consecutive-429 run
                err_msg = (data.get("error") or {}).get("message") or f"HTTP {resp.status_code}"
                is_context = _is_context_limit_error(data, body_text)
                _log(f"[amd] Error {resp.status_code} (key {key[-8:]}): {err_msg} body={body_text[:200]!r}")
                if resp.status_code in (401, 403):
                    # Bad/revoked key — rotate away (not rate-limit but unusable).
                    amd_keys.report_rate_limit(key)
                if is_context:
                    # Context window exhaustion is per-model/request, not key.
                    return JSONResponse(
                        status_code=resp.status_code,
                        content={"error": {"message": err_msg, "type": "context_window_error"}},
                    )
                # 404 (model not authorized for this key) → advance to the next
                # key with no delay; a different key may have access.
                if resp.status_code < 500 and attempt < key_limit + attempts - 1:
                    continue
                # 5xx → escalating key cooldown (504 fast-breaks, no cycling);
                # then transient retry with backoff.
                if not _report_amd_status(key, resp.status_code):
                    return JSONResponse(
                        status_code=resp.status_code,
                        content={"error": {"message": err_msg, "type": "upstream_error"}},
                    )
                if attempt < key_limit + attempts - 1:
                    await _backoff(attempt)
                    continue
                return JSONResponse(
                    status_code=resp.status_code,
                    content={"error": {"message": err_msg, "type": "upstream_error"}},
                )

            amd_keys.report_success(key)
            consecutive_429 = 0
            usage = data.get("usage") or {}
            if isinstance(usage, dict) and ("prompt_tokens" in usage or "completion_tokens" in usage):
                _add_tokens(model,
                    usage.get("prompt_tokens") or 0,
                    usage.get("completion_tokens") or 0,
                    usage.get("prompt_cache_hit_tokens") or (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                    usage.get("prompt_cache_miss_tokens") or 0,
                )
            return data
        finally:
            try:
                await client.aclose()
            except Exception:
                pass

    last_err = f"AMD error after retries (last key {last_key[-8:] if last_key else 'none'}: HTTP {last_status})"
    if last_error:
        last_err += f": {_exc_desc(last_error)}"
    return JSONResponse(
        status_code=last_status or 502,
        content={"error": {"message": last_err, "type": "upstream_error"}},
    )


async def amd_stream_with_retry(
    req_body: dict,
    user: str,
    model: str,
    max_retries: int = None,
):
    """Streaming AMD TokenFactory call with key rotation on consecutive 429s.

    Mirrors nvidia_stream_with_retry. Yields OpenAI SSE lines. A key that 429s
    rotates away (cooldown) and the request is retried with the next key.
    """
    attempts = max_retries if max_retries is not None else MAX_RETRIES
    last_error = None
    key_limit = len(amd_keys.keys) or 1
    used_keys_in_request = 0
    # Consecutive pool-wide 429s → fail fast when the whole pool is saturated.
    consecutive_429 = 0
    # Track stream throughput so we don't replay once bytes reached the client.
    streamed_any = False

    attempt = 0
    while attempt <= attempts + key_limit * 2:
        key = amd_keys.select()
        if key is None:
            yield _openai_stream_error(
                "all AMD keys are in rate-limit cooldown",
                "rate_limit_error", "rate_limit_exceeded",
            )
            return
        used_keys_in_request += 1
        client = _amd_client(streaming=True)
        try:
            upstream_request = client.build_request(
                "POST", "/chat/completions", json=req_body,
                headers=_amd_headers(key),
            )
            resp = await client.send(upstream_request, stream=True)
        except Exception as e:
            last_error = e
            _log(f"[amd] Stream request failed (attempt {attempt}, key {key[-8:]}): {_exc_desc(e)}")
            await client.aclose()
            if attempt < attempts + key_limit * 2:
                attempt += 1
                await _backoff(attempt)
                continue
            yield _openai_stream_error(f"Stream request failed: {_exc_desc(e)}", "upstream_error", "transport_error")
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
                err_msg = f"Rate limit ({_exc_desc(e)})"
            rotated = amd_keys.report_rate_limit(key)
            _log(f"[amd] Stream 429 on key {key[-8:]} (attempt {attempt}): {err_msg} rotated={rotated}")
            await resp.aclose()
            await client.aclose()
            consecutive_429 += 1
            # Whole pool saturated → fail fast with a clear 429 instead of
            # silently looping every key for minutes (looks like "thinking").
            if consecutive_429 >= AMD_POOL_429_FAILFAST:
                _log(f"[amd] {consecutive_429} consecutive 429s → pool saturated; failing fast")
                yield _openai_stream_error(
                    f"AMD pool rate-limited ({consecutive_429} consecutive 429s); try again in a moment",
                    "rate_limit_error", "rate_limit_exceeded",
                )
                return
            if rotated and used_keys_in_request < key_limit:
                attempt += 1
                await _backoff(attempt)
                continue
            if attempt < attempts + key_limit * 2:
                attempt += 1
                await _backoff(attempt)
                continue
            yield _openai_stream_error(err_msg, "rate_limit_error", "rate_limit_exceeded")
            return

        if resp.status_code >= 400:
            try:
                raw = await resp.aread()
            except Exception:
                raw = b""
            try:
                data = json.loads(raw) if raw else {}
            except Exception:
                data = {}
            err_msg = (data.get("error") or {}).get("message") or f"HTTP {resp.status_code}"
            body_text = raw.decode("utf-8", errors="replace")
            consecutive_429 = 0  # a non-429 breaks the consecutive-429 run
            _log(f"[amd] Stream error {resp.status_code} (key {key[-8:]}): {err_msg}")
            if resp.status_code in (401, 403):
                amd_keys.report_rate_limit(key)
            await resp.aclose()
            await client.aclose()
            # No bytes have streamed yet -> safe to try another key. A 404 means
            # this key lacks the model; a 5xx may be transient. Sweep the pool.
            # 5xx records escalating key cooldown (504 fast-breaks, no cycling).
            if resp.status_code < 500 and attempt < attempts + key_limit:
                attempt += 1
                await _backoff(attempt)
                continue
            if resp.status_code >= 500:
                if not _report_amd_status(key, resp.status_code):
                    yield _openai_stream_error(err_msg, "upstream_error", str(resp.status_code))
                    return
                if attempt < attempts + key_limit * 2:
                    attempt += 1
                    await _backoff(attempt)
                    continue
            yield _openai_stream_error(err_msg, "upstream_error", str(resp.status_code))
            return

        # Success — stream it through
        amd_keys.report_success(key)
        consecutive_429 = 0
        stream_completed = False
        try:
            try:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        stream_completed = True
                        yield line + "\n\n"
                        streamed_any = True
                        break
                    try:
                        piece = json.loads(payload)
                    except Exception:
                        piece = {}
                    if not isinstance(piece, dict):
                        continue
                    # In-stream error (e.g. rate limit surfaced mid-body)
                    ev = _first_chunk_error(line)
                    if ev:
                        err_msg, is_rate_limit = ev
                        if is_rate_limit:
                            rotated = amd_keys.report_rate_limit(key)
                            _log(f"[amd] Stream body rate-limit on key {key[-8:]}: {err_msg} rotated={rotated}")
                            yield _openai_stream_error(err_msg, "rate_limit_error", "rate_limit_exceeded")
                        else:
                            yield _openai_stream_error(err_msg)
                        return
                    if '"usage"' in line:
                        u = piece.get("usage") or {}
                        if isinstance(u, dict) and (u.get("prompt_tokens") or u.get("completion_tokens")):
                            _add_tokens(model,
                                u.get("prompt_tokens") or 0,
                                u.get("completion_tokens") or 0,
                                u.get("prompt_cache_hit_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                                u.get("prompt_cache_miss_tokens") or 0,
                            )
                    yield line + "\n\n"
                    streamed_any = True
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.TransportError) as e:
                _log(f"[amd] Stream interrupted (key {key[-8:]}): {_exc_desc(e)}")
                last_error = e
                if not streamed_any and attempt < attempts + key_limit * 2:
                    attempt += 1
                    await _backoff(attempt)
                    continue
                yield _openai_stream_error(_transport_error_message(e), "upstream_error", "transport_error")
                return
        finally:
            try:
                await resp.aclose()
            except Exception:
                pass
            try:
                await client.aclose()
            except Exception:
                pass

        if not stream_completed:
            _log(f"[amd] Stream ended before [DONE] on key {key[-8:]}")
            if not streamed_any and attempt < attempts + key_limit * 2:
                attempt += 1
                await _backoff(attempt)
                continue
            yield _openai_stream_error("AMD stream ended before completion", "upstream_error", "incomplete_stream")
            return
        return

    yield _openai_stream_error(
        f"connection error: AMD stream failed after retries: {_exc_desc(last_error)}",
        "upstream_error", "transport_error",
    )


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


# ── Routes: Ollama format ─────────────────────────────────────────
# Minimal Ollama-compatible shim for clients that only speak Ollama
# (e.g. some VS Code extensions): GET /api/tags lists the same models as
# /v1/models, POST /api/chat and POST /api/generate fan out to the same
# Zen / NVIDIA-NIM upstream paths as /v1/chat/completions. Streaming uses
# Ollama NDJSON (one JSON object per line), not SSE. Structured output
# via `format` is NOT translated — the model returns plain text.

OLLAMA_VERSION = "0.11.4"
OLLAMA_MODIFIED_AT = "2026-01-01T00:00:00Z"
# Keep in sync with the nvidia/* entries exposed by list_models.
_OLLAMA_NVIDIA_ALIASES = ("kimi-k3", "deepseek-v4-pro-0813", "deepseek-v4-flash-0731", "deepseek-coder")


def _ollama_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())



async def _ollama_names() -> list[str]:
    """Model names served on the Ollama shim (same set as /v1/models)."""
    names = _served_models()
    seen = set(names)
    for alias, canonical in sorted(MODEL_ALIASES.items()):
        if canonical in seen and alias not in seen and not _is_blocked_model(canonical):
            names.append(alias)
    if nvidia_keys.ready:
        for alias in _OLLAMA_NVIDIA_ALIASES:
            nid = f"nvidia/{alias}"
            if nid not in seen:
                names.append(nid)
                seen.add(nid)
    if amd_keys.ready:
        try:
            amd_ids = await _amd_fetch_dynamic_ids()
        except Exception:
            amd_ids = amd_models()
        for mid in amd_ids:
            nid = f"amd/{mid}"
            if nid not in seen:
                names.append(nid)
                seen.add(nid)
    return names


def _ollama_details(name: str) -> dict:
    if is_nvidia_model(name):
        canon = nvidia_model_id(name)
    elif is_amd_model(name):
        canon = amd_model_id(name)
    else:
        canon = _normalize_model(name) or ""
    meta = _models_meta.get(canon) or {}
    modalities = meta.get("modalities") or {}
    caps = ["completion", "tools"]
    inp = modalities.get("input") if isinstance(modalities, dict) else None
    if isinstance(inp, list) and any(str(x).lower() == "image" for x in inp):
        caps.append("vision")
    return {
        "digest": "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest(),
        "details": {
            "parent_model": "",
            "format": "",
            "family": "opencode-free",
            "families": ["opencode-free"],
            "parameter_size": "",
            "quantization_level": "",
        },
        "capabilities": caps,
    }


async def ollama_tags(request: Request):
    new_request_id()
    if not _models_cache:
        await _fetch_free_models()
    models = []
    for n in await _ollama_names():
        d = _ollama_details(n)
        models.append({
            "name": n,
            "model": n,
            "modified_at": OLLAMA_MODIFIED_AT,
            "size": 0,
            "digest": d["digest"],
            "details": d["details"],
        })
    return {"models": models}


async def ollama_version(request: Request):
    new_request_id()
    return {"version": OLLAMA_VERSION}


async def ollama_ps(request: Request):
    new_request_id()
    return {"models": []}


async def ollama_root(request: Request):
    new_request_id()
    return PlainTextResponse("Ollama is running")


async def ollama_show(request: Request):
    new_request_id()
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    name = body.get("model") or body.get("name") or ""
    if not name:
        return JSONResponse(status_code=400, content={"error": "model name required"})
    if is_nvidia_model(name):
        if not nvidia_keys.ready:
            return JSONResponse(
                status_code=503,
                content={"error": "No NVIDIA API keys loaded (nvidia-api-keys.txt empty/missing)"},
            )
    elif is_amd_model(name):
        if not amd_keys.ready:
            return JSONResponse(
                status_code=503,
                content={"error": "No AMD API keys loaded (amd-api-keys.txt empty/missing)"},
            )
    elif not await _ensure_model_known(_normalize_model(name)):
        return JSONResponse(status_code=404, content={"error": f'model "{name}" not found'})
    d = _ollama_details(name)
    return {
        "modelfile": f"FROM {name}",
        "parameters": "",
        "template": "",
        "details": d["details"],
        "model_info": {"general.architecture": "opencode-free"},
        "capabilities": d["capabilities"],
    }


def _ollama_messages_to_openai(msgs, system=None) -> list[dict]:
    """Ollama chat messages -> OpenAI chat messages (images become parts)."""
    out: list[dict] = []
    if isinstance(system, str) and system:
        out.append({"role": "system", "content": system})
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "user"
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        content = m.get("content")
        if content is None:
            content = ""
        images = m.get("images") or []
        if images and isinstance(content, str):
            parts: list[dict] = []
            if content:
                parts.append({"type": "text", "text": content})
            for img in images:
                if not isinstance(img, str) or not img:
                    continue
                url = img if img.startswith("data:") else f"data:image/jpeg;base64,{img}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
            entry: dict = {"role": role, "content": parts or content}
        elif isinstance(content, str):
            entry = {"role": role, "content": content}
        else:
            entry = {"role": role, "content": json.dumps(content, ensure_ascii=False) if content else ""}
        if role == "assistant":
            oai_tcs = []
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                if not isinstance(fn, dict):
                    continue
                args = fn.get("arguments")
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                oai_tcs.append({
                    "id": tc.get("id") or oc_id("call"),
                    "type": "function",
                    "function": {"name": fn.get("name") or "", "arguments": args or ""},
                })
            if oai_tcs:
                entry["tool_calls"] = oai_tcs
            thinking = m.get("thinking")
            if isinstance(thinking, str) and thinking:
                entry["reasoning_content"] = thinking
        if role == "tool" and m.get("tool_call_id"):
            entry["tool_call_id"] = m["tool_call_id"]
        out.append(entry)
    return out


def _ollama_tools_to_openai(tools):
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
                    "parameters": t.get("parameters") or t.get("input_schema") or {},
                },
            })
    return out or None


def _ollama_sampling_from(options: dict, body: dict) -> dict:
    sampling = {}
    for k in ("temperature", "top_p", "stop"):
        v = options.get(k, body.get(k))
        if v is not None:
            sampling[k] = v
    return sampling


def _ollama_max_tokens(options: dict, body: dict):
    """Ollama num_predict (-1 = unlimited) -> max_tokens."""
    for k in ("num_predict", "max_tokens", "max_completion_tokens"):
        v = options.get(k, body.get(k))
        if v is None:
            continue
        try:
            v = int(v)
        except (TypeError, ValueError):
            continue
        if v < 0:
            return None
        if v == 0:
            continue
        return v
    return None


def _ollama_stream_flag(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)


def _ollama_think_enabled(think) -> bool:
    if think is None:
        return False
    if isinstance(think, bool):
        return think
    if isinstance(think, dict):
        return True
    if isinstance(think, str):
        return think.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(think)


def _ollama_effort_body(body: dict, options: dict, think) -> dict:
    """Merge reasoning-effort knobs for _resolve_muse_effort + Responses sampling."""
    effort: dict = {}
    for src in (body, options):
        if not isinstance(src, dict):
            continue
        if isinstance(src.get("reasoning"), dict) and "reasoning" not in effort:
            effort["reasoning"] = src["reasoning"]
        if src.get("reasoning_effort") and not effort.get("reasoning_effort"):
            effort["reasoning_effort"] = src["reasoning_effort"]
    if isinstance(think, dict) and think.get("effort"):
        effort["reasoning_effort"] = think["effort"]
    elif think is False:
        effort["reasoning_effort"] = "none"
    for k in ("temperature", "top_p"):
        if isinstance(options, dict) and options.get(k) is not None:
            effort[k] = options[k]
        elif body.get(k) is not None:
            effort[k] = body[k]
    return effort


def _ollama_status(resp: JSONResponse) -> int:
    try:
        return int(resp.status_code)
    except Exception:
        return 502


def _ollama_message(resp: JSONResponse) -> str:
    try:
        data = json.loads(resp.body.decode("utf-8"))
    except Exception:
        return "Upstream error"
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return err.get("message") or "Upstream error"
    if isinstance(err, str):
        return err
    return "Upstream error"


async def _ollama_upstream(request, user, model_raw, oai_messages, tools, stream, max_tokens, sampling, effort_body):
    """Shared Zen/NIM fan-out for the Ollama shim (mirrors chat_completions).

    Returns (kind, payload) where kind is:
      "sse"   -> payload is an async generator of OpenAI SSE strings
      "data"  -> payload is a buffered chat.completion dict
      "error" -> payload is (status_code, message) for an Ollama {"error": ...} body
    """
    if is_nvidia_model(model_raw):
        if not nvidia_keys.ready:
            return ("error", (503, "No NVIDIA API keys loaded (nvidia-api-keys.txt empty/missing)"))
        nim_model = nvidia_model_id(model_raw)
        norm_messages = _normalize_messages(oai_messages) or []
        req_body = nvidia_request_body(
            nim_model, norm_messages, stream, tools, None,
            max_tokens, None, sampling,
        )
        if stream:
            return ("sse", nvidia_stream_with_retry(req_body, user, nim_model))
        data = await nvidia_request_with_retry(req_body, user, nim_model)
        if isinstance(data, JSONResponse):
            return ("error", (_ollama_status(data), _ollama_message(data)))
        if data is None:
            return ("error", (502, "Client disconnected"))
        if not data.get("choices"):
            return ("error", (502, "Invalid upstream response"))
        return ("data", data)

    if is_amd_model(model_raw):
        if not amd_keys.ready:
            return ("error", (503, "No AMD API keys loaded (amd-api-keys.txt empty/missing)"))
        amd_model = amd_model_id(model_raw)
        norm_messages = _normalize_messages(oai_messages) or []
        req_body = amd_request_body(
            amd_model, norm_messages, stream, tools, None,
            max_tokens, None, sampling,
        )
        if stream:
            return ("sse", amd_stream_with_retry(req_body, user, amd_model))
        data = await amd_request_with_retry(req_body, user, amd_model)
        if isinstance(data, JSONResponse):
            return ("error", (_ollama_status(data), _ollama_message(data)))
        if data is None:
            return ("error", (502, "Client disconnected"))
        if not data.get("choices"):
            return ("error", (502, "Invalid upstream response"))
        return ("data", data)

    model = _normalize_model(model_raw)
    if not await _ensure_model_known(model):
        return ("error", (400, f"Unknown model: {model}. Available: {', '.join(_served_models())}"))
    session_id = get_session(user, oai_messages)
    up_messages = _prepare_upstream_messages(session_id, _normalize_messages(oai_messages))
    req_body, headers = zen_request(model, up_messages, stream, tools, None, session_id, max_tokens, None, sampling)

    if _is_muse_spark(model) or _model_wire_api(model) == "openai-responses":
        resp_body = _zen_responses_body(
            model, up_messages, tools,
            effort=_resolve_muse_effort(model_raw, effort_body),
            max_tokens=max_tokens,
            sampling=_responses_sampling_from(effort_body or {}),
        )
        gen = _zen_responses_stream_with_retry(request, resp_body, headers, user, oai_messages or [], session_id, model)
        if stream:
            return ("sse", gen)
        data = await _aggregate_upstream_completion(
            request, resp_body, headers, user, oai_messages or [], session_id, model, gen=gen,
        )
        if isinstance(data, JSONResponse):
            return ("error", (_ollama_status(data), _ollama_message(data)))
        if data is None:
            return ("error", (502, "Client disconnected"))
        return ("data", data)

    if stream:
        if _needs_buffered_fallback(model, tools, stream):
            _log(f"[zen] buffered fallback: stream:true -> stream:false upstream (model={model})")
            buffered_body = dict(req_body)
            buffered_body["stream"] = False
            data = await _zen_request_with_retry(request, buffered_body, headers, user, oai_messages or [], session_id, model)
            if isinstance(data, JSONResponse):
                return ("error", (_ollama_status(data), _ollama_message(data)))
            if data is None:
                return ("error", (502, "Client disconnected"))
            return ("data", data)
        return ("sse", _zen_stream_with_retry(request, req_body, headers, user, oai_messages or [], session_id, model))

    data = await _zen_request_with_retry(request, req_body, headers, user, oai_messages or [], session_id, model)
    if isinstance(data, JSONResponse):
        return ("error", (_ollama_status(data), _ollama_message(data)))
    if data is None:
        return ("error", (502, "Client disconnected"))
    if not data.get("choices"):
        return ("error", (502, "Invalid upstream response"))
    return ("data", data)


def _ollama_done_reason(finish) -> str:
    return "length" if finish == "length" else "stop"


def _ollama_tool_args(slot_args: str) -> dict:
    try:
        parsed = json.loads(slot_args) if slot_args else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _ollama_ndjson_stream(sse_gen, model_name: str, mode: str, include_thinking: bool):
    """Fold an OpenAI SSE stream (Zen/NIM/muse) into Ollama NDJSON chunks."""
    created_at = _ollama_now()
    content_parts: list[str] = []
    think_parts: list[str] = []
    tool_acc: dict[int, dict] = {}
    finish_reason = None
    usage = None
    async for raw in sse_gen:
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
            if isinstance(piece.get("error"), dict):
                yield json.dumps({"error": piece["error"].get("message") or "Upstream error"}, ensure_ascii=False) + "\n"
                return
            choices = piece.get("choices")
            if not choices:
                if isinstance(piece.get("usage"), dict):
                    usage = piece["usage"]
                continue
            ch = choices[0] if isinstance(choices, list) else {}
            if not isinstance(ch, dict):
                continue
            delta = ch.get("delta") or {}
            if not isinstance(delta, dict):
                delta = {}
            text = delta.get("content")
            if text:
                content_parts.append(text)
                if mode == "generate":
                    yield json.dumps({"model": model_name, "created_at": created_at, "response": text, "done": False}, ensure_ascii=False) + "\n"
                else:
                    yield json.dumps({"model": model_name, "created_at": created_at, "message": {"role": "assistant", "content": text}, "done": False}, ensure_ascii=False) + "\n"
            thinking = delta.get("reasoning_content")
            if thinking:
                think_parts.append(thinking)
                if include_thinking:
                    if mode == "generate":
                        yield json.dumps({"model": model_name, "created_at": created_at, "response": "", "thinking": thinking, "done": False}, ensure_ascii=False) + "\n"
                    else:
                        yield json.dumps({"model": model_name, "created_at": created_at, "message": {"role": "assistant", "content": "", "thinking": thinking}, "done": False}, ensure_ascii=False) + "\n"
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                idx = tc.get("index", 0)
                slot = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if isinstance(fn, dict):
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
            u = piece.get("usage")
            if isinstance(u, dict) and (u.get("prompt_tokens") or u.get("completion_tokens")):
                usage = u

    full = "".join(content_parts)
    done_reason = _ollama_done_reason(finish_reason)
    if mode == "generate":
        final: dict = {"model": model_name, "created_at": created_at, "response": full, "done": True, "done_reason": done_reason}
        if include_thinking and think_parts:
            final["thinking"] = "".join(think_parts)
    else:
        message: dict = {"role": "assistant", "content": full}
        if include_thinking and think_parts:
            message["thinking"] = "".join(think_parts)
        if tool_acc:
            message["tool_calls"] = [
                {"function": {"name": tool_acc[idx]["name"], "arguments": _ollama_tool_args(tool_acc[idx]["arguments"])}}
                for idx in sorted(tool_acc)
            ]
        final = {"model": model_name, "created_at": created_at, "message": message, "done": True, "done_reason": done_reason}
    if isinstance(usage, dict):
        final["prompt_eval_count"] = usage.get("prompt_tokens") or 0
        final["eval_count"] = usage.get("completion_tokens") or 0
    yield json.dumps(final, ensure_ascii=False) + "\n"


def _ollama_chat_object(data: dict, model_name: str, include_thinking: bool) -> dict:
    choice = (data.get("choices") or [{}])[0] or {}
    msg = choice.get("message") or {}
    message: dict = {"role": "assistant", "content": msg.get("content") or ""}
    if include_thinking and msg.get("reasoning_content"):
        message["thinking"] = msg["reasoning_content"]
    tcs = []
    for tc in msg.get("tool_calls") or []:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        tcs.append({"function": {"name": fn.get("name") or "", "arguments": _ollama_tool_args(fn.get("arguments") or "")}})
    if tcs:
        message["tool_calls"] = tcs
    out: dict = {
        "model": model_name,
        "created_at": _ollama_now(),
        "message": message,
        "done": True,
        "done_reason": _ollama_done_reason(choice.get("finish_reason")),
    }
    usage = data.get("usage") or {}
    if usage:
        out["prompt_eval_count"] = usage.get("prompt_tokens") or 0
        out["eval_count"] = usage.get("completion_tokens") or 0
    return out


def _ollama_generate_object(data: dict, model_name: str, include_thinking: bool) -> dict:
    obj = _ollama_chat_object(data, model_name, include_thinking)
    out: dict = {
        "model": model_name,
        "created_at": obj["created_at"],
        "response": (obj.get("message") or {}).get("content") or "",
        "done": True,
        "done_reason": obj["done_reason"],
    }
    if "thinking" in (obj.get("message") or {}):
        out["thinking"] = obj["message"]["thinking"]
    if "prompt_eval_count" in obj:
        out["prompt_eval_count"] = obj["prompt_eval_count"]
        out["eval_count"] = obj.get("eval_count", 0)
    return out


async def _ollama_json_body(request):
    """Return (body, None) or (None, error_response)."""
    try:
        body = await request.json()
    except Exception:
        return None, JSONResponse(status_code=400, content={"error": "invalid JSON"})
    if not isinstance(body, dict):
        return None, JSONResponse(status_code=400, content={"error": "invalid JSON"})
    return body, None


async def ollama_chat(request: Request):
    new_request_id()
    user = auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Invalid API key"})
    body, err = await _ollama_json_body(request)
    if err is not None:
        return err
    model_raw = body.get("model") or ""
    if not model_raw:
        return JSONResponse(status_code=400, content={"error": "model required"})
    oai_messages = _ollama_messages_to_openai(body.get("messages") or [], body.get("system"))
    tools = _ollama_tools_to_openai(body.get("tools"))
    options = body.get("options") or {}
    if not isinstance(options, dict):
        options = {}
    sampling = _ollama_sampling_from(options, body)
    max_tokens = _ollama_max_tokens(options, body)
    stream = _ollama_stream_flag(body.get("stream"), True)
    think = body.get("think", False)
    include_thinking = _ollama_think_enabled(think)
    effort_body = _ollama_effort_body(body, options, think)

    kind, payload = await _ollama_upstream(request, user, model_raw, oai_messages, tools, stream, max_tokens, sampling, effort_body)
    if kind == "error":
        status, msg = payload
        return JSONResponse(status_code=status, content={"error": msg})
    if kind == "data":
        return JSONResponse(_ollama_chat_object(payload, str(model_raw), include_thinking))
    return StreamingResponse(
        _ollama_ndjson_stream(payload, str(model_raw), "chat", include_thinking),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


async def ollama_generate(request: Request):
    new_request_id()
    user = auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Invalid API key"})
    body, err = await _ollama_json_body(request)
    if err is not None:
        return err
    model_raw = body.get("model") or body.get("name") or ""
    if not model_raw:
        return JSONResponse(status_code=400, content={"error": "model required"})
    prompt = body.get("prompt") or ""
    suffix = body.get("suffix") or ""
    gen_msgs = []
    if body.get("system"):
        gen_msgs.append({"role": "system", "content": str(body["system"])})
    user_msg: dict = {"role": "user", "content": str(prompt) + str(suffix)}
    if isinstance(body.get("images"), list) and body["images"]:
        user_msg["images"] = body["images"]
    gen_msgs.append(user_msg)
    oai_messages = _ollama_messages_to_openai(gen_msgs, None)
    options = body.get("options") or {}
    if not isinstance(options, dict):
        options = {}
    sampling = _ollama_sampling_from(options, body)
    max_tokens = _ollama_max_tokens(options, body)
    stream = _ollama_stream_flag(body.get("stream"), True)
    think = body.get("think", False)
    include_thinking = _ollama_think_enabled(think)
    effort_body = _ollama_effort_body(body, options, think)

    kind, payload = await _ollama_upstream(request, user, model_raw, oai_messages, None, stream, max_tokens, sampling, effort_body)
    if kind == "error":
        status, msg = payload
        return JSONResponse(status_code=status, content={"error": msg})
    if kind == "data":
        return JSONResponse(_ollama_generate_object(payload, str(model_raw), include_thinking))
    return StreamingResponse(
        _ollama_ndjson_stream(payload, str(model_raw), "generate", include_thinking),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


# ── Routes: OpenAI format ─────────────────────────────────────────

async def list_models(request: Request):
    new_request_id()
    if not _models_cache:
        await _fetch_free_models()
    _sanitize_models_cache()
    data = []
    try:
        _zen_free_ids: set[str] | None = set(_served_models())
    except Exception:
        _zen_free_ids = None
    for m in _served_models():
        if not _stealth_admitted(m, _zen_free_ids):
            continue
        entry = {"id": m, "object": "model", "created": 1779000000, "owned_by": "opencode-free"}
        meta = _models_meta.get(m)
        if meta:
            if meta.get("limit"):
                entry["limits"] = meta["limit"]
            else:
                entry["limits"] = dict(_DEFAULT_LIMIT)
            if meta.get("modalities"):
                entry["modalities"] = meta["modalities"]
            else:
                entry["modalities"] = dict(_DEFAULT_MODALITIES)
        else:
            # Conservative defaults when models.dev was unreachable.
            entry["limits"] = dict(_DEFAULT_LIMIT)
            entry["modalities"] = dict(_DEFAULT_MODALITIES)
        data.append(entry)
    # Slash-free picker aliases for OpenCode Zen models (no slash/colon):
    # expose each alias whose canonical id is actually served (blocked
    # canonicals never appear in _served_models, so their aliases stay hidden).
    _seen_ids = {e["id"] for e in data}
    for alias, canonical in sorted(MODEL_ALIASES.items()):
        if canonical in _seen_ids and alias not in _seen_ids and not _is_blocked_model(canonical):
            data.append({"id": alias, "object": "model", "created": 1779000000, "owned_by": "opencode-free"})
    # NVIDIA NIM models exposed behind the nvidia/ (and nvimin/) prefix.
    if nvidia_keys.ready:
        for alias in ("kimi-k3", "deepseek-v4-pro-0813", "deepseek-v4-flash-0731", "deepseek-coder"):
            data.append({"id": f"nvidia/{alias}", "object": "model", "created": 1779000000, "owned_by": "opencode-free"})
    # AMD Radeon TokenFactory models exposed behind the amd/ prefix.
    # Dynamic discovery via GET /models when the pool is ready; hardcoded
    # amd_models() fallback on failure — never wiped on empty.
    if amd_keys.ready:
        try:
            amd_ids = await _amd_fetch_dynamic_ids()
        except Exception:
            amd_ids = amd_models()
        for mid in amd_ids:
            data.append({"id": f"amd/{mid}", "object": "model", "created": 1779000000, "owned_by": "opencode-free"})
    return {"object": "list", "data": data, "models_checked_at": _models_checked_at}


async def chat_completions(request: Request):
    new_request_id()
    user = auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": {"message": "Invalid API key"}})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "invalid JSON body", "type": "invalid_request_error"}},
        )
    model = body.get("model")
    messages = body.get("messages")
    stream = body.get("stream")
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")

    if not isinstance(messages, list) or not messages:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "messages is required", "type": "invalid_request_error"}},
        )

    # NVIDIA NIM direct route: models addressed with nvidia/ or nvimin/ prefix
    if is_nvidia_model(model):
        if not nvidia_keys.ready:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "No NVIDIA API keys loaded (nvidia-api-keys.txt empty/missing)", "type": "upstream_error"}},
            )
        nim_model = nvidia_model_id(model)
        norm_messages = _normalize_messages(messages) or []
        req_body = nvidia_request_body(
            nim_model, norm_messages, stream, tools, tool_choice,
            body.get("max_tokens"), body.get("max_completion_tokens"), _nvidia_sampling_from(body),
        )
        if stream:
            return StreamingResponse(
                nvidia_stream_with_retry(req_body, user, nim_model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
            )
        data = await nvidia_request_with_retry(req_body, user, nim_model)
        if isinstance(data, JSONResponse):
            return data
        if data is None:
            return JSONResponse(status_code=502, content={"error": {"message": "Client disconnected", "type": "upstream_error"}})
        if not data.get("choices"):
            return JSONResponse(status_code=502, content={"error": {"message": "Invalid upstream response", "type": "upstream_error"}})
        return data

    # AMD Radeon TokenFactory direct route: models addressed with amd/ or radeon/ prefix
    if is_amd_model(model):
        if not amd_keys.ready:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "No AMD API keys loaded (amd-api-keys.txt empty/missing)", "type": "upstream_error"}},
            )
        amd_model = amd_model_id(model)
        norm_messages = _normalize_messages(messages) or []
        req_body = amd_request_body(
            amd_model, norm_messages, stream, tools, tool_choice,
            body.get("max_tokens"), body.get("max_completion_tokens"), _amd_sampling_from(body),
        )
        if stream:
            return StreamingResponse(
                amd_stream_with_retry(req_body, user, amd_model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
            )
        data = await amd_request_with_retry(req_body, user, amd_model)
        if isinstance(data, JSONResponse):
            return data
        if data is None:
            return JSONResponse(status_code=502, content={"error": {"message": "Client disconnected", "type": "upstream_error"}})
        if not data.get("choices"):
            return JSONResponse(status_code=502, content={"error": {"message": "Invalid upstream response", "type": "upstream_error"}})
        return data

    model = _normalize_model(model)
    if not await _ensure_model_known(model):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Unknown model: {model}. Available: {', '.join(_served_models())}"}},
        )

    session_id = get_session(user, messages)

    up_messages = _prepare_upstream_messages(session_id, _normalize_messages(messages))
    req_body, headers = zen_request(model, up_messages, stream, tools, tool_choice, session_id, body.get("max_tokens"), body.get("max_completion_tokens"), _sampling_from(body))

    if _is_muse_spark(model) or _model_wire_api(model) == "openai-responses":
        # Responses-routed models (muse-spark substring or models.dev
        # provider.npm == "@ai-sdk/openai" meta): chat/completions returns a
        # deterministic 500. Translate to Responses and translate back.
        resp_body = _zen_responses_body(
            model, up_messages, tools,
            effort=_resolve_muse_effort(body.get("model"), body),
            max_tokens=body.get("max_tokens"),
            max_completion_tokens=body.get("max_completion_tokens"),
            sampling=_responses_sampling_from(body),
        )
        if stream:
            return StreamingResponse(
                _zen_responses_stream_with_retry(request, resp_body, headers, user, messages or [], session_id, model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
            )
        data = await _aggregate_upstream_completion(
            request, resp_body, headers, user, messages or [], session_id, model,
            gen=_zen_responses_stream_with_retry(request, resp_body, headers, user, messages or [], session_id, model),
        )
        if isinstance(data, JSONResponse):
            return data
        if data is None:
            return JSONResponse(status_code=502, content={"error": {"message": "Client disconnected", "type": "upstream_error"}})
        return data

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
    new_request_id()
    user = auth(request)
    if not user:
        return JSONResponse(
            status_code=401,
            content={"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key"}},
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "invalid JSON body", "type": "invalid_request_error"}},
        )
    model = body.get("model")
    stream = body.get("stream")

    # NVIDIA NIM direct route (Anthropic format): convert to OpenAI, call NIM,
    # convert the completion back to Anthropic. Streams are bridged through the
    # buffered path (same guarantee as the Zen muse-spark fallback).
    if is_nvidia_model(model):
        if not nvidia_keys.ready:
            return JSONResponse(
                status_code=503,
                content={"type": "error", "error": {"type": "upstream_error", "message": "No NVIDIA API keys loaded"}},
            )
        nim_model = nvidia_model_id(model)
        oai_nv_messages, nv_tools = anthropic_to_openai(body)
        oai_nv_messages = _normalize_messages(oai_nv_messages) or []
        input_tokens = len(json.dumps(oai_nv_messages)) // 4
        nv_req_body = nvidia_request_body(
            nim_model, oai_nv_messages, False, nv_tools, None,
            body.get("max_tokens"), body.get("max_completion_tokens"), _nvidia_sampling_from(body),
        )
        nv_data = await nvidia_request_with_retry(nv_req_body, user, nim_model)
        if isinstance(nv_data, JSONResponse):
            return nv_data
        if nv_data is None:
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "upstream_error", "message": "Client disconnected"}},
            )
        if not nv_data.get("choices"):
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "upstream_error", "message": "Invalid upstream response"}},
            )
        anth = openai_to_anthropic(nv_data, nim_model, input_tokens)
        if stream:
            async def _nv_anthropic_sse():
                # Stream the already-complete buffered Anthropic conversion.
                start_payload = json.dumps({"type": "message_start", "message": anth})
                yield f"event: message_start\ndata: {start_payload}\n\n"
                content = anth.get("content", [])
                for idx, block in enumerate(content):
                    cbs_payload = json.dumps({"type": "content_block_start", "index": idx, "content_block": block})
                    yield f"event: content_block_start\ndata: {cbs_payload}\n\n"
                    if block.get("type") == "text":
                        delta_payload = json.dumps({"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": block.get("text", "")}})
                        yield f"event: content_block_delta\ndata: {delta_payload}\n\n"
                    cbstop_payload = json.dumps({"type": "content_block_stop", "index": idx})
                    yield f"event: content_block_stop\ndata: {cbstop_payload}\n\n"
                stop_payload = json.dumps({"type": "message_stop"})
                yield f"event: message_stop\ndata: {stop_payload}\n\n"
            return StreamingResponse(
                _nv_anthropic_sse(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
            )
        return anth

    # AMD TokenFactory direct route (Anthropic format): convert to OpenAI, call
    # TokenFactory, convert the completion back to Anthropic. Streams are
    # bridged through the buffered path (same guarantee as the NVIDIA route).
    if is_amd_model(model):
        if not amd_keys.ready:
            return JSONResponse(
                status_code=503,
                content={"type": "error", "error": {"type": "upstream_error", "message": "No AMD API keys loaded"}},
            )
        amd_model = amd_model_id(model)
        oai_amd_messages, amd_tools = anthropic_to_openai(body)
        oai_amd_messages = _normalize_messages(oai_amd_messages) or []
        input_tokens = len(json.dumps(oai_amd_messages)) // 4
        amd_req_body = amd_request_body(
            amd_model, oai_amd_messages, False, amd_tools, None,
            body.get("max_tokens"), body.get("max_completion_tokens"), _amd_sampling_from(body),
        )
        amd_data = await amd_request_with_retry(amd_req_body, user, amd_model)
        if isinstance(amd_data, JSONResponse):
            return amd_data
        if amd_data is None:
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "upstream_error", "message": "Client disconnected"}},
            )
        if not amd_data.get("choices"):
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "upstream_error", "message": "Invalid upstream response"}},
            )
        anth = openai_to_anthropic(amd_data, amd_model, input_tokens)
        if stream:
            async def _amd_anthropic_sse():
                # Stream the already-complete buffered Anthropic conversion.
                start_payload = json.dumps({"type": "message_start", "message": anth})
                yield f"event: message_start\ndata: {start_payload}\n\n"
                content = anth.get("content", [])
                for idx, block in enumerate(content):
                    cbs_payload = json.dumps({"type": "content_block_start", "index": idx, "content_block": block})
                    yield f"event: content_block_start\ndata: {cbs_payload}\n\n"
                    if block.get("type") == "text":
                        delta_payload = json.dumps({"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": block.get("text", "")}})
                        yield f"event: content_block_delta\ndata: {delta_payload}\n\n"
                    cbstop_payload = json.dumps({"type": "content_block_stop", "index": idx})
                    yield f"event: content_block_stop\ndata: {cbstop_payload}\n\n"
                stop_payload = json.dumps({"type": "message_stop"})
                yield f"event: message_stop\ndata: {stop_payload}\n\n"
            return StreamingResponse(
                _amd_anthropic_sse(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
            )
        return anth

    model = _normalize_model(model)

    if not await _ensure_model_known(model):
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": f"Unknown model: {model}. Available: {', '.join(_served_models())}"}},
        )

    oai_messages, tools = anthropic_to_openai(body)
    session_id = get_session(user, oai_messages)
    input_tokens = len(json.dumps(oai_messages)) // 4

    up_messages = _prepare_upstream_messages(session_id, _normalize_messages(oai_messages))

    if _is_muse_spark(model) or _model_wire_api(model) == "openai-responses":
        # Muse Spark via /zen/v1/responses (see chat route); aggregate and
        # reframe as Anthropic SSE (or a plain message when not streaming).
        _, headers = zen_request(model, up_messages, True, tools, None, session_id, body.get("max_tokens"), body.get("max_completion_tokens"), _sampling_from(body))
        resp_body = _zen_responses_body(
            model, up_messages, tools,
            effort=_resolve_muse_effort(body.get("model"), body),
            max_tokens=body.get("max_tokens"),
            max_completion_tokens=body.get("max_completion_tokens"),
            sampling=_responses_sampling_from(body),
        )
        data = await _aggregate_upstream_completion(
            request, resp_body, headers, user, oai_messages, session_id, model,
            gen=_zen_responses_stream_with_retry(request, resp_body, headers, user, oai_messages, session_id, model),
        )
        if isinstance(data, JSONResponse):
            return data
        if data is None:
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "upstream_error", "message": "Client disconnected"}},
            )
        if stream:
            async def _muse_anthropic_sse():
                for chunk in _buffered_to_anthropic_sse(data, model, input_tokens):
                    yield chunk
            return StreamingResponse(
                _muse_anthropic_sse(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
            )
        return openai_to_anthropic(data, model, input_tokens)

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
            if not isinstance(data, dict):
                continue
            # Surface upstream error chunks explicitly: a {"error": {...}}
            # chunk carries no choices/usage and must NOT be swallowed into
            # a terminal status=incomplete. Emit a failed response instead.
            err_obj = data.get("error")
            if isinstance(err_obj, dict):
                err_msg = err_obj.get("message") or "upstream error"
                _log(f"[zen] Responses upstream error chunk: {_redact_for_log(str(err_msg))[:300]}")
                yield _responses_sse("response.failed", {
                    "type": "response.failed",
                    "response": _response_obj(resp_id, model, "failed", [], _chat_usage_to_responses(usage)),
                })
                yield _responses_sse("response.completed", {
                    "type": "response.completed",
                    "response": _response_obj(resp_id, model, "failed", [], _chat_usage_to_responses(usage)),
                })
                return
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
    new_request_id()
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
                content={"error": {"message": f"Unknown model: {zen_model}. Available: {', '.join(_served_models())}"}},
            )
        messages = _normalize_messages(messages)
        session_id = get_session(user, messages)
        up_messages = _prepare_upstream_messages(session_id, messages)
        req_body, headers = zen_request(zen_model, up_messages, stream, tools, tool_choice, session_id, body.get("max_tokens"), body.get("max_completion_tokens"), _sampling_from(body))

        if _is_muse_spark(zen_model) or _model_wire_api(zen_model) == "openai-responses":
            # Client speaks Responses already, but the Zen upstream for muse is
            # /zen/v1/responses, not chat/completions (which 500s). Stream there
            # natively, then map the events to client Responses events as usual.
            resp_body = _zen_responses_body(
                zen_model, up_messages, tools,
                effort=_resolve_muse_effort(model, body),
                max_tokens=body.get("max_tokens"),
                max_completion_tokens=body.get("max_completion_tokens"),
                sampling=_responses_sampling_from(body),
            )
            gen = _zen_responses_stream_with_retry(request, resp_body, headers, user, messages, session_id, zen_model)
            if stream:
                return StreamingResponse(
                    _stream_as_responses(gen, model, session_id),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
                )
            data = await _aggregate_upstream_completion(
                request, resp_body, headers, user, messages, session_id, zen_model,
                gen=_zen_responses_stream_with_retry(request, resp_body, headers, user, messages, session_id, zen_model),
            )
            if isinstance(data, JSONResponse):
                return data
            if data is None:
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": "Client disconnected", "type": "upstream_error"}},
                )
            return _chat_to_responses(data, model)

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
    Blocklisted ids are excluded, so the default/fast/smart fallbacks never
    land on an always-failing model.
    """
    served = _served_models() or _models_cache
    if not served:
        return _normalize_model(model)
    m = model.lower().replace("-", "").replace("_", "")
    if m == "opencodedefault":
        return served[0]
    if m == "opencodefast":
        return next((x for x in served if "flash" in x or "lightning" in x or "mini" in x), served[0])
    if m == "opencodesmart":
        return next((x for x in served if "ultra" in x or "pro" in x or "smart" in x), served[0])
    return _normalize_model(model)


async def health(request: Request):
    pool_state = None
    if PROXY_POOL_ENABLED:
        pool_state = proxy_pool.get_pool_state() if proxy_pool.ready else "loading"
    return {
        "status": "ok",
        "version": f"v{PROXY_VERSION}",
        "models": len(_models_cache),
        "models_checked_at": _models_checked_at,
        "socks5": STATIC_PROXY,
        "proxy_pool": PROXY_POOL_ENABLED,
        "proxy_port_filter": sorted(ALLOWED_PROXY_PORTS) if PROXY_PORT_FILTER_ENABLED else None,
        "pool_state": pool_state,
        "pool_size": len(proxy_pool.hot) if PROXY_POOL_ENABLED else None,
        "tokens": dict(_tokens),
        "nvidia_keys": len(nvidia_keys.keys) if nvidia_keys.ready else 0,
        "nvidia_models": [f"nvidia/{m}" for m in nvidia_models()] if nvidia_keys.ready else [],
        "amd_keys": len(amd_keys.keys) if amd_keys.ready else 0,
        "amd_models": [f"amd/{m}" for m in amd_models()] if amd_keys.ready else [],
        "endpoints": ["/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/models", "/api/tags", "/api/show", "/api/chat", "/api/generate"],
    }


app.add_route("/v1/models", _json(list_models), methods=["GET"])
app.add_route("/v1/chat/completions", _json(chat_completions), methods=["POST"])
app.add_route("/v1/messages", _json(messages), methods=["POST"])
app.add_route("/v1/responses", _json(handle_responses), methods=["POST"])
app.add_route("/health", _json(health), methods=["GET"])
app.add_route("/api/tags", _json(ollama_tags), methods=["GET"])
app.add_route("/api/version", _json(ollama_version), methods=["GET"])
app.add_route("/api/ps", _json(ollama_ps), methods=["GET"])
app.add_route("/api/show", _json(ollama_show), methods=["POST"])
app.add_route("/api/chat", ollama_chat, methods=["POST"])
app.add_route("/api/generate", ollama_generate, methods=["POST"])
app.add_route("/", ollama_root, methods=["GET"])


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
    print("  Ollama:    GET  /api/tags, POST /api/chat|generate")
    print("  Models:    GET  /v1/models")
    print("  Health:    GET  /health")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
