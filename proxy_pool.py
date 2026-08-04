import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

log_file = None
_base_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
_data_dir = _base_dir / "data"


def _log(*a):
    msg = f"[{time.strftime('%H:%M:%S')}] [proxy-pool] " + " ".join(str(x) for x in a)
    print(f"\x1b[33m{msg}\x1b[0m", flush=True)
    global log_file
    if log_file is None:
        log_file = _data_dir / "proxy-pool.log"
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except OSError:
        pass


SOCKS5_SOURCES = [
    "https://cdn.jsdelivr.net/gh/proxyscrape/free-proxy-list@main/proxies/protocols/socks5/data.txt",
    "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/socks5/data.txt",
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks5.txt",
    "https://databay.com/free-proxy-list/socks5.txt",
    "https://raw.githubusercontent.com/mohammedcha/ProxRipper/main/full_proxies/socks5.txt",
    "https://raw.githubusercontent.com/iplocate/free-proxy-list/main/protocols/socks5.txt",
    "https://raw.githubusercontent.com/r00tee/Proxy-List/main/Socks5.txt",
    "https://raw.githubusercontent.com/gproxynet/free-proxy-list/main/socks5.txt",
]

MAX_PER_SOURCE = 1000
MAX_POOL_SIZE = 5000
POLL_INTERVAL = 30 * 60
# Candidates stay valid across restarts for a full poll cycle; a 60s TTL made
# the on-disk cache useless since refreshes only run every 30 minutes.
CACHE_TTL = POLL_INTERVAL
VERIFY_TIMEOUT = 6
VERIFY_BATCH_SIZE = 20
HOT_TARGET = 15
HOT_MIN = 5
# Hot entries must outlive a refill cycle (~45s) plus typical idle gaps,
# otherwise every request after a pause pays on-the-fly verification.
HOT_TTL = 600
RATE_LIMIT_TTL = 30 * 60
BLACKLIST_TTL = 120 * 60
# Retries after the first attempt: 3 -> 4 total attempts, worst-case backoff
# 1+2+4 = 7s. Each retry rotates to a fresh verified proxy, so more attempts
# mostly add latency (up to ~94s at 10) for marginal recovery.
MAX_RETRIES = 3
EXHAUSTED_FORCE_REFRESH_INTERVAL = 30
REQUEST_CONNECT_TIMEOUT = 5
REQUEST_READ_TIMEOUT = 120
# Streaming requests may legitimately spend several minutes in the model's
# thinking phase between SSE events. Keep the shorter timeout for buffered
# requests, but give streaming clients a longer idle window.
STREAM_READ_TIMEOUT = 300

# Public SOCKS lists contain many stale or mislabelled HTTP endpoints. Keep the
# default candidate set focused on the two SOCKS ports that are most common in
# these lists; set OPENCODE_PROXY_PORT_FILTER=false to allow every port.
PROXY_PORT_FILTER_ENABLED = os.environ.get(
    "OPENCODE_PROXY_PORT_FILTER", "true"
).lower() not in ("0", "false", "no", "off")
ALLOWED_PROXY_PORTS = frozenset((4145, 1080))

CACHE_FILE = _data_dir / "proxy-pool-cache.json"


def _is_socks5_addr(addr: str) -> bool:
    parts = addr.strip().split(":")
    if len(parts) != 2:
        return False
    host, port = parts
    if not port.isdigit() or not (1 <= int(port) <= 65535):
        return False
    if not host or host.startswith(".") or host.endswith("."):
        return False
    return True


def _load_json(path: Path, default=None):
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return default or {}


def _save_json(path: Path, data):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(path)
    except OSError:
        pass


class ProxyPool:
    def __init__(self):
        self.candidates: list[dict] = []
        self.hot: list[dict] = []
        self.blacklist: dict[str, float] = {}
        self.rate_limits: dict[str, float] = {}
        self.current: dict | None = None
        self.verifying: set[str] = set()
        self.last_refresh = 0.0
        self.last_force_refresh = 0.0
        self._ready = False
        self._refill_task: asyncio.Task | None = None
        self._source_task: asyncio.Task | None = None
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._stream_clients: dict[str, httpx.AsyncClient] = {}
        self._no_proxy_client: httpx.AsyncClient | None = None
        self._stream_no_proxy_client: httpx.AsyncClient | None = None
        self._verify_sem = asyncio.Semaphore(25)
        self._select_lock = asyncio.Lock()
        self._try_load_cache()

    def _try_load_cache(self) -> bool:
        data = _load_json(CACHE_FILE, {})
        if isinstance(data, dict):
            saved_at = float(data.get("saved_at", 0))
            candidates = data.get("candidates")
            age = time.time() - saved_at
            has_sources = (
                isinstance(candidates, list)
                and all(
                    isinstance(p, dict) and p.get("address") and p.get("source")
                    for p in candidates
                )
            )
            if isinstance(candidates, list) and candidates and has_sources and age <= CACHE_TTL:
                candidates = self._filter_allowed_ports(candidates)
                if candidates:
                    self.candidates = candidates
                    self.last_refresh = saved_at
                    _log(f"Loaded {len(candidates)} cached candidates ({age:.0f}s old)")
                    return True
                _log("Ignored cached candidates: none use an allowed proxy port")
            if candidates and isinstance(candidates, list) and not has_sources:
                _log("Ignored proxy cache without source provenance")
            if candidates:
                _log(f"Ignored stale proxy cache ({age:.0f}s old)")
        return False

    def _save_cache(self):
        _save_json(CACHE_FILE, {
            "saved_at": time.time(),
            "candidates": self.candidates,
        })

    @staticmethod
    def _proxy_port(addr: str) -> int | None:
        try:
            return int(addr.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            return None

    @classmethod
    def _port_allowed(cls, addr: str) -> bool:
        return (
            not PROXY_PORT_FILTER_ENABLED
            or cls._proxy_port(addr) in ALLOWED_PROXY_PORTS
        )

    @classmethod
    def _filter_allowed_ports(cls, candidates: list[dict]) -> list[dict]:
        if not PROXY_PORT_FILTER_ENABLED:
            return candidates
        filtered = [
            p for p in candidates
            if cls._port_allowed(p.get("address", ""))
        ]
        rejected = len(candidates) - len(filtered)
        if rejected:
            _log(
                f"Ignored {rejected} candidates on unsupported ports "
                f"(allowed: {', '.join(map(str, sorted(ALLOWED_PROXY_PORTS)))})"
            )
        return filtered

    def _is_bad(self, addr: str) -> bool:
        if not self._port_allowed(addr):
            return True
        now = time.time()
        bl = self.blacklist.get(addr)
        if bl and bl > now:
            return True
        rl = self.rate_limits.get(addr)
        if rl and rl > now:
            return True
        return False

    async def load(self):
        """Fetch sources in background. Returns immediately — pool is ready."""
        self._ready = True
        if self.candidates:
            self._trigger_refill()
        if self._source_task is None or self._source_task.done():
            self._source_task = asyncio.create_task(self._refresh_candidates())

    async def _refresh_candidates(self):
        """Fetch sources, replace candidates, trigger refill."""
        now = time.time()
        if self.candidates and now - self.last_refresh < POLL_INTERVAL:
            self._trigger_refill()
            return

        _log("Fetching SOCKS5 proxy sources...")

        async def fetch(url: str) -> list[str]:
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(15.0), follow_redirects=True
                ) as c:
                    r = await c.get(url)
                    if r.status_code == 200:
                        lines = []
                        for line in r.text.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            if "://" in line:
                                line = line.split("://", 1)[1]
                            if _is_socks5_addr(line):
                                lines.append(line)
                        if lines:
                            _log(f"  {url}: {len(lines)} proxies")
                        return lines
                    _log(f"  {url}: HTTP {r.status_code}")
            except Exception as e:
                _log(f"  {url}: {e.__class__.__name__}")
            return []

        tasks = [fetch(src) for src in SOCKS5_SOURCES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen = set()
        proxies = []
        for source_url, lines in zip(SOCKS5_SOURCES, results):
            if isinstance(lines, list):
                random.shuffle(lines)
                for addr in lines[:MAX_PER_SOURCE]:
                    if addr not in seen:
                        seen.add(addr)
                        proxies.append({
                            "address": addr,
                            "protocol": "socks5",
                            "source": source_url,
                        })

        random.shuffle(proxies)
        if len(proxies) > MAX_POOL_SIZE:
            proxies = proxies[:MAX_POOL_SIZE]

        proxies = self._filter_allowed_ports(proxies)
        if proxies:
            self.candidates = proxies
            self.last_refresh = time.time()
            self._save_cache()
            _log(f"Got {len(self.candidates)} unique candidates from all sources")
        else:
            _log("No candidates loaded; keeping existing candidates")

        self._trigger_refill()

    async def _verify(self, proxy: dict) -> bool:
        """Full-path check: SOCKS5 tunnel + real HTTPS request through the proxy.

        A bare SOCKS5 CONNECT passes proxies that then blackhole, throttle, or
        break TLS on real traffic; requiring a 2xx from the actual Zen endpoint
        keeps those out of the hot buffer. The httpx phase timeouts do NOT
        bound the whole check (SOCKS handshake / DNS / TLS can stall beyond
        them), so wait_for is the hard deadline: one stalled proxy must never
        block an entire verify batch.
        """
        addr = proxy["address"]
        url = f"socks5://{addr}"

        async def _check() -> bool:
            timeout = httpx.Timeout(VERIFY_TIMEOUT)
            async with httpx.AsyncClient(proxy=url, verify=False, timeout=timeout) as c:
                r = await c.get(
                    "https://opencode.ai/zen/v1/models",
                    headers={
                        "User-Agent": "opencode/1.15.0",
                        "x-opencode-client": "cli",
                    },
                )
                return 200 <= r.status_code < 300

        async with self._verify_sem:
            try:
                return await asyncio.wait_for(_check(), timeout=VERIFY_TIMEOUT + 2)
            except Exception:
                return False

    def _take_verification_batch(
        self,
        limit: int,
        excluded: set[str] | None = None,
    ) -> list[dict]:
        """Take a round-robin batch across source lists.

        Candidates are shuffled within each source at refresh time, while the
        source round-robin prevents a large/noisy list from monopolizing a
        verification batch.
        """
        excluded = excluded or set()
        by_source: dict[str, list[dict]] = {}
        for p in self.candidates:
            addr = p.get("address")
            if (
                not addr
                or addr in excluded
                or addr in self.verifying
                or self._is_bad(addr)
            ):
                continue
            source = p.get("source") or "legacy"
            by_source.setdefault(source, []).append(p)

        source_order = list(by_source)
        random.shuffle(source_order)
        positions = {source: 0 for source in source_order}
        batch: list[dict] = []
        while len(batch) < limit and source_order:
            progressed = False
            for source in source_order:
                position = positions[source]
                candidates = by_source[source]
                if position >= len(candidates):
                    continue
                p = candidates[position]
                positions[source] = position + 1
                self.verifying.add(p["address"])
                batch.append(p)
                progressed = True
                if len(batch) >= limit:
                    break
            if not progressed:
                break
        return batch

    async def select(self) -> bool:
        """Get the next usable proxy. Checks hot buffer first, then verifies one on-the-fly.

        Serialized with a lock so concurrent requests share one verification
        pass instead of each blacklisting its own failed batch.
        """
        async with self._select_lock:
            return await self._select_unlocked()

    async def _select_unlocked(self) -> bool:
        """select() body; callers must hold _select_lock."""
        if self.current:
            if not self._is_bad(self.current["address"]):
                return True
            self.current = None

        if not self.candidates and self._source_task:
            await self._source_task

        # 1. Hot buffer — verified and ready
        while self.hot:
            p = self.hot.pop(0)
            verified_at = float(p.get("verified_at", 0))
            if time.time() - verified_at > HOT_TTL:
                continue
            if not self._is_bad(p["address"]):
                self.current = p
                _log(f"Selected from hot: {p['address']}")
                self._trigger_refill()
                return True

        # 2. On-the-fly: verify a batch of candidates in parallel
        batch = self._take_verification_batch(VERIFY_BATCH_SIZE)

        if batch:
            _log(
                f"Verifying {len(batch)} candidates on-the-fly across "
                f"{len({p.get('source', 'legacy') for p in batch})} sources..."
            )
            pending = {asyncio.create_task(self._verify(p)): p for p in batch}
            try:
                while pending:
                    done, _ = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for fut in done:
                        p = pending.pop(fut)
                        if fut.cancelled():
                            continue
                        addr = p["address"]
                        if fut.result() is True:
                            # First verified proxy wins; don't wait for the rest
                            # (up to ~8s of latency saved per cold request).
                            for f in pending:
                                f.cancel()
                            self.current = p
                            _log(f"Selected after on-the-fly verify: {addr}")
                            self._trigger_refill()
                            return True
                        self.blacklist[addr] = time.time() + BLACKLIST_TTL
                # All failed
                _log(f"No usable proxy ({self.get_pool_state()})")
                return False
            finally:
                # Cancel stragglers (success path or caller cancellation) and
                # always release the in-flight markers, even if the request
                # handler is cancelled mid-verify.
                if pending:
                    for f in pending:
                        f.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                for p in batch:
                    self.verifying.discard(p["address"])

        # 3. Nothing usable
        _log(f"No usable proxy ({self.get_pool_state()})")
        return False

    def _trigger_refill(self):
        """Ensure background refill is running to keep hot buffer full."""
        if len(self.hot) >= HOT_MIN:
            return
        if self._refill_task and not self._refill_task.done():
            return
        self._refill_task = asyncio.ensure_future(self._refill())

    async def _refill(self):
        """Background: keep HOT_TARGET verified proxies in hot buffer."""
        checked = 0
        added = 0
        max_checks = 200

        while len(self.hot) < HOT_TARGET and checked < max_checks:
            hot_addresses = {p["address"] for p in self.hot}
            if self.current:
                hot_addresses.add(self.current["address"])
            batch = self._take_verification_batch(VERIFY_BATCH_SIZE, hot_addresses)

            if not batch:
                break

            _log(
                f"Refilling hot buffer: verifying {len(batch)} candidates across "
                f"{len({p.get('source', 'legacy') for p in batch})} sources "
                f"(hot={len(self.hot)}/{HOT_TARGET})"
            )
            try:
                results = await asyncio.gather(
                    *[self._verify(p) for p in batch], return_exceptions=True
                )
            finally:
                # Always release in-flight markers, even on task cancellation.
                for p in batch:
                    self.verifying.discard(p["address"])
            checked += len(batch)
            for p, ok in zip(batch, results):
                addr = p["address"]
                if (
                    ok is True
                    and len(self.hot) < HOT_TARGET
                    and addr not in {x["address"] for x in self.hot}
                ):
                    self.hot.append({**p, "verified_at": time.time()})
                    added += 1
                elif ok is not True:
                    self.blacklist[addr] = time.time() + BLACKLIST_TTL

        _log(f"Refill done: +{added}, checked={checked}, hot={len(self.hot)}/{HOT_TARGET}")

    def report_ratelimit(self, addr: str | None = None):
        target = addr or (self.current and self.current["address"])
        if not target:
            return
        self.rate_limits[target] = time.time() + RATE_LIMIT_TTL
        self._evict_client(target)
        _log(f"Rate-limited {target} for {RATE_LIMIT_TTL // 60}m; rotating")
        if self.current and self.current["address"] == target:
            self.current = None

    def report_failure(self, addr: str | None = None):
        target = addr or (self.current and self.current["address"])
        if not target:
            return
        self.blacklist[target] = time.time() + BLACKLIST_TTL
        self._evict_client(target)
        _log(f"Blacklisted {target} for {BLACKLIST_TTL // 60}m")
        if self.current and self.current["address"] == target:
            self.current = None

    def _evict_client(self, addr: str):
        """Drop the pooled AsyncClient for a proxy that just failed, releasing
        its connection pool (and any half-dead keep-alive sockets) and keeping
        self._clients from growing without bound as proxies rotate."""
        for clients in (self._clients, self._stream_clients):
            c = clients.pop(f"socks5://{addr}", None)
            if c is not None:
                try:
                    asyncio.ensure_future(c.aclose())
                except Exception:
                    pass

    def get_pool_state(self) -> str:
        now = time.time()
        bl = sum(1 for a in self.blacklist if self.blacklist.get(a, 0) > now)
        rl = sum(1 for a in self.rate_limits if self.rate_limits.get(a, 0) > now)
        return f"candidates={len(self.candidates)} hot={len(self.hot)} verifying={len(self.verifying)} blacklisted={bl} rate_limited={rl}"

    async def force_refresh(self):
        now = time.time()
        if now - self.last_force_refresh < EXHAUSTED_FORCE_REFRESH_INTERVAL:
            return
        self.last_force_refresh = now
        # Never wipe live blacklist knowledge: clearing it would re-test known-
        # dead proxies on every request. Only let entries expire on their TTLs.
        expired = [a for a, exp in self.blacklist.items() if exp <= now]
        for a in expired:
            del self.blacklist[a]
        expired_rate_limits = [a for a, exp in self.rate_limits.items() if exp <= now]
        for a in expired_rate_limits:
            del self.rate_limits[a]
        _log(f"Force-refreshing (expired {len(expired)} blacklist and "
             f"{len(expired_rate_limits)} rate-limit entries, hot={len(self.hot)}, "
             f"candidates={len(self.candidates)})")
        if self._source_task is None or self._source_task.done():
            self._source_task = asyncio.create_task(self._refresh_candidates())
        await self._source_task

    def get_client(
        self,
        proxy_url: str | None = None,
        streaming: bool = False,
    ) -> httpx.AsyncClient:
        read_timeout = STREAM_READ_TIMEOUT if streaming else REQUEST_READ_TIMEOUT
        timeout = httpx.Timeout(
            connect=REQUEST_CONNECT_TIMEOUT,
            read=read_timeout,
            write=read_timeout,
            pool=REQUEST_CONNECT_TIMEOUT,
        )
        clients = self._stream_clients if streaming else self._clients
        if not proxy_url:
            client_attr = "_stream_no_proxy_client" if streaming else "_no_proxy_client"
            client = getattr(self, client_attr)
            if client is None:
                client = httpx.AsyncClient(
                    base_url="https://opencode.ai",
                    timeout=timeout,
                )
                setattr(self, client_attr, client)
            return client
        if proxy_url not in clients:
            clients[proxy_url] = httpx.AsyncClient(
                base_url="https://opencode.ai",
                timeout=timeout,
                proxy=proxy_url,
                verify=False,
            )
        return clients[proxy_url]

    async def close(self):
        for c in [*self._clients.values(), *self._stream_clients.values()]:
            await c.aclose()
        for client in (self._no_proxy_client, self._stream_no_proxy_client):
            if client:
                await client.aclose()
        self._clients.clear()
        self._stream_clients.clear()
        self._no_proxy_client = None
        self._stream_no_proxy_client = None

    @property
    def ready(self) -> bool:
        return self._ready


pool = ProxyPool()
