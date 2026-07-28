import asyncio
import json
import random
import time
from pathlib import Path

import httpx

log_file = None
_data_dir = Path(__file__).parent / "data"


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
CACHE_TTL = 60
VERIFY_TIMEOUT = 4
HOT_TARGET = 15
HOT_MIN = 5
HOT_TTL = 60
RATE_LIMIT_TTL = 30 * 60
BLACKLIST_TTL = 120 * 60
MAX_RETRIES = 10
EXHAUSTED_FORCE_REFRESH_INTERVAL = 30
REQUEST_CONNECT_TIMEOUT = 5
REQUEST_READ_TIMEOUT = 120

RATE_LIMIT_FILE = _data_dir / "proxy-rate-limits.json"
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
        self._no_proxy_client: httpx.AsyncClient | None = None
        self._verify_sem = asyncio.Semaphore(25)
        self._load_rate_limits()
        self._try_load_cache()

    def _load_rate_limits(self):
        data = _load_json(RATE_LIMIT_FILE, {})
        now = time.time()
        changed = False
        for addr, expires_at in list(data.items()):
            if expires_at < now:
                del data[addr]
                changed = True
        if changed:
            _save_json(RATE_LIMIT_FILE, data)
        self.rate_limits = {k: float(v) for k, v in data.items()}
        if self.rate_limits:
            _log(f"Loaded {len(self.rate_limits)} rate-limited proxies from cache")

    def _save_rate_limits(self):
        _save_json(RATE_LIMIT_FILE, self.rate_limits)

    def _try_load_cache(self) -> bool:
        data = _load_json(CACHE_FILE, {})
        if isinstance(data, dict):
            saved_at = float(data.get("saved_at", 0))
            candidates = data.get("candidates")
            age = time.time() - saved_at
            if isinstance(candidates, list) and candidates and age <= CACHE_TTL:
                self.candidates = candidates
                self.last_refresh = saved_at
                _log(f"Loaded {len(candidates)} cached candidates ({age:.0f}s old)")
                return True
            if candidates:
                _log(f"Ignored stale proxy cache ({age:.0f}s old)")
        return False

    def _save_cache(self):
        _save_json(CACHE_FILE, {
            "saved_at": time.time(),
            "candidates": self.candidates,
        })

    def _is_bad(self, addr: str) -> bool:
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
        for lines in results:
            if isinstance(lines, list):
                random.shuffle(lines)
                for addr in lines[:MAX_PER_SOURCE]:
                    if addr not in seen:
                        seen.add(addr)
                        proxies.append({"address": addr, "protocol": "socks5"})

        random.shuffle(proxies)
        if len(proxies) > MAX_POOL_SIZE:
            proxies = proxies[:MAX_POOL_SIZE]

        if proxies:
            self.candidates = proxies
            self.last_refresh = time.time()
            self._save_cache()
            _log(f"Got {len(self.candidates)} unique candidates from all sources")
        else:
            _log("No candidates loaded; keeping existing candidates")

        self._trigger_refill()

    async def _verify(self, proxy: dict) -> bool:
        addr = proxy["address"]
        host, port_str = addr.split(":", 1)
        port = int(port_str)

        async def _socks5_handshake():
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port), timeout=VERIFY_TIMEOUT
                )
                try:
                    writer.write(bytes([0x05, 0x01, 0x00]))
                    await writer.drain()
                    resp = await asyncio.wait_for(
                        reader.read(2), timeout=VERIFY_TIMEOUT
                    )
                    if len(resp) != 2 or resp[0] != 0x05 or resp[1] != 0x00:
                        return False

                    # Require the proxy to connect to the actual Zen host.
                    target = b"opencode.ai"
                    request = (
                        bytes([0x05, 0x01, 0x00, 0x03, len(target)])
                        + target
                        + (443).to_bytes(2, "big")
                    )
                    writer.write(request)
                    await writer.drain()
                    reply = await asyncio.wait_for(
                        reader.readexactly(4), timeout=VERIFY_TIMEOUT
                    )
                    return reply[0] == 0x05 and reply[1] == 0x00
                finally:
                    writer.close()
                    await writer.wait_closed()
            except Exception:
                return False

        async with self._verify_sem:
            try:
                return await asyncio.wait_for(
                    _socks5_handshake(), timeout=VERIFY_TIMEOUT + 1
                )
            except (asyncio.TimeoutError, Exception):
                return False

    async def select(self) -> bool:
        """Get the next usable proxy. Checks hot buffer first, then verifies one on-the-fly."""
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
        batch = []
        for p in self.candidates:
            addr = p["address"]
            if self._is_bad(addr) or addr in self.verifying:
                continue
            self.verifying.add(addr)
            batch.append(p)
            if len(batch) >= 10:
                break

        if batch:
            _log(f"Verifying {len(batch)} candidates on-the-fly...")
            results = await asyncio.gather(
                *[self._verify(p) for p in batch], return_exceptions=True
            )
            for p, ok in zip(batch, results):
                addr = p["address"]
                self.verifying.discard(addr)
                if ok is True:
                    self.current = p
                    _log(f"Selected after on-the-fly verify: {addr}")
                    self._trigger_refill()
                    return True
                self.blacklist[addr] = time.time() + BLACKLIST_TTL

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
            batch = []
            for p in self.candidates:
                addr = p["address"]
                if addr in hot_addresses or addr in self.verifying or self._is_bad(addr):
                    continue
                self.verifying.add(addr)
                batch.append(p)
                if len(batch) >= 20:
                    break

            if not batch:
                break

            _log(f"Refilling hot buffer: verifying {len(batch)} candidates (hot={len(self.hot)}/{HOT_TARGET})")
            results = await asyncio.gather(
                *[self._verify(p) for p in batch], return_exceptions=True
            )
            checked += len(batch)
            for p, ok in zip(batch, results):
                addr = p["address"]
                self.verifying.discard(addr)
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
        self._save_rate_limits()
        _log(f"Rate-limited {target} for {RATE_LIMIT_TTL // 60}m")
        self.current = None

    def report_failure(self, addr: str | None = None):
        target = addr or (self.current and self.current["address"])
        if not target:
            return
        self.blacklist[target] = time.time() + BLACKLIST_TTL
        _log(f"Blacklisted {target} for {BLACKLIST_TTL // 60}m")
        self.current = None

    def get_pool_state(self) -> str:
        now = time.time()
        rl = sum(1 for a in self.rate_limits if self.rate_limits.get(a, 0) > now)
        bl = sum(1 for a in self.blacklist if self.blacklist.get(a, 0) > now)
        return f"candidates={len(self.candidates)} hot={len(self.hot)} verifying={len(self.verifying)} blacklisted={bl} rate_limited={rl}"

    async def force_refresh(self):
        now = time.time()
        if now - self.last_force_refresh < EXHAUSTED_FORCE_REFRESH_INTERVAL:
            return
        self.last_force_refresh = now
        self.blacklist.clear()
        cleared = 0
        for addr in list(self.rate_limits.keys()):
            if any(c["address"] == addr for c in self.candidates):
                del self.rate_limits[addr]
                cleared += 1
        if cleared:
            self._save_rate_limits()
        self.hot.clear()
        _log(f"Force-refreshing (cleared {cleared} rate-limits, blacklist reset)")
        self.last_refresh = 0
        if self._source_task is None or self._source_task.done():
            self._source_task = asyncio.create_task(self._refresh_candidates())
        await self._source_task

    def get_client(self, proxy_url: str | None = None) -> httpx.AsyncClient:
        timeout = httpx.Timeout(
            connect=REQUEST_CONNECT_TIMEOUT,
            read=REQUEST_READ_TIMEOUT,
            write=REQUEST_READ_TIMEOUT,
            pool=REQUEST_CONNECT_TIMEOUT,
        )
        if not proxy_url:
            if self._no_proxy_client is None:
                self._no_proxy_client = httpx.AsyncClient(
                    base_url="https://opencode.ai",
                    timeout=timeout,
                )
            return self._no_proxy_client
        if proxy_url not in self._clients:
            self._clients[proxy_url] = httpx.AsyncClient(
                base_url="https://opencode.ai",
                timeout=timeout,
                proxy=proxy_url,
                verify=False,
            )
        return self._clients[proxy_url]

    async def close(self):
        for c in self._clients.values():
            await c.aclose()
        if self._no_proxy_client:
            await self._no_proxy_client.aclose()
        self._clients.clear()
        self._no_proxy_client = None

    @property
    def ready(self) -> bool:
        return self._ready


pool = ProxyPool()
