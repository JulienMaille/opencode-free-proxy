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


_LOG_MAX_BYTES = 5 * 1024 * 1024  # rotate proxy-pool.log past 5 MB
_LOG_KEEP_BYTES = 1 * 1024 * 1024  # ...keeping the last 1 MB


def _append_log(path: Path, msg: str):
    """Append one line, rotating the file down to its tail past the cap."""
    try:
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
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except OSError:
        pass


def _log(*a):
    msg = f"[{time.strftime('%H:%M:%S')}] [proxy-pool] " + " ".join(str(x) for x in a)
    print(f"\x1b[33m{msg}\x1b[0m", flush=True)
    global log_file
    if log_file is None:
        log_file = _data_dir / "proxy-pool.log"
    _append_log(log_file, msg)


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
# Retries after the first attempt: 5 -> 6 total attempts. Each retry rotates
# to a fresh verified proxy; hard transport errors (connect timeout / proxy
# unreachable) blacklist the exit immediately instead of consuming a second
# attempt on it.
MAX_RETRIES = 5
# Max simultaneous streams per proxy. A single sticky proxy handles normal
# sequential requests, but bursts of concurrent requests (omp subagent spawns)
# spill across a few verified proxies so one proxy isn't overloaded / 429'd.
MAX_PER_PROXY = 2
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
        self.transport_failures: dict[str, int] = {}
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
        self._inflight: dict[str, int] = {}
        self._inflight_at: dict[str, float] = {}
        # Per-address stability stats: {addr: {ok, fail, ema}}. Persisted with
        # the candidate cache so proxies that survived yesterday get picked
        # first today; a proxy list refresh never resets this history.
        self.stats: dict[str, dict] = {}
        self._stats_dirty = False
        self._last_stats_save = 0.0
        self._try_load_cache()

    # ── Stability stats ──────────────────────────────────────────────

    _STATS_SAVE_INTERVAL = 60.0  # disk writes at most once/min

    def _stat(self, addr: str) -> dict:
        return self.stats.setdefault(addr, {"ok": 0, "fail": 0, "ema": 0.0})

    def _mark_ok(self, addr: str):
        s = self._stat(addr)
        s["ok"] += 1
        self._stats_dirty = True
        self._maybe_save_stats()

    def _mark_fail(self, addr: str):
        s = self._stat(addr)
        s["fail"] += 1
        self._stats_dirty = True
        self._maybe_save_stats()

    def _record_latency(self, addr: str, seconds: float):
        s = self._stat(addr)
        s["ema"] = seconds if not s["ema"] else 0.7 * s["ema"] + 0.3 * seconds
        self._stats_dirty = True

    def _quality(self, addr: str) -> float:
        """Lower is better: expected-seconds-per-success.

        Combines measured latency (EWMA) with Laplace-smoothed reliability.
        Unknowns sit mid-pack: a brand-new verified proxy gets a fair chance
        but proven fast+reliable ones win the slot.
        """
        s = self.stats.get(addr)
        if not s:
            return VERIFY_TIMEOUT * 2
        reliability = (s["ok"] + 1) / (s["ok"] + s["fail"] + 2)
        ema = s["ema"] or VERIFY_TIMEOUT
        return ema / max(reliability, 0.05)

    def _maybe_save_stats(self):
        now = time.time()
        if self._stats_dirty and now - self._last_stats_save >= self._STATS_SAVE_INTERVAL:
            self._last_stats_save = now
            self._stats_dirty = False
            self._save_cache()

    def _try_load_cache(self) -> bool:
        data = _load_json(CACHE_FILE, {})
        if isinstance(data, dict):
            saved_at = float(data.get("saved_at", 0))
            saved_stats = data.get("stats")
            if isinstance(saved_stats, dict):
                self.stats = {
                    a: {"ok": int(s.get("ok", 0)), "fail": int(s.get("fail", 0)), "ema": float(s.get("ema", 0.0))}
                    for a, s in saved_stats.items()
                    if isinstance(a, str) and isinstance(s, dict)
                }
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
                    _log(f"Loaded {len(candidates)} cached candidates ({age:.0f}s old, {len(self.stats)} stability stats)")
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
            "stats": self.stats,
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
            started = time.monotonic()
            async with httpx.AsyncClient(proxy=url, verify=False, timeout=timeout) as c:
                r = await c.get(
                    "https://opencode.ai/zen/v1/models",
                    headers={
                        "User-Agent": "opencode/1.15.0",
                        "x-opencode-client": "cli",
                    },
                )
                if 200 <= r.status_code < 300:
                    self._record_latency(addr, time.monotonic() - started)
                    return True
                return False

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

    async def select(self) -> dict | None:
        """Get the next usable proxy. Checks hot buffer first, then verifies one on-the-fly.

        Serialized with a lock so concurrent requests share one verification
        pass instead of each blacklisting its own failed batch.

        Returns the selected proxy entry (callers read the address off it and
        MUST pair each successful select with a release() when the request
        attempt finishes). Returns None if nothing usable is available.
        """
        async with self._select_lock:
            return await self._select_unlocked()

    async def _select_unlocked(self) -> dict | None:
        """select() body; callers must hold _select_lock.

        Sticky by default: reuse the current proxy while it is healthy and
        under its per-proxy concurrency cap. When the cap is reached (a burst
        of concurrent requests, e.g. omp subagent spawns), spill to the
        least-loaded verified hot proxy instead of overloading the single one.
        """
        if self.current:
            addr = self.current["address"]
            if not self._is_bad(addr):
                if self._load(addr) < MAX_PER_PROXY:
                    self._inflight[addr] = self._inflight.get(addr, 0) + 1
                    self._inflight_at[addr] = time.time()
                    return self.current
                # current is at capacity — fall through to spread the load
                overloaded_current = addr
            else:
                self.current = None
                overloaded_current = None
        else:
            overloaded_current = None

        if not self.candidates and self._source_task:
            await self._source_task

        # 1. Hot buffer — least-loaded verified proxy with spare capacity,
        #    skipping the overloaded current so it isn't hit even harder.
        #    Prune stale/bad entries first so dead weight doesn't accumulate
        #    and block the background refill from topping the buffer up.
        now = time.time()
        self.hot = [
            p for p in self.hot
            if (now - float(p.get("verified_at", 0))) <= HOT_TTL
            and not self._is_bad(p["address"])
        ]
        best = None
        best_score = None
        for p in self.hot:
            verified_at = float(p.get("verified_at", 0))
            if now - verified_at > HOT_TTL:
                continue
            if self._is_bad(p["address"]):
                continue
            if p["address"] == overloaded_current:
                continue
            load = self._load(p["address"])
            if load >= MAX_PER_PROXY:
                continue
            # Rank by stability: latency EWMA discounted by reliability
            # (Laplace-smoothed ok/fail), with a mild penalty per extra
            # in-flight request so bursts still spread across exits.
            score = self._quality(p["address"]) * (1 + load)
            if best is None or score < best_score:
                best = p
                best_score = score
        if best is not None:
            self.current = best
            self._inflight[best["address"]] = self._inflight.get(best["address"], 0) + 1
            self._inflight_at[best["address"]] = time.time()
            _log(f"Selected from hot: {best['address']}")
            self._trigger_refill()
            return best

        # Current at capacity and no spare verified proxy: reuse it rather than
        # pay a full on-the-fly verification pass.
        if overloaded_current:
            self._inflight[overloaded_current] = self._inflight.get(overloaded_current, 0) + 1
            self._inflight_at[overloaded_current] = time.time()
            return self.current

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
                            self._inflight[addr] = self._inflight.get(addr, 0) + 1
                            self._inflight_at[addr] = time.time()
                            _log(f"Selected after on-the-fly verify: {addr}")
                            self._trigger_refill()
                            return p
                        self.blacklist[addr] = time.time() + BLACKLIST_TTL
                # All failed
                _log(f"No usable proxy ({self.get_pool_state()})")
                return None
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
        return None

    def release(self, addr: str | None = None):
        """Release the per-proxy concurrency slot reserved by a prior select().

        Callers MUST pair every successful select() with a release() once the
        request attempt finishes (success, failure, rate-limit or cancellation).
        """
        target = addr or (self.current and self.current["address"])
        if not target:
            return
        n = self._inflight.get(target, 0)
        if n > 1:
            self._inflight[target] = n - 1
        else:
            self._inflight.pop(target, None)
            self._inflight_at.pop(target, None)

    def _load(self, addr: str) -> int:
        """In-flight request count for a proxy, self-healing any slots that
        were never released (e.g. a stream cancelled before its finally ran).
        A stale slot beyond the max stream duration is treated as free."""
        n = self._inflight.get(addr, 0)
        if n == 0:
            return 0
        at = self._inflight_at.get(addr, 0)
        if time.time() - at > STREAM_READ_TIMEOUT:
            self._inflight.pop(addr, None)
            self._inflight_at.pop(addr, None)
            return 0
        return n

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
        self.transport_failures.pop(target, None)
        self.rate_limits[target] = time.time() + RATE_LIMIT_TTL
        self._evict_client(target)
        _log(f"Rate-limited {target} for {RATE_LIMIT_TTL // 60}m; rotating")
        if self.current and self.current["address"] == target:
            self.current = None

    def rotate_without_blacklist(self, addr: str | None = None):
        """Drop sticky selection for ``addr`` WITHOUT blacklisting it.

        For upstream capacity errors (503 "Endpoint is unavailable"): the same
        proxy succeeds seconds later, so blacklisting would only shrink the
        pool. Clearing ``current`` makes the next select() prefer a different
        exit while this one stays eligible via the hot buffer.
        """
        target = addr or (self.current and self.current["address"])
        if not target:
            return
        if self.current and self.current["address"] == target:
            self.current = None

    def report_failure(self, addr: str | None = None, hard: bool = False):
        """Record a transport-level failure for ``addr``.

        Soft failures (e.g. a single ReadError) get one grace retry on the
        same exit; ``hard`` failures (connect timeout, proxy unreachable)
        blacklist immediately — an exit that refuses connections is dead for
        the rest of this request anyway.
        """
        target = addr or (self.current and self.current["address"])
        if not target:
            return
        failures = self.transport_failures.get(target, 0) + 1
        self.transport_failures[target] = failures
        self._mark_fail(target)
        self._evict_client(target)
        # Drop the proxy from the hot buffer on ANY transport failure so the
        # next select() doesn't hand it to concurrent requests; it can only
        # come back after passing a brand-new full verification in a later
        # refill. Hard failures (connect timeout, proxy unreachable) blacklist
        # immediately; soft failures (<2) get one grace retry on the same exit
        # before it is blacklisted.
        self.hot = [p for p in self.hot if p["address"] != target]
        if not hard and failures < 2:
            _log(f"Transport failure {target} ({failures}/2); retrying same proxy")
            return
        self.blacklist[target] = time.time() + BLACKLIST_TTL
        self.transport_failures.pop(target, None)
        reason = "hard transport failure" if hard else f"{failures} transport failures"
        _log(f"Blacklisted {target} after {reason} for {BLACKLIST_TTL // 60}m")
        if self.current and self.current["address"] == target:
            self.current = None

    def _blacklist_and_rotate(self, addr: str | None, reason_tag: str):
        """Shared: blacklist ``addr`` for BLACKLIST_TTL and clear sticky current."""
        target = addr or (self.current and self.current["address"])
        if not target:
            return None
        self.transport_failures.pop(target, None)
        self.blacklist[target] = time.time() + BLACKLIST_TTL
        self._mark_fail(target)
        self._evict_client(target)
        _log(f"{reason_tag} {target}; blacklisted for {BLACKLIST_TTL // 60}m; rotating")
        if self.current and self.current["address"] == target:
            self.current = None
        return target

    def report_stream_failure(self, addr: str | None = None):
        """A stream died mid-body (torn tunnel / incomplete chunked read)."""
        self._blacklist_and_rotate(addr, "Stream failure; blacklisted")

    def report_region_block(self, addr: str | None = None):
        """A proxy returned a geo-restriction (RegionError) for the model."""
        self._blacklist_and_rotate(addr, "Region-blocked")

    def report_success(self, addr: str | None = None):
        target = addr or (self.current and self.current["address"])
        if not target:
            return
        self._mark_ok(target)
        had_failure = self.transport_failures.pop(target, None)
        if had_failure and self.current and self.current["address"] == target:
            # This request only succeeded after a transport failure on the
            # same proxy (e.g. a stale keep-alive tunnel or a flaky SOCKS
            # CONNECT). Don't keep it sticky for the next request: the 1/2
            # strike never escalates because this success clears it, so the
            # proxy would otherwise pay a ConnectError + retry on every
            # subsequent request. Rotate so the next request starts on a
            # fresh verified proxy; the flaky one stays in the hot buffer
            # and still gets selected occasionally.
            self.current = None

    def _evict_client(self, addr: str):
        """Drop the pooled AsyncClient for a proxy that just failed.

        The client is only removed from the selection dicts — it is NOT
        closed here. Closing it would cancel every in-flight stream riding
        the same exit (observed as sibling streams on other models dying at
        the same second when one request tripped the blacklist). In-flight
        responses hold their own transport references and finish naturally;
        garbage collection reclaims the client once its last stream ends.
        Bound by MAX_POOL_SIZE (≤5000 distinct proxy URLs → ≤10k clients).
        Future: add LRU idle-close after HOT_TTL for idle clients.
        """
        self._clients.pop(f"socks5://{addr}", None)
        self._stream_clients.pop(f"socks5://{addr}", None)

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
