"""NVIDIA NIM API key pool with rate-limit-aware rotation.

Reads nvapi- keys from ``nvidia-api-keys.txt`` (one per line) and hands them
out round-robin. Any key that hits a rate limit (429) is immediately rotated
away into a cooldown whose length grows **exponentially per key** (60s, 120s,
240s, ... capped at 10m), so a key that keeps getting rate-limited backs off
progressively and stops being offered. This avoids the thrash that happened
with a "2 consecutive" threshold under a saturated shared free-tier quota.
"""
import os
import sys
import time
from pathlib import Path

_log_stdout = True

# Resolve the keys file across all run modes:
#   - frozen (PyInstaller onefile): bundled data lives in the temp _MEIPASS dir
#   - frozen (PyInstaller onefile / exe): also fall back to the exe's own dir
#   - non-frozen (`python server.py`): resolve relative to this module
if getattr(sys, "frozen", False):
    _CANDIDATES = [
        Path(getattr(sys, "_MEIPASS", "")),
        Path(sys.executable).parent,
        Path.cwd(),
    ]
else:
    _CANDIDATES = [Path(__file__).parent, Path.cwd()]


def _find_keys_file() -> Path:
    for d in _CANDIDATES:
        if d and d.is_dir():
            candidate = d / "nvidia-api-keys.txt"
            if candidate.is_file():
                return candidate
    # Default to the first candidate for the "not found" log message.
    return (_CANDIDATES[0] / "nvidia-api-keys.txt")


KEYS_FILE = _find_keys_file()

# A key must be cooled down this many seconds once rotated due to rate limits.
# Kept for back-compat; the live scheme is the escalating cooldown below
# (pi-freeflow port): 30s base, 45s on 5xx, 60s on 504,
# 90s * min(4, consecutive_failures) on 429, capped at COOLDOWN_MAX_SECS.
COOLDOWN_SECS = 60
COOLDOWN_MAX_SECS = 600  # cap the per-key exponential cooldown at 10 minutes
# ── Sticky-primary + healthy-first + escalating cooldown (pi-freeflow port) ──
COOLDOWN_BASE_MS = 30_000
COOLDOWN_5XX_MS = 45_000
COOLDOWN_504_MS = 60_000
COOLDOWN_429_MS = 90_000
COOLDOWN_MAX_MULT = 4
# 60s sliding window: >5 429s/min across the pool logs one burst-warning
# (throttled to at most one log per window).
BURST_WINDOW_SECS = 60.0
BURST_THRESHOLD = 5
# Rotate a key away after this many *consecutive* 429s. Set to 1 so a **single**
# 429 immediately cools the key down (with exponential escalating cooldown),
# preventing the useless "thrash" seen when the shared free-tier quota is
# saturated: previously a key had to 429 twice *consecutively*, but rotation
# moved to a new key after 1, so keys never accumulated 2-in-a-row and nothing
# ever cooled down. With a threshold of 1, each 429ed key backs off and stops
# being offered, letting truly-starved keys drain while the rest of the pool
# stays usable.
RATE_LIMIT_THRESHOLD = 1

# NVIDIA NIM API endpoint + client timeout. Long-thinking models (e.g.
# deepseek-v4-pro) emit zero bytes while reasoning, so give them generous
# read windows; the connect phase stays short.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_CONNECT_TIMEOUT = 5
# Long-thinking models (deepseek-v4-pro / -flash) can stay silent for 5+ minutes
# while reasoning; keep these windows generous (>= the 10m cooldown cap) so a
# slow first token isn't killed mid-reasoning.
NVIDIA_READ_TIMEOUT = 600
NVIDIA_STREAM_READ_TIMEOUT = 600


def _log(*a):
    if not _log_stdout:
        return
    print(f"[{time.strftime('%H:%M:%S')}] [nvidia] " + " ".join(str(x) for x in a), flush=True)


def _load_keys() -> list[str]:
    """Return non-empty trimmed nvapi- keys from the keys file (any order)."""
    keys: list[str] = []
    try:
        raw = KEYS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        _log(f"Keys file not found: {KEYS_FILE}")
        return keys
    for line in raw.splitlines():
        line = line.strip()
        # A leading BOM (\ufeff) can survive a utf-8 read on the first line;
        # strip it so the fixed 70-char check still matches.
        line = line.lstrip("\ufeff").strip()
        # Only genuine keys load: NVIDIA nvapi- keys are a fixed 70-char total.
        if line.startswith("nvapi-") and len(line) == 70:
            keys.append(line)
    if not keys:
        _log("No nvapi- keys found in nvidia-api-keys.txt")
    else:
        _log(f"Loaded {len(keys)} NVIDIA API keys from {KEYS_FILE.name}")
    return keys


class NVIDIAKeyPool:
    """Sticky-primary pool of NVIDIA NIM keys with 429-aware rotation + cooldown."""

    def __init__(self):
        self.keys: list[str] = _load_keys()
        self._idx = 0
        # key -> consecutive 429 counter (reset on success)
        self._consecutive_429: dict[str, int] = {}
        # key -> number of cooldown rotation events (drives exponential backoff;
        # kept for back-compat alongside the escalating entry cooldown)
        self._cooldown_events: dict[str, int] = {}
        # key -> epoch time at which it may be used again (0 = no cooldown)
        self._cooldown_until: dict[str, float] = {}
        # ── pi-freeflow sticky-primary entry state ──
        # {key: {consecutive_failures, last429At, lastLatencyMs,
        #        success, fail, cooling_until}}. cooling_until is epoch seconds.
        self.entries: dict[str, dict] = {}
        self.current: str | None = None
        self._429_times: list[float] = []
        self._last_burst_warn = 0.0

    @property
    def ready(self) -> bool:
        return bool(self.keys)

    def reload(self):
        """Re-read the keys file, preserving cooldown for keys that persist."""
        fresh = _load_keys()
        if not fresh:
            return
        # keep existing 429/cooldown state keyed by key value
        self.keys = fresh
        self._idx = 0
        _log(f"Reloaded pool: {len(self.keys)} keys")

    def _usable(self, key: str) -> bool:
        return time.time() >= self._cooldown_until.get(key, 0) and not self._is_cooling(key)

    def _entry(self, key: str) -> dict:
        return self.entries.setdefault(key, {
            "consecutive_failures": 0,
            "last429At": 0.0,
            "lastLatencyMs": 0,
            "success": 0,
            "fail": 0,
            "cooling_until": 0.0,
        })

    def _is_cooling(self, key: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return float(self.entries.get(key, {}).get("cooling_until", 0.0)) > now

    @staticmethod
    def _cooldown_ms(status: int | None, consecutive_failures: int) -> int:
        """Escalating cooldown: 30s base, 45s on 5xx, 60s on 504,
        90s * min(4, consecutive_failures) on 429 (capped at 10m)."""
        if status == 429:
            mult = max(1, min(COOLDOWN_MAX_MULT, consecutive_failures))
            return min(COOLDOWN_429_MS * mult, COOLDOWN_MAX_SECS * 1000)
        if status == 504:
            return COOLDOWN_504_MS
        if status is not None and 500 <= status < 600:
            return COOLDOWN_5XX_MS
        return COOLDOWN_BASE_MS

    def _note_429(self, now: float):
        self._429_times.append(now)
        cutoff = now - BURST_WINDOW_SECS
        self._429_times = [t for t in self._429_times if t > cutoff]
        if len(self._429_times) > BURST_THRESHOLD and now - self._last_burst_warn >= BURST_WINDOW_SECS:
            self._last_burst_warn = now
            _log(f"429 burst-warning: {len(self._429_times)} 429s in last 60s across pool")

    def _apply_entry_cooldown(self, key: str, status: int | None) -> int:
        now = time.time()
        e = self._entry(key)
        e["consecutive_failures"] = int(e.get("consecutive_failures", 0)) + 1
        e["fail"] = int(e.get("fail", 0)) + 1
        if status == 429:
            e["last429At"] = now
            self._note_429(now)
        ms = self._cooldown_ms(status, e["consecutive_failures"])
        e["cooling_until"] = now + ms / 1000.0
        return ms

    def record_latency(self, key: str, ms: int):
        self._entry(key)["lastLatencyMs"] = int(ms)

    def _effective_cooling_until(self, key: str) -> float:
        return max(
            float(self._cooldown_until.get(key, 0.0)),
            float(self.entries.get(key, {}).get("cooling_until", 0.0)),
        )

    def getOrdered(self) -> list[str]:
        """Healthy-first ordering: sticky current first (if healthy), then
        healthy keys in pool order, then cooling keys by earliest recovery."""
        now = time.time()
        seen: set[str] = set()
        healthy: list[str] = []
        cooling: list[str] = []
        ordered: list[str] = []
        if self.current and self.current in self.keys:
            ordered.append(self.current)
        ordered.extend(self.keys)
        for k in ordered:
            if k in seen:
                continue
            seen.add(k)
            if self._effective_cooling_until(k) > now:
                cooling.append(k)
            else:
                healthy.append(k)
        cooling.sort(key=lambda k: self._effective_cooling_until(k))
        return healthy + cooling

    @staticmethod
    def is_cancelled(exc: BaseException | None) -> bool:
        """True for client AbortError / cancellation — never penalize these."""
        if exc is None:
            return False
        name = type(exc).__name__
        if name in ("AbortError", "CancelledError", "Cancel", "ClientDisconnect", "Disconnect"):
            return True
        try:
            import asyncio as _asyncio
            if isinstance(exc, _asyncio.CancelledError):
                return True
        except Exception:
            pass
        msg = str(exc)
        if "AbortError" in msg or "aborted" in msg.lower() or "client disconnect" in msg.lower():
            return True
        return False

    def report_cancelled(self, key: str | None = None, exc: BaseException | None = None):
        """Client aborted / cancelled — no penalty, keep sticky current."""
        return

    def _rotate_key(self, key: str, ms: int, reason: str):
        self._cooldown_until[key] = max(
            self._cooldown_until.get(key, 0), time.time() + ms / 1000.0
        )
        if self.current == key:
            self.current = None
        usable = sum(1 for k in self.keys if self._usable(k))
        _log(f"Key {key[-8:]} {reason} (+{ms // 1000}s cooldown; {usable} key(s) usable)")

    def report_http_status(self, key: str, status: int | None) -> bool:
        """Record an upstream HTTP status with escalating cooldown.

        Returns True if the caller should roll to the next key, False on
        504 fast-break (don't cycle the pool).
        """
        if status == 504:
            ms = self._apply_entry_cooldown(key, 504)
            _log(f"504 on key {key[-8:]} (+{ms // 1000}s cooldown); fast-break, not cycling pool")
            return False
        if status == 429:
            self.report_rate_limit(key)
            return True
        if status is not None and 500 <= status < 600:
            ms = self._apply_entry_cooldown(key, status)
            self._rotate_key(key, ms, f"HTTP {status}")
            return True
        return True

    def _next_index(self) -> int | None:
        """Return the index of the next usable key, or None if all are cooling."""
        n = len(self.keys)
        if n == 0:
            return None
        for _ in range(n):  # wrap around at most once
            idx = self._idx % n
            self._idx += 1
            if self._usable(self.keys[idx]):
                return idx
        return None

    def select(self) -> str | None:
        """Sticky-primary pick: keep returning current while healthy, else the
        first healthy key from getOrdered() (healthy-first, then cooling).
        When every key is cooling, fall back to the earliest-recovery cooling
        entry (mirrors the proxy-pool fallback) instead of returning None."""
        if self.current and self.current in self.keys and self._usable(self.current):
            return self.current
        if self.current and (self.current not in self.keys or not self._usable(self.current)):
            self.current = None
        ordered = self.getOrdered()
        fallback: str | None = None
        for k in ordered:
            if self._usable(k):
                self.current = k
                # Keep round-robin cursor roughly aligned so a fresh select
                # after rotation doesn't always restart at index 0.
                try:
                    self._idx = self.keys.index(k) + 1
                except ValueError:
                    pass
                return k
            if fallback is None:
                fallback = k
        if fallback is not None:
            self.current = fallback
            try:
                self._idx = self.keys.index(fallback) + 1
            except ValueError:
                pass
            _log(f"Selected cooling key (earliest recovery): ...{fallback[-8:]}")
            return fallback
        return None

    def report_success(self, key: str):
        """A request succeeded: the key is healthy, clear its counters."""
        self._consecutive_429.pop(key, None)
        self._cooldown_events.pop(key, None)
        e = self._entry(key)
        e["consecutive_failures"] = 0
        e["cooling_until"] = 0.0
        e["success"] = int(e.get("success", 0)) + 1

    def _cooldown_for(self, n: int) -> int:
        """Exponential per-key cooldown in seconds for a key that has been
        rotated ``n`` times (1-based event count)."""
        return min(COOLDOWN_SECS * (2 ** (n - 1)), COOLDOWN_MAX_SECS)

    def _check_rotation(self, key: str) -> bool:
        """Return True once a key's consecutive-429 count reaches the threshold."""
        n = self._consecutive_429.get(key, 0)
        if n >= RATE_LIMIT_THRESHOLD:
            # Number of distinct rotation events this key has already had
            # (drives the exponential cooldown escalation).
            events = self._cooldown_events.get(key, 0) + 1
            self._cooldown_events[key] = events
            legacy_cd = self._cooldown_for(events)
            ms = self._apply_entry_cooldown(key, 429)
            # Escalating entry cooldown wins when longer; legacy exponential
            # floor (60s doubling to 600s) is preserved otherwise.
            cd = max(legacy_cd, ms // 1000)
            self._cooldown_until[key] = time.time() + cd
            if self.current == key:
                self.current = None
            self._log_rotate(key, n, cd)
            return True
        return False

    def _log_rotate(self, key: str, n: int, cd: int):
        usable = sum(1 for k in self.keys if self._usable(k))
        _log(
            f"Key {key[-8:]} hit {n} consecutive rate limits; cooling down "
            f"{cd}s ({usable} key(s) usable)"
        )

    def report_rate_limit(self, key: str) -> bool:
        """Record a 429 for ``key``. Returns True if it was rotated away (this
        is the 2nd consecutive rate limit) and the caller should switch keys."""
        self._consecutive_429[key] = self._consecutive_429.get(key, 0) + 1
        return self._check_rotation(key)


pool = NVIDIAKeyPool()
