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
# The cooldown grows *exponentially per key*: each time a key is rotated away
# again, its next cooldown doubles (60s -> 120s -> 240s -> ...), so a key that
# keeps getting rate-limited backs off increasingly hard instead of thrashing
# the pool on a fixed 60s timer.
COOLDOWN_SECS = 60
COOLDOWN_MAX_SECS = 600  # cap the per-key exponential cooldown at 10 minutes
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
    """Round-robin pool of NVIDIA NIM keys with 429-aware rotation + cooldown."""

    def __init__(self):
        self.keys: list[str] = _load_keys()
        self._idx = 0
        # key -> consecutive 429 counter (reset on success)
        self._consecutive_429: dict[str, int] = {}
        # key -> number of cooldown rotation events (drives exponential backoff)
        self._cooldown_events: dict[str, int] = {}
        # key -> epoch time at which it may be used again (0 = no cooldown)
        self._cooldown_until: dict[str, float] = {}

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
        return time.time() >= self._cooldown_until.get(key, 0)

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
        """Pick the next key to use (skips keys still in cooldown)."""
        idx = self._next_index()
        if idx is None:
            return None
        return self.keys[idx]

    def report_success(self, key: str):
        """A request succeeded: the key is healthy, clear its counters."""
        self._consecutive_429.pop(key, None)
        self._cooldown_events.pop(key, None)

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
            cd = self._cooldown_for(events)
            self._cooldown_until[key] = time.time() + cd
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
