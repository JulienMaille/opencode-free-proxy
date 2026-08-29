"""NVIDIA NIM API key pool with rate-limit-aware rotation.

Reads nvapi- keys from ``nvidia-api-keys.txt`` (one per line) and hands them
out round-robin. Tracks consecutive 429 rate-limit responses per key; a key
that hits a rate limit **twice in a row** is rotated away and sent to a
temporary cooldown before it can be reused.
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
COOLDOWN_SECS = 60
# Rotate a key away after this many *consecutive* 429s (2 = rotate on the 2nd
# consecutive rate-limit in a row, per the shared-window requirement).
RATE_LIMIT_THRESHOLD = 2

# NVIDIA NIM API endpoint + client timeout. Long-thinking models (e.g.
# deepseek-v4-pro) emit zero bytes while reasoning, so give them generous
# read windows; the connect phase stays short.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_CONNECT_TIMEOUT = 5
NVIDIA_READ_TIMEOUT = 300
NVIDIA_STREAM_READ_TIMEOUT = 300


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
        if line.startswith("nvapi-"):
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
        """A request succeeded: the key is healthy, clear its 429 counter."""
        self._consecutive_429.pop(key, None)

    def _check_rotation(self, key: str) -> bool:
        """Return True once a key's consecutive-429 count reaches the threshold."""
        n = self._consecutive_429.get(key, 0)
        if n >= RATE_LIMIT_THRESHOLD:
            self._cooldown_until[key] = time.time() + COOLDOWN_SECS
            self._log_rotate(key, n)
            return True
        return False

    def _log_rotate(self, key: str, n: int):
        usable = sum(1 for k in self.keys if self._usable(k))
        _log(
            f"Key {key[-8:]} hit {n} consecutive rate limits; cooling down "
            f"{COOLDOWN_SECS}s ({usable} key(s) usable)"
        )

    def report_rate_limit(self, key: str) -> bool:
        """Record a 429 for ``key``. Returns True if it was rotated away (this
        is the 2nd consecutive rate limit) and the caller should switch keys."""
        self._consecutive_429[key] = self._consecutive_429.get(key, 0) + 1
        return self._check_rotation(key)


pool = NVIDIAKeyPool()
