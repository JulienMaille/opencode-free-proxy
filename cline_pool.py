"""Per-user Cline login-slot pool (JSON) + legacy static-key slots.

Mechanical port of pi-freeflow's ``src/cline-accounts.ts``: Cline serves its
free models direct-only (never through the SOCKS pool), one bearer per login
slot. Slots live in ``cline-pool.json`` (0600, repo root, gitignored) with a
largest-wins ``.bak`` recovery copy; a corrupt file reads back as an empty
pool (never throws, never logs secrets).

Static keys from ``cline-api-keys.txt`` keep working: non-empty non-``#``
lines auto-import as key slots (``key-N``, no refresh/expiry), merged with
the JSON pool and deduped by token. ``sk_`` keys are accepted too (the
pre-existing user key keeps loading).

429 free-limit bookkeeping is keyed by the REQUESTED model id (Cline spells its
own name in the 429 body, a different namespace). Roll ordering is
activeSlot-first then capped-last; capped slots are still tried. Success
clears the cap, bumps usage, and marks the slot active. All-capped rolls
report ``limit_only=True`` + ``earliest_reset_at``.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from cline_device_auth import is_workos_jwt, refresh_cline_token, ClineAuthError, to_api_key

_log_stdout = True

# Resolve the pool file across all run modes (mirrors the keys-file logic):
# frozen exe -> exe dir; otherwise beside this module; honour
# CLINE_POOL_FILE env override (tests/CI sandbox).
if getattr(sys, "frozen", False):
    _POOL_CANDIDATES = [Path(sys.executable).parent, Path.cwd()]
else:
    _POOL_CANDIDATES = [Path(__file__).parent, Path.cwd()]

CLINE_POOL_FILE_ENV = "CLINE_POOL_FILE"


def _default_pool_path() -> Path:
    base = _POOL_CANDIDATES[0] if _POOL_CANDIDATES else Path.cwd()
    try:
        if not base.is_dir():
            base = Path.cwd()
    except Exception:
        base = Path.cwd()
    return base / "cline-pool.json"


def resolve_pool_path() -> Path:
    override = (os.environ.get(CLINE_POOL_FILE_ENV) or "").strip()
    if override:
        return Path(override)
    return _default_pool_path()


POOL_FILE = resolve_pool_path()
POOL_BACKUP_FILE = Path(str(POOL_FILE) + ".bak")

# Static keys file (kept for paying key slots).
if getattr(sys, "frozen", False):
    _KEY_CANDIDATES = [
        Path(getattr(sys, "_MEIPASS", "")),
        Path(sys.executable).parent,
        Path.cwd(),
    ]
else:
    _KEY_CANDIDATES = [Path(__file__).parent, Path.cwd()]


def _find_keys_file() -> Path:
    for d in _KEY_CANDIDATES:
        try:
            if d and d.is_dir():
                candidate = d / "cline-api-keys.txt"
                if candidate.is_file():
                    return candidate
        except Exception:
            continue
    return (_KEY_CANDIDATES[0] / "cline-api-keys.txt")


KEYS_FILE = _find_keys_file()

# Cline OpenAI-compatible API endpoint + client timeout. Long-thinking models
# emit zero bytes while reasoning, so read windows stay generous.
CLINE_BASE_URL = "https://api.cline.bot/api/v1"
CLINE_MODELS_URL = CLINE_BASE_URL + "/models"
CLINE_CHAT_URL = CLINE_BASE_URL + "/chat/completions"
CLINE_CONNECT_TIMEOUT = 5
CLINE_READ_TIMEOUT = 600
CLINE_STREAM_READ_TIMEOUT = 600

# Full desktop-client identity set (pi-freeflow CLINE_CLIENT_HEADERS):
# cline-free/* 403s with a bare bearer, 200s with this set.
CLINE_CLIENT_VERSION = "3.5.54"
CLINE_CLIENT_HEADERS = {
    "HTTP-Referer": "https://cline.bot",
    "X-Title": "Cline",
    "X-IS-MULTIROOT": "false",
    "X-CLIENT-TYPE": "cline-desktop",
    "X-CLIENT-VERSION": CLINE_CLIENT_VERSION,
    "User-Agent": f"Cline/{CLINE_CLIENT_VERSION}",
}

# Skew so a token expiring mid-flight counts as stale before it breaks a call.
CLINE_REFRESH_SKEW_MS = 60_000


def _log(*a):
    if not _log_stdout:
        return
    print(f"[{time.strftime('%H:%M:%S')}] [cline] " + " ".join(str(x) for x in a), flush=True)


def redacted_token(token: str) -> str:
    """Last-4 display for a token; the full value is never shown or logged."""
    tail = (token or "")[-4:] or "????"
    return f"…{tail}"


# ── token acceptance ─────────────────────────────────────────────

def _accepts_token(token: str) -> bool:
    """workos: prefix, JWT shape, clp_ keys, or sk_ keys."""
    if not isinstance(token, str) or not token:
        return False
    return (token.startswith("workos:") or is_workos_jwt(token)
            or token.startswith("clp_") or token.startswith("sk_"))


# ── static key slots ─────────────────────────────────────────────

def _load_static_keys() -> list[str]:
    """Non-empty non-# lines from cline-api-keys.txt (BOM-tolerant)."""
    keys: list[str] = []
    try:
        raw = KEYS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return keys
    except OSError:
        return keys
    for line in raw.splitlines():
        line = line.strip().lstrip("﻿").strip()
        if not line or line.startswith("#"):
            continue
        if _accepts_token(line):
            keys.append(line)
    return keys


# ── pool document parse ──────────────────────────────────────────

def _empty_pool() -> dict:
    return {"accounts": []}


def _parse_limits(raw, now_ms: int):
    if not isinstance(raw, dict):
        return None
    limits: dict = {}
    for slot, models_raw in raw.items():
        if not isinstance(slot, str) or not slot.strip():
            continue
        if not isinstance(models_raw, dict):
            continue
        models: dict = {}
        for model, reset_raw in models_raw.items():
            if not isinstance(model, str) or not model.strip():
                continue
            if not isinstance(reset_raw, (int, float)) or not (reset_raw == reset_raw) or reset_raw <= now_ms:
                continue
            models[model] = reset_raw
        if models:
            limits[slot.strip()] = models
    return limits or None


def _parse_usage(raw):
    if not isinstance(raw, dict):
        return None
    usage: dict = {}
    for slot, entry_raw in raw.items():
        if not isinstance(slot, str) or not slot.strip():
            continue
        if not isinstance(entry_raw, dict):
            continue
        served = entry_raw.get("served")
        last_at = entry_raw.get("lastAt")
        last_model = entry_raw.get("lastModel")
        if not isinstance(served, (int, float)) or not (served == served) or served < 0:
            continue
        if not isinstance(last_at, (int, float)) or not (last_at == last_at) or last_at <= 0:
            continue
        if not isinstance(last_model, str):
            continue
        usage[slot.strip()] = {"served": int(served), "lastAt": last_at, "lastModel": last_model}
    return usage or None


def _parse_pool_doc(raw: str):
    """Parse a pool document. None when unusable (bad JSON/shape) so callers
    fall back to backup. A valid doc with zero accounts is a real empty pool.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        _log(f"cline pool file corrupt: {POOL_FILE}")
        return None
    if not isinstance(parsed, dict):
        _log(f"cline pool file unusable: {POOL_FILE}")
        return None
    doc = parsed
    if not isinstance(doc.get("accounts"), list):
        _log(f"cline pool file unusable: {POOL_FILE}")
        return None
    accounts: list = []
    for entry in doc["accounts"]:
        if not isinstance(entry, dict):
            continue
        slot = entry.get("slot")
        token = entry.get("token")
        slot = slot.strip() if isinstance(slot, str) else ""
        if not slot or not isinstance(token, str) or not _accepts_token(token):
            continue
        acc: dict = {
            "slot": slot,
            "token": token,
            "addedAt": entry.get("addedAt") if isinstance(entry.get("addedAt"), str) else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        refresh = entry.get("refreshToken")
        if isinstance(refresh, str) and refresh:
            acc["refreshToken"] = refresh
        expires = entry.get("expiresAt")
        if isinstance(expires, (int, float)) and expires == expires and expires > 0:
            acc["expiresAt"] = expires
        for f in ("accountId", "email"):
            v = entry.get(f)
            if isinstance(v, str) and v.strip():
                acc[f] = v.strip()
        accounts.append(acc)
    active_raw = doc.get("activeSlot")
    active = active_raw.strip() if isinstance(active_raw, str) else ""
    state: dict = {"accounts": accounts}
    if active and any(a["slot"] == active for a in accounts):
        state["activeSlot"] = active
    limits = _parse_limits(doc.get("limits"), int(time.time() * 1000))
    if limits:
        state["limits"] = limits
    usage = _parse_usage(doc.get("usage"))
    if usage:
        state["usage"] = usage
    return state


def _account_count_in(raw: str) -> int:
    try:
        doc = json.loads(raw)
        accts = doc.get("accounts") if isinstance(doc, dict) else None
        return len(accts) if isinstance(accts, list) else -1
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return -1


def _recover_from_backup(description: str):
    """Recover from .bak and heal the main file. Only for missing/unusable
    main — never overrides a valid empty pool."""
    try:
        raw = POOL_BACKUP_FILE.read_text(encoding="utf-8")
    except OSError:
        return None
    parsed = _parse_pool_doc(raw)
    if parsed and parsed["accounts"]:
        _log(f"cline pool {description} — recovered from backup ({len(parsed['accounts'])} slots)")
        save_pool(parsed)
        return parsed
    return None


def _read_pool_file() -> dict:
    try:
        if not POOL_FILE.is_file():
            return _recover_from_backup("main file missing") or _empty_pool()
        parsed = _parse_pool_doc(POOL_FILE.read_text(encoding="utf-8"))
        if parsed is not None:
            return parsed
        return _recover_from_backup("file unusable") or _empty_pool()
    except OSError:
        return _empty_pool()


_cached = None
_cached_mtime = -1


def _disk_mtime() -> float:
    try:
        return POOL_FILE.stat().st_mtime
    except OSError:
        return -1.0


def reset_pool_cache_for_test():
    """Test-only: drop the load_pool mtime cache."""
    global _cached, _cached_mtime
    _cached = None
    _cached_mtime = -1


def load_pool() -> dict:
    """Load the pool, re-reading only when another process changed the file."""
    global _cached, _cached_mtime
    m = _disk_mtime()
    if _cached is not None and m == _cached_mtime:
        return _cached
    _cached = _read_pool_file()
    _cached_mtime = _disk_mtime()
    return _cached


def _keep_largest_backup(raw: str):
    """Largest-wins backup: ties never replace (a run of equally-sized
    destructive writes cannot evict a good copy)."""
    try:
        count = _account_count_in(raw)
        if count <= 0:
            return
        try:
            backup_count = _account_count_in(POOL_BACKUP_FILE.read_text(encoding="utf-8"))
        except OSError:
            backup_count = -1
        if count <= backup_count:
            return
        POOL_BACKUP_FILE.write_text(raw, encoding="utf-8")
        try:
            os.chmod(POOL_BACKUP_FILE, 0o600)
        except OSError:
            pass
    except OSError:
        pass


def save_pool(pool: dict):
    """Atomically persist the pool (tmp+rename) with owner-only permissions."""
    global _cached, _cached_mtime
    try:
        parent = POOL_FILE.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        try:
            if POOL_FILE.is_file():
                _keep_largest_backup(POOL_FILE.read_text(encoding="utf-8"))
        except OSError:
            pass
        tmp = POOL_FILE.parent / f"{POOL_FILE.name}.{int(time.time() * 1000)}.tmp"
        serialized = json.dumps(pool, indent=2)
        tmp.write_text(serialized, encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, POOL_FILE)
        _keep_largest_backup(serialized)
        _cached = pool
        _cached_mtime = _disk_mtime()
    except OSError:
        _log(f"could not persist cline pool ({len(pool.get('accounts', []))} slots)")


def _patch_account_on_disk(slot: str, patch: dict):
    """Persist a patch to one slot against the file as it is on disk now
    (never write a stale snapshot back; a removed slot stays removed)."""
    live = _read_pool_file()
    target = next((a for a in live["accounts"] if a.get("slot") == slot), None)
    if target is None:
        return
    target.update(patch)
    save_pool(live)


def _mark_active_on_disk(slot: str):
    live = _read_pool_file()
    if not any(a.get("slot") == slot for a in live["accounts"]):
        return
    live["activeSlot"] = slot
    save_pool(live)


def record_usage_on_disk(slot: str, model_id: str):
    """Bump one slot's serve counters on disk (counters only — never token)."""
    slot = (slot or "").strip()
    if not slot:
        return
    live = _read_pool_file()
    if not any(a.get("slot") == slot for a in live["accounts"]):
        return
    usage = live.get("usage") or {}
    prev = usage.get(slot) or {}
    served = prev.get("served")
    served = int(served) + 1 if isinstance(served, (int, float)) and served >= 0 else 1
    usage[slot] = {"served": served, "lastAt": int(time.time() * 1000),
                   "lastModel": cap_model_key(model_id)}
    live["usage"] = usage
    save_pool(live)


# Routing prefix the proxy adds in front of a canonical Cline id
# ("cline/cline-free/kimi-k3"). The cap gate compares CANONICAL ids, so a
# limits/usage entry keyed by the prefixed form is invisible to it: the next
# request misses the cap, calls upstream and 429s again.
_ROUTING_PREFIX_RE = re.compile(r"^\s*cline/", re.IGNORECASE)


def cap_model_key(model: str | None) -> str:
    """Canonical limits/usage map key for a requested model id.

    Applied at the persistence boundary so no call site can key bookkeeping by
    the routing-prefixed form the gate will never match.
    """
    return _ROUTING_PREFIX_RE.sub("", str(model or "").strip())


def _cap_keys(model: str | None) -> list:
    """Limits-entry lookup order: canonical key first, then the raw form.

    The raw fallback still reads entries written before this normalisation (or
    by older builds), so an existing pool file keeps failing fast.
    """
    canon = cap_model_key(model)
    if not canon:
        return []
    raw = str(model).strip()
    return [canon] if canon == raw else [canon, raw]


def _active_limit_for(pool: dict, slot: str, model: str, now_ms: int):
    if not model:
        return None
    caps = (pool.get("limits") or {}).get(slot) or {}
    at = next((caps[k] for k in _cap_keys(model) if isinstance(caps.get(k), (int, float))), None)
    return at if isinstance(at, (int, float)) and at == at and at > now_ms else None


def record_limit_on_disk(slot: str, model: str, reset_at: int):
    """Persist one free-limit observation on disk (keeps the later window)."""
    keys = _cap_keys(model)
    if not keys:
        return
    live = _read_pool_file()
    if not any(a.get("slot") == slot for a in live["accounts"]):
        return
    limits = live.get("limits") or {}
    slot_limits = dict(limits.get(slot) or {})
    # Keep the later window across every key this model may live under, so a
    # legacy prefixed entry can neither shorten nor duplicate the new one.
    known = max((slot_limits[k] for k in keys if isinstance(slot_limits.get(k), (int, float))),
                default=None)
    if known is not None and known >= reset_at:
        return
    for k in keys:  # drop stale duplicates, keep one canonical entry
        slot_limits.pop(k, None)
    slot_limits[keys[0]] = reset_at
    limits[slot] = slot_limits
    live["limits"] = limits
    save_pool(live)


def clear_limit_on_disk(slot: str, model: str):
    """Drop one slot+model limit once that slot serves the model again."""
    keys = _cap_keys(model)
    if not keys:
        return
    live = _read_pool_file()
    limits = live.get("limits")
    for_slot = (limits or {}).get(slot)
    if not limits or not isinstance(for_slot, dict):
        return
    if not any(isinstance(for_slot.get(k), (int, float)) for k in keys):
        return
    for k in keys:  # clear every form so no stale cap survives the success
        for_slot.pop(k, None)
    if not for_slot:
        del limits[slot]
    if not limits:
        live.pop("limits", None)
    save_pool(live)


# ── same-account guard ───────────────────────────────────────────

def _same_account(account: dict, identity: dict) -> bool:
    """accountId authoritative; email fallback; else identical bearer."""
    id_a = (account.get("accountId") or "").strip()
    id_b = (identity.get("accountId") or "").strip() if isinstance(identity.get("accountId"), str) else ""
    if id_a and id_b:
        return id_a == id_b
    mail_a = (account.get("email") or "").strip().lower()
    mail_b = (identity.get("email") or "").strip().lower() if isinstance(identity.get("email"), str) else ""
    if mail_a and mail_b:
        return mail_a == mail_b
    token = identity.get("token")
    return isinstance(token, str) and bool(token) and account.get("token") == token


def find_account_slot(pool: dict, identity: dict, except_slot=None):
    """Other slot already holding this account, or None. `except_slot` skips
    the slot being written so re-logging is never a duplicate."""
    hit = next((a for a in pool.get("accounts", [])
                if a.get("slot") != except_slot and _same_account(a, identity)), None)
    return hit["slot"] if hit else None


def add_account(slot: str, token: str, extras=None) -> dict:
    """Save (or replace) one login slot. Refuses an account another slot
    already holds. Raises on bad input — messages never echo the token."""
    extras = extras or {}
    clean_slot = (slot or "").strip()
    if not clean_slot:
        raise ValueError("Cline slot name cannot be empty")
    if len(clean_slot) > 64:
        raise ValueError("Cline slot name is too long (max 64 characters)")
    if not _accepts_token(token):
        raise ValueError("Cline token must be a workos: login grant, JWT, clp_ key, or sk_ key")
    pool = load_pool()
    existing = next((a for a in pool["accounts"] if a.get("slot") == clean_slot), None)
    duplicate = find_account_slot(
        pool,
        {"token": token, "accountId": extras.get("accountId"), "email": extras.get("email")},
        clean_slot,
    )
    if duplicate:
        who = (extras.get("email") or "").strip() or ((existing or {}).get("email") or "").strip() or "this account"
        raise ValueError(
            f"That Cline account ({who}) is already saved as [{duplicate}] — "
            f"log in with a different account, or remove {duplicate} first")
    if existing:
        existing["token"] = token
        if extras.get("refreshToken"):
            existing["refreshToken"] = extras["refreshToken"]
        if extras.get("expiresAt") is not None:
            if isinstance(extras["expiresAt"], (int, float)) and extras["expiresAt"] > 0:
                existing["expiresAt"] = extras["expiresAt"]
            else:
                existing.pop("expiresAt", None)
        for f in ("accountId", "email"):
            if extras.get(f) is not None:
                if isinstance(extras[f], str) and extras[f].strip():
                    existing[f] = extras[f].strip()
                else:
                    existing.pop(f, None)
    else:
        acc: dict = {
            "slot": clean_slot,
            "token": token,
            "addedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extras.get("refreshToken"):
            acc["refreshToken"] = extras["refreshToken"]
        if isinstance(extras.get("expiresAt"), (int, float)) and extras["expiresAt"] > 0:
            acc["expiresAt"] = extras["expiresAt"]
        for f in ("accountId", "email"):
            if isinstance(extras.get(f), str) and extras[f].strip():
                acc[f] = extras[f].strip()
        pool["accounts"].append(acc)
    if not pool.get("activeSlot"):
        pool["activeSlot"] = clean_slot
    save_pool(pool)
    return pool


def remove_account(slot: str) -> bool:
    """Drop one login slot. False when the slot was not saved."""
    clean_slot = (slot or "").strip()
    pool = load_pool()
    idx = next((i for i, a in enumerate(pool["accounts"]) if a.get("slot") == clean_slot), -1)
    if idx < 0:
        return False
    pool["accounts"].pop(idx)
    usage = pool.get("usage")
    if isinstance(usage, dict) and clean_slot in usage:
        del usage[clean_slot]
        if not usage:
            pool.pop("usage", None)
    if pool.get("activeSlot") == clean_slot:
        pool["activeSlot"] = pool["accounts"][0]["slot"] if pool["accounts"] else None
        if not pool.get("activeSlot"):
            pool.pop("activeSlot", None)
    save_pool(pool)
    return True


def is_token_stale(account: dict, skew_ms: int = CLINE_REFRESH_SKEW_MS) -> bool:
    """True when the slot carries a known expiry within the skew window.
    Slots without expiresAt (every key slot) are never stale."""
    exp = account.get("expiresAt")
    if not isinstance(exp, (int, float)) or not (exp == exp):
        return False
    return int(time.time() * 1000) + skew_ms >= exp


def refresh_account_in_place(account: dict):
    """Refresh one slot. True=applied, False=dead grant, None=transient (keep
    stale). Persists only the refreshed fields (patch-on-disk)."""
    current = account.get("refreshToken")
    if not current:
        return None
    try:
        fresh = refresh_cline_token(refresh_token=current)
    except ClineAuthError as e:
        if e.error_code == "invalid_grant" or e.status in (400, 401):
            _log(f"cline slot refresh rejected — grant is dead [{account.get('slot')}]")
            return False
        _log(f"cline slot refresh failed — keeping stale bearer [{account.get('slot')}]")
        return None
    except Exception:
        _log(f"cline slot refresh failed — keeping stale bearer [{account.get('slot')}]")
        return None
    token = fresh.get("access")
    if not isinstance(token, str) or not _accepts_token(token):
        _log(f"cline slot refresh rejected — grant is dead [{account.get('slot')}]")
        return False
    # Wire form (pi-freeflow clineRefreshImpl): WorkOS JWTs must ride as
    # workos:<jwt>; the refresh endpoint returns them bare.
    token = to_api_key(token)
    patch = {"token": token}
    if fresh.get("refresh"):
        patch["refreshToken"] = fresh["refresh"]
        account["refreshToken"] = fresh["refresh"]
    if isinstance(fresh.get("expires"), (int, float)) and fresh["expires"] > 0:
        patch["expiresAt"] = fresh["expires"]
    else:
        patch["expiresAt"] = None
    if fresh.get("accountId"):
        patch["accountId"] = fresh["accountId"]
    if fresh.get("email"):
        patch["email"] = fresh["email"]
    account["token"] = token
    for k, v in patch.items():
        if v is None:
            account.pop(k, None)
        else:
            account[k] = v
    on_disk_patch = {k: v for k, v in patch.items() if v is not None}
    if "expiresAt" in patch and patch["expiresAt"] is None:
        # patch-on-disk: explicitly clear the stale expiry.
        live = _read_pool_file()
        target = next((a for a in live["accounts"] if a.get("slot") == account.get("slot")), None)
        if target is not None:
            target.update(on_disk_patch)
            target.pop("expiresAt", None)
            save_pool(live)
    else:
        _patch_account_on_disk(account.get("slot"), on_disk_patch)
    return True


# ── 429 free-limit cap parser (pi-freeflow port) ─────────────────
# Marker: /free limit reached on model/i. Delays: 'try again in Nh Mm|Nm|
# bare N as HOURS'. Model name: trailing punctuation stripped.

_FREE_LIMIT_MARKER_RE = re.compile(r"free limit reached on model", re.IGNORECASE)
_FREE_LIMIT_MODEL_RE = re.compile(r"free limit reached on model\s+(\S+)", re.IGNORECASE)
_RESET_HM_RE = re.compile(r"try again in\s+([0-9]+)\s*h(?:ours?|rs?)?(?:\s*([0-9]+)\s*m(?:in(?:utes?)?)?)?", re.IGNORECASE)
_RESET_M_RE = re.compile(r"try again in\s+([0-9]+)\s*m(?:in(?:utes?)?)?", re.IGNORECASE)
_RESET_BARE_RE = re.compile(r"try again in\s+([0-9]+)(?![0-9a-z])", re.IGNORECASE)


def _parse_reset_delay_ms(text: str):
    m = _RESET_HM_RE.search(text or "")
    if m:
        hours = int(m.group(1))
        minutes = int(m.group(2)) if m.group(2) else 0
        return (hours * 60 + minutes) * 60_000
    m = _RESET_M_RE.search(text or "")
    if m:
        return int(m.group(1)) * 60_000
    m = _RESET_BARE_RE.search(text or "")
    return int(m.group(1)) * 3_600_000 if m else None


def parse_cline_free_limit(raw: str, now_ms=None):
    """Parse an upstream 429 body for the daily free-limit marker.

    Returns {modelId?, resetAt?} or None for anything else (a plain edge 429
    keeps its existing behavior). resetAt is epoch-ms.
    """
    if not isinstance(raw, str) or not _FREE_LIMIT_MARKER_RE.search(raw):
        return None
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    out = {}
    m = _FREE_LIMIT_MODEL_RE.search(raw)
    if m:
        # The model name ends the sentence: only TRAILING punctuation belongs
        # to the prose (pi-freeflow: /[.,;:]+$/). Interior dots (v4.1) stay.
        out["modelId"] = re.sub(r"[.,;:]+$", "", m.group(1))
    delay = _parse_reset_delay_ms(raw)
    if delay is not None:
        out["resetAt"] = now_ms + delay
    return out


# Legacy cap-delay parser (kept: existing tests pin it).
# "try again in ..." window, then up to three "<n><unit>" duration parts
# (hours/minutes/seconds, full words allowed, case-insensitive) or a bare
# number (treated as seconds).
_CAP_WINDOW_RE = re.compile(r"try\s+again\s+in\s+([^\n,;.]{1,48})", re.IGNORECASE)
_CAP_FRAG_RE = re.compile(
    r"^\s*(\d+)\s*(h(?:ours?)?|m(?:in(?:utes?)?)?|s(?:ec(?:onds?)?)?)?"
    r"(?:\s+(\d+)\s*(h(?:ours?)?|m(?:in(?:utes?)?)?|s(?:ec(?:onds?)?)?)?)?"
    r"(?:\s+(\d+)\s*(h(?:ours?)?|m(?:in(?:utes?)?)?|s(?:ec(?:onds?)?)?)?)?\s*$",
    re.IGNORECASE,
)


def _cap_unit_seconds(unit) -> int:
    u = (unit or "").strip().lower()
    if u.startswith("h"):
        return 3600
    if u.startswith("m"):
        return 60
    return 1  # seconds, or a bare number with no unit


def parse_cap_delay_seconds(text) -> int | None:
    """Legacy delay parser: hours/minutes/seconds + bare number as seconds."""
    if not text:
        return None
    m = _CAP_WINDOW_RE.search(str(text))
    if not m:
        return None
    frag = m.group(1).strip()
    fm = _CAP_FRAG_RE.match(frag)
    if not fm:
        return None
    total = 0
    for num, unit in ((fm.group(1), fm.group(2)), (fm.group(3), fm.group(4)), (fm.group(5), fm.group(6))):
        if num is None:
            continue
        total += int(num) * _cap_unit_seconds(unit)
    return total


def _format_cap_delay(secs: int) -> str:
    """Render seconds as a human ``2h 15m``-style delay."""
    secs = max(0, int(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    bits: list[str] = []
    if h:
        bits.append(f"{h}h")
    if m:
        bits.append(f"{m}m")
    if s or not bits:
        bits.append(f"{s}s")
    return " ".join(bits)


def map_cline_error(status) -> str:
    """Machine-classified outcome of one Cline chat attempt."""
    try:
        code = int(status)
    except (TypeError, ValueError):
        return "client"
    if 200 <= code < 300:
        return "ok"
    if code in (401, 403):
        return "auth"
    if code == 429:
        return "rate-limit"
    if code == 402:
        return "exhausted"
    if code >= 500:
        return "server"
    return "client"


# ── unified slot pool (JSON logins + static key slots) ───────────

def _key_slot_accounts() -> list:
    """Static-key slots (slot 'key-N'), deduped by token against JSON slots."""
    pool = load_pool()
    have = {a.get("token") for a in pool.get("accounts", [])}
    out = []
    n = 0
    for token in _load_static_keys():
        if token in have:
            continue
        have.add(token)
        n += 1
        out.append({"slot": f"key-{n}", "token": token,
                    "addedAt": "", "keySlot": True})
    return out


def all_accounts() -> list:
    """JSON login slots in file order, then static key slots."""
    return list(load_pool().get("accounts", [])) + _key_slot_accounts()


def pool_account_count() -> int:
    return len(all_accounts())


def _find_account(slot: str):
    return next((a for a in all_accounts() if a.get("slot") == slot), None)


def _roll_candidates(model: str | None):
    """activeSlot-first, then pool order; with a model, capped-last (capped
    still tried). Returns (candidates, pool, now_ms)."""
    pool = load_pool()
    now_ms = int(time.time() * 1000)
    key_slots = _key_slot_accounts()
    key_by_slot = {a["slot"]: a for a in key_slots}
    candidates = list(pool.get("accounts", [])) + key_slots
    active = pool.get("activeSlot")
    if active:
        at = next((i for i, a in enumerate(candidates) if a.get("slot") == active), -1)
        if at > 0:
            candidates = [candidates[at]] + candidates[:at] + candidates[at + 1:]
    if model:
        open_slots, capped = [], []
        for a in candidates:
            # Key slots carry no limits map entries keyed by slot... they can
            # still be capped; limits are keyed by slot name either way.
            (capped if _active_limit_for(pool, a["slot"], model, now_ms) is not None
             else open_slots).append(a)
        candidates = open_slots + capped
    _ = key_by_slot
    return candidates, pool, now_ms


def roll_order(model: str | None = None) -> list:
    """Slot names in roll order (active-first, capped-last). For tests/UI."""
    return [a["slot"] for a in _roll_candidates(model)[0]]


def record_cap(slot: str, requested_model: str, body_text: str):
    """Record a 429 free-limit cap for (slot, requested model). Returns the
    parsed resetAt (epoch-ms) or None."""
    limit = parse_cline_free_limit(body_text or "")
    if not limit or not isinstance(limit.get("resetAt"), (int, float)):
        return None
    record_limit_on_disk(slot, requested_model or "", int(limit["resetAt"]))
    return int(limit["resetAt"])


def record_slot_success(slot: str, requested_model: str):
    """Success: clear cap + bump usage + mark active (patch-on-disk)."""
    clear_limit_on_disk(slot, requested_model or "")
    record_usage_on_disk(slot, requested_model or "")
    acct = _find_account(slot)
    if acct is not None and not acct.get("keySlot"):
        _mark_active_on_disk(slot)


def earliest_reset_hint(model: str | None = None):
    """Shortest remaining cap delay for model (or pool-wide) as epoch-ms
    delta text, or None when nothing is capped.

    Accepts the routing-prefixed or canonical model id (see ``cap_model_key``).
    """
    pool = load_pool()
    now_ms = int(time.time() * 1000)
    best = None
    wanted = set(_cap_keys(model)) if model else None
    for slot, models in (pool.get("limits") or {}).items():
        if not isinstance(models, dict):
            continue
        for m, reset in models.items():
            if not isinstance(reset, (int, float)) or reset <= now_ms:
                continue
            if wanted is not None and m not in wanted:
                continue
            if best is None or reset < best:
                best = reset
    if best is None:
        return None
    return "try again in " + _format_cap_delay((best - now_ms) / 1000)


# ── legacy ClineKeyPool facade (kept for server.py compat) ───────
# The transport block calls pool.select/report_*/record_limit helpers; they
# now delegate to the JSON slot pool. Static-key-only operation is unchanged
# in behavior (key slots roll in file order, sticky current first).

class ClineKeyPool:
    """Compat facade over the login-slot pool: callers see bearer strings."""

    def __init__(self):
        self.keys: list[str] = _load_static_keys()
        self.current: str | None = None
        self.caps: dict = {}
        self._sync_caps()

    @property
    def ready(self) -> bool:
        return pool_account_count() > 0

    def reload(self):
        self.keys = _load_static_keys()
        self._sync_caps()

    def _sync_caps(self):
        pool = load_pool()
        now = time.time()
        self.caps = {}
        for slot, models in (pool.get("limits") or {}).items():
            for m, reset_ms in models.items():
                if reset_ms > now * 1000:
                    token = self._token_for_slot(slot)
                    if token:
                        self.caps[(token, m)] = reset_ms / 1000.0

    def _token_for_slot(self, slot: str):
        acct = _find_account(slot)
        return acct["token"] if acct else None

    def _slot_for_token(self, token: str):
        acct = next((a for a in all_accounts() if a.get("token") == token), None)
        return acct["slot"] if acct else None

    def select(self, model: str | None = None, skip=None):
        """First roll candidate not in ``skip``.

        ``skip`` matches EITHER form of an account: its slot name or its
        bearer token. Callers legitimately hold one or the other (the roll
        loops deal in tokens, ``is_capped_for`` in slot names), and matching
        on slot alone meant a token in ``skip`` was silently ignored — the
        same slot got re-selected forever.
        """
        skipped = set(skip or ())
        for acct in _roll_candidates(model)[0]:
            if acct.get("slot") in skipped or acct.get("token") in skipped:
                continue
            if acct.get("keySlot"):
                self.current = acct["token"]
                return acct["token"]
            # JSON slot: stale -> refresh-once pre-attempt (transient keeps
            # stale; dead grant skips the slot for this roll).
            if acct.get("refreshToken") and is_token_stale(acct):
                if refresh_account_in_place(acct) is False:
                    continue
            self.current = acct["token"]
            return acct["token"]
        return None

    def cap_reset_ms_for(self, slot: str, model: str | None):
        """Epoch-ms reset of ``slot``'s active free-limit cap on ``model``.

        None when uncapped / no model / cap already expired. One ``load_pool``
        (mtime-cached) call — the cap gate uses this instead of a bool so it
        can report the reset hint without a second pool read. Accepts the
        routing-prefixed or canonical model id (see ``cap_model_key``).
        """
        if not model:
            return None
        return _active_limit_for(load_pool(), slot, model, int(time.time() * 1000))

    def is_capped_for(self, slot: str, model: str | None) -> bool:
        """True if ``slot`` has an active free-limit cap for ``model``."""
        return self.cap_reset_ms_for(slot, model) is not None

    def slot_name_for(self, token: str) -> str | None:
        """Slot name behind a bearer token (for cap gating)."""
        return self._slot_for_token(token)

    def slot_count(self) -> int:
        """Total slots (JSON logins + static key slots)."""
        return pool_account_count()

    def refresh_once(self, key: str):
        """401/403 path: refresh the slot behind ``key`` exactly once.

        Returns the fresh bearer (retry once with it), False (dead grant —
        skip the slot), or None (key slot or transient fault — keep the
        stale bearer and its normal accounting).
        """
        slot = self._slot_for_token(key)
        if not slot:
            return None
        acct = _find_account(slot)
        if acct is None or acct.get("keySlot") or not acct.get("refreshToken"):
            return None
        pool_state = load_pool()
        live = next((a for a in pool_state.get("accounts", []) if a.get("slot") == slot), None)
        if live is None:
            return False
        outcome = refresh_account_in_place(live)
        if outcome is True:
            return live["token"]
        if outcome is False:
            return False
        return None

    def report_slot_success(self, key: str, model: str | None):
        """Success: clear the (slot, model) cap + bump usage + mark active."""
        slot = self._slot_for_token(key)
        if slot:
            record_slot_success(slot, model or "")
        self._sync_caps()

    def report_success(self, key: str):
        slot = self._slot_for_token(key)
        # requested-model cap clearing happens in the transport (it knows the
        # model); here clear pool-wide slot caps bookkeeping + sync.
        if slot:
            acct = _find_account(slot)
            if acct is not None and not acct.get("keySlot"):
                _mark_active_on_disk(slot)
        self._sync_caps()

    def report_rate_limit(self, key: str) -> bool:
        return True

    def report_capped(self, key: str, model: str | None, body_text: str | None):
        slot = self._slot_for_token(key)
        if slot:
            record_cap(slot, model or "", body_text or "")
        self._sync_caps()
        return parse_cap_delay_seconds(body_text)

    def cap_reset_at(self, key: str, model: str | None):
        slot = self._slot_for_token(key)
        if not slot:
            return None
        at = _active_limit_for(load_pool(), slot, model or "", int(time.time() * 1000))
        return at / 1000.0 if at is not None else None

    def nearest_reset_hint(self, model: str | None = None):
        return earliest_reset_hint(model)

    def report_http_status(self, key: str, status) -> bool:
        return True

    def record_latency(self, key: str, ms: int):
        return

    @staticmethod
    def is_cancelled(exc) -> bool:
        return False

    def report_cancelled(self, key=None, exc=None):
        return

    def getOrdered(self, model: str | None = None) -> list[str]:
        return [a["token"] for a in _roll_candidates(model)[0]]


pool = ClineKeyPool()
