"""Cline WorkOS device-flow auth (extension login, no paste).

Mechanical port of pi-freeflow's ``src/cline-device-auth.ts`` (which mirrors
the Cline SDK device authorize / poll / register / refresh flow): all network
goes through :mod:`urllib` here so the CLI stays dependency-free. No live
calls happen at import time.
"""
import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_WORKOS_BASE = "https://api.workos.com"
DEFAULT_API_BASE = "https://api.cline.bot"
WORKOS_TOKEN_PREFIX = "workos:"

CLINE_WORKOS_CLIENT_ID_ENV = "CLINE_WORKOS_CLIENT_ID"

# Bundled production client id (pi-freeflow
# BUNDLED_CLINE_WORKOS_CLIENT_ID). Env override wins when set and non-blank.
BUNDLED_CLINE_WORKOS_CLIENT_ID = "client_01K3A541FN8TA3EPPHTD2325AR"

# Cline's own sign-out route, for switching accounts in the browser.
CLINE_BROWSER_SIGNOUT_URL = "https://api.cline.bot/api/v1/auth/logout"

HTTP_TIMEOUT_SECS = 30


def resolve_workos_client_id() -> str:
    """Bundled client id, unless the env override is set and non-blank."""
    import os

    from_env = (os.environ.get(CLINE_WORKOS_CLIENT_ID_ENV) or "").strip()
    return from_env or BUNDLED_CLINE_WORKOS_CLIENT_ID


class ClineAuthError(Exception):
    """Device-login / register / refresh failure with optional status + code."""

    def __init__(self, message: str, status=None, error_code=None):
        super().__init__(message)
        self.status = status
        self.error_code = error_code


def is_workos_jwt(token: str) -> bool:
    """True for WorkOS JWT-shaped access tokens (base64url eyJ header)."""
    return bool(re.match(r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", (token or "").strip()))


def to_api_key(access_token: str) -> str:
    """Wire form of a Cline credential.

    Cline OAuth access tokens are WorkOS JWTs and must ride as ``workos:<jwt>``;
    dashboard keys (``clp_``) and other bearers go verbatim — prefixing them
    401s.
    """
    token = (access_token or "").strip()
    if token.lower().startswith(WORKOS_TOKEN_PREFIX):
        return token
    return f"{WORKOS_TOKEN_PREFIX}{token}" if is_workos_jwt(token) else token


def _to_epoch_ms(iso_datetime: str) -> int:
    from datetime import datetime, timezone

    text = (iso_datetime or "").strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        raise ValueError(f"Invalid expiresAt value: {iso_datetime}")


def _decode_jwt_exp_ms(access_token: str):
    """JWT ``exp`` claim in epoch-ms, or None when absent/undecodable."""
    try:
        parts = (access_token or "").split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1].replace("-", "+").replace("_", "/")
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.b64decode(payload_b64).decode("utf-8"))
        exp = payload.get("exp")
        return int(exp) * 1000 if isinstance(exp, (int, float)) and exp > 0 else None
    except Exception:
        return None


def derive_expiry(explicit_ms, access_token: str) -> int:
    """Expiry derivation order: explicit server expiry -> JWT exp claim ->
    already-expired (forces refresh)."""
    if isinstance(explicit_ms, (int, float)) and explicit_ms and explicit_ms > 0:
        return int(explicit_ms)
    decoded = _decode_jwt_exp_ms(access_token or "")
    if decoded is not None:
        return decoded
    return int(time.time() * 1000) - 1


def _join_url(base: str, path: str) -> str:
    return (base or "").rstrip("/") + path


def _post_form(url: str, fields: dict, timeout=HTTP_TIMEOUT_SECS) -> tuple[int, dict]:
    """POST urlencoded form; returns (status, parsed-json-or-{})."""
    body = urllib.parse.urlencode(fields).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        status = e.code
    try:
        payload = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        payload = {}
    return status, payload if isinstance(payload, dict) else {}


def _post_json(url: str, payload: dict, timeout=HTTP_TIMEOUT_SECS) -> tuple[int, dict, str]:
    """POST JSON; returns (status, parsed-json-or-{}, raw-text)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        status = e.code
    try:
        parsed = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        parsed = {}
    return status, parsed if isinstance(parsed, dict) else {}, raw


def start_device_auth(workos_base: str = DEFAULT_WORKOS_BASE) -> dict:
    """Begin the WorkOS device flow. Returns {deviceCode, userCode,
    verificationUri, verificationUriComplete?, expiresIn, interval}."""
    status, payload = _post_form(
        _join_url(workos_base, "/user_management/authorize/device"),
        {"client_id": resolve_workos_client_id()},
    )
    if status < 200 or status >= 300:
        detail = payload.get("error_description") if isinstance(payload.get("error_description"), str) else ""
        raise ClineAuthError(
            f"Device authorization failed: {status}" + (f" - {detail}" if detail else ""),
            status=status,
            error_code=payload.get("error") if isinstance(payload.get("error"), str) else None,
        )
    if not (isinstance(payload.get("device_code"), str)
            and isinstance(payload.get("user_code"), str)
            and isinstance(payload.get("verification_uri"), str)):
        raise ClineAuthError("Invalid WorkOS device authorization response")
    out = {
        "deviceCode": payload["device_code"],
        "userCode": payload["user_code"],
        "verificationUri": payload["verification_uri"],
        "expiresIn": _to_seconds(payload.get("expires_in"), 300),
        "interval": _to_seconds(payload.get("interval"), 5),
    }
    if isinstance(payload.get("verification_uri_complete"), str):
        out["verificationUriComplete"] = payload["verification_uri_complete"]
    return out


def _to_seconds(value, fallback: int) -> int:
    return int(value) if isinstance(value, (int, float)) and value > 0 else fallback


def poll_device_token(workos_base: str = DEFAULT_WORKOS_BASE, device_code: str = "",
                      interval_seconds: int = 5, timeout_secs: int = HTTP_TIMEOUT_SECS,
                      max_wait_ms=None) -> dict:
    """Poll until the user approves/denies or the code expires.

    Returns {accessToken, refreshToken}. Raises ClineAuthError on
    denied/expired/cancelled, generic Error on timeout.
    """
    if not device_code:
        raise ValueError("device_code is required")
    deadline = time.time() * 1000 + (max_wait_ms if max_wait_ms is not None
                                     else interval_seconds * 1000 * 60 * 10)
    interval = max(1, int(interval_seconds))
    url = _join_url(workos_base, "/user_management/authenticate")
    fields = {
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "device_code": device_code,
        "client_id": resolve_workos_client_id(),
    }
    while True:
        try:
            status, payload = _post_form(url, fields, timeout=timeout_secs)
            transport_ok = True
        except Exception:
            # Transient transport blip: keep polling until the deadline.
            if time.time() * 1000 > deadline:
                raise ClineAuthError("WorkOS device authorization timed out")
            _sleep_remaining(deadline, interval)
            continue
        _ = transport_ok
        if 200 <= status < 300:
            if not (isinstance(payload.get("access_token"), str)
                    and isinstance(payload.get("refresh_token"), str)):
                raise ClineAuthError("Invalid WorkOS token response")
            return {"accessToken": payload["access_token"],
                    "refreshToken": payload["refresh_token"]}
        err = payload.get("error")
        if err == "authorization_pending":
            pass  # keep polling at the current interval
        elif err == "slow_down":
            interval += 1
        elif err in ("access_denied", "expired_token", "invalid_grant"):
            detail = payload.get("error_description") if isinstance(
                payload.get("error_description"), str) else f"Device login refused ({err})"
            raise ClineAuthError(detail, status=status,
                                 error_code=err if isinstance(err, str) else None)
        elif status >= 500 or status == 429:
            pass  # transient WorkOS blip: keep polling until the deadline
        else:
            detail = payload.get("error_description") if isinstance(
                payload.get("error_description"), str) else f"WorkOS token polling failed: {status}"
            raise ClineAuthError(detail, status=status,
                                 error_code=err if isinstance(err, str) else None)
        if time.time() * 1000 > deadline:
            raise ClineAuthError("WorkOS device authorization timed out")
        _sleep_remaining(deadline, interval)


def _sleep_remaining(deadline_ms: float, interval_seconds: int):
    """Sleep before the next poll, never past the device code's own expiry."""
    ms = max(0, min(interval_seconds * 1000, deadline_ms - time.time() * 1000))
    if ms > 0:
        time.sleep(ms / 1000.0)


def _to_credentials(data: dict, fallback_refresh=None) -> dict:
    refresh = data.get("refreshToken") or fallback_refresh
    if not refresh:
        raise ClineAuthError("Token response did not include a refresh token")
    try:
        explicit = _to_epoch_ms(data.get("expiresAt") or "")
    except ValueError:
        explicit = None
    user_info = data.get("userInfo") or {}
    return {
        "access": data.get("accessToken"),
        "refresh": refresh,
        "expires": derive_expiry(explicit, data.get("accessToken") or ""),
        "accountId": user_info.get("clineUserId") or None,
        "email": user_info.get("email") or None,
        "metadata": {"tokenType": data.get("tokenType")},
    }


def _read_token_data(payload: dict, message: str) -> dict:
    data = payload.get("data") or {}
    if not payload.get("success") or not data.get("accessToken"):
        raise ClineAuthError(f"Invalid token response: {message}")
    return data


def register_cline_token(api_base: str = DEFAULT_API_BASE,
                         access_token: str = "", refresh_token: str = "") -> dict:
    """Exchange WorkOS tokens for Cline credentials. Returns credentials dict
    {access, refresh, expires, accountId?, email?, metadata}."""
    if not access_token or not refresh_token:
        raise ValueError("access_token and refresh_token are required")
    status, payload, raw = _post_json(
        _join_url(api_base, "/api/v1/auth/register"),
        {"accessToken": access_token, "refreshToken": refresh_token},
    )
    if not (200 <= status < 300):
        raise ClineAuthError(
            f"Token registration failed: {status}" + (f" - {raw[:200]}" if raw else ""),
            status=status,
        )
    return _to_credentials(_read_token_data(payload, "register"))


def refresh_cline_token(api_base: str = DEFAULT_API_BASE, refresh_token: str = "") -> dict:
    """Refresh Cline credentials. Returns credentials dict. Raises
    ClineAuthError with error_code 'invalid_grant' (or status 400/401) when
    the grant is dead and the slot must be logged in again."""
    if not refresh_token:
        raise ValueError("refresh_token is required")
    status, payload, raw = _post_json(
        _join_url(api_base, "/api/v1/auth/refresh"),
        {"refreshToken": refresh_token, "grantType": "refresh_token"},
    )
    if not (200 <= status < 300):
        code = payload.get("error") if isinstance(payload.get("error"), str) else None
        text = (raw or "").lower()
        if status in (400, 401) or code == "invalid_grant" or "invalid_grant" in text:
            raise ClineAuthError(f"Token refresh rejected (dead grant): {status}",
                                 status=status, error_code="invalid_grant")
        raise ClineAuthError(
            f"Token refresh failed: {status}" + (f" - {raw[:200]}" if raw else ""),
            status=status, error_code=code,
        )
    return _to_credentials(_read_token_data(payload, "refresh"), refresh_token)
