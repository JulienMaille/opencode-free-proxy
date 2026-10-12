#!/usr/bin/env python3
"""Cline device-login CLI: save a per-user login slot into cline-pool.json.

Usage:
    python cline_login.py [--slot default]

Flow: startDeviceAuth -> print user_code + verification URIs -> poll until
approved/cancelled/timeout -> registerClineToken -> addAccount(slot, ...).

Exit codes are non-zero on failure with an actionable message. Full tokens
are never printed (last-4 only).
"""
import argparse
import sys
import time

from cline_device_auth import (
    ClineAuthError,
    poll_device_token,
    register_cline_token,
    start_device_auth,
    to_api_key,
)
from cline_pool import add_account, redacted_token


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Log a Cline account into cline-pool.json (device flow)")
    p.add_argument("--slot", default="default", help="Login slot name (default: default)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    slot = (args.slot or "").strip() or "default"
    try:
        auth = start_device_auth()
    except ClineAuthError as e:
        print(f"FAILED to start device login: {e} (status={e.status})", flush=True)
        return 1
    except Exception as e:
        print(f"FAILED to start device login: {e}", flush=True)
        return 1
    print("Cline device login — approve in your browser:", flush=True)
    print(f"  code: {auth['userCode']}", flush=True)
    print(f"  open: {auth['verificationUri']}", flush=True)
    if auth.get("verificationUriComplete"):
        print(f"  open (pre-filled): {auth['verificationUriComplete']}", flush=True)
    print(f"Waiting for approval (code expires in ~{int(auth.get('expiresIn', 300))}s)…", flush=True)
    max_wait_ms = max(1000, int(auth.get("expiresIn", 300)) * 1000)
    try:
        tokens = poll_device_token(device_code=auth["deviceCode"],
                                   interval_seconds=int(auth.get("interval", 5)),
                                   max_wait_ms=max_wait_ms)
    except ClineAuthError as e:
        code = e.error_code or ""
        if code in ("access_denied",):
            print("FAILED: login denied in the browser — re-run and approve the request.", flush=True)
        elif code in ("expired_token", "invalid_grant"):
            print("FAILED: device code expired — re-run python cline_login.py for a fresh code.", flush=True)
        elif code == "cancelled":
            print("FAILED: login cancelled.", flush=True)
        elif "timed out" in str(e).lower():
            print("FAILED: approval timed out — re-run python cline_login.py for a fresh code.", flush=True)
        else:
            print(f"FAILED while waiting for approval: {e}", flush=True)
        return 1
    except Exception as e:
        print(f"FAILED while waiting for approval: {e}", flush=True)
        return 1
    try:
        creds = register_cline_token(access_token=tokens["accessToken"],
                                     refresh_token=tokens["refreshToken"])
    except ClineAuthError as e:
        print(f"FAILED to register login with Cline: {e} (status={e.status})", flush=True)
        return 1
    except Exception as e:
        print(f"FAILED to register login with Cline: {e}", flush=True)
        return 1
    wire_token = to_api_key(creds["access"])
    try:
        pool = add_account(slot, wire_token, {
            "refreshToken": creds.get("refresh"),
            "expiresAt": creds.get("expires"),
            "accountId": creds.get("accountId"),
            "email": creds.get("email"),
        })
    except ValueError as e:
        print(f"FAILED to save slot: {e}", flush=True)
        return 1
    saved = next((a for a in pool.get("accounts", []) if a.get("slot") == slot), {})
    email = saved.get("email") or creds.get("email") or "(no email reported)"
    print(f"Saved Cline login slot [{slot}] ({email}), token {redacted_token(wire_token)}", flush=True)
    print(f"Pool now holds {len(pool.get('accounts', []))} slot(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
