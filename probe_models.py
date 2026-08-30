"""Probe each discovered free model with a super-short prompt.

For every model the local proxy serves (from GET /v1/models), send a tiny
non-streaming chat completion and classify the outcome so you can see at a
glance which models still work for free:

    OK            got a real completion (may be reasoning-only / truncated)
    RATE LIMITED  free-tier quota 429
    DEAD          promotion ended / ModelError (retired upstream)
    UNKNOWN       no longer discovered
    REGION        geo-block currently
    ERROR/1xx–5xx explicit failure
    TIMEOUT       no answer within the per-request window
"""
import json
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:6446"
TIMEOUT = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
PROMPT = "hi"
MAX_TOKENS = 16


def classify(status: int, payload) -> tuple[str, str]:
    text = payload.get("error", {}).get("message") or payload.get("error", {}).get("type") or ""
    text_l = f"{text} {payload.get('error', {})}".lower()
    if status == 400 and "unknown model" in text_l:
        return "UNKNOWN", text
    if payload.get("error", {}).get("type") == "modelerror" or "promotion has ended" in text_l:
        return "DEAD", text
    if status == 429 or "rate limit" in text_l or payload.get("error", {}).get("type") == "rate_limit_error":
        return "RATE LIMITED", text
    if status == 200:
        choices = payload.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or "").strip()
            detail = f"{len(content)}c/{len(reasoning)}r"
            if not content:
                return "OK(reasoning only)", detail
            return "OK", detail
        return "OK(no choices)", ""
    if status >= 500:
        return f"ERROR{status}", text
    return f"ERROR{status}", text


def main():
    with httpx.Client(timeout=TIMEOUT) as client:
        try:
            r = client.get(f"{BASE}/v1/models")
            r.raise_for_status()
            models = [m["id"] for m in r.json().get("data", [])]
        except Exception as e:
            print(f"FAIL could not list models from {BASE}: {e}")
            return 1

    print(f"Probing {len(models)} models via {BASE} (prompt={PROMPT!r}, max_tokens={MAX_TOKENS}, timeout={TIMEOUT}s)\n")
    dead = []
    rate = []
    for m in models:
        started = time.time()
        body = {
            "model": m,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False,
            "max_completion_tokens": MAX_TOKENS,
        }
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                resp = client.post(f"{BASE}/v1/chat/completions", json=body)
                try:
                    data = resp.json()
                except json.JSONDecodeError:
                    data = {"error": {"message": f"non-JSON body: {resp.text[:120]!r}"}}
            state, detail = classify(resp.status_code, data)
        except httpx.TimeoutException:
            state, detail = "TIMEOUT", f">{TIMEOUT:.0f}s"
        except Exception as e:
            state, detail = "ERROR", f"{type(e).__name__}: {e}"
        elapsed = time.time() - started
        tag = f"[{state}]"
        print(f"  {m:<32} {tag:<18} {detail:<24} {elapsed:5.1f}s")
        if state == "DEAD":
            dead.append(m)
        elif state == "RATE LIMITED":
            rate.append(m)

    print()
    if dead:
        print(f"DEAD ({len(dead)}):   {', '.join(dead)}")
    if rate:
        print(f"RATE LIMITED ({len(rate)}): {', '.join(rate)}")
    if not dead and not rate:
        print("All models responded (may need a region/retry for exact grades).")
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())