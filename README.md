# opencode-free-proxy

Free AI models from [OpenCode](https://opencode.ai) (the Zen API free tier), exposed as standard OpenAI and Anthropic APIs. Works with any tool that speaks those formats: opencode CLI, Cursor, Claude Code, Cline, aider, [oh-my-pi](docs/omp.md), raw `curl`, etc.

## Quick start (Windows)

```bat
pip install -r requirements.txt
start.bat
```

A standalone `dist\opencode-free-proxy.exe` (~12 MB) is also provided — same defaults (port 6446 + proxy pool), no Python required. Rebuild it with `pyinstaller --onefile --exclude-module fastapi server.py` (see the full exclude list in git history).

`start.bat` runs `python server.py` — the defaults are `--port 6446 --proxy-pool`, so the rotating SOCKS5 proxy pool is on by default (recommended — free-tier rate limits per IP are aggressive). Use `start-simple.bat` (equivalent to `python server.py --no-proxy-pool`) for direct connections without a proxy.

`stop.bat` kills the server on port 6446.

Server is at `http://localhost:6446`.

## CLI arguments

```bash
python server.py                    # default: port 6446 + proxy pool
python server.py --no-proxy-pool    # direct connections
python server.py --port 8080 --proxy socks5://127.0.0.1:9150
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | `6446` | Listen port |
| `--host` | `0.0.0.0` | Listen host |
| `--proxy` | _(none)_ | Static SOCKS5 proxy (e.g. `socks5://127.0.0.1:9150`) |
| `--proxy-pool` | on | Rotating SOCKS5 proxy pool with transport-failure and per-proxy 429 rotation (`--no-proxy-pool` disables) |
| `--api-key` | _(none)_ | API key for client auth (see env vars) |

## Environment variables

| Variable | What |
|----------|------|
| `PORT` / `HOST` | Override listen port/host |
| `SOCKS5_PROXY` | Static SOCKS5 proxy (used when the proxy pool is off) |
| `OPENCODE_PROXY_POOL` | `0`/`false` to disable the proxy pool via env |
| `OPENCODE_PROXY_PORT_FILTER` | Enabled by default; set `0`/`false` to allow proxy ports other than `4145` and `1080` |
| `LOCAL_KEY` / `API_KEY` | API key for client auth; if unset, the server accepts any request |
| `OPENCODE_ENABLE_EXA=1` | Enables the `websearch` tool for opencode CLI (set in `start.bat`) |

## Models

The model list is fetched dynamically from the Zen API (`opencode.ai/zen/v1/models`) and enriched with context limits / modalities from `models.dev`. It refreshes every 5 hours. Typical free models:

- `space-bunny-free` (text + image, 1M context, 512K output — best specs)
- `mimo-v2.6-flash-free` / `mimo-v2.5-free` (text + image, reasoning)
- `muse-spark-1.3-contributor-free` / `muse-spark-1.2-contributor-free` (text + image, routed to `/zen/v1/responses`)
- `nemotron-3.5-lightning-free` / `nemotron-3-ultra-free` (text, reasoning)
- `big-pickle` (reasoning model, text-only)
- `fledge-alpha-free` (text, 1M context, 131K output)
- `longcat-2.5-preview-free`, `stealth/space-bunny-alpha` (auto-discovered)

> Removed from the curated set (blocked upstream, plus a persistent dead-model denylist in `data/dead-models.json` for retired `-free` ids):
> `deepseek-v4-flash-free` — listed by the Zen API but currently returns
> `Error from provider (Console): Upstream request failed: Model is unavailable.`;
> `north-mini-code-free` — opaque `400 Provider returned error` on multi-turn
> tool calls; `ling-3.0-flash-free` — no longer on the free tier upstream
> (404: "use this slug instead: inclusionai/ling-3.0-flash"); and
> `ling-3.0-flash-fin-free` — FreeTier-gated on the chat path on all exits
> (403). These four ids are filtered from discovery and every
> listing/request path, so clients never see them.

## API

### OpenAI format — `POST /v1/chat/completions`

```bash
curl http://localhost:6446/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "big-pickle",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Streaming responses are forwarded as SSE as soon as the upstream emits them.
Use `"stream": true` for long-thinking models; the proxy allows up to 300
seconds of silence between streaming events while keeping the shorter timeout
for buffered requests.

### Anthropic format — `POST /v1/messages`

```bash
curl http://localhost:6446/v1/messages \
  -H "x-api-key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "big-pickle",
    "system": "You are helpful.",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1024,
    "stream": true
  }'
```

### Ollama format — `POST /api/chat`, `POST /api/generate`

```bash
curl http://localhost:6446/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "big-pickle",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

`GET /api/tags` mirrors `/v1/models` in Ollama's shape; `GET /api/version`,
`GET /api/ps`, `POST /api/show`, and `GET /` (`Ollama is running`) are also
served. `stream` defaults to `true` (Ollama convention) with NDJSON chunks;
pass `"stream": false` for a single JSON object. The `think` flag maps to the
muse reasoning effort (`false` disables thinking); `options.num_predict`
maps to `max_tokens`. Structured output via `format` is not translated.


### Other endpoints

| Method | Path | What |
|--------|------|------|
| `GET` | `/v1/models` | List models (includes limits + modalities) |
| `GET` | `/health` | Health + version |
| `GET` | `/api/tags` | List models (Ollama shape) |
| `POST` | `/api/chat` | Chat (Ollama shape, NDJSON stream) |
| `POST` | `/api/generate` | Single-prompt completion (Ollama shape) |

### Auth

Both `Authorization: Bearer KEY` and `x-api-key: KEY` work on all endpoints.

## How it works

```
Your tool (opencode CLI, Cursor, curl, etc.)
        │
        ▼
  opencode-free-proxy        ← translates formats, manages sessions & proxies
        │
        ▼  HTTPS
  opencode.ai/zen/v1/       ← free tier API
```

> **Sessions**: the proxy hashes the message prefix to reuse the upstream session, so multi-turn conversations stay coherent.

## Cline (free tier + key slots)

In addition to the Zen-tier models, the
proxy serves Cline models directly from
[Cline](https://api.cline.bot/api/v1) (OpenAI-compatible, SSE streaming,
direct-only — never through the SOCKS pool). Address a Cline model with the
`cline/` prefix, e.g. `cline/cline-free/kimi-k3`. Model IDs pass through
unchanged — the `cline/` routing prefix is stripped but the upstream ID
(including the `cline-free/` segment) is never rewritten; requests send IDs
verbatim.

### Free tier: device login

The free tier uses per-user login slots (a faithful port of pi-freeflow's
Cline FREE tier). Log in with the device flow — no pasting tokens:

```bash
python cline_login.py [--slot default]
```

This prints a code + browser URLs; approve in the browser and the login is
saved into `cline-pool.json` (repo root, gitignored, 0600, with a
largest-wins `.bak` recovery copy). Log in once per Cline account; requests
rotate across all saved slots (active-slot first, daily-capped slots last —
capped slots are still tried). A slot that serves a turn becomes active, its
usage counters bump, and its cap clears. Stale bearers refresh once
pre-attempt (transient faults keep the stale bearer); a 401/403 refreshes
once then retries, skipping the slot on a dead grant. Duplicate accounts are
refused (same WorkOS account id, else email, else identical bearer).

Free canonical IDs (served by login slots, daily caps per account):

- `cline/cline-free/deepseek-v4.1-flash`
- `cline/cline-free/muse-spark-1.3-contributor`
- `cline/z-ai/glm-5.3-flash`
- `cline/cline-free/kimi-k3`
- `cline/cline-free/solar-pro4`

Friendly slugs (`cline/deepseek-v4.1-flash`, `cline/solar-pro4`,
`cline/kimi-k3`) resolve to the canonical free IDs. Every Cline request
carries the full desktop-client identity set (`HTTP-Referer:
https://cline.bot`, `X-Title: Cline`, `X-IS-MULTIROOT: false`,
`X-CLIENT-TYPE: cline-desktop`, `X-CLIENT-VERSION: 3.5.54`, `User-Agent:
Cline/3.5.54`) — required: `cline-free/*` answers 403 to a bare bearer.

```bash
curl http://localhost:6446/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cline/cline-free/kimi-k3",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### Key slots (paying keys)

`cline-api-keys.txt` (same folder as `server.py`) still works: one key per
line (any non-empty line not starting with `#`; `workos:`, JWT, `clp_`, and
`sk_` forms accepted). Keys auto-import as `key-N` slots (no refresh/expiry),
merged with the JSON pool and deduped by token. Key slots serve the
ClinePass + usage-billing IDs:

- `cline/cline-pass/glm-5.3`, `cline-pass/glm-5.3-flash`,
  `cline-pass/kimi-k3`, `cline-pass/deepseek-v4-pro`,
  `cline-pass/deepseek-v4.1-flash`, `cline-pass/mimo-v2.5`,
  `cline-pass/mimo-v2.5-pro`, `cline-pass/minimax-m3`,
  `cline-pass/muse-spark-1.3-contributor`, `cline-pass/qwen3.8-max`,
  `cline-pass/qwen3.7-max`, `cline-pass/qwen3.7-plus`
- `cline/anthropic/claude-sonnet-4-6`, `cline/google/gemini-2.5-pro`,
  `cline/deepseek/deepseek-chat`, `cline/openai/gpt-4o`,
  `cline/minimax/minimax-m2.5` (usage billing)

The paying IDs keep working as passthrough for key slots. Retired IDs
(`glm-5.2`, `kimi-k2.7-code`, `kimi-k2.6`, `deepseek-v4-flash`) have no
aliases and pass through as-is so Cline errors on them itself.
`GET /v1/models` dynamically discovers IDs from the Cline `/models` endpoint
when a slot is loaded, merged with the hardcoded list (which carries the
`cline-pass/*` IDs the live endpoint omits) — never wiped on empty.
`POST /v1/messages` (Anthropic) is supported the same way (convert to OpenAI, call Cline, convert back).

Error map: `401/403` auth, `429` rate-limit (a body saying `free limit
reached on model ... try again in ...` parks that slot+requested-model until
the parsed reset; bare numbers mean hours), `402` exhausted (hint: top up at
app.cline.bot / not subscribed), `5xx` server.

Check slot status with:

```
curl http://localhost:6446/health   # cline_slots: <count>, cline_models: [...]
curl http://localhost:6446/v1/models # cline/* entries list available Cline routes
```

- **Proxy pool**: on by default, SOCKS5 proxies are scraped from public lists, verified, and rotated. A proxy gets one transport retry, then is blacklisted after its second transport failure; `429` responses temporarily skip the current proxy so the caller can retry through another IP.
- **Auth headers**: the proxy adds the `x-opencode-*` headers the Zen API requires (discovered by reverse-engineering the opencode binary):

```
Authorization: Bearer public
User-Agent: opencode/1.15.0 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.13
x-opencode-client: cli
x-opencode-project: global
x-opencode-request: msg_<unique_id>
x-opencode-session: ses_<unique_id>
```

## Client configuration

Point any OpenAI/Anthropic-compatible tool at `http://127.0.0.1:6446/v1` (OpenAI)
or `http://127.0.0.1:6446` (Anthropic `/v1/messages`). No auth is required
unless you set an API key (see env vars above).

### opencode

Add a custom provider to `~/.config/opencode/opencode.jsonc` (or a project
`opencode.jsonc`). Use `@ai-sdk/openai-compatible` and declare modalities per
model so opencode knows which accept images — the `capabilities` key is
**not** read by opencode; use `modalities` (+ `attachment: true` for image
models):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "local-openai/big-pickle",
  "provider": {
    "local-openai": {
      "name": "Local OpenAI",
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:6446/v1" },
      "models": {
        "big-pickle": {
          "name": "Big Pickle",
          "id": "big-pickle",
          "reasoning": true,
          "modalities": { "input": ["text"], "output": ["text"] },
          "limit": { "context": 200000, "output": 32000 }
        },
        "mimo-v2.5-free": {
          "name": "Mimo v2.5",
          "id": "mimo-v2.5-free",
          "reasoning": true,
          "attachment": true,
          "modalities": { "input": ["text", "image"], "output": ["text"] },
          "limit": { "context": 200000, "output": 32000 }
        }
      }
    }
  }
}
```

Notes:

- The `id` must match the model the proxy serves (Zen free slugs like
  `big-pickle`, or `cline/<id>` for Cline models). The picker key is
  `local-openai/<id>`.
- `modalities.input` with `"image"` (plus `attachment: true`) is what makes
  opencode allow image attachments; models without it are text-only and opencode
  will refuse images for them.
- Config is read **once at startup** — restart opencode after editing it.
- select the model with `/models`.

### oh-my-pi (omp)

Full step-by-step config (provider + models in `~/.omp/agent/models.yml`,
per-role thinking levels in `~/.omp/agent/config.yml`, verify commands) lives in
[`docs/omp.md`](docs/omp.md). In short, reference the proxy's models as:

```yaml
providers:
  opencode-local:
    baseUrl: http://127.0.0.1:6446/v1
    auth: none
    api: openai-completions
    # ... compat knobs (see docs/omp.md) ...
    models:
      - id: nemotron-3.5-lightning-free
        reasoning: true
        input: [text]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 1000000
        maxTokens: 128000
```

then point roles at models with a `:low`/`:high`/`:max`/`:auto` reasoning suffix:

```yaml
modelRoles:
  default: opencode-local/space-bunny-free:auto
  vision: opencode-local/space-bunny-free:auto
```

Cline models ride the same provider (`opencode-local/cline/cline-free/kimi-k3:auto`).

## License

MIT
