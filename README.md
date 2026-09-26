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
- `jev-1.13-free`, `ling-3.0-flash-fin-free` (auto-discovered)

> Removed from the curated set (still auto-discovered if the Zen API lists them):
> `deepseek-v4-flash-free` — listed by the Zen API but currently returns
> `Error from provider (Console): Upstream request failed: Model is unavailable.`;
> `north-mini-code-free` — opaque `400 Provider returned error` on multi-turn
> tool calls; and `ling-3.0-flash-free` — no longer on the free tier upstream
> (404: "use this slug instead: inclusionai/ling-3.0-flash"). These three ids are
> filtered from discovery and every listing/request path, so clients never see them.

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

## NVIDIA NIM models

In addition to the free Zen-tier models, the proxy can serve models directly
from [NVIDIA NIM](https://integrate.api.nvidia.com). Address an NVIDIA model with
the `nvidia/` (or `nvimin/`) prefix, e.g. `nvidia/deepseek-v4-pro-0813`.

This requires `nvidia-api-keys.txt` in the same folder as `server.py` — one
`nvapi-...` key per line. Add your own keys there (the file ships with a set of
free-tier keys that are validated and cleaned periodically). The proxy auto-rotates
that file on each request.

```bash
curl http://localhost:6446/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/deepseek-v4-pro-0813",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Supported model aliases (also work as `nvidia/<alias>`):

- `nvidia/deepseek-v4-pro-0813`
- `nvidia/deepseek-v4-flash-0731`
- `nvidia/llama-3.2-11b-vision-instruct` (also exposed to the Anthropic
  `POST /v1/messages` endpoint, with OpenAI↔Anthropic conversions applied)

Any model id that already contains a `/` after the prefix (e.g.
`nvidia/deepseek-ai/deepseek-v4-pro-0813`) is passed through to the NIM slug
verbatim, so the full catalog of 80+ NIM models is usable.

### NVIDIA key rotation

NVIDIA free-tier `nvapi-*` keys are capped at ~40 RPM/account and are gated
per-model on the free tier. The proxy rotates keys automatically:

- Keys are handed out round-robin from `nvidia-api-keys.txt`.
- Only well-formed keys load: NVIDIA `nvapi-*` keys are a **fixed 70 characters**
  (the `nvapi-` prefix plus a 64-char body of `[A-Za-z0-9_-]`). Anything shorter
  or longer is skipped at load time and rejected by the key hunter's validator.
- A request that hits a **rate limit (429 / `FreeUsageLimitError`)** immediately
  cools that key down. The cooldown grows **exponentially per key**
  (60s → 120s → 240s → … capped at 10 minutes), so a key that keeps getting
  rate-limited backs off progressively and stops being offered, letting the
  rest of the pool stay usable.
- If **12 consecutive 429s** happen (the whole pool is saturated by the shared
  free-tier quota), the proxy fails fast with a clear `rate limit exceeded`
  error instead of silently looping every key with backoff for minutes.
- A per-key **404** (this key is not entitled to the model) advances to the next
  key immediately so the whole pool is tried before failing.
- NVIDIA read/stream timeouts are generous (600s) so long-thinking models
  (deepseek-v4-pro / -flash) that stay silent for minutes aren't killed mid-reasoning.
- `POST /v1/messages` (Anthropic) is supported: the proxy converts to OpenAI
  format, calls NIM, and converts the result back to Anthropic SSE.

Check key status with:

```
curl http://localhost:6446/health   # nvidia_keys: <count>, nvidia_models: [...]
curl http://localhost:6446/v1/models # nvidia/* entries list available NIM routes
```

> Free-tier NIM keys have tight per-model quotas; for heavy/long-thinking
> models a 404 may mean quota exhaustion for that model on that key, in which
> case a different key in the file is used. Replace `nvidia-api-keys.txt` with
> your own paid-tier keys for higher limits.

## AMD Radeon TokenFactory

In addition to the Zen-tier and NVIDIA NIM models, the proxy can serve models
directly from [AMD Radeon TokenFactory](https://developer.amd.com.cn/radeon).
Address an AMD model with the `amd/` (or `radeon/`) prefix, e.g.
`amd/DeepSeek-V4-Flash`.

This requires `amd-api-keys.txt` in the same folder as `server.py` — one
`rc-...` key per line. Keys are a fixed 51 characters: `rc-` plus a 48-char
hex body (`[0-9a-f]`). The proxy auto-rotates that file on each request.

```bash
curl http://localhost:6446/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amd/DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Supported canonical IDs (also work as `amd/<id>`):

- `amd/DeepSeek-V4-Flash` (Cline `defaultModelId`)
- `amd/DeepSeek-V4-Flash-Vision-Exp`
- `amd/DeepSeek-V4.1-Flash`
- `amd/GLM-5.3-Flash`
- `amd/MinerU2.5-Pro`
- `amd/MiniCPM5-2B`
- `amd/Qwen3.8-27B`
- `amd/Qwen3.8-Flash-Next`

Friendly slugs (`amd/deepseek-v4-flash`) and legacy IDs
(`DeepSeek-V4-Flash-0731`, `MiniCPM-V46`, `MiniCPM5-1B`, `Qwen3.6-35B-A3B`)
resolve to the canonical verbatim TokenFactory IDs. `GET /v1/models`
dynamically discovers IDs from the TokenFactory `/models` endpoint when a key
is loaded, falling back to the hardcoded list on failure. `POST /v1/messages`
(Anthropic) is supported the same way as the NVIDIA route.

Check key status with:

```
curl http://localhost:6446/health   # amd_keys: <count>, amd_models: [...]
curl http://localhost:6446/v1/models # amd/* entries list available TokenFactory routes
```

## Cline

In addition to the Zen-tier, NVIDIA NIM, and AMD TokenFactory models, the
proxy can serve models directly from
[Cline](https://api.cline.bot/api/v1) (OpenAI-compatible, SSE streaming).
Address a Cline model with the `cline/` prefix, e.g.
`cline/cline-pass/glm-5.3`. Model IDs pass through unchanged — the `cline/`
routing prefix is stripped but the upstream ID (including the `cline-pass/`
segment) is never rewritten.

This requires `cline-api-keys.txt` in the same folder as `server.py` — one
key per line (any non-empty line not starting with `#`; the Cline key format
is not strict). Get a key at app.cline.bot under Settings -> API Keys. The
proxy sends the desktop-client identity headers (`X-CLIENT-TYPE:
cline-desktop`, `User-Agent: Cline/3.5.54`) on every Cline request and
auto-rotates the key file on each request.

```bash
curl http://localhost:6446/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cline/cline-pass/glm-5.3",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Supported canonical IDs (also work as `cline/<id>`):

- `cline/cline-pass/glm-5.3`
- `cline/cline-pass/glm-5.3-flash`
- `cline/cline-pass/kimi-k3`
- `cline/cline-pass/deepseek-v4-pro`
- `cline/cline-pass/deepseek-v4.1-flash`
- `cline/cline-pass/mimo-v2.5`
- `cline/cline-pass/mimo-v2.5-pro`
- `cline/cline-pass/minimax-m3`
- `cline/cline-pass/muse-spark-1.3-contributor`
- `cline/cline-pass/qwen3.8-max`
- `cline/cline-pass/qwen3.7-max`
- `cline/cline-pass/qwen3.7-plus`
- `cline/anthropic/claude-sonnet-4-6` (usage billing)
- `cline/google/gemini-2.5-pro` (usage billing)
- `cline/deepseek/deepseek-chat` (usage billing)
- `cline/openai/gpt-4o` (usage billing)
- `cline/minimax/minimax-m2.5` (usage billing)

Friendly slugs (`cline/glm-5.3`, `cline/kimi-k3`) and bare-suffix IDs resolve
to the canonical Cline IDs. Retired IDs (`glm-5.2`, `kimi-k2.7-code`,
`kimi-k2.6`, `deepseek-v4-flash`) have no aliases and pass through as-is so
Cline errors on them itself. `GET /v1/models` dynamically discovers IDs from
the Cline `/models` endpoint when a key is loaded, merged with the hardcoded
list (which carries the `cline-pass/*` IDs the live endpoint omits) —
never wiped on empty. `POST /v1/messages` (Anthropic) is supported the same
way as the AMD route.

Error notes: a `402` means the key has an empty balance (hint: top up at
app.cline.bot / not subscribed). A `429` whose body says `free limit reached
on model ... try again in ...` parks that (key, model) pair until the parsed
reset time; capped keys sort last but are still tried.

Check key status with:

```
curl http://localhost:6446/health   # cline_keys: <count>, cline_models: [...]
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
        },
        "nvidia/kimi-k3": {
          "name": "Kimi K3 (NVIDIA)",
          "id": "nvidia/kimi-k3",
          "reasoning": true,
          "attachment": true,
          "modalities": { "input": ["text", "image"], "output": ["text"] },
          "limit": { "context": 131072, "output": 32768 }
        }
      }
    }
  }
}
```

Notes:

- The `id` must match the model the proxy serves (Zen free slugs like
  `big-pickle`, or `nvidia/<alias>` for NIM models). The picker key is
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

NVIDIA models ride the same provider (`opencode-local/nvidia/kimi-k3:auto`).

## License

MIT
