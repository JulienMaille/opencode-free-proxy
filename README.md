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

## Deploying on a Free Cloud Host

You can deploy this service for free on platforms like **Render**, **Koyeb**, or **Hugging Face Spaces**.

### Option A: Deploy on Render (Recommended)

1. Push or fork this repository to your GitHub account.
2. Sign in to [Render](https://render.com).
3. Click **New +** -> **Blueprint**.
4. Connect your GitHub repository. Render will automatically detect `render.yaml` and configure the Web Service on the Free Tier.
5. Click **Apply**. Once deployed, Render will provide a live URL (e.g. `https://your-service.onrender.com`).

Alternatively, create a **Web Service** manually on Render:
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python server.py --host 0.0.0.0`
- **Environment Variables**: Set `PORT` (or let Render set it automatically).

### Option B: Deploy with Docker (Koyeb / Hugging Face Spaces / Fly.io)

This repo includes a production `Dockerfile`.

1. Push this repo to GitHub.
2. On your host of choice (e.g., Koyeb or Hugging Face Spaces Docker SDK):
   - Choose **Docker** build mode.
   - Set start port/container port to `6446` (or `$PORT` override).
3. Deploy! Access health checks at `https://your-app-url/health`.

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

- `muse-spark-1.3-contributor-free` / `muse-spark-1.2-contributor-free` (routed to `/zen/v1/responses` — Zen does not serve Muse Spark on chat/completions; requests are translated to the Responses API and translated back)
- `nemotron-3.5-lightning-free`
- `mimo-v2.5-free` (only one that also accepts image/audio/video input)
- `nemotron-3-ultra-free`
- `laguna-s-2.1-free`
- `longcat-2.0-free`
- `big-pickle` (reasoning model, text-only)

> Removed from the curated set (still auto-discovered if the Zen API lists them):
> `deepseek-v4-flash-free` — listed by the Zen API but currently returns
> `Error from provider (Console): Upstream request failed: Model is unavailable.`;
> `north-mini-code-free` — opaque `400 Provider returned error` on multi-turn
> tool calls; and `ling-3.0-flash-free` — no longer on the free tier upstream
> (404: "use this slug instead: inclusionai/ling-3.0-flash"). If your client
> configures models by hand, drop them; the proxy itself serves whatever the
> Zen API returns.

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

### Other endpoints

| Method | Path | What |
|--------|------|------|
| `GET` | `/v1/models` | List models (includes limits + modalities) |
| `GET` | `/health` | Health + version |

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

then point roles at models with a `:low`/`:high`/`:max` reasoning suffix:

```yaml
modelRoles:
  default: opencode-local/nemotron-3.5-lightning-free:high
  vision: opencode-local/mimo-v2.5-free:high
```

NVIDIA models ride the same provider (`opencode-local/nvidia/kimi-k3:high`).

## License

MIT
