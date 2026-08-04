# opencode-free-proxy

Free AI models from [OpenCode](https://opencode.ai) (the Zen API free tier), exposed as standard OpenAI and Anthropic APIs. Works with any tool that speaks those formats: opencode CLI, Cursor, Claude Code, Cline, aider, raw `curl`, etc.

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

- `deepseek-v4-flash-free`
- `mimo-v2.5-free` (only one that also accepts image/audio/video input)
- `ling-3.0-flash-free`
- `nemotron-3-ultra-free`
- `north-mini-code-free`
- `laguna-s-2.1-free`

## API

### OpenAI format — `POST /v1/chat/completions`

```bash
curl http://localhost:6446/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-free",
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
    "model": "deepseek-v4-flash-free",
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

- **Sessions**: the proxy hashes the message prefix to reuse the upstream session, so multi-turn conversations stay coherent.
- **Proxy pool**: on by default, SOCKS5 proxies are scraped from public lists, verified, and rotated. Transport failures are blacklisted; `429` responses temporarily skip the current proxy so the caller can retry through another IP.
- **Auth headers**: the proxy adds the `x-opencode-*` headers the Zen API requires (discovered by reverse-engineering the opencode binary):

```
Authorization: Bearer public
User-Agent: opencode/1.15.0 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.13
x-opencode-client: cli
x-opencode-project: global
x-opencode-request: msg_<unique_id>
x-opencode-session: ses_<unique_id>
```

## License

MIT
