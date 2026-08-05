# oh-my-pi (omp) setup

Point [oh-my-pi](https://github.com/can1357/oh-my-pi) at this proxy's local OpenAI-compatible endpoint instead of a cloud provider. The provider (`opencode-local`) and per-role thinking levels below are the merged best of the two configs we run.

## Install

```powershell
irm https://omp.sh/install.ps1 | iex
```

Restart your terminal afterwards so the `omp` command is on `PATH`.

## 1. Provider + models — `~/.omp/agent/models.yml`

Model metadata (context window, max output, reasoning, modalities) mirrors what the proxy serves (`GET /v1/models`). `cost` is pinned to zero everywhere because the tier is free.

> `ling-3.0-flash-free` and `north-mini-code-free` are intentionally not configured here: the former is gone from the free tier upstream (404 → paid `inclusionai/ling-3.0-flash`), and the latter returns opaque `400 Provider returned error` on multi-turn tool calls. See `README.md`.

```yaml
providers:
  opencode-local:
    baseUrl: http://127.0.0.1:6446/v1
    auth: none
    api: openai-completions
    compat:
      supportsDeveloperRole: false
      supportsMultipleSystemMessages: true
      supportsUsageInStreaming: false
      maxTokensField: max_tokens
      supportsToolChoice: true
      supportsForcedToolChoice: false
      reasoningContentField: reasoning_content
      requiresReasoningContentForToolCalls: true
      allowsSyntheticReasoningContentForToolCalls: false
    models:
      - id: deepseek-v4-flash-free
        name: DeepSeek V4 Flash Free
        reasoning: true
        input: [text]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 200000
        maxTokens: 128000
      - id: mimo-v2.5-free
        name: MiMo V2.5 Free
        reasoning: true
        input: [text, image]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 200000
        maxTokens: 32000
      - id: nemotron-3-ultra-free
        name: Nemotron 3 Ultra Free
        reasoning: true
        input: [text]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 1000000
        maxTokens: 128000
      - id: laguna-s-2.1-free
        name: Laguna S 2.1 Free
        reasoning: true
        input: [text]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 256000
        maxTokens: 32000
      - id: longcat-2.0-free
        name: LongCat 2.0 Free
        reasoning: true
        input: [text]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 1000000
        maxTokens: 131072
```

Notes on the `compat` knobs (validated against the Zen gateway behavior the proxy fronts):

- `auth: none` — the proxy doesn't validate client auth, so omp resolves to its keyless sentinel and sends no `Authorization` header. Don't set `apiKey` here.
- `supportsDeveloperRole: false` — the upstream rejects the newer `developer` role (only `system`/`user`/`assistant`/`tool`). omp sends `system` directly; the proxy also rewrites `developer` → `system` as a safety net.
- `reasoningContentField: reasoning_content` + `requiresReasoningContentForToolCalls: true` + `allowsSyntheticReasoningContentForToolCalls: false` — opencode-zen 400s follow-up requests when a prior assistant tool-call turn lacks exact `reasoning_content`; DeepSeek-family and MiMo reject synthetic placeholder values, hence the false.
- `supportsForcedToolChoice: false` — any model in thinking mode rejects forced `tool_choice` (`Thinking mode does not support this tool_choice`, upstream 400). Tell omp not to hard-force a single tool.
- `supportsUsageInStreaming: false` — the proxy ignores `stream_options.include_usage`; don't ask for streamed usage.
- `maxTokensField: max_tokens` — the proxy accepts both, `max_tokens` is the safest.
- `supportsMultipleSystemMessages: true` — allow multiple `system` turns; default anyway.

## 2. Default model roles — `~/.omp/agent/config.yml`

```yaml
modelRoles:
  default: opencode-local/deepseek-v4-flash-free:high
  smol: opencode-local/nemotron-3-ultra-free:medium
  slow: opencode-local/nemotron-3-ultra-free:max
  vision: opencode-local/mimo-v2.5-free:high
  plan: opencode-local/deepseek-v4-flash-free:high
  designer: opencode-local/deepseek-v4-flash-free:high
  commit: opencode-local/nemotron-3-ultra-free:medium
  tiny: opencode-local/nemotron-3-ultra-free:medium
  task: opencode-local/deepseek-v4-flash-free:high
  advisor: opencode-local/nemotron-3-ultra-free:max
cycleOrder:
  - smol
  - default
  - slow
  - vision
  - plan
```

Pre-assigning every role avoids oh-my-pi's first-run model picker. The `:high`/`:medium`/`:max` suffixes set per-role reasoning `effort` (the proxy accepts `reasoning_effort`); keep them when editing.

## 3. Verify

```powershell
# list models (should show opencode-local with 5 entries)
omp models

# one-shot prompt through the proxy
omp -p --model "opencode-local/deepseek-v4-flash-free" "hello"
```

Start the proxy on `http://127.0.0.1:6446` before using omp.