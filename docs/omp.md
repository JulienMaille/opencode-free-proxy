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

NVIDIA NIM models exposed by the proxy (`nvidia/*`, e.g. `nvidia/kimi-k3`) ride
the same provider — reference them as `opencode-local/nvidia/kimi-k3:high`.
They use plain OpenAI messages (no reasoning-content round-trip requirement),
and the proxy forwards `reasoning_effort` (`low`/`high`/`max`) on the NVIDIA
route, so the per-role `:low`/`:high`/`:max` suffixes work as-is. Kimi-K3 is
1M context / 131K max output, thinking always on.

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
      - id: space-bunny-free
        name: Space Bunny Free
        reasoning: true
        input: [text, image]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 1048576
        maxTokens: 524288
      - id: mimo-v2.6-flash-free
        name: MiMo V2.6 Flash Free
        reasoning: true
        input: [text, image]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 200000
        maxTokens: 32000
      - id: mimo-v2.5-free
        name: MiMo V2.5 Free
        reasoning: true
        input: [text, image]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 200000
        maxTokens: 32000
      - id: muse-spark-1.3-contributor-free
        name: muse-spark-1.3-contributor-free
        reasoning: true
        input: [text, image]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 1048576
        maxTokens: 131072
      - id: muse-spark-1.2-contributor-free
        name: muse-spark-1.2-contributor-free
        reasoning: true
        input: [text, image]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 1048576
        maxTokens: 131072
      - id: nemotron-3.5-lightning-free
        name: nemotron-3.5-lightning-free
        reasoning: true
        input: [text]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 262144
        maxTokens: 262144
      - id: nemotron-3-ultra-free
        name: nemotron-3-ultra-free
        reasoning: true
        input: [text]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 1000000
        maxTokens: 128000
      - id: ling-3.0-flash-fin-free
        name: ling-3.0-flash-fin-free
        reasoning: true
        input: [text]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 262144
        maxTokens: 32768
      - id: big-pickle
        name: Big Pickle
        reasoning: true
        input: [text]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 200000
        maxTokens: 32000
      - id: jev-1.13-free
        name: jev-1.13-free
        reasoning: true
        input: [text]
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        contextWindow: 128000
        maxTokens: 16384
```

Notes on the `compat` knobs (validated against the Zen gateway behavior the proxy fronts):

- `auth: none` — the proxy doesn't validate client auth, so omp resolves to its keyless sentinel and sends no `Authorization` header. Don't set `apiKey` here.
- `supportsDeveloperRole: false` — the upstream rejects the newer `developer` role (only `system`/`user`/`assistant`/`tool`). omp sends `system` directly; the proxy also rewrites `developer` → `system` as a safety net.
- `reasoningContentField: reasoning_content` + `requiresReasoningContentForToolCalls: true` + `allowsSyntheticReasoningContentForToolCalls: false` — opencode-zen 400s follow-up requests when a prior assistant tool-call turn lacks exact `reasoning_content`; MiMo and DeepSeek-family reject synthetic placeholder values, hence the false.
- `supportsForcedToolChoice: false` — any model in thinking mode rejects forced `tool_choice` (`Thinking mode does not support this tool_choice`, upstream 400). Tell omp not to hard-force a single tool.
- `supportsUsageInStreaming: false` — the proxy ignores `stream_options.include_usage`; don't ask for streamed usage.
- `maxTokensField: max_tokens` — the proxy accepts both, `max_tokens` is the safest.
- `supportsMultipleSystemMessages: true` — allow multiple `system` turns; default anyway.

## 2. Default model roles — `~/.omp/agent/config.yml`

MiMo V2.6 Flash Free is the single model for all roles. A YAML anchor
(`&model`) defines it once; every role references `*model` to avoid
duplication. Change the anchor value and every role follows.

```yaml
# Change model+effort here; all roles reference the anchor.
_model: &model opencode-local/space-bunny-free:auto

modelRoles:
  default: *model
  vision: *model
  reasoning: *model
  fast: opencode-local/mimo-v2.6-flash-free:auto
cycleOrder:
  - default
  - vision
  - reasoning
  - fast
defaultThinkingLevel: auto
symbolPreset: unicode
theme:
  dark: win11
  light: win11
setupVersion: 2
composer:
  shape: box
```

Pre-assigning every role avoids oh-my-pi's first-run model picker. Space Bunny Free has the best specs: 1M context, 512K output, vision. The `:auto` suffix lets the classifier pick the right thinking effort per turn. To vary effort per role, use `:low`/`:high`/`:max` instead.

## 3. Verify

```powershell
# list models (should show opencode-local entries)
omp models

# one-shot prompt through the proxy
omp -p --model "opencode-local/mimo-v2.6-flash-free" "hello"
```

Start the proxy on `http://127.0.0.1:6446` before using omp.