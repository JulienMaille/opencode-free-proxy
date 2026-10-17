# opencode / OpenChamber setup

Point [opencode](https://opencode.ai) at this proxy's local OpenAI-compatible
endpoint, and keep the model list it offers in sync with what the proxy actually
serves. OpenChamber is the desktop/TUI host around opencode: it does **not**
define models itself, it reads them from opencode and only stores UI state about
them.

## Where the model list lives

| What | File | Notes |
|---|---|---|
| Model **definitions** (the list opencode/OpenChamber offer) | `~/.config/opencode/opencode.jsonc` → `provider.local-openai.models` | The only place to add/remove a model. Pickers show it as `local-openai/<id>`. |
| OpenChamber **UI state** | `~/.config/openchamber/preferences.json` | Shape `fields.<key> = { updatedAt, value }`. |
| OpenChamber **UI state** (duplicate) | `~/.config/openchamber/settings.json` | Flat keys. |

OpenChamber keeps the same keys in **both** state files:

- `recentModels` — the "recent" row of the picker
- `hiddenModels` — models hidden from the picker
- `favoriteModels`, `defaultModel`, `lastSelectedModel`, `recentEfforts`

> **Gotcha:** because the two files duplicate these keys, edit **both** or the
> next UI change by OpenChamber will overwrite your edit from the other file.
> In `preferences.json` respect the `{ updatedAt, value }` wrapper.

## Source of truth for limits

Never invent a context/output size: read it from the proxy.

```powershell
curl http://127.0.0.1:6446/v1/models
```

Each served model carries `limits.context` / `limits.output` (and the alias keys
`contextWindow` / `maxTokens`). Alias entries without the `-free` suffix
(`longcat-2.5`, `mimo-v2.6`, `space-bunny`, …) are convenience aliases and may
omit `limits` — configure the `-free` slug, not the alias.

Copy the values verbatim into the opencode entry as
`"limit": { "context": <context>, "output": <output> }`. A wrong limit is worse
than a missing one: too small truncates usable context, too large makes opencode
send requests the model then rejects.

## Minimal provider block — `~/.config/opencode/opencode.jsonc`

Use `@ai-sdk/openai-compatible`; declare `modalities` per model (the
`capabilities` key is **not** read by opencode; use `modalities` + `attachment:
true` for image models):

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
        "nemotron-3.5-super-vl-free": {
          "name": "Nemotron 3.5 Super VL",
          "id": "nemotron-3.5-super-vl-free",
          "reasoning": true,
          "attachment": true,
          "modalities": { "input": ["text", "image"], "output": ["text"] },
          "limit": { "context": 1048576, "output": 131072 }
        }
      }
    }
  }
}
```

## Keeping the list in sync

1. Read the real catalogue from the proxy: `GET /v1/models`.
2. In `provider.local-openai.models`:
   - **add** models the proxy serves but the config lacks (copy the shape of a
     neighbouring entry);
   - **remove** entries the proxy no longer serves (retired / dead / deprecated);
   - **fix** any drift in `limit.context`, `limit.output` or `modalities`;
   - leave every other provider and the rest of the file untouched.
3. If the root `model` (default) points at a model you just removed, switch it to
   one that still exists.
4. Validate the JSON (strict parse) and summarise: added / removed / corrected.
5. Optionally clean OpenChamber state: update `defaultModel` /
   `lastSelectedModel` if they point at a removed model, and drop dead entries
   from `hiddenModels` / `recentModels` — in **both** state files.
6. Restart opencode: the config is read **once at startup**.

## Gotchas

- **Restart required** — opencode reads `opencode.jsonc` once at startup.
- **`id` must match the proxy slug** — a Zen free slug (`big-pickle`) or a Cline
  route (`cline/<id>`). The picker key is `local-openai/<id>`.
- **Images** — only models with `"image"` in `modalities.input` (plus
  `attachment: true`) accept attachments; the rest are text-only.
- **Damaged default** — if the root `model` names a removed model, opencode falls
  back and the picker looks broken. Keep it valid.
- **Aliases are not models** — configure the `-free` slug the proxy advertises in
  `/v1/models`, not a bare alias.
- **This page is about client config only** — syncing `opencode.jsonc` touches no
  proxy code and no proxy state.
