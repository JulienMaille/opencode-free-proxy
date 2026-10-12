"""Live probe: what do our free models ACTUALLY emit for a nested-array tool arg?

Port of pi-freeflow's scripts/live-tool-args.ts (commit 37c2cb6). Sends the real
OMP `todo` schema through the local proxy and prints the RAW arguments string
each model returns, classified against the intended shape, so malformation is
observed rather than assumed.

Opt-in diagnostic, never part of the test suite. Costs free-tier quota.

Usage:
    python probe_tool_args.py                 # free models, full + single tool
    python probe_tool_args.py MODEL [MODEL...]  # specific models
    FF_STREAM=1 python probe_tool_args.py     # streamed replies, reassembled
"""

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("PROBE_BASE", "http://127.0.0.1:6446")
TIMEOUT_MS = int(os.environ.get("FF_TIMEOUT_MS", "90000"))

OP_ENUM = ["init", "start", "done", "rm", "drop", "block", "unblock", "append", "view"]

# Verbatim from pi-freeflow's test/tool-translation-todo-full-schema fixture,
# so results are directly comparable with upstream's measurements.
TODO_PARAMS = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": OP_ENUM, "description": "operation to apply"},
        "list": {
            "type": "array",
            "description": "phased task list (init)",
            "items": {
                "type": "object",
                "properties": {
                    "phase": {"type": "string", "description": "phase name"},
                    "items": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["phase", "items"],
            },
        },
        "task": {"type": "string", "description": "task content"},
        "phase": {"type": "string", "description": "phase name"},
        "items": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string", "description": "blocker note (block op)"},
    },
    "required": ["op"],
}
TODO_DESC = "apply a single todo operation"
TODO_FN = {
    "type": "function",
    "function": {"name": "todo", "description": TODO_DESC, "parameters": TODO_PARAMS},
}

PROMPT = (
    "Call the todo tool exactly once with op=init to record a two-phase plan: "
    "phase 'Research' with tasks 'Map repo structure' and 'Trace auth flow', then "
    "phase 'Synthesis' with task 'Write summary'. Do not write any text."
)

# A realistic host tool inventory. pi-freeflow found malformation is triggered by
# tool-set DENSITY: a lone tool stays canonical, a full inventory does not.
HOST_TOOLS = [
    "bash", "read", "write", "edit", "glob", "grep", "todowrite", "task",
    "webfetch", "websearch", "skill", "question", "patch", "multiedit",
    "notebookread", "notebookedit", "lsp", "sql", "context7", "atlas",
]


def chat_tools(full: bool):
    if not full:
        return [TODO_FN]
    out = [
        {"type": "function", "function": {"name": n, "description": f"{n} tool",
                                          "parameters": {"type": "object", "properties": {}}}}
        for n in HOST_TOOLS
        if n != "todo"
    ]
    out.append(TODO_FN)
    return out


def classify(args: str) -> str:
    """Label an emitted argument payload against the intended shape."""
    try:
        v = json.loads(args)
    except Exception:
        return "UNPARSEABLE"
    if not v or not isinstance(v, dict) or isinstance(v, list):
        return "NOT-AN-OBJECT"
    rec = v
    notes = []
    if "list" not in rec and "items" in rec:
        notes.append("WRONG-FIELD(items-instead-of-list)")
    if isinstance(rec.get("list"), list):
        for i, entry in enumerate(rec["list"]):
            if isinstance(entry, str) and entry.strip().startswith("{"):
                notes.append("JSON-STRING-IN-LIST")
                continue
            if not isinstance(entry, dict):
                notes.append(f"list[{i}]:{type(entry).__name__}")
                continue
            entry_keys = set(entry)
            unknown = entry_keys - {"phase", "items"}
            if unknown:
                notes.append(f"NESTED-UNKNOWN-KEYS{sorted(unknown)}")
            if "items" not in entry:
                notes.append("NESTED-MISSING-items")
            elif not isinstance(entry["items"], list):
                notes.append(f"NESTED-items:{type(entry['items']).__name__}")
            if "phase" not in entry:
                notes.append("NESTED-MISSING-phase")
    elif rec.get("list") is not None:
        notes.append(f"list:{type(rec['list']).__name__}")
    if isinstance(rec.get("items"), list):
        first = rec["items"][0] if rec["items"] else None
        notes.append(f"items:array<{type(first).__name__}>")
        if isinstance(first, str) and first.strip().startswith("{"):
            notes.append("JSON-STRING-IN-ITEMS")
    if rec.get("op") != "init":
        notes.append(f"op:{rec.get('op')!r}")
    unknown = set(rec) - {"op", "list", "task", "phase", "items", "reason"}
    if unknown:
        notes.append(f"UNKNOWN-KEYS{sorted(unknown)}")
    return " ".join(notes) if notes else "canonical"


def args_from_completion(body: dict):
    """(name, args) out of a buffered chat.completion, Responses, or Anthropic body."""
    rec = body or {}
    if isinstance(rec.get("output"), list):
        for item in rec["output"]:
            if isinstance(item, dict) and item.get("type") == "function_call":
                return str(item.get("name") or ""), str(item.get("arguments") or "")
    for ch in rec.get("choices") or []:
        if not isinstance(ch, dict):
            continue
        for c in (ch.get("message") or {}).get("tool_calls") or []:
            fn = (c.get("function") or {}) if isinstance(c, dict) else {}
            if fn:
                return str(fn.get("name") or ""), str(fn.get("arguments") or "")
    for blk in rec.get("content") or []:
        if isinstance(blk, dict) and blk.get("type") == "tool_use":
            return str(blk.get("name") or ""), json.dumps(blk.get("input"), ensure_ascii=False)
    return None


def args_from_sse(text: str):
    """Reassemble a streamed reply the way a host does: concat per tool-call index."""
    by_index = {}
    for block in text.replace("\r\n", "\n").split("\n\n"):
        line = next((c for c in block.split("\n") if c.startswith("data:")), None)
        if not line:
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        for choice in parsed.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            for tc in (choice.get("delta") or {}).get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                idx = tc.get("index", 0)
                entry = by_index.setdefault(idx, {"name": "", "args": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    entry["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    entry["args"] += fn["arguments"]
        # Responses-wire function calls carry the name on the item, args on deltas
        t = parsed.get("type")
        idx = parsed.get("output_index", 0)
        item = parsed.get("item") or {}
        if t in ("response.output_item.added", "response.output_item.done") and isinstance(item, dict) \
                and item.get("type") == "function_call":
            entry = by_index.setdefault(idx, {"name": "", "args": ""})
            if item.get("name"):
                entry["name"] = item["name"]
            if t == "response.output_item.done" and isinstance(item.get("arguments"), str):
                entry["args"] = item["arguments"]
        elif t == "response.function_call_arguments.done" and isinstance(parsed.get("arguments"), str):
            entry = by_index.setdefault(idx, {"name": "", "args": ""})
            entry["args"] = parsed["arguments"]
    for entry in by_index.values():
        if entry["name"] and entry["args"]:
            return entry["name"], entry["args"]
    return None


def call_model(model: str, full: bool, streaming: bool):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": streaming,
        "max_tokens": 4096,
        "tools": chat_tools(full),
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_MS / 1000) as r:
            text = r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code} {e.read().decode()[:160]}"
    except Exception as e:
        return None, f"FAILED {e!r}"
    if streaming:
        return args_from_sse(text), None
    try:
        return args_from_completion(json.loads(text)), None
    except Exception:
        return None, f"BAD-JSON {text[:160]}"


def served_free_models():
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=20) as r:
        data = json.loads(r.read().decode())
    ids = [m.get("id", "") for m in data.get("data", [])]
    return [i for i in ids if i.endswith("-free") and "/" not in i]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    streaming = os.environ.get("FF_STREAM") == "1"
    models = args or served_free_models()
    print(f"# probe: {'stream' if streaming else 'buffered'} | {len(models)} model(s) | full-inventory + single-tool")
    print(f"# models: {', '.join(models)}\n")
    rows = []
    for model in models:
        for full in (True, False):
            label = "full " if full else "alone"
            call, err = call_model(model, full, streaming)
            if err:
                rows.append((model, label, "ERROR", err[:120], ""))
                print(f"{model:34s} {label} ERROR  {err[:110]}")
                continue
            if not call:
                rows.append((model, label, "NO-TOOL-CALL", "", ""))
                print(f"{model:34s} {label} NO-TOOL-CALL")
                continue
            name, raw = call
            verdict = classify(raw)
            rows.append((model, label, verdict, name, raw))
            print(f"{model:34s} {label} [{verdict}]")
            print(f"{'':34s}   raw: {raw[:300]}")

    print("\n=== SUMMARY ===")
    bad = [r for r in rows if r[2] not in ("canonical",)]
    print(f"total {len(rows)} calls | canonical {len(rows) - len(bad)} | malformed/other {len(bad)}")
    for model, label, verdict, _, _ in bad:
        print(f"  {model:34s} {label} -> {verdict}")


if __name__ == "__main__":
    main()
