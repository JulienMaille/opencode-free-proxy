"""Model-retirement policy and bounded tool-arg salvage (pi-freeflow 1.39.0).

Two September-October review findings ported from pi-freeflow 7427771:

- an intermittent upstream ``503 Endpoint is unavailable`` must never retire
  (or persist) a model: the old counter called ``_mark_model_dead(persistent=
  True)`` after 3 sightings in an hour, so deepseek-v4.1-flash-free vanished
  from the served list and ``data/dead-models.json`` even though it answered
  again seconds later. Persistence stays reserved for the explicit "promotion
  has ended" signal.
- the prefix-glue salvage in ``_parse_tool_args`` is bounded (64 KiB / 128
  candidates): it re-slices and re-parses at every brace, a quadratic scan
  that stalled the single-threaded proxy for 34.5 s on 80 KB.

No network here: only pure helpers are exercised.
"""
import json
import sys

import pytest

# server parses its own CLI args at import; keep pytest's argv out of that.
_argv = sys.argv
sys.argv = [sys.argv[0]]
import server as srv  # noqa: E402

sys.argv = _argv

MODEL = "deepseek-v4.1-flash-free"


@pytest.fixture
def isolated(monkeypatch):
    """Fresh retirement state; persistence is observed, never written."""
    persisted: list[int] = []
    monkeypatch.setattr(srv, "_backend_down_hits", {})
    monkeypatch.setattr(srv, "_dead_models", set())
    monkeypatch.setattr(srv, "DEAD_IDS", set())
    monkeypatch.setattr(srv, "_models_cache", [MODEL])
    monkeypatch.setattr(srv, "_models_meta", {})
    monkeypatch.setattr(srv, "_persist_dead_ids", lambda: persisted.append(1))
    return persisted


def test_backend_unavailable_never_retires_or_persists(isolated):
    for _ in range(10):
        srv._note_backend_unavailable(MODEL)

    assert MODEL not in srv._dead_models
    assert MODEL not in srv.DEAD_IDS
    assert isolated == []
    # The model is still offered: the failing turn surfaced the 503 itself.
    assert MODEL in srv._served_models()


def test_backend_unavailable_counter_is_per_window(isolated):
    srv._note_backend_unavailable(MODEL)
    srv._note_backend_unavailable(MODEL)
    srv._note_backend_unavailable(MODEL)
    # The threshold only logs; the counter keeps counting without retiring.
    srv._note_backend_unavailable(MODEL)
    assert srv._backend_down_hits[MODEL][0] == 4
    assert MODEL not in srv._dead_models


def test_promotion_ended_still_persists(isolated):
    assert srv._mark_model_dead(MODEL, persistent=True, reason="promotion has ended")
    assert MODEL in srv.DEAD_IDS
    assert isolated == [1]
    assert MODEL not in srv._served_models()


def test_salvage_still_repairs_live_glue():
    value, salvaged = srv._parse_tool_args(',"timeout":400}{"command":"ls"}')
    assert value == {"command": "ls"}
    assert salvaged is True


def test_salvage_refuses_payload_over_the_size_bound():
    args = '{"junk":"' + ("x" * srv._SALVAGE_MAX_BYTES) + '"}{"command":"ls"}'
    assert len(args) > srv._SALVAGE_MAX_BYTES
    # Without the bound the trailing object would be salvaged; over it the
    # host must see its own schema error instead of a quadratic scan.
    assert srv._parse_tool_args(args) == (None, False)


def test_salvage_stops_at_the_candidate_bound():
    args = ("}{" * srv._SALVAGE_MAX_CANDIDATES) + '{"command":"ls"}'
    assert srv._parse_tool_args(args) == (None, False)


def test_salvage_still_reaches_a_call_within_the_bound():
    args = ("}{" * (srv._SALVAGE_MAX_CANDIDATES - 1)) + '{"command":"ls"}'
    assert srv._parse_tool_args(args) == ({"command": "ls"}, True)


def test_wellformed_large_args_parse_before_the_bound():
    args = json.dumps({"command": "x" * (80 * 1024)})
    assert len(args) > srv._SALVAGE_MAX_BYTES
    value, salvaged = srv._parse_tool_args(args)
    assert value is not None and len(value["command"]) == 80 * 1024
    assert salvaged is False
