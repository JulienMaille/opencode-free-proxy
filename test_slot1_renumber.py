"""slot-1 first-head renumber: upstream emits the first call head on
index 1; the client must see it on index 0. Offline."""
import sys

sys.argv = ["test_slot1"]
import server as srv


def _piece(idx, name="glob", cid="call_01_x"):
    return {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": idx, "id": cid, "type": "function",
         "function": {"name": name, "arguments": "{}"}}]},
        "finish_reason": None}]}


def test_first_head_slot1_renumbered_to_0():
    cell = [False]
    p = _piece(1)
    srv._strip_injected_tool_deltas(p, True, cell)
    tcs = p["choices"][0]["delta"]["tool_calls"]
    assert tcs[0]["index"] == 0, tcs
    assert cell == [True]


def test_second_head_keeps_index():
    cell = [False]
    p1 = _piece(1, "glob", "call_a")
    srv._strip_injected_tool_deltas(p1, True, cell)
    p2 = _piece(1, "read", "call_b")
    srv._strip_injected_tool_deltas(p2, True, cell)
    assert p2["choices"][0]["delta"]["tool_calls"][0]["index"] == 1


def test_fragments_untouched_and_no_cell_ok():
    p = {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": "xx"}}]}, "finish_reason": None}]}
    srv._strip_injected_tool_deltas(p, True, None)
    assert p["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    # no-cell path never renumbers
    cell_none = _piece(1)
    srv._strip_injected_tool_deltas(cell_none, True, None)
    assert cell_none["choices"][0]["delta"]["tool_calls"][0]["index"] == 1


def test_injected_strip_still_works():
    p = {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "x", "type": "function",
         "function": {"name": "bash", "arguments": ""}}]}, "finish_reason": None}]}
    srv._strip_injected_tool_deltas(p, False, [False])
    assert p["choices"][0]["delta"]["tool_calls"] == []
