from __future__ import annotations

# GT names tools as `calendar.create_event` with args `start`/`end`; the Worker
# emits `calendar` with `start_time`/`end_time`. Map both vocabularies.
_TOOL_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "calendar.create_event": ("calendar", {"title": "title", "start": "start_time", "end": "end_time"}),
    "reply_draft.create": ("send_email", {"to": "to", "body": "body"}),
}


def _agent_tools_for(predicted_email: dict) -> list[dict]:
    out: list[dict] = []
    for src in ("proposed_actions", "external_actions"):
        for c in predicted_email.get(src, []) or []:
            if isinstance(c, dict) and c.get("tool"):
                out.append(c)
    return out


def _matches(expected: dict, agent_call: dict) -> bool:
    mapping = _TOOL_MAP.get(expected["tool"])
    if mapping is None:
        return False
    target_tool, arg_map = mapping
    if agent_call.get("tool") != target_tool:
        return False
    params = agent_call.get("parameters") or {}
    for gt_arg in expected.get("required_args", []):
        v = params.get(arg_map.get(gt_arg, gt_arg))
        if v is None or v == "" or v == []:
            return False
    return True


def score_tool_actions(predicted: list[dict], expected_tool_calls: list[dict]) -> float:
    if not expected_tool_calls:
        return 1.0

    pred_by_id: dict[str, list[dict]] = {p["id"]: _agent_tools_for(p) for p in predicted}
    target_tools = {t for t, _ in _TOOL_MAP.values()}

    matched = sum(
        1
        for exp in expected_tool_calls
        if any(_matches(exp, c) for c in pred_by_id.get(exp["id"], []))
    )
    relevant_emitted = sum(
        1
        for calls in pred_by_id.values()
        for c in calls
        if c.get("tool") in target_tools
    )
    denom = max(len(expected_tool_calls), relevant_emitted)
    return matched / denom if denom else 0.0
