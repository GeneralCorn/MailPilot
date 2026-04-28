from triage.schemas import AgentMessage


def to_cached_request(all_msgs: list[AgentMessage]) -> tuple[str, list[dict]]:
    """Split AgentMessage list into (system_text, messages) and tag the last
    assistant content block with cache_control so the stable system + few-shot
    prefix is cached server-side.

    Anthropic caches everything from the start of the input up to and including
    a cache_control block. The last few-shot's assistant turn is the natural
    cut point — anything appended after it (the actual email, feedback,
    repair-retry messages) varies per call and stays uncached.
    """
    system_text = next((m.content for m in all_msgs if m.role == "system"), "")
    non_system = [m for m in all_msgs if m.role != "system"]

    last_assistant = -1
    for i, m in enumerate(non_system):
        if m.role == "assistant":
            last_assistant = i

    out: list[dict] = []
    for i, m in enumerate(non_system):
        block: dict = {"type": "text", "text": m.content}
        if i == last_assistant:
            block["cache_control"] = {"type": "ephemeral"}
        out.append({"role": m.role, "content": [block]})

    return system_text, out
