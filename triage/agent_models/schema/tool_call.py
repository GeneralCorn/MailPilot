from typing import Any, NotRequired, TypedDict
class ToolCall(TypedDict):
    tool: str
    parameters: NotRequired[dict[str, Any]]
    reason: NotRequired[str]
