from typing import Any, NotRequired, TypedDict

class ToolResult(TypedDict):
    tool: str
    success: bool
    message: NotRequired[str]
    data: NotRequired[dict[str, Any]]
