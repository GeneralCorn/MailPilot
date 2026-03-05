from typing import Any, NotRequired, TypedDict
class Tool(TypedDict):
    name: str
    description: str
    parameters: NotRequired[dict[str, Any]]
