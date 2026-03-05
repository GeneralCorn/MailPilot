from typing import Literal, TypedDict
class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str