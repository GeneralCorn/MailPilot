from __future__ import annotations

import abc
import uuid
from typing import Any, TYPE_CHECKING

from triage.schemas import AgentMessage, ToolCall, ToolResult

if TYPE_CHECKING:
    from triage.runtime.state import RuntimeState


class AgentInterface(abc.ABC):
    def __init__(
        self,
        name: str,
        *,
        system_prompt: str = "",
        max_steps: int = 5,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.max_steps = max_steps

        self._run_id: str = ""
        self._step_count: int = 0
        self._done: bool = False
        self._messages: list[AgentMessage] = []
        self._result: Any = None

    @abc.abstractmethod
    def think(self, payload: dict[str, Any], state: "RuntimeState") -> list[AgentMessage]:
        """Build prompt messages or decide on next action."""
        ...

    @abc.abstractmethod
    def act(self, messages: list[AgentMessage], state: "RuntimeState") -> Any:
        """Execute the decided action (LLM call, tool invocation, etc.)."""
        ...

    @abc.abstractmethod
    def observe(self, action_output: Any, state: "RuntimeState") -> bool:
        """Process act() result, update state. Return True to stop, False to continue."""
        ...


    def step(self, payload: dict[str, Any], state: "RuntimeState") -> Any:
        """Run one *think → act → observe* iteration."""
        self._step_count += 1
        messages = self.think(payload, state)
        self._messages.extend(messages)
        output = self.act(messages, state)
        self._done = self.observe(output, state)
        return output

    def run(self, payload: dict[str, Any], state: "RuntimeState") -> Any:
        """Loop step() until observe() returns True or max_steps reached."""
        self._run_id = uuid.uuid4().hex[:12]
        self._step_count = 0
        self._done = False
        self._messages = []
        self._result = None

        self.on_start(payload, state)

        try:
            while not self._done and self._step_count < self.max_steps:
                self._result = self.step(payload, state)
        except Exception as exc:
            self._result = self.on_error(exc, state)

        self.on_finish(self._result, state)
        return self._result

    def handle(self, payload: dict[str, Any], state: "RuntimeState") -> Any:
        """AgentHandler-compatible entry point; delegates to run()."""
        return self.run(payload, state)

    def as_handler(self):
        """Return callable matching AgentHandler signature for Runtime.register_agent()."""
        return self.handle

    def build_system_message(self) -> AgentMessage:
        """Wrap self.system_prompt into an AgentMessage."""
        return AgentMessage(role="system", content=self.system_prompt)

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def result(self) -> Any:
        return self._result

    @property
    def messages(self) -> list[AgentMessage]:
        return list(self._messages)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.name!r} steps={self._step_count} done={self._done}>"
        )
