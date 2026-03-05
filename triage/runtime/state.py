"""Runtime state container for the task pipeline."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class RuntimeState(BaseModel):
    """Mutable state maintained by the runtime for the current run."""
    task: str = Field(default="", description="Current task description")
    plan: list[str] = Field(default_factory=list, description="Steps or plan items")
    artifacts: dict[str, Any] = Field(default_factory=dict, description="Named outputs (e.g. search results, drafts)")
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional state for agents")

    def append_plan(self, step: str) -> None:
        self.plan.append(step)

    def set_artifact(self, name: str, value: Any) -> None:
        self.artifacts[name] = value

    def get_artifact(self, name: str, default: Any = None) -> Any:
        return self.artifacts.get(name, default)
