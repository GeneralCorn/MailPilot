"""
Wire triage pipeline agents (router, evaluator, ranker, worker) to Runtime.
Handlers adapt (payload, state) -> call existing agent functions.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

from triage.agent import route, evaluate, rank, work
from triage.schemas import Message, State

from .state import RuntimeState

if TYPE_CHECKING:
    from .runtime import Runtime


def _message_from_payload(payload: dict[str, Any], key: str = "email") -> Message:
    raw = payload.get(key)
    if raw is None:
        raise ValueError(f"payload must contain '{key}'")
    return Message(**raw) if isinstance(raw, dict) else raw


def _pipeline_state(payload: dict[str, Any], state: RuntimeState) -> State:
    ps = state.get_artifact("pipeline_state")
    if ps is None:
        ps = payload.get("pipeline_state") or State()
        state.set_artifact("pipeline_state", ps)
    return ps


def router_handler(payload: dict[str, Any], state: RuntimeState) -> Any:
    """Agent: classify email into category. Payload: { \"email\": Message|dict }."""
    email = _message_from_payload(payload)
    ps = _pipeline_state(payload, state)
    route(email, ps)
    return ps


def evaluator_handler(payload: dict[str, Any], state: RuntimeState) -> Any:
    """Agent: validate classification, gate risk. Payload: { \"email\": Message|dict }."""
    email = _message_from_payload(payload)
    ps = _pipeline_state(payload, state)
    evaluate(email, ps)
    return ps


def ranker_handler(payload: dict[str, Any], state: RuntimeState) -> Any:
    """Agent: score and order emails. Payload: { \"pipeline_state\": State|dict } (optional)."""
    ps = _pipeline_state(payload, state)
    rank(ps)
    return ps


def worker_handler(payload: dict[str, Any], state: RuntimeState) -> Any:
    """Agent: execute approved actions. Payload: { \"email\", \"approved_actions\" }."""
    email = _message_from_payload(payload)
    approved_actions = payload.get("approved_actions", [])
    return work(email, approved_actions)


def register_triage_agents(runtime: "Runtime") -> None:
    """Register router, evaluator, ranker, worker with the runtime."""
    runtime.register_agent("router", router_handler)
    runtime.register_agent("evaluator", evaluator_handler)
    runtime.register_agent("ranker", ranker_handler)
    runtime.register_agent("worker", worker_handler)
