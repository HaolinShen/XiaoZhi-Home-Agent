"""Agent 模块"""

from .context import AgentContext, ContextValidationError, SpaceDirectory
from .prompts import build_system_prompt
from .session import SessionManager, build_agent_request
from .state import AgentState


def build_graph(*args, **kwargs):
    """Lazily import graph construction to avoid tool/context import cycles."""
    from .graph import build_graph as _build_graph

    return _build_graph(*args, **kwargs)


def build_llm(*args, **kwargs):
    """Lazily import the LLM factory to avoid tool/context import cycles."""
    from .graph import build_llm as _build_llm

    return _build_llm(*args, **kwargs)

__all__ = [
    "AgentState",
    "build_graph",
    "build_llm",
    "build_system_prompt",
    "AgentContext",
    "ContextValidationError",
    "SpaceDirectory",
    "SessionManager",
    "build_agent_request",
]
