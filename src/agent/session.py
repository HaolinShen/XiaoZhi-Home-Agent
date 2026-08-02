"""Session lifecycle helpers for LangGraph checkpoints."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timedelta

from .context import AgentContext, SpaceDirectory
from ..memory.store import cleanup_expired_checkpoints


@dataclass
class SessionManager:
    space_directory: SpaceDirectory
    checkpointer: object
    ttl: timedelta = timedelta(days=7)

    def create(
        self,
        *,
        home_id: str,
        user_id: str,
        client_id: str,
        room_id: str | None = None,
        device_id: str | None = None,
        is_admin: bool = False,
        session_id: str | None = None,
    ) -> AgentContext:
        context = AgentContext(
            home_id=home_id,
            user_id=user_id,
            session_id=session_id or f"session-{uuid.uuid4().hex}",
            client_id=client_id,
            room_id=room_id,
            device_id=device_id,
            is_admin=is_admin,
        )
        return self.space_directory.validate(context)

    def resume(self, context: AgentContext) -> AgentContext:
        """Validate a caller-supplied stable session before reuse."""
        return self.space_directory.validate(context)

    def end(self, context: AgentContext) -> None:
        """Delete a session checkpoint when the configured saver supports it."""
        delete_thread = getattr(self.checkpointer, "delete_thread", None)
        if delete_thread is None:
            raise RuntimeError("configured checkpointer cannot delete sessions")
        delete_thread(context.session_id)

    def cleanup_expired(self) -> int:
        """Apply the configured checkpoint retention policy."""
        return cleanup_expired_checkpoints(self.checkpointer, self.ttl)


def build_agent_request(message, context: AgentContext) -> tuple[dict, dict]:
    """Build graph input and config without allowing model-controlled identity."""
    state_input = {"messages": [message], **context.to_state_input()}
    return state_input, context.to_config()
