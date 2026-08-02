"""Acceptance-focused tests for iteration 001 phase one."""

import tempfile
import unittest
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from src.agent.context import (
    AgentContext,
    ContextValidationError,
    DeviceLocation,
    SpaceDirectory,
)
from src.agent.session import SessionManager, build_agent_request
from src.memory.store import close_checkpointer, create_checkpointer


class AgentContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = SpaceDirectory(
            {"home-a": {"living_room", "bedroom"}, "home-b": {"living_room"}},
            {
                "light-a": DeviceLocation("home-a", "living_room"),
                "light-b": DeviceLocation("home-b", "living_room"),
            },
        )

    def test_session_id_is_langgraph_thread_id(self) -> None:
        context = AgentContext(
            home_id="home-a",
            user_id="user-a",
            session_id="session-a",
            client_id="phone-a",
        )
        self.assertEqual(
            context.to_config()["configurable"]["thread_id"],
            "session-a",
        )

    def test_rejects_room_from_another_home(self) -> None:
        context = AgentContext(
            home_id="home-b",
            user_id="user-a",
            session_id="session-a",
            client_id="phone-a",
            room_id="bedroom",
        )
        with self.assertRaises(ContextValidationError):
            self.directory.validate(context)

    def test_rejects_device_from_another_home(self) -> None:
        context = AgentContext(
            home_id="home-a",
            user_id="user-a",
            session_id="session-a",
            client_id="phone-a",
            device_id="light-b",
        )
        with self.assertRaises(ContextValidationError):
            self.directory.validate(context)

    def test_explicit_room_mention_resolves_to_business_id(self) -> None:
        self.assertEqual(
            self.directory.resolve_room_mention("先看看主卧"),
            "bedroom",
        )

    def test_build_request_keeps_identity_out_of_message(self) -> None:
        context = AgentContext(
            home_id="home-a",
            user_id="user-a",
            session_id="session-a",
            client_id="phone-a",
            room_id="living_room",
        )
        state, config = build_agent_request("turn", context)
        self.assertEqual(state["request_room_id"], "living_room")
        self.assertEqual(config["configurable"]["home_id"], "home-a")


class SessionManagerTests(unittest.TestCase):
    def test_create_and_resume_stable_session(self) -> None:
        directory = SpaceDirectory(
            {"home-a": {"living_room"}},
            {"light-a": DeviceLocation("home-a", "living_room")},
        )
        manager = SessionManager(directory, MemorySaver())
        created = manager.create(
            home_id="home-a",
            user_id="user-a",
            client_id="phone-a",
            session_id="stable-session",
        )
        resumed = manager.resume(created)
        self.assertEqual(resumed.session_id, "stable-session")


class CheckpointerTests(unittest.TestCase):
    def test_memory_mode_is_explicit(self) -> None:
        self.assertIsInstance(create_checkpointer(None), MemorySaver)

    def test_sqlite_mode_creates_database_or_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "checkpoints.db"
            try:
                checkpointer = create_checkpointer(str(db_path))
            except RuntimeError as exc:
                self.assertIn("SQLite", str(exc))
            else:
                self.assertEqual(checkpointer.__class__.__name__, "SqliteSaver")
                self.assertTrue(db_path.exists())
                close_checkpointer(checkpointer)


if __name__ == "__main__":
    unittest.main()
