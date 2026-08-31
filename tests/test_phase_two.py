"""Acceptance-focused tests for iteration 001 phase two."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.context import AgentContext, DeviceLocation, SpaceDirectory
from src.agent.graph import build_graph
from src.devices import DeviceRegistry, SimulatorBackend
from src.memory import (
    MemoryPermissionError,
    MemoryRepository,
    MemoryScope,
    MemoryService,
    MemoryType,
    MemoryWrite,
)
from src.tools.memory import build_memory_tools


class MemoryPhaseTwoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "memories.db"
        self.repository = MemoryRepository(str(self.db_path))
        self.directory = SpaceDirectory(
            {
                "home-a": {"living_room", "bedroom"},
                "home-b": {"living_room"},
            },
            {
                "light-a": DeviceLocation("home-a", "living_room"),
                "ac-a": DeviceLocation("home-a", "bedroom"),
                "light-b": DeviceLocation("home-b", "living_room"),
            },
        )
        self.service = MemoryService(self.repository, self.directory)

    def tearDown(self) -> None:
        self.repository.close()
        self.temp_dir.cleanup()

    def memory_tools(self):
        """P1: 记忆工具由工厂按依赖显式构建，不再经模块级单例。"""
        return {
            tool.name: tool
            for tool in build_memory_tools(self.service)
        }

    @staticmethod
    def context(
        home_id: str = "home-a",
        user_id: str = "user-a",
        *,
        room_id: str | None = None,
        device_id: str | None = None,
        is_admin: bool = False,
    ) -> AgentContext:
        return AgentContext(
            home_id=home_id,
            user_id=user_id,
            session_id=f"session-{user_id}",
            client_id="phone",
            room_id=room_id,
            device_id=device_id,
            is_admin=is_admin,
        )

    def save_personal(
        self,
        context: AgentContext,
        key: str,
        value: dict,
        *,
        room_id: str | None = None,
        device_id: str | None = None,
    ):
        return self.service.save(
            context,
            MemoryWrite(
                scope=MemoryScope.USER,
                memory_type=MemoryType.PREFERENCE,
                memory_key=key,
                memory_value=value,
                room_id=room_id,
                device_id=device_id,
                source="用户明确要求记住",
            ),
        )

    def test_schema_and_indexes_are_created(self) -> None:
        connection = sqlite3.connect(self.db_path)
        try:
            table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
            ).fetchone()
            indexes = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memories'"
                )
            }
        finally:
            connection.close()
        self.assertIsNotNone(table)
        self.assertIn("idx_memories_home_scope", indexes)
        self.assertIn("idx_memories_user", indexes)
        self.assertIn("uq_memories_business_key", indexes)

    def test_upsert_survives_repository_restart(self) -> None:
        context = self.context()
        first = self.save_personal(context, "lighting.color", {"color": "暖光"})
        second = self.save_personal(context, "lighting.color", {"color": "冷白光"})
        self.assertEqual(first.id, second.id)
        self.assertEqual(second.memory_value, {"color": "冷白光"})

        self.repository.close()
        self.repository = MemoryRepository(str(self.db_path))
        self.service = MemoryService(self.repository, self.directory)
        records = self.service.list_memories(context)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].memory_value, {"color": "冷白光"})

    def test_homes_and_users_are_isolated(self) -> None:
        self.save_personal(self.context(), "lighting.color", {"color": "暖光"})
        self.save_personal(
            self.context("home-b", "user-b"),
            "lighting.color",
            {"color": "蓝光"},
        )

        self.assertEqual(len(self.service.list_memories(self.context())), 1)
        self.assertEqual(len(self.service.list_memories(self.context(user_id="user-b"))), 0)
        home_b = self.service.list_memories(self.context("home-b", "user-b"))
        self.assertEqual(home_b[0].memory_value, {"color": "蓝光"})

    def test_shared_rules_are_visible_but_personal_preferences_are_private(self) -> None:
        admin = self.context(is_admin=True)
        self.service.save(
            admin,
            MemoryWrite(
                scope=MemoryScope.HOME,
                memory_type=MemoryType.CONSTRAINT,
                memory_key="quiet_hours",
                memory_value={"after": "23:00"},
            ),
        )
        self.save_personal(admin, "lighting.color", {"color": "暖光"})

        other_user_records = self.service.list_memories(self.context(user_id="user-b"))
        self.assertEqual([record.memory_key for record in other_user_records], ["quiet_hours"])

    def test_shared_memory_write_requires_admin(self) -> None:
        with self.assertRaises(MemoryPermissionError):
            self.service.save(
                self.context(),
                MemoryWrite(
                    scope=MemoryScope.HOME,
                    memory_type=MemoryType.CONSTRAINT,
                    memory_key="quiet_hours",
                    memory_value={"after": "23:00"},
                ),
            )

    def test_room_device_and_personal_combinations_are_filtered(self) -> None:
        admin = self.context(is_admin=True)
        self.service.save(
            admin,
            MemoryWrite(
                scope=MemoryScope.ROOM,
                memory_type=MemoryType.ALIAS,
                memory_key="living_room.alias",
                memory_value={"alias": "大厅"},
                room_id="living_room",
            ),
        )
        self.service.save(
            admin,
            MemoryWrite(
                scope=MemoryScope.DEVICE,
                memory_type=MemoryType.ALIAS,
                memory_key="light-a.alias",
                memory_value={"alias": "主灯"},
                device_id="light-a",
            ),
        )
        self.save_personal(
            admin,
            "lighting.brightness",
            {"brightness": 30},
            room_id="living_room",
        )
        self.save_personal(
            admin,
            "light-a.color",
            {"color": "暖光"},
            device_id="light-a",
        )

        room_keys = {
            record.memory_key
            for record in self.service.list_memories(
                self.context(room_id="living_room")
            )
        }
        self.assertEqual(
            room_keys,
            {"living_room.alias", "lighting.brightness"},
        )

        device_keys = {
            record.memory_key
            for record in self.service.list_memories(self.context(device_id="light-a"))
        }
        self.assertEqual(
            device_keys,
            {
                "living_room.alias",
                "light-a.alias",
                "lighting.brightness",
                "light-a.color",
            },
        )

    def test_invalid_scope_combinations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.service.save(
                self.context(is_admin=True),
                MemoryWrite(
                    scope=MemoryScope.ROOM,
                    memory_type=MemoryType.ALIAS,
                    memory_key="invalid",
                    memory_value={"alias": "x"},
                ),
            )
        with self.assertRaises(ValueError):
            self.service.save(
                self.context(is_admin=True),
                MemoryWrite(
                    scope=MemoryScope.DEVICE,
                    memory_type=MemoryType.ALIAS,
                    memory_key="invalid",
                    memory_value={"alias": "x"},
                    device_id="light-b",
                ),
            )

    def test_view_update_and_delete_respect_ownership(self) -> None:
        owner = self.context()
        record = self.save_personal(owner, "lighting.color", {"color": "暖光"})
        self.assertEqual(self.service.get(owner, record.id).id, record.id)

        with self.assertRaises(MemoryPermissionError):
            self.service.get(self.context(user_id="user-b"), record.id)
        with self.assertRaises(MemoryPermissionError):
            self.service.update(
                self.context(user_id="user-b"),
                record.id,
                {"color": "蓝光"},
            )

        updated = self.service.update(owner, record.id, {"color": "冷白光"})
        self.assertEqual(updated.memory_value, {"color": "冷白光"})
        self.assertTrue(self.service.delete(owner, record.id))
        self.assertEqual(self.service.list_memories(owner), [])
        with self.assertRaises(KeyError):
            self.service.get(owner, record.id)

    def test_prompt_format_contains_only_accessible_memories(self) -> None:
        self.save_personal(self.context(), "lighting.color", {"color": "暖光"})
        self.save_personal(
            self.context(user_id="user-b"),
            "lighting.color",
            {"color": "蓝光"},
        )
        prompt = self.service.format_for_prompt(self.context())
        self.assertIn("暖光", prompt)
        self.assertNotIn("蓝光", prompt)

    def test_tools_use_identity_from_runnable_config(self) -> None:
        tools = self.memory_tools()
        config = self.context().to_config()
        saved_message = tools["save_personal_memory"].invoke(
            {
                "memory_key": "lighting.color",
                "memory_value": {"color": "暖光"},
                "source": "记住我喜欢暖光",
            },
            config=config,
        )
        self.assertIn("已保存个人记忆", saved_message)
        listed = tools["list_personal_memories"].invoke({}, config=config)
        self.assertIn("lighting.color", listed)

        record = self.service.list_memories(self.context())[0]
        updated = tools["update_personal_memory"].invoke(
            {
                "memory_id": record.id,
                "memory_value": {"color": "冷白光"},
            },
            config=config,
        )
        self.assertIn("已更新个人记忆", updated)
        deleted = tools["delete_personal_memory"].invoke(
            {"memory_id": record.id},
            config=config,
        )
        self.assertEqual(deleted, "记忆已删除")

    def test_home_rule_tool_requires_trusted_admin_flag(self) -> None:
        tools = self.memory_tools()
        payload = {
            "memory_key": "quiet_hours",
            "memory_value": {"after": "23:00"},
            "source": "记住晚上十一点后使用安静模式",
        }
        with self.assertRaises(MemoryPermissionError):
            tools["save_home_rule"].invoke(payload, config=self.context().to_config())

        admin_config = self.context(is_admin=True).to_config()
        result = tools["save_home_rule"].invoke(payload, config=admin_config)
        self.assertIn("已保存家庭规则", result)

    def test_graph_injects_accessible_memory_before_model_call(self) -> None:
        self.save_personal(self.context(), "lighting.color", {"color": "暖光"})
        self.save_personal(
            self.context(user_id="user-b"),
            "lighting.color",
            {"color": "蓝光"},
        )
        self.repository.close()

        captured_messages = []

        class FakeLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                captured_messages.extend(messages)
                return AIMessage(content="好的")

        settings = SimpleNamespace(
            memory=SimpleNamespace(
                enable_long_term=True,
                long_term_db_path=str(self.db_path),
                db_path="",
            )
        )
        registry = DeviceRegistry(SimulatorBackend())
        with patch("src.agent.graph.build_llm", return_value=FakeLLM()):
            graph = build_graph(registry, settings, self.directory)
            context = self.context()
            graph.invoke(
                {"messages": [HumanMessage(content="新会话中帮我开灯")], **context.to_state_input()},
                context.to_config(),
            )

        system_prompt = captured_messages[0].content
        self.assertIn("lighting.color", system_prompt)
        self.assertIn("暖光", system_prompt)
        self.assertNotIn("蓝光", system_prompt)
        graph.memory_repository.close()

        # The graph owned the reopened connection; prevent tearDown double-close.
        self.repository = MemoryRepository(str(self.db_path))
        self.service = MemoryService(self.repository, self.directory)


if __name__ == "__main__":
    unittest.main()
