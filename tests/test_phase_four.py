import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from src.agent.context import AgentContext, DeviceLocation, SpaceDirectory
from src.memory import MemoryRepository, MemoryScope, MemoryService, MemoryType, MemoryWrite
from src.memory.models import utc_now
from src.tools.memory import build_memory_tools
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.tools.devices import build_device_tools


class PhaseFourTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = MemoryRepository(str(Path(self.temp_dir.name) / "memory.db"))
        self.directory = SpaceDirectory(
            {"home-a": {"bedroom"}},
            {"ac-a": DeviceLocation("home-a", "bedroom")},
        )
        self.service = MemoryService(self.repository, self.directory)
        self.context = AgentContext(
            home_id="home-a", user_id="user-a", session_id="session-a",
            client_id="phone", room_id="bedroom", device_id="ac-a",
        )

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    def memory_tools(self):
        """P1: 记忆工具由工厂显式构建（闭包持有 service，无模块级单例）。"""
        return {tool.name: tool for tool in build_memory_tools(self.service)}

    def test_repeated_operations_create_candidate_but_not_memory(self):
        self.assertIsNone(self.service.record_operation(
            self.context, "ac.temperature", {"temperature": 25}
        ))
        self.assertIsNone(self.service.record_operation(
            self.context, "ac.temperature", {"temperature": 25}
        ))
        candidate = self.service.record_operation(
            self.context, "ac.temperature", {"temperature": 25}
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.observation_count, 3)
        self.assertEqual(self.service.list(self.context), [])

    def test_successful_device_settings_feed_candidate_observations(self):
        registry = DeviceRegistry(SimulatorBackend())
        graph_directory = SpaceDirectory.from_registry(registry, "home-a")
        self.service.spaces = graph_directory
        # P1: 设备工具显式持有 memory_service，偏好观察开启（图路径语义）。
        device_tools = {
            tool.name: tool
            for tool in build_device_tools(registry, self.service)
        }
        tool_context = self.context.model_copy(update={
            "room_id": "living_room", "device_id": "living_room_ac"
        })
        for _ in range(3):
            device_tools["control_ac"].invoke(
                {"device_name": "客厅空调", "action": "set_temp", "temperature": 25},
                config=tool_context.to_config(),
            )
        candidates = self.service.list_candidates(tool_context)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].memory_key, "ac.temperature")

    def test_confirm_candidate_saves_memory_and_reject_does_not(self):
        for _ in range(3):
            first = self.service.record_operation(
                self.context, "ac.temperature", {"temperature": 25}
            )
        record = self.service.confirm_candidate(self.context, first.id)
        self.assertEqual(record.memory_value, {"temperature": 25})
        self.assertEqual(self.service.list_candidates(self.context), [])

        for _ in range(3):
            second = self.service.record_operation(
                self.context, "ac.fan", {"fan_speed": "low"}
            )
        self.assertTrue(self.service.reject_candidate(self.context, second.id))
        self.assertEqual([item.memory_key for item in self.service.list(self.context)], ["ac.temperature"])

    def test_candidates_are_isolated_by_user_and_tools_require_trusted_context(self):
        for _ in range(3):
            candidate = self.service.record_operation(
                self.context, "ac.temperature", {"temperature": 25}
            )
        other = self.context.model_copy(update={"user_id": "user-b", "session_id": "session-b"})
        self.assertEqual(self.service.list_candidates(other), [])
        with self.assertRaises(KeyError):
            self.service.confirm_candidate(other, candidate.id)

        tools = self.memory_tools()
        listed = tools["list_preference_candidates"].invoke({}, config=self.context.to_config())
        self.assertIn(candidate.id, listed)
        confirmed = tools["confirm_preference_candidate"].invoke(
            {"candidate_id": candidate.id}, config=self.context.to_config()
        )
        self.assertIn("已确认并保存偏好", confirmed)

    def test_conflicts_are_recorded_and_complementary_values_are_merged(self):
        original = self.service.save(self.context, MemoryWrite(
            scope=MemoryScope.USER, memory_type=MemoryType.PREFERENCE,
            memory_key="ac.comfort", memory_value={"temperature": 25, "mode": "cool"},
            room_id="bedroom", device_id="ac-a",
        ))
        updated = self.service.save(self.context, MemoryWrite(
            scope=MemoryScope.USER, memory_type=MemoryType.PREFERENCE,
            memory_key="ac.comfort", memory_value={"temperature": 26, "fan": "low"},
            room_id="bedroom", device_id="ac-a",
        ))
        self.assertEqual(updated.id, original.id)
        self.assertEqual(updated.memory_value, {
            "temperature": 26, "mode": "cool", "fan": "low"
        })
        conflicts = self.repository.list_conflicts("home-a", original.id)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].resolution, "merged")

    def test_stale_confidence_decays_with_floor(self):
        record = self.service.save(self.context, MemoryWrite(
            scope=MemoryScope.USER, memory_type=MemoryType.PREFERENCE,
            memory_key="ac.temperature", memory_value={"temperature": 25},
            room_id="bedroom", device_id="ac-a", confidence=0.8,
        ))
        old = (utc_now() - timedelta(days=100)).isoformat()
        self.repository.connection.execute(
            "UPDATE memories SET updated_at=? WHERE id=?", (old, record.id)
        )
        self.repository.connection.commit()
        self.assertEqual(self.service.decay_stale_confidence(), 1)
        self.assertAlmostEqual(self.service.get(self.context, record.id).confidence, 0.72)

    def test_vector_retrieval_is_only_recommended_after_scale_threshold(self):
        report = self.service.evaluate_vector_retrieval("home-a", threshold=1)
        self.assertFalse(report["recommend_vector_retrieval"])
        self.service.save(self.context, MemoryWrite(
            scope=MemoryScope.USER, memory_type=MemoryType.PREFERENCE,
            memory_key="ac.temperature", memory_value={"temperature": 25},
            room_id="bedroom", device_id="ac-a",
        ))
        report = self.service.evaluate_vector_retrieval("home-a", threshold=1)
        self.assertTrue(report["recommend_vector_retrieval"])
        self.assertEqual(report["reason"], "memory_scale_threshold_reached")

    def test_reject_tool_changes_candidate_status(self):
        for _ in range(3):
            candidate = self.service.record_operation(
                self.context, "ac.temperature", {"temperature": 25}
            )
        tools = self.memory_tools()
        result = tools["reject_preference_candidate"].invoke(
            {"candidate_id": candidate.id}, config=self.context.to_config()
        )
        self.assertEqual(result, "已拒绝该偏好候选")


if __name__ == "__main__":
    unittest.main()
