"""P3 守卫用例：收敛到 heuristics.py 的三个判定，行为与迁移前逐字一致。

routing 兜底、Planner 判定、自动化强制工具三类启发式的关键词表以前散落在
routing.py / planning.py / graph.py，现在同源于 heuristics.py。这里钉住：
  1. 每个判定的代表性输入输出（迁移前后必须一致）；
  2. 共享词表（ACTION_CORE）确实被三方复用 —— 新增关键词时能同时看到
     所有受影响方。
"""

import unittest

from src.agent.heuristics import (
    ACTION_CORE,
    AUTOMATION_ACTION_MARKERS,
    PLANNER_ACTION_PATTERNS,
    ROUTING_CONTROL_WORDS,
    required_automation_tool,
    should_use_planner,
)
from src.agent.routing import classify_intent_fallback


class PlannerHeuristicTests(unittest.TestCase):
    def test_multi_action_custom_requests_route_to_planner(self):
        self.assertTrue(should_use_planner("关闭客厅灯，然后打开卧室空调到25度"))
        self.assertTrue(should_use_planner("打开客厅灯并且打开电视"))
        self.assertFalse(should_use_planner("关闭客厅灯"))
        self.assertFalse(should_use_planner("打开空调"))

    def test_predefined_scene_phrases_never_route_to_planner(self):
        for text in ("我要出门了", "开启离家模式", "我要睡了", "看电影", "起床了", "我回来了"):
            with self.subTest(text=text):
                self.assertFalse(should_use_planner(text))

    def test_empty_input_does_not_route_to_planner(self):
        self.assertFalse(should_use_planner(""))
        self.assertFalse(should_use_planner("   "))


class RoutingFallbackTests(unittest.TestCase):
    def test_automation_phrases_and_future_times_are_deterministic(self):
        for text in (
            "明天早上6点设置闹钟并准备热水",
            "车辆到家前提前打开空调",
            "帮我查看有哪些自动化",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    classify_intent_fallback(text).intent, "automation_management"
                )

    def test_memory_knowledge_scene_query_and_control_words(self):
        cases = {
            "记住我喜欢暖光": "memory_management",
            "空调显示故障代码E1是什么意思": "device_knowledge",
            "开启睡眠模式": "scene_control",
            "现在屋里温度多少": "device_query",
            "打开客厅灯": "device_control",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(classify_intent_fallback(text).intent, expected)

    def test_empty_and_tiny_inputs_fall_back_to_clarification(self):
        self.assertEqual(classify_intent_fallback("").intent, "clarification")
        self.assertEqual(classify_intent_fallback("好").intent, "clarification")


class AutomationRequiredToolTests(unittest.TestCase):
    def test_query_and_cancel_requests_are_never_forced_to_create(self):
        for text in (
            "当前有多少个定时任务",
            "现在有几个定时任务",
            "帮我看看有哪些自动化",
            "列出所有定时例程",
            "查询一下明天的定时任务",
            "现在有没有定时任务",
            "取消明天的起床计划",
            "删除那个定时任务",
            "停用车辆回家例程",
        ):
            with self.subTest(text=text):
                self.assertIsNone(required_automation_tool(text))

    def test_real_creation_requests_lock_the_matching_tool(self):
        self.assertEqual(
            required_automation_tool("我今天下午5点打球回到家，帮我提前准备洗澡水，同时提前打开客厅空调降温"),
            "create_scheduled_routine",
        )
        self.assertEqual(required_automation_tool("明天早上7点叫我起床"), "schedule_wake_routine")
        self.assertEqual(
            required_automation_tool("车辆到家前提前打开空调"),
            "create_vehicle_arrival_routine",
        )

    def test_trigger_without_action_is_not_forced(self):
        self.assertIsNone(required_automation_tool("定时任务"))
        self.assertIsNone(required_automation_tool(""))


class SharedWordTableTests(unittest.TestCase):
    def test_all_three_heuristics_share_the_core_action_words(self):
        """共享词表被三方复用：核心动作词改一处，三个判定同时看到。"""
        for word in ACTION_CORE:
            with self.subTest(word=word):
                self.assertIn(word, PLANNER_ACTION_PATTERNS)
                self.assertIn(word, ROUTING_CONTROL_WORDS)
                self.assertIn(word, AUTOMATION_ACTION_MARKERS)

    def test_each_heuristic_extends_core_with_its_own_semantics(self):
        # Planner 独有：窗帘/静音/切换这类精确动作（意图线索不需要，精确动作才要）。
        self.assertIn("拉开", PLANNER_ACTION_PATTERNS)
        self.assertNotIn("拉开", ROUTING_CONTROL_WORDS)
        # 路由独有：宽泛的控制线索。
        self.assertIn("设置", ROUTING_CONTROL_WORDS)
        # 自动化独有：定时语义下的动作表达。
        self.assertIn("预热", AUTOMATION_ACTION_MARKERS)
        self.assertNotIn("预热", PLANNER_ACTION_PATTERNS)


if __name__ == "__main__":
    unittest.main()
