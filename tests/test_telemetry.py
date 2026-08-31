"""P2 守卫用例：token/延迟采集与进度事件的双写。

钉住三件事：
  1. UsageTracer 能从 openai 兼容接口的两代字段名里提取 token 用量；
  2. emit_progress 在 stream 之外也有日志痕迹（invoke / 后台路径可查）；
  3. traced_node 装饰器保留节点原签名，并记录节点级延迟。
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.agent.telemetry import (
    UsageTracer,
    _extract_model,
    _extract_usage,
    _normalize_usage,
    traced_node,
)


class UsageExtractionTests(unittest.TestCase):
    def test_extracts_usage_metadata_from_generation_messages(self):
        message = SimpleNamespace(response_metadata={
            "usage_metadata": {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}
        })
        chunk = SimpleNamespace(message=message, generation_info={})
        response = SimpleNamespace(llm_output=None, generations=[[chunk]])
        self.assertEqual(_normalize_usage(_extract_usage(response)), {
            "input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
        })

    def test_extracts_legacy_token_usage_from_llm_output(self):
        response = SimpleNamespace(
            llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            generations=[],
        )
        self.assertEqual(_normalize_usage(_extract_usage(response)), {
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        })

    def test_missing_usage_yields_none_without_crashing(self):
        response = SimpleNamespace(llm_output=None, generations=[])
        self.assertEqual(_normalize_usage(_extract_usage(response)), {
            "input_tokens": None, "output_tokens": None, "total_tokens": None,
        })

    def test_extracts_model_name_from_llm_output(self):
        response = SimpleNamespace(llm_output={"model_name": "deepseek-chat"}, generations=[])
        self.assertEqual(_extract_model(response), "deepseek-chat")

    def test_extracts_model_name_from_response_metadata(self):
        chunk = SimpleNamespace(
            message=SimpleNamespace(response_metadata={"model": "gpt-4o-mini"}),
            generation_info={},
        )
        response = SimpleNamespace(llm_output=None, generations=[[chunk]])
        self.assertEqual(_extract_model(response), "gpt-4o-mini")

    def test_missing_model_falls_back_to_unknown(self):
        """取不到模型名也必须返回字符串——否则 _log 的格式串会缺键。"""
        response = SimpleNamespace(llm_output=None, generations=[])
        self.assertEqual(_extract_model(response), "unknown")


class UsageTracerTests(unittest.TestCase):
    def test_tracer_records_one_entry_per_llm_call(self):
        tracer = UsageTracer(sink=lambda record: None)
        tracer.on_llm_start([], [], run_id="run-1")
        response = SimpleNamespace(
            llm_output={"token_usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
            generations=[],
        )
        tracer.on_llm_end(response, run_id="run-1")
        self.assertEqual(len(tracer.records), 1)
        record = tracer.records[0]
        self.assertEqual(record["run_id"], "run-1")
        self.assertEqual(record["total_tokens"], 5)
        self.assertIsNotNone(record["latency_ms"])

    def test_tracer_sink_receives_normalized_record(self):
        seen = []
        tracer = UsageTracer(sink=seen.append)
        tracer.on_llm_start([], [], run_id="run-2")
        tracer.on_llm_end(SimpleNamespace(llm_output=None, generations=[]), run_id="run-2")
        self.assertEqual(len(seen), 1)
        self.assertIn("total_tokens", seen[0])

    def test_default_sink_actually_writes_a_log_line(self):
        """默认 sink（_log）必须真能格式化成功。

        两个用真实依赖的理由，缺一不可：
          1. 不能注入自定义 sink —— 上面两个用例都注入了，于是 `_log` 这条
             唯一有 bug 的路径从未被覆盖：格式串引用的 `model` 键当时并不存在。
          2. 不能 patch 成 MagicMock —— Mock 的 .info() 接受任何参数，
             而 KeyError 恰恰发生在 loguru 内部的 str.format 里。
        `_log` 抛出的异常会被 LangChain 回调管理器吞掉（只打一行 stderr），
        所以这类故障不会让业务失败，只会让度量静默归零，必须靠测试兜住。
        """
        from loguru import logger

        lines: list[str] = []
        sink_id = logger.add(
            lines.append,
            level="INFO",
            format="{message}",
            filter=lambda record: record["extra"].get("channel") == "llm_usage",
        )
        self.addCleanup(logger.remove, sink_id)

        tracer = UsageTracer()  # 刻意用默认 sink
        tracer.on_llm_start([], [], run_id="run-3")
        tracer.on_llm_end(
            SimpleNamespace(
                llm_output={
                    "model_name": "deepseek-chat",
                    "token_usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
                },
                generations=[],
            ),
            run_id="run-3",
        )

        self.assertEqual(len(lines), 1)
        self.assertIn("model=deepseek-chat", lines[0])
        self.assertIn("input=7", lines[0])
        self.assertIn("total=8", lines[0])


class ProgressEmissionTests(unittest.TestCase):
    def test_emit_progress_writes_log_when_stream_writer_is_silent(self):
        """invoke 模式下 get_stream_writer 返回空写入器，日志侧必须仍有痕迹。"""
        from src.agent.observability import emit_progress

        records = []
        with patch("src.agent.observability.logger") as fake_logger:
            fake_logger.bind.return_value = fake_logger
            fake_logger.debug.side_effect = (
                lambda fmt, **kw: records.append((fmt, kw))
            )
            emit_progress("planning_selected", goal="开灯")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][1]["event"], "planning_selected")
        self.assertIn("开灯", records[0][1]["payload"])


class TracedNodeTests(unittest.TestCase):
    def test_decorator_preserves_signature_and_records_latency(self):
        calls = []

        with patch("src.agent.telemetry.logger") as fake_logger:
            fake_logger.bind.return_value = fake_logger
            fake_logger.debug.side_effect = lambda fmt, **kw: calls.append((fmt, kw))

            @traced_node("test_node")
            def node(state):
                return {"ok": state["value"]}

            result = node({"value": 42})
            self.assertEqual(result, {"ok": 42})
            self.assertEqual(node.__name__, "node")

        self.assertTrue(calls)
        self.assertEqual(calls[0][1]["name"], "test_node")
        self.assertIsNotNone(calls[0][1]["elapsed_ms"])


if __name__ == "__main__":
    unittest.main()
