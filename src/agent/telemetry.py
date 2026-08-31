"""LLM 调用级的 token / 延迟度量（P2）。

以前项目里没有任何 token / 耗时采集，后台自动化执行器和 `graph.invoke()`
路径（进度事件被 LangGraph 静默丢弃）更是完全无观测。这里补两层：

  1. `UsageTracer`（LangChain 回调）: 挂在 build_llm 的 LLM 实例上，
     每次 LLM 调用结束都记录 input/output/total token 与耗时，写入结构化日志，
     并保留在内存 records 里供测试与聚合。
  2. `traced_node` 装饰器: 给图节点计时，覆盖"没有 LLM 调用的节点"（如
     verifier / sync_context），让 invoke 路径也有端到端的节点级延迟痕迹。

数据先落 loguru 结构化日志（channel=llm_usage / node_latency），后续要接
Langfuse 等可视化时只需换 sink，数据来源不变。
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from loguru import logger


def _merge_usage(target: dict, source: dict | None) -> None:
    if not isinstance(source, dict):
        return
    for key in source:
        if key not in target:
            target[key] = source[key]


def _extract_usage(response) -> dict:
    """从 LLM 响应里尽力提取 token 用量。

    兼容 openai 兼容接口的两代字段名：
      - usage_metadata:  input_tokens / output_tokens / total_tokens
      - token_usage:     prompt_tokens / completion_tokens / total_tokens
    提取不到时返回空 dict，绝不让度量本身把业务链路打断。
    """
    usage: dict[str, Any] = {}

    # 1) llm_output（ChatOpenAI 把 token_usage 放在这里）
    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict):
        for key in ("usage", "usage_metadata", "token_usage"):
            _merge_usage(usage, llm_output.get(key))

    # 2) 每代消息的 response_metadata / generation_info
    for generation in getattr(response, "generations", None) or []:
        for chunk in generation:
            message = getattr(chunk, "message", None)
            metadata = getattr(message, "response_metadata", None) or {}
            if isinstance(metadata, dict):
                for key in ("usage_metadata", "token_usage"):
                    _merge_usage(usage, metadata.get(key))
            _merge_usage(usage, getattr(chunk, "generation_info", None) or {})

    return usage


def _extract_model(response) -> str:
    """尽力提取模型名，取不到就返回 "unknown"。

    这里必须"永远返回一个字符串"：`_log` 的格式串直接 `**record` 展开，
    loguru 在 `logger.info()` 里就地做 `str.format`，缺键会当场抛 KeyError。
    而这个异常会被 LangChain 的回调管理器吞成一行 stderr 提示
    （`Error in UsageTracer.on_llm_end callback: KeyError('model')`），
    业务链路照常跑完，token 日志却一条都不落盘 —— 度量静默失效比没有度量更糟。
    曾经就是因为格式串引用了一个从未被赋值的 `model` 键踩了这个坑。
    """
    for source in _metadata_sources(response):
        for key in ("model_name", "model"):
            value = source.get(key)
            if value:
                return str(value)
    return "unknown"


def _metadata_sources(response):
    """按优先级产出可能带元信息的 dict：llm_output → 每代消息的 response_metadata。"""
    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict):
        yield llm_output
    for generation in getattr(response, "generations", None) or []:
        for chunk in generation:
            message = getattr(chunk, "message", None)
            metadata = getattr(message, "response_metadata", None) or {}
            if isinstance(metadata, dict):
                yield metadata


def _normalize_usage(raw: dict) -> dict:
    """把两代字段名统一成 input/output/total。"""
    input_tokens = (
        raw.get("input_tokens")
        or raw.get("prompt_tokens")
    )
    output_tokens = (
        raw.get("output_tokens")
        or raw.get("completion_tokens")
    )
    total_tokens = raw.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


class UsageTracer(BaseCallbackHandler):
    """记录每次 LLM 调用的 token 用量与耗时。"""

    def __init__(self, sink: Callable[[dict], None] | None = None):
        self.records: list[dict] = []
        self._starts: dict[str, float] = {}
        self._sink = sink or self._log

    @staticmethod
    def _log(record: dict) -> None:
        logger.bind(channel="llm_usage").info(
            "model={model} | run_id={run_id} | input={input_tokens} | "
            "output={output_tokens} | total={total_tokens} | latency_ms={latency_ms}",
            **record,
        )

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._starts[str(run_id)] = time.monotonic()

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        started = self._starts.pop(str(run_id), None)
        latency_ms = round((time.monotonic() - started) * 1000, 1) if started is not None else None
        normalized = _normalize_usage(_extract_usage(response))
        record = {
            "run_id": str(run_id),
            "model": _extract_model(response),
            "latency_ms": latency_ms,
            **normalized,
        }
        self.records.append(record)
        self._sink(record)


def traced_node(name: str):
    """给图节点计时的装饰器：invoke / stream / 后台路径都会留下节点级延迟日志。

    节点函数签名（state）/（state, config）不受影响——装饰器只透传参数。
    """

    def decorate(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            started = time.monotonic()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = round((time.monotonic() - started) * 1000, 1)
                logger.bind(channel="node_latency").debug(
                    "node={name} | latency_ms={elapsed_ms}", name=name, elapsed_ms=elapsed_ms
                )

        return wrapper

    return decorate
