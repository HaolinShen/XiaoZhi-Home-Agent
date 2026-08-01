"""
中间件模块
==========
提供 Agent 运行时的横切关注点（Cross-cutting Concerns）处理。

中间件栈:
  ┌──────────────────┐
  │  LoggingInterceptor  │ ← 记录每次 LLM 调用和工具执行
  ├──────────────────┤
  │  RetryInterceptor    │ ← LLM 调用失败时自动重试
  ├──────────────────┤
  │  MetricsInterceptor  │ ← (规划中) 收集性能指标
  └──────────────────┘

设计模式: 装饰器模式 (Decorator Pattern)
  每个中间件独立封装一种能力，可自由组合。
"""

import time
from typing import Any, Callable
from loguru import logger


# ============================================================
# 日志中间件
# ============================================================

class LoggingInterceptor:
    """
    Agent 调用日志记录器。

    记录每次 Agent 调用的:
      - 输入内容（摘要）
      - 处理耗时
      - 输出结果（摘要）

    用途:
      - 调试: 追踪 Agent 的决策过程
      - 监控: 发现异常慢的调用
      - 审计: 记录用户操作历史
    """

    @staticmethod
    def wrap(func: Callable) -> Callable:
        """
        包装一个函数，添加日志记录。

        使用方式:
          @LoggingInterceptor.wrap
          def my_agent_call(messages):
              ...
        """
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()

            # 记录输入（截断过长内容）
            input_preview = _truncate(str(args[0]) if args else str(kwargs), 200)
            logger.info(f"🚀 Agent 调用开始 | input={input_preview}")

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                # 记录输出（截断过长内容）
                output_preview = _truncate(str(result), 200)
                logger.info(
                    f"✅ Agent 调用完成 | elapsed={elapsed:.2f}s | output={output_preview}"
                )
                return result

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"❌ Agent 调用失败 | elapsed={elapsed:.2f}s | error={e}"
                )
                raise

        return wrapper


# ============================================================
# 重试中间件
# ============================================================

class RetryInterceptor:
    """
    LLM 调用失败自动重试器。

    处理以下类型的错误:
      - 网络超时（百炼 API 临时不可达）
      - 速率限制（并发过高被限流）
      - 服务端临时错误（5xx）

    重试策略:
      - 最多重试 3 次
      - 指数退避: 1s → 2s → 4s
      - 可通过环境变量配置最大重试次数
    """

    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # 基础等待秒数

    @classmethod
    def wrap(cls, func: Callable, max_retries: int = None) -> Callable:
        """
        包装一个函数，添加自动重试。

        参数:
          func: 要包装的异步或同步函数
          max_retries: 最大重试次数（默认 3）
        """
        retries = max_retries if max_retries is not None else cls.MAX_RETRIES

        def wrapper(*args, **kwargs) -> Any:
            last_error = None

            for attempt in range(retries + 1):
                try:
                    if attempt > 0:
                        delay = cls.BASE_DELAY * (2 ** (attempt - 1))
                        logger.info(f"🔄 第 {attempt} 次重试 | delay={delay:.1f}s")
                        time.sleep(delay)

                    return func(*args, **kwargs)

                except Exception as e:
                    last_error = e
                    error_type = type(e).__name__
                    if attempt < retries:
                        logger.warning(
                            f"⚠️ 调用失败，准备重试 | "
                            f"attempt={attempt+1}/{retries} | "
                            f"error={error_type}: {e}"
                        )
                    else:
                        logger.error(
                            f"❌ 重试耗尽 | attempts={retries+1} | "
                            f"error={error_type}: {e}"
                        )

            raise last_error  # type: ignore[misc]

        return wrapper


# ============================================================
# 工具函数
# ============================================================

def _truncate(text: str, max_len: int = 200) -> str:
    """截断过长文本，避免日志爆炸"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...(共 {len(text)} 字)"


def apply_all_middleware(func: Callable) -> Callable:
    """
    应用所有中间件。

    中间件执行顺序（从外到内）:
      1. RetryInterceptor → 最外层，捕获所有异常并重试
      2. LoggingInterceptor → 记录调用和耗时
      3. 原始函数 → 实际执行

    使用方式:
      original_func = lambda: llm.invoke(messages)
      safeguarded = apply_all_middleware(original_func)
      result = safeguarded()
    """
    # 从外到内包装
    wrapped = RetryInterceptor.wrap(func)
    wrapped = LoggingInterceptor.wrap(wrapped)
    return wrapped
