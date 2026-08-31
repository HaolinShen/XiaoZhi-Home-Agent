"""
记忆存储模块
============
管理 Agent 的对话记忆和历史记录。

记忆层级:
  1. 短期记忆（Checkpoint）: LangGraph 内置的检查点机制
     - 保存每次对话的状态（消息列表）
     - 自动在多轮对话中恢复上下文
     - 支持内存模式（快速，重启丢失）和 SQLite 模式（持久化）

  2. 长期记忆（Long-term Memory）: 用户偏好学习（规划中）
     - 记住用户习惯（如"喜欢暖光"、"通常设 25°C"）
     - 跨会话保留
     - 可以用于个性化推荐

当前实现: 短期记忆（LangGraph Checkpoint）
规划中:   长期记忆（向量数据库 + 用户画像）
"""

import os
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _HAS_SQLITE = True
except ImportError:
    SqliteSaver = None  # type: ignore[assignment,misc]  # 条件导入的标准回退
    _HAS_SQLITE = False


def create_checkpointer(db_path: str | None = None):
    """
    创建检查点存储器。

    SQLite is explicit: a configured path must initialize successfully.

    参数:
      db_path: SQLite 数据库文件路径。
               None 或 "" 表示使用内存模式。
               数据库文件不存在时会自动创建。

    返回:
      MemorySaver 或 SqliteSaver 实例

    使用示例:
      # 持久化模式
      checkpointer = create_checkpointer("data/checkpoints.db")

      # 内存模式
      checkpointer = create_checkpointer(None)
    """
    if db_path:
        if not _HAS_SQLITE:
            raise RuntimeError(
                "SQLite checkpointing is configured but "
                "langgraph-checkpoint-sqlite is not installed"
            )
        try:
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            logger.info(f"✅ SQLite 检查点已就绪 | path={db_path}")
            return checkpointer
        except Exception as exc:
            raise RuntimeError(
                f"failed to initialize SQLite checkpointer at {db_path!r}: {exc}"
            ) from exc

    logger.info("📝 使用内存检查点（会话记忆在重启后丢失）")
    return MemorySaver()


def close_checkpointer(checkpointer) -> None:
    """Close the underlying SQLite connection when one is present."""
    connection = getattr(checkpointer, "conn", None)
    if connection is not None:
        connection.close()


def cleanup_expired_checkpoints(
    checkpointer,
    ttl: timedelta,
    *,
    now: datetime | None = None,
) -> int:
    """Delete checkpoint threads whose latest snapshot is older than ``ttl``."""
    now = now or datetime.now(UTC)
    latest_by_thread: dict[str, Any] = {}
    for item in checkpointer.list(None):
        configurable = item.config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if not thread_id:
            continue
        timestamp = item.checkpoint.get("ts")
        if timestamp:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            previous = latest_by_thread.get(thread_id)
            if previous is None or parsed > previous:
                latest_by_thread[thread_id] = parsed

    expired = [
        thread_id
        for thread_id, timestamp in latest_by_thread.items()
        if now - timestamp.astimezone(UTC) >= ttl
    ]
    for thread_id in expired:
        checkpointer.delete_thread(thread_id)
    return len(expired)
