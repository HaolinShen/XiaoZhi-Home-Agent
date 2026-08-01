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
from typing import Optional
from loguru import logger

from langgraph.checkpoint.memory import MemorySaver

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _HAS_SQLITE = True
except ImportError:
    SqliteSaver = None
    _HAS_SQLITE = False


def create_checkpointer(db_path: Optional[str] = None):
    """
    创建检查点存储器。

    策略:
      1. 如果提供了 db_path → SQLite 持久化模式
      2. 如果 db_path 为空或创建失败 → 内存模式（fallback）

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
    if db_path and _HAS_SQLITE:
        try:
            import sqlite3
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            logger.info(f"✅ SQLite 检查点已就绪 | path={db_path}")
            return checkpointer
        except Exception as e:
            logger.warning(f"SQLite 初始化失败，回退到内存模式: {e}")

    logger.info("📝 使用内存检查点（会话记忆在重启后丢失）")
    return MemorySaver()
