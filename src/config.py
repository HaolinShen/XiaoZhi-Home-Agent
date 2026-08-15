"""
配置管理模块
============
使用 pydantic-settings 加载和管理所有配置项。

支持两种环境变量命名风格（优先级: 通用名 > 百炼专用名）:
  | 用途       | 通用名        | 百炼专用名       |
  |------------|---------------|-------------------|
  | API Key    | LLM_API_KEY   | BAILIAN_API_KEY   |
  | 服务地址   | LLM_BASE_URL  | BAILIAN_BASE_URL  |
  | 模型名称   | LLM_MODEL_ID  | BAILIAN_MODEL     |
  | 超时时间   | LLM_TIMEOUT   | -                 |

特性:
  - 自动从 .env 文件加载环境变量
  - 类型验证（避免 API Key 为空等低级错误）
  - 嵌套配置（MCP、记忆等子配置）
  - IDE 友好（完整的类型提示和文档字符串）
"""

from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# 子配置：MCP 服务器
# ============================================================
class MCPServerConfig(BaseSettings):
    """内置 MCP 服务器的配置"""
    model_config = SettingsConfigDict(env_prefix="MCP_SERVER_")

    enabled: bool = Field(
        default=True,
        description="是否在 Agent 启动时同时启动 MCP 服务器",
    )
    port: int = Field(
        default=8765,
        ge=1024, le=65535,
        description="MCP SSE 模式的监听端口（stdio 模式忽略此值）",
    )


# ============================================================
# 子配置：记忆
# ============================================================
class MemoryConfig(BaseSettings):
    """对话记忆与持久化配置"""
    model_config = SettingsConfigDict(env_prefix="CHECKPOINT_")

    db_path: str = Field(
        default="data/checkpoints.db",
        description="SQLite 检查点数据库路径（设为空使用内存模式）",
    )
    long_term_db_path: str = Field(
        default="data/memories.db",
        description="SQLite 长期记忆数据库路径",
    )
    enable_long_term: bool = Field(
        default=True,
        validation_alias="ENABLE_LONG_TERM_MEMORY",
        description="是否启用长期记忆",
    )
    context_max_messages: int = Field(default=12, ge=2)
    context_max_tokens: int = Field(default=2400, ge=100)
    tool_result_max_chars: int = Field(default=1200, ge=100)
    summary_max_chars: int = Field(default=1800, ge=200)
    session_ttl_hours: int = Field(default=168, ge=1)
    retrieval_top_k: int = Field(default=6, ge=1, le=20)


class PlanningConfig(BaseSettings):
    """Planner–Executor–Verifier loop limits."""
    model_config = SettingsConfigDict(env_prefix="PLANNING_")

    enabled: bool = Field(default=True)
    max_steps: int = Field(default=8, ge=2, le=12)
    max_step_retries: int = Field(default=1, ge=0, le=3)
    max_replans: int = Field(default=1, ge=0, le=3)


class RoutingConfig(BaseSettings):
    """Structured intent router settings."""
    model_config = SettingsConfigDict(env_prefix="ROUTING_")

    enabled: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.6, ge=0, le=1)


class MultiAgentConfig(BaseSettings):
    """Supervisor and specialised-agent collaboration limits."""
    model_config = SettingsConfigDict(env_prefix="MULTI_AGENT_")

    enabled: bool = Field(default=True)
    max_handoffs: int = Field(default=2, ge=1, le=5)


class RAGConfig(BaseSettings):
    """Local Agentic RAG settings."""
    model_config = SettingsConfigDict(env_prefix="RAG_")

    enabled: bool = Field(default=True)
    knowledge_path: str = Field(default="docs/knowledge")
    top_k: int = Field(default=3, ge=1, le=10)
    max_rewrites: int = Field(default=1, ge=0, le=3)


class AutomationConfig(BaseSettings):
    """Persistent event-driven routine scheduler settings."""
    model_config = SettingsConfigDict(env_prefix="AUTOMATION_")

    enabled: bool = Field(default=True)
    db_path: str = Field(default="data/automation.db")
    timezone: str = Field(default="Asia/Shanghai")
    poll_seconds: float = Field(default=1.0, ge=0.1, le=60)


# ============================================================
# 主配置
# ============================================================
class Settings(BaseSettings):
    """
    应用主配置。

    环境变量读取优先级:
      1. 通用名 (LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL)
      2. 百炼专用名 (BAILIAN_API_KEY, BAILIAN_MODEL, BAILIAN_BASE_URL)
      3. 代码默认值

    示例 .env:
      LLM_MODEL_ID=qwen-plus
      LLM_API_KEY=sk-xxxxxxxx
      LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- LLM API 配置（支持双命名）----
    # 注意: 通用名和百炼专用名都定义为可选字段，
    # 在 __init__ 后通过 _resolve_names() 统一解析

    # 通用名（优先）
    llm_api_key: Optional[str] = Field(
        default=None,
        alias="LLM_API_KEY",
        description="LLM API Key（通用名，优先）",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        alias="LLM_BASE_URL",
        description="LLM 服务地址（通用名，优先）",
    )
    llm_model_id: Optional[str] = Field(
        default=None,
        alias="LLM_MODEL_ID",
        description="模型名称（通用名，优先）",
    )
    llm_timeout: int = Field(
        default=60,
        alias="LLM_TIMEOUT",
        description="LLM 请求超时秒数",
    )

    # 百炼专用名（fallback）
    bailian_api_key: Optional[str] = Field(
        default=None,
        alias="BAILIAN_API_KEY",
        description="百炼 API Key（备选）",
    )
    bailian_base_url: Optional[str] = Field(
        default=None,
        alias="BAILIAN_BASE_URL",
        description="百炼服务地址（备选）",
    )
    bailian_model: Optional[str] = Field(
        default=None,
        alias="BAILIAN_MODEL",
        description="百炼模型名（备选）",
    )

    # ---- 日志 ----
    log_level: str = Field(
        default="INFO",
        description="日志级别: DEBUG / INFO / WARNING / ERROR",
    )

    # ---- 外部 MCP 服务 ----
    external_mcp_servers: str = Field(
        default="",
        description="外部 MCP 服务配置（JSON 字符串，多个用逗号分隔）",
    )

    # ---- 子配置 ----
    mcp_server: MCPServerConfig = Field(default_factory=MCPServerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    multi_agent: MultiAgentConfig = Field(default_factory=MultiAgentConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    automation: AutomationConfig = Field(default_factory=AutomationConfig)

    # ---- 最终解析后的值（非 Field，用于内部使用）----
    _api_key: str = ""
    _base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    _model: str = "qwen-plus"

    def model_post_init(self, __context) -> None:
        """
        Pydantic 初始化后回调: 统一解析双命名变量。

        优先级: LLM_* > BAILIAN_* > 默认值
        """
        # API Key
        self._api_key = (
            self.llm_api_key
            or self.bailian_api_key
            or ""
        )
        # 去掉可能的外层引号（兼容 .env 中带引号的值）
        self._api_key = self._api_key.strip().strip('"').strip("'")

        # Base URL
        self._base_url = (
            self.llm_base_url
            or self.bailian_base_url
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self._base_url = self._base_url.strip().strip('"').strip("'")

        # Model
        self._model = (
            self.llm_model_id
            or self.bailian_model
            or "qwen-plus"
        )
        self._model = self._model.strip().strip('"').strip("'")

        # 验证 API Key
        if self._api_key in ("", "sk-your-api-key-here", "your-api-key"):
            raise ValueError(
                "请在 .env 文件中设置 LLM_API_KEY 或 BAILIAN_API_KEY。\n"
                "获取地址: https://bailian.console.aliyun.com/"
            )

    # ---- 便捷属性 ----
    @property
    def api_key(self) -> str:
        """最终的 API Key"""
        return self._api_key

    @property
    def base_url(self) -> str:
        """最终的服务地址"""
        return self._base_url

    @property
    def model(self) -> str:
        """最终的模型名称"""
        return self._model

    # 向后兼容别名
    @property
    def bailian_api_key_resolved(self) -> str:
        return self._api_key

    @property
    def bailian_base_url_resolved(self) -> str:
        return self._base_url

    @property
    def bailian_model_resolved(self) -> str:
        return self._model


# ============================================================
# 全局配置单例
# ============================================================
settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    获取配置单例（懒加载）。
    第一次调用时完成验证，后续直接返回缓存。

    使用:
      from src.config import get_settings
      cfg = get_settings()
      print(cfg.model)    # qwen-plus
      print(cfg.api_key)  # sk-xxx
    """
    global settings
    if settings is None:
        settings = Settings()
    return settings
