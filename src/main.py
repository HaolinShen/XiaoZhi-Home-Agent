"""
主入口 — 命令行交互界面
=======================
基于 Typer + Rich 构建的现代化 CLI 终端界面。

特性:
  - 美观的 Rich 格式化输出（面板、颜色、表格）
  - Typer 命令行参数解析（--model、--debug 等）
  - 完整的交互式对话循环（/help, /status, /reset, /quit）
  - 可选 MCP 服务器后台启动
  - 优雅的错误处理和用户提示

启动方式:
  # 默认启动
  python -m src.main

  # 指定模型
  python -m src.main --model qwen-max

  # 调试模式（详细日志）
  python -m src.main --debug

  # 查看所有选项
  python -m src.main --help
"""

import sys
import os
import uuid
from typing import Optional

# 确保项目根目录在 Python 路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 第三方库 ----
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich import box
from loguru import logger

# ---- 项目模块 ----
from src.config import get_settings, Settings
from src.devices import DeviceRegistry, SimulatorBackend
from src.models import DeviceType
from src.tools import set_registry as set_tools_registry
from src.tools import get_all_tools
from src.agent import build_graph
from src.agent.state import AgentState
from langchain_core.messages import HumanMessage

# ============================================================
# 全局对象
# ============================================================
app = typer.Typer(
    name="smart-home",
    help="智能家居管家 — 基于 LangGraph + MCP 的 AI Agent",
    add_completion=False,
    invoke_without_command=True,  # 无子命令时执行默认回调（启动对话）
)
console = Console()


# ============================================================
# 启动辅助函数
# ============================================================

def _print_banner(settings: Settings) -> None:
    """打印欢迎横幅"""
    banner = Panel(
        f"[bold cyan]🏠 智能家居管家 — 小智[/bold cyan]\n\n"
        f"[dim]模型:[/dim] {settings.model}\n"
        f"[dim]框架:[/dim] LangGraph + LangChain + MCP\n"
        f"[dim]平台:[/dim] 阿里百炼 (Alibaba Bailian)\n"
        f"[dim]会话:[/dim] {uuid.uuid4().hex[:8]}",
        title="Smart Home Agent v0.1.0",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print()
    console.print(banner)


def _print_help() -> None:
    """打印帮助信息"""
    help_table = Table(title="📖 使用指南", box=box.ROUNDED, border_style="dim")
    help_table.add_column("类别", style="bold cyan", width=12)
    help_table.add_column("示例", style="white")

    help_table.add_row("💡 灯光", "打开客厅灯 / 关掉卧室灯 / 把灯光调暗到30% / 灯调成白光")
    help_table.add_row("❄️ 空调", "打开客厅空调 / 空调调到25度 / 空调切到制热 / 风速调高")
    help_table.add_row("📺 电视", "打开电视 / 电视音量调到50 / 静音 / 切换到HDMI 2")
    help_table.add_row("🪟 窗帘", "打开窗帘 / 关上窗帘 / 窗帘打开一半")
    help_table.add_row("🎬 场景", "我回来了 / 我要睡了 / 看电影 / 起床了 / 我出门了")
    help_table.add_row("📊 状态", "现在家里什么状态? / 灯开着吗?")

    cmd_table = Table(title="⌨️  特殊命令", box=box.ROUNDED, border_style="dim")
    cmd_table.add_column("命令", style="bold yellow", width=16)
    cmd_table.add_column("说明", style="white")
    cmd_table.add_row("/status", "查看所有设备状态（直接查询，不经过 LLM）")
    cmd_table.add_row("/scenes", "列出所有可用的场景模式")
    cmd_table.add_row("/reset", "重置对话记忆（开始新对话）")
    cmd_table.add_row("/help", "显示此帮助信息")
    cmd_table.add_row("/quit, /exit", "退出程序")

    console.print(help_table)
    console.print(cmd_table)


def _print_device_status(registry: DeviceRegistry) -> None:
    """打印设备状态面板"""
    console.print()
    console.print(Panel(
        registry.get_status_summary(),
        title="📊 设备状态",
        border_style="green",
    ))
    console.print()


def _print_scenes() -> None:
    """打印可用场景列表"""
    from src.tools.scenes import SCENE_META
    console.print()
    table = Table(title="🎬 可用场景模式", box=box.ROUNDED, border_style="dim")
    table.add_column("场景", style="bold cyan")
    table.add_column("说明", style="white")
    for name, meta in SCENE_META.items():
        table.add_row(f"{meta['emoji']} {name}", meta['description'])
    console.print(table)
    console.print()


# ============================================================
# Agent 对话循环
# ============================================================

def run_interactive_loop(
    graph,
    registry: DeviceRegistry,
    thread_id: str,
) -> None:
    """
    交互式对话主循环。

    流程:
      1. 读取用户输入
      2. 处理特殊命令
      3. 调用 Agent 处理普通消息
      4. 显示 Agent 回复
      5. 循环
    """
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            # ---- 读取输入 ----
            try:
                user_input = console.input("\n[bold green]👤 你:[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n\n[dim]👋 再见！小智随时为你服务~[/dim]\n")
                break

            if not user_input:
                continue

            # ---- 处理特殊命令 ----
            if user_input.lower() in ("/quit", "/exit", "/q"):
                console.print("\n[bold cyan]👋 再见！小智随时为你服务~[/bold cyan]\n")
                break

            elif user_input.lower() == "/status":
                _print_device_status(registry)
                continue

            elif user_input.lower() == "/scenes":
                _print_scenes()
                continue

            elif user_input.lower() in ("/help", "/h"):
                console.print()
                _print_help()
                continue

            elif user_input.lower() == "/reset":
                # 生成新的 thread_id（新会话）
                old_id = thread_id
                new_id = f"user-{uuid.uuid4().hex[:8]}"
                config = {"configurable": {"thread_id": new_id}}
                console.print(
                    f"\n[dim]✅ 对话记忆已重置 | {old_id} → {new_id}[/dim]"
                )
                continue

            # ---- 正常对话 ----
            console.print("\n[bold cyan]🤖 小智:[/bold cyan] ", end="")

            with console.status("[dim]思考中...[/dim]", spinner="dots"):
                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config,
                )

            # 提取最终回复
            final_msg = result["messages"][-1]
            response = final_msg.content

            # 用 Markdown 渲染回复（支持表格、列表等）
            console.print(Markdown(response))

        except KeyboardInterrupt:
            console.print("\n\n[dim]👋 检测到 Ctrl+C，小智先下班啦~[/dim]\n")
            break

        except Exception as e:
            logger.error(f"对话异常: {e}", exc_info=True)
            console.print(f"\n[red]😵 出现错误: {e}[/red]")
            console.print("[dim]请检查网络和 API 配置。输入 /quit 退出。[/dim]")


# ============================================================
# 默认回调: 启动交互式对话（无子命令时自动执行）
# ============================================================

@app.callback()
def chat(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="覆盖 .env 中的模型配置（qwen-turbo / qwen-plus / qwen-max）",
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d",
        help="开启调试模式（显示详细日志）",
    ),
):
    """
    启动智能家居管家交互式对话。

    直接运行 python -m src.main 进入对话模式。
    使用 python -m src.main status 查看设备状态。
    使用 python -m src.main mcp-server 启动 MCP 服务器。
    """
    # 如果用户指定了子命令（status / mcp-server），跳过对话循环
    if ctx.invoked_subcommand is not None:
        return
    # ---- 加载配置 ----
    try:
        settings = get_settings()
    except ValueError as e:
        console.print(f"\n[red]❌ 配置错误: {e}[/red]\n")
        console.print(
            "[dim]请编辑 .env 文件，填入真实的 BAILIAN_API_KEY。\n"
            "获取地址: https://bailian.console.aliyun.com/[/dim]\n"
        )
        raise typer.Exit(code=1)

    # ---- 覆盖配置 ----
    if model:
        settings._model = model
    if debug:
        settings.log_level = "DEBUG"

    # ---- 配置日志 ----
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
        colorize=True,
    )

    # ---- 打印横幅 ----
    _print_banner(settings)

    # ---- 初始化设备注册中心 ----
    backend = SimulatorBackend()
    registry = DeviceRegistry(backend)

    # ---- 注入注册中心到工具层 ----
    set_tools_registry(registry)

    # ---- 构建 Agent 图 ----
    console.print("[dim]正在初始化 Agent...[/dim]")
    try:
        graph = build_graph(registry, settings)
    except Exception as e:
        console.print(f"\n[red]❌ Agent 初始化失败: {e}[/red]\n")
        console.print(
            "[dim]可能的原因:\n"
            "  1. API Key 无效或过期\n"
            "  2. 网络无法访问百炼 API\n"
            "  3. 模型名称不正确[/dim]\n"
        )
        raise typer.Exit(code=1)

    # ---- 打印提示 ----
    console.print()
    console.print(
        "[dim]试着说:[/dim] "
        "[bold]打开客厅灯[/bold] / "
        "[bold]空调调到25度[/bold] / "
        "[bold]我要睡觉了[/bold] / "
        "[bold]现在家里什么状态?[/bold]"
    )
    console.print("[dim]输入 /help 查看更多用法，/quit 退出[/dim]")

    # ---- 启动对话循环 ----
    thread_id = f"user-{uuid.uuid4().hex[:8]}"
    run_interactive_loop(graph, registry, thread_id)


@app.command()
def status():
    """快速查看所有设备状态（不启动对话）"""
    backend = SimulatorBackend()
    registry = DeviceRegistry(backend)
    _print_device_status(registry)


@app.command()
def mcp_server(
    transport: str = typer.Option(
        "stdio", "--transport", "-t",
        help="传输模式: stdio 或 sse",
    ),
    port: int = typer.Option(
        8765, "--port", "-p",
        help="SSE 模式端口",
    ),
):
    """启动 MCP 服务器（供 Claude Desktop 等外部客户端连接）"""
    from src.mcp.server import create_mcp_server

    backend = SimulatorBackend()
    registry = DeviceRegistry(backend)

    mcp = create_mcp_server(registry, server_name="Smart Home Agent")
    console.print(f"[bold cyan]🚀 MCP 服务器启动[/bold cyan] | transport={transport}")
    mcp.run(transport=transport, port=port if transport == "sse" else None)


# ============================================================
# 入口
# ============================================================

def main():
    """CLI 入口（兼容 python -m src.main 和 python src/main.py）"""
    app()


if __name__ == "__main__":
    main()
