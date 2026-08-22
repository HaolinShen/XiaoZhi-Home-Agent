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
from datetime import timedelta
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
from rich import box
from loguru import logger

# ---- 项目模块 ----
from src.config import get_settings, Settings
from src.devices import DeviceRegistry, SimulatorBackend
from src.progress_view import PlanProgressView, format_arguments, format_state
from src.tools import set_registry as set_tools_registry
from src.tools import set_automation_runtime
from src.automation.runtime import AutomationRuntime
from src.mcp import load_external_tools
from src.agent import (
    AgentContext,
    SessionManager,
    SpaceDirectory,
    build_agent_request,
    build_graph,
)
from langchain_core.messages import HumanMessage
from langgraph.types import Command

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

def _print_banner(settings: Settings, context: AgentContext) -> None:
    """打印欢迎横幅"""
    banner = Panel(
        f"[bold cyan]🏠 智能家居管家 — 小智[/bold cyan]\n\n"
        f"[dim]模型:[/dim] {settings.model}\n"
        f"[dim]框架:[/dim] LangGraph + LangChain + MCP\n"
        f"[dim]平台:[/dim] 阿里百炼 (Alibaba Bailian)\n"
        f"[dim]住宅:[/dim] {context.home_id}\n"
        f"[dim]用户:[/dim] {context.user_id}\n"
        f"[dim]会话:[/dim] {context.session_id}",
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
    help_table.add_row("🌡️ 传感器", "屋里多少度 / 客厅湿度怎么样 / 家里有人吗 / 有点干（会先读数再决定）")
    help_table.add_row("🎬 场景", "我回来了 / 我要睡了 / 看电影 / 起床了 / 我出门了")
    help_table.add_row("📊 状态", "现在家里什么状态? / 灯开着吗?")

    cmd_table = Table(title="⌨️  特殊命令", box=box.ROUNDED, border_style="dim")
    cmd_table.add_column("命令", style="bold yellow", width=16)
    cmd_table.add_column("说明", style="white")
    cmd_table.add_row("/status", "查看所有设备状态（直接查询，不经过 LLM）")
    cmd_table.add_row("/scenes", "列出所有可用的场景模式")
    cmd_table.add_row("/reset", "重置对话记忆（开始新对话）")
    cmd_table.add_row("/history", "查看当前会话最近的 Checkpoint 状态历史")
    cmd_table.add_row("/plan", "复盘最近一次多步骤计划：Planner 产出 + 逐步验证结果")
    cmd_table.add_row("/routines", "查看起床/车辆回家例程及其待执行任务")
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


def _print_checkpoint_history(graph, config: dict) -> None:
    from src.agent.time_travel import list_state_history
    history = list_state_history(graph, config, limit=10)
    table = Table(title="🕰️ Checkpoint 状态历史", box=box.ROUNDED, border_style="cyan")
    table.add_column("Checkpoint", style="dim")
    table.add_column("时间")
    table.add_column("下一节点")
    for item in history:
        table.add_row(
            str(item["checkpoint_id"] or "-")[:12],
            str(item["created_at"] or "-"),
            ", ".join(item["next"]) or "END",
        )
    console.print(table)


def _print_last_plan(graph, config: dict) -> None:
    """打印本会话最近一次计划的逐步执行与验证结果。

    进度事件是"流过去就没了"，这个命令从 checkpoint 里把同一份轨迹再取出来，
    用于事后复盘：哪一步重试过、Verifier 比对的期望值和实测值分别是什么。
    """
    values = graph.get_state(config).values
    plan = values.get("plan")
    if not plan:
        console.print("\n[dim]本会话还没有生成过多步骤计划。[/dim]")
        return

    console.print()
    console.print(Panel(
        f"[bold]目标[/bold] {plan.get('goal', '')}\n"
        f"[dim]理由[/dim] {plan.get('rationale') or '—'}\n"
        f"[dim]版本[/dim] v{values.get('plan_revision', 1)}"
        f" · [dim]状态[/dim] {values.get('planning_status', 'unknown')}"
        f" · [dim]重新规划[/dim] {values.get('replan_count', 0)} 次",
        title="📋 最近一次计划",
        border_style="magenta",
    ))

    plan_table = Table(
        title="Planner 产出（执行前就已确定）", box=box.ROUNDED,
        border_style="magenta", title_justify="left",
    )
    plan_table.add_column("步", width=3, justify="right", style="bold")
    plan_table.add_column("要做什么", style="white")
    plan_table.add_column("工具", style="cyan")
    plan_table.add_column("参数", style="dim")
    for step in plan.get("steps", []):
        plan_table.add_row(
            str(step.get("step_id", "?")),
            str(step.get("description", "")),
            str(step.get("tool_name", "")),
            format_arguments(step.get("arguments")),
        )
    console.print(plan_table)

    results = values.get("planning_results", [])
    if not results:
        console.print("[dim]计划尚未执行（可能仍在等待确认，或已被拒绝）。[/dim]")
        return

    result_table = Table(
        title="Executor + Verifier 轨迹（每次尝试一行）", box=box.ROUNDED,
        border_style="blue", title_justify="left",
    )
    result_table.add_column("计划", width=4, justify="right", style="dim")
    result_table.add_column("步", width=3, justify="right", style="bold")
    result_table.add_column("工具返回", style="white")
    result_table.add_column("验证", width=6)
    result_table.add_column("期望 vs 实测", style="dim")
    for item in results:
        verification = item.get("verification", {})
        ok = verification.get("success")
        result_table.add_row(
            f"v{item.get('plan_revision', 1)}",
            str(item.get("step_id", "?")),
            str(item.get("tool_result", "")),
            "[green]通过[/green]" if ok else f"[red]{verification.get('problem_type', '失败')}[/red]",
            f"{format_state(verification.get('expected_state'))}"
            f" {'≡' if ok else '≠'} "
            f"{format_state(verification.get('actual_state'))}",
        )
    console.print(result_table)


def _print_routines(runtime: AutomationRuntime | None, home_id: str) -> None:
    if runtime is None:
        console.print("[dim]自动化调度器未启用。[/dim]")
        return
    routines = runtime.store.list_routines(home_id)
    table = Table(title="⏰ 自动化例程", box=box.ROUNDED, border_style="cyan")
    table.add_column("ID", style="dim")
    table.add_column("名称")
    table.add_column("触发器")
    table.add_column("任务状态")
    for routine in routines:
        tasks = runtime.store.list_tasks(routine.id)
        counts: dict[str, int] = {}
        for task in tasks:
            counts[task.status] = counts.get(task.status, 0) + 1
        summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "尚未触发"
        table.add_row(routine.id[:10], routine.name, routine.trigger_type, summary)
    console.print(table)


def _ask_for_approval(payload: dict) -> bool:
    """Render one approval request and return a strict yes/no decision."""
    console.print()
    console.print(Panel(
        payload.get("question", "该操作需要确认，是否继续？"),
        title=f"⚠️ 操作确认 · {payload.get('risk_level', 'unknown')}",
        border_style="yellow",
    ))
    answer = console.input(
        "[bold yellow]输入 y 确认执行，直接回车或其他键取消：[/bold yellow] "
    ).strip().lower()
    return answer in {"y", "yes", "确认", "同意", "继续", "执行", "确定", "好"}


def _stream_segment(graph, payload, config: dict, view: PlanProgressView) -> dict | None:
    """流式跑图的一段，边跑边渲染进度，返回遇到的第一个 interrupt。

    用 stream 而不是 invoke 是关键：只有 stream 才会把节点里
    `emit_progress` 发出的 custom 事件交给调用方。invoke 模式下这些事件
    会被 LangGraph 丢弃，Planner / Executor / Verifier 的分工也就看不见了。
    """
    pending = None
    # status 是 rich 的 Live 区域：view 打印的进度会正常滚动在上方，
    # 只有"思考中"这一行留在底部，所以等 LLM 时依旧有反馈。
    with console.status("[dim]思考中...[/dim]", spinner="dots"):
        for mode, chunk in graph.stream(
            payload, config, stream_mode=["custom", "updates"]
        ):
            if mode == "custom":
                view.handle(chunk)
            elif mode == "updates":
                interrupts = chunk.get("__interrupt__") if isinstance(chunk, dict) else None
                if interrupts:
                    pending = interrupts[0].value
    return pending


def _invoke_with_approval(
    graph, state_input: dict, config: dict, view: PlanProgressView
) -> dict:
    """Stream a graph run and resume any approval interrupts on the same thread."""
    view.reset()
    pending = _stream_segment(graph, state_input, config, view)
    while pending:
        approved = _ask_for_approval(pending)
        pending = _stream_segment(
            graph, Command(resume={"approved": approved}), config, view
        )
    # 进度事件已经边跑边打了，最终状态从 checkpoint 读一次即可。
    return graph.get_state(config).values


def _release_runtime_resources(automation_runtime, graph) -> None:
    """释放 CLI 持有的后台线程和 SQLite 连接。

    长期记忆库由 build_graph 内部创建，只作为 graph.memory_repository 暴露出来，
    所以退出时必须由调用方来关 —— 之前这里只关了 automation_runtime，那个连接
    一直靠进程退出兜底，一旦 build_graph 被用在长驻服务里就是泄漏。
    graph 可能为 None（初始化过程中途失败），memory_repository 也可能为 None
    （关掉了长期记忆），两者都用 getattr 兜住。
    """
    if automation_runtime is not None:
        automation_runtime.close()
        set_automation_runtime(None)
    repository = getattr(graph, "memory_repository", None)
    if repository is not None:
        repository.close()


# ============================================================
# Agent 对话循环
# ============================================================

def run_interactive_loop(
    graph,
    registry: DeviceRegistry,
    context: AgentContext,
    sessions: SessionManager,
    view: PlanProgressView,
    automation_runtime: AutomationRuntime | None = None,
) -> None:
    """
    交互式对话主循环。

    流程:
      1. 读取用户输入
      2. 处理特殊命令
      3. 流式调用 Agent，边跑边渲染 Planner / Executor / Verifier 过程
      4. 显示 Agent 最终回复
      5. 循环
    """
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
                old_id = context.session_id
                context = sessions.create(
                    home_id=context.home_id,
                    user_id=context.user_id,
                    client_id=context.client_id,
                    room_id=context.room_id,
                    device_id=context.device_id,
                )
                console.print(
                    f"\n[dim]✅ 已创建新会话 | {old_id} → {context.session_id}[/dim]"
                )
                continue

            elif user_input.lower() == "/history":
                _print_checkpoint_history(graph, context.to_config())
                continue

            elif user_input.lower() == "/plan":
                _print_last_plan(graph, context.to_config())
                continue

            elif user_input.lower() == "/routines":
                _print_routines(automation_runtime, context.home_id)
                continue

            # ---- 正常对话 ----
            state_input, config = build_agent_request(
                HumanMessage(content=user_input), context
            )
            # 进度事件会在这里边跑边打印，所以 "小智:" 标签要等它跑完再打，
            # 否则规划过程会插在标签和回复中间。
            result = _invoke_with_approval(graph, state_input, config, view)

            # 提取最终回复
            final_msg = result["messages"][-1]
            response = final_msg.content

            console.print("\n[bold cyan]🤖 小智:[/bold cyan] ", end="")
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
    home_id: str = typer.Option("demo-home", help="住宅 ID"),
    user_id: str = typer.Option("demo-user", help="用户 ID"),
    session_id: Optional[str] = typer.Option(None, help="已有会话 ID；留空则新建"),
    client_id: str = typer.Option("cli", help="终端 ID"),
    room_id: Optional[str] = typer.Option(None, help="当前房间 ID"),
    device_id: Optional[str] = typer.Option(None, help="当前设备 ID"),
    admin: bool = typer.Option(
        False,
        "--admin",
        help="以家庭管理员身份运行（仅供可信后端或本地演示设置）",
    ),
    trace: bool = typer.Option(
        False,
        "--trace",
        help="额外显示路由、记忆判断等诊断事件（规划过程默认已显示）",
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

    # ---- 初始化设备注册中心 ----
    backend = SimulatorBackend()
    registry = DeviceRegistry(backend)

    # ---- 注入注册中心到工具层 ----
    set_tools_registry(registry)

    automation_runtime = None
    if settings.automation.enabled:
        automation_runtime = AutomationRuntime(
            registry,
            db_path=settings.automation.db_path,
            timezone_name=settings.automation.timezone,
            event_sink=lambda event: logger.info("自动化事件 | {}", event),
        )
        automation_runtime.scheduler.poll_seconds = settings.automation.poll_seconds
        automation_runtime.scheduler.start()
        set_automation_runtime(automation_runtime)

    space_directory = SpaceDirectory.from_registry(registry, home_id=home_id)

    # ---- 构建 Agent 图 ----
    console.print("[dim]正在初始化 Agent...[/dim]")
    graph = None
    try:
        external_tools = load_external_tools(settings.external_mcp_servers)
        graph = build_graph(registry, settings, space_directory, external_tools=external_tools)
        sessions = SessionManager(
            space_directory,
            graph.checkpointer,
            ttl=timedelta(hours=settings.memory.session_ttl_hours),
        )
        expired_sessions = sessions.cleanup_expired()
        if expired_sessions:
            logger.info(f"已清理 {expired_sessions} 个过期会话")
        context = sessions.create(
            home_id=home_id,
            user_id=user_id,
            client_id=client_id,
            room_id=room_id,
            device_id=device_id,
            is_admin=admin,
            session_id=session_id,
        )
    except Exception as e:
        _release_runtime_resources(automation_runtime, graph)
        console.print(f"\n[red]❌ Agent 初始化失败: {e}[/red]\n")
        console.print(
            "[dim]可能的原因:\n"
            "  1. API Key 无效或过期\n"
            "  2. 网络无法访问百炼 API\n"
            "  3. 模型名称不正确[/dim]\n"
        )
        raise typer.Exit(code=1)

    _print_banner(settings, context)

    # ---- 打印提示 ----
    console.print()
    console.print(
        "[dim]试着说:[/dim] "
        "[bold]打开客厅灯[/bold] / "
        "[bold]空调调到25度[/bold] / "
        "[bold]我要睡觉了[/bold] / "
        "[bold]现在家里什么状态?[/bold] / "
        "[bold]杭州今天天气怎么样?[/bold]"
    )
    console.print(
        "[dim]多动作请求（如「关掉客厅灯，然后把卧室空调调到25度」）会走 "
        "Planner → Executor → Verifier，过程会逐步显示[/dim]"
    )
    console.print("[dim]输入 /help 查看更多用法，/quit 退出[/dim]")

    # ---- 启动对话循环 ----
    try:
        run_interactive_loop(
            graph, registry, context, sessions,
            PlanProgressView(console, show_trace=trace),
            automation_runtime,
        )
    finally:
        _release_runtime_resources(automation_runtime, graph)


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
