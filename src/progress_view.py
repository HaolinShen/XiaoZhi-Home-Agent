"""
运行时进度视图
==============
把图发出的 `custom` 进度事件渲染成终端可读的分阶段过程。

为什么需要它:
  图内部早就把 Planner / Executor / Verifier 拆成了三个独立节点，但 CLI 之前用
  `graph.invoke()` 一次性拿结果 —— LangGraph 在 invoke 模式下给节点一个空的
  stream writer，所有 `emit_progress` 都被静默丢弃。于是用户只能看到
  "已生成 N 步执行计划" 和最后一句 "任务已完成"，中间谁在规划、谁在执行、
  谁在验证完全看不出来。改用 `graph.stream(stream_mode=["custom","updates"])`
  之后，这个模块负责把事件翻译成人能看懂的过程。

分层原则:
  - 事件由 `src/agent/observability.py` 定义并发出（不依赖任何终端库）
  - 渲染只在这里做（依赖 rich，属于表现层）
  - 图和工具层不知道有终端存在
"""

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent.observability import PLANNING_EVENTS, TRACE_EVENTS

# 单个参数值在表格里的展示上限，避免长文本把表格撑破。
_MAX_VALUE_CHARS = 40


def format_arguments(arguments: dict[str, Any] | None) -> str:
    """把工具参数压成一行 `k=v` 文本，供表格和单行日志共用。"""
    if not arguments:
        return "—"
    parts = []
    for key, value in arguments.items():
        text = str(value)
        if len(text) > _MAX_VALUE_CHARS:
            text = text[: _MAX_VALUE_CHARS - 1] + "…"
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def format_state(state: dict[str, Any] | None) -> str:
    """把期望/实测状态压成一行，用于 Verifier 的对比展示。"""
    if not state:
        return "—"
    return " ".join(f"{key}={value}" for key, value in state.items())


class PlanProgressView:
    """按事件顺序渲染 Planner → Executor → Verifier 的运行过程。

    参数:
      console: rich Console（测试里可传入写入 StringIO 的 Console）
      show_trace: 是否额外显示路由/记忆/上下文这类诊断事件
    """

    def __init__(self, console: Console, show_trace: bool = False):
        self.console = console
        self.show_trace = show_trace
        self._planning_seen = False

    # ---- 对外入口 ----

    def handle(self, event: dict[str, Any]) -> None:
        """处理一个进度事件。未知事件一律忽略，避免图新增事件时 CLI 崩掉。"""
        name = event.get("event", "")
        if name in PLANNING_EVENTS:
            self._planning_seen = True
            getattr(self, f"_on_{name}")(event)
        elif name in TRACE_EVENTS and self.show_trace:
            self._on_trace(name, event)

    @property
    def planning_seen(self) -> bool:
        """本轮是否走过规划分支（调用方据此决定是否再加分隔线）。"""
        return self._planning_seen

    def reset(self) -> None:
        self._planning_seen = False

    # ---- 规划阶段 ----

    def _on_planning_selected(self, event: dict[str, Any]) -> None:
        self.console.print()
        self.console.print(Panel(
            f"[bold]目标[/bold] {event.get('goal', '')}\n"
            f"[dim]{event.get('reason', '')}[/dim]",
            title="🧭 Planner 分支",
            border_style="magenta",
            padding=(0, 1),
        ))

    def _on_plan_generated(self, event: dict[str, Any]) -> None:
        steps = event.get("steps", [])
        table = Table(
            title=(
                f"📋 计划 v{event.get('revision', 1)}"
                f"（{len(steps)} 步 · 此刻尚未触碰任何设备）"
            ),
            box=box.ROUNDED,
            border_style="magenta",
            title_justify="left",
        )
        table.add_column("步", style="bold", width=3, justify="right")
        table.add_column("要做什么", style="white")
        table.add_column("工具", style="cyan")
        table.add_column("参数", style="dim")
        for step in steps:
            table.add_row(
                str(step.get("step_id", "?")),
                str(step.get("description", "")),
                str(step.get("tool_name", "")),
                format_arguments(step.get("arguments")),
            )
        self.console.print(table)
        rationale = event.get("rationale") or ""
        if rationale:
            self.console.print(f"[dim]规划理由: {rationale}[/dim]")

    def _on_plan_decision(self, event: dict[str, Any]) -> None:
        if event.get("approved"):
            self.console.print(
                f"[green]▶ 计划 v{event.get('revision', 1)} 已批准，"
                f"开始逐步执行[/green]"
            )
        else:
            self.console.print(
                f"[yellow]■ 计划 v{event.get('revision', 1)} 已拒绝，"
                f"没有任何设备被操作[/yellow]"
            )

    # ---- 执行与验证阶段 ----

    def _on_step_started(self, event: dict[str, Any]) -> None:
        attempt = event.get("attempt", 1)
        retry_hint = f" [yellow](第 {attempt} 次尝试)[/yellow]" if attempt > 1 else ""
        self.console.print(
            f"\n[bold blue]⚙ Executor[/bold blue] "
            f"步骤 {event.get('step_index', '?')}/{event.get('step_total', '?')} "
            f"· {event.get('description', '')}{retry_hint}\n"
            f"  [dim]调用[/dim] [cyan]{event.get('tool_name', '')}[/cyan]"
            f"([dim]{format_arguments(event.get('arguments'))}[/dim])"
        )

    def _on_step_executed(self, event: dict[str, Any]) -> None:
        self.console.print(
            f"  [dim]工具返回[/dim] {event.get('tool_result', '')}"
        )

    def _on_step_verified(self, event: dict[str, Any]) -> None:
        expected = format_state(event.get("expected_state"))
        actual = format_state(event.get("actual_state"))
        if event.get("success"):
            self.console.print(
                f"[bold green]✔ Verifier[/bold green] "
                f"步骤 {event.get('step_index', '?')}/{event.get('step_total', '?')} 通过"
                f" · [dim]期望[/dim] {expected} [dim]≡ 实测[/dim] {actual}"
            )
            return
        self.console.print(
            f"[bold red]✘ Verifier[/bold red] "
            f"步骤 {event.get('step_index', '?')}/{event.get('step_total', '?')} 未通过"
            f" · [red]{event.get('problem_type', 'unknown')}[/red]\n"
            f"  [dim]期望[/dim] {expected}\n"
            f"  [dim]实测[/dim] {actual}\n"
            f"  [dim]{event.get('reason', '')}[/dim]"
        )

    def _on_step_retry(self, event: dict[str, Any]) -> None:
        self.console.print(
            f"[yellow]↻ 重试步骤 {event.get('step_id', '?')}"
            f"（第 {event.get('attempt', '?')}/{event.get('max_attempts', '?')} 次尝试）[/yellow]"
        )

    def _on_replan_requested(self, event: dict[str, Any]) -> None:
        if event.get("accepted"):
            self.console.print(
                f"[yellow]⟲ 重试额度已用尽，把失败原因交回 Planner 重新规划"
                f"（第 {event.get('replan_count', '?')}/{event.get('max_replans', '?')} 次）[/yellow]"
            )
        else:
            self.console.print(
                "[red]⊘ 重新规划额度也已用尽，任务停止，不再继续调用工具[/red]"
            )

    def _on_planning_finished(self, event: dict[str, Any]) -> None:
        status = event.get("status", "unknown")
        style = {
            "completed": "green", "cancelled": "yellow", "failed": "red",
        }.get(status, "dim")
        # total 是"尝试次数"而不是"步数"：失败重试和被放弃的旧版本计划都记在里面，
        # 所以这里写成 通过 X 次 / 共 Y 次尝试，避免看成 X/Y 步。
        self.console.print(
            f"[{style}]🏁 规划结束 · {status} · "
            f"验证通过 {event.get('succeeded', 0)} 次 / 共 {event.get('total', 0)} 次尝试 · "
            f"最终计划 v{event.get('plan_revision', 0)}[/{style}]"
        )

    # ---- 诊断事件（--trace）----

    def _on_trace(self, name: str, event: dict[str, Any]) -> None:
        detail = " ".join(
            f"{key}={value}" for key, value in event.items() if key != "event"
        )
        self.console.print(f"[dim]· {name} {detail}[/dim]")
