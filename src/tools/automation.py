"""Agent tools for creating and managing confirmed automation routines."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from ..automation.planning import (
    ScheduledActionInput,
    ScheduledRoutineInput,
    VehicleRoutineInput,
)
if TYPE_CHECKING:
    from ..automation.runtime import AutomationRuntime


_runtime = None

UTC = timezone.utc


def set_automation_runtime(runtime) -> None:
    global _runtime
    _runtime = runtime


def _r():
    if _runtime is None:
        raise RuntimeError("AutomationRuntime 尚未初始化")
    return _runtime


def _identity(config: RunnableConfig | None) -> tuple[str, str]:
    values = (config or {}).get("configurable", {})
    home_id = values.get("home_id")
    user_id = values.get("user_id")
    if not home_id or not user_id:
        raise RuntimeError("缺少可信 home_id/user_id")
    return str(home_id), str(user_id)


@tool(args_schema=ScheduledRoutineInput)
def create_scheduled_routine(
    name: str,
    anchor_at_iso: str,
    actions: list[ScheduledActionInput],
    config: RunnableConfig = None,
) -> str:
    """创建通用的一次性定时例程。

    anchor_at_iso 是用户目标发生时间。每个 action 用 offset_minutes 表示相对
    时间：提前执行必须为负数。根据用户目标动态选择任意受支持设备工具，不能
    添加用户没有要求的动作，也不能定时解锁门锁。设备动作的 arguments 必须
    使用设备中文名 device_name；action 使用 on/off/set_temp 等现有控制工具值，
    不要使用 device_id 或 turn_on/turn_off。
    """
    home_id, user_id = _identity(config)
    anchor_at = datetime.fromisoformat(anchor_at_iso)
    parsed = [
        item if isinstance(item, ScheduledActionInput) else ScheduledActionInput.model_validate(item)
        for item in actions
    ]
    routine, tasks = _r().create_scheduled_routine(
        home_id, user_id, name, anchor_at, parsed
    )
    return f"已创建定时例程 {routine.id}，共安排 {len(tasks)} 个任务。"


@tool(args_schema=VehicleRoutineInput)
def create_vehicle_arrival_routine(
    vehicle_id: str,
    name: str,
    actions: list[ScheduledActionInput],
    config: RunnableConfig = None,
) -> str:
    """创建通用车辆 ETA 例程，动作时间相对于预计到家时间。"""
    home_id, user_id = _identity(config)
    parsed = [
        item if isinstance(item, ScheduledActionInput) else ScheduledActionInput.model_validate(item)
        for item in actions
    ]
    routine = _r().create_vehicle_arrival_routine(
        home_id, user_id, vehicle_id, name, parsed
    )
    return f"已创建车辆回家例程，routine_id={routine.id}。"


@tool
def schedule_wake_routine(wake_at_iso: str, config: RunnableConfig = None) -> str:
    """创建一次起床自动化。

    wake_at_iso 必须是带日期的 ISO 8601 时间，例如 2026-08-16T06:00:00+08:00。
    例程会设置卧室音响闹钟，并在起床前准备洗澡热水和冲奶热水。
    """
    home_id, user_id = _identity(config)
    wake_at = datetime.fromisoformat(wake_at_iso)
    routine, tasks = _r().schedule_wake(home_id, user_id, wake_at)
    return f"已创建起床例程 {routine.id}，共安排 {len(tasks)} 个任务。"


@tool
def enable_vehicle_arrival_routine(
    vehicle_id: str,
    config: RunnableConfig = None,
) -> str:
    """启用车辆回家联动；实际执行由车辆 ETA 或地理围栏事件触发。"""
    home_id, user_id = _identity(config)
    routine = _r().enable_vehicle_arrival(home_id, user_id, vehicle_id)
    return f"已启用车辆 {vehicle_id} 的回家例程，routine_id={routine.id}。"


_TASK_STATUS_TEXT = {
    "pending": "待执行",
    "running": "执行中",
    "completed": "已完成",
    "failed": "执行失败",
    "cancelled": "已取消",
}


def _local_time(value: datetime | None, zone: ZoneInfo) -> str | None:
    """Render a stored UTC timestamp in the routine's own timezone."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(zone).strftime("%Y-%m-%d %H:%M:%S")


def _offset_text(minutes: int) -> str:
    if minutes == 0:
        return "准点执行"
    if minutes < 0:
        return f"提前{abs(minutes)}分钟"
    return f"延后{minutes}分钟"


@tool
def list_automation_routines(config: RunnableConfig = None) -> str:
    """列出当前住宅的自动化例程，含每个动作的设备、参数、时间和执行状态。

    用户询问定时任务的具体内容时使用本工具，返回结果已包含全部动作明细，
    可直接向用户逐条说明，不要声称无法查看动作详情。
    """
    home_id, _ = _identity(config)
    store = _r().store
    routines = store.list_routines(home_id)
    payload = []
    for item in routines:
        try:
            zone = ZoneInfo(item.timezone)
        except Exception:
            zone = UTC
        runs = store.list_runs(item.id)
        latest_run = runs[-1] if runs else None
        # 同一动作在多次排期（如车辆 ETA 更新）下会有多条任务，取最新那条的状态。
        latest_task = {}
        for task in store.list_tasks(item.id):
            current = latest_task.get(task.action_id)
            if current is None or task.created_at >= current.created_at:
                latest_task[task.action_id] = task

        actions = []
        for step, action in enumerate(item.actions, start=1):
            task = latest_task.get(action.id)
            entry = {
                "step": step,
                "description": action.description or action.tool_name,
                "tool_name": action.tool_name,
                "arguments": action.arguments,
                "enabled": action.enabled,
            }
            if action.tool_name == "set_alarm":
                # 闹钟在例程创建时就已设置，due_at 是设置时刻而不是响铃时刻。
                entry["timing"] = "创建时立即设置闹钟，响铃时间为目标时间"
            else:
                entry["timing"] = _offset_text(action.offset_minutes)
                entry["offset_minutes"] = action.offset_minutes
            if task is None:
                entry["status"] = "未排期"
            else:
                entry["status"] = _TASK_STATUS_TEXT.get(task.status, task.status)
                entry["scheduled_at"] = _local_time(task.due_at, zone)
                if task.executed_at:
                    entry["executed_at"] = _local_time(task.executed_at, zone)
                if task.error:
                    entry["error"] = task.error
            actions.append(entry)

        payload.append({
            "routine_id": item.id,
            "name": item.name,
            "trigger_type": item.trigger_type,
            "enabled": item.enabled,
            "timezone": item.timezone,
            "target_time": _local_time(latest_run.anchor_at, zone) if latest_run else None,
            "action_count": len(item.actions),
            "actions": actions,
        })
    return json.dumps(payload, ensure_ascii=False)


@tool
def cancel_automation_routine(routine_id: str, config: RunnableConfig = None) -> str:
    """取消某个例程尚未执行的全部任务。"""
    home_id, user_id = _identity(config)
    count = _r().cancel(routine_id, home_id, user_id)
    return f"已取消 {count} 个尚未执行的自动化任务。"
