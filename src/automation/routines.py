"""Factories for the two supported automation scenarios."""

from __future__ import annotations

from .models import Routine, RoutineAction
from .planning import ScheduledActionInput


def build_scheduled_routine(
    home_id: str,
    user_id: str,
    name: str,
    actions: list[ScheduledActionInput],
    *,
    metadata: dict | None = None,
) -> Routine:
    return Routine(
        home_id=home_id,
        user_id=user_id,
        name=name,
        trigger_type="fixed_time",
        actions=[
            RoutineAction(
                offset_minutes=action.offset_minutes,
                tool_name=action.tool_name,
                arguments=action.arguments,
                description=action.description,
            )
            for action in actions
        ],
        metadata=metadata or {"kind": "scheduled", "requires_initial_confirmation": True},
    )


def build_vehicle_routine(
    home_id: str,
    user_id: str,
    vehicle_id: str,
    name: str,
    actions: list[ScheduledActionInput],
) -> Routine:
    routine = build_scheduled_routine(home_id, user_id, name, actions)
    routine.trigger_type = "vehicle_eta"
    routine.metadata.update({
        "kind": "vehicle_arrival",
        "vehicle_id": vehicle_id,
        "never_unlock_door": True,
    })
    return routine


def build_wake_routine(home_id: str, user_id: str) -> Routine:
    return Routine(
        home_id=home_id,
        user_id=user_id,
        name="起床准备",
        trigger_type="fixed_time",
        actions=[
            RoutineAction(
                offset_minutes=0,
                tool_name="set_alarm",
                description="设置卧室音响起床闹钟",
                arguments={"speaker_name": "卧室音响", "label": "起床"},
            ),
            RoutineAction(
                offset_minutes=-30,
                tool_name="control_water_heater",
                description="提前准备洗澡热水",
                arguments={
                    "device_name": "卫生间电热水器", "action": "set_temp", "target_temp": 45,
                },
            ),
            RoutineAction(
                offset_minutes=-10,
                tool_name="control_kettle",
                description="准备冲牛奶所需热水",
                arguments={
                    "device_name": "厨房烧水壶", "action": "set_temp", "target_temp": 80,
                },
            ),
            RoutineAction(
                offset_minutes=0,
                tool_name="control_curtain",
                description="打开卧室窗帘",
                arguments={"device_name": "卧室窗帘", "action": "open"},
            ),
            RoutineAction(
                offset_minutes=0,
                tool_name="control_light",
                description="打开卧室灯",
                arguments={"device_name": "卧室灯", "action": "set_brightness", "brightness": 40},
            ),
        ],
        metadata={"kind": "wake", "requires_initial_confirmation": True},
    )


def build_arrival_routine(home_id: str, user_id: str) -> Routine:
    return Routine(
        home_id=home_id,
        user_id=user_id,
        name="车辆回家准备",
        trigger_type="vehicle_eta",
        actions=[
            RoutineAction(
                offset_minutes=-15,
                tool_name="control_water_heater",
                description="提前准备洗澡热水",
                arguments={
                    "device_name": "卫生间电热水器", "action": "set_temp", "target_temp": 45,
                },
            ),
            RoutineAction(
                offset_minutes=-10,
                tool_name="control_ac",
                description="提前降低客厅温度",
                arguments={
                    "device_name": "客厅空调", "action": "on", "temperature": 25, "mode": "cool",
                },
            ),
            RoutineAction(
                offset_minutes=-2,
                tool_name="control_curtain",
                description="打开客厅窗帘透光",
                arguments={"device_name": "客厅窗帘", "action": "open"},
            ),
            RoutineAction(
                offset_minutes=0,
                tool_name="control_light",
                description="到家时打开客厅灯",
                arguments={"device_name": "客厅灯", "action": "on"},
            ),
        ],
        metadata={
            "kind": "vehicle_arrival",
            "requires_initial_confirmation": True,
            "never_unlock_door": True,
        },
    )
