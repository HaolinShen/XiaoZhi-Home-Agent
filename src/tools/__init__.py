"""工具模块"""

from .devices import (
    control_light,
    control_ac,
    control_tv,
    control_curtain,
    control_humidifier,
    read_sensor,
    get_device_status,
    set_registry as set_device_tools_registry,
)
from .scenes import (
    activate_scene,
    list_scenes,
    set_registry as set_scene_tools_registry,
)
from .memory import (
    delete_personal_memory,
    list_preference_candidates,
    confirm_preference_candidate,
    reject_preference_candidate,
    list_memory_versions,
    list_personal_memories,
    save_home_rule,
    save_personal_memory,
    set_memory_service,
    update_personal_memory,
)


def set_registry(registry) -> None:
    """同时注入注册中心到所有工具模块"""
    set_device_tools_registry(registry)
    set_scene_tools_registry(registry)


def get_all_tools() -> list:
    """获取所有已注册的工具（用于 bind_tools）"""
    return [
        control_light,
        control_ac,
        control_tv,
        control_curtain,
        control_humidifier,
        activate_scene,
        list_scenes,
        read_sensor,
        get_device_status,
        save_personal_memory,
        save_home_rule,
        list_personal_memories,
        update_personal_memory,
        delete_personal_memory,
        list_preference_candidates,
        confirm_preference_candidate,
        reject_preference_candidate,
        list_memory_versions,
    ]


__all__ = [
    "get_all_tools",
    "set_registry",
    "control_light",
    "control_ac",
    "control_tv",
    "control_curtain",
    "control_humidifier",
    "activate_scene",
    "list_scenes",
    "read_sensor",
    "get_device_status",
    "set_memory_service",
    "save_personal_memory",
    "save_home_rule",
    "list_personal_memories",
    "update_personal_memory",
    "delete_personal_memory",
    "list_preference_candidates",
    "confirm_preference_candidate",
    "reject_preference_candidate",
    "list_memory_versions",
]
