"""工具模块"""

from .devices import (
    control_light,
    control_ac,
    control_tv,
    control_curtain,
    get_device_status,
    set_registry as set_device_tools_registry,
)
from .scenes import (
    activate_scene,
    list_scenes,
    set_registry as set_scene_tools_registry,
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
        activate_scene,
        list_scenes,
        get_device_status,
    ]


__all__ = [
    "get_all_tools",
    "set_registry",
    "control_light",
    "control_ac",
    "control_tv",
    "control_curtain",
    "activate_scene",
    "list_scenes",
    "get_device_status",
]
