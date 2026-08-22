"""工具包（P0/P1 改造后的统一入口）。

所有工具由 `build_all_tools()` 工厂按依赖（registry / memory_service /
automation_runtime）显式构建，不再存在模块级可变单例：
  - 工具以闭包持有依赖，调用方无需先"注入全局"再"记得复位"；
  - 后台执行器 / MCP 等无可信身份的调用方在构造期显式关闭偏好观察。

`build_all_tools` 返回的工具列表顺序即图里 bind_tools 的顺序；
新增设备/工具只需改各子工厂内部的能力声明，本文件不再需要登记任何清单。
"""

from .automation import build_automation_tools
from .devices import build_device_tools
from .memory import build_memory_tools
from .scenes import SCENE_META, build_scene_tools


def build_all_tools(
    registry,
    *,
    memory_service=None,
    automation_runtime=None,
    external_tools=None,
    enable_preference_tracking: bool = True,
) -> list:
    """构建 Agent 可见的全部工具（含可选外部 MCP 工具）。

    enable_preference_tracking: 图路径保持 True（偏好观察需要可信身份，
    缺身份会 fail-fast）；后台执行器与 MCP 服务器应显式传 False。
    """
    tools = [
        *build_device_tools(
            registry,
            memory_service,
            enable_preference_tracking=enable_preference_tracking,
        ),
        *build_scene_tools(registry),
        *build_memory_tools(memory_service),
        *build_automation_tools(automation_runtime),
    ]
    if external_tools:
        tools.extend(external_tools)
    return tools


__all__ = [
    "build_all_tools",
    "build_device_tools",
    "build_scene_tools",
    "build_memory_tools",
    "build_automation_tools",
    "SCENE_META",
]
