"""
场景模式工具
============
一键设置多个设备到预设状态，模拟真实智能家居的"场景"功能。

场景设计原则:
  - 每个场景控制 3-6 个设备，提供完整的场景体验
  - 场景之间互不冲突（激活新场景会覆盖旧场景的设置）
  - 只操作执行器，绝不碰只读传感器。批量关闭按 `devices/capabilities.py`
    的 scene_exit 分组派生（P0）：新增设备类型时，只要声明了 scene_exit，
    "离家/睡眠"模式就会自动处理它，不用再手改这里的类型清单——
    "离家模式"也因此永远不会把温湿度计关了。

当前场景:
  - 回家模式: 开客厅灯 + 客厅空调 + 开窗帘
  - 离家模式: 关闭所有电器（安全节能）
  - 睡眠模式: 关灯关电视 + 卧室空调低风速 + 关窗帘
  - 观影模式: 氛围灯光 + 开电视 + 关窗帘
  - 起床模式: 开卧室窗帘 + 关空调 + 渐亮灯光

P1 改造: `build_scene_tools(registry)` 工厂以闭包持有 registry，
不再经过模块级单例。
"""

from collections.abc import Callable

from langchain_core.tools import StructuredTool
from loguru import logger

from ..devices.base import DeviceRegistry
from ..devices.capabilities import SCENE_EXIT_TYPES
from ..models import DeviceType

# 离家/睡眠批量操作按 scene_exit 分组的类型集合（从能力声明派生，禁止手写）。
_POWER_OFF_ON_EXIT = SCENE_EXIT_TYPES["power_off"]
_CURTAINS_ON_EXIT = SCENE_EXIT_TYPES["curtain_close"]
_LOCKS_ON_EXIT = SCENE_EXIT_TYPES["lock"]

# 睡眠模式只关灯/电视/加湿器和窗帘：空调单独设置，热水器/烧水壶睡眠时不强制关。
_SLEEP_POWER_OFF = frozenset({DeviceType.LIGHT, DeviceType.TV, DeviceType.HUMIDIFIER})


# ============================================================
# 场景定义（可扩展）
# ============================================================

# 所有支持的场景及其描述
SCENE_META = {
    "回家模式": {
        "description": "打开客厅灯光和空调，打开窗帘，营造温馨的回家氛围",
        "emoji": "🏠",
    },
    "离家模式": {
        "description": "关闭所有灯光、空调、电视、加湿器、热水器和烧水壶，关闭所有窗帘并锁好门锁，确保安全节能",
        "emoji": "👋",
    },
    "睡眠模式": {
        "description": "关闭所有灯光和电视，窗帘全关，卧室空调设为 26°C 低风速制冷",
        "emoji": "🌙",
    },
    "观影模式": {
        "description": "灯光调暗，关窗帘，开电视，打造家庭影院体验",
        "emoji": "🎬",
    },
    "起床模式": {
        "description": "打开卧室窗帘，关闭卧室空调，渐亮卧室灯光，温柔唤醒",
        "emoji": "🌅",
    },
}


def _activate_scene(registry: DeviceRegistry) -> Callable[[str], str]:

    def activate_scene(scene_name: str) -> str:
        """
        激活智能场景模式。一键设置多个设备到预设状态。

        可用的场景模式:
          🏠 回家模式 — 打开客厅灯和空调，打开窗帘
          👋 离家模式 — 关闭所有电器，关窗帘（安全节能）
          🌙 睡眠模式 — 关灯关电视，卧室空调低风速，关窗帘
          🎬 观影模式 — 氛围灯光，开电视，关窗帘
          🌅 起床模式 — 开窗帘，关空调，开卧室灯

        何时使用:
          - 用户说"我回来了"、"到家了" → 推荐回家模式
          - 用户说"我要出门了"、"走了" → 推荐离家模式
          - 用户说"我要睡了"、"困了" → 推荐睡眠模式
          - 用户说"我要看电影"、"看剧" → 推荐观影模式
          - 用户说"早上好"、"起床了" → 推荐起床模式

        参数:
            scene_name: 场景名称，如"回家模式"、"睡眠模式"等

        返回:
            场景执行结果的文本描述。
        """
        logger.info(f"激活场景: {scene_name}")

        results: list[str] = []

        # ========================================
        # 🏠 回家模式
        # ========================================
        if scene_name == "回家模式":
            registry.update("living_room_light", power=True, brightness=80, color="暖白")
            registry.update("living_room_ac", power=True, temperature=26, mode="cool")
            registry.update("living_room_curtain", position=100)
            results = [
                "✅ 已激活「🏠 回家模式」",
                "  · 客厅灯已打开（亮度 80%，暖白）",
                "  · 客厅空调已开启（制冷 26°C）",
                "  · 客厅窗帘已完全打开",
                "🏠 欢迎回家！",
            ]

        # ========================================
        # 👋 离家模式
        # ========================================
        elif scene_name == "离家模式":
            all_devices = registry.get_all()
            for dev_id, dev in all_devices.items():
                if dev.device_type in _POWER_OFF_ON_EXIT:
                    registry.update(dev_id, power=False)
                elif dev.device_type in _CURTAINS_ON_EXIT:
                    registry.update(dev_id, position=0)
            # 门锁单独处理：离家要"上锁"而不是"关闭"（power 表示在线，必须保持）。
            for dev_id, dev in all_devices.items():
                if dev.device_type in _LOCKS_ON_EXIT:
                    registry.update(dev_id, locked=True)
            results = [
                "✅ 已激活「👋 离家模式」",
                "  · 所有灯光、空调、电视、加湿器、热水器和烧水壶已关闭",
                "  · 所有窗帘已关闭",
                "  · 所有门锁已上锁",
                "👋 再见，所有设备已安全关闭！",
            ]

        # ========================================
        # 🌙 睡眠模式
        # ========================================
        elif scene_name == "睡眠模式":
            all_devices = registry.get_all()
            for dev_id, dev in all_devices.items():
                if dev.device_type in _SLEEP_POWER_OFF:
                    registry.update(dev_id, power=False)
                elif dev.device_type in _CURTAINS_ON_EXIT:
                    registry.update(dev_id, position=0)
            # 卧室空调特殊设置
            registry.update("bedroom_ac", power=True, temperature=26, mode="cool", fan_speed="low")
            registry.update("living_room_ac", power=False)
            results = [
                "✅ 已激活「🌙 睡眠模式」",
                "  · 所有灯光、电视和加湿器已关闭",
                "  · 所有窗帘已关闭",
                "  · 卧室空调已设为 26°C 低风速制冷",
                "  · 客厅空调已关闭",
                "🌙 晚安，好梦！",
            ]

        # ========================================
        # 🎬 观影模式
        # ========================================
        elif scene_name == "观影模式":
            registry.update("living_room_light", power=True, brightness=10, color="暖黄")
            registry.update("bedroom_light", power=False)
            registry.update("kitchen_light", power=False)
            registry.update("living_room_tv", power=True)
            registry.update("living_room_curtain", position=0)
            registry.update("bedroom_curtain", position=0)
            registry.update("living_room_ac", power=True, temperature=25, mode="cool", fan_speed="low")
            results = [
                "✅ 已激活「🎬 观影模式」",
                "  · 客厅灯已调暗至 10%（氛围光）",
                "  · 其他灯光已关闭",
                "  · 客厅电视已打开",
                "  · 所有窗帘已关闭",
                "  · 客厅空调已设为 25°C 低风速",
                "🎬 享受你的观影时光！",
            ]

        # ========================================
        # 🌅 起床模式
        # ========================================
        elif scene_name == "起床模式":
            registry.update("bedroom_curtain", position=100)
            registry.update("bedroom_ac", power=False)
            registry.update("bedroom_light", power=True, brightness=50, color="暖白")
            results = [
                "✅ 已激活「🌅 起床模式」",
                "  · 卧室窗帘已完全打开，阳光洒进来",
                "  · 卧室空调已关闭",
                "  · 卧室灯已打开（亮度 50%，暖白）",
                "🌅 早上好，新的一天开始了！",
            ]

        else:
            available = "、".join(SCENE_META.keys())
            return f"❌ 未知场景「{scene_name}」。当前支持: {available}"

        return "\n".join(results)

    return activate_scene


def _list_scenes() -> Callable[[str], str]:

    def list_scenes(query: str = "") -> str:
        """
        列出所有可用的场景模式及其功能描述。

        当用户不确定有哪些场景，或想了解场景功能时调用。

        参数:
            query: 保留参数，当前未使用

        返回:
            场景列表和对应的描述。
        """
        lines = ["📋 **可用的场景模式:**", ""]
        for name, meta in SCENE_META.items():
            lines.append(f"  {meta['emoji']} **{name}**: {meta['description']}")
        lines.append("")
        lines.append('说出场景名称即可激活，如"我要开启回家模式"。')
        return "\n".join(lines)

    return list_scenes


def build_scene_tools(registry: DeviceRegistry) -> list[StructuredTool]:
    """构建场景工具集（activate_scene + list_scenes）。"""
    activate_scene = _activate_scene(registry)
    list_scenes = _list_scenes()
    return [
        StructuredTool.from_function(
            activate_scene,
            name="activate_scene",
            description=activate_scene.__doc__,
        ),
        StructuredTool.from_function(
            list_scenes,
            name="list_scenes",
            description=list_scenes.__doc__,
        ),
    ]


__all__ = ["SCENE_META", "build_scene_tools"]
