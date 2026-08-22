"""确定性启发式判定（P3：散落正则/关键词收敛后的唯一归属）。

这里集中了三类"不靠 LLM、必须确定"的判定：

  1. `should_use_planner`      —— 是否把请求交给 Planner（原 planning.py）
  2. `classify_intent_fallback`—— 意图路由的确定性兜底（原 routing.py 的关键词表）
  3. `required_automation_tool`—— 自动化创建请求的强制工具锁定（原 graph.py）

为什么要收敛成一个模块：
  这三处以前各自维护一份关键词表（"打开""关闭"至少出现三次），语义上重叠、
  改一处容易漏另一处；而且只有 automation 那份写清楚了"为什么这样判"。
  现在：
  - 真正相同的词表只定义一次（ACTION_CORE 等基表），各判定在其上做有语义的扩展；
  - 每个扩展都注明"为什么与别处不同"；
  - 三个函数同文件，新增关键词时能同时看到所有受影响方。

行为兼容性：每个函数的判定结果与迁移前逐字一致，由 tests/test_heuristics.py
和既有用例（test_phase_seven / test_automation_routines 等）共同钉住。
"""

from __future__ import annotations

import re

# ============================================================
# 共享词表
# ============================================================

# 三个判定共有的核心动作词。各判定在它之上做**有语义的扩展**，扩展不得互相污染：
#  - 路由兜底要的是"意图线索"，越宽越好 → 加 设置/调高/调低；
#  - Planner 判定要的是"显式的自定义动作"，加的词都是用户对设备的直接命令；
#  - 自动化强制要的是"定时场景下的设备动作"，加的"准备/预热/烧水"等是
#    定时语义下的动作表达。
ACTION_CORE = ("打开", "开启", "关闭", "关掉", "调到")

# ============================================================
# 1. Planner 判定（原 planning.should_use_planner）
# ============================================================

# 预定义场景请求留在 ReAct + 场景审批路径。注意和 routing 的 scene_words 不同：
# 这里匹配的是**整句惯用表达**（"我要出门"），routing 匹配的是**子串线索**
# （"离家"），宽窄不同是有意的——Planner 的排除必须精确，误伤一句"我要出门
# 前关灯"就会把两个动作的请求送去走场景分支。
PLANNER_SCENE_MARKERS = (
    "回家模式", "离家模式", "睡眠模式", "观影模式", "起床模式",
    "我回来了", "我要出门", "我要睡", "看电影", "起床了",
)

PLANNER_ACTION_PATTERNS = ACTION_CORE + ("设为", "调成", "拉开", "拉上", "静音", "切换")

PLANNER_DEVICE_KINDS = ("灯", "空调", "电视", "窗帘", "加湿器", "热水器", "门锁", "烧水壶")

PLANNER_CONNECTORS = ("并且", "然后", "同时", "再把", "再将", "以及", "顺便")


def should_use_planner(text: str) -> bool:
    """Conservatively route explicit custom multi-action requests to Planner."""
    normalized = text.strip()
    if not normalized:
        return False

    # Predefined scene requests remain on the existing ReAct + scene approval path.
    if any(marker in normalized for marker in PLANNER_SCENE_MARKERS):
        return False

    action_count = sum(len(re.findall(pattern, normalized)) for pattern in PLANNER_ACTION_PATTERNS)
    device_kinds = sum(
        1 for keyword in PLANNER_DEVICE_KINDS if keyword in normalized
    )
    connectors = any(connector in normalized for connector in PLANNER_CONNECTORS)
    return action_count >= 2 and (device_kinds >= 2 or connectors)


# ============================================================
# 2. 意图路由的确定性兜底（原 routing.classify_intent_fallback）
# ============================================================

ROUTING_MEMORY_WORDS = ("记住", "忘记", "删除记忆", "偏好", "喜欢", "家庭规则", "记忆")
ROUTING_AUTOMATION_WORDS = (
    "定时", "闹钟", "车辆回家", "汽车回家", "到家前", "回家前",
    "取消例程", "自动化", "提前准备", "提前打开",
)
ROUTING_KNOWLEDGE_WORDS = ("故障", "错误代码", "说明书", "怎么清洗", "怎么维护", "支持什么", "是什么意思")
ROUTING_SCENE_WORDS = ("场景", "模式", "睡眠", "离家", "回家", "观影", "起床")
ROUTING_QUERY_WORDS = ("查询", "状态", "温度", "开着吗", "在线", "有哪些设备")
# 比 Planner 的 action 表宽：路由只需要"这是控制类意图"的线索，不需要精确动作。
ROUTING_CONTROL_WORDS = ACTION_CORE + ("设置", "调高", "调低")

# 未来时间信号是 automation_management 的确定性判据（见 routing.py 的硬约束注释）。
ROUTING_FUTURE_TIME_PATTERN = re.compile(
    r"(?:今天|明天|后天|周[一二三四五六日天]|星期[一二三四五六日天]).{0,10}"
    r"(?:上午|下午|晚上|早上|凌晨|\d{1,2}\s*[点时])",
)


def routing_word_hits(text: str, words: tuple[str, ...]) -> bool:
    """routing 的关键词命中判断（对英文不敏感，text 已 lower）。"""
    return any(word in text for word in words)


def has_future_time(text: str) -> bool:
    return bool(ROUTING_FUTURE_TIME_PATTERN.search(text))


# ============================================================
# 3. 自动化创建请求的强制工具锁定（原 graph._required_automation_tool）
# ============================================================

AUTOMATION_READ_MARKERS = (
    "有哪些", "有多少", "多少个", "多少条", "几个", "几条", "查看", "看看",
    "看一下", "列出", "列表", "查询", "有没有", "是否有", "都有什么",
    "什么任务", "哪些任务", "什么例程",
)
AUTOMATION_CANCEL_MARKERS = ("取消", "删除", "停用", "撤销", "清空", "不要了")
AUTOMATION_TRIGGER_MARKERS = (
    "提前", "定时", "闹钟", "起床", "叫我", "到家前", "回家前",
    "车辆", "汽车", "车快到", "eta", "地理围栏",
)
# 定时语义下的动作词：在核心动作词之外加上"准备/预热/烧水"这类
# 只有在"未来某时刻"语境里才成立的表达。注意 eta 是小写关键词，
# 所以这个判定和 routing 一样先 lower 再匹配。
AUTOMATION_ACTION_MARKERS = ACTION_CORE + (
    "调高", "调低", "设置", "设为", "准备", "预热", "烧水",
    "降温", "升温", "制冷", "制热", "启动", "叫我",
)
AUTOMATION_TIME_PATTERN = re.compile(
    r"\d{1,2}\s*[:：]\d{2}"
    r"|\d{1,2}\s*[点時时]"
    r"|\d{1,3}\s*(?:分钟|个小时|小时)后"
    r"|今天|今晚|明天|明早|明晚|后天"
    r"|周[一二三四五六日天]|星期[一二三四五六日天]"
)


def required_automation_tool(text: str) -> str | None:
    """Return the mutation tool a creation request must call, if any.

    强制机制的默认值必须是"不强制"：只有同时出现未来触发信号和设备动作信号
    时才锁定创建工具。查询和取消类请求（例如"当前有多少个定时任务"）必须返回
    None，否则 Agent 会被反复要求为一个问句创建例程，两轮都失败后只能对用户
    报错，而它本来应该调用 list_automation_routines。
    """
    normalized = text.strip().lower()
    if not normalized:
        return None
    if any(marker in normalized for marker in AUTOMATION_READ_MARKERS):
        return None
    if any(marker in normalized for marker in AUTOMATION_CANCEL_MARKERS):
        return None
    has_trigger = bool(AUTOMATION_TIME_PATTERN.search(normalized)) or any(
        marker in normalized for marker in AUTOMATION_TRIGGER_MARKERS
    )
    has_action = any(marker in normalized for marker in AUTOMATION_ACTION_MARKERS)
    if not (has_trigger and has_action):
        return None
    if any(marker in normalized for marker in ("车辆", "汽车", "车快到", "eta", "地理围栏")):
        return "create_vehicle_arrival_routine"
    if any(marker in normalized for marker in ("起床", "闹钟", "叫我起床")):
        return "schedule_wake_routine"
    return "create_scheduled_routine"
