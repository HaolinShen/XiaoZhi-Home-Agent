"""检索之前，先把「用户说的这台」定位到「哪一份说明书」。

为什么这一步必须在检索之前，而且必须能拒答
------------------------------------------
说明书检索有一个和普通文档问答不同的性质：**查到错型号的说明书比查不到更危险。**
"E4 是排水泵异常，请检查排水管排水是否通畅"读起来完全权威、格式专业、步骤具体，
用户没有任何线索能判断这段话其实属于另一台空调。查不到只是没帮上忙，
查错型号是拿着别人家的答案指挥用户拆机。

所以解析结果是四态，而不是"型号或 None"：

- ``resolved``   定位到唯一一台设备，且它登记了型号 → 可以检索
- ``ambiguous``  条件对上了多台（家里两台空调型号还不一样）→ 反问是哪一台
- ``no_model``   定位到设备了，但这台设备没登记说明书型号 → 说清楚是"没有这份资料"
- ``unknown``    连是哪台设备都没解析出来 → 反问

后三种都**不允许**退化成"不带型号过滤搜整个语料库"。`KnowledgeBase.search` 的
`model` 参数刻意做成关键字必填，就是为了让这种退化只能是调用方写出来的显式选择，
而不是漏传参数的副产品。

型号为什么读设备对象而不是这里另开一张表
----------------------------------------
`BaseDevice.model` 是型号的唯一数据源。这个模块里如果再放一份
``{"living_room_ac": "SmartCool-AC2024"}`` 的映射，换设备、加设备时就有两处要改，
而漏改的那一处不会报错——只会让检索安静地按旧型号查。所以这里只做"措辞 → 设备实例"，
型号一律从解析出的设备身上读。

消解顺序的取舍
--------------
显式设备名 > 类型关键词 > 可信上下文（`active_device_id`）。用户明确说了"卧室空调"，
就不能被 App 当前所在房间覆盖掉；反过来，用户只说"空调"时，可信空间
（`sync_context` 落定的 `active_room_id` / `active_device_id`）才用来消解多候选。
这里用的是同一套可信身份边界：房间与设备来自 `RunnableConfig`，不是模型生成的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..agent.context import location_to_room_id
from ..devices.base import _MATCH_STOPWORDS, DeviceRegistry
from ..devices.capabilities import TYPE_KEYWORDS
from ..models import AnyDevice, DeviceType

ResolutionStatus = Literal["resolved", "ambiguous", "no_model", "unknown"]


@dataclass(frozen=True)
class DeviceResolution:
    """一次实体消解的结果。

    `basis` 记录"靠什么定位到的"（设备名 / 类型关键词 / 可信上下文），
    它会进 RAG 轨迹。排查"为什么答的是客厅那台"时，需要的正是这一句。
    """

    status: ResolutionStatus
    device_id: str | None = None
    device_name: str | None = None
    model: str | None = None
    # ambiguous 时给用户看的候选设备名，已排序（保证同一句话每次反问文案一致）
    candidates: tuple[str, ...] = field(default_factory=tuple)
    basis: str = ""


def resolve_device(
    query: str,
    registry: DeviceRegistry,
    *,
    active_device_id: str | None = None,
    active_room_id: str | None = None,
) -> DeviceResolution:
    """把一句自然语言解析到唯一一台设备，解析不出来就明确说解析不出来。"""
    devices = registry.get_all()
    if not devices:
        return DeviceResolution(status="unknown", basis="住宅内没有已注册设备")

    candidates, basis = _collect_candidates(query, devices, active_device_id)
    if not candidates:
        return DeviceResolution(status="unknown", basis="用户措辞里没有可识别的设备或设备类型")

    if len(candidates) > 1:
        candidates, basis = _narrow(candidates, basis, active_device_id, active_room_id)

    if len(candidates) > 1:
        return DeviceResolution(
            status="ambiguous",
            candidates=tuple(sorted(device.name for device in candidates)),
            basis=basis,
        )

    device = candidates[0]
    if not device.model:
        # 定位到设备了，但它没登记型号。这里**不能**回退成全库检索：
        # 灯、窗帘这些设备本就没挂说明书，全库检索只会把空调的说明书递给用户。
        return DeviceResolution(
            status="no_model",
            device_id=device.device_id,
            device_name=device.name,
            basis=basis,
        )

    return DeviceResolution(
        status="resolved",
        device_id=device.device_id,
        device_name=device.name,
        model=device.model,
        basis=basis,
    )


def _collect_candidates(
    query: str,
    devices: dict[str, AnyDevice],
    active_device_id: str | None,
) -> tuple[list[AnyDevice], str]:
    """按"显式设备名 → 类型关键词 → 可信上下文"依次取候选集。"""
    named = _match_by_name(query, devices)
    if named:
        return named, "用户显式点名了设备"

    typed = _match_by_type_keyword(query, devices)
    if typed:
        return typed, "用户提到了设备类型"

    if active_device_id and active_device_id in devices:
        # 用户一个字没提设备（"这个 E4 什么意思"），但请求是从某台设备的面板发起的。
        return [devices[active_device_id]], "沿用可信上下文里的当前设备"

    return [], ""


def _match_by_name(query: str, devices: dict[str, AnyDevice]) -> list[AnyDevice]:
    """设备名子串匹配，取最长命中。

    先剥掉"的/那/台"这类虚词，"客厅的空调"才能命中设备名"客厅空调"。
    停用词表直接用 `registry.find` 的那一份：用户的措辞习惯只该有一处定义，
    抄一份到这里，以后加"我家的"之类的虚词就会两边不一致。

    只保留最长命中是必要的：假如设备名里同时有"空调"和"客厅空调"，
    "客厅空调坏了"会同时命中两者，取最长才是用户真正指的那台。
    """
    stripped = "".join(char for char in query if char not in _MATCH_STOPWORDS)
    matched = [device for device in devices.values() if device.name and device.name in stripped]
    if not matched:
        return []
    longest = max(len(device.name) for device in matched)
    return [device for device in matched if len(device.name) == longest]


def _match_by_type_keyword(query: str, devices: dict[str, AnyDevice]) -> list[AnyDevice]:
    """类型关键词匹配，多类型命中时按"关键词更长、出现更早"择一。

    为什么需要择一：`TYPE_KEYWORDS` 里"温度"属于温湿度传感器，而"空调"属于空调，
    "空调设置温度多少合适"会同时命中两类。全都当候选就会退化成一句
    "你说的是客厅空调、卧室空调还是客厅温湿度传感器"——问得毫无必要。

    择一规则是两条确定性的语言习惯，不问模型：
    1. **更长的关键词更具体**："温湿度传感器"命中时不该被"温度"拉走。
    2. **出现更早的通常是句子主语**："空调设置温度"里"空调"在句首，问的是空调。

    两条都平手才认定为真歧义（例如"空调和电视都出问题了"），交回给用户。
    """
    ranked: list[tuple[int, int, DeviceType]] = []
    for device_type, keywords in TYPE_KEYWORDS.items():
        positions = [(len(kw), query.find(kw)) for kw in keywords if kw in query]
        if not positions:
            continue
        best_length = max(length for length, _ in positions)
        earliest = min(position for _, position in positions)
        ranked.append((-best_length, earliest, device_type))
    if not ranked:
        return []

    ranked.sort()
    best_key = ranked[0][:2]
    winning_types = {entry[2] for entry in ranked if entry[:2] == best_key}
    return [device for device in devices.values() if device.device_type in winning_types]


def _narrow(
    candidates: list[AnyDevice],
    basis: str,
    active_device_id: str | None,
    active_room_id: str | None,
) -> tuple[list[AnyDevice], str]:
    """多候选时用可信空间消解：先看当前设备，再看当前房间。"""
    for device in candidates:
        if device.device_id == active_device_id:
            return [device], f"{basis}，由可信上下文的当前设备消解"

    if active_room_id:
        in_room = [
            device
            for device in candidates
            if location_to_room_id(device.location or "") == active_room_id
        ]
        # 只在房间过滤后**恰好剩一台**时才采纳。剩两台仍是歧义；
        # 剩零台说明用户问的不是本房间的设备，此时缩窄反而会把正确候选删光。
        if len(in_room) == 1:
            return in_room, f"{basis}，由可信上下文的当前房间消解"

    return candidates, basis
