"""Agentic RAG 子图：实体消解 → 检索 → 自证核对 → 综合作答，任一环节不成立就拒答。

这条链路和"把文档喂给模型让它回答"的区别在于三处刻意的约束：

1. **检索之前先消解实体。** 型号过滤要有型号，型号来自解析出的那台设备
   （`resolution.py`）。解析不出唯一设备就拒答，绝不退化成搜整个语料库——
   查到错型号的说明书比查不到更危险。
2. **检查项分流。** 说明书的排查清单里，"确认设置温度低于室温"这类系统读一下
   设备状态就能核对，"检查滤网是否积尘"必须人到现场。前者由 `selfcheck` 直接
   核对掉（读注册中心真实状态，不问模型），后者原样交给用户。
   文档问答由此变成可执行诊断。
3. **模型只负责组织语言，事实由代码陈述。** 自动核对结论和来源清单都是代码拼的，
   不经模型改写；模型拿到片段后只写处理建议，且被明确要求不要复述这两块。
   这样"引用"是结构保证，不是提示词里的一句请求。

`llm=None` 时整条链路仍然可用（退化成片段拼接 + 词表重写），所以测试是确定性的，
也不会因为没配 API Key 就整个功能不可用。
"""

from __future__ import annotations

import re

from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from loguru import logger
from typing_extensions import TypedDict

from ..devices.base import DeviceRegistry
from ..models import DeviceType
from .base import KnowledgeBase
from .resolution import DeviceResolution, resolve_device
from .selfcheck import CheckContext, run_self_check
from .tokenizer import CODE_PATTERN

# 命中分数下限。013 起分数是 `HybridScores.confidence`——[0,1] 的**绝对**置信度，
# 由 BM25 饱和归一与余弦重标定加权而来（见 retrieval.py 的模块 docstring）。
# 它刻意不是 RRF 融合分：RRF 只有名次信息，第一名恒为满分，拿它守门等于不守门。
#
# 两个数字都由 `python -m src.evaluation.recall --sweep` 在 63 条 golden 用例上扫出来，
# 不是手调的。关键的一格是**首轮下限 0.35**：向量权重取 0.5~1.0 的任何值，
# 拒答准确率都在 0.30→0.35 这一步从 85.7% 跳到 100%，所以它不是拟合某一条用例。
# 代价是 Recall@1 从 89.3%（下限 0.20）降到 87.5%，离线纯 BM25 从 62.5% 降到 58.9%。
# 这个方向的取舍是刻意的：查到错章节的说明书比查不到更危险。
#
# 需要知道的一处薄弱：拒答里最高的那个假阳性是 0.336（"卧室空调有点响" 在
# FrostLine 上命中「制冷效果不佳」），离 0.35 只有 4% 余量。为什么这么薄，
# 实测给了答案——同型号不同症状小节的余弦（困难负例，中位 0.568）和正例
# （中位 0.653）大幅重叠，**绝对置信度本质上分不开"对的症状"和"同一台电器的
# 另一个症状"**。真正能说出"这份说明书没讲这件事"的是词法通道返回 0
# （"响/噪音"在 FrostLine 全文不存在）。所以这道闸门的可靠性来自 BM25，
# 不是来自阈值调得准——这也是不能为了召回率把 BM25 权重清零的真正理由。
DEFAULT_MIN_SCORE = 0.35

# 重写后查询的分数下限，比原句严。理由和 012 时代一致但数字换了量纲：
# 重写后是纯说明书用词，真命中的分数明显更高（实测离线 0.60~0.82、混合 0.52~0.86），
# 而"该型号根本没有这一章"的通用词误命中是 0.000（离线）/ 0.005（混合）。
# 取 0.42：比首轮那档高 20%，又比最弱的真命中（混合下"排水 排水泵 冷凝水"
# 命中「排水不畅」的 0.517）低 19%，两边都留出余量。
DEFAULT_REWRITTEN_MIN_SCORE = 0.42

# 相对截断：置信度不到第一名这个比例的候选不进答案。
#
# 它解决的是 013 自己引入的一个精度回退。语料从 12 个 chunk 扩到 124 个之后，
# 同一份症状手册里的**兄弟小节**会一起被召回——"客厅空调开着但一点都不凉"
# 除了正确的「制冷效果不佳」(0.832)，还带出「有异味」(0.541) 和「噪音异常」(0.512)。
# 它们都过了 0.35 的绝对下限（因为同一台电器的不同症状语义本来就近，
# 实测困难负例余弦中位 0.568），于是答案挂上三条引用而只有一条相关。
# 012 时代同一个查询只返回一条。
#
# 绝对下限管不了这件事：把它提到 0.55 会连正确答案一起砍掉。
# 该用的是相对判断——**比最佳证据弱这么多的东西，不该出现在引用里**。
#
# 0.7 是在 63 条 golden 用例上实测选的：Recall@1 / Recall@3 / MRR / 拒答准确率
# 在 0.0~0.8 的任何取值下**一个数字都不变**（说明它从不砍掉正确答案），
# 而平均引用数从 1.81 降到 1.50。取 0.7 而不是 0.8 是留余量。
#
# 注意它**不是拒答闸门**：第一名永远保留，所以这道截断只会让引用更干净，
# 不会把"答得出来"变成"拒答"。
DEFAULT_RELATIVE_FLOOR = 0.7
# 口语症状 → 说明书用词。用于**没有语义通道**时的确定性重写。
#
# 013 之后这张词表的地位变了：它不再是语义泛化的唯一手段，而是"没配 embedding 时"
# 的确定性兜底。启用远程 embedding 后，"不凉"→「制冷效果不佳」这一跳由向量通道
# 直接完成（实测余弦 0.673），根本走不到重写这一步。词表留着有两个用处：
# 离线可用，以及给 LLM 重写一个可校验的对照答案。
#
# 关于**替换**而不是**追加**：012 给的理由是"分数分母是查询词总数，留着原话会稀释分数"。
# 换成 BM25 后这个理由**不再成立**——BM25 是对查询词求和，匹配不上的词贡献 0，
# 不存在稀释。这里仍然用替换，理由换成了：重写后的查询要和
# `_REWRITTEN_MIN_SCORE` 那一档阈值配套，而那一档是按"纯说明书用词"的分数分布标定的。
# 追加式会让分数落在两档之间，阈值就没有稳定含义了。
#
# "响"用了负向断言排除"影响"，否则"会影响制冷吗"会被判成噪音问题。
_SYMPTOM_LEXICON: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"不凉|不制冷|不够冷|不冷|没有冷风|没冷风|降不下|制冷效果"), "制冷效果不佳 制冷 室温"),
    (re.compile(r"异响|噪音|杂音|咔哒|嗡嗡|响声|声音大|(?<!影)响"), "噪音异常 噪音"),
    (re.compile(r"异味|霉味|臭味|发臭|味道|气味"), "有异味 异味 滤网"),
    (re.compile(r"滤网|清洗|清洁|保养|多久洗"), "滤网 清洗周期 清洁步骤"),
    (re.compile(r"漏水|滴水|排水|积水"), "排水 排水泵 冷凝水"),
)


class RagState(TypedDict, total=False):
    query: str
    rewritten_query: str
    device_id: str | None
    device_name: str | None
    device_model: str | None
    resolution_status: str
    active_room_id: str | None
    active_device_id: str | None
    hits: list[dict]
    check_outcomes: list[dict]
    manual_items: list[str]
    answer: str
    citations: list[str]
    rag_status: str
    refusal_reason: str
    rewrite_count: int
    trajectory: list[dict]


def build_knowledge_rag_subgraph(
    knowledge: KnowledgeBase,
    registry: DeviceRegistry,
    *,
    llm=None,
    top_k: int = 3,
    max_rewrites: int = 1,
    min_score: float = DEFAULT_MIN_SCORE,
    rewritten_min_score: float = DEFAULT_REWRITTEN_MIN_SCORE,
    relative_floor: float = DEFAULT_RELATIVE_FLOOR,
):
    """构建说明书检索子图。

    `registry` 是必需的位置参数：型号从设备实例上读，没有注册中心就没有型号，
    也就没有型号过滤——那正是这个子图最该防住的失败模式，所以不给它默认值。
    `llm` 可以是 None，此时重写走词表、作答走片段拼接。
    """

    def identify(state: RagState) -> dict:
        resolution = resolve_device(
            state["query"],
            registry,
            active_device_id=state.get("active_device_id"),
            active_room_id=state.get("active_room_id"),
        )
        trajectory = [{
            "step": "identify",
            "device_id": resolution.device_id,
            "model": resolution.model,
            "status": resolution.status,
            "basis": resolution.basis,
        }]
        return {
            "device_id": resolution.device_id,
            "device_name": resolution.device_name,
            "device_model": resolution.model,
            "resolution_status": resolution.status,
            "refusal_reason": _refusal_reason(resolution),
            "rewrite_count": 0,
            "trajectory": trajectory,
        }

    def after_identify(state: RagState) -> str:
        # 只有解析到唯一设备且该设备登记了型号，才允许进检索。
        # 其余三种状态一律拒答：这里若"兜底"成不带型号搜全库，
        # 用户就会拿到另一台设备的说明书，而且完全看不出来。
        return "retrieve" if state.get("resolution_status") == "resolved" else "refuse"

    def retrieve(state: RagState) -> dict:
        query = state.get("rewritten_query") or state["query"]
        hits = knowledge.search(query, model=state["device_model"], top_k=top_k)
        # 分数下限分三档，对应三种检索机制：
        # 1. 查询带错误码 —— `search` 已经把不含该码的小节全部滤掉，活下来的必然就是
        #    讲这个码的那一节，相似度分数毫无意义："空调显示 E4 是什么意思"里
        #    "是什么意思"永远匹配不上说明书，分数被摊低，
        #    但那一节正是唯一正确答案。拿模糊阈值卡精确命中会把对的判成"查不到"。
        # 2. 原始口语 —— 用 min_score，滤掉靠通用词凑出来的噪声命中。
        # 3. 重写之后 —— 用更严的 rewritten_min_score，理由见常量定义处。
        if CODE_PATTERN.search(query.lower()):
            floor = 0.0
        elif state.get("rewrite_count", 0) > 0:
            floor = rewritten_min_score
        else:
            floor = min_score
        serialized = [hit.model_dump() for hit in hits if hit.score >= floor]
        # 再做一次**相对**截断：比最佳证据弱太多的候选不进引用。
        # 参照点用第一名而不是最高置信度，这样第一名永远保留——
        # 这道截断只负责让引用更干净，不负责把"答得出来"变成"拒答"。
        if serialized:
            cutoff = serialized[0]["score"] * relative_floor
            serialized = [hit for hit in serialized if hit["score"] >= cutoff]
        trajectory = list(state.get("trajectory", []))
        trajectory.append({
            "step": "retrieve",
            "query": query,
            "model": state["device_model"],
            "hit_count": len(serialized),
            "top_score": round(hits[0].score, 3) if hits else 0.0,
            "score_floor": floor,
            # 语义通道这次有没有参与。降级必须在轨迹里可见，否则"召回突然变差"
            # 只能靠翻日志才知道是 embedding 没接上，而不是检索逻辑退化。
            "dense": knowledge.dense_enabled,
            # 两个通道各自的原始分与名次，排查"为什么这一节排第一"时要看它。
            "signals": hits[0].signals if hits else {},
        })
        return {"hits": serialized, "trajectory": trajectory}

    def after_retrieve(state: RagState) -> str:
        if state.get("hits"):
            return "self_check"
        if state.get("rewrite_count", 0) >= max_rewrites:
            return "refuse"
        # 带错误码的查询不重写。错误码是精确键：语料里没有 E9，
        # 换任何说法也变不出 E9 来，重写只会把用户引到某个"看起来相关"的小节，
        # 得到一段格式权威但答的不是这个码的内容。宁可直说没查到。
        if CODE_PATTERN.search(state["query"].lower()):
            return "refuse"
        return "rewrite"

    def rewrite(state: RagState) -> dict:
        rewritten, source = _rewrite_query(
            state["query"], state["device_model"], knowledge=knowledge, llm=llm
        )
        trajectory = list(state.get("trajectory", []))
        trajectory.append({"step": "rewrite", "query": rewritten, "source": source})
        return {
            "rewritten_query": rewritten,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "trajectory": trajectory,
        }

    def self_check(state: RagState) -> dict:
        """把最相关小节的排查清单分成"系统已核对"和"需你确认"两摞。

        只处理排名第一的小节：第二三名是"也沾点关系"的小节，
        把它们的清单一起核对会让答案里冒出一堆和当前症状无关的核对结论。
        """
        top_chunk = state["hits"][0]["chunk"]
        checklist = top_chunk.get("checklist", [])
        outcomes: list[dict] = []
        manual: list[str] = []

        context = _build_check_context(registry, state.get("device_id"))
        for item in checklist:
            check_id = item.get("check_id")
            if check_id is None or context is None:
                # 标了 <!--manual--> 的，或设备已不在注册中心（无法读状态）的，
                # 都退回人工确认。缺状态时假定"核对通过"是最危险的做法。
                manual.append(item["text"])
                continue
            outcomes.append(run_self_check(check_id, item["text"], context).model_dump())

        trajectory = list(state.get("trajectory", []))
        trajectory.append({
            "step": "self_check",
            "section": top_chunk["section"],
            "auto": len(outcomes),
            "manual": len(manual),
            "problems": sum(1 for outcome in outcomes if outcome["verdict"] == "problem"),
        })
        return {"check_outcomes": outcomes, "manual_items": manual, "trajectory": trajectory}

    def answer(state: RagState) -> dict:
        hits = state["hits"]
        citations = [f"{hit['chunk']['source']}#{hit['chunk']['section']}" for hit in hits]
        outcomes = state.get("check_outcomes", [])
        manual = state.get("manual_items", [])

        body, synthesized = _synthesize(
            llm,
            query=state["query"],
            device_name=state.get("device_name") or "该设备",
            # 走到 answer 必然经过 identify 的 resolved 分支，model 不会是 None；
            # 兜底成空串只是给类型检查器一个确定性，语义上用不到。
            device_model=state["device_model"] or "",
            hits=hits,
            outcomes=outcomes,
        )

        sections = [body]
        if outcomes:
            sections.append(_render_outcomes(outcomes))
        if manual:
            sections.append(
                "需你确认（系统读不到，得人到现场）：\n"
                + "\n".join(f"- {text}" for text in manual)
            )
        sections.append("来源：\n" + "\n".join(f"- {citation}" for citation in citations))

        trajectory = list(state.get("trajectory", []))
        trajectory.append({
            "step": "answer",
            "citation_count": len(citations),
            "synthesized": synthesized,
        })
        return {
            "answer": "\n\n".join(sections),
            "citations": citations,
            "rag_status": "answered",
            "trajectory": trajectory,
        }

    def refuse(state: RagState) -> dict:
        reason = state.get("refusal_reason") or _NO_DOCUMENT_REFUSAL
        trajectory = list(state.get("trajectory", []))
        trajectory.append({
            "step": "refuse",
            "reason": state.get("resolution_status") or "no_supported_document",
        })
        return {
            "answer": reason,
            "citations": [],
            "rag_status": "refused",
            "trajectory": trajectory,
        }

    graph = StateGraph(RagState)
    graph.add_node("identify", identify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rewrite", rewrite)
    graph.add_node("self_check", self_check)
    graph.add_node("answer", answer)
    graph.add_node("refuse", refuse)
    graph.set_entry_point("identify")
    graph.add_conditional_edges("identify", after_identify, {"retrieve": "retrieve", "refuse": "refuse"})
    graph.add_conditional_edges(
        "retrieve",
        after_retrieve,
        {"self_check": "self_check", "rewrite": "rewrite", "refuse": "refuse"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("self_check", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("refuse", END)
    return graph.compile()


# ============================================================
# 拒答文案
# ============================================================
# 四种拒答都必须说清"是哪一步没成立"，并且都带上"不能可靠确认"这句。
# 拒答纪律是要专门设计的：模型的默认行为是尽量给个答案，
# 而这里"给不出答案"往往才是正确输出。

_NO_DOCUMENT_REFUSAL = (
    "这台设备的说明书里没有能支撑答案的对应条目，我不能可靠确认。"
    "请补充故障现象的更多细节，或把这份说明书加进知识库。"
)


def _refusal_reason(resolution: DeviceResolution) -> str:
    if resolution.status == "ambiguous":
        names = "、".join(resolution.candidates)
        return (
            f"家里有多台设备符合你的描述（{names}），它们的说明书并不通用，"
            "我不能可靠确认你问的是哪一台。请说明具体是哪一台。"
        )
    if resolution.status == "no_model":
        return (
            f"{resolution.device_name}没有登记设备型号，我找不到对应的说明书，"
            "不能可靠确认它的故障处理方式。请补充型号，或直接联系售后。"
        )
    if resolution.status == "unknown":
        return (
            "我没能确认你说的是哪一台设备，因此不能可靠确认该查哪份说明书。"
            "请说明设备名称（例如「客厅空调」）。"
        )
    return ""


# ============================================================
# 查询重写
# ============================================================


def _rewrite_query(
    query: str,
    model: str | None,
    *,
    knowledge: KnowledgeBase,
    llm=None,
) -> tuple[str, str]:
    """把口语症状换成说明书用词，返回（新查询, 来源）。

    先让模型在**该型号说明书真实存在的小节标题**里挑一个——不是让它自由发挥关键词。
    模型给的标题必须能在标题清单里对上，否则视为无效、退回词表。
    这一层校验是必要的：一个凭空编出来的小节名会让重写后的检索比重写前更差，
    而且失败得很安静。

    模型还必须能说"都不符合"，见 `_llm_rewrite`。
    """
    titles = knowledge.section_titles(model)
    if llm is not None and titles:
        candidate, no_match = _llm_rewrite(llm, query, titles)
        if no_match:
            # 模型看着该型号真实的小节清单，明确说没有一个相符。这比词表更可信：
            # 词表只看用户措辞（"响" → "噪音异常 噪音"），根本不知道这个型号
            # 到底有没有噪音章节。此时保持原句去查一次然后拒答，
            # 不要再拿词表凑一个"看起来查到了"的结果。
            return query, "llm-no-match"
        if candidate:
            return candidate, "llm"
    lexical = _lexical_rewrite(query)
    if lexical:
        return lexical, "lexicon"
    # 两条路都没结论时保持原句：重写不出来就让它按原样再查一次然后拒答，
    # 而不是瞎改一版制造出个"看起来查到了"的结果。
    return query, "unchanged"


# 模型用来表示"没有任何小节相符"的输出。
_NO_MATCH_REPLY = "无"


def _llm_rewrite(llm, query: str, titles: list[str]) -> tuple[str | None, bool]:
    """让模型在真实小节里挑一个，返回（重写后的查询, 模型是否明确说都不符合）。

    为什么必须给模型一个"都不符合"的出口
    ----------------------------------
    这是端到端实测抓到的一个拒答纪律漏洞。原来的 prompt 是"请判断最可能属于哪一个小节"，
    **强制单选**——当正确答案根本不在清单里时，模型必然挑一个最像的。
    而校验只检查"标题真实存在"，存在不等于相关，于是重写后的查询轻松过了阈值：

        "卧室空调有点响是怎么了" → FrostLine 说明书没有噪音章节
          → 模型被迫挑了「蒸发器结霜」
          → 检索命中，答案挂上三条引用，rag_status=answered

    模型自己在正文里都写了"说明书片段中未提及空调异响的相关信息"，但系统仍然
    以权威格式把引用递了出去——这正是拒答纪律要防的那种输出。

    这个漏洞 012 就存在，只是当时测试都传 `llm=None`（走词表重写），从没暴露。

    取舍：信"无"会让"其实有相符小节但模型看漏了"的情况变成一次拒答（少答一次）；
    不信会让"没有相符小节"变成一次权威的错答。后者是这个项目明确认定更危险的那一类。
    """
    prompt = (
        "用户在描述一台家电的故障，原话是：\n"
        f"{query}\n\n"
        "这台设备的说明书包含以下小节：\n"
        + "\n".join(f"- {title}" for title in titles)
        + "\n\n请判断用户描述的最可能属于哪一个小节。"
        "只输出该小节的标题，后面可以再跟不超过 3 个说明书里会出现的关键词，用空格分隔。"
        f"如果没有任何小节与用户描述的问题相符，只输出「{_NO_MATCH_REPLY}」这一个字，"
        "不要勉强挑一个最接近的。"
        "不要输出解释、编号或标点。"
    )
    try:
        response = llm.invoke(prompt)
    except GraphInterrupt:
        # 人工审批是靠抛 GraphInterrupt 实现的，吞掉它等于用户还没回答就重跑节点。
        raise
    except Exception as error:  # noqa: BLE001 —— 重写失败不该让整条链路挂掉
        logger.warning(f"知识检索查询重写失败，退回词表 | error={error}")
        return None, False

    text = getattr(response, "content", response)
    if not isinstance(text, str):
        return None, False
    text = text.strip()
    if text == _NO_MATCH_REPLY:
        return None, True
    # 必须真的对上一个已有小节标题，否则当作无效重写。
    if text and any(title in text for title in titles):
        return text, False
    return None, False


def _lexical_rewrite(query: str) -> str | None:
    terms = [replacement for pattern, replacement in _SYMPTOM_LEXICON if pattern.search(query)]
    return " ".join(terms) if terms else None


# ============================================================
# 自证核对
# ============================================================


def _build_check_context(registry: DeviceRegistry, device_id: str | None) -> CheckContext | None:
    """组装自证核对能看到的事实：这台设备 + 同房间的温湿度读数。

    刻意**不**调用 `registry.tick_environment()`。环境推演只允许 `read_sensor` 触发，
    在这里顺手推一把，会让同一轮对话里的室温随调用次数漂移，
    诊断结论就再也复现不了了。get / get_by_type 都是稳定快照读。
    """
    if device_id is None:
        return None
    device = registry.get(device_id)
    if device is None:
        return None

    temperature: float | None = None
    humidity: int | None = None
    sensor_name: str | None = None
    for sensor in registry.get_by_type(DeviceType.TEMP_HUMIDITY_SENSOR).values():
        # 同房间配对用的是模拟器自己的约定：location 字面量相等。
        if sensor.location and sensor.location == device.location:
            temperature = getattr(sensor, "temperature", None)
            humidity = getattr(sensor, "humidity", None)
            sensor_name = sensor.name
            break

    return CheckContext(
        device=device,
        room_temperature=temperature,
        room_humidity=humidity,
        sensor_name=sensor_name,
    )


_VERDICT_MARKS = {"problem": "✗", "ok": "✓", "unknown": "?"}


def _render_outcomes(outcomes: list[dict]) -> str:
    """核对结果由代码渲染，不经模型改写。

    模型会把"设定温度 26°C，室温 26°C"复述成"温度设置正常"——
    只有实测值能反驳它，所以这一段必须是代码直接陈述事实。
    """
    lines = ["自动核对结果（已读取设备真实状态）："]
    for outcome in outcomes:
        mark = _VERDICT_MARKS.get(outcome["verdict"], "?")
        lines.append(f"{mark} {outcome['item_text']}\n    {outcome['detail']}")
    return "\n".join(lines)


# ============================================================
# 答案综合
# ============================================================


def _synthesize(
    llm,
    *,
    query: str,
    device_name: str,
    device_model: str,
    hits: list[dict],
    outcomes: list[dict],
) -> tuple[str, bool]:
    """返回（正文, 是否由模型综合）。

    模型只写处理建议，且被明确告知不要复述核对结果和来源——那两块由代码拼接。
    模型不可用或调用失败时退回原文拼接：信息量下降，但引用与核对结论一个不少。
    """
    excerpts = "\n\n".join(
        f"【{hit['chunk']['title']} - {hit['chunk']['section']}】\n{hit['chunk']['content']}"
        for hit in hits
    )
    if llm is None:
        return f"根据{device_name}（{device_model}）的说明书：\n\n{excerpts}", False

    problems = [outcome["detail"] for outcome in outcomes if outcome["verdict"] == "problem"]
    problem_block = (
        "系统已核对出的异常（请优先围绕它给建议）：\n" + "\n".join(f"- {text}" for text in problems)
        if problems
        else "系统自动核对没发现异常。"
    )
    prompt = (
        f"用户的{device_name}（型号 {device_model}）出现问题，原话是：{query}\n\n"
        f"以下是该型号说明书的相关片段：\n{excerpts}\n\n"
        f"{problem_block}\n\n"
        "请基于上述片段写一段处理建议，要求：\n"
        "1. 只使用片段里的信息，片段没提到的不要补充；\n"
        "2. 不超过 4 句话；\n"
        "3. 不要罗列检查清单、不要复述核对结果、不要写来源——这些由系统另行附上；\n"
        "4. 涉及断电、联系售后这类安全提示时必须保留。"
    )
    try:
        response = llm.invoke(prompt)
    except GraphInterrupt:
        raise
    except Exception as error:  # noqa: BLE001 —— 综合失败要降级，不能让用户什么都拿不到
        logger.warning(f"知识检索答案综合失败，退回片段拼接 | error={error}")
        return f"根据{device_name}（{device_model}）的说明书：\n\n{excerpts}", False

    text = getattr(response, "content", response)
    if not isinstance(text, str) or not text.strip():
        return f"根据{device_name}（{device_model}）的说明书：\n\n{excerpts}", False
    return text.strip(), True
