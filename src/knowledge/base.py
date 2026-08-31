"""Markdown 说明书索引 + 混合检索入口。

关于说明书里的行内标注
--------------------
排查清单的条目末尾可以挂一个 Markdown 注释：

    1. 确认空调处于开机状态。<!--check:device_is_on-->
    4. 检查进风口滤网是否积尘。<!--manual-->

`check:xxx` 表示这一项系统能自己核对（xxx 是 `selfcheck.SELF_CHECKS` 里的 id），
`manual` 表示必须人到现场动手。

为什么标注写在正文行内，而不是在 catalog.json 里另开一张表：

- **同位。** 标注跟着它描述的那一行走。写在索引里就要靠"第几节第几条"定位，
  说明书改一行顺序，全部标注错位，而且错位是静默的。
- **不可见。** Markdown 注释渲染时不显示，说明书在任何阅读器里仍然是一份说明书。
- **正文不受污染。** 解析时标注被剥离：既不进给用户看的文本，也不进检索词表
  （否则 `ac_target_temp_below_room` 这种 id 会被切成 latin token 参与打分，
  搜"target"能搜出空调说明书）。

检索分两层：这里只做**硬过滤**，打分交给 `retrieval.HybridRetriever`
---------------------------------------------------------------
`search` 自己只负责挑候选：型号必须相等、查询里的错误码必须全部出现在该小节。
这两道是**准入条件而不是打分项**，所以必须在任何通道之前生效——
理由见 `retrieval.py` 的模块 docstring（向量检索会把 E4 和 E7 判为近邻）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

from .embeddings import EmbeddingProvider, NullEmbeddings
from .retrieval import HybridRetriever
from .selfcheck import KNOWN_CHECK_IDS
from .tokenizer import extract_codes

# <!--check:xxx--> 或 <!--manual-->。允许注释内有空格，因为编辑器格式化会加。
_ANNOTATION_PATTERN = re.compile(r"<!--\s*(?:check:([a-z0-9_]+)|manual)\s*-->")


class ChecklistItem(BaseModel):
    """说明书排查清单里的一条，带上"谁来核对"的标注。

    check_id 为 None 表示这一项标了 `<!--manual-->`——必须人工确认。
    """

    text: str
    check_id: str | None = None


class KnowledgeChunk(BaseModel):
    document_id: str
    title: str
    model: str
    source: str
    section: str
    content: str
    # 只有带标注的小节才非空（故障码说明是散文，没有清单）。
    checklist: list[ChecklistItem] = []


class KnowledgeHit(BaseModel):
    chunk: KnowledgeChunk
    # score 是 [0,1] 的绝对置信度（`HybridScores.confidence`），三档下限套在它身上。
    # 名次由 signals["rrf"] 决定，两者分工见 retrieval.py 的模块 docstring。
    score: float
    # 两个通道各自的原始分与名次，写进 RAG 轨迹用于排查"为什么这一节排第一"。
    signals: dict = {}


class KnowledgeBase:
    def __init__(
        self,
        root: str | Path,
        *,
        embeddings: EmbeddingProvider | None = None,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        """加载语料并建索引。

        `embeddings` 默认是 `NullEmbeddings`——**明确宣告没有语义通道**，检索退化为
        纯 BM25。这样测试不需要 API Key，而且"没配模型"和"模型坏了"是两条可分的路径，
        不会互相掩盖。真实语义能力由调用方注入 `ApiEmbeddings` 提供
        （见 `graph.py` 与 `embeddings.build_embeddings`）。

        为什么不留一个离线的假向量通道兜底：试过，被实测否决了，
        理由写在 `embeddings.py` 的模块 docstring 里——没有 IDF 的向量通道
        会按"共享了几个常见词"排序，等于从旁路把 BM25 刚压掉的噪声放回来。
        """
        self.root = Path(root)
        self.chunks = self._load()
        self.retriever = HybridRetriever(
            [_searchable_text(chunk) for chunk in self.chunks],
            embeddings=embeddings or NullEmbeddings(),
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
        # 每个小节里出现过的错误码，构造期算一次。旧实现在**每次检索的每个小节上**
        # 重新跑一遍正则和分词——加载是一次性的，索引却不是。
        self._codes = [extract_codes(_searchable_text(chunk)) for chunk in self.chunks]

    @property
    def dense_enabled(self) -> bool:
        """语义通道这次到底有没有建起来。降级必须可见，不能只写在日志里。"""
        return self.retriever.dense_enabled

    def _load(self) -> list[KnowledgeChunk]:
        catalog_path = self.root / "catalog.json"
        if not catalog_path.exists():
            return []
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        chunks = []
        for item in catalog.get("documents", []):
            path = self.root / item["file"]
            if not path.exists():
                raise FileNotFoundError(
                    f"catalog.json 登记了 {item['file']}，但 {path} 不存在。"
                    "这份说明书永远检索不到；宁可构造期失败，不要静默跳过。"
                )
            for section, raw in _split_markdown(path.read_text(encoding="utf-8")):
                content, checklist = _extract_checklist(raw)
                _validate_check_ids(checklist, source=item["file"], section=section)
                chunks.append(KnowledgeChunk(
                    document_id=item["id"], title=item["title"], model=item["model"],
                    source=item["file"], section=section, content=content,
                    checklist=checklist,
                ))
        return chunks

    def search(self, query: str, *, model: str | None, top_k: int = 3) -> list[KnowledgeHit]:
        """按型号过滤后做混合检索（BM25 + 向量，RRF 融合名次）。

        `model` 是**关键字必填**参数，不给默认值：`model=None` 意味着搜索整个语料库，
        跨型号的说明书会一起被召回，而"查错型号的说明书比查不到更危险"。
        这个降级必须是调用方写出来的显式选择，不能是漏传参数的副产品。
        """
        query_codes = extract_codes(query)
        candidates = [
            index
            for index, chunk in enumerate(self.chunks)
            # 硬过滤一：型号必须逐字相等。
            if not (model and chunk.model != model)
            # 硬过滤二：查询里的错误码必须**全部**出现在这一节。错误码是精确键，
            # 不能降级成相似度的一部分——E4 的语义近邻是 E5/E7，讲的是别的故障。
            and (not query_codes or query_codes.issubset(self._codes[index]))
        ]

        scored = self.retriever.score(query, candidates)
        hits = [
            KnowledgeHit(
                chunk=self.chunks[index],
                score=scores.confidence,
                signals=scores.as_dict(),
            )
            for index, scores in scored.items()
        ]
        # 名次由 RRF 决定，confidence 只在 RRF 同分时参与；末两级是稳定 tiebreak。
        # RRF 是 `1/(k+rank)` 的和，取值离散，同分概率比连续分数高得多，
        # 没有确定性 tiebreak 会让测试随机失败。
        hits.sort(
            key=lambda hit: (
                -hit.signals["rrf"],
                -hit.score,
                hit.chunk.document_id,
                hit.chunk.section,
            )
        )
        return hits[:top_k]

    def section_titles(self, model: str | None) -> list[str]:
        """该型号说明书里真实存在的小节标题，保持语料顺序。查询重写要拿它做校验。"""
        seen: list[str] = []
        for chunk in self.chunks:
            if model and chunk.model != model:
                continue
            if chunk.section not in seen:
                seen.append(chunk.section)
        return seen


def _searchable_text(chunk: KnowledgeChunk) -> str:
    """一个小节参与检索的全文。

    用换行而不是空格拼接：向量通道对"这看起来像一份文档"敏感，
    标题、小节名、正文分行更接近它训练时见过的形态。
    对 BM25 没有区别，两者用同一份文本是为了避免两个通道看到不同的语料
    （那会让"某个词到底在不在索引里"变成两个答案）。
    """
    return f"{chunk.title}\n{chunk.section}\n{chunk.content}"


def _split_markdown(text: str) -> list[tuple[str, str]]:
    sections = []
    current_title = "正文"
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        elif not line.startswith("# "):
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(title, content) for title, content in sections if content]


def _extract_checklist(raw: str) -> tuple[str, list[ChecklistItem]]:
    """剥离行内标注，返回（干净正文, 清单条目）。

    没有任何标注的小节返回空清单——不去猜"看起来像清单的行"。
    自动核对是要读设备真实状态并给用户下结论的，识别范围必须由语料显式声明，
    不能靠正文长得像不像列表来推断。
    """
    clean_lines: list[str] = []
    checklist: list[ChecklistItem] = []
    for line in raw.splitlines():
        match = _ANNOTATION_PATTERN.search(line)
        clean = _ANNOTATION_PATTERN.sub("", line).rstrip()
        clean_lines.append(clean)
        if match is None:
            continue
        # 条目文本用剥离标注后的原文（含"1. "序号前缀），
        # 这样回给用户时跟说明书逐字一致，用户能对上号。
        item_text = clean.strip()
        if item_text:
            checklist.append(ChecklistItem(text=item_text, check_id=match.group(1)))
    return "\n".join(clean_lines).strip(), checklist


def _validate_check_ids(checklist: list[ChecklistItem], *, source: str, section: str) -> None:
    """语料引用了未声明的 check id 就直接炸——构造期失败，不留到运行期。

    这里是语料与代码之间唯一的引用关系。若容忍未知 id（比如降级成"需人工确认"），
    一个拼错的 id 就会让一条本该自动核对的检查项无声消失：功能变弱但不报错，
    是最难定位的那类缺陷。宁可启动失败。
    """
    for item in checklist:
        if item.check_id is not None and item.check_id not in KNOWN_CHECK_IDS:
            raise ValueError(
                f"{source} 的「{section}」小节引用了未声明的自证检查 id "
                f"'{item.check_id}'；已声明的是 {sorted(KNOWN_CHECK_IDS)}。"
                "请在 src/knowledge/selfcheck.py 的 SELF_CHECKS 里补声明，"
                "或把该条目改标为 <!--manual-->。"
            )
