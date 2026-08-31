"""说明书检索的召回评测：旧词法 vs BM25 vs 纯向量 vs 混合。

为什么需要这个
------------
013 的主张是"混合检索比 012 的词法检索召回更好"。这句话如果没有数字，就只是
一句话。`docs/gap-analysis.md` 把"没有评测体系"列为学习价值最高的缺口，理由正是
这个：一行 prompt 或一个常量改动让召回从 0.9 掉到 0.6，现有单元测试**全绿**，
没有任何信号。

这个 runner 测的是**首轮检索**，刻意不跑查询重写。
理由是它要回答的问题就是"语义通道能不能替掉那张人工维护的症状词表"——
把重写也跑上，等于让词表替检索背书，测不出通道本身的能力。

四种配置的意义
------------
- ``legacy``  012 的算法（bigram 集合交集 ÷ 查询词数），冻结在本文件里做基线。
- ``bm25``    只有词法通道：换成 jieba 分词 + BM25（有 IDF、有 TF、有长度归一）。
              和 legacy 的差值就是"把覆盖率换成 BM25"单独带来的收益。
- ``dense``   只有向量通道。用来回答"是不是干脆别要 BM25 了"。
- ``hybrid``  生产配置。

跑法::

    PYTHONIOENCODING=utf-8 python -m src.evaluation.recall            # 全部四种
    PYTHONIOENCODING=utf-8 python -m src.evaluation.recall --offline  # 跳过需要 API 的两种
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from ..knowledge.base import KnowledgeBase, _searchable_text
from ..knowledge.embeddings import ApiEmbeddings, EmbeddingProvider, NullEmbeddings
from ..knowledge.rag import DEFAULT_MIN_SCORE, DEFAULT_RELATIVE_FLOOR
from ..knowledge.tokenizer import CODE_PATTERN

DATASET_PATH = Path("evals/knowledge_recall.json")
TOP_K = 3

# 012 的三档下限里，本 runner 只会用到两档：带错误码是 0，原始口语是这一档。
# 第三档（重写后）用不上，因为这里刻意不跑重写。
_LEGACY_MIN_SCORE = 0.15


# ============================================================
# 基线：012 的词法打分，冻结副本
# ============================================================


class LegacyLexicalRetriever:
    """012 的检索算法，**只为做对比而保留**，不要在生产代码里引用它。

    刻意抄成一份独立实现而不是从 git 历史里翻：基线必须能和新实现在同一次运行里
    跑同一份数据，否则"提升了多少"这个数字没法复现。

    算法：中文按相邻两字切 bigram（拼接后滑窗，跨字段边界），英文按正则切词，
    两边求集合交集，除以查询词数。没有 TF、没有 IDF、没有长度归一。
    """

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self._chunks = knowledge.chunks
        self._texts = [_searchable_text(chunk).lower() for chunk in knowledge.chunks]
        self._terms = [_legacy_terms(text) for text in self._texts]
        self._codes = [set(re.findall(r"\b(?:[a-z]\d+|\d{3,})\b", text)) for text in self._texts]

    def search(self, query: str, *, model: str | None, top_k: int = TOP_K) -> list[tuple[str, float]]:
        query_terms = _legacy_terms(query.lower())
        query_codes = set(re.findall(r"\b(?:[a-z]\d+|\d{3,})\b", query.lower()))
        hits: list[tuple[float, str, str, str]] = []
        for index, chunk in enumerate(self._chunks):
            if model and chunk.model != model:
                continue
            if query_codes and not query_codes.issubset(self._codes[index]):
                continue
            overlap = query_terms & self._terms[index]
            if not overlap:
                continue
            score = len(overlap) / max(1, len(query_terms))
            hits.append((score, chunk.document_id, chunk.section, f"{chunk.source}#{chunk.section}"))
        hits.sort(key=lambda row: (-row[0], row[1], row[2]))
        return [(row[3], row[0]) for row in hits[:top_k]]


def _legacy_terms(text: str) -> set[str]:
    latin = set(re.findall(r"[a-z0-9_-]+", text))
    chinese = "".join(re.findall(r"[一-鿿]", text))
    grams = {chinese[index:index + 2] for index in range(max(0, len(chinese) - 1))}
    return latin | grams


# ============================================================
# 指标
# ============================================================


@dataclass
class Metrics:
    name: str
    graded: int = 0            # 有期望答案的用例数
    hit_at_1: int = 0
    hit_at_3: int = 0
    reciprocal_ranks: float = 0.0
    refusals: int = 0          # 应当查不到的用例数
    refusals_correct: int = 0
    by_kind: dict[str, list[int]] = field(default_factory=dict)  # kind -> [正确数, 总数]

    def record(self, kind: str, correct: bool) -> None:
        bucket = self.by_kind.setdefault(kind, [0, 0])
        bucket[0] += int(correct)
        bucket[1] += 1

    @property
    def recall_at_1(self) -> float:
        return self.hit_at_1 / self.graded if self.graded else 0.0

    @property
    def recall_at_3(self) -> float:
        return self.hit_at_3 / self.graded if self.graded else 0.0

    @property
    def mrr(self) -> float:
        return self.reciprocal_ranks / self.graded if self.graded else 0.0

    @property
    def refusal_accuracy(self) -> float:
        return self.refusals_correct / self.refusals if self.refusals else 0.0


def _admitted(ranked: list[tuple[str, float]], query: str, floor: float) -> list[str]:
    """套用 rag.py 的准入规则：带错误码的查询不设绝对下限，其余用 floor；
    然后再做一次相对截断（比第一名弱太多的不进引用）。

    评测必须和生产用**同一套**准入规则，否则测出来的召回率里混着生产根本不会采纳的命中。
    `legacy` 那一档也套同样的相对截断——不然新旧对比里新方案会因为"引用更少"而吃亏，
    比的就不是同一件事了。
    """
    effective = 0.0 if CODE_PATTERN.search(query.lower()) else floor
    admitted = [(citation, score) for citation, score in ranked if score >= effective]
    if admitted:
        cutoff = admitted[0][1] * DEFAULT_RELATIVE_FLOOR
        admitted = [(c, s) for c, s in admitted if s >= cutoff]
    return [citation for citation, _ in admitted]


def evaluate(cases: list[dict], search, *, name: str, floor: float) -> Metrics:
    metrics = Metrics(name=name)
    for case in cases:
        expected = set(case["expected"])
        if case["model"] is None:
            # 型号为 null 表示实体消解那一层就没定下是哪台设备（客厅灯没登记型号、
            # 或者问的根本不是设备）。生产路径在 `after_identify` 就拒答了，**不进检索**，
            # 所以这里也不能调 search。
            #
            # 这一条不是形式主义：`KnowledgeBase.search(model=None)` 的语义是
            # "搜整个语料库"而不是"拒答"。把 None 透传下去，"客厅灯的灯罩怎么拆"
            # 会命中 GlowSoft 的清洁步骤——看起来还挺像正确答案，
            # 于是评测会给一条本该拒答的用例打满分。
            admitted: list[str] = []
        else:
            admitted = _admitted(search(case["query"], case["model"]), case["query"], floor)

        if not expected:
            metrics.refusals += 1
            correct = not admitted
            metrics.refusals_correct += int(correct)
            metrics.record(case["kind"], correct)
            continue

        metrics.graded += 1
        rank = next((i for i, c in enumerate(admitted, start=1) if c in expected), None)
        correct = rank == 1
        metrics.hit_at_1 += int(correct)
        metrics.hit_at_3 += int(rank is not None)
        metrics.reciprocal_ranks += 1.0 / rank if rank else 0.0
        metrics.record(case["kind"], correct)
    return metrics


# ============================================================
# 配置与主流程
# ============================================================


def _api_embeddings() -> EmbeddingProvider | None:
    """从环境变量拼出远程 provider。缺配置就返回 None，让调用方跳过那两种配置。

    这里显式 `load_dotenv(".env")`：这个 runner 是独立入口，不经过 `Settings`，
    不加载就只能读到进程环境变量，表现是"明明 .env 里配了却一直跳过 hybrid"。
    路径写死而不用无参调用：无参版会回溯调用栈去找 .env，从 `python -c` /stdin
    调进来时拿不到调用者文件名，直接 AssertionError。
    """
    load_dotenv(".env")
    model = os.getenv("RAG_EMBEDDING_MODEL_ID") or "text-embedding-v4"
    base_url = os.getenv("RAG_EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL") or ""
    api_key = os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY") or ""
    if not (model and base_url and api_key):
        return None
    return ApiEmbeddings(model=model, base_url=base_url, api_key=api_key)


def run(knowledge_path: str = "docs/knowledge", *, offline: bool = False) -> list[Metrics]:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]

    lexical_kb = KnowledgeBase(knowledge_path, embeddings=NullEmbeddings())
    # 索引在循环外建一次。放进 lambda 里会让每条用例重算一遍全语料的 bigram 集合，
    # 60 条用例就是 60 次——评测本身慢到没人愿意跑，就等于没有评测。
    legacy = LegacyLexicalRetriever(lexical_kb)
    results = [
        evaluate(
            cases,
            lambda q, m: legacy.search(q, model=m),
            name="legacy（012 词法覆盖率）",
            floor=_LEGACY_MIN_SCORE,
        ),
        evaluate(
            cases,
            lambda q, m: [
                (f"{h.chunk.source}#{h.chunk.section}", h.score)
                for h in lexical_kb.search(q, model=m, top_k=TOP_K)
            ],
            name="bm25（仅词法通道）",
            floor=DEFAULT_MIN_SCORE,
        ),
    ]

    embeddings = None if offline else _api_embeddings()
    if embeddings is None:
        print("跳过 dense / hybrid：未配置 embedding（或指定了 --offline）\n")
        return results

    for name, bm25_weight, dense_weight in (
        ("dense（仅向量通道）", 0.0, 1.0),
        ("hybrid（BM25 + 向量，生产配置）", 0.5, 0.5),
    ):
        kb = KnowledgeBase(
            knowledge_path,
            embeddings=embeddings,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
        results.append(
            evaluate(
                cases,
                lambda q, m, kb=kb: [
                    (f"{h.chunk.source}#{h.chunk.section}", h.score)
                    for h in kb.search(q, model=m, top_k=TOP_K)
                ],
                name=name,
                floor=DEFAULT_MIN_SCORE,
            )
        )
    return results


def _report(results: list[Metrics]) -> None:
    print(f"{'配置':32s} {'Recall@1':>9s} {'Recall@3':>9s} {'MRR':>7s} {'拒答准确':>9s}")
    print("-" * 72)
    for metrics in results:
        print(
            f"{metrics.name:32s} {metrics.recall_at_1:>8.1%} {metrics.recall_at_3:>9.1%} "
            f"{metrics.mrr:>7.3f} {metrics.refusal_accuracy:>9.1%}"
        )

    kinds = sorted({kind for metrics in results for kind in metrics.by_kind})
    print("\n分类 Recall@1（拒答类显示的是拒答准确率）")
    header = "".join(f"{kind:>14s}" for kind in kinds)
    print(f"{'配置':32s}{header}")
    print("-" * (32 + 14 * len(kinds)))
    for metrics in results:
        cells = ""
        for kind in kinds:
            correct, total = metrics.by_kind.get(kind, [0, 0])
            cells += f"{f'{correct}/{total}':>14s}" if total else f"{'—':>14s}"
        print(f"{metrics.name:32s}{cells}")


def sweep(knowledge_path: str = "docs/knowledge") -> None:
    """在（向量权重 × 首轮分数下限）网格上扫一遍，用来重新标定。

    换语料、换 embedding 模型之后阈值必须重标，这个入口就是为那件事准备的：
    只看单条查询的分数很容易把参数拧到只对那一条最优。
    这里同时打印召回和拒答准确——它们是**互相拉扯**的两个指标，
    只优化一个必然把另一个搞坏（下限调低召回涨、拒答烂；调高反之）。
    """
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    embeddings = _api_embeddings()
    if embeddings is None:
        print("需要 embedding 配置才能扫参")
        return

    print(f"{'向量权重':>8s} {'下限':>6s} {'Recall@1':>9s} {'Recall@3':>9s} {'MRR':>7s} {'拒答准确':>9s} {'口语类':>8s}")
    print("-" * 62)
    for dense_weight in (0.5, 0.6, 0.7, 0.8, 1.0):
        kb = KnowledgeBase(
            knowledge_path,
            embeddings=embeddings,
            bm25_weight=1.0 - dense_weight,
            dense_weight=dense_weight,
        )
        for floor in (0.12, 0.20, 0.30, 0.35, 0.40, 0.45):
            metrics = evaluate(
                cases,
                lambda q, m, kb=kb: [
                    (f"{h.chunk.source}#{h.chunk.section}", h.score)
                    for h in kb.search(q, model=m, top_k=TOP_K)
                ],
                name="sweep",
                floor=floor,
            )
            spoken = metrics.by_kind.get("colloquial", [0, 0])
            print(
                f"{dense_weight:>8.1f} {floor:>6.2f} {metrics.recall_at_1:>8.1%} "
                f"{metrics.recall_at_3:>9.1%} {metrics.mrr:>7.3f} "
                f"{metrics.refusal_accuracy:>9.1%} {f'{spoken[0]}/{spoken[1]}':>8s}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="说明书检索召回评测")
    parser.add_argument("--knowledge-path", default="docs/knowledge")
    parser.add_argument(
        "--offline", action="store_true",
        help="只跑不需要 embedding 接口的两种配置（legacy / bm25）",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="在（向量权重 × 分数下限）网格上扫参，用于换语料/换模型后重新标定",
    )
    args = parser.parse_args()
    if args.sweep:
        sweep(args.knowledge_path)
        return
    _report(run(args.knowledge_path, offline=args.offline))


if __name__ == "__main__":
    main()
