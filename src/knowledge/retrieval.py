"""混合检索：BM25 词法通道 + 向量语义通道 + RRF 融合。

为什么是"融合定名次、绝对分守门"两套分数
--------------------------------------
这是本模块最要紧的一个设计，先说清楚它解决什么问题。

012 的拒答纪律依赖一个**绝对**判断：所有候选都不够好时必须说"查不到"，
而不是把最不差的那个递出去。它靠三档分数下限实现（`rag.py` 的 0 / 0.15 / 0.4），
成立的前提是分数带量纲——0.11 是噪声、0.75 是真命中。

而 RRF（Reciprocal Rank Fusion）是**纯名次**的：`Σ 1/(k + rank)`。
名次第一的文档永远拿到最高的 RRF 分，**跟它到底像不像毫无关系**。
一个语料里根本没有对应章节的查询，它的第一名照样是满分 RRF。
所以如果直接拿 RRF 分去套三档下限，拒答纪律会当场失效：
所有查询都会有一个"高分"命中，永远不会走到拒答分支。

结论是两套分数各管一件事，不能混：

- ``rrf``        —— 只决定**名次**。名次融合的好处正是不用标定量纲：
                   BM25 无上界、余弦在 [0,1]，直接加权求和需要先把两者拉到同一尺度，
                   而 RRF 对尺度免疫，两个通道谁强谁弱都能稳定融合。
- ``confidence`` —— 只决定**放不放行**，取值锁死在 [0,1]，由两个通道的
                   归一化绝对分加权而来。三档下限套在它身上。

先按 confidence 筛掉不够格的，再按 rrf 排名次。

两个通道各自的归一化都是实测标定的
--------------------------------
- **BM25 无上界**，不能用"除以本次查询的最高分"来归一：那样第一名恒为 1.0，
  又变回"名次分"，绝对性再次丢失。这里用饱和函数 ``x / (x + s)``，
  s 是实测常量，含义是"BM25 到多少分算半信"。
- **余弦有基线偏移**。真实 embedding 上两段完全无关的中文也有 0.3~0.4 的余弦
  （向量空间各向异性），直接当置信度用会让垃圾看起来有四成把握。
  所以要减掉基线再拉回 [0,1]，基线值属于具体模型，挂在 provider 上。

硬过滤为什么必须在两个通道之前
----------------------------
型号相等和错误码 issubset 这两道硬过滤（见 `base.py`）**不是打分项，是准入条件**，
所以它们在候选集阶段就要生效，绝不能变成融合公式里的一个权重。
向量检索尤其危险：E4 的语义近邻天然是 E5、E7——它们讲的是完全不同的故障。
一旦把错误码从硬过滤降级成"相似度的一部分"，用户问 E4 就会拿到 E7 的处理步骤，
而且读起来完全通顺。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from .embeddings import EmbeddingProvider
from .tokenizer import tokenize

# BM25 的噪声基线与饱和常量。两个都是实测标定的（110 个 chunk 的语料）：
#   靠"异常""检查""卧室"这类通用词凑出来的噪声命中   BM25 ≤ 2.96
#   真实词法命中                                    BM25 ≥ 4.26（一路到 30.6）
# 基线取 3.5，落在这条缝里。**低于基线的 BM25 分一律记 0**，而不是让它折算成
# 0.27 那样的小正数——这一点是必需的，理由见 `_channel_strengths`：
# 置信度是"任一通道有确凿证据"，两个都是弱信号时必须得到 0，不能靠累加凑出个中间值。
#
# **这个基线依赖语料规模，换语料必须重标。** BM25 分数由 IDF 与平均文档长度决定，
# 都是语料级统计量。踩过一次：拿三个 chunk 的合成语料写测试，所有 BM25 分数都远低于
# 3.5，于是什么都进不了名次——测出来的是"基线没标定"，而不是被测的机制。
# 重标方法是 `python -m src.evaluation.recall --sweep`。
#
# 饱和常量 6.0 把减掉基线后的分数压进 [0,1) 且保留量纲。
# 别改成"除以本次查询的最高分"——那会让第一名恒等于 1.0，绝对性当场消失，
# 三档下限就再也守不住任何东西（同样的理由让 RRF 不能用来守门）。
_BM25_NOISE_FLOOR = 3.5
_BM25_SATURATION = 6.0

# RRF 的 k。60 是原论文（Cormack 2009）的取值，也是业界默认。
# k 越大，名次之间的差距越平缓、越照顾"两个通道都排中游"的文档；
# k 越小越偏袒各通道的第一名。这里不调它：本语料每个型号只有十来个小节，
# 名次差距本来就小，k 的影响远不如两个通道自身的质量。
_RRF_K = 60


@dataclass(frozen=True)
class HybridScores:
    """一个候选小节在本次查询下的全部分数。两套分数各管一件事，见模块 docstring。"""

    bm25: float
    dense: float
    bm25_rank: int | None
    dense_rank: int | None
    rrf: float
    confidence: float

    def as_dict(self) -> dict:
        """写进 RAG 轨迹用。排查"为什么这一节排第一"时需要看到两个通道各自的贡献。"""
        return {
            "bm25": round(self.bm25, 3),
            "dense": round(self.dense, 3),
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
            "rrf": round(self.rrf, 5),
            "confidence": round(self.confidence, 3),
        }


class HybridRetriever:
    """对一组固定文档做混合检索。文档在构造期分词、建索引、算向量。

    `documents` 的顺序就是候选下标的含义，调用方（`KnowledgeBase`）负责把下标
    映射回 chunk。这样检索层完全不认识"说明书""型号"这些概念，只认下标和文本。
    """

    def __init__(
        self,
        documents: list[str],
        *,
        embeddings: EmbeddingProvider,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = _RRF_K,
        bm25_saturation: float = _BM25_SATURATION,
        bm25_noise_floor: float = _BM25_NOISE_FLOOR,
    ) -> None:
        self._documents = documents
        self._rrf_k = rrf_k
        self._saturation = bm25_saturation
        self._noise_floor = bm25_noise_floor
        self._embeddings = embeddings
        # 余弦的两个刻度属于具体模型，从 provider 上读，不在这里写死。
        self._baseline = float(getattr(embeddings, "baseline_similarity", 0.0))
        self._strong = float(getattr(embeddings, "strong_similarity", 1.0))

        total = bm25_weight + dense_weight
        if total <= 0:
            raise ValueError("bm25_weight 与 dense_weight 不能同时为 0，那样没有任何通道在工作")
        self._bm25_weight = bm25_weight / total
        self._dense_weight = dense_weight / total
        # ---- 词法通道 ----
        # BM25 在**全语料**上建索引，而不是每次按型号过滤后重建。
        # IDF 必须是语料级常量：若按候选子集算，同一个 (查询, 小节) 组合会因为
        # 这次解析到哪个型号而得到不同的分数，测试和调参都失去意义。
        self._tokenized = [tokenize(text) for text in documents]
        self._bm25 = BM25Okapi(self._tokenized) if documents else None

        # ---- 语义通道 ----
        self._matrix: np.ndarray | None = None
        self.dense_enabled = False
        if documents:
            self._build_dense_index()

    # ------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------
    def _build_dense_index(self) -> None:
        """算文档向量。provider 没有语义能力就不建；建失败就整条通道关掉并**大声**记下来。

        先看 `semantic` 再调用，是因为"没有语义通道"和"语义通道坏了"是两件事：
        前者是配置结果（不该有 ERROR 日志），后者是故障。混在一起会让真正的故障
        被"反正没配 Key"的噪声盖住。

        为什么建失败是降级而不是抛：embedding 是外部服务，可用性问题和 LLM 综合失败同类，
        不该让整个应用起不来。但也绝不能静默——降级后跑出来的召回数字
        和双通道完全不是一回事，必须能从日志和轨迹里看出来。
        """
        if not getattr(self._embeddings, "semantic", False):
            return
        if self._dense_weight <= 0:
            # 配成纯 BM25 时不该去打 embedding 接口。消融实验会跑很多次，
            # 每次白算一遍全语料向量既慢又花钱。
            return
        try:
            vectors = self._embeddings.embed_documents(self._documents)
        except Exception as error:  # noqa: BLE001 —— 外部服务不可用要降级，见上
            logger.error(
                f"知识检索语义通道构建失败，本次退化为纯词法检索 | error={error}"
            )
            return

        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(self._documents):
            logger.error(
                f"embedding 返回形状异常，语义通道关闭 | shape={getattr(matrix, 'shape', None)}"
            )
            return
        self._matrix = _l2_normalize(matrix)
        self.dense_enabled = True

    # ------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------
    def score(self, query: str, candidates: list[int]) -> dict[int, HybridScores]:
        """给候选下标打分。只返回**至少在一个通道里过了噪声基线**的候选。

        `candidates` 已经过硬过滤（型号、错误码）。传空列表就返回空——
        这正是"型号定不下来就不检索"那条纪律在这一层的表现。
        """
        if not candidates or self._bm25 is None:
            return {}

        query_tokens = tokenize(query)
        # 权重为 0 表示"这个通道不参与"，名次和置信度都不参与——不是只把置信度权重清零。
        # 只清置信度权重会留下一个隐蔽的错：RRF 名次仍然算进去，于是"纯向量检索"
        # 的排序里还混着 BM25 的意见，消融实验测出来的就不是纯向量。
        use_lexical = self._bm25_weight > 0
        use_dense = self._dense_weight > 0

        lexical = (
            self._bm25.get_scores(query_tokens)
            if query_tokens and use_lexical
            else np.zeros(len(self._documents))
        )

        semantic, dense_live = self._dense_scores(query) if use_dense else (
            np.zeros(len(self._documents)), False
        )

        # 名次也要过各自通道的噪声基线，不能只在合成 confidence 时才过。
        #
        # 这是评测逼出来的第二处修正。RRF 会**奖励"在两个通道都出现"**，
        # 而 BM25 的"出现"极其廉价——任何一个词有一点重叠就有正分、就有名次。
        # 于是一个只靠通用词沾上边的小节，会因为"BM25 排第一 + 向量排第五"
        # 而击败"BM25 没排上 + 向量排第一"的正确答案：
        #   0.2/(60+1) + 0.8/(60+5) = 0.0156  >  0.8/(60+1) = 0.0131
        # 给弱通道降权解决不了这个问题（上面那组数就是权重 0.2 时算的），
        # 因为问题不在票数权重，而在**噪声本来就不该有投票权**。
        lexical_ranks = (
            _ranks({
                index: float(lexical[index])
                for index in candidates
                if float(lexical[index]) > self._noise_floor
            })
            if use_lexical
            else {}
        )
        semantic_ranks = (
            _ranks({
                index: float(semantic[index])
                for index in candidates
                # 同理，余弦要高过这个模型的基线才算"向量通道认为相关"。
                # 无关文本在 text-embedding-v4 上也有 0.31~0.42 的余弦，
                # 让它们进名次等于让全语料都有投票权。
                if float(semantic[index]) > self._baseline
            })
            if dense_live
            else {}
        )

        scored: dict[int, HybridScores] = {}
        for index in candidates:
            bm25_raw = float(lexical[index])
            dense_raw = float(semantic[index]) if dense_live else 0.0
            lexical_rank = lexical_ranks.get(index)
            semantic_rank = semantic_ranks.get(index)
            if lexical_rank is None and semantic_rank is None:
                # 两个通道都没过各自的噪声基线，这一节和查询没有确凿关系。
                continue

            # 加权 RRF：每个通道的名次票按该通道的权重计。
            #
            # 说清它的实际作用范围：**权重相等时，加权 RRF 与不加权 RRF 的名次完全一致**
            # （只差一个公共系数）。所以生产配置（0.5/0.5）并不因为这行而改变什么，
            # 它存在是为了让权重真的能表达"这个通道说话更算数"，
            # 以及让消融配置（0/1、1/0）在名次层面也干净。
            #
            # 真正把"混合反而比单通道差"修掉的不是这里，而是上面那道**名次的噪声基线**。
            # 记下当时的数字免得后人重走：等票混合 Recall@1 85.7%、仅向量 91.1%，
            # 而把 BM25 降权到 0.2 也只挽回一部分——因为问题不在票数权重，
            # 在噪声本来就不该有投票权。加上名次基线后混合回到 87.5%，与纯向量持平。
            rrf = 0.0
            if lexical_rank is not None:
                rrf += self._bm25_weight / (self._rrf_k + lexical_rank)
            if semantic_rank is not None:
                rrf += self._dense_weight / (self._rrf_k + semantic_rank)

            scored[index] = HybridScores(
                bm25=bm25_raw,
                dense=dense_raw,
                bm25_rank=lexical_rank,
                dense_rank=semantic_rank,
                rrf=rrf,
                confidence=self._confidence(bm25_raw, dense_raw if dense_live else None),
            )
        return scored

    def _confidence(self, bm25_raw: float, dense_raw: float | None) -> float:
        """把两个通道的绝对分合成一个 [0,1] 置信度。

        规则是：**只对有确凿证据的通道求加权平均**，弱信号通道不参与，
        而不是当 0 分参与平均。这一条是实测逼出来的，代价很具体：

        直接按固定权重求平均（0.5/0.5）时，"只有一个通道确信"的命中上限就是 0.5。
        而那恰恰是混合检索存在的理由——实测「洗澡水不够热」在 AquaWarm 说明书上
        BM25 = 0.00（一个词都对不上）、余弦 0.623（语义完全对上），
        固定权重把它压到 0.362，和"离线时靠通用词凑出来的错误命中 0.331"只差 9%，
        阈值根本没有可用的落点。按"有证据的通道"平均之后它是 0.725，
        而噪声因为两个通道都低于各自基线，得到的是 0。

        反过来也要防住：不能用 max 或概率或（soft-OR）。那会让**单通道的噪声**
        直接成为置信度——两个弱信号本该互相否证，soft-OR 却把它们相加。
        所以关键不在"用哪种合成函数"，而在于**每个通道各自先有噪声基线**：
        弱信号必须先归零，再谈合成。

        这也顺带取消了"语义通道不可用时把权重收拢到词法通道"的特例：
        通道缺席和通道弱本来就该走同一条路。
        """
        strengths: list[tuple[float, float]] = []
        lexical = _lexical_strength(bm25_raw, self._noise_floor, self._saturation)
        if lexical > 0:
            strengths.append((self._bm25_weight, lexical))
        if dense_raw is not None:
            semantic = _rebase(dense_raw, self._baseline, self._strong)
            if semantic > 0:
                strengths.append((self._dense_weight, semantic))
        if not strengths:
            return 0.0
        total = sum(weight for weight, _ in strengths)
        return sum(weight * strength for weight, strength in strengths) / total

    def _dense_scores(self, query: str) -> tuple[np.ndarray, bool]:
        """查询向量 × 文档矩阵。单次查询失败只影响这一次，不关掉整个通道。"""
        if not self.dense_enabled or self._matrix is None:
            return np.zeros(len(self._documents)), False
        try:
            vector = np.asarray(self._embeddings.embed_query(query), dtype=np.float64)
        except Exception as error:  # noqa: BLE001 —— 单次调用失败退化为纯词法
            logger.warning(f"查询向量计算失败，本次退化为纯词法检索 | error={error}")
            return np.zeros(len(self._documents)), False
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return np.zeros(len(self._documents)), False
        return self._matrix @ (vector / norm), True


# ============================================================
# 归一化与名次
# ============================================================


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 零向量（空文本）除以 1，保持为零向量而不是变 NaN。
    norms[norms == 0] = 1.0
    return matrix / norms


def _lexical_strength(score: float, noise_floor: float, saturation: float) -> float:
    """把无上界的 BM25 分压到 [0,1)，且保留量纲（不是名次）。

    低于噪声基线的一律记 0：实测通用词噪声命中 BM25 ≤ 2.96、真实命中 ≥ 4.26，
    基线 3.5 落在缝里。折算成 0.27 那样的小正数会让"两个弱信号"在合成时凑出
    一个中间值，而它们本该互相否证——详见 `HybridRetriever._confidence`。
    """
    if score <= noise_floor:
        return 0.0
    return (score - noise_floor) / (score - noise_floor + saturation)


def _rebase(similarity: float, baseline: float, strong: float) -> float:
    """把余弦从"模型自己的区间"拉到 [0,1]：减掉基线，再按强命中刻度归一。

    不做这一步的后果：真实 embedding 上两段无关中文余弦也有 0.31~0.42
    （向量空间各向异性），于是任何查询对任何小节都有三四成的"置信度"，
    绝对阈值就再也分不开"有点像"和"没关系"。

    只减基线不除强命中也不够：`(0.60-0.42)/(1-0.42)` 只有 0.31，
    而 0.60 在实测里已经是一个确凿的相关命中了。两个刻度都要用上，
    真命中才能落到 0.65~1.0 这一档，和噪声的 0 拉开距离。
    """
    span = strong - baseline
    if span <= 0:
        return 0.0
    return float(min(1.0, max(0.0, (similarity - baseline) / span)))


def _ranks(scores: dict[int, float]) -> dict[int, int]:
    """按分数降序给出 1 起的名次。

    传进来的 `scores` 已经由调用方过掉各自通道的噪声基线——"能进这个字典"
    就等于"这个通道认为它相关"。这一点是 RRF 的前提：
    RRF 奖励在多个通道都出现，所以"出现"必须意味着有证据，不能是"分数大于 0"。

    排序键带上下标，保证同分时名次稳定——否则 RRF 分会随字典顺序抖动。
    """
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {index: rank for rank, (index, _) in enumerate(ranked, start=1)}
