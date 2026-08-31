"""说明书检索的向量通道：可插拔的 embedding provider。

为什么要做成协议 + 显式的"没有语义通道"
------------------------------------
混合检索的另一半是语义相似度，它**真的需要一个模型**。而这个项目有两条同时成立
的既有承诺：

1. 生产上要能做语义泛化——"不凉"和"制冷效果不佳"一个字都不重叠，
   词法通道永远召不回来，这正是 012 挂在"已知边界"里的那笔债。
2. 测试不需要 API Key，而且必须是确定性的。换一版 embedding 就让断言全变，
   是 `docs/guide/13` 明确点名"最贵的一项"代价。

所以 embedding 是显式注入的依赖，`semantic` 标志决定向量通道开不开：

- `ApiEmbeddings`  —— 调远程 `/embeddings`（OpenAI 兼容），生产用。
                      结果按内容哈希落盘缓存，改文档才重算，不是每次检索都联网。
- `NullEmbeddings` —— 明确宣告"没有语义通道"。未配置模型时用，检索退化为纯 BM25，
                      并且这件事会写进日志和 RAG 轨迹，不是静默的。

一个被实测否决的设计：离线哈希向量
--------------------------------
013 最初的方案是留一个 `HashingEmbeddings` 兜底——把词特征哈希到固定维度当"向量"，
好处是离线、确定、让混合检索永远是双通道。实测把它否决了：

    查询「客厅空调开着但一点都不凉」（正确答案是「制冷效果不佳」小节）
      哈希向量给「噪音异常」余弦 0.243，排第一
      BM25                     给「噪音异常」0 分
    六组"口语 → 说明书书面语"的相关对，哈希向量有四组余弦恰好是 0.000

原因很直接：**哈希向量没有 IDF，它按"共享了几个常见词"排序。**
上面那个 0.243 就是靠一个「空调」凑出来的。而 IDF 压制通用词恰恰是 BM25 已经做好的事，
把这条通道放回来等于从旁路重新引入了 BM25 刚刚干掉的噪声——而且它绕过了 IDF，
BM25 那边的修正管不到它。

它还破坏了阈值标定：真命中与噪声命中的 confidence 曾一度只差 0.007
（0.129 vs 0.122），三档下限没有任何可用的落点。去掉它之后同一组查询变成
0.110 vs 0.000。

结论：**双通道是要靠真模型换来的，不能靠一个假的第二通道凑出来。**
宁可诚实地退化成单通道并且说出来。

真实 embedding 的两个刻度是实测的
--------------------------------
余弦有基线偏移：`text-embedding-v4` 上两段完全无关的中文也有 0.31~0.42 的余弦
（向量空间各向异性），直接当置信度用会让垃圾看起来有四成把握。
所以要减掉基线、再按"多高算强命中"拉回 [0,1]，两个刻度都属于具体模型，挂在 provider 上。
换模型必须重新测，不能沿用。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
from loguru import logger

# text-embedding-v4 上实测（见模块 docstring 与 evals/）：
#   六组"用户口语 → 说明书书面语"相关对   余弦 0.601 ~ 0.737，中位 0.687
#   六组无关对（症状问句 vs 保修/除垢章节）余弦 0.306 ~ 0.418
# 基线取无关对的**上界** 0.42：宁可把边缘的相关对压成 0，也不让无关对拿到正分——
# 下游是绝对分数守门，假阳性会直接变成"拿着别人家的章节权威作答"。
# 强命中取相关对的中位数 0.70：到这个余弦就给满分，再往上没有区分意义。
_V4_BASELINE_SIMILARITY = 0.42
_V4_STRONG_SIMILARITY = 0.70


@runtime_checkable
class EmbeddingProvider(Protocol):
    """向量提供者。

    `semantic` 为假时检索层会关掉向量通道并把权重收拢到 BM25 上，
    而不是拿一堆零向量去参与融合。
    `baseline_similarity` / `strong_similarity` 是这个模型的余弦刻度，
    检索层用它们把余弦拉成可以和阈值比较的 [0,1] 绝对分。
    """

    name: str
    dimension: int
    semantic: bool
    baseline_similarity: float
    strong_similarity: float

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class NullEmbeddings:
    """明确宣告"这套配置下没有语义通道"。

    刻意不返回零向量而是直接抛：零向量会安静地参与融合、贡献 0 分，
    看起来一切正常，只有召回质量偷偷退化。检索层应该先看 `semantic` 再决定要不要调，
    真调到这里就是检索层的 bug，应该炸出来。
    """

    name = "none"
    dimension = 0
    semantic = False
    baseline_similarity = 0.0
    strong_similarity = 1.0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("NullEmbeddings 没有语义通道，调用方应先检查 provider.semantic")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("NullEmbeddings 没有语义通道，调用方应先检查 provider.semantic")


class ApiEmbeddings:
    """OpenAI 兼容的远程 embedding 接口，带内容哈希磁盘缓存。

    缓存键是 `sha256(文本)`，缓存文件按模型名分开存：换模型不会读到上一版模型算出来的
    向量（那种串味不会报错，只会让相似度整体失真）。改一个字的说明书只重算那一节。

    调用失败**不在这里兜底**。这个类只负责"要么给出向量，要么抛"；
    降级成纯词法检索是检索层的决定，而且必须留下日志和轨迹痕迹——
    在这里静默返回零向量会让"语义通道其实没在工作"完全不可见。
    """

    semantic = True

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        dimension: int = 1024,
        baseline_similarity: float = _V4_BASELINE_SIMILARITY,
        strong_similarity: float = _V4_STRONG_SIMILARITY,
        cache_path: str | Path | None = "data/embeddings",
        timeout: float = 30.0,
        batch_size: int = 10,
    ) -> None:
        self.name = model
        self.dimension = dimension
        self.baseline_similarity = baseline_similarity
        self.strong_similarity = strong_similarity
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._batch_size = batch_size

        self._cache: dict[str, list[float]] = {}
        self._cache_file: Path | None = None
        if cache_path:
            safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in model)
            self._cache_file = Path(cache_path) / f"{safe}.json"
            self._load_cache()

    # ---- 缓存 ----
    def _load_cache(self) -> None:
        if self._cache_file is None or not self._cache_file.exists():
            return
        try:
            payload = json.loads(self._cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(f"embedding 缓存读取失败，本轮重新计算 | error={error}")
            return
        # 维度对不上就整份丢掉：拼接不同维度的向量会在矩阵乘法处炸得莫名其妙。
        if payload.get("dimension") != self.dimension:
            return
        self._cache = {key: list(value) for key, value in payload.get("vectors", {}).items()}

    def _save_cache(self) -> None:
        if self._cache_file is None:
            return
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(
                json.dumps(
                    {"model": self._model, "dimension": self.dimension, "vectors": self._cache},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as error:
            # 缓存写不进去不该让检索失败，下次重算就是了。
            logger.warning(f"embedding 缓存写入失败 | error={error}")

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ---- 远程调用 ----
    def _request(self, batch: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._model, "input": batch},
            timeout=self._timeout,
        )
        if response.status_code != 200:
            # 不用 raise_for_status()：它只给出 "400 Bad Request"，把响应体丢掉了。
            # 而供应商的真实原因全在响应体里——batch size 超限、模型名拼错、
            # 单条文本过长，全都是 400，靠状态码分不开。踩过一次：
            # 批量默认值 24 触发阿里云 "batch size should not be larger than 10"，
            # 日志里只有一句 400，只能靠二分批量大小才定位到。
            raise RuntimeError(
                f"embedding 接口返回 {response.status_code}："
                f"{response.text[:300]}（batch={len(batch)} model={self._model}）"
            )
        payload = response.json()
        # 接口不保证按输入顺序返回，按 index 排回来。顺序错乱不会报错，
        # 只会让每个小节挂上别人的向量——最难发现的那类失真。
        rows = sorted(payload["data"], key=lambda item: item["index"])
        return [list(row["embedding"]) for row in rows]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if self._key(text) not in self._cache]
        # 同一批里可能有重复文本，去重后再发请求。
        unique: list[str] = list(dict.fromkeys(missing))
        for start in range(0, len(unique), self._batch_size):
            batch = unique[start:start + self._batch_size]
            for text, vector in zip(batch, self._request(batch)):
                self._cache[self._key(text)] = vector
        if unique:
            self._save_cache()
        return [self._cache[self._key(text)] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        key = self._key(text)
        if key not in self._cache:
            self._cache[key] = self._request([text])[0]
            self._save_cache()
        return self._cache[key]


def build_embeddings(
    *,
    model_id: str = "",
    base_url: str = "",
    api_key: str = "",
    dimension: int = 1024,
    cache_path: str = "data/embeddings",
) -> EmbeddingProvider:
    """按配置选一个 provider：型号、地址、Key 都齐就走远程，否则明确没有语义通道。

    这个选择是**显式的、并且会打日志**。悄悄降级的后果是有人以为自己在跑混合检索、
    实际只有 BM25 在工作，然后拿这组数字去比较召回质量。
    """
    if model_id and base_url and api_key:
        logger.info(f"知识检索语义通道：远程 embedding | model={model_id} dim={dimension}")
        return ApiEmbeddings(
            model=model_id,
            base_url=base_url,
            api_key=api_key,
            dimension=dimension,
            cache_path=cache_path,
        )
    logger.info(
        "知识检索语义通道：未启用（缺 RAG_EMBEDDING_MODEL_ID / 地址 / Key）。"
        "检索退化为纯 BM25 词法通道——同义不同字的口语症状会召不回来，"
        "这是配置结果而不是故障。"
    )
    return NullEmbeddings()
