"""说明书 RAG：实体消解、自证核对、引用拼接、拒答纪律。

这些用例验证的是**边界与副作用**，不是返回文案：解析到哪台设备、读的是哪份型号的
说明书、核对读的是设备真实状态还是模型说法、拿不准时会不会闭嘴。
"""

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.errors import GraphInterrupt

from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.knowledge import KnowledgeBase, build_knowledge_rag_subgraph, resolve_device
from src.knowledge.base import _searchable_text
from src.knowledge.embeddings import NullEmbeddings
from src.knowledge.rag import (  # 私有 _synthesize：GraphInterrupt 透传要在函数级验证
    DEFAULT_MIN_SCORE,
    DEFAULT_RELATIVE_FLOOR,
    _synthesize,
)
from src.knowledge.selfcheck import KNOWN_CHECK_IDS, CheckContext, run_self_check
from src.knowledge.tokenizer import extract_codes, tokenize
from src.models import ACMode

KNOWLEDGE_PATH = "docs/knowledge"


def _steps(result: dict, name: str) -> list[dict]:
    return [item for item in result.get("trajectory", []) if item["step"] == name]


class EchoLLM:
    """把 prompt 原样记下来，返回一段固定散文。"""

    def __init__(self, reply: str = "建议先断电 5 分钟再上电，仍报错请联系售后。"):
        self.reply = reply
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.reply)


class RaisingLLM:
    def __init__(self, error: BaseException):
        self.error = error

    def invoke(self, prompt):
        raise self.error


class DeviceResolutionTests(unittest.TestCase):
    def setUp(self):
        self.registry = DeviceRegistry(SimulatorBackend())

    def test_explicit_device_name_beats_trusted_room_context(self):
        """用户明确说了哪台，就不能被 App 当前所在房间覆盖掉。"""
        resolution = resolve_device(
            "卧室空调显示 E4", self.registry, active_room_id="living_room"
        )
        self.assertEqual(resolution.status, "resolved")
        self.assertEqual(resolution.device_id, "bedroom_ac")
        self.assertEqual(resolution.model, "FrostLine-AC310")

    def test_type_keyword_with_two_models_is_ambiguous(self):
        """两台空调型号不同、说明书不通用，只说"空调"必须反问而不是挑一台。"""
        resolution = resolve_device("空调显示 E4 是什么意思", self.registry)
        self.assertEqual(resolution.status, "ambiguous")
        self.assertEqual(resolution.candidates, ("卧室空调", "客厅空调"))
        self.assertIsNone(resolution.model)

    def test_trusted_room_disambiguates_type_keyword(self):
        resolution = resolve_device(
            "空调显示 E4 是什么意思", self.registry, active_room_id="bedroom"
        )
        self.assertEqual(resolution.status, "resolved")
        self.assertEqual(resolution.device_id, "bedroom_ac")

    def test_room_context_is_ignored_when_it_matches_nothing(self):
        """房间过滤后剩零台时不能采纳过滤结果，否则会把正确候选删光。"""
        resolution = resolve_device(
            "空调显示 E4", self.registry, active_room_id="kitchen"
        )
        self.assertEqual(resolution.status, "ambiguous")

    def test_device_without_registered_model_is_reported_as_no_model(self):
        resolution = resolve_device("客厅灯一直闪怎么办", self.registry)
        self.assertEqual(resolution.status, "no_model")
        self.assertEqual(resolution.device_id, "living_room_light")
        self.assertIsNone(resolution.model)

    def test_unrelated_question_resolves_to_unknown(self):
        resolution = resolve_device("外面下雨了吗", self.registry)
        self.assertEqual(resolution.status, "unknown")
        self.assertIsNone(resolution.device_id)

    def test_type_keyword_ranking_prefers_sentence_subject(self):
        """"空调设置温度"同时命中空调和温湿度传感器，靠"出现更早"择一。

        判错的话会解析到卧室温湿度传感器（没登记型号）→ no_model，和这里期望的
        resolved 明显可分，不会是个模棱两可的失败。
        """
        resolution = resolve_device(
            "空调设置温度多少合适", self.registry, active_room_id="bedroom"
        )
        self.assertEqual(resolution.status, "resolved")
        self.assertEqual(resolution.device_id, "bedroom_ac")

    def test_longer_type_keyword_wins_over_shorter_one(self):
        resolution = resolve_device("温湿度传感器读数准吗", self.registry, active_room_id="bedroom")
        self.assertEqual(resolution.device_id, "bedroom_th_sensor")


class ChecklistParsingTests(unittest.TestCase):
    def setUp(self):
        self.knowledge = KnowledgeBase(KNOWLEDGE_PATH)

    def test_annotations_never_reach_content(self):
        for chunk in self.knowledge.chunks:
            self.assertNotIn("<!--", chunk.content)
            self.assertNotIn("check:", chunk.content)

    def test_check_ids_are_not_searchable_terms(self):
        """标注被剥离前若进了词表，搜 "target" 就能搜出空调说明书。"""
        self.assertEqual(self.knowledge.search("ac_target_temp_below_room", model=None), [])

    def test_checklist_splits_auto_and_manual_by_annotation(self):
        section = next(
            chunk for chunk in self.knowledge.chunks if chunk.section == "制冷效果不佳"
        )
        auto = [item for item in section.checklist if item.check_id]
        manual = [item for item in section.checklist if item.check_id is None]
        self.assertEqual(len(auto), 3)
        self.assertEqual(len(manual), 2)
        # 条目文本要跟说明书逐字一致（含序号），用户才能对上号。
        self.assertTrue(section.checklist[0].text.startswith("1. "))

    def test_prose_section_has_empty_checklist(self):
        """故障码说明是散文，不能被猜成清单。"""
        section = next(chunk for chunk in self.knowledge.chunks if chunk.section == "E3")
        self.assertEqual(section.checklist, [])

    def test_unknown_check_id_fails_at_construction(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        (root / "catalog.json").write_text(
            json.dumps({"documents": [
                {"id": "x", "title": "X", "model": "X-1", "file": "x.md"}
            ]}),
            encoding="utf-8",
        )
        (root / "x.md").write_text(
            "# X\n\n## 排查\n\n1. 确认某件事。<!--check:no_such_check-->\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as ctx:
            KnowledgeBase(root)
        message = str(ctx.exception)
        self.assertIn("no_such_check", message)
        self.assertIn("x.md", message)


class SelfCheckTests(unittest.TestCase):
    def setUp(self):
        self.registry = DeviceRegistry(SimulatorBackend())
        self.graph = build_knowledge_rag_subgraph(
            KnowledgeBase(KNOWLEDGE_PATH), self.registry, llm=None
        )

    def test_missing_room_sensor_yields_unknown_not_ok(self):
        """读不到室温时这一项必须退回人工，绝不能算"核对通过"。"""
        device = self.registry.get("living_room_ac")
        outcome = run_self_check(
            "ac_target_temp_below_room", "确认设置温度低于当前室温。",
            CheckContext(device=device, room_temperature=None),
        )
        self.assertEqual(outcome.verdict, "unknown")

    def test_verdict_follows_real_device_state(self):
        """同一台设备、同一条检查项，结论只由实测状态决定。"""
        # 客厅实测室温 27.0°C
        self.registry.update("living_room_ac", temperature=28)
        problem = run_self_check(
            "ac_target_temp_below_room", "确认设置温度低于当前室温。",
            CheckContext(device=self.registry.get("living_room_ac"), room_temperature=27.0),
        )
        self.assertEqual(problem.verdict, "problem")
        self.assertIn("28", problem.detail)

        self.registry.update("living_room_ac", temperature=24)
        ok = run_self_check(
            "ac_target_temp_below_room", "确认设置温度低于当前室温。",
            CheckContext(device=self.registry.get("living_room_ac"), room_temperature=27.0),
        )
        self.assertEqual(ok.verdict, "ok")

    def test_subclass_fields_survive_check_context(self):
        """CheckContext.device 标注成 BaseDevice，子类字段不能被 Pydantic 抹掉。

        抹掉的表现是 mode 变 None → 结论从 ok/problem 退化成 unknown，
        整个自证功能静默失效。
        """
        context = CheckContext(device=self.registry.get("living_room_ac"))
        self.assertEqual(getattr(context.device, "mode", None), ACMode.COOL)
        self.assertEqual(run_self_check("ac_mode_is_cool", "x", context).verdict, "ok")

    def test_non_ac_device_gets_unknown_instead_of_false_pass(self):
        context = CheckContext(device=self.registry.get("living_room_light"))
        self.assertEqual(run_self_check("ac_mode_is_cool", "x", context).verdict, "unknown")

    def test_subgraph_splits_checklist_and_reports_problem(self):
        result = self.graph.invoke({"query": "客厅空调开着但一点都不凉"})
        self.assertEqual(result["rag_status"], "answered")
        check = _steps(result, "self_check")[0]
        self.assertEqual((check["auto"], check["manual"]), (3, 2))
        # 默认状态下客厅空调是关机的，第一条检查项应判为异常。
        verdicts = [outcome["verdict"] for outcome in result["check_outcomes"]]
        self.assertIn("problem", verdicts)
        self.assertIn("需你确认", result["answer"])
        self.assertIn("自动核对结果", result["answer"])

    def test_self_check_does_not_advance_environment(self):
        """核对读的是稳定快照。顺手推一把环境推演会让室温随调用次数漂移。"""
        before = self.registry.get("living_room_th_sensor").temperature
        self.graph.invoke({"query": "客厅空调开着但一点都不凉"})
        self.graph.invoke({"query": "客厅空调开着但一点都不凉"})
        self.assertEqual(self.registry.get("living_room_th_sensor").temperature, before)


class ModelFilterTests(unittest.TestCase):
    def setUp(self):
        self.registry = DeviceRegistry(SimulatorBackend())
        self.graph = build_knowledge_rag_subgraph(
            KnowledgeBase(KNOWLEDGE_PATH), self.registry, llm=None
        )

    def test_same_error_code_answers_differently_per_model(self):
        """两台空调都有 E4，含义完全不同。型号过滤失效时这条会立刻发现。"""
        living = self.graph.invoke({"query": "客厅空调显示 E4 是什么意思"})
        bedroom = self.graph.invoke({"query": "卧室空调显示 E4 是什么意思"})

        self.assertEqual(living["rag_status"], "answered")
        self.assertEqual(bedroom["rag_status"], "answered")
        self.assertIn("通信", living["answer"])
        self.assertNotIn("排水泵", living["answer"])
        self.assertIn("排水泵", bedroom["answer"])
        self.assertNotIn("通信", bedroom["answer"])
        self.assertEqual(living["citations"], ["smartcool-ac2024-errors.md#E4"])
        self.assertEqual(bedroom["citations"], ["frostline-ac310-errors.md#E4"])

    def test_every_hit_belongs_to_the_resolved_model(self):
        result = self.graph.invoke({"query": "客厅空调开着但一点都不凉"})
        self.assertTrue(result["hits"])
        for hit in result["hits"]:
            self.assertEqual(hit["chunk"]["model"], "SmartCool-AC2024")

    def test_search_requires_explicit_model_argument(self):
        """model 是关键字必填参数：全库检索只能是调用方写出来的显式选择。"""
        with self.assertRaises(TypeError):
            KnowledgeBase(KNOWLEDGE_PATH).search("E4")


class RefusalTests(unittest.TestCase):
    def setUp(self):
        self.registry = DeviceRegistry(SimulatorBackend())
        self.knowledge = KnowledgeBase(KNOWLEDGE_PATH)
        self.graph = build_knowledge_rag_subgraph(self.knowledge, self.registry, llm=None)

    def assert_refused(self, result: dict):
        self.assertEqual(result["rag_status"], "refused")
        self.assertEqual(result["citations"], [])
        self.assertIn("不能可靠确认", result["answer"])

    def test_ambiguous_device_refuses_before_retrieving(self):
        result = self.graph.invoke({"query": "空调显示 E4 是什么意思"})
        self.assert_refused(result)
        # 关键：没有退化成不带型号的全库检索。
        self.assertEqual(_steps(result, "retrieve"), [])
        self.assertIn("卧室空调", result["answer"])

    def test_device_without_model_refuses_before_retrieving(self):
        result = self.graph.invoke({"query": "客厅灯一直闪怎么办"})
        self.assert_refused(result)
        self.assertEqual(_steps(result, "retrieve"), [])

    def test_unresolvable_device_refuses_before_retrieving(self):
        result = self.graph.invoke({"query": "外面下雨了吗"})
        self.assert_refused(result)
        self.assertEqual(_steps(result, "retrieve"), [])

    def test_unsupported_error_code_refuses_without_rewriting(self):
        """错误码是精确键：语料里没有 E9，换任何说法也变不出来，重写只会引到错的小节。"""
        result = self.graph.invoke({"query": "客厅空调显示 E9 是什么意思"})
        self.assert_refused(result)
        self.assertEqual(_steps(result, "rewrite"), [])

    def test_refuses_when_this_model_manual_lacks_the_symptom_section(self):
        """FrostLine 说明书没有噪音章节。

        重写成"噪音异常 噪音"后会靠"异常"这个通用词以 0.25 分命中 E4（排水异常）——
        用原句那档阈值就会把一段讲排水的内容当成噪音问题的答案递出去。
        """
        result = self.graph.invoke({"query": "卧室空调有点响是怎么了"})
        self.assert_refused(result)
        retrieves = _steps(result, "retrieve")
        self.assertEqual(len(retrieves), 2)
        self.assertGreater(retrieves[1]["score_floor"], retrieves[0]["score_floor"])


class QueryRewriteTests(unittest.TestCase):
    def setUp(self):
        self.registry = DeviceRegistry(SimulatorBackend())
        self.knowledge = KnowledgeBase(KNOWLEDGE_PATH)

    def build(self, llm=None):
        return build_knowledge_rag_subgraph(self.knowledge, self.registry, llm=llm)

    def test_rewrite_replaces_rather_than_appends(self):
        """分母是查询词总数，留着原话会让永远匹配不上的口语 bigram 一直稀释分数。"""
        result = self.build().invoke({"query": "客厅空调开着但一点都不凉"})
        rewrite = _steps(result, "rewrite")[0]
        self.assertEqual(rewrite["query"], "制冷效果不佳 制冷 室温")
        self.assertNotIn("一点都不凉", rewrite["query"])
        retrieves = _steps(result, "retrieve")
        self.assertGreater(retrieves[1]["top_score"], retrieves[0]["top_score"] * 3)

    def test_rewrite_maps_symptom_to_another_models_error_section(self):
        """口语"漏水"要能跨到说明书的"排水"用词上。

        断言的是不变量而不是引用列表逐字相等：013 把语料从 5 份扩到 39 份之后，
        FrostLine 多了「排水不畅」和「冷凝水管检查」两节，同一个查询合理地召回三条。
        真正该钉住的是两件事——跨接落在讲排水泵的 E4 上（词表桥接生效），
        以及所有引用都属于这一个型号（型号过滤没被融合层绕过）。
        """
        result = self.build().invoke({"query": "卧室空调好像在漏水"})
        self.assertEqual(result["rag_status"], "answered")
        self.assertEqual(result["citations"][0], "frostline-ac310-errors.md#E4")
        for citation in result["citations"]:
            self.assertTrue(
                citation.startswith("frostline-ac310-"),
                f"型号过滤失效，混进了别的型号：{citation}",
            )

    def test_strong_first_pass_is_not_rewritten(self):
        """首轮已经查到确凿答案时不该再重写——重写会多跑一轮，还可能改差。

        013 换了查询：原来用的是"客厅空调吹出来一股霉味"，它在旧的覆盖率打分下
        是 0.20、稳稳高于当时的 0.15 下限；换成混合检索后同一句在**纯 BM25**
        配置下只有 0.278，低于实测标定出来的 0.35，于是会走重写（重写后仍答对）。
        这里改用说明书原词的查询：它在两种配置下都是确凿的强命中
        （纯 BM25 0.626 / 混合 0.813），测的仍然是同一条不变量。
        """
        result = self.build().invoke({"query": "客厅空调制冷效果不佳"})
        self.assertEqual(_steps(result, "rewrite"), [])
        self.assertEqual(result["citations"][0], "smartcool-ac2024-symptoms.md#制冷效果不佳")

    def test_llm_rewrite_must_name_an_existing_section(self):
        """模型编出一个不存在的小节名时退回词表，而不是拿去检索。"""
        llm = EchoLLM(reply="压缩机启动失败")
        result = self.build(llm).invoke({"query": "客厅空调开着但一点都不凉"})
        rewrite = _steps(result, "rewrite")[0]
        self.assertEqual(rewrite["source"], "lexicon")
        self.assertEqual(rewrite["query"], "制冷效果不佳 制冷 室温")

    def test_llm_rewrite_is_grounded_in_real_section_titles(self):
        llm = EchoLLM(reply="制冷效果不佳")
        result = self.build(llm).invoke({"query": "客厅空调开着但一点都不凉"})
        rewrite = _steps(result, "rewrite")[0]
        self.assertEqual(rewrite["source"], "llm")
        # prompt 里必须给出该型号真实存在的小节清单，而不是让模型自由发挥。
        self.assertIn("制冷效果不佳", llm.prompts[0])
        self.assertIn("噪音异常", llm.prompts[0])
        self.assertNotIn("排水", llm.prompts[0])

    def test_llm_saying_no_section_matches_leads_to_refusal(self):
        """模型明确说"都不符合"时必须拒答，不能退回词表凑一个。

        端到端实测抓到的漏洞：原来的重写 prompt 是强制单选，正确答案不在清单里时
        模型必然挑一个最像的，而校验只查"标题真实存在"——存在不等于相关。
        于是"卧室空调有点响"（FrostLine 没有噪音章节）会拿到「蒸发器结霜」的内容
        并挂上三条引用。模型自己在答案里都写了"说明书未提及异响"，
        系统却仍以权威格式把引用递出去了。

        这个漏洞 012 就存在，只是当时所有用例都传 llm=None（走词表），从没暴露。
        """
        llm = EchoLLM(reply="无")
        result = self.build(llm).invoke({"query": "卧室空调有点响是怎么了"})

        rewrite = _steps(result, "rewrite")[0]
        self.assertEqual(rewrite["source"], "llm-no-match")
        # 保持原句去查一次然后拒答，而不是拿词表再凑一版。
        self.assertEqual(rewrite["query"], "卧室空调有点响是怎么了")
        self.assertEqual(result["rag_status"], "refused")
        self.assertEqual(result["citations"], [])

    def test_max_rewrites_bounds_the_loop(self):
        result = self.build().invoke({"query": "卧室空调有点响是怎么了"})
        self.assertEqual(len(_steps(result, "rewrite")), 1)


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.registry = DeviceRegistry(SimulatorBackend())
        self.knowledge = KnowledgeBase(KNOWLEDGE_PATH)

    def build(self, llm):
        return build_knowledge_rag_subgraph(self.knowledge, self.registry, llm=llm)

    def test_citations_and_check_results_are_assembled_by_code(self):
        """模型只写散文，引用块和核对块由代码拼——引用是结构保证，不是提示词请求。"""
        llm = EchoLLM(reply="先确认开机，再看设定温度。")
        result = self.build(llm).invoke({"query": "客厅空调开着但一点都不凉"})
        self.assertEqual(result["rag_status"], "answered")
        self.assertIn("先确认开机，再看设定温度。", result["answer"])
        self.assertIn("来源：\n- smartcool-ac2024-symptoms.md#制冷效果不佳", result["answer"])
        self.assertIn("自动核对结果", result["answer"])
        self.assertTrue(_steps(result, "answer")[0]["synthesized"])

    def test_answer_degrades_to_excerpts_when_llm_fails(self):
        result = self.build(RaisingLLM(RuntimeError("rate limited"))).invoke(
            {"query": "客厅空调显示 E3 是什么意思"}
        )
        self.assertEqual(result["rag_status"], "answered")
        self.assertFalse(_steps(result, "answer")[0]["synthesized"])
        # 信息量下降，但引用与原文一个不少。
        self.assertIn("温度传感器", result["answer"])
        self.assertEqual(result["citations"], ["smartcool-ac2024-errors.md#E3"])

    def test_graph_interrupt_is_not_swallowed_by_synthesis(self):
        """人工审批靠抛 GraphInterrupt 实现，无条件 except 会破坏审批语义。"""
        hits = [{
            "chunk": {"title": "T", "section": "S", "content": "C",
                      "source": "t.md", "model": "M", "document_id": "d", "checklist": []},
            "score": 1.0,
        }]
        with self.assertRaises(GraphInterrupt):
            _synthesize(
                RaisingLLM(GraphInterrupt(())),
                query="q", device_name="客厅空调", device_model="M",
                hits=hits, outcomes=[],
            )


class StubEmbeddings:
    """按"文本里出现了哪个主题词"返回设计好的单位向量。

    向量通道必须能被**确定性**地测：用真实 embedding 就要 API Key，而且换一版模型
    断言全变（`docs/guide/13` 把这一项列为上向量检索最贵的代价）。
    这个 stub 不是在模拟语义，而是用来**构造出想验证的那种局面**——
    比如"向量通道把某一节判为最相似，但那一节不含查询里的错误码"，
    用来证明硬过滤真的在两个通道之前生效，而不是融合公式里的一个权重。

    `query_vectors` 让测试能直接指定某个查询的向量，用来造出
    "向量通道有意见、词法通道一个字都对不上"这种局面（那正是混合检索存在的理由）。
    """

    semantic = True
    name = "stub"
    baseline_similarity = 0.0
    strong_similarity = 1.0

    def __init__(self, topics: list[str], query_vectors: dict[str, list[float]] | None = None):
        self.topics = topics
        self.dimension = len(topics)
        self._query_vectors = query_vectors or {}

    def _vector(self, text: str) -> list[float]:
        return [1.0 if topic in text else 0.0 for topic in self.topics]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._query_vectors.get(text) or self._vector(text)


class BrokenEmbeddings(StubEmbeddings):
    """构建索引就抛。用来验证 embedding 服务不可用时是降级而不是整个应用起不来。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service down")


class QueryOnlyBrokenEmbeddings(StubEmbeddings):
    """文档向量算得出来，单次查询向量算不出来。降级范围应该只有这一次查询。"""

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("rate limited")


class HybridRetrievalTests(unittest.TestCase):
    """混合检索的机制：硬过滤的位置、两套分数的分工、降级的可见性。

    这些用例跑**真实语料**而不是合成小语料，是踩过一次才定的：BM25 的分数依赖语料
    统计量，噪声基线（3.5）是按这份 124 个 chunk 的语料标定的。
    三个 chunk 的合成语料里所有 BM25 分数都远低于该基线，于是什么都进不了名次——
    测出来的是"基线没标定"，不是被测的机制。
    """

    def setUp(self):
        self.registry = DeviceRegistry(SimulatorBackend())

    def test_error_code_filter_is_not_bypassed_by_the_vector_channel(self):
        """错误码是精确键，向量通道再怎么"觉得像"也不能把它绕过去。

        这里刻意让向量通道站在错误的一边：stub 把含"排水"的小节全判为与查询最相似，
        而 FrostLine 讲排水的除了 E4 还有「排水不畅」「冷凝水管检查」。
        查询点名了 E4，那两节就必须一个都不出现——若把错误码从硬过滤降级成
        相似度的一部分，用户问 E4 会拿到讲排水维护的内容，而且读起来完全通顺。
        """
        knowledge = KnowledgeBase(KNOWLEDGE_PATH, embeddings=StubEmbeddings(["排水"]))
        hits = knowledge.search("卧室空调 E4 排水", model="FrostLine-AC310", top_k=10)

        self.assertTrue(hits)
        for hit in hits:
            self.assertIn("e4", extract_codes(_searchable_text(hit.chunk)))
        sections = {hit.chunk.section for hit in hits}
        self.assertNotIn("排水不畅", sections)
        self.assertNotIn("冷凝水管检查", sections)

    def test_model_filter_is_not_bypassed_by_the_vector_channel(self):
        """型号过滤同理：向量通道看不到别的型号，因为候选集在它之前就定了。"""
        knowledge = KnowledgeBase(KNOWLEDGE_PATH, embeddings=StubEmbeddings(["排水", "通信"]))
        hits = knowledge.search("排水", model="FrostLine-AC310", top_k=10)

        self.assertTrue(hits)
        for hit in hits:
            self.assertEqual(hit.chunk.model, "FrostLine-AC310")

    def test_rank_one_does_not_imply_admissible(self):
        """RRF 决定名次，confidence 决定放不放行——两套分数不能混。

        RRF 是 `Σ w/(k+rank)`，名次第一的必然拿到该通道的最高分，**跟像不像无关**。
        所以一个语料里没有对应章节的查询，它的第一名照样有满分 RRF。
        若拿 RRF 去套三档下限，拒答分支永远走不到。
        """
        knowledge = KnowledgeBase(KNOWLEDGE_PATH, embeddings=NullEmbeddings())
        hits = knowledge.search("卧室窗帘关不严", model="QuietTrack-C60", top_k=3)

        self.assertTrue(hits, "应当有候选进入名次，否则这条测不到想测的东西")
        top = hits[0]
        self.assertGreater(top.signals["rrf"], 0.0, "它是名次第一，RRF 必然为正")
        self.assertLess(
            top.score, DEFAULT_MIN_SCORE,
            "名次第一但证据不足，绝对置信度必须低于准入下限",
        )

    def test_semantic_only_hit_is_not_halved_by_the_absent_channel(self):
        """只有一个通道确信时，置信度取"有证据的通道"的加权平均，不是固定权重平均。

        固定 0.5/0.5 平均会让单通道命中上限变成 0.5，而那恰恰是混合检索存在的理由：
        实测「洗澡水不够热」在 AquaWarm 说明书上 BM25 = 0（一个词都对不上）、
        余弦 0.62（语义完全对上）。压到 0.31 就和噪声挤在一起，阈值没有落点。
        """
        query = "完全对不上的说法"
        # 让向量通道单独指向讲排水的小节，词法通道对这句话一个词都匹配不上。
        knowledge = KnowledgeBase(
            KNOWLEDGE_PATH,
            embeddings=StubEmbeddings(["排水"], query_vectors={query: [1.0]}),
        )
        hits = knowledge.search(query, model="FrostLine-AC310", top_k=3)

        self.assertTrue(hits, "向量通道应当独立召回")
        top = hits[0]
        self.assertIsNone(top.signals["bm25_rank"], "词法通道不该有意见")
        self.assertGreater(
            top.score, DEFAULT_MIN_SCORE,
            "纯语义命中不能因为另一个通道缺席就被腰斩到进不了门",
        )

    def test_below_noise_floor_lexical_hit_gets_no_rank(self):
        """通用词凑出来的弱 BM25 分不该有投票权。

        RRF 奖励"在两个通道都出现"，所以"出现"必须意味着有证据。
        若弱信号也能进名次，一个只靠通用词沾边的小节会因为
        "BM25 排第一 + 向量排第五"击败"BM25 没排上 + 向量排第一"的正确答案。
        """
        query = "完全对不上的说法"
        knowledge = KnowledgeBase(
            KNOWLEDGE_PATH,
            embeddings=StubEmbeddings(["排水"], query_vectors={query: [1.0]}),
        )
        hits = knowledge.search(query, model="FrostLine-AC310", top_k=5)

        self.assertTrue(hits)
        for hit in hits:
            self.assertIsNone(
                hit.signals["bm25_rank"],
                f"BM25 原始分 {hit.signals['bm25']} 低于噪声基线，不该有名次",
            )

    def test_missing_semantic_channel_is_visible_in_the_trajectory(self):
        """降级必须能从轨迹里看出来，不能只写在日志里。

        否则"召回突然变差"只能靠翻日志才知道是 embedding 没接上，
        而不是检索逻辑退化——两者的排查方向完全不同。
        """
        knowledge = KnowledgeBase(KNOWLEDGE_PATH, embeddings=NullEmbeddings())
        self.assertFalse(knowledge.dense_enabled)

        graph = build_knowledge_rag_subgraph(knowledge, self.registry, llm=None)
        result = graph.invoke({"query": "客厅空调显示 E3 是什么意思"})
        retrieves = _steps(result, "retrieve")
        self.assertTrue(retrieves)
        self.assertFalse(retrieves[0]["dense"])

    def test_broken_embedding_service_degrades_instead_of_crashing(self):
        """embedding 服务挂了要退化成纯词法检索，不是让整个应用起不来。

        它和"语料引用了未声明的 check id"是两类问题：后者是代码与语料不一致，
        必须构造期失败；前者是外部服务可用性，和 LLM 综合失败同类，应当降级。
        """
        knowledge = KnowledgeBase(KNOWLEDGE_PATH, embeddings=BrokenEmbeddings(["排水"]))
        self.assertFalse(knowledge.dense_enabled)

        hits = knowledge.search("客厅空调显示 E3 是什么意思", model="SmartCool-AC2024", top_k=3)
        self.assertTrue(hits, "词法通道应当照常工作")
        for hit in hits:
            self.assertIsNone(hit.signals["dense_rank"])

    def test_query_embedding_failure_only_affects_that_query(self):
        """单次查询向量算不出来，不该把整条通道关掉。"""
        knowledge = KnowledgeBase(
            KNOWLEDGE_PATH, embeddings=QueryOnlyBrokenEmbeddings(["排水"])
        )
        self.assertTrue(knowledge.dense_enabled, "文档向量建成功了，通道应当仍然是开的")

        hits = knowledge.search("客厅空调显示 E3 是什么意思", model="SmartCool-AC2024", top_k=3)
        self.assertTrue(hits)
        for hit in hits:
            self.assertIsNone(hit.signals["dense_rank"])

    def test_ordering_is_deterministic_across_runs(self):
        """RRF 分是 `w/(k+rank)` 的和，取值离散，同分概率比连续分数高得多。

        没有确定性 tiebreak，名次会随字典顺序抖动，测试就会随机失败。
        """
        knowledge = KnowledgeBase(KNOWLEDGE_PATH, embeddings=StubEmbeddings(["排水", "通信"]))
        first = [h.chunk.section for h in knowledge.search("排水", model="FrostLine-AC310", top_k=5)]
        second = [h.chunk.section for h in knowledge.search("排水", model="FrostLine-AC310", top_k=5)]
        self.assertEqual(first, second)

    def test_catalog_pointing_at_a_missing_file_fails_at_construction(self):
        """catalog 登记了却不存在的文件必须构造期报错。

        012 时代这里是静默 `continue`：那份说明书永远检索不到，但什么都不报，
        表现是"我明明加了说明书，它却说查不到"——最难定位的那类缺陷。
        """
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        (root / "catalog.json").write_text(
            json.dumps({"documents": [
                {"id": "ghost", "title": "Ghost", "model": "G-1", "file": "ghost.md"}
            ]}),
            encoding="utf-8",
        )
        with self.assertRaises(FileNotFoundError) as ctx:
            KnowledgeBase(root)
        self.assertIn("ghost.md", str(ctx.exception))


class TokenizerTests(unittest.TestCase):
    """分词层：错误码整体保留、不靠空格断词、不造幻影词。"""

    def test_error_code_survives_tokenization(self):
        """"E3" 若被切成「e」和「3」，错误码在检索里就彻底消失了。"""
        self.assertIn("e3", tokenize("客厅空调显示 E3 是什么意思"))
        self.assertIn("1002", tokenize("电视报 1002"))

    def test_error_code_is_recognised_without_a_space(self):
        """`\\b` 在中文里不成立：Unicode 下汉字也算 `\\w`，"示E3是"里没有词边界。

        旧实现只在用户恰好打了空格时才认得出错误码，不打空格就静默走成普通口语查询，
        于是三档下限里"精确键那一档"根本没生效。
        """
        self.assertEqual(extract_codes("客厅空调显示E3是什么意思"), {"e3"})
        self.assertEqual(extract_codes("客厅空调显示 E3 是什么意思"), {"e3"})

    def test_model_strings_are_not_mistaken_for_error_codes(self):
        """型号里的数字段不能被当成错误码，否则型号一出现就会触发精确键过滤。"""
        self.assertEqual(extract_codes("SmartCool-AC2024 怎么保养"), set())
        self.assertEqual(extract_codes("FrostLine AC310 说明"), set())

    def test_error_code_is_counted_once(self):
        """错误码被摘出来一次、又被 jieba 切出一次，TF 就凭空翻倍——那是巧合不是设计。"""
        self.assertEqual(tokenize("空调显示 E4 是什么意思").count("e4"), 1)

    def test_no_phantom_word_glued_across_a_colloquial_phrase(self):
        """jieba 的 HMM 新词发现会把"灯不太亮"粘成一个词，而它在任何说明书里都不存在。

        后果是召回归零：BM25 只剩「卧室」可用，而「卧室」恰好出现在讲清洁周期的那节、
        不出现在正确的「亮度明显偏低」里。
        """
        tokens = tokenize("卧室灯不太亮了")
        self.assertNotIn("灯不太亮", tokens)
        self.assertIn("亮", tokens)


class CorpusConsistencyTests(unittest.TestCase):
    """语料与代码之间的双向一致性。

    这些是生成式断言：新增设备或说明书时漏掉一处同步，这里失败，
    而不是让检索在运行期安静地什么都查不到。
    """

    def setUp(self):
        self.registry = DeviceRegistry(SimulatorBackend())
        self.knowledge = KnowledgeBase(KNOWLEDGE_PATH)
        self.catalog = json.loads(
            (Path(KNOWLEDGE_PATH) / "catalog.json").read_text(encoding="utf-8")
        )

    def test_every_device_model_has_a_registered_manual(self):
        """设备填了 model 却没有对应说明书，检索会一直拒答且看不出原因。"""
        documented = {chunk.model for chunk in self.knowledge.chunks}
        for device in self.registry.get_all().values():
            if not device.model:
                continue
            with self.subTest(device=device.name):
                self.assertIn(device.model, documented)

    def test_every_manual_model_belongs_to_a_real_device(self):
        """说明书的 model 与设备的 model 必须逐字相等。

        差一个字符（大小写、连字符）不会报错，只会让型号过滤全部落空——
        表现是"明明有说明书却一直拒答"。
        """
        device_models = {
            device.model for device in self.registry.get_all().values() if device.model
        }
        for document in self.catalog["documents"]:
            with self.subTest(document=document["id"]):
                self.assertIn(document["model"], device_models)

    def test_catalog_title_matches_the_markdown_h1(self):
        """catalog 的 title 会进检索词表，和文件里的 H1 不一致就有两个"文档名"。"""
        for document in self.catalog["documents"]:
            path = Path(KNOWLEDGE_PATH) / document["file"]
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            with self.subTest(document=document["id"]):
                self.assertEqual(first_line, f"# {document['title']}")

    def test_every_declared_check_is_referenced_by_the_corpus(self):
        """反向校验：声明了却没有任何语料引用的自证检查是死代码。

        `base.py` 只校验了一个方向（语料引用的 id 必须已声明）。
        另一个方向同样会出问题：加了 SELF_CHECKS 条目却忘了在语料里标注，
        那条检查永远不会执行，而"诊断能力没有变强"是不会报错的。
        """
        referenced = {
            item.check_id
            for chunk in self.knowledge.chunks
            for item in chunk.checklist
            if item.check_id
        }
        for check_id in sorted(KNOWN_CHECK_IDS):
            with self.subTest(check_id=check_id):
                self.assertIn(check_id, referenced)

    def test_at_least_one_device_deliberately_has_no_model(self):
        """no_model 拒答路径需要一台没登记型号的设备才可见。

        013 给全部设备类型补齐了说明书，如果顺手把每台设备都填上型号，
        这条路径就在演示和测试里同时消失了——留一台是刻意的。
        """
        without_model = [
            device.name for device in self.registry.get_all().values() if not device.model
        ]
        self.assertTrue(without_model, "至少要保留一台没登记型号的设备")

    def test_no_section_leaks_annotations_into_the_searchable_text(self):
        """标注既不能进给用户看的正文，也不能进检索词表。"""
        for chunk in self.knowledge.chunks:
            with self.subTest(section=f"{chunk.source}#{chunk.section}"):
                self.assertNotIn("<!--", _searchable_text(chunk))


class NewDeviceSelfCheckTests(unittest.TestCase):
    """013 给非空调设备补的自证检查。

    重点不是"判定对不对"，而是**挂错设备类型时必须是 unknown**：
    返回 ok 等于宣称核对通过，那是最危险的输出。
    """

    def setUp(self):
        self.registry = DeviceRegistry(SimulatorBackend())

    def test_checks_read_real_device_state(self):
        cases = [
            # (check_id, device_id, 期望判定)
            ("light_brightness_not_zero", "living_room_light", "ok"),
            ("curtain_not_fully_closed", "living_room_curtain", "problem"),  # 出厂位置 0
            ("tv_is_not_muted", "living_room_tv", "ok"),
            ("humidifier_tank_has_water", "living_room_humidifier", "ok"),
            ("water_heater_target_is_high_enough", "bathroom_water_heater", "ok"),
            ("kettle_target_is_boiling", "kitchen_kettle", "ok"),
            ("device_battery_not_low", "entryway_lock", "ok"),
            ("ac_fan_speed_not_lowest", "living_room_ac", "ok"),
        ]
        for check_id, device_id, expected in cases:
            with self.subTest(check=check_id):
                context = CheckContext(device=self.registry.get(device_id), room_humidity=42)
                self.assertEqual(run_self_check(check_id, "x", context).verdict, expected)

    def test_verdict_follows_state_changes(self):
        """同一条检查项，结论只由实测状态决定。"""
        self.registry.update("living_room_light", brightness=0)
        context = CheckContext(device=self.registry.get("living_room_light"))
        outcome = run_self_check("light_brightness_not_zero", "x", context)
        self.assertEqual(outcome.verdict, "problem")
        self.assertIn("0", outcome.detail)

    def test_wrong_device_type_yields_unknown_never_ok(self):
        """挂到不具备该字段的设备上时只能说"无法自动核对"。"""
        device_checks = [
            ("light_brightness_not_zero", "living_room_ac"),
            ("curtain_not_fully_closed", "living_room_ac"),
            ("tv_is_not_muted", "living_room_ac"),
            ("humidifier_tank_has_water", "living_room_ac"),
            ("water_heater_target_is_high_enough", "living_room_light"),
            ("kettle_target_is_boiling", "living_room_light"),
            ("device_battery_not_low", "living_room_ac"),
            ("ac_fan_speed_not_lowest", "living_room_light"),
            ("humidifier_target_above_room", "living_room_ac"),
        ]
        for check_id, device_id in device_checks:
            with self.subTest(check=check_id, device=device_id):
                context = CheckContext(device=self.registry.get(device_id))
                self.assertEqual(run_self_check(check_id, "x", context).verdict, "unknown")

    def test_missing_room_humidity_yields_unknown_not_ok(self):
        """房间没有湿度读数时退回人工，绝不假定通过——和室温那条同一个原则。"""
        context = CheckContext(device=self.registry.get("living_room_humidifier"))
        outcome = run_self_check("humidifier_target_above_room", "x", context)
        self.assertEqual(outcome.verdict, "unknown")


class CitationPrecisionTests(unittest.TestCase):
    """相对截断：比最佳证据弱太多的候选不该出现在引用里。

    这道截断修的是 013 自己引入的精度回退——语料扩到 124 个 chunk 后，
    同一份症状手册的**兄弟小节**会一起过绝对下限（同一台电器的不同症状语义本来就近）。
    """

    def setUp(self):
        self.registry = DeviceRegistry(SimulatorBackend())
        self.knowledge = KnowledgeBase(KNOWLEDGE_PATH, embeddings=NullEmbeddings())
        self.graph = build_knowledge_rag_subgraph(self.knowledge, self.registry, llm=None)

    def test_every_citation_is_close_to_the_best_evidence(self):
        """性质断言：留下来的每一条都不低于第一名的 relative_floor 倍。"""
        result = self.graph.invoke({"query": "客厅空调开着但一点都不凉"})
        hits = result["hits"]
        self.assertTrue(hits)
        top = hits[0]["score"]
        for hit in hits:
            self.assertGreaterEqual(hit["score"], top * DEFAULT_RELATIVE_FLOOR)

    def test_weak_sibling_sections_do_not_get_cited(self):
        """「不凉」不该把同一份手册里讲异味、讲噪音的小节也引用上。

        它们过得了 0.35 的绝对下限（实测 0.54 / 0.51），但比正确答案（0.83）弱得多。
        绝对下限管不了这件事：提到 0.55 会连正确答案一起砍掉。
        """
        result = self.graph.invoke({"query": "客厅空调开着但一点都不凉"})
        self.assertEqual(
            result["citations"], ["smartcool-ac2024-symptoms.md#制冷效果不佳"]
        )

    def test_comparable_sections_are_all_kept(self):
        """相对截断不能变成"永远只留一条"——几条证据确实相当时要都留下。

        "漏水"在 FrostLine 上同时命中讲排水泵的 E4 和讲排水不畅的症状节，
        两者都是这个问题的合理依据。
        """
        result = self.graph.invoke({"query": "卧室空调好像在漏水"})
        self.assertGreater(len(result["citations"]), 1)
        for citation in result["citations"]:
            self.assertTrue(citation.startswith("frostline-ac310-"))

    def test_relative_floor_never_turns_an_answer_into_a_refusal(self):
        """第一名永远保留：这道截断只让引用更干净，不承担拒答职责。"""
        graph = build_knowledge_rag_subgraph(
            self.knowledge, self.registry, llm=None, relative_floor=1.0
        )
        result = graph.invoke({"query": "客厅空调制冷效果不佳"})
        self.assertEqual(result["rag_status"], "answered")
        self.assertEqual(len(result["citations"]), 1)


if __name__ == "__main__":
    unittest.main()
