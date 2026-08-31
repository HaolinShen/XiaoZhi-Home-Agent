# 从 Agent 学习者视角的差距分析

> **快照**：2026-08-22 盘点，`HEAD = e7d1113`（011 工程收口已提交），工作区干净
> **测试基线**：盘点时 `190 passed, 137 subtests passed`
> **方法**：代码审查 + 实际运行复现。每条结论标注证据强度：
> - 【实测】= 写了脚本跑真实代码并观察到输出
> - 【审查】= 读代码推断，未运行验证
>
> **2026-08-23 更新**：1.1 已修复（`src/memory/summarizer.py` + 4 条回归用例），
> 修复过程与效果见 [1.1 的「修复」小节](#修复窗口起点对齐到合法边界)。
> 测试基线随之变为 `198 passed, 137 subtests passed`（新增 4 条配对回归 + 4 条
> `UsageTracer` 用例，后者来自同期修掉的 `KeyError('model')`，性质属于 2.1）。
> 其余各条尚未动工。
>
> **2026-08-30 更新**：3.4 主体已由 012 迭代（`docs/iterations/012-manual-rag-upgrade.md`）
> 修复——实体消解四态、检查项自证分流、LLM 综合与代码拼接引用，详见 3.4 的改写。
> 剩余缺口（词法检索无语义泛化、自证检查项仅 3 条且全空调）仍成立。
> 测试基线变为 `234 passed, 137 subtests passed`（新增 `tests/test_knowledge_rag.py` 36 项）。
>
> **2026-08-31 更新**：上面那两条剩余缺口**都已由 013 迭代**
> （`docs/iterations/013-hybrid-retrieval.md`）**修掉**——BM25 + 向量双通道混合检索、
> 名次（RRF）与准入（confidence）分离、jieba 分词、语料 5 → 39 份、自证检查 3 → 12 条
> 且覆盖全部 10 种设备类型。3.4 换上一组**新的**诚实边界，见该节改写。
> 同时 **2.1 已部分修复**：新增 `evals/knowledge_recall.json`（63 条 golden 用例）
> + `src/evaluation/recall.py`（runner），**RAG 召回质量**这一维第一次有了真实数字
> （Recall@1 51.8% → 87.5%）。另外三维——路由准确率、Planner 计划质量、
> 端到端任务成功率——**仍然全缺**，所以 2.1 的「学习价值最高」定位不变。
> 测试基线变为 `264 passed, 383 subtests passed`（`tests/test_knowledge_rag.py` 36 → 66 项）。

本文不是迭代方案，是一份决策前的现状盘点。定稿的改造计划另开 `docs/iterations/012-*.md`。

---

## 摘要

项目定位是「教学 + 答辩作品」（README 面向想学现代 Agent 开发的开发者，`docs/defense-script.md` 是答辩讲稿）。所以判断「值不值得补」的标准是双重的：**这个缺口会不会被面试官问到** + **补它能学到哪个 Agent 工程概念**。

**编排层已经成熟**：五条互斥路径、`planning_status` 显式状态机、`verify_step` 用真实设备状态而非模型自述做验证、身份靠 `RunnableConfig` 做到「模型不可表达」、011 的能力单一数据源。

**缺的是编排之外的三层**：模型调用层的韧性、质量的可度量性、安全边界的覆盖范围。

| # | 问题 | 证据 | 学到的概念 | 优先级 |
|---|---|---|---|---|
| 1.1 | ~~对话裁剪切断「工具调用 ↔ 工具结果」配对~~ **已修复** | 【实测】30.8% 的裁剪 → 0% | 消息协议不变量 | ~~高~~ 已收 |
| 1.2 | MCP server 旁路零审批暴露 `unlock` | 【实测】 | 边界要在每个入口重复 | 高 |
| 1.3 | 审批绑定「工具名」而非「效果」 | 【实测】 | 审批锚点的选择 | 高 |
| 1.4 | Planner 路径 unlock 降级为 medium + 模型自述文案 | 【实测】 | HITL 展示必须是机器事实 | 高 |
| 2.1 | 没有评测体系（唯一的 18 行打分器喂字面量）—— **013 已补 RAG 召回一维**（63 条 golden + runner）；路由 / Planner / 端到端三维仍缺 | 【实测】 | 非确定性系统的质量度量 | 高（学习价值最高） |
| 2.2 | LLM 调用层零韧性（3 处裸 `invoke`、结构化输出无兜底） | 【审查】 | LLM 是不可靠依赖 | 中 |
| 3.1 | 上下文预算只算历史，漏掉 system prompt 与工具 schema | 【实测】 | 预算要对完整请求体计数 | 中 |
| 3.2 | `Send` fan-out 的分支是零 IO 内存查表；全同步架构 | 【审查】 | 并行的收益来自 IO 等待 | 中 |
| 3.3 | 记忆内容与远端 MCP 工具描述直入 system prompt | 【审查】 | 不可信输入的隔离与标注 | 中 |
| 3.4 | ~~RAG 全路径零 LLM 调用（检索粘贴）~~ **012 已升级** / ~~剩余缺口：词法无语义泛化~~ **013 已修**（BM25 + 向量双通道，口语召回 3/30 → 23/30） | 【实测】 | 检索 ≠ RAG 的 G；词法 ≠ 语义 | ~~低~~ ~~已收大半~~ 已收口，剩四条诚实边界 |
| 3.5 | 零锁 + 3 处 `check_same_thread=False` | 【审查】 | 已在答辩文档自陈 | 低 |

---

## 零、已经做对的（改动时不要回退）

盘点缺口之前先钉住不该动的部分，否则后续重构容易把这些一起「优化」掉。

1. **`verify_step` 读注册中心真实状态做验证**（`src/agent/planning.py:197`）。不信模型自述成败，是 Planner 分支最有价值的设计。
2. **失败分支按 `problem_type` 分流**：`unsupported_action` / `device_not_found` 是确定性错误，跳过重试直接 replan。识别出「同样参数重放不可能成功」这一点，比无脑重试高一个层次。
3. **身份对模型不可表达**：工具用 `config: RunnableConfig` 命名参数接身份，该参数不出现在 JSON Schema 里，模型在语法上无法伪造 `home_id`。这比「校验模型传来的 home_id」强得多。
4. **RAG 的拒答纪律 + 强制引用**（`src/knowledge/rag.py`）。检索不到、型号定不下来都明确说「不能可靠确认」，而不是让模型硬答。012 之后这条进一步升级：引用与核对块由代码拼接（结构保证），消解不唯一直接拒答（不退化成全库检索）。改动时不要回退。013 上了混合检索之后拒答准确率**仍然是 100%**（63 条 golden 用例里 7 条应当拒答，全部拒对），但实测暴露了一件不直观的事：**这道闸门的可靠性来自词法通道，不来自阈值调得准**。「这份说明书没讲这件事」只有 BM25 说得出来——问「卧室空调有点响」时，「响 / 噪音」在 FrostLine 全文一个都没有，BM25 直接返回 0；而向量通道给「制冷效果不佳」的余弦照样有 0.336（正例余弦中位 0.653、困难负例中位 0.568，两者大幅重叠，绝对余弦分不开「对的症状」和「同一台电器的另一个症状」）。所以：**不许为了召回率把 `RAG_BM25_WEIGHT` 清零**。纯向量的 Recall@3（94.6% vs 89.3%）与 MRR（0.908 vs 0.881）确实比混合更好，这个数字照实摆着，但那条路会让 embedding 一挂就**全部**拒答，而混合配置下还有 BM25 兜着（离线 Recall@1 58.9%）。
5. **`dedupe_key` 用 UNIQUE 索引去重**，而不是「先查再插」，避开了 TOCTOU 竞态。
6. **确定性分支刻意不问 LLM**：`classify_intent()` 命中 `automation_management` 信号时直接返回 fallback。知道什么时候**不该**用模型，是 Agent 工程的成熟标志。
7. **`src/middleware/interceptors.py:6-21` 的自陈**：完整写清了「为什么这段 RetryInterceptor 不接入」的三条理由（无条件 `except Exception` 会吞掉 `GraphInterrupt` 破坏审批语义、叠加在 `ChatOpenAI(max_retries=2)` 上会放大无效 API Key 这类确定性错误、loguru + `emit_progress()` 已覆盖可观测性）。这种「保留代码并诚实说明为何未启用」的工程判断，比删掉它更有教学价值。

---

## 一、四个实测确认的缺陷

### 1.1 对话裁剪会切断「工具调用 ↔ 工具结果」的配对【实测 · 已修复】

> **状态**：2026-08-23 已修复。下面从「为什么会有这个坑」讲到「怎么修的、修完效果如何」，
> 缺陷分析部分完整保留 —— 这条的教学价值主要在**根因推导过程**，而不在最终那十几行补丁。

#### 背景 A：为什么这个项目必须裁剪历史

大模型是**无状态**的。它不记得上一轮说过什么，所谓「多轮对话」全靠客户端每次把**整段历史重新发一遍**来伪造记忆。历史越长，每轮请求体越大 —— 直到超出模型的上下文窗口，或者账单变得离谱。

本项目的历史还是**落盘持久化**的，这让问题更突出。`build_graph()` 编译时挂了 checkpointer：

```python
checkpointer = create_checkpointer(settings.memory.db_path)   # SqliteSaver → data/checkpoints.db
graph = workflow.compile(checkpointer=checkpointer)
```

`AgentContext.to_config()` 把 `session_id` 同时用作 LangGraph 的 `thread_id`，所以**同一个 session 的消息会一直累积在 SQLite 里**，进程重启也不清空。没有裁剪，一个长期使用的 session 会无限膨胀。

#### 背景 B：裁剪由三层配合完成

| 层 | 位置 | 职责 |
|---|---|---|
| 状态层 | `src/agent/state.py:29`<br>`messages: Annotated[list, add_messages]` | `add_messages` 是 reducer：节点返回的消息**默认追加**；返回同 id 的消息则**替换**；返回 `RemoveMessage(id=...)` 则**删除** |
| 编排层 | `src/agent/graph.py:286` `compact_context_node` | 图节点。读当前 state，算出该删哪些、摘要写什么，把结果作为状态更新返回 |
| 算法层 | `src/memory/summarizer.py` | 三个纯函数：`estimate_tokens` / `compact_messages` / `build_compaction_update`。不碰状态、不碰 LLM，可独立测试 |

关键点：**裁剪不是「这一轮少发几条」，而是真的从 checkpoint 里删数据**。因为 `add_messages` 把 `RemoveMessage` 解释成删除指令，写回 SQLite 时那几行就没了，下一轮再也读不回来。这是刻意设计（控制存储无限增长），但也意味着切错的后果是不可逆的。

#### 背景 C：裁剪节点在图里的位置 —— 它在 ReAct 循环**内部**

`graph.py` 的边定义里，`compact_context` 有**三个**入口：

```python
workflow.add_conditional_edges("task_router", ..., {"compact_context": "compact_context", ...})
workflow.add_edge("compact_context", "agent")
workflow.add_edge("tools", "compact_context")          # ← 工具执行完，回来先裁剪
workflow.add_edge("reject_tools", "compact_context")   # ← 审批被拒也一样
```

画出来是这样：

```
task_router ──→ compact_context ──→ agent ──┬──→ END（直接文本回复）
                     ↑                      └──→ tools ──┐
                     └──────────────────────────────────┘
```

也就是说：**每一次工具执行之后，都会立刻走一遍裁剪，然后才把消息发给模型**。这一点非常重要 —— 裁剪运行的时刻，正好是 `tools` 节点刚刚往历史尾部追加了新 `ToolMessage` 的时刻。裁剪和「工具调用配对」这两件事在时间上贴得最近，所以踩坑概率不是偶发的边角情况，而是主路径上的高频事件。

#### 背景 D：一次裁剪具体做了 4 件事

`compact_context_node` 把当前全部消息和已有摘要交给 `build_compaction_update`，参数来自 `settings.memory`（`src/config.py:64-67`）：

```python
context_max_messages = 12     # 最多保留 12 条
context_max_tokens   = 2400   # 保留部分的估算 token 上限
tool_result_max_chars = 1200  # 单条工具结果最长 1200 字符
summary_max_chars    = 1800   # 摘要最长 1800 字符
```

然后依次做 4 件事：

**① 先把过长的工具结果就地截断**（`summarizer.py:26-31`）

```python
messages = [_truncate_message(m, max_tool_result_chars) if isinstance(m, ToolMessage) else m
            for m in messages]
```

超过 1200 字符的 `ToolMessage` 内容被砍掉尾部并加上 `…（工具结果已裁剪）` 标记。这一步不删消息，只瘦身。

**② 算出「保留窗口」的起点 `keep_from`**（`summarizer.py:32-34`）

```python
keep_from = max(0, len(messages) - max_messages)              # 先按条数：留最后 12 条
while keep_from < len(messages) - 1 and estimate_tokens(messages[keep_from:]) > max_tokens:
    keep_from += 1                                            # 还超 token 就继续往后挪
```

两道闸门串联：先按**条数**定初始位置，再按**估算 token** 一格一格往后推，直到不超 2400。

**③ 窗口外的消息压成一行行摘要**（`summarizer.py:37-43`）

```python
summary_parts.append(f"{role}: {content[:240]}")     # 每条截到 240 字符
summary = "\n".join(summary_parts)[-max_summary_chars:]   # 整体保尾部 1800 字符
```

注意这里**没有调用 LLM** —— 所谓「摘要」就是 `role: 内容前 240 字` 的字符串拼接，再和已有摘要合并。这个摘要最终会由 `agent_node` 塞进 **system prompt**（`graph.py:676` 的 `conversation_summary={...}`），所以被裁掉的信息不是彻底消失，而是**降级**成了 system prompt 里的一段粗略文本。

**④ 生成状态更新**（`summarizer.py:64-76`）

```python
removals     = [RemoveMessage(id=m.id) for m in messages if m.id not in recent_ids]  # 删窗口外的
replacements = [被截断过内容的消息]                                                    # 同 id 覆盖，让①的截断落盘
return removals + replacements, merged_summary, estimate_tokens(recent)
```

之后 `agent_node`（`graph.py:664-691`）组装最终请求体：把 SystemMessage 插到第 0 位，后面**原样跟上 `state["messages"]`**，直接 `invoke`。

```python
messages = list(state["messages"])
messages.insert(0, SystemMessage(content=system_prompt + context_prompt + role_context))
response = active_llm.invoke(messages)
```

**整条链路里没有任何一处检查消息之间的配对关系** —— 这就是缺陷的落点。

---

以上是背景。下面是问题本身。

#### 前提：一次工具调用在历史里是**一对**消息，不是一条

Agent 每调用一次工具，消息历史里会留下两条互相绑定的消息：

| 顺序 | 消息 | 内容 | 绑定关系 |
|---|---|---|---|
| 1 | `AIMessage` | 模型说「我要调 `control_light`，参数是客厅灯 / turn_on」 | 带 `tool_calls`，每个调用有唯一 id（如 `c1`） |
| 2 | `ToolMessage` | 程序执行完回填「c1 的结果：客厅灯已打开」 | 带同一个 `tool_call_id=c1` |

可以理解成**一问一答**：第 1 条是问题（模型的请求），第 2 条是答案（执行结果）。

OpenAI 兼容协议对这个配对是**强制要求**的：`tool` 角色的消息必须紧跟在带对应 `tool_calls` 的助手消息之后。如果请求体里出现了答案却找不到问题，服务端不会宽容地忽略，而是**直接 400 拒绝整个请求**。

#### 问题：`keep_from` 是按位置数出来的，跟配对关系毫无关系

回看背景 D 的第 ② 步：`keep_from` 由**条数**和**字符长度**两个量决定，这两个量都跟「哪条消息是哪条的答案」没有任何关系。刀口落在哪纯属巧合。

举个具体例子。假设某个 session 在 SQLite 里已经存了 13 条历史，本轮用户又问了一句，`sync_context` 把新 `HumanMessage` 追加进来变成 14 条（`AI*` = 带 `tool_calls` 的助手消息）：

```
索引  消息                        说明
 0   Human   "客厅灯开一下"
 1   AI*c1   tool_calls=[c1]       ← 问题
 2   Tool c1 "客厅灯已打开"         ← 答案（绑定 c1）
 3   AI      "已经打开了"
 4   Human   "空调也开一下"
 5   AI*c2   tool_calls=[c2]
 6   Tool c2
 7   AI
 8   Human   "卧室窗帘和灯都关掉"
 9   AI*     tool_calls=[c3, c4]   ← 一条消息里两个调用
10   Tool c3
11   Tool c4
12   AI
13   Human   "现在几度"             ← 本轮新追加
```

裁剪一算：

```
keep_from = max(0, 14 - 12) = 2
recent = messages[2:]   → 窗口首条是索引 2 的 Tool c1
old    = messages[:2]   → 索引 0、1 被压成摘要，并发 RemoveMessage 从 SQLite 删除
                          其中索引 1 正是 Tool c1 的父消息 AI*c1
```

于是 `agent_node` 发出的请求体是：

```
[0] SystemMessage
[1] ToolMessage(tool_call_id=c1)   ← 孤儿：有答案，找不到问题 → 服务端 400
[2] AIMessage "已经打开了"
[3] HumanMessage "空调也开一下"
...
```

换成 13 条或 15 条，`keep_from` 就会变成 1 或 3，落在合法位置上，这轮又完全正常。**同一份代码、同一个用户，只因为历史条数的奇偶巧合而时好时坏** —— 而且第 ② 步的 token 循环还会把 `keep_from` 继续往后推，推到哪一格取决于每条消息的字符长度，更加无法预测。所以这不是能靠调 `context_max_messages` 避开的问题。

`RemoveMessage` 的存在让后果加重一层：父消息不是「这轮没带上」，而是**从 SQLite 里删掉了**，就算之后想补也补不回来。

#### 实测：三成的裁剪会切错刀口

盘点阶段最初报的数字是「53.4% 的对话会踩到」，那份语料是随机生成的一次性脚本，已不可复现。
修复时改成**穷举式可复现语料**重新测量，口径写清楚：

> 穷举「每轮 0 / 1 / 2 次工具调用」的 5 轮组合 = `3^5 = 243` 段对话（内容长度随轮次变化）；
> 对每段对话的**每个前缀长度**各跑一次裁剪 = 4455 个裁剪点。默认参数 `max_messages=12, max_tokens=2400`。
> 这份语料就是 `tests/test_phase_three.py::ToolCallPairingTests` 用的那一份。

| 口径 | 修复前 | 修复后 |
|---|---|---|
| 真正发生了裁剪的调用中，产出孤儿的比例 | **474 / 1541 = 30.8%** | **0 / 1541 = 0%** |
| 全部裁剪点（含历史还没超预算、原样返回的） | 474 / 4455 = 10.6% | 0 / 4455 = 0% |
| 会话维度：至少有一轮会踩到的对话 | **207 / 243 = 85.2%** | 0 / 243 = 0% |

第一行是最有意义的口径 —— 分母是「护栏真正动手的次数」。**每 3 次有效裁剪就有 1 次切断配对**。

再端到端跑真实 `build_graph()`（真实图、真实 checkpointer，只把 LLM 换成 FakeLLM），
按 `[1,2,0,1,1,2,0,2,1,0,1,2]` 的工具调用次数连打 12 轮，拦下每一次实际发往模型的请求体：

```
修复前（还原两处 bug）: 21 次 LLM 入参，含孤儿 6 次   ← 真实端点会各返回一次 400
      第 7次调用: body 13 条, 孤儿=['call-0-0']
      第 9次调用: body 13 条, 孤儿=['call-1-0', 'call-1-1']
      第12次调用: body 13 条, 孤儿=['call-3-0']
      第16次调用: body 13 条, 孤儿=['call-5-1']
      第19次调用: body 13 条, 孤儿=['call-7-0', 'call-7-1']
修复后（当前代码）    : 21 次 LLM 入参，含孤儿 0 次
```

> **这里本身有个教训**：第一次写端到端复现脚本时用的是「每轮固定 1 次工具调用」的
> FakeLLM，结果修复前后都是 0 孤儿，一度以为复现不了。原因是那种对话每轮恰好 4 条消息
> （Human / AI\*call / Tool / AI），历史长度永远是 4 的倍数，`keep_from` 必然落在
> `Human` 上 —— **规整的语料会系统性地掩盖这个 bug**。把每轮工具调用次数变成 0/1/2
> 混合、破坏长度周期性之后，立刻稳定复现。这也解释了为什么这个 bug 能活到现在：
> 手写测试 fixture 天然是规整的。

> **修正一处早期结论**：探索阶段曾认为这会「永久损坏该 thread，之后每轮都重发孤儿」。实测**不成立**。接着上面的例子往下走一步就清楚了：
>
> ```
> 本轮裁剪后 checkpoint 剩 12 条，首条是孤儿 Tool c1
> 下一轮 sync_context 追加新 Human → 13 条
> keep_from = max(0, 13 - 12) = 1 → 窗口从第 2 条开始 → 孤儿 Tool c1 被挤出窗口并删除
> ```
>
> 孤儿把自己顶掉了，**问题自愈**。

#### 严重度：单轮失败，下一轮自己好

所以症状不是「彻底坏掉」，而是「长对话偶尔莫名报一次错，重问一遍就正常了」—— 这类间歇性故障恰恰最难查。而且 `src/main.py:423` 会把原始英文异常直接打给用户：

```
messages with role 'tool' must be a response to a preceding message with 'tool_calls'
```

看到这句话，第一反应通常是怀疑 API Key 或网络，很难联想到是自己的裁剪逻辑切错了地方。

#### 顺带发现：token 估算把工具调用当成了 1 token

`estimate_tokens` 是把每条消息的 `content` 按 `(字符数+1)//2` 相加。但带 `tool_calls` 的 `AIMessage`，它的 `content` 通常是空字符串 `""` —— 工具名和参数都存在 `tool_calls` 字段里，不在 `content` 里。于是整个工具调用载荷被 `max(1, ...)` 兜底成 **1 token**。（这条也是 3.1「预算算错对象」的一部分；已随本条一起修掉，见下。）

#### 修复：窗口起点对齐到合法边界

**思路**：`keep_from` 由条数和字符长度算出，这两个量永远不会知道配对关系 —— 所以不去改它们，
而是在它们算完之后**加一道对齐**，把起点推到合法边界上。窗口首条不能是 `ToolMessage`，就这一条不变量。

**选型：为什么没直接用 `trim_messages(start_on="human")`。** LangChain 内置的
`trim_messages` 正是为这个坑准备的，但它不认识本项目的两个约束：一是裁剪结果要转成
`RemoveMessage` 写回 checkpoint（需要保留原消息的 `id` 身份，而 `trim_messages`
的语义是「返回一个新列表」），二是窗口外的消息要拼成滚动摘要。接它的成本高于自己写
七行对齐，且会让「条数闸门 → token 闸门 → 对齐」这条链路的可读性变差。所以自己实现，
把踩过的坑就地记在 docstring 里。

**改动一：`_align_window_start()`（`src/memory/summarizer.py`）**

```python
keep_from = max(0, len(messages) - max_messages)
while keep_from < len(messages) - 1 and estimate_tokens(messages[keep_from:]) > max_tokens:
    keep_from += 1
# 条数与 token 两道闸门都算完之后再对齐边界：这两道闸门只看长度，
# 不认识 tool_calls ↔ ToolMessage 的配对，必须由对齐兜住协议不变量。
keep_from = _align_window_start(messages, keep_from)
```

对齐本身是「**优先前进，前进不通才后退**」：

```python
index = keep_from
while index < len(messages) and isinstance(messages[index], ToolMessage):
    index += 1                      # ① 向后跳过孤儿结果
if index >= len(messages):          # ② 前进会把窗口清空，才退回去
    index = keep_from
    while index > 0 and isinstance(messages[index], ToolMessage):
        index -= 1                  #    退到带 tool_calls 的父消息
return index
```

两个方向的取舍不是随手写的：

- **默认前进**（丢掉那几条孤儿结果）。窗口只会变小，两道预算闸门天然仍然满足，不需要重算。代价是丢掉一次工具执行的结果 —— 但那次调用的**请求**已经被切走了，结果本身在模型看来也无从解释，丢掉不损失可用信息。
- **只在前进会清空窗口时后退**。这发生在「超小预算 + 一轮里多次工具调用」的极端场景（`keep_from` 之后全是 `ToolMessage`）。此时后退会带上父 `AIMessage`，窗口略微超出 `max_tokens` —— **刻意接受这个超支**：超预算最坏是账单大一点，发出非法请求体是必然 400。

**改动二：`_billable_chars()` —— 顺手把上面「顺带发现」的 token 漏算一起修掉**

`estimate_tokens` 原本只数 `content`，而带 `tool_calls` 的 `AIMessage` 其 `content` 是空串，
整个调用载荷被 `max(1, ...)` 兜底成 1 token。现在把工具名与参数的字符数也计入：

```python
chars = len(str(getattr(message, "content", "")))
for call in getattr(message, "tool_calls", None) or []:
    if isinstance(call, dict):
        chars += len(str(call.get("name", ""))) + len(str(call.get("args", "")))
```

这两处必须一起改：只修对齐不修估算，token 闸门在工具密集的长对话里几乎不生效
（正是最需要它的场景）；只修估算不修对齐，`keep_from` 被推得更远，切错刀口的概率反而更高。

**改动三：回归用例（`tests/test_phase_three.py::ToolCallPairingTests`）**

| 用例 | 钉住什么 |
|---|---|
| `test_documented_orphan_case_is_fixed` | 上文走查的那段 14 条历史（断言 `len(messages) == 14`，防止走查和代码漂移） |
| `test_no_orphan_across_generated_corpus` | 穷举语料的 4455 个裁剪点，逐个断言无孤儿（`assertGreater(checked, 1000)` 防止语料被改小后静默失效） |
| `test_tiny_budget_retreats_to_include_the_parent_call` | 极端预算下**必须后退**，`recent[0] is messages[1]` |
| `test_estimate_counts_tool_call_payload` | 带 `tool_calls` 的空 content 消息，估算必须大于纯空消息 |

断言的是协议不变量本身（`assertNotIsInstance(recent[0], ToolMessage)` + 无孤儿 id），不是返回文本。

**顺带修正一个把 bug 当成预期的旧用例。** `test_tool_results_are_trimmed_and_checkpoint_update_removes_old_messages`
原来的 fixture 是 `[Human, Tool(call-1), AI]` + `max_messages=2` —— 一条**没有父消息的裸
`ToolMessage`**，窗口首条恰好就是它。修复后这条用例失败了（`AssertionError: '已裁剪' not found in ''`），
因为 `tool-1` 现在落到窗口外、变成 `RemoveMessage` 而非被截断的替换消息。
根因不是新代码有问题，而是**这个 fixture 当初就把「切断配对」的错误行为编码成了预期**。
改成带 `tool_calls` 父消息的拟真历史 + `max_messages=3`，并在注释里写明原委。

> 这是本次修复里最值得记的一点：bug 能长期存活，往往是因为有测试在**保护**它。
> 一份不合法的 fixture（现实中不会出现的裸 ToolMessage）让绿灯变成了噪音。

#### 效果

- **孤儿请求体归零**：1541 次有效裁剪 474 → **0**；端到端 21 次真实入参 6 → **0**。
- **代价是平均每次裁剪少保留 0.42 条消息**（12.00 → 11.58）。对齐只会向后跳过孤儿结果，所以损失有上界：最多等于一轮里的工具调用条数。
- 测试基线 190 → 198（本条 +4，同期 `UsageTracer` +4）。
- 1.1 与 3.1 的重叠部分（`tool_calls` 载荷算成 1 token）已消除；**3.1 剩下的部分仍未修** —— system prompt 与工具 JSON Schema 依然不在预算内，护栏给的仍是偏乐观的估算，只是不再乐观得离谱。

---

### 1.2 MCP server 旁路零审批暴露 `unlock`【实测】

图内 `control_lock(action="unlock")` 是全项目唯一 `risk_level="high"` 的动作，必须人工批准（`src/agent/approval.py:97`）。但 `src/mcp/server.py:205` 的 `control_lock_mcp(device_name, action)` 直接包装 `built["control_lock"]`，docstring（`:209`）还明确文档化了 `unlock`：

```
control_lock(action='unlock')      -> 需要审批 risk=high     ← 图内
control_lock_mcp(action='unlock')  -> 无任何拦截             ← MCP 入口
```

对 `src/mcp/server.py` 全文 grep `interrupt` / `approval` / 身份校验：**零命中**。`activate_scene_mcp`（`:248`）同理。

**根因不是漏写，是架构层次**：审批实现在编排层（图节点里 `interrupt()`），而 MCP server 是**另一个进程入口**，根本不经过图。任何绕过编排层的入口都自动绕过审批。

> **概念**：安全边界必须在**每个入口**重复，或者下沉到所有入口共用的那一层。「在编排器里做鉴权」是分布式系统里的经典反模式。

---

### 1.3 审批绑定「工具名」而非「效果」【实测】

`src/agent/approval.py:78` 的 `risky_calls` 按工具名过滤（`activate_scene` / unlock / 自动化创建）。于是等效操作的审批状态不一致：

```
activate_scene('回家模式')                            -> 需要审批 risk=medium
control_light + control_ac + control_curtain（等效）  -> *** 零审批直接执行 ***
```

**这不需要恶意提示词**。用户说「帮我把客厅调成回家的样子」时，模型**自然可能**选单设备路径，审批就静默消失了。审批锚定的是「模型挑了哪个工具」，不是「产生了什么物理效果」。

（澄清：`control_lock(action='lock')` 不需审批是**正确的刻意设计** —— 上锁是安全方向。这条不算缺陷。）

---

### 1.4 Planner 路径 unlock 降级为 medium + 模型自述文案【实测】

`src/agent/graph.py:441` 的 `plan_approval_node` 只调 `plan_approval_payload`，**从不调用 `build_approval_request`**。而 `src/agent/planning.py:148-159` 里：

- `risk_level` 硬编码 `"medium"`（`:156`），与计划是否含 unlock 无关
- `question` 由模型自己写的 `step["description"]` 拼成（`:151`）

`src/main.py:255-266` 的 `_ask_for_approval` 又只渲染 `payload.get("question")`，从不展示底层 `tool_calls`。三者叠加，实测：

```
control_lock 可被 Planner 规划: True   （在 PLANNING_TOOL_NAMES 中）
risk_level = medium                    （硬编码）

CLI 展示给用户的全部内容：
  已生成 2 步执行计划，是否开始执行？
  1. 打开客厅灯
  2. 准备门口环境          ← 模型措辞完全没提"解锁"

实际会执行：
  control_light({'device_name': '客厅灯', 'action': 'turn_on'})
  control_lock({'device_name': '入户门锁', 'action': 'unlock'})
```

> **概念**：HITL 的展示内容必须是**机器从 `tool_calls` 生成的事实**，不能是模型撰写的摘要。否则「人在回路」批准的是模型的**说法**，而不是模型的**行为** —— 这恰好废掉了 HITL 的全部意义。风险等级同理，必须从实际动作推导。

这条和 1.2 是答辩时最容易被追到的两条：「你的 HITL 怎么防止模型绕过？」顺着问一层就到。反过来说，**修完之后它们会变成最有说服力的深度案例**。

---

## 二、两个学科级缺口

### 2.1 没有评测体系 —— 学习价值最高的缺口【实测 · 013 已补一维】

> **状态**：2026-08-31，013 迭代补上了**四维中的一维**（RAG 召回质量）。
> 下面的盘点原文保留 —— 它描述的是「零评测」这个起点，而**四分之三仍然是这个状态**。

`docs/iterations/README.md` 把 008「Agentic RAG 与轨迹评测」标为已实现，但实际上：

- `src/evaluation/trajectory.py` 全文 **18 行**，一个纯函数，自创建后再没改过
- 唯一调用点 `tests/test_phase_twelve.py:80` 喂给它的是**手写字面量 dict**，不是 `graph.invoke()` 的真实输出
- ~~全仓库 `.jsonl` / `.yaml` / `.csv` 数据集数量为 **0**~~ → 013 起有一份：
  `evals/knowledge_recall.json`，**63 条 golden 用例**（口语 30 / 说明书原词 10 /
  错误码 10 / 同码异义 6 / 应当拒答 7）
- ~~无 eval 命令~~ → `python -m src.evaluation.recall`（`--offline` 跳过需要 API 的两档、
  `--sweep` 在「向量权重 × 分数下限」网格上扫参）；**无 Makefile、无 CI（`.github` 目录仍不存在）**
- 所有测试用 FakeLLM —— **没有一项在衡量真实模型行为**（评测 runner 也一样：它测检索，不测 LLM）

~~四个关键维度**全缺**测量~~ —— 013 之后是**一有三缺**：

| 维度 | 状态 |
|---|---|
| RAG 召回质量 | **013 已补**。Recall@1 51.8% → 87.5%，口语查询 3/30 → 23/30，拒答准确率全程 100% |
| 路由准确率 | 仍缺 |
| Planner 计划质量 | 仍缺 |
| 端到端任务成功率 | 仍缺 |

**所以这条的「学习价值最高」定位不变。** 补掉的是最容易补的那一维——检索是纯函数，
`source#section` 是可以逐字比对的标签，golden 用例写起来是机械劳动。而剩下三维都要
面对「正确答案不唯一」这个真问题：一个多步计划有几种合法写法？端到端「成功」怎么定义？
这三个问题 013 一个都没回答。

**为什么这是「学科级」而不是「还缺个功能」**：Agent 工程与传统软件最本质的区别，就是被测系统是非确定性的。现有 264 项测试钉住的是**机制正确性**（权限边界、数据库状态、设备副作用、事件顺序）—— 这部分做得很好，是很多教学项目都没有的。但一行 prompt 改动导致意图分类准确率从 92% 掉到 71%，当前测试**全绿**，没有任何信号。

**同一类盲区的两个实例**（2026-08-23 修 1.1 时撞上的，都不是模型行为问题，而是「测试钉住了数据结构，没钉住真实输出路径」）：

1. `UsageTracer._log` 的格式串引用了一个从未被赋值的 `model` 键。真实 CLI 每轮打三行
   `Error in UsageTracer.on_llm_end callback: KeyError('model')`，token 日志一条都不落盘。
   原有两条用例都注入了自定义 sink，`_log` 这条唯一有 bug 的路径从未被执行过 ——
   而且异常被 LangChain 回调管理器吞成一行 stderr，业务照常跑完，**度量静默归零比没有度量更糟**。
   补的守卫用例刻意用默认 sink + 真实 loguru（不能 patch 成 `MagicMock`，Mock 的 `.info()`
   接受任何参数，而 `KeyError` 恰恰发生在 loguru 内部的 `str.format` 里）。
2. 1.1 那条把 bug 当成预期的 fixture（见上文）。

两条都说明：**测试通过 ≠ 那条路径被执行过**。这正是评测体系要覆盖的另一半 —— 不只是「模型答得对不对」，还有「真实链路上到底发生了什么」。

项目自己也意识到了 —— `docs/iterations/011-engineering-hardening.md` 结尾写「结构收口后，拟真语料回归集才有稳定的底座」。**底座已经有了，回归集还没建**。

**可以分两步做**，第一步不需要 API Key：

1. 离线档：golden 数据集（`query → 期望路由 / 期望是否走 Planner / 期望 RAG 命中的 source`）+ runner + 分维度打分。用 FakeLLM 也能测确定性分支（`classify_intent` 的关键词兜底、`should_use_planner` 的正则、RAG 的词法检索），这三块本来就是纯函数，正好能测出真实准确率。
   > **013 起这一步有了可照抄的样板**：`evals/knowledge_recall.json` + `src/evaluation/recall.py`
   > 就是照这个形状做出来的（数据集是纯声明、runner 遍历它、输出一张分维度分数表、
   > `--offline` 保证无 Key 可跑）。剩下两块纯函数——`classify_intent_fallback` 的词表和
   > `should_use_planner` 的正则——把 `recall.py` 的骨架换个打分函数就能开工。
   > 有两条纪律值得从 013 抄走：**阈值要扫参扫出来而不是手调**（`--sweep`
   > 证明了 0.35 这一格对权重不敏感，所以不是拟合某一条用例），
   > 以及**数据集里必须有你真实踩过的坑**（013 刻意留了两处语料缺口当被测行为）。
2. 真实模型档：同一份数据集打真实端点，产出可对比的分数。这一档要能被跳过（无 Key 时），否则 CI 跑不了。

---

### 2.2 LLM 被当成「总会成功返回合法结果」的依赖【审查】

| 调用点 | 位置 | 防护 |
|---|---|---|
| 意图分类 | `src/agent/routing.py:89-95` | ✅ try/except + 关键词兜底 |
| ReAct 主调用 | `src/agent/graph.py:691` | ❌ 裸奔 |
| 纠正重试 | `src/agent/graph.py:719` | ❌ 裸奔 |
| Planner 结构化输出 | `src/agent/graph.py:400` | ❌ 裸奔 |

全库 `with_fallbacks` / `include_raw` **零命中**。

Planner 这处尤其值得注意：`ExecutionPlan` 的约束很紧（`steps` 有 `min_length=2`、`tool_name` 是从能力声明派生的 `Literal`），schema 违约直接 `ValidationError` 冒泡。而 `src/agent/planning.py:42-46` 的注释详细记录了「模型会写 Home Assistant 风格的 `turn_on`，这是规划第一版反复失败的根因」—— 解法是把合法值写进 prompt，**纯 prompt 约束，零运行时兜底**。prompt 是概率性的，约束不是。

> **概念**：LLM 是网络依赖 + 概率性生成器，两种失败都要处理。routing.py 已经示范了正确做法，把同一模式推广到另外三处即可 —— 但要小心 `interceptors.py:6-21` 已经指出的坑：无条件 `except Exception` 会吞掉 `GraphInterrupt`，破坏审批语义。

---

## 三、次要但值得知道

### 3.1 上下文预算算错了对象【实测】

`compact_context_node`（`src/agent/graph.py:286`）的 `max_tokens=2400` 只约束消息历史。而实际请求体里还有：

- system prompt 本身（同口径估算约 2200 token，**单它就接近整个预算**）
- 全部工具的 JSON Schema（`bind_tools` 注入）
- ~~`tool_calls` 的参数载荷（如 1.1 所述被算成 1 token）~~ **已随 1.1 修复**
- `memory_context` + `conversation_summary`

日志打出 `上下文规模 | messages=12 | estimated_tokens=42` 时，真实请求可能是它的几十倍。护栏给的是虚假安全感。（1.1 修完后这个倍数变小了，但性质不变：**前三项里仍有两项完全不在预算内**。）

> **概念**：上下文预算必须对**完整请求体**计数，而且要用真实 tokenizer 而非字符数近似。

### 3.2 `Send` 动态并行没有并行任何东西【审查】

`src/agent/parallel.py:55` 的 `query_device` 分支只做 `registry.get()` 内存字典查找 —— 零 IO、零 LLM。fan-out + reducer 的调度开销**大于**直接循环。它演示了 `Send` 的**机制**，但没演示 `Send` 解决的**问题**。

而真正该并行的地方反而是串行的：`connect_external_tools` 在 `for` 循环里逐个发现服务；`src/mcp/client.py:157` 每次工具调用都 `async with _open_session(service)` 重建会话，stdio 下等于**每次调用拉起一个新子进程**。

整个 `src/` 除 `src/mcp/` 外零 `async def`；图只用 `graph.invoke` / `graph.stream` 驱动，不用 `ainvoke` / `astream`。

> **概念**：并行的收益来自 IO 等待的重叠。把 fan-out 用在纯内存操作上，学到的是 API 而不是判断力。如果要展示 `Send`，应该让分支里有真实等待（多个外部 MCP 调用、或多设备的模拟延迟）。

### 3.3 Prompt injection 面【审查】

两处不可信输入无隔离地进入 system prompt：

1. `src/agent/graph.py:195` 把远端 MCP server 返回的 `tool.description` **无条件拼进** system prompt。远端服务改一行描述就能改变本地 Agent 的行为。
2. `src/agent/graph.py:250-263` 把记忆的 `memory_value` 原样格式化进 `memory_context`，再由 `:677` 拼进 system prompt。记忆是**跨会话持久化**的，所以这是存储型注入面。

当前唯一防护是 `src/agent/graph.py:680` 那句自然语言祈使句：「这些标识来自受信任的业务上下文，不得根据用户文本改写。」它保护的是身份字段，不覆盖记忆内容和远端描述。

> **概念**：不可信输入要么隔离到 user 消息里（而不是 system），要么显式标注边界。「用一句祈使句请模型别照做」不是控制手段。

### 3.4 RAG：012 升级、013 补上语义通道，边界换了一批【实测 · 已修复】

> **状态**：012 迭代（`docs/iterations/012-manual-rag-upgrade.md`）已把本条的主体修掉——
> 原先「子串匹配猜型号 + 两次 `.replace()` 改写 + 拼原始 chunk」的路径全部替换。
> **013 迭代**（`docs/iterations/013-hybrid-retrieval.md`）又把 012 留下的两条缺口
> （词法无语义泛化、自证检查仅 3 条）也修掉了。本条保留：改了什么，以及**换掉之后的新边界**。

012 后的现状：

- 检索前先做**实体消解**（`src/knowledge/resolution.py`）：用户措辞 → 设备实例 → 型号，
  结果四态，**仅唯一命中放行检索**，其余三态（ambiguous / no_model / unknown）直接拒答，
  不存在「兜底成全库检索」的分支。型号唯一数据源是 `BaseDevice.model` 字段，
  知识模块不再持有第二份型号表。`KnowledgeBase.search()` 的 `model` 改为关键字必填参数
  ——全库检索必须是调用方的显式选择，不能是漏传参数的副产品。
- **检查项自证分流**（`src/knowledge/selfcheck.py`）：说明书排查项末尾挂行内 Markdown
  注释（`<!--check:xxx-->` 可自证 / `<!--manual-->` 需人工），可自证项直接读设备与传感器
  真实状态核对（不问模型），判定三值 problem / ok / unknown——读不到传感器退回人工，
  绝不默认通过。语料引用未声明的 check id 会在 `KnowledgeBase` 构造期抛 `ValueError`（fail-fast）。
- **LLM 已接入**（原先「`graph.py:183` 为 knowledge 角色准备的专用 LLM 永远用不到」
  这句已过时）：`build_knowledge_rag_subgraph(knowledge_base, registry, llm=llm, ...)`，
  LLM 负责查询重写（从该型号说明书真实存在的小节标题清单里挑）与答案正文；
  「自动核对结果」块和「来源」块由代码拼接、不经模型改写，LLM 失败时退回片段拼接
  且引用一个不少。
- 语料从 4 份约 1.5KB 扩到 ~~**5 份 12 个小节约 5.4KB**~~ → 013 再扩到 **39 份、124 个
  chunk、13 个型号，覆盖全部 10 种设备类型**（每个型号三份：故障代码 / 常见症状排查 /
  保养与清洁）；`tests/test_knowledge_rag.py` ~~36 项~~ → **61 项**，
  全量基线 ~~234 项 + 137 subtests~~ → **`264 passed, 383 subtests passed`**。

**013 又改了什么**（把 012 那两条缺口换成了机制）：

- **检索变成双通道**：`rank_bm25.BM25Okapi` 词法通道（jieba `cut_for_search` 分词、
  在**全语料**上建索引以保证 IDF 是语料级常量）+ `ApiEmbeddings` 向量通道（文档向量按内容
  sha256 落盘缓存）。索引在 `KnowledgeBase` 构造期建一次，不再每次检索重跑分词。
- **名次与准入是两套分数，刻意不混**：`rrf`（`Σ w/(60+rank)`，对量纲免疫）只决定名次；
  `confidence`（锁死 `[0,1]`）只决定放不放行，三档下限套在它身上。这一处是本轮最要紧的判断——
  **RRF 是纯名次的，名次第一必然拿满分，跟它到底像不像毫无关系**，拿它去套下限会让拒答分支
  永远走不到。
- **零 Key 测试与真语义泛化的冲突靠显式注入解决**：`EmbeddingProvider` 协议 +
  `NullEmbeddings`（`embed_*` 直接抛，而不是返回零向量——零向量会安静地贡献 0 分参与融合，
  只有召回质量偷偷退化）。测试注入 `StubEmbeddings`，仍然确定性、仍然不要 Key。
- **自证检查 3 条 → 12 条**，覆盖全部设备类型，不再只有空调；`CheckContext` 一个字段都没加。
- **实测收益**：口语查询召回 **3/30 → 23/30**，Recall@1 **51.8% → 87.5%**（+35.7pp），
  拒答准确率**全程 100%**。其中「把覆盖率换成 BM25」单独贡献 +7.1pp，剩下 +28.6pp 是语义通道给的。

**新的诚实边界**（四条，都是 013 亲手量出来的，别吹过头）：

- **绝对置信度分不开「对的症状」和「同一台电器的另一个症状」。** 在 63 条用例上统计
  `text-embedding-v4` 的余弦分布：正例中位 **0.653**、困难负例（同型号的**其它**症状小节）
  中位 **0.568**，两条分布**大幅重叠**，中位只差 0.085。基线提到 0.55 也只挡掉 35% 的困难负例、
  同时丢掉 15% 的正例。代价直接体现在闸门余量上：拒答用例里最高的假阳性是 **0.336**
  （「卧室空调有点响」命中 FrostLine 的「制冷效果不佳」），离阈值 0.35 **只有 4% 余量**。
  真正要认清的是——**这道闸门的可靠性其实来自被保留的 BM25 通道，而不是来自阈值调得准**
  （「响 / 噪音」在 FrostLine 全文不存在，词法通道直接返回 0）。下一步该做的不是继续拧阈值，
  而是加一道 top1 与 top2 的差距门；013 没做，因为那是第三种机制，得先有数据证明它比调阈值更稳。
- **确定性的代价真实发生了，只是发生在生产侧。** 余弦的两个刻度（噪声基线 0.42 /
  强命中 0.70）是对 `text-embedding-v4` **实测标定**的，**换 embedding 模型必须重测**。
  测试侧用 stub 躲掉了这个问题，生产侧躲不掉——这正是 012 当初拒绝上向量检索时列出的
  最贵那一项，013 没有消灭它，只是把它从测试挪到了部署。
- **召回评测只测首轮检索。** 不测端到端任务成功率、不测真实 LLM 的综合质量、
  也不测查询重写（刻意的：要回答的问题是「语义通道能不能替掉那张人工症状词表」，
  把重写也跑上等于让词表替检索背书）。所以 87.5% 是**检索层**的数字，不是「说明书问答答对率」。
- **BM25 的噪声基线依赖语料规模。** 3.5 这个数是由 IDF 与平均文档长度决定的，两者都是
  语料级统计量，**换语料必须 `--sweep` 重标**。已经踩过一次：拿三个 chunk 的合成语料写测试，
  所有 BM25 分数都远低于 3.5，测出来的是「基线没标定」而不是被测的机制。

**仍然成立的旧缺口**：

- ~~**检索仍是词法 bigram 重叠 + 人工症状词表，没有语义泛化**~~ —— **013 已修**（见上）。
  唯一残留：`_SYMPTOM_LEXICON` 那五条空调正则**还在**，但定位从「语义泛化的唯一手段」降级成
  「没配 embedding 时的确定性兜底」。要不要给另外十二个型号补词表，取决于离线配置是不是一个
  需要认真支持的场景——013 没有下这个判断。
- ~~**自证检查项只有 3 条，且全是空调的**~~ —— **013 已修**（3 → 12 条，覆盖全部设备类型）。
  013 还顺手补上了 012 缺的反方向校验：**声明了却没有任何语料引用的检查项是死代码**，
  现在也会在测试里失败（012 只校验了「语料引用的 id 必须已声明」这一个方向）。
- **只核对排名第一的小节**。第二三名不核对，理由是"也沾点关系"的小节会把无关的
  核对结论混进答案。（这条 013 没动。）

### 3.5 并发安全【审查】

零锁 + 3 处 `check_same_thread=False`。后台调度线程与主对话流程并发修改共享 `registry`。项目已在 `docs/defense-deep-dive.md` 自陈为已知局限 —— 自陈本身就是答辩加分项，优先级可以放低。

---

## 四、明天决策用的三条路线

不建议同时开工。三条路线各自独立、都能单独收尾。

### A. 修 HITL 的两个洞（1.2 + 1.4，可带上 1.3）

- 改动范围：`src/agent/approval.py`、`src/agent/planning.py`、`src/agent/graph.py:441`、`src/main.py:255`、`src/mcp/server.py`
- 核心动作：`plan_approval_payload` 改为从实际 `tool_calls` 机器生成动作明细并推导 `risk_level`；`_ask_for_approval` 展示明细；MCP server 的 `unlock` 加边界
- 配套测试：钉住「含 unlock 的计划必须 risk=high」「展示文本必须包含 action 字面量」
- **答辩价值最高**，也最容易被追问

### B. 建评测回归集（2.1）—— 2026-08-31 起是**部分完成**

- ~~改动范围：新增 `evals/`（数据集 + runner），扩写 `src/evaluation/`~~ ——
  013 已建：`evals/knowledge_recall.json` + `src/evaluation/recall.py`
- 先做离线档：`classify_intent` 兜底路径、~~`should_use_planner` 正则~~、~~RAG 词法检索~~
  三块都是纯函数，能立刻产出真实准确率数字 —— **RAG 那块已做完**（Recall@1 51.8% → 87.5%），
  另两块还没碰，但骨架可以直接照 `recall.py` 抄
- 再加真实模型档（无 Key 时可跳过）+ 一个 CI workflow —— **两样都还没有**（`.github` 目录仍不存在）
- ~~**学习价值最高**，是唯一还没碰的学科~~ —— 学习价值依然最高，因为**难的三维一维没动**：
  路由、Planner 计划质量、端到端成功率都要面对「正确答案不唯一」，而 RAG 召回是四维里
  唯一能逐字比对标签的那一维。011 收口了结构，013 给了样板，剩下的是照着做

### ~~C. 修消息裁剪 bug（1.1）~~ —— 2026-08-23 已完成

- 实际改动范围：`src/memory/summarizer.py` 单文件 `+50 −1`（两个新私有函数各七八行可执行代码，其余是记录踩坑原因的注释）+ `tests/test_phase_three.py` 4 条回归用例 + 修正 1 条把 bug 当预期的旧用例
- 效果：有效裁剪的孤儿率 30.8% → 0%，代价是平均每次少保留 0.42 条消息
- 结论：确实如预估「范围最小、最快见效」，可以作为 A 或 B 的开胃菜 —— 现在开胃菜吃完了

如果只能选一个：**答辩导向选 A，学 Agent 工程选 B**。
