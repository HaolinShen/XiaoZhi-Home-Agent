# 智能家居 Agent — 从零理解一个真实的 LangGraph 项目

> **这本教程写给谁**：想学 AI Agent 开发、但被"Agent"这个词绕晕过的人。
> 你只需要会 Python（类、装饰器、类型注解），知道"大模型是个能聊天的 API"就够了。
> 向量数据库、RAG、ReAct、状态机这些词，遇到时会从零解释。
>
> **代码版本**：`HEAD = e7d1113`（011 工程收口）。全文行号引用均按此版本实测核对。
> **测试基线**：`264 passed, 383 subtests passed`（实测，约 22 秒）。
> **阅读方式**：第 0 章必读（10 分钟跑起来）。之后每章都是"上一章留下了什么问题 → 这一章怎么解"，建议顺序读一遍，再当参考手册查。

---

## 目录

**入门**
- [第 0 章 这本教程怎么读](#第-0-章-这本教程怎么读)
- [第 1 章 Agent 到底是什么 —— 从"开客厅灯"说起](#第-1-章-agent-到底是什么--从开客厅灯说起)
- [第 2 章 最小可用：LLM + 一个工具 + 循环](#第-2-章-最小可用llm--一个工具--循环)

**打地基**
- [第 3 章 Agent 的手：设备层与能力声明](#第-3-章-agent-的手设备层与能力声明)
- [第 4 章 Agent 的工具箱：工厂与显式依赖注入](#第-4-章-agent-的工具箱工厂与显式依赖注入)

**让它可靠**
- [第 5 章 Planner：让 Agent 先说清要做什么](#第-5-章-planner让-agent-先说清要做什么)
- [第 6 章 Executor 与 Verifier：不听模型自述，去查真实状态](#第-6-章-executor-与-verifier不听模型自述去查真实状态)
- [第 7 章 人在回路：让暂停活过进程重启](#第-7-章-人在回路让暂停活过进程重启)
- [第 8 章 可信身份边界：让模型物理上无法伪造身份](#第-8-章-可信身份边界让模型物理上无法伪造身份)

**让它变快、变专**
- [第 9 章 意图路由：什么时候**不该**用 LLM](#第-9-章-意图路由什么时候不该用-llm)
- [第 10 章 并行子图与 Send](#第-10-章-并行子图与-send)
- [第 11 章 多智能体：安全边界建立在"看不见"上](#第-11-章-多智能体安全边界建立在看不见上)

**让它有记忆、有知识、有未来**
- [第 12 章 记忆：Checkpoint 与长期记忆](#第-12-章-记忆checkpoint-与长期记忆)
- [第 13 章 知识检索与拒答纪律](#第-13-章-知识检索与拒答纪律)
- [第 14 章 事件驱动自动化：把计划投射到未来](#第-14-章-事件驱动自动化把计划投射到未来)

**工程收口**
- [第 15 章 可观测性：出了问题怎么看见](#第-15-章-可观测性出了问题怎么看见)
- [第 16 章 MCP：把工具给别的 AI 用](#第-16-章-mcp把工具给别的-ai-用)
- [第 17 章 怎么验证你的改动](#第-17-章-怎么验证你的改动)
- [第 18 章 全景图与 13 条不变量](#第-18-章-全景图与-13-条不变量)
- [第 19 章 已知边界与下一步](#第-19-章-已知边界与下一步)

**附录**
- [附录 A 动手实验清单](#附录-a-动手实验清单)
- [附录 B 常见报错速查](#附录-b-常见报错速查)

---

## 第 0 章 这本教程怎么读

### 0.1 这个项目是什么

一句话：**你对着终端说人话，它去控制家里的灯、空调、窗帘、门锁**。

```
你: 打开客厅灯，空调调到 26 度
🤖 已生成 2 步执行计划，是否开始执行？
    1. 打开客厅灯
    2. 空调设为 26 度
你: 好
🤖 ✅ 客厅灯已打开（亮度 80%）
   ✅ 客厅空调已设为 26℃
```

家里没有智能设备也能跑——项目内置了一个**设备模拟器**，所有设备状态存在内存里，行为跟真的一样（开了灯，湿度传感器读数会跟着变）。

它同时是一个**教学项目**。README 写明面向"想学习现代 AI Agent 开发的开发者"，所以代码里注释密度很高，而且踩过的坑就地记在注释里。本教程的很多"为什么"直接来自那些注释。

### 0.2 环境准备

**必须用已有的 Conda 环境**（不要 `uv run`，不要新建 `.venv`）：

```bash
# 解释器路径固定
F:\Software\Anaconda\envs\langgraph\python.exe
```

Windows 下运行任何会打印设备名或 emoji 的命令，**都要加 `PYTHONIOENCODING=utf-8`**，否则直接 `UnicodeEncodeError` 崩掉：

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m pytest -q
```

这不是可选的讲究。中文设备名（"客厅灯"）+ emoji（🟢）在 Windows 默认的 GBK 控制台编码下必然报错，是本项目最高频的入门障碍。

### 0.3 十分钟跑起来

**第 1 步：确认测试全绿**（不需要 API Key，这一步就能验证代码是好的）

```bash
cd "G:/大厂学习/minimind/langgraph"
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m pytest -q
```

期望看到：

```
264 passed, 383 subtests passed in 22.36s
```

**为什么不需要 API Key 就能跑测试？** 因为需要模型的用例都用一个假的 LLM（`FakeLLM`）替掉了真实模型调用，其余 264 项里的大多数根本不碰模型。013 上了向量检索之后这条纪律也没松：语义通道是**显式注入的依赖**，测试注入一个 `StubEmbeddings`，所以仍然离线、仍然确定性、仍然不要 Key。这是本项目一个重要的工程决策，第 17 章会讲透。现在你只需要知道：**测试验证的是代码机制，不是模型智力**。

**第 2 步：配 API Key**（要真的对话就需要它）

```bash
cp .env.example .env
# 编辑 .env，至少填 LLM_API_KEY
```

**第 3 步：启动对话**

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m src.main
```

试着说：`打开客厅灯`。

**第 4 步（最重要的一步）：打开 trace 再说一遍**

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m src.main --trace
```

`--trace` 会把 Agent 内部的判断过程打出来：意图分类结果、走了哪条路径、记忆是否适用。**建议你从此以后一直开着它读这本教程**——本书讲的每一个机制，都能在 trace 输出里看到对应的一行。

### 0.4 全书地图：13 次迭代长出来的 5 条路径

这个项目不是一次写成的。它有 13 份迭代方案文档（`docs/iterations/001~013`），每一份都是"上一版遇到了什么问题 → 这一版怎么解"。**本教程的章节顺序就是这个演进顺序**，因为理解"为什么需要它"永远比"它是什么"重要。

最终长成的样子，是一张有 17 个节点的图，里面藏着**5 条互斥的业务路径**：

```
你说的话
   │
   ▼
sync_context ──► memory_reasoner ──► task_router
（定身份和空间）  （记忆适用性判断）      │
                                        │  ★ 全图唯一的业务分叉点
      ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
      ▼                ▼               ▼              ▼             ▼
  clarification   device_query_    knowledge_rag    planner    compact_context
  （信息不足     subgraph          （查说明书）    （多步任务）   （ReAct 主路）
    直接反问）   （多设备并行查）        │             │             │
      │              │                 │        plan_approval      agent ⇄ tools
      ▼              ▼                 ▼        （人工审批）        （思考-行动循环）
     END            END               END            │
                                              executor ⇄ verifier
                                              （执行）  （对账）
```

五条路径的对应章节：

| 路径 | 干什么 | 主讲章节 |
|---|---|---|
| **ReAct 主路** | 单步/简单任务，边想边做 | 第 2 章 |
| **Planner** | 多步任务，先列计划再执行，每步验证 | 第 5、6 章 |
| **并行查询** | 一次问多个设备状态 | 第 10 章 |
| **知识 RAG** | 查设备说明书类问题（BM25 词法 + 向量语义**双通道**混合检索） | 第 13 章 |
| **澄清** | 信息不足或记忆冲突，直接反问 | 第 9、12 章 |

> **第 13 章还没写到这一版**（本文目前写到第 7 章）。知识检索这条路径在 012 / 013 两轮迭代里
> 变化最大：现在是 **BM25 词法通道 + 向量语义通道**的混合检索，**名次**由 RRF 决定、
> **准入**由一个独立的 `confidence` 决定（两者刻意不混——RRF 是纯名次的，名次第一必然拿满分，
> 拿它去套分数下限会让拒答分支永远走不到）。语义通道是**显式注入的依赖**，所以测试仍然
> 不需要 API Key。设计与全部实测数字见
> [`iterations/013-hybrid-retrieval.md`](iterations/013-hybrid-retrieval.md)，
> 成稿章节见 [`guide/13-知识检索与RAG.md`](guide/13-知识检索与RAG.md)。

外加两个**跑在图之外**的子系统：

| 子系统 | 干什么 | 主讲章节 |
|---|---|---|
| **自动化** | 定时/事件触发的例程（起床、车辆到家） | 第 14 章 |
| **MCP** | 把工具暴露给 Claude Desktop 等外部 AI | 第 16 章 |

### 0.5 每章的结构

从第 1 章起，每章固定五段：

1. **要解决什么问题** —— 上一章的做法在什么情况下会失灵，配具体的失败例子
2. **代码怎么写的** —— 项目里的真实代码，带 `file:line`，可以点开对照
3. **关键设计决策** —— 为什么这样而不那样。这是本书最有价值的部分
4. **动手试一试** —— 跑一条命令、改一个参数、加一行 print，亲眼看到机制运转
5. **踩坑与局限** —— 已知的坑、以及这一步**没有**解决的问题

第 5 段不是客套。这个项目有一份诚实的现状盘点（`docs/gap-analysis.md`），列了 10 条已知缺口。本教程会如实标注它们，因为**把一个有边界的系统讲成完美的，是教学里最坏的一种谎**。你会在第 19 章看到完整清单。

---

## 第 1 章 Agent 到底是什么 —— 从"开客厅灯"说起

### 1.1 要解决什么问题

假设你想让大模型帮你开灯。你直接问它：

```
你: 打开客厅灯
GPT: 好的，我已经帮你打开客厅灯了！
```

**它在撒谎。** 大模型是一个"文本进、文本出"的函数，它没有手，没有网络连接，碰不到你家的灯。它只是根据训练数据判断"这种情况下人类会这么回答"，然后生成了一句最像样的话。

这就是 Agent 要解决的根本问题：**怎么让一个只会生成文本的模型，真的产生副作用**。

### 1.2 三个零件

答案出人意料地朴素，只需要三个零件。

**零件一：一个真的能开灯的函数。**

```python
def control_light(device_name: str, action: str) -> str:
    # 真的去改设备状态
    ...
    return "客厅灯已打开"
```

**零件二：把这个函数"描述"给模型。**

这一步是关键。你不能把 Python 函数对象传给模型——模型只吃文本。你要传的是一份**说明书**（业内叫 function schema / tool schema），大致长这样：

```json
{
  "name": "control_light",
  "description": "控制灯光设备的开关、亮度和色温",
  "parameters": {
    "device_name": {"type": "string", "description": "设备名称，如'客厅灯'"},
    "action": {"type": "string", "enum": ["on", "off", "set_brightness"]}
  }
}
```

模型看到这份说明书后，不再回答"我已经帮你开灯了"，而是回答一个**结构化的请求**：

```json
{"tool_calls": [{"name": "control_light", "args": {"device_name": "客厅灯", "action": "on"}}]}
```

翻译成人话：**"我想调用 control_light 这个函数，参数是这些，麻烦你帮我执行一下。"**

注意这里的分工：**模型只负责决定"调什么、传什么参数"，真正的执行是你的代码干的**。模型依然没有手，但它现在会指挥了。

**零件三：一个循环。**

模型说要调工具 → 你的代码执行 → 把执行结果告诉模型 → 模型看到结果，决定下一步（可能还要调工具，也可能可以回答了）→ 循环。

这个"思考-行动-观察"的循环有个名字叫 **ReAct**（Reasoning + Acting）。它就是 Agent 的全部核心。

### 1.3 一次工具调用在对话历史里是**一对**消息

这个细节很小，但后面第 12 章会有一个 bug 完全建立在它之上，所以现在就讲清楚。

Agent 每调用一次工具，对话历史里会留下**两条互相绑定**的消息：

| 顺序 | 消息类型 | 内容 | 绑定关系 |
|---|---|---|---|
| 1 | `AIMessage` | 模型说"我要调 control_light，参数是客厅灯/on" | 带 `tool_calls`，每个调用有唯一 id（如 `c1`） |
| 2 | `ToolMessage` | 你的代码回填"c1 的结果：客厅灯已打开" | 带同一个 `tool_call_id=c1` |

可以理解成**一问一答**：第 1 条是问题（模型的请求），第 2 条是答案（执行结果）。

**OpenAI 兼容协议对这个配对是强制的**：`tool` 角色的消息必须紧跟在带对应 `tool_calls` 的助手消息之后。如果请求体里出现了答案却找不到问题，服务端不会宽容地忽略，而是**直接 400 拒绝整个请求**：

```
messages with role 'tool' must be a response to a preceding message with 'tool_calls'
```

记住这句报错。第 12 章你会再见到它。

### 1.4 动手试一试

不启动 Agent，直接看"说明书"长什么样。在项目根目录开一个 Python REPL：

```python
from src.devices.simulator import SimulatorBackend
from src.devices.base import DeviceRegistry
from src.tools import build_all_tools

registry = DeviceRegistry(SimulatorBackend())
tools = build_all_tools(registry, memory_service=None, automation_runtime=None,
                        external_tools=None, enable_preference_tracking=False)

# 挑出开灯工具，看它的说明书
light = next(t for t in tools if t.name == "control_light")
print(light.name)
print(light.description)
print(light.args_schema.model_json_schema())
```

最后那行打出来的 JSON，就是**真正发给模型的东西**。教程后面讲的所有"工具面收窄""能力声明单一数据源"，最终都体现在这份 JSON 上。

### 1.5 踩坑与局限

**这一章的模型还太天真。** 我们假设模型看到工具说明书，就会正确地调用它。实际上：

- 模型可能把 `action` 写成 `turn_on`（Home Assistant 风格），而项目用的是 `on`。第 5 章会讲这个真实发生过的根因。
- 模型可能声称调用成功，但设备其实没变。第 6 章会讲怎么对账。
- 模型可能被用户的话诱导去解锁门锁。第 7、8 章会讲怎么拦。

**下一章的问题**：三个零件拼起来之后，多步任务为什么会做一半就跑偏？

---

## 第 2 章 最小可用：LLM + 一个工具 + 循环

### 2.1 要解决什么问题

第 1 章的三个零件要落成真代码，得回答几个具体问题：

- 循环写在哪？用 `while True` 吗？
- 对话历史存在哪？进程重启就丢了吗？
- 模型说"要调 3 个工具"的时候，谁负责挨个执行？

LangGraph 给的答案是：**把 Agent 画成一张图**。节点是"要做的事"，边是"下一步去哪"，图自带一个状态字典在节点间传递。

### 2.2 代码怎么写的

本项目的图定义在 `src/agent/graph.py:109-936` 的 `build_graph()` 里。ReAct 主路只涉及三个节点：

```
compact_context ──► agent ──► tools ──┐
      ▲              │                │
      └──────────────┴────────────────┘
```

对应的建图代码（`graph.py`）：

| 代码 | 位置 | 作用 |
|---|---|---|
| `workflow.add_node("agent", agent_node)` | `graph.py:806` | 调模型 |
| `workflow.add_node("tools", ToolNode(tools))` | `graph.py:809` | 执行工具（LangGraph 预置节点） |
| `workflow.add_edge("compact_context", "agent")` | `graph.py:850` | 压缩完就去调模型 |
| `workflow.add_edge("tools", "compact_context")` | `graph.py:922` | **工具执行完回压缩，不是回 agent** |
| `workflow.add_conditional_edges("agent", router, ...)` | `graph.py:900-909` | 模型说完话，决定去哪 |

**循环的出口只有一个判断**，在 `router` 函数里（`graph.py:889-898`）：

```python
def router(state) -> Literal["approval", "tools", "supervisor_finalize", "__end__"]:
    last_msg = state["messages"][-1]                                      # 891
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:           # 892
        if build_approval_request(last_msg.tool_calls) is not None:       # 893
            return "approval"                                              # 894  ← 敏感动作，要人批
        return "tools"                                                     # 895  ← 继续转
    if getattr(getattr(settings, "multi_agent", None), "enabled", False):  # 896
        return "supervisor_finalize"                                       # 897
    return "__end__"                                                       # 898  ← 出环
```

判断依据只有一条：**模型这次返回的消息里有没有 `tool_calls`**。有就继续转，没有就出环。

`agent_node`（`graph.py:655-751`）每轮做的事，简化后是：

```python
messages = list(state["messages"])                       # 664
# 拼一条 SystemMessage 插到最前面（每轮都新拼一条，不复用旧的）
messages.insert(0, SystemMessage(content=system_prompt + context_prompt + role_context))   # 685
response = active_llm.invoke(messages)                   # 691
return {"messages": [response]}                          # 742
```

### 2.3 关键设计决策

#### 决策一：状态里只有 `messages` 有 reducer，其余 43 个字段是"后写覆盖"

`AgentState`（`src/agent/state.py:16-72`）一共 44 个字段，但**只有一个带 reducer**：

```python
messages: Annotated[list, add_messages]      # state.py:29
```

其余 43 个全是裸 `NotRequired[...]`，语义是**后写覆盖前写**（last-write-wins）。

这件事必须讲清楚，因为有两个字段**看起来**像累加器，其实不是：

- `planning_results`（`state.py:72`）：累加是在 `verifier_node` 里**手写**的——先 `list(state.get("planning_results", []))` 拷一份，`.append(...)`，再整体覆盖写回（`graph.py:529-536`、542）。不是 reducer 干的。
- `parallel_query_results`（`state.py:43`）：一次性整体写入（`graph.py:634`）。

#### 决策二：`add_messages` 不只是 append —— 它还能删

这是小白 100% 会误解的地方。看到 `Annotated[list, add_messages]`，直觉是"节点返回的消息会被追加到列表末尾"。实际上 `add_messages` 有三种行为：

| 节点返回什么 | `add_messages` 干什么 |
|---|---|
| 一条新消息 | 追加到末尾 |
| 一条**已有 id** 的消息 | **替换**掉原来那条 |
| 一个 `RemoveMessage(id=...)` | **删除**对应消息 |

第三条是项目控制历史膨胀的手段：`compact_context_node` 把 `RemoveMessage` 对象混在返回值里（`graph.py:307-308`），reducer 看到就按 id 删。

**这意味着裁剪不是"这一轮少发几条"，而是真的从 SQLite 里删数据**——删了就再也读不回来。第 12 章会讲这个设计带来的一个真实 bug。

#### 决策三：工具执行完回 `compact_context`，不是回 `agent`

教科书 ReAct 是 `tools → agent`。本项目是 `tools → compact_context → agent`（`graph.py:922`）。

理由（`defense-deep-dive.md` 1.4 节）：**只有工具结果回流时消息才会膨胀**。把压缩放在回路上而不是入口，用户新一轮输入和工具结果回流就能复用同一个节点。

读者如果按教科书心智模型去数环长度，会数错。

#### 决策四：图里没有任何循环计数器

全项目 `src/` 下 grep `recursion_limit` **零命中**。ReAct 分支唯一的死循环保险是 LangGraph 的默认 `recursion_limit=25`，超限抛 `GraphRecursionError`。

真正需要步数预算的场景走 Planner 分支——那里有三重显式预算，第 6 章会讲。

### 2.4 动手试一试

**实验 A：让图自己画出来**（一行 REPL，最推荐）

```python
print(graph.get_graph().draw_mermaid())
```

把输出粘到任何支持 Mermaid 的地方（如 GitHub Markdown），就得到一张官方渲染的拓扑图。拿它跟本教程 0.4 的图对照，能立刻确认你有没有读错边。

**实验 B：数一数环转了几圈**

在 `graph.py:891` 之后插一行：

```python
print(f"[ROUTER] msgs={len(state['messages'])} tool_calls={[c.get('name') for c in getattr(last_msg,'tool_calls',[])]}")
```

这一行同时暴露三件事：环转了几圈、每圈调了什么工具、以及"循环终止就是 tool_calls 变空"。

**实验 C：亲手体会 reducer 的存在意义**

把 `state.py:29` 临时改成：

```python
messages: list          # 去掉 Annotated[..., add_messages]
```

跑一轮带工具调用的对话——会立刻炸。因为每个节点返回的 `{"messages": [response]}` 从"追加"变成了"整体覆盖"，历史消失，`tool_call` 和 `tool_result` 配不上对，服务端 400。

**做完记得改回来。** 这是理解 reducer 为什么存在最快的方式。

**实验 D：看完整 state**

一轮对话之后：

```python
print(graph.get_state(config).values)
```

44 个字段全打出来。特别对比 `planning_results` 的长度和 `plan["steps"]` 的长度，感受"除了 messages，其他都是被覆盖写的"。

### 2.5 踩坑与局限

**坑一：`interrupt()` 恢复时会把整个节点重跑一遍。**

这个第 7 章会详讲，但先埋个伏笔：小白的心智模型是"函数在 `interrupt` 那行暂停、原地续跑"，实际是"节点整体重放到 `interrupt` 处，取出 resume 值"。所以 `interrupt` 之前的代码会执行两次——必须是纯函数，或者至少无副作用。

**坑二：`build_graph` 的返回类型注解是错的。**

`graph.py:115` 写 `-> StateGraph`，但 `graph.py:928` 实际返回 `workflow.compile(...)` 的产物，是 `CompiledStateGraph`。而且还往这个对象上挂了两个属性（`graph.py:930-931`）：

```python
graph.memory_service = memory_service
graph.memory_repository = memory_repository
```

用途是退出时关 SQLite 连接（`main.py:307-320`）。Windows 上不关会导致临时目录删不掉，第 17 章会详讲。

**坑三：`sync_context` 这个名字在撒谎——它会写数据库。**

`sync_context_node`（`graph.py:213-265`）里第 252 行调了 `memory_service.extract_candidates_from_text(...)`，会把用户话里的偏好抽成候选**写进长期记忆库**。名字叫"同步上下文"，听起来是只读的，其实不是。

**这一章的局限**：ReAct 循环处理"打开客厅灯"这种单步任务很好。但用户说"把客厅灯打开，空调调到 26 度，再把窗帘关上"——模型要在一个循环里连续调 3 次工具，中间任何一次判断失误，后面全跑偏，而且**你无法在它动手之前看到它的完整意图**。

**下一章的问题**：在讲多步任务之前，我们得先把"手"和"工具箱"讲清楚——因为后面所有的可靠性机制（验证、审批、规划），全都建立在"设备状态是可信的、工具能力是明确的"之上。

---

## 第 3 章 Agent 的手：设备层与能力声明

### 3.1 要解决什么问题

第 1 章说"你要有一个真的能开灯的函数"。听起来很简单，直到你数一数这个项目有多少种设备：

灯、空调、电视、窗帘、加湿器、热水器、门锁、烧水壶（8 种**可控**设备）+ 温湿度传感器、人体传感器（2 种**只读**设备）。共 16 台实例。

每种设备有自己的动作（灯能调亮度，空调能调温度和模式，门锁只能锁/解锁），自己的合法范围（亮度 0-100，空调温度 16-30），自己的中文名。

**朴素做法会长成什么样？** 这个项目真实经历过：`tools/devices.py` 曾经是**669 行手写 `if action == ...` 的 if/elif**。而新增一种设备要同步改 **9 处**。

更糟的不是工作量，是**失败模式**。`capabilities.py:5-9` 的模块注释原话：

> 以前"新增一种设备"要手工改 9 处（models、simulator 默认实例、base 的 keywords_map、tools/devices 的 if/elif、tools/\_\_init\_\_、graph 的 device_tool_names、planning 的 DEVICE_ACTION_SPECS 与 PlanStep 的 Literal、mcp/server、scenes 的类型清单），其中工具实现的 if/elif 和 PlanStep 的 Literal **无法反射**，漏改一处的表现是"**Planner 第一版计划稳定失败，且不报错**"。

"稳定失败且不报错"——这是最坏的一种 bug。你看不到异常，只看到 Agent 变笨了。

`011-engineering-hardening.md:5-6` 把根因总结成一句话：

> 全部指向同一个根因：**信息与依赖分散在多份手工副本里，漏改不报错，错误延迟到运行期才爆发**。

### 3.2 代码怎么写的

设备层是四个文件，职责分得很干净：

| 文件 | 一句话职责 |
|---|---|
| `src/models.py` | 用 Pydantic + Enum 定义 10 种设备的**合法状态空间** |
| `src/devices/base.py` | 后端抽象接口 `DeviceBackend` + 门面 `DeviceRegistry` |
| `src/devices/simulator.py` | 内存字典实现的假硬件，含**确定性**环境推演 |
| `src/devices/capabilities.py` | **单一数据源**：8 种可控设备的全部声明 |

#### 3.2.1 `capabilities.py`：一条声明派生出十五样东西

先看一条**完整的真实声明**。烧水壶是最短的一个（`capabilities.py:718-760`），完整读一遍：

```python
DeviceCapability(
    device_type=DeviceType.KETTLE,                    # 718 起
    tool_name="control_kettle",
    device_label="烧水壶",
    tool_summary="控制智能烧水壶的开关与目标水温。",
    usage_examples=('"烧水" → action="boil"', ...),
    device_examples='如"厨房烧水壶"',
    not_found_text="❌ 找不到名为「{device_name}」的烧水壶设备。...",
    common_params=(
        ParamSpec("target_temp", int, 100, "目标水温 40-100°C"),   # 731-733
    ),
    actions=(
        ActionSpec("boil", "boil", "一键烧开（开机并加热到 100°C）",
                   expected=lambda a, d: {"power": True, "target_temp": 100},
                   handler=_kettle_handlers()["boil"]),
        ActionSpec("on",  "on",  "打开烧水壶", ...),
        ActionSpec("off", "off", "关闭烧水壶", ...),
        ActionSpec("set_temp", "set_temp(target_temp)", "设置目标水温（需配合 target_temp 参数）",
                   expected=..., handler=..., preference=PreferenceSpec(...)),  # 753
    ),
    default_devices=((KettleDevice, {"device_id": "kitchen_kettle", ...}),),   # 756-758
    scene_exit="power_off",                                                    # 759
)
```

**这条声明有 12 个字段，每个字段都在下游长出东西。** 关键的四个：

| 字段 | 长成什么 |
|---|---|
| `common_params` | 直接变成工具 JSON Schema 的一个 property + docstring 的一行 |
| `actions[].signature` | 喂给 Planner 的**合法值文本**（括号里是这个 action 要带的参数） |
| `actions[].expected` | `(参数, 设备) → 期望状态 dict`，**给验证器对账用**（第 6 章的核心） |
| `actions[].handler` | 真正的副作用，返回 `(结果文本, 生效后的参数)` |

`handler` 和 `expected` 是**两个独立的 lambda**：一个负责"做"，一个负责"说做完之后应该长什么样"。分开写是为了让验证器能独立判断成败，而不用相信 handler 的返回文本。

**完整的派生清单（15 项）**，从这一条声明自动长出来：

| # | 派生物 | 生成位置 | 消费方 |
|---|---|---|---|
| 1 | 工具 docstring（LLM 读的文本） | `tools/devices.py:38-49` | 模型 |
| 2 | 工具入参 JSON Schema | `tools/devices.py:52-66`（`pydantic.create_model`） | 模型 |
| 3 | 工具函数体 | `tools/devices.py:69-110` | ToolNode |
| 4 | 8 个 `control_xxx` 工具实例 | `tools/devices.py:234-236` | `bind_tools` |
| 5 | `CONTROL_TOOL_NAMES` | `capabilities.py:801` | 多智能体 device 角色（`graph.py:165`） |
| 6 | `CAPABILITIES_BY_TOOL` | `capabilities.py:803` | 测试反射 |
| 7 | `SCENE_EXIT_TYPES` | `capabilities.py:806-809` | 场景批量操作（`scenes.py:33-35`） |
| 8 | Planner 期望状态表 `DEVICE_ACTION_SPECS` | `planning.py:52-61` | 验证器 |
| 9 | Planner 合法工具名 `PLANNING_TOOL_NAMES` | `planning.py:64` | Planner prompt |
| 10 | Planner 合法 action 词表 `TOOL_ACTIONS` | `planning.py:65-68` | Planner prompt |
| 11 | **`PlanStep.tool_name` 的 `Literal`** | `planning.py:81` | structured output 的 JSON `enum` |
| 12 | 自动化允许的工具名 `AutomationToolName` | `automation/planning.py:14-16` | 自动化 Schema |
| 13 | `registry.find` 的类型关键词 | `capabilities.py:779-794` | 模糊查找（`base.py:183`） |
| 14 | 模拟器默认设备实例 | `simulator.py:233-248` | 启动时注册 16 台 |
| 15 | MCP 工具 | `mcp/server.py:81-84` | 外部 AI |

看第 4 项和第 14 项的代码量对比就懂了收敛的价值（`011-engineering-hardening.md:36-46`）：

- 控制工具：**669 行 if/elif** → 从声明生成（整个 `tools/devices.py` 现在 239 行）
- 模拟器默认设备：**140 行手写** → 9 行（`simulator.py:240-248`）

**现行规则（`CLAUDE.md:97-108`）：新增一种设备 = 1 处声明 + 2 处手工。** 手工的只剩：

1. `src/models.py` —— 数据模型 + `DeviceType` 枚举 + 字段范围约束
2. `src/agent/approval.py` —— 对外敏感动作（如解锁）的审批判定

#### 3.2.2 `DeviceRegistry`：精确查找 vs 模糊查找

这是设备层最需要理解的一组对比。

**精确查找** `get(device_id)`（`base.py:130-132`）：参数是 `living_room_light` 这种英文 ID，直接 `dict.get`。**给程序内部用**（并行子图、验证器）。

**模糊查找** `find(user_input, device_type)`（`base.py:142-191`）：参数是"客厅的灯"这种中文自然语言。**给 LLM 用**。三级策略：

| 策略 | 行号 | 规则 |
|---|---|---|
| 1 精确名匹配 | `base.py:164-167` | `device.name == user_input` |
| 2 字符包含（去虚词） | `base.py:169-177` | 过滤掉虚词后，要求**每个剩余字符**都出现在设备名里 |
| 3 类型关键词 | `base.py:179-191` | 命中关键词后，**若该类型有多台设备就返回 None** |

策略 3 的"多候选返回 None"是有意的（`base.py:186-189` 注释）：**多候选时拒绝猜测，让 Agent 向用户澄清**。

虚词表在 `base.py:38`：`的之那这台个只盏`。注释（`base.py:36-37`）解释了这张表为什么这么短：

> 这里只放纯虚词/量词，**绝不放可能作为设备名区分字的实义字**（如"室""厅"）。

放进去会让"卧室灯"和"客厅灯"互相误撞。

#### 3.2.3 `tick_environment()`：世界会动，但只在你说动的时候动

模拟器不是死的。空调开着制冷，温湿度传感器读数会跟着降。这个推演由 `tick_environment()` 驱动（`simulator.py:137-154`），**全部确定性、无随机数**（这样测试才能断言）。

实测效果（空调设 24°C 制冷 + 加湿器开着）：

```
初始:  客厅温湿度传感器: 温度 27.0°C | 湿度 42%
tick1: 温度 26.5°C | 湿度 44%
tick2: 温度 26.0°C | 湿度 46%
tick3: 温度 25.5°C | 湿度 48%
tick4: 温度 25.0°C | 湿度 50%
不 tick 连读三次: 25.0/50, 25.0/50, 25.0/50   ← 快照稳定
```

**关键约束：它只应该由"读环境"的入口调用。** 全项目只有 3 个调用点：`read_sensor`（`tools/devices.py:132`）、`get_device_status`（`tools/devices.py:196`）、并行查询子图的 dispatch（`parallel.py:49`）。

为什么不写在 `get()` 里（`simulator.py:141-146` 原话）：

> `get()`/`get_all()` 被场景模式、验证器等大量调用，如果每次读取都改状态，"读一下看看" 就会变成 "读一下顺便改了"，**验证器读到的值也会随调用次数漂移，非常难排查**。

并行子图那个调用点位置也是刻意的（`parallel.py:47-48`）：放在 `dispatch` 而不是每个 `query_device` 分支里，否则**并行分支的数量会直接改变读数**。

#### 3.2.4 `models.py`：用类型挡住模型的胡说

大模型会输出任何东西，包括 `brightness=120`。Pydantic 的 `Field` 约束是第一道防线：

```python
brightness: int = Field(default=80, ge=0, le=100, description="亮度百分比 0-100")   # models.py:249
temperature: int = Field(default=26, ge=16, le=30, description="目标温度 16-30°C")   # models.py:313
device_id: str = Field(..., pattern=r"^[a-z0-9_]+$")                                # models.py:194-198
```

有一个约束值得单独说：加湿器目标湿度下限是 **30**，不是 0（`models.py:397`）。

```python
target_humidity: int = Field(default=60, ge=30, le=80)
```

技术上 0% 是合法整数，但加湿器设 5% 毫无意义。**这说明 Schema 不只是类型检查，它是产品规格的一部分。**

还有 `frozen=True`（每个子类都有，如 `models.py:248`）——锁死 `device_type`，灯永远是灯，谁都改不成空调。

### 3.3 关键设计决策

#### 决策一：传感器只读，靠 6 处机制协同

传感器故意**不在** `CAPABILITIES` 里，只出现在 `SENSOR_DEFAULT_DEVICES`（`capabilities.py:768-775`）。这一个决定就让所有下游派生自动排除传感器：没有 `control_temp_humidity_sensor` 工具、Planner 词表里没有它、`PlanStep` 的 `Literal` 不含它。

但只靠这一处不够。完整的 6 处协同，以及**漏掉任何一处的具体症状**：

| # | 机制 | 位置 | 漏掉会怎样 |
|---|---|---|---|
| 1 | 传感器不在 `CAPABILITIES` | `capabilities.py:768-775` | 自动生成 `control_temp_humidity_sensor`，Agent 就能"打开温湿度传感器" |
| 2 | 提示词把传感器**单列成只读组** | `base.py:220-241`（判断在 `:232`） | Planner 给传感器下 `control_xxx` → Literal 拒掉 → **第一版计划稳定失败** |
| 3 | `PlanStep.tool_name` 的 Literal 不含 `read_sensor` | `planning.py:81` | 读取操作进了计划，但读取不改状态 → 验证器无从判断成败 |
| 4 | 场景批量用**正向白名单** | `scenes.py:33-35` + `:120-128` | 见下方"最难排查的坑" |
| 5 | 门锁离家用 `locked=True` 而非 `power=False` | `scenes.py:125-128` | 门锁变砖（见 3.5 坑五） |
| 6 | `tick_environment` 只由读入口调 | `base.py:214-217` | 读数随调用次数漂移 |

第 2 处的注释（`base.py:224-226`）把因果说得最直白：

> 执行器和传感器分开列出…**如果混在一张清单里，规划器会试图给温湿度传感器下"打开"指令。**

**第 4 处是本项目最难排查的坑，值得完整讲一遍。** 假设"离家模式"的实现是"遍历所有设备，全部关掉"。传感器被 `power=False` 之后，会发生一串连锁：

```
power=False
  → simulator.py:163-164 / :195-196 提前 return（不再推演环境）
  → models.py:531-532 / :570-571 的 to_status_text() 返回「⚠️ 离线」
  → read_sensor 读到离线
  → Agent 回答"读不到温度 / 家里没人"
```

用户看到的现象是**"Agent 说家里没人"**。你会去查人体传感器逻辑、查提示词、查模型——很难联想到根因是三天前写的"离家模式"。`CLAUDE.md:122` 原话：「很难联想到根因」。

现在的解法是**正向白名单**（`scenes.py:9-13`）：

> 只操作执行器，绝不碰只读传感器。批量关闭按 `devices/capabilities.py` 的 `scene_exit` 分组派生：新增设备类型时，只要声明了 `scene_exit`，"离家/睡眠"模式就会自动处理它，不用再手改这里的类型清单——**"离家模式"也因此永远不会把温湿度计关了**。

白名单比黑名单强的地方在于：传感器根本没声明 `scene_exit`，所以它**天然**不在集合里，不需要任何人记得排除它。

#### 决策二：`tick_environment` 不是抽象方法

`DeviceBackend` 有 6 个方法，5 个是 `@abstractmethod`，只有 `tick_environment` 是普通方法 + 空实现（`base.py:82-91`）。

理由（`base.py:87-90`）：真实后端（Home Assistant / MQTT）的传感器由硬件自行上报，不需要推演。**做成抽象方法会强制每个新后端都写一个空实现。**

这是"抽象基类该抽象到什么程度"的一个好例子：**抽象只应该覆盖所有实现都必须有的东西**。

#### 决策三：一致性由生成式测试钉住，不靠人记得

单一数据源解决了"改一处"，但怎么保证派生真的一致？答案是 `tests/test_capabilities.py` 的**生成式断言**：

| 测试 | 位置 | 钉住什么 |
|---|---|---|
| Literal / `PLANNING_TOOL_NAMES` / `DEVICE_ACTION_SPECS` 三者集合相等 | `test_capabilities.py:53-63` | 三份派生不许漂移 |
| `AutomationToolName == {*PLANNING_TOOL_NAMES, "set_alarm"}` | `:65-72` | 自动化 Schema 跟主 Schema 同源 |
| 每个类型都有查找关键词（含传感器） | `:74-84` | 新设备一定能被找到 |
| 每台默认设备都能在声明里找到出处 | `:86-94` | 模拟器不许凭空多设备 |
| Schema 覆盖声明参数、**`config` 绝不出现在 properties**、`turn_on` 被拒 | `:96-111` | 身份不泄漏 + action 名正确 |
| MCP 工具与图内工具**同源**（同设备同动作，结果逐字一致） | `:129-141` | 两条路径不许分叉 |

`capabilities.py:21-22` 的说法是：**漏任何一处都在测试阶段失败，而不是运行期静默。**

### 3.4 动手试一试

以下片段全部实测通过。**都不需要 API Key**——设备层完全不碰 `Settings`。

**实验 A：零配置看整个世界**

```bash
cd "G:/大厂学习/minimind/langgraph"
PYTHONIOENCODING=utf-8 F:/Software/Anaconda/envs/langgraph/python.exe -m src.main status
```

打出 16 台设备的分组状态报告。

**实验 B：精确 vs 模糊查找，亲手撞一次"拒绝猜测"**

```python
from loguru import logger; logger.remove()          # 关掉日志噪音
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.models import DeviceType

registry = DeviceRegistry(SimulatorBackend())
print('设备总数:', len(registry.get_all()))                    # 16

print(registry.get('living_room_light').to_status_text())
# → 客厅灯 (living_room_light): 🔴 关闭 | 亮度: 80% | 色温: 暖白

print(registry.find('客厅的灯', DeviceType.LIGHT).device_id)   # living_room_light（"的"被过滤）

# 三级策略的分界线 —— 注意这两行结果不一样
print(registry.find('灯',   DeviceType.LIGHT))   # 命中！返回客厅灯（见 3.5 坑一）
print(registry.find('灯光', DeviceType.LIGHT))   # None ← 3 台灯，拒绝猜测
print(registry.find('冷气', DeviceType.AC))      # None ← 2 台空调，拒绝猜测
```

**实验 C：让世界动起来**

```python
r = DeviceRegistry(SimulatorBackend())
print('初始:', r.get('living_room_th_sensor').to_status_text())

r.update('living_room_humidifier', power=True)
r.update('living_room_ac', power=True, temperature=24, mode='cool')

for i in range(1, 5):
    r.tick_environment()
    print(f'tick{i}:', r.get('living_room_th_sensor').to_status_text())

# 不 tick 就是稳定快照
for _ in range(3):
    s = r.get('living_room_th_sensor'); print(s.temperature, s.humidity)
```

**实验 D：一眼看完所有派生（证明"单一数据源"不是口号）**

```python
from src.devices.capabilities import CONTROL_TOOL_NAMES, SCENE_EXIT_TYPES, TYPE_KEYWORDS
from src.agent.planning import TOOL_ACTIONS, PlanStep
from src.automation.planning import AutomationToolName
from typing import get_args

print(CONTROL_TOOL_NAMES)                                   # 8 个 control_xxx
for k, v in TOOL_ACTIONS.items(): print(k, '->', v)         # Planner 看到的 action 词表
print(get_args(PlanStep.model_fields['tool_name'].annotation))
print(get_args(AutomationToolName))                         # = 上面 + set_alarm
for k, v in SCENE_EXIT_TYPES.items(): print(k, sorted(t.value for t in v))

# 传感器不在计划里 —— 这一行会抛 ValidationError
PlanStep(step_id=1, description='读湿度', tool_name='read_sensor', arguments={})
```

`TOOL_ACTIONS` 的实测内容（这就是 Planner 眼里的全部世界）：

```
control_light        -> on / off / set_brightness(brightness) / set_color(color)
control_ac           -> on(可带 temperature、mode) / off / set_temp(temperature) / set_mode(mode) / set_fan(fan_speed)
control_tv           -> on / off / set_volume(volume) / mute / set_channel(channel)
control_curtain      -> open / close / set_position(percentage)
control_humidifier   -> on / off / set_humidity(target_humidity) / set_mist_level(mist_level)
control_water_heater -> on / off / set_temp(target_temp)
control_lock         -> lock / unlock
control_kettle       -> boil / on / off / set_temp(target_temp)
```

**实验 E：把一条声明注释掉，看测试怎么骂你**

临时注释掉 `capabilities.py:400` 起 `CAPABILITIES` 里的任意一条声明，然后：

```bash
PYTHONIOENCODING=utf-8 F:/Software/Anaconda/envs/langgraph/python.exe -m pytest \
  tests/test_capabilities.py tests/test_sensors.py -q
# 正常时: 41 passed, 8 subtests passed in 2.00s
```

你会看到**多条断言同时失败**（`:53-63`、`:74-84`、`:86-94`）。这就是"漏改在测试阶段失败，而不是运行期静默"的样子。**做完记得改回来。**

### 3.5 踩坑与局限

**坑一：`find('灯')` 会静默返回第一台灯，"拒绝猜测"的覆盖面比想象的窄。**

策略 2（字符包含）**先于**策略 3（类型关键词）执行，所以它会绕过策略 3 的多候选保护。实测：

```
find('灯',   LIGHT) -> living_room_light   ← 3 台灯，策略2命中，静默返回注册顺序第一台
find('灯光', LIGHT) -> None                ← 策略2失败（"光"不在"客厅灯"里），策略3拒绝猜测
find('空调', AC)    -> living_room_ac      ← 2 台空调，同样静默选中客厅的
find('冷气', AC)    -> None
```

后果：用户说"把空调关了"会关**客厅**的，而不是问清楚。这不是 bug（"客厅"是合理默认），但**它跟"多候选拒绝猜测"这个说法给人的印象不符**，读代码时容易高估这层保护。

**坑二：`registry.update()` 的返回值被所有 handler 忽略——ReAct 分支会报告假成功。**

`update()` 失败时返回 `False`（`simulator.py:104-106` 捕获校验异常）。但所有 handler（`capabilities.py:158/163/167/172` 等）都不检查这个返回值，照样返回 `"✅ ...已打开"`。

实测三层防线的分工：

```
LightDevice(..., brightness=120)                      -> ValidationError（模型层拒绝）
registry.update('living_room_light', brightness=120)  -> False（状态不变，仍是 80）
control_light(action='set_brightness', brightness=120) -> "✅ 客厅灯亮度已调至 100%"（handler 先 clamp 到 100）
```

第三行是**正确的**（clamp 是设计意图）。但如果失败原因不是越界而是别的，工具层依然会说成功。唯一的安全网是 **Planner 分支的验证器**（读真实状态对账，第 6 章）；**ReAct 分支没有这层**，会把假成功直接告诉用户。

**坑三：`models.py` 里那批 clamp 验证器其实是死代码。**

作者自己在注释里承认了（`models.py:280-283`）：

> 注意：当前验证器使用 `field_validator` 的默认 `mode="after"`，会在字段的基础类型和 `Field` 约束验证之后执行。而 `brightness` 同时声明了 `ge=0`、`le=100`，**所以超出范围的值通常会先触发 Pydantic ValidationError，无法运行到这里**。

也就是说 `clamp_brightness`（`:259-293`）、`clamp_temperature`（`:317-321`）等一批验证器都跑不到。真正的截断在**工具层的 handler**：`capabilities.py:119-120` 的 `_clamp()` 和 `:123-138` 的 `_int_arg()`。

这是个诚实的例子：**代码里可以存在跑不到的防御，只要注释说清楚了，它就不是陷阱而是待办。**

**坑四：裸 `int()` 会掀翻整张图。**

`capabilities.py:126-130` 的注释记了这个根因：

> `expected_state_for_step()` 在 executor 的 **try 块之外**被调用，模型只要写出 `brightness="很亮"`，裸 `int()` 抛出的 `ValueError` 就会掀翻整张图。

解法是自定义异常 `InvalidPlanArgument`（`:141-146`），而且**绝不静默套默认值**：

> 也不会静默套用默认值把"调到很亮"悄悄执行成"调到 50%"再报告成功。

实测：

```python
step = {'tool_name': 'control_light',
        'arguments': {'device_name': '客厅灯', 'action': 'set_brightness', 'brightness': '很亮'}}
print(expected_state_for_step(step, registry))
# ('living_room_light', {}, "invalid argument: brightness 需要 0-100 之间的整数，收到 '很亮'")
```

**坑五：门锁的锁态不能塞进 `power`。**

`models.py:471-474` 原话：

> 把锁态塞进 `power` 会和其他执行器的"开机/关机"语义打架，**验证器读期望状态时也会误判**。

所以 `LockDevice.power` 默认 `True`（`:477`）表示"在线"，锁态是独立的 `locked` 字段。离家模式必须写 `locked=True` 而不是 `power=False`（`scenes.py:125-128`），否则门锁 `power=False` → 状态文本变"⚠️ 离线" → precheck（`capabilities.py:353-356`）此后拒绝一切操作 → **门锁彻底变砖**。

**坑六（本教程新发现）：`simulator.get_status_summary()` 里还有一份硬编码类型清单。**

`simulator.py:112-123` 有个 `type_order` 列表，`:125` 的循环只遍历它。所以如果你按"1 处声明"的说法只加 `CAPABILITIES`，新设备会正常注册、能被控制、能进计划，但**在 `get_device_status` 的报告里完全不显示**。

而 `tests/test_capabilities.py:86-94` 只断言 `registry.get(device_id)` 非空，**没有覆盖状态报告**，所以这个漏点不会被测试抓到。

**这是"1 处声明 + 2 处手工"故事里未被提及的第 3 处手工同步。** 同类问题还有 `read_sensor` 内部硬编码的 `type_map`（`tools/devices.py:117-121`），新增第 3 种传感器要手改，漏改的症状是"❌ 不支持的传感器类型"。

**坑七：`get_all()` 返回的是内部字典的引用，不是副本。**

`simulator.py:82-83` 直接 `return self._devices`。调用方（`scenes.py:119`、`context.py:93`、`parallel.py:24`）拿到的是**可写的活字典**，能绕过 `update()` 的校验直接篡改状态。目前没人利用这点，但它是个潜在的口子。（对比 `get_by_type()` 在 `:85-89` 返回新建字典。）

**坑八：`get_device_status` 的 `query` 参数是假的。**

`tools/devices.py:193-194` 写着 `_ = query  # 保留参数给未来扩展`，但 docstring（`:202`、`:205`）却告诉 LLM「也可以指定类型关键词来筛选，如"灯光"、"空调"」。模型按提示传了 `query="灯光"`，仍然拿到全量 16 台设备的报告。**docstring 与实现不一致，浪费 token 也可能让模型困惑。**（`list_scenes` 的同名参数更诚实：`scenes.py:214` 写"保留参数，当前未使用"。）

**这一章的局限**：传感器侧还没有单一数据源（新增传感器要改 5 处），`handler` 和 `expected` 是同一条声明里的两个独立 lambda，仍要人工保持对齐——例如空调 `on` 动作的 `expected`（`capabilities.py:477-481`）声明了 `power`/`temperature`/`mode`，但 handler（`:185-199`）还会写 `fan_speed`，所以验证器不校验风速。单一数据源解决了"跨文件副本"，**没解决"同一条声明内部两个 lambda 不一致"**。

**下一章的问题**：这 8 个工具怎么拿到 `registry`？如果用全局变量，测试会互相污染，后台任务会没有身份。

---

## 第 4 章 Agent 的工具箱：工厂与显式依赖注入

### 4.1 要解决什么问题

工具函数需要 `registry` 才能开灯，需要 `memory_service` 才能记偏好。这些依赖怎么进到工具里？

最省事的写法是模块级全局变量：

```python
# 曾经的写法
_registry = None
def set_registry(r):
    global _registry
    _registry = r
```

这个项目真的这么写过，直到踩了两类坑：

**坑 A：测试互相污染。** 测试 1 设了 registry，测试 2 忘了复位，于是测试 2 操作的是测试 1 的设备。这类 bug 表现为"单独跑过，一起跑就挂"，或者更糟——"一起跑能过，单独跑挂"。

**坑 B：后台执行器没有身份，而守卫拦不住。** 这一条值得完整看根因链（`defense-deep-dive.md` 4.9 节）：

```
RoutineExecutor 直接 tool.invoke(arguments)      # 后台执行，没有可信身份
  → LangChain 仍然注入一个 config 对象，但 configurable 为空 {}
  → record_preference_operation 里的守卫写的是 `if config is not None`
  → 空 configurable 不是 None，守卫失效
  → _context() 下标访问 config["configurable"]["home_id"] → KeyError
```

现象是定时例程执行时抛 `KeyError: 'home_id'`，而且**只有窗帘活下来**（窗帘那条动作恰好没声明偏好观察）。

`tools/memory.py:46-48` 把这个教训写成了注释，值得逐字记住：

> 曾经的实现靠 `if config is not None` 守卫，LangChain **总会注入空 configurable 的 config**，那个判断恒为真、一个字都拦不住。

### 4.2 代码怎么写的

011 的解法是**工厂函数 + 闭包**。唯一入口 `build_all_tools`（`src/tools/__init__.py:18-43`）：

```python
def build_all_tools(
    registry,
    *,
    memory_service=None,
    automation_runtime=None,
    external_tools=None,
    enable_preference_tracking: bool = True,
) -> list:
    tools = []
    tools += build_device_tools(registry, memory_service,
                                enable_preference_tracking=enable_preference_tracking)   # 32-36
    tools += build_scene_tools(registry)                                                  # 37
    tools += build_memory_tools(memory_service)                                           # 38
    tools += build_automation_tools(automation_runtime)                                   # 39
    if external_tools:
        tools += external_tools                                                           # 41-42
    return tools
```

文件注释（`__init__.py:8-9`）说明了一个重要性质：

> 返回的工具列表**顺序即图里 `bind_tools` 的顺序**；新增设备/工具只需改各子工厂内部的能力声明，本文件不再需要登记任何清单。

依赖怎么进到工具里？看 `build_device_tools`（`tools/devices.py:221-239`）的三层闭包：

```python
def build_device_tools(registry, memory_service=None, *, enable_preference_tracking=True):
    recorder = make_preference_recorder(memory_service, enable_preference_tracking)   # 233
    return [_make_control_tool(cap, registry, recorder) for cap in CAPABILITIES]      # 235
```

`_make_control_tool` 内部定义的 `_fn`（`:72-100`）直接引用外层的 `cap` / `registry` / `recorder` —— **这三个变量被闭包捕获**，工具自带依赖，不需要任何全局变量。

`devices.py:7-9` 的设计动机原话：

> **闭包注入（P1）**：工具不再通过模块级 `_registry` 单例拿依赖…这消除了"测试忘记复位单例"和"后台执行器无身份调用"两类隐患。

#### 4.2.1 一个工具从声明到被模型看见的完整链条

```
① capabilities.py:718-760   DeviceCapability 声明
        ↓
② tools/devices.py:38-49    _build_docstring(cap)   → 中文 docstring 字符串
   tools/devices.py:52-66    _build_args_schema(cap) → pydantic.create_model(...)
        ↓
③ tools/devices.py:72-100   定义闭包 _fn（捕获 cap/registry/recorder）
   tools/devices.py:102-103  _fn.__name__ / _fn.__doc__ 赋值
        ↓
④ tools/devices.py:104-110  StructuredTool.from_function(_fn, ..., infer_schema=False)
        ↓
⑤ tools/__init__.py:31-40   build_all_tools 汇总成 list
        ↓
⑥ agent/graph.py:162        llm.bind_tools(tools)   ← 转成 OpenAI function-calling JSON
   agent/graph.py:178-185   6 个角色各 bind_tools 一个子集（第 11 章）
        ↓
⑦ HTTP 请求 body 的 tools[] 数组 → 模型
        ↓
⑧ 模型回 tool_calls → ToolNode 按 name 查表（graph.py:161）→ .invoke(args)
```

**第 ④ 步的 `infer_schema=False` 是安全关键**，第 8 章会详讲。简单说：它禁止 LangChain 从函数签名反推 Schema，这才让 `_fn` 的签名里能安全地存在 `config: RunnableConfig = None`（`devices.py:75`）而**不暴露给模型**。注释（`devices.py:78-79`）原话：

> LangChain 依据签名注入可信身份，而 JSON Schema 里没有它（**模型看不到，也填不了**）。

#### 4.2.2 工具执行体的 6 步

`_fn` 的函数体（`devices.py:72-100`）：

| 步 | 行号 | 做什么 |
|---|---|---|
| 1 | `:80` | 用声明的 default 补全所有 `common_params`（模型没填的参数不会变成 `KeyError`） |
| 2 | `:81-83` | `registry.find` → 失败返回 `not_found_text.format(...)` |
| 3 | `:84-87` | action 查表 → 失败返回"不支持的操作"+ 支持列表 |
| 4 | `:88-91` | `precheck` → 非空字符串则拒绝（水箱空、门锁离线） |
| 5 | `:92` | `spec.handler(registry, device, args)` → `(text, effective)` |
| 6 | `:93-99` | **仅当**声明了 `preference` **且** `effective` 非空才记偏好 |

第 6 步的设计在 `capabilities.py:50-52` 有注释：`effective`

> 只在动作成功后非空，用于偏好观察（拿到 **clamp 后的真实值**，而不是模型写的原始值）；失败路径返回 `None`，**绝不记录偏好**。

实测四条路径：

```
set_brightness=120   -> ✅ 客厅灯亮度已调至 100%。      （步 5，handler 先 clamp）
action='turn_on'     -> ❌ 不支持的操作「turn_on」。灯光支持: on / off / set_brightness / set_color
device_name='浴室灯' -> ❌ 找不到名为「浴室灯」的灯光设备。当前可用的灯光有: 客厅灯、卧室灯、厨房灯。
门锁离线时 unlock    -> ❌ 玄关门锁离线，无法操作。      （步 4 precheck）
```

### 4.3 关键设计决策

#### 决策一：`enable_preference_tracking` 是构造期的显式选择，不是运行期的兜底

同一批工具有两条完全不同的调用路径，它们对"偏好观察"的需求正好相反：

| 路径 | 取值 | 位置 | 理由 |
|---|---|---|---|
| **图路径**（用户对话） | `True`（默认） | `graph.py:155-160` | 用户手动重复操作 → 应该学成偏好 |
| **后台自动化执行器** | `False` | `automation/executor.py:22-25` | 机器触发，不该计入"重复手动操作" |
| **MCP 服务器** | `False` | `mcp/server.py:81-84` | 无可信身份（`server.py:30-31` 注释） |
| **测试** | `False` | `test_capabilities.py:46` 等 | 免去造身份 |

`make_preference_recorder`（`tools/memory.py:39-64`）的两个分支：

- `enabled=False` 或 `service=None` → 返回 `_noop`（`:50-54`）
- 否则 → `recorder`（`:56-62`），先检查身份键（`home_id`/`user_id`/`thread_id`/`client_id`），**缺任一键直接 `raise RuntimeError`**

注意最后这一句：**缺身份是 fail-fast，不是静默跳过**。docstring（`:42-44`）强调这是「构造期的显式选择」，不是以前「逐键检查后安静跳过」的隐式兜底。

**为什么 fail-fast 比兜底好？** 因为"定时动作因缺身份被判失败"这类 bug 会立刻暴露，而不是静默吞掉、等你三个月后发现偏好库里少了一半数据。

#### 决策二：依赖为 `None` 时，工具整组消失

实测 `automation_runtime=None`（自动化未启用）时，`build_all_tools` 返回 **21 个工具**：

```
control_light, control_ac, control_tv, control_curtain, control_humidifier,
control_water_heater, control_lock, control_kettle,
read_sensor, get_device_status,
activate_scene, list_scenes,
save_personal_memory, save_home_rule, list_personal_memories, update_personal_memory,
delete_personal_memory, list_preference_candidates, confirm_preference_candidate,
reject_preference_candidate, list_memory_versions
```

**6 个自动化工具整组消失了。** 这是有意的（`011-engineering-hardening.md:61-62`）：

> 自动化未启用（`runtime=None`）时自动化工具**根本不出现在 Agent 面前**，比旧行为「调用时报尚未初始化」更安全。

差别在哪？旧行为下模型能看到工具、会去调、然后拿到一句"尚未初始化"的错误，接着可能重试、可能编个回答。新行为下模型压根不知道这个能力存在。**能力的存在性由构造决定，而不是由运行期的错误提示决定。**

（这个决定有个连锁后果，第 11 章会讲：自动化关闭时用户说"明天 6 点叫我起床"，会走进一条硬编码兜底文案的路径。）

### 4.4 动手试一试

**实验 A：看 LLM 眼中的工具箱**

```python
from loguru import logger; logger.remove()
import json
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.tools import build_all_tools

r = DeviceRegistry(SimulatorBackend())
tools = build_all_tools(r, enable_preference_tracking=False)
print(len(tools), [t.name for t in tools])                       # 21 个

by = {t.name: t for t in tools}
print(by['control_lock'].description)                            # 生成的 docstring
print(json.dumps(by['control_kettle'].args_schema.model_json_schema(),
                 ensure_ascii=False, indent=2))                  # 生成的 JSON Schema
```

`control_kettle` 的实测 Schema（对照 3.2.1 那条声明逐字段看）：

```json
{
  "properties": {
    "device_name": {"description": "设备名称，如\"厨房烧水壶\"", "type": "string"},
    "action": {"description": "boil: 一键烧开（开机并加热到 100°C） / on: 打开烧水壶 / off: 关闭烧水壶 / set_temp: 设置目标水温（需配合 target_temp 参数）", "type": "string"},
    "target_temp": {"default": 100, "description": "目标水温 40-100°C", "type": "integer"}
  },
  "required": ["device_name", "action"],
  "title": "control_kettleInput",
  "type": "object"
}
```

**注意 `action` 是 `type: string` 而不是 `enum`** —— 合法值只写在 description 里。这个细节直接导致第 5 章的一个真实 bug。

**实验 B：绕过 LLM 直接调工具**

```python
print(by['control_kettle'].invoke({'device_name': '厨房烧水壶', 'action': 'boil'}))
print(by['control_light'].invoke({'device_name': '客厅灯', 'action': 'turn_on'}))   # 被拒
print(by['read_sensor'].invoke({'sensor_type': 'temp_humidity', 'location': '客厅'}))
```

这是调试工具最快的方式——**不烧 token，不看模型脸色**。

**实验 C：亲眼看"工具整组消失"**

```python
a = build_all_tools(r, enable_preference_tracking=False)                     # automation_runtime 默认 None
print(len(a))                                                                # 21
print([t.name for t in a if 'routine' in t.name or 'alarm' in t.name])       # []
```

再对比启动带自动化的 CLI（`src/main.py`），在 trace 里看工具数量的变化。

**实验 D：撞一次 fail-fast**

```python
from src.tools.memory import make_preference_recorder
rec = make_preference_recorder(some_memory_service, True)    # 开启观察
rec({}, "lighting.brightness", 80)                           # 传空 config
# RuntimeError: ...（缺身份直接炸，不是静默跳过）
```

这一行的意义是让你**记住 fail-fast 的手感**：错误发生在最接近根因的地方。

### 4.5 踩坑与局限

**坑一：`set_registry` / `set_memory_service` / `set_automation_runtime` 已经删了。**

`CLAUDE.md` 明确写着「测试别再 import 它们」。如果你在网上/旧文档里看到这些函数，那是 011 之前的写法。

**坑二：多智能体的角色工具集不全是派生的。**

`graph.py:165` 的 device 角色从 `CONTROL_TOOL_NAMES` 派生（新增设备自动进入），但 `scene_tool_names`（`:167`）、`memory_tool_names`（`:168-172`）、`automation_tool_names`（`:173-177`）**都是手写字面量集合**。

`CLAUDE.md:91` 的警告：**新增非设备工具若不加进对应角色的名字集合，该角色就永远调不到它。** 这个坑第 11 章会详讲，症状是"工具明明存在，某个角色就是不用它"。

**坑三：`Settings` 在 import 时就可能抛 `ValueError`。**

`config.py:243-247` 的 `model_post_init` 里校验 API Key，空值或占位符直接 raise。所以任何走 `get_settings()` 的路径都需要 `.env`。

但设备层和工具层**完全不碰 settings** —— 这就是为什么 4.4 的所有实验和 `python -m src.main status` 都不需要 API Key。这个边界值得记住：**离模型越远的层，越容易单独测试**。

**这一章的局限**：工厂解决了"依赖怎么进来"，但工具的**能力边界**还是靠 docstring 里的文字约定（`action` 是 `type: string` 不是 `enum`）。下一章你会看到这个松散约定的代价。

**下一章的问题**：用户说"把客厅灯打开，空调调到 26 度，再把窗帘关上"——怎么让 Agent 先列出完整计划让人过一眼，而不是边想边做？

---

## 第 5 章 Planner：让 Agent 先说清要做什么

### 5.1 要解决什么问题

第 2 章的 ReAct 循环有个特点：**它是走一步看一步的**。

模型先想「我要开灯」，调 `control_light`，看到结果，再想「接下来调空调」，调 `control_ac`……每一步的决定都发生在上一步的结果出来之后。

这个特点在单步任务里是优点（灵活、省一次调用），在多步任务里变成三个具体的麻烦：

**麻烦一：不可预览。**

你说「把客厅灯打开，空调调到 26 度，再把窗帘关上」，ReAct 会先开灯，然后才决定下一步。**在第一个动作执行之前，谁也不知道后面还有什么。** 你想在动手前把整张清单看一眼——ReAct 结构上给不了你这个。

**麻烦二：不可验证。**

工具返回「✅ 客厅灯已打开」，模型就认为灯开了。但工具返回的是**它自己拼的字符串**——第 3 章 3.5 的坑里我们已经看到，`registry.update()` 的返回值被所有 handler 忽略，更新失败时工具照样回 ✅。

`docs/defense-script.md:333` 把这件事说得很直白：

> 因为 LLM 自评会说谎 —— 它会把"我调用了工具"当成"设备状态正确"。……**唯一的事实来源是设备状态，不是模型的自述。**

**麻烦三：不可收敛。**

失败了怎么办？ReAct 的答案是「让模型自己再想想」。模型可能重试同样的参数、可能换个说法再试、也可能试到你的 API 额度用完。没有任何机制规定「最多试几次」。

003 迭代文档里唯一一句明确点出失败模式的话（`docs/iterations/003-planner-executor-verifier.md:53`）：

> 达到限制后任务明确结束，**避免无限工具调用和无限重新规划**。

三个麻烦，一个解法：**把「想」和「做」拆开**。先让模型一次性生成完整计划（Planner），再由确定性代码逐步执行（Executor），每步执行完读真实设备状态对账（Verifier）。

| 痛点 | ReAct 的表现 | 本项目的对策 |
|---|---|---|
| 不可预览 | 走一步才知道下一步 | 计划一次生成完再整体审批（`graph.py:441-451`） |
| 不可验证 | 把「调了工具」当成「状态对了」 | `verify_step()` 读注册中心真实状态（`planning.py:197-256`） |
| 不可收敛 | 无限重试 / 无限反思 | 三个预算常量（`config.py:77-79`），超出即明确结束 |

**注意这不是「Planner 取代 ReAct」。** 两条路并存，由一个判定函数决定走哪条。这一章讲判定和 Planner，第 6 章讲 Executor 和 Verifier。

### 5.2 代码怎么写的

#### 5.2.1 谁决定走 Planner：一个纯正则函数

`should_use_planner()` 在 `src/agent/heuristics.py:56-71`。**它完全不调用 LLM**——判定用不着模型。

```python
def should_use_planner(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False

    # 预定义场景请求短路，一律不走 Planner
    if any(marker in normalized for marker in PLANNER_SCENE_MARKERS):
        return False

    action_count = sum(len(re.findall(p, normalized)) for p in PLANNER_ACTION_PATTERNS)
    device_kinds = sum(1 for keyword in PLANNER_DEVICE_KINDS if keyword in normalized)
    connectors = any(word in normalized for word in PLANNER_CONNECTORS)
    return action_count >= 2 and (device_kinds >= 2 or connectors)
```

一句话：**≥2 个动作词，且（≥2 类设备 或 出现连接词）**。

四张词表都在 `heuristics.py`：

| 词表 | 位置 | 内容 |
|---|---|---|
| `ACTION_CORE` | `:34` | `打开 / 开启 / 关闭 / 关掉 / 调到` —— 三处判定共享的基表 |
| `PLANNER_ACTION_PATTERNS` | `:49` | `ACTION_CORE` + `设为 / 调成 / 拉开 / 拉上 / 静音 / 切换`，共 11 个 |
| `PLANNER_DEVICE_KINDS` | `:51` | `灯 / 空调 / 电视 / 窗帘 / 加湿器 / 热水器 / 门锁 / 烧水壶`，8 类 |
| `PLANNER_CONNECTORS` | `:53` | `并且 / 然后 / 同时 / 再把 / 再将 / 以及 / 顺便` |
| `PLANNER_SCENE_MARKERS` | `:44-47` | `回家模式 / 离家模式 / 我要出门 / 我要睡 / 看电影 / 起床了` … |

三个计数方式**各不相同**，这是刻意的：

- `action_count` 用 `re.findall` 的**长度求和**——「打开客厅灯并且打开电视」里两个「打开」计 2。
- `device_kinds` 是**去重的类别数**——一句话里两盏灯只算 1 类。
- `connectors` 是**布尔**，出现即为真，不计数。

具体例子（前两条是 `tests/test_heuristics.py:25-26` 的原文断言）：

| 句子 | 动作 | 设备类 | 连接词 | 走 Planner？ |
|---|---|---|---|---|
| 关闭客厅灯，然后打开卧室空调到 25 度 | 2 | 2（灯、空调） | 然后 | ✅ |
| 打开客厅灯并且打开电视 | 2 | 2（灯、电视） | 并且 | ✅ |
| 打开卧室灯，再把灯调到 30 | 2 | 1（灯） | 再把 | ✅ 靠连接词过 |
| 关闭客厅灯 | 1 | 1 | 无 | ❌ 动作不够 |
| 打开客厅灯和卧室灯 | 1 | 1 | 无 | ❌ 只有一个「打开」，且「和」不在连接词表里 |
| 我要出门前关闭客厅灯，再把空调关掉 | 2 | 2 | 再把 | ❌ **场景标记短路** |

最后一条值得停一下：这句话客观上有两个动作、两类设备、一个连接词，三项全部达标，**本该走 Planner**，但因为含「我要出门」被短路了。把「我要」两个字去掉（`出门前关闭客厅灯，再把空调关掉`），同一句话立刻改走 Planner——这是实测的，5.4 的实验 A 会让你亲眼看到这个翻转。

这是作者明知会误判、仍然选择的保守。为什么，见 5.3。

> **顺便一个容易被忽略的细节**：动作词表里是「关闭 / 关掉」，**没有「关灯」**。所以口语化的「关灯」压根不计入 `action_count`——「我要出门前关灯，再把空调关掉」只算 1 个动作词（来自「空调关掉」），它走 ReAct 其实是**动作不够**，跟场景短路无关。读词表的时候，要读的是**字面量**，不是你脑子里的同义词。

**判定的调用点**：`src/agent/graph.py:327`

```python
use_planner = planning_enabled and should_use_planner(latest_text)
```

`graph.py:328` 之后的四个分支判断**全部带 `not use_planner` 前缀**。也就是说：**Planner 判定优先级最高**，一旦命中，意图分类的结果（RAG / 并行查询 / 澄清）全被压制。`PLANNING_ENABLED=false` 可以把整条分支关掉（`config.py:76`）。

#### 5.2.2 计划长什么样：两个 Pydantic 模型

`src/agent/planning.py:71-98`。

```python
class PlanStep(BaseModel):
    step_id: int = Field(ge=1)
    description: str = Field(min_length=1)
    tool_name: Literal[tuple(PLANNING_TOOL_NAMES)]   # :81
    arguments: dict[str, Any]

class ExecutionPlan(BaseModel):
    goal: str = Field(min_length=1)
    rationale: str = ""
    steps: list[PlanStep] = Field(min_length=2, max_length=8)   # :90

    @model_validator(mode="after")
    def normalize_step_ids(self):        # :92-98
        # 模型乱填 step_id 也不影响执行，强制重编为 1..N
        ...
```

三个约束各有理由：

- **`min_length=2`**：只有 1 步的任务本就不该走规划分支——那是 ReAct 的活。Schema 层直接拒绝。
- **`max_length=8`**：预算上限。运行期还有第二道截断（`graph.py:403-405` 按 `settings.planning.max_steps` 再切一次）。
- **`normalize_step_ids`**：模型经常把 step_id 写成 `[1, 1, 2]` 或 `[0, 1, 2]`。这个 validator 不报错，直接重编号——**能自动修好的就别麻烦用户**。

`tool_name` 的 `Literal` 是第 3 章那条派生链的终点：

```
CAPABILITIES (capabilities.py)           一条 DeviceCapability 声明
   → DEVICE_ACTION_SPECS (planning.py:52-61)
   → PLANNING_TOOL_NAMES (planning.py:64)
   → Literal[tuple(...)] (planning.py:81)
   → structured output 的 JSON Schema enum
```

注释（`planning.py:76-80`）解释了为什么它必须派生，以及一条刻意的排除：

> Literal 也从能力声明派生（P0）……structured output 靠它生成 JSON Schema 的 enum，从而在模型侧就约束住工具名；以前这里必须手写、靠一致性用例钉住，漏加时 Planner 明明该用新工具却被 pydantic 拒掉。现在新增设备自动带上，一致性由测试兜底。**传感器故意不在其中：规划分支只做写操作，read_sensor 不该出现在计划里。**

#### 5.2.3 这一章最重要的一个陷阱：`with_structured_output` 会丢掉工具语义

Planner 节点这样调模型（`graph.py:393`、`:400`）：

```python
structured_planner = llm.with_structured_output(ExecutionPlan)
plan = structured_planner.invoke(prompt)
```

注意：**不是 `bind_tools`**。这个选择带来一个非常反直觉的后果。

`bind_tools` 会把每个工具的 JSON Schema 和 **docstring** 一起发给模型，所以模型知道 `control_light` 支持 `on / off / set_brightness / set_color`。

`with_structured_output` 只发一份 Schema——`ExecutionPlan` 的 Schema。而 `ExecutionPlan` 里 `arguments` 的类型是 `dict[str, Any]`，**没有任何关于 action 取值的信息**。

`src/agent/planning.py:42-46` 的注释记录了这件事的代价：

> 为什么这份声明必须存在：Planner 走 `llm.with_structured_output(ExecutionPlan)`，不像 ReAct 分支那样 `bind_tools`，所以工具 docstring 里写的 "on / off / ..." **一个字都到不了模型面前**。模型只能凭常识猜 action，于是会写出 Home Assistant 风格的 turn_on / turn_off —— 这正是规划第一版反复失败的根因。把合法值显式喂给它，第一版就该是对的。

「Home Assistant 风格的 `turn_on`」——模型没有瞎猜，它是按业界最常见的智能家居 API 命名写的。**它不知道你这套用的是 `on`。**

`docs/defense-script.md:339` 把归因说得更准：

> **根因是"结构化输出丢失了工具语义"这件事本身，不是模型笨。**

#### 5.2.4 补救办法：把丢掉的语义手动塞回 prompt

既然 Schema 传不了 action 取值，就写进 prompt 文本。

**第一步，从能力声明派生一张动作表**（`planning.py:65-68`）：

```python
TOOL_ACTIONS: dict[str, str] = {
    tool_name: " / ".join(action.signature for action in spec.actions.values())
    for tool_name, spec in DEVICE_ACTION_SPECS.items()
}
```

`signature` 就是第 3 章 `ActionSpec` 里那个字段（`capabilities.py:88`），例如 `"set_brightness(brightness)"`。所以 `TOOL_ACTIONS["control_light"]` 渲染出来是：

```
on / off / set_brightness(brightness) / set_color(color)
```

`planning.py:63` 上方有一行注释：**「以下两个常量都是 DEVICE_ACTION_SPECS 的派生视图，不要手写。」**

**第二步，渲染成文本**（`planning.py:121`）：

```python
actions = "\n".join(f"  · {tool}: {spec}" for tool, spec in TOOL_ACTIONS.items())
```

**第三步，注入 prompt**（`planning.py:136-139`）：

```
1. 每一步只能调用一个工具，工具必须是 {', '.join(PLANNING_TOOL_NAMES)} 之一。
2. arguments.action 只能取下列合法值，括号内是该 action 需要附带的参数；
   不要使用 turn_on / turn_off 这类其他平台的命名：
{actions}
```

第 138 行那句「**不要使用 turn_on / turn_off 这类其他平台的命名**」是踩坑之后补的负例约束。这个细节很值得记住：

> **光给正例不够，还得把模型最可能猜错的那个写法直接否掉。**

Planner prompt 一共 7 条规则（`planning.py:135-144`），其余几条也各有来由：

| 规则 | 位置 | 为什么 |
|---|---|---|
| 4. 不使用 `activate_scene` | `:141` | 本分支专用于自定义多步骤目标；场景有自己的路径 |
| 5. 不添加用户没要求的设备操作 | `:142` | 防止模型"贴心"地顺手把窗帘也关了 |
| 6. 重新规划时针对失败原因调整，但保持原目标 | `:143` | 第 6 章的 replan 用 |
| 7. 只输出结构化 ExecutionPlan，不输出额外文本 | `:144` | 结构化通道不接受寒暄 |

prompt 还会注入设备清单（`:127`，`registry.get_device_list_prompt()`）、长期记忆（`:130`）、上次失败反馈（`:133`，第一次规划时是「无，这是第一次规划。」）。

### 5.3 关键设计决策

#### 决策一：为什么不用 `bind_tools`（既然它自带语义）

`bind_tools` 让模型「一次调一个工具」，这正是 ReAct 的形态——**它拿不到完整计划**。

两条路的取舍是清楚的：

| | `bind_tools` | `with_structured_output` |
|---|---|---|
| 拿到完整计划 | ❌ 一次一个工具 | ✅ 一次一整份 |
| 工具语义（docstring） | ✅ 自动带上 | ❌ **完全丢失** |
| 工具名约束 | 靠模型自觉 | ✅ JSON Schema enum 硬约束 |
| 参数形状约束 | ✅ 每个工具的 Schema | ❌ `dict[str, Any]`，无约束 |

本项目的选择是：**要完整计划，然后把丢掉的语义手工补回 prompt。**

代价你已经看到了——`TOOL_ACTIONS` 这套派生机制存在的唯一理由就是补这个洞。收益是 5.1 那三件事（可预览、可验证、可收敛）全都建立在「计划是一个完整对象」这个前提上。

#### 决策二：为什么预定义场景一律不走 Planner

`heuristics.py:40-43` 的注释给了两层理由：

> 预定义场景请求留在 ReAct + 场景审批路径。注意和 routing 的 scene_words 不同：这里匹配的是**整句惯用表达**（"我要出门"），routing 匹配的是**子串线索**（"离家"），宽窄不同是有意的——Planner 的排除必须精确，误伤一句"我要出门前关灯"就会把两个动作的请求送去走场景分支。

**业务理由**：「离家模式」已经是一个原子工具 `activate_scene`，有自己的审批通道。让 Planner 把它拆成 5 步，等于把一个已验证的原子操作降级成 5 个可能各自失败的步骤。

**工程理由**（就是上面那段注释）：排除表必须比路由表**窄**。宽了会误伤。

这里有个可以拿来锻炼判断力的观察：作者明知「我要出门前关灯，再把空调关掉」会被误判，仍然保留了短路。因为两种错误的代价不对称——**把场景请求错送进 Planner，会把一个可靠操作拆成多个不可靠步骤；把多动作请求错送进场景分支，最坏结果是模型多说一句「我不太确定你要哪个场景」。**

#### 决策三：判定函数为什么是纯正则，不问 LLM

三个理由，一个比一个实在：

1. **省一次调用。** 每轮对话都要判定，用 LLM 判定意味着每轮多一次往返。
2. **确定性。** 同一句话永远走同一条路。测试可以直接断言（`tests/test_heuristics.py`），不需要 FakeLLM。
3. **它本来就够用。** 「有几个动作词」「涉及几类设备」这种问题，正则的准确率不比模型低。

这条原则在项目里出现了三次（`should_use_planner`、路由兜底、`required_automation_tool`），011 迭代把它们收敛进了同一个文件 `heuristics.py`。`docs/gap-analysis.md:49` 附近把它列为「已经做对的」之一：

> **确定性分支刻意不问 LLM**……知道什么时候**不该**用模型，是 Agent 工程的成熟标志。

### 5.4 动手试一试

#### 实验 A：摸出判定阈值（不需要 API Key）

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -c "
from src.agent.heuristics import should_use_planner
cases = [
    '关闭客厅灯',
    '打开客厅灯和卧室灯',
    '打开客厅灯并且打开电视',
    '关闭客厅灯，然后打开卧室空调到25度',
    '打开卧室灯，再把灯调到30',
    '我要出门前关闭客厅灯，再把空调关掉',
    '出门前关闭客厅灯，再把空调关掉',
    '开启离家模式',
]
for c in cases:
    print(('走 Planner ' if should_use_planner(c) else '走 ReAct   '), c)
"
```

实测输出：

```
走 ReAct    关闭客厅灯
走 ReAct    打开客厅灯和卧室灯
走 Planner  打开客厅灯并且打开电视
走 Planner  关闭客厅灯，然后打开卧室空调到25度
走 Planner  打开卧室灯，再把灯调到30
走 ReAct    我要出门前关闭客厅灯，再把空调关掉      ← 场景标记短路
走 Planner  出门前关闭客厅灯，再把空调关掉          ← 少了「我要」，翻转
走 ReAct    开启离家模式
```

重点看第 6、7 两行：**同一句话，差两个字，走的是完全不同的路径**。这就是 `PLANNER_SCENE_MARKERS` 的威力，也是它的风险。

#### 实验 B：看到 Planner 只写不做

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m src.main --trace
```

先输入 `/status` 记下当前状态（比如客厅灯是关的、卧室空调是关的），再输入 `关闭客厅灯，然后打开卧室空调到25度`。你会依次看到：

```
planning_selected     ← reason: 请求包含多个自定义动作，交由 Planner 先出计划再执行
plan_generated        ← 带每一步的 tool_name 和 arguments
（暂停，等你确认）
```

`plan_generated` 事件携带每步的工具名和参数。`graph.py:419-420` 的注释解释了为什么要带：

> 带上每步的工具名和参数：这是"Planner 只写不做"最直观的证据 —— **此刻设备状态还没有任何变化，参数却已经全部定下来了**。

**现在输入 `n` 取消**，然后再敲一次 `/status`。两次输出一模一样——参数早就定好了，设备一个字节都没变。这就是「Planner 只写不做」。

> **为什么不能"另开一个终端查状态"**：设备模拟器是**纯内存**的（`simulator.py:62-67` 明写「程序重启后状态会重置为默认值」）。`python -m src.main status` 这个子命令会 `SimulatorBackend()` 新建一份默认状态（`main.py:585-586`），跟正在运行的那个进程毫无关系。想观察运行中的状态，只能在**同一个会话里**用 `/status`。
>
> 这个坑值得记住：一旦你把模拟器换成真实的 Home Assistant，"另开终端查"就又成立了。**状态住在哪里，决定了你能从哪里观察它。**

#### 实验 C：亲手制造 `turn_on` 那个坑

`tests/test_phase_seven.py:234-247` 已经把这个坑钉住了，你可以直接看它怎么骂人：

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -c "
from src.devices.simulator import SimulatorBackend
from src.devices.base import DeviceRegistry
from src.agent.planning import expected_state_for_step, TOOL_ACTIONS

registry = DeviceRegistry(SimulatorBackend())
# 模型写了 Home Assistant 风格的 turn_off
step = {'tool_name': 'control_light', 'arguments': {'device_name': '客厅灯', 'action': 'turn_off'}}
print(expected_state_for_step(step, registry))
print()
print('合法值:', TOOL_ACTIONS['control_light'])
"
```

实测输出（省略两行 loguru 初始化日志）：

```
('living_room_light', {}, 'unsupported action: turn_off（control_light 仅支持 on / off / set_brightness(brightness) / set_color(color)）')

合法值: on / off / set_brightness(brightness) / set_color(color)
```

注意返回的是**三元组** `(device_id, expected_state, preparation_error)`：设备找到了（`living_room_light`），但期望状态是空的 `{}`，第三项带着错误原因。这个三元组的每一项在第 6 章都有用。

输出里的错误信息会**带上合法值列表**——这不是巧合，是刻意的，5.5 会讲为什么。

#### 实验 D：看 prompt 里到底写了什么

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -c "
from src.devices.simulator import SimulatorBackend
from src.devices.base import DeviceRegistry
from src.agent.planning import planner_prompt

registry = DeviceRegistry(SimulatorBackend())
print(planner_prompt('关闭客厅灯，然后打开卧室空调', registry, '', ''))
"
```

把输出通读一遍。这就是模型看到的**全部**信息——没有工具 docstring，只有你手工写进去的这些。理解这一点，你就理解了 5.2.3。

### 5.5 踩坑与局限

**坑一：`int()` 在 try 块外面抛异常会掀翻整张图。**

`expected_state_for_step()` 的调用点是 `graph.py:469`，而 executor 的 `try` 从 `graph.py:471` 才开始。**中间那两行没有保护。**

模型只要写出 `brightness="很亮"`，裸 `int()` 抛的 `ValueError` 就会一路冒泡掀翻整个图。`capabilities.py:126-129` 的注释：

> 这里必须容错：`expected_state_for_step()` 在 executor 的 try 块之外被调用，模型只要写出 brightness="很亮" 这种值，**裸 int() 抛出的 ValueError 就会掀翻整张图**。转成 preparation_error 后会被判成确定性错误直接 replan —— 既不崩，也不会**静默套用默认值把"调到很亮"悄悄执行成"调到 50%"再报告成功**。

这段注释一次讲了两个坑，而且第二个更阴险：**静默套默认值会报告成功**。用户说「调到很亮」，系统调成 50%，然后回一句「✅ 已完成」——没人会发现。

所以项目单独立了一个异常类型（`capabilities.py:141-145`）：

> 单独立一个异常类型，是为了把"参数写错"和真正的程序 bug 分开：前者应该变成 preparation_error 回喂给 Planner 重写，后者才该往上抛。

**坑二：错误信息不带合法值，会白耗一轮重规划额度。**

`_unsupported_action()` 的 docstring（`planning.py:164-167`）：

> 重新规划时这条 feedback 会回喂给 Planner，所以带上合法值列表比只说 "unsupported action" 有用得多 —— 否则模型只能凭常识反推正确写法，**弱一点的模型可能改成 close/disable 又错一轮，白白耗掉重新规划额度**。

「改成 close 又错一轮」——这是真实发生过的。默认只有 1 次 replan 额度（`config.py:79`），错两轮就直接失败了。

**坑三：`max_steps` 是硬截断，可能截出半个计划。**

`graph.py:403-405` 的 `plan.steps[:max_steps]` 就是一刀切。作者自己承认这是缺口（`docs/defense-script.md:345`）：

> 会。`plan.steps[:max_steps]` 是硬截断，极端情况下目标只完成一半……严格来说应该在截断时显式告知用户"这次只安排了前 8 步"。**这是已知缺口。**

**坑四：`PLANNER_DEVICE_KINDS` 是手写的，不在第 3 章那条派生链里。**

`heuristics.py:51` 那 8 个词是手打的字面量。新增第 9 种设备时忘了加，**表现是降级而不是崩溃**：带连接词的请求仍能进 Planner，但「打开烧水壶 关闭热水器」这种无连接词的两动作请求会漏判、退回 ReAct。

`tests/test_heuristics.py` 只钉住了 `ACTION_CORE` 被三方复用，**没有把这张表和 `CAPABILITIES` 关联起来**。这是第 3 章「1 处声明 + 2 处手工」故事之外的第三处手工同步点。

**这一章的局限**：Planner 让计划变得可预览了，但到这里为止**它还只是一份文本**——谁来执行、执行完怎么知道真的做到了、做不到怎么办，一个都没解决。

**下一章的问题**：工具返回「✅ 已打开」，怎么确认灯真的亮了？

---

## 第 6 章 Executor 与 Verifier：不听模型自述，去查真实状态

### 6.1 要解决什么问题

第 5 章末尾留下的问题很具体：**工具返回的字符串不是事实。**

回顾第 3 章 3.5 的坑三——所有 handler 都忽略 `registry.update()` 的返回值：

```python
# capabilities.py 里 handler 的典型写法
lambda args, device, registry: (
    registry.update(device.device_id, {"power": True}),   # 返回值没人看
    f"✅ {device.name} 已打开",                            # 无条件返回成功
)[1]
```

更新失败了？工具照样回 ✅。ReAct 分支没有任何机制发现这件事——它只有工具返回的文本。

第二个问题是**失败之后怎么办**。「怎么办」有两个层次：

- 这次失败是**瞬时的**（超时、状态竞态）还是**确定性的**（计划本身写错了）？
- 如果是计划写错，重试同样的参数有意义吗？

这两个问题的答案完全不同，而 ReAct 把它们混在一起交给模型「自己想想」。

第三个问题是**控制流靠什么驱动**。ReAct 里模型说「我做完了」，循环就结束。如果模型判断错了呢？

本章讲的三样东西正好各解一个：

| 问题 | 解法 |
|---|---|
| 工具返回的不是事实 | `verify_step()` 执行后读注册中心真实状态对账 |
| 失败原因分不清 | `problem_type` 五分类，确定性错误跳过重试直接 replan |
| 控制流靠模型自述 | `planning_status` 显式状态机驱动全部条件边 |

### 6.2 代码怎么写的

#### 6.2.1 `planning_status`：一个字段驱动全部条件边

`src/agent/state.py:66-68`：

```python
planning_status: NotRequired[
    Literal["planning", "awaiting_approval", "executing", "completed", "failed", "cancelled"]
]
```

六个取值。`CLAUDE.md:58` 一句话概括了它的地位：

> `planning_status` 字段驱动全部条件边，**不靠 LLM 自述成败**。

完整的状态流转：

```
task_router (graph.py:311)
  │  should_use_planner → planning_status="planning"          [graph.py:384]
  ▼
planner (graph.py:390)
  │  with_structured_output(ExecutionPlan)                     [graph.py:393]
  │  → "awaiting_approval", current_step_index=0, revision+=1  [graph.py:434-438]
  ▼ 固定边（无条件）                                            [graph.py:852]
plan_approval (graph.py:441)
  │  interrupt(request)  ⏸ 图暂停，状态落 checkpoint            [graph.py:444]
  ├─ 批准 → "executing"                                        [graph.py:450]
  └─ 拒绝 → "cancelled" ────────────────────────┐              [graph.py:450]
  ▼                                              │
executor (graph.py:453)                          │
  │  执行 1 步，status 保持 "executing"           │            [graph.py:502]
  ▼ 固定边                                        │            [graph.py:862]
verifier (graph.py:506)
  │  verify_step() 读真实设备状态                              [graph.py:510]
  │
  ├─ 成功 且还有下一步 → index+1, retry=0, "executing" → 回 executor   [:538-545]
  ├─ 成功 且是最后一步 → "completed" ──────────┐                       [:545]
  └─ 失败 → deterministic?                     │                       [:554]
       ├─ 否 且 retry ≤ max → "executing" → 回 executor                [:556]
       └─ 是 或额度耗尽 → replan_count+=1                              [:566]
            ├─ ≤ max_replans → "planning" → 回 planner（带失败反馈）    [:568]
            └─ > max_replans → "failed" ──┐                            [:568]
  ▼                                        ▼
planning_finalize (graph.py:596) ← ────────┘
  │  planning_active=False                                             [:617]
  ▼
END                                                                    [:883]
```

三条条件边的位置：

| 路由函数 | 定义 | 读哪个字段 |
|---|---|---|
| `route_task` | `graph.py:825-834` | `planning_active`（不读 status） |
| `route_after_plan_approval` | `graph.py:854-855` | `planning_status` |
| `route_after_verifier` | `graph.py:864-872` | `planning_status` |

**注意 `"awaiting_approval"` 这个值没有任何条件边读它**——`planner → plan_approval` 是固定边（`graph.py:852`）。它纯粹是给 `/plan` 命令和外部观察者看的语义标记。

**整个循环里 LLM 只被调用一次**（`graph.py:400`，生成计划）。执行、验证、路由、重试、终止全是确定性代码。这是本章最值得记住的一句话。

#### 6.2.2 Executor：一步只调一个工具

`executor_node` 在 `graph.py:453-504`。骨架很朴素：

```python
step = steps[state["current_step_index"]]
device_id, expected_state, preparation_error = expected_state_for_step(registry, step)  # :469

try:                                     # :471
    if tool is None:
        tool_result = "❌ 未知工具"
    elif preparation_error:              # :474-475
        tool_result = f"❌ {preparation_error}"     # 工具根本不被调用
    else:
        tool_result = tool.invoke(step["arguments"])
except Exception as exc:
    tool_result = f"❌ {exc}"

return {"last_execution": {...}, "planning_status": "executing"}   # :494-502
```

两个细节：

**细节一：`expected_state_for_step()` 在 `try` 之外**（`:469` vs `:471`）。这就是 5.5 坑一的位置。

**细节二：`preparation_error` 会让工具根本不被调用**（`:474-475`）。计划写错的那类失败**在碰设备之前就被拦下**——零副作用地失败，然后重规划。

Executor 自己**不判断成败**。它只把「执行了什么、拿到什么」写进 `last_execution` 字段（`:494-501`），成败交给下一个节点。

#### 6.2.3 Verifier：对账的三行核心

`verify_step()` 在 `src/agent/planning.py:197-256`。docstring 第一句（`:204`）：

> Verify execution using the actual registry state, **not model self-report**.

判定顺序严格短路，**顺序本身就是设计**：

| 序 | 条件 | 行 | `problem_type` |
|---|---|---|---|
| 1 | `device_id is None`（设备名解析不出来） | `:206-211` | `device_not_found` |
| 2 | `preparation_error` 非空 | `:212-218` | `unsupported_action` |
| 3 | `tool_result` 以 `❌` 开头 | `:219-225` | `tool_error` |
| 4 | 执行后 `registry.get(device_id)` 为 None | `:227-234` | `device_not_found` |
| 5 | 逐字段比对有差异 | `:237-249` | `state_mismatch` |
| 6 | 全部一致 | `:250-256` | `none` |

对账的核心是这三行（`planning.py:235-241`）：

```python
actual = {name: _plain_value(getattr(device, name, None)) for name in expected_state}
expected = {name: _plain_value(value) for name, value in expected_state.items()}
mismatches = {
    name: {"expected": expected[name], "actual": actual[name]}
    for name in expected
    if actual[name] != expected[name]
}
```

三个要点：

**1. 只比 `expected_state` 里出现的字段**（`for name in expected_state`）。

这是「部分验证」。`set_volume` 的期望状态只声明了 `{"volume": ...}`（`capabilities.py:543`），所以只查音量，不管电源；而 `set_brightness` 声明了 `{"power": True, "brightness": ...}`（`capabilities.py:428-431`），会连电源一起查。

**期望状态里的字段集就是验证的范围**——写在第 3 章那条声明里。

**2. `getattr(device, name, None)`** —— 期望状态的 key 必须和设备对象的属性名同名。这是一条隐式契约，**写错 key 会静默拿到 `None` 然后判 mismatch**（不会报错说「你 key 写错了」）。

**3. `_plain_value()`**（`planning.py:259-260`）：`getattr(value, "value", value)`，把枚举拆成裸值再比。不做这一步，`DeviceType.AC` 和 `"ac"` 会被判不等。

`mismatches` 会原样进 `reason`（`:246`），所以 replan 反馈里能看到「期望 X，实测 Y」的逐字段明细。

#### 6.2.4 期望状态从哪来：第 3 章那条链的另一半

第 3 章讲了 `ActionSpec` 有 `handler` 和 `expected` 两个 lambda。现在看 `expected` 这一半怎么用。

`capabilities.py:59-62` 的类型注释说清了两者的分工：

> expected 签名：(arguments, device) -> 期望状态字典。与 handler 的差异：handler 写真实副作用，expected 只描述"做完之后设备应该是什么样"，**Verifier 拿它跟注册中心的真实状态比对。mute 这类翻转语义必须读 device。**

几个实例：

| action | 期望状态 | 位置 |
|---|---|---|
| `control_light` `on` / `off` | `{"power": True}` / `{"power": False}` | `capabilities.py:422-423` |
| `control_light` `set_brightness` | `{"power": True, "brightness": clamp(0,100)}` ← **顺带断言开机** | `:428-431` |
| `control_curtain` `open` / `close` | `{"position": 100}` / `{"position": 0}` | `:585-586` |
| `control_tv` `set_volume` | `{"volume": clamp(0,100)}` ← 只验音量 | `:543` |
| `control_tv` `mute` | `{"muted": not device.muted}` ← **唯一读 device 的翻转语义** | `:552` |

`expected_state_for_step()`（`planning.py:174-194`）负责把声明变成实际的期望字典，返回三元组 `(device_id, expected_state, error)`。五个分支：

| 分支 | 行 | 返回 |
|---|---|---|
| 工具名不认识 | `:180-182` | `(None, {}, "unsupported tool")` |
| 设备解析不出来 | `:183-185` | `(None, {}, "device not found or ambiguous")` |
| action 不在合法集里 | `:187-190` | `(device_id, {}, _unsupported_action(...))` |
| 参数解析失败 | `:191-194` | `(device_id, {}, str(exc))` |
| 正常 | `:192` | `(device_id, action_spec.expected(args, device), None)` |

#### 6.2.5 失败分流：这是整章最精妙的十行

`graph.py:548-556`。**注释比代码长，而且注释就是设计文档**：

```python
retry_count = state.get("step_retry_count", 0) + 1
max_retries = getattr(getattr(settings, "planning", None), "max_step_retries", 1)
# unsupported_action / device_not_found 是确定性错误：计划本身写错了，
# 用同一批参数原样重放不可能成功，只会白白耗掉重试额度、拖慢自愈。
# 直接跳到 replan 分支，把失败原因（已带合法值列表）交回 Planner 重写。
# tool_error / state_mismatch 可能是瞬时的（超时、状态竞态），仍然可重试。
deterministic = verification.problem_type in ("unsupported_action", "device_not_found")
if retry_count <= max_retries and not deterministic:
```

这十行背后是一个**归因分层**：

| 错误类别 | 谁错了 | 该在哪一层修 | 重试有意义吗 |
|---|---|---|---|
| `unsupported_action` / `device_not_found` | **计划错了**（Planner 的输出错） | Planner → replan | **无。** 同参数重放必然同样失败 |
| `tool_error` / `state_mismatch` | **执行环境有问题**（超时、竞态、设备离线） | Executor → retry | 有。可能是瞬时的 |

注意 `and not` 的写法（`:555`）：确定性错误**连一次重试都不做**。`retry_count` 虽然被 +1 写回（`:590`），但不影响本次分流。

`docs/gap-analysis.md` 把这个设计列为「已经做对的、改动时不要回退」的第二条：

> 识别出「同样参数重放不可能成功」这一点，比无脑重试高一个层次。

#### 6.2.6 replan 反馈：让重试有信息增量

失败后回喂给 Planner 的不是一句「失败了」。`graph.py:577-586`：

```python
feedback = (
    f"步骤 {execution['step']['step_id']}（{execution['step']['description']}）失败："
    f"{verification.reason}。工具结果：{execution['tool_result']}"
)
if verification.problem_type == "device_not_found":
    feedback += (
        "\n设备名未能解析。请仅从下列可用设备中选择，"
        "device_name 必须与设备名称逐字一致（不要添加'的''那台'等修饰）：\n"
        + registry.get_device_list_prompt()
    )
```

三层信息：

1. **无条件带上**：失败步骤的 id、原文 description、`verification.reason`（对 `state_mismatch` 而言就是逐字段的 expected/actual 明细）、工具返回的原始文本。
2. **`device_not_found` 额外追加**：完整可用设备清单 + 一条「逐字一致」的格式指令。
3. **`unsupported_action` 走另一条路**（更早注入）：`_unsupported_action()` 在 `expected_state_for_step` 阶段就把合法值拼进 `preparation_error`（`planning.py:169-171`），随后成为 `reason`，再被拼进 feedback。

回喂链路：`graph.py:593` 写进 state → `graph.py:398` 读出 → `planning.py:117` 形参 → `planning.py:132-133` 注入 prompt 的「上一次执行失败信息：」段。

测试钉住了这条链路（`tests/test_phase_seven.py:344`）：

```python
self.assertIn("device not found", fake.planner_prompts[-1])
```

**replan 时被清空和被保留的字段不一样**（画状态图时要注意）：

| 字段 | replan 时 | 为什么 |
|---|---|---|
| `current_step_index` | 归零（`:434`） | 新计划从第一步开始 |
| `step_retry_count` | 归零（`:435`） | 新计划的重试额度重置 |
| `plan_revision` | +1（`:438`） | v1 → v2 |
| `replan_count` | +1（`:566`） | 消耗全局额度 |
| `planning_results` | **跨版本保留**（`:529-536` 累加） | `/plan` 要能看到两版计划的完整轨迹 |

#### 6.2.7 三个预算常量

`src/config.py:73-79`（`env_prefix="PLANNING_"`）：

```python
enabled: bool = True
max_steps: int = Field(default=8, ge=2, le=12)
max_step_retries: int = Field(default=1, ge=0, le=3)
max_replans: int = Field(default=1, ge=0, le=3)
```

默认值意味着：**每步最多执行 2 次**（1 + 1 retry），**整个任务最多 2 版计划**（1 + 1 replan），**每版最多 8 步**。最坏情况 32 次工具调用 + 2 次 LLM 规划调用，**有界**。

读取处全部用双层 `getattr` 兜底（`graph.py:403`、`:549`、`:567`），settings 缺字段也不崩——这是为了让测试能用 `SimpleNamespace` 手搓 settings（第 17 章讲）。

### 6.3 关键设计决策

#### 决策一：为什么值得多一次 LLM 调用

`docs/defense-script.md:330`：

> 换来三件 ReAct 给不了的东西：执行前人能看到**完整**动作清单（ReAct 是走一步才知道下一步）、每步有客观验证、失败有预算化的重试和重新规划。所以我只在'多设备多动作'时付这个代价……而且故意写得保守。

「故意写得保守」对应的就是第 5 章那个判定函数——它宁可漏判，不肯多判。

#### 决策二：为什么不让 LLM 自评成败

已经引过一次，值得再看一遍（`docs/defense-script.md:333`）：

> 因为 LLM 自评会说谎 —— 它会把'我调用了工具'当成'设备状态正确'。……**唯一的事实来源是设备状态，不是模型的自述。** 反思的产物是结构化的 `VerificationResult`，里面带 `problem_type`，这个字段直接决定控制流走重试还是重新规划。

最后半句是关键：**`problem_type` 不是给人看的日志，它是控制流的输入。** 一个模型自述的字符串没法驱动条件边，一个五值枚举可以。

#### 决策三：`state_mismatch` 这个分类的价值

五个 `problem_type` 里，`state_mismatch` 是最能体现整套设计价值的那个：**工具报告成功，但设备真的没变。**

在 ReAct 分支里，这种情况**完全不可见**——工具回了 ✅，模型信了，用户也信了。只有「执行后再读一次真实状态」这个动作能把它揪出来。

第 3 章 3.5 坑三里那个「`registry.update()` 返回值被忽略」的问题，在 Planner 路径上被 Verifier 兜住了，在 ReAct 路径上没有。**这是两条路径真实存在的可靠性差异。**

### 6.4 动手试一试

#### 实验 A：看到 Verifier 的对账账本

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m src.main --trace
```

输入 `关闭客厅灯，然后打开卧室空调到25度`，批准计划，然后盯住 `step_verified` 事件里的两个字段（`graph.py:525-526`）：

```
step_verified  step_id=1  success=true
               expected_state={'power': False}
               actual_state={'power': False}
```

这两个字典就是账本。左边来自 `capabilities.py` 的声明，右边来自 `registry.get()` 的真实读取。

执行完输入 `/plan`，从 checkpoint 里把同一份轨迹再取一遍。`main.py:169-170` 的注释解释了为什么要有这个命令：

> 进度事件是'流过去就没了'，这个命令从 checkpoint 里把同一份轨迹再取出来，用于事后复盘：哪一步重试过、Verifier 比对的期望值和实测值分别是什么。

#### 实验 B：制造一次 replan（`device_not_found` 路线，最容易复现）

**为什么这条最容易**：默认有 3 盏灯（客厅灯 / 卧室灯 / 厨房灯，`capabilities.py:447-449`），所以任何不存在的灯名都会被 `registry.find` 的策略 3 拒绝猜测（第 3 章讲过：多候选时拒绝猜测）。

输入：

```
关闭书房灯，然后打开卧室空调
```

必走 Planner（2 动作 + 2 设备类 + 连接词「然后」），且必失败（「书房灯」三级策略全落空）。批准后观察事件流：

```
step_started      step_id=1
step_executed     tool_result=❌ device not found or ambiguous
                  ↑ 工具根本没被调用，这是 preparation_error 变成的文本
step_verified     success=false  problem_type=device_not_found
（注意：没有 step_retry 事件！）
replan_requested  replan_count=1  max_replans=1  accepted=true
plan_generated    ← v2 计划，设备名应该已被纠正
（第二次审批面板）
```

**「没有 step_retry 事件」就是「确定性错误跳过重试」的可观测证据**（`graph.py:554-555`）。

批准 v2 后 `/plan` 复盘：`plan_revision` = 2，`replan_count` = 1。

#### 实验 C：把额度调成 0，看明确失败

```bash
PLANNING_MAX_REPLANS=0 PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m src.main --trace
```

重跑实验 B。这次第一次失败之后直接 `planning_status="failed"`，输出确定性文案（`graph.py:607`）：

```
多步骤任务未能完成，已停止继续执行。最后失败原因：...
```

注意这句话不是模型生成的，是代码写死的。**终止路径上不调 LLM**——失败时最不该依赖的就是可能也在出问题的那个组件。

#### 实验 D：看到唯一能触发 `step_retry` 的场景

`state_mismatch` 在模拟器里很难自然发生（内存字典不会写失败）。测试用打桩造出来了（`tests/test_phase_seven.py:290-314`）：

```python
with patch.object(self.registry, "update", side_effect=flaky_update):
    # 第一次 update 返回 False，第二次正常
```

跑它并看断言：

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m pytest -q tests/test_phase_seven.py -k "retry" -v
```

断言是「step 1 出现**两条**结果记录，第一条失败第二条成功，最终 `completed`」——这是 `max_step_retries=1` 生效的样子。

### 6.5 踩坑与局限

**坑一：`mute` 的期望状态不幂等。**

`{"muted": not device.muted}`（`capabilities.py:552`）读的是**执行前**的状态。所以同一步重试时，期望值会**跟着翻转**。

作者自陈（`docs/defense-script.md:348`）：

> 有一个 —— 静音是取反语义……期望值依赖执行前的状态，所以**同一步重试时期望值会跟着翻转**。目前静音不在多设备规划的常见路径上，但这是设计上的真实局限，正确做法是把取反类动作在规划阶段就固化成绝对目标值。

后台自动化执行器有同款问题（`max_attempts=2` 的重试假设动作幂等）。

**坑二：`handler` 和 `expected` 是两个独立 lambda，靠人对齐。**

第 3 章末尾提过这一点，这里能看到具体后果：空调 `on` 的 `expected` 不含 `fan_speed`，但 `handler` 会写 `fan_speed`。**结果是验证器不校验风速**——风速写错了，Verifier 不会发现。

同一条声明里两个函数，没有任何机制保证它们描述的是同一件事。

**坑三：期望状态的 key 写错不会报错。**

`getattr(device, name, None)` 在属性不存在时返回 `None`，所以 key 打错（比如 `"powre"`）的表现是**永久 mismatch**，而不是「你 key 写错了」。debug 时会一直怀疑设备，不怀疑声明。

**坑四：ReAct 路径没有 Verifier。**

这不是 bug，是范围。但值得明确写下来：**只有 Planner 路径有对账**。单步请求「打开客厅灯」走 ReAct，工具回 ✅ 就是 ✅，没有第二次确认。

两条路径的可靠性不同，这件事对使用者是不可见的。

**这一章的局限**：状态机让执行过程可控、可验证、可收敛了。但有一个环节我们一直跳过没讲——`plan_approval` 那个「⏸ 图暂停」到底是怎么实现的？暂停期间程序在干什么？如果这时候进程挂了会怎样？

**下一章的问题**：怎么让一次「等用户点确认」的暂停，在进程重启之后还能接着跑？

---

## 第 7 章 人在回路：让暂停活过进程重启

前两章出现过两次「⏸ 图在这里停住，等你确认」，我们一路跳过没讲。这一章把它拆开。

「人在回路」（Human-in-the-Loop，常缩写 HITL）说的是：**在 Agent 动手之前，插入一个人的决定**。听起来简单，实现起来有个特别容易被低估的难点——**这个「等」要等多久，程序在等的时候是什么状态**。

### 7.1 要解决什么问题

#### 问题一：有些操作，错了就撤不回来

第 6 章的 Verifier 很有用，但它是**事后**的：先动手，再对账。对大部分操作这没问题——灯开错了关掉就行。

但有两类操作不是：

| 操作 | 为什么不能事后补救 |
|---|---|
| `activate_scene('离家模式')` | 一次调用改**多台**设备。错了要一台台恢复，而且你不知道改之前每台是什么状态 |
| `control_lock(action='unlock')` | 门开了就是开了。屋里有没有人、有没有别人进来，取决于这一次调用 |

第 002 号迭代方案（`docs/iterations/002-*.md:9`）就是为这个立的：批量场景一次改多台设备，且不可恢复。

#### 问题二：最朴素的写法是个陷阱

刚接触这个需求的人，十有八九会先写出这个：

```python
def approval_node(state):
    answer = input("确定要解锁吗？(y/n) ")      # ← 看起来完全能用
    if answer != "y":
        return {"approval_decision": "rejected"}
    return {"approval_decision": "approved"}
```

它在你自己的终端里跑，是能用的。但它有四个问题，从轻到重：

1. **图代码绑死了 CLI。** 这个节点里出现了 `input()`，意味着这张图永远只能跑在有终端的地方。想给它做个 Web 界面？改图。
2. **没法测。** 单元测试跑到这一行就挂住了，等一个永远不会来的键盘输入。
3. **暂停活在内存里。** 用户去接了个电话，这期间进程被 Ctrl-C 了、服务器重启了、部署滚动更新了——这次暂停连着前面所有对话历史一起消失。
4. **最根本的**：它把「等一个人」这件事，实现成了「阻塞一个线程」。这两件事的时间尺度差了好几个数量级。人可能 3 秒回答，也可能 3 小时后才回来。为了等 3 小时而占住一个进程，是不成立的。

作者对这一点的表述很直接：

> `input()` 阻塞在内存里，进程一死什么都没了；`interrupt()` 的暂停是**持久化**的，可以跨进程、跨机器恢复。

#### 问题三：拒绝之后，消息序列会坏掉

这个坑很隐蔽，几乎所有人第一次都会踩。

回想第 2 章的 ReAct 循环：模型发出 `tool_calls` → 工具执行 → `ToolMessage` 回喂 → 模型继续。现在用户点了「拒绝」，工具不执行了。那消息历史里就留下了一条**发出了 tool_call、却没有任何回执**的 AIMessage。

下一轮对话把这段历史再发给模型 API，**请求本身就是不合法的**。OpenAI 兼容接口会直接报错。

所以「拒绝」不是「什么都不做」。拒绝也必须**产生一条结果**。

#### 三个问题，三个对策

| 问题 | 对策 | 在哪 |
|---|---|---|
| 不可撤销的操作需要人批 | 敏感动作分类 + 风险等级 | `src/agent/approval.py:71-137` |
| 暂停必须能跨进程 | LangGraph `interrupt()` + checkpointer | `graph.py:444` / `graph.py:777` |
| 拒绝要产生合法的结果 | 为每条 tool_call 造一条错误 `ToolMessage` | `approval.py:152-162` |

### 7.2 代码怎么写的

#### 7.2.1 `interrupt()` 到底做了什么

先看最小形态：

```python
from langgraph.types import interrupt

def approval_node(state):
    decision = interrupt({"question": "确定要解锁吗？"})   # ← 图在这里停住
    return {"approval_decision": "approved" if decision else "rejected"}
```

`interrupt()` 这一行发生的事，跟 `input()` 完全不是一回事：

1. **把当前整张图的状态写进 checkpoint**（消息历史、计划、执行进度，全部）
2. **抛出一个特殊异常**，让这次 `graph.stream()` / `graph.invoke()` 提前返回
3. 返回值里带上 `__interrupt__` 字段，里面是你传给 `interrupt()` 的那个 payload

注意第 2 点：**函数返回了，栈退掉了，没有任何东西在等待**。进程可以去干别的，可以退出，可以被杀掉。

恢复的时候，调用方带着**同一个 thread_id** 再调一次，payload 换成 `Command(resume=...)`：

```python
graph.invoke(Command(resume={"approved": True}), config)   # config 里 thread_id 不变
```

LangGraph 从 checkpoint 把状态读回来，**重跑那个节点**，这一次 `interrupt()` 不再中断，而是**直接返回你 resume 传进去的值**。

> **`interrupt()` 必须有 checkpointer。** 这不是可选项——暂停是靠「存盘」实现的，没有存的地方就无法暂停。`build_graph` 里 `compile(checkpointer=...)` 的那个参数就是这么来的。

「重跑那个节点」这五个字，是本章最大的坑，7.5 会专门讲。

#### 7.2.2 两个中断点，两种策略

这个项目有两处 `interrupt()`，对应第 6 章那张五路径图里的两条路：

| 中断点 | 位置 | 触发条件 | payload 由谁造 |
|---|---|---|---|
| `approval` 节点 | `graph.py:769-787` | ReAct 主路：**看模型这次发的 tool_calls 里有没有敏感动作**，没有就不停 | `build_approval_request()`（`approval.py:71`） |
| `plan_approval` 节点 | `graph.py:441-451` | Planner 路径：**每次生成或修订计划后，无条件停** | `plan_approval_payload()`（`planning.py:148`） |

这个不对称是刻意的：

- ReAct 一次只做一件事，**按内容判断**够用了——「打开客厅灯」不该弹窗。
- Planner 一次可能做 8 件事，而且这 8 件事是**模型写的**。所以 `graph.py:852` 是一条**固定边** `planner → plan_approval`，没有条件判断，一律停。

看 `approval_node` 的完整实现（`graph.py:769-787`）：

```python
def approval_node(state: AgentState) -> dict:
    """Pause before executing a batch scene and wait for trusted approval."""
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", [])
    request = build_approval_request(tool_calls)
    if request is None:                                      # :774 不敏感 → 直接放行
        return {"approval_request": None, "approval_decision": "approved"}

    decision = interrupt(request)                            # :777 ⏸
    approved = approval_is_granted(decision)                 # :778
    logger.info("人工审批结果 | approved={} | tools={}", approved,
                [call.get("name") for call in tool_calls])
    return {
        "approval_request": request,
        "approval_decision": "approved" if approved else "rejected",
    }
```

注意 `:774`：**不敏感的调用也会走进这个节点**，只是立刻返回 `approved` 就出去了。路由函数 `router()`（`graph.py:889-894`）在决定去不去 `approval` 的时候，其实已经调了一次 `build_approval_request`：

```python
if build_approval_request(last_msg.tool_calls) is not None:
    return "approval"
```

所以这个函数被调用了两次，两次结果必须一致——它是**纯函数**，只读 `tool_calls`，这就是纯函数在这里的价值。

#### 7.2.3 什么算「敏感动作」

`build_approval_request()`（`approval.py:71-137`）认三类，写在 `:78-81` 的过滤器里：

| 类别 | 判据 | 风险等级 | 文案里写什么 |
|---|---|---|---|
| 批量场景 | `name == "activate_scene"` | `medium` | 场景名 + `SCENE_META` 里的官方描述 |
| 门锁解锁 | `_is_unlock_call()`（`:23`） | **`high`** | 设备名 +「（对外敏感动作）」 |
| 创建自动化 | `_is_automation_call()`（`:35`），4 个工具 | `medium` | 触发时间 + 每个动作的偏移分钟数 |

`unlock` 是**全项目唯一**的 `high`（`:97`）。而同一个工具的另一个 action `lock` **完全不需要审批**。`_is_unlock_call` 的 docstring（`approval.py:24-28`）解释了为什么：

> 解锁是"对外"的敏感动作（屋里有没有人取决于它），所以和批量场景一样需要人工确认后才能执行。**上锁（lock）是收拢安全边界，不需要审批。**

这句话值得抄下来。风险不在**操作对象**上（「锁」不危险），在**操作方向**上（「开锁」危险，「上锁」不危险）。

多个敏感动作同时出现时，风险等级**取最高，不取平均**（`:123`）：

```python
risk_level = "high" if "high" in risk_levels else "medium"
```

这也是 fail-safe 的一种形态：一批操作里只要有一个是 high，整批就按 high 提示。

#### 7.2.4 payload 为什么是 TypedDict

```python
class ApprovalRequest(TypedDict):
    """Serializable payload exposed by LangGraph ``interrupt``."""
    kind: Literal["tool_approval"]
    question: str
    risk_level: Literal["medium", "high"]
    summary: str
    tool_calls: list[dict[str, Any]]
```

这里没用 Pydantic，用的是 `TypedDict`。docstring 只有一个词是关键：**Serializable**。

这个 payload 要被写进 SQLite checkpoint，再原样交给调用方（可能是 CLI，也可能是个 HTTP 响应）。它必须是纯粹的 JSON 可序列化结构。`TypedDict` 在运行时**就是个普通 dict**，天然满足；Pydantic 模型还得 `.model_dump()` 一道。

`kind` 字段是给调用方分辨用的：`"tool_approval"`（ReAct 路径）还是 `"plan_approval"`（Planner 路径，`planning.py:154`）。同一个 interrupt 通道，两种 payload。

**一个特别容易看漏的细节**在 `:129-136`：

```python
"tool_calls": [
    {"id": call.get("id"), "name": call.get("name"), "args": call.get("args", {})}
    for call in tool_calls          # ← 注意：是 tool_calls，不是 risky_calls
]
```

遍历的是**全部** `tool_calls`，不是上面筛出来的 `risky_calls`。这决定了**审批粒度是整批**：

- 批准 = 这一批全做
- 拒绝 = 这一批全不做

没有「只批准其中第 2 条」这回事。7.5 会讲这个设计的代价。

#### 7.2.5 `approval_is_granted`：唯一的默认方向是「不批准」

从 CLI 传回来的东西可能长什么样？`True`、`{"approved": True}`、`"y"`、`"确认"`、`None`、或者某个前端框架塞过来的奇怪对象。全部收敛到一个 bool：

```python
def approval_is_granted(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return value.get("approved") is True          # :146 注意是 is True
    if isinstance(value, str):
        return value.strip().lower() in {"y", "yes", "true", "确认", "同意",
                                         "继续", "执行", "确定", "好"}
    return False                                      # :149 兜底 = 拒绝
```

实测各种输入（这就是 7.4 的实验 D）：

| 输入 | 结果 | 为什么 |
|---|---|---|
| `True` | ✅ | bool 直接用 |
| `{"approved": True}` | ✅ | 标准形态 |
| `{"approved": "yes"}` | ❌ | **不是 `True` 这个对象** |
| `{"approved": 1}` | ❌ | Python 里 `1 == True` 但 `1 is True` 是 False |
| `"y"` / `"确认"` / `"YES "` | ✅ | 白名单，会 strip + lower |
| `"n"` / `""` / `None` / `{}` / `1` / `[...]` | ❌ | 落到 `:149` |

`{"approved": 1}` 返回 False 这一条，第一次看会觉得是 bug。它不是。这里选 `is True` 而不是真值判断，理由是：**真值判断在这个位置是 fail-open 的。**

假如写成 `return bool(value.get("approved"))`，那么前端传 `{"approved": "no"}` 会怎样？非空字符串为真 → **批准**。用户明明点了「不」。

这类函数只有一个正确的默认方向。写代码的时候问自己：**「如果我完全看不懂传进来的这个东西，应该当成同意还是不同意？」** 答案永远是不同意。

#### 7.2.6 拒绝的正确姿势

回到 7.1 的问题三。拒绝之后必须给每条 tool_call 一个回执：

```python
def rejection_tool_messages(tool_calls: list[dict[str, Any]]) -> list[ToolMessage]:
    """Close every proposed tool call without executing it after rejection."""
    return [
        ToolMessage(
            content="用户未批准该操作，工具没有执行，任何设备状态都未改变。",
            tool_call_id=str(call.get("id", "unknown-tool-call")),
            name=call.get("name"),
            status="error",
        )
        for call in tool_calls
    ]
```

这几行同时干了两件事，缺一件都不行：

| 层面 | 做了什么 | 不做的后果 |
|---|---|---|
| **协议层** | 为每个 `tool_call_id` 补一条回执 | 下一轮 API 请求**不合法**，直接报错 |
| **语义层** | `status="error"` + 明确写「任何设备状态都未改变」 | 模型可能在后续对话里**假装已经执行了** |

第二点比第一点更值得琢磨。作者的说法：

> 消息序列里每个 `tool_call_id` 必须有对应的工具回执，否则下一轮请求不合法……模型也"知道"这件事没做成，**不会在后续对话里假装已经执行了**。

模型没有独立的记忆，它对「发生过什么」的全部认知就是这段消息历史。你在历史里写什么，它就相信什么。含糊的回执（比如空字符串）会让它自己编一个结论。

`tests/test_phase_six.py:97-115` 就是钉这件事的：拒绝之后断言设备**没变**、最终回复里有「取消」、且历史里**恰好有 1 条**含「未批准」的 ToolMessage。

路由部分（`graph.py:912-917`）：

```python
def route_after_approval(state) -> Literal["tools", "reject_tools"]:
    return "tools" if state.get("approval_decision") == "approved" else "reject_tools"
```

批准 → 真的执行；拒绝 → 去 `reject_tools_node`（`graph.py:789-796`）造回执。

**Planner 路径不需要这一套。** `route_after_plan_approval`（`graph.py:854-855`）只看状态：

```python
return "executor" if state.get("planning_status") == "executing" else "planning_finalize"
```

因为 `plan_approval_node` 在拒绝时写的是 `planning_status = "cancelled"`（`graph.py:450`），直接去收尾节点。**state 里根本没有悬空的 tool_calls**——Planner 走的是 `with_structured_output`，从头到尾没产生过 tool_call。

这是第 5 章那个技术选择的一个意外好处：选了 `with_structured_output` 丢掉了工具语义（那是代价），但也顺带绕开了整个「拒绝要补回执」的麻烦。**架构决策的后果总是成对出现的。**

#### 7.2.7 调用方怎么恢复

CLI 侧的完整逻辑只有 12 行（`main.py:292-304`）：

```python
def _invoke_with_approval(graph, state_input, config, view) -> dict:
    """Stream a graph run and resume any approval interrupts on the same thread."""
    view.reset()
    pending = _stream_segment(graph, state_input, config, view)
    while pending:                                        # ← 一轮对话可能停多次
        approved = _ask_for_approval(pending)
        pending = _stream_segment(
            graph, Command(resume={"approved": approved}), config, view
        )
    # 进度事件已经边跑边打了，最终状态从 checkpoint 读一次即可。
    return graph.get_state(config).values
```

四个要点：

1. **`while` 而不是 `if`。** 一次对话里可能连续遇到多个中断点——比如 Planner 出计划要批一次，计划里又含 `activate_scene` 还要批一次；replan 之后又是一次。
2. **第二次调用传的是 `Command(resume=...)`，不是新的 state。** 这是「继续」而不是「重新开始」。传新 state 会当成新一轮对话。
3. **`config` 必须是同一个**——里面的 `thread_id` 是找回 checkpoint 的唯一钥匙。
4. **最终状态从 `graph.get_state(config)` 读，不从返回值拿。** 因为中间可能停过好几次，每段 stream 的返回值都只是一个片段。

`_stream_segment`（`main.py:269-289`）的 docstring 点出了一个关键约束：

> 用 stream 而不是 invoke 是关键：只有 stream 才会把节点里 `emit_progress` 发出的 custom 事件交给调用方。**invoke 模式下这些事件会被 LangGraph 丢弃**，Planner / Executor / Verifier 的分工也就看不见了。

它同时监听两个通道（`main.py:280-288`）：

```python
for mode, chunk in graph.stream(payload, config, stream_mode=["custom", "updates"]):
    if mode == "custom":
        view.handle(chunk)                    # 进度事件 → 渲染
    elif mode == "updates":
        interrupts = chunk.get("__interrupt__")
        if interrupts:
            pending = interrupts[0].value     # 中断 → 交给上层去问人
```

`custom` 通道是进度，`updates` 通道里才有 `__interrupt__`。

#### 7.2.8 暂停凭什么活过重启

到这里所有零件都齐了，只剩最后一个问题：checkpoint 存在哪。

`src/memory/store.py:37-79`，三种情况：

| `db_path` | 结果 | 暂停能跨进程吗 |
|---|---|---|
| 有值，且初始化成功 | `SqliteSaver`（`:70`） | ✅ 能 |
| `None` 或 `""` | `MemorySaver` + 一行 warning（`:78`） | ❌ 只在同进程内有效 |
| 有值，但初始化失败 | **`RuntimeError`**（`:73-76`） | 直接崩，不启动 |

第三行是这个函数最重要的设计。docstring 一句话：

> SQLite is explicit: a configured path must initialize successfully.

为什么不静默降级成 `MemorySaver`？因为降级之后**一切看起来都是正常的**：图能跑、审批能弹、批准能执行。只有在进程重启之后，用户才会发现所有历史和所有待批准的暂停凭空消失了——而日志里只有一行 info 说「使用内存检查点」。

**你配置了持久化，系统没给你持久化，还不告诉你**——这是最坏的一类失败。宁可启动就崩。

（这条原则在第 15 章会再出现一次，措辞是「度量静默归零比没有度量更糟」。）

### 7.3 关键设计决策

#### 决策一：为什么必须是 `interrupt()` 而不是 `input()`

| 维度 | `input()` | `interrupt()` |
|---|---|---|
| 暂停存在哪 | 一个阻塞的线程栈 | checkpoint（SQLite 文件） |
| 进程死了 | 全丢 | 重启后能接着跑 |
| 谁负责「怎么问人」 | 图节点自己 | **调用方** |
| 能测吗 | 挂住 | `Command(resume={"approved": True})` |

第三行是最容易被忽略、但影响最深远的一行。

`interrupt()` 把「怎么问人」从图里挪了出去。图只负责说一句「**我需要一个决定，这是相关信息**」，至于怎么呈现、谁来点：

- CLI 用 Rich 画一个黄框（`main.py:255-266`）
- Web 就是一个按钮
- 企业微信就是一条卡片消息
- 测试直接 `Command(resume={"approved": True})`

**图代码一个字都不用改。** 这就是为什么 `ApprovalRequest` 里要放 `question` / `risk_level` / `summary` / `tool_calls` 四个字段——它们是给**任意**前端准备的原料。

#### 决策二：为什么审批的默认方向必须是「拒绝」

两种错误的代价严重不对称：

| 错误方向 | 后果 | 代价 |
|---|---|---|
| 该批准的判成拒绝 | 用户皱个眉，重说一遍 | 一次重试 |
| 该拒绝的判成批准 | 门开了 / 5 台设备被改了 | **不可撤销** |

所有默认值都朝代价小的那边倒。这个模式在项目里出现了至少三次：

| 位置 | 默认值 | 章节 |
|---|---|---|
| `approval_is_granted`（`approval.py:149`） | `False` = 不批准 | 本章 |
| `is_admin`（`context.py`） | `False` = 不是管理员 | 第 8 章 |
| `required_automation_tool`（`heuristics.py:136`） | `None` = 不强制 | 第 11 章 |

第三个尤其反直觉——一个「强制」机制，默认值居然是「不强制」。它的 docstring（`heuristics.py:136-142`）自己写着：

> **强制机制的默认值必须是"不强制"。**

因为强制错了（把查询请求判成创建请求）会让 Agent 去创建一个用户根本没要的自动化。同样是「错的方向决定代价」。

#### 决策三：为什么 ReAct 按内容停、Planner 无条件停

这条不对称如果反过来会怎样？

- **ReAct 也无条件停**：每次「打开客厅灯」都弹窗。用户第三次就会开始无脑点确认——**审批疲劳会让 HITL 退化成一个多余的回车键**。这比没有审批更糟，因为它给了虚假的安全感。
- **Planner 也按内容停**：那就得先扫一遍 8 个步骤，判断有没有敏感动作。但计划本身**就是模型刚写出来的、还没验证过的东西**——第 5 章讲过它可能把 action 写成 `turn_on`。在一个还没验证的结构上做安全判断，不靠谱。

所以：**动作已经具体到可以逐条检查时，按内容判断；面对一份还没验证的计划时，一律停。**

### 7.4 动手试一试

#### 实验 A：四个测试，四件事（不需要 API Key）

`tests/test_phase_six.py` 一共 4 个用例，正好覆盖本章四个核心行为：

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m pytest -q tests/test_phase_six.py -v
```

实测 `4 passed`。逐个看它们在断言什么（这比看源码快）：

| 测试 | 在钉什么 | 关键断言 |
|---|---|---|
| `test_scene_waits_for_approval_before_changing_devices`（`:73`） | 暂停时设备**真的还没动** | `:85-86` 断言灯和电视**仍然是开的**，然后 resume 之后才变成关的 |
| `test_rejection_closes_tool_call_without_changing_devices`（`:97`） | 拒绝路径完整 | `:108-115` 设备没变 + 回复含「取消」+ 恰好 1 条「未批准」ToolMessage |
| `test_single_device_control_does_not_require_approval`（`:117`） | 不敏感动作**不弹窗** | `:141` `assertNotIn("__interrupt__", result)` |
| `test_sqlite_checkpoint_resumes_after_graph_is_rebuilt`（`:144`） | **暂停能跨图对象恢复** | 见实验 B |

第一个测试的 `:85-86` 那两行特别值得看：

```python
self.assertTrue(self.registry.get("living_room_light").power)      # 暂停时还是开的
self.assertTrue(self.registry.get("living_room_tv").power)
```

这不是在断言「灯是开的」，而是在断言「**审批期间工具确实一次都没跑**」。这就是 7.1 问题一的机器化表述。

#### 实验 B：亲眼看到暂停跨进程存活

`test_phase_six.py:144-163` 这个用例，是整章最有说服力的一段。它的流程：

```python
first_graph = self._build_graph()
interrupted = self._start(first_graph)              # 跑到 interrupt，停住
self.assertIn("__interrupt__", interrupted)
self.assertTrue(self.registry.get("living_room_light").power)
close_checkpointer(first_graph.checkpointer)        # ← 把数据库连接关掉

second_graph = self._build_graph()                  # ← 全新的图对象
completed = second_graph.invoke(
    Command(resume={"approved": True}),             # ← 只传一个 resume
    self.context.to_config(),                       # ← 同一个 thread_id
)
self.assertFalse(self.registry.get("living_room_light").power)   # 执行了
self.assertIn("执行完成", completed["messages"][-1].content)
```

停下来想想 `second_graph` 知道些什么：它是刚 `build_graph()` 出来的，**内存里什么都没有**。用户说过什么、模型发了什么 tool_call、待批准的是哪个场景——全不知道。

它拿到的只有两样东西：一个 `thread_id`，一个 `{"approved": True}`。

然后它把整轮对话跑完了。

**所有上下文都是从 SQLite 里读回来的。** 把 `second_graph` 换成另一台机器上的另一个进程，逻辑完全一样。这就是「持久化的暂停」的含金量。

> 顺便注意 `:153` 和 `:163` 两处 `close_checkpointer(...)`。这不是洁癖——Windows 上 `TemporaryDirectory.cleanup()` 遇到没关闭的 SQLite 连接会抛 `PermissionError: [WinError 32]`，**测试断言全过也会判失败**。第 17 章会专门讲这个坑和它的执行顺序陷阱。

#### 实验 C：真实 CLI 里的两个风险等级（需要 API Key）

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m src.main --trace
```

依次试这四句，观察黄框的标题：

| 你说 | 预期 | 看什么 |
|---|---|---|
| `我要出门了` | ⏸ 弹窗，`risk_level = medium` | summary 里有场景的官方描述 |
| `把入户门锁打开` | ⏸ 弹窗，`risk_level = high` | 文案里有「（对外敏感动作）」 |
| `把入户门锁上` | **不弹窗，直接执行** | 这就是 7.2.3 那个非对称 |
| `打开客厅灯` | **不弹窗** | 单设备控制不是敏感动作 |

第 2、3 句是本章最该亲手跑一遍的对比：**同一个工具、同一台设备，两个方向，一个要批准一个不要。**

（如果第 3 句也弹窗了，说明模型把它翻译成了 `unlock`——那是模型的问题，不是审批的问题。看 `--trace` 里的 tool_calls 就知道。）

#### 实验 D：手动摸 fail-safe 的边界（不需要 API Key）

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -c "
from src.agent.approval import approval_is_granted as g
cases = [True, False, {'approved': True}, {'approved': 'yes'}, {'approved': 1},
         {}, 'y', '确认', 'YES ', 'n', '', None, 1, 0, ['approved']]
for c in cases:
    print(f'{str(c)!r:22} -> {g(c)}')
"
```

实测输出：

```
'True'                 -> True
'False'                -> False
"{'approved': True}"   -> True
"{'approved': 'yes'}"  -> False        ← 不是 True 这个对象
"{'approved': 1}"      -> False        ← 1 == True 但 1 is not True
'{}'                   -> False
'y'                    -> True
'确认'                  -> True
'YES '                 -> True         ← strip + lower
'n'                    -> False
''                     -> False
'None'                 -> False
'1'                    -> False        ← 数字不在任何分支里
'0'                    -> False
"['approved']"         -> False
```

**练习**：把 `approval.py:146` 的 `is True` 改成 `bool(...)`，重跑这个脚本，看哪几行翻转了。然后想一想：如果前端传的是 `{"approved": "no"}`，改之后会发生什么？

（改完记得改回来。）

### 7.5 踩坑与局限

#### 坑一：含 `interrupt()` 的节点会**从头重跑**，所以必须幂等

这是本章最重要的一个坑，也是 LangGraph HITL 最常见的 bug 来源。

回想 7.2.1 说的恢复机制：**LangGraph 重跑那个节点**，只是这次 `interrupt()` 直接返回 resume 值。作者的原话：

> 节点会从头重跑，`interrupt()` 这次直接返回 resume 值而不再中断。所以**含 interrupt 的节点必须幂等**……如果在 interrupt 之前写了库或调了工具，那部分会被执行两次。

看 `approval_node` 在 `interrupt()` **之前**做了什么（`graph.py:771-775`）：

```python
last_msg = state["messages"][-1]                  # 读
tool_calls = getattr(last_msg, "tool_calls", [])  # 读
request = build_approval_request(tool_calls)      # 纯函数
```

**全是读，没有一次写。** `plan_approval_node`（`graph.py:443`）同样：

```python
request = plan_approval_payload(state["plan"])    # 纯函数
decision = interrupt(request)
```

这不是巧合，是纪律。如果你在这里加一行「先记一条审批日志到数据库」，那条日志会出现**两次**。

顺着这条纪律还能发现一个细节：这两个节点都**没有** `@traced_node` 装饰器（第 15 章会讲这个装饰器），而其他关键节点都有。因为给一个会重跑的节点计时，测出来的数字没有意义。

> **自查清单**：往任何含 `interrupt()` 的节点里加代码之前，问一句——「这行如果跑两次，会怎样？」

#### 坑二：审批粒度是整批，没有「部分批准」

7.2.4 提过 payload 里遍历的是全量 `tool_calls`。所以如果模型一次发出三个调用，其中一个是 `unlock`：

```
[control_light(客厅灯, on), control_ac(卧室空调, on), control_lock(入户门锁, unlock)]
```

你只有两个选择：**三个全做**，或者**三个全不做**。想「只开灯和空调、别开锁」，做不到——只能拒绝，然后重新说一遍。

这在当前实现里是可接受的（模型很少一次发多个调用），但如果换成一个更爱并行调工具的模型，就会变成真实的体验问题。

#### 坑三：Planner 路径的 unlock 被降级成 medium，而且展示的是模型自己写的文案

这是本章最需要警惕的一条，也是整个审批设计里最真实的缺口。

`plan_approval_node`（`graph.py:441`）只调 `plan_approval_payload`，**从不调用 `build_approval_request`**。而 `plan_approval_payload`（`planning.py:148-159`）里：

- `risk_level` 硬编码 `"medium"`（`:156`），跟计划里有没有 `unlock` 完全无关
- `question` 是把模型写的 `step["description"]` 拼起来的（`:151`）

而 `control_lock` **是可以被 Planner 规划的**（它在 `PLANNING_TOOL_NAMES` 里）。再加上 CLI 的 `_ask_for_approval`（`main.py:255-266`）只渲染 `payload["question"]`、从不展示底层 `tool_calls`，三者叠加的实测结果是：

```
CLI 展示给用户的全部内容：
  已生成 2 步执行计划，是否开始执行？
  1. 打开客厅灯
  2. 准备门口环境          ← 模型措辞完全没提"解锁"

实际会执行：
  control_light({'device_name': '客厅灯', 'action': 'turn_on'})
  control_lock({'device_name': '入户门锁', 'action': 'unlock'})
```

用户批准的是「准备门口环境」，实际发生的是开锁。

> **这条的原则性教训**：HITL 展示的内容必须是**机器从 `tool_calls` 生成的事实**，不能是模型撰写的摘要。否则「人在回路」批准的是模型的**说法**，而不是模型的**行为**——这恰好废掉了 HITL 的全部意义。风险等级同理，必须从实际动作推导，不能硬编码。

修法不难（`plan_approval_payload` 里对每步调一次 `build_approval_request` 的判据、拼出机器文案、聚合风险等级），但它是个**设计缺陷**而不是笔误——两条路径各自造 payload，从一开始就没共用敏感动作判据。

#### 坑四：绕过图的入口，自动绕过审批

审批实现在**编排层**（图节点里的 `interrupt()`）。而 `src/mcp/server.py` 是**另一个进程入口**，根本不经过图。

```
control_lock(action='unlock')      →  需要审批 risk=high     ← 图内
control_lock_mcp(action='unlock')  →  无任何拦截             ← MCP 入口
```

对 `src/mcp/server.py` 全文搜 `interrupt` / `approval`：**零命中**。第 16 章会详细讲这个入口。

**根因不是漏写，是架构层次**：安全边界必须在**每个入口**重复，或者下沉到所有入口共用的那一层。「在编排器里做鉴权」是分布式系统里的经典反模式。

#### 坑五：审批绑定「工具名」，不是「效果」

`approval.py:78` 的过滤器按**工具名**判断。于是等效操作的审批状态不一致：

```
activate_scene('回家模式')                            →  需要审批 medium
control_light + control_ac + control_curtain（等效）  →  *** 零审批直接执行 ***
```

**这不需要恶意提示词。** 用户说「帮我把客厅调成回家的样子」时，模型**自然可能**选单设备路径——审批就静默消失了。

审批锚定的是「模型挑了哪个工具」，不是「产生了什么物理效果」。

（澄清：`control_lock(action='lock')` 不需审批是**正确的刻意设计**，那是安全方向。这里说的是另一回事。）

---

**这一章的局限**：坑三到坑五指向同一个方向——审批机制本身实现得很干净（fail-safe、可持久化、前后端解耦），但它的**触发判据**和**呈现内容**还不够严密。这是学 HITL 时最值得记住的：难点从来不在「怎么暂停」，而在「暂停的时机对不对」和「给人看的东西是不是事实」。

**下一章的问题**：现在我们有一个人来批准危险操作了。但——**批准的这个人，凭什么证明他是他？** 更要紧的是：模型能不能自己说一句「我是管理员，家庭 ID 是 home-b」，然后就去改别人家的设备？

---
