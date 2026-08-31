[← 第 1 章 Agent 到底是什么](01-Agent是什么.md) · [目录](README.md) · [第 3 章 Agent 的手：设备层与能力声明 →](03-设备层与能力声明.md)

---

# 第 2 章 最小可用：LLM + 一个工具 + 循环

## 2.1 要解决什么问题

第 1 章的三个零件要落成真代码，得回答几个具体问题：

- 循环写在哪？用 `while True` 吗？
- 对话历史存在哪？进程重启就丢了吗？
- 模型说"要调 3 个工具"的时候，谁负责挨个执行？

LangGraph 给的答案是：**把 Agent 画成一张图**。节点是"要做的事"，边是"下一步去哪"，图自带一个状态字典在节点间传递。

## 2.2 代码怎么写的

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

## 2.3 关键设计决策

### 决策一：状态里只有 `messages` 有 reducer，其余 41 个字段是"后写覆盖"

`AgentState`（`src/agent/state.py:16-72`）一共 42 个字段，但**只有一个带 reducer**：

```python
messages: Annotated[list, add_messages]      # state.py:29
```

其余 41 个全是裸 `NotRequired[...]`，语义是**后写覆盖前写**（last-write-wins）。

这件事必须讲清楚，因为有两个字段**看起来**像累加器，其实不是：

- `planning_results`（`state.py:72`）：累加是在 `verifier_node` 里**手写**的——先 `list(state.get("planning_results", []))` 拷一份，`.append(...)`，再整体覆盖写回（`graph.py:529-536`、542）。不是 reducer 干的。
- `parallel_query_results`（`state.py:43`）：一次性整体写入（`graph.py:634`）。

### 决策二：`add_messages` 不只是 append —— 它还能删

这是小白 100% 会误解的地方。看到 `Annotated[list, add_messages]`，直觉是"节点返回的消息会被追加到列表末尾"。实际上 `add_messages` 有三种行为：

| 节点返回什么 | `add_messages` 干什么 |
|---|---|
| 一条新消息 | 追加到末尾 |
| 一条**已有 id** 的消息 | **替换**掉原来那条 |
| 一个 `RemoveMessage(id=...)` | **删除**对应消息 |

第三条是项目控制历史膨胀的手段：`compact_context_node` 把 `RemoveMessage` 对象混在返回值里（`graph.py:307-308`），reducer 看到就按 id 删。

**这意味着裁剪不是"这一轮少发几条"，而是真的从 SQLite 里删数据**——删了就再也读不回来。第 12 章会讲这个设计带来的一个真实 bug。

### 决策三：工具执行完回 `compact_context`，不是回 `agent`

教科书 ReAct 是 `tools → agent`。本项目是 `tools → compact_context → agent`（`graph.py:922`）。

理由（`defense-deep-dive.md` 1.4 节）：**只有工具结果回流时消息才会膨胀**。把压缩放在回路上而不是入口，用户新一轮输入和工具结果回流就能复用同一个节点。

读者如果按教科书心智模型去数环长度，会数错。

### 决策四：图里没有任何循环计数器

全项目 `src/` 下 grep `recursion_limit` **零命中**。ReAct 分支唯一的死循环保险是 LangGraph 的默认 `recursion_limit=25`，超限抛 `GraphRecursionError`。

真正需要步数预算的场景走 Planner 分支——那里有三重显式预算，第 6 章会讲。

## 2.4 动手试一试

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

42 个字段全打出来。特别对比 `planning_results` 的长度和 `plan["steps"]` 的长度，感受"除了 messages，其他都是被覆盖写的"。

## 2.5 踩坑与局限

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

[← 第 1 章 Agent 到底是什么](01-Agent是什么.md) · [目录](README.md) · [第 3 章 Agent 的手：设备层与能力声明 →](03-设备层与能力声明.md)
