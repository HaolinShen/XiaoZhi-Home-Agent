[← 第 9 章 意图路由：什么时候不该用 LLM](09-意图路由.md) · [目录](README.md) · [第 11 章 多智能体：安全边界建立在看不见上 →](11-多智能体.md)

---

# 第 10 章 并行子图与 Send

这一章讲怎么用一个**独立的小图**处理"一次要查很多设备"的请求，以及 LangGraph 里最容易撞的那面墙——多个分支同时往一个状态字段里写，会直接报错。

## 10.1 要解决什么问题

第 9 章的 `task_router` 把请求分了五条路。其中一条叫 `parallel_query`，进来的是这种句子：

> 客厅和卧室的灯都开着吗？空调多少度？

这句话要读的设备不是一个，是**一批**。而且"一批"有多大，写代码的时候你不知道——用户说"客厅"是 8 个设备，说"客厅和卧室"是 11 个，说"所有设备"是 16 个。数量由**运行时的用户输入**决定。

先看最朴素的两种写法为什么都不好。

**写法一：交给 ReAct 主路，让模型自己一个一个查。** 模型调 `get_device_status`，看结果，再调，再看。11 个设备就是 11 轮"模型 → 工具 → 模型"。每一轮都是一次真实的 HTTP 请求打到大模型上，而这 11 次调用之间**没有任何依赖**——查客厅灯的结果不影响你怎么查卧室空调。花 11 轮 token 去做一件本来可以一次做完的事，纯浪费。更糟的是模型可能查到第 7 个就觉得"差不多了"，开始编剩下的。

**写法二：在图里画死。** 那就不用模型了，画一条链：

```
查客厅灯 → 查卧室灯 → 查客厅空调 → 查卧室空调 → ...
```

问题是这条链得画多长？你得为"用户可能提到的设备组合"预先画出所有分支。新增一台设备就要改图。而且这条链是**顺序**的——第 4 个节点必须等前 3 个跑完，即使它们互不相干。

真正合适的结构是**扇出（fan-out）**：一个节点把工作**摊开**成 N 份互不相干的分支，N 份各自算完，再由一个节点把结果**收拢（fan-in）**成一份答案。

```
                    ┌→ 查 living_room_light ┐
dispatch ──扇出──── ├→ 查 bedroom_ac        ├──收拢──→ aggregate → 一段文本
                    └→ 查 living_room_th... ┘
```

这个结构本身跟"设备"没关系，它是一个可以单独测、单独讲的东西。所以项目把它做成了一个**子图（subgraph）**——一个独立编译出来的小 LangGraph，有自己的状态定义、自己的节点、自己的入口和出口，被主图当成一个普通节点调用。你可以把它理解成"图里的函数"：调用方只关心传进去什么、返回什么，不关心里面有几个节点。

这一章要解决的三个问题：

1. 分支数量运行时才知道，静态边画不出来 → `Send`
2. N 个分支同时往同一个状态字段写结果 → **reducer**
3. 子图的状态和主图的状态怎么划界

## 10.2 代码怎么写的

全部代码在 `src/agent/parallel.py`，**整个文件只有 77 行**。主图那边的接线在 `src/agent/graph.py`。

### 10.2.1 主图这边：一个普通节点，里面调子图

先看外面。`build_graph` 在构造期就把子图编译好（`src/agent/graph.py:199`）：

```python
device_query_subgraph = build_device_query_subgraph(registry)
```

注意**它不是被 `add_node` 直接挂上去的**。挂上去的是一个普通函数 `parallel_query_node`（`src/agent/graph.py:816`）：

```python
workflow.add_node("device_query_subgraph", parallel_query_node)
```

节点名叫 `device_query_subgraph`，但节点体是普通 Python 函数（`src/agent/graph.py:623-635`）：

```python
def parallel_query_node(state: AgentState) -> dict:
    latest_text = getattr(state["messages"][-1], "content", "")
    targets = extract_query_targets(latest_text, registry)
    result = device_query_subgraph.invoke({
        "query": latest_text,
        "targets": targets,
        "parallel_results": [],
    })
    emit_progress("parallel_query_completed", target_count=len(targets))
    return {
        "messages": [AIMessage(content=result.get("response", "没有找到可查询的设备。"))],
        "parallel_query_results": result.get("parallel_results", []),
    }
```

四步：算出目标清单 → `.invoke()` 子图 → 发一条进度事件 → 把子图的输出翻译成主图的状态更新。10.3 会讲为什么是 `.invoke()` 而不是把子图当节点。

路由那边的判断在 `src/agent/graph.py:335-340`：意图是 `device_query` **且** `should_use_parallel_query()` 为真才走这条路；`route_task`（`src/agent/graph.py:825-834`）把它翻译成节点名；跑完直接 `END`（`src/agent/graph.py:848`）——**不回 ReAct，不调模型**。

`should_use_parallel_query`（`parallel.py:38-39`）的判断朴素到一行：

```python
def should_use_parallel_query(query: str, registry: DeviceRegistry) -> bool:
    return len(extract_query_targets(query, registry)) >= 2
```

目标 ≥ 2 个就扇出。目标提取 `extract_query_targets`（`parallel.py:21-35`）是三级递降的纯字符串匹配，没有 LLM：

1. 句子里有"所有设备"/"全部设备"/"家里设备" → 返回全部（`:25-26`）
2. 否则先找**精确设备名**命中，只要命中就只返回这些（`:27-29`）——说"客厅灯和卧室灯"就只查这两盏
3. 都没有，才按**设备名或房间名**子串匹配，去重后保持注册中心顺序（`:30-35`）

### 10.2.2 子图内部：四个节点

`build_device_query_subgraph`（`parallel.py:42-76`）在函数内部定义四个闭包节点，最后接线编译。注意 `registry` 是被闭包捕获的（和第 4 章的工具工厂同一套路），子图不读任何全局变量。

**dispatch（`:44-50`）** 什么状态都不改，只干一件事：

```python
def dispatch_node(state: QueryState):
    registry.tick_environment()
    return {}
```

`tick_environment()` 是模拟器的"环境推演"——按同房间空调/加湿器的状态往前推一次传感器读数。它放在这里有讲究，10.3 讲。

**fan_out（`:52-53`）** 这是这一章的主角：

```python
def fan_out(state: QueryState):
    return [Send("query_device", {"device_id": device_id}) for device_id in state["targets"]]
```

**query_device（`:55-61`）** 每个分支执行一次：

```python
def query_device(state: QueryState):
    device = registry.get(state["device_id"])
    if device is None:
        result = {"device_id": state["device_id"], "ok": False, "text": "设备不存在"}
    else:
        result = {"device_id": device.device_id, "ok": True, "text": device.to_status_text()}
    return {"parallel_results": [result]}
```

注意它返回的是 `[result]`——**一个只有一个元素的 list**。为什么不直接返回 `result`？下一节讲。

**aggregate（`:63-66`）** 收拢：按 `device_id` 排序，拼成多行文本。

接线（`:68-76`）：

```python
graph.set_entry_point("dispatch")
graph.add_conditional_edges("dispatch", fan_out, ["query_device"])
graph.add_edge("query_device", "aggregate")
graph.add_edge("aggregate", END)
return graph.compile()
```

一共四条边，就把"不定数量的并行分支"表达完了。

### 10.2.3 `Send` 是什么：静态条件边 vs 动态派发

`add_conditional_edges` 你在第 5、6 章见过，那时候它的返回值是**一个节点名字符串**：

```python
# 静态条件边：从若干个"预先声明过"的目标里挑一个
def route_task(state) -> Literal["planner", "compact_context", ...]:
    if state.get("intent_route") == "clarification":
        return "clarification"
    ...
```

关键词是"预先声明过"。可能的去向写在 `Literal[...]` 里，也写在 `add_conditional_edges` 第三个参数的映射表里（`graph.py:836-846`）。图的形状在编译时就定了，运行时只是**选一条**。

`Send` 打破的正是这一点。同一个 `add_conditional_edges`，返回值换成 `list[Send]`：

```python
[Send("query_device", {"device_id": "living_room_light"}),
 Send("query_device", {"device_id": "bedroom_ac"}),
 Send("query_device", {"device_id": "living_room_th_sensor"})]
```

`Send(节点名, 载荷)` 的意思是"**再跑一份** `query_device`，它这次看到的状态就是这个载荷"。返回三个 `Send`，`query_device` 就被执行三次。列表长度由 `state["targets"]` 决定，也就是由用户那句话决定。

两个容易误解的点：

- **`Send` 的第二个参数会成为目标节点的整个 state**，不是"合并进 state"。所以 `query_device` 里能读 `state["device_id"]`（Send 传的），但**读不到** `state["query"]`（父级子图状态里的）。这也是为什么 `query_device` 只用 `device_id` 一个字段。
- 编译时仍然要在 `add_conditional_edges` 的第三个参数里声明 `["query_device"]`。这个声明不限制**数量**，只声明**可能的去向**——LangGraph 需要它来画图和做校验。

对照记一句：**静态条件边决定"往哪走"，`Send` 决定"分成几份走"。**

### 10.2.4 reducer：并发写状态的唯一合法通道

现在有 11 个 `query_device` 分支，每个都要往 `parallel_results` 里放一条结果。它们属于**同一个超步（superstep）**——LangGraph 一轮一轮推进，一轮里被激活的所有节点算完，才统一把状态更新写回去。

问题来了：11 份更新，同一个 key，写谁的？

LangGraph 的回答是**它不猜**。默认规则是"一个 key 一步只能收一个值"，收到第二个就报错。实测（10.4 实验 A）：

```
InvalidUpdateError
At key 'results': Can receive only one value per step. Use an Annotated key to handle multiple values.
```

这个设计是对的：如果 LangGraph 默默取最后一个，你会得到"11 个设备只查出 1 个"，而且没有任何报错，只能靠肉眼发现结果少了。**在并发写上，报错比猜好。**

合法的办法是给这个 key 声明一个 **reducer（合并器）**——一个函数，告诉 LangGraph"收到多个值时怎么合成一个"。声明方式是 `Annotated`（`parallel.py:13-19`）：

```python
class QueryState(TypedDict):
    query: str
    targets: list[str]
    device_id: NotRequired[str]
    parallel_results: Annotated[list[dict], operator.add]
    response: NotRequired[str]
```

`Annotated[list[dict], operator.add]` 读作：这个字段类型是 `list[dict]`，多个值到达时用 `operator.add` 合并。`operator.add` 对 list 就是 `+`，也就是拼接。于是 `[{a}] + [{b}] + [{c}]` → `[{a},{b},{c}]`。

**回头看 `query_device` 为什么返回 `[result]` 而不是 `result`：因为 reducer 是 `operator.add`，两边必须都是 list。**返回裸 dict 会变成 `list + dict`，直接 `TypeError`。这是 reducer 类型和节点返回值类型的契约——写 reducer 的时候顺手在脑子里过一遍"我这个 reducer 接受什么类型"。

这套机制你其实早就在用。`src/agent/state.py:29`：

```python
messages: Annotated[list, add_messages]
```

`add_messages` 是 LangChain 给的 reducer，比 `operator.add` 聪明：它按消息 id 去重、能处理 id 相同的消息覆盖。整本教程里每个节点 `return {"messages": [AIMessage(...)]}` 都是追加而不是覆盖——就是这个 reducer 在起作用。

`QueryState` 五个字段里**只有 `parallel_results` 有 reducer**（10.4 实验 C 会打印出来）。其余四个字段全程只有一个节点写，不需要。**reducer 不是"越多越安全"，加了 reducer 就意味着"这个字段会被并发写"，是一个语义声明，别乱加。**

### 10.2.5 子图状态与父图状态怎么对接

这套代码里，答案干脆得有点意外：**几乎完全隔离，靠一次手工翻译对接。**

`QueryState`（`parallel.py:13-19`）和 `AgentState`（`src/agent/state.py:16-73`）是两个毫无关系的 TypedDict。没有共享字段、没有继承、没有 LangGraph 的状态自动映射。全部交接发生在 `parallel_query_node` 那 13 行里：

```
主图 AgentState                          子图 QueryState
──────────────────────────────────────────────────────────
messages[-1].content   ──手工映射──→     query
（算出来的 targets）    ──手工映射──→     targets
                                          parallel_results（初始 []）
                                          ↓ 子图内部跑完
messages（追加 AIMessage） ←──────────    response
parallel_query_results     ←──────────    parallel_results
```

进去 3 个 key，出来 2 个 key，其余 60 多个 `AgentState` 字段子图一个都看不到——身份字段、审批状态、计划、记忆上下文，全部隔离在外。

这带来一个很值得注意的推论：**`AgentState` 里的 `parallel_query_results`（`state.py:43`）没有 reducer。**

```python
parallel_query_results: NotRequired[list[dict[str, Any]]]
```

为什么不需要？因为在主图看来，`parallel_query_node` 是**一个**节点，写**一次**。扇出、并发写、合并全部发生在子图边界之内，早在 `.invoke()` 返回时就已经收拢成一个普通 list 了。

**这就是子图边界的价值：并发的复杂度被关在一个盒子里，盒子外面的状态模型保持简单。** 如果换成把 11 个 `Send` 直接打在主图上，`AgentState` 就得给 `parallel_query_results` 加 reducer，而且这个"某些超步里我会被并发写"的性质会成为整个主图的一部分——以后每个改主图的人都得记着它。

## 10.3 关键设计决策

### 决策一：子图用 `.invoke()` 包在普通节点里，不 `add_node(subgraph)`

LangGraph 允许把编译好的子图直接当节点挂上去。这里没这么做，代价和收益都很明确。

代价：状态映射得手写（10.2.5 那张表），子图的内部步骤不会出现在主图的 stream 事件里——CLI 看不到 11 个分支逐个完成，只看到一条 `parallel_query_completed`（`graph.py:631`）。

收益是隔离：子图不需要认识 `AgentState` 的任何字段，因此可以脱离主图单独 `invoke`（`tests/test_phase_nine.py:54-63` 就是这么测的），也不会因为主图加字段而受影响。

还有一个不显眼但重要的后果：**子图编译时没传 checkpointer**（`parallel.py:76` 是裸 `graph.compile()`，实测 `sub.checkpointer is None`），而主图传了（`graph.py:928`）。所以第 7 章那套 `interrupt` + 恢复在这个子图里**不能用**。对只读查询来说无所谓——查询没有需要审批的副作用。但如果哪天想把扇出扩展到**控制**动作，这一点会立刻变成硬约束。

### 决策二：`tick_environment()` 放在 dispatch，而不是 query_device

`parallel.py:45-48` 的注释把理由写在了原地：

> 放在 dispatch 而不是 query_device 里，是为了保证一次查询只推演一次，
> 否则并行分支数量会直接改变读数。

想清楚这个 bug 长什么样：如果每个分支各推演一次，那"客厅和卧室"（11 个目标）会把环境往前推 11 步，"只查客厅灯"（1 个目标）推 1 步。**用户问的设备越多，模拟出的温度离目标温度越近。**读数变成了问句长度的函数。这种 bug 没人会往"因为我多问了一台设备"上想。

顺带纠正一个说法：`CLAUDE.md` 写"`tick_environment()` 只应由 `read_sensor` 调用"。实际有三个调用点，`src/tools/devices.py:128-131` 的注释才是准确版本——只有"显式看一眼环境"的入口才该调它：`read_sensor`、`get_device_status`、以及这里的子图 dispatch。控制路径和 Verifier 路径都不该调。

### 决策三：aggregate 里排序，不靠完成顺序

`parallel.py:64`：

```python
results = sorted(state.get("parallel_results", []), key=lambda item: item["device_id"])
```

`operator.add` 拼接的顺序 = 更新到达的顺序，而并发分支的完成顺序**不保证**。不排序的话，同一句话问两次可能得到两种行序，用户会觉得系统在抽风。迭代文档（`docs/iterations/005-subgraph-dynamic-parallel.md:12`）把这条列成了明确要求：

> 最终按设备 ID 排序，避免并行完成顺序影响用户看到的结果。

代价是行序按 `device_id` 字母序（`bedroom_ac` 在 `living_room_light` 前面），不是用户提问的顺序。确定性优先于自然度——这是个可以再改进的取舍。

### 决策四：只扇出只读查询，绝不扇出控制

`docs/iterations/005-subgraph-dynamic-parallel.md:16` 划了这条线：多设备**控制**仍走第 5~6 章的 Planner 顺序执行，保留计划确认、Verifier 和重试/重规划。

理由是失败语义完全不同。查询失败了，那一行显示"设备不存在"，其余 10 行照样有用（`parallel.py:57-58`）。控制失败了，你面对的是"3 个设备开了、1 个失败了"这种半完成状态——需要判断能不能重试、要不要回滚、剩下的还该不该继续。**扇出天然不擅长表达"部分失败后怎么办"，而顺序执行的状态机天然擅长。**

## 10.4 动手试一试

三个实验。**实验 A 不依赖本项目的任何代码，也不需要 API Key**，只要装了 langgraph 就能跑——它演示的是本章最容易撞的那面墙。

### 实验 A：亲手撞一次 `InvalidUpdateError`，再修好它

同一个图跑两遍，唯一区别是 `results` 有没有 reducer：

```bash
cd "G:/大厂学习/minimind/langgraph" && PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -c "
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from langgraph.types import Send

def build(state_cls):
    g = StateGraph(state_cls)
    g.add_node('dispatch', lambda s: {})
    g.add_node('worker', lambda s: {'results': [s['n']]})
    g.set_entry_point('dispatch')
    g.add_conditional_edges('dispatch', lambda s: [Send('worker', {'n': i}) for i in (1, 2, 3)], ['worker'])
    g.add_edge('worker', END)
    return g.compile()

class NoReducer(TypedDict):
    results: list

class WithReducer(TypedDict):
    results: Annotated[list, operator.add]

try:
    print('A 没有 reducer:', build(NoReducer).invoke({'results': []}))
except Exception as exc:
    print('A 没有 reducer:', type(exc).__name__)
    print(str(exc)[:200])
print('B 加 operator.add:', build(WithReducer).invoke({'results': []}))
"
```

实测输出：

```
A 没有 reducer: InvalidUpdateError
At key 'results': Can receive only one value per step. Use an Annotated key to handle multiple values.
For troubleshooting, visit: https://docs.langchain.com/oss/python/langgraph/errors/INVALID_CONCUR
B 加 operator.add: {'results': [1, 2, 3]}
```

**两行代码之差。** 把这段留着——以后你自己写扇出撞上这个报错时，会立刻知道是哪一行的锅。

改着玩：把 `worker` 改成返回裸 `{'results': s['n']}`（不加方括号），看 `operator.add` 怎么报 `TypeError`。

### 实验 B：`Send` 分支真的是并发的（但要有等待才看得出来）

给每个分支塞一个 `sleep(0.5)`，看总耗时：

```bash
cd "G:/大厂学习/minimind/langgraph" && PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -c "
import operator, time, threading
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from langgraph.types import Send

class S(TypedDict):
    results: Annotated[list, operator.add]

def worker(s):
    time.sleep(0.5)
    return {'results': [threading.current_thread().name]}

g = StateGraph(S)
g.add_node('dispatch', lambda s: {})
g.add_node('worker', worker)
g.set_entry_point('dispatch')
g.add_conditional_edges('dispatch', lambda s: [Send('worker', {'n': i}) for i in range(4)], ['worker'])
g.add_edge('worker', END)
t0 = time.perf_counter()
out = g.compile().invoke({'results': []})
print('4 个分支各 sleep 0.5s，总耗时 %.2f s' % (time.perf_counter() - t0))
print(out['results'])
"
```

实测输出：

```
4 个分支各 sleep 0.5s，总耗时 0.51 s
['ThreadPoolExecutor-1_0', 'ThreadPoolExecutor-1_1', 'ThreadPoolExecutor-1_2', 'ThreadPoolExecutor-1_3']
```

两条信息很关键：4 × 0.5s 只用了 0.51s，**等待真的重叠了**；而且线程名说明它们跑在不同线程上——同步的 `.invoke()` 就有线程池，**不需要写 async**。记住这个结论，下一节要用它。

### 实验 C：跑真实子图，看结构、看结果、看耗时

```bash
cd "G:/大厂学习/minimind/langgraph" && PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -c "
from loguru import logger; logger.remove()
from src.agent.parallel import build_device_query_subgraph, extract_query_targets, QueryState
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend

r = DeviceRegistry(SimulatorBackend())
sub = build_device_query_subgraph(r)
print('checkpointer =', sub.checkpointer)
for e in sub.get_graph().edges:
    print(' ', e.source, '->', e.target, '(conditional)' if e.conditional else '')
print('带 reducer 的字段:', [k for k, v in QueryState.__annotations__.items() if 'Annotated' in str(v)])
print('精确设备名短路:', extract_query_targets('客厅灯和卧室灯开着吗', r))
print('房间名匹配:', len(extract_query_targets('客厅和卧室的灯都开着吗，空调多少度', r)), '个目标')
out = sub.invoke({'query': 'x', 'targets': ['bedroom_ac', 'living_room_light'], 'parallel_results': []})
print(out['response'])
"
```

实测输出：

```
checkpointer = None
  __start__ -> dispatch
  dispatch -> query_device (conditional)
  query_device -> aggregate
  aggregate -> __end__
带 reducer 的字段: ['parallel_results']
精确设备名短路: ['living_room_light', 'bedroom_light']
房间名匹配: 11 个目标
卧室空调 (bedroom_ac): 🔴 关闭 | 温度: 26°C | 模式: 制冷 | 风速: 自动
客厅灯 (living_room_light): 🔴 关闭 | 亮度: 80% | 色温: 暖白
```

四条边、一个 reducer 字段、没有 checkpointer——10.2 和 10.3 说的每一条都在这里。

### 实验 D：跑测试

这一章对应的测试是 `tests/test_phase_nine.py`（**不是** `test_phase_five.py`，那是记忆系统的）：

```bash
cd "G:/大厂学习/minimind/langgraph" && PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m pytest -q tests/test_phase_nine.py
```

实测输出：

```
...                                                                      [100%]
3 passed in 1.99s
```

三个用例分工很清楚，**注意它们验证的是结构和合并结果，一个字都没测耗时**：

| 用例 | 行 | 验证什么 |
|---|---|---|
| `test_target_extraction_and_parallel_decision` | `:41-52` | 目标清单**逐个 ID 精确相等**（含 3 个传感器，不含未提到的玄关）；2 个目标才扇出，1 个不扇出 |
| `test_subgraph_fanout_aggregates_sorted_results` | `:54-63` | 单独 `invoke` 子图，断言 `parallel_results` 的**顺序**是排过序的，且两台设备都出现在 `response` 里 |
| `test_main_graph_uses_parallel_query_subgraph_without_react` | `:65-82` | 走完整主图，断言路由到 `parallel_query`、结果 ≥ 2 条 |

第三个用例有个漂亮的写法值得抄（`:66-71`）：它塞进去的 FakeLLM 的 `invoke` 直接 `raise AssertionError("parallel query should not invoke ReAct")`。**"不该发生的事"用崩溃来断言**，比事后检查调用次数更狠、也更难写错。

## 10.5 踩坑与局限

### 坑一：忘了 reducer，报错信息还挺容易看漏

`InvalidUpdateError: At key 'X': Can receive only one value per step` 就是"你并发写了一个没 reducer 的 key"。修法固定：给那个 key 加 `Annotated[list, operator.add]`。

反过来也是坑：**加了 reducer，节点返回值的类型必须和 reducer 匹配。**`operator.add` 配 list，就得返回 `[result]`；返回裸 dict 是 `TypeError`。

还有一个更隐蔽的：`Send` 的载荷会**成为**目标节点的整个 state。分支里读一个"父级状态里明明有"的字段会 `KeyError`，因为它压根没被传进来。要用就在每个 `Send` 的载荷里显式带上。

### 坑二：这些 `Send` 分支没有并行任何东西（诚实的一节）

必须把话说白：**当前实现的扇出带来的是结构上的清晰，不是速度。**

`query_device`（`parallel.py:55-61`）的全部工作是 `registry.get()`——一次内存字典查找，加一次字符串格式化。零 IO、零 LLM、零等待。而实验 B 已经证明并行的收益来自**等待的重叠**。没有等待，就没有可重叠的东西。

更难听的是：调度开销**远大于**直接循环。同一批 11 个目标，扇出 20 次和裸 `for` 循环 20 次的实测对比：

```
run0  fanout 133.5 ms | loop 0.30 ms | 451x
run1  fanout 176.5 ms | loop 0.30 ms | 595x
```

慢**几百倍**。绝对值当然无所谓（一次查询几毫秒，用户感知不到），但方向必须说清楚：这是一次**为了展示机制而付出的开销**，不是性能优化。`docs/gap-analysis.md:508-516`（3.2 节）把这条列为审查项，原话最到位：

> 并行的收益来自 IO 等待的重叠。把 fan-out 用在纯内存操作上，学到的是 API 而不是判断力。

**什么时候才会有真收益？** 分支体里得有真实的等待：

- 每台设备是一次真实的厂商云 HTTP 调用（几十到几百毫秒）
- 每个分支各做一次 LLM 调用（比如 6 个角色并行给意见，再投票）
- 每个分支查一个不同的外部 MCP 服务

而**那时候要补三件当前完全没做的事**：

1. **并发安全。** 实验 B 已经证明分支真的跑在不同线程上。现在的分支只**读** registry 所以侥幸安全；一旦分支里有写操作，就是真实的数据竞争。项目当前是**零锁**，还有 3 处 `check_same_thread=False`（`docs/gap-analysis.md` 3.5 节自陈）。
2. **超时。** 一个分支挂住，整个超步就卡住——LangGraph 要等这一轮所有节点结束才推进。没有超时就意味着一台离线设备能让整句回答永远出不来。
3. **部分失败策略。** 见下面的坑三。

顺带修正一个可能的误解：不需要为了并行去改 async。gap-analysis 提到"`src/` 除 `src/mcp/` 外零 `async def`，图只用 `invoke`/`stream`"，这是事实，但实验 B 说明同步 `.invoke()` 本身就走线程池。**缺的不是 async，是分支里没有等待。**

### 坑三：一个分支失败了，没有人管

`query_device` 唯一的失败处理是"设备不存在 → 塞一条 `ok: False` 的结果"（`parallel.py:57-58`）。这个字段进了 `parallel_results`，然后：

- `aggregate`（`:63-66`）只取 `item["text"]` 拼进回答，**从不看 `ok`**
- `parallel_query_node`（`graph.py:632-635`）也不看，直接把 `response` 当 AIMessage 发给用户

于是用户看到的是 10 行正常状态夹着一行"设备不存在"。可接受，但**这已经是最好的情况了**。如果 `query_device` 里抛出真异常（未来接真实设备的网络超时），LangGraph 默认行为是整个超步失败 → 整次调用抛出 → 用户什么都没拿到。**11 个设备里 1 个网络抖动，10 个成功的结果一起丢掉。**

真接了 I/O 就必须在分支内部把异常**吃掉并转成 `ok: False` 的结果**，让失败成为数据而不是控制流。然后 `aggregate` 得真的读 `ok`，回答里区分"查到了"和"查不到"。

### 坑四：结果顺序不由完成顺序决定，也不由用户的提问顺序决定

`operator.add` 的拼接顺序 = 到达顺序 = 不保证。`aggregate` 用排序把它压成确定的（决策三），但排的是 `device_id` 字母序。用户问"客厅和卧室的灯"，答案先出 `bedroom_*` 再出 `living_room_*`——确定，但不自然。想按提问顺序排，得让 `Send` 载荷带上序号，`aggregate` 按序号排。

### 坑五：目标提取是子串匹配，一个房间名把整屋拉进来

实验 C 里"客厅和卧室的灯都开着吗，空调多少度"提取出 **11 个**目标——因为第三级降级是按**房间名**匹配（`parallel.py:30-33`），提到"客厅"就把客厅所有设备拉进来，电视、窗帘、加湿器、传感器全在里面。用户只问了灯和空调。

`extract_query_targets` 的 docstring（`:22`）把这个取舍写明了：*"Resolve room/device mentions without guessing ambiguous device types."* 它**故意不猜**"灯"指哪几盏——多查几台只读设备的代价是回答啰嗦，猜错设备类型的代价是答错。宁可啰嗦。

但代价是真实的：11 行状态文本可能盖住用户真正问的那两台。改进方向是在 `aggregate` 里做一次相关性排序，或者按设备类型关键词二次过滤。

### 坑六：子图内部的步骤在 CLI 里看不见

`emit_progress("parallel_query_completed", ...)`（`graph.py:631`）在**主图节点**里发，子图内部四个节点一条事件都不发。而且这条事件属于 `TRACE_EVENTS`（`src/agent/observability.py:37-41`），**默认不显示**——得 `python -m src.main --trace` 才看得到。

所以扇出出问题时（比如某个分支静默返回空），你在 CLI 里看到的只是一段答案短了一截。要调试就直接单独 `invoke` 子图（实验 C），别在整条链上找。

**这一章的局限**：`Send` 的机制讲全了，但这个项目里它的**收益**没兑现——分支里没有等待，扇出只买到了结构清晰。真正需要并行的地方（多个外部 MCP 服务发现、每次工具调用重建 stdio 会话）反而还是串行的。这一章要带走的不是"用 `Send` 能加速"，而是"什么时候值得扇出，以及扇出之后要为并发安全、超时、部分失败额外付什么账"。

**下一章的问题**：这一章的隔离靠"子图看不到主图的 60 个字段"。同一个思路推到工具上会怎样——如果一个角色**根本看不见**某个工具，它就永远不可能误用它。但工具集一旦按角色切开，"某个角色死活不调那个工具"就会变成一类新的、极难排查的 bug。

---

[← 第 9 章 意图路由：什么时候不该用 LLM](09-意图路由.md) · [目录](README.md) · [第 11 章 多智能体：安全边界建立在看不见上 →](11-多智能体.md)
