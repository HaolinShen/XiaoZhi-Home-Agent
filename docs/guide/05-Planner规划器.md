[← 第 4 章 Agent 的工具箱：工厂与显式依赖注入](04-工具工厂与依赖注入.md) · [目录](README.md) · [第 6 章 Executor 与 Verifier：去查真实状态 →](06-Executor与Verifier.md)

---

# 第 5 章 Planner：让 Agent 先说清要做什么

## 5.1 要解决什么问题

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

## 5.2 代码怎么写的

### 5.2.1 谁决定走 Planner：一个纯正则函数

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

`graph.py:331` 起的四个分支判断**全部带 `not use_planner` 前缀**。也就是说：**Planner 判定优先级最高**，一旦命中，意图分类的结果（RAG / 并行查询 / 澄清）全被压制。`PLANNING_ENABLED=false` 可以把整条分支关掉（`config.py:76`）。

### 5.2.2 计划长什么样：两个 Pydantic 模型

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

### 5.2.3 这一章最重要的一个陷阱：`with_structured_output` 会丢掉工具语义

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

### 5.2.4 补救办法：把丢掉的语义手动塞回 prompt

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

## 5.3 关键设计决策

### 决策一：为什么不用 `bind_tools`（既然它自带语义）

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

### 决策二：为什么预定义场景一律不走 Planner

`heuristics.py:40-43` 的注释给了两层理由：

> 预定义场景请求留在 ReAct + 场景审批路径。注意和 routing 的 scene_words 不同：这里匹配的是**整句惯用表达**（"我要出门"），routing 匹配的是**子串线索**（"离家"），宽窄不同是有意的——Planner 的排除必须精确，误伤一句"我要出门前关灯"就会把两个动作的请求送去走场景分支。

**业务理由**：「离家模式」已经是一个原子工具 `activate_scene`，有自己的审批通道。让 Planner 把它拆成 5 步，等于把一个已验证的原子操作降级成 5 个可能各自失败的步骤。

**工程理由**（就是上面那段注释）：排除表必须比路由表**窄**。宽了会误伤。

这里有个可以拿来锻炼判断力的观察：作者明知「我要出门前关灯，再把空调关掉」会被误判，仍然保留了短路。因为两种错误的代价不对称——**把场景请求错送进 Planner，会把一个可靠操作拆成多个不可靠步骤；把多动作请求错送进场景分支，最坏结果是模型多说一句「我不太确定你要哪个场景」。**

### 决策三：判定函数为什么是纯正则，不问 LLM

三个理由，一个比一个实在：

1. **省一次调用。** 每轮对话都要判定，用 LLM 判定意味着每轮多一次往返。
2. **确定性。** 同一句话永远走同一条路。测试可以直接断言（`tests/test_heuristics.py`），不需要 FakeLLM。
3. **它本来就够用。** 「有几个动作词」「涉及几类设备」这种问题，正则的准确率不比模型低。

这条原则在项目里出现了三次（`should_use_planner`、路由兜底、`required_automation_tool`），011 迭代把它们收敛进了同一个文件 `heuristics.py`。`docs/gap-analysis.md:52` 附近把它列为「已经做对的」之一：

> **确定性分支刻意不问 LLM**……知道什么时候**不该**用模型，是 Agent 工程的成熟标志。

## 5.4 动手试一试

### 实验 A：摸出判定阈值（不需要 API Key）

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

### 实验 B：看到 Planner 只写不做

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

### 实验 C：亲手制造 `turn_on` 那个坑

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

### 实验 D：看 prompt 里到底写了什么

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

## 5.5 踩坑与局限

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

[← 第 4 章 Agent 的工具箱：工厂与显式依赖注入](04-工具工厂与依赖注入.md) · [目录](README.md) · [第 6 章 Executor 与 Verifier：去查真实状态 →](06-Executor与Verifier.md)
