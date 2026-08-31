[← 第 0 章 如何阅读本教程](00-如何阅读本教程.md) · [目录](README.md) · [第 2 章 最小可用：LLM + 一个工具 + 循环 →](02-最小可用的Agent.md)

---

# 第 1 章 Agent 到底是什么 —— 从"开客厅灯"说起

## 1.1 要解决什么问题

假设你想让大模型帮你开灯。你直接问它：

```
你: 打开客厅灯
GPT: 好的，我已经帮你打开客厅灯了！
```

**它在撒谎。** 大模型是一个"文本进、文本出"的函数，它没有手，没有网络连接，碰不到你家的灯。它只是根据训练数据判断"这种情况下人类会这么回答"，然后生成了一句最像样的话。

这就是 Agent 要解决的根本问题：**怎么让一个只会生成文本的模型，真的产生副作用**。

## 1.2 三个零件

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

## 1.3 一次工具调用在对话历史里是**一对**消息

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

## 1.4 动手试一试

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

## 1.5 踩坑与局限

**这一章的模型还太天真。** 我们假设模型看到工具说明书，就会正确地调用它。实际上：

- 模型可能把 `action` 写成 `turn_on`（Home Assistant 风格），而项目用的是 `on`。第 5 章会讲这个真实发生过的根因。
- 模型可能声称调用成功，但设备其实没变。第 6 章会讲怎么对账。
- 模型可能被用户的话诱导去解锁门锁。第 7、8 章会讲怎么拦。

**下一章的问题**：三个零件拼起来之后，多步任务为什么会做一半就跑偏？

---

[← 第 0 章 如何阅读本教程](00-如何阅读本教程.md) · [目录](README.md) · [第 2 章 最小可用：LLM + 一个工具 + 循环 →](02-最小可用的Agent.md)
