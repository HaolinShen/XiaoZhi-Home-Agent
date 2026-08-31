[← 附录 A 动手实验清单](附录A-动手实验清单.md) · [目录](README.md)

---

# 附录 B 常见报错速查

用法：**用报错里最独特的那个词去搜这一页**（`InvalidUpdateError`、`WinError 32`、`home_id`）。
每条给的是**根因**和**修法**，展开的推导在对应章节里。

这一页分两半，第二半更重要：

- [B.1 会报错的](#b1-会报错的) —— 有异常、有堆栈，好查
- [B.2 不报错的](#b2-不报错的) —— **只是行为变怪**。这一半才是真正会烧掉你半天的

所有报错原文都是在 `HEAD = e7d1113` 上真实触发后粘贴的，没有一条是凭印象写的。

## B.1 会报错的

### 环境类

```
UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f7e2' in position 4: illegal multibyte sequence
```

**根因**：Windows 控制台默认 GBK 编码，而设备名是中文、状态是 emoji（🟢）。
**修法**：命令前面加 `PYTHONIOENCODING=utf-8`。**这是本项目最高频的入门障碍**，不是可选的讲究。
详见[第 0 章](00-如何阅读本教程.md)。

---

```
1 validation error for Settings
  Value error, 请在 .env 文件中设置 LLM_API_KEY 或 BAILIAN_API_KEY。
获取地址: https://bailian.console.aliyun.com/ [type=value_error, ...]
```

**根因**：`Settings` 在构造期就校验 API Key，没配就起不来。
**修法**：把 Key 写进项目根目录的 `.env`。

但先问自己**是不是真的需要它** —— 设备层、工具层、路由层、检索层、全部测试都不碰模型。
[附录 A](附录A-动手实验清单.md) 里标"不需要 API Key"的实验有三十多个。
测试也不需要：需要模型的用例都用 `FakeLLM` 替掉了。详见[第 17 章](17-如何验证你的改动.md)。

---

```
PermissionError: [WinError 32] 另一个程序正在使用此文件，进程无法访问。
```

**根因**：`TemporaryDirectory.cleanup()` 撞上没关闭的 SQLite 连接。Windows 上文件被占用就删不掉。
**测试断言全过也会判失败**，看起来像"测试莫名其妙红了"。

**修法**：凡在临时目录里建过 `AutomationStore` / `MemoryRepository` / `build_graph()`
（它内部会建 `MemoryRepository`，通过 `graph.memory_repository` 暴露）的，都要在目录删除前 `close()`。

**还有一层顺序陷阱**：unittest 先跑完整个 `tearDown()` 才轮到 `doCleanups()`，
所以把 `temp_dir.cleanup()` 写在 `tearDown` 里会**早于**测试方法内 `addCleanup(repo.close)` 注册的关闭动作。
正确做法是在 `setUp` 里 `self.addCleanup(self.temp_dir.cleanup)` —— `doCleanups` 是 LIFO，
最先注册就最后执行。详见[第 17 章](17-如何验证你的改动.md)坑一、坑二。

### LangGraph 机制类

```
InvalidUpdateError: At key 'X': Can receive only one value per step
```

**根因**：两个并发分支（多个 `Send`）同时往同一个 state key 写，而这个 key 没有 reducer。
**修法**：给那个 key 加 reducer，最常用的是 `Annotated[list, operator.add]`。
详见[第 10 章](10-并行子图与Send.md)坑一。

---

```
TypeError: can only concatenate list (not "dict") to list
```

**根因**：加了 `operator.add` 当 reducer，但节点返回的是裸 `dict` 而不是 `[dict]`。
**修法**：reducer 类型和节点返回值类型是一份契约 —— `operator.add` 配 list，就必须返回 `[result]`。
写 reducer 的时候顺手在脑子里过一遍"我这个 reducer 接受什么类型"。详见[第 10 章](10-并行子图与Send.md)。

---

```
KeyError: '<某个父级状态里明明有的字段>'
```

（发生在 `Send` 分支节点里。）

**根因**：`Send` 的载荷会**成为**目标节点的整个 state，父级状态不会自动流进去。
**修法**：要用的字段在每个 `Send` 的载荷里显式带上。详见[第 10 章](10-并行子图与Send.md)坑一。

---

```
RuntimeError: ... get_stream_writer ...
```

**根因**：在图上下文之外调用 `get_stream_writer()`。后台自动化执行器直接 `tool.invoke`，连图上下文都没有。
**修法**：这正是 `emit_progress()` 要双写 stream + 日志的原因 —— 图外路径靠日志留痕。
详见[第 15 章](15-可观测性.md)。

### Pydantic 校验类（这些是设计意图，不是 bug）

```
1 validation error for PlanStep
tool_name
  Input should be 'control_light', 'control_ac', 'control_tv', 'control_curtain',
  'control_humidifier', 'control_water_heater', 'control_lock' or 'control_kettle'
  [type=literal_error, input_value='read_sensor', input_type=str]
```

**根因**：`read_sensor` 不在 `PlanStep.tool_name` 的 `Literal` 里 —— **传感器是只读的，不能进执行计划**。
**这不是要修的错误**，这是不变量七在起作用。详见[第 3 章](03-设备层与能力声明.md)、[第 18 章](18-全景图与不变量.md)。

---

```
1 validation error for LightDevice
brightness
  Input should be less than or equal to 100 [type=less_than_equal, input_value=120, input_type=int]
```

**根因**：`models.py` 用 `Field(ge=0, le=100)` 挡住越界值。模型胡说的参数在数据层就被拒。
**这也不是要修的错误**。详见[第 3 章](03-设备层与能力声明.md)。

---

```
1 validation error for ScheduledActionInput
  Value error, 定时自动化禁止解锁门锁 [type=value_error, ...]
```

**根因**：`automation/planning.py` 的 `reject_scheduled_unlock` 显式拒绝定时解锁。
一条定时任务在无人确认的凌晨解锁大门，后果不需要解释。

**注意禁令的粒度是 per-action 而不是 per-tool**：定时**上锁**是允许的，
`control_lock` 也在自动化工具白名单里。检查点只有这一个 `model_validator` ——
绕过 `ScheduledActionInput` 直写 store 就能执行 unlock。详见[第 14 章](14-事件驱动自动化.md)。

### 身份与权限类

```
RuntimeError: 记录设备偏好缺少可信身份: ['home_id', 'user_id', 'thread_id', 'client_id']
```

**根因**：开启了偏好观察（`enable_preference_tracking=True`），但 `config["configurable"]` 里没有身份。
**这是刻意的 fail-fast**，不是缺陷 —— 错误发生在最接近根因的地方。

旧写法是"逐键检查后安静跳过"，于是一个空 `configurable` 的 config 让 `if config is not None` 恒为真，
权限检查静默失效。**这个教训值得带走：不要写静默兜底，要么在构造期把选择做明确，要么当场炸。**
详见[第 4 章](04-工具工厂与依赖注入.md)、[第 8 章](08-可信身份边界.md)。

---

```
KeyError: 'home_id'
```

**根因**：工具用下标访问 `config["configurable"]["home_id"]`，故意不用 `.get()` ——
拿不到身份就当场炸，不允许带着 `None` 往下走。
**修法**：身份必须经 `AgentContext.to_config()` 注入。如果你在后台路径（自动化执行器、MCP）
撞到这个，说明该路径应该显式关掉需要身份的功能，而不是补一个假身份。
详见[第 8 章](08-可信身份边界.md)。

---

```
MemoryPermissionError: home shared memory requires administrator permission
```

**根因**：非管理员写家庭共享记忆。家庭/房间/设备级共享记忆需要 `is_admin`。
**修法**：用 `python -m src.main --admin` 启动。

**但 `--admin` 不是全路径生效的**：`tools/memory.py:_context()` 构造 `AgentContext` 时不传 `is_admin`，
只有 `save_home_rule` 显式传。所以管理员用 `delete_personal_memory` 删家庭规则**照样被拒**。
详见[第 8 章](08-可信身份边界.md)坑一。

---

```
ContextValidationError
```

**根因**：`SpaceDirectory.validate()` 发现 `room_id` / `device_id` 不属于当前住宅。
**这是跨住宅越权的最后一道门**，报错就是它在干活。详见[第 8 章](08-可信身份边界.md)。

### LLM API 协议类

```
HTTP 400（请求体里出现了 tool 消息，但找不到对应的 tool_calls）
```

**根因**：上下文裁剪切断了 `tool_calls ↔ ToolMessage` 的配对。
OpenAI 兼容协议对这个配对是**强制**的 —— `tool` 角色的消息必须紧跟在带对应 `tool_calls` 的助手消息之后，
服务端不会宽容地忽略，而是直接 400 拒绝整个请求。

**根因的根因**：条数闸门和 token 闸门都只看长度，**它们不认识这个配对**。
**修法**：两道长度闸门算完之后，由专门一步对齐窗口起点。
**通用教训**：协议不变量不能指望每道闸门各自照顾，必须由**专门一步**兜住。
详见[第 1 章](01-Agent是什么.md)、[第 12 章](12-记忆系统.md)。

### 依赖漂移类

```
TypeError: got an unexpected keyword argument 'port'
```

**根因**：`FastMCP.run()` 在当前依赖版本里没有 `port` 参数，新版 SDK 把端口挪到了构造函数
（`FastMCP(name, port=8765)`）。
**当前状态**：**已修**（013 收尾）——端口改为构造期传入 `create_mcp_server(..., port=...)`，
`main.py` 按 transport 分支调用 `run()`。这个 bug 存活了一整轮迭代，最后是装上 mypy 后
被它的 `call-arg` 检查揪出来的——「装了类型检查工具却从来不跑」等于没装。

**这个坑的普适教训**：MCP 生态还年轻，SDK 的 API 在小版本间会动。
**MCP 相关代码尤其需要一个真的起进程的集成测试** —— 本项目 client 侧有，server 侧的启动路径没有。
详见[第 16 章](16-MCP工具共享.md)坑二。

## B.2 不报错的

这一半没有堆栈可查。你只能从**症状**反查根因 —— 这也是[第 18 章](18-全景图与不变量.md)
把十七条不变量单独列出来的原因：破坏它们往往不报错。

| 症状 | 根因 | 详见 |
| --- | --- | --- |
| **Agent 说"家里没人"** | 人体传感器被场景批量操作关掉了。你在排查"为什么感应不到人"，根本不会联想到"上次执行离家模式" | [3](03-设备层与能力声明.md)、[18](18-全景图与不变量.md) 不变量七 |
| **同一句话问两遍，温度不一样** | 有别的地方调了 `registry.tick_environment()`。它只应由 `read_sensor` 调用 | [18](18-全景图与不变量.md) 不变量八 |
| **Agent 自信地报告"客厅灯已开"，灯是关的** | 验证读了模型的自述而不是真实设备状态。**这是 Agent 项目里最常见的失效形态** | [6](06-Executor与Verifier.md) |
| **定时请求被当场执行**（用户明天早上发现空调开了一整夜） | `automation_management` 的确定性短路失效了。模型会把"明天 6 点回家前开空调"识别成预定义的「回家模式」然后立刻执行 —— 改 prompt 治不好 | [9](09-意图路由.md)、[18](18-全景图与不变量.md) 不变量十二 |
| **用户问"我有哪些例程"，系统给他新建了一条** | `required_automation_tool()` 过度触发。它的默认返回值必须是 `None` | [11](11-多智能体.md)、[18](18-全景图与不变量.md) 不变量十三 |
| **某个角色突然"变傻"，明明有工具却不用** | 新增的非设备工具忘了加进该角色的名字集合。**它不报错，那个角色只是永远看不见这个工具**。而且这条约束当前没有测试兜底 | [11](11-多智能体.md)坑一 |
| **说"我要出门了"，Agent 什么都不做** | 两张场景词表不同源：`PLANNER_SCENE_MARKERS` 含"我要出门"，`ROUTING_SCENE_WORDS` 不含"出门"。实测落到 `chat` 角色、**0 个工具** | [9](09-意图路由.md)坑四 |
| **`find('灯')` 返回了客厅灯**，而你以为它会拒绝猜测 | "拒绝猜测"的覆盖面比想象的窄，短关键词会静默命中第一台 | [3](03-设备层与能力声明.md)坑一 |
| **问某个症状，答的是同一台电器的另一个症状** | 013 前是 bigram 假命中（唯一重叠的词只有"空调"，三段同分全部越过阈值）；013 后换成了更难缠的版本：**同一台电器不同症状的语义本来就近**，实测正例余弦中位 0.653、这类困难负例 0.568，绝对置信度分不开。引用层面靠相对截断（弱于第一名 0.7 倍的不进引用）压掉了，但**拒答那一半没解决**。两个时代的共同点是 `rag_status` 为 `answered`、`citation_count` 非零，**指标层面毫无异常信号** | [13](13-知识检索与RAG.md)实验二、13.5 局限一 |
| **说明书里根本没这一章，系统却引经据典答了** | 查询重写的 prompt 强制单选（"判断最可能属于哪一个小节"），正确答案不在清单里时模型必然挑一个最像的，而校验只查"标题真实存在"——**存在不等于相关**。模型甚至会在正文里写"说明书未提及"，引用块照样挂三条。**只有跑真实 LLM 的端到端验证抓得到**：单元测试传 `llm=None` 走词表，召回评测只测首轮检索 | [13](13-知识检索与RAG.md)、`docs/iterations/013-hybrid-retrieval.md` §3.5 |
| **口语症状全查不到，`dense` 却写着 false** | 语义通道没启用（`RAG_EMBEDDING_MODEL_ID` 为空或 Key 缺失），检索退化成纯 BM25。**这是配置结果不是故障**，`build_embeddings` 会打一条 INFO 说明。纯词法在口语类上只有 7/30 | [13](13-知识检索与RAG.md)实验六 |
| **`llm_usage` 日志一条都不落盘**，业务却照常跑完 | 观测代码自己炸了，异常被 LangChain 回调管理器吞成一行 stderr。**度量静默失效比没有度量更糟** | [15](15-可观测性.md)坑一、[19](19-已知边界与下一步.md) |
| **加了 `--trace` 却在日志文件里找不到 `graph_progress`** | `--trace` 只改终端渲染，不抬日志级别。`graph_progress` / `node_latency` 是 DEBUG 通道，要 `--debug` | [15](15-可观测性.md)坑二 |
| **例程列表报的时间是旧的** | `routine_runs.anchor_at` 不随 ETA 更新（`create_run` 用 `INSERT OR IGNORE`）。任务已经挪走了，报给用户的锚点还是老的 | [14](14-事件驱动自动化.md)坑三 |
| **配了 `AUTOMATION_TIMEZONE` 但渲染出的时间没变** | `Routine.timezone` 是模型硬编码默认值，运行时配置从不传下去。配置只影响 naive datetime 的解释 | [14](14-事件驱动自动化.md)坑四 |
| **一条任务永久卡在 `running`，既不重试也不标失败** | 进程在 `tick()` 中途被 kill。`due_tasks` 的 `status='pending'` 过滤永远捞不回来，没有回收逻辑 | [14](14-事件驱动自动化.md) |
| **通过 MCP 开门锁，没有任何人确认** | MCP server 是第二个进程入口，**完全不经过图**。凡是实现在图节点里的东西（首当其冲是审批）在这个入口上都不存在 | [7](07-人在回路审批.md)坑四、[8](08-可信身份边界.md)坑三、[16](16-MCP工具共享.md) |
| **加了一种新设备，MCP 客户端看不到它** | 新增设备不会自动出现在 MCP 面前 —— 这一处的"自动派生"是没有的，而测试全绿 | [16](16-MCP工具共享.md)坑一 |
| **`interrupt()` 之前的设备控制执行了两次** | 含 `interrupt()` 的节点恢复时会**从头重跑**。这是 LangGraph 的机制，不是 bug | [7](07-人在回路审批.md)坑一、[18](18-全景图与不变量.md) 不变量十一 |
| **步骤失败了，系统卡住重试三次然后放弃** | 确定性错误（`unsupported_action` / `device_not_found`）本该跳过重试直接 replan。同样参数原样重放一百次也不会成功 | [6](06-Executor与Verifier.md)、[18](18-全景图与不变量.md) 不变量五 |
| **同样的 `turn_off` 错误无限重复** | replan 没把合法值回喂给 Planner。Planner 走 `with_structured_output` 而不是 `bind_tools`，**工具 docstring 根本到不了模型面前** —— 它写错动作名不是它笨，是没人告诉过它 | [5](05-Planner规划器.md)、[18](18-全景图与不变量.md) 不变量六 |

## 排查顺序建议

撞到看不懂的问题时，按这个顺序省时间：

```
1. 先跑一次全量测试
   PYTHONIOENCODING=utf-8 F:/Software/Anaconda/envs/langgraph/python.exe -m pytest -q
   → 全绿说明是你这一轮的输入/配置问题，不是代码坏了
   → 有红先看红的那几条，别急着看现象

2. 绕过 LLM 直接调工具（附录 A 的 4.4 实验 B）
   → 工具本身好的，问题就在"模型有没有调对"这一层
   → 不烧 token，不看模型脸色

3. 加 --trace 看路由和记忆判断
   python -m src.main --trace
   → 走错路径的问题（第 9 章那一类）在这里一眼可见

4. 加 --debug 让 DEBUG 通道落盘
   → graph_progress / node_latency 才会出现

5. 直接查设备真实状态
   PYTHONIOENCODING=utf-8 F:/Software/Anaconda/envs/langgraph/python.exe -m src.main status
   → "Agent 说做完了"和"真的做完了"是两件事
```

## 这个附录的局限

B.1 收的是**已经撞过**的报错。B.2 收的是**已经想到**的症状。
两者都没有任何机制保证完整 —— 它们是从十三次迭代的踩坑记录里归纳的，
也就是说：**这里的每一条都对应过一次真实的排查。**

如果你撞到了不在这一页上的东西，那是这本教程该补的，不是你的问题。

---

[← 附录 A 动手实验清单](附录A-动手实验清单.md) · [目录](README.md)
