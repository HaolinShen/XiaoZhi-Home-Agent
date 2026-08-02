# 001 智慧生活智能体记忆系统 MVP

## 1. 背景

当前项目使用 LangGraph Checkpoint 保存消息历史，并通过 `thread_id` 隔离对话。项目最终会嵌入智慧生活 App，由一个智能体管控一个房屋，面向多家庭成员、多终端和多次会话提供设备控制服务。

现有实现可以支持基础多轮对话，但还存在以下问题：

- 每次启动随机生成 `thread_id`，无法由 App 稳定地恢复会话。
- 对话消息持续累积，模型调用成本和上下文溢出风险会不断增加。
- 家庭共享规则、个人偏好和临时会话上下文没有分层。
- 长期记忆配置尚未实现，缺少查看、修改和删除能力。
- SQLite Checkpointer 的创建逻辑重复，并且依赖缺失时会静默回退到内存。

## 2. 迭代目标

本次迭代实现一套可运行、可验证、可扩展的记忆系统 MVP：

1. 保留 LangGraph `thread_id`，但将其定位为内部会话检查点键。
2. 使用 `home_id` 隔离住宅，使用 `room_id` 和 `device_id` 定位空间与设备。
3. 使用 `user_id` 隔离个人偏好，使用 `session_id` 管理一次连续对话。
4. 结合 App 页面和终端位置提供可选的房间、设备上下文。
5. 建立家庭、房间、设备和个人等不同作用域的长期记忆。
6. 控制历史消息长度，避免上下文无限增长。
7. 允许用户查看、修改和删除长期记忆。

本次迭代暂不引入向量数据库、知识图谱和自动习惯推断。

## 3. 总体架构

```text
智慧生活 App
    │ home_id / user_id / session_id / client_id
    │ room_id? / device_id?
    ▼
Agent 服务
    ├── 实时状态层：设备平台
    ├── 会话记忆层：LangGraph Checkpoint
    ├── 家庭记忆层：SQLite
    └── 个人记忆层：SQLite
```

各层职责如下：

| 层级 | 隔离键 | 保存内容 | 生命周期 |
| --- | --- | --- | --- |
| 实时状态 | `home_id + room_id + device_id` | 开关、温度、在线状态 | 实时 |
| 会话记忆 | `session_id` | 对话、指代、当前任务、摘要 | 单次会话 |
| 家庭记忆 | `home_id` | 房间别名、家庭规则、公共场景 | 长期 |
| 空间记忆 | `home_id + room_id` | 房间别名、房间默认模式 | 长期 |
| 个人记忆 | `home_id + user_id`，可附带空间或设备 | 温度、灯光和场景偏好 | 长期 |

设备实时状态必须从设备平台读取，不能将历史对话中的设备状态当作当前事实。

## 4. 业务标识设计

每次 App 请求携带以下上下文：

```python
class AgentContext:
    home_id: str
    user_id: str
    session_id: str
    client_id: str
    room_id: str | None = None
    device_id: str | None = None
```

| 标识 | 用途 |
| --- | --- |
| `home_id` | 隔离不同房屋、设备和家庭记忆 |
| `room_id` | 定位同一住宅中的客厅、卧室等空间，可选 |
| `device_id` | 定位一台具体设备，可选 |
| `user_id` | 区分家庭成员、权限和个人偏好 |
| `session_id` | 隔离一次连续对话 |
| `client_id` | 区分手机、音箱和中控屏等入口 |

LangGraph 配置使用 `session_id` 作为 `thread_id`：

```python
config = {
    "configurable": {
        "thread_id": context.session_id,
        "home_id": context.home_id,
        "user_id": context.user_id,
        "client_id": context.client_id,
        "room_id": context.room_id,
        "device_id": context.device_id,
    }
}
```

建议由业务后端管理会话：连续交互复用当前 `session_id`，空闲 20 至 30 分钟、用户点击新对话或业务明确结束时创建新会话。不同终端默认使用独立会话。

标识的空间层级如下：

```text
home_id（一套住宅）
└── room_id（住宅内的房间或区域）
    └── device_id（房间内的具体设备）
```

`room_id` 和 `device_id` 是可选请求上下文。用户可能从 App 首页发起请求，此时没有预选空间；用户从房间页、设备详情页、固定音箱或中控屏发起请求时，App 应提供对应标识。

必须校验 `room_id` 属于当前 `home_id`，并校验 `device_id` 属于当前住宅以及对应房间，不能只信任客户端传值。

### 4.1 房间与设备定位规则

当用户指令没有完整说明操作目标时，按以下优先级解析：

```text
用户本轮明确提到的房间或设备
> App 当前页面提供的 device_id 或 room_id
> 固定终端绑定的 room_id
> 当前会话 active_device_id 或 active_room_id
> 用户个人默认空间（仅适用于已确认的偏好）
> 向用户澄清
```

定位规则：

- 用户本轮明确表达始终优先于历史上下文。
- `device_id` 已提供时，仍需校验设备类型是否符合指令。
- 用户说“打开客厅灯”时，可操作客厅唯一的灯；存在多盏灯时按名称、别名或设备组继续解析。
- 用户说“打开所有灯”时，必须根据已确定的 `home_id` 或 `room_id` 限定操作范围。
- 无法唯一定位时不得猜测，应询问用户具体房间或设备。
- 涉及门锁、燃气、报警器等高风险设备时，不使用模糊的历史指代直接执行。

### 4.2 请求上下文与会话状态的区别

- `AgentContext.room_id/device_id` 表示 App 或终端为本次请求提供的初始位置。
- `AgentState.active_room_id/active_device_id` 表示当前会话解析出的关注对象。
- 每次请求先使用可信请求上下文初始化或校正会话状态，再结合用户本轮表达更新关注对象。
- 新建会话时不继承旧会话的关注房间和设备。

## 5. 会话记忆设计

扩展 Agent 状态，显式保存摘要和当前关注对象：

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    conversation_summary: str
    active_room_id: str | None
    active_device_id: str | None
```

上下文治理规则：

- 保留最近 10 至 20 条原始消息。
- 达到 token 阈值时，将更早的对话合并进滚动摘要。
- 工具结果只保留设备 ID、执行结果和必要状态字段。
- 新会话只加载长期记忆，不加载全部旧会话消息。
- 摘要中的旧设备状态必须明确标记为历史信息。

每轮模型输入按以下顺序组织：

```text
系统规则
+ 请求身份和权限上下文
+ 当前 App 页面、房间和设备上下文
+ 家庭共享记忆
+ 当前用户偏好
+ 会话摘要
+ 最近消息
+ 当前输入
```

## 6. 长期记忆设计

MVP 使用 SQLite 保存结构化长期记忆：

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    home_id TEXT NOT NULL,
    user_id TEXT,
    room_id TEXT,
    device_id TEXT,
    scope TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    memory_value TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    expires_at DATETIME
);
```

字段约定：

- `scope`：`home`、`room`、`device` 或 `user`。
- `memory_type`：`preference`、`alias`、`routine` 或 `constraint`。
- `memory_key`：稳定的业务键，例如 `bedroom.sleep_temperature`。
- `memory_value`：JSON 格式的结构化值。
- `source`：用户原话或来源消息 ID，方便追溯。
- `expires_at`：临时信息的失效时间。

建议为以下字段增加唯一约束，重复表达同一偏好时更新原记录：

```text
home_id + user_id + room_id + device_id + scope + memory_type + memory_key
```

不同作用域的示例：

| 作用域 | 标识组合 | 示例 |
| --- | --- | --- |
| 家庭 | `home_id` | 晚上 23:00 后使用安静模式 |
| 房间 | `home_id + room_id` | 主卧也叫“爸爸妈妈房间” |
| 设备 | `home_id + device_id` | 客厅主灯支持色温调节 |
| 个人 | `home_id + user_id` | 用户整体偏好暖光 |
| 个人房间 | `home_id + user_id + room_id` | 用户在书房偏好冷白光 |
| 个人设备 | `home_id + user_id + device_id` | 用户睡眠时偏好主卧空调 25°C |

设备能力、型号和实时状态原则上属于设备平台数据，不重复保存为模型长期记忆；设备作用域主要保存用户命名、已确认偏好和业务别名。

## 7. 记忆写入规则

### 7.1 直接写入

用户明确提出长期意图时写入：

- “记住我喜欢暖光。”
- “以后睡觉时把空调设为 25 度。”

写入后向用户明确反馈保存的内容。

### 7.2 确认后写入

系统根据重复行为得到候选偏好时，必须先询问用户。本 MVP 只预留接口，不实现自动推断。

### 7.3 禁止写入

- “今天有点冷”等临时感受。
- “这次调到 27 度”等单次指令。
- 当前设备开关和在线状态。
- 可从设备平台实时获取的设备能力和归属关系。
- 未经用户确认的敏感推断。
- 权限和安全策略；这些内容应由业务权限系统管理。

冲突优先级为：

```text
安全与权限规则 > 用户当前明确指令 > 个人偏好 > 家庭默认偏好
```

## 8. 安全与数据隔离

- 记忆服务从可信请求上下文读取 `home_id`、`user_id`、`room_id` 和 `device_id`。
- 不允许模型通过工具参数自行指定任意家庭或用户。
- Repository 的所有查询必须包含 `home_id` 条件。
- 使用空间或设备过滤时，必须验证其属于当前 `home_id`。
- 读取个人记忆时同时校验 `user_id` 和成员关系。
- 删除家庭共享记忆需要家庭管理员权限。
- 日志不得记录完整敏感记忆或用户隐私原文。

## 9. 代码结构调整

```text
src/
├── agent/
│   ├── context.py
│   ├── graph.py
│   ├── prompts.py
│   └── state.py
├── memory/
│   ├── checkpoint.py
│   ├── extractor.py
│   ├── models.py
│   ├── repository.py
│   ├── service.py
│   └── summarizer.py
└── tools/
    └── memory.py
```

模块职责：

- `context.py`：定义和校验请求身份上下文。
- `checkpoint.py`：唯一负责创建 LangGraph Checkpointer。
- `models.py`：定义长期记忆数据模型。
- `repository.py`：实现 SQLite 增删改查。
- `service.py`：处理权限、去重、冲突和过期。
- `extractor.py`：从明确表达中提取候选记忆。
- `summarizer.py`：压缩长对话。
- `tools/memory.py`：提供受上下文约束的记忆工具。

## 10. 实施步骤

### 阶段一：会话基础

- 将 `langgraph-checkpoint-sqlite` 加入正式依赖。
- 合并重复的 Checkpointer 创建逻辑。
- SQLite 配置启用但初始化失败时明确报错。
- 接收并校验六类业务标识，其中 `room_id` 和 `device_id` 可选。
- 使用 `session_id` 作为 LangGraph `thread_id`。
- 支持新建、恢复和结束会话。
- 实现请求空间上下文与会话关注对象的同步规则。

### 阶段二：长期记忆

- 创建 `memories` 表和索引。
- 实现 Repository 和 Memory Service。
- 支持明确个人偏好和家庭规则的写入。
- 支持家庭、房间、设备和个人组合范围的记忆查询。
- 在模型调用前注入当前请求可访问的长期记忆。
- 提供查看、修改和删除接口。

### 阶段三：上下文治理

- 增加 token 或消息规模统计。
- 实现滚动摘要和最近消息窗口。
- 裁剪冗余工具结果。
- 设置会话过期和 Checkpoint 清理策略。
- 清理已过期长期记忆。

### 阶段四：后续增强

- 根据重复操作生成候选偏好。
- 用户确认后保存候选偏好。
- 记忆规模明显增长后评估向量检索。
- 增加记忆合并、冲突识别和置信度衰减。

实现说明：

- 相同用户、记忆键和值累计达到 3 次操作后生成候选，候选不会自动进入长期记忆。
- 候选支持查看、确认和拒绝；确认时才转换为个人偏好，并保留候选来源和置信度。
- 同一业务键再次写入时合并互补字段，显式新值覆盖旧值，并记录冲突审计。
- 长期未更新的记忆支持按周期衰减置信度，并设置最低置信度下限。
- 当前继续使用结构化 SQLite 查询；仅当活跃记忆达到配置的规模阈值时建议评估向量检索，未引入额外依赖。

## 11. App 侧能力

智慧生活 App 增加“智能体记忆”页面，支持：

- 查看个人记忆和家庭共享规则。
- 按房间和设备筛选相关记忆。
- 查看记忆来源和更新时间。
- 修改或删除单条记忆。
- 清除全部个人记忆。
- 关闭个人长期记忆功能。
- 由管理员管理家庭共享记忆。

## 12. 验收标准

1. 用户明确要求记住暖光偏好后，新会话仍能正确应用。
2. 两个家庭的数据完全隔离。
3. 同一家庭的公共规则可共享，个人偏好不会被其他成员读取。
4. 手机和音箱同时交互时，会话上下文不会串线。
5. App 从客厅页面发出“打开灯”时，Agent 能使用请求中的 `room_id` 定位客厅。
6. 用户先说“看看主卧”，再说“把灯关了”时，Agent 能使用会话中的 `active_room_id`。
7. 同一房间存在多台匹配设备且无法唯一确定时，Agent 会询问而不是猜测。
8. 客户端传入不属于当前住宅的 `room_id` 或 `device_id` 时请求被拒绝。
9. 单次设备指令不会覆盖长期偏好。
10. 用户删除记忆后，后续对话不再使用该记忆。
11. Agent 始终从设备平台查询实时状态，不相信历史消息中的旧状态。
12. 长对话的模型输入规模受到限制，不会无限增长。
13. SQLite 持久化配置失败时给出明确错误。
14. 模型工具调用无法访问请求上下文之外的家庭、房间和设备数据。

## 13. 本次迭代交付物

- 包含六类标识的 App 请求上下文和会话管理接口。
- 房间、设备解析优先级和歧义澄清机制。
- 稳定的 SQLite Checkpointer。
- 结构化长期记忆表、Repository 和 Service。
- 家庭记忆及个人记忆的读取、写入和删除能力。
- 基础滚动摘要和最近消息窗口。
- 家庭、房间、设备和用户记忆隔离测试。
- 记忆写入、目标定位、歧义处理和上下文压缩测试。
- README、配置示例和接口使用说明更新。
