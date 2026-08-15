# 阶段十三：事件驱动家庭自动化

## 目标

让 Agent 不只执行当前对话中的即时动作，还能创建未来固定时间或车辆事件触发的持久化例程。

## 已实现

- `src/automation/models.py`：例程、相对动作、运行批次、调度任务和车辆事件模型。
- `src/automation/planning.py`：LLM 生成动作在入库前的 Schema、工具白名单、设备名和安全边界校验。
- `src/automation/store.py`：SQLite 持久化、任务去重、取消和状态查询。
- `src/automation/scheduler.py`：后台 worker 与可注入虚拟时间的 `tick()`。
- `src/automation/executor.py`：复用现有设备工具和 Verifier。
- `src/automation/speaker.py`：音响闹钟适配接口与模拟实现。
- `src/automation/vehicle.py`：车辆 ETA/地理围栏事件与动态到家锚点。
- `src/tools/automation.py`：起床、车辆回家、列表和取消工具；列表工具返回每个动作的设备、参数、提前量、排期时间和执行状态。
- Automation Agent：独立职责提示词和工具权限。
- `create_scheduled_routine`：接受 LLM 动态生成的目标时间和任意受支持动作列表。
- `create_vehicle_arrival_routine`：接受 LLM 动态生成的车辆 ETA 相对动作。

## 场景

### 通用固定时间计划

```text
用户给出目标时间和目标
Automation Agent 生成 actions + offset_minutes
Schema 校验工具、设备、action 和安全边界
用户确认后持久化并调度
```

### 车辆回家

```text
到家前 15 分钟准备洗澡热水
到家前 10 分钟开启客厅空调
到家前 2 分钟打开窗帘
到家时打开客厅灯
```

同一趟行程的 ETA 更新会移动尚未执行任务，不会重复已经完成的动作。

## 安全边界

- 创建持续自动化必须经过 Human-in-the-loop 确认。
- 自动化例程不包含门锁解锁。
- 取消起床例程时同步取消已经写入音响后端的闹钟。
- 用户和住宅身份只从可信 `RunnableConfig` 获取。
- 每个设备动作执行后检查真实注册表状态。
- 真实汽车和音响通过适配接口接入，访问令牌不进入模型上下文。

## 测试

```powershell
F:\Software\Anaconda\envs\langgraph\python.exe -m pytest -q `
  -p no:cacheprovider tests/test_automation_routines.py
```

测试包含完整 LangGraph 路径：中文请求进入 Automation Agent，生成动态动作并在
`approval` 中断；批准前数据库为空，批准后才创建例程和调度任务。
