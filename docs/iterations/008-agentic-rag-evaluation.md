# 阶段十二：Agentic RAG 与轨迹评测

## 实现内容

- `docs/knowledge/catalog.json`：设备型号与说明书文件映射。
- `src/knowledge/base.py`：本地 Markdown 知识索引和可解释词法检索。
- `src/knowledge/rag.py`：识别设备型号、检索、查询改写、带来源回答和无答案拒答子图。
- `src/evaluation/trajectory.py`：路由、状态、来源、检索、改写和拒答指标。

当前知识查询会被结构化 Router 识别为 `device_knowledge`，进入 RAG 子图，不调用设备控制工具。故障代码查询要求文档中出现相同代码，避免用相似文档猜测答案。

## 边界

当前版本使用本地 Markdown 和标准库词法检索，未安装向量数据库或 PDF 解析依赖。后续可在文档规模增长后替换为向量检索，并保留当前的型号过滤、引用和轨迹评测接口。
