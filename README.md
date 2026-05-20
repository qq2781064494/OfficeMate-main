# OfficeMate

声明：本项目在黑马程序员 RAG 案例基础上做了扩展、重构和工程化整理，感谢原始案例的分享。

OfficeMate 是一个面向企业内部制度、流程和常见办公问题的智能知识问答系统。当前版本已经从早期的 `Streamlit + Chroma + JSON` 重构为 `FastAPI + Vue + MySQL + Milvus`，目标不再只是“做一个能问答的小 Demo”，而是搭建一套更完整的企业知识库服务：

- 文档入库
- RAG 问答
- Agent 问答
- 标准测评与本地测评
- 可追溯引用
- 任务异步执行
- 会话日志与反馈持久化

## 当前技术架构

当前主架构如下：

- 后端：`FastAPI`
- 前端：`Vue 3 + Vite`
- 关系型存储：`MySQL`
- 向量数据库：`Milvus`
- LLM 编排：`LangChain`
- 模型接口：`OpenAI-compatible API`

当前系统和旧版的关系：

- `FastAPI + Vue` 是当前主入口
- `MySQL + Milvus` 是当前主存储
- 旧的 `Streamlit` 页面和部分 JSON/Chroma 逻辑仍保留在仓库中，主要用于对照、兼容和学习，不再是主运行方式

## 项目介绍

OfficeMate 面向企业内部高频办公场景，例如：

- HR 制度：年假、病假、调休、补卡、转正
- 财务制度：报销、借款、发票、补贴、冲销
- 行政流程：采购申请、办公用品、用印、访客、工位调整
- IT 支持：账号权限、VPN、设备报修、软件安装
- 综合公告：入职流程、跨部门协作、制度通知、活动安排

项目目标不是做泛聊天机器人，而是围绕企业知识场景，搭建一套完整的：

- `文档上传 -> 文档解析 -> 切片向量化 -> 检索增强生成 -> 引用溯源 -> 日志反馈 -> 测评验证`

这套链路既适合课程项目、毕设展示和简历项目，也适合用来系统学习一个完整的 RAG 应用如何落地。

## 系统流程

### 1. 文档入库流程

主知识库入库链路如下：

1. 用户上传 `txt/pdf/docx/xlsx/csv/zip` 文档
2. 后端解析文档内容并清洗文本
3. 按切片规则拆成 chunk
4. 生成 embedding
5. 把向量写入 Milvus
6. 把文档元数据写入 MySQL
7. 原始文件保存在文件系统
8. 如果是批量任务，则通过 `task_id` 轮询执行状态

### 2. 标准 RAG 问答流程

标准 RAG 的主流程如下：

1. 接收用户问题与会话 ID
2. 识别问题类型与分类范围
3. Query Rewrite，对问题做规范化和术语扩展
4. Task Planning，对复杂问题拆解子任务
5. Hybrid Retrieval，执行向量检索 / BM25 / 混合检索
6. Evidence Selection，对候选片段做重排与证据筛选
7. 针对子任务生成答案
8. 汇总最终回答并补充引用来源
9. 把问答日志写入 MySQL

### 3. Agent 问答流程

项目中保留了两套 Agent 化实验链路：

- `Decision-ReAct`
  先判断问题复杂度，再决定是否走 rewrite、planning 和整合流程

- `Agent-ReAct-RAG`
  把改写、规划、检索、生成封装成工具，由 Agent 决定调用顺序

### 4. 实时思考过程展示

当前前端已经支持标准 RAG 的流式过程展示：

- 实时状态更新
- 分步骤显示当前执行阶段
- 问答完成后展示每一步的结构化结果

例如可以看到：

- 问题类型与改写结果
- 子任务拆解结果
- 检索与重排命中标题
- 子任务答案摘要
- 最终答案整理结果

## 我做的 RAG

### 1. 文档入库

知识库入库流程被拆成明确阶段：

- 文件接收
- 文档解析
- 文本清洗
- 切片
- embedding
- 写入 Milvus
- 元数据写入 MySQL

这样做的好处是文档管理、向量检索和业务日志解耦，后续扩展比较方便。

### 2. Query Rewrite

我没有直接用原始问题检索，而是先做问题改写：

- 规范口语化表达
- 做企业术语扩展
- 补充检索查询词

例如“补签怎么办”会扩展到“补卡、考勤更正、打卡异常”等相关术语。

### 3. Task Planning

如果问题比较复杂，系统会先拆成多个子任务，再分别检索和回答，最后统一汇总。  
这可以减少复杂问题中不同业务线信息混杂的问题。

### 4. Hybrid Retrieval

检索层支持三种策略：

- `vector`
- `bm25`
- `hybrid`

其中：

- 向量检索负责语义召回
- BM25 负责术语、标题和关键词命中
- 混合检索兼顾语义与规则词匹配

### 5. Evidence Selection 与 Rerank

检索回来后不会直接拼接上下文，而是继续做：

- rerank 模型重排
- 分类匹配
- 标题命中 bonus
- 任务 hints 命中
- 证据预算控制

这一层主要用来提升最终引用质量和答案准确率。

### 6. Structured Answer

最终回答尽量保持结构化输出，通常会包含：

- `最终回答`
- `步骤 / 材料 / 条件`
- `风险提示`
- `引用来源`

## 我做的 Agent

### 1. Decision-ReAct

这套方案会先判断当前问题的复杂度，再决定是否启用：

- 改写
- 任务拆分
- 汇总生成

它的意义在于让简单问题走轻链路，复杂问题走重链路。

### 2. Agent-ReAct-RAG

这套方案是真正的工具式 Agent。当前封装的核心工具包括：

- `rewrite_tool`
- `plan_tool`
- `retrieve_and_rerank_tool`
- `generate_final_answer_tool`

Agent 可以自己决定什么时候调用这些工具，而不是被动走固定流水线。

## 我做的测评

### 1. RAGBench 标准测评

项目支持 `RAGBench` 数据集测评，当前代码中常见 subset 包括：

- `techqa`
- `emanual`
- `delucionqa`

RAGBench 主要用于：

- 检索效果对比
- 不同参数组合对比
- 标准 benchmark 验证

### 2. 本地题库测评

项目还内置了本地题库，用来评估企业办公场景下的真实效果。常见样例包括：

- `evaluation_samples.json`
- `complex_eval_samples.json`
- `manual_test_questions.json`
- `rag_question_bank_200.json`

### 3. 测评中心支持的核心参数

当前前端测评中心已经支持这些实验参数：

- 切片规则
  - `chunk_size`
  - `chunk_overlap`
  - `max_split_char_number`
- 检索策略
  - `vector`
  - `bm25`
  - `hybrid`
- 是否启用 Query Rewrite
- 是否启用 Rerank
- 是否评测 Faithfulness

### 4. 历史 run 与详情查看

当前系统已经支持：

- 历史 benchmark run 补录到 MySQL
- 前端查看 run 列表
- 点击查看 run 详情
- 查看题目级明细
- 查看标准文档、检索文档、最终回答和命中上下文

更多指标解释可参考：

- [测评指标说明.md](/Users/weijiaxin/Documents/pythonwork/OfficeMate-main/OfficeMate-main/测评指标说明.md:1)

## 数据集与数据来源

项目当前主要使用三类数据：

### 1. 示例知识库数据

主系统演示知识库主要来自：

- [sample_docs](/Users/weijiaxin/Documents/pythonwork/OfficeMate-main/OfficeMate-main/sample_docs)

覆盖内容包括：

- 员工手册
- 请假与考勤制度
- 差旅与报销制度
- 采购申请流程
- IT 服务台常见问题
- 入职与权限开通流程

### 2. 本地题库测评数据

本地题库和人工测试问题主要来自：

- [sample_docs/evaluation_samples.json](/Users/weijiaxin/Documents/pythonwork/OfficeMate-main/OfficeMate-main/sample_docs/evaluation_samples.json:1)
- [sample_docs/complex_eval_samples.json](/Users/weijiaxin/Documents/pythonwork/OfficeMate-main/OfficeMate-main/sample_docs/complex_eval_samples.json:1)
- [sample_docs/manual_test_questions.json](/Users/weijiaxin/Documents/pythonwork/OfficeMate-main/OfficeMate-main/sample_docs/manual_test_questions.json:1)
- [sample_docs/rag_question_bank_200.json](/Users/weijiaxin/Documents/pythonwork/OfficeMate-main/OfficeMate-main/sample_docs/rag_question_bank_200.json:1)

### 3. RAGBench 标准测评数据

RAGBench 数据会被组织到本地 benchmark 目录，并用于构建语料与向量索引。相关代码主要在：

- [services/benchmark_eval_service.py](/Users/weijiaxin/Documents/pythonwork/OfficeMate-main/OfficeMate-main/services/benchmark_eval_service.py:1)
- [services/benchmark_store.py](/Users/weijiaxin/Documents/pythonwork/OfficeMate-main/OfficeMate-main/services/benchmark_store.py:1)
- [scripts/download_ragbench.py](/Users/weijiaxin/Documents/pythonwork/OfficeMate-main/OfficeMate-main/scripts/download_ragbench.py:1)

## 页面功能

当前 Vue 控制台的主要页面包括：

### 1. 总览页

- 服务状态查看
- 文档数量与测评概览
- 最近任务和系统摘要

### 2. 智能问答页

- 标准 RAG 问答
- Decision-ReAct 问答
- Agent-ReAct-RAG 问答
- 会话 ID 持久化
- 历史会话回填
- 流式思考过程展示

### 3. 文档上传页

- 上传知识库文档
- 触发异步入库任务

### 4. 文档管理页

- 查看已入库文档
- 删除文档

### 5. 测评中心页

- RAGBench 测评
- 本地题库测评
- 参数调节
- 历史 run 查看
- run 详情查看

### 6. 任务中心页

- 查看异步任务状态
- 轮询上传、建索引、测评任务进度

## 项目结构

```text
OfficeMate-main/
├── api/                    # FastAPI 路由层
├── agent_react_rag/        # 工具型 Agent 链路
├── decision_react/         # 决策型 Agent 链路
├── core/                   # 启动、数据库、基础设施
├── frontend/               # Vue 3 + Vite 前端
├── models/                 # MySQL ORM 模型
├── sample_docs/            # 示例知识库与本地题库
├── services/               # 主业务服务层
├── storage/                # 本地文件、benchmark 产物、历史结果
├── vectorstores/           # Milvus 适配层
├── main.py                 # FastAPI 启动入口
├── README.md
└── 项目代码导读.md
```

## 运行方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果你需要 benchmark 相关扩展依赖：

```bash
pip install -r requirements-benchmark.txt
```

### 2. 准备基础服务

你需要先准备：

- MySQL
- Milvus

常见本地开发配置示例：

- MySQL：`127.0.0.1:3306`
- Milvus：`127.0.0.1:19530`

### 3. 配置环境变量

请在本地准备 `.env` 或等价配置，至少包括：

- 数据库连接信息
- Milvus 连接信息
- Chat / Embedding / Rerank 模型接口信息

敏感信息不要提交到 Git。

### 4. 启动后端

```bash
uvicorn main:app --reload --port 8001
```

后端接口文档：

- Swagger UI: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

### 5. 启动前端

```bash
cd frontend
npm install
npm run dev
```

前端开发地址：

- [http://127.0.0.1:5173](http://127.0.0.1:5173)

### 6. 一体化访问

如果前端已经构建完成，也可以直接让 FastAPI 托管静态页面。

```bash
cd frontend
npm run build
cd ..
uvicorn main:app --reload --port 8001
```

然后访问：

- [http://127.0.0.1:8001](http://127.0.0.1:8001)

## 当前数据库设计

当前 MySQL 核心表包括：

- `documents`
- `qa_logs`
- `feedback_logs`
- `task_runs`
- `benchmark_runs`
- `benchmark_run_details`
- `local_eval_knowledge_bases`
- `benchmark_corpus_registry`

这些表分别负责：

- 文档元数据
- 问答日志
- 用户反馈
- 异步任务状态
- benchmark 运行摘要与明细
- 本地题库知识库注册
- benchmark 语料注册

## 当前向量存储设计

当前 Milvus 主要使用三类 collection：

- `officemate_main_chunks`
- `officemate_benchmark_chunks`
- `officemate_local_eval_chunks`

分别对应：

- 主知识库
- benchmark 语料
- 本地题库测评知识库

## 项目亮点

### 1. 不只是“检索后拼答案”，而是完整 RAG 链路

项目覆盖了：

- 文档上传
- 文档解析
- 切片向量化
- 检索增强生成
- 引用溯源
- 日志反馈
- 测评验证

### 2. 做了 Query Rewrite 和 Task Planning

这让系统不再只是“用户问题 -> 检索 -> 生成”，而是先做问题理解和任务拆解。

### 3. 支持多种检索策略对比

你可以直接比较：

- 纯向量检索
- 纯 BM25
- 混合检索

### 4. 同时实现了 RAG 与 Agent 两条路线

除了主 RAG 链路，还实现了：

- `Decision-ReAct`
- `Agent-ReAct-RAG`

适合做“从固定 RAG 到 Agent”的对比展示。

### 5. 补齐了评测闭环

项目同时具备：

- RAGBench 标准测评
- 本地题库测评
- 历史 run 保存
- 前端详情展示

### 6. 已经完成服务化重构

当前版本不再依赖单页脚本式 Demo，而是具备：

- FastAPI 服务接口
- Vue 控制台前端
- MySQL 持久化
- Milvus 向量库
- 后台任务执行

## 补充说明

- 仓库里仍然保留了一些旧版 `Streamlit`、`pages/`、`app.py` 相关内容，主要用于参考和兼容，不是当前主入口。
- 当前主入口是：
  - 后端：[main.py](/Users/weijiaxin/Documents/pythonwork/OfficeMate-main/OfficeMate-main/main.py:1)
  - 前端：[frontend/src/App.vue](/Users/weijiaxin/Documents/pythonwork/OfficeMate-main/OfficeMate-main/frontend/src/App.vue:1)
- 如果你想更快理解代码结构，建议继续阅读：
  - [项目代码导读.md](/Users/weijiaxin/Documents/pythonwork/OfficeMate-main/OfficeMate-main/项目代码导读.md:1)
