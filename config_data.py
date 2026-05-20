"""全局配置文件。

建议阅读顺序：
1. 先看路径配置，理解项目把数据存到了哪里。
2. 再看文档分类和支持的文件类型，理解项目面向什么场景。
3. 最后看模型、切分、示例文档等参数，理解问答链路的默认行为。
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 缺少依赖时允许继续运行，只是不会自动加载 .env
    load_dotenv = None


# BASE_DIR 指向当前项目根目录，后面的路径几乎都基于它来拼接。
BASE_DIR = Path(__file__).resolve().parent
APP_NAME = "OfficeMate"

# 优先读取项目根目录下的 .env，保证 Streamlit 从任意入口启动时都能拿到环境变量。
ENV_PATH = BASE_DIR / ".env"
if load_dotenv and ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=False)

# storage 目录下会同时保存原始文档、JSON 日志和 Chroma 向量库。
STORAGE_DIR = BASE_DIR / "storage"
RAW_DOCUMENT_DIR = STORAGE_DIR / "raw_documents"
JSON_STORE_DIR = STORAGE_DIR / "json_store"
SAMPLE_DOC_DIR = BASE_DIR / "sample_docs"
RAGBENCH_DIR = STORAGE_DIR / "ragbench"
BENCHMARK_CORPUS_DIR = STORAGE_DIR / "benchmark_corpus"
BENCHMARK_CHROMA_DIR = STORAGE_DIR / "benchmark_chroma"
BENCHMARK_RUN_DIR = STORAGE_DIR / "benchmark_runs"
BENCHMARK_RUN_INDEX_PATH = BENCHMARK_RUN_DIR / "run_index.json"
LOCAL_EVAL_CORPUS_DIR = STORAGE_DIR / "local_eval_corpus"
LOCAL_EVAL_CHROMA_DIR = STORAGE_DIR / "local_eval_chroma"
LOCAL_EVAL_KB_DIR = STORAGE_DIR / "local_eval_kb"
LOCAL_EVAL_KB_INDEX_PATH = STORAGE_DIR / "local_eval_kb_index.json"

# 这三个 JSON 文件分别承担“文档索引 / 问答日志 / 用户反馈”职责。
DOCUMENT_INDEX_PATH = JSON_STORE_DIR / "documents.json"
QA_LOG_PATH = JSON_STORE_DIR / "qa_logs.json"
FEEDBACK_PATH = JSON_STORE_DIR / "feedback_logs.json"

# 页面上的分类下拉框、过滤条件和示例数据都会复用这里的分类定义。
DOCUMENT_CATEGORIES = [
    "员工手册",
    "HR制度",
    "财务制度",
    "IT支持",
    "行政流程",
    "综合公告",
]
CATEGORY_FILTER_OPTIONS = ["全部"] + DOCUMENT_CATEGORIES
SUPPORTED_FILE_TYPES = ["txt", "pdf", "docx", "xlsx", "csv", "zip"]
DEFAULT_VERSION = "v1.0"
EVALUATION_SAMPLE_PATH = BASE_DIR / "sample_docs" / "evaluation_samples.json"
COMPLEX_EVALUATION_SAMPLE_PATH = BASE_DIR / "sample_docs" / "complex_eval_samples.json"
benchmark_collection_prefix = "officemate_benchmark"
local_eval_collection_prefix = "officemate_local_eval"
benchmark_default_top_k = 5
benchmark_default_question_limit = 50
benchmark_default_subsets = ["techqa", "emanual", "delucionqa"]
benchmark_default_splits = ["test"]
benchmark_chunk_size = 1000
benchmark_chunk_overlap = 100
benchmark_max_split_char_number = 1200
upload_prepare_workers = 4
upload_embedding_workers = 3
embedding_batch_size = 16
upload_task_history_limit = 5

# 下面是向量库和文本切分相关配置。
# collection_name/persist_directory 控制 Chroma 把向量数据保存到哪里。
collection_name = "officemate_knowledge_base"
persist_directory = str(STORAGE_DIR / "chroma_db")
# chunk_size 和 chunk_overlap 控制文档切片大小及片段之间的重叠。
chunk_size = 800
chunk_overlap = 120
separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", " ", ""]
# 文本不超过这个长度时，直接作为一个片段入库，不再继续切分。
max_split_char_number = 800
# 这里实际用于控制默认返回多少条相似片段。
similarity_threshold = 4
hybrid_vector_weight = 0.65
hybrid_bm25_weight = 0.35
hybrid_fetch_k = 8
parallel_subtask_workers = 3

# 模型 provider 统一走 provider + model + base_url 配置。
# 当前默认的 `mlx` 仍按 OpenAI-compatible 协议接入，这样后续可以在不改业务层的情况下替换实现。
chat_provider = os.getenv("CHAT_PROVIDER", "openai_compatible").strip().lower()
rewrite_provider = os.getenv("REWRITE_PROVIDER", chat_provider).strip().lower()
task_provider = os.getenv("TASK_PROVIDER", chat_provider).strip().lower()
embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai_compatible").strip().lower()
rerank_provider = os.getenv("RERANK_PROVIDER", embedding_provider).strip().lower()

rewrite_model_name = os.getenv(
    "REWRITE_MODEL",
    os.getenv("OPENAI_REWRITE_MODEL", os.getenv("OPENAI_MODEL", os.getenv("MINIMAX_CHAT_MODEL", "MiniMax-M2.7"))),
)
rewrite_api_key = os.getenv(
    "REWRITE_API_KEY",
    os.getenv("OPENAI_REWRITE_API_KEY", os.getenv("OPENAI_API_KEY", os.getenv("MINIMAX_API_KEY", ""))),
)
rewrite_base_url = os.getenv(
    "REWRITE_BASE_URL",
    os.getenv("OPENAI_REWRITE_BASE_URL", os.getenv("OPENAI_BASE_URL", os.getenv("MINIMAX_BASE_URL", ""))),
)
task_model_name = os.getenv(
    "TASK_MODEL",
    os.getenv("OPENAI_TASK_MODEL", os.getenv("OPENAI_MODEL", os.getenv("MINIMAX_CHAT_MODEL", "MiniMax-M2.7"))),
)
task_api_key = os.getenv(
    "TASK_API_KEY",
    os.getenv("OPENAI_TASK_API_KEY", os.getenv("OPENAI_API_KEY", os.getenv("MINIMAX_API_KEY", ""))),
)
task_base_url = os.getenv(
    "TASK_BASE_URL",
    os.getenv("OPENAI_TASK_BASE_URL", os.getenv("OPENAI_BASE_URL", os.getenv("MINIMAX_BASE_URL", ""))),
)

embedding_model_name = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-0.6B-4bit-DWQ")
embedding_api_key = os.getenv("EMBEDDING_API_KEY", "local")
embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8000/v1")

rerank_model_name = os.getenv("RERANK_MODEL", "Qwen3-Reranker-0.6B-mlx-8Bit")
rerank_api_key = os.getenv("RERANK_API_KEY", embedding_api_key)
rerank_base_url = os.getenv("RERANK_BASE_URL", embedding_base_url)
rerank_timeout_seconds = float(os.getenv("RERANK_TIMEOUT_SECONDS", "30"))

chat_model_name = os.getenv("OPENAI_MODEL", os.getenv("MINIMAX_CHAT_MODEL", "MiniMax-M2.7"))
chat_api_key = os.getenv("OPENAI_API_KEY", os.getenv("MINIMAX_API_KEY", ""))
chat_base_url = os.getenv("OPENAI_BASE_URL", os.getenv("MINIMAX_BASE_URL", ""))

# benchmark 允许单独指定一套 chat / embedding / rerank 模型，避免影响主问答链路。
benchmark_chat_provider = os.getenv("BENCHMARK_CHAT_PROVIDER", chat_provider).strip().lower()
benchmark_chat_model_name = os.getenv("BENCHMARK_CHAT_MODEL", chat_model_name)
benchmark_chat_api_key = os.getenv("BENCHMARK_CHAT_API_KEY", chat_api_key)
benchmark_chat_base_url = os.getenv("BENCHMARK_CHAT_BASE_URL", chat_base_url)

benchmark_embedding_provider = os.getenv("BENCHMARK_EMBEDDING_PROVIDER", embedding_provider).strip().lower()
benchmark_embedding_model_name = os.getenv("BENCHMARK_EMBEDDING_MODEL", embedding_model_name)
benchmark_embedding_api_key = os.getenv("BENCHMARK_EMBEDDING_API_KEY", embedding_api_key)
benchmark_embedding_base_url = os.getenv("BENCHMARK_EMBEDDING_BASE_URL", embedding_base_url)

benchmark_rerank_provider = os.getenv("BENCHMARK_RERANK_PROVIDER", rerank_provider).strip().lower()
benchmark_rerank_model_name = os.getenv("BENCHMARK_RERANK_MODEL", rerank_model_name)
benchmark_rerank_api_key = os.getenv("BENCHMARK_RERANK_API_KEY", rerank_api_key)
benchmark_rerank_base_url = os.getenv("BENCHMARK_RERANK_BASE_URL", rerank_base_url)
benchmark_rerank_timeout_seconds = float(os.getenv("BENCHMARK_RERANK_TIMEOUT_SECONDS", str(rerank_timeout_seconds)))
# 会话历史和引用文档数量也在这里统一约束。
max_history_rounds = 10
max_reference_documents = 10
default_session_prefix = "officemate"
max_subtasks = 10

# 企业办公术语中存在大量简称、别名和模板化说法。
# 这里维护一份轻量同义词词典，供 QueryRewriter 做查询扩展。
QUERY_SYNONYMS = {
    "补卡": ["补签", "考勤更正", "打卡异常"],
    "调休": ["补休", "倒休"],
    "蓝票": ["增值税普通发票", "普通发票"],
    "红票": ["红字发票", "负数发票"],
    "VPN": ["虚拟专用网络", "远程访问"],
    "报销": ["费用报销", "报账"],
    "补贴": ["津贴", "出差补助", "差旅补贴"],
    "审批": ["审批流程", "发起申请", "流程审批"],
}

# 当检索不到足够材料时，系统会直接返回这个固定模板，避免模型硬编答案。
NO_EVIDENCE_MESSAGE = (
    "### 最终回答\n"
    "未找到明确依据，当前知识库中没有足够材料支撑这个问题。\n\n"
    "### 操作步骤/材料清单\n"
    "请先切换更准确的分类，或补充上传对应制度、流程和通知文档。\n\n"
    "### 风险提示\n"
    "在没有明确制度依据前，请不要直接执行流程，建议联系对应部门进一步确认。"
)

QUESTION_TYPE_LABELS = {
    "policy_qa": "制度问答",
    "process_guide": "流程指引",
    "material_list": "材料清单",
    "notice_summary": "通知总结",
}

# 示例文档配置：一键导入示例知识库时，就按这里定义的文件和元数据入库。
SAMPLE_DOCS = [
    {
        "file_name": "员工手册.txt",
        "category": "员工手册",
        "title": "员工手册（节选）",
        "version": "v2026.04",
    },
    {
        "file_name": "员工行为与职业发展规范.txt",
        "category": "员工手册",
        "title": "员工行为与职业发展规范",
        "version": "v2026.04",
    },
    {
        "file_name": "请假与考勤制度.txt",
        "category": "HR制度",
        "title": "请假与考勤制度",
        "version": "v2026.04",
    },
    {
        "file_name": "招聘转正与员工关怀制度.txt",
        "category": "HR制度",
        "title": "招聘转正与员工关怀制度",
        "version": "v2026.04",
    },
    {
        "file_name": "差旅与报销制度.txt",
        "category": "财务制度",
        "title": "差旅与报销制度",
        "version": "v2026.04",
    },
    {
        "file_name": "预算借款与付款管理制度.txt",
        "category": "财务制度",
        "title": "预算借款与付款管理制度",
        "version": "v2026.04",
    },
    {
        "file_name": "合同付款与发票管理规范.txt",
        "category": "财务制度",
        "title": "合同付款与发票管理规范",
        "version": "v2026.04",
    },
    {
        "file_name": "采购申请流程.txt",
        "category": "行政流程",
        "title": "采购申请流程",
        "version": "v2026.04",
    },
    {
        "file_name": "行政资产与办公支持管理办法.txt",
        "category": "行政流程",
        "title": "行政资产与办公支持管理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "用印、访客与档案流转流程.txt",
        "category": "行政流程",
        "title": "用印、访客与档案流转流程",
        "version": "v2026.04",
    },
    {
        "file_name": "IT服务台常见问题.txt",
        "category": "IT支持",
        "title": "IT服务台常见问题",
        "version": "v2026.04",
    },
    {
        "file_name": "账号权限与密码安全规范.txt",
        "category": "IT支持",
        "title": "账号权限与密码安全规范",
        "version": "v2026.04",
    },
    {
        "file_name": "终端设备与软件安装管理办法.txt",
        "category": "IT支持",
        "title": "终端设备与软件安装管理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "入职与权限开通流程.txt",
        "category": "综合公告",
        "title": "入职与权限开通流程",
        "version": "v2026.04",
    },
    {
        "file_name": "审批流转与通知发布规范.txt",
        "category": "综合公告",
        "title": "审批流转与通知发布规范",
        "version": "v2026.04",
    },
    {
        "file_name": "会议与跨部门协作规范.txt",
        "category": "综合公告",
        "title": "会议与跨部门协作规范",
        "version": "v2026.04",
    },
    {
        "file_name": "绩效目标与季度考核办法.txt",
        "category": "HR制度",
        "title": "绩效目标与季度考核办法",
        "version": "v2026.04",
    },
    {
        "file_name": "费用退款与冲销处理规范.txt",
        "category": "财务制度",
        "title": "费用退款与冲销处理规范",
        "version": "v2026.04",
    },
    {
        "file_name": "供应商准入与合作管理办法.txt",
        "category": "行政流程",
        "title": "供应商准入与合作管理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "会议室与公共资源预约管理办法.txt",
        "category": "行政流程",
        "title": "会议室与公共资源预约管理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "数据分级与对外发送规范.txt",
        "category": "IT支持",
        "title": "数据分级与对外发送规范",
        "version": "v2026.04",
    },
    {
        "file_name": "远程办公与居家协作规范.txt",
        "category": "员工手册",
        "title": "远程办公与居家协作规范",
        "version": "v2026.04",
    },
    {
        "file_name": "利益冲突与礼品接待规范.txt",
        "category": "员工手册",
        "title": "利益冲突与礼品接待规范",
        "version": "v2026.04",
    },
    {
        "file_name": "员工申诉与内部沟通机制.txt",
        "category": "员工手册",
        "title": "员工申诉与内部沟通机制",
        "version": "v2026.04",
    },
    {
        "file_name": "知识沉淀与交接记录规范.txt",
        "category": "员工手册",
        "title": "知识沉淀与交接记录规范",
        "version": "v2026.04",
    },
    {
        "file_name": "内部导师与带教管理办法.txt",
        "category": "员工手册",
        "title": "内部导师与带教管理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "办公行为与环境维护规范.txt",
        "category": "员工手册",
        "title": "办公行为与环境维护规范",
        "version": "v2026.04",
    },
    {
        "file_name": "员工荣誉与即时激励办法.txt",
        "category": "员工手册",
        "title": "员工荣誉与即时激励办法",
        "version": "v2026.04",
    },
    {
        "file_name": "试用期沟通与辅导规范.txt",
        "category": "员工手册",
        "title": "试用期沟通与辅导规范",
        "version": "v2026.04",
    },
    {
        "file_name": "加班与值班管理细则.txt",
        "category": "HR制度",
        "title": "加班与值班管理细则",
        "version": "v2026.04",
    },
    {
        "file_name": "实习生与外包人员协作管理规范.txt",
        "category": "HR制度",
        "title": "实习生与外包人员协作管理规范",
        "version": "v2026.04",
    },
    {
        "file_name": "员工异动与岗位交接制度.txt",
        "category": "HR制度",
        "title": "员工异动与岗位交接制度",
        "version": "v2026.04",
    },
    {
        "file_name": "薪资核对与个税资料提交流程.txt",
        "category": "HR制度",
        "title": "薪资核对与个税资料提交流程",
        "version": "v2026.04",
    },
    {
        "file_name": "员工档案与证明开具管理办法.txt",
        "category": "HR制度",
        "title": "员工档案与证明开具管理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "年度培训计划与学习档案办法.txt",
        "category": "HR制度",
        "title": "年度培训计划与学习档案办法",
        "version": "v2026.04",
    },
    {
        "file_name": "招聘面试评价与背调规范.txt",
        "category": "HR制度",
        "title": "招聘面试评价与背调规范",
        "version": "v2026.04",
    },
    {
        "file_name": "出勤异常与长期缺勤处理办法.txt",
        "category": "HR制度",
        "title": "出勤异常与长期缺勤处理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "备用金管理办法.txt",
        "category": "财务制度",
        "title": "备用金管理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "发票遗失与替代凭证处理规范.txt",
        "category": "财务制度",
        "title": "发票遗失与替代凭证处理规范",
        "version": "v2026.04",
    },
    {
        "file_name": "预付款与分期付款管理办法.txt",
        "category": "财务制度",
        "title": "预付款与分期付款管理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "成本归集与项目费用分摊规则.txt",
        "category": "财务制度",
        "title": "成本归集与项目费用分摊规则",
        "version": "v2026.04",
    },
    {
        "file_name": "营销活动费用管理规范.txt",
        "category": "财务制度",
        "title": "营销活动费用管理规范",
        "version": "v2026.04",
    },
    {
        "file_name": "供应商对账与结算周期规范.txt",
        "category": "财务制度",
        "title": "供应商对账与结算周期规范",
        "version": "v2026.04",
    },
    {
        "file_name": "低值物资报销与集采规则.txt",
        "category": "财务制度",
        "title": "低值物资报销与集采规则",
        "version": "v2026.04",
    },
    {
        "file_name": "押金与保证金管理规范.txt",
        "category": "财务制度",
        "title": "押金与保证金管理规范",
        "version": "v2026.04",
    },
    {
        "file_name": "快递收发与重要文件签收规范.txt",
        "category": "行政流程",
        "title": "快递收发与重要文件签收规范",
        "version": "v2026.04",
    },
    {
        "file_name": "办公区工位调整与搬迁流程.txt",
        "category": "行政流程",
        "title": "办公区工位调整与搬迁流程",
        "version": "v2026.04",
    },
    {
        "file_name": "车辆预约与公务用车管理办法.txt",
        "category": "行政流程",
        "title": "车辆预约与公务用车管理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "门禁卡与工牌补办流程.txt",
        "category": "行政流程",
        "title": "门禁卡与工牌补办流程",
        "version": "v2026.04",
    },
    {
        "file_name": "印刷品与宣传物料申请流程.txt",
        "category": "行政流程",
        "title": "印刷品与宣传物料申请流程",
        "version": "v2026.04",
    },
    {
        "file_name": "办公用品领用与盘点制度.txt",
        "category": "行政流程",
        "title": "办公用品领用与盘点制度",
        "version": "v2026.04",
    },
    {
        "file_name": "清洁与安全巡检管理规范.txt",
        "category": "行政流程",
        "title": "清洁与安全巡检管理规范",
        "version": "v2026.04",
    },
    {
        "file_name": "外部活动与场地布置申请流程.txt",
        "category": "行政流程",
        "title": "外部活动与场地布置申请流程",
        "version": "v2026.04",
    },
    {
        "file_name": "员工宿舍与临时住宿管理办法.txt",
        "category": "行政流程",
        "title": "员工宿舍与临时住宿管理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "邮箱组与通讯录管理规范.txt",
        "category": "IT支持",
        "title": "邮箱组与通讯录管理规范",
        "version": "v2026.04",
    },
    {
        "file_name": "数据备份与恢复申请流程.txt",
        "category": "IT支持",
        "title": "数据备份与恢复申请流程",
        "version": "v2026.04",
    },
    {
        "file_name": "账号异常登录与安全告警处置办法.txt",
        "category": "IT支持",
        "title": "账号异常登录与安全告警处置办法",
        "version": "v2026.04",
    },
    {
        "file_name": "软件许可证与订阅管理规范.txt",
        "category": "IT支持",
        "title": "软件许可证与订阅管理规范",
        "version": "v2026.04",
    },
    {
        "file_name": "硬件报修与备机借用流程.txt",
        "category": "IT支持",
        "title": "硬件报修与备机借用流程",
        "version": "v2026.04",
    },
    {
        "file_name": "移动设备与BYOD接入规范.txt",
        "category": "IT支持",
        "title": "移动设备与BYOD接入规范",
        "version": "v2026.04",
    },
    {
        "file_name": "系统变更与发布窗口管理办法.txt",
        "category": "IT支持",
        "title": "系统变更与发布窗口管理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "视频会议与远程演示支持规范.txt",
        "category": "IT支持",
        "title": "视频会议与远程演示支持规范",
        "version": "v2026.04",
    },
    {
        "file_name": "打印与扫描设备使用管理办法.txt",
        "category": "IT支持",
        "title": "打印与扫描设备使用管理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "节假日安排与调休通知规范.txt",
        "category": "综合公告",
        "title": "节假日安排与调休通知规范",
        "version": "v2026.04",
    },
    {
        "file_name": "项目立项与里程碑同步规范.txt",
        "category": "综合公告",
        "title": "项目立项与里程碑同步规范",
        "version": "v2026.04",
    },
    {
        "file_name": "例外审批与紧急事项升级办法.txt",
        "category": "综合公告",
        "title": "例外审批与紧急事项升级办法",
        "version": "v2026.04",
    },
    {
        "file_name": "客户接待与高层来访协同规范.txt",
        "category": "综合公告",
        "title": "客户接待与高层来访协同规范",
        "version": "v2026.04",
    },
    {
        "file_name": "周报月报与经营数据提交流程.txt",
        "category": "综合公告",
        "title": "周报月报与经营数据提交流程",
        "version": "v2026.04",
    },
    {
        "file_name": "公司级活动报名与名额管理办法.txt",
        "category": "综合公告",
        "title": "公司级活动报名与名额管理办法",
        "version": "v2026.04",
    },
    {
        "file_name": "内部制度修订与版本发布流程.txt",
        "category": "综合公告",
        "title": "内部制度修订与版本发布流程",
        "version": "v2026.04",
    },
    {
        "file_name": "公共邮箱与共享账号使用规范.txt",
        "category": "综合公告",
        "title": "公共邮箱与共享账号使用规范",
        "version": "v2026.04",
    },
]


def ensure_runtime_dirs() -> None:
    """确保运行时依赖的目录和 JSON 文件存在。

    这个函数的作用很像“项目启动时的本地初始化脚本”：
    - 没有目录就创建目录
    - 没有 JSON 文件就写入空数组
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_STORE_DIR.mkdir(parents=True, exist_ok=True)
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    RAGBENCH_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_EVAL_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_EVAL_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_EVAL_KB_DIR.mkdir(parents=True, exist_ok=True)

    for path in (DOCUMENT_INDEX_PATH, QA_LOG_PATH, FEEDBACK_PATH):
        if not path.exists():
            path.write_text("[]", encoding="utf-8")

    if not BENCHMARK_RUN_INDEX_PATH.exists():
        BENCHMARK_RUN_INDEX_PATH.write_text("[]", encoding="utf-8")
    if not LOCAL_EVAL_KB_INDEX_PATH.exists():
        LOCAL_EVAL_KB_INDEX_PATH.write_text("[]", encoding="utf-8")


# 模块导入时就执行一次初始化，这样其他模块几乎可以默认目录已经准备好了。
ensure_runtime_dirs()


# 下面两项是为了兼容旧版 demo 中可能仍然引用的配置变量。
md5_path = str(STORAGE_DIR / "legacy_md5.txt")
session_config = {
    "configurable": {
        "session_id": f"{default_session_prefix}_default",
    }
}
