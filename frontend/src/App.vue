<script setup>
import { computed, onMounted, onUnmounted, reactive, ref, watch } from "vue";
import { marked } from "marked";
import {
  askAgentReact,
  askChat,
  askDecisionReact,
  buildBenchmarkCorpus,
  buildBenchmarkIndex,
  createLocalEvalKnowledgeBase,
  deleteDocument,
  getBenchmarkRunDetail,
  getHealth,
  getStats,
  getTask,
  listBenchmarkRuns,
  listBenchmarkSubsets,
  listDocuments,
  listLocalEvalDatasets,
  listLocalEvalKnowledgeBases,
  listSessionLogs,
  listTasks,
  runBenchmark,
  runLocalEval,
  saveFeedback,
  seedDocuments,
  streamChat,
  uploadDocuments
} from "./api";
import { CATEGORY_OPTIONS, NAV_ITEMS, RETRIEVER_OPTIONS } from "./constants";

const CHAT_SESSION_STORAGE_KEY = "officemate_chat_session_id";

function buildSessionId() {
  return `console_${Date.now()}`;
}

function loadStoredSessionId() {
  if (typeof window === "undefined") {
    return buildSessionId();
  }
  return window.localStorage.getItem(CHAT_SESSION_STORAGE_KEY) || buildSessionId();
}

const activeView = ref("overview");
const loading = reactive({
  overview: false,
  upload: false,
  chat: false,
  documents: false,
  benchmark: false,
  tasks: false
});
const alerts = ref([]);

const health = ref(null);
const stats = ref(null);
const documents = ref([]);
const tasks = ref([]);
const benchmarkSubsets = ref([]);
const benchmarkRuns = ref([]);
const benchmarkRunDetail = ref(null);
const localEvalDatasets = ref([]);
const localEvalKnowledgeBases = ref([]);
const sessionLogs = ref([]);
const chatStatusText = ref("");
const liveStatusSteps = ref([]);
const livePhaseResults = ref([]);
const streamedAnswerText = ref("");

const uploadForm = reactive({
  files: [],
  category: "综合公告",
  version: "v2026.04",
  customTitle: ""
});

const chatForm = reactive({
  mode: "standard",
  question: "",
  sessionId: loadStoredSessionId(),
  category: "全部"
});
const chatResult = ref(null);
const feedbackForm = reactive({
  rating: "up",
  comment: ""
});

const benchmarkForm = reactive({
  subset: "",
  split: "test",
  retrieverStrategy: "hybrid",
  topK: 5,
  questionLimit: 20,
  enableQueryRewrite: true,
  enableRerank: true,
  enableFaithfulness: true,
  chunkSize: 1000,
  chunkOverlap: 100,
  maxSplitCharNumber: 1200,
  rebuildCorpus: false,
  rebuildIndex: false
});

const localEvalBuildForm = reactive({
  knowledgeBaseName: `sampledocs_${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "")}`,
  rebuild: false,
  chunkSize: 1000,
  chunkOverlap: 100,
  maxSplitCharNumber: 1200
});

const localEvalRunForm = reactive({
  knowledgeBaseId: "",
  knowledgeBaseName: "",
  datasetKey: "",
  datasetLabel: "",
  samplePath: "",
  retrieverStrategy: "hybrid",
  topK: 5,
  questionLimit: 20,
  enableQueryRewrite: true,
  enableRerank: true,
  enableFaithfulness: true,
  chunkSize: 1000,
  chunkOverlap: 100,
  maxSplitCharNumber: 1200
});

const currentTaskId = ref("");
let pollingTimer = null;

function pushAlert(message, tone = "info") {
  alerts.value = [{ id: Date.now(), message, tone }, ...alerts.value].slice(0, 4);
}

function taskTone(status) {
  if (status === "completed") return "success";
  if (status === "failed") return "danger";
  if (status === "running") return "warning";
  return "muted";
}

const selectedTask = computed(() => tasks.value.find((item) => item.id === currentTaskId.value) || null);
const renderedAnswerHtml = computed(() => markdownToHtml(chatResult.value?.answer || streamedAnswerText.value || ""));
const chatTrace = computed(() => {
  const trace = chatResult.value?.trace;
  return Array.isArray(trace) ? trace : [];
});
const chatProcessSummary = computed(() => {
  if (!chatResult.value) {
    return [];
  }
  return [
    { label: "问题类型", value: chatResult.value.question_type || "-" },
    { label: "规范化问题", value: chatResult.value.normalized_query || chatForm.question || "-" },
    { label: "检索查询", value: joinList(chatResult.value.retrieval_queries, " | ") },
    { label: "命中术语", value: joinList(chatResult.value.matched_terms) },
    { label: "预重排标题", value: joinList(chatResult.value.pre_rerank_titles) },
    { label: "最终检索标题", value: joinList(chatResult.value.retrieved_titles) }
  ];
});
const chatDecisionSummary = computed(() => {
  const decision = chatResult.value?.decision;
  if (!decision || typeof decision !== "object") {
    return [];
  }
  return [
    { label: "复杂度", value: decision.complexity || "-" },
    { label: "是否重写", value: decision.use_rewrite ? "是" : "否" },
    { label: "是否规划", value: decision.use_planner ? "是" : "否" },
    { label: "是否汇总", value: decision.use_synthesize ? "是" : "否" },
    { label: "建议分类", value: decision.suggested_category || "-" },
    { label: "决策理由", value: decision.reason || "-" }
  ];
});
const standardProcessCards = computed(() => {
  if (!chatResult.value) {
    return [];
  }
  const plannedTasks = Array.isArray(chatResult.value.planned_tasks) ? chatResult.value.planned_tasks : [];
  const answerPreview = (chatResult.value.answer || streamedAnswerText.value || "").split("### 引用文档")[0].trim();
  return [
    {
      title: "步骤 1：问题理解与改写",
      status: "completed",
      statusLabel: "已完成",
      items: [
        { label: "问题类型", value: chatResult.value.question_type || "-" },
        { label: "规范化问题", value: chatResult.value.normalized_query || chatForm.question || "-" },
        { label: "检索查询", value: joinList(chatResult.value.retrieval_queries, " | ") },
        { label: "命中术语", value: joinList(chatResult.value.matched_terms) }
      ]
    },
    {
      title: "步骤 2：任务拆解",
      status: "completed",
      statusLabel: "已完成",
      items: plannedTasks.length
        ? plannedTasks.map((task, index) => ({
            label: `子任务 ${index + 1}`,
            value: `${task.description || "-"} | 分类：${task.category || "-"} | 类型：${task.intent || "-"}`
          }))
        : [{ label: "拆解结果", value: "单任务回答，本轮未额外拆分子任务。" }]
    },
    {
      title: "步骤 3：检索与重排",
      status: "completed",
      statusLabel: "已完成",
      items: [
        { label: "预重排标题", value: joinList(chatResult.value.pre_rerank_titles) },
        { label: "最终检索标题", value: joinList(chatResult.value.retrieved_titles) },
        { label: "命中文本片段", value: `${(chatResult.value.retrieved_contexts || []).length || 0} 段` }
      ]
    },
    {
      title: "步骤 4：答案生成",
      status: "completed",
      statusLabel: "已完成",
      items: [
        { label: "回答摘要", value: answerPreview || "无" }
      ]
    }
  ];
});
const liveThinkingSteps = computed(() => {
  if (liveStatusSteps.value.length) {
    return liveStatusSteps.value;
  }
  return ["等待后端返回实时状态..."];
});
const liveThinkingStepDetails = computed(() =>
  liveThinkingSteps.value.map((message, index) => ({
    index,
    label: `步骤 ${index + 1}`,
    message,
    status: index === liveThinkingSteps.value.length - 1 ? "running" : "completed",
    statusLabel: index === liveThinkingSteps.value.length - 1 ? "进行中" : "已完成"
  }))
);
const liveProcessCards = computed(() => {
  if (!livePhaseResults.value.length) {
    return [];
  }
  const lastIndex = livePhaseResults.value.length - 1;
  return livePhaseResults.value.map((card, index) => ({
    ...card,
    statusLabel: index === lastIndex && loading.chat ? "进行中" : "已完成",
    statusClass: index === lastIndex && loading.chat ? "warning" : "success"
  }));
});
const benchmarkSummary = computed(() => benchmarkRunDetail.value?.summary || null);
const benchmarkDetails = computed(() => benchmarkRunDetail.value?.details || []);
const benchmarkMetricCards = computed(() => {
  if (!benchmarkSummary.value) {
    return [];
  }
  const retrieval = benchmarkSummary.value.retrieval_metrics || {};
  const rerank = benchmarkSummary.value.rerank_metrics || {};
  const ragas = benchmarkSummary.value.ragas_metrics || {};
  return [
    { label: "问题数", value: benchmarkSummary.value.question_count ?? "-" },
    { label: "Recall@1", value: safeMetric(retrieval.recall_at_1) },
    { label: "Recall@5", value: safeMetric(retrieval.recall_at_5) },
    { label: "MRR", value: safeMetric(retrieval.mrr) },
    { label: "重排后 MRR", value: safeMetric(rerank.post_mrr) },
    { label: "Faithfulness", value: safeMetric(ragas.faithfulness) }
  ];
});
const benchmarkMetricSections = computed(() => {
  if (!benchmarkSummary.value) {
    return [];
  }
  const retrieval = benchmarkSummary.value.retrieval_metrics || {};
  const rerank = benchmarkSummary.value.rerank_metrics || {};
  const ragas = benchmarkSummary.value.ragas_metrics || {};
  const chunkConfig = benchmarkSummary.value.chunk_config || {};
  return [
    {
      title: "切片与运行配置",
      items: [
        { label: "chunk_size", value: safeMetric(chunkConfig.chunk_size) },
        { label: "chunk_overlap", value: safeMetric(chunkConfig.chunk_overlap) },
        { label: "max_split_char_number", value: safeMetric(chunkConfig.max_split_char_number) },
        { label: "question_limit", value: safeMetric(benchmarkSummary.value.question_limit) },
        { label: "document_count", value: safeMetric(benchmarkSummary.value.document_count) },
        { label: "sample_count", value: safeMetric(benchmarkSummary.value.sample_count) }
      ]
    },
    {
      title: "检索指标",
      items: [
        { label: "Recall@1", value: safeMetric(retrieval.recall_at_1) },
        { label: "Recall@3", value: safeMetric(retrieval.recall_at_3) },
        { label: "Recall@5", value: safeMetric(retrieval.recall_at_5) },
        { label: "Hit@1", value: safeMetric(retrieval.hit_rate_at_1) },
        { label: "Hit@3", value: safeMetric(retrieval.hit_rate_at_3) },
        { label: "Hit@5", value: safeMetric(retrieval.hit_rate_at_5) },
        { label: "MRR", value: safeMetric(retrieval.mrr) }
      ]
    },
    {
      title: "重排指标",
      items: [
        { label: "status", value: safeMetric(rerank.status) },
        { label: "pre_hit_rate@1", value: safeMetric(rerank.pre_hit_rate_at_1) },
        { label: "post_hit_rate@1", value: safeMetric(rerank.post_hit_rate_at_1) },
        { label: "pre_hit_rate@3", value: safeMetric(rerank.pre_hit_rate_at_3) },
        { label: "post_hit_rate@3", value: safeMetric(rerank.post_hit_rate_at_3) },
        { label: "pre_mrr", value: safeMetric(rerank.pre_mrr) },
        { label: "post_mrr", value: safeMetric(rerank.post_mrr) },
        { label: "delta_mrr", value: safeMetric(rerank.delta_mrr) },
        { label: "win_rate", value: safeMetric(rerank.win_rate) },
        { label: "tie_rate", value: safeMetric(rerank.tie_rate) },
        { label: "lose_rate", value: safeMetric(rerank.lose_rate) },
        { label: "avg_rank_improvement", value: safeMetric(rerank.avg_rank_improvement) }
      ]
    },
    {
      title: "RAGAS 指标",
      items: [
        { label: "status", value: safeMetric(ragas.status) },
        { label: "faithfulness", value: safeMetric(ragas.faithfulness) },
        { label: "answer_relevancy", value: safeMetric(ragas.answer_relevancy) },
        { label: "context_precision", value: safeMetric(ragas.context_precision) },
        { label: "context_recall", value: safeMetric(ragas.context_recall) },
        { label: "error", value: safeMetric(ragas.error) }
      ]
    }
  ];
});

marked.setOptions({
  breaks: true,
  gfm: true
});

async function refreshOverview() {
  loading.overview = true;
  try {
    const [healthData, statsData, taskData] = await Promise.all([
      getHealth(),
      getStats(),
      listTasks(8)
    ]);
    health.value = healthData;
    stats.value = statsData;
    tasks.value = taskData;
  } catch (error) {
    pushAlert(error.message, "danger");
  } finally {
    loading.overview = false;
  }
}

async function refreshDocuments() {
  loading.documents = true;
  try {
    documents.value = await listDocuments();
  } catch (error) {
    pushAlert(error.message, "danger");
  } finally {
    loading.documents = false;
  }
}

async function refreshTasks() {
  try {
    tasks.value = await listTasks(20);
    if (currentTaskId.value) {
      const task = await getTask(currentTaskId.value);
      const existingIndex = tasks.value.findIndex((item) => item.id === task.id);
      if (existingIndex >= 0) {
        tasks.value.splice(existingIndex, 1, task);
      } else {
        tasks.value.unshift(task);
      }
    }
  } catch (error) {
    pushAlert(error.message, "danger");
  }
}

async function refreshSessionLogsForCurrentSession({ silent = false, clearCurrentAnswer = false } = {}) {
  const sessionId = (chatForm.sessionId || "").trim();
  if (!sessionId) {
    sessionLogs.value = [];
    if (clearCurrentAnswer) {
      chatResult.value = null;
      streamedAnswerText.value = "";
      liveStatusSteps.value = [];
      livePhaseResults.value = [];
    }
    if (!silent) {
      chatStatusText.value = "当前没有可用的会话 ID。";
    }
    return;
  }
  try {
    const logs = await listSessionLogs(sessionId, 20);
    sessionLogs.value = logs;
    if (clearCurrentAnswer) {
      chatResult.value = null;
      streamedAnswerText.value = "";
      liveStatusSteps.value = [];
      livePhaseResults.value = [];
    }
    if (!silent) {
      chatStatusText.value = logs.length
        ? `已载入当前会话的 ${logs.length} 条历史记录。`
        : "当前会话暂无历史记录。";
    }
  } catch (error) {
    sessionLogs.value = [];
    if (!silent) {
      chatStatusText.value = "读取会话历史失败，请稍后重试。";
      pushAlert(error.message, "danger");
    }
  }
}

async function refreshBenchmarkData() {
  loading.benchmark = true;
  try {
    const [subsets, runs, datasets, knowledgeBases] = await Promise.all([
      listBenchmarkSubsets(),
      listBenchmarkRuns(10),
      listLocalEvalDatasets(),
      listLocalEvalKnowledgeBases()
    ]);
    benchmarkSubsets.value = subsets;
    benchmarkRuns.value = runs;
    localEvalDatasets.value = datasets;
    localEvalKnowledgeBases.value = knowledgeBases;
    if (!benchmarkForm.subset && subsets.length) {
      benchmarkForm.subset = subsets[0].subset;
    }
    if (!localEvalRunForm.datasetKey && datasets.length) {
      applyDataset(datasets[0]);
    }
    if (!localEvalRunForm.knowledgeBaseId && knowledgeBases.length) {
      applyKnowledgeBase(knowledgeBases[0]);
    }
  } catch (error) {
    pushAlert(error.message, "danger");
  } finally {
    loading.benchmark = false;
  }
}

async function openBenchmarkRunDetail(runId) {
  try {
    loading.benchmark = true;
    benchmarkRunDetail.value = await getBenchmarkRunDetail(runId);
  } catch (error) {
    pushAlert(error.message, "danger");
  } finally {
    loading.benchmark = false;
  }
}

function startPolling() {
  stopPolling();
  pollingTimer = window.setInterval(refreshTasks, 4000);
}

function stopPolling() {
  if (pollingTimer) {
    window.clearInterval(pollingTimer);
    pollingTimer = null;
  }
}

async function submitUpload() {
  if (!uploadForm.files.length) {
    pushAlert("请先选择至少一个文件。", "warning");
    return;
  }
  loading.upload = true;
  try {
    const task = await uploadDocuments({
      files: uploadForm.files,
      category: uploadForm.category,
      version: uploadForm.version,
      customTitle: uploadForm.customTitle
    });
    currentTaskId.value = task.task_id;
    pushAlert(`上传任务已创建：${task.task_id}`, "success");
    await refreshTasks();
  } catch (error) {
    pushAlert(error.message, "danger");
  } finally {
    loading.upload = false;
  }
}

async function seedSampleDocuments() {
  loading.upload = true;
  try {
    const task = await seedDocuments(true);
    currentTaskId.value = task.task_id;
    pushAlert(`示例文档导入任务已创建：${task.task_id}`, "success");
    await refreshTasks();
  } catch (error) {
    pushAlert(error.message, "danger");
  } finally {
    loading.upload = false;
  }
}

async function submitChat() {
  if (!chatForm.question.trim()) {
    pushAlert("请输入问题。", "warning");
    return;
  }
  loading.chat = true;
  chatResult.value = null;
  streamedAnswerText.value = "";
  liveStatusSteps.value = [];
  livePhaseResults.value = [];
  chatStatusText.value = "正在调用问答链路，请稍候...";
  try {
    const payload = {
      question: chatForm.question,
      session_id: chatForm.sessionId,
      category: chatForm.category,
      use_history: true,
      persist_log: true,
      include_references: true,
      enable_query_rewrite: true,
      enable_rerank: true
    };
    if (chatForm.mode === "standard") {
      await streamChat(payload, {
        onStatus(event) {
          const message = event.message || "";
          chatStatusText.value = message || "正在处理...";
          if (message && liveStatusSteps.value[liveStatusSteps.value.length - 1] !== message) {
            liveStatusSteps.value = [...liveStatusSteps.value, message];
          }
        },
        onChunk(event) {
          streamedAnswerText.value += event.content || "";
        },
        onPhaseResult(event) {
          const nextCard = {
            phase: event.phase,
            title: event.title || event.phase || "阶段结果",
            items: Array.isArray(event.items) ? event.items : []
          };
          const existingIndex = livePhaseResults.value.findIndex((item) => item.phase === nextCard.phase);
          if (existingIndex >= 0) {
            const next = [...livePhaseResults.value];
            next.splice(existingIndex, 1, nextCard);
            livePhaseResults.value = next;
          } else {
            livePhaseResults.value = [...livePhaseResults.value, nextCard];
          }
        },
        onMeta(event) {
          chatResult.value = event;
          if (!chatResult.value.answer) {
            chatResult.value.answer = streamedAnswerText.value;
          }
        },
        onError(event) {
          const message = event.message || "流式问答失败。";
          chatStatusText.value = message;
          pushAlert(message, "danger");
        },
        onDone() {
          if (!chatResult.value) {
            chatResult.value = {
              answer: streamedAnswerText.value,
              question_type: "",
              retrieved_titles: [],
              retrieval_queries: [],
              pre_rerank_titles: [],
              matched_terms: []
            };
          }
        }
      });
    } else if (chatForm.mode === "decision") {
      chatResult.value = await askDecisionReact(payload);
    } else {
      chatResult.value = await askAgentReact(payload);
    }
    sessionLogs.value = await listSessionLogs(chatForm.sessionId, 20);
    chatStatusText.value = "问答完成。";
    pushAlert("问答完成。", "success");
    await refreshOverview();
  } catch (error) {
    chatStatusText.value = "问答失败，请查看提示信息。";
    pushAlert(error.message, "danger");
  } finally {
    loading.chat = false;
  }
}

async function submitFeedback() {
  if (!chatResult.value?.qa_log_id) {
    pushAlert("当前没有可反馈的问答记录。", "warning");
    return;
  }
  try {
    await saveFeedback({
      qa_log_id: chatResult.value.qa_log_id,
      rating: feedbackForm.rating,
      comment: feedbackForm.comment,
      session_id: chatForm.sessionId
    });
    pushAlert("反馈已提交。", "success");
    feedbackForm.comment = "";
  } catch (error) {
    pushAlert(error.message, "danger");
  }
}

async function removeDocument(documentId) {
  if (!window.confirm("确认删除这份文档吗？这会同步删除向量索引。")) {
    return;
  }
  try {
    await deleteDocument(documentId);
    pushAlert("文档已删除。", "success");
    await refreshDocuments();
    await refreshOverview();
  } catch (error) {
    pushAlert(error.message, "danger");
  }
}

async function launchBenchmarkCorpus() {
  try {
    const task = await buildBenchmarkCorpus({
      subset: benchmarkForm.subset,
      splits: [benchmarkForm.split],
      rebuild: benchmarkForm.rebuildCorpus
    });
    currentTaskId.value = task.task_id;
    pushAlert(`RAGBench 语料任务已创建：${task.task_id}`, "success");
  } catch (error) {
    pushAlert(error.message, "danger");
  }
}

async function launchBenchmarkIndex() {
  try {
    const task = await buildBenchmarkIndex({
      subset: benchmarkForm.subset,
      rebuild: benchmarkForm.rebuildIndex,
      chunk_size: benchmarkForm.chunkSize,
      chunk_overlap: benchmarkForm.chunkOverlap,
      max_split_char_number: benchmarkForm.maxSplitCharNumber
    });
    currentTaskId.value = task.task_id;
    pushAlert(`RAGBench 索引任务已创建：${task.task_id}`, "success");
  } catch (error) {
    pushAlert(error.message, "danger");
  }
}

async function launchBenchmarkRun() {
  try {
    const task = await runBenchmark({
      subset: benchmarkForm.subset,
      split: benchmarkForm.split,
      retriever_strategy: benchmarkForm.retrieverStrategy,
      top_k: benchmarkForm.topK,
      question_limit: benchmarkForm.questionLimit,
      enable_query_rewrite: benchmarkForm.enableQueryRewrite,
      enable_ragas: true,
      enable_faithfulness: benchmarkForm.enableFaithfulness,
      enable_rerank: benchmarkForm.enableRerank,
      rebuild_corpus: benchmarkForm.rebuildCorpus,
      rebuild_index: benchmarkForm.rebuildIndex,
      chunk_size: benchmarkForm.chunkSize,
      chunk_overlap: benchmarkForm.chunkOverlap,
      max_split_char_number: benchmarkForm.maxSplitCharNumber
    });
    currentTaskId.value = task.task_id;
    pushAlert(`RAGBench 评测任务已创建：${task.task_id}`, "success");
  } catch (error) {
    pushAlert(error.message, "danger");
  }
}

async function launchLocalEvalBuild() {
  try {
    const task = await createLocalEvalKnowledgeBase({
      knowledge_base_name: localEvalBuildForm.knowledgeBaseName,
      rebuild: localEvalBuildForm.rebuild,
      chunk_size: localEvalBuildForm.chunkSize,
      chunk_overlap: localEvalBuildForm.chunkOverlap,
      max_split_char_number: localEvalBuildForm.maxSplitCharNumber
    });
    currentTaskId.value = task.task_id;
    pushAlert(`本地题库构建任务已创建：${task.task_id}`, "success");
  } catch (error) {
    pushAlert(error.message, "danger");
  }
}

async function launchLocalEvalRun() {
  try {
    const task = await runLocalEval({
      knowledge_base_id: localEvalRunForm.knowledgeBaseId,
      knowledge_base_name: localEvalRunForm.knowledgeBaseName,
      dataset_key: localEvalRunForm.datasetKey,
      dataset_label: localEvalRunForm.datasetLabel,
      sample_path: localEvalRunForm.samplePath,
      retriever_strategy: localEvalRunForm.retrieverStrategy,
      top_k: localEvalRunForm.topK,
      question_limit: localEvalRunForm.questionLimit,
      selected_question_ids: [],
      enable_query_rewrite: localEvalRunForm.enableQueryRewrite,
      enable_ragas: true,
      enable_faithfulness: localEvalRunForm.enableFaithfulness,
      enable_rerank: localEvalRunForm.enableRerank,
      chunk_size: localEvalRunForm.chunkSize,
      chunk_overlap: localEvalRunForm.chunkOverlap,
      max_split_char_number: localEvalRunForm.maxSplitCharNumber
    });
    currentTaskId.value = task.task_id;
    pushAlert(`本地题库评测任务已创建：${task.task_id}`, "success");
  } catch (error) {
    pushAlert(error.message, "danger");
  }
}

function applyDataset(dataset) {
  localEvalRunForm.datasetKey = dataset.dataset_key;
  localEvalRunForm.datasetLabel = dataset.dataset_label;
  localEvalRunForm.samplePath = dataset.sample_path;
}

function applyKnowledgeBase(knowledgeBase) {
  localEvalRunForm.knowledgeBaseId = knowledgeBase.knowledge_base_id;
  localEvalRunForm.knowledgeBaseName = knowledgeBase.knowledge_base_name;
}

function markdownToHtml(markdownText) {
  return marked.parse(markdownText || "");
}

function safeMetric(value) {
  if (value === null || value === undefined || value === "" || Number.isNaN(value)) {
    return "-";
  }
  if (typeof value === "number") {
    return value.toFixed(4);
  }
  return String(value);
}

function renderedLogAnswer(log) {
  return markdownToHtml(log.answer || "");
}

function hitLabel(detail) {
  return detail.retrieval_hit ? "命中" : "未命中";
}

function hitTone(detail) {
  return detail.retrieval_hit ? "success" : "danger";
}

function joinList(values, separator = "、") {
  if (!Array.isArray(values) || !values.length) {
    return "无";
  }
  return values.join(separator);
}

function formatTraceStep(step) {
  const labels = {
    understand_question_tool: "问题理解",
    retrieve_evidence_tool: "证据检索",
    generate_answers_tool: "生成子答案",
    finalize_answer_tool: "汇总最终答案",
    rewrite_tool: "问题改写",
    plan_tool: "任务拆解",
    retrieve_and_rerank_tool: "检索与重排",
    generate_final_answer_tool: "生成最终答案"
  };
  return labels[step] || step || "未知步骤";
}

function createNewSession() {
  chatForm.sessionId = buildSessionId();
  chatResult.value = null;
  sessionLogs.value = [];
  chatStatusText.value = "已创建新会话，当前还没有历史记录。";
  streamedAnswerText.value = "";
  liveStatusSteps.value = [];
  livePhaseResults.value = [];
  pushAlert(`已创建新会话：${chatForm.sessionId}`, "success");
}

watch(
  () => chatForm.sessionId,
  async (value, oldValue) => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(CHAT_SESSION_STORAGE_KEY, value || buildSessionId());
    if (!value || value === oldValue) {
      return;
    }
    await refreshSessionLogsForCurrentSession({ silent: false, clearCurrentAnswer: true });
  }
);

onMounted(async () => {
  startPolling();
  await Promise.all([refreshOverview(), refreshDocuments(), refreshBenchmarkData()]);
  await refreshSessionLogsForCurrentSession({ silent: false, clearCurrentAnswer: true });
});

onUnmounted(() => {
  stopPolling();
});
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <span class="brand-mark">OM</span>
        <div>
          <h1>OfficeMate</h1>
          <p>FastAPI + Vue 工作台</p>
        </div>
      </div>

      <nav class="nav-list">
        <button
          v-for="item in NAV_ITEMS"
          :key="item.key"
          class="nav-item"
          :class="{ active: activeView === item.key }"
          @click="activeView = item.key"
        >
          {{ item.label }}
        </button>
      </nav>

      <div class="sidebar-card" v-if="health">
        <div class="sidebar-label">服务状态</div>
        <div class="status-row">
          <span class="dot success"></span>
          {{ health.status }} · MySQL {{ health.mysql_database }}
        </div>
        <div class="status-row muted">Milvus {{ health.milvus_host }}:{{ health.milvus_port }}</div>
      </div>
    </aside>

    <main class="main-panel">
      <header class="topbar">
        <div>
          <h2>{{ NAV_ITEMS.find((item) => item.key === activeView)?.label }}</h2>
          <p>面向文档入库、问答、Agent 与测评的统一控制台。</p>
        </div>
        <button class="ghost-button" @click="refreshOverview">刷新总览</button>
      </header>

      <section class="alerts" v-if="alerts.length">
        <article
          v-for="alert in alerts"
          :key="alert.id"
          class="alert"
          :class="alert.tone"
        >
          {{ alert.message }}
        </article>
      </section>

      <section v-if="activeView === 'overview'" class="view-grid">
        <article class="panel">
          <div class="panel-title">核心指标</div>
          <div class="stat-grid" v-if="stats">
            <div class="stat-card">
              <strong>{{ stats.document_count }}</strong>
              <span>知识文档</span>
            </div>
            <div class="stat-card">
              <strong>{{ stats.category_count }}</strong>
              <span>分类数量</span>
            </div>
            <div class="stat-card">
              <strong>{{ stats.qa_count }}</strong>
              <span>问答记录</span>
            </div>
            <div class="stat-card">
              <strong>{{ stats.feedback_count }}</strong>
              <span>反馈数量</span>
            </div>
          </div>
        </article>

        <article class="panel">
          <div class="panel-title">最近任务</div>
          <div class="list-stack">
            <div v-for="task in tasks.slice(0, 6)" :key="task.id" class="task-row" @click="currentTaskId = task.id">
              <div>
                <div class="task-name">{{ task.task_type }}</div>
                <div class="task-meta">{{ task.id }}</div>
              </div>
              <span class="badge" :class="taskTone(task.status)">{{ task.status }}</span>
            </div>
          </div>
        </article>
      </section>

      <section v-else-if="activeView === 'upload'" class="view-grid two-up">
        <article class="panel">
          <div class="panel-title">上传知识文档</div>
          <div class="form-grid">
            <label class="field">
              <span>分类</span>
              <select v-model="uploadForm.category">
                <option v-for="category in CATEGORY_OPTIONS.filter((item) => item !== '全部')" :key="category" :value="category">
                  {{ category }}
                </option>
              </select>
            </label>
            <label class="field">
              <span>版本</span>
              <input v-model="uploadForm.version" placeholder="例如 v2026.04" />
            </label>
            <label class="field span-2">
              <span>自定义标题</span>
              <input v-model="uploadForm.customTitle" placeholder="可选，留空则默认用文件名" />
            </label>
            <label class="field span-2">
              <span>文件</span>
              <input type="file" multiple @change="uploadForm.files = Array.from($event.target.files || [])" />
            </label>
          </div>

          <div class="action-row">
            <button class="primary-button" :disabled="loading.upload" @click="submitUpload">创建上传任务</button>
            <button class="ghost-button" :disabled="loading.upload" @click="seedSampleDocuments">导入示例文档</button>
          </div>
        </article>

        <article class="panel">
          <div class="panel-title">当前任务</div>
          <div v-if="selectedTask" class="task-detail">
            <div><strong>ID：</strong>{{ selectedTask.id }}</div>
            <div><strong>类型：</strong>{{ selectedTask.task_type }}</div>
            <div><strong>状态：</strong><span class="badge" :class="taskTone(selectedTask.status)">{{ selectedTask.status }}</span></div>
            <div><strong>阶段：</strong>{{ selectedTask.progress_stage }}</div>
            <div><strong>说明：</strong>{{ selectedTask.progress_message }}</div>
            <pre class="json-box">{{ JSON.stringify(selectedTask.result_json, null, 2) }}</pre>
          </div>
          <div v-else class="empty-state">选择一个任务或先创建上传任务。</div>
        </article>
      </section>

      <section v-else-if="activeView === 'chat'" class="view-grid two-up">
        <article class="panel">
          <div class="panel-title">智能问答</div>
          <div class="form-grid">
            <label class="field">
              <span>模式</span>
              <select v-model="chatForm.mode">
                <option value="standard">标准 RAG</option>
                <option value="decision">Decision-ReAct</option>
                <option value="agent">Agent-ReAct-RAG</option>
              </select>
            </label>
            <label class="field">
              <span>分类</span>
              <select v-model="chatForm.category">
                <option v-for="category in CATEGORY_OPTIONS" :key="category" :value="category">{{ category }}</option>
              </select>
            </label>
            <label class="field span-2">
              <span>会话 ID</span>
              <input v-model="chatForm.sessionId" />
            </label>
            <div class="session-toolbar span-2">
              <button class="ghost-button" type="button" @click="createNewSession">新建会话</button>
            </div>
            <label class="field span-2">
              <span>问题</span>
              <textarea v-model="chatForm.question" rows="8" placeholder="例如：请假需要提前多久申请？"></textarea>
            </label>
          </div>
          <div class="action-row">
            <button class="primary-button" :disabled="loading.chat" @click="submitChat">
              {{ loading.chat ? "问答进行中..." : "开始问答" }}
            </button>
          </div>
          <div v-if="chatStatusText" class="status-note">
            {{ chatStatusText }}
          </div>
        </article>

        <article class="panel">
          <div class="panel-title">回答结果</div>
          <div v-if="loading.chat || chatResult" class="result-stack">
            <div v-if="loading.chat" class="subsection process-section loading-process">
              <h3>思考过程</h3>
              <div class="summary-box">
                <div class="summary-title">当前正在执行</div>
                <div class="summary-item">
                  <span>当前状态</span>
                  <strong>{{ chatStatusText || "正在处理..." }}</strong>
                </div>
                <div
                  v-for="step in liveThinkingStepDetails"
                  :key="`live-step-${step.index}`"
                  class="live-step-card"
                >
                  <div class="live-step-head">
                    <span>{{ step.label }}</span>
                    <span class="badge" :class="step.status === 'running' ? 'warning' : 'success'">
                      {{ step.statusLabel }}
                    </span>
                  </div>
                  <div class="live-step-result">{{ step.message }}</div>
                </div>
                <div v-if="liveProcessCards.length" class="subsection live-result-section">
                  <h3>步骤结果</h3>
                  <div class="list-stack">
                    <div v-for="card in liveProcessCards" :key="card.phase" class="detail-card compact">
                      <div class="live-step-head">
                        <div class="task-name">{{ card.title }}</div>
                        <span class="badge" :class="card.statusClass">{{ card.statusLabel }}</span>
                      </div>
                      <div class="detail-info-grid single">
                        <div v-for="item in card.items" :key="`${card.phase}-${item.label}`" class="process-result-item">
                          <strong>{{ item.label }}：</strong>{{ item.value }}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div v-if="chatResult" class="meta-grid">
              <div><strong>问答 ID：</strong>{{ chatResult.qa_log_id }}</div>
              <div><strong>问题类型：</strong>{{ chatResult.question_type }}</div>
            </div>
            <div v-if="chatResult" class="subsection process-section">
              <h3>思考过程</h3>
              <div class="list-stack" v-if="chatForm.mode === 'standard'">
                <div v-for="card in standardProcessCards" :key="card.title" class="detail-card compact">
                  <div class="live-step-head">
                    <div class="task-name">{{ card.title }}</div>
                    <span class="badge success">{{ card.statusLabel }}</span>
                  </div>
                  <div class="detail-info-grid single">
                    <div v-for="item in card.items" :key="`${card.title}-${item.label}`" class="process-result-item">
                      <strong>{{ item.label }}：</strong>{{ item.value }}
                    </div>
                  </div>
                </div>
              </div>
              <div v-else class="detail-summary-grid expanded">
                <div class="summary-box">
                  <div class="summary-title">标准链路过程</div>
                  <div v-for="item in chatProcessSummary" :key="item.label" class="summary-item">
                    <span>{{ item.label }}</span>
                    <strong>{{ item.value }}</strong>
                  </div>
                </div>
                <div v-if="chatDecisionSummary.length" class="summary-box">
                  <div class="summary-title">Decision-ReAct 决策过程</div>
                  <div v-for="item in chatDecisionSummary" :key="`decision-${item.label}`" class="summary-item">
                    <span>{{ item.label }}</span>
                    <strong>{{ item.value }}</strong>
                  </div>
                </div>
              </div>
            </div>
            <div v-if="chatResult && chatTrace.length" class="subsection process-section">
              <h3>执行轨迹</h3>
              <div class="list-stack">
                <div v-for="(item, index) in chatTrace" :key="`${item.step}-${index}`" class="detail-card compact">
                  <div class="detail-card-head">
                    <div class="task-name">{{ index + 1 }}. {{ formatTraceStep(item.step) }}</div>
                    <span class="badge muted">{{ item.duration_ms ?? 0 }} ms</span>
                  </div>
                  <div class="detail-info-grid single">
                    <div><strong>步骤标识：</strong>{{ item.step }}</div>
                    <div><strong>步骤摘要：</strong>{{ item.summary || "无" }}</div>
                  </div>
                </div>
              </div>
            </div>
            <article v-if="chatResult" class="markdown-body answer-box" v-html="renderedAnswerHtml"></article>
            <div v-if="chatResult" class="meta-grid">
              <div><strong>检索标题：</strong>{{ (chatResult.retrieved_titles || []).join("、") || "无" }}</div>
              <div><strong>改写查询：</strong>{{ (chatResult.retrieval_queries || []).join(" | ") || "无" }}</div>
            </div>

            <div v-if="chatResult" class="subsection">
              <h3>提交反馈</h3>
              <div class="action-row">
                <label class="inline-field"><input type="radio" value="up" v-model="feedbackForm.rating" /> 👍 满意</label>
                <label class="inline-field"><input type="radio" value="down" v-model="feedbackForm.rating" /> 👎 不满意</label>
              </div>
              <textarea v-model="feedbackForm.comment" rows="3" placeholder="可选，补充说明哪里不对或哪里做得好"></textarea>
              <button class="ghost-button" @click="submitFeedback">提交反馈</button>
            </div>
          </div>
          <div v-else class="empty-state">回答结果会显示在这里。</div>
        </article>

        <article class="panel span-full">
          <div class="panel-title">会话历史</div>
          <div class="chat-log-list">
            <div v-for="log in sessionLogs" :key="log.id" class="chat-log-card">
              <div class="chat-log-question">{{ log.question }}</div>
              <article class="markdown-body chat-log-answer" v-html="renderedLogAnswer(log)"></article>
            </div>
          </div>
        </article>
      </section>

      <section v-else-if="activeView === 'documents'" class="view-grid">
        <article class="panel span-full">
          <div class="panel-title">知识文档列表</div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>标题</th>
                  <th>分类</th>
                  <th>版本</th>
                  <th>状态</th>
                  <th>分块数</th>
                  <th>上传时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="document in documents" :key="document.id">
                  <td>{{ document.title }}</td>
                  <td>{{ document.category }}</td>
                  <td>{{ document.version }}</td>
                  <td><span class="badge" :class="taskTone(document.status)">{{ document.status }}</span></td>
                  <td>{{ document.chunk_count }}</td>
                  <td>{{ document.uploaded_at }}</td>
                  <td><button class="danger-button" @click="removeDocument(document.id)">删除</button></td>
                </tr>
              </tbody>
            </table>
          </div>
        </article>
      </section>

      <section v-else-if="activeView === 'benchmark'" class="view-grid two-up">
        <article class="panel">
          <div class="panel-title">RAGBench 全局知识库评测</div>
          <div class="form-grid">
            <label class="field">
              <span>subset</span>
              <select v-model="benchmarkForm.subset">
                <option v-for="subset in benchmarkSubsets" :key="subset.subset" :value="subset.subset">
                  {{ subset.subset }} ({{ subset.question_count }} 题)
                </option>
              </select>
            </label>
            <label class="field">
              <span>split</span>
              <input v-model="benchmarkForm.split" />
            </label>
            <label class="field">
              <span>检索策略</span>
              <select v-model="benchmarkForm.retrieverStrategy">
                <option v-for="item in RETRIEVER_OPTIONS" :key="item.value" :value="item.value">{{ item.label }}</option>
              </select>
            </label>
            <label class="field">
              <span>Top K</span>
              <input v-model.number="benchmarkForm.topK" type="number" min="1" />
            </label>
            <label class="field">
              <span>题目数</span>
              <input v-model.number="benchmarkForm.questionLimit" type="number" min="1" />
            </label>
            <label class="field">
              <span>Chunk Size</span>
              <input v-model.number="benchmarkForm.chunkSize" type="number" min="100" />
            </label>
            <label class="field">
              <span>Chunk Overlap</span>
              <input v-model.number="benchmarkForm.chunkOverlap" type="number" min="0" />
            </label>
            <label class="field span-2">
              <span>Max Split Char Number</span>
              <input v-model.number="benchmarkForm.maxSplitCharNumber" type="number" min="100" />
            </label>
            <label class="inline-field"><input type="checkbox" v-model="benchmarkForm.enableQueryRewrite" /> 启用重写</label>
            <label class="inline-field"><input type="checkbox" v-model="benchmarkForm.enableRerank" /> 启用重排</label>
            <label class="inline-field"><input type="checkbox" v-model="benchmarkForm.enableFaithfulness" /> 测评 Faithfulness</label>
            <label class="inline-field"><input type="checkbox" v-model="benchmarkForm.rebuildCorpus" /> 重建语料</label>
            <label class="inline-field"><input type="checkbox" v-model="benchmarkForm.rebuildIndex" /> 重建索引</label>
          </div>
          <div class="action-row">
            <button class="ghost-button" @click="launchBenchmarkCorpus">构建语料</button>
            <button class="ghost-button" @click="launchBenchmarkIndex">构建索引</button>
            <button class="primary-button" @click="launchBenchmarkRun">启动评测</button>
          </div>
        </article>

        <article class="panel">
          <div class="panel-title">本地题库评测</div>
          <div class="form-grid">
            <label class="field span-2">
              <span>知识库名称</span>
              <input v-model="localEvalBuildForm.knowledgeBaseName" />
            </label>
            <label class="field">
              <span>切片大小</span>
              <input v-model.number="localEvalBuildForm.chunkSize" type="number" min="100" />
            </label>
            <label class="field">
              <span>切片重叠</span>
              <input v-model.number="localEvalBuildForm.chunkOverlap" type="number" min="0" />
            </label>
            <label class="field span-2">
              <span>最大免切分长度</span>
              <input v-model.number="localEvalBuildForm.maxSplitCharNumber" type="number" min="100" />
            </label>
            <label class="field span-2">
              <span>现有知识库</span>
              <select @change="applyKnowledgeBase(localEvalKnowledgeBases.find((item) => item.knowledge_base_id === $event.target.value))">
                <option value="">请选择</option>
                <option v-for="item in localEvalKnowledgeBases" :key="item.knowledge_base_id" :value="item.knowledge_base_id">
                  {{ item.knowledge_base_name }}
                </option>
              </select>
            </label>
            <label class="field span-2">
              <span>题库数据集</span>
              <select @change="applyDataset(localEvalDatasets.find((item) => item.dataset_key === $event.target.value))">
                <option value="">请选择</option>
                <option v-for="item in localEvalDatasets" :key="item.dataset_key" :value="item.dataset_key">
                  {{ item.dataset_label }}
                </option>
              </select>
            </label>
            <label class="field">
              <span>Top K</span>
              <input v-model.number="localEvalRunForm.topK" type="number" min="1" />
            </label>
            <label class="field">
              <span>题目数</span>
              <input v-model.number="localEvalRunForm.questionLimit" type="number" min="1" />
            </label>
            <label class="field">
              <span>检索策略</span>
              <select v-model="localEvalRunForm.retrieverStrategy">
                <option v-for="item in RETRIEVER_OPTIONS" :key="item.value" :value="item.value">{{ item.label }}</option>
              </select>
            </label>
            <label class="field">
              <span>切片大小</span>
              <input v-model.number="localEvalRunForm.chunkSize" type="number" min="100" />
            </label>
            <label class="field">
              <span>切片重叠</span>
              <input v-model.number="localEvalRunForm.chunkOverlap" type="number" min="0" />
            </label>
            <label class="field">
              <span>最大免切分长度</span>
              <input v-model.number="localEvalRunForm.maxSplitCharNumber" type="number" min="100" />
            </label>
            <label class="inline-field"><input type="checkbox" v-model="localEvalRunForm.enableQueryRewrite" /> 启用重写</label>
            <label class="inline-field"><input type="checkbox" v-model="localEvalRunForm.enableRerank" /> 启用重排</label>
            <label class="inline-field"><input type="checkbox" v-model="localEvalRunForm.enableFaithfulness" /> 测评 Faithfulness</label>
          </div>
          <div class="action-row">
            <button class="ghost-button" @click="launchLocalEvalBuild">构建知识库</button>
            <button class="primary-button" @click="launchLocalEvalRun">启动本地评测</button>
          </div>
        </article>

        <article class="panel span-full">
          <div class="panel-title">最近评测记录</div>
          <div class="list-stack">
            <div v-for="run in benchmarkRuns" :key="run.run_id" class="task-row" @click="openBenchmarkRunDetail(run.run_id)">
              <div>
                <div class="task-name">{{ run.run_id }}</div>
                <div class="task-meta">
                  {{ run.subset }} · {{ run.question_count }} 题 · top_k={{ run.top_k }} ·
                  {{ run.retriever_strategy || run.mode }} ·
                  重写={{ run.enable_query_rewrite === false ? "关" : "开" }} ·
                  Faithfulness={{ run.enable_faithfulness === false ? "关" : "开" }}
                </div>
              </div>
              <span class="badge success">{{ run.retriever_strategy || run.mode }}</span>
            </div>
          </div>
        </article>

        <article class="panel span-full">
          <div class="panel-title">评测详情</div>
          <div v-if="benchmarkRunDetail" class="result-stack">
            <div class="stat-grid metric-grid">
              <div v-for="item in benchmarkMetricCards" :key="item.label" class="stat-card compact">
                <strong>{{ item.value }}</strong>
                <span>{{ item.label }}</span>
              </div>
            </div>
            <div class="meta-grid">
              <div><strong>Run ID：</strong>{{ benchmarkRunDetail.summary.run_id }}</div>
              <div><strong>Subset：</strong>{{ benchmarkRunDetail.summary.subset }}</div>
              <div><strong>Split：</strong>{{ benchmarkRunDetail.summary.split }}</div>
              <div><strong>知识库：</strong>{{ benchmarkRunDetail.summary.knowledge_base_name || "无" }}</div>
              <div><strong>检索策略：</strong>{{ benchmarkRunDetail.summary.retriever_strategy }}</div>
              <div><strong>Top K：</strong>{{ benchmarkRunDetail.summary.top_k }}</div>
              <div><strong>重写：</strong>{{ benchmarkRunDetail.summary.enable_query_rewrite === false ? "关闭" : "开启" }}</div>
              <div><strong>重排：</strong>{{ benchmarkRunDetail.summary.enable_rerank === false ? "关闭" : "开启" }}</div>
              <div><strong>Faithfulness：</strong>{{ benchmarkRunDetail.summary.enable_faithfulness === false ? "关闭" : "开启" }}</div>
            </div>
            <div class="subsection">
              <h3>完整指标面板</h3>
              <div class="detail-summary-grid expanded">
                <div v-for="section in benchmarkMetricSections" :key="section.title" class="summary-box">
                  <div class="summary-title">{{ section.title }}</div>
                  <div v-for="item in section.items" :key="`${section.title}-${item.label}`" class="summary-item">
                    <span>{{ item.label }}</span>
                    <strong>{{ item.value }}</strong>
                  </div>
                </div>
              </div>
            </div>
            <div class="subsection">
              <h3>题目明细总览（前 20 条）</h3>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>问题</th>
                      <th>标准文档</th>
                      <th>命中</th>
                      <th>首命中排名</th>
                      <th>检索标题</th>
                      <th>预重排标题</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="detail in benchmarkDetails.slice(0, 20)" :key="`${benchmarkRunDetail.summary.run_id}-${detail.question_id}`">
                      <td>{{ detail.question_id }}</td>
                      <td class="question-cell">{{ detail.question }}</td>
                      <td>{{ joinList(detail.expected_titles) }}</td>
                      <td><span class="badge" :class="hitTone(detail)">{{ hitLabel(detail) }}</span></td>
                      <td>{{ detail.first_hit_rank ?? "-" }}</td>
                      <td>{{ joinList(detail.retrieved_titles) }}</td>
                      <td>{{ joinList(detail.pre_rerank_titles) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div class="subsection">
              <h3>题目详情（前 10 条）</h3>
              <div class="list-stack">
                <div
                  v-for="detail in benchmarkDetails.slice(0, 10)"
                  :key="`detail-${benchmarkRunDetail.summary.run_id}-${detail.question_id}`"
                  class="detail-card"
                >
                  <div class="detail-card-head">
                    <div class="task-name">Q{{ detail.question_id }}：{{ detail.question }}</div>
                    <span class="badge" :class="hitTone(detail)">{{ hitLabel(detail) }}</span>
                  </div>
                  <div class="detail-info-grid">
                    <div><strong>标准文档：</strong>{{ joinList(detail.expected_titles) }}</div>
                    <div><strong>预重排标题：</strong>{{ joinList(detail.pre_rerank_titles) }}</div>
                    <div><strong>最终检索标题：</strong>{{ joinList(detail.retrieved_titles) }}</div>
                    <div><strong>首命中排名：</strong>{{ detail.first_hit_rank ?? "-" }}</div>
                  </div>
                  <div class="detail-answer-grid">
                    <div class="answer-panel">
                      <div class="summary-title">标准答案</div>
                      <div class="answer-text">{{ detail.gold_answer || "无" }}</div>
                    </div>
                    <div class="answer-panel">
                      <div class="summary-title">最终回答</div>
                      <div class="answer-text">{{ detail.predicted_answer || "无" }}</div>
                    </div>
                  </div>
                  <div class="context-panel">
                    <div class="summary-title">检索命中文本</div>
                    <div v-if="detail.retrieved_contexts?.length" class="context-list">
                      <div
                        v-for="(context, index) in detail.retrieved_contexts.slice(0, 3)"
                        :key="`ctx-${benchmarkRunDetail.summary.run_id}-${detail.question_id}-${index}`"
                        class="context-item"
                      >
                        <div class="context-index">片段 {{ index + 1 }}</div>
                        <div class="context-text">{{ context }}</div>
                      </div>
                    </div>
                    <div v-else class="empty-state compact">没有返回检索片段。</div>
                  </div>
                  <details class="raw-detail">
                    <summary>查看原始 JSON</summary>
                    <pre class="json-box">{{ JSON.stringify(detail, null, 2) }}</pre>
                  </details>
                </div>
              </div>
            </div>
            <div class="subsection">
              <h3>题目原始详情（前 3 条）</h3>
              <div class="list-stack">
                <div v-for="detail in benchmarkDetails.slice(0, 3)" :key="`raw-${benchmarkRunDetail.summary.run_id}-${detail.question_id}`" class="chat-log-card">
                  <div class="task-name">Q{{ detail.question_id }}：{{ detail.question }}</div>
                  <div class="task-meta">
                    标准文档：{{ joinList(detail.expected_titles) }} | 最终检索：{{ joinList(detail.retrieved_titles) }}
                  </div>
                  <pre class="json-box">{{ JSON.stringify(detail, null, 2) }}</pre>
                </div>
              </div>
            </div>
          </div>
          <div v-else class="empty-state">点击上面的历史 run 查看完整详情。</div>
        </article>
      </section>

      <section v-else-if="activeView === 'tasks'" class="view-grid two-up">
        <article class="panel">
          <div class="panel-title">任务列表</div>
          <div class="list-stack">
            <div v-for="task in tasks" :key="task.id" class="task-row" @click="currentTaskId = task.id">
              <div>
                <div class="task-name">{{ task.task_type }}</div>
                <div class="task-meta">{{ task.created_at }}</div>
              </div>
              <span class="badge" :class="taskTone(task.status)">{{ task.status }}</span>
            </div>
          </div>
        </article>

        <article class="panel">
          <div class="panel-title">任务详情</div>
          <div v-if="selectedTask" class="task-detail">
            <div><strong>ID：</strong>{{ selectedTask.id }}</div>
            <div><strong>状态：</strong>{{ selectedTask.status }}</div>
            <div><strong>阶段：</strong>{{ selectedTask.progress_stage }}</div>
            <div><strong>消息：</strong>{{ selectedTask.progress_message }}</div>
            <pre class="json-box">{{ JSON.stringify(selectedTask, null, 2) }}</pre>
          </div>
          <div v-else class="empty-state">点击左侧任务查看详情。</div>
        </article>
      </section>
    </main>
  </div>
</template>
