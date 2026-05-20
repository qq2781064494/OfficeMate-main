"""问答服务层。

升级后的链路包含：
1. 识别问题类型
2. 查询改写，扩展企业术语和同义词
3. 任务规划，把复杂问题拆成多个子任务
4. 混合检索，结合向量召回与 BM25 关键词召回
5. 轻量重排，按业务规则提升更相关的证据片段
6. 分子任务生成答案，再统一汇总为最终回答
"""

from __future__ import annotations

import json
from typing import Callable, Iterable, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import config_data as config
from services.model_provider import ModelProviderFactory
from services.rag.answering import (
    AnswerSynthesizer,
    ThinkBlockStreamFilter,
    build_final_answer,
    finalize_task_answers,
    format_reference_markdown,
    generate_task_answers_parallel,
    strip_think_blocks,
)
from services.rag.contracts import (
    ChatRequest,
    ChatResponse,
    ChatResultHolder,
    PipelineState,
    StreamingChatSession,
)
from services.rag.planning import PlannerFactory
from services.rag.query import QueryRewriteResult, QueryRewriter, infer_question_type, resolve_question_type_label
from services.rag.retrieval import RetrievalCoordinator, RetrieverFactory
from services.rag.selection import SimpleReranker, select_task_evidence
from services.storage_service import JsonStorageService
from utils.log_tool import get_logger


logger = get_logger("chat_service")

_strip_think_blocks = strip_think_blocks
_ThinkBlockStreamFilter = ThinkBlockStreamFilter


class OfficeMateChatService:
    """企业知识问答的总编排器。

    这个类现在扮演的是 Facade / Orchestrator 的角色：
    - 页面层只需要调用 answer_question
    - 具体的 query rewrite、task planning、retrieval、rerank、synthesize
      都被收敛在这里统一编排

    对面试来说，这个类非常适合作为“主流程讲解入口”。
    """

    def __init__(self, retriever=None, retrieval_coordinator=None):
        self.storage = JsonStorageService()
        self.chat_model = None
        self.rewrite_model = None
        self.task_model = None
        self.hybrid_retriever = retriever
        self.retrieval_coordinator = retrieval_coordinator
        self.query_rewriter = QueryRewriter(chat_model_factory=self._get_rewrite_model)
        self.task_planner = PlannerFactory.create("hybrid", chat_model_factory=self._get_task_model)
        self.reranker = SimpleReranker()
        # 这里的 prompt 只负责“单个子任务”的回答。
        # 多个子任务之间的合并，会交给 AnswerSynthesizer 单独处理。
        # 子任务回答时，模型必须围绕“当前任务”作答，避免多意图问题一次说乱。
        self.task_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是 OfficeMate，负责回答企业内部制度与流程问题。"
                    "你只能依据给定的参考材料回答，不能编造制度、审批规则或联系方式。"
                    "如果材料不足，请明确写“未找到明确依据”。"
                    "输出必须严格使用以下 Markdown 标题："
                    "### 最终回答\n### 操作步骤/材料清单\n### 风险提示\n"
                    "如果某一部分不适用，请写“无”。",
                ),
                MessagesPlaceholder("history"),
                (
                    "human",
                    "原始问题：{question}\n"
                    "当前子任务：{task_description}\n"
                    "当前任务类型：{question_type}\n"
                    "当前分类过滤：{category}\n\n"
                    "参考材料如下：\n{context}\n\n"
                    "请只回答当前子任务，不要编造材料中不存在的规则。",
                ),
            ]
        )

    def answer_question(
        self,
        question,
        session_id,
        category="全部",
        status_callback: Callable[[str], None] | None = None,
        *,
        use_history: bool = True,
        persist_log: bool = True,
        include_references: bool = True,
        enable_query_rewrite: bool = True,
        enable_rerank: bool = True,
        reference_limit: int | None = None,
    ):
        """同步问答入口。"""
        request = ChatRequest(
            question=question,
            session_id=session_id,
            category=category,
            use_history=use_history,
            persist_log=persist_log,
            include_references=include_references,
            enable_query_rewrite=enable_query_rewrite,
            enable_rerank=enable_rerank,
            reference_limit=reference_limit,
            status_callback=status_callback,
        )
        return self.run_chat(request).to_legacy_dict()

    def stream_answer_question(
        self,
        question,
        session_id,
        category="全部",
        status_callback: Callable[[str], None] | None = None,
    ) -> tuple[Iterable[str], dict]:
        """流式问答入口，主要给 Streamlit 聊天页使用。"""
        request = ChatRequest(
            question=question,
            session_id=session_id,
            category=category,
            use_history=True,
            persist_log=True,
            include_references=True,
            enable_query_rewrite=True,
            enable_rerank=True,
            reference_limit=config.max_reference_documents,
            status_callback=status_callback,
        )
        session = self.stream_chat(request)
        return session.stream, session.result_holder.to_legacy_dict()

    def run_chat(self, request: ChatRequest) -> ChatResponse:
        """执行完整问答流程并一次性返回结果。"""
        pipeline_state = self._prepare_pipeline_state(request)
        reference_limit = self._resolve_reference_limit(request.reference_limit)
        evidence = self._select_task_evidence(
            pipeline_state=pipeline_state,
            selected_category=request.category,
            reference_limit=reference_limit,
            enable_rerank=request.enable_rerank,
            log_prefix="answer_question",
            session_id=request.session_id,
        )
        task_plans = evidence["task_plans"]
        references = evidence["references"]
        retrieved_contexts = evidence["retrieved_contexts"]
        if not references:
            return self._build_empty_response(request, pipeline_state)

        task_answers = generate_task_answers_parallel(
            question=request.question,
            question_type=pipeline_state.question_type,
            history=pipeline_state.history,
            task_plans=task_plans,
            session_id=request.session_id,
            prompt_template=self.task_prompt_template,
            chat_model_factory=self.get_chat_model,
            question_type_labels=config.QUESTION_TYPE_LABELS,
            logger=logger,
            status_callback=request.status_callback,
        )
        answer_body = self._resolve_answer_body(
            question=request.question,
            status_callback=request.status_callback,
            task_answers=task_answers,
            task_reference_groups=evidence["task_reference_groups"],
        )
        qa_log_id = self._persist_qa_log(
            request=request,
            answer=build_final_answer(answer_body, references, request.include_references),
            question_type=pipeline_state.question_type,
            references=references,
        )
        response = ChatResponse(
            answer=build_final_answer(answer_body, references, request.include_references),
            question_type=pipeline_state.question_type,
            qa_log_id=qa_log_id,
            source_docs=references,
            retrieved_contexts=retrieved_contexts,
            normalized_query=pipeline_state.rewrite_result.normalized_query,
            retrieval_queries=pipeline_state.rewrite_result.retrieval_queries,
            matched_terms=pipeline_state.rewrite_result.matched_terms,
            pre_rerank_titles=evidence["pre_rerank_titles"],
            retrieved_titles=evidence["retrieved_titles"],
            planned_tasks=self._serialize_planned_tasks(pipeline_state.planned_tasks),
        )
        logger.info(
            "answer_question completed | session_id=%s | question_type=%s | reference_count=%s | answer_length=%s",
            request.session_id,
            response.question_type,
            len(references),
            len(response.answer),
        )
        return response

    def stream_chat(self, request: ChatRequest) -> StreamingChatSession:
        """执行完整问答流程，但把答案正文按流式方式产出。"""
        result_holder = ChatResultHolder()

        def generator():
            pipeline_state = self._prepare_pipeline_state(request)
            reference_limit = self._resolve_reference_limit(request.reference_limit)
            evidence = self._select_task_evidence(
                pipeline_state=pipeline_state,
                selected_category=request.category,
                reference_limit=reference_limit,
                enable_rerank=request.enable_rerank,
                log_prefix="stream_answer_question",
                session_id=request.session_id,
            )
            self._emit_event(
                request.event_callback,
                "evidence_selection",
                {
                    "title": "步骤 3：检索与重排",
                    "items": [
                        {"label": "预重排标题", "value": "、".join(evidence["pre_rerank_titles"]) or "无"},
                        {"label": "最终检索标题", "value": "、".join(evidence["retrieved_titles"]) or "无"},
                        {"label": "命中文本片段数", "value": str(len(evidence["retrieved_contexts"]))},
                    ],
                },
            )
            task_plans = evidence["task_plans"]
            references = evidence["references"]
            retrieved_contexts = evidence["retrieved_contexts"]
            if not references:
                response = self._build_empty_response(request, pipeline_state)
                result_holder.set_response(response)
                yield response.answer
                return

            # 某些任务在证据选择后可能已经没有候选了，这里先过滤掉空任务。
            non_empty_task_plans = [item for item in task_plans if item.candidates]
            if not non_empty_task_plans:
                response = self._build_response(
                    request=request,
                    pipeline_state=pipeline_state,
                    pre_rerank_titles=evidence["pre_rerank_titles"],
                    retrieved_titles=evidence["retrieved_titles"],
                    references=references,
                    retrieved_contexts=retrieved_contexts,
                    answer_body=config.NO_EVIDENCE_MESSAGE,
                )
                result_holder.set_response(response)
                yield response.answer
                return

            task_answers = generate_task_answers_parallel(
                question=request.question,
                question_type=pipeline_state.question_type,
                history=pipeline_state.history,
                task_plans=non_empty_task_plans,
                session_id=request.session_id,
                prompt_template=self.task_prompt_template,
                chat_model_factory=self.get_chat_model,
                question_type_labels=config.QUESTION_TYPE_LABELS,
                logger=logger,
                status_callback=request.status_callback,
            )
            self._emit_event(
                request.event_callback,
                "task_answers",
                {
                    "title": "步骤 4：子任务答案生成",
                    "items": [
                        {
                            "label": item.task_description,
                            "value": item.answer[:180] + ("..." if len(item.answer) > 180 else ""),
                        }
                        for item in task_answers
                    ] or [{"label": "生成结果", "value": "未生成子任务答案。"}],
                },
            )
            streamed_parts: List[str] = []
            answer_body_parts: List[str] = []

            # 单任务和多任务的输出路径不同：
            # - 单任务：直接输出正文
            # - 多任务：先走 synthesizer 汇总，再追加引用
            if len(task_answers) <= 1:
                answer_body, _ = finalize_task_answers(
                    question=request.question,
                    task_answers=task_answers,
                    use_synthesize=False,
                    chat_model_factory=self.get_chat_model,
                    logger=logger,
                    task_reference_groups=evidence["task_reference_groups"],
                )
                for chunk in self._yield_text_chunks(answer_body):
                    streamed_parts.append(chunk)
                    answer_body_parts.append(chunk)
                    yield chunk
            else:
                self._emit_status(request.status_callback, "正在整合多个子任务答案...")
                try:
                    for chunk in AnswerSynthesizer(self._get_chat_model()).stream_synthesize(request.question, task_answers):
                        streamed_parts.append(chunk)
                        answer_body_parts.append(chunk)
                        yield chunk
                except Exception as exc:
                    logger.exception(
                        "stream_answer_question synthesizer_failed_fallback_to_concat | session_id=%s | task_ids=%s | error=%s",
                        request.session_id,
                        [task.task_id for task in task_answers],
                        exc,
                    )
                    answer_body, _ = finalize_task_answers(
                        question=request.question,
                        task_answers=task_answers,
                        use_synthesize=False,
                        chat_model_factory=self.get_chat_model,
                        logger=logger,
                        task_reference_groups=evidence["task_reference_groups"],
                    )
                    streamed_parts.append(answer_body)
                    answer_body_parts.append(answer_body)
                    yield answer_body

            self._emit_status(request.status_callback, "正在整理引用并生成最终回答...")
            if request.include_references:
                references_block = f"\n\n### 引用文档\n{format_reference_markdown(references)}"
                streamed_parts.append(references_block)
                yield references_block

            response = self._build_response(
                request=request,
                pipeline_state=pipeline_state,
                pre_rerank_titles=evidence["pre_rerank_titles"],
                retrieved_titles=evidence["retrieved_titles"],
                references=references,
                retrieved_contexts=retrieved_contexts,
                answer_body="".join(answer_body_parts).strip(),
            )
            self._emit_event(
                request.event_callback,
                "final_answer",
                {
                    "title": "步骤 5：最终答案整理",
                    "items": [
                        {
                            "label": "回答摘要",
                            "value": response.answer[:220] + ("..." if len(response.answer) > 220 else ""),
                        }
                    ],
                },
            )
            logger.info(
                "stream_answer_question completed | session_id=%s | question_type=%s | reference_count=%s | answer_length=%s",
                request.session_id,
                response.question_type,
                len(references),
                len(response.answer),
            )
            result_holder.set_response(response)

        return StreamingChatSession(stream=generator(), result_holder=result_holder)

    def _prepare_pipeline_state(self, request: ChatRequest) -> PipelineState:
        """完成问答链路前半段准备工作。"""
        self._emit_status(request.status_callback, "正在识别问题类型并改写检索词...")
        question_type_key = infer_question_type(request.question)
        question_type = resolve_question_type_label(question_type_key)
        if request.enable_query_rewrite:
            rewrite_result = self.query_rewriter.rewrite(request.question)
        else:
            rewrite_result = QueryRewriteResult(
                original_query=request.question,
                normalized_query=request.question,
                retrieval_queries=[request.question],
                matched_terms={},
            )
        self._emit_event(
            request.event_callback,
            "query_understanding",
            {
                "title": "步骤 1：问题理解与改写",
                "items": [
                    {"label": "问题类型", "value": question_type},
                    {"label": "规范化问题", "value": rewrite_result.normalized_query},
                    {"label": "检索查询", "value": " | ".join(rewrite_result.retrieval_queries)},
                    {"label": "命中术语", "value": json.dumps(rewrite_result.matched_terms, ensure_ascii=False)},
                ],
            },
        )
        self._emit_status(request.status_callback, "正在拆解子任务并规划检索范围...")
        planned_tasks = self.task_planner.plan(rewrite_result.normalized_query, question_type_key, request.category)
        self._emit_event(
            request.event_callback,
            "task_planning",
            {
                "title": "步骤 2：任务拆解",
                "items": [
                    {
                        "label": f"子任务 {index + 1}",
                        "value": f"{task.description} | 分类：{task.category} | 类型：{task.intent}"
                    }
                    for index, task in enumerate(planned_tasks)
                ] or [{"label": "拆解结果", "value": "未拆分子任务。"}],
            },
        )
        # 多轮对话时，把同一 session 的历史问答转成 LangChain message。
        history = self._build_history(request.session_id) if request.use_history else []
        logger.info(
            "chat_pipeline prepared | session_id=%s | category=%s | question=%s | question_type=%s | rewrite=%s | matched_terms=%s | planned_tasks=%s | history_rounds=%s",
            request.session_id,
            request.category,
            request.question,
            question_type,
            rewrite_result.retrieval_queries,
            rewrite_result.matched_terms,
            planned_tasks,
            len(history) // 2,
        )
        return PipelineState(
            question_type_key=question_type_key,
            question_type=question_type,
            rewrite_result=rewrite_result,
            planned_tasks=planned_tasks,
            history=history,
        )

    def _select_task_evidence(
        self,
        pipeline_state: PipelineState,
        selected_category: str,
        reference_limit: int,
        enable_rerank: bool,
        log_prefix: str,
        session_id: str,
    ) -> dict:
        """为所有子任务选择最终证据。"""
        evidence = select_task_evidence(
            rewrite_result=pipeline_state.rewrite_result,
            planned_tasks=pipeline_state.planned_tasks,
            selected_category=selected_category,
            top_k=reference_limit,
            query_rewriter=self.query_rewriter,
            retrieval_coordinator=self.get_retrieval_coordinator(),
            reranker=self.reranker,
            enable_rerank=enable_rerank,
        )
        for item in evidence["task_plans"]:
            logger.info(
                "%s task_processed | session_id=%s | task_id=%s | task_category=%s | reranked_candidates=%s",
                log_prefix,
                session_id,
                item.planned_task.task_id,
                item.task_category,
                len(item.candidates),
            )
        return evidence

    def _resolve_reference_limit(self, reference_limit: int | None) -> int:
        """把引用上限规整成合法整数。"""
        return max(1, int(reference_limit or config.max_reference_documents))

    def _resolve_answer_body(self, question: str, status_callback, task_answers: list, task_reference_groups: list[dict]) -> str:
        """根据子任务答案生成最终正文。"""
        if not task_answers:
            return config.NO_EVIDENCE_MESSAGE
        self._emit_status(status_callback, "正在整合多个子任务答案...")
        answer_body, metadata = finalize_task_answers(
            question=question,
            task_answers=task_answers,
            use_synthesize=True,
            chat_model_factory=self.get_chat_model,
            logger=logger,
            task_reference_groups=task_reference_groups,
        )
        logger.info(
            "answer_question finalize_completed | mode=%s | task_count=%s | answer_length=%s",
            metadata.get("mode"),
            metadata.get("task_count"),
            len(answer_body),
        )
        return answer_body

    def _build_empty_response(self, request: ChatRequest, pipeline_state: PipelineState) -> ChatResponse:
        """在没有找到证据时，构造统一兜底响应。"""
        answer = build_final_answer(config.NO_EVIDENCE_MESSAGE, [], request.include_references)
        qa_log_id = self._persist_qa_log(
            request=request,
            answer=answer,
            question_type=pipeline_state.question_type,
            references=[],
        )
        return ChatResponse(
            answer=answer,
            question_type=pipeline_state.question_type,
            qa_log_id=qa_log_id,
            source_docs=[],
            retrieved_contexts=[],
            normalized_query=pipeline_state.rewrite_result.normalized_query,
            retrieval_queries=pipeline_state.rewrite_result.retrieval_queries,
            matched_terms=pipeline_state.rewrite_result.matched_terms,
            pre_rerank_titles=[],
            retrieved_titles=[],
            planned_tasks=self._serialize_planned_tasks(pipeline_state.planned_tasks),
        )

    def _build_response(
        self,
        request: ChatRequest,
        pipeline_state: PipelineState,
        pre_rerank_titles: list[str],
        retrieved_titles: list[str],
        references: list[dict],
        retrieved_contexts: list[str],
        answer_body: str,
    ) -> ChatResponse:
        """把正文、引用和检索元数据组装成标准响应对象。"""
        full_answer = build_final_answer(answer_body, references, request.include_references)
        qa_log_id = self._persist_qa_log(
            request=request,
            answer=full_answer,
            question_type=pipeline_state.question_type,
            references=references,
        )
        return ChatResponse(
            answer=full_answer,
            question_type=pipeline_state.question_type,
            qa_log_id=qa_log_id,
            source_docs=references,
            retrieved_contexts=retrieved_contexts,
            normalized_query=pipeline_state.rewrite_result.normalized_query,
            retrieval_queries=pipeline_state.rewrite_result.retrieval_queries,
            matched_terms=pipeline_state.rewrite_result.matched_terms,
            pre_rerank_titles=pre_rerank_titles,
            retrieved_titles=retrieved_titles,
            planned_tasks=self._serialize_planned_tasks(pipeline_state.planned_tasks),
        )

    def _persist_qa_log(
        self,
        request: ChatRequest,
        answer: str,
        question_type: str,
        references: list[dict],
    ) -> int | None:
        """把问答结果写入本地 JSON 日志。"""
        if not request.persist_log:
            return None
        qa_log = self.storage.add_qa_log(
            {
                "session_id": request.session_id,
                "question": request.question,
                "answer": answer,
                "category": request.category,
                "question_type": question_type,
                "source_docs": references,
            }
        )
        return qa_log["id"]

    def _yield_text_chunks(self, text: str, chunk_size: int = 64):
        """把完整文本按小块切开，给页面做流式展示。"""
        for index in range(0, len(text), chunk_size):
            yield text[index:index + chunk_size]

    def _emit_status(self, status_callback: Callable[[str], None] | None, message: str) -> None:
        """向页面层发送当前处理阶段的简短说明。

        这里故意只发送“流程状态”，不发送模型内部推理内容。
        这样既能让用户知道系统正在做什么，也不会暴露冗长、不稳定的
        thinking 文本。
        """
        if status_callback is not None:
            status_callback(message)

    def _emit_event(self, event_callback: Callable[[str, dict], None] | None, phase: str, payload: dict) -> None:
        if event_callback is not None:
            event_callback(phase, payload)

    def _serialize_planned_tasks(self, planned_tasks: list[object]) -> list[dict]:
        results = []
        for item in planned_tasks:
            results.append(
                {
                    "task_id": getattr(item, "task_id", ""),
                    "description": getattr(item, "description", ""),
                    "category": getattr(item, "category", ""),
                    "intent": getattr(item, "intent", ""),
                    "hints": list(getattr(item, "hints", []) or []),
                }
            )
        return results

    def infer_question_type(self, question):
        """向外暴露问题类型判断，方便实验链路复用。"""
        return infer_question_type(question)

    def _build_history(self, session_id):
        """把当前会话的历史问答转成 LangChain message 列表。"""
        logs = self.storage.list_session_logs(session_id, limit=config.max_history_rounds)
        messages = []
        for log in logs:
            messages.append(HumanMessage(content=log["question"]))
            messages.append(AIMessage(content=log["answer"]))
        return messages

    def build_history(self, session_id):
        """公共包装方法，供其他模式读取同一套会话历史。"""
        return self._build_history(session_id)

    def format_reference_markdown(self, references):
        """公共包装方法，供外层模块复用引用格式化逻辑。"""
        return format_reference_markdown(references)

    def _get_hybrid_retriever(self):
        """延迟创建混合检索器。

        这里通过 RetrieverFactory 显式应用工厂模式，
        方便后续切换成仅向量检索、仅 BM25 或别的策略。
        """
        if self.hybrid_retriever is None:
            self.hybrid_retriever = RetrieverFactory.create("hybrid")
            logger.info("chat_service hybrid_retriever_initialized")
        return self.hybrid_retriever

    def get_hybrid_retriever(self):
        return self._get_hybrid_retriever()

    def _get_retrieval_coordinator(self):
        """延迟创建共享检索协调器。

        初学时可以重点理解“为什么不在 __init__ 里直接创建”：
        - RetrievalCoordinator 依赖 retriever
        - retriever 又会初始化向量库
        - 向量库初始化通常比较重
        - 所以只有真正问问题时才创建，会更省资源
        """
        if self.retrieval_coordinator is None:
            self.retrieval_coordinator = RetrievalCoordinator(
                retriever=self._get_hybrid_retriever(),
                query_rewriter=self.query_rewriter,
            )
            logger.info("chat_service retrieval_coordinator_initialized")
        return self.retrieval_coordinator

    def get_retrieval_coordinator(self):
        return self._get_retrieval_coordinator()

    def _get_chat_model(self):
        """延迟创建聊天模型对象。"""
        if self.chat_model is None:
            self.chat_model = ModelProviderFactory.create_chat_provider().build_chat_model(
                temperature=0.3,
            )
            logger.info(
                "chat_service chat_model_initialized | provider=%s | model=%s | base_url=%s",
                config.chat_provider,
                config.chat_model_name,
                config.chat_base_url,
            )
        return self.chat_model

    def get_chat_model(self):
        return self._get_chat_model()

    def _get_rewrite_model(self):
        """延迟创建查询改写模型对象。"""
        if self.rewrite_model is None:
            self.rewrite_model = ModelProviderFactory.create_rewrite_provider().build_chat_model(
                temperature=0.1,
            )
            logger.info(
                "chat_service rewrite_model_initialized | provider=%s | model=%s | base_url=%s",
                config.rewrite_provider,
                config.rewrite_model_name,
                config.rewrite_base_url,
            )
        return self.rewrite_model

    def get_rewrite_model(self):
        return self._get_rewrite_model()

    def _get_task_model(self):
        """延迟创建任务规划模型对象。"""
        if self.task_model is None:
            self.task_model = ModelProviderFactory.create_task_provider().build_chat_model(
                temperature=0.1,
                extra_body={"reasoning_split": True},
            )
            logger.info(
                "chat_service task_model_initialized | provider=%s | model=%s | base_url=%s | reasoning_split=%s",
                config.task_provider,
                config.task_model_name,
                config.task_base_url,
                True,
            )
        return self.task_model
