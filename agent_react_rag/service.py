"""基于 create_agent 的真正工具调用式 ReAct-RAG 服务。"""

from __future__ import annotations

import config_data as config
from agent_react_rag.compat import get_create_agent
from agent_react_rag.schemas import AgentReactResult
from agent_react_rag.tools import AgentReactToolbox
from services.chat_service import OfficeMateChatService
from services.rag.answering import strip_think_blocks
from utils.log_tool import get_logger


logger = get_logger("agent_react_rag_service")


class AgentReactRagService:
    """让 create_agent 自己决定是否调用 rewrite / plan / retrieve / final_answer。"""

    def __init__(self):
        self.runtime = OfficeMateChatService()
        self.create_agent = get_create_agent()

    def _build_system_prompt(self, selected_category: str) -> str:
        return (
            "你是 OfficeMate 的 ReAct 风格 RAG Agent。"
            "你的任务不是直接凭常识回答，而是自己判断是否需要调用工具。"
            "你可以使用以下工具："
            "1. rewrite_tool：当问题口语化、模糊或省略较多时，先改写问题。"
            "2. plan_tool：当问题包含多个并列子问题、多个制度域或多个条件时，先拆解任务。"
            "3. retrieve_and_rerank_tool：基于当前问题或已拆解任务执行检索和重排；如果你已经判断问题明显属于某个制度域，请传入 target_category 缩窄检索范围。"
            "4. generate_final_answer_tool：基于检索后的证据生成唯一最终答案。"
            "必须遵守以下规则："
            "A. 你可以决定是否调用 rewrite_tool。"
            "B. 你可以决定是否调用 plan_tool。"
            "C. 在给用户最终回答之前，必须调用 retrieve_and_rerank_tool。"
            "D. 在 retrieve_and_rerank_tool 之后，必须调用 generate_final_answer_tool。"
            "E. generate_final_answer_tool 的返回结果就是最终答案，不要再改写、复述、总结或追加任何内容。"
            "F. 最终不要输出 <think>、解释过程、工具调用说明。"
            "G. 不要自己编造制度内容。"
            "H. 不要重复调用同一个工具超过 1 次，除非上一步明确失败。"
            f"I. 当前页面选择的分类是：{selected_category}。如果不是“全部”，请优先遵守该分类限制。"
        )

    def answer_question(self, question: str, session_id: str, category: str = "全部") -> AgentReactResult:
        question_type_key = self.runtime.infer_question_type(question)
        question_type = config.QUESTION_TYPE_LABELS[question_type_key]
        history = self.runtime.build_history(session_id)
        # 预热模型与检索协调器，避免在 LangGraph 工具线程里首次初始化
        # Chroma / provider client 时触发线程相关异常。
        self.runtime.get_chat_model()
        self.runtime.get_retrieval_coordinator()

        toolbox = AgentReactToolbox(
            runtime=self.runtime,
            question=question,
            selected_category=category,
            question_type_key=question_type_key,
            question_type_label=question_type,
            history=history,
        )
        agent = self.create_agent(
            model=self.runtime.get_chat_model(),
            system_prompt=self._build_system_prompt(category),
            tools=toolbox.get_tools(),
        )

        input_dict = {
            "messages": history + [
                {"role": "user", "content": question},
            ]
        }

        for chunk in agent.stream(input_dict, stream_mode="values"):
            if toolbox.state.get("final_answer"):
                # 一旦最终答案工具已经产出结果，就不再等待 Agent 后续再组织一遍文字，
                # 直接把工具结果作为最终答案，减少额外的模型收尾耗时。
                break

        answer = strip_think_blocks(toolbox.state.get("final_answer", "").strip())
        if not answer:
            answer = config.NO_EVIDENCE_MESSAGE + "\n\n### 引用文档\n无"

        references = toolbox.state.get("references", [])
        qa_log = self.runtime.storage.add_qa_log(
            {
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "category": category,
                "question_type": question_type,
                "source_docs": references,
                "mode": "agent_react_rag",
                "trace": [item.__dict__ for item in toolbox.trace],
            }
        )
        logger.info(
            "agent_react_rag answer_completed | session_id=%s | qa_log_id=%s | trace_steps=%s",
            session_id,
            qa_log["id"],
            [item.step for item in toolbox.trace],
        )
        return AgentReactResult(
            answer=answer,
            question_type=question_type,
            qa_log_id=qa_log["id"],
            source_docs=references,
            trace=[item.__dict__ for item in toolbox.trace],
        )
