"""answering 阶段的共享 helper。

这一层关注“如何把证据组织成最终文本”：
- 单任务时，直接基于当前证据回答
- 多任务时，先分别回答，再做统一汇总
- 最后补上引用文档列表
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re

import config_data as config
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from services.rag.planning import PlannedTask


def strip_think_blocks(text: str) -> str:
    """清理模型可能返回的 `<think>...</think>` 思考块。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _strip_answer_section_headings(text: str) -> str:
    """去掉子答案里已有的三级标题，便于后续统一汇总。"""
    cleaned = re.sub(
        r"^\s*###\s*(最终回答|操作步骤/材料清单|风险提示)\s*$",
        "",
        text,
        flags=re.M,
    )
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


@dataclass
class TaskAnswer:
    """单个子任务的回答结果。"""

    task_id: str
    task_description: str
    category: str
    answer: str


class AnswerSynthesizer:
    """把多个子任务答案聚合成一份最终回复。"""

    def __init__(self, chat_model):
        self.chat_model = chat_model
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是企业知识助手的答案汇总器。"
                    "你的唯一任务是：围绕“原始问题”生成一份最终答复，而不是改写、转述或逐条复刻子任务答案。"
                    "原始问题的优先级高于所有子答案；每一段内容都必须直接服务于原始问题，不能偏离。"
                    "子任务答案只是候选素材，可能有重复、交叉、噪音、局部视角，甚至自带小标题；你必须先理解原始问题，再对素材去重、筛选、归并后统一作答。"
                    "禁止按子任务逐条回答，禁止保留“任务1/任务2”等痕迹，禁止把每个子答案分别生成为一套结构。"
                    "整份输出只能出现一次以下 3 个 Markdown 标题，且顺序必须严格一致："
                    "### 最终回答\n### 操作步骤/材料清单\n### 风险提示\n"
                    "注意：上面这 3 个标题是针对整份最终回复，不是针对每个子答案。除这 3 个标题外，不要再生成同级重复标题。"
                    "如果子答案之间重复，请去重；如果存在冲突，优先保守表述并明确说明“未找到明确依据”；如果材料不足，也必须明确标注“未找到明确依据”。"
                    "输出时先回答用户真正想问什么，再补充步骤和风险；不要被子答案原有格式带偏。"
                ),
                (
                    "human",
                    "请先牢记并紧扣这个原始问题，最终回复必须始终围绕它展开：\n"
                    "【原始问题】\n{question}\n\n"
                    "下面是多个子任务答案，它们只是素材，不是最终输出格式。"
                    "请你先内部整合，再输出一份统一答案；不要按素材顺序逐条作答，不要复用素材里的标题结构。\n\n"
                    "【子任务答案素材】\n{task_answers}",
                ),
            ]
        )

    def synthesize(self, question: str, task_answers: list[TaskAnswer]) -> str:
        """把多个子任务答案整合成一份最终答案。"""
        if len(task_answers) == 1:
            return strip_think_blocks(task_answers[0].answer)

        task_blocks = []
        for index, task_answer in enumerate(task_answers, start=1):
            task_blocks.append(
                f"[任务 {index}] {task_answer.task_description} | 分类：{task_answer.category}\n"
                f"{_strip_answer_section_headings(task_answer.answer)}"
            )

        chain = self.prompt | self.chat_model | StrOutputParser()
        result = chain.invoke(
            {
                "question": question,
                "task_answers": "\n\n".join(task_blocks),
            }
        )
        return strip_think_blocks(result)

    def stream_synthesize(self, question: str, task_answers: list[TaskAnswer], chunk_size: int = 64):
        """给页面层提供流式版本的答案汇总输出。"""
        result = self.synthesize(question, task_answers)
        for index in range(0, len(result), max(1, int(chunk_size))):
            yield result[index:index + max(1, int(chunk_size))]


class ThinkBlockStreamFilter:
    """按流式 chunk 过滤 `<think>...</think>` 思考块。"""

    def __init__(self):
        self.buffer = ""
        self.in_think_block = False

    def feed(self, chunk: str) -> list[str]:
        """增量处理流式文本，屏蔽 `<think>` 内容。"""
        self.buffer += chunk
        outputs: list[str] = []

        while self.buffer:
            if self.in_think_block:
                end_index = self.buffer.find("</think>")
                if end_index == -1:
                    if len(self.buffer) > 32:
                        self.buffer = self.buffer[-32:]
                    break
                self.buffer = self.buffer[end_index + len("</think>"):]
                self.in_think_block = False
                continue

            start_index = self.buffer.find("<think>")
            if start_index == -1:
                safe_length = max(0, len(self.buffer) - 32)
                if safe_length:
                    outputs.append(self.buffer[:safe_length])
                    self.buffer = self.buffer[safe_length:]
                break

            if start_index > 0:
                outputs.append(self.buffer[:start_index])
            self.buffer = self.buffer[start_index + len("<think>"):]
            self.in_think_block = True

        return [item for item in outputs if item]

    def flush(self) -> list[str]:
        """在流结束时把缓冲区里剩余的可见文本吐出来。"""
        if self.in_think_block:
            self.buffer = ""
            return []
        output = self.buffer.replace("<think>", "").replace("</think>", "")
        self.buffer = ""
        return [output] if output else []


def build_context(candidates: list[object], limit: int) -> str:
    """把候选证据整理成适合 prompt 消费的上下文文本。"""
    blocks = []
    for index, candidate in enumerate(candidates[:limit], start=1):
        blocks.append(
            f"[{index}] 标题：{candidate.title}\n"
            f"分类：{candidate.category} | "
            f"版本：{candidate.version} | "
            f"综合得分：{candidate.score:.4f}\n"
            f"内容：{candidate.content[:600]}"
        )
    return "\n\n".join(blocks)


def build_references(candidates: list[object], limit: int) -> list[dict]:
    """从候选证据里提炼“去重后的引用文档列表”。"""
    references = []
    seen_document_ids = set()
    for candidate in candidates:
        document_id = candidate.document_id
        if document_id in seen_document_ids:
            continue
        seen_document_ids.add(document_id)
        references.append(
            {
                "document_id": document_id,
                "title": candidate.title,
                "category": candidate.category,
                "version": candidate.version,
                "file_name": candidate.file_name,
                "score": round(float(candidate.score), 4),
            }
        )
        if len(references) >= limit:
            break
    return references


def format_reference_markdown(references: list[dict]) -> str:
    """把引用文档转成最终回答里展示的 Markdown 列表。"""
    lines = []
    for index, item in enumerate(references, start=1):
        lines.append(
            f"{index}. {item['title']} | 分类：{item['category']} | "
            f"版本：{item['version']} | 文件：{item['file_name']}"
        )
    return "\n".join(lines) if lines else "无"


def build_final_answer(answer_body: str, references: list[dict], include_references: bool) -> str:
    """把正文和引用文档拼成最终回答。"""
    if not include_references:
        return answer_body
    return f"{answer_body.strip()}\n\n### 引用文档\n{format_reference_markdown(references)}"


def answer_single_task(
    *,
    question: str,
    question_type: str,
    planned_task: PlannedTask,
    category: str,
    history,
    candidates: list[object],
    prompt_template,
    chat_model_factory,
    question_type_labels: dict,
    logger,
) -> TaskAnswer | None:
    """回答单个子任务。"""
    if not candidates:
        logger.warning("answer_single_task no_candidates | task_id=%s | category=%s", planned_task.task_id, category)
        return None

    # 先把候选证据压成结构化上下文，减少 prompt 中的噪声。
    context = build_context(candidates, limit=len(candidates))
    chain = prompt_template | chat_model_factory() | StrOutputParser()
    logger.info(
        "answer_single_task invoke_model | task_id=%s | category=%s | question_type=%s | context_blocks=%s",
        planned_task.task_id,
        category,
        planned_task.intent,
        len(candidates),
    )
    answer_body = chain.invoke(
        {
            "question": question,
            "task_description": planned_task.description,
            "question_type": question_type_labels.get(planned_task.intent, question_type),
            "category": category,
            "context": context,
            "history": history,
        }
    )
    answer_body = strip_think_blocks(answer_body)
    logger.info(
        "answer_single_task completed | task_id=%s | category=%s | answer_length=%s",
        planned_task.task_id,
        category,
        len(answer_body),
    )
    return TaskAnswer(
        task_id=planned_task.task_id,
        task_description=planned_task.description,
        category=category,
        answer=answer_body,
    )


def generate_task_answers_parallel(
    *,
    question: str,
    question_type: str,
    history,
    task_plans: list[object],
    session_id: str,
    prompt_template,
    chat_model_factory,
    question_type_labels: dict,
    logger,
    status_callback=None,
    parallel_workers: int | None = None,
    log_prefix: str = "task_answers",
) -> list[TaskAnswer]:
    """并发生成多个子任务答案。"""
    valid_task_plans = [item for item in task_plans if getattr(item, "candidates", None)]
    if not valid_task_plans:
        return []

    if len(valid_task_plans) == 1:
        item = valid_task_plans[0]
        if status_callback is not None:
            status_callback(f"正在生成子任务答案：{item.planned_task.description}")
        task_answer = answer_single_task(
            question=question,
            question_type=question_type,
            planned_task=item.planned_task,
            category=item.task_category,
            history=history,
            candidates=item.candidates,
            prompt_template=prompt_template,
            chat_model_factory=chat_model_factory,
            question_type_labels=question_type_labels,
            logger=logger,
        )
        return [task_answer] if task_answer else []

    max_workers = min(parallel_workers or config.parallel_subtask_workers, len(valid_task_plans))
    if status_callback is not None:
        status_callback(f"正在并行生成 {len(valid_task_plans)} 个子任务答案...")
    logger.info(
        "%s parallel_start | session_id=%s | task_count=%s | max_workers=%s | task_ids=%s",
        log_prefix,
        session_id,
        len(valid_task_plans),
        max_workers,
        [item.planned_task.task_id for item in valid_task_plans],
    )

    ordered_answers: dict[str, TaskAnswer] = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="task_answer") as executor:
        future_map = {
            executor.submit(
                answer_single_task,
                question=question,
                question_type=question_type,
                planned_task=item.planned_task,
                category=item.task_category,
                history=history,
                candidates=item.candidates,
                prompt_template=prompt_template,
                chat_model_factory=chat_model_factory,
                question_type_labels=question_type_labels,
                logger=logger,
            ): item
            for item in valid_task_plans
        }
        for future in as_completed(future_map):
            item = future_map[future]
            try:
                task_answer = future.result()
                if task_answer:
                    ordered_answers[item.planned_task.task_id] = task_answer
                    logger.info(
                        "%s parallel_completed | session_id=%s | task_id=%s | answer_length=%s",
                        log_prefix,
                        session_id,
                        item.planned_task.task_id,
                        len(task_answer.answer),
                    )
            except Exception as exc:
                logger.exception(
                    "%s parallel_failed | session_id=%s | task_id=%s | error=%s",
                    log_prefix,
                    session_id,
                    item.planned_task.task_id,
                    exc,
                )

    return [
        ordered_answers[item.planned_task.task_id]
        for item in valid_task_plans
        if item.planned_task.task_id in ordered_answers
    ]


def should_skip_synthesizer(task_answers: list[TaskAnswer], task_reference_groups: list[dict] | None = None) -> bool:
    del task_reference_groups
    return len(task_answers) <= 1


def merge_task_answers(task_answers: list[TaskAnswer]) -> str:
    sections = {
        "最终回答": [],
        "操作步骤/材料清单": [],
        "风险提示": [],
    }
    for item in task_answers:
        parsed = split_sections(item.answer)
        sections["最终回答"].extend(parsed["最终回答"])
        sections["操作步骤/材料清单"].extend(parsed["操作步骤/材料清单"])
        sections["风险提示"].extend(parsed["风险提示"])

    merged_parts = []
    for title in ("最终回答", "操作步骤/材料清单", "风险提示"):
        merged_parts.append(f"### {title}")
        merged_parts.append(dedupe_section_items(sections[title]))
    return "\n\n".join(part.strip() for part in merged_parts if part.strip()).strip()


def split_sections(text: str) -> dict[str, list[str]]:
    cleaned = strip_think_blocks(text).strip()
    pattern = re.compile(
        r"###\s*(最终回答|操作步骤/材料清单|风险提示)\s*\n(.*?)(?=\n###\s*(最终回答|操作步骤/材料清单|风险提示)\s*\n|\Z)",
        re.S,
    )
    sections = {
        "最终回答": [],
        "操作步骤/材料清单": [],
        "风险提示": [],
    }
    matches = list(pattern.finditer(cleaned))
    if not matches:
        sections["最终回答"].append(cleaned)
        return sections
    for match in matches:
        title = match.group(1)
        body = match.group(2).strip()
        if body:
            sections[title].append(body)
    return sections


def dedupe_section_items(blocks: list[str]) -> str:
    if not blocks:
        return "无"
    seen = set()
    normalized_items = []
    for block in blocks:
        pieces = [piece.strip() for piece in re.split(r"\n{2,}", block) if piece.strip()]
        if not pieces:
            pieces = [block.strip()]
        for piece in pieces:
            key = re.sub(r"\s+", " ", piece)
            if key in seen:
                continue
            seen.add(key)
            normalized_items.append(piece)
    return "\n\n".join(normalized_items) if normalized_items else "无"


def finalize_task_answers(
    *,
    question: str,
    task_answers: list[TaskAnswer],
    use_synthesize: bool,
    chat_model_factory,
    logger,
    task_reference_groups: list[dict] | None = None,
) -> tuple[str, dict]:
    if not task_answers:
        return config.NO_EVIDENCE_MESSAGE, {"mode": "empty", "task_count": 0}
    if len(task_answers) == 1:
        return strip_think_blocks(task_answers[0].answer), {"mode": "single_task_passthrough", "task_count": 1}
    skip = should_skip_synthesizer(task_answers, task_reference_groups)
    if use_synthesize and not skip:
        answer = AnswerSynthesizer(chat_model_factory()).synthesize(question, task_answers)
        logger.info(
            "finalize_task_answers completed | mode=%s | task_count=%s | answer_length=%s",
            "synthesizer",
            len(task_answers),
            len(answer),
        )
        return answer, {"mode": "synthesizer", "task_count": len(task_answers), "answer_length": len(answer)}
    answer = merge_task_answers(task_answers)
    logger.info(
        "finalize_task_answers completed | mode=%s | task_count=%s | answer_length=%s",
        "structured_merge",
        len(task_answers),
        len(answer),
    )
    return answer, {"mode": "structured_merge", "task_count": len(task_answers), "answer_length": len(answer)}


__all__ = [
    "AnswerSynthesizer",
    "TaskAnswer",
    "ThinkBlockStreamFilter",
    "answer_single_task",
    "build_context",
    "build_final_answer",
    "build_references",
    "dedupe_section_items",
    "finalize_task_answers",
    "format_reference_markdown",
    "generate_task_answers_parallel",
    "merge_task_answers",
    "should_skip_synthesizer",
    "split_sections",
    "strip_think_blocks",
]
