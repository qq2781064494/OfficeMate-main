"""为 sample_docs/evaluation_samples.json 批量补全 gold_answer。

本脚本使用本地抽取式规则生成标准答案，不依赖外部模型。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import sys

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample_docs"
EVAL_PATH = SAMPLE_DOCS_DIR / "evaluation_samples.json"


def load_title_to_document() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(SAMPLE_DOCS_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            continue
        mapping[lines[0]] = content
    return mapping


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def tokenize(text: str) -> set[str]:
    normalized = normalize_text(text)
    chars = [ch for ch in normalized if "\u4e00" <= ch <= "\u9fff"]
    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    words = re.findall(r"[A-Za-z0-9]+", normalized.lower())
    singles = [ch for ch in chars if ch not in "的是了和或及前后内外在将应需可按与由为把被对等前后最先再还均各类个天月年时分人款项单表中不"]  # noqa: E501
    return set(bigrams + words + singles)


def clean_line(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^[一二三四五六七八九十]+、", "", cleaned)
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    cleaned = re.sub(r"^问：", "", cleaned)
    cleaned = re.sub(r"^答：", "", cleaned)
    return cleaned.strip()


def split_candidates(content: str) -> list[str]:
    lines = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(line.startswith(prefix) for prefix in ("版本：", "生效日期：", "适用范围：", "适用对象：", "分类：")):
            continue
        if re.match(r"^[一二三四五六七八九十]+、", line):
            continue
        if line.startswith(("问：", "答：")):
            continue
        if "：" not in line and "。" not in line and len(line) <= 12:
            continue
        lines.append(line)
    return lines


def score_line(question: str, line: str) -> float:
    q_tokens = tokenize(question)
    l_tokens = tokenize(line)
    if not q_tokens or not l_tokens:
        return 0.0
    overlap = len(q_tokens & l_tokens)
    score = overlap / max(len(q_tokens), 1)

    if any(marker in question for marker in ("几", "多久", "几点", "多少", "多长", "是否", "能不能", "可以")):
        if re.search(r"\d|天|小时|分钟|工作日|自然日|元|前|内|可以|不得|不予|原则上", line):
            score += 0.15
    if "为什么" in question and any(marker in line for marker in ("确保", "避免", "便于", "用于", "原因", "风险")):
        score += 0.15
    if any(marker in question for marker in ("哪些", "包括", "需要什么", "附哪些", "分别负责什么")) and any(marker in line for marker in ("包括", "应包含", "需注明", "填写", "上传", "确认", "核对")):
        score += 0.15
    if any(marker in question for marker in ("除了", "还要", "还需要")):
        if any(marker in line for marker in ("说明", "材料", "凭证", "记录", "审批", "核对", "确认", "附件")):
            score += 0.2
        if any(marker in line for marker in ("不予", "不得", "不能")):
            score -= 0.1
    if line.startswith("答："):
        score += 0.1
    return score


def extract_gold_answer(query: str, expected_titles: list[str], title_to_document: dict[str, str]) -> str:
    ranked: list[tuple[float, str]] = []
    for title in expected_titles:
        content = title_to_document.get(title, "")
        if not content:
            continue
        raw_lines = [line.strip() for line in content.splitlines() if line.strip()]
        lines = split_candidates(content)
        for index, line in enumerate(lines):
            score = score_line(query, line)
            ranked.append((score, line))
        for index, line in enumerate(raw_lines[:-1]):
            next_line = raw_lines[index + 1]
            if line.startswith("问：") and next_line.startswith("答："):
                qa_score = score_line(query, line)
                if qa_score > 0:
                    ranked.append((qa_score + 0.45, next_line))

    ranked.sort(key=lambda item: item[0], reverse=True)
    ranked = [item for item in ranked if item[0] > 0]
    if not ranked:
        return "未找到明确依据"

    selected: list[str] = []
    seen = set()
    allow_second = len(expected_titles) > 1 and any(
        marker in query for marker in ("为什么", "怎么", "怎么办", "处理", "流程", "哪些", "包括", "除了", "分别", "还要")
    )
    for score, line in ranked:
        cleaned = clean_line(line)
        if not cleaned or cleaned in seen:
            continue
        selected.append(cleaned)
        seen.add(cleaned)
        if len(selected) >= 1 and not allow_second:
            break
        if len(selected) >= 2:
            break

    if not selected:
        return "未找到明确依据"
    return " ".join(selected)


def main() -> None:
    title_to_document = load_title_to_document()
    samples = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    output_samples: list[dict] = []
    for index, sample in enumerate(tqdm(samples, desc="Generating gold answers")):
        if index >= 200:
            output_samples.append(sample)
            continue
        expected_titles = sample.get("expected_titles", [])
        gold_answer = extract_gold_answer(sample["query"], expected_titles, title_to_document)
        output_samples.append(
            {
                "query": sample["query"],
                "category": sample["category"],
                "expected_titles": expected_titles,
                "gold_answer": gold_answer,
            }
        )

    EVAL_PATH.write_text(
        json.dumps(output_samples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Updated {EVAL_PATH} with {len(output_samples)} samples.")


if __name__ == "__main__":
    main()
