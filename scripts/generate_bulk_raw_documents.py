"""批量生成模拟办公文档。

这个脚本的定位不是“正式业务逻辑”，而是“造测试数据”：
1. 给 raw_documents 快速补大量文本文件
2. 方便测试上传、管理、检索和批量处理能力
3. 生成的只是演示语料，不会自动写入 documents.json 或向量库

所以可以把它理解为：
“给 RAG 项目准备大批量原始文档样本的辅助脚本”
"""

from __future__ import annotations

import hashlib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "storage" / "raw_documents"
# 这里直接基于项目根目录定位 raw_documents，保证脚本从哪里执行都能找到目标目录。

CATEGORIES = [
    # 每个 category 对应一类企业办公知识主题。
    # 后面会按“每类若干文档”的方式批量生成。
    {
        "slug": "hr",
        "title": "人力资源制度",
        "topics": [
            "请假与考勤",
            "绩效评估",
            "入职培训",
            "试用期管理",
            "员工关怀",
            "晋升评审",
        ],
    },
    {
        "slug": "finance",
        "title": "财务与报销制度",
        "topics": [
            "差旅报销",
            "采购付款",
            "借款冲销",
            "预算控制",
            "发票归档",
            "成本核算",
        ],
    },
    {
        "slug": "it",
        "title": "IT服务与信息安全",
        "topics": [
            "账号开通",
            "VPN使用",
            "设备领用",
            "密码重置",
            "邮件安全",
            "权限回收",
        ],
    },
    {
        "slug": "admin",
        "title": "行政与采购流程",
        "topics": [
            "办公用品申请",
            "固定资产采购",
            "会议室管理",
            "访客接待",
            "印章使用",
            "车辆预约",
        ],
    },
    {
        "slug": "ops",
        "title": "综合运营通知",
        "topics": [
            "季度重点工作",
            "应急值班安排",
            "节假日通知",
            "供应商协同",
            "项目复盘",
            "稽核整改",
        ],
    },
    {
        "slug": "legal",
        "title": "合规与审计规范",
        "topics": [
            "合同审批",
            "档案留存",
            "审计配合",
            "数据脱敏",
            "授权管理",
            "风险排查",
        ],
    },
]

SECTIONS = [
    # 这些 section 会出现在每份模拟文档中，目的是让语料更像制度/流程说明文。
    "适用范围",
    "角色职责",
    "触发条件",
    "操作步骤",
    "材料清单",
    "时效要求",
    "常见异常处理",
    "风险提示",
]


def build_paragraph(category_title: str, topic: str, section: str, index: int, sub_index: int) -> str:
    """构造一小段文本。

    这一步相当于“文档片段模板”：
    - section 决定这一段属于哪个章节
    - topic 决定这份文档围绕什么主题
    - index / sub_index 只是让文本看起来不完全重复
    """
    return (
        f"{section}：第 {index} 号文档围绕“{topic}”进行说明，适用于 {category_title} 场景。"
        f" 负责人需要在业务发生前完成信息确认、模板核验、审批链检查和留痕要求核对。"
        f" 若出现跨部门协同，需同步直属主管、流程管理员和支持团队。"
        f" 第 {sub_index + 1} 条强调执行时限、补充材料、版本差异以及系统录入口径保持一致。"
        f" 对于历史流程与新制度冲突的情况，应以最近发布版本为准，并在流程备注中写明例外原因。"
        f" 所有沟通记录、附件、截图、审批回执与台账编号均应保留，以便复核、抽检和后续追踪。"
    )


def build_document(category: dict[str, object], doc_index: int) -> tuple[str, str]:
    """构造一整份文档的文件名和正文内容。"""
    topic = category["topics"][doc_index % len(category["topics"])]
    title = f"{category['title']} - {topic} 执行说明第 {doc_index:04d} 版"
    lines = [
        title,
        f"文档编号：{category['slug'].upper()}-{doc_index:04d}",
        f"主题标签：{topic}、流程规范、内部知识库、培训示例",
        "版本：v2026.03",
        "更新说明：本文件为批量生成的演示语料，用于扩充本地知识文档规模。",
        "",
    ]

    for section_idx, section in enumerate(SECTIONS):
        # 每个 section 下再生成几段内容，模拟真实制度文档的章节结构。
        lines.append(f"## {section}")
        for paragraph_idx in range(5):
            lines.append(
                build_paragraph(
                    category_title=str(category["title"]),
                    topic=str(topic),
                    section=section,
                    index=doc_index,
                    sub_index=section_idx * 10 + paragraph_idx,
                )
            )
        lines.append("")

    lines.extend(
        [
            "## FAQ",
            f"Q1：如果“{topic}”涉及跨部门会签，应由谁发起？",
            "A1：默认由需求提出部门在业务系统发起，若系统中无入口，则由流程管理员代提并补齐审批说明。",
            f"Q2：若“{topic}”材料不齐是否可以先审批后补件？",
            "A2：原则上不建议。确有紧急业务时，应由审批人明确备注临时放行依据、补件时限和责任人。",
            "Q3：执行后是否需要归档？",
            "A3：需要。归档材料至少包括申请单、审批结果、关键附件、截图和处理结论。",
            "",
            "## 培训提示",
            "建议新同事在正式操作前先阅读本说明、查看最近一次案例记录，并与对应接口人完成一次流程演练。",
        ]
    )

    file_name = f"{category['slug']}_{doc_index:04d}_{str(topic).replace(' ', '_')}.txt"
    return file_name, "\n".join(lines) + "\n"


def main() -> None:
    """批量写入模拟文档。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    docs_per_category = 400
    created = 0
    skipped = 0

    for category in CATEGORIES:
        for doc_index in range(1, docs_per_category + 1):
            file_name, content = build_document(category, doc_index)
            # 这里用内容哈希作为文件名前缀，目的是减少重复写入和文件名冲突。
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
            output_path = RAW_DIR / f"{digest}_{file_name}"
            if output_path.exists():
                skipped += 1
                continue
            output_path.write_text(content, encoding="utf-8")
            created += 1

    # 最后打印统计结果，便于在命令行里快速确认脚本效果。
    print(f"created={created}")
    print(f"skipped={skipped}")
    print(f"total_expected={len(CATEGORIES) * docs_per_category}")


if __name__ == "__main__":
    main()
