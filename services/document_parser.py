"""文档解析器：把不同格式的文件统一转成纯文本。

它是知识入库链路的第一步。
后面的向量切分、embedding、检索都依赖这里先把文件内容提取出来。
"""

from io import BytesIO
from pathlib import Path

import pandas as pd
from docx import Document as DocxDocument
from pypdf import PdfReader
from utils.log_tool import get_logger


logger = get_logger("document_parser")


class DocumentParser:
    def parse(self, file_name, file_bytes):
        """根据文件后缀选择合适的解析方式。

        返回值：
        - normalized: 规范化后的纯文本
        - suffix: 原始文件后缀，后续会写入文档元数据
        """
        suffix = Path(file_name).suffix.lower()
        if suffix == ".txt":
            text = self._decode_text(file_bytes)
        elif suffix == ".pdf":
            text = self._parse_pdf(file_bytes)
        elif suffix == ".docx":
            text = self._parse_docx(file_bytes)
        elif suffix == ".xlsx":
            text = self._parse_excel(file_bytes)
        elif suffix == ".csv":
            text = self._parse_csv(file_bytes)
        else:
            raise ValueError(f"暂不支持的文件类型: {suffix}")

        normalized = self._normalize_text(text)
        if not normalized:
            raise ValueError("文档中没有提取到可用文本，请检查文件内容。")
        logger.info(
            "document_parser parsed | file_name=%s | suffix=%s | text_length=%s",
            file_name,
            suffix,
            len(normalized),
        )
        return normalized, suffix

    def _decode_text(self, file_bytes):
        """按常见中文办公文档编码顺序尝试解码纯文本文件。"""
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("文本文件编码无法识别，建议转为 UTF-8 后再上传。")

    def _parse_pdf(self, file_bytes):
        """逐页提取 PDF 文本，并把每页内容拼接起来。"""
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    def _parse_docx(self, file_bytes):
        """提取 Word 文档中的非空段落文本。"""
        document = DocxDocument(BytesIO(file_bytes))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        return "\n".join(paragraphs)

    def _parse_excel(self, file_bytes):
        """把 Excel 每个工作表转换成“表头 + 行内容”的文本表示。"""
        excel_file = pd.ExcelFile(BytesIO(file_bytes))
        sheet_texts = []
        for sheet_name in excel_file.sheet_names:
            dataframe = excel_file.parse(sheet_name, dtype=str).fillna("")
            sheet_texts.append(f"工作表：{sheet_name}\n{self._dataframe_to_text(dataframe)}")
        return "\n\n".join(sheet_texts)

    def _parse_csv(self, file_bytes):
        """按常见编码读取 CSV，并转成统一的文本表格格式。"""
        dataframe = None
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                dataframe = pd.read_csv(BytesIO(file_bytes), dtype=str, encoding=encoding).fillna("")
                break
            except UnicodeDecodeError:
                continue
        if dataframe is None:
            raise ValueError("CSV 文件编码无法识别，建议转为 UTF-8 后再上传。")
        return self._dataframe_to_text(dataframe)

    def _dataframe_to_text(self, dataframe):
        """把 DataFrame 序列化成便于向量检索的多行文本。"""
        headers = [str(column).strip() for column in dataframe.columns.tolist()]
        lines = [" | ".join(headers)] if headers else []
        for _, row in dataframe.iterrows():
            values = [str(value).strip() for value in row.tolist()]
            if any(values):
                lines.append(" | ".join(values))
        return "\n".join(lines)

    def _normalize_text(self, text):
        """做一层轻量清洗，去掉 BOM、空行和多余换行差异。"""
        clean_text = text.replace("\ufeff", "")
        lines = [line.strip() for line in clean_text.replace("\r\n", "\n").split("\n")]
        filtered = [line for line in lines if line]
        return "\n".join(filtered)
