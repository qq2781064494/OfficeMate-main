"""文档服务层。

它是“知识上传”和“知识删除”的总调度器：
- 解析文件
- 计算哈希避免重复导入
- 保存原始文件
- 写入向量库
- 更新文档索引
"""

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import config_data as config
from services.document_parser import DocumentParser
from services.storage_service import JsonStorageService
from services.vector_store import OfficeMateVectorStore
from utils.log_tool import get_logger


logger = get_logger("document_service")


class DocumentService:
    def __init__(self):
        # storage 管理 JSON 元数据，parser 负责提取文本，vector_store 负责向量入库。
        self.storage = JsonStorageService()
        self.parser = DocumentParser()
        self.vector_store = None

    def ingest_uploaded_file(self, uploaded_file, category, version, custom_title=""):
        """处理来自 Streamlit 上传组件的文件对象。"""
        if Path(uploaded_file.name).suffix.lower() == ".zip":
            return self.ingest_zip_file(
                uploaded_file={
                    "file_name": uploaded_file.name,
                    "file_bytes": uploaded_file.getvalue(),
                },
                category=category,
                version=version,
            )
        # 如果用户没有输入自定义标题，就默认用文件名去掉后缀作为标题。
        title = custom_title.strip() or Path(uploaded_file.name).stem
        logger.info(
            "document_upload received | file_name=%s | category=%s | version=%s | title=%s",
            uploaded_file.name,
            category,
            version,
            title,
        )
        return self.ingest_bytes(
            file_name=uploaded_file.name,
            file_bytes=uploaded_file.getvalue(),
            category=category,
            title=title,
            version=version.strip() or config.DEFAULT_VERSION,
            source_label="manual_upload",
        )

    def ingest_zip_file(self, uploaded_file, category, version):
        """处理 zip 压缩包，解包后批量导入其中的支持格式文件。"""
        zip_bytes = uploaded_file["file_bytes"]
        results = []
        try:
            with ZipFile(BytesIO(zip_bytes)) as zip_file:
                for member_name in zip_file.namelist():
                    member_path = Path(member_name)
                    if member_name.endswith("/") or member_path.name.startswith("."):
                        continue
                    suffix = member_path.suffix.lower().lstrip(".")
                    if suffix not in config.SUPPORTED_FILE_TYPES or suffix == "zip":
                        continue
                    results.append(
                        self.ingest_bytes(
                            file_name=member_path.name,
                            file_bytes=zip_file.read(member_name),
                            category=category,
                            title=member_path.stem,
                            version=version.strip() or config.DEFAULT_VERSION,
                            source_label="manual_upload_zip",
                        )
                    )
        except BadZipFile as exc:
            logger.warning("document_zip invalid | file_name=%s | error=%s", uploaded_file["file_name"], exc)
            return [
                {
                    "status": "failed",
                    "message": f"压缩包《{uploaded_file['file_name']}》无法解压，请确认 zip 文件未损坏。",
                }
            ]

        if not results:
            logger.info("document_zip empty | file_name=%s | size=%s", uploaded_file["file_name"], len(zip_bytes))
            return [
                {
                    "status": "failed",
                    "message": f"压缩包《{uploaded_file['file_name']}》里没有可导入的 txt/pdf/docx/xlsx/csv 文件。",
                }
            ]

        logger.info(
            "document_zip imported | file_name=%s | extracted_count=%s | category=%s | version=%s",
            uploaded_file["file_name"],
            len(results),
            category,
            version,
        )
        return results

    def ingest_bytes(self, file_name, file_bytes, category, title, version, source_label):
        """真正的入库主流程。

        可以给上传文件用，也可以给示例文档导入用，因为二者最后都会变成
        “文件名 + 字节流 + 元数据”这一套通用输入。
        """
        prepared = self.prepare_upload_item(
            {
                "file_name": file_name,
                "file_bytes": file_bytes,
                "category": category,
                "title": title,
                "version": version,
                "source_label": source_label,
            }
        )
        registration = self.register_prepared_document(prepared)
        if registration["status"] == "duplicate":
            return registration["result"]
        embedded = self.embed_prepared_document(registration["prepared"])
        return self.finalize_prepared_document(embedded)
        

    def delete_document(self, document_id):
        """删除文档时，同时删除向量索引、原始文件和 JSON 元数据。"""
        record = self.storage.get_document_by_id(document_id)
        if not record:
            return {
                "status": "not_found",
                "message": "未找到对应的知识文档。",
            }

        title = record.get("title", "未命名文档")
        logger.info("document_delete start | document_id=%s | title=%s", document_id, title)
        try:
            if record.get("status") == "success":
                # 已成功入库的文档，需要把向量片段一起清掉。
                self._get_vector_store().delete_document(
                    document_id=document_id,
                    chunk_count=record.get("chunk_count", 0),
                )

            raw_path = record.get("raw_path")
            if raw_path:
                raw_file = config.BASE_DIR / raw_path
                if raw_file.exists():
                    raw_file.unlink()

            deleted = self.storage.delete_document(document_id)
            if not deleted:
                return {
                    "status": "not_found",
                    "message": "未找到对应的知识文档。",
                }
        except Exception as exc:
            return {
                "status": "failed",
                "message": f"删除《{title}》失败：{exc}",
                "document": record,
            }

        return {
            "status": "success",
            "message": f"已删除《{title}》，并同步移除原始文件与知识库索引。",
            "document": record,
        }

    def seed_sample_documents(self):
        """批量导入项目内置示例文档。"""
        results = []
        for sample in config.SAMPLE_DOCS:
            file_path = config.SAMPLE_DOC_DIR / sample["file_name"]
            results.append(
                self.ingest_bytes(
                    file_name=sample["file_name"],
                    file_bytes=file_path.read_bytes(),
                    category=sample["category"],
                    title=sample["title"],
                    version=sample["version"],
                    source_label="sample_docs",
                )
            )
        logger.info("document_seed completed | count=%s", len(results))
        return results

    def list_documents(self):
        """给页面层提供文档列表。"""
        return self.storage.list_documents()

    def expand_upload_items(self, source_files, category, version, custom_title=""):
        """把上传源展开成统一的文档项列表，zip 会在这里解包。"""
        items = []
        default_version = version.strip() or config.DEFAULT_VERSION
        for source_file in source_files:
            file_name = source_file["file_name"]
            file_bytes = source_file["file_bytes"]
            suffix = Path(file_name).suffix.lower()
            if suffix == ".zip":
                items.extend(self._expand_zip_items(file_name, file_bytes, category, default_version))
                continue
            title = custom_title.strip() or Path(file_name).stem
            items.append(
                {
                    "file_name": file_name,
                    "file_bytes": file_bytes,
                    "category": category,
                    "title": title,
                    "version": default_version,
                    "source_label": "manual_upload",
                }
            )
        return items

    def prepare_upload_item(self, item):
        """并行阶段使用：解析文本、计算哈希、切片，但不写库。"""
        file_name = item["file_name"]
        file_bytes = item["file_bytes"]
        text, file_suffix = self.parser.parse(file_name, file_bytes)
        chunks = self._get_vector_store().split_text(text)
        prepared = {
            **item,
            "file_hash": hashlib.sha256(file_bytes).hexdigest(),
            "file_type": file_suffix.lstrip("."),
            "text_length": len(text),
            "chunks": chunks,
        }
        logger.info(
            "document_prepare success | file_name=%s | chunk_count=%s | text_length=%s",
            file_name,
            len(chunks),
            len(text),
        )
        return prepared

    def register_prepared_document(self, prepared):
        """串行阶段使用：处理重复、落 processing 记录、保存原始文件。"""
        existing = self.storage.get_document_by_hash(prepared["file_hash"])
        if existing and existing.get("status") == "success":
            logger.info("document_ingest duplicate | file_name=%s | existing_id=%s", prepared["file_name"], existing["id"])
            return {
                "status": "duplicate",
                "result": {
                    "status": "duplicate",
                    "message": f"《{existing['title']}》已存在，已跳过重复导入。",
                    "document": existing,
                },
            }

        raw_path = self._save_raw_file(prepared["file_hash"], prepared["file_name"], prepared["file_bytes"])
        record_payload = {
            "file_hash": prepared["file_hash"],
            "file_name": prepared["file_name"],
            "file_type": prepared["file_type"],
            "title": prepared["title"],
            "category": prepared["category"],
            "version": prepared["version"],
            "source_label": prepared["source_label"],
            "raw_path": str(raw_path.relative_to(config.BASE_DIR)),
            "text_length": prepared["text_length"],
            "chunk_count": 0,
            "status": "processing",
            "error": "",
        }
        if existing:
            record = self.storage.update_document(existing["id"], record_payload)
        else:
            record = self.storage.add_document(record_payload)
        prepared["record"] = record
        return {
            "status": "processing",
            "prepared": prepared,
        }

    def embed_prepared_document(self, prepared):
        """并发 embedding 阶段使用。"""
        prepared["embeddings"] = self._get_vector_store().embed_chunks(prepared["chunks"])
        return prepared

    def finalize_prepared_document(self, prepared):
        """串行阶段使用：把 embeddings 写入向量库并更新成功状态。"""
        record = prepared["record"]
        try:
            chunk_count = self._get_vector_store().add_embeddings(
                document_id=record["id"],
                chunks=prepared["chunks"],
                embeddings=prepared.get("embeddings", []),
                metadata={
                    "title": prepared["title"],
                    "category": prepared["category"],
                    "version": prepared["version"],
                    "file_name": prepared["file_name"],
                    "uploaded_at": record["uploaded_at"],
                },
            )
            updated_record = self.storage.update_document(
                record["id"],
                {
                    "chunk_count": chunk_count,
                    "status": "success",
                    "error": "",
                },
            )
            return {
                "status": "success",
                "message": f"《{prepared['title']}》导入成功，共切分 {chunk_count} 个片段。",
                "document": updated_record,
            }
        except Exception as exc:
            return self.mark_prepared_document_failed(prepared, exc)

    def mark_prepared_document_failed(self, prepared, exc):
        """更新已登记文档的失败状态。"""
        logger.exception(
            "document_ingest failed | file_name=%s | title=%s | error=%s",
            prepared["file_name"],
            prepared["title"],
            exc,
        )
        record = prepared.get("record")
        updated_record = None
        if record:
            updated_record = self.storage.update_document(
                record["id"],
                {
                    "status": "failed",
                    "error": str(exc),
                },
            )
        return {
            "status": "failed",
            "message": f"《{prepared['title']}》导入失败：{exc}",
            "document": updated_record or {},
            "title": prepared["title"],
            "file_name": prepared["file_name"],
        }

    def build_failed_result(self, title, file_name, exc):
        """构造预处理阶段失败结果。"""
        logger.exception("document_prepare failed | file_name=%s | title=%s | error=%s", file_name, title, exc)
        return {
            "status": "failed",
            "message": f"《{title}》导入失败：{exc}",
            "document": {},
            "title": title,
            "file_name": file_name,
        }

    def _save_raw_file(self, file_hash, file_name, file_bytes):
        """把原始文件保存到 storage/raw_documents。"""
        # 文件名前缀加入哈希片段，既便于追踪，也能减少同名覆盖风险。
        safe_name = file_name.replace(" ", "_")
        raw_path = config.RAW_DOCUMENT_DIR / f"{file_hash[:12]}_{safe_name}"
        raw_path.write_bytes(file_bytes)
        return raw_path

    def _get_vector_store(self):
        """延迟初始化向量库对象，避免页面一打开就提前创建。"""
        if self.vector_store is None:
            self.vector_store = OfficeMateVectorStore()
        return self.vector_store

    def _expand_zip_items(self, file_name, file_bytes, category, version):
        """把 zip 展开成文档项；只保留支持的文件类型。"""
        items = []
        try:
            with ZipFile(BytesIO(file_bytes)) as zip_file:
                for member_name in zip_file.namelist():
                    member_path = Path(member_name)
                    if member_name.endswith("/") or member_path.name.startswith("."):
                        continue
                    suffix = member_path.suffix.lower().lstrip(".")
                    if suffix not in config.SUPPORTED_FILE_TYPES or suffix == "zip":
                        continue
                    items.append(
                        {
                            "file_name": member_path.name,
                            "file_bytes": zip_file.read(member_name),
                            "category": category,
                            "title": member_path.stem,
                            "version": version,
                            "source_label": "manual_upload_zip",
                        }
                    )
        except BadZipFile as exc:
            logger.warning("document_zip invalid | file_name=%s | error=%s", file_name, exc)
        return items
