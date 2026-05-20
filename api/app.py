"""FastAPI 应用入口。"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config_data as config
from api.helpers import build_chunk_config, resolve_sample_path
from api.schemas import (
    BenchmarkBuildCorpusRequest,
    BenchmarkBuildIndexRequest,
    BenchmarkRunRequest,
    ChatAskRequest,
    FeedbackRequest,
    LocalEvalKnowledgeBaseCreateRequest,
    LocalEvalRunRequest,
    SeedDocumentsRequest,
)
from core.bootstrap import bootstrap_runtime
from decision_react.service import DecisionReactService
from agent_react_rag.service import AgentReactRagService
from services.background_executor import get_background_executor
from services.benchmark_eval_service import BenchmarkEvalConfig, BenchmarkEvalService
from services.benchmark_results import BenchmarkResultStore
from services.benchmark_store import BenchmarkCorpusStore
from services.chat_service import OfficeMateChatService
from services.document_service import DocumentService
from services.local_eval_service import LocalEvalConfig, LocalEvalService
from services.local_eval_store import LocalEvalCorpusStore
from services.rag.contracts import ChatRequest
from services.storage_service import JsonStorageService
from services.task_run_service import TaskRunService


app = FastAPI(title="OfficeMate API", version="1.0.0")
frontend_dist_dir = config.BASE_DIR / "frontend" / "dist"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _serialize_response_object(result: object):
    """兼容 dataclass / Pydantic / 普通对象，统一转成 API 可返回的数据。"""
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    raise TypeError(f"Unsupported response type: {type(result)!r}")


@app.on_event("startup")
def on_startup() -> None:
    bootstrap_runtime()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": config.APP_NAME,
        "mysql_database": config.mysql_database,
        "milvus_host": config.milvus_host,
        "milvus_port": config.milvus_port,
    }


@app.get("/admin/stats")
def admin_stats():
    return JsonStorageService().get_stats()


@app.get("/documents")
def list_documents():
    return JsonStorageService().list_documents()


@app.get("/documents/{document_id}")
def get_document(document_id: str):
    record = JsonStorageService().get_document_by_id(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="未找到对应文档。")
    return record


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    result = DocumentService().delete_document(document_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=result["message"])
    if result["status"] == "failed":
        raise HTTPException(status_code=500, detail=result["message"])
    return result


@app.post("/documents/seed")
def seed_documents(request: SeedDocumentsRequest):
    service = DocumentService()
    if not request.run_async:
        return service.seed_sample_documents()
    task = get_background_executor().submit(
        task_type="seed_documents",
        payload=request.model_dump(),
        runner=service.seed_sample_documents,
    )
    return {"task_id": task["id"], "status": task["status"]}


@app.post("/documents/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    category: str = "综合公告",
    version: str = config.DEFAULT_VERSION,
    custom_title: str = "",
):
    source_files = []
    for file in files:
        source_files.append(
            {
                "file_name": file.filename or "upload.bin",
                "file_bytes": await file.read(),
            }
        )

    def runner():
        service = DocumentService()
        items = service.expand_upload_items(source_files, category=category, version=version, custom_title=custom_title)
        results = []
        for item in items:
            try:
                prepared = service.prepare_upload_item(item)
                registration = service.register_prepared_document(prepared)
                if registration["status"] == "duplicate":
                    results.append(registration["result"])
                    continue
                embedded = service.embed_prepared_document(registration["prepared"])
                results.append(service.finalize_prepared_document(embedded))
            except Exception as exc:
                results.append(service.build_failed_result(item["title"], item["file_name"], exc))
        return {"results": results}

    task = get_background_executor().submit(
        task_type="upload_documents",
        payload={"file_names": [item["file_name"] for item in source_files], "category": category, "version": version},
        runner=runner,
    )
    return {"task_id": task["id"], "status": task["status"]}


@app.post("/chat/ask")
def chat_ask(request: ChatAskRequest):
    service = OfficeMateChatService()
    return service.answer_question(
        question=request.question,
        session_id=request.session_id,
        category=request.category,
        use_history=request.use_history,
        persist_log=request.persist_log,
        include_references=request.include_references,
        enable_query_rewrite=request.enable_query_rewrite,
        enable_rerank=request.enable_rerank,
        reference_limit=request.reference_limit,
    )


@app.post("/chat/stream")
def chat_stream(request: ChatAskRequest):
    service = OfficeMateChatService()
    event_queue: Queue[tuple[str, dict]] = Queue()

    def update_status(message: str) -> None:
        event_queue.put(("status", {"message": message}))

    def update_event(phase: str, payload: dict) -> None:
        event_queue.put(("phase_result", {"phase": phase, **payload}))

    session = service.stream_chat(
        ChatRequest(
            question=request.question,
            session_id=request.session_id,
            category=request.category,
            use_history=request.use_history,
            persist_log=request.persist_log,
            include_references=request.include_references,
            enable_query_rewrite=request.enable_query_rewrite,
            enable_rerank=request.enable_rerank,
            reference_limit=request.reference_limit,
            status_callback=update_status,
            event_callback=update_event,
        )
    )

    def sse_event(event_name: str, payload: dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def producer() -> None:
        try:
            for chunk in session.stream:
                event_queue.put(("chunk", {"content": chunk}))
            if session.result_holder.response is not None:
                event_queue.put(("meta", session.result_holder.response.to_legacy_dict()))
            event_queue.put(("done", {"status": "completed"}))
        except Exception as exc:
            event_queue.put(("error", {"message": str(exc)}))
            event_queue.put(("done", {"status": "failed"}))

    def event_stream():
        producer_thread = Thread(target=producer, daemon=True)
        producer_thread.start()
        while True:
            try:
                event_name, payload = event_queue.get(timeout=15)
            except Empty:
                yield sse_event("ping", {"status": "waiting"})
                continue
            yield sse_event(event_name, payload)
            if event_name == "done":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/feedback")
def save_feedback(request: FeedbackRequest):
    return JsonStorageService().upsert_feedback(
        qa_log_id=request.qa_log_id,
        rating=request.rating,
        comment=request.comment,
        session_id=request.session_id,
    )


@app.get("/sessions/{session_id}/logs")
def list_session_logs(session_id: str, limit: int | None = None):
    return JsonStorageService().list_session_logs(session_id, limit=limit)


@app.post("/agent/decision-react/ask")
def decision_react_ask(request: ChatAskRequest):
    result = DecisionReactService().answer_question(
        question=request.question,
        session_id=request.session_id,
        category=request.category,
    )
    return _serialize_response_object(result)


@app.post("/agent/react-rag/ask")
def agent_react_ask(request: ChatAskRequest):
    result = AgentReactRagService().answer_question(
        question=request.question,
        session_id=request.session_id,
        category=request.category,
    )
    return _serialize_response_object(result)


@app.post("/benchmark/ragbench/build-corpus")
def benchmark_build_corpus(request: BenchmarkBuildCorpusRequest):
    def runner():
        return BenchmarkCorpusStore().build_subset_corpus(
            subset=request.subset,
            splits=request.splits,
            rebuild=request.rebuild,
        )

    task = get_background_executor().submit(
        task_type="benchmark_build_corpus",
        payload=request.model_dump(),
        runner=runner,
    )
    return {"task_id": task["id"], "status": task["status"]}


@app.post("/benchmark/ragbench/build-index")
def benchmark_build_index(request: BenchmarkBuildIndexRequest):
    chunk_config = build_chunk_config(
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        max_split_char_number=request.max_split_char_number,
    )

    def runner():
        return BenchmarkCorpusStore().ensure_vector_index(
            subset=request.subset,
            rebuild=request.rebuild,
            chunk_config=chunk_config,
        )

    task = get_background_executor().submit(
        task_type="benchmark_build_index",
        payload=request.model_dump(),
        runner=runner,
    )
    return {"task_id": task["id"], "status": task["status"]}


@app.post("/benchmark/ragbench/run")
def benchmark_run(request: BenchmarkRunRequest):
    chunk_config = build_chunk_config(
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        max_split_char_number=request.max_split_char_number,
    )

    def runner():
        service = BenchmarkEvalService()
        return service.run_evaluation(
            BenchmarkEvalConfig(
                subset=request.subset,
                split=request.split,
                retriever_strategy=request.retriever_strategy,
                top_k=request.top_k,
                question_limit=request.question_limit,
                enable_query_rewrite=request.enable_query_rewrite,
                enable_ragas=request.enable_ragas,
                enable_faithfulness=request.enable_faithfulness,
                enable_rerank=request.enable_rerank,
                rebuild_corpus=request.rebuild_corpus,
                rebuild_index=request.rebuild_index,
                chunk_config=chunk_config,
            )
        )

    task = get_background_executor().submit(
        task_type="benchmark_run",
        payload=request.model_dump(),
        runner=runner,
    )
    return {"task_id": task["id"], "status": task["status"]}


@app.get("/benchmark/runs")
def benchmark_runs(limit: int = 20, subset: str | None = None):
    return BenchmarkResultStore().list_runs(limit=limit, subset=subset)


@app.get("/benchmark/subsets")
def benchmark_subsets():
    return BenchmarkCorpusStore().list_available_subsets()


@app.get("/benchmark/runs/{run_id}")
def benchmark_run_detail(run_id: str):
    store = BenchmarkResultStore()
    summary = store.get_run_summary(run_id)
    if not summary:
        raise HTTPException(status_code=404, detail="未找到对应评测结果。")
    return {"summary": summary, "details": store.load_run_details(run_id)}


@app.post("/local-eval/knowledge-bases")
def create_local_eval_knowledge_base(request: LocalEvalKnowledgeBaseCreateRequest):
    chunk_config = build_chunk_config(
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        max_split_char_number=request.max_split_char_number,
    )

    def runner():
        return LocalEvalCorpusStore().build_knowledge_base(
            knowledge_base_name=request.knowledge_base_name,
            chunk_config=chunk_config,
            rebuild=request.rebuild,
        )

    task = get_background_executor().submit(
        task_type="local_eval_build_kb",
        payload=request.model_dump(),
        runner=runner,
    )
    return {"task_id": task["id"], "status": task["status"]}


@app.get("/local-eval/knowledge-bases")
def list_local_eval_knowledge_bases():
    return LocalEvalCorpusStore().list_knowledge_bases()


@app.get("/local-eval/datasets")
def list_local_eval_datasets():
    return LocalEvalCorpusStore().list_available_datasets()


@app.post("/local-eval/run")
def run_local_eval(request: LocalEvalRunRequest):
    chunk_config = build_chunk_config(
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        max_split_char_number=request.max_split_char_number,
    )

    def runner():
        service = LocalEvalService()
        return service.run_evaluation(
            LocalEvalConfig(
                knowledge_base_id=request.knowledge_base_id,
                knowledge_base_name=request.knowledge_base_name,
                dataset_key=request.dataset_key,
                dataset_label=request.dataset_label,
                sample_path=resolve_sample_path(request.sample_path),
                retriever_strategy=request.retriever_strategy,
                top_k=request.top_k,
                question_limit=request.question_limit,
                selected_question_ids=request.selected_question_ids,
                enable_query_rewrite=request.enable_query_rewrite,
                enable_ragas=request.enable_ragas,
                enable_faithfulness=request.enable_faithfulness,
                enable_rerank=request.enable_rerank,
                chunk_config=chunk_config,
            )
        )

    task = get_background_executor().submit(
        task_type="local_eval_run",
        payload=request.model_dump(),
        runner=runner,
    )
    return {"task_id": task["id"], "status": task["status"]}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = TaskRunService().get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="未找到对应任务。")
    return task


@app.get("/tasks")
def list_tasks(limit: int = 20, task_type: str | None = None, status: str | None = None):
    return TaskRunService().list_tasks(limit=limit, task_type=task_type, status=status)


if frontend_dist_dir.exists():
    assets_dir = frontend_dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/")
    def frontend_index():
        return FileResponse(frontend_dist_dir / "index.html")

    @app.get("/{full_path:path}")
    def frontend_spa_fallback(full_path: str):
        if full_path.startswith(("documents", "chat", "feedback", "sessions", "agent", "benchmark", "local-eval", "tasks", "admin", "health", "openapi.json", "docs", "redoc")):
            raise HTTPException(status_code=404, detail="未找到对应接口。")
        candidate = frontend_dist_dir / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(frontend_dist_dir / "index.html")
