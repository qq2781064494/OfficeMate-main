const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function resolveStreamBaseUrl() {
  if (API_BASE_URL) {
    return API_BASE_URL;
  }
  if (typeof window !== "undefined" && window.location.port === "5173") {
    return "http://127.0.0.1:8001";
  }
  return "";
}

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    let detail = `请求失败: ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  if (response.status === 204) {
    return null;
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

export function getHealth() {
  return apiFetch("/health");
}

export function getStats() {
  return apiFetch("/admin/stats");
}

export function listDocuments() {
  return apiFetch("/documents");
}

export function deleteDocument(documentId) {
  return apiFetch(`/documents/${documentId}`, { method: "DELETE" });
}

export function uploadDocuments({ files, category, version, customTitle }) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  const params = new URLSearchParams({
    category,
    version,
    custom_title: customTitle || ""
  });
  return apiFetch(`/documents/upload?${params.toString()}`, {
    method: "POST",
    body: formData
  });
}

export function seedDocuments(runAsync = true) {
  return apiFetch("/documents/seed", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_async: runAsync })
  });
}

export function askChat(payload) {
  return apiFetch("/chat/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function streamChat(payload, handlers = {}) {
  const response = await fetch(`${resolveStreamBaseUrl()}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    let detail = `请求失败: ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  if (!response.body) {
    throw new Error("浏览器当前环境不支持流式响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const emitEvent = (rawEvent) => {
    const lines = rawEvent.split("\n");
    let eventName = "message";
    const dataLines = [];
    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }
    if (!dataLines.length) {
      return;
    }
    const payloadText = dataLines.join("\n");
    const payloadObject = JSON.parse(payloadText);
    if (eventName === "status" && handlers.onStatus) {
      handlers.onStatus(payloadObject);
    } else if (eventName === "chunk" && handlers.onChunk) {
      handlers.onChunk(payloadObject);
    } else if (eventName === "meta" && handlers.onMeta) {
      handlers.onMeta(payloadObject);
    } else if (eventName === "phase_result" && handlers.onPhaseResult) {
      handlers.onPhaseResult(payloadObject);
    } else if (eventName === "error" && handlers.onError) {
      handlers.onError(payloadObject);
    } else if (eventName === "done" && handlers.onDone) {
      handlers.onDone(payloadObject);
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let separatorIndex = buffer.indexOf("\n\n");
    while (separatorIndex >= 0) {
      const rawEvent = buffer.slice(0, separatorIndex).trim();
      buffer = buffer.slice(separatorIndex + 2);
      if (rawEvent) {
        emitEvent(rawEvent);
      }
      separatorIndex = buffer.indexOf("\n\n");
    }
    if (done) {
      const rawEvent = buffer.trim();
      if (rawEvent) {
        emitEvent(rawEvent);
      }
      break;
    }
  }
}

export function askDecisionReact(payload) {
  return apiFetch("/agent/decision-react/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function askAgentReact(payload) {
  return apiFetch("/agent/react-rag/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function listSessionLogs(sessionId, limit = 20) {
  return apiFetch(`/sessions/${encodeURIComponent(sessionId)}/logs?limit=${limit}`);
}

export function saveFeedback(payload) {
  return apiFetch("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function listTasks(limit = 20) {
  return apiFetch(`/tasks?limit=${limit}`);
}

export function getTask(taskId) {
  return apiFetch(`/tasks/${taskId}`);
}

export function listBenchmarkSubsets() {
  return apiFetch("/benchmark/subsets");
}

export function listBenchmarkRuns(limit = 20) {
  return apiFetch(`/benchmark/runs?limit=${limit}`);
}

export function getBenchmarkRunDetail(runId) {
  return apiFetch(`/benchmark/runs/${encodeURIComponent(runId)}`);
}

export function buildBenchmarkCorpus(payload) {
  return apiFetch("/benchmark/ragbench/build-corpus", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function buildBenchmarkIndex(payload) {
  return apiFetch("/benchmark/ragbench/build-index", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function runBenchmark(payload) {
  return apiFetch("/benchmark/ragbench/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function listLocalEvalKnowledgeBases() {
  return apiFetch("/local-eval/knowledge-bases");
}

export function listLocalEvalDatasets() {
  return apiFetch("/local-eval/datasets");
}

export function createLocalEvalKnowledgeBase(payload) {
  return apiFetch("/local-eval/knowledge-bases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function runLocalEval(payload) {
  return apiFetch("/local-eval/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}
