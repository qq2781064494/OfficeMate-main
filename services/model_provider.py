"""统一的模型 Provider 与工厂。"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, List

import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config_data as config
from utils.langchain_compat import ensure_langchain_runtime_compatibility


@dataclass(frozen=True)
class ProviderConfig:
    """单个能力所需的 provider 配置。"""

    provider_name: str
    model_name: str
    api_key: str
    base_url: str
    timeout_seconds: float | None = None


class BaseLLMProvider(ABC):
    """统一的模型 provider 基类。"""

    provider_name = "base"

    def __init__(self, provider_config: ProviderConfig):
        self.provider_config = provider_config

    def build_chat_model(self, **kwargs) -> Any:
        raise NotImplementedError(f"{self.provider_name} does not implement chat model creation")

    def build_embedding_client(self, **kwargs) -> Any:
        raise NotImplementedError(f"{self.provider_name} does not implement embedding client creation")

    def rerank(self, query: str, documents: List[str], top_n: int) -> list[dict] | None:
        raise NotImplementedError(f"{self.provider_name} does not implement rerank")


class OpenAICompatibleProvider(BaseLLMProvider):
    """适配 OpenAI-compatible / vLLM / oMLX 风格接口。"""

    provider_name = "openai_compatible"

    def build_chat_model(self, **kwargs) -> ChatOpenAI:
        ensure_langchain_runtime_compatibility()
        return ChatOpenAI(
            model=self.provider_config.model_name,
            api_key=self.provider_config.api_key,
            base_url=self.provider_config.base_url or None,
            **kwargs,
        )

    def build_embedding_client(self, **kwargs) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self.provider_config.model_name,
            api_key=self.provider_config.api_key or "local",
            base_url=self.provider_config.base_url,
            **kwargs,
        )

    def rerank(self, query: str, documents: List[str], top_n: int) -> list[dict] | None:
        base_url = (self.provider_config.base_url or "").rstrip("/")
        if not base_url or not documents:
            return None

        endpoint = f"{base_url}/rerank"
        payload = {
            "model": self.provider_config.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.provider_config.api_key:
            headers["Authorization"] = f"Bearer {self.provider_config.api_key}"

        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=self.provider_config.timeout_seconds,
        )
        response.raise_for_status()
        response_json = response.json()
        if isinstance(response_json, dict):
            if isinstance(response_json.get("results"), list):
                return response_json["results"]
            if isinstance(response_json.get("data"), list):
                return response_json["data"]
        return None


class MLXProvider(OpenAICompatibleProvider):
    """oMLX Provider。

    当前实现沿用 OpenAI-compatible 协议，但把 provider 身份显式区分出来，
    方便后续接入 MLX 专属逻辑而不影响上层业务代码。
    """

    provider_name = "mlx"


class ModelProviderFactory:
    """统一创建 chat / embedding / rerank provider。"""

    _PROVIDER_MAP = {
        "openai_compatible": OpenAICompatibleProvider,
        "mlx": MLXProvider,
    }

    @classmethod
    def create(cls, provider_name: str, *, model_name: str, api_key: str, base_url: str, timeout_seconds: float | None = None) -> BaseLLMProvider:
        normalized_name = (provider_name or "openai_compatible").strip().lower()
        provider_cls = cls._PROVIDER_MAP.get(normalized_name)
        if provider_cls is None:
            raise ValueError(f"Unsupported model provider: {provider_name}")
        return provider_cls(
            ProviderConfig(
                provider_name=normalized_name,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
        )

    @classmethod
    def create_chat_provider(cls) -> BaseLLMProvider:
        return cls.create(
            config.chat_provider,
            model_name=config.chat_model_name,
            api_key=config.chat_api_key,
            base_url=config.chat_base_url,
        )

    @classmethod
    def create_rewrite_provider(cls) -> BaseLLMProvider:
        return cls.create(
            config.rewrite_provider,
            model_name=config.rewrite_model_name,
            api_key=config.rewrite_api_key,
            base_url=config.rewrite_base_url,
        )

    @classmethod
    def create_task_provider(cls) -> BaseLLMProvider:
        return cls.create(
            config.task_provider,
            model_name=config.task_model_name,
            api_key=config.task_api_key,
            base_url=config.task_base_url,
        )

    @classmethod
    def create_embedding_provider(cls) -> BaseLLMProvider:
        return cls.create(
            config.embedding_provider,
            model_name=config.embedding_model_name,
            api_key=config.embedding_api_key or "local",
            base_url=config.embedding_base_url,
        )

    @classmethod
    def create_rerank_provider(cls) -> BaseLLMProvider:
        return cls.create(
            config.rerank_provider,
            model_name=config.rerank_model_name,
            api_key=config.rerank_api_key,
            base_url=config.rerank_base_url,
            timeout_seconds=config.rerank_timeout_seconds,
        )

    @classmethod
    def create_benchmark_chat_provider(cls) -> BaseLLMProvider:
        return cls.create(
            config.benchmark_chat_provider,
            model_name=config.benchmark_chat_model_name,
            api_key=config.benchmark_chat_api_key,
            base_url=config.benchmark_chat_base_url,
        )

    @classmethod
    def create_benchmark_embedding_provider(cls) -> BaseLLMProvider:
        return cls.create(
            config.benchmark_embedding_provider,
            model_name=config.benchmark_embedding_model_name,
            api_key=config.benchmark_embedding_api_key or "local",
            base_url=config.benchmark_embedding_base_url,
        )

    @classmethod
    def create_benchmark_rerank_provider(cls) -> BaseLLMProvider:
        return cls.create(
            config.benchmark_rerank_provider,
            model_name=config.benchmark_rerank_model_name,
            api_key=config.benchmark_rerank_api_key,
            base_url=config.benchmark_rerank_base_url,
            timeout_seconds=config.benchmark_rerank_timeout_seconds,
        )
