from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from finagent.packages.common.config import Settings
from finagent.packages.common.errors import ExternalDependencyError


class LLMGenerationError(ExternalDependencyError):
    """Raised when a configured LLM backend cannot return usable text."""


class OllamaClient:
    provider = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        num_predict: int = 768,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.num_predict = num_predict

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": self.num_predict,
            },
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise LLMGenerationError(f"Ollama 服务返回 HTTP {exc.code}：{self._read_error(exc)}") from exc
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMGenerationError(f"Ollama 请求失败：{exc}") from exc

        if data.get("error"):
            raise LLMGenerationError(f"Ollama 生成失败：{data['error']}")
        message = data.get("message") or {}
        content = (message.get("content") or data.get("response") or "").strip()
        if not content:
            raise LLMGenerationError("Ollama 返回了空响应")
        return content

    @staticmethod
    def _read_error(exc: HTTPError) -> str:
        try:
            return exc.read().decode("utf-8", errors="replace")[:500]
        except OSError:
            return exc.reason


class OpenAICompatibleChatClient:
    provider = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_seconds: float = 120.0,
        max_tokens: int = 768,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self._chat_completions_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise LLMGenerationError(f"LLM 网关返回 HTTP {exc.code}：{OllamaClient._read_error(exc)}") from exc
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMGenerationError(f"LLM 网关请求失败：{exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMGenerationError("LLM 网关返回了无效的对话补全结果") from exc
        if not content:
            raise LLMGenerationError("LLM 网关返回了空响应")
        return content

    def _chat_completions_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"


def build_llm_client(settings: Settings) -> OpenAICompatibleChatClient | OllamaClient | None:
    if settings.runtime_mode == "test":
        return None
    if settings.llm_provider == "openai_compatible":
        if not settings.llm_gateway_base_url:
            raise LLMGenerationError("使用 LLM 网关生成回答时必须配置 FINAGENT_LLM_GATEWAY_BASE_URL")
        return OpenAICompatibleChatClient(
            base_url=settings.llm_gateway_base_url,
            api_key=settings.llm_gateway_api_key,
            model=settings.llm_gateway_model,
            timeout_seconds=settings.llm_gateway_timeout_seconds,
            max_tokens=settings.llm_gateway_max_tokens,
            temperature=settings.llm_gateway_temperature,
            top_p=settings.llm_gateway_top_p,
        )
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        num_predict=settings.ollama_num_predict,
    )
