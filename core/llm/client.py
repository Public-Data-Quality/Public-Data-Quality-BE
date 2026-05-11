from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config.constants import LLM_DEFAULT_API_URL, LLM_DEFAULT_MODEL, LLM_REQUEST_TIMEOUT_SECONDS

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _load_env() -> None:
    if load_dotenv is None:
        return
    package_env = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(package_env)
    load_dotenv()


@dataclass
class ChatLLMResponse:
    content: str


class ChatLLMClient:
    def __init__(
        self,
        model_name: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int = LLM_REQUEST_TIMEOUT_SECONDS,
    ):
        _load_env()
        self.model_name = model_name or os.getenv("LLM_MODEL") or LLM_DEFAULT_MODEL
        self.api_url = self._normalize_api_url(api_url or os.getenv("LLM_API_URL") or LLM_DEFAULT_API_URL)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
        self.timeout_seconds = timeout_seconds
        self.last_error = ""
        self.last_response_preview = ""

    def _normalize_api_url(self, value: str | None) -> str:
        normalized = (value or "").strip()
        return normalized or LLM_DEFAULT_API_URL

    def _configuration_error(self) -> str:
        if not self.model_name:
            return "LLM_MODEL missing"
        if not self.api_url:
            return "LLM_API_URL missing"
        if not self.api_key:
            return "OPENAI_API_KEY missing"
        return ""

    @property
    def enabled(self) -> bool:
        error = self._configuration_error()
        if error:
            self.last_error = error
            return False
        return True

    def invoke(self, prompt: str, *, system_prompt: str | None = None) -> ChatLLMResponse | None:
        if not self.enabled:
            return None

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "developer", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._invoke_messages(messages)
        if response is not None:
            return response

        return None

    def _build_payload(self, messages: list[dict[str, str]], *, json_response: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        if json_response:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post_chat_completion(self, payload: dict[str, Any]) -> ChatLLMResponse | None:
        if not self.enabled:
            return None

        data = json.dumps(payload).encode("utf-8")
        request = Request(
            self.api_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                error_body = ""
            self.last_error = f"HTTP {exc.code}: {error_body or exc.reason}"
            return None
        except URLError as exc:
            self.last_error = f"URL error: {exc.reason}"
            return None
        except TimeoutError:
            self.last_error = "request timeout"
            return None
        except json.JSONDecodeError:
            self.last_error = "invalid JSON response"
            return None

        content = self._extract_content(body)
        if not content:
            self.last_error = "empty response content"
            self.last_response_preview = ""
            return None
        self.last_response_preview = content[:300]
        self.last_error = ""
        return ChatLLMResponse(content=content)

    def _invoke_messages(self, messages: list[dict[str, str]]) -> ChatLLMResponse | None:
        return self._post_chat_completion(self._build_payload(messages))

    def invoke_json(self, prompt: str, *, system_prompt: str | None = None) -> ChatLLMResponse | None:
        if not self.enabled:
            return None

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "developer", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._invoke_messages_json(messages)
        if response is not None:
            return response

        return None

    def _invoke_messages_json(self, messages: list[dict[str, str]]) -> ChatLLMResponse | None:
        return self._post_chat_completion(self._build_payload(messages, json_response=True))

    @staticmethod
    def _extract_content(body: dict[str, Any]) -> str:
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content", "")
                    if isinstance(content, str):
                        return content
        return ""
