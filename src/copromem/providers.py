"""Chat providers supported by the generic agent interface."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Callable


class OpenRouterProvider:
    """Non-streaming OpenRouter chat-completions tool client."""

    DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str, *, api_key: str | None = None,
                 api_key_env: str = "OPENROUTER_API_KEY", url: str = DEFAULT_URL,
                 timeout: float = 60, temperature: float = 0, max_tokens: int = 1024,
                 opener: Callable[..., Any] | None = None):
        if not model or timeout <= 0 or max_tokens < 1:
            raise ValueError("model, positive timeout and positive max_tokens are required")
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"missing API key in {api_key_env}")
        if not url.startswith("https://"):
            raise ValueError("OpenRouter endpoint must use HTTPS")
        self.url = url
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.opener = opener or urllib.request.urlopen

    def complete(self, messages: Sequence[Mapping[str, Any]],
                 tools: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        body = {"model": self.model, "messages": list(messages), "tools": list(tools),
                "tool_choice": "auto", "parallel_tool_calls": False, "stream": False,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        request = urllib.request.Request(
            self.url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST")
        try:
            with self.opener(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"OpenRouter request failed with HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError("OpenRouter connection failed") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("OpenRouter response is not an object")
        if payload.get("error"):
            raise RuntimeError("OpenRouter returned an API error")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise ValueError("OpenRouter response has no completion choice")
        message = choices[0].get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            raise ValueError("OpenRouter response has no assistant message")
        metadata = {key: payload[key] for key in ("id", "model", "usage") if key in payload}
        return {**message, "_provider": metadata}


class ScriptedProvider:
    """Deterministic provider for local agent smoke tests."""

    def __init__(self, responses: Sequence[Mapping[str, Any]]):
        self.responses = [dict(item) for item in responses]
        self.requests: list[dict[str, Any]] = []

    def complete(self, messages: Sequence[Mapping[str, Any]],
                 tools: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        self.requests.append({"messages": list(messages), "tools": list(tools)})
        if not self.responses:
            raise RuntimeError("scripted provider ran out of responses")
        return self.responses.pop(0)
