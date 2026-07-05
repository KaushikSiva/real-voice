from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def stream_chat(
        self,
        model: str,
        user_text: str,
        system_prompt: str,
        temperature: float = 0.4,
    ) -> Iterator[str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        yield from self.stream_messages(model=model, messages=messages, temperature=temperature)

    def stream_messages(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
    ) -> Iterator[str]:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests is not installed. Run ./scripts/setup_macos.sh") from exc

        payload = {
            "model": model,
            "stream": True,
            "messages": messages,
            "options": {"temperature": temperature},
        }
        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                event = json.loads(raw_line.decode("utf-8"))
                if "error" in event:
                    raise RuntimeError(event["error"])
                content = event.get("message", {}).get("content", "")
                if content:
                    yield content
                if event.get("done"):
                    break

    def chat(
        self,
        model: str,
        user_text: str,
        system_prompt: str,
        temperature: float = 0.4,
    ) -> str:
        return "".join(self.stream_chat(model, user_text, system_prompt, temperature)).strip()

    def chat_messages(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
    ) -> str:
        return "".join(self.stream_messages(model=model, messages=messages, temperature=temperature)).strip()

    def tags(self) -> dict[str, Any]:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests is not installed. Run ./scripts/setup_macos.sh") from exc

        response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        return dict(response.json())
