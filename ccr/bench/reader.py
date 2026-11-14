"""Local Ollama reader: answers a question from retrieved context, never scores."""

from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.request
from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3.8:27b"
DEFAULT_HOST = "http://127.0.0.1:11434"
INSTRUCTION = (
    "Answer with a short phrase. "
    "If the context does not contain the answer, say unknown."
)

PROMPT_MARGIN = 256  # tokens reserved for instruction and question

Transport = Callable[[str, dict], dict]


class ReaderUnavailable(Exception):
    """Raised when the Ollama server cannot be reached."""


def _post_json(timeout_s: float) -> Transport:
    def post(url: str, payload: dict) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    return post


def _resolve_host(host: str | None) -> str:
    raw = host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
    return raw if "://" in raw else f"http://{raw}"


class OllamaReader:
    """Deterministic (temperature 0, no thinking) short-answer generator."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
        num_predict: int = 64,
        num_ctx: int = 16384,
        timeout_s: float = 600.0,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.host = _resolve_host(host).rstrip("/")
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self._post = transport or _post_json(timeout_s)
        self.empty_count = 0
        self.timeout_count = 0
        self.truncated_count = 0

    @property
    def tag(self) -> str:
        """Model tag recorded in cache keys and reports."""
        return self.model

    @property
    def context_limit_tokens(self) -> int:
        """Largest context the reader can take without Ollama silently truncating."""
        return self.num_ctx - self.num_predict - PROMPT_MARGIN

    def answer(self, question: str, context: str) -> str:
        """Return the model's short answer; empty string on empty output or timeout."""
        prompt = f"{INSTRUCTION}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }
        try:
            body = self._post(f"{self.host}/api/generate", payload)
        except (TimeoutError, socket.timeout):
            self.timeout_count += 1
            logger.warning("reader timed out for question %r", question[:60])
            return ""
        except (ConnectionError, urllib.error.URLError, OSError) as exc:
            if isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout)):
                self.timeout_count += 1
                return ""
            raise ReaderUnavailable(
                f"cannot reach Ollama at {self.host} ({exc}); "
                "start it with `ollama serve` and pull the model"
            ) from exc
        if int(body.get("prompt_eval_count") or 0) >= self.num_ctx - self.num_predict:
            self.truncated_count += 1
            logger.warning("prompt may have been truncated at num_ctx=%d", self.num_ctx)
        text = str(body.get("response", "")).strip()
        if not text:
            self.empty_count += 1
        return text
