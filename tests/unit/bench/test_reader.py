"""Tests for the local Ollama reader (fake transport, no server needed)."""

from __future__ import annotations

import pytest

from ccr.bench.reader import OllamaReader, ReaderUnavailable


class FakeTransport:
    def __init__(self, response: dict | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, payload: dict) -> dict:
        self.calls.append((url, payload))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_request_pins_deterministic_non_thinking_generation() -> None:
    t = FakeTransport({"response": " Paris \n"})
    reader = OllamaReader(host="http://h:1", transport=t)
    assert reader.answer("Capital?", "France's capital is Paris.") == "Paris"
    url, body = t.calls[0]
    assert url == "http://h:1/api/generate"
    assert body["model"] == "qwen3.8:27b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["temperature"] == 0
    assert body["options"]["num_predict"] == 64


def test_prompt_contains_instruction_context_and_question() -> None:
    t = FakeTransport({"response": "x"})
    OllamaReader(host="http://h:1", transport=t).answer("Q-TEXT", "CTX-TEXT")
    prompt = t.calls[0][1]["prompt"]
    assert "Answer with a short phrase. If the context does not contain the answer, say unknown." in prompt
    assert "CTX-TEXT" in prompt and "Q-TEXT" in prompt
    assert prompt.index("CTX-TEXT") < prompt.index("Q-TEXT")


def test_empty_response_returns_empty_string() -> None:
    reader = OllamaReader(host="http://h:1", transport=FakeTransport({"response": ""}))
    assert reader.answer("q", "c") == ""
    assert reader.empty_count == 1


def test_connection_error_raises_reader_unavailable_naming_host() -> None:
    reader = OllamaReader(
        host="http://down:11434", transport=FakeTransport(ConnectionError("refused"))
    )
    with pytest.raises(ReaderUnavailable, match="down:11434"):
        reader.answer("q", "c")


def test_timeout_returns_empty_and_is_counted() -> None:
    reader = OllamaReader(host="http://h:1", transport=FakeTransport(TimeoutError("slow")))
    assert reader.answer("q", "c") == ""
    assert reader.timeout_count == 1


def test_host_defaults_from_env_and_gains_scheme(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "localhost:9999")
    t = FakeTransport({"response": "x"})
    OllamaReader(transport=t).answer("q", "c")
    assert t.calls[0][0] == "http://localhost:9999/api/generate"
    monkeypatch.delenv("OLLAMA_HOST")
    t2 = FakeTransport({"response": "x"})
    OllamaReader(transport=t2).answer("q", "c")
    assert t2.calls[0][0] == "http://127.0.0.1:11434/api/generate"


def test_tag_is_the_model_name() -> None:
    assert OllamaReader(host="http://h:1", transport=FakeTransport({})).tag == "qwen3.8:27b"
