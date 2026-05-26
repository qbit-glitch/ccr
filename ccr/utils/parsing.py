"""Parsing utilities — code block extraction, JSON extraction, Anthropic API format helpers."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from ccr.core.types import CCRRequest, TokenUsage


# --- JSON extraction from LLM responses ---

def extract_json_from_llm(text: str) -> dict[str, Any] | None:
    """Extract and parse JSON from an LLM response.

    Handles common LLM output patterns in order of likelihood:
    1. Direct JSON string
    2. JSON inside ```json code blocks
    3. JSON found via balanced brace scanning

    Returns:
        Parsed dict/list, or None if no valid JSON found.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # 1. Direct parse (most common for well-prompted models)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract from ```json code blocks
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Also try bare ``` blocks
    bare_match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if bare_match:
        try:
            return json.loads(bare_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Balanced brace scanning (handles surrounding text/explanation)
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1

    return None


def extract_json_string(text: str) -> str:
    """Extract raw JSON string from LLM response (without parsing).

    For callers that need the raw string for further processing.
    Falls back to text.strip() if no JSON structure found.
    """
    if not text:
        return ""

    # Try code blocks first
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find JSON object or array
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if match:
        return match.group(0)

    return text.strip()


# --- Code block extraction (adapted from RLM utils/parsing.py) ---

def extract_code_blocks(text: str, language: str = "repl") -> list[str]:
    """Extract fenced code blocks with the given language tag."""
    pattern = rf"```{re.escape(language)}\s*\n(.*?)\n```"
    return re.findall(pattern, text, re.DOTALL)


def find_final_answer(text: str) -> str | None:
    """Find FINAL(...) or FINAL_VAR(...) in text."""
    # FINAL_VAR(variable_name) — returns the variable name
    match = re.search(r"^\s*FINAL_VAR\(\s*['\"]?(\w+)['\"]?\s*\)", text, re.MULTILINE)
    if match:
        return match.group(1)

    # FINAL(answer text)
    match = re.search(r"^\s*FINAL\((.*)\)\s*$", text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


# --- Anthropic API format helpers ---

def parse_anthropic_request(body: bytes) -> CCRRequest:
    """Parse an Anthropic Messages API request body into a CCRRequest."""
    data = json.loads(body)

    messages = data.get("messages", [])
    system_prompt = data.get("system")
    if isinstance(system_prompt, list):
        # system can be a list of content blocks
        system_prompt = "\n".join(
            b.get("text", "") for b in system_prompt if b.get("type") == "text"
        )

    return CCRRequest(
        original_messages=messages,
        system_prompt=system_prompt,
        model_requested=data.get("model", ""),
        max_tokens=data.get("max_tokens", 4096),
        raw_body=body,
    )


def format_anthropic_response(
    content: str,
    model: str,
    usage: TokenUsage,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Format a response to match Anthropic Messages API response schema."""
    return {
        "id": f"msg_{request_id or uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        },
    }


def format_anthropic_error(message: str, error_type: str = "invalid_request_error") -> dict:
    """Format an error response matching Anthropic API error schema."""
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def build_messages_with_context(
    original_messages: list[dict],
    context_pack_text: str | None = None,
    memory_context: str | None = None,
    playbook_text: str | None = None,
) -> list[dict]:
    """Inject context pack, memory, and playbook into message list before the last user message."""
    if not context_pack_text and not memory_context and not playbook_text:
        return original_messages

    messages = list(original_messages)
    injected_parts = []
    if playbook_text:
        injected_parts.append(f"<ace_playbook>\n{playbook_text}\n</ace_playbook>")
    if memory_context:
        injected_parts.append(f"<session_memory>\n{memory_context}\n</session_memory>")
    if context_pack_text:
        injected_parts.append(f"<context_pack>\n{context_pack_text}\n</context_pack>")
    injection = "\n\n".join(injected_parts)

    # Find last user message and prepend context to it
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            content = messages[i].get("content", "")
            if isinstance(content, str):
                messages[i] = {
                    **messages[i],
                    "content": f"{injection}\n\n{content}",
                }
            elif isinstance(content, list):
                # Prepend as a text block
                messages[i] = {
                    **messages[i],
                    "content": [{"type": "text", "text": injection}] + content,
                }
            break

    return messages
