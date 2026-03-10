"""RLM Orchestrator — the completion loop that drives iterative code execution.

Adapted from vendor/rlm RLM class. Key differences:
- No TCP sockets — uses CCR's model clients directly
- Integrated with CCR's context packer and repo index
- Simpler resource management (no LMHandler)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ccr.core.types import (
    REPLResult,
    RLMConfig,
    RLMResult,
    TokenUsage,
)
from ccr.rlm.repl import CCRRepl
from ccr.utils.parsing import extract_code_blocks, find_final_answer
from ccr.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


def _make_metadata_preview(text: str, max_preview: int = 120) -> str:
    """Create a metadata preview string: length + short prefix."""
    preview = text[:max_preview].replace("\n", "\\n")
    if len(text) > max_preview:
        preview += "..."
    return f'{len(text)} chars, preview: "{preview}"'


def format_execution_result(result: REPLResult, max_output: int = 2000) -> str:
    """Format a REPL execution result as metadata-only for message history.

    Per the RLM paper (Algorithm 1): "Only (constant-size) metadata about stdout,
    like a short prefix and length, is appended to M's history for the next iteration.
    This forces M to rely on variables and sub-calls to manage long strings instead
    of polluting its window."
    """
    parts = []
    if result.stdout:
        parts.append(f"[stdout: {_make_metadata_preview(result.stdout)}]")
    if result.stderr:
        # Errors get more detail to help debugging (but still capped)
        err = result.stderr[:500]
        parts.append(f"[stderr: {err}]")
    if result.final_answer is not None:
        parts.append(f"[FINAL_VAR returned]: {result.final_answer[:500]}")
    if result.locals_snapshot:
        vars_str = ", ".join(f"{k}: {v[:60]}" for k, v in list(result.locals_snapshot.items())[:10])
        parts.append(f"[variables]: {vars_str}")
    return "\n".join(parts) if parts else "(no output)"


def format_iteration(response: str, code_blocks: list[str], results: list[REPLResult]) -> list[dict]:
    """Format one iteration into message history entries."""
    messages = [{"role": "assistant", "content": response}]

    if code_blocks:
        outputs = []
        for i, (code, result) in enumerate(zip(code_blocks, results)):
            outputs.append(f"=== Code Block {i+1} ===\n```python\n{code}\n```\n{format_execution_result(result)}")
        messages.append({
            "role": "user",
            "content": "REPL execution results:\n" + "\n\n".join(outputs) + "\n\nContinue.",
        })

    return messages


class RLMError(Exception):
    """Base RLM error with partial answer."""
    def __init__(self, message: str, partial_answer: str = ""):
        super().__init__(message)
        self.partial_answer = partial_answer


class BudgetExceededError(RLMError):
    pass


class TimeoutExceededError(RLMError):
    pass


class ErrorThresholdExceededError(RLMError):
    pass


class CCRRlm:
    """Recursive Language Model orchestrator for CCR.

    The main completion loop:
        1. Send prompt + history to sub-model
        2. Parse response for ```repl``` code blocks
        3. Execute each block in the sandboxed REPL
        4. Check for FINAL_VAR/FINAL termination
        5. If not done, append results to history and repeat
        6. On max iterations, ask model for best answer from trajectory

    Supports recursive sub-calls: rlm_query() in the REPL spawns a child
    CCRRlm at depth+1 with its own REPL instance.
    """

    def __init__(
        self,
        sub_client: Any,
        config: RLMConfig | None = None,
        repo_index: Any = None,
        system_prompt: str | None = None,
        depth: int = 0,
        custom_tools: dict[str, Any] | None = None,
        playbook_text: str | None = None,
    ):
        self.sub_client = sub_client
        self.config = config or RLMConfig()
        self.repo_index = repo_index
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.depth = depth
        self.custom_tools = custom_tools or {}
        self.playbook_text = playbook_text

        # Tracking state
        self._consecutive_errors = 0
        self._best_partial_answer: str | None = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._trajectory: list[dict[str, Any]] = []
        self._compaction_history: list[str] = []
        self._loop_start_time: float = 0.0

    def _default_system_prompt(self) -> str:
        from ccr.context.prompts import RLM_SYSTEM_PROMPT
        return RLM_SYSTEM_PROMPT

    def completion(self, prompt: str, root_prompt: str | None = None) -> RLMResult:
        """Run the RLM completion loop on a prompt.

        Per the RLM paper (Algorithm 1): The prompt P is loaded as a variable
        inside the REPL environment, and only metadata about P (length, prefix)
        is given to the LLM. This forces the model to manipulate the prompt
        programmatically rather than copying it into its context window.

        Args:
            prompt: The task/context to process.
            root_prompt: Optional short version of the original question.

        Returns:
            RLMResult with the final answer and metadata.
        """
        time_start = time.perf_counter()
        self._loop_start_time = time_start
        self._consecutive_errors = 0
        self._best_partial_answer = None
        self._trajectory = []
        self._compaction_history = []

        # At max depth, fall back to plain LLM call
        if self.depth >= self.config.max_depth:
            return self._fallback_completion(prompt, time_start)

        # Create REPL with tools
        repl = CCRRepl(
            sub_client=self.sub_client,
            repo_index=self.repo_index,
            subcall_fn=self._subcall,
            custom_tools=self.custom_tools,
        )

        # RLM paper key insight: load prompt as a REPL variable, not in LLM context
        repl.add_context(prompt, name="task_prompt")

        # ACE: inject playbook as REPL variable so RLM can reference strategies
        if self.playbook_text:
            repl.add_context(self.playbook_text, name="playbook")

        try:
            return self._run_loop(prompt, root_prompt, repl, time_start)
        except RLMError as e:
            partial = e.partial_answer or self._best_partial_answer or f"Error: {e}"
            return RLMResult(
                response=partial,
                iterations_used=len(self._trajectory),
                depth=self.depth,
                execution_time=time.perf_counter() - time_start,
                usage=TokenUsage(
                    input_tokens=self._total_input_tokens,
                    output_tokens=self._total_output_tokens,
                    total_tokens=self._total_input_tokens + self._total_output_tokens,
                ),
                trajectory=self._trajectory,
                final_answer_source="error",
            )
        finally:
            repl.cleanup()

    def _run_loop(
        self,
        prompt: str,
        root_prompt: str | None,
        repl: CCRRepl,
        time_start: float,
    ) -> RLMResult:
        """The main generate→execute→check loop."""
        message_history = self._setup_messages(prompt)
        compaction_count = 0
        iterations_completed = 0

        # Make compaction history accessible to REPL code
        repl.locals["history"] = self._compaction_history

        for i in range(self.config.max_iterations):
            # Check timeout
            elapsed = time.perf_counter() - time_start
            if elapsed > self.config.max_timeout_seconds:
                logger.info(f"RLM timeout after {elapsed:.1f}s at iteration {i}")
                raise TimeoutExceededError(
                    f"RLM timeout after {elapsed:.1f}s",
                    self._best_partial_answer or "",
                )

            # Check budget
            if self.config.max_budget_usd is not None:
                from ccr.utils.costs import calculate_cost
                cost = calculate_cost(
                    "gpt-oss-20b",
                    self._total_input_tokens,
                    self._total_output_tokens,
                ) or 0.0
                if cost > self.config.max_budget_usd:
                    logger.info(f"RLM budget exceeded: ${cost:.4f} > ${self.config.max_budget_usd}")
                    raise BudgetExceededError(
                        f"Budget exceeded: ${cost:.4f}",
                        self._best_partial_answer or "",
                    )

            # Check token limit
            if self.config.max_total_tokens > 0:
                total = self._total_input_tokens + self._total_output_tokens
                if total > self.config.max_total_tokens:
                    raise BudgetExceededError(
                        f"Token limit exceeded: {total}",
                        self._best_partial_answer or "",
                    )

            # Compaction: summarize history if approaching context limit
            compaction_threshold = self.config.compaction_threshold
            if compaction_threshold == 0:
                # Auto: 80% of estimated model context (default ~32K tokens)
                compaction_threshold = int(32000 * 0.80)

            history_tokens = count_tokens(str(message_history))
            if history_tokens > compaction_threshold:
                compaction_count += 1
                message_history = self._compact_history(message_history, compaction_count)

            # Build iteration prompt
            iter_suffix = self._build_iteration_suffix(i, root_prompt)
            current_messages = message_history + [{"role": "user", "content": iter_suffix}]

            # Call the sub-model
            try:
                response_text = self.sub_client.completion(current_messages)
                usage = self.sub_client.get_last_usage()
                if usage:
                    self._total_input_tokens += usage.input_tokens
                    self._total_output_tokens += usage.output_tokens
            except Exception as e:
                logger.error(f"RLM iteration {i} LLM call failed: {e}")
                self._consecutive_errors += 1
                if self._consecutive_errors >= self.config.max_consecutive_errors:
                    raise ErrorThresholdExceededError(
                        f"Error threshold: {self._consecutive_errors}",
                        self._best_partial_answer or "",
                    )
                continue

            iterations_completed = i + 1

            # Parse code blocks
            code_blocks = extract_code_blocks(response_text, "repl")
            if not code_blocks:
                code_blocks = extract_code_blocks(response_text, "python")

            # Execute code blocks in REPL
            results: list[REPLResult] = []
            final_answer = None
            for code in code_blocks:
                result = repl.execute_code(code)
                results.append(result)

                if result.error:
                    self._consecutive_errors += 1
                else:
                    self._consecutive_errors = 0

                if result.final_answer is not None:
                    final_answer = result.final_answer
                    break

            # Check for FINAL/FINAL_VAR in the text itself
            if final_answer is None:
                text_final = find_final_answer(response_text)
                if text_final is not None:
                    # It's a variable name — look it up in REPL
                    if text_final in repl.locals:
                        val = repl.locals[text_final]
                        import json as _json
                        final_answer = _json.dumps(val) if isinstance(val, (dict, list)) else str(val)
                    else:
                        final_answer = text_final

            # Track trajectory
            self._trajectory.append({
                "iteration": i,
                "code_blocks": len(code_blocks),
                "has_errors": any(r.error for r in results),
                "has_final": final_answer is not None,
            })

            # Store best partial
            if response_text and response_text.strip():
                self._best_partial_answer = response_text

            # Check error threshold
            if self._consecutive_errors >= self.config.max_consecutive_errors:
                logger.info(f"RLM error threshold reached: {self._consecutive_errors} consecutive errors")
                raise ErrorThresholdExceededError(
                    f"Error threshold: {self._consecutive_errors}",
                    self._best_partial_answer or "",
                )

            # Return if we have a final answer
            if final_answer is not None:
                return RLMResult(
                    response=final_answer,
                    iterations_used=i + 1,
                    depth=self.depth,
                    execution_time=time.perf_counter() - time_start,
                    usage=TokenUsage(
                        input_tokens=self._total_input_tokens,
                        output_tokens=self._total_output_tokens,
                        total_tokens=self._total_input_tokens + self._total_output_tokens,
                    ),
                    trajectory=self._trajectory,
                    final_answer_source="FINAL_VAR",
                )

            # Append iteration results to history
            new_messages = format_iteration(response_text, code_blocks, results)
            message_history.extend(new_messages)

        # Exhausted iterations — ask for best answer
        return self._default_answer(message_history, time_start, iterations_completed)

    def _setup_messages(self, prompt: str) -> list[dict[str, Any]]:
        """Build initial message history with metadata-only prompt.

        Per the RLM paper (Algorithm 1): "hist ← [Metadata(state)]"
        The LLM receives only metadata about the prompt (length, prefix,
        access instructions), NOT the full prompt text. The full prompt
        is available as `task_prompt` variable in the REPL.
        """
        prompt_metadata = self._build_prompt_metadata(prompt)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt_metadata},
        ]

    def _build_prompt_metadata(self, prompt: str) -> str:
        """Build metadata-only representation of the prompt for the LLM.

        Instead of sending the full prompt (which wastes context window),
        we send: length, token estimate, short prefix, and how to access it.
        """
        token_est = count_tokens(prompt)
        prefix = prompt[:300].replace("\n", "\\n")
        if len(prompt) > 300:
            prefix += "..."

        parts = [
            "## Task",
            f"A task prompt has been loaded into your REPL as `task_prompt` (string variable).",
            f"- Length: {len(prompt)} chars (~{token_est} tokens)",
            f'- Preview: "{prefix}"',
            "",
            "## Available REPL Variables",
            "- `task_prompt` — the full task/prompt text (use slicing, regex, etc. to examine)",
        ]

        # Add context info if repo index is loaded
        if self.repo_index is not None:
            if hasattr(self.repo_index, "files"):
                parts.append(f"- `context` — repo metadata dict ({len(self.repo_index.files)} files indexed)")
            else:
                parts.append("- `context` — repo metadata dict")

        # Add playbook info if available
        if self.playbook_text:
            parts.append(f"- `playbook` — ACE playbook with strategies and insights ({len(self.playbook_text)} chars)")

        parts.extend([
            "",
            "## Instructions",
            "Read `task_prompt` in the REPL to understand the task, then solve it.",
            "Use `FINAL_VAR('result_variable')` when done.",
        ])

        return "\n".join(parts)

    def _build_iteration_suffix(self, iteration: int, root_prompt: str | None) -> str:
        """Build the user message suffix for each iteration."""
        parts = [f"[Iteration {iteration + 1}/{self.config.max_iterations}]"]
        if root_prompt and iteration == 0:
            parts.append(f"Original question: {root_prompt}")
        if iteration > 0:
            parts.append("Continue working. Use FINAL_VAR('variable_name') when done.")
        else:
            parts.append(
                "Write ```repl``` code blocks to examine the context and solve the task. "
                "Use FINAL_VAR('result_variable') when you have the answer."
            )
        return " ".join(parts)

    def _compact_history(
        self,
        message_history: list[dict[str, Any]],
        compaction_count: int,
    ) -> list[dict[str, Any]]:
        """Summarize the conversation history to stay within context limits."""
        summary_messages = message_history + [{
            "role": "user",
            "content": (
                "Summarize your progress so far. Include:\n"
                "1. Steps completed and remaining\n"
                "2. Key intermediate results (preserve values exactly)\n"
                "3. Next action\n"
                "Be concise (1-3 paragraphs)."
            ),
        }]
        try:
            summary = self.sub_client.completion(summary_messages)
        except Exception:
            summary = self._best_partial_answer or "Progress summary unavailable."

        # Store history in trajectory for REPL access (per RLM paper)
        self._compaction_history.append(summary)

        return message_history[:2] + [
            {"role": "assistant", "content": summary},
            {
                "role": "user",
                "content": (
                    f"History compacted ({compaction_count}x). "
                    "Continue from above. Use SHOW_VARS() to check state. "
                    "Do NOT repeat completed work."
                ),
            },
        ]

    def _default_answer(
        self,
        message_history: list[dict[str, Any]],
        time_start: float,
        iterations_used: int | None = None,
    ) -> RLMResult:
        """When iterations are exhausted, generate a final answer from trajectory."""
        try:
            final_prompt = message_history + [{
                "role": "user",
                "content": "Iterations exhausted. Provide your best final answer based on all work so far.",
            }]
            answer = self.sub_client.completion(final_prompt)
        except Exception:
            answer = self._best_partial_answer or "Unable to generate answer."

        return RLMResult(
            response=answer,
            iterations_used=iterations_used if iterations_used is not None else self.config.max_iterations,
            depth=self.depth,
            execution_time=time.perf_counter() - time_start,
            usage=TokenUsage(
                input_tokens=self._total_input_tokens,
                output_tokens=self._total_output_tokens,
                total_tokens=self._total_input_tokens + self._total_output_tokens,
            ),
            trajectory=self._trajectory,
            final_answer_source="default",
        )

    def _fallback_completion(self, prompt: str, time_start: float) -> RLMResult:
        """At max depth, just do a plain LLM call."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.sub_client.completion(messages)
        except Exception as e:
            response = f"Error: {e}"

        return RLMResult(
            response=response,
            iterations_used=1,
            depth=self.depth,
            execution_time=time.perf_counter() - time_start,
            final_answer_source="fallback",
        )

    def _subcall(self, prompt: str, model: str | None = None) -> RLMResult:
        """Spawn a child RLM for recursive sub-calls."""
        # Calculate remaining timeout from loop start
        elapsed = time.perf_counter() - self._loop_start_time
        remaining_timeout = max(10.0, self.config.max_timeout_seconds - elapsed)

        # Calculate remaining budget
        remaining_budget = self.config.max_budget_usd
        if remaining_budget is not None:
            from ccr.utils.costs import calculate_cost
            spent = calculate_cost(
                "gpt-oss-20b",
                self._total_input_tokens,
                self._total_output_tokens,
            ) or 0.0
            remaining_budget = max(0.001, remaining_budget - spent)

        child = CCRRlm(
            sub_client=self.sub_client,
            config=RLMConfig(
                max_depth=self.config.max_depth,
                max_iterations=max(5, self.config.max_iterations // 2),
                max_timeout_seconds=remaining_timeout,
                max_budget_usd=remaining_budget,
                max_consecutive_errors=self.config.max_consecutive_errors,
                max_total_tokens=self.config.max_total_tokens,
            ),
            repo_index=self.repo_index,
            system_prompt=self.system_prompt,
            depth=self.depth + 1,
            custom_tools=self.custom_tools,
            playbook_text=self.playbook_text,
        )
        return child.completion(prompt)
