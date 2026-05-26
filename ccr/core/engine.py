"""CCR Engine — the main orchestrator.

Owns memory, router, packer, model clients.
Processes each request through the full pipeline.
"""

from __future__ import annotations

import logging
import os
import time

from ccr.context.indexer import RepoIndex
from ccr.context.packer import ContextPacker
from ccr.core.exceptions import (
    ModelAuthError,
    ModelError,
    PlaybookError,
)
from ccr.core.memory import MemoryManager
from ccr.core.router import TaskRouter
from ccr.core.hooks import HookManager, create_default_hooks
from ccr.core.types import (
    CCREngineConfig,
    CCRRequest,
    CCRResponse,
    HookEvent,
    RouteDecision,
    TokenUsage,
)
from ccr.ace.engine import ACEEngine
from ccr.models.anthropic_client import ClaudeClient
from ccr.models.openai_compat import OpenAICompatClient
from ccr.rlm.orchestrator import CCRRlm
from ccr.utils.costs import CostTracker
from ccr.utils.parsing import build_messages_with_context

logger = logging.getLogger(__name__)


class CCREngine:
    """Orchestrates the full CCR pipeline for a single project."""

    def __init__(self, project_root: str, config: CCREngineConfig):
        self.project_root = project_root
        self.config = config
        self.memory = MemoryManager(project_root, config.memory)
        self.cost_tracker = CostTracker()
        self._initialized = False

        # Lazy-initialized
        self._claude_client: ClaudeClient | None = None
        self._sub_client: OpenAICompatClient | None = None
        self._repo_index: RepoIndex | None = None
        self._packer: ContextPacker | None = None
        self._router: TaskRouter | None = None
        self._hooks: HookManager | None = None
        self._ace_engine: ACEEngine | None = None

    def initialize(self) -> None:
        """Idempotent startup. Call before first process().

        Raises:
            ConfigError: If configuration is invalid.
        """
        if self._initialized:
            return

        logger.info(f"Initializing CCR for: {self.project_root}")

        # 0. Validate configuration
        warnings = self.config.validate()
        for w in warnings:
            logger.warning(f"Config warning: {w}")

        # 1. Ensure .ccr/ structure
        created = self.memory.ensure_structure()
        if created:
            logger.info("Created .ccr/ directory")

        # 2. Build/load repo index
        self._build_index()

        # 3. Initialize model clients
        self._claude_client = ClaudeClient(
            api_key=self.config.anthropic_api_key,
            model_name=self.config.claude_model,
        )

        self._sub_client = OpenAICompatClient(
            model_name=self.config.sub_model,
            base_url=self.config.sub_model_base_url,
            api_key=self.config.sub_model_api_key,
        )

        # 3b. Connectivity checks (warn, don't block)
        try:
            sub_ok, sub_detail = self._sub_client.check_connectivity()
            if sub_ok:
                logger.info(f"Sub-model connectivity: {sub_detail}")
            else:
                logger.warning(
                    f"Sub-model at {self.config.sub_model_base_url} is not reachable: {sub_detail}. "
                    f"Requests routed to sub-model will fail."
                )
        except Exception as e:
            logger.warning(f"Sub-model connectivity check failed: {e}")

        # 4. Context packer
        self._packer = ContextPacker(
            repo_index=self._repo_index,
            sub_client=self._sub_client,
            token_budget=self.config.pack_token_budget,
            max_search_candidates=self.config.max_search_candidates,
            min_relevance_score=self.config.min_relevance_score,
        )

        # 5. Router
        self._router = TaskRouter(
            classifier_client=self._sub_client,
            context_packer=self._packer,
            memory=self.memory,
            config=self.config.router,
        )

        # 6. Hook system
        self._hooks = create_default_hooks(self.memory)

        # 7. ACE engine (online adaptation)
        if self.config.ace.enabled:
            playbook_path = self.config.ace.playbook_path or os.path.join(
                self.project_root, ".ccr", "playbook.txt"
            )
            self._ace_engine = ACEEngine(
                sub_client=self._sub_client,
                config=self.config.ace,
                playbook_path=playbook_path,
            )
            logger.info("ACE engine initialized")

        self._initialized = True

        # Fire SessionStart hook
        self._hooks.fire(HookEvent.SESSION_START, {"project": self.project_root})
        logger.info("CCR engine initialized")

    def process(self, request: CCRRequest) -> CCRResponse:
        """Process a request through the full CCR pipeline.

        1. Classify task complexity
        2. Route to appropriate model
        3. Build context pack if needed
        4. Call model with enriched context
        5. Auto-commit if warranted
        6. Return response
        """
        if not self._initialized:
            self.initialize()

        start = time.time()

        # 1. Classify
        classification = self._router.classify(request)
        logger.info(
            f"[{request.request_id}] Classified as {classification.tier.value} "
            f"(confidence={classification.confidence:.2f}, reason={classification.reasoning})"
        )

        # 2. Route
        decision = self._router.route(request, classification)
        logger.info(f"[{request.request_id}] Routing to: {decision.target}")

        # 3. Build enriched messages
        context_text = None
        memory_text = None

        if decision.context_pack:
            context_text = decision.context_pack.to_prompt_text()
            classification.packed_tokens = decision.context_pack.total_tokens

        if decision.memory_context_level > 0:
            memory_text = self.memory.get_context(level=decision.memory_context_level)

        # Inject ACE playbook into context if available
        playbook_text = None
        if self._ace_engine:
            playbook_text = self._ace_engine.get_playbook_for_prompt()

        enriched_messages = build_messages_with_context(
            request.original_messages,
            context_pack_text=context_text,
            memory_context=memory_text,
            playbook_text=playbook_text,
        )

        # Fire PreCompact hook if message context is getting large
        total_content_len = sum(len(str(m.get("content", ""))) for m in enriched_messages)
        if total_content_len > self.config.pre_compact_char_threshold and self._hooks:
            self._hooks.fire(HookEvent.PRE_COMPACT, {
                "content_length": total_content_len,
                "branch": self.memory.get_active_branch(),
            })

        # 4. Call the target model
        response_text, usage = self._call_model(
            decision, enriched_messages, request
        )

        # 5. Track costs
        self.cost_tracker.record(
            model=self._get_model_for_target(decision.target),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

        # 6. Auto-commit if warranted
        if decision.should_autocommit and self.config.memory.proactive_commits:
            self._auto_commit(request, response_text, decision)

        # 7. ACE online adaptation — learn from this task execution
        if self._ace_engine:
            try:
                ace_result = self._ace_engine.adapt_online(
                    task=request.last_user_message[:500],
                    execution_trace=response_text[:3000],
                    execution_result=response_text[:500],
                    was_successful=True,  # assume success unless env feedback says otherwise
                    context=context_text or "",
                )
                if self._hooks:
                    self._hooks.fire(HookEvent.POST_TASK_COMPLETE, {
                        "task": request.last_user_message[:100],
                        "bullets_added": ace_result.bullets_added,
                        "bullets_pruned": ace_result.bullets_pruned,
                        "playbook_size": ace_result.playbook_size,
                    })
            except (PlaybookError, ModelError) as e:
                logger.warning(f"ACE adaptation failed: {e}")
            except Exception as e:
                logger.warning(f"ACE adaptation failed (unexpected): {e}")

        # 8. Log OTA
        elapsed = time.time() - start
        self.memory.log_ota(
            f"ccr-{decision.target}",
            f"req={request.request_id} tier={classification.tier.value} time={elapsed:.1f}s",
        )

        return CCRResponse(
            content=response_text,
            model_used=self._get_model_for_target(decision.target),
            classification=classification,
            context_pack=decision.context_pack,
            usage=usage,
            request_id=request.request_id,
            routed_to=decision.target,
        )

    def get_usage_report(self) -> dict:
        """Get cost tracking report with savings calculation."""
        return self.cost_tracker.get_report()

    def shutdown(self) -> None:
        """Clean shutdown — flush logs, save index, fire Stop hook."""
        if self._repo_index:
            try:
                self.memory.save_index(self._repo_index.to_json())
            except Exception as e:
                logger.warning(f"Failed to save index: {e}")

        # Update metadata with final file tree
        if self._repo_index:
            try:
                file_list = [f["path"] for f in self._repo_index.files]
                self.memory.update_metadata_file_tree(file_list)
            except Exception:
                pass

        # Fire Stop hook
        if self._hooks:
            self._hooks.fire(HookEvent.STOP, {"project": self.project_root})
        logger.info("CCR engine shut down")

    # --- Internal ---

    def _build_index(self) -> None:
        """Build or load repo index."""
        cached = self.memory.load_index()
        if cached:
            idx = RepoIndex.from_cache(self.project_root, cached)
            if idx and idx.files:
                self._repo_index = idx
                logger.info(f"Loaded index from cache: {len(idx.files)} files")
                # Reload contents since cache doesn't store them
                self._repo_index = RepoIndex.build(
                    self.project_root,
                    max_file_size_kb=self.config.memory.index_max_file_size_kb,
                    extensions=set(self.config.index_extensions),
                )
                return

        self._repo_index = RepoIndex.build(
            self.project_root,
            max_file_size_kb=self.config.memory.index_max_file_size_kb,
            extensions=set(self.config.index_extensions),
        )
        logger.info(f"Built index: {len(self._repo_index.files)} files")
        try:
            self.memory.save_index(self._repo_index.to_json())
        except Exception as e:
            logger.warning(f"Failed to cache index: {e}")

    def _call_model(
        self,
        decision: RouteDecision,
        messages: list[dict],
        request: CCRRequest,
    ) -> tuple[str, TokenUsage]:
        """Call the appropriate model based on routing decision."""
        target = decision.target

        # RLM path: use iterative code execution for COMPLEX tasks.
        # RLM builds a richer context pack via REPL, then we still send to Claude.
        if decision.use_rlm and self._repo_index and self._sub_client:
            try:
                rlm_context = self._run_rlm_for_context(request, decision)
                if rlm_context:
                    # Inject RLM-generated context into messages for Claude
                    from ccr.utils.parsing import build_messages_with_context
                    messages = build_messages_with_context(
                        request.original_messages,
                        context_pack_text=rlm_context,
                    )
            except ModelAuthError:
                raise  # Credential errors must propagate
            except (ModelError, Exception) as e:
                logger.warning(f"RLM execution failed, falling back to direct: {e}")

        if target in ("qwen_direct", "qwen_with_context"):
            client = self._sub_client
        else:
            client = self._claude_client

        try:
            text = client.completion(messages)
            usage = client.get_last_usage()
            return text, usage
        except ModelAuthError:
            raise  # Non-recoverable — don't fallback with bad creds
        except ModelError as e:
            logger.error(f"Model call failed ({target}): {e}")
            # Fallback: try Claude directly for sub-model failures
            if client != self._claude_client:
                try:
                    text = self._claude_client.completion(request.original_messages)
                    usage = self._claude_client.get_last_usage()
                    return text, usage
                except ModelError as e2:
                    logger.error(f"Claude fallback also failed: {e2}")
            raise

    def _run_rlm_for_context(
        self,
        request: CCRRequest,
        decision: RouteDecision,
    ) -> str | None:
        """Run the RLM orchestrator to build enriched context via REPL.

        Returns context text to inject into the Claude prompt, or None if RLM
        didn't produce useful output.
        """
        from ccr.context.prompts import RLM_PACKING_SYSTEM

        rlm_playbook = None
        if self._ace_engine:
            rlm_playbook = self._ace_engine.get_playbook_for_prompt()

        rlm = CCRRlm(
            sub_client=self._sub_client,
            config=self.config.rlm,
            repo_index=self._repo_index,
            system_prompt=RLM_PACKING_SYSTEM,
            playbook_text=rlm_playbook,
        )

        # Ask RLM to find relevant files for this task
        task_prompt = (
            f"Task: {request.last_user_message}\n\n"
            f"Token budget: {self.config.pack_token_budget}\n"
            f"Find the relevant files and build a context pack using FINAL_VAR('pack_result')."
        )

        result = rlm.completion(task_prompt, root_prompt=request.last_user_message)

        logger.info(
            f"[{request.request_id}] RLM context build: "
            f"iterations={result.iterations_used}, "
            f"source={result.final_answer_source}, "
            f"time={result.execution_time:.1f}s"
        )

        # Log RLM tool usage via hooks
        if self._hooks:
            self._hooks.fire(HookEvent.POST_TOOL_USE, {
                "tool_name": "rlm-context-build",
                "file_path": f"iterations={result.iterations_used}",
                "observation": f"RLM built context in {result.iterations_used} iterations",
                "thought": f"Source: {result.final_answer_source}",
                "action": f"Generated context ({len(result.response)} chars)",
            })

        # Only use RLM result if it came from FINAL_VAR (structured output)
        if result.final_answer_source == "FINAL_VAR" and result.response:
            return f"<rlm_context>\n{result.response}\n</rlm_context>"

        return None

    def _get_model_for_target(self, target: str) -> str:
        if target in ("qwen_direct", "qwen_with_context"):
            return self.config.sub_model
        return self.config.claude_model

    def _auto_commit(
        self,
        request: CCRRequest,
        response: str,
        decision: RouteDecision,
    ) -> None:
        """Auto-commit after significant responses."""
        try:
            task_desc = request.last_user_message[:100]
            files = []
            if decision.context_pack:
                files = [f[0] for f in decision.context_pack.files[:5]]

            self.memory.commit(
                title=f"Auto: {task_desc}",
                what=f"Processed request via {decision.target}",
                why=task_desc,
                files_changed=files,
                next_step="Continue development",
            )
        except Exception as e:
            logger.warning(f"Auto-commit failed: {e}")
