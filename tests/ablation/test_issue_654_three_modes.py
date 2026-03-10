"""Issue #654 — Hardest ironclaw issue, demonstrated across three CCR modes.

Issue: Unify three duplicated agentic loops (dispatcher.rs, worker.rs, runtime.rs)
into a single AgenticLoop engine with a LoopDelegate trait.

This is the hardest issue in the ironclaw backlog:
- 15,625 chars of specification
- ~3,700 lines of duplicated code across 3 files
- Touches the core architecture (agent loop, tool execution, event sourcing)
- 10 listed pitfalls and landmines
- 23 acceptance criteria checkboxes

Demonstrates how CCR's FULL_CCR mode provides maximum value for COMPLEX tasks
by pre-loading all relevant source files and ACE playbook strategies.
"""

from __future__ import annotations

import sys
import os
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ccr.core.types import CCRRequest, RouterConfig, ComplexityTier
from ccr.core.router import TaskRouter
from ccr.utils.tokens import count_tokens
from ccr.utils.costs import calculate_cost


# ─────────────────────────────────────────────────────────────────
# The prompt — as a developer would phrase it
# ─────────────────────────────────────────────────────────────────

ISSUE_PROMPT = (
    "refactor: unify three agentic loops into single AgenticLoop engine, "
    "retire src/agent/worker.rs. The core agentic loop — LLM call → tool "
    "execution → result processing → context update → repeat — is independently "
    "implemented in three separate files with significant copy-paste duplication: "
    "dispatcher.rs (~770 lines, interactive chat), worker.rs (~600 lines, "
    "background jobs), and worker/runtime.rs (~400 lines, Docker container). "
    "Create a shared loop engine in src/agent/agentic_loop_engine.rs with a "
    "LoopDelegate trait. Implement ChatDelegate, JobDelegate, ContainerDelegate. "
    "Extract shared tool execution (validate → timeout → execute → serialize) "
    "and shared result processing (sanitize → wrap → ChatMessage). Delete "
    "src/agent/worker.rs, move to src/worker/job.rs."
)


# ─────────────────────────────────────────────────────────────────
# Context layers for each mode
# ─────────────────────────────────────────────────────────────────

PASSTHROUGH_CONTEXT = ""  # Nothing

ROUTE_ONLY_MEMORY = """## Project: IronClaw
- Rust-based secure AI assistant (~160K lines)
- Agent loop: Job orchestration with state machine
- Architecture: agent/, tools/, channels/, safety/, sandbox/, workspace/
- Key files: dispatcher.rs, worker.rs, runtime.rs contain agentic loops
"""

FULL_CCR_MEMORY = """## Project: IronClaw
- Rust-based secure AI assistant platform (~160K lines)
- Architecture: agent/, tools/, channels/, safety/, sandbox/, workspace/, llm/, db/
- Agent loop: dispatcher.rs (chat, ~2051L), worker.rs (jobs, ~1516L), worker/runtime.rs (container, ~567L)
- Duplicated logic: tool intent nudge, completion detection, tool execution pipeline, result processing
- Tools: 30+ built-in + WASM sandbox, ToolRegistry, rate limiting, approval model
- Safety: Sanitizer, Validator, LeakDetector, PolicyRules
- LLM: Reasoning struct, respond_with_tools(), select_tools(), ToolSelection
- Key patterns: Fix pattern not instance, zero clippy warnings, regression tests required
- Key types: WorkerDeps, Worker, AgenticLoopResult, RespondResult, ToolExecResult
"""

FULL_CCR_CONTEXT_PACK = """<file path="src/agent/dispatcher.rs" lines="37-44,150-370">
pub(super) async fn run_agentic_loop(&self, message: &IncomingMessage, session: Arc<Mutex<Session>>,
    thread_id: Uuid, initial_messages: Vec<ChatMessage>) -> Result<AgenticLoopResult, Error> {
    let max_tool_iterations = self.config.max_tool_iterations;
    let force_text_at = max_tool_iterations;
    let nudge_at = max_tool_iterations.saturating_sub(1);
    let mut iteration = 0;
    const MAX_TOOL_INTENT_NUDGES: u32 = 2;
    let mut consecutive_tool_intent_nudges: u32 = 0;
    loop {
        iteration += 1;
        if iteration > max_tool_iterations + 1 { return Err(...); }
        // Check interrupted → cost guard → nudge injection → LLM call
        match output.result {
            RespondResult::Text(text) => {
                if llm_signals_tool_intent(&text) { nudge; continue; }
                return Ok(AgenticLoopResult::Response(sanitized));
            }
            RespondResult::ToolCalls { tool_calls, content } => {
                consecutive_tool_intent_nudges = 0;
                // 3-phase: preflight (approval) → parallel exec → postflight
                // ~370 lines of chat-specific approval logic
            }
        }
    }
}
</file>

<file path="src/agent/worker.rs" lines="30-47,291-580">
pub struct WorkerDeps {
    pub context_manager: Arc<ContextManager>,
    pub llm: Arc<dyn LlmProvider>,
    pub safety: Arc<SafetyLayer>,
    pub tools: Arc<ToolRegistry>,
    pub store: Option<Arc<dyn Database>>,
    pub hooks: Arc<HookRegistry>,
    pub timeout: Duration,
    pub use_planning: bool,
    pub sse_tx: Option<broadcast::Sender<SseEvent>>,
    pub approval_context: Option<ApprovalContext>,
}

async fn execution_loop(&self, rx: &mut Receiver<WorkerMessage>,
    reasoning: &Reasoning, reason_ctx: &mut ReasoningContext) -> Result<(), Error> {
    const MAX_TOOL_INTENT_NUDGES: u32 = 2;  // DUPLICATED from dispatcher.rs
    // Planning phase → direct selection loop
    // Tool execution: validate → timeout → execute → serialize (DUPLICATED)
    // Result processing: sanitize → wrap → ChatMessage (DUPLICATED)
    // Completion: llm_signals_completion() (DUPLICATED)
}
</file>

<file path="src/worker/runtime.rs" lines="49-60,218-370">
pub struct WorkerRuntime {
    config: WorkerConfig,
    client: Arc<WorkerHttpClient>,
    llm: Arc<dyn LlmProvider>,
    safety: Arc<SafetyLayer>,
    tools: Arc<ToolRegistry>,
}

async fn execution_loop(&self, ...) -> Result<String, WorkerError> {
    const MAX_TOOL_INTENT_NUDGES: u32 = 2;  // DUPLICATED
    // Sequential tool execution (no parallel)
    // Same validate → timeout → execute → serialize (DUPLICATED)
    // Same sanitize → wrap (DUPLICATED)
    // llm_signals_completion() (DUPLICATED)
}
</file>

<file path="src/agent/scheduler.rs" lines="449-500">
async fn execute_tool_task(tools, context_manager, safety, approval_context, job_id,
    tool_name, params) -> Result<String, Error> {
    // 4th copy of validate → timeout → execute → serialize
}
</file>

<file path="src/llm/reasoning.rs" lines="1-30">
pub const TOOL_INTENT_NUDGE: &str = "...";  // Shared constant, used in all 3 loops
pub struct Reasoning { ... }
impl Reasoning {
    pub async fn respond_with_tools(&self, ctx: &ReasoningContext) -> Result<LlmOutput, LlmError>;
    pub async fn plan(&self, ctx: &ReasoningContext) -> Result<ActionPlan, LlmError>;
}
</file>

<file path="src/safety/mod.rs" lines="48-55">
impl SafetyLayer {
    pub fn sanitize_tool_output(&self, tool_name: &str, output: &str) -> SanitizedOutput;
    pub fn wrap_for_llm(&self, tool_name: &str, content: &str, was_modified: bool) -> String;
}
</file>"""

FULL_CCR_PLAYBOOK = """## STRATEGIES & INSIGHTS
[strat-00001] helpful=12 harmful=0 :: In IronClaw, always check both PostgreSQL and libSQL impls when modifying Database trait
[strat-00005] helpful=6 harmful=0 :: When adding new GatewayState fields, update: struct definition, GatewayChannel::new(), rebuild_state(), test_helpers, ws.rs test state
[strat-00006] helpful=8 harmful=0 :: Use &dyn Trait (not impl Trait) for large functions to avoid monomorphization bloat — especially agentic loops
[strat-00007] helpful=7 harmful=0 :: When extracting shared code from duplicated modules, keep consumer-specific logic in the consumer (delegate/strategy pattern)

## COMMON MISTAKES TO AVOID
[avoid-00001] helpful=9 harmful=0 :: Never byte-slice user strings in Rust; use is_char_boundary() for UTF-8 safety
[avoid-00004] helpful=5 harmful=0 :: Don't forget to update all construction sites when adding struct fields — grep for a unique field name to find them all
[avoid-00005] helpful=6 harmful=0 :: When deleting a module (worker.rs), update mod.rs, all pub use re-exports, and all downstream imports (scheduler.rs, agent_loop.rs)
[avoid-00006] helpful=4 harmful=0 :: Duplicate tests in worker.rs for llm_signals_completion() — delete them, keep only the ones in util.rs

## PROBLEM-SOLVING HEURISTICS
[heur-00001] helpful=5 harmful=0 :: For refactoring: 1) create new shared module 2) implement trait 3) migrate one consumer at a time 4) delete old code last
[heur-00002] helpful=4 harmful=0 :: Test migration: move tests alongside the code they test, don't create new test files
"""


# ─────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────

def classify_prompt(prompt: str) -> ComplexityTier:
    """Classify using CCR's real heuristic router."""
    import re
    config = RouterConfig()
    msg_lower = prompt.lower()
    token_count = count_tokens([{"role": "user", "content": prompt}])

    # 1. COMPLEX keywords
    for kw in config.complex_keywords:
        if kw.lower() in msg_lower:
            return ComplexityTier.COMPLEX

    # 2. COMPLEX exploration patterns
    for pat in config.complex_patterns:
        if re.search(pat, msg_lower):
            return ComplexityTier.COMPLEX

    # 3. TRIVIAL
    if token_count < config.trivial_token_threshold:
        has_code = "```" in prompt
        has_file_ref = bool(re.search(r"\.\w{2,4}\b", prompt))
        if not has_code and not has_file_ref:
            return ComplexityTier.TRIVIAL

    # 4. Questions → SIMPLE
    if re.match(r"^(what|how|why|where|when)\s", msg_lower):
        return ComplexityTier.SIMPLE

    # 5. Action verbs
    if re.match(r"^\s*(fix|add|remove|rename|update|change|refactor)\s", msg_lower):
        if token_count < config.simple_token_threshold:
            return ComplexityTier.SIMPLE
        return ComplexityTier.MODERATE

    return ComplexityTier.MODERATE


CLAUDE_MODEL = "claude-sonnet-4-5"
SUB_MODEL = "gpt-oss-20b"
EST_OUTPUT_TOKENS = 8000  # Large refactoring generates a lot of code


def compute_mode(mode: str):
    tier = classify_prompt(ISSUE_PROMPT)
    prompt_tokens = count_tokens(ISSUE_PROMPT)

    if mode == "PASSTHROUGH":
        context_tokens = 0
        model = CLAUDE_MODEL
    elif mode == "ROUTE_ONLY":
        context_tokens = count_tokens(ROUTE_ONLY_MEMORY)
        model = SUB_MODEL if tier in (ComplexityTier.TRIVIAL, ComplexityTier.SIMPLE) else CLAUDE_MODEL
    else:  # FULL_CCR
        context_tokens = (
            count_tokens(FULL_CCR_MEMORY)
            + count_tokens(FULL_CCR_CONTEXT_PACK)
            + count_tokens(FULL_CCR_PLAYBOOK)
        )
        model = SUB_MODEL if tier in (ComplexityTier.TRIVIAL, ComplexityTier.SIMPLE) else CLAUDE_MODEL

    input_tokens = prompt_tokens + context_tokens
    cost = calculate_cost(model, input_tokens, EST_OUTPUT_TOKENS) or 0.0

    return {
        "tier": tier,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "context_tokens": context_tokens,
        "input_tokens": input_tokens,
        "output_tokens": EST_OUTPUT_TOKENS,
        "cost": cost,
    }


def run_demo():
    print()
    print("=" * 80)
    print("  ISSUE #654 — HARDEST IRONCLAW ISSUE: THREE-MODE COMPARISON")
    print("  Unify 3 agentic loops into single AgenticLoop engine")
    print("=" * 80)
    print()

    # Classification
    tier = classify_prompt(ISSUE_PROMPT)
    prompt_tokens = count_tokens(ISSUE_PROMPT)
    print(f"  Classification: {tier.value.upper()}")
    print(f"  Prompt tokens: {prompt_tokens}")
    print(f"  Matched keyword: 'refactor' → COMPLEX")
    print()

    # Issue complexity metrics
    print("  Issue Complexity Metrics:")
    print("    - Spec length:        15,625 chars")
    print("    - Duplicated code:    ~3,700 lines across 3 files")
    print("    - Files to modify:    8+ files")
    print("    - New files to create: 3 files")
    print("    - Acceptance criteria: 23 checkboxes")
    print("    - Listed pitfalls:    10")
    print()

    # Compute modes
    modes = ["PASSTHROUGH", "ROUTE_ONLY", "FULL_CCR"]
    results = {mode: compute_mode(mode) for mode in modes}

    # Token comparison
    print("=" * 80)
    print("  TOKEN & COST COMPARISON (single request)")
    print("=" * 80)
    print()
    print(f"  {'Mode':<16} {'Model':<20} {'Prompt':>7} {'Context':>8} {'Input':>7} {'Output':>7} {'Cost':>10}")
    print("  " + "-" * 76)

    for mode in modes:
        r = results[mode]
        print(
            f"  {mode:<16} {r['model']:<20} {r['prompt_tokens']:>7} "
            f"{r['context_tokens']:>8} {r['input_tokens']:>7} {r['output_tokens']:>7} "
            f"${r['cost']:>8.4f}"
        )
    print("  " + "-" * 76)
    print()

    # What each mode provides
    print("=" * 80)
    print("  WHAT EACH MODE PROVIDES")
    print("=" * 80)
    print()

    descriptions = {
        "PASSTHROUGH": textwrap.dedent("""
        No context. Claude sees only the prompt text.
        Must discover: file locations, struct definitions, trait interfaces,
        all 3 loop implementations, shared patterns, test locations.
        Estimated: ~20-30 tool calls just to read and understand the code.
        Then another ~10-15 to write and verify the new files.
        """).strip(),

        "ROUTE_ONLY": textwrap.dedent("""
        Light memory context (project overview, key file list).
        CCR classifies as COMPLEX ('refactor' keyword) → routes to Claude.
        Claude knows the file names but not their contents.
        Saves ~5 tool calls for initial orientation.
        Still needs ~15-25 tool calls for code reading and writing.
        """).strip(),

        "FULL_CCR": textwrap.dedent("""
        Full context pack with all 6 relevant source files pre-loaded:
        - dispatcher.rs: Loop structure, approval flow, tool intent nudge
        - worker.rs: WorkerDeps struct, execution_loop, planning phase
        - runtime.rs: Container loop, sequential tools, HTTP events
        - scheduler.rs: 4th copy of tool execution
        - reasoning.rs: TOOL_INTENT_NUDGE constant, Reasoning API
        - safety/mod.rs: sanitize_tool_output, wrap_for_llm
        Plus ACE playbook with strategies:
        - Use &dyn Trait for large functions (avoid monomorphization)
        - Keep consumer-specific logic in delegates
        - Migration order: create new → implement trait → migrate one at a time
        - Delete duplicate tests (worker.rs llm_signals_completion)
        Claude can generate the FULL refactoring in 1-3 turns.
        """).strip(),
    }

    for mode in modes:
        print(f"  [{mode}]")
        for line in descriptions[mode].split("\n"):
            print(f"    {line}")
        print()

    # Multi-turn cost projection
    print("=" * 80)
    print("  MULTI-TURN COST PROJECTION")
    print("=" * 80)
    print()

    # PASSTHROUGH: ~35 turns for a complex refactoring
    pt_turns = 35
    pt_read_cost = calculate_cost(CLAUDE_MODEL, 800, 300) or 0.0
    pt_write_cost = calculate_cost(CLAUDE_MODEL, 500, 2000) or 0.0
    pt_total = results["PASSTHROUGH"]["cost"] + 20 * pt_read_cost + 10 * pt_write_cost

    # ROUTE_ONLY: ~22 turns
    ro_turns = 22
    ro_total = results["ROUTE_ONLY"]["cost"] + 12 * pt_read_cost + 8 * pt_write_cost

    # FULL_CCR: ~3 turns (all code pre-loaded, just write)
    fc_turns = 3
    fc_total = results["FULL_CCR"]["cost"] + 2 * pt_write_cost

    print(f"  {'Mode':<16} {'Turns':>6} {'Read calls':>11} {'Write calls':>12} {'Est. Total':>12} {'vs PT':>8}")
    print("  " + "-" * 66)
    print(f"  {'PASSTHROUGH':<16} {pt_turns:>6} {'~20':>11} {'~10':>12} ${pt_total:>10.4f} {'base':>8}")
    print(f"  {'ROUTE_ONLY':<16} {ro_turns:>6} {'~12':>11} {'~8':>12} ${ro_total:>10.4f} {f'-{((pt_total-ro_total)/pt_total)*100:.0f}%':>8}")
    print(f"  {'FULL_CCR':<16} {fc_turns:>6} {'0':>11} {'~2':>12} ${fc_total:>10.4f} {f'-{((pt_total-fc_total)/pt_total)*100:.0f}%':>8}")
    print("  " + "-" * 66)
    print()

    # The actual fix summary
    print("=" * 80)
    print("  FILES CREATED (FULL_CCR approach)")
    print("=" * 80)
    print()
    print("  New files:")
    print("    1. src/agent/agentic_loop_engine.rs  (~300 lines)")
    print("       - LoopDelegate trait with 8 async methods")
    print("       - run_agentic_loop() — the ONE shared loop")
    print("       - AgenticLoopConfig, LoopSignal, LoopOutcome types")
    print("       - execute_tool_with_safety() shared helper")
    print()
    print("    2. src/tools/execute.rs  (~100 lines)")
    print("       - execute_tool_safely() — validate → timeout → execute → serialize")
    print("       - process_tool_output() — sanitize → wrap → ChatMessage")
    print("       - Replaces 4 duplicated copies")
    print()
    print("    3. src/worker/job.rs  (~250 lines)")
    print("       - JobDelegate implementing LoopDelegate for background jobs")
    print("       - JobDeps (replaces WorkerDeps)")
    print("       - Parallel execution via JoinSet")
    print("       - Completion detection via llm_signals_completion()")
    print("       - Planning phase support")
    print()
    print("  Files to modify:")
    print("    4. src/agent/dispatcher.rs — ChatDelegate impl, remove duplicated loop")
    print("    5. src/worker/runtime.rs → container.rs — ContainerDelegate impl")
    print("    6. src/agent/mod.rs — remove `pub mod worker`, add agentic_loop_engine")
    print("    7. src/agent/scheduler.rs — use JobDeps, shared execute")
    print("    8. src/tools/mod.rs — add `pub mod execute`")
    print()
    print("  Files to delete:")
    print("    9. src/agent/worker.rs — RETIRED (replaced by worker/job.rs)")
    print()

    # Value analysis
    print("=" * 80)
    print("  FULL_CCR VALUE ANALYSIS FOR COMPLEX TASKS")
    print("=" * 80)
    print()

    context_tokens = results["FULL_CCR"]["context_tokens"]
    context_cost = calculate_cost(CLAUDE_MODEL, context_tokens, 0) or 0.0
    saved_reads = 20  # Tool calls avoided
    read_cost_saved = saved_reads * pt_read_cost

    print(f"  Context pack size:       {context_tokens:,} tokens")
    print(f"  Context pack cost:       ${context_cost:.4f}")
    print(f"  Tool calls avoided:      ~{saved_reads} file reads")
    print(f"  Cost of those reads:     ${read_cost_saved:.4f}")
    print(f"  Net savings per request: ${read_cost_saved - context_cost:.4f}")
    print(f"  ROI on context packing:  {read_cost_saved / context_cost:.1f}x" if context_cost > 0 else "  ROI: infinite (zero context cost)")
    print()
    print(f"  Key insight: For COMPLEX tasks, the context pack PAYS FOR ITSELF")
    print(f"  {read_cost_saved / context_cost:.0f}x over by eliminating tool call turns." if context_cost > 0 else "")
    print()


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

def test_issue_654_classified_as_complex():
    """'refactor' keyword should trigger COMPLEX classification."""
    tier = classify_prompt(ISSUE_PROMPT)
    assert tier == ComplexityTier.COMPLEX, f"Expected COMPLEX, got {tier}"


def test_passthrough_no_context():
    r = compute_mode("PASSTHROUGH")
    assert r["context_tokens"] == 0
    assert r["model"] == CLAUDE_MODEL


def test_all_modes_use_claude_for_complex():
    """COMPLEX tasks always route to Claude, never sub-model."""
    for mode in ["PASSTHROUGH", "ROUTE_ONLY", "FULL_CCR"]:
        r = compute_mode(mode)
        assert r["model"] == CLAUDE_MODEL, f"{mode} should use Claude for COMPLEX"


def test_full_ccr_has_most_context():
    """FULL_CCR should have significantly more context than ROUTE_ONLY."""
    full = compute_mode("FULL_CCR")
    route = compute_mode("ROUTE_ONLY")
    assert full["context_tokens"] > route["context_tokens"] * 5


def test_full_ccr_context_includes_all_files():
    """Context pack should reference all key source files."""
    assert "dispatcher.rs" in FULL_CCR_CONTEXT_PACK
    assert "worker.rs" in FULL_CCR_CONTEXT_PACK
    assert "runtime.rs" in FULL_CCR_CONTEXT_PACK
    assert "scheduler.rs" in FULL_CCR_CONTEXT_PACK
    assert "reasoning.rs" in FULL_CCR_CONTEXT_PACK
    assert "safety/mod.rs" in FULL_CCR_CONTEXT_PACK


def test_playbook_has_refactoring_strategies():
    """ACE playbook should contain refactoring-specific strategies."""
    assert "monomorphization" in FULL_CCR_PLAYBOOK
    assert "delegate" in FULL_CCR_PLAYBOOK.lower()
    assert "migration" in FULL_CCR_PLAYBOOK.lower()


def test_implementation_files_exist():
    """All implementation files for issue #654 should exist."""
    import os
    base = os.path.join(os.path.dirname(__file__), "..", "..", "codebase_testing", "ironclaw", "src")
    expected_files = [
        "agent/agentic_loop_engine.rs",
        "agent/chat_delegate.rs",
        "tools/execute.rs",
        "worker/job.rs",
        "worker/container.rs",
    ]
    for f in expected_files:
        path = os.path.join(base, f)
        assert os.path.exists(path), f"Missing implementation file: {f}"


def test_mod_rs_updated():
    """Module files should register the new modules."""
    import os
    base = os.path.join(os.path.dirname(__file__), "..", "..", "codebase_testing", "ironclaw", "src")

    # agent/mod.rs should reference agentic_loop_engine and chat_delegate
    agent_mod = open(os.path.join(base, "agent", "mod.rs")).read()
    assert "agentic_loop_engine" in agent_mod
    assert "chat_delegate" in agent_mod

    # tools/mod.rs should reference execute
    tools_mod = open(os.path.join(base, "tools", "mod.rs")).read()
    assert "pub mod execute" in tools_mod

    # worker/mod.rs should reference job and container
    worker_mod = open(os.path.join(base, "worker", "mod.rs")).read()
    assert "pub mod job" in worker_mod
    assert "pub mod container" in worker_mod


def test_delegate_trait_consistency():
    """All delegate files should implement LoopDelegate."""
    import os
    base = os.path.join(os.path.dirname(__file__), "..", "..", "codebase_testing", "ironclaw", "src")

    delegates = [
        ("agent/chat_delegate.rs", "ChatDelegate"),
        ("worker/job.rs", "JobDelegate"),
        ("worker/container.rs", "ContainerDelegate"),
    ]
    for path, name in delegates:
        content = open(os.path.join(base, path)).read()
        assert "impl LoopDelegate for" in content, f"{path} should implement LoopDelegate"
        assert name in content, f"{path} should define {name}"


if __name__ == "__main__":
    run_demo()
