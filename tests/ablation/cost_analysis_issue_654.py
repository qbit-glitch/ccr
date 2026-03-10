"""Comprehensive cost analysis: CCR three-mode comparison for ironclaw issue #654.

Issue #654: Unify three agentic loops into a single AgenticLoop engine.
This is the hardest issue in the ironclaw backlog — a deep architectural
refactoring across ~3,700 lines of duplicated code.

Compares PASSTHROUGH, ROUTE_ONLY, and FULL_CCR modes across:
- Per-turn token costs with growing context windows
- Multi-turn cumulative costs
- Token usage breakdown (prompt vs context vs output)
- ROI metrics: cost per line of code, context pack ROI, time savings
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ccr.utils.costs import calculate_cost, CostTracker, MODEL_COSTS
from ccr.utils.tokens import count_tokens, estimate_tokens


# ─────────────────────────────────────────────────────────────────
# Constants — Issue #654 specifics
# ─────────────────────────────────────────────────────────────────

CLAUDE_MODEL = "claude-sonnet-4-5"
SUB_MODEL = "gpt-oss-20b"

# Issue metrics
LINES_DUPLICATED = 3700
LINES_NEW_CODE = 1050  # 377 + 109 + 293 created + ~271 remaining (delegates, mod.rs)
LINES_DELETED = 2500
FILES_TOUCHED = 9
ACCEPTANCE_CRITERIA = 23
PITFALLS = 10

# The prompt as a developer would phrase it
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

# Context text blocks (from test_issue_654_three_modes.py)
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
        match output.result {
            RespondResult::Text(text) => {
                if llm_signals_tool_intent(&text) { nudge; continue; }
                return Ok(AgenticLoopResult::Response(sanitized));
            }
            RespondResult::ToolCalls { tool_calls, content } => {
                consecutive_tool_intent_nudges = 0;
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
    const MAX_TOOL_INTENT_NUDGES: u32 = 2;
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
    const MAX_TOOL_INTENT_NUDGES: u32 = 2;
}
</file>

<file path="src/agent/scheduler.rs" lines="449-500">
async fn execute_tool_task(tools, context_manager, safety, approval_context, job_id,
    tool_name, params) -> Result<String, Error> {
}
</file>

<file path="src/llm/reasoning.rs" lines="1-30">
pub const TOOL_INTENT_NUDGE: &str = "...";
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
# Data model for multi-turn simulation
# ─────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """A single turn in the multi-turn conversation."""
    turn_number: int
    turn_type: str  # "read", "write", "reasoning"
    description: str
    input_tokens: int    # prompt + context + conversation history
    output_tokens: int
    model: str = CLAUDE_MODEL

    @property
    def cost(self) -> float:
        return calculate_cost(self.model, self.input_tokens, self.output_tokens) or 0.0


@dataclass
class ModeSimulation:
    """Full multi-turn simulation for one CCR mode."""
    mode: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(t.cost for t in self.turns)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def reads(self) -> int:
        return sum(1 for t in self.turns if t.turn_type == "read")

    @property
    def writes(self) -> int:
        return sum(1 for t in self.turns if t.turn_type == "write")

    @property
    def reasonings(self) -> int:
        return sum(1 for t in self.turns if t.turn_type == "reasoning")


# ─────────────────────────────────────────────────────────────────
# Token counts from real CCR utilities
# ─────────────────────────────────────────────────────────────────

def get_token_counts() -> dict:
    """Compute real token counts for all context blocks."""
    prompt_tokens = count_tokens(ISSUE_PROMPT)
    route_memory_tokens = count_tokens(ROUTE_ONLY_MEMORY)
    full_memory_tokens = count_tokens(FULL_CCR_MEMORY)
    context_pack_tokens = count_tokens(FULL_CCR_CONTEXT_PACK)
    playbook_tokens = count_tokens(FULL_CCR_PLAYBOOK)

    return {
        "prompt": prompt_tokens,
        "route_memory": route_memory_tokens,
        "full_memory": full_memory_tokens,
        "context_pack": context_pack_tokens,
        "playbook": playbook_tokens,
        "full_ccr_total_context": full_memory_tokens + context_pack_tokens + playbook_tokens,
    }


# ─────────────────────────────────────────────────────────────────
# Multi-turn simulation builders
# ─────────────────────────────────────────────────────────────────

def build_passthrough_simulation() -> ModeSimulation:
    """PASSTHROUGH: ~35 turns. No context — Claude discovers everything via tools.

    Realistic turn sequence for a complex refactoring:
    - Turns 1-3: Initial reasoning about the task
    - Turns 4-15: Reading source files to understand the codebase
    - Turns 16-20: Reading more files (tests, imports, related modules)
    - Turns 21-25: Planning and partial writes (may need to re-read)
    - Turns 26-32: Writing new files and modifying existing ones
    - Turns 33-35: Verification reads and final reasoning
    """
    tc = get_token_counts()
    sim = ModeSimulation(mode="PASSTHROUGH")
    history_tokens = 0  # Grows with each turn

    # Turn 1: Initial prompt — Claude receives bare prompt, reasons about approach
    t1_input = tc["prompt"]
    t1_output = 800  # "I'll need to read the source files first..."
    sim.turns.append(Turn(1, "reasoning", "Initial task analysis — no context, must discover everything", t1_input, t1_output))
    history_tokens += t1_input + t1_output

    # Turns 2-6: Read the three main loop files
    read_files = [
        ("Read dispatcher.rs (~2051 lines)", 1200, 6000),
        ("Read worker.rs (~1516 lines)", 1000, 4500),
        ("Read worker/runtime.rs (~567 lines)", 800, 1700),
        ("Read scheduler.rs (tool execution copy)", 900, 1500),
        ("Read reasoning.rs (shared types)", 700, 900),
    ]
    for i, (desc, prompt_overhead, file_output) in enumerate(read_files, 2):
        t_input = history_tokens + prompt_overhead
        t_output = file_output
        sim.turns.append(Turn(i, "read", desc, t_input, t_output))
        history_tokens += prompt_overhead + t_output

    # Turns 7-11: Read supporting files
    support_reads = [
        ("Read safety/mod.rs", 500, 600),
        ("Read agent/mod.rs (module structure)", 400, 300),
        ("Read tools/mod.rs", 400, 400),
        ("Read error.rs (error types)", 500, 800),
        ("Read llm types (ChatMessage, ToolSelection)", 600, 1000),
    ]
    for i, (desc, prompt_overhead, file_output) in enumerate(support_reads, 7):
        t_input = history_tokens + prompt_overhead
        t_output = file_output
        sim.turns.append(Turn(i, "read", desc, t_input, t_output))
        history_tokens += prompt_overhead + t_output

    # Turns 12-16: Read test files and more implementation details
    more_reads = [
        ("Read worker.rs tests section", 500, 1200),
        ("Read dispatcher.rs approval flow detail", 600, 1500),
        ("Read context/state.rs (JobState, JobContext)", 500, 800),
        ("Read hooks/mod.rs (HookRegistry interface)", 400, 500),
        ("Read channels/web/types.rs (SseEvent)", 400, 400),
    ]
    for i, (desc, prompt_overhead, file_output) in enumerate(more_reads, 12):
        t_input = history_tokens + prompt_overhead
        t_output = file_output
        sim.turns.append(Turn(i, "read", desc, t_input, t_output))
        history_tokens += prompt_overhead + t_output

    # Turns 17-21: Additional reads for imports and edge cases
    edge_reads = [
        ("Read rate_limiter.rs (RateLimitResult)", 400, 500),
        ("Read tools/tool.rs (Tool trait)", 400, 600),
        ("Read agent/task.rs (Job types)", 400, 500),
        ("Read worker/claude_bridge.rs (container patterns)", 500, 700),
        ("Re-read dispatcher.rs tool execution section", 600, 2000),
    ]
    for i, (desc, prompt_overhead, file_output) in enumerate(edge_reads, 17):
        t_input = history_tokens + prompt_overhead
        t_output = file_output
        sim.turns.append(Turn(i, "read", desc, t_input, t_output))
        history_tokens += prompt_overhead + t_output

    # Turn 22: Reasoning — plan the implementation
    t22_input = history_tokens + 500
    t22_output = 2000  # Detailed implementation plan
    sim.turns.append(Turn(22, "reasoning", "Plan implementation: traits, types, migration order", t22_input, t22_output))
    history_tokens += 500 + t22_output

    # Turns 23-27: Write new files
    writes = [
        ("Write agentic_loop_engine.rs (LoopDelegate trait + run_agentic_loop)", 4000),
        ("Write tools/execute.rs (execute_tool_safely + process_tool_output)", 1500),
        ("Write worker/job.rs (JobDelegate + JobDeps)", 3500),
        ("Write ChatDelegate impl in dispatcher.rs", 2500),
        ("Write ContainerDelegate impl", 2000),
    ]
    for i, (desc, output) in enumerate(writes, 23):
        t_input = history_tokens + 600
        sim.turns.append(Turn(i, "write", desc, t_input, output))
        history_tokens += 600 + output

    # Turns 28-30: Modify existing files
    mods = [
        ("Update agent/mod.rs — module declarations", 800),
        ("Update tools/mod.rs — add execute module", 500),
        ("Update scheduler.rs — use shared execute", 1200),
    ]
    for i, (desc, output) in enumerate(mods, 28):
        t_input = history_tokens + 500
        sim.turns.append(Turn(i, "write", desc, t_input, output))
        history_tokens += 500 + output

    # Turns 31-33: Verification reads
    verify_reads = [
        ("Re-read new agentic_loop_engine.rs to verify", 500, 1000),
        ("Check dispatcher.rs compiles with new delegate", 500, 800),
        ("Verify worker/job.rs imports are correct", 400, 600),
    ]
    for i, (desc, overhead, output) in enumerate(verify_reads, 31):
        t_input = history_tokens + overhead
        sim.turns.append(Turn(i, "read", desc, t_input, output))
        history_tokens += overhead + output

    # Turns 34-35: Final writes (fixes from verification)
    final_writes = [
        ("Fix import errors found during verification", 1000),
        ("Add missing test stubs and module documentation", 1200),
    ]
    for i, (desc, output) in enumerate(final_writes, 34):
        t_input = history_tokens + 500
        sim.turns.append(Turn(i, "write", desc, t_input, output))
        history_tokens += 500 + output

    return sim


def build_route_only_simulation() -> ModeSimulation:
    """ROUTE_ONLY: ~22 turns. Has project memory, knows file names.

    CCR classifies as COMPLEX, routes to Claude with light memory context.
    Claude knows file names and project structure but not file contents.
    Saves ~5 orientation reads vs PASSTHROUGH.
    """
    tc = get_token_counts()
    sim = ModeSimulation(mode="ROUTE_ONLY")
    history_tokens = 0

    # Turn 1: Prompt + memory context — Claude knows file names
    t1_input = tc["prompt"] + tc["route_memory"]
    t1_output = 600  # "I know the files, let me read them..."
    sim.turns.append(Turn(1, "reasoning", "Task analysis with project memory — knows file locations", t1_input, t1_output))
    history_tokens += t1_input + t1_output

    # Turns 2-5: Read the three main files (skips orientation reads)
    reads = [
        ("Read dispatcher.rs (knows exact path from memory)", 800, 6000),
        ("Read worker.rs", 700, 4500),
        ("Read worker/runtime.rs", 600, 1700),
        ("Read scheduler.rs tool execution", 600, 1500),
    ]
    for i, (desc, overhead, output) in enumerate(reads, 2):
        t_input = history_tokens + overhead
        sim.turns.append(Turn(i, "read", desc, t_input, output))
        history_tokens += overhead + output

    # Turns 6-9: Read supporting types (fewer than PASSTHROUGH — memory gives hints)
    support = [
        ("Read safety/mod.rs + reasoning.rs", 500, 1500),
        ("Read llm types", 500, 1000),
        ("Read error.rs", 400, 800),
        ("Read context/state.rs", 400, 800),
    ]
    for i, (desc, overhead, output) in enumerate(support, 6):
        t_input = history_tokens + overhead
        sim.turns.append(Turn(i, "read", desc, t_input, output))
        history_tokens += overhead + output

    # Turns 10-13: Deeper reads for edge cases
    deeper = [
        ("Read worker.rs tests", 500, 1200),
        ("Read dispatcher.rs approval section", 500, 1500),
        ("Read hooks interface", 400, 500),
        ("Read rate_limiter.rs", 400, 500),
    ]
    for i, (desc, overhead, output) in enumerate(deeper, 10):
        t_input = history_tokens + overhead
        sim.turns.append(Turn(i, "read", desc, t_input, output))
        history_tokens += overhead + output

    # Turn 14: Reasoning
    t14_input = history_tokens + 500
    t14_output = 1800
    sim.turns.append(Turn(14, "reasoning", "Plan implementation with known architecture", t14_input, t14_output))
    history_tokens += 500 + t14_output

    # Turns 15-20: Write files
    writes = [
        ("Write agentic_loop_engine.rs", 4000),
        ("Write tools/execute.rs", 1500),
        ("Write worker/job.rs", 3500),
        ("Write ChatDelegate", 2500),
        ("Write ContainerDelegate", 2000),
        ("Update mod.rs files + scheduler.rs", 1500),
    ]
    for i, (desc, output) in enumerate(writes, 15):
        t_input = history_tokens + 600
        sim.turns.append(Turn(i, "write", desc, t_input, output))
        history_tokens += 600 + output

    # Turns 21-22: Verification and fixes
    sim.turns.append(Turn(21, "write", "Fix import issues", history_tokens + 500, 1000))
    history_tokens += 500 + 1000
    sim.turns.append(Turn(22, "write", "Add test stubs", history_tokens + 400, 1200))

    return sim


def build_full_ccr_simulation() -> ModeSimulation:
    """FULL_CCR: ~3 turns. All source files pre-loaded + ACE playbook.

    Context pack includes all 6 relevant source files, project memory,
    and ACE playbook with refactoring strategies. Claude has everything
    it needs to generate the complete refactoring in one shot.
    """
    tc = get_token_counts()
    sim = ModeSimulation(mode="FULL_CCR")

    # Full context = prompt + memory + context pack + playbook
    full_context = (
        tc["prompt"]
        + tc["full_memory"]
        + tc["context_pack"]
        + tc["playbook"]
    )

    # Turn 1: Generate the main implementation (agentic_loop_engine.rs + execute.rs)
    t1_input = full_context
    t1_output = 5500  # ~400 lines of Rust = ~5500 tokens
    sim.turns.append(Turn(
        1, "write",
        "Generate agentic_loop_engine.rs + tools/execute.rs (all source pre-loaded)",
        t1_input, t1_output,
    ))

    # Turn 2: Generate remaining files (job.rs + delegates + mod.rs updates)
    t2_input = full_context + t1_output + 200  # Previous output in history
    t2_output = 6500  # job.rs + ChatDelegate + ContainerDelegate + mod.rs
    sim.turns.append(Turn(
        2, "write",
        "Generate worker/job.rs + ChatDelegate + ContainerDelegate + mod.rs updates",
        t2_input, t2_output,
    ))

    # Turn 3: Final reasoning — verify completeness, list tests needed
    t3_input = full_context + t1_output + t2_output + 400
    t3_output = 1500  # Summary + test checklist
    sim.turns.append(Turn(
        3, "reasoning",
        "Verify completeness against 23 acceptance criteria, list test stubs",
        t3_input, t3_output,
    ))

    return sim


# ─────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────

def compute_roi_metrics(sim: ModeSimulation) -> dict:
    """Compute ROI metrics for a simulation."""
    total_lines_changed = LINES_NEW_CODE + LINES_DELETED
    cost = sim.total_cost
    return {
        "cost_per_line_new": cost / LINES_NEW_CODE if LINES_NEW_CODE > 0 else 0,
        "cost_per_line_changed": cost / total_lines_changed if total_lines_changed > 0 else 0,
        "cost_per_turn": cost / sim.num_turns if sim.num_turns > 0 else 0,
        "tokens_per_turn": sim.total_tokens / sim.num_turns if sim.num_turns > 0 else 0,
        "input_output_ratio": sim.total_input_tokens / sim.total_output_tokens if sim.total_output_tokens > 0 else 0,
        "estimated_minutes": sim.num_turns * 0.5,  # ~30s per turn average
    }


def compute_context_pack_roi(tc: dict) -> dict:
    """Compute ROI of the context pack itself."""
    pack_tokens = tc["full_ccr_total_context"]
    pack_input_cost = calculate_cost(CLAUDE_MODEL, pack_tokens, 0) or 0.0

    # What it replaces: ~23 file read turns
    # Each read turn: ~600 tokens overhead + growing history
    # Average read turn input: ~15000 tokens (midpoint of growing history)
    avg_read_input = 15000
    avg_read_output = 1500
    reads_replaced = 23
    read_cost_total = sum(
        (calculate_cost(CLAUDE_MODEL, avg_read_input + i * 2000, avg_read_output) or 0.0)
        for i in range(reads_replaced)
    )

    return {
        "pack_tokens": pack_tokens,
        "pack_cost": pack_input_cost,
        "reads_replaced": reads_replaced,
        "read_cost_replaced": read_cost_total,
        "net_savings": read_cost_total - pack_input_cost,
        "roi_multiple": read_cost_total / pack_input_cost if pack_input_cost > 0 else float("inf"),
    }


def compute_cost_tracker_report(sim: ModeSimulation) -> dict:
    """Use CCR's CostTracker to build a report for a simulation."""
    tracker = CostTracker(baseline_model=CLAUDE_MODEL)
    for turn in sim.turns:
        tracker.record(turn.model, turn.input_tokens, turn.output_tokens)
    return tracker.get_report()


# ─────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────

def print_report():
    """Print the full formatted cost analysis report."""
    tc = get_token_counts()

    pt = build_passthrough_simulation()
    ro = build_route_only_simulation()
    fc = build_full_ccr_simulation()
    sims = {"PASSTHROUGH": pt, "ROUTE_ONLY": ro, "FULL_CCR": fc}

    print()
    print("=" * 90)
    print("  ISSUE #654 COST ANALYSIS — THREE CCR MODES")
    print("  Unify 3 agentic loops into single AgenticLoop engine")
    print("=" * 90)
    print()

    # ── Section 1: Issue Metrics ──
    print("  ISSUE COMPLEXITY METRICS")
    print("  " + "-" * 50)
    print(f"    Duplicated code:       {LINES_DUPLICATED:,} lines across 3 files")
    print(f"    New code written:      {LINES_NEW_CODE:,} lines across 3+ files")
    print(f"    Code deleted:          {LINES_DELETED:,} lines")
    print(f"    Net change:            -{LINES_DELETED - LINES_NEW_CODE:,} lines")
    print(f"    Files touched:         {FILES_TOUCHED}")
    print(f"    Acceptance criteria:   {ACCEPTANCE_CRITERIA}")
    print(f"    Listed pitfalls:       {PITFALLS}")
    print(f"    Classification:        COMPLEX (keyword: 'refactor')")
    print()

    # ── Section 2: Token Counts ──
    print("  TOKEN COUNTS (measured by CCR)")
    print("  " + "-" * 50)
    print(f"    Issue prompt:          {tc['prompt']:,} tokens")
    print(f"    ROUTE_ONLY memory:     {tc['route_memory']:,} tokens")
    print(f"    FULL_CCR memory:       {tc['full_memory']:,} tokens")
    print(f"    FULL_CCR context pack: {tc['context_pack']:,} tokens")
    print(f"    FULL_CCR playbook:     {tc['playbook']:,} tokens")
    print(f"    FULL_CCR total ctx:    {tc['full_ccr_total_context']:,} tokens")
    print()

    # ── Section 3: Model Pricing ──
    inp_cost, out_cost = MODEL_COSTS[CLAUDE_MODEL]
    print(f"  MODEL PRICING ({CLAUDE_MODEL})")
    print("  " + "-" * 50)
    print(f"    Input:  ${inp_cost:.2f} / 1M tokens")
    print(f"    Output: ${out_cost:.2f} / 1M tokens")
    print()

    # ── Section 4: Multi-Turn Summary ──
    print("=" * 90)
    print("  MULTI-TURN SIMULATION SUMMARY")
    print("=" * 90)
    print()
    header = (
        f"  {'Mode':<14} {'Turns':>6} {'Reads':>6} {'Writes':>7} "
        f"{'Reason':>7} {'Input Tok':>12} {'Output Tok':>12} {'Total Cost':>12} {'vs PT':>8}"
    )
    print(header)
    print("  " + "-" * 86)

    for name in ["PASSTHROUGH", "ROUTE_ONLY", "FULL_CCR"]:
        s = sims[name]
        vs_pt = ""
        if name != "PASSTHROUGH":
            pct = ((pt.total_cost - s.total_cost) / pt.total_cost) * 100
            vs_pt = f"-{pct:.0f}%"
        else:
            vs_pt = "base"
        print(
            f"  {name:<14} {s.num_turns:>6} {s.reads:>6} {s.writes:>7} "
            f"{s.reasonings:>7} {s.total_input_tokens:>12,} {s.total_output_tokens:>12,} "
            f"${s.total_cost:>10.4f} {vs_pt:>8}"
        )
    print("  " + "-" * 86)
    print()

    # ── Section 5: Per-Turn Cost Breakdown ──
    for name in ["PASSTHROUGH", "ROUTE_ONLY", "FULL_CCR"]:
        s = sims[name]
        print(f"  [{name}] Per-Turn Breakdown")
        print(f"  {'#':>4} {'Type':<10} {'Input':>10} {'Output':>8} {'Cost':>10}  Description")
        print("  " + "-" * 80)
        for t in s.turns:
            print(
                f"  {t.turn_number:>4} {t.turn_type:<10} {t.input_tokens:>10,} "
                f"{t.output_tokens:>8,} ${t.cost:>8.4f}  {t.description[:40]}"
            )
        print(f"  {'':>4} {'TOTAL':<10} {s.total_input_tokens:>10,} "
              f"{s.total_output_tokens:>8,} ${s.total_cost:>8.4f}")
        print()

    # ── Section 6: ROI Metrics ──
    print("=" * 90)
    print("  ROI METRICS")
    print("=" * 90)
    print()

    header = f"  {'Metric':<35} {'PASSTHROUGH':>14} {'ROUTE_ONLY':>14} {'FULL_CCR':>14}"
    print(header)
    print("  " + "-" * 78)

    metrics = {name: compute_roi_metrics(sims[name]) for name in sims}

    print(f"  {'Cost per line (new code)':<35} "
          f"${metrics['PASSTHROUGH']['cost_per_line_new']:>12.5f} "
          f"${metrics['ROUTE_ONLY']['cost_per_line_new']:>12.5f} "
          f"${metrics['FULL_CCR']['cost_per_line_new']:>12.5f}")
    print(f"  {'Cost per line (all changes)':<35} "
          f"${metrics['PASSTHROUGH']['cost_per_line_changed']:>12.6f} "
          f"${metrics['ROUTE_ONLY']['cost_per_line_changed']:>12.6f} "
          f"${metrics['FULL_CCR']['cost_per_line_changed']:>12.6f}")
    print(f"  {'Cost per turn':<35} "
          f"${metrics['PASSTHROUGH']['cost_per_turn']:>12.4f} "
          f"${metrics['ROUTE_ONLY']['cost_per_turn']:>12.4f} "
          f"${metrics['FULL_CCR']['cost_per_turn']:>12.4f}")
    print(f"  {'Tokens per turn':<35} "
          f"{metrics['PASSTHROUGH']['tokens_per_turn']:>14,.0f} "
          f"{metrics['ROUTE_ONLY']['tokens_per_turn']:>14,.0f} "
          f"{metrics['FULL_CCR']['tokens_per_turn']:>14,.0f}")
    print(f"  {'Input/output ratio':<35} "
          f"{metrics['PASSTHROUGH']['input_output_ratio']:>14.1f}x "
          f"{metrics['ROUTE_ONLY']['input_output_ratio']:>14.1f}x "
          f"{metrics['FULL_CCR']['input_output_ratio']:>14.1f}x")
    print(f"  {'Estimated time (minutes)':<35} "
          f"{metrics['PASSTHROUGH']['estimated_minutes']:>14.1f} "
          f"{metrics['ROUTE_ONLY']['estimated_minutes']:>14.1f} "
          f"{metrics['FULL_CCR']['estimated_minutes']:>14.1f}")
    print()

    # ── Section 7: Context Pack ROI ──
    print("=" * 90)
    print("  CONTEXT PACK ROI ANALYSIS")
    print("=" * 90)
    print()

    roi = compute_context_pack_roi(tc)
    print(f"    Context pack size:        {roi['pack_tokens']:,} tokens")
    print(f"    Context pack cost:        ${roi['pack_cost']:.4f}")
    print(f"    File reads replaced:      {roi['reads_replaced']}")
    print(f"    Cost of those reads:      ${roi['read_cost_replaced']:.4f}")
    print(f"    Net savings:              ${roi['net_savings']:.4f}")
    print(f"    ROI multiple:             {roi['roi_multiple']:.1f}x")
    print()
    print(f"    Insight: The context pack costs ${roi['pack_cost']:.4f} but saves")
    print(f"    ${roi['read_cost_replaced']:.4f} in eliminated file-reading turns,")
    print(f"    a {roi['roi_multiple']:.0f}x return on the context token investment.")
    print()

    # ── Section 8: Cumulative Cost Curve ──
    print("=" * 90)
    print("  CUMULATIVE COST CURVE (cost after N turns)")
    print("=" * 90)
    print()

    # Show at key milestones
    milestones = [1, 3, 5, 10, 15, 20, 25, 30, 35]
    print(f"  {'Turn':>6} {'PASSTHROUGH':>14} {'ROUTE_ONLY':>14} {'FULL_CCR':>14}")
    print("  " + "-" * 50)
    for m in milestones:
        pt_cum = sum(t.cost for t in pt.turns[:m])
        ro_cum = sum(t.cost for t in ro.turns[:m])
        fc_cum = sum(t.cost for t in fc.turns[:m])
        pt_str = f"${pt_cum:.4f}" if m <= pt.num_turns else "--"
        ro_str = f"${ro_cum:.4f}" if m <= ro.num_turns else "--"
        fc_str = f"${fc_cum:.4f}" if m <= fc.num_turns else "--"
        print(f"  {m:>6} {pt_str:>14} {ro_str:>14} {fc_str:>14}")
    print()

    # ── Section 9: Cost Tracker Reports ──
    print("=" * 90)
    print("  CCR COST TRACKER REPORTS")
    print("=" * 90)
    print()

    for name in sims:
        report = compute_cost_tracker_report(sims[name])
        print(f"  [{name}]")
        print(f"    Total cost:     ${report['total_cost_usd']:.6f}")
        print(f"    Baseline cost:  ${report['baseline_cost_usd']:.6f}")
        print(f"    Savings:        ${report['savings_usd']:.6f} ({report['savings_pct']:.1f}%)")
        for model_name, model_data in report["models"].items():
            print(f"    {model_name}: {model_data['calls']} calls, "
                  f"{model_data['input_tokens']:,} in / {model_data['output_tokens']:,} out, "
                  f"${model_data['cost_usd']:.6f}")
        print()

    # ── Section 10: Final Summary ──
    print("=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)
    print()

    savings_ro = pt.total_cost - ro.total_cost
    savings_fc = pt.total_cost - fc.total_cost
    pct_ro = (savings_ro / pt.total_cost) * 100
    pct_fc = (savings_fc / pt.total_cost) * 100

    turn_reduction_ro = pt.num_turns - ro.num_turns
    turn_reduction_fc = pt.num_turns - fc.num_turns

    print(f"    PASSTHROUGH total:    ${pt.total_cost:.4f}  ({pt.num_turns} turns)")
    print(f"    ROUTE_ONLY total:     ${ro.total_cost:.4f}  ({ro.num_turns} turns)  "
          f"saves ${savings_ro:.4f} ({pct_ro:.0f}%), {turn_reduction_ro} fewer turns")
    print(f"    FULL_CCR total:       ${fc.total_cost:.4f}  ({fc.num_turns} turns)   "
          f"saves ${savings_fc:.4f} ({pct_fc:.0f}%), {turn_reduction_fc} fewer turns")
    print()
    print(f"    FULL_CCR achieves {pct_fc:.0f}% cost reduction and {turn_reduction_fc}x fewer")
    print(f"    turns vs PASSTHROUGH for the hardest issue in the ironclaw backlog.")
    print()
    print(f"    Key drivers:")
    print(f"    1. Context pack eliminates ~20 file-reading turns")
    print(f"    2. ACE playbook provides refactoring strategies upfront")
    print(f"    3. Growing context window in PASSTHROUGH makes later turns expensive")
    print(f"    4. FULL_CCR front-loads context cost but avoids cumulative growth")
    print()


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

def test_token_counts_are_positive():
    """All token counts should be positive integers."""
    tc = get_token_counts()
    for key, val in tc.items():
        assert val > 0, f"{key} should be positive, got {val}"


def test_full_ccr_context_larger_than_route_only():
    """FULL_CCR context should be significantly larger than ROUTE_ONLY."""
    tc = get_token_counts()
    assert tc["full_ccr_total_context"] > tc["route_memory"] * 5


def test_passthrough_has_most_turns():
    """PASSTHROUGH should have the most turns."""
    pt = build_passthrough_simulation()
    ro = build_route_only_simulation()
    fc = build_full_ccr_simulation()
    assert pt.num_turns > ro.num_turns > fc.num_turns


def test_full_ccr_cheapest():
    """FULL_CCR should be the cheapest mode."""
    pt = build_passthrough_simulation()
    ro = build_route_only_simulation()
    fc = build_full_ccr_simulation()
    assert fc.total_cost < ro.total_cost < pt.total_cost


def test_passthrough_no_context():
    """PASSTHROUGH first turn should have only prompt tokens as input."""
    pt = build_passthrough_simulation()
    tc = get_token_counts()
    assert pt.turns[0].input_tokens == tc["prompt"]


def test_full_ccr_has_3_turns():
    """FULL_CCR should complete in exactly 3 turns."""
    fc = build_full_ccr_simulation()
    assert fc.num_turns == 3


def test_passthrough_has_35_turns():
    """PASSTHROUGH should have exactly 35 turns."""
    pt = build_passthrough_simulation()
    assert pt.num_turns == 35


def test_route_only_has_22_turns():
    """ROUTE_ONLY should have exactly 22 turns."""
    ro = build_route_only_simulation()
    assert ro.num_turns == 22


def test_passthrough_23_reads():
    """PASSTHROUGH should have 23 read turns (5+5+5+5+3 verification)."""
    pt = build_passthrough_simulation()
    assert pt.reads == 23


def test_route_only_12_reads():
    """ROUTE_ONLY should have 12 read turns."""
    ro = build_route_only_simulation()
    assert ro.reads == 12


def test_full_ccr_0_reads():
    """FULL_CCR should have 0 read turns."""
    fc = build_full_ccr_simulation()
    assert fc.reads == 0


def test_all_modes_use_claude():
    """All modes should use Claude for COMPLEX tasks."""
    for sim in [build_passthrough_simulation(), build_route_only_simulation(), build_full_ccr_simulation()]:
        for turn in sim.turns:
            assert turn.model == CLAUDE_MODEL, f"{sim.mode} turn {turn.turn_number} uses {turn.model}"


def test_cost_per_turn_increases_in_passthrough():
    """Later turns in PASSTHROUGH should cost more due to growing context."""
    pt = build_passthrough_simulation()
    # First 5 turns vs last 5 turns
    early_avg = sum(t.cost for t in pt.turns[:5]) / 5
    late_avg = sum(t.cost for t in pt.turns[-5:]) / 5
    assert late_avg > early_avg, "Later turns should cost more than early turns"


def test_context_pack_roi_positive():
    """Context pack ROI should be positive (saves more than it costs)."""
    tc = get_token_counts()
    roi = compute_context_pack_roi(tc)
    assert roi["net_savings"] > 0
    assert roi["roi_multiple"] > 1.0


def test_cost_tracker_report_matches():
    """CostTracker report should match manual calculation."""
    fc = build_full_ccr_simulation()
    report = compute_cost_tracker_report(fc)
    assert abs(report["total_cost_usd"] - fc.total_cost) < 0.001


def test_roi_cost_per_line_positive():
    """Cost per line should be positive for all modes."""
    for name, sim in [("PT", build_passthrough_simulation()),
                      ("RO", build_route_only_simulation()),
                      ("FC", build_full_ccr_simulation())]:
        m = compute_roi_metrics(sim)
        assert m["cost_per_line_new"] > 0, f"{name} cost_per_line_new should be positive"
        assert m["cost_per_line_changed"] > 0, f"{name} cost_per_line_changed should be positive"


def test_full_ccr_savings_over_50_percent():
    """FULL_CCR should save at least 50% vs PASSTHROUGH."""
    pt = build_passthrough_simulation()
    fc = build_full_ccr_simulation()
    pct = ((pt.total_cost - fc.total_cost) / pt.total_cost) * 100
    assert pct > 50, f"Expected >50% savings, got {pct:.1f}%"


def test_turn_types_consistent():
    """Each simulation should have the expected turn type distribution."""
    pt = build_passthrough_simulation()
    assert pt.reads == 23
    assert pt.writes == 10
    assert pt.reasonings == 2
    assert pt.reads + pt.writes + pt.reasonings == pt.num_turns

    ro = build_route_only_simulation()
    assert ro.reads == 12
    assert ro.writes == 8
    assert ro.reasonings == 2
    assert ro.reads + ro.writes + ro.reasonings == ro.num_turns

    fc = build_full_ccr_simulation()
    assert fc.reads == 0
    assert fc.writes == 2
    assert fc.reasonings == 1
    assert fc.reads + fc.writes + fc.reasonings == fc.num_turns


def test_cumulative_cost_monotonic():
    """Cumulative cost should be monotonically increasing."""
    for sim in [build_passthrough_simulation(), build_route_only_simulation(), build_full_ccr_simulation()]:
        cumulative = 0.0
        for turn in sim.turns:
            new_cumulative = cumulative + turn.cost
            assert new_cumulative >= cumulative, f"{sim.mode} turn {turn.turn_number} decreased cumulative cost"
            cumulative = new_cumulative


def test_calculate_cost_uses_real_function():
    """Verify we're using CCR's real calculate_cost, not a mock."""
    cost = calculate_cost(CLAUDE_MODEL, 1_000_000, 0)
    inp_rate = MODEL_COSTS[CLAUDE_MODEL][0]
    assert cost == inp_rate, f"Expected ${inp_rate}, got ${cost}"


def test_count_tokens_uses_real_function():
    """Verify we're using CCR's real count_tokens, not a mock."""
    tokens = count_tokens("hello world")
    assert isinstance(tokens, int)
    assert tokens > 0


if __name__ == "__main__":
    print_report()
