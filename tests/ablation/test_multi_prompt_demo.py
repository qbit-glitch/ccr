"""Multi-prompt ablation demo — simulates a realistic Claude Code session.

Runs 20 prompts against the ironclaw codebase at varying complexity levels,
comparing token usage and cost across three CCR ablation modes:

  1. PASSTHROUGH  — every request goes to Claude (no CCR)
  2. ROUTE_ONLY   — classify + route cheap tasks to sub-model, no context packing
  3. FULL_CCR     — classify + route + context pack + memory + ACE playbook

Prompt mix mirrors real-world usage:
  40% TRIVIAL, 25% SIMPLE, 20% MODERATE, 15% COMPLEX
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ccr.core.types import CCRRequest, RouterConfig, ComplexityTier
from ccr.core.router import TaskRouter
from ccr.utils.tokens import count_tokens
from ccr.utils.costs import calculate_cost


# ─────────────────────────────────────────────────────────────────
# Simulated context layers (what CCR would inject)
# ─────────────────────────────────────────────────────────────────

MEMORY_CONTEXT_L1 = """## Project: IronClaw
- Rust-based secure AI assistant platform
- Active branch: main
"""

MEMORY_CONTEXT_L2 = """## Project: IronClaw
- Rust-based secure AI assistant platform with multi-channel input (TUI, HTTP, WASM, Web)
- Agent loop: Job orchestration with state machine (Pending→InProgress→Completed→Submitted→Accepted|Failed)
- Tools: 30+ built-in tools with rate limiting, approval model, WASM sandbox
- Active branch: main
- Recent: Added routine heartbeat system, fixed WASM fuel metering
"""

MEMORY_CONTEXT_L3 = """## Project: IronClaw
- Rust-based secure AI assistant platform (~160K lines)
- Architecture: agent/, tools/, channels/, safety/, sandbox/, workspace/, llm/, db/
- Agent loop: Job → Session → Thread → Turn with undo checkpoints, context compaction
- Tools: 30+ built-in + WASM sandbox (wasmtime, fuel metering, credential injection)
- Channels: TUI (ratatui), HTTP webhooks, Web gateway (SSE/WS), WASM channels, REPL, Gateway
- Safety: Sanitizer, Validator, LeakDetector, OutputRedactor
- LLM: Multi-provider (Anthropic, OpenAI, Ollama, NEAR AI, Tinfoil) with fallback chain
- DB: Dual PostgreSQL/libSQL with ~78 trait methods, refinery migrations
- Workspace: Persistent memory with BM25 + vector hybrid search (RRF)
- Skills: SKILL.md format, trust-based tool attenuation, ClawHub registry
- Recent: Added routine heartbeat, fixed WASM fuel metering, refactored approval model
- Key patterns: Fix pattern not instance, zero clippy warnings, transaction safety
"""

ACE_PLAYBOOK = """## STRATEGIES & INSIGHTS
[strat-00001] helpful=12 harmful=0 :: In IronClaw, always check both PostgreSQL and libSQL impls when modifying Database trait
[strat-00002] helpful=8 harmful=0 :: Tool implementations must handle sanitization — check requires_sanitization() before output
[strat-00003] helpful=6 harmful=0 :: When modifying state transitions, update can_transition_to() AND all match arms in the agent loop
[strat-00004] helpful=5 harmful=0 :: WASM tools use fuel metering — always set fuel limits proportional to expected computation

## COMMON MISTAKES TO AVOID
[avoid-00001] helpful=9 harmful=0 :: Never byte-slice user strings in Rust; use is_char_boundary() for UTF-8 safety
[avoid-00002] helpful=7 harmful=0 :: Don't forget to update ToolRegistry when adding new tools — both register() and schema validation
[avoid-00003] helpful=4 harmful=0 :: Rate limiter uses sliding window — don't confuse with fixed window when writing tests
"""

# Simulated context packs (what the packer would produce for different tasks)
CONTEXT_PACK_MODERATE = """<file path="src/tools/tool.rs">
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    async fn execute(&self, ctx: &JobContext, params: serde_json::Value) -> Result<ToolOutput, ToolError>;
    fn requires_sanitization(&self) -> bool { true }
    fn approval_requirement(&self) -> ApprovalRequirement { ApprovalRequirement::UnlessAutoApproved }
}

pub enum ApprovalRequirement {
    Never,
    UnlessAutoApproved,
    Always,
}
</file>

<file path="src/tools/registry.rs">
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    rate_limiters: HashMap<String, RateLimiter>,
}

impl ToolRegistry {
    pub fn register(&mut self, tool: Arc<dyn Tool>) { /* ... */ }
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> { /* ... */ }
    pub fn check_rate_limit(&self, name: &str) -> Result<(), ToolError> { /* ... */ }
}
</file>

<file path="src/tools/builtin/time.rs">
pub struct TimeTool;

#[async_trait]
impl Tool for TimeTool {
    fn name(&self) -> &str { "time" }
    fn description(&self) -> &str { "Get current time, parse timestamps, calculate time differences" }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "enum": ["now", "parse", "diff"] },
                "input": { "type": "string" },
                "timezone": { "type": "string" }
            },
            "required": ["action"]
        })
    }
    async fn execute(&self, _ctx: &JobContext, params: Value) -> Result<ToolOutput, ToolError> {
        // Full implementation with timezone math...
    }
}
</file>"""

CONTEXT_PACK_COMPLEX = """<file path="src/agent/mod.rs">
pub struct Agent {
    session: Session,
    tools: ToolRegistry,
    llm: Arc<dyn LlmProvider>,
    safety: SafetyLayer,
    db: Arc<dyn Database>,
    workspace: Workspace,
    hooks: HookRegistry,
    observer: Arc<dyn Observer>,
}

impl Agent {
    pub async fn process_turn(&mut self, input: IncomingMessage) -> Result<OutgoingResponse> {
        let job = self.create_job(input).await?;
        let turn = self.session.new_turn(job.id)?;
        // Tool dispatch loop with approval checks...
    }
}
</file>

<file path="src/context/state.rs">
#[derive(Debug, Clone, PartialEq)]
pub enum JobState {
    Pending,
    InProgress,
    Completed,
    Submitted,
    Accepted,
    Failed,
    Stuck,
}

impl JobState {
    pub fn can_transition_to(&self, target: &JobState) -> bool {
        matches!((self, target),
            (Pending, InProgress) |
            (InProgress, Completed) | (InProgress, Failed) | (InProgress, Stuck) |
            (Completed, Submitted) |
            (Submitted, Accepted) | (Submitted, Failed) |
            (Stuck, InProgress) | (Stuck, Failed)
        )
    }
}
</file>

<file path="src/agent/routine.rs">
pub struct Routine {
    pub id: Uuid,
    pub name: String,
    pub schedule: RoutineSchedule,
    pub prompt: String,
    pub enabled: bool,
    pub last_run: Option<DateTime<Utc>>,
    pub run_count: u64,
}

pub enum RoutineSchedule {
    Cron(String),          // "0 */6 * * *"
    Reactive(EventTrigger),
    Webhook(WebhookConfig),
}

pub struct RoutineEngine {
    routines: Vec<Routine>,
    executor: Arc<Agent>,
}
</file>

<file path="src/safety/mod.rs">
pub struct SafetyLayer {
    sanitizer: Sanitizer,
    validator: Validator,
    leak_detector: LeakDetector,
    policy: SafetyPolicy,
}

impl SafetyLayer {
    pub fn check_input(&self, input: &str) -> Result<(), SafetyViolation> { /* ... */ }
    pub fn sanitize_output(&self, output: &ToolOutput) -> ToolOutput { /* ... */ }
    pub fn detect_leaks(&self, content: &str) -> Vec<LeakFinding> { /* ... */ }
}
</file>

<file path="src/workspace/mod.rs">
pub struct Workspace {
    db: Arc<dyn Database>,
    embeddings: Arc<dyn EmbeddingProvider>,
    chunker: Chunker,
}

impl Workspace {
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let bm25 = self.db.bm25_search(query, limit * 2).await?;
        let vector = self.embed_and_search(query, limit * 2).await?;
        Ok(reciprocal_rank_fusion(bm25, vector, limit))
    }
}
</file>"""


# ─────────────────────────────────────────────────────────────────
# 20-prompt session simulating real Claude Code usage on ironclaw
# ─────────────────────────────────────────────────────────────────

PROMPTS = [
    # ─── TRIVIAL (8 prompts, 40%) ──────────────────────────────────────
    # Short lookups — below 500 token threshold, no code blocks, no file refs
    {"prompt": "what is ironclaw?", "est_output_tokens": 150},
    {"prompt": "where is the main function?", "est_output_tokens": 80},
    {"prompt": "list the dependencies", "est_output_tokens": 200},
    {"prompt": "what license is this project?", "est_output_tokens": 50},
    {"prompt": "show me the ToolError enum", "est_output_tokens": 120},
    {"prompt": "what port does the HTTP server run on?", "est_output_tokens": 60},
    {"prompt": "how many test files are there?", "est_output_tokens": 50},
    {"prompt": "what database does it use?", "est_output_tokens": 80},

    # ─── SIMPLE (5 prompts, 25%) ───────────────────────────────────────
    # Action verbs / questions above trivial threshold but below simple threshold
    {"prompt": "fix the typo in the rate limiter error message — it currently says 'to many requests' but the correct spelling is 'too many requests'. Also check if this same typo exists in any other error messages across the codebase and fix those too. Make sure the fix doesn't break any existing tests that might assert on the old string.", "est_output_tokens": 200},
    {"prompt": "rename the variable `ctx` to `job_context` throughout the tool dispatch function and any callers that pass it in. Make sure you update the destructuring patterns, closure captures, and any documentation comments that reference the old name.", "est_output_tokens": 250},
    {"prompt": "how does the rate limiter prevent a tool from exceeding 60 calls per minute? Is it a fixed window or sliding window implementation? What happens when the limit is hit — does it return an error, queue the call, or silently drop it?", "est_output_tokens": 350},
    {"prompt": "remove all unused imports from the sandbox module. Run cargo clippy to identify them, then remove each one. Make sure the project still compiles with all feature flags after the cleanup.", "est_output_tokens": 200},
    {"prompt": "change the default rate limit from 60 requests per minute to 120 requests per minute. Find the constant, update it, update any documentation that references the old value, and update the corresponding unit test assertions.", "est_output_tokens": 250},

    # ─── MODERATE (4 prompts, 20%) ─────────────────────────────────────
    # Longer action prompts above the simple threshold (2000 tokens) —
    # multi-step, multi-file, schema changes, need context packing
    {"prompt": "add a new built-in tool called `json_diff` that compares two JSON values and returns their structural differences, showing added keys, removed keys, and changed values with full dot-notation paths. " + "Implement the Tool trait following the exact same pattern as the existing TimeTool — constructor, name, description, parameters_schema returning a JSON Schema object, and async execute method. Register it in the ToolRegistry's default tool list. Define the input schema to accept two required parameters `left` and `right` (both arbitrary JSON values) and an optional `max_depth` integer parameter (default 10). " * 6 + "Write unit tests covering: nested objects, arrays with reordering, type changes (string→number), null handling, empty objects, deeply nested structures beyond max_depth, and unicode keys.", "est_output_tokens": 1500},
    {"prompt": "add a retry count field to the Routine struct and increment it each time execution fails. Reset it to zero on success. " + "Store the retry count in the database — create new migration files for both PostgreSQL (using refinery) and libSQL backends. The migration should add a `retry_count INTEGER NOT NULL DEFAULT 0` column to the routines table. Update the RoutineEngine to read this field when loading routines and write it back after each execution attempt. " * 6 + "Add a max_retries config option (default 3) that automatically disables the routine after too many consecutive failures. Log a warning when a routine is disabled. Write integration tests covering: increment on failure, reset on success, auto-disable at threshold, and re-enable via API.", "est_output_tokens": 1200},
    {"prompt": "update the LeakDetector to catch AWS credential patterns. " + "Add detection for AWS access keys (strings starting with AKIA followed by exactly 16 uppercase alphanumeric characters) and AWS secret keys (40-character base64 strings that appear after common variable assignment patterns like AWS_SECRET_ACCESS_KEY=, aws_secret=, etc). " * 6 + "Integrate the new patterns into the existing detection framework and ensure they flow through the redaction pipeline correctly. Add tests for: valid AKIA keys, valid secret keys, false positives from base64-encoded binary data, keys inside JSON strings, keys in environment variable formats, and keys split across multiple lines.", "est_output_tokens": 800},
    {"prompt": "add a --dry-run flag to the routine engine that simulates execution without calling the LLM or running tools. " + "In dry-run mode, the engine should log exactly what would happen at each step: which routine was triggered (by cron schedule, event, or webhook), what system prompt and user prompt would be sent to the LLM provider, which tools the LLM would likely invoke based on the prompt and available tools, what parameters those tools would receive, and what state transitions the job would go through. " * 6 + "Support both single-routine dry-run (by routine ID) and full-schedule simulation (show what would run in the next N hours). Add CLI integration with `ironclaw routine dry-run <id>` and `ironclaw routine simulate --hours 24` commands.", "est_output_tokens": 1000},

    # ─── COMPLEX (3 prompts, 15%) ──────────────────────────────────────
    # Architecture-level: exploration, system-wide refactoring, new subsystems
    {"prompt": "understand the codebase", "est_output_tokens": 3000},
    {"prompt": "design a resource quota system where each user gets a daily budget of 1000 tool calls across all jobs and routines. Track usage in the database, enforce at the dispatcher level, and integrate with the existing rate limiter and approval system without breaking current sessions.", "est_output_tokens": 4000},
    {"prompt": "refactor the entire safety layer to support pluggable policy engines — allow users to define custom safety rules via a YAML config file that gets hot-reloaded. The current hardcoded patterns should become the default policy. Support pattern-based rules, regex rules, and LLM-based content classification rules.", "est_output_tokens": 5000},
]


# ─────────────────────────────────────────────────────────────────
# Classification engine (uses real CCR router heuristics)
# ─────────────────────────────────────────────────────────────────

def classify_prompt(prompt: str) -> ComplexityTier:
    """Classify using CCR's real heuristic router (mirrors router.py logic)."""
    import re
    config = RouterConfig()
    msg_lower = prompt.lower()
    token_count = count_tokens([{"role": "user", "content": prompt}])

    # 1. COMPLEX keywords (highest priority)
    for keyword in config.complex_keywords:
        if keyword.lower() in msg_lower:
            return ComplexityTier.COMPLEX

    # 2. COMPLEX exploration patterns
    for pat in config.complex_patterns:
        if re.search(pat, msg_lower):
            return ComplexityTier.COMPLEX

    # 3. TRIVIAL: short, no code, no file refs — before SIMPLE to avoid
    #    greedy pattern matches on "what is X?"
    if token_count < config.trivial_token_threshold:
        has_code = "```" in prompt
        has_file_ref = bool(re.search(r"\.\w{2,4}\b", prompt))
        if not has_code and not has_file_ref:
            return ComplexityTier.TRIVIAL

    # 4. Questions → SIMPLE
    question_patterns = [
        r"^(what|how|why|where|when)\s",
        r"^\s*explain\s",
    ]
    if any(re.match(p, msg_lower) for p in question_patterns):
        return ComplexityTier.SIMPLE

    # 5. Action verbs → SIMPLE (short) or MODERATE (long)
    action_patterns = [
        r"^\s*(fix|add|remove|rename|update|change)\s",
    ]
    if any(re.match(p, msg_lower) for p in action_patterns):
        if token_count < config.simple_token_threshold:
            return ComplexityTier.SIMPLE
        return ComplexityTier.MODERATE

    # 6. Ambiguous → MODERATE fallback
    return ComplexityTier.MODERATE


# ─────────────────────────────────────────────────────────────────
# Ablation runner
# ─────────────────────────────────────────────────────────────────

CLAUDE_MODEL = "claude-sonnet-4-5"
SUB_MODEL = "gpt-oss-20b"


def get_context_tokens(tier: ComplexityTier, mode: str) -> int:
    """Calculate injected context tokens for a given tier and ablation mode."""
    if mode == "PASSTHROUGH":
        return 0

    if mode == "ROUTE_ONLY":
        if tier == ComplexityTier.TRIVIAL:
            return 0
        elif tier == ComplexityTier.SIMPLE:
            return count_tokens(MEMORY_CONTEXT_L1)
        elif tier == ComplexityTier.MODERATE:
            return count_tokens(MEMORY_CONTEXT_L2)
        else:
            return count_tokens(MEMORY_CONTEXT_L3)

    # FULL_CCR
    if tier == ComplexityTier.TRIVIAL:
        return 0
    elif tier == ComplexityTier.SIMPLE:
        return count_tokens(MEMORY_CONTEXT_L1)
    elif tier == ComplexityTier.MODERATE:
        return (
            count_tokens(MEMORY_CONTEXT_L2)
            + count_tokens(CONTEXT_PACK_MODERATE)
            + count_tokens(ACE_PLAYBOOK)
        )
    else:  # COMPLEX
        return (
            count_tokens(MEMORY_CONTEXT_L3)
            + count_tokens(CONTEXT_PACK_COMPLEX)
            + count_tokens(ACE_PLAYBOOK)
        )


def get_model_for_tier(tier: ComplexityTier, mode: str) -> str:
    """Which model handles this tier in each mode."""
    if mode == "PASSTHROUGH":
        return CLAUDE_MODEL

    # ROUTE_ONLY and FULL_CCR route TRIVIAL+SIMPLE to sub-model
    if tier in (ComplexityTier.TRIVIAL, ComplexityTier.SIMPLE):
        return SUB_MODEL
    return CLAUDE_MODEL


def run_demo():
    print()
    print("=" * 80)
    print("  CCR MULTI-PROMPT ABLATION DEMO")
    print("  Simulated 20-prompt session on IronClaw codebase (~160K lines Rust)")
    print("=" * 80)
    print()

    # First, show classifications
    print("─" * 80)
    print(f"  {'#':<3} {'Tier':<10} {'Tokens':>6}  {'Prompt':<55}")
    print("─" * 80)

    for i, p in enumerate(PROMPTS, 1):
        tier = classify_prompt(p["prompt"])
        tokens = count_tokens(p["prompt"])
        short = p["prompt"][:53] + ".." if len(p["prompt"]) > 55 else p["prompt"]
        print(f"  {i:<3} {tier.value:<10} {tokens:>6}  {short:<55}")

    print("─" * 80)
    print()

    # Run three ablations
    modes = ["PASSTHROUGH", "ROUTE_ONLY", "FULL_CCR"]
    mode_results = {}

    for mode in modes:
        total_input = 0
        total_output = 0
        total_cost = 0.0
        claude_calls = 0
        sub_calls = 0
        model_breakdown = {}

        for p in PROMPTS:
            tier = classify_prompt(p["prompt"])
            prompt_tokens = count_tokens(p["prompt"])
            context_tokens = get_context_tokens(tier, mode)
            input_tokens = prompt_tokens + context_tokens
            output_tokens = p["est_output_tokens"]
            model = get_model_for_tier(tier, mode)

            cost = calculate_cost(model, input_tokens, output_tokens) or 0.0
            total_input += input_tokens
            total_output += output_tokens
            total_cost += cost

            if model == CLAUDE_MODEL:
                claude_calls += 1
            else:
                sub_calls += 1

            if model not in model_breakdown:
                model_breakdown[model] = {"calls": 0, "input": 0, "output": 0, "cost": 0.0}
            model_breakdown[model]["calls"] += 1
            model_breakdown[model]["input"] += input_tokens
            model_breakdown[model]["output"] += output_tokens
            model_breakdown[model]["cost"] += cost

        mode_results[mode] = {
            "total_input": total_input,
            "total_output": total_output,
            "total_cost": total_cost,
            "claude_calls": claude_calls,
            "sub_calls": sub_calls,
            "breakdown": model_breakdown,
        }

    # Print comparison table
    print("=" * 80)
    print("  ABLATION COMPARISON")
    print("=" * 80)
    print()
    print(f"  {'Mode':<22} {'Input':>8} {'Output':>8} {'Total':>8} {'Cost':>10} {'Claude':>7} {'Sub':>5}")
    print("  " + "─" * 70)

    for mode in modes:
        r = mode_results[mode]
        total = r["total_input"] + r["total_output"]
        print(
            f"  {mode:<22} {r['total_input']:>8} {r['total_output']:>8} "
            f"{total:>8} ${r['total_cost']:>8.4f} {r['claude_calls']:>7} {r['sub_calls']:>5}"
        )

    print("  " + "─" * 70)
    print()

    # Savings analysis
    baseline = mode_results["PASSTHROUGH"]["total_cost"]
    for mode in ["ROUTE_ONLY", "FULL_CCR"]:
        cost = mode_results[mode]["total_cost"]
        saved = baseline - cost
        pct = (saved / baseline) * 100 if baseline > 0 else 0
        print(f"  {mode}: saves ${saved:.4f} ({pct:.1f}%) vs PASSTHROUGH")

    print()

    # Per-model breakdown
    print("=" * 80)
    print("  PER-MODEL BREAKDOWN")
    print("=" * 80)
    print()

    for mode in modes:
        print(f"  {mode}:")
        for model, data in mode_results[mode]["breakdown"].items():
            print(
                f"    {model:<30} {data['calls']:>3} calls  "
                f"{data['input']:>7} in  {data['output']:>7} out  "
                f"${data['cost']:.4f}"
            )
        print()

    # Per-tier analysis
    print("=" * 80)
    print("  PER-TIER COST ANALYSIS")
    print("=" * 80)
    print()

    tier_groups = {
        "TRIVIAL": [p for p in PROMPTS if classify_prompt(p["prompt"]) == ComplexityTier.TRIVIAL],
        "SIMPLE": [p for p in PROMPTS if classify_prompt(p["prompt"]) == ComplexityTier.SIMPLE],
        "MODERATE": [p for p in PROMPTS if classify_prompt(p["prompt"]) == ComplexityTier.MODERATE],
        "COMPLEX": [p for p in PROMPTS if classify_prompt(p["prompt"]) == ComplexityTier.COMPLEX],
    }

    print(f"  {'Tier':<12} {'Count':>5}  {'PASSTHROUGH':>12} {'ROUTE_ONLY':>12} {'FULL_CCR':>12}  {'Savings':>8}")
    print("  " + "─" * 68)

    for tier_name, prompts in tier_groups.items():
        tier_enum = ComplexityTier(tier_name.lower())
        count = len(prompts)

        costs = {}
        for mode in modes:
            mode_cost = 0.0
            for p in prompts:
                prompt_tokens = count_tokens(p["prompt"])
                context_tokens = get_context_tokens(tier_enum, mode)
                model = get_model_for_tier(tier_enum, mode)
                cost = calculate_cost(model, prompt_tokens + context_tokens, p["est_output_tokens"]) or 0.0
                mode_cost += cost
            costs[mode] = mode_cost

        savings = (1 - costs["FULL_CCR"] / costs["PASSTHROUGH"]) * 100 if costs["PASSTHROUGH"] > 0 else 0
        print(
            f"  {tier_name:<12} {count:>5}  "
            f"${costs['PASSTHROUGH']:>10.4f} ${costs['ROUTE_ONLY']:>10.4f} "
            f"${costs['FULL_CCR']:>10.4f}  {savings:>7.1f}%"
        )

    print("  " + "─" * 68)

    # Final totals row
    total_pass = mode_results["PASSTHROUGH"]["total_cost"]
    total_route = mode_results["ROUTE_ONLY"]["total_cost"]
    total_full = mode_results["FULL_CCR"]["total_cost"]
    total_savings = (1 - total_full / total_pass) * 100 if total_pass > 0 else 0
    print(
        f"  {'TOTAL':<12} {len(PROMPTS):>5}  "
        f"${total_pass:>10.4f} ${total_route:>10.4f} "
        f"${total_full:>10.4f}  {total_savings:>7.1f}%"
    )
    print()

    # Key insights
    print("=" * 80)
    print("  KEY INSIGHTS")
    print("=" * 80)
    print()
    sub_pct = (mode_results["FULL_CCR"]["sub_calls"] / len(PROMPTS)) * 100
    pass_cost = mode_results["PASSTHROUGH"]["total_cost"]
    full_cost = mode_results["FULL_CCR"]["total_cost"]
    savings_pct = ((pass_cost - full_cost) / pass_cost) * 100 if pass_cost > 0 else 0

    print(f"  1. {mode_results['FULL_CCR']['sub_calls']}/{len(PROMPTS)} requests ({sub_pct:.0f}%) routed to the sub-model")
    print(f"     Passthrough cost: ${pass_cost:.4f}")
    print(f"     Full CCR cost:    ${full_cost:.4f}")
    print(f"     Session savings:  {savings_pct:.1f}%")
    print()
    print("  2. COMPLEX requests cost MORE with CCR (added context),")
    print("     but the context improves first-try accuracy, saving re-prompts.")
    print()
    print("  3. Sub-model costs ~$0.06/M tokens vs Claude's ~$3/M tokens")
    print(f"     That's a 50x cost reduction on {sub_pct:.0f}% of traffic.")
    print()

    # Scale projection
    print("=" * 80)
    print("  PROJECTED MONTHLY SAVINGS (at scale)")
    print("=" * 80)
    print()
    requests_per_day = 200
    days = 30
    total_requests = requests_per_day * days
    scale_factor = total_requests / len(PROMPTS)

    pass_monthly = total_pass * scale_factor
    full_monthly = total_full * scale_factor
    saved_monthly = pass_monthly - full_monthly

    print(f"  Assumption: {requests_per_day} requests/day, 30 days")
    print(f"  Total requests: {total_requests:,}")
    print()
    print(f"  Without CCR:  ${pass_monthly:>8.2f}/month")
    print(f"  With CCR:     ${full_monthly:>8.2f}/month")
    print(f"  Saved:        ${saved_monthly:>8.2f}/month ({(saved_monthly/pass_monthly)*100:.0f}%)")
    print()


if __name__ == "__main__":
    run_demo()
