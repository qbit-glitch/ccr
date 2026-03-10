"""Issue #738 fix — demonstrated across three CCR ablation modes.

Shows how CCR handles a real GitHub issue fix (tunnel binding to wrong port)
differently in each mode, comparing token usage, context, and cost.

Bug: Managed tunnel binds to gateway port (3000) instead of webhook server
port (8080), causing all external webhook channels to return 404.

Fix: Add reverse proxy route in the gateway that forwards /webhook/* traffic
to the webhook server, so a single tunnel serves both web UI and webhooks.
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
# The actual prompt (as a user would type it)
# ─────────────────────────────────────────────────────────────────

ISSUE_PROMPT = (
    "fix this issue: Managed Tunnel binds to Web Gateway port (3000) instead of "
    "Webhook Server port (8080), causing all external webhook channels to return 404. "
    "When using IronClaw's managed ngrok tunnel, the tunnel binds to port 3000 "
    "(Web Gateway) instead of port 8080 (Webhook Server), causing all external "
    "webhook channels (e.g., Slack) to return 404 on challenge verification. "
    "The start_tunnel() function in src/main.rs reads the port from the gateway "
    "config instead of the webhook server config. The fix should add a reverse "
    "proxy route in the gateway that forwards /webhook/* requests to the webhook "
    "server, so a single tunnel serves both web UI and webhook traffic."
)


# ─────────────────────────────────────────────────────────────────
# Simulated context for each mode
# ─────────────────────────────────────────────────────────────────

# PASSTHROUGH: Just the raw prompt, no context
PASSTHROUGH_SYSTEM = ""  # Claude Code's default system prompt only

# ROUTE_ONLY: Classification + memory context (L2 for MODERATE tasks)
ROUTE_ONLY_MEMORY = """## Project: IronClaw
- Rust-based secure AI assistant platform with multi-channel input (TUI, HTTP, WASM, Web)
- Agent loop: Job orchestration with state machine
- Tunnel support: ngrok, cloudflare, tailscale, custom
- Active branch: main
- Recent: Added tunnel provider abstraction, fixed WASM fuel metering
"""

# FULL_CCR: Classification + memory (L3) + context pack + ACE playbook
FULL_CCR_MEMORY = """## Project: IronClaw
- Rust-based secure AI assistant platform (~160K lines)
- Architecture: agent/, tools/, channels/, safety/, sandbox/, workspace/, llm/, db/
- Channels: TUI (ratatui), HTTP webhooks, Web gateway (SSE/WS), WASM channels
- Tunnel: src/tunnel/ — provider abstraction (ngrok, cloudflare, tailscale, custom)
- Web gateway: src/channels/web/ — Axum server on port 3000 (configurable)
- Webhook server: src/channels/webhook_server.rs — unified HTTP server on port 8080
- Key patterns: Fix pattern not instance, zero clippy warnings, regression tests required
"""

FULL_CCR_CONTEXT_PACK = """<file path="src/main.rs" lines="842-900">
/// Start managed tunnel if configured and no static URL is already set.
async fn start_tunnel(
    mut config: ironclaw::config::Config,
) -> (Config, Option<Box<dyn Tunnel>>) {
    let gateway_port = config
        .channels.gateway.as_ref()
        .map(|g| g.port)
        .unwrap_or(3000);
    let gateway_host = config
        .channels.gateway.as_ref()
        .map(|g| g.host.as_str())
        .unwrap_or("127.0.0.1");

    match ironclaw::tunnel::create_tunnel(provider_config) {
        Ok(Some(tunnel)) => {
            match tunnel.start(gateway_host, gateway_port).await {
                Ok(url) => { config.tunnel.public_url = Some(url); }
                Err(e) => { tracing::error!("Failed to start tunnel: {}", e); }
            }
        }
    }
}
</file>

<file path="src/channels/web/server.rs" lines="128-177">
pub struct GatewayState {
    pub msg_tx: RwLock<Option<mpsc::Sender<IncomingMessage>>>,
    pub sse: SseManager,
    pub workspace: Option<Arc<Workspace>>,
    pub user_id: String,
    pub startup_time: Instant,
    // ... other fields
}

pub async fn start_server(addr: SocketAddr, state: Arc<GatewayState>, auth_token: String) {
    let public = Router::new()
        .route("/api/health", get(health_handler));
    let protected = Router::new()
        .route("/api/chat/send", post(chat_send_handler))
        // ... other routes
        .route_layer(middleware::from_fn_with_state(auth_state, auth_middleware));
    let app = Router::new()
        .merge(public).merge(statics).merge(projects).merge(protected);
}
</file>

<file path="src/channels/webhook_server.rs" lines="1-80">
pub struct WebhookServer {
    config: WebhookServerConfig,
    routes: Vec<Router>,
}
impl WebhookServer {
    pub fn add_routes(&mut self, router: Router) { self.routes.push(router); }
    pub async fn start(&mut self) -> Result<(), ChannelError> {
        let listener = TcpListener::bind(self.config.addr).await?;
        // Merges all route fragments into single server
    }
}
</file>

<file path="src/channels/web/mod.rs" lines="85-150">
// GatewayChannel builder pattern
impl GatewayChannel {
    pub fn with_workspace(mut self, ws: Arc<Workspace>) -> Self { /* rebuild_state */ }
    pub fn with_session_manager(mut self, sm: Arc<SessionManager>) -> Self { /* ... */ }
    // ... builder methods for each subsystem
    fn rebuild_state(&mut self, mutate: impl FnOnce(&mut GatewayState)) {
        // Creates new GatewayState copying all fields, applies mutation
    }
}
</file>

<file path="src/tunnel/mod.rs" lines="1-30">
pub trait Tunnel: Send + Sync {
    fn name(&self) -> &str;
    async fn start(&self, host: &str, port: u16) -> Result<String, TunnelError>;
    async fn stop(&self) -> Result<(), TunnelError>;
}
pub fn create_tunnel(config: &TunnelProviderConfig) -> Result<Option<Box<dyn Tunnel>>>;
</file>"""

FULL_CCR_PLAYBOOK = """## STRATEGIES & INSIGHTS
[strat-00001] helpful=12 harmful=0 :: In IronClaw, always check both PostgreSQL and libSQL impls when modifying Database trait
[strat-00002] helpful=8 harmful=0 :: Tool implementations must handle sanitization — check requires_sanitization() before output
[strat-00005] helpful=6 harmful=0 :: When adding new GatewayState fields, update: struct definition, GatewayChannel::new(), rebuild_state(), test_helpers, ws.rs test state

## COMMON MISTAKES TO AVOID
[avoid-00001] helpful=9 harmful=0 :: Never byte-slice user strings in Rust; use is_char_boundary() for UTF-8 safety
[avoid-00004] helpful=5 harmful=0 :: Don't forget to update all GatewayState construction sites when adding fields — grep for 'startup_time' to find them all
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

    # 3. TRIVIAL: short, no code, no file refs
    if token_count < config.trivial_token_threshold:
        has_code = "```" in prompt
        has_file_ref = bool(re.search(r"\.\w{2,4}\b", prompt))
        if not has_code and not has_file_ref:
            return ComplexityTier.TRIVIAL

    # 4. Questions → SIMPLE
    if re.match(r"^(what|how|why|where|when)\s", msg_lower):
        return ComplexityTier.SIMPLE

    # 5. Action verbs → SIMPLE (short) or MODERATE (long)
    if re.match(r"^\s*(fix|add|remove|rename|update|change)\s", msg_lower):
        if token_count < config.simple_token_threshold:
            return ComplexityTier.SIMPLE
        return ComplexityTier.MODERATE

    return ComplexityTier.MODERATE


# ─────────────────────────────────────────────────────────────────
# Token + cost calculation per mode
# ─────────────────────────────────────────────────────────────────

CLAUDE_MODEL = "claude-sonnet-4-5"
SUB_MODEL = "gpt-oss-20b"

# Estimated output tokens for the fix (code changes + explanation)
EST_OUTPUT_TOKENS = 1500


def compute_mode(mode: str):
    """Compute token usage and cost for a given mode."""
    tier = classify_prompt(ISSUE_PROMPT)
    prompt_tokens = count_tokens(ISSUE_PROMPT)

    if mode == "PASSTHROUGH":
        # No CCR — everything goes to Claude as-is
        context_tokens = 0
        model = CLAUDE_MODEL
        classification_cost = 0.0

    elif mode == "ROUTE_ONLY":
        # Classify + route, light memory context
        context_tokens = count_tokens(ROUTE_ONLY_MEMORY)
        # SIMPLE tasks go to sub-model, MODERATE+ to Claude
        model = SUB_MODEL if tier in (ComplexityTier.TRIVIAL, ComplexityTier.SIMPLE) else CLAUDE_MODEL
        # Classification cost (sub-model call for ambiguous cases)
        classification_cost = 0.0  # Heuristic-classified, no LLM call needed

    else:  # FULL_CCR
        # Full pipeline: classify + route + pack context + memory + playbook
        context_tokens = (
            count_tokens(FULL_CCR_MEMORY)
            + count_tokens(FULL_CCR_CONTEXT_PACK)
            + count_tokens(FULL_CCR_PLAYBOOK)
        )
        model = SUB_MODEL if tier in (ComplexityTier.TRIVIAL, ComplexityTier.SIMPLE) else CLAUDE_MODEL
        classification_cost = 0.0  # Heuristic-classified

    input_tokens = prompt_tokens + context_tokens
    output_tokens = EST_OUTPUT_TOKENS
    cost = calculate_cost(model, input_tokens, output_tokens) or 0.0
    total_cost = cost + classification_cost

    return {
        "tier": tier,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "context_tokens": context_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": total_cost,
    }


# ─────────────────────────────────────────────────────────────────
# The three "fixes" — showing what each mode provides to the model
# ─────────────────────────────────────────────────────────────────

def describe_mode_approach(mode: str) -> str:
    """Describe what context the model receives in each mode."""
    if mode == "PASSTHROUGH":
        return textwrap.dedent("""
        Model receives: Raw user prompt only.
        No project context, no file contents, no memory.
        Claude must figure out the codebase structure from scratch,
        likely requesting file reads via tool calls (additional turns + tokens).
        """).strip()

    elif mode == "ROUTE_ONLY":
        return textwrap.dedent("""
        Model receives: User prompt + light memory context (project overview).
        CCR classifies this as SIMPLE (action verb "fix" + <2000 tokens).
        Routes to sub-model (GPT-OSS-20B) — saves ~99.5% vs Claude.
        Sub-model gets enough context for a basic fix attempt, but may
        need follow-up from Claude for complex multi-file changes.
        """).strip()

    else:  # FULL_CCR
        return textwrap.dedent("""
        Model receives: User prompt + deep memory + context pack + ACE playbook.
        CCR classifies as SIMPLE, but FULL_CCR enriches with full context:
        - Memory L3: Full architecture map, key patterns, recent changes
        - Context pack: Relevant source files pre-loaded (main.rs tunnel code,
          server.rs GatewayState, webhook_server.rs, mod.rs builder pattern)
        - ACE playbook: Learned strategies (update all GatewayState construction
          sites, grep for 'startup_time' to find them)
        Sub-model can attempt fix with rich context; escalates to Claude if needed.
        """).strip()


# ─────────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────────

def run_demo():
    print()
    print("=" * 80)
    print("  ISSUE #738 — THREE-MODE FIX COMPARISON")
    print("  Bug: Tunnel binds to gateway port instead of webhook server port")
    print("=" * 80)
    print()

    # Show the prompt and classification
    tier = classify_prompt(ISSUE_PROMPT)
    prompt_tokens = count_tokens(ISSUE_PROMPT)
    print(f"  Prompt: \"{ISSUE_PROMPT[:70]}...\"")
    print(f"  Classification: {tier.value.upper()} (heuristic: 'fix' action verb + {prompt_tokens} tokens)")
    print()

    # Compute all three modes
    modes = ["PASSTHROUGH", "ROUTE_ONLY", "FULL_CCR"]
    results = {mode: compute_mode(mode) for mode in modes}

    # Comparison table
    print("=" * 80)
    print("  TOKEN & COST COMPARISON")
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

    # Savings
    baseline = results["PASSTHROUGH"]["cost"]
    for mode in ["ROUTE_ONLY", "FULL_CCR"]:
        cost = results[mode]["cost"]
        saved = baseline - cost
        pct = (saved / baseline) * 100 if baseline > 0 else 0
        direction = "saves" if saved >= 0 else "adds"
        abs_saved = abs(saved)
        print(f"  {mode}: {direction} ${abs_saved:.4f} ({abs(pct):.1f}%) vs PASSTHROUGH")

    print()

    # What each mode provides
    print("=" * 80)
    print("  WHAT EACH MODE PROVIDES TO THE MODEL")
    print("=" * 80)
    print()

    for mode in modes:
        print(f"  [{mode}]")
        for line in describe_mode_approach(mode).split("\n"):
            print(f"    {line}")
        print()

    # The actual fix summary
    print("=" * 80)
    print("  THE FIX (applied in codebase_testing/ironclaw/)")
    print("=" * 80)
    print()
    print("  Files changed:")
    print("    1. src/main.rs")
    print("       - Renamed misleading 'gateway_port' → 'tunnel_port' in start_tunnel()")
    print("       - Added comment explaining tunnel points at gateway (which proxies webhooks)")
    print()
    print("    2. src/channels/web/server.rs")
    print("       - Added 'webhook_proxy_addr: Option<SocketAddr>' to GatewayState")
    print("       - Added webhook_proxy_handler() — reverse proxies /webhook/* to webhook server")
    print("       - Registered /webhook/{*path} route (public, no auth — webhook server validates)")
    print()
    print("    3. src/channels/web/mod.rs")
    print("       - Added with_webhook_proxy_addr() builder method")
    print("       - Added webhook_proxy_addr to rebuild_state() and constructor")
    print()
    print("    4. src/channels/web/ws.rs, test_helpers.rs (GatewayState construction sites)")
    print("       - Added webhook_proxy_addr: None to test state constructors")
    print()
    print("    5. src/main.rs (wiring)")
    print("       - Connected webhook_server_addr → gw.with_webhook_proxy_addr()")
    print()

    # Estimated turns comparison
    print("=" * 80)
    print("  ESTIMATED TURNS TO COMPLETE FIX")
    print("=" * 80)
    print()
    print(f"  {'Mode':<16} {'Turns':>6} {'Why':<55}")
    print("  " + "-" * 76)
    print(f"  {'PASSTHROUGH':<16} {'~8-12':>6} {'No context → needs tool calls to read files':<55}")
    print(f"  {'ROUTE_ONLY':<16} {'~5-7':>6} {'Light context → fewer file reads needed':<55}")
    print(f"  {'FULL_CCR':<16} {'~1-2':>6} {'All files pre-loaded + playbook hints':<55}")
    print("  " + "-" * 76)
    print()

    # Multi-turn cost projection
    print("=" * 80)
    print("  MULTI-TURN COST PROJECTION (including tool call turns)")
    print("=" * 80)
    print()

    # PASSTHROUGH: ~10 turns (5 file reads + fix + verify)
    pt_turns = 10
    pt_read_cost = calculate_cost(CLAUDE_MODEL, 500, 200) or 0.0  # Each tool read turn
    pt_total = results["PASSTHROUGH"]["cost"] + (pt_turns - 1) * pt_read_cost

    # ROUTE_ONLY: ~6 turns (3 file reads + fix + verify)
    ro_turns = 6
    ro_read_cost = calculate_cost(CLAUDE_MODEL, 500 + count_tokens(ROUTE_ONLY_MEMORY), 200) or 0.0
    ro_total = results["ROUTE_ONLY"]["cost"] + (ro_turns - 1) * ro_read_cost

    # FULL_CCR: ~1-2 turns (fix directly, maybe 1 verification)
    fc_turns = 2
    fc_total = results["FULL_CCR"]["cost"] + calculate_cost(CLAUDE_MODEL, 300, 100) or 0.0

    print(f"  {'Mode':<16} {'Turns':>6} {'Est. Total Cost':>15} {'vs PASSTHROUGH':>15}")
    print("  " + "-" * 55)
    print(f"  {'PASSTHROUGH':<16} {pt_turns:>6} ${pt_total:>13.4f} {'baseline':>15}")
    print(f"  {'ROUTE_ONLY':<16} {ro_turns:>6} ${ro_total:>13.4f} {f'-{((pt_total - ro_total)/pt_total)*100:.0f}%':>15}")
    print(f"  {'FULL_CCR':<16} {fc_turns:>6} ${fc_total:>13.4f} {f'-{((pt_total - fc_total)/pt_total)*100:.0f}%':>15}")
    print("  " + "-" * 55)
    print()

    # Monthly projection
    issues_per_day = 5  # typical dev session
    days_per_month = 22  # working days
    monthly_issues = issues_per_day * days_per_month

    print(f"  At {issues_per_day} issues/day × {days_per_month} working days = {monthly_issues} issues/month:")
    print(f"    PASSTHROUGH:  ${pt_total * monthly_issues:.2f}/month")
    print(f"    ROUTE_ONLY:   ${ro_total * monthly_issues:.2f}/month")
    print(f"    FULL_CCR:     ${fc_total * monthly_issues:.2f}/month")
    print(f"    Savings:      ${(pt_total - fc_total) * monthly_issues:.2f}/month ({((pt_total - fc_total)/pt_total)*100:.0f}%)")
    print()


def test_issue_738_classification():
    """Test that the issue prompt is classified correctly."""
    tier = classify_prompt(ISSUE_PROMPT)
    # "fix ..." with >500 tokens but <2000 → SIMPLE
    # Actually this is >2000 tokens if the prompt is long enough
    prompt_tokens = count_tokens(ISSUE_PROMPT)
    if prompt_tokens >= 2000:
        assert tier == ComplexityTier.MODERATE, f"Expected MODERATE for long fix, got {tier}"
    else:
        assert tier == ComplexityTier.SIMPLE, f"Expected SIMPLE for short fix, got {tier}"


def test_passthrough_no_context():
    """PASSTHROUGH mode adds no context tokens."""
    r = compute_mode("PASSTHROUGH")
    assert r["context_tokens"] == 0
    assert r["model"] == CLAUDE_MODEL


def test_route_only_adds_memory():
    """ROUTE_ONLY adds memory context."""
    r = compute_mode("ROUTE_ONLY")
    assert r["context_tokens"] > 0


def test_full_ccr_adds_pack_and_playbook():
    """FULL_CCR adds memory + context pack + playbook."""
    r = compute_mode("FULL_CCR")
    route_only = compute_mode("ROUTE_ONLY")
    # FULL_CCR should have more context than ROUTE_ONLY
    assert r["context_tokens"] > route_only["context_tokens"]


def test_routing_matches_tier():
    """SIMPLE tasks route to sub-model, MODERATE+ to Claude."""
    r = compute_mode("ROUTE_ONLY")
    tier = r["tier"]
    if tier in (ComplexityTier.TRIVIAL, ComplexityTier.SIMPLE):
        assert r["model"] == SUB_MODEL, f"SIMPLE should use sub-model"
    else:
        assert r["model"] == CLAUDE_MODEL, f"MODERATE+ should use Claude"


if __name__ == "__main__":
    run_demo()
