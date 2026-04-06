# CCR for Students and Researchers

CCR gives Claude Code persistent memory across sessions. Without it, Claude forgets
everything when you close the terminal. With it, Claude remembers your project's history,
decisions, and experiment results — even months later. Works with Claude Max ($20/mo),
no API keys needed.

---

## The Problem It Solves

**Without CCR — Monday morning:**
> You: "Continue working on my neural network training script."
> Claude: "I don't have context about your project. Can you share your code and explain your goals?"
> You: *[pastes 500 lines, re-explains architecture, re-describes last week's findings — ~3,000 tokens]*

**With CCR — Monday morning:**
> You: "Continue working on my neural network training script."
> Claude: *[already knows]* "Picking up from experiment exp-042 (C023). Last session: val_loss=0.23 at epoch 15. Next: try lr=1e-4 with cosine scheduler."
> *[~200 tokens of context instead of 3,000]*

---

## 3-Step Setup

```bash
# Step 1: Install CCR
pip install -e .   # from source (or: pip install ccr-memory when on PyPI)

# Step 2: Install hooks and register with Claude Code
ccr install        # writes .mcp.json + all 4 hooks automatically

# Step 3: Test it
# Open a new Claude Code session in your project and ask:
# "What do you remember about this project?"
# Then call: gcc_status()
```

That's it. Claude will now automatically load your project memory at the start of every session.

---

## 5 Commands You'll Actually Use

| Command | When to use |
|---------|-------------|
| `gcc_commit(title=..., what=..., why=..., files_changed=[], next_step=...)` | After finishing a task or experiment — saves your progress |
| `gcc_context(level=2)` | At session start — loads what Claude knew last session |
| `gcc_status` | Quick overview: active branch, recent milestones |
| `index_search(query="...")` | Find files or past decisions by description |
| `ace_get_playbook` | View strategies Claude has learned about your project |

### Minimal example for a research session:

```python
# At the start of a session (CCR does this automatically via hooks):
gcc_context(level=2)

# After completing work:
gcc_commit(
    title="Baseline BERT fine-tune on arxiv-cs",
    what="Fine-tuned bert-base-uncased for 3 epochs, batch=16",
    why="Establish baseline before trying LoRA",
    files_changed=["train.py", "configs/baseline.yaml"],
    next_step="Compare with LoRA r=16 run",
)
```

---

## Tracking Experiments

When running ML experiments, add an `experiment=` dict to capture structured results:

```python
gcc_commit(
    title="LoRA r=16 vs BERT baseline",
    what="Trained LoRA adapter for 3 epochs on arxiv-cs",
    why="Test if LoRA r=16 matches full fine-tune at 12% param count",
    files_changed=["train.py", "configs/lora_r16.yaml"],
    next_step="Try r=8 to reduce memory further",
    experiment={
        "id": "exp-042",
        "hypothesis": "LoRA r=16 matches full fine-tune performance",
        "metrics": {"val_loss": 0.23, "accuracy": 0.87, "params_pct": 12},
        "conclusion": "Confirmed — 98% of full FT perf at 12% param count",
    },
)
```

**Finding experiments later — use `gcc_experiments` (searches metric values, not just titles):**
```python
# Find all runs where val_loss < 0.3
gcc_experiments(metric_filter={"val_loss": {"lt": 0.3}})

# Compare two specific runs side-by-side
gcc_experiments(compare=["C041", "C053"])

# Best runs by accuracy (highest first)
gcc_experiments(top_n="accuracy:desc")

# All LoRA-related experiments
gcc_experiments(hypothesis_contains="LoRA")

# Unified search across everything (commits, experiments, discussions, sessions)
gcc_search("val_loss")
```

> **Note:** `gcc_context(level=5, search_term=...)` searches commit titles and text. Use `gcc_experiments` when you need to filter by metric values like `val_loss < 0.3`.

---

## PhD Researcher Workflow (3-Month Project)

### Month 1: Establishing baselines

```python
# Run 1: baseline
gcc_commit(title="Baseline run", ..., experiment={"id": "run-001", "metrics": {"val_loss": 0.45}})

# Run 2: first improvement
gcc_commit(title="LR schedule ablation", ..., experiment={"id": "run-002", "metrics": {"val_loss": 0.38}})
```

### Month 2: Iterating on hypotheses

Use `gcc_branch` to isolate a hypothesis without polluting main memory:
```python
gcc_branch(name="hypothesis-lora")
# ... run experiments, commit results ...
gcc_merge(source="hypothesis-lora", message="LoRA confirmed: 98% perf, 12% params")
```

### Month 3: Writing the paper

Claude reconstructs your decision chain from memory:
```python
# Find all experiment runs sorted by best val_loss
gcc_experiments(top_n="val_loss:asc")
# Returns a table: commit, date, hypothesis, metrics, conclusion

# Find decisions you logged about model architecture
gcc_discussions(search="architecture")

# Full-text search across commits + experiments + discussions + sessions
gcc_search("preprocessing decision")
# Returns: all places this topic appeared, grouped by source
```

---

## Token Cost Framing

| Scenario | Tokens per session start |
|----------|--------------------------|
| Without CCR (re-explaining project) | 3,000–10,000 |
| With CCR `gcc_context(level=2)` | ~200–500 |
| Savings over 30 sessions | ~90,000–285,000 tokens |

Claude Max ($20/mo) gives effectively unlimited usage in Claude Code. CCR's value isn't
reducing your bill — it's making every session productive from minute zero by eliminating
the "re-explain everything" tax.

---

## How Automatic Memory Capture Works

CCR captures your session in two layers — you don't need to remember to log anything:

**Layer 1 — Transcript reconciliation (always fires):**  
When your session ends, the Stop hook reads the Claude Code transcript file and inserts
any Q&A turns that weren't explicitly logged. Even if Claude forgets to call
`session_log_turn`, the turns are recovered from the transcript.

**Layer 2 — Auto-baseline commit (fires when no explicit commit was made):**  
If you end a session without calling `gcc_commit`, CCR creates a structural `[auto]`
commit summarizing what you did (files touched, user questions asked). This is a
guaranteed safety net — no session is completely lost.

> **When to call `gcc_commit` manually:** Auto-baseline creates a minimal record.
> For experiments with metrics, architecture decisions, or anything you'll want to
> *search* later, call `gcc_commit` with full context. Quality of retrieval depends
> on quality of commit messages.

**`ace_update_counters` requires your (or Claude's) judgment:**  
This tool marks strategies as helpful or harmful based on what worked in a session.
Because it requires evaluating outcomes, it can't be automated from hooks — call it
when you notice a pattern ("this approach kept causing bugs", "this template saved time").

---

## FAQ

**Q: Does CCR work without Claude Max?**
A: CCR requires Claude Code, which requires **Claude Max ($20/mo)**. This is distinct
from Claude Pro ($20/mo) — Claude Pro is for claude.ai chat, while Claude Max is for
Claude Code (the terminal CLI). If you can't afford $20/mo, CCR also works with an
Anthropic API key:

```bash
# API key path (pay-per-token, but you control costs):
export ANTHROPIC_API_KEY=sk-ant-...
claude  # Claude Code opens with API key billing
```

At typical student usage (1-2 hours/day), API costs are roughly $2-8/month. Claude Max
is better value for heavy users (unlimited within fair-use caps).

> **Note on global pricing:** $20/mo is US-priced. In purchasing-power-parity terms,
> this can be $40-80/mo equivalent in many countries. The API-key path above is the
> most accessible for budget-constrained researchers.

**Q: Will Claude automatically commit my progress?**
A: Yes, if you ran `ccr install`. The Stop hook auto-commits when you end the session.
See "How Automatic Memory Capture Works" above for details on the two-layer capture
system. Call `gcc_commit` manually for finer-grained control (especially for experiments).

**Q: How much memory does CCR keep?**
A: CCR stores all commits indefinitely in `.ccr/` (plain markdown files). The rolling
summary (what Claude loads by default) is capped at 1,500 chars to stay concise. When it
gets long, CCR warns you and asks for a compressed summary.

**Q: Can I use CCR for a non-coding research project (papers, writing)?**
A: Yes. Just use any folder as your "project". Track drafts, feedback, and decisions with
`gcc_commit`. The `files_changed` field can list document names.

**Q: Is my research data private?**
A: All data lives in `.ccr/` inside your project folder. Nothing is sent to external
servers. CCR is pure local storage + the Claude Code session you already have.
