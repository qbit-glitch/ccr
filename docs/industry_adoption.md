# CCR Industry Adoption Posture

CCR's product direction is Git-style memory infrastructure for coding agents:
local-first, inspectable, versioned, evidence-backed, and agent-agnostic.

## Implemented Adoption Surface

### Evidence And Evals

- `gcc_recall(query=...)` returns an answer, confidence, evidence IDs, stale notes, and conflict notes.
- `gcc_facts(...)` stores temporal facts with `observed_at`, validity fields, supersession, confidence, source, and pinning.
- `gcc_conflicts(...)` detects active contradictory facts.
- `ccr memory-eval` produces text, JSON, and HTML reports for citation and abstention checks.

### Memory Tiers

CCR now exposes the operational tier model:

```text
scratchpad -> session -> commit -> fact -> pattern -> playbook
```

Use:

```bash
ccr tiers --project .
```

### Governance

Local governance is intentionally file-based:

- `.ccr/governance.json` stores roles, retention, redaction, approved tools, and project boundary.
- `.ccr/governance_audit.jsonl` is append-only and hash-chained.
- `ccr governance scan` checks CCR text memory for common secrets and PII.
- `ccr export --redacted` redacts secrets/PII before export.

This maps to the practical security questions buyers ask first: read/write
control, leak detection, auditability, and deletion/retention posture.

### Observability

CCR has optional OpenTelemetry-compatible spans and local JSONL traces.

Environment switches:

```bash
CCR_OTEL_ENABLED=1     # emit OpenTelemetry spans when opentelemetry is installed
CCR_TRACE_LOCAL=1      # append local spans to .ccr/traces.jsonl
```

Current span names include:

- `ccr.memory.write.commit`
- `ccr.memory.read.context`
- `ccr.memory.recall`
- `ccr.memory.retrieve.*`
- `ccr.memory.rerank`

### Rerankers

Recall has a provider layer. SQLite/FTS remains the default retrieval base, and
reranking is optional:

```bash
CCR_RERANKER=lexical
CCR_RERANKER=fastembed
CCR_RERANKER=sentence-transformers
CCR_RERANKER=bge-reranker
CCR_RERANKER=ollama
```

Unavailable optional providers fall back to lexical reranking.

### Memory Browser

Generate an inspectable local UI:

```bash
ccr browser --project .
```

It writes `.ccr/browser/index.html` with timeline, tier counts, facts, stale
facts, and conflicts.

### Cross-Agent Event Contract

CCR records canonical events in `.ccr/agent_events.jsonl`:

```text
AgentEvent:
  agent
  session_id
  project
  event_type
  user_intent
  files_touched
  tools_used
  outcome
  memory_candidates
```

Validate event streams with:

```bash
ccr governance events-validate .ccr/agent_events.jsonl
```

### Local-First Sync

Git-backed sync is optional and local-first:

```bash
ccr sync init .
ccr sync push .
ccr sync pull .
ccr sync resolve .
```

Without a remote, this still creates a local Git-backed sync repo under
`.ccr/sync/repo`.

### Reliability And Ops

```bash
ccr doctor --enterprise
ccr backup
ccr restore <backup.zip>
ccr verify
ccr repair
ccr migrate
ccr export facts . --redacted
```

Backups include a SHA-256 manifest next to the zip.

### Enterprise Gateway Mode

The first gateway slice is a local policy gate:

```bash
ccr enterprise-gateway init .
ccr enterprise-gateway check . --tool gcc_commit --role writer
```

It validates approved tools, actor roles, project boundaries, and proposed
memory writes for high-severity secret findings.

## Compliance Mapping

CCR is not claiming certification. The current posture maps implementation
surfaces to common review frameworks:

- NIST AI RMF / GenAI Profile: governance policy, evaluation reports, audit trails, and incident-visible findings.
- OWASP LLM Top 10: secret/PII redaction, evidence-backed recall, provenance, and local-first controls.
- OWASP Agentic Skills Top 10: approved-tool policy, agent event schema, and auditable memory writes.
- MCP authorization guidance: local policy boundary now exists; external identity and token enforcement remain future enterprise-gateway work.

## Windows And Distribution

New code in this slice uses Python standard library path APIs and avoids
platform-specific shell behavior. Before release, Windows CI should run:

```bash
python -m pytest tests/unit/test_enterprise_features.py tests/unit/test_memory_recall_eval.py
python -m ccr.cli doctor --enterprise .
python -m ccr.cli memory-eval --project . --suite temporal
```

Distribution targets remain: PyPI, uvx, Homebrew, Docker, and direct source
install. Windows support should be treated as release-blocking for enterprise
adoption.
