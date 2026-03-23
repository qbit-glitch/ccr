"""CCR MCP Server — exposes GCC memory, ACE playbook, RLM sandbox, and repo index as tools.

Claude Code connects to this server and gains persistent memory, self-evolving
playbooks, a sandboxed REPL, and repo search — all without a sub-model or API keys.

Usage:
    python -m ccr.mcp_server              # stdio transport (for Claude Code)
    python -m ccr.mcp_server --project .  # explicit project root
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ccr.ace.playbook import DeltaOperation, FailureLesson, Playbook, parse_delta_operations
from ccr.context.indexer import RepoIndex
from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig, PlaybookSchema, SchemaMetrics
from ccr.rlm.repl import CCRRepl
from ccr.utils.parsing import extract_json_string

# ---------------------------------------------------------------------------
# Globals — initialized once at startup
# ---------------------------------------------------------------------------

# M3: Module-level lock for thread safety on global state mutations
_state_lock = threading.Lock()

_project_root: str = ""
_memory: MemoryManager | None = None
_playbook: Playbook | None = None
_playbook_path: str = ""
_failure_lessons_path: str = ""
_global_playbook: Playbook | None = None
_global_playbook_path: str = ""
_global_failure_lessons_path: str = ""
_repo_index: RepoIndex | None = None
_repl: CCRRepl | None = None
_schema_path: str = ""
_global_schema_path: str = ""
_embedding_model: object | None = None  # EmbeddingModel when available
_embeddings_path: str = ""
_chunk_embeddings_path: str = ""

mcp = FastMCP(
    "ccr",
    instructions=(
        "CCR gives you persistent project memory (GCC), self-evolving strategy "
        "playbooks (ACE), a sandboxed Python REPL (RLM), and repo indexing. "
        "Use gcc_* tools for memory, ace_* for playbook management, rlm_* for "
        "sandboxed code execution, and index_* for repo search."
    ),
)


def _get_sub_client() -> object | None:
    """Return a sub-model client, or None when no backend is configured.

    Priority order (first available wins):
      1. Ollama (CCR_OLLAMA_MODEL env var, e.g. "qwen2.5:7b") — free, local
      2. Anthropic Haiku (ANTHROPIC_API_KEY_SUB or ANTHROPIC_API_KEY)

    Graceful no-op: when nothing is configured, all Phase 2 features are silently skipped.
    """
    ollama_model = os.environ.get("CCR_OLLAMA_MODEL")
    if ollama_model:
        try:
            from ccr.models.openai_compat import OpenAICompatClient  # noqa: PLC0415
            return OpenAICompatClient(
                model_name=ollama_model,
                base_url=os.environ.get("CCR_OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key="ollama",
                max_tokens=1024,
            )
        except Exception:
            pass

    api_key = os.environ.get("ANTHROPIC_API_KEY_SUB") or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from ccr.models.anthropic_client import ClaudeClient  # noqa: PLC0415
            return ClaudeClient(api_key=api_key, model_name="claude-haiku-4-5-20251001", max_tokens=1024)
        except Exception:
            pass

    return None


def _init(project_root: str | None = None) -> None:
    """Initialize all subsystems for the given project root."""
    global _project_root, _memory, _playbook, _playbook_path, _failure_lessons_path
    global _global_playbook, _global_playbook_path, _global_failure_lessons_path, _repo_index
    global _schema_path, _global_schema_path
    global _embedding_model, _embeddings_path, _chunk_embeddings_path

    _project_root = os.path.abspath(project_root or os.getcwd())
    _memory = MemoryManager(_project_root, CCRConfig())
    _memory.ensure_structure()

    # Wire optional sub-model (Phase 2): activates GCC LLM rolling summary + ACE synthesis
    sub = _get_sub_client()
    if sub is not None:
        _memory.set_sub_client(sub)

    # Clear session marker so hooks detect new session on first prompt
    session_marker = os.path.join(_project_root, ".ccr", ".session_active")
    if os.path.isfile(session_marker):
        try:
            os.remove(session_marker)
        except OSError:
            pass

    _playbook_path = os.path.join(_project_root, ".ccr", "playbook.txt")
    _failure_lessons_path = os.path.join(_project_root, ".ccr", "failure_lessons.json")
    _playbook = _load_playbook()

    # Initialize global playbook (~/.ccr/)
    global_ccr = os.path.expanduser("~/.ccr")
    os.makedirs(global_ccr, exist_ok=True)
    _global_playbook_path = os.path.join(global_ccr, "global_playbook.txt")
    _global_failure_lessons_path = os.path.join(global_ccr, "global_failure_lessons.json")
    _global_playbook = _load_global_playbook()

    # Schema paths (MCE-inspired schema evolution)
    _schema_path = os.path.join(_project_root, ".ccr", "playbook_schema.json")
    _global_schema_path = os.path.join(global_ccr, "global_playbook_schema.json")

    # Embeddings paths (A-RAG semantic search)
    _embeddings_path = os.path.join(_project_root, ".ccr", "index_embeddings.json.gz")
    _chunk_embeddings_path = os.path.join(_project_root, ".ccr", "index_chunk_embeddings.json.gz")

    # Build repo index
    _repo_index = RepoIndex.build(_project_root)

    # Load cached embeddings if available
    if os.path.isfile(_embeddings_path):
        _repo_index.load_embeddings(_embeddings_path)

    # Load cached chunk embeddings if available (A-RAG §3.1 sentence-level chunks)
    if os.path.isfile(_chunk_embeddings_path):
        _repo_index.load_chunk_embeddings(_chunk_embeddings_path)

    # Cache index
    try:
        _memory.save_index(_repo_index.to_json())
    except Exception:
        pass


def _atomic_write(path: str, content: str) -> None:
    """Write content to path atomically via tmp + fsync + os.replace (H5)."""
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_playbook() -> Playbook:
    """Load playbook from disk or create empty. Also loads failure lessons."""
    if os.path.isfile(_playbook_path):
        with open(_playbook_path, "r", encoding="utf-8") as f:
            pb = Playbook(f.read())
    else:
        pb = Playbook()
    # Load structured failure lessons from companion JSON
    if _failure_lessons_path:
        pb.load_failure_lessons(_failure_lessons_path)
    return pb


def _save_playbook() -> None:
    """Persist playbook and failure lessons to disk (H5: atomic writes)."""
    if _playbook is not None:
        _atomic_write(_playbook_path, _playbook.serialize())
        # Save failure lessons to companion JSON
        if _failure_lessons_path:
            _playbook.save_failure_lessons(_failure_lessons_path)


def _load_global_playbook() -> Playbook:
    """Load global playbook from ~/.ccr/ or create empty."""
    if os.path.isfile(_global_playbook_path):
        with open(_global_playbook_path, "r", encoding="utf-8") as f:
            pb = Playbook(f.read())
    else:
        pb = Playbook()
    if _global_failure_lessons_path:
        pb.load_failure_lessons(_global_failure_lessons_path)
    return pb


def _save_global_playbook() -> None:
    """Persist global playbook to ~/.ccr/ (H5: atomic writes)."""
    if _global_playbook is not None:
        _atomic_write(_global_playbook_path, _global_playbook.serialize())
        if _global_failure_lessons_path:
            _global_playbook.save_failure_lessons(_global_failure_lessons_path)


def _load_schema(path: str) -> PlaybookSchema:
    """Load schema from JSON or return default (MCE schema persistence)."""
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            return PlaybookSchema.from_dict(data.get("current", {}))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return PlaybookSchema.default()


def _load_schema_history(path: str) -> list[dict]:
    """Load schema version history from JSON."""
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            return data.get("history", [])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return []


def _save_schema(schema: PlaybookSchema, history: list[dict], path: str) -> None:
    """Save schema + version history to JSON via atomic write."""
    data = {
        "current": schema.to_dict(),
        "history": history,
        "next_version": schema.version + 1,
    }
    _atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False))


def _ensure_global_playbook() -> Playbook:
    if _global_playbook is None:
        _init()
    assert _global_playbook is not None
    return _global_playbook


def _ensure_memory() -> MemoryManager:
    if _memory is None:
        _init()
    assert _memory is not None
    return _memory


def _ensure_playbook() -> Playbook:
    if _playbook is None:
        _init()
    assert _playbook is not None
    return _playbook


def _ensure_index() -> RepoIndex:
    if _repo_index is None:
        _init()
    assert _repo_index is not None
    return _repo_index


def _extract_patterns_from_commit(
    title: str, what: str, why: str, files_changed: list[str], sub_client
) -> list[str]:
    """Auto-extract transferable patterns from a commit via sub-model (CER §3.2).

    Returns "When X, do Y" pattern strings, or [] on any error.
    Failures are transparent and never block the commit path.
    """
    try:
        files_str = ", ".join(files_changed[:5])
        prompt = (
            "Extract 1-2 transferable patterns from this commit for future reference.\n\n"
            f"Commit: {title}\n"
            f"What: {what}\n"
            f"Why: {why}\n"
            f"Files: {files_str}\n\n"
            'Write patterns as abstract "When X, do Y" rules that would help in similar future situations.\n'
            'Respond with a JSON array of strings: ["When X, do Y", "When A, do B"]\n'
            "Be concise. Each pattern should be 1 sentence."
        )
        response = sub_client.completion(prompt)
        raw = extract_json_string(response)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(p) for p in parsed if p]
        return []
    except Exception:
        return []


# ===========================================================================
# GCC Memory Tools
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_commit(
    title: str,
    what: str,
    why: str,
    files_changed: list[str],
    next_step: str,
    patterns_learned: list[str] | None = None,
    admission_threshold: float = 0.85,
    rejection_threshold: float = 0.0,
    compressed_summary: str | None = None,
) -> str:
    """Commit progress to project memory.

    Creates a structured commit record in .ccr/ with what was done, why,
    which files changed, and what to do next. Builds a rolling summary
    that progressively captures the full project history.

    Use after meaningful progress — completing a feature, fixing a bug,
    reaching a milestone, or before context gets large.

    Admission control (A-MAC Algorithm 1, correct polarity):
    1. Computes S(m) and S(m_conflict) — admission scores (higher = more valuable)
    2. Novelty N(m) = 1 - max similarity across k=5 recent commits (Eq. 3)
    3. Similarity is recency-modulated: old conflicts dampened via R = exp(-0.01·hours) (Eq. 4)
    4. If S(m) < rejection_threshold → reject (low value, correct polarity)
    5. If similarity >= admission_threshold → FindConflict found
       - If S(m) > S(m_conflict) → create new (new outranks existing)
       - If S(m) <= S(m_conflict) → merge into existing
    6. Otherwise → create new commit

    Rolling summary compression (GCC paper G4 fix):
    When the rolling summary exceeds 1200 chars, the return value includes a
    compression prompt. To restore the GCC paper's S_t = f(S_{t-1}, D_t)
    property, compress the summary and pass it back via compressed_summary on
    your next gcc_commit call. This two-call pattern mirrors gcc_consolidate's
    project tier — Claude Code acts as the LLM that compresses the summary.

    Args:
        patterns_learned: Optional list of transferable patterns/skills observed
            during this task (CER-inspired, arXiv:2506.06698). Abstract and
            parameterized, e.g., "When adding {feature_type}, update tests +
            docs + CLAUDE.md together". Patterns are deduped against the existing
            buffer and tracked across commits. Recurring patterns (3+ commits)
            are suggested for ACE playbook promotion.
        admission_threshold: Similarity score (0-1) above which FindConflict
            detects a conflict. Default 0.85 per paper §3.3. Set to 1.0 to disable.
        rejection_threshold: Admission score below which commits are rejected.
            Default 0.0 (disabled). Low score = low value = reject.
        compressed_summary: Optional LLM-compressed rolling summary. When
            provided, replaces the mechanical rolling summary entirely. Use this
            to respond to the compression prompt — compress the current summary
            into a concise paragraph capturing project context, key milestones,
            and current direction. Max 1500 chars.
    """
    try:
        with _state_lock:
            mem = _ensure_memory()
            # Phase 2: Auto-extract patterns via sub-model when not provided (CER §3.2)
            if patterns_learned is None and mem.config.auto_extract_patterns and len(what) > 100:
                sub = _get_sub_client()
                if sub is not None:
                    patterns_learned = _extract_patterns_from_commit(
                        title, what, why, files_changed or [], sub
                    ) or None
            result = mem.commit(title, what, why, files_changed, next_step,
                                patterns_learned,
                                admission_threshold, rejection_threshold,
                                compressed_summary)

        # ACE §3.1-3.3: Generator → Reflector → Curator pipeline (fire-and-forget)
        sub = _get_sub_client()
        if sub is not None and result and not result.startswith("Error"):
            _run_ace_pipeline(title, what, why, sub)

        return result
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_branch(name: str, purpose: str, hypothesis: str) -> str:
    """Create an exploration branch for experimental work.

    Branches isolate experimental changes from the main line. Must be on
    main to create a branch. Use kebab-case names (e.g., 'try-new-parser').

    Args:
        name: Branch name in kebab-case.
        purpose: What this branch explores.
        hypothesis: What you expect to learn or achieve.
    """
    try:
        with _state_lock:
            mem = _ensure_memory()
            return mem.create_branch(name, purpose, hypothesis)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False))
def gcc_merge(branch: str, outcome: str, conclusion: str) -> str:
    """Merge an exploration branch back into main.

    Must be on the branch being merged. Integrates the branch's rolling
    summary and OTA log into main.

    Args:
        branch: Branch name to merge.
        outcome: One of 'success', 'failure', or 'partial'.
        conclusion: Summary of what was learned.
    """
    try:
        with _state_lock:
            mem = _ensure_memory()
            return mem.merge(branch, outcome, conclusion)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_context(
    level: int = 2,
    search_term: str | None = None,
    commit_id: str | None = None,
    log_window: int = 0,
    follow_links: bool = False,
) -> str:
    """Retrieve project memory at the specified depth.

    Levels:
        1: Project overview only (~200 tokens)
        2: + rolling summary + last 3 commits
        3: + branch summary (purpose/hypothesis/conclusion)
        4: + last 10 commits
        5: + specific commit by ID or keyword search + cross-links

    Use level 2 at session start for grounding. Use level 5 with
    search_term to find specific past work.

    Args:
        level: Depth of context retrieval (1-5).
        search_term: Keyword to search commits (level 5 only).
        commit_id: Specific commit ID to retrieve (level 5 only).
        log_window: Number of recent OTA log entries to include.
        follow_links: If True and level >= 5, include linked commit summaries (1-hop BFS).
    """
    with _state_lock:
        mem = _ensure_memory()
    return mem.get_context(
        level=level,
        search_term=search_term,
        commit_id=commit_id,
        log_window=log_window,
        follow_links=follow_links,
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_links(
    commit_id: str,
    link_types: str | None = None,
    max_hops: int = 1,
    query: str | None = None,
) -> str:
    """Retrieve cross-links for a commit.

    Shows which other commits are related via shared files (entity),
    explicit C### references (causal), replacement language (supersession),
    or keyword overlap (semantic). Useful for tracing the evolution of
    a feature or understanding why a decision was made.

    Link taxonomy inspired by A-MEM (bidirectional links) and MAGMA (typed
    edges), but uses mechanical heuristics — not the papers' LLM inference
    or dense vector embeddings. See CLAUDE.md for limitations.

    Args:
        commit_id: The commit to look up (e.g., "C012").
        link_types: Comma-separated link types to filter (default: all).
                    Options: entity, causal, supersession, semantic.
        max_hops: How many link hops to traverse (1 = direct, 2 = friends-of-friends).
        query: Optional natural language search intent. When provided, BFS traversal
               is weighted by query relevance (MAGMA intent-aware traversal, Alg. 1).
               Example: query="why was the auth system changed"
    """
    with _state_lock:
        mem = _ensure_memory()
    types = [t.strip() for t in link_types.split(",")] if link_types else None
    linked = mem.get_linked_commits(commit_id, link_types=types, max_hops=max_hops, query=query)
    if not linked:
        # Still show direct link summary even if BFS returned nothing
        direct = mem.get_commit_links(commit_id)
        if any(v for v in direct.values()):
            return mem._format_links_for_context(commit_id, direct)
        return f"No links found for {commit_id}."
    # Group by link type for readability
    lines = [f"# Links for {commit_id} (max_hops={max_hops})"]
    by_type: dict[str, list[dict]] = {}
    for entry in linked:
        by_type.setdefault(entry["link_type"], []).append(entry)
    for lt in ("entity", "causal", "supersession", "semantic"):
        entries = by_type.get(lt, [])
        if not entries:
            continue
        lines.append(f"\n## {lt.capitalize()} Links")
        for e in entries:
            hop_tag = f" (hop {e['hop']})" if e.get("hop", 1) > 1 else ""
            title = e.get("title", "")
            what = e.get("what", "")[:120]
            detail = ""
            if lt == "entity" and e.get("shared_files"):
                detail = f" | shared: {', '.join(e['shared_files'])}"
            elif lt in ("causal", "supersession") and e.get("snippet"):
                detail = f' | "{e["snippet"]}"'
            score_tag = ""
            if "query_score" in e:
                score_tag = f" [q: {e['query_score']:.3f}]"
            elif "embedding_score" in e:
                score_tag = f" [emb: {e['embedding_score']:.3f}]"
            lines.append(f"- **[{e['id']}]**{hop_tag}{score_tag} {title}{detail}")
            if what:
                lines.append(f"  {what}")
    return "\n".join(lines)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_evolve_memory(commit_id: str | None = None) -> str:
    """Manually trigger A-MEM evolution for a commit or all recent commits.

    When a sub-model is available, rewrites commit summaries to incorporate
    context from related commits (A-MEM §3.3 Eq.7 memory evolution).

    Args:
        commit_id: Specific commit to evolve (e.g. "C001"). If None, evolves
                   all recent commits that have semantic/supersession links.
    """
    sub = _get_sub_client()
    if sub is None:
        return (
            "Sub-model not available. Set CCR_OLLAMA_MODEL or ANTHROPIC_API_KEY "
            "to enable A-MEM memory evolution."
        )

    with _state_lock:
        mem = _ensure_memory()
    mem.sub_client = sub

    branch = mem.get_active_branch()
    evolutions_performed: list[str] = []

    try:
        from ccr.core.types import CommitLink  # noqa: PLC0415
        links_data = mem._load_links()

        def _evolve_commit_with_links(cid: str) -> int:
            """Evolve one commit's linked peers. Returns count of newly evolved entries."""
            node = links_data.get("links", {}).get(cid, {})
            candidate_links: list[CommitLink] = []
            for lt in ("semantic", "supersession"):
                for entry in node.get(lt, []):
                    score = entry.get("score", 0.0)
                    if score > 0.5:
                        candidate_links.append(CommitLink(
                            target=entry["target"],
                            link_type=lt,
                            score=score,
                        ))
            if not candidate_links:
                return 0
            before = set(mem._evolved_summaries.keys())
            mem._trigger_memory_evolution(cid, candidate_links)
            after = set(mem._evolved_summaries.keys())
            return len(after - before)

        if commit_id:
            n = _evolve_commit_with_links(commit_id)
            if n > 0:
                evolutions_performed.append(f"{commit_id}: {n} new evolution(s)")
            else:
                evolutions_performed.append(
                    f"{commit_id}: no new evolutions "
                    "(no eligible links or sub-model produced no changes)"
                )
        else:
            # Scan last 10 commits
            recent = mem._parse_recent_commit_data(branch, k=10)
            for commit in recent:
                cid = commit.get("id", "")
                if not cid:
                    continue
                n = _evolve_commit_with_links(cid)
                if n > 0:
                    evolutions_performed.append(f"{cid}: {n} new evolution(s)")

    except Exception as e:
        return f"Error during memory evolution: {type(e).__name__}: {e}"

    if not evolutions_performed or all("no new evolutions" in e for e in evolutions_performed):
        return "No evolutions performed. Ensure commits have semantic/supersession links with score > 0.5."
    return "A-MEM memory evolution complete:\n" + "\n".join(f"  - {e}" for e in evolutions_performed)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_log_ota(observation: str, thought: str, action: str) -> str:
    """Log an Observation-Thought-Action triple to the project log.

    OTA triples track your reasoning process. They're attached to
    commits and preserved across sessions. Each call appends a new entry.

    Args:
        observation: What you observed (e.g., "Test failure in router.py").
        thought: Your reasoning (e.g., "The regex pattern is too greedy").
        action: What you did (e.g., "Fixed pattern to use non-greedy match").
    """
    try:
        with _state_lock:  # H3: protect memory state mutation
            mem = _ensure_memory()
            mem.log_ota(
                tool_name="claude-code",
                observation=observation,
                thought=thought,
                action=action,
            )
        return "OTA logged."
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_status() -> str:
    """Show current project memory status.

    Returns the active branch, recent milestones, open branches,
    and metadata summary.
    """
    with _state_lock:
        mem = _ensure_memory()

    parts = []
    branch = mem.get_active_branch()
    parts.append(f"Active branch: {branch}")

    # Level-1 context (overview)
    overview = mem.get_context(level=1)
    parts.append(overview)

    return "\n\n".join(parts)


# ===========================================================================
# GCC Hierarchical Summary Tools (TiMem-inspired)
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_consolidate(tier: str = "session", content: str | None = None) -> str:
    """Generate or save a hierarchical memory summary (TiMem-inspired).

    Three tiers of consolidation, inspired by TiMem's Temporal Memory Tree
    (note: TMT tree structure and §3.3 recall pipeline are NOT implemented —
    this uses flat aggregation with mechanical consolidation):

    Tiers:
        "session": Generate a session summary from recent commits (mechanical, immediate).
                   Consolidates the last N commits into a structured paragraph.
        "phase": Generate a phase summary from recent sessions (mechanical, immediate).
                 Aggregates session summaries + branch metadata into a strategic view.
        "project": Returns a prompt for Claude Code to generate a project overview.
                   Call again with content= to save the generated overview.

    The session and phase tiers produce summaries immediately using mechanical
    consolidation (template-based extraction from commits). The project tier
    requires two calls: first to get the prompt, then to save the result.

    Args:
        tier: "session", "phase", or "project".
        content: For tier="project" only — the generated overview text to save.
    """
    try:
        with _state_lock:
            mem = _ensure_memory()
            if tier == "project" and content is not None:
                mem.save_overview(content)
                return "Project overview saved."
            return mem.get_consolidation_prompt(tier=tier)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_summaries(tier: str = "all", count: int = 5) -> str:
    """Retrieve hierarchical memory summaries.

    Returns consolidated summaries at the specified tier:
        "session": Recent session summaries from the active branch.
        "phase": Recent phase summaries (cross-branch).
        "project": The current project overview.
        "all": All tiers combined.

    Args:
        tier: Which tier(s) to retrieve ("session", "phase", "project", "all").
        count: Maximum number of summaries per tier (default 5).
    """
    with _state_lock:
        mem = _ensure_memory()
    count = max(1, min(count, 50))
    return mem.get_summaries(tier=tier, count=count)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_patterns(
    min_occurrences: int = 1,
    include_promoted: bool = True,
    search_term: str | None = None,
) -> str:
    """Query the CER-inspired pattern buffer.

    Patterns are transferable decision-making skills observed across commits
    (CER arXiv:2506.06698). They are deduped by word similarity and tracked
    by occurrence count. Patterns appearing in 3+ commits are suggested for
    ACE playbook promotion.

    Args:
        min_occurrences: Minimum occurrence count to include (default 1).
        include_promoted: Whether to include already-promoted patterns (default True).
        search_term: Optional keyword filter on pattern text.
    """
    with _state_lock:
        mem = _ensure_memory()
    result = mem.get_patterns(
        min_occurrences=min_occurrences,
        include_promoted=include_promoted,
        search_term=search_term,
    )

    if not result["patterns"]:
        return f"No patterns found (total buffer: {result['total']}, filter: min_occurrences={min_occurrences})."

    lines = [f"# Pattern Buffer ({result['matching']}/{result['total']} shown)"]
    for p in result["patterns"][:25]:
        promoted_tag = " [PROMOTED]" if p.get("promoted") else ""
        lines.append(
            f"- **[{p['id']}]** ({p['occurrence_count']}x, first: {p['first_seen']}){promoted_tag}\n"
            f"  {p['text']}\n"
            f"  Commits: {', '.join(p['commit_ids'])}"
        )

    # Promotion candidates
    threshold = mem.config.pattern_promotion_count
    candidates = [p for p in result["patterns"] if p["occurrence_count"] >= threshold and not p.get("promoted")]
    if candidates:
        lines.append(f"\n## Promotion Candidates (>= {threshold} occurrences)")
        for c in candidates:
            lines.append(f"- [{c['id']}] \"{c['text'][:100]}\" ({c['occurrence_count']}x)")

    return "\n".join(lines)


# ===========================================================================
# ACE Playbook Tools
# ===========================================================================


def _resolve_playbook(scope: str) -> tuple[Playbook, callable]:
    """Resolve playbook and save function based on scope."""
    if scope == "global":
        return _ensure_global_playbook(), _save_global_playbook
    return _ensure_playbook(), _save_playbook


def _serialize_playbook(pb: Playbook) -> str:
    """Serialize a playbook, using failure-aware format if lessons exist."""
    has_lessons = any(b.has_failure_lessons for b in pb.bullets)
    return pb.serialize_with_failures() if has_lessons else pb.serialize()


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def ace_get_playbook(task_context: str = "") -> str:
    """Get the current ACE playbook — both global and project-specific strategies.

    Returns two tiers:
    - GLOBAL: Universal heuristics that transfer across all projects (~/.ccr/)
    - PROJECT: Project-specific strategies (.ccr/)

    Review this after loading context to learn from past successes and failures.

    Args:
        task_context: Optional task description. When provided, prepends a
            policy-ranked section showing the top 5 most relevant bullets
            weighted by GRPO group-relative advantage (SkillRL Eq.3).
    """
    gpb = _ensure_global_playbook()
    ppb = _ensure_playbook()

    parts = []

    # Policy-ranked section when task_context is provided
    if task_context.strip():
        ranked = ppb.get_policy_ranked(task_context, top_k=5)
        if ranked:
            ranked_lines = ["# Policy-ranked skills for this task:"]
            for b in ranked:
                adv = f"{b.grpo_advantage:+.3f}"
                ranked_lines.append(f"[{b.id}] (advantage={adv}) :: {b.content}")
            parts.append("\n".join(ranked_lines))

    g_text = _serialize_playbook(gpb)
    if g_text.strip():
        parts.append(f"# GLOBAL PLAYBOOK (applies to all projects)\n{g_text}")
    else:
        parts.append("# GLOBAL PLAYBOOK (applies to all projects)\n(empty)")

    p_text = _serialize_playbook(ppb)
    if p_text.strip():
        parts.append(f"# PROJECT PLAYBOOK (this project only)\n{p_text}")
    else:
        parts.append("# PROJECT PLAYBOOK (this project only)\n(empty)")

    return "\n\n".join(parts)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def ace_apply_delta(operations: list[dict], scope: str = "project") -> str:
    """Apply delta operations to the playbook.

    Supported operations:
        ADD: {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "..."}
        UPDATE: {"type": "UPDATE", "bullet_id": "str-00001", "content": "new content"}
        MERGE: {"type": "MERGE", "bullet_id": "str-00001", "merge_target": "str-00002", "content": "merged"}
        REMOVE: {"type": "REMOVE", "bullet_id": "str-00001"}

    Use ADD when you discover a new insight or strategy.
    Use UPDATE to refine existing bullets.
    Use MERGE to combine similar bullets.
    Use REMOVE to delete unhelpful bullets.

    Args:
        operations: List of delta operation dicts.
        scope: "project" (default) or "global". Global bullets apply across all projects.
    """
    try:
        with _state_lock:
            pb, save_fn = _resolve_playbook(scope)
            ops = parse_delta_operations({"operations": operations})
            applied = pb.apply_delta(ops)
            save_fn()

        # Mark matching patterns as promoted (CER buffer close-the-loop)
        promoted_count = 0
        with _state_lock:
            mem = _ensure_memory()
        for op in ops:
            if op.op_type == "ADD" and op.content:
                try:
                    promoted_count += mem.mark_pattern_promoted_by_content(op.content)
                except Exception:
                    pass  # Never fail the delta apply

        result = f"Applied {applied} operation(s) to {scope} playbook. Now has {len(pb.bullets)} bullets."
        if promoted_count:
            result += f" Marked {promoted_count} pattern(s) as promoted in CER buffer."
        return result
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def ace_update_counters(bullet_tags: list[dict], scope: str = "project") -> str:
    """Update helpful/harmful counters for playbook bullets.

    After completing a task, reflect on which strategies helped or hurt,
    then update their counters. This drives evolutionary selection — high-scoring
    bullets persist, low-scoring ones get pruned.

    When tagging a bullet as "harmful", you SHOULD include a structured failure
    lesson explaining WHY it failed. This makes the harmful tag actionable:

    Args:
        bullet_tags: List of dicts. Each dict has:
            - "id": bullet ID (e.g., "str-00001")
            - "tag": "helpful" or "harmful"
            - "failure_lesson" (optional, for harmful tags): {
                "failure_point": "Where the strategy broke down",
                "flawed_reasoning": "What incorrect assumption was made",
                "counterfactual": "What should have been done instead",
                "prevention_principle": "General rule to avoid this failure"
              }
        scope: "project" (default) or "global".
    """
    try:
        with _state_lock:
            pb, save_fn = _resolve_playbook(scope)
            updated = pb.update_bullet_counts(bullet_tags)
            # Recompute GRPO advantages after counter update (SkillRL Eq.3)
            pb.recompute_grpo_advantages()
            save_fn()

        # Count how many had structured lessons
        lessons_added = sum(
            1 for t in bullet_tags
            if t.get("tag") == "harmful" and isinstance(t.get("failure_lesson"), dict) and t["failure_lesson"]
        )
        parts = [f"Updated {updated} bullet(s) in {scope} playbook."]
        if lessons_added:
            parts.append(f"Recorded {lessons_added} structured failure lesson(s).")
        return " ".join(parts)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def ace_get_stats() -> str:
    """Get playbook statistics — bullet counts, section breakdown, health metrics.

    Returns stats for both global and project playbooks. Includes evolution
    trigger info (SkillRL §3.3) and temporal decay stats.
    """
    gpb = _ensure_global_playbook()
    ppb = _ensure_playbook()
    return json.dumps({
        "global": asdict(gpb.get_stats()),
        "project": asdict(ppb.get_stats()),
    }, indent=2)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def ace_find_similar(threshold: float = 0.6, scope: str = "project") -> str:
    """Find similar bullet pairs that may be candidates for merging.

    Uses Jaccard + trigram similarity. Returns pairs above the threshold
    so you can decide whether to MERGE them.

    Args:
        threshold: Similarity threshold (0.0-1.0, default 0.6).
        scope: "project" (default), "global", or "cross" (find duplicates between tiers).
    """
    if scope == "cross":
        gpb = _ensure_global_playbook()
        ppb = _ensure_playbook()
        # Cross-tier: compare every global bullet against every project bullet
        pairs = []
        for gb in gpb.bullets:
            words_a = set(gb.content.lower().split())
            trigrams_a = Playbook._char_trigrams(gb.content.lower())
            if len(words_a) < 2:
                continue
            for pb_bullet in ppb.bullets:
                words_b = set(pb_bullet.content.lower().split())
                trigrams_b = Playbook._char_trigrams(pb_bullet.content.lower())
                if len(words_b) < 2:
                    continue
                word_inter = words_a & words_b
                word_union = words_a | words_b
                word_jaccard = len(word_inter) / len(word_union) if word_union else 0.0
                tri_inter = trigrams_a & trigrams_b
                tri_union = trigrams_a | trigrams_b
                tri_jaccard = len(tri_inter) / len(tri_union) if tri_union else 0.0
                combined = 0.4 * word_jaccard + 0.6 * tri_jaccard
                if combined >= threshold:
                    pairs.append((gb, pb_bullet, combined))
        pairs.sort(key=lambda x: x[2], reverse=True)
        if not pairs:
            return "No cross-tier similar bullet pairs found."
        lines = ["Cross-tier similarities (global vs project):"]
        for a, b, sim in pairs[:10]:
            lines.append(f"[{a.id}] (global) vs [{b.id}] (project) (similarity={sim:.2f})")
            lines.append(f"  G: {a.content[:100]}")
            lines.append(f"  P: {b.content[:100]}")
        return "\n".join(lines)

    pb, _ = _resolve_playbook(scope)
    pairs = pb.find_similar_pairs(threshold)
    if not pairs:
        return f"No similar bullet pairs found in {scope} playbook."
    lines = []
    for a, b, sim in pairs[:10]:
        lines.append(f"[{a.id}] vs [{b.id}] (similarity={sim:.2f})")
        lines.append(f"  A: {a.content[:100]}")
        lines.append(f"  B: {b.content[:100]}")
    return "\n".join(lines)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False))
def ace_prune(scope: str = "project") -> str:
    """Prune problematic bullets and enforce token budget.

    First evolves failure lessons into new skills (threshold=1, aggressive),
    then removes bullets where harmful >= helpful and harmful >= 3,
    then trims lowest-scoring bullets if playbook exceeds 80K chars.

    This ordering ensures failure lessons are distilled into new heuristic
    bullets before the harmful source bullets are removed.

    Args:
        scope: "project" (default) or "global".
    """
    try:
        with _state_lock:
            pb, save_fn = _resolve_playbook(scope)
            # Load schema for parameterized thresholds (MCE)
            sp = _schema_path if scope == "project" else _global_schema_path
            schema = _load_schema(sp) if sp else PlaybookSchema.default()
            # Evolve failure lessons BEFORE pruning to prevent permanent loss
            evolved = pb.evolve_from_failures(threshold=max(1, schema.evolution_threshold - 2))
            pruned = pb.prune_problematic(min_harmful=schema.prune_min_harmful)
            budget_pruned = pb.enforce_token_budget(
                max_chars=schema.token_budget, decay_rate=schema.decay_rate,
            )
            save_fn()
        total = len(pruned) + len(budget_pruned)
        parts = []
        if evolved:
            parts.append(f"Evolved {len(evolved)} new skill(s) from failure lessons.")
        parts.append(f"Pruned {total} bullet(s) from {scope} playbook. Now has {len(pb.bullets)} bullets.")
        return " ".join(parts)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _word_jaccard(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity between two strings."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _auto_synthesize_skills(candidates: list[dict], sub_client) -> list[dict]:
    """Cross-synthesize generalized skills from related failure lessons (SkillRL §3.3).

    Groups candidates by task_context similarity (word Jaccard >= 0.5), then calls
    the sub-model once per group of 2+ candidates to generate 1-2 generalized
    "When X, do Y" skills that abstract across all prevention_principles in the group.

    Single-candidate groups are skipped — they were already handled by the mechanical
    path (prevention_principle verbatim copy).

    Args:
        candidates: List of dicts with keys: bullet_slug, failure_lesson (dict with
            prevention_principle, task_context, failure_point).
        sub_client: A ClaudeClient instance (or compatible) with .completion() method.

    Returns:
        List of dicts with keys: content (str), when_to_apply (str), scope (str).
        Returns [] on any error (graceful no-op).
    """
    try:
        from ccr.utils.parsing import extract_json_string  # noqa: PLC0415

        if not candidates:
            return []

        # Greedy grouping by task_context similarity (threshold 0.5)
        groups: list[list[dict]] = []
        for cand in candidates:
            tc = cand.get("failure_lesson", {}).get("task_context", "")
            placed = False
            for group in groups:
                anchor_tc = group[0].get("failure_lesson", {}).get("task_context", "")
                if _word_jaccard(tc, anchor_tc) >= 0.5:
                    group.append(cand)
                    placed = True
                    break
            if not placed:
                groups.append([cand])

        synthesized: list[dict] = []
        for group in groups:
            if len(group) < 2:
                # Single-candidate groups handled by mechanical path — skip
                continue
            principles = [
                c.get("failure_lesson", {}).get("prevention_principle", "")
                for c in group
            ]
            # Filter empty principles
            principles = [p for p in principles if p.strip()]
            if not principles:
                continue

            principles_text = "\n".join(f"- {p}" for p in principles)
            prompt = (
                "You are synthesizing generalized skills from related failure lessons.\n\n"
                f"Failure lessons (all from similar task contexts):\n{principles_text}\n\n"
                'Write 1-2 generalized "When X, do Y" skills that abstract across all of these.\n'
                'Respond with a JSON array: [{"content": "...", "when_to_apply": "..."}]\n'
                "Be concise. Each skill should be 1-2 sentences."
            )

            try:
                raw = sub_client.completion([{"role": "user", "content": prompt}])
                json_str = extract_json_string(raw)
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and item.get("content", "").strip():
                            synthesized.append({
                                "content": item["content"].strip(),
                                "when_to_apply": item.get("when_to_apply", ""),
                                "scope": "project",
                            })
            except Exception:
                # Silent failure per spec — one bad group doesn't block others
                continue

        return synthesized
    except Exception:
        return []


# ===========================================================================
# ACE 3-Agent Pipeline (Generator → Reflector → Curator) — ACE §3.1-3.3
# ===========================================================================


def _ace_generator(trajectory: str, sub_client) -> list[str]:
    """Generate candidate strategy bullets from task trajectory (ACE §3.1)."""
    try:
        prompt = (
            "You are an AI strategy generator. Given this task trajectory:\n"
            f"{trajectory}\n\n"
            "Generate 2-3 actionable strategy bullets for a coding assistant playbook.\n"
            'Each bullet should be: specific, actionable, abstract (not tied to this exact task).\n'
            'Format as "When X, do Y" or "Always Y when Z".\n'
            'Respond with a JSON array: ["bullet 1", "bullet 2"]'
        )
        response = sub_client.completion([{"role": "user", "content": prompt}])
        raw = extract_json_string(response)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(p) for p in parsed if p]
        return []
    except Exception:
        return []


def _ace_reflector(candidates: list[str], trajectory: str, sub_client) -> list[str]:
    """Filter/score candidates by quality and relevance (ACE §3.2)."""
    try:
        if not candidates:
            return []
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
        prompt = (
            "Rate each strategy bullet for quality (1-5):\n"
            "- 5: Highly actionable, specific, generalizable\n"
            "- 3: Decent but could be improved\n"
            "- 1: Too vague or too specific\n\n"
            f"Task context: {trajectory}\n"
            f"Candidates:\n{numbered}\n\n"
            'Respond with JSON: [{"bullet": "...", "score": N}, ...]\n'
            "Only include bullets with score >= 3."
        )
        response = sub_client.completion([{"role": "user", "content": prompt}])
        raw = extract_json_string(response)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [
                item["bullet"]
                for item in parsed
                if isinstance(item, dict) and item.get("score", 0) >= 3 and item.get("bullet", "").strip()
            ]
        return candidates  # fallback: return original unfiltered
    except Exception:
        return candidates  # fallback: return original unfiltered on error


def _ace_curator(existing_bullets: list[str], candidates: list[str], sub_client) -> list[dict]:
    """Decide ADD/MERGE/SKIP for each candidate vs existing playbook (ACE §3.3)."""
    try:
        if not candidates:
            return []
        existing_sample = "\n".join(f"- {b}" for b in existing_bullets[:10])
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
        prompt = (
            "For each candidate strategy bullet, decide: ADD (novel), MERGE (similar to existing), SKIP (duplicate).\n\n"
            f"Existing bullets (first 10):\n{existing_sample}\n\n"
            f"Candidates:\n{numbered}\n\n"
            'Respond with JSON: [{"bullet": "...", "action": "ADD"|"MERGE"|"SKIP", "merge_with": "existing bullet text if MERGE else null"}]'
        )
        response = sub_client.completion([{"role": "user", "content": prompt}])
        raw = extract_json_string(response)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, dict) and item.get("bullet", "").strip():
                    result.append({
                        "bullet": item["bullet"],
                        "action": item.get("action", "ADD"),
                        "merge_with": item.get("merge_with"),
                    })
            return result
        return [{"bullet": c, "action": "ADD", "merge_with": None} for c in candidates]
    except Exception:
        return [{"bullet": c, "action": "ADD", "merge_with": None} for c in candidates]


def _run_ace_pipeline(title: str, what: str, why: str, sub_client) -> None:
    """Run the ACE 3-agent Generator → Reflector → Curator pipeline after a commit.

    Failures are silently swallowed — this must never affect the commit result.
    """
    try:
        trajectory = f"Title: {title}\nWhat: {what}\nWhy: {why}"

        # Step 1: Generator
        candidates = _ace_generator(trajectory, sub_client)
        if not candidates:
            return

        # Step 2: Reflector
        filtered = _ace_reflector(candidates, trajectory, sub_client)
        if not filtered:
            return

        # Step 3: Curator
        with _state_lock:
            pb = _ensure_playbook()
        existing_sample = [b.content for b in pb.bullets[:10]]
        decisions = _ace_curator(existing_sample, filtered, sub_client)

        # Step 4: Apply decisions
        with _state_lock:
            pb = _ensure_playbook()
            for decision in decisions:
                action = decision.get("action", "ADD")
                bullet_text = decision.get("bullet", "").strip()
                if not bullet_text:
                    continue
                if action == "ADD":
                    op = DeltaOperation(
                        op_type="ADD",
                        section="STRATEGIES & INSIGHTS",
                        content=bullet_text,
                    )
                    pb.apply_delta([op])
                elif action == "MERGE":
                    merge_with = decision.get("merge_with", "")
                    if merge_with:
                        # Find closest existing bullet by word Jaccard
                        best_id = None
                        best_sim = 0.0
                        for b in pb.bullets:
                            sim = _word_jaccard(b.content, merge_with)
                            if sim > best_sim:
                                best_sim = sim
                                best_id = b.id
                        if best_id and best_sim > 0.0:
                            op = DeltaOperation(
                                op_type="UPDATE",
                                section="",
                                content=bullet_text,
                                bullet_id=best_id,
                            )
                            pb.apply_delta([op])
                # SKIP: do nothing
            _save_playbook()
    except Exception:
        pass  # Never raise — commit result must not be affected


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def ace_generate_bullets(context: str, auto_apply: bool = False) -> str:
    """Generate and optionally apply strategy bullets via the ACE 3-agent pipeline.

    Runs Generator → Reflector → Curator to produce candidate playbook bullets
    from a task context or trajectory. Returns a preview of all decisions.

    Args:
        context: Task context or trajectory to generate bullets from.
        auto_apply: If True, automatically apply ADD decisions. Default False (preview only).
    """
    try:
        sub = _get_sub_client()
        if sub is None:
            return "No sub-model configured. Set CCR_OLLAMA_MODEL or ANTHROPIC_API_KEY to enable ACE pipeline."

        candidates = _ace_generator(context, sub)
        if not candidates:
            return "Generator produced no candidates."

        filtered = _ace_reflector(candidates, context, sub)
        if not filtered:
            return f"Reflector filtered all {len(candidates)} candidate(s). None passed quality threshold."

        with _state_lock:
            pb = _ensure_playbook()
        existing_sample = [b.content for b in pb.bullets[:10]]
        decisions = _ace_curator(existing_sample, filtered, sub)

        if not decisions:
            return "Curator produced no decisions."

        lines = ["# ACE Pipeline Results"]
        applied_count = 0
        if auto_apply:
            with _state_lock:
                pb = _ensure_playbook()
                for decision in decisions:
                    action = decision.get("action", "ADD")
                    bullet_text = decision.get("bullet", "").strip()
                    if not bullet_text:
                        continue
                    if action == "ADD":
                        op = DeltaOperation(
                            op_type="ADD",
                            section="STRATEGIES & INSIGHTS",
                            content=bullet_text,
                        )
                        pb.apply_delta([op])
                        applied_count += 1
                    elif action == "MERGE":
                        merge_with = decision.get("merge_with", "")
                        if merge_with:
                            best_id = None
                            best_sim = 0.0
                            for b in pb.bullets:
                                sim = _word_jaccard(b.content, merge_with)
                                if sim > best_sim:
                                    best_sim = sim
                                    best_id = b.id
                            if best_id and best_sim > 0.0:
                                op = DeltaOperation(
                                    op_type="UPDATE",
                                    section="",
                                    content=bullet_text,
                                    bullet_id=best_id,
                                )
                                pb.apply_delta([op])
                                applied_count += 1
                _save_playbook()
            lines.append(f"Applied {applied_count} decision(s) to project playbook.")

        lines.append(f"\n## Decisions ({len(decisions)} total):")
        for d in decisions:
            action = d.get("action", "?")
            bullet = d.get("bullet", "")[:120]
            merge_with = d.get("merge_with", "")
            if merge_with:
                lines.append(f"- [{action}] {bullet}\n  (merge with: {str(merge_with)[:80]})")
            else:
                lines.append(f"- [{action}] {bullet}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
def ace_evolve_from_failures(
    threshold: int = 3,
    scope: str = "project",
    synthesized_skills: list[dict] | None = None,
) -> str:
    """Evolve new skills from accumulated failure lessons (SkillRL §3.3 / Prompt B.1).

    Two-call pattern (caller-driven synthesis):
      1. Call with no synthesized_skills → returns failure lessons + prompt for
         Claude Code to synthesize new skills (teacher-model role per SkillRL §4.2).
      2. Call again with synthesized_skills=[{"content": "...", "when_to_apply": "..."}]
         → saves the synthesized skills as new playbook bullets.

    If synthesized_skills is not provided and the threshold is met, the tool
    falls back to mechanical evolution (prevention_principle extraction) for
    backward compatibility. The caller-driven path is preferred for higher
    quality synthesis per SkillRL's teacher-model pattern.

    Args:
        threshold: Minimum harmful-with-lessons bullets to trigger (default 3).
        scope: "project" (default) or "global".
        synthesized_skills: Optional list of dicts with "content" and "when_to_apply"
            keys, provided by Claude Code after reviewing the failure lessons prompt.
    """
    try:
        with _state_lock:
            pb, save_fn = _resolve_playbook(scope)

            # Path 2: Save synthesized skills from Claude Code
            if synthesized_skills:
                ops = []
                for skill in synthesized_skills:
                    content = skill.get("content", "").strip()
                    when = skill.get("when_to_apply", "")
                    if content:
                        ops.append(DeltaOperation(
                            op_type="ADD",
                            section="PROBLEM-SOLVING HEURISTICS",
                            content=content,
                        ))
                applied = pb.apply_delta(ops)
                # Set when_to_apply on newly added bullets
                for skill, bullet in zip(synthesized_skills, pb.bullets[-applied:]):
                    if skill.get("when_to_apply"):
                        bullet.when_to_apply = skill["when_to_apply"]
                    bullet.scope = skill.get("scope", "general")
                save_fn()
                return f"Saved {applied} synthesized skill(s) to {scope} playbook."

            # Path 1: Check if evolution is needed
            check = pb.check_evolution_needed(threshold)
            if not check.get("needed"):
                return (
                    f"Evolution not triggered in {scope} playbook. Need {threshold} harmful bullets with failure "
                    f"lessons, currently have {check['candidate_count']}."
                )

            # Run mechanical evolution (backward compat + fallback)
            new_bullets = pb.evolve_from_failures(threshold)
            if new_bullets:
                save_fn()

            # Build synthesis prompt for Claude Code (teacher-model per SkillRL §4.2)
            # Include the candidates so Claude can optionally synthesize better skills
            candidate_ids = set(check.get("candidate_ids", []))
            candidates = [b for b in pb.bullets if b.id in candidate_ids]

            # Phase 2: LLM cross-synthesis of related failure lessons (SkillRL §3.3)
            # Runs when sub-client is available; graceful no-op otherwise.
            sub = _get_sub_client()
            synthesized_count = 0
            if sub is not None and candidates:
                # Build candidate dicts expected by _auto_synthesize_skills
                cand_dicts = []
                for b in candidates:
                    for lesson in b.failure_lessons:
                        fl_dict = lesson if isinstance(lesson, dict) else {
                            "prevention_principle": getattr(lesson, "prevention_principle", ""),
                            "task_context": getattr(lesson, "task_context", ""),
                            "failure_point": getattr(lesson, "failure_point", ""),
                        }
                        cand_dicts.append({
                            "bullet_slug": b.id,
                            "failure_lesson": fl_dict,
                        })
                if cand_dicts:
                    synthesized = _auto_synthesize_skills(cand_dicts, sub)
                    for skill in synthesized:
                        content = skill.get("content", "").strip()
                        if content:
                            op = DeltaOperation(
                                op_type="ADD",
                                section="PROBLEM-SOLVING HEURISTICS",
                                content=content,
                            )
                            applied_count = pb.apply_delta([op])
                            # Set when_to_apply and scope on the newly added bullet
                            if applied_count > 0:
                                new_b = pb.bullets[-1]
                                if skill.get("when_to_apply"):
                                    new_b.when_to_apply = skill["when_to_apply"]
                                new_b.scope = skill.get("scope", "project")
                                synthesized_count += 1
                    if synthesized_count > 0:
                        save_fn()

            lessons_text = []
            for b in candidates:
                lessons_text.append(f"## Bullet [{b.id}]: {b.content[:100]}")
                for lesson in b.failure_lessons:
                    fp = getattr(lesson, 'failure_point', '?') if not isinstance(lesson, dict) else lesson.get('failure_point', '?')
                    fr = getattr(lesson, 'flawed_reasoning', '?') if not isinstance(lesson, dict) else lesson.get('flawed_reasoning', '?')
                    cf = getattr(lesson, 'counterfactual', '?') if not isinstance(lesson, dict) else lesson.get('counterfactual', '?')
                    pp = getattr(lesson, 'prevention_principle', '?') if not isinstance(lesson, dict) else lesson.get('prevention_principle', '?')
                    lessons_text.append(
                        f"  - Failure: {fp}\n"
                        f"    Flawed reasoning: {fr}\n"
                        f"    Counterfactual: {cf}\n"
                        f"    Prevention: {pp}"
                    )

            result_parts = []
            if new_bullets:
                result_parts.append(f"Evolved {len(new_bullets)} new skill(s) from {scope} failure lessons:")
                for b in new_bullets:
                    result_parts.append(f"  [{b.id}] {b.content[:100]}")
                    if b.when_to_apply:
                        result_parts.append(f"    When: {b.when_to_apply[:100]}")
            if synthesized_count > 0:
                result_parts.append(f"Synthesized {synthesized_count} cross-lesson skill(s) via sub-model (SkillRL §3.3).")

            if lessons_text:
                result_parts.append(
                    f"\n---\n# Optional: Caller-driven synthesis (SkillRL §4.2)\n"
                    f"For higher-quality skill synthesis, review these failure lessons and call\n"
                    f"ace_evolve_from_failures again with synthesized_skills=[\n"
                    f'  {{"content": "skill text", "when_to_apply": "condition", "scope": "general"}}\n'
                    f"]\n\n" + "\n".join(lessons_text)
                )

            return "\n".join(result_parts) if result_parts else "No skills evolved."
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
def ace_evolve_schema(
    scope: str = "project",
    apply_proposal: int | None = None,
    rollback: bool = False,
) -> str:
    """Evolve playbook schema via MCE-inspired (1+1)-ES (arXiv:2601.21557).

    Meta-level evolution of playbook STRUCTURE — sections, thresholds,
    parameters. Computes health metrics, compares to baseline,
    proposes one improvement per call.

    Three-call pattern:
      1. No args → evaluate metrics, show proposals
      2. apply_proposal=N → apply Nth proposal (1-indexed)
      3. rollback=True → revert to parent schema version

    Args:
        scope: "project" or "global".
        apply_proposal: 1-indexed proposal to apply (None = evaluate only).
        rollback: Revert to parent schema version.
    """
    from datetime import datetime, timezone

    try:
        with _state_lock:
            pb, save_fn = _resolve_playbook(scope)
            sp = _schema_path if scope == "project" else _global_schema_path
            schema = _load_schema(sp) if sp else PlaybookSchema.default()
            history = _load_schema_history(sp) if sp else []

            if rollback:
                # Revert to parent schema
                if schema.parent_version is None:
                    return "Cannot rollback: no parent schema version exists."
                parent = None
                for h in history:
                    if h.get("version") == schema.parent_version:
                        parent = PlaybookSchema.from_dict(h)
                        break
                if parent is None:
                    return f"Cannot rollback: parent version {schema.parent_version} not found in history."
                # Apply parent schema to playbook
                moved = pb.apply_schema(parent)
                # Compute new metrics
                new_metrics = pb.compute_metrics(parent)
                parent.baseline_metrics = new_metrics
                # Push current to history, restore parent
                history.append(schema.to_dict())
                # Enforce history cap
                max_hist = 20
                if len(history) > max_hist:
                    history = history[-max_hist:]
                _save_schema(parent, history, sp)
                save_fn()
                return (
                    f"Rolled back to schema v{parent.version}.\n"
                    f"Moved {moved} bullet(s) between sections.\n"
                    f"Current health: {new_metrics.overall_health:.3f}"
                )

            # Compute current metrics
            metrics = pb.compute_metrics(schema)

            if apply_proposal is not None:
                # Re-compute proposals to validate index
                proposals = pb.propose_schema_changes(
                    schema, metrics,
                    overflow_threshold=0.5,
                    min_cluster_size=3,
                    stop_health_threshold=0.8,
                    rollback_health_delta=-0.05,
                )
                idx = apply_proposal - 1
                if idx < 0 or idx >= len(proposals):
                    return f"Invalid proposal index {apply_proposal}. Available: {len(proposals)}."
                proposal = proposals[idx]

                # Create new schema version
                new_schema = PlaybookSchema(
                    version=schema.version + 1,
                    sections=list(schema.sections),
                    slug_map=dict(schema.slug_map),
                    decay_rate=schema.decay_rate,
                    prune_min_harmful=schema.prune_min_harmful,
                    evolution_threshold=schema.evolution_threshold,
                    token_budget=schema.token_budget,
                    parent_version=schema.version,
                    change_description=proposal.description,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    baseline_metrics=metrics,
                )

                # Apply the change
                ct = proposal.change_type
                details = proposal.details

                if ct == "ADD_SECTION":
                    new_name = details["name"]
                    slug_prefix = details.get("slug_prefix", new_name[:3].lower())
                    new_schema.sections.append(new_name)
                    from ccr.ace.playbook import _normalize_section
                    new_schema.slug_map[_normalize_section(new_name)] = slug_prefix
                    # Move specified bullets
                    bullet_ids = set(details.get("bullet_ids", []))
                    for bullet in pb.bullets:
                        if bullet.id in bullet_ids:
                            bullet.section = new_name
                    pb.apply_schema(new_schema)

                elif ct == "REMOVE_SECTION":
                    rm_name = details["name"]
                    if rm_name in new_schema.sections:
                        new_schema.sections.remove(rm_name)
                    pb.apply_schema(new_schema)

                elif ct == "ADJUST_DECAY":
                    new_schema.decay_rate = details["new_rate"]

                elif ct == "ADJUST_PRUNING":
                    new_schema.prune_min_harmful = details["new_min_harmful"]

                elif ct == "ADJUST_EVOLUTION":
                    new_schema.evolution_threshold = details["new_threshold"]

                elif ct == "ADJUST_BUDGET":
                    new_schema.token_budget = details["new_budget"]

                elif ct == "REBALANCE":
                    moves = details.get("moves", [])
                    for move in moves:
                        b = pb.get_bullet(move["bullet_id"])
                        if b:
                            b.section = move["to"]

                elif ct == "ROLLBACK":
                    # Handled above, shouldn't reach here
                    pass

                # Push old schema to history
                history.append(schema.to_dict())
                max_hist = 20
                if len(history) > max_hist:
                    history = history[-max_hist:]

                _save_schema(new_schema, history, sp)
                save_fn()

                new_metrics = pb.compute_metrics(new_schema)
                delta = new_metrics.overall_health - metrics.overall_health
                return (
                    f"Applied {ct} → schema v{new_schema.version}.\n"
                    f"{proposal.description}\n"
                    f"Health: {metrics.overall_health:.3f} → {new_metrics.overall_health:.3f} "
                    f"(Δ{delta:+.3f})"
                )

            # Evaluation mode: compute proposals
            proposals = pb.propose_schema_changes(
                schema, metrics,
                overflow_threshold=0.5,
                min_cluster_size=3,
                stop_health_threshold=0.8,
                rollback_health_delta=-0.05,
            )

            parts = [
                f"# Schema Health Report (v{schema.version})",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Section Balance | {metrics.section_balance:.3f} |",
                f"| Utilization Rate | {metrics.utilization_rate:.3f} |",
                f"| Harmful Ratio | {metrics.harmful_ratio:.3f} |",
                f"| Unused Ratio | {metrics.unused_ratio:.3f} |",
                f"| Decay Impact | {metrics.decay_impact:.3f} |",
                f"| **Overall Health** | **{metrics.overall_health:.3f}** |",
                f"| Total Bullets | {metrics.total_bullets} |",
                f"| Total Sections | {metrics.total_sections} |",
            ]

            if metrics.empty_sections:
                parts.append(f"| Empty Sections | {', '.join(metrics.empty_sections)} |")
            if metrics.overflow_sections:
                parts.append(f"| Overflow Sections | {', '.join(metrics.overflow_sections)} |")

            # Baseline comparison
            if schema.baseline_metrics:
                bl = schema.baseline_metrics
                delta = metrics.overall_health - bl.overall_health
                parts.extend([
                    f"",
                    f"## Baseline Comparison (v{schema.version} adopted at {bl.overall_health:.3f})",
                    f"Health delta: {delta:+.3f}",
                ])

            # Schema parameters
            parts.extend([
                f"",
                f"## Schema Parameters",
                f"- Decay rate: {schema.decay_rate}",
                f"- Prune min harmful: {schema.prune_min_harmful}",
                f"- Evolution threshold: {schema.evolution_threshold}",
                f"- Token budget: {schema.token_budget}",
                f"- Sections: {len(schema.sections)}",
            ])

            # Proposals
            if proposals:
                parts.extend([f"", f"## Proposals"])
                for i, p in enumerate(proposals, 1):
                    parts.append(
                        f"  {i}. **{p.change_type}** (confidence: {p.confidence:.2f})\n"
                        f"     {p.description}"
                    )
                parts.append(
                    f"\nCall `ace_evolve_schema(apply_proposal=N)` to apply, "
                    f"or `ace_evolve_schema(rollback=True)` to revert."
                )
            else:
                parts.append(f"\n## No proposals — playbook schema is healthy.")

            return "\n".join(parts)
    except ValueError:
        raise
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ===========================================================================
# RLM Sandbox Tools
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def rlm_init(task_prompt: str) -> str:
    """Initialize a sandboxed Python REPL for structured problem-solving.

    Provides the REPL execution component from the RLM paper (arXiv:2512.24601).
    Note: The paper's autonomous generate-execute loop (Algorithm 1) is NOT active
    in MCP mode — Claude Code drives the iteration loop manually via rlm_execute
    calls. This is the REPL substrate only, not the full RLM system.

    Sets up a REPL with these variables and tools pre-loaded:
        - task_prompt: Your problem statement (string)
        - context: Repo index metadata (dict with file paths, symbols, imports)
        - get_file(path): Fetch full content of any indexed file
        - search_repo(query): Search files by content/symbol/path
        - estimate_tokens(text): Estimate token count
        - FINAL_VAR(name): Signal completion and return a variable's value
        - SHOW_VARS(): List all user-created variables

    Use this for complex analysis that benefits from iterative exploration.

    Args:
        task_prompt: The problem or question to solve.
    """
    try:
        global _repl
        with _state_lock:
            idx = _ensure_index()

            # Clean up previous REPL if any
            if _repl is not None:
                _repl.cleanup()

            # Kernel sandbox runs code in a subprocess, which means in-process tools
            # (search_repo, get_file, etc.) are NOT available there. Use Python-level
            # sandboxing (AST validation + restricted builtins) for the RLM REPL,
            # which needs these tools. The kernel sandbox is available for standalone
            # code execution via KernelSandbox directly.
            _repl = CCRRepl(repo_index=idx, project_root=_project_root, use_kernel_sandbox=False)
            _repl.locals["task_prompt"] = task_prompt

        # Load playbook as variable if available
        pb = _ensure_playbook()
        pb_text = pb.serialize()
        if pb_text.strip():
            _repl.locals["playbook"] = pb_text

        prompt_preview = task_prompt[:120] + "..." if len(task_prompt) > 120 else task_prompt
        file_count = len(idx.files) if idx.files else 0

        return (
            f"REPL initialized.\n"
            f"- task_prompt: {len(task_prompt)} chars — \"{prompt_preview}\"\n"
            f"- context: repo metadata ({file_count} files indexed)\n"
            f"- Tools: get_file(), search_repo(), estimate_tokens(), FINAL_VAR(), SHOW_VARS()\n"
            f"\nUse rlm_execute to run code. Use rlm_finalize when done."
        )
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _summarize_stdout(stdout: str, threshold: int = 1000) -> str:
    """Summarize long stdout per RLM paper Section 3 ('Metadata-only stdout').

    If stdout exceeds *threshold* chars, replace with a metadata summary
    showing line/char counts plus the first and last 5 lines.  Short output
    (at or below the threshold) is returned unchanged.

    This enforces the paper's principle that stdout should carry metadata
    (lengths, counts, type info) rather than full content, saving tokens.
    """
    if len(stdout) <= threshold:
        return stdout

    lines = stdout.splitlines()
    line_count = len(lines)
    char_count = len(stdout)

    head = "\n".join(lines[:5])
    tail = "\n".join(lines[-5:]) if line_count > 10 else ""

    parts = [f"[stdout truncated: {line_count} lines, {char_count} chars]"]
    parts.append(head)
    if tail:
        parts.append("...")
        parts.append(tail)

    return "\n".join(parts)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def rlm_execute(code: str, metadata_only: bool = True) -> str:
    """Execute Python code in the sandboxed REPL.

    The REPL persists variables across calls. Use it to:
        - Explore the repo: search_repo("pattern"), get_file("path")
        - Process data: parse, filter, transform results
        - Build answers incrementally across multiple execute calls

    Stdout is captured and returned. Variables persist between calls.
    By default, long stdout is summarized to metadata per the RLM paper
    (Section 3, 'Metadata-only stdout') to save tokens.

    Args:
        code: Python code to execute.
        metadata_only: If True (default), stdout exceeding 1000 chars is
            replaced with a metadata summary (line count, char count, first/last
            5 lines). Set to False to return full stdout regardless of length.
    """
    try:
        with _state_lock:
            if _repl is None:
                return "Error: REPL not initialized. Call rlm_init first."
            repl_ref = _repl

        result = repl_ref.execute_code(code)

        parts = []
        if result.stdout.strip():
            stdout = result.stdout.strip()
            if metadata_only:
                stdout = _summarize_stdout(stdout)
            parts.append(f"stdout: {stdout}")

        if result.error:
            parts.append(f"error: {result.error}")

        if result.final_answer is not None:
            parts.append(f"FINAL_ANSWER: {result.final_answer}")

        if result.locals_snapshot:
            var_summary = ", ".join(
                f"{k}: {v[:60]}" for k, v in list(result.locals_snapshot.items())[:10]
            )
            parts.append(f"vars: {var_summary}")

        parts.append(f"time: {result.execution_time:.3f}s")

        return "\n".join(parts)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False))
def rlm_finalize(variable_name: str) -> str:
    """Finalize the REPL session and return a variable's value as the result.

    Calls FINAL_VAR internally to extract and serialize the named variable.
    Cleans up the REPL after extraction.

    Args:
        variable_name: Name of the variable to return.
    """
    try:
        global _repl
        with _state_lock:
            if _repl is None:
                return "Error: REPL not initialized. Call rlm_init first."
            repl_ref = _repl

        # H1: Validate variable_name to prevent code injection
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', variable_name):
            return f"Error: Invalid variable name '{variable_name}'. Must be a valid Python identifier."

        # Call _final_var directly instead of constructing code string (H1)
        answer = repl_ref._final_var(variable_name)
        with _state_lock:
            repl_ref.cleanup()
            _repl = None  # M4: Reset _repl after cleanup

        if answer is not None and not answer.startswith("Error:"):
            return answer
        return f"Error: Variable '{variable_name}' not found."
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ===========================================================================
# Repo Index Tools
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
def index_build() -> str:
    """Build or rebuild the repo index.

    Scans the project directory for source files, extracts symbols (classes,
    functions) and imports per file. The index enables search_repo and
    get_file in the RLM sandbox.

    If onnxruntime + tokenizers are installed, also computes dense embeddings
    for semantic search (A-RAG §3.1). Otherwise, BM25 fallback is available.
    """
    try:
        global _repo_index, _embedding_model
        with _state_lock:  # H2: protect global state mutation
            _repo_index = RepoIndex.build(_project_root)

            # Cache
            mem = _ensure_memory()
            try:
                mem.save_index(_repo_index.to_json())
                mem.update_metadata_file_tree([f for f in _repo_index.files.keys()])
            except Exception:
                pass

        # Semantic: compute embeddings if available (A-RAG §3.1)
        emb_status = ""
        try:
            from ccr.context.embeddings import SEMANTIC_AVAILABLE, get_embedding_model

            if SEMANTIC_AVAILABLE:
                model = get_embedding_model()
                if model is not None:
                    count = _repo_index.build_embeddings(model)
                    _repo_index.save_embeddings(_embeddings_path)
                    _embedding_model = model
                    emb_status = f"\nEmbeddings: {count} files ({model.MODEL_NAME})"
                    # A-RAG §3.1: build chunk-level embeddings for snippet extraction
                    try:
                        chunk_count, chunk_files = _repo_index.build_chunk_embeddings(model)
                        _repo_index.save_chunk_embeddings(_chunk_embeddings_path)
                        emb_status += f"\nChunk embeddings: {chunk_count} chunks across {chunk_files} files"
                    except Exception as ce:
                        emb_status += f"\nChunk embeddings: skipped ({ce})"
            else:
                emb_status = (
                    "\nEmbeddings: unavailable (install onnxruntime + tokenizers for semantic search)"
                )
        except Exception as e:
            emb_status = f"\nEmbeddings: skipped ({e})"

        return _repo_index.get_summary() + emb_status
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def index_search(query: str, mode: str = "hybrid", top_k: int = 10) -> str:
    """Search the repo index for files matching a query.

    Three search modes:
      - "keyword": Exact substring matching on paths, symbols, content (fast).
        Inspired by A-RAG §3.2 keyword_search, but uses substring matching
        not the paper's frequency*length scoring (Eq 1).
      - "semantic": Meaning-based search via ONNX dense embeddings
        (A-RAG §3.2 Eq 3 inspired) or BM25 fallback (CCR's own zero-dep
        alternative — BM25 is not from the A-RAG paper).
      - "hybrid": Combines keyword + semantic scores via weighted score fusion
        (default, best results). This is CCR's own design — A-RAG uses
        agent-driven tool selection, not mechanical score fusion.

    Args:
        query: Search term or natural language description.
        mode: "keyword", "semantic", or "hybrid" (default).
        top_k: Maximum results to return (default 10).
    """
    if mode not in ("keyword", "semantic", "hybrid"):
        return f"Error: invalid mode '{mode}'. Use 'keyword', 'semantic', or 'hybrid'."

    top_k = max(1, min(top_k, 100))  # M3: bound top_k to prevent excessive results
    idx = _ensure_index()

    suffix = ""
    if mode == "keyword":
        results = idx.search(query)[:top_k]
    elif mode == "semantic":
        if _embedding_model is not None and idx._embeddings:
            results = idx.semantic_search(query, _embedding_model, top_k=top_k)
        else:
            results = idx.bm25_search(query, top_k=top_k)
            suffix = " (BM25 fallback)"
    else:  # hybrid
        results = idx.hybrid_search(query, model=_embedding_model, top_k=top_k)
        if _embedding_model is None or not idx._embeddings:
            suffix = " (BM25 fallback)"

    if not results:
        return f"No files matching '{query}' ({mode} mode)."

    lines = [f"# {mode} search{suffix}"]
    for r in results:
        syms = ", ".join(r["symbols"][:5]) if r["symbols"] else ""
        lines.append(
            f"[{r['score']}] {r['path']} ({r['language']}, {r['lines']} lines)"
            + (f" — {syms}" if syms else "")
        )
    return "\n".join(lines)


# ===========================================================================
# Entry point
# ===========================================================================


def main():
    """Run the MCP server with stdio transport."""
    import argparse

    parser = argparse.ArgumentParser(description="CCR MCP Server")
    parser.add_argument(
        "--project", "-p",
        default=os.getcwd(),
        help="Project root directory (default: cwd)",
    )
    args = parser.parse_args()

    _init(args.project)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
