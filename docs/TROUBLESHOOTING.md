# CCR Troubleshooting Guide

Run `ccr doctor` first — it diagnoses the most common issues automatically.

---

## Common Errors

### 1. `ccr: command not found`

**Cause**: CCR installed into a directory that isn't in your PATH.

**Fix**:
```bash
# Option A: Install from source (recommended for development)
cd /path/to/ccr && pip install -e .

# Option B: Check that ~/.local/bin is in PATH
echo $PATH | grep -o ~/.local/bin
# If missing, add to ~/.zshrc or ~/.bashrc:
export PATH="$HOME/.local/bin:$PATH"

# Then reload:
source ~/.zshrc   # or source ~/.bashrc
```

---

### 2. `ModuleNotFoundError: No module named 'ccr'`

**Cause**: Running `ccr` outside the virtual environment where it was installed.

**Fix**:
```bash
# Activate the venv first:
source .venv/bin/activate

# Then verify:
python -c "import ccr; print(ccr.__version__)"
```

---

### 3. `claude: command not found`

**Cause**: Claude Code CLI isn't installed or isn't in PATH.

**Fix**:
```bash
# Install Claude Code (requires Node.js 18+):
npm install -g @anthropic-ai/claude-code

# Verify:
claude --version

# If Node.js is missing:
# macOS: brew install node
# Linux: https://nodejs.org/en/download
```

---

### 4. `No .ccr/ directory found — run 'ccr init'`

**Cause**: CCR not initialized in this project directory.

**Fix**:
```bash
cd /your/project
ccr init          # creates .ccr/ directory
ccr install       # registers hooks and MCP server
```

---

### 5. Memory not loading at session start

**Symptom**: Claude doesn't seem to know past context. `gcc_status()` shows no commits.

**Diagnosis**:
```bash
ccr doctor    # shows which checks pass/fail
```

**Common causes and fixes**:

| Symptom | Cause | Fix |
|---------|-------|-----|
| `[!!] Hooks not configured` | `ccr install` not run | `ccr install` |
| `[WARN] Hook path stale` | Reinstalled CCR with pip | `ccr uninstall && ccr install` |
| `[!!] Claude Code CLI not found` | `claude` not in PATH | Install Claude Code (see #3 above) |
| `[!!] .mcp.json not found` | MCP server not registered | `ccr install` |
| Hooks active but still no memory | Python version mismatch | `ccr doctor` → check Python line |

**Manual verification**:
```bash
# Check hooks are configured:
cat .claude/settings.local.json | python3 -m json.tool | grep -A2 "Stop"

# Check MCP server is registered:
cat .mcp.json

# Check hook paths still exist:
ccr doctor
```

---

## ccr doctor Output Explained

```
[OK]  Python 3.11.9               ← Python version is compatible
[OK]  Claude Code CLI found        ← claude command is in PATH
[OK]  .ccr/ exists at ...          ← project initialized
[OK]  .mcp.json configured         ← MCP server registered
[OK]  Auto-commit hooks configured ← UserPromptSubmit + Stop hooks active
[OK]  Hook paths valid             ← python exe + 4 scripts all exist on disk
[OK]  No hook errors logged        ← no recent failures

[!!]  ...  ← Issues that will break CCR; must fix before using
[--]  ...  ← Notices; optional improvements or non-critical warnings
[WARN] Hook path stale  ← Python path changed (e.g. after pip upgrade in new venv)
```

**Fix any `[!!]` items before opening Claude Code.**

---

## Getting More Help

- Run `ccr doctor` — most issues show up here with a fix command
- Check hook errors: `tail -60 .ccr/.hook_errors.log`
- GitHub issues: https://github.com/qbit-glitch/ccr/issues
