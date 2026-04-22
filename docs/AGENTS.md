# CCR Agent Compatibility Matrix

CCR integrates with AI agents through a pluggable adapter architecture. Each agent gets the best available integration strategy based on its capabilities.

## Quick Reference

| Agent | MCP | Hooks | Level | Status | Cost | Install Command |
|-------|:---:|:-----:|:-----:|--------|------|-----------------|
| **Claude Code** | ✅ | ✅ JSON | 5 (Full) | ✅ Verified | $20/mo Pro or API key | `ccr install-global --agents claude-code` |
| **Kimi Code CLI** | ✅ | ✅ TOML | 5 (Full) | ✅ Verified | Free tier | `ccr install-global --agents kimi` |
| **Continue** | ✅ | ❌ | 4 (MCP) | 🧪 Experimental | LLM backend costs* | `ccr install-global --agents continue-dev` |
| **Ollama** | ❌ | ❌ | 3 (File) | 🧪 Experimental | Free (needs RAM/GPU) | `ccr install-global --agents ollama` |
| **OpenAI API** | ❌ | ❌ | 2 (SDK) | 🧪 Experimental | Pay-per-token | `ccr install-global --agents openai` |
| **Generic MCP** | ✅ | ❌ | 4 (MCP) | ✅ Verified | Depends on LLM | `ccr install-global --agents generic-mcp` |

**Status legend:**
- ✅ **Verified** — Integration tested with real software
- 🧪 **Experimental** — Adapter generates correct config/wrapper, but not yet tested against the actual agent. Community testing welcome!

**Cost legend:**
- CCR itself is **free and open source** (MIT license)
- *Continue extension is free, but the LLM backends (OpenAI, Anthropic, etc.) require paid API keys

## Integration Levels

### Level 5 — MCP + Hooks (Full Experience)
The agent calls CCR tools via MCP, and CCR automatically injects context via lifecycle hooks. Zero manual work.

**Supported agents:** Claude Code, Kimi Code CLI

### Level 4 — MCP Only (Tool Access)
The agent calls CCR tools via MCP. You must manually call `gcc_context()` when you want memory loaded.

**Supported agents:** Continue.dev, any MCP-capable agent via Generic MCP adapter

### Level 3 — File Watcher (No Hooks, No MCP)
CCR writes context to `.ccr/context_inject.md`. A wrapper script or file-watcher injects it into the agent's input.

**Supported agents:** Ollama

### Level 2 — SDK Wrapper / CLI Prefix (Programmatic)
Python decorator `@ccr.wrap` or CLI wrapper `ccr-agent` prepends context to any LLM call.

**Supported agents:** OpenAI API

### Level 1 — Manual (Documentation Only)
Export context with `ccr export-context` and copy-paste into your agent.

**Universal fallback** — works with any agent.

---

## Per-Agent Setup

### Claude Code

```bash
ccr install-global --agents claude-code
# Or simply (default):
ccr install-global
```

Then run `claude` from any project directory. CCR auto-creates `.ccr/` on first use.

### Kimi Code CLI

```bash
ccr install-global --agents kimi
# Or with Claude (default):
ccr install-global
```

Then run `kimi` from any project directory.

### Continue (VS Code / JetBrains extension)

```bash
ccr install-global --agents continue-dev
```

Then reload Continue in your IDE. CCR tools will appear in the MCP tools panel. Call `gcc_context(level=2)` manually when you want memory loaded.

> Note: The extension is called **"Continue"** in the VS Code marketplace (not "Continue.dev"). Visit [continue.dev](https://continue.dev) for installation instructions.

### Ollama

```bash
ccr install-global --agents ollama
```

This creates `~/.ccr/bin/ollama-ccr`, a wrapper script that prepends project context before each prompt:

```bash
ollama-ccr llama3.2 "Explain this codebase"
```

### OpenAI API

```bash
ccr install-global --agents openai
```

This creates:
1. `~/.ccr/bin/ccr-openai` — CLI wrapper
2. `ccr/wrap_openai.py` — Python SDK wrapper

```python
import openai
from ccr.wrap_openai import wrap_openai

client = openai.OpenAI()
wrapped = wrap_openai(client)
response = wrapped.chat.completions.create(...)
```

### Generic MCP (manual import)

```bash
ccr install-global --agents generic-mcp
```

This writes `~/.ccr/mcp-config.json`. Copy the `mcpServers.ccr` block into your agent's MCP configuration file. Refer to your agent's documentation for the correct location.

---

## Auto-Detection

CCR can detect which agents are installed and set up all of them at once:

```bash
# Install CCR for every detected agent
ccr install-global --agents auto

# Install CCR for ALL supported agents (even if not detected)
ccr install-global --agents all
```

Check what CCR detected:

```bash
ccr agents list
ccr agents info <agent-name>
```

---

## Adding a New Agent Adapter

To add support for a new agent, create a file in `ccr/adapters/`:

```python
# ccr/adapters/my_agent.py
from ccr.adapters import BaseAgentAdapter, InstallResult, UninstallResult

class MyAgentAdapter(BaseAgentAdapter):
    name = "my-agent"
    display_name = "My Agent"

    def is_installed(self) -> bool:
        return shutil.which("my-agent") is not None

    def supports_mcp(self) -> bool:
        return True  # or False

    def supports_hooks(self) -> bool:
        return False  # or True

    def install(self, python_exe: str, hooks_dir: str, ccr_pkg: str) -> InstallResult:
        # Set up MCP config, hooks, wrapper scripts, etc.
        return InstallResult(success=True, message="Done", files_modified=[])

    def uninstall(self) -> UninstallResult:
        return UninstallResult(success=True, message="Removed", files_modified=[])
```

Then import it in `ccr/adapters/__init__.py`:

```python
from ccr.adapters import my_agent
# Add my_agent to the module iteration list
```

No other code changes needed — the adapter will automatically appear in `ccr agents list` and `ccr install-global --agents auto`.
