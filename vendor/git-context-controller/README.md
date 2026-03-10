# Git Context Controller (GCC)

[![Release](https://img.shields.io/github/v/release/faugustdev/git-context-controller)](https://github.com/faugustdev/git-context-controller/releases/tag/v1.0.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Skills.sh](https://img.shields.io/badge/skills.sh-compatible-blue)](https://skills.sh)

**Structured context management framework for LLM agents.**

GCC implements Git-like operations (COMMIT, BRANCH, MERGE, CONTEXT) to manage long-horizon agent memory as a persistent, versioned file system.

> Based on the research paper: [Git Context Controller](https://arxiv.org/abs/2508.00031)

---

## Why GCC?

LLM agents lose context as conversations grow. Critical decisions, technical reasoning, and intermediate results vanish behind token limits. GCC solves this by giving agents a structured memory system:

- **No more lost context** -- milestones are persisted with full technical reasoning
- **Safe experimentation** -- branches isolate alternative approaches without polluting the main flow
- **Cross-session continuity** -- agents recover exactly where they left off
- **Multi-agent handoff** -- one agent's work is readable by another

## How It Works

```
                          ┌─────────────────────────────────┐
                          │           .GCC/                  │
                          │                                  │
                          │  main.md         (roadmap)       │
                          │  metadata.yaml   (state)         │
                          │  commit.md       (history)       │
                          │  log.md          (OTA traces)    │
                          │                                  │
                          │  branches/                       │
                          │    ├── feature-x/                │
                          │    │   ├── summary.md            │
                          │    │   ├── commit.md             │
                          │    │   └── log.md                │
                          │    └── experiment-y/             │
                          │        ├── summary.md            │
                          │        ├── commit.md             │
                          │        └── log.md                │
                          └─────────────────────────────────┘
```

### The OTA Cycle

Agents operate through **Observation-Thought-Action** cycles, logged in real time:

```
┌───────────┐     ┌───────────┐     ┌───────────┐
│ OBSERVE   │────>│  THINK    │────>│   ACT     │
│           │     │           │     │           │
│ Read logs │     │ Analyze   │     │ Execute   │
│ Check     │     │ Decide    │     │ COMMIT    │
│ state     │     │ strategy  │     │ BRANCH    │
└───────────┘     └───────────┘     │ MERGE     │
      ^                             └─────┬─────┘
      │                                   │
      └───────────────────────────────────┘
              Logged to log.md
```

### Command Flow

```
  User works on task
         │
         ▼
  ┌──────────────┐    milestone?    ┌──────────┐
  │  OTA Cycle   │────────────────> │  COMMIT  │──> commit.md
  │  (ongoing)   │                  └──────────┘
  └──────┬───────┘
         │
         │  need alternative?
         ▼
  ┌──────────────┐                  ┌──────────┐
  │   BRANCH     │────────────────> │ Isolated │──> branches/<name>/
  │              │                  │ workspace│
  └──────────────┘                  └────┬─────┘
                                         │
                                    validated?
                                         │
                                         ▼
                                    ┌──────────┐
                                    │  MERGE   │──> main.md updated
                                    └──────────┘

  ┌──────────────┐
  │  CONTEXT     │──> Retrieve memory at any resolution
  │  --branch    │    (summary, traces, metadata, full)
  │  --log       │
  │  --metadata  │
  │  --full      │
  └──────────────┘
```

## Installation

### As a Claude Code Skill

```bash
# Via skills.sh
npx skills add faugustdev/git-context-controller

# Manual installation
cp -r gcc/ your-project/.claude/skills/gcc/
```

### Standalone

```bash
git clone https://github.com/faugustdev/git-context-controller.git

# Initialize GCC in your project
./scripts/gcc_init.sh /path/to/your/project/.GCC
```

## Quick Start

Once installed, GCC activates automatically. Use commands or natural language:

| Action | Command | Natural Language |
|---|---|---|
| Save progress | `/gcc commit` | "save this milestone" |
| Try alternative | `/gcc branch experiment` | "branch to try a different approach" |
| Integrate results | `/gcc merge experiment` | "merge the experiment results" |
| Recover context | `/gcc context --full` | "where were we?" |
| View recent work | `/gcc context --log` | "show recent activity" |
| Check structure | `/gcc context --metadata` | "what files do we have?" |

## Commands Reference

### COMMIT

Persists a milestone with full technical context.

```
/gcc commit <summary>
```

Each commit captures:
- Sequential ID and timestamp
- Branch context and purpose
- Previous progress summary
- Detailed technical contribution
- Files touched

### BRANCH

Creates an isolated workspace for experimentation.

```
/gcc branch <name>
```

Creates a new directory under `.GCC/branches/<name>/` with its own commit history, OTA log, and summary.

### MERGE

Synthesizes a completed branch back into the main flow.

```
/gcc merge <branch-name>
```

Updates `main.md` with results, marks the branch as merged or abandoned, and creates a synthesis commit.

### CONTEXT

Retrieves historical memory at different resolution levels.

```
/gcc context [--branch [name] | --log [n] | --metadata | --full]
```

| Flag | Returns | Use Case |
|---|---|---|
| `--branch` | Branch summary + recent commits | Understand what happened on a branch |
| `--log [n]` | Last N OTA entries (default: 20) | Debug or resume interrupted work |
| `--metadata` | Project structure, dependencies | Recover file tree and config |
| `--full` | Complete roadmap from main.md | Cross-session recovery or agent handoff |

## File Formats

### main.md

Global roadmap with objectives, milestones, and active branches.

### commit.md

Structured commit entries with branch purpose, previous progress, and detailed contribution.

### log.md

Sequential OTA (Observation-Thought-Action) trace entries. Capped at 50 entries.

### metadata.yaml

Infrastructure state: branch registry, file tree, dependencies, configuration.

See [`references/file_formats.md`](references/file_formats.md) for complete format specifications with examples.

## Configuration

GCC behavior is controlled via `metadata.yaml`:

```yaml
proactive_commits: true   # Auto-suggest commits after milestones
proactive_commits: false  # Only commit when explicitly requested
```

Toggle with natural language: "enable/disable proactive commits"

## Project Structure

```
git-context-controller/
├── SKILL.md              # Claude Code skill definition
├── README.md             # This file
├── LICENSE               # MIT License
├── CONTRIBUTING.md       # Contribution guidelines
├── .gitignore            # Excludes .GCC/ and local files
├── scripts/
│   └── gcc_init.sh       # Initialization script
├── references/
│   └── file_formats.md   # Detailed format specifications
└── examples/
    └── sample_session.md # Example GCC session walkthrough
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## References

- **Paper**: [Git Context Controller: Structured Context Management for LLM Agents](https://arxiv.org/abs/2508.00031)
- **Concept**: [Emergent Mind - GCC Topic](https://www.emergentmind.com/topics/git-context-controller-gcc)

## License

[MIT](LICENSE)
