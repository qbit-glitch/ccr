# Example GCC Session

This walkthrough demonstrates a typical GCC-managed session building a REST API.

## 1. Initialization

The agent starts working on a new project. GCC auto-initializes:

```
> "Build me a REST API for a todo app"

[GCC] Initializing .GCC/ structure...
[GCC] Created: main.md, metadata.yaml, commit.md, log.md, branches/
```

**main.md** is populated with:

```markdown
# Project Roadmap

## Objectives
- [ ] Build REST API for todo app
- [ ] CRUD endpoints for todos
- [ ] Error handling and validation
```

## 2. First Milestone (COMMIT)

After implementing the basic Express server and todo model:

```
[GCC] COMMIT: Implement Express server with Todo model and in-memory store

commit.md entry:
  [C001] Implement Express server with Todo model
  - Branch: main
  - Contribution: Created server.js with Express setup on port 3000.
    Defined Todo model with id, title, completed fields.
    Implemented in-memory array store with helper functions.
  - Files: server.js, models/todo.js
```

## 3. Branching (BRANCH)

The agent considers two database options:

```
> "Should we use SQLite or PostgreSQL?"

[GCC] BRANCH: try-sqlite
  Purpose: Evaluate SQLite as lightweight storage for the todo API

[GCC] BRANCH: try-postgres
  Purpose: Evaluate PostgreSQL for production-grade storage
```

Each branch gets its own workspace under `.GCC/branches/`.

## 4. Working on a Branch

On `try-sqlite`, the agent implements and tests:

```
[OTA-012] Branch: try-sqlite
  Observation: SQLite requires no separate server process
  Thought: Good fit for single-user deployments, simpler setup
  Action: Implementing better-sqlite3 integration in db/sqlite.js

[GCC] COMMIT: SQLite integration with migration script
  [C001] on try-sqlite
  - Contribution: Implemented db/sqlite.js using better-sqlite3.
    Created migration script for todos table.
    All CRUD operations tested and passing.
```

## 5. Merging (MERGE)

After evaluating both branches:

```
> "SQLite is better for this use case, merge it"

[GCC] MERGE: try-sqlite -> main
  Synthesis: Evaluated SQLite vs PostgreSQL. SQLite chosen for
  simplicity and zero-config deployment. PostgreSQL branch
  abandoned (overkill for single-user todo app).

main.md updated:
  ### M2: Database selection - SQLite
  - Branch: try-sqlite (merged)
  - Branch: try-postgres (abandoned)
```

## 6. Context Recovery (CONTEXT)

In a new session, the agent recovers state:

```
> "Where were we?"

[GCC] CONTEXT --full
  Reading main.md...

  Project: REST API for todo app
  Completed:
    - M1: Express server + Todo model
    - M2: SQLite database integration
  Active branches: none
  Remaining objectives:
    - [ ] Error handling and validation

  Ready to continue with error handling.
```

## Resulting .GCC/ Structure

```
.GCC/
├── main.md              # Updated with 2 milestones
├── metadata.yaml        # File tree, dependencies updated
├── commit.md            # 3 commits on main (init + C001 + merge)
├── log.md               # 15 OTA entries
└── branches/
    ├── try-sqlite/
    │   ├── summary.md   # Status: merged
    │   ├── commit.md    # 1 commit
    │   └── log.md       # 5 OTA entries
    └── try-postgres/
        ├── summary.md   # Status: abandoned
        ├── commit.md    # 1 commit
        └── log.md       # 3 OTA entries
```
