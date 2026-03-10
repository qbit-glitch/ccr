## Some very useful papers which must be explored and understood properly

1. https://arxiv.org/pdf/2508.00031
2. https://arxiv.org/pdf/2512.24601
3. https://arxiv.org/abs/2510.04618


## A very important note

Claude Code is brilliant… until the repo is way too big. I stopped “prompting” and started “compiling.”
New stack:
Claude Code → CCR → Recursive Language Model Gateway (REPL brain) → vLLM → MiniMax-M2.5

What the RLM layer actually does:
• loads the entire repo into a REPL workspace (not into tokens)
• writes program code to walk it, slices it, search it
• builds a tiny context pack for this task
• hands that to the model like a precompiled header

Make sure to Implement this completely end to end, so we have far less tokens usage, far less cost with real-infra engineers.
