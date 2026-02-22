# AGENTS Instructions

Scope: entire repository.

## Required Startup Behavior
- At the start of every user conversation and at the start of each new task, read `AGENT_MEMORY.md`.
- Treat `AGENT_MEMORY.md` as the default preference source for collaboration style, workflow, and output expectations.
- If a user request conflicts with `AGENT_MEMORY.md`, follow the user request for that turn and keep the rest of `AGENT_MEMORY.md` preferences.

## Ongoing Behavior
- Re-check `AGENT_MEMORY.md` before making substantial changes or if the user asks for a style/process change.
- Keep responses and edits consistent with `AGENT_MEMORY.md` unless explicitly overridden by the user.

