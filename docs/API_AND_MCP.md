# API and MCP Notes

## LLM API

Main chat and most days use OpenRouter Chat Completions through helpers in `main.go`:

- `callOpenRouter(...)`
- `callOpenRouterDetailed(...)`

Environment variables used frequently:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`
- `OPENROUTER_EMBEDDING_MODEL`

## MCP basics in this project

Common client flow in day commands:

1. Create MCP client (`inprocess` or `stdio`)
2. `Initialize`
3. `ListTools`
4. `CallTool`

Day 31 requires at least `git_branch` tool and may use additional tools:

- `git_branch`: current branch + dirty status
- `list_files`: file listing for project context
- `git_diff`: current local diff snapshot

This MCP context is combined with retrieved docs context in `/help` answers.
