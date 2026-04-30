# AGENT CHALLENGE CLI

`openrouter-cli` is a Go-based learning project that incrementally builds an LLM assistant over 30+ daily tasks.

## What is in this repository

- CLI commands `day3` ... `day33`
- Agent workflows with memory and state (`agent.go`, `day11`..`day15`)
- MCP examples (`day16`..`day20` + `internal/day17mcp`, `internal/day19mcp`, `internal/day31mcp`, `internal/day33mcp`)
- RAG pipeline and local LLM integrations (`day21`..`day29`)

## Core entrypoints

- `main.go` routes all subcommands.
- `dayXX.go` files hold each day's implementation.
- `cmd/*` contains stdio MCP server launchers.
- `internal/*` contains reusable MCP server logic.

## Day 33 scope

Day 33 adds a support assistant command that uses:

1. RAG over `README` + `docs/` + support FAQ.
2. MCP tools for support context (`get_user_profile`, `get_ticket`, `list_user_tickets`).
3. User/ticket-aware answers for support questions.

See `docs/PROJECT_STRUCTURE.md`, `docs/API_AND_MCP.md`, and `docs/SUPPORT_FAQ.md`.
