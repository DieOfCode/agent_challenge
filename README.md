# AGENT CHALLENGE CLI

`openrouter-cli` is a Go-based learning project that incrementally builds an LLM assistant over 30+ daily tasks.

## What is in this repository

- CLI commands `day3` ... `day31`
- Agent workflows with memory and state (`agent.go`, `day11`..`day15`)
- MCP examples (`day16`..`day20` + `internal/day17mcp`, `internal/day19mcp`, `internal/day31mcp`)
- RAG pipeline and local LLM integrations (`day21`..`day29`)

## Core entrypoints

- `main.go` routes all subcommands.
- `dayXX.go` files hold each day's implementation.
- `cmd/*` contains stdio MCP server launchers.
- `internal/*` contains reusable MCP server logic.

## Day 31 scope

Day 31 adds a developer assistant command that supports `/help` and uses:

1. RAG over this `README` + `docs/`
2. MCP tool `git_branch` (and optional file/diff tools)
3. Project-aware answers for structure and workflow questions

See `docs/PROJECT_STRUCTURE.md` and `docs/API_AND_MCP.md` for details.
