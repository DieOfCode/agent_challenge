# Project Structure

## Top-level files

- `main.go`: CLI routing and shared OpenRouter helper types.
- `agent.go`: base agent abstraction for prompt/response encapsulation.
- `day3.go` ... `day33.go`: feature tasks per day.

## MCP layout

- `cmd/day16_stdio_server`: day16 stdio MCP server.
- `cmd/day17_mcp_server`: day17 stdio MCP server.
- `cmd/day18_mcp_server`: day18 stdio MCP server.
- `cmd/day19_mcp_server`: day19 stdio MCP server.
- `cmd/day31_mcp_server`: day31 stdio MCP server.
- `cmd/day33_mcp_server`: day33 stdio MCP server.
- `internal/day17mcp`: TODO API MCP tools.
- `internal/day19mcp`: pipeline tools (search/summarize/save/verify).
- `internal/day31mcp`: project tools (`git_branch`, `list_files`, `git_diff`).
- `internal/day33mcp`: support tools (`get_user_profile`, `get_ticket`, `list_user_tickets`).

## RAG layout

- `day21.go`: index build/chunking/embeddings.
- `day22.go` .. `day25.go`: RAG retrieval, filters, grounding, and mini-chat.
- `DAY21_INDEX_*.json`: local index artifacts.

## Testing

- Each day can include `dayXX_test.go`.
- MCP internals can include tests under `internal/dayXXmcp/*_test.go`.
