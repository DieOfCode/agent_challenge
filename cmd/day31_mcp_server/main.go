package main

import (
	"fmt"
	"os"
	"strings"

	"openrouter-cli/internal/day31mcp"

	"github.com/mark3labs/mcp-go/server"
)

func main() {
	workspace := strings.TrimSpace(os.Getenv("DAY31_WORKSPACE"))
	mcpServer := day31mcp.NewServer("day31-dev-assistant-mcp-server", "1.0.0", workspace)
	if err := server.ServeStdio(mcpServer); err != nil {
		fmt.Fprintf(os.Stderr, "failed to start day31 MCP stdio server: %v\n", err)
		os.Exit(1)
	}
}
