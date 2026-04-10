package main

import (
	"fmt"
	"os"

	"openrouter-cli/internal/day19mcp"

	"github.com/mark3labs/mcp-go/server"
)

func main() {
	mcpServer := day19mcp.NewServer("day19-pipeline-mcp-server", "1.0.0")
	if err := server.ServeStdio(mcpServer); err != nil {
		fmt.Fprintf(os.Stderr, "failed to start day19 MCP stdio server: %v\n", err)
		os.Exit(1)
	}
}
