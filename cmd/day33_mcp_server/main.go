package main

import (
	"fmt"
	"os"
	"strings"

	"openrouter-cli/internal/day33mcp"

	"github.com/mark3labs/mcp-go/server"
)

func main() {
	usersFile := strings.TrimSpace(os.Getenv("DAY33_USERS_FILE"))
	ticketsFile := strings.TrimSpace(os.Getenv("DAY33_TICKETS_FILE"))
	mcpServer := day33mcp.NewServer("day33-support-mcp-server", "1.0.0", usersFile, ticketsFile)
	if err := server.ServeStdio(mcpServer); err != nil {
		fmt.Fprintf(os.Stderr, "failed to start day33 MCP stdio server: %v\n", err)
		os.Exit(1)
	}
}
