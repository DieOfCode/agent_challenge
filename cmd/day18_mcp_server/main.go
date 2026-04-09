package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"openrouter-cli/internal/day18mcp"

	"github.com/mark3labs/mcp-go/server"
)

func main() {
	storePath := strings.TrimSpace(os.Getenv("DAY18_STORE_PATH"))
	interval := parseIntervalEnv(os.Getenv("DAY18_INTERVAL_SECONDS"))

	mcpServer := day18mcp.NewServer("day18-scheduler-mcp-server", "1.0.0", storePath, interval)
	if err := server.ServeStdio(mcpServer); err != nil {
		fmt.Fprintf(os.Stderr, "failed to start day18 MCP stdio server: %v\n", err)
		os.Exit(1)
	}
}

func parseIntervalEnv(raw string) time.Duration {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 5 * time.Second
	}
	val, err := strconv.Atoi(raw)
	if err != nil || val <= 0 {
		return 5 * time.Second
	}
	return time.Duration(val) * time.Second
}

