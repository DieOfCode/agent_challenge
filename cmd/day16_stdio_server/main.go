package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func main() {
	mcpServer := server.NewMCPServer(
		"day16-stdio-server",
		"1.0.0",
		server.WithToolCapabilities(true),
	)

	mcpServer.AddTool(
		mcp.NewTool(
			"upper_text",
			mcp.WithDescription("Convert text to upper case"),
			mcp.WithString("text", mcp.Description("Input text")),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			value := mcp.ParseString(request, "text", "")
			return mcp.NewToolResultText(strings.ToUpper(strings.TrimSpace(value))), nil
		},
	)

	mcpServer.AddTool(
		mcp.NewTool(
			"system_time",
			mcp.WithDescription("Return current UTC time"),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return mcp.NewToolResultText(time.Now().UTC().Format(time.RFC3339)), nil
		},
	)

	if err := server.ServeStdio(mcpServer); err != nil {
		fmt.Fprintf(os.Stderr, "failed to serve stdio MCP server: %v\n", err)
		os.Exit(1)
	}
}
