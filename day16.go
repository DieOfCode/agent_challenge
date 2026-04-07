package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

type day16Result struct {
	Transport     string
	ServerName    string
	ServerVersion string
	Protocol      string
	Connected     bool
	ToolNames     []string
}

func runDay16Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day16", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	transportMode := fs.String("transport", "inprocess", "MCP transport: inprocess|stdio")
	serverName := fs.String("server-name", "day16-local-mcp-server", "MCP server name")
	serverVersion := fs.String("server-version", "1.0.0", "MCP server version")
	stdioCommand := fs.String("stdio-command", "", "Stdio MCP server command (required for -transport stdio)")
	stdioArgsCSV := fs.String("stdio-args", "", "Comma-separated args for stdio command")
	stdioEnvCSV := fs.String("stdio-env", "", "Comma-separated env values KEY=VALUE for stdio command")
	timeout := fs.Duration("timeout", 8*time.Second, "Connection timeout")
	reportPath := fs.String("report", "DAY16_RESULTS.md", "Markdown report output path")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day16 flags: %w", err)
	}
	if *help {
		printDay16Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day16 arguments: %s", strings.Join(fs.Args(), " "))
	}

	mode := strings.ToLower(strings.TrimSpace(*transportMode))
	var (
		result day16Result
		err    error
	)
	switch mode {
	case "inprocess":
		result, err = connectAndListMCPToolsInProcess(*serverName, *serverVersion, *timeout)
	case "stdio":
		cmd := strings.TrimSpace(*stdioCommand)
		if cmd == "" {
			return fmt.Errorf("stdio transport requires -stdio-command (example: -stdio-command \"go\" -stdio-args \"run,./cmd/day16_stdio_server\")")
		}
		result, err = connectAndListMCPToolsStdio(cmd, parseCSVList(*stdioArgsCSV), parseCSVList(*stdioEnvCSV), *timeout)
	default:
		return fmt.Errorf("unsupported transport: %s (allowed: inprocess|stdio)", mode)
	}
	if err != nil {
		return err
	}

	printDay16Result(result)
	if err := writeDay16Report(*reportPath, result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)
	return nil
}

func connectAndListMCPToolsInProcess(serverName, serverVersion string, timeout time.Duration) (day16Result, error) {
	mcpServer := createDay16MCPServer(serverName, serverVersion)

	mcpClient, err := client.NewInProcessClient(mcpServer)
	if err != nil {
		return day16Result{}, fmt.Errorf("failed to create MCP client: %w", err)
	}
	defer mcpClient.Close()

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	if err := mcpClient.Start(ctx); err != nil {
		return day16Result{}, fmt.Errorf("failed to start MCP client transport: %w", err)
	}

	return initializeAndListTools(ctx, mcpClient, "inprocess")
}

func connectAndListMCPToolsStdio(command string, args []string, env []string, timeout time.Duration) (day16Result, error) {
	mcpClient, err := client.NewStdioMCPClient(command, env, args...)
	if err != nil {
		return day16Result{}, fmt.Errorf("failed to create stdio MCP client: %w", err)
	}
	defer mcpClient.Close()

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	return initializeAndListTools(ctx, mcpClient, "stdio")
}

func initializeAndListTools(ctx context.Context, mcpClient *client.Client, mode string) (day16Result, error) {
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{
		Name:    "openrouter-cli-day16-client",
		Version: "1.0.0",
	}
	initReq.Params.Capabilities = mcp.ClientCapabilities{}

	initRes, err := mcpClient.Initialize(ctx, initReq)
	if err != nil {
		return day16Result{}, fmt.Errorf("failed to initialize MCP connection: %w", err)
	}

	toolRes, err := mcpClient.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return day16Result{}, fmt.Errorf("failed to list MCP tools: %w", err)
	}
	if toolRes == nil {
		return day16Result{}, fmt.Errorf("nil tool list received from MCP server")
	}

	names := collectDay16ToolNames(toolRes.Tools)
	return day16Result{
		Transport:     mode,
		ServerName:    initRes.ServerInfo.Name,
		ServerVersion: initRes.ServerInfo.Version,
		Protocol:      initRes.ProtocolVersion,
		Connected:     true,
		ToolNames:     names,
	}, nil
}

func collectDay16ToolNames(tools []mcp.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		name := strings.TrimSpace(tool.Name)
		if name == "" {
			continue
		}
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func createDay16MCPServer(serverName, serverVersion string) *server.MCPServer {
	mcpServer := server.NewMCPServer(
		strings.TrimSpace(serverName),
		strings.TrimSpace(serverVersion),
		server.WithToolCapabilities(true),
	)

	mcpServer.AddTool(
		mcp.NewTool(
			"echo_text",
			mcp.WithDescription("Echo input text"),
			mcp.WithString("text", mcp.Description("Input text")),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			text := strings.TrimSpace(mcp.ParseString(request, "text", ""))
			if text == "" {
				text = "empty"
			}
			return mcp.NewToolResultText("echo: " + text), nil
		},
	)

	mcpServer.AddTool(
		mcp.NewTool(
			"sum_numbers",
			mcp.WithDescription("Sum two numbers"),
			mcp.WithNumber("a", mcp.Description("First number")),
			mcp.WithNumber("b", mcp.Description("Second number")),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			a := mcp.ParseFloat64(request, "a", 0)
			b := mcp.ParseFloat64(request, "b", 0)
			return mcp.NewToolResultText(fmt.Sprintf("%.2f", a+b)), nil
		},
	)

	return mcpServer
}

func printDay16Result(result day16Result) {
	fmt.Println("=== Day 16: MCP Client Connection ===")
	fmt.Printf("connection_established=%t transport=%s server=%s version=%s protocol=%s\n",
		result.Connected, day16EmptyFallback(result.Transport, "unknown"), result.ServerName, result.ServerVersion, result.Protocol,
	)
	fmt.Printf("tools_count=%d\n", len(result.ToolNames))
	for i, name := range result.ToolNames {
		fmt.Printf("%d. %s\n", i+1, name)
	}
}

func writeDay16Report(path string, result day16Result) error {
	var b strings.Builder
	b.WriteString("# Day 16 Results: MCP Connection and Tool Discovery\n\n")
	b.WriteString(fmt.Sprintf("- connection established: `%t`\n", result.Connected))
	b.WriteString(fmt.Sprintf("- transport: `%s`\n", day16EmptyFallback(result.Transport, "unknown")))
	b.WriteString(fmt.Sprintf("- server: `%s`\n", result.ServerName))
	b.WriteString(fmt.Sprintf("- server version: `%s`\n", result.ServerVersion))
	b.WriteString(fmt.Sprintf("- protocol: `%s`\n", result.Protocol))
	b.WriteString(fmt.Sprintf("- tools count: `%d`\n\n", len(result.ToolNames)))
	b.WriteString("## Tools\n")
	for _, name := range result.ToolNames {
		b.WriteString("- `" + name + "`\n")
	}
	b.WriteString("\nConclusion: MCP connection is established and tools are returned successfully.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay16Usage() {
	fmt.Println("Usage: openrouter-cli day16 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -transport string       MCP transport: inprocess|stdio")
	fmt.Println("  -server-name string     MCP server name")
	fmt.Println("  -server-version string  MCP server version")
	fmt.Println("  -stdio-command string   Stdio MCP server command")
	fmt.Println("  -stdio-args string      Comma-separated args for stdio command")
	fmt.Println("  -stdio-env string       Comma-separated env vars for stdio command")
	fmt.Println("  -timeout duration       Connection timeout (e.g. 8s)")
	fmt.Println("  -report string          Markdown report output path")
	fmt.Println("  -help                   Show help")
}

func day16EmptyFallback(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}
