package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"openrouter-cli/internal/day18mcp"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

type day18RunResult struct {
	Transport     string
	ServerName    string
	ServerVersion string
	Protocol      string
	Tools         []string
	Summary       day18mcp.Summary
	WindowMinutes int
}

func runDay18Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day18", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	transportMode := fs.String("transport", "stdio", "MCP transport: stdio|inprocess")
	storePath := fs.String("store", day18mcp.DefaultStorePath, "Path to scheduler JSON store")
	interval := fs.Duration("interval", 5*time.Second, "Scheduler interval (e.g. 5s)")
	windowMinutes := fs.Int("window-minutes", 60, "Window minutes for summary aggregation")
	stdioCommand := fs.String("stdio-command", "go", "Stdio MCP server command")
	stdioArgsCSV := fs.String("stdio-args", "run,./cmd/day18_mcp_server", "Comma-separated args for stdio command")
	stdioEnvCSV := fs.String("stdio-env", "", "Comma-separated env KEY=VALUE for stdio command")
	wait := fs.Duration("wait", 6*time.Second, "Time to wait for scheduler ticks")
	timeout := fs.Duration("timeout", 25*time.Second, "Total timeout")
	reportPath := fs.String("report", "DAY18_RESULTS.md", "Markdown report output path")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day18 flags: %w", err)
	}
	if *help {
		printDay18Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day18 arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *windowMinutes <= 0 {
		return fmt.Errorf("window-minutes must be positive")
	}
	if *interval <= 0 {
		return fmt.Errorf("interval must be positive")
	}

	result, err := runDay18Flow(
		strings.ToLower(strings.TrimSpace(*transportMode)),
		strings.TrimSpace(*storePath),
		*interval,
		*windowMinutes,
		*wait,
		strings.TrimSpace(*stdioCommand),
		parseCSVList(*stdioArgsCSV),
		parseCSVList(*stdioEnvCSV),
		*timeout,
	)
	if err != nil {
		return err
	}

	printDay18Result(result)
	if err := writeDay18Report(*reportPath, result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)
	return nil
}

func runDay18Flow(transportMode, storePath string, interval time.Duration, windowMinutes int, wait time.Duration, stdioCommand string, stdioArgs, stdioEnv []string, timeout time.Duration) (day18RunResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	client, transportLabel, err := newDay18MCPClient(ctx, transportMode, storePath, interval, stdioCommand, stdioArgs, stdioEnv)
	if err != nil {
		return day18RunResult{}, err
	}
	defer client.Close()

	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{Name: "openrouter-cli-day18", Version: "1.0.0"}
	initReq.Params.Capabilities = mcp.ClientCapabilities{}

	initRes, err := client.Initialize(ctx, initReq)
	if err != nil {
		return day18RunResult{}, fmt.Errorf("failed to initialize MCP connection: %w", err)
	}

	toolRes, err := client.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return day18RunResult{}, fmt.Errorf("failed to list MCP tools: %w", err)
	}
	if toolRes == nil {
		return day18RunResult{}, fmt.Errorf("nil tools response")
	}

	if wait > 0 {
		time.Sleep(wait)
	}

	summary, err := callDay18SummaryTool(ctx, client, windowMinutes)
	if err != nil {
		return day18RunResult{}, err
	}

	return day18RunResult{
		Transport:     transportLabel,
		ServerName:    strings.TrimSpace(initRes.ServerInfo.Name),
		ServerVersion: strings.TrimSpace(initRes.ServerInfo.Version),
		Protocol:      strings.TrimSpace(initRes.ProtocolVersion),
		Tools:         day18CollectToolNames(toolRes.Tools),
		Summary:       summary,
		WindowMinutes: windowMinutes,
	}, nil
}

func newDay18MCPClient(ctx context.Context, transportMode, storePath string, interval time.Duration, stdioCommand string, stdioArgs, stdioEnv []string) (*client.Client, string, error) {
	switch transportMode {
	case "inprocess":
		mcpServer := day18mcp.NewServer("day18-inprocess-server", "1.0.0", storePath, interval)
		mcpClient, err := client.NewInProcessClient(mcpServer)
		if err != nil {
			return nil, "", fmt.Errorf("failed to create inprocess MCP client: %w", err)
		}
		if err := mcpClient.Start(ctx); err != nil {
			return nil, "", fmt.Errorf("failed to start inprocess MCP transport: %w", err)
		}
		return mcpClient, "inprocess", nil
	case "stdio":
		cmd := strings.TrimSpace(stdioCommand)
		if cmd == "" {
			return nil, "", fmt.Errorf("stdio command is empty")
		}
		env := append([]string(nil), stdioEnv...)
		env = append(env,
			"DAY18_STORE_PATH="+storePath,
			fmt.Sprintf("DAY18_INTERVAL_SECONDS=%d", int(interval.Seconds())),
		)
		mcpClient, err := client.NewStdioMCPClient(cmd, env, stdioArgs...)
		if err != nil {
			return nil, "", fmt.Errorf("failed to create stdio MCP client: %w", err)
		}
		return mcpClient, "stdio", nil
	default:
		return nil, "", fmt.Errorf("unsupported transport: %s (allowed: stdio|inprocess)", transportMode)
	}
}

func callDay18SummaryTool(ctx context.Context, mcpClient *client.Client, windowMinutes int) (day18mcp.Summary, error) {
	toolReq := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: day18mcp.ToolGetSummary,
			Arguments: map[string]any{
				"window_minutes": windowMinutes,
			},
		},
	}
	result, err := mcpClient.CallTool(ctx, toolReq)
	if err != nil {
		return day18mcp.Summary{}, fmt.Errorf("summary tool call failed: %w", err)
	}
	if result == nil {
		return day18mcp.Summary{}, fmt.Errorf("summary tool result is nil")
	}
	if result.IsError {
		return day18mcp.Summary{}, fmt.Errorf("summary tool error: %s", day18ToolResultText(result))
	}
	summary, err := parseDay18Summary(result)
	if err != nil {
		return day18mcp.Summary{}, err
	}
	return summary, nil
}

func parseDay18Summary(result *mcp.CallToolResult) (day18mcp.Summary, error) {
	if result.StructuredContent != nil {
		raw, err := jsonMarshal(result.StructuredContent)
		if err == nil {
			var summary day18mcp.Summary
			if err := jsonUnmarshal(raw, &summary); err == nil {
				return summary, nil
			}
		}
	}
	return day18mcp.Summary{}, fmt.Errorf("failed to parse summary result")
}

func day18CollectToolNames(tools []mcp.Tool) []string {
	out := make([]string, 0, len(tools))
	for _, tool := range tools {
		name := strings.TrimSpace(tool.Name)
		if name == "" {
			continue
		}
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

func printDay18Result(result day18RunResult) {
	fmt.Println("=== Day 18: Scheduler and Background Tasks ===")
	fmt.Printf("connection_established=%t transport=%s server=%s version=%s protocol=%s\n",
		true,
		day16EmptyFallback(result.Transport, "unknown"),
		day16EmptyFallback(result.ServerName, "unknown"),
		day16EmptyFallback(result.ServerVersion, "unknown"),
		day16EmptyFallback(result.Protocol, "unknown"),
	)
	fmt.Printf("tools_count=%d\n", len(result.Tools))
	for i, name := range result.Tools {
		fmt.Printf("%d. %s\n", i+1, name)
	}
	fmt.Printf("summary_window_minutes=%d total_runs=%d window_runs=%d last_run=%s store=%s active=%t\n",
		result.Summary.WindowMinutes,
		result.Summary.TotalRuns,
		result.Summary.WindowRuns,
		day16EmptyFallback(result.Summary.LastRunAt, "-"),
		result.Summary.StorePath,
		result.Summary.SchedulerActive,
	)
}

func writeDay18Report(path string, result day18RunResult) error {
	var b strings.Builder
	b.WriteString("# Day 18 Results: Scheduler and Background Tasks\n\n")
	b.WriteString(fmt.Sprintf("- connection established: `%t`\n", true))
	b.WriteString(fmt.Sprintf("- transport: `%s`\n", result.Transport))
	b.WriteString(fmt.Sprintf("- server: `%s`\n", result.ServerName))
	b.WriteString(fmt.Sprintf("- server version: `%s`\n", result.ServerVersion))
	b.WriteString(fmt.Sprintf("- protocol: `%s`\n", result.Protocol))
	b.WriteString(fmt.Sprintf("- tools count: `%d`\n\n", len(result.Tools)))
	b.WriteString("## Tools\n")
	for _, name := range result.Tools {
		b.WriteString("- `" + name + "`\n")
	}
	b.WriteString("\n## Summary\n")
	b.WriteString(fmt.Sprintf("- total runs: `%d`\n", result.Summary.TotalRuns))
	b.WriteString(fmt.Sprintf("- window runs (%d min): `%d`\n", result.Summary.WindowMinutes, result.Summary.WindowRuns))
	b.WriteString(fmt.Sprintf("- last run: `%s`\n", day16EmptyFallback(result.Summary.LastRunAt, "-")))
	b.WriteString(fmt.Sprintf("- store path: `%s`\n", result.Summary.StorePath))
	b.WriteString(fmt.Sprintf("- scheduler active: `%t`\n", result.Summary.SchedulerActive))
	b.WriteString("\nConclusion: scheduler runs periodically, stores data, and aggregated summary is returned via MCP tool.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay18Usage() {
	fmt.Println("Usage: openrouter-cli day18 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -transport string       MCP transport: stdio|inprocess")
	fmt.Println("  -store string           Path to scheduler JSON store")
	fmt.Println("  -interval duration      Scheduler interval (e.g. 5s)")
	fmt.Println("  -window-minutes int     Window minutes for summary aggregation")
	fmt.Println("  -stdio-command string   Stdio MCP server command")
	fmt.Println("  -stdio-args string      Comma-separated args for stdio command")
	fmt.Println("  -stdio-env string       Comma-separated env vars for stdio command")
	fmt.Println("  -wait duration          Wait time before calling summary")
	fmt.Println("  -timeout duration       Total timeout (e.g. 25s)")
	fmt.Println("  -report string          Markdown report path")
	fmt.Println("  -help                   Show help")
}

func day18ToolResultText(result *mcp.CallToolResult) string {
	if result == nil {
		return ""
	}
	parts := make([]string, 0, len(result.Content))
	for _, content := range result.Content {
		if textContent, ok := content.(mcp.TextContent); ok {
			text := strings.TrimSpace(textContent.Text)
			if text != "" {
				parts = append(parts, text)
			}
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

// Minimal JSON helpers to avoid extra imports in day18.go.
func jsonMarshal(v any) ([]byte, error) {
	return json.Marshal(v)
}

func jsonUnmarshal(data []byte, v any) error {
	return json.Unmarshal(data, v)
}
