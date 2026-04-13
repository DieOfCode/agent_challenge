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
	"openrouter-cli/internal/day19mcp"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

type day20Step struct {
	Task   string
	Tool   string
	Server string
	Reason string
}

type day20ServerInfo struct {
	Name      string
	Version   string
	Protocol  string
	Transport string
	Tools     []string
}

type day20RunResult struct {
	Servers       []day20ServerInfo
	Steps         []day20Step
	CorpusSource  string
	Search        day19mcp.SearchResult
	Summary       day19mcp.SummaryResult
	SummarySave   day19mcp.SaveResult
	SummaryVerify day19mcp.VerifyResult
	Scheduler     day18mcp.Summary
	FinalSave     day19mcp.SaveResult
	Query         string
}

func runDay20Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day20", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	transportMode := fs.String("transport", "inprocess", "MCP transport: inprocess|stdio")
	query := fs.String("query", "pipeline", "Search query")
	corpus := fs.String("corpus", "", "Inline corpus override")
	corpusFile := fs.String("corpus-file", "", "Path to corpus file")
	summaryOutput := fs.String("summary-output", "DAY20_SUMMARY.txt", "Path to save the summary")
	finalOutput := fs.String("output", "DAY20_FLOW_OUTPUT.txt", "Final output file path")
	maxSentences := fs.Int("max-sentences", 2, "Max sentences in summary")
	day18Interval := fs.Duration("day18-interval", 2*time.Second, "Day18 scheduler interval")
	day18Wait := fs.Duration("day18-wait", 3*time.Second, "Wait time before fetching scheduler summary")
	day18Command := fs.String("day18-stdio-command", "go", "Day18 stdio MCP server command")
	day18Args := fs.String("day18-stdio-args", "run,./cmd/day18_mcp_server", "Comma-separated args for day18 stdio command")
	day19Command := fs.String("day19-stdio-command", "go", "Day19 stdio MCP server command")
	day19Args := fs.String("day19-stdio-args", "run,./cmd/day19_mcp_server", "Comma-separated args for day19 stdio command")
	timeout := fs.Duration("timeout", 25*time.Second, "Total timeout")
	reportPath := fs.String("report", "DAY20_RESULTS.md", "Markdown report output path")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day20 flags: %w", err)
	}
	if *help {
		printDay20Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day20 arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *maxSentences <= 0 {
		return fmt.Errorf("max-sentences must be positive")
	}
	if *day18Interval <= 0 {
		return fmt.Errorf("day18-interval must be positive")
	}

	corpusText, corpusSource, err := resolveDay19Corpus(strings.TrimSpace(*corpusFile), strings.TrimSpace(*corpus))
	if err != nil {
		return err
	}

	result, err := runDay20Flow(day20FlowInput{
		Transport:     strings.ToLower(strings.TrimSpace(*transportMode)),
		Query:         strings.TrimSpace(*query),
		Corpus:        corpusText,
		CorpusSource:  corpusSource,
		SummaryOutput: strings.TrimSpace(*summaryOutput),
		FinalOutput:   strings.TrimSpace(*finalOutput),
		MaxSentences:  *maxSentences,
		Day18Interval: *day18Interval,
		Day18Wait:     *day18Wait,
		Day18Command:  strings.TrimSpace(*day18Command),
		Day18Args:     parseCSVList(*day18Args),
		Day19Command:  strings.TrimSpace(*day19Command),
		Day19Args:     parseCSVList(*day19Args),
		Timeout:       *timeout,
	})
	if err != nil {
		return err
	}

	printDay20Result(result)
	if err := writeDay20Report(*reportPath, result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)
	return nil
}

type day20FlowInput struct {
	Transport     string
	Query         string
	Corpus        string
	CorpusSource  string
	SummaryOutput string
	FinalOutput   string
	MaxSentences  int
	Day18Interval time.Duration
	Day18Wait     time.Duration
	Day18Command  string
	Day18Args     []string
	Day19Command  string
	Day19Args     []string
	Timeout       time.Duration
}

type day20Orchestrator struct {
	servers    map[string]*client.Client
	serverInfo map[string]day20ServerInfo
	toolIndex  map[string]string
}

func newDay20Orchestrator() *day20Orchestrator {
	return &day20Orchestrator{
		servers:    make(map[string]*client.Client),
		serverInfo: make(map[string]day20ServerInfo),
		toolIndex:  make(map[string]string),
	}
}

func (o *day20Orchestrator) Close() {
	for _, client := range o.servers {
		client.Close()
	}
}

func (o *day20Orchestrator) RegisterServer(ctx context.Context, key, transport string, client *client.Client) error {
	if _, exists := o.servers[key]; exists {
		return fmt.Errorf("server %s already registered", key)
	}
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{Name: "openrouter-cli-day20", Version: "1.0.0"}
	initReq.Params.Capabilities = mcp.ClientCapabilities{}

	initRes, err := client.Initialize(ctx, initReq)
	if err != nil {
		return fmt.Errorf("failed to initialize MCP connection for %s: %w", key, err)
	}

	toolRes, err := client.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return fmt.Errorf("failed to list tools for %s: %w", key, err)
	}
	if toolRes == nil {
		return fmt.Errorf("nil tool response for %s", key)
	}

	tools := day20CollectToolNames(toolRes.Tools)
	for _, toolName := range tools {
		o.toolIndex[toolName] = key
	}

	info := day20ServerInfo{
		Name:      strings.TrimSpace(initRes.ServerInfo.Name),
		Version:   strings.TrimSpace(initRes.ServerInfo.Version),
		Protocol:  strings.TrimSpace(initRes.ProtocolVersion),
		Transport: transport,
		Tools:     tools,
	}

	o.servers[key] = client
	o.serverInfo[key] = info
	return nil
}

func (o *day20Orchestrator) CallTool(ctx context.Context, toolName string, args map[string]any) (*mcp.CallToolResult, string, error) {
	serverKey, ok := o.toolIndex[toolName]
	if !ok {
		return nil, "", fmt.Errorf("tool %s not registered", toolName)
	}
	client := o.servers[serverKey]
	if client == nil {
		return nil, "", fmt.Errorf("server %s not found for tool %s", serverKey, toolName)
	}
	toolReq := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      toolName,
			Arguments: args,
		},
	}
	result, err := client.CallTool(ctx, toolReq)
	return result, serverKey, err
}

func runDay20Flow(input day20FlowInput) (day20RunResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), input.Timeout)
	defer cancel()

	orchestrator := newDay20Orchestrator()
	defer orchestrator.Close()

	day19Client, day19Transport, err := newDay20Day19Client(ctx, input.Transport, input.Day19Command, input.Day19Args)
	if err != nil {
		return day20RunResult{}, err
	}
	if err := orchestrator.RegisterServer(ctx, "day19", day19Transport, day19Client); err != nil {
		return day20RunResult{}, err
	}

	day18Client, day18Transport, err := newDay20Day18Client(ctx, input.Transport, input.Day18Interval, input.Day18Command, input.Day18Args)
	if err != nil {
		return day20RunResult{}, err
	}
	if err := orchestrator.RegisterServer(ctx, "day18", day18Transport, day18Client); err != nil {
		return day20RunResult{}, err
	}

	steps := make([]day20Step, 0)

	searchTool, reason := selectDay20Tool("search for query", orchestrator.toolIndex, []string{day19mcp.ToolSearch})
	searchResultRaw, serverKey, err := orchestrator.CallTool(ctx, searchTool, map[string]any{
		"query":  input.Query,
		"corpus": input.Corpus,
	})
	if err != nil {
		return day20RunResult{}, fmt.Errorf("search tool call failed: %w", err)
	}
	steps = append(steps, day20Step{Task: "search", Tool: searchTool, Server: serverKey, Reason: reason})
	searchResult, err := parseDay20SearchResult(searchResultRaw)
	if err != nil {
		return day20RunResult{}, err
	}

	summaryInput := strings.Join(searchResult.Matches, "\n")
	if strings.TrimSpace(summaryInput) == "" {
		summaryInput = fmt.Sprintf("No matches found for query: %s", input.Query)
	}

	summarizeTool, reason := selectDay20Tool("summarize text", orchestrator.toolIndex, []string{day19mcp.ToolSummarize})
	summaryRaw, serverKey, err := orchestrator.CallTool(ctx, summarizeTool, map[string]any{
		"text":          summaryInput,
		"max_sentences": input.MaxSentences,
	})
	if err != nil {
		return day20RunResult{}, fmt.Errorf("summarize tool call failed: %w", err)
	}
	steps = append(steps, day20Step{Task: "summarize", Tool: summarizeTool, Server: serverKey, Reason: reason})
	summaryResult, err := parseDay20SummaryResult(summaryRaw)
	if err != nil {
		return day20RunResult{}, err
	}

	saveTool, reason := selectDay20Tool("save summary", orchestrator.toolIndex, []string{day19mcp.ToolSaveToFile})
	summarySaveRaw, serverKey, err := orchestrator.CallTool(ctx, saveTool, map[string]any{
		"path":    input.SummaryOutput,
		"content": summaryResult.Summary,
	})
	if err != nil {
		return day20RunResult{}, fmt.Errorf("save summary tool call failed: %w", err)
	}
	steps = append(steps, day20Step{Task: "save summary", Tool: saveTool, Server: serverKey, Reason: reason})
	summarySave, err := parseDay20SaveResult(summarySaveRaw)
	if err != nil {
		return day20RunResult{}, err
	}

	verifyTool, reason := selectDay20Tool("verify summary", orchestrator.toolIndex, []string{day19mcp.ToolVerifyFile})
	verifyRaw, serverKey, err := orchestrator.CallTool(ctx, verifyTool, map[string]any{
		"path":              input.SummaryOutput,
		"expected_contains": summaryResult.Summary,
	})
	if err != nil {
		return day20RunResult{}, fmt.Errorf("verify tool call failed: %w", err)
	}
	steps = append(steps, day20Step{Task: "verify summary", Tool: verifyTool, Server: serverKey, Reason: reason})
	summaryVerify, err := parseDay20VerifyResult(verifyRaw)
	if err != nil {
		return day20RunResult{}, err
	}

	if input.Day18Wait > 0 {
		time.Sleep(input.Day18Wait)
	}

	schedulerTool, reason := selectDay20Tool("scheduler summary", orchestrator.toolIndex, []string{day18mcp.ToolGetSummary})
	schedulerRaw, serverKey, err := orchestrator.CallTool(ctx, schedulerTool, map[string]any{
		"window_minutes": 60,
	})
	if err != nil {
		return day20RunResult{}, fmt.Errorf("scheduler tool call failed: %w", err)
	}
	steps = append(steps, day20Step{Task: "scheduler summary", Tool: schedulerTool, Server: serverKey, Reason: reason})
	schedulerSummary, err := parseDay20SchedulerResult(schedulerRaw)
	if err != nil {
		return day20RunResult{}, err
	}

	finalContent := buildDay20FinalContent(summaryResult, schedulerSummary)
	finalSaveTool, reason := selectDay20Tool("save final output", orchestrator.toolIndex, []string{day19mcp.ToolSaveToFile})
	finalSaveRaw, serverKey, err := orchestrator.CallTool(ctx, finalSaveTool, map[string]any{
		"path":    input.FinalOutput,
		"content": finalContent,
	})
	if err != nil {
		return day20RunResult{}, fmt.Errorf("final save tool call failed: %w", err)
	}
	steps = append(steps, day20Step{Task: "save final output", Tool: finalSaveTool, Server: serverKey, Reason: reason})
	finalSave, err := parseDay20SaveResult(finalSaveRaw)
	if err != nil {
		return day20RunResult{}, err
	}

	servers := make([]day20ServerInfo, 0, len(orchestrator.serverInfo))
	for _, info := range orchestrator.serverInfo {
		servers = append(servers, info)
	}
	sort.Slice(servers, func(i, j int) bool {
		return servers[i].Name < servers[j].Name
	})

	return day20RunResult{
		Servers:       servers,
		Steps:         steps,
		CorpusSource:  input.CorpusSource,
		Search:        searchResult,
		Summary:       summaryResult,
		SummarySave:   summarySave,
		SummaryVerify: summaryVerify,
		Scheduler:     schedulerSummary,
		FinalSave:     finalSave,
		Query:         input.Query,
	}, nil
}

func newDay20Day19Client(ctx context.Context, transport, command string, args []string) (*client.Client, string, error) {
	switch transport {
	case "inprocess":
		mcpServer := day19mcp.NewServer("day19-orchestrator-server", "1.0.0")
		mcpClient, err := client.NewInProcessClient(mcpServer)
		if err != nil {
			return nil, "", fmt.Errorf("failed to create day19 inprocess client: %w", err)
		}
		if err := mcpClient.Start(ctx); err != nil {
			return nil, "", fmt.Errorf("failed to start day19 inprocess transport: %w", err)
		}
		return mcpClient, "inprocess", nil
	case "stdio":
		cmd := strings.TrimSpace(command)
		if cmd == "" {
			return nil, "", fmt.Errorf("day19 stdio command is empty")
		}
		mcpClient, err := client.NewStdioMCPClient(cmd, nil, args...)
		if err != nil {
			return nil, "", fmt.Errorf("failed to create day19 stdio client: %w", err)
		}
		return mcpClient, "stdio", nil
	default:
		return nil, "", fmt.Errorf("unsupported transport: %s (allowed: inprocess|stdio)", transport)
	}
}

func newDay20Day18Client(ctx context.Context, transport string, interval time.Duration, command string, args []string) (*client.Client, string, error) {
	switch transport {
	case "inprocess":
		mcpServer := day18mcp.NewServer("day18-orchestrator-server", "1.0.0", day18mcp.DefaultStorePath, interval)
		mcpClient, err := client.NewInProcessClient(mcpServer)
		if err != nil {
			return nil, "", fmt.Errorf("failed to create day18 inprocess client: %w", err)
		}
		if err := mcpClient.Start(ctx); err != nil {
			return nil, "", fmt.Errorf("failed to start day18 inprocess transport: %w", err)
		}
		return mcpClient, "inprocess", nil
	case "stdio":
		cmd := strings.TrimSpace(command)
		if cmd == "" {
			return nil, "", fmt.Errorf("day18 stdio command is empty")
		}
		env := []string{fmt.Sprintf("DAY18_INTERVAL_SECONDS=%d", int(interval.Seconds()))}
		mcpClient, err := client.NewStdioMCPClient(cmd, env, args...)
		if err != nil {
			return nil, "", fmt.Errorf("failed to create day18 stdio client: %w", err)
		}
		return mcpClient, "stdio", nil
	default:
		return nil, "", fmt.Errorf("unsupported transport: %s (allowed: inprocess|stdio)", transport)
	}
}

func selectDay20Tool(task string, toolIndex map[string]string, preferred []string) (string, string) {
	lower := strings.ToLower(task)
	for _, tool := range preferred {
		if _, ok := toolIndex[tool]; ok {
			reason := fmt.Sprintf("selected %s for task '%s'", tool, lower)
			return tool, reason
		}
	}
	keywordMap := map[string]string{
		"search":   day19mcp.ToolSearch,
		"summar":   day19mcp.ToolSummarize,
		"save":     day19mcp.ToolSaveToFile,
		"verify":   day19mcp.ToolVerifyFile,
		"schedule": day18mcp.ToolGetSummary,
	}
	for keyword, tool := range keywordMap {
		if strings.Contains(lower, keyword) {
			if _, ok := toolIndex[tool]; ok {
				reason := fmt.Sprintf("matched keyword '%s'", keyword)
				return tool, reason
			}
		}
	}
	for tool := range toolIndex {
		reason := "fallback to first available tool"
		return tool, reason
	}
	return "", "no tool available"
}

func parseDay20SearchResult(result *mcp.CallToolResult) (day19mcp.SearchResult, error) {
	var out day19mcp.SearchResult
	if parseDay20Structured(result, &out) && out.Query != "" {
		return out, nil
	}
	return day19mcp.SearchResult{}, fmt.Errorf("failed to parse search result")
}

func parseDay20SummaryResult(result *mcp.CallToolResult) (day19mcp.SummaryResult, error) {
	var out day19mcp.SummaryResult
	if parseDay20Structured(result, &out) && out.Summary != "" {
		return out, nil
	}
	return day19mcp.SummaryResult{}, fmt.Errorf("failed to parse summary result")
}

func parseDay20SaveResult(result *mcp.CallToolResult) (day19mcp.SaveResult, error) {
	var out day19mcp.SaveResult
	if parseDay20Structured(result, &out) && out.Path != "" {
		return out, nil
	}
	return day19mcp.SaveResult{}, fmt.Errorf("failed to parse save result")
}

func parseDay20VerifyResult(result *mcp.CallToolResult) (day19mcp.VerifyResult, error) {
	var out day19mcp.VerifyResult
	if parseDay20Structured(result, &out) && out.Path != "" {
		return out, nil
	}
	return day19mcp.VerifyResult{}, fmt.Errorf("failed to parse verify result")
}

func parseDay20SchedulerResult(result *mcp.CallToolResult) (day18mcp.Summary, error) {
	var out day18mcp.Summary
	if parseDay20Structured(result, &out) && out.WindowMinutes > 0 {
		return out, nil
	}
	return day18mcp.Summary{}, fmt.Errorf("failed to parse scheduler summary")
}

func parseDay20Structured(result *mcp.CallToolResult, out any) bool {
	if result == nil {
		return false
	}
	if result.StructuredContent != nil {
		raw, err := json.Marshal(result.StructuredContent)
		if err == nil && json.Unmarshal(raw, out) == nil {
			return true
		}
	}
	for _, content := range result.Content {
		textContent, ok := content.(mcp.TextContent)
		if !ok {
			continue
		}
		text := strings.TrimSpace(textContent.Text)
		if text == "" {
			continue
		}
		if json.Unmarshal([]byte(text), out) == nil {
			return true
		}
	}
	return false
}

func day20CollectToolNames(tools []mcp.Tool) []string {
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

func buildDay20FinalContent(summary day19mcp.SummaryResult, scheduler day18mcp.Summary) string {
	var b strings.Builder
	b.WriteString("Pipeline Summary\n")
	b.WriteString("--------------\n")
	b.WriteString(summary.Summary + "\n\n")
	b.WriteString("Scheduler Snapshot\n")
	b.WriteString("------------------\n")
	b.WriteString(fmt.Sprintf("total_runs=%d window_runs=%d window_minutes=%d last_run=%s\n", scheduler.TotalRuns, scheduler.WindowRuns, scheduler.WindowMinutes, scheduler.LastRunAt))
	return b.String()
}

func printDay20Result(result day20RunResult) {
	fmt.Println("=== Day 20: MCP Orchestration ===")
	fmt.Printf("servers=%d steps=%d\n", len(result.Servers), len(result.Steps))
	for i, server := range result.Servers {
		fmt.Printf("server_%d=%s transport=%s tools=%d\n", i+1, day16EmptyFallback(server.Name, "unknown"), day16EmptyFallback(server.Transport, "unknown"), len(server.Tools))
	}
	for i, step := range result.Steps {
		fmt.Printf("step_%d task=%s tool=%s server=%s\n", i+1, step.Task, step.Tool, step.Server)
	}
	fmt.Printf("query=%s corpus_source=%s matches=%d\n", result.Query, result.CorpusSource, result.Search.Count)
	fmt.Printf("summary_saved=%s verify_contains=%t\n", result.SummarySave.Path, result.SummaryVerify.Contains)
	fmt.Printf("scheduler_total_runs=%d final_output=%s\n", result.Scheduler.TotalRuns, result.FinalSave.Path)
}

func writeDay20Report(path string, result day20RunResult) error {
	var b strings.Builder
	b.WriteString("# Day 20 Results: MCP Orchestration\n\n")
	b.WriteString(fmt.Sprintf("- servers registered: `%d`\n", len(result.Servers)))
	b.WriteString(fmt.Sprintf("- steps executed: `%d`\n", len(result.Steps)))
	b.WriteString(fmt.Sprintf("- query: `%s`\n", result.Query))
	b.WriteString(fmt.Sprintf("- corpus source: `%s`\n\n", result.CorpusSource))

	b.WriteString("## Servers\n")
	for _, server := range result.Servers {
		b.WriteString(fmt.Sprintf("- %s (transport: `%s`, tools: %d)\n", day16EmptyFallback(server.Name, "unknown"), server.Transport, len(server.Tools)))
	}

	b.WriteString("\n## Tool Routing\n")
	for _, step := range result.Steps {
		b.WriteString(fmt.Sprintf("- task: %s -> tool: `%s` (server: %s, reason: %s)\n", step.Task, step.Tool, step.Server, step.Reason))
	}

	b.WriteString("\n## Pipeline Outputs\n")
	b.WriteString(fmt.Sprintf("- search matches: `%d`\n", result.Search.Count))
	b.WriteString(fmt.Sprintf("- summary saved: `%s`\n", result.SummarySave.Path))
	b.WriteString(fmt.Sprintf("- verify contains: `%t`\n", result.SummaryVerify.Contains))
	b.WriteString(fmt.Sprintf("- scheduler total runs: `%d`\n", result.Scheduler.TotalRuns))
	b.WriteString(fmt.Sprintf("- final output: `%s`\n", result.FinalSave.Path))
	b.WriteString("\nConclusion: orchestrator routed calls across multiple MCP servers in a long flow (search -> summarize -> save -> verify -> scheduler summary -> save).\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay20Usage() {
	fmt.Println("Usage: openrouter-cli day20 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -transport string        MCP transport: inprocess|stdio")
	fmt.Println("  -query string            Search query")
	fmt.Println("  -corpus string           Inline corpus override")
	fmt.Println("  -corpus-file string      Path to corpus file")
	fmt.Println("  -summary-output string   Path to save summary")
	fmt.Println("  -output string           Path to save final output")
	fmt.Println("  -max-sentences int       Max sentences in summary")
	fmt.Println("  -day18-interval duration Scheduler interval for day18")
	fmt.Println("  -day18-wait duration     Wait before calling day18 summary")
	fmt.Println("  -day18-stdio-command     Day18 stdio command")
	fmt.Println("  -day18-stdio-args string Comma-separated day18 args")
	fmt.Println("  -day19-stdio-command     Day19 stdio command")
	fmt.Println("  -day19-stdio-args string Comma-separated day19 args")
	fmt.Println("  -timeout duration        Total timeout")
	fmt.Println("  -report string           Markdown report path")
	fmt.Println("  -help                    Show help")
}
