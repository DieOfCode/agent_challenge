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

	"openrouter-cli/internal/day19mcp"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

type day19RunResult struct {
	Transport     string
	ServerName    string
	ServerVersion string
	Protocol      string
	Tools         []string
	CorpusSource  string
	Search        day19mcp.SearchResult
	Summary       day19mcp.SummaryResult
	Save          day19mcp.SaveResult
}

func runDay19Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day19", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	transportMode := fs.String("transport", "stdio", "MCP transport: stdio|inprocess")
	query := fs.String("query", "pipeline", "Search query")
	corpus := fs.String("corpus", "", "Inline corpus override")
	corpusFile := fs.String("corpus-file", "", "Path to corpus file")
	outputPath := fs.String("output", "DAY19_PIPELINE_OUTPUT.txt", "Output file to save summary")
	maxSentences := fs.Int("max-sentences", 2, "Max sentences in summary")
	stdioCommand := fs.String("stdio-command", "go", "Stdio MCP server command")
	stdioArgsCSV := fs.String("stdio-args", "run,./cmd/day19_mcp_server", "Comma-separated args for stdio command")
	stdioEnvCSV := fs.String("stdio-env", "", "Comma-separated env KEY=VALUE for stdio command")
	timeout := fs.Duration("timeout", 20*time.Second, "Total timeout")
	reportPath := fs.String("report", "DAY19_RESULTS.md", "Markdown report output path")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day19 flags: %w", err)
	}
	if *help {
		printDay19Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day19 arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *maxSentences <= 0 {
		return fmt.Errorf("max-sentences must be positive")
	}

	corpusText, corpusSource, err := resolveDay19Corpus(strings.TrimSpace(*corpusFile), strings.TrimSpace(*corpus))
	if err != nil {
		return err
	}

	result, err := runDay19Flow(
		strings.ToLower(strings.TrimSpace(*transportMode)),
		strings.TrimSpace(*query),
		corpusText,
		corpusSource,
		strings.TrimSpace(*outputPath),
		*maxSentences,
		strings.TrimSpace(*stdioCommand),
		parseCSVList(*stdioArgsCSV),
		parseCSVList(*stdioEnvCSV),
		*timeout,
	)
	if err != nil {
		return err
	}

	printDay19Result(result)
	if err := writeDay19Report(*reportPath, result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)
	return nil
}

func resolveDay19Corpus(corpusFile, corpusInline string) (string, string, error) {
	if corpusFile != "" {
		text, err := day19mcp.LoadCorpusFromFile(corpusFile)
		if err != nil {
			return "", "", fmt.Errorf("failed to read corpus file: %w", err)
		}
		return text, corpusFile, nil
	}
	if corpusInline != "" {
		return corpusInline, "inline", nil
	}
	return day19mcp.DefaultCorpus, "default", nil
}

func runDay19Flow(transportMode, query, corpus, corpusSource, outputPath string, maxSentences int, stdioCommand string, stdioArgs, stdioEnv []string, timeout time.Duration) (day19RunResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	mcpClient, transportLabel, err := newDay19MCPClient(ctx, transportMode, stdioCommand, stdioArgs, stdioEnv)
	if err != nil {
		return day19RunResult{}, err
	}
	defer mcpClient.Close()

	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{Name: "openrouter-cli-day19", Version: "1.0.0"}
	initReq.Params.Capabilities = mcp.ClientCapabilities{}

	initRes, err := mcpClient.Initialize(ctx, initReq)
	if err != nil {
		return day19RunResult{}, fmt.Errorf("failed to initialize MCP connection: %w", err)
	}

	toolRes, err := mcpClient.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return day19RunResult{}, fmt.Errorf("failed to list MCP tools: %w", err)
	}
	if toolRes == nil {
		return day19RunResult{}, fmt.Errorf("nil tools response")
	}

	searchResult, err := callDay19Search(ctx, mcpClient, query, corpus)
	if err != nil {
		return day19RunResult{}, err
	}

	summaryInput := strings.Join(searchResult.Matches, "\n")
	if strings.TrimSpace(summaryInput) == "" {
		summaryInput = fmt.Sprintf("No matches found for query: %s", query)
	}

	summaryResult, err := callDay19Summarize(ctx, mcpClient, summaryInput, maxSentences)
	if err != nil {
		return day19RunResult{}, err
	}

	saveResult, err := callDay19Save(ctx, mcpClient, outputPath, summaryResult.Summary)
	if err != nil {
		return day19RunResult{}, err
	}

	return day19RunResult{
		Transport:     transportLabel,
		ServerName:    strings.TrimSpace(initRes.ServerInfo.Name),
		ServerVersion: strings.TrimSpace(initRes.ServerInfo.Version),
		Protocol:      strings.TrimSpace(initRes.ProtocolVersion),
		Tools:         day19CollectToolNames(toolRes.Tools),
		CorpusSource:  corpusSource,
		Search:        searchResult,
		Summary:       summaryResult,
		Save:          saveResult,
	}, nil
}

func newDay19MCPClient(ctx context.Context, transportMode, stdioCommand string, stdioArgs, stdioEnv []string) (*client.Client, string, error) {
	switch transportMode {
	case "inprocess":
		mcpServer := day19mcp.NewServer("day19-inprocess-server", "1.0.0")
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
		mcpClient, err := client.NewStdioMCPClient(cmd, stdioEnv, stdioArgs...)
		if err != nil {
			return nil, "", fmt.Errorf("failed to create stdio MCP client: %w", err)
		}
		return mcpClient, "stdio", nil
	default:
		return nil, "", fmt.Errorf("unsupported transport: %s (allowed: stdio|inprocess)", transportMode)
	}
}

func callDay19Search(ctx context.Context, mcpClient *client.Client, query, corpus string) (day19mcp.SearchResult, error) {
	toolReq := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: day19mcp.ToolSearch,
			Arguments: map[string]any{
				"query":  query,
				"corpus": corpus,
			},
		},
	}
	result, err := mcpClient.CallTool(ctx, toolReq)
	if err != nil {
		return day19mcp.SearchResult{}, fmt.Errorf("search tool call failed: %w", err)
	}
	if result == nil {
		return day19mcp.SearchResult{}, fmt.Errorf("search tool result is nil")
	}
	if result.IsError {
		return day19mcp.SearchResult{}, fmt.Errorf("search tool error: %s", day19ToolResultText(result))
	}
	searchResult, err := parseDay19SearchResult(result)
	if err != nil {
		return day19mcp.SearchResult{}, err
	}
	return searchResult, nil
}

func callDay19Summarize(ctx context.Context, mcpClient *client.Client, text string, maxSentences int) (day19mcp.SummaryResult, error) {
	toolReq := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: day19mcp.ToolSummarize,
			Arguments: map[string]any{
				"text":          text,
				"max_sentences": maxSentences,
			},
		},
	}
	result, err := mcpClient.CallTool(ctx, toolReq)
	if err != nil {
		return day19mcp.SummaryResult{}, fmt.Errorf("summarize tool call failed: %w", err)
	}
	if result == nil {
		return day19mcp.SummaryResult{}, fmt.Errorf("summarize tool result is nil")
	}
	if result.IsError {
		return day19mcp.SummaryResult{}, fmt.Errorf("summarize tool error: %s", day19ToolResultText(result))
	}
	summaryResult, err := parseDay19SummaryResult(result)
	if err != nil {
		return day19mcp.SummaryResult{}, err
	}
	return summaryResult, nil
}

func callDay19Save(ctx context.Context, mcpClient *client.Client, path, content string) (day19mcp.SaveResult, error) {
	toolReq := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: day19mcp.ToolSaveToFile,
			Arguments: map[string]any{
				"path":    path,
				"content": content,
			},
		},
	}
	result, err := mcpClient.CallTool(ctx, toolReq)
	if err != nil {
		return day19mcp.SaveResult{}, fmt.Errorf("save_to_file tool call failed: %w", err)
	}
	if result == nil {
		return day19mcp.SaveResult{}, fmt.Errorf("save_to_file tool result is nil")
	}
	if result.IsError {
		return day19mcp.SaveResult{}, fmt.Errorf("save_to_file tool error: %s", day19ToolResultText(result))
	}
	saveResult, err := parseDay19SaveResult(result)
	if err != nil {
		return day19mcp.SaveResult{}, err
	}
	return saveResult, nil
}

func parseDay19SearchResult(result *mcp.CallToolResult) (day19mcp.SearchResult, error) {
	var out day19mcp.SearchResult
	if parseDay19Structured(result, &out) && out.Query != "" {
		return out, nil
	}
	return day19mcp.SearchResult{}, fmt.Errorf("failed to parse search result")
}

func parseDay19SummaryResult(result *mcp.CallToolResult) (day19mcp.SummaryResult, error) {
	var out day19mcp.SummaryResult
	if parseDay19Structured(result, &out) && out.Summary != "" {
		return out, nil
	}
	return day19mcp.SummaryResult{}, fmt.Errorf("failed to parse summary result")
}

func parseDay19SaveResult(result *mcp.CallToolResult) (day19mcp.SaveResult, error) {
	var out day19mcp.SaveResult
	if parseDay19Structured(result, &out) && out.Path != "" {
		return out, nil
	}
	return day19mcp.SaveResult{}, fmt.Errorf("failed to parse save result")
}

func parseDay19Structured(result *mcp.CallToolResult, out any) bool {
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

func day19CollectToolNames(tools []mcp.Tool) []string {
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

func day19ToolResultText(result *mcp.CallToolResult) string {
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

func printDay19Result(result day19RunResult) {
	fmt.Println("=== Day 19: MCP Tool Composition ===")
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
	fmt.Printf("corpus_source=%s query=%s matches=%d\n", result.CorpusSource, result.Search.Query, result.Search.Count)
	fmt.Printf("summary_sentences=%d saved_path=%s bytes=%d\n",
		result.Summary.SentenceCount,
		result.Save.Path,
		result.Save.Bytes,
	)
}

func writeDay19Report(path string, result day19RunResult) error {
	var b strings.Builder
	b.WriteString("# Day 19 Results: MCP Tool Composition\n\n")
	b.WriteString(fmt.Sprintf("- connection established: `%t`\n", true))
	b.WriteString(fmt.Sprintf("- transport: `%s`\n", result.Transport))
	b.WriteString(fmt.Sprintf("- server: `%s`\n", result.ServerName))
	b.WriteString(fmt.Sprintf("- server version: `%s`\n", result.ServerVersion))
	b.WriteString(fmt.Sprintf("- protocol: `%s`\n", result.Protocol))
	b.WriteString(fmt.Sprintf("- corpus source: `%s`\n", result.CorpusSource))
	b.WriteString(fmt.Sprintf("- query: `%s`\n", result.Search.Query))
	b.WriteString(fmt.Sprintf("- matches: `%d`\n\n", result.Search.Count))

	b.WriteString("## Tools\n")
	for _, name := range result.Tools {
		b.WriteString("- `" + name + "`\n")
	}

	b.WriteString("\n## Pipeline Output\n")
	b.WriteString("### Search Matches\n")
	if len(result.Search.Matches) == 0 {
		b.WriteString("- (no matches)\n")
	} else {
		for _, match := range result.Search.Matches {
			b.WriteString("- " + match + "\n")
		}
	}

	b.WriteString("\n### Summary\n")
	b.WriteString(result.Summary.Summary + "\n")
	b.WriteString(fmt.Sprintf("\n### Saved File\n- path: `%s`\n- bytes: `%d`\n", result.Save.Path, result.Save.Bytes))
	b.WriteString("\nConclusion: pipeline executed automatically (search -> summarize -> save_to_file) with data passed between tools.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay19Usage() {
	fmt.Println("Usage: openrouter-cli day19 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -transport string       MCP transport: stdio|inprocess")
	fmt.Println("  -query string           Search query")
	fmt.Println("  -corpus string          Inline corpus override")
	fmt.Println("  -corpus-file string     Path to corpus file")
	fmt.Println("  -output string          Output file to save summary")
	fmt.Println("  -max-sentences int      Max sentences in summary")
	fmt.Println("  -stdio-command string   Stdio MCP server command")
	fmt.Println("  -stdio-args string      Comma-separated args for stdio command")
	fmt.Println("  -stdio-env string       Comma-separated env vars for stdio command")
	fmt.Println("  -timeout duration       Total timeout (e.g. 20s)")
	fmt.Println("  -report string          Markdown report path")
	fmt.Println("  -help                   Show help")
}
