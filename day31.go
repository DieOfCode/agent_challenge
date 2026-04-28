package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"openrouter-cli/internal/day31mcp"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

const (
	day31DefaultTopK       = 4
	day31DefaultChunkSize  = 900
	day31DefaultOverlap    = 120
	day31DefaultTurnTime   = 20 * time.Second
	day31DefaultReportPath = "DAY31_RESULTS.md"
)

type day31Chunk struct {
	ChunkID string
	Source  string
	Section string
	Text    string
}

type day31RetrievedChunk struct {
	Chunk day31Chunk
	Score float64
}

type day31Connection struct {
	Transport     string
	ServerName    string
	ServerVersion string
	Protocol      string
	Tools         []string
}

type day31HelpResult struct {
	Question  string
	Answer    string
	Branch    day31mcp.GitBranchResult
	Retrieved []day31RetrievedChunk
	Latency   time.Duration
	Usage     usageStats
	Simulated bool
}

type day31Assistant struct {
	mcpClient *client.Client
	apiKey    string
	model     string
	maxTokens int
	temp      *float64
	simulate  bool
	topK      int
	timeout   time.Duration
	chunks    []day31Chunk
}

func runDay31Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day31", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	question := fs.String("question", "", "One-shot /help question")
	interactive := fs.Bool("interactive", true, "Run interactive /help mode")
	workspace := fs.String("workspace", ".", "Project workspace root")
	readmePath := fs.String("readme", "README.md", "README path (relative to workspace)")
	docsDir := fs.String("docs", "docs", "Docs directory (relative to workspace)")
	topK := fs.Int("top-k", day31DefaultTopK, "Top-K chunks for /help retrieval")
	model := fs.String("model", getDefaultModel(), "OpenRouter chat model")
	maxTokens := fs.Int("max-tokens", 320, "Max response tokens")
	temperature := fs.Float64("temperature", 0.2, "Model temperature")
	simulate := fs.Bool("simulate", false, "Run offline deterministic mode (no LLM API calls)")
	transport := fs.String("transport", "inprocess", "MCP transport: inprocess|stdio")
	stdioCommand := fs.String("stdio-command", "go", "Stdio MCP server command")
	stdioArgsCSV := fs.String("stdio-args", "run,./cmd/day31_mcp_server", "Comma-separated args for stdio command")
	stdioEnvCSV := fs.String("stdio-env", "", "Comma-separated env KEY=VALUE for stdio command")
	timeout := fs.Duration("timeout", day31DefaultTurnTime, "Per-operation timeout")
	reportPath := fs.String("report", "", "Optional report path for one-shot mode")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day31 flags: %w", err)
	}
	if *help {
		printDay31Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day31 arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *topK <= 0 {
		return fmt.Errorf("top-k must be positive")
	}
	if *maxTokens <= 0 {
		return fmt.Errorf("max-tokens must be positive")
	}
	if *timeout < time.Second {
		return fmt.Errorf("timeout must be >= 1s")
	}
	if !*interactive && strings.TrimSpace(*question) == "" {
		return fmt.Errorf("empty question: provide -question or run interactive mode")
	}

	workspacePath := strings.TrimSpace(*workspace)
	if workspacePath == "" {
		workspacePath = "."
	}

	chunks, err := loadDay31Corpus(workspacePath, strings.TrimSpace(*readmePath), strings.TrimSpace(*docsDir))
	if err != nil {
		return err
	}
	if len(chunks) == 0 {
		return fmt.Errorf("day31 corpus is empty: check README/docs paths")
	}

	apiKey := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	simulateMode := *simulate
	if !simulateMode && apiKey == "" {
		fmt.Println("warning: OPENROUTER_API_KEY is not set, switching to simulate mode")
		simulateMode = true
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	mcpClient, transportLabel, err := newDay31MCPClient(
		ctx,
		strings.ToLower(strings.TrimSpace(*transport)),
		workspacePath,
		strings.TrimSpace(*stdioCommand),
		parseCSVList(*stdioArgsCSV),
		parseCSVList(*stdioEnvCSV),
	)
	if err != nil {
		return err
	}
	defer mcpClient.Close()

	conn, err := initializeDay31MCP(ctx, mcpClient, transportLabel)
	if err != nil {
		return err
	}
	if !day17ContainsTool(conn.Tools, day31mcp.ToolGitBranch) {
		return fmt.Errorf("required MCP tool %q not found", day31mcp.ToolGitBranch)
	}

	temp := *temperature
	assistant := &day31Assistant{
		mcpClient: mcpClient,
		apiKey:    apiKey,
		model:     strings.TrimSpace(*model),
		maxTokens: *maxTokens,
		temp:      &temp,
		simulate:  simulateMode,
		topK:      *topK,
		timeout:   *timeout,
		chunks:    chunks,
	}

	fmt.Println("=== Day 31: Developer Assistant ===")
	fmt.Printf("connection_established=true transport=%s server=%s version=%s protocol=%s\n",
		conn.Transport,
		day16EmptyFallback(conn.ServerName, "unknown"),
		day16EmptyFallback(conn.ServerVersion, "unknown"),
		day16EmptyFallback(conn.Protocol, "unknown"),
	)
	fmt.Printf("tools_count=%d\n", len(conn.Tools))
	for i, name := range conn.Tools {
		fmt.Printf("%d. %s\n", i+1, name)
	}
	fmt.Printf("rag_chunks=%d readme=%s docs=%s\n", len(chunks), strings.TrimSpace(*readmePath), strings.TrimSpace(*docsDir))
	fmt.Printf("mode=%s\n", map[bool]string{true: "simulate", false: "llm"}[simulateMode])

	if strings.TrimSpace(*question) != "" {
		result, err := assistant.Help(strings.TrimSpace(*question))
		if err != nil {
			return err
		}
		printDay31HelpResult(result)

		report := strings.TrimSpace(*reportPath)
		if report != "" {
			if err := writeDay31Report(report, result, conn); err != nil {
				return err
			}
			fmt.Printf("Отчёт: %s\n", report)
		}
	}

	if *interactive {
		return runDay31Interactive(assistant)
	}
	return nil
}

func newDay31MCPClient(ctx context.Context, transport, workspace, stdioCommand string, stdioArgs, stdioEnv []string) (*client.Client, string, error) {
	switch transport {
	case "inprocess":
		mcpServer := day31mcp.NewServer("day31-inprocess-mcp-server", "1.0.0", workspace)
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
		env = append(env, "DAY31_WORKSPACE="+workspace)
		mcpClient, err := client.NewStdioMCPClient(cmd, env, stdioArgs...)
		if err != nil {
			return nil, "", fmt.Errorf("failed to create stdio MCP client: %w", err)
		}
		return mcpClient, "stdio", nil
	default:
		return nil, "", fmt.Errorf("unsupported transport: %s (allowed: inprocess|stdio)", transport)
	}
}

func initializeDay31MCP(ctx context.Context, mcpClient *client.Client, transport string) (day31Connection, error) {
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{Name: "openrouter-cli-day31", Version: "1.0.0"}
	initReq.Params.Capabilities = mcp.ClientCapabilities{}

	initRes, err := mcpClient.Initialize(ctx, initReq)
	if err != nil {
		return day31Connection{}, fmt.Errorf("failed to initialize MCP connection: %w", err)
	}

	toolRes, err := mcpClient.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return day31Connection{}, fmt.Errorf("failed to list MCP tools: %w", err)
	}
	if toolRes == nil {
		return day31Connection{}, fmt.Errorf("nil tools response")
	}

	return day31Connection{
		Transport:     transport,
		ServerName:    strings.TrimSpace(initRes.ServerInfo.Name),
		ServerVersion: strings.TrimSpace(initRes.ServerInfo.Version),
		Protocol:      strings.TrimSpace(initRes.ProtocolVersion),
		Tools:         day19CollectToolNames(toolRes.Tools),
	}, nil
}

func (a *day31Assistant) Help(question string) (day31HelpResult, error) {
	question = strings.TrimSpace(question)
	if question == "" {
		return day31HelpResult{}, fmt.Errorf("empty /help question")
	}

	ctx, cancel := context.WithTimeout(context.Background(), a.timeout)
	defer cancel()

	branch, err := day31CallGitBranch(ctx, a.mcpClient)
	if err != nil {
		return day31HelpResult{}, err
	}

	retrieved := day31RetrieveChunks(a.chunks, question, a.topK)
	result := day31HelpResult{
		Question:  question,
		Branch:    branch,
		Retrieved: retrieved,
		Simulated: a.simulate,
	}

	if a.simulate {
		result.Answer = day31BuildSimulatedAnswer(question, branch, retrieved)
		return result, nil
	}

	resp, err := day31AskModel(a.apiKey, a.model, a.maxTokens, a.temp, question, branch, retrieved)
	if err != nil {
		return day31HelpResult{}, err
	}
	result.Answer = strings.TrimSpace(resp.Answer)
	result.Usage = resp.Usage
	result.Latency = resp.Latency
	return result, nil
}

func runDay31Interactive(assistant *day31Assistant) error {
	fmt.Println("Interactive mode. Commands: /help <question>, /branch, /sources <question>, /exit")
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1024), 1<<20)
	for {
		fmt.Print("you> ")
		if !scanner.Scan() {
			fmt.Println("")
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		switch {
		case line == "/exit" || line == "exit" || line == "quit":
			return nil
		case line == "/branch":
			ctx, cancel := context.WithTimeout(context.Background(), assistant.timeout)
			branch, err := day31CallGitBranch(ctx, assistant.mcpClient)
			cancel()
			if err != nil {
				fmt.Printf("error: %v\n\n", err)
				continue
			}
			fmt.Printf("branch> %s (dirty=%t head=%s)\n\n", branch.Branch, branch.Dirty, day16EmptyFallback(branch.Head, "-"))
			continue
		case strings.HasPrefix(line, "/sources"):
			query := strings.TrimSpace(strings.TrimPrefix(line, "/sources"))
			if query == "" {
				fmt.Println("usage> /sources <question>")
				fmt.Println("")
				continue
			}
			retrieved := day31RetrieveChunks(assistant.chunks, query, assistant.topK)
			fmt.Printf("sources> %s\n", day31RenderSourcesInline(retrieved))
			fmt.Println("")
			continue
		}

		query := line
		if strings.HasPrefix(line, "/help") {
			query = strings.TrimSpace(strings.TrimPrefix(line, "/help"))
			if query == "" {
				fmt.Println("usage> /help <question>")
				fmt.Println("")
				continue
			}
		}

		result, err := assistant.Help(query)
		if err != nil {
			fmt.Printf("error: %v\n\n", err)
			continue
		}
		printDay31HelpResult(result)
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("failed to read stdin: %w", err)
	}
	return nil
}

func day31CallGitBranch(ctx context.Context, mcpClient *client.Client) (day31mcp.GitBranchResult, error) {
	toolReq := mcp.CallToolRequest{Params: mcp.CallToolParams{Name: day31mcp.ToolGitBranch, Arguments: map[string]any{}}}
	result, err := mcpClient.CallTool(ctx, toolReq)
	if err != nil {
		return day31mcp.GitBranchResult{}, fmt.Errorf("git_branch tool call failed: %w", err)
	}
	if result == nil {
		return day31mcp.GitBranchResult{}, fmt.Errorf("git_branch tool returned nil")
	}
	if result.IsError {
		return day31mcp.GitBranchResult{}, fmt.Errorf("git_branch tool error: %s", day19ToolResultText(result))
	}
	var out day31mcp.GitBranchResult
	if parseDay19Structured(result, &out) && strings.TrimSpace(out.Branch) != "" {
		return out, nil
	}
	return day31mcp.GitBranchResult{}, fmt.Errorf("failed to parse git_branch result")
}

func day31AskModel(apiKey, model string, maxTokens int, temp *float64, question string, branch day31mcp.GitBranchResult, retrieved []day31RetrievedChunk) (openRouterResult, error) {
	systemPrompt := "Ты ассистент разработчика проекта. Отвечай только на основе контекста проекта и git-контекста. Если информации не хватает, честно скажи: 'Не знаю, уточните вопрос'. Обязательно добавь раздел 'Источники:' и перечисли source#chunk_id, которые использовал."
	userPrompt := fmt.Sprintf(
		"Git context:\nbranch=%s\ndirty=%t\nhead=%s\n\nProject docs context:\n%s\n\n/help question:\n%s",
		branch.Branch,
		branch.Dirty,
		day16EmptyFallback(branch.Head, "-"),
		day31RenderContext(retrieved),
		question,
	)
	return callOpenRouterDetailed(
		apiKey,
		model,
		[]message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
		maxTokens,
		temp,
		nil,
		"openrouter-cli-day31-help",
	)
}

func day31BuildSimulatedAnswer(question string, branch day31mcp.GitBranchResult, retrieved []day31RetrievedChunk) string {
	best := ""
	if len(retrieved) > 0 {
		best = strings.TrimSpace(retrieved[0].Chunk.Text)
		if len(best) > 220 {
			best = best[:220]
		}
	}
	if best == "" {
		best = "В контексте не найдено подходящих фрагментов."
	}
	return strings.TrimSpace(fmt.Sprintf(
		"[simulate] По вопросу: %s\nТекущая ветка: %s (dirty=%t).\nКлючевой фрагмент: %s\nИсточники: %s",
		question,
		branch.Branch,
		branch.Dirty,
		best,
		day31RenderSourcesInline(retrieved),
	))
}

func loadDay31Corpus(workspace, readmePath, docsDir string) ([]day31Chunk, error) {
	workspaceAbs, err := filepath.Abs(strings.TrimSpace(workspace))
	if err != nil {
		return nil, fmt.Errorf("failed to resolve workspace: %w", err)
	}

	readmeAbs := day31ResolvePath(workspaceAbs, readmePath)
	readmeText, err := os.ReadFile(readmeAbs)
	if err != nil {
		return nil, fmt.Errorf("failed to read README for day31 RAG: %w", err)
	}

	docsAbs := day31ResolvePath(workspaceAbs, docsDir)
	stat, err := os.Stat(docsAbs)
	if err != nil {
		return nil, fmt.Errorf("failed to read docs dir for day31 RAG: %w", err)
	}
	if !stat.IsDir() {
		return nil, fmt.Errorf("docs path is not a directory: %s", docsDir)
	}

	chunks := day31ChunkMarkdownFile(day31RelPath(workspaceAbs, readmeAbs), string(readmeText))
	walkErr := filepath.WalkDir(docsAbs, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(d.Name()))
		switch ext {
		case ".md", ".txt", ".json", ".yaml", ".yml":
		default:
			return nil
		}
		content, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		rel := day31RelPath(workspaceAbs, path)
		chunks = append(chunks, day31ChunkMarkdownFile(rel, string(content))...)
		return nil
	})
	if walkErr != nil {
		return nil, fmt.Errorf("failed to walk docs: %w", walkErr)
	}
	return chunks, nil
}

func day31ResolvePath(workspace, path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		path = "."
	}
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	return filepath.Clean(filepath.Join(workspace, path))
}

func day31RelPath(workspace, path string) string {
	rel, err := filepath.Rel(workspace, path)
	if err != nil {
		return filepath.ToSlash(path)
	}
	rel = filepath.ToSlash(rel)
	if rel == "" {
		return "."
	}
	return rel
}

func day31ChunkMarkdownFile(source, text string) []day31Chunk {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	lines := strings.Split(text, "\n")
	section := "general"
	var sectionBuf strings.Builder
	chunks := make([]day31Chunk, 0)
	chunkIdx := 0

	flush := func() {
		body := strings.TrimSpace(sectionBuf.String())
		sectionBuf.Reset()
		if body == "" {
			return
		}
		for _, piece := range day31SplitText(body, day31DefaultChunkSize, day31DefaultOverlap) {
			chunkID := fmt.Sprintf("%s#%03d", source, chunkIdx)
			chunks = append(chunks, day31Chunk{
				ChunkID: chunkID,
				Source:  source,
				Section: section,
				Text:    piece,
			})
			chunkIdx++
		}
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") {
			flush()
			heading := strings.TrimSpace(strings.TrimLeft(trimmed, "#"))
			if heading == "" {
				heading = "section"
			}
			section = heading
			continue
		}
		sectionBuf.WriteString(line)
		sectionBuf.WriteByte('\n')
	}
	flush()

	if len(chunks) == 0 {
		chunks = append(chunks, day31Chunk{
			ChunkID: source + "#000",
			Source:  source,
			Section: "full",
			Text:    text,
		})
	}
	return chunks
}

func day31SplitText(text string, size, overlap int) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	if size <= 0 {
		size = day31DefaultChunkSize
	}
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= size {
		overlap = size / 4
	}
	runes := []rune(text)
	if len(runes) <= size {
		return []string{text}
	}

	out := make([]string, 0)
	start := 0
	for start < len(runes) {
		end := start + size
		if end > len(runes) {
			end = len(runes)
		}
		piece := strings.TrimSpace(string(runes[start:end]))
		if piece != "" {
			out = append(out, piece)
		}
		if end == len(runes) {
			break
		}
		next := end - overlap
		if next <= start {
			next = end
		}
		start = next
	}
	return out
}

func day31RetrieveChunks(chunks []day31Chunk, question string, topK int) []day31RetrievedChunk {
	if len(chunks) == 0 || topK <= 0 {
		return nil
	}
	queryTokens := day31Tokens(question)
	if len(queryTokens) == 0 {
		queryTokens = []string{strings.ToLower(strings.TrimSpace(question))}
	}

	scored := make([]day31RetrievedChunk, 0, len(chunks))
	for _, chunk := range chunks {
		score := day31ChunkScore(queryTokens, chunk)
		if score <= 0 {
			continue
		}
		scored = append(scored, day31RetrievedChunk{Chunk: chunk, Score: score})
	}

	if len(scored) == 0 {
		limit := topK
		if limit > len(chunks) {
			limit = len(chunks)
		}
		fallback := make([]day31RetrievedChunk, 0, limit)
		for i := 0; i < limit; i++ {
			fallback = append(fallback, day31RetrievedChunk{Chunk: chunks[i], Score: 0})
		}
		return fallback
	}

	sort.Slice(scored, func(i, j int) bool {
		if scored[i].Score == scored[j].Score {
			return scored[i].Chunk.ChunkID < scored[j].Chunk.ChunkID
		}
		return scored[i].Score > scored[j].Score
	})

	if topK > len(scored) {
		topK = len(scored)
	}
	return scored[:topK]
}

func day31ChunkScore(queryTokens []string, chunk day31Chunk) float64 {
	if len(queryTokens) == 0 {
		return 0
	}
	docSet := tokenSet(strings.ToLower(chunk.Text + " " + chunk.Section + " " + chunk.Source))
	if len(docSet) == 0 {
		return 0
	}

	querySet := make(map[string]struct{}, len(queryTokens))
	for _, token := range queryTokens {
		if token == "" {
			continue
		}
		querySet[token] = struct{}{}
	}
	if len(querySet) == 0 {
		return 0
	}

	match := 0
	for token := range querySet {
		if _, ok := docSet[token]; ok {
			match++
		}
	}
	if match == 0 {
		return 0
	}

	coverage := float64(match) / float64(len(querySet))
	density := float64(match) / float64(len(docSet))
	return coverage*0.85 + density*0.15
}

func day31Tokens(text string) []string {
	parts := wordTokenRE.FindAllString(strings.ToLower(text), -1)
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func day31RenderContext(retrieved []day31RetrievedChunk) string {
	if len(retrieved) == 0 {
		return "(no chunks found)"
	}
	var b strings.Builder
	for i, item := range retrieved {
		b.WriteString(fmt.Sprintf("[%d] source=%s section=%s chunk_id=%s score=%.4f\n", i+1, item.Chunk.Source, item.Chunk.Section, item.Chunk.ChunkID, item.Score))
		b.WriteString(item.Chunk.Text)
		b.WriteString("\n---\n")
	}
	return strings.TrimSpace(b.String())
}

func day31RenderSourcesInline(retrieved []day31RetrievedChunk) string {
	if len(retrieved) == 0 {
		return "(no sources)"
	}
	parts := make([]string, 0, len(retrieved))
	for _, item := range retrieved {
		parts = append(parts, fmt.Sprintf("%s (%s, %.3f)", item.Chunk.Source, item.Chunk.ChunkID, item.Score))
	}
	return strings.Join(parts, "; ")
}

func printDay31HelpResult(result day31HelpResult) {
	fmt.Printf("\n/help> %s\n", result.Question)
	fmt.Printf("assistant> %s\n", strings.TrimSpace(result.Answer))
	fmt.Printf("branch> %s (dirty=%t head=%s)\n", result.Branch.Branch, result.Branch.Dirty, day16EmptyFallback(result.Branch.Head, "-"))
	fmt.Printf("sources> %s\n", day31RenderSourcesInline(result.Retrieved))
	if !result.Simulated {
		fmt.Printf("usage> prompt=%d completion=%d total=%d latency=%s\n", result.Usage.PromptTokens, result.Usage.CompletionTokens, result.Usage.TotalTokens, result.Latency.Round(time.Millisecond))
	}
	fmt.Println("")
}

func writeDay31Report(path string, result day31HelpResult, conn day31Connection) error {
	if strings.TrimSpace(path) == "" {
		path = day31DefaultReportPath
	}
	var b strings.Builder
	b.WriteString("# Day 31 Results: Developer Assistant\n\n")
	b.WriteString(fmt.Sprintf("- transport: `%s`\n", conn.Transport))
	b.WriteString(fmt.Sprintf("- server: `%s`\n", conn.ServerName))
	b.WriteString(fmt.Sprintf("- server version: `%s`\n", conn.ServerVersion))
	b.WriteString(fmt.Sprintf("- protocol: `%s`\n", conn.Protocol))
	b.WriteString(fmt.Sprintf("- branch: `%s`\n", result.Branch.Branch))
	b.WriteString(fmt.Sprintf("- dirty: `%t`\n", result.Branch.Dirty))
	b.WriteString(fmt.Sprintf("- mode: `%s`\n\n", map[bool]string{true: "simulate", false: "llm"}[result.Simulated]))

	b.WriteString("## Question\n")
	b.WriteString(result.Question + "\n\n")

	b.WriteString("## Answer\n")
	b.WriteString(strings.TrimSpace(result.Answer) + "\n\n")

	b.WriteString("## Retrieved Sources\n")
	for _, item := range result.Retrieved {
		b.WriteString(fmt.Sprintf("- `%s` (%s, score=%.4f)\n", item.Chunk.Source, item.Chunk.ChunkID, item.Score))
	}

	if !result.Simulated {
		b.WriteString("\n## Usage\n")
		b.WriteString(fmt.Sprintf("- prompt tokens: `%d`\n", result.Usage.PromptTokens))
		b.WriteString(fmt.Sprintf("- completion tokens: `%d`\n", result.Usage.CompletionTokens))
		b.WriteString(fmt.Sprintf("- total tokens: `%d`\n", result.Usage.TotalTokens))
		b.WriteString(fmt.Sprintf("- latency: `%s`\n", result.Latency.Round(time.Millisecond)))
	}

	b.WriteString("\nConclusion: /help uses README+docs retrieval and MCP git branch context to answer project questions.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay31Usage() {
	fmt.Println("Usage: openrouter-cli day31 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -question string       One-shot /help question")
	fmt.Println("  -interactive           Run interactive /help mode")
	fmt.Println("  -workspace string      Project workspace root")
	fmt.Println("  -readme string         README path")
	fmt.Println("  -docs string           Docs directory path")
	fmt.Println("  -top-k int             Top-K chunks for retrieval")
	fmt.Println("  -model string          OpenRouter chat model")
	fmt.Println("  -max-tokens int        Max response tokens")
	fmt.Println("  -temperature float     Model temperature")
	fmt.Println("  -simulate              Run offline deterministic mode")
	fmt.Println("  -transport string      MCP transport: inprocess|stdio")
	fmt.Println("  -stdio-command string  Stdio MCP server command")
	fmt.Println("  -stdio-args string     Comma-separated args for stdio command")
	fmt.Println("  -stdio-env string      Comma-separated env vars for stdio command")
	fmt.Println("  -timeout duration      Per-operation timeout")
	fmt.Println("  -report string         Optional report path for one-shot mode")
	fmt.Println("  -help                  Show help")
}
