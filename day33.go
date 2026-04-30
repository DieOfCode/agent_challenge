package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"openrouter-cli/internal/day33mcp"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

const (
	day33DefaultTopK      = 6
	day33DefaultMaxTokens = 420
)

type day33Connection struct {
	Transport     string
	ServerName    string
	ServerVersion string
	Protocol      string
	Tools         []string
}

type day33SupportContext struct {
	User           *day33mcp.UserProfile
	Ticket         *day33mcp.SupportTicket
	RelatedTickets []day33mcp.SupportTicket
}

type day33AnswerResult struct {
	Question  string
	Context   day33SupportContext
	Retrieved []day31RetrievedChunk
	Answer    string
	Simulated bool
	Latency   time.Duration
	Usage     usageStats
}

type day33Assistant struct {
	mcpClient *client.Client
	apiKey    string
	model     string
	maxTokens int
	temp      *float64
	simulate  bool
	topK      int
	timeout   time.Duration
	ragChunks []day31Chunk
}

func runDay33Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day33", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	question := fs.String("question", "", "Question to support assistant")
	interactive := fs.Bool("interactive", true, "Run interactive support mode")
	ticketID := fs.Int("ticket-id", 5001, "Ticket ID (0 to skip)")
	userID := fs.Int("user-id", 0, "User ID override (0 = auto from ticket)")
	workspace := fs.String("workspace", ".", "Project workspace root")
	readmePath := fs.String("readme", "README.md", "README path for RAG")
	docsDir := fs.String("docs", "docs", "Docs directory for RAG")
	usersFile := fs.String("users-file", "support_data/users.json", "Users JSON path")
	ticketsFile := fs.String("tickets-file", "support_data/tickets.json", "Tickets JSON path")
	transport := fs.String("transport", "inprocess", "MCP transport: inprocess|stdio")
	stdioCommand := fs.String("stdio-command", "go", "Stdio MCP server command")
	stdioArgsCSV := fs.String("stdio-args", "run,./cmd/day33_mcp_server", "Comma-separated args for stdio command")
	stdioEnvCSV := fs.String("stdio-env", "", "Comma-separated env values KEY=VALUE")
	topK := fs.Int("top-k", day33DefaultTopK, "Top-K RAG chunks")
	model := fs.String("model", getDefaultModel(), "OpenRouter chat model")
	maxTokens := fs.Int("max-tokens", day33DefaultMaxTokens, "Max response tokens")
	temperature := fs.Float64("temperature", 0.2, "Model temperature")
	simulate := fs.Bool("simulate", false, "Offline deterministic mode")
	timeout := fs.Duration("timeout", 25*time.Second, "Timeout for MCP/LLM operations")
	reportPath := fs.String("report", "", "Optional report path for one-shot mode")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day33 flags: %w", err)
	}
	if *help {
		printDay33Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day33 arguments: %s", strings.Join(fs.Args(), " "))
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
		return fmt.Errorf("empty question: set -question or enable -interactive")
	}

	workspaceAbs, err := filepath.Abs(strings.TrimSpace(*workspace))
	if err != nil {
		return fmt.Errorf("failed to resolve workspace: %w", err)
	}

	ragChunks, err := loadDay31Corpus(workspaceAbs, strings.TrimSpace(*readmePath), strings.TrimSpace(*docsDir))
	if err != nil {
		return err
	}
	if len(ragChunks) == 0 {
		return fmt.Errorf("RAG context is empty (README/docs)")
	}

	apiKey := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	simulateMode := *simulate
	if !simulateMode && apiKey == "" {
		simulateMode = true
		fmt.Println("warning: OPENROUTER_API_KEY is not set, switched to simulate mode")
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	mcpClient, transportLabel, err := newDay33MCPClient(
		ctx,
		strings.ToLower(strings.TrimSpace(*transport)),
		workspaceAbs,
		strings.TrimSpace(*usersFile),
		strings.TrimSpace(*ticketsFile),
		strings.TrimSpace(*stdioCommand),
		parseCSVList(*stdioArgsCSV),
		parseCSVList(*stdioEnvCSV),
	)
	if err != nil {
		return err
	}
	defer mcpClient.Close()

	conn, err := initializeDay33MCP(ctx, mcpClient, transportLabel)
	if err != nil {
		return err
	}
	if !day17ContainsTool(conn.Tools, day33mcp.ToolGetUserProfile) || !day17ContainsTool(conn.Tools, day33mcp.ToolGetTicket) {
		return fmt.Errorf("required MCP tools are not available: need %q and %q", day33mcp.ToolGetUserProfile, day33mcp.ToolGetTicket)
	}

	temp := *temperature
	assistant := &day33Assistant{
		mcpClient: mcpClient,
		apiKey:    apiKey,
		model:     strings.TrimSpace(*model),
		maxTokens: *maxTokens,
		temp:      &temp,
		simulate:  simulateMode,
		topK:      *topK,
		timeout:   *timeout,
		ragChunks: ragChunks,
	}

	fmt.Println("=== Day 33: Support Assistant ===")
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
	fmt.Printf("rag_chunks=%d mode=%s\n", len(ragChunks), map[bool]string{true: "simulate", false: "llm"}[simulateMode])

	if strings.TrimSpace(*question) != "" {
		result, err := assistant.Answer(strings.TrimSpace(*question), *ticketID, *userID)
		if err != nil {
			return err
		}
		printDay33AnswerResult(result)

		report := strings.TrimSpace(*reportPath)
		if report != "" {
			if err := writeDay33Report(report, result, conn); err != nil {
				return err
			}
			fmt.Printf("Отчёт: %s\n", report)
		}
	}

	if *interactive {
		return runDay33Interactive(assistant, *ticketID, *userID)
	}
	return nil
}

func newDay33MCPClient(ctx context.Context, transport, workspace, usersFile, ticketsFile, stdioCommand string, stdioArgs, stdioEnv []string) (*client.Client, string, error) {
	switch transport {
	case "inprocess":
		mcpServer := day33mcp.NewServer("day33-inprocess-support-mcp", "1.0.0", usersFile, ticketsFile)
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
		env = append(env, "DAY33_WORKSPACE="+workspace)
		env = append(env, "DAY33_USERS_FILE="+usersFile)
		env = append(env, "DAY33_TICKETS_FILE="+ticketsFile)
		mcpClient, err := client.NewStdioMCPClient(cmd, env, stdioArgs...)
		if err != nil {
			return nil, "", fmt.Errorf("failed to create stdio MCP client: %w", err)
		}
		return mcpClient, "stdio", nil
	default:
		return nil, "", fmt.Errorf("unsupported transport: %s (allowed: inprocess|stdio)", transport)
	}
}

func initializeDay33MCP(ctx context.Context, mcpClient *client.Client, transport string) (day33Connection, error) {
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{Name: "openrouter-cli-day33", Version: "1.0.0"}
	initReq.Params.Capabilities = mcp.ClientCapabilities{}

	initRes, err := mcpClient.Initialize(ctx, initReq)
	if err != nil {
		return day33Connection{}, fmt.Errorf("failed to initialize MCP connection: %w", err)
	}
	toolRes, err := mcpClient.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return day33Connection{}, fmt.Errorf("failed to list MCP tools: %w", err)
	}
	if toolRes == nil {
		return day33Connection{}, fmt.Errorf("nil tools response")
	}
	return day33Connection{
		Transport:     transport,
		ServerName:    strings.TrimSpace(initRes.ServerInfo.Name),
		ServerVersion: strings.TrimSpace(initRes.ServerInfo.Version),
		Protocol:      strings.TrimSpace(initRes.ProtocolVersion),
		Tools:         day19CollectToolNames(toolRes.Tools),
	}, nil
}

func (a *day33Assistant) Answer(question string, ticketID, userID int) (day33AnswerResult, error) {
	question = strings.TrimSpace(question)
	if question == "" {
		return day33AnswerResult{}, fmt.Errorf("empty question")
	}
	ctx, cancel := context.WithTimeout(context.Background(), a.timeout)
	defer cancel()

	supportCtx, err := a.loadSupportContext(ctx, ticketID, userID)
	if err != nil {
		return day33AnswerResult{}, err
	}

	query := day33BuildRAGQuery(question, supportCtx)
	retrieved := day31RetrieveChunks(a.ragChunks, query, a.topK)
	result := day33AnswerResult{
		Question:  question,
		Context:   supportCtx,
		Retrieved: retrieved,
		Simulated: a.simulate,
	}

	if a.simulate {
		result.Answer = day33SimulateAnswer(question, supportCtx, retrieved)
		return result, nil
	}

	resp, err := day33AskModel(a.apiKey, a.model, a.maxTokens, a.temp, question, supportCtx, retrieved)
	if err != nil {
		return day33AnswerResult{}, err
	}
	result.Answer = strings.TrimSpace(resp.Answer)
	result.Usage = resp.Usage
	result.Latency = resp.Latency
	return result, nil
}

func (a *day33Assistant) loadSupportContext(ctx context.Context, ticketID, userID int) (day33SupportContext, error) {
	out := day33SupportContext{}

	if ticketID > 0 {
		ticket, err := day33CallGetTicket(ctx, a.mcpClient, ticketID)
		if err != nil {
			return out, err
		}
		out.Ticket = &ticket
		if userID == 0 {
			userID = ticket.UserID
		}
	}

	if userID > 0 {
		user, err := day33CallGetUser(ctx, a.mcpClient, userID)
		if err != nil {
			return out, err
		}
		out.User = &user
		tickets, err := day33CallListUserTickets(ctx, a.mcpClient, userID, 4)
		if err == nil {
			out.RelatedTickets = tickets
		}
	}
	return out, nil
}

func day33CallGetUser(ctx context.Context, mcpClient *client.Client, userID int) (day33mcp.UserProfile, error) {
	request := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: day33mcp.ToolGetUserProfile,
			Arguments: map[string]any{
				"user_id": userID,
			},
		},
	}
	res, err := mcpClient.CallTool(ctx, request)
	if err != nil {
		return day33mcp.UserProfile{}, fmt.Errorf("get_user_profile failed: %w", err)
	}
	if res == nil {
		return day33mcp.UserProfile{}, fmt.Errorf("get_user_profile returned nil")
	}
	if res.IsError {
		return day33mcp.UserProfile{}, fmt.Errorf("get_user_profile error: %s", day19ToolResultText(res))
	}
	user, err := day33mcp.ParseUserFromCallToolResult(res)
	if err != nil {
		return day33mcp.UserProfile{}, err
	}
	return user, nil
}

func day33CallGetTicket(ctx context.Context, mcpClient *client.Client, ticketID int) (day33mcp.SupportTicket, error) {
	request := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: day33mcp.ToolGetTicket,
			Arguments: map[string]any{
				"ticket_id": ticketID,
			},
		},
	}
	res, err := mcpClient.CallTool(ctx, request)
	if err != nil {
		return day33mcp.SupportTicket{}, fmt.Errorf("get_ticket failed: %w", err)
	}
	if res == nil {
		return day33mcp.SupportTicket{}, fmt.Errorf("get_ticket returned nil")
	}
	if res.IsError {
		return day33mcp.SupportTicket{}, fmt.Errorf("get_ticket error: %s", day19ToolResultText(res))
	}
	ticket, err := day33mcp.ParseTicketFromCallToolResult(res)
	if err != nil {
		return day33mcp.SupportTicket{}, err
	}
	return ticket, nil
}

func day33CallListUserTickets(ctx context.Context, mcpClient *client.Client, userID, limit int) ([]day33mcp.SupportTicket, error) {
	request := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: day33mcp.ToolListUserTickets,
			Arguments: map[string]any{
				"user_id": userID,
				"limit":   limit,
			},
		},
	}
	res, err := mcpClient.CallTool(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("list_user_tickets failed: %w", err)
	}
	if res == nil || res.IsError {
		if res == nil {
			return nil, fmt.Errorf("list_user_tickets returned nil")
		}
		return nil, fmt.Errorf("list_user_tickets error: %s", day19ToolResultText(res))
	}
	out, err := day33mcp.ParseTicketListFromCallToolResult(res)
	if err != nil {
		return nil, err
	}
	return out.Tickets, nil
}

func day33BuildRAGQuery(question string, ctx day33SupportContext) string {
	var b strings.Builder
	b.WriteString(question)
	if ctx.Ticket != nil {
		t := ctx.Ticket
		b.WriteString(" ")
		b.WriteString(t.Subject)
		b.WriteString(" ")
		b.WriteString(t.Category)
		b.WriteString(" ")
		b.WriteString(t.LastError)
		b.WriteString(" ")
		b.WriteString(t.LastErrorCode)
	}
	if ctx.User != nil {
		u := ctx.User
		b.WriteString(" ")
		b.WriteString(u.Plan)
		b.WriteString(" ")
		b.WriteString(u.AuthProvider)
	}
	return b.String()
}

func day33AskModel(apiKey, model string, maxTokens int, temp *float64, question string, ctx day33SupportContext, retrieved []day31RetrievedChunk) (openRouterResult, error) {
	systemPrompt := "Ты AI-ассистент поддержки пользователей. Отвечай коротко и практично. Используй только контекст тикета/пользователя и RAG-контекст. Формат ответа: `Краткий ответ`, `Почему`, `Шаги для пользователя`, `Шаги для поддержки`, `Источники`."
	userPrompt := day33BuildLLMPrompt(question, ctx, retrieved)
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
		"openrouter-cli-day33-support",
	)
}

func day33BuildLLMPrompt(question string, ctx day33SupportContext, retrieved []day31RetrievedChunk) string {
	var b strings.Builder
	b.WriteString("Вопрос пользователя:\n")
	b.WriteString(question)
	b.WriteString("\n\n")

	b.WriteString("Контекст пользователя:\n")
	if ctx.User == nil {
		b.WriteString("(нет)\n")
	} else {
		u := ctx.User
		b.WriteString(fmt.Sprintf("id=%d name=%s plan=%s locale=%s auth_provider=%s status=%s tags=%s\n", u.ID, u.Name, u.Plan, u.Locale, u.AuthProvider, u.Status, strings.Join(u.Tags, ",")))
	}
	b.WriteString("\n")

	b.WriteString("Контекст тикета:\n")
	if ctx.Ticket == nil {
		b.WriteString("(нет)\n")
	} else {
		t := ctx.Ticket
		b.WriteString(fmt.Sprintf("id=%d subject=%s status=%s priority=%s category=%s attempts=%d last_error=%s last_error_code=%s feature=%s labels=%s\nsummary=%s\n",
			t.ID, t.Subject, t.Status, t.Priority, t.Category, t.Attempts, t.LastError, t.LastErrorCode, t.AffectedFeature, strings.Join(t.Labels, ","), t.Summary))
	}
	b.WriteString("\n")

	if len(ctx.RelatedTickets) > 0 {
		b.WriteString("Похожие тикеты пользователя:\n")
		for _, ticket := range ctx.RelatedTickets {
			b.WriteString(fmt.Sprintf("- #%d [%s/%s] %s (error=%s)\n", ticket.ID, ticket.Priority, ticket.Status, ticket.Subject, ticket.LastErrorCode))
		}
		b.WriteString("\n")
	}

	b.WriteString("RAG контекст (FAQ + docs):\n")
	b.WriteString(day31RenderContext(retrieved))
	return b.String()
}

func day33SimulateAnswer(question string, ctx day33SupportContext, retrieved []day31RetrievedChunk) string {
	var short string
	var why string
	userSteps := []string{}
	supportSteps := []string{}

	if ctx.Ticket != nil {
		t := ctx.Ticket
		subjectLower := strings.ToLower(t.Subject + " " + t.Category + " " + t.LastError + " " + t.LastErrorCode)
		switch {
		case strings.Contains(subjectLower, "auth") || strings.Contains(subjectLower, "автор") || strings.Contains(subjectLower, "sso"):
			short = "Проблема похожа на сбой в auth-потоке (токен/сессия/2FA/SSO)."
			why = "В тикете есть auth-признаки: last_error=" + t.LastError + ", code=" + t.LastErrorCode + "."
			userSteps = append(userSteps,
				"Попросить пользователя выйти из всех сессий и войти повторно.",
				"Проверить корректность времени на устройстве и актуальность метода входа.",
			)
			supportSteps = append(supportSteps,
				"Проверить auth-логи по request_id/времени попытки.",
				"Проверить лимиты попыток, статус 2FA и состояние SSO-конфига.",
			)
		default:
			short = "Похоже на проблему в клиентской части или конфигурации окружения."
			why = "Категория тикета: " + t.Category + "."
			userSteps = append(userSteps, "Сделать hard reload и повторить сценарий.")
			supportSteps = append(supportSteps, "Проверить логи фронтенда/бэкенда по времени инцидента.")
		}
	} else {
		short = "Нужно уточнить ID тикета или контекст пользователя, чтобы дать точный ответ."
		why = "Без контекста тикета рекомендация будет слишком общей."
		userSteps = append(userSteps, "Сообщить точный текст ошибки и время попытки.")
		supportSteps = append(supportSteps, "Запросить ticket_id/user_id и повторно провести диагностику.")
	}

	if ctx.User != nil {
		supportSteps = append(supportSteps, fmt.Sprintf("Учесть профиль пользователя: plan=%s, auth_provider=%s.", ctx.User.Plan, ctx.User.AuthProvider))
	}

	var b strings.Builder
	b.WriteString("Краткий ответ:\n")
	b.WriteString(short + "\n\n")
	b.WriteString("Почему:\n")
	b.WriteString(why + "\n\n")
	b.WriteString("Шаги для пользователя:\n")
	for _, step := range day33UniqueStrings(userSteps) {
		b.WriteString("- " + step + "\n")
	}
	b.WriteString("\nШаги для поддержки:\n")
	for _, step := range day33UniqueStrings(supportSteps) {
		b.WriteString("- " + step + "\n")
	}
	b.WriteString("\nИсточники:\n")
	b.WriteString("- " + day31RenderSourcesInline(retrieved) + "\n")
	b.WriteString("\n[simulate] question=" + question)
	return strings.TrimSpace(b.String())
}

func runDay33Interactive(assistant *day33Assistant, startTicketID, startUserID int) error {
	fmt.Println("Interactive support mode. Commands: /ticket <id>, /user <id>, /context, /exit")
	currentTicketID := startTicketID
	currentUserID := startUserID
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1024), 1<<20)
	for {
		fmt.Print("support> ")
		if !scanner.Scan() {
			fmt.Println("")
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if line == "/exit" || line == "exit" || line == "quit" {
			return nil
		}
		if strings.HasPrefix(line, "/ticket") {
			parts := strings.Fields(line)
			if len(parts) != 2 {
				fmt.Println("usage> /ticket <id>")
				continue
			}
			id := parseIntOrZero(parts[1])
			if id <= 0 {
				fmt.Println("ticket id must be positive")
				continue
			}
			currentTicketID = id
			fmt.Printf("context> ticket_id=%d\n", currentTicketID)
			continue
		}
		if strings.HasPrefix(line, "/user") {
			parts := strings.Fields(line)
			if len(parts) != 2 {
				fmt.Println("usage> /user <id>")
				continue
			}
			id := parseIntOrZero(parts[1])
			if id <= 0 {
				fmt.Println("user id must be positive")
				continue
			}
			currentUserID = id
			fmt.Printf("context> user_id=%d\n", currentUserID)
			continue
		}
		if line == "/context" {
			fmt.Printf("context> ticket_id=%d user_id=%d\n", currentTicketID, currentUserID)
			continue
		}

		result, err := assistant.Answer(line, currentTicketID, currentUserID)
		if err != nil {
			fmt.Printf("error: %v\n", err)
			continue
		}
		printDay33AnswerResult(result)
		if result.Context.Ticket != nil {
			currentTicketID = result.Context.Ticket.ID
		}
		if result.Context.User != nil {
			currentUserID = result.Context.User.ID
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("failed to read stdin: %w", err)
	}
	return nil
}

func parseIntOrZero(text string) int {
	text = strings.TrimSpace(text)
	if text == "" {
		return 0
	}
	value := 0
	for _, r := range text {
		if r < '0' || r > '9' {
			return 0
		}
		value = value*10 + int(r-'0')
	}
	return value
}

func printDay33AnswerResult(result day33AnswerResult) {
	fmt.Printf("\nquestion> %s\n", result.Question)
	if result.Context.Ticket != nil {
		t := result.Context.Ticket
		fmt.Printf("ticket> #%d [%s/%s] %s (error=%s)\n", t.ID, t.Priority, t.Status, t.Subject, t.LastErrorCode)
	}
	if result.Context.User != nil {
		u := result.Context.User
		fmt.Printf("user> #%d %s plan=%s auth=%s\n", u.ID, u.Email, u.Plan, u.AuthProvider)
	}
	fmt.Printf("sources> %s\n", day31RenderSourcesInline(result.Retrieved))
	fmt.Printf("answer>\n%s\n", strings.TrimSpace(result.Answer))
	if !result.Simulated {
		fmt.Printf("usage> prompt=%d completion=%d total=%d latency=%s\n", result.Usage.PromptTokens, result.Usage.CompletionTokens, result.Usage.TotalTokens, result.Latency.Round(time.Millisecond))
	}
	fmt.Println("")
}

func writeDay33Report(path string, result day33AnswerResult, conn day33Connection) error {
	if strings.TrimSpace(path) == "" {
		path = "DAY33_RESULTS.md"
	}
	var b strings.Builder
	b.WriteString("# Day 33 Results: Support Assistant\n\n")
	b.WriteString(fmt.Sprintf("- transport: `%s`\n", conn.Transport))
	b.WriteString(fmt.Sprintf("- server: `%s`\n", conn.ServerName))
	b.WriteString(fmt.Sprintf("- server version: `%s`\n", conn.ServerVersion))
	b.WriteString(fmt.Sprintf("- protocol: `%s`\n", conn.Protocol))
	if result.Context.Ticket != nil {
		b.WriteString(fmt.Sprintf("- ticket: `#%d`\n", result.Context.Ticket.ID))
	}
	if result.Context.User != nil {
		b.WriteString(fmt.Sprintf("- user: `#%d`\n", result.Context.User.ID))
	}
	b.WriteString(fmt.Sprintf("- mode: `%s`\n\n", map[bool]string{true: "simulate", false: "llm"}[result.Simulated]))

	b.WriteString("## Question\n")
	b.WriteString(result.Question + "\n\n")
	b.WriteString("## Answer\n")
	b.WriteString(strings.TrimSpace(result.Answer) + "\n\n")

	b.WriteString("## Sources\n")
	if len(result.Retrieved) == 0 {
		b.WriteString("- (none)\n")
	} else {
		sources := append([]day31RetrievedChunk(nil), result.Retrieved...)
		sort.Slice(sources, func(i, j int) bool { return sources[i].Score > sources[j].Score })
		for _, item := range sources {
			b.WriteString(fmt.Sprintf("- `%s` (%s, score=%.4f)\n", item.Chunk.Source, item.Chunk.ChunkID, item.Score))
		}
	}

	if !result.Simulated {
		b.WriteString("\n## Usage\n")
		b.WriteString(fmt.Sprintf("- prompt tokens: `%d`\n", result.Usage.PromptTokens))
		b.WriteString(fmt.Sprintf("- completion tokens: `%d`\n", result.Usage.CompletionTokens))
		b.WriteString(fmt.Sprintf("- total tokens: `%d`\n", result.Usage.TotalTokens))
		b.WriteString(fmt.Sprintf("- latency: `%s`\n", result.Latency.Round(time.Millisecond)))
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func day33UniqueStrings(items []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(items))
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		if _, ok := seen[item]; ok {
			continue
		}
		seen[item] = struct{}{}
		out = append(out, item)
	}
	return out
}

func printDay33Usage() {
	fmt.Println("Usage: openrouter-cli day33 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -question string       User support question")
	fmt.Println("  -interactive           Run interactive support mode")
	fmt.Println("  -ticket-id int         Ticket ID (0 to skip)")
	fmt.Println("  -user-id int           User ID override (0 = infer from ticket)")
	fmt.Println("  -workspace string      Project workspace root")
	fmt.Println("  -readme string         README path for RAG")
	fmt.Println("  -docs string           Docs dir for RAG")
	fmt.Println("  -users-file string     Users JSON path")
	fmt.Println("  -tickets-file string   Tickets JSON path")
	fmt.Println("  -transport string      MCP transport: inprocess|stdio")
	fmt.Println("  -stdio-command string  Stdio MCP server command")
	fmt.Println("  -stdio-args string     Comma-separated args for stdio command")
	fmt.Println("  -stdio-env string      Comma-separated env values KEY=VALUE")
	fmt.Println("  -top-k int             Top-K RAG chunks")
	fmt.Println("  -model string          OpenRouter chat model")
	fmt.Println("  -max-tokens int        Max response tokens")
	fmt.Println("  -temperature float     Model temperature")
	fmt.Println("  -simulate              Offline deterministic mode")
	fmt.Println("  -timeout duration      Timeout for MCP/LLM operations")
	fmt.Println("  -report string         Report path for one-shot mode")
	fmt.Println("  -help                  Show help")
}
