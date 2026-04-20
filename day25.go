package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"
)

type day25TaskState struct {
	Goal           string   `json:"goal"`
	Constraints    []string `json:"constraints"`
	Terms          []string `json:"terms"`
	Clarifications []string `json:"clarifications"`
	UpdatedAtUTC   string   `json:"updated_at_utc"`
}

type day25Session struct {
	Messages     []message      `json:"messages"`
	TaskState    day25TaskState `json:"task_state"`
	UpdatedAtUTC string         `json:"updated_at_utc"`
}

type day25ModelTaskUpdate struct {
	Goal           string   `json:"goal"`
	Constraints    []string `json:"constraints"`
	Terms          []string `json:"terms"`
	Clarifications []string `json:"clarifications"`
}

type day25ModelOutput struct {
	Answer          string               `json:"answer"`
	Sources         []day24SourceRef     `json:"sources"`
	TaskStateUpdate day25ModelTaskUpdate `json:"task_state_update"`
}

type day25TurnResult struct {
	UserInput       string
	QueryUsed       string
	BestScore       float64
	WeakContext     bool
	WeakReason      string
	Output          day25ModelOutput
	RetrievedBefore []day22RetrievedChunk
	RetrievedAfter  []day22RetrievedChunk
	Usage           usageStats
	Latency         time.Duration
}

type day25Config struct {
	IndexPath           string
	SessionFile         string
	WindowSize          int
	TopKBefore          int
	TopKAfter           int
	SimilarityThreshold float64
	UnsureThreshold     float64
	Model               string
	RewriteModel        string
	EmbeddingModel      string
	MaxTokens           int
	Temperature         *float64
	Simulate            bool
	ShowTokens          bool
}

type day25Scenario struct {
	Name     string
	Messages []string
}

type day25ScenarioTurn struct {
	Index       int
	UserInput   string
	Answer      string
	Sources     int
	WeakContext bool
	Goal        string
}

type day25ScenarioResult struct {
	Name             string
	Turns            []day25ScenarioTurn
	FinalState       day25TaskState
	AllWithSources   bool
	GoalStable       bool
	TurnsWithSources int
	UnknownTurns     int
}

type day25RunResult struct {
	Scenarios        []day25ScenarioResult
	TotalTurns       int
	TurnsWithSources int
	UnknownTurns     int
	AllWithSources   bool
	AllGoalsStable   bool
}

func runDay25Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day25", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	prompt := fs.String("prompt", "", "Single user question (non-interactive mode)")
	interactive := fs.Bool("interactive", true, "Run interactive mini-chat")
	runScenarios := fs.Bool("run-scenarios", false, "Run 2 long scenarios (10-15 messages each)")
	simulate := fs.Bool("simulate", false, "No API calls: use local deterministic responder (useful for offline tests)")
	reportPath := fs.String("report", "DAY25_RESULTS.md", "Scenario report path")
	indexPath := fs.String("index", "DAY21_INDEX_structured.json", "Path to Day21 index")
	sessionFile := fs.String("session-file", "/tmp/day25-session.json", "Path to persisted chat session JSON")
	resetSession := fs.Bool("reset-session", false, "Reset persisted session before run")
	windowSize := fs.Int("window-size", 10, "Recent history window for response generation")
	topKBefore := fs.Int("top-k-before", 12, "Top-K candidates before rerank/filter")
	topKAfter := fs.Int("top-k-after", 4, "Top-K chunks after rerank/filter")
	similarityThreshold := fs.Float64("similarity-threshold", 0.35, "Chunk relevance threshold")
	unsureThreshold := fs.Float64("unsure-threshold", 0.18, "If best score below threshold, assistant answers 'Не знаю'")
	model := fs.String("model", getDefaultModel(), "OpenRouter chat model")
	rewriteModel := fs.String("rewrite-model", getDefaultModel(), "OpenRouter rewrite model")
	embeddingModel := fs.String("embedding-model", defaultEmbeddingModel(), "OpenRouter embedding model")
	maxTokens := fs.Int("max-tokens", 320, "Max response tokens")
	temperature := fs.Float64("temperature", 0.2, "Model temperature")
	showTokens := fs.Bool("show-tokens", false, "Print token/latency stats in chat mode")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day25 flags: %w", err)
	}
	if *help {
		printDay25Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day25 arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *windowSize <= 0 {
		return fmt.Errorf("window-size must be positive")
	}
	if *topKBefore <= 0 || *topKAfter <= 0 {
		return fmt.Errorf("top-k-before/top-k-after must be positive")
	}
	if *topKAfter > *topKBefore {
		return fmt.Errorf("top-k-after cannot exceed top-k-before")
	}
	if *maxTokens <= 0 {
		return fmt.Errorf("max-tokens must be positive")
	}
	if *similarityThreshold < 0 || *similarityThreshold > 1.5 {
		return fmt.Errorf("similarity-threshold should be in range [0..1.5]")
	}
	if *unsureThreshold < 0 || *unsureThreshold > 1.5 {
		return fmt.Errorf("unsure-threshold should be in range [0..1.5]")
	}
	if !*interactive && !*runScenarios && strings.TrimSpace(*prompt) == "" {
		return fmt.Errorf("empty prompt: use -prompt, -interactive, or -run-scenarios")
	}

	index, err := loadDay22Index(strings.TrimSpace(*indexPath))
	if err != nil {
		return err
	}
	if len(index.Chunks) == 0 {
		return fmt.Errorf("index has no chunks")
	}

	var apiKey string
	if !*simulate {
		apiKey = getAPIKey()
	}
	temp := *temperature
	cfg := day25Config{
		IndexPath:           strings.TrimSpace(*indexPath),
		SessionFile:         strings.TrimSpace(*sessionFile),
		WindowSize:          *windowSize,
		TopKBefore:          *topKBefore,
		TopKAfter:           *topKAfter,
		SimilarityThreshold: *similarityThreshold,
		UnsureThreshold:     *unsureThreshold,
		Model:               *model,
		RewriteModel:        *rewriteModel,
		EmbeddingModel:      *embeddingModel,
		MaxTokens:           *maxTokens,
		Temperature:         &temp,
		Simulate:            *simulate,
		ShowTokens:          *showTokens,
	}

	ctx := context.Background()
	if *runScenarios {
		result, err := runDay25Scenarios(ctx, cfg, apiKey, index)
		if err != nil {
			return err
		}
		printDay25ScenarioSummary(result)
		if err := writeDay25Report(strings.TrimSpace(*reportPath), result, cfg); err != nil {
			return err
		}
		fmt.Printf("Отчёт: %s\n", strings.TrimSpace(*reportPath))
		return nil
	}

	if *resetSession {
		if err := os.Remove(cfg.SessionFile); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("failed to reset session: %w", err)
		}
	}

	session, err := loadDay25Session(cfg.SessionFile)
	if err != nil {
		return err
	}

	if strings.TrimSpace(*prompt) != "" {
		turn, err := day25ProcessTurn(ctx, cfg, apiKey, index, &session, strings.TrimSpace(*prompt))
		if err != nil {
			return err
		}
		if err := saveDay25Session(cfg.SessionFile, session); err != nil {
			return err
		}
		printDay25Turn(turn, cfg.ShowTokens)
		return nil
	}

	if *interactive {
		return runDay25Interactive(ctx, cfg, apiKey, index, &session)
	}
	return nil
}

func runDay25Interactive(ctx context.Context, cfg day25Config, apiKey string, index day21Index, session *day25Session) error {
	fmt.Println("Day25 mini-chat mode. Type /exit to quit.")
	fmt.Println("Commands:")
	fmt.Println("  /state")
	fmt.Println("  /history")
	fmt.Println("")
	if len(session.Messages) > 0 {
		fmt.Printf("session> restored %d messages from %s\n", len(session.Messages), cfg.SessionFile)
	}

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
		switch line {
		case "/exit", "exit", "quit":
			return nil
		case "/state":
			fmt.Printf("state>\n%s\n\n", day25RenderTaskState(session.TaskState))
			continue
		case "/history":
			fmt.Printf("history>\n%s\n\n", day25RenderHistory(keepLastMessages(session.Messages, 12)))
			continue
		}

		turn, err := day25ProcessTurn(ctx, cfg, apiKey, index, session, line)
		if err != nil {
			fmt.Printf("error: %v\n\n", err)
			continue
		}
		if err := saveDay25Session(cfg.SessionFile, *session); err != nil {
			return err
		}
		printDay25Turn(turn, cfg.ShowTokens)
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("failed to read stdin: %w", err)
	}
	return nil
}

func day25ProcessTurn(ctx context.Context, cfg day25Config, apiKey string, index day21Index, session *day25Session, userInput string) (day25TurnResult, error) {
	userInput = strings.TrimSpace(userInput)
	if userInput == "" {
		return day25TurnResult{}, fmt.Errorf("empty user input")
	}

	day25UpdateTaskStateHeuristic(&session.TaskState, userInput)
	queryForRAG := day25BuildQuery(userInput, session.TaskState)
	queryUsed, before, after, bestScore, err := day25RetrieveContext(ctx, cfg, apiKey, index, queryForRAG, userInput, session.TaskState)
	if err != nil {
		return day25TurnResult{}, err
	}

	weak := bestScore < cfg.UnsureThreshold
	turn := day25TurnResult{
		UserInput:       userInput,
		QueryUsed:       queryUsed,
		BestScore:       bestScore,
		WeakContext:     weak,
		RetrievedBefore: before,
		RetrievedAfter:  after,
	}
	if weak {
		turn.WeakReason = fmt.Sprintf("best score %.4f below unsure threshold %.4f", bestScore, cfg.UnsureThreshold)
	}

	var output day25ModelOutput
	if weak {
		output = day25BuildUnsureOutput(bestScore, after)
	} else if cfg.Simulate {
		output = day25BuildSimulatedOutput(userInput, session.TaskState, after)
	} else {
		resp, err := day25AskModel(cfg, apiKey, session.TaskState, keepLastMessages(session.Messages, cfg.WindowSize), userInput, after)
		if err != nil {
			return day25TurnResult{}, err
		}
		turn.Usage = resp.Usage
		turn.Latency = resp.Latency
		parsed, err := day25ParseModelOutput(resp.Answer)
		if err != nil {
			parsed = day25BuildFallbackOutput(resp.Answer, after)
		}
		output = parsed
	}
	output = day25NormalizeOutput(output, userInput, after)
	day25ApplyTaskUpdate(&session.TaskState, output.TaskStateUpdate)

	assistantText := strings.TrimSpace(output.Answer)
	session.Messages = append(session.Messages,
		message{Role: "user", Content: userInput},
		message{Role: "assistant", Content: assistantText},
	)
	session.UpdatedAtUTC = time.Now().UTC().Format(time.RFC3339)
	session.TaskState.UpdatedAtUTC = session.UpdatedAtUTC

	turn.Output = output
	return turn, nil
}

func day25BuildQuery(userInput string, state day25TaskState) string {
	var b strings.Builder
	b.WriteString(strings.TrimSpace(userInput))
	if goal := strings.TrimSpace(state.Goal); goal != "" {
		b.WriteString("\ngoal: ")
		b.WriteString(goal)
	}
	if len(state.Constraints) > 0 {
		b.WriteString("\nconstraints: ")
		b.WriteString(strings.Join(state.Constraints, "; "))
	}
	if len(state.Terms) > 0 {
		b.WriteString("\nterms: ")
		b.WriteString(strings.Join(state.Terms, ", "))
	}
	return b.String()
}

func day25RetrieveContext(ctx context.Context, cfg day25Config, apiKey string, index day21Index, queryForRAG, userInput string, state day25TaskState) (string, []day22RetrievedChunk, []day22RetrievedChunk, float64, error) {
	queryUsed := strings.TrimSpace(queryForRAG)
	if queryUsed == "" {
		queryUsed = strings.TrimSpace(userInput)
	}
	if !cfg.Simulate && apiKey != "" {
		if rewritten, err := day23RewriteQuery(apiKey, cfg.RewriteModel, queryUsed); err == nil && strings.TrimSpace(rewritten) != "" {
			queryUsed = strings.TrimSpace(rewritten)
		}
	}

	var before []day22RetrievedChunk
	var err error
	if cfg.Simulate || apiKey == "" {
		before, err = day25RetrieveLexical(index, queryUsed, cfg.TopKBefore)
	} else {
		before, err = day23RetrieveByEmbedding(ctx, apiKey, cfg.EmbeddingModel, index, queryUsed, cfg.TopKBefore)
		if err != nil {
			before, err = day25RetrieveLexical(index, queryUsed, cfg.TopKBefore)
		}
	}
	if err != nil {
		return "", nil, nil, 0, err
	}
	hints := day25BuildSourceHints(state)
	scored := day24Rerank(before, userInput, queryUsed, hints)
	if len(scored) == 0 {
		return "", nil, nil, 0, fmt.Errorf("no chunks after rerank")
	}
	best := scored[0].Score
	after := day24FilterByThreshold(scored, cfg.SimilarityThreshold, cfg.TopKAfter)
	if len(after) == 0 {
		after = day23TakeTop(scored, minInt(1, len(scored)))
	}
	return queryUsed, before, after, best, nil
}

func day25RetrieveLexical(index day21Index, query string, topK int) ([]day22RetrievedChunk, error) {
	queryTokens := day22Tokens(query)
	scored := make([]day22RetrievedChunk, 0, len(index.Chunks))
	for _, chunk := range index.Chunks {
		score := day22LexicalScore(queryTokens, chunk)
		scored = append(scored, day22RetrievedChunk{
			ChunkID: chunk.ChunkID,
			Source:  chunk.Source,
			Title:   chunk.Title,
			Section: chunk.Section,
			Score:   score,
			Text:    chunk.Text,
		})
	}
	if len(scored) == 0 {
		return nil, fmt.Errorf("index has no chunks")
	}
	sort.Slice(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })
	if topK > len(scored) {
		topK = len(scored)
	}
	return scored[:topK], nil
}

func day25BuildSourceHints(state day25TaskState) []string {
	hints := make([]string, 0, len(state.Terms)+len(state.Constraints)+1)
	if state.Goal != "" {
		hints = append(hints, state.Goal)
	}
	hints = append(hints, state.Terms...)
	hints = append(hints, state.Constraints...)
	return hints
}

func day25AskModel(cfg day25Config, apiKey string, state day25TaskState, history []message, userInput string, chunks []day22RetrievedChunk) (openRouterResult, error) {
	systemPrompt := strings.Join([]string{
		"Ты production-like RAG ассистент mini-chat.",
		"Используй только контекст и task state.",
		"Всегда возвращай JSON без markdown:",
		`{"answer":"...","sources":[{"source":"...","section":"...","chunk_id":"..."}],"task_state_update":{"goal":"...","constraints":[],"terms":[],"clarifications":[]}}`,
		"Если контекста недостаточно: answer начинается с 'Не знаю' и просит уточнение.",
		"sources должен содержать реально использованные источники.",
	}, "\n")
	userPrompt := strings.Join([]string{
		"Task state:",
		day25RenderTaskState(state),
		"",
		"Recent history:",
		day25RenderHistory(history),
		"",
		"RAG context:",
		day24BuildContext(chunks),
		"",
		"Current user question:",
		userInput,
	}, "\n")
	resp, err := callOpenRouterDetailed(
		apiKey,
		cfg.Model,
		[]message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
		cfg.MaxTokens,
		cfg.Temperature,
		nil,
		"openrouter-cli-day25",
	)
	if err != nil {
		return openRouterResult{}, fmt.Errorf("day25 model call failed: %w", err)
	}
	return resp, nil
}

func day25ParseModelOutput(raw string) (day25ModelOutput, error) {
	candidate := strings.TrimSpace(raw)
	candidate = strings.TrimPrefix(candidate, "```json")
	candidate = strings.TrimPrefix(candidate, "```JSON")
	candidate = strings.TrimPrefix(candidate, "```")
	candidate = strings.TrimSuffix(candidate, "```")
	candidate = strings.TrimSpace(candidate)

	if !strings.HasPrefix(candidate, "{") || !strings.HasSuffix(candidate, "}") {
		start := strings.Index(candidate, "{")
		end := strings.LastIndex(candidate, "}")
		if start == -1 || end == -1 || end <= start {
			return day25ModelOutput{}, fmt.Errorf("json object not found")
		}
		candidate = candidate[start : end+1]
	}
	var out day25ModelOutput
	if err := json.Unmarshal([]byte(candidate), &out); err != nil {
		return day25ModelOutput{}, fmt.Errorf("invalid day25 output JSON: %w", err)
	}
	return out, nil
}

func day25BuildFallbackOutput(rawAnswer string, chunks []day22RetrievedChunk) day25ModelOutput {
	answer := strings.TrimSpace(rawAnswer)
	if answer == "" {
		answer = "Не знаю. Уточните вопрос."
	}
	out := day25ModelOutput{
		Answer: answer,
	}
	if len(chunks) > 0 {
		out.Sources = append(out.Sources, day24SourceRef{
			Source:  chunks[0].Source,
			Section: chunks[0].Section,
			ChunkID: chunks[0].ChunkID,
		})
	}
	return out
}

func day25BuildUnsureOutput(bestScore float64, chunks []day22RetrievedChunk) day25ModelOutput {
	out := day25ModelOutput{
		Answer: fmt.Sprintf("Не знаю: релевантного контекста недостаточно (best score %.3f). Уточните вопрос и укажите нужный файл/термин.", bestScore),
	}
	if len(chunks) > 0 {
		out.Sources = append(out.Sources, day24SourceRef{
			Source:  chunks[0].Source,
			Section: chunks[0].Section,
			ChunkID: chunks[0].ChunkID,
		})
	}
	return out
}

func day25BuildSimulatedOutput(userInput string, state day25TaskState, chunks []day22RetrievedChunk) day25ModelOutput {
	answer := "По найденному контексту: "
	if len(chunks) > 0 {
		answer += day24ExtractQuote(chunks[0].Text, userInput)
	} else {
		answer += "данных недостаточно."
	}
	if strings.TrimSpace(state.Goal) != "" {
		answer += " Текущая цель: " + strings.TrimSpace(state.Goal) + "."
	}
	out := day25ModelOutput{
		Answer: answer,
	}
	for _, chunk := range day23TakeTop(chunks, 2) {
		out.Sources = append(out.Sources, day24SourceRef{
			Source:  chunk.Source,
			Section: chunk.Section,
			ChunkID: chunk.ChunkID,
		})
	}
	return out
}

func day25NormalizeOutput(out day25ModelOutput, userInput string, chunks []day22RetrievedChunk) day25ModelOutput {
	out.Answer = strings.TrimSpace(out.Answer)
	if out.Answer == "" {
		out.Answer = "Не знаю. Уточните вопрос."
	}

	normalized := make([]day24SourceRef, 0, len(out.Sources))
	seen := map[string]struct{}{}
	for _, src := range out.Sources {
		src.Source = strings.TrimSpace(src.Source)
		src.Section = strings.TrimSpace(src.Section)
		src.ChunkID = strings.TrimSpace(src.ChunkID)
		if src.Source == "" && src.ChunkID == "" {
			continue
		}
		if src.Source == "" && src.ChunkID != "" {
			for _, ch := range chunks {
				if strings.EqualFold(ch.ChunkID, src.ChunkID) {
					src.Source = ch.Source
					if src.Section == "" {
						src.Section = ch.Section
					}
					break
				}
			}
		}
		key := strings.ToLower(src.Source + "|" + src.Section + "|" + src.ChunkID)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		normalized = append(normalized, src)
	}
	out.Sources = normalized
	if len(out.Sources) == 0 && len(chunks) > 0 {
		out.Sources = append(out.Sources, day24SourceRef{
			Source:  chunks[0].Source,
			Section: chunks[0].Section,
			ChunkID: chunks[0].ChunkID,
		})
	}
	return out
}

var (
	day25GoalPrefixRe       = regexp.MustCompile(`(?i)(?:цель|goal)\s*:\s*(.+)$`)
	day25ConstraintMarkerRe = regexp.MustCompile(`(?i)(?:ограничение|constraint)\s*:\s*(.+)$`)
	day25QuotedTermRe       = regexp.MustCompile("[`\"«](.*?)[`\"»]")
	day25TermMarkerRe       = regexp.MustCompile(`(?i)термин\s*:\s*(.+)$`)
)

func day25UpdateTaskStateHeuristic(state *day25TaskState, userInput string) {
	userInput = strings.TrimSpace(userInput)
	if userInput == "" {
		return
	}
	state.Clarifications = appendUnique(state.Clarifications, userInput)

	if m := day25GoalPrefixRe.FindStringSubmatch(userInput); len(m) == 2 {
		state.Goal = strings.TrimSpace(m[1])
	} else if state.Goal == "" {
		lower := strings.ToLower(userInput)
		if strings.Contains(lower, "хочу") || strings.Contains(lower, "нужно") || strings.Contains(lower, "задача") {
			state.Goal = day25TrimSentence(userInput, 140)
		}
	}

	if m := day25ConstraintMarkerRe.FindStringSubmatch(userInput); len(m) == 2 {
		state.Constraints = appendUnique(state.Constraints, strings.TrimSpace(m[1]))
	}
	for _, sent := range day25SplitSentences(userInput) {
		lower := strings.ToLower(sent)
		if strings.Contains(lower, "огранич") || strings.Contains(lower, "только") || strings.Contains(lower, "нельзя") || strings.Contains(lower, "без ") {
			state.Constraints = appendUnique(state.Constraints, strings.TrimSpace(sent))
		}
	}

	for _, m := range day25QuotedTermRe.FindAllStringSubmatch(userInput, -1) {
		if len(m) == 2 {
			term := strings.TrimSpace(m[1])
			if len(term) >= 2 {
				state.Terms = appendUnique(state.Terms, term)
			}
		}
	}
	if m := day25TermMarkerRe.FindStringSubmatch(userInput); len(m) == 2 {
		term := strings.TrimSpace(m[1])
		if term != "" {
			state.Terms = appendUnique(state.Terms, term)
		}
	}

	state.UpdatedAtUTC = time.Now().UTC().Format(time.RFC3339)
	if len(state.Clarifications) > 50 {
		state.Clarifications = append([]string(nil), state.Clarifications[len(state.Clarifications)-50:]...)
	}
	if len(state.Constraints) > 25 {
		state.Constraints = append([]string(nil), state.Constraints[len(state.Constraints)-25:]...)
	}
	if len(state.Terms) > 25 {
		state.Terms = append([]string(nil), state.Terms[len(state.Terms)-25:]...)
	}
}

func day25ApplyTaskUpdate(state *day25TaskState, upd day25ModelTaskUpdate) {
	if goal := strings.TrimSpace(upd.Goal); goal != "" {
		if state.Goal == "" {
			state.Goal = goal
		}
	}
	for _, item := range upd.Constraints {
		item = strings.TrimSpace(item)
		if item != "" {
			state.Constraints = appendUnique(state.Constraints, item)
		}
	}
	for _, item := range upd.Terms {
		item = strings.TrimSpace(item)
		if item != "" {
			state.Terms = appendUnique(state.Terms, item)
		}
	}
	for _, item := range upd.Clarifications {
		item = strings.TrimSpace(item)
		if item != "" {
			state.Clarifications = appendUnique(state.Clarifications, item)
		}
	}
	state.UpdatedAtUTC = time.Now().UTC().Format(time.RFC3339)
}

func day25SplitSentences(text string) []string {
	parts := strings.FieldsFunc(text, func(r rune) bool {
		return r == '.' || r == '\n' || r == ';' || r == '!' || r == '?'
	})
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func day25TrimSentence(text string, limit int) string {
	text = strings.TrimSpace(strings.Join(strings.Fields(text), " "))
	if len(text) <= limit {
		return text
	}
	return text[:limit]
}

func day25RenderTaskState(state day25TaskState) string {
	var b strings.Builder
	if strings.TrimSpace(state.Goal) == "" {
		b.WriteString("- goal: (empty)\n")
	} else {
		b.WriteString("- goal: " + state.Goal + "\n")
	}
	b.WriteString("- constraints: " + day25RenderList(state.Constraints, 6) + "\n")
	b.WriteString("- terms: " + day25RenderList(state.Terms, 6) + "\n")
	b.WriteString("- clarifications: " + day25RenderList(state.Clarifications, 6) + "\n")
	if state.UpdatedAtUTC != "" {
		b.WriteString("- updated_at_utc: " + state.UpdatedAtUTC + "\n")
	}
	return b.String()
}

func day25RenderList(items []string, limit int) string {
	if len(items) == 0 {
		return "(none)"
	}
	if len(items) > limit {
		items = items[len(items)-limit:]
	}
	return strings.Join(items, " | ")
}

func day25RenderHistory(history []message) string {
	if len(history) == 0 {
		return "(empty)"
	}
	var b strings.Builder
	for i, m := range history {
		b.WriteString(fmt.Sprintf("%d. %s: %s\n", i+1, m.Role, sanitizeCodeFences(strings.TrimSpace(m.Content))))
	}
	return strings.TrimSpace(b.String())
}

func printDay25Turn(turn day25TurnResult, showTokens bool) {
	fmt.Printf("assistant> %s\n", turn.Output.Answer)
	fmt.Println("sources>")
	for i, src := range turn.Output.Sources {
		fmt.Printf("  %d. source=%s section=%s chunk_id=%s\n", i+1, src.Source, src.Section, src.ChunkID)
	}
	if turn.WeakContext {
		fmt.Printf("state> weak_context=true reason=%s\n", turn.WeakReason)
	}
	if showTokens {
		fmt.Printf("tokens> prompt=%d completion=%d total=%d latency=%s best_score=%.4f\n",
			turn.Usage.PromptTokens,
			turn.Usage.CompletionTokens,
			turn.Usage.TotalTokens,
			turn.Latency.Round(time.Millisecond),
			turn.BestScore,
		)
	}
	fmt.Println("")
}

func loadDay25Session(path string) (day25Session, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return day25Session{}, fmt.Errorf("session file path is empty")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return day25Session{}, nil
		}
		return day25Session{}, fmt.Errorf("failed to read session file: %w", err)
	}
	var out day25Session
	if err := json.Unmarshal(data, &out); err != nil {
		return day25Session{}, fmt.Errorf("failed to parse session JSON: %w", err)
	}
	return out, nil
}

func saveDay25Session(path string, session day25Session) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return fmt.Errorf("session file path is empty")
	}
	if dir := strings.TrimSpace(strings.TrimSuffix(path, "/"+filepathBase(path))); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("failed to create session dir: %w", err)
		}
	}
	payload, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode session JSON: %w", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, payload, 0o644); err != nil {
		return fmt.Errorf("failed to write temp session: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("failed to replace session file: %w", err)
	}
	return nil
}

func filepathBase(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	parts := strings.Split(path, "/")
	return parts[len(parts)-1]
}

func runDay25Scenarios(ctx context.Context, cfg day25Config, apiKey string, index day21Index) (day25RunResult, error) {
	scenarios := defaultDay25Scenarios()
	out := day25RunResult{
		Scenarios: make([]day25ScenarioResult, 0, len(scenarios)),
	}
	for _, scenario := range scenarios {
		session := day25Session{}
		result := day25ScenarioResult{
			Name:           scenario.Name,
			Turns:          make([]day25ScenarioTurn, 0, len(scenario.Messages)),
			AllWithSources: true,
		}
		goalSnapshots := make([]string, 0, len(scenario.Messages))
		for i, userInput := range scenario.Messages {
			turn, err := day25ProcessTurn(ctx, cfg, apiKey, index, &session, userInput)
			if err != nil {
				return day25RunResult{}, fmt.Errorf("scenario %s turn %d failed: %w", scenario.Name, i+1, err)
			}
			sourcesCount := len(turn.Output.Sources)
			if sourcesCount == 0 {
				result.AllWithSources = false
			}
			if turn.WeakContext {
				result.UnknownTurns++
			}
			result.TurnsWithSources += sourcesCount
			result.Turns = append(result.Turns, day25ScenarioTurn{
				Index:       i + 1,
				UserInput:   userInput,
				Answer:      turn.Output.Answer,
				Sources:     sourcesCount,
				WeakContext: turn.WeakContext,
				Goal:        session.TaskState.Goal,
			})
			goalSnapshots = append(goalSnapshots, session.TaskState.Goal)
		}
		result.FinalState = session.TaskState
		result.GoalStable = day25GoalStable(goalSnapshots)
		out.Scenarios = append(out.Scenarios, result)
	}

	out.TotalTurns = 0
	out.TurnsWithSources = 0
	out.UnknownTurns = 0
	out.AllWithSources = true
	out.AllGoalsStable = true
	for _, scenario := range out.Scenarios {
		out.TotalTurns += len(scenario.Turns)
		for _, turn := range scenario.Turns {
			if turn.Sources > 0 {
				out.TurnsWithSources++
			}
		}
		out.UnknownTurns += scenario.UnknownTurns
		if !scenario.AllWithSources {
			out.AllWithSources = false
		}
		if !scenario.GoalStable {
			out.AllGoalsStable = false
		}
	}
	return out, nil
}

func day25GoalStable(goals []string) bool {
	first := ""
	for _, g := range goals {
		g = strings.TrimSpace(g)
		if g != "" {
			first = g
			break
		}
	}
	if first == "" {
		return false
	}
	for _, g := range goals {
		g = strings.TrimSpace(g)
		if g == "" {
			return false
		}
		if !day25GoalsCompatible(first, g) {
			return false
		}
	}
	return true
}

func day25GoalsCompatible(a, b string) bool {
	al := strings.ToLower(strings.TrimSpace(a))
	bl := strings.ToLower(strings.TrimSpace(b))
	if al == "" || bl == "" {
		return false
	}
	if strings.Contains(al, bl) || strings.Contains(bl, al) {
		return true
	}
	at := day25MeaningTokens(day22Tokens(al))
	bt := day25MeaningTokens(day22Tokens(bl))
	if len(at) == 0 || len(bt) == 0 {
		return false
	}
	set := map[string]struct{}{}
	for _, t := range at {
		set[t] = struct{}{}
	}
	hits := 0
	for _, t := range bt {
		if _, ok := set[t]; ok {
			hits++
		}
	}
	return hits >= 2
}

func day25MeaningTokens(tokens []string) []string {
	stop := map[string]struct{}{
		"это": {}, "как": {}, "что": {}, "для": {}, "или": {}, "чтобы": {}, "проект": {}, "goal": {}, "task": {},
	}
	out := make([]string, 0, len(tokens))
	for _, t := range tokens {
		if len(t) < 3 {
			continue
		}
		if _, ok := stop[t]; ok {
			continue
		}
		out = append(out, t)
	}
	return out
}

func defaultDay25Scenarios() []day25Scenario {
	return []day25Scenario{
		{
			Name: "scenario_1_mvp_rag_cli",
			Messages: []string{
				"Цель: собрать production-like MVP мини-чата с RAG для внутренних документов.",
				"Ограничение: только CLI на Go, без веб-интерфейса.",
				"Термин: Source-of-Truth = DAY21_INDEX_structured.json.",
				"Нужно, чтобы каждый ответ всегда показывал источники.",
				"Уточнение: историю диалога храним в JSON-файле.",
				"Ограничение: нельзя терять цель диалога между сообщениями.",
				"Добавь память задачи: что уже уточнили и какие ограничения зафиксированы.",
				"Какие шаги реализации сделать в первую очередь?",
				"Напомни текущую цель и ограничения.",
				"Если контекст слабый, ассистент должен говорить не знаю. Зафиксируй это.",
				"Собери короткий план на следующий спринт.",
				"Финально: дай краткий итог с источниками.",
			},
		},
		{
			Name: "scenario_2_team_knowledge_chat",
			Messages: []string{
				"Цель: построить чат для команды, который отвечает по результатам Day10-Day24.",
				"Ограничение: в ответе всегда минимум один источник source+section+chunk_id.",
				"Термин: task memory = goal + constraints + terms + clarifications.",
				"Уточнение: если пользователь меняет требования, сохраняем это как clarification.",
				"Нужно чтобы чат учитывал и историю, и RAG-контекст одновременно.",
				"Ограничение: нельзя выдумывать факты без подтверждения в контексте.",
				"Сформулируй архитектуру мини-чата из 4-6 пунктов.",
				"Проверь, что цель не потерялась после длинного диалога.",
				"Напомни ключевые термины, которые мы уже закрепили.",
				"Как будем тестировать на 10-15 сообщениях?",
				"Сделай короткий action plan внедрения.",
				"Заверши ответом с источниками и текущим состоянием задачи.",
			},
		},
	}
}

func printDay25ScenarioSummary(result day25RunResult) {
	fmt.Println("=== Day 25: Mini-chat + RAG + Task Memory ===")
	fmt.Printf("scenarios=%d total_turns=%d turns_with_sources=%d unknown_turns=%d\n",
		len(result.Scenarios), result.TotalTurns, result.TurnsWithSources, result.UnknownTurns)
	fmt.Printf("all_with_sources=%t all_goals_stable=%t\n", result.AllWithSources, result.AllGoalsStable)
	for _, scenario := range result.Scenarios {
		fmt.Printf("- %s: turns=%d goal_stable=%t all_with_sources=%t unknown_turns=%d final_goal=%q\n",
			scenario.Name,
			len(scenario.Turns),
			scenario.GoalStable,
			scenario.AllWithSources,
			scenario.UnknownTurns,
			scenario.FinalState.Goal,
		)
	}
}

func writeDay25Report(path string, result day25RunResult, cfg day25Config) error {
	var b strings.Builder
	b.WriteString("# Day 25 Results: Mini-chat with RAG + Task Memory\n\n")
	b.WriteString(fmt.Sprintf("- index: `%s`\n", cfg.IndexPath))
	b.WriteString(fmt.Sprintf("- mode: `%s`\n", map[bool]string{true: "simulate", false: "api"}[cfg.Simulate]))
	b.WriteString(fmt.Sprintf("- top-k before: `%d`\n", cfg.TopKBefore))
	b.WriteString(fmt.Sprintf("- top-k after: `%d`\n", cfg.TopKAfter))
	b.WriteString(fmt.Sprintf("- similarity threshold: `%.2f`\n", cfg.SimilarityThreshold))
	b.WriteString(fmt.Sprintf("- unsure threshold: `%.2f`\n", cfg.UnsureThreshold))
	b.WriteString(fmt.Sprintf("- scenarios: `%d`\n", len(result.Scenarios)))
	b.WriteString(fmt.Sprintf("- total turns: `%d`\n", result.TotalTurns))
	b.WriteString(fmt.Sprintf("- turns with sources: `%d/%d`\n", result.TurnsWithSources, result.TotalTurns))
	b.WriteString(fmt.Sprintf("- all with sources: `%t`\n", result.AllWithSources))
	b.WriteString(fmt.Sprintf("- all goals stable: `%t`\n", result.AllGoalsStable))
	b.WriteString(fmt.Sprintf("- unknown turns: `%d`\n\n", result.UnknownTurns))

	for _, scenario := range result.Scenarios {
		b.WriteString("## " + scenario.Name + "\n")
		b.WriteString(fmt.Sprintf("- turns: `%d`\n", len(scenario.Turns)))
		b.WriteString(fmt.Sprintf("- all with sources: `%t`\n", scenario.AllWithSources))
		b.WriteString(fmt.Sprintf("- goal stable: `%t`\n", scenario.GoalStable))
		b.WriteString(fmt.Sprintf("- unknown turns: `%d`\n", scenario.UnknownTurns))
		b.WriteString(fmt.Sprintf("- final goal: `%s`\n", escapeDay22Table(scenario.FinalState.Goal)))
		b.WriteString("### Final Task State\n")
		b.WriteString("```text\n")
		b.WriteString(day25RenderTaskState(scenario.FinalState))
		b.WriteString("```\n")
		b.WriteString("### Turns\n")
		b.WriteString("| # | User | Sources | Weak Context | Goal Snapshot |\n")
		b.WriteString("| --- | --- | ---: | ---: | --- |\n")
		for _, turn := range scenario.Turns {
			b.WriteString(fmt.Sprintf("| %d | %s | %d | %t | %s |\n",
				turn.Index,
				escapeDay22Table(turn.UserInput),
				turn.Sources,
				turn.WeakContext,
				escapeDay22Table(turn.Goal),
			))
		}
		b.WriteString("\n")
	}
	b.WriteString("Conclusion: mini-chat persists dialogue + task state, performs RAG each turn, and keeps sources visible in responses.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay25Usage() {
	fmt.Println("Usage: openrouter-cli day25 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -prompt string               Single user question")
	fmt.Println("  -interactive                 Run interactive mini-chat")
	fmt.Println("  -run-scenarios               Run 2 long scenarios")
	fmt.Println("  -simulate                    No API calls (local deterministic mode)")
	fmt.Println("  -report string               Scenario report path")
	fmt.Println("  -index string                Path to Day21 index JSON")
	fmt.Println("  -session-file string         Session JSON file")
	fmt.Println("  -reset-session               Reset session file before run")
	fmt.Println("  -window-size int             Recent history window")
	fmt.Println("  -top-k-before int            Top-K before rerank/filter")
	fmt.Println("  -top-k-after int             Top-K after rerank/filter")
	fmt.Println("  -similarity-threshold float  Relevance threshold")
	fmt.Println("  -unsure-threshold float      'Не знаю' threshold")
	fmt.Println("  -model string                Chat model")
	fmt.Println("  -rewrite-model string        Query rewrite model")
	fmt.Println("  -embedding-model string      Embedding model")
	fmt.Println("  -max-tokens int              Max response tokens")
	fmt.Println("  -temperature float           Temperature")
	fmt.Println("  -show-tokens                 Show token stats")
	fmt.Println("  -help                        Show help")
}
