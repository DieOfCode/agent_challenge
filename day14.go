package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type day14Invariant struct {
	ID          string   `json:"id"`
	Category    string   `json:"category"`
	Rule        string   `json:"rule"`
	Rationale   string   `json:"rationale"`
	Forbidden   []string `json:"forbidden,omitempty"`
	Alternative string   `json:"alternative,omitempty"`
}

type day14InvariantFile struct {
	Version    int              `json:"version"`
	UpdatedAt  string           `json:"updated_at"`
	Invariants []day14Invariant `json:"invariants"`
}

type day14HistoryFile struct {
	Version  int             `json:"version"`
	Messages []storedMessage `json:"messages"`
}

type day14InvariantConflict struct {
	Invariant day14Invariant
	Matched   []string
}

type day14AgentResponse struct {
	Text      string
	Refused   bool
	Conflicts []day14InvariantConflict
	Tokens    tokenStats
}

type day14InvariantAgent struct {
	apiKey                   string
	model                    string
	maxTokens                int
	temperature              *float64
	title                    string
	offline                  bool
	windowSize               int
	historyFile              string
	invariantsFile           string
	history                  []message
	invariants               []day14Invariant
	cumulativePromptTokens   int
	cumulativeResponseTokens int
}

type day14ScenarioResult struct {
	Name      string
	Request   string
	Response  string
	Refused   bool
	Conflicts []string
	Tokens    tokenStats
}

type day14DemoResult struct {
	Model              string
	Offline            bool
	InvariantsFile     string
	HistoryFile        string
	InvariantCount     int
	ConflictCasePassed bool
	ExplanationPassed  bool
	AllowedCasePassed  bool
	Scenarios          []day14ScenarioResult
}

func runDay14Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day14", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	model := fs.String("model", getDefaultModel(), "OpenRouter model")
	maxTokens := fs.Int("max-tokens", 260, "Maximum response tokens")
	temperature := fs.Float64("temperature", 0.2, "Temperature")
	historyFile := fs.String("history-file", "/tmp/day14-history.json", "Path to dialogue history JSON")
	invariantsFile := fs.String("invariants-file", "/tmp/day14-invariants.json", "Path to invariants JSON")
	windowSize := fs.Int("window-size", 12, "How many latest dialogue messages are sent to model")
	reportPath := fs.String("report", "DAY14_RESULTS.md", "Markdown report path")
	interactive := fs.Bool("interactive", false, "Run interactive mode")
	showTokens := fs.Bool("show-tokens", false, "Show token stats in interactive mode")
	resetHistory := fs.Bool("reset-history", true, "Reset dialogue history before run")
	resetInvariants := fs.Bool("reset-invariants", false, "Reset invariants file to defaults before run")
	offline := fs.Bool("offline", false, "Run without API call for allowed requests")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day14 flags: %w", err)
	}
	if *help {
		printDay14Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day14 arguments: %s", strings.Join(fs.Args(), " "))
	}

	if *resetHistory {
		if err := os.Remove(*historyFile); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	if *resetInvariants {
		if err := os.Remove(*invariantsFile); err != nil && !os.IsNotExist(err) {
			return err
		}
	}

	invariants, err := loadOrCreateDay14Invariants(*invariantsFile)
	if err != nil {
		return err
	}
	history, err := loadDay14History(*historyFile)
	if err != nil {
		return err
	}

	apiKey := ""
	if !*offline {
		apiKey = strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
		if apiKey == "" {
			return fmt.Errorf("OPENROUTER_API_KEY is not set (or use -offline)")
		}
	}

	t := *temperature
	agent := &day14InvariantAgent{
		apiKey:         apiKey,
		model:          *model,
		maxTokens:      *maxTokens,
		temperature:    &t,
		title:          "day14-invariants-agent",
		offline:        *offline,
		windowSize:     *windowSize,
		historyFile:    *historyFile,
		invariantsFile: *invariantsFile,
		history:        cloneMessages(history),
		invariants:     normalizeDay14Invariants(invariants),
	}

	if *interactive {
		return runDay14Interactive(agent, *showTokens)
	}

	result, err := runDay14Demo(agent)
	if err != nil {
		return err
	}
	printDay14DemoResult(result)
	if err := writeDay14Report(*reportPath, result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)
	return nil
}

func runDay14Demo(agent *day14InvariantAgent) (day14DemoResult, error) {
	cases := []struct {
		name    string
		request string
	}{
		{
			name:    "Conflict Request",
			request: "Переведём backend на Node.js и MongoDB, а KYC в оплате пока отключим.",
		},
		{
			name:    "Allowed Request",
			request: "Составь план оптимизации текущего Go + PostgreSQL монолита без изменения стека.",
		},
	}

	scenarios := make([]day14ScenarioResult, 0, len(cases))
	for _, item := range cases {
		response, err := agent.Reply(item.request)
		if err != nil {
			return day14DemoResult{}, fmt.Errorf("%s failed: %w", item.name, err)
		}
		conflicts := make([]string, 0, len(response.Conflicts))
		for _, c := range response.Conflicts {
			conflicts = append(conflicts, c.Invariant.ID)
		}
		scenarios = append(scenarios, day14ScenarioResult{
			Name:      item.name,
			Request:   item.request,
			Response:  strings.TrimSpace(response.Text),
			Refused:   response.Refused,
			Conflicts: conflicts,
			Tokens:    response.Tokens,
		})
	}

	conflictPassed := len(scenarios) > 0 && scenarios[0].Refused && len(scenarios[0].Conflicts) > 0
	explanationPassed := false
	if len(scenarios) > 0 {
		lower := strings.ToLower(scenarios[0].Response)
		explanationPassed = strings.Contains(lower, "инвариант") || strings.Contains(lower, "invariant")
		explanationPassed = explanationPassed && strings.Contains(lower, "конфликт")
	}
	allowedPassed := len(scenarios) > 1 && !scenarios[1].Refused

	return day14DemoResult{
		Model:              agent.model,
		Offline:            agent.offline,
		InvariantsFile:     agent.invariantsFile,
		HistoryFile:        agent.historyFile,
		InvariantCount:     len(agent.invariants),
		ConflictCasePassed: conflictPassed,
		ExplanationPassed:  explanationPassed,
		AllowedCasePassed:  allowedPassed,
		Scenarios:          scenarios,
	}, nil
}

func runDay14Interactive(agent *day14InvariantAgent, showTokens bool) error {
	fmt.Println("Day14 interactive mode. Type /exit to quit.")
	fmt.Println("Commands:")
	fmt.Println("  /inv show")
	fmt.Println("  /inv reload")
	fmt.Println("  /history show")
	fmt.Println("  /history clear")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("you> ")
		if !scanner.Scan() {
			return scanner.Err()
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		switch strings.ToLower(line) {
		case "/exit", "exit", "quit":
			return nil
		}
		if strings.HasPrefix(line, "/") {
			handled, err := handleDay14Command(agent, line)
			if handled {
				if err != nil {
					fmt.Fprintf(os.Stderr, "command error: %v\n", err)
				}
				continue
			}
		}

		response, err := agent.Reply(line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "agent error: %v\n", err)
			continue
		}
		fmt.Printf("agent> %s\n\n", response.Text)
		if showTokens {
			printTokenStats(response.Tokens)
		}
	}
}

func handleDay14Command(agent *day14InvariantAgent, line string) (bool, error) {
	fields := strings.Fields(strings.TrimSpace(line))
	if len(fields) == 0 {
		return false, nil
	}

	switch strings.ToLower(fields[0]) {
	case "/inv":
		if len(fields) < 2 {
			return true, fmt.Errorf("usage: /inv show|reload")
		}
		switch strings.ToLower(fields[1]) {
		case "show":
			fmt.Printf("invariants>\n%s\n\n", renderDay14InvariantsForHumans(agent.invariants))
			return true, nil
		case "reload":
			items, err := loadOrCreateDay14Invariants(agent.invariantsFile)
			if err != nil {
				return true, err
			}
			agent.invariants = normalizeDay14Invariants(items)
			fmt.Printf("invariants> loaded %d items from %s\n\n", len(agent.invariants), agent.invariantsFile)
			return true, nil
		default:
			return true, fmt.Errorf("unknown /inv action: %s", fields[1])
		}
	case "/history":
		if len(fields) < 2 {
			return true, fmt.Errorf("usage: /history show|clear")
		}
		switch strings.ToLower(fields[1]) {
		case "show":
			fmt.Printf("history>\n%s\n\n", renderDay14History(agent.history))
			return true, nil
		case "clear":
			agent.history = nil
			if err := saveDay14History(agent.historyFile, agent.history); err != nil {
				return true, err
			}
			fmt.Println("history> cleared")
			fmt.Println()
			return true, nil
		default:
			return true, fmt.Errorf("unknown /history action: %s", fields[1])
		}
	}
	return false, nil
}

func (a *day14InvariantAgent) Reply(userInput string) (day14AgentResponse, error) {
	userInput = strings.TrimSpace(userInput)
	if userInput == "" {
		return day14AgentResponse{}, fmt.Errorf("empty user input")
	}

	a.history = append(a.history, message{Role: "user", Content: userInput})

	conflicts := detectDay14InvariantConflicts(userInput, a.invariants)
	if len(conflicts) > 0 {
		refusal := renderDay14Refusal(conflicts)
		a.history = append(a.history, message{Role: "assistant", Content: refusal})
		if err := saveDay14History(a.historyFile, a.history); err != nil {
			return day14AgentResponse{}, err
		}
		stats := day14EstimatedTokenStats(a.history, refusal, a.windowSize, a.cumulativePromptTokens, a.cumulativeResponseTokens)
		return day14AgentResponse{
			Text:      refusal,
			Refused:   true,
			Conflicts: conflicts,
			Tokens:    stats,
		}, nil
	}

	if a.offline {
		answer := renderDay14OfflineAnswer(a.invariants, userInput)
		a.history = append(a.history, message{Role: "assistant", Content: answer})
		if err := saveDay14History(a.historyFile, a.history); err != nil {
			return day14AgentResponse{}, err
		}
		stats := day14EstimatedTokenStats(a.history, answer, a.windowSize, a.cumulativePromptTokens, a.cumulativeResponseTokens)
		return day14AgentResponse{
			Text:    answer,
			Refused: false,
			Tokens:  stats,
		}, nil
	}

	requestMessages := buildDay14RequestMessages(a.invariants, keepLastMessages(a.history, a.windowSize))
	result, err := callOpenRouterDetailed(
		a.apiKey,
		a.model,
		requestMessages,
		a.maxTokens,
		a.temperature,
		nil,
		a.title,
	)
	if err != nil {
		return day14AgentResponse{}, err
	}

	answer := strings.TrimSpace(result.Answer)
	postConflicts := detectDay14InvariantConflicts(answer, a.invariants)
	refused := false
	if len(postConflicts) > 0 {
		refused = true
		answer = renderDay14GeneratedConflictRefusal(postConflicts)
	}

	a.history = append(a.history, message{Role: "assistant", Content: answer})
	if err := saveDay14History(a.historyFile, a.history); err != nil {
		return day14AgentResponse{}, err
	}

	promptTokens := result.Usage.PromptTokens
	responseTokens := result.Usage.CompletionTokens
	totalTokens := result.Usage.TotalTokens
	if promptTokens == 0 {
		promptTokens = estimateMessagesTokens(requestMessages)
	}
	if responseTokens == 0 {
		responseTokens = estimateTextTokens(answer)
	}
	if totalTokens == 0 {
		totalTokens = promptTokens + responseTokens
	}

	a.cumulativePromptTokens += promptTokens
	a.cumulativeResponseTokens += responseTokens

	stats := tokenStats{
		EstimatedHistoryTokens:   estimateMessagesTokens(keepLastMessages(a.history, a.windowSize)),
		EstimatedRequestTokens:   estimateMessagesTokens(requestMessages),
		EstimatedResponseTokens:  estimateTextTokens(answer),
		PromptTokens:             promptTokens,
		ResponseTokens:           responseTokens,
		TotalTokens:              totalTokens,
		ConversationTokens:       estimateMessagesTokens(a.history),
		CumulativePromptTokens:   a.cumulativePromptTokens,
		CumulativeResponseTokens: a.cumulativeResponseTokens,
		CumulativeTotalTokens:    a.cumulativePromptTokens + a.cumulativeResponseTokens,
	}

	return day14AgentResponse{
		Text:      answer,
		Refused:   refused,
		Conflicts: postConflicts,
		Tokens:    stats,
	}, nil
}

func buildDay14RequestMessages(invariants []day14Invariant, history []message) []message {
	system := strings.TrimSpace(
		"Ты ассистент с жёсткими инвариантами. Нарушать их нельзя.\n" +
			"Перед ответом явно сделай проверку инвариантов.\n" +
			"Если запрос нарушает инварианты, откажись и объясни почему.\n" +
			"Формат ответа:\n" +
			"STATUS: OK|REFUSE\n" +
			"INVARIANT_CHECK:\n- ...\n" +
			"ANSWER:\n...\n",
	)
	invariantBlock := renderDay14InvariantsForPrompt(invariants)

	out := make([]message, 0, 2+len(history))
	out = append(out, message{Role: "system", Content: system})
	out = append(out, message{Role: "system", Content: invariantBlock})
	out = append(out, history...)
	return out
}

func detectDay14InvariantConflicts(text string, invariants []day14Invariant) []day14InvariantConflict {
	lower := strings.ToLower(strings.TrimSpace(text))
	if lower == "" || len(invariants) == 0 {
		return nil
	}

	out := make([]day14InvariantConflict, 0)
	for _, inv := range invariants {
		matched := make([]string, 0)
		for _, term := range inv.Forbidden {
			needle := strings.ToLower(strings.TrimSpace(term))
			if needle == "" {
				continue
			}
			if strings.Contains(lower, needle) {
				matched = append(matched, needle)
			}
		}
		// Extra guard for KYC invariant: catch phrasing like "KYC ... отключим".
		if inv.ID == "biz-kyc-required" || strings.Contains(strings.ToLower(inv.Rule), "kyc") {
			if strings.Contains(lower, "kyc") {
				kycConflictMarkers := []string{"без ", "without", "skip", "disable", "отключ"}
				for _, marker := range kycConflictMarkers {
					if strings.Contains(lower, marker) {
						matched = append(matched, "kyc+"+strings.TrimSpace(marker))
					}
				}
			}
		}
		if len(matched) == 0 {
			continue
		}
		out = append(out, day14InvariantConflict{
			Invariant: inv,
			Matched:   matched,
		})
	}
	return out
}

func renderDay14Refusal(conflicts []day14InvariantConflict) string {
	var b strings.Builder
	b.WriteString("STATUS: REFUSE\n")
	b.WriteString("INVARIANT_CHECK:\n")
	b.WriteString("- Обнаружен конфликт запроса с зафиксированными инвариантами.\n")
	b.WriteString("ANSWER:\n")
	b.WriteString("Не могу предложить решение в таком виде, потому что запрос нарушает обязательные ограничения:\n")
	for i, c := range conflicts {
		b.WriteString(fmt.Sprintf("%d) [%s] %s\n", i+1, c.Invariant.ID, c.Invariant.Rule))
		b.WriteString(fmt.Sprintf("   Причина: %s\n", c.Invariant.Rationale))
		if len(c.Matched) > 0 {
			b.WriteString(fmt.Sprintf("   Совпадения: %s\n", strings.Join(c.Matched, ", ")))
		}
		if strings.TrimSpace(c.Invariant.Alternative) != "" {
			b.WriteString(fmt.Sprintf("   Безопасная альтернатива: %s\n", c.Invariant.Alternative))
		}
	}
	return strings.TrimSpace(b.String())
}

func renderDay14GeneratedConflictRefusal(conflicts []day14InvariantConflict) string {
	refusal := renderDay14Refusal(conflicts)
	return refusal + "\n\nПримечание: черновой ответ модели был отклонён, так как выходил за рамки инвариантов."
}

func renderDay14OfflineAnswer(invariants []day14Invariant, userInput string) string {
	var b strings.Builder
	b.WriteString("STATUS: OK\n")
	b.WriteString("INVARIANT_CHECK:\n")
	b.WriteString("- Запрос не конфликтует с инвариантами архитектуры/стека/бизнес-правил.\n")
	b.WriteString("- Решение остаётся в рамках зафиксированных технических решений.\n")
	b.WriteString("ANSWER:\n")
	b.WriteString("Предлагаю план в рамках текущих ограничений:\n")
	b.WriteString("1. Уточнить метрики и узкие места в текущей реализации.\n")
	b.WriteString("2. Оптимизировать Go-сервис и SQL-запросы без смены стека.\n")
	b.WriteString("3. Сохранить бизнес-правила и существующие архитектурные договорённости.\n")
	_ = userInput
	_ = invariants
	return strings.TrimSpace(b.String())
}

func day14EstimatedTokenStats(history []message, answer string, windowSize, cumulativePrompt, cumulativeResponse int) tokenStats {
	requestHistory := keepLastMessages(history, windowSize)
	responseTokens := estimateTextTokens(answer)
	promptTokens := estimateMessagesTokens(requestHistory)
	totalTokens := promptTokens + responseTokens
	return tokenStats{
		EstimatedHistoryTokens:   promptTokens,
		EstimatedRequestTokens:   promptTokens,
		EstimatedResponseTokens:  responseTokens,
		PromptTokens:             promptTokens,
		ResponseTokens:           responseTokens,
		TotalTokens:              totalTokens,
		ConversationTokens:       estimateMessagesTokens(history),
		CumulativePromptTokens:   cumulativePrompt,
		CumulativeResponseTokens: cumulativeResponse,
		CumulativeTotalTokens:    cumulativePrompt + cumulativeResponse,
	}
}

func renderDay14InvariantsForPrompt(invariants []day14Invariant) string {
	var b strings.Builder
	b.WriteString("Обязательные инварианты:\n")
	for i, inv := range invariants {
		b.WriteString(fmt.Sprintf("%d. [%s] (%s) %s\n", i+1, inv.ID, inv.Category, inv.Rule))
		if strings.TrimSpace(inv.Rationale) != "" {
			b.WriteString("   why: " + inv.Rationale + "\n")
		}
		if len(inv.Forbidden) > 0 {
			b.WriteString("   forbidden: " + strings.Join(inv.Forbidden, ", ") + "\n")
		}
		if strings.TrimSpace(inv.Alternative) != "" {
			b.WriteString("   safe alternative: " + inv.Alternative + "\n")
		}
	}
	return strings.TrimSpace(b.String())
}

func renderDay14InvariantsForHumans(invariants []day14Invariant) string {
	if len(invariants) == 0 {
		return "Invariants: (empty)"
	}
	var b strings.Builder
	b.WriteString("Invariants:\n")
	for _, inv := range invariants {
		b.WriteString(fmt.Sprintf("- [%s] %s | %s\n", inv.ID, inv.Category, inv.Rule))
		if len(inv.Forbidden) > 0 {
			b.WriteString("  forbidden: " + strings.Join(inv.Forbidden, ", ") + "\n")
		}
		if strings.TrimSpace(inv.Alternative) != "" {
			b.WriteString("  alternative: " + inv.Alternative + "\n")
		}
	}
	return strings.TrimSpace(b.String())
}

func renderDay14History(history []message) string {
	if len(history) == 0 {
		return "History: (empty)"
	}
	var b strings.Builder
	b.WriteString("History:\n")
	for i, m := range history {
		b.WriteString(fmt.Sprintf("%d. %s: %s\n", i+1, m.Role, m.Content))
	}
	return strings.TrimSpace(b.String())
}

func loadOrCreateDay14Invariants(path string) ([]day14Invariant, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			defaults := defaultDay14Invariants()
			if err := saveDay14Invariants(path, defaults); err != nil {
				return nil, err
			}
			return defaults, nil
		}
		return nil, err
	}
	if len(raw) == 0 {
		defaults := defaultDay14Invariants()
		if err := saveDay14Invariants(path, defaults); err != nil {
			return nil, err
		}
		return defaults, nil
	}

	var payload day14InvariantFile
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, fmt.Errorf("failed to parse invariants file: %w", err)
	}
	items := normalizeDay14Invariants(payload.Invariants)
	if len(items) == 0 {
		items = defaultDay14Invariants()
		if err := saveDay14Invariants(path, items); err != nil {
			return nil, err
		}
	}
	return items, nil
}

func saveDay14Invariants(path string, invariants []day14Invariant) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	payload := day14InvariantFile{
		Version:    1,
		UpdatedAt:  time.Now().UTC().Format(time.RFC3339),
		Invariants: normalizeDay14Invariants(invariants),
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func defaultDay14Invariants() []day14Invariant {
	return []day14Invariant{
		{
			ID:          "arch-monolith",
			Category:    "architecture",
			Rule:        "Сохраняем модульный монолит; не переходим на микросервисную архитектуру.",
			Rationale:   "Команда и сроки не позволяют безопасно перейти на микросервисы.",
			Forbidden:   []string{"microservice", "microservices", "микросервис", "service mesh"},
			Alternative: "Оптимизировать границы модулей внутри монолита.",
		},
		{
			ID:          "stack-go-postgres",
			Category:    "stack",
			Rule:        "Базовый стек зафиксирован: Go + PostgreSQL.",
			Rationale:   "Под это уже подготовлены инфраструктура, экспертиза и SLA.",
			Forbidden:   []string{"node.js", "nodejs", "nestjs", "python", "django", "mongodb", "mongo"},
			Alternative: "Улучшать текущие Go-сервисы и SQL-модель без смены платформы.",
		},
		{
			ID:          "biz-kyc-required",
			Category:    "business-rule",
			Rule:        "Платёжный поток не может работать без KYC-проверки.",
			Rationale:   "Это регуляторное и юридическое требование.",
			Forbidden:   []string{"без kyc", "without kyc", "skip kyc", "disable kyc", "отключить kyc", "отключим kyc", "kyc пока отключ"},
			Alternative: "Сокращать friction KYC, но не убирать саму проверку.",
		},
		{
			ID:          "decision-provider",
			Category:    "technical-decision",
			Rule:        "Для push-уведомлений остаёмся на OneSignal.",
			Rationale:   "Решение уже принято и зафиксировано в текущем roadmap.",
			Forbidden:   []string{"firebase cloud messaging", "fcm", "amazon sns", "webpush provider swap"},
			Alternative: "Тюнить сегментацию и доставку в рамках OneSignal.",
		},
	}
}

func normalizeDay14Invariants(in []day14Invariant) []day14Invariant {
	if len(in) == 0 {
		return nil
	}
	out := make([]day14Invariant, 0, len(in))
	for i, item := range in {
		id := strings.TrimSpace(item.ID)
		if id == "" {
			id = fmt.Sprintf("inv-%02d", i+1)
		}
		category := strings.TrimSpace(item.Category)
		if category == "" {
			category = "general"
		}
		rule := strings.TrimSpace(item.Rule)
		if rule == "" {
			continue
		}
		rationale := strings.TrimSpace(item.Rationale)
		alternative := strings.TrimSpace(item.Alternative)
		forbidden := make([]string, 0, len(item.Forbidden))
		for _, f := range item.Forbidden {
			value := strings.TrimSpace(f)
			if value == "" {
				continue
			}
			forbidden = append(forbidden, value)
		}
		out = append(out, day14Invariant{
			ID:          id,
			Category:    category,
			Rule:        rule,
			Rationale:   rationale,
			Forbidden:   forbidden,
			Alternative: alternative,
		})
	}
	return out
}

func loadDay14History(path string) ([]message, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	if len(raw) == 0 {
		return nil, nil
	}
	var payload day14HistoryFile
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, fmt.Errorf("failed to parse history file: %w", err)
	}
	out := make([]message, 0, len(payload.Messages))
	for _, item := range payload.Messages {
		role := strings.TrimSpace(item.Role)
		content := strings.TrimSpace(item.Content)
		if role == "" || content == "" {
			continue
		}
		out = append(out, message{
			Role:    role,
			Content: content,
		})
	}
	return out, nil
}

func saveDay14History(path string, history []message) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	payload := day14HistoryFile{
		Version:  1,
		Messages: make([]storedMessage, 0, len(history)),
	}
	now := time.Now().UTC().Format(time.RFC3339)
	for _, m := range history {
		role := strings.TrimSpace(m.Role)
		content := strings.TrimSpace(m.Content)
		if role == "" || content == "" {
			continue
		}
		payload.Messages = append(payload.Messages, storedMessage{
			Role:      role,
			Content:   content,
			Timestamp: now,
		})
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func printDay14DemoResult(result day14DemoResult) {
	fmt.Println("=== Day 14: Invariants & State Constraints ===")
	fmt.Printf("model=%s offline=%t invariants=%d\n", result.Model, result.Offline, result.InvariantCount)
	fmt.Printf("history_file=%s invariants_file=%s\n", result.HistoryFile, result.InvariantsFile)
	fmt.Printf("checks: conflict=%t explanation=%t allowed=%t\n",
		result.ConflictCasePassed, result.ExplanationPassed, result.AllowedCasePassed,
	)
	for _, s := range result.Scenarios {
		fmt.Printf("- %s: refused=%t conflicts=%d total_tokens=%d\n",
			s.Name, s.Refused, len(s.Conflicts), s.Tokens.TotalTokens,
		)
	}
}

func writeDay14Report(path string, result day14DemoResult) error {
	var b strings.Builder
	b.WriteString("# Day 14 Results: Invariants and Constraints\n\n")
	b.WriteString(fmt.Sprintf("- model: `%s`\n", result.Model))
	b.WriteString(fmt.Sprintf("- offline mode: `%t`\n", result.Offline))
	b.WriteString(fmt.Sprintf("- invariants file: `%s`\n", result.InvariantsFile))
	b.WriteString(fmt.Sprintf("- history file: `%s`\n", result.HistoryFile))
	b.WriteString(fmt.Sprintf("- invariants count: `%d`\n", result.InvariantCount))
	b.WriteString(fmt.Sprintf("- conflict case passed: `%t`\n", result.ConflictCasePassed))
	b.WriteString(fmt.Sprintf("- refusal explanation passed: `%t`\n", result.ExplanationPassed))
	b.WriteString(fmt.Sprintf("- allowed case passed: `%t`\n\n", result.AllowedCasePassed))

	for _, s := range result.Scenarios {
		b.WriteString("## " + s.Name + "\n\n")
		b.WriteString(fmt.Sprintf("- refused: `%t`\n", s.Refused))
		b.WriteString(fmt.Sprintf("- conflicts: `%s`\n", strings.Join(s.Conflicts, ", ")))
		b.WriteString(fmt.Sprintf("- tokens prompt/response/total: `%d / %d / %d`\n\n", s.Tokens.PromptTokens, s.Tokens.ResponseTokens, s.Tokens.TotalTokens))
		b.WriteString("Request:\n")
		b.WriteString("```text\n" + sanitizeCodeFences(s.Request) + "\n```\n\n")
		b.WriteString("Assistant response:\n")
		b.WriteString("```text\n" + sanitizeCodeFences(s.Response) + "\n```\n\n")
	}

	b.WriteString("Conclusion: invariants are stored separately from dialogue, checked before generation, and conflicts lead to explicit refusal with explanation and safe alternatives.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay14Usage() {
	fmt.Println("Usage: openrouter-cli day14 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -model string            OpenRouter model")
	fmt.Println("  -max-tokens int          Maximum response tokens")
	fmt.Println("  -temperature float       Temperature")
	fmt.Println("  -history-file string     Path to dialogue history JSON")
	fmt.Println("  -invariants-file string  Path to invariants JSON")
	fmt.Println("  -window-size int         Last N dialogue messages sent to model")
	fmt.Println("  -report string           Markdown report path")
	fmt.Println("  -interactive             Run interactive mode")
	fmt.Println("  -show-tokens             Show token stats in interactive mode")
	fmt.Println("  -reset-history           Reset dialogue history before run")
	fmt.Println("  -reset-invariants        Recreate invariants file from defaults")
	fmt.Println("  -offline                 Do not call API for allowed requests")
	fmt.Println("  -help                    Show help")
}
