package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type userProfile struct {
	ID          string
	Name        string
	Role        string
	Style       string
	Format      string
	Constraints []string
}

type personalizedMemoryAgent struct {
	apiKey                   string
	model                    string
	maxTokens                int
	temperature              *float64
	title                    string
	router                   *MemoryRouter
	profile                  userProfile
	cumulativePromptTokens   int
	cumulativeResponseTokens int
}

type day12ProfileResult struct {
	Profile         userProfile
	ShortLayer      string
	WorkingLayer    string
	LongLayer       string
	Question        string
	Answer          string
	PromptTokens    int
	TotalTokens     int
	CostUSD         float64
	AdaptationScore int
	AdaptationMax   int
}

func newPersonalizedMemoryAgent(apiKey, model string, maxTokens int, temperature *float64, router *MemoryRouter, profile userProfile) *personalizedMemoryAgent {
	return &personalizedMemoryAgent{
		apiKey:      apiKey,
		model:       model,
		maxTokens:   maxTokens,
		temperature: temperature,
		title:       "day12-personalized-agent",
		router:      router,
		profile:     profile,
	}
}

func (a *personalizedMemoryAgent) Reply(userInput string) (openRouterResult, error) {
	userInput = strings.TrimSpace(userInput)
	if userInput == "" {
		return openRouterResult{}, fmt.Errorf("empty user input")
	}
	if err := a.router.SaveShortMessage("user", userInput); err != nil {
		return openRouterResult{}, err
	}

	systemMemory, err := a.router.BuildSystemMemoryMessages()
	if err != nil {
		return openRouterResult{}, err
	}
	snapshot, err := a.router.Snapshot()
	if err != nil {
		return openRouterResult{}, err
	}

	requestMessages := make([]message, 0, 2+len(systemMemory)+len(snapshot.Short))
	requestMessages = append(requestMessages,
		message{
			Role: "system",
			Content: "Ты персонализированный ассистент. Всегда автоматически учитывай профиль пользователя, " +
				"его формат ответа и ограничения. Если есть конфликт, приоритет: safety > user constraints > style.",
		},
		message{
			Role:    "system",
			Content: renderProfileSystemBlock(a.profile),
		},
	)
	requestMessages = append(requestMessages, systemMemory...)
	requestMessages = append(requestMessages, snapshot.Short...)

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
		return openRouterResult{}, err
	}

	answer := strings.TrimSpace(result.Answer)
	if err := a.router.SaveShortMessage("assistant", answer); err != nil {
		return openRouterResult{}, err
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

	result.Answer = answer
	result.Tokens = tokenStats{
		PromptTokens:             promptTokens,
		ResponseTokens:           responseTokens,
		TotalTokens:              totalTokens,
		CumulativePromptTokens:   a.cumulativePromptTokens,
		CumulativeResponseTokens: a.cumulativeResponseTokens,
		CumulativeTotalTokens:    a.cumulativePromptTokens + a.cumulativeResponseTokens,
	}
	return result, nil
}

func (a *personalizedMemoryAgent) SetProfile(profile userProfile) error {
	if err := persistUserProfile(a.router, profile); err != nil {
		return err
	}
	a.profile = profile
	return nil
}

func runDay12Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day12", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	model := fs.String("model", getDefaultModel(), "OpenRouter model")
	maxTokens := fs.Int("max-tokens", 260, "Maximum response tokens")
	temperature := fs.Float64("temperature", 0.2, "Temperature")
	memoryRoot := fs.String("memory-root", "/tmp/day12-memory", "Root directory for memory files")
	taskID := fs.String("task-id", "day12-personalization-task", "Working memory task ID")
	shortWindow := fs.Int("short-window", 10, "Recent short-term messages sent to model")
	profileID := fs.String("profile", "all", "Profile id or 'all'")
	reportPath := fs.String("report", "DAY12_RESULTS.md", "Markdown report path")
	interactive := fs.Bool("interactive", false, "Run interactive personalized chat")
	reset := fs.Bool("reset", true, "Reset memory files before run")
	showTokens := fs.Bool("show-tokens", true, "Show token stats in interactive mode")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day12 flags: %w", err)
	}
	if *help {
		printDay12Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day12 arguments: %s", strings.Join(fs.Args(), " "))
	}

	profiles := builtInProfiles()
	selected, err := resolveProfiles(profiles, *profileID, *interactive)
	if err != nil {
		return err
	}

	apiKey := getAPIKey()

	if *interactive {
		profile := selected[0]
		store := newMemoryFileStore(filepath.Join(*memoryRoot, profile.ID))
		if *reset {
			if err := store.reset(); err != nil {
				return err
			}
		}
		router := newMemoryRouter(store, *taskID, *shortWindow)
		if err := seedDay12WorkingMemory(router); err != nil {
			return err
		}
		if err := persistUserProfile(router, profile); err != nil {
			return err
		}
		t := *temperature
		agent := newPersonalizedMemoryAgent(apiKey, *model, *maxTokens, &t, router, profile)
		return runDay12Interactive(agent, router, profiles, *showTokens)
	}

	metadata, metaErr := fetchModelMetadata(apiKey)
	if metaErr != nil {
		fmt.Fprintf(os.Stderr, "warning: failed to fetch model metadata for day12 cost estimate: %v\n", metaErr)
	}

	results := make([]day12ProfileResult, 0, len(selected))
	t := *temperature
	question := "Подготовь план запуска функции push-уведомлений для MVP доставки: приоритеты, риски и формат отчета команде."
	for _, profile := range selected {
		store := newMemoryFileStore(filepath.Join(*memoryRoot, profile.ID))
		if *reset {
			if err := store.reset(); err != nil {
				return err
			}
		}
		router := newMemoryRouter(store, *taskID, *shortWindow)
		if err := seedDay12WorkingMemory(router); err != nil {
			return err
		}
		if err := persistUserProfile(router, profile); err != nil {
			return err
		}

		agent := newPersonalizedMemoryAgent(apiKey, *model, *maxTokens, &t, router, profile)
		response, err := agent.Reply(question)
		if err != nil {
			return err
		}

		shortLayer, err := router.Render("short")
		if err != nil {
			return err
		}
		workingLayer, err := router.Render("working")
		if err != nil {
			return err
		}
		longLayer, err := router.Render("long")
		if err != nil {
			return err
		}
		score, maxScore := scoreProfileAdaptation(profile, response.Answer)

		item := day12ProfileResult{
			Profile:         profile,
			Question:        question,
			Answer:          strings.TrimSpace(response.Answer),
			ShortLayer:      shortLayer,
			WorkingLayer:    workingLayer,
			LongLayer:       longLayer,
			PromptTokens:    response.Tokens.PromptTokens,
			TotalTokens:     response.Tokens.TotalTokens,
			AdaptationScore: score,
			AdaptationMax:   maxScore,
		}
		if meta, ok := resolveModelMeta(metadata, response.Model, *model); ok {
			if cost, known := estimateCostUSD(response.Usage, meta); known {
				item.CostUSD = cost
			}
		}
		results = append(results, item)
	}

	printDay12Results(*model, results)
	if err := writeDay12Report(*reportPath, *model, results); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)
	return nil
}

func runDay12Interactive(agent *personalizedMemoryAgent, router *MemoryRouter, profiles map[string]userProfile, showTokens bool) error {
	fmt.Println("Day12 interactive mode. Type /exit to quit.")
	fmt.Println("Commands:")
	fmt.Println("  /profile show")
	fmt.Println("  /profile list")
	fmt.Println("  /profile set <id>")
	fmt.Println("  /mem save|show|clear ...")
	fmt.Println("  /task show|set <task_id>")

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
			if handled, err := handleDay12Command(line, agent, router, profiles); handled {
				if err != nil {
					fmt.Fprintf(os.Stderr, "command error: %v\n", err)
				}
				continue
			}
		}

		result, err := agent.Reply(line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "agent error: %v\n", err)
			continue
		}
		fmt.Printf("agent> %s\n\n", result.Answer)
		if showTokens {
			printTokenStats(result.Tokens)
		}
	}
}

func handleDay12Command(line string, agent *personalizedMemoryAgent, router *MemoryRouter, profiles map[string]userProfile) (bool, error) {
	fields := strings.Fields(strings.TrimSpace(line))
	if len(fields) == 0 {
		return false, nil
	}
	if strings.ToLower(fields[0]) == "/profile" {
		if len(fields) < 2 {
			return true, fmt.Errorf("usage: /profile show|list|set <id>")
		}
		switch strings.ToLower(fields[1]) {
		case "show":
			fmt.Printf("profile> %s\n\n", renderProfileSystemBlock(agent.profile))
			return true, nil
		case "list":
			keys := make([]string, 0, len(profiles))
			for key := range profiles {
				keys = append(keys, key)
			}
			sortStrings(keys)
			fmt.Println("profile> available:")
			for _, key := range keys {
				fmt.Printf("- %s\n", key)
			}
			fmt.Println()
			return true, nil
		case "set":
			if len(fields) < 3 {
				return true, fmt.Errorf("usage: /profile set <id>")
			}
			profile, ok := profiles[strings.ToLower(strings.TrimSpace(fields[2]))]
			if !ok {
				return true, fmt.Errorf("unknown profile: %s", fields[2])
			}
			if err := agent.SetProfile(profile); err != nil {
				return true, err
			}
			fmt.Printf("profile> active profile: %s\n\n", profile.ID)
			return true, nil
		default:
			return true, fmt.Errorf("unknown profile action: %s", fields[1])
		}
	}
	return handleDay11Command(router, line)
}

func seedDay12WorkingMemory(router *MemoryRouter) error {
	seed := []struct {
		Layer MemoryLayer
		Kind  string
		Key   string
		Value string
	}{
		{Layer: MemoryLayerWorking, Kind: "goal", Value: "Запустить push-уведомления в MVP доставки"},
		{Layer: MemoryLayerWorking, Kind: "constraint", Value: "Срок 2 недели"},
		{Layer: MemoryLayerWorking, Kind: "constraint", Value: "Минимизировать риск спама"},
		{Layer: MemoryLayerWorking, Kind: "decision", Value: "Провайдер OneSignal"},
		{Layer: MemoryLayerWorking, Kind: "value", Key: "kpi", Value: "CTR push > 6%"},
		{Layer: MemoryLayerWorking, Kind: "value", Key: "reporting", Value: "еженедельный отчет PM"},
	}
	for _, item := range seed {
		if err := router.SaveExplicit(item.Layer, item.Kind, item.Key, item.Value); err != nil {
			return err
		}
	}
	return nil
}

func persistUserProfile(router *MemoryRouter, profile userProfile) error {
	pairs := []struct {
		Kind  string
		Key   string
		Value string
	}{
		{Kind: "profile", Key: "id", Value: profile.ID},
		{Kind: "profile", Key: "name", Value: profile.Name},
		{Kind: "profile", Key: "role", Value: profile.Role},
		{Kind: "preference", Key: "style", Value: profile.Style},
		{Kind: "preference", Key: "format", Value: profile.Format},
	}
	for _, item := range pairs {
		if strings.TrimSpace(item.Value) == "" {
			continue
		}
		if err := router.SaveExplicit(MemoryLayerLong, item.Kind, item.Key, item.Value); err != nil {
			return err
		}
	}
	for i, c := range profile.Constraints {
		value := strings.TrimSpace(c)
		if value == "" {
			continue
		}
		key := fmt.Sprintf("constraint_%02d", i+1)
		if err := router.SaveExplicit(MemoryLayerLong, "knowledge", key, value); err != nil {
			return err
		}
	}
	return nil
}

func builtInProfiles() map[string]userProfile {
	return map[string]userProfile{
		"founder-brief": {
			ID:     "founder-brief",
			Name:   "Ivan",
			Role:   "Startup founder",
			Style:  "Очень кратко, деловой тон, без лишней воды",
			Format: "Список из 5-7 пунктов",
			Constraints: []string{
				"Ответ не длиннее 90 слов",
				"Сначала приоритеты, потом риски",
			},
		},
		"pm-table": {
			ID:     "pm-table",
			Name:   "Ivan",
			Role:   "Product manager",
			Style:  "Структурно и нейтрально",
			Format: "Таблица Markdown: Приоритет | Действие | Риск | Митигейшн",
			Constraints: []string{
				"Добавь KPI и владельца шага",
				"Не используй художественные формулировки",
			},
		},
		"dev-json": {
			ID:     "dev-json",
			Name:   "Ivan",
			Role:   "Backend engineer",
			Style:  "Технический и конкретный",
			Format: "Только JSON-объект с полями priorities, risks, mitigations, rollout",
			Constraints: []string{
				"Никакого текста вне JSON",
				"Явно укажи monitoring и rollback",
			},
		},
	}
}

func resolveProfiles(profiles map[string]userProfile, raw string, interactive bool) ([]userProfile, error) {
	raw = strings.ToLower(strings.TrimSpace(raw))
	if raw == "" && interactive {
		raw = "founder-brief"
	}
	if raw == "" || raw == "all" {
		list := []string{"founder-brief", "pm-table", "dev-json"}
		out := make([]userProfile, 0, len(list))
		for _, key := range list {
			out = append(out, profiles[key])
		}
		if interactive && len(out) > 0 {
			return []userProfile{out[0]}, nil
		}
		return out, nil
	}
	profile, ok := profiles[raw]
	if !ok {
		return nil, fmt.Errorf("unknown profile: %s", raw)
	}
	return []userProfile{profile}, nil
}

func renderProfileSystemBlock(profile userProfile) string {
	var b strings.Builder
	b.WriteString("User profile:\n")
	b.WriteString("- id: " + profile.ID + "\n")
	b.WriteString("- name: " + profile.Name + "\n")
	b.WriteString("- role: " + profile.Role + "\n")
	b.WriteString("- style: " + profile.Style + "\n")
	b.WriteString("- response format: " + profile.Format + "\n")
	if len(profile.Constraints) > 0 {
		b.WriteString("- constraints:\n")
		for _, item := range profile.Constraints {
			b.WriteString("  - " + item + "\n")
		}
	}
	b.WriteString("Always adapt automatically to this profile.")
	return strings.TrimSpace(b.String())
}

func scoreProfileAdaptation(profile userProfile, answer string) (int, int) {
	text := strings.TrimSpace(answer)
	lower := strings.ToLower(text)
	score := 0
	maxScore := 4
	if profile.ID == "dev-json" {
		maxScore = 5
	}

	switch profile.ID {
	case "dev-json":
		if strings.Contains(text, "{") && strings.Contains(text, "}") {
			score++
		}
		if !strings.Contains(lower, "```") {
			score++
		}
		if strings.Contains(lower, "monitor") || strings.Contains(lower, "rollback") {
			score++
		}
	case "pm-table":
		if strings.Contains(text, "|") {
			score++
		}
		if strings.Contains(lower, "kpi") {
			score++
		}
	case "founder-brief":
		if countWords(text) <= 95 {
			score++
		}
		if strings.Contains(text, "1.") || strings.Contains(text, "- ") {
			score++
		}
	}

	if strings.Contains(lower, "onesignal") {
		score++
	}
	if strings.Contains(lower, "риск") || strings.Contains(lower, "risk") {
		score++
	}
	return score, maxScore
}

func countWords(text string) int {
	return len(strings.Fields(strings.TrimSpace(text)))
}

func printDay12Results(model string, results []day12ProfileResult) {
	fmt.Println("=== Day 12: Personalization ===")
	fmt.Printf("Модель: %s\n\n", model)
	for _, item := range results {
		fmt.Printf("--- profile=%s ---\n", item.Profile.ID)
		fmt.Printf("adaptation=%d/%d tokens(prompt=%d total=%d) cost=$%.6f\n",
			item.AdaptationScore, item.AdaptationMax, item.PromptTokens, item.TotalTokens, item.CostUSD,
		)
	}
}

func writeDay12Report(path, model string, results []day12ProfileResult) error {
	var b strings.Builder
	b.WriteString("# Day 12 Results: Personalized Assistant\n\n")
	b.WriteString("Model: `" + model + "`\n\n")
	b.WriteString("| Profile | Adaptation | Prompt Tokens | Total Tokens | Cost USD |\n")
	b.WriteString("|---|---:|---:|---:|---:|\n")
	for _, item := range results {
		b.WriteString(fmt.Sprintf(
			"| %s | %d/%d | %d | %d | %.6f |\n",
			item.Profile.ID, item.AdaptationScore, item.AdaptationMax, item.PromptTokens, item.TotalTokens, item.CostUSD,
		))
	}
	b.WriteString("\n")

	for _, item := range results {
		b.WriteString("## Profile: " + item.Profile.ID + "\n\n")
		b.WriteString("### Profile Settings\n")
		b.WriteString("```text\n" + sanitizeCodeFences(renderProfileSystemBlock(item.Profile)) + "\n```\n\n")
		b.WriteString("### Stored Memory Layers\n")
		b.WriteString("Short-term:\n")
		b.WriteString("```text\n" + sanitizeCodeFences(item.ShortLayer) + "\n```\n\n")
		b.WriteString("Working:\n")
		b.WriteString("```text\n" + sanitizeCodeFences(item.WorkingLayer) + "\n```\n\n")
		b.WriteString("Long-term:\n")
		b.WriteString("```text\n" + sanitizeCodeFences(item.LongLayer) + "\n```\n\n")
		b.WriteString("Question:\n")
		b.WriteString("```text\n" + sanitizeCodeFences(item.Question) + "\n```\n\n")
		b.WriteString("Answer:\n")
		b.WriteString("```text\n" + sanitizeCodeFences(item.Answer) + "\n```\n\n")
		b.WriteString(fmt.Sprintf("- adaptation score: `%d/%d`\n", item.AdaptationScore, item.AdaptationMax))
		b.WriteString(fmt.Sprintf("- prompt tokens: `%d`\n", item.PromptTokens))
		b.WriteString(fmt.Sprintf("- total tokens: `%d`\n", item.TotalTokens))
		b.WriteString(fmt.Sprintf("- cost: `$%.6f`\n\n", item.CostUSD))
	}
	b.WriteString("Conclusion: profile preferences are injected automatically in every request and lead to different answer styles and formats.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay12Usage() {
	fmt.Println("Usage: openrouter-cli day12 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -model string         OpenRouter model")
	fmt.Println("  -max-tokens int       Maximum response tokens")
	fmt.Println("  -temperature float    Temperature")
	fmt.Println("  -memory-root string   Root dir for memory files")
	fmt.Println("  -task-id string       Working memory task ID")
	fmt.Println("  -short-window int     Recent short-term messages sent to model")
	fmt.Println("  -profile string       Profile id or 'all'")
	fmt.Println("  -report string        Markdown report path")
	fmt.Println("  -interactive          Run interactive personalized mode")
	fmt.Println("  -show-tokens          Show token stats in interactive mode")
	fmt.Println("  -reset                Reset memory files before run")
	fmt.Println("  -help                 Show help")
}

func sortStrings(values []string) {
	if len(values) < 2 {
		return
	}
	for i := 0; i < len(values); i++ {
		for j := i + 1; j < len(values); j++ {
			if values[j] < values[i] {
				values[i], values[j] = values[j], values[i]
			}
		}
	}
}
