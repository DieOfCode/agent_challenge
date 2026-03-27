package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type day9Turn struct {
	Turn        int
	Prompt      string
	Answer      string
	PromptTok   int
	ResponseTok int
	TotalTok    int
	TurnCostUSD float64
}

type day9ScenarioResult struct {
	Name             string
	Compressed       bool
	KeepLast         int
	SummaryEvery     int
	Turns            []day9Turn
	PromptTokens     int
	ResponseTokens   int
	TotalTokens      int
	CostUSD          float64
	QualityScore     int
	QualityMax       int
	QualityDetails   []string
	FinalAnswer      string
	CompressionStats string
}

func runDay9Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day9", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	model := fs.String("model", getDefaultModel(), "OpenRouter model")
	maxTokens := fs.Int("max-tokens", 220, "Maximum response tokens per turn")
	temperature := fs.Float64("temperature", 0.2, "Temperature")
	keepLast := fs.Int("keep-last", 6, "How many latest messages are kept without compression")
	summaryEvery := fs.Int("summary-every", 10, "How many older messages are compressed into one summary block")
	historyDir := fs.String("history-dir", "/tmp/day9-agent-context", "Directory for run history files")
	reportPath := fs.String("report", "DAY9_RESULTS.md", "Markdown report path")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day9 flags: %w", err)
	}
	if *help {
		printDay9Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day9 arguments: %s", strings.Join(fs.Args(), " "))
	}

	if err := os.MkdirAll(*historyDir, 0o755); err != nil {
		return fmt.Errorf("failed to create day9 history dir: %w", err)
	}

	apiKey := getAPIKey()
	metadata, metaErr := fetchModelMetadata(apiKey)
	if metaErr != nil {
		fmt.Fprintf(os.Stderr, "warning: failed to fetch metadata for day9 cost estimate: %v\n", metaErr)
	}

	prompts := day9Prompts()

	baseline, err := runDay9Scenario(day9RunConfig{
		Name:      "without_compression",
		Model:     *model,
		APIKey:    apiKey,
		MaxTokens: *maxTokens,
		Temp:      *temperature,
		History:   filepath.Join(*historyDir, "baseline.json"),
		Prompts:   prompts,
		Metadata:  metadata,
		Compression: CompressionConfig{
			Enabled: false,
		},
	})
	if err != nil {
		return err
	}

	compressed, err := runDay9Scenario(day9RunConfig{
		Name:      "with_compression",
		Model:     *model,
		APIKey:    apiKey,
		MaxTokens: *maxTokens,
		Temp:      *temperature,
		History:   filepath.Join(*historyDir, "compressed.json"),
		Prompts:   prompts,
		Metadata:  metadata,
		Compression: CompressionConfig{
			Enabled:       true,
			KeepLastN:     *keepLast,
			SummaryEveryN: *summaryEvery,
		},
	})
	if err != nil {
		return err
	}

	printDay9Results(*model, baseline, compressed)
	if err := writeDay9Report(*reportPath, *model, baseline, compressed); err != nil {
		return fmt.Errorf("failed to write day9 report: %w", err)
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)
	return nil
}

type day9RunConfig struct {
	Name        string
	Model       string
	APIKey      string
	MaxTokens   int
	Temp        float64
	History     string
	Prompts     []string
	Metadata    map[string]modelInfo
	Compression CompressionConfig
}

func runDay9Scenario(cfg day9RunConfig) (day9ScenarioResult, error) {
	store := NewJSONHistoryStore(cfg.History)
	if err := store.Reset(); err != nil {
		return day9ScenarioResult{}, fmt.Errorf("failed to reset %s history: %w", cfg.Name, err)
	}

	t := cfg.Temp
	agent, err := NewLLMAgent(LLMAgentConfig{
		APIKey:       cfg.APIKey,
		Model:        cfg.Model,
		MaxTokens:    cfg.MaxTokens,
		Temperature:  &t,
		SystemPrompt: "Ты ассистент-память. Сохраняй факты из диалога и точно возвращай их по запросу.",
		Title:        "day9-context-compression",
		HistoryStore: store,
		Compression:  cfg.Compression,
	})
	if err != nil {
		return day9ScenarioResult{}, err
	}

	result := day9ScenarioResult{
		Name:         cfg.Name,
		Compressed:   cfg.Compression.Enabled,
		KeepLast:     cfg.Compression.KeepLastN,
		SummaryEvery: cfg.Compression.SummaryEveryN,
		Turns:        make([]day9Turn, 0, len(cfg.Prompts)),
	}

	for i, prompt := range cfg.Prompts {
		reply, err := agent.Reply(prompt)
		if err != nil {
			return day9ScenarioResult{}, fmt.Errorf("%s failed on turn %d: %w", cfg.Name, i+1, err)
		}

		turn := day9Turn{
			Turn:        i + 1,
			Prompt:      prompt,
			Answer:      strings.TrimSpace(reply.Answer),
			PromptTok:   reply.Tokens.PromptTokens,
			ResponseTok: reply.Tokens.ResponseTokens,
			TotalTok:    reply.Tokens.TotalTokens,
		}

		if meta, ok := resolveModelMeta(cfg.Metadata, reply.Model, cfg.Model); ok {
			if cost, known := estimateCostUSD(reply.Usage, meta); known {
				turn.TurnCostUSD = cost
				result.CostUSD += cost
			}
		}

		result.PromptTokens += turn.PromptTok
		result.ResponseTokens += turn.ResponseTok
		result.TotalTokens += turn.TotalTok
		result.Turns = append(result.Turns, turn)

		if i == len(cfg.Prompts)-1 {
			result.FinalAnswer = turn.Answer
			score, maxScore, details := day9QualityScore(turn.Answer)
			result.QualityScore = score
			result.QualityMax = maxScore
			result.QualityDetails = details
		}
	}

	if result.Compressed {
		result.CompressionStats = fmt.Sprintf("enabled keep_last=%d summary_every=%d", result.KeepLast, result.SummaryEvery)
	} else {
		result.CompressionStats = "disabled"
	}

	return result, nil
}

func day9Prompts() []string {
	prompts := []string{
		"Запомни: имя клиента Иван.",
		"Запомни: любимый стек клиента Go и PostgreSQL.",
		"Запомни: проект называется Atlas.",
		"Запомни: дедлайн проекта 15 апреля 2026.",
		"Запомни: сервер eu-west-1, порт 7443.",
		"Запомни: бюджет 12000 USD, контакт Anna.",
	}

	for i := 1; i <= 8; i++ {
		prompts = append(prompts, fmt.Sprintf(
			"Служебная заметка %d: %s",
			i,
			strings.Repeat("обсудили логирование, ретраи, кэш, мониторинг, очереди и SLA; ", 16),
		))
	}

	prompts = append(prompts,
		`Верни JSON с полями name, stack, project, deadline, region, port, budget_usd, contact. Используй только факты из диалога.`,
	)
	return prompts
}

func day9QualityScore(answer string) (int, int, []string) {
	text := strings.ToLower(answer)

	checks := []struct {
		Label    string
		Patterns []string
	}{
		{Label: "name", Patterns: []string{"иван", "ivan"}},
		{Label: "stack-go", Patterns: []string{"go"}},
		{Label: "stack-postgresql", Patterns: []string{"postgresql", "postgres"}},
		{Label: "project", Patterns: []string{"atlas"}},
		{Label: "deadline", Patterns: []string{"15 апреля 2026", "2026-04-15", "15.04.2026"}},
		{Label: "region", Patterns: []string{"eu-west-1"}},
		{Label: "port", Patterns: []string{"7443"}},
		{Label: "budget", Patterns: []string{"12000"}},
		{Label: "contact", Patterns: []string{"anna", "анна"}},
	}

	score := 0
	details := make([]string, 0, len(checks))
	for _, c := range checks {
		matched := false
		for _, pattern := range c.Patterns {
			if strings.Contains(text, strings.ToLower(pattern)) {
				matched = true
				break
			}
		}
		if matched {
			score++
			details = append(details, fmt.Sprintf("%s: ok", c.Label))
		} else {
			details = append(details, fmt.Sprintf("%s: miss", c.Label))
		}
	}

	return score, len(checks), details
}

func printDay9Results(model string, baseline, compressed day9ScenarioResult) {
	fmt.Println("=== День 9: Сжатие истории ===")
	fmt.Printf("Модель: %s\n\n", model)

	printOneDay9Scenario(baseline)
	printOneDay9Scenario(compressed)

	promptSaved := baseline.PromptTokens - compressed.PromptTokens
	totalSaved := baseline.TotalTokens - compressed.TotalTokens

	fmt.Println("=== Сравнение ===")
	fmt.Printf("Качество без сжатия: %d/%d\n", baseline.QualityScore, baseline.QualityMax)
	fmt.Printf("Качество со сжатием: %d/%d\n", compressed.QualityScore, compressed.QualityMax)
	fmt.Printf("Prompt tokens: without=%d with=%d saved=%d\n", baseline.PromptTokens, compressed.PromptTokens, promptSaved)
	fmt.Printf("Total tokens:  without=%d with=%d saved=%d\n", baseline.TotalTokens, compressed.TotalTokens, totalSaved)
	fmt.Printf("Cost (USD):    without=$%.6f with=$%.6f\n", baseline.CostUSD, compressed.CostUSD)
}

func printOneDay9Scenario(s day9ScenarioResult) {
	fmt.Printf("--- %s ---\n", s.Name)
	fmt.Printf("compression: %s\n", s.CompressionStats)
	fmt.Printf("quality: %d/%d\n", s.QualityScore, s.QualityMax)
	fmt.Printf("tokens: prompt=%d response=%d total=%d\n", s.PromptTokens, s.ResponseTokens, s.TotalTokens)
	if s.CostUSD > 0 {
		fmt.Printf("cost: $%.6f\n", s.CostUSD)
	}
	fmt.Println()
}

func writeDay9Report(path, model string, baseline, compressed day9ScenarioResult) error {
	var b strings.Builder
	b.WriteString("# Day 9 Results: Context Compression\n\n")
	b.WriteString("Model: `" + model + "`\n\n")

	writeDay9ReportScenario(&b, baseline)
	writeDay9ReportScenario(&b, compressed)

	promptSaved := baseline.PromptTokens - compressed.PromptTokens
	totalSaved := baseline.TotalTokens - compressed.TotalTokens

	b.WriteString("## Comparison\n")
	b.WriteString(fmt.Sprintf("- quality without compression: `%d/%d`\n", baseline.QualityScore, baseline.QualityMax))
	b.WriteString(fmt.Sprintf("- quality with compression: `%d/%d`\n", compressed.QualityScore, compressed.QualityMax))
	b.WriteString(fmt.Sprintf("- prompt tokens without: `%d`\n", baseline.PromptTokens))
	b.WriteString(fmt.Sprintf("- prompt tokens with: `%d`\n", compressed.PromptTokens))
	b.WriteString(fmt.Sprintf("- prompt tokens saved: `%d`\n", promptSaved))
	b.WriteString(fmt.Sprintf("- total tokens without: `%d`\n", baseline.TotalTokens))
	b.WriteString(fmt.Sprintf("- total tokens with: `%d`\n", compressed.TotalTokens))
	b.WriteString(fmt.Sprintf("- total tokens saved: `%d`\n", totalSaved))
	b.WriteString(fmt.Sprintf("- cost without: `$%.6f`\n", baseline.CostUSD))
	b.WriteString(fmt.Sprintf("- cost with: `$%.6f`\n", compressed.CostUSD))
	b.WriteString("\n")
	b.WriteString("Conclusion: keeping recent turns raw and summarizing older chunks reduces token usage while preserving most task-critical facts.\n")

	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func writeDay9ReportScenario(b *strings.Builder, s day9ScenarioResult) {
	b.WriteString("## " + s.Name + "\n\n")
	b.WriteString("- compression: `" + s.CompressionStats + "`\n")
	b.WriteString(fmt.Sprintf("- quality: `%d/%d`\n", s.QualityScore, s.QualityMax))
	b.WriteString(fmt.Sprintf("- tokens prompt/response/total: `%d / %d / %d`\n", s.PromptTokens, s.ResponseTokens, s.TotalTokens))
	b.WriteString(fmt.Sprintf("- cost: `$%.6f`\n", s.CostUSD))
	b.WriteString("- quality details: `" + strings.Join(s.QualityDetails, ", ") + "`\n\n")
	b.WriteString("Final answer sample:\n")
	b.WriteString("```text\n" + sanitizeCodeFences(strings.TrimSpace(s.FinalAnswer)) + "\n```\n\n")
}

func sanitizeCodeFences(text string) string {
	if text == "" {
		return text
	}
	return strings.ReplaceAll(text, "```", "'''")
}

func printDay9Usage() {
	fmt.Println("Usage: openrouter-cli day9 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -model string         OpenRouter model")
	fmt.Println("  -max-tokens int       Max completion tokens per turn")
	fmt.Println("  -temperature float    Temperature")
	fmt.Println("  -keep-last int        Keep latest N messages as-is")
	fmt.Println("  -summary-every int    Compress each N older messages into summary")
	fmt.Println("  -history-dir string   Directory for scenario history files")
	fmt.Println("  -report string        Markdown report path")
	fmt.Println("  -help                 Show help")
}
