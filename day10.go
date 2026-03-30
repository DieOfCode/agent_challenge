package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type day10StrategyResult struct {
	Name                 string
	Strategy             ContextStrategy
	WindowSize           int
	PromptTokens         int
	ResponseTokens       int
	TotalTokens          int
	CostUSD              float64
	QualityScore         int
	QualityMax           int
	StabilityScore       int
	StabilityMax         int
	BranchIsolationScore int
	BranchIsolationMax   int
	UsabilityNote        string
	FinalAnswer          string
	StabilityAnswer      string
	BranchingNotes       []string
}

type day10RunConfig struct {
	Name       string
	Strategy   ContextStrategy
	Model      string
	APIKey     string
	MaxTokens  int
	Temp       float64
	WindowSize int
	History    string
	Metadata   map[string]modelInfo
	Progress   bool
}

func runDay10Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day10", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	model := fs.String("model", getDefaultModel(), "OpenRouter model")
	maxTokens := fs.Int("max-tokens", 220, "Maximum response tokens per turn")
	temperature := fs.Float64("temperature", 0.2, "Temperature")
	windowSize := fs.Int("window-size", 8, "Window size for sliding/facts strategies")
	historyDir := fs.String("history-dir", "/tmp/day10-agent-context", "Directory for scenario history files")
	reportPath := fs.String("report", "DAY10_RESULTS.md", "Markdown report path")
	progress := fs.Bool("progress", true, "Show live progress")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day10 flags: %w", err)
	}
	if *help {
		printDay10Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day10 arguments: %s", strings.Join(fs.Args(), " "))
	}

	if err := os.MkdirAll(*historyDir, 0o755); err != nil {
		return fmt.Errorf("failed to create day10 history dir: %w", err)
	}

	apiKey := getAPIKey()
	metadata, metaErr := fetchModelMetadata(apiKey)
	if metaErr != nil {
		fmt.Fprintf(os.Stderr, "warning: failed to fetch metadata for day10 cost estimate: %v\n", metaErr)
	}

	sliding, err := runDay10LinearStrategy(day10RunConfig{
		Name:       "sliding_window",
		Strategy:   ContextStrategySliding,
		Model:      *model,
		APIKey:     apiKey,
		MaxTokens:  *maxTokens,
		Temp:       *temperature,
		WindowSize: *windowSize,
		History:    filepath.Join(*historyDir, "sliding.json"),
		Metadata:   metadata,
		Progress:   *progress,
	})
	if err != nil {
		return err
	}

	facts, err := runDay10LinearStrategy(day10RunConfig{
		Name:       "sticky_facts",
		Strategy:   ContextStrategyFacts,
		Model:      *model,
		APIKey:     apiKey,
		MaxTokens:  *maxTokens,
		Temp:       *temperature,
		WindowSize: *windowSize,
		History:    filepath.Join(*historyDir, "facts.json"),
		Metadata:   metadata,
		Progress:   *progress,
	})
	if err != nil {
		return err
	}

	branching, err := runDay10BranchingStrategy(day10RunConfig{
		Name:       "branching",
		Strategy:   ContextStrategyBranching,
		Model:      *model,
		APIKey:     apiKey,
		MaxTokens:  *maxTokens,
		Temp:       *temperature,
		WindowSize: *windowSize,
		History:    filepath.Join(*historyDir, "branching.json"),
		Metadata:   metadata,
		Progress:   *progress,
	})
	if err != nil {
		return err
	}

	results := []day10StrategyResult{sliding, facts, branching}
	printDay10Results(*model, results)
	if err := writeDay10Report(*reportPath, *model, results); err != nil {
		return fmt.Errorf("failed to write day10 report: %w", err)
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)
	return nil
}

func runDay10LinearStrategy(cfg day10RunConfig) (day10StrategyResult, error) {
	store := NewJSONHistoryStore(cfg.History)
	if err := store.Reset(); err != nil {
		return day10StrategyResult{}, fmt.Errorf("failed to reset %s history: %w", cfg.Name, err)
	}

	t := cfg.Temp
	agent, err := NewLLMAgent(LLMAgentConfig{
		APIKey:       cfg.APIKey,
		Model:        cfg.Model,
		MaxTokens:    cfg.MaxTokens,
		Temperature:  &t,
		SystemPrompt: "Ты помощник по сбору ТЗ. Отвечай точно и не теряй договоренности.",
		Title:        "day10-context-strategy-" + cfg.Name,
		HistoryStore: store,
		Strategy:     cfg.Strategy,
		WindowSize:   cfg.WindowSize,
	})
	if err != nil {
		return day10StrategyResult{}, err
	}

	result := day10StrategyResult{
		Name:          cfg.Name,
		Strategy:      cfg.Strategy,
		WindowSize:    cfg.WindowSize,
		UsabilityNote: usabilityNote(cfg.Strategy),
	}

	basePrompts := day10BasePrompts()
	for i, prompt := range basePrompts {
		if cfg.Progress {
			fmt.Printf("[%s] base turn %d/%d\n", cfg.Name, i+1, len(basePrompts))
		}
		reply, err := agent.Reply(prompt)
		if err != nil {
			return day10StrategyResult{}, fmt.Errorf("%s base turn %d failed: %w", cfg.Name, i+1, err)
		}
		day10Accumulate(&result, reply, cfg.Metadata, cfg.Model)
	}

	stabilityQuestion := "Назови 4 критичных пункта: бюджет, срок, SLA и push-провайдер."
	stabilityReply, err := agent.Reply(stabilityQuestion)
	if err != nil {
		return day10StrategyResult{}, fmt.Errorf("%s stability turn failed: %w", cfg.Name, err)
	}
	day10Accumulate(&result, stabilityReply, cfg.Metadata, cfg.Model)
	result.StabilityAnswer = strings.TrimSpace(stabilityReply.Answer)
	result.StabilityScore, result.StabilityMax = day10StabilityScore(result.StabilityAnswer)

	finalQuestion := "Собери финальное ТЗ в JSON с полями goal, budget_usd, deadline_weeks, backend, database, auth, push_provider, sla, reports."
	finalReply, err := agent.Reply(finalQuestion)
	if err != nil {
		return day10StrategyResult{}, fmt.Errorf("%s final turn failed: %w", cfg.Name, err)
	}
	day10Accumulate(&result, finalReply, cfg.Metadata, cfg.Model)
	result.FinalAnswer = strings.TrimSpace(finalReply.Answer)
	result.QualityScore, result.QualityMax = day10QualityScore(result.FinalAnswer)

	return result, nil
}

func runDay10BranchingStrategy(cfg day10RunConfig) (day10StrategyResult, error) {
	store := NewJSONHistoryStore(cfg.History)
	if err := store.Reset(); err != nil {
		return day10StrategyResult{}, fmt.Errorf("failed to reset branching history: %w", err)
	}

	t := cfg.Temp
	agent, err := NewLLMAgent(LLMAgentConfig{
		APIKey:       cfg.APIKey,
		Model:        cfg.Model,
		MaxTokens:    cfg.MaxTokens,
		Temperature:  &t,
		SystemPrompt: "Ты помощник по сбору ТЗ. Учитывай активную ветку диалога.",
		Title:        "day10-context-strategy-branching",
		HistoryStore: store,
		Strategy:     ContextStrategyBranching,
		WindowSize:   cfg.WindowSize,
	})
	if err != nil {
		return day10StrategyResult{}, err
	}

	result := day10StrategyResult{
		Name:          cfg.Name,
		Strategy:      ContextStrategyBranching,
		WindowSize:    cfg.WindowSize,
		UsabilityNote: usabilityNote(ContextStrategyBranching),
	}

	basePrompts := day10BasePrompts()
	for i, prompt := range basePrompts {
		if cfg.Progress {
			fmt.Printf("[branching] base turn %d/%d (branch=%s)\n", i+1, len(basePrompts), agent.ActiveBranch())
		}
		reply, err := agent.Reply(prompt)
		if err != nil {
			return day10StrategyResult{}, fmt.Errorf("branching base turn %d failed: %w", i+1, err)
		}
		day10Accumulate(&result, reply, cfg.Metadata, cfg.Model)
		if i == 5 {
			if err := agent.SaveCheckpoint("spec_v1"); err != nil {
				return day10StrategyResult{}, fmt.Errorf("failed to save checkpoint: %w", err)
			}
		}
	}

	stabilityQuestion := "Назови 4 критичных пункта: бюджет, срок, SLA и push-провайдер."
	stabilityReply, err := agent.Reply(stabilityQuestion)
	if err != nil {
		return day10StrategyResult{}, fmt.Errorf("branching stability turn failed: %w", err)
	}
	day10Accumulate(&result, stabilityReply, cfg.Metadata, cfg.Model)
	result.StabilityAnswer = strings.TrimSpace(stabilityReply.Answer)
	result.StabilityScore, result.StabilityMax = day10StabilityScore(result.StabilityAnswer)

	finalQuestion := "Собери финальное ТЗ в JSON с полями goal, budget_usd, deadline_weeks, backend, database, auth, push_provider, sla, reports."
	finalReply, err := agent.Reply(finalQuestion)
	if err != nil {
		return day10StrategyResult{}, fmt.Errorf("branching final turn failed: %w", err)
	}
	day10Accumulate(&result, finalReply, cfg.Metadata, cfg.Model)
	result.FinalAnswer = strings.TrimSpace(finalReply.Answer)
	result.QualityScore, result.QualityMax = day10QualityScore(result.FinalAnswer)

	if err := agent.CreateBranch("option_a", "spec_v1"); err != nil {
		return day10StrategyResult{}, fmt.Errorf("failed to create option_a: %w", err)
	}
	if err := agent.CreateBranch("option_b", "spec_v1"); err != nil {
		return day10StrategyResult{}, fmt.Errorf("failed to create option_b: %w", err)
	}

	if err := agent.SwitchBranch("option_a"); err != nil {
		return day10StrategyResult{}, err
	}
	aAnswer, err := runDay10BranchOption(agent, cfg, &result, "option_a", []string{
		"Решение в ветке: фронтенд React Web.",
		"Срок ветки: 6 недель.",
	})
	if err != nil {
		return day10StrategyResult{}, err
	}

	if err := agent.SwitchBranch("option_b"); err != nil {
		return day10StrategyResult{}, err
	}
	bAnswer, err := runDay10BranchOption(agent, cfg, &result, "option_b", []string{
		"Решение в ветке: нативно iOS + Android.",
		"Срок ветки: 12 недель.",
	})
	if err != nil {
		return day10StrategyResult{}, err
	}

	result.BranchIsolationScore, result.BranchIsolationMax = day10BranchIsolationScore(aAnswer, bAnswer)
	result.BranchingNotes = []string{
		"checkpoint: spec_v1",
		"branches: option_a, option_b",
		"switching: main -> option_a -> option_b",
	}
	return result, nil
}

func runDay10BranchOption(agent *LLMAgent, cfg day10RunConfig, result *day10StrategyResult, branchName string, prompts []string) (string, error) {
	for i, prompt := range prompts {
		if cfg.Progress {
			fmt.Printf("[branching] %s turn %d/%d\n", branchName, i+1, len(prompts))
		}
		reply, err := agent.Reply(prompt)
		if err != nil {
			return "", fmt.Errorf("%s prompt %d failed: %w", branchName, i+1, err)
		}
		day10Accumulate(result, reply, cfg.Metadata, cfg.Model)
	}

	finalReply, err := agent.Reply("Собери итог ветки в JSON с полями platform, deadline_weeks, notes.")
	if err != nil {
		return "", fmt.Errorf("%s final failed: %w", branchName, err)
	}
	day10Accumulate(result, finalReply, cfg.Metadata, cfg.Model)
	return strings.TrimSpace(finalReply.Answer), nil
}

func day10Accumulate(result *day10StrategyResult, reply openRouterResult, metadata map[string]modelInfo, requestedModel string) {
	result.PromptTokens += reply.Tokens.PromptTokens
	result.ResponseTokens += reply.Tokens.ResponseTokens
	result.TotalTokens += reply.Tokens.TotalTokens

	if meta, ok := resolveModelMeta(metadata, reply.Model, requestedModel); ok {
		if cost, known := estimateCostUSD(reply.Usage, meta); known {
			result.CostUSD += cost
		}
	}
}

func day10BasePrompts() []string {
	return []string{
		"Цель: собрать ТЗ для MVP приложения доставки.",
		"Ограничение: бюджет 15000 USD.",
		"Ограничение: срок запуска 8 недель.",
		"Предпочтение: backend на Go, база PostgreSQL.",
		"Решение: авторизация через email + OTP.",
		"Договоренность: отчет по прогрессу каждую пятницу.",
		"Требование: офлайн-режим для курьеров.",
		"Требование: push-уведомления обязательны.",
		"Изменение решения: вместо Firebase используем OneSignal.",
		"Ограничение: SLA API не ниже 99.9%.",
	}
}

func day10QualityScore(answer string) (int, int) {
	text := strings.ToLower(answer)
	checks := [][]string{
		{"mvp", "достав"},
		{"15000"},
		{"8", "недель"},
		{"go"},
		{"postgres"},
		{"otp"},
		{"onesignal"},
		{"99.9"},
		{"пятниц", "friday"},
	}

	score := 0
	for _, patterns := range checks {
		matched := false
		for _, p := range patterns {
			if strings.Contains(text, strings.ToLower(p)) {
				matched = true
				break
			}
		}
		if matched {
			score++
		}
	}
	return score, len(checks)
}

func day10StabilityScore(answer string) (int, int) {
	text := strings.ToLower(answer)
	checks := [][]string{
		{"15000"},
		{"8", "недель"},
		{"99.9"},
		{"onesignal"},
	}
	score := 0
	for _, patterns := range checks {
		matched := false
		for _, p := range patterns {
			if strings.Contains(text, strings.ToLower(p)) {
				matched = true
				break
			}
		}
		if matched {
			score++
		}
	}
	return score, len(checks)
}

func day10BranchIsolationScore(optionA, optionB string) (int, int) {
	a := strings.ToLower(optionA)
	b := strings.ToLower(optionB)
	score := 0

	if strings.Contains(a, "react") || strings.Contains(a, "web") {
		score++
	}
	if strings.Contains(a, "6") && !strings.Contains(a, "12") {
		score++
	}
	if strings.Contains(b, "native") || strings.Contains(b, "ios") || strings.Contains(b, "android") {
		score++
	}
	if strings.Contains(b, "12") && !strings.Contains(b, "6 недель") {
		score++
	}
	return score, 4
}

func usabilityNote(strategy ContextStrategy) string {
	switch strategy {
	case ContextStrategySliding:
		return "Самый простой режим, но легко теряет ранние договоренности."
	case ContextStrategyFacts:
		return "Хороший баланс: важные факты стабильнее при умеренной цене."
	case ContextStrategyBranching:
		return "Лучший режим для альтернатив, но требует команд для управления ветками."
	default:
		return "Базовый режим."
	}
}

func printDay10Results(model string, results []day10StrategyResult) {
	fmt.Println("=== День 10: Стратегии контекста (без summary) ===")
	fmt.Printf("Модель: %s\n\n", model)

	for _, r := range results {
		fmt.Printf("--- %s (%s) ---\n", r.Name, r.Strategy)
		fmt.Printf("quality=%d/%d stability=%d/%d tokens(prompt=%d total=%d) cost=$%.6f\n",
			r.QualityScore, r.QualityMax,
			r.StabilityScore, r.StabilityMax,
			r.PromptTokens, r.TotalTokens, r.CostUSD,
		)
		if r.BranchIsolationMax > 0 {
			fmt.Printf("branch_isolation=%d/%d\n", r.BranchIsolationScore, r.BranchIsolationMax)
		}
		fmt.Printf("ux: %s\n\n", r.UsabilityNote)
	}
}

func writeDay10Report(path, model string, results []day10StrategyResult) error {
	var b strings.Builder
	b.WriteString("# Day 10 Results: Context Strategies (No Summary)\n\n")
	b.WriteString("Model: `" + model + "`\n\n")
	b.WriteString("| Strategy | Quality | Stability | Prompt Tokens | Total Tokens | Cost USD | Branch Isolation |\n")
	b.WriteString("|---|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range results {
		branchCol := "-"
		if r.BranchIsolationMax > 0 {
			branchCol = fmt.Sprintf("%d/%d", r.BranchIsolationScore, r.BranchIsolationMax)
		}
		b.WriteString(fmt.Sprintf(
			"| %s | %d/%d | %d/%d | %d | %d | %.6f | %s |\n",
			r.Name, r.QualityScore, r.QualityMax, r.StabilityScore, r.StabilityMax,
			r.PromptTokens, r.TotalTokens, r.CostUSD, branchCol,
		))
	}
	b.WriteString("\n")

	for _, r := range results {
		b.WriteString("## " + r.Name + "\n\n")
		b.WriteString("- strategy: `" + string(r.Strategy) + "`\n")
		b.WriteString(fmt.Sprintf("- quality: `%d/%d`\n", r.QualityScore, r.QualityMax))
		b.WriteString(fmt.Sprintf("- stability: `%d/%d`\n", r.StabilityScore, r.StabilityMax))
		if r.BranchIsolationMax > 0 {
			b.WriteString(fmt.Sprintf("- branch isolation: `%d/%d`\n", r.BranchIsolationScore, r.BranchIsolationMax))
		}
		b.WriteString(fmt.Sprintf("- prompt tokens: `%d`\n", r.PromptTokens))
		b.WriteString(fmt.Sprintf("- total tokens: `%d`\n", r.TotalTokens))
		b.WriteString(fmt.Sprintf("- cost: `$%.6f`\n", r.CostUSD))
		b.WriteString("- usability: " + r.UsabilityNote + "\n")
		if len(r.BranchingNotes) > 0 {
			b.WriteString("- branching flow: `" + strings.Join(r.BranchingNotes, " | ") + "`\n")
		}
		b.WriteString("\nFinal answer sample:\n")
		b.WriteString("```text\n" + sanitizeCodeFences(strings.TrimSpace(r.FinalAnswer)) + "\n```\n\n")
	}

	bestQuality := results[0]
	cheapest := results[0]
	mostStable := results[0]
	for _, r := range results[1:] {
		if r.QualityScore > bestQuality.QualityScore {
			bestQuality = r
		}
		if r.StabilityScore > mostStable.StabilityScore {
			mostStable = r
		}
		if r.TotalTokens < cheapest.TotalTokens {
			cheapest = r
		}
	}

	b.WriteString("## Summary\n")
	b.WriteString(fmt.Sprintf("- Best quality: `%s` (%d/%d)\n", bestQuality.Name, bestQuality.QualityScore, bestQuality.QualityMax))
	b.WriteString(fmt.Sprintf("- Best stability: `%s` (%d/%d)\n", mostStable.Name, mostStable.StabilityScore, mostStable.StabilityMax))
	b.WriteString(fmt.Sprintf("- Lowest token usage: `%s` (%d total tokens)\n", cheapest.Name, cheapest.TotalTokens))
	b.WriteString("- Branching keeps alternative lines of reasoning isolated at higher token cost.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay10Usage() {
	fmt.Println("Usage: openrouter-cli day10 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -model string         OpenRouter model")
	fmt.Println("  -max-tokens int       Max completion tokens per turn")
	fmt.Println("  -temperature float    Temperature")
	fmt.Println("  -window-size int      Context window for sliding/facts")
	fmt.Println("  -history-dir string   Directory for scenario history files")
	fmt.Println("  -report string        Markdown report path")
	fmt.Println("  -progress             Show live progress")
	fmt.Println("  -help                 Show help")
}
