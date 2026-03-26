package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type day8Turn struct {
	Turn              int
	Input             string
	Output            string
	Tokens            tokenStats
	TurnCostUSD       float64
	CumulativeCostUSD float64
	Error             string
	Overflow          bool
}

type day8ScenarioResult struct {
	Name  string
	Turns []day8Turn
}

func runDay8Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day8", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	model := fs.String("model", getDefaultModel(), "OpenRouter model")
	maxTokens := fs.Int("max-tokens", 180, "Maximum response tokens per turn")
	contextLimit := fs.Int("context-limit", 700, "Context limit for short/long scenarios (local pre-check)")
	overflowLimit := fs.Int("overflow-limit", 320, "Context limit for overflow scenario")
	temperature := fs.Float64("temperature", 0.2, "Temperature")
	historyDir := fs.String("history-dir", "/tmp/day8-agent-context", "Directory for scenario history files")
	reportPath := fs.String("report", "DAY8_RESULTS.md", "Markdown report path")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day8 flags: %w", err)
	}
	if *help {
		printDay8Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day8 arguments: %s", strings.Join(fs.Args(), " "))
	}

	if err := os.MkdirAll(*historyDir, 0o755); err != nil {
		return fmt.Errorf("failed to create history dir: %w", err)
	}

	apiKey := getAPIKey()
	metadata, metaErr := fetchModelMetadata(apiKey)
	if metaErr != nil {
		fmt.Fprintf(os.Stderr, "warning: failed to fetch model metadata for day8 cost estimate: %v\n", metaErr)
	}

	shortInputs := []string{
		"Привет. Меня зовут Алиса. Запомни это.",
		"Как меня зовут?",
	}

	longInputs := []string{
		"Запомни этот длинный контекст: " + strings.Repeat("мобильный api и протоколы; ", 45),
		"Добавь к памяти ещё блок: " + strings.Repeat("latency throughput reliability; ", 45),
		"Сохрани дополнительный блок: " + strings.Repeat("quic tcp udp congestion control; ", 45),
		"Теперь суммируй всё кратко в 3 пункта.",
	}

	overflowInputs := []string{
		"Сохрани текст: " + strings.Repeat("контекст ", 120),
		"Сохрани текст: " + strings.Repeat("история ", 120),
		"Сохрани текст: " + strings.Repeat("память ", 120),
		"Какой был самый первый маркер?",
	}

	shortResult, err := runDay8Scenario(day8ScenarioConfig{
		Name:         "short_dialog",
		Model:        *model,
		APIKey:       apiKey,
		MaxTokens:    *maxTokens,
		ContextLimit: *contextLimit,
		Temperature:  *temperature,
		HistoryFile:  filepath.Join(*historyDir, "short.json"),
		Inputs:       shortInputs,
		Metadata:     metadata,
	})
	if err != nil {
		return err
	}

	longResult, err := runDay8Scenario(day8ScenarioConfig{
		Name:         "long_dialog",
		Model:        *model,
		APIKey:       apiKey,
		MaxTokens:    *maxTokens,
		ContextLimit: *contextLimit,
		Temperature:  *temperature,
		HistoryFile:  filepath.Join(*historyDir, "long.json"),
		Inputs:       longInputs,
		Metadata:     metadata,
	})
	if err != nil {
		return err
	}

	overflowResult, err := runDay8Scenario(day8ScenarioConfig{
		Name:         "overflow_dialog",
		Model:        *model,
		APIKey:       apiKey,
		MaxTokens:    *maxTokens,
		ContextLimit: *overflowLimit,
		Temperature:  *temperature,
		HistoryFile:  filepath.Join(*historyDir, "overflow.json"),
		Inputs:       overflowInputs,
		Metadata:     metadata,
	})
	if err != nil {
		return err
	}

	scenarios := []day8ScenarioResult{shortResult, longResult, overflowResult}
	printDay8Results(*model, scenarios)

	if err := writeDay8Report(*reportPath, *model, scenarios); err != nil {
		return fmt.Errorf("failed to write day8 report: %w", err)
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)

	return nil
}

type day8ScenarioConfig struct {
	Name         string
	Model        string
	APIKey       string
	MaxTokens    int
	ContextLimit int
	Temperature  float64
	HistoryFile  string
	Inputs       []string
	Metadata     map[string]modelInfo
}

func runDay8Scenario(cfg day8ScenarioConfig) (day8ScenarioResult, error) {
	store := NewJSONHistoryStore(cfg.HistoryFile)
	if err := store.Reset(); err != nil {
		return day8ScenarioResult{}, fmt.Errorf("failed to reset %s history: %w", cfg.Name, err)
	}

	t := cfg.Temperature
	agent, err := NewLLMAgent(LLMAgentConfig{
		APIKey:       cfg.APIKey,
		Model:        cfg.Model,
		MaxTokens:    cfg.MaxTokens,
		Temperature:  &t,
		SystemPrompt: "Ты полезный ассистент. Отвечай кратко.",
		Title:        "day8-token-analysis",
		HistoryStore: store,
		ContextLimit: cfg.ContextLimit,
	})
	if err != nil {
		return day8ScenarioResult{}, err
	}

	result := day8ScenarioResult{
		Name:  cfg.Name,
		Turns: make([]day8Turn, 0, len(cfg.Inputs)),
	}

	cumulativeCost := 0.0
	for i, input := range cfg.Inputs {
		turn := day8Turn{
			Turn:  i + 1,
			Input: input,
		}

		reply, err := agent.Reply(input)
		if err != nil {
			turn.Error = err.Error()
			var overflowErr *ContextLimitError
			if errors.As(err, &overflowErr) {
				turn.Overflow = true
			}
			result.Turns = append(result.Turns, turn)
			break
		}

		turn.Output = reply.Answer
		turn.Tokens = reply.Tokens

		if meta, ok := resolveModelMeta(cfg.Metadata, reply.Model, cfg.Model); ok {
			if cost, known := estimateCostUSD(reply.Usage, meta); known {
				turn.TurnCostUSD = cost
				cumulativeCost += cost
				turn.CumulativeCostUSD = cumulativeCost
			}
		}

		result.Turns = append(result.Turns, turn)
	}

	return result, nil
}

func resolveModelMeta(metadata map[string]modelInfo, actualModel, requestedModel string) (modelInfo, bool) {
	if metadata == nil {
		return modelInfo{}, false
	}
	if m, ok := metadata[strings.TrimSpace(actualModel)]; ok {
		return m, true
	}
	if m, ok := metadata[strings.TrimSpace(requestedModel)]; ok {
		return m, true
	}
	return modelInfo{}, false
}

func printDay8Results(model string, scenarios []day8ScenarioResult) {
	fmt.Println("=== День 8: Работа с токенами ===")
	fmt.Printf("Модель: %s\n\n", model)

	for _, scenario := range scenarios {
		fmt.Printf("--- %s ---\n", scenario.Name)
		for _, turn := range scenario.Turns {
			if turn.Error != "" {
				fmt.Printf("turn=%d ERROR: %s\n", turn.Turn, turn.Error)
				continue
			}

			fmt.Printf(
				"turn=%d tokens(history_est=%d request_est=%d prompt=%d response=%d total=%d cumulative_total=%d)\n",
				turn.Turn,
				turn.Tokens.EstimatedHistoryTokens,
				turn.Tokens.EstimatedRequestTokens,
				turn.Tokens.PromptTokens,
				turn.Tokens.ResponseTokens,
				turn.Tokens.TotalTokens,
				turn.Tokens.CumulativeTotalTokens,
			)

			if turn.CumulativeCostUSD > 0 || turn.TurnCostUSD > 0 {
				fmt.Printf("         cost(turn=$%.6f cumulative=$%.6f)\n", turn.TurnCostUSD, turn.CumulativeCostUSD)
			}
		}
		fmt.Println()
	}

	shortLast := lastSuccessfulTurn(scenarios[0])
	longLast := lastSuccessfulTurn(scenarios[1])
	overflowErrTurn := firstErrorTurn(scenarios[2])

	fmt.Println("=== Сравнение ===")
	fmt.Printf("Короткий диалог: cumulative_total_tokens=%d cumulative_cost=$%.6f\n", shortLast.Tokens.CumulativeTotalTokens, shortLast.CumulativeCostUSD)
	fmt.Printf("Длинный диалог: cumulative_total_tokens=%d cumulative_cost=$%.6f\n", longLast.Tokens.CumulativeTotalTokens, longLast.CumulativeCostUSD)
	if overflowErrTurn != nil {
		fmt.Printf("Переполнение: сломалось на turn=%d, ошибка: %s\n", overflowErrTurn.Turn, overflowErrTurn.Error)
	}
}

func lastSuccessfulTurn(s day8ScenarioResult) day8Turn {
	best := day8Turn{}
	for _, turn := range s.Turns {
		if turn.Error == "" {
			best = turn
		}
	}
	return best
}

func firstErrorTurn(s day8ScenarioResult) *day8Turn {
	for i := range s.Turns {
		if s.Turns[i].Error != "" {
			return &s.Turns[i]
		}
	}
	return nil
}

func writeDay8Report(path, model string, scenarios []day8ScenarioResult) error {
	var b strings.Builder
	b.WriteString("# Day 8 Results: Token Behavior\n\n")
	b.WriteString("Model: `" + model + "`\n\n")

	for _, s := range scenarios {
		b.WriteString("## " + s.Name + "\n\n")
		for _, turn := range s.Turns {
			if turn.Error != "" {
				b.WriteString(fmt.Sprintf("- turn=%d ERROR: `%s`\n", turn.Turn, turn.Error))
				continue
			}
			b.WriteString(fmt.Sprintf(
				"- turn=%d prompt=%d response=%d total=%d cumulative_total=%d",
				turn.Turn,
				turn.Tokens.PromptTokens,
				turn.Tokens.ResponseTokens,
				turn.Tokens.TotalTokens,
				turn.Tokens.CumulativeTotalTokens,
			))
			if turn.CumulativeCostUSD > 0 || turn.TurnCostUSD > 0 {
				b.WriteString(fmt.Sprintf(" cost(turn=$%.6f cumulative=$%.6f)", turn.TurnCostUSD, turn.CumulativeCostUSD))
			}
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}

	shortLast := lastSuccessfulTurn(scenarios[0])
	longLast := lastSuccessfulTurn(scenarios[1])
	overflowErrTurn := firstErrorTurn(scenarios[2])

	b.WriteString("## Summary\n")
	b.WriteString(fmt.Sprintf("- short dialog cumulative tokens: `%d`\n", shortLast.Tokens.CumulativeTotalTokens))
	b.WriteString(fmt.Sprintf("- long dialog cumulative tokens: `%d`\n", longLast.Tokens.CumulativeTotalTokens))
	if overflowErrTurn != nil {
		b.WriteString(fmt.Sprintf("- overflow breakage at turn `%d`: `%s`\n", overflowErrTurn.Turn, overflowErrTurn.Error))
	}
	b.WriteString("- Token/cost growth is visible turn-by-turn; overflow is blocked by context-limit pre-check.\n")

	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay8Usage() {
	fmt.Println("Usage: openrouter-cli day8 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -model string         OpenRouter model")
	fmt.Println("  -max-tokens int       Max completion tokens per turn")
	fmt.Println("  -context-limit int    Context limit for short/long scenarios")
	fmt.Println("  -overflow-limit int   Context limit used for overflow scenario")
	fmt.Println("  -temperature float    Temperature")
	fmt.Println("  -history-dir string   Directory for scenario history files")
	fmt.Println("  -report string        Markdown report path")
	fmt.Println("  -help                 Show help")
}
