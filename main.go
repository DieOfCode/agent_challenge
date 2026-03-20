package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"
)

const openRouterURL = "https://openrouter.ai/api/v1/chat/completions"

type chatRequest struct {
	Model       string    `json:"model"`
	Messages    []message `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	Stop        []string  `json:"stop,omitempty"`
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type methodResult struct {
	Name   string
	Answer string
	Score  int
}

type temperatureResult struct {
	Temperature float64
	Answer      string
	Accuracy    int
	Creativity  int
	Diversity   int
}

func main() {
	if err := loadDotEnv(".env"); err != nil {
		exitf("failed to load .env: %v", err)
	}

	if len(os.Args) > 1 && !strings.HasPrefix(os.Args[1], "-") {
		switch os.Args[1] {
		case "day3":
			if err := runDay3Command(os.Args[2:]); err != nil {
				exitf("day3 failed: %v", err)
			}
			return
		case "day4":
			if err := runDay4Command(os.Args[2:]); err != nil {
				exitf("day4 failed: %v", err)
			}
			return
		case "help":
			printRootUsage()
			return
		default:
			exitf("unknown subcommand: %s\n\n%s", os.Args[1], rootUsage())
		}
	}

	if err := runChatCommand(os.Args[1:]); err != nil {
		exitf("%v", err)
	}
}

func runChatCommand(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	model := fs.String("model", getDefaultModel(), "OpenRouter model")
	prompt := fs.String("prompt", "", "Prompt text (if empty, will read from stdin)")
	format := fs.String("format", `JSON object: {"answer":"..."}`, "Explicit output format instruction")
	maxTokens := fs.Int("max-tokens", 200, "Maximum response tokens")
	stopSequence := fs.String("stop", "<END>", "Stop sequence")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse flags: %w", err)
	}
	if *help {
		printChatUsage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected arguments: %s", strings.Join(fs.Args(), " "))
	}

	userPrompt := strings.TrimSpace(*prompt)
	if userPrompt == "" {
		input, err := io.ReadAll(os.Stdin)
		if err != nil {
			return fmt.Errorf("failed to read stdin: %w", err)
		}
		userPrompt = strings.TrimSpace(string(input))
	}
	if userPrompt == "" {
		return fmt.Errorf("empty prompt: use -prompt or pipe text to stdin")
	}

	systemInstruction := fmt.Sprintf(
		"Answer format: %s\nLength limit: no more than 80 words.\nHard cap: no more than %d tokens.\nAfter the formatted answer, emit stop sequence: %s",
		strings.TrimSpace(*format),
		*maxTokens,
		strings.TrimSpace(*stopSequence),
	)

	var stop []string
	if seq := strings.TrimSpace(*stopSequence); seq != "" {
		stop = []string{seq}
	}

	answer, err := callOpenRouter(
		getAPIKey(),
		*model,
		[]message{
			{Role: "system", Content: systemInstruction},
			{Role: "user", Content: userPrompt},
		},
		*maxTokens,
		nil,
		stop,
		"minimal-go-cli",
	)
	if err != nil {
		return err
	}

	fmt.Println(strings.TrimSpace(answer))
	return nil
}

func runDay3Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day3", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	model := fs.String("model", getDefaultModel(), "OpenRouter model")
	task := fs.String("task", `Сколько различных перестановок у слова "LEVEL"? Дайте ответ и короткое обоснование.`, "Task to solve")
	expected := fs.String("expected", "30", "Expected exact value for simple accuracy scoring")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day3 flags: %w", err)
	}
	if *help {
		printDay3Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day3 arguments: %s", strings.Join(fs.Args(), " "))
	}

	apiKey := getAPIKey()

	direct, err := callOpenRouter(apiKey, *model, []message{
		{Role: "user", Content: *task},
	}, 400, nil, nil, "reasoning-day3-cli")
	if err != nil {
		return fmt.Errorf("direct method failed: %w", err)
	}

	stepByStep, err := callOpenRouter(apiKey, *model, []message{
		{Role: "user", Content: "Решай пошагово.\n\nЗадача:\n" + *task},
	}, 500, nil, nil, "reasoning-day3-cli")
	if err != nil {
		return fmt.Errorf("step-by-step method failed: %w", err)
	}

	promptForTask, err := callOpenRouter(apiKey, *model, []message{
		{
			Role: "user",
			Content: "Составь лучший краткий промпт для точного решения задачи. " +
				"Верни только сам промпт без пояснений.\n\nЗадача:\n" + *task,
		},
	}, 200, nil, nil, "reasoning-day3-cli")
	if err != nil {
		return fmt.Errorf("prompt-constructor method (phase 1) failed: %w", err)
	}
	promptForTask = strings.TrimSpace(promptForTask)

	promptGeneratedSolution, err := callOpenRouter(apiKey, *model, []message{
		{Role: "user", Content: promptForTask},
	}, 500, nil, nil, "reasoning-day3-cli")
	if err != nil {
		return fmt.Errorf("prompt-constructor method (phase 2) failed: %w", err)
	}

	expertsSolution, err := callOpenRouter(apiKey, *model, []message{
		{
			Role: "user",
			Content: "Ты группа экспертов. Дай решение отдельно от каждого:\n" +
				"1) Аналитик\n2) Инженер\n3) Критик\n" +
				"После этого дай общий вывод.\n\nЗадача:\n" + *task,
		},
	}, 900, nil, nil, "reasoning-day3-cli")
	if err != nil {
		return fmt.Errorf("experts method failed: %w", err)
	}

	results := []methodResult{
		{Name: "1) Прямой ответ", Answer: direct},
		{Name: "2) Пошагово", Answer: stepByStep},
		{Name: "3) Сначала промпт, затем решение", Answer: promptGeneratedSolution},
		{Name: "4) Группа экспертов", Answer: expertsSolution},
	}

	for i := range results {
		results[i].Score = scoreAccuracy(results[i].Answer, *expected)
	}

	best := append([]methodResult(nil), results...)
	sort.SliceStable(best, func(i, j int) bool { return best[i].Score > best[j].Score })
	topScore := best[0].Score
	var topMethods []string
	for _, item := range best {
		if item.Score == topScore {
			topMethods = append(topMethods, item.Name)
		}
	}

	fmt.Println("=== День 3: Разные способы рассуждения ===")
	fmt.Printf("Модель: %s\n", *model)
	fmt.Printf("Задача: %s\n\n", *task)

	for _, r := range results {
		fmt.Printf("--- %s ---\n%s\n\n", r.Name, strings.TrimSpace(r.Answer))
	}

	fmt.Println("=== Сравнение ===")
	for _, r := range results {
		fmt.Printf("%s -> score=%d, содержит точный ответ %q: %v\n", r.Name, r.Score, *expected, containsExactAnswer(r.Answer, *expected))
	}
	fmt.Printf("Наиболее точный результат: %s\n", strings.Join(topMethods, ", "))
	return nil
}

func runDay4Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day4", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	model := fs.String("model", getDefaultModel(), "OpenRouter model")
	prompt := fs.String("prompt", `Ответь в одну строку формата: result=<число>; reason=<до 12 слов>; metaphor=<до 7 слов>. Сколько уникальных перестановок у слова LEVEL?`, "Prompt to evaluate across temperatures")
	expected := fs.String("expected", "result=30", "Expected exact value used for accuracy scoring")
	maxTokens := fs.Int("max-tokens", 220, "Maximum response tokens for each run")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day4 flags: %w", err)
	}
	if *help {
		printDay4Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day4 arguments: %s", strings.Join(fs.Args(), " "))
	}

	apiKey := getAPIKey()
	temps := []float64{0, 0.7, 1.2}
	results := make([]temperatureResult, 0, len(temps))

	for _, temp := range temps {
		t := temp
		answer, err := callOpenRouter(apiKey, *model, []message{
			{Role: "user", Content: *prompt},
		}, *maxTokens, &t, nil, "temperature-day4-cli")
		if err != nil {
			return fmt.Errorf("temperature %.1f failed: %w", temp, err)
		}

		results = append(results, temperatureResult{
			Temperature: temp,
			Answer:      answer,
			Accuracy:    exactAccuracyScore(answer, *expected),
			Creativity:  creativityScore(answer),
		})
	}

	for i := range results {
		results[i].Diversity = diversityScore(i, results)
	}

	bestAccuracy := pickBestTemperature(results, func(r temperatureResult) int { return r.Accuracy })
	bestCreativity := pickBestTemperature(results, func(r temperatureResult) int { return r.Creativity })
	bestDiversity := pickBestTemperature(results, func(r temperatureResult) int { return r.Diversity })

	fmt.Println("=== День 4: Температура ===")
	fmt.Printf("Модель: %s\n", *model)
	fmt.Printf("Промпт: %s\n\n", *prompt)

	for _, r := range results {
		fmt.Printf("--- temperature = %.1f ---\n%s\n\n", r.Temperature, strings.TrimSpace(r.Answer))
	}

	fmt.Println("=== Сравнение ===")
	for _, r := range results {
		fmt.Printf("temperature=%.1f -> accuracy=%d, creativity=%d, diversity=%d\n", r.Temperature, r.Accuracy, r.Creativity, r.Diversity)
	}
	fmt.Printf("Лучшая точность: temperature=%.1f\n", bestAccuracy)
	fmt.Printf("Лучшая креативность: temperature=%.1f\n", bestCreativity)
	fmt.Printf("Лучшее разнообразие: temperature=%.1f\n", bestDiversity)

	fmt.Println("=== Для каких задач подходит ===")
	fmt.Println("temperature=0.0 -> точные, формальные и повторяемые задачи: извлечение фактов, проверка формул, детерминированные ответы.")
	fmt.Println("temperature=0.7 -> универсальный баланс: объяснения, черновики текстов, продуктовые описания, где важны и точность, и живость.")
	fmt.Println("temperature=1.2 -> брейншторм и вариативность: идеи, сторителлинг, поиск необычных формулировок.")

	return nil
}

func callOpenRouter(apiKey, model string, messages []message, maxTokens int, temperature *float64, stop []string, title string) (string, error) {
	reqPayload := chatRequest{
		Model:    model,
		Messages: messages,
	}
	if maxTokens > 0 {
		reqPayload.MaxTokens = maxTokens
	}
	if temperature != nil {
		reqPayload.Temperature = temperature
	}
	if len(stop) > 0 {
		reqPayload.Stop = stop
	}

	reqBody, err := json.Marshal(reqPayload)
	if err != nil {
		return "", fmt.Errorf("failed to encode request: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, openRouterURL, bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("HTTP-Referer", "https://localhost")
	req.Header.Set("X-Title", title)

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	var out chatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", fmt.Errorf("invalid JSON response: %w\nraw: %s", err, string(raw))
	}

	if resp.StatusCode >= 400 {
		if out.Error != nil && out.Error.Message != "" {
			return "", fmt.Errorf("API error (%s): %s", resp.Status, out.Error.Message)
		}
		return "", fmt.Errorf("API error (%s): %s", resp.Status, string(raw))
	}

	if len(out.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	return strings.TrimSpace(out.Choices[0].Message.Content), nil
}

func containsExactAnswer(answer, expected string) bool {
	re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(expected) + `\b`)
	return re.MatchString(answer)
}

func scoreAccuracy(answer, expected string) int {
	score := 0
	lower := strings.ToLower(answer)

	if containsExactAnswer(answer, expected) {
		score += 2
	}
	if strings.Contains(lower, "5!") || strings.Contains(lower, "120") {
		score++
	}
	if strings.Contains(lower, "2!") || strings.Contains(lower, "повтор") || strings.Contains(lower, "одинаков") {
		score++
	}

	return score
}

func exactAccuracyScore(answer, expected string) int {
	expected = strings.ToLower(strings.TrimSpace(expected))
	answerLower := strings.ToLower(answer)
	if expected == "" {
		return 0
	}

	score := 20

	if strings.Contains(answerLower, expected) {
		score = 100
	} else {
		candidate := expected
		if idx := strings.Index(expected, "="); idx != -1 && idx+1 < len(expected) {
			candidate = strings.TrimSpace(expected[idx+1:])
		}
		if candidate != "" && containsExactAnswer(answerLower, candidate) {
			score = 70
		}
	}

	if strings.Contains(answerLower, "result=60") && strings.Contains(expected, "30") {
		score -= 50
	}
	if strings.Contains(answerLower, "не "+expected) || strings.Contains(answerLower, "not "+expected) {
		score -= 40
	}

	if score > 100 {
		return 100
	}
	if score < 0 {
		return 0
	}
	return score
}

var wordTokenRE = regexp.MustCompile(`[\pL\pN]+`)

func creativityScore(answer string) int {
	tokens := wordTokenRE.FindAllString(strings.ToLower(answer), -1)
	if len(tokens) == 0 {
		return 0
	}

	unique := make(map[string]struct{}, len(tokens))
	for _, t := range tokens {
		unique[t] = struct{}{}
	}

	richness := float64(len(unique)) / float64(len(tokens))
	score := int(richness * 70)

	if len(tokens) > 80 {
		score += 10
	} else if len(tokens) > 40 {
		score += 5
	}

	if strings.ContainsAny(answer, "!?") {
		score += 5
	}
	if strings.Contains(strings.ToLower(answer), "как") || strings.Contains(strings.ToLower(answer), "словно") || strings.Contains(strings.ToLower(answer), "будто") {
		score += 10
	}

	if score > 100 {
		return 100
	}
	if score < 0 {
		return 0
	}
	return score
}

func diversityScore(index int, all []temperatureResult) int {
	if len(all) <= 1 {
		return 0
	}

	sum := 0.0
	for i, other := range all {
		if i == index {
			continue
		}
		sum += jaccardDistance(all[index].Answer, other.Answer)
	}

	avg := sum / float64(len(all)-1)
	score := int(avg * 100)
	if score > 100 {
		return 100
	}
	if score < 0 {
		return 0
	}
	return score
}

func jaccardDistance(a, b string) float64 {
	setA := tokenSet(a)
	setB := tokenSet(b)

	if len(setA) == 0 && len(setB) == 0 {
		return 0
	}

	intersection := 0
	for token := range setA {
		if _, ok := setB[token]; ok {
			intersection++
		}
	}

	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return 0
	}

	return 1 - (float64(intersection) / float64(union))
}

func tokenSet(text string) map[string]struct{} {
	tokens := wordTokenRE.FindAllString(strings.ToLower(text), -1)
	set := make(map[string]struct{}, len(tokens))
	for _, token := range tokens {
		set[token] = struct{}{}
	}
	return set
}

func pickBestTemperature(results []temperatureResult, selector func(temperatureResult) int) float64 {
	if len(results) == 0 {
		return 0
	}
	best := results[0]
	bestScore := selector(best)
	for _, item := range results[1:] {
		s := selector(item)
		if s > bestScore {
			best = item
			bestScore = s
		}
	}
	return best.Temperature
}

func getDefaultModel() string {
	model := strings.TrimSpace(os.Getenv("OPENROUTER_MODEL"))
	if model == "" {
		return "openai/gpt-4o-mini"
	}
	return model
}

func getAPIKey() string {
	apiKey := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	if apiKey == "" {
		exitf("OPENROUTER_API_KEY is not set")
	}
	return apiKey
}

func rootUsage() string {
	return "Usage:\n  openrouter-cli [flags]\n  openrouter-cli day3 [flags]\n  openrouter-cli day4 [flags]\n\nUse `openrouter-cli --help` for chat flags, `openrouter-cli day3 --help` for Day 3 flags, and `openrouter-cli day4 --help` for Day 4 flags."
}

func printRootUsage() {
	fmt.Println(rootUsage())
}

func printChatUsage() {
	fmt.Println("Usage: openrouter-cli [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -model string       OpenRouter model")
	fmt.Println("  -prompt string      Prompt text (if empty, reads from stdin)")
	fmt.Println("  -format string      Explicit output format instruction")
	fmt.Println("  -max-tokens int     Maximum response tokens")
	fmt.Println("  -stop string        Stop sequence")
	fmt.Println("  -help               Show help")
	fmt.Println("Subcommands:")
	fmt.Println("  day3                Run four reasoning strategies and compare results")
	fmt.Println("  day4                Run same prompt with different temperatures and compare")
}

func printDay3Usage() {
	fmt.Println("Usage: openrouter-cli day3 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -model string       OpenRouter model")
	fmt.Println(`  -task string        Task to solve (default: "Сколько различных перестановок у слова \"LEVEL\"?...")`)
	fmt.Println(`  -expected string    Expected exact value used for scoring (default: "30")`)
	fmt.Println("  -help               Show help")
}

func printDay4Usage() {
	fmt.Println("Usage: openrouter-cli day4 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -model string       OpenRouter model")
	fmt.Println(`  -prompt string      Prompt to run at temperatures 0, 0.7, 1.2`)
	fmt.Println(`  -expected string    Expected exact value used for accuracy scoring`)
	fmt.Println("  -max-tokens int     Maximum response tokens for each run")
	fmt.Println("  -help               Show help")
}

func exitf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

var envLineRE = regexp.MustCompile(`^([A-Za-z_][A-Za-z0-9_]*)=(.*)$`)

func loadDotEnv(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	for i, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		matches := envLineRE.FindStringSubmatch(line)
		if len(matches) != 3 {
			return fmt.Errorf("invalid .env format at line %d", i+1)
		}

		key := matches[1]
		value := strings.TrimSpace(matches[2])
		if len(value) >= 2 {
			if (value[0] == '"' && value[len(value)-1] == '"') || (value[0] == '\'' && value[len(value)-1] == '\'') {
				value = value[1 : len(value)-1]
			}
		}

		// Keep explicit shell env vars higher priority than values from .env.
		if _, exists := os.LookupEnv(key); !exists {
			if err := os.Setenv(key, value); err != nil {
				return err
			}
		}
	}

	return nil
}
