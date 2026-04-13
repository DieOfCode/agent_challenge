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
	"strconv"
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
	Model   string `json:"model,omitempty"`
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage usageStats `json:"usage,omitempty"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type usageStats struct {
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
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

type openRouterResult struct {
	Answer  string
	Model   string
	Usage   usageStats
	Latency time.Duration
	Tokens  tokenStats
}

type tokenStats struct {
	EstimatedHistoryTokens   int
	EstimatedRequestTokens   int
	EstimatedResponseTokens  int
	PromptTokens             int
	ResponseTokens           int
	TotalTokens              int
	ConversationTokens       int
	CumulativePromptTokens   int
	CumulativeResponseTokens int
	CumulativeTotalTokens    int
	ContextLimit             int
}

type benchmarkResult struct {
	Tier           string
	ModelID        string
	ModelURL       string
	HuggingFaceURL string
	Answer         string
	Latency        time.Duration
	Usage          usageStats
	CostUSD        float64
	CostKnown      bool
	QualityScore   int
}

type modelInfo struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	CanonicalSlug string `json:"canonical_slug"`
	HuggingFaceID string `json:"hugging_face_id"`
	Pricing       struct {
		Prompt     string `json:"prompt"`
		Completion string `json:"completion"`
		Request    string `json:"request"`
	} `json:"pricing"`
}

type modelsResponse struct {
	Data []modelInfo `json:"data"`
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
		case "day5":
			if err := runDay5Command(os.Args[2:]); err != nil {
				exitf("day5 failed: %v", err)
			}
			return
		case "day8":
			if err := runDay8Command(os.Args[2:]); err != nil {
				exitf("day8 failed: %v", err)
			}
			return
		case "day9":
			if err := runDay9Command(os.Args[2:]); err != nil {
				exitf("day9 failed: %v", err)
			}
			return
		case "day10":
			if err := runDay10Command(os.Args[2:]); err != nil {
				exitf("day10 failed: %v", err)
			}
			return
		case "day11":
			if err := runDay11Command(os.Args[2:]); err != nil {
				exitf("day11 failed: %v", err)
			}
			return
		case "day12":
			if err := runDay12Command(os.Args[2:]); err != nil {
				exitf("day12 failed: %v", err)
			}
			return
		case "day13":
			if err := runDay13Command(os.Args[2:]); err != nil {
				exitf("day13 failed: %v", err)
			}
			return
		case "day14":
			if err := runDay14Command(os.Args[2:]); err != nil {
				exitf("day14 failed: %v", err)
			}
			return
		case "day15":
			if err := runDay15Command(os.Args[2:]); err != nil {
				exitf("day15 failed: %v", err)
			}
			return
		case "day16":
			if err := runDay16Command(os.Args[2:]); err != nil {
				exitf("day16 failed: %v", err)
			}
			return
		case "day17":
			if err := runDay17Command(os.Args[2:]); err != nil {
				exitf("day17 failed: %v", err)
			}
			return
		case "day19":
			if err := runDay19Command(os.Args[2:]); err != nil {
				exitf("day19 failed: %v", err)
			}
			return
		case "day20":
			if err := runDay20Command(os.Args[2:]); err != nil {
				exitf("day20 failed: %v", err)
			}
			return
		case "day18":
			if err := runDay18Command(os.Args[2:]); err != nil {
				exitf("day18 failed: %v", err)
			}
			return
		case "agent":
			if err := runAgentCommand(os.Args[2:]); err != nil {
				exitf("agent failed: %v", err)
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

func runDay5Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day5", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	modelWeak := fs.String("weak-model", "openai/gpt-4o-mini", "Weak model ID")
	modelMid := fs.String("mid-model", "openai/gpt-4.1-mini", "Mid model ID")
	modelStrong := fs.String("strong-model", "openai/gpt-4.1", "Strong model ID")
	prompt := fs.String("prompt", "Сравни HTTP/1.1, HTTP/2 и HTTP/3 для мобильного API. Формат: 3 буллета отличий и 1 практическая рекомендация. Обязательно упомяни: multiplexing, head-of-line blocking, QUIC.", "Prompt for all models")
	keywords := fs.String("quality-keywords", "multiplexing,head-of-line,quic,рекомендац", "Comma-separated quality markers")
	maxTokens := fs.Int("max-tokens", 350, "Maximum response tokens")
	temperature := fs.Float64("temperature", 0.2, "Temperature for all compared models")
	includeCost := fs.Bool("include-cost", true, "Calculate cost if pricing metadata is available")
	reportPath := fs.String("report", "DAY5_RESULTS.md", "Markdown report output path")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day5 flags: %w", err)
	}
	if *help {
		printDay5Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day5 arguments: %s", strings.Join(fs.Args(), " "))
	}

	apiKey := getAPIKey()
	metadata, metaErr := fetchModelMetadata(apiKey)
	if metaErr != nil {
		fmt.Fprintf(os.Stderr, "warning: failed to fetch model metadata: %v\n", metaErr)
	}

	t := *temperature
	models := []struct {
		tier    string
		modelID string
	}{
		{tier: "weak", modelID: strings.TrimSpace(*modelWeak)},
		{tier: "mid", modelID: strings.TrimSpace(*modelMid)},
		{tier: "strong", modelID: strings.TrimSpace(*modelStrong)},
	}

	qualityKeywords := parseCSVList(*keywords)
	results := make([]benchmarkResult, 0, len(models))

	for _, item := range models {
		if item.modelID == "" {
			return fmt.Errorf("%s model is empty", item.tier)
		}

		resp, err := callOpenRouterDetailed(apiKey, item.modelID, []message{
			{Role: "user", Content: *prompt},
		}, *maxTokens, &t, nil, "models-day5-cli")
		if err != nil {
			return fmt.Errorf("%s model (%s) failed: %w", item.tier, item.modelID, err)
		}

		modelID := item.modelID
		if strings.TrimSpace(resp.Model) != "" {
			modelID = strings.TrimSpace(resp.Model)
		}

		modelMeta, hasMeta := metadata[modelID]
		if !hasMeta {
			modelMeta, hasMeta = metadata[item.modelID]
		}

		result := benchmarkResult{
			Tier:         item.tier,
			ModelID:      modelID,
			ModelURL:     "https://openrouter.ai/" + modelID,
			Answer:       strings.TrimSpace(resp.Answer),
			Latency:      resp.Latency,
			Usage:        resp.Usage,
			QualityScore: qualityScore(resp.Answer, qualityKeywords),
		}

		if hasMeta {
			if modelMeta.ID != "" {
				result.ModelURL = "https://openrouter.ai/" + modelMeta.ID
			}
			if modelMeta.HuggingFaceID != "" {
				result.HuggingFaceURL = "https://huggingface.co/" + modelMeta.HuggingFaceID
			}
			if *includeCost {
				cost, known := estimateCostUSD(resp.Usage, modelMeta)
				result.CostUSD = cost
				result.CostKnown = known
			}
		}

		results = append(results, result)
	}

	fastest := pickFastestModel(results)
	bestQuality := pickBestQualityModel(results)
	efficient := pickMostEfficientModel(results)

	fmt.Println("=== День 5: Версии моделей ===")
	fmt.Printf("Промпт: %s\n\n", *prompt)
	for _, r := range results {
		fmt.Printf("--- %s model: %s ---\n", strings.ToUpper(r.Tier), r.ModelID)
		fmt.Printf("Latency: %s\n", r.Latency.Round(time.Millisecond))
		fmt.Printf("Tokens: prompt=%d, completion=%d, total=%d\n", r.Usage.PromptTokens, r.Usage.CompletionTokens, r.Usage.TotalTokens)
		if r.CostKnown {
			fmt.Printf("Cost: $%.6f\n", r.CostUSD)
		} else {
			fmt.Println("Cost: N/A")
		}
		fmt.Printf("Quality score: %d/100\n", r.QualityScore)
		fmt.Printf("OpenRouter: %s\n", r.ModelURL)
		if r.HuggingFaceURL != "" {
			fmt.Printf("HuggingFace: %s\n", r.HuggingFaceURL)
		}
		fmt.Printf("Answer:\n%s\n\n", r.Answer)
	}

	fmt.Println("=== Сравнение ===")
	fmt.Printf("Качество: лучший -> %s (%s)\n", strings.ToUpper(bestQuality.Tier), bestQuality.ModelID)
	fmt.Printf("Скорость: лучший -> %s (%s)\n", strings.ToUpper(fastest.Tier), fastest.ModelID)
	fmt.Printf("Ресурсоёмкость: лучший -> %s (%s)\n", strings.ToUpper(efficient.Tier), efficient.ModelID)
	fmt.Printf("Короткий вывод: слабая модель обычно дешевле и быстрее, сильная — качественнее, средняя — компромисс по качеству/цене.\n")

	if err := writeDay5Report(*reportPath, *prompt, results, fastest, bestQuality, efficient); err != nil {
		return fmt.Errorf("failed to write report: %w", err)
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)

	return nil
}

func callOpenRouter(apiKey, model string, messages []message, maxTokens int, temperature *float64, stop []string, title string) (string, error) {
	result, err := callOpenRouterDetailed(apiKey, model, messages, maxTokens, temperature, stop, title)
	if err != nil {
		return "", err
	}
	return result.Answer, nil
}

func callOpenRouterDetailed(apiKey, model string, messages []message, maxTokens int, temperature *float64, stop []string, title string) (openRouterResult, error) {
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
		return openRouterResult{}, fmt.Errorf("failed to encode request: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, openRouterURL, bytes.NewReader(reqBody))
	if err != nil {
		return openRouterResult{}, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("HTTP-Referer", "https://localhost")
	req.Header.Set("X-Title", title)

	client := &http.Client{Timeout: 60 * time.Second}
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return openRouterResult{}, fmt.Errorf("request failed: %w", err)
	}
	latency := time.Since(start)
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return openRouterResult{}, fmt.Errorf("failed to read response: %w", err)
	}

	var out chatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return openRouterResult{}, fmt.Errorf("invalid JSON response: %w\nraw: %s", err, string(raw))
	}

	if resp.StatusCode >= 400 {
		if out.Error != nil && out.Error.Message != "" {
			return openRouterResult{}, fmt.Errorf("API error (%s): %s", resp.Status, out.Error.Message)
		}
		return openRouterResult{}, fmt.Errorf("API error (%s): %s", resp.Status, string(raw))
	}

	if len(out.Choices) == 0 {
		return openRouterResult{}, fmt.Errorf("no choices in response")
	}

	usage := out.Usage
	if usage.TotalTokens == 0 && (usage.PromptTokens > 0 || usage.CompletionTokens > 0) {
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}

	return openRouterResult{
		Answer:  strings.TrimSpace(out.Choices[0].Message.Content),
		Model:   strings.TrimSpace(out.Model),
		Usage:   usage,
		Latency: latency,
	}, nil
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

func parseCSVList(csv string) []string {
	parts := strings.Split(csv, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		v := strings.TrimSpace(strings.ToLower(part))
		if v != "" {
			out = append(out, v)
		}
	}
	return out
}

func qualityScore(answer string, keywords []string) int {
	if strings.TrimSpace(answer) == "" {
		return 0
	}

	lower := strings.ToLower(answer)
	score := 0

	if len(keywords) > 0 {
		perKeyword := 60 / len(keywords)
		remainder := 60 - (perKeyword * len(keywords))
		matched := 0
		for _, kw := range keywords {
			if strings.Contains(lower, kw) {
				matched++
				score += perKeyword
			}
		}
		if matched == len(keywords) {
			score += remainder
		}
	}

	bulletCount := strings.Count(answer, "\n-") + strings.Count(answer, "\n*")
	if bulletCount >= 3 {
		score += 20
	} else if bulletCount == 2 {
		score += 12
	} else if bulletCount == 1 {
		score += 6
	}

	if strings.Contains(lower, "рекомендац") || strings.Contains(lower, "recommend") {
		score += 20
	}

	if score > 100 {
		return 100
	}
	if score < 0 {
		return 0
	}
	return score
}

func fetchModelMetadata(apiKey string) (map[string]modelInfo, error) {
	req, err := http.NewRequest(http.MethodGet, "https://openrouter.ai/api/v1/models", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create models request: %w", err)
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("models request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed reading models response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("models API error (%s): %s", resp.Status, string(raw))
	}

	var parsed modelsResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse models response: %w", err)
	}

	out := make(map[string]modelInfo, len(parsed.Data))
	for _, item := range parsed.Data {
		if strings.TrimSpace(item.ID) != "" {
			out[item.ID] = item
		}
	}
	return out, nil
}

func estimateCostUSD(usage usageStats, meta modelInfo) (float64, bool) {
	promptPrice, hasPrompt := parseFloat(meta.Pricing.Prompt)
	completionPrice, hasCompletion := parseFloat(meta.Pricing.Completion)
	requestPrice, hasRequest := parseFloat(meta.Pricing.Request)

	if !hasPrompt && !hasCompletion && !hasRequest {
		return 0, false
	}

	total := 0.0
	if hasPrompt {
		total += float64(usage.PromptTokens) * promptPrice
	}
	if hasCompletion {
		total += float64(usage.CompletionTokens) * completionPrice
	}
	if hasRequest {
		total += requestPrice
	}
	return total, true
}

func parseFloat(value string) (float64, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, false
	}
	return parsed, true
}

func pickFastestModel(results []benchmarkResult) benchmarkResult {
	best := results[0]
	for _, r := range results[1:] {
		if r.Latency < best.Latency {
			best = r
		}
	}
	return best
}

func pickBestQualityModel(results []benchmarkResult) benchmarkResult {
	best := results[0]
	for _, r := range results[1:] {
		if r.QualityScore > best.QualityScore {
			best = r
		}
	}
	return best
}

func pickMostEfficientModel(results []benchmarkResult) benchmarkResult {
	anyCost := false
	for _, r := range results {
		if r.CostKnown {
			anyCost = true
			break
		}
	}

	best := results[0]
	for _, r := range results[1:] {
		if anyCost && r.CostKnown && best.CostKnown {
			if r.CostUSD < best.CostUSD {
				best = r
			}
			continue
		}
		if anyCost && r.CostKnown && !best.CostKnown {
			best = r
			continue
		}
		if r.Usage.TotalTokens < best.Usage.TotalTokens {
			best = r
		}
	}
	return best
}

func writeDay5Report(path, prompt string, results []benchmarkResult, fastest, bestQuality, efficient benchmarkResult) error {
	var b strings.Builder
	b.WriteString("# Day 5 Results: Model Versions\n\n")
	b.WriteString("## Prompt\n")
	b.WriteString(prompt + "\n\n")
	b.WriteString("## Models\n\n")

	for _, r := range results {
		b.WriteString("### " + strings.ToUpper(r.Tier) + " model: `" + r.ModelID + "`\n")
		b.WriteString("- Latency: `" + r.Latency.Round(time.Millisecond).String() + "`\n")
		b.WriteString(fmt.Sprintf("- Tokens: `prompt=%d completion=%d total=%d`\n", r.Usage.PromptTokens, r.Usage.CompletionTokens, r.Usage.TotalTokens))
		if r.CostKnown {
			b.WriteString(fmt.Sprintf("- Cost: `$%.6f`\n", r.CostUSD))
		} else {
			b.WriteString("- Cost: `N/A`\n")
		}
		b.WriteString(fmt.Sprintf("- Quality score: `%d/100`\n", r.QualityScore))
		b.WriteString("- OpenRouter: " + r.ModelURL + "\n")
		if r.HuggingFaceURL != "" {
			b.WriteString("- HuggingFace: " + r.HuggingFaceURL + "\n")
		}
		b.WriteString("\n")
		b.WriteString("Answer:\n")
		b.WriteString("```text\n" + strings.TrimSpace(r.Answer) + "\n```\n\n")
	}

	b.WriteString("## Comparison\n")
	b.WriteString(fmt.Sprintf("- Quality winner: `%s (%s)`\n", strings.ToUpper(bestQuality.Tier), bestQuality.ModelID))
	b.WriteString(fmt.Sprintf("- Speed winner: `%s (%s)`\n", strings.ToUpper(fastest.Tier), fastest.ModelID))
	b.WriteString(fmt.Sprintf("- Efficiency winner: `%s (%s)`\n", strings.ToUpper(efficient.Tier), efficient.ModelID))
	b.WriteString("\n")
	b.WriteString("Short conclusion: weak models are often cheaper/faster, strong models tend to give better quality, and mid-tier models are a practical balance.\n")

	return os.WriteFile(path, []byte(b.String()), 0o644)
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
	return "Usage:\n  openrouter-cli [flags]\n  openrouter-cli agent [flags]\n  openrouter-cli day3 [flags]\n  openrouter-cli day4 [flags]\n  openrouter-cli day5 [flags]\n  openrouter-cli day8 [flags]\n  openrouter-cli day9 [flags]\n  openrouter-cli day10 [flags]\n  openrouter-cli day11 [flags]\n  openrouter-cli day12 [flags]\n  openrouter-cli day13 [flags]\n  openrouter-cli day14 [flags]\n  openrouter-cli day15 [flags]\n  openrouter-cli day16 [flags]\n  openrouter-cli day17 [flags]\n  openrouter-cli day18 [flags]\n  openrouter-cli day19 [flags]\n  openrouter-cli day20 [flags]\n\nUse `openrouter-cli --help` for chat flags, `openrouter-cli agent --help` for Agent flags, `openrouter-cli day3 --help` for Day 3 flags, `openrouter-cli day4 --help` for Day 4 flags, `openrouter-cli day5 --help` for Day 5 flags, `openrouter-cli day8 --help` for Day 8 flags, `openrouter-cli day9 --help` for Day 9 flags, `openrouter-cli day10 --help` for Day 10 flags, `openrouter-cli day11 --help` for Day 11 flags, `openrouter-cli day12 --help` for Day 12 flags, `openrouter-cli day13 --help` for Day 13 flags, `openrouter-cli day14 --help` for Day 14 flags, `openrouter-cli day15 --help` for Day 15 flags, `openrouter-cli day16 --help` for Day 16 flags, `openrouter-cli day17 --help` for Day 17 flags, `openrouter-cli day18 --help` for Day 18 flags, `openrouter-cli day19 --help` for Day 19 flags, and `openrouter-cli day20 --help` for Day 20 flags."
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
	fmt.Println("  agent               Run encapsulated LLM agent (single prompt or interactive)")
	fmt.Println("  day3                Run four reasoning strategies and compare results")
	fmt.Println("  day4                Run same prompt with different temperatures and compare")
	fmt.Println("  day5                Compare weak/mid/strong models by quality, speed, and cost")
	fmt.Println("  day8                Show token growth on short/long/overflow dialogues")
	fmt.Println("  day9                Compare full history vs compressed context")
	fmt.Println("  day10               Compare sliding/facts/branching context strategies")
	fmt.Println("  day11               Run layered memory model (short/working/long)")
	fmt.Println("  day12               Run personalized assistant on top of memory layers")
	fmt.Println("  day13               Run formal task state machine (planning->execution->validation->done)")
	fmt.Println("  day14               Enforce invariants and refuse conflicting requests")
	fmt.Println("  day15               Enforce controlled lifecycle with explicit state transitions")
	fmt.Println("  day16               Connect to MCP and list available tools")
	fmt.Println("  day17               Use custom MCP tool (API-backed) from an agent")
	fmt.Println("  day18               Run scheduler MCP tool with periodic summary")
	fmt.Println("  day19               Run MCP pipeline (search -> summarize -> save)")
	fmt.Println("  day20               Orchestrate multiple MCP servers in a long flow")
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

func printDay5Usage() {
	fmt.Println("Usage: openrouter-cli day5 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -weak-model string      Weak model ID")
	fmt.Println("  -mid-model string       Mid model ID")
	fmt.Println("  -strong-model string    Strong model ID")
	fmt.Println("  -prompt string          Prompt for all compared models")
	fmt.Println("  -quality-keywords string  Comma-separated quality markers")
	fmt.Println("  -temperature float      Temperature for all compared models")
	fmt.Println("  -max-tokens int         Maximum response tokens")
	fmt.Println("  -include-cost bool      Calculate cost if pricing metadata is available")
	fmt.Println("  -report string          Markdown report output path")
	fmt.Println("  -help                   Show help")
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
