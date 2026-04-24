package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

type day29ChatOptions struct {
	Temperature float64 `json:"temperature,omitempty"`
	NumPredict  int     `json:"num_predict,omitempty"`
	NumCtx      int     `json:"num_ctx,omitempty"`
}

type day29ChatRequest struct {
	Model    string               `json:"model"`
	Messages []day27OllamaMessage `json:"messages"`
	Stream   bool                 `json:"stream"`
	Options  day29ChatOptions     `json:"options,omitempty"`
}

type day29Run struct {
	Answer         string
	PromptTokens   int
	ResponseTokens int
	TotalTokens    int
	Latency        time.Duration
	LoadDuration   time.Duration
	ModelSizeBytes int64
	ModelVRAMBytes int64
}

type day29ModelTagDetails struct {
	Family            string `json:"family"`
	ParameterSize     string `json:"parameter_size"`
	QuantizationLevel string `json:"quantization_level"`
}

type day29ModelTag struct {
	Name    string               `json:"name"`
	Model   string               `json:"model"`
	Size    int64                `json:"size"`
	Details day29ModelTagDetails `json:"details"`
}

type day29TagsResponse struct {
	Models []day29ModelTag `json:"models"`
}

type day29PSModel struct {
	Name     string `json:"name"`
	Model    string `json:"model"`
	Size     int64  `json:"size"`
	SizeVRAM int64  `json:"size_vram"`
}

type day29PSResponse struct {
	Models []day29PSModel `json:"models"`
}

type day29ProfileConfig struct {
	Name          string
	Model         string
	SystemPrompt  string
	Temperature   float64
	MaxTokens     int
	ContextWindow int
	Quantization  string
	ParameterSize string
}

type day29QuestionBenchmark struct {
	ID               string
	Question         string
	Retrieved        []day22RetrievedChunk
	BaselineRuns     []day29Run
	OptimizedRuns    []day29Run
	BaselineScore    int
	OptimizedScore   int
	BaselineStable   bool
	OptimizedStable  bool
	BaselineLatency  time.Duration
	OptimizedLatency time.Duration
}

type day29ProfileMetrics struct {
	AvgQuality       int
	AvgLatency       time.Duration
	Stability        int
	AvgPromptTokens  int
	AvgResponseToken int
	AvgTotalTokens   int
	AvgLoadDuration  time.Duration
	MaxModelSize     int64
	MaxModelVRAM     int64
}

type day29RunResult struct {
	IndexPath        string
	IndexStrategy    string
	IndexChunks      int
	TopK             int
	Repeats          int
	ControlsFile     string
	LocalBaseURL     string
	LocalVersion     string
	OptimizationNote string
	BaselineProfile  day29ProfileConfig
	OptimizedProfile day29ProfileConfig
	SingleQuestion   string
	SingleRetrieved  []day22RetrievedChunk
	SingleBaseline   day29Run
	SingleOptimized  day29Run
	Benchmarks       []day29QuestionBenchmark
	Before           day29ProfileMetrics
	After            day29ProfileMetrics
}

func runDay29Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day29", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	question := fs.String("question", "Какие инструменты обнаружены в Day16?", "Single question for before/after comparison")
	indexPath := fs.String("index", "DAY21_INDEX_structured.json", "Path to Week 6 index JSON")
	topK := fs.Int("top-k", 4, "Retrieved chunks count")
	repeats := fs.Int("repeats", 1, "Runs per mode for stability estimation")
	runControls := fs.Bool("run-controls", true, "Run benchmark on control questions")
	maxControls := fs.Int("max-controls", 5, "Limit count of control questions")
	controlsPath := fs.String("controls-file", "DAY22_CONTROL_QUESTIONS.json", "Control questions JSON path")
	reportPath := fs.String("report", "DAY29_RESULTS.md", "Markdown report path")
	localBaseURL := fs.String("local-base-url", "http://127.0.0.1:11434", "Local Ollama base URL")
	baselineModel := fs.String("baseline-model", "qwen2.5:0.5b", "Baseline local model")
	optimizedModel := fs.String("optimized-model", "", "Optimized local model (optional; if empty and -try-quant=true, auto-select quantized variant)")
	tryQuant := fs.Bool("try-quant", true, "Try to auto-pick quantized model variant for optimized profile")
	timeoutSec := fs.Int("timeout-sec", 120, "HTTP timeout seconds")
	baselineTemp := fs.Float64("baseline-temperature", 0.2, "Baseline temperature")
	optimizedTemp := fs.Float64("optimized-temperature", 0.1, "Optimized temperature")
	baselineMaxTokens := fs.Int("baseline-max-tokens", 220, "Baseline max response tokens (num_predict)")
	optimizedMaxTokens := fs.Int("optimized-max-tokens", 160, "Optimized max response tokens (num_predict)")
	baselineCtx := fs.Int("baseline-context-window", 2048, "Baseline context window (num_ctx)")
	optimizedCtx := fs.Int("optimized-context-window", 4096, "Optimized context window (num_ctx)")
	baselineSystem := fs.String("baseline-system", "Ты локальный RAG ассистент. Отвечай строго на основе контекста.", "Baseline system prompt")
	optimizedSystem := fs.String("optimized-system", "Ты оптимизированный RAG-ассистент AGENT CHALLENGE. Отвечай только фактами из контекста. Формат: 1) Краткий ответ. 2) Подтверждения с [source|chunk_id]. 3) Если данных не хватает, напиши: не знаю, уточните запрос.", "Optimized system prompt")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day29 flags: %w", err)
	}
	if *help {
		printDay29Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day29 arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *topK <= 0 {
		return fmt.Errorf("top-k must be positive")
	}
	if *repeats <= 0 {
		return fmt.Errorf("repeats must be positive")
	}
	if *maxControls <= 0 {
		return fmt.Errorf("max-controls must be positive")
	}
	if *timeoutSec <= 0 {
		return fmt.Errorf("timeout-sec must be positive")
	}
	if *baselineTemp < 0 || *baselineTemp > 2 || *optimizedTemp < 0 || *optimizedTemp > 2 {
		return fmt.Errorf("temperature should be in [0..2]")
	}
	if *baselineMaxTokens <= 0 || *optimizedMaxTokens <= 0 {
		return fmt.Errorf("max tokens must be positive")
	}
	if *baselineCtx <= 0 || *optimizedCtx <= 0 {
		return fmt.Errorf("context window must be positive")
	}

	index, err := loadDay22Index(strings.TrimSpace(*indexPath))
	if err != nil {
		return err
	}
	if len(index.Chunks) == 0 {
		return fmt.Errorf("index has no chunks")
	}

	client := &http.Client{Timeout: time.Duration(*timeoutSec) * time.Second}
	base := strings.TrimRight(strings.TrimSpace(*localBaseURL), "/")
	version, err := day26CheckServer(client, base)
	if err != nil {
		return err
	}

	tags, err := day29ListLocalModels(client, base)
	if err != nil {
		return err
	}

	baselineName := strings.TrimSpace(*baselineModel)
	if baselineName == "" {
		return fmt.Errorf("baseline-model is empty")
	}
	if err := day26EnsureModelPresent(client, base, baselineName); err != nil {
		return err
	}

	optimizedName, optimizationNote := day29SelectOptimizedModel(baselineName, strings.TrimSpace(*optimizedModel), *tryQuant, tags)
	if optimizedName == "" {
		optimizedName = baselineName
	}
	if err := day26EnsureModelPresent(client, base, optimizedName); err != nil {
		return err
	}

	baselineTag := day29FindTagByName(tags, baselineName)
	optimizedTag := day29FindTagByName(tags, optimizedName)

	baselineProfile := day29ProfileConfig{
		Name:          "before",
		Model:         baselineName,
		SystemPrompt:  strings.TrimSpace(*baselineSystem),
		Temperature:   *baselineTemp,
		MaxTokens:     *baselineMaxTokens,
		ContextWindow: *baselineCtx,
		Quantization:  day29TagQuantization(baselineTag),
		ParameterSize: strings.TrimSpace(baselineTag.Details.ParameterSize),
	}
	optimizedProfile := day29ProfileConfig{
		Name:          "after",
		Model:         optimizedName,
		SystemPrompt:  strings.TrimSpace(*optimizedSystem),
		Temperature:   *optimizedTemp,
		MaxTokens:     *optimizedMaxTokens,
		ContextWindow: *optimizedCtx,
		Quantization:  day29TagQuantization(optimizedTag),
		ParameterSize: strings.TrimSpace(optimizedTag.Details.ParameterSize),
	}

	singleRetrieved := day28RetrieveLocal(index, strings.TrimSpace(*question), *topK)
	if len(singleRetrieved) == 0 {
		return fmt.Errorf("retrieval returned 0 chunks for single question")
	}

	singleBefore, err := day29AskLocalRAG(client, base, baselineProfile, strings.TrimSpace(*question), singleRetrieved)
	if err != nil {
		return fmt.Errorf("baseline single call failed: %w", err)
	}
	singleAfter, err := day29AskLocalRAG(client, base, optimizedProfile, strings.TrimSpace(*question), singleRetrieved)
	if err != nil {
		return fmt.Errorf("optimized single call failed: %w", err)
	}

	benchmarks := make([]day29QuestionBenchmark, 0)
	if *runControls {
		controls, err := ensureDay22Controls(strings.TrimSpace(*controlsPath))
		if err != nil {
			return err
		}
		if *maxControls < len(controls) {
			controls = controls[:*maxControls]
		}
		for _, cq := range controls {
			retrieved := day28RetrieveLocal(index, cq.Question, *topK)
			if len(retrieved) == 0 {
				continue
			}
			item := day29QuestionBenchmark{
				ID:            cq.ID,
				Question:      cq.Question,
				Retrieved:     retrieved,
				BaselineRuns:  make([]day29Run, 0, *repeats),
				OptimizedRuns: make([]day29Run, 0, *repeats),
			}
			for i := 0; i < *repeats; i++ {
				run, err := day29AskLocalRAG(client, base, baselineProfile, cq.Question, retrieved)
				if err != nil {
					return fmt.Errorf("baseline run failed for %s repeat %d: %w", cq.ID, i+1, err)
				}
				item.BaselineRuns = append(item.BaselineRuns, run)

				run, err = day29AskLocalRAG(client, base, optimizedProfile, cq.Question, retrieved)
				if err != nil {
					return fmt.Errorf("optimized run failed for %s repeat %d: %w", cq.ID, i+1, err)
				}
				item.OptimizedRuns = append(item.OptimizedRuns, run)
			}
			item.BaselineScore = day29AverageAnswerScore(item.BaselineRuns, cq.ExpectedTerms)
			item.OptimizedScore = day29AverageAnswerScore(item.OptimizedRuns, cq.ExpectedTerms)
			item.BaselineStable = day29IsStable(item.BaselineRuns)
			item.OptimizedStable = day29IsStable(item.OptimizedRuns)
			item.BaselineLatency = day29AverageLatency(item.BaselineRuns)
			item.OptimizedLatency = day29AverageLatency(item.OptimizedRuns)

			benchmarks = append(benchmarks, item)
		}
	}

	result := day29RunResult{
		IndexPath:        strings.TrimSpace(*indexPath),
		IndexStrategy:    index.Strategy,
		IndexChunks:      len(index.Chunks),
		TopK:             *topK,
		Repeats:          *repeats,
		ControlsFile:     strings.TrimSpace(*controlsPath),
		LocalBaseURL:     base,
		LocalVersion:     version,
		OptimizationNote: optimizationNote,
		BaselineProfile:  baselineProfile,
		OptimizedProfile: optimizedProfile,
		SingleQuestion:   strings.TrimSpace(*question),
		SingleRetrieved:  singleRetrieved,
		SingleBaseline:   singleBefore,
		SingleOptimized:  singleAfter,
		Benchmarks:       benchmarks,
	}
	result.Before = day29AggregateProfile(benchmarks, true)
	result.After = day29AggregateProfile(benchmarks, false)

	printDay29Result(result)
	if err := writeDay29Report(strings.TrimSpace(*reportPath), result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", strings.TrimSpace(*reportPath))
	return nil
}

func day29AskLocalRAG(client *http.Client, baseURL string, profile day29ProfileConfig, question string, retrieved []day22RetrievedChunk) (day29Run, error) {
	contextText := day22BuildContext(retrieved)
	userPrompt := "Контекст:\n" + contextText + "\n\nВопрос:\n" + strings.TrimSpace(question)
	messages := []day27OllamaMessage{
		{Role: "system", Content: profile.SystemPrompt},
		{Role: "user", Content: userPrompt},
	}
	resp, err := day29Chat(client, baseURL, profile.Model, messages, day29ChatOptions{
		Temperature: profile.Temperature,
		NumPredict:  profile.MaxTokens,
		NumCtx:      profile.ContextWindow,
	})
	if err != nil {
		return day29Run{}, err
	}
	answer := strings.TrimSpace(resp.Message.Content)
	if answer == "" {
		return day29Run{}, fmt.Errorf("empty model answer")
	}
	size, vram, _ := day29ReadModelResources(client, baseURL, profile.Model)
	return day29Run{
		Answer:         answer,
		PromptTokens:   resp.PromptEvalCount,
		ResponseTokens: resp.EvalCount,
		TotalTokens:    resp.PromptEvalCount + resp.EvalCount,
		Latency:        time.Duration(resp.TotalDuration),
		LoadDuration:   time.Duration(resp.LoadDuration),
		ModelSizeBytes: size,
		ModelVRAMBytes: vram,
	}, nil
}

func day29Chat(client *http.Client, baseURL, model string, messages []day27OllamaMessage, opts day29ChatOptions) (day27ChatResponse, error) {
	payload := day29ChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
		Options:  opts,
	}
	reqBody, err := json.Marshal(payload)
	if err != nil {
		return day27ChatResponse{}, fmt.Errorf("failed to encode chat request: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, baseURL+"/api/chat", bytes.NewReader(reqBody))
	if err != nil {
		return day27ChatResponse{}, fmt.Errorf("failed to build chat request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return day27ChatResponse{}, fmt.Errorf("chat request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return day27ChatResponse{}, fmt.Errorf("failed to read chat response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return day27ChatResponse{}, fmt.Errorf("chat API error (%s): %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	var out day27ChatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return day27ChatResponse{}, fmt.Errorf("invalid chat response JSON: %w", err)
	}
	if strings.TrimSpace(out.Error) != "" {
		return day27ChatResponse{}, fmt.Errorf("ollama returned error: %s", out.Error)
	}
	return out, nil
}

func day29ListLocalModels(client *http.Client, baseURL string) ([]day29ModelTag, error) {
	req, err := http.NewRequest(http.MethodGet, baseURL+"/api/tags", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to build tags request: %w", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to query local model list: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read tags response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("tags API error (%s): %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	var out day29TagsResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("invalid tags JSON: %w", err)
	}
	if len(out.Models) == 0 {
		return nil, fmt.Errorf("no local models found; pull one with ollama pull <model>")
	}
	return out.Models, nil
}

func day29ReadModelResources(client *http.Client, baseURL, model string) (int64, int64, error) {
	req, err := http.NewRequest(http.MethodGet, baseURL+"/api/ps", nil)
	if err != nil {
		return 0, 0, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, 0, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, 0, err
	}
	if resp.StatusCode >= 400 {
		return 0, 0, fmt.Errorf("ps API error (%s): %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	var out day29PSResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return 0, 0, err
	}
	for _, item := range out.Models {
		if strings.EqualFold(strings.TrimSpace(item.Name), strings.TrimSpace(model)) || strings.EqualFold(strings.TrimSpace(item.Model), strings.TrimSpace(model)) {
			return item.Size, item.SizeVRAM, nil
		}
	}
	return 0, 0, nil
}

func day29FindTagByName(tags []day29ModelTag, model string) day29ModelTag {
	model = strings.TrimSpace(model)
	for _, item := range tags {
		if strings.EqualFold(strings.TrimSpace(item.Name), model) || strings.EqualFold(strings.TrimSpace(item.Model), model) {
			return item
		}
	}
	return day29ModelTag{Name: model}
}

func day29TagQuantization(tag day29ModelTag) string {
	q := strings.TrimSpace(tag.Details.QuantizationLevel)
	if q != "" {
		return q
	}
	name := strings.ToLower(strings.TrimSpace(tag.Name))
	switch {
	case strings.Contains(name, "q2"):
		return "Q2"
	case strings.Contains(name, "q3"):
		return "Q3"
	case strings.Contains(name, "q4"):
		return "Q4"
	case strings.Contains(name, "q5"):
		return "Q5"
	case strings.Contains(name, "q6"):
		return "Q6"
	case strings.Contains(name, "q8"):
		return "Q8"
	default:
		return "unknown"
	}
}

func day29SelectOptimizedModel(baselineModel, explicitOptimized string, tryQuant bool, tags []day29ModelTag) (string, string) {
	baselineModel = strings.TrimSpace(baselineModel)
	explicitOptimized = strings.TrimSpace(explicitOptimized)
	if explicitOptimized != "" {
		return explicitOptimized, "optimized model is explicitly set by -optimized-model"
	}
	if !tryQuant {
		return baselineModel, "quantization auto-pick disabled by -try-quant=false"
	}

	baselineTag := day29FindTagByName(tags, baselineModel)
	baseFamily := strings.TrimSpace(baselineTag.Details.Family)
	basePrefix := strings.SplitN(strings.ToLower(baselineModel), ":", 2)[0]
	baseQuant := strings.ToUpper(day29TagQuantization(baselineTag))

	candidates := make([]day29ModelTag, 0)
	for _, tag := range tags {
		name := strings.TrimSpace(tag.Name)
		if name == "" || strings.EqualFold(name, baselineModel) {
			continue
		}
		tagFamily := strings.TrimSpace(tag.Details.Family)
		tagPrefix := strings.SplitN(strings.ToLower(name), ":", 2)[0]
		sameFamily := baseFamily != "" && strings.EqualFold(baseFamily, tagFamily)
		samePrefix := basePrefix != "" && basePrefix == tagPrefix
		if !sameFamily && !samePrefix {
			continue
		}
		q := strings.ToUpper(day29TagQuantization(tag))
		if q == "" || q == "UNKNOWN" {
			continue
		}
		if baseQuant != "" && q == baseQuant {
			continue
		}
		candidates = append(candidates, tag)
	}
	if len(candidates) == 0 {
		return baselineModel, "quantized alternative not found locally; optimized uses baseline model with tuned params + prompt"
	}

	sort.Slice(candidates, func(i, j int) bool {
		a, b := candidates[i], candidates[j]
		if a.Size > 0 && b.Size > 0 && a.Size != b.Size {
			return a.Size < b.Size
		}
		return strings.ToLower(a.Name) < strings.ToLower(b.Name)
	})
	picked := candidates[0]
	return picked.Name, fmt.Sprintf("auto-selected quantized local model: %s (%s)", picked.Name, day29TagQuantization(picked))
}

func day29AverageAnswerScore(runs []day29Run, expectedTerms []string) int {
	if len(runs) == 0 {
		return 0
	}
	sum := 0
	for _, run := range runs {
		sum += day22ScoreAnswer(run.Answer, expectedTerms)
	}
	return sum / len(runs)
}

func day29AverageLatency(runs []day29Run) time.Duration {
	if len(runs) == 0 {
		return 0
	}
	total := time.Duration(0)
	for _, run := range runs {
		total += run.Latency
	}
	return total / time.Duration(len(runs))
}

func day29IsStable(runs []day29Run) bool {
	if len(runs) <= 1 {
		return true
	}
	ref := day28NormalizeAnswer(runs[0].Answer)
	for i := 1; i < len(runs); i++ {
		if day28NormalizeAnswer(runs[i].Answer) != ref {
			return false
		}
	}
	return true
}

func day29AggregateProfile(benchmarks []day29QuestionBenchmark, baseline bool) day29ProfileMetrics {
	if len(benchmarks) == 0 {
		return day29ProfileMetrics{}
	}
	qualitySum := 0
	latencySum := time.Duration(0)
	stableCount := 0
	promptTokenSum := 0
	responseTokenSum := 0
	totalTokenSum := 0
	loadSum := time.Duration(0)
	runCount := 0
	var maxSize int64
	var maxVRAM int64

	for _, b := range benchmarks {
		var score int
		var stable bool
		var latency time.Duration
		var runs []day29Run
		if baseline {
			score = b.BaselineScore
			stable = b.BaselineStable
			latency = b.BaselineLatency
			runs = b.BaselineRuns
		} else {
			score = b.OptimizedScore
			stable = b.OptimizedStable
			latency = b.OptimizedLatency
			runs = b.OptimizedRuns
		}
		qualitySum += score
		latencySum += latency
		if stable {
			stableCount++
		}
		for _, run := range runs {
			promptTokenSum += run.PromptTokens
			responseTokenSum += run.ResponseTokens
			totalTokenSum += run.TotalTokens
			loadSum += run.LoadDuration
			runCount++
			if run.ModelSizeBytes > maxSize {
				maxSize = run.ModelSizeBytes
			}
			if run.ModelVRAMBytes > maxVRAM {
				maxVRAM = run.ModelVRAMBytes
			}
		}
	}

	out := day29ProfileMetrics{
		AvgQuality:   qualitySum / len(benchmarks),
		AvgLatency:   latencySum / time.Duration(len(benchmarks)),
		Stability:    (stableCount * 100) / len(benchmarks),
		MaxModelSize: maxSize,
		MaxModelVRAM: maxVRAM,
	}
	if runCount > 0 {
		out.AvgPromptTokens = promptTokenSum / runCount
		out.AvgResponseToken = responseTokenSum / runCount
		out.AvgTotalTokens = totalTokenSum / runCount
		out.AvgLoadDuration = loadSum / time.Duration(runCount)
	}
	return out
}

func printDay29Result(result day29RunResult) {
	fmt.Println("=== Day 29: Local LLM Optimization ===")
	fmt.Printf("index=%s strategy=%s chunks=%d top_k=%d repeats=%d\n", result.IndexPath, result.IndexStrategy, result.IndexChunks, result.TopK, result.Repeats)
	fmt.Printf("local_server=%s local_version=%s\n", result.LocalBaseURL, result.LocalVersion)
	fmt.Printf("before_model=%s before_quant=%s before_temp=%.2f before_max_tokens=%d before_ctx=%d\n",
		result.BaselineProfile.Model,
		result.BaselineProfile.Quantization,
		result.BaselineProfile.Temperature,
		result.BaselineProfile.MaxTokens,
		result.BaselineProfile.ContextWindow,
	)
	fmt.Printf("after_model=%s after_quant=%s after_temp=%.2f after_max_tokens=%d after_ctx=%d\n",
		result.OptimizedProfile.Model,
		result.OptimizedProfile.Quantization,
		result.OptimizedProfile.Temperature,
		result.OptimizedProfile.MaxTokens,
		result.OptimizedProfile.ContextWindow,
	)
	fmt.Printf("optimization_note=%s\n", result.OptimizationNote)
	fmt.Printf("single_question=%s\n", result.SingleQuestion)
	fmt.Printf("single_before_answer=%s\n", sanitizeCodeFences(result.SingleBaseline.Answer))
	fmt.Printf("single_after_answer=%s\n", sanitizeCodeFences(result.SingleOptimized.Answer))
	fmt.Printf("benchmarks=%d\n", len(result.Benchmarks))
	fmt.Printf("quality_before=%d quality_after=%d\n", result.Before.AvgQuality, result.After.AvgQuality)
	fmt.Printf("speed_before=%s speed_after=%s\n", result.Before.AvgLatency.Round(time.Millisecond), result.After.AvgLatency.Round(time.Millisecond))
	fmt.Printf("tokens_before=%d tokens_after=%d\n", result.Before.AvgTotalTokens, result.After.AvgTotalTokens)
	fmt.Printf("stability_before=%d%% stability_after=%d%%\n", result.Before.Stability, result.After.Stability)
}

func writeDay29Report(path string, result day29RunResult) error {
	var b strings.Builder
	b.WriteString("# Day 29 Results: Local LLM Optimization\n\n")
	b.WriteString("## Setup\n")
	b.WriteString(fmt.Sprintf("- index: `%s`\n", result.IndexPath))
	b.WriteString(fmt.Sprintf("- index strategy: `%s`\n", result.IndexStrategy))
	b.WriteString(fmt.Sprintf("- chunks: `%d`\n", result.IndexChunks))
	b.WriteString(fmt.Sprintf("- server: `%s` (v%s)\n", result.LocalBaseURL, result.LocalVersion))
	b.WriteString(fmt.Sprintf("- retrieval: `local lexical top-k=%d`\n", result.TopK))
	b.WriteString(fmt.Sprintf("- repeats: `%d`\n", result.Repeats))
	b.WriteString(fmt.Sprintf("- optimization note: `%s`\n\n", escapeDay22Table(result.OptimizationNote)))

	b.WriteString("## Before (baseline)\n")
	b.WriteString(fmt.Sprintf("- model: `%s`\n", result.BaselineProfile.Model))
	b.WriteString(fmt.Sprintf("- quantization: `%s`\n", result.BaselineProfile.Quantization))
	if result.BaselineProfile.ParameterSize != "" {
		b.WriteString(fmt.Sprintf("- parameter size: `%s`\n", result.BaselineProfile.ParameterSize))
	}
	b.WriteString(fmt.Sprintf("- temperature: `%.2f`\n", result.BaselineProfile.Temperature))
	b.WriteString(fmt.Sprintf("- max tokens (num_predict): `%d`\n", result.BaselineProfile.MaxTokens))
	b.WriteString(fmt.Sprintf("- context window (num_ctx): `%d`\n", result.BaselineProfile.ContextWindow))
	b.WriteString(fmt.Sprintf("- prompt template: `%s`\n\n", escapeDay22Table(result.BaselineProfile.SystemPrompt)))

	b.WriteString("## After (optimized)\n")
	b.WriteString(fmt.Sprintf("- model: `%s`\n", result.OptimizedProfile.Model))
	b.WriteString(fmt.Sprintf("- quantization: `%s`\n", result.OptimizedProfile.Quantization))
	if result.OptimizedProfile.ParameterSize != "" {
		b.WriteString(fmt.Sprintf("- parameter size: `%s`\n", result.OptimizedProfile.ParameterSize))
	}
	b.WriteString(fmt.Sprintf("- temperature: `%.2f`\n", result.OptimizedProfile.Temperature))
	b.WriteString(fmt.Sprintf("- max tokens (num_predict): `%d`\n", result.OptimizedProfile.MaxTokens))
	b.WriteString(fmt.Sprintf("- context window (num_ctx): `%d`\n", result.OptimizedProfile.ContextWindow))
	b.WriteString(fmt.Sprintf("- prompt template: `%s`\n\n", escapeDay22Table(result.OptimizedProfile.SystemPrompt)))

	b.WriteString("## Single Question\n")
	b.WriteString(fmt.Sprintf("Question: %s\n\n", escapeDay22Table(result.SingleQuestion)))
	b.WriteString("Retrieved chunks:\n")
	for _, chunk := range result.SingleRetrieved {
		b.WriteString(fmt.Sprintf("- `%s` section=`%s` chunk_id=`%s` score=%.4f\n", chunk.Source, chunk.Section, chunk.ChunkID, chunk.Score))
	}
	b.WriteString("\nBefore answer:\n")
	b.WriteString("```text\n" + sanitizeCodeFences(strings.TrimSpace(result.SingleBaseline.Answer)) + "\n```\n")
	b.WriteString("\nAfter answer:\n")
	b.WriteString("```text\n" + sanitizeCodeFences(strings.TrimSpace(result.SingleOptimized.Answer)) + "\n```\n")

	if len(result.Benchmarks) > 0 {
		b.WriteString("\n## Benchmark Table\n")
		b.WriteString("| ID | Question | Quality Before | Quality After | Latency Before | Latency After | Tokens Before | Tokens After | Stable Before | Stable After |\n")
		b.WriteString("| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |\n")
		for _, item := range result.Benchmarks {
			beforeTokens := day29AverageTotalTokens(item.BaselineRuns)
			afterTokens := day29AverageTotalTokens(item.OptimizedRuns)
			b.WriteString(fmt.Sprintf("| %s | %s | %d | %d | %s | %s | %d | %d | %t | %t |\n",
				item.ID,
				escapeDay22Table(item.Question),
				item.BaselineScore,
				item.OptimizedScore,
				item.BaselineLatency.Round(time.Millisecond),
				item.OptimizedLatency.Round(time.Millisecond),
				beforeTokens,
				afterTokens,
				item.BaselineStable,
				item.OptimizedStable,
			))
		}
	}

	b.WriteString("\n## Comparison\n")
	b.WriteString(fmt.Sprintf("- quality before: `%d`\n", result.Before.AvgQuality))
	b.WriteString(fmt.Sprintf("- quality after: `%d`\n", result.After.AvgQuality))
	b.WriteString(fmt.Sprintf("- speed before: `%s`\n", result.Before.AvgLatency.Round(time.Millisecond)))
	b.WriteString(fmt.Sprintf("- speed after: `%s`\n", result.After.AvgLatency.Round(time.Millisecond)))
	b.WriteString(fmt.Sprintf("- prompt tokens before: `%d`\n", result.Before.AvgPromptTokens))
	b.WriteString(fmt.Sprintf("- prompt tokens after: `%d`\n", result.After.AvgPromptTokens))
	b.WriteString(fmt.Sprintf("- response tokens before: `%d`\n", result.Before.AvgResponseToken))
	b.WriteString(fmt.Sprintf("- response tokens after: `%d`\n", result.After.AvgResponseToken))
	b.WriteString(fmt.Sprintf("- total tokens before: `%d`\n", result.Before.AvgTotalTokens))
	b.WriteString(fmt.Sprintf("- total tokens after: `%d`\n", result.After.AvgTotalTokens))
	b.WriteString(fmt.Sprintf("- load duration before: `%s`\n", result.Before.AvgLoadDuration.Round(time.Millisecond)))
	b.WriteString(fmt.Sprintf("- load duration after: `%s`\n", result.After.AvgLoadDuration.Round(time.Millisecond)))
	b.WriteString(fmt.Sprintf("- max model RAM before: `%s`\n", day29HumanBytes(result.Before.MaxModelSize)))
	b.WriteString(fmt.Sprintf("- max model RAM after: `%s`\n", day29HumanBytes(result.After.MaxModelSize)))
	b.WriteString(fmt.Sprintf("- max model VRAM before: `%s`\n", day29HumanBytes(result.Before.MaxModelVRAM)))
	b.WriteString(fmt.Sprintf("- max model VRAM after: `%s`\n", day29HumanBytes(result.After.MaxModelVRAM)))
	b.WriteString(fmt.Sprintf("- stability before: `%d%%`\n", result.Before.Stability))
	b.WriteString(fmt.Sprintf("- stability after: `%d%%`\n", result.After.Stability))
	b.WriteString("\nConclusion: optimized profile applies parameter tuning (`temperature`, `num_predict`, `num_ctx`), prompt specialization, and optional quantized model selection when locally available.\n")

	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func day29AverageTotalTokens(runs []day29Run) int {
	if len(runs) == 0 {
		return 0
	}
	total := 0
	for _, run := range runs {
		total += run.TotalTokens
	}
	return total / len(runs)
}

func day29HumanBytes(n int64) string {
	if n <= 0 {
		return "n/a"
	}
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for val := n / unit; val >= unit; val /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

func printDay29Usage() {
	fmt.Println("Usage: openrouter-cli day29 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -question string                  Single comparison question")
	fmt.Println("  -index string                     Path to Week 6 index JSON")
	fmt.Println("  -top-k int                        Retrieved chunks count")
	fmt.Println("  -repeats int                      Runs per mode for stability")
	fmt.Println("  -run-controls                     Run control benchmark")
	fmt.Println("  -max-controls int                 Max control questions")
	fmt.Println("  -controls-file string             Control questions JSON")
	fmt.Println("  -report string                    Markdown report path")
	fmt.Println("  -local-base-url string            Local Ollama base URL")
	fmt.Println("  -baseline-model string            Baseline local model")
	fmt.Println("  -optimized-model string           Optimized local model")
	fmt.Println("  -try-quant                        Auto-pick quantized optimized model")
	fmt.Println("  -baseline-temperature float       Baseline temperature")
	fmt.Println("  -optimized-temperature float      Optimized temperature")
	fmt.Println("  -baseline-max-tokens int          Baseline max response tokens")
	fmt.Println("  -optimized-max-tokens int         Optimized max response tokens")
	fmt.Println("  -baseline-context-window int      Baseline context window")
	fmt.Println("  -optimized-context-window int     Optimized context window")
	fmt.Println("  -baseline-system string           Baseline system prompt")
	fmt.Println("  -optimized-system string          Optimized system prompt")
	fmt.Println("  -timeout-sec int                  HTTP timeout seconds")
	fmt.Println("  -help                             Show help")
}
