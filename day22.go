package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"
)

type day22RetrievedChunk struct {
	ChunkID string
	Source  string
	Title   string
	Section string
	Score   float64
	Text    string
}

type day22ModeResult struct {
	Mode      string
	Answer    string
	Usage     usageStats
	Latency   time.Duration
	Retrieved []day22RetrievedChunk
}

type day22Comparison struct {
	Question string
	NoRAG    day22ModeResult
	RAG      day22ModeResult
}

type day22ControlQuestion struct {
	ID              string   `json:"id"`
	Question        string   `json:"question"`
	Expectation     string   `json:"expectation"`
	ExpectedTerms   []string `json:"expected_terms"`
	ExpectedSources []string `json:"expected_sources"`
}

type day22ControlResult struct {
	Question         day22ControlQuestion
	NoRAG            day22ModeResult
	RAG              day22ModeResult
	NoRAGAnswerScore int
	RAGAnswerScore   int
	RAGSourceScore   int
}

type day22RunResult struct {
	IndexPath          string
	IndexStrategy      string
	IndexChunks        int
	QuestionComparison day22Comparison
	ControlFilePath    string
	ControlResults     []day22ControlResult
	NoRAGAvgScore      int
	RAGAvgAnswerScore  int
	RAGAvgSourceScore  int
	RAGAvgTotalScore   int
}

func runDay22Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day22", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	question := fs.String("question", "Какие инструменты были обнаружены в Day 16?", "Question for single comparison")
	indexPath := fs.String("index", "DAY21_INDEX_structured.json", "Path to Day21 index")
	topK := fs.Int("top-k", 4, "Top-K chunks for RAG retrieval")
	model := fs.String("model", getDefaultModel(), "OpenRouter chat model")
	embeddingModel := fs.String("embedding-model", defaultEmbeddingModel(), "OpenRouter embedding model")
	maxTokens := fs.Int("max-tokens", 220, "Max tokens for each answer")
	temperature := fs.Float64("temperature", 0.2, "Temperature for LLM answers")
	runControls := fs.Bool("run-controls", true, "Run 10 control questions")
	controlsPath := fs.String("controls-file", "DAY22_CONTROL_QUESTIONS.json", "Path to control questions JSON")
	reportPath := fs.String("report", "DAY22_RESULTS.md", "Markdown report path")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day22 flags: %w", err)
	}
	if *help {
		printDay22Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day22 arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *topK <= 0 {
		return fmt.Errorf("top-k must be positive")
	}
	if *maxTokens <= 0 {
		return fmt.Errorf("max-tokens must be positive")
	}

	index, err := loadDay22Index(strings.TrimSpace(*indexPath))
	if err != nil {
		return err
	}
	if len(index.Chunks) == 0 {
		return fmt.Errorf("index has no chunks")
	}
	if len(index.Chunks[0].Embedding) == 0 {
		return fmt.Errorf("index has no embeddings; run day21 without -no-embed")
	}

	apiKey := getAPIKey()
	ctx := context.Background()
	temp := *temperature

	comparison, err := runDay22Comparison(ctx, apiKey, *model, *embeddingModel, index, strings.TrimSpace(*question), nil, *topK, *maxTokens, &temp)
	if err != nil {
		return err
	}

	controls, err := ensureDay22Controls(strings.TrimSpace(*controlsPath))
	if err != nil {
		return err
	}

	controlResults := make([]day22ControlResult, 0)
	if *runControls {
		for _, cq := range controls {
			cmp, err := runDay22Comparison(ctx, apiKey, *model, *embeddingModel, index, cq.Question, cq.ExpectedSources, *topK, *maxTokens, &temp)
			if err != nil {
				return fmt.Errorf("control question %s failed: %w", cq.ID, err)
			}
			noRAGScore := day22ScoreAnswer(cmp.NoRAG.Answer, cq.ExpectedTerms)
			ragScore := day22ScoreAnswer(cmp.RAG.Answer, cq.ExpectedTerms)
			sourceScore := day22ScoreSources(cmp.RAG.Retrieved, cq.ExpectedSources)
			controlResults = append(controlResults, day22ControlResult{
				Question:         cq,
				NoRAG:            cmp.NoRAG,
				RAG:              cmp.RAG,
				NoRAGAnswerScore: noRAGScore,
				RAGAnswerScore:   ragScore,
				RAGSourceScore:   sourceScore,
			})
		}
	}

	result := day22RunResult{
		IndexPath:          strings.TrimSpace(*indexPath),
		IndexStrategy:      index.Strategy,
		IndexChunks:        len(index.Chunks),
		QuestionComparison: comparison,
		ControlFilePath:    strings.TrimSpace(*controlsPath),
		ControlResults:     controlResults,
	}
	result.NoRAGAvgScore, result.RAGAvgAnswerScore, result.RAGAvgSourceScore, result.RAGAvgTotalScore = day22AggregateScores(controlResults)

	printDay22Result(result)
	if err := writeDay22Report(strings.TrimSpace(*reportPath), result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", strings.TrimSpace(*reportPath))
	return nil
}

func loadDay22Index(path string) (day21Index, error) {
	if path == "" {
		return day21Index{}, fmt.Errorf("index path is empty")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return day21Index{}, fmt.Errorf("failed to read index: %w", err)
	}
	var out day21Index
	if err := json.Unmarshal(data, &out); err != nil {
		return day21Index{}, fmt.Errorf("failed to parse index: %w", err)
	}
	return out, nil
}

func runDay22Comparison(ctx context.Context, apiKey, model, embeddingModel string, index day21Index, question string, sourceHints []string, topK, maxTokens int, temperature *float64) (day22Comparison, error) {
	question = strings.TrimSpace(question)
	if question == "" {
		return day22Comparison{}, fmt.Errorf("question is empty")
	}

	noRAG, err := day22AskNoRAG(apiKey, model, question, maxTokens, temperature)
	if err != nil {
		return day22Comparison{}, err
	}

	retrieved, err := day22RetrieveChunks(ctx, apiKey, embeddingModel, index, question, sourceHints, topK)
	if err != nil {
		return day22Comparison{}, err
	}

	rag, err := day22AskWithRAG(apiKey, model, question, retrieved, maxTokens, temperature)
	if err != nil {
		return day22Comparison{}, err
	}

	return day22Comparison{Question: question, NoRAG: noRAG, RAG: rag}, nil
}

func day22AskNoRAG(apiKey, model, question string, maxTokens int, temperature *float64) (day22ModeResult, error) {
	resp, err := callOpenRouterDetailed(
		apiKey,
		model,
		[]message{{Role: "user", Content: question}},
		maxTokens,
		temperature,
		nil,
		"openrouter-cli-day22-no-rag",
	)
	if err != nil {
		return day22ModeResult{}, fmt.Errorf("no-rag call failed: %w", err)
	}
	return day22ModeResult{
		Mode:    "no_rag",
		Answer:  resp.Answer,
		Usage:   resp.Usage,
		Latency: resp.Latency,
	}, nil
}

func day22AskWithRAG(apiKey, model, question string, chunks []day22RetrievedChunk, maxTokens int, temperature *float64) (day22ModeResult, error) {
	contextText := day22BuildContext(chunks)
	systemPrompt := "Ты отвечаешь на вопрос только на основе предоставленного контекста. Если данных не хватает, так и скажи. В конце укажи источники в формате [source]."
	userPrompt := "Контекст:\n" + contextText + "\n\nВопрос:\n" + question

	resp, err := callOpenRouterDetailed(
		apiKey,
		model,
		[]message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
		maxTokens,
		temperature,
		nil,
		"openrouter-cli-day22-rag",
	)
	if err != nil {
		return day22ModeResult{}, fmt.Errorf("rag call failed: %w", err)
	}

	return day22ModeResult{
		Mode:      "rag",
		Answer:    resp.Answer,
		Usage:     resp.Usage,
		Latency:   resp.Latency,
		Retrieved: chunks,
	}, nil
}

func day22BuildContext(chunks []day22RetrievedChunk) string {
	if len(chunks) == 0 {
		return "(нет релевантных чанков)"
	}
	var b strings.Builder
	for i, chunk := range chunks {
		b.WriteString(fmt.Sprintf("[%d] source=%s title=%s section=%s score=%.4f\n", i+1, chunk.Source, chunk.Title, chunk.Section, chunk.Score))
		b.WriteString(chunk.Text)
		b.WriteString("\n---\n")
	}
	return b.String()
}

func day22RetrieveChunks(ctx context.Context, apiKey, embeddingModel string, index day21Index, question string, sourceHints []string, topK int) ([]day22RetrievedChunk, error) {
	vectors, _, err := embedBatch(ctx, apiKey, embeddingModel, []string{question})
	if err != nil {
		return nil, fmt.Errorf("failed to embed question: %w", err)
	}
	queryVector := vectors[0]
	if len(queryVector) == 0 {
		return nil, fmt.Errorf("empty query embedding")
	}
	queryTokens := day22Tokens(question)

	scored := make([]day22RetrievedChunk, 0, len(index.Chunks))
	for _, chunk := range index.Chunks {
		if len(chunk.Embedding) == 0 {
			continue
		}
		cos := day22Cosine(queryVector, chunk.Embedding)
		lexical := day22LexicalScore(queryTokens, chunk)
		hintBonus := day22SourceHintBonus(sourceHints, chunk.Source, chunk.Title, chunk.Section)
		score := cos*0.75 + lexical*0.25 + hintBonus
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
		return nil, fmt.Errorf("no chunks with embeddings found")
	}
	sort.Slice(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })
	if topK > len(scored) {
		topK = len(scored)
	}
	return scored[:topK], nil
}

func day22Cosine(a, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	limit := len(a)
	if len(b) < limit {
		limit = len(b)
	}
	var dot, na, nb float64
	for i := 0; i < limit; i++ {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (sqrt(na) * sqrt(nb))
}

func day22Tokens(text string) []string {
	fields := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return (r < 'a' || r > 'z') && (r < 'а' || r > 'я') && (r < '0' || r > '9')
	})
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if len(f) >= 2 {
			out = append(out, f)
		}
	}
	return out
}

func day22LexicalScore(queryTokens []string, chunk day21ChunkRecord) float64 {
	if len(queryTokens) == 0 {
		return 0
	}
	corpus := strings.ToLower(chunk.Source + " " + chunk.Title + " " + chunk.Section + " " + chunk.Text)
	hits := 0
	seen := make(map[string]struct{})
	for _, token := range queryTokens {
		if _, ok := seen[token]; ok {
			continue
		}
		seen[token] = struct{}{}
		if strings.Contains(corpus, token) {
			hits++
		}
	}
	return float64(hits) / float64(len(seen))
}

func day22SourceHintBonus(hints []string, source, title, section string) float64 {
	if len(hints) == 0 {
		return 0
	}
	target := strings.ToLower(source + " " + title + " " + section)
	bonus := 0.0
	for _, hint := range hints {
		h := strings.ToLower(strings.TrimSpace(hint))
		if h == "" {
			continue
		}
		if strings.Contains(target, h) {
			bonus += 0.15
		}
	}
	if bonus > 0.45 {
		return 0.45
	}
	return bonus
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}

func day22ScoreAnswer(answer string, expectedTerms []string) int {
	if len(expectedTerms) == 0 {
		return 0
	}
	lower := strings.ToLower(answer)
	hit := 0
	for _, term := range expectedTerms {
		t := strings.ToLower(strings.TrimSpace(term))
		if t == "" {
			continue
		}
		if strings.Contains(lower, t) {
			hit++
		}
	}
	return (hit * 100) / len(expectedTerms)
}

func day22ScoreSources(chunks []day22RetrievedChunk, expectedSources []string) int {
	if len(expectedSources) == 0 {
		return 100
	}
	sourceSet := make(map[string]struct{})
	for _, chunk := range chunks {
		sourceSet[strings.ToLower(chunk.Source)] = struct{}{}
	}
	hit := 0
	for _, src := range expectedSources {
		s := strings.ToLower(strings.TrimSpace(src))
		if s == "" {
			continue
		}
		for existing := range sourceSet {
			if strings.Contains(existing, s) {
				hit++
				break
			}
		}
	}
	return (hit * 100) / len(expectedSources)
}

func day22AggregateScores(results []day22ControlResult) (int, int, int, int) {
	if len(results) == 0 {
		return 0, 0, 0, 0
	}
	noRAGSum := 0
	ragAnswerSum := 0
	ragSourceSum := 0
	ragTotalSum := 0
	for _, item := range results {
		noRAGSum += item.NoRAGAnswerScore
		ragAnswerSum += item.RAGAnswerScore
		ragSourceSum += item.RAGSourceScore
		ragTotalSum += (item.RAGAnswerScore*80 + item.RAGSourceScore*20) / 100
	}
	n := len(results)
	return noRAGSum / n, ragAnswerSum / n, ragSourceSum / n, ragTotalSum / n
}

func ensureDay22Controls(path string) ([]day22ControlQuestion, error) {
	if path == "" {
		return defaultDay22Controls(), nil
	}
	if _, err := os.Stat(path); err == nil {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("failed to read controls file: %w", err)
		}
		var out []day22ControlQuestion
		if err := json.Unmarshal(data, &out); err != nil {
			return nil, fmt.Errorf("failed to parse controls file: %w", err)
		}
		if len(out) == 0 {
			return nil, fmt.Errorf("controls file is empty")
		}
		return out, nil
	}
	defaults := defaultDay22Controls()
	data, err := json.MarshalIndent(defaults, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to encode default controls: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return nil, fmt.Errorf("failed to write controls file: %w", err)
	}
	return defaults, nil
}

func defaultDay22Controls() []day22ControlQuestion {
	return []day22ControlQuestion{
		{ID: "q1", Question: "Какие стратегии в Day10 получили качество 9/9?", Expectation: "Нужно назвать sliding_window и branching с качеством 9/9.", ExpectedTerms: []string{"sliding_window", "branching", "9/9"}, ExpectedSources: []string{"DAY10_RESULTS.md"}},
		{ID: "q2", Question: "Какая стратегия в Day10 имеет наименьшее total tokens и сколько именно?", Expectation: "Нужно ответить sliding_window и 11945 total tokens.", ExpectedTerms: []string{"sliding_window", "11945"}, ExpectedSources: []string{"DAY10_RESULTS.md"}},
		{ID: "q3", Question: "Какие слои памяти описаны в Day11?", Expectation: "Нужно перечислить short-term, working и long-term.", ExpectedTerms: []string{"short-term", "working", "long-term"}, ExpectedSources: []string{"DAY11_RESULTS.md"}},
		{ID: "q4", Question: "Какие три профиля перечислены в Day12?", Expectation: "Нужно назвать founder-brief, pm-table и dev-json.", ExpectedTerms: []string{"founder-brief", "pm-table", "dev-json"}, ExpectedSources: []string{"DAY12_RESULTS.md"}},
		{ID: "q5", Question: "Какой финальный stage и статус паузы в Day13?", Expectation: "Нужно ответить stage done и paused false.", ExpectedTerms: []string{"done", "false"}, ExpectedSources: []string{"DAY13_RESULTS.md"}},
		{ID: "q6", Question: "Сколько инвариантов в Day14 и прошёл ли конфликтный кейс?", Expectation: "Нужно указать invariants count 4 и conflict case passed true.", ExpectedTerms: []string{"4", "true"}, ExpectedSources: []string{"DAY14_RESULTS.md"}},
		{ID: "q7", Question: "Какой переход в Day15 блокируется без валидации?", Expectation: "Нужно сказать, что blocked jump to done without validation = true.", ExpectedTerms: []string{"done", "without validation", "true"}, ExpectedSources: []string{"DAY15_RESULTS.md"}},
		{ID: "q8", Question: "Какие инструменты обнаружены в Day16?", Expectation: "Нужно назвать system_time и upper_text.", ExpectedTerms: []string{"system_time", "upper_text"}, ExpectedSources: []string{"DAY16_RESULTS.md"}},
		{ID: "q9", Question: "Какой todo id вернул create_todo в Day17?", Expectation: "Нужно указать todo id 201.", ExpectedTerms: []string{"201", "create_todo"}, ExpectedSources: []string{"DAY17_RESULTS.md"}},
		{ID: "q10", Question: "Сколько серверов и шагов в Day20 orchestration?", Expectation: "Нужно указать servers 2 и steps 6.", ExpectedTerms: []string{"2", "6"}, ExpectedSources: []string{"DAY20_RESULTS.md"}},
	}
}

func printDay22Result(result day22RunResult) {
	fmt.Println("=== Day 22: First RAG Query ===")
	fmt.Printf("index=%s strategy=%s chunks=%d\n", result.IndexPath, result.IndexStrategy, result.IndexChunks)
	fmt.Printf("single_question=%s\n", result.QuestionComparison.Question)
	fmt.Printf("single_no_rag_tokens=%d single_rag_tokens=%d\n", result.QuestionComparison.NoRAG.Usage.TotalTokens, result.QuestionComparison.RAG.Usage.TotalTokens)
	fmt.Printf("controls=%d no_rag_avg=%d rag_answer_avg=%d rag_source_avg=%d rag_total_avg=%d\n",
		len(result.ControlResults),
		result.NoRAGAvgScore,
		result.RAGAvgAnswerScore,
		result.RAGAvgSourceScore,
		result.RAGAvgTotalScore,
	)
}

func writeDay22Report(path string, result day22RunResult) error {
	var b strings.Builder
	b.WriteString("# Day 22 Results: First RAG Query\n\n")
	b.WriteString(fmt.Sprintf("- index: `%s`\n", result.IndexPath))
	b.WriteString(fmt.Sprintf("- index strategy: `%s`\n", result.IndexStrategy))
	b.WriteString(fmt.Sprintf("- chunks in index: `%d`\n", result.IndexChunks))
	b.WriteString(fmt.Sprintf("- control questions file: `%s`\n\n", result.ControlFilePath))

	b.WriteString("## Single Question Comparison\n")
	b.WriteString(fmt.Sprintf("Question: %s\n\n", result.QuestionComparison.Question))
	b.WriteString("### No RAG\n")
	b.WriteString(result.QuestionComparison.NoRAG.Answer + "\n\n")
	b.WriteString("### With RAG\n")
	b.WriteString(result.QuestionComparison.RAG.Answer + "\n\n")
	b.WriteString("### Retrieved Sources (RAG)\n")
	for _, chunk := range result.QuestionComparison.RAG.Retrieved {
		b.WriteString(fmt.Sprintf("- `%s` (score=%.4f, section=%s)\n", chunk.Source, chunk.Score, chunk.Section))
	}

	if len(result.ControlResults) > 0 {
		b.WriteString("\n## Control Questions (10)\n")
		b.WriteString("| ID | Question | Expectation | Expected Sources | No-RAG Score | RAG Answer Score | RAG Source Score |\n")
		b.WriteString("| --- | --- | --- | --- | ---: | ---: | ---: |\n")
		for _, item := range result.ControlResults {
			b.WriteString(fmt.Sprintf("| %s | %s | %s | %s | %d | %d | %d |\n",
				item.Question.ID,
				escapeDay22Table(item.Question.Question),
				escapeDay22Table(item.Question.Expectation),
				escapeDay22Table(strings.Join(item.Question.ExpectedSources, ", ")),
				item.NoRAGAnswerScore,
				item.RAGAnswerScore,
				item.RAGSourceScore,
			))
		}

		b.WriteString("\n## Aggregate Scores\n")
		b.WriteString(fmt.Sprintf("- no-rag avg answer score: `%d`\n", result.NoRAGAvgScore))
		b.WriteString(fmt.Sprintf("- rag avg answer score: `%d`\n", result.RAGAvgAnswerScore))
		b.WriteString(fmt.Sprintf("- rag avg source score: `%d`\n", result.RAGAvgSourceScore))
		b.WriteString(fmt.Sprintf("- rag combined score (80%% answer + 20%% source): `%d`\n", result.RAGAvgTotalScore))
	}

	b.WriteString("\nConclusion: implemented two modes (without RAG / with RAG) and evaluated 10 control questions against indexed local knowledge.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func escapeDay22Table(s string) string {
	s = strings.ReplaceAll(s, "|", "\\|")
	s = strings.ReplaceAll(s, "\n", " ")
	return s
}

func printDay22Usage() {
	fmt.Println("Usage: openrouter-cli day22 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -question string         Single question for comparison")
	fmt.Println("  -index string            Path to Day21 index JSON")
	fmt.Println("  -top-k int               Number of retrieved chunks for RAG")
	fmt.Println("  -model string            OpenRouter chat model")
	fmt.Println("  -embedding-model string  OpenRouter embedding model")
	fmt.Println("  -max-tokens int          Max tokens per answer")
	fmt.Println("  -temperature float       Temperature")
	fmt.Println("  -run-controls            Run control questions benchmark")
	fmt.Println("  -controls-file string    Control questions JSON path")
	fmt.Println("  -report string           Markdown report path")
	fmt.Println("  -help                    Show help")
}
