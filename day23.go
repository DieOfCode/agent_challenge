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

type day23ModeResult struct {
	Mode            string
	QueryUsed       string
	Answer          string
	Usage           usageStats
	Latency         time.Duration
	RetrievedBefore []day22RetrievedChunk
	RetrievedAfter  []day22RetrievedChunk
	Threshold       float64
}

type day23Comparison struct {
	Question string
	Basic    day23ModeResult
	Enhanced day23ModeResult
}

type day23ControlResult struct {
	Question            day22ControlQuestion
	Basic               day23ModeResult
	Enhanced            day23ModeResult
	BasicAnswerScore    int
	EnhancedAnswerScore int
	BasicSourceScore    int
	EnhancedSourceScore int
}

type day23RunResult struct {
	IndexPath           string
	IndexStrategy       string
	IndexChunks         int
	TopKBefore          int
	TopKAfter           int
	Threshold           float64
	ControlFilePath     string
	Single              day23Comparison
	ControlResults      []day23ControlResult
	BasicAvgAnswerScore int
	EnhancedAvgAnswer   int
	BasicAvgSourceScore int
	EnhancedAvgSource   int
	BasicAvgTotal       int
	EnhancedAvgTotal    int
}

func runDay23Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day23", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	question := fs.String("question", "Какие инструменты обнаружены в Day16?", "Question for single comparison")
	indexPath := fs.String("index", "DAY21_INDEX_structured.json", "Path to Day21 index")
	topKBefore := fs.Int("top-k-before", 12, "Top-K candidates before filtering")
	topKAfter := fs.Int("top-k-after", 4, "Top-K chunks after filtering/reranking")
	threshold := fs.Float64("similarity-threshold", 0.35, "Relevance threshold after reranking")
	model := fs.String("model", getDefaultModel(), "OpenRouter chat model")
	rewriteModel := fs.String("rewrite-model", getDefaultModel(), "OpenRouter model for query rewrite")
	embeddingModel := fs.String("embedding-model", defaultEmbeddingModel(), "OpenRouter embedding model")
	maxTokens := fs.Int("max-tokens", 220, "Max tokens for each answer")
	temperature := fs.Float64("temperature", 0.2, "Temperature for LLM answers")
	runControls := fs.Bool("run-controls", true, "Run 10 control questions")
	controlsPath := fs.String("controls-file", "DAY23_CONTROL_QUESTIONS.json", "Path to control questions JSON")
	reportPath := fs.String("report", "DAY23_RESULTS.md", "Markdown report path")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day23 flags: %w", err)
	}
	if *help {
		printDay23Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day23 arguments: %s", strings.Join(fs.Args(), " "))
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
	if *threshold < 0 || *threshold > 1.5 {
		return fmt.Errorf("similarity-threshold should be in range [0..1.5]")
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

	controls, err := ensureDay23Controls(strings.TrimSpace(*controlsPath))
	if err != nil {
		return err
	}

	apiKey := getAPIKey()
	ctx := context.Background()
	temp := *temperature

	single, err := runDay23Comparison(ctx, apiKey, *model, *rewriteModel, *embeddingModel, index, strings.TrimSpace(*question), *topKBefore, *topKAfter, *threshold, *maxTokens, &temp)
	if err != nil {
		return err
	}

	controlResults := make([]day23ControlResult, 0)
	if *runControls {
		for _, cq := range controls {
			cmp, err := runDay23Comparison(ctx, apiKey, *model, *rewriteModel, *embeddingModel, index, cq.Question, *topKBefore, *topKAfter, *threshold, *maxTokens, &temp)
			if err != nil {
				return fmt.Errorf("control question %s failed: %w", cq.ID, err)
			}
			controlResults = append(controlResults, day23ControlResult{
				Question:            cq,
				Basic:               cmp.Basic,
				Enhanced:            cmp.Enhanced,
				BasicAnswerScore:    day22ScoreAnswer(cmp.Basic.Answer, cq.ExpectedTerms),
				EnhancedAnswerScore: day22ScoreAnswer(cmp.Enhanced.Answer, cq.ExpectedTerms),
				BasicSourceScore:    day22ScoreSources(cmp.Basic.RetrievedAfter, cq.ExpectedSources),
				EnhancedSourceScore: day22ScoreSources(cmp.Enhanced.RetrievedAfter, cq.ExpectedSources),
			})
		}
	}

	result := day23RunResult{
		IndexPath:       strings.TrimSpace(*indexPath),
		IndexStrategy:   index.Strategy,
		IndexChunks:     len(index.Chunks),
		TopKBefore:      *topKBefore,
		TopKAfter:       *topKAfter,
		Threshold:       *threshold,
		ControlFilePath: strings.TrimSpace(*controlsPath),
		Single:          single,
		ControlResults:  controlResults,
	}
	result.BasicAvgAnswerScore, result.EnhancedAvgAnswer, result.BasicAvgSourceScore, result.EnhancedAvgSource, result.BasicAvgTotal, result.EnhancedAvgTotal = day23Aggregate(controlResults)

	printDay23Result(result)
	if err := writeDay23Report(strings.TrimSpace(*reportPath), result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", strings.TrimSpace(*reportPath))
	return nil
}

func runDay23Comparison(ctx context.Context, apiKey, model, rewriteModel, embeddingModel string, index day21Index, question string, topKBefore, topKAfter int, threshold float64, maxTokens int, temperature *float64) (day23Comparison, error) {
	basic, err := day23RunBasicRAG(ctx, apiKey, model, embeddingModel, index, question, topKBefore, topKAfter, maxTokens, temperature)
	if err != nil {
		return day23Comparison{}, err
	}
	enhanced, err := day23RunEnhancedRAG(ctx, apiKey, model, rewriteModel, embeddingModel, index, question, topKBefore, topKAfter, threshold, maxTokens, temperature)
	if err != nil {
		return day23Comparison{}, err
	}
	return day23Comparison{Question: question, Basic: basic, Enhanced: enhanced}, nil
}

func day23RunBasicRAG(ctx context.Context, apiKey, model, embeddingModel string, index day21Index, question string, topKBefore, topKAfter, maxTokens int, temperature *float64) (day23ModeResult, error) {
	before, err := day23RetrieveByEmbedding(ctx, apiKey, embeddingModel, index, question, topKBefore)
	if err != nil {
		return day23ModeResult{}, err
	}
	after := day23TakeTop(before, topKAfter)
	resp, err := day22AskWithRAG(apiKey, model, question, after, maxTokens, temperature)
	if err != nil {
		return day23ModeResult{}, err
	}
	return day23ModeResult{
		Mode:            "basic_rag",
		QueryUsed:       question,
		Answer:          resp.Answer,
		Usage:           resp.Usage,
		Latency:         resp.Latency,
		RetrievedBefore: before,
		RetrievedAfter:  after,
		Threshold:       0,
	}, nil
}

func day23RunEnhancedRAG(ctx context.Context, apiKey, model, rewriteModel, embeddingModel string, index day21Index, question string, topKBefore, topKAfter int, threshold float64, maxTokens int, temperature *float64) (day23ModeResult, error) {
	rewritten, err := day23RewriteQuery(apiKey, rewriteModel, question)
	if err != nil {
		return day23ModeResult{}, err
	}
	queryUsed := rewritten
	if strings.TrimSpace(queryUsed) == "" {
		queryUsed = question
	}

	before, err := day23RetrieveByEmbedding(ctx, apiKey, embeddingModel, index, queryUsed, topKBefore)
	if err != nil {
		return day23ModeResult{}, err
	}
	after := day23RerankAndFilter(before, question, queryUsed, threshold, topKAfter)
	if len(after) == 0 {
		after = day23TakeTop(before, minInt(1, len(before)))
	}

	resp, err := day22AskWithRAG(apiKey, model, question, after, maxTokens, temperature)
	if err != nil {
		return day23ModeResult{}, err
	}
	return day23ModeResult{
		Mode:            "enhanced_rag",
		QueryUsed:       queryUsed,
		Answer:          resp.Answer,
		Usage:           resp.Usage,
		Latency:         resp.Latency,
		RetrievedBefore: before,
		RetrievedAfter:  after,
		Threshold:       threshold,
	}, nil
}

func day23RewriteQuery(apiKey, model, question string) (string, error) {
	system := "Ты преобразуешь вопрос в короткий поисковый запрос для retrieval по локальным markdown/json/go документам. Верни только одну строку запроса, без пояснений."
	user := "Вопрос: " + strings.TrimSpace(question)
	resp, err := callOpenRouterDetailed(
		apiKey,
		model,
		[]message{{Role: "system", Content: system}, {Role: "user", Content: user}},
		40,
		floatPtr(0),
		nil,
		"openrouter-cli-day23-query-rewrite",
	)
	if err != nil {
		return "", fmt.Errorf("query rewrite failed: %w", err)
	}
	out := strings.TrimSpace(resp.Answer)
	out = strings.Trim(out, "`\"'")
	out = strings.ReplaceAll(out, "\n", " ")
	out = strings.TrimSpace(out)
	if out == "" {
		return question, nil
	}
	if len(out) > 220 {
		out = out[:220]
	}
	return out, nil
}

func day23RetrieveByEmbedding(ctx context.Context, apiKey, embeddingModel string, index day21Index, query string, topK int) ([]day22RetrievedChunk, error) {
	vectors, _, err := embedBatch(ctx, apiKey, embeddingModel, []string{query})
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}
	qv := vectors[0]
	if len(qv) == 0 {
		return nil, fmt.Errorf("empty query embedding")
	}

	scored := make([]day22RetrievedChunk, 0, len(index.Chunks))
	for _, chunk := range index.Chunks {
		if len(chunk.Embedding) == 0 {
			continue
		}
		score := day22Cosine(qv, chunk.Embedding)
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

func day23RerankAndFilter(candidates []day22RetrievedChunk, question, rewritten string, threshold float64, topKAfter int) []day22RetrievedChunk {
	origTokens := day22Tokens(question)
	rewriteTokens := day22Tokens(rewritten)
	reranked := make([]day22RetrievedChunk, 0, len(candidates))

	for _, item := range candidates {
		text := strings.ToLower(item.Source + " " + item.Title + " " + item.Section + " " + item.Text)
		origLex := day23TokenHitRatio(origTokens, text)
		rewriteLex := day23TokenHitRatio(rewriteTokens, text)
		final := item.Score*0.65 + origLex*0.20 + rewriteLex*0.15
		item.Score = final
		if final >= threshold {
			reranked = append(reranked, item)
		}
	}

	sort.Slice(reranked, func(i, j int) bool { return reranked[i].Score > reranked[j].Score })
	if len(reranked) > topKAfter {
		reranked = reranked[:topKAfter]
	}
	return reranked
}

func day23TokenHitRatio(tokens []string, text string) float64 {
	if len(tokens) == 0 {
		return 0
	}
	seen := map[string]struct{}{}
	hits := 0
	for _, t := range tokens {
		if _, ok := seen[t]; ok {
			continue
		}
		seen[t] = struct{}{}
		if strings.Contains(text, t) {
			hits++
		}
	}
	if len(seen) == 0 {
		return 0
	}
	return float64(hits) / float64(len(seen))
}

func day23TakeTop(items []day22RetrievedChunk, n int) []day22RetrievedChunk {
	if n <= 0 || len(items) == 0 {
		return nil
	}
	if n > len(items) {
		n = len(items)
	}
	out := make([]day22RetrievedChunk, n)
	copy(out, items[:n])
	return out
}

func day23Aggregate(items []day23ControlResult) (int, int, int, int, int, int) {
	if len(items) == 0 {
		return 0, 0, 0, 0, 0, 0
	}
	var bAns, eAns, bSrc, eSrc, bTotal, eTotal int
	for _, it := range items {
		bAns += it.BasicAnswerScore
		eAns += it.EnhancedAnswerScore
		bSrc += it.BasicSourceScore
		eSrc += it.EnhancedSourceScore
		bTotal += (it.BasicAnswerScore*80 + it.BasicSourceScore*20) / 100
		eTotal += (it.EnhancedAnswerScore*80 + it.EnhancedSourceScore*20) / 100
	}
	n := len(items)
	return bAns / n, eAns / n, bSrc / n, eSrc / n, bTotal / n, eTotal / n
}

func ensureDay23Controls(path string) ([]day22ControlQuestion, error) {
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

func writeDay23Report(path string, result day23RunResult) error {
	var b strings.Builder
	b.WriteString("# Day 23 Results: Reranking and Filtering\n\n")
	b.WriteString(fmt.Sprintf("- index: `%s`\n", result.IndexPath))
	b.WriteString(fmt.Sprintf("- index strategy: `%s`\n", result.IndexStrategy))
	b.WriteString(fmt.Sprintf("- chunks in index: `%d`\n", result.IndexChunks))
	b.WriteString(fmt.Sprintf("- top-k before: `%d`\n", result.TopKBefore))
	b.WriteString(fmt.Sprintf("- top-k after: `%d`\n", result.TopKAfter))
	b.WriteString(fmt.Sprintf("- similarity threshold: `%.2f`\n", result.Threshold))
	b.WriteString(fmt.Sprintf("- control questions file: `%s`\n\n", result.ControlFilePath))

	b.WriteString("## Single Question\n")
	b.WriteString(fmt.Sprintf("Question: %s\n\n", result.Single.Question))
	b.WriteString("### Basic RAG (without rewrite/filter)\n")
	b.WriteString(fmt.Sprintf("Query used: `%s`\n\n", result.Single.Basic.QueryUsed))
	b.WriteString(result.Single.Basic.Answer + "\n\n")
	b.WriteString("Retrieved:\n")
	for _, item := range result.Single.Basic.RetrievedAfter {
		b.WriteString(fmt.Sprintf("- `%s` (score=%.4f)\n", item.Source, item.Score))
	}

	b.WriteString("\n### Enhanced RAG (query rewrite + rerank/filter)\n")
	b.WriteString(fmt.Sprintf("Query used: `%s`\n\n", result.Single.Enhanced.QueryUsed))
	b.WriteString(result.Single.Enhanced.Answer + "\n\n")
	b.WriteString("Retrieved (after filter):\n")
	for _, item := range result.Single.Enhanced.RetrievedAfter {
		b.WriteString(fmt.Sprintf("- `%s` (score=%.4f)\n", item.Source, item.Score))
	}

	if len(result.ControlResults) > 0 {
		b.WriteString("\n## Control Questions Comparison\n")
		b.WriteString("| ID | Question | Basic Answer | Enhanced Answer | Basic Source | Enhanced Source |\n")
		b.WriteString("| --- | --- | ---: | ---: | ---: | ---: |\n")
		for _, item := range result.ControlResults {
			b.WriteString(fmt.Sprintf("| %s | %s | %d | %d | %d | %d |\n",
				item.Question.ID,
				escapeDay22Table(item.Question.Question),
				item.BasicAnswerScore,
				item.EnhancedAnswerScore,
				item.BasicSourceScore,
				item.EnhancedSourceScore,
			))
		}

		b.WriteString("\n## Aggregate\n")
		b.WriteString(fmt.Sprintf("- basic avg answer score: `%d`\n", result.BasicAvgAnswerScore))
		b.WriteString(fmt.Sprintf("- enhanced avg answer score: `%d`\n", result.EnhancedAvgAnswer))
		b.WriteString(fmt.Sprintf("- basic avg source score: `%d`\n", result.BasicAvgSourceScore))
		b.WriteString(fmt.Sprintf("- enhanced avg source score: `%d`\n", result.EnhancedAvgSource))
		b.WriteString(fmt.Sprintf("- basic combined score: `%d`\n", result.BasicAvgTotal))
		b.WriteString(fmt.Sprintf("- enhanced combined score: `%d`\n", result.EnhancedAvgTotal))
	}

	conclusion := "enhanced mode applies query rewrite and relevance filtering/reranking; metrics are comparable to basic mode."
	if len(result.ControlResults) > 0 {
		switch {
		case result.EnhancedAvgTotal > result.BasicAvgTotal:
			conclusion = "enhanced mode applies query rewrite and relevance filtering/reranking and improves combined quality versus basic mode."
		case result.EnhancedAvgTotal < result.BasicAvgTotal:
			conclusion = "enhanced mode applies query rewrite and relevance filtering/reranking, but on this benchmark basic mode scored higher."
		}
	}
	b.WriteString("\nConclusion: " + conclusion + "\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay23Result(result day23RunResult) {
	fmt.Println("=== Day 23: Reranking and Filtering ===")
	fmt.Printf("index=%s strategy=%s chunks=%d\n", result.IndexPath, result.IndexStrategy, result.IndexChunks)
	fmt.Printf("top_k_before=%d top_k_after=%d threshold=%.2f\n", result.TopKBefore, result.TopKAfter, result.Threshold)
	fmt.Printf("single_question=%s\n", result.Single.Question)
	fmt.Printf("single_basic_tokens=%d single_enhanced_tokens=%d\n", result.Single.Basic.Usage.TotalTokens, result.Single.Enhanced.Usage.TotalTokens)
	fmt.Printf("controls=%d basic_answer_avg=%d enhanced_answer_avg=%d basic_source_avg=%d enhanced_source_avg=%d basic_total_avg=%d enhanced_total_avg=%d\n",
		len(result.ControlResults),
		result.BasicAvgAnswerScore,
		result.EnhancedAvgAnswer,
		result.BasicAvgSourceScore,
		result.EnhancedAvgSource,
		result.BasicAvgTotal,
		result.EnhancedAvgTotal,
	)
}

func printDay23Usage() {
	fmt.Println("Usage: openrouter-cli day23 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -question string            Single question for comparison")
	fmt.Println("  -index string               Path to Day21 index JSON")
	fmt.Println("  -top-k-before int           Top-K before rerank/filter")
	fmt.Println("  -top-k-after int            Top-K after rerank/filter")
	fmt.Println("  -similarity-threshold float Filtering threshold after rerank")
	fmt.Println("  -model string               Chat model")
	fmt.Println("  -rewrite-model string       Query rewrite model")
	fmt.Println("  -embedding-model string     Embedding model")
	fmt.Println("  -max-tokens int             Max tokens per answer")
	fmt.Println("  -temperature float          Temperature")
	fmt.Println("  -run-controls               Run 10 control questions")
	fmt.Println("  -controls-file string       Control questions JSON path")
	fmt.Println("  -report string              Markdown report path")
	fmt.Println("  -help                       Show help")
}

func floatPtr(v float64) *float64 {
	return &v
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
