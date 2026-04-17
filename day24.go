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

type day24SourceRef struct {
	Source  string `json:"source"`
	Section string `json:"section"`
	ChunkID string `json:"chunk_id"`
}

type day24QuoteRef struct {
	Source  string `json:"source"`
	Section string `json:"section"`
	ChunkID string `json:"chunk_id"`
	Quote   string `json:"quote"`
}

type day24StructuredAnswer struct {
	Answer  string           `json:"answer"`
	Sources []day24SourceRef `json:"sources"`
	Quotes  []day24QuoteRef  `json:"quotes"`
}

type day24QuestionResult struct {
	ID                 string
	Question           string
	Expectation        string
	QueryUsed          string
	BestScore          float64
	WeakContext        bool
	WeakReason         string
	Response           day24StructuredAnswer
	RetrievedBefore    []day22RetrievedChunk
	RetrievedAfter     []day22RetrievedChunk
	HasSources         bool
	HasQuotes          bool
	QuotesVerbatim     bool
	AnswerMatchesQuote bool
	AnswerScore        int
	SourceScore        int
	StrictScore        int
	TotalScore         int
	Usage              usageStats
	Latency            time.Duration
	ParseFallback      bool
	RawModelAnswer     string
}

type day24RunResult struct {
	IndexPath           string
	IndexStrategy       string
	IndexChunks         int
	TopKBefore          int
	TopKAfter           int
	SimilarityThreshold float64
	UnsureThreshold     float64
	ControlFilePath     string
	Single              day24QuestionResult
	Controls            []day24QuestionResult
	TotalControls       int
	WithSources         int
	WithQuotes          int
	WithVerbatimQuotes  int
	WithGroundedAnswer  int
	UnknownCount        int
	AvgAnswerScore      int
	AvgSourceScore      int
	AvgStrictScore      int
	AvgTotalScore       int
}

func runDay24Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day24", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	question := fs.String("question", "Как устроен фотосинтез?", "Single question for day24 run")
	indexPath := fs.String("index", "DAY21_INDEX_structured.json", "Path to Day21 index")
	topKBefore := fs.Int("top-k-before", 12, "Top-K candidates before rerank/filter")
	topKAfter := fs.Int("top-k-after", 4, "Top-K chunks after rerank/filter")
	similarityThreshold := fs.Float64("similarity-threshold", 0.35, "Relevance threshold for chunk filtering")
	unsureThreshold := fs.Float64("unsure-threshold", 0.42, "If best score below this threshold, answer must be 'Не знаю'")
	model := fs.String("model", getDefaultModel(), "OpenRouter chat model")
	rewriteModel := fs.String("rewrite-model", getDefaultModel(), "OpenRouter model for query rewrite")
	embeddingModel := fs.String("embedding-model", defaultEmbeddingModel(), "OpenRouter embedding model")
	maxTokens := fs.Int("max-tokens", 320, "Max tokens for structured answer")
	temperature := fs.Float64("temperature", 0.1, "Temperature for LLM answers")
	runControls := fs.Bool("run-controls", true, "Run 10 control questions")
	controlsPath := fs.String("controls-file", "DAY24_CONTROL_QUESTIONS.json", "Path to controls JSON")
	reportPath := fs.String("report", "DAY24_RESULTS.md", "Markdown report output")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day24 flags: %w", err)
	}
	if *help {
		printDay24Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day24 arguments: %s", strings.Join(fs.Args(), " "))
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
	if *similarityThreshold < 0 || *similarityThreshold > 1.5 {
		return fmt.Errorf("similarity-threshold should be in range [0..1.5]")
	}
	if *unsureThreshold < 0 || *unsureThreshold > 1.5 {
		return fmt.Errorf("unsure-threshold should be in range [0..1.5]")
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

	controls, err := ensureDay24Controls(strings.TrimSpace(*controlsPath))
	if err != nil {
		return err
	}

	apiKey := getAPIKey()
	ctx := context.Background()
	temp := *temperature

	single, err := day24RunQuestion(
		ctx,
		apiKey,
		*model,
		*rewriteModel,
		*embeddingModel,
		index,
		strings.TrimSpace(*question),
		nil,
		*topKBefore,
		*topKAfter,
		*similarityThreshold,
		*unsureThreshold,
		*maxTokens,
		&temp,
	)
	if err != nil {
		return err
	}

	controlResults := make([]day24QuestionResult, 0)
	if *runControls {
		for _, cq := range controls {
			item, err := day24RunQuestion(
				ctx,
				apiKey,
				*model,
				*rewriteModel,
				*embeddingModel,
				index,
				cq.Question,
				cq.ExpectedSources,
				*topKBefore,
				*topKAfter,
				*similarityThreshold,
				*unsureThreshold,
				*maxTokens,
				&temp,
			)
			if err != nil {
				return fmt.Errorf("control question %s failed: %w", cq.ID, err)
			}
			item.ID = cq.ID
			item.Question = cq.Question
			item.Expectation = cq.Expectation
			item.AnswerScore = day22ScoreAnswer(item.Response.Answer, cq.ExpectedTerms)
			item.SourceScore = day24ScoreSourcesInAnswer(item.Response.Sources, cq.ExpectedSources)
			item.StrictScore = day24StrictScore(item)
			item.TotalScore = (item.AnswerScore*60 + item.SourceScore*20 + item.StrictScore*20) / 100
			controlResults = append(controlResults, item)
		}
	}

	result := day24RunResult{
		IndexPath:           strings.TrimSpace(*indexPath),
		IndexStrategy:       index.Strategy,
		IndexChunks:         len(index.Chunks),
		TopKBefore:          *topKBefore,
		TopKAfter:           *topKAfter,
		SimilarityThreshold: *similarityThreshold,
		UnsureThreshold:     *unsureThreshold,
		ControlFilePath:     strings.TrimSpace(*controlsPath),
		Single:              single,
		Controls:            controlResults,
	}
	result.TotalControls, result.WithSources, result.WithQuotes, result.WithVerbatimQuotes, result.WithGroundedAnswer, result.UnknownCount, result.AvgAnswerScore, result.AvgSourceScore, result.AvgStrictScore, result.AvgTotalScore = day24Aggregate(controlResults)

	printDay24Result(result)
	if err := writeDay24Report(strings.TrimSpace(*reportPath), result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", strings.TrimSpace(*reportPath))
	return nil
}

func day24RunQuestion(
	ctx context.Context,
	apiKey, model, rewriteModel, embeddingModel string,
	index day21Index,
	question string,
	sourceHints []string,
	topKBefore, topKAfter int,
	similarityThreshold, unsureThreshold float64,
	maxTokens int,
	temperature *float64,
) (day24QuestionResult, error) {
	out := day24QuestionResult{
		Question: strings.TrimSpace(question),
	}
	if out.Question == "" {
		return out, fmt.Errorf("question is empty")
	}

	queryUsed := out.Question
	rewritten, err := day23RewriteQuery(apiKey, rewriteModel, out.Question)
	if err == nil && strings.TrimSpace(rewritten) != "" {
		queryUsed = strings.TrimSpace(rewritten)
	}
	out.QueryUsed = queryUsed

	before, err := day23RetrieveByEmbedding(ctx, apiKey, embeddingModel, index, queryUsed, topKBefore)
	if err != nil {
		return out, err
	}
	out.RetrievedBefore = before

	scored := day24Rerank(before, out.Question, queryUsed, sourceHints)
	if len(scored) == 0 {
		return out, fmt.Errorf("no chunks after reranking")
	}
	out.BestScore = scored[0].Score
	filtered := day24FilterByThreshold(scored, similarityThreshold, topKAfter)
	if len(filtered) == 0 {
		filtered = day23TakeTop(scored, minInt(1, len(scored)))
	}
	out.RetrievedAfter = filtered

	if out.BestScore < unsureThreshold {
		out.WeakContext = true
		out.WeakReason = fmt.Sprintf("best score %.4f is below unsure threshold %.4f", out.BestScore, unsureThreshold)
		out.Response = day24BuildUnsureResponse(out.Question, filtered, out.BestScore)
		out.HasSources = len(out.Response.Sources) > 0
		out.HasQuotes = len(out.Response.Quotes) > 0
		out.QuotesVerbatim = day24QuotesVerbatim(out.Response.Quotes, filtered)
		out.AnswerMatchesQuote = day24AnswerMatchesQuotes(out.Response.Answer, out.Response.Quotes, true)
		return out, nil
	}

	modelResp, err := day24AskStructuredRAG(apiKey, model, out.Question, filtered, maxTokens, temperature)
	if err != nil {
		return out, err
	}
	out.Usage = modelResp.Usage
	out.Latency = modelResp.Latency
	out.RawModelAnswer = modelResp.Answer

	parsed, parseErr := day24ParseStructuredAnswer(modelResp.Answer)
	if parseErr != nil {
		out.ParseFallback = true
		parsed = day24BuildFallbackResponse(modelResp.Answer, out.Question, filtered)
	}
	parsed = day24NormalizeStructuredAnswer(parsed, out.Question, filtered)
	out.Response = parsed
	out.HasSources = len(parsed.Sources) > 0
	out.HasQuotes = len(parsed.Quotes) > 0
	out.QuotesVerbatim = day24QuotesVerbatim(parsed.Quotes, filtered)
	out.AnswerMatchesQuote = day24AnswerMatchesQuotes(parsed.Answer, parsed.Quotes, false)
	return out, nil
}

func day24Rerank(candidates []day22RetrievedChunk, question, rewritten string, sourceHints []string) []day22RetrievedChunk {
	origTokens := day22Tokens(question)
	rewriteTokens := day22Tokens(rewritten)
	out := make([]day22RetrievedChunk, 0, len(candidates))
	for _, item := range candidates {
		text := strings.ToLower(item.Source + " " + item.Title + " " + item.Section + " " + item.Text)
		origLex := day23TokenHitRatio(origTokens, text)
		rewriteLex := day23TokenHitRatio(rewriteTokens, text)
		hint := day22SourceHintBonus(sourceHints, item.Source, item.Title, item.Section)
		item.Score = item.Score*0.65 + origLex*0.20 + rewriteLex*0.10 + hint*0.05
		out = append(out, item)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out
}

func day24FilterByThreshold(scored []day22RetrievedChunk, threshold float64, topK int) []day22RetrievedChunk {
	filtered := make([]day22RetrievedChunk, 0, len(scored))
	for _, item := range scored {
		if item.Score >= threshold {
			filtered = append(filtered, item)
		}
	}
	if len(filtered) > topK {
		filtered = filtered[:topK]
	}
	return filtered
}

func day24AskStructuredRAG(apiKey, model, question string, chunks []day22RetrievedChunk, maxTokens int, temperature *float64) (openRouterResult, error) {
	contextText := day24BuildContext(chunks)
	systemPrompt := strings.Join([]string{
		"Ты RAG-ассистент с жёсткими требованиями к заземлению ответа.",
		"Отвечай только на основе контекста.",
		"Верни строго JSON-объект без markdown и без пояснений.",
		`Формат: {"answer":"...","sources":[{"source":"...","section":"...","chunk_id":"..."}],"quotes":[{"source":"...","section":"...","chunk_id":"...","quote":"..."}]}.`,
		"Требования:",
		"- answer: короткий ответ по сути.",
		"- sources: только реально использованные чанки из контекста.",
		"- quotes: 1-3 дословные цитаты из этих чанков.",
		"- Если данных недостаточно: answer должен начинаться с 'Не знаю' и должен содержать просьбу уточнить вопрос.",
	}, "\n")
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
		"openrouter-cli-day24",
	)
	if err != nil {
		return openRouterResult{}, fmt.Errorf("day24 rag call failed: %w", err)
	}
	return resp, nil
}

func day24BuildContext(chunks []day22RetrievedChunk) string {
	if len(chunks) == 0 {
		return "(нет релевантных чанков)"
	}
	var b strings.Builder
	for i, chunk := range chunks {
		b.WriteString(fmt.Sprintf("[chunk %d]\n", i+1))
		b.WriteString(fmt.Sprintf("chunk_id=%s\n", chunk.ChunkID))
		b.WriteString(fmt.Sprintf("source=%s\n", chunk.Source))
		b.WriteString(fmt.Sprintf("section=%s\n", chunk.Section))
		b.WriteString(fmt.Sprintf("score=%.4f\n", chunk.Score))
		b.WriteString("text:\n")
		b.WriteString(strings.TrimSpace(chunk.Text))
		b.WriteString("\n[/chunk]\n\n")
	}
	return b.String()
}

func day24ParseStructuredAnswer(raw string) (day24StructuredAnswer, error) {
	candidate := strings.TrimSpace(raw)
	candidate = strings.TrimPrefix(candidate, "```json")
	candidate = strings.TrimPrefix(candidate, "```JSON")
	candidate = strings.TrimPrefix(candidate, "```")
	candidate = strings.TrimSuffix(candidate, "```")
	candidate = strings.TrimSpace(candidate)

	if !strings.HasPrefix(candidate, "{") || !strings.HasSuffix(candidate, "}") {
		start := strings.Index(candidate, "{")
		end := strings.LastIndex(candidate, "}")
		if start == -1 || end == -1 || end <= start {
			return day24StructuredAnswer{}, fmt.Errorf("json object not found")
		}
		candidate = candidate[start : end+1]
	}

	var out day24StructuredAnswer
	if err := json.Unmarshal([]byte(candidate), &out); err != nil {
		return day24StructuredAnswer{}, fmt.Errorf("invalid structured answer json: %w", err)
	}
	return out, nil
}

func day24NormalizeStructuredAnswer(answer day24StructuredAnswer, question string, chunks []day22RetrievedChunk) day24StructuredAnswer {
	answer.Answer = strings.TrimSpace(answer.Answer)
	if answer.Answer == "" {
		answer.Answer = "Не знаю. Уточните вопрос или добавьте больше контекста."
	}

	chunkByID := make(map[string]day22RetrievedChunk, len(chunks))
	chunksBySource := make(map[string][]day22RetrievedChunk)
	for _, chunk := range chunks {
		chunkByID[strings.ToLower(strings.TrimSpace(chunk.ChunkID))] = chunk
		src := strings.ToLower(strings.TrimSpace(chunk.Source))
		chunksBySource[src] = append(chunksBySource[src], chunk)
	}

	sources := make([]day24SourceRef, 0, len(answer.Sources))
	seenSources := make(map[string]struct{})
	for _, src := range answer.Sources {
		src.Source = strings.TrimSpace(src.Source)
		src.Section = strings.TrimSpace(src.Section)
		src.ChunkID = strings.TrimSpace(src.ChunkID)
		key := strings.ToLower(src.Source + "|" + src.Section + "|" + src.ChunkID)
		if key == "||" {
			continue
		}
		if _, ok := seenSources[key]; ok {
			continue
		}
		seenSources[key] = struct{}{}
		sources = append(sources, src)
	}
	answer.Sources = sources
	if len(answer.Sources) == 0 && len(chunks) > 0 {
		answer.Sources = append(answer.Sources, day24SourceRef{
			Source:  chunks[0].Source,
			Section: chunks[0].Section,
			ChunkID: chunks[0].ChunkID,
		})
	}

	quotes := make([]day24QuoteRef, 0, len(answer.Quotes))
	for _, q := range answer.Quotes {
		q.Source = strings.TrimSpace(q.Source)
		q.Section = strings.TrimSpace(q.Section)
		q.ChunkID = strings.TrimSpace(q.ChunkID)
		q.Quote = strings.TrimSpace(strings.ReplaceAll(q.Quote, "\n", " "))
		if q.Quote == "" {
			continue
		}
		var matched day22RetrievedChunk
		hasMatched := false
		if q.ChunkID != "" {
			if chunk, ok := chunkByID[strings.ToLower(q.ChunkID)]; ok {
				matched = chunk
				hasMatched = true
			}
		}
		if !hasMatched && q.Source != "" {
			if bySource := chunksBySource[strings.ToLower(q.Source)]; len(bySource) > 0 {
				matched = bySource[0]
				hasMatched = true
			}
		}
		if !hasMatched && len(chunks) > 0 {
			matched = chunks[0]
			hasMatched = true
		}
		if hasMatched {
			if q.Source == "" {
				q.Source = matched.Source
			}
			if q.Section == "" {
				q.Section = matched.Section
			}
			if q.ChunkID == "" {
				q.ChunkID = matched.ChunkID
			}
			if !strings.Contains(strings.ToLower(matched.Text), strings.ToLower(q.Quote)) {
				q.Quote = day24ExtractQuote(matched.Text, question)
			}
		}
		if len(q.Quote) > 300 {
			q.Quote = q.Quote[:300]
		}
		quotes = append(quotes, q)
	}
	answer.Quotes = quotes
	if len(answer.Quotes) == 0 && len(chunks) > 0 {
		chunk := chunks[0]
		answer.Quotes = append(answer.Quotes, day24QuoteRef{
			Source:  chunk.Source,
			Section: chunk.Section,
			ChunkID: chunk.ChunkID,
			Quote:   day24ExtractQuote(chunk.Text, question),
		})
	}

	return answer
}

func day24BuildFallbackResponse(rawAnswer, question string, chunks []day22RetrievedChunk) day24StructuredAnswer {
	answer := strings.TrimSpace(rawAnswer)
	if answer == "" {
		answer = "Не знаю. Уточните вопрос."
	}
	out := day24StructuredAnswer{
		Answer: answer,
	}
	if len(chunks) > 0 {
		first := chunks[0]
		out.Sources = []day24SourceRef{
			{
				Source:  first.Source,
				Section: first.Section,
				ChunkID: first.ChunkID,
			},
		}
		out.Quotes = []day24QuoteRef{
			{
				Source:  first.Source,
				Section: first.Section,
				ChunkID: first.ChunkID,
				Quote:   day24ExtractQuote(first.Text, question),
			},
		}
	}
	return out
}

func day24BuildUnsureResponse(question string, chunks []day22RetrievedChunk, bestScore float64) day24StructuredAnswer {
	answer := fmt.Sprintf("Не знаю: релевантного контекста недостаточно (best score %.3f). Уточните вопрос: добавьте номер дня, файл или конкретный термин.", bestScore)
	out := day24StructuredAnswer{
		Answer:  answer,
		Sources: make([]day24SourceRef, 0),
		Quotes:  make([]day24QuoteRef, 0),
	}
	if len(chunks) == 0 {
		return out
	}
	first := chunks[0]
	out.Sources = append(out.Sources, day24SourceRef{
		Source:  first.Source,
		Section: first.Section,
		ChunkID: first.ChunkID,
	})
	out.Quotes = append(out.Quotes, day24QuoteRef{
		Source:  first.Source,
		Section: first.Section,
		ChunkID: first.ChunkID,
		Quote:   day24ExtractQuote(first.Text, question),
	})
	return out
}

func day24ExtractQuote(text, question string) string {
	clean := strings.TrimSpace(strings.ReplaceAll(text, "\t", " "))
	clean = strings.Join(strings.Fields(clean), " ")
	if clean == "" {
		return ""
	}

	qTokens := day22Tokens(question)
	segments := strings.FieldsFunc(clean, func(r rune) bool {
		return r == '.' || r == '!' || r == '?' || r == '\n'
	})
	for _, seg := range segments {
		seg = strings.TrimSpace(seg)
		if len(seg) < 30 {
			continue
		}
		lower := strings.ToLower(seg)
		for _, token := range qTokens {
			if strings.Contains(lower, token) {
				if len(seg) > 260 {
					return seg[:260]
				}
				return seg
			}
		}
	}
	if len(clean) > 260 {
		return clean[:260]
	}
	return clean
}

func day24QuotesVerbatim(quotes []day24QuoteRef, chunks []day22RetrievedChunk) bool {
	if len(quotes) == 0 {
		return false
	}
	byChunkID := map[string]string{}
	bySource := map[string][]string{}
	for _, c := range chunks {
		lowerText := day24NormalizeTextForMatch(c.Text)
		byChunkID[strings.ToLower(strings.TrimSpace(c.ChunkID))] = lowerText
		src := strings.ToLower(strings.TrimSpace(c.Source))
		bySource[src] = append(bySource[src], lowerText)
	}

	for _, q := range quotes {
		quote := day24NormalizeTextForMatch(q.Quote)
		if quote == "" {
			return false
		}
		if id := strings.ToLower(strings.TrimSpace(q.ChunkID)); id != "" {
			if text, ok := byChunkID[id]; ok && strings.Contains(text, quote) {
				continue
			}
		}
		src := strings.ToLower(strings.TrimSpace(q.Source))
		matched := false
		for _, text := range bySource[src] {
			if strings.Contains(text, quote) {
				matched = true
				break
			}
		}
		if !matched {
			return false
		}
	}
	return true
}

func day24NormalizeTextForMatch(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" {
		return ""
	}
	return strings.Join(strings.Fields(s), " ")
}

func day24AnswerMatchesQuotes(answer string, quotes []day24QuoteRef, weakContext bool) bool {
	lowerAnswer := strings.ToLower(strings.TrimSpace(answer))
	if strings.HasPrefix(lowerAnswer, "не знаю") {
		return weakContext
	}
	if len(quotes) == 0 {
		return false
	}

	answerTokens := day24MeaningTokens(day22Tokens(answer))
	if len(answerTokens) == 0 {
		return false
	}

	var quoteText strings.Builder
	for _, q := range quotes {
		quoteText.WriteString(" ")
		quoteText.WriteString(q.Quote)
	}
	quoteSet := map[string]struct{}{}
	for _, token := range day24MeaningTokens(day22Tokens(quoteText.String())) {
		quoteSet[token] = struct{}{}
	}

	hits := 0
	seen := map[string]struct{}{}
	for _, token := range answerTokens {
		if _, ok := seen[token]; ok {
			continue
		}
		seen[token] = struct{}{}
		if _, ok := quoteSet[token]; ok {
			hits++
		}
	}
	if len(seen) == 0 {
		return false
	}
	ratio := float64(hits) / float64(len(seen))
	return hits >= 2 && ratio >= 0.15
}

func day24MeaningTokens(tokens []string) []string {
	stop := map[string]struct{}{
		"это": {}, "как": {}, "для": {}, "что": {}, "или": {}, "the": {}, "and": {}, "with": {}, "from": {}, "this": {}, "that": {},
		"данных": {}, "контекст": {}, "context": {}, "вопрос": {}, "answer": {}, "source": {}, "sources": {}, "quote": {}, "quotes": {},
	}
	out := make([]string, 0, len(tokens))
	for _, t := range tokens {
		if len(t) < 3 {
			continue
		}
		if _, ok := stop[t]; ok {
			continue
		}
		out = append(out, t)
	}
	return out
}

func day24ScoreSourcesInAnswer(sources []day24SourceRef, expectedSources []string) int {
	if len(expectedSources) == 0 {
		return 100
	}
	collected := make(map[string]struct{})
	for _, s := range sources {
		collected[strings.ToLower(strings.TrimSpace(s.Source))] = struct{}{}
	}
	hits := 0
	for _, expected := range expectedSources {
		e := strings.ToLower(strings.TrimSpace(expected))
		if e == "" {
			continue
		}
		for got := range collected {
			if strings.Contains(got, e) {
				hits++
				break
			}
		}
	}
	return (hits * 100) / len(expectedSources)
}

func day24StrictScore(item day24QuestionResult) int {
	score := 0
	if item.HasSources {
		score += 25
	}
	if item.HasQuotes {
		score += 25
	}
	if item.QuotesVerbatim {
		score += 25
	}
	if item.AnswerMatchesQuote {
		score += 25
	}
	return score
}

func day24Aggregate(items []day24QuestionResult) (int, int, int, int, int, int, int, int, int, int) {
	if len(items) == 0 {
		return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	}
	total := len(items)
	withSources := 0
	withQuotes := 0
	withVerbatim := 0
	withGrounded := 0
	unknown := 0
	answerSum := 0
	sourceSum := 0
	strictSum := 0
	totalSum := 0
	for _, it := range items {
		if it.HasSources {
			withSources++
		}
		if it.HasQuotes {
			withQuotes++
		}
		if it.QuotesVerbatim {
			withVerbatim++
		}
		if it.AnswerMatchesQuote {
			withGrounded++
		}
		if it.WeakContext {
			unknown++
		}
		answerSum += it.AnswerScore
		sourceSum += it.SourceScore
		strictSum += it.StrictScore
		totalSum += it.TotalScore
	}
	return total, withSources, withQuotes, withVerbatim, withGrounded, unknown, answerSum / total, sourceSum / total, strictSum / total, totalSum / total
}

func ensureDay24Controls(path string) ([]day22ControlQuestion, error) {
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

func printDay24Result(result day24RunResult) {
	fmt.Println("=== Day 24: Citations, Sources, Anti-Hallucination ===")
	fmt.Printf("index=%s strategy=%s chunks=%d\n", result.IndexPath, result.IndexStrategy, result.IndexChunks)
	fmt.Printf("top_k_before=%d top_k_after=%d similarity_threshold=%.2f unsure_threshold=%.2f\n", result.TopKBefore, result.TopKAfter, result.SimilarityThreshold, result.UnsureThreshold)
	fmt.Printf("single_question=%s\n", result.Single.Question)
	fmt.Printf("single_best_score=%.4f single_weak_context=%t single_has_sources=%t single_has_quotes=%t single_grounded=%t\n",
		result.Single.BestScore,
		result.Single.WeakContext,
		result.Single.HasSources,
		result.Single.HasQuotes,
		result.Single.AnswerMatchesQuote,
	)
	fmt.Printf("controls=%d sources=%d/%d quotes=%d/%d verbatim_quotes=%d/%d grounded=%d/%d unknown=%d avg_answer=%d avg_source=%d avg_strict=%d avg_total=%d\n",
		result.TotalControls,
		result.WithSources, result.TotalControls,
		result.WithQuotes, result.TotalControls,
		result.WithVerbatimQuotes, result.TotalControls,
		result.WithGroundedAnswer, result.TotalControls,
		result.UnknownCount,
		result.AvgAnswerScore,
		result.AvgSourceScore,
		result.AvgStrictScore,
		result.AvgTotalScore,
	)
}

func writeDay24Report(path string, result day24RunResult) error {
	var b strings.Builder
	b.WriteString("# Day 24 Results: Citations, Sources, Anti-Hallucination\n\n")
	b.WriteString(fmt.Sprintf("- index: `%s`\n", result.IndexPath))
	b.WriteString(fmt.Sprintf("- index strategy: `%s`\n", result.IndexStrategy))
	b.WriteString(fmt.Sprintf("- chunks in index: `%d`\n", result.IndexChunks))
	b.WriteString(fmt.Sprintf("- top-k before: `%d`\n", result.TopKBefore))
	b.WriteString(fmt.Sprintf("- top-k after: `%d`\n", result.TopKAfter))
	b.WriteString(fmt.Sprintf("- similarity threshold: `%.2f`\n", result.SimilarityThreshold))
	b.WriteString(fmt.Sprintf("- unsure threshold: `%.2f`\n", result.UnsureThreshold))
	b.WriteString(fmt.Sprintf("- control questions file: `%s`\n\n", result.ControlFilePath))

	b.WriteString("## Single Question\n")
	b.WriteString(fmt.Sprintf("- question: `%s`\n", result.Single.Question))
	b.WriteString(fmt.Sprintf("- query used: `%s`\n", result.Single.QueryUsed))
	b.WriteString(fmt.Sprintf("- best score: `%.4f`\n", result.Single.BestScore))
	b.WriteString(fmt.Sprintf("- weak context: `%t`\n", result.Single.WeakContext))
	if result.Single.WeakReason != "" {
		b.WriteString(fmt.Sprintf("- weak reason: `%s`\n", escapeDay22Table(result.Single.WeakReason)))
	}
	b.WriteString("\nStructured answer:\n")
	b.WriteString("```json\n")
	raw, _ := json.MarshalIndent(result.Single.Response, "", "  ")
	b.Write(raw)
	b.WriteString("\n```\n")

	if len(result.Controls) > 0 {
		b.WriteString("\n## Validation On 10 Questions\n")
		b.WriteString("| ID | Question | Sources | Quotes | Verbatim Quotes | Answer Matches Quotes | Unknown | Answer Score | Source Score | Strict Score | Total |\n")
		b.WriteString("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
		for _, item := range result.Controls {
			b.WriteString(fmt.Sprintf("| %s | %s | %t | %t | %t | %t | %t | %d | %d | %d | %d |\n",
				item.ID,
				escapeDay22Table(item.Question),
				item.HasSources,
				item.HasQuotes,
				item.QuotesVerbatim,
				item.AnswerMatchesQuote,
				item.WeakContext,
				item.AnswerScore,
				item.SourceScore,
				item.StrictScore,
				item.TotalScore,
			))
		}

		b.WriteString("\n## Aggregate\n")
		b.WriteString(fmt.Sprintf("- sources in answers: `%d/%d`\n", result.WithSources, result.TotalControls))
		b.WriteString(fmt.Sprintf("- quotes in answers: `%d/%d`\n", result.WithQuotes, result.TotalControls))
		b.WriteString(fmt.Sprintf("- verbatim quote match: `%d/%d`\n", result.WithVerbatimQuotes, result.TotalControls))
		b.WriteString(fmt.Sprintf("- answer meaning matches quotes: `%d/%d`\n", result.WithGroundedAnswer, result.TotalControls))
		b.WriteString(fmt.Sprintf("- unknown mode triggered: `%d`\n", result.UnknownCount))
		b.WriteString(fmt.Sprintf("- avg answer score: `%d`\n", result.AvgAnswerScore))
		b.WriteString(fmt.Sprintf("- avg source score: `%d`\n", result.AvgSourceScore))
		b.WriteString(fmt.Sprintf("- avg strict score: `%d`\n", result.AvgStrictScore))
		b.WriteString(fmt.Sprintf("- avg total score: `%d`\n", result.AvgTotalScore))
	}

	b.WriteString("\nConclusion: day24 returns structured grounded responses (answer + sources + quotes) and enforces 'Не знаю' when context relevance is weak.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay24Usage() {
	fmt.Println("Usage: openrouter-cli day24 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -question string              Single question for day24 run")
	fmt.Println("  -index string                 Path to Day21 index JSON")
	fmt.Println("  -top-k-before int             Top-K before rerank/filter")
	fmt.Println("  -top-k-after int              Top-K after rerank/filter")
	fmt.Println("  -similarity-threshold float   Filtering threshold")
	fmt.Println("  -unsure-threshold float       Below this score, assistant must answer 'Не знаю'")
	fmt.Println("  -model string                 Chat model")
	fmt.Println("  -rewrite-model string         Query rewrite model")
	fmt.Println("  -embedding-model string       Embedding model")
	fmt.Println("  -max-tokens int               Max tokens for structured answer")
	fmt.Println("  -temperature float            Temperature")
	fmt.Println("  -run-controls                 Run control questions benchmark")
	fmt.Println("  -controls-file string         Controls JSON path")
	fmt.Println("  -report string                Markdown report path")
	fmt.Println("  -help                         Show help")
}
