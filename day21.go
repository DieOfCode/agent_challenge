package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const openRouterEmbeddingsURL = "https://openrouter.ai/api/v1/embeddings"

const (
	day21DefaultChunkSize     = 1200
	day21DefaultChunkOverlap  = 200
	day21DefaultStructuredMax = 2000
)

type day21Document struct {
	Path  string
	Title string
	Text  string
	Ext   string
}

type day21Chunk struct {
	ChunkID      string
	Source       string
	Title        string
	Section      string
	Strategy     string
	ChunkIndex   int
	StartOffset  int
	EndOffset    int
	CharCount    int
	ApproxTokens int
	Text         string
}

type day21ChunkRecord struct {
	ChunkID      string    `json:"chunk_id"`
	Source       string    `json:"source"`
	Title        string    `json:"title"`
	Section      string    `json:"section"`
	Strategy     string    `json:"strategy"`
	ChunkIndex   int       `json:"chunk_index"`
	StartOffset  int       `json:"start_offset"`
	EndOffset    int       `json:"end_offset"`
	CharCount    int       `json:"char_count"`
	ApproxTokens int       `json:"approx_tokens"`
	Text         string    `json:"text"`
	Embedding    []float64 `json:"embedding"`
}

type day21Index struct {
	Strategy          string             `json:"strategy"`
	Model             string             `json:"model"`
	CreatedAtUTC      string             `json:"created_at_utc"`
	TotalChunks       int                `json:"total_chunks"`
	TotalChars        int                `json:"total_chars"`
	ApproxTokens      int                `json:"approx_tokens"`
	UsagePromptTokens int                `json:"usage_prompt_tokens,omitempty"`
	UsageTotalTokens  int                `json:"usage_total_tokens,omitempty"`
	EmbeddingDim      int                `json:"embedding_dim"`
	Chunks            []day21ChunkRecord `json:"chunks"`
}

type day21Stats struct {
	Docs              int
	TotalChars        int
	ApproxPages       float64
	Chunks            int
	AvgChunkChars     int
	MaxChunkChars     int
	ApproxTokens      int
	EmbeddingDim      int
	UsagePromptTokens int
	UsageTotalTokens  int
}

type embeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Usage usageStats `json:"usage,omitempty"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type day21RunResult struct {
	Docs            int
	TotalChars      int
	ApproxPages     float64
	Model           string
	FixedStats      day21Stats
	StructuredStats day21Stats
	FixedIndexPath  string
	StructIndexPath string
	Warnings        []string
}

func runDay21Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day21", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	inputDir := fs.String("input-dir", ".", "Root directory to scan")
	extsCSV := fs.String("exts", "md,txt,go,js,ts,py,rs,java,swift", "Comma-separated file extensions (no dots)")
	includePDF := fs.Bool("include-pdf", true, "Include PDF files (requires pdftotext)")
	maxFileSize := fs.Int64("max-file-size", 2_000_000, "Max file size to read (bytes)")
	maxFiles := fs.Int("max-files", 200, "Maximum number of files to index")
	chunkSize := fs.Int("chunk-size", day21DefaultChunkSize, "Chunk size (characters) for fixed strategy")
	chunkOverlap := fs.Int("chunk-overlap", day21DefaultChunkOverlap, "Chunk overlap (characters)")
	structuredMax := fs.Int("structured-max-size", day21DefaultStructuredMax, "Max chunk size for structured strategy")
	batchSize := fs.Int("batch-size", 16, "Embedding batch size")
	maxChunks := fs.Int("max-chunks", 0, "Optional max chunks per strategy (0 = no limit)")
	noEmbed := fs.Bool("no-embed", false, "Skip embedding generation")
	model := fs.String("embedding-model", defaultEmbeddingModel(), "Embedding model")
	outputDir := fs.String("output-dir", ".", "Output directory for index files")
	reportPath := fs.String("report", "DAY21_RESULTS.md", "Markdown report path")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day21 flags: %w", err)
	}
	if *help {
		printDay21Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day21 arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *chunkSize <= 0 {
		return fmt.Errorf("chunk-size must be positive")
	}
	if *structuredMax <= 0 {
		return fmt.Errorf("structured-max-size must be positive")
	}
	if *batchSize <= 0 {
		return fmt.Errorf("batch-size must be positive")
	}

	extSet := parseExtSet(*extsCSV, *includePDF)
	docs, warnings, err := collectDay21Documents(strings.TrimSpace(*inputDir), extSet, *includePDF, *maxFileSize, *maxFiles)
	if err != nil {
		return err
	}
	if len(docs) == 0 {
		return fmt.Errorf("no documents found to index")
	}

	totalChars := 0
	for _, doc := range docs {
		totalChars += len(doc.Text)
	}
	approxPages := float64(totalChars) / 2000.0
	if approxPages < 20 {
		warnings = append(warnings, fmt.Sprintf("total content is below 20 pages (approx %.1f pages)", approxPages))
	}

	fixedChunks := chunkFixedStrategy(docs, *chunkSize, *chunkOverlap, *maxChunks)
	structuredChunks := chunkStructuredStrategy(docs, *structuredMax, *chunkOverlap, *maxChunks)

	ctx := context.Background()
	apiKey := ""
	if !*noEmbed {
		apiKey = getAPIKey()
	}

	fixedIndexPath := filepath.Join(*outputDir, "DAY21_INDEX_fixed.json")
	structIndexPath := filepath.Join(*outputDir, "DAY21_INDEX_structured.json")

	fixedIndex, fixedStats, err := buildDay21Index(ctx, apiKey, *model, "fixed", fixedChunks, *batchSize, *noEmbed)
	if err != nil {
		return err
	}
	if err := writeDay21Index(fixedIndexPath, fixedIndex); err != nil {
		return err
	}

	structuredIndex, structuredStats, err := buildDay21Index(ctx, apiKey, *model, "structured", structuredChunks, *batchSize, *noEmbed)
	if err != nil {
		return err
	}
	if err := writeDay21Index(structIndexPath, structuredIndex); err != nil {
		return err
	}

	result := day21RunResult{
		Docs:            len(docs),
		TotalChars:      totalChars,
		ApproxPages:     approxPages,
		Model:           *model,
		FixedStats:      fixedStats,
		StructuredStats: structuredStats,
		FixedIndexPath:  fixedIndexPath,
		StructIndexPath: structIndexPath,
		Warnings:        warnings,
	}

	printDay21Result(result)
	if err := writeDay21Report(*reportPath, result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)
	return nil
}

func defaultEmbeddingModel() string {
	if model := strings.TrimSpace(os.Getenv("OPENROUTER_EMBEDDING_MODEL")); model != "" {
		return model
	}
	return "openai/text-embedding-3-small"
}

func parseExtSet(csv string, includePDF bool) map[string]struct{} {
	set := make(map[string]struct{})
	parts := strings.Split(csv, ",")
	for _, part := range parts {
		trimmed := strings.ToLower(strings.TrimSpace(part))
		if trimmed == "" {
			continue
		}
		if !strings.HasPrefix(trimmed, ".") {
			trimmed = "." + trimmed
		}
		set[trimmed] = struct{}{}
	}
	if includePDF {
		set[".pdf"] = struct{}{}
	}
	return set
}

func collectDay21Documents(root string, extSet map[string]struct{}, includePDF bool, maxFileSize int64, maxFiles int) ([]day21Document, []string, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return nil, nil, fmt.Errorf("input-dir is empty")
	}
	info, err := os.Stat(root)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to stat input-dir: %w", err)
	}
	if !info.IsDir() {
		return nil, nil, fmt.Errorf("input-dir is not a directory")
	}

	skippedDirs := map[string]struct{}{
		".git":         {},
		"node_modules": {},
		"vendor":       {},
		"dist":         {},
		"build":        {},
		".idea":        {},
		".codex":       {},
	}

	docs := make([]day21Document, 0)
	warnings := make([]string, 0)

	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			if _, skip := skippedDirs[d.Name()]; skip {
				return filepath.SkipDir
			}
			return nil
		}
		if maxFiles > 0 && len(docs) >= maxFiles {
			return errors.New("max files reached")
		}
		ext := strings.ToLower(filepath.Ext(path))
		if _, ok := extSet[ext]; !ok {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			warnings = append(warnings, fmt.Sprintf("failed to stat file %s", path))
			return nil
		}
		if maxFileSize > 0 && info.Size() > maxFileSize {
			warnings = append(warnings, fmt.Sprintf("skipping large file %s", path))
			return nil
		}

		text, err := readDay21File(path, ext, includePDF)
		if err != nil {
			warnings = append(warnings, fmt.Sprintf("skipping file %s: %v", path, err))
			return nil
		}
		text = strings.TrimSpace(text)
		if text == "" {
			return nil
		}
		docs = append(docs, day21Document{
			Path:  path,
			Title: filepath.Base(path),
			Text:  text,
			Ext:   ext,
		})
		return nil
	})
	if err != nil {
		if err.Error() != "max files reached" {
			return nil, warnings, err
		}
	}

	return docs, warnings, nil
}

func readDay21File(path, ext string, includePDF bool) (string, error) {
	if ext == ".pdf" {
		if !includePDF {
			return "", fmt.Errorf("pdf skipped")
		}
		return readPDFText(path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func readPDFText(path string) (string, error) {
	pdftotextPath, err := exec.LookPath("pdftotext")
	if err != nil {
		return "", fmt.Errorf("pdftotext not found")
	}
	cmd := exec.Command(pdftotextPath, "-layout", "-nopgbrk", path, "-")
	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("pdftotext failed")
	}
	return string(output), nil
}

func chunkFixedStrategy(docs []day21Document, size, overlap, maxChunks int) []day21Chunk {
	chunks := make([]day21Chunk, 0)
	chunkIndex := 0
	for _, doc := range docs {
		segments := splitFixed(doc.Text, size, overlap)
		for _, seg := range segments {
			chunkID := fmt.Sprintf("fixed:%s:%d", doc.Title, chunkIndex)
			chunks = append(chunks, buildChunk(doc, "fixed", "fixed", chunkID, chunkIndex, seg))
			chunkIndex++
			if maxChunks > 0 && len(chunks) >= maxChunks {
				return chunks
			}
		}
	}
	return chunks
}

func chunkStructuredStrategy(docs []day21Document, size, overlap, maxChunks int) []day21Chunk {
	chunks := make([]day21Chunk, 0)
	chunkIndex := 0
	for _, doc := range docs {
		sections := splitStructuredSections(doc)
		for _, section := range sections {
			segments := splitFixed(section.Text, size, overlap)
			for _, seg := range segments {
				chunkID := fmt.Sprintf("structured:%s:%d", doc.Title, chunkIndex)
				chunks = append(chunks, buildChunk(doc, "structured", section.Title, chunkID, chunkIndex, seg))
				chunkIndex++
				if maxChunks > 0 && len(chunks) >= maxChunks {
					return chunks
				}
			}
		}
	}
	return chunks
}

type textSegment struct {
	Text  string
	Start int
	End   int
}

func splitFixed(text string, size, overlap int) []textSegment {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	if overlap >= size {
		overlap = 0
	}
	step := size - overlap
	if step <= 0 {
		step = size
	}
	segments := make([]textSegment, 0)
	for start := 0; start < len(text); start += step {
		end := start + size
		if end > len(text) {
			end = len(text)
		}
		chunk := strings.TrimSpace(text[start:end])
		if chunk == "" {
			continue
		}
		segments = append(segments, textSegment{Text: chunk, Start: start, End: end})
		if end == len(text) {
			break
		}
	}
	return segments
}

type structuredSection struct {
	Title string
	Text  string
}

func splitStructuredSections(doc day21Document) []structuredSection {
	if doc.Ext == ".md" || doc.Ext == ".markdown" {
		return splitMarkdownSections(doc.Text)
	}
	return []structuredSection{{Title: doc.Title, Text: doc.Text}}
}

func splitMarkdownSections(text string) []structuredSection {
	lines := strings.Split(text, "\n")
	sections := make([]structuredSection, 0)
	currentTitle := "Introduction"
	var buf []string
	flush := func() {
		content := strings.TrimSpace(strings.Join(buf, "\n"))
		if content != "" {
			sections = append(sections, structuredSection{Title: currentTitle, Text: currentTitle + "\n" + content})
		}
		buf = nil
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") {
			flush()
			title := strings.TrimSpace(strings.TrimLeft(trimmed, "#"))
			if title == "" {
				title = "Section"
			}
			currentTitle = title
			continue
		}
		buf = append(buf, line)
	}
	flush()
	if len(sections) == 0 {
		return []structuredSection{{Title: "Introduction", Text: text}}
	}
	return sections
}

func buildChunk(doc day21Document, strategy, section, chunkID string, idx int, seg textSegment) day21Chunk {
	charCount := len(seg.Text)
	return day21Chunk{
		ChunkID:      chunkID,
		Source:       doc.Path,
		Title:        doc.Title,
		Section:      section,
		Strategy:     strategy,
		ChunkIndex:   idx,
		StartOffset:  seg.Start,
		EndOffset:    seg.End,
		CharCount:    charCount,
		ApproxTokens: approxTokens(seg.Text),
		Text:         seg.Text,
	}
}

func approxTokens(text string) int {
	if text == "" {
		return 0
	}
	return (len(text) + 3) / 4
}

func buildDay21Index(ctx context.Context, apiKey, model, strategy string, chunks []day21Chunk, batchSize int, noEmbed bool) (day21Index, day21Stats, error) {
	index := day21Index{
		Strategy:     strategy,
		Model:        model,
		CreatedAtUTC: time.Now().UTC().Format(time.RFC3339),
		Chunks:       make([]day21ChunkRecord, len(chunks)),
	}

	stats := day21Stats{}
	stats.Chunks = len(chunks)

	for i, chunk := range chunks {
		stats.TotalChars += chunk.CharCount
		stats.ApproxTokens += chunk.ApproxTokens
		if chunk.CharCount > stats.MaxChunkChars {
			stats.MaxChunkChars = chunk.CharCount
		}
		index.Chunks[i] = day21ChunkRecord{
			ChunkID:      chunk.ChunkID,
			Source:       chunk.Source,
			Title:        chunk.Title,
			Section:      chunk.Section,
			Strategy:     chunk.Strategy,
			ChunkIndex:   chunk.ChunkIndex,
			StartOffset:  chunk.StartOffset,
			EndOffset:    chunk.EndOffset,
			CharCount:    chunk.CharCount,
			ApproxTokens: chunk.ApproxTokens,
			Text:         chunk.Text,
		}
	}
	if stats.Chunks > 0 {
		stats.AvgChunkChars = stats.TotalChars / stats.Chunks
	}

	if noEmbed {
		index.TotalChunks = stats.Chunks
		index.TotalChars = stats.TotalChars
		index.ApproxTokens = stats.ApproxTokens
		return index, stats, nil
	}

	if apiKey == "" {
		return day21Index{}, day21Stats{}, fmt.Errorf("OPENROUTER_API_KEY is not set")
	}

	usagePrompt := 0
	usageTotal := 0
	embeddingDim := 0

	for i := 0; i < len(chunks); i += batchSize {
		end := i + batchSize
		if end > len(chunks) {
			end = len(chunks)
		}
		batch := chunks[i:end]
		texts := make([]string, len(batch))
		for j, chunk := range batch {
			texts[j] = chunk.Text
		}
		vectors, usage, err := embedBatch(ctx, apiKey, model, texts)
		if err != nil {
			return day21Index{}, day21Stats{}, err
		}
		usagePrompt += usage.PromptTokens
		usageTotal += usage.TotalTokens
		for j := range batch {
			index.Chunks[i+j].Embedding = vectors[j]
			if embeddingDim == 0 {
				embeddingDim = len(vectors[j])
			}
		}
	}

	index.TotalChunks = stats.Chunks
	index.TotalChars = stats.TotalChars
	index.ApproxTokens = stats.ApproxTokens
	index.UsagePromptTokens = usagePrompt
	index.UsageTotalTokens = usageTotal
	index.EmbeddingDim = embeddingDim

	stats.UsagePromptTokens = usagePrompt
	stats.UsageTotalTokens = usageTotal
	stats.EmbeddingDim = embeddingDim

	return index, stats, nil
}

func embedBatch(ctx context.Context, apiKey, model string, texts []string) ([][]float64, usageStats, error) {
	reqPayload := embeddingRequest{Model: model, Input: texts}
	body, err := json.Marshal(reqPayload)
	if err != nil {
		return nil, usageStats{}, fmt.Errorf("failed to encode embeddings request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, openRouterEmbeddingsURL, bytes.NewReader(body))
	if err != nil {
		return nil, usageStats{}, fmt.Errorf("failed to create embeddings request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("HTTP-Referer", "https://localhost")
	req.Header.Set("X-Title", "openrouter-cli-day21")

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, usageStats{}, fmt.Errorf("embeddings request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, usageStats{}, fmt.Errorf("failed to read embeddings response: %w", err)
	}

	var out embeddingResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, usageStats{}, fmt.Errorf("invalid embeddings JSON response: %w", err)
	}
	if resp.StatusCode >= 400 {
		if out.Error != nil && out.Error.Message != "" {
			return nil, usageStats{}, fmt.Errorf("embeddings API error (%s): %s", resp.Status, out.Error.Message)
		}
		return nil, usageStats{}, fmt.Errorf("embeddings API error (%s): %s", resp.Status, string(raw))
	}

	vectors := make([][]float64, len(texts))
	for _, item := range out.Data {
		if item.Index < 0 || item.Index >= len(texts) {
			continue
		}
		vectors[item.Index] = item.Embedding
	}
	for i, vec := range vectors {
		if len(vec) == 0 {
			return nil, usageStats{}, fmt.Errorf("missing embedding for index %d", i)
		}
	}

	usage := out.Usage
	if usage.TotalTokens == 0 && (usage.PromptTokens > 0 || usage.CompletionTokens > 0) {
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	return vectors, usage, nil
}

func writeDay21Index(path string, index day21Index) error {
	data, err := json.MarshalIndent(index, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode index: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

func printDay21Result(result day21RunResult) {
	fmt.Println("=== Day 21: Document Indexing ===")
	fmt.Printf("documents=%d total_chars=%d approx_pages=%.1f\n", result.Docs, result.TotalChars, result.ApproxPages)
	fmt.Printf("model=%s\n", result.Model)
	fmt.Printf("fixed_chunks=%d structured_chunks=%d\n", result.FixedStats.Chunks, result.StructuredStats.Chunks)
	fmt.Printf("fixed_index=%s\n", result.FixedIndexPath)
	fmt.Printf("structured_index=%s\n", result.StructuredStatsIndexPath())
	for _, warn := range result.Warnings {
		fmt.Printf("warning=%s\n", warn)
	}
}

func (r day21RunResult) StructuredStatsIndexPath() string {
	return r.StructIndexPath
}

func writeDay21Report(path string, result day21RunResult) error {
	var b strings.Builder
	b.WriteString("# Day 21 Results: Document Indexing\n\n")
	b.WriteString(fmt.Sprintf("- documents indexed: `%d`\n", result.Docs))
	b.WriteString(fmt.Sprintf("- total chars: `%d`\n", result.TotalChars))
	b.WriteString(fmt.Sprintf("- approx pages: `%.1f`\n", result.ApproxPages))
	b.WriteString(fmt.Sprintf("- embedding model: `%s`\n", result.Model))
	b.WriteString(fmt.Sprintf("- fixed index: `%s`\n", result.FixedIndexPath))
	b.WriteString(fmt.Sprintf("- structured index: `%s`\n\n", result.StructIndexPath))

	if len(result.Warnings) > 0 {
		b.WriteString("## Warnings\n")
		for _, warn := range result.Warnings {
			b.WriteString("- " + warn + "\n")
		}
		b.WriteString("\n")
	}

	b.WriteString("## Fixed Chunking Strategy\n")
	writeDay21Stats(&b, result.FixedStats)
	b.WriteString("\n## Structured Chunking Strategy\n")
	writeDay21Stats(&b, result.StructuredStats)

	b.WriteString("\n## Comparison\n")
	b.WriteString(fmt.Sprintf("- fixed chunks: `%d` | structured chunks: `%d`\n", result.FixedStats.Chunks, result.StructuredStats.Chunks))
	b.WriteString(fmt.Sprintf("- fixed avg chunk chars: `%d` | structured avg chunk chars: `%d`\n", result.FixedStats.AvgChunkChars, result.StructuredStats.AvgChunkChars))
	b.WriteString(fmt.Sprintf("- fixed max chunk chars: `%d` | structured max chunk chars: `%d`\n", result.FixedStats.MaxChunkChars, result.StructuredStats.MaxChunkChars))
	b.WriteString(fmt.Sprintf("- fixed approx tokens: `%d` | structured approx tokens: `%d`\n", result.FixedStats.ApproxTokens, result.StructuredStats.ApproxTokens))
	b.WriteString("\nConclusion: both strategies produced embeddings with metadata; fixed chunking is uniform, while structured chunking preserves section boundaries.\n")

	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func writeDay21Stats(b *strings.Builder, stats day21Stats) {
	b.WriteString(fmt.Sprintf("- chunks: `%d`\n", stats.Chunks))
	b.WriteString(fmt.Sprintf("- avg chunk chars: `%d`\n", stats.AvgChunkChars))
	b.WriteString(fmt.Sprintf("- max chunk chars: `%d`\n", stats.MaxChunkChars))
	b.WriteString(fmt.Sprintf("- approx tokens: `%d`\n", stats.ApproxTokens))
	if stats.EmbeddingDim > 0 {
		b.WriteString(fmt.Sprintf("- embedding dim: `%d`\n", stats.EmbeddingDim))
	}
	if stats.UsageTotalTokens > 0 {
		b.WriteString(fmt.Sprintf("- embedding usage total tokens: `%d`\n", stats.UsageTotalTokens))
	}
}

func printDay21Usage() {
	fmt.Println("Usage: openrouter-cli day21 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -input-dir string          Root directory to scan")
	fmt.Println("  -exts string               File extensions to include (comma-separated, no dots)")
	fmt.Println("  -include-pdf               Include PDF files if pdftotext is available")
	fmt.Println("  -max-file-size int         Max file size to read (bytes)")
	fmt.Println("  -max-files int             Max number of files to index")
	fmt.Println("  -chunk-size int            Chunk size for fixed strategy")
	fmt.Println("  -chunk-overlap int         Overlap size for fixed strategy")
	fmt.Println("  -structured-max-size int   Chunk size for structured strategy")
	fmt.Println("  -batch-size int            Embedding batch size")
	fmt.Println("  -max-chunks int            Optional max chunks per strategy")
	fmt.Println("  -embedding-model string    Embedding model")
	fmt.Println("  -output-dir string         Output directory for index files")
	fmt.Println("  -report string             Markdown report path")
	fmt.Println("  -no-embed                  Skip embedding generation")
	fmt.Println("  -help                      Show help")
}
