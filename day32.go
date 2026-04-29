package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

const (
	day32DefaultTopK         = 8
	day32DefaultMaxDiffChars = 20000
	day32DefaultMaxFiles     = 40
	day32DefaultMaxFileChars = 2400
)

type day32PRMeta struct {
	Number  int
	Title   string
	URL     string
	BaseRef string
	HeadRef string
	BaseSHA string
	HeadSHA string
}

type day32ReviewResult struct {
	Meta         day32PRMeta
	Base         string
	Head         string
	ChangedFiles []string
	Diff         string
	Retrieved    []day31RetrievedChunk
	ReviewText   string
	Simulated    bool
	Latency      time.Duration
	Usage        usageStats
	Warnings     []string
}

type day32GitHubEvent struct {
	PullRequest struct {
		Number  int    `json:"number"`
		Title   string `json:"title"`
		HTMLURL string `json:"html_url"`
		Base    struct {
			Ref string `json:"ref"`
			SHA string `json:"sha"`
		} `json:"base"`
		Head struct {
			Ref string `json:"ref"`
			SHA string `json:"sha"`
		} `json:"head"`
	} `json:"pull_request"`
}

func runDay32Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day32", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	workspace := fs.String("workspace", ".", "Project workspace root")
	fromGitHubEvent := fs.Bool("from-github-event", false, "Read PR base/head from GitHub event payload")
	eventPath := fs.String("event-path", strings.TrimSpace(os.Getenv("GITHUB_EVENT_PATH")), "Path to GitHub event payload JSON")
	base := fs.String("base", "", "Base ref/SHA for diff")
	head := fs.String("head", "HEAD", "Head ref/SHA for diff")
	diffFile := fs.String("diff-file", "", "Optional path to prebuilt diff file")
	filesFile := fs.String("files-file", "", "Optional path to changed-files list")
	readmePath := fs.String("readme", "README.md", "README path (relative to workspace)")
	docsDir := fs.String("docs", "docs", "Docs dir (relative to workspace)")
	topK := fs.Int("top-k", day32DefaultTopK, "Top-K RAG chunks for review")
	maxDiffChars := fs.Int("max-diff-chars", day32DefaultMaxDiffChars, "Max diff chars to include in prompt")
	maxFiles := fs.Int("max-files", day32DefaultMaxFiles, "Max changed files to load into code context")
	maxFileChars := fs.Int("max-file-chars", day32DefaultMaxFileChars, "Max chars per changed file for context")
	model := fs.String("model", getDefaultModel(), "OpenRouter chat model")
	maxTokens := fs.Int("max-tokens", 900, "Max response tokens")
	temperature := fs.Float64("temperature", 0.1, "Model temperature")
	simulate := fs.Bool("simulate", false, "No API calls, use local heuristic reviewer")
	reportPath := fs.String("report", "DAY32_RESULTS.md", "Markdown report output path")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day32 flags: %w", err)
	}
	if *help {
		printDay32Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day32 arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *topK <= 0 {
		return fmt.Errorf("top-k must be positive")
	}
	if *maxDiffChars <= 0 {
		return fmt.Errorf("max-diff-chars must be positive")
	}
	if *maxFiles <= 0 {
		return fmt.Errorf("max-files must be positive")
	}
	if *maxFileChars <= 0 {
		return fmt.Errorf("max-file-chars must be positive")
	}
	if *maxTokens <= 0 {
		return fmt.Errorf("max-tokens must be positive")
	}

	workspaceAbs, err := filepath.Abs(strings.TrimSpace(*workspace))
	if err != nil {
		return fmt.Errorf("failed to resolve workspace: %w", err)
	}

	meta := day32PRMeta{}
	baseRef := strings.TrimSpace(*base)
	headRef := strings.TrimSpace(*head)
	if headRef == "" {
		headRef = "HEAD"
	}

	warnings := make([]string, 0)
	if *fromGitHubEvent {
		parsed, err := loadDay32MetaFromEvent(strings.TrimSpace(*eventPath))
		if err != nil {
			warnings = append(warnings, fmt.Sprintf("failed to parse GitHub event: %v", err))
		} else {
			meta = parsed
			if strings.TrimSpace(meta.BaseSHA) != "" {
				baseRef = strings.TrimSpace(meta.BaseSHA)
			}
			if strings.TrimSpace(meta.HeadSHA) != "" {
				headRef = strings.TrimSpace(meta.HeadSHA)
			}
		}
	}
	if baseRef == "" {
		baseRef = day32InferBaseRef(workspaceAbs)
		warnings = append(warnings, fmt.Sprintf("base ref was empty; inferred base=%s", baseRef))
	}

	diffText, changedFiles, err := day32CollectDiffAndFiles(workspaceAbs, baseRef, headRef, strings.TrimSpace(*diffFile), strings.TrimSpace(*filesFile))
	if err != nil {
		return err
	}
	if strings.TrimSpace(diffText) == "" {
		warnings = append(warnings, "diff is empty")
	}
	if len(diffText) > *maxDiffChars {
		diffText = diffText[:*maxDiffChars]
		warnings = append(warnings, fmt.Sprintf("diff truncated to %d chars", *maxDiffChars))
	}

	docChunks, err := loadDay31Corpus(workspaceAbs, strings.TrimSpace(*readmePath), strings.TrimSpace(*docsDir))
	if err != nil {
		return err
	}
	codeChunks := day32LoadChangedCodeChunks(workspaceAbs, changedFiles, *maxFiles, *maxFileChars)
	if len(codeChunks) == 0 {
		warnings = append(warnings, "changed code context is empty")
	}

	allChunks := append([]day31Chunk{}, docChunks...)
	allChunks = append(allChunks, codeChunks...)
	reviewQuery := day32BuildReviewQuery(meta, changedFiles, diffText)
	retrieved := day31RetrieveChunks(allChunks, reviewQuery, *topK)

	simulateMode := *simulate
	apiKey := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	if !simulateMode && apiKey == "" {
		simulateMode = true
		warnings = append(warnings, "OPENROUTER_API_KEY is not set, switched to simulate mode")
	}

	result := day32ReviewResult{
		Meta:         meta,
		Base:         baseRef,
		Head:         headRef,
		ChangedFiles: changedFiles,
		Diff:         diffText,
		Retrieved:    retrieved,
		Simulated:    simulateMode,
		Warnings:     warnings,
	}

	if simulateMode {
		result.ReviewText = day32BuildHeuristicReview(meta, baseRef, headRef, changedFiles, diffText, retrieved)
	} else {
		temp := *temperature
		llmResult, err := day32GenerateReviewWithLLM(apiKey, strings.TrimSpace(*model), *maxTokens, &temp, meta, baseRef, headRef, changedFiles, diffText, retrieved)
		if err != nil {
			return err
		}
		result.ReviewText = strings.TrimSpace(llmResult.Answer)
		result.Usage = llmResult.Usage
		result.Latency = llmResult.Latency
	}

	printDay32Result(result)
	if err := writeDay32Report(strings.TrimSpace(*reportPath), result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", strings.TrimSpace(*reportPath))
	return nil
}

func loadDay32MetaFromEvent(path string) (day32PRMeta, error) {
	if strings.TrimSpace(path) == "" {
		return day32PRMeta{}, fmt.Errorf("event path is empty")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return day32PRMeta{}, fmt.Errorf("failed to read event payload: %w", err)
	}
	var ev day32GitHubEvent
	if err := json.Unmarshal(raw, &ev); err != nil {
		return day32PRMeta{}, fmt.Errorf("failed to parse event payload: %w", err)
	}
	if ev.PullRequest.Number == 0 {
		return day32PRMeta{}, fmt.Errorf("pull_request block not found in event payload")
	}
	return day32PRMeta{
		Number:  ev.PullRequest.Number,
		Title:   strings.TrimSpace(ev.PullRequest.Title),
		URL:     strings.TrimSpace(ev.PullRequest.HTMLURL),
		BaseRef: strings.TrimSpace(ev.PullRequest.Base.Ref),
		HeadRef: strings.TrimSpace(ev.PullRequest.Head.Ref),
		BaseSHA: strings.TrimSpace(ev.PullRequest.Base.SHA),
		HeadSHA: strings.TrimSpace(ev.PullRequest.Head.SHA),
	}, nil
}

func day32InferBaseRef(workspace string) string {
	out, err := day32RunGit(workspace, "symbolic-ref", "refs/remotes/origin/HEAD")
	if err == nil {
		ref := strings.TrimSpace(out)
		ref = strings.TrimPrefix(ref, "refs/remotes/")
		if ref != "" {
			return ref
		}
	}
	return "HEAD~1"
}

func day32CollectDiffAndFiles(workspace, base, head, diffFile, filesFile string) (string, []string, error) {
	var diffText string
	var changedFiles []string
	var err error

	if diffFile != "" {
		raw, readErr := os.ReadFile(diffFile)
		if readErr != nil {
			return "", nil, fmt.Errorf("failed to read diff-file: %w", readErr)
		}
		diffText = string(raw)
	} else {
		diffText, err = day32RunGit(workspace, "diff", "--no-color", "--unified=0", base+"..."+head)
		if err != nil {
			return "", nil, fmt.Errorf("failed to collect git diff: %w", err)
		}
	}

	if filesFile != "" {
		raw, readErr := os.ReadFile(filesFile)
		if readErr != nil {
			return "", nil, fmt.Errorf("failed to read files-file: %w", readErr)
		}
		changedFiles = day32ParseLines(string(raw))
	} else {
		out, runErr := day32RunGit(workspace, "diff", "--name-only", base+"..."+head)
		if runErr != nil {
			return "", nil, fmt.Errorf("failed to collect changed files: %w", runErr)
		}
		changedFiles = day32ParseLines(out)
	}

	if len(changedFiles) == 0 {
		return strings.TrimSpace(diffText), changedFiles, nil
	}
	sort.Strings(changedFiles)
	return strings.TrimSpace(diffText), changedFiles, nil
}

func day32ParseLines(text string) []string {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			out = append(out, filepath.ToSlash(trimmed))
		}
	}
	return out
}

func day32RunGit(workspace string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", append([]string{"-C", workspace}, args...)...)
	output, err := cmd.CombinedOutput()
	text := strings.TrimSpace(string(output))
	if err != nil {
		if text == "" {
			text = err.Error()
		}
		return "", fmt.Errorf("git %s failed: %s", strings.Join(args, " "), text)
	}
	return text, nil
}

func day32LoadChangedCodeChunks(workspace string, files []string, maxFiles, maxChars int) []day31Chunk {
	chunks := make([]day31Chunk, 0)
	if maxFiles <= 0 {
		maxFiles = day32DefaultMaxFiles
	}
	if maxChars <= 0 {
		maxChars = day32DefaultMaxFileChars
	}

	count := 0
	for _, rel := range files {
		if count >= maxFiles {
			break
		}
		if day32SkipFileForContext(rel) {
			continue
		}
		abs := filepath.Join(workspace, filepath.FromSlash(rel))
		data, err := os.ReadFile(abs)
		if err != nil {
			continue
		}
		if len(data) == 0 || day32LooksBinary(data) {
			continue
		}

		text := strings.TrimSpace(string(data))
		if text == "" {
			continue
		}
		if len(text) > maxChars {
			text = text[:maxChars]
		}
		prefixed := fmt.Sprintf("FILE: %s\n%s", rel, text)
		for i, part := range day31SplitText(prefixed, 900, 120) {
			chunkID := fmt.Sprintf("code:%s#%03d", rel, i)
			chunks = append(chunks, day31Chunk{
				ChunkID: chunkID,
				Source:  rel,
				Section: "code",
				Text:    part,
			})
		}
		count++
	}
	return chunks
}

func day32SkipFileForContext(path string) bool {
	path = strings.ToLower(strings.TrimSpace(path))
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".jar", ".ico", ".mp4", ".mov", ".woff", ".woff2", ".ttf":
		return true
	}
	if strings.Contains(path, "node_modules/") || strings.HasPrefix(path, "vendor/") || strings.Contains(path, "/vendor/") {
		return true
	}
	if strings.HasSuffix(path, ".min.js") {
		return true
	}
	return false
}

func day32LooksBinary(data []byte) bool {
	limit := len(data)
	if limit > 4096 {
		limit = 4096
	}
	for i := 0; i < limit; i++ {
		if data[i] == 0 {
			return true
		}
	}
	return false
}

func day32BuildReviewQuery(meta day32PRMeta, files []string, diffText string) string {
	var b strings.Builder
	b.WriteString("PR review context\n")
	if meta.Number > 0 {
		b.WriteString(fmt.Sprintf("PR #%d\n", meta.Number))
	}
	if meta.Title != "" {
		b.WriteString("title: " + meta.Title + "\n")
	}
	if meta.BaseRef != "" || meta.HeadRef != "" {
		b.WriteString(fmt.Sprintf("base=%s head=%s\n", meta.BaseRef, meta.HeadRef))
	}
	if len(files) > 0 {
		b.WriteString("files: " + strings.Join(files, " ") + "\n")
	}
	if diffText != "" {
		if len(diffText) > 2500 {
			diffText = diffText[:2500]
		}
		b.WriteString("diff: " + diffText)
	}
	return b.String()
}

func day32GenerateReviewWithLLM(apiKey, model string, maxTokens int, temperature *float64, meta day32PRMeta, base, head string, files []string, diffText string, retrieved []day31RetrievedChunk) (openRouterResult, error) {
	systemPrompt := "Ты Senior reviewer. Сделай практичное code review только по фактам из diff и контекста. Верни Markdown строго с разделами: `## Потенциальные баги`, `## Архитектурные проблемы`, `## Рекомендации`. Для каждого пункта укажи файл/зону риска. Если раздел пустой, напиши `- Не обнаружено`"
	userPrompt := day32BuildLLMUserPrompt(meta, base, head, files, diffText, retrieved)

	return callOpenRouterDetailed(
		apiKey,
		model,
		[]message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
		maxTokens,
		temperature,
		nil,
		"openrouter-cli-day32-review",
	)
}

func day32BuildLLMUserPrompt(meta day32PRMeta, base, head string, files []string, diffText string, retrieved []day31RetrievedChunk) string {
	var b strings.Builder
	b.WriteString("PR metadata:\n")
	if meta.Number > 0 {
		b.WriteString(fmt.Sprintf("number=%d\n", meta.Number))
	}
	if meta.Title != "" {
		b.WriteString("title=" + meta.Title + "\n")
	}
	if meta.URL != "" {
		b.WriteString("url=" + meta.URL + "\n")
	}
	b.WriteString(fmt.Sprintf("base=%s\nhead=%s\n\n", base, head))

	b.WriteString("Changed files:\n")
	if len(files) == 0 {
		b.WriteString("(none)\n")
	} else {
		for _, f := range files {
			b.WriteString("- " + f + "\n")
		}
	}
	b.WriteString("\n")

	b.WriteString("Diff:\n")
	if strings.TrimSpace(diffText) == "" {
		b.WriteString("(empty diff)\n")
	} else {
		b.WriteString(diffText)
		b.WriteString("\n")
	}
	b.WriteString("\nRAG context (docs + code):\n")
	b.WriteString(day31RenderContext(retrieved))
	return b.String()
}

func day32BuildHeuristicReview(meta day32PRMeta, base, head string, files []string, diffText string, retrieved []day31RetrievedChunk) string {
	addedLines := day32CollectPrefixedLines(diffText, '+')
	removedLines := day32CollectPrefixedLines(diffText, '-')

	potentialBugs := make([]string, 0)
	arch := make([]string, 0)
	recs := make([]string, 0)

	panicRE := regexp.MustCompile(`\bpanic\s*\(`)
	printRE := regexp.MustCompile(`\bfmt\.Print(ln|f)?\s*\(`)
	todoRE := regexp.MustCompile(`\bTODO\b`)
	sleepRE := regexp.MustCompile(`\btime\.Sleep\s*\(`)

	for _, line := range addedLines {
		if panicRE.MatchString(line) {
			potentialBugs = append(potentialBugs, "Добавлен `panic(...)`: риск падения приложения при runtime-ошибках.")
			break
		}
	}
	for _, line := range removedLines {
		if strings.Contains(line, "if err != nil") {
			potentialBugs = append(potentialBugs, "Удалена/изменена явная обработка `if err != nil`: проверьте, что ошибка не проглатывается.")
			break
		}
	}
	for _, line := range addedLines {
		if sleepRE.MatchString(line) {
			potentialBugs = append(potentialBugs, "Добавлен `time.Sleep(...)`: возможна деградация latency/throughput.")
			break
		}
	}

	if len(files) > 20 {
		arch = append(arch, fmt.Sprintf("Большой PR (%d файлов): review и rollback становятся сложнее, стоит дробить изменения.", len(files)))
	}
	if strings.Contains(strings.ToLower(diffText), "main.go") && len(files) > 8 {
		arch = append(arch, "Изменения затрагивают `main.go` вместе со многими файлами: есть риск избыточной связности CLI-маршрутизации.")
	}

	for _, line := range addedLines {
		if printRE.MatchString(line) {
			recs = append(recs, "В прод-коде лучше заменить `fmt.Print*` на структурированный логгер или ограничить debug-вывод флагом.")
			break
		}
	}
	for _, line := range addedLines {
		if todoRE.MatchString(line) {
			recs = append(recs, "Обнаружен `TODO` в добавленном коде: стоит создать явную задачу и связать с PR.")
			break
		}
	}
	if len(retrieved) > 0 {
		recs = append(recs, "Проверьте соответствие изменений документации/контрактам в найденных источниках RAG (README/docs).")
	}
	if len(recs) == 0 {
		recs = append(recs, "Добавьте targeted-тесты на измененные ветки кода и граничные кейсы.")
	}

	var b strings.Builder
	b.WriteString("## Потенциальные баги\n")
	if len(potentialBugs) == 0 {
		b.WriteString("- Не обнаружено\n")
	} else {
		for _, item := range day32Unique(potentialBugs) {
			b.WriteString("- " + item + "\n")
		}
	}

	b.WriteString("\n## Архитектурные проблемы\n")
	if len(arch) == 0 {
		b.WriteString("- Не обнаружено\n")
	} else {
		for _, item := range day32Unique(arch) {
			b.WriteString("- " + item + "\n")
		}
	}

	b.WriteString("\n## Рекомендации\n")
	for _, item := range day32Unique(recs) {
		b.WriteString("- " + item + "\n")
	}

	b.WriteString("\n---\n")
	if meta.Number > 0 {
		b.WriteString(fmt.Sprintf("PR: #%d\n", meta.Number))
	}
	if meta.Title != "" {
		b.WriteString("Title: " + meta.Title + "\n")
	}
	b.WriteString(fmt.Sprintf("Diff range: %s...%s\n", base, head))
	if len(files) > 0 {
		b.WriteString(fmt.Sprintf("Changed files: %d\n", len(files)))
	}
	if len(retrieved) > 0 {
		b.WriteString("RAG sources: " + day31RenderSourcesInline(retrieved) + "\n")
	}
	return strings.TrimSpace(b.String())
}

func day32CollectPrefixedLines(diffText string, prefix byte) []string {
	lines := strings.Split(strings.ReplaceAll(diffText, "\r\n", "\n"), "\n")
	out := make([]string, 0)
	for _, line := range lines {
		if len(line) == 0 || line[0] != prefix {
			continue
		}
		if strings.HasPrefix(line, "+++") || strings.HasPrefix(line, "---") {
			continue
		}
		out = append(out, strings.TrimSpace(strings.TrimPrefix(line, string(prefix))))
	}
	return out
}

func day32Unique(items []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(items))
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		if _, ok := seen[item]; ok {
			continue
		}
		seen[item] = struct{}{}
		out = append(out, item)
	}
	return out
}

func printDay32Result(result day32ReviewResult) {
	fmt.Println("=== Day 32: Automated PR Review ===")
	if result.Meta.Number > 0 {
		fmt.Printf("pr=%d title=%q\n", result.Meta.Number, result.Meta.Title)
	}
	fmt.Printf("range=%s...%s\n", result.Base, result.Head)
	fmt.Printf("changed_files=%d diff_chars=%d rag_chunks=%d mode=%s\n",
		len(result.ChangedFiles),
		len(result.Diff),
		len(result.Retrieved),
		map[bool]string{true: "simulate", false: "llm"}[result.Simulated],
	)
	if len(result.Warnings) > 0 {
		fmt.Println("warnings:")
		for _, warning := range result.Warnings {
			fmt.Printf("- %s\n", warning)
		}
	}
	fmt.Println("")
	fmt.Println(result.ReviewText)
	fmt.Println("")
	if !result.Simulated {
		fmt.Printf("usage> prompt=%d completion=%d total=%d latency=%s\n", result.Usage.PromptTokens, result.Usage.CompletionTokens, result.Usage.TotalTokens, result.Latency.Round(time.Millisecond))
	}
}

func writeDay32Report(path string, result day32ReviewResult) error {
	if strings.TrimSpace(path) == "" {
		path = "DAY32_RESULTS.md"
	}
	var b strings.Builder
	b.WriteString("# Day 32 Results: Automated AI Code Review\n\n")
	if result.Meta.Number > 0 {
		b.WriteString(fmt.Sprintf("- PR: `#%d`\n", result.Meta.Number))
	}
	if result.Meta.Title != "" {
		b.WriteString(fmt.Sprintf("- Title: `%s`\n", result.Meta.Title))
	}
	if result.Meta.URL != "" {
		b.WriteString(fmt.Sprintf("- URL: %s\n", result.Meta.URL))
	}
	b.WriteString(fmt.Sprintf("- Diff range: `%s...%s`\n", result.Base, result.Head))
	b.WriteString(fmt.Sprintf("- Changed files: `%d`\n", len(result.ChangedFiles)))
	b.WriteString(fmt.Sprintf("- Diff chars: `%d`\n", len(result.Diff)))
	b.WriteString(fmt.Sprintf("- RAG chunks used: `%d`\n", len(result.Retrieved)))
	b.WriteString(fmt.Sprintf("- Mode: `%s`\n", map[bool]string{true: "simulate", false: "llm"}[result.Simulated]))
	if len(result.Warnings) > 0 {
		b.WriteString("- Warnings:\n")
		for _, warning := range result.Warnings {
			b.WriteString("  - " + warning + "\n")
		}
	}
	if !result.Simulated {
		b.WriteString(fmt.Sprintf("- Tokens: `prompt=%d completion=%d total=%d`\n", result.Usage.PromptTokens, result.Usage.CompletionTokens, result.Usage.TotalTokens))
		b.WriteString(fmt.Sprintf("- Latency: `%s`\n", result.Latency.Round(time.Millisecond)))
	}
	b.WriteString("\n## Changed Files\n")
	if len(result.ChangedFiles) == 0 {
		b.WriteString("- (none)\n")
	} else {
		for _, file := range result.ChangedFiles {
			b.WriteString("- `" + file + "`\n")
		}
	}

	b.WriteString("\n## Review\n")
	b.WriteString(result.ReviewText)
	b.WriteString("\n\n## RAG Sources\n")
	if len(result.Retrieved) == 0 {
		b.WriteString("- (none)\n")
	} else {
		for _, item := range result.Retrieved {
			b.WriteString(fmt.Sprintf("- `%s` (%s, score=%.4f)\n", item.Chunk.Source, item.Chunk.ChunkID, item.Score))
		}
	}

	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay32Usage() {
	fmt.Println("Usage: openrouter-cli day32 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -workspace string         Project workspace root")
	fmt.Println("  -from-github-event        Read PR metadata/base/head from GitHub event payload")
	fmt.Println("  -event-path string        Path to GitHub event payload JSON")
	fmt.Println("  -base string              Base ref/SHA")
	fmt.Println("  -head string              Head ref/SHA")
	fmt.Println("  -diff-file string         Optional prebuilt diff file")
	fmt.Println("  -files-file string        Optional changed-files list")
	fmt.Println("  -readme string            README path for RAG")
	fmt.Println("  -docs string              Docs dir for RAG")
	fmt.Println("  -top-k int                Top-K RAG chunks")
	fmt.Println("  -max-diff-chars int       Max diff chars in prompt")
	fmt.Println("  -max-files int            Max changed files for code context")
	fmt.Println("  -max-file-chars int       Max chars per changed file")
	fmt.Println("  -model string             OpenRouter model")
	fmt.Println("  -max-tokens int           Max response tokens")
	fmt.Println("  -temperature float        Model temperature")
	fmt.Println("  -simulate                 Use local heuristic reviewer (no API)")
	fmt.Println("  -report string            Markdown report path")
	fmt.Println("  -help                     Show help")
}
