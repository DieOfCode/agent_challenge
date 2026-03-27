package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Agent is a separate entity that owns request/response interaction with the LLM.
type Agent interface {
	Reply(userInput string) (openRouterResult, error)
}

type LLMAgentConfig struct {
	APIKey       string
	Model        string
	MaxTokens    int
	Temperature  *float64
	SystemPrompt string
	Title        string
	HistoryStore HistoryStore
	ContextLimit int
	Compression  CompressionConfig
}

type CompressionConfig struct {
	Enabled       bool
	KeepLastN     int
	SummaryEveryN int
}

type LLMAgent struct {
	apiKey                   string
	model                    string
	maxTokens                int
	temperature              *float64
	title                    string
	systemPrompt             string
	history                  []message
	summaries                []string
	store                    HistoryStore
	contextLimit             int
	cumulativePromptTokens   int
	cumulativeResponseTokens int
	compression              CompressionConfig
}

type HistoryStore interface {
	Load() (conversationMemory, error)
	Save(conversationMemory) error
	Reset() error
}

type JSONHistoryStore struct {
	path string
}

type conversationMemory struct {
	Messages  []message
	Summaries []string
}

type conversationSnapshot struct {
	Version   int             `json:"version"`
	Messages  []storedMessage `json:"messages"`
	Summaries []storedSummary `json:"summaries,omitempty"`
}

type storedMessage struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	Timestamp string `json:"timestamp,omitempty"`
}

type storedSummary struct {
	Content   string `json:"content"`
	Timestamp string `json:"timestamp,omitempty"`
}

func NewJSONHistoryStore(path string) *JSONHistoryStore {
	return &JSONHistoryStore{path: path}
}

func (s *JSONHistoryStore) Load() (conversationMemory, error) {
	raw, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return conversationMemory{}, nil
		}
		return conversationMemory{}, err
	}

	var snapshot conversationSnapshot
	if err := json.Unmarshal(raw, &snapshot); err != nil {
		return conversationMemory{}, fmt.Errorf("failed to parse history JSON: %w", err)
	}

	memory := conversationMemory{
		Messages:  make([]message, 0, len(snapshot.Messages)),
		Summaries: make([]string, 0, len(snapshot.Summaries)),
	}
	for _, m := range snapshot.Messages {
		role := strings.TrimSpace(m.Role)
		content := strings.TrimSpace(m.Content)
		if role == "" || content == "" {
			continue
		}
		memory.Messages = append(memory.Messages, message{
			Role:    role,
			Content: content,
		})
	}
	for _, s := range snapshot.Summaries {
		content := strings.TrimSpace(s.Content)
		if content == "" {
			continue
		}
		memory.Summaries = append(memory.Summaries, content)
	}
	return memory, nil
}

func (s *JSONHistoryStore) Save(memory conversationMemory) error {
	snapshot := conversationSnapshot{
		Version:   2,
		Messages:  make([]storedMessage, 0, len(memory.Messages)),
		Summaries: make([]storedSummary, 0, len(memory.Summaries)),
	}
	now := time.Now().UTC().Format(time.RFC3339)
	for _, m := range memory.Messages {
		role := strings.TrimSpace(m.Role)
		content := strings.TrimSpace(m.Content)
		if role == "" || content == "" {
			continue
		}
		snapshot.Messages = append(snapshot.Messages, storedMessage{
			Role:      role,
			Content:   content,
			Timestamp: now,
		})
	}
	for _, summary := range memory.Summaries {
		content := strings.TrimSpace(summary)
		if content == "" {
			continue
		}
		snapshot.Summaries = append(snapshot.Summaries, storedSummary{
			Content:   content,
			Timestamp: now,
		})
	}

	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode history JSON: %w", err)
	}

	dir := filepath.Dir(s.path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("failed to create history dir: %w", err)
		}
	}

	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return fmt.Errorf("failed to write history temp file: %w", err)
	}
	if err := os.Rename(tmp, s.path); err != nil {
		return fmt.Errorf("failed to replace history file: %w", err)
	}
	return nil
}

func (s *JSONHistoryStore) Reset() error {
	err := os.Remove(s.path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func NewLLMAgent(cfg LLMAgentConfig) (*LLMAgent, error) {
	history := make([]message, 0, 16)
	summaries := make([]string, 0, 8)
	systemPrompt := strings.TrimSpace(cfg.SystemPrompt)

	if cfg.HistoryStore != nil {
		memory, err := cfg.HistoryStore.Load()
		if err != nil {
			return nil, fmt.Errorf("failed to load history: %w", err)
		}
		if len(memory.Summaries) > 0 {
			summaries = append(summaries, memory.Summaries...)
		}
		for _, loaded := range memory.Messages {
			role := strings.ToLower(strings.TrimSpace(loaded.Role))
			content := strings.TrimSpace(loaded.Content)
			if content == "" {
				continue
			}
			if role == "system" {
				if systemPrompt == "" {
					systemPrompt = content
				}
				continue
			}
			history = append(history, message{
				Role:    loaded.Role,
				Content: content,
			})
		}
	}

	compression := cfg.Compression
	if compression.KeepLastN <= 0 {
		compression.KeepLastN = 12
	}
	if compression.SummaryEveryN <= 0 {
		compression.SummaryEveryN = 10
	}

	return &LLMAgent{
		apiKey:       cfg.APIKey,
		model:        cfg.Model,
		maxTokens:    cfg.MaxTokens,
		temperature:  cfg.Temperature,
		title:        cfg.Title,
		systemPrompt: systemPrompt,
		history:      history,
		summaries:    summaries,
		store:        cfg.HistoryStore,
		contextLimit: cfg.ContextLimit,
		compression:  compression,
	}, nil
}

type ContextLimitError struct {
	EstimatedPromptTokens int
	MaxTokens             int
	ContextLimit          int
}

func (e *ContextLimitError) Error() string {
	return fmt.Sprintf(
		"context limit exceeded: estimated_prompt_tokens=%d max_tokens=%d context_limit=%d",
		e.EstimatedPromptTokens,
		e.MaxTokens,
		e.ContextLimit,
	)
}

func (a *LLMAgent) Reply(userInput string) (openRouterResult, error) {
	userInput = strings.TrimSpace(userInput)
	if userInput == "" {
		return openRouterResult{}, fmt.Errorf("empty user input")
	}

	a.compressHistoryIfNeeded()

	contextMessages := a.buildContextMessages()
	historyEstimate := estimateMessagesTokens(contextMessages)
	userEstimate := estimateMessageTokens(message{Role: "user", Content: userInput})
	requestEstimate := historyEstimate + userEstimate
	if a.contextLimit > 0 && requestEstimate+a.maxTokens > a.contextLimit {
		return openRouterResult{}, &ContextLimitError{
			EstimatedPromptTokens: requestEstimate,
			MaxTokens:             a.maxTokens,
			ContextLimit:          a.contextLimit,
		}
	}

	requestMessages := append([]message(nil), contextMessages...)
	requestMessages = append(requestMessages, message{
		Role:    "user",
		Content: userInput,
	})

	result, err := callOpenRouterDetailed(
		a.apiKey,
		a.model,
		requestMessages,
		a.maxTokens,
		a.temperature,
		nil,
		a.title,
	)
	if err != nil {
		return openRouterResult{}, err
	}

	// Store conversation state inside the agent to keep it as a true chat entity.
	a.history = append(a.history,
		message{Role: "user", Content: userInput},
		message{Role: "assistant", Content: strings.TrimSpace(result.Answer)},
	)
	a.compressHistoryIfNeeded()

	if a.store != nil {
		if err := a.store.Save(conversationMemory{
			Messages:  a.history,
			Summaries: a.summaries,
		}); err != nil {
			return openRouterResult{}, fmt.Errorf("failed to save conversation history: %w", err)
		}
	}

	promptTokens := result.Usage.PromptTokens
	responseTokens := result.Usage.CompletionTokens
	totalTokens := result.Usage.TotalTokens
	if promptTokens == 0 {
		promptTokens = requestEstimate
	}
	if responseTokens == 0 {
		responseTokens = estimateTextTokens(result.Answer)
	}
	if totalTokens == 0 {
		totalTokens = promptTokens + responseTokens
	}

	a.cumulativePromptTokens += promptTokens
	a.cumulativeResponseTokens += responseTokens

	result.Tokens = tokenStats{
		EstimatedHistoryTokens:   historyEstimate,
		EstimatedRequestTokens:   requestEstimate,
		EstimatedResponseTokens:  estimateTextTokens(result.Answer),
		PromptTokens:             promptTokens,
		ResponseTokens:           responseTokens,
		TotalTokens:              totalTokens,
		ConversationTokens:       estimateMessagesTokens(a.history),
		CumulativePromptTokens:   a.cumulativePromptTokens,
		CumulativeResponseTokens: a.cumulativeResponseTokens,
		CumulativeTotalTokens:    a.cumulativePromptTokens + a.cumulativeResponseTokens,
		ContextLimit:             a.contextLimit,
	}

	return result, nil
}

func (a *LLMAgent) buildContextMessages() []message {
	context := make([]message, 0, 2+len(a.history))
	if a.systemPrompt != "" {
		context = append(context, message{
			Role:    "system",
			Content: a.systemPrompt,
		})
	}
	if summary := combineSummaries(a.summaries); summary != "" {
		context = append(context, message{
			Role:    "system",
			Content: "Conversation summary (compressed older turns):\n" + summary,
		})
	}
	context = append(context, a.history...)
	return context
}

func (a *LLMAgent) compressHistoryIfNeeded() {
	if !a.compression.Enabled {
		return
	}
	keepLast := a.compression.KeepLastN
	summaryEvery := a.compression.SummaryEveryN
	if keepLast < 0 {
		keepLast = 0
	}
	if summaryEvery <= 0 {
		return
	}

	for len(a.history)-keepLast >= summaryEvery {
		chunk := make([]message, 0, summaryEvery)
		for _, m := range a.history[:summaryEvery] {
			if strings.EqualFold(strings.TrimSpace(m.Role), "system") {
				continue
			}
			chunk = append(chunk, m)
		}
		if summary := summarizeHistoryChunk(chunk); summary != "" {
			a.summaries = append(a.summaries, summary)
		}
		a.history = append([]message(nil), a.history[summaryEvery:]...)
	}
}

func summarizeHistoryChunk(chunk []message) string {
	if len(chunk) == 0 {
		return ""
	}
	parts := make([]string, 0, len(chunk))
	for _, m := range chunk {
		role := strings.ToLower(strings.TrimSpace(m.Role))
		if role == "" {
			role = "message"
		}
		content := limitWords(normalizeWhitespace(m.Content), 16)
		if content == "" {
			continue
		}
		parts = append(parts, role+": "+content)
	}
	if len(parts) == 0 {
		return ""
	}
	combined := strings.Join(parts, " | ")
	r := []rune(combined)
	if len(r) > 420 {
		return string(r[:420]) + "..."
	}
	return combined
}

func combineSummaries(summaries []string) string {
	if len(summaries) == 0 {
		return ""
	}
	lines := make([]string, 0, len(summaries))
	for i, summary := range summaries {
		clean := strings.TrimSpace(summary)
		if clean == "" {
			continue
		}
		lines = append(lines, fmt.Sprintf("%d) %s", i+1, clean))
	}
	return strings.Join(lines, "\n")
}

func normalizeWhitespace(text string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(text)), " ")
}

func limitWords(text string, maxWords int) string {
	text = strings.TrimSpace(text)
	if text == "" || maxWords <= 0 {
		return text
	}
	words := strings.Fields(text)
	if len(words) <= maxWords {
		return strings.Join(words, " ")
	}
	return strings.Join(words[:maxWords], " ") + "..."
}

func runAgentCommand(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli agent", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	model := fs.String("model", getDefaultModel(), "OpenRouter model")
	prompt := fs.String("prompt", "", "Single user prompt (if empty, reads stdin unless -interactive)")
	systemPrompt := fs.String("system", "You are a helpful assistant.", "System prompt for the agent")
	maxTokens := fs.Int("max-tokens", 400, "Maximum response tokens")
	temperature := fs.Float64("temperature", 0.2, "Agent temperature")
	contextLimit := fs.Int("context-limit", 0, "Optional context limit in tokens (0 disables local pre-check)")
	showTokens := fs.Bool("show-tokens", false, "Print token stats after each response")
	compressHistory := fs.Bool("compress-history", false, "Enable context compression (summary + last N messages)")
	keepLast := fs.Int("keep-last", 12, "How many latest messages to keep without compression")
	summaryEvery := fs.Int("summary-every", 10, "Compress each N older messages into one summary block")
	historyFile := fs.String("history-file", ".agent_history.json", "Path to JSON file for saved dialogue context")
	noHistory := fs.Bool("no-history", false, "Disable context persistence")
	resetHistory := fs.Bool("reset-history", false, "Delete saved history before start")
	interactive := fs.Bool("interactive", false, "Interactive chat mode")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse agent flags: %w", err)
	}
	if *help {
		printAgentUsage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected agent arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *noHistory && *resetHistory {
		return fmt.Errorf("cannot use -no-history and -reset-history together")
	}

	var historyStore HistoryStore
	if !*noHistory {
		store := NewJSONHistoryStore(strings.TrimSpace(*historyFile))
		if *resetHistory {
			if err := store.Reset(); err != nil {
				return fmt.Errorf("failed to reset history: %w", err)
			}
		}
		historyStore = store
	}

	t := *temperature
	agent, err := NewLLMAgent(LLMAgentConfig{
		APIKey:       getAPIKey(),
		Model:        *model,
		MaxTokens:    *maxTokens,
		Temperature:  &t,
		SystemPrompt: *systemPrompt,
		Title:        "encapsulated-agent-cli",
		HistoryStore: historyStore,
		ContextLimit: *contextLimit,
		Compression: CompressionConfig{
			Enabled:       *compressHistory,
			KeepLastN:     *keepLast,
			SummaryEveryN: *summaryEvery,
		},
	})
	if err != nil {
		return err
	}

	if *interactive {
		return runAgentInteractive(agent, *showTokens)
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
		return fmt.Errorf("empty prompt: use -prompt, -interactive, or pipe text to stdin")
	}

	result, err := agent.Reply(userPrompt)
	if err != nil {
		return err
	}
	fmt.Println(strings.TrimSpace(result.Answer))
	if *showTokens {
		printTokenStats(result.Tokens)
	}
	return nil
}

func runAgentInteractive(agent Agent, showTokens bool) error {
	fmt.Println("Agent interactive mode. Type /exit to quit.")
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("you> ")
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				return fmt.Errorf("stdin scan failed: %w", err)
			}
			return nil
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch strings.ToLower(line) {
		case "/exit", "exit", "quit":
			return nil
		}

		result, err := agent.Reply(line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "agent error: %v\n", err)
			continue
		}

		fmt.Printf("agent> %s\n\n", strings.TrimSpace(result.Answer))
		if showTokens {
			printTokenStats(result.Tokens)
		}
	}
}

func printTokenStats(stats tokenStats) {
	fmt.Printf(
		"tokens> history_est=%d request_est=%d prompt=%d response=%d total=%d conversation_est=%d cumulative_total=%d context_limit=%d\n",
		stats.EstimatedHistoryTokens,
		stats.EstimatedRequestTokens,
		stats.PromptTokens,
		stats.ResponseTokens,
		stats.TotalTokens,
		stats.ConversationTokens,
		stats.CumulativeTotalTokens,
		stats.ContextLimit,
	)
}

func printAgentUsage() {
	fmt.Println("Usage: openrouter-cli agent [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -model string        OpenRouter model")
	fmt.Println("  -prompt string       Single prompt (if empty, reads stdin)")
	fmt.Println("  -system string       System prompt for the agent")
	fmt.Println("  -temperature float   Temperature")
	fmt.Println("  -max-tokens int      Maximum response tokens")
	fmt.Println("  -context-limit int   Optional context limit in tokens")
	fmt.Println("  -show-tokens         Print token stats for each response")
	fmt.Println("  -compress-history    Enable history compression")
	fmt.Println("  -keep-last int       Keep latest N messages as-is")
	fmt.Println("  -summary-every int   Compress every N older messages")
	fmt.Println("  -history-file string JSON file for saved context")
	fmt.Println("  -no-history          Disable context persistence")
	fmt.Println("  -reset-history       Clear saved context before start")
	fmt.Println("  -interactive         Interactive chat mode")
	fmt.Println("  -help                Show help")
}

func estimateTextTokens(text string) int {
	runes := len([]rune(strings.TrimSpace(text)))
	if runes == 0 {
		return 0
	}
	// Rough heuristic: ~4 chars per token + small message overhead.
	return (runes+3)/4 + 1
}

func estimateMessageTokens(m message) int {
	return 4 + estimateTextTokens(m.Content)
}

func estimateMessagesTokens(messages []message) int {
	total := 2
	for _, msg := range messages {
		total += estimateMessageTokens(msg)
	}
	return total
}
