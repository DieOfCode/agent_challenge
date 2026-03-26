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
}

type LLMAgent struct {
	apiKey                   string
	model                    string
	maxTokens                int
	temperature              *float64
	title                    string
	history                  []message
	store                    HistoryStore
	contextLimit             int
	cumulativePromptTokens   int
	cumulativeResponseTokens int
}

type HistoryStore interface {
	Load() ([]message, error)
	Save([]message) error
	Reset() error
}

type JSONHistoryStore struct {
	path string
}

type conversationSnapshot struct {
	Version  int             `json:"version"`
	Messages []storedMessage `json:"messages"`
}

type storedMessage struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	Timestamp string `json:"timestamp,omitempty"`
}

func NewJSONHistoryStore(path string) *JSONHistoryStore {
	return &JSONHistoryStore{path: path}
}

func (s *JSONHistoryStore) Load() ([]message, error) {
	raw, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var snapshot conversationSnapshot
	if err := json.Unmarshal(raw, &snapshot); err != nil {
		return nil, fmt.Errorf("failed to parse history JSON: %w", err)
	}

	history := make([]message, 0, len(snapshot.Messages))
	for _, m := range snapshot.Messages {
		role := strings.TrimSpace(m.Role)
		content := strings.TrimSpace(m.Content)
		if role == "" || content == "" {
			continue
		}
		history = append(history, message{
			Role:    role,
			Content: content,
		})
	}
	return history, nil
}

func (s *JSONHistoryStore) Save(history []message) error {
	snapshot := conversationSnapshot{
		Version:  1,
		Messages: make([]storedMessage, 0, len(history)),
	}
	now := time.Now().UTC().Format(time.RFC3339)
	for _, m := range history {
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

	if cfg.HistoryStore != nil {
		loadedHistory, err := cfg.HistoryStore.Load()
		if err != nil {
			return nil, fmt.Errorf("failed to load history: %w", err)
		}
		if len(loadedHistory) > 0 {
			history = append(history, loadedHistory...)
		}
	}

	if len(history) == 0 && strings.TrimSpace(cfg.SystemPrompt) != "" {
		history = append(history, message{
			Role:    "system",
			Content: strings.TrimSpace(cfg.SystemPrompt),
		})
	}

	return &LLMAgent{
		apiKey:       cfg.APIKey,
		model:        cfg.Model,
		maxTokens:    cfg.MaxTokens,
		temperature:  cfg.Temperature,
		title:        cfg.Title,
		history:      history,
		store:        cfg.HistoryStore,
		contextLimit: cfg.ContextLimit,
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

	historyEstimate := estimateMessagesTokens(a.history)
	userEstimate := estimateMessageTokens(message{Role: "user", Content: userInput})
	requestEstimate := historyEstimate + userEstimate
	if a.contextLimit > 0 && requestEstimate+a.maxTokens > a.contextLimit {
		return openRouterResult{}, &ContextLimitError{
			EstimatedPromptTokens: requestEstimate,
			MaxTokens:             a.maxTokens,
			ContextLimit:          a.contextLimit,
		}
	}

	requestMessages := append([]message(nil), a.history...)
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

	if a.store != nil {
		if err := a.store.Save(a.history); err != nil {
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
