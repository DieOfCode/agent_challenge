package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Agent is a separate entity that owns request/response interaction with the LLM.
type Agent interface {
	Reply(userInput string) (openRouterResult, error)
}

type ContextStrategy string

const (
	ContextStrategyFull      ContextStrategy = "full"
	ContextStrategySliding   ContextStrategy = "sliding"
	ContextStrategyFacts     ContextStrategy = "facts"
	ContextStrategyBranching ContextStrategy = "branching"
)

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
	Strategy     ContextStrategy
	WindowSize   int
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
	facts                    map[string]string
	strategy                 ContextStrategy
	windowSize               int
	branches                 map[string][]message
	checkpoints              map[string]branchCheckpoint
	activeBranch             string
	store                    HistoryStore
	contextLimit             int
	cumulativePromptTokens   int
	cumulativeResponseTokens int
	compression              CompressionConfig
}

type branchCheckpoint struct {
	Branch string `json:"branch"`
	Index  int    `json:"index"`
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
	Messages     []message
	Summaries    []string
	Facts        map[string]string
	Branches     map[string][]message
	Checkpoints  map[string]branchCheckpoint
	ActiveBranch string
	Strategy     string
}

type conversationSnapshot struct {
	Version      int                         `json:"version"`
	Messages     []storedMessage             `json:"messages"`
	Summaries    []storedSummary             `json:"summaries,omitempty"`
	Facts        map[string]string           `json:"facts,omitempty"`
	Branches     map[string][]storedMessage  `json:"branches,omitempty"`
	Checkpoints  map[string]branchCheckpoint `json:"checkpoints,omitempty"`
	ActiveBranch string                      `json:"active_branch,omitempty"`
	Strategy     string                      `json:"strategy,omitempty"`
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
		Messages:     make([]message, 0, len(snapshot.Messages)),
		Summaries:    make([]string, 0, len(snapshot.Summaries)),
		Facts:        cloneFacts(snapshot.Facts),
		Branches:     make(map[string][]message, len(snapshot.Branches)),
		Checkpoints:  cloneCheckpoints(snapshot.Checkpoints),
		ActiveBranch: strings.TrimSpace(snapshot.ActiveBranch),
		Strategy:     strings.TrimSpace(snapshot.Strategy),
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
	for branchName, storedMessages := range snapshot.Branches {
		name := strings.TrimSpace(branchName)
		if name == "" {
			continue
		}
		converted := make([]message, 0, len(storedMessages))
		for _, m := range storedMessages {
			role := strings.TrimSpace(m.Role)
			content := strings.TrimSpace(m.Content)
			if role == "" || content == "" {
				continue
			}
			converted = append(converted, message{
				Role:    role,
				Content: content,
			})
		}
		memory.Branches[name] = converted
	}
	return memory, nil
}

func (s *JSONHistoryStore) Save(memory conversationMemory) error {
	snapshot := conversationSnapshot{
		Version:      3,
		Messages:     make([]storedMessage, 0, len(memory.Messages)),
		Summaries:    make([]storedSummary, 0, len(memory.Summaries)),
		Facts:        cloneFacts(memory.Facts),
		Branches:     make(map[string][]storedMessage, len(memory.Branches)),
		Checkpoints:  cloneCheckpoints(memory.Checkpoints),
		ActiveBranch: strings.TrimSpace(memory.ActiveBranch),
		Strategy:     strings.TrimSpace(memory.Strategy),
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
	for branchName, msgs := range memory.Branches {
		name := strings.TrimSpace(branchName)
		if name == "" {
			continue
		}
		stored := make([]storedMessage, 0, len(msgs))
		for _, m := range msgs {
			role := strings.TrimSpace(m.Role)
			content := strings.TrimSpace(m.Content)
			if role == "" || content == "" {
				continue
			}
			stored = append(stored, storedMessage{
				Role:      role,
				Content:   content,
				Timestamp: now,
			})
		}
		snapshot.Branches[name] = stored
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
	facts := make(map[string]string)
	branches := make(map[string][]message)
	checkpoints := make(map[string]branchCheckpoint)
	activeBranch := "main"
	systemPrompt := strings.TrimSpace(cfg.SystemPrompt)

	if cfg.HistoryStore != nil {
		memory, err := cfg.HistoryStore.Load()
		if err != nil {
			return nil, fmt.Errorf("failed to load history: %w", err)
		}
		if len(memory.Summaries) > 0 {
			summaries = append(summaries, memory.Summaries...)
		}
		if len(memory.Facts) > 0 {
			facts = cloneFacts(memory.Facts)
		}
		if len(memory.Branches) > 0 {
			branches = cloneBranches(memory.Branches)
		}
		if len(memory.Checkpoints) > 0 {
			checkpoints = cloneCheckpoints(memory.Checkpoints)
		}
		if strings.TrimSpace(memory.ActiveBranch) != "" {
			activeBranch = strings.TrimSpace(memory.ActiveBranch)
		}
		if cfg.Strategy == "" && strings.TrimSpace(memory.Strategy) != "" {
			cfg.Strategy = ContextStrategy(strings.TrimSpace(memory.Strategy))
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
	strategy := normalizeStrategy(cfg.Strategy)
	windowSize := cfg.WindowSize
	if windowSize <= 0 {
		windowSize = 12
	}
	if len(branches) == 0 {
		branches["main"] = cloneMessages(history)
	}
	if _, ok := branches[activeBranch]; !ok {
		activeBranch = "main"
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
		facts:        facts,
		strategy:     strategy,
		windowSize:   windowSize,
		branches:     branches,
		checkpoints:  checkpoints,
		activeBranch: activeBranch,
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

	history := a.getActiveHistory()
	if a.strategy == ContextStrategyFull {
		history = a.compressHistoryIfNeeded(history)
		a.setActiveHistory(history)
	}
	if a.strategy == ContextStrategyFacts {
		a.updateFactsFromUserInput(userInput)
	}

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
	history = append(history,
		message{Role: "user", Content: userInput},
		message{Role: "assistant", Content: strings.TrimSpace(result.Answer)},
	)
	if a.strategy == ContextStrategySliding {
		history = keepLastMessages(history, a.windowSize)
	}
	if a.strategy == ContextStrategyFull {
		history = a.compressHistoryIfNeeded(history)
	}
	a.setActiveHistory(history)

	if err := a.saveState(); err != nil {
		return openRouterResult{}, fmt.Errorf("failed to save conversation history: %w", err)
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
		ConversationTokens:       estimateMessagesTokens(a.getActiveHistory()),
		CumulativePromptTokens:   a.cumulativePromptTokens,
		CumulativeResponseTokens: a.cumulativeResponseTokens,
		CumulativeTotalTokens:    a.cumulativePromptTokens + a.cumulativeResponseTokens,
		ContextLimit:             a.contextLimit,
	}

	return result, nil
}

func (a *LLMAgent) buildContextMessages() []message {
	history := a.getActiveHistory()
	context := make([]message, 0, 3+len(history))
	if a.systemPrompt != "" {
		context = append(context, message{
			Role:    "system",
			Content: a.systemPrompt,
		})
	}

	switch a.strategy {
	case ContextStrategySliding:
		context = append(context, keepLastMessages(history, a.windowSize)...)
	case ContextStrategyFacts:
		if factsBlock := buildFactsBlock(a.facts); factsBlock != "" {
			context = append(context, message{
				Role:    "system",
				Content: factsBlock,
			})
		}
		context = append(context, keepLastMessages(history, a.windowSize)...)
	case ContextStrategyBranching:
		context = append(context, history...)
	default:
		if summary := combineSummaries(a.summaries); summary != "" {
			context = append(context, message{
				Role:    "system",
				Content: "Conversation summary (compressed older turns):\n" + summary,
			})
		}
		context = append(context, history...)
	}
	return context
}

func (a *LLMAgent) compressHistoryIfNeeded(history []message) []message {
	if !a.compression.Enabled {
		return history
	}
	keepLast := a.compression.KeepLastN
	summaryEvery := a.compression.SummaryEveryN
	if keepLast < 0 {
		keepLast = 0
	}
	if summaryEvery <= 0 {
		return history
	}

	for len(history)-keepLast >= summaryEvery {
		chunk := make([]message, 0, summaryEvery)
		for _, m := range history[:summaryEvery] {
			if strings.EqualFold(strings.TrimSpace(m.Role), "system") {
				continue
			}
			chunk = append(chunk, m)
		}
		if summary := summarizeHistoryChunk(chunk); summary != "" {
			a.summaries = append(a.summaries, summary)
		}
		history = append([]message(nil), history[summaryEvery:]...)
	}
	return history
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

func (a *LLMAgent) getActiveHistory() []message {
	if a.strategy == ContextStrategyBranching {
		branch := strings.TrimSpace(a.activeBranch)
		if branch == "" {
			branch = "main"
			a.activeBranch = branch
		}
		history, ok := a.branches[branch]
		if !ok {
			history = make([]message, 0, 16)
			a.branches[branch] = history
		}
		return history
	}
	return a.history
}

func (a *LLMAgent) setActiveHistory(history []message) {
	copied := cloneMessages(history)
	if a.strategy == ContextStrategyBranching {
		branch := strings.TrimSpace(a.activeBranch)
		if branch == "" {
			branch = "main"
			a.activeBranch = branch
		}
		a.branches[branch] = copied
		return
	}
	a.history = copied
}

func (a *LLMAgent) saveState() error {
	if a.store == nil {
		return nil
	}

	active := a.getActiveHistory()
	memory := conversationMemory{
		Messages:     cloneMessages(active),
		Summaries:    append([]string(nil), a.summaries...),
		Facts:        cloneFacts(a.facts),
		Branches:     cloneBranches(a.branches),
		Checkpoints:  cloneCheckpoints(a.checkpoints),
		ActiveBranch: strings.TrimSpace(a.activeBranch),
		Strategy:     string(a.strategy),
	}
	return a.store.Save(memory)
}

func keepLastMessages(history []message, keep int) []message {
	if keep <= 0 {
		return nil
	}
	if len(history) <= keep {
		return cloneMessages(history)
	}
	return cloneMessages(history[len(history)-keep:])
}

func cloneMessages(in []message) []message {
	if len(in) == 0 {
		return nil
	}
	out := make([]message, 0, len(in))
	for _, m := range in {
		role := strings.TrimSpace(m.Role)
		content := strings.TrimSpace(m.Content)
		if role == "" || content == "" {
			continue
		}
		out = append(out, message{
			Role:    role,
			Content: content,
		})
	}
	return out
}

func cloneBranches(in map[string][]message) map[string][]message {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string][]message, len(in))
	for name, history := range in {
		cleanName := strings.TrimSpace(name)
		if cleanName == "" {
			continue
		}
		out[cleanName] = cloneMessages(history)
	}
	return out
}

func cloneFacts(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		key := strings.TrimSpace(k)
		val := strings.TrimSpace(v)
		if key == "" || val == "" {
			continue
		}
		out[key] = val
	}
	return out
}

func cloneCheckpoints(in map[string]branchCheckpoint) map[string]branchCheckpoint {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]branchCheckpoint, len(in))
	for name, cp := range in {
		key := strings.TrimSpace(name)
		if key == "" {
			continue
		}
		if cp.Index < 0 {
			cp.Index = 0
		}
		cp.Branch = strings.TrimSpace(cp.Branch)
		if cp.Branch == "" {
			cp.Branch = "main"
		}
		out[key] = cp
	}
	return out
}

func normalizeStrategy(raw ContextStrategy) ContextStrategy {
	switch ContextStrategy(strings.ToLower(strings.TrimSpace(string(raw)))) {
	case ContextStrategySliding:
		return ContextStrategySliding
	case ContextStrategyFacts:
		return ContextStrategyFacts
	case ContextStrategyBranching:
		return ContextStrategyBranching
	default:
		return ContextStrategyFull
	}
}

func parseContextStrategy(raw string) (ContextStrategy, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "full":
		return ContextStrategyFull, nil
	case "sliding":
		return ContextStrategySliding, nil
	case "facts", "sticky-facts", "sticky_facts":
		return ContextStrategyFacts, nil
	case "branching", "branch":
		return ContextStrategyBranching, nil
	default:
		return "", fmt.Errorf("unknown strategy: %s (supported: full|sliding|facts|branching)", raw)
	}
}

func buildFactsBlock(facts map[string]string) string {
	if len(facts) == 0 {
		return ""
	}
	keys := make([]string, 0, len(facts))
	for key := range facts {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	var b strings.Builder
	b.WriteString("Sticky facts extracted from the dialogue:\n")
	for _, key := range keys {
		value := strings.TrimSpace(facts[key])
		if value == "" {
			continue
		}
		b.WriteString("- ")
		b.WriteString(key)
		b.WriteString(": ")
		b.WriteString(limitWords(normalizeWhitespace(value), 24))
		b.WriteString("\n")
	}
	return strings.TrimSpace(b.String())
}

func (a *LLMAgent) updateFactsFromUserInput(userInput string) {
	if a.facts == nil {
		a.facts = make(map[string]string)
	}
	text := normalizeWhitespace(userInput)
	if text == "" {
		return
	}

	for _, part := range splitFactSegments(text) {
		if strings.Contains(part, ":") {
			left, right, ok := strings.Cut(part, ":")
			if !ok {
				continue
			}
			key := normalizeFactKey(left)
			val := strings.TrimSpace(right)
			if key != "" && val != "" {
				a.facts[key] = val
			}
		}
	}

	lower := strings.ToLower(text)
	switch {
	case strings.Contains(lower, "цель"):
		a.facts["goal"] = text
	case strings.Contains(lower, "огранич"):
		a.facts["constraints"] = text
	case strings.Contains(lower, "предпочт"):
		a.facts["preferences"] = text
	case strings.Contains(lower, "решени"):
		a.facts["decisions"] = text
	case strings.Contains(lower, "договор"):
		a.facts["agreements"] = text
	case strings.Contains(lower, "deadline"), strings.Contains(lower, "дедлайн"):
		a.facts["deadline"] = text
	case strings.Contains(lower, "budget"), strings.Contains(lower, "бюджет"):
		a.facts["budget"] = text
	case strings.Contains(lower, "stack"), strings.Contains(lower, "стек"):
		a.facts["stack"] = text
	}

	if strings.Contains(lower, "запомни") || strings.Contains(lower, "remember") {
		next := len(a.facts) + 1
		a.facts[fmt.Sprintf("note_%02d", next)] = text
	}
}

func splitFactSegments(text string) []string {
	raw := strings.FieldsFunc(text, func(r rune) bool {
		return r == ';' || r == '\n'
	})
	out := make([]string, 0, len(raw))
	for _, segment := range raw {
		trimmed := strings.TrimSpace(segment)
		if trimmed == "" {
			continue
		}
		out = append(out, trimmed)
	}
	return out
}

func normalizeFactKey(key string) string {
	key = strings.ToLower(normalizeWhitespace(key))
	key = strings.ReplaceAll(key, " ", "_")
	key = strings.Trim(key, "_-")
	if key == "" {
		return ""
	}
	if len(key) > 32 {
		return key[:32]
	}
	return key
}

func (a *LLMAgent) SaveCheckpoint(name string) error {
	if a.strategy != ContextStrategyBranching {
		return fmt.Errorf("checkpoints are available only in branching strategy")
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("checkpoint name is required")
	}
	history := a.getActiveHistory()
	if a.checkpoints == nil {
		a.checkpoints = make(map[string]branchCheckpoint)
	}
	a.checkpoints[name] = branchCheckpoint{
		Branch: a.activeBranch,
		Index:  len(history),
	}
	return a.saveState()
}

func (a *LLMAgent) CreateBranch(name, checkpoint string) error {
	if a.strategy != ContextStrategyBranching {
		return fmt.Errorf("branches are available only in branching strategy")
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("branch name is required")
	}
	if _, exists := a.branches[name]; exists {
		return fmt.Errorf("branch already exists: %s", name)
	}

	sourceBranch := a.activeBranch
	index := len(a.getActiveHistory())
	checkpoint = strings.TrimSpace(checkpoint)
	if checkpoint != "" {
		cp, ok := a.checkpoints[checkpoint]
		if !ok {
			return fmt.Errorf("checkpoint not found: %s", checkpoint)
		}
		sourceBranch = cp.Branch
		index = cp.Index
	}

	sourceHistory, ok := a.branches[sourceBranch]
	if !ok {
		return fmt.Errorf("source branch not found: %s", sourceBranch)
	}
	if index < 0 {
		index = 0
	}
	if index > len(sourceHistory) {
		index = len(sourceHistory)
	}

	a.branches[name] = cloneMessages(sourceHistory[:index])
	return a.saveState()
}

func (a *LLMAgent) SwitchBranch(name string) error {
	if a.strategy != ContextStrategyBranching {
		return fmt.Errorf("branch switching is available only in branching strategy")
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("branch name is required")
	}
	if _, ok := a.branches[name]; !ok {
		return fmt.Errorf("branch not found: %s", name)
	}
	a.activeBranch = name
	return a.saveState()
}

func (a *LLMAgent) ListBranches() []string {
	if len(a.branches) == 0 {
		return nil
	}
	names := make([]string, 0, len(a.branches))
	for name := range a.branches {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (a *LLMAgent) ActiveBranch() string {
	if a.strategy != ContextStrategyBranching {
		return "main"
	}
	if strings.TrimSpace(a.activeBranch) == "" {
		return "main"
	}
	return a.activeBranch
}

func (a *LLMAgent) FactsSnapshot() map[string]string {
	return cloneFacts(a.facts)
}

func (a *LLMAgent) Strategy() ContextStrategy {
	return a.strategy
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
	strategyRaw := fs.String("strategy", "full", "Context strategy: full|sliding|facts|branching")
	windowSize := fs.Int("window-size", 12, "Window size for sliding/facts strategies")
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
	strategy, err := parseContextStrategy(*strategyRaw)
	if err != nil {
		return err
	}
	if strategy != ContextStrategyFull && *compressHistory {
		return fmt.Errorf("-compress-history works only with -strategy=full")
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
		Strategy:     strategy,
		WindowSize:   *windowSize,
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
	fmt.Println("Extra commands: /strategy, /facts, /branches, /checkpoint <name>, /branch <name> [checkpoint], /switch <name>")
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

		if strings.HasPrefix(line, "/") {
			if handleInteractiveCommand(agent, line) {
				continue
			}
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

func handleInteractiveCommand(agent Agent, input string) bool {
	fields := strings.Fields(strings.TrimSpace(input))
	if len(fields) == 0 {
		return false
	}

	cmd := strings.ToLower(fields[0])
	switch cmd {
	case "/exit", "exit", "quit":
		return false
	case "/strategy":
		llm, ok := agent.(*LLMAgent)
		if !ok {
			fmt.Println("agent> strategy info is unavailable")
			return true
		}
		fmt.Printf("agent> strategy=%s active_branch=%s\n\n", llm.Strategy(), llm.ActiveBranch())
		return true
	case "/facts":
		llm, ok := agent.(*LLMAgent)
		if !ok {
			fmt.Println("agent> facts are unavailable")
			return true
		}
		facts := llm.FactsSnapshot()
		if len(facts) == 0 {
			fmt.Println("agent> no sticky facts yet")
			fmt.Println()
			return true
		}
		keys := make([]string, 0, len(facts))
		for key := range facts {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		fmt.Println("agent> sticky facts:")
		for _, key := range keys {
			fmt.Printf("- %s: %s\n", key, facts[key])
		}
		fmt.Println()
		return true
	case "/branches":
		llm, ok := agent.(*LLMAgent)
		if !ok {
			fmt.Println("agent> branches are unavailable")
			return true
		}
		names := llm.ListBranches()
		if len(names) == 0 {
			fmt.Println("agent> no branches")
			fmt.Println()
			return true
		}
		active := llm.ActiveBranch()
		fmt.Println("agent> branches:")
		for _, name := range names {
			if name == active {
				fmt.Printf("- %s (active)\n", name)
			} else {
				fmt.Printf("- %s\n", name)
			}
		}
		fmt.Println()
		return true
	case "/checkpoint":
		if len(fields) < 2 {
			fmt.Println("agent> usage: /checkpoint <name>")
			fmt.Println()
			return true
		}
		llm, ok := agent.(*LLMAgent)
		if !ok {
			fmt.Println("agent> checkpoints are unavailable")
			fmt.Println()
			return true
		}
		if err := llm.SaveCheckpoint(fields[1]); err != nil {
			fmt.Printf("agent> %v\n\n", err)
			return true
		}
		fmt.Printf("agent> checkpoint saved: %s\n\n", fields[1])
		return true
	case "/branch":
		if len(fields) < 2 {
			fmt.Println("agent> usage: /branch <name> [checkpoint]")
			fmt.Println()
			return true
		}
		llm, ok := agent.(*LLMAgent)
		if !ok {
			fmt.Println("agent> branching is unavailable")
			fmt.Println()
			return true
		}
		checkpoint := ""
		if len(fields) > 2 {
			checkpoint = fields[2]
		}
		if err := llm.CreateBranch(fields[1], checkpoint); err != nil {
			fmt.Printf("agent> %v\n\n", err)
			return true
		}
		fmt.Printf("agent> branch created: %s\n\n", fields[1])
		return true
	case "/switch":
		if len(fields) < 2 {
			fmt.Println("agent> usage: /switch <name>")
			fmt.Println()
			return true
		}
		llm, ok := agent.(*LLMAgent)
		if !ok {
			fmt.Println("agent> branch switching is unavailable")
			fmt.Println()
			return true
		}
		if err := llm.SwitchBranch(fields[1]); err != nil {
			fmt.Printf("agent> %v\n\n", err)
			return true
		}
		fmt.Printf("agent> active branch: %s\n\n", fields[1])
		return true
	default:
		return false
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
	fmt.Println("  -strategy string     full|sliding|facts|branching")
	fmt.Println("  -window-size int     Context window for sliding/facts strategies")
	fmt.Println("  -show-tokens         Print token stats for each response")
	fmt.Println("  -compress-history    Enable history compression")
	fmt.Println("  -keep-last int       Keep latest N messages as-is")
	fmt.Println("  -summary-every int   Compress every N older messages")
	fmt.Println("  -history-file string JSON file for saved context")
	fmt.Println("  -no-history          Disable context persistence")
	fmt.Println("  -reset-history       Clear saved context before start")
	fmt.Println("  -interactive         Interactive chat mode")
	fmt.Println("Interactive commands:")
	fmt.Println("  /strategy            Show active strategy and branch")
	fmt.Println("  /facts               Show sticky facts")
	fmt.Println("  /branches            List branches")
	fmt.Println("  /checkpoint <name>   Save checkpoint (branching)")
	fmt.Println("  /branch <name> [cp]  Create branch from active/cp")
	fmt.Println("  /switch <name>       Switch active branch")
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
