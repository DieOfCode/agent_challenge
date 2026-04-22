package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type day27OllamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type day27ChatRequest struct {
	Model    string               `json:"model"`
	Messages []day27OllamaMessage `json:"messages"`
	Stream   bool                 `json:"stream"`
	Options  day26GenerateOpts    `json:"options,omitempty"`
}

type day27ChatResponse struct {
	Model   string `json:"model,omitempty"`
	Message struct {
		Role    string `json:"role,omitempty"`
		Content string `json:"content,omitempty"`
	} `json:"message"`
	Done            bool   `json:"done,omitempty"`
	TotalDuration   int64  `json:"total_duration,omitempty"`
	LoadDuration    int64  `json:"load_duration,omitempty"`
	PromptEvalCount int    `json:"prompt_eval_count,omitempty"`
	EvalCount       int    `json:"eval_count,omitempty"`
	Error           string `json:"error,omitempty"`
}

type day27Session struct {
	Messages     []day27OllamaMessage `json:"messages"`
	UpdatedAtUTC string               `json:"updated_at_utc"`
}

type day27Turn struct {
	UserInput       string
	AssistantAnswer string
	PromptTokens    int
	ResponseTokens  int
	TotalDuration   time.Duration
	LoadDuration    time.Duration
}

type day27RunResult struct {
	BaseURL string
	Model   string
	Version string
	Turns   []day27Turn
}

func runDay27Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day27", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	baseURL := fs.String("base-url", "http://127.0.0.1:11434", "Local Ollama base URL")
	model := fs.String("model", "qwen2.5:0.5b", "Local model name")
	prompt := fs.String("prompt", "", "Single user prompt")
	interactive := fs.Bool("interactive", true, "Run local interactive CLI chat")
	runDemo := fs.Bool("run-demo", false, "Run built-in 3-prompt demo and save report")
	reportPath := fs.String("report", "DAY27_RESULTS.md", "Markdown report path (used by -run-demo)")
	sessionFile := fs.String("session-file", "/tmp/day27-local-chat.json", "Session JSON file for local chat history")
	resetSession := fs.Bool("reset-session", false, "Clear stored session before run")
	timeoutSec := fs.Int("timeout-sec", 120, "HTTP timeout seconds")
	temperature := fs.Float64("temperature", 0.2, "Generation temperature")
	maxHistory := fs.Int("max-history", 20, "Max history messages kept in session")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day27 flags: %w", err)
	}
	if *help {
		printDay27Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day27 arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *timeoutSec <= 0 {
		return fmt.Errorf("timeout-sec must be positive")
	}
	if *temperature < 0 || *temperature > 2 {
		return fmt.Errorf("temperature should be in [0..2]")
	}
	if *maxHistory <= 0 {
		return fmt.Errorf("max-history must be positive")
	}
	if !*interactive && !*runDemo && strings.TrimSpace(*prompt) == "" {
		return fmt.Errorf("empty prompt: use -prompt, -interactive, or -run-demo")
	}

	client := &http.Client{Timeout: time.Duration(*timeoutSec) * time.Second}
	base := strings.TrimRight(strings.TrimSpace(*baseURL), "/")
	mdl := strings.TrimSpace(*model)
	if base == "" {
		return fmt.Errorf("base-url is empty")
	}
	if mdl == "" {
		return fmt.Errorf("model is empty")
	}

	version, err := day26CheckServer(client, base)
	if err != nil {
		return err
	}
	if err := day26EnsureModelPresent(client, base, mdl); err != nil {
		return err
	}

	if *runDemo {
		result, err := runDay27Demo(client, base, mdl, version, *temperature)
		if err != nil {
			return err
		}
		printDay27Result(result)
		if err := writeDay27Report(strings.TrimSpace(*reportPath), result); err != nil {
			return err
		}
		fmt.Printf("Отчёт: %s\n", strings.TrimSpace(*reportPath))
		return nil
	}

	if *resetSession {
		if err := os.Remove(strings.TrimSpace(*sessionFile)); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("failed to reset session: %w", err)
		}
	}
	session, err := loadDay27Session(strings.TrimSpace(*sessionFile))
	if err != nil {
		return err
	}

	if strings.TrimSpace(*prompt) != "" {
		turn, err := day27ProcessTurn(client, base, mdl, &session, strings.TrimSpace(*prompt), *temperature, *maxHistory)
		if err != nil {
			return err
		}
		if err := saveDay27Session(strings.TrimSpace(*sessionFile), session); err != nil {
			return err
		}
		printDay27Turn(turn)
		return nil
	}

	if *interactive {
		return runDay27Interactive(client, base, mdl, strings.TrimSpace(*sessionFile), &session, *temperature, *maxHistory)
	}
	return nil
}

func runDay27Demo(client *http.Client, baseURL, model, version string, temperature float64) (day27RunResult, error) {
	demoPrompts := []string{
		"Сколько будет 15 + 27? Ответ только числом.",
		"Продолжи последовательность 3, 6, 12, 24 и объясни правило в 1 предложении.",
		"Составь краткий план из 4 шагов для локального CLI-чата на LLM.",
	}
	session := day27Session{}
	result := day27RunResult{
		BaseURL: baseURL,
		Model:   model,
		Version: version,
		Turns:   make([]day27Turn, 0, len(demoPrompts)),
	}
	for _, p := range demoPrompts {
		turn, err := day27ProcessTurn(client, baseURL, model, &session, p, temperature, 20)
		if err != nil {
			return day27RunResult{}, err
		}
		result.Turns = append(result.Turns, turn)
	}
	return result, nil
}

func runDay27Interactive(client *http.Client, baseURL, model, sessionFile string, session *day27Session, temperature float64, maxHistory int) error {
	fmt.Println("Day27 local LLM chat (Ollama). Type /exit to quit.")
	fmt.Println("Commands:")
	fmt.Println("  /history")
	fmt.Println("  /reset")
	fmt.Println("")
	if len(session.Messages) > 0 {
		fmt.Printf("session> restored %d messages from %s\n\n", len(session.Messages), sessionFile)
	}

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1024), 1<<20)
	for {
		fmt.Print("you> ")
		if !scanner.Scan() {
			fmt.Println("")
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		switch line {
		case "/exit", "exit", "quit":
			return nil
		case "/history":
			fmt.Printf("history>\n%s\n\n", day27RenderHistory(session.Messages))
			continue
		case "/reset":
			session.Messages = nil
			session.UpdatedAtUTC = time.Now().UTC().Format(time.RFC3339)
			if err := saveDay27Session(sessionFile, *session); err != nil {
				return err
			}
			fmt.Println("session> reset complete")
			fmt.Println("")
			continue
		}

		turn, err := day27ProcessTurn(client, baseURL, model, session, line, temperature, maxHistory)
		if err != nil {
			fmt.Printf("error: %v\n\n", err)
			continue
		}
		if err := saveDay27Session(sessionFile, *session); err != nil {
			return err
		}
		printDay27Turn(turn)
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("failed to read stdin: %w", err)
	}
	return nil
}

func day27ProcessTurn(client *http.Client, baseURL, model string, session *day27Session, userInput string, temperature float64, maxHistory int) (day27Turn, error) {
	userInput = strings.TrimSpace(userInput)
	if userInput == "" {
		return day27Turn{}, fmt.Errorf("empty user input")
	}

	messages := append([]day27OllamaMessage{}, session.Messages...)
	messages = append(messages, day27OllamaMessage{Role: "user", Content: userInput})

	resp, err := day27Chat(client, baseURL, model, messages, temperature)
	if err != nil {
		return day27Turn{}, err
	}
	answer := strings.TrimSpace(resp.Message.Content)
	if answer == "" {
		return day27Turn{}, fmt.Errorf("empty assistant answer")
	}

	session.Messages = append(session.Messages,
		day27OllamaMessage{Role: "user", Content: userInput},
		day27OllamaMessage{Role: "assistant", Content: answer},
	)
	if len(session.Messages) > maxHistory {
		session.Messages = append([]day27OllamaMessage(nil), session.Messages[len(session.Messages)-maxHistory:]...)
	}
	session.UpdatedAtUTC = time.Now().UTC().Format(time.RFC3339)

	return day27Turn{
		UserInput:       userInput,
		AssistantAnswer: answer,
		PromptTokens:    resp.PromptEvalCount,
		ResponseTokens:  resp.EvalCount,
		TotalDuration:   time.Duration(resp.TotalDuration),
		LoadDuration:    time.Duration(resp.LoadDuration),
	}, nil
}

func day27Chat(client *http.Client, baseURL, model string, messages []day27OllamaMessage, temperature float64) (day27ChatResponse, error) {
	payload := day27ChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
		Options: day26GenerateOpts{
			Temperature: temperature,
		},
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

func loadDay27Session(path string) (day27Session, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return day27Session{}, fmt.Errorf("session file path is empty")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return day27Session{}, nil
		}
		return day27Session{}, fmt.Errorf("failed to read session file: %w", err)
	}
	var out day27Session
	if err := json.Unmarshal(data, &out); err != nil {
		return day27Session{}, fmt.Errorf("failed to parse session JSON: %w", err)
	}
	return out, nil
}

func saveDay27Session(path string, session day27Session) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return fmt.Errorf("session file path is empty")
	}
	dir := strings.TrimSpace(filepath.Dir(path))
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("failed to create session dir: %w", err)
		}
	}
	body, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode session JSON: %w", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, body, 0o644); err != nil {
		return fmt.Errorf("failed to write temp session file: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("failed to replace session file: %w", err)
	}
	return nil
}

func day27RenderHistory(messages []day27OllamaMessage) string {
	if len(messages) == 0 {
		return "(empty)"
	}
	var b strings.Builder
	for i, msg := range messages {
		b.WriteString(fmt.Sprintf("%d. %s: %s\n", i+1, msg.Role, sanitizeCodeFences(strings.TrimSpace(msg.Content))))
	}
	return strings.TrimSpace(b.String())
}

func printDay27Turn(turn day27Turn) {
	fmt.Printf("assistant> %s\n", turn.AssistantAnswer)
	fmt.Printf("tokens> prompt=%d response=%d total_duration=%s load_duration=%s\n\n",
		turn.PromptTokens,
		turn.ResponseTokens,
		turn.TotalDuration.Round(time.Millisecond),
		turn.LoadDuration.Round(time.Millisecond),
	)
}

func printDay27Result(result day27RunResult) {
	fmt.Println("=== Day 27: Local LLM Integration ===")
	fmt.Printf("server=%s version=%s model=%s\n", result.BaseURL, result.Version, result.Model)
	for i, turn := range result.Turns {
		fmt.Printf("%d) prompt=%s\n", i+1, turn.UserInput)
		fmt.Printf("answer=%s\n", turn.AssistantAnswer)
		fmt.Printf("tokens prompt=%d response=%d total_duration=%s\n\n",
			turn.PromptTokens, turn.ResponseTokens, turn.TotalDuration.Round(time.Millisecond))
	}
}

func writeDay27Report(path string, result day27RunResult) error {
	var b strings.Builder
	b.WriteString("# Day 27 Results: Local LLM Integration\n\n")
	b.WriteString(fmt.Sprintf("- server: `%s`\n", result.BaseURL))
	b.WriteString(fmt.Sprintf("- version: `%s`\n", result.Version))
	b.WriteString(fmt.Sprintf("- model: `%s`\n", result.Model))
	b.WriteString("- app type: `CLI utility (local chat)`\n")
	b.WriteString("- cloud models: `not used`\n\n")

	b.WriteString("## Demo Queries (3)\n")
	for i, turn := range result.Turns {
		b.WriteString(fmt.Sprintf("### %d\n", i+1))
		b.WriteString("Prompt:\n")
		b.WriteString("```text\n" + strings.TrimSpace(turn.UserInput) + "\n```\n\n")
		b.WriteString("Answer:\n")
		b.WriteString("```text\n" + sanitizeCodeFences(strings.TrimSpace(turn.AssistantAnswer)) + "\n```\n\n")
		b.WriteString(fmt.Sprintf("- prompt tokens: `%d`\n", turn.PromptTokens))
		b.WriteString(fmt.Sprintf("- response tokens: `%d`\n", turn.ResponseTokens))
		b.WriteString(fmt.Sprintf("- total duration: `%s`\n", turn.TotalDuration.Round(time.Millisecond)))
		b.WriteString(fmt.Sprintf("- load duration: `%s`\n\n", turn.LoadDuration.Round(time.Millisecond)))
	}

	b.WriteString("Conclusion: application sends requests to local Ollama model, receives and displays responses, and works without cloud APIs.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay27Usage() {
	fmt.Println("Usage: openrouter-cli day27 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -base-url string      Local Ollama base URL")
	fmt.Println("  -model string         Local model name")
	fmt.Println("  -prompt string        Single prompt")
	fmt.Println("  -interactive          Run interactive local chat")
	fmt.Println("  -run-demo             Run 3 built-in prompts")
	fmt.Println("  -report string        Report path (for -run-demo)")
	fmt.Println("  -session-file string  Session JSON file")
	fmt.Println("  -reset-session        Reset session file before run")
	fmt.Println("  -timeout-sec int      HTTP timeout seconds")
	fmt.Println("  -temperature float    Generation temperature")
	fmt.Println("  -max-history int      Max kept history messages")
	fmt.Println("  -help                 Show help")
}
