package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

type day26GenerateRequest struct {
	Model   string            `json:"model"`
	Prompt  string            `json:"prompt"`
	Stream  bool              `json:"stream"`
	Options day26GenerateOpts `json:"options,omitempty"`
}

type day26GenerateOpts struct {
	Temperature float64 `json:"temperature,omitempty"`
}

type day26GenerateResponse struct {
	Model           string `json:"model,omitempty"`
	Response        string `json:"response,omitempty"`
	Done            bool   `json:"done,omitempty"`
	TotalDuration   int64  `json:"total_duration,omitempty"`
	LoadDuration    int64  `json:"load_duration,omitempty"`
	PromptEvalCount int    `json:"prompt_eval_count,omitempty"`
	EvalCount       int    `json:"eval_count,omitempty"`
	Error           string `json:"error,omitempty"`
}

type day26ModelTag struct {
	Name string `json:"name"`
}

type day26TagsResponse struct {
	Models []day26ModelTag `json:"models"`
}

type day26PromptResult struct {
	Name           string
	Prompt         string
	Answer         string
	PromptTokens   int
	ResponseTokens int
	TotalDuration  time.Duration
	LoadDuration   time.Duration
}

type day26RunResult struct {
	BaseURL     string
	Model       string
	Version     string
	Prompts     []day26PromptResult
	UsedCLI     string
	UsedHTTPAPI string
}

func runDay26Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day26", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	baseURL := fs.String("base-url", "http://127.0.0.1:11434", "Local Ollama base URL")
	model := fs.String("model", "qwen2.5:0.5b", "Local model name")
	reportPath := fs.String("report", "DAY26_RESULTS.md", "Markdown report output")
	timeoutSec := fs.Int("timeout-sec", 120, "HTTP timeout seconds")
	temperature := fs.Float64("temperature", 0.2, "Generation temperature")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day26 flags: %w", err)
	}
	if *help {
		printDay26Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day26 arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *timeoutSec <= 0 {
		return fmt.Errorf("timeout-sec must be positive")
	}
	if *temperature < 0 || *temperature > 2 {
		return fmt.Errorf("temperature should be in [0..2]")
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

	prompts := []day26PromptResult{
		{
			Name:   "simple_math",
			Prompt: "Сколько будет 17 + 25? Ответ только числом.",
		},
		{
			Name:   "logic",
			Prompt: "Дана последовательность: 2, 4, 8, 16. Какое следующее число и почему?",
		},
		{
			Name: "analysis",
			Prompt: "Составь краткий план из 5 шагов: как запустить локальный мини-чат с RAG, " +
				"источниками и памятью задачи.",
		},
	}

	for i := range prompts {
		result, err := day26Generate(client, base, mdl, prompts[i].Prompt, *temperature)
		if err != nil {
			return fmt.Errorf("prompt %s failed: %w", prompts[i].Name, err)
		}
		prompts[i].Answer = result.Answer
		prompts[i].PromptTokens = result.PromptTokens
		prompts[i].ResponseTokens = result.ResponseTokens
		prompts[i].TotalDuration = result.TotalDuration
		prompts[i].LoadDuration = result.LoadDuration
	}

	run := day26RunResult{
		BaseURL: base,
		Model:   mdl,
		Version: version,
		Prompts: prompts,
		UsedCLI: "ollama run " + mdl + " \"Привет!\"",
		UsedHTTPAPI: "POST " + base + "/api/generate " +
			`{"model":"` + mdl + `","prompt":"...","stream":false}`,
	}

	printDay26Result(run)
	if err := writeDay26Report(strings.TrimSpace(*reportPath), run); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", strings.TrimSpace(*reportPath))
	return nil
}

func day26CheckServer(client *http.Client, baseURL string) (string, error) {
	req, err := http.NewRequest(http.MethodGet, baseURL+"/api/version", nil)
	if err != nil {
		return "", fmt.Errorf("failed to build version request: %w", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to reach local Ollama at %s: %w", baseURL, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read version response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("ollama version API error (%s): %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	var out struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", fmt.Errorf("invalid version JSON: %w", err)
	}
	if strings.TrimSpace(out.Version) == "" {
		return "", fmt.Errorf("version field is empty in response")
	}
	return out.Version, nil
}

func day26EnsureModelPresent(client *http.Client, baseURL, model string) error {
	req, err := http.NewRequest(http.MethodGet, baseURL+"/api/tags", nil)
	if err != nil {
		return fmt.Errorf("failed to build tags request: %w", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to query model list: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read tags response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return fmt.Errorf("ollama tags API error (%s): %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	var tags day26TagsResponse
	if err := json.Unmarshal(raw, &tags); err != nil {
		return fmt.Errorf("invalid tags JSON: %w", err)
	}

	for _, item := range tags.Models {
		name := strings.TrimSpace(item.Name)
		if name == model || strings.EqualFold(name, model) {
			return nil
		}
	}
	return fmt.Errorf("model %q not found locally; run: ollama pull %s", model, model)
}

func day26Generate(client *http.Client, baseURL, model, prompt string, temperature float64) (day26PromptResult, error) {
	payload := day26GenerateRequest{
		Model:  model,
		Prompt: prompt,
		Stream: false,
		Options: day26GenerateOpts{
			Temperature: temperature,
		},
	}
	reqBody, err := json.Marshal(payload)
	if err != nil {
		return day26PromptResult{}, fmt.Errorf("failed to encode generate request: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, baseURL+"/api/generate", bytes.NewReader(reqBody))
	if err != nil {
		return day26PromptResult{}, fmt.Errorf("failed to build generate request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return day26PromptResult{}, fmt.Errorf("generate request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return day26PromptResult{}, fmt.Errorf("failed to read generate response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return day26PromptResult{}, fmt.Errorf("generate API error (%s): %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	var out day26GenerateResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return day26PromptResult{}, fmt.Errorf("invalid generate JSON: %w", err)
	}
	if strings.TrimSpace(out.Error) != "" {
		return day26PromptResult{}, fmt.Errorf("ollama returned error: %s", out.Error)
	}
	answer := strings.TrimSpace(out.Response)
	if answer == "" {
		return day26PromptResult{}, fmt.Errorf("empty model response")
	}

	return day26PromptResult{
		Answer:         answer,
		PromptTokens:   out.PromptEvalCount,
		ResponseTokens: out.EvalCount,
		TotalDuration:  time.Duration(out.TotalDuration),
		LoadDuration:   time.Duration(out.LoadDuration),
	}, nil
}

func printDay26Result(result day26RunResult) {
	fmt.Println("=== Day 26: Local LLM Run ===")
	fmt.Printf("server=%s version=%s model=%s\n", result.BaseURL, result.Version, result.Model)
	fmt.Printf("cli_example=%s\n", result.UsedCLI)
	fmt.Printf("http_api_example=%s\n", result.UsedHTTPAPI)
	for i, item := range result.Prompts {
		fmt.Printf("%d) %s prompt_tokens=%d response_tokens=%d total_duration=%s\n",
			i+1,
			item.Name,
			item.PromptTokens,
			item.ResponseTokens,
			item.TotalDuration.Round(time.Millisecond),
		)
		fmt.Printf("prompt: %s\n", item.Prompt)
		fmt.Printf("answer: %s\n\n", item.Answer)
	}
}

func writeDay26Report(path string, result day26RunResult) error {
	var b strings.Builder
	b.WriteString("# Day 26 Results: Local LLM\n\n")
	b.WriteString(fmt.Sprintf("- server: `%s`\n", result.BaseURL))
	b.WriteString(fmt.Sprintf("- version: `%s`\n", result.Version))
	b.WriteString(fmt.Sprintf("- model: `%s`\n", result.Model))
	b.WriteString(fmt.Sprintf("- CLI access example: `%s`\n", result.UsedCLI))
	b.WriteString(fmt.Sprintf("- HTTP API access example: `%s`\n\n", result.UsedHTTPAPI))

	b.WriteString("## Prompts\n")
	for i, item := range result.Prompts {
		b.WriteString(fmt.Sprintf("### %d. %s\n", i+1, item.Name))
		b.WriteString(fmt.Sprintf("- prompt tokens: `%d`\n", item.PromptTokens))
		b.WriteString(fmt.Sprintf("- response tokens: `%d`\n", item.ResponseTokens))
		b.WriteString(fmt.Sprintf("- total duration: `%s`\n", item.TotalDuration.Round(time.Millisecond)))
		b.WriteString(fmt.Sprintf("- load duration: `%s`\n\n", item.LoadDuration.Round(time.Millisecond)))
		b.WriteString("Prompt:\n")
		b.WriteString("```text\n" + strings.TrimSpace(item.Prompt) + "\n```\n\n")
		b.WriteString("Answer:\n")
		b.WriteString("```text\n" + sanitizeCodeFences(strings.TrimSpace(item.Answer)) + "\n```\n\n")
	}

	b.WriteString("Conclusion: local model is running and responds to simple, logical, and analytical prompts via local API.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay26Usage() {
	fmt.Println("Usage: openrouter-cli day26 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -base-url string      Local Ollama base URL")
	fmt.Println("  -model string         Local model name")
	fmt.Println("  -report string        Markdown report path")
	fmt.Println("  -timeout-sec int      HTTP timeout seconds")
	fmt.Println("  -temperature float    Generation temperature")
	fmt.Println("  -help                 Show help")
}
