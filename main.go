package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
)

const openRouterURL = "https://openrouter.ai/api/v1/chat/completions"

type chatRequest struct {
	Model    string    `json:"model"`
	Messages []message `json:"messages"`
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func main() {
	if err := loadDotEnv(".env"); err != nil {
		exitf("failed to load .env: %v", err)
	}

	defaultModel := strings.TrimSpace(os.Getenv("OPENROUTER_MODEL"))
	if defaultModel == "" {
		defaultModel = "openai/gpt-4o-mini"
	}

	model := flag.String("model", defaultModel, "OpenRouter model")
	prompt := flag.String("prompt", "", "Prompt text (if empty, will read from stdin)")
	flag.Parse()

	apiKey := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	if apiKey == "" {
		exitf("OPENROUTER_API_KEY is not set")
	}

	userPrompt := strings.TrimSpace(*prompt)
	if userPrompt == "" {
		input, err := io.ReadAll(os.Stdin)
		if err != nil {
			exitf("failed to read stdin: %v", err)
		}
		userPrompt = strings.TrimSpace(string(input))
	}
	if userPrompt == "" {
		exitf("empty prompt: use -prompt or pipe text to stdin")
	}

	reqBody, err := json.Marshal(chatRequest{
		Model: *model,
		Messages: []message{
			{Role: "user", Content: userPrompt},
		},
	})
	if err != nil {
		exitf("failed to encode request: %v", err)
	}

	req, err := http.NewRequest(http.MethodPost, openRouterURL, bytes.NewReader(reqBody))
	if err != nil {
		exitf("failed to create request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	// Optional but recommended by OpenRouter for analytics/rate attribution.
	req.Header.Set("HTTP-Referer", "https://localhost")
	req.Header.Set("X-Title", "minimal-go-cli")

	client := &http.Client{Timeout: 45 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		exitf("request failed: %v", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		exitf("failed to read response: %v", err)
	}

	var out chatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		exitf("invalid JSON response: %v\nraw: %s", err, string(raw))
	}

	if resp.StatusCode >= 400 {
		if out.Error != nil && out.Error.Message != "" {
			exitf("API error (%s): %s", resp.Status, out.Error.Message)
		}
		exitf("API error (%s): %s", resp.Status, string(raw))
	}

	if len(out.Choices) == 0 {
		exitf("no choices in response")
	}

	fmt.Println(strings.TrimSpace(out.Choices[0].Message.Content))
}

func exitf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

var envLineRE = regexp.MustCompile(`^([A-Za-z_][A-Za-z0-9_]*)=(.*)$`)

func loadDotEnv(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	for i, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		matches := envLineRE.FindStringSubmatch(line)
		if len(matches) != 3 {
			return fmt.Errorf("invalid .env format at line %d", i+1)
		}

		key := matches[1]
		value := strings.TrimSpace(matches[2])
		if len(value) >= 2 {
			if (value[0] == '"' && value[len(value)-1] == '"') || (value[0] == '\'' && value[len(value)-1] == '\'') {
				value = value[1 : len(value)-1]
			}
		}

		// Keep explicit shell env vars higher priority than values from .env.
		if _, exists := os.LookupEnv(key); !exists {
			if err := os.Setenv(key, value); err != nil {
				return err
			}
		}
	}

	return nil
}
