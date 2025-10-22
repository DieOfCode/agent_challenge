package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

const baseURL = "https://openrouter.ai/api/v1"

type Model struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type modelsResponse struct {
	Data []Model `json:"data"`
}

func ListModels(ctx context.Context, token string) ([]Model, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/models", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		b, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("openrouter models error: %s", string(b))
	}
	var mr modelsResponse
	if err := json.NewDecoder(res.Body).Decode(&mr); err != nil {
		return nil, err
	}
	return mr.Data, nil
}

// Chat API types (OpenAI-compatible)

type ChatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

type ChatCompletionRequest struct {
	Model          string        `json:"model"`
	Messages       []ChatMessage `json:"messages"`
	Tools          []Tool        `json:"tools,omitempty"`
	ToolChoice     string        `json:"tool_choice,omitempty"`
	ResponseFormat any           `json:"response_format,omitempty"`
	MaxTokens      int           `json:"max_tokens,omitempty"`
	Stream         bool          `json:"stream,omitempty"`
	Stop           []string      `json:"stop,omitempty"`
	Temperature    float64       `json:"temperature,omitempty"`
}

type Choice struct {
	Index        int         `json:"index"`
	FinishReason string      `json:"finish_reason"`
	Message      ChatMessage `json:"message"`
}

type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

// Usage tokens information (OpenAI-compatible). Not all providers return it.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func CreateChatCompletion(ctx context.Context, token string, reqBody ChatCompletionRequest) (*ChatCompletionResponse, error) {
	buf := new(bytes.Buffer)
	if err := json.NewEncoder(buf).Encode(reqBody); err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/chat/completions", buf)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second}
	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		b, _ := io.ReadAll(res.Body)
		return nil, errors.New(string(b))
	}
	var cr ChatCompletionResponse
	if err := json.NewDecoder(res.Body).Decode(&cr); err != nil {
		return nil, err
	}
	return &cr, nil
}
