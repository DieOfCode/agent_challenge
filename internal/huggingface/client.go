package huggingface

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const baseURL = "https://api-inference.huggingface.co/models/"
const hubAPI = "https://huggingface.co/api/models"

type requestBody struct {
	Inputs     string                 `json:"inputs"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	Options    map[string]interface{} `json:"options,omitempty"`
}

// Generic HF text generation response (best-effort). Different backends return different shapes.
// We try a couple of common formats.
type genTextItem struct {
	GeneratedText string `json:"generated_text"`
}

type genTextDetails struct {
	FinishReason string `json:"finish_reason"`
}

type genTextRich struct {
	GeneratedText string         `json:"generated_text"`
	Details       genTextDetails `json:"details"`
}

type Options struct {
	Temperature  float64
	MaxNewTokens int
	Stop         []string
	TopP         float64
}

type Result struct {
	Text         string
	FinishReason string
}

func Generate(ctx context.Context, token, model, prompt string, opts Options) (*Result, error) {
	rb := requestBody{Inputs: prompt}
	rb.Parameters = map[string]interface{}{
		"return_full_text": false,
	}
	if opts.MaxNewTokens > 0 {
		rb.Parameters["max_new_tokens"] = opts.MaxNewTokens
	}
	if opts.Temperature > 0 {
		rb.Parameters["temperature"] = opts.Temperature
	}
	if opts.TopP > 0 {
		rb.Parameters["top_p"] = opts.TopP
	}
	if len(opts.Stop) > 0 {
		rb.Parameters["stop"] = opts.Stop
	}
	// Inference API may queue cold models; set wait_for_model to true
	rb.Options = map[string]interface{}{"wait_for_model": true}

	buf := new(bytes.Buffer)
	if err := json.NewEncoder(buf).Encode(rb); err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+model, buf)
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
		return nil, fmt.Errorf("huggingface error: %s", string(b))
	}
	// Try to decode a few possible shapes
	var anyResp any
	if err := json.NewDecoder(res.Body).Decode(&anyResp); err != nil {
		return nil, err
	}
	// Case 1: array of {generated_text}
	if arr, ok := anyResp.([]any); ok && len(arr) > 0 {
		// try rich
		if m, ok := arr[0].(map[string]any); ok {
			if _, has := m["generated_text"]; has {
				jb, _ := json.Marshal(arr[0])
				var rich genTextRich
				if err := json.Unmarshal(jb, &rich); err == nil && rich.GeneratedText != "" {
					return &Result{Text: rich.GeneratedText, FinishReason: rich.Details.FinishReason}, nil
				}
				var item genTextItem
				if err := json.Unmarshal(jb, &item); err == nil && item.GeneratedText != "" {
					return &Result{Text: item.GeneratedText}, nil
				}
			}
		}
	}
	// Case 2: plain string
	if s, ok := anyResp.(string); ok {
		return &Result{Text: s}, nil
	}
	// Fallback: marshal back to string
	jb, _ := json.Marshal(anyResp)
	return &Result{Text: string(jb)}, nil
}

// ---- List public text-generation models (non-gated/non-private) from the Hub ----
type hubModel struct {
	ID          string `json:"id"`
	PipelineTag string `json:"pipeline_tag"`
	Private     bool   `json:"private"`
	Gated       bool   `json:"gated"`
}

func ListTextGenModels(ctx context.Context, token string, limit int) ([]string, error) {
	if limit <= 0 {
		limit = 50
	}
	url := fmt.Sprintf("%s?pipeline_tag=text-generation&sort=likes&direction=-1&limit=%d", hubAPI, limit)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	client := &http.Client{Timeout: 30 * time.Second}
	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		b, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("huggingface hub error: %s", string(b))
	}
	var arr []hubModel
	if err := json.NewDecoder(res.Body).Decode(&arr); err != nil {
		return nil, err
	}
	var ids []string
	for _, m := range arr {
		if m.PipelineTag != "text-generation" {
			continue
		}
		if m.Private || m.Gated {
			continue
		}
		ids = append(ids, m.ID)
	}
	return ids, nil
}
