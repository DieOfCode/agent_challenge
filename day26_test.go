package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestWriteDay26Report(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "DAY26_RESULTS.md")

	result := day26RunResult{
		BaseURL: "http://127.0.0.1:11434",
		Model:   "qwen2.5:0.5b",
		Version: "0.21.0",
		Prompts: []day26PromptResult{
			{
				Name:           "simple_math",
				Prompt:         "2+2?",
				Answer:         "4",
				PromptTokens:   4,
				ResponseTokens: 1,
				TotalDuration:  120 * time.Millisecond,
				LoadDuration:   10 * time.Millisecond,
			},
		},
		UsedCLI:     "ollama run qwen2.5:0.5b \"Привет\"",
		UsedHTTPAPI: "POST /api/generate",
	}

	if err := writeDay26Report(path, result); err != nil {
		t.Fatalf("writeDay26Report failed: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read report: %v", err)
	}
	content := string(data)
	if !strings.Contains(content, "Day 26 Results") {
		t.Fatalf("report does not contain header")
	}
	if !strings.Contains(content, "qwen2.5:0.5b") {
		t.Fatalf("report does not contain model name")
	}
}
