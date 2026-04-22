package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDay27SessionRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "day27-session.json")

	in := day27Session{
		Messages: []day27OllamaMessage{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello"},
		},
		UpdatedAtUTC: "2026-04-22T00:00:00Z",
	}
	if err := saveDay27Session(path, in); err != nil {
		t.Fatalf("saveDay27Session failed: %v", err)
	}
	out, err := loadDay27Session(path)
	if err != nil {
		t.Fatalf("loadDay27Session failed: %v", err)
	}
	if len(out.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(out.Messages))
	}
	if out.Messages[1].Content != "hello" {
		t.Fatalf("unexpected message content: %q", out.Messages[1].Content)
	}
}

func TestWriteDay27Report(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "DAY27_RESULTS.md")

	result := day27RunResult{
		BaseURL: "http://127.0.0.1:11434",
		Version: "0.21.0",
		Model:   "qwen2.5:0.5b",
		Turns: []day27Turn{
			{
				UserInput:       "2+2?",
				AssistantAnswer: "4",
				PromptTokens:    4,
				ResponseTokens:  1,
				TotalDuration:   120 * time.Millisecond,
				LoadDuration:    10 * time.Millisecond,
			},
		},
	}
	if err := writeDay27Report(path, result); err != nil {
		t.Fatalf("writeDay27Report failed: %v", err)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read report: %v", err)
	}
	content := string(body)
	if !strings.Contains(content, "Day 27 Results") {
		t.Fatalf("report header missing")
	}
	if !strings.Contains(content, "cloud models: `not used`") {
		t.Fatalf("report should mention local-only mode")
	}
}
