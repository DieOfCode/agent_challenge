package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadDay31CorpusAndRetrieve(t *testing.T) {
	workspace := t.TempDir()
	if err := os.WriteFile(filepath.Join(workspace, "README.md"), []byte("# Project\nCLI project with developer assistant."), 0o644); err != nil {
		t.Fatalf("failed to write README.md: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(workspace, "docs"), 0o755); err != nil {
		t.Fatalf("failed to create docs dir: %v", err)
	}
	docText := "# MCP\nTool git_branch returns current git branch and dirty status."
	if err := os.WriteFile(filepath.Join(workspace, "docs", "API.md"), []byte(docText), 0o644); err != nil {
		t.Fatalf("failed to write docs/API.md: %v", err)
	}

	chunks, err := loadDay31Corpus(workspace, "README.md", "docs")
	if err != nil {
		t.Fatalf("loadDay31Corpus() error = %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks, got %d", len(chunks))
	}

	retrieved := day31RetrieveChunks(chunks, "как узнать текущую git branch", 2)
	if len(retrieved) == 0 {
		t.Fatalf("expected retrieved chunks")
	}

	topSource := retrieved[0].Chunk.Source
	if !strings.Contains(topSource, "docs/API.md") {
		t.Fatalf("expected top source from API doc, got %s", topSource)
	}
}

func TestDay31SplitText(t *testing.T) {
	text := strings.Repeat("a", 1500)
	chunks := day31SplitText(text, 500, 50)
	if len(chunks) < 3 {
		t.Fatalf("expected split into several chunks, got %d", len(chunks))
	}
	if strings.TrimSpace(chunks[0]) == "" {
		t.Fatalf("first chunk is empty")
	}
}
