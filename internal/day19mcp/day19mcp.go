package day19mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

const (
	ToolSearch     = "search"
	ToolSummarize  = "summarize"
	ToolSaveToFile = "save_to_file"
	DefaultCorpus  = "OpenRouter CLI demo corpus.\n\nThis corpus contains multiple sentences about MCP tools, pipelines, and background tasks.\nIt is designed for testing the search tool, summarization tool, and save-to-file tool.\n\nMCP pipelines pass data from one tool to another.\nSummaries should capture the key points without losing accuracy.\nSaving to file should persist the final output for review."
)

type SearchResult struct {
	Query   string   `json:"query"`
	Matches []string `json:"matches"`
	Count   int      `json:"count"`
}

type SummaryResult struct {
	Summary        string `json:"summary"`
	SentenceCount  int    `json:"sentence_count"`
	InputLength    int    `json:"input_length"`
	GeneratedAtUTC string `json:"generated_at_utc"`
}

type SaveResult struct {
	Path       string `json:"path"`
	Bytes      int    `json:"bytes"`
	SavedAtUTC string `json:"saved_at_utc"`
}

func NewServer(name, version string) *server.MCPServer {
	serverName := strings.TrimSpace(name)
	if serverName == "" {
		serverName = "day19-pipeline-mcp-server"
	}
	serverVersion := strings.TrimSpace(version)
	if serverVersion == "" {
		serverVersion = "1.0.0"
	}

	mcpServer := server.NewMCPServer(
		serverName,
		serverVersion,
		server.WithToolCapabilities(true),
	)

	mcpServer.AddTool(
		mcp.NewTool(
			ToolSearch,
			mcp.WithDescription("Search corpus for query and return matching lines"),
			mcp.WithString("query", mcp.Description("Query string"), mcp.Required()),
			mcp.WithString("corpus", mcp.Description("Corpus to search"), mcp.Required()),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			query := strings.TrimSpace(mcp.ParseString(request, "query", ""))
			corpus := mcp.ParseString(request, "corpus", "")
			if query == "" {
				return mcp.NewToolResultError("query must be provided"), nil
			}
			matches := searchCorpus(corpus, query)
			result := SearchResult{Query: query, Matches: matches, Count: len(matches)}
			return structuredResult(result)
		},
	)

	mcpServer.AddTool(
		mcp.NewTool(
			ToolSummarize,
			mcp.WithDescription("Summarize provided text (simple heuristic)"),
			mcp.WithString("text", mcp.Description("Text to summarize"), mcp.Required()),
			mcp.WithNumber("max_sentences", mcp.Description("Max sentences in summary")),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			text := strings.TrimSpace(mcp.ParseString(request, "text", ""))
			if text == "" {
				return mcp.NewToolResultError("text must be provided"), nil
			}
			maxSentences := mcp.ParseInt(request, "max_sentences", 2)
			if maxSentences <= 0 {
				maxSentences = 2
			}
			summary := summarizeText(text, maxSentences)
			result := SummaryResult{
				Summary:        summary,
				SentenceCount:  countSentences(summary),
				InputLength:    len(text),
				GeneratedAtUTC: time.Now().UTC().Format(time.RFC3339),
			}
			return structuredResult(result)
		},
	)

	mcpServer.AddTool(
		mcp.NewTool(
			ToolSaveToFile,
			mcp.WithDescription("Save content to a file"),
			mcp.WithString("path", mcp.Description("File path"), mcp.Required()),
			mcp.WithString("content", mcp.Description("Content to write"), mcp.Required()),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			path := strings.TrimSpace(mcp.ParseString(request, "path", ""))
			content := mcp.ParseString(request, "content", "")
			if path == "" {
				return mcp.NewToolResultError("path must be provided"), nil
			}
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return mcp.NewToolResultErrorFromErr("failed to create directory", err), nil
			}
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				return mcp.NewToolResultErrorFromErr("failed to write file", err), nil
			}
			result := SaveResult{
				Path:       path,
				Bytes:      len(content),
				SavedAtUTC: time.Now().UTC().Format(time.RFC3339),
			}
			return structuredResult(result)
		},
	)

	return mcpServer
}

func structuredResult(payload any) (*mcp.CallToolResult, error) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return mcp.NewToolResultErrorFromErr("failed to encode response", err), nil
	}
	return mcp.NewToolResultStructured(payload, string(raw)), nil
}

func searchCorpus(corpus, query string) []string {
	if corpus == "" {
		return nil
	}
	needle := strings.ToLower(query)
	lines := strings.Split(corpus, "\n")
	matches := make([]string, 0)
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		if strings.Contains(strings.ToLower(trimmed), needle) {
			matches = append(matches, trimmed)
		}
	}
	return matches
}

func summarizeText(text string, maxSentences int) string {
	if maxSentences <= 0 {
		maxSentences = 1
	}
	sentences := splitSentences(text)
	if len(sentences) == 0 {
		return strings.TrimSpace(text)
	}
	if len(sentences) > maxSentences {
		sentences = sentences[:maxSentences]
	}
	return strings.TrimSpace(strings.Join(sentences, " "))
}

func splitSentences(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	separators := []string{".", "!", "?"}
	for _, sep := range separators {
		text = strings.ReplaceAll(text, sep, sep+"|")
	}
	parts := strings.Split(text, "|")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed == "" {
			continue
		}
		out = append(out, trimmed)
	}
	return out
}

func countSentences(text string) int {
	return len(splitSentences(text))
}

func LoadCorpusFromFile(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", fmt.Errorf("corpus file path is empty")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}
