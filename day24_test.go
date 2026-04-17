package main

import (
	"strings"
	"testing"
)

func TestDay24ParseStructuredAnswerWithCodeFence(t *testing.T) {
	raw := "```json\n{\"answer\":\"ok\",\"sources\":[{\"source\":\"DAY16_RESULTS.md\",\"section\":\"root\",\"chunk_id\":\"c1\"}],\"quotes\":[{\"source\":\"DAY16_RESULTS.md\",\"section\":\"root\",\"chunk_id\":\"c1\",\"quote\":\"tools_count=2\"}]}\n```"
	got, err := day24ParseStructuredAnswer(raw)
	if err != nil {
		t.Fatalf("day24ParseStructuredAnswer failed: %v", err)
	}
	if got.Answer != "ok" {
		t.Fatalf("unexpected answer: %q", got.Answer)
	}
	if len(got.Sources) != 1 || got.Sources[0].ChunkID != "c1" {
		t.Fatalf("unexpected sources: %#v", got.Sources)
	}
	if len(got.Quotes) != 1 || got.Quotes[0].Quote != "tools_count=2" {
		t.Fatalf("unexpected quotes: %#v", got.Quotes)
	}
}

func TestDay24QuotesVerbatim(t *testing.T) {
	chunks := []day22RetrievedChunk{
		{
			ChunkID: "c1",
			Source:  "DAY16_RESULTS.md",
			Section: "root",
			Text:    "tools_count=2\n1. system_time\n2. upper_text",
		},
	}
	quotes := []day24QuoteRef{
		{
			Source:  "DAY16_RESULTS.md",
			Section: "root",
			ChunkID: "c1",
			Quote:   "tools_count=2",
		},
	}
	if !day24QuotesVerbatim(quotes, chunks) {
		t.Fatalf("expected verbatim quote to pass")
	}
	quotes[0].Quote = "missing text"
	if day24QuotesVerbatim(quotes, chunks) {
		t.Fatalf("expected non-verbatim quote to fail")
	}
}

func TestDay24BuildUnsureResponse(t *testing.T) {
	chunks := []day22RetrievedChunk{
		{
			ChunkID: "c1",
			Source:  "DAY16_RESULTS.md",
			Section: "root",
			Text:    "tools_count=2",
		},
	}
	resp := day24BuildUnsureResponse("Что в Day99?", chunks, 0.2)
	if !strings.HasPrefix(strings.ToLower(resp.Answer), "не знаю") {
		t.Fatalf("expected unsure answer, got %q", resp.Answer)
	}
	if len(resp.Sources) == 0 {
		t.Fatalf("expected sources in unsure response")
	}
	if len(resp.Quotes) == 0 {
		t.Fatalf("expected quotes in unsure response")
	}
}
