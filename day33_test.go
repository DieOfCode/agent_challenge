package main

import (
	"strings"
	"testing"

	"openrouter-cli/internal/day33mcp"
)

func TestDay33BuildRAGQuery(t *testing.T) {
	ctx := day33SupportContext{
		User:   &day33mcp.UserProfile{ID: 10, Plan: "pro", AuthProvider: "email_password"},
		Ticket: &day33mcp.SupportTicket{ID: 20, Subject: "Не работает авторизация", Category: "auth", LastError: "invalid_grant", LastErrorCode: "AUTH_401"},
	}
	query := day33BuildRAGQuery("Почему не работает авторизация?", ctx)
	for _, needle := range []string{"авториза", "invalid_grant", "AUTH_401", "pro"} {
		if !strings.Contains(strings.ToLower(query), strings.ToLower(needle)) {
			t.Fatalf("expected query to include %q, got: %s", needle, query)
		}
	}
}

func TestDay33SimulateAnswer(t *testing.T) {
	ctx := day33SupportContext{
		User:   &day33mcp.UserProfile{ID: 101, Plan: "pro", AuthProvider: "email_password", Email: "u@example.com"},
		Ticket: &day33mcp.SupportTicket{ID: 5001, Subject: "Не работает авторизация", Category: "auth", LastError: "invalid_grant", LastErrorCode: "AUTH_401", Priority: "high", Status: "open"},
	}
	retrieved := []day31RetrievedChunk{{Chunk: day31Chunk{Source: "docs/SUPPORT_FAQ.md", ChunkID: "docs/SUPPORT_FAQ.md#000"}, Score: 0.9}}
	answer := day33SimulateAnswer("Почему не работает авторизация?", ctx, retrieved)

	for _, section := range []string{"Краткий ответ", "Почему", "Шаги для пользователя", "Шаги для поддержки", "Источники"} {
		if !strings.Contains(answer, section) {
			t.Fatalf("expected section %q in answer:\n%s", section, answer)
		}
	}
	if !strings.Contains(strings.ToLower(answer), "auth") {
		t.Fatalf("expected auth hint in answer: %s", answer)
	}
}

func TestParseIntOrZero(t *testing.T) {
	cases := map[string]int{
		"42":   42,
		"":     0,
		"x":    0,
		" 77 ": 77,
		"01":   1,
	}
	for in, expected := range cases {
		if got := parseIntOrZero(in); got != expected {
			t.Fatalf("parseIntOrZero(%q)=%d, want %d", in, got, expected)
		}
	}
}
