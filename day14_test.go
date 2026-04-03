package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDetectDay14InvariantConflicts_StackAndKYC(t *testing.T) {
	invariants := defaultDay14Invariants()
	input := "Переведём backend на Node.js и MongoDB, а KYC в оплате пока отключим."

	conflicts := detectDay14InvariantConflicts(input, invariants)
	if len(conflicts) < 2 {
		t.Fatalf("expected at least 2 conflicts, got %d", len(conflicts))
	}

	ids := map[string]bool{}
	for _, c := range conflicts {
		ids[c.Invariant.ID] = true
	}
	if !ids["stack-go-postgres"] {
		t.Fatalf("expected stack-go-postgres conflict, got %+v", ids)
	}
	if !ids["biz-kyc-required"] {
		t.Fatalf("expected biz-kyc-required conflict, got %+v", ids)
	}
}

func TestLoadOrCreateDay14Invariants_CreatesDefaultFile(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "invariants.json")

	invariants, err := loadOrCreateDay14Invariants(path)
	if err != nil {
		t.Fatalf("loadOrCreateDay14Invariants returned error: %v", err)
	}
	if len(invariants) == 0 {
		t.Fatalf("expected default invariants, got empty list")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected invariants file to be created, stat error: %v", err)
	}
}

func TestDay14InvariantAgentReply_RefusesConflictingRequest(t *testing.T) {
	tmpDir := t.TempDir()
	historyPath := filepath.Join(tmpDir, "history.json")
	invariantsPath := filepath.Join(tmpDir, "invariants.json")

	invariants, err := loadOrCreateDay14Invariants(invariantsPath)
	if err != nil {
		t.Fatalf("failed to prepare invariants: %v", err)
	}

	temp := 0.2
	agent := &day14InvariantAgent{
		model:          "test-model",
		maxTokens:      200,
		temperature:    &temp,
		title:          "test-day14",
		offline:        true,
		windowSize:     10,
		historyFile:    historyPath,
		invariantsFile: invariantsPath,
		invariants:     invariants,
	}

	resp, err := agent.Reply("Перейдём на Node.js и MongoDB, KYC можно отключить.")
	if err != nil {
		t.Fatalf("Reply returned error: %v", err)
	}
	if !resp.Refused {
		t.Fatalf("expected refusal for conflicting request")
	}
	if len(resp.Conflicts) == 0 {
		t.Fatalf("expected conflicts to be returned")
	}
	if !strings.Contains(resp.Text, "STATUS: REFUSE") {
		t.Fatalf("expected refusal status in response, got: %s", resp.Text)
	}

	history, err := loadDay14History(historyPath)
	if err != nil {
		t.Fatalf("failed to load history file: %v", err)
	}
	if len(history) != 2 {
		t.Fatalf("expected 2 history messages (user+assistant), got %d", len(history))
	}
}

func TestDay14InvariantAgentReply_AllowsValidRequest(t *testing.T) {
	tmpDir := t.TempDir()
	historyPath := filepath.Join(tmpDir, "history.json")
	invariantsPath := filepath.Join(tmpDir, "invariants.json")

	invariants, err := loadOrCreateDay14Invariants(invariantsPath)
	if err != nil {
		t.Fatalf("failed to prepare invariants: %v", err)
	}

	temp := 0.2
	agent := &day14InvariantAgent{
		model:          "test-model",
		maxTokens:      200,
		temperature:    &temp,
		title:          "test-day14",
		offline:        true,
		windowSize:     10,
		historyFile:    historyPath,
		invariantsFile: invariantsPath,
		invariants:     invariants,
	}

	resp, err := agent.Reply("Дай план оптимизации текущего Go + PostgreSQL монолита.")
	if err != nil {
		t.Fatalf("Reply returned error: %v", err)
	}
	if resp.Refused {
		t.Fatalf("expected non-refusal for valid request")
	}
	if strings.Contains(strings.ToLower(resp.Text), "status: refuse") {
		t.Fatalf("unexpected refusal in response: %s", resp.Text)
	}
	if !strings.Contains(resp.Text, "STATUS: OK") {
		t.Fatalf("expected STATUS: OK in response, got: %s", resp.Text)
	}
}
