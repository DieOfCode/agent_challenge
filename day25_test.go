package main

import "testing"

func TestDay25UpdateTaskStateHeuristic(t *testing.T) {
	state := day25TaskState{}
	day25UpdateTaskStateHeuristic(&state, "Цель: собрать mini-chat с RAG.")
	day25UpdateTaskStateHeuristic(&state, "Ограничение: только CLI.")
	day25UpdateTaskStateHeuristic(&state, "Термин: task memory.")

	if state.Goal != "собрать mini-chat с RAG." {
		t.Fatalf("unexpected goal: %q", state.Goal)
	}
	if len(state.Constraints) == 0 {
		t.Fatalf("expected constraints to be collected")
	}
	if len(state.Terms) == 0 || state.Terms[0] != "task memory." {
		t.Fatalf("expected term extraction, got: %#v", state.Terms)
	}
}

func TestDay25GoalStable(t *testing.T) {
	okGoals := []string{
		"собрать mini-chat с RAG",
		"собрать mini-chat с RAG",
		"собрать mini-chat с RAG",
	}
	if !day25GoalStable(okGoals) {
		t.Fatalf("expected stable goals to pass")
	}

	badGoals := []string{
		"собрать mini-chat с RAG",
		"",
		"другая цель",
	}
	if day25GoalStable(badGoals) {
		t.Fatalf("expected unstable goals to fail")
	}
}

func TestDay25NormalizeOutputAddsSourceFallback(t *testing.T) {
	chunks := []day22RetrievedChunk{
		{Source: "DAY16_RESULTS.md", Section: "root", ChunkID: "c1"},
	}
	out := day25NormalizeOutput(day25ModelOutput{Answer: "ok"}, "q", chunks)
	if len(out.Sources) == 0 {
		t.Fatalf("expected fallback source")
	}
	if out.Sources[0].ChunkID != "c1" {
		t.Fatalf("unexpected source fallback: %#v", out.Sources[0])
	}
}
