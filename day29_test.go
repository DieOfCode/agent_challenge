package main

import (
	"testing"
	"time"
)

func TestDay29SelectOptimizedModelAutoQuant(t *testing.T) {
	tags := []day29ModelTag{
		{
			Name: "qwen2.5:7b-q8_0",
			Size: 8_000,
			Details: day29ModelTagDetails{
				Family:            "qwen2",
				QuantizationLevel: "Q8_0",
			},
		},
		{
			Name: "qwen2.5:7b-q4_k_m",
			Size: 4_000,
			Details: day29ModelTagDetails{
				Family:            "qwen2",
				QuantizationLevel: "Q4_K_M",
			},
		},
		{
			Name: "qwen2.5:7b-q5_k_m",
			Size: 5_000,
			Details: day29ModelTagDetails{
				Family:            "qwen2",
				QuantizationLevel: "Q5_K_M",
			},
		},
	}

	picked, note := day29SelectOptimizedModel("qwen2.5:7b-q8_0", "", true, tags)
	if picked != "qwen2.5:7b-q4_k_m" {
		t.Fatalf("expected q4 model, got %s", picked)
	}
	if note == "" {
		t.Fatalf("expected non-empty optimization note")
	}
}

func TestDay29SelectOptimizedModelFallback(t *testing.T) {
	tags := []day29ModelTag{{
		Name: "qwen2.5:0.5b",
		Details: day29ModelTagDetails{
			Family:            "qwen2",
			QuantizationLevel: "Q4_K_M",
		},
	}}

	picked, _ := day29SelectOptimizedModel("qwen2.5:0.5b", "", true, tags)
	if picked != "qwen2.5:0.5b" {
		t.Fatalf("expected fallback to baseline, got %s", picked)
	}
}

func TestDay29AggregateProfile(t *testing.T) {
	bench := []day29QuestionBenchmark{
		{
			BaselineScore:    40,
			OptimizedScore:   60,
			BaselineStable:   true,
			OptimizedStable:  false,
			BaselineLatency:  2 * time.Second,
			OptimizedLatency: 1 * time.Second,
			BaselineRuns: []day29Run{{PromptTokens: 10, ResponseTokens: 20, TotalTokens: 30, LoadDuration: 500 * time.Millisecond, ModelSizeBytes: 1000, ModelVRAMBytes: 100},
				{PromptTokens: 20, ResponseTokens: 10, TotalTokens: 30, LoadDuration: 300 * time.Millisecond, ModelSizeBytes: 900, ModelVRAMBytes: 90}},
			OptimizedRuns: []day29Run{{PromptTokens: 8, ResponseTokens: 12, TotalTokens: 20, LoadDuration: 200 * time.Millisecond, ModelSizeBytes: 800, ModelVRAMBytes: 80}},
		},
		{
			BaselineScore:    60,
			OptimizedScore:   80,
			BaselineStable:   false,
			OptimizedStable:  true,
			BaselineLatency:  4 * time.Second,
			OptimizedLatency: 2 * time.Second,
			BaselineRuns:     []day29Run{{PromptTokens: 15, ResponseTokens: 15, TotalTokens: 30, LoadDuration: 400 * time.Millisecond, ModelSizeBytes: 1100, ModelVRAMBytes: 110}},
			OptimizedRuns:    []day29Run{{PromptTokens: 9, ResponseTokens: 11, TotalTokens: 20, LoadDuration: 100 * time.Millisecond, ModelSizeBytes: 700, ModelVRAMBytes: 70}},
		},
	}

	before := day29AggregateProfile(bench, true)
	after := day29AggregateProfile(bench, false)

	if before.AvgQuality != 50 || after.AvgQuality != 70 {
		t.Fatalf("unexpected avg quality before=%d after=%d", before.AvgQuality, after.AvgQuality)
	}
	if before.AvgLatency != 3*time.Second || after.AvgLatency != 1500*time.Millisecond {
		t.Fatalf("unexpected latency before=%s after=%s", before.AvgLatency, after.AvgLatency)
	}
	if before.Stability != 50 || after.Stability != 50 {
		t.Fatalf("unexpected stability before=%d after=%d", before.Stability, after.Stability)
	}
	if before.AvgTotalTokens != 30 || after.AvgTotalTokens != 20 {
		t.Fatalf("unexpected avg total tokens before=%d after=%d", before.AvgTotalTokens, after.AvgTotalTokens)
	}
	if before.MaxModelSize != 1100 || after.MaxModelVRAM != 80 {
		t.Fatalf("unexpected max resources before_size=%d after_vram=%d", before.MaxModelSize, after.MaxModelVRAM)
	}
}
