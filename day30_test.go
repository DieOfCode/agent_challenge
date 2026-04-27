package main

import (
	"testing"
	"time"
)

func TestDay30RateLimiter(t *testing.T) {
	limiter := newDay30RateLimiter(2, time.Minute)
	now := time.Unix(1000, 0)
	limiter.windowFrom = now

	ok, _, _ := limiter.Allow(now)
	if !ok {
		t.Fatalf("first request should be allowed")
	}
	ok, _, _ = limiter.Allow(now.Add(1 * time.Second))
	if !ok {
		t.Fatalf("second request should be allowed")
	}
	ok, _, _ = limiter.Allow(now.Add(2 * time.Second))
	if ok {
		t.Fatalf("third request in same window should be blocked")
	}
	ok, _, _ = limiter.Allow(now.Add(61 * time.Second))
	if !ok {
		t.Fatalf("request after window reset should be allowed")
	}
}

func TestDay30TrimMessages(t *testing.T) {
	in := []day27OllamaMessage{
		{Role: "user", Content: "1"},
		{Role: "assistant", Content: "2"},
		{Role: "user", Content: "3"},
		{Role: "assistant", Content: "4"},
	}
	out := day30TrimMessages(in, 2)
	if len(out) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(out))
	}
	if out[0].Content != "3" || out[1].Content != "4" {
		t.Fatalf("unexpected trimmed order: %+v", out)
	}
}

func TestDay30IsNonLoopbackURL(t *testing.T) {
	if day30IsNonLoopbackURL("http://127.0.0.1:8090") {
		t.Fatalf("loopback URL must be false")
	}
	if day30IsNonLoopbackURL("http://localhost:8090") {
		t.Fatalf("localhost URL must be false")
	}
	if !day30IsNonLoopbackURL("http://192.168.1.10:8090") {
		t.Fatalf("LAN URL must be true")
	}
	if !day30IsNonLoopbackURL("http://my-vps.example.com:8090") {
		t.Fatalf("domain URL must be true")
	}
}
