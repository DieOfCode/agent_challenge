package main

import (
	"testing"
	"time"
)

func TestConnectAndListMCPToolsInProcess(t *testing.T) {
	result, err := connectAndListMCPToolsInProcess("test-day16-server", "1.0.0", 5*time.Second)
	if err != nil {
		t.Fatalf("connectAndListMCPToolsInProcess returned error: %v", err)
	}
	if !result.Connected {
		t.Fatalf("expected connected=true")
	}
	if result.Transport != "inprocess" {
		t.Fatalf("expected transport=inprocess, got %s", result.Transport)
	}
	if result.ServerName != "test-day16-server" {
		t.Fatalf("expected server name test-day16-server, got %s", result.ServerName)
	}
	if len(result.ToolNames) == 0 {
		t.Fatalf("expected at least one tool")
	}

	if !containsDay16Tool(result.ToolNames, "echo_text") || !containsDay16Tool(result.ToolNames, "sum_numbers") {
		t.Fatalf("expected tools echo_text and sum_numbers, got %+v", result.ToolNames)
	}
}

func TestConnectAndListMCPToolsStdio(t *testing.T) {
	result, err := connectAndListMCPToolsStdio("go", []string{"run", "./cmd/day16_stdio_server"}, nil, 15*time.Second)
	if err != nil {
		t.Fatalf("connectAndListMCPToolsStdio returned error: %v", err)
	}
	if !result.Connected {
		t.Fatalf("expected connected=true")
	}
	if result.Transport != "stdio" {
		t.Fatalf("expected transport=stdio, got %s", result.Transport)
	}
	if result.ServerName != "day16-stdio-server" {
		t.Fatalf("expected server name day16-stdio-server, got %s", result.ServerName)
	}
	if len(result.ToolNames) == 0 {
		t.Fatalf("expected at least one tool")
	}
	if !containsDay16Tool(result.ToolNames, "upper_text") || !containsDay16Tool(result.ToolNames, "system_time") {
		t.Fatalf("expected tools upper_text and system_time, got %+v", result.ToolNames)
	}
}

func containsDay16Tool(names []string, target string) bool {
	for _, name := range names {
		if name == target {
			return true
		}
	}
	return false
}
