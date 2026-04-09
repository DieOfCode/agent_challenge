package day18mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

const (
	DefaultStorePath = "/tmp/day18_scheduler_store.json"
	ToolGetSummary   = "get_summary"
)

type SchedulerStore struct {
	mu      sync.Mutex
	path    string
	entries []RunEntry
}

type RunEntry struct {
	At   string `json:"at"`
	Kind string `json:"kind"`
}

type storePayload struct {
	Version int        `json:"version"`
	Entries []RunEntry `json:"entries"`
}

type Summary struct {
	TotalRuns       int    `json:"total_runs"`
	WindowRuns      int    `json:"window_runs"`
	WindowMinutes   int    `json:"window_minutes"`
	LastRunAt       string `json:"last_run_at,omitempty"`
	StorePath       string `json:"store_path"`
	SchedulerActive bool   `json:"scheduler_active"`
}

type Scheduler struct {
	interval time.Duration
	store    *SchedulerStore
	stopCh   chan struct{}
	running  bool
}

func NewServer(name, version, storePath string, interval time.Duration) *server.MCPServer {
	serverName := strings.TrimSpace(name)
	if serverName == "" {
		serverName = "day18-scheduler-mcp-server"
	}
	serverVersion := strings.TrimSpace(version)
	if serverVersion == "" {
		serverVersion = "1.0.0"
	}
	if interval <= 0 {
		interval = 5 * time.Second
	}
	store := NewSchedulerStore(storePath)
	scheduler := NewScheduler(interval, store)
	scheduler.Start()

	mcpServer := server.NewMCPServer(
		serverName,
		serverVersion,
		server.WithToolCapabilities(true),
	)

	mcpServer.AddTool(
		mcp.NewTool(
			ToolGetSummary,
			mcp.WithDescription("Return aggregated scheduler summary (counts per window)"),
			mcp.WithNumber("window_minutes", mcp.Description("Window size in minutes"), mcp.Required()),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			windowMinutes := mcp.ParseInt(request, "window_minutes", 60)
			if windowMinutes <= 0 {
				return mcp.NewToolResultError("window_minutes must be positive"), nil
			}

			summary, err := BuildSummary(store, windowMinutes, scheduler.running)
			if err != nil {
				return mcp.NewToolResultErrorFromErr("failed to build summary", err), nil
			}
			raw, err := json.Marshal(summary)
			if err != nil {
				return mcp.NewToolResultErrorFromErr("failed to encode summary", err), nil
			}
			return mcp.NewToolResultStructured(summary, string(raw)), nil
		},
	)

	return mcpServer
}

func NewSchedulerStore(path string) *SchedulerStore {
	if strings.TrimSpace(path) == "" {
		path = DefaultStorePath
	}
	return &SchedulerStore{path: path}
}

func (s *SchedulerStore) AddRun(kind string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.load(); err != nil {
		return err
	}
	s.entries = append(s.entries, RunEntry{
		At:   time.Now().UTC().Format(time.RFC3339),
		Kind: strings.TrimSpace(kind),
	})
	return s.save()
}

func (s *SchedulerStore) Snapshot() ([]RunEntry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.load(); err != nil {
		return nil, err
	}
	out := make([]RunEntry, len(s.entries))
	copy(out, s.entries)
	return out, nil
}

func (s *SchedulerStore) Path() string {
	return s.path
}

func (s *SchedulerStore) load() error {
	raw, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			s.entries = nil
			return nil
		}
		return err
	}
	if len(raw) == 0 {
		s.entries = nil
		return nil
	}
	var payload storePayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		return fmt.Errorf("failed to parse store: %w", err)
	}
	s.entries = payload.Entries
	return nil
}

func (s *SchedulerStore) save() error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	payload := storePayload{
		Version: 1,
		Entries: s.entries,
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

func NewScheduler(interval time.Duration, store *SchedulerStore) *Scheduler {
	return &Scheduler{
		interval: interval,
		store:    store,
		stopCh:   make(chan struct{}),
	}
}

func (s *Scheduler) Start() {
	if s.running {
		return
	}
	s.running = true
	go func() {
		ticker := time.NewTicker(s.interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				_ = s.store.AddRun("tick")
			case <-s.stopCh:
				return
			}
		}
	}()
}

func (s *Scheduler) Stop() {
	if !s.running {
		return
	}
	s.running = false
	close(s.stopCh)
}

func BuildSummary(store *SchedulerStore, windowMinutes int, running bool) (Summary, error) {
	entries, err := store.Snapshot()
	if err != nil {
		return Summary{}, err
	}
	total := len(entries)
	window := time.Duration(windowMinutes) * time.Minute
	cutoff := time.Now().UTC().Add(-window)
	windowCount := 0
	lastRun := ""
	for _, entry := range entries {
		ts, err := time.Parse(time.RFC3339, entry.At)
		if err == nil && ts.After(cutoff) {
			windowCount++
		}
		lastRun = entry.At
	}
	return Summary{
		TotalRuns:       total,
		WindowRuns:      windowCount,
		WindowMinutes:   windowMinutes,
		LastRunAt:       lastRun,
		StorePath:       store.Path(),
		SchedulerActive: running,
	}, nil
}

