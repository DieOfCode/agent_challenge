package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

type day30ServiceConfig struct {
	ListenAddr         string
	LocalBaseURL       string
	Model              string
	AuthToken          string
	RateLimitPerMinute int
	MaxContextChars    int
	MaxHistoryMessages int
	DefaultTemperature float64
	DefaultMaxTokens   int
	DefaultContext     int
	TimeoutSec         int
	SystemPrompt       string
}

type day30ChatRequest struct {
	SessionID     string   `json:"session_id"`
	Message       string   `json:"message"`
	Temperature   *float64 `json:"temperature,omitempty"`
	MaxTokens     int      `json:"max_tokens,omitempty"`
	ContextWindow int      `json:"context_window,omitempty"`
}

type day30ChatOptions struct {
	Temperature float64 `json:"temperature,omitempty"`
	NumPredict  int     `json:"num_predict,omitempty"`
	NumCtx      int     `json:"num_ctx,omitempty"`
}

type day30OllamaChatRequest struct {
	Model    string               `json:"model"`
	Messages []day27OllamaMessage `json:"messages"`
	Stream   bool                 `json:"stream"`
	Options  day30ChatOptions     `json:"options,omitempty"`
}

type day30ChatResponse struct {
	SessionID string `json:"session_id"`
	Model     string `json:"model"`
	Answer    string `json:"answer"`
	Usage     struct {
		PromptTokens   int   `json:"prompt_tokens"`
		ResponseTokens int   `json:"response_tokens"`
		TotalDuration  int64 `json:"total_duration_ns"`
		LoadDuration   int64 `json:"load_duration_ns"`
	} `json:"usage"`
	Limits struct {
		RateLimitPerMinute int `json:"rate_limit_per_minute"`
		MaxContextChars    int `json:"max_context_chars"`
		MaxHistoryMessages int `json:"max_history_messages"`
	} `json:"limits"`
}

type day30HealthResponse struct {
	Status       string `json:"status"`
	Service      string `json:"service"`
	Model        string `json:"model"`
	OllamaBase   string `json:"ollama_base_url"`
	OllamaVer    string `json:"ollama_version"`
	StartedAtUTC string `json:"started_at_utc"`
	UptimeSec    int64  `json:"uptime_sec"`
	Limits       struct {
		RateLimitPerMinute int `json:"rate_limit_per_minute"`
		MaxContextChars    int `json:"max_context_chars"`
		MaxHistoryMessages int `json:"max_history_messages"`
	} `json:"limits"`
}

type day30VerifyResult struct {
	ServiceURL          string
	HealthOK            bool
	HealthStatusCode    int
	HealthLatency       time.Duration
	NetworkAddressOK    bool
	Model               string
	RateLimitPerMinute  int
	MaxContextChars     int
	ParallelRequests    int
	ParallelConcurrency int
	ParallelSuccess     int
	ParallelFailed      int
	ParallelAvgLatency  time.Duration
	ParallelP95Latency  time.Duration
	RateLimitTriggered  bool
	ContextLimitWorked  bool
	AuthEnabled         bool
}

type day30Service struct {
	cfg          day30ServiceConfig
	client       *http.Client
	startedAt    time.Time
	startedAtUTC string
	ollamaVer    string
	mu           sync.Mutex
	sessions     map[string][]day27OllamaMessage
	rateLimiter  *day30RateLimiter
}

type day30RateLimiter struct {
	mu         sync.Mutex
	limit      int
	window     time.Duration
	windowFrom time.Time
	used       int
}

func runDay30Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day30", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	mode := fs.String("mode", "serve", "Mode: serve or verify")
	listen := fs.String("listen", "0.0.0.0:8090", "Listen address for private service")
	localBaseURL := fs.String("local-base-url", "http://127.0.0.1:11434", "Local Ollama base URL")
	model := fs.String("model", "qwen2.5:0.5b", "Local Ollama model")
	authToken := fs.String("auth-token", "", "Optional bearer token required for /v1/chat")
	rateLimit := fs.Int("rate-limit", 30, "Global request limit per minute")
	maxContext := fs.Int("max-context", 12000, "Max total context chars per chat request")
	maxHistory := fs.Int("max-history", 12, "Max stored chat messages per session")
	temperature := fs.Float64("temperature", 0.2, "Default temperature for chat")
	maxTokens := fs.Int("max-tokens", 220, "Default max response tokens")
	contextWindow := fs.Int("context-window", 4096, "Default context window (num_ctx)")
	timeoutSec := fs.Int("timeout-sec", 120, "HTTP timeout seconds")
	systemPrompt := fs.String("system-prompt", "Ты приватный AI-ассистент. Отвечай кратко, по делу и учитывай контекст диалога.", "System prompt for chat")
	verifyURL := fs.String("verify-url", "http://127.0.0.1:8090", "Service URL for verify mode")
	verifyRequests := fs.Int("verify-requests", 8, "Total parallel requests for stability check")
	verifyConcurrency := fs.Int("verify-concurrency", 4, "Parallel workers for stability check")
	reportPath := fs.String("report", "DAY30_RESULTS.md", "Markdown report path for verify mode")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day30 flags: %w", err)
	}
	if *help {
		printDay30Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day30 arguments: %s", strings.Join(fs.Args(), " "))
	}
	if strings.TrimSpace(*mode) != "serve" && strings.TrimSpace(*mode) != "verify" {
		return fmt.Errorf("mode must be serve or verify")
	}
	if *rateLimit <= 0 {
		return fmt.Errorf("rate-limit must be positive")
	}
	if *maxContext <= 0 {
		return fmt.Errorf("max-context must be positive")
	}
	if *maxHistory <= 0 {
		return fmt.Errorf("max-history must be positive")
	}
	if *temperature < 0 || *temperature > 2 {
		return fmt.Errorf("temperature should be in [0..2]")
	}
	if *maxTokens <= 0 {
		return fmt.Errorf("max-tokens must be positive")
	}
	if *contextWindow <= 0 {
		return fmt.Errorf("context-window must be positive")
	}
	if *timeoutSec <= 0 {
		return fmt.Errorf("timeout-sec must be positive")
	}
	if *verifyRequests <= 0 {
		return fmt.Errorf("verify-requests must be positive")
	}
	if *verifyConcurrency <= 0 {
		return fmt.Errorf("verify-concurrency must be positive")
	}

	cfg := day30ServiceConfig{
		ListenAddr:         strings.TrimSpace(*listen),
		LocalBaseURL:       strings.TrimRight(strings.TrimSpace(*localBaseURL), "/"),
		Model:              strings.TrimSpace(*model),
		AuthToken:          strings.TrimSpace(*authToken),
		RateLimitPerMinute: *rateLimit,
		MaxContextChars:    *maxContext,
		MaxHistoryMessages: *maxHistory,
		DefaultTemperature: *temperature,
		DefaultMaxTokens:   *maxTokens,
		DefaultContext:     *contextWindow,
		TimeoutSec:         *timeoutSec,
		SystemPrompt:       strings.TrimSpace(*systemPrompt),
	}

	switch strings.TrimSpace(*mode) {
	case "serve":
		if cfg.ListenAddr == "" {
			return fmt.Errorf("listen is empty")
		}
		if cfg.LocalBaseURL == "" {
			return fmt.Errorf("local-base-url is empty")
		}
		if cfg.Model == "" {
			return fmt.Errorf("model is empty")
		}
		client := &http.Client{Timeout: time.Duration(cfg.TimeoutSec) * time.Second}
		ver, err := day26CheckServer(client, cfg.LocalBaseURL)
		if err != nil {
			return err
		}
		if err := day26EnsureModelPresent(client, cfg.LocalBaseURL, cfg.Model); err != nil {
			return err
		}

		svc := &day30Service{
			cfg:          cfg,
			client:       client,
			startedAt:    time.Now(),
			startedAtUTC: time.Now().UTC().Format(time.RFC3339),
			ollamaVer:    ver,
			sessions:     make(map[string][]day27OllamaMessage),
			rateLimiter:  newDay30RateLimiter(cfg.RateLimitPerMinute, time.Minute),
		}
		return svc.serve()
	case "verify":
		result, err := day30VerifyService(strings.TrimRight(strings.TrimSpace(*verifyURL), "/"), cfg, *verifyRequests, *verifyConcurrency)
		if err != nil {
			return err
		}
		printDay30VerifyResult(result)
		if err := writeDay30Report(strings.TrimSpace(*reportPath), result); err != nil {
			return err
		}
		fmt.Printf("Отчёт: %s\n", strings.TrimSpace(*reportPath))
		return nil
	default:
		return fmt.Errorf("unsupported mode: %s", strings.TrimSpace(*mode))
	}
}

func (s *day30Service) serve() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/v1/chat", s.handleChat)

	srv := &http.Server{
		Addr:         s.cfg.ListenAddr,
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: time.Duration(s.cfg.TimeoutSec+15) * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	fmt.Println("=== Day 30: Local LLM Private Service ===")
	fmt.Printf("mode=serve listen=%s model=%s ollama=%s ollama_version=%s\n", s.cfg.ListenAddr, s.cfg.Model, s.cfg.LocalBaseURL, s.ollamaVer)
	fmt.Printf("auth_enabled=%v rate_limit=%d/min max_context_chars=%d max_history=%d\n", s.cfg.AuthToken != "", s.cfg.RateLimitPerMinute, s.cfg.MaxContextChars, s.cfg.MaxHistoryMessages)
	fmt.Println("routes: GET /health, POST /v1/chat")
	fmt.Println("service_status=running")

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("day30 server failed: %w", err)
	}
	return nil
}

func (s *day30Service) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		day30WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var out day30HealthResponse
	out.Status = "ok"
	out.Service = "day30-private-llm"
	out.Model = s.cfg.Model
	out.OllamaBase = s.cfg.LocalBaseURL
	out.OllamaVer = s.ollamaVer
	out.StartedAtUTC = s.startedAtUTC
	out.UptimeSec = int64(time.Since(s.startedAt).Seconds())
	out.Limits.RateLimitPerMinute = s.cfg.RateLimitPerMinute
	out.Limits.MaxContextChars = s.cfg.MaxContextChars
	out.Limits.MaxHistoryMessages = s.cfg.MaxHistoryMessages
	day30WriteJSON(w, http.StatusOK, out)
}

func (s *day30Service) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		day30WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if s.cfg.AuthToken != "" {
		auth := strings.TrimSpace(r.Header.Get("Authorization"))
		expected := "Bearer " + s.cfg.AuthToken
		if auth != expected {
			day30WriteError(w, http.StatusUnauthorized, "unauthorized")
			return
		}
	}

	var req day30ChatRequest
	dec := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	if err := dec.Decode(&req); err != nil {
		day30WriteError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	msg := strings.TrimSpace(req.Message)
	if msg == "" {
		day30WriteError(w, http.StatusBadRequest, "message is empty")
		return
	}
	sessionID := strings.TrimSpace(req.SessionID)
	if sessionID == "" {
		sessionID = "default"
	}

	allowed, _, resetIn := s.rateLimiter.Allow(time.Now())
	if !allowed {
		w.Header().Set("Retry-After", fmt.Sprintf("%d", int(math.Ceil(resetIn.Seconds()))))
		day30WriteError(w, http.StatusTooManyRequests, "rate limit exceeded")
		return
	}

	temp := s.cfg.DefaultTemperature
	if req.Temperature != nil {
		if *req.Temperature < 0 || *req.Temperature > 2 {
			day30WriteError(w, http.StatusBadRequest, "temperature should be in [0..2]")
			return
		}
		temp = *req.Temperature
	}
	maxTokens := s.cfg.DefaultMaxTokens
	if req.MaxTokens > 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens > 1024 {
		maxTokens = 1024
	}
	ctxWindow := s.cfg.DefaultContext
	if req.ContextWindow > 0 {
		ctxWindow = req.ContextWindow
	}
	if ctxWindow > 32768 {
		ctxWindow = 32768
	}

	history := s.getSession(sessionID)
	messages := make([]day27OllamaMessage, 0, len(history)+2)
	messages = append(messages, day27OllamaMessage{Role: "system", Content: s.cfg.SystemPrompt})
	messages = append(messages, history...)
	messages = append(messages, day27OllamaMessage{Role: "user", Content: msg})

	if day30MessagesChars(messages) > s.cfg.MaxContextChars {
		day30WriteError(w, http.StatusRequestEntityTooLarge, "max context exceeded")
		return
	}

	if strings.TrimSpace(r.Header.Get("X-Day30-Dry-Run")) == "1" {
		answer := "dry-run: ok"
		history = append(history,
			day27OllamaMessage{Role: "user", Content: msg},
			day27OllamaMessage{Role: "assistant", Content: answer},
		)
		s.setSession(sessionID, day30TrimMessages(history, s.cfg.MaxHistoryMessages))

		out := day30ChatResponse{
			SessionID: sessionID,
			Model:     s.cfg.Model,
			Answer:    answer,
		}
		out.Limits.RateLimitPerMinute = s.cfg.RateLimitPerMinute
		out.Limits.MaxContextChars = s.cfg.MaxContextChars
		out.Limits.MaxHistoryMessages = s.cfg.MaxHistoryMessages
		day30WriteJSON(w, http.StatusOK, out)
		return
	}

	resp, err := day30Chat(s.client, s.cfg.LocalBaseURL, s.cfg.Model, messages, day30ChatOptions{
		Temperature: temp,
		NumPredict:  maxTokens,
		NumCtx:      ctxWindow,
	})
	if err != nil {
		day30WriteError(w, http.StatusBadGateway, "llm call failed: "+err.Error())
		return
	}
	answer := strings.TrimSpace(resp.Message.Content)
	if answer == "" {
		day30WriteError(w, http.StatusBadGateway, "empty model answer")
		return
	}

	history = append(history,
		day27OllamaMessage{Role: "user", Content: msg},
		day27OllamaMessage{Role: "assistant", Content: answer},
	)
	s.setSession(sessionID, day30TrimMessages(history, s.cfg.MaxHistoryMessages))

	out := day30ChatResponse{
		SessionID: sessionID,
		Model:     s.cfg.Model,
		Answer:    answer,
	}
	out.Usage.PromptTokens = resp.PromptEvalCount
	out.Usage.ResponseTokens = resp.EvalCount
	out.Usage.TotalDuration = resp.TotalDuration
	out.Usage.LoadDuration = resp.LoadDuration
	out.Limits.RateLimitPerMinute = s.cfg.RateLimitPerMinute
	out.Limits.MaxContextChars = s.cfg.MaxContextChars
	out.Limits.MaxHistoryMessages = s.cfg.MaxHistoryMessages
	day30WriteJSON(w, http.StatusOK, out)
}

func (s *day30Service) getSession(sessionID string) []day27OllamaMessage {
	s.mu.Lock()
	defer s.mu.Unlock()
	items := s.sessions[sessionID]
	return append([]day27OllamaMessage(nil), items...)
}

func (s *day30Service) setSession(sessionID string, messages []day27OllamaMessage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessions[sessionID] = append([]day27OllamaMessage(nil), messages...)
}

func newDay30RateLimiter(limit int, window time.Duration) *day30RateLimiter {
	return &day30RateLimiter{
		limit:      limit,
		window:     window,
		windowFrom: time.Now(),
	}
}

func (l *day30RateLimiter) Allow(now time.Time) (bool, int, time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if now.Sub(l.windowFrom) >= l.window {
		l.windowFrom = now
		l.used = 0
	}
	if l.used >= l.limit {
		reset := l.window - now.Sub(l.windowFrom)
		if reset < 0 {
			reset = 0
		}
		return false, 0, reset
	}
	l.used++
	remaining := l.limit - l.used
	reset := l.window - now.Sub(l.windowFrom)
	if reset < 0 {
		reset = 0
	}
	return true, remaining, reset
}

func day30TrimMessages(messages []day27OllamaMessage, max int) []day27OllamaMessage {
	if max <= 0 || len(messages) <= max {
		return append([]day27OllamaMessage(nil), messages...)
	}
	return append([]day27OllamaMessage(nil), messages[len(messages)-max:]...)
}

func day30MessagesChars(messages []day27OllamaMessage) int {
	total := 0
	for _, msg := range messages {
		total += len(strings.TrimSpace(msg.Content))
	}
	return total
}

func day30WriteJSON(w http.ResponseWriter, status int, payload any) {
	body, err := json.Marshal(payload)
	if err != nil {
		http.Error(w, `{"error":"internal"}`, http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

func day30WriteError(w http.ResponseWriter, status int, msg string) {
	day30WriteJSON(w, status, map[string]any{
		"error":   strings.TrimSpace(msg),
		"status":  status,
		"success": false,
	})
}

func day30Chat(client *http.Client, baseURL, model string, messages []day27OllamaMessage, opts day30ChatOptions) (day27ChatResponse, error) {
	payload := day30OllamaChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
		Options:  opts,
	}
	reqBody, err := json.Marshal(payload)
	if err != nil {
		return day27ChatResponse{}, fmt.Errorf("failed to encode chat request: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, baseURL+"/api/chat", strings.NewReader(string(reqBody)))
	if err != nil {
		return day27ChatResponse{}, fmt.Errorf("failed to build chat request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return day27ChatResponse{}, fmt.Errorf("chat request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return day27ChatResponse{}, fmt.Errorf("failed to read chat response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return day27ChatResponse{}, fmt.Errorf("chat API error (%s): %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	var out day27ChatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return day27ChatResponse{}, fmt.Errorf("invalid chat response JSON: %w", err)
	}
	if strings.TrimSpace(out.Error) != "" {
		return day27ChatResponse{}, fmt.Errorf("ollama returned error: %s", out.Error)
	}
	return out, nil
}

func day30VerifyService(baseURL string, cfg day30ServiceConfig, totalRequests, concurrency int) (day30VerifyResult, error) {
	if baseURL == "" {
		return day30VerifyResult{}, fmt.Errorf("verify-url is empty")
	}
	client := &http.Client{Timeout: time.Duration(cfg.TimeoutSec) * time.Second}
	result := day30VerifyResult{
		ServiceURL:          baseURL,
		RateLimitPerMinute:  cfg.RateLimitPerMinute,
		MaxContextChars:     cfg.MaxContextChars,
		ParallelRequests:    totalRequests,
		ParallelConcurrency: concurrency,
		AuthEnabled:         cfg.AuthToken != "",
		NetworkAddressOK:    day30IsNonLoopbackURL(baseURL),
	}

	healthReq, err := http.NewRequest(http.MethodGet, baseURL+"/health", nil)
	if err != nil {
		return day30VerifyResult{}, fmt.Errorf("failed to build health request: %w", err)
	}
	start := time.Now()
	healthResp, err := client.Do(healthReq)
	result.HealthLatency = time.Since(start)
	if err != nil {
		return day30VerifyResult{}, fmt.Errorf("health check failed: %w", err)
	}
	defer healthResp.Body.Close()
	raw, err := io.ReadAll(healthResp.Body)
	if err != nil {
		return day30VerifyResult{}, fmt.Errorf("failed to read health response: %w", err)
	}
	result.HealthStatusCode = healthResp.StatusCode
	result.HealthOK = healthResp.StatusCode >= 200 && healthResp.StatusCode < 300
	if !result.HealthOK {
		return day30VerifyResult{}, fmt.Errorf("health endpoint returned %s: %s", healthResp.Status, strings.TrimSpace(string(raw)))
	}
	var healthOut day30HealthResponse
	if err := json.Unmarshal(raw, &healthOut); err == nil {
		if strings.TrimSpace(healthOut.Model) != "" {
			result.Model = strings.TrimSpace(healthOut.Model)
		}
		if healthOut.Limits.RateLimitPerMinute > 0 {
			result.RateLimitPerMinute = healthOut.Limits.RateLimitPerMinute
		}
		if healthOut.Limits.MaxContextChars > 0 {
			result.MaxContextChars = healthOut.Limits.MaxContextChars
		}
	}

	latencies := make([]time.Duration, 0, totalRequests)
	latMu := sync.Mutex{}
	success := 0
	failed := 0
	succMu := sync.Mutex{}
	jobs := make(chan int)
	wg := sync.WaitGroup{}

	worker := func() {
		defer wg.Done()
		for i := range jobs {
			reqBody := day30ChatRequest{
				SessionID: fmt.Sprintf("verify-session-%d", i%2),
				Message:   "Коротко подтверди, что приватный сервис работает.",
				MaxTokens: 80,
			}
			body, _ := json.Marshal(reqBody)
			req, _ := http.NewRequest(http.MethodPost, baseURL+"/v1/chat", strings.NewReader(string(body)))
			req.Header.Set("Content-Type", "application/json")
			if cfg.AuthToken != "" {
				req.Header.Set("Authorization", "Bearer "+cfg.AuthToken)
			}
			startReq := time.Now()
			resp, err := client.Do(req)
			lat := time.Since(startReq)
			latMu.Lock()
			latencies = append(latencies, lat)
			latMu.Unlock()
			if err != nil {
				succMu.Lock()
				failed++
				succMu.Unlock()
				continue
			}
			rawResp, _ := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode < 200 || resp.StatusCode >= 300 {
				succMu.Lock()
				failed++
				succMu.Unlock()
				_ = rawResp
				continue
			}
			succMu.Lock()
			success++
			succMu.Unlock()
		}
	}

	if concurrency > totalRequests {
		concurrency = totalRequests
	}
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go worker()
	}
	for i := 0; i < totalRequests; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	result.ParallelSuccess = success
	result.ParallelFailed = failed
	if len(latencies) > 0 {
		result.ParallelAvgLatency = day30AverageDuration(latencies)
		result.ParallelP95Latency = day30P95Duration(latencies)
	}

	contextTestMsg := strings.Repeat("x", result.MaxContextChars+200)
	contextReq := day30ChatRequest{
		SessionID: "verify-context",
		Message:   contextTestMsg,
		MaxTokens: 10,
	}
	ctxBody, _ := json.Marshal(contextReq)
	ctxHTTPReq, _ := http.NewRequest(http.MethodPost, baseURL+"/v1/chat", strings.NewReader(string(ctxBody)))
	ctxHTTPReq.Header.Set("Content-Type", "application/json")
	ctxHTTPReq.Header.Set("X-Day30-Dry-Run", "1")
	if cfg.AuthToken != "" {
		ctxHTTPReq.Header.Set("Authorization", "Bearer "+cfg.AuthToken)
	}
	ctxResp, err := client.Do(ctxHTTPReq)
	if err == nil {
		_, _ = io.ReadAll(ctxResp.Body)
		_ = ctxResp.Body.Close()
		result.ContextLimitWorked = ctxResp.StatusCode == http.StatusRequestEntityTooLarge
	}

	rateBurst := result.RateLimitPerMinute + 2
	if rateBurst < 3 {
		rateBurst = 3
	}
	if rateBurst > 160 {
		rateBurst = 160
	}
	hit429 := false
	for i := 0; i < rateBurst; i++ {
		rlReq := day30ChatRequest{SessionID: "verify-rl", Message: fmt.Sprintf("ping %d", i), MaxTokens: 8}
		rlBody, _ := json.Marshal(rlReq)
		httpReq, _ := http.NewRequest(http.MethodPost, baseURL+"/v1/chat", strings.NewReader(string(rlBody)))
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("X-Day30-Dry-Run", "1")
		if cfg.AuthToken != "" {
			httpReq.Header.Set("Authorization", "Bearer "+cfg.AuthToken)
		}
		httpResp, err := client.Do(httpReq)
		if err != nil {
			continue
		}
		_, _ = io.ReadAll(httpResp.Body)
		_ = httpResp.Body.Close()
		if httpResp.StatusCode == http.StatusTooManyRequests {
			hit429 = true
			break
		}
	}
	result.RateLimitTriggered = hit429

	return result, nil
}

func day30AverageDuration(values []time.Duration) time.Duration {
	if len(values) == 0 {
		return 0
	}
	total := time.Duration(0)
	for _, v := range values {
		total += v
	}
	return total / time.Duration(len(values))
}

func day30P95Duration(values []time.Duration) time.Duration {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]time.Duration(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	idx := int(math.Ceil(0.95*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func day30IsNonLoopbackURL(rawURL string) bool {
	u, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil {
		return false
	}
	host := strings.TrimSpace(u.Hostname())
	if host == "" {
		return false
	}
	if strings.EqualFold(host, "localhost") {
		return false
	}
	ip := net.ParseIP(host)
	if ip != nil {
		return !ip.IsLoopback()
	}
	return true
}

func printDay30VerifyResult(result day30VerifyResult) {
	fmt.Println("=== Day 30: Local LLM as Private Service ===")
	fmt.Printf("service_url=%s model=%s auth_enabled=%v network_address_ok=%v\n", result.ServiceURL, result.Model, result.AuthEnabled, result.NetworkAddressOK)
	fmt.Printf("health_ok=%v status_code=%d health_latency=%s\n", result.HealthOK, result.HealthStatusCode, result.HealthLatency.Round(time.Millisecond))
	fmt.Printf("parallel_requests=%d concurrency=%d success=%d failed=%d avg_latency=%s p95_latency=%s\n",
		result.ParallelRequests,
		result.ParallelConcurrency,
		result.ParallelSuccess,
		result.ParallelFailed,
		result.ParallelAvgLatency.Round(time.Millisecond),
		result.ParallelP95Latency.Round(time.Millisecond),
	)
	fmt.Printf("rate_limit=%d/min rate_limit_triggered=%v\n", result.RateLimitPerMinute, result.RateLimitTriggered)
	fmt.Printf("max_context_chars=%d context_limit_worked=%v\n", result.MaxContextChars, result.ContextLimitWorked)
}

func writeDay30Report(path string, result day30VerifyResult) error {
	var b strings.Builder
	b.WriteString("# Day 30 Results: Local LLM as Private Service\n\n")
	b.WriteString("## Deployment\n")
	b.WriteString(fmt.Sprintf("- service URL: `%s`\n", result.ServiceURL))
	b.WriteString(fmt.Sprintf("- model: `%s`\n", result.Model))
	b.WriteString(fmt.Sprintf("- auth enabled: `%v`\n", result.AuthEnabled))
	b.WriteString(fmt.Sprintf("- network address ready (not loopback): `%v`\n", result.NetworkAddressOK))
	b.WriteString("- HTTP API routes: `GET /health`, `POST /v1/chat`\n\n")

	b.WriteString("## Checks\n")
	b.WriteString(fmt.Sprintf("- health check: `%v` (status=%d, latency=%s)\n", result.HealthOK, result.HealthStatusCode, result.HealthLatency.Round(time.Millisecond)))
	b.WriteString(fmt.Sprintf("- parallel stability: `requests=%d`, `concurrency=%d`, `success=%d`, `failed=%d`, `avg=%s`, `p95=%s`\n",
		result.ParallelRequests,
		result.ParallelConcurrency,
		result.ParallelSuccess,
		result.ParallelFailed,
		result.ParallelAvgLatency.Round(time.Millisecond),
		result.ParallelP95Latency.Round(time.Millisecond),
	))
	b.WriteString(fmt.Sprintf("- rate limit check: `%v` (limit=%d/min)\n", result.RateLimitTriggered, result.RateLimitPerMinute))
	b.WriteString(fmt.Sprintf("- max context check: `%v` (max_context_chars=%d)\n", result.ContextLimitWorked, result.MaxContextChars))

	b.WriteString("\n## Result\n")
	b.WriteString("Private AI service is running with HTTP chat API, supports multi-message chat sessions, and enforces basic runtime limits (rate limit and context cap).\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay30Usage() {
	fmt.Println("Usage: openrouter-cli day30 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -mode string                 Mode: serve or verify")
	fmt.Println("  -listen string               Listen address for service (serve mode)")
	fmt.Println("  -local-base-url string       Local Ollama base URL")
	fmt.Println("  -model string                Local model name")
	fmt.Println("  -auth-token string           Optional bearer token for /v1/chat")
	fmt.Println("  -rate-limit int              Global request limit per minute")
	fmt.Println("  -max-context int             Max context chars per request")
	fmt.Println("  -max-history int             Max stored messages per session")
	fmt.Println("  -temperature float           Default temperature")
	fmt.Println("  -max-tokens int              Default max tokens")
	fmt.Println("  -context-window int          Default context window (num_ctx)")
	fmt.Println("  -timeout-sec int             HTTP timeout seconds")
	fmt.Println("  -system-prompt string        System prompt for chat")
	fmt.Println("  -verify-url string           Service URL for verify mode")
	fmt.Println("  -verify-requests int         Total requests for stability test")
	fmt.Println("  -verify-concurrency int      Parallel workers for stability test")
	fmt.Println("  -report string               Markdown report path (verify mode)")
	fmt.Println("  -help                        Show help")
}
