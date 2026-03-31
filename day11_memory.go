package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type MemoryLayer string

const (
	MemoryLayerShort   MemoryLayer = "short"
	MemoryLayerWorking MemoryLayer = "working"
	MemoryLayerLong    MemoryLayer = "long"
)

type memoryFileStore struct {
	dir string
}

type shortTermMemory struct {
	Version  int             `json:"version"`
	Messages []storedMessage `json:"messages"`
}

type workingTaskMemory struct {
	Goal        string            `json:"goal,omitempty"`
	Constraints []string          `json:"constraints,omitempty"`
	Decisions   []string          `json:"decisions,omitempty"`
	Preferences []string          `json:"preferences,omitempty"`
	Notes       []string          `json:"notes,omitempty"`
	Values      map[string]string `json:"values,omitempty"`
	UpdatedAt   string            `json:"updated_at,omitempty"`
}

type workingLayerMemory struct {
	Version int                          `json:"version"`
	Tasks   map[string]workingTaskMemory `json:"tasks"`
}

type longTermLayerMemory struct {
	Version     int               `json:"version"`
	Profile     map[string]string `json:"profile,omitempty"`
	Preferences map[string]string `json:"preferences,omitempty"`
	Knowledge   map[string]string `json:"knowledge,omitempty"`
	Decisions   []string          `json:"decisions,omitempty"`
	Notes       []string          `json:"notes,omitempty"`
	UpdatedAt   string            `json:"updated_at,omitempty"`
}

type memorySnapshot struct {
	Short   []message
	Working workingTaskMemory
	Long    longTermLayerMemory
}

type MemoryRouter struct {
	store         *memoryFileStore
	taskID        string
	shortWindow   int
	shortStoreCap int
}

func newMemoryFileStore(dir string) *memoryFileStore {
	return &memoryFileStore{dir: dir}
}

func newMemoryRouter(store *memoryFileStore, taskID string, shortWindow int) *MemoryRouter {
	if shortWindow <= 0 {
		shortWindow = 10
	}
	return &MemoryRouter{
		store:         store,
		taskID:        normalizeTaskID(taskID),
		shortWindow:   shortWindow,
		shortStoreCap: 80,
	}
}

func (r *MemoryRouter) SetTask(taskID string) {
	r.taskID = normalizeTaskID(taskID)
}

func (r *MemoryRouter) TaskID() string {
	return r.taskID
}

func (r *MemoryRouter) SaveShortMessage(role, content string) error {
	role = strings.TrimSpace(role)
	content = strings.TrimSpace(content)
	if role == "" || content == "" {
		return nil
	}
	messages, err := r.store.loadShortMessages()
	if err != nil {
		return err
	}
	messages = append(messages, message{
		Role:    role,
		Content: content,
	})
	messages = keepLastMessages(messages, r.shortStoreCap)
	return r.store.saveShortMessages(messages)
}

func (r *MemoryRouter) SaveWorking(kind, key, value string) error {
	return r.store.upsertWorking(r.taskID, kind, key, value)
}

func (r *MemoryRouter) SaveLong(kind, key, value string) error {
	return r.store.upsertLong(kind, key, value)
}

func (r *MemoryRouter) SaveExplicit(layer MemoryLayer, kind, key, value string) error {
	switch layer {
	case MemoryLayerShort:
		if strings.TrimSpace(value) == "" {
			value = strings.TrimSpace(key)
		}
		return r.SaveShortMessage("user", value)
	case MemoryLayerWorking:
		return r.SaveWorking(kind, key, value)
	case MemoryLayerLong:
		return r.SaveLong(kind, key, value)
	default:
		return fmt.Errorf("unsupported memory layer: %s", layer)
	}
}

func (r *MemoryRouter) Snapshot() (memorySnapshot, error) {
	short, err := r.store.loadShortMessages()
	if err != nil {
		return memorySnapshot{}, err
	}
	workingLayer, err := r.store.loadWorkingLayer()
	if err != nil {
		return memorySnapshot{}, err
	}
	longLayer, err := r.store.loadLongLayer()
	if err != nil {
		return memorySnapshot{}, err
	}

	task := workingLayer.Tasks[r.taskID]
	return memorySnapshot{
		Short:   keepLastMessages(short, r.shortWindow),
		Working: task,
		Long:    longLayer,
	}, nil
}

func (r *MemoryRouter) Clear(layer string) error {
	switch strings.ToLower(strings.TrimSpace(layer)) {
	case "short":
		return r.store.saveShortMessages(nil)
	case "working":
		return r.store.clearWorkingTask(r.taskID)
	case "long":
		return r.store.saveLongLayer(longTermLayerMemory{Version: 1})
	case "all":
		return r.store.reset()
	default:
		return fmt.Errorf("unknown memory layer to clear: %s", layer)
	}
}

func (r *MemoryRouter) Render(layer string) (string, error) {
	snapshot, err := r.Snapshot()
	if err != nil {
		return "", err
	}
	switch strings.ToLower(strings.TrimSpace(layer)) {
	case "short":
		return renderShortMemory(snapshot.Short), nil
	case "working":
		return renderWorkingMemory(r.taskID, snapshot.Working), nil
	case "long":
		return renderLongMemory(snapshot.Long), nil
	case "all":
		return strings.TrimSpace(
			renderShortMemory(snapshot.Short) + "\n\n" +
				renderWorkingMemory(r.taskID, snapshot.Working) + "\n\n" +
				renderLongMemory(snapshot.Long),
		), nil
	default:
		return "", fmt.Errorf("unknown memory layer: %s", layer)
	}
}

func (r *MemoryRouter) BuildSystemMemoryMessages() ([]message, error) {
	snapshot, err := r.Snapshot()
	if err != nil {
		return nil, err
	}

	out := make([]message, 0, 2)
	if block := renderLongMemoryBlock(snapshot.Long); block != "" {
		out = append(out, message{
			Role:    "system",
			Content: block,
		})
	}
	if block := renderWorkingMemoryBlock(r.taskID, snapshot.Working); block != "" {
		out = append(out, message{
			Role:    "system",
			Content: block,
		})
	}
	return out, nil
}

func (s *memoryFileStore) reset() error {
	for _, file := range []string{s.shortPath(), s.workingPath(), s.longPath()} {
		if err := os.Remove(file); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

func (s *memoryFileStore) shortPath() string {
	return filepath.Join(s.dir, "short_term.json")
}

func (s *memoryFileStore) workingPath() string {
	return filepath.Join(s.dir, "working_memory.json")
}

func (s *memoryFileStore) longPath() string {
	return filepath.Join(s.dir, "long_term.json")
}

func (s *memoryFileStore) ensureDir() error {
	return os.MkdirAll(s.dir, 0o755)
}

func (s *memoryFileStore) loadShortMessages() ([]message, error) {
	var payload shortTermMemory
	if err := s.readJSON(s.shortPath(), &payload); err != nil {
		return nil, err
	}
	out := make([]message, 0, len(payload.Messages))
	for _, m := range payload.Messages {
		role := strings.TrimSpace(m.Role)
		content := strings.TrimSpace(m.Content)
		if role == "" || content == "" {
			continue
		}
		out = append(out, message{
			Role:    role,
			Content: content,
		})
	}
	return out, nil
}

func (s *memoryFileStore) saveShortMessages(messages []message) error {
	if err := s.ensureDir(); err != nil {
		return err
	}
	payload := shortTermMemory{
		Version:  1,
		Messages: make([]storedMessage, 0, len(messages)),
	}
	now := time.Now().UTC().Format(time.RFC3339)
	for _, m := range messages {
		role := strings.TrimSpace(m.Role)
		content := strings.TrimSpace(m.Content)
		if role == "" || content == "" {
			continue
		}
		payload.Messages = append(payload.Messages, storedMessage{
			Role:      role,
			Content:   content,
			Timestamp: now,
		})
	}
	return s.writeJSONAtomic(s.shortPath(), payload)
}

func (s *memoryFileStore) loadWorkingLayer() (workingLayerMemory, error) {
	payload := workingLayerMemory{
		Version: 1,
		Tasks:   map[string]workingTaskMemory{},
	}
	if err := s.readJSON(s.workingPath(), &payload); err != nil {
		return workingLayerMemory{}, err
	}
	if payload.Tasks == nil {
		payload.Tasks = map[string]workingTaskMemory{}
	}
	return payload, nil
}

func (s *memoryFileStore) saveWorkingLayer(payload workingLayerMemory) error {
	if err := s.ensureDir(); err != nil {
		return err
	}
	if payload.Version == 0 {
		payload.Version = 1
	}
	if payload.Tasks == nil {
		payload.Tasks = map[string]workingTaskMemory{}
	}
	return s.writeJSONAtomic(s.workingPath(), payload)
}

func (s *memoryFileStore) clearWorkingTask(taskID string) error {
	payload, err := s.loadWorkingLayer()
	if err != nil {
		return err
	}
	delete(payload.Tasks, normalizeTaskID(taskID))
	return s.saveWorkingLayer(payload)
}

func (s *memoryFileStore) upsertWorking(taskID, kind, key, value string) error {
	payload, err := s.loadWorkingLayer()
	if err != nil {
		return err
	}
	taskID = normalizeTaskID(taskID)
	kind = strings.ToLower(strings.TrimSpace(kind))
	key = strings.TrimSpace(key)
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}

	task := payload.Tasks[taskID]
	if task.Values == nil {
		task.Values = map[string]string{}
	}
	switch kind {
	case "goal":
		task.Goal = value
	case "constraint", "constraints":
		task.Constraints = appendUnique(task.Constraints, value)
	case "decision", "decisions":
		task.Decisions = appendUnique(task.Decisions, value)
	case "preference", "preferences":
		task.Preferences = appendUnique(task.Preferences, value)
	case "note", "notes":
		task.Notes = appendUnique(task.Notes, value)
	case "value", "kv":
		key = normalizeMemoryKey(key)
		if key == "" {
			return fmt.Errorf("working kv requires key")
		}
		task.Values[key] = value
	default:
		return fmt.Errorf("unknown working kind: %s", kind)
	}
	task.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	payload.Tasks[taskID] = task
	return s.saveWorkingLayer(payload)
}

func (s *memoryFileStore) loadLongLayer() (longTermLayerMemory, error) {
	payload := longTermLayerMemory{
		Version:     1,
		Profile:     map[string]string{},
		Preferences: map[string]string{},
		Knowledge:   map[string]string{},
	}
	if err := s.readJSON(s.longPath(), &payload); err != nil {
		return longTermLayerMemory{}, err
	}
	if payload.Profile == nil {
		payload.Profile = map[string]string{}
	}
	if payload.Preferences == nil {
		payload.Preferences = map[string]string{}
	}
	if payload.Knowledge == nil {
		payload.Knowledge = map[string]string{}
	}
	return payload, nil
}

func (s *memoryFileStore) saveLongLayer(payload longTermLayerMemory) error {
	if err := s.ensureDir(); err != nil {
		return err
	}
	if payload.Version == 0 {
		payload.Version = 1
	}
	if payload.Profile == nil {
		payload.Profile = map[string]string{}
	}
	if payload.Preferences == nil {
		payload.Preferences = map[string]string{}
	}
	if payload.Knowledge == nil {
		payload.Knowledge = map[string]string{}
	}
	return s.writeJSONAtomic(s.longPath(), payload)
}

func (s *memoryFileStore) upsertLong(kind, key, value string) error {
	payload, err := s.loadLongLayer()
	if err != nil {
		return err
	}
	kind = strings.ToLower(strings.TrimSpace(kind))
	key = normalizeMemoryKey(key)
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}

	switch kind {
	case "profile":
		if key == "" {
			return fmt.Errorf("long profile requires key")
		}
		payload.Profile[key] = value
	case "preference", "preferences":
		if key == "" {
			return fmt.Errorf("long preference requires key")
		}
		payload.Preferences[key] = value
	case "knowledge":
		if key == "" {
			return fmt.Errorf("long knowledge requires key")
		}
		payload.Knowledge[key] = value
	case "decision", "decisions":
		payload.Decisions = appendUnique(payload.Decisions, value)
	case "note", "notes":
		payload.Notes = appendUnique(payload.Notes, value)
	default:
		return fmt.Errorf("unknown long kind: %s", kind)
	}

	payload.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	return s.saveLongLayer(payload)
}

func (s *memoryFileStore) readJSON(path string, target any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if len(raw) == 0 {
		return nil
	}
	if err := json.Unmarshal(raw, target); err != nil {
		return fmt.Errorf("failed to parse %s: %w", path, err)
	}
	return nil
}

func (s *memoryFileStore) writeJSONAtomic(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func appendUnique(slice []string, value string) []string {
	value = strings.TrimSpace(value)
	if value == "" {
		return slice
	}
	for _, item := range slice {
		if strings.EqualFold(strings.TrimSpace(item), value) {
			return slice
		}
	}
	return append(slice, value)
}

func normalizeTaskID(taskID string) string {
	taskID = strings.TrimSpace(taskID)
	if taskID == "" {
		return "default-task"
	}
	return taskID
}

func normalizeMemoryKey(key string) string {
	key = strings.ToLower(strings.TrimSpace(key))
	key = strings.ReplaceAll(key, " ", "_")
	key = strings.Trim(key, "_-")
	if len(key) > 64 {
		key = key[:64]
	}
	return key
}

func renderShortMemory(short []message) string {
	if len(short) == 0 {
		return "Short-term memory: (empty)"
	}
	var b strings.Builder
	b.WriteString("Short-term memory:\n")
	for i, m := range short {
		b.WriteString(fmt.Sprintf("%d. %s: %s\n", i+1, m.Role, m.Content))
	}
	return strings.TrimSpace(b.String())
}

func renderWorkingMemory(taskID string, working workingTaskMemory) string {
	var b strings.Builder
	b.WriteString("Working memory (task: " + normalizeTaskID(taskID) + "):\n")
	if strings.TrimSpace(working.Goal) != "" {
		b.WriteString("- goal: " + working.Goal + "\n")
	}
	if len(working.Constraints) > 0 {
		b.WriteString("- constraints: " + strings.Join(working.Constraints, " | ") + "\n")
	}
	if len(working.Decisions) > 0 {
		b.WriteString("- decisions: " + strings.Join(working.Decisions, " | ") + "\n")
	}
	if len(working.Preferences) > 0 {
		b.WriteString("- preferences: " + strings.Join(working.Preferences, " | ") + "\n")
	}
	if len(working.Notes) > 0 {
		b.WriteString("- notes: " + strings.Join(working.Notes, " | ") + "\n")
	}
	if len(working.Values) > 0 {
		keys := make([]string, 0, len(working.Values))
		for key := range working.Values {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		pairs := make([]string, 0, len(keys))
		for _, key := range keys {
			pairs = append(pairs, key+"="+working.Values[key])
		}
		b.WriteString("- values: " + strings.Join(pairs, " | ") + "\n")
	}
	if b.String() == "Working memory (task: "+normalizeTaskID(taskID)+"):\n" {
		b.WriteString("(empty)\n")
	}
	return strings.TrimSpace(b.String())
}

func renderLongMemory(long longTermLayerMemory) string {
	var b strings.Builder
	b.WriteString("Long-term memory:\n")
	if len(long.Profile) > 0 {
		b.WriteString("- profile: " + mapToLine(long.Profile) + "\n")
	}
	if len(long.Preferences) > 0 {
		b.WriteString("- preferences: " + mapToLine(long.Preferences) + "\n")
	}
	if len(long.Knowledge) > 0 {
		b.WriteString("- knowledge: " + mapToLine(long.Knowledge) + "\n")
	}
	if len(long.Decisions) > 0 {
		b.WriteString("- decisions: " + strings.Join(long.Decisions, " | ") + "\n")
	}
	if len(long.Notes) > 0 {
		b.WriteString("- notes: " + strings.Join(long.Notes, " | ") + "\n")
	}
	if b.String() == "Long-term memory:\n" {
		b.WriteString("(empty)\n")
	}
	return strings.TrimSpace(b.String())
}

func renderLongMemoryBlock(long longTermLayerMemory) string {
	if len(long.Profile) == 0 && len(long.Preferences) == 0 && len(long.Knowledge) == 0 && len(long.Decisions) == 0 && len(long.Notes) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Long-term memory (stable profile and known decisions):\n")
	if len(long.Profile) > 0 {
		b.WriteString("- profile: " + mapToLine(long.Profile) + "\n")
	}
	if len(long.Preferences) > 0 {
		b.WriteString("- preferences: " + mapToLine(long.Preferences) + "\n")
	}
	if len(long.Knowledge) > 0 {
		b.WriteString("- knowledge: " + mapToLine(long.Knowledge) + "\n")
	}
	if len(long.Decisions) > 0 {
		b.WriteString("- decisions: " + strings.Join(long.Decisions, " | ") + "\n")
	}
	if len(long.Notes) > 0 {
		b.WriteString("- notes: " + strings.Join(long.Notes, " | ") + "\n")
	}
	return strings.TrimSpace(b.String())
}

func renderWorkingMemoryBlock(taskID string, working workingTaskMemory) string {
	if strings.TrimSpace(working.Goal) == "" &&
		len(working.Constraints) == 0 &&
		len(working.Decisions) == 0 &&
		len(working.Preferences) == 0 &&
		len(working.Notes) == 0 &&
		len(working.Values) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString("Working memory for current task `" + normalizeTaskID(taskID) + "`:\n")
	if strings.TrimSpace(working.Goal) != "" {
		b.WriteString("- goal: " + working.Goal + "\n")
	}
	if len(working.Constraints) > 0 {
		b.WriteString("- constraints: " + strings.Join(working.Constraints, " | ") + "\n")
	}
	if len(working.Decisions) > 0 {
		b.WriteString("- decisions: " + strings.Join(working.Decisions, " | ") + "\n")
	}
	if len(working.Preferences) > 0 {
		b.WriteString("- preferences: " + strings.Join(working.Preferences, " | ") + "\n")
	}
	if len(working.Notes) > 0 {
		b.WriteString("- notes: " + strings.Join(working.Notes, " | ") + "\n")
	}
	if len(working.Values) > 0 {
		b.WriteString("- values: " + mapToLine(working.Values) + "\n")
	}
	return strings.TrimSpace(b.String())
}

func mapToLine(values map[string]string) string {
	if len(values) == 0 {
		return ""
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, key := range keys {
		out = append(out, key+"="+values[key])
	}
	return strings.Join(out, " | ")
}
