package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type TaskStage string

const (
	TaskStagePlanning   TaskStage = "planning"
	TaskStageExecution  TaskStage = "execution"
	TaskStageValidation TaskStage = "validation"
	TaskStageDone       TaskStage = "done"
)

type taskTransition struct {
	At             string    `json:"at"`
	Event          string    `json:"event"`
	From           TaskStage `json:"from"`
	To             TaskStage `json:"to"`
	CurrentStep    string    `json:"current_step"`
	ExpectedAction string    `json:"expected_action"`
	Note           string    `json:"note,omitempty"`
}

type taskState struct {
	TaskID         string           `json:"task_id"`
	Stage          TaskStage        `json:"stage"`
	CurrentStep    string           `json:"current_step"`
	ExpectedAction string           `json:"expected_action"`
	Paused         bool             `json:"paused"`
	PauseReason    string           `json:"pause_reason,omitempty"`
	UpdatedAt      string           `json:"updated_at"`
	History        []taskTransition `json:"history,omitempty"`
}

type taskStateFile struct {
	Version int       `json:"version"`
	State   taskState `json:"state"`
}

type taskStateMachine struct {
	state taskState
}

type day13DemoResult struct {
	TaskID             string
	StateFile          string
	Transitions        []taskTransition
	PauseChecksPassed  bool
	ResumeChecksPassed bool
	ReloadCheckPassed  bool
	FinalState         taskState
}

func newTaskStateMachine(taskID string) *taskStateMachine {
	now := time.Now().UTC().Format(time.RFC3339)
	return &taskStateMachine{
		state: taskState{
			TaskID:         normalizeTaskID(taskID),
			Stage:          TaskStagePlanning,
			CurrentStep:    "Определить план работы",
			ExpectedAction: "Подтвердить план",
			UpdatedAt:      now,
			History: []taskTransition{
				{
					At:             now,
					Event:          "init",
					From:           "",
					To:             TaskStagePlanning,
					CurrentStep:    "Определить план работы",
					ExpectedAction: "Подтвердить план",
				},
			},
		},
	}
}

func (m *taskStateMachine) State() taskState {
	return m.state
}

func (m *taskStateMachine) Pause(reason string) {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "manual pause"
	}
	m.state.Paused = true
	m.state.PauseReason = reason
	m.touch("pause", m.state.Stage, m.state.Stage, reason)
}

func (m *taskStateMachine) Resume(note string) {
	m.state.Paused = false
	m.state.PauseReason = ""
	m.touch("resume", m.state.Stage, m.state.Stage, strings.TrimSpace(note))
}

func (m *taskStateMachine) UpdateStep(step, expectedAction string) {
	step = strings.TrimSpace(step)
	expectedAction = strings.TrimSpace(expectedAction)
	if step != "" {
		m.state.CurrentStep = step
	}
	if expectedAction != "" {
		m.state.ExpectedAction = expectedAction
	}
	m.touch("update", m.state.Stage, m.state.Stage, "")
}

func (m *taskStateMachine) TransitionTo(next TaskStage, step, expectedAction, note string) error {
	next = normalizeTaskStage(next)
	if !isValidStage(next) {
		return fmt.Errorf("invalid stage: %s", next)
	}
	if !canTransition(m.state.Stage, next) {
		return fmt.Errorf("invalid transition: %s -> %s", m.state.Stage, next)
	}
	if strings.TrimSpace(step) != "" {
		m.state.CurrentStep = strings.TrimSpace(step)
	}
	if strings.TrimSpace(expectedAction) != "" {
		m.state.ExpectedAction = strings.TrimSpace(expectedAction)
	}
	prev := m.state.Stage
	m.state.Stage = next
	if next == TaskStageDone {
		m.state.Paused = false
		m.state.PauseReason = ""
	}
	m.touch("transition", prev, next, strings.TrimSpace(note))
	return nil
}

func (m *taskStateMachine) Render() string {
	return fmt.Sprintf(
		"stage=%s\ncurrent_step=%s\nexpected_action=%s\npaused=%t\npause_reason=%s",
		m.state.Stage,
		emptyFallback(m.state.CurrentStep, "(none)"),
		emptyFallback(m.state.ExpectedAction, "(none)"),
		m.state.Paused,
		emptyFallback(m.state.PauseReason, "(none)"),
	)
}

func (m *taskStateMachine) touch(event string, from, to TaskStage, note string) {
	now := time.Now().UTC().Format(time.RFC3339)
	m.state.UpdatedAt = now
	m.state.History = append(m.state.History, taskTransition{
		At:             now,
		Event:          event,
		From:           from,
		To:             to,
		CurrentStep:    m.state.CurrentStep,
		ExpectedAction: m.state.ExpectedAction,
		Note:           note,
	})
}

func runDay13Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day13", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	taskID := fs.String("task-id", "day13-task", "Task ID")
	stateFile := fs.String("state-file", "/tmp/day13-task-state.json", "Path to task state JSON")
	reportPath := fs.String("report", "DAY13_RESULTS.md", "Markdown report path")
	interactive := fs.Bool("interactive", false, "Run interactive task-state assistant mode")
	reset := fs.Bool("reset", true, "Reset state file before run")
	model := fs.String("model", getDefaultModel(), "OpenRouter model (interactive mode)")
	maxTokens := fs.Int("max-tokens", 200, "Maximum response tokens (interactive mode)")
	temperature := fs.Float64("temperature", 0.2, "Temperature (interactive mode)")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day13 flags: %w", err)
	}
	if *help {
		printDay13Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day13 arguments: %s", strings.Join(fs.Args(), " "))
	}

	if *reset {
		if err := os.Remove(*stateFile); err != nil && !os.IsNotExist(err) {
			return err
		}
	}

	machine, err := loadOrCreateTaskMachine(*stateFile, *taskID)
	if err != nil {
		return err
	}
	if err := saveTaskMachine(*stateFile, machine); err != nil {
		return err
	}

	if *interactive {
		t := *temperature
		return runDay13Interactive(machine, *stateFile, getAPIKey(), *model, *maxTokens, &t)
	}

	result, err := runDay13Demo(*stateFile, *taskID)
	if err != nil {
		return err
	}
	printDay13DemoResult(result)
	if err := writeDay13Report(*reportPath, result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)
	return nil
}

func runDay13Demo(stateFile, taskID string) (day13DemoResult, error) {
	machine := newTaskStateMachine(taskID)
	if err := saveTaskMachine(stateFile, machine); err != nil {
		return day13DemoResult{}, err
	}

	pauseChecks := true
	resumeChecks := true

	// 1) Planning pause/resume
	machine.UpdateStep("Определить scope MVP", "Подтвердить scope и зависимости")
	machine.Pause("waiting_for_scope_confirmation")
	pauseChecks = pauseChecks && machine.State().Paused
	machine.Resume("scope_confirmed")
	resumeChecks = resumeChecks && !machine.State().Paused

	// 2) Transition to execution and pause
	if err := machine.TransitionTo(TaskStageExecution, "Реализовать API и интеграцию OneSignal", "Подготовить PR и тесты", "plan approved"); err != nil {
		return day13DemoResult{}, err
	}
	machine.Pause("waiting_for_dev_window")
	pauseChecks = pauseChecks && machine.State().Paused
	if err := saveTaskMachine(stateFile, machine); err != nil {
		return day13DemoResult{}, err
	}

	// 3) Reload from disk and continue without repeating explanation.
	loaded, err := loadOrCreateTaskMachine(stateFile, taskID)
	if err != nil {
		return day13DemoResult{}, err
	}
	reloadCheck := loaded.State().Stage == TaskStageExecution &&
		loaded.State().Paused &&
		strings.TrimSpace(loaded.State().CurrentStep) != ""

	loaded.Resume("dev_window_opened")
	resumeChecks = resumeChecks && !loaded.State().Paused

	// 4) Validation pause/resume
	if err := loaded.TransitionTo(TaskStageValidation, "Проверить тесты и acceptance criteria", "Подтвердить readiness к релизу", "implementation finished"); err != nil {
		return day13DemoResult{}, err
	}
	loaded.Pause("waiting_for_qa_results")
	pauseChecks = pauseChecks && loaded.State().Paused
	loaded.Resume("qa_passed")
	resumeChecks = resumeChecks && !loaded.State().Paused

	// 5) Done
	if err := loaded.TransitionTo(TaskStageDone, "Закрыть задачу и заархивировать артефакты", "Никаких действий", "validation passed"); err != nil {
		return day13DemoResult{}, err
	}
	if err := saveTaskMachine(stateFile, loaded); err != nil {
		return day13DemoResult{}, err
	}

	return day13DemoResult{
		TaskID:             normalizeTaskID(taskID),
		StateFile:          stateFile,
		Transitions:        append([]taskTransition(nil), loaded.State().History...),
		PauseChecksPassed:  pauseChecks,
		ResumeChecksPassed: resumeChecks,
		ReloadCheckPassed:  reloadCheck,
		FinalState:         loaded.State(),
	}, nil
}

func runDay13Interactive(machine *taskStateMachine, stateFile, apiKey, model string, maxTokens int, temperature *float64) error {
	fmt.Println("Day13 interactive mode. Type /exit to quit.")
	fmt.Println("Commands:")
	fmt.Println("  /state show")
	fmt.Println("  /state pause [reason]")
	fmt.Println("  /state resume")
	fmt.Println("  /state set <planning|execution|validation|done>")
	fmt.Println("  /state step <text>")
	fmt.Println("  /state expect <text>")
	fmt.Println("  /state next <stage>|<step>|<expected_action>")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("you> ")
		if !scanner.Scan() {
			return scanner.Err()
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		switch strings.ToLower(line) {
		case "/exit", "exit", "quit":
			return nil
		}
		if strings.HasPrefix(line, "/") {
			handled, err := handleDay13StateCommand(machine, stateFile, line)
			if err != nil {
				fmt.Fprintf(os.Stderr, "state error: %v\n", err)
			}
			if handled {
				continue
			}
		}

		if machine.State().Paused {
			fmt.Printf("agent> Task paused at `%s`. Expected action: %s. Use /state resume.\n\n",
				machine.State().Stage,
				emptyFallback(machine.State().ExpectedAction, "(none)"),
			)
			continue
		}

		reply, err := day13StateAwareReply(apiKey, model, maxTokens, temperature, machine.State(), line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "agent error: %v\n", err)
			continue
		}
		fmt.Printf("agent> %s\n\n", reply)
	}
}

func day13StateAwareReply(apiKey, model string, maxTokens int, temperature *float64, state taskState, userInput string) (string, error) {
	system := fmt.Sprintf(
		"Ты помощник выполнения задачи. Текущее состояние:\nstage=%s\ncurrent_step=%s\nexpected_action=%s\npaused=%t\n"+
			"Отвечай только в контексте текущего шага и ожидаемого действия.",
		state.Stage,
		emptyFallback(state.CurrentStep, "(none)"),
		emptyFallback(state.ExpectedAction, "(none)"),
		state.Paused,
	)

	result, err := callOpenRouterDetailed(
		apiKey,
		model,
		[]message{
			{Role: "system", Content: system},
			{Role: "user", Content: strings.TrimSpace(userInput)},
		},
		maxTokens,
		temperature,
		nil,
		"day13-state-machine",
	)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(result.Answer), nil
}

func handleDay13StateCommand(machine *taskStateMachine, stateFile, line string) (bool, error) {
	fields := strings.Fields(strings.TrimSpace(line))
	if len(fields) == 0 || strings.ToLower(fields[0]) != "/state" {
		return false, nil
	}
	if len(fields) < 2 {
		return true, fmt.Errorf("usage: /state show|pause|resume|set|step|expect|next")
	}

	action := strings.ToLower(fields[1])
	switch action {
	case "show":
		fmt.Printf("state>\n%s\n\n", machine.Render())
	case "pause":
		reason := strings.TrimSpace(strings.TrimPrefix(line, "/state pause"))
		machine.Pause(reason)
		fmt.Printf("state> paused at %s\n\n", machine.State().Stage)
	case "resume":
		machine.Resume("manual resume")
		fmt.Printf("state> resumed at %s\n\n", machine.State().Stage)
	case "set":
		if len(fields) < 3 {
			return true, fmt.Errorf("usage: /state set <stage>")
		}
		stage := normalizeTaskStage(TaskStage(fields[2]))
		if err := machine.TransitionTo(stage, "", "", "manual set"); err != nil {
			return true, err
		}
		fmt.Printf("state> stage=%s\n\n", machine.State().Stage)
	case "step":
		step := strings.TrimSpace(strings.TrimPrefix(line, "/state step"))
		if step == "" {
			return true, fmt.Errorf("usage: /state step <text>")
		}
		machine.UpdateStep(step, "")
		fmt.Printf("state> current_step updated\n\n")
	case "expect":
		expected := strings.TrimSpace(strings.TrimPrefix(line, "/state expect"))
		if expected == "" {
			return true, fmt.Errorf("usage: /state expect <text>")
		}
		machine.UpdateStep("", expected)
		fmt.Printf("state> expected_action updated\n\n")
	case "next":
		payload := strings.TrimSpace(strings.TrimPrefix(line, "/state next"))
		parts := strings.Split(payload, "|")
		if len(parts) < 3 {
			return true, fmt.Errorf("usage: /state next <stage>|<step>|<expected_action>")
		}
		stage := normalizeTaskStage(TaskStage(parts[0]))
		step := strings.TrimSpace(parts[1])
		expected := strings.TrimSpace(parts[2])
		if err := machine.TransitionTo(stage, step, expected, "manual next"); err != nil {
			return true, err
		}
		fmt.Printf("state> transitioned to %s\n\n", machine.State().Stage)
	default:
		return true, fmt.Errorf("unknown /state action: %s", action)
	}
	if err := saveTaskMachine(stateFile, machine); err != nil {
		return true, err
	}
	return true, nil
}

func loadOrCreateTaskMachine(path, taskID string) (*taskStateMachine, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return newTaskStateMachine(taskID), nil
		}
		return nil, err
	}
	if len(raw) == 0 {
		return newTaskStateMachine(taskID), nil
	}

	var file taskStateFile
	if err := json.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("failed to parse state file: %w", err)
	}
	state := file.State
	if !isValidStage(state.Stage) {
		state.Stage = TaskStagePlanning
	}
	if strings.TrimSpace(state.TaskID) == "" {
		state.TaskID = normalizeTaskID(taskID)
	}
	if state.History == nil {
		state.History = []taskTransition{}
	}
	return &taskStateMachine{state: state}, nil
}

func saveTaskMachine(path string, machine *taskStateMachine) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	payload := taskStateFile{
		Version: 1,
		State:   machine.State(),
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func printDay13DemoResult(result day13DemoResult) {
	fmt.Println("=== Day 13: Task State Machine ===")
	fmt.Printf("task_id=%s stage=%s paused=%t\n", result.TaskID, result.FinalState.Stage, result.FinalState.Paused)
	fmt.Printf("pause_checks=%t resume_checks=%t reload_check=%t\n",
		result.PauseChecksPassed, result.ResumeChecksPassed, result.ReloadCheckPassed,
	)
	fmt.Printf("transitions=%d state_file=%s\n", len(result.Transitions), result.StateFile)
}

func writeDay13Report(path string, result day13DemoResult) error {
	var b strings.Builder
	b.WriteString("# Day 13 Results: Task State Machine\n\n")
	b.WriteString(fmt.Sprintf("- task_id: `%s`\n", result.TaskID))
	b.WriteString(fmt.Sprintf("- state_file: `%s`\n", result.StateFile))
	b.WriteString(fmt.Sprintf("- pause checks passed: `%t`\n", result.PauseChecksPassed))
	b.WriteString(fmt.Sprintf("- resume checks passed: `%t`\n", result.ResumeChecksPassed))
	b.WriteString(fmt.Sprintf("- reload check passed: `%t`\n", result.ReloadCheckPassed))
	b.WriteString(fmt.Sprintf("- final stage: `%s`\n", result.FinalState.Stage))
	b.WriteString(fmt.Sprintf("- final paused: `%t`\n", result.FinalState.Paused))
	b.WriteString("\n## Final State\n")
	b.WriteString("```text\n" + sanitizeCodeFences((&taskStateMachine{state: result.FinalState}).Render()) + "\n```\n\n")
	b.WriteString("## Transition Log\n")
	for _, item := range result.Transitions {
		b.WriteString(fmt.Sprintf(
			"- `%s` event=`%s` from=`%s` to=`%s` step=`%s` expected=`%s` note=`%s`\n",
			item.At, item.Event, item.From, item.To, item.CurrentStep, item.ExpectedAction, item.Note,
		))
	}
	b.WriteString("\nConclusion: task state machine supports pause/resume at multiple stages and continues after reload without повторных объяснений.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay13Usage() {
	fmt.Println("Usage: openrouter-cli day13 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -task-id string      Task ID")
	fmt.Println("  -state-file string   Path to task state JSON")
	fmt.Println("  -report string       Markdown report path")
	fmt.Println("  -interactive         Run interactive mode")
	fmt.Println("  -reset               Reset state file before run")
	fmt.Println("  -model string        OpenRouter model (interactive mode)")
	fmt.Println("  -max-tokens int      Max response tokens (interactive mode)")
	fmt.Println("  -temperature float   Temperature (interactive mode)")
	fmt.Println("  -help                Show help")
}

func isValidStage(stage TaskStage) bool {
	switch stage {
	case TaskStagePlanning, TaskStageExecution, TaskStageValidation, TaskStageDone:
		return true
	default:
		return false
	}
}

func canTransition(from, to TaskStage) bool {
	if from == to {
		return true
	}
	switch from {
	case TaskStagePlanning:
		return to == TaskStageExecution
	case TaskStageExecution:
		return to == TaskStageValidation
	case TaskStageValidation:
		return to == TaskStageDone
	case TaskStageDone:
		return false
	default:
		return false
	}
}

func normalizeTaskStage(stage TaskStage) TaskStage {
	switch strings.ToLower(strings.TrimSpace(string(stage))) {
	case "planning":
		return TaskStagePlanning
	case "execution":
		return TaskStageExecution
	case "validation":
		return TaskStageValidation
	case "done":
		return TaskStageDone
	default:
		return stage
	}
}

func emptyFallback(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}
