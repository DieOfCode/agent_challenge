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

type day15Stage string

const (
	day15StagePlanning   day15Stage = "planning"
	day15StageExecution  day15Stage = "execution"
	day15StageValidation day15Stage = "validation"
	day15StageDone       day15Stage = "done"
)

type day15Transition struct {
	At             string     `json:"at"`
	Event          string     `json:"event"`
	From           day15Stage `json:"from"`
	To             day15Stage `json:"to"`
	CurrentStep    string     `json:"current_step"`
	ExpectedAction string     `json:"expected_action"`
	PlanApproved   bool       `json:"plan_approved"`
	Paused         bool       `json:"paused"`
	Note           string     `json:"note,omitempty"`
}

type day15State struct {
	TaskID         string            `json:"task_id"`
	Stage          day15Stage        `json:"stage"`
	PlanApproved   bool              `json:"plan_approved"`
	CurrentStep    string            `json:"current_step"`
	ExpectedAction string            `json:"expected_action"`
	Paused         bool              `json:"paused"`
	PauseReason    string            `json:"pause_reason,omitempty"`
	UpdatedAt      string            `json:"updated_at"`
	History        []day15Transition `json:"history,omitempty"`
}

type day15StateFile struct {
	Version int        `json:"version"`
	State   day15State `json:"state"`
}

type day15StateMachine struct {
	state day15State
}

type day15DemoResult struct {
	TaskID                       string
	StateFile                    string
	Transitions                  []day15Transition
	BlockedWithoutApproval       bool
	BlockedJumpToDone            bool
	BlockedTransitionWhilePaused bool
	ReloadResumePassed           bool
	FinalState                   day15State
}

func runDay15Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day15", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	taskID := fs.String("task-id", "day15-task", "Task ID")
	stateFile := fs.String("state-file", "/tmp/day15-task-state.json", "Path to task state JSON")
	reportPath := fs.String("report", "DAY15_RESULTS.md", "Markdown report path")
	interactive := fs.Bool("interactive", false, "Run interactive lifecycle mode")
	reset := fs.Bool("reset", true, "Reset state file before run")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day15 flags: %w", err)
	}
	if *help {
		printDay15Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day15 arguments: %s", strings.Join(fs.Args(), " "))
	}

	if *reset {
		if err := os.Remove(*stateFile); err != nil && !os.IsNotExist(err) {
			return err
		}
	}

	machine, err := loadOrCreateDay15Machine(*stateFile, *taskID)
	if err != nil {
		return err
	}
	if err := saveDay15Machine(*stateFile, machine); err != nil {
		return err
	}

	if *interactive {
		return runDay15Interactive(machine, *stateFile)
	}

	result, err := runDay15Demo(*stateFile, *taskID)
	if err != nil {
		return err
	}
	printDay15DemoResult(result)
	if err := writeDay15Report(*reportPath, result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)
	return nil
}

func newDay15StateMachine(taskID string) *day15StateMachine {
	now := time.Now().UTC().Format(time.RFC3339)
	return &day15StateMachine{
		state: day15State{
			TaskID:         normalizeTaskID(taskID),
			Stage:          day15StagePlanning,
			PlanApproved:   false,
			CurrentStep:    "Подготовить и согласовать план",
			ExpectedAction: "Утвердить план",
			UpdatedAt:      now,
			History: []day15Transition{
				{
					At:             now,
					Event:          "init",
					From:           "",
					To:             day15StagePlanning,
					CurrentStep:    "Подготовить и согласовать план",
					ExpectedAction: "Утвердить план",
					PlanApproved:   false,
					Paused:         false,
				},
			},
		},
	}
}

func (m *day15StateMachine) State() day15State {
	return m.state
}

func (m *day15StateMachine) ApprovePlan(note string) error {
	if m.state.Stage != day15StagePlanning {
		return fmt.Errorf("plan can be approved only in planning stage (current: %s)", m.state.Stage)
	}
	m.state.PlanApproved = true
	if strings.TrimSpace(m.state.ExpectedAction) == "" || strings.Contains(strings.ToLower(m.state.ExpectedAction), "утверд") {
		m.state.ExpectedAction = "Начать реализацию"
	}
	m.touch("approve_plan", m.state.Stage, m.state.Stage, strings.TrimSpace(note))
	return nil
}

func (m *day15StateMachine) Pause(reason string) error {
	if m.state.Stage == day15StageDone {
		return fmt.Errorf("cannot pause done stage")
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "manual pause"
	}
	m.state.Paused = true
	m.state.PauseReason = reason
	m.touch("pause", m.state.Stage, m.state.Stage, reason)
	return nil
}

func (m *day15StateMachine) Resume(note string) error {
	if !m.state.Paused {
		return fmt.Errorf("task is not paused")
	}
	m.state.Paused = false
	m.state.PauseReason = ""
	m.touch("resume", m.state.Stage, m.state.Stage, strings.TrimSpace(note))
	return nil
}

func (m *day15StateMachine) UpdateStep(step, expectedAction string) {
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

func (m *day15StateMachine) TransitionTo(next day15Stage, step, expectedAction, note string) error {
	next = normalizeDay15Stage(next)
	if !isValidDay15Stage(next) {
		return fmt.Errorf("invalid stage: %s", next)
	}
	if m.state.Stage == day15StageDone && next != day15StageDone {
		return fmt.Errorf("cannot transition from done to %s", next)
	}
	if m.state.Paused && next != m.state.Stage {
		return fmt.Errorf("task is paused at %s; resume first", m.state.Stage)
	}
	if !canTransitionDay15(m.state.Stage, next) {
		return fmt.Errorf("invalid transition: %s -> %s (allowed next: %s)", m.state.Stage, next, strings.Join(day15AllowedNext(m.state.Stage), ", "))
	}
	if m.state.Stage == day15StagePlanning && next == day15StageExecution && !m.state.PlanApproved {
		return fmt.Errorf("cannot move to execution: plan is not approved")
	}

	if strings.TrimSpace(step) != "" {
		m.state.CurrentStep = strings.TrimSpace(step)
	}
	if strings.TrimSpace(expectedAction) != "" {
		m.state.ExpectedAction = strings.TrimSpace(expectedAction)
	}

	prev := m.state.Stage
	m.state.Stage = next
	if next == day15StageDone {
		m.state.Paused = false
		m.state.PauseReason = ""
		m.state.ExpectedAction = "Никаких действий"
	}
	m.touch("transition", prev, next, strings.TrimSpace(note))
	return nil
}

func (m *day15StateMachine) Render() string {
	return fmt.Sprintf(
		"stage=%s\nplan_approved=%t\ncurrent_step=%s\nexpected_action=%s\npaused=%t\npause_reason=%s",
		m.state.Stage,
		m.state.PlanApproved,
		day15EmptyFallback(m.state.CurrentStep, "(none)"),
		day15EmptyFallback(m.state.ExpectedAction, "(none)"),
		m.state.Paused,
		day15EmptyFallback(m.state.PauseReason, "(none)"),
	)
}

func (m *day15StateMachine) touch(event string, from, to day15Stage, note string) {
	now := time.Now().UTC().Format(time.RFC3339)
	m.state.UpdatedAt = now
	m.state.History = append(m.state.History, day15Transition{
		At:             now,
		Event:          event,
		From:           from,
		To:             to,
		CurrentStep:    m.state.CurrentStep,
		ExpectedAction: m.state.ExpectedAction,
		PlanApproved:   m.state.PlanApproved,
		Paused:         m.state.Paused,
		Note:           note,
	})
}

func runDay15Demo(stateFile, taskID string) (day15DemoResult, error) {
	machine := newDay15StateMachine(taskID)
	if err := saveDay15Machine(stateFile, machine); err != nil {
		return day15DemoResult{}, err
	}

	blockedWithoutApproval := false
	blockedJumpToDone := false
	blockedWhilePaused := false
	reloadResumePassed := false

	if err := machine.TransitionTo(day15StageExecution, "Начать разработку", "Открыть PR", "attempt before approval"); err != nil {
		blockedWithoutApproval = true
	}

	if err := machine.ApprovePlan("plan approved by owner"); err != nil {
		return day15DemoResult{}, err
	}

	if err := machine.TransitionTo(day15StageExecution, "Реализовать задачу", "Подготовить PR и тесты", "start execution"); err != nil {
		return day15DemoResult{}, err
	}

	if err := machine.TransitionTo(day15StageDone, "Закрыть задачу", "Никаких действий", "jump attempt"); err != nil {
		blockedJumpToDone = true
	}

	if err := machine.Pause("waiting_for_dev_window"); err != nil {
		return day15DemoResult{}, err
	}

	if err := machine.TransitionTo(day15StageValidation, "Прогнать проверки", "Подтвердить качество", "attempt while paused"); err != nil {
		blockedWhilePaused = true
	}
	if err := saveDay15Machine(stateFile, machine); err != nil {
		return day15DemoResult{}, err
	}

	loaded, err := loadOrCreateDay15Machine(stateFile, taskID)
	if err != nil {
		return day15DemoResult{}, err
	}
	reloadResumePassed = loaded.State().Paused &&
		loaded.State().Stage == day15StageExecution &&
		loaded.State().PlanApproved

	if err := loaded.Resume("dev window opened"); err != nil {
		return day15DemoResult{}, err
	}
	if err := loaded.TransitionTo(day15StageValidation, "Провести валидацию", "Подтвердить релиз readiness", "execution completed"); err != nil {
		return day15DemoResult{}, err
	}
	if err := loaded.TransitionTo(day15StageDone, "Закрыть задачу", "Никаких действий", "validation passed"); err != nil {
		return day15DemoResult{}, err
	}
	if err := saveDay15Machine(stateFile, loaded); err != nil {
		return day15DemoResult{}, err
	}

	return day15DemoResult{
		TaskID:                       normalizeTaskID(taskID),
		StateFile:                    stateFile,
		Transitions:                  append([]day15Transition(nil), loaded.State().History...),
		BlockedWithoutApproval:       blockedWithoutApproval,
		BlockedJumpToDone:            blockedJumpToDone,
		BlockedTransitionWhilePaused: blockedWhilePaused,
		ReloadResumePassed:           reloadResumePassed,
		FinalState:                   loaded.State(),
	}, nil
}

func runDay15Interactive(machine *day15StateMachine, stateFile string) error {
	fmt.Println("Day15 interactive mode. Type /exit to quit.")
	fmt.Println("Commands:")
	fmt.Println("  /state show")
	fmt.Println("  /state approve-plan [note]")
	fmt.Println("  /state pause [reason]")
	fmt.Println("  /state resume [note]")
	fmt.Println("  /state step <text>")
	fmt.Println("  /state expect <text>")
	fmt.Println("  /state next <execution|validation|done>|<step>|<expected_action>")
	fmt.Println("  /state history")

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
			handled, err := handleDay15StateCommand(machine, stateFile, line)
			if handled {
				if err != nil {
					fmt.Fprintf(os.Stderr, "state error: %v\n", err)
				}
				continue
			}
		}

		if machine.State().Paused {
			fmt.Printf("agent> Task paused at `%s`. Resume first using /state resume.\n\n", machine.State().Stage)
			continue
		}

		fmt.Printf("agent> stage=%s plan_approved=%t expected_action=%s\n\n",
			machine.State().Stage,
			machine.State().PlanApproved,
			day15EmptyFallback(machine.State().ExpectedAction, "(none)"),
		)
	}
}

func handleDay15StateCommand(machine *day15StateMachine, stateFile, line string) (bool, error) {
	fields := strings.Fields(strings.TrimSpace(line))
	if len(fields) == 0 || strings.ToLower(fields[0]) != "/state" {
		return false, nil
	}
	if len(fields) < 2 {
		return true, fmt.Errorf("usage: /state show|approve-plan|pause|resume|step|expect|next|history")
	}

	action := strings.ToLower(fields[1])
	switch action {
	case "show":
		fmt.Printf("state>\n%s\n\n", machine.Render())
	case "approve-plan":
		note := strings.TrimSpace(strings.TrimPrefix(line, "/state approve-plan"))
		if err := machine.ApprovePlan(note); err != nil {
			return true, err
		}
		fmt.Println("state> plan approved")
		fmt.Println()
	case "pause":
		reason := strings.TrimSpace(strings.TrimPrefix(line, "/state pause"))
		if err := machine.Pause(reason); err != nil {
			return true, err
		}
		fmt.Printf("state> paused at %s\n\n", machine.State().Stage)
	case "resume":
		note := strings.TrimSpace(strings.TrimPrefix(line, "/state resume"))
		if err := machine.Resume(note); err != nil {
			return true, err
		}
		fmt.Printf("state> resumed at %s\n\n", machine.State().Stage)
	case "step":
		step := strings.TrimSpace(strings.TrimPrefix(line, "/state step"))
		if step == "" {
			return true, fmt.Errorf("usage: /state step <text>")
		}
		machine.UpdateStep(step, "")
		fmt.Println("state> current_step updated")
		fmt.Println()
	case "expect":
		expected := strings.TrimSpace(strings.TrimPrefix(line, "/state expect"))
		if expected == "" {
			return true, fmt.Errorf("usage: /state expect <text>")
		}
		machine.UpdateStep("", expected)
		fmt.Println("state> expected_action updated")
		fmt.Println()
	case "next":
		payload := strings.TrimSpace(strings.TrimPrefix(line, "/state next"))
		parts := strings.Split(payload, "|")
		if len(parts) < 3 {
			return true, fmt.Errorf("usage: /state next <execution|validation|done>|<step>|<expected_action>")
		}
		next := normalizeDay15Stage(day15Stage(parts[0]))
		step := strings.TrimSpace(parts[1])
		expected := strings.TrimSpace(parts[2])
		if err := machine.TransitionTo(next, step, expected, "manual next"); err != nil {
			return true, err
		}
		fmt.Printf("state> transitioned to %s\n\n", machine.State().Stage)
	case "history":
		fmt.Println("state> transitions:")
		for i, item := range machine.State().History {
			fmt.Printf("%d) %s event=%s from=%s to=%s plan_approved=%t paused=%t note=%s\n",
				i+1, item.At, item.Event, item.From, item.To, item.PlanApproved, item.Paused, day15EmptyFallback(item.Note, "-"),
			)
		}
		fmt.Println()
	default:
		return true, fmt.Errorf("unknown /state action: %s", action)
	}

	if err := saveDay15Machine(stateFile, machine); err != nil {
		return true, err
	}
	return true, nil
}

func loadOrCreateDay15Machine(path, taskID string) (*day15StateMachine, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return newDay15StateMachine(taskID), nil
		}
		return nil, err
	}
	if len(raw) == 0 {
		return newDay15StateMachine(taskID), nil
	}

	var file day15StateFile
	if err := json.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("failed to parse state file: %w", err)
	}
	state := file.State
	if !isValidDay15Stage(state.Stage) {
		state.Stage = day15StagePlanning
	}
	if strings.TrimSpace(state.TaskID) == "" {
		state.TaskID = normalizeTaskID(taskID)
	}
	if state.History == nil {
		state.History = []day15Transition{}
	}
	return &day15StateMachine{state: state}, nil
}

func saveDay15Machine(path string, machine *day15StateMachine) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	payload := day15StateFile{
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

func printDay15DemoResult(result day15DemoResult) {
	fmt.Println("=== Day 15: Controlled State Transitions ===")
	fmt.Printf("task_id=%s stage=%s paused=%t plan_approved=%t\n",
		result.TaskID, result.FinalState.Stage, result.FinalState.Paused, result.FinalState.PlanApproved,
	)
	fmt.Printf("checks: blocked_without_approval=%t blocked_jump_to_done=%t blocked_while_paused=%t reload_resume=%t\n",
		result.BlockedWithoutApproval, result.BlockedJumpToDone, result.BlockedTransitionWhilePaused, result.ReloadResumePassed,
	)
	fmt.Printf("transitions=%d state_file=%s\n", len(result.Transitions), result.StateFile)
}

func writeDay15Report(path string, result day15DemoResult) error {
	var b strings.Builder
	b.WriteString("# Day 15 Results: Controlled State Transitions\n\n")
	b.WriteString(fmt.Sprintf("- task_id: `%s`\n", result.TaskID))
	b.WriteString(fmt.Sprintf("- state_file: `%s`\n", result.StateFile))
	b.WriteString(fmt.Sprintf("- blocked transition before plan approval: `%t`\n", result.BlockedWithoutApproval))
	b.WriteString(fmt.Sprintf("- blocked jump to done without validation: `%t`\n", result.BlockedJumpToDone))
	b.WriteString(fmt.Sprintf("- blocked transition while paused: `%t`\n", result.BlockedTransitionWhilePaused))
	b.WriteString(fmt.Sprintf("- continue after pause/reload: `%t`\n", result.ReloadResumePassed))
	b.WriteString(fmt.Sprintf("- final stage: `%s`\n", result.FinalState.Stage))
	b.WriteString(fmt.Sprintf("- final paused: `%t`\n", result.FinalState.Paused))
	b.WriteString(fmt.Sprintf("- plan approved: `%t`\n\n", result.FinalState.PlanApproved))

	b.WriteString("## Final State\n")
	b.WriteString("```text\n" + sanitizeCodeFences((&day15StateMachine{state: result.FinalState}).Render()) + "\n```\n\n")

	b.WriteString("## Transition Log\n")
	for _, item := range result.Transitions {
		b.WriteString(fmt.Sprintf("- `%s` event=`%s` from=`%s` to=`%s` approved=`%t` paused=`%t` note=`%s`\n",
			item.At, item.Event, item.From, item.To, item.PlanApproved, item.Paused, item.Note,
		))
	}
	b.WriteString("\nConclusion: lifecycle is controlled by explicit states and allowed transitions; invalid jumps are rejected, and work continues correctly after pause/resume and reload.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay15Usage() {
	fmt.Println("Usage: openrouter-cli day15 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -task-id string      Task ID")
	fmt.Println("  -state-file string   Path to task state JSON")
	fmt.Println("  -report string       Markdown report path")
	fmt.Println("  -interactive         Run interactive mode")
	fmt.Println("  -reset               Reset state file before run")
	fmt.Println("  -help                Show help")
}

func isValidDay15Stage(stage day15Stage) bool {
	switch stage {
	case day15StagePlanning, day15StageExecution, day15StageValidation, day15StageDone:
		return true
	default:
		return false
	}
}

func normalizeDay15Stage(stage day15Stage) day15Stage {
	switch strings.ToLower(strings.TrimSpace(string(stage))) {
	case "planning":
		return day15StagePlanning
	case "execution":
		return day15StageExecution
	case "validation":
		return day15StageValidation
	case "done":
		return day15StageDone
	default:
		return stage
	}
}

func canTransitionDay15(from, to day15Stage) bool {
	if from == to {
		return true
	}
	switch from {
	case day15StagePlanning:
		return to == day15StageExecution
	case day15StageExecution:
		return to == day15StageValidation
	case day15StageValidation:
		return to == day15StageDone
	case day15StageDone:
		return false
	default:
		return false
	}
}

func day15AllowedNext(stage day15Stage) []string {
	switch stage {
	case day15StagePlanning:
		return []string{"planning", "execution (after approve-plan)"}
	case day15StageExecution:
		return []string{"execution", "validation"}
	case day15StageValidation:
		return []string{"validation", "done"}
	case day15StageDone:
		return []string{"done"}
	default:
		return []string{}
	}
}

func day15EmptyFallback(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}
