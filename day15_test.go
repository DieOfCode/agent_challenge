package main

import (
	"path/filepath"
	"testing"
)

func TestDay15TransitionRequiresPlanApproval(t *testing.T) {
	m := newDay15StateMachine("task-a")

	if err := m.TransitionTo(day15StageExecution, "", "", "try before approve"); err == nil {
		t.Fatalf("expected transition error without plan approval")
	}

	if err := m.ApprovePlan("approved"); err != nil {
		t.Fatalf("ApprovePlan failed: %v", err)
	}
	if err := m.TransitionTo(day15StageExecution, "", "", "start execution"); err != nil {
		t.Fatalf("expected transition to execution after approval, got: %v", err)
	}
}

func TestDay15CannotJumpToDone(t *testing.T) {
	m := newDay15StateMachine("task-b")
	if err := m.ApprovePlan("approved"); err != nil {
		t.Fatalf("ApprovePlan failed: %v", err)
	}
	if err := m.TransitionTo(day15StageExecution, "", "", "to execution"); err != nil {
		t.Fatalf("transition to execution failed: %v", err)
	}
	if err := m.TransitionTo(day15StageDone, "", "", "jump to done"); err == nil {
		t.Fatalf("expected error when jumping execution -> done")
	}
}

func TestDay15PauseBlocksTransitionsUntilResume(t *testing.T) {
	m := newDay15StateMachine("task-c")
	if err := m.ApprovePlan("approved"); err != nil {
		t.Fatalf("ApprovePlan failed: %v", err)
	}
	if err := m.TransitionTo(day15StageExecution, "", "", "to execution"); err != nil {
		t.Fatalf("transition to execution failed: %v", err)
	}
	if err := m.Pause("wait"); err != nil {
		t.Fatalf("Pause failed: %v", err)
	}
	if err := m.TransitionTo(day15StageValidation, "", "", "while paused"); err == nil {
		t.Fatalf("expected error for transition while paused")
	}
	if err := m.Resume("continue"); err != nil {
		t.Fatalf("Resume failed: %v", err)
	}
	if err := m.TransitionTo(day15StageValidation, "", "", "after resume"); err != nil {
		t.Fatalf("expected transition after resume, got: %v", err)
	}
}

func TestDay15SaveAndLoadState(t *testing.T) {
	tmpDir := t.TempDir()
	statePath := filepath.Join(tmpDir, "day15-state.json")

	m := newDay15StateMachine("task-d")
	if err := m.ApprovePlan("approved"); err != nil {
		t.Fatalf("ApprovePlan failed: %v", err)
	}
	if err := m.TransitionTo(day15StageExecution, "Implement", "Prepare PR", "execution"); err != nil {
		t.Fatalf("transition failed: %v", err)
	}
	if err := m.Pause("window"); err != nil {
		t.Fatalf("Pause failed: %v", err)
	}
	if err := saveDay15Machine(statePath, m); err != nil {
		t.Fatalf("saveDay15Machine failed: %v", err)
	}

	loaded, err := loadOrCreateDay15Machine(statePath, "task-d")
	if err != nil {
		t.Fatalf("loadOrCreateDay15Machine failed: %v", err)
	}
	s := loaded.State()
	if s.Stage != day15StageExecution {
		t.Fatalf("expected stage execution, got %s", s.Stage)
	}
	if !s.PlanApproved {
		t.Fatalf("expected plan approved state")
	}
	if !s.Paused {
		t.Fatalf("expected paused state")
	}
}
