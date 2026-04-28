package day31mcp

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestGetGitBranchAndDirty(t *testing.T) {
	repo := createTempGitRepo(t)

	ctx := context.Background()
	branch, err := GetGitBranch(ctx, repo)
	if err != nil {
		t.Fatalf("GetGitBranch() error = %v", err)
	}
	if strings.TrimSpace(branch.Branch) == "" {
		t.Fatalf("expected non-empty branch")
	}
	if branch.Dirty {
		t.Fatalf("expected clean repo after initial commit")
	}

	filePath := filepath.Join(repo, "README.md")
	if err := os.WriteFile(filePath, []byte("changed\n"), 0o644); err != nil {
		t.Fatalf("failed to modify file: %v", err)
	}

	branchAfter, err := GetGitBranch(ctx, repo)
	if err != nil {
		t.Fatalf("GetGitBranch() dirty error = %v", err)
	}
	if !branchAfter.Dirty {
		t.Fatalf("expected dirty repo after file modification")
	}
}

func TestListFilesAndDiff(t *testing.T) {
	repo := createTempGitRepo(t)
	ctx := context.Background()

	if err := os.MkdirAll(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatalf("failed to create docs dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repo, "docs", "one.md"), []byte("a"), 0o644); err != nil {
		t.Fatalf("failed to write docs/one.md: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repo, "docs", "two.md"), []byte("b"), 0o644); err != nil {
		t.Fatalf("failed to write docs/two.md: %v", err)
	}

	files, err := ListFiles(ctx, repo, ".", 2)
	if err != nil {
		t.Fatalf("ListFiles() error = %v", err)
	}
	if files.Count == 0 {
		t.Fatalf("expected at least one file")
	}
	if files.Count > 2 {
		t.Fatalf("expected limited file count <= 2, got %d", files.Count)
	}

	if err := os.WriteFile(filepath.Join(repo, "README.md"), []byte("changed\n"), 0o644); err != nil {
		t.Fatalf("failed to modify README.md: %v", err)
	}
	diff, err := GetGitDiff(ctx, repo, 2048)
	if err != nil {
		t.Fatalf("GetGitDiff() error = %v", err)
	}
	if strings.TrimSpace(diff.Diff) == "" || strings.Contains(diff.Diff, "(no local diff)") {
		t.Fatalf("expected non-empty git diff")
	}
}

func createTempGitRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	runCmd(t, repo, "git", "init")
	runCmd(t, repo, "git", "config", "user.email", "day31@example.com")
	runCmd(t, repo, "git", "config", "user.name", "Day31 Bot")
	if err := os.WriteFile(filepath.Join(repo, "README.md"), []byte("initial\n"), 0o644); err != nil {
		t.Fatalf("failed to write README.md: %v", err)
	}
	runCmd(t, repo, "git", "add", "README.md")
	runCmd(t, repo, "git", "commit", "-m", "initial")
	return repo
}

func runCmd(t *testing.T, dir string, name string, args ...string) {
	t.Helper()
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("command failed: %s %s\nerr=%v\nout=%s", name, strings.Join(args, " "), err, string(out))
	}
}
