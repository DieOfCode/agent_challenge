package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadDay32MetaFromEvent(t *testing.T) {
	tmp := t.TempDir()
	eventPath := filepath.Join(tmp, "event.json")
	payload := `{
  "pull_request": {
    "number": 42,
    "title": "Add day32 review",
    "html_url": "https://example.com/pr/42",
    "base": {"ref": "main", "sha": "aaa111"},
    "head": {"ref": "feature/day32", "sha": "bbb222"}
  }
}`
	if err := os.WriteFile(eventPath, []byte(payload), 0o644); err != nil {
		t.Fatalf("failed to write event file: %v", err)
	}

	meta, err := loadDay32MetaFromEvent(eventPath)
	if err != nil {
		t.Fatalf("loadDay32MetaFromEvent() error = %v", err)
	}
	if meta.Number != 42 || meta.BaseSHA != "aaa111" || meta.HeadSHA != "bbb222" {
		t.Fatalf("unexpected meta: %+v", meta)
	}
}

func TestDay32CollectDiffAndFiles(t *testing.T) {
	repo := day32CreateTempRepo(t)

	if err := os.WriteFile(filepath.Join(repo, "sample.txt"), []byte("line1\nline2\n"), 0o644); err != nil {
		t.Fatalf("failed to modify file: %v", err)
	}

	diffText, files, err := day32CollectDiffAndFiles(repo, "HEAD~1", "HEAD", "", "")
	if err != nil {
		t.Fatalf("day32CollectDiffAndFiles() error = %v", err)
	}
	if strings.TrimSpace(diffText) == "" {
		t.Fatalf("expected non-empty diff")
	}
	if len(files) == 0 {
		t.Fatalf("expected changed files")
	}
}

func TestDay32BuildHeuristicReview(t *testing.T) {
	diff := `diff --git a/x.go b/x.go
index 111..222 100644
--- a/x.go
+++ b/x.go
@@ -1,2 +1,4 @@
-if err != nil { return err }
+panic("boom")
+fmt.Println("debug")
+// TODO remove`
	text := day32BuildHeuristicReview(day32PRMeta{Number: 9, Title: "T"}, "base", "head", []string{"x.go"}, diff, nil)
	if !strings.Contains(text, "## Потенциальные баги") {
		t.Fatalf("expected bugs section")
	}
	if !strings.Contains(text, "panic") {
		t.Fatalf("expected panic finding")
	}
	if !strings.Contains(text, "## Рекомендации") {
		t.Fatalf("expected recommendations section")
	}
}

func day32CreateTempRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	day32RunCmd(t, repo, "git", "init")
	day32RunCmd(t, repo, "git", "config", "user.email", "day32@example.com")
	day32RunCmd(t, repo, "git", "config", "user.name", "Day32 Bot")
	if err := os.WriteFile(filepath.Join(repo, "sample.txt"), []byte("line1\n"), 0o644); err != nil {
		t.Fatalf("failed to write sample file: %v", err)
	}
	day32RunCmd(t, repo, "git", "add", "sample.txt")
	day32RunCmd(t, repo, "git", "commit", "-m", "init")

	if err := os.WriteFile(filepath.Join(repo, "sample.txt"), []byte("line1\nline-changed\n"), 0o644); err != nil {
		t.Fatalf("failed to update sample file: %v", err)
	}
	day32RunCmd(t, repo, "git", "add", "sample.txt")
	day32RunCmd(t, repo, "git", "commit", "-m", "update")
	return repo
}

func day32RunCmd(t *testing.T, dir, name string, args ...string) {
	t.Helper()
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("command failed: %s %s\nerr=%v\nout=%s", name, strings.Join(args, " "), err, string(out))
	}
}
