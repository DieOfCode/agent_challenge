package day31mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

const (
	ToolGitBranch = "git_branch"
	ToolListFiles = "list_files"
	ToolGitDiff   = "git_diff"
)

const (
	defaultFilesLimit = 80
	maxFilesLimit     = 500
	defaultDiffBytes  = 4000
	maxDiffBytes      = 20000
)

type GitBranchResult struct {
	Branch string `json:"branch"`
	Dirty  bool   `json:"dirty"`
	Head   string `json:"head,omitempty"`
}

type ListFilesResult struct {
	Root      string   `json:"root"`
	Files     []string `json:"files"`
	Count     int      `json:"count"`
	Truncated bool     `json:"truncated"`
}

type GitDiffResult struct {
	Branch    string `json:"branch"`
	Diff      string `json:"diff"`
	Bytes     int    `json:"bytes"`
	Truncated bool   `json:"truncated"`
}

func NewServer(name, version, workspace string) *server.MCPServer {
	serverName := strings.TrimSpace(name)
	if serverName == "" {
		serverName = "day31-dev-assistant-mcp"
	}
	serverVersion := strings.TrimSpace(version)
	if serverVersion == "" {
		serverVersion = "1.0.0"
	}
	workspacePath := resolveWorkspace(workspace)

	mcpServer := server.NewMCPServer(
		serverName,
		serverVersion,
		server.WithToolCapabilities(true),
	)

	mcpServer.AddTool(
		mcp.NewTool(
			ToolGitBranch,
			mcp.WithDescription("Return current git branch and dirty flag for the workspace"),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			result, err := GetGitBranch(ctx, workspacePath)
			if err != nil {
				return mcp.NewToolResultErrorFromErr("failed to read git branch", err), nil
			}
			raw, err := json.Marshal(result)
			if err != nil {
				return mcp.NewToolResultErrorFromErr("failed to encode git branch response", err), nil
			}
			return mcp.NewToolResultStructured(result, string(raw)), nil
		},
	)

	mcpServer.AddTool(
		mcp.NewTool(
			ToolListFiles,
			mcp.WithDescription("List project files from workspace or subdirectory"),
			mcp.WithString("root", mcp.Description("Relative root path from workspace (default: .)")),
			mcp.WithNumber("limit", mcp.Description("Maximum number of files in result")),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			root := strings.TrimSpace(mcp.ParseString(request, "root", "."))
			limit := mcp.ParseInt(request, "limit", defaultFilesLimit)
			result, err := ListFiles(ctx, workspacePath, root, limit)
			if err != nil {
				return mcp.NewToolResultErrorFromErr("failed to list files", err), nil
			}
			raw, err := json.Marshal(result)
			if err != nil {
				return mcp.NewToolResultErrorFromErr("failed to encode list_files response", err), nil
			}
			return mcp.NewToolResultStructured(result, string(raw)), nil
		},
	)

	mcpServer.AddTool(
		mcp.NewTool(
			ToolGitDiff,
			mcp.WithDescription("Return git diff snippet for current workspace"),
			mcp.WithNumber("max_bytes", mcp.Description("Maximum diff bytes to return")),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			maxBytes := mcp.ParseInt(request, "max_bytes", defaultDiffBytes)
			result, err := GetGitDiff(ctx, workspacePath, maxBytes)
			if err != nil {
				return mcp.NewToolResultErrorFromErr("failed to read git diff", err), nil
			}
			raw, err := json.Marshal(result)
			if err != nil {
				return mcp.NewToolResultErrorFromErr("failed to encode git_diff response", err), nil
			}
			return mcp.NewToolResultStructured(result, string(raw)), nil
		},
	)

	return mcpServer
}

func GetGitBranch(ctx context.Context, workspace string) (GitBranchResult, error) {
	workspacePath := resolveWorkspace(workspace)

	branch, err := runGit(ctx, workspacePath, "rev-parse", "--abbrev-ref", "HEAD")
	if err != nil {
		return GitBranchResult{}, err
	}
	branch = strings.TrimSpace(branch)
	if branch == "" {
		return GitBranchResult{}, fmt.Errorf("git returned empty branch")
	}

	head, err := runGit(ctx, workspacePath, "rev-parse", "--short", "HEAD")
	if err != nil {
		head = ""
	}

	status, err := runGit(ctx, workspacePath, "status", "--porcelain")
	if err != nil {
		return GitBranchResult{}, err
	}

	return GitBranchResult{
		Branch: branch,
		Dirty:  strings.TrimSpace(status) != "",
		Head:   strings.TrimSpace(head),
	}, nil
}

func ListFiles(ctx context.Context, workspace, root string, limit int) (ListFilesResult, error) {
	_ = ctx

	workspacePath := resolveWorkspace(workspace)
	rootPath, err := normalizeSubPath(workspacePath, root)
	if err != nil {
		return ListFilesResult{}, err
	}
	if limit <= 0 {
		limit = defaultFilesLimit
	}
	if limit > maxFilesLimit {
		limit = maxFilesLimit
	}

	files := make([]string, 0, limit)
	truncated := false
	err = filepath.WalkDir(rootPath, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		name := d.Name()
		if d.IsDir() {
			if name == ".git" || name == "node_modules" || name == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}
		rel, relErr := filepath.Rel(workspacePath, path)
		if relErr != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		if strings.HasPrefix(rel, "../") || rel == ".." {
			return nil
		}
		if len(files) >= limit {
			truncated = true
			return fs.SkipAll
		}
		files = append(files, rel)
		return nil
	})
	if err != nil && err != fs.SkipAll {
		return ListFilesResult{}, fmt.Errorf("failed to walk files: %w", err)
	}

	sort.Strings(files)
	relRoot, _ := filepath.Rel(workspacePath, rootPath)
	relRoot = filepath.ToSlash(relRoot)
	if relRoot == "" {
		relRoot = "."
	}

	return ListFilesResult{
		Root:      relRoot,
		Files:     files,
		Count:     len(files),
		Truncated: truncated,
	}, nil
}

func GetGitDiff(ctx context.Context, workspace string, maxBytes int) (GitDiffResult, error) {
	workspacePath := resolveWorkspace(workspace)
	if maxBytes <= 0 {
		maxBytes = defaultDiffBytes
	}
	if maxBytes > maxDiffBytes {
		maxBytes = maxDiffBytes
	}

	branchInfo, err := GetGitBranch(ctx, workspacePath)
	if err != nil {
		return GitDiffResult{}, err
	}

	diffText, err := runGit(ctx, workspacePath, "diff", "--", ".")
	if err != nil {
		return GitDiffResult{}, err
	}
	diffText = strings.TrimSpace(diffText)
	if diffText == "" {
		return GitDiffResult{Branch: branchInfo.Branch, Diff: "(no local diff)", Bytes: 0, Truncated: false}, nil
	}

	truncated := false
	if len(diffText) > maxBytes {
		diffText = diffText[:maxBytes]
		truncated = true
	}

	return GitDiffResult{
		Branch:    branchInfo.Branch,
		Diff:      diffText,
		Bytes:     len(diffText),
		Truncated: truncated,
	}, nil
}

func runGit(parent context.Context, workspace string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(parent, 6*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "git", append([]string{"-C", workspace}, args...)...)
	out, err := cmd.CombinedOutput()
	text := strings.TrimSpace(string(out))
	if err != nil {
		if text == "" {
			text = err.Error()
		}
		return "", fmt.Errorf("git %s failed: %s", strings.Join(args, " "), text)
	}
	return text, nil
}

func resolveWorkspace(workspace string) string {
	workspace = strings.TrimSpace(workspace)
	if workspace == "" {
		workspace = "."
	}
	abs, err := filepath.Abs(workspace)
	if err != nil {
		return workspace
	}
	return abs
}

func normalizeSubPath(workspace, sub string) (string, error) {
	base := resolveWorkspace(workspace)
	sub = strings.TrimSpace(sub)
	if sub == "" {
		sub = "."
	}
	candidate := filepath.Clean(filepath.Join(base, sub))
	rel, err := filepath.Rel(base, candidate)
	if err != nil {
		return "", fmt.Errorf("failed to normalize path: %w", err)
	}
	rel = filepath.ToSlash(rel)
	if rel == ".." || strings.HasPrefix(rel, "../") {
		return "", fmt.Errorf("path escapes workspace")
	}
	info, err := os.Stat(candidate)
	if err != nil {
		return "", fmt.Errorf("failed to stat path: %w", err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("path is not a directory: %s", sub)
	}
	return candidate, nil
}
