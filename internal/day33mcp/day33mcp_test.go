package day33mcp

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

func TestLoadStoreDefaults(t *testing.T) {
	store, err := LoadStore("", "")
	if err != nil {
		t.Fatalf("LoadStore() error = %v", err)
	}
	if len(store.UsersByID) == 0 {
		t.Fatalf("expected default users")
	}
	if len(store.TicketsByID) == 0 {
		t.Fatalf("expected default tickets")
	}
	if _, ok := store.UsersByID[101]; !ok {
		t.Fatalf("expected user 101 in defaults")
	}
	if _, ok := store.TicketsByID[5001]; !ok {
		t.Fatalf("expected ticket 5001 in defaults")
	}
}

func TestLoadStoreFromFiles(t *testing.T) {
	tmp := t.TempDir()
	usersPath := filepath.Join(tmp, "users.json")
	ticketsPath := filepath.Join(tmp, "tickets.json")

	usersJSON := `{"users":[{"id":1,"name":"U","email":"u@example.com","plan":"free","locale":"ru-KZ","auth_provider":"email","status":"active","tags":["a"]}]}`
	ticketsJSON := `{"tickets":[{"id":11,"user_id":1,"subject":"S","status":"open","priority":"low","category":"auth","summary":"sm","last_error":"x","last_error_code":"E","last_login_at":"2026-01-01T00:00:00Z","attempts":1,"affected_feature":"login","labels":["l"]}]}`

	if err := os.WriteFile(usersPath, []byte(usersJSON), 0o644); err != nil {
		t.Fatalf("write users: %v", err)
	}
	if err := os.WriteFile(ticketsPath, []byte(ticketsJSON), 0o644); err != nil {
		t.Fatalf("write tickets: %v", err)
	}

	store, err := LoadStore(usersPath, ticketsPath)
	if err != nil {
		t.Fatalf("LoadStore() error = %v", err)
	}
	if len(store.UsersByID) != 1 || len(store.TicketsByID) != 1 {
		t.Fatalf("unexpected store size: users=%d tickets=%d", len(store.UsersByID), len(store.TicketsByID))
	}
	if store.TicketsByID[11].UserID != 1 {
		t.Fatalf("unexpected ticket user mapping")
	}
}

func TestServerToolsInProcess(t *testing.T) {
	server := NewServer("test-day33", "1.0.0", "", "")
	mcpClient, err := client.NewInProcessClient(server)
	if err != nil {
		t.Fatalf("NewInProcessClient error = %v", err)
	}
	defer mcpClient.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := mcpClient.Start(ctx); err != nil {
		t.Fatalf("Start error = %v", err)
	}
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{Name: "test-day33-client", Version: "1.0.0"}
	if _, err := mcpClient.Initialize(ctx, initReq); err != nil {
		t.Fatalf("Initialize error = %v", err)
	}

	userRes, err := mcpClient.CallTool(ctx, mcp.CallToolRequest{Params: mcp.CallToolParams{Name: ToolGetUserProfile, Arguments: map[string]any{"user_id": 101}}})
	if err != nil {
		t.Fatalf("CallTool get_user_profile error = %v", err)
	}
	user, err := ParseUserFromCallToolResult(userRes)
	if err != nil {
		t.Fatalf("ParseUserFromCallToolResult error = %v", err)
	}
	if user.ID != 101 {
		t.Fatalf("unexpected user id: %d", user.ID)
	}

	ticketRes, err := mcpClient.CallTool(ctx, mcp.CallToolRequest{Params: mcp.CallToolParams{Name: ToolGetTicket, Arguments: map[string]any{"ticket_id": 5001}}})
	if err != nil {
		t.Fatalf("CallTool get_ticket error = %v", err)
	}
	ticket, err := ParseTicketFromCallToolResult(ticketRes)
	if err != nil {
		t.Fatalf("ParseTicketFromCallToolResult error = %v", err)
	}
	if ticket.ID != 5001 {
		t.Fatalf("unexpected ticket id: %d", ticket.ID)
	}

	listRes, err := mcpClient.CallTool(ctx, mcp.CallToolRequest{Params: mcp.CallToolParams{Name: ToolListUserTickets, Arguments: map[string]any{"user_id": 101, "limit": 2}}})
	if err != nil {
		t.Fatalf("CallTool list_user_tickets error = %v", err)
	}
	list, err := ParseTicketListFromCallToolResult(listRes)
	if err != nil {
		t.Fatalf("ParseTicketListFromCallToolResult error = %v", err)
	}
	if list.UserID != 101 || list.Count == 0 {
		t.Fatalf("unexpected list result: %+v", list)
	}

	raw, _ := json.Marshal(list)
	if len(raw) == 0 {
		t.Fatalf("expected serializable list result")
	}
}
