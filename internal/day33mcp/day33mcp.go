package day33mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

const (
	ToolGetUserProfile  = "get_user_profile"
	ToolGetTicket       = "get_ticket"
	ToolListUserTickets = "list_user_tickets"
)

type UserProfile struct {
	ID           int      `json:"id"`
	Name         string   `json:"name"`
	Email        string   `json:"email"`
	Plan         string   `json:"plan"`
	Locale       string   `json:"locale"`
	AuthProvider string   `json:"auth_provider"`
	Status       string   `json:"status"`
	Tags         []string `json:"tags"`
}

type SupportTicket struct {
	ID              int      `json:"id"`
	UserID          int      `json:"user_id"`
	Subject         string   `json:"subject"`
	Status          string   `json:"status"`
	Priority        string   `json:"priority"`
	Category        string   `json:"category"`
	Summary         string   `json:"summary"`
	LastError       string   `json:"last_error"`
	LastErrorCode   string   `json:"last_error_code"`
	LastLoginAt     string   `json:"last_login_at"`
	Attempts        int      `json:"attempts"`
	AffectedFeature string   `json:"affected_feature"`
	Labels          []string `json:"labels"`
}

type UserTicketList struct {
	UserID  int             `json:"user_id"`
	Count   int             `json:"count"`
	Tickets []SupportTicket `json:"tickets"`
}

type supportData struct {
	Users   []UserProfile   `json:"users"`
	Tickets []SupportTicket `json:"tickets"`
}

type Store struct {
	UsersByID       map[int]UserProfile
	TicketsByID     map[int]SupportTicket
	TicketsByUserID map[int][]SupportTicket
}

func NewServer(name, version, usersFile, ticketsFile string) *server.MCPServer {
	serverName := strings.TrimSpace(name)
	if serverName == "" {
		serverName = "day33-support-mcp-server"
	}
	serverVersion := strings.TrimSpace(version)
	if serverVersion == "" {
		serverVersion = "1.0.0"
	}

	store, loadErr := LoadStore(usersFile, ticketsFile)

	mcpServer := server.NewMCPServer(
		serverName,
		serverVersion,
		server.WithToolCapabilities(true),
	)

	mcpServer.AddTool(
		mcp.NewTool(
			ToolGetUserProfile,
			mcp.WithDescription("Get user profile by user_id"),
			mcp.WithNumber("user_id", mcp.Description("User ID"), mcp.Required()),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			if loadErr != nil {
				return mcp.NewToolResultErrorFromErr("failed to load support data", loadErr), nil
			}
			userID := mcp.ParseInt(request, "user_id", 0)
			if userID <= 0 {
				return mcp.NewToolResultError("user_id must be positive"), nil
			}
			user, ok := store.UsersByID[userID]
			if !ok {
				return mcp.NewToolResultError(fmt.Sprintf("user %d not found", userID)), nil
			}
			raw, err := json.Marshal(user)
			if err != nil {
				return mcp.NewToolResultErrorFromErr("failed to encode user", err), nil
			}
			return mcp.NewToolResultStructured(user, string(raw)), nil
		},
	)

	mcpServer.AddTool(
		mcp.NewTool(
			ToolGetTicket,
			mcp.WithDescription("Get support ticket by ticket_id"),
			mcp.WithNumber("ticket_id", mcp.Description("Ticket ID"), mcp.Required()),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			if loadErr != nil {
				return mcp.NewToolResultErrorFromErr("failed to load support data", loadErr), nil
			}
			ticketID := mcp.ParseInt(request, "ticket_id", 0)
			if ticketID <= 0 {
				return mcp.NewToolResultError("ticket_id must be positive"), nil
			}
			ticket, ok := store.TicketsByID[ticketID]
			if !ok {
				return mcp.NewToolResultError(fmt.Sprintf("ticket %d not found", ticketID)), nil
			}
			raw, err := json.Marshal(ticket)
			if err != nil {
				return mcp.NewToolResultErrorFromErr("failed to encode ticket", err), nil
			}
			return mcp.NewToolResultStructured(ticket, string(raw)), nil
		},
	)

	mcpServer.AddTool(
		mcp.NewTool(
			ToolListUserTickets,
			mcp.WithDescription("List tickets for a user"),
			mcp.WithNumber("user_id", mcp.Description("User ID"), mcp.Required()),
			mcp.WithNumber("limit", mcp.Description("Max number of tickets")),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			if loadErr != nil {
				return mcp.NewToolResultErrorFromErr("failed to load support data", loadErr), nil
			}
			userID := mcp.ParseInt(request, "user_id", 0)
			limit := mcp.ParseInt(request, "limit", 5)
			if userID <= 0 {
				return mcp.NewToolResultError("user_id must be positive"), nil
			}
			if limit <= 0 {
				limit = 5
			}
			if limit > 50 {
				limit = 50
			}
			tickets := append([]SupportTicket(nil), store.TicketsByUserID[userID]...)
			if len(tickets) > limit {
				tickets = tickets[:limit]
			}
			result := UserTicketList{UserID: userID, Count: len(tickets), Tickets: tickets}
			raw, err := json.Marshal(result)
			if err != nil {
				return mcp.NewToolResultErrorFromErr("failed to encode ticket list", err), nil
			}
			return mcp.NewToolResultStructured(result, string(raw)), nil
		},
	)

	return mcpServer
}

func LoadStore(usersFile, ticketsFile string) (Store, error) {
	users, err := loadUsers(usersFile)
	if err != nil {
		return Store{}, err
	}
	tickets, err := loadTickets(ticketsFile)
	if err != nil {
		return Store{}, err
	}

	store := Store{
		UsersByID:       make(map[int]UserProfile, len(users)),
		TicketsByID:     make(map[int]SupportTicket, len(tickets)),
		TicketsByUserID: make(map[int][]SupportTicket),
	}
	for _, user := range users {
		if user.ID <= 0 {
			continue
		}
		store.UsersByID[user.ID] = user
	}
	for _, ticket := range tickets {
		if ticket.ID <= 0 {
			continue
		}
		store.TicketsByID[ticket.ID] = ticket
		store.TicketsByUserID[ticket.UserID] = append(store.TicketsByUserID[ticket.UserID], ticket)
	}

	for userID := range store.TicketsByUserID {
		sort.Slice(store.TicketsByUserID[userID], func(i, j int) bool {
			left := store.TicketsByUserID[userID][i]
			right := store.TicketsByUserID[userID][j]
			if left.Priority == right.Priority {
				return left.ID > right.ID
			}
			return priorityRank(left.Priority) < priorityRank(right.Priority)
		})
	}

	return store, nil
}

func loadUsers(path string) ([]UserProfile, error) {
	raw, err := readDataFileOrDefault(path, defaultUsersJSON)
	if err != nil {
		return nil, err
	}
	var payload struct {
		Users []UserProfile `json:"users"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, fmt.Errorf("failed to parse users JSON: %w", err)
	}
	return payload.Users, nil
}

func loadTickets(path string) ([]SupportTicket, error) {
	raw, err := readDataFileOrDefault(path, defaultTicketsJSON)
	if err != nil {
		return nil, err
	}
	var payload struct {
		Tickets []SupportTicket `json:"tickets"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, fmt.Errorf("failed to parse tickets JSON: %w", err)
	}
	return payload.Tickets, nil
}

func readDataFileOrDefault(path string, fallback string) ([]byte, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return []byte(fallback), nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s: %w", path, err)
	}
	return raw, nil
}

func priorityRank(priority string) int {
	switch strings.ToLower(strings.TrimSpace(priority)) {
	case "critical":
		return 0
	case "high":
		return 1
	case "medium":
		return 2
	default:
		return 3
	}
}

func ParseUserFromCallToolResult(result *mcp.CallToolResult) (UserProfile, error) {
	if result == nil {
		return UserProfile{}, fmt.Errorf("nil tool result")
	}
	if result.StructuredContent != nil {
		raw, err := json.Marshal(result.StructuredContent)
		if err == nil {
			var user UserProfile
			if json.Unmarshal(raw, &user) == nil && user.ID > 0 {
				return user, nil
			}
		}
	}
	for _, content := range result.Content {
		text, ok := content.(mcp.TextContent)
		if !ok {
			continue
		}
		var user UserProfile
		if json.Unmarshal([]byte(strings.TrimSpace(text.Text)), &user) == nil && user.ID > 0 {
			return user, nil
		}
	}
	return UserProfile{}, fmt.Errorf("failed to parse user from tool result")
}

func ParseTicketFromCallToolResult(result *mcp.CallToolResult) (SupportTicket, error) {
	if result == nil {
		return SupportTicket{}, fmt.Errorf("nil tool result")
	}
	if result.StructuredContent != nil {
		raw, err := json.Marshal(result.StructuredContent)
		if err == nil {
			var ticket SupportTicket
			if json.Unmarshal(raw, &ticket) == nil && ticket.ID > 0 {
				return ticket, nil
			}
		}
	}
	for _, content := range result.Content {
		text, ok := content.(mcp.TextContent)
		if !ok {
			continue
		}
		var ticket SupportTicket
		if json.Unmarshal([]byte(strings.TrimSpace(text.Text)), &ticket) == nil && ticket.ID > 0 {
			return ticket, nil
		}
	}
	return SupportTicket{}, fmt.Errorf("failed to parse ticket from tool result")
}

func ParseTicketListFromCallToolResult(result *mcp.CallToolResult) (UserTicketList, error) {
	if result == nil {
		return UserTicketList{}, fmt.Errorf("nil tool result")
	}
	if result.StructuredContent != nil {
		raw, err := json.Marshal(result.StructuredContent)
		if err == nil {
			var out UserTicketList
			if json.Unmarshal(raw, &out) == nil && out.UserID > 0 {
				return out, nil
			}
		}
	}
	for _, content := range result.Content {
		text, ok := content.(mcp.TextContent)
		if !ok {
			continue
		}
		var out UserTicketList
		if json.Unmarshal([]byte(strings.TrimSpace(text.Text)), &out) == nil && out.UserID > 0 {
			return out, nil
		}
	}
	return UserTicketList{}, fmt.Errorf("failed to parse ticket list from tool result")
}

const defaultUsersJSON = `{
  "users": [
    {
      "id": 101,
      "name": "Ivan Sokolov",
      "email": "ivan@example.com",
      "plan": "pro",
      "locale": "ru-KZ",
      "auth_provider": "email_password",
      "status": "active",
      "tags": ["beta", "mobile"]
    },
    {
      "id": 102,
      "name": "Aigerim N.",
      "email": "aigerim@example.com",
      "plan": "free",
      "locale": "ru-KZ",
      "auth_provider": "google",
      "status": "active",
      "tags": ["new_user"]
    },
    {
      "id": 103,
      "name": "John Doe",
      "email": "john@example.com",
      "plan": "team",
      "locale": "en-US",
      "auth_provider": "sso",
      "status": "active",
      "tags": ["enterprise"]
    }
  ]
}`

const defaultTicketsJSON = `{
  "tickets": [
    {
      "id": 5001,
      "user_id": 101,
      "subject": "Не работает авторизация",
      "status": "open",
      "priority": "high",
      "category": "auth",
      "summary": "Пользователь не может войти после сброса пароля",
      "last_error": "invalid_grant",
      "last_error_code": "AUTH_401",
      "last_login_at": "2026-04-25T08:11:00Z",
      "attempts": 6,
      "affected_feature": "login",
      "labels": ["login", "password_reset"]
    },
    {
      "id": 5002,
      "user_id": 101,
      "subject": "2FA код не приходит",
      "status": "pending",
      "priority": "medium",
      "category": "auth",
      "summary": "SMS задерживается более 5 минут",
      "last_error": "sms_delivery_timeout",
      "last_error_code": "AUTH_2FA_TIMEOUT",
      "last_login_at": "2026-04-24T19:04:00Z",
      "attempts": 2,
      "affected_feature": "2fa",
      "labels": ["2fa", "sms"]
    },
    {
      "id": 5003,
      "user_id": 102,
      "subject": "Не открывается дашборд",
      "status": "open",
      "priority": "medium",
      "category": "ui",
      "summary": "После логина белый экран",
      "last_error": "chunk_load_error",
      "last_error_code": "UI_3001",
      "last_login_at": "2026-04-28T07:02:00Z",
      "attempts": 3,
      "affected_feature": "dashboard",
      "labels": ["frontend", "cache"]
    },
    {
      "id": 5004,
      "user_id": 103,
      "subject": "SSO не пускает пользователей",
      "status": "open",
      "priority": "critical",
      "category": "auth",
      "summary": "Ошибка подписи токена при SSO-входе",
      "last_error": "jwt_signature_invalid",
      "last_error_code": "SSO_403",
      "last_login_at": "2026-04-29T09:43:00Z",
      "attempts": 11,
      "affected_feature": "sso",
      "labels": ["sso", "enterprise", "blocking"]
    }
  ]
}`
