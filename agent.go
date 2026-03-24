package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

// Agent is a separate entity that owns request/response interaction with the LLM.
type Agent interface {
	Reply(userInput string) (openRouterResult, error)
}

type LLMAgentConfig struct {
	APIKey       string
	Model        string
	MaxTokens    int
	Temperature  *float64
	SystemPrompt string
	Title        string
}

type LLMAgent struct {
	apiKey      string
	model       string
	maxTokens   int
	temperature *float64
	title       string
	history     []message
}

func NewLLMAgent(cfg LLMAgentConfig) *LLMAgent {
	history := make([]message, 0, 16)
	if strings.TrimSpace(cfg.SystemPrompt) != "" {
		history = append(history, message{
			Role:    "system",
			Content: strings.TrimSpace(cfg.SystemPrompt),
		})
	}

	return &LLMAgent{
		apiKey:      cfg.APIKey,
		model:       cfg.Model,
		maxTokens:   cfg.MaxTokens,
		temperature: cfg.Temperature,
		title:       cfg.Title,
		history:     history,
	}
}

func (a *LLMAgent) Reply(userInput string) (openRouterResult, error) {
	userInput = strings.TrimSpace(userInput)
	if userInput == "" {
		return openRouterResult{}, fmt.Errorf("empty user input")
	}

	requestMessages := append([]message(nil), a.history...)
	requestMessages = append(requestMessages, message{
		Role:    "user",
		Content: userInput,
	})

	result, err := callOpenRouterDetailed(
		a.apiKey,
		a.model,
		requestMessages,
		a.maxTokens,
		a.temperature,
		nil,
		a.title,
	)
	if err != nil {
		return openRouterResult{}, err
	}

	// Store conversation state inside the agent to keep it as a true chat entity.
	a.history = append(a.history,
		message{Role: "user", Content: userInput},
		message{Role: "assistant", Content: strings.TrimSpace(result.Answer)},
	)

	return result, nil
}

func runAgentCommand(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli agent", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	model := fs.String("model", getDefaultModel(), "OpenRouter model")
	prompt := fs.String("prompt", "", "Single user prompt (if empty, reads stdin unless -interactive)")
	systemPrompt := fs.String("system", "You are a helpful assistant.", "System prompt for the agent")
	maxTokens := fs.Int("max-tokens", 400, "Maximum response tokens")
	temperature := fs.Float64("temperature", 0.2, "Agent temperature")
	interactive := fs.Bool("interactive", false, "Interactive chat mode")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse agent flags: %w", err)
	}
	if *help {
		printAgentUsage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected agent arguments: %s", strings.Join(fs.Args(), " "))
	}

	t := *temperature
	agent := NewLLMAgent(LLMAgentConfig{
		APIKey:       getAPIKey(),
		Model:        *model,
		MaxTokens:    *maxTokens,
		Temperature:  &t,
		SystemPrompt: *systemPrompt,
		Title:        "encapsulated-agent-cli",
	})

	if *interactive {
		return runAgentInteractive(agent)
	}

	userPrompt := strings.TrimSpace(*prompt)
	if userPrompt == "" {
		input, err := io.ReadAll(os.Stdin)
		if err != nil {
			return fmt.Errorf("failed to read stdin: %w", err)
		}
		userPrompt = strings.TrimSpace(string(input))
	}
	if userPrompt == "" {
		return fmt.Errorf("empty prompt: use -prompt, -interactive, or pipe text to stdin")
	}

	result, err := agent.Reply(userPrompt)
	if err != nil {
		return err
	}
	fmt.Println(strings.TrimSpace(result.Answer))
	return nil
}

func runAgentInteractive(agent Agent) error {
	fmt.Println("Agent interactive mode. Type /exit to quit.")
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("you> ")
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				return fmt.Errorf("stdin scan failed: %w", err)
			}
			return nil
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch strings.ToLower(line) {
		case "/exit", "exit", "quit":
			return nil
		}

		result, err := agent.Reply(line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "agent error: %v\n", err)
			continue
		}

		fmt.Printf("agent> %s\n\n", strings.TrimSpace(result.Answer))
	}
}

func printAgentUsage() {
	fmt.Println("Usage: openrouter-cli agent [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -model string        OpenRouter model")
	fmt.Println("  -prompt string       Single prompt (if empty, reads stdin)")
	fmt.Println("  -system string       System prompt for the agent")
	fmt.Println("  -temperature float   Temperature")
	fmt.Println("  -max-tokens int      Maximum response tokens")
	fmt.Println("  -interactive         Interactive chat mode")
	fmt.Println("  -help                Show help")
}
