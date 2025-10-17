package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"agent_challenge/internal/agent"
	"agent_challenge/internal/openrouter"
)

func main() {
	reader := bufio.NewReader(os.Stdin)

	// Token
	token := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	if token == "" {
		secret, err := readSecret("Введите OpenRouter API токен: ")
		if err != nil {
			fmt.Printf("Не удалось прочитать токен: %v\n", err)
			return
		}
		token = strings.TrimSpace(secret)
		if token == "" {
			fmt.Println("Токен обязателен. Завершение.")
			return
		}
	}

	// Model selection
	model := selectModel(token, reader)
	if model == "" {
		fmt.Println("Модель не выбрана. Завершение.")
		return
	}

	// Select answer format
	format := selectFormat(reader)

	// System prompt (format-aware)
	sysPrompt := buildSystemPrompt(format)
	messages := []openrouter.ChatMessage{{Role: "system", Content: sysPrompt}}

	tools := agent.GetToolDefinitions()
	maxTokens := 512
	ctx := context.Background()

	fmt.Println("Готово. Введите сообщение (или 'exit' для выхода). Команды: /help, /format <text|markdown|json>")
	for {
		fmt.Print("You> ")
		line, _ := reader.ReadString('\n')
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.EqualFold(line, "exit") || strings.EqualFold(line, "quit") {
			fmt.Println("Пока!")
			return
		}

		// Commands handling
		if strings.HasPrefix(line, "/") {
			parts := strings.Fields(line)
			cmd := strings.ToLower(parts[0])
			switch cmd {
			case "/help":
				printHelp()
			case "/format":
				if len(parts) < 2 {
					fmt.Println("Укажите формат: /format text | /format markdown | /format json")
					break
				}
				newFmt := normalizeFormat(parts[1])
				if newFmt == "" {
					fmt.Println("Неизвестный формат. Доступно: text, markdown, json")
					break
				}
				format = newFmt
				// Обновляем системный промпт
				sysPrompt = buildSystemPrompt(format)
				messages = append(messages, openrouter.ChatMessage{Role: "system", Content: sysPrompt})
				fmt.Printf("Формат установлен: %s\n", format)
			case "/maxtokens", "/max":
				if len(parts) < 2 {
					fmt.Println("Укажите число: /max 256")
					break
				}
				if v, err := strconv.Atoi(parts[1]); err == nil && v > 0 {
					maxTokens = v
					fmt.Printf("max_tokens установлен: %d\n", maxTokens)
				} else {
					fmt.Println("Некорректное значение. Пример: /max 512")
				}
			default:
				fmt.Println("Неизвестная команда. Введите /help для справки.")
			}
			continue
		}

		messages = append(messages, openrouter.ChatMessage{Role: "user", Content: line})

		// Tool-calling loop (max 3 steps)
		var assistantOut string
		for step := 0; step < 3; step++ {
			var assistantMsg openrouter.ChatMessage
			stopSpin := startSpinner("Думаю…")
			req := openrouter.ChatCompletionRequest{
				Model:      model,
				Messages:   messages,
				Tools:      tools,
				ToolChoice: "auto",
				MaxTokens:  maxTokens,
			}
			// hint model to return JSON if format requires it
			if strings.HasPrefix(format, "json") {
				req.ResponseFormat = map[string]any{"type": "json_object"}
			}
			resp, err := openrouter.CreateChatCompletion(ctx, token, req)
			if err != nil {
				stopSpin()
				// Фолбэк: выбранная модель/провайдер не поддерживает инструменты
				errLower := strings.ToLower(err.Error())
				if strings.Contains(errLower, "support tool use") {
					fmt.Println("Предупреждение: модель не поддерживает инструменты. Продолжаю без tools…")
					stopSpin = startSpinner("Думаю…")
					req2 := openrouter.ChatCompletionRequest{Model: model, Messages: messages, MaxTokens: maxTokens}
					if strings.HasPrefix(format, "json") {
						req2.ResponseFormat = map[string]any{"type": "json_object"}
					}
					resp2, err2 := openrouter.CreateChatCompletion(ctx, token, req2)
					stopSpin()
					if err2 != nil || len(resp2.Choices) == 0 {
						fmt.Printf("Ошибка запроса: %v\n", err2)
						break
					}
					assistantOut = resp2.Choices[0].Message.Content
					messages = append(messages, resp2.Choices[0].Message)
					break
				}
				// Credit/max_tokens issue → reduce and retry once
				if strings.Contains(errLower, "requires more credits") || strings.Contains(errLower, "fewer max_tokens") {
					newMax := maxTokens / 2
					if newMax < 128 {
						newMax = 128
					}
					fmt.Printf("Недостаточно кредитов/слишком большой max_tokens. Понижаю до %d и повторяю…\n", newMax)
					maxTokens = newMax
					stopSpin = startSpinner("Думаю…")
					reqRetry := openrouter.ChatCompletionRequest{Model: model, Messages: messages, MaxTokens: maxTokens, Tools: tools, ToolChoice: "auto"}
					if strings.HasPrefix(format, "json") {
						reqRetry.ResponseFormat = map[string]any{"type": "json_object"}
					}
					respRetry, errRetry := openrouter.CreateChatCompletion(ctx, token, reqRetry)
					stopSpin()
					if errRetry != nil || len(respRetry.Choices) == 0 {
						fmt.Printf("Ошибка запроса после понижения max_tokens: %v\n", errRetry)
						break
					}
					assistantMsg = respRetry.Choices[0].Message
					messages = append(messages, assistantMsg)
				} else {
					fmt.Printf("Ошибка запроса: %v\n", err)
					break
				}
			} else {
				stopSpin()
				if len(resp.Choices) == 0 {
					fmt.Println("Пустой ответ модели")
					break
				}
				assistantMsg = resp.Choices[0].Message
				messages = append(messages, assistantMsg)
			}

			if len(assistantMsg.ToolCalls) == 0 {
				assistantOut = assistantMsg.Content
				break
			}

			for _, tc := range assistantMsg.ToolCalls {
				result := agent.ExecuteTool(tc)
				messages = append(messages, openrouter.ChatMessage{
					Role:       "tool",
					Content:    result,
					ToolCallID: tc.ID,
					Name:       tc.Function.Name,
				})
				// small pause to simulate tool latency
				time.Sleep(50 * time.Millisecond)
			}
		}

		if assistantOut == "" {
			// Try to get final answer after tools
			stopSpin := startSpinner("Думаю…")
			req := openrouter.ChatCompletionRequest{Model: model, Messages: messages, MaxTokens: maxTokens}
			if strings.HasPrefix(format, "json") {
				req.ResponseFormat = map[string]any{"type": "json_object"}
			}
			resp, err := openrouter.CreateChatCompletion(ctx, token, req)
			stopSpin()
			if err == nil && len(resp.Choices) > 0 {
				assistantOut = resp.Choices[0].Message.Content
			}
		}

		if assistantOut == "" {
			assistantOut = "(нет ответа)"
		}
		// If JSON expected, try to pretty print/validate
		if strings.HasPrefix(format, "json") {
			if pretty, ok := tryPrettyJSON(assistantOut); ok {
				assistantOut = pretty
			}
		}
		fmt.Printf("Agent> %s\n", assistantOut)
	}
}

func selectModel(token string, reader *bufio.Reader) string {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	models, err := openrouter.ListModels(ctx, token)
	if err != nil || len(models) == 0 {
		fmt.Println("Не удалось получить список моделей. Введите ID модели вручную (пример: openrouter/auto):")
		fmt.Print("Модель: ")
		line, _ := reader.ReadString('\n')
		return strings.TrimSpace(line)
	}

	// Разделим модели: бесплатные (по признаку ":free" в ID) и платные
	var freeIDs []string
	var paidIDs []string
	for _, m := range models {
		if strings.Contains(m.ID, ":free") {
			freeIDs = append(freeIDs, m.ID)
		} else {
			paidIDs = append(paidIDs, m.ID)
		}
	}

	// Возьмем до 15 из каждой категории
	limit := func(list []string, n int) []string {
		if len(list) > n {
			return list[:n]
		}
		return list
	}
	freeShow := limit(freeIDs, 15)
	paidShow := limit(paidIDs, 15)

	// Построим единую нумерацию
	fmt.Println("Доступные модели (15 бесплатных и 15 платных, если доступны):")
	idx := 1
	indexToID := make(map[string]string)
	if len(freeShow) > 0 {
		fmt.Println("Бесплатные:")
		for _, id := range freeShow {
			fmt.Printf("%2d) %s\n", idx, id)
			indexToID[fmt.Sprintf("%d", idx)] = id
			idx++
		}
	}
	if len(paidShow) > 0 {
		fmt.Println("Платные:")
		for _, id := range paidShow {
			fmt.Printf("%2d) %s\n", idx, id)
			indexToID[fmt.Sprintf("%d", idx)] = id
			idx++
		}
	}

	fmt.Println("Введите номер из списка или полный ID (Enter по умолчанию: openrouter/auto):")
	fmt.Print("Модель: ")
	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)
	if line == "" {
		return "openrouter/auto"
	}
	if id, ok := indexToID[line]; ok {
		return id
	}
	return line
}

func printHelp() {
	fmt.Println("Доступные команды:")
	fmt.Println("  /help                      — показать эту справку")
	fmt.Println("  /format <text|markdown|json> — сменить формат ответа")
	fmt.Println("  exit | quit                — выйти")
}

func normalizeFormat(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	switch s {
	case "text", "1":
		return "text"
	case "markdown", "md", "2":
		return "markdown"
	case "json", "json:qa_v1", "3":
		return "json:qa_v1"
	default:
		return ""
	}
}

func selectFormat(reader *bufio.Reader) string {
	fmt.Println("Выберите формат ответа: 1) text  2) markdown  3) json (по умолчанию: text)")
	fmt.Print("Формат: ")
	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(strings.ToLower(line))
	switch line {
	case "2", "markdown":
		return "markdown"
	case "3", "json", "json:qa_v1":
		return "json:qa_v1"
	default:
		return "text"
	}
}

func buildSystemPrompt(format string) string {
	base := "Ты — полезный ассистент. Отвечай по-русски. Используй инструменты, когда уместно."
	if strings.HasPrefix(format, "json") {
		schema := `{"type":"object","required":["answer"],"properties":{"answer":{"type":"string"},"citations":{"type":"array","items":{"type":"object","required":["url"],"properties":{"url":{"type":"string"},"title":{"type":"string"}}}},"used_tools":{"type":"array","items":{"type":"object","required":["name","result"],"properties":{"name":{"type":"string"},"arguments":{"type":"object"},"result":{"type":"string"}}}},"followups":{"type":"array","items":{"type":"string"}}}}`
		return base + " Всегда возвращай ТОЛЬКО валидный JSON по схеме qa_v1 без текста вокруг. Схема qa_v1: " + schema
	}
	if format == "markdown" {
		return base + " Отвечай в Markdown (заголовки, списки, ссылки)."
	}
	return base
}

func tryPrettyJSON(s string) (string, bool) {
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return s, false
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return s, false
	}
	return string(b), true
}

// readSecret reads a line from stdin with echo disabled (Unix/macOS only).
func readSecret(prompt string) (string, error) {
	fmt.Print(prompt)
	// save current tty state
	get := exec.Command("stty", "-g")
	get.Stdin = os.Stdin
	stateBytes, _ := get.Output()
	state := strings.TrimSpace(string(stateBytes))

	off := exec.Command("stty", "-echo")
	off.Stdin = os.Stdin
	_ = off.Run()
	defer func() {
		restore := exec.Command("stty", state)
		restore.Stdin = os.Stdin
		_ = restore.Run()
		fmt.Println()
	}()
	reader := bufio.NewReader(os.Stdin)
	line, err := reader.ReadString('\n')
	return strings.TrimSpace(line), err
}

// startSpinner prints a spinner with a message until the returned stop function is called.
func startSpinner(message string) func() {
	done := make(chan struct{})
	go func() {
		frames := []rune{'⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'}
		idx := 0
		for {
			select {
			case <-done:
				fmt.Print("\r")
				return
			default:
				fmt.Printf("\r%s %s", string(frames[idx]), message)
				idx = (idx + 1) % len(frames)
				time.Sleep(80 * time.Millisecond)
			}
		}
	}()
	return func() {
		close(done)
		fmt.Print("\r\x1b[2K") // clear line
	}
}
