package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
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

	// System prompt
	messages := []openrouter.ChatMessage{
		{Role: "system", Content: "Ты — полезный ассистент. Используй инструменты, когда это уместно. Отвечай по-русски."},
	}

	tools := agent.GetToolDefinitions()
	ctx := context.Background()

	fmt.Println("Готово. Введите сообщение (или 'exit' для выхода).")
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

		messages = append(messages, openrouter.ChatMessage{Role: "user", Content: line})

		// Tool-calling loop (max 3 steps)
		var assistantOut string
		for step := 0; step < 3; step++ {
			stopSpin := startSpinner("Думаю…")
			resp, err := openrouter.CreateChatCompletion(ctx, token, openrouter.ChatCompletionRequest{
				Model:      model,
				Messages:   messages,
				Tools:      tools,
				ToolChoice: "auto",
			})
			if err != nil {
				stopSpin()
				// Фолбэк: выбранная модель/провайдер не поддерживает инструменты
				if strings.Contains(strings.ToLower(err.Error()), "support tool use") {
					fmt.Println("Предупреждение: модель не поддерживает инструменты. Продолжаю без tools…")
					stopSpin = startSpinner("Думаю…")
					resp2, err2 := openrouter.CreateChatCompletion(ctx, token, openrouter.ChatCompletionRequest{
						Model:    model,
						Messages: messages,
					})
					stopSpin()
					if err2 != nil || len(resp2.Choices) == 0 {
						fmt.Printf("Ошибка запроса: %v\n", err2)
						break
					}
					assistantOut = resp2.Choices[0].Message.Content
					messages = append(messages, resp2.Choices[0].Message)
					break
				}
				fmt.Printf("Ошибка запроса: %v\n", err)
				break
			}
			stopSpin()
			if len(resp.Choices) == 0 {
				fmt.Println("Пустой ответ модели")
				break
			}
			assistantMsg := resp.Choices[0].Message
			messages = append(messages, assistantMsg)

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
			resp, err := openrouter.CreateChatCompletion(ctx, token, openrouter.ChatCompletionRequest{
				Model:    model,
				Messages: messages,
			})
			stopSpin()
			if err == nil && len(resp.Choices) > 0 {
				assistantOut = resp.Choices[0].Message.Content
			}
		}

		if assistantOut == "" {
			assistantOut = "(нет ответа)"
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
