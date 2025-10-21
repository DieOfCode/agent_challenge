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

	// TZ mode controls
	tzMode := false
	tzEndMarker := "END_OF_TZ"
	nextUseStop := false
	lastAnswer := ""

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
		ranCommand := false
		runNow := false
		if strings.HasPrefix(line, "/") {
			parts := strings.Fields(line)
			cmd := strings.ToLower(parts[0])
			ranCommand = true
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
			case "/save":
				if lastAnswer == "" {
					fmt.Println("Нет данных для сохранения. Сначала получите ответ агента.")
					break
				}
				var path string
				if len(parts) >= 2 {
					path = parts[1]
				} else {
					ext := "md"
					if strings.HasPrefix(format, "json") {
						ext = "json"
					}
					path = fmt.Sprintf("TZ_%s.%s", time.Now().Format("20060102_150405"), ext)
				}
				if err := os.WriteFile(path, []byte(lastAnswer), 0644); err != nil {
					fmt.Printf("Ошибка сохранения: %v\n", err)
				} else {
					fmt.Printf("Сохранено: %s\n", path)
				}
			case "/tz":
				if len(parts) < 2 {
					fmt.Println("Использование: /tz on | /tz off | /tz finalize")
					break
				}
				sub := strings.ToLower(parts[1])
				switch sub {
				case "on":
					tzMode = true
					sysPrompt = buildTZSystemPrompt(format)
					messages = append(messages, openrouter.ChatMessage{Role: "system", Content: sysPrompt})
					fmt.Println("Режим ТЗ включён. Модель будет собирать требования и оформлять ТЗ.")
				case "off":
					tzMode = false
					sysPrompt = buildSystemPrompt(format)
					messages = append(messages, openrouter.ChatMessage{Role: "system", Content: sysPrompt})
					fmt.Println("Режим ТЗ выключен.")
				case "finalize":
					if !tzMode {
						fmt.Println("Команда доступна только в режиме ТЗ. Включите: /tz on")
						break
					}
					nextUseStop = true
					messages = append(messages, openrouter.ChatMessage{Role: "user", Content: "Утвердить"})
					fmt.Printf("Финализирую ТЗ. Будет применён stop-маркер: %s\n", tzEndMarker)
					runNow = true
				default:
					fmt.Println("Неизвестная подкоманда. Использование: /tz on | /tz off | /tz finalize")
				}
			default:
				fmt.Println("Неизвестная команда. Введите /help для справки.")
			}
			if !runNow {
				continue
			}
		}

		if !ranCommand {
			messages = append(messages, openrouter.ChatMessage{Role: "user", Content: line})
		}

		// Tool-calling loop (max 5 steps)
		var assistantOut string
		finalizeComplete := false
		var finalBuffer strings.Builder
		for step := 0; step < 5; step++ {
			var assistantMsg openrouter.ChatMessage
			stopSpin := startSpinner("Думаю…")
			// Увеличиваем лимит токенов на финальном шаге, чтобы не обрывалось по длине
			reqMax := maxTokens
			if nextUseStop && reqMax < 2000 {
				reqMax = 2000
			}
			req := openrouter.ChatCompletionRequest{
				Model:      model,
				Messages:   messages,
				Tools:      tools,
				ToolChoice: "auto",
				MaxTokens:  reqMax,
			}
			// hint model to return JSON if format requires it
			if strings.HasPrefix(format, "json") {
				req.ResponseFormat = map[string]any{"type": "json_object"}
			}
			// apply stop marker on finalize
			if nextUseStop {
				req.Stop = []string{tzEndMarker}
				// disable tool-use on finalize to force direct final output
				req.Tools = nil
				req.ToolChoice = ""
			}
			resp, err := openrouter.CreateChatCompletion(ctx, token, req)
			if err != nil {
				stopSpin()
				// Фолбэк: выбранная модель/провайдер не поддерживает инструменты
				errLower := strings.ToLower(err.Error())
				if strings.Contains(errLower, "support tool use") {
					fmt.Println("Предупреждение: модель не поддерживает инструменты. Продолжаю без tools…")
					stopSpin = startSpinner("Думаю…")
					req2 := openrouter.ChatCompletionRequest{Model: model, Messages: messages, MaxTokens: reqMax}
					if strings.HasPrefix(format, "json") {
						req2.ResponseFormat = map[string]any{"type": "json_object"}
					}
					if nextUseStop {
						req2.Stop = []string{tzEndMarker}
						// На финале выключаем инструменты
						req2.Tools = nil
						req2.ToolChoice = ""
					}
					resp2, err2 := openrouter.CreateChatCompletion(ctx, token, req2)
					stopSpin()
					if err2 != nil || len(resp2.Choices) == 0 {
						fmt.Printf("Ошибка запроса: %v\n", err2)
						break
					}
					finish := resp2.Choices[0].FinishReason
					assistantMsg = resp2.Choices[0].Message
					messages = append(messages, assistantMsg)
					if nextUseStop {
						finalBuffer.WriteString(assistantMsg.Content)
					}
					if nextUseStop && strings.EqualFold(finish, "length") {
						messages = append(messages, openrouter.ChatMessage{Role: "user", Content: "Продолжай финальный вывод ТЗ с того места, где остановился. Заверши и выведи END_OF_TZ."})
						continue
					}
					if nextUseStop && (strings.EqualFold(finish, "stop") || strings.Contains(assistantMsg.Content, tzEndMarker)) {
						finalizeComplete = true
					}
					assistantOut = assistantMsg.Content
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
					reqMax = maxTokens
					if nextUseStop && reqMax < 1500 {
						reqMax = 1500
					}
					reqRetry := openrouter.ChatCompletionRequest{Model: model, Messages: messages, MaxTokens: reqMax, Tools: tools, ToolChoice: "auto"}
					if strings.HasPrefix(format, "json") {
						reqRetry.ResponseFormat = map[string]any{"type": "json_object"}
					}
					if nextUseStop {
						reqRetry.Stop = []string{tzEndMarker}
						// disable tools on retry finalize as well
						reqRetry.Tools = nil
						reqRetry.ToolChoice = ""
					}
					respRetry, errRetry := openrouter.CreateChatCompletion(ctx, token, reqRetry)
					stopSpin()
					if errRetry != nil || len(respRetry.Choices) == 0 {
						fmt.Printf("Ошибка запроса после понижения max_tokens: %v\n", errRetry)
						break
					}
					finish := respRetry.Choices[0].FinishReason
					assistantMsg = respRetry.Choices[0].Message
					messages = append(messages, assistantMsg)
					if nextUseStop {
						finalBuffer.WriteString(assistantMsg.Content)
					}
					if nextUseStop && strings.EqualFold(finish, "length") {
						messages = append(messages, openrouter.ChatMessage{Role: "user", Content: "Продолжай финальный вывод ТЗ с того места, где остановился. Заверши и выведи END_OF_TZ."})
						continue
					}
					if nextUseStop && (strings.EqualFold(finish, "stop") || strings.Contains(assistantMsg.Content, tzEndMarker)) {
						finalizeComplete = true
					}
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
				finish := resp.Choices[0].FinishReason
				assistantMsg = resp.Choices[0].Message
				messages = append(messages, assistantMsg)
				if nextUseStop {
					finalBuffer.WriteString(assistantMsg.Content)
				}
				if nextUseStop && strings.EqualFold(finish, "length") {
					messages = append(messages, openrouter.ChatMessage{Role: "user", Content: "Продолжай финальный вывод ТЗ с того места, где остановился. Заверши и выведи END_OF_TZ."})
					continue
				}
				if nextUseStop && (strings.EqualFold(finish, "stop") || strings.Contains(assistantMsg.Content, tzEndMarker)) {
					finalizeComplete = true
				}
			}

			if len(assistantMsg.ToolCalls) == 0 {
				// если финализируем — берём накопленный буфер, иначе — одиночный ответ
				if nextUseStop {
					assistantOut = finalBuffer.String()
				} else {
					assistantOut = assistantMsg.Content
				}
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
			reqMax2 := maxTokens
			if nextUseStop && reqMax2 < 2000 {
				reqMax2 = 2000
			}
			req := openrouter.ChatCompletionRequest{Model: model, Messages: messages, MaxTokens: reqMax2}
			if strings.HasPrefix(format, "json") {
				req.ResponseFormat = map[string]any{"type": "json_object"}
			}
			if nextUseStop {
				req.Stop = []string{tzEndMarker}
				// и здесь тоже фиксируем без tools
				req.Tools = nil
				req.ToolChoice = ""
			}
			resp, err := openrouter.CreateChatCompletion(ctx, token, req)
			stopSpin()
			if err == nil && len(resp.Choices) > 0 {
				if nextUseStop {
					assistantOut = resp.Choices[0].Message.Content
				} else {
					assistantOut = resp.Choices[0].Message.Content
				}
			}
		}

		// Если финализация — используем буфер и завершаем
		if nextUseStop {
			buf := finalBuffer.String()
			if buf != "" {
				assistantOut = buf
			}
			if finalizeComplete || strings.Contains(assistantOut, tzEndMarker) {
				nextUseStop = false
			}
		}

		// Сохраняем последний ответ и автосохранение ТЗ в файл при финализации
		lastAnswer = assistantOut
		justFinalized := tzMode && (finalizeComplete || strings.Contains(assistantOut, tzEndMarker))
		if justFinalized {
			if path, err := saveFinalTZ(assistantOut, format); err == nil {
				fmt.Printf("[auto-save] ТЗ сохранено: %s\n", path)
			} else {
				fmt.Printf("[auto-save] Не удалось сохранить ТЗ: %v\n", err)
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

	// Рекомендуемые модели (предпочтения и стабильность JSON/tool-use)
	recommended := []string{
		"anthropic/claude-3.5-sonnet",
		"openai/gpt-4o-mini",
		"openrouter/auto",
	}
	recSet := map[string]struct{}{}
	for _, id := range recommended {
		recSet[id] = struct{}{}
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
	fmt.Println("Доступные модели:")
	idx := 1
	indexToID := make(map[string]string)
	// Секция: Рекомендуемые
	if len(recommended) > 0 {
		fmt.Println("Рекомендуемые:")
		for _, id := range recommended {
			fmt.Printf("%2d) %s\n", idx, id)
			indexToID[fmt.Sprintf("%d", idx)] = id
			idx++
		}
	}
	if len(freeShow) > 0 {
		fmt.Println("Бесплатные:")
		for _, id := range freeShow {
			if _, ok := recSet[id]; ok { // пропустим дубликаты из рекомендуемых
				continue
			}
			fmt.Printf("%2d) %s\n", idx, id)
			indexToID[fmt.Sprintf("%d", idx)] = id
			idx++
		}
	}
	if len(paidShow) > 0 {
		fmt.Println("Платные:")
		for _, id := range paidShow {
			if _, ok := recSet[id]; ok { // пропустим дубликаты из рекомендуемых
				continue
			}
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
	fmt.Println("  /tz on|off|finalize       — режим подготовки ТЗ и финализация по маркеру")
	fmt.Println("  /save [path]              — сохранить последний ответ в файл")
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

func buildTZSystemPrompt(format string) string {
	// Сжатая инструкция BA-режима с маркером финала
	base := "Ты — бизнес-аналитик. Сначала собираешь требования, затем оформляешь полное ТЗ. Работаешь циклами: задать до 3 уточняющих вопросов → обновить черновик секций → проверить чек-лист полноты → запросить подтверждение → итог. Если пользователь напишет ‘Утвердить’ или не ответит два шага подряд, выдай финальный результат. Финальный ответ строго заканчивай строкой END_OF_TZ. Всегда отвечай по-русски. Не раскрывай внутренние рассуждения."
	if strings.HasPrefix(format, "json") {
		// В ТЗ-режиме JSON может использоваться для структурированного итога
		return base + " Если запрошен JSON-формат, возвращай валидный JSON по запрошенной схеме без текста вокруг."
	}
	if format == "markdown" {
		return base + " Оформляй ответы в Markdown."
	}
	return base
}

func saveFinalTZ(content, format string) (string, error) {
	ext := "md"
	if strings.HasPrefix(format, "json") {
		ext = "json"
	}
	name := fmt.Sprintf("TZ_%s.%s", time.Now().Format("20060102_150405"), ext)
	if err := os.WriteFile(name, []byte(content), 0644); err != nil {
		return "", err
	}
	return name, nil
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
