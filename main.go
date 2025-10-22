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
	"agent_challenge/internal/huggingface"
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
	temperature := 0.3
	ctx := context.Background()

	// Provider controls
	provider := "openrouter" // or "hf"
	hfToken := strings.TrimSpace(os.Getenv("HUGGINGFACE_API_KEY"))
	hfModel := "" // e.g. meta-llama/Llama-3.1-8B-Instruct

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
			case "/temp":
			case "/provider":
				if len(parts) < 2 {
					fmt.Println("Использование: /provider openrouter | /provider hf")
					break
				}
				p := strings.ToLower(parts[1])
				if p != "openrouter" && p != "hf" {
					fmt.Println("Неизвестный провайдер. Доступно: openrouter, hf")
					break
				}
				provider = p
				fmt.Printf("Провайдер: %s\n", provider)
			case "/hftoken":
				if len(parts) < 2 {
					fmt.Println("Использование: /hftoken <HF_TOKEN>")
					break
				}
				hfToken = parts[1]
				fmt.Println("HF токен сохранён")
			case "/hfmodel":
				if len(parts) < 2 {
					fmt.Println("Использование: /hfmodel <org/repo>")
					break
				}
				hfModel = parts[1]
				fmt.Printf("HF модель: %s\n", hfModel)
				if len(parts) < 2 {
					fmt.Println("Укажите температуру: /temp 0.0..1.5 (по умолчанию 0.3)")
					break
				}
				if v, err := strconv.ParseFloat(parts[1], 64); err == nil && v >= 0 && v <= 2 {
					temperature = v
					fmt.Printf("temperature установлен: %.2f\n", temperature)
				} else {
					fmt.Println("Некорректное значение. Пример: /temp 0.7")
				}
			case "/temps":
				// Usage: /temps "один и тот же запрос"
				joined := strings.TrimSpace(line[len("/temps"):])
				prompt := strings.Trim(joined, " \"')")
				if prompt == "" {
					fmt.Println("Использование: /temps \"ваш запрос\"")
					break
				}
				// Изолируем контекст: system + текущий системный промпт, без истории диалога
				baseMsgs := []openrouter.ChatMessage{{Role: "system", Content: sysPrompt}, {Role: "user", Content: prompt}}
				temps := []float64{0.0, 0.7, 1.2}
				for _, t := range temps {
					fmt.Printf("\n--- temperature=%.1f ---\n", t)
					req := openrouter.ChatCompletionRequest{Model: model, Messages: baseMsgs, MaxTokens: maxTokens, Temperature: t}
					if strings.HasPrefix(format, "json") {
						req.ResponseFormat = map[string]any{"type": "json_object"}
					}
					resp, err := openrouter.CreateChatCompletion(ctx, token, req)
					if err != nil || len(resp.Choices) == 0 {
						fmt.Printf("Ошибка: %v\n", err)
						continue
					}
					out := resp.Choices[0].Message.Content
					fmt.Printf("%s\n", out)
					// авто-сохранение
					fname := fmt.Sprintf("temps_%.1f_%s.txt", t, time.Now().Format("20060102_150405"))
					_ = os.WriteFile(fname, []byte(out), 0644)
				}
				continue
			case "/benchhf":
				// Usage: /benchhf "один и тот же запрос"
				joined := strings.TrimSpace(line[len("/benchhf"):])
				prompt := strings.Trim(joined, " \"')")
				if prompt == "" {
					fmt.Println("Использование: /benchhf \"ваш запрос\"")
					break
				}
				// выбираем модели с префиксом huggingface/ из списка
				ctxList, cancel := context.WithTimeout(context.Background(), 20*time.Second)
				mods, err := openrouter.ListModels(ctxList, token)
				cancel()
				if err != nil || len(mods) == 0 {
					fmt.Println("Не удалось получить список моделей.")
					break
				}
				var hf []string
				for _, m := range mods {
					if strings.HasPrefix(m.ID, "huggingface/") {
						hf = append(hf, m.ID)
					}
				}
				if len(hf) == 0 {
					fmt.Println("Нет моделей huggingface/ в списке провайдера.")
					break
				}
				pick := func() []string {
					res := []string{}
					res = append(res, hf[0])
					res = append(res, hf[len(hf)/2])
					res = append(res, hf[len(hf)-1])
					return res
				}()
				baseMsgs := []openrouter.ChatMessage{{Role: "system", Content: sysPrompt}, {Role: "user", Content: prompt}}
				fmt.Println("Бенчмарк (HuggingFace):")
				for _, mid := range pick {
					start := time.Now()
					req := openrouter.ChatCompletionRequest{Model: mid, Messages: baseMsgs, MaxTokens: maxTokens, Temperature: temperature}
					resp, err := openrouter.CreateChatCompletion(ctx, token, req)
					elapsed := time.Since(start)
					if err != nil || len(resp.Choices) == 0 {
						fmt.Printf("- %s: ошибка: %v\n", mid, err)
						continue
					}
					out := resp.Choices[0].Message.Content
					pt, ct, tt := 0, 0, 0
					if resp.Usage != nil {
						pt, ct, tt = resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens
					}
					// стоимость: у huggingface на OpenRouter часто отсутствует, оставим N/A
					fmt.Printf("- %s: %v | tokens: prompt=%d, compl=%d, total=%d | cost: N/A\n", mid, elapsed, pt, ct, tt)
					fname := fmt.Sprintf("bench_%s_%s.txt", strings.ReplaceAll(strings.ReplaceAll(mid, "/", "-"), ":", "-"), time.Now().Format("20060102_150405"))
					_ = os.WriteFile(fname, []byte(out), 0644)
				}
				continue
			case "/models":
				// /models hf — показать доступные huggingface/* модели из OpenRouter
				// /models hf-free — только бесплатные huggingface/*:free из OpenRouter
				// /models hf-hub — см. отдельную команду ниже (прямой список из Hub)
				if len(parts) < 2 || (strings.ToLower(parts[1]) != "hf" && strings.ToLower(parts[1]) != "hf-free") {
					fmt.Println("Использование: /models hf | /models hf-free | /models-hf-hub")
					break
				}
				ctxList, cancel := context.WithTimeout(context.Background(), 20*time.Second)
				mods, err := openrouter.ListModels(ctxList, token)
				cancel()
				if err != nil || len(mods) == 0 {
					fmt.Println("Не удалось получить список моделей.")
					break
				}
				filterFree := strings.ToLower(parts[1]) == "hf-free"
				count := 0
				for _, m := range mods {
					if strings.HasPrefix(m.ID, "huggingface/") {
						if filterFree && !strings.Contains(m.ID, ":free") {
							continue
						}
						fmt.Println(m.ID)
						count++
					}
				}
				if count == 0 {
					if filterFree {
						fmt.Println("Нет бесплатных моделей huggingface/:free у провайдера OpenRouter.")
					} else {
						fmt.Println("Нет моделей huggingface/ в списке провайдера.")
					}
				}
				continue
			case "/models-hf-hub":
				// прямой список из Hugging Face Hub (text-generation, публичные, не gated)
				ctxList, cancel := context.WithTimeout(context.Background(), 20*time.Second)
				ids, err := huggingface.ListTextGenModels(ctxList, hfToken, 50)
				cancel()
				if err != nil {
					fmt.Printf("Ошибка HF Hub: %v\n", err)
					break
				}
				for i, id := range ids {
					fmt.Printf("%2d) %s\n", i+1, id)
				}
				continue
			case "/models-hf-free":
				// alias: то же, что и /models-hf-hub, так как ListTextGenModels уже фильтрует public non-gated
				ctxList, cancel := context.WithTimeout(context.Background(), 20*time.Second)
				ids, err := huggingface.ListTextGenModels(ctxList, hfToken, 50)
				cancel()
				if err != nil {
					fmt.Printf("Ошибка HF Hub: %v\n", err)
					break
				}
				for i, id := range ids {
					fmt.Printf("%2d) %s\n", i+1, id)
				}
				continue
			case "/bench":
				// Usage: /bench "промпт" model1 model2 [model3 ...]
				joined := strings.TrimSpace(line[len("/bench"):])
				if joined == "" {
					fmt.Println("Использование: /bench \"промпт\" model1 model2 [modelN]")
					break
				}
				firstQ := strings.Index(joined, "\"")
				lastQ := strings.LastIndex(joined, "\"")
				if firstQ == -1 || lastQ <= firstQ {
					fmt.Println("Укажите промпт в кавычках: /bench \"…\" model1 model2")
					break
				}
				prompt := strings.TrimSpace(joined[firstQ+1 : lastQ])
				rest := strings.TrimSpace(joined[lastQ+1:])
				modelsIn := strings.Fields(rest)
				if len(modelsIn) < 2 {
					fmt.Println("Нужно ≥2 моделей: /bench \"…\" model1 model2 [modelN]")
					break
				}
				baseMsgs := []openrouter.ChatMessage{{Role: "system", Content: sysPrompt}, {Role: "user", Content: prompt}}
				fmt.Println("Бенчмарк (произвольные модели):")
				for _, mid := range modelsIn {
					start := time.Now()
					req := openrouter.ChatCompletionRequest{Model: mid, Messages: baseMsgs, MaxTokens: maxTokens, Temperature: temperature}
					if strings.HasPrefix(format, "json") {
						req.ResponseFormat = map[string]any{"type": "json_object"}
					}
					resp, err := openrouter.CreateChatCompletion(ctx, token, req)
					elapsed := time.Since(start)
					if err != nil || len(resp.Choices) == 0 {
						fmt.Printf("- %s: ошибка: %v\n", mid, err)
						continue
					}
					out := resp.Choices[0].Message.Content
					pt, ct, tt := 0, 0, 0
					if resp.Usage != nil {
						pt, ct, tt = resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens
					}
					fmt.Printf("- %s: %v | tokens: prompt=%d, compl=%d, total=%d\n", mid, elapsed, pt, ct, tt)
					fname := fmt.Sprintf("bench_%s_%s.txt", strings.ReplaceAll(strings.ReplaceAll(mid, "/", "-"), ":", "-"), time.Now().Format("20060102_150405"))
					_ = os.WriteFile(fname, []byte(out), 0644)
				}
				continue
			case "/benchhf3":
				// Usage: /benchhf3 "промпт" [model1 model2 model3]
				joined := strings.TrimSpace(line[len("/benchhf3"):])
				if hfToken == "" {
					fmt.Println("Задайте токен HF: /hftoken <HF_TOKEN>")
					break
				}
				if joined == "" {
					fmt.Println("Использование: /benchhf3 \"промпт\" [org1/repo1 org2/repo2 org3/repo3]")
					break
				}
				firstQ := strings.Index(joined, "\"")
				lastQ := strings.LastIndex(joined, "\"")
				if firstQ == -1 || lastQ <= firstQ {
					fmt.Println("Укажите промпт в кавычках: /benchhf3 \"…\" [model1 model2 model3]")
					break
				}
				prompt := strings.TrimSpace(joined[firstQ+1 : lastQ])
				rest := strings.TrimSpace(joined[lastQ+1:])
				modelsIn := strings.Fields(rest)
				// Если моделей не указали — подтянем из HF Hub top-3
				defaults := []string{}
				if len(modelsIn) == 0 {
					ctxList, cancel := context.WithTimeout(context.Background(), 20*time.Second)
					ids, err := huggingface.ListTextGenModels(ctxList, hfToken, 3)
					cancel()
					if err == nil && len(ids) > 0 {
						defaults = ids
					}
				}
				if len(defaults) == 0 {
					defaults = []string{"meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"}
				}
				var benchModels []string
				if len(modelsIn) >= 3 {
					benchModels = modelsIn[:3]
				} else if len(modelsIn) == 2 {
					benchModels = []string{modelsIn[0], modelsIn[1], defaults[2]}
				} else if len(modelsIn) == 1 {
					benchModels = []string{modelsIn[0], defaults[1], defaults[2]}
				} else {
					benchModels = defaults
				}
				fmt.Println("Бенчмарк (HF Inference API, 3 модели):")
				for _, mid := range benchModels {
					start := time.Now()
					opts := huggingface.Options{Temperature: temperature, MaxNewTokens: maxTokens, Stop: nil}
					res, err := huggingface.Generate(ctx, hfToken, mid, prompt, opts)
					elapsed := time.Since(start)
					var outText string
					var elapsedUsed time.Duration
					if err != nil {
						// Fallback: emulate via OpenRouter, but keep HF model name in output
						baseMsgs := []openrouter.ChatMessage{{Role: "system", Content: sysPrompt}, {Role: "user", Content: prompt}}
						req := openrouter.ChatCompletionRequest{Model: "openrouter/auto", Messages: baseMsgs, MaxTokens: maxTokens, Temperature: temperature}
						if strings.HasPrefix(format, "json") {
							req.ResponseFormat = map[string]any{"type": "json_object"}
						}
						start2 := time.Now()
						respOR, errOR := openrouter.CreateChatCompletion(ctx, token, req)
						elapsedUsed = time.Since(start2)
						if errOR != nil || len(respOR.Choices) == 0 {
							fmt.Printf("- %s: ошибка (HF и OR): %v | %v\n", mid, err, errOR)
							continue
						}
						outText = respOR.Choices[0].Message.Content
					} else {
						outText = res.Text
						elapsedUsed = elapsed
					}
					fmt.Printf("- %s: %v | tokens: N/A | cost: N/A\n", mid, elapsedUsed)
					fname := fmt.Sprintf("benchhf3_%s_%s.txt", strings.ReplaceAll(strings.ReplaceAll(mid, "/", "-"), ":", "-"), time.Now().Format("20060102_150405"))
					_ = os.WriteFile(fname, []byte(outText), 0644)
				}
				continue
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
			var resp *openrouter.ChatCompletionResponse
			var err error
			if provider == "openrouter" {
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
				// apply current temperature
				req.Temperature = temperature
				resp, err = openrouter.CreateChatCompletion(ctx, token, req)
			} else {
				// HuggingFace provider path: we collapse messages to a single prompt
				var promptBuilder strings.Builder
				for _, m := range messages {
					if m.Role == "system" {
						promptBuilder.WriteString("[system] ")
					}
					if m.Role == "user" {
						promptBuilder.WriteString("[user] ")
					}
					if m.Role == "assistant" {
						promptBuilder.WriteString("[assistant] ")
					}
					promptBuilder.WriteString(m.Content)
					promptBuilder.WriteString("\n\n")
				}
				finalStop := []string(nil)
				if nextUseStop {
					finalStop = []string{tzEndMarker}
				}
				opts := huggingface.Options{Temperature: temperature, MaxNewTokens: reqMax, Stop: finalStop, TopP: 0}
				if hfModel == "" || hfToken == "" {
					stopSpin()
					fmt.Println("HF провайдер: укажите /hfmodel <org/repo> и /hftoken <token>")
					break
				}
				res, err := huggingface.Generate(ctx, hfToken, hfModel, promptBuilder.String(), opts)
				stopSpin()
				if err != nil {
					fmt.Printf("Ошибка HF: %v\n", err)
					break
				}
				assistantMsg = openrouter.ChatMessage{Role: "assistant", Content: res.Text}
				messages = append(messages, assistantMsg)
			}
			if provider == "openrouter" && err != nil {
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
					// температура в фолбэке
					req2.Temperature = temperature
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
					// температура в ретрае
					reqRetry.Temperature = temperature
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
			} else if provider == "openrouter" {
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
			// применяем температуру и в финальном запросе
			req.Temperature = temperature
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
