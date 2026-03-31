package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

type layeredMemoryAgent struct {
	apiKey                   string
	model                    string
	maxTokens                int
	temperature              *float64
	title                    string
	systemPrompt             string
	router                   *MemoryRouter
	cumulativePromptTokens   int
	cumulativeResponseTokens int
}

type day11DemoResult struct {
	Model             string
	TaskID            string
	ShortLayer        string
	WorkingLayer      string
	LongLayer         string
	Actions           []string
	Question          string
	BaselineAnswer    string
	LayeredAnswer     string
	BaselinePromptTok int
	BaselineTotalTok  int
	LayeredPromptTok  int
	LayeredTotalTok   int
	BaselineScore     int
	LayeredScore      int
	MaxScore          int
}

func newLayeredMemoryAgent(apiKey, model string, maxTokens int, temperature *float64, router *MemoryRouter) *layeredMemoryAgent {
	return &layeredMemoryAgent{
		apiKey:       apiKey,
		model:        model,
		maxTokens:    maxTokens,
		temperature:  temperature,
		title:        "day11-memory-layers",
		systemPrompt: "Ты ассистент с многослойной памятью. Используй long-term и working memory как факты, short-term как текущий контекст.",
		router:       router,
	}
}

func (a *layeredMemoryAgent) Reply(userInput string) (openRouterResult, error) {
	userInput = strings.TrimSpace(userInput)
	if userInput == "" {
		return openRouterResult{}, fmt.Errorf("empty user input")
	}
	if err := a.router.SaveShortMessage("user", userInput); err != nil {
		return openRouterResult{}, err
	}

	systemMemory, err := a.router.BuildSystemMemoryMessages()
	if err != nil {
		return openRouterResult{}, err
	}
	snapshot, err := a.router.Snapshot()
	if err != nil {
		return openRouterResult{}, err
	}

	requestMessages := make([]message, 0, 1+len(systemMemory)+len(snapshot.Short))
	requestMessages = append(requestMessages, message{
		Role:    "system",
		Content: a.systemPrompt,
	})
	requestMessages = append(requestMessages, systemMemory...)
	requestMessages = append(requestMessages, snapshot.Short...)

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

	answer := strings.TrimSpace(result.Answer)
	if err := a.router.SaveShortMessage("assistant", answer); err != nil {
		return openRouterResult{}, err
	}

	promptTokens := result.Usage.PromptTokens
	responseTokens := result.Usage.CompletionTokens
	totalTokens := result.Usage.TotalTokens
	if promptTokens == 0 {
		promptTokens = estimateMessagesTokens(requestMessages)
	}
	if responseTokens == 0 {
		responseTokens = estimateTextTokens(answer)
	}
	if totalTokens == 0 {
		totalTokens = promptTokens + responseTokens
	}
	a.cumulativePromptTokens += promptTokens
	a.cumulativeResponseTokens += responseTokens

	result.Answer = answer
	result.Tokens = tokenStats{
		PromptTokens:             promptTokens,
		ResponseTokens:           responseTokens,
		TotalTokens:              totalTokens,
		CumulativePromptTokens:   a.cumulativePromptTokens,
		CumulativeResponseTokens: a.cumulativeResponseTokens,
		CumulativeTotalTokens:    a.cumulativePromptTokens + a.cumulativeResponseTokens,
	}
	return result, nil
}

func runDay11Command(args []string) error {
	fs := flag.NewFlagSet("openrouter-cli day11", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	model := fs.String("model", getDefaultModel(), "OpenRouter model")
	maxTokens := fs.Int("max-tokens", 260, "Maximum response tokens")
	temperature := fs.Float64("temperature", 0.2, "Temperature")
	memoryDir := fs.String("memory-dir", "/tmp/day11-memory-layers", "Directory with separated memory layer files")
	taskID := fs.String("task-id", "day11-task-spec", "Working memory task ID")
	shortWindow := fs.Int("short-window", 10, "Number of recent short-term messages sent to model")
	reportPath := fs.String("report", "DAY11_RESULTS.md", "Report path")
	interactive := fs.Bool("interactive", false, "Run interactive mode")
	showTokens := fs.Bool("show-tokens", true, "Show token stats in interactive mode")
	reset := fs.Bool("reset", true, "Reset memory files before run")
	help := fs.Bool("help", false, "Show help")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("failed to parse day11 flags: %w", err)
	}
	if *help {
		printDay11Usage()
		return nil
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected day11 arguments: %s", strings.Join(fs.Args(), " "))
	}

	store := newMemoryFileStore(*memoryDir)
	if *reset {
		if err := store.reset(); err != nil {
			return fmt.Errorf("failed to reset memory dir: %w", err)
		}
	}
	router := newMemoryRouter(store, *taskID, *shortWindow)

	t := *temperature
	agent := newLayeredMemoryAgent(getAPIKey(), *model, *maxTokens, &t, router)

	if *interactive {
		return runDay11Interactive(agent, router, *showTokens)
	}

	result, err := runDay11Demo(agent, router)
	if err != nil {
		return err
	}
	result.Model = *model
	result.TaskID = router.TaskID()

	printDay11DemoResult(result)
	if err := writeDay11Report(*reportPath, result); err != nil {
		return err
	}
	fmt.Printf("Отчёт: %s\n", *reportPath)
	return nil
}

func runDay11Interactive(agent *layeredMemoryAgent, router *MemoryRouter, showTokens bool) error {
	fmt.Println("Day11 interactive mode. Type /exit to quit.")
	fmt.Println("Commands:")
	fmt.Println("  /task show")
	fmt.Println("  /task set <task_id>")
	fmt.Println("  /mem save short <text>")
	fmt.Println("  /mem save work <goal|constraint|decision|preference|note|value> <text or key=value>")
	fmt.Println("  /mem save long <profile|preference|knowledge|decision|note> <text or key=value>")
	fmt.Println("  /mem show <short|working|long|all>")
	fmt.Println("  /mem clear <short|working|long|all>")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("you> ")
		if !scanner.Scan() {
			return scanner.Err()
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		switch strings.ToLower(line) {
		case "/exit", "exit", "quit":
			return nil
		}

		if strings.HasPrefix(line, "/") {
			handled, err := handleDay11Command(router, line)
			if err != nil {
				fmt.Fprintf(os.Stderr, "memory error: %v\n", err)
			}
			if handled {
				continue
			}
		}

		result, err := agent.Reply(line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "agent error: %v\n", err)
			continue
		}
		fmt.Printf("agent> %s\n\n", result.Answer)
		if showTokens {
			printTokenStats(result.Tokens)
		}
	}
}

func handleDay11Command(router *MemoryRouter, line string) (bool, error) {
	fields := strings.Fields(strings.TrimSpace(line))
	if len(fields) == 0 {
		return false, nil
	}
	switch strings.ToLower(fields[0]) {
	case "/task":
		if len(fields) < 2 {
			return true, fmt.Errorf("usage: /task show|set <task_id>")
		}
		switch strings.ToLower(fields[1]) {
		case "show":
			fmt.Printf("memory> active task: %s\n\n", router.TaskID())
			return true, nil
		case "set":
			if len(fields) < 3 {
				return true, fmt.Errorf("usage: /task set <task_id>")
			}
			router.SetTask(fields[2])
			fmt.Printf("memory> active task switched to: %s\n\n", router.TaskID())
			return true, nil
		default:
			return true, fmt.Errorf("unknown /task action: %s", fields[1])
		}
	case "/mem":
		if len(fields) < 2 {
			return true, fmt.Errorf("usage: /mem save|show|clear ...")
		}
		action := strings.ToLower(fields[1])
		switch action {
		case "show":
			if len(fields) < 3 {
				return true, fmt.Errorf("usage: /mem show short|working|long|all")
			}
			text, err := router.Render(fields[2])
			if err != nil {
				return true, err
			}
			fmt.Printf("memory> %s\n\n", text)
			return true, nil
		case "clear":
			if len(fields) < 3 {
				return true, fmt.Errorf("usage: /mem clear short|working|long|all")
			}
			if err := router.Clear(fields[2]); err != nil {
				return true, err
			}
			fmt.Printf("memory> cleared %s layer\n\n", strings.ToLower(fields[2]))
			return true, nil
		case "save":
			return true, day11HandleSave(router, line, fields)
		default:
			return true, fmt.Errorf("unknown /mem action: %s", action)
		}
	default:
		return false, nil
	}
}

func day11HandleSave(router *MemoryRouter, line string, fields []string) error {
	if len(fields) < 4 {
		return fmt.Errorf("usage: /mem save <short|work|long> ...")
	}
	layer := strings.ToLower(fields[2])
	rest := strings.TrimSpace(strings.TrimPrefix(line, strings.Join(fields[:4], " ")))
	if rest != "" {
		rest = strings.TrimSpace(fields[3] + " " + rest)
	} else {
		rest = strings.TrimSpace(strings.TrimPrefix(line, strings.Join(fields[:3], " ")))
	}

	switch layer {
	case "short":
		value := strings.TrimSpace(strings.TrimPrefix(line, strings.Join(fields[:3], " ")))
		if value == "" {
			return fmt.Errorf("usage: /mem save short <text>")
		}
		if err := router.SaveExplicit(MemoryLayerShort, "", "", value); err != nil {
			return err
		}
		fmt.Println("memory> saved to short-term layer")
		fmt.Println()
		return nil
	case "work", "working":
		kind := strings.ToLower(fields[3])
		value := strings.TrimSpace(strings.TrimPrefix(line, strings.Join(fields[:4], " ")))
		key, parsedValue := parseKV(value)
		if kind == "value" || kind == "kv" {
			if key == "" || parsedValue == "" {
				return fmt.Errorf("for working value use key=value")
			}
			value = parsedValue
		} else {
			key = ""
			value = strings.TrimSpace(value)
		}
		if value == "" {
			return fmt.Errorf("empty working value")
		}
		if err := router.SaveExplicit(MemoryLayerWorking, kind, key, value); err != nil {
			return err
		}
		fmt.Println("memory> saved to working layer")
		fmt.Println()
		return nil
	case "long":
		kind := strings.ToLower(fields[3])
		value := strings.TrimSpace(strings.TrimPrefix(line, strings.Join(fields[:4], " ")))
		key, parsedValue := parseKV(value)
		switch kind {
		case "profile", "preference", "knowledge":
			if key == "" || parsedValue == "" {
				return fmt.Errorf("for long %s use key=value", kind)
			}
			value = parsedValue
		default:
			key = ""
			value = strings.TrimSpace(value)
			if value == "" {
				return fmt.Errorf("empty long value")
			}
		}
		if err := router.SaveExplicit(MemoryLayerLong, kind, key, value); err != nil {
			return err
		}
		fmt.Println("memory> saved to long-term layer")
		fmt.Println()
		return nil
	default:
		return fmt.Errorf("unknown layer: %s", layer)
	}
}

func parseKV(text string) (string, string) {
	left, right, ok := strings.Cut(strings.TrimSpace(text), "=")
	if !ok {
		return "", ""
	}
	return strings.TrimSpace(left), strings.TrimSpace(right)
}

func runDay11Demo(agent *layeredMemoryAgent, router *MemoryRouter) (day11DemoResult, error) {
	actions := []struct {
		Layer MemoryLayer
		Kind  string
		Key   string
		Value string
	}{
		{Layer: MemoryLayerLong, Kind: "profile", Key: "name", Value: "Ivan"},
		{Layer: MemoryLayerLong, Kind: "profile", Key: "role", Value: "backend engineer"},
		{Layer: MemoryLayerLong, Kind: "preference", Key: "language", Value: "Go"},
		{Layer: MemoryLayerLong, Kind: "knowledge", Key: "timezone", Value: "Asia/Almaty"},
		{Layer: MemoryLayerWorking, Kind: "goal", Value: "Собрать ТЗ для MVP приложения доставки"},
		{Layer: MemoryLayerWorking, Kind: "constraint", Value: "Бюджет 15000 USD"},
		{Layer: MemoryLayerWorking, Kind: "constraint", Value: "Срок запуска 8 недель"},
		{Layer: MemoryLayerWorking, Kind: "decision", Value: "Push-провайдер OneSignal"},
		{Layer: MemoryLayerWorking, Kind: "value", Key: "sla", Value: "99.9%"},
	}

	actionLines := make([]string, 0, len(actions))
	for _, action := range actions {
		if err := router.SaveExplicit(action.Layer, action.Kind, action.Key, action.Value); err != nil {
			return day11DemoResult{}, err
		}
		label := fmt.Sprintf("%s.%s", action.Layer, action.Kind)
		if strings.TrimSpace(action.Key) != "" {
			label += "." + action.Key
		}
		actionLines = append(actionLines, label+"="+action.Value)
	}

	shortSeed := []message{
		{Role: "user", Content: "Мы обсудили экран логина и корзины."},
		{Role: "assistant", Content: "Принято, фиксирую логин, корзину и оформление заказа."},
		{Role: "user", Content: "Добавим фильтр по городу и историю заказов."},
	}
	for _, msg := range shortSeed {
		if err := router.SaveShortMessage(msg.Role, msg.Content); err != nil {
			return day11DemoResult{}, err
		}
	}

	question := "Сделай краткую сводку: имя пользователя, цель задачи, бюджет, push-провайдер и что обсуждали в последних сообщениях."

	snapshotBefore, err := router.Snapshot()
	if err != nil {
		return day11DemoResult{}, err
	}
	baselineMessages := make([]message, 0, 2+len(snapshotBefore.Short))
	baselineMessages = append(baselineMessages, message{
		Role:    "system",
		Content: "Ты помощник. Отвечай кратко и по фактам из доступного контекста.",
	})
	baselineMessages = append(baselineMessages, snapshotBefore.Short...)
	baselineMessages = append(baselineMessages, message{Role: "user", Content: question})

	baseline, err := callOpenRouterDetailed(
		agent.apiKey,
		agent.model,
		baselineMessages,
		agent.maxTokens,
		agent.temperature,
		nil,
		"day11-memory-baseline",
	)
	if err != nil {
		return day11DemoResult{}, err
	}

	layered, err := agent.Reply(question)
	if err != nil {
		return day11DemoResult{}, err
	}

	shortLayer, err := router.Render("short")
	if err != nil {
		return day11DemoResult{}, err
	}
	workingLayer, err := router.Render("working")
	if err != nil {
		return day11DemoResult{}, err
	}
	longLayer, err := router.Render("long")
	if err != nil {
		return day11DemoResult{}, err
	}

	bScore, maxScore := day11ScoreAnswer(strings.TrimSpace(baseline.Answer))
	lScore, _ := day11ScoreAnswer(strings.TrimSpace(layered.Answer))

	bPrompt := baseline.Usage.PromptTokens
	bTotal := baseline.Usage.TotalTokens
	if bPrompt == 0 {
		bPrompt = estimateMessagesTokens(baselineMessages)
	}
	if bTotal == 0 {
		bTotal = bPrompt + estimateTextTokens(baseline.Answer)
	}

	return day11DemoResult{
		Actions:           actionLines,
		Question:          question,
		ShortLayer:        shortLayer,
		WorkingLayer:      workingLayer,
		LongLayer:         longLayer,
		BaselineAnswer:    strings.TrimSpace(baseline.Answer),
		LayeredAnswer:     strings.TrimSpace(layered.Answer),
		BaselinePromptTok: bPrompt,
		BaselineTotalTok:  bTotal,
		LayeredPromptTok:  layered.Tokens.PromptTokens,
		LayeredTotalTok:   layered.Tokens.TotalTokens,
		BaselineScore:     bScore,
		LayeredScore:      lScore,
		MaxScore:          maxScore,
	}, nil
}

func day11ScoreAnswer(answer string) (int, int) {
	text := strings.ToLower(answer)
	checks := []string{
		"ivan",
		"мvp",
		"15000",
		"onesignal",
		"фильтр",
		"корзин",
	}
	score := 0
	for _, check := range checks {
		if strings.Contains(text, check) {
			score++
		}
	}
	return score, len(checks)
}

func printDay11DemoResult(result day11DemoResult) {
	fmt.Println("=== Day 11: Memory Layers ===")
	fmt.Printf("task: %s\n", result.TaskID)
	fmt.Printf("quality baseline=%d/%d layered=%d/%d\n", result.BaselineScore, result.MaxScore, result.LayeredScore, result.MaxScore)
	fmt.Printf("tokens baseline(prompt=%d total=%d) layered(prompt=%d total=%d)\n",
		result.BaselinePromptTok, result.BaselineTotalTok, result.LayeredPromptTok, result.LayeredTotalTok,
	)
}

func writeDay11Report(path string, result day11DemoResult) error {
	var b strings.Builder
	b.WriteString("# Day 11 Results: Memory Layers\n\n")
	b.WriteString("Model: `" + result.Model + "`\n")
	b.WriteString("Task: `" + result.TaskID + "`\n\n")

	b.WriteString("## Explicit Memory Routing\n")
	for _, action := range result.Actions {
		b.WriteString("- `" + action + "`\n")
	}
	b.WriteString("\n")

	b.WriteString("## Stored Layers\n\n")
	b.WriteString("### Short-term\n")
	b.WriteString("```text\n" + sanitizeCodeFences(result.ShortLayer) + "\n```\n\n")
	b.WriteString("### Working\n")
	b.WriteString("```text\n" + sanitizeCodeFences(result.WorkingLayer) + "\n```\n\n")
	b.WriteString("### Long-term\n")
	b.WriteString("```text\n" + sanitizeCodeFences(result.LongLayer) + "\n```\n\n")

	b.WriteString("## Same Question Comparison\n")
	b.WriteString("Question:\n")
	b.WriteString("```text\n" + sanitizeCodeFences(result.Question) + "\n```\n\n")
	b.WriteString("Without memory layers (short-term only):\n")
	b.WriteString("```text\n" + sanitizeCodeFences(result.BaselineAnswer) + "\n```\n\n")
	b.WriteString("With memory layers:\n")
	b.WriteString("```text\n" + sanitizeCodeFences(result.LayeredAnswer) + "\n```\n\n")

	b.WriteString("## Metrics\n")
	b.WriteString(fmt.Sprintf("- quality without layers: `%d/%d`\n", result.BaselineScore, result.MaxScore))
	b.WriteString(fmt.Sprintf("- quality with layers: `%d/%d`\n", result.LayeredScore, result.MaxScore))
	b.WriteString(fmt.Sprintf("- prompt tokens without layers: `%d`\n", result.BaselinePromptTok))
	b.WriteString(fmt.Sprintf("- prompt tokens with layers: `%d`\n", result.LayeredPromptTok))
	b.WriteString(fmt.Sprintf("- total tokens without layers: `%d`\n", result.BaselineTotalTok))
	b.WriteString(fmt.Sprintf("- total tokens with layers: `%d`\n", result.LayeredTotalTok))
	b.WriteString("\n")
	b.WriteString("Conclusion: memory layers improve retrieval of persistent and task-level facts, while short-term memory remains focused on recent dialogue.\n")

	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printDay11Usage() {
	fmt.Println("Usage: openrouter-cli day11 [flags]")
	fmt.Println("Flags:")
	fmt.Println("  -model string         OpenRouter model")
	fmt.Println("  -max-tokens int       Maximum response tokens")
	fmt.Println("  -temperature float    Temperature")
	fmt.Println("  -memory-dir string    Directory with short/working/long files")
	fmt.Println("  -task-id string       Current task ID")
	fmt.Println("  -short-window int     Recent short-term messages sent to model")
	fmt.Println("  -report string        Markdown report path")
	fmt.Println("  -interactive          Run interactive mode")
	fmt.Println("  -show-tokens          Show token stats in interactive mode")
	fmt.Println("  -reset                Clear memory files before run")
	fmt.Println("  -help                 Show help")
}
