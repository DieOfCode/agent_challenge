# Day 27 Results: Local LLM Integration

- server: `http://127.0.0.1:11434`
- version: `0.21.0`
- model: `qwen2.5:0.5b`
- app type: `CLI utility (local chat)`
- cloud models: `not used`

## Demo Queries (3)
### 1
Prompt:
```text
Сколько будет 15 + 27? Ответ только числом.
```

Answer:
```text
15 + 27 = 42
```

- prompt tokens: `47`
- response tokens: `11`
- total duration: `2.238s`
- load duration: `1.392s`

### 2
Prompt:
```text
Продолжи последовательность 3, 6, 12, 24 и объясни правило в 1 предложении.
```

Answer:
```text
Правило: каждое число после первого является разумно увеличенным квадратом предыдущего числа.
```

- prompt tokens: `100`
- response tokens: `30`
- total duration: `330ms`
- load duration: `90ms`

### 3
Prompt:
```text
Составь краткий план из 4 шагов для локального CLI-чата на LLM.
```

Answer:
```text
1. Введите команду `llm-start` и выберите язык вашего приложения.
2. Введите команду `llm-set-language` и выберите язык, который вы хотите использовать.
3. Введите команду `llm-set-configuration` и выберите конфигурацию вашего приложения.
4. Введите команду `llm-start` и выберите начальную версию вашего приложения.
```

- prompt tokens: `163`
- response tokens: `96`
- total duration: `1.113s`
- load duration: `61ms`

Conclusion: application sends requests to local Ollama model, receives and displays responses, and works without cloud APIs.
