# Day 29 Results: Local LLM Optimization

## Setup
- index: `DAY21_INDEX_structured.json`
- index strategy: `structured`
- chunks: `200`
- server: `http://127.0.0.1:11434` (v0.21.0)
- retrieval: `local lexical top-k=4`
- repeats: `1`
- optimization note: `quantized alternative not found locally; optimized uses baseline model with tuned params + prompt`

## Before (baseline)
- model: `qwen2.5:0.5b`
- quantization: `Q4_K_M`
- parameter size: `494.03M`
- temperature: `0.20`
- max tokens (num_predict): `220`
- context window (num_ctx): `2048`
- prompt template: `Ты локальный RAG ассистент. Отвечай строго на основе контекста.`

## After (optimized)
- model: `qwen2.5:0.5b`
- quantization: `Q4_K_M`
- parameter size: `494.03M`
- temperature: `0.10`
- max tokens (num_predict): `160`
- context window (num_ctx): `4096`
- prompt template: `Ты оптимизированный RAG-ассистент AGENT CHALLENGE. Отвечай только фактами из контекста. Формат: 1) Краткий ответ. 2) Подтверждения с [source\|chunk_id]. 3) Если данных не хватает, напиши: не знаю, уточните запрос.`

## Single Question
Question: Какие инструменты обнаружены в Day16?

Retrieved chunks:
- `DAY14_RESULTS.md` section=`Allowed Request` chunk_id=`structured:DAY14_RESULTS.md:172` score=2.6000
- `DAY10_FULL_IO.md` section=`Output` chunk_id=`structured:DAY10_FULL_IO.md:52` score=2.4000
- `DAY14_RESULTS_ONLINE.md` section=`Allowed Request` chunk_id=`structured:DAY14_RESULTS_ONLINE.md:176` score=2.4000
- `DAY10_FULL_IO.md` section=`2.2. Курьерская часть` chunk_id=`structured:DAY10_FULL_IO.md:100` score=2.2000

Before answer:
```text
Инструменты обнаружены в Day16:

- pprof (Proton Profiler): инструмент для анализа производительности Go-сервисов.
- Redis: инструмент для хранения данных в встроенной библиотеке Go.
- Explain: инструмент для анализа плана выполнения SQL-запросов.

Эти инструменты помогут вам провести аудит производительности и улучшить эффективность вашего Go-модуля.
```

After answer:
```text
Конечно, в день 16 вы обнаружили несколько инструментов для анализа производительности Go-сервисов:

1. **pprof (Profiler)**: Инструмент для анализа использования CPU и памяти.
2. **Redis**: Инструмент для хранения данных в Redis, который может быть использован для кэширования запросов и выполнений SQL-запросов.
3. **InfluxDB**: Инструмент для анализа данных в формате influxdb, который может быть использован для сбора и анализа данных из Go-сервисов.

Вот пример кода, который можно использовать для запуска инструмента pprof
```

## Benchmark Table
| ID | Question | Quality Before | Quality After | Latency Before | Latency After | Tokens Before | Tokens After | Stable Before | Stable After |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| q1 | Какие стратегии в Day10 получили качество 9/9? | 66 | 33 | 12.725s | 8.184s | 1947 | 2000 | true | true |
| q2 | Какая стратегия в Day10 имеет наименьшее total tokens и сколько именно? | 0 | 0 | 4.286s | 1.819s | 2065 | 2077 | true | true |
| q3 | Какие слои памяти описаны в Day11? | 0 | 0 | 3.063s | 2.019s | 1859 | 1800 | true | true |

## Comparison
- quality before: `22`
- quality after: `11`
- speed before: `6.691s`
- speed after: `4.007s`
- prompt tokens before: `1819`
- prompt tokens after: `1878`
- response tokens before: `137`
- response tokens after: `80`
- total tokens before: `1957`
- total tokens after: `1959`
- load duration before: `4.79s`
- load duration after: `2.539s`
- max model RAM before: `684.2 MiB`
- max model RAM after: `696.2 MiB`
- max model VRAM before: `684.2 MiB`
- max model VRAM after: `696.2 MiB`
- stability before: `100%`
- stability after: `100%`

Conclusion: optimized profile applies parameter tuning (`temperature`, `num_predict`, `num_ctx`), prompt specialization, and optional quantized model selection when locally available.
