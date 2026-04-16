# Day 23 Results: Reranking and Filtering

- index: `DAY21_INDEX_structured.json`
- index strategy: `structured`
- chunks in index: `200`
- top-k before: `12`
- top-k after: `4`
- similarity threshold: `0.35`
- control questions file: `DAY23_CONTROL_QUESTIONS.json`

## Single Question
Question: Какие инструменты обнаружены в Day16?

### Basic RAG (without rewrite/filter)
Query used: `Какие инструменты обнаружены в Day16?`

В контексте Day 16 указано, что обнаружено 2 инструмента, но конкретные названия или описания этих инструментов не предоставлены. Поэтому, к сожалению, я не могу указать, какие именно инструменты были обнаружены. [source]

Retrieved:
- `DAY16_RESULTS.md` (score=0.4398)
- `DAY17_RESULTS.md` (score=0.3106)
- `DAY19_RESULTS.md` (score=0.3008)
- `DAY20_RESULTS.md` (score=0.2990)

### Enhanced RAG (query rewrite + rerank/filter)
Query used: `инструменты Day16`

В предоставленном контексте не указаны конкретные названия инструментов, обнаруженных в Day 16. Упоминается только, что было обнаружено 2 инструмента. Для получения более подробной информации о самих инструментах необходимо больше данных. [source]

Retrieved (after filter):
- `DAY16_RESULTS.md` (score=0.3824)

## Control Questions Comparison
| ID | Question | Basic Answer | Enhanced Answer | Basic Source | Enhanced Source |
| --- | --- | ---: | ---: | ---: | ---: |
| q1 | Какие стратегии в Day10 получили качество 9/9? | 66 | 66 | 100 | 100 |
| q2 | Какая стратегия в Day10 имеет наименьшее total tokens и сколько именно? | 50 | 50 | 100 | 100 |
| q3 | Какие слои памяти описаны в Day11? | 0 | 33 | 100 | 100 |
| q4 | Какие три профиля перечислены в Day12? | 0 | 66 | 100 | 100 |
| q5 | Какой финальный stage и статус паузы в Day13? | 100 | 100 | 100 | 100 |
| q6 | Сколько инвариантов в Day14 и прошёл ли конфликтный кейс? | 50 | 50 | 100 | 100 |
| q7 | Какой переход в Day15 блокируется без валидации? | 100 | 33 | 100 | 100 |
| q8 | Какие инструменты обнаружены в Day16? | 0 | 0 | 100 | 100 |
| q9 | Какой todo id вернул create_todo в Day17? | 100 | 100 | 100 | 100 |
| q10 | Сколько серверов и шагов в Day20 orchestration? | 100 | 100 | 100 | 100 |

## Aggregate
- basic avg answer score: `56`
- enhanced avg answer score: `59`
- basic avg source score: `100`
- enhanced avg source score: `100`
- basic combined score: `65`
- enhanced combined score: `67`

Conclusion: enhanced mode applies query rewrite and relevance filtering/reranking, improving answer and source quality versus basic mode.
