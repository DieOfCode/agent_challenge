# Day 22 Results: First RAG Query

- index: `DAY21_INDEX_structured.json`
- index strategy: `structured`
- chunks in index: `200`
- control questions file: `DAY22_CONTROL_QUESTIONS.json`

## Single Question Comparison
Question: Какие инструменты были обнаружены в Day 16?

### No RAG
В "Day 16" игры "The Legend of Zelda: Breath of the Wild" не существует, так как игра не разделена на дни. Однако, если вы имеете в виду какой-то конкретный день в контексте прохождения или событий в игре, пожалуйста, уточните, и я постараюсь помочь вам с информацией о найденных инструментах или предметах.

### With RAG
В контексте Day 16 указано, что было обнаружено 2 инструмента, но конкретные названия или описание этих инструментов не предоставлены. Поэтому я не могу ответить на вопрос о том, какие именно инструменты были обнаружены. [source]

### Retrieved Sources (RAG)
- `DAY16_RESULTS.md` (score=0.4614, section=Day 16 Results: MCP Connection and Tool Discovery)
- `DAY19_RESULTS.md` (score=0.3525, section=Day 19 Results: MCP Tool Composition)
- `DAY17_RESULTS.md` (score=0.3474, section=Day 17 Results: First MCP Tool)
- `DAY13_RESULTS.md` (score=0.3168, section=Day 13 Results: Task State Machine)

## Control Questions (10)
| ID | Question | Expectation | Expected Sources | No-RAG Score | RAG Answer Score | RAG Source Score |
| --- | --- | --- | --- | ---: | ---: | ---: |
| q1 | Какие стратегии в Day10 получили качество 9/9? | Нужно назвать sliding_window и branching с качеством 9/9. | DAY10_RESULTS.md | 0 | 100 | 100 |
| q2 | Какая стратегия в Day10 имеет наименьшее total tokens и сколько именно? | Нужно ответить sliding_window и 11945 total tokens. | DAY10_RESULTS.md | 0 | 100 | 100 |
| q3 | Какие слои памяти описаны в Day11? | Нужно перечислить short-term, working и long-term. | DAY11_RESULTS.md | 100 | 0 | 100 |
| q4 | Какие три профиля перечислены в Day12? | Нужно назвать founder-brief, pm-table и dev-json. | DAY12_RESULTS.md | 0 | 0 | 100 |
| q5 | Какой финальный stage и статус паузы в Day13? | Нужно ответить stage done и paused false. | DAY13_RESULTS.md | 0 | 100 | 100 |
| q6 | Сколько инвариантов в Day14 и прошёл ли конфликтный кейс? | Нужно указать invariants count 4 и conflict case passed true. | DAY14_RESULTS.md | 50 | 50 | 100 |
| q7 | Какой переход в Day15 блокируется без валидации? | Нужно сказать, что blocked jump to done without validation = true. | DAY15_RESULTS.md | 0 | 100 | 100 |
| q8 | Какие инструменты обнаружены в Day16? | Нужно назвать system_time и upper_text. | DAY16_RESULTS.md | 0 | 100 | 100 |
| q9 | Какой todo id вернул create_todo в Day17? | Нужно указать todo id 201. | DAY17_RESULTS.md | 50 | 100 | 100 |
| q10 | Сколько серверов и шагов в Day20 orchestration? | Нужно указать servers 2 и steps 6. | DAY20_RESULTS.md | 50 | 100 | 100 |

## Aggregate Scores
- no-rag avg answer score: `25`
- rag avg answer score: `75`
- rag avg source score: `100`
- rag combined score (80% answer + 20% source): `80`

Conclusion: implemented two modes (without RAG / with RAG) and evaluated 10 control questions against indexed local knowledge.
