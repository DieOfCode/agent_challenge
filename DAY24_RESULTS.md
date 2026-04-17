# Day 24 Results: Citations, Sources, Anti-Hallucination

- index: `DAY21_INDEX_structured.json`
- index strategy: `structured`
- chunks in index: `200`
- top-k before: `12`
- top-k after: `4`
- similarity threshold: `0.35`
- unsure threshold: `0.42`
- control questions file: `DAY24_CONTROL_QUESTIONS.json`

## Single Question
- question: `Как устроен фотосинтез?`
- query used: `фотосинтез устройство`
- best score: `0.1772`
- weak context: `true`
- weak reason: `best score 0.1772 is below unsure threshold 0.4200`

Structured answer:
```json
{
  "answer": "Не знаю: релевантного контекста недостаточно (best score 0.177). Уточните вопрос: добавьте номер дня, файл или конкретный термин.",
  "sources": [
    {
      "source": "DAY10_FULL_IO.md",
      "section": "Output",
      "chunk_id": "structured:DAY10_FULL_IO.md:53"
    }
  ],
  "quotes": [
    {
      "source": "DAY10_FULL_IO.md",
      "section": "Output",
      "chunk_id": "structured:DAY10_FULL_IO.md:53",
      "quote": "- Оформление заказа (без сложных функций, таких как подписки или предзаказы)"
    }
  ]
}
```

## Validation On 10 Questions
| ID | Question | Sources | Quotes | Verbatim Quotes | Answer Matches Quotes | Unknown | Answer Score | Source Score | Strict Score | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q1 | Какие стратегии в Day10 получили качество 9/9? | true | true | true | false | false | 100 | 0 | 75 | 75 |
| q2 | Какая стратегия в Day10 имеет наименьшее total tokens и сколько именно? | true | true | true | true | false | 50 | 0 | 100 | 50 |
| q3 | Какие слои памяти описаны в Day11? | true | true | true | true | true | 0 | 100 | 100 | 40 |
| q4 | Какие три профиля перечислены в Day12? | true | true | false | true | true | 0 | 100 | 75 | 35 |
| q5 | Какой финальный stage и статус паузы в Day13? | true | true | true | true | false | 100 | 100 | 100 | 100 |
| q6 | Сколько инвариантов в Day14 и прошёл ли конфликтный кейс? | true | true | true | true | true | 0 | 0 | 100 | 20 |
| q7 | Какой переход в Day15 блокируется без валидации? | true | true | true | true | true | 0 | 100 | 100 | 40 |
| q8 | Какие инструменты обнаружены в Day16? | true | true | true | true | true | 0 | 100 | 100 | 40 |
| q9 | Какой todo id вернул create_todo в Day17? | true | true | true | false | false | 50 | 100 | 75 | 65 |
| q10 | Сколько серверов и шагов в Day20 orchestration? | true | true | true | false | false | 100 | 100 | 75 | 95 |

## Aggregate
- sources in answers: `10/10`
- quotes in answers: `10/10`
- verbatim quote match: `9/10`
- answer meaning matches quotes: `7/10`
- unknown mode triggered: `5`
- avg answer score: `40`
- avg source score: `70`
- avg strict score: `90`
- avg total score: `56`

Conclusion: day24 returns structured grounded responses (answer + sources + quotes) and enforces 'Не знаю' when context relevance is weak.
