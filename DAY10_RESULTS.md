# Day 10 Results: Context Strategies (No Summary)

Model: `openai/gpt-4o-mini`

| Strategy | Quality | Stability | Prompt Tokens | Total Tokens | Cost USD | Branch Isolation |
|---|---:|---:|---:|---:|---:|---:|
| sliding_window | 9/9 | 3/4 | 9554 | 11945 | 0.002868 | - |
| sticky_facts | 7/9 | 2/4 | 10816 | 13177 | 0.003039 | - |
| branching | 9/9 | 3/4 | 26605 | 30384 | 0.006258 | 4/4 |

## sliding_window

- strategy: `sliding`
- quality: `9/9`
- stability: `3/4`
- prompt tokens: `9554`
- total tokens: `11945`
- cost: `$0.002868`
- usability: Самый простой режим, но легко теряет ранние договоренности.

Final answer sample:
```text
'''json
{
  "goal": "Разработка MVP приложения для доставки товаров.",
  "budget_usd": 15000,
  "deadline_weeks": 8,
  "backend": "Node.js или аналогичный",
  "database": "PostgreSQL или аналогичный",
  "auth": "Регистрация через email с подтверждением по OTP",
  "push_provider": "OneSignal",
  "sla": "Не ниже 99.9%",
  "reports": "Отчет по прогрессу каждую пятницу"
}
'''
```

## sticky_facts

- strategy: `facts`
- quality: `7/9`
- stability: `2/4`
- prompt tokens: `10816`
- total tokens: `13177`
- cost: `$0.003039`
- usability: Хороший баланс: важные факты стабильнее при умеренной цене.

Final answer sample:
```text
Вот финальное ТЗ в формате JSON:

'''json
{
  "goal": "Собрать ТЗ для MVP приложения доставки.",
  "budget_usd": null,  // Укажите бюджет в долларах
  "deadline_weeks": null,  // Укажите срок выполнения в неделях
  "backend": "Go",
  "database": "PostgreSQL",
  "auth": "Авторизация через email + OTP.",
  "push_provider": "OneSignal",
  "sla": "Не ниже 99.9%.",
  "reports": "Отчет по прогрессу каждую пятницу."
}
'''

Пожалуйста, заполните поля `budget_usd` и `deadline_weeks` соответствующими значениями. Если есть дополнительные изменения или уточнения, дайте знать!
```

## branching

- strategy: `branching`
- quality: `9/9`
- stability: `3/4`
- branch isolation: `4/4`
- prompt tokens: `26605`
- total tokens: `30384`
- cost: `$0.006258`
- usability: Лучший режим для альтернатив, но требует команд для управления ветками.
- branching flow: `checkpoint: spec_v1 | branches: option_a, option_b | switching: main -> option_a -> option_b`

Final answer sample:
```text
Вот финальное ТЗ в формате JSON:

'''json
{
  "goal": "Создание MVP приложения доставки.",
  "budget_usd": 15000,
  "deadline_weeks": 8,
  "backend": "Go",
  "database": "PostgreSQL",
  "auth": {
    "method": "email",
    "otp": true
  },
  "push_provider": "OneSignal",
  "sla": "99.9%",
  "reports": {
    "frequency": "еженедельно",
    "day": "пятница"
  },
  "features": {
    "user_registration": true,
    "product_selection": true,
    "order_processing": true,
    "delivery_tracking": true,
    "payment_integration": true,
    "offline_mode_for_couriers": true,
    "push_notifications": true
  }
}
'''

Этот JSON включает все ключевые аспекты вашего проекта и может быть использован для дальнейшей работы. Если нужно внести изменения или добавить дополнительные поля, дайте знать
```

## Summary
- Best quality: `sliding_window` (`9/9`, tie with `branching`)
- Best stability: `sliding_window` and `branching` (`3/4`)
- Lowest token usage: `sliding_window` (`11945` total tokens)
- Branching keeps alternatives isolated (`4/4`), but costs more tokens than linear strategies.
