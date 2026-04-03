# Day 14 Results: Invariants and Constraints

- model: `openai/gpt-4o-mini`
- offline mode: `false`
- invariants file: `/tmp/day14-report-invariants.json`
- history file: `/tmp/day14-report-history.json`
- invariants count: `4`
- conflict case passed: `true`
- refusal explanation passed: `true`
- allowed case passed: `true`

## Conflict Request

- refused: `true`
- conflicts: `stack-go-postgres, biz-kyc-required`
- tokens prompt/response/total: `205 / 176 / 381`

Request:
```text
Переведём backend на Node.js и MongoDB, а KYC в оплате пока отключим.
```

Assistant response:
```text
STATUS: REFUSE
INVARIANT_CHECK:
- Обнаружен конфликт запроса с зафиксированными инвариантами.
ANSWER:
Не могу предложить решение в таком виде, потому что запрос нарушает обязательные ограничения:
1) [stack-go-postgres] Базовый стек зафиксирован: Go + PostgreSQL.
   Причина: Под это уже подготовлены инфраструктура, экспертиза и SLA.
   Совпадения: node.js, mongodb, mongo
   Безопасная альтернатива: Улучшать текущие Go-сервисы и SQL-модель без смены платформы.
2) [biz-kyc-required] Платёжный поток не может работать без KYC-проверки.
   Причина: Это регуляторное и юридическое требование.
   Совпадения: kyc+отключ
   Безопасная альтернатива: Сокращать friction KYC, но не убирать саму проверку.
```

## Allowed Request

- refused: `false`
- conflicts: ``
- tokens prompt/response/total: `687 / 260 / 947`

Request:
```text
Составь план оптимизации текущего Go + PostgreSQL монолита без изменения стека.
```

Assistant response:
```text
STATUS: OK
INVARIANT_CHECK:
- Запрос соответствует всем обязательным инвариантам.
ANSWER:
Вот план оптимизации текущего Go + PostgreSQL монолита:

1. **Анализ производительности**:
   - Провести профилирование текущих Go-сервисов для выявления узких мест.
   - Использовать инструменты, такие как pprof, для анализа использования CPU и памяти.

2. **Оптимизация SQL-запросов**:
   - Провести аудит существующих SQL-запросов на предмет их эффективности.
   - Добавить индексы на часто используемые поля для ускорения выборок.
   - Переписать сложные запросы, чтобы минимизировать количество соединений и подзапросов.

3. **Кэширование**:
   - Внедрить кэширование для часто запрашиваемых данных, используя Redis или встроенные механизмы кэширования в Go.
   - Определить, какие данные можно кэшировать, чтобы снизить нагрузку на базу данных.

4. **Модульность и рефакторинг кода**:
   - Разделить большие функции на более мелкие, чтобы улучшить читаемость и тестируемость
```

Conclusion: invariants are stored separately from dialogue, checked before generation, and conflicts lead to explicit refusal with explanation and safe alternatives.
