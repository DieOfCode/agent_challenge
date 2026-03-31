# Day 11 Results: Memory Layers

Model: `openai/gpt-4o-mini`
Task: `day11-task-spec`

## Explicit Memory Routing
- `long.profile.name=Ivan`
- `long.profile.role=backend engineer`
- `long.preference.language=Go`
- `long.knowledge.timezone=Asia/Almaty`
- `working.goal=Собрать ТЗ для MVP приложения доставки`
- `working.constraint=Бюджет 15000 USD`
- `working.constraint=Срок запуска 8 недель`
- `working.decision=Push-провайдер OneSignal`
- `working.value.sla=99.9%`

## Stored Layers

### Short-term
```text
Short-term memory:
1. user: Мы обсудили экран логина и корзины.
2. assistant: Принято, фиксирую логин, корзину и оформление заказа.
3. user: Добавим фильтр по городу и историю заказов.
4. user: Сделай краткую сводку: имя пользователя, цель задачи, бюджет, push-провайдер и что обсуждали в последних сообщениях.
5. assistant: Имя пользователя: Иван  
Цель задачи: Собрать ТЗ для MVP приложения доставки  
Бюджет: 15000 USD  
Push-провайдер: OneSignal  
Обсужденные элементы: экран логина, корзина, фильтр по городу, история заказов.
```

### Working
```text
Working memory (task: day11-task-spec):
- goal: Собрать ТЗ для MVP приложения доставки
- constraints: Бюджет 15000 USD | Срок запуска 8 недель
- decisions: Push-провайдер OneSignal
- values: sla=99.9%
```

### Long-term
```text
Long-term memory:
- profile: name=Ivan | role=backend engineer
- preferences: language=Go
- knowledge: timezone=Asia/Almaty
```

## Same Question Comparison
Question:
```text
Сделай краткую сводку: имя пользователя, цель задачи, бюджет, push-провайдер и что обсуждали в последних сообщениях.
```

Without memory layers (short-term only):
```text
- **Имя пользователя:** Не указано.
- **Цель задачи:** Разработка функционала для экрана логина, корзины, фильтра по городу и истории заказов.
- **Бюджет:** Не указан.
- **Push-провайдер:** Не указан.
- **Обсуждение:** Экран логина, корзина, добавление фильтра по городу и истории заказов.
```

With memory layers:
```text
Имя пользователя: Иван  
Цель задачи: Собрать ТЗ для MVP приложения доставки  
Бюджет: 15000 USD  
Push-провайдер: OneSignal  
Обсужденные элементы: экран логина, корзина, фильтр по городу, история заказов.
```

## Metrics
- quality without layers: `2/6`
- quality with layers: `4/6`
- prompt tokens without layers: `113`
- prompt tokens with layers: `234`
- total tokens without layers: `200`
- total tokens with layers: `296`

Conclusion: memory layers improve retrieval of persistent and task-level facts, while short-term memory remains focused on recent dialogue.
