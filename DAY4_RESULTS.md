# Day 4 Results: Temperature Comparison

## Prompt
`Ответь в одну строку формата: result=<число>; reason=<до 12 слов>; metaphor=<до 7 слов>. Сколько уникальных перестановок у слова LEVEL?`

## Responses

### temperature = 0.0
`result=30; reason=Слово содержит повторяющиеся буквы L и E; metaphor=Слово как замок с повторяющимися ключами.`

### temperature = 0.7
`result=30; reason=3 уникальные буквы, 2 одинаковые L; metaphor=Слово танцует разными шагами.`

### temperature = 1.2
`result=30; reason=5 букв, включая повторяющиеся буквы L и E; metaphor=Композиция внутри симфонии.`

## Comparison (from CLI)
- `temperature=0.0 -> accuracy=100, creativity=75, diversity=64`
- `temperature=0.7 -> accuracy=100, creativity=70, diversity=71`
- `temperature=1.2 -> accuracy=100, creativity=70, diversity=66`

## Conclusions
- `temperature=0.0`: best for precise, deterministic, repeatable tasks.
- `temperature=0.7`: best balance for daily work (quality + natural language).
- `temperature=1.2`: best for brainstorming and unconventional phrasing.

## Video
Use [DAY4_VIDEO_SCRIPT.md](./DAY4_VIDEO_SCRIPT.md) as the narration and demo plan.
