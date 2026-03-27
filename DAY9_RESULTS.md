# Day 9 Results: Context Compression

Model: `openai/gpt-4o-mini`

## without_compression

- compression: `disabled`
- quality: `9/9`
- tokens prompt/response/total: `18765 / 415 / 19180`
- cost: `$0.003064`
- quality details: `name: ok, stack-go: ok, stack-postgresql: ok, project: ok, deadline: ok, region: ok, port: ok, budget: ok, contact: ok`

Final answer sample:
```text
Вот JSON с запомненными фактами:

'''json
{
  "name": "Иван",
  "stack": ["Go", "PostgreSQL"],
  "project": "Atlas",
  "deadline": "2026-04-15",
  "region": "eu-west-1",
  "port": 7443,
  "budget_usd": 12000,
  "contact": "Anna"
}
'''
```

## with_compression

- compression: `enabled keep_last=6 summary_every=10`
- quality: `8/9`
- tokens prompt/response/total: `15899 / 412 / 16311`
- cost: `$0.002632`
- quality details: `name: ok, stack-go: ok, stack-postgresql: ok, project: ok, deadline: ok, region: ok, port: miss, budget: ok, contact: ok`

Final answer sample:
```text
'''json
{
  "name": "Иван",
  "stack": ["Go", "PostgreSQL"],
  "project": "Atlas",
  "deadline": "2026-04-15",
  "region": "eu-west-1",
  "port": null,
  "budget_usd": 12000,
  "contact": "Anna"
}
'''
```

## Comparison
- quality without compression: `9/9`
- quality with compression: `8/9`
- prompt tokens without: `18765`
- prompt tokens with: `15899`
- prompt tokens saved: `2866`
- total tokens without: `19180`
- total tokens with: `16311`
- total tokens saved: `2869`
- cost without: `$0.003064`
- cost with: `$0.002632`

Conclusion: keeping recent turns raw and summarizing older chunks reduces token usage while preserving most task-critical facts.
