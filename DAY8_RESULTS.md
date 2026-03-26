# Day 8 Results: Token Behavior

Model: `openai/gpt-4o-mini`

## short_dialog

- turn=1 prompt=39 response=24 total=63 cumulative_total=63 cost(turn=$0.000020 cumulative=$0.000020)
- turn=2 prompt=76 response=8 total=84 cumulative_total=147 cost(turn=$0.000016 cumulative=$0.000036)

## long_dialog

- turn=1 prompt=393 response=31 total=424 cumulative_total=424 cost(turn=$0.000078 cumulative=$0.000078)
- turn=2 ERROR: `context limit exceeded: estimated_prompt_tokens=739 max_tokens=180 context_limit=700`

## overflow_dialog

- turn=1 ERROR: `context limit exceeded: estimated_prompt_tokens=296 max_tokens=180 context_limit=320`

## Summary
- short dialog cumulative tokens: `147`
- long dialog cumulative tokens: `424`
- overflow breakage at turn `1`: `context limit exceeded: estimated_prompt_tokens=296 max_tokens=180 context_limit=320`
- Token/cost growth is visible turn-by-turn; overflow is blocked by context-limit pre-check.
