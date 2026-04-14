# Day 21 Results: Document Indexing

- documents indexed: `55`
- total chars: `544813`
- approx pages: `272.4`
- embedding model: `openai/text-embedding-3-small`
- fixed index: `DAY21_INDEX_fixed.json`
- structured index: `DAY21_INDEX_structured.json`

## Fixed Chunking Strategy
- chunks: `200`
- avg chunk chars: `1132`
- max chunk chars: `1200`
- approx tokens: `56636`
- embedding dim: `1536`
- embedding usage total tokens: `56250`

## Structured Chunking Strategy
- chunks: `200`
- avg chunk chars: `855`
- max chunk chars: `2000`
- approx tokens: `42836`
- embedding dim: `1536`
- embedding usage total tokens: `42019`

## Comparison
- fixed chunks: `200` | structured chunks: `200`
- fixed avg chunk chars: `1132` | structured avg chunk chars: `855`
- fixed max chunk chars: `1200` | structured max chunk chars: `2000`
- fixed approx tokens: `56636` | structured approx tokens: `42836`

Conclusion: both strategies produced embeddings with metadata; fixed chunking is uniform, while structured chunking preserves section boundaries.
