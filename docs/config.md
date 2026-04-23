# Config reference

All options can be set in `~/.config/scope/config.json` (global) or `.scope.json` (project root). CLI flags take highest priority. `ignorePatterns` is merged (union) across all sources.

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `embedder` | `"transformers" \| "openai" \| "ollama"` | `"transformers"` | Embedding backend |
| `openaiApiKey` | string | `OPENAI_API_KEY` env | OpenAI key |
| `openaiModel` | string | `"text-embedding-3-small"` | OpenAI model |
| `ollamaBaseUrl` | string | `"http://localhost:11434"` | Ollama base URL |
| `ollamaModel` | string | `"nomic-embed-text"` | Ollama model |
| `ollamaDimensions` | number | `768` | Ollama embedding dimensions |
| `transformersModel` | string | `"jinaai/jina-embeddings-v2-base-code"` | ONNX model |
| `chunkMaxLines` | number | `50` | Max lines per chunk |
| `chunkOverlap` | number | `0.2` | Sliding-window overlap ratio |
| `chunkStrategy` | `"function" \| "sliding-window"` | `"function"` | Chunking strategy |
| `chunkMaxTokens` | number | `512` | Max tokens per chunk (hard truncation) |
| `concurrency` | number | `10` | Parallel file processing |
| `batchSize` | number | `32` | Embedding batch size |
| `searchTopK` | number | `5` | Default top-k results |
| `watchEnabled` | boolean | `false` | Auto-reindex on file changes |
| `watchDebounceMs` | number | `2000` | Watcher debounce delay |
| `ignorePatterns` | string[] | see defaults | Glob patterns to exclude |

## Default ignore patterns

`node_modules`, `.git`, `dist`, `build`, `out`, `__pycache__`, `.next`, `.nuxt`, `coverage`, `*.min.js`, `*.d.ts`, `*.map`, `*.lock`, `*.snap`, `.DS_Store`

## Example `.scope.json`

```json
{
  "embedder": "openai",
  "searchTopK": 10,
  "ignorePatterns": ["fixtures/", "*.generated.ts"]
}
```
