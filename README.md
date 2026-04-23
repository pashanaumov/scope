# scope

Semantic code search for Claude Code. Index your codebase with ONNX embeddings + LanceDB; search it with hybrid vector + keyword lookup via MCP.

## Install as Claude Code plugin

```bash
claude plugin marketplace add pashanaumov/scope
claude plugin install scope@scope
```

## MCP tools

| Tool | Description |
|------|-------------|
| `index_codebase` | Index a project for semantic search |
| `search_code` | Hybrid vector + keyword search |
| `get_indexing_status` | Check index freshness and model readiness |
| `clear_index` | Wipe the index for the current project |

## Slash commands

| Command | Description |
|---------|-------------|
| `/scope:index` | Index the current project |
| `/scope:search <query>` | Search the codebase |
| `/scope:status` | Show index status |
| `/scope:clear` | Clear the index |

## Config

Scope is zero-config by default. Override with:

- **Global:** `~/.config/scope/config.json`
- **Per-project:** `.scope.json` in project root

See [docs/config.md](docs/config.md) for all options.

## Embedders

| Embedder | Config | Notes |
|----------|--------|-------|
| `transformers` (default) | — | Local ONNX; downloads on first run (~100MB) |
| `openai` | `OPENAI_API_KEY` | `text-embedding-3-small` |
| `ollama` | `ollamaBaseUrl` | e.g. `nomic-embed-text` |

## Dev

```bash
npm install
npm run build      # tsc → dist/
npm test           # node:test suite (95 tests)
npm run bench      # search quality + token savings benchmark
npm run csn-bench  # CodeSearchNet NDCG@10
```

## Data paths

| Path | Purpose |
|------|---------|
| `~/.config/scope/indexes/<hash>/` | Per-project vector index |
| `~/.config/scope/models/` | ONNX model files |
| `~/.config/scope/grammars/` | tree-sitter grammars |
