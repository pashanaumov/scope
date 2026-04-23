---
name: scope
description: Semantic code search — index your codebase and find relevant code with hybrid vector + keyword search (ONNX embeddings + LanceDB).
allowed-tools: mcp__scope__index_codebase, mcp__scope__search_code, mcp__scope__get_indexing_status, mcp__scope__clear_index
---

You have access to the scope semantic search plugin. Use it proactively when you need to find code.

## Available MCP tools

- **`index_codebase`** — Index the project. Call once before searching, or when code has changed significantly.
- **`search_code`** — Search with a natural-language query. Returns top-k chunks with file path, line range, and content.
- **`get_indexing_status`** — Check if the project is indexed, how many files/chunks, and when.
- **`clear_index`** — Wipe the index for the current project.

## When to use scope

- User asks to find code ("where is X implemented?", "find all usages of Y")
- You need context from a large codebase before editing
- You want to verify a change won't break similar patterns elsewhere

## Workflow

1. Check status with `get_indexing_status`
2. If not indexed (or stale), run `index_codebase` first
3. Search with descriptive natural-language queries
4. Use results to navigate to the right files and line ranges
