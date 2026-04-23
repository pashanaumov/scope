---
description: Index the current codebase for semantic search
argument-hint: [path] [--force]
allowed-tools: mcp__scope__index_codebase, mcp__scope__get_indexing_status
---

Index the codebase so it can be searched semantically.

Arguments: $ARGUMENTS

Parse any path or --force flag from the arguments, then call the `index_codebase` MCP tool with `path` set to the current working directory. Indexing runs in the background — immediately call `get_indexing_status` (with the same path) once to confirm the job started, then report the status to the user and suggest they can check progress with `/scope:status`.
