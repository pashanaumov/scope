---
description: Index the current codebase for semantic search
argument-hint: [path] [--force]
allowed-tools: mcp__scope__index_codebase
---

Index the codebase so it can be searched semantically.

Arguments: $ARGUMENTS

Parse any path or --force flag from the arguments, then call the `index_codebase` MCP tool. Report progress and final stats (files indexed, chunks, duration, any errors).
