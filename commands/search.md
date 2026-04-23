---
description: Search the indexed codebase with semantic + keyword hybrid search
argument-hint: <query>
allowed-tools: mcp__scope__search_code
---

Search the codebase for code relevant to the query.

Arguments: $ARGUMENTS

Use the `search_code` MCP tool with the provided query and `path` set to the current working directory. Return the top results with file paths, line ranges, and relevant code snippets.
