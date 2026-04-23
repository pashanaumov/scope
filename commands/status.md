---
description: Check indexing status and model readiness
allowed-tools: mcp__scope__get_indexing_status
---

Check whether the current project has been indexed and when.

Call the `get_indexing_status` MCP tool with `path` set to the current working directory. Report: indexed (yes/no), file count, chunk count, last indexed timestamp (human-readable), model ready status, any missing grammars. If a job is active, report its phase and progress (filesTotal/filesDone, chunksTotal/chunksDone). If a job failed, show the error.
