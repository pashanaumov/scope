#!/bin/sh
# Launcher for scope MCP server.
# Installs node_modules on first run, then starts the server via tsx.
set -e

PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PLUGIN_ROOT"

if [ ! -d "$PLUGIN_ROOT/node_modules" ]; then
  npm install --silent
fi

exec ./node_modules/.bin/tsx ./scope/mcp/index.ts
