#!/bin/sh
# Launcher for scope MCP server.
# Installs node_modules on first run, then starts the server via tsx.
set -e

PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$PLUGIN_ROOT/node_modules" ]; then
  npm --prefix "$PLUGIN_ROOT" install --silent
fi

exec npx --prefix "$PLUGIN_ROOT" tsx "$PLUGIN_ROOT/scope/mcp/index.ts"
