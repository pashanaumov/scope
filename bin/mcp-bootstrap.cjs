#!/usr/bin/env node
// Bootstrap for scope MCP server.
// Uses __dirname to locate plugin root — no CLAUDE_PLUGIN_ROOT needed.
// Runs npm ci on first install, then spawns the server via tsx.
const { execSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const root = path.resolve(__dirname, '..');

if (!fs.existsSync(path.join(root, 'node_modules'))) {
  execSync('npm ci --silent', { cwd: root, stdio: ['pipe', 'pipe', 'inherit'] });
}

const tsx = path.join(root, 'node_modules', '.bin', 'tsx');
const entry = path.join(root, 'scope', 'mcp', 'index.ts');

const child = spawn(tsx, [entry], { cwd: root, stdio: 'inherit' });
child.on('exit', code => process.exit(code ?? 0));
