import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { after, describe, test } from 'node:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import type { ServerDeps } from '../../scope/mcp/server.js';
import { createServer, detectProjectRoot } from '../../scope/mcp/server.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type MockIndexer = {
  index: (path: string) => Promise<{
    filesIndexed: number;
    chunksIndexed: number;
    durationMs: number;
    errors: { file: string; error: string }[];
    filesDeleted: number;
    filesUnchanged: number;
  }>;
  search: (
    query: string,
    topK?: number,
  ) => Promise<
    {
      filePath: string;
      startLine: number;
      endLine: number;
      body: string;
      score: number;
      language: string;
      id: string;
      header: string;
    }[]
  >;
  clear: (path: string) => Promise<void>;
};

function makeMockIndexer(overrides?: Partial<MockIndexer>): MockIndexer {
  return {
    index: async (_path) => ({
      filesIndexed: 3,
      chunksIndexed: 10,
      durationMs: 50,
      errors: [],
      filesDeleted: 0,
      filesUnchanged: 0,
    }),
    search: async () => [],
    clear: async () => {},
    ...overrides,
  };
}

function makeMockDeps(overrides?: Partial<ServerDeps>): ServerDeps {
  const mockIndexer = makeMockIndexer();
  return {
    resolveConfig: async (_path) => ({
      indexDir: '/mock/index',
      modelsDir: '/mock/models',
      grammarsDir: '/mock/grammars',
      embedder: 'transformers' as const,
      openaiModel: 'text-embedding-3-small',
      ollamaBaseUrl: 'http://localhost:11434',
      ollamaModel: 'nomic-embed-text',
      ollamaDimensions: 768,
      transformersModel: 'jinaai/jina-embeddings-v2-base-code',
      chunkMaxLines: 50,
      chunkOverlap: 0.2,
      chunkStrategy: 'function' as const,
      chunkMaxTokens: 512,
      concurrency: 10,
      batchSize: 32,
      searchTopK: 5,
      watchEnabled: false,
      watchDebounceMs: 2000,
      ignorePatterns: [],
    }),
    createIndexer: async () => mockIndexer as never,
    checkSetupStatus: async () => ({ modelReady: true, grammarsMissing: [] }),
    runSetup: async () => {},
    startWatcher: () => {},
    ...overrides,
  };
}

async function makeTestClient(
  deps: ServerDeps = {},
): Promise<{ client: Client; close: () => Promise<void> }> {
  const server = createServer(deps);
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  const client = new Client({ name: 'test-client', version: '1.0.0' }, { capabilities: {} });
  await client.connect(clientTransport);
  return {
    client,
    close: async () => {
      await client.close();
    },
  };
}

// ---------------------------------------------------------------------------
// 1. detectProjectRoot: SCOPE_PROJECT_PATH env var
// ---------------------------------------------------------------------------

describe('detectProjectRoot', () => {
  test('uses SCOPE_PROJECT_PATH env var when set', async () => {
    const original = process.env.SCOPE_PROJECT_PATH;
    process.env.SCOPE_PROJECT_PATH = '/custom/project/path';
    try {
      const result = await detectProjectRoot();
      assert.equal(result, '/custom/project/path');
    } finally {
      if (original === undefined) delete process.env.SCOPE_PROJECT_PATH;
      else process.env.SCOPE_PROJECT_PATH = original;
    }
  });

  // 2. detectProjectRoot: walks up to find .git
  test('walks up to find .git directory', async () => {
    const original = process.env.SCOPE_PROJECT_PATH;
    delete process.env.SCOPE_PROJECT_PATH;
    try {
      const result = await detectProjectRoot();
      // Running from the workbench repo — should find .git
      assert.ok(existsSync(join(result, '.git')), `Expected .git at ${result}`);
    } finally {
      if (original !== undefined) process.env.SCOPE_PROJECT_PATH = original;
    }
  });

  // 3. detectProjectRoot: falls back to cwd
  test('falls back to cwd when no .git found', async () => {
    const original = process.env.SCOPE_PROJECT_PATH;
    delete process.env.SCOPE_PROJECT_PATH;
    const origCwd = process.cwd;
    // Override cwd to filesystem root — no .git will be found walking up
    process.cwd = () => '/';
    try {
      const result = await detectProjectRoot();
      assert.equal(result, '/');
    } finally {
      process.cwd = origCwd;
      if (original !== undefined) process.env.SCOPE_PROJECT_PATH = original;
    }
  });
});

// ---------------------------------------------------------------------------
// 4. Tool list: server exposes all 4 tools
// ---------------------------------------------------------------------------

test('server exposes all 4 tools', async () => {
  const { client, close } = await makeTestClient(makeMockDeps());
  after(close);

  const { tools } = await client.listTools();
  const names = tools.map((t) => t.name);

  assert.ok(names.includes('index_codebase'), 'missing index_codebase');
  assert.ok(names.includes('search_code'), 'missing search_code');
  assert.ok(names.includes('clear_index'), 'missing clear_index');
  assert.ok(names.includes('get_indexing_status'), 'missing get_indexing_status');
  assert.equal(names.length, 4);
});

// ---------------------------------------------------------------------------
// 5. get_indexing_status: returns correct shape
// ---------------------------------------------------------------------------

test('get_indexing_status returns correct shape with mocked core', async () => {
  const deps = makeMockDeps({
    checkSetupStatus: async () => ({
      modelReady: true,
      grammarsMissing: ['tree-sitter-ruby.wasm'],
    }),
  });
  const { client, close } = await makeTestClient(deps);
  after(close);

  const result = await client.callTool({ name: 'get_indexing_status', arguments: {} });
  assert.ok(!result.isError, `Expected no error, got: ${JSON.stringify(result.content)}`);

  const text = (result.content as { type: string; text: string }[])[0]?.text;
  const status = JSON.parse(text) as {
    indexed: boolean;
    filesCount: number;
    chunksCount: number;
    modelReady: boolean;
    grammarsMissing: string[];
  };

  assert.ok('indexed' in status, 'missing indexed');
  assert.ok('filesCount' in status, 'missing filesCount');
  assert.ok('chunksCount' in status, 'missing chunksCount');
  assert.ok('modelReady' in status, 'missing modelReady');
  assert.ok(Array.isArray(status.grammarsMissing), 'grammarsMissing should be array');
  assert.equal(status.modelReady, true);
  assert.deepEqual(status.grammarsMissing, ['tree-sitter-ruby.wasm']);
  assert.equal(status.indexed, false); // no index_codebase called yet
});

// ---------------------------------------------------------------------------
// 6. search_code: validates required query param
// ---------------------------------------------------------------------------

test('search_code returns error when query is missing', async () => {
  const { client, close } = await makeTestClient(makeMockDeps());
  after(close);

  const result = await client.callTool({ name: 'search_code', arguments: {} });
  assert.ok(result.isError, 'Expected isError to be true');

  const text = (result.content as { type: string; text: string }[])[0]?.text;
  const body = JSON.parse(text) as { error: string };
  assert.ok(
    body.error.toLowerCase().includes('query'),
    `Expected error about query, got: ${body.error}`,
  );
});

// ---------------------------------------------------------------------------
// 7. Watcher: NOT started when watchEnabled = false
// ---------------------------------------------------------------------------

test('watcher is not started when watchEnabled is false', async () => {
  let watcherCalled = false;
  const deps = makeMockDeps({
    startWatcher: () => {
      watcherCalled = true;
    },
    resolveConfig: async (_path) => ({
      indexDir: '/mock/index',
      modelsDir: '/mock/models',
      grammarsDir: '/mock/grammars',
      embedder: 'transformers' as const,
      openaiModel: 'text-embedding-3-small',
      ollamaBaseUrl: 'http://localhost:11434',
      ollamaModel: 'nomic-embed-text',
      ollamaDimensions: 768,
      transformersModel: 'jinaai/jina-embeddings-v2-base-code',
      chunkMaxLines: 50,
      chunkOverlap: 0.2,
      chunkStrategy: 'function' as const,
      chunkMaxTokens: 512,
      concurrency: 10,
      batchSize: 32,
      searchTopK: 5,
      watchEnabled: false, // explicitly false
      watchDebounceMs: 2000,
      ignorePatterns: [],
    }),
  });

  const { client, close } = await makeTestClient(deps);
  after(close);

  // Trigger index_codebase — watcher should NOT be started
  await client.callTool({ name: 'index_codebase', arguments: {} });

  assert.equal(watcherCalled, false, 'startWatcher should not be called when watchEnabled=false');
});
