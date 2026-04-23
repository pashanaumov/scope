import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import type { ScopeConfig } from '../scope/core/config-resolver.js';
import {
  createEmbedder,
  OllamaEmbedder,
  OpenAIEmbedder,
  TransformersEmbedder,
} from '../scope/core/embedder.js';

// ---------------------------------------------------------------------------
// Shared minimal config fixture
// ---------------------------------------------------------------------------

function baseConfig(overrides: Partial<ScopeConfig> = {}): ScopeConfig {
  return {
    indexDir: '/tmp/index',
    modelsDir: '/tmp/models',
    grammarsDir: '/tmp/grammars',
    embedder: 'transformers',
    openaiApiKey: 'sk-test',
    openaiModel: 'text-embedding-3-small',
    ollamaBaseUrl: 'http://localhost:11434',
    ollamaModel: 'nomic-embed-text',
    ollamaDimensions: 768,
    transformersModel: 'jinaai/jina-embeddings-v2-base-code',
    chunkMaxLines: 50,
    chunkOverlap: 0.2,
    chunkStrategy: 'function',
    chunkMaxTokens: 512,
    concurrency: 10,
    batchSize: 32,
    searchTopK: 5,
    watchEnabled: false,
    watchDebounceMs: 2000,
    ignorePatterns: [],
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// 1. createEmbedder factory
// ---------------------------------------------------------------------------

describe('createEmbedder', () => {
  it('returns TransformersEmbedder when embedder=transformers', () => {
    const e = createEmbedder(baseConfig({ embedder: 'transformers' }));
    assert.ok(e instanceof TransformersEmbedder);
  });

  it('returns OpenAIEmbedder when embedder=openai', () => {
    const e = createEmbedder(baseConfig({ embedder: 'openai' }));
    assert.ok(e instanceof OpenAIEmbedder);
  });

  it('returns OllamaEmbedder when embedder=ollama', () => {
    const e = createEmbedder(baseConfig({ embedder: 'ollama' }));
    assert.ok(e instanceof OllamaEmbedder);
  });
});

// ---------------------------------------------------------------------------
// 2. OpenAIEmbedder — injected mock, batching
// ---------------------------------------------------------------------------

describe('OpenAIEmbedder', () => {
  function makeCreateFn(calls: { count: number }) {
    return async (opts: { model: string; input: string[] }) => {
      calls.count++;
      return {
        data: opts.input.map((_, i) => ({ embedding: new Array(1536).fill(i * 0.01) })),
      };
    };
  }

  it('dimensions returns 1536', () => {
    const e = new OpenAIEmbedder({
      openaiApiKey: 'sk-test',
      openaiModel: 'text-embedding-3-small',
      batchSize: 32,
    });
    assert.equal(e.dimensions, 1536);
  });

  it('32 texts → 1 batch (batchSize=32)', async () => {
    const calls = { count: 0 };
    const embedder = new OpenAIEmbedder(
      { openaiApiKey: 'sk-test', openaiModel: 'text-embedding-3-small', batchSize: 32 },
      makeCreateFn(calls),
    );
    const texts = Array.from({ length: 32 }, (_, i) => `text ${i}`);
    const result = await embedder.embed(texts);
    assert.equal(result.length, 32);
    assert.equal(calls.count, 1);
  });

  it('65 texts → 3 batches (batchSize=32)', async () => {
    const calls = { count: 0 };
    const embedder = new OpenAIEmbedder(
      { openaiApiKey: 'sk-test', openaiModel: 'text-embedding-3-small', batchSize: 32 },
      makeCreateFn(calls),
    );
    const texts = Array.from({ length: 65 }, (_, i) => `text ${i}`);
    const result = await embedder.embed(texts);
    assert.equal(result.length, 65);
    assert.equal(calls.count, 3); // batches: 32, 32, 1
  });
});

// ---------------------------------------------------------------------------
// 3. OllamaEmbedder — mocked fetch
// ---------------------------------------------------------------------------

describe('OllamaEmbedder', () => {
  const DIMS = 512;

  function makeFakeFetch(calls: { urls: string[] }) {
    return async (url: string, _opts: unknown) => {
      calls.urls.push(url as string);
      return {
        json: async () => ({ embedding: new Array(DIMS).fill(0.1) }),
      };
    };
  }

  it('calls correct endpoint and extracts dimensions', async () => {
    const calls = { urls: [] as string[] };
    const origFetch = globalThis.fetch;
    globalThis.fetch = makeFakeFetch(calls) as typeof fetch;

    try {
      const embedder = new OllamaEmbedder({
        ollamaBaseUrl: 'http://localhost:11434',
        ollamaModel: 'nomic-embed-text',
        ollamaDimensions: 768,
        batchSize: 32,
      });
      const result = await embedder.embed(['hello', 'world']);
      assert.equal(result.length, 2);
      assert.equal(embedder.dimensions, DIMS);
      assert.ok(calls.urls.every((u) => u === 'http://localhost:11434/api/embeddings'));
      assert.equal(calls.urls.length, 2);
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('dimensions returns configured value before first embed call', () => {
    const embedder = new OllamaEmbedder({
      ollamaBaseUrl: 'http://localhost:11434',
      ollamaModel: 'nomic-embed-text',
      ollamaDimensions: 768,
      batchSize: 32,
    });
    assert.equal(embedder.dimensions, 768);
  });
});

// ---------------------------------------------------------------------------
// 4. TransformersEmbedder — no model loaded, just structural checks
// ---------------------------------------------------------------------------

describe('TransformersEmbedder', () => {
  it('is instantiable without errors', () => {
    const embedder = new TransformersEmbedder({
      transformersModel: 'jinaai/jina-embeddings-v2-base-code',
      modelsDir: '/tmp/models',
      batchSize: 32,
    });
    assert.ok(embedder instanceof TransformersEmbedder);
  });

  it('dimensions returns 768', () => {
    const embedder = new TransformersEmbedder({
      transformersModel: 'jinaai/jina-embeddings-v2-base-code',
      modelsDir: '/tmp/models',
      batchSize: 32,
    });
    assert.equal(embedder.dimensions, 768);
  });
});
