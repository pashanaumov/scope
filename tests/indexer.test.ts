import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, stat, unlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, describe, it } from 'node:test';
import type { Chunk, Chunker } from '../scope/core/chunker.js';
import type { ScopeConfig } from '../scope/core/config-resolver.js';
import type { Embedder } from '../scope/core/embedder.js';
import type { IndexProgress } from '../scope/core/indexer.js';
import { Indexer } from '../scope/core/indexer.js';
import type { SearchResult, VectorRecord, VectorStore } from '../scope/core/vector-store.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function makeTmp(): Promise<string> {
  return mkdtemp(join(tmpdir(), 'indexer-test-'));
}

function makeChunk(filePath: string, n = 0): Chunk {
  return {
    id: `${filePath}:${n + 1}:${n + 5}`,
    filePath,
    language: 'typescript',
    startLine: n + 1,
    endLine: n + 5,
    header: filePath,
    context: '',
    body: 'const x = 1;',
    embedText: `${filePath}\n\nconst x = 1;`,
  };
}

function makeConfig(indexDir: string): ScopeConfig {
  return {
    indexDir,
    modelsDir: join(indexDir, 'models'),
    grammarsDir: join(indexDir, 'grammars'),
    embedder: 'transformers',
    openaiModel: 'text-embedding-3-small',
    ollamaBaseUrl: 'http://localhost:11434',
    ollamaModel: 'nomic-embed-text',
    ollamaDimensions: 768,
    transformersModel: 'jinaai/jina-embeddings-v2-base-code',
    chunkMaxLines: 50,
    chunkOverlap: 0.2,
    chunkStrategy: 'sliding-window',
    chunkMaxTokens: 512,
    concurrency: 4,
    batchSize: 32,
    searchTopK: 5,
    watchEnabled: false,
    watchDebounceMs: 2000,
    ignorePatterns: [],
  };
}

function makeMockEmbedder(dims = 3): Embedder {
  return {
    embed: async (texts: string[]) => texts.map(() => new Array<number>(dims).fill(0)),
    get dimensions() {
      return dims;
    },
  };
}

interface MockVectorStore {
  store: VectorStore;
  upserted: VectorRecord[];
  deleted: string[];
  searches: Array<{ vector: number[]; text: string; topK?: number }>;
  clearCalled: boolean;
  openCalled: boolean;
}

function makeMockVectorStore(): MockVectorStore {
  const upserted: VectorRecord[] = [];
  const deleted: string[] = [];
  const searches: Array<{ vector: number[]; text: string; topK?: number }> = [];
  let clearCalled = false;
  let openCalled = false;

  const store = {
    open: async (_dims: number) => {
      openCalled = true;
    },
    upsert: async (records: VectorRecord[]) => {
      upserted.push(...records);
    },
    deleteByFile: async (path: string) => {
      deleted.push(path);
    },
    hybridSearch: async (
      vector: number[],
      text: string,
      topK?: number,
    ): Promise<SearchResult[]> => {
      searches.push({ vector, text, topK });
      return [];
    },
    clear: async () => {
      clearCalled = true;
    },
    count: async () => upserted.length,
  } as unknown as VectorStore;

  return {
    store,
    get upserted() {
      return upserted;
    },
    get deleted() {
      return deleted;
    },
    get searches() {
      return searches;
    },
    get clearCalled() {
      return clearCalled;
    },
    get openCalled() {
      return openCalled;
    },
  };
}

function makeMockChunker(override?: (filePath: string, content: string) => Promise<Chunk[]>): {
  chunker: Chunker;
  calledWith: string[];
} {
  const calledWith: string[] = [];
  const chunker = {
    chunkFile: async (filePath: string, content: string): Promise<Chunk[]> => {
      calledWith.push(filePath);
      if (override) return override(filePath, content);
      return [makeChunk(filePath)];
    },
  } as unknown as Chunker;
  return { chunker, calledWith };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Indexer', () => {
  // ─── scan: chunks all files ───────────────────────────────────────────────
  describe('scan', () => {
    let projectDir: string;
    let indexDir: string;

    before(async () => {
      projectDir = await makeTmp();
      indexDir = await makeTmp();
      await writeFile(join(projectDir, 'a.ts'), 'const a = 1;');
      await writeFile(join(projectDir, 'b.ts'), 'const b = 2;');
      await writeFile(join(projectDir, 'c.ts'), 'const c = 3;');
    });
    after(async () => {
      await rm(projectDir, { recursive: true, force: true });
      await rm(indexDir, { recursive: true, force: true });
    });

    it('calls chunkFile for each file', async () => {
      const { chunker, calledWith } = makeMockChunker();
      const mvs = makeMockVectorStore();
      const indexer = new Indexer(makeConfig(indexDir), makeMockEmbedder(), mvs.store, chunker);

      const result = await indexer.index(projectDir);

      assert.equal(calledWith.length, 3);
      assert.equal(result.filesIndexed, 3);
      assert.equal(result.errors.length, 0);
    });
  });

  // ─── incremental: unchanged files skipped ────────────────────────────────
  describe('incremental', () => {
    let projectDir: string;
    let indexDir: string;

    before(async () => {
      projectDir = await makeTmp();
      indexDir = await makeTmp();
      await writeFile(join(projectDir, 'x.ts'), 'const x = 1;');
      await writeFile(join(projectDir, 'y.ts'), 'const y = 2;');
    });
    after(async () => {
      await rm(projectDir, { recursive: true, force: true });
      await rm(indexDir, { recursive: true, force: true });
    });

    it('skips unchanged files on second run', async () => {
      const { chunker, calledWith } = makeMockChunker();
      const mvs = makeMockVectorStore();
      const indexer = new Indexer(makeConfig(indexDir), makeMockEmbedder(), mvs.store, chunker);

      await indexer.index(projectDir);
      const firstCount = calledWith.length;

      // Second run: nothing changed
      await indexer.index(projectDir);
      assert.equal(calledWith.length, firstCount, 'no new chunkFile calls on second run');
    });
  });

  // ─── changed file: re-indexed ─────────────────────────────────────────────
  describe('changed file', () => {
    let projectDir: string;
    let indexDir: string;

    before(async () => {
      projectDir = await makeTmp();
      indexDir = await makeTmp();
      await writeFile(join(projectDir, 'mod.ts'), 'const v = 1;');
      await writeFile(join(projectDir, 'stable.ts'), 'const s = 0;');
    });
    after(async () => {
      await rm(projectDir, { recursive: true, force: true });
      await rm(indexDir, { recursive: true, force: true });
    });

    it('re-indexes only the modified file', async () => {
      const { chunker, calledWith } = makeMockChunker();
      const mvs = makeMockVectorStore();
      const indexer = new Indexer(makeConfig(indexDir), makeMockEmbedder(), mvs.store, chunker);

      await indexer.index(projectDir);
      calledWith.length = 0; // reset

      // Modify one file — wait a ms so mtime changes
      await new Promise((r) => setTimeout(r, 10));
      await writeFile(join(projectDir, 'mod.ts'), 'const v = 999; // changed');

      const result = await indexer.index(projectDir);
      assert.equal(calledWith.length, 1);
      assert.ok(calledWith[0]?.endsWith('mod.ts'));
      assert.equal(result.filesIndexed, 1);
    });
  });

  // ─── deleted file: deleteByFile called ───────────────────────────────────
  describe('deleted file', () => {
    let projectDir: string;
    let indexDir: string;

    before(async () => {
      projectDir = await makeTmp();
      indexDir = await makeTmp();
      await writeFile(join(projectDir, 'keep.ts'), 'const k = 1;');
      await writeFile(join(projectDir, 'gone.ts'), 'const g = 2;');
    });
    after(async () => {
      await rm(projectDir, { recursive: true, force: true });
      await rm(indexDir, { recursive: true, force: true });
    });

    it('calls deleteByFile for removed files', async () => {
      const { chunker } = makeMockChunker();
      const mvs = makeMockVectorStore();
      const indexer = new Indexer(makeConfig(indexDir), makeMockEmbedder(), mvs.store, chunker);

      await indexer.index(projectDir);

      await unlink(join(projectDir, 'gone.ts'));
      await indexer.index(projectDir);

      assert.ok(mvs.deleted.some((p) => p.endsWith('gone.ts')));
    });
  });

  // ─── error isolation ──────────────────────────────────────────────────────
  describe('error isolation', () => {
    let projectDir: string;
    let indexDir: string;

    before(async () => {
      projectDir = await makeTmp();
      indexDir = await makeTmp();
      await writeFile(join(projectDir, 'ok.ts'), 'const ok = 1;');
      await writeFile(join(projectDir, 'bad.ts'), 'const bad = 2;');
    });
    after(async () => {
      await rm(projectDir, { recursive: true, force: true });
      await rm(indexDir, { recursive: true, force: true });
    });

    it('captures error and still indexes other files', async () => {
      const { chunker } = makeMockChunker(async (filePath) => {
        if (filePath.endsWith('bad.ts')) throw new Error('chunker exploded');
        return [makeChunk(filePath)];
      });
      const mvs = makeMockVectorStore();
      const indexer = new Indexer(makeConfig(indexDir), makeMockEmbedder(), mvs.store, chunker);

      const result = await indexer.index(projectDir);

      assert.equal(result.errors.length, 1);
      assert.ok(result.errors[0]?.file.endsWith('bad.ts'));
      assert.equal(result.filesIndexed, 1);
      assert.ok(mvs.upserted.some((r) => r.filePath.endsWith('ok.ts')));
    });
  });

  // ─── ignore patterns: node_modules excluded ───────────────────────────────
  describe('ignore patterns', () => {
    let projectDir: string;
    let indexDir: string;

    before(async () => {
      projectDir = await makeTmp();
      indexDir = await makeTmp();
      await writeFile(join(projectDir, 'src.ts'), 'const src = 1;');
      await mkdir(join(projectDir, 'node_modules', 'pkg'), { recursive: true });
      await writeFile(join(projectDir, 'node_modules', 'pkg', 'index.ts'), 'module');
    });
    after(async () => {
      await rm(projectDir, { recursive: true, force: true });
      await rm(indexDir, { recursive: true, force: true });
    });

    it('does not index files in node_modules', async () => {
      const { chunker, calledWith } = makeMockChunker();
      const mvs = makeMockVectorStore();
      const indexer = new Indexer(makeConfig(indexDir), makeMockEmbedder(), mvs.store, chunker);

      await indexer.index(projectDir);

      const indexedNM = calledWith.some((p) => p.includes('node_modules'));
      assert.equal(indexedNM, false);
      assert.equal(calledWith.length, 1); // only src.ts
    });
  });

  // ─── progress callbacks ───────────────────────────────────────────────────
  describe('progress callbacks', () => {
    let projectDir: string;
    let indexDir: string;

    before(async () => {
      projectDir = await makeTmp();
      indexDir = await makeTmp();
      await writeFile(join(projectDir, 'p.ts'), 'const p = 1;');
    });
    after(async () => {
      await rm(projectDir, { recursive: true, force: true });
      await rm(indexDir, { recursive: true, force: true });
    });

    it('emits scan → embed → store → done phases', async () => {
      const { chunker } = makeMockChunker();
      const mvs = makeMockVectorStore();
      const indexer = new Indexer(makeConfig(indexDir), makeMockEmbedder(), mvs.store, chunker);

      const phases: string[] = [];
      await indexer.index(projectDir, (p: IndexProgress) => phases.push(p.phase));

      assert.ok(phases.includes('scan'));
      assert.ok(phases.includes('embed'));
      assert.ok(phases.includes('store'));
      assert.ok(phases.includes('done'));
      assert.equal(phases[0], 'scan');
      assert.equal(phases[phases.length - 1], 'done');
    });
  });

  // ─── search: delegates to hybridSearch ────────────────────────────────────
  describe('search', () => {
    let indexDir: string;

    before(async () => {
      indexDir = await makeTmp();
    });
    after(async () => {
      await rm(indexDir, { recursive: true, force: true });
    });

    it('delegates to vectorStore.hybridSearch with embedded query', async () => {
      const { chunker } = makeMockChunker();
      const embedder = makeMockEmbedder(3);
      const mvs = makeMockVectorStore();
      const indexer = new Indexer(makeConfig(indexDir), embedder, mvs.store, chunker);

      await indexer.search('find me', 7);

      assert.equal(mvs.searches.length, 1);
      assert.deepEqual(mvs.searches[0]?.vector, [0, 0, 0]);
      assert.equal(mvs.searches[0]?.text, 'find me');
      assert.equal(mvs.searches[0]?.topK, 7);
    });
  });

  // ─── clear: clears store and manifest ─────────────────────────────────────
  describe('clear', () => {
    let projectDir: string;
    let indexDir: string;

    before(async () => {
      projectDir = await makeTmp();
      indexDir = await makeTmp();
      await writeFile(join(projectDir, 'z.ts'), 'const z = 1;');
    });
    after(async () => {
      await rm(projectDir, { recursive: true, force: true });
      await rm(indexDir, { recursive: true, force: true });
    });

    it('clears vector store and deletes manifest', async () => {
      const { chunker } = makeMockChunker();
      const mvs = makeMockVectorStore();
      const indexer = new Indexer(makeConfig(indexDir), makeMockEmbedder(), mvs.store, chunker);

      await indexer.index(projectDir);

      const manifestFile = join(indexDir, 'manifest.json');
      // Manifest should exist after indexing
      await assert.doesNotReject(stat(manifestFile));

      await indexer.clear(projectDir);

      assert.ok(mvs.clearCalled, 'vectorStore.clear() should be called');
      // Manifest should be gone
      await assert.rejects(stat(manifestFile), /ENOENT/);
    });
  });
});
