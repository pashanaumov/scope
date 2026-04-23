import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, describe, it } from 'node:test';
import type { VectorRecord } from '../scope/core/vector-store.js';
import { VectorStore } from '../scope/core/vector-store.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DIMS = 8;
const TOP_K = 5;

// ---------------------------------------------------------------------------
// Fixtures
//
// 8-dim vectors are manually crafted so each domain is orthogonal:
//   auth     → dominant in dim 0
//   database → dominant in dim 1
//   ui       → dominant in dim 2
// embedText contains plain lowercase words that FTS can tokenise.
// ---------------------------------------------------------------------------

function makeAuthChunk(n: 1 | 2): VectorRecord {
  return {
    id: `auth.ts:${n}:10`,
    filePath: 'auth.ts',
    language: 'typescript',
    startLine: n,
    endLine: 10,
    header: `auth.ts > authenticateUser (function)`,
    body: `function authenticateUser(token: string) { /* jwt validation */ }`,
    embedText: `auth.ts > authenticateUser (function)\n\njwt authentication token login validation user password bearer`,
    sourceUrl: '',
    vector:
      n === 1 ? [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] : [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  };
}

function makeDbChunk(n: 1 | 2): VectorRecord {
  return {
    id: `database.ts:${n}:20`,
    filePath: 'database.ts',
    language: 'typescript',
    startLine: n,
    endLine: 20,
    header: `database.ts > queryPool (function)`,
    body: `function queryPool(sql: string) { /* connection pool */ }`,
    embedText: `database.ts > queryPool (function)\n\nsql query connection pool database transaction execute`,
    sourceUrl: '',
    vector:
      n === 1 ? [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] : [0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  };
}

function makeUiChunk(n: 1 | 2): VectorRecord {
  return {
    id: `ui.ts:${n}:30`,
    filePath: 'ui.ts',
    language: 'typescript',
    startLine: n,
    endLine: 30,
    header: `ui.ts > renderComponent (function)`,
    body: `function renderComponent(props: Props) { /* react render */ }`,
    embedText: `ui.ts > renderComponent (function)\n\nreact component render view button state hook`,
    sourceUrl: '',
    vector:
      n === 1 ? [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] : [0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
  };
}

const CORPUS: VectorRecord[] = [
  makeAuthChunk(1),
  makeAuthChunk(2),
  makeDbChunk(1),
  makeDbChunk(2),
  makeUiChunk(1),
  makeUiChunk(2),
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function makeTmp(): Promise<string> {
  return mkdtemp(join(tmpdir(), 'vector-store-test-'));
}

async function openStore(dir: string): Promise<VectorStore> {
  const store = new VectorStore({ indexDir: dir, searchTopK: TOP_K });
  await store.open(DIMS);
  return store;
}

// ---------------------------------------------------------------------------
// open()
// ---------------------------------------------------------------------------

describe('open', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('creates a new table if one does not exist', async () => {
    const store = await openStore(tmp);
    const n = await store.count();
    assert.equal(n, 0);
  });

  it('opens an existing table without error', async () => {
    // Re-open the same directory; table already exists from previous test.
    const store = await openStore(tmp);
    const n = await store.count();
    assert.equal(n, 0);
  });
});

// ---------------------------------------------------------------------------
// upsert() + count()
// ---------------------------------------------------------------------------

describe('upsert and count', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('upsert adds records and count() returns the right number', async () => {
    const store = await openStore(tmp);
    await store.upsert([makeAuthChunk(1), makeDbChunk(1)]);
    assert.equal(await store.count(), 2);
  });

  it('upsert with duplicate IDs replaces existing records (idempotent)', async () => {
    const store = await openStore(tmp);
    // At this point count is 2 from the previous test (same dir, re-opened).
    const updated = { ...makeAuthChunk(1), body: 'UPDATED BODY' };
    await store.upsert([updated]);
    assert.equal(await store.count(), 2, 'should still be 2 after idempotent upsert');
  });
});

// ---------------------------------------------------------------------------
// deleteByFile()
// ---------------------------------------------------------------------------

describe('deleteByFile', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('removes only records for the specified file', async () => {
    const store = await openStore(tmp);
    await store.upsert([makeAuthChunk(1), makeAuthChunk(2), makeDbChunk(1)]);
    assert.equal(await store.count(), 3);

    await store.deleteByFile('auth.ts');
    assert.equal(await store.count(), 1);
  });

  it('deleting a file that does not exist is a no-op', async () => {
    const store = await openStore(tmp);
    // count is 1 (database.ts chunk) from prior test
    await store.deleteByFile('nonexistent.ts');
    assert.equal(await store.count(), 1);
  });
});

// ---------------------------------------------------------------------------
// clear()
// ---------------------------------------------------------------------------

describe('clear', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('empties the table', async () => {
    const store = await openStore(tmp);
    await store.upsert(CORPUS);
    assert.equal(await store.count(), CORPUS.length);

    await store.clear();
    assert.equal(await store.count(), 0);
  });

  it('count() returns 0 after clear(), further upserts work normally', async () => {
    const store = await openStore(tmp);
    // still 0 from previous test
    await store.upsert([makeAuthChunk(1)]);
    assert.equal(await store.count(), 1);
  });
});

// ---------------------------------------------------------------------------
// hybridSearch — retrieval quality gate
// ---------------------------------------------------------------------------

describe('hybridSearch retrieval quality', () => {
  let tmp: string;
  let store: VectorStore;

  before(async () => {
    tmp = await makeTmp();
    store = await openStore(tmp);
    await store.upsert(CORPUS);
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('query "jwt token validation" → auth.ts in top 3', async () => {
    // Query vector is aligned with auth chunks (dim 0).
    const queryVec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    const results = await store.hybridSearch(queryVec, 'jwt token validation', 3);
    assert.ok(results.length > 0, 'should return results');
    const filePaths = results.map((r) => r.filePath);
    assert.ok(
      filePaths.includes('auth.ts'),
      `expected auth.ts in top 3, got: ${filePaths.join(', ')}`,
    );
  });

  it('query "sql connection pool" → database.ts in top 3', async () => {
    const queryVec = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    const results = await store.hybridSearch(queryVec, 'sql connection pool', 3);
    assert.ok(results.length > 0, 'should return results');
    const filePaths = results.map((r) => r.filePath);
    assert.ok(
      filePaths.includes('database.ts'),
      `expected database.ts in top 3, got: ${filePaths.join(', ')}`,
    );
  });

  it('query "react component render" → ui.ts in top 3', async () => {
    const queryVec = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    const results = await store.hybridSearch(queryVec, 'react component render', 3);
    assert.ok(results.length > 0, 'should return results');
    const filePaths = results.map((r) => r.filePath);
    assert.ok(filePaths.includes('ui.ts'), `expected ui.ts in top 3, got: ${filePaths.join(', ')}`);
  });

  it('results have valid SearchResult shape', async () => {
    const results = await store.hybridSearch(
      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      'jwt token',
      2,
    );
    assert.ok(results.length > 0);
    const r = results[0];
    assert.ok(typeof r.id === 'string' && r.id.length > 0, 'id must be non-empty string');
    assert.ok(typeof r.filePath === 'string', 'filePath must be string');
    assert.ok(typeof r.language === 'string', 'language must be string');
    assert.ok(typeof r.startLine === 'number', 'startLine must be number');
    assert.ok(typeof r.endLine === 'number', 'endLine must be number');
    assert.ok(typeof r.header === 'string', 'header must be string');
    assert.ok(typeof r.body === 'string', 'body must be string');
    assert.ok(typeof r.sourceUrl === 'string', 'sourceUrl must be string');
    assert.ok(typeof r.score === 'number', 'score must be number');
  });
});
