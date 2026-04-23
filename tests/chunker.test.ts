import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { Chunker } from '../scope/core/chunker.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeChunker(
  overrides: Partial<{
    chunkMaxLines: number;
    chunkOverlap: number;
    chunkStrategy: 'sliding-window' | 'function';
    chunkMaxTokens: number;
    grammarsDir: string;
  }> = {},
) {
  return new Chunker({
    chunkMaxLines: overrides.chunkMaxLines ?? 50,
    chunkOverlap: overrides.chunkOverlap ?? 0.2,
    chunkStrategy: overrides.chunkStrategy ?? 'sliding-window',
    chunkMaxTokens: overrides.chunkMaxTokens ?? 512,
    grammarsDir: overrides.grammarsDir ?? '/nonexistent-grammars',
  });
}

function makeLines(n: number): string {
  return Array.from({ length: n }, (_, i) => `line ${i + 1}`).join('\n');
}

// ---------------------------------------------------------------------------
// 1. Language detection (via chunkFile returning correct language field)
// ---------------------------------------------------------------------------

describe('language detection', () => {
  it('.ts → typescript', async () => {
    const chunker = makeChunker();
    const chunks = await chunker.chunkFile('src/foo.ts', 'const x = 1;\n');
    assert.equal(chunks[0]?.language, 'typescript');
  });

  it('.py → python', async () => {
    const chunker = makeChunker();
    const chunks = await chunker.chunkFile('src/foo.py', 'x = 1\n');
    assert.equal(chunks[0]?.language, 'python');
  });

  it('unknown extension → unknown', async () => {
    const chunker = makeChunker();
    const chunks = await chunker.chunkFile('src/foo.xyz', 'hello\n');
    assert.equal(chunks[0]?.language, 'unknown');
  });

  it('.rs → rust', async () => {
    const chunker = makeChunker();
    const chunks = await chunker.chunkFile('src/main.rs', 'fn main() {}\n');
    assert.equal(chunks[0]?.language, 'rust');
  });

  it('.go → go', async () => {
    const chunker = makeChunker();
    const chunks = await chunker.chunkFile('src/main.go', 'package main\n');
    assert.equal(chunks[0]?.language, 'go');
  });
});

// ---------------------------------------------------------------------------
// 2. Ignored files
// ---------------------------------------------------------------------------

describe('ignored files', () => {
  it('*.d.ts → empty array', async () => {
    const chunker = makeChunker();
    const chunks = await chunker.chunkFile('types/foo.d.ts', 'export type Foo = string;');
    assert.deepEqual(chunks, []);
  });

  it('*.min.js → empty array', async () => {
    const chunker = makeChunker();
    const chunks = await chunker.chunkFile('dist/bundle.min.js', 'var x=1;');
    assert.deepEqual(chunks, []);
  });

  it('*.map → empty array', async () => {
    const chunker = makeChunker();
    const chunks = await chunker.chunkFile('dist/bundle.js.map', '{"version":3}');
    assert.deepEqual(chunks, []);
  });
});

// ---------------------------------------------------------------------------
// 3. Sliding-window strategy
// ---------------------------------------------------------------------------

describe('sliding-window strategy', () => {
  it('small file (3 lines) → single chunk', async () => {
    const chunker = makeChunker({ chunkMaxLines: 50 });
    const content = 'line 1\nline 2\nline 3';
    const chunks = await chunker.chunkFile('src/small.ts', content);
    assert.equal(chunks.length, 1);
    assert.equal(chunks[0].startLine, 1);
    assert.equal(chunks[0].endLine, 3);
  });

  it('file with exactly chunkMaxLines lines → 1 chunk', async () => {
    const chunker = makeChunker({ chunkMaxLines: 10, chunkOverlap: 0.2 });
    const content = makeLines(10);
    const chunks = await chunker.chunkFile('src/exact.ts', content);
    assert.equal(chunks.length, 1);
    assert.equal(chunks[0].startLine, 1);
    assert.equal(chunks[0].endLine, 10);
  });

  it('file with chunkMaxLines + 1 lines → 2 chunks with overlap', async () => {
    // chunkMaxLines=10, overlap=0.2 → overlapLines=2, step=8
    // chunk1: lines 1-10, chunk2: lines 9-11 (start=8, end=11 but file has 11 lines)
    const chunker = makeChunker({ chunkMaxLines: 10, chunkOverlap: 0.2 });
    const content = makeLines(11);
    const chunks = await chunker.chunkFile('src/overlap.ts', content);
    assert.equal(chunks.length, 2);
    assert.equal(chunks[0].startLine, 1);
    assert.equal(chunks[0].endLine, 10);
    assert.equal(chunks[1].startLine, 9);
    assert.equal(chunks[1].endLine, 11);
  });

  it('embedText = header + "\\n\\n" + body (no context for sliding-window)', async () => {
    const chunker = makeChunker({ chunkMaxLines: 50 });
    const content = 'hello\nworld';
    const chunks = await chunker.chunkFile('src/file.ts', content);
    assert.equal(chunks.length, 1);
    const expected = 'src/file.ts\n\nhello\nworld';
    assert.equal(chunks[0].embedText, expected);
  });

  it('context is empty string for sliding-window', async () => {
    const chunker = makeChunker({ chunkMaxLines: 50 });
    const chunks = await chunker.chunkFile('src/file.ts', 'a\nb\nc');
    assert.equal(chunks[0].context, '');
  });

  it('header is just the file path for sliding-window', async () => {
    const chunker = makeChunker({ chunkMaxLines: 50 });
    const chunks = await chunker.chunkFile('src/file.ts', 'a\nb\nc');
    assert.equal(chunks[0].header, 'src/file.ts');
  });
});

// ---------------------------------------------------------------------------
// 4. Token limit enforcement
// ---------------------------------------------------------------------------

describe('token limit', () => {
  it('oversized chunk embedText is truncated to chunkMaxTokens * 4 chars', async () => {
    // chunkMaxTokens=10 → maxChars=40
    const chunker = makeChunker({ chunkMaxTokens: 10, chunkMaxLines: 200 });
    // Body that is definitely > 40 chars
    const longBody = 'x'.repeat(200);
    const chunks = await chunker.chunkFile('src/big.ts', longBody);
    assert.equal(chunks.length, 1);
    assert.ok(chunks[0].embedText.length <= 10 * 4);
  });

  it('non-oversized chunk is not truncated', async () => {
    const chunker = makeChunker({ chunkMaxTokens: 512, chunkMaxLines: 200 });
    const content = 'short content';
    const chunks = await chunker.chunkFile('src/small.ts', content);
    assert.equal(chunks.length, 1);
    assert.ok(chunks[0].embedText.length < 512 * 4);
    assert.ok(chunks[0].embedText.includes('short content'));
  });
});

// ---------------------------------------------------------------------------
// 5. Chunk ID stability
// ---------------------------------------------------------------------------

describe('chunk ID stability', () => {
  it('same file + lines → same chunk ID', async () => {
    const chunker = makeChunker({ chunkMaxLines: 50 });
    const content = 'line 1\nline 2\nline 3';

    const chunks1 = await chunker.chunkFile('src/file.ts', content);
    const chunks2 = await chunker.chunkFile('src/file.ts', content);

    assert.equal(chunks1[0].id, chunks2[0].id);
  });

  it('chunk ID format is <path>:<startLine>:<endLine>', async () => {
    const chunker = makeChunker({ chunkMaxLines: 50 });
    const content = 'line 1\nline 2\nline 3';
    const chunks = await chunker.chunkFile('src/file.ts', content);
    assert.equal(chunks[0].id, 'src/file.ts:1:3');
  });

  it('different files produce different IDs', async () => {
    const chunker = makeChunker({ chunkMaxLines: 50 });
    const content = 'line 1\nline 2';
    const [a] = await chunker.chunkFile('src/a.ts', content);
    const [b] = await chunker.chunkFile('src/b.ts', content);
    assert.notEqual(a.id, b.id);
  });
});
