#!/usr/bin/env node
/**
 * Scope Benchmark
 * ============================
 * Measures search quality and token savings for a given project.
 *
 * Usage:
 *   node --experimental-strip-types benchmarks/benchmark.ts [project-path]
 *   node --experimental-strip-types benchmarks/benchmark.ts [project-path] --queries queries.json
 *
 * The script:
 *   1. Verifies the project is indexed (exits with a hint if not).
 *   2. Runs a set of representative queries.
 *   3. For each result, computes:
 *        - tokens served by the indexer (chunk body length ÷ 4)
 *        - tokens in the matched full file
 *        - tokens in the whole repo
 *   4. Prints a per-query table and an aggregate summary.
 */

import { glob, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import {
  Chunker,
  createEmbedder,
  Indexer,
  resolveConfig,
  type SearchResult,
  VectorStore,
  type ScopeConfig,
} from '../scope/core/index.js';

// ---------------------------------------------------------------------------
// Default queries — representative of real Claude Code usage
// ---------------------------------------------------------------------------

interface QuerySpec {
  query: string;
  /** Optional: a substring of the file path we *expect* to appear in top-k results */
  expectedFile?: string;
}

const DEFAULT_QUERIES: QuerySpec[] = [
  { query: 'how to resolve workbench config from disk', expectedFile: 'config-resolver' },
  { query: 'file manifest mtime sha256 incremental', expectedFile: 'manifest' },
  { query: 'tree-sitter chunk function sliding window', expectedFile: 'chunker' },
  { query: 'download onnx model grammar setup', expectedFile: 'setup' },
  { query: 'lancedb vector store upsert hybrid search', expectedFile: 'vector-store' },
  { query: 'mcp server detect project root tool handler', expectedFile: 'server' },
  { query: 'cli progress bar spinner index command', expectedFile: 'progress' },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Rough token estimate: 1 token ≈ 4 bytes (GPT-4 average). */
function approxTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Right-pad a string to width. */
function pad(s: string, width: number): string {
  return s.length >= width ? s : s + ' '.repeat(width - s.length);
}

/** Truncate a string with … if longer than max. */
function trunc(s: string, max: number): string {
  return s.length <= max ? s : `${s.slice(0, max - 1)}…`;
}

/** Format a number with thousands separator. */
function fmt(n: number): string {
  return n.toLocaleString('en-US');
}

/** Return a coloured string for the terminal. */
const c = {
  bold: (s: string) => `\x1b[1m${s}\x1b[0m`,
  green: (s: string) => `\x1b[32m${s}\x1b[0m`,
  yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,
  red: (s: string) => `\x1b[31m${s}\x1b[0m`,
  dim: (s: string) => `\x1b[2m${s}\x1b[0m`,
  cyan: (s: string) => `\x1b[36m${s}\x1b[0m`,
};

// ---------------------------------------------------------------------------
// Source file scanner
// ---------------------------------------------------------------------------

interface RepoStats {
  fileCount: number;
  totalChars: number;
  totalTokens: number;
  fileSizes: Map<string, number>; // absPath → char count
}

// Node 22+ built-in glob (no extra dependency)
const SOURCE_PATTERNS =
  '{**/*.ts,**/*.tsx,**/*.js,**/*.jsx,**/*.mjs,**/*.py,**/*.rs,**/*.go,**/*.java,**/*.c,**/*.cpp,**/*.rb}';

async function scanRepo(projectPath: string): Promise<RepoStats> {
  const fileIter = glob(SOURCE_PATTERNS, {
    cwd: projectPath,
    exclude: (name) => ['node_modules', 'dist', 'build', '.git', 'coverage'].includes(name),
  });

  const fileSizes = new Map<string, number>();
  let totalChars = 0;

  const readPromises: Promise<void>[] = [];
  for await (const rel of fileIter) {
    // Skip generated / binary-ish files
    if (rel.endsWith('.d.ts') || rel.endsWith('.min.js') || rel.endsWith('.map')) continue;
    const absPath = resolve(projectPath, rel);
    readPromises.push(
      readFile(absPath, 'utf-8')
        .then((content) => {
          fileSizes.set(absPath, content.length);
          totalChars += content.length;
        })
        .catch(() => {
          // skip unreadable files
        }),
    );
  }
  await Promise.all(readPromises);

  return {
    fileCount: fileSizes.size,
    totalChars,
    totalTokens: approxTokens(' '.repeat(totalChars)),
    fileSizes,
  };
}

// ---------------------------------------------------------------------------
// Per-query analysis
// ---------------------------------------------------------------------------

interface QueryResult {
  query: string;
  expectedFile: string | undefined;
  hits: SearchResult[];
  latencyMs: number;
  /** top-k chunk tokens summed */
  chunkTokens: number;
  /** full-file tokens for the top-1 matched file */
  topFileTokens: number;
  /** whether expectedFile appeared in top-3 */
  hitAtTop3: boolean | null; // null when no expectedFile given
}

async function runQuery(
  indexer: Indexer,
  spec: QuerySpec,
  topK: number,
  fileSizes: Map<string, number>,
  projectPath: string,
): Promise<QueryResult> {
  const t0 = Date.now();
  const hits = await indexer.search(spec.query, topK);
  const latencyMs = Date.now() - t0;

  const chunkTokens = hits.reduce((sum, h) => sum + approxTokens(h.body), 0);

  // Find file size for top-1 result
  let topFileTokens = 0;
  if (hits.length > 0) {
    const absPath = resolve(projectPath, hits[0].filePath);
    const chars = fileSizes.get(absPath) ?? fileSizes.get(hits[0].filePath) ?? 0;
    topFileTokens = approxTokens(' '.repeat(chars));
  }

  // Check precision — did expected file appear in top 3?
  let hitAtTop3: boolean | null = null;
  if (spec.expectedFile) {
    hitAtTop3 = hits
      .slice(0, 3)
      .some((h) => h.filePath.toLowerCase().includes(spec.expectedFile?.toLowerCase()));
  }

  return {
    query: spec.query,
    expectedFile: spec.expectedFile,
    hits,
    latencyMs,
    chunkTokens,
    topFileTokens,
    hitAtTop3,
  };
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

function printQueryResult(r: QueryResult, repoTokens: number, idx: number): void {
  const hitIcon = r.hitAtTop3 === null ? c.dim('  ') : r.hitAtTop3 ? c.green(' ✓') : c.red(' ✗');

  console.log(`\n${c.bold(c.cyan(`Query ${idx + 1}:`))} ${c.bold(r.query)}${hitIcon}`);
  console.log(c.dim(`  Latency: ${r.latencyMs} ms   Chunks returned: ${r.hits.length}`));

  if (r.hits.length === 0) {
    console.log(c.red('  No results — is the project indexed?'));
    return;
  }

  // Results table
  const COL = [35, 10, 8, 6];
  console.log(
    c.dim(
      `  ${pad('File', COL[0])} ${pad('Lines', COL[1])} ${pad('~Tokens', COL[2])} ${pad('Score', COL[3])}`,
    ),
  );
  console.log(
    c.dim(
      `  ${'─'.repeat(COL[0])} ${'─'.repeat(COL[1])} ${'─'.repeat(COL[2])} ${'─'.repeat(COL[3])}`,
    ),
  );

  for (const hit of r.hits) {
    const loc = `${trunc(hit.filePath, 33)}:${hit.startLine}`;
    const lines = `${hit.endLine - hit.startLine + 1} lines`;
    const tokens = approxTokens(hit.body).toString();
    const score = hit.score.toFixed(3);
    console.log(`  ${pad(loc, COL[0])} ${pad(lines, COL[1])} ${pad(tokens, COL[2])} ${score}`);
  }

  // Token savings
  const savedVsFile =
    r.topFileTokens > 0 ? Math.round((1 - r.chunkTokens / r.topFileTokens) * 100) : 0;
  const savedVsRepo = repoTokens > 0 ? Math.round((1 - r.chunkTokens / repoTokens) * 100) : 0;

  const savingsColor = savedVsRepo >= 90 ? c.green : savedVsRepo >= 70 ? c.yellow : c.red;

  console.log(`\n  ${c.bold('Token savings:')}`);
  console.log(
    `    vs. reading the top-matched file: ${savingsColor(`${savedVsFile}%`)} (${fmt(r.chunkTokens)} vs ${fmt(r.topFileTokens)} tokens)`,
  );
  console.log(
    `    vs. reading the whole repo:       ${savingsColor(`${savedVsRepo}%`)} (${fmt(r.chunkTokens)} vs ${fmt(repoTokens)} tokens)`,
  );
}

function printSummary(results: QueryResult[], repo: RepoStats, topK: number): void {
  console.log(`\n${'═'.repeat(70)}`);
  console.log(c.bold('BENCHMARK SUMMARY'));
  console.log('═'.repeat(70));

  console.log(
    `\nRepo:   ${repo.fileCount} source files  ·  ~${fmt(repo.totalTokens)} tokens total`,
  );
  console.log(`Search: top-${topK} chunks per query\n`);

  // Aggregate token savings
  const totalChunkTokens = results.reduce((s, r) => s + r.chunkTokens, 0);
  const avgChunkTokens = Math.round(totalChunkTokens / results.length);
  const avgSavingsVsRepo = Math.round((1 - avgChunkTokens / repo.totalTokens) * 100);
  const avgLatency = Math.round(results.reduce((s, r) => s + r.latencyMs, 0) / results.length);

  const hitCount = results.filter((r) => r.hitAtTop3 === true).length;
  const hitTotal = results.filter((r) => r.hitAtTop3 !== null).length;
  const precisionPct = hitTotal > 0 ? Math.round((hitCount / hitTotal) * 100) : null;

  console.log(
    `${c.bold(`Average tokens per search (top-${topK} chunks):`)} ${fmt(avgChunkTokens)}`,
  );
  console.log(
    `${c.bold('Average tokens (whole repo):')}                   ${fmt(repo.totalTokens)}`,
  );
  console.log(
    `${c.bold('Average token savings vs. whole repo:')}          ${c.green(`${avgSavingsVsRepo}%`)}`,
  );
  console.log(`${c.bold('Average search latency:')}                        ${avgLatency} ms`);

  if (precisionPct !== null) {
    const precColor = precisionPct >= 80 ? c.green : precisionPct >= 60 ? c.yellow : c.red;
    console.log(
      `${c.bold('Precision@3 (expected file in top 3):')}         ${precColor(`${precisionPct}%`)} (${hitCount}/${hitTotal} queries)`,
    );
  }

  // Per-query table
  console.log(`\n${c.bold('Per-query breakdown:')}`);
  const Q = [38, 10, 10, 10, 5];
  console.log(
    c.dim(
      `${pad('Query', Q[0])} ${pad('Latency', Q[1])} ${pad('Chunks tok', Q[2])} ${pad('Savings', Q[3])} P@3`,
    ),
  );
  console.log(
    c.dim(
      `${'─'.repeat(Q[0])} ${'─'.repeat(Q[1])} ${'─'.repeat(Q[2])} ${'─'.repeat(Q[3])} ${'─'.repeat(Q[4])}`,
    ),
  );

  for (const r of results) {
    const savings =
      repo.totalTokens > 0 ? `${Math.round((1 - r.chunkTokens / repo.totalTokens) * 100)}%` : 'n/a';
    const p3 = r.hitAtTop3 === null ? c.dim('n/a') : r.hitAtTop3 ? c.green('✓') : c.red('✗');
    console.log(
      `${pad(trunc(r.query, 37), Q[0])} ${pad(`${r.latencyMs} ms`, Q[1])} ${pad(fmt(r.chunkTokens), Q[2])} ${pad(savings, Q[3])} ${p3}`,
    );
  }

  console.log(`\n${'═'.repeat(70)}\n`);

  // Guidance
  if (avgSavingsVsRepo >= 90) {
    console.log(c.green('✓ Excellent token savings — the indexer is working as expected.'));
  } else if (avgSavingsVsRepo >= 70) {
    console.log(
      c.yellow('⚠ Good savings, but consider increasing searchTopK for better coverage.'),
    );
  } else {
    console.log(c.red('✗ Low savings — the repo may be too small for the effect to be visible.'));
  }

  if (precisionPct !== null && precisionPct < 70) {
    console.log(
      c.yellow('⚠ Precision@3 is below 70%. Try re-indexing with --force, or check that'),
    );
    console.log(c.yellow('  the expected files are not excluded by ignorePatterns.'));
  }

  console.log('');
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const { values, positionals } = parseArgs({
    args: process.argv.slice(2),
    options: {
      queries: { type: 'string' },
      'top-k': { type: 'string', default: '5' },
      help: { type: 'boolean', short: 'h', default: false },
    },
    allowPositionals: true,
    strict: true,
  });

  if (values.help) {
    console.log(`
scope benchmark — measure search quality and token savings

Usage:
  node --experimental-strip-types benchmarks/benchmark.ts [project-path] [options]

Options:
  --queries <path>   JSON file with array of { query, expectedFile? } objects
  --top-k <n>        Number of results per query (default: 5)
  -h, --help         Show this message

Examples:
  node --experimental-strip-types benchmarks/benchmark.ts
  node --experimental-strip-types benchmarks/benchmark.ts ~/projects/my-app
  node --experimental-strip-types benchmarks/benchmark.ts . --top-k 10
    `);
    process.exit(0);
  }

  const projectPath = resolve(positionals[0] ?? process.cwd());
  const topK = Math.max(1, parseInt(values['top-k'] ?? '5', 10));

  // Load queries
  let queries: QuerySpec[] = DEFAULT_QUERIES;
  if (values.queries) {
    const raw = await readFile(values.queries, 'utf-8');
    queries = JSON.parse(raw) as QuerySpec[];
    console.log(`Loaded ${queries.length} queries from ${values.queries}`);
  }

  console.log(c.bold('\n  Scope Benchmark'));
  console.log(c.dim(`  Project: ${projectPath}`));
  console.log(c.dim(`  Queries: ${queries.length}   top-k: ${topK}\n`));

  // Resolve config and instantiate indexer
  const config: ScopeConfig = await resolveConfig(projectPath);

  const embedder = createEmbedder(config);
  const vectorStore = new VectorStore(config);
  const chunker = new Chunker(config);
  const indexer = new Indexer(config, embedder, vectorStore, chunker);

  // Test that the index exists by running a dummy search first
  console.log(c.dim('  Checking index…'));
  try {
    await indexer.search('__warmup__', 1);
  } catch (err) {
    const msg = (err as Error).message ?? String(err);
    if (msg.includes('ENOENT') || msg.includes('not found') || msg.includes('No such')) {
      console.error(c.red(`\nError: no index found for ${projectPath}`));
      console.error(c.yellow(`  Run: scope index ${projectPath}`));
      process.exit(1);
    }
    // Other errors — let them surface naturally
  }

  // Scan repo for token counting
  console.log(c.dim('  Scanning source files for token baseline…'));
  const repo = await scanRepo(projectPath);
  console.log(
    c.dim(
      `  Found ${repo.fileCount} source files — ~${fmt(repo.totalTokens)} tokens (whole-repo baseline)\n`,
    ),
  );

  // Run queries
  const queryResults: QueryResult[] = [];

  for (let i = 0; i < queries.length; i++) {
    process.stdout.write(c.dim(`  Running query ${i + 1}/${queries.length}…\r`));
    const result = await runQuery(indexer, queries[i], topK, repo.fileSizes, projectPath);
    queryResults.push(result);
    printQueryResult(result, repo.totalTokens, i);
  }

  // Print summary
  printSummary(queryResults, repo, topK);
}

main().catch((err) => {
  console.error(c.red(`\nFatal: ${(err as Error).message ?? String(err)}`));
  process.exit(1);
});
