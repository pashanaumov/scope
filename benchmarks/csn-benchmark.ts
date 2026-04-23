#!/usr/bin/env node
/**
 * CodeSearchNet NDCG@10 Benchmark — scope
 *
 * Downloads the Python test split of CodeSearchNet (~22K functions) and the
 * official annotation store (99 queries × ~21 labels each), indexes all
 * functions with our indexer, runs the 99 queries, and computes NDCG@10 —
 * the same metric used in the original CSN paper.
 *
 * All downloaded data is cached in ~/.scope-csn-cache so subsequent runs
 * are fast (no re-download, incremental re-index of only new/changed files).
 *
 * Baselines from the CSN paper (Python, NDCG@10):
 *   BM25:                       ~0.17
 *   NBOW (neural bag-of-words): ~0.26
 *   jina-embeddings-v2-base-code (published): ~0.35
 *
 * Usage:
 *   npm run csn-bench                  # run full benchmark
 *   npm run csn-bench --skip-index     # skip indexing (re-use existing index)
 *   npm run csn-bench --clear-cache    # wipe cache and start fresh
 *   npm run csn-bench --help
 */

import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';
import {
  Chunker,
  createEmbedder,
  Indexer,
  resolveConfig,
  VectorStore,
} from '../scope/core/index.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LANGUAGE = 'python';
const DATASET_ID = 'code-search-net/code_search_net';
const ANNOTATION_CSV_URL =
  'https://raw.githubusercontent.com/github/CodeSearchNet/master/resources/annotationStore.csv';
const HF_ROWS_API = 'https://datasets-server.huggingface.co/rows';
const PAGE_SIZE = 100;
const EVAL_K = 10;
const RECALL_KS = [1, 3, 5] as const;
// Fetch more results per query so we have headroom after deduplication.
const SEARCH_OVERSAMPLE = EVAL_K * 5;

const CACHE_DIR = join(homedir(), '.scope-csn-cache');
const CORPUS_DIR = join(CACHE_DIR, 'corpus', LANGUAGE);
const INDEX_DIR = join(CACHE_DIR, 'index');
const ROWS_CACHE_DIR = join(CACHE_DIR, 'rows', LANGUAGE);
const URL_INDEX_PATH = join(CACHE_DIR, 'url-index.json');
const RESULTS_PATH = join(CACHE_DIR, 'results.jsonl');

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CsnRow {
  func_code_url: string;
  whole_func_string: string;
}

// Map<query → Map<githubUrl → relevance (0-3)>>
type AnnotationMap = Map<string, Map<string, number>>;

// ---------------------------------------------------------------------------
// Logging helpers
// ---------------------------------------------------------------------------

function log(msg: string): void {
  process.stdout.write(`${msg}\n`);
}

/** Overwrite the current line (progress indicator). */
function prog(msg: string): void {
  process.stdout.write(`\r${msg.padEnd(80)}`);
}

function clearLine(): void {
  process.stdout.write(`\r${' '.repeat(80)}\r`);
}

// ---------------------------------------------------------------------------
// HTTP with retry + exponential back-off
// ---------------------------------------------------------------------------

async function fetchRetry(url: string, retries = 4): Promise<Response> {
  let lastErr: unknown;
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const res = await fetch(url);
      if (res.ok) return res;
      if (res.status === 429 || res.status >= 500) {
        const delay = 2 ** attempt * 1500;
        prog(`  HTTP ${res.status} — retry in ${delay}ms (attempt ${attempt + 1})`);
        await sleep(delay);
        continue;
      }
      throw new Error(`HTTP ${res.status} fetching ${url}`);
    } catch (err) {
      lastErr = err;
      if (attempt < retries - 1) await sleep(1000 * (attempt + 1));
    }
  }
  throw lastErr;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// ---------------------------------------------------------------------------
// Step 1 — Annotation store
// ---------------------------------------------------------------------------

async function loadAnnotations(): Promise<AnnotationMap> {
  const cachePath = join(CACHE_DIR, 'annotationStore.csv');
  let csv: string;

  if (existsSync(cachePath)) {
    csv = await readFile(cachePath, 'utf-8');
    log(`✓ Annotation store: loaded from cache`);
  } else {
    log('↓ Downloading annotation store...');
    const res = await fetchRetry(ANNOTATION_CSV_URL);
    csv = await res.text();
    await mkdir(CACHE_DIR, { recursive: true });
    await writeFile(cachePath, csv);
    log(`✓ Annotation store: saved to ${cachePath}`);
  }

  // CSV columns: Language,Query,GitHubUrl,Relevance,Notes
  const map: AnnotationMap = new Map();
  let pyAnnotations = 0;

  for (const line of csv.split('\n').slice(1)) {
    const [lang, query, url, relStr] = line.split(',');
    if (lang?.trim() !== 'Python' || !query?.trim() || !url?.trim()) continue;

    const relevance = Number.parseInt(relStr?.trim() ?? '0', 10);
    if (!map.has(query.trim())) map.set(query.trim(), new Map());
    const qMap = map.get(query.trim()) ?? new Map<string, number>();

    // Average repeated annotations for the same (query, url) pair
    const prev = qMap.get(url.trim());
    qMap.set(url.trim(), prev !== undefined ? (prev + relevance) / 2 : relevance);
    pyAnnotations++;
  }

  log(`  → ${map.size} queries, ${pyAnnotations} annotations`);
  return map;
}

// ---------------------------------------------------------------------------
// Step 2 — Corpus download
//
// Returns Map<relPath → githubUrl> where relPath is the filename within
// CORPUS_DIR (e.g. "abc123def.py"). This is the key used by the Indexer's
// sourceUrlProvider.
// ---------------------------------------------------------------------------

function urlToFilename(url: string): string {
  return `${createHash('sha256').update(url).digest('hex').slice(0, 32)}.py`;
}

async function buildCorpus(): Promise<Map<string, string>> {
  await mkdir(ROWS_CACHE_DIR, { recursive: true });
  await mkdir(CORPUS_DIR, { recursive: true });

  // Load existing URL index (filename → githubUrl)
  let urlIndex: Record<string, string> = {};
  if (existsSync(URL_INDEX_PATH)) {
    urlIndex = JSON.parse(await readFile(URL_INDEX_PATH, 'utf-8'));
  }

  // Fetch page 0 to get total count
  const page0Path = join(ROWS_CACHE_DIR, 'page_000000.json');
  let page0Data: { num_rows_total: number; rows: Array<{ row: CsnRow }> };

  if (existsSync(page0Path)) {
    page0Data = JSON.parse(await readFile(page0Path, 'utf-8'));
  } else {
    log('↓ Fetching first page to determine corpus size...');
    const url = hfRowsUrl(0);
    const res = await fetchRetry(url);
    page0Data = (await res.json()) as typeof page0Data;
    await writeFile(page0Path, JSON.stringify(page0Data));
  }

  const total = page0Data.num_rows_total;
  const totalPages = Math.ceil(total / PAGE_SIZE);
  log(`↓ Corpus: ${total} functions, ${totalPages} pages`);

  let written = 0;
  let cached = 0;
  let urlIndexDirty = false;

  for (let page = 0; page < totalPages; page++) {
    const pagePath = join(ROWS_CACHE_DIR, `page_${String(page).padStart(6, '0')}.json`);
    let pageData: { rows: Array<{ row: CsnRow }> };

    if (existsSync(pagePath)) {
      pageData = JSON.parse(await readFile(pagePath, 'utf-8'));
    } else {
      const res = await fetchRetry(hfRowsUrl(page * PAGE_SIZE));
      pageData = (await res.json()) as typeof pageData;
      await writeFile(pagePath, JSON.stringify(pageData));
    }

    for (const { row } of pageData.rows) {
      if (!row.func_code_url || !row.whole_func_string) continue;
      const filename = urlToFilename(row.func_code_url);
      const filePath = join(CORPUS_DIR, filename);

      if (!urlIndex[filename]) {
        urlIndex[filename] = row.func_code_url;
        urlIndexDirty = true;
      }

      if (existsSync(filePath)) {
        cached++;
      } else {
        await writeFile(filePath, row.whole_func_string, 'utf-8');
        written++;
      }
    }

    if ((page + 1) % 20 === 0 || page + 1 === totalPages) {
      prog(`  Pages: ${page + 1}/${totalPages} — ${written} written, ${cached} cached`);
    }
  }

  if (urlIndexDirty) {
    await writeFile(URL_INDEX_PATH, JSON.stringify(urlIndex));
  }

  clearLine();
  log(`✓ Corpus ready: ${written + cached} functions (${written} new, ${cached} cached)`);

  // Build the relPath → url map for sourceUrlProvider
  const urlMap = new Map<string, string>();
  for (const [filename, url] of Object.entries(urlIndex)) {
    urlMap.set(filename, url);
  }
  return urlMap;
}

function hfRowsUrl(offset: number): string {
  return `${HF_ROWS_API}?dataset=${encodeURIComponent(DATASET_ID)}&config=${LANGUAGE}&split=test&offset=${offset}&length=${PAGE_SIZE}`;
}

// ---------------------------------------------------------------------------
// Step 3 — Index
// ---------------------------------------------------------------------------

async function indexCorpus(urlMap: Map<string, string>): Promise<void> {
  const config = await resolveConfig(CORPUS_DIR, {
    indexDir: INDEX_DIR,
    ignorePatterns: [],
    chunkStrategy: 'function',
    searchTopK: SEARCH_OVERSAMPLE,
  });

  const embedder = createEmbedder(config);
  const vectorStore = new VectorStore(config);
  const chunker = new Chunker(config);
  const indexer = new Indexer(config, embedder, vectorStore, chunker);

  log('⚙  Indexing corpus...');
  log('   First run: ~5-15 min to embed 22K functions. Subsequent runs are incremental.');

  let lastPhase = '';
  const t0 = Date.now();

  const result = await indexer.index(
    CORPUS_DIR,
    (p) => {
      if (p.phase !== lastPhase) {
        if (lastPhase) clearLine();
        lastPhase = p.phase;
        log(`  Phase: ${p.phase}`);
      }
      if (p.phase === 'embed') {
        prog(`    Embedding: ${p.filesDone}/${p.filesTotal} files`);
      }
    },
    false,
    (relPath) => urlMap.get(relPath),
  );

  clearLine();
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  log(
    `✓ Index complete: ${result.filesIndexed} new, ${result.filesUnchanged} unchanged, ${result.chunksIndexed} chunks (${elapsed}s)`,
  );
  if (result.errors.length > 0) {
    log(`  ⚠ ${result.errors.length} errors (first: ${result.errors[0]?.error})`);
  }
}

// ---------------------------------------------------------------------------
// Step 4 — Evaluate (NDCG@K)
// ---------------------------------------------------------------------------

/** DCG@K using gain = 2^rel − 1, denominator = log2(rank + 1), 1-indexed rank. */
function dcgAtK(relevances: number[], k: number): number {
  let dcg = 0;
  for (let i = 0; i < Math.min(k, relevances.length); i++) {
    dcg += (2 ** relevances[i] - 1) / Math.log2(i + 2); // i+2 = rank+1 (rank is 1-indexed)
  }
  return dcg;
}

/**
 * Returns NDCG@K, or -1 to signal "skip this query" when IDCG=0.
 * Results are deduplicated by URL (best score per URL) before scoring.
 */
function computeNdcg(
  retrievedUrls: Array<{ url: string; score: number }>,
  annotations: Map<string, number>,
  k: number,
): number {
  // Ideal DCG from annotation store (perfect ranking)
  const idealRels = [...annotations.values()].sort((a, b) => b - a).slice(0, k);
  const idcg = dcgAtK(idealRels, k);
  if (idcg === 0) return -1; // no relevant items annotated — skip

  const relList = retrievedUrls.slice(0, k).map((r) => annotations.get(r.url) ?? 0);
  return dcgAtK(relList, k) / idcg;
}

/**
 * Returns MRR: reciprocal rank of the first result with relevance > 0.
 * Uses the full retrieved list (not capped at K) so MRR isn't artificially
 * zero for queries where the first relevant item lands outside the top-10.
 */
function computeMrr(
  retrievedUrls: Array<{ url: string; score: number }>,
  annotations: Map<string, number>,
): number {
  for (let i = 0; i < retrievedUrls.length; i++) {
    if ((annotations.get(retrievedUrls[i]?.url) ?? 0) > 0) return 1 / (i + 1);
  }
  return 0;
}

/**
 * Returns 1 if at least one result in the top-K has relevance > 0, else 0.
 * This is the "hit rate" / "success@K" variant of Recall, appropriate for
 * sparsely-annotated datasets like CSN where only ~21 URLs are labelled per query.
 */
function computeRecallAtK(
  retrievedUrls: Array<{ url: string; score: number }>,
  annotations: Map<string, number>,
  k: number,
): number {
  return retrievedUrls.slice(0, k).some((r) => (annotations.get(r.url) ?? 0) > 0) ? 1 : 0;
}

interface QueryResult {
  query: string;
  ndcg10: number;
  dcg10: number;
  idcg10: number;
  mrr: number;
  recall1: number;
  recall3: number;
  recall5: number;
  numAnnotations: number;
  results: Array<{ url: string; rank: number; relevance: number; score: number }>;
}

async function runEval(annotations: AnnotationMap, _urlMap: Map<string, string>): Promise<void> {
  const config = await resolveConfig(CORPUS_DIR, {
    indexDir: INDEX_DIR,
    ignorePatterns: [],
    searchTopK: SEARCH_OVERSAMPLE,
  });

  const embedder = createEmbedder(config);
  const vectorStore = new VectorStore(config);
  const chunker = new Chunker(config);
  const indexer = new Indexer(config, embedder, vectorStore, chunker);

  const queries = [...annotations.keys()];
  log(
    `\n▶  Running ${queries.length} queries (NDCG@${EVAL_K}, MRR, Recall@${RECALL_KS.join('/')})...`,
  );

  const queryResults: QueryResult[] = [];
  const ndcgScores: number[] = [];
  const t0 = Date.now();

  for (let qi = 0; qi < queries.length; qi++) {
    const query = queries[qi] ?? '';
    if (!query) continue;
    prog(`  Query ${qi + 1}/${queries.length}: "${query.slice(0, 55)}"`);

    const rawResults = await indexer.search(query, SEARCH_OVERSAMPLE);

    // Deduplicate by sourceUrl (keep best score per URL)
    const bestByUrl = new Map<string, number>();
    for (const r of rawResults) {
      if (!r.sourceUrl) continue;
      const existing = bestByUrl.get(r.sourceUrl);
      if (existing === undefined || r.score > existing) {
        bestByUrl.set(r.sourceUrl, r.score);
      }
    }

    const deduped = [...bestByUrl.entries()]
      .map(([url, score]) => ({ url, score }))
      .sort((a, b) => b.score - a.score);
    // Keep full list for MRR/Recall; slice to EVAL_K only for NDCG.
    const dedupedTop = deduped.slice(0, EVAL_K);

    const qAnnotations = annotations.get(query) ?? new Map<string, number>();
    const ndcg = computeNdcg(dedupedTop, qAnnotations, EVAL_K);
    const mrr = computeMrr(deduped, qAnnotations);
    const recall1 = computeRecallAtK(deduped, qAnnotations, 1);
    const recall3 = computeRecallAtK(deduped, qAnnotations, 3);
    const recall5 = computeRecallAtK(deduped, qAnnotations, 5);

    // Compute DCG/IDCG for the result record even when skipping
    const relList = dedupedTop.map((r) => qAnnotations.get(r.url) ?? 0);
    const idealRels = [...qAnnotations.values()].sort((a, b) => b - a).slice(0, EVAL_K);

    if (ndcg >= 0) ndcgScores.push(ndcg);

    queryResults.push({
      query,
      ndcg10: ndcg,
      dcg10: dcgAtK(relList, EVAL_K),
      idcg10: dcgAtK(idealRels, EVAL_K),
      mrr,
      recall1,
      recall3,
      recall5,
      numAnnotations: qAnnotations.size,
      results: dedupedTop.map((r, i) => ({
        url: r.url,
        rank: i + 1,
        relevance: qAnnotations.get(r.url) ?? 0,
        score: r.score,
      })),
    });
  }

  clearLine();
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  log(`✓ Queries done (${elapsed}s)`);

  // Write results.jsonl
  await writeFile(RESULTS_PATH, `${queryResults.map((r) => JSON.stringify(r)).join('\n')}\n`);
  log(`✓ Results saved → ${RESULTS_PATH}`);

  // ── Summary ──────────────────────────────────────────────────────────────
  const scored = ndcgScores.length;
  const skipped = queries.length - scored;
  const meanNdcg = scored > 0 ? ndcgScores.reduce((a, b) => a + b, 0) / scored : 0;

  const scoredResults = queryResults.filter((r) => r.ndcg10 >= 0);
  const n = scoredResults.length || 1;
  const meanMrr = scoredResults.reduce((a, r) => a + r.mrr, 0) / n;
  const meanRecalls = RECALL_KS.map(
    (k) =>
      scoredResults.reduce((a, r) => a + r[`recall${k}` as 'recall1' | 'recall3' | 'recall5'], 0) /
      n,
  );

  // Top / bottom 5 queries
  const sorted = scoredResults.sort((a, b) => b.ndcg10 - a.ndcg10);
  const top5 = sorted.slice(0, 5);
  const bot5 = sorted.slice(-5).reverse();

  log('\n═══════════════════════════════════════════════════════════════');
  log(`  CodeSearchNet Retrieval Metrics — Python test split`);
  log('═══════════════════════════════════════════════════════════════');
  log(
    `  Mean NDCG@${EVAL_K}:      ${meanNdcg.toFixed(4)}  (${scored} queries scored, ${skipped} skipped)`,
  );
  log(`  MRR:             ${meanMrr.toFixed(4)}`);
  for (let i = 0; i < RECALL_KS.length; i++) {
    log(`  Recall@${RECALL_KS[i]}:        ${(meanRecalls[i] ?? 0).toFixed(4)}  (hit rate)`);
  }
  log('');
  log('  Baselines (CSN paper, Python, NDCG@10):');
  log('    BM25                             0.1716');
  log('    NBOW (neural bag-of-words)       0.2602');
  log('    jina-embeddings-v2-base-code     ~0.35  (published)');
  log('');
  log('  Top 5 queries by NDCG@10:');
  for (const r of top5) {
    log(`    ${r.ndcg10.toFixed(4)}  "${r.query}"`);
  }
  log('');
  log('  Bottom 5 queries by NDCG@10:');
  for (const r of bot5) {
    log(`    ${r.ndcg10.toFixed(4)}  "${r.query}"`);
  }
  log('═══════════════════════════════════════════════════════════════');
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    log('Usage: npm run csn-bench [options]');
    log('');
    log('Options:');
    log('  --skip-index    Skip indexing phase, run queries against existing index');
    log('  --clear-cache   Delete all cached data and start fresh');
    log('  --help          Show this help');
    log('');
    log(`Cache directory: ${CACHE_DIR}`);
    log(`Results file:    ${RESULTS_PATH}`);
    return;
  }

  if (args.includes('--clear-cache')) {
    log(`⚠  Clearing cache at ${CACHE_DIR}...`);
    await rm(CACHE_DIR, { recursive: true, force: true });
    log('✓ Cache cleared\n');
  }

  const skipIndex = args.includes('--skip-index');

  log('═══════════════════════════════════════════════════════════════');
  log('  CodeSearchNet NDCG@10 Benchmark — scope');
  log('═══════════════════════════════════════════════════════════════\n');

  // Step 1: annotations
  const annotations = await loadAnnotations();

  // Step 2: corpus
  let urlMap: Map<string, string>;
  if (skipIndex && existsSync(URL_INDEX_PATH)) {
    const urlIndex = JSON.parse(await readFile(URL_INDEX_PATH, 'utf-8')) as Record<string, string>;
    urlMap = new Map(Object.entries(urlIndex));
    log(`✓ URL index loaded from cache (${urlMap.size} entries)`);
  } else {
    urlMap = await buildCorpus();
  }

  // Step 3: index (unless --skip-index)
  if (!skipIndex) {
    await indexCorpus(urlMap);
  } else {
    log('⏭  Skipping index phase (--skip-index)');
  }

  // Step 4: eval
  await runEval(annotations, urlMap);
}

main().catch((err: unknown) => {
  process.stderr.write(`\nFatal error: ${err}\n`);
  process.exit(1);
});
