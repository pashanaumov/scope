import { mkdir } from 'node:fs/promises';
import type { Connection, Table } from '@lancedb/lancedb';
import * as lancedb from '@lancedb/lancedb';
import type { ScopeConfig } from './config-resolver.js';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface VectorRecord {
  id: string;
  filePath: string;
  language: string;
  startLine: number;
  endLine: number;
  header: string;
  body: string;
  embedText: string;
  /** Back-link to the original source (e.g. GitHub URL). Empty string when none. */
  sourceUrl: string;
  vector: number[];
}

export interface SearchResult {
  id: string;
  filePath: string;
  language: string;
  startLine: number;
  endLine: number;
  header: string;
  body: string;
  /** Original source URL, if the record was indexed with one. Empty string otherwise. */
  sourceUrl: string;
  score: number;
}

// ---------------------------------------------------------------------------
// VectorStore
// ---------------------------------------------------------------------------

export class VectorStore {
  private readonly indexDir: string;
  private readonly defaultTopK: number;
  private dimensions = 0;

  private db: Connection | null = null;
  private table: Table | null = null;
  private hasFtsIndex: boolean = false;

  constructor(config: Pick<ScopeConfig, 'indexDir' | 'searchTopK'>) {
    this.indexDir = config.indexDir;
    this.defaultTopK = config.searchTopK;
  }

  /**
   * Open (or create) the LanceDB table. Must be called before other methods.
   */
  async open(dimensions: number): Promise<void> {
    this.dimensions = dimensions;
    await mkdir(this.indexDir, { recursive: true });
    this.db = await lancedb.connect(this.indexDir);

    const names = await this.db.tableNames();
    if (names.includes('chunks')) {
      this.table = await this.db.openTable('chunks');
      const indices = await this.table.listIndices();
      this.hasFtsIndex = indices.some((idx) => idx.name === 'embedText_idx');
    } else {
      // LanceDB requires at least one row to infer schema (including vector dims).
      const dummy = makeDummyRecord(dimensions);
      this.table = await this.db.createTable('chunks', [dummy]);
      await this.table.delete("id = '__dummy__'");
      this.hasFtsIndex = false;
    }
  }

  /**
   * Upsert a batch of records. Existing IDs are replaced.
   */
  async upsert(records: VectorRecord[]): Promise<void> {
    if (!this.table) throw new Error('VectorStore not opened');
    if (records.length === 0) return;

    await this.table
      .mergeInsert('id')
      .whenMatchedUpdateAll()
      .whenNotMatchedInsertAll()
      .execute(records as unknown as Record<string, unknown>[]);

    // Rebuild FTS index after data changes (replace: true is the default).
    await this.table.createIndex('embedText', { config: lancedb.Index.fts() });
    await this.table.waitForIndex(['embedText_idx'], 30);
    this.hasFtsIndex = true;
  }

  /**
   * Hybrid search: vector + FTS (RRF fusion). Falls back to vector-only when
   * no FTS index exists or the query text is empty.
   */
  async hybridSearch(
    queryVector: number[],
    queryText: string,
    topK?: number,
  ): Promise<SearchResult[]> {
    if (!this.table) throw new Error('VectorStore not opened');
    const k = topK ?? this.defaultTopK;

    if (this.hasFtsIndex && queryText.trim()) {
      const reranker = await lancedb.rerankers.RRFReranker.create();
      const rows = await this.table
        .vectorSearch(queryVector)
        .fullTextSearch(queryText, { columns: ['embedText'] })
        .rerank(reranker)
        .limit(k)
        .toArray();
      return rows.map((row) => toSearchResult(row));
    }

    // Vector-only fallback
    const rows = await this.table.vectorSearch(queryVector).limit(k).toArray();
    return rows.map((row) => toSearchResult(row));
  }

  /**
   * Delete all chunks belonging to a file.
   */
  async deleteByFile(filePath: string): Promise<void> {
    if (!this.table) throw new Error('VectorStore not opened');
    const escaped = filePath.replace(/'/g, "''");
    await this.table.delete(`filePath = '${escaped}'`);
  }

  /**
   * Clear all data.
   */
  async clear(): Promise<void> {
    if (!this.table) throw new Error('VectorStore not opened');
    await this.table.delete('id IS NOT NULL');
    this.hasFtsIndex = false;
  }

  /**
   * Return number of records in the store.
   */
  async count(): Promise<number> {
    if (!this.table) throw new Error('VectorStore not opened');
    return this.table.countRows();
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeDummyRecord(dimensions: number): Record<string, unknown> {
  return {
    id: '__dummy__',
    filePath: '',
    language: '',
    startLine: 0,
    endLine: 0,
    header: '',
    body: '',
    embedText: '',
    sourceUrl: '',
    vector: new Array<number>(dimensions).fill(0),
  };
}

function toSearchResult(row: Record<string, unknown>): SearchResult {
  // Hybrid (RRF) returns _score; vector-only returns _distance (lower = better).
  let score: number;
  if (typeof row._score === 'number') {
    score = row._score as number;
  } else {
    const dist = typeof row._distance === 'number' ? (row._distance as number) : 1;
    score = 1 / (1 + dist);
  }

  return {
    id: row.id as string,
    filePath: row.filePath as string,
    language: row.language as string,
    startLine: row.startLine as number,
    endLine: row.endLine as number,
    header: row.header as string,
    body: row.body as string,
    sourceUrl: (row.sourceUrl as string) ?? '',
    score,
  };
}
