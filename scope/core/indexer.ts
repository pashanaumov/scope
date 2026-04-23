import { mkdir, readFile, stat, unlink, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import fg from 'fast-glob';
import ignore from 'ignore';
import pLimit from 'p-limit';
import type { Chunk, Chunker } from './chunker.js';
import type { ScopeConfig } from './config-resolver.js';
import type { Embedder } from './embedder.js';
import type { Manifest } from './manifest.js';
import { diffManifest, loadManifest, saveManifest } from './manifest.js';
import type { SearchResult, VectorRecord, VectorStore } from './vector-store.js';

// ---------------------------------------------------------------------------
// Re-exports for consumers
// ---------------------------------------------------------------------------

export type { Chunk } from './chunker.js';
export { Chunker } from './chunker.js';
export type { ScopeConfig } from './config-resolver.js';
export { resolveConfig } from './config-resolver.js';
export type { Embedder } from './embedder.js';
export { createEmbedder } from './embedder.js';
export type { Manifest, ManifestDiff } from './manifest.js';
export { checkSetupStatus, setup } from './setup.js';
export type { SearchResult } from './vector-store.js';
export { VectorStore } from './vector-store.js';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type IndexPhase = 'scan' | 'embed' | 'store' | 'done';

export interface IndexProgress {
  phase: IndexPhase;
  filesTotal: number;
  filesDone: number;
  chunksTotal: number;
  chunksDone: number;
  errors: Array<{ file: string; error: string }>;
}

export type ProgressCallback = (progress: IndexProgress) => void;

export interface IndexResult {
  filesIndexed: number;
  filesDeleted: number;
  filesUnchanged: number;
  chunksIndexed: number;
  errors: Array<{ file: string; error: string }>;
  durationMs: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HARD_CODED_IGNORE: string[] = [
  'node_modules',
  '.git',
  'dist',
  'build',
  'out',
  '__pycache__',
  '.next',
  '.nuxt',
  'coverage',
  '*.min.js',
  '*.d.ts',
  '*.map',
  '*.lock',
  '*-lock.yaml',
  '*.snap',
  '.DS_Store',
];

// ---------------------------------------------------------------------------
// Indexer
// ---------------------------------------------------------------------------

export class Indexer {
  private readonly config: ScopeConfig;
  private readonly embedder: Embedder;
  private readonly vectorStore: VectorStore;
  private readonly chunker: Chunker;
  private storeOpened = false;

  constructor(
    config: ScopeConfig,
    embedder: Embedder,
    vectorStore: VectorStore,
    chunker: Chunker,
  ) {
    this.config = config;
    this.embedder = embedder;
    this.vectorStore = vectorStore;
    this.chunker = chunker;
  }

  /** Index a project directory incrementally. */
  async index(
    projectPath: string,
    onProgress?: ProgressCallback,
    force = false,
    sourceUrlProvider?: (relPath: string) => string | undefined,
  ): Promise<IndexResult> {
    const startTime = Date.now();
    const absProjectPath = resolve(projectPath);
    const errors: Array<{ file: string; error: string }> = [];

    await this.ensureOpen();

    const manifestFilePath = join(this.config.indexDir, 'manifest.json');
    // force=true: start with empty manifest so everything is treated as new
    const storedManifest: Manifest = force ? {} : await loadManifest(manifestFilePath);

    // ----- SCAN -----
    const ig = await buildIgnoreFilter(absProjectPath, this.config.ignorePatterns);

    const rawPaths = await fg('**/*', {
      cwd: absProjectPath,
      dot: true,
      onlyFiles: true,
      followSymbolicLinks: false,
      ignore: [...HARD_CODED_IGNORE, ...this.config.ignorePatterns].map((p) =>
        p.includes('/') ? p : `**/${p}/**`,
      ),
    });

    const filteredPaths = rawPaths.filter((relPath) => !ig.ignores(relPath));

    const currentFiles: Record<string, { mtime: number; size: number }> = {};
    await Promise.all(
      filteredPaths.map(async (relPath) => {
        try {
          const s = await stat(resolve(absProjectPath, relPath));
          currentFiles[relPath] = { mtime: s.mtimeMs, size: s.size };
        } catch {
          // file disappeared between glob and stat — skip
        }
      }),
    );

    const totalFiles = Object.keys(currentFiles).length;
    onProgress?.({
      phase: 'scan',
      filesTotal: totalFiles,
      filesDone: totalFiles,
      chunksTotal: 0,
      chunksDone: 0,
      errors: [],
    });

    const { diff, updated: updatedManifest } = await diffManifest(
      absProjectPath,
      currentFiles,
      storedManifest,
    );

    const filesToProcess = [...diff.new, ...diff.changed];
    const filesUnchanged = totalFiles - filesToProcess.length - diff.deleted.length;

    // ----- EMBED -----
    const limit = pLimit(this.config.concurrency);
    const fileChunks = new Map<string, Chunk[]>();
    let filesDone = 0;

    await Promise.all(
      filesToProcess.map((relPath) =>
        limit(async () => {
          try {
            const content = await readFile(resolve(absProjectPath, relPath), 'utf-8');
            const chunks = await this.chunker.chunkFile(relPath, content);
            fileChunks.set(relPath, chunks);
          } catch (err) {
            errors.push({ file: relPath, error: String(err) });
          } finally {
            filesDone++;
            onProgress?.({
              phase: 'embed',
              filesTotal: filesToProcess.length,
              filesDone,
              chunksTotal: 0,
              chunksDone: 0,
              errors: [...errors],
            });
          }
        }),
      ),
    );

    // Collect all chunks across files
    const allChunkEntries: Array<{ chunk: Chunk }> = [];
    for (const chunks of fileChunks.values()) {
      for (const chunk of chunks) {
        allChunkEntries.push({ chunk });
      }
    }

    // Report chunksTotal now that chunking is complete
    onProgress?.({
      phase: 'embed',
      filesTotal: filesToProcess.length,
      filesDone: filesToProcess.length,
      chunksTotal: allChunkEntries.length,
      chunksDone: 0,
      errors: [...errors],
    });

    // Embed in batches
    const vectors: number[][] = [];
    for (let i = 0; i < allChunkEntries.length; i += this.config.batchSize) {
      const batch = allChunkEntries.slice(i, i + this.config.batchSize);
      const texts = batch.map((c) => c.chunk.embedText);
      const batchVectors = await this.embedder.embed(texts);
      vectors.push(...batchVectors);
      onProgress?.({
        phase: 'embed',
        filesTotal: filesToProcess.length,
        filesDone: filesToProcess.length,
        chunksTotal: allChunkEntries.length,
        chunksDone: vectors.length,
        errors: [...errors],
      });
    }

    if (vectors.length !== allChunkEntries.length) {
      throw new Error(
        `Embedder returned ${vectors.length} vectors for ${allChunkEntries.length} chunks — mismatch`,
      );
    }

    // Build VectorRecords
    const records: VectorRecord[] = allChunkEntries.map((c, i) => ({
      id: c.chunk.id,
      filePath: c.chunk.filePath,
      language: c.chunk.language,
      startLine: c.chunk.startLine,
      endLine: c.chunk.endLine,
      header: c.chunk.header,
      body: c.chunk.body,
      embedText: c.chunk.embedText,
      sourceUrl: sourceUrlProvider?.(c.chunk.filePath) ?? '',
      vector: vectors[i] as number[],
    }));

    // ----- STORE -----
    onProgress?.({
      phase: 'store',
      filesTotal: totalFiles,
      filesDone: totalFiles,
      chunksTotal: records.length,
      chunksDone: 0,
      errors: [...errors],
    });

    await Promise.all(diff.deleted.map((relPath) => this.vectorStore.deleteByFile(relPath)));

    if (records.length > 0) {
      await this.vectorStore.upsert(records);
    }

    // Exclude errored files from manifest so they get retried next run
    const failedFiles = new Set(errors.map((e) => e.file));
    const finalManifest: Manifest = { ...updatedManifest };
    for (const relPath of failedFiles) {
      if (diff.new.includes(relPath)) {
        delete finalManifest[relPath];
      } else if (diff.changed.includes(relPath)) {
        const old = storedManifest[relPath];
        if (old) finalManifest[relPath] = old;
        else delete finalManifest[relPath];
      }
    }

    await saveManifest(manifestFilePath, finalManifest);

    // Write stats.json so `workbench status` and MCP `get_indexing_status` can report persistent state.
    const statsPath = join(this.config.indexDir, 'stats.json');
    await mkdir(this.config.indexDir, { recursive: true });
    await writeFile(
      statsPath,
      JSON.stringify({
        lastIndexedAt: Date.now(),
        chunkCount: await this.vectorStore.count(),
        fileCount: Object.keys(finalManifest).length,
      }),
    );

    // ----- DONE -----
    onProgress?.({
      phase: 'done',
      filesTotal: totalFiles,
      filesDone: totalFiles,
      chunksTotal: records.length,
      chunksDone: records.length,
      errors: [...errors],
    });

    return {
      filesIndexed: filesToProcess.length - errors.length,
      filesDeleted: diff.deleted.length,
      filesUnchanged,
      chunksIndexed: records.length,
      errors,
      durationMs: Date.now() - startTime,
    };
  }

  /** Search the index. */
  async search(queryText: string, topK?: number): Promise<SearchResult[]> {
    await this.ensureOpen();
    const [queryVector = []] = await this.embedder.embed([queryText]);
    return this.vectorStore.hybridSearch(queryVector, queryText, topK ?? this.config.searchTopK);
  }

  /** Clear the entire index (VectorStore + Manifest). */
  async clear(projectPath: string): Promise<void> {
    void projectPath; // projectPath kept for API symmetry; manifest path comes from config
    await this.ensureOpen();
    await this.vectorStore.clear();
    const manifestFilePath = join(this.config.indexDir, 'manifest.json');
    try {
      await unlink(manifestFilePath);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
    }
  }

  private async ensureOpen(): Promise<void> {
    if (!this.storeOpened) {
      await this.vectorStore.open(this.embedder.dimensions);
      this.storeOpened = true;
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function buildIgnoreFilter(
  projectPath: string,
  extraPatterns: string[],
): Promise<ReturnType<typeof ignore>> {
  const ig = ignore();
  ig.add(HARD_CODED_IGNORE);
  ig.add(extraPatterns);

  // Walk up from projectPath, innermost first, collecting .gitignore files
  const dirs: string[] = [];
  let dir = resolve(projectPath);
  while (true) {
    dirs.push(dir);
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }

  // Add from outermost → innermost so inner rules take precedence
  for (const d of dirs.reverse()) {
    try {
      const content = await readFile(join(d, '.gitignore'), 'utf-8');
      ig.add(content);
    } catch {
      // no .gitignore here
    }
  }

  return ig;
}
