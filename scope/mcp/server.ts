import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import type { Indexer, ScopeConfig } from '../core/index.js';

// ---------------------------------------------------------------------------
// DI interface
// ---------------------------------------------------------------------------

export interface ServerDeps {
  resolveConfig?: (projectPath: string) => Promise<ScopeConfig>;
  createIndexer?: (config: ScopeConfig) => Indexer | Promise<Indexer>;
  checkSetupStatus?: (
    config: Pick<ScopeConfig, 'modelsDir' | 'grammarsDir' | 'transformersModel'>,
  ) => Promise<{ modelReady: boolean; grammarsMissing: string[] }>;
  runSetup?: (config: ScopeConfig) => Promise<void>;
  startWatcher?: (
    projectPath: string,
    indexFn: () => Promise<void>,
    opts: { debounceMs: number },
  ) => void;
}

// ---------------------------------------------------------------------------
// Project root detection
// ---------------------------------------------------------------------------

export async function detectProjectRoot(): Promise<string> {
  // Explicit overrides (highest priority)
  const envPath = process.env.SCOPE_PROJECT_PATH ?? process.env.CLAUDE_PROJECT_ROOT;
  if (envPath) return envPath;

  // Walk up from cwd looking for .git — but skip plugin cache dirs to avoid
  // accidentally indexing the scope plugin itself when running as an MCP server.
  let dir = process.cwd();
  while (true) {
    if (existsSync(join(dir, '.git'))) {
      // Sanity check: don't return a path inside the Claude plugin cache
      if (!dir.includes('/.claude/plugins/')) return dir;
    }
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return process.cwd();
}

// ---------------------------------------------------------------------------
// Default implementations (lazy-loaded to avoid heavy deps at startup)
// ---------------------------------------------------------------------------

async function defaultResolveConfig(projectPath: string): Promise<ScopeConfig> {
  const { resolveConfig } = await import('../core/index.js');
  return resolveConfig(projectPath);
}

async function defaultCreateIndexer(config: ScopeConfig): Promise<Indexer> {
  const {
    createEmbedder,
    Chunker,
    VectorStore,
    Indexer: IndexerClass,
  } = await import('../core/index.js');
  const embedder = createEmbedder(config);
  const vectorStore = new VectorStore(config);
  const chunker = new Chunker(config);
  return new IndexerClass(config, embedder, vectorStore, chunker);
}

async function defaultCheckSetupStatus(
  config: Pick<ScopeConfig, 'modelsDir' | 'grammarsDir' | 'transformersModel'>,
): Promise<{ modelReady: boolean; grammarsMissing: string[] }> {
  const { checkSetupStatus } = await import('../core/index.js');
  return checkSetupStatus(config);
}

async function defaultRunSetup(config: ScopeConfig): Promise<void> {
  const { setup } = await import('../core/index.js');
  await setup({ config });
}

function defaultStartWatcher(
  projectPath: string,
  indexFn: () => Promise<void>,
  opts: { debounceMs: number },
): void {
  import('chokidar')
    .then(({ watch }) => {
      let debounceTimer: NodeJS.Timeout | undefined;
      let indexing = false;

      const watcher = watch(projectPath, {
        ignored: /(^|[/\\])(\.git|node_modules|dist)($|[/\\])/,
        persistent: false,
        ignoreInitial: true,
      });

      watcher.on('all', () => {
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(async () => {
          if (indexing) return;
          indexing = true;
          try {
            await indexFn();
          } finally {
            indexing = false;
          }
        }, opts.debounceMs);
      });
    })
    .catch(() => {
      // chokidar unavailable — skip watcher
    });
}

// ---------------------------------------------------------------------------
// Server factory
// ---------------------------------------------------------------------------

export function createServer(deps: ServerDeps = {}): Server {
  const {
    resolveConfig = defaultResolveConfig,
    createIndexer = defaultCreateIndexer,
    checkSetupStatus = defaultCheckSetupStatus,
    runSetup = defaultRunSetup,
    startWatcher = defaultStartWatcher,
  } = deps;

  const server = new Server(
    { name: 'scope', version: '0.1.0' },
    { capabilities: { tools: {} } },
  );

  // Lazy singleton state
  let indexer: Indexer | null = null;
  let currentConfig: ScopeConfig | null = null;
  let currentProjectPath: string | null = null;
  let watcherStarted = false;

  async function getIndexer(projectPath: string): Promise<{ idx: Indexer; cfg: ScopeConfig }> {
    if (indexer && currentConfig && currentProjectPath === projectPath) {
      return { idx: indexer, cfg: currentConfig };
    }
    // Path changed or first call — (re)create indexer
    currentProjectPath = projectPath;
    currentConfig = await resolveConfig(projectPath);
    indexer = await createIndexer(currentConfig);
    watcherStarted = false; // reset watcher state for new project
    return { idx: indexer, cfg: currentConfig };
  }

  // ------------------------------------------------------------------
  // Tool list
  // ------------------------------------------------------------------

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: 'index_codebase',
        description: 'Index the current codebase for semantic search. Run this before searching.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            path: { type: 'string', description: 'Project root path (defaults to auto-detected)' },
            force: { type: 'boolean', description: 'Re-index all files even if unchanged' },
          },
        },
      },
      {
        name: 'search_code',
        description: 'Search the indexed codebase using semantic + keyword hybrid search.',
        inputSchema: {
          type: 'object' as const,
          required: ['query'],
          properties: {
            query: { type: 'string', description: 'Search query' },
            topK: { type: 'number', description: 'Number of results (default 5)' },
            path: { type: 'string', description: 'Project root path (defaults to auto-detected)' },
          },
        },
      },
      {
        name: 'clear_index',
        description: 'Clear the search index for the current project.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            path: { type: 'string', description: 'Project root path (defaults to auto-detected)' },
          },
        },
      },
      {
        name: 'get_indexing_status',
        description: 'Check whether the codebase has been indexed and when.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            path: { type: 'string', description: 'Project root path (defaults to auto-detected)' },
          },
        },
      },
    ],
  }));

  // ------------------------------------------------------------------
  // Tool calls
  // ------------------------------------------------------------------

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args = {} } = request.params;

    try {
      switch (name) {
        case 'index_codebase': {
          const { path: pathArg, force } = args as { path?: string; force?: boolean };
          const projectPath = pathArg ?? (await detectProjectRoot());
          const { idx, cfg } = await getIndexer(projectPath);
          await runSetup(cfg);
          const result = await idx.index(projectPath, undefined, force ?? false);

          if (cfg.watchEnabled && !watcherStarted) {
            startWatcher(
              projectPath,
              async () => {
                await idx.index(projectPath);
              },
              { debounceMs: cfg.watchDebounceMs },
            );
            watcherStarted = true;
          }

          return {
            content: [
              {
                type: 'text' as const,
                text: JSON.stringify({
                  filesIndexed: result.filesIndexed,
                  chunksIndexed: result.chunksIndexed,
                  durationMs: result.durationMs,
                  errors: result.errors,
                }),
              },
            ],
          };
        }

        case 'search_code': {
          const { query, topK, path: pathArg } = args as { query?: string; topK?: number; path?: string };
          if (!query) {
            return {
              content: [
                {
                  type: 'text' as const,
                  text: JSON.stringify({ error: 'query parameter is required' }),
                },
              ],
              isError: true,
            };
          }
          const projectPath = pathArg ?? (await detectProjectRoot());
          const { idx } = await getIndexer(projectPath);
          const results = await idx.search(query, topK);
          return {
            content: [
              {
                type: 'text' as const,
                text: JSON.stringify(
                  results.map((r) => ({
                    file: r.filePath,
                    startLine: r.startLine,
                    endLine: r.endLine,
                    content: r.body,
                    score: r.score,
                    language: r.language,
                  })),
                ),
              },
            ],
          };
        }

        case 'clear_index': {
          const { path: pathArg } = args as { path?: string };
          const projectPath = pathArg ?? (await detectProjectRoot());
          const { idx } = await getIndexer(projectPath);
          await idx.clear(projectPath);
          indexer = null;
          watcherStarted = false;
          return {
            content: [{ type: 'text' as const, text: JSON.stringify({ cleared: true }) }],
          };
        }

        case 'get_indexing_status': {
          const { path: pathArg } = args as { path?: string };
          const projectPath = pathArg ?? (await detectProjectRoot());
          const cfg = currentConfig ?? (await resolveConfig(projectPath));
          const { modelReady, grammarsMissing } = await checkSetupStatus(cfg);

          // Prefer on-disk stats so status survives server restarts
          let indexed = false;
          let filesCount = 0;
          let chunksCount = 0;
          let lastIndexedAt: number | null = null;

          try {
            const statsPath = join(cfg.indexDir, 'stats.json');
            const raw = await readFile(statsPath, 'utf-8');
            const stats = JSON.parse(raw) as {
              lastIndexedAt?: number;
              chunkCount?: number;
              fileCount?: number;
            };
            indexed = true;
            chunksCount = stats.chunkCount ?? 0;
            filesCount = stats.fileCount ?? 0;
            lastIndexedAt = stats.lastIndexedAt ?? null;
          } catch {
            // No stats.json yet — not indexed
          }

          return {
            content: [
              {
                type: 'text' as const,
                text: JSON.stringify({
                  indexed,
                  filesCount,
                  chunksCount,
                  lastIndexedAt,
                  modelReady,
                  grammarsMissing,
                }),
              },
            ],
          };
        }

        default:
          return {
            content: [
              { type: 'text' as const, text: JSON.stringify({ error: `Unknown tool: ${name}` }) },
            ],
            isError: true,
          };
      }
    } catch (err) {
      return {
        content: [{ type: 'text' as const, text: JSON.stringify({ error: String(err) }) }],
        isError: true,
      };
    }
  });

  return server;
}
