export const VERSION = '0.1.0';
export type { Chunk } from './chunker.js';
export { Chunker } from './chunker.js';
export type { ScopeConfig } from './config-resolver.js';
export { resolveConfig } from './config-resolver.js';
export type {
  Embedder,
  IndexPhase,
  IndexProgress,
  IndexResult,
  ProgressCallback,
} from './indexer.js';
export {
  checkSetupStatus,
  createEmbedder,
  Indexer,
  setup,
} from './indexer.js';
export type { Manifest, ManifestDiff, ManifestEntry } from './manifest.js';
export { diffManifest, loadManifest, saveManifest } from './manifest.js';
export type { SearchResult, VectorRecord } from './vector-store.js';
export { VectorStore } from './vector-store.js';
