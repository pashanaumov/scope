import { createWriteStream } from 'node:fs';
import { access, mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import type { ScopeConfig } from './config-resolver.js';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type SetupPhase = 'model' | 'grammars' | 'done';

export interface SetupProgress {
  phase: SetupPhase;
  item: string;
  bytesTotal: number; // 0 if unknown (no content-length)
  bytesDone: number;
  skipped: boolean; // true if already cached
}

export type ProgressCallback = (progress: SetupProgress) => void;

export interface SetupOptions {
  config: Pick<ScopeConfig, 'modelsDir' | 'grammarsDir' | 'embedder' | 'transformersModel'>;
  onProgress?: ProgressCallback;
  onTip?: (message: string) => void;
  /** @internal Injectable fetch for testing — defaults to globalThis.fetch */
  _fetch?: (input: string | URL | Request, init?: RequestInit) => Promise<Response>;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HF_FILES = [
  'onnx/model_quantized.onnx',
  'tokenizer.json',
  'tokenizer_config.json',
  'config.json',
  'special_tokens_map.json',
] as const;

const GRAMMAR_LANGUAGES = [
  'typescript',
  'javascript',
  'python',
  'rust',
  'go',
  'java',
  'c',
  'cpp',
  'ruby',
] as const;

const OPENAI_TIP = [
  '✓ Using local embeddings (jina-embeddings-v2-base-code, ~87 chunks/s, code-specialized)',
  '  Tip: large codebases index 5-20x faster using OpenAI:',
  '  Set OPENAI_API_KEY or add "embedder": "openai" to .workbench.json',
].join('\n');

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function downloadFile(
  url: string,
  destPath: string,
  phase: SetupPhase,
  item: string,
  onProgress: ProgressCallback | undefined,
  fetchFn: (input: string | URL | Request, init?: RequestInit) => Promise<Response>,
): Promise<void> {
  await mkdir(dirname(destPath), { recursive: true });

  const response = await fetchFn(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} downloading ${item} from ${url}`);
  }

  const bytesTotal = parseInt(response.headers.get('content-length') ?? '0', 10) || 0;
  let bytesDone = 0;

  const ws = createWriteStream(destPath);
  const reader = response.body?.getReader();
  if (!reader) throw new Error(`Download failed: response body is null for ${destPath}`);

  await new Promise<void>((resolve, reject) => {
    ws.on('finish', resolve);
    ws.on('error', reject);

    (async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          bytesDone += value.length;
          onProgress?.({ phase, item, bytesTotal, bytesDone, skipped: false });
          // Respect backpressure
          if (!ws.write(value)) {
            await new Promise<void>((res, rej) => {
              ws.once('drain', res);
              ws.once('error', rej);
            });
          }
        }
        ws.end();
      } catch (e) {
        reject(e);
      }
    })();
  });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Check which assets are missing without downloading anything. */
export async function checkSetupStatus(
  config: Pick<ScopeConfig, 'modelsDir' | 'grammarsDir' | 'transformersModel'>,
): Promise<{ modelReady: boolean; grammarsMissing: string[] }> {
  const modelDir = join(config.modelsDir, ...config.transformersModel.split('/'));

  const modelChecks = await Promise.all(
    HF_FILES.map((file) => fileExists(join(modelDir, ...file.split('/')))),
  );
  const modelReady = modelChecks.every(Boolean);

  const grammarsMissing: string[] = [];

  if (!(await fileExists(join(config.grammarsDir, 'tree-sitter.wasm')))) {
    grammarsMissing.push('tree-sitter.wasm');
  }
  for (const lang of GRAMMAR_LANGUAGES) {
    const filename = `tree-sitter-${lang}.wasm`;
    if (!(await fileExists(join(config.grammarsDir, filename)))) {
      grammarsMissing.push(filename);
    }
  }

  return { modelReady, grammarsMissing };
}

/**
 * Run first-time setup: download model + grammars as needed.
 * Returns list of items that were downloaded (empty if everything was cached).
 */
export async function setup(options: SetupOptions): Promise<string[]> {
  const { config, onProgress, onTip, _fetch = globalThis.fetch } = options;
  const downloaded: string[] = [];

  // ------------------------------------------------------------------
  // HF model (only for transformers embedder)
  // ------------------------------------------------------------------
  if (config.embedder === 'transformers') {
    const modelDir = join(config.modelsDir, ...config.transformersModel.split('/'));

    for (const file of HF_FILES) {
      const destPath = join(modelDir, ...file.split('/'));

      if (await fileExists(destPath)) {
        onProgress?.({ phase: 'model', item: file, bytesTotal: 0, bytesDone: 0, skipped: true });
        continue;
      }

      const url = `https://huggingface.co/${config.transformersModel}/resolve/main/${file}`;
      await downloadFile(url, destPath, 'model', file, onProgress, _fetch);
      downloaded.push(file);
    }

    onTip?.(OPENAI_TIP);
  }

  // ------------------------------------------------------------------
  // Tree-sitter WASM grammars (always)
  // ------------------------------------------------------------------
  await mkdir(config.grammarsDir, { recursive: true });

  const wasmCorePath = join(config.grammarsDir, 'tree-sitter.wasm');
  if (await fileExists(wasmCorePath)) {
    onProgress?.({
      phase: 'grammars',
      item: 'tree-sitter.wasm',
      bytesTotal: 0,
      bytesDone: 0,
      skipped: true,
    });
  } else {
    await downloadFile(
      'https://cdn.jsdelivr.net/npm/web-tree-sitter@latest/tree-sitter.wasm',
      wasmCorePath,
      'grammars',
      'tree-sitter.wasm',
      onProgress,
      _fetch,
    );
    downloaded.push('tree-sitter.wasm');
  }

  for (const lang of GRAMMAR_LANGUAGES) {
    const filename = `tree-sitter-${lang}.wasm`;
    const destPath = join(config.grammarsDir, filename);

    if (await fileExists(destPath)) {
      onProgress?.({
        phase: 'grammars',
        item: filename,
        bytesTotal: 0,
        bytesDone: 0,
        skipped: true,
      });
      continue;
    }

    const url = `https://cdn.jsdelivr.net/npm/tree-sitter-${lang}@latest/tree-sitter-${lang}.wasm`;
    await downloadFile(url, destPath, 'grammars', filename, onProgress, _fetch);
    downloaded.push(filename);
  }

  onProgress?.({ phase: 'done', item: '', bytesTotal: 0, bytesDone: 0, skipped: false });

  return downloaded;
}
