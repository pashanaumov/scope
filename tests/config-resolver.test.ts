import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { after, before, describe, it } from 'node:test';
import { resolveConfig } from '../scope/core/config-resolver.js';

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

async function makeTmp(): Promise<string> {
  return mkdtemp(join(tmpdir(), 'config-test-'));
}

function sha256hex12(s: string): string {
  return createHash('sha256').update(s).digest('hex').slice(0, 12);
}

async function writeJson(dir: string, filename: string, data: object): Promise<void> {
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, filename), JSON.stringify(data), 'utf-8');
}

// ---------------------------------------------------------------------------
// Default config
// ---------------------------------------------------------------------------

describe('default config', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('returns all correct defaults when no files or flags present', async () => {
    const projectPath = join(tmp, 'project');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home');
    await mkdir(homeDir, { recursive: true });

    const config = await resolveConfig(projectPath, undefined, homeDir);

    const hash = sha256hex12(resolve(projectPath));
    assert.equal(config.indexDir, join(homeDir, '.config', 'scope', 'indexes', hash));
    assert.equal(config.modelsDir, join(homeDir, '.config', 'scope', 'models'));
    assert.equal(config.grammarsDir, join(homeDir, '.config', 'scope', 'grammars'));
    assert.equal(config.embedder, 'transformers');
    assert.equal(config.openaiModel, 'text-embedding-3-small');
    assert.equal(config.ollamaBaseUrl, 'http://localhost:11434');
    assert.equal(config.ollamaModel, 'nomic-embed-text');
    assert.equal(config.ollamaDimensions, 768);
    assert.equal(config.transformersModel, 'jinaai/jina-embeddings-v2-base-code');
    assert.equal(config.chunkMaxLines, 50);
    assert.equal(config.chunkOverlap, 0.2);
    assert.equal(config.chunkStrategy, 'function');
    assert.equal(config.chunkMaxTokens, 512);
    assert.equal(config.concurrency, 10);
    assert.equal(config.batchSize, 32);
    assert.equal(config.searchTopK, 5);
    assert.equal(config.watchEnabled, false);
    assert.equal(config.watchDebounceMs, 2000);
    assert.ok(config.ignorePatterns.includes('node_modules'));
    assert.ok(config.ignorePatterns.includes('.git'));
    assert.ok(config.ignorePatterns.includes('*.d.ts'));
    assert.ok(config.ignorePatterns.includes('.DS_Store'));
  });
});

// ---------------------------------------------------------------------------
// Project hash in default indexDir
// ---------------------------------------------------------------------------

describe('project hash', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('indexDir encodes SHA256(projectPath).slice(0,12)', async () => {
    const projectPath = join(tmp, 'myproject');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home');
    await mkdir(homeDir, { recursive: true });

    const config = await resolveConfig(projectPath, undefined, homeDir);

    const expectedHash = sha256hex12(resolve(projectPath));
    assert.ok(config.indexDir.endsWith(expectedHash));
    assert.equal(config.indexDir, join(homeDir, '.config', 'scope', 'indexes', expectedHash));
  });

  it('different project paths produce different indexDir hashes', async () => {
    const pathA = join(tmp, 'projA');
    const pathB = join(tmp, 'projB');
    await mkdir(pathA, { recursive: true });
    await mkdir(pathB, { recursive: true });
    const homeDir = join(tmp, 'home');
    await mkdir(homeDir, { recursive: true });

    const [cfgA, cfgB] = await Promise.all([
      resolveConfig(pathA, undefined, homeDir),
      resolveConfig(pathB, undefined, homeDir),
    ]);

    assert.notEqual(cfgA.indexDir, cfgB.indexDir);
  });
});

// ---------------------------------------------------------------------------
// CLI flags override defaults
// ---------------------------------------------------------------------------

describe('CLI flags override defaults', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('overrides scalar fields', async () => {
    const projectPath = join(tmp, 'project');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home');
    await mkdir(homeDir, { recursive: true });

    const config = await resolveConfig(
      projectPath,
      { embedder: 'openai', concurrency: 20, searchTopK: 10, watchEnabled: true },
      homeDir,
    );

    assert.equal(config.embedder, 'openai');
    assert.equal(config.concurrency, 20);
    assert.equal(config.searchTopK, 10);
    assert.equal(config.watchEnabled, true);
    // Untouched defaults remain
    assert.equal(config.batchSize, 32);
  });
});

// ---------------------------------------------------------------------------
// .workbench.json overrides defaults
// ---------------------------------------------------------------------------

describe('.workbench.json overrides defaults', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('project .scope.json values override defaults', async () => {
    const projectPath = join(tmp, 'project');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home');
    await mkdir(homeDir, { recursive: true });

    await writeJson(projectPath, '.scope.json', {
      embedder: 'ollama',
      chunkMaxLines: 100,
      batchSize: 16,
    });

    const config = await resolveConfig(projectPath, undefined, homeDir);

    assert.equal(config.embedder, 'ollama');
    assert.equal(config.chunkMaxLines, 100);
    assert.equal(config.batchSize, 16);
    // Defaults preserved for unset keys
    assert.equal(config.concurrency, 10);
    assert.equal(config.searchTopK, 5);
  });
});

// ---------------------------------------------------------------------------
// ~/.workbench/config.json overrides defaults
// ---------------------------------------------------------------------------

describe('global config (~/.config/scope/config.json) overrides defaults', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('global config overrides defaults', async () => {
    const projectPath = join(tmp, 'project');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home');
    await writeJson(join(homeDir, '.config', 'scope'), 'config.json', {
      embedder: 'openai',
      watchEnabled: true,
      watchDebounceMs: 5000,
    });

    const config = await resolveConfig(projectPath, undefined, homeDir);

    assert.equal(config.embedder, 'openai');
    assert.equal(config.watchEnabled, true);
    assert.equal(config.watchDebounceMs, 5000);
    // Defaults preserved
    assert.equal(config.concurrency, 10);
  });
});

// ---------------------------------------------------------------------------
// CLI flags win over all file configs
// ---------------------------------------------------------------------------

describe('CLI flags win over file configs', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('CLI overrides .scope.json, preserves other file values', async () => {
    const projectPath = join(tmp, 'project');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home');
    await mkdir(homeDir, { recursive: true });

    await writeJson(projectPath, '.scope.json', { embedder: 'ollama', concurrency: 5 });

    const config = await resolveConfig(projectPath, { embedder: 'openai' }, homeDir);

    assert.equal(config.embedder, 'openai'); // CLI wins
    assert.equal(config.concurrency, 5); // from file
  });

  it('CLI overrides global config', async () => {
    const projectPath = join(tmp, 'project2');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home2');
    await writeJson(join(homeDir, '.config', 'scope'), 'config.json', { batchSize: 64 });

    const config = await resolveConfig(projectPath, { batchSize: 8 }, homeDir);

    assert.equal(config.batchSize, 8);
  });
});

// ---------------------------------------------------------------------------
// ignorePatterns merging
// ---------------------------------------------------------------------------

describe('ignorePatterns merging', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('unions patterns from all sources', async () => {
    const projectPath = join(tmp, 'project');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home');

    await writeJson(join(homeDir, '.config', 'scope'), 'config.json', {
      ignorePatterns: ['*.log'],
    });
    await writeJson(projectPath, '.scope.json', {
      ignorePatterns: ['tmp/'],
    });

    const config = await resolveConfig(projectPath, { ignorePatterns: ['secrets/'] }, homeDir);

    assert.ok(config.ignorePatterns.includes('node_modules')); // default
    assert.ok(config.ignorePatterns.includes('*.log')); // global
    assert.ok(config.ignorePatterns.includes('tmp/')); // project
    assert.ok(config.ignorePatterns.includes('secrets/')); // CLI
  });

  it('deduplicates patterns that appear in multiple sources', async () => {
    const projectPath = join(tmp, 'dedup');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'homedeup');
    await mkdir(homeDir, { recursive: true });

    // 'node_modules' is already a default; adding it again should yield only one entry
    await writeJson(projectPath, '.scope.json', { ignorePatterns: ['node_modules'] });

    const config = await resolveConfig(projectPath, undefined, homeDir);

    const count = config.ignorePatterns.filter((p) => p === 'node_modules').length;
    assert.equal(count, 1);
  });

  it('project config does not replace default ignorePatterns', async () => {
    const projectPath = join(tmp, 'noreplace');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'homenoreplace');
    await mkdir(homeDir, { recursive: true });

    await writeJson(projectPath, '.scope.json', { ignorePatterns: ['custom/'] });

    const config = await resolveConfig(projectPath, undefined, homeDir);

    // Default patterns still present
    assert.ok(config.ignorePatterns.includes('.git'));
    assert.ok(config.ignorePatterns.includes('custom/'));
  });
});

// ---------------------------------------------------------------------------
// openaiApiKey from environment variable
// ---------------------------------------------------------------------------

describe('openaiApiKey from env', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('picks up OPENAI_API_KEY from env when not in any config', async () => {
    const projectPath = join(tmp, 'project');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home');
    await mkdir(homeDir, { recursive: true });

    process.env.OPENAI_API_KEY = 'test-key-from-env';
    try {
      const config = await resolveConfig(projectPath, undefined, homeDir);
      assert.equal(config.openaiApiKey, 'test-key-from-env');
    } finally {
      delete process.env.OPENAI_API_KEY;
    }
  });

  it('CLI flag openaiApiKey takes priority over env', async () => {
    const projectPath = join(tmp, 'project2');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home2');
    await mkdir(homeDir, { recursive: true });

    process.env.OPENAI_API_KEY = 'env-key';
    try {
      const config = await resolveConfig(projectPath, { openaiApiKey: 'cli-key' }, homeDir);
      assert.equal(config.openaiApiKey, 'cli-key');
    } finally {
      delete process.env.OPENAI_API_KEY;
    }
  });

  it('openaiApiKey is undefined when not set anywhere', async () => {
    const projectPath = join(tmp, 'project3');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home3');
    await mkdir(homeDir, { recursive: true });

    const savedKey = process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_API_KEY;
    try {
      const config = await resolveConfig(projectPath, undefined, homeDir);
      assert.equal(config.openaiApiKey, undefined);
    } finally {
      if (savedKey !== undefined) process.env.OPENAI_API_KEY = savedKey;
    }
  });
});

// ---------------------------------------------------------------------------
// Walk up directory tree to find .workbench.json
// ---------------------------------------------------------------------------

describe('walk up directory tree', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('finds .workbench.json in a parent directory', async () => {
    const rootProject = join(tmp, 'root-project');
    const subDir = join(rootProject, 'a', 'b', 'c');
    await mkdir(subDir, { recursive: true });
    const homeDir = join(tmp, 'home');
    await mkdir(homeDir, { recursive: true });

    await writeJson(rootProject, '.scope.json', { concurrency: 42 });

    // Pass the deep sub-directory — should walk up and find config
    const config = await resolveConfig(subDir, undefined, homeDir);
    assert.equal(config.concurrency, 42);
  });

  it('uses the closest .workbench.json when multiple exist', async () => {
    const parent = join(tmp, 'parent');
    const child = join(parent, 'child');
    await mkdir(child, { recursive: true });
    const homeDir = join(tmp, 'home2');
    await mkdir(homeDir, { recursive: true });

    await writeJson(parent, '.scope.json', { concurrency: 1 });
    await writeJson(child, '.scope.json', { concurrency: 99 });

    const config = await resolveConfig(child, undefined, homeDir);
    assert.equal(config.concurrency, 99);
  });
});

// ---------------------------------------------------------------------------
// Full priority stack: defaults < global < project < CLI
// ---------------------------------------------------------------------------

describe('full priority stack', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('global < project < CLI for the same field', async () => {
    const projectPath = join(tmp, 'project');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home');

    await writeJson(join(homeDir, '.config', 'scope'), 'config.json', { concurrency: 2 });
    await writeJson(projectPath, '.scope.json', { concurrency: 5 });

    const config = await resolveConfig(projectPath, { concurrency: 9 }, homeDir);
    assert.equal(config.concurrency, 9); // CLI wins
  });

  it('project overrides global when CLI absent', async () => {
    const projectPath = join(tmp, 'project2');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home2');

    await writeJson(join(homeDir, '.config', 'scope'), 'config.json', { batchSize: 8 });
    await writeJson(projectPath, '.scope.json', { batchSize: 64 });

    const config = await resolveConfig(projectPath, undefined, homeDir);
    assert.equal(config.batchSize, 64); // project wins over global
  });

  it('global overrides defaults when project and CLI absent', async () => {
    const projectPath = join(tmp, 'project3');
    await mkdir(projectPath, { recursive: true });
    const homeDir = join(tmp, 'home3');

    await writeJson(join(homeDir, '.config', 'scope'), 'config.json', { searchTopK: 20 });

    const config = await resolveConfig(projectPath, undefined, homeDir);
    assert.equal(config.searchTopK, 20); // global wins over defaults
  });
});
