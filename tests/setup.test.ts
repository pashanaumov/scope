import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, describe, it } from 'node:test';
import { checkSetupStatus, setup } from '../scope/core/setup.js';

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

async function makeTmp(): Promise<string> {
  return mkdtemp(join(tmpdir(), 'setup-test-'));
}

/** Returns a mock fetch that streams tiny content and records called URLs. */
function makeMockFetch(content: Uint8Array = new Uint8Array([0x01, 0x02, 0x03])) {
  const calls: string[] = [];

  const fn = async (url: string | URL | Request): Promise<Response> => {
    calls.push(String(url));
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(content);
        controller.close();
      },
    });
    return {
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: new Headers({ 'content-length': String(content.length) }),
      body: stream,
    } as unknown as Response;
  };

  (fn as typeof fn & { calls: string[] }).calls = calls;
  return fn as typeof fn & { calls: string[] };
}

const BASE_MODEL = 'jinaai/jina-embeddings-v2-base-code';

// ---------------------------------------------------------------------------
// checkSetupStatus — non-existent dirs → all missing
// ---------------------------------------------------------------------------

describe('checkSetupStatus with non-existent dirs', () => {
  it('reports modelReady:false and all grammars missing', async () => {
    const result = await checkSetupStatus({
      modelsDir: '/nonexistent/models',
      grammarsDir: '/nonexistent/grammars',
      transformersModel: BASE_MODEL,
    });

    assert.equal(result.modelReady, false);
    // 9 languages + tree-sitter.wasm
    assert.equal(result.grammarsMissing.length, 10);
    assert.ok(result.grammarsMissing.includes('tree-sitter.wasm'));
    assert.ok(result.grammarsMissing.includes('tree-sitter-typescript.wasm'));
    assert.ok(result.grammarsMissing.includes('tree-sitter-python.wasm'));
  });
});

// ---------------------------------------------------------------------------
// checkSetupStatus — pre-populated dirs → all ready
// ---------------------------------------------------------------------------

describe('checkSetupStatus with pre-populated dirs', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('reports modelReady:true and no grammars missing', async () => {
    const modelsDir = join(tmp, 'models');
    const grammarsDir = join(tmp, 'grammars');

    // Create all HF model files
    const modelDir = join(modelsDir, 'jinaai', 'jina-embeddings-v2-base-code');
    const hfFiles = [
      'onnx/model_quantized.onnx',
      'tokenizer.json',
      'tokenizer_config.json',
      'config.json',
      'special_tokens_map.json',
    ];
    for (const file of hfFiles) {
      const filePath = join(modelDir, ...file.split('/'));
      await mkdir(join(modelDir, ...file.split('/').slice(0, -1)), { recursive: true });
      await writeFile(filePath, 'placeholder');
    }

    // Create all grammar files
    await mkdir(grammarsDir, { recursive: true });
    const langs = ['typescript', 'javascript', 'python', 'rust', 'go', 'java', 'c', 'cpp', 'ruby'];
    await writeFile(join(grammarsDir, 'tree-sitter.wasm'), 'placeholder');
    for (const lang of langs) {
      await writeFile(join(grammarsDir, `tree-sitter-${lang}.wasm`), 'placeholder');
    }

    const result = await checkSetupStatus({
      modelsDir,
      grammarsDir,
      transformersModel: BASE_MODEL,
    });

    assert.equal(result.modelReady, true);
    assert.deepEqual(result.grammarsMissing, []);
  });
});

// ---------------------------------------------------------------------------
// setup with mock fetch — calls onProgress for each file
// ---------------------------------------------------------------------------

describe('setup with mock fetch', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('calls onProgress for all model + grammar files and returns downloaded list', async () => {
    const mockFetch = makeMockFetch();
    const progressEvents: Array<{ item: string; skipped: boolean }> = [];

    const downloaded = await setup({
      config: {
        modelsDir: join(tmp, 'models'),
        grammarsDir: join(tmp, 'grammars'),
        embedder: 'transformers',
        transformersModel: BASE_MODEL,
      },
      onProgress: (p) => progressEvents.push({ item: p.item, skipped: p.skipped }),
      _fetch: mockFetch,
    });

    // 5 model files + 10 grammar files should be downloaded
    assert.equal(downloaded.length, 15);
    assert.ok(downloaded.includes('onnx/model_quantized.onnx'));
    assert.ok(downloaded.includes('tokenizer.json'));
    assert.ok(downloaded.includes('tree-sitter.wasm'));
    assert.ok(downloaded.includes('tree-sitter-typescript.wasm'));

    // All non-done progress events should be non-skipped (fresh dirs)
    const nonDone = progressEvents.filter((e) => e.item !== '');
    assert.ok(
      nonDone.every((e) => !e.skipped),
      'all non-done events should not be skipped',
    );

    // done event emitted at end
    const doneEvents = progressEvents.filter((e) => e.item === '');
    assert.equal(doneEvents.length, 1);

    // fetch called once per downloaded file
    assert.equal(mockFetch.calls.length, 15);
  });
});

// ---------------------------------------------------------------------------
// setup — skips files already present
// ---------------------------------------------------------------------------

describe('setup skips already-present files', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('skips model files that exist on disk', async () => {
    const modelsDir = join(tmp, 'models');
    const grammarsDir = join(tmp, 'grammars');
    const mockFetch = makeMockFetch();
    const skippedItems: string[] = [];

    // Pre-create two HF model files
    const modelDir = join(modelsDir, 'jinaai', 'jina-embeddings-v2-base-code');
    await mkdir(join(modelDir, 'onnx'), { recursive: true });
    await writeFile(join(modelDir, 'onnx', 'model_quantized.onnx'), 'cached');
    await writeFile(join(modelDir, 'tokenizer.json'), 'cached');

    await setup({
      config: { modelsDir, grammarsDir, embedder: 'transformers', transformersModel: BASE_MODEL },
      onProgress: (p) => {
        if (p.skipped) skippedItems.push(p.item);
      },
      _fetch: mockFetch,
    });

    assert.ok(skippedItems.includes('onnx/model_quantized.onnx'));
    assert.ok(skippedItems.includes('tokenizer.json'));
    // fetch should NOT have been called for the two cached files
    assert.ok(!mockFetch.calls.some((url) => url.includes('model_quantized.onnx')));
    assert.ok(!mockFetch.calls.some((url) => url.includes('tokenizer.json')));
  });

  it('skips grammar files that already exist', async () => {
    const modelsDir = join(tmp, 'models2');
    const grammarsDir = join(tmp, 'grammars2');
    await mkdir(grammarsDir, { recursive: true });
    await writeFile(join(grammarsDir, 'tree-sitter-python.wasm'), 'cached');

    const mockFetch = makeMockFetch();
    const skippedItems: string[] = [];

    await setup({
      config: { modelsDir, grammarsDir, embedder: 'openai', transformersModel: BASE_MODEL },
      onProgress: (p) => {
        if (p.skipped) skippedItems.push(p.item);
      },
      _fetch: mockFetch,
    });

    assert.ok(skippedItems.includes('tree-sitter-python.wasm'));
    assert.ok(!mockFetch.calls.some((url) => url.includes('tree-sitter-python')));
  });
});

// ---------------------------------------------------------------------------
// setup — emits OpenAI tip when embedder is 'transformers'
// ---------------------------------------------------------------------------

describe('setup emits OpenAI tip for transformers embedder', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('calls onTip with the local-embeddings message', async () => {
    const tips: string[] = [];

    await setup({
      config: {
        modelsDir: join(tmp, 'models'),
        grammarsDir: join(tmp, 'grammars'),
        embedder: 'transformers',
        transformersModel: BASE_MODEL,
      },
      onTip: (msg) => tips.push(msg),
      _fetch: makeMockFetch(),
    });

    assert.equal(tips.length, 1);
    assert.ok(tips[0]?.includes('jina-embeddings-v2-base-code'));
    assert.ok(tips[0]?.includes('OpenAI'));
  });
});

// ---------------------------------------------------------------------------
// setup — does NOT emit tip when embedder is 'openai'
// ---------------------------------------------------------------------------

describe('setup does not emit tip for openai embedder', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('onTip is never called when embedder is openai', async () => {
    const tips: string[] = [];

    await setup({
      config: {
        modelsDir: join(tmp, 'models'),
        grammarsDir: join(tmp, 'grammars'),
        embedder: 'openai',
        transformersModel: BASE_MODEL,
      },
      onTip: (msg) => tips.push(msg),
      _fetch: makeMockFetch(),
    });

    assert.equal(tips.length, 0);
  });
});
