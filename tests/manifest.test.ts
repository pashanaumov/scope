import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { after, before, describe, it } from 'node:test';
import {
  diffManifest,
  loadManifest,
  manifestPath,
  saveManifest,
  updateManifest,
} from '../scope/core/manifest.js';

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

async function makeTmp(): Promise<string> {
  return mkdtemp(join(tmpdir(), 'manifest-test-'));
}

async function touch(filePath: string, content: string = 'hello'): Promise<void> {
  await mkdir(join(filePath, '..'), { recursive: true });
  await writeFile(filePath, content, 'utf8');
}

function sha256(s: string): string {
  return createHash('sha256').update(s).digest('hex');
}

// ---------------------------------------------------------------------------
// loadManifest
// ---------------------------------------------------------------------------

describe('loadManifest', () => {
  it('returns empty object for non-existent file', async () => {
    const result = await loadManifest('/nonexistent/path/manifest.json');
    assert.deepEqual(result, {});
  });
});

// ---------------------------------------------------------------------------
// saveManifest / round-trip
// ---------------------------------------------------------------------------

describe('saveManifest', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('creates directories and writes correctly', async () => {
    const mPath = join(tmp, 'deep', 'nested', 'manifest.json');
    const manifest = { 'src/foo.ts': { mtime: 1000, size: 42, sha256: 'abc' } };
    await saveManifest(mPath, manifest);
    const loaded = await loadManifest(mPath);
    assert.deepEqual(loaded, manifest);
  });

  it('round-trip: save then load gives same result', async () => {
    const mPath = join(tmp, 'round-trip', 'manifest.json');
    const manifest = {
      'a.ts': { mtime: 111, size: 10, sha256: sha256('a') },
      'b.ts': { mtime: 222, size: 20, sha256: sha256('b') },
    };
    await saveManifest(mPath, manifest);
    const loaded = await loadManifest(mPath);
    assert.deepEqual(loaded, manifest);
  });
});

// ---------------------------------------------------------------------------
// diffManifest
// ---------------------------------------------------------------------------

describe('diffManifest', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('identifies new files not in stored manifest', async () => {
    const content = 'new file content';
    await touch(join(tmp, 'new.ts'), content);
    const s = await stat(join(tmp, 'new.ts'));

    const { diff } = await diffManifest(tmp, { 'new.ts': { mtime: s.mtimeMs, size: s.size } }, {});
    assert.deepEqual(diff.new, ['new.ts']);
    assert.deepEqual(diff.changed, []);
    assert.deepEqual(diff.deleted, []);
  });

  it('identifies deleted files (in stored but not in currentFiles)', async () => {
    const stored = { 'gone.ts': { mtime: 1, size: 1, sha256: 'aaa' } };
    const { diff } = await diffManifest(tmp, {}, stored);
    assert.deepEqual(diff.deleted, ['gone.ts']);
    assert.deepEqual(diff.new, []);
    assert.deepEqual(diff.changed, []);
  });

  it('unchanged file (mtime+size match) → not in diff', async () => {
    const content = 'stable';
    await touch(join(tmp, 'stable.ts'), content);
    const s = await stat(join(tmp, 'stable.ts'));
    const stored = {
      'stable.ts': { mtime: s.mtimeMs, size: s.size, sha256: sha256(content) },
    };

    const { diff } = await diffManifest(
      tmp,
      { 'stable.ts': { mtime: s.mtimeMs, size: s.size } },
      stored,
    );
    assert.deepEqual(diff.new, []);
    assert.deepEqual(diff.changed, []);
    assert.deepEqual(diff.deleted, []);
  });

  it('mtime changed, SHA256 unchanged → not in changed list', async () => {
    const content = 'touchable';
    await touch(join(tmp, 'touched.ts'), content);
    const s = await stat(join(tmp, 'touched.ts'));
    const stored = {
      'touched.ts': { mtime: s.mtimeMs - 1000, size: s.size, sha256: sha256(content) },
    };

    const { diff, updated } = await diffManifest(
      tmp,
      { 'touched.ts': { mtime: s.mtimeMs, size: s.size } },
      stored,
    );
    assert.deepEqual(diff.changed, []);
    assert.deepEqual(diff.new, []);
    // mtime updated in-place
    assert.equal(updated['touched.ts'].mtime, s.mtimeMs);
  });

  it('mtime changed, SHA256 changed → in changed list', async () => {
    const oldContent = 'old content';
    const newContent = 'new content';
    await touch(join(tmp, 'modified.ts'), newContent);
    const s = await stat(join(tmp, 'modified.ts'));
    const stored = {
      'modified.ts': { mtime: s.mtimeMs - 1000, size: s.size, sha256: sha256(oldContent) },
    };

    const { diff } = await diffManifest(
      tmp,
      { 'modified.ts': { mtime: s.mtimeMs, size: s.size } },
      stored,
    );
    assert.deepEqual(diff.changed, ['modified.ts']);
  });
});

// ---------------------------------------------------------------------------
// updateManifest
// ---------------------------------------------------------------------------

describe('updateManifest', () => {
  let tmp: string;
  before(async () => {
    tmp = await makeTmp();
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  it('writes correct entries for new files', async () => {
    const content = 'update me';
    await touch(join(tmp, 'upd.ts'), content);
    const s = await stat(join(tmp, 'upd.ts'));

    const result = await updateManifest(tmp, {}, ['upd.ts']);
    assert.equal(result['upd.ts'].sha256, sha256(content));
    assert.equal(result['upd.ts'].size, s.size);
    assert.equal(typeof result['upd.ts'].mtime, 'number');
  });

  it('preserves existing entries not in filePaths', async () => {
    const existing = { 'old.ts': { mtime: 1, size: 2, sha256: 'xyz' } };
    await touch(join(tmp, 'newfile.ts'), 'hello');
    const result = await updateManifest(tmp, existing, ['newfile.ts']);
    assert.deepEqual(result['old.ts'], existing['old.ts']);
    assert.ok(result['newfile.ts']);
  });
});

// ---------------------------------------------------------------------------
// manifestPath
// ---------------------------------------------------------------------------

describe('manifestPath', () => {
  it('generates correct path from project path', () => {
    const absPath = resolve('/some/project');
    const expectedHash = createHash('sha256').update(absPath).digest('hex').slice(0, 12);
    const result = manifestPath(absPath, '/home/user');
    assert.equal(result, `/home/user/.workbench/${expectedHash}/manifest.json`);
  });

  it('uses homedir() when homeDir not provided', () => {
    // just verify it contains .workbench and ends with manifest.json
    const result = manifestPath('/any/path');
    assert.ok(result.includes('.workbench'));
    assert.ok(result.endsWith('manifest.json'));
  });
});
