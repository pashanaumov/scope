import { createHash } from 'node:crypto';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';

export interface ManifestEntry {
  mtime: number;
  size: number;
  sha256: string;
}

export type Manifest = Record<string, ManifestEntry>;

export interface ManifestDiff {
  new: string[];
  changed: string[];
  deleted: string[];
}

export async function loadManifest(manifestFilePath: string): Promise<Manifest> {
  try {
    const content = await readFile(manifestFilePath, 'utf8');
    return JSON.parse(content) as Manifest;
  } catch (err: unknown) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return {};
    throw err;
  }
}

export async function saveManifest(manifestFilePath: string, manifest: Manifest): Promise<void> {
  await mkdir(dirname(manifestFilePath), { recursive: true });
  await writeFile(manifestFilePath, JSON.stringify(manifest, null, 2), 'utf8');
}

function sha256(content: Buffer): string {
  return createHash('sha256').update(content).digest('hex');
}

export async function diffManifest(
  projectPath: string,
  currentFiles: Record<string, { mtime: number; size: number }>,
  stored: Manifest,
): Promise<{ diff: ManifestDiff; updated: Manifest }> {
  const updated: Manifest = { ...stored };
  const diff: ManifestDiff = { new: [], changed: [], deleted: [] };

  // Deleted: in stored but not in currentFiles
  for (const relPath of Object.keys(stored)) {
    if (!(relPath in currentFiles)) {
      diff.deleted.push(relPath);
      delete updated[relPath];
    }
  }

  // New or potentially changed
  for (const [relPath, stat] of Object.entries(currentFiles)) {
    const entry = stored[relPath];

    if (!entry) {
      // New file — compute SHA256
      const content = await readFile(resolve(projectPath, relPath));
      updated[relPath] = { mtime: stat.mtime, size: stat.size, sha256: sha256(content) };
      diff.new.push(relPath);
      continue;
    }

    if (entry.mtime === stat.mtime && entry.size === stat.size) {
      // Fast-path: unchanged
      continue;
    }

    // mtime or size changed — compute SHA256 to confirm
    const content = await readFile(resolve(projectPath, relPath));
    const hash = sha256(content);

    if (hash !== entry.sha256) {
      updated[relPath] = { mtime: stat.mtime, size: stat.size, sha256: hash };
      diff.changed.push(relPath);
    } else {
      // Content unchanged — update mtime/size only (e.g. touch'd file)
      updated[relPath] = { mtime: stat.mtime, size: stat.size, sha256: hash };
    }
  }

  return { diff, updated };
}

export async function updateManifest(
  projectPath: string,
  manifest: Manifest,
  filePaths: string[],
): Promise<Manifest> {
  const result: Manifest = { ...manifest };

  await Promise.all(
    filePaths.map(async (relPath) => {
      const absPath = resolve(projectPath, relPath);
      const [content, s] = await Promise.all([readFile(absPath), stat(absPath)]);
      result[relPath] = { mtime: s.mtimeMs, size: s.size, sha256: sha256(content) };
    }),
  );

  return result;
}

export function manifestPath(projectPath: string, homeDir?: string): string {
  const absPath = resolve(projectPath);
  const hash = createHash('sha256').update(absPath).digest('hex').slice(0, 12);
  const base = homeDir ?? homedir();
  return join(base, '.workbench', hash, 'manifest.json');
}
