// Verify the complete bundled runtime asset inventory, without network access.
import { readFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const manifest = JSON.parse(await readFile(path.join(root, 'docs/asset-provenance.json'), 'utf8'));
const errors = [];
const expected = new Map();
for (const item of manifest.bundled_files) {
  if (expected.has(item.path)) errors.push(`Duplicate entry: ${item.path}`);
  expected.set(item.path, item);
  const absolute = path.resolve(root, item.path);
  if (!absolute.startsWith(path.join(root, 'public') + path.sep)) {
    errors.push(`Asset outside public/: ${item.path}`);
    continue;
  }
  try {
    const bytes = await readFile(absolute);
    const hash = createHash('sha256').update(bytes).digest('hex');
    if (hash !== item.sha256 || bytes.length !== item.bytes) errors.push(`Changed: ${item.path}`);
    if (!item.license || !item.source_id) errors.push(`Missing provenance: ${item.path}`);
  } catch {
    errors.push(`Missing: ${item.path}`);
  }
}
async function visit(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) await visit(absolute);
    else {
      const relative = path.relative(root, absolute).split(path.sep).join('/');
      if (!expected.has(relative)) errors.push(`Unlisted public file: ${relative}`);
    }
  }
}
await visit(path.join(root, 'public'));
if (errors.length) {
  console.error(errors.join('\n'));
  process.exitCode = 1;
} else {
  console.log(`Verified ${expected.size} bundled files: hashes, sizes, and provenance coverage.`);
}
