// Ashore / Jellys geometry is baked once, not rebuilt in the visitor's browser.
import { palmGeometry } from './vendor/date-palm.js';
import fs from 'node:fs';
import { gzipSync } from 'node:zlib';
const manifest = [];
for (const seed of [44, 288]) for (let tier = 0; tier < 3; tier++) {
  const tree = palmGeometry(seed, tier);
  for (const part of ['trunk', 'leaves']) {
    const g = tree[part], chunks = [], attributes = {};
    let offset = 0;
    for (const name of ['position', 'normal', 'uv', 'color', ...(part === 'leaves' ? ['frondContact'] : [])]) {
      const source = g.attributes[name];
      let data = source.array, normalized = false;
      if (name === 'normal') { data = Int16Array.from(data, x => Math.round(Math.max(-1, Math.min(1, x)) * 32767)); normalized = true; }
      if (name === 'color') { data = Uint8Array.from(data, x => Math.round(Math.max(0, Math.min(1, x)) * 255)); normalized = true; }
      const pad = (4 - offset % 4) % 4; if (pad) { chunks.push(Buffer.alloc(pad)); offset += pad; }
      attributes[name] = { offset, length: data.length, type: data.constructor.name, itemSize: source.itemSize, normalized };
      chunks.push(Buffer.from(data.buffer, data.byteOffset, data.byteLength)); offset += data.byteLength;
    }
    const pad = (4 - offset % 4) % 4; if (pad) { chunks.push(Buffer.alloc(pad)); offset += pad; }
    const indices = g.index.array;
    const index = { offset, length: indices.length, type: indices.constructor.name };
    chunks.push(Buffer.from(indices.buffer, indices.byteOffset, indices.byteLength));
    const file = `palm-${seed}-${tier}-${part}.bin`;
    const data=Buffer.concat(chunks);
    fs.writeFileSync(new URL(`../public/assets/palms-r6/${file}`, import.meta.url), data);
    fs.writeFileSync(new URL(`../public/assets/palms-r6/${file}.gz`, import.meta.url), gzipSync(data,{level:9}));
    manifest.push({ seed, tier, part, file, attributes, index, triangles: indices.length / 3 });
    g.dispose();
  }
}
fs.writeFileSync(new URL('../public/assets/palms-r6/manifest.json', import.meta.url), JSON.stringify(manifest));
console.log(manifest.map(x => ({file:x.file,triangles:x.triangles})));
