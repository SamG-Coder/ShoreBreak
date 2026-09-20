// Mulberry32: bryc JavaScript implementation of Tommy Ettinger's generator, public domain.
// See licenses/Mulberry32-Public-Domain.txt.
import * as THREE from 'three';

// Tileable cellular "foam" noise baked once at startup (RGBA8, mip-mapped), sampled by
// every splash blob so each one is a cluster of bubbles instead of a smooth ball:
//   R: coarse lumps  (1 - F1, 7 cells per tile)       -> cauliflower bumps
//   G: fine lumps    (1 - F1, 15 cells per tile)      -> ragged silhouettes
//   B: F1 of a 10-cell layer (0 at a cell centre, ~1 on the walls) -> erosion into lace
//      (holes open from the cell centres, the walls survive as webs)
//   A: F2 - F1 ridges of the same layer (thin webs)
function mulberry(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function worleyLayer(n, cells, seed) {
  const rnd = mulberry(seed);
  const pts = new Float32Array(cells * cells * 2);
  for (let i = 0; i < cells * cells; i++) { pts[i * 2] = rnd(); pts[i * 2 + 1] = rnd(); }
  const f1 = new Float32Array(n * n), f2 = new Float32Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const px = (x + 0.5) / n * cells, py = (y + 0.5) / n * cells;
      const cx = Math.floor(px), cy = Math.floor(py);
      let d1 = 9, d2 = 9;
      for (let j = -1; j <= 1; j++) {
        for (let i = -1; i <= 1; i++) {
          const gx = cx + i, gy = cy + j;
          const wx = ((gx % cells) + cells) % cells, wy = ((gy % cells) + cells) % cells;
          const k = (wy * cells + wx) * 2;
          const dx = gx + pts[k] - px, dy = gy + pts[k + 1] - py;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) d2 = d;
        }
      }
      f1[y * n + x] = d1; f2[y * n + x] = d2;
    }
  }
  return { f1, f2 };
}

export function makeFoamNoise(n = 128) {
  const A = worleyLayer(n, 7, 11);
  const B = worleyLayer(n, 15, 23);
  const C = worleyLayer(n, 10, 37);
  const data = new Uint8Array(n * n * 4);
  const q = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));
  for (let i = 0; i < n * n; i++) {
    const lumpA = Math.pow(Math.max(1 - A.f1[i] / 0.75, 0), 0.8);
    const lumpB = Math.pow(Math.max(1 - B.f1[i] / 0.75, 0), 0.8);
    data[i * 4] = q(lumpA);
    data[i * 4 + 1] = q(lumpB);
    data[i * 4 + 2] = q(Math.min(C.f1[i] / 0.7, 1));
    data[i * 4 + 3] = q(Math.min((C.f2[i] - C.f1[i]) / 0.5, 1));
  }
  const tex = new THREE.DataTexture(data, n, n, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}
