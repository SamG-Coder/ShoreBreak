// Mulberry32: bryc JavaScript implementation of Tommy Ettinger's generator, public domain.
// See licenses/Mulberry32-Public-Domain.txt.
import * as THREE from 'three';

// Tileable smooth-noise texture for the beach's low-frequency fields (sorting patches, drying
// patches, upper-beach swells, scarps). One texture fetch replaces a gradient-noise evaluation
// (cheaper to run and far cheaper to compile), and its mip chain band-limits the fields at
// grazing distances for free.
//   RGBA8, 256^2, repeat. 32 lattice cells per period (8 texels per cell), two octaves, four
//   independent channels. GLSL: macroNoise(p, lam) = texture(uMacroNoise, p / (32 * lam)) * 2 - 1,
//   features ~lam metres, values ~ +-1.

export const MACRO_NOISE_GLSL = /* glsl */ `
uniform sampler2D uMacroNoise;
vec4 macroNoise(vec2 p, vec2 lam) { return texture(uMacroNoise, p / (32.0 * lam)) * 2.0 - 1.0; }
vec4 macroNoiseLod(vec2 p, vec2 lam) { return textureLod(uMacroNoise, p / (32.0 * lam), 0.0) * 2.0 - 1.0; }
// explicit derivatives (dx, dy = screen derivatives of p): safe inside non-uniform branches
vec4 macroNoiseG(vec2 p, vec2 lam, vec2 dx, vec2 dy) { vec2 s = 1.0 / (32.0 * lam); return textureGrad(uMacroNoise, p * s, dx * s, dy * s) * 2.0 - 1.0; }
`;

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeMacroNoise(N = 256, cells = 32) {
  const data = new Uint8Array(N * N * 4);
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
  for (let ch = 0; ch < 4; ch++) {
    const rnd = mulberry32(9127 + ch * 7919);
    const octs = [[cells, 0.68], [cells * 2, 0.32]].map(([c, w]) => {
      const g = new Float32Array(c * c);
      for (let i = 0; i < g.length; i++) g[i] = rnd() * 2 - 1;
      return { c, w, g };
    });
    const v = new Float32Array(N * N);
    let lo = Infinity, hi = -Infinity;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      let s = 0;
      for (const { c, w, g } of octs) {
        const fx = (x / N) * c, fy = (y / N) * c;
        const ix = Math.floor(fx), iy = Math.floor(fy);
        const tx = fade(fx - ix), ty = fade(fy - iy);
        const at = (i, j) => g[((j + c) % c) * c + ((i + c) % c)];
        const a = at(ix, iy), b = at(ix + 1, iy), cc = at(ix, iy + 1), d = at(ix + 1, iy + 1);
        s += w * (a + (b - a) * tx + (cc - a) * ty + (a - b - cc + d) * tx * ty);
      }
      v[y * N + x] = s;
      lo = Math.min(lo, s); hi = Math.max(hi, s);
    }
    // symmetric normalisation about 0 (the channel mean stays ~0.5)
    const m = Math.max(-lo, hi);
    for (let i = 0; i < N * N; i++) data[i * 4 + ch] = Math.round(Math.min(Math.max(0.5 + 0.5 * v[i] / m, 0), 1) * 255);
  }
  const t = new THREE.DataTexture(data, N, N, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.generateMipmaps = true;
  t.anisotropy = 8;
  t.colorSpace = THREE.NoColorSpace;
  t.needsUpdate = true;
  return t;
}
