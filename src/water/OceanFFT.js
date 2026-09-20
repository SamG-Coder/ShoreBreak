// Mulberry32: bryc JavaScript implementation of Tommy Ettinger's generator, public domain.
// See licenses/Mulberry32-Public-Domain.txt.
import * as THREE from 'three';
import { FullscreenPass, makeShader } from '../core/gpu.js';
import { OCEAN } from '../glsl/water.js';

// GPU FFT wind sea (Tessendorf 2001) in three non-commensurate cascades.
//
// Spectrum: JONSWAP for the measured light onshore sea breeze (far_sea.md §6, §10.2):
// U10 = 3 m/s, fetch 2 km -> peak 1.13 s / 2.0 m, Hs ≈ 0.07 m, gamma 3.3, cos² spreading
// about +z (counter-propagating energy suppressed x0.1), capillary cut-off at ~1.2 cm.
//
// Each cascade holds one band of the spectrum so the sum is the whole spectrum:
//   C0  41.0 m patch   2.5 m < λ           (swell-like chop, displaces the mesh)
//   C1   7.3 m patch   0.30 m < λ < 2.5 m  (dominant wind waves, displacement + normals)
//   C2   1.37 m patch  λ < 0.30 m          (short gravity-capillary waves, normals)
// Per frame: spectrum evolution -> 8 horizontal + 8 vertical radix-2 Stockham passes on a
// 3N x N atlas (two RGBA32F targets = 3 complex fields: h+i·sx, sz+i·dx, dz) -> per-cascade
// compose into mipmapped RGBA16F textures:
//   D_c = (dx, h, dz, 0)           displacement (m)
//   S_c = (sx, sz, sx², sz²)       slopes and second moments (LEAN mapping: the trilinear /
//                                  anisotropic mip filter gives the per-pixel mean slope and
//                                  the unresolved slope variance E[s²] − E[s]²)

export const OCEAN_N = OCEAN.N;
export const OCEAN_L = OCEAN.L;
const BAND = [[0, (2 * Math.PI) / 2.5], [(2 * Math.PI) / 2.5, (2 * Math.PI) / 0.3], [(2 * Math.PI) / 0.3, 1e9]];
export const OCEAN_CHOP = [0.85, 0.8, 0.5];   // Tessendorf choppiness λ per cascade

const G = 9.81;
// normalisation of the spreading function cos^s (forward) + b cos^s (backward) over [-pi, pi]
const NORM = (s, b) => { let a = 0; for (let i = 0; i < 2000; i++) { const t = -Math.PI + (2 * Math.PI * (i + 0.5)) / 2000; const c = Math.cos(t); a += (c > 0 ? Math.pow(c, s) : b * Math.pow(-c, s)) * (2 * Math.PI / 2000); } return a; };
const KM = 370.0;          // gravity-capillary crossover (rad/m)

function jonswap(w, { U10 = 3.0, fetch = 2000.0, gamma = 3.3 } = {}) {
  const wp = 22 * Math.pow((G * G) / (U10 * fetch), 1 / 3);
  const alpha = 0.076 * Math.pow((U10 * U10) / (fetch * G), 0.22);
  const sig = w <= wp ? 0.07 : 0.09;
  const r = Math.exp(-((w - wp) ** 2) / (2 * sig * sig * wp * wp));
  return (alpha * G * G) / w ** 5 * Math.exp(-1.25 * (wp / w) ** 4) * Math.pow(gamma, r);
}

function makeRng(seed) {
  let a = seed >>> 0;
  const rnd = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return () => { // standard normal pair (Box-Muller)
    const u = Math.max(rnd(), 1e-9), v = rnd();
    const m = Math.sqrt(-2 * Math.log(u));
    return [m * Math.cos(2 * Math.PI * v), m * Math.sin(2 * Math.PI * v)];
  };
}

/** Initial spectrum h0(k) for all cascades, natural FFT order: texel (c*N + n, m) = (h0(k), h0(-k)). */
function buildH0({ seed = 1234, windDeg = 5, spread = [1.2, 0.6, 0.5], back = [0.1, 0.3, 0.4], ampScale = 1.0 } = {}) {
  const N = OCEAN_N;
  const data = new Float32Array(3 * N * N * 4);
  const h0 = [];
  const stats = [];
  const gauss = makeRng(seed);
  for (let c = 0; c < 3; c++) {
    // each cascade is sampled in a rotated frame (u = R(rot) xz / L): rotate its spectrum back
    const thw = ((windDeg - OCEAN.ROT[c]) * Math.PI) / 180;
    const L = OCEAN_L[c], dk = (2 * Math.PI) / L;
    const sp = spread[c], bk = back[c], spreadingNorm = NORM(sp, bk);
    const re = new Float32Array(N * N), im = new Float32Array(N * N);
    let var_ = 0, sx2 = 0, sz2 = 0;
    for (let m = 0; m < N; m++) for (let n = 0; n < N; n++) {
      const kx = dk * (n < N / 2 ? n : n - N), kz = dk * (m < N / 2 ? m : m - N);
      const k = Math.hypot(kx, kz);
      const [g1, g2] = gauss();
      if (k < 1e-6 || k < BAND[c][0] || k >= BAND[c][1]) continue;
      const w = Math.sqrt(G * k * (1 + (k / KM) ** 2));
      const dwdk = (G * (1 + 3 * (k / KM) ** 2)) / (2 * w);
      // directional spreading about the wind (angle measured from +z toward +x)
      let d = Math.atan2(kx, kz) - thw;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      const cs = Math.cos(d);
      // short waves are shorter-crested (broader, partly bimodal spreading): the far sea
      // reads as a chaotic network of short crest lines, not long parallel rows
      const D = (cs > 0 ? Math.pow(cs, sp) : bk * Math.pow(-cs, sp)) / spreadingNorm;
      const cap = Math.exp(-((k / 520) ** 2));          // capillary damping (λ < ~1.2 cm)
      const Sk = (jonswap(w) * dwdk / k) * D * cap * ampScale;
      const e = Math.sqrt((Sk * dk * dk) / 2) / Math.SQRT2;
      re[m * N + n] = g1 * e; im[m * N + n] = g2 * e;
      var_ += Sk * dk * dk; sx2 += Sk * dk * dk * kx * kx; sz2 += Sk * dk * dk * kz * kz;
    }
    for (let m = 0; m < N; m++) for (let n = 0; n < N; n++) {
      const nm = (N - n) % N, mm = (N - m) % N;
      const o = ((m * 3 * N) + c * N + n) * 4;
      data[o] = re[m * N + n]; data[o + 1] = im[m * N + n];
      data[o + 2] = re[mm * N + nm]; data[o + 3] = im[mm * N + nm];
    }
    h0.push({ re, im });
    stats.push({ L, Hs: 4 * Math.sqrt(var_), mssX: sx2, mssZ: sz2 });
  }
  return { data, stats };
}

const SPECTRUM = /* glsl */ `
precision highp float;
precision highp int;
uniform sampler2D uH0;
uniform float uT;
uniform vec3 uL;
layout(location = 1) out vec4 out1;
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int c = p.x / ${OCEAN_N};
  int n = p.x - c * ${OCEAN_N};
  int m = p.y;
  float L = c == 0 ? uL.x : (c == 1 ? uL.y : uL.z);
  float dk = 6.283185307 / L;
  vec2 k = dk * vec2(n < ${OCEAN_N / 2} ? n : n - ${OCEAN_N}, m < ${OCEAN_N / 2} ? m : m - ${OCEAN_N});
  float kl = length(k);
  vec4 h0 = texelFetch(uH0, p, 0);
  float w = sqrt(${G} * kl * (1.0 + kl * kl / ${KM * KM}.0));
  // wrap the phase in double-ish precision: w*t can be large
  float ph = mod(w * uT, 6.283185307);
  vec2 e = vec2(cos(ph), sin(ph));
  // h(k,t) = h0(k) e^{-iwt} + conj(h0(-k)) e^{+iwt}   (energy of h0(k) travels along +k)
  vec2 h = cmul(h0.xy, vec2(e.x, -e.y)) + cmul(vec2(h0.z, -h0.w), e);
  vec2 ih = vec2(-h.y, h.x);                     // i h
  vec2 kn = kl > 1e-6 ? k / kl : vec2(0.0);
  // pack pairs of real fields: A = h + i sx, B = sz + i dx, C = dz
  vec2 sx = k.x * ih, sz = k.y * ih;             // i kx h, i kz h
  vec2 dx = -kn.x * ih, dz = -kn.y * ih;         // -i kx/k h, -i kz/k h
  vec2 A = h + vec2(-sx.y, sx.x);
  vec2 B = sz + vec2(-dx.y, dx.x);
  gl_FragColor = vec4(A, B);
  out1 = vec4(dz, 0.0, 0.0);
}`;

const FFT = /* glsl */ `
precision highp float;
precision highp int;
uniform sampler2D uIn0;
uniform sampler2D uIn1;
uniform int uSub;
uniform int uHoriz;
layout(location = 1) out vec4 out1;
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int idx = uHoriz == 1 ? (p.x & ${OCEAN_N - 1}) : p.y;
  int hs = uSub >> 1;
  int ev = (idx / uSub) * hs + (idx & (hs - 1));
  int od = ev + ${OCEAN_N / 2};
  ivec2 pe = uHoriz == 1 ? ivec2(p.x - idx + ev, p.y) : ivec2(p.x, ev);
  ivec2 po = uHoriz == 1 ? ivec2(p.x - idx + od, p.y) : ivec2(p.x, od);
  float ang = 6.283185307 * float(idx & (uSub - 1)) / float(uSub);
  vec2 w = vec2(cos(ang), sin(ang));
  vec4 e0 = texelFetch(uIn0, pe, 0), o0 = texelFetch(uIn0, po, 0);
  vec4 e1 = texelFetch(uIn1, pe, 0), o1 = texelFetch(uIn1, po, 0);
  gl_FragColor = vec4(e0.xy + cmul(w, o0.xy), e0.zw + cmul(w, o0.zw));
  out1 = vec4(e1.xy + cmul(w, o1.xy), e1.zw + cmul(w, o1.zw));
}`;

const COMPOSE = /* glsl */ `
precision highp float;
precision highp int;
uniform sampler2D uIn0;
uniform sampler2D uIn1;
uniform int uCascade;
uniform float uChopK;
layout(location = 1) out vec4 outS;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 q = ivec2(p.x + uCascade * ${OCEAN_N}, p.y);
  vec4 a = texelFetch(uIn0, q, 0);
  vec4 b = texelFetch(uIn1, q, 0);
  float h = a.x, sx = a.y, sz = a.z, dx = a.w, dz = b.x;
  gl_FragColor = vec4(dx * uChopK, h, dz * uChopK, 0.0);
  outS = vec4(sx, sz, sx * sx, sz * sz);
}`;

export class OceanFFT {
  constructor(opts = {}) {
    const N = OCEAN_N;
    const { data, stats } = buildH0(opts);
    this.stats = stats;
    this.h0 = new THREE.DataTexture(data, 3 * N, N, THREE.RGBAFormat, THREE.FloatType);
    this.h0.minFilter = this.h0.magFilter = THREE.NearestFilter;
    this.h0.needsUpdate = true;
    const mk = () => {
      const rt = new THREE.WebGLRenderTarget(3 * N, N, {
        count: 2, type: THREE.FloatType, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
        depthBuffer: false, stencilBuffer: false, generateMipmaps: false,
      });
      rt.textures.forEach((t) => { t.colorSpace = THREE.NoColorSpace; });
      return rt;
    };
    this.ping = mk();
    this.pong = mk();
    const options = { type: THREE.HalfFloatType, minFilter: THREE.LinearMipmapLinearFilter,
      magFilter: THREE.LinearFilter, wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping,
      depthBuffer: false, stencilBuffer: false, generateMipmaps: true };
    this.cascades = new THREE.WebGLArrayRenderTarget(N, N, 3, options);
    const slopes = new THREE.DataArrayTexture(null, N, N, 3);
    Object.assign(slopes, options, {isRenderTargetTexture: true});
    this.cascades.textures.push(slopes);
    this.cascades.textures.forEach(t => { t.colorSpace = THREE.NoColorSpace; t.anisotropy = 8; });
    const L = new THREE.Vector3(...OCEAN_L);
    this.pSpec = new FullscreenPass(makeShader(SPECTRUM, { uH0: { value: this.h0 }, uT: { value: 0 }, uL: { value: L } }));
    this.pFFT = new FullscreenPass(makeShader(FFT, { uIn0: { value: null }, uIn1: { value: null }, uSub: { value: 2 }, uHoriz: { value: 1 } }));
    this.pCompose = new FullscreenPass(makeShader(COMPOSE, { uIn0: { value: null }, uIn1: { value: null }, uCascade: { value: 0 }, uChopK: { value: 1 } }));
    this.time = NaN;
  }

  /** Displacement / slope textures of cascade c. */
  disp() { return this.cascades.textures[0]; }
  slope() { return this.cascades.textures[1]; }

  update(renderer, t) {
    if (t === this.time) return;
    this.time = t;
    if (!this.initialized) {
      // Allocate the full mip chains while generateMipmaps is enabled.
      renderer.initRenderTarget?.(this.cascades);
      this.initialized = true;
    }
    const prev = renderer.getRenderTarget();
    const prevCube = renderer.getActiveCubeFace(), prevMip = renderer.getActiveMipmapLevel();
    this.pSpec.material.uniforms.uT.value = t;
    this.pSpec.render(renderer, this.ping);
    let src = this.ping, dst = this.pong;
    const u = this.pFFT.material.uniforms;
    for (const horiz of [1, 0]) {
      for (let sub = 2; sub <= OCEAN_N; sub *= 2) {
        u.uIn0.value = src.textures[0]; u.uIn1.value = src.textures[1];
        u.uSub.value = sub; u.uHoriz.value = horiz;
        this.pFFT.render(renderer, dst);
        const tmp = src; src = dst; dst = tmp;
      }
    }
    const cu = this.pCompose.material.uniforms;
    cu.uIn0.value = src.textures[0]; cu.uIn1.value = src.textures[1];
    for (let c = 0; c < 3; c++) {
      cu.uCascade.value = c; cu.uChopK.value = OCEAN_CHOP[c];
      // Build both mip chains only after every array layer is current.
      this.cascades.textures.forEach(texture => { texture.generateMipmaps = c === 2; });
      this.pCompose.render(renderer, this.cascades, c);
    }
    renderer.setRenderTarget(prev, prevCube, prevMip);
  }
}
