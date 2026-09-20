import * as THREE from 'three';
import { glslDefines, CONFIG } from '../config.js';
import { NOISE, BED, SKY, OPTICS, DEPTH } from '../glsl/common.js';
import { BREAKER } from '../glsl/breaker.js';
import { CHOP, SWE_SAMPLE } from '../glsl/water.js';
import { SWASH_FAR } from '../glsl/swashfar.js';
import { FullscreenPass, makeShader } from '../core/gpu.js';
import { K, NKIND, CLASS_KINDS, EMIT_GLSL, emitterRates, BORE_BG, exploreBinEdges, exploreEdgeFade, lodAt, LOD, EXPLORE_HALF } from './emitters.js';
import { makeFoamNoise } from './foamNoise.js';

// Whitewater: splash, spray and crown break-up as GPU particles (math doc §9, §14, §44, §53;
// breaker_morphology.md §5.2-§5.4).
//
// Simulation: fixed-step GPGPU state (pos+age | vel+kind | radius, density, t_frag, seed+foam
// share), three ring buffers (blobs, droplets, ligaments), deterministic CPU emission (see
// emitters.js). The splash dome is one broad arch: the first launches are the fastest, later ones
// fill the body, and the launch clusters only jitter the crown (+-6 %); its crown erodes into lace
// after the apex while the droplets / ligaments that were travelling inside it are revealed. Blobs
// that fall back ride the bore (SWE velocity) and fade into the surface foam.
//
// Two phases: every blob parcel is part white foam (opaque, lumpy, bright) and part aerated water
// (smooth, translucent, a milky in-scatter with a sheen on its edges). The water share grows as a
// splash parcel falls back, low on a roller / the dome base / a spilling roller (glassy, bubbly
// water near the surface), on rafts and in the patchy pockets of the churn (patchiness scales only
// the foam share); jets, sheets and the curtain veil are mostly water.
//
// Rendering (per frame; clip: half-res buffers, explore: 1/2 of the output):
//  1. front pre-pass: MAX of 1 / depth of the dense white foam (where the eye meets the foam);
//  2. blobs -> order independent additive buffers: (tau foam, tau water, tau * view z, tau lump) |
//     front-weighted (detail, weight) + tearing state | front-weighted lump normal. Each parcel's
//     chord is clipped against the scene depth; its weight falls off a few cm behind the front of
//     the dense foam, so the front lumps keep their own lumpy caps and bubbly texture instead of
//     averaging into a grey mottle;
//  3. resolve at full res: foam coverage 1 - exp(-(k tau)^2) (thin high spray keeps a translucent
//     veil) over the aerated water; lighting from the lump normals + the log optical depth at
//     1-4 cm and (mip chain) 9-22 cm billows, downward faces see less sky, cavities / crevices
//     (front-depth steps), self-shadow towards the sun, sparse wet glints; low foam on the bore
//     gets flow-advected world-space billows / folds (two-phase flow map); the torn crown is an
//     analytic Voronoi lace of white-rimmed webs around glassy, speckled windows;
//  4. crisp pass: droplets as sub-pixel-correct dots (coverage AA, no motion blur in the clip,
//     sparkle) and ligaments as rim threads / radial fingers with a head bead, glassy up close,
//     occluded by the scene depth and by the foam in front of them.
//
// Explore mode (CONFIG.explore, ?explore; the clip mode is unchanged):
//  - emission over focus.x +- 40 m (bins recentred on the focus snapped to 1 m, 5 cm wide at the
//    player, 0.8 m far away; edge fade over the last 8 m), with a distance LOD: fewer, larger
//    parcels far from the player (count x f, radius x f^-1/2, density x f^1/2; emitters.js lodAt)
//    and a 512^2 particle pool;
//  - beyond the scrolling shallow-water window the particles land on / ride / find bore fronts in
//    a far-field water state baked each step over the emission window (FARBAKE: glsl/swashfar.js
//    swash + the breaker surface);
//  - view independence: rafts are horizontal discs (screen ellipse from the view), the crown lace
//    web uses the screen's horizontal axis;
//  - close range: soft near-plane clip and a screen-size cap on parcels, bubbles at a fixed world
//    scale (Voronoi cells of 3.5 / 7 mm: bright Plateau borders, clear interiors, cap bumps) in the
//    front parcels, a third shading ring; drops are short exposure streaks, within ~1 m tiny lenses
//    (refracted sky / ground, dark rim, sun glint).
const EXPLORE = !!CONFIG.explore;
const RES = EXPLORE ? 512 : 384;
const N = RES * RES;
const CLASS_SIZE = EXPLORE ? [163840, 81920, 16384] : [90112, 49152, 8192];   // blobs, droplets, ligaments
const CLASS_BASE = [0, CLASS_SIZE[0], CLASS_SIZE[0] + CLASS_SIZE[1]];
const X0 = -2.9, X1 = 2.9;
const EDGES = EXPLORE ? exploreBinEdges() : Float64Array.from({ length: 65 }, (_, i) => X0 + i * (X1 - X0) / 64);
const BINS = EDGES.length - 1;
const NBIT = Math.ceil(Math.log2(BINS)) + 1;
const FOCUS_SNAP = 1.0;           // m: the bins move with the player in whole metres
// far-field water state for the particles beyond the shallow-water window (explore mode)
const FAR_NX = 320, FAR_NZ = 200;
const FAR_W = 2 * EXPLORE_HALF + 1.2, FAR_Z0 = -7.0, FAR_Z1 = 4.0;

const PRELUDE = (zpass) => glslDefines() + NOISE + BED + SKY + OPTICS + DEPTH + BREAKER + CHOP + SWE_SAMPLE + (EXPLORE ? SWASH_FAR : '') + EMIT_GLSL + /* glsl */ `
#define WW_EXPLORE ${EXPLORE ? 1 : 0}
#define WW_ZPASS ${zpass ? 1 : 0}
#define NBIN ${BINS}
#define NBIT ${NBIT}
#define RES ${RES}
#define NB ${CLASS_SIZE[0]}
#define ND ${CLASS_SIZE[1]}
#define NL ${CLASS_SIZE[2]}
bool isBlob(int k) { return k <= K_SPILL; }
bool isLig(int k) { return k >= K_LIG; }
bool isRoller(int k) { return k == K_ROLL || k == K_BOIL || k == K_BORE || k == K_SPILL; }
// erosion time constant of fragmenting blobs (s)
float fragT(int k) { return isRoller(k) ? 0.35 : k == K_DOME ? 0.22 : (k == K_VEIL ? 0.25 : (k == K_FSHEET ? 0.10 : (k == K_JETB ? 0.12 : 0.3))); }
float lifeMax(int k) {
  if (k == K_ROLL || k == K_DOME) return 1.7;
  if (k == K_VEIL) return 0.6;
  if (k == K_FSHEET) return 0.55;
  if (k == K_JETB) return 1.2;
  if (k == K_BOIL) return 1.3;
  if (k == K_BORE) return 0.95;
  if (k == K_SPILL) return 1.2;
  return 2.0;
}
// how sheet-like (streaky, translucent-edged) a blob kind is, 0 = lumpy foam
float sheetness(int k) { return k == K_DOME ? 0.68 : (k == K_VEIL || k == K_FSHEET || k == K_JETB ? 1.0 : 0.35); }
`;

// ---------------------------------------------------------------------------- simulation
const UPDATE = /* glsl */ `
uniform sampler2D uPos;
uniform sampler2D uVel;
uniform sampler2D uAttr;
uniform sampler2D uCdf;      // (NBIN + 1) x (NKIND + 1) (R32F): per-emitter CDF over the bins, last row = bin edges (world x)
uniform vec4 uLod;           // (focus x, focus z, D0, power): distance LOD of the emission (emitters.js lodAt)
uniform float uSpillHov;     // how high (x parcel radius) the spilling roller's parcels tumble above the surface
uniform vec4 uSpillWin;      // spilling roller: hover rise end, fall start / end, life on the surface (s)
#define SPILL_HOV uSpillHov
uniform float uDtS;
uniform int uHead[3];
uniform int uCount[NKIND];
uniform float uStepSeed;
layout(location = 1) out vec4 outVel;
layout(location = 2) out vec4 outAttr;

uniform sampler2D uWaterH;   // total surface elevation baked by SwashSim each step
#if WW_EXPLORE
// beyond the (scrolling) shallow-water window: the far-field swash (glsl/swashfar.js) and the
// breaker surface, baked each step over the emission window (FARBAKE)
uniform sampler2D uFarA;     // (h, u, v, w) far-field swash
uniform sampler2D uFarB;     // (total surface elevation incl. breakers, whitewater foam, 0, 0)
uniform vec4 uFarDom;        // xMin, zMin, width, depth
vec2 farUV(vec2 xz) { return (xz - uFarDom.xy) / uFarDom.zw; }
float farWt(vec2 xz) { return max(swashFarWeight(xz), 1.0 - sweInside(xz)); }
#endif
float waterLevel(vec2 xz) {
  vec2 uv = sweUV(xz);
  float s = (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) ? 0.0 : textureLod(uWaterH, uv, 0.0).r;
#if WW_EXPLORE
  float wf = farWt(xz);
  if (wf > 1e-3) s = mix(s, textureLod(uFarB, farUV(xz), 0.0).x, wf);
#endif
  return s;
}
// shallow-water state (h, u, v, w) (and the whitewater foam) where the particles are
vec4 swState(vec2 xz) {
  vec4 s = textureLod(uSweView, sweUV(xz), 0.0);
#if WW_EXPLORE
  float wf = farWt(xz);
  if (wf > 1e-3) {
    vec4 f = textureLod(uFarA, farUV(xz), 0.0);
    // (depth from the total level: robust where the far model carries no water)
    f.x = max(textureLod(uFarB, farUV(xz), 0.0).x - bedProfile(xz.y), 0.0);
    s = mix(s, f, wf);
  }
#endif
  return s;
}
float swFoam(vec2 xz) {
  float f = textureLod(uSweFoam, sweUV(xz), 0.0).r;
#if WW_EXPLORE
  float wf = farWt(xz);
  if (wf > 1e-3) f = mix(f, textureLod(uFarB, farUV(xz), 0.0).y, wf);
#endif
  return f;
}
#if WW_EXPLORE
vec2 sweVel(vec2 xz) { vec2 v = textureLod(uSweView, sweUV(xz), 0.0).yz * sweInside(xz); float wf = farWt(xz); return wf > 1e-3 ? mix(v, textureLod(uFarA, farUV(xz), 0.0).yz, wf) : v; }
#else
vec2 sweVel(vec2 xz) { return textureLod(uSweView, sweUV(xz), 0.0).yz * sweInside(xz); }
#endif

// x of a newborn of emitter k: inverse CDF over the bins (binary search, then uniform in the bin)
float cdfAt(int i, int k) { return texelFetch(uCdf, ivec2(i, k), 0).r; }
float sampleX(int k, float u) {
  int lo = 0, hi = NBIN - 1;
  for (int it = 0; it < NBIT + min(uEvtCount, 0); it++) {   // dynamic bound (FXC)
    if (lo >= hi) break;
    int mid = (lo + hi) / 2;
    if (cdfAt(mid, k) >= u) hi = mid; else lo = mid + 1;
  }
  float c = cdfAt(lo, k);
  float prev = lo > 0 ? cdfAt(lo - 1, k) : 0.0;
  float f = clamp((u - prev) / max(c - prev, 1e-6), 0.0, 1.0);
  float e0 = texelFetch(uCdf, ivec2(lo, NKIND), 0).r, e1 = texelFetch(uCdf, ivec2(lo + 1, NKIND), 0).r;
  return e0 + f * (e1 - e0);
}
float lodAt(float x) { return clamp(pow(uLod.z / max(length(vec2(x - uLod.x, ${LOD.ZB.toFixed(2)} - uLod.y)), 1e-3), uLod.w), ${LOD.MIN.toFixed(3)}, 1.0); }
float gauss(vec2 r) { return sqrt(-2.0 * log(max(r.x, 1e-6))) * cos(6.2831853 * r.y); }
// smooth 3D value noise (x, z, t) for the patchiness of the roller foam
float vnoise(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float n000 = hash11(dot(i, vec3(1.0, 57.0, 113.0))), n100 = hash11(dot(i + vec3(1, 0, 0), vec3(1.0, 57.0, 113.0)));
  float n010 = hash11(dot(i + vec3(0, 1, 0), vec3(1.0, 57.0, 113.0))), n110 = hash11(dot(i + vec3(1, 1, 0), vec3(1.0, 57.0, 113.0)));
  float n001 = hash11(dot(i + vec3(0, 0, 1), vec3(1.0, 57.0, 113.0))), n101 = hash11(dot(i + vec3(1, 0, 1), vec3(1.0, 57.0, 113.0)));
  float n011 = hash11(dot(i + vec3(0, 1, 1), vec3(1.0, 57.0, 113.0))), n111 = hash11(dot(i + vec3(1, 1, 1), vec3(1.0, 57.0, 113.0)));
  return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y), mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}
// aerated water is patchy: dense white billows with translucent aerated water between them
// (0.2-0.5 m patches that drift and re-form). The factor is fixed at birth so a patch moves
// with the parcels that make it up; it scales only the white-foam phase of a parcel (a pocket
// is bubbly water, not a hole): 0 = aerated water only
float patchiness(vec2 xz, float seed) {
  float n = 0.65 * vnoise(vec3(xz.x / 0.4, xz.y / 0.4, uTime / 0.4 + seed * 17.0)) + 0.35 * vnoise(vec3(xz.x / 0.17, xz.y / 0.17, uTime / 0.25 + 5.3));
  return smoothstep(0.36, 0.54, n);
}
// coherent launch cluster of the splash-up: particles born in the same ~0.14 m x 0.035 s
// cell share one launch velocity, so the dome grows as lumps (cauliflower)
vec3 domeCluster(float x, float tl, float seed) {
  float slot = floor(tl / 0.035);
  float j = hash11(slot * 1.37 + seed * 53.1);
  float cw = mix(0.06, 0.15, hash11(slot * 7.77 + seed * 3.3));    // irregular cluster widths
  float cx = floor(x / cw + j);
  return hash33(vec3(cx * 1.13 + 0.5, slot * 0.71 + 3.1, seed * 97.0 + 1.7));
}
// Bore fronts in the shallow-water state along the cross-shore line at x, marching shoreward
// from z0 in 10 cm steps: a drop of the water SURFACE (not the depth: the still water over the
// steep pebble step is no front) of > 2.5 cm over 20 cm, with >= 4 cm of broken (foamy) water
// behind it. Picks one front cell by reservoir sampling weighted by its strength.
// Returns (z of the picked front, summed strength, u behind it, h behind it); zLand = the
// landward-most clear front (-99 if none).
vec4 boreFront(float x, float z0, float rnd, out float zLand) {
  vec4 a = swState(vec2(x, z0));
  vec4 b = swState(vec2(x, z0 + 0.1));
  float W = 0.0;
  vec4 pick = vec4(-99.0, 0.0, 0.0, 0.0);
  zLand = -99.0;
  for (int it = 2; it < 48 + min(uEvtCount, 0); it++) {    // dynamic bound (FXC)
    float z = z0 + float(it) * 0.1;
    vec4 c = swState(vec2(x, z));
    float drop = a.w - c.w;
    // (the water behind must not be draining: a backwash running into still water also makes a
    // surface step, but it has no roller)
    float s = smoothstep(0.035, 0.13, drop) * smoothstep(0.05, 0.18, a.x) * clamp((a.z + 0.3) / 0.8, 0.0, 1.0);
    if (s > 0.02) {
      // only broken water carries a roller (a steep unbroken swell face does not)
      s *= smoothstep(0.2, 0.9, swFoam(vec2(x, z - 0.2)));
      W += s;
      if (hash11(rnd * 91.7 + float(it) * 13.13) * W < s) pick = vec4(z - 0.1, 0.0, a.z, a.x);
      if (s > 0.2) zLand = z - 0.1;
    }
    a = b; b = c;
  }
  pick.y = W;
  return pick;
}
// the white front of a breaker's bore runs at a nearly constant speed until it stalls ~2.5-2.8 m
// shoreward of the plunge line (wave B: 3.7-4.2 m/s for ~0.7 s; the spilling wave C: ~1.7 m/s
// for ~1.4 s); the shallow-water bore lags it, so the roller follows whichever is ahead
float surgeZ(Brk b, float tau, out float spd) {
  float P = plungeFactor(b.style);
  float V = 4.2 * sqrt(b.H / 0.45) * mix(0.5, 1.0, P);
  // (how far the white water runs before it stalls varies from wave to wave: 2.2-2.9 m)
  float D = 2.8 * mix(0.9, 1.0, P) * mix(0.78, 1.02, hash11(b.seed * 40.7));
  float q = max(V * tau / D, 0.0);
  float q4 = q * q * q * q;
  float k = pow(1.0 + q4 * q4, -0.125);
  spd = V * pow(1.0 + q4 * q4, -1.125);
  return b.zI + D * q * k;
}
float surgeW(Brk b, float tau) {
  float P = plungeFactor(b.style);
  // (after it stalls the roller keeps tumbling at the stall line for ~0.5 s: wave B holds at the
  //  beach step until ~4.6 s)
  float t1 = 1.2 + 0.4 * (1.0 - P);
  // (small breakers have no surge worth the name)
  return smoothstep(0.1, 0.25, tau) * (1.0 - smoothstep(t1, t1 + 0.17, tau)) * smoothstep(0.4, 0.6, b.H / 0.45);
}

#define DEAD { gl_FragColor = vec4(0.0, -50.0, 0.0, 99.0); outVel = vec4(0.0); outAttr = vec4(0.0); return; }

void main() {
  ivec2 ij = ivec2(gl_FragCoord.xy);
  int id = ij.y * RES + ij.x;
  vec4 P = texelFetch(uPos, ij, 0);
  vec4 Vl = texelFetch(uVel, ij, 0);
  vec4 At = texelFetch(uAttr, ij, 0);
  int cls = id < NB ? 0 : (id < NB + ND ? 1 : 2);
  int base = cls == 0 ? 0 : (cls == 1 ? NB : NB + ND);
  int size = cls == 0 ? NB : (cls == 1 ? ND : NL);
  int k0 = cls == 0 ? 0 : (cls == 1 ? K_SPRAY : K_LIG);
  int k1 = cls == 0 ? K_SPRAY : (cls == 1 ? K_LIG : NKIND);
  int slotK = (id - base - uHead[cls] + 2 * size) % size;
  int acc = 0; int kind = -1;
  for (int j = 0; j < NKIND + min(uEvtCount, 0); j++) {   // dynamic bound (FXC)
    if (j < k0 || j >= k1) continue;
    if (kind < 0 && slotK < acc + uCount[j]) kind = j;
    acc += uCount[j];
  }

  if (kind >= 0) {
    // ------------------------------------------------------------ birth
    vec3 h1 = hash33(vec3(float(id) * 0.713, uStepSeed, 1.7));
    vec3 h2 = hash33(vec3(float(id) * 1.931, uStepSeed * 1.37, 5.3));
    vec3 h3 = hash33(vec3(float(id) * 0.377, uStepSeed * 0.71, 9.1));
    float x = sampleX(kind, h1.x);
    int best = -1; float bestW = 0.0;
    for (int i = 0; i < uEvtCount; i++) {  // dynamic bound: keeps D3D/FXC from unrolling
      Brk bb = brkAt(i, x);
      if (bb.str <= 0.0) continue;
      float w = emitWeight(kind, i, bb, x);
      if (w > bestW) { bestW = w; best = i; }
    }
    // bore rollers also live on fronts whose event has already retired
    bool hasEvt = best >= 0;
    if (!hasEvt && kind != K_BORE) DEAD
    Brk b = brkAt(max(best, 0), x);
    float tau = hasEvt ? uTime - b.ti : 9.0;
    float hs = hasEvt ? b.H / 0.45 : 1.0;
    float Pl = hasEvt ? plungeFactor(b.style) : 1.0;
    float bseed = hasEvt ? b.seed : 0.5;
    float en0 = clamp(hs * sqrt(max(b.str, 0.0)), 0.6, 1.4);
    float en = en0 * mix(0.55, 1.0, Pl);           // plunging impacts throw higher
    float enR = en0 * mix(0.8, 1.0, Pl);           // roller churn
    float A = domeField(x, bseed);
    float Wtop = min(mix(2.8, 3.95, A) * en, 4.15);  // splash-up launch speed of the dome top (m/s; wave B: <= +0.8 m)
    float g1 = gauss(h2.xy), g2 = gauss(h2.yz), g3 = gauss(h3.xy), g4 = gauss(h3.yz), g5 = gauss(vec2(h1.y, h3.z));
    float zI = hasEvt ? b.zI : -1.9;
    // the bore: its front (shallow-water state + the early splash-driven surge) for the rollers
    float zFr = zI + 0.5, spd = 0.0, zLand = -99.0;
    vec4 bf = vec4(-99.0, 0.0, 0.0, 0.0);
    float zk = zI, wk = 0.0;
    if (kind == K_BORE || kind == K_BOIL) {
      bf = boreFront(x, hasEvt ? zI - 0.5 : -1.8, h3.z, zLand);
      if (hasEvt) { zk = surgeZ(b, tau, spd); wk = surgeW(b, tau); }
      zFr = zLand > -50.0 ? clamp(zLand, zI + 0.2, zk + 0.8) : max(zk, zI + 0.3);
      zFr = mix(zFr, max(zFr, zk), wk);
    }
    // the crest apex (z, y) and its speed, for the crest emitters (one evaluation site)
    vec2 Ap = vec2(0.0); float cs = 0.0;
    if (kind == K_SPILL || kind == K_FSHEET || kind == K_FDROP) {
      float tn0 = brkTn(b, uTime), tn1 = brkTn(b, uTime + 0.02);
      Ap = brkApex(b, brkStage(tn0), tn0);
      cs = (brkApex(b, brkStage(tn1), tn1).x - Ap.x) / 0.02;
    }
    vec3 pos, vel;
    float rad, dens, tF = 99.0, fph = 1.0;   // fph: white-foam share of the parcel (the rest is aerated water)
    if (kind == K_SPILL) {
      // spilling whitecap: aerated water tumbling down the front face from the crest to ~0.3 H
      // below it, riding the crest (a roller, not a splash)
      float slot = floor(uTime / 0.07);
      // lumpy top: 0.2-0.4 m lumps with gaps (windows) in between
      float cwid = mix(0.2, 0.4, hash11(slot * 3.7 + bseed * 7.1));
      float xo = hash11(slot * 1.3 + bseed) * cwid;
      float cell = floor((x + xo) / cwid);
      float uc = ((x + xo) / cwid - cell) * 2.0 - 1.0;
      vec3 hc = hash33(vec3(cell, slot, bseed * 53.0 + 9.0));
      float s = h1.y * h1.y;                                     // 0 = crest .. 1 = 0.3 H down the face
      float dn = 0.3 * max(Ap.y, 0.1) * s;
      pos = vec3(x + g1 * 0.03, Ap.y - dn + 0.01, Ap.x + 0.02 + 1.3 * dn + 0.03 * g2);
      // (the rendered crest can stand above the baked height field, where the lip sheet is: the
      // parcel remembers how far above it rides and settles onto the surface as it slides down)
      tF = max(pos.y - waterLevel(pos.xz), 0.0);
      float prof = sqrt(max(1.0 - uc * uc, 0.0));
      // it rides the face (born on the surface) and slowly slides down it: slightly slower than
      // the crest, so the band's lower edge frays
      vel = vec3(g3 * 0.1, 0.0, cs * mix(0.82, 1.0, h2.z) + g5 * 0.05);
      rad = mix(0.035, 0.07, h1.z) * sqrt(hs);
      // 15-25 % of the band shows the water through (windows between the lumps); the foam
      // draped down the face is thinner than the whitecap on the crest
      dens = 7.0 * mix(0.45, 1.0, smoothstep(0.2, 0.45, hc.z)) * mix(0.6, 1.0, prof) * mix(1.0, 0.45, s);
      fph = mix(0.1, 1.0, smoothstep(0.2, 0.45, hc.z));
    } else if (kind == K_ROLL) {
      // impact roll: low, lumpy billow thrown shoreward along the whole touched-down length;
      // it grows in behind the peel point (a fresh impact is still low)
      float slot = floor(tau / 0.04);
      float cw = mix(0.2, 0.4, hash11(slot * 5.1 + bseed * 3.3));
      vec3 hc = hash33(vec3(floor(x / cw + hash11(slot * 3.1 + bseed)), slot, bseed * 71.0 + 4.0));
      float late = smoothstep(0.08, 0.3, tau);
      float grow = mix(0.4, 1.0, smoothstep(-0.03, 0.12, tau));
      float w0 = mix(0.4, 2.0, hc.x * hc.x) * enR * mix(1.0, 0.55, late) * grow;
      pos = vec3(x + g1 * 0.04, 0.0, zI + 0.02 + g2 * 0.08 + late * 0.35 * tau);
      vec2 fl = sweVel(pos.xz);
      pos.y = max(waterLevel(pos.xz), 0.0) + 0.02 + 0.06 * h1.y;
      vel = vec3((hc.z - 0.5) * 0.35 + g3 * 0.1, w0 + g4 * 0.15, mix(b.c * (0.25 + 0.35 * hc.y), fl.y * (0.6 + 0.4 * hc.y), late) + g5 * 0.15);
      // a third of the parcels are thrown down and forward: the ragged lower edge of the roll
      if (h2.z < 0.33) vel = vec3(vel.x, -mix(0.2, 0.9, h3.x), b.c * mix(0.4, 0.8, h1.y));
      rad = mix(0.035, 0.075, h1.z) * sqrt(hs);
      // lumpy roller: dense billows separated by thin, dark gaps (not a uniform tube)
      dens = 16.0 * mix(0.4, 1.0, smoothstep(0.3, 0.5, hc.z)) * mix(0.5, 1.0, grow);
      fph = mix(0.15, 1.0, smoothstep(0.3, 0.5, hc.z));
      tF = 0.18 + 0.15 * h3.x;                                // the roller tears open (dark holes) as it ages
      // the roller tears open into translucent patches (turbid water shows through) as it ages
      fph *= mix(1.0, patchiness(pos.xz, bseed), smoothstep(0.1, 0.3, tau));
    } else if (kind == K_DOME || kind == K_DROP || kind == K_LIG) {
      // cauliflower splash-up (h = V^2 / 2g): clusters with a common launch velocity
      float tl = uTime - domeLaunch(best, x, b.ti);
      vec3 hc = domeCluster(x, tl, bseed);
      float dAdx = (domeField(x + 0.1, bseed) - domeField(x - 0.1, bseed)) / 0.2;
      float q;
      // The splash sheet: the first launches are the fastest, later ones fill the body below; at a
      // given launch time the speeds spread continuously (the body is filled from the base to the
      // crown), and the crown's envelope is shared along the dome with a small (+-6 %) lumpy
      // jitter per launch cluster: the dome rises as one broad arch, not as towers of clusters
      if (kind == K_DOME) q = mix(1.0, 0.5, smoothstep(0.0, 0.2, tl)) * mix(0.5, 1.0, sqrt(h3.z)) * (1.0 + 0.06 * (2.0 * hc.x - 1.0));
      else q = kind == K_LIG ? mix(0.6, 0.95, sqrt(h3.z)) : mix(0.55, 1.03, sqrt(h3.z));
      // late launches only feed the boiling base of the dome
      q = min(q, mix(1.02, 0.42, smoothstep(0.04, 0.22, tl)));
      // a few clusters are fast, thin sheets of glassy water: the translucent scalloped rim and
      // fingers above the white lobes (breaker_morphology 5.4: +0.37 s)
      bool jet = kind == K_DOME && hash11(hc.x * 97.1 + hc.y * 13.3) < 0.25;
      if (jet) q = min(q * 1.06, 1.08);
      // a fraction of the dome is fine spray travelling with it: a translucent veil around the lumps
      bool haze = kind == K_DOME && h2.z < 0.045;

      float w0 = Wtop * q * (1.0 - 0.2 * clamp(tl / 0.15, 0.0, 1.0));
      // crown sheets flare away from the dome axis; launch clusters barely diverge (a random lateral
      // velocity per cluster tears the dome into separate columns)
      float vx = -clamp(dAdx, -1.2, 1.2) * 0.35 * q + (hc.z - 0.5) * (kind == K_DOME ? 0.12 : 0.5);
      float vz = 0.45 + 1.0 * hc.y;                             // shoreward drift onto the bore
      // each cluster is a jet: speed spread along its direction (it stretches into a finger),
      // little spread across it
      float sv = kind == K_DOME ? (haze ? 0.3 : 0.08) : 0.28;
      vec3 Vc = vec3(vx, w0, vz);
      pos = vec3(x + g1 * 0.02, 0.0, zI + 0.08 + g2 * 0.06);
      pos.y = max(waterLevel(pos.xz), 0.0) + 0.04 + 0.06 * h1.y;     // launched from the churned surface
      vel = Vc * (1.0 + clamp(g4, -2.0, 1.6) * (kind == K_DOME ? 0.03 : 0.07)) + vec3(g3, g5 * 0.5, g1 * 0.7) * sv;
      if (kind == K_DOME) {
        rad = mix(0.022, 0.05, h1.z * h1.z) * sqrt(hs) * (1.0 + 0.5 * smoothstep(0.6, 1.0, q)) * (jet ? 0.7 : 1.0);
        // some launch clusters are thin, glassy sheets of water (translucent), most are dense foam
        dens = 7.2 * mix(0.25, 1.0, smoothstep(0.12, 0.3, hc.z * hc.y + 0.1 * hc.x));
        // the slow body parcels fall back first and stay low: a thin, translucent lower dome
        // (the dark water behind shows through under the crown)
        dens *= mix(0.35, 1.0, smoothstep(0.35, 0.8, q));
        if (haze) { rad *= 1.4; dens = 1.2; fph = 0.35; }
        if (jet) fph = 0.2;                                     // (glassy sheet: vertex shader BLOB_VERT)
        // the crown tears into lace near its apex; the lower dome tears later and slower
        tF = w0 > 3.1 ? 0.68 * w0 / G_ACC + 0.12 * h3.x : 0.5 + 0.3 * h3.x;
      } else if (kind == K_DROP) {
        rad = mix(0.0015, 0.0055, pow(h1.z, 2.5));
        dens = 1.0;
        tF = 0.8 * w0 / G_ACC + 0.05 + 0.2 * h3.x;              // revealed when the crown opens
      } else {
        // rim threads of the torn crown: lateral, 0.6-1.4 cm thick, pinching into beads
        rad = mix(0.003, 0.007, h1.z * h1.z);
        dens = mix(0.04, 0.12, h3.x * h3.x);                    // ligament length (m)
        tF = w0 > 2.9 ? mix(0.55, 0.9, h3.x) * w0 / G_ACC : 99.0;
      }
    } else if (kind == K_VEIL) {
      // falling curtain turning into a semi-transparent veil of streaks and spray
      float sig = pow(h1.y, 1.5) * 0.95;
      vec3 c = brkJet(b, sig, min(uTime, b.ti));
      pos = vec3(x + g1 * 0.04, max(c.y, 0.02) + g2 * 0.03, c.x + g3 * 0.05);
      // it keeps the curtain's fall at first, then the impact blows it back up as spray
      vel = vec3(g4 * 0.2, mix(-1.2, 1.6, h1.z), b.c * 0.35 + g5 * 0.25);
      rad = mix(0.04, 0.08, h3.x);
      dens = 0.5;
      tF = 0.1;
    } else if (kind == K_SPRAY) {
      pos = vec3(x + g1 * 0.03, 0.0, zI + 0.05 + g2 * 0.1);
      float w = Wtop * mix(0.2, 0.75, pow(h1.z, 0.6));
      if (tau > 0.2) {
        // spray thrown off the churning roller
        pos.z = zI - 0.1 + 0.8 * h1.y * h1.y;
        w = mix(0.8, 2.2, pow(h1.z, 1.5));
      }
      pos.y = max(waterLevel(pos.xz), 0.0) + 0.03 + 0.15 * h1.y;
      vel = vec3(g3 * 0.55, w, 0.3 + 1.8 * h3.x);
      rad = mix(0.0015, 0.006, h3.y * h3.y * h3.y);
      dens = 1.0;
      tF = 0.0;
    } else if (kind == K_FSHEET || kind == K_FDROP) {
      // crest feathering: thin sheets / jets torn off the sharpening crest, fanning +-25 deg and
      // leaning back in the wind
      pos = vec3(x + g1 * 0.02, Ap.y + 0.005, Ap.x + 0.02 + g2 * 0.02);
      // coherent jets: material torn off the same ~0.2 m of crest in the same 60 ms shares its
      // launch speed, log-normally distributed (median 1.75 m/s: 10-35 cm, the odd one ~0.45 m)
      vec3 hc = hash33(vec3(floor(x / 0.2 + hash11(floor(uTime / 0.06))), floor(uTime / 0.06), bseed * 37.0 + 2.0));
      float up = clamp(1.75 * exp(0.3 * gauss(hc.xy)), 0.9, 3.0);
      float fan = (hc.z - 0.5) * 0.93;                         // +-25 deg
      vel = vec3(fan * up + g3 * 0.12, up * (1.0 + 0.1 * clamp(g4, -2.0, 1.5)), cs * 0.65 + g5 * 0.12);
      if (kind == K_FSHEET) { rad = mix(0.02, 0.045, h3.x); dens = 2.8 * mix(0.45, 1.0, hc.y); tF = 0.08; }
      else { rad = mix(0.0015, 0.0055, pow(h3.x, 2.5)); dens = 1.0; tF = 0.0; }
    } else if (kind == K_LIPDROP) {
      // droplets pinching off the flying lip edge
      float sig = h1.y * h1.y * 0.3;
      vec3 c = brkJet(b, sig, uTime);
      vec3 c2 = brkJet(b, sig, uTime - 0.01);
      vec2 v = (c.xy - c2.xy) / 0.01;
      pos = vec3(x + g1 * 0.02, c.y, c.x);
      vel = vec3(g3 * 0.25, v.y + g2 * 0.35, v.x + h3.z * 0.5);
      rad = mix(0.002, 0.008, h3.x * h3.x);
      dens = 1.0; tF = 0.0;
    } else if (kind == K_JETB || kind == K_JETD || kind == K_JETL) {
      // second splash cycle: a narrow jet splitting into lobes and ligament fingers
      float xc = jetCentre(x, bseed);
      float m = jetMask(x, bseed);
      float w0 = 4.15 * en * mix(0.5, 1.0, m) * mix(0.55, 1.0, h1.z);
      pos = vec3(x + g1 * 0.02, 0.0, zI + 0.3 + g2 * 0.05);
      pos.y = max(waterLevel(pos.xz), 0.0) + 0.03 + 0.05 * h1.y;
      vel = vec3((x - xc) * 1.6 + g3 * 0.2, w0 + g4 * 0.2, 0.25 + g5 * 0.2);
      // the sheet body stays whole almost to its apex, then splits into beaded fingers
      if (kind == K_JETB) { rad = mix(0.02, 0.04, h3.x); dens = 4.0; tF = 0.8 * w0 / G_ACC; }
      else if (kind == K_JETD) { rad = mix(0.001, 0.0035, pow(h3.x, 2.0)); dens = 1.0; tF = 0.7 * w0 / G_ACC + 0.1 * h3.y; }
      else { rad = mix(0.0018, 0.0035, h3.x); dens = mix(0.12, 0.35, h3.y); tF = 0.6 * w0 / G_ACC + 0.1 * h3.z; }
    } else if (kind == K_BORE) {
      // bore-front roller: aerated water tumbling on the front of the bore (also a stalled one);
      // right after a plunge it rides the splash-driven surge where that is ahead of the bore
      float zf = bf.x, uB = bf.z, hB = bf.w;
      float accept = min(bf.y / 2.5, 1.0);
      // a roller circulates on its front: it moves with the front, which is slower than the water
      // behind a (slowing / stalled) bore
      float vFront = 0.3 * uB;
      if (hasEvt && wk > 0.0 && zk > (zLand > -50.0 ? zLand : zI) + 0.25 && h1.y < wk
          && (tau < 0.85 || swState(vec2(x, zk - 0.1)).x > 0.012)) {
        zf = zk; vFront = 0.7 * spd; hB = 0.16 * hs; accept = 1.0;
      }
      if (zf < -50.0 || h2.z > accept) DEAD
      float slot = floor(uTime / 0.06);
      vec3 hc = hash33(vec3(floor(x / 0.3 + hash11(slot * 2.3 + 7.1)), slot, 5.0 + bseed * 19.0));
      // on the face of the bore, just behind its toe
      pos = vec3(x + g1 * 0.03, 0.0, zf - 0.08 - 0.35 * h2.y);
      pos.y = waterLevel(pos.xz) + 0.005;
      vec2 fl = sweVel(pos.xz);
      float hh = clamp(hB / 0.15, 0.6, 1.4);
      float vz = vFront * mix(0.5, 1.2, h1.z);
      vel = vec3(fl.x * 0.5 + g3 * 0.15, mix(0.35, 1.1, hc.x * hc.x) * hh + g4 * 0.1, vz + g5 * 0.1);
      rad = mix(0.03, 0.055, h3.x) * sqrt(clamp(hasEvt ? hs : hh, 0.6, 1.3));
      dens = 24.0 * mix(0.6, 1.0, smoothstep(0.3, 0.5, hc.z));
      fph = patchiness(pos.xz, bseed + 0.3);
      tF = 0.25 + 0.2 * h3.y;
    } else {
      // turbulent roller body: after the splash falls back, the water between a back edge that
      // moves shoreward (the old plunge line calms first) and the bore front keeps churning;
      // lumps boil up 5-10 cm (more where the bore is deep) and fall back (§5.5)
      float zB = zI - 0.2 + 1.3 * max(tau - 0.3, 0.0);
      float zF = max(mix(zI + 0.4, zFr, max(smoothstep(0.45, 0.7, tau), 1.0 - Pl)), zB + 0.45);
      float zz = mix(zB, zF, pow(h1.y, 0.7));
      vec4 sv = swState(vec2(x, zz));
      if (h2.z > smoothstep(0.02, 0.16, sv.x)) DEAD
      pos = vec3(x + g1 * 0.03, 0.0, zz + g2 * 0.05);
      pos.y = waterLevel(pos.xz) + 0.01;
      // coherent mounds: the churn heaves up in 0.3-0.7 m wide lumps; the launch speed has a
      // rounded profile across each lump so it rises as a smooth white mound, not a column
      float slot = floor(tau / 0.07);
      float cwid = mix(0.3, 0.7, hash11(slot * 3.3 + bseed * 9.1));
      float xo = hash11(slot * 5.7 + bseed) * cwid;
      float cell = floor((x + xo) / cwid);
      float uc = ((x + xo) / cwid - cell) * 2.0 - 1.0;              // -1..1 across the lump
      vec3 hc = hash33(vec3(cell, slot, floor(zz / 0.35) + bseed * 31.0));
      float prof = sqrt(max(1.0 - uc * uc, 0.0));
      float w0 = mix(0.45, 1.3, hc.x) * enR * mix(0.45, 1.0, prof) * mix(2.3, 1.0, Pl) * mix(0.8, 1.2, smoothstep(0.05, 0.3, sv.x));
      if (tau > 0.6) w0 = min(w0, 1.2);
      // the splash that falls back re-impacts: where the secondary jet goes up, the churn at the
      // plunge line is lifted too (§4 S13)
      float cyc2 = exp(-0.5 * pow((tau - JET_DELAY) / 0.12, 2.0)) * jetMask(x, bseed) * hasJetG(bseed) * (1.0 - smoothstep(0.3, 0.8, zz - zI));
      w0 += 1.2 * cyc2;
      vec2 fl = sweVel(pos.xz);
      vel = vec3(uc * 0.25 + g3 * 0.12, w0 * (1.0 + 0.08 * clamp(g4, -2.0, 2.0)), fl.y * 0.85 + g5 * 0.2);
      rad = mix(0.035, 0.065, h3.x) * sqrt(hs);
      dens = 22.0 * mix(0.5, 1.0, smoothstep(0.25, 0.45, hc.y)) * mix(1.0, 0.7, smoothstep(0.5, 1.0, tau));
      tF = 0.25 + 0.2 * h3.y;
      fph = min(patchiness(pos.xz, bseed) * 1.25, 1.0);
    }
#if WW_EXPLORE
    // far from the player: fewer, larger parcels with the same coverage and optical depth per
    // parcel (the emission density was scaled by lodAt on the CPU)
    float lod = lodAt(x);
    if (lod < 0.999) {
      float sc = inversesqrt(lod);
      if (isBlob(kind)) { rad *= sc; dens /= sc; }
      else if (!isLig(kind)) rad *= min(sc, 2.5);
    }
#endif
    // spread births over the step (no 1/120 s layering)
    float sub = h3.z * uDtS;
    pos += vel * sub;
    gl_FragColor = vec4(pos, sub);
    outVel = vec4(vel, float(kind + (kind == K_SPILL ? 16 : 0)));   // SPILL rides the surface from birth
    outAttr = vec4(rad, dens, tF, floor(fract(h2.x * 7.31 + h3.y) * 4095.0) + clamp(fph, 0.0, 0.999));
    return;
  }

  // ------------------------------------------------------------ integrate
  float age = P.w;
  if (age > 90.0) { gl_FragColor = P; outVel = Vl; outAttr = At; return; }
  int kd = int(Vl.w + 0.5);
  bool landed = kd >= 16;
  if (landed) kd -= 16;
  age += uDtS;
  vec3 pos = P.xyz, vel = Vl.xyz;
  if (landed) {
    if (kd == K_SPILL) {
      // the whitecap rides the front face with the crest; it is left behind as the face flattens
      vel.xz *= exp(-uDtS * 0.8);
      vel.y = 0.0;
      pos.xz += vel.xz * uDtS;
      // the whitecap grows into a roller of aerated water standing on the collapsing crest: its
      // parcels tumble at 0-25 cm above the surface (a volume, not a film)
      // (the roller stands tallest once its crest has collapsed: wave C's spilling roller is
      //  0.3-0.4 m tall at 7.0-7.5 s)
      float hov = SPILL_HOV * At.x * pow(fract(floor(At.w) * 0.0617), 0.7) * smoothstep(0.15, uSpillWin.x, age) * (1.0 - smoothstep(uSpillWin.y, uSpillWin.z, age));
      pos.y = waterLevel(pos.xz) + max(At.z * exp(-age / 0.5), hov) + 0.2 * At.x;
    } else {
      // riding the bore / swash: advected by the shallow-water flow, handing over to the SWE foam
#if WW_EXPLORE
      vec4 sw = swState(pos.xz);
      vec2 fv = sw.yz;
      bool inW = true;
#else
      vec4 sw = textureLod(uSweView, sweUV(pos.xz), 0.0);
      vec2 fv = sw.yz * sweInside(pos.xz);
      bool inW = sweInside(pos.xz) > 0.5;
#endif
      vel.xz = mix(vel.xz, fv, 1.0 - exp(-uDtS * 5.0));
      // a raft stranded on the thin uprush / backwash film (< 3-4 cm) drains away and pops quickly
      if (sw.x < 0.02 && inW) age += uDtS * 2.0 * (1.0 - smoothstep(0.008, 0.02, sw.x));
      // rafts drawn back seaward by the backwash break up quickly (the seaward edge of the
      // whitewater thins to a low, streaky band)
      age += uDtS * 1.5 * smoothstep(0.2, 0.8, -fv.y);
      vel.y = 0.0;
      pos.xz += vel.xz * uDtS;
      // floats on the surface as a flattening, tearing raft of foam (no submerged grey balls)
      pos.y = waterLevel(pos.xz) + 0.15 * At.x;
    }
    if (age - (kd == K_SPILL ? 0.0 : At.z) > (kd == K_SPILL ? uSpillWin.w : (kd == K_BORE ? 0.55 : 0.8))) age = 99.0;
  } else {
    float drag = kd == K_BOIL || kd == K_BORE || kd == K_SPILL ? 0.8 : (kd == K_VEIL ? 1.6 : (kd == K_FSHEET ? 1.2 : (kd == K_ROLL ? 0.8 : (isBlob(kd) ? 0.3 : 0.2))));
    // aerated lumps tumbling in a roller are carried by its turbulence: they hang in the air for
    // ~0.5 s instead of falling ballistically (the roller has volume, not a flat carpet)
    vel.y -= (kd == K_BOIL || kd == K_BORE ? 0.86 : 1.0) * G_ACC * uDtS;
    if (!isBlob(kd) && !isLig(kd)) {
      // Quadratic spherical-drop drag, a = 3 rho_air Cd |v-w|(w-v)/(8 rho_water r).
      // Smaller droplets follow the onshore breeze; the dense crest remains ballistic.
      vec3 wind = vec3(0.35, 0.0, 0.9);
      vec3 relative = vel - wind;
      float kAir = 3.0 * 1.225 * 0.47 / (8.0 * 1000.0 * max(At.x, 0.00035));
      vel = wind + relative / (1.0 + kAir * length(relative) * uDtS);
    } else vel *= exp(-drag * uDtS);
    pos += vel * uDtS;
    float surf = waterLevel(pos.xz);
    if (pos.y < surf && vel.y < 0.0) {
      if (isBlob(kd) && kd != K_VEIL && kd != K_FSHEET) {
        landed = true;
        At.z = age;                        // landing time
        pos.y = surf;
        vel.y = 0.0;
      } else age = 99.0;                    // droplets / ligaments rejoin the water
    }
  }
  if (isLig(kd) && !landed && age > At.z + (kd == K_JETL ? 0.08 : 0.12 + 0.2 * fract(At.w * 0.37))) {
    // ligament pinches off into a droplet at its head (Rayleigh-Plateau)
    kd = kd == K_LIG ? K_DROP : K_JETD;
    At.x *= 1.3;
    At.y = 1.0;
  }
  if (age > lifeMax(kd)) age = 99.0;
  gl_FragColor = vec4(pos, age);
  outVel = vec4(vel, float(kd + (landed ? 16 : 0)));
  outAttr = At;
}`;

// far-field water state over the emission window (explore mode), baked each sim step: what the
// particles beyond the shallow-water window land on, ride and find their bore fronts in
const FARBAKE = /* glsl */ `
uniform vec4 uFarDom;
layout(location = 1) out vec4 outB;
void main() {
  vec2 xz = uFarDom.xy + gl_FragCoord.xy / vec2(${FAR_NX}.0, ${FAR_NZ}.0) * uFarDom.zw;
  vec4 s = swashFar(xz, uTime);
  gl_FragColor = s;
  outB = vec4(s.w + brkSurfaceOnly(xz, uTime), swashFarFoam(xz, uTime).x, 0.0, 0.0);
}`;

// ---------------------------------------------------------------------------- blobs -> optical depth
const BLOB_VERT = /* glsl */ `
uniform sampler2D uPos;
uniform sampler2D uVel;
uniform sampler2D uAttr;
uniform float uPxScale;     // metres per buffer pixel at 1 m depth
in vec2 aCorner;
out vec2 vC;
out vec4 vB;     // (density, view depth, radius, erosion)
out vec4 vN;     // noise offset xy, rotation cos/sin
out vec2 vPh;    // share of the parcel's optical depth in (white foam, aerated water)
out float vDrop; // vertical displacement of this parcel since its crown started to tear (m)
out float vNw;   // weight in the surface-normal channel (flat foam on the water: low)
out float vSheet;// 0 lumpy foam .. 1 streaky water sheet
out vec2 vAsp;   // sprite half extents / radius (keeps the bubble lumps round on squashed sprites)
out float vLand; // 1 = raft floating on the water (no motion streaks)
out float vFine; // close range: weight of the fixed-scale bubble detail (explore mode)
out vec2 vDir;   // sprite x axis on screen (the parcel's surface normal in the front buffer)
uniform int uHideMask;   // debug: bit k hides kind k (bit 15 = landed blobs)
uniform float uBufH;     // thickness buffer height (px)
uniform float uBaseH;    // the white foam of the dome / spilling roller fades out over this height above the water (m)
uniform sampler2D uWaterH;   // total surface elevation over the shallow-water window (SwashSim)
#if WW_EXPLORE
uniform sampler2D uFarB;     // far field: (total surface elevation, foam) over the emission window
uniform vec4 uFarDom;
#endif
float wLevel(vec2 xz) {
  vec2 uv = sweUV(xz);
  float s = (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) ? 0.0 : textureLod(uWaterH, uv, 0.0).r;
#if WW_EXPLORE
  float wf = max(swashFarWeight(xz), 1.0 - sweInside(xz));
  if (wf > 1e-3) s = mix(s, textureLod(uFarB, (xz - uFarDom.xy) / uFarDom.zw, 0.0).x, wf);
#endif
  return s;
}
void main() {
  int id = gl_InstanceID;
  ivec2 ij = ivec2(id % RES, id / RES);
  vec4 P = texelFetch(uPos, ij, 0);
  vec4 Vl = texelFetch(uVel, ij, 0);
  vec4 At = texelFetch(uAttr, ij, 0);
  float age = P.w;
  int kd = int(Vl.w + 0.5);
  bool landed = kd >= 16;
  if (landed) kd -= 16;
  if (age > 90.0 || kd > K_SPILL) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  if (((uHideMask >> kd) & 1) == 1 || (landed && ((uHideMask >> 15) & 1) == 1)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float seed = floor(At.w) / 4096.0;
  float fph = fract(At.w);       // white-foam share given at birth (patchiness; glassy dome sheets)
  // crown parcels (fast launches that tear near their apex) keep the density they had when the
  // sheet started to tear: the tearing itself is done on the aggregate in the resolve (lace)
  bool crown = kd == K_DOME && At.z < 0.45;
  float ga = crown && !landed ? min(age, At.z) : age;
  float grow = 1.0 + (kd == K_DOME ? 0.48 : (isRoller(kd) ? 0.38 : 0.45)) * ga;
  float r = At.x * (1.0 + (kd == K_DOME ? 0.48 : (isRoller(kd) ? 0.38 : 0.45)) * age);
  // (spray sheets thicken as they tear off the crest: no bright seam along the crest line)
  float d = At.y * smoothstep(0.0, kd == K_FSHEET ? 0.09 : 0.02, age) / (grow * grow);
  float ero = 0.0;
  if (landed) {
    float la = age - (kd == K_SPILL ? 0.0 : At.z);
    // landed foam flattens into a raft and hands over to the surface (SWE) foam within ~0.5 s
    d *= exp(-la / (kd == K_SPILL ? 0.4 : (kd == K_BORE || kd == K_BOIL ? 0.3 : 0.25)));
    r *= 1.0 + 0.5 * la;
  } else if (age > At.z) {
    float fa = (age - At.z) / fragT(kd);
    ero = isRoller(kd) ? 0.0 : 1.0 - exp(-fa);
    if (!crown) d *= exp(-0.35 * fa);
  }
  if ((kd == K_DOME || kd == K_JETB) && !landed) {
    // a torn splash parcel is gone ~0.2-0.3 s after its apex: it falls as lace, drops and spray,
    // which is why the visible top of the dome falls faster than g (14-16 m/s^2)
    float tFall = max(-Vl.y / G_ACC, 0.0);
    d *= exp(-pow(tFall / (crown || kd == K_JETB ? 0.13 : 0.2), 2.0));
  }
  d *= 1.0 - smoothstep(0.75, 1.0, age / lifeMax(kd));
  // churning roller: its parcels boil (the density of each lump pulses as it turns over)
  bool churn = isRoller(kd) && !landed;
  if (churn) d *= 0.84 + 0.16 * sin(age * 7.0 + seed * 6.2832);
  // Two phases: white foam (opaque, bright) and aerated water (translucent, grey-green; the
  // bubbly water between and under the foam). The share of the aerated water grows as a splash
  // parcel falls back, low on a roller (the lowest ~12 cm is glassy bubbly water) and on the rafts.
  float hw = P.y - wLevel(P.xz);
  float sh = landed ? (kd == K_SPILL ? 0.5 : 0.0) : sheetness(kd);
  float wfr;
  if (landed) wfr = kd == K_SPILL ? 0.45 : 0.4;
  else if (kd == K_DOME) wfr = mix(0.56, 0.79, smoothstep(0.3, -1.2, Vl.y));
  else if (kd == K_VEIL) wfr = 0.7;
  else if (kd == K_FSHEET) wfr = 0.66;                   // feathering: translucent white sheets
  else if (kd == K_JETB) wfr = 0.74;                      // glassy sheet, white where it frays
  else if (isRoller(kd)) wfr = mix(0.8, 0.5, smoothstep(0.03, 0.12, hw));
  else wfr = 0.5;
  // (a glassy sheet of the crown: mostly water)
  if (kd == K_DOME && !landed) { float gs = smoothstep(0.45, 0.2, fph); sh = mix(sh, 1.0, gs); wfr = mix(wfr, 0.85, gs); }
  float fF = (1.0 - wfr) * (kd == K_DOME ? smoothstep(0.1, 0.9, fph) * 0.8 + 0.2 : fph);
  // white foam fades out towards the water it floats on (a roller's base is bubbly water)
  // (and so does the base of the splash dome and of the spilling roller: glassy, bubbly water)
  if (churn) fF *= smoothstep(-0.01, 0.15, hw);
  else if ((kd == K_DOME && !landed) || (kd == K_SPILL && landed)) fF *= smoothstep(-0.01, uBaseH, hw);
  // (a tearing crown sheet frays into spray and lace: its glassy water goes with it)
  if (kd == K_DOME) {wfr *= 1.0 - 0.25 * ero;fF *= 1.0 - 0.45 * ero;}
  vPh = vec2(fF, wfr);
  vec4 mv = viewMatrix * vec4(P.xyz, 1.0);
#if WW_EXPLORE
  // close to the eye: soft clip against the near plane, and no parcel may fill the view (a lump
  // larger than ~15 % of the screen height fades out instead of becoming a blurry disc)
  float zc = -mv.z;
  d *= smoothstep(0.1, 0.45, zc - 0.5 * r);
  d *= 1.0 - smoothstep(0.13, 0.22, r / max(zc * uPxScale, 1e-6) / uBufH);
#endif
  // cull parcels whose optical depth no longer matters (d * chord)
  if (d * r < 0.012) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  // (no motion stretch: the phone's shutter is short; only thin sheets are drawn out a little
  //  along their motion)
  vec3 vv = mat3(viewMatrix) * Vl.xyz;
  float sp = length(vv.xy);
  float st = landed ? 1.0 : 1.0 + 0.38 * smoothstep(0.6, 1.0, sh) * smoothstep(0.5, 3.0, sp);
  vec2 dir = sp > 1e-3 ? vv.xy / sp : vec2(0.0, 1.0);
  vec2 perp = vec2(-dir.y, dir.x);
  float ra = r * st, rb = r / sqrt(st);
  if (isRoller(kd) && !landed) {
    float an = seed * 71.3, asp = mix(1.0, 2.2, fract(seed * 29.7));
    dir = vec2(cos(an), sin(an)); perp = vec2(-dir.y, dir.x);
    ra = r * sqrt(asp); rb = r / sqrt(asp);
  }
#if WW_EXPLORE
  if (landed) {
    // a raft is a horizontal disc: its screen ellipse follows the view (round seen from above,
    // flat at grazing angles; ~0.42 at the clip camera)
    vec3 upV = mat3(viewMatrix)[1];
    float ratio = clamp(abs(dot(upV, normalize(mv.xyz))), 0.12, 1.0);
    perp = length(upV.xy) > 1e-3 ? normalize(upV.xy) : vec2(0.0, 1.0);
    dir = vec2(perp.y, -perp.x);
    float ext = kd == K_SPILL ? 1.6 : 1.25;
    ra = r * ext; rb = r * ext * mix(ratio, 1.0, kd == K_SPILL ? 0.1 : 0.0);
  }
#else
  if (landed) { dir = vec2(1.0, 0.0); perp = vec2(0.0, 1.0); ra = r * 1.2; rb = r * 0.5; }
  if (landed && kd == K_SPILL) { ra = r * 1.6; rb = r * 0.75; }
#endif
  // never smaller than ~1.3 buffer pixels (keeps tiny blobs from aliasing)
  float pxm = -mv.z * uPxScale;
#if WW_EXPLORE
  vFine = smoothstep(7e-3, 2.5e-3, pxm);
#else
  vFine = 0.0;
#endif
  float grow2 = max(1.0, 1.3 * pxm / max(rb, 1e-4));
  ra *= grow2; rb *= grow2; d /= grow2 * grow2;
  mv.xy += dir * aCorner.x * ra + perp * aCorner.y * rb;
  vDir = dir;
  vC = aCorner;
  vB = vec4(d, -mv.z, r, ero);
  float ta = landed ? 0.0 : max(age - At.z, 0.0);
  vDrop = Vl.y * ta + 0.5 * G_ACC * ta * ta;          // y(t) - y(t_tear), ballistic
  vNw = landed ? 0.22 : (kd == K_BORE ? 0.46 : 0.82);
  vSheet = sh;
  vAsp = vec2(ra, rb) / (grow2 * max(r, 1e-4));
  vLand = landed ? 1.0 : 0.0;
  float ang = seed * 43.98;
  // the lumps of a churning roller tumble (3-6 rad/s)
  // (rafts riding the bore deform and turn more slowly)
  if (churn || landed) ang += (fract(seed * 17.3) < 0.5 ? -1.0 : 1.0) * mix(3.0, 6.0, fract(seed * 5.71)) * (landed ? 0.4 : 1.0) * age;
  vN = vec4(fract(seed * vec2(13.7, 71.3)), cos(ang), sin(ang));
  gl_Position = projectionMatrix * mv;
}`;

const BLOB_FRAG = /* glsl */ `
uniform sampler2D uSceneDepth;
uniform sampler2D uNoise;
uniform vec2 uBufSize;
uniform mat3 uCamRot;
in vec2 vC;
in vec4 vB;
in vec4 vN;
in vec2 vPh;
in float vDrop;
in float vNw;
in float vSheet;
in vec2 vAsp;
in float vLand;
in float vFine;
in vec2 vDir;
uniform float uPxScaleB;
uniform float uFrontSph;   // how much of a parcel's round cap shows in its surface normal
uniform vec2 uFrontL;      // (depth fall-off (m) of the front weighting, dense threshold of the front pre-pass)
#if !WW_ZPASS
uniform sampler2D uFrontZ; // front pre-pass: MAX of 1 / depth of the dense foam at this pixel
layout(location = 1) out vec4 outDetail;
layout(location = 2) out vec4 outLump;
#endif
void main() {
  float d2 = dot(vC, vC);
  vec2 cI = mix(vC * vAsp, vC, vSheet);
  // close range (explore): bubble clumps at a fixed world scale (5 mm and 1.1 cm), anchored to the
  // parcel, so a lump seen from 1-2 m is a mass of bubbles and not a magnified blur
  // (evaluated before any discard: its screen derivatives give the bump slope of the surface)
  float fineA = 0.0, fineE = 0.0, bubF = 0.5;
  vec2 bumpG = vec2(0.0);
#if WW_EXPLORE && !WW_ZPASS
  if (vFine > 0.0) {
    // bubbles at a fixed world scale, anchored to the parcel: Voronoi cells of ~7 mm and ~3.5 mm
    // (1.5-8 mm bubbles; the scale varies a little from parcel to parcel). Bright Plateau borders
    // (the walls within ~1 px of the cell edges: F2 - F1), clear, slightly concave cell interiors
    // (the water behind shows through), a cap bump per bubble (glints in the resolve).
    float pxb = max(vB.y * uPxScaleB, 1e-6);
    vec2 pw = mat2(vN.w, -vN.z, vN.z, vN.w) * cI * vB.z * mix(0.8, 1.25, vN.x);   // m, sprite plane
    vec4 t1 = texture(uNoise, pw / 0.07 + vN.yx * 7.3);                            // 10 cells / 7 cm
    vec4 t2 = texture(uNoise, mat2(0.8, -0.6, 0.6, 0.8) * pw / 0.035 + vN.xy * 3.1);
    float b1 = 1.0 - smoothstep(0.35, 1.1, t1.a * 0.25 * 0.007 / pxb);
    float b2 = 1.0 - smoothstep(0.35, 1.1, t2.a * 0.25 * 0.0035 / pxb);
    float wallB = max(b1, 0.75 * b2);
    fineA = vFine * 0.9 * (wallB - 0.45);
    fineE = vFine * 0.5 * (1.0 - wallB);
    bubF = 0.3 + 0.7 * wallB;
    float cap1 = 1.0 - min(t1.b * 1.4, 1.0), cap2 = 1.0 - min(t2.b * 1.4, 1.0);
    float hb = 0.0025 * cap1 * cap1 + 0.0012 * cap2 * cap2;           // bump height (m)
    bumpG = vec2(dFdx(hb), dFdy(hb)) / pxb;
  }
#endif
  vec2 q = mat2(vN.z, vN.w, -vN.w, vN.z) * cI * (vB.z / 0.4) + vN.xy;
  vec4 nz = texture(uNoise, q);
  float prof = sqrt(max(1.0 - d2, 0.0));
#if !WW_ZPASS
  // the lump's own surface (a lumpy cap; flat for a raft on the water): its slope on screen
  float hL = vB.z * prof * (0.55 + 0.75 * nz.r) * mix(1.0, 0.35, vLand);
  vec2 gL = vec2(dFdx(hL), dFdy(hL)) / max(vB.y * uPxScaleB, 1e-6);
#endif
  if (d2 > 1.0) discard;
  // fibres of a thin sheet, sheared along its motion (sprite x axis = motion direction); foam
  // lumps and rafts get isotropic bubble clumps instead
  float sM = smoothstep(0.5, 0.9, vSheet) * (1.0 - vLand);
  float streak = mix(0.5, texture(uNoise, vec2(vC.x * 0.05, vC.y * 0.5) * (vB.z / 0.1) + vN.yx).a, sM);
  // foam parcels: bubbly lumps with a ragged edge; water sheets: a soft, translucent edge
  // (the fibres modulate the inside of a sheet, never its outline: a streak-cut outline reads as
  //  a comb of hairs / fur)
  float edge = (1.0 - d2) - 0.45 * (1.0 - nz.g) * (1.0 - 0.6 * vSheet) - 0.3 * fineE * (0.3 + 0.7 * d2);
  float sil = smoothstep(0.0, mix(0.22, 0.75, vSheet) * (1.0 - 0.5 * vFine), edge);
  float body = mix((0.45 + 0.9 * nz.r) * (0.55 + 0.9 * streak), (0.5 + 0.7 * nz.r) * (0.4 + 1.0 * streak), vSheet);
  body *= max(1.0 + 0.9 * fineA, 0.15);
  float chord = 2.0 * vB.z * prof * body * sil;
  float sceneZ = linearizeDepth(textureLod(uSceneDepth, gl_FragCoord.xy / uBufSize, 0.0).x);
  float zf = vB.y - 0.5 * chord;
#if WW_ZPASS
  // front pre-pass: where the dense white foam starts (parcels thick enough to hide what is
  // behind them, in front of the water surface)
  if (vB.x * vPh.x * chord < uFrontL.y || zf > sceneZ + 0.03) discard;
  gl_FragColor = vec4(1.0 / max(zf, 0.05), 0.0, 0.0, 0.0);
#else
  // the aerated water of the parcel: a smooth, irregular body (no bubble lumps)
  float chordW = 2.0 * vB.z * prof * smoothstep(0.0, 0.45, (1.0 - d2) - 0.3 * (1.0 - nz.r));
  if (max(chord, chordW) <= 1e-5) discard;
  float tauFull = vB.x * vPh.x * chord;
  float vis = clamp(sceneZ - zf, 0.0, chord);
  // foam behind the (aerated, turbulent) water surface still shows through it, fading
  // over ~12 cm of water in front of it
  float hid = chord - vis;
  float d0 = max(zf - sceneZ, 0.0);                       // water in front of the blob's front
  float seen = 0.12 * exp(-d0 / 0.12) * (1.0 - exp(-hid / 0.12));
  float tau = vB.x * vPh.x * (vis + 0.55 * seen);
  float zfW = vB.y - 0.5 * chordW;
  float visW = clamp(sceneZ - zfW, 0.0, chordW);
  float tauW = vB.x * vPh.y * (visW + 0.4 * 0.12 * exp(-max(zfW - sceneZ, 0.0) / 0.12) * (1.0 - exp(-(chordW - visW) / 0.12)));
  // the unclipped optical depth drives the surface normal (the water line is not a foam edge)
  float keep = smoothstep(-0.25, 0.0, sceneZ - zf);
  gl_FragColor = vec4(tau, tauW, (tau + tauW) * vB.y, tauFull * keep * vNw * (1.0 - 0.6 * vSheet));
  // What the eye sees of foam is its front surface: the parcels at the front of the dense foam
  // (front pre-pass) are weighted in, those more than a few cm behind it fade out. Their lumpy
  // caps (surface normal) and bubbly texture are accumulated with that weight, so the lumps of
  // the front billows keep their own shading and texture instead of averaging into a grey mottle.
  float kF = texture(uFrontZ, gl_FragCoord.xy / uBufSize).r;   // (filtered: no per-texel steps)
  float zFr = kF > 0.0 ? 1.0 / kF : 1e4;
  // (a lump's rim weighs less: its steep edge must not draw a ring over the lumps around it)
  float om = exp(-max(zf - zFr, 0.0) / uFrontL.x) * min(tau, 2.0) * smoothstep(0.0, 0.45, 1.0 - d2);
  // particle-anchored detail (1-3 cm bubble clumps + fibres sheared along the motion)
  float fine = texture(uNoise, mat2(vN.z, vN.w, -vN.w, vN.z) * cI * (vB.z / 0.3) + vN.yx * 3.7).g;
  float fib = texture(uNoise, mix(vec2(vC.x * vB.z / 0.6, vC.y * vB.z / 0.1), cI * (vB.z / 0.2), vLand) + vN.xy * 5.3).a;
  float d = 1.1 * (fib - 0.5) + 0.8 * (fine - 0.45) + 1.4 * fineA + 1.2 * vFine * (bubF - 0.5);
  // lump normal (view space xy): the lumpy cap (+ the close-range bubble bumps)
  vec2 nxy = -(gL * uFrontSph + bumpG * 1.4);
  nxy /= sqrt(1.0 + dot(nxy, nxy));
  // tearing state for the lace in the resolve: sum tau*erosion, sum tau*drop since tearing
  float tauA = tau + tauW;
  outDetail = vec4(om * d, om, tauA * vB.w, tauA * vDrop);
  outLump = vec4(om * nxy, 0.0, 0.0);
#endif
}`;

// ---------------------------------------------------------------------------- resolve (full res)
const RESOLVE = /* glsl */ `
uniform sampler2D uThick;     // (tau white foam, tau aerated water, tau * view z, tau lump), mip-mapped
uniform sampler2D uDetail;    // front-weighted (sum w d, sum w) + tearing state (sum tau erosion, sum tau drop)
uniform sampler2D uLump;      // front-weighted lump normal (sum w n.x, sum w n.y), view space
uniform sampler2D uFrontZ;    // front pre-pass: 1 / depth of the dense foam (MAX)
uniform sampler2D uSceneDepth;
uniform vec2 uThickTexel;
uniform float uPxScale;
uniform mat3 uCamRot;
uniform vec4 uFoamK;      // (kappa foam, normal gain, log-height scale (m), cavity gain)
uniform vec4 uFoamK2;     // (foam transfer exponent, kappa aerated water, billow normal gain, billow cavity gain)
uniform vec3 uWaterCol;   // in-scatter of the aerated water (scene-linear, sunlit)
uniform vec4 uFoamK3;     // (lowest normal y in view space, sky seen by a downward face, cavity floor, lump normal gain)
uniform vec4 uBil;        // low foam billows: (weight, coverage gain, crevice gain, height (m))
uniform vec4 uCrev;       // front outlines: (crevice darkening, near-side lift, depth step (m), glint gain)
uniform vec4 uLaceK;      // torn crown: (cell size (m), glass alpha, rim gain, speck density)
uniform vec2 uViewSize;   // output size (px)
uniform float uBilUp;     // how much the low (floating) foam's normal turns up
uniform float uBilT;      // flow-map cycle of the low billows (s)
uniform vec3 uSplashK;    // splash thrown up high: (lump normal gain, billow normal gain, cavity reduction)
uniform vec4 uLight;      // (sun gain, ambient gain, sun-facing boost, detail gain)
uniform float uDetCol;    // brightness modulation by the particle-anchored detail
uniform float uGlass;     // how glassy (sheen on the aerated water's sheets and edges)
uniform vec3 uCamPos;
uniform vec2 uTanFov;     // (tan(hfov/2), tan(vfov/2))
uniform vec2 uSunScreen;  // screen-space direction towards the sun (unit, in buffer texels)
uniform float uShadow;    // sun visibility deep inside the foam mass
uniform vec3 uAmbCol;     // sky ambient colour on the foam (blue)
uniform vec3 uAlbedo;     // foam albedo (multiple scattering in aerated water: a cool white)
uniform float uPxFull;    // metres per output pixel at 1 m depth
uniform vec4 uLaceAx;     // horizontal axes of the lace web: (screen right xz, its normal xz); clip: (1, 0, 0, 1)
uniform sampler2D uNoise;
uniform int uDbg;
in vec2 vUv;
float hOf(float tau) { return uFoamK.z * log(1.0 + tau); }
float H(vec2 uv) { return hOf(textureLod(uThick, uv, 0.0).a); }
float HL(vec2 uv, float lod) { return hOf(textureLod(uThick, uv, lod).a); }
// smooth 2D value noise (quintic), -1..1
float vn2(vec2 p) { return vnoise(p) * 2.0 - 1.0; }
// Voronoi: exact distance to the nearest cell border, distance to the cell's centre (in cells)
// and the cell's hash
vec3 voronoiB(vec2 x) {
  vec2 n = floor(x), f = fract(x);
  vec2 mg = vec2(0.0), mr = vec2(0.0);
  float md = 8.0;
  for (int j = -1; j <= 1 + min(uEvtCount, 0); j++) for (int i = -1; i <= 1; i++) {
    vec2 g = vec2(float(i), float(j));
    vec2 r = g + hash22(n + g) - f;
    float d = dot(r, r);
    if (d < md) { md = d; mr = r; mg = g; }
  }
  float mb = 8.0;
  for (int j = -2; j <= 2 + min(uEvtCount, 0); j++) for (int i = -2; i <= 2; i++) {
    vec2 g = mg + vec2(float(i), float(j));
    vec2 r = g + hash22(n + g) - f;
    vec2 dr = r - mr;
    if (dot(dr, dr) > 1e-5) mb = min(mb, dot(0.5 * (mr + r), normalize(dr)));
  }
  return vec3(mb, sqrt(md), hash12(n + mg));
}
void main() {
  vec4 T = texture(uThick, vUv);
  float tauF = T.r, tauW = T.g;
  float tauA = tauF + tauW;
  if (tauA < 2e-3) { gl_FragColor = vec4(0.0); return; }
  float zAvg = T.b / tauA;
  vec2 e = uThickTexel;
  float pxm = zAvg * uPxScale;              // metres per buffer texel at the foam
  float px = zAvg * uPxFull;                // metres per output pixel
  vec3 ray = vec3((vUv * 2.0 - 1.0) * uTanFov, -1.0);
  vec3 Pw = uCamPos + uCamRot * ray * zAvg;
  // particle-anchored detail of the front lumps (front-weighted mean): log-normal modulation of
  // the white foam's optical depth, so thin foam frays into lace while the dense core only gets a
  // bubbly texture
  vec4 Dt = texture(uDetail, vUv);
  float det = Dt.y > 1e-3 ? clamp(Dt.x / Dt.y, -1.5, 1.5) * uLight.w * smoothstep(0.0, 0.05, Dt.y) : 0.0;
  float tauD = tauF * exp(0.6 * det);
  // Crown tearing into lace (breaker_morphology 5.4, +0.47 s): white-rimmed webs of foam on glassy
  // water. Cells of a Voronoi web carried with the falling crown open as the sheet tears (the
  // walls thin from 3.5 cm to 0.8 cm half-width: at the apex 60-70 % of the crown is glass); the
  // windows are glassy water, darker and greyer than the sea behind, speckled with droplets.
  float eroA = Dt.z / tauA;
  float hole = 0.0, rimL = 0.0, speck = 0.0, ee = 0.0;
  if (eroA > 0.01) {
    float dropA = Dt.w / tauA;
    // sheets are stretched along their (mostly vertical) motion: cells elongate vertically
    float ax = dot(Pw.xz, uLaceAx.xy), az = dot(Pw.xz, uLaceAx.zw);
    vec2 lq = vec2(ax + 0.35 * az, (Pw.y - dropA - 0.2 * az) * 0.8) / uLaceK.x;
    // smooth warp (no creases): curved, looping webs
    float sheetScale = 0.78 + 0.46 * vn2(lq * 0.18 + 5.4);
    vec2 wq = lq * sheetScale + 0.68 * vec2(vn2(lq * 0.6 + 3.1), vn2(lq * 0.6 + 17.7)) + 0.18 * vec2(vn2(lq * 2.1), vn2(lq * 2.1 + 9.2));
    vec3 vb = voronoiB(wq);
    ee = smoothstep(0.025 + vb.z * 0.075, 0.46 + vb.z * 0.23, eroA);
    // each window is a rounded hole (surface tension) growing in its cell: inside the cell's
    // border by the web half-width (from 4 cm to 0.5-0.9 cm as the sheet tears) and within a
    // growing radius of its centre (some cells stay closed longer)
    float hw = mix(0.04, mix(0.005, 0.009, vb.z), ee);
    float R = uLaceK.x * mix(0.08, 0.81, ee) * mix(0.34, 1.18, vb.z);
    float hd = min(vb.x * uLaceK.x - hw, R - vb.y * uLaceK.x);   // > 0 inside the window (m)
    float aa = max(0.6 * px, .7 * fwidth(hd));
    hole = smoothstep(-aa, aa, hd) * smoothstep(0.02, 0.12, eroA);
    // the rim of each window: a bright, sunlit edge (1-2 px) on the web side
    rimL = smoothstep(-2.4 * px, -0.6 * px, hd) * (1.0 - smoothstep(-aa, aa, hd)) * ee;
    // droplets in the windows (0.5-1.5 cm, world-anchored with the crown)
    vec2 sq = lq * uLaceK.x / 0.018;
    vec2 sc = floor(sq);
    vec2 sp = hash22(sc + 7.1);
    float sd = length((fract(sq) - sp) * 0.018) / px;
    speck = hole * step(hash12(sc + 3.7), uLaceK.w) * (1.0 - smoothstep(0.35, 1.1, sd));
  }
  // Low foam riding the bore / swash (rafts, bore rollers): billows and folds (5-15 cm lumps) in a
  // world-space field on the water plane, advected with the shallow-water flow (two-phase flow
  // map, 0.5 s cycles, the phase drifting smoothly over the surface: the lumps churn and re-form
  // instead of sliding through a fixed texture). Seen at a grazing angle they are horizontal
  // folds. The field lifts the foam surface (shading), opens gaps between the lumps and darkens
  // the crevices.
  float bil = 0.5, wLow = 0.0;
  vec2 bilG = vec2(0.0);
  vec4 sv = textureLod(uSweView, sweUV(Pw.xz), 0.0);
  float inW = sweInside(Pw.xz);
  // splash thrown well above the water (dome, jets): sheets and spray, crisp and bright rather
  // than shaded lumps
  float splash = smoothstep(0.25, 0.6, Pw.y - sv.w * inW);
  if (uBil.x > 0.0) {
    wLow = inW * (1.0 - smoothstep(0.1, 0.3, Pw.y - sv.w)) * uBil.x;
    if (wLow > 0.01) {
      vec2 vf = sv.yz * inW;
      float ph = uTime / uBilT + 0.8 * textureLod(uNoise, Pw.xz * 0.37 + 0.13, 0.0).r;
      float f1 = fract(ph), f2 = fract(ph + 0.5);
      float w1 = 1.0 - abs(2.0 * f1 - 1.0);
      vec2 o1 = -vf * f1 * uBilT, o2 = -vf * f2 * uBilT + vec2(0.37, 0.71);
      // three points: this pixel and 2 px to the right / up (screen-space slope of the field)
      vec2 du = 2.0 / uViewSize;
      vec3 b3 = vec3(0.0);
      for (int k = 0; k < 3; k++) {
        vec2 uv = vUv + (k == 1 ? vec2(du.x, 0.0) : (k == 2 ? vec2(0.0, du.y) : vec2(0.0)));
        vec3 Pk = uCamPos + uCamRot * vec3((uv * 2.0 - 1.0) * uTanFov, -1.0) * zAvg;
        vec2 q1 = Pk.xz + o1, q2 = Pk.xz + o2;
        // The short-cycle displacement carries folds with the actual flow;
        // fixed world axes prevent velocity changes from rotating the pattern.
        // (smooth value noise: rounded lumps without the creases of a cellular field)
        float keepFine=1.0-smoothstep(.009,.022,pxm);
        float a1 = 0.70 * vnoise(q1 / 0.16) + .30 * mix(.5,vnoise(q1 / 0.05 + 7.31),keepFine);
        float a2 = 0.70 * vnoise(q2 / 0.16) + .30 * mix(.5,vnoise(q2 / 0.05 + 7.31),keepFine);
        b3[k] = mix(a2, a1, w1);
      }
      bil = mix(0.5, b3.x, wLow);
      bilG = wLow * uBil.w * vec2(b3.y - b3.x, b3.z - b3.x) / (2.0 * px);
      tauD *= exp(uBil.y * (b3.x - 0.45) * 2.0 * wLow);

    }
  }
  // Two phases, both order independent sums over the parcels:
  //  - white foam: surface-like (bubble walls), a mildly steep optical-depth -> coverage transfer
  //    (the body is mostly opaque, its thinner parts let the water phase through);
  //  - aerated water: bubbly, translucent water (in-scatter over what lies behind).
  float aF0 = 1.0 - exp(-pow(uFoamK.x * tauD, uFoamK2.x));
  // thin sheets and spray thrown up high (jets, feathering, the curtain veil) stay a translucent
  // veil (alpha ~0.3-0.5) rather than vanishing (the steep transfer is for the foam body)
  aF0 = max(aF0, splash * 0.6 * (1.0 - exp(-4.0 * uFoamK.x * tauD)));
  float aF = aF0 * (1.0 - hole);
  float aW = 1.0 - exp(-uFoamK2.y * tauW);
  // foam surface normal from the gradient of the (log) optical depth on two rings of fixed
  // world size (1.3 cm and 4 cm): the lumps of the mass
  float r1 = clamp(0.013 / pxm, 1.0, 6.0), r2 = clamp(0.04 / pxm, 2.5, 16.0);
  vec2 g1 = vec2(0.0), g2 = vec2(0.0), gw = vec2(0.0);
  float ring1 = 0.0, ring2 = 0.0;
  for (int i = 0; i < 8; i++) {
    float a = float(i) * 0.785398 + 0.3927;
    vec2 d = vec2(cos(a), sin(a));
    vec2 t1 = textureLod(uThick, vUv + d * e * r1, 0.0).ag;
    float h1 = hOf(t1.x);
    float h2 = H(vUv + d * e * r2);
    g1 += d * h1; g2 += d * h2;
    ring1 += h1; ring2 += h2;
    gw += d * log(1.0 + uFoamK2.y * t1.y);
  }
  g1 /= 4.0 * r1 * pxm;       // sum_i d_i (d_i . g r) = 4 g r
  g2 /= 4.0 * r2 * pxm;
  gw /= 4.0 * r1 * pxm;       // slope of the aerated water's log optical depth (1/m): its edges
  // billows (9 and 22 cm): the same height field on the mip chain, so the 10-40 cm lumps of the
  // mass are shaded (lit tops, grey undersides, darker crevices between them)
  vec2 gB = vec2(0.0);
  float lapB = 0.0;
  for (int k = 0; k < 2; k++) {
    float s = k == 0 ? 0.09 : 0.22;
    float rt = s / pxm;
    float lod = log2(max(0.5 * rt, 1.0));
    float c0 = HL(vUv, lod);
    float hxp = HL(vUv + vec2(e.x * rt, 0.0), lod), hxm = HL(vUv - vec2(e.x * rt, 0.0), lod);
    float hyp = HL(vUv + vec2(0.0, e.y * rt), lod), hym = HL(vUv - vec2(0.0, e.y * rt), lod);
    float wk = k == 0 ? 0.6 : 0.4;
    gB += wk * vec2(hxp - hxm, hyp - hym) / (2.0 * s);
    lapB += wk * (hxp + hxm + hyp + hym - 4.0 * c0) / s;
  }
  // (no 1-texel micro-bump term: the reference whitewater is smooth at the pixel scale, its
  //  texture lives in the 1-10 cm streaks and bubble clumps)
  vec2 grad = (0.6 * g1 + 0.4 * g2) * uFoamK.y + gB * uFoamK2.z * mix(1.0, uSplashK.y, splash) + bilG;
#if WW_EXPLORE
  // close range: the bubble clumps (3-5 mm) shade too (a third ring of ~3.5 mm)
  float wNear = smoothstep(4.5e-3, 1.8e-3, pxm);
  if (wNear > 0.0) {
    float r0 = clamp(0.0035 / pxm, 1.0, 2.5);
    vec2 g0 = vec2(0.0);
    for (int i = 0; i < 4; i++) {
      float a = float(i) * 1.5707963 + 0.7853982;
      vec2 d = vec2(cos(a), sin(a));
      g0 += d * H(vUv + d * e * r0);
    }
    g0 /= 2.0 * r0 * pxm;
    grad = mix(grad, (0.35 * g0 + 0.4 * g1 + 0.25 * g2) * uFoamK.y + gB * uFoamK2.z + bilG, wNear);
  }
#endif
  // the front lumps' own caps (front-weighted lump normals): each lump is lit from the sun side,
  // shaded on its underside
  vec2 nL = Dt.y > 1e-3 ? texture(uLump, vUv).xy / Dt.y : vec2(0.0);
  nL *= smoothstep(0.0, 0.08, Dt.y) * mix(1.0, uSplashK.x, splash);
  // (undersides may face down: they see neither the sun nor much of the sky - the grey of the
  //  lower billows)
  vec3 nV = normalize(vec3(-grad.x + uFoamK3.w * nL.x, max(-grad.y + uFoamK3.w * nL.y, uFoamK3.x), 1.0));
  // (foam floating on the water faces up rather than towards the camera)
  vec3 N = normalize(uCamRot * nV + vec3(0.0, 0.35 + uBilUp * wLow, 0.0));
  // cavities: the centre lies below its surroundings (lumps: 1-4 cm; crevices between billows)
  float lap = (ring2 - ring1) / (8.0 * r2 * pxm);
  float cav = clamp(1.0 - uFoamK.w * lap * 1.5 - uFoamK2.w * lapB, uFoamK3.z, 1.08);
  cav *= 1.0 - uBil.z * (0.5 - bil);                   // crevices between the low billows
  cav = mix(cav, 1.0, uSplashK.z * splash);
  // Crisp outlines of the front lumps: where a lump lies in front of the foam next to it, the foam
  // just behind its edge is in a crevice (contact shadow, 2-4 px) and the lump's own edge catches
  // a little more light.
  float kS = textureLod(uFrontZ, vUv, 0.0).r;
  // (not on the low foam riding the water: neighbouring rafts lie at different depths without any
  //  crevice between them)
  if (kS > 0.0 && uCrev.x > 0.0 && wLow < 0.99) {
    float zS = 1.0 / kS, dzF = 0.0, dzN = 0.0;
    for (int i = 0; i < 6; i++) {
      float a = float(i) * 1.0471976 + 0.3;
      float rr = i < 3 ? 2.5 : 5.0;
      float kN = textureLod(uFrontZ, vUv + vec2(cos(a), sin(a)) * rr * e, 0.0).r;
      if (kN > 0.0) {
        float dz = zS - 1.0 / kN;                        // > 0: the neighbour is in front
        dzF = max(dzF, dz);
        dzN = max(dzN, -dz);
      }
    }
    float kc = 1.0 - wLow;
    cav *= 1.0 - uCrev.x * kc * (1.0 - 0.5 * splash) * smoothstep(uCrev.z, 3.0 * uCrev.z, dzF);
    cav *= 1.0 + uCrev.y * kc * smoothstep(uCrev.z, 3.0 * uCrev.z, dzN);
  }
  // self-shadowing: foam above (towards the high sun) shades the lower parts of the mass,
  // screen-space march of the optical depth over 2-32 cm
  float occ = 0.0;
  for (int i = 1; i <= 4; i++) {
    float dm = 0.02 * float(i * i);                    // 2, 8, 18, 32 cm -> keep within the lump scale
    vec2 ts = textureLod(uThick, vUv + uSunScreen * min(dm / pxm, 60.0) * e, 0.0).rg;
    occ += (1.0 - exp(-0.5 * (ts.x + 0.3 * ts.y))) * (1.2 - 0.2 * float(i));
  }
  float shadow = mix(uShadow, 1.0, exp(-0.45 * occ));
  // foam = dense multiple-scattering medium: the sun is high and ahead, yet the faces towards the
  // camera are bright too (sunlight diffuses through 5-20 cm of foam); what darkens foam is other
  // foam between it and the sun (self-shadow), downward faces and the crevices between the billows
  float ndl = dot(N, uSunDir);
  float wrap = 0.5 + 0.5 * clamp((ndl + 0.8) / 1.8, 0.0, 1.0);
  wrap *= 1.0 + uLight.z * smoothstep(0.5, 0.9, ndl);   // sunlit caps: the brightest whites
  vec3 amb = uAmbCol * mix(uFoamK3.y, 1.05, 0.5 + 0.5 * N.y);
  const float SUNVIS = 0.87;               // mean sun visibility of the parcels in the mass
  vec3 col = uAlbedo * (uSunColor * (uLight.x / 3.14159) * wrap * SUNVIS * shadow + amb * uLight.y * mix(0.8, 1.0, shadow)) * cav;
  col *= clamp(1.0 + uDetCol * det * mix(1.0, 1.6, splash), 0.78, 1.15);
  // sparse wet glints: a bubble cap here and there mirrors the sun (1-2 px, 1.3-1.6 x white)
  vec3 V = normalize(uCamRot * ray);
  float nh = dot(N, normalize(uSunDir - V));
  vec2 capQ = (Pw.xz - sv.yz * inW * 0.03) / 0.009;
  vec2 gc = floor(capQ);
  vec2 gp = hash22(gc + 0.37);
  float gd = length(fract(capQ) - gp) * 0.009 / max(px, 0.0005);
  float glint = uCrev.w * smoothstep(0.985, 0.998, nh) * step(hash12(gc * 1.7 + 0.3), 0.3) * (1.0 - smoothstep(0.4, 1.2, gd)) * shadow;
  col += uSunColor * 0.25 * glint;

  // torn crown: the web's window rims catch the sun
  col += uSunColor * uLaceK.z * rimL * shadow;
  // aerated water: glassy, bubbly water - sunlit single scattering (shaded like the foam around
  // it) and a bright sheen where a sheet turns away from the view (its edges catch the sky / sun)
  vec3 colW = uWaterCol * mix(0.7, 1.0, shadow) * mix(0.85, 1.05, wrap);
  float sheen = smoothstep(4.0, 16.0, length(gw));
  colW += (vec3(0.55, 0.62, 0.66) + 0.25 * uSunColor * shadow) * sheen * uGlass;
  // the crown's windows: glassy water with a little spray haze (alpha 0.3-0.45)
  float aH = hole * uLaceK.y * mix(0.8, 1.2, ee) * smoothstep(0.05, 0.5, aF0);
  aW = max(aW, aH);
  colW = mix(colW, vec3(0.3, 0.5, 0.52) + 0.06 * uSunColor, hole);
  // droplet specks in the windows
  aF = max(aF, 0.85 * speck);
  col = mix(col, uAlbedo * uSunColor * 0.62 + 0.1, speck);
  // crisp full-res occlusion by water that lies well in front of the foam mass
  float sceneZ = linearizeDepth(textureLod(uSceneDepth, vUv, 0.0).x);
  float vis = smoothstep(-0.6, -0.2, sceneZ - zAvg);
  aF *= vis; aW *= vis;
  // foam over aerated water over the scene (premultiplied)
  float alpha = 1.0 - (1.0 - aF) * (1.0 - aW);
  gl_FragColor = vec4(col * aF + colW * aW * (1.0 - aF), alpha);
  if (uDbg == 1) gl_FragColor = vec4(vec3(tauF / 20.0, tauW / 20.0, aF), 1.0);
  if (uDbg == 2) gl_FragColor = vec4(N * 0.5 + 0.5, 1.0);
  if (uDbg == 3) gl_FragColor = vec4(vec3(shadow, wrap, cav - 0.5), 1.0);
  if (uDbg == 6) gl_FragColor = vec4(vec3(0.5 + 0.2 * det), 1.0);
  if (uDbg == 7) gl_FragColor = vec4(eroA, hole, rimL, 1.0);
  if (uDbg == 8) gl_FragColor = vec4(nL * 0.5 + 0.5, kS * 0.2, 1.0);
  if (uDbg == 10) gl_FragColor = vec4(aF, aW, alpha, 1.0);
  if (uDbg == 11) gl_FragColor = vec4(vec3(shadow), 1.0);
  if (uDbg == 12) gl_FragColor = vec4(vec3(cav), 1.0);
  if (uDbg == 13) gl_FragColor = vec4(vec3(wrap), 1.0);
  if (uDbg == 14) gl_FragColor = vec4(vec3(0.5 + 0.5 * N.y), 1.0);
}`;

// ---------------------------------------------------------------------------- droplets
const DROP_VERT = /* glsl */ `
uniform sampler2D uPos;
uniform sampler2D uVel;
uniform sampler2D uAttr;
uniform int uBase;
uniform float uPxScaleFull;   // metres per output pixel at 1 m depth
uniform int uHideMask;        // debug: bit k hides kind k
in vec2 aCorner;
out vec2 vC;
out vec4 vD;   // (draw radius px, coverage alpha, view depth, sparkle)
out float vL;  // half length (px) of the exposure streak along the sprite's x axis (explore)
void main() {
  int id = uBase + gl_InstanceID;
  ivec2 ij = ivec2(id % RES, id / RES);
  vec4 P = texelFetch(uPos, ij, 0);
  vec4 Vl = texelFetch(uVel, ij, 0);
  vec4 At = texelFetch(uAttr, ij, 0);
  int kd = int(Vl.w + 0.5);
  if (P.w > 90.0 || kd < K_SPRAY || kd > K_JETD || P.w < At.z || ((uHideMask >> kd) & 1) == 1) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec4 mv = viewMatrix * vec4(P.xyz, 1.0);
  float pxm = -mv.z * uPxScaleFull;
  float rpx = At.x / pxm;
  float R = max(rpx, 0.9);
  float a = min(1.0, (rpx / 0.9) * (rpx / 0.9));
  a *= smoothstep(0.0, 0.04, P.w - max(At.z, 0.0)) * (1.0 - smoothstep(0.8, 1.0, P.w / lifeMax(kd)));
#if WW_EXPLORE
  // soft near clip; a drop right in front of the eye is out of focus and gone
  a *= smoothstep(0.12, 0.35, -mv.z) * (1.0 - smoothstep(40.0, 70.0, rpx));
#endif
  float seed = floor(At.w) / 4096.0;
  float L = 0.0;
  vec2 sdir = vec2(1.0, 0.0);
#if WW_EXPLORE
  // phone shutter (~1/200 s): a fast drop close by is a short streak along its motion on screen
  // with its coverage spread over it (a drop resolved as a lens, within ~0.5 m, stays a lens)
  vec2 vs = (mat3(viewMatrix) * Vl.xyz).xy / pxm;                      // px / s
  L = min(length(vs) / 650.0, 7.0) * (1.0 - smoothstep(2.5, 5.0, R));
  if (length(vs) > 1e-3) sdir = normalize(vs);
  a *= (2.0 * R) / (2.0 * R + L);
#endif
  vec2 sq = vec2(-sdir.y, sdir.x);
  mv.xy += (sdir * aCorner.x * (R + 1.0 + 0.5 * L) + sq * aCorner.y * (R + 1.0)) * pxm;
  vC = vec2(aCorner.x * (R + 1.0 + 0.5 * L), aCorner.y * (R + 1.0));
  vL = 0.5 * L;
  vD = vec4(R, a, -mv.z, seed);
  gl_Position = projectionMatrix * mv;
}`;

const OCCLUDE = /* glsl */ `
uniform sampler2D uSceneDepth;
uniform sampler2D uThick;
uniform vec2 uViewport;
uniform vec3 uOccK;       // (kappa foam, foam transfer exponent, kappa aerated water) as in the resolve
float occlusion(float myZ) {
  vec2 suv = gl_FragCoord.xy / uViewport;
  float sceneZ = linearizeDepth(textureLod(uSceneDepth, suv, 0.0).x);
  float o = smoothstep(-0.02, 0.03, sceneZ - myZ);
  vec4 T = texture(uThick, suv);
  if (T.r + T.g > 1e-3) {
    float zA = T.b / (T.r + T.g);
    // fraction of the foam optical depth that lies in front of this point
    float f = clamp(0.5 + (myZ - zA) / 0.3, 0.0, 1.0);
    // same coverage transfer as the foam resolve (white foam, aerated water)
    o *= exp(-pow(uOccK.x * T.r * f, uOccK.y) - uOccK.z * T.g * f);
    // white on white: threads / drops seen against dense foam disappear into it
    o *= 1.0 - 0.8 * (1.0 - exp(-pow(uOccK.x * T.r, uOccK.y)));
  }
  return o;
}
`;

const DROP_FRAG = /* glsl */ OCCLUDE + `
in vec2 vC;
in vec4 vD;
in float vL;
uniform vec2 uSunScr;   // screen-space direction towards the sun
void main() {
  float r = length(vec2(max(abs(vC.x) - vL, 0.0), vC.y));
  float cov = 1.0 - smoothstep(vD.x - 0.6, vD.x + 0.6, r);
  // clear water drops: they mostly show the (refracted) sky and water behind them, so they are
  // translucent; only ~10 % catch the sun as clipped glints
  bool glint = hash11(vD.w * 57.31) < 0.1;
  float a = cov * vD.y * (glint ? 0.9 : 0.6);
  if (a < 0.004) discard;
  a *= occlusion(vD.z);
  if (a < 0.004) discard;
  vec3 col = vec3(0.76, 0.89, 0.94) * (1.0 - 0.2 * smoothstep(0.3, 1.0, r / max(vD.x, 0.9)));
  float tw = glint ? step(0.35, hash11(vD.w * 97.3 + floor(uTime * 24.0))) : 0.0;
  col += uSunColor * 1.4 * tw * (1.0 - smoothstep(0.0, 0.8, r / max(vD.x, 0.9)));
#if WW_EXPLORE
  // resolved drops (close range): a tiny fish-eye lens. The sky is seen refracted in its lower
  // half, the sea / beach in the upper half, a dark total-reflection rim and a sun glint.
  float lens = smoothstep(2.5, 6.0, vD.x);
  if (lens > 0.0) {
    vec2 p = vC / max(vD.x, 1e-3);
    float rr = length(p);
    vec3 lc = mix(vec3(0.30, 0.48, 0.50), vec3(0.82, 0.92, 0.98), smoothstep(0.35, -0.55, p.y));
    lc *= 1.0 - 0.6 * smoothstep(0.7, 0.97, rr);
    vec2 gp = p - uSunScr * 0.42;
    float gl = exp(-dot(gp, gp) / 0.015);
    lc += uSunColor * 1.1 * gl;
    col = mix(col, lc, lens);
    float al = cov * vD.y * mix(0.8, 1.0, max(gl, smoothstep(0.75, 0.97, rr))) * occlusion(vD.z);
    a = mix(a, al, lens);
  }
#endif
  gl_FragColor = vec4(col * a, a);
}`;

// ---------------------------------------------------------------------------- ligaments
const LIG_SEGS = 7;
const LIG_VERT = /* glsl */ `
uniform sampler2D uPos;
uniform sampler2D uVel;
uniform sampler2D uAttr;
uniform int uBase;
uniform float uPxScaleFull;
uniform int uHideMask;
in vec2 aLig;          // (s: 0 head .. 1 tail, side -1 / +1)
out vec2 vW;           // (signed distance from the centreline in px, half width px)
out vec2 vA;           // (alpha, view depth)
out float vS;          // 0 head .. 1 tail
void main() {
  int id = uBase + gl_InstanceID;
  ivec2 ij = ivec2(id % RES, id / RES);
  vec4 P = texelFetch(uPos, ij, 0);
  vec4 Vl = texelFetch(uVel, ij, 0);
  vec4 At = texelFetch(uAttr, ij, 0);
  int kd = int(Vl.w + 0.5);
  if (P.w > 90.0 || kd < K_LIG || kd > K_JETL || P.w < At.z || ((uHideMask >> kd) & 1) == 1) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float age = P.w;
  vec3 V = Vl.xyz;
  float seed = floor(At.w) / 4096.0;
  float s = aLig.x;
  float life = max(age - At.z, 0.0);
  // crown rim ligament: while its parent sheet still exists (the first ~0.1 s after it tears) it
  // lies along the torn rim (mostly lateral, gently bent), carried ballistically with its parcel;
  // then it is drawn out along its own motion (radially out of the dome) into a nearly straight
  // finger with a bead at its head
  float arc = kd == K_LIG ? 1.0 - smoothstep(0.04, 0.12, life) : 0.0;
  float phi = (hash11(seed * 91.7) - 0.5) * 2.4;                  // +-70 deg from horizontal
  float sgn = hash11(seed * 13.3) < 0.5 ? -1.0 : 1.0;
  vec3 T0 = normalize(vec3(sgn * cos(phi), sin(phi), 0.35 * (hash11(seed * 5.9) - 0.5)));
  vec3 B0 = normalize(vec3(T0.y, -T0.x, 0.0) + vec3(0.0, -0.4, 0.0));
  float Lr = At.y * (1.0 + 0.8 * life);                           // stretched while it flies
  float bend = (hash11(seed * 3.1) - 0.5) * 0.7;                  // gently bent (|bend| <= 0.35)
  float uu = s - 0.5;
  vec3 Qa = P.xyz + T0 * (Lr * uu) + B0 * (bend * Lr * (uu * uu - 0.25));
  vec3 Ta = T0 + B0 * (bend * 2.0 * uu);
  // jet finger: the streakline of the fluid that left the jet with this parcel
  float sp = max(length(V), 0.4);
  float span = min(At.y / sp, max(age - 0.03, 0.0));
  float sg = s * span;
  vec3 Qs = P.xyz - V * sg - vec3(0.0, 0.5 * G_ACC, 0.0) * sg * sg;
  vec3 Ts = -(V + vec3(0.0, G_ACC * sg, 0.0));
  // jet finger: a curved thread along the parcel's motion, splaying away from the jet axis
  vec3 T1 = normalize(V + vec3(0.0, 0.0, 1e-3));
  vec3 B1 = normalize(cross(T1, vec3(0.0, 0.0, 1.0)) + vec3(0.0, 0.0, 0.3 * (hash11(seed * 7.7) - 0.5)));
  float bend1 = (hash11(seed * 3.1) - 0.5) * 0.7;
  float L1 = min(At.y * (0.6 + 0.6 * life), kd == K_JETL ? 0.35 : 0.3);
  float v1 = s - 0.3;
  vec3 Qj = P.xyz - T1 * (L1 * v1) + B1 * (bend1 * L1 * v1 * v1);
  vec3 Tj = -T1 + B1 * (bend1 * 2.0 * v1);
  vec3 Q = mix(Qj, Qa, arc);
  vec3 Tn = mix(Tj, Ta, arc);
  vec4 mv = viewMatrix * vec4(Q, 1.0);
  vec3 tv = mat3(viewMatrix) * Tn;
  vec2 td = length(tv.xy) > 1e-5 ? normalize(tv.xy) : vec2(0.0, 1.0);
  vec2 nd = vec2(-td.y, td.x);
  float pxm = -mv.z * uPxScaleFull;
  // beaded thread: thick in the middle, pinching towards the ends (Rayleigh-Plateau beads)
  float bead = 0.75 + 0.35 * sin(s * 17.0 + seed * 40.0);
  // (a finger: tapering from a bead at its head, 1.5 x the thread)
  float fing = mix(1.1, 0.4, s) * (1.0 + 0.5 * (1.0 - smoothstep(0.0, 0.14, s)));
  float wpx = At.x * mix(fing, (0.35 + 0.65 * sin(3.14159 * s)) * bead, arc) / pxm;
#if WW_EXPLORE
  // (seen from afar the thread width stands for a whole bundle of fine threads; resolved up close
  //  a single thread is 2-6 mm thick)
  wpx *= mix(1.0, 0.45, smoothstep(4.5e-3, 1.8e-3, pxm));
#endif
  float W = max(wpx, 0.55);
  float a = min(1.0, wpx / 0.55);
  a *= smoothstep(0.0, 0.03, life);
  vS = s * (1.0 - arc) + arc;
#if WW_EXPLORE
  a *= smoothstep(0.12, 0.35, -mv.z);    // soft near clip
#endif
  mv.xy += nd * aLig.y * (W + 1.0) * pxm;
  vW = vec2(aLig.y * (W + 1.0), W);
  vA = vec2(a, -mv.z);
  gl_Position = projectionMatrix * mv;
}`;

const LIG_FRAG = /* glsl */ OCCLUDE + `
in vec2 vW;
in vec2 vA;
in float vS;
void main() {
  float cov = clamp(vW.y + 0.5 - abs(vW.x), 0.0, 1.0);
  float side = vW.x / max(vW.y, 1e-3);
  // clear water thread: translucent body (it shows the water / sky behind it, tinted), a bright
  // specular line along the side that faces the sun
  float hl = smoothstep(0.1, -0.5, side) * smoothstep(-1.1, -0.6, side);
  float a = cov * vA.x * mix(0.6, 0.9, hl);
  if (a < 0.004) discard;
  a *= occlusion(vA.y);
  if (a < 0.004) discard;
  vec3 col = vec3(0.74, 0.87, 0.9) + vec3(0.25, 0.2, 0.18) * hl;
  // the drop at a finger's head catches the sun
  col += uSunColor * 0.35 * (1.0 - smoothstep(0.03, 0.1, vS)) * hl;
  // resolved threads: clear, glassy water - a faint tinted body, a narrow bright specular line and
  // darker refracting edges instead of an opaque white ribbon
  {
  float lensL = smoothstep(1.2, 2.6, vW.y);
  if (lensL > 0.0) {
    float hlN = smoothstep(0.1, -0.25, side) * smoothstep(-0.75, -0.45, side);
    float rimD = smoothstep(0.7, 0.98, abs(side));
    vec3 gc = mix(vec3(0.55, 0.74, 0.77), vec3(1.05, 1.02, 0.98), hlN) * (1.0 - 0.35 * rimD);
    col = mix(col, gc, lensL);
    a *= mix(1.0, mix(0.4, 1.0, max(hlN, 0.6 * rimD)) / mix(0.6, 0.9, hl), lensL);
  }
  }
  gl_FragColor = vec4(col * a, a);
}`;

// ---------------------------------------------------------------------------- class
export class Whitewater {
  constructor(renderer, shared, schedule) {
    this.renderer = renderer;
    this.schedule = schedule;
    this.shared = shared;
    const mk = () => {
      const rt = new THREE.WebGLRenderTarget(RES, RES, {
        count: 3, type: THREE.FloatType, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
        depthBuffer: false, stencilBuffer: false, generateMipmaps: false,
      });
      for (const t of rt.textures) t.colorSpace = THREE.NoColorSpace;
      return rt;
    };
    this.rt = [mk(), mk()];
    this.classSize = CLASS_SIZE;   // (debug / tools)
    this.cur = 0;
    this.head = [0, 0, 0];
    this.stepCount = 0;
    // per-emitter CDF over the bins (rows 0..NKIND-1) + the bin edges in world x (row NKIND)
    this.cdfData = new Float32Array((BINS + 1) * (NKIND + 1));
    this.cdfTex = new THREE.DataTexture(this.cdfData, BINS + 1, NKIND + 1, THREE.RedFormat, THREE.FloatType);
    this.fxs = 0;
    for (let i = 0; i <= BINS; i++) this.cdfData[NKIND * (BINS + 1) + i] = EDGES[i];
    this.cdfTex.minFilter = this.cdfTex.magFilter = THREE.NearestFilter;
    this.cdfTex.colorSpace = THREE.NoColorSpace;
    this.cdfTex.needsUpdate = true;
    this.counts = new Int32Array(NKIND);
    this.carry = new Float32Array(NKIND);
    this.rates = new Float32Array(NKIND);
    this.binRates = new Float32Array(BINS * NKIND);
    this.lod = new THREE.Vector4(0, 4.1, 1e6, LOD.P);   // clip mode: no LOD
    this.noise = makeFoamNoise(128);
    this.dbg = parseInt(new URLSearchParams(globalThis.location?.search || '').get('wwdbg') || '0', 10);
    // foam look: kappa / fexp: white-foam coverage 1 - exp(-(kappa tau)^fexp); kw: aerated water
    // (1 - exp(-kw tau_w)), wr/wg/wb its in-scatter colour; normalGain / cavity: lump (1-4 cm)
    // shading, nB / cavB: billows (9-22 cm); boost: extra sun on sun-facing caps
    this.params = { kappa: 0.86, fexp: 1.45, kw: 0.085, normalGain: 0.60, nB: 0.82, hSat: 0.043, cavity: 0.12, cavB: 0.27, nyMin: -0.5, skyDown: 0.45, cavMin: 0.55, bil: 1.0, bilCov: 0.95, bilCav: 0.22, bilH: 0.021, bilUp: 0.2, bilT: 0.5, baseH: 0.18, spN: 0.5, spB: 1.0, spC: 0.5, lumpN: 0.24, frontL: 0.06, frontT: 0.3, crev: 0.085, crevLift: 0.03, crevDz: 0.15, glint: 1.0, cell: 0.09, glassA: 0.20, rimK: 0.15, speck: 0.22, bufH: 960, detail: 1.05, sun: 1.12, amb: 1.08, boost: 0.12, shadow: 0.45, ar: 0.33, ag: 0.35, ab: 0.375, alr: 0.9, alg: 0.915, alb: 0.94, wr: 0.36, wg: 0.46, wb: 0.46, dcol: 0.16, glass: 0.7, sph: 0.25, bufS: 0.5, hov: 3.5, hw0: 0.5, hw1: 0.55, hw2: 0.85, hlife: 0.9 };
    // debug / tuning overrides from the URL: ?ww=kappa:1.2,sun:0.7
    const ov = new URLSearchParams(globalThis.location?.search || '').get('ww');
    if (ov) for (const kv of ov.split(',')) { const [k, v] = kv.split(':'); if (k in this.params) this.params[k] = parseFloat(v); }

    const prelude = PRELUDE(false);
    const farU = {};
    if (EXPLORE) {
      // far-field water state over the emission window (see FARBAKE)
      this.far = new THREE.WebGLRenderTarget(FAR_NX, FAR_NZ, {
        count: 2, type: THREE.HalfFloatType, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
        depthBuffer: false, stencilBuffer: false, generateMipmaps: false,
      });
      for (const t of this.far.textures) t.colorSpace = THREE.NoColorSpace;
      this.farDom = new THREE.Vector4(-FAR_W / 2, FAR_Z0, FAR_W, FAR_Z1 - FAR_Z0);
      Object.assign(farU, { uFarA: { value: this.far.textures[0] }, uFarB: { value: this.far.textures[1] }, uFarDom: { value: this.farDom } });
      this.pFar = new FullscreenPass(makeShader(prelude + FARBAKE, { ...shared, uFarDom: { value: this.farDom } }));
    }
    this.pUpdate = new FullscreenPass(makeShader(prelude + UPDATE, {
      ...shared, ...farU,
      uPos: { value: null }, uVel: { value: null }, uAttr: { value: null }, uCdf: { value: this.cdfTex }, uLod: { value: this.lod }, uSpillHov: { value: this.params.hov }, uSpillWin: { value: new THREE.Vector4(this.params.hw0, this.params.hw1, this.params.hw2, this.params.hlife) },
      uDtS: { value: 0 }, uHead: { value: [0, 0, 0] }, uCount: { value: Array.from(this.counts) }, uStepSeed: { value: 0 },
    }));

    const state = { uPos: { value: null }, uVel: { value: null }, uAttr: { value: null } };
    this.state = state;
    const camRot = { value: new THREE.Matrix3() };
    const pxScale = { value: 1 }, pxScaleFull = { value: 1 };
    const sceneDepth = { value: null };
    this.u = { camRot, pxScale, pxScaleFull, sceneDepth };

    // --- blob pass (optical depth, additive)
    const hideMask = { value: parseInt(new URLSearchParams(globalThis.location?.search || '').get('wwhide') || '0', 10) };
    const quad = new THREE.InstancedBufferGeometry();
    quad.setAttribute('aCorner', new THREE.Float32BufferAttribute([-1, -1, 1, -1, 1, 1, -1, 1], 2));
    quad.setIndex([0, 1, 2, 0, 2, 3]);
    quad.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5);
    const blobGeo = quad.clone();
    blobGeo.instanceCount = CLASS_SIZE[0];
    blobGeo.boundingSphere = quad.boundingSphere;
    this.blobMat = new THREE.ShaderMaterial({
      uniforms: {
        ...shared, ...state, uSceneDepth: sceneDepth, uNoise: { value: this.noise },
        uBufSize: { value: new THREE.Vector2(1, 1) }, uBufH: { value: 1 }, uCamRot: camRot, uPxScale: pxScale, uPxScaleB: pxScale, uFrontSph: { value: this.params.sph },
        uFrontZ: { value: null }, uFrontL: { value: new THREE.Vector2(0.06, 0.3) }, uBaseH: { value: 0.18 },
        uHideMask: hideMask, ...(EXPLORE ? { uFarB: farU.uFarB, uFarDom: farU.uFarDom } : {}),
      },
      vertexShader: prelude + BLOB_VERT,
      fragmentShader: prelude + BLOB_FRAG,
      transparent: true, depthTest: false, depthWrite: false,
      blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneFactor,
    });
    this.blobScene = new THREE.Scene();
    const bm = new THREE.Mesh(blobGeo, this.blobMat); bm.frustumCulled = false; this.blobScene.add(bm);
    // front pre-pass: 1 / depth of the front of the dense white foam (MAX), same parcels
    const preludeZ = PRELUDE(true);
    this.blobZMat = new THREE.ShaderMaterial({
      uniforms: this.blobMat.uniforms,
      vertexShader: preludeZ + BLOB_VERT,
      fragmentShader: preludeZ + BLOB_FRAG,
      transparent: true, depthTest: false, depthWrite: false,
      blending: THREE.CustomBlending, blendEquation: THREE.MaxEquation,
      blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneFactor,
    });
    this.blobZScene = new THREE.Scene();
    const bz = new THREE.Mesh(blobGeo, this.blobZMat); bz.frustumCulled = false; this.blobZScene.add(bz);

    // --- resolve
    this.pResolve = new FullscreenPass(makeShader(prelude + RESOLVE, {
      ...shared, uThick: { value: null }, uDetail: { value: null }, uLump: { value: null }, uFrontZ: { value: null }, uDetCol: { value: 0.06 }, uGlass: { value: 0 }, uSceneDepth: sceneDepth, uNoise: { value: this.noise },
      uCrev: { value: new THREE.Vector4() }, uLaceK: { value: new THREE.Vector4() },
      uFoamK2: { value: new THREE.Vector4() }, uFoamK3: { value: new THREE.Vector4() }, uBil: { value: new THREE.Vector4() }, uBilUp: { value: 0 }, uBilT: { value: 0.5 }, uSplashK: { value: new THREE.Vector3() }, uViewSize: { value: new THREE.Vector2(1, 1) }, uWaterCol: { value: new THREE.Vector3() }, uAlbedo: { value: new THREE.Vector3() },
      uCamPos: { value: new THREE.Vector3() }, uTanFov: { value: new THREE.Vector2() },
      uSunScreen: { value: new THREE.Vector2(0, 1) }, uShadow: { value: 0.55 }, uThickTexel: { value: new THREE.Vector2() },
      uAmbCol: { value: new THREE.Vector3() }, uPxFull: pxScaleFull, uLaceAx: { value: new THREE.Vector4(1, 0, 0, 1) },
      uPxScale: pxScale, uCamRot: camRot, uFoamK: { value: new THREE.Vector4() }, uLight: { value: new THREE.Vector4() },
      uDbg: { value: this.dbg },
    }));

    // --- crisp pass: droplets + ligaments (premultiplied over)
    const over = {
      transparent: true, depthTest: false, depthWrite: false,
      blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
    };
    const occl = { uThick: { value: null }, uViewport: { value: new THREE.Vector2(1, 1) }, uOccK: { value: new THREE.Vector3() } };
    this.occl = occl;
    const dropGeo = quad.clone();
    dropGeo.instanceCount = CLASS_SIZE[1];
    dropGeo.boundingSphere = quad.boundingSphere;
    this.dropMat = new THREE.ShaderMaterial({
      uniforms: { ...shared, ...state, ...occl, uSceneDepth: sceneDepth, uBase: { value: CLASS_BASE[1] }, uPxScaleFull: pxScaleFull, uHideMask: hideMask, uSunScr: { value: new THREE.Vector2(0, 1) } },
      vertexShader: prelude + DROP_VERT, fragmentShader: prelude + DROP_FRAG, ...over,
    });
    const ligGeo = new THREE.InstancedBufferGeometry();
    const la = [], li = [];
    for (let i = 0; i <= LIG_SEGS; i++) { la.push(i / LIG_SEGS, -1, i / LIG_SEGS, 1); }
    for (let i = 0; i < LIG_SEGS; i++) { const a = i * 2; li.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    ligGeo.setAttribute('aLig', new THREE.Float32BufferAttribute(la, 2));
    ligGeo.setIndex(li);
    ligGeo.instanceCount = CLASS_SIZE[2];
    ligGeo.boundingSphere = quad.boundingSphere;
    this.ligMat = new THREE.ShaderMaterial({
      uniforms: { ...shared, ...state, ...occl, uSceneDepth: sceneDepth, uBase: { value: CLASS_BASE[2] }, uPxScaleFull: pxScaleFull, uHideMask: hideMask },
      vertexShader: prelude + LIG_VERT, fragmentShader: prelude + LIG_FRAG, ...over, side: THREE.DoubleSide,
    });
    this.crispScene = new THREE.Scene();
    const dm = new THREE.Mesh(dropGeo, this.dropMat); dm.frustumCulled = false;
    const lm = new THREE.Mesh(ligGeo, this.ligMat); lm.frustumCulled = false;
    this.crispScene.add(dm, lm);

    this.thick = null;
    this.reset();
  }

  reset() {
    const r = this.renderer;
    const prev = r.getClearAlpha();
    const prevC = new THREE.Color(); r.getClearColor(prevC);
    // all channels start "dead": pos.w (age) = 99
    r.setClearColor(new THREE.Color(0, 0, 0), 99);
    for (const rt of this.rt) { r.setRenderTarget(rt); r.clear(true, false, false); }
    r.setClearColor(prevC, prev);
    this.head = [0, 0, 0]; this.stepCount = 0; this.carry.fill(0);
    this._publish();
  }

  _emission(t, dt) {
    const events = this.schedule.active;
    const R = this.binRates;
    R.fill(0);
    const tmp = this.rates;
    // explore mode: the bins follow the player (snapped), the emission density falls off with
    // the distance from the player (LOD; the GPU enlarges the far parcels to match)
    let fxs = 0, fx = 0, fz = 4.1;
    if (EXPLORE) {
      const f = this.shared.uFocus.value;
      fx = f.x; fz = f.y;
      fxs = Math.round(fx / FOCUS_SNAP) * FOCUS_SNAP;
      this.lod.set(fx, fz, LOD.D0, LOD.P);
      if (fxs !== this.fxs) {
        this.fxs = fxs;
        for (let i = 0; i <= BINS; i++) this.cdfData[NKIND * (BINS + 1) + i] = fxs + EDGES[i];
      }
    }
    for (let i = 0; i < BINS; i++) {
      const x = fxs + 0.5 * (EDGES[i] + EDGES[i + 1]);
      const bw = (EDGES[i + 1] - EDGES[i]) * (EXPLORE ? lodAt(x, fx, fz) * exploreEdgeFade(x - fx) : 1);
      tmp.fill(0);
      for (const e of events) emitterRates(e, x, t, tmp);
      tmp[K.BORE] += BORE_BG;
      for (let k = 0; k < NKIND; k++) R[k * BINS + i] = tmp[k] * bw;
    }
    const totals = [0, 0, 0];
    const W1 = BINS + 1;
    for (let k = 0; k < NKIND; k++) {
      let sum = 0;
      for (let i = 0; i < BINS; i++) { sum += R[k * BINS + i]; this.cdfData[k * W1 + i] = sum; }
      for (let i = 0; i < BINS; i++) this.cdfData[k * W1 + i] = sum > 0 ? this.cdfData[k * W1 + i] / sum : (i + 1) / BINS;
      const want = sum * dt + this.carry[k];
      const n = Math.floor(want);
      this.carry[k] = want - n;
      this.counts[k] = n;
    }
    CLASS_KINDS.forEach(([a, b], c) => {
      let tot = 0;
      for (let k = a; k < b; k++) tot += this.counts[k];
      const cap = Math.floor(CLASS_SIZE[c] / 5);
      if (tot > cap) {
        const s = cap / tot; tot = 0;
        for (let k = a; k < b; k++) { this.counts[k] = Math.floor(this.counts[k] * s); tot += this.counts[k]; }
      }
      totals[c] = tot;
    });
    this.cdfTex.needsUpdate = true;
    return totals;
  }

  step(t, dt) {
    const totals = this._emission(t, dt);
    if (EXPLORE) {
      // far-field water state around the (snapped) focus for this step
      this.farDom.set(this.fxs - FAR_W / 2, FAR_Z0, FAR_W, FAR_Z1 - FAR_Z0);
      this.pFar.render(this.renderer, this.far);
    }
    const u = this.pUpdate.material.uniforms;
    const src = this.rt[this.cur], dst = this.rt[1 - this.cur];
    u.uPos.value = src.textures[0];
    u.uVel.value = src.textures[1];
    u.uAttr.value = src.textures[2];
    u.uDtS.value = dt;
    u.uHead.value = this.head.slice();
    u.uCount.value = Array.from(this.counts);
    u.uStepSeed.value = (this.stepCount % 9973) * 0.6180339 + 0.123;
    this.pUpdate.render(this.renderer, dst);
    this.cur = 1 - this.cur;
    for (let c = 0; c < 3; c++) this.head[c] = (this.head[c] + totals[c]) % CLASS_SIZE[c];
    this.stepCount++;
    this._publish();
  }

  _publish() {
    const rt = this.rt[this.cur];
    this.state.uPos.value = rt.textures[0];
    this.state.uVel.value = rt.textures[1];
    this.state.uAttr.value = rt.textures[2];
  }

  _targets(w, h) {
    // (clip: half-res buffers (bufH 960 of 1920); explore: bufS = 1/2 of the output resolution - the
    //  full-res resolve reconstructs the detail, the blob passes cost ~4x less fill)
    const s = Math.min(1, this.params.bufH / h, EXPLORE ? this.params.bufS : 1);
    const bw = Math.max(1, Math.round(w * s)), bh = Math.max(1, Math.round(h * s));
    if (this.thick && this.thick.width === bw && this.thick.height === bh) return;
    this.thick?.dispose();
    this.frontZ?.dispose();
    // (tau foam, tau water, tau z, tau lump) | front-weighted (detail, weight) + tearing | lump normal
    this.thick = new THREE.WebGLRenderTarget(bw, bh, {
      count: 3, type: THREE.HalfFloatType, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
      depthBuffer: false, stencilBuffer: false, generateMipmaps: false,
    });
    for (const t of this.thick.textures) t.colorSpace = THREE.NoColorSpace;
    // the optical depth is mip-mapped: the resolve shades the 10-40 cm billows from its mip chain
    const t0 = this.thick.textures[0];
    t0.generateMipmaps = true;
    t0.minFilter = THREE.LinearMipmapLinearFilter;
    this.thick.textures[2].format = THREE.RGFormat;
    // front pre-pass: 1 / depth of the dense foam (MAX blended)
    this.frontZ = new THREE.WebGLRenderTarget(bw, bh, {
      type: THREE.HalfFloatType, format: THREE.RedFormat, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
      depthBuffer: false, stencilBuffer: false, generateMipmaps: false,
    });
    this.frontZ.texture.colorSpace = THREE.NoColorSpace;
  }

  // debug: GPU time per stage (window.__scene.whitewater.stageTimes = {} to enable)
  _tq(name) {
    if (!this.stageTimes) return;
    const gl = this.renderer.getContext();
    const ext = this._tqExt || (this._tqExt = gl.getExtension('EXT_disjoint_timer_query_webgl2'));
    if (!ext) return;
    if (this._tqOpen) { gl.endQuery(ext.TIME_ELAPSED_EXT); (this._tqList ||= []).push(this._tqOpen); this._tqOpen = null; }
    if (name) { const q = gl.createQuery(); gl.beginQuery(ext.TIME_ELAPSED_EXT, q); this._tqOpen = [name, q]; }
  }
  // GPU ms per stage, averaged over the frames whose results are available (the others stay queued
  // for the next call: read it a few hundred ms after the frames were rendered)
  stageReport() {
    const gl = this.renderer.getContext(), out = {}, pending = [];
    for (const [n, q] of this._tqList || []) {
      if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) { pending.push([n, q]); continue; }
      (out[n] ||= []).push(gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6);
      gl.deleteQuery(q);
    }
    this._tqList = pending;
    for (const k in out) out[k] = +(out[k].reduce((a, b) => a + b, 0) / out[k].length).toFixed(3);
    return out;
  }

  render(camera, target, sceneDepth, w, h) {
    const r = this.renderer;
    this._targets(w, h);
    const bw = this.thick.width, bh = this.thick.height;
    const tanH = Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5);
    this.u.pxScale.value = (2 * tanH) / bh;
    this.u.pxScaleFull.value = (2 * tanH) / h;
    this.u.camRot.value.setFromMatrix4(camera.matrixWorld);
    this.u.sceneDepth.value = sceneDepth;
    this.blobMat.uniforms.uBufSize.value.set(bw, bh);
    this.blobMat.uniforms.uBufH.value = bh;
    const prevA = r.getClearAlpha();
    const P = this.params;
    this.blobMat.uniforms.uFrontSph.value = P.sph;
    this.blobMat.uniforms.uFrontL.value.set(P.frontL, P.frontT);
    this.blobMat.uniforms.uBaseH.value = P.baseH;
    r.setClearColor(0x000000, 0);
    // 1) front of the dense foam (pre-pass), 2) optical depth of the foam mass + its front lumps
    this._tq('front');
    r.setRenderTarget(this.frontZ);
    r.clear(true, false, false);
    this.blobMat.uniforms.uFrontZ.value = null;
    if (this.dbg !== 4) r.render(this.blobZScene, camera);
    this._tq('blobs');
    this.blobMat.uniforms.uFrontZ.value = this.frontZ.texture;
    r.setRenderTarget(this.thick);
    r.clear(true, false, false);
    if (this.dbg !== 4) r.render(this.blobScene, camera);
    // 3) resolve into the particle target
    const ru = this.pResolve.material.uniforms;
    ru.uThick.value = this.thick.textures[0];
    ru.uDetail.value = this.thick.textures[1];
    ru.uLump.value = this.thick.textures[2];
    ru.uFrontZ.value = this.frontZ.texture;
    ru.uCrev.value.set(P.crev, P.crevLift, P.crevDz, P.glint);
    ru.uLaceK.value.set(P.cell, P.glassA, P.rimK, P.speck);
    ru.uThickTexel.value.set(1 / bw, 1 / bh);
    ru.uFoamK.value.set(P.kappa, P.normalGain, P.hSat, P.cavity);
    ru.uFoamK2.value.set(P.fexp, P.kw, P.nB, P.cavB);
    ru.uFoamK3.value.set(P.nyMin, P.skyDown, P.cavMin, P.lumpN);
    ru.uBil.value.set(P.bil, P.bilCov, P.bilCav, P.bilH);
    ru.uBilUp.value = P.bilUp;
    ru.uBilT.value = P.bilT;
    ru.uSplashK.value.set(P.spN, P.spB, P.spC);
    ru.uViewSize.value.set(w, h);
    ru.uWaterCol.value.set(P.wr, P.wg, P.wb);
    ru.uAlbedo.value.set(P.alr, P.alg, P.alb);
    this.occl.uOccK.value.set(P.kappa, P.fexp, P.kw);
    ru.uLight.value.set(P.sun, P.amb, P.boost, P.detail);
    ru.uDetCol.value = P.dcol;
    ru.uGlass.value = P.glass;
    ru.uCamPos.value.setFromMatrixPosition(camera.matrixWorld);
    ru.uTanFov.value.set(tanH * camera.aspect, tanH);
    // project the sun direction to a screen-space direction (view space x/y)
    const sd = this.shared.uSunDir.value.clone().transformDirection(camera.matrixWorldInverse);
    // (looking straight at / away from the sun: no screen direction -> straight up)
    if (Math.hypot(sd.x, sd.y) > 1e-4) ru.uSunScreen.value.set(sd.x, sd.y).normalize(); else ru.uSunScreen.value.set(0, 1);
    this.dropMat.uniforms.uSunScr.value.copy(ru.uSunScreen.value);
    // Fixed along-shore axes. Looking around never rotates or regenerates lace.
    ru.uLaceAx.value.set(1,0,0,1);
    ru.uShadow.value = P.shadow;
    ru.uAmbCol.value.set(P.ar, P.ag, P.ab);
    this._tq('resolve');
    this.pResolve.render(r, target);
    // 3) droplets + ligaments over it
    this.occl.uThick.value = this.thick.textures[0];
    this.occl.uViewport.value.set(w, h);
    r.setRenderTarget(target);
    this._tq('crisp');
    if (this.dbg !== 5) r.render(this.crispScene, camera);
    this._tq(null);
    r.setClearColor(0x000000, prevA);
  }
}
