import * as THREE from 'three';

// One-time GPU bake of a periodic pebble-beach tile (swash analysis §6): a packed gravel mosaic
// (matrix grains ~9.5 mm on a 9 mm pitch, grit ~4 mm on a 4.2 mm pitch: the grains overlap, so
// what shows are 2-6 mm chip faces with only ~15 % wet grit between them), a few coarse granules,
// and scattered flat, well-rounded pebbles 18-55 mm (~60 /m^2), a few pale / pierced / pitted ones.
// Sizes and contrast were fitted to the 1080p frame bottom of the clip (luminance ACF e-fold
// 1.9 px vs 1.96, band-pass spectrum, luma percentiles). Small grains are angular chips with a
// crease (two facets catch the sun differently) and a bright crown / dark wet flank, grain faces
// carry a 1 mm mineral speckle (mica / quartz specks, pits), the darkest grains stop at a dark
// brown-grey (no black pinholes), and the tile is 2x2 supersampled so grain edges stay crisp
// without stair-stepping.
// Palette = the measured wet classes (colour analysis §5), converted to pre-grade
// scene-linear albedo and calibrated on renders against the clip's frame-bottom statistics.
// Chroma: small grains are pooled toward the mean tint (the video's chroma is far coarser than a
// grain); pink / ochre pebbles keep theirs; 2-6 cm patches of pinker / greyer gravel carry the
// clip's low-frequency chroma variation (the runtime takes its chroma from the local mean albedo).
// All loops have dynamic trip counts (uZero) so FXC compiles the bake quickly.
//
// Outputs (all periodic over TILE metres, mipmapped):
//   A  RGBA16F : wet albedo (linear, pre-grade), height above the gravel datum (m)
//   B  RGBA8   : tile-space normal (xyz * 0.5 + 0.5), ambient occlusion
//   C  RGBA8   : roughness, crown mask (wet skin on stone tops), pebble flag, stone random

export const TILE = 1.2;        // tile size (m)
export const TILE_RES = 2048;   // texels per side (0.59 mm / texel)

const SAT = 1.35;               // post saturation (the palette is authored pre-grade)
// Effective radiance / albedo of the gravel in our light rig, measured on renders (grain
// slopes, AO and sun shadows included): calibrates the class medians to the frame-bottom
// statistics of the reference (pre-grade mean (0.215, 0.174, 0.149) at t = 8 s).
const E_EFF = 0.71;
const TINT = [1.02, 1.0, 0.935];  // the sky-ambient share of the rig is cooler than the sun

const srgbToLin = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
const lumOf = (l) => 0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2];
function wetAlbedo(srgb) {
  const l = srgb.map(srgbToLin);
  const Y = lumOf(l);
  return l.map((c, i) => Math.max((Y + (c - Y) / SAT) / E_EFF * TINT[i], 0));
}

// wet sRGB classes and weights (swash analysis §6.2). pool: how much of the class chroma is
// pulled toward the mean tint (video chroma is soft), the pink / ochre stones keep theirs.
export const CLASSES = [
  { name: 'brown-grey', srgb: [123, 105, 96], w: 0.62, rough: 0.20, pool: 0.45 },
  { name: 'dark', srgb: [72, 60, 52], w: 0.14, rough: 0.18, pool: 0.45 },
  { name: 'cool-grey', srgb: [116, 106, 100], w: 0.09, rough: 0.22, pool: 0.35 },
  { name: 'quartz', srgb: [196, 178, 168], w: 0.07, rough: 0.14, pool: 0.45 },
  { name: 'salmon', srgb: [162, 120, 104], w: 0.06, rough: 0.20, pool: 0.0 },
  { name: 'ochre', srgb: [142, 108, 74], w: 0.02, rough: 0.24, pool: 0.05 },
];

const f3 = (v) => `vec3(${v.map((x) => x.toFixed(4)).join(', ')})`;

// class weights of the fine gravel grains (more pale quartz/limestone specks and dark grains
// than among the pebbles: the salt-and-pepper look of the matrix)
export const GRAIN_WEIGHTS = [0.52, 0.16, 0.10, 0.10, 0.09, 0.03];
// the finest grit carries most of the pale quartz / limestone specks
export const GRIT_WEIGHTS = [0.42, 0.17, 0.08, 0.22, 0.08, 0.03];

/** chroma of the dominant (brown-grey) class relative to its luminance: vec3 with lum 1 */
export function meanTint(tinted = true) {
  const base = tinted ? wetAlbedo(CLASSES[0].srgb) : wetAlbedo(CLASSES[0].srgb).map((c, i) => c / TINT[i]);
  return base.map((c) => c / lumOf(base));
}

const cum = (w) => { let a = 0; const s = w.reduce((x, y) => x + y, 0); return w.slice(0, 5).map((x) => (a += x / s)); };

// Bake parameters (uniforms, so the tile can be re-baked while tuning: Beach.rebake()).
// Size classes (one call site, looped): packed parameters per layer
//   L0: grid pitch (m), seed, kind, occupancy       L1: D median (m), log sigma, D min, D max
//   L2: b/a min, max, half-thickness/a min, max     L3: top height min, max (m), profile exponent, tilt
//   L4: angularity (0 ellipse .. 1 random hexagon), crease slope (two facets)
// K: x pebble prominence (top += K.x * a), y mineral speck gain, z grain chroma pooling scale,
//    w pebble chroma pooling scale
export const BAKE_DEFAULTS = {
  // r4: fewer, larger and more distinct pebbles (~60 /m^2, 18-55 mm, luminance jitter 0.21 log,
  // class colours kept), few granules (the grey 'marbles'), slightly flatter chips, softer
  // crown / rim shading, fewer mineral specks (the 1080 frame bottom of the clip, t = 4.8 / 8.1 s)
  L0: [[0.0042, 11.0, 1.0, 0.95], [0.0090, 23.0, 2.0, 0.97], [0.0190, 37.0, 3.0, 0.08], [0.080, 53.0, 4.0, 0.51]],
  L1: [[0.0042, 0.28, 0.0024, 0.0070], [0.0095, 0.30, 0.0055, 0.0170], [0.0150, 0.22, 0.0110, 0.0240], [0.0320, 0.36, 0.0180, 0.0650]],
  L2: [[0.60, 1.0, 0.45, 0.65], [0.58, 0.95, 0.40, 0.60], [0.50, 0.90, 0.33, 0.48], [0.55, 0.88, 0.38, 0.50]],
  L3: [[-0.0016, 0.0000, 2.3, 0.40], [-0.0005, 0.0018, 2.5, 0.40], [0.0006, 0.0026, 2.5, 0.28], [0.0024, 0.0046, 2.3, 0.10]],
  L4: [[0.60, 0.30], [0.50, 0.35], [0.35, 0.20], [0.0, 0.0]],
  K: [0.11, 0.48, 0.85, 0.22],
  K2: [0.6, 1.0, 1.0, 1.4],
  pebW: CLASSES.map((c) => c.w),
  grainW: GRAIN_WEIGHTS,
  gritW: GRIT_WEIGHTS,
};

/** uniforms of the stone bake for a parameter set (missing keys: defaults) */
export function bakeUniforms(p = {}, u = null) {
  const q = { ...BAKE_DEFAULTS, ...p };
  const v4 = (a) => a.map((x) => new THREE.Vector4(x[0], x[1], x[2] ?? 0, x[3] ?? 0));
  const vals = {
    uL0: v4(q.L0), uL1: v4(q.L1), uL2: v4(q.L2), uL3: v4(q.L3), uL4: v4(q.L4),
    uBakeK: new THREE.Vector4(...q.K), uBakeK2: new THREE.Vector4(...q.K2),
    uPebW: cum(q.pebW), uGrainW: cum(q.grainW), uGritW: cum(q.gritW),
  };
  if (!u) { u = {}; for (const k in vals) u[k] = { value: vals[k] }; return u; }
  for (const k in vals) u[k].value = vals[k];
  return u;
}

function paletteGLSL() {
  const lines = CLASSES.map((c, i) => `  if (i == ${i}) { rough = ${c.rough.toFixed(3)}; pool = ${c.pool.toFixed(3)}; return ${f3(wetAlbedo(c.srgb))}; }`);
  const base = wetAlbedo(CLASSES[0].srgb);
  const tint = base.map((c) => c / lumOf(base));
  return `
const vec3 MEAN_TINT = ${f3(tint)};
uniform float uPebW[5], uGrainW[5], uGritW[5];
float pickClass(float u, int set) {
  for (int i = 0; i < 5; i++) {
    float w = set == 0 ? uPebW[i] : (set == 1 ? uGrainW[i] : uGritW[i]);
    if (u < w) return float(i);
  }
  return 5.0;
}
vec3 classAlbedo(int i, out float rough, out float pool) {
${lines.join('\n')}
  pool = 0.0;
  return vec3(0.0);
}`;
}

const COMMON = /* glsl */ `
#define TILE ${TILE.toFixed(4)}
#define RES ${TILE_RES}.0
uvec3 pcg3d(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
vec3 rnd3(vec2 c, int s) { return vec3(pcg3d(uvec3(uint(c.x), uint(c.y), uint(s + 7)))) * (1.0 / 4294967296.0); }
float gauss(vec2 u) { return sqrt(-2.0 * log(max(u.x, 1e-6))) * cos(6.2831853 * u.y); }
// periodic value noise: n lattice cells per tile
float pnoise(vec2 p, float n, int s) {
  vec2 q = p * n; vec2 i = floor(q); vec2 f = q - i;
  f = f * f * (3.0 - 2.0 * f);
  float a = rnd3(mod(i, n), s).x, b = rnd3(mod(i + vec2(1, 0), n), s).x;
  float c = rnd3(mod(i + vec2(0, 1), n), s).x, d = rnd3(mod(i + vec2(1, 1), n), s).x;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
// non-periodic value noise for stone-local detail
float lnoise(vec2 p, int s) {
  vec2 i = floor(p); vec2 f = p - i;
  f = f * f * (3.0 - 2.0 * f);
  vec2 o = vec2(4096.0);
  float a = rnd3(i + o, s).x, b = rnd3(i + o + vec2(1, 0), s).x;
  float c = rnd3(i + o + vec2(0, 1), s).x, d = rnd3(i + o + vec2(1, 1), s).x;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
`;

// ------------------------------------------------------------------------ pass 1
export const BAKE_STONES = COMMON + paletteGLSL() + /* glsl */ `
uniform int uZero;   // 0: dynamic loop bounds keep FXC from unrolling the bake (fast compile)
in vec2 vUv;
layout(location = 1) out vec4 outC;

// winner of the height test so far
float gH;        // top height (m)
vec3 gR0, gR1;   // its random numbers
vec2 gLoc;       // fragment position in the stone frame (m)
float gA;        // semi-major axis (m)
float gRR;       // normalised radius 0..1
float gKind;     // 0 base grit, 1 grit grain, 2 matrix grain, 3 granule, 4 pebble

// Size classes (BAKE_DEFAULTS above). Packed gravel: grain diameters exceed the grid pitch, so
// the grains overlap into a mosaic and only ~15 % of the surface shows the wet grit between them.
uniform vec4 uL0[4], uL1[4], uL2[4], uL3[4], uL4[4];
uniform vec4 uBakeK, uBakeK2;

// One size class on a periodic jittered grid of n cells per tile.
void layer(vec2 p, float n, int seed, float kind, vec2 dMed_dSig, vec2 dRange, float occ,
           vec2 asp, vec2 thk, vec2 zr, float pr, float tilt, float ang, float ridge) {
  float cs = TILE / n;
  vec2 q = p * n;
  vec2 i0 = floor(q);
  vec2 f = q - i0;
  for (int k9 = 0; k9 < 9 + uZero; k9++) {
    int x = k9 % 3 - 1, y = k9 / 3 - 1;
    vec2 c = mod(i0 + vec2(x, y), n);
    vec3 r0 = rnd3(c, seed), r1 = rnd3(c, seed + 101), r2 = rnd3(c, seed + 202);
    if (r2.x > occ) continue;
    vec2 ctr = vec2(x, y) + 0.15 + 0.7 * r0.xy;
    vec2 d = (f - ctr) * cs;
    float D = clamp(dMed_dSig.x * exp(dMed_dSig.y * gauss(r1.xy)), dRange.x, dRange.y);
    float a = 0.5 * D;
    if (length(d) > a * 1.45) continue;
    float ar = mix(asp.x, asp.y, r0.z);
    if (kind > 3.5 && r2.z < 0.08) ar *= 0.62;       // a few elongated pebbles
    float an = r1.z * 6.2831853;
    float ca = cos(an), sa = sin(an);
    vec2 l = vec2(ca * d.x + sa * d.y, -sa * d.x + ca * d.y);
    vec2 e = vec2(l.x / a, l.y / (a * ar));
    float th = atan(e.y, e.x);
    float wob = 1.0 + 0.07 * r2.z * sin(2.0 * th + r2.y * 31.0) + 0.06 * sin(3.0 * th + r2.z * 17.0)
              + 0.035 * sin(5.0 * th + r0.x * 23.0) + (kind < 3.5 ? 0.03 * sin(7.0 * th + r1.y * 11.0) : 0.0);
    float re = length(e);
    if (ang > 0.0) {
      // chipped outline: a random convex hexagon blended with the ellipse
      float pd = 0.0, ph = r2.y * 6.2831853;
      for (int k = 0; k < 6 + uZero; k++) {
        float t = ph + float(k) * 1.0471976 + 0.5 * (fract(r0.x * 13.7 * float(k + 1)) - 0.5);
        pd = max(pd, dot(e, vec2(cos(t), sin(t))));
      }
      re = mix(re, pd * 1.08, ang);
    }
    float rr = re / wob;
    if (rr >= 1.0) continue;
    // pierced pebbles (swash analysis: a 25 mm grey disc with a hole)
    if (kind > 3.5 && r0.y > 0.972 && D > 0.02 && length(e - vec2(0.12, -0.08) * (r1.x - 0.5)) < mix(0.17, 0.27, r1.y)) continue;
    float half_ = a * mix(thk.x, thk.y, r2.y);                   // half thickness
    float top = mix(zr.x, zr.y, fract(r2.z * 7.13 + r0.x * 3.1));
    if (kind > 3.5) top += uBakeK.x * a;                        // larger pebbles stand a bit prouder
    float prof = pow(max(1.0 - pow(rr, pr), 0.0), 1.0 / pr);
    // grains lie at random tilts: flat chips catch the sun differently (the 'crunchy' look)
    vec2 tv = (vec2(r0.z, r2.y) * 2.0 - 1.0) * tilt;
    // crease: a ridge across the chip splits it into two facets (lit / shaded side)
    float rt = r0.y * 6.2831853;
    float cr = abs(dot(l, vec2(cos(rt), sin(rt))) - (r1.x - 0.5) * 0.6 * a);
    float h = top - half_ + half_ * prof + dot(tv, l) * prof - ridge * cr * prof;
    if (h > gH) { gH = h; gR0 = r0; gR1 = r1; gLoc = l; gA = a; gRR = rr; gKind = kind; }
  }
}

// full surface at tile point p (0..1): albedo, height, and the C channels
void surface(vec2 p, out vec4 A, out vec4 C) {
  // base: dark wet grit / coarse sand between the stones
  gH = -0.0020 + 0.0006 * pnoise(p, 640.0, 5) + 0.0003 * pnoise(p, 1280.0, 6);
  gKind = 0.0; gR0 = gR1 = vec3(0.0); gLoc = vec2(0.0); gA = 0.001; gRR = 1.0;
  for (int i = 0; i < 4 + uZero; i++) {
    vec4 a0 = uL0[i], a1 = uL1[i], a2 = uL2[i], a3 = uL3[i]; vec2 a4 = uL4[i].xy;
    layer(p, floor(TILE / a0.x + 0.5), int(a0.y), a0.z, a1.xy, a1.zw, a0.w, a2.xy, a2.zw, a3.xy, a3.z, a3.w, a4.x, a4.y);
  }

  vec3 alb; float rough, cls, pool;
  float crown = 0.0, rnd = 0.0;
  if (gKind < 0.5) {
    // interstitial grit and sand: dark (wet, self-shadowed), finely speckled with pale sand
    float s = pnoise(p, 2560.0, 9);
    float sp = step(0.86, pnoise(p, 1900.0, 10));
    alb = vec3(0.092, 0.075, 0.064) * (0.75 + 0.5 * s) + vec3(0.10, 0.09, 0.08) * sp;
    rough = 0.55;
  } else {
    bool peb = gKind > 3.5;
    // class draw: measured pebble weights; gravel grains have more pale (quartz) and dark ones
    cls = pickClass(gR0.x * 0.999, peb ? 0 : (gKind < 1.5 ? 2 : 1));
    alb = classAlbedo(int(cls), rough, pool);
    // per-stone luminance jitter, log-normal (sigma 0.15 pebbles, more for the small grains)
    float sig = peb ? 0.15 * uBakeK2.w : (gKind > 2.5 ? 0.28 : 0.46) * uBakeK2.z;
    alb *= exp(sig * clamp(gauss(gR1.yx), -1.7, 2.2));   // no pitch-black outliers
    // hue jitter: warm-pink <-> grey axis (+-4 deg)
    alb *= 1.0 + 0.07 * (gR1.z - 0.5) * vec3(1.0, -0.15, -0.8);
    // inside the stone: 1 mm mineral speckle, blotches; pitted / veined pebbles
    float sp = lnoise(gLoc / 0.0006 + gR0.yz * 900.0, 3);
    float sp2 = lnoise(gLoc / 0.0011 + gR1.zx * 700.0, 5);
    float bl = lnoise(gLoc / 0.004 + gR1.xy * 300.0, 4);
    alb *= 0.92 + 0.14 * sp + 0.12 * (bl - 0.5);
    alb *= 1.0 + (0.45 * step(0.83, sp2) - 0.18 * step(sp2, 0.12)) * uBakeK.y;   // pale mica / quartz specks, dark pits
    if (peb) {
      if (cls < 0.5 && gR1.x > 0.7) alb *= 0.9 + 0.2 * step(0.72, lnoise(gLoc / 0.0012 + 17.0, 8)); // granitic speckle
      if (gR0.z > 0.85) alb *= 1.0 - 0.35 * step(0.86, lnoise(gLoc / 0.0016 + 3.0, 12));            // pits
      if (cls < 1.5 && gR1.y > 0.8) {                                                              // quartz vein
        float vd = abs(dot(gLoc, vec2(cos(gR1.z * 9.0), sin(gR1.z * 9.0))) - (gR0.y - 0.5) * gA);
        alb = mix(alb, vec3(0.55, 0.50, 0.46), smoothstep(0.0009, 0.0004, vd) * 0.8);
      }
      // weathered rim darker, top paler (wear + wet skin), broad blotches
      alb *= mix(1.05, 0.86, smoothstep(0.45, 1.0, gRR)) * (1.0 + 0.10 * (bl - 0.5));
    } else {
      // convex chips: sky-lit crown, occluded wet flanks (the bright-cap / dark-rim speckle)
      alb *= mix(1.0 + 0.12 * uBakeK2.x, 1.0 - 0.38 * uBakeK2.x, smoothstep(0.30, 1.0, gRR));
    }
    // small grains: their chroma is below the video's chroma resolution (4:2:0 + denoise)
    pool = mix(pool, 1.0, gKind < 1.5 ? 0.75 : (gKind < 2.5 ? 0.5 : (gKind < 3.5 ? 0.25 : 0.0))) * (peb ? uBakeK.w : uBakeK.z);
    pool = min(pool, 1.0);
    alb = mix(alb, dot(alb, vec3(0.2126, 0.7152, 0.0722)) * MEAN_TINT, pool);
    // 2-6 cm patches of pinker / greyer gravel (the clip's chroma varies at this scale, not per grain)
    float pk = pnoise(p, floor(TILE / 0.045), 14) * 0.65 + pnoise(p, floor(TILE / 0.018), 15) * 0.35 - 0.5;
    alb *= 1.0 + 0.15 * uBakeK2.y * pk * vec3(1.0, -0.35, -0.55);
    // soft knee on the palest grains (the clip's pale specks stay below ~225 sRGB unless glinting)
    float Lg = dot(alb, vec3(0.2126, 0.7152, 0.0722));
    if (Lg > 0.46) alb *= (0.46 + (Lg - 0.46) * 0.5) / Lg;
    // and on the darkest: wet dark grains still read dark brown-grey, never as black pinholes
    if (Lg < 0.05) alb *= (0.025 + 0.5 * Lg) / max(Lg, 1e-4);
    crown = 1.0 - gRR * gRR;
    rnd = fract(gR0.x * 13.7 + gR1.y * 7.1);
    rough += 0.08 * (gR1.x - 0.5);
  }
  A = vec4(alb, gH);
  C = vec4(clamp(rough, 0.05, 1.0), crown, gKind > 3.5 ? 1.0 : 0.0, rnd);
}

void main() {
  // 2x2 supersampling: crisp, unaliased grain edges in the top mip
  vec4 A = vec4(0.0), C = vec4(0.0);
  for (int s = 0; s < 4 + uZero; s++) {
    vec2 o = (vec2(float(s & 1), float(s >> 1)) - 0.5) * 0.5 / RES;
    vec4 a, c;
    surface(vUv + o, a, c);
    A += a; C += c;
  }
  gl_FragColor = A * 0.25;
  outC = C * 0.25;
}`;

// ------------------------------------------------------------------------ pass 2
// normals (Sobel on the float height) and ambient occlusion (horizon scan)
export const BAKE_NORMALS = COMMON + /* glsl */ `
uniform sampler2D uA;
uniform int uZero;
in vec2 vUv;
float hAt(ivec2 q) { q = ivec2(mod(vec2(q), RES)); return texelFetch(uA, q, 0).a; }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float px = TILE / RES;
  float h00 = hAt(p + ivec2(-1, -1)), h10 = hAt(p + ivec2(0, -1)), h20 = hAt(p + ivec2(1, -1));
  float h01 = hAt(p + ivec2(-1, 0)), h11 = hAt(p), h21 = hAt(p + ivec2(1, 0));
  float h02 = hAt(p + ivec2(-1, 1)), h12 = hAt(p + ivec2(0, 1)), h22 = hAt(p + ivec2(1, 1));
  float dx = ((h20 + 2.0 * h21 + h22) - (h00 + 2.0 * h01 + h02)) / (8.0 * px);
  float dy = ((h02 + 2.0 * h12 + h22) - (h00 + 2.0 * h10 + h20)) / (8.0 * px);
  vec3 n = normalize(vec3(-dx, -dy, 1.0));
  // horizon-based AO over 12 directions, radii 0.8-6 mm (soft: the gaps are lit by the sky
  // through a wet, translucent grit fill, never black holes)
  float occ = 0.0;
  for (int i = 0; i < 12 + uZero; i++) {
    float a = (float(i) + 0.5) * 0.5235988;
    vec2 dir = vec2(cos(a), sin(a));
    float tmax = 0.0;
    for (int k = 1; k <= 5 + uZero; k++) {
      float r = 0.0005 * pow(1.65, float(k));
      ivec2 q = p + ivec2(round(dir * r / px));
      tmax = max(tmax, (hAt(q) - h11) / r);
    }
    occ += tmax / sqrt(1.0 + tmax * tmax);      // sin(horizon elevation)
  }
  float ao = clamp(1.0 - occ / 12.0 * 1.15, 0.0, 1.0);
  gl_FragColor = vec4(n * 0.5 + 0.5, ao);
}`;
