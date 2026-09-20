import * as THREE from 'three';
import { CONFIG, glslDefines, bedProfileJS } from '../config.js';
import { NOISE, BED, SKY, OPTICS, DEPTH } from '../glsl/common.js';
import { BREAKER } from '../glsl/breaker.js';
import { CHOP, SWE_SAMPLE } from '../glsl/water.js';
import { SWASH_FAR } from '../glsl/swashfar.js';
import { FullscreenPass, makeShader } from '../core/gpu.js';
import { halfWidthAt } from '../water/WaterSurface.js';
import { TILE, TILE_RES, BAKE_STONES, BAKE_NORMALS, bakeUniforms } from './pebbleBake.js';
import { buildExploreGeometry, LAND, LAND_MORPH_GLSL, TERRAIN_GLSL, BEND_GLSL, groundHeightJS } from './terrain.js';
import { HAZE_GLSL } from './haze.js';
import { Promenade } from './Promenade.js';
import { Backdrop } from './Backdrop.js';
import { makeMacroNoise, MACRO_NOISE_GLSL } from './macroNoise.js';
import { SAND_FRAG, photoUniforms, loadCoastTextures } from './SandMaterial.js';
import { CoastalRocks } from './CoastalRocks.js';
import { rockTopAt } from './CoastalBed.js';
import { COAST_SHADOW, coastUniforms } from './CoastMaterial.js';

// Pebble beach + seabed.
// A periodic gravel/pebble tile is baked once on the GPU (pebbleBake.js). At run time it
// is sampled with stochastic hex tiling (random offset + rotation per hex vertex) and
// height-based blending, so seams follow stone outlines and nothing repeats. Near the
// camera a short parallax-occlusion march and a sun-shadow probe give the stones relief.
// Wetness comes from the swash solver: damp wet gravel everywhere, darker stones under
// the water, a glossy skin that fades ~0.35 s after the film leaves, drained gravel that stays
// darker (brim-full pores) for ~1 s and greyer for ~1-2 s (chroma recovers with the time since the
// film left), the 1-1.5 mm film the pebbles retain after a backwash drawn as brim-full pores (not a
// sky mirror), a turbid grey-green veil under the thin (1.5-3.6 mm) films the water mesh does not
// draw (matched to its ragged cut-off), a glitter band of point glints just behind a retreating
// film edge, and weak caustics under 1-8 cm of water. The phone's chroma
// handling is emulated per pixel: the colour comes from the local mean albedo at a luminance-
// independent display-space strength (see 'camera chroma').
// Above the reach of the swash the pebbles dry out (wet -> damp -> dry, crowns first): the
// damp memory of the solver inside its window, swashFarWet() beyond it. The upper beach is
// coarser (sorted pebbles) and the storm berm is bleached.
// Seaward of the step the bed turns into pale sand/pebbles (albedo ~0.4) so the water
// optics can build the luminous turquoise band; the mesh runs out to z = -60 m.
// Clip mode keeps the fixed-camera mesh; explore mode (?explore) draws a focus-centred
// terrain (terrain.js: dense around the player, world-aligned levels, far skirts to +-4 km on the
// curved bay, the back-beach and the promenade wall behind the berm, Promenade.js) and the city,
// palms and hills around the bay (Backdrop.js).

export const TILE_METERS = TILE;
const EXPLORE = !!CONFIG.explore;

const VERT_CLIP = /* glsl */ `
out vec3 vWorld;
out float vViewZ;
out vec3 vNb;
void main() {
  vec2 xz = position.xz;
  vec3 P = vec3(xz.x, bedHeight(xz), xz.y);
  float e = 0.04;
  float bx = bedHeight(xz + vec2(e, 0.0)) - bedHeight(xz - vec2(e, 0.0));
  float bz = bedHeight(xz + vec2(0.0, e)) - bedHeight(xz - vec2(0.0, e));
  vNb = normalize(vec3(-bx / (2.0 * e), 1.0, -bz / (2.0 * e)));
  vWorld = P;
  vec4 mv = viewMatrix * vec4(P, 1.0);
  vViewZ = -mv.z;
  gl_Position = projectionMatrix * mv;
}`;

// explore: columns are offsets from the snapped focus (world-aligned levels, terrain.js),
// rows are world z. The drape relief fades with the distance from the player where the
// columns get coarse; the drawn surface is lowered by the relief it no longer carries so the
// water films (tested against the exact bed) never disappear under it.
const VERT_EXPLORE = /* glsl */ `
out vec3 vWorld;
out float vViewZ;
out vec3 vNb;
${LAND_MORPH_GLSL}
void main() {
  vec2 xz = vec2(landColumnX(position),position.z);
  vec3 k = terrainKeep(xz);
  // height + central differences in a dynamic loop: one inlined copy of the terrain function
  float e = 0.04;
  float y = 0.0, bias = 0.0;
  vec2 g = vec2(0.0);
  for (int i = 0; i < 5 + min(uEvtCount, 0); i++) {
    vec2 o = vec2(i == 0 ? e : (i == 1 ? -e : 0.0), i == 2 ? e : (i == 3 ? -e : 0.0));
    float b0;
    float h = terrainHeight(xz + o, k, b0);
    if (i == 4) { y = h; bias = b0; } else g += o / e * h;
  }
  vNb = normalize(vec3(-g.x / (2.0 * e), 1.0, -g.y / (2.0 * e)));
  vec3 P = vec3(xz.x, y, xz.y);
  vWorld = P;
  // (drawn on the curved bay: shading keeps the straight-beach coordinates, terrain.js)
  vec4 mv = viewMatrix * vec4(P - vec3(0.0, bias, bayBend(P.x)), 1.0);
  vViewZ = -mv.z;
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = /* glsl */ `
uniform sampler2D uTexA;   // wet albedo, height (m)
uniform sampler2D uTexB;   // tile normal, AO
uniform sampler2D uTexC;   // roughness, crown, pebble flag, random
uniform float uBeachGain;  // global albedo calibration
uniform vec3 uSeabed;      // albedo of the pale bed beyond the step
uniform vec4 uBeachDebug;  // x: ignore the water state, y: no specular/glints, z: albedo only, w: wetness debug
uniform float uBeachFilmTau; // time constant of the solver's film channel (swash/SwashSim.js WET pass)
uniform vec4 uDry;         // x: dry albedo gain, y: dry saturation, z: berm bleach gain, w: dry roughness
in vec3 vWorld;
in float vViewZ;
in vec3 vNb;

#define TILE ${TILE.toFixed(4)}
#define HEX 0.55
#define HMAX 0.0120
#define NORMAL_GAIN 1.6
#define COARSE_SCALE 1.2      // coarser lag near the top of the swash (clip frame bottom)
#define GLOSS_TAU 0.35
// Just-drained gravel (clip frame bottom, AE-corrected, v 0.96-1: 0.112 when the film leaves at
// t = 3.0 s, 0.14 at 3.75 s, still 0.13-0.14 at 4.75 s; the long-exposed damp state is 0.165): the
// pores brim with water for a moment (fast), then a residual skin of water between the grains
// dries over several seconds (slow), both darker and greyer than damp
#define DRAIN_DARK 0.17       // fast: pores brim-full
#define DRAIN_TAU 0.45
#define RESID_DARK 0.17       // slow: residual moisture skin
#define RESID_TAU 4.0
// Chroma of the wet states (display R-B at the clip frame bottom, v 0.97-1): 22 when the film has
// just left (t = 3.0 s), 28 / 29 / 32 / 33 after 0.3 / 0.6 / 1.2 / 1.8 s; a film of a mm or so
// held in the pores keeps it greyer (lower face, t = 0-0.9 s)
#define GREY_DAMP 0.24        // long-exposed damp gravel
#define GREY_FRESH 0.47       // + just drained, decaying with TAIL_TAU
#define TAIL_TAU 0.6
#define GREY_PORES 0.2        // + a retained film in the pores (0.2-1.5 mm)
#define GREY_SKIN 0.3         // + ... thicker than ~1 mm
#define GREY_UNDER 0.45       // the bed under the backwash sheet (seen through the water mesh)
#define GLINT_GAIN 12.0      // point glints: 1250 /m^2 (0.3-0.9 mm) in the glitter band, 125 /m^2 on damp crowns
#define VEIL_MAX 0.35         // turbid veil of a thin (< 3.6 mm) backwash film
const vec3 VEIL_ALB = vec3(0.40, 0.45, 0.43);
#define HMIN -0.0026
#define BLEND_H 0.0060
#define BLEND_D 0.00045
#define PI 3.14159265
// mip bias: crisp at the frame bottom (a 4K->1080 downscale in the clip), progressively softer with
// distance (the phone's denoise/codec removes detail faster than the geometry alone would)
#ifdef BEACH_EXPLORE
float lodScale(float d) { return mix(0.9, 1.2, smoothstep(2.45, 5.0, d)); }   // (full-window: a little crisper)
#else
float lodScale(float d) { return mix(0.9, 1.5, smoothstep(2.45, 3.6, d)); }
#endif
#define CHROMA_YREF 0.179  // mean gravel luminance (pre-grade)
#define CHROMA_EXP 0.30    // linear chroma ~ Y^(1 - CHROMA_EXP): ~constant display-space chroma after the grade
#define CAM_CHROMA 1.0
#define CHROMA_GAIN 1.1
#define PIX_KEEP 0.1    // share of the per-grain (pixel-scale) chroma the camera keeps
const vec3 GRAVEL_WARM = vec3(1.03, 0.941, 0.87);   // (G-R -19..-20 in every luma bin, frame bottom)

vec2 rotF(vec2 cs, vec2 v) { return vec2(cs.x * v.x - cs.y * v.y, cs.y * v.x + cs.x * v.y); }
vec2 rotB(vec2 cs, vec2 v) { return vec2(cs.x * v.x + cs.y * v.y, -cs.y * v.x + cs.x * v.y); }
float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
// band-limit a noise octave of wavelength lam to the pixel footprint (no moire at grazing views)
float octKeep(float lam, float foot) { return 1.0 - smoothstep(0.3, 0.6, 2.0 * foot / lam); }

// ---- stochastic hex tiling (Heitz-Neyret / Mikkelsen triangle grid)
float gDist;                   // view depth of the fragment
float gBlur;                   // extra filtering of the bed seen through a water film
vec3 hw;                       // barycentric weights of the 3 hex vertices
vec2 hc1, hc2, hc3;            // per-vertex rotation (cos, sin)
vec2 ho1, ho2, ho3;            // per-vertex uv offset
vec2 gx1, gy1, gx2, gy2, gx3, gy3;
float hs1, hs2, hs3;           // per-vertex scale: coarser lag gravel in the upper swash
// upper swash near the camera: lag of coarse clasts (bigger grains, more pebbles, warmer)
float coarseAt(vec2 xz) { return smoothstep(1.25, 1.95, xz.y + 0.13 * sin(xz.x * 2.3 + 1.3) + 0.09 * sin(xz.x * 5.1 + 0.4)); }
// cross-shore sorting above the swash: the dry upper beach is pebbles, the storm berm the coarsest
// (D50 of the matrix: ~5 mm in the swash, 15-20 mm at z = 6-12, 30-40 mm on the storm berm and
// the coarsest band along its crest, 25-30 mm on the back-beach)
float zoneScale(vec2 xz) {
  float z = xz.y + 0.6 * sin(xz.x * 0.21 + 0.7) + 0.4 * sin(xz.x * 0.53 + 2.1);
  float zc = z - 19.6 - 0.8 * sin(xz.x * 0.07 + 1.3);
  // (the storm cusps below the crest sort the pebbles: coarser on the horns, terrain.js)
  float horn = cos(6.2831853 * (xz.x / 21.0 + 0.3 * sin(xz.x / 37.0 + 0.4))) * exp(-(z - 16.3) * (z - 16.3) / 4.8);
  return 1.0 + 1.0 * smoothstep(4.0, 9.0, z) + 1.3 * smoothstep(10.0, 17.0, z) + 0.9 * exp(-zc * zc / 2.0)
       + 0.45 * horn - 0.5 * smoothstep(22.0, 28.0, z);
}
float vScale(vec2 v) {
  vec2 st = vec2(v.x, (v.y + 0.57735027 * v.x) / 1.15470054);
  vec2 p = st * HEX;
  return mix(1.0, COARSE_SCALE, coarseAt(p)) * zoneScale(p);
}
void hexSetup(vec2 xz, vec2 dx, vec2 dy) {
  vec2 st = xz / HEX;
  vec2 sk = vec2(st.x, -0.57735027 * st.x + 1.15470054 * st.y);
  vec2 b = floor(sk);
  vec3 t = vec3(fract(sk), 0.0);
  t.z = 1.0 - t.x - t.y;
  float s = step(0.0, -t.z), s2 = 2.0 * s - 1.0;
  hw = vec3(-t.z * s2, s - t.y * s2, s - t.x * s2);
  vec2 v1 = b + vec2(s, s), v2 = b + vec2(s, 1.0 - s), v3 = b + vec2(1.0 - s, s);
  vec3 r1 = hash32(v1 * vec2(1.13, 0.87) + 31.7), r2 = hash32(v2 * vec2(1.13, 0.87) + 31.7), r3 = hash32(v3 * vec2(1.13, 0.87) + 31.7);
  hc1 = vec2(cos(r1.z * 6.2831853), sin(r1.z * 6.2831853)); ho1 = r1.xy;
  hc2 = vec2(cos(r2.z * 6.2831853), sin(r2.z * 6.2831853)); ho2 = r2.xy;
  hc3 = vec2(cos(r3.z * 6.2831853), sin(r3.z * 6.2831853)); ho3 = r3.xy;
  hs1 = vScale(v1); hs2 = vScale(v2); hs3 = vScale(v3);
  float ls = lodScale(gDist) * gBlur;
  vec2 dxt = dx * ls / TILE, dyt = dy * ls / TILE;
  gx1 = rotF(hc1, dxt / hs1); gy1 = rotF(hc1, dyt / hs1);
  gx2 = rotF(hc2, dxt / hs2); gy2 = rotF(hc2, dyt / hs2);
  gx3 = rotF(hc3, dxt / hs3); gy3 = rotF(hc3, dyt / hs3);
}
vec2 uv1(vec2 xz) { return rotF(hc1, xz / (TILE * hs1)) + ho1; }
vec2 uv2(vec2 xz) { return rotF(hc2, xz / (TILE * hs2)) + ho2; }
vec2 uv3(vec2 xz) { return rotF(hc3, xz / (TILE * hs3)) + ho3; }
// height of the dominant hex sample (used by the parallax march / shadow probe)
vec2 hcD, hoD, gxD, gyD; float hsD;
void pickDominant() {
  if (hw.x >= hw.y && hw.x >= hw.z) { hcD = hc1; hoD = ho1; gxD = gx1; gyD = gy1; hsD = hs1; }
  else if (hw.y >= hw.z) { hcD = hc2; hoD = ho2; gxD = gx2; gyD = gy2; hsD = hs2; }
  else { hcD = hc3; hoD = ho3; gxD = gx3; gyD = gy3; hsD = hs3; }
}
float domHeight(vec2 xz) { return textureGrad(uTexA, rotF(hcD, xz / (TILE * hsD)) + hoD, gxD, gyD).a * hsD; }

// ---- point glints (sub-pixel, energy-preserving footprint), one candidate per cell
float sparkle(vec2 xz, float cell, float dens, float tseed, float foot, float sz) {
  vec2 q = xz / cell;
  vec2 c = floor(q);
  vec2 f = q - c;
  vec3 r = hash33(vec3(c, tseed));
  if (r.x > dens * cell * cell) return 0.0;
  vec2 pos = 0.2 + 0.6 * hash22(c * 1.37 + vec2(tseed * 3.7, 11.0));
  float rg = mix(0.0005, 0.0015, r.y) * sz;
  vec2 d = (f - pos) * cell;
  float re = max(rg, foot);
  return (rg * rg) / (re * re) * exp(-dot(d, d) / (re * re) * 1.6) * (0.25 + 0.75 * r.z * r.z);
}

// ---- weak sun caustics under a thin film: warped cellular focus lines, 4-10 cm cells
float caustics(vec2 p, float t) {
  vec2 q = p / 0.07;
  q += 0.35 * vec2(gnoise(q * 0.31 + vec2(t * 0.55, 0.0)), gnoise(q * 0.31 + vec2(5.2, -t * 0.47)));
  vec3 w = worley(q + vec2(t * 0.45, -t * 0.3));
  return pow(clamp(1.0 - (w.y - w.x) * 2.4, 0.0, 1.0), 4.0);
}

void main() {
  vec3 P = vWorld;
#ifdef BEACH_EXPLORE
  vec3 V = normalize(cameraPosition - P + vec3(0.0, 0.0, bayBend(P.x)));
#else
  vec3 V = normalize(cameraPosition - P);
#endif
  float dist = vViewZ;
  gDist = dist;
  vec2 dx = dFdx(P.xz), dy = dFdy(P.xz);
  float foot = 0.5 * max(length(dx), length(dy));        // pixel footprint radius on the bed (m)
  vec3 Nb = normalize(vNb);
  vec3 Tx = normalize(vec3(Nb.y, -Nb.x, 0.0));
  vec3 Tz = cross(Tx, Nb);
  vec3 Vs = vec3(dot(V, Tx), dot(V, Tz), dot(V, Nb));
#ifdef BEACH_EXPLORE
  float texW = 1.0 - smoothstep(38.0, 60.0, dist);   // (grazing views along the beach keep the stones' mottle)
#else
  float texW = 1.0 - smoothstep(13.0, 18.0, dist);
#endif
  float pomW = 1.0 - smoothstep(4.4, 5.6, dist);

  // ---- water state: the swash solver inside its (scrolling) window, the analytic far-field
  // swash beyond it (glsl/swashfar.js). Sampled early: a film blurs the bed a little (ripple
  // refraction smears 2-5 mm of detail)
  vec2 suv = sweUV(P.xz);
  float wOn = 1.0 - uBeachDebug.x;
  float wSolver = sweInside(P.xz) * (1.0 - swashFarWeight(P.xz));
  float wIn = wSolver * wOn;                 // solver weight
  float wFar = (1.0 - wSolver) * wOn;        // far-field weight
  vec4 sv = textureLod(uSweView, suv, 0.0);
  vec4 swF = vec4(0.0), fmF = vec4(0.0);
  vec2 wtF = vec2(0.0);
  if (wFar > 0.0 && P.z < 6.0) { swF = swashFar(P.xz, uTime); fmF = swashFarFoam(P.xz, uTime); }
  if (wFar > 0.0 && P.z < 12.0) wtF = swashFarWet(P.xz, uTime);
  float hS = wIn * sv.x;                                           // solver depth over the smooth bed
  float hF = wFar * swF.x;                                         // far-field swash depth
  gBlur = 1.0 + 0.8 * smoothstep(0.004, 0.03, hS + hF);

  // tile mean (1x1 mip) for the far field and the seabed
  vec3 meanA = textureLod(uTexA, vec2(0.5), 14.0).rgb;
  vec3 alb = meanA;
  vec3 albLo = meanA;               // local mean albedo (~8 px footprint): carries the camera chroma
  vec3 nS = vec3(0.0, 0.0, 1.0);    // normal in the slope frame (Tx, Tz, Nb)
  float ao = 0.8, rough = 0.35, crown = 0.3, peb = 0.0, hgt = 0.0, rnd = 0.5, nLen = 1.0;
  float hN = 0.5;                   // stone height, normalised 0 (gaps) .. 1 (crowns)
  float shadow = 1.0;
  vec2 xz = P.xz;
  vec3 L = uSunDir;

  if (texW > 0.0) {
    hexSetup(P.xz, dx, dy);
    pickDominant();
    // taller relief where the grains are scaled up (sorted pebbles of the upper beach)
    float hsK = max(max(hs1, max(hs2, hs3)) / COARSE_SCALE, 1.0);
    float hmax = HMAX * hsK, hmin = HMIN * hsK;
    // ---- parallax occlusion (march down from HMAX along the view ray)
    if (pomW > 0.0) {
      vec2 sh = Vs.xy / max(Vs.z, 0.25) * pomW;
#ifdef BEACH_EXPLORE
      // (fewer steps when looking down: the parallax is short)
      int NS = int(mix(5.0, 12.0, smoothstep(0.95, 0.4, Vs.z))) + min(uEvtCount, 0);
#else
      const int NS = 10;
#endif
      float dh = (hmax - hmin) / float(NS);
      float hk = hmax, Hk = domHeight(xz + sh * hk);
      float hp = hk, Hp = Hk;
      for (int k = 0; k < NS; k++) {
        if (Hk >= hk) break;
        hp = hk; Hp = Hk;
        hk -= dh;
        Hk = domHeight(xz + sh * hk);
      }
      float d0 = Hp - hp, d1 = Hk - hk;
      float tt = clamp(d0 / min(d0 - d1, -1e-6), 0.0, 1.0);
      if (Hk < hk) tt = 1.0;
      xz += sh * mix(hp, hk, tt);
    }
    vec2 u1 = uv1(xz), u2 = uv2(xz), u3 = uv3(xz);
    vec4 A1 = textureGrad(uTexA, u1, gx1, gy1);
    vec4 A2 = textureGrad(uTexA, u2, gx2, gy2);
    vec4 A3 = textureGrad(uTexA, u3, gx3, gy3);
    A1.a *= hs1; A2.a *= hs2; A3.a *= hs3;
    vec3 sb = vec3(A1.a, A2.a, A3.a) + BLEND_H * hw;
    float sm = max(sb.x, max(sb.y, sb.z));
    vec3 bw = max(sb - sm + BLEND_D, 0.0);
    bw /= bw.x + bw.y + bw.z;
    alb = A1.rgb * bw.x + A2.rgb * bw.y + A3.rgb * bw.z;
#ifdef BEACH_EXPLORE
    // (beyond ~14 m an 8-pixel footprint averages the tile's 2-6 cm chroma patches out)
    if (dist > 14.0) albLo = meanA * (lum(alb) / max(lum(meanA), 1e-4));
    else
#endif
    albLo = textureGrad(uTexA, u1, gx1 * 8.0, gy1 * 8.0).rgb * hw.x + textureGrad(uTexA, u2, gx2 * 8.0, gy2 * 8.0).rgb * hw.y
          + textureGrad(uTexA, u3, gx3 * 8.0, gy3 * 8.0).rgb * hw.z;
    hgt = dot(vec3(A1.a, A2.a, A3.a), bw);
    vec3 nw = vec3(0.0); vec4 cc = vec4(0.0); float aoS = 0.0;
    if (bw.x > 0.0) { vec4 B = textureGrad(uTexB, u1, gx1, gy1); vec3 n = B.xyz * 2.0 - 1.0; nw += bw.x * vec3(rotB(hc1, n.xy), n.z); aoS += bw.x * B.w; cc += bw.x * textureGrad(uTexC, u1, gx1, gy1); }
    if (bw.y > 0.0) { vec4 B = textureGrad(uTexB, u2, gx2, gy2); vec3 n = B.xyz * 2.0 - 1.0; nw += bw.y * vec3(rotB(hc2, n.xy), n.z); aoS += bw.y * B.w; cc += bw.y * textureGrad(uTexC, u2, gx2, gy2); }
    if (bw.z > 0.0) { vec4 B = textureGrad(uTexB, u3, gx3, gy3); vec3 n = B.xyz * 2.0 - 1.0; nw += bw.z * vec3(rotB(hc3, n.xy), n.z); aoS += bw.z * B.w; cc += bw.z * textureGrad(uTexC, u3, gx3, gy3); }
    nLen = clamp(length(nw), 0.05, 1.0);
    nS = nw / nLen;
    // relief gain: the mip-filtered tile normals lose the grain-scale slopes that give the clip's chips
    // their lit far rim / shaded near face (sun ahead of the camera)
    nS = normalize(vec3(nS.xy * NORMAL_GAIN, nS.z));
    ao = aoS; rough = cc.x; crown = cc.y; peb = cc.z; rnd = cc.w;
    hN = mix(0.5, clamp((hgt / hsK - HMIN) / (HMAX - HMIN), 0.0, 1.0), texW);
    // ---- sun shadow probe on the dominant sample
    if (pomW > 0.0) {
      vec2 ls = vec2(dot(L, Tx), dot(L, Tz)) / max(dot(L, Nb), 0.2);
      float h0 = hgt;
      for (int k = 1; k <= 4; k++) {
        float up = 0.0007 * hsK * float(k * k);
        float hs = domHeight(xz + ls * up);
        shadow = min(shadow, clamp(1.0 - (hs - (h0 + up)) / (0.0007 * hsK), 0.0, 1.0));
      }
      shadow = mix(1.0, shadow, pomW);
    }
    // fade to the mean far away (the texture is far below pixel size there)
    alb = mix(meanA, alb, texW);
    albLo = mix(meanA, albLo, texW);
    nS = normalize(mix(vec3(0.0, 0.0, 1.0), nS, texW));
    ao = mix(0.8, ao, texW);
  }

  // ---- macro variation: mottle at 0.3 m (+-6 % lum, +-3 deg hue) and broad patches (each octave
  // band-limited to the pixel footprint)
  float m1 = gnoise(P.xz / 0.30) * 0.6 * octKeep(0.30, foot) + gnoise(P.xz / 0.13 + 3.7) * 0.3 * octKeep(0.13, foot);
  float m2 = gnoise(P.xz / 2.3 + 11.0);
  alb *= (1.0 + 0.09 * m1 + 0.075 * m2) * uBeachGain;
  float cz = coarseAt(P.xz);
  vec3 cMac = (1.0 + 0.035 * gnoise(P.xz / 0.45 + 7.1) * octKeep(0.45, foot) * vec3(1.0, 0.0, -1.0)) * mix(vec3(1.0), vec3(0.965, 0.95, 0.93), cz);
  // (warmer than the baked palette: the clip's exposed gravel reads G-B ~ 12-15 codes at every time,
  // the render ~5 with the round-2 grade)
  cMac *= GRAVEL_WARM;
  // Broad mineral sorting follows former run-up lines, with no repeating stripes.
  float mineral = gnoise(P.xz * vec2(0.16, 0.45) + vec2(17.8, 2.6));
  cMac *= mix(vec3(0.955, 0.982, 1.025), vec3(1.03, 1.005, 0.967), 0.5 + 0.5 * mineral);
  alb *= cMac;
  albLo *= cMac;
  alb = max(mix(alb, lum(meanA) + (alb - lum(meanA)) * 1.18, cz), 0.0);   // livelier contrast near the camera
  // above the swash: along-shore sorting patches (finer / coarser, greyer / warmer pebbles) and a
  // patchy strand line of dried wrack and fine debris at the storm run-up; they carry the beach's
  // structure at every distance along the shore
  float upW = smoothstep(2.6, 4.5, P.z);
  if (upW > 0.0) {
    vec4 nA = macroNoiseG(P.xz + vec2(31.0, 7.0), vec2(23.0, 3.1), dx, dy);
    vec4 nB = macroNoiseG(P.xz - vec2(5.0, 11.0), vec2(7.5, 1.3), dx, dy);
    float s1 = nA.x * 0.6 + nB.y * 0.4;
    vec3 tint = vec3(1.0) + 0.035 * nA.z * vec3(1.0, 0.1, -0.9);
    float zs = 6.3 + 0.3 * sin(P.x / 13.0 + 0.7 * sin(P.x / 37.0)) + 0.15 * sin(P.x * 0.47);
    float wS = 0.07 + 0.035 * sin(P.x / 3.0 + 1.3 * sin(P.x / 7.7));
    float sd = (P.z - zs) / wS;
    float strand = exp(-sd * sd) * smoothstep(-0.1, 0.5, nB.w)
                 * min(1.0, wS / max(3.0 * foot, 1e-4));
    vec3 m = (1.0 + 0.12 * s1 * upW) * tint * mix(vec3(1.0), vec3(0.62, 0.56, 0.48), 0.6 * strand * upW);
    alb *= m;
    albLo *= m;
  }

  // ---- seabed beyond the step: pale sand / washed pebbles, contrast reduced. The pale bed only
  // shows where the water is deep enough to tint it (below the step toe): the steep step itself
  // is coarse gravel, so no pale rim shows through the clear shallow water at its foot
  float sea = smoothstep(-0.9, -2.5, P.z + 0.35 * gnoise(P.xz * vec2(0.9, 0.6) + 2.0));
  vec3 seaAlb = uSeabed * pow(max(lum(alb) / max(lum(meanA * uBeachGain), 1e-3), 0.0), 0.55) * (1.0 + 0.08 * m2);
  alb = mix(alb, seaAlb, sea);

  vec3 N = normalize(Tx * nS.x + Tz * nS.y + Nb * nS.z);

  // ---- water state
  vec2 wt = textureLod(uSweWet, suv, 0.0).xy * wIn + wtF * wFar;
  vec4 fm = textureLod(uSweFoam, suv, 0.0) * wIn + fmF * wFar;
  float brk = (P.z < 1.2 && P.z > -8.0) ? brkSurfaceOnly(P.xz, uTime) : 0.0;
  // (beyond the window: the breaker water over the bed, or the far-field swash sheet up the face)
  float dSmooth = mix(max(max(brk - P.y, 0.0), swF.x), (hS > 2e-4 ? hS : 0.0) + brk, wIn);
  dSmooth = max(dSmooth, 0.0);
  float wDepth = dSmooth > 2e-4 ? max(dSmooth - hgt, 0.0) : 0.0;  // over the local stone
  float sub = smoothstep(0.0002, 0.0016, wDepth);                  // this stone is under water
  // the water surface is drawn above: same ragged cut-off as WaterSurface.js (depth > 0.9 mm + edge
  // noise), so the thin-film look below takes over exactly in the mesh's holes and fringe
  float edgeN = vnoise(P.xz * 41.0) * 0.0016 + vnoise(P.xz * 7.0) * 0.0018;
  float meshCover = smoothstep(0.0009 + edgeN, 0.0021 + edgeN, dSmooth);   // film look runs ~1 mm under the fringe
  // the solver's film channel decays as exp(-t / uSweFilmTau) once the water has gone: reshape it
  // to the measured gloss decay (tau ~0.35 s) and recover the time since the film left (half-float
  // W.x stalls near 1e-6)
  float fTau = max(uBeachFilmTau, 0.05);
  float gloss = max(pow(clamp(wt.x, 0.0, 1.0), fTau / GLOSS_TAU), sub);
  // Porous stones and sheltered gaps retain water longer than exposed crowns.
  float retention=mix(.70,1.55,rnd)*mix(1.28,.82,hN);
  float since = -fTau * log(max(wt.x, 2e-6))/retention;
  gloss=max(exp(-since/GLOSS_TAU),sub);
  float foam = clamp(fm.r + 0.5 * fm.g, 0.0, 1.0);
  // a film thinner than the grain crowns (the 1-1.5 mm the pebbles retain after every backwash)
  // is brim-full pores, not a mirror: no sky film / veil below ~1.5 mm, the drained state instead
  float thick = smoothstep(0.0015, 0.003, dSmooth);
  float pores = smoothstep(0.0002, 0.0006, dSmooth) * (1.0 - thick);

  // ---- drying above the swash: the damp memory (1 = swashed within the last minute) sets how dry
  // the gravel is; the stone crowns dry first, the gaps between the grains last
  // The damp memory only marks the latest run-ups (the solver's half-float channel fades in ~15 s):
  // above them a damp fringe (capillary rise, spray, the bigger waves of the last minutes) dries
  // out over ~1 m, its upper edge meandering along the shore
  float recent = smoothstep(0.65, 0.95, wt.y);
  // (scalloped by the beach cusps: the swash runs further up in the embayments between the horns)
  float cph = P.x / 17.0 + 0.25 * sin(P.x / 23.0 + 1.1);
  float horn = 1.0 - abs(sin(3.14159 * cph));
  float bEff = P.y + 0.035 * sin(P.x / 9.0 + 0.6 * sin(P.x / 21.0)) + 0.018 * sin(P.x / 2.7 + 2.0) - 0.06 * (1.0 - horn * horn);
  float wetMem = max(recent, 0.82 * (1.0 - smoothstep(0.36, 0.72, bEff)) * wOn);
  float dry0 = (1.0 - smoothstep(0.12, 0.85, wetMem)) * (1.0 - sea);
  // patchy: some clusters of stones hold their water longer
  if (dry0 > 0.0 && dry0 < 1.0) {
    float nd = macroNoiseG(P.xz, vec2(0.7), dx, dy).y * 0.7 + macroNoiseG(P.xz + 3.0, vec2(0.23), dx, dy).z * 0.3;
    dry0 = clamp(dry0 + 0.8 * dry0 * (1.0 - dry0) * nd, 0.0, 1.0);
  }
  float dth = 0.5 + 0.34 * (0.5 - hN) + 0.24 * (rnd - 0.5);
  float dryK = smoothstep(dth - 0.16, dth + 0.16, dry0) * (1.0 - gloss) * (1.0 - sub);
  {
    // dry = damp x1.7 in linear, saturation x0.7 (swash analysis 6.2); the storm berm is bleached
    float bleach = smoothstep(7.5, 18.0, P.z);
    // (dry stones scatter more white light from their surface: half of the wet gravel's warm cast goes)
    vec3 gD = uDry.x * (1.0 + uDry.z * bleach) * mix(vec3(1.0), 1.0 / GRAVEL_WARM, 0.5);
    float sD = uDry.y * (1.0 - 0.22 * bleach);
    vec3 a2 = alb * gD;   a2 = max(lum(a2) + (a2 - lum(a2)) * sD, 0.0);
    // (soft knee: dry quartz / limestone stones top out near albedo 0.65, not paper white)
    float La = lum(a2);
    if (La > 0.5) a2 *= (0.5 + (La - 0.5) * 0.35) / La;
    vec3 l2 = albLo * gD; l2 = max(lum(l2) + (l2 - lum(l2)) * sD, 0.0);
    alb = mix(alb, a2, dryK);
    albLo = mix(albLo, l2, dryK);
    rough = mix(rough, uDry.w, dryK);
  }

  vec3 albC = alb;                  // albedo before the wetness tweaks (their chroma is re-applied via chS)
  // ---- wet albedo: damp everywhere; under water darker; just drained: darker, less chroma
  vec3 albSub = alb * 0.80;
  albSub = lum(albSub) + (albSub - lum(albSub)) * 1.12;
  alb = mix(alb, max(albSub, 0.0), sub * (1.0 - sea));
  float wetW = min(wIn + wFar * step(1e-5, wtF.x), 1.0);       // (films only where a swash model runs)
  // The chroma states use area fractions, not this stone's own state: the camera's chroma is far
  // coarser than a grain (under a mm-thin film the submerged gaps would otherwise stay warm and
  // every exposed crown turn grey: grey 'marbles' on a brown matrix). subC: submerged share of the
  // gravel under a film of dSmooth; dryS: dried share.
  float subC = dSmooth > 2e-4 ? smoothstep(-0.001, 0.004, dSmooth) : 0.0;
  float dryS = smoothstep(0.34, 0.66, dry0);
  float tail = exp(-since / TAIL_TAU) * (1.0 - subC) * wetW;
  // (the retained film and the just-drained pores are the same water: the larger of the two)
  // (a retained film over ~1 mm skins the grains: greyer still; clip t = 0-0.9 s, v 0.90-0.97)
  float poreGrey = (GREY_PORES * pores * (1.0 - subC) + GREY_SKIN * smoothstep(0.0008, 0.0015, dSmooth) * (1.0 - thick)) * wetW;
  float greyW = GREY_DAMP * (1.0 - dryS) * (1.0 - subC) * (1.0 - sea) + max(GREY_FRESH * tail, poreGrey);
  // under a real sheet of backwash (> 2-6 mm) the bed reads grey-green through the turbid water
  greyW += GREY_UNDER * smoothstep(0.002, 0.006, dSmooth) * (1.0 - sea);
  alb = mix(alb, vec3(lum(alb)), greyW);
  // drainage: the film has gone but the pores between the grains are still brim-full (fast, ~0.5 s,
  // continuous with the under-water darkening at the film edge), then a residual water skin between
  // the grains keeps the gravel darker for a few seconds
  float wetOut = (1.0 - sub) * wetW * (1.0 - sea);
  float drain = max(exp(-since / DRAIN_TAU), pores) * wetOut;
  alb *= (1.0 - DRAIN_DARK * drain) * (1.0 - RESID_DARK * exp(-since / RESID_TAU) * wetOut);

  // ---- lighting (display-referred rig: Lambert sun + soft sky ambient)
  float NoL = max(dot(N, L), 0.0);
  vec3 sigT = WATER_SIGMA_A + WATER_SIGMA_S;
  vec3 sunT = exp(-sigT * wDepth / max(L.y, 0.2));
  vec3 skyT = exp(-sigT * wDepth * 1.3);
  vec3 col = alb * (uSunColor * (NoL * shadow / PI) * sunT + skyAmbient(N) * ao * skyT);

  // caustics under 1-8 cm films (<= 0.15 x direct), none under foam
  if (wDepth > 0.006 && wDepth < 0.09 && foam < 0.3 && dist < 12.0) {
    float cA = smoothstep(0.006, 0.02, wDepth) * (1.0 - smoothstep(0.05, 0.09, wDepth)) * (1.0 - foam / 0.3);
    float cPat = caustics(P.xz, uTime);
    col += alb * uSunColor * (NoL * shadow / PI) * sunT * 0.15 * (cPat * 1.15 - 0.15) * cA;
  }

  // ---- camera chroma. In the clip the gravel's colour offset is ~constant in display space at every
  // luminance (R-B ~32 in every luma bin) and varies only at the 2-6 cm scale: the phone's chroma
  // denoise + 4:2:0 + codec make chroma a smooth field. So the diffuse colour takes its chroma from
  // the local mean albedo (an ~8 px footprint), scaled to linear chroma ~ Y^0.7 (dark gaps stay brown,
  // specks read warm-white), keeping a little of the per-grain chroma. Lighting and water tints are
  // kept (the correction only swaps the albedo's tint).
  // Wetness does not raise the display chroma the way shading does: the clip's just-drained / filmed
  // gravel is darker AND greyer (t = 3.5 s, v 0.92: R-G 13 vs 18 dry), so the camera law is driven
  // by the luminance the stone would have dry and out of the water.
  // (the wet states' chroma enters only here, linearly: chS; the albedo's own tint is divided out)
  float chS = mix(1.0, 0.96, subC * (1.0 - sea)) * (1.0 - greyW);
  {
    float Yd = max(lum(col), 1e-6);
    float wetK = lum(alb) / max(lum(albC), 1e-5) * (0.6 * lum(sunT) + 0.4 * lum(skyT));
    float g = pow(Yd / max(wetK, 0.05) / CHROMA_YREF, -CHROMA_EXP) * CHROMA_GAIN * chS;
    vec3 tp = clamp(albC / max(lum(albC), 1e-4), 0.4, 2.0);
    vec3 tn = clamp(alb / max(lum(alb), 1e-4), 0.4, 2.0);
    vec3 tl = clamp(albLo / max(lum(albLo), 1e-4), 0.4, 2.0);
    vec3 tt = max(1.0 + (mix(tl, tp, PIX_KEEP) - 1.0) * g, 0.0);
    vec3 cC = col * clamp(tt / tn, 0.4, 2.5);
    cC *= Yd / max(lum(cC), 1e-6);
    col = mix(col, cC, CAM_CHROMA * (1.0 - sea) * (1.0 - uBeachDebug.w));
  }

  // ---- specular: water skin on stone crowns (F0 0.02), glossy film just after drainage; dry stone
  // (F0 0.04, rough) once dried out
  float NoV = max(dot(N, V), 1e-3);
  vec3 H = normalize(V + L);
  float NoH = max(dot(N, H), 0.0);
  float r = mix(rough, 0.05, gloss);
  float toks = (1.0 - nLen) / nLen;                      // normal-map filtering (Toksvig)
  vec3 dnx = dFdx(N), dny = dFdy(N);
  float kap = min(0.5 * (dot(dnx, dnx) + dot(dny, dny)), 0.12);
  float a = sqrt(r * r * r * r + toks * 0.6 + kap);
  a = clamp(a, 0.002, 1.0);
  float F0 = mix(0.02, 0.04, dryK);
  float F = F0 + (1.0 - F0) * pow(1.0 - max(dot(V, H), 0.0), 5.0);
  float skin = mix(mix(0.25 + 0.75 * crown, 1.0, gloss), 1.0, dryK);
  vec3 spec = uSunColor * D_GGX(NoH, a) * V_SmithGGXCorrelated(NoV, NoL, a) * NoL * F * shadow * skin * (1.0 - sub);
  float skyW = 0.25 * gloss * (1.0 - sub) * (1.0 - meshCover);   // drained gravel reads dark with glints, not sky-blue
  if (skyW > 0.002) {
    vec3 R = reflect(-V, normalize(mix(Nb, N, 0.5)));   // the film skin smooths the grain relief
    R.y = max(R.y, 0.02);
    spec += skyRadiance(normalize(R)) * fresnelSchlick(NoV, 0.02) * skyW;
  }

  // thin film the water mesh does not draw (< ~3 mm): its own sky reflection + ripples
  float film = (1.0 - meshCover) * thick;
  if (film > 0.0) {
    vec3 rg = gnoised(P.xz / 0.07 + vec2(uTime * 0.7, -uTime * 0.5)) * octKeep(0.07, foot);
    vec3 Nf = normalize(Nb + vec3(rg.y, 0.0, rg.z) * 0.12);
    vec3 Rf = reflect(-V, Nf);
    Rf.y = max(Rf.y, 0.02);
    float Ff = fresnelWater(max(dot(Nf, V), 0.0));
    col = col * (1.0 - Ff * film) + skyRadiance(normalize(Rf)) * Ff * film;
    // turbid backwash film: suspended sand / micro-bubbles veil the stones grey-green as it thickens
    // toward the water mesh (clip t = 3.2 s: contrast about -40 % under 3-4 mm of backwash)
    float veil = film * smoothstep(0.0004, 0.0036, dSmooth) * VEIL_MAX;
    col = mix(col, VEIL_ALB * (uSunColor * (max(L.y, 0.0) / PI) + skyAmbient(Nb)), veil);
  }

  // ---- point glints
  vec3 Hm = normalize(V + L);
  float align = smoothstep(0.80, 0.97, dot(Nb, Hm));    // where wet facets can mirror the sun
  // glitter band: pebble crowns and menisci breaking a thin film just inside its edge
  // (only where the film edge has just passed: 10-25 cm behind a retreating edge, not over the
  // film the pebbles retain for seconds after it)
  float band = smoothstep(0.0002, 0.0009, hS) * (1.0 - smoothstep(0.004, 0.010, hS)) * (1.0 - foam)
             * (1.0 - smoothstep(0.2, 0.45, since));
  float gl = 0.0;
  if (band > 0.0 && dist < 9.0) gl += band * sparkle(P.xz, 0.012, 1250.0, 8.7, foot, 0.6) * (.72+.28*sin(uTime*5.0+P.x*21.0+P.z*17.0));
  // sparse sparkle of wet crowns on the exposed damp gravel (dry stones only show a few mica specks)
  if (dist < 9.0) gl += (1.0 - meshCover) * (1.0 - sub) * mix(1.0, 0.08, dryK) * sparkle(xz, 0.02, 125.0, 0.0, foot, 1.0) * shadow;
  col += uSunColor * GLINT_GAIN * gl * mix(0.35, 1.0, align);

  if (uBeachDebug.y > 0.5) { spec = vec3(0.0); col -= uSunColor * GLINT_GAIN * gl * mix(0.35, 1.0, align); }
  if (uBeachDebug.z > 0.5) { col = (uBeachDebug.z > 1.5 ? albLo : alb) * 0.95; spec = vec3(0.0); }
  if (uBeachDebug.w > 0.5) { col = vec3(sub, gloss * (1.0 - sub), band) * 0.5 + vec3(film * 0.3) + vec3(dryK * 0.4, dryK * 0.2, 0.0); spec = vec3(0.0); }
  vec3 outc = col + spec;
  float Yo = max(lum(outc), 0.0);
  // measured at every exposed time: the gravel's chroma rises toward the camera (R-B 17 at z 0.7 m ->
  // 34 at z 1.9 m): mostly the time since the film left (greyW above), plus a coarser, pinker lag near
  // the top of the swash and finer greyer grit below
  float chZ = clamp(0.98 + 0.12 * (P.z - 1.82), 0.45, 1.0);
  outc = max(Yo + (outc - Yo) * mix(1.0, chZ, (1.0 - sea) * (1.0 - uBeachDebug.w)), 0.0);
  // aerial perspective far along the beach (km): the land fades into the horizon haze
  float haze = landHaze(dist);
  if (haze > 0.0) outc = mix(outc, landHazeColor(-V), haze);
  gl_FragColor = vec4(outc, 1.0);
}`;

function buildGeometry(cols) {
  const rows = [];
  let z = 3.4;
  let s = 0.02;
  while (z > -60) {
    rows.push(z);
    z -= s;
    if (z < 0.6) s = Math.min(s * (z > -8 ? 1.012 : 1.04), z > -8 ? 0.12 : 1.5);
  }
  rows.push(-60);
  const nr = rows.length;
  const pos = new Float32Array(nr * cols * 3);
  let k = 0;
  for (let j = 0; j < nr; j++) {
    const hw = halfWidthAt(rows[j]) + 0.4;
    for (let i = 0; i < cols; i++) {
      pos[k * 3] = -hw + (2 * hw * i) / (cols - 1);
      pos[k * 3 + 2] = rows[j];
      k++;
    }
  }
  const idx = new Uint32Array((nr - 1) * (cols - 1) * 6);
  let t = 0;
  for (let j = 0; j < nr - 1; j++) for (let i = 0; i < cols - 1; i++) {
    const a = j * cols + i, b = a + 1, c = a + cols, d = c + 1;
    idx[t++] = a; idx[t++] = b; idx[t++] = c; idx[t++] = b; idx[t++] = d; idx[t++] = c;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5);
  return g;
}

export class Beach {
  constructor(renderer, shared) {
    const aniso = renderer.capabilities.getMaxAnisotropy();
    if (!EXPLORE) {
    // pass 1: stones -> A (albedo, height) half-float + C (roughness, crown, flags) 8-bit
    this.stoneRT = new THREE.WebGLRenderTarget(TILE_RES, TILE_RES, {
      count: 2, type: THREE.HalfFloatType, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter,
      wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping, generateMipmaps: true, depthBuffer: false,
    });
    this.stoneRT.textures[1].type = THREE.UnsignedByteType;
    for (const t of this.stoneRT.textures) {
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.colorSpace = THREE.NoColorSpace;
      t.anisotropy = aniso;
    }
    // pass 2: normals + AO
    this.nrmRT = new THREE.WebGLRenderTarget(TILE_RES, TILE_RES, {
      type: THREE.UnsignedByteType, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter,
      wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping, generateMipmaps: true, depthBuffer: false,
    });
    this.nrmRT.texture.colorSpace = THREE.NoColorSpace;
    this.nrmRT.texture.anisotropy = aniso;
    this.rebake(renderer);
    }

    const prelude = glslDefines() + NOISE + BED + SKY + OPTICS + DEPTH + BREAKER + CHOP + SWE_SAMPLE + SWASH_FAR + MACRO_NOISE_GLSL + HAZE_GLSL + (EXPLORE ? BEND_GLSL + COAST_SHADOW : '');
    const uFilmTau = shared.uSweFilmTau || { value: 2.6 };
    this.macroNoise = makeMacroNoise();
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        ...shared,
        ...(EXPLORE ? {...coastUniforms(shared),...photoUniforms()} : {}),
        uBeachFocus: shared.uFocus || { value: new THREE.Vector2(0, 4.1) },
        uBeachFilmTau: uFilmTau,
        uTexA: { value: this.stoneRT?.textures[0] },
        uTexB: { value: this.nrmRT?.texture },
        uTexC: { value: this.stoneRT?.textures[1] },
        uBeachGain: { value: 1.0 },
        uSeabed: { value: new THREE.Vector3(0.45, 0.43, 0.37) },
        uBeachDebug: { value: new THREE.Vector4(0, 0, 0, 0) },
        uDry: { value: new THREE.Vector4(1.7, 0.7, 0.14, 0.7) },
        uMacroNoise: { value: this.macroNoise },
      },
      vertexShader: prelude + (EXPLORE ? 'uniform vec2 uBeachFocus;\n' + TERRAIN_GLSL + VERT_EXPLORE : VERT_CLIP),
      fragmentShader: prelude + (EXPLORE ? SAND_FRAG : FRAG),
    });
    this.mesh = new THREE.Mesh(EXPLORE ? buildExploreGeometry() : buildGeometry(300), this.material);
    this.mesh.frustumCulled = false;
    if (EXPLORE) {
      this.rocks=new CoastalRocks(shared);
      this.mesh.add(this.rocks.mesh);
      // the promenade seawall closing the beach behind the storm berm
      this.promenade = new Promenade(shared, this.macroNoise);
      this.mesh.add(this.promenade.mesh);
      // the city, the palms of the promenade and the hills around the bay (Backdrop.js)
      this.backdrop = new Backdrop(shared);
      this.mesh.add(this.backdrop.group);
    }
  }

  async loadAssets(renderer) {
    if (EXPLORE) await Promise.all([loadCoastTextures(renderer,'sand',this.material.uniforms),this.rocks.loadAssets(renderer)]);
  }

  /** (Re)bake the pebble tile; params override BAKE_DEFAULTS (pebbleBake.js) - used while tuning. */
  rebake(renderer, params = {}) {
    const bake = new FullscreenPass(makeShader(BAKE_STONES, { uZero: { value: 0 }, ...bakeUniforms(params) }));
    bake.render(renderer, this.stoneRT);
    const bakeN = new FullscreenPass(makeShader(BAKE_NORMALS, { uA: { value: this.stoneRT?.textures[0] }, uZero: { value: 0 } }));
    bakeN.render(renderer, this.nrmRT);
    renderer.setRenderTarget(null);
    bake.material.dispose();
    bakeN.material.dispose();
    for (const p of [bake, bakeN]) { const i = FullscreenPass.all.indexOf(p); if (i >= 0) FullscreenPass.all.splice(i, 1); }
  }

  /** Height of the rendered ground at (x, z) (explore terrain incl. the upper-beach swells and
   *  storm scarps; the swash-face drape of a few cm is left out) - for the player's eye height. */
  groundAt(x, z) { return EXPLORE ? Math.max(groundHeightJS(x, z, bedProfileJS, this.macroNoise),rockTopAt(x,z)) : bedProfileJS(z); }
}
