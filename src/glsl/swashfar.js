import { CONFIG } from '../config.js';

// Far-field swash (explore mode): outside the scrolling shallow-water window the swash is
// rendered from a cheap analytic model instead (math doc §18-§24, calibrated on the solver).
// OWNED BY THE SWASH MODULE.
//
// Model, per along-shore position x and per breaker event (impact time t_i(x), run-up R_max(x),
// landing line z_I(x), energy E(x), evaluated with the breaker's own along-crest timing), with
// tau = t - t_i:
//   bore      the collapsed crest runs from the landing line to the shoreline, reaching it at
//             tau_c = 1.2 s (x sqrt(H / 0.45)): S = z_I (1 - tau / tau_c)^1.5
//   uprush    ballistic: R(s) = U0 s - a s^2 / 2 (s = tau - tau_c), a = 1.3 m/s^2, U0 = sqrt(2 a R_max);
//             R_max = 4.2 H + 0.35 (x bore gain^0.7, lobes and along-shore undulation per event):
//             2.2 m for the clip's waves, reached ~3 s after the plunge (solver median 2.2 m / 2.8 s)
//   backwash  the film edge retreats to the saturated lower face: E = z_r + (R_max - z_r) exp(-(s'/1 s)^1.5)
//   lens      a sheet from the step toe to the front, h = h0(tau) (1 - q)^p (1 - q^6), q = (z - z_toe) / (S - z_toe),
//             p 0.6 -> 0.75 (blunt bore -> tapered sheet), h0 0.22 m -> 3.5 cm; seaward the bore level
//             (0.17 m at the shoreline) spreads back over the step and a small drawdown follows the backwash
//   foam      plunge-zone whitewater, the bore blanket (tau ~0.5 s) and a roller at the advancing front;
//             lace and milk live on the wet film (die when it drains), turbulence in the lens
//   wetness   film = 1 under water, exp(-since / 2.6 s) after the edge has passed; damp over the run-up
//             zone (tau 55 s) and the capillary fringe of the face
// A newer event's water hides (covers) the foam of the older ones, and a crest coming in hides the
// older foam on its rising face (as brkSweFoamKeep). The profiles were fitted to the solver's
// tau-aligned medians (h, v, R, G, M, K) over 18 waves, so the far field continues the simulated
// swash statistically when seen along the beach.
//
// SwashSim evaluates the model once per simulation step into three textures (FAR_BAKE_MODEL below)
// over x = window centre +- 128 m (asinh-warped: 3.6 cm texels near the window, ~10 cm at 30 m,
// ~40 cm at 128 m) and z = -4.1 .. 6 m (3 cm rows), so a consumer only samples textures:
//   float swashFarWeight(vec2 xz)        0 inside the SWE window .. 1 beyond it (smooth blend)
//   vec4  swashFar(vec2 xz, float t)     (h depth m, u, v velocity m/s, w surface elevation m) like uSweView
//   vec4  swashFarFoam(vec2 xz, float t) (whitewater, lace, turbulence, milk) like uSweFoam
//   vec2  swashFarWet(vec2 xz, float t)  (film, damp) like uSweWet
//   vec4  swashFarLace(vec2 xz, float t) advected-lace offsets like uSweLace (the pattern rides the flow)
//   vec2  swashFarGrad(vec2 xz, float t) d(w)/d(x, z) of the far-field surface (4 extra fetches)
//   void  swashFarAll(xz, t, out view, out foam, out wet, out lace)
// (t is accepted for the interface; the textures hold the state of the last simulation step.)
// Blended helpers (the solver inside its window, the far field beyond it):
//   vec4  sweBlendView(vec2 xz, float t, out float wAll)  (h, u, v, w) x wAll; wAll = coverage of either
//   vec4  sweBlendFoam(vec2 xz, float t)   vec2 sweBlendWet(vec2 xz, float t)   vec4 sweBlendLace(vec2 xz, float t)
// Requires glslDefines() (bedProfile); the blended helpers also need SWE_SAMPLE. Include-guarded.
// In clip mode (no ?explore) only trivial stubs are emitted and SWASH_FAR_ON is not defined.

const STUB = /* glsl */ `
#ifndef SWASH_FAR_CORE_DEF
#define SWASH_FAR_CORE_DEF 1
float swashFarWeight(vec2 xz) { return 0.0; }
vec4 swashFar(vec2 xz, float t) { float B = bedProfile(xz.y); return vec4(max(-B, 0.0), 0.0, 0.0, max(B, 0.0)); }
vec4 swashFarFoam(vec2 xz, float t) { return vec4(0.0); }
vec2 swashFarWet(vec2 xz, float t) { return vec2(0.0, 1.0 - smoothstep(0.35, 0.75, bedProfile(xz.y))); }
vec4 swashFarLace(vec2 xz, float t) { return vec4(0.0); }
vec2 swashFarGrad(vec2 xz, float t) { return vec2(0.0); }
void swashFarAll(vec2 xz, float t, out vec4 view, out vec4 foam, out vec2 wet, out vec4 lace) {
  view = swashFar(xz, t); foam = vec4(0.0); wet = swashFarWet(xz, t); lace = vec4(0.0);
}
#endif
`;

// mapping between world xz and the far-field textures (shared by the bake and the samplers)
const MAP = /* glsl */ `
uniform vec4 uSwashFarMap;        // texture centre x, asinh scale L (m), half range in g (m), enabled
uniform vec4 uSwashFarMapZ;       // z of the first row's centre, row spacing (m), rows, columns
uniform vec4 uSwashFarWin;        // SWE window centre x, blend start / end (|x - centre|), enabled
uniform vec2 uSwashFocus;         // continuous player position, independent of solver cell scrolling
float swashFarU(float x) {
  float g = uSwashFarMap.y * asinh((x - uSwashFarMap.x) / uSwashFarMap.y);
  return 0.5 + 0.5 * g / uSwashFarMap.z;
}
float swashFarXofColumn(float i) {   // centre of column i
  float g = ((i + 0.5) / uSwashFarMapZ.w - 0.5) * 2.0 * uSwashFarMap.z;
  return uSwashFarMap.x + uSwashFarMap.y * sinh(g / uSwashFarMap.y);
}
vec2 swashFarUV(vec2 xz) { return vec2(swashFarU(xz.x), (xz.y - uSwashFarMapZ.x) / (uSwashFarMapZ.y * uSwashFarMapZ.z) + 0.5 / uSwashFarMapZ.z); }
float swashVisualFocusX() {
  // A paused solver cannot follow an arbitrarily travelling camera. Keep
  // its visible overlap inside the region the far bake actually prepares.
  return clamp(uSwashFocus.x, uSwashFarWin.x - 1.1, uSwashFarWin.x + 1.1);
}
float swashFarWeight(vec2 xz) {
  // The solver moves by whole cells about once per metre. Its allocation
  // centre must not drive the visible near/far handoff: that stepped foam,
  // water height and wetness ahead of a moving player. Keep the normal band
  // inside the fully valid solver cells even just before a scroll.
  float visible = smoothstep(uSwashFarWin.y, uSwashFarWin.z, abs(xz.x - swashVisualFocusX()));
  // If the camera travels while the waves are paused, the solver stays put.
  // Retire its field before the 0.884 m allocation-edge fringe in that case.
  float bounds = smoothstep(9.95, 10.14, abs(xz.x - uSwashFarWin.x));
  return uSwashFarWin.w * max(visible, bounds);
}
`;

const FULL = /* glsl */ `
#ifndef SWASH_FAR_CORE_DEF
#define SWASH_FAR_CORE_DEF 1
#define SWASH_FAR_ON 1
${MAP}
uniform sampler2D uSwashFarView;  // (h, u, v, w)
uniform sampler2D uSwashFarFoamT; // (R, G, K, M)
uniform sampler2D uSwashFarWetT;  // (film, damp, pattern displacement z, 0)
float swashFarInZ(vec2 uv) { return step(0.0, uv.y) * step(uv.y, 1.0); }
vec4 swashFar(vec2 xz, float t) {
  vec2 uv = swashFarUV(xz);
  vec4 result = vec4(0.0);
  if (swashFarInZ(uv) < 0.5) { float B = bedProfile(xz.y); result = vec4(max(-B, 0.0), 0.0, 0.0, max(B, 0.0)); }
  else result = textureLod(uSwashFarView, uv, 0.0);
  return result;
}
vec4 swashFarFoam(vec2 xz, float t) {
  vec2 uv = swashFarUV(xz);
  return textureLod(uSwashFarFoamT, uv, 0.0) * swashFarInZ(uv);
}
vec2 swashFarWet(vec2 xz, float t) {
  vec2 uv = swashFarUV(xz);
  vec2 result = vec2(0.0);
  if (swashFarInZ(uv) < 0.5) result = vec2(0.0, 1.0 - smoothstep(0.35, 0.75, bedProfile(xz.y)));
  else result = textureLod(uSwashFarWetT, uv, 0.0).xy;
  return result;
}
vec4 swashFarLace(vec2 xz, float t) {
  vec2 uv = swashFarUV(xz);
  float d = textureLod(uSwashFarWetT, uv, 0.0).z * swashFarInZ(uv);
  return vec4(0.0, -d, 0.0, -d);
}
vec2 swashFarGrad(vec2 xz, float t) {
  float ex = 0.05, ez = uSwashFarMapZ.y;
  float wE = swashFar(xz + vec2(ex, 0.0), t).w, wW = swashFar(xz - vec2(ex, 0.0), t).w;
  float wN = swashFar(xz + vec2(0.0, ez), t).w, wS = swashFar(xz - vec2(0.0, ez), t).w;
  return vec2((wE - wW) / (2.0 * ex), (wN - wS) / (2.0 * ez));
}
void swashFarAll(vec2 xz, float t, out vec4 view, out vec4 foam, out vec2 wet, out vec4 lace) {
  vec2 uv = swashFarUV(xz);
  float inZ = swashFarInZ(uv);
  float B = bedProfile(xz.y);
  view = inZ > 0.5 ? textureLod(uSwashFarView, uv, 0.0) : vec4(max(-B, 0.0), 0.0, 0.0, max(B, 0.0));
  foam = textureLod(uSwashFarFoamT, uv, 0.0) * inZ;
  vec4 w = textureLod(uSwashFarWetT, uv, 0.0);
  wet = inZ > 0.5 ? w.xy : vec2(0.0, 1.0 - smoothstep(0.35, 0.75, B));
  lace = vec4(0.0, -w.z, 0.0, -w.z) * inZ;
}
#endif
`;

// Blended views for consumers (the solver inside its window, the far field beyond it).
const SWASH_FAR_BLEND = /* glsl */ `
#ifndef SWASH_FAR_BLEND_DEF
#define SWASH_FAR_BLEND_DEF 1
vec4 sweBlendView(vec2 xz, float t, out float wAll) {
  float wIn = sweInside(xz), wf = swashFarWeight(xz);
  vec4 s = textureLod(uSweView, sweUV(xz), 0.0);
  wAll = max(wIn, wf);
  if (wf <= 0.0) return s * wIn;
  return mix(s * wIn, swashFar(xz, t), wf);
}
vec4 sweBlendFoam(vec2 xz, float t) {
  float wIn = sweInside(xz), wf = swashFarWeight(xz);
  vec4 s = textureLod(uSweFoam, sweUV(xz), 0.0) * wIn;
  return wf <= 0.0 ? s : mix(s, swashFarFoam(xz, t), wf);
}
vec2 sweBlendWet(vec2 xz, float t) {
  float wIn = sweInside(xz), wf = swashFarWeight(xz);
  vec2 s = textureLod(uSweWet, sweUV(xz), 0.0).xy * wIn;
  return wf <= 0.0 ? s : mix(s, swashFarWet(xz, t), wf);
}
vec4 sweBlendLace(vec2 xz, float t) {
  float wIn = sweInside(xz), wf = swashFarWeight(xz);
  vec4 s = textureLod(uSweLace, sweUV(xz), 0.0) * wIn;
  return wf <= 0.0 ? s : mix(s, swashFarLace(xz, t), wf);
}
#endif
`;

export const SWASH_FAR_CORE = CONFIG.explore ? FULL : STUB;
export const FAR_MAP = MAP;
export const SWASH_FAR = SWASH_FAR_CORE + SWASH_FAR_BLEND;

// ---------------------------------------------------------------- the model (SwashSim bake pass)
// Evaluated per texel of the far-field textures (MRT: view, foam, wet). Requires glslDefines(),
// NOISE, BED. uSwashFarParam holds, per column (same x mapping) and remembered event (rows,
// oldest first): (t_i, R_max, z_I, E) - baked by SwashSim from the breaker events.
export const FAR_BAKE_MODEL = MAP + /* glsl */ `
uniform sampler2D uSwashFarParam;
uniform vec4 uSwashFarK;          // a (uprush deceleration), crossing time, backwash time scale, residual-film line
uniform vec4 uSwashFarK2;         // bore level at the shoreline, residual film depth, film tau, step toe z
uniform float uTime;
uniform int uFarSlots;
layout(location = 1) out vec4 outFoam;
layout(location = 2) out vec4 outWet;
// approach of the next crest (for hiding the older foam on its rising face, like brkSweFoamKeep):
// apex position relative to the landing line and front-face length vs normalised time (BK_TABLE fit)
float sfApexZ(float tn) { float d = max(-0.33 - tn, 0.0); return -0.48 - 3.3 * d * (1.0 + 0.12 * d); }
float sfFaceL(float tn) { return mix(0.4, 4.0, smoothstep(-0.4, -2.0, tn)); }
void main() {
  ivec2 ij = ivec2(gl_FragCoord.xy);
  float x = swashFarXofColumn(float(ij.x));
  float z = uSwashFarMapZ.x + float(ij.y) * uSwashFarMapZ.y;
  vec2 xz = vec2(x, z);
  float t = uTime;
  float B = bedProfile(z);
  // (inside the solver window, where the far field is never shown, only the quiescent state)
  if (abs(x - uSwashFarWin.x) < uSwashFarWin.y - 1.4) {
    gl_FragColor = vec4(max(-B, 0.0), 0.0, 0.0, max(B, 0.0));
    outFoam = vec4(0.0); outWet = vec4(0.0, 1.0 - smoothstep(0.35, 0.75, B), 0.0, 0.0);
    return;
  }
  float a = uSwashFarK.x, zr0 = uSwashFarK.w, zt = uSwashFarK2.w;
  // ragged 1-5 cm fringe of the fronts, and a 0.3-1 m meander (the finer detail fades out where the
  // columns are too coarse to carry it: sampled per 10-40 cm column it aliases into a regular sawtooth)
  float dxc = swashFarXofColumn(float(ij.x) + 0.5) - swashFarXofColumn(float(ij.x) - 0.5);
  float rag = 0.022 * gnoise(vec2(x * 13.0, t * 0.6)) * (1.0 - smoothstep(0.1, 0.22, dxc))
            + 0.012 * gnoise(vec2(x * 41.0 + 3.7, t * 1.3)) * (1.0 - smoothstep(0.03, 0.07, dxc))
            + 0.05 * gnoise(vec2(x * 2.3 + 1.9, t * 0.15)) * (1.0 - smoothstep(0.6, 1.2, dxc));
  float h = 0.0, vS = 0.0, wS = 1.0e-9, cov = 0.0, since = 1.0e3, keepF = 1.0, dS = 0.0, dW = 0.0;
  vec4 F = vec4(0.0);
  // newest event first (rows are sorted by time)
  for (int k = uFarSlots - 1; k >= 0; k--) {
    vec4 P = texelFetch(uSwashFarParam, ivec2(ij.x, 2 * k), 0);
    float tau = t - P.x, Rm = P.y, zI = P.z, E = P.w;
    if (E <= 1.0e-4 || Rm <= 0.01) continue;
    float rs = clamp(pow(E, 0.25), 0.6, 1.3);
    if (tau < 0.0) {
      // the next crest is coming in: its rising face renews the surface (no older foam on it)
      float tn = tau / rs;
      if (tn > -2.4) {
        float zc = zI + sfApexZ(tn) * rs * rs, L = sfFaceL(tn) * rs * rs;
        float face = smoothstep(zc - 0.8, zc - 0.2, z) * (1.0 - smoothstep(zc + 0.6 * L, zc + L, z));
        keepF = min(keepF, 1.0 - face * smoothstep(-2.4, -1.5, tn) * step(z, 0.6));
      }
      continue;
    }
    if (tau > 16.0 || (z > Rm + 0.3 && z > 0.3)) continue;   // (beyond this event's reach)
    float U0 = sqrt(2.0 * a * Rm);
    float tc = uSwashFarK.y * rs, tup = U0 / a, tm = tc + tup;
    float zr = min(zr0, 0.8 * Rm);
    bool adv = tau < tm;
    float S;
    if (tau < tc) S = zI * pow(1.0 - tau / tc, 1.5);
    else if (adv) { float s = tau - tc; S = U0 * s - 0.5 * a * s * s; }
    else S = zr + (Rm - zr) * exp(-pow((tau - tm) / uSwashFarK.z, 1.5));
    S += rag * smoothstep(-0.3, 0.3, S);
    // ---- water: the lens over the face (from the step toe) and the bore level seaward of the shoreline
    float hb = uSwashFarK2.x * rs * rs;
    float rise = smoothstep(tc - 0.7 * rs, tc + 0.1 * rs, tau);
    float sheet = (0.035 + 0.19 * exp(-max(tau - tc - 0.3, 0.0) / 1.4)) * rs * (adv ? rise : 0.4 + 0.6 * exp(-(tau - tm) / 0.8));
    float Sp = max(S - zt, 1.0e-3);
    float q = clamp((z - zt) / Sp, 0.0, 1.0);
    float p = adv ? 0.6 + 0.15 * smoothstep(0.0, 0.5, (tau - tc) / tup) : 0.9;
    float lens = (z < S && z >= zt && S > zt) ? sheet * pow(1.0 - q, p) * (1.0 - pow(q, 6.0)) : 0.0;
    float eta;
    if (tau < tc) eta = hb * smoothstep(S + 0.12, S - 0.2, z) * exp(min(z - S, 0.0) / 0.9);
    else {
      float lvl = hb * rise * exp(-max(tau - tc - 0.3 * rs, 0.0) / 0.9);
      eta = lvl * exp(min(z, 0.0) / 1.1) - (adv ? 0.0 : 0.03 * rs * smoothstep(0.0, 0.8, tau - tm)) * exp(min(z, 0.0) / 0.7);
    }
    float he = max(lens, z < 0.3 ? eta - B : 0.0);
    // ---- velocity: ~1.2 m/s behind the advancing front (the thinning tip is slower), linear from
    // the toe; the backwash accelerates to ~1.4 m/s at the step; weak undertow seaward of the toe
    float ve;
    if (tau < tc) ve = -zI * 1.5 / tc * sqrt(max(1.0 - tau / tc, 0.0)) * smoothstep(zI - 0.5, S, z) * step(z, S + 0.05);
    else if (adv) {
      float s = tau - tc;
      float vf = 1.2 * rs * (1.0 - pow(s / tup, 3.0)) + 0.4 * max(U0 - a * s - 1.2 * rs, 0.0);
      ve = vf * pow(clamp((z - zt) / Sp, 0.0, 1.0), 0.8) * step(z, S + 0.05) - 0.12 * step(z, zt);
    } else {
      float Vb = 1.4 * rs * (1.0 - exp(-(tau - tm) / 0.6));
      ve = -Vb * pow(clamp(1.0 - (z - zt) / Sp, 0.0, 1.0), 0.7) * step(zt - 0.2, z) * step(z, S) - 0.12 * step(z, zt);
    }
    float keep = 1.0 - cov;
    h = max(h, he);
    float w2 = he * he * (keep + 0.05);
    vS += w2 * ve; wS += w2;
    float cover = adv ? smoothstep(S + 0.02, S - 0.25, z) : smoothstep(S + 0.02, S - 0.1, z);
    if (tau < tc) cover *= step(zI - 0.8, z);
    // pattern displacement: the foam rides up with the uprush and back down with the backwash
    float Dz = adv ? 0.8 * (S - Rm) : S - Rm;
    dS += Dz * cover * keep; dW += cover * keep;
    // ---- arrival of the front at z and the time the film edge left it (relative to t_i)
    float tArr = z < 0.0 ? tc * (1.0 - pow(clamp(z / min(zI, -0.1), 0.0, 1.0), 0.6667))
                         : tc + (U0 - sqrt(max(U0 * U0 - 2.0 * a * clamp(z, 0.0, 0.999 * Rm), 0.0))) / a;
    float tLeft = z > zr ? tm + uSwashFarK.z * pow(-log(clamp((z - zr) / max(Rm - zr, 1.0e-3), 1.0e-6, 1.0)), 0.6667) : 1.0e9;
    bool reached = z < Rm && tau > tArr;
    if (reached) since = min(since, max(tau - tLeft, 0.0));
    // ---- foam: patchy along the beach (0.5-4 m clumps of each event's whitewater, riding the flow)
    float sd = texelFetch(uSwashFarParam, ivec2(ij.x, 2 * k + 1), 0).x * 37.0;
    vec2 pq = vec2(x / 2.2, (z - Dz) / 0.8) + vec2(sd, -sd * 0.7);
    float patchF = smoothstep(-0.6, 0.6, gnoise(pq) + 0.55 * gnoise(pq * 2.4 + 5.1));
    float patchP = smoothstep(-0.7, 0.7, gnoise(vec2(x / 3.1 + sd * 1.3, tau * 0.25)));
    float ageF = max(tau - tArr, 0.0);
    float zp = zI + 0.15 + 0.1 * min(tau, 2.0);
    float dp = (z - zp) / (0.6 + 0.15 * tau);
    float Rp = 2.3 * mix(0.45, 1.2, patchP) * exp(-max(tau - 1.2, 0.0) / 0.9) * smoothstep(0.1, 0.9, tau) * exp(-0.5 * dp * dp);
    // (the bore foam builds up over ~0.35 s after the impact, as the solver's: switched on at the
    // impact it grows along the peeling crest as a slab with a square end)
    float Rb = 1.25 * mix(0.3, 1.3, patchF) * exp(-max(tau - tc - 0.5 * rs, 0.0) / 0.5) * cover * smoothstep(-0.95, -0.65, z) * smoothstep(0.0, 0.35, tau);
    float df = (S - z) / 0.4;
    float Rf = adv ? 0.8 * mix(0.6, 1.15, patchF) * exp(-df * df) * cover * smoothstep(0.0, 0.3, tau - tc + 0.3) * (1.0 - smoothstep(0.6, 1.0, (tau - tc) / tup)) : 0.0;
    // backwash toe: the sheet plunges under the still water at the step - a turbulent, slightly
    // foamy band that comes and goes along the beach (swash §7)
    float zj = zt - 0.12 + 0.08 * gnoise(vec2(x * 1.7 + sd, tau * 0.8));
    float dj = (z - zj) / 0.1;
    float Rt = adv ? 0.0 : 0.45 * smoothstep(0.2, 1.0, (tau - tm) / 0.6) * smoothstep(0.35, 0.8, patchP + 0.3) * exp(-dj * dj);
    float Eg = sqrt(min(E, 1.5));
    float lg = (z - zp) / 1.0;
    float G = max(1.3 * mix(0.45, 1.2, patchF) * smoothstep(0.0, 0.5, ageF) * exp(-max(tau - tc - 1.6 * rs, 0.0) / 2.2) * ((reached && tau < tLeft + 0.3) ? 1.0 : 0.0),
                  1.8 * mix(0.5, 1.1, patchP) * smoothstep(0.2, 0.8, tau) * exp(-max(tau - 1.5, 0.0) / 2.2) * exp(-lg * lg));
    float sea = smoothstep(zI - 1.4, zI - 0.2, z);
    float Mk = (0.25 + 0.55 * smoothstep(0.0, 0.4, ageF) * exp(-max(tau - tc - 1.2 * rs, 0.0) / 1.25)) * max(reached ? 1.0 : 0.0, step(z, 0.0) * sea);
    float Kp = 1.6 * exp(-max(tau - 1.3, 0.0) / 1.2) * smoothstep(0.1, 0.9, tau) * exp(-0.5 * (z - zp) * (z - zp) / 0.36);
    float Kl = 2.8 * (adv ? exp(-max(tau - tc - 1.4, 0.0) / 1.2) : 0.55 + 0.45 * exp(-(tau - tm))) * cover * step(zt - 0.3, z);
    Kl = max(Kl, 2.0 * Rt);
    F = max(F, vec4(E * (Rp + Rb + Rf) + Eg * Rt, min(G * Eg, 2.0), max(Kp, Kl) * Eg, Mk * sqrt(min(E, 1.2))) * keep);
    cov = max(cov, cover);
  }
  // residual film on the saturated lower face (the water table follows the mean swash level)
  h = max(h, uSwashFarK2.y * smoothstep(zr0 + 0.1, zr0 - 0.3, z) * step(-0.3, z));
  float dr = smoothstep(0.0005, 0.002, h);
  float film = h > 0.001 ? 1.0 : exp(-since / uSwashFarK2.z);
  float damp = max(h > 0.001 ? 1.0 : exp(-since / 55.0), 1.0 - smoothstep(0.35, 0.75, B));
  // surface: thin films drape the rendered relief (as the solver's view pass), deep water is level.
  // Well beyond the solver window (> 12-18 m from its centre: seen from afar) dry cells and films
  // under ~5 mm sit a few cm below the bed: the coarse water mesh cannot follow the pebble relief
  // between its vertices and would poke through it in grey dashes (the beach draws those films' gloss
  // itself, from the depth and wetness channels); faded in gradually, no seam at the window's edge
  float lowF = (1.0 - smoothstep(0.002, 0.008, h)) * smoothstep(12.0, 18.0, abs(x - swashVisualFocusX()));
  float w = mix(bedHeight(xz) + h, B + h, smoothstep(0.04, 0.12, h)) - 0.03 * lowF;
  gl_FragColor = vec4(h, 0.1 * smoothstep(0.003, 0.02, h), vS / wS, w);
  // (faded out toward the seaward end of the texture: nothing is cut off at its edge)
  outFoam = vec4(F.x * keepF, F.y * keepF, F.z, F.w * keepF) * dr * smoothstep(uSwashFarMapZ.x, uSwashFarMapZ.x + 0.6, z);
  outWet = vec4(film, damp, dS / max(dW, 1.0e-3) * min(dW, 1.0), 0.0);
}`;
