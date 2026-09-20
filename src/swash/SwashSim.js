import * as THREE from 'three';
import { CONFIG, glslDefines } from '../config.js';
import { NOISE, BED } from '../glsl/common.js';
import { BREAKER } from '../glsl/breaker.js';
import { SWASH_FAR_CORE, FAR_BAKE_MODEL, FAR_MAP } from '../glsl/swashfar.js';
import { FullscreenPass, makeShader, floatRT, PingPong } from '../core/gpu.js';
import { RockWetness } from '../beach/RockWetness.js';

// GPU shallow-water solver for the surf/swash zone (math doc §15-§28).
//
// Scheme: Kurganov & Petrova (2007) second-order well-balanced, positivity-preserving
// central-upwind finite volumes on a Cartesian grid. Conserved state U = (w, hu, hv)
// with w = h + B (surface elevation), piecewise-bilinear bed defined at cell corners,
// generalized-minmod reconstruction, desingularized velocities for wet/dry fronts.
// Per sim step: [CONFIG.swe.substeps x flux] -> sources (bore injection from the breaker
// model, Manning friction, infiltration into the pebbles above the water table, sponge)
// -> foam (blanket / lace / turbulence / milk + advected lace coordinates) -> wetness -> view
// -> height (total surface incl. breaker events) -> foam view (what renderers see: the foam under
// a breaking face is hidden).
//
// Swash kinematics are calibrated against the clip (timeline analysis §9, tools/swashprobe.mjs):
// each bore stalls against the previous backwash, the main uprush starts ~1.65 s after the
// plunge, runs out past the frame bottom (z ≈ 1.9) and drains back at ~1 m/s.
//
// Explore mode (CONFIG.explore): the domain is a window 22 m wide along-shore (same 3 cm cells, same
// extended dry reserve) that scrolls with the player: when the focus (shared.uFocus.x) is more than 1 m from its
// centre, every field (solver state, foam + lace coordinates, wetness) is shifted by whole cells;
// the columns entering at the edge continue the old edge column, varied along-shore like the
// analytic far field (glsl/swashfar.js). The window's lateral edges are open (zero-gradient: nothing
// reflects off them), and beyond them the far field takes over for rendering (blended over 2 m,
// see swashFarWeight). The far field is driven by the breaker events: SwashSim remembers every
// event it has seen (the packed schedule drops an event shortly after its impact over the clip
// frame; the swash lasts much longer, and along a moving window a late section still injects),
// bakes each event's per-x parameters (impact time, run-up, landing line, energy) whenever the
// history or the window changes, and evaluates the far field into textures at <= 60 Hz. Clip mode
// is unchanged by all this (fixed 9.6 m window at x = 0, reflective walls, no far field).

const COMMON = glslDefines() + NOISE + BED;
const EXPLORE = !!CONFIG.explore;
const XDEF = EXPLORE ? '#define SWE_EXPLORE 1\n' : '';

const DOMAIN = /* glsl */ `
uniform vec4 uDom;    // xMin, zMin, dx, dz
uniform ivec2 uN;
vec2 cellXZ(ivec2 ij) { return vec2(uDom.x + (float(ij.x) + 0.5) * uDom.z, uDom.y + (float(ij.y) + 0.5) * uDom.w); }
`;

// The bed the flow sees. Only 2-D lumps of the pebble face (the z-invariant cusp ridges of
// bedRelief would act as fixed runnels: every wave's front would get the same saw-tooth, which the
// clip does not show; lobes are made per wave by the injection instead). The step knee meanders
// along-shore by up to +-16 cm (lambda 0.65-2.7 m; <= 5 cm of bed height, so thin films still drape
// the rendered bed), so the backwash jump at the toe is an irregular band, not a straight line.
const BED_INIT = COMMON + DOMAIN + /* glsl */ `
in vec2 vUv;
uniform float uRelief;   // fraction of the pebble-face lumps the flow feels (the rest is roughness)
float sweBed(vec2 xz) {
  float face = smoothstep(-1.2, 0.2, xz.y) * smoothstep(5.0, 2.5, xz.y);
  float lumps = 0.010 * gnoise(xz * 1.3 + 3.1) + 0.005 * gnoise(xz * 4.1 - 1.3);
  float cusp = 0.012 * sin(xz.x * 5.2 + 0.8 * sin(xz.x * 1.7)) + 0.010 * sin(xz.x * 14.5 + 1.3);
  float knee = smoothstep(-1.9, -1.1, xz.y) * (1.0 - smoothstep(-0.2, 0.4, xz.y));
  float shift = knee * 0.16 * (0.5 * sin(xz.x * 2.3 + 1.0) + 0.3 * sin(xz.x * 5.1 + 2.0) + 0.2 * sin(xz.x * 9.7 + 0.4));
  return max(bedProfile(xz.y - shift) + uRelief * (lumps + 0.15 * cusp) * face,rockBedHeight(xz));
}
void main() {
  ivec2 ij = ivec2(gl_FragCoord.xy);
  vec2 c = cellXZ(ij);
  vec2 h = 0.5 * uDom.zw;
  float bNE = sweBed(c + vec2(h.x, h.y)), bNW = sweBed(c + vec2(-h.x, h.y));
  float bSE = sweBed(c + vec2(h.x, -h.y)), bSW = sweBed(c + vec2(-h.x, -h.y));
  float Bc = 0.25 * (bNE + bNW + bSE + bSW);
  // Hydrostatic interface crests are baked once (also when the window scrolls),
  // saving two extra bed fetches in every flux substep.
  float Bce=.25*(bNE+bSE+sweBed(c+vec2(3.0*h.x,h.y))+sweBed(c+vec2(3.0*h.x,-h.y)));
  float Bcn=.25*(bNE+bNW+sweBed(c+vec2(h.x,3.0*h.y))+sweBed(c+vec2(-h.x,3.0*h.y)));
  gl_FragColor = vec4(Bc,max(Bc,Bce),max(Bc,Bcn),0.0);
}`;

const STATE_INIT = DOMAIN + /* glsl */ `
uniform sampler2D uBed;
void main() {
  ivec2 ij = ivec2(gl_FragCoord.xy);
  float B = texelFetch(uBed, ij, 0).x;
  gl_FragColor = vec4(max(B, 0.0), 0.0, 0.0, 0.0);
}`;

const FLUX = XDEF + `#define G_ACC ${CONFIG.g.toFixed(2)}\n` + DOMAIN + /* glsl */ `
uniform sampler2D uState;
uniform sampler2D uBed;
uniform float uDt;
const float THETA = 1.3;
const float EPS4 = 1.0e-8;
const float VMAX = 7.5;

// Boundaries: reflective walls at the beach top and the lateral edges (mirrored ghost
// cells with the normal momentum negated), open zero-gradient + sponge at the seaward edge.
// The scrolling window of explore mode is open (zero-gradient) at its lateral edges too: the
// swash beyond them is the far field, water flows out (and in) freely.
ivec2 ghost(ivec2 p) {
  ivec2 q = p;
#ifndef SWE_EXPLORE
  if (q.x < 0) q.x = -q.x - 1; else if (q.x >= uN.x) q.x = 2 * uN.x - q.x - 1;
#endif
  if (q.y >= uN.y) q.y = 2 * uN.y - q.y - 1;
  return clamp(q, ivec2(0), uN - 1);
}
vec3 St(ivec2 p) {
  vec3 U = texelFetch(uState, ghost(p), 0).xyz;
#ifndef SWE_EXPLORE
  if (p.x < 0 || p.x >= uN.x) U.y = -U.y;
#endif
  if (p.y >= uN.y) U.z = -U.z;
  return U;
}
vec3 Bd(ivec2 p) { return texelFetch(uBed, ghost(p), 0).xyz; }

float mm3(float a, float b, float c) {
  if (a > 0.0 && b > 0.0 && c > 0.0) return min(a, min(b, c));
  if (a < 0.0 && b < 0.0 && c < 0.0) return max(a, max(b, c));
  return 0.0;
}
vec3 mmv(vec3 l, vec3 c, vec3 r) {
  vec3 a = THETA * (c - l), b = 0.5 * (r - l), d = THETA * (r - c);
  return vec3(mm3(a.x, b.x, d.x), mm3(a.y, b.y, d.y), mm3(a.z, b.z, d.z));
}
// Limited reconstruction with a hydrostatic fallback at wet/dry interfaces
void recon(vec3 Ul, vec3 Uc, vec3 Ur, float Bm, float Bp, out vec3 Um, out vec3 Up) {
  vec3 s = mmv(Ul, Uc, Ur);
  Up = Uc + 0.5 * s; Um = Uc - 0.5 * s;
  // Hydrostatic first-order fallback only where a reconstructed face dries.
  // Preserve the neighbouring wet face's free-surface elevation instead of
  // reflecting it around the cell average (which creates flow at a resting rock).
  if (Up.x < Bp || Um.x < Bm) {
    Up=Uc;Um=Uc;
  }
}
struct Face { float w, h, u, v; };
Face mkFace(vec3 U, float B) {
  Face f; f.h = max(U.x - B, 0.0); f.w = B + f.h;
  float h4 = f.h * f.h * f.h * f.h;
  float den = sqrt(h4 + max(h4, EPS4));
  f.u = clamp(1.41421356 * f.h * U.y / den, -VMAX, VMAX);
  f.v = clamp(1.41421356 * f.h * U.z / den, -VMAX, VMAX);
  return f;
}
// central-upwind numerical flux; dir = 0 for x faces (normal velocity u), 1 for z faces (v)
vec3 kpFlux(Face L, Face R, int dir) {
  float unL = dir == 0 ? L.u : L.v, unR = dir == 0 ? R.u : R.v;
  float cL = sqrt(G_ACC * L.h), cR = sqrt(G_ACC * R.h);
  float ap = max(max(unL + cL, unR + cR), 0.0);
  float am = min(min(unL - cL, unR - cR), 0.0);
  float d = ap - am;
  if (d < 1.0e-7) return vec3(0.0);
  float pL = 0.5 * G_ACC * L.h * L.h, pR = 0.5 * G_ACC * R.h * R.h;
  vec3 FL, FR;
  if (dir == 0) {
    FL = vec3(L.h * L.u, L.h * L.u * L.u + pL, L.h * L.u * L.v);
    FR = vec3(R.h * R.u, R.h * R.u * R.u + pR, R.h * R.u * R.v);
  } else {
    FL = vec3(L.h * L.v, L.h * L.u * L.v, L.h * L.v * L.v + pL);
    FR = vec3(R.h * R.v, R.h * R.u * R.v, R.h * R.v * R.v + pR);
  }
  vec3 UL = vec3(L.w, L.h * L.u, L.h * L.v), UR = vec3(R.w, R.h * R.u, R.h * R.v);
  return (ap * FL - am * FR) / d + (ap * am / d) * (UR - UL);
}

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 ex = ivec2(1, 0), ez = ivec2(0, 1);
  vec3 C = St(p), E = St(p + ex), W = St(p - ex), N = St(p + ez), S = St(p - ez);
  vec3 EE = St(p + 2 * ex), WW = St(p - 2 * ex), NN = St(p + 2 * ez), SS = St(p - 2 * ez);
  vec3 bC = Bd(p), bE = Bd(p + ex), bW = Bd(p - ex), bN = Bd(p + ez), bS = Bd(p - ez);
  vec3 bWW = Bd(p - 2 * ex), bSS = Bd(p - 2 * ez);
  // DRY_STENCIL_BEGIN
#ifdef SWE_EXPLORE
  // All nine reconstruction samples are dry: every hydrostatic flux is zero.
  // Skip the four Riemann solves over the dry reserve; wake immediately when
  // water reaches ANY sample, two cells before it can reach this cell.
  if(C.x<=bC.x+1e-7&&E.x<=bE.x+1e-7&&W.x<=bW.x+1e-7&&N.x<=bN.x+1e-7&&S.x<=bS.x+1e-7
    &&WW.x<=bWW.x+1e-7&&SS.x<=bSS.x+1e-7&&EE.x<=Bd(p+2*ex).x+1e-7&&NN.x<=Bd(p+2*ez).x+1e-7){
    gl_FragColor=vec4(bC.x,0.,0.,0.);return;
  }
#endif
  // DRY_STENCIL_END
  // face beds: .y = east face of that cell, .z = north face
  float BeC=bC.y,BwC=bW.y,BnC=bC.z,BsC=bS.z;

  vec3 UwC, UeC, UsC, UnC, UwE, UeE, UwW, UeW, UsN, UnN, UsS, UnS;
  recon(W, C, E, BwC, BeC, UwC, UeC);
  recon(C, E, EE, BeC, bE.y, UwE, UeE);
  recon(WW, W, C, bWW.y, BwC, UwW, UeW);
  recon(S, C, N, BsC, BnC, UsC, UnC);
  recon(C, N, NN, BnC, bN.z, UsN, UnN);
  recon(SS, S, C, bSS.z, BsC, UsS, UnS);

  vec3 HE = kpFlux(mkFace(UeC, BeC), mkFace(UwE, BeC), 0);
  vec3 HW = kpFlux(mkFace(UeW, BwC), mkFace(UwC, BwC), 0);
  vec3 GN = kpFlux(mkFace(UnC, BnC), mkFace(UsN, BnC), 1);
  vec3 GS = kpFlux(mkFace(UnS, BsC), mkFace(UsC, BsC), 1);

  // Well-balanced hydrostatic bed source.
  // Difference of hydrostatic face pressures: agrees with KP on fully wet
  // linear reconstructions, and remains balanced at a partially dry shoulder.
  vec4 h0=max(vec4(C.x)-vec4(BeC,BwC,BnC,BsC),vec4(0.0));
  float Sx=.5*G_ACC*(h0.x*h0.x-h0.y*h0.y)/uDom.z;
  float Sz=.5*G_ACC*(h0.z*h0.z-h0.w*h0.w)/uDom.w;

  vec3 U = C - uDt / uDom.z * (HE - HW) - uDt / uDom.w * (GN - GS) + uDt * vec3(0.0, Sx, Sz);
  float h = U.x - bC.x;
  if (h < 1.0e-5) { U = vec3(bC.x, 0.0, 0.0); }
  else {
    // cap velocities (desingularized) to keep the explicit step inside CFL
    float sp = length(U.yz) / h;
    if (sp > VMAX) U.yz *= VMAX / sp;
  }
  gl_FragColor = vec4(U, 0.0);
}`;

const SOURCES = XDEF + COMMON + DOMAIN + BREAKER + (EXPLORE ? SWASH_FAR_CORE : '') + /* glsl */ `
uniform sampler2D uState;
uniform sampler2D uBed;
uniform float uDtS;
uniform float uManning;
uniform float uFricH;
uniform float uInfil;
uniform vec2 uInfZ;      // bed elevation range over which infiltration ramps in (water table)
uniform float uInfLow;   // infiltration fraction of a residual thin film below the water table
uniform float uThinN;    // Manning n multiplier for films thinner than the pebbles
uniform vec4 uRetain;    // retained film: depth (m), fades out between these bed elevations (m), its soak-in time (s)
uniform float uEdgeNudge; // explore: relaxation (per step) of the outer metre of the window toward the far field
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 U = texelFetch(uState, p, 0).xyz;
  float B = texelFetch(uBed, p, 0).x;
  vec2 xz = cellXZ(p);
  // bore injection: the collapsing crest + plunging lip become shallow-water mass & momentum
  vec3 inj = brkInjection(xz, uTime);
  U.x += inj.x * uDtS;
  U.y += inj.z * uDtS;   // along-shore momentum (oblique bores drift +x)
  U.z += inj.y * uDtS;   // cross-shore (+z) momentum
  float h = U.x - B;
  if (h > 1.0e-4) {
    vec2 vel = U.yz / h;
    float sp = length(vel);
    // Manning bottom friction (§20), implicit: n^2 g |u| / h^(4/3); the depth floor keeps the
    // thin swash tip nearly ballistic (measured front deceleration ≈ g sinβ). A film thinner than
    // the pebbles (< 1 cm) is dragged much harder: the late run-up tip strands, and the backwash
    // tail creeps instead of sliding.
    float n = uManning * mix(uThinN, 1.0, smoothstep(0.003, 0.012, h));
    float cf = G_ACC * n * n * sp / pow(max(h, uFricH), 1.3333);
    U.yz /= 1.0 + uDtS * cf;
    // infiltration into the permeable pebble face above the water table (§28); the lower
    // swash stays saturated (the water table follows the mean swash level), so its backwash
    // film stays continuous instead of breaking into rivulets
    // (a residual film of a few mm still soaks in on the saturated lower face)
    float dry = max(smoothstep(uInfZ.x, uInfZ.y, B), uInfLow * smoothstep(0.006, 0.0, h) * step(0.0, B));
    // the last millimetre or two on the lower face is held between the pebbles (menisci) and only
    // soaks in slowly: the face stays filmed and glossy for a few seconds after the backwash edge
    // has passed (clip t = 0-0.6 s: Z's film at z 1.1-1.7 until A's bore arrives)
    float hRet = uRetain.x * (1.0 - smoothstep(uRetain.y, uRetain.z, B)) * step(0.0, B);
    float dh = min(max(h - hRet, 0.0), uInfil * uDtS * dry * (0.4 + 0.6 * smoothstep(0.05, 0.0, h)))
             + min(h, hRet) * uDtS / uRetain.w;
    U.x -= dh;
    h -= dh;
    if (h < 1.0e-4) U.yz = vec2(0.0);
  } else {
    U = vec3(max(U.x, B), 0.0, 0.0);
  }
#ifdef SWE_EXPLORE
  // absorbing sponge at the seaward boundary; the lateral edges of the scrolling window are open
  // (FLUX) and may also be nudged toward the analytic far field shown beyond them (uEdgeNudge,
  // off by default: a nudged band moving with a running player pumps far-field water into the window)
  float sp = 1.0 - smoothstep(SWE_ZMIN, SWE_ZMIN + 0.9, xz.y);
  if (B < 0.0) U.x = mix(U.x, 0.0, sp * 0.10);
  U.yz *= 1.0 - sp * 0.10;
  float xR = uDom.x + float(uN.x) * uDom.z;
  float sx = 1.0 - smoothstep(0.0, 1.0, min(xz.x - uDom.x, xR - xz.x));
  if (sx * uEdgeNudge > 0.0) {
    vec4 fv = swashFar(xz, uTime);
    float wt = mix(B + fv.x, max(bedProfile(xz.y) + fv.x, B), smoothstep(0.03, 0.1, fv.x));
    U = mix(U, vec3(wt, fv.x * fv.y, fv.x * fv.z), sx * uEdgeNudge);
  }
#else
  // absorbing sponge at the seaward boundary and the lateral edges
  float sp = 1.0 - smoothstep(SWE_ZMIN, SWE_ZMIN + 0.9, xz.y);
  float sx = smoothstep(SWE_XMAX - 0.8, SWE_XMAX, abs(xz.x));
  float relax = max(sp * 0.10, sx * 0.03);
  if (B < 0.0) U.x = mix(U.x, 0.0, relax);
  U.yz *= 1.0 - relax;
#endif
  gl_FragColor = vec4(U, 0.0);
}`;

// Foam transport (§13, §31, §32): whitewater R (dense aerated, from the crash and
// the turbulent bore front) decays into surface foam G (the lace on the swash).
// Two advected lace coordinate sets (phase-offset resets) let the pattern stretch
// with the flow without smearing forever.
const FOAM = COMMON + DOMAIN + BREAKER + /* glsl */ `
uniform sampler2D uState;
uniform sampler2D uBed;
uniform sampler2D uFoamPrev;
uniform sampler2D uUVPrev;
uniform float uDtS;
uniform float uResetA;   // 1 when lace set A resets this step
uniform float uResetB;
uniform vec4 uFoamK;     // x: blanket tau (s), y: lace yield, z: lace tau on the swash (s), w: lace tau on deep water (s)
uniform vec4 uFoamS;     // x: injection -> blanket, y: bore front -> blanket, z: milk yield, w: front convergence scale (1/s)
uniform vec4 uFoamM;     // x: milk tau (s), y: uprush aeration (1/s), z: blanket tau factor on deep water, w: standing-jump foam
uniform vec4 uFoamK2;    // x: shear turbulence (1/s), y: shear milk (1/s), z: swash-tip foam, w: jump convergence scale (1/s)
uniform float uFoamRMax; // ceiling of the blanket amount (thick foam lasts longer, see uFoamS2.x)
uniform float uFoamCap;  // blanket amount a sheet can carry per metre of depth
uniform float uFoamPatch; // patchiness of the whitewater production (relative amplitude)
uniform vec4 uFoamS2;    // x: blanket persistence per unit thickness, y: air entrainment by turbulence (1/s), z: tau factor of a stalled thin sheet, w: turbulence tau in deep water (s)
layout(location = 1) out vec4 outUV;

vec3 st(ivec2 q) { return texelFetch(uState, clamp(q, ivec2(0), uN - 1), 0).xyz; }
float bd(ivec2 q) { return texelFetch(uBed, clamp(q, ivec2(0), uN - 1), 0).x; }
vec2 velAt(ivec2 q) {
  vec3 U = st(q); float h = max(U.x - bd(q), 0.0);
  float h4 = h * h * h * h;
  return 1.41421356 * h * U.yz / sqrt(h4 + max(h4, 1.0e-8));
}
float hAt(ivec2 q) { return max(st(q).x - bd(q), 0.0); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 xz = cellXZ(p);
  vec3 U = st(p);
  float B = bd(p);
  float h = max(U.x - B, 0.0);
  vec2 vel = velAt(p);
  vec2 texel = 1.0 / vec2(uN);
  vec2 uv = (vec2(p) + 0.5) * texel;
  vec2 back = uv - vel * uDtS / (vec2(uN) * uDom.zw);
  vec4 F = textureLod(uFoamPrev, back, 0.0);
  vec4 L = textureLod(uUVPrev, back, 0.0);
  // a draining film only millimetres thick runs off in rivulets between the pebble lumps; the foam
  // floating on it spreads sideways instead of being combed into cross-shore streaks with them
  float thinF = (1.0 - smoothstep(0.02, 0.05, h)) * step(1.0e-4, h);
  if (thinF > 0.0) {
    vec2 t2 = vec2(2.0 / float(uN.x), 0.0);
    F = mix(F, 0.5 * (textureLod(uFoamPrev, back + t2, 0.0) + textureLod(uFoamPrev, back - t2, 0.0)), 0.3 * thinF);
  }
  // lace coordinates are stored as offsets from the cell position (keeps precision)
  L -= vec4(vel * uDtS, vel * uDtS);
  // Where the flow tears the advected frame apart (the thin uprush tongue sliding away from the
  // slow film ahead, a backwash sheet leaving its stranded tail) the offsets would stretch the
  // pattern into long drips. There the offset field diffuses (nonlinear: only where its gradient
  // exceeds ~1), so the pattern smears into the mean instead of streaking.
  {
    vec2 tx = vec2(1.0 / float(uN.x), 0.0), tz = vec2(0.0, 1.0 / float(uN.y));
    vec4 LE = textureLod(uUVPrev, back + tx, 0.0), LW = textureLod(uUVPrev, back - tx, 0.0);
    vec4 LN = textureLod(uUVPrev, back + tz, 0.0), LS = textureLod(uUVPrev, back - tz, 0.0);
    vec4 gx = (LE - LW) / (2.0 * uDom.z), gz = (LN - LS) / (2.0 * uDom.w);
    float gA = length(vec4(gx.xy, gz.xy)), gB = length(vec4(gx.zw, gz.zw));
    vec4 avg = 0.25 * (LE + LW + LN + LS) - vec4(vel * uDtS, vel * uDtS);
    L.xy = mix(L.xy, avg.xy, 0.5 * smoothstep(0.8, 2.5, gA));
    L.zw = mix(L.zw, avg.zw, 0.5 * smoothstep(0.8, 2.5, gB));
  }
  if (uResetA > 0.5) L.xy = vec2(0.0);
  if (uResetB > 0.5) L.zw = vec2(0.0);

  // bore-front detector: where a turbulent front passes, the depth jumps within a fraction of
  // a second: dh/dt = -div(h u) reaches several m/s, while the smooth swash lens changes by
  // only a few cm/s. Foam production ~ dissipation ~ rate of rise times the flow speed.
  ivec2 ex = ivec2(1, 0), ez = ivec2(0, 1);
  vec2 vE = velAt(p + ex), vW = velAt(p - ex), vN = velAt(p + ez), vS = velAt(p - ez);
  float hE = hAt(p + ex), hW = hAt(p - ex), hN = hAt(p + ez), hS = hAt(p - ez);
  float dhdt = -((hE * vE.x - hW * vW.x) / (2.0 * uDom.z) + (hN * vN.y - hS * vS.y) / (2.0 * uDom.w));
  float sp = length(vel);
  float front = smoothstep(uFoamS.w * 0.25, uFoamS.w, dhdt) * smoothstep(0.012, 0.045, max(h, max(hS, hN))) * max(sp, 0.6);
  // (only a front running up the beach is a breaking bore: the mound a section that broke early
  // spreads sideways into its still unbroken neighbours as a lateral surge, which carries no roller
  // - a white slab with a square corner sliding along the shore otherwise)
  front *= smoothstep(-0.1, 0.5, vel.y / max(sp, 0.25));
  // the uprush is the collapsed bore itself: fast supercritical onshore sheet flow stays aerated
  float Fr = sp / sqrt(G_ACC * max(h, 0.003));
  front += uFoamM.y * smoothstep(0.4, 1.2, vel.y) * smoothstep(1.1, 2.0, Fr) * smoothstep(0.01, 0.035, h);
  // hydraulic jumps: strong convergence of the velocity (a bore held against the backwash:
  // +1 m/s meeting -1.5 m/s over ~10 cm) even though the depth no longer changes. Where bore
  // water arrives from the sea side it is a white roller; the backwash's own toe jump at the
  // step is a turbulent, only slightly foamy band (swash §7)
  float conv = -((vE.x - vW.x) / (2.0 * uDom.z) + (vN.y - vS.y) / (2.0 * uDom.w));
  float jump = smoothstep(uFoamK2.w * 0.3, uFoamK2.w, conv) * smoothstep(0.008, 0.03, h);
  // (the toe roller boils up in patches that come and go along the step: an irregular band)
  // Kinetic contact aeration around submerged rock shoulders. No source at rest.
  vec2 bedSlope=vec2(bd(p+ex)-bd(p-ex),bd(p+ez)-bd(p-ez))/(2.0*uDom.zw);
  float rockRise=max(bd(p)-bedHeight(xz),0.0);
  float contact=clamp(dot(vel,bedSlope),0.0,1.8)*smoothstep(.035,.12,rockRise)*smoothstep(.12,.70,sp)*smoothstep(.008,.035,h);
  front+=.065*contact;
  float toe = smoothstep(0.3, 0.75, 0.5 + 0.5 * gnoise(vec2(xz.x * 2.6, uTime * 0.9 + xz.y * 1.5)));
  front += uFoamM.w * jump * mix(0.2 + 1.2 * toe, 1.0, smoothstep(0.1, 0.5, vS.y));
  // the leading edge of a still-turbulent bore sheet rolls over itself (a foam roller along the
  // edge, foam lumps right on the pebbles); the late, laminar run-up tip carries none
  // (a bore front held against the backwash or pushing slowly up; the fast sheet of a surging uprush -
  // a splash sheet running out - carries none: it is torn and translucent, swash §4)
  float edge = smoothstep(0.002, 0.006, h) * (1.0 - smoothstep(0.3 * h, 0.7 * h, hN)) * (1.0 - smoothstep(0.9, 1.5, vel.y));
  front += uFoamK2.z * edge * smoothstep(0.1, 0.6, vel.y) * smoothstep(0.2, 0.8, F.b + 0.3 * F.r);
  vec3 inj = brkInjection(xz, uTime);
  float splash = brkFoam(xz, uTime);

  // two-layer foam (swash analysis §4, colour §4.5): the opaque blanket R (whitewater from
  // the crash and the turbulent bore front) tears into lace G (yield ~0.5 of what the blanket
  // loses); rates in FOAM_SIM, calibrated on the clip's foam-cover curves per image band
  // (the step zone stays white ~2 s after a plunge, the swash blanket tears within ~1 s).
  // Both die in place when the film drains.
  // Milkiness M (bubble/sediment cloud in the water column) decays with tau ~1.25 s.
  float R = F.r, Gf = F.g, K = F.b, Mk = F.a;
  // while the bore is still turbulent it keeps entraining air all along its path (not only at
  // the front and the plunge line): no clear-water gap between the crash foam and the swash foam
  float entrain = uFoamS2.y * smoothstep(0.35, 1.3, K) * smoothstep(0.02, 0.07, h);
  // whitewater is produced in patches (splash fingers, bursts of the roller): 0.4-1.2 m, slowly
  // changing; the flow draws them out into the soft bands and swirls of the torn blanket
  float pn = gnoise(xz * vec2(1.1, 1.5) + vec2(0.0, uTime * 0.35)) + 0.5 * gnoise(xz * vec2(2.6, 3.2) - vec2(uTime * 0.5, 1.7));
  float prod = 1.0 + uFoamPatch * (2.0 * smoothstep(-0.7, 0.7, pn) - 1.0);
  R += uDtS * prod * (uFoamS.x * inj.x + uFoamS.y * front + splash + entrain);
  // a sheet a few centimetres thick cannot carry a thick blanket: the thin, late uprush tip (a
  // splash sheet running out) is torn and translucent (swash §4: B's tip tears within ~0.3 s)
  R = min(R, uFoamCap * max(h - 0.002, 0.0) + 0.02);
  // a thick foam layer takes longer to drain and collapse than a thin one (tau grows with R);
  // the churning roller over the step (deep, turbulent water) keeps its whitewater much longer;
  // a stalled thin sheet tears a little faster
  // (the churning roller is the step's, shoreward of the plunge zone, and it is fed by the incoming
  // bore: in a strong backwash, which dives under at the toe, it clears quickly - a streaky,
  // translucent band, not a white one)
  float persist = smoothstep(0.05, 0.13, h) * (1.0 - smoothstep(0.2, 0.9, xz.y)) * smoothstep(0.15, 0.7, K)
                * smoothstep(-0.9, -0.3, vel.y) * smoothstep(-1.9, -1.2, xz.y);
  float tauR = uFoamK.x * mix(1.0, uFoamM.z, persist) * (1.0 + uFoamS2.x * max(R - 0.5, 0.0))
             * mix(uFoamS2.z, 1.0, max(smoothstep(0.15, 0.5, sp), smoothstep(0.03, 0.08, h)));
  float loss = R * (1.0 - exp(-uDtS / tauR));
  R -= loss;
  float tauG = mix(uFoamK.z, uFoamK.w, smoothstep(0.12, 0.45, h));
  Gf += uFoamK.y * loss;
  Gf *= exp(-uDtS / tauG);
  float drained = 1.0 - smoothstep(0.0004, 0.0022, h);
  R *= exp(-uDtS * 10.0 * drained);
  Gf *= exp(-uDtS * 7.0 * drained);
  // turbulent kinetic energy proxy k (§11): produced by injection & bore, by the hydraulic jumps
  // (also the backwash toe) and by bed shear in the fast thin sheet over the pebbles (keeps the
  // swash surface rippled, not glassy); it lasts longer in the deep churning water over the step
  float shear = smoothstep(0.1, 0.9, sp) * smoothstep(0.002, 0.012, h) * (1.0 - smoothstep(0.15, 0.4, h));
  // (the backwash's own toe jump: turbulent but hardly aerated - the dark toe band)
  K += uDtS * (6.0 * inj.x + 1.5 * front + uFoamK2.x * shear + 2.0 * jump * mix(1.0, 0.35, smoothstep(0.2, 0.6, -vN.y)));
  K *= exp(-uDtS / mix(0.7, uFoamS2.w, smoothstep(0.05, 0.2, h)));
  // milk: bubbles released by the decaying blanket, plus sediment/bubbles kept in suspension
  // (a film of a few cm cannot hold a bubble cloud: bed shear only clouds water deeper than ~3 cm,
  // and the bubbles of a thinning late film escape twice as fast)
  // (a backwash carries its bubbles down and dives under the next bore at the toe: no new cloud from
  // its bed shear, and the jump there pulls the milk under - the dark toe band, swash §7)
  Mk += uFoamS.z * (loss + uDtS * inj.x) + uDtS * uFoamK2.y * shear * smoothstep(0.02, 0.045, h) * (1.0 - smoothstep(0.2, 0.7, -vel.y));
  Mk *= exp(-uDtS / (uFoamM.x * mix(0.5, 1.0, smoothstep(0.01, 0.04, h)))) * exp(-uDtS * (4.0 * drained + 2.5 * jump * smoothstep(0.1, 0.5, -vN.y)));
  gl_FragColor = vec4(clamp(R, 0.0, uFoamRMax), clamp(Gf, 0.0, 2.0), clamp(K, 0.0, 4.0), clamp(Mk, 0.0, 1.0));
  outUV = L;
}`;

// Render view of the foam state: what the water / beach shaders see. The SWE foam under the
// raised face of a shoaling or breaking wave is hidden (brkSweFoamKeep: the face is stretched and
// renewed, and the heightfield would smear the xz-sampled foam into vertical streaks).
const FOAM_VIEW = COMMON + DOMAIN + BREAKER + /* glsl */ `
uniform sampler2D uFoamT;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 F = texelFetch(uFoamT, p, 0);
  F = any(isnan(F)) ? vec4(0.0) : clamp(F, vec4(0.0), vec4(8.0, 2.0, 4.0, 1.0));
  gl_FragColor = F * brkSweFoamKeep(cellXZ(p), uTime);
}`;

// Beach wetness (§27): film = glossy water film, damp = darkened pebbles.
const WET = XDEF + COMMON + DOMAIN + /* glsl */ `
uniform sampler2D uState;
uniform sampler2D uBed;
uniform sampler2D uWetPrev;
uniform float uDtS;
uniform float uFilmTau;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float h = max(texelFetch(uState, p, 0).x - texelFetch(uBed, p, 0).x, 0.0);
  vec2 W = texelFetch(uWetPrev, p, 0).xy;
  float cover = smoothstep(0.0008, 0.004, h);
  // film channel: 1 under water, then exp(-t / uFilmTau) once the film has gone; the beach
  // recovers the time since the film left from it and reshapes it to the measured gloss decay
  // (pebble contrast returns 0.3-0.4 s after the film edge passes, swash §5)
  W.x = max(cover, W.x * exp(-uDtS / uFilmTau));
  W.y = max(cover, W.y * exp(-uDtS / 55.0));
#ifdef SWE_EXPLORE
  // the capillary fringe above the run-up line stays damp (as the far field's; no drying with the
  // time since the window last moved)
  W.y = max(W.y, 1.0 - smoothstep(0.35, 0.75, bedHeight(cellXZ(p))));
#endif
  float oldLine=texelFetch(uWetPrev,p,0).z;
  float B=bedHeight(cellXZ(p));
  float line=mix(B,oldLine,exp(-uDtS/20.0));
  if(h>.001) line=max(line,texelFetch(uState,p,0).x);
  gl_FragColor = vec4(W,line,0.0);
}`;

const WET_INIT = COMMON + DOMAIN + /* glsl */ `
void main() {
  vec2 xz = cellXZ(ivec2(gl_FragCoord.xy));
  float B = bedHeight(xz);
  gl_FragColor = vec4(0.0, 1.0 - smoothstep(0.35, 0.75, B), B, 0.0);
}`;

// Render view: (h, u, v, w) half-float, linearly filterable for the water/beach shaders.
// The flow sees a smoothed bed (uRelief, no cusp ridges); the rendered surface drapes thin films
// over the full pebble-face relief (w = bedHeight + h) so dry cells stay exactly dry for the
// renderers, and shows the (lightly smoothed) solver level where the water is deep.
// Total water-surface elevation (shallow-water level + analytic breaker heightfield) baked
// once per step, so particles and other consumers sample one texture instead of
// re-evaluating every breaker event (cheaper at runtime and far faster to compile).
const HEIGHT = COMMON + DOMAIN + BREAKER + /* glsl */ `
uniform sampler2D uState;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 xz = cellXZ(p);
  float w = texelFetch(uState, p, 0).x;
  vec2 uv = (vec2(p) + 0.5) / vec2(uN);
  vec2 e = min(uv, 1.0 - uv);
  float wIn = smoothstep(0.0, 0.04, min(e.x, e.y));
  gl_FragColor = vec4(mix(0.0, w, wIn) + brkSurfaceOnly(xz, uTime), 0.0, 0.0, 1.0);
}`;

const VIEW = COMMON + DOMAIN + /* glsl */ `
uniform sampler2D uState;
uniform sampler2D uBed;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 U = texelFetch(uState, p, 0).xyz;
  float B = texelFetch(uBed, p, 0).x;
  float h = max(U.x - B, 0.0);
  float h4 = h * h * h * h;
  vec2 vel = 1.41421356 * h * U.yz / sqrt(h4 + max(h4, 1.0e-8));
  // thin films are draped over the full rendered relief (dry stays exactly dry); deeper water
  // shows the solver's own level (its bed meanders at the step knee), lightly smoothed: a
  // hydraulic jump is a 1-2 cell step in the solver, which would flip the reflections along a line
  float Bfull = hydraulicBedHeight(cellXZ(p));
  ivec2 ex = ivec2(1, 0), ez = ivec2(0, 1), hi = uN - 1;
  float ws = 0.4 * U.x + 0.1 * (texelFetch(uState, clamp(p + ex, ivec2(0), hi), 0).x + texelFetch(uState, clamp(p - ex, ivec2(0), hi), 0).x
                              + texelFetch(uState, clamp(p + ez, ivec2(0), hi), 0).x + texelFetch(uState, clamp(p - ez, ivec2(0), hi), 0).x)
           + 0.05 * (texelFetch(uState, clamp(p + 2 * ex, ivec2(0), hi), 0).x + texelFetch(uState, clamp(p - 2 * ex, ivec2(0), hi), 0).x
                   + texelFetch(uState, clamp(p + 2 * ez, ivec2(0), hi), 0).x + texelFetch(uState, clamp(p - 2 * ez, ivec2(0), hi), 0).x);
  gl_FragColor = vec4(h, vel, mix(Bfull + h, max(ws, Bfull + h * 0.5), smoothstep(0.04, 0.12, h)));
}`;

// ---- explore mode: scrolling window + far field
// Whole-cell scroll of a field: the new cell i takes the old cell i + k. A column entering from
// beyond the old window continues the old edge column, changed along-shore as the far field
// changes between the two positions (edge + far(x) - far(x_edge)): continuous at the seam (no
// dam-break between the solver's water and a far-field state), yet not an x-invariant smear. Thin
// films follow the far field's depth, deeper water its level; velocities are the edge's. After a
// jump longer than the window the columns start from the far field itself. uDom (and the bed)
// already hold the new window.
const SHIFT = COMMON + DOMAIN + SWASH_FAR_CORE + /* glsl */ `
uniform sampler2D uSrc;
uniform sampler2D uBed;
uniform int uShift;
uniform int uMode;       // 0: solver state, 1: wetness
uniform float uTime;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 q = p + ivec2(uShift, 0);
  if (q.x >= 0 && q.x < uN.x) { gl_FragColor = texelFetch(uSrc, q, 0); return; }
  vec2 xz = cellXZ(p);
  ivec2 qe = ivec2(clamp(q.x, 0, uN.x - 1), p.y);    // old edge column (old indices)
  ivec2 pe = ivec2(qe.x - uShift, p.y);               // ... in the new window
  bool edge = pe.x >= 0 && pe.x < uN.x;
  vec2 xe = cellXZ(pe);
  if (uMode == 0) {
    float B = texelFetch(uBed, p, 0).x;
    vec4 fv = swashFar(xz, uTime);
    if (!edge) {
      float w = mix(B + fv.x, max(bedProfile(xz.y) + fv.x, B), smoothstep(0.03, 0.1, fv.x));
      float h = max(w - B, 0.0);
      gl_FragColor = vec4(max(w, B), h * fv.y, h * fv.z, 0.0);
      return;
    }
    vec3 Ue = texelFetch(uSrc, qe, 0).xyz;
    float Be = texelFetch(uBed, pe, 0).x;
    vec4 fe = swashFar(xe, uTime);
    float he = max(Ue.x - Be, 0.0);
    float hn = max(he + fv.x - fe.x, 0.0);
    float w = mix(B + hn, max(Ue.x + fv.w - fe.w, B), smoothstep(0.05, 0.12, he));
    float h = max(w - B, 0.0);
    vec2 vel = he > 1.0e-3 ? Ue.yz / he : vec2(0.0);
    vel *= min(1.0, 3.0 / max(length(vel), 1.0e-4));
    gl_FragColor = vec4(max(w, B), h * vel, 0.0);
  } else {
    vec2 wf = swashFarWet(xz, uTime);
    if (!edge) { gl_FragColor = vec4(wf, max(bedHeight(xz),swashFar(xz,uTime).w), 0.0); return; }
    vec2 We = texelFetch(uSrc, qe, 0).xy;
    gl_FragColor = vec4(clamp(We + wf - swashFarWet(xe, uTime), 0.0, 1.0), max(bedHeight(xz),texelFetch(uSrc,qe,0).z), 0.0);
  }
}`;
// ... the foam MRT (blanket / lace / turbulence / milk + advected lace coordinates)
const SHIFT_FOAM = COMMON + DOMAIN + SWASH_FAR_CORE + /* glsl */ `
uniform sampler2D uSrcA;
uniform sampler2D uSrcB;
uniform int uShift;
uniform float uTime;
layout(location = 1) out vec4 outUV;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 q = p + ivec2(uShift, 0);
  if (q.x >= 0 && q.x < uN.x) { gl_FragColor = texelFetch(uSrcA, q, 0); outUV = texelFetch(uSrcB, q, 0); return; }
  vec2 xz = cellXZ(p);
  ivec2 qe = ivec2(clamp(q.x, 0, uN.x - 1), p.y);
  ivec2 pe = ivec2(qe.x - uShift, p.y);
  vec4 ff = swashFarFoam(xz, uTime);
  if (pe.x < 0 || pe.x >= uN.x) { gl_FragColor = ff; outUV = vec4(0.0); return; }
  gl_FragColor = max(texelFetch(uSrcA, qe, 0) + ff - swashFarFoam(cellXZ(pe), uTime), 0.0);
  outUV = texelFetch(uSrcB, qe, 0);
}`;
// Far-field parameters of every remembered breaker event (rows) along x (columns: the far-field
// textures' asinh-warped x mapping), evaluated with the breaker's own along-crest timing (brkAt on
// this pass's copy of the event arrays): (impact time, run-up R_max, landing line z_I, energy).
// R_max = 4.2 H + 0.35 (solver: 2.2 m for the clip's waves; spilling breakers run up ~15 % less)
// x bore gain^0.7, with a per-event along-shore undulation (5.3 / 2.3 m) and swash-front lobes
// (1.2 / 0.42 m, swash §3.2); second row per event: (seed, style, H, spill). Re-baked only when
// the event history or the window changes.
const FAR_PARAM = COMMON + BREAKER + FAR_MAP + /* glsl */ `
void main() {
  ivec2 ij = ivec2(gl_FragCoord.xy);
  float x = swashFarXofColumn(float(ij.x));
  int k = ij.y / 2;
  if (k >= uEvtCount) { gl_FragColor = vec4(-1.0e4, 0.0, -2.0, 0.0); return; }
  // column width: far from the window the columns are 10-40 cm wide, so every along-shore detail
  // finer than two columns (the crest's 0.45 m staircase and timing wobble, the finest lobes) is
  // averaged over the column (4 sub-samples) or faded out - sampled per column it aliases into a
  // regular sawtooth of the fronts and the wet line
  float dxc = swashFarXofColumn(float(ij.x) + 0.5) - swashFarXofColumn(float(ij.x) - 0.5);
  int ns = dxc > 0.05 ? 4 : 1;
  float ti = 0.0, zI = 0.0, H = 0.0;
  for (int j = 0; j < 4 + min(uEvtCount, 0); j++) {   // (one call site of brkAt; dynamic bound: FXC)
    if (j >= ns) break;
    Brk bj = brkAt(k, x + dxc * ((float(j) + 0.5) / float(ns) - 0.5));
    ti += bj.ti; zI += bj.zI; H += bj.H;
  }
  ti /= float(ns); zI /= float(ns); H /= float(ns);
  vec4 A = uEvtA[k], C = uEvtC[k], E4 = uEvtE[k];
  float sd = A.w, gB = max(uEvtG[k].x, 0.0), str = max(uEvtB[k].w, 0.0), style = C.w, spill = E4.z;
  // second row: (seed, style, H, spill)
  if (ij.y - 2 * k == 1) { gl_FragColor = vec4(sd, style, H, spill); return; }
  float ph = 2.5 * sin(0.37 * x + sd * 5.1 + 1.7 * sin(0.13 * x));   // (wandering phase: no regular comb)
  float lp1 = 1.0 - smoothstep(0.2, 0.45, 5.236 * dxc / 6.2832), lp2 = 1.0 - smoothstep(0.2, 0.45, 14.96 * dxc / 6.2832);
  float lob = 0.12 * sin(x * 1.186 + sd * 41.3 + 0.4 * ph) + 0.07 * sin(x * 2.732 + sd * 17.9 + 1.3 + 0.8 * ph)
            + (0.06 * lp1 * sin(x * 5.236 + sd * 37.1 + ph) + 0.025 * lp2 * sin(x * 14.96 + sd * 91.7 + 0.7 * sin(2.1 * x) + 1.9 * ph))
              * (0.45 + 0.9 * (0.5 + 0.5 * sin(0.29 * x + sd * 3.3) * sin(0.71 * x + 1.1)));
  // (x the 10-40 m undulation and the cusps, as the bore injection: brkRunupMod)
  float Rm = (4.2 * H + 0.35) * pow(gB, 0.7) * (0.85 + 0.15 * smoothstep(0.3, 0.8, style)) * str * (1.0 + lob) * (1.0 + brkRunupMod(x, sd));
  float s = H / 0.45;
  float E = s * s * str * mix(0.6, 1.0, style) * gB;
  gl_FragColor = vec4(ti, Rm, zI, E);
}`;
// The far field itself (glsl/swashfar.js FAR_BAKE_MODEL), every step: (h, u, v, w), foam, wetness.
const FAR_MODEL = COMMON + FAR_BAKE_MODEL;
const EXPLORE_WIN = {
  nx: 736,               // 22.1 m of 3 cm cells along-shore; extra dry rows in Explore
  follow: 1.0,           // recentre when the focus is this far (m) from the window centre
  blend: [5.8, 8.8],     // continuous player-centred handoff; includes the 1 m solver-follow reserve
  nudge: 0.0,            // relaxation per step of the outer metre toward the far field (0: open edges only)
};
const FAR = {
  cols: 2048,            // x = window centre + L sinh(g / L), g uniform over +- L asinh(range / L):
  L: 12.0, range: 128.0, //   3.6 cm columns near the window, ~10 cm at 30 m, ~40 cm at 128 m
  z0: -4.05, dz: 0.03, rows: 552,   // z rows (centres) -4.05 .. 12.48 m; high-run-up reserve
  slots: 6,              // remembered events (= MAX_EVENTS of the breaker arrays)
  // a, crossing time, backwash time scale, residual-film line | bore level, residual film, (film tau), step toe
  k: [1.3, 1.2, 1.0, 0.95], k2: [0.17, 0.004, 0, -0.55],
};

// Tuned injection defaults (see brkInjection in glsl/breaker.js).
const INJ = {
  mass: 2.4,
  speed: 1.0,
  jet: [0.17, 0.15, 2.875, 3.55],  // mass fraction, pulse T, zone drift, water speed
  // (the roller is held at the step ~0.55 s, then surges: the first, jet-driven pulse stalls against
  // the previous backwash, the main uprush follows - timeline finding 4; tuned on the in-frame edge
  // tracks with the run-up held at the measured 0.2-0.8 m past the frame bottom, z_max ~2.45 m)
  roller: [0.55, 0.3, 1.2, 2.2],   // delay, pulse T, zone drift, water speed
  lobes: [0.05, 0.025, 0.012, 0.15],   // along-crest momentum modulation at 1.2 / 0.42 m / fine quasi-random, along-shore drift ratio
};
// Foam life cycle (FOAM pass) and look (glsl/foam.js).
const FOAM_SIM = {
  k: [0.3, 0.29, 1.65, 2.0],       // blanket tau, lace yield, lace tau swash, lace tau deep
  s: [1.0, 31.0, 0.3, 1.2],       // injection gain, bore-front gain, milk yield, front convergence scale
  m: [1.25, 0.0, 2.5, 0.18],      // milk tau, uprush aeration, roller persistence (x blanket tau in deep water), standing-jump foam
  k2: [4.0, 0.1, 3.0, 11.0],      // shear turbulence, shear milk, swash-tip roller foam, jump convergence scale (1/s)
  s2: [0.0, 0.0, 0.15, 1.45],     // blanket persistence per unit thickness, turbulent entrainment, stalled-sheet tau factor, deep turbulence tau
  rMax: 2.5,                      // ceiling of the blanket amount (a fresh blanket tears ~0.4 s after production stops)
  cap: 20.0,                      // blanket amount a sheet can carry per metre of depth (a 2-3 cm late uprush tip: torn)
  patch: 0.35,                    // patchiness of the whitewater production (relative amplitude)
};
const FOAM_LOOK = [10.5, 0.042, 0.30, 1.0];    // lace cells per metre, filament width (cell units), lace opacity, blanket opacity
const FOAM_LOOK2 = [6.0, 1.6, 0.45, 0.14];    // torn-foam hole cells per metre, max hole elongation along the flow, bubble grain, swirl warp (m)
const FOAM_SRC = [6.0, 0.43, 2.0, 1.0];      // splash-sheet foam, its pulse T, roller foam, splash zone width scale
const FOAM_SPILL = 24.0;                      // whitewater of a spilling roller (1/s per unit spill at H = 0.45)
const FOAM_SPILL_K = [1.5, 1.8];              // its drift speed (m/s), end of production (s after the impact)
// Solver source parameters (CONFIG.swe holds the grid; these are the tuned physics).
const SWE = {
  manning: 0.028,                  // wet gravel (swash analysis: n ≈ 0.022-0.025; tuned on the edge tracks)
  fricH: 0.005,                    // depth floor in the friction law (m)
  infiltration: 0.08,              // m/s into the permeable pebble face (swash analysis: 1-2 cm/s x 1.27; tuned: the backwash film edge re-exposes the upper swash on time, t 2.8-3.0 s)
  infZ: [0.105, 0.26],             // infiltration ramps in between these bed elevations (m): water table
  infLow: 0.25,                    // residual thin film (< 6 mm) below the water table still soaks in
  thinN: 1.0,                      // Manning n multiplier for films < 1 cm (thinner than the pebbles)
  relief: 0.5,                     // fraction of bedRelief() (cusps, lumps) seen by the flow
  retain: [0.0015, 0.17, 0.28, 3.0], // film held by the pebbles below bed elevation ~0.2 m (z < 1.5-2 m): depth, fade, soak-in time
};

export class SwashSim {
  constructor(renderer, shared) {
    const s = CONFIG.swe;
    this.renderer = renderer;
    this.shared = shared;
    this.explore = EXPLORE;
    this.dx = (s.xMax - s.xMin) / s.nx;
    this.dz = (s.zMax - s.zMin) / s.nz;
    // clip mode: the fixed 9.6 m window at x = 0; explore mode: a 22 m window that scrolls with the focus
    this.nx = EXPLORE ? EXPLORE_WIN.nx : s.nx; this.nz = s.nz;
    this.cx = EXPLORE ? 0 : 0.5 * (s.xMin + s.xMax);   // window centre (world x)
    const x0 = EXPLORE ? this.cx - 0.5 * this.nx * this.dx : s.xMin;
    this.substeps = s.substeps;
    const dom = new THREE.Vector4(x0, s.zMin, this.dx, this.dz);
    this.dom = dom;
    const N = new THREE.Vector2(this.nx, this.nz);
    const base = { uRockBed:shared.uRockBed || {value:null}, uRockDomain:shared.uRockDomain || {value:new THREE.Vector4()}, uDom: { value: dom }, uN: { value: N } };
    const nx = this.nx, nz = this.nz;

    this.bed = floatRT(nx, nz);
    this.state = new PingPong(nx, nz);
    this.foamMRT = [this._mrt(), this._mrt()];
    this.foamIdx = 0;
    this.foamView = floatRT(nx, nz, { type: THREE.HalfFloatType, filter: THREE.LinearFilter });
    this.wet = new PingPong(nx, nz, { type: THREE.HalfFloatType, filter: THREE.LinearFilter });
    this.view = floatRT(nx, nz, { type: THREE.HalfFloatType, filter: THREE.LinearFilter });
    this.height = floatRT(nx, nz, { type: THREE.HalfFloatType, filter: THREE.LinearFilter });

    // bore injection (breaker.js "SWE coupling"); URL ?inj= / ?injv= still override the mass / speed
    const qs = new URLSearchParams(typeof location !== 'undefined' ? location.search : '');
    if (!qs.has('inj')) shared.uInjMass.value = INJ.mass;
    if (!qs.has('injv')) shared.uInjSpeed.value = INJ.speed;
    shared.uInjJ = { value: new THREE.Vector4(...INJ.jet) };
    shared.uInjR = { value: new THREE.Vector4(...INJ.roller) };
    shared.uInjL = { value: new THREE.Vector4(...INJ.lobes) };
    // time constant of the wetness film channel (Beach.js FILM_TAU_SOLVER reshapes it)
    shared.uSweFilmTau = { value: 2.6 };
    shared.uFoamLook = { value: new THREE.Vector4(...FOAM_LOOK) };
    shared.uFoamLook2 = { value: new THREE.Vector4(...FOAM_LOOK2) };
    shared.uFoamSrc = { value: new THREE.Vector4(...FOAM_SRC) };
    shared.uFoamSpill = { value: FOAM_SPILL };
    shared.uFoamSpillK = { value: new THREE.Vector2(...FOAM_SPILL_K) };
    // (the milk colour itself belongs to the water optics: WaterSurface.js sets shared.uMilkColor)

    // far-field swash (glsl/swashfar.js): textures over the window centre +- 128 m, the blend beyond
    // the window (all zero / disabled in clip mode)
    const gHalf = FAR.L * Math.asinh(FAR.range / FAR.L);
    shared.uSwashFarMap = { value: new THREE.Vector4(this.cx, FAR.L, gHalf, 0) };
    shared.uSwashFarMapZ = { value: new THREE.Vector4(FAR.z0, FAR.dz, FAR.rows, FAR.cols) };
    shared.uSwashFarWin = { value: new THREE.Vector4(this.cx, EXPLORE_WIN.blend[0], EXPLORE_WIN.blend[1], 0) };
    shared.uSwashFocus = shared.uFocus;
    shared.uSwashFarK = { value: new THREE.Vector4(...FAR.k) };
    shared.uSwashFarK2 = { value: new THREE.Vector4(FAR.k2[0], FAR.k2[1], shared.uSweFilmTau.value, FAR.k2[3]) };
    shared.uSwashFarView = { value: null };
    shared.uSwashFarFoamT = { value: null };
    shared.uSwashFarWetT = { value: null };
    const U = shared;
    this.pBed = new FullscreenPass(makeShader(BED_INIT, { ...base, uRelief: { value: SWE.relief } }));
    this.pInit = new FullscreenPass(makeShader(STATE_INIT, { ...base, uBed: { value: this.bed.texture } }));
    this.pFlux = new FullscreenPass(makeShader(FLUX, { ...base, uState: { value: null }, uBed: { value: this.bed.texture }, uDt: { value: 0 } }));
    this.pSrc = new FullscreenPass(makeShader(SOURCES, {
      ...base, ...U, uState: { value: null }, uBed: { value: this.bed.texture }, uDtS: { value: 0 },
      uManning: { value: SWE.manning }, uFricH: { value: SWE.fricH }, uInfil: { value: SWE.infiltration },
      uInfZ: { value: new THREE.Vector2(...SWE.infZ) }, uInfLow: { value: SWE.infLow }, uThinN: { value: SWE.thinN }, uRetain: { value: new THREE.Vector4(...SWE.retain) }, uEdgeNudge: { value: EXPLORE_WIN.nudge },
    }));
    this.pFoam = new FullscreenPass(makeShader(FOAM, {
      ...base, ...U, uState: { value: null }, uBed: { value: this.bed.texture }, uFoamPrev: { value: null }, uUVPrev: { value: null },
      uDtS: { value: 0 }, uResetA: { value: 0 }, uResetB: { value: 0 },
      uFoamK: { value: new THREE.Vector4(...FOAM_SIM.k) }, uFoamS: { value: new THREE.Vector4(...FOAM_SIM.s) },
      uFoamM: { value: new THREE.Vector4(...FOAM_SIM.m) }, uFoamK2: { value: new THREE.Vector4(...FOAM_SIM.k2) },
      uFoamS2: { value: new THREE.Vector4(...FOAM_SIM.s2) }, uFoamRMax: { value: FOAM_SIM.rMax }, uFoamCap: { value: FOAM_SIM.cap }, uFoamPatch: { value: FOAM_SIM.patch },
    }));
    this.pWet = new FullscreenPass(makeShader(WET, { ...base, uState: { value: null }, uBed: { value: this.bed.texture }, uWetPrev: { value: null }, uDtS: { value: 0 }, uFilmTau: shared.uSweFilmTau }));
    this.pWetInit = new FullscreenPass(makeShader(WET_INIT, { ...base }));
    this.pView = new FullscreenPass(makeShader(VIEW, { ...base, uState: { value: null }, uBed: { value: this.bed.texture } }));
    this.pHeight = new FullscreenPass(makeShader(HEIGHT, { ...base, ...shared, uState: { value: null } }));
    this.pFoamView = new FullscreenPass(makeShader(FOAM_VIEW, { ...base, ...shared, uFoamT: { value: null } }));

    // expose to every other shader
    shared.uSweView = { value: this.view.texture };
    shared.uWaterH = { value: this.height.texture };   // total surface elevation over the SWE domain
    shared.uSweFoam = { value: null };
    shared.uSweLace = { value: null };
    shared.uSweWet = { value: null };
    shared.uSweDom = { value: new THREE.Vector4(x0, s.zMin, nx * this.dx, s.zMax - s.zMin) };
    shared.uSweTexel = { value: new THREE.Vector2(1 / nx, 1 / nz) };
    shared.uLacePhase = { value: new THREE.Vector2() };
    // one pattern seed per advected coordinate set, changed only when that set is reset (its
    // weight is then 0), so the lace never re-rolls on screen
    shared.uLaceSeeds = { value: new THREE.Vector2(0.3, 0.7) };

    if (EXPLORE) this._initExplore(base);
    if (EXPLORE) this.rockWetness=new RockWetness(renderer,shared);
    this.reset();
  }

  // ---------------------------------------------------------------- explore mode
  _initExplore(base) {
    const sh = this.shared;
    // per-x event parameters (re-baked when the history / window changes) and the far field itself
    this.farParam = floatRT(FAR.cols, 2 * FAR.slots);   // per event: (t_i, R_max, z_I, E), (seed, style, H, spill)
    this.farRT = new THREE.WebGLRenderTarget(FAR.cols, FAR.rows, {
      count: 3, type: THREE.HalfFloatType, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
      depthBuffer: false, stencilBuffer: false, generateMipmaps: false,
    });
    for (const t of this.farRT.textures) t.colorSpace = THREE.NoColorSpace;
    sh.uSwashFarView.value = this.farRT.textures[0];
    sh.uSwashFarFoamT.value = this.farRT.textures[1];
    sh.uSwashFarWetT.value = this.farRT.textures[2];
    sh.uSwashFarMap.value.w = 1;
    sh.uSwashFarWin.value.w = 1;
    // event history: every packed breaker event seen so far (the schedule drops an event ~1.4 s
    // after its impact over the clip window; its swash lasts ~10 s), newest FAR.slots kept
    this.hist = new Map();
    this.evtKeys = Object.keys(sh).filter((k) => /^uEvt[A-Z]$/.test(k));
    const hu = { uEvtCount: { value: 0 } };
    for (const k of this.evtKeys) hu[k] = { value: new Float32Array(FAR.slots * 4) };
    this.histU = hu;
    const map = { uSwashFarMap: sh.uSwashFarMap, uSwashFarMapZ: sh.uSwashFarMapZ, uSwashFarWin: sh.uSwashFarWin, uSwashFocus: sh.uSwashFocus };
    this.pParam = new FullscreenPass(makeShader(FAR_PARAM, { ...sh, ...hu, ...map }));
    // the bore injection and foam sources also run on the remembered events: an event that has left
    // the packed schedule still injects wherever along the (moving) window it breaks late
    for (const p of [this.pSrc, this.pFoam]) Object.assign(p.material.uniforms, hu);
    this.pFar = new FullscreenPass(makeShader(FAR_MODEL, {
      ...map, uSwashFarParam: { value: this.farParam.texture }, uSwashFarK: sh.uSwashFarK, uSwashFarK2: sh.uSwashFarK2,
      uTime: sh.uTime, uFarSlots: { value: FAR.slots },
    }));
    const far = {
      ...map, uSwashFarView: sh.uSwashFarView, uSwashFarFoamT: sh.uSwashFarFoamT, uSwashFarWetT: sh.uSwashFarWetT, uTime: sh.uTime,
    };
    this.pShift = new FullscreenPass(makeShader(SHIFT, { ...base, ...far, uSrc: { value: null }, uBed: { value: this.bed.texture }, uShift: { value: 0 }, uMode: { value: 0 } }));
    this.pShiftFoam = new FullscreenPass(makeShader(SHIFT_FOAM, { ...base, ...far, uSrcA: { value: null }, uSrcB: { value: null }, uShift: { value: 0 } }));
    this.synced = false;
  }

  // shared uniforms added by modules constructed after this one (and new per-event arrays of the
  // breaker) reach this module's passes on the first step
  _syncShared() {
    this.synced = true;
    const sh = this.shared;
    for (const k of Object.keys(sh)) {
      if (/^uEvt[A-Z]$/.test(k) && !this.evtKeys.includes(k)) {
        this.evtKeys.push(k);
        this.histU[k] = { value: new Float32Array(FAR.slots * 4) };
        for (const p of [this.pParam, this.pSrc, this.pFoam]) { p.material.uniforms[k] = this.histU[k]; p.material.needsUpdate = true; }
      }
    }
    for (const p of [this.pSrc, this.pFoam, this.pHeight, this.pFoamView, this.pParam]) {
      const u = p.material.uniforms;
      let add = false;
      for (const k of Object.keys(sh)) if (!(k in u)) { u[k] = sh[k]; add = true; }
      if (add) p.material.needsUpdate = true;
    }
  }

  // remember newly packed events (true when the history changed)
  _updateHistory() {
    const sh = this.shared;
    const n = sh.uEvtCount.value, A = sh.uEvtA.value;
    let changed = false;
    for (let i = 0; i < n; i++) {
      const t0 = A[i * 4], seed = A[i * 4 + 3];
      const key = `${t0.toFixed(4)}|${seed.toFixed(5)}`;
      if (this.hist.has(key)) continue;
      const rec = { t0, v: {} };
      for (const k of this.evtKeys) rec.v[k] = Array.from(sh[k].value.subarray(i * 4, i * 4 + 4));
      this.hist.set(key, rec);
      changed = true;
    }
    if (!changed) return false;
    const list = [...this.hist.entries()].sort((a, b) => a[1].t0 - b[1].t0);
    while (list.length > FAR.slots) this.hist.delete(list.shift()[0]);
    list.forEach(([, rec], i) => { for (const k of this.evtKeys) if (rec.v[k]) this.histU[k].value.set(rec.v[k], i * 4); });
    this.histU.uEvtCount.value = list.length;
    return true;
  }

  _bakeParams() {
    this.shared.uSwashFarMap.value.x = this.cx;
    this.pParam.render(this.renderer, this.farParam);
  }

  _bakeFar(t) { this.farT = t ?? this.shared.uTime.value; this.pFar.render(this.renderer, this.farRT); }

  // scroll the window by k whole cells (+k: toward +x)
  _shift(k) {
    const r = this.renderer;
    this.cx += k * this.dx;
    this.dom.x += k * this.dx;
    this.shared.uSweDom.value.x = this.dom.x;
    this.shared.uSwashFarWin.value.x = this.cx;
    this.pBed.render(r, this.bed);
    this._bakeParams();
    this._bakeFar();
    const su = this.pShift.material.uniforms;
    su.uShift.value = k;
    su.uMode.value = 0; su.uSrc.value = this.state.read.texture;
    this.pShift.render(r, this.state.write); this.state.swap();
    su.uMode.value = 1; su.uSrc.value = this.wet.read.texture;
    this.pShift.render(r, this.wet.write); this.wet.swap();
    const fu = this.pShiftFoam.material.uniforms;
    const src = this.foamMRT[this.foamIdx];
    fu.uShift.value = k; fu.uSrcA.value = src.textures[0]; fu.uSrcB.value = src.textures[1];
    this.pShiftFoam.render(r, this.foamMRT[1 - this.foamIdx]);
    this.foamIdx = 1 - this.foamIdx;
  }

  // The focus: shared.uFocus (main.js sets it from the camera every rendered frame). A seek runs the
  // warm-up before the next render, so when the explore controls are around their position is
  // used directly (it is what the camera will be placed at).
  _focusX() {
    const c = typeof window !== 'undefined' ? window.__explore : null;
    return c && c.pos && Number.isFinite(c.pos.x) ? c.pos.x : this.shared.uFocus.value.x;
  }

  _follow() {
    const d = this._focusX() - this.cx;
    if (Math.abs(d) <= EXPLORE_WIN.follow) return false;
    const k = Math.round(d / this.dx);
    if (k !== 0) this._shift(k);
    return true;
  }

  _mrt() {
    const rt = new THREE.WebGLRenderTarget(this.nx, this.nz, {
      count: 2, type: THREE.FloatType, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
      depthBuffer: false, stencilBuffer: false, generateMipmaps: false,
    });
    rt.textures[0].type = THREE.HalfFloatType;
    for (const t of rt.textures) t.colorSpace = THREE.NoColorSpace;
    return rt;
  }

  reset() {
    const r = this.renderer;
    if (this.explore) {
      // restart centred on the focus (snapped to whole cells), with an empty event history
      this.cx = Math.round(this._focusX() / this.dx) * this.dx;
      this.dom.x = this.cx - 0.5 * this.nx * this.dx;
      this.shared.uSweDom.value.x = this.dom.x;
      this.shared.uSwashFarWin.value.x = this.cx;
      this.shared.uSwashFarMap.value.x = this.cx;
      this.hist.clear();
      this.histU.uEvtCount.value = 0;
      this.bakeDirty = true;
      this.farT = -1e9;
    }
    this.pBed.render(r, this.bed);
    this.pInit.render(r, this.state.read);
    this.pInit.render(r, this.state.write);
    const prevClear = r.getClearAlpha();
    r.setClearColor(0x000000, 0);
    for (const m of this.foamMRT) { r.setRenderTarget(m); r.clear(true, false, false); }
    r.setClearAlpha(prevClear);
    this.pWetInit.render(r, this.wet.read);
    this.pWetInit.render(r, this.wet.write);
    this.lastT = null;
    this._publish();
    this.rockWetness?.reset();
  }

  step(t, dt) {
    const r = this.renderer;
    if (this.explore) {
      if (!this.synced) this._syncShared();
      const newEvents = this._updateHistory();
      const moved = this._follow();                 // (re-bakes the far field when it scrolls)
      if (!moved) {
        if (newEvents || this.bakeDirty) this._bakeParams();
        // the far field is rendered once per frame: re-evaluate it at most at 60 Hz (0.13 ms)
        if (newEvents || this.bakeDirty || !(t - this.farT < 1 / 60 - 1e-6)) this._bakeFar(t);
      }
      this.bakeDirty = false;
    }
    const sub = this.substeps;
    const fu = this.pFlux.material.uniforms;
    fu.uDt.value = dt / sub;
    for (let i = 0; i < sub; i++) {
      fu.uState.value = this.state.read.texture;
      this.pFlux.render(r, this.state.write);
      this.state.swap();
    }
    const su = this.pSrc.material.uniforms;
    su.uState.value = this.state.read.texture;
    su.uDtS.value = dt;
    this.pSrc.render(r, this.state.write);
    this.state.swap();

    // foam + lace coordinates (MRT ping-pong)
    // each coordinate set is shown from 0 s to 0.94 s of age: a longer-lived frame is sheared into
    // comet streaks by the backwash (x3-5 in a fast, diverging sheet)
    const LACE_T = 1.3;
    const phase = (v) => ((v % 1) + 1) % 1;
    const phA = phase(t / LACE_T), phB = phase(t / LACE_T + 0.5);
    const prevA = this.lastT === null ? phA : phase(this.lastT / LACE_T);
    const prevB = this.lastT === null ? phB : phase(this.lastT / LACE_T + 0.5);
    const src = this.foamMRT[this.foamIdx], dst = this.foamMRT[1 - this.foamIdx];
    const fo = this.pFoam.material.uniforms;
    fo.uState.value = this.state.read.texture;
    fo.uFoamPrev.value = src.textures[0];
    fo.uUVPrev.value = src.textures[1];
    fo.uDtS.value = dt;
    fo.uResetA.value = phA < prevA ? 1 : 0;
    fo.uResetB.value = phB < prevB ? 1 : 0;
    this.pFoam.render(r, dst);
    this.foamIdx = 1 - this.foamIdx;
    this.shared.uLacePhase.value.set(phA, phB);
    const cycA = Math.floor(t / LACE_T), cycB = Math.floor(t / LACE_T + 0.5);
    this.shared.uLaceSeeds.value.set(((cycA * 0.6180339 + 0.13) % 1 + 1) % 1 * 10, ((cycB * 0.7548776 + 0.57) % 1 + 1) % 1 * 10);

    const wu = this.pWet.material.uniforms;
    wu.uState.value = this.state.read.texture;
    wu.uWetPrev.value = this.wet.read.texture;
    wu.uDtS.value = dt;
    this.pWet.render(r, this.wet.write);
    this.wet.swap();

    this.pView.material.uniforms.uState.value = this.state.read.texture;
    this.pView.render(r, this.view);
    this.pHeight.material.uniforms.uState.value = this.state.read.texture;
    this.pHeight.render(r, this.height);
    this.lastT = t;
    this._publish();
    this.rockWetness?.step(dt);
  }

  _publish() {
    const cur = this.foamMRT[this.foamIdx];
    const fv = this.pFoamView.material.uniforms;
    fv.uFoamT.value = cur.textures[0];
    this.pFoamView.render(this.renderer, this.foamView);
    this.shared.uSweFoam.value = this.foamView.texture;
    this.shared.uSweLace.value = cur.textures[1];
    this.shared.uSweWet.value = this.wet.read.texture;
  }

  // Debug readback (tools/swashprobe.mjs): full-precision copies of the solver fields.
  // Returns Float32Arrays of nx*nz*4: s = (h, u, v, B), f = foam (R, G, K, M), w = (film, damp, 0, 0).
  readback() {
    const r = this.renderer, n = this.nx * this.nz * 4;
    if (!this.pProbe) {
      this.probeRT = floatRT(this.nx, this.nz);
      this.pProbe = new FullscreenPass(makeShader(DOMAIN + /* glsl */ `
uniform sampler2D uState; uniform sampler2D uBed; uniform sampler2D uFoamT; uniform sampler2D uWetT; uniform int uSel;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 U = texelFetch(uState, p, 0).xyz; float B = texelFetch(uBed, p, 0).x; float h = max(U.x - B, 0.0);
  float h4 = h * h * h * h; vec2 vel = 1.41421356 * h * U.yz / sqrt(h4 + max(h4, 1.0e-8));
  if (uSel == 0) gl_FragColor = vec4(h, vel, B);
  else if (uSel == 1) gl_FragColor = texelFetch(uFoamT, p, 0);
  else gl_FragColor = vec4(texelFetch(uWetT, p, 0).xy, 0.0, 0.0);
}`, { uDom: this.pView.material.uniforms.uDom, uN: this.pView.material.uniforms.uN, uState: { value: null }, uBed: { value: this.bed.texture },
        uFoamT: { value: null }, uWetT: { value: null }, uSel: { value: 0 } }));
    }
    const u = this.pProbe.material.uniforms;
    u.uState.value = this.state.read.texture;
    u.uFoamT.value = this.foamMRT[this.foamIdx].textures[0];
    u.uWetT.value = this.wet.read.texture;
    const out = {};
    for (const [k, sel] of [['s', 0], ['f', 1], ['w', 2]]) {
      u.uSel.value = sel;
      this.pProbe.render(r, this.probeRT);
      out[k] = new Float32Array(n);
      r.readRenderTargetPixels(this.probeRT, 0, 0, this.nx, this.nz, out[k]);
    }
    r.setRenderTarget(null);
    return out;
  }
}
