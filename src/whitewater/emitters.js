// Hash-without-Sine routines: Copyright (c) 2014 David Hoskins, MIT.
// See licenses/Hash-without-Sine-MIT.txt and THIRD_PARTY_NOTICES.md.
// Whitewater emitters: what is thrown up, where along the crest and when.
//
// Every emitter is tied to a stage of the breaker (breaking_wave_crash_swash_math.md):
//   crest steepening   -> FSHEET / FDROP   crest feathering (patchy spray sheets + drops)
//   spilling crest     -> SPILL            whitecap roller riding the front face (spilling waves)
//   lip flight         -> LIPDROP          droplets pinching off the curtain edge
//   lip impact (V^2)   -> VEIL, ROLL, SPRAY  curtain veil, impact roll, impact droplets
//   splash-up          -> DOME, DROP, LIG  cauliflower dome (blob clusters), crown droplets
//                                            and ligaments revealed as the crown fragments
//   2nd splash cycle   -> JETB/JETD/JETL   narrow secondary jet at t_i + ~0.5 s
//   bore / roller      -> BOIL             churning roller body between the (shoreward-moving)
//                                            back edge and the bore front
//   bore front         -> BORE             aerated roller tumbling on the front of the bore; the
//                                            front is found in the shallow-water state (plus the
//                                            splash-driven surge right after a plunge), so it also
//                                            fires for stalled bores and after the event retires
// Rates are integrated on the CPU (deterministic CDF over x per emitter); the GPU draws x
// from the CDF and evaluates the exact local breaker state (glsl/breaker.js) there.
// Per-event gains from the schedule (e.splash / e.bore, uEvtG) scale the splash-up (dome, drops,
// ligaments, secondary jet) and the bore-front roller. In explore mode the bins cover
// focus.x +- EXPLORE_HALF with a distance LOD (see the end of this file); the time-independent part
// of each event at each bin x is cached on the event.
//
// Along-shore structure (breaker_morphology.md §3.2, §5.4): splash domes every ~2.7 m,
// alternating strong / weak, with low saddles between them (dome amplitude also scales
// with the local H^2), plus narrow secondary jets between the domes. The phase is derived
// from the event seed; for the clip's wave B (seed 0.73) the domes sit at x = -1.2 / +1.5
// and the secondary jet at x = +0.9, as measured.

import { brkTiming } from '../core/schedule.js';

export const K = {
  ROLL: 0, DOME: 1, VEIL: 2, FSHEET: 3, JETB: 4, BOIL: 5, BORE: 6, SPILL: 7, // blobs  (thickness pass)
  SPRAY: 8, DROP: 9, FDROP: 10, LIPDROP: 11, JETD: 12,                       // droplets (crisp pass)
  LIG: 13, JETL: 14,                                                         // ligaments (ribbon pass)
};
export const NKIND = 15;
// particle classes (separate ring buffers so each draw call only touches its own range)
export const CLASS_KINDS = [[0, 8], [8, 13], [13, 15]];

// splash-up launch after the local lip impact (s): where the break closes out the splash
// needs ~0.13 s to build up, where it peels fast (|dt/dx| > 0.1 s/m) the neighbouring, already
// broken section feeds it and it rises right away (measured on wave B, left vs right dome) - but
// never before the local impact (the lip draws its own white line there in the last ~0.07 s; an
// earlier launch put white ~0.5 m ahead of the reference peel point)
export const DOME_MIN_DELAY = 0.03;
export const DTDX_H = 0.45;      // half baseline of the peel rate dt_i/dx (m)
export const domeDelay = (dtdx) => Math.min(Math.max(0.25 - 2.5 * Math.abs(dtdx), DOME_MIN_DELAY), 0.25);
// the curtain veil starts this long before the local impact (the lip itself draws the veil before)
export const VEIL_LEAD = 0.01;
export const JET_DELAY = 0.5;     // secondary jet launch after local impact (s; wave B's jet tops out at ~4.0 s)
export const DOME_SPACING = 2.25; // m
// bore-front rollers found in the shallow-water state regardless of any event (per m per s)
export const BORE_BG = 1400;

const fract = (v) => v - Math.floor(v);
export function hash11(p) { p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
const gp = (tau, T) => (tau <= 0 ? 0 : (tau / (T * T)) * Math.exp(-tau / T));
const sstep = (a, b, v) => { const t = Math.min(Math.max((v - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };

// roller churn envelope after the local impact (a spilling break starts churning at once)
export const boilEnv = (tau, dec = 0.5, t0 = 0.12) => (tau <= t0 ? 0 : Math.min((tau - t0) / 0.15, 1) * Math.exp(-Math.max(tau - t0 - 0.15, 0) / dec));
// not every breaker throws a coherent secondary jet (wave B does, A and C do not)
export const hasJet = (seed) => (hash11(seed * 131.9) > 0.35 ? 1 : 0);
export const plungeFactor = (style) => sstep(0.35, 0.75, style);
// bore-front roller envelope: from the bore's formation until it has run up and stalled
export const boreEnv = (tau) => sstep(0.1, 0.3, tau) * (1 - sstep(1.5, 2.3, tau));
// spilling whitecap: from the moment the crest arrives and stands (spills) at the break until
// the roller has become the bore (D = stand time of this section, see brkTiming); a small
// reformed crest (clip's 'a') spills earlier and dies out before it would break
const spillStart = (D, hs) => Math.min(Math.max(-D - 0.1, -0.9), -0.3) - 0.3 * (1 - sstep(0.4, 0.7, hs));
const spillEnd = (hs) => 0.05 - 0.4 * (1 - sstep(0.4, 0.7, hs));
export const spillEnv = (tau, D, hs) => sstep(spillStart(D, hs), spillStart(D, hs) + 0.15, tau) * (1 - sstep(spillEnd(hs), spillEnd(hs) + 0.3, tau));
function domePhase(seed) { return DOME_SPACING * fract(seed * 3.7) - 2.8273; }

/** splash-dome amplitude 0..1 along the crest */
export function domeField(x, seed) {
  const S = DOME_SPACING, ph = domePhase(seed);
  const k0 = Math.floor((x - ph) / S + 0.5);
  let a = 0;
  for (let j = -1; j <= 1; j++) {
    const k = k0 + j;
    const c = ph + k * S + 0.2 * (hash11(k * 7.13 + seed * 91.7) - 0.5);
    const even = ((k % 2) + 2) % 2 === 0;
    const amp = (even ? 1.0 : 0.85) * (0.92 + 0.16 * hash11(k * 3.71 + seed * 17.3));
    const sg = even ? (x < c ? 0.3 : 0.6) : (x < c ? 0.4 : 0.6);   // skewed like wave B's domes
    a = Math.max(a, amp * Math.exp(-0.5 * ((x - c) / sg) ** 2));
  }
  return a;
}
/** secondary-jet mask 0..1: a 0.4 m wide jet between each strong dome and the next one, and a
 *  weaker one on the other side of the strong dome (wave B: x = +0.9 and x = -0.9) */
const JET_OFF = [[2.15, 1.0], [0.35, 0.6]];
export function jetMask(x, seed) {
  const S = DOME_SPACING, ph = domePhase(seed);
  let m = 0;
  for (const [off, amp] of JET_OFF) {
    const k0 = Math.floor((x - ph - off) / (2 * S) + 0.5);
    for (let j = -1; j <= 1; j++) {
      const c = ph + (k0 + j) * 2 * S + off;
      m = Math.max(m, amp * Math.exp(-0.5 * ((x - c) / 0.17) ** 2));
    }
  }
  return m;
}
/** crest feathering: patches 0.4-0.8 m long, 1-2 m apart, each lasting 0.2-0.4 s between
 *  t_i - 0.75 and t_i - 0.15 s (it starts as soon as the front face passes ~50 deg). Returns a
 *  rate density that integrates to 1 over a patch life. */
export function featherPatch(x, tau, seed) {
  let r = 0;
  for (let slot = 0; slot < 3; slot++) {
    const xo = hash11(slot * 5.3 + seed * 13.1) * 1.9;
    const cell = Math.floor((x + xo) / 1.9);
    const h1 = hash11(cell * 1.91 + slot * 7.7 + seed * 31.3);
    const h2 = hash11(cell * 3.17 + slot * 2.9 + seed * 11.9);
    const h3 = hash11(cell * 0.73 + slot * 4.1 + seed * 5.7);
    const cx = (cell + 0.3 + 0.4 * h1) * 1.9 - xo;
    const half = 0.2 + 0.2 * h2;
    const t0 = -0.75 + 0.25 * h3 + slot * 0.12;
    const dur = 0.2 + 0.2 * h2;
    if (Math.abs(x - cx) < half && tau > t0 && tau < t0 + dur) r += 1 / dur;
  }
  return r;
}

// particles per metre of crest (per event, at H = 0.45 m) and their time profiles
const N = {
  ROLL: 1500, DOME: 7500, VEIL: 800, SPRAY: 900, DROP: 2800, LIG: 400,
  FSHEET: 650, FDROP: 550, LIPDROP: 225, JETB: 900, JETD: 6000, JETL: 160, BOIL: 3200, BORE: 4200,
  SPILL: 6400,
};

// Time-independent part of an event at x (timing, dome / jet structure), cached per event:
// the emission is integrated every sim step over up to ~200 bins (explore mode: focus +- 25 m)
// and only the time profiles change from step to step.
function evtSig(e) { return e.t0 * 1.37 + e.H * 7.1 + e.seed * 13.3 + (e.x0 || 0) * 17.9 + (e.aR1 || 0) * 3.1 + (e.aL1 || 0) * 5.3 + (e.zI || 0) * 2.9; }
function evtAt(e, x) {
  let m = e._ww;
  const sig = evtSig(e);
  if (!m || e._wwSig !== sig) { m = e._ww = new Map(); e._wwSig = sig; }
  const key = Math.round(x * 1e4);
  let c = m.get(key);
  if (!c) {
    if (m.size > 8192) m.clear();
    const b = brkTiming(e, x);
    // (the peel rate over +-0.45 m: the along-crest staircase of the break must not make each 0.45 m
    //  section launch its own dome at its own time - towers)
    const tR = brkTiming(e, x + DTDX_H).ti, tL = brkTiming(e, x - DTDX_H).ti;
    const dtdx = (tR - tL) / (2 * DTDX_H);
    // the splash-up launch clock: the local impact time smoothed over +-0.45 m (the lip lands in
    // 0.45 m sections, but the splash of neighbouring sections merges into one dome)
    const tiS = 0.25 * (tL + tR) + 0.5 * b.ti;
    c = { b, dtdx, tiS, A: domeField(x, e.seed), jm: jetMask(x, e.seed) * hasJet(e.seed), gS: e.splash ?? 1, gB: e.bore ?? 1 };
    m.set(key, c);
  }
  return c;
}

/** out[k] += rate (particles / s / m) of emitter k for event e at (x, t) */
export function emitterRates(e, x, t, out) {
  if (!(e.strength > 0)) return;
  const c = evtAt(e, x);
  const b = c.b;
  const tau = t - b.ti;
  if (tau < -1.0 || tau > 2.4) return;
  const hs = b.H / 0.45;
  const E = hs * hs * e.strength;
  const A = c.A;
  // splash-up needs a plunging impact: spilling / collapsing breakers (style < 0.5) barely dome
  // (per-event splash gain: schedule e.splash, uEvtG.y)
  const P = plungeFactor(e.style);
  const dome = E * (0.3 + 0.7 * A) * (0.25 + 0.75 * P) * c.gS;
  const dtdx = c.dtdx;
  const tl = t - c.tiS - domeDelay(dtdx);
  const pDome = gp(tl + 0.03, 0.065);
  const pBase = 0.12 * gp(tl - 0.08, 0.2);   // the base keeps boiling up after the main splash
  // impact roll burst, then the plunge point keeps boiling (roller) for ~0.8 s
  out[K.ROLL] += N.ROLL * E * (gp(tau + 0.02, 0.045) + 0.4 * gp(tau - 0.04, 0.22));
  // spilling crest roller (whitecap tumbling down the front before and while it breaks)
  // (a small reformed crest only gets the breaker's own crest foam)
  out[K.SPILL] += N.SPILL * hs * Math.sqrt(hs) * e.strength * sstep(0.4, 0.85, e.spill) * sstep(0.42, 0.6, hs) * spillEnv(tau, b.D, hs);
  out[K.DOME] += N.DOME * dome * (pDome + pBase);
  out[K.DROP] += N.DROP * dome * pDome;
  out[K.LIG] += N.LIG * dome * pDome;
  out[K.VEIL] += N.VEIL * E * gp(tau + VEIL_LEAD, 0.06);
  out[K.SPRAY] += N.SPRAY * E * (gp(tau + 0.01, 0.07) + 0.6 * boilEnv(tau) * (tau > 0.2 ? 1 : 0));
  const fp = featherPatch(x, tau, e.seed) * hs;
  out[K.FSHEET] += N.FSHEET * fp;
  out[K.FDROP] += N.FDROP * fp;
  if (t > b.tb + 0.08 && t < b.ti) out[K.LIPDROP] += N.LIPDROP * hs;
  const jm = c.jm * P;
  if (jm > 0.02) {
    const pj = gp(tau - JET_DELAY + 0.03, 0.045) * jm * E * c.gS;
    out[K.JETB] += N.JETB * pj;
    out[K.JETD] += N.JETD * pj;
    out[K.JETL] += N.JETL * pj;
  }
  // the energy a spilling / collapsing breaker does not throw up into a dome goes into a
  // long-lived turbulent roller and a stronger bore roller
  // (a small reformed crest spills without a churning roller worth the name)
  out[K.BOIL] += N.BOIL * E * (1 + 1.2 * (1 - P)) * sstep(0.3, 0.55, hs) * boilEnv(tau, 0.5 + 0.6 * (1 - P), 0.12 * P);
  out[K.BORE] += N.BORE * E * (1 + 1.0 * (1 - P)) * boreEnv(tau) * Math.sqrt(c.gB);
}

// GLSL mirror of the functions above (+ kind ids)
export const EMIT_GLSL = /* glsl */ `
#define K_ROLL 0
#define K_DOME 1
#define K_VEIL 2
#define K_FSHEET 3
#define K_JETB 4
#define K_BOIL 5
#define K_BORE 6
#define K_SPILL 7
#define K_SPRAY 8
#define K_DROP 9
#define K_FDROP 10
#define K_LIPDROP 11
#define K_JETD 12
#define K_LIG 13
#define K_JETL 14
#define NKIND 15
#define JET_DELAY ${JET_DELAY.toFixed(3)}
#define VEIL_LEAD ${VEIL_LEAD.toFixed(3)}
#define DOME_S ${DOME_SPACING.toFixed(3)}
float plungeFactor(float style) { return smoothstep(0.35, 0.75, style); }
float gpulse(float tau, float T) { return tau <= 0.0 ? 0.0 : tau / (T * T) * exp(-tau / T); }
float boilEnvG(float tau, float dec, float t0) { return tau <= t0 ? 0.0 : min((tau - t0) / 0.15, 1.0) * exp(-max(tau - t0 - 0.15, 0.0) / dec); }
float spillStart(float D, float hs) { return clamp(-D - 0.1, -0.9, -0.3) - 0.3 * (1.0 - smoothstep(0.4, 0.7, hs)); }
float spillEnd(float hs) { return 0.05 - 0.4 * (1.0 - smoothstep(0.4, 0.7, hs)); }
float domePhase(float seed) { return DOME_S * fract(seed * 3.7) - 2.8273; }
float domeField(float x, float seed) {
  float ph = domePhase(seed);
  float k0 = floor((x - ph) / DOME_S + 0.5);
  float a = 0.0;
  for (int j = -1; j <= 1; j++) {
    float k = k0 + float(j);
    float c = ph + k * DOME_S + 0.2 * (hash11(k * 7.13 + seed * 91.7) - 0.5);
    bool even = mod(k, 2.0) < 0.5;
    float amp = (even ? 1.0 : 0.85) * (0.92 + 0.16 * hash11(k * 3.71 + seed * 17.3));
    float sg = even ? (x < c ? 0.3 : 0.6) : (x < c ? 0.4 : 0.6);
    a = max(a, amp * exp(-0.5 * pow((x - c) / sg, 2.0)));
  }
  return a;
}
float jetCentre(float x, float seed) {
  float ph = domePhase(seed);
  float c1 = ph + floor((x - ph - 2.15) / (2.0 * DOME_S) + 0.5) * 2.0 * DOME_S + 2.15;
  float c2 = ph + floor((x - ph - 0.35) / (2.0 * DOME_S) + 0.5) * 2.0 * DOME_S + 0.35;
  return abs(x - c1) < abs(x - c2) ? c1 : c2;
}
float jetMask(float x, float seed) {
  float ph = domePhase(seed);
  float c1 = ph + floor((x - ph - 2.15) / (2.0 * DOME_S) + 0.5) * 2.0 * DOME_S + 2.15;
  float c2 = ph + floor((x - ph - 0.35) / (2.0 * DOME_S) + 0.5) * 2.0 * DOME_S + 0.35;
  return max(exp(-0.5 * pow((x - c1) / 0.17, 2.0)), 0.6 * exp(-0.5 * pow((x - c2) / 0.17, 2.0)));
}
float hasJetG(float seed) { return step(0.35, hash11(seed * 131.9)); }
float featherPatch(float x, float tau, float seed) {
  float r = 0.0;
  for (int slot = 0; slot < 3; slot++) {
    float fs = float(slot);
    float xo = hash11(fs * 5.3 + seed * 13.1) * 1.9;
    float cell = floor((x + xo) / 1.9);
    float h1 = hash11(cell * 1.91 + fs * 7.7 + seed * 31.3);
    float h2 = hash11(cell * 3.17 + fs * 2.9 + seed * 11.9);
    float h3 = hash11(cell * 0.73 + fs * 4.1 + seed * 5.7);
    float cx = (cell + 0.3 + 0.4 * h1) * 1.9 - xo;
    float half_ = 0.2 + 0.2 * h2;
    float t0 = -0.75 + 0.25 * h3 + fs * 0.12;
    float dur = 0.2 + 0.2 * h2;
    if (abs(x - cx) < half_ && tau > t0 && tau < t0 + dur) r += 1.0 / dur;
  }
  return r;
}
// splash-up launch time of event i at x: the smoothed impact time (see evtAt) + the build-up delay
float domeLaunch(int i, float x, float ti) {
  float tR = brkAt(i, x + ${DTDX_H.toFixed(2)}).ti, tL = brkAt(i, x - ${DTDX_H.toFixed(2)}).ti;
  float dtdx = (tR - tL) / ${(2 * DTDX_H).toFixed(2)};
  return 0.25 * (tL + tR) + 0.5 * ti + clamp(0.25 - 2.5 * abs(dtdx), ${DOME_MIN_DELAY.toFixed(3)}, 0.25);
}
// time profile of emitter k for event i (used to attribute a newborn to its event)
float emitWeight(int k, int i, Brk b, float x) {
  float tau = uTime - b.ti;
  float P = plungeFactor(b.style);
  float w = 0.0;
  if (k == K_ROLL) w = gpulse(tau + 0.02, 0.045) + 0.4 * gpulse(tau - 0.04, 0.22);
  else if (k == K_SPILL) { float hs = b.H / 0.45, s0 = spillStart(b.D, hs), s1 = spillEnd(hs); w = smoothstep(0.4, 0.85, b.spill) * smoothstep(s0, s0 + 0.15, tau) * (1.0 - smoothstep(s1, s1 + 0.3, tau)); }
  else if (k == K_DOME || k == K_DROP || k == K_LIG) {
    float tl = uTime - domeLaunch(i, x, b.ti);
    w = gpulse(tl + 0.03, 0.065) + (k == K_DOME ? 0.12 * gpulse(tl - 0.08, 0.2) : 0.0);
  }
  else if (k == K_VEIL) w = gpulse(tau + VEIL_LEAD, 0.06);
  else if (k == K_SPRAY) w = gpulse(tau + 0.01, 0.07) + (tau > 0.2 ? 0.6 * boilEnvG(tau, 0.5, 0.12) : 0.0);
  else if (k == K_FSHEET || k == K_FDROP) w = featherPatch(x, tau, b.seed);
  else if (k == K_LIPDROP) w = (uTime > b.tb + 0.08 && uTime < b.ti) ? 1.0 : 0.0;
  else if (k == K_JETB || k == K_JETD || k == K_JETL) w = gpulse(tau - JET_DELAY + 0.03, 0.045) * jetMask(x, b.seed) * hasJetG(b.seed);
  else if (k == K_BORE) w = (1.0 + (1.0 - P)) * smoothstep(0.1, 0.3, tau) * (1.0 - smoothstep(1.5, 2.3, tau));
  else w = (1.0 + 1.2 * (1.0 - P)) * boilEnvG(tau, 0.5 + 0.6 * (1.0 - P), 0.12 * P);
  // per-event gains (schedule e.splash / e.bore, uEvtG), as in emitterRates()
  if (k == K_DOME || k == K_DROP || k == K_LIG || k == K_JETB || k == K_JETD || k == K_JETL) w *= uEvtG[i].y;
  else if (k == K_BORE) w *= sqrt(max(uEvtG[i].x, 0.0));
  return w;
}
`;

// ---------------------------------------------------------------- explore mode: emission window
// The whitewater is emitted over focus.x +- EXPLORE_HALF m (like the lip ribbons: +- 40 m), fading
// out over the last EXPLORE_FADE m. Bins are fine (5 cm) at the focus and widen with the along-shore
// distance (0.45 m at 13 m, 0.8 m beyond 25 m), so the along-crest structure (domes, jets) is
// resolved where the player can see it and the CPU integration stays cheap.
export const EXPLORE_HALF = 40.0, EXPLORE_FADE = 8.0;
export const exploreEdgeFade = (dx) => { const t = Math.min(Math.max((Math.abs(dx) - (EXPLORE_HALF - EXPLORE_FADE)) / EXPLORE_FADE, 0), 1); return 1 - t * t * (3 - 2 * t); };
export function exploreBinEdges(R = EXPLORE_HALF, a = 0.05, b = 0.03, c = 0.8) {
  const half = [0];
  let d = 0;
  while (d < R - 1e-6) {
    const w0 = Math.min(a + b * d, c);
    const w = Math.min(a + b * (d + 0.5 * w0), c);
    d = Math.min(d + w, R);
    if (R - d < 0.3 * w) d = R;          // no sliver at the end
    half.push(d);
  }
  const edges = [];
  for (let i = half.length - 1; i > 0; i--) edges.push(-half[i]);
  for (const v of half) edges.push(v);
  return Float64Array.from(edges);
}
// Level of detail by distance from the player: fewer, larger parcels far away (count x f, radius
// x f^-1/2, density x f^1/2: same coverage and optical depth per parcel). Mirrored in GLSL (lodAt).
export const LOD = { D0: 6.5, P: 1.75, MIN: 0.07, ZB: -2.0 };
export function lodAt(x, fx, fz) {
  const d = Math.hypot(x - fx, LOD.ZB - fz);
  return Math.min(Math.max(Math.pow(LOD.D0 / Math.max(d, 1e-3), LOD.P), LOD.MIN), 1);
}
