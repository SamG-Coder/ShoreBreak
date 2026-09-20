import { CONFIG } from '../config.js';
import { SWELL_TRANSPORT } from './swell.js';
// Analytic shore-break model shared by the water surface, the plunging-lip sheet
// (water/LipRibbon.js), the whitewater particles and the shallow-water injection.
//
// Causal chain (breaking_wave_crash_swash_math.md):
//   shoaling swell (long flat back, front face 7 m -> 1 m, crest +0.24 -> +0.55 m)
//   -> steepening: the front goes from a raised-cosine face to a vertical wall (t_i - 0.40 s)
//   -> lip ejection at t_i - 0.33 s: the crest front is thrown ballistically,
//      x = x0 + u0 t, y = y0 + w0 t - g t^2 / 2 (§6); the apex collapses in place
//   -> impact at t_i on the trough in front of the toe; impact speed Vimp drives
//      splash / foam / spray ∝ Vimp^2 (§9, §44)
//   -> the crest remnant collapses; its mass and momentum are injected into the
//      shallow-water solver, which carries the bore, swash and backwash (§15-§26).
//
// The profile is keyed to the reference clip. The stage table below is the wave-B
// morphology (breaker_morphology.md §6, stages S1-S8) re-derived for OUR camera by
// matching image rows (heights scaled x1.15, trough -0.07 m; the transformed tip
// trace reproduces the measured lip-line v(t) to ~1 px) plus the approach track of
// the hump. Along-crest timing t_i(x) is a logistic peel (§42/§43) plus an optional
// quadratic peel about a nucleation point (wave C, softened: a U, not a V), evaluated on a
// mildly stepped x (the crest closes out in sections), and a small wobble. Along a long beach
// (explore mode) that local peel saturates beyond |x| > 5 m and a bounded multi-nucleation
// modulation takes over (peaks every 10-25 m peeling both ways; schedule.js has the formulas and
// the exact JS mirror brkTiming). The ballistic lip tip (the "knee" of the sheet) follows the
// white lip-line traces of the reference timestacks.
//
// Units: the event height H is the NOMINAL breaker height of the analyses (0.45 m for
// wave B). Rendered heights are camera-matched: crest +0.553 m, trough -0.068 m for
// H = 0.45 (BRK_VIS = 1.38). Size scales with s = H / 0.45, times with sqrt(s).


// ---------------------------------------------------------------- stage table (s = 1)
// tau: normalised time to impact. za: apex z relative to the landing line. ya: apex
// height, yt: level of the trough ahead (m, rendered). lf: front-face length. Front
// face shape: mix(raised cosine, superellipse cap + concave power foot, st), the cap
// ends in a vertical wall at uw*lf whose foot starts at height fraction hw. lb: back
// face length (sech^2). S1..S8 of the morphology are the knots -0.8 .. 0.0.
export const BK_TABLE = {
  T:  [-6.0, -4.0, -3.0, -2.0, -1.4, -1.0, -0.8, -0.55, -0.40, -0.33, -0.22, -0.12, -0.04, 0.0, 0.12, 0.30, 0.60, 0.85],
  ZA: [-23.0, -15.5, -11.3, -7.0, -4.3, -2.55, -1.85, -1.08, -0.625, -0.511, -0.461, -0.472, -0.477, -0.481, -0.475, -0.46, -0.44, -0.43],
  YA: [0.20, 0.24, 0.28, 0.33, 0.41, 0.46, 0.50, 0.548, 0.56, 0.553, 0.484, 0.392, 0.346, 0.300, 0.200, 0.080, 0.010, 0.0],
  YT: [-0.10, -0.10, -0.10, -0.10, -0.09, -0.08, -0.068, -0.068, -0.068, -0.068, -0.068, -0.068, -0.068, -0.068, -0.06, -0.04, -0.02, 0.0],
  LF: [8.0, 7.0, 6.0, 4.0, 2.2, 1.4, 1.04, 0.60, 0.46, 0.42, 0.42, 0.40, 0.38, 0.36, 0.40, 0.60, 1.0, 1.2],
  UW: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.26, 0.12, 0.04, 0.02, 0.02, 0.02, 0.02, 0.05, 0.2, 0.3, 0.3],
  HW: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.62, 0.66, 0.54, 0.56, 0.53, 0.47, 0.41, 0.40, 0.5, 0.5, 0.5],
  M:  [1.6, 1.6, 1.6, 1.6, 1.6, 1.8, 2.0, 2.6, 3.2, 3.6, 4.5, 5.0, 5.5, 6.0, 4.0, 2.0, 2.0, 2.0],
  P:  [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0],
  ST: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.3, 0.0, 0.0],
  LB: [4.0, 3.8, 3.6, 3.2, 2.9, 2.7, 2.6, 2.5, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.6, 3.0, 3.0],
};

// Catmull-Rom (non-uniform knots) cubic of segment i of a column, as polynomial
// coefficients in f = (tn - T[i]) / (T[i+1] - T[i]).
function hermCoef(A, T, i) {
  const n = T.length, i0 = Math.max(i - 1, 0), i3 = Math.min(i + 2, n - 1);
  const h = T[i + 1] - T[i];
  const m1 = ((A[i + 1] - A[i0]) / Math.max(T[i + 1] - T[i0], 1e-4)) * h;
  const m2 = ((A[i3] - A[i]) / Math.max(T[i3] - T[i], 1e-4)) * h;
  const a = A[i], b = A[i + 1];
  // h00 a + h10 m1 + h01 b + h11 m2
  return [a, m1, -3 * a - 2 * m1 + 3 * b - m2, 2 * a + m1 - 2 * b + m2];
}
/** JS evaluation of the stage table (mirrors the generated GLSL exactly). */
export function brkStageJS(tn) {
  const K = BK_TABLE, T = K.T, n = T.length;
  let i = 0;
  for (let k = 1; k < n - 1; k++) if (tn >= T[k]) i = k;
  const f = Math.min(Math.max((tn - T[i]) / (T[i + 1] - T[i]), 0), 1);
  const poly = (c) => c[0] + f * (c[1] + f * (c[2] + f * c[3]));
  const lin = (A) => A[i] + (A[i + 1] - A[i]) * f;
  return { za: poly(hermCoef(K.ZA, T, i)), ya: poly(hermCoef(K.YA, T, i)), yt: lin(K.YT), lf: lin(K.LF), uw: lin(K.UW),
    hw: lin(K.HW), m: lin(K.M), p: lin(K.P), st: lin(K.ST), lb: lin(K.LB) };
}
function stageGLSL() {
  const K = BK_TABLE, T = K.T, n = T.length;
  const f = (v) => { const s = (+v).toPrecision(7); return /[.e]/.test(s) ? s : s + '.0'; };
  const v4 = (a) => `vec4(${a.map(f).join(', ')})`;
  let code = `BrkStage brkStage(float tn) {
  float t0, iw;
  vec4 cz, cy, l0, l1, q0, q1;   // cubic za, cubic ya, (yt, lf, uw, hw) and (m, p, st, lb) start / delta
`;
  for (let i = 0; i < n - 1; i++) {
    const lin0 = [K.YT[i], K.LF[i], K.UW[i], K.HW[i]], lin1 = [K.YT[i + 1] - K.YT[i], K.LF[i + 1] - K.LF[i], K.UW[i + 1] - K.UW[i], K.HW[i + 1] - K.HW[i]];
    const q0 = [K.M[i], K.P[i], K.ST[i], K.LB[i]], q1 = [K.M[i + 1] - K.M[i], K.P[i + 1] - K.P[i], K.ST[i + 1] - K.ST[i], K.LB[i + 1] - K.LB[i]];
    const cond = i < n - 2 ? `if (tn < ${f(T[i + 1])}) ` : '';
    code += `  ${i ? 'else ' : ''}${cond}{ t0 = ${f(T[i])}; iw = ${f(1 / (T[i + 1] - T[i]))}; cz = ${v4(hermCoef(K.ZA, T, i))}; cy = ${v4(hermCoef(K.YA, T, i))};
    l0 = ${v4(lin0)}; l1 = ${v4(lin1)}; q0 = ${v4(q0)}; q1 = ${v4(q1)}; }
`;
  }
  code += `  float fr = clamp((tn - t0) * iw, 0.0, 1.0);
  vec4 fp = vec4(1.0, fr, fr * fr, fr * fr * fr);
  vec4 l = l0 + l1 * fr, q = q0 + q1 * fr;
  BrkStage g;
  g.za = dot(cz, fp); g.ya = dot(cy, fp);
  g.yt = l.x; g.lf = l.y; g.uw = l.z; g.hw = l.w;
  g.m = q.x; g.p = q.y; g.st = q.z; g.lb = q.w;
  return g;
}`;
  return code;
}

export const BREAKER = (CONFIG.explore ? '#ifndef OPT_EXPLORE\n#define OPT_EXPLORE\n#endif\n' : '') + /* glsl */ `
#define MAX_EVENTS 6
${SWELL_TRANSPORT}
uniform vec4 uEvtA[MAX_EVENTS]; // x: t0 impact-time base (s), y: H nominal breaker height (m), z: impact (lip landing) z at x=0 (m), w: seed
uniform vec4 uEvtB[MAX_EVENTS]; // x: c bore / crest speed at the break (m/s), y: x0 nucleation x (m), z: timing wobble (s), w: strength (0 = off)
uniform vec4 uEvtC[MAX_EVENTS]; // x: lip speed offset kappa, y: landing-line waviness (m), z: height variation (rel), w: plunge style 0..1
uniform vec4 uEvtD[MAX_EVENTS]; // quadratic peel about x0: ti += aR1 ds + aR2 ds^2 (x >= x0), aL1 ds + aL2 ds^2 (x < x0), ds = sqrt(d^2 + 0.25) - 0.5
uniform vec4 uEvtE[MAX_EVENTS]; // x: height gain along the logistic peel (rel), y: stand time (s), z: spill (whitecap) 0..1, w: landing-line tilt dz/dx
uniform vec4 uEvtF[MAX_EVENTS]; // logistic peel: ti += x / (1 + exp(-(x - y) / z)); w: stall = fraction of the peel delay spent standing
uniform vec4 uEvtG[MAX_EVENTS]; // x: bore gain, y: splash gain (per-event multipliers, schedule e.bore / e.splash; read by the
                                // swash / whitewater modules), z: along-shore timing modulation amplitude Af (s; > 0: applies
                                // everywhere, < 0: only beyond |x| > 5 m), w: its phase (schedule e.far / e.farPhase)
uniform int uEvtCount;
uniform float uTime;

#define BRK_HREF 0.45
#define BRK_TB (-0.42)     // lip birth (normalised tau)
#define BRK_TE (-0.33)     // lip release: ballistic from here on
#define BRK_TEND 0.85      // end of the event (normalised tau)

struct Brk {
  float ti;    // impact time of the lip tip at this x
  float tb;    // lip release (break onset)
  float Tj;    // flight time of the lip tip
  float H;     // nominal breaker height (trough to crest, analysis metric)
  float c;     // bore / crest speed at the break (for the swash coupling and particles)
  float cFar;  // height gain along the logistic peel
  float tauS;  // stand time at the break (s) common to the whole crest
  float zI;    // tip landing z
  float zB;    // apex z at lip release
  float kap;   // lip speed offset
  float vy0;   // vertical launch speed of the lip tip
  float Tr;    // crest collapse time scale
  float style;
  float spill;
  float str;
  float seed;
  float s;     // size scale H / 0.45
  float rs;    // sqrt(s): time / speed scale
  float x;     // along-shore position of this slice
  float D;     // stall (s): the late-breaking sections arrive earlier and stand before they throw
  float hold;  // post-impact hold of a spilling crest (normalised s; 0 for plungers, brkTn)
};

// ---------------------------------------------------------------- stage table (s = 1)
// Generated from BK_TABLE (JS, above): one branch per segment with literal polynomial
// coefficients (no dynamic indexing of constant arrays, which is slow under ANGLE/D3D).
struct BrkStage { float za, ya, yt, lf, uw, hw, m, p, st, lb; };
${stageGLSL()}

// rendered crest elevation / trough level for a nominal height H (at the break)
float brkEtaC(float H) { return 1.229 * H; }   // +0.553 m for H = 0.45
float brkEtaT(float H) { return -0.151 * H; }  // -0.068 m

// Along-crest staircase: the crest closes out in ~0.45 m sections (the measured tb(x) of wave B
// has plateaus and jumps). The peel delay is evaluated at a stepped x: a mild staircase (a full
// one turns the lip into boxes; the steep curtain edge at the peel point comes from the curtain
// hanging below the knee, LipRibbon). The phase comes from the seed (wave B, seed 0.73: fitted
// to the timeline tb table). (Softened: with K 0.35 and 30 % ramps the local peel rate jumped
// between 0.65x and 2.4x at each step and the face shading creased along x there; now 0.75x ..
// 1.45x over 55 % ramps: the sections still close out in steps, no crease.)
#define BRK_STAIR_L 0.45
#define BRK_STAIR_K 0.25
float brkStairX(float x, float seed) {
  float s = x / BRK_STAIR_L + fract(seed * 3.0 + 0.485);
  return x + BRK_STAIR_K * BRK_STAIR_L * (floor(s) + smoothstep(0.45, 1.0, fract(s)) - s + 0.175);
}
// Peel delay at x: quadratic about the nucleation point x0 in a softened distance (a U with a flat
// ~1 m bottom, not a V cusp; a negative quadratic term saturates at its maximum) + logistic peel.
float brkPeel(vec4 D, vec4 F, float x0, float x) {
  float d = x - x0;
  float ds = sqrt(d * d + 0.25) - 0.5;
  float a1 = d >= 0.0 ? D.x : D.z, a2 = d >= 0.0 ? D.y : D.w;
  if (a2 < 0.0) ds = min(ds, -a1 / (2.0 * a2));
  return a1 * ds + a2 * ds * ds + F.x / (1.0 + exp(-(x - F.y) / max(F.z, 1e-3)));
}
// The local peel is evaluated at a clamped x: identity within the clip's view (|x| <= 5 m), then
// saturating smoothly (slope 1 at the knee, rational: cheaper than tanh) toward +-8 m, so a
// quadratic peel stays bounded along a long beach.
#define BRK_XCLAMP 5.0
#define BRK_XSAT 3.0
float brkClampX(float x) {
  float e = max(abs(x) - BRK_XCLAMP, 0.0);
  return x - sign(x) * (e - BRK_XSAT * e / (e + BRK_XSAT));
}
// Bounded multi-nucleation modulation of the timing along the beach (|S| <= 1): its minima are the
// peaks where a section breaks first (median spacing 16 m, 10-23 m) and it peels both ways from
// there (median 8-10 m/s for Af = 1-1.5 s). The phase comes from the schedule (e.farPhase).
// Returns (S, dS/dx) (the staircase offsets x by < 7 cm: S(xs) = S + S' (xs - x) to 1e-3 s).
vec2 brkFarSD(float x, float ph) {
  vec3 a = vec3(0.185, 0.33, 0.57) * x + vec3(6.2831853 * ph, 17.1 * ph + 1.3, 41.7 * ph + 2.9);
  return vec2(dot(vec3(0.55, 0.30, 0.15), sin(a)), dot(vec3(0.10175, 0.099, 0.0855), cos(a)));
}
// (no ternaries around calls, no && in these small helpers: ANGLE turns them into if/else, and with
// brkAt / brkTn inlined at dozens of call sites the D3D compile time explodes)
float brkFarW(float far, float x) { return mix(smoothstep(5.0, 13.0, abs(x)), 1.0, step(1e-9, far)) * step(1e-9, abs(far)); }
#define BRK_FAR_STAND 0.25   // fraction of the modulation delay spent standing (feathering shoulder)
#define BRK_FAR_H 0.12       // the peak is the bigger wave ...
#define BRK_FAR_Z 0.25       // ... and breaks a little further out (m)
#define BRK_TILT_L 12.0      // the landing-line tilt is a bounded bend (m)
#define BRK_HOLD 0.5         // post-impact hold of a spilling crest (normalised s, brkTn)
Brk brkAt(int i, float x) {
  vec4 A = uEvtA[i], B = uEvtB[i], C = uEvtC[i], D = uEvtD[i], E = uEvtE[i], F = uEvtF[i], G = uEvtG[i];
  Brk b;
  float sd = A.w;
  b.x = x;
  float xs = brkStairX(x, sd);
  // measured local peel (video events; the modulation replaces it everywhere when G.z > 0)
  // (branch-free: brkAt is inlined at many call sites)
  float useL = 1.0 - step(1e-9, G.z);
  float xc = brkClampX(x);
  float peelS = useL * brkPeel(D, F, B.y, xc);               // smooth: clocks the arrival
  float peel = useL * brkPeel(D, F, B.y, brkClampX(xs));     // stepped: the break itself
  float lg = useL * F.x / (1.0 + exp(-(xc - F.y) / max(F.z, 1e-3)));
  float stand = F.w * peelS;
  // along a long beach (beyond |x| > 5 m for the video events) the local peel hands over to the
  // bounded modulation
  float Af = abs(G.z);
  float wS = brkFarW(G.z, x);
  vec2 fm = brkFarSD(x, G.w);
  peelS = mix(peelS, Af * fm.x, wS);
  peel = mix(peel, Af * (fm.x + fm.y * (xs - x)), wS);
  stand = mix(stand, BRK_FAR_STAND * Af * (fm.x + 1.0), wS);
  lg *= 1.0 - wS;
  float fS = fm.x * wS;
  // the sections arrive smoothly; the steps are taken up by standing (>= 0.05 s) at the break
  b.D = max(max(E.y, 0.05) + stand + (peel - peelS), 0.0);
  float wob = 0.62 * sin(x * 2.3 + sd * 6.1) + 0.38 * sin(x * 5.1 + sd * 11.7);
  b.ti = A.x + peel + B.z * wob;
  b.H = A.y * (1.0 + C.z * (0.65 * sin(x * 1.1 + sd * 3.3) + 0.35 * sin(x * 2.9 + sd * 8.9)));
  // the late (peeling) end of a crest is the bigger wave: taller, breaks further out, longer throw
  // (and along a long beach the peak of each section is the bigger wave)
  b.H *= (1.0 + E.x * (abs(F.x) > 1e-4 ? lg / F.x : 0.0)) * (1.0 - BRK_FAR_H * fS);
  b.s = b.H / BRK_HREF;
  b.rs = sqrt(b.s);
  b.c = B.x;
  b.cFar = E.x;
  b.tauS = E.y;
  b.spill = E.z;
  b.kap = C.x;
  b.style = C.w;
  b.str = B.w;
  b.seed = sd;
  b.Tj = -BRK_TE * b.rs;
  b.tb = b.ti - b.Tj;
  b.Tr = 0.3 * b.rs;
  b.vy0 = -0.10 * b.rs;
  b.zI = A.z + E.w * BRK_TILT_L * sin(x / BRK_TILT_L) + BRK_FAR_Z * fS + C.y * (0.6 * sin(x * 1.37 + sd * 4.7) + 0.4 * sin(x * 3.1 + sd * 2.3));
  b.zB = b.zI + ${BK_TABLE.ZA[9]} * b.s;
  b.hold = BRK_HOLD * (1.0 - smoothstep(0.3, 0.5, b.style));
  return b;
}

// Normalised stage time. The approach is clocked by the arrival (ti - D): a section that
// breaks late arrives with the rest of the crest and STANDS (a steep wave held up by the
// backwash, §26) between the S2 and S3 shapes before it throws its lip at ti - 0.33 s.
// (The thrown lip itself runs on this clock: brkLip.)
float brkTnLip(Brk b, float t) {
  float Dn = b.D / b.rs;
  float a = (t - b.ti) / b.rs + Dn;
  float Ts = -0.47 - 0.6 * Dn;
  return a - Dn * smoothstep(Ts, Ts + max(1.6 * Dn, 1e-3), a);
}
// Spilling breakers (style < 0.5) do not collapse after the impact: the crest keeps standing
// ~0.3-0.5 s while its white roller rides it (clip, wave C 6.8-7.4 s: the roller sits on a green
// water face), then decays. The crest's stage clock slows to BRK_HOLD_R right after the impact and
// catches up smoothly (net delay (1 - BRK_HOLD_R) x hold). Heightfield, crest optics and particles
// follow this clock; the thrown lip does not (brkTnLip).
#define BRK_HOLD_R 0.15
// (branch-free: brkTn is inlined at many call sites, and an early return in each of them multiplies
// the D3D compile time)
float brkTn(Brk b, float t) {
  float tn = brkTnLip(b, t);
  float a = 0.6 * b.hold, w = max(0.8 * b.hold, 1e-4);
  float u = clamp((tn - a) / w, 0.0, 1.0);
  // integral of the rate mix(R, 1, smoothstep(a, a + w, tn)) from 0
  float I = w * u * u * u * (1.0 - 0.5 * u) + max(tn - a - w, 0.0);
  return mix(tn, BRK_HOLD_R * tn + (1.0 - BRK_HOLD_R) * I, step(0.0, tn) * step(1e-9, b.hold));
}
// apex (z, y) at normalised time tn; far from the break the track scales with sqrt(s) (speed ~ const)
vec2 brkApex(Brk b, BrkStage g, float tn) {
  float k = mix(b.s, b.rs, smoothstep(-0.8, -2.0, tn));
  return vec2(b.zI + g.za * k, g.ya * b.s);
}

// Continue each scheduled crest offshore with finite-depth phase speed. At tn=-3
// position, height and spatial derivatives match the measured shoaling profile.
void brkTransport(Brk b, float tn, out BrkStage g, out vec2 A) {
  g=brkStage(max(tn,-3.0)); A=brkApex(b,g,max(tn,-3.0));
#ifdef OPT_EXPLORE
  if(tn < -3.0) {
    vec4 anchor=swellTravel(A.x);
    vec4 offshore=swellPosition(anchor.x+(tn+3.0)*b.rs);
    float stretch=clamp(offshore.y/anchor.y,.75,2.4);
    float shoal=clamp(sqrt(anchor.z/max(offshore.z,.1)),.62,1.25);
    A.x=offshore.w; g.lf*=stretch; g.lb*=stretch;
    g.ya*=shoal; g.yt*=shoal;
  }
#endif
}

// Crest (apex) position along z and its speed.
float brkCrestZ(Brk b, float t) {
  float tn = brkTn(b, t);
  return brkApex(b, brkStage(tn), tn).x;
}
float brkCrestSpeed(Brk b, float t) { return (brkCrestZ(b, t + 0.01) - brkCrestZ(b, t - 0.01)) / 0.02; }

// Nominal trough-to-crest height over the life of the event (brkEtaC(brkHeight) = apex).
float brkHeight(Brk b, float t) {
  BrkStage g = brkStage(brkTn(b, t));
  return (g.ya - g.yt) * b.s * (BRK_HREF / 0.621);
}
float brkSteep(Brk b, float t) { return brkStage(brkTn(b, t)).st; }

// front-face shape functions: value and d/du
vec2 brkFcos(float u) { u = clamp(u, 0.0, 1.0); return vec2(0.5 + 0.5 * cos(3.14159265 * u), -1.5707963 * sin(3.14159265 * u)); }
vec2 brkFsteep(float u, float uw, float hw, float m, float p) {
  u = clamp(u, 0.0, 1.0);
  if (u < uw) {
    float r = u / uw;
    float rp = pow(r, p);
    float q = max(1.0 - rp, 1e-5);
    float v = hw + (1.0 - hw) * pow(q, 1.0 / p);
    float dv = -(1.0 - hw) * pow(r, p - 1.0) * pow(q, 1.0 / p - 1.0) / uw;
    return vec2(v, max(dv, -60.0));
  }
  float w = max((1.0 - u) / (1.0 - uw), 0.0);
  return vec2(hw * pow(w, m), -hw * m * pow(w, max(m - 1.0, 0.0)) / (1.0 - uw));
}
// Normalised single-valued profile about the apex: (eta, d eta / d xi) for xi = (z - Za) / s.
vec2 brkShape(BrkStage g, float xi) {
  if (xi < 0.0) {
    float a = xi / g.lb;
    float s2 = sech2(a);
    return vec2(g.ya * s2, g.ya * (-2.0 / g.lb) * tanh(a) * s2);
  }
  if (xi > g.lf) {
    const float La = 1.5;
    float e = (xi - g.lf) / La;
    float ex = exp(-e * e);
    return vec2(g.yt * ex, g.yt * ex * (-2.0 * e / La));
  }
  float u = xi / g.lf;
  vec2 fc = brkFcos(u), fs = brkFsteep(u, g.uw, g.hw, g.m, g.p);
  vec2 F = mix(fc, fs, g.st);
  return vec2(g.yt + (g.ya - g.yt) * F.x, (g.ya - g.yt) * F.y / g.lf);
}

// Normalised front-face position (xi / s) where the face has dropped to height fraction h
// (1 = apex, 0 = trough), by bisection of the monotonic shape function.
float brkFrontXi(BrkStage g, float h) {
  float lo = 0.0, hi = 1.0;
  int nIt = 12 + min(uEvtCount, 0);   // dynamic trip count: not unrolled by FXC
  for (int k = 0; k < nIt; k++) {
    float u = 0.5 * (lo + hi);
    float F = mix(brkFcos(u).x, brkFsteep(u, g.uw, g.hw, g.m, g.p).x, g.st);
    if (F > h) lo = u; else hi = u;
  }
  return 0.5 * (lo + hi) * g.lf;
}

// Heightfield part of one event: elevation above the local mean level at z.
// Returns (eta, d eta / dz, crest-proximity weight)
// brkProfileS: the same from the event's stage g, apex A at stage time tn = brkTn(b, t) (callers that
// need the profile at several z share one stage evaluation: fewer inlined copies of brkStage)
vec3 brkProfileS(Brk b, BrkStage g, vec2 A, float tn, float z) {
  if (b.str <= 0.0 || tn < ${BK_TABLE.T[0].toFixed(1)} || tn > BRK_TEND) return vec3(0.0);
  vec2 e = brkShape(g, (z - A.x) / b.s);
  // the swell is born offshore (the clip shows no swell beyond D ~ 17-18 m, z ~ -13: C's hump is a
  // faint ridge there), and only exists seaward of the beach face
  float birth=smoothstep(-18.0,-11.0,A.x);
#ifdef OPT_EXPLORE
  birth=1.0;
#endif
  float amp = birth * (1.0 - smoothstep(0.6, BRK_TEND, tn)) * b.str;
  float fade = 1.0 - smoothstep(-1.0, 0.1, z);
  float k = amp * fade;
  // crest proximity: a narrow strip on the crest top (on the steep front measured
  // vertically below the apex, so foam never smears down a vertical face)
  float xi = (z - A.x) / b.s;
  // (a spilling crest carries its whitecap further down the face)
  // (while the lip sheet is out, it carries the white rim itself: nothing on the face under it)
  float crestW = xi < 0.0 ? exp(-pow(xi / 0.12, 2.0))
               : exp(-pow((g.ya - e.x) / (0.025 + 0.1 * b.spill), 2.0)) * (1.0 - smoothstep(BRK_TB - 0.06, BRK_TB, tn) * (1.0 - smoothstep(0.0, 0.1, tn)));
  return vec3(e.x * b.s * k, e.y * k, crestW * k);
}
vec3 brkProfile(Brk b, float z, float t) {
  float tn = brkTn(b, t);
  vec3 result = vec3(0.0);
#ifdef OPT_EXPLORE
  if(tn < -3.0) {
    BrkStage g = brkStage(-3.0); vec2 A = vec2(0.0); brkTransport(b,tn,g,A);
    vec2 e=brkShape(g,(z-A.x)/b.s);
    result = vec3(e.x*b.s*b.str,e.y*b.str,0.0);
  } else
#endif
  {
  BrkStage g = brkStage(tn);
  result = brkProfileS(b, g, brkApex(b, g, tn), tn, z);
  }
  return result;
}

// Feathering / spilling whitecap coverage on the crest before the lip is thrown (drawn on a
// fine "crest skin" by the lip sheet: the coarse heightfield would smear it down the face).
// Feathering is patchy along the crest (patches 0.4-0.8 m, 1-2 m apart); a spilling
// breaker carries a continuous whitecap.
// The onset is on the ARRIVAL clock (the approach, without the stand at the break): a spilling
// crest whitecaps along its whole length at nearly the same time (wave C: 6.25-6.35 s from the
// nucleation point to the right edge, which plunges 0.7 s later), ~0.2 s before its lip is born.
float brkArrival(Brk b, float t) { return (t - b.ti + b.D) / b.rs; }
// (a small reformed crest - clip's 'a', s ~0.36 - is white along its whole length from its arrival
// on, ~0.45 earlier: it spills along v 0.60 at 0.1-0.7 s)
float brkWhitecapOnset(Brk b) { return mix(-0.85, -0.30, smoothstep(0.3, 0.9, b.spill)) - 0.45 * (1.0 - smoothstep(0.35, 0.6, b.s)); }
float brkWhitecap(Brk b, float t) {
  float patchy = smoothstep(0.15, 0.75, 0.5 + 0.5 * sin(b.x * 3.3 + b.seed * 17.0) * sin(b.x * 1.7 + b.seed * 5.0 + 1.0));
  float spill = smoothstep(0.3, 0.9, b.spill);
  float a = brkArrival(b, t) - brkWhitecapOnset(b);
  // feathering: patchy, builds over 0.4; spilling: continuous, grows in ~0.1 s
  return mix(smoothstep(0.0, 0.4, a) * 0.45 * patchy, smoothstep(0.0, 0.12, a), spill);
}
// Aerated crest on the heightfield. The whitecap is drawn by the lip sheet's crest skin and
// the whitened remnant by the lip roll + the injected bore foam, so only a faint trace of
// torn crest remains here (texture on the near-vertical heightfield face streaks vertically).
float brkCrestFoam(Brk b, float t, float crestW) {
  float tn = brkTn(b, t);
  // (kept faint: at 0.25 the lace texture thresholds into a row of bright dashes along the crest)
  float post = 0.08 * smoothstep(0.02, 0.12, tn) * (1.0 - smoothstep(0.15, 0.35, tn));
  return crestW * post;
}

// Sum of all event heightfields at (x,z). d = (d/dx, d/dz) of eta (d.x is filled by the caller).
float brkSurface(vec2 xz, float t, out vec2 d, out float crestFoam) {
  float eta = 0.0; d = vec2(0.0); crestFoam = 0.0;
  for (int i = 0; i < uEvtCount; i++) {  // dynamic bound: keeps D3D/FXC from unrolling
    Brk b = brkAt(i, xz.x);
    vec3 p = brkProfile(b, xz.y, t);
    eta += p.x;
    d.y += p.y;
    if (p.z > 0.01) crestFoam = max(crestFoam, brkCrestFoam(b, t, p.z));
  }
  return eta;
}
// Horizontal thickness (m) of the crest at height y through the point xz: the water between
// the front face and the back face at that level (the light path through a thin crest; the
// high sun behind the wave makes thin crests glow, colour report §4.1). -1 if xz is not on
// the upper part of a breaker crest. For the optics: thickHint of a heightfield fragment.
float brkCrestWidth(vec2 xz, float y, float t) {
  float best = -1.0;
  for (int i = 0; i < uEvtCount; i++) {  // dynamic bound: keeps D3D/FXC from unrolling
    Brk b = brkAt(i, xz.x);
    if (b.str <= 0.0) continue;
    float tn = brkTn(b, t);
    if (tn < -2.0 || tn > 0.3) continue;
    BrkStage g = brkStage(tn);
    vec2 A = brkApex(b, g, tn);
    float amp = smoothstep(-18.0, -11.0, A.x) * b.str;
    float ya = g.ya * b.s * amp, yt = g.yt * b.s * amp;
    float h = (y - yt) / max(ya - yt, 1e-3);          // height fraction on this crest
    if (h < 0.35 || h > 1.02) continue;
    float xi = (xz.y - A.x) / b.s;
    if (xi < -1.5 || xi > g.lf) continue;
    // back face: ya sech^2(xi / lb) = y  ->  xi_b = -lb acosh(sqrt(ya / y))
    float r = sqrt(max(ya, 1e-3) / max(y, 1e-3));
    float xb = y > 0.0 ? -g.lb * log(r + sqrt(max(r * r - 1.0, 0.0))) : -3.0;
    float xf = brkFrontXi(g, clamp(h, 0.0, 1.0));
    float w = max(xf - xb, 0.0) * b.s;
    best = best < 0.0 ? w : min(best, w);
  }
  return best;
}
float brkSurfaceOnly(vec2 xz, float t) {
  float eta = 0.0;
  for (int i = 0; i < uEvtCount; i++) {  // dynamic bound: keeps D3D/FXC from unrolling
    eta += brkProfile(brkAt(i, xz.x), xz.y, t).x;
  }
  return eta;
}

// ---------------------------------------------------------------- the lip (jet)
// The lip is a curved sheet: centreline = cubic Bezier from the root R (just ahead of
// the apex, horizontal tangent) to the tip T. The tip grows out of the steepening crest
// between BRK_TB and BRK_TE, then flies ballistically (launch 0.17 m ahead of the apex,
// u0 = 1.05, w0 = -0.10 m/s for s = 1): the apex meanwhile collapses in place.
struct BrkLip {
  vec2 A;      // apex (z, y)
  vec2 R;      // root (centreline)
  vec2 T;      // tip
  vec2 P1, P2; // Bezier handles
  vec2 vT;     // tip velocity
  float wR;    // thickness at the root
  float wT;    // thickness at the tip
  float fl;    // time of flight of the tip since release (s)
  float grow;  // 0 at birth -> 1
  float tn;
  BrkStage g;  // stage at tn
};
vec2 brkLipV0(Brk b) { return vec2(1.05 * (1.0 + b.kap) * mix(0.55, 1.0, b.style), -0.10) * b.rs; }
BrkLip brkLip(Brk b, float t) {
  BrkLip L;
  float tn = max(brkTnLip(b, t), BRK_TB);
  L.tn = tn;
  BrkStage g = brkStage(tn);
  L.g = g;
  L.A = brkApex(b, g, tn);
  L.grow = smoothstep(BRK_TB, BRK_TB + 0.06, tn);
  L.wR = 0.075 * b.s;
  // the root of the jet (where the barrel ceiling turns down into the wall) moves out
  // from inside the crest as the lip forms
  L.R = L.A + vec2(mix(0.02, 0.10, smoothstep(BRK_TB, BRK_TE + 0.05, tn)) * b.s, -0.5 * L.wR);
  vec2 rel = vec2(0.17, -0.043) * b.s;
  if (tn < BRK_TE) {
    float f = smoothstep(BRK_TB, BRK_TE, tn);
    L.T = L.A + mix(vec2(0.03, -0.06) * b.s, rel, f);
    L.vT = vec2(1.5, -0.3) * b.rs;
    L.fl = 0.0;
  } else {
    BrkStage gE = brkStage(BRK_TE);
    vec2 Te = brkApex(b, gE, BRK_TE) + rel;
    vec2 V0 = brkLipV0(b);
    L.fl = (tn - BRK_TE) * b.rs;
    float fb = min(L.fl, b.Tj);
    L.T = Te + V0 * fb + vec2(0.0, -0.5 * G_ACC * fb * fb);
    L.vT = V0 + vec2(0.0, -G_ACC * fb);
    if (L.fl > b.Tj) {
      // after touch-down the curtain is absorbed: its foot slides on and sinks below the trough
      float a = L.fl - b.Tj;
      L.T += vec2(V0.x * a * 0.6, -0.25 * b.s * smoothstep(0.0, 0.25 * b.rs, a));
    }
  }
  vec2 ch = L.T - L.R;
  float len = max(length(ch), 1e-4);
  vec2 dT = normalize(ch / len + normalize(L.vT));
  L.P1 = L.R + vec2(0.45 * max(ch.x, 0.02 * b.s) + 0.1 * len, 0.0);
  L.P2 = L.T - dT * 0.35 * len;
  L.wT = 0.035 * b.s * (1.0 - 0.5 * min(L.fl / 0.3, 1.0)) * L.grow;
  return L;
}
// centreline point and unit tangent at sigma (0 = root .. 1 = tip)
vec4 brkLipCurve(BrkLip L, float sg) {
  float u = 1.0 - sg;
  vec2 p = u * u * u * L.R + 3.0 * u * u * sg * L.P1 + 3.0 * u * sg * sg * L.P2 + sg * sg * sg * L.T;
  vec2 dp = 3.0 * u * u * (L.P1 - L.R) + 6.0 * u * sg * (L.P2 - L.P1) + 3.0 * sg * sg * (L.T - L.P2);
  float l = length(dp);
  return vec4(p, l > 1e-6 ? dp / l : vec2(1.0, 0.0));
}
float brkLipWidth(BrkLip L, float sg) { return L.wT + (L.wR - L.wT) * pow(1.0 - clamp(sg, 0.0, 1.0), 0.8); }

// Centreline position of the lip at fraction sigma (0 = tip, 1 = root): returns (z, y, thickness).
vec3 brkJet(Brk b, float sigma, float t) {
  BrkLip L = brkLip(b, t);
  float sg = 1.0 - clamp(sigma, 0.0, 1.0);
  vec4 c = brkLipCurve(L, sg);
  return vec3(c.xy, brkLipWidth(L, sg));
}
// Impact speed of the lip tip (for splash / foam scaling, §9 / §53)
float brkImpactSpeed(Brk b) {
  vec2 v = brkLipV0(b) + vec2(0.0, -G_ACC * b.Tj);
  return length(v);
}

// ---------------------------------------------------------------- SWE coupling
// (owned by the swash module: keep the brkInjection() signature stable)
// Mass (m/s) and momentum (m^2/s^2) sources injected into the shallow-water solver
// as each event collapses: the plunging lip and the collapsing crest become the bore.
uniform float uInjMass;     // total injected volume per unit crest width per unit H^2 (m^2 / m^2)
uniform float uInjSpeed;    // scales the injected water speeds
uniform vec4 uInjJ;         // jet + splash sheet: x mass fraction, y pulse time scale (s), z zone drift (m/s), w water speed (m/s)
uniform vec4 uInjR;         // collapsing roller:  x delay after impact (s), y pulse time scale (s), z zone drift (m/s), w water speed (m/s)
uniform vec4 uInjL;         // along-crest momentum modulation (relative) at 1.2 m / 0.42 m / fine (quasi-random, 0.15-0.3 m); w: along-shore drift (x / z momentum)
// Two components (breaker analysis §5.4-5.5, timeline §2, §9): the plunging jet and its splash-up
// throw a fast, thin sheet shoreward right at impact (the first pulse, which stalls against the
// previous backwash); the crest remnant collapses more slowly into the turbulent roller that becomes
// the main bore, held at the step until the backwash has run out and then surging up the face (the
// stall-then-surge of every bore in the clip, timeline finding 4). Each is a gamma pulse in time
// (unit integral) times a Gaussian in z whose centre drifts shoreward from the landing line.
float brkGamma(float tau, float T) { return tau > 0.0 ? tau / (T * T) * exp(-tau / T) : 0.0; }
// only a plunging jet throws its splash sheet far shoreward; a spilling crest's whitewater just
// rides the (slower) bore
float brkThrow(Brk b) { return mix(0.3, 1.0, smoothstep(0.4, 0.75, b.style)); }
// Swash-front lobes (swash §3.2): convex tongues 0.25-0.6 m apart on a 1.0-1.3 m undulation, new
// for every wave. A slight excess of bore momentum on the crests of this pattern grows the fingers
// (and they smooth out as the front slows); the phases come from the event seed.
// Beyond the clip's frame (|x| > 4-8 m, explore mode) the pattern's amplitude and phase wander along
// the beach (a pure sinusoid seen down 20 m of shoreline reads as a comb).
// (the finest component is quasi-random: two incommensurate, phase-wandering waves, 0.15-0.3 m, so
// the front breaks into irregular fingers, never a regular comb)
float brkLobes(Brk b) {
  float x = b.x;
  float far = smoothstep(4.0, 8.0, abs(x));
  float ph = far * 2.5 * sin(0.37 * x + b.seed * 5.1 + 1.7 * sin(0.13 * x));
  float am = mix(1.0, 0.45 + 0.9 * (0.5 + 0.5 * sin(0.29 * x + b.seed * 3.3) * sin(0.71 * x + 1.1)), far);
  // (the medium lobes wander in phase along the crest too: spacing 0.3-0.6 m)
  float wm = 1.1 * sin(1.9 * x + b.seed * 13.7) + 0.6 * sin(4.4 * x + b.seed * 29.3);
  float fine = 0.6 * sin(24.1 * x + b.seed * 53.3 + 2.1 * sin(3.7 * x + b.seed * 9.0)) + 0.4 * sin(37.9 * x + b.seed * 71.9 + 1.7 * sin(5.9 * x + b.seed * 4.0));
  return am * (uInjL.x * sin(5.236 * x + b.seed * 37.1 + ph) + uInjL.y * sin(14.96 * x + b.seed * 91.7 + wm + 1.9 * ph)
       + uInjL.z * fine);
}
// Along a long beach the run-up of a wave varies by +-20-30 % over 10-40 m (swash §3.2, explore):
// a per-event undulation (seeded, ~13 / ~30 m) plus the beach cusps (Beach.js: the swash runs further
// up in the embayments between the horns, 17 m spacing). Relative run-up change; faded in beyond the
// clip's view (|x| > 4-8 m), so the video events keep their measured swash. Shared by the bore
// injection (as a speed change, R ~ U^2) and the far-field swash (SwashSim FAR_PARAM).
float brkRunupMod(float x, float seed) {
  float far = smoothstep(4.0, 8.0, abs(x));
  float und = 0.6 * sin(0.21 * x + seed * 7.3 + 0.5 * sin(0.09 * x + seed * 3.1)) + 0.4 * sin(0.47 * x + seed * 13.1);
  float cph = x / 17.0 + 0.25 * sin(x / 23.0 + 1.1);
  float horn = 1.0 - abs(sin(3.14159 * cph));
  return far * (0.14 * und + 0.13 * (1.0 - 2.0 * horn * horn));
}
vec3 brkInjection(vec2 xz, float t) {
  vec3 src = vec3(0.0);
  for (int i = 0; i < uEvtCount; i++) {  // dynamic bound: keeps D3D/FXC from unrolling
    Brk b = brkAt(i, xz.x);
    if (b.str <= 0.0) continue;
    float tau = t - b.ti;
    if (tau < -0.1 || tau > 3.0) continue;
    float hs = sqrt(b.H / 0.45);
    // (x the event's bore gain, uEvtG.x: the breaker's per-wave strength of the collapsed bore)
    float M = uInjMass * 0.2025 * pow(b.H / 0.45, 1.5) * b.str * max(uEvtG[i].x, 0.0);   // ∝ H L with L ∝ sqrt(H); = uInjMass H^2 at H = 0.45
    // jet + splash sheet
    float tJ = tau + 0.04;
    float zJ = b.zI + 0.10 + uInjJ.z * brkThrow(b) * tJ;
    float sJ = 0.22 * hs + 0.20 * tJ;
    float qJ = M * uInjJ.x * brkGamma(tJ, uInjJ.y) * exp(-0.5 * pow((xz.y - zJ) / sJ, 2.0)) / (2.5066 * sJ);
    // collapsing crest -> roller -> bore
    float tR = tau - uInjR.x;
    float zR = mix(brkCrestZ(b, b.ti), b.zI, 0.5) + uInjR.z * max(tR, 0.0);
    float sR = 0.30 * hs + 0.15 * max(tR, 0.0);
    float qR = M * (1.0 - uInjJ.x) * brkGamma(tR, uInjR.y) * exp(-0.5 * pow((xz.y - zR) / sR, 2.0)) / (2.5066 * sR);
    // a spilling bore enters the swash slower than a plunging one
    float thr = brkThrow(b);
    float mz = uInjSpeed * (qJ * uInjJ.w * mix(0.5, 1.0, thr) + qR * uInjR.w * mix(0.6, 1.0, thr))
             * (1.0 + brkLobes(b)) * (1.0 + 0.5 * brkRunupMod(b.x, b.seed));
    src += vec3(qJ + qR, mz, mz * uInjL.w);
  }
  return src;
}
// Surface whitewater production (1/s, in units of the SWE blanket-foam amount): the plunge and
// its splash-up throw an aerated sheet shoreward that lands progressively further up (the white
// front runs from the landing line to the waterline in ~0.8 s, timeline §9 bore.X), and the
// collapsing roller churns in place. Independent of how the mass is split between components.
uniform vec4 uFoamSrc;      // x: splash-sheet foam, y: its pulse time scale (s), z: roller foam, w: splash zone width scale
uniform float uFoamSpill;   // whitewater of a spilling roller (1/s per unit spill at H = 0.45)
uniform vec2 uFoamSpillK;   // its drift speed (m/s), end of production (s after the impact)
float brkFoam(vec2 xz, float t) {
  float f = 0.0;
  for (int i = 0; i < uEvtCount; i++) {  // dynamic bound: keeps D3D/FXC from unrolling
    Brk b = brkAt(i, xz.x);
    if (b.str <= 0.0) continue;
    float tau = t - b.ti;
    // A spilling crest carries its whitewater roller down the front face for the whole break and
    // its bore keeps churning shoreward afterwards: a small spiller whitens the inner surf for far
    // longer than its size suggests (wave "a" keeps A's step zone white until ~1.8 s, timeline §2)
    // (a pure spiller: a crest that spills and then plunges throws its whitewater like a plunger)
    float spl = smoothstep(0.5, 0.9, b.spill) * (1.0 - smoothstep(0.1, 0.3, b.style)) * b.str;
    // (before the impact the whitecap is drawn on the crest by the lip sheet)
    if (spl > 0.0 && tau > -0.3 && tau < 2.4) {
      // (the spilled whitewater rides the reformed bore in over the step and up the lower face:
      // wave "a" whitens v 0.72-0.84 until ~2.4 s, timeline §2 / swash §4)
      float on = smoothstep(-0.3, 0.0, tau) * (1.0 - smoothstep(uFoamSpillK.y - 1.1, uFoamSpillK.y, tau));
      float zc = b.zI + 0.1 + uFoamSpillK.x * max(tau, 0.0);
      f += uFoamSpill * spl * (b.H / 0.45) * on * exp(-0.5 * pow((xz.y - zc) / (0.3 + 0.2 * max(tau, 0.0)), 2.0));
    }
    if (tau < -0.1 || tau > 3.0) continue;
    float E = (b.H * b.H) / (0.45 * 0.45) * b.str * mix(0.6, 1.0, b.style);
    vec2 gain = max(uEvtG[i].xy, vec2(0.0));   // per-event (bore, splash) gains
    float tJ = tau + 0.04;
    float zJ = b.zI + 0.10 + uInjJ.z * brkThrow(b) * tJ;
    float sJ = uFoamSrc.w * (0.25 + 0.30 * max(tJ, 0.0));
    f += E * gain.y * uFoamSrc.x * brkGamma(tJ, uFoamSrc.y) * exp(-0.5 * pow((xz.y - zJ) / sJ, 2.0));
    // (the crash zone churns in place from the impact on, whenever the roller's water surges on)
    float tR = tau - 0.21;
    float zR = mix(brkCrestZ(b, b.ti), b.zI, 0.5) + 0.55 * max(tR, 0.0);
    float sR = 0.35 + 0.2 * max(tR, 0.0);
    f += E * gain.x * uFoamSrc.z * brkGamma(tR, 0.4) * exp(-0.5 * pow((xz.y - zR) / sR, 2.0));
  }
  return f;
}
// How much of the shallow-water surface foam is shown at xz under the analytic breaker (1 = all).
// The shoaling face is stretched and renewed as it stands up (strain thins the foam, §32): in the
// clip the approaching face is clean glassy green even over the previous wave's lace, and the SWE
// foam sampled at xz on a near-vertical face would smear into vertical streaks. Clocked by the
// arrival (brkTn: a standing section is covered), from the first steepening until the collapsed
// crest is back down; measured on the rendered elevation, so the trough and the toe in front of
// the face keep the previous bore's torn whitewater. Baked per SWE cell (SwashSim FOAM_VIEW).
float brkSweFoamKeep(vec2 xz, float t) {
  float keep = 1.0;
  if (xz.y < -9.0 || xz.y > 0.6) return keep;
  float fade = 1.0 - smoothstep(-1.0, 0.1, xz.y);   // as brkProfile: the heightfield dies on the beach
  for (int i = 0; i < uEvtCount; i++) {  // dynamic bound: keeps D3D/FXC from unrolling
    Brk b = brkAt(i, xz.x);
    if (b.str <= 0.0) continue;
    float tn = brkTn(b, t);
    if (tn < -2.4 || tn > 0.5) continue;
    BrkStage g = brkStage(tn);
    vec2 A = brkApex(b, g, tn);
    float xi = (xz.y - A.x) / b.s;
    if (xi < -0.8 || xi > g.lf) continue;
    float amp = smoothstep(-23.0, -13.0, A.x) * (1.0 - smoothstep(0.6, BRK_TEND, tn)) * b.str * fade;
    float eta = brkShape(g, xi).x * b.s * amp;                         // rendered elevation (m)
    float raised = smoothstep(0.012, 0.06, eta - 0.5 * g.yt * b.s * amp);
    float front = smoothstep(-0.8, -0.2, xi);                          // front face + top of the back
    float grown = smoothstep(-2.4, -1.5, tn);
    float live = 1.0 - smoothstep(0.1, 0.5, tn);
    keep = min(keep, 1.0 - front * raised * grown * live);
  }
  return keep;
}
`;
