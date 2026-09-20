// Mulberry32: bryc JavaScript implementation of Tommy Ettinger's generator, public domain.
// See licenses/Mulberry32-Public-Domain.txt.
// Breaker event schedule.
//
// The first events reproduce the reference clip (t = 0 is its first frame), from the
// timeline / morphology analyses: Z (≈ -4.6 s), A (impact ≈ -0.8 s at x = 0, already a
// collapsing bore at t = 0), B (the showcase plunge: logistic peel left -> right,
// t_i = 2.89 s on the left edge, 3.31 s on the right), C (spilling-then-plunging, nucleates at
// u = 0.45 at 6.70 s and spreads both ways) and D (≈ 10.5 s). After that the schedule
// continues procedurally with the same statistics (period 3.6-4.1 s), deterministic
// for a given seed.
//
// Along-crest timing (math doc §42/§43), mirrored exactly by brkAt() in glsl/breaker.js:
//   ti(x) = t0 + peel_step(x) + wobble * (0.62 sin(2.3 x + 6.1 seed) + 0.38 sin(5.1 x + 11.7 seed))
//   peel_step(x) = mix(local(clampX(xs)), Af (farS(x) + farS'(x) (xs - x)), farW(x))  (the break)
//   peel(x)      = mix(local(clampX(x)),  Af farS(x), farW(x))                        (the arrival)
//   local(x) = quadratic about x0 in the softened distance ds = sqrt(d^2 + 0.25) - 0.5, d = x - x0
//             (aR1 ds + aR2 ds^2 right, aL1 ds + aL2 ds^2 left; a negative quadratic saturates)
//             + F[0] / (1 + exp(-(x - F[1]) / F[2]))            (logistic peel)
//   clampX: identity for |x| <= 5 m (everything the clip camera sees), then saturates smoothly
//           toward ±8 m: the video events' local peels stay bounded along a long beach (explore)
//   farS(x) = 0.55 sin(0.185 x + 2π ψ) + 0.30 sin(0.33 x + 17.1 ψ + 1.3) + 0.15 sin(0.57 x + 41.7 ψ + 2.9)
//           a bounded multi-nucleation modulation (|farS| <= 1): its minima are the peaks where a
//           section breaks first (median spacing 16 m, 10-23 m) and it peels both ways from there
//           (median 8-10 m/s); Af = |far| s (e.far); ψ = e.farPhase
//   farW(x) = 1 (procedural events: far > 0, the local peel is not used) or smoothstep(5, 13, |x|)
//           (video events: far < 0, so nothing changes inside the clip's view)
//   xs = stairX(x, seed): a mild along-crest staircase (sections of 0.45 m close out together)
//   stall D = max(stand, 0.05) + mix(F[3] local(x), FAR_STAND Af (farS + 1), farW) + (peel_step - peel):
//           the arrival (ti - D) is smooth; a section that breaks late on the far modulation also
//           stands up a while (feathering shoulder) before it throws
//   H(x) *= 1 - FAR_H farW farS (the peak is the bigger wave), landing line zI += FAR_Z farW farS;
//   the logistic height gain (hPeel) fades out with farW
// Our x scale: the frame spans x = -2.35 .. +2.30 at the break line (u = (x + 2.35) / 4.65).

import { CONFIG } from '../config.js';

const MAX_EVENTS = 6;
const H_REF = 0.45;   // nominal height of wave B (see glsl/breaker.js BRK_HREF)
const T_SKIN = -1.4, T_RELEASE = -0.33, T_END = 0.85; // normalised stage times (breaker.js)
// along-shore extent over which every event's timing is bounded and tracked (explore mode)
const X_BEACH = 200;
const XS_STEP = 0.5;
// half-width (m) of the lip ribbons around the focus (water/LipRibbon.js, explore mode)
export const LIP_HALF = 40.32;
// bounded along-shore structure (breaker.js BRK_*)
const XCLAMP = 5.0, XSAT = 3.0;
const FAR_STAND = 0.25, FAR_H = 0.12, FAR_Z = 0.25;
const TILT_L = 12.0;

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// wave B's impact timing: the logistic of breaker_morphology §3.1 (converted to our x scale:
// 2.940 + 0.546 / (1 + exp(-(x - 1.055) / 0.93))) refitted to the frame where the falling
// curtain crosses the row v = 0.62 in each 0.06-u column of the reference (rms 0.02 s):
// the far left touches down first (2.89 s), the right peels at 6-8 m/s to 3.31 s. Checked
// against the white lip-line traces of the reference timestacks (u 0.2-0.95): the knee
// (ballistic tip, brkLip) follows them to ~0.02 s. (The timeline's tb(u) table, a whitening
// onset, leads the lip trace by 0.05-0.08 s: spray and the lip line whiten the lip zone first.)
const PEEL_B = { x0: 0, aR1: 0, aR2: 0, aL1: 0, aL2: 0, F: [0.65, 0.75, 1.35, 0.3] };

export const EVENT_DEFAULTS = {
  H: 0.45,        // nominal breaker height, trough to crest (m); rendered x1.38 (camera-matched)
  zI: -1.90,      // lip landing z (image row v ≈ 0.645 at the trough level)
  c: 2.5,         // bore / crest speed at the break (swash coupling, particles)
  hPeel: 0.0,     // relative height gain along the logistic peel (the late end is the bigger wave)
  stand: 0.0,     // s the crest stands at the break before throwing (all x), plus F[3] x peel delay
  kappa: 0.0,     // lip ground speed = 1.05 (1 + kappa) m/s (s = 1)
  wobble: 0.012,  // s, residual along-crest timing noise
  waviness: 0.03, // m, landing-line waviness
  tilt: 0.0,      // dz/dx of the landing line near x = 0 (bounded: TILT_L sin(x / TILT_L))
  hvar: 0.08,
  style: 1.0,     // 0 spilling/collapsing .. 1 plunging
  spill: 0.2,
  strength: 1.0,
  // per-event multipliers read by the other modules (uEvtG.xy): bore strength (swash) and
  // splash strength (whitewater emitters; also returned by brkTiming() as .splash)
  bore: 1.0,
  splash: 1.0,
  // bounded along-shore timing modulation (uEvtG.zw): amplitude Af (s) and phase; > 0 applies
  // everywhere and replaces the local peel (x0, aR/aL, F[0..3] are ignored), < 0 only beyond
  // |x| > 5 m (the video events keep their measured timing in view), 0 none
  far: -1.0,
  farPhase: 0.0,
  ...PEEL_B,
};

export const VIDEO_EVENTS = [
  { name: 'Z', t0: -4.84, H: 0.42, seed: 0.11, farPhase: 0.62 },
  // (A's bore runs a little short of B's: its early bore stops at v ~0.85 at 0-0.6 s)
  { name: 'A', t0: -1.04, H: 0.50, seed: 0.37, bore: 0.85, farPhase: 0.18 },
  // a: the small reformed crest that spills along v ≈ 0.60 at 0.1-0.7 s behind A's bore
  { name: 'a', t0: 0.85, H: 0.16, seed: 0.51, F: [0.1, 0.0, 1.2, 0.0], stand: 0.35, style: 0.0, spill: 1.0, zI: -2.2, strength: 0.8, farPhase: 0.47 },
  { name: 'B', t0: 2.840, H: 0.45, seed: 0.73, hPeel: 0.18, farPhase: 0.83 },
  // C: nucleates at u = 0.45 (x = -0.23) at 6.72 s; left section follows within 0.05-0.15 s,
  // right lags up to 0.72 s (timeline §3 tb table, fitted with the softened peel + staircase,
  // rms 0.06 s: the left half has no clear order); smaller, spills first. Its whitecap sits at image row
  // v 0.585-0.60 at 6.3 s and it plunges onto v ~0.66 (raw rows): landing line 0.15 m shoreward of B's
  // Its late right section is the smaller wave (crest ~0.02 v lower than the left at 6.9 s, still a
  // spilling whitecap there, only small barrels at 7.1-7.5 s): hPeel -0.2 along a logistic of
  // negligible delay (F[0] = 1 ms) is a pure height ramp, -20 % beyond u ~0.7.
  { name: 'C', t0: 6.722, H: 0.30, seed: 0.29, x0: -0.23, aR1: 0.40, aR2: 0.004, aL1: 0.294, aL2: -0.155, F: [0.001, 0.6, 0.3, 0.8], stand: 0.3,
    hPeel: -0.2, style: 0.35, spill: 0.9, zI: -2.05, kappa: -0.1, farPhase: 0.36 },
  { name: 'D', t0: 10.37, H: 0.47, seed: 0.91, farPhase: 0.05 },
];

// along-crest staircase (brkStairX in breaker.js): sections of STAIR_L m close out together
const STAIR_L = 0.45, STAIR_K = 0.25;
const fract = (v) => v - Math.floor(v);
const mix = (a, b, k) => a * (1 - k) + b * k;
const sstep = (a, b, v) => { const t = Math.min(Math.max((v - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
export function stairX(x, seed) {
  const s = x / STAIR_L + fract(seed * 3.0 + 0.485);
  return x + STAIR_K * STAIR_L * (Math.floor(s) + sstep(0.45, 1.0, fract(s)) - s + 0.175);
}
// brkClampX: identity within the clip's view, saturating toward ±(XCLAMP + XSAT)
export function clampX(x) {
  const e = Math.max(Math.abs(x) - XCLAMP, 0);
  return x - Math.sign(x) * (e - (XSAT * e) / (e + XSAT));
}
// brkFarSD: the modulation and its slope
export function farSD(x, ph) {
  const a1 = 0.185 * x + 6.2831853 * ph, a2 = 0.33 * x + 17.1 * ph + 1.3, a3 = 0.57 * x + 41.7 * ph + 2.9;
  return [0.55 * Math.sin(a1) + 0.30 * Math.sin(a2) + 0.15 * Math.sin(a3),
    0.10175 * Math.cos(a1) + 0.099 * Math.cos(a2) + 0.0855 * Math.cos(a3)];
}
export const farS = (x, ph) => farSD(x, ph)[0];
const farW = (far, x) => (far > 0 ? 1 : sstep(5.0, 13.0, Math.abs(x)));
// local peel delay (brkPeel in breaker.js): softened quadratic about x0 + logistic
function peelAt(e, F, x) {
  const d = x - e.x0;
  let ds = Math.sqrt(d * d + 0.25) - 0.5;
  const a1 = d >= 0 ? e.aR1 : e.aL1, a2 = d >= 0 ? e.aR2 : e.aL2;
  if (a2 < 0) ds = Math.min(ds, -a1 / (2 * a2));
  return a1 * ds + a2 * ds * ds + F[0] / (1 + Math.exp(-(x - F[1]) / Math.max(F[2], 1e-3)));
}

/** JS mirror of brkAt() timing in glsl/breaker.js (for CPU-side emission bookkeeping).
 *  Also returns the event's splash multiplier (e.splash, default 1) for the whitewater emitters. */
export function brkTiming(e, x) {
  const s = e.seed;
  const F = e.F || [0, 0, 1, 0];
  const xs = stairX(x, s);
  const far = e.far || 0, Af = Math.abs(far), ph = e.farPhase || 0;
  // measured local peel (video events; the modulation replaces it everywhere when far > 0)
  const useL = far > 0 ? 0 : 1;
  const xc = clampX(x);
  let peelS = useL * peelAt(e, F, xc);
  let peel = useL * peelAt(e, F, clampX(xs));
  let lg = (useL * F[0]) / (1 + Math.exp(-(xc - F[1]) / Math.max(F[2], 1e-3)));
  let stand = F[3] * peelS;
  // along a long beach (beyond |x| > 5 m for the video events) the local peel hands over to the
  // bounded modulation
  const wS = far === 0 ? 0 : farW(far, x);
  const [fm, dfm] = farSD(x, ph);
  peelS = mix(peelS, Af * fm, wS);
  peel = mix(peel, Af * (fm + dfm * (xs - x)), wS);
  stand = mix(stand, FAR_STAND * Af * (fm + 1), wS);
  lg *= 1 - wS;
  const fS = fm * wS;
  const wob = 0.62 * Math.sin(x * 2.3 + s * 6.1) + 0.38 * Math.sin(x * 5.1 + s * 11.7);
  const ti = e.t0 + peel + e.wobble * wob;
  // stand: the section arrives early and stands (brkTn in breaker.js)
  const D = Math.max(Math.max(e.stand || 0, 0.05) + stand + (peel - peelS), 0);
  const H = e.H * (1 + e.hvar * (0.65 * Math.sin(x * 1.1 + s * 3.3) + 0.35 * Math.sin(x * 2.9 + s * 8.9)))
    * (1 + (e.hPeel || 0) * (Math.abs(F[0]) > 1e-4 ? lg / F[0] : 0)) * (1 - FAR_H * fS);
  const rs = Math.sqrt(H / H_REF);
  const Tj = -T_RELEASE * rs;
  const vx = 1.05 * (1 + e.kappa) * (0.55 + 0.45 * e.style) * rs;
  const vy = -0.10 * rs - CONFIG.g * Tj;
  const vimp = Math.hypot(vx, vy);
  const zI = e.zI + e.tilt * TILT_L * Math.sin(x / TILT_L) + FAR_Z * fS
    + e.waviness * (0.6 * Math.sin(x * 1.37 + s * 4.7) + 0.4 * Math.sin(x * 3.1 + s * 2.3));
  return { ti, tb: ti - Tj, H, Tj, vimp, rs, D, zI, splash: e.splash ?? 1, bore: e.bore ?? 1 };
}

// time range of the event's impact line across x in [xa, xb] (lo2: earliest arrival ti - D)
function tiRange(e, xa, xb, step = 0.25) {
  let lo = Infinity, hi = -Infinity, rsMax = 0, lo2 = Infinity;
  for (let x = xa; x <= xb + 1e-6; x += step) {
    const b = brkTiming(e, x);
    lo = Math.min(lo, b.ti); hi = Math.max(hi, b.ti); rsMax = Math.max(rsMax, b.rs);
    lo2 = Math.min(lo2, b.ti - b.D);
  }
  return [lo, hi, rsMax, lo2];
}

export class Schedule {
  constructor({ seed = 7, events = VIDEO_EVENTS } = {}) {
    this.events = events.map((e) => Schedule.prepare({ ...EVENT_DEFAULTS, ...e }));
    this.events.sort((a, b) => a.t0 - b.t0);
    this.rng = mulberry32(seed);
    this.nextSet = this.events.at(-1).t0 + 19 + this.rng() * 7;
    this.setStage = -1; this.quietWaves = 0;
    this.A = new Float32Array(MAX_EVENTS * 4);
    this.B = new Float32Array(MAX_EVENTS * 4);
    this.C = new Float32Array(MAX_EVENTS * 4);
    this.D = new Float32Array(MAX_EVENTS * 4);
    this.E = new Float32Array(MAX_EVENTS * 4);
    this.F = new Float32Array(MAX_EVENTS * 4); // logistic peel (amplitude, centre, width, -)
    this.G = new Float32Array(MAX_EVENTS * 4); // (bore gain, splash gain, far amplitude, far phase)
    this.count = 0;
    this.active = [];
    this.focusX = 0;   // along-shore focus (explore: the player; set by the caller of jetSlots)
  }

  static prepare(e) {
    // influence window (swell born ~6 normalised s before impact, injection / splash last
    // ~1.3 s after) over the shallow-water domain (clip) or the whole beach (explore: the
    // heightfield, the scrolling shallow-water window and the far-field swash can be anywhere)
    const [xa, xb] = CONFIG.explore ? [-X_BEACH, X_BEACH] : [CONFIG.swe.xMin, CONFIG.swe.xMax];
    const [, hi, rs, lo2] = tiRange(e, xa, xb, CONFIG.explore ? XS_STEP : 0.25);
    e._window = [lo2 - 6.2 * rs, hi + 1.4];
    // the lip sheet (with its crest skin for the whitecap) over the visible crest
    const [, vhi, vrs, vlo2] = tiRange(e, -4.0, 4.0);
    e._lip = [vlo2 + T_SKIN * vrs - 0.05, vhi + T_END * vrs + 0.05];
    if (CONFIG.explore) {
      // explore: per 0.5 m along the beach, the lip window there (jetSlots picks the range
      // covered by the ribbons around the focus)
      const n = Math.round((2 * X_BEACH) / XS_STEP) + 1;
      e._lipLo = new Float32Array(n); e._lipHi = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const b = brkTiming(e, -X_BEACH + i * XS_STEP);
        e._lipLo[i] = b.ti - b.D + T_SKIN * b.rs - 0.05;
        e._lipHi[i] = b.ti + T_END * b.rs + 0.05;
      }
    }
    return e;
  }

  /** Make sure procedural events exist up to time t. */
  extendTo(t) {
    let last = this.events[this.events.length - 1];
    while (last.t0 < t + 20) {
      const r = this.rng;
      const gap = CONFIG.explore ? 4.05 + r() * .75 : (r() < 0.1 ? 4.6 + r() * 1.2 : 3.55 + r() * .55);
      const t0 = last.t0 + gap;
      if(CONFIG.explore && this.setStage < 0 && t0 >= this.nextSet) this.setStage=0;
      const stage=this.setStage, large=stage===2;
      const setHeights=[.50,.64,.84,.56];
      const H=stage>=0 ? setHeights[stage]+(r()-.5)*.08 : (this.quietWaves>0?.27:.34)+r()*.14;
      if(stage>=0) {
        this.setStage++;
        if(this.setStage===4){this.setStage=-1;this.quietWaves=3;this.nextSet=t0+29+r()*26;}
      } else if(this.quietWaves>0)this.quietWaves--;
      // Bounded multi-nucleation timing along the whole beach: no local peel, the far modulation
      // everywhere (peaks every 10-25 m peeling both ways at ~5-10 m/s, the timing bounded by
      // ±far s), so a long beach shows several sections breaking and peeling at once.
      const ev = Schedule.prepare({
        ...EVENT_DEFAULTS,
        t0, H: CONFIG.explore ? H : .32+r()*.2,
        zI: -1.8-r()*.3-Math.max(H-.45,0)*2.4,
        large, setStage: stage,
        bore: large?1.12:1, splash: large?1.25:1,
        x0: 0, aR1: 0, aR2: 0, aL1: 0, aL2: 0,
        F: [0, 0, 1, 0],
        far: 1.0 + r() * 0.5,
        farPhase: r(),
        wobble: 0.02 + r() * 0.03,
        waviness: 0.02 + r() * 0.04,
        tilt: (r() - 0.5) * 0.06,
        kappa: -0.15 + r() * 0.2,
        style: large ? .98 : .5+r()*.5,
        spill: large ? .25 : r()*.8,
        seed: r(),
      });
      this.events.push(ev);
      last = ev;
    }
  }

  static window(e) { return e._window || Schedule.prepare(e)._window; }

  /** Spread (s) of the impact time across the shallow-water domain (kept for callers of the old API). */
  static spread(e) {
    const [lo, hi] = tiRange(e, CONFIG.swe.xMin, CONFIG.swe.xMax);
    return hi - lo;
  }

  pack(t) {
    this.extendTo(t);
    const act = [];
    for (const e of this.events) {
      const [a, b] = Schedule.window(e);
      if (t >= a && t <= b) act.push(e);
    }
    act.sort((a, b) => a.t0 - b.t0);
    while (act.length > MAX_EVENTS) act.shift();
    this.active = act;
    this.count = act.length;
    for (let i = 0; i < MAX_EVENTS; i++) {
      const e = act[i];
      const o = i * 4;
      if (!e) {
        this.A.set([0, 0, 0, 0], o); this.B.set([1, 0, 0, 0], o); this.C.set([0, 0, 0, 0], o);
        this.D.set([0, 0, 0, 0], o); this.E.set([1, 1, 0, 0], o); this.F.set([0, 0, 1, 0], o); this.G.set([1, 1, 0, 0], o);
        continue;
      }
      this.A.set([e.t0, e.H, e.zI, e.seed], o);
      this.B.set([e.c, e.x0, e.wobble, e.strength], o);
      this.C.set([e.kappa, e.waviness, e.hvar, e.style], o);
      this.D.set([e.aR1, e.aR2, e.aL1, e.aL2], o);
      this.E.set([e.hPeel || 0, e.stand || 0, e.spill, e.tilt], o);
      this.F.set(e.F || [0, 0, 1, 0], o);
      this.G.set([e.bore ?? 1, e.splash ?? 1, e.far || 0, e.farPhase || 0], o);
    }
    return this;
  }

  /** Indices (into the packed arrays) of events whose lip sheet may be visible near time t, most
   *  important first (the caller draws as many as it has ribbons for). Explore mode: over the
   *  ribbons' range around the along-shore focus (Schedule.focusRef, the shared uFocus, when the
   *  caller passes none), those whose lip is up near the focus first. */
  jetSlots(t, focusX) {
    if (!CONFIG.explore) {
      const out = [];
      this.active.forEach((e, i) => {
        const [a, b] = e._lip || Schedule.prepare(e)._lip;
        if (t > a && t < b) out.push(i);
      });
      return out;
    }
    const fx = focusX ?? Schedule.focusRef?.value.x ?? this.focusX;
    this.focusX = fx;
    const cand = [];
    const n = Math.round((2 * X_BEACH) / XS_STEP) + 1;
    const i0 = Math.max(0, Math.floor((fx - LIP_HALF + X_BEACH) / XS_STEP));
    const i1 = Math.min(n - 1, Math.ceil((fx + LIP_HALF + X_BEACH) / XS_STEP));
    this.active.forEach((e, i) => {
      if (!e._lipLo) Schedule.prepare(e);
      // distance from the focus to the nearest place where this event's lip (or crest skin) is up
      let dmin = Infinity;
      for (let k = i0; k <= i1; k++) {
        if (t > e._lipLo[k] && t < e._lipHi[k]) dmin = Math.min(dmin, Math.abs(-X_BEACH + k * XS_STEP - fx));
      }
      // (quantised to 4 m so the order does not flicker between two lips at similar distances)
      if (dmin < Infinity) cand.push([i, Math.floor(dmin / 4)]);
    });
    cand.sort((p, q) => p[1] - q[1] || p[0] - q[0]);
    return cand.map((c) => c[0]);
  }
}

export { MAX_EVENTS };
