// Measured "look" data from the reference clip: sky LUT, light rig and camera sway.
import { CONFIG } from '../config.js';

const SAT = 1.35; // saturation applied by the phone-look post (see post/Post.js)

// Sky, display-linear Rec.709 medians by "dv above the horizon" (colour analysis §2).
const SKY_DV = [
  [0.004, 0.2918, 0.4851, 0.5906], [0.008, 0.2961, 0.4910, 0.5972], [0.015, 0.3140, 0.5029, 0.6038],
  [0.025, 0.3231, 0.5089, 0.6240], [0.035, 0.3325, 0.5271, 0.6376], [0.050, 0.3185, 0.5395, 0.6584],
  [0.070, 0.2789, 0.5395, 0.6939], [0.090, 0.2462, 0.5457, 0.7157], [0.120, 0.2086, 0.5457, 0.7305],
  [0.150, 0.1946, 0.5395, 0.7454], [0.180, 0.1590, 0.5271, 0.7454], [0.220, 0.1329, 0.5029, 0.7305],
  [0.260, 0.1119, 0.4735, 0.7305], [0.300, 0.1022, 0.4678, 0.7231], [0.340, 0.0887, 0.4564, 0.7084],
];
// beyond the top of the frame (reflections only): deepening blue toward the zenith
const SKY_HIGH = [[35, 0.060, 0.380, 0.680], [90, 0.035, 0.270, 0.580]];

function unsaturate([r, g, b]) {
  const Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return [Y + (r - Y) / SAT, Y + (g - Y) / SAT, Y + (b - Y) / SAT].map((v, i) => Math.max(v * (CONFIG.explore ? [1.06, 0.94, 1.045][i] : 1), 0));
}

/** vec4[18] (elevation deg, r, g, b) in pre-grade scene-linear for our camera model. */
export function skyLut() {
  const C = CONFIG.camera;
  const fpx = 0.5 / Math.tan((C.vfov * Math.PI) / 360); // focal length in image heights
  const vh = 0.34658;                                   // mean horizon v
  const rows = [];
  for (const [dv, r, g, b] of SKY_DV) {
    const v = vh - dv;
    const el = Math.atan((0.5 - v) / fpx) * (180 / Math.PI) + C.pitchDeg; // pitchDeg is negative
    rows.push([el, ...unsaturate([r, g, b])]);
  }
  rows.unshift([0, ...rows[0].slice(1)]);
  for (const h of SKY_HIGH) rows.push([h[0], ...unsaturate(h.slice(1))]);
  const out = new Float32Array(18 * 4);
  rows.slice(0, 18).forEach((row, i) => out.set(row, i * 4));
  return out;
}

// Sky above the frame top as seen in reflections (display-linear, like SKY_DV): a clear
// Mediterranean summer sky deepens quickly toward the zenith. This part of the sky is what
// the camera-facing wind-ripple facets of the far sea reflect.
export const SKY_HIGH_REFL = [[30, 0.075, 0.415, 0.700], [45, 0.045, 0.330, 0.630], [65, 0.030, 0.260, 0.560], [90, 0.022, 0.215, 0.500]];

/**
 * The same sky as a 256 x 1 float texture indexed by sqrt(sin(elevation)) (fine near the horizon),
 * for cheap multi-tap reflection lookups in the water shader (glsl/water.js skyRefl()).
 * Piecewise-linear interpolation of skyLut(), identical to skyRadiance() in common.js.
 */
export function skyTexture(THREE, high = SKY_HIGH_REFL) {
  const lut = skyLut();
  const rows = [];
  for (let i = 0; i < 16; i++) rows.push(Array.from(lut.slice(i * 4, i * 4 + 4)));
  for (const h of high) rows.push([h[0], ...unsaturate(h.slice(1))]);
  const n = 256;
  const data = new Float32Array(n * 4);
  for (let j = 0; j < n; j++) {
    const u = (j + 0.5) / n;
    const e = (Math.asin(Math.min(u * u, 1)) * 180) / Math.PI;
    let c = rows[0].slice(1);
    for (let i = 1; i < rows.length; i++) {
      const a = rows[i - 1], b = rows[i];
      if (e >= a[0]) { const f = Math.min(Math.max((e - a[0]) / (b[0] - a[0]), 0), 1); c = [1, 2, 3].map((k) => a[k] + (b[k] - a[k]) * f); }
    }
    data.set([c[0], c[1], c[2], 1], j * 4);
  }
  const tex = new THREE.DataTexture(data, n, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

// Light rig (colour analysis §8): sun ≈ 2.1 x (1.00, 0.95, 0.91); soft sky ambient that
// reproduces the phone's lifted shadows (effective sun:sky ≈ 1.6:1).
export const LIGHT = {
  sun: CONFIG.sun.color.map((c) => c * CONFIG.sun.intensity),
  skyAmb: [0.32, 0.36, 0.425],
};

// Camera sway measured from the horizon + buoy track at 5 Hz: [t, rotX, rotY, rotZ] (rad).
const KF = [[0.0,-0.18565,0.00319,-0.01331],[0.2,-0.18533,0.00243,-0.01284],[0.4,-0.1852,0.00169,-0.01251],
[0.6,-0.18559,0.00084,-0.01211],[0.8,-0.18643,0.00087,-0.01203],[1.0,-0.18739,0.00081,-0.01223],
[1.2,-0.18846,0.00054,-0.01208],[1.4,-0.18941,0.00081,-0.01247],[1.6,-0.19026,0.00129,-0.01271],
[1.8,-0.19083,0.00145,-0.01278],[2.0,-0.19129,0.00192,-0.01274],[2.2,-0.19201,0.0018,-0.0125],
[2.4,-0.19273,0.00161,-0.01169],[2.6,-0.19339,0.00142,-0.0109],[2.8,-0.19406,0.00089,-0.00947],
[3.0,-0.19468,0.00054,-0.00836],[3.2,-0.19528,0.00035,-0.00646],[3.4,-0.19587,0.00017,-0.00499],
[3.6,-0.19652,0.0003,-0.00368],[3.8,-0.19707,0.00048,-0.00281],[4.0,-0.19762,0.00078,-0.00198],
[4.2,-0.19798,0.0015,-0.00157],[4.4,-0.19786,0.00172,-0.00159],[4.6,-0.19735,0.00188,-0.00194],
[4.8,-0.19615,0.00144,-0.00234],[5.0,-0.19447,0.00116,-0.00292],[5.2,-0.19244,0.00117,-0.00359],
[5.4,-0.19005,0.00114,-0.00396],[5.6,-0.18785,0.00059,-0.00399],[5.8,-0.18547,0.00052,-0.00408],
[6.0,-0.1835,0.00022,-0.00378],[6.2,-0.18183,0.00018,-0.00374],[6.4,-0.18047,-0.00052,-0.00349],
[6.6,-0.17957,-0.00094,-0.00377],[6.8,-0.17897,-0.00201,-0.00371],[7.0,-0.17865,-0.00258,-0.00355],
[7.2,-0.1787,-0.00292,-0.00341],[7.4,-0.1792,-0.00317,-0.00292],[7.6,-0.18096,-0.00316,-0.00253],
[7.8,-0.18397,-0.00359,-0.00226],[8.0,-0.18823,-0.00396,-0.00192],[8.2,-0.19359,-0.00488,-0.00202],
[8.4,-0.19958,-0.00593,-0.00242]];

function catmull(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}
function keyed(t) {
  const n = KF.length;
  const f = Math.min(Math.max(t / 0.2, 0), n - 1.0001);
  const i = Math.floor(f), u = f - i;
  const k = (j) => KF[Math.min(Math.max(j, 0), n - 1)];
  return [1, 2, 3].map((c) => catmull(k(i - 1)[c], k(i)[c], k(i + 1)[c], k(i + 2)[c], u));
}
// loopable fit of the same sway (camera analysis §7), in radians
function procedural(t) {
  const s = (a, f, p) => a * Math.sin(2 * Math.PI * f * t + p);
  const d2r = Math.PI / 180;
  return [
    -(10.921 + s(0.516, 0.160, -1.843) + s(0.255, 0.240, 0.269)) * d2r,
    (-0.141 + s(0.268, 0.050, 0.976) + s(0.075, 0.140, 2.754)) * d2r,
    (-0.561 + s(0.434, 0.060, -0.799) + s(0.133, 0.240, 1.505)) * d2r,
  ];
}
/** Camera rotation (x pitch, y yaw, z roll) at clip time t: measured inside the clip, fitted loop outside. */
export function cameraSway(t) {
  if (t >= 0 && t <= 8.4) return keyed(t);
  const p = procedural(t);
  const edge = t < 0 ? 0 : 8.4;
  const w = Math.min(Math.abs(t - edge) / 0.8, 1);
  if (w >= 1) return p;
  const k = keyed(edge);
  return k.map((v, i) => v + (p[i] - v) * (w * w * (3 - 2 * w)));
}
