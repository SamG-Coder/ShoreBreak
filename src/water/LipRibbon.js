import * as THREE from 'three';
import { CONFIG } from '../config.js';
import { Schedule, LIP_HALF } from '../core/schedule.js';
import { WATER_PRELUDE } from './WaterSurface.js';
import { WATER_SHADE } from '../glsl/water.js';
import { LIP_SNAP, makeLipColumns, LIP_MORPH_GLSL } from './lipLod.js';
export { LIP_SNAP } from './lipLod.js';

// The plunging lip ("jet") and the barrel ceiling of one breaker event, as a sheet
// swept along the crest (x). Its cross-section (glsl/breaker.js brkLip) runs:
//   seg 0  apex -> lip root (top)            (continues the heightfield back face)
//   seg 1  outer surface, root -> knee -> foot: the jet (brkLip Bezier, whose end is the
//          ballistic tip = the KNEE) and the curtain hanging below it (grows with the flight)
//   seg 2  rounded foot of the curtain (fingers)
//   seg 3  inner surface, foot -> root        (the barrel ceiling)
//   seg 4  fillet from the underside root down to the vertical wall of the heightfield
//          face (the barrel's back wall)
//   seg 5  the barrel wall and floor: lies exactly on the heightfield face down to near the
//          toe (polygon offset)
// Seen from the beach: a torn, aerated line on the knee (spray whitening the jet above it as the
// flight goes on) over a short translucent curtain, the olive-teal barrel showing through and
// below it (breaker_morphology §5.3, colour report §4).
// Rows 4-5 are drawn twice with the same program: by a "barrel tint" pass (the lip's shadow on
// the pocket: the water surface drawn behind pulled toward the pocket's in-scatter colour,
// premultiplied over) and by the sheet pass (only the white roller of a spilling crest there).
// The heightfield (WaterSurface) carries the back face, the wall and the foot; its crest
// cap is hidden inside this sheet. Lip parts that fall into the water fade out, so the
// curtain is absorbed by the trough at impact; after touch-down the sheet fades out (the
// white water there is the whitewater particles'). Chop, shallow-water set-up and other
// events are added exactly as on the heightfield so the seams coincide.

const SEGS = [8, 32, 4, 16, 8, 12]; // rows per segment (seg 0 .. 5); last row closes seg 5

// Columns (along-shore x of each cross-section).
// Clip: fixed x in [xMin, xMax], denser in the middle (the lip / barrel at the break line: 1.5 cm),
// coarser toward the ends, which only carry the far crest skin (4 cm).
function columnsClip(xMin, xMax, cols) {
  const xs = [];
  for (let i = 0; i < cols; i++) {
    const u = (2 * i) / (cols - 1) - 1;
    xs.push(0.5 * (xMin + xMax) + 0.5 * (xMax - xMin) * (0.65 * u + 0.35 * u * u * u));
  }
  return xs;
}
// Explore: offsets from a centre that follows the focus (the player) in steps of LIP_SNAP, in LOD
// bands that double their spacing with the distance (2 cm near the player .. 32 cm at 20-40 m; the
// crest is seen ever more end-on further along the beach). Every band starts on a multiple of its
// own spacing and LIP_SNAP is a multiple of all of them, so each column stays on the same world x
// while the ribbon recentres. The vertex morph merges departing columns before
// those band edges move, including the back-depth and barrel-tint passes.

function buildGeometry(xs) {
  const cols = xs.length;
  const rows = [];
  SEGS.forEach((n, s) => { for (let i = 0; i < n; i++) rows.push(s + i / n); });
  rows.push(SEGS.length);
  const nr = rows.length;
  const pos = new Float32Array(nr * cols * 3);
  const row = new Float32Array(nr * cols);
  let k = 0;
  for (let j = 0; j < nr; j++) {
    for (let i = 0; i < cols; i++) {
      pos[k * 3] = xs[i];
      row[k] = rows[j];
      k++;
    }
  }
  // triangles are emitted back to front within the one draw call: barrel wall (seg 5),
  // fillet (seg 4), then the lip itself, which blends over them
  const idx = [], idxBarrel = [];
  const bands = [...Array(nr - 1).keys()];
  const order = (j) => { const sg = Math.floor(rows[j]); return sg === 5 ? 0 : sg === 4 ? 1 : 2; };
  bands.sort((p, q) => order(p) - order(q) || p - q);
  for (const j of bands) {
    for (let i = 0; i < cols - 1; i++) {
      const a = j * cols + i, b = a + 1, c = a + cols, d = c + 1;
      idx.push(a, c, b, b, c, d); // counter-clockwise seen from outside the water body
      if (rows[j] >= 4) idxBarrel.push(a, c, b, b, c, d);
    }
  }
  const mk = (ix) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', posAttr);
    g.setAttribute('aRow', rowAttr);
    g.setIndex(ix);
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5);
    return g;
  };
  const posAttr = new THREE.BufferAttribute(pos, 3), rowAttr = new THREE.BufferAttribute(row, 1);
  return { sheet: mk(idx), barrel: mk(idxBarrel) };   // (the barrel pass shares the vertex buffers)
}

// shared by the vertex shaders of both passes
const SLOT = /* glsl */ `
uniform int uSlot;
uniform int uPass;       // 0: sheet, 1: barrel tint (both premultiplied over)
// 0 off (URL ?lipdbg=N), 3 aeration, 4 opacity / Fresnel / scattering, 5 barrel rows, 6 overdraw (additive),
// 7 crest skin (coverage, brightness, normal.y)
uniform int uLipDebug;
// x: world x of the geometry's x = 0 (explore: the focus snapped to LIP_SNAP), y: half-width of
// the ribbon, z: focus x (the player; own copy of the shared uFocus.x)
uniform vec4 uLipX0;
${LIP_MORPH_GLSL}
// the sheet fades out over the last ~6 m of the ribbon (explore): no hard end along the beach
float lipEdge(float x) { return 1.0 - smoothstep(uLipX0.y - 6.3, uLipX0.y - 0.6, abs(x - uLipX0.z)); }
`;

const SHEET = /* glsl */ `
// the lower edge of the hanging curtain is uneven: a gentle, non-periodic undulation (the torn
// foot itself is an alpha dissolve in the fragment shader - regular geometric lobes read as a
// scalloped glass edge)
float lipFinger(float x, float seed) {
  return 0.6 * vnoise(vec2(x * 6.3 + seed * 31.0, 0.5)) + 0.4 * vnoise(vec2(x * 15.7 - seed * 7.0, 1.5));
}
// Length of the curtain hanging below the knee (the ballistic tip of brkLip). It forms under the
// lip during the flight (S5) and reaches ~0.14 m: a short translucent veil under the aerated lip line,
// the dark barrel showing through it and below it (breaker_morphology §5.3); its foot reaches the
// trough just before the knee lands, which closes each section with a near-vertical curtain edge.
float lipDrape(Brk b, BrkLip L) { return 0.14 * b.s * smoothstep(0.0, 0.15, L.fl) * L.grow; }
#define LIP_TGONE 0.32

// (explore) The cross-section of a column stays inside the box z in [zI - 2.6 k, zI + 0.9 k],
// y in [-0.45 k, 0.9 k] (k = max(s, 1): crest skin from the arrival on, lip, curtain, barrel, plus the
// set-up and chop). A column whose box lies entirely outside the view frustum (widened by 30 %) is
// not evaluated at all: it collapses like a hidden one. (Its neighbours' transition triangles are
// off-screen too.) Never in clip mode. From the event's nominal values (EA = uEvtA: landing z,
// height), with margins for their along-shore variation (zI +-0.45 m, s +30 %): no brkAt here.
bool lipCulled(float x, vec4 EA) {
  if (uLipX0.y > 1e3) return false;
  float k = max(1.3 * EA.y / BRK_HREF, 1.0);
  mat4 VP = projectionMatrix * viewMatrix;
  vec4 c0 = VP * vec4(x, -0.45 * k, EA.z - 0.45 - 2.6 * k, 1.0);
  vec4 c1 = VP * vec4(x, 0.9 * k, EA.z - 0.45 - 2.6 * k, 1.0);
  vec4 c2 = VP * vec4(x, -0.45 * k, EA.z + 0.45 + 0.9 * k, 1.0);
  vec4 c3 = VP * vec4(x, 0.9 * k, EA.z + 0.45 + 0.9 * k, 1.0);
  const float M = 1.3;
  vec4 xs = vec4(c0.x, c1.x, c2.x, c3.x), ys = vec4(c0.y, c1.y, c2.y, c3.y), ws = vec4(c0.w, c1.w, c2.w, c3.w) * M;
  return all(lessThan(xs + ws, vec4(0.0))) || all(lessThan(ws - xs, vec4(0.0)))
      || all(lessThan(ys + ws, vec4(0.0))) || all(lessThan(ws - ys, vec4(0.0)))
      || all(lessThan(ws, vec4(0.0)));
}

// Raw (event-only) cross-section point of row r at slice b.x: (z, y).
// meta = (sigma, thickness, tn, flight); meta2 = (knee sigma, sheet length (m), curtain length, -);
// knee: (z, y) of the knee
vec2 sheetRaw(Brk b, float r, float t, out vec4 meta, out vec4 meta2, out vec2 knee) {
  float tn = brkTnLip(b, t);   // (the thrown sheet: not held by a spilling crest, brkTn)
  // the heightfield's own stage (held clock), shared by every profile evaluation below (one
  // inlined brkStage instead of five: D3D compile time)
  float tnH = brkTn(b, t);
  BrkStage gH = brkStage(tnH);
  vec2 AH = brkApex(b, gH, tnH);
  meta = vec4(0.0, 0.0, tn, 0.0);
  meta2 = vec4(1.0, 0.0, 0.0, 0.0);
  knee = vec2(0.0);
  float seg = floor(r), f = r - seg;
  vec2 q = AH;
  if (tn < BRK_TB - 0.01 && brkArrival(b, t) > brkWhitecapOnset(b) - 0.02 && seg < 4.0 && b.str > 0.0) {
    // crest skin: before the lip is born, rows 0..4 cover the crest from 15 cm behind the
    // apex down the front face (0.16 m drop, 0.32 m for a spilling crest), 4 mm above the
    // water, for the whitecap
    BrkStage g = gH;   // (before the lip: tnH = tn)
    vec2 A = AH;
    // (rows spread evenly over the skin: seg 0-3 have 8 / 32 / 4 / 16 rows - mapped by row value the
    // coarse seg 2 made a visible horizontal seam across the whitecap)
    float ri = seg < 1.0 ? r * 8.0 : (seg < 2.0 ? 8.0 + f * 32.0 : (seg < 3.0 ? 40.0 + f * 4.0 : 44.0 + f * 16.0));
    float pp = ri / 60.0;
    float xi = pp < 0.3 ? mix(-0.15, 0.0, pp / 0.3)
                        : brkFrontXi(g, 1.0 - mix(0.0, mix(0.16, 0.32, smoothstep(0.4, 0.9, b.spill)), (pp - 0.3) / 0.7) / max((g.ya - g.yt) * b.s, 0.05));
    float z = A.x + xi * b.s;
    vec3 pr = brkProfileS(b, gH, AH, tnH, z);
    vec2 nrm = normalize(vec2(-pr.y, 1.0));
    meta.xy = vec2(pp, -1.0);
    q = vec2(z, pr.x) + vec2(nrm.x, nrm.y) * 0.004;
  } else
  // (nothing of the sheet is left once it has sunk into the bore, tn > 0.3: the crest top, the
  // last part to go, is gone by then)
  if (tn < BRK_TB - 0.01 || tn > LIP_TGONE || b.str <= 0.0) {
    // collapsed inside the wave body (hidden; thickness tag -2)
    meta.y = -2.0;
    q = AH + vec2(-0.1, -0.25) * b.s;
  } else {
  BrkLip L = brkLip(b, t);
  meta.w = L.fl;
  // root of the jet (sigma 0): point and tangent (P1 - R is horizontal)
  vec2 R0 = L.R, t0 = normalize(L.P1 - L.R), n0 = vec2(-t0.y, t0.x);
  // knee (ballistic tip) and the hanging curtain below it, its foot hanging lower in lobes
  vec4 cK = brkLipCurve(L, 1.0);
  float Ld = lipDrape(b, L);
  vec2 Fp = cK.xy + Ld * vec2(0.216, -0.976);
  Fp.y -= lipFinger(b.x, b.seed) * 0.035 * b.s * smoothstep(0.06, 0.2, L.fl);
  float lenJ = 0.5 * (length(L.T - L.R) + length(L.P1 - L.R) + length(L.P2 - L.P1) + length(L.T - L.P2));
  float lenD = length(Fp - cK.xy);
  float sk = lenJ / max(lenJ + lenD, 1e-4);   // rows spread evenly along the whole sheet
  meta2 = vec4(sk, lenJ + lenD, lenD, 0.0);
  knee = cK.xy;
  if (seg < 1.0) {
    // from 8 cm behind the apex on the back face (where it leaves the heightfield) over the
    // crest to the lip root, lifted a few mm so the coarser heightfield cap never pokes through
    float zb = L.A.x - 0.16 * b.s;
    vec3 bp = brkProfileS(b, gH, AH, tnH, zb);
    vec2 Bk = vec2(zb, bp.x);
    vec2 top = R0 + n0 * (0.5 * L.wR + 0.006 * b.s);
    // C1 shoulder: match the body slope at the back and the jet's horizontal
    // tangent at the root. No piecewise-linear crest plate.
    float f2=f*f, f3=f2*f, span=top.x-zb;
    q=vec2(mix(zb,top.x,f), (2.0*f3-3.0*f2+1.0)*Bk.y
      +(f3-2.0*f2+f)*span*bp.y+(-2.0*f3+3.0*f2)*top.y);
    meta.xy = vec2(0.0, L.wR);
  } else if (seg < 4.0) {
    // centreline point of this row: jet (Bezier) up to the knee, then the curtain (Hermite from
    // the knee, leaving along the jet, hanging near-vertically at the foot)
    float sg = seg < 2.0 ? f : (seg < 3.0 ? 1.0 : 1.0 - f);
    vec4 c = vec4(0.0); float w = 0.0;
    if (sg <= sk || lenD < 1e-4) {
      float sj = min(sg / max(sk, 1e-4), 1.0);
      c = brkLipCurve(L, sj);
      w = brkLipWidth(L, sj);
    } else {
      float u = (sg - sk) / max(1.0 - sk, 1e-4);
      vec2 m0 = cK.zw * lenD * 0.8, m1 = vec2(0.1, -0.995) * lenD * 0.8;
      float u2 = u * u, u3 = u2 * u;
      vec2 p = (2.0 * u3 - 3.0 * u2 + 1.0) * cK.xy + (u3 - 2.0 * u2 + u) * m0 + (-2.0 * u3 + 3.0 * u2) * Fp + (u3 - u2) * m1;
      vec2 dp = (6.0 * u2 - 6.0 * u) * cK.xy + (3.0 * u2 - 4.0 * u + 1.0) * m0 + (6.0 * u - 6.0 * u2) * Fp + (3.0 * u2 - 2.0 * u) * m1;
      float ld = length(dp);
      c = vec4(p, ld > 1e-6 ? dp / ld : vec2(0.0, -1.0));
      w = L.wT * mix(1.0, 0.55, u);
    }
    vec2 nrm = vec2(-c.w, c.z);
    if (seg < 2.0) q = c.xy + nrm * (0.5 * w + 0.006 * b.s);
    else if (seg < 3.0) { float th = 3.14159265 * f; q = c.xy + (nrm * cos(th) + c.zw * sin(th)) * 0.5 * w; }
    else q = c.xy - nrm * 0.5 * w;
    meta.xy = vec2(sg, w);
  } else if (seg >= 5.0) {
    // barrel wall (f < 0.65: on the heightfield face from the cap bottom down to the trough) and
    // floor (the trough in front of it, in the shadow of the curtain, to just past the landing line)
    BrkStage g = L.g;
    float hw = max(g.hw, 0.05);
    float h = mix(hw, 0.02, min(f / 0.65, 1.0));
    float u = 1.0 - (1.0 - g.uw) * pow(h / hw, 1.0 / max(g.m, 1.0));
    float z = L.A.x + u * g.lf * b.s;
    z = mix(z, max(b.zI + 0.12 * b.s, z + 0.05), clamp((f - 0.65) / 0.35, 0.0, 1.0));
    q = vec2(z, brkProfileS(b, gH, AH, tnH, z).x);
    meta.xy = vec2(-1.0 - f, L.wR);
  } else {
    // fillet: underside root -> wall point W on the heightfield face (tangent join)
    vec2 u0 = R0 - n0 * 0.5 * L.wR;
    // (kept 1 cm behind the wall so the fillet never z-fights the discretised cap)
    BrkStage g = L.g;
    float zW = L.A.x + g.uw * g.lf * b.s * 0.999;
    vec3 pw = brkProfileS(b, gH, AH, tnH, zW);
    vec2 W = vec2(zW - 0.01 * b.s, pw.x);
    vec2 dW = vec2(0.0, -1.0);
    float k = 0.9 * length(W - u0);
    float f2 = f * f, f3 = f2 * f;
    q = (2.0 * f3 - 3.0 * f2 + 1.0) * u0 + (f3 - 2.0 * f2 + f) * vec2(-k, 0.0) + (-2.0 * f3 + 3.0 * f2) * W + (f3 - f2) * dW * k;
    meta.xy = vec2(-f, L.wR);
  }
  if (seg < 4.0) {
    // at birth the lip lies just inside the crest and rises out of it (no seam strip)
    // (a spilling crest is already white: its lip carries on from the whitecap, not hidden)
    q.y -= 0.012 * b.s * (1.0 - L.grow) * (1.0 - smoothstep(0.4, 0.9, b.spill));
    // after touch-down the sheet sinks into the bore while it fades out
    q.y -= 0.5 * b.s * smoothstep(0.06, 0.5, tn);
  } else if (tn > 0.18) {
    // the pocket is gone: fold the fillet and wall rows back into the body
    meta.y = -2.0;
    q = AH + vec2(-0.1, -0.25) * b.s;
  }
  }
  return q;
}

// Everything the heightfield adds on top of this event (chop, set-up, other events) at xz,
// with its gradient (so neighbouring points can be offset without re-evaluating it).
// own: this event's own profile at xz (brkProfile), ownDX: its elevation at (xz.x + 0.04, xz.y)
// (from the caller's neighbour slice).
vec3 sheetResidual(vec3 own, vec2 xz, float t, float ownDX) {
  vec2 gr; vec4 inf;
  float wh = waterHeight(xz, t, 0.016, gr, inf);
  // waterHeight's d/dx is a forward difference over 4 cm: cancel our own part the same way
  return vec3(wh - own.x, gr.x - (ownDX - own.x) / 0.04, gr.y - own.y);
}
`;

const VERT = /* glsl */ `
in float aRow;
out vec3 vWorld;
out vec3 vNrm;
out float vViewZ;
out vec4 vLip;    // (row, sigma, thickness, height above the heightfield)
out vec4 vLip2;   // (tn, flight time, residual, x)
out vec4 vLip3;   // (knee sigma, sheet length, curtain length, -)
out vec2 vKnee;   // knee (z, y)

void main() {
  float x = lipColumn(position.x + uLipX0.x);
  float t = uTime;
  vec4 EA = uEvtA[uSlot];
  // (a column outside the view: collapsed, nothing evaluated)
  vec4 meta = vec4(0.0, -2.0, 9.0, 0.0), meta2 = vec4(1.0, 0.0, 0.0, 0.0); vec2 knee = vec2(0.0);
  // The section point and its three finite-difference neighbours are evaluated in ONE loop
  // with a trip count the compiler cannot know (uEvtCount >= 0): D3D/FXC otherwise inlines
  // sheetRaw() four times and the program takes about a minute to compile under ANGLE.
  // The slice (brkAt) is evaluated in the loop too (one call site; the along-crest neighbour at
  // x + 4 cm also gives the own part of the residual's forward difference).
  // (the barrel-tint pass needs no normal: one evaluation)
  float dr = 0.004, dx = 0.04;
  vec2 q = vec2(EA.z - 0.5, -0.25 * EA.y / BRK_HREF), qa = vec2(0.0), qb = vec2(0.0), qx = vec2(0.0);
  Brk b = brkAt(uSlot, x); float ownDX = 0.0;
  int nEval = (lipCulled(x, EA) ? 0 : (uPass == 1 ? 1 : 4)) + min(uEvtCount, 0);
  for (int k = 0; k < nEval; k++) {
    float rr = k == 1 ? min(aRow + dr, 6.0) : (k == 2 ? max(aRow - dr, 0.0) : aRow);
    Brk bk = brkAt(uSlot, x + (k == 3 ? dx : 0.0));
    vec4 mk, mk2; vec2 kn;
    vec2 qk = sheetRaw(bk, rr, t, mk, mk2, kn);
    if (k == 0) { b = bk; q = qk; meta = mk; meta2 = mk2; knee = kn; if (meta.y < -1.5) break; }
    else if (k == 1) qa = qk;
    else if (k == 2) qb = qk;
    else { qx = qk; ownDX = brkProfile(bk, q.x, t).x; }
  }
  if (meta.y < -1.5) {
    // hidden vertex: no residual / normal work (most of the sheet, most of the time)
    vWorld = vec3(x, q.y, q.x); vNrm = vec3(0.0, 1.0, 0.0);
    vLip = vec4(aRow, 0.0, -2.0, -1.0); vLip2 = vec4(meta.z, 0.0, 0.0, x); vLip3 = vec4(1.0, 0.0, 0.0, 0.0); vKnee = vec2(0.0);
    vec4 mv0 = viewMatrix * vec4(vWorld, 1.0);
    vViewZ = -mv0.z;
    gl_Position = projectionMatrix * mv0;
    return;
  }
  vec3 own = brkProfile(b, q.x, t);
  float above = q.y - own.x;
  vec3 res = sheetResidual(own, vec2(x, q.x), t, ownDX);
  vec3 P = vec3(x, q.y + res.x, q.x);
  // normal from the actual sheet: finite differences of the raw section along the profile
  // and along x (evaluated in the loop above), plus the (linearised) residual
  vec3 N = vec3(0.0, 1.0, 0.0);
  if (uPass == 0) {
    vec3 Tp = vec3(0.0, qa.y - qb.y + res.z * (qa.x - qb.x), qa.x - qb.x);
    vec3 Tx = vec3(dx, qx.y - q.y + res.y * dx + res.z * (qx.x - q.x), qx.x - q.x);
    N = cross(Tp, Tx);
    float ln = length(N);
    N = ln > 1e-12 ? N / ln : vec3(0.0, 1.0, 0.0);
  }
  vWorld = P;
  vNrm = N;
  vLip = vec4(aRow, meta.x, meta.y, above);
  vLip2 = vec4(meta.z, meta.w, res.x, x);
  vLip3 = meta2;
  vKnee = knee;
  // A camera-directed offset pulled the barrel through the crest from behind.
  // Offset only a few millimetres toward its physical front; depth handles occlusion.
  if (aRow >= 4.0) P.z += 0.002;
  vec4 mv = viewMatrix * vec4(P, 1.0);
  vViewZ = -mv.z;
  gl_Position = projectionMatrix * mv;
}`;

const VERT_BACK = /* glsl */ `
in float aRow;
out vec3 vWorld;
out vec3 vNrm;
out float vViewZ;
out vec4 vLip;
out vec4 vLip2;
out vec4 vLip3;
out vec2 vKnee;
void main() {
  float x = lipColumn(position.x + uLipX0.x);
  vec4 meta = vec4(0.0, -2.0, 9.0, 0.0), meta2 = vec4(1.0, 0.0, 0.0, 0.0); vec2 knee = vec2(0.0);
  vec4 EA = uEvtA[uSlot];
  vec2 q = vec2(EA.z - 0.5, -0.25 * EA.y / BRK_HREF);
  float above = -1.0;
  if (!lipCulled(x, EA)) {
    Brk b = brkAt(uSlot, x);
    q = sheetRaw(b, aRow, uTime, meta, meta2, knee);
    above = q.y - brkProfile(b, q.x, uTime).x;
  }
  vec2 gr; vec4 inf;
  bool hidden = meta.y < -1.5;
  float res = hidden ? 0.0 : waterHeight(vec2(x, q.x), uTime, 0.016, gr, inf) - (q.y - above);   // same position as the main pass
  if (hidden) { meta.y = -2.0; above = -1.0; }
  vec3 P = vec3(x, q.y + res, q.x);
  vWorld = P; vNrm = vec3(0.0, 1.0, 0.0);
  vLip = vec4(aRow, meta.x, meta.y, above);
  vLip2 = vec4(meta.z, meta.w, res, x);
  vLip3 = meta2;
  vKnee = knee;
  vec4 mv = viewMatrix * vec4(P, 1.0);
  vViewZ = -mv.z;
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = /* glsl */ `
in vec3 vWorld;
in vec3 vNrm;
in float vViewZ;
in vec4 vLip;
in vec4 vLip2;
in vec4 vLip3;
in vec2 vKnee;

// torn, bubbly feathering foam in world units (clumps 10-30 cm, holes), tumbling down the face
vec2 spillFoam(vec3 P, float seed) {
  float n = 0.5 + 0.3 * gnoise(vec2(P.x * 3.5 + seed * 9.0, P.y * 22.0 + uTime * 1.6))
                + 0.2 * gnoise(vec2(P.x * 11.0 - uTime * 0.2, P.y * 45.0 + uTime * 2.5));
  float holes = 0.5 + 0.5 * gnoise(vec2(P.x * 8.0 + 3.1, P.y * 30.0 + uTime * 2.0));
  return vec2(n, holes);
}
// Spilling whitecap / roller (wave C): ONE world-space field shared by the crest skin, the lip and
// the barrel rows, so the white stays continuous where a column switches from the skin to the lip.
// White from the crest down to a torn lower edge that creeps down the face (0.28 -> 0.6 of the
// face height over ~0.4 s); bubbly (holes of 3-10 cm, ~isotropic on screen), tumbling down; it
// rides the collapsing crest and hands over to the whitewater at the plunge.
vec2 spillWhite(Brk b, vec3 P, float t) {
  float tn = brkTn(b, t);
  vec2 A = brkApex(b, brkStage(tn), tn);
  float drop = A.y - P.y;
  float age = brkArrival(b, t) - brkWhitecapOnset(b);
  float sv = 0.95 * (P.y - A.y) - 0.31 * (P.z - A.x);
  // torn 2D foam tumbling down the face: three octaves ~isotropic on screen (world x and the height
  // seen along the view ray) - strands elongated down-slope read as a comb of white hair hanging from
  // the crest, and a single strong octave with a hard threshold as leopard spots
  float small = 1.0 - smoothstep(0.35, 0.6, b.s);
  float n = 0.5 + 0.28 * gnoise(vec2(P.x + b.seed * 0.7, (sv + 0.3 * t) * 1.3) * mix(17.0, 14.0, small))
                + 0.16 * gnoise(mat2(0.8, -0.6, 0.6, 0.8) * vec2(P.x - b.seed * 0.3, sv + 0.45 * t) * 41.0)
                + 0.09 * gnoise(mat2(0.6, 0.8, -0.8, 0.6) * vec2(P.x, sv + 0.5 * t) * 97.0);
  // (tongues of white spill further down the face in places)
  float tongue = smoothstep(0.45, 0.9, vnoise(vec2(P.x * 7.0 + b.seed * 3.0, 0.5 * t)));
  // (a small crest - clip's 'a' - is white over most of its low face)
  float reachF = mix(0.6, 0.9, small);
  float reach = 1.38 * b.H * mix(0.1, reachF, smoothstep(0.0, 0.7, age)) + 0.08 * (n - 0.5) + 0.07 * b.s * tongue * smoothstep(0.1, 0.5, age);
  float lower = 1.0 - smoothstep(reach - 0.05, reach, drop);
  // ends over the back of the crest in a lumpy but dense edge (lumps of 10-20 cm: fine noise in a
  // wide fade leaves a grey half-transparent strip over the crest's shaded back - dirty flecks
  // along the top of the roller - where the clip's roller has a bright white top)
  float lumpB = 0.65 * vnoise(vec2(P.x * 5.0 + b.seed * 13.0, 0.5 * t)) + 0.35 * vnoise(vec2(P.x * 13.0 - b.seed * 5.0, 0.9 * t));
  float back = smoothstep(-0.07 * b.s - 0.03, -0.02 * b.s, P.z - A.x + 0.1 * (lumpB - 0.5));
  float dd = clamp(drop / max(reach, 0.05), 0.0, 1.0);
  // (soft threshold: dense white on the tumbling top of the roller, torn translucent patches with
  // holes further down the face)
  float dens = smoothstep(0.32 - 0.4 * (1.0 - dd), 0.72 - 0.3 * (1.0 - dd), n) * mix(0.95, 0.5, dd);
  // (x: coverage, y: brightness: the lumps of the roller shade each other, and the roller's
  // underside is in its own shade - a bright top, greyer toward its lower edge)
  return vec2(smoothstep(0.0, 0.12, age) * lower * back * dens * (1.0 - smoothstep(0.05, 0.3, tn)) * 0.9,
              (0.8 + 0.36 * clamp(n - 0.4, 0.0, 0.5)) * mix(1.06, 0.72, smoothstep(0.15, 0.95, dd)));
}

void lipMain() {
  if (uLipDebug == 6) { gl_FragColor = vec4(uPass == 1 ? vec3(0.0) : vec3(0.06, 0.0, 0.0), 0.0); return; }   // overdraw heat map
  float row = vLip.x;
  if (uPass == 1) {
    // ---- barrel tint (multiplicative blend: dst * rgb). The lip shades the pocket from the high
    // sun behind the wave, but the water there stays lit through: a turquoise tint that deepens
    // toward the foot of the wall (colour report: face under the lip #7BA8A6, barrel foot
    // (79-88, 122-138, 111-127)), fading out over the last few cm above the toe.
    if (row < 4.0 || vLip.z < -1.5) discard;
    float tn = vLip2.x;
    // (the pocket opens as the curtain forms under the thrown lip)
    float cover = smoothstep(-0.33, -0.2, tn) * (1.0 - smoothstep(-0.02, 0.16, tn));
    float fw = clamp(row - 5.0, 0.0, 1.0);   // 0 top of the wall .. 0.65 trough .. 1 past the landing line
    float foot = row < 5.0 ? 0.0 : smoothstep(0.2, 0.45, fw) * (1.0 - smoothstep(0.82, 1.0, fw));
    // (the shaded pocket is olive-teal, not cyan: clip barrel foot (83-100, 124-133, 113-126), G > B
    // by 7-10 - the fines stirred up by the breaker scatter the little light that reaches it: the
    // water behind is pulled toward that in-scatter colour, premultiplied over)
    // The pocket is visible from the shore-facing side only; soften its
    // contribution at a grazing side view instead of exposing a dark sheet.
    vec3 viewDir=normalize(cameraPosition-vWorld);
    float facing=smoothstep(0.02,0.32,viewDir.z);
    float k = mix(0.06, 0.30, foot) * cover * facing;
    gl_FragColor = vec4(vec3(0.15, 0.185, 0.155) * k, k);
    if (uLipDebug == 5) gl_FragColor = vec4(0.1, 0.1, 0.5, 0.5);
    return;
  }
  vec3 P = vWorld;
  Brk b = brkAt(uSlot, P.x);
  float spillK = smoothstep(0.3, 0.9, b.spill);
  vec2 swv = spillK > 0.0 ? spillWhite(b, P, uTime) : vec2(0.0, 1.0);
  float sw = spillK * swv.x;
  // triangles spanning a crest-skin column (thickness tag -1) and a lip / hidden column are
  // stretched across the state change: drop them (a spilling crest: only its white, continuous)
  if (vLip.z < -0.001 && abs(vLip.z + 1.0) > 0.001) {
    if (sw < 0.004 || vLip.z < -1.0) discard;
    gl_FragColor = vec4(shadeFoam(vec3(0.0, 1.0, 0.0), sw) * swv.y * sw, sw);
    return;
  }
  if (vLip.z < -0.5) {
    // ---- crest skin (before the lip exists): feathering / spilling whitecap, premultiplied
    float pp = vLip.y;
    float cap = brkWhitecap(b, uTime);
    float age = brkArrival(b, uTime) - brkWhitecapOnset(b);   // normalised time since onset
    vec2 fn = spillFoam(P, b.seed);
    float n = fn.x, holes = fn.y;
    // feathering: a torn line on the apex, widening down the face
    float reachF = 0.36 * smoothstep(0.0, 0.45, age) - 0.15 * (n - 0.5);
    float bandF = smoothstep(0.12, 0.3, pp) * (1.0 - smoothstep(reachF - 0.15, reachF, pp));
    float densF = smoothstep(0.35, 0.8, n) * smoothstep(0.2, 0.6, holes + 0.1) * mix(1.0, 0.55, pp) * mix(0.45, 0.85, smoothstep(0.1, 1.0, age));
    // spilling (wave C): the torn whitecap band / roller (spillWhite); the face under it stays glassy
    float a = clamp(max(cap * bandF * densF * (1.0 - spillK), sw), 0.0, 1.0);
    if (a < 0.004) discard;
    vec3 N = normalize(vNrm);
    // (the foam lumps scatter light all round: mostly the up-facing lighting, the shade of the roller's
    // underside comes from swv.y - the analytic skin normal flips across a steep front and would
    // print a horizontal band and grey flecks into the white)
    vec3 fc = shadeFoam(normalize(mix(N, vec3(0.0, 1.0, 0.0), 0.8)), a) * mix(1.0, swv.y, spillK);
    gl_FragColor = vec4(fc * a, a);
    if (uLipDebug == 7) gl_FragColor = vec4(a, swv.y * 0.8, N.y * 0.5 + 0.5, 1.0);
    return;
  }
  float tn = vLip2.x, fl = vLip2.y;
  // the lip emerges from the crest (no hard edge along the crest) and fades out after touch-down
  // (under a spilling whitecap the glassy lip only shows once it has grown out of the white)
  float born = smoothstep(BRK_TB + 0.02 * spillK, BRK_TB + mix(0.08, 0.22, spillK), tn);
  float gone = 1.0 - smoothstep(0.0, 0.12, tn);
  if (row >= 4.0) {
    // ---- barrel rows in the sheet pass: only the white roller of a spilling crest, which keeps
    // pouring down the upper face under the lip (same foam field as the whitecap and the lip)
    float fa = sw;
    if (fa < 0.004) discard;
    vec3 N = normalize(vNrm);
    vec3 fc = shadeFoam(normalize(mix(N, vec3(0.0, 1.0, 0.0), 0.5)), fa) * swv.y;
    gl_FragColor = vec4(fc * fa, fa);
    if (uLipDebug == 5) gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
    return;
  }

  vec3 V = normalize(cameraPosition - P);
  vec3 N = normalize(vNrm);
  if (!gl_FrontFacing) N = -N;
  // the side we see must face the eye (the interpolated finite-difference normal can tip past
  // grazing on the curling sheet): keep N.V >= 0.05
  float nv0 = dot(N, V);
  if (nv0 < 0.05) N = normalize(N + V * (0.05 - nv0));
  float sg = vLip.y, w = vLip.z;
  float sk = vLip3.x, lenT = vLip3.y, lenD = vLip3.z;
  float dK = (sg - sk) * lenT;   // along the sheet from the knee (m): < 0 on the jet, > 0 on the curtain
  float eF = lenD - dK;          // up the curtain from its foot (m)
  // world size of a pixel here: thresholds on fine noise are widened to >= ~1 px (anti-aliasing, and
  // no stencil-cut holes when the sheet passes a wading viewer); nearK: the sheet within ~3 m
  float pxW = 2.0 * vViewZ / (projectionMatrix[1][1] * uResolution.y);
  float nearK = 1.0 - smoothstep(1.5, 3.5, vViewZ);
  float late = smoothstep(-0.05, 0.0, tn);   // the last instants before touch-down
  // the part of the free sheet that has fallen into the water is absorbed (soft edge, no alpha test)
  float wet = sg > 0.35 ? smoothstep(-0.012, -0.002, vLip.w) : 1.0;
  // the foot of the curtain dissolves over 3-5 cm in a torn, non-periodic edge (no pane edge, no rim)
  float nFt = 0.6 * gnoise(vec2(P.x * 40.0 + b.seed * 17.0, dK * 22.0 - fl * 3.0)) + 0.4 * gnoise(vec2(P.x * 95.0 - b.seed * 5.0, dK * 45.0 + 3.0));
  float footA = lenD > 0.02 ? smoothstep(0.004, 0.05 + 2.0 * pxW, eF + 0.02 * nFt) : 1.0;
  wet *= footA;
  // (the reflection and its Fresnel rise toward grazing on the rounded edges: faded near the free
  // edge, or the outline of the sheet shows as a thin bright line)
  float edgeK = lenD > 0.02 ? smoothstep(0.004, 0.03, eF + 0.012 * nFt) : 1.0;

  // noise coordinates ~isotropic on screen (world x, and height seen along the ~18 deg view ray),
  // moving with the knee (explore: any view direction - isotropic on the sheet itself: x and the
  // distance along the sheet from the knee)
  vec2 bp = vec2(P.x, uLipX0.y > 1e3 ? 0.95 * (P.y - vKnee.y) - 0.31 * (P.z - vKnee.x) : dK);
  // ---- aeration. One torn foam texture (three octaves, 1-5 cm, drifting with the fall) thresholded
  // at the local foam density D: coverage = smoothstep(1 - D -+ tw, th). Dense where D ~ 1, torn
  // patches with holes of 3-10 cm at D ~ 0.5, sparse flecks at D ~ 0.3; the threshold width tw is at
  // least ~1 px (anti-aliased at any distance: from close up, holes fade to translucent water instead
  // of being stencil-cut). A faint milky haze ~ D fills in between.
  float th = 0.5 + 0.28 * gnoise(bp * 21.0 + b.seed * 13.0) + 0.16 * gnoise(mat2(0.8, -0.6, 0.6, 0.8) * bp * 53.0 - vec2(0.0, fl * 3.0))
                 + 0.09 * gnoise(mat2(0.6, 0.8, -0.8, 0.6) * bp * 127.0 + vec2(b.seed * 5.0, -fl * 6.0));
  float tw = max(0.06, 22.0 * pxW);
  // The white lip line is a torn, bubbly band on the jet just behind the knee (6-13 cm, holes of 3-10
  // cm), thrown white ~0.25 s before touch-down (tn -0.30 .. -0.19, appearing first as broken,
  // feathered patches growing out of the whitening crest root: t_lip = t_i - 0.23 s, timeline §2); the
  // overhang forming before that is glassy. The onset is ragged in 2D (not x-only dashes).
  float nOn = vnoise(vec2(P.x * 5.0 + b.seed * 19.0, 3.7));
  float on = smoothstep(0.0, 0.07, tn - mix(-0.30, -0.24, nOn) + 0.08 * (th - 0.5) + 0.04 * gnoise(vec2(P.x * 14.0 + b.seed * 3.0, dK * 30.0)));
  // (the band covers the top of the jet toward the knee: foreshortened to 0.01-0.015 v)
  // it widens back over the whole jet as the flight goes on: just before touch-down the thrown
  // sheet is a white veil from the crest down to the curtain (the white curtain at the peel point)
  float bw = b.s * mix(0.06, 0.13, vnoise(vec2(P.x * 4.0 - b.seed * 7.0, 8.1))) * smoothstep(0.0, 0.15, fl);
  float wide = smoothstep(-0.085, -0.015, tn);   // the last ~0.07 s of the flight
  bw = mix(bw, (lenT - lenD) * 0.75, wide);
  // both edges feathered over 1.5-2 cm and torn (the root side of the widened band thins out: no
  // hard edge along the crest)
  float eN = gnoise(vec2(P.x * 19.0 - b.seed * 9.0, fl * 2.0 + 1.7));
  float band = smoothstep(-bw - 0.045 - 0.08 * wide, -bw + 0.01 + 0.06 * wide, dK + 0.012 * eN)
             * (1.0 - smoothstep(-0.008, 0.03, dK - 0.008 * eN));
  // (soft and torn at first - a thin bubbly rim - denser as the flight goes on)
  float Dband = band * on * mix(mix(0.5, 0.82, smoothstep(0.04, 0.18, fl)), 0.62, wide);
  // (white streaks hang from the band into the curtain in places)
  float drip = smoothstep(0.55, 0.8, vnoise(vec2(P.x * 30.0 + b.seed * 5.0, 2.3 + dK * 12.0))) * (1.0 - smoothstep(0.0, 0.06, dK)) * smoothstep(-0.02, 0.0, dK) * on;
  // the jet between the crest root and the band whitens over the flight: first the crest root (S5,
  // tn ~ -0.22), then spray and haze over the whole jet from the crest top down to the lip line
  // (S6-S7; the glassy part is the curtain under it)
  float jetPos = sg / max(sk, 0.05);   // 0 root .. 1 knee
  // (the root whitens in patches along the crest; over the crest top itself - the silhouette - only
  // faintly: a white outline along the crest reads as a drawn line)
  float rootP = mix(0.35, 1.0, smoothstep(0.25, 0.7, vnoise(vec2(P.x * 3.1 + b.seed * 11.0, fl * 1.5))));
  float Djet = max(0.5 * rootP * smoothstep(-0.30, -0.18, tn) * (1.0 - smoothstep(0.1, 0.7, jetPos + 0.25 * (th - 0.5))),
                   0.5 * smoothstep(-0.15, -0.04, tn))
             * (1.0 - smoothstep(-0.03, 0.0, dK)) * smoothstep(0.0, 0.04, fl) * (row < 1.0 ? smoothstep(0.55, 1.0, row) : 1.0);
  float Dlip = max(Dband, 0.55 * drip);
  // (a wide ramp above the threshold: translucent torn edges thickening into dense cores)
  float aerLip = (smoothstep(1.0 - Dlip - tw, 1.0 - Dlip + tw + 0.3, th) * 0.92 + 0.25 * Dlip) * min(Dlip * 10.0, 1.0);
  // (the spray over the jet: finer, drawn out ~2:1 along the fall - spray streaming off the lip)
  float thJ = 0.5 + 0.26 * gnoise(vec2(bp.x * 34.0 + b.seed * 3.0, bp.y * 15.0 - fl * 4.0)) + 0.16 * gnoise(vec2(bp.x * 83.0 - b.seed, bp.y * 38.0 - fl * 7.0))
                  + 0.1 * (th - 0.5) * 2.0;
  aerLip = max(aerLip, (smoothstep(1.0 - Djet - tw, 1.0 - Djet + tw + 0.3, thJ) * 0.85 + 0.25 * Djet) * min(Djet * 10.0, 1.0));
  // In the last instants the stretched curtain tears into a misty white veil of short streaks (2-5 cm
  // apart, a few cm long, jittered in spacing and broken along their length; fainter from close up)
  float xw = P.x + 0.012 * gnoise(vec2(P.x * 7.0 + b.seed * 5.0, dK * 9.0 - fl * 1.5));
  float thV = 0.5 + 0.3 * gnoise(vec2(xw * 33.0 + b.seed * 9.0, dK * 9.0 - fl * 5.0)) + 0.2 * gnoise(vec2(xw * 80.0 - b.seed * 3.0, dK * 22.0 - fl * 7.0));
  float Dveil = smoothstep(-0.035, 0.0, tn) * smoothstep(0.0, 0.03, dK) * mix(0.3, 0.6, late) * (1.0 - 0.4 * nearK);
  float aerVeil = (smoothstep(1.0 - Dveil - tw, 1.0 - Dveil + tw + 0.25, thV) * 0.75 + 0.35 * Dveil) * min(Dveil * 10.0, 1.0);
  // where the curtain enters the water it breaks up: a ragged 5-10 cm line, whitening 5-10 cm up the
  // curtain at touch-down (the first white at the base, S7) before the whitewater takes over
  // (only the free sheet: the crest-top rows lie on the heightfield's back face by construction)
  float plunge = smoothstep(0.05 + 0.08 * late, 0.0, vLip.w) * smoothstep(0.15, 0.3, fl) * smoothstep(0.35 - 0.2 * late - tw, 0.6 + tw, th) * mix(0.45, 0.8, late)
               * smoothstep(-0.06, -0.02, dK) * step(1.0, row);
  // spilling breakers (wave C): the torn white roller continues from the crest-skin whitecap on
  // the jet (same world-space foam field); the little curtain below stays glassy
  float spillW = sw * (1.0 - 0.6 * smoothstep(0.0, 0.06, dK));
  float aer = clamp(max(aerLip, aerVeil) * (1.0 - spillK) + plunge, 0.0, 1.0);

  WIn s;
  s.P = P; s.N = N; s.V = V; s.viewZ = vViewZ; s.thickHint = -1.0;
  s.foam = aer; s.foamDense = aer; s.turb = aer; s.crest = 1.0; s.rough = 0.06; s.milk = aer * 0.4;

  // ---- lit-through water: the sheet is the same water as the face. Reflection: the sky over the
  // visible facet distribution with Smith shadowing, exactly as on the heightfield (no bright
  // mirror line at the crest). Body: sunlight forward-scattered by the micro-bubbles of the thin
  // sheet (sigma_s ~ 32 /m) glows turquoise (uGlowTint, as the heightfield's thin crests);
  // the rest of the view ray passes through to the barrel behind (already in the colour buffer).
  vec3 gn = gnoised(vec2(P.x * 23.0 + b.seed * 3.0, dK * 7.0 - uTime * 1.5));
  // (the falling curtain is lumpy: 5-10 cm undulations catch the sky and thin / thicken it)
  // (and the jet's surface carries the ripples of the face it was thrown from, 5-15 cm: a smooth
  // analytic jet reflects one flat sky colour - a pale slab from close up)
  vec3 gj = gnoised(vec2(P.x * 11.0 - b.seed * 5.0, dK * 13.0 - uTime * 1.1));
  N = normalize(N + mix(0.1, 0.22, smoothstep(0.0, 0.05, dK)) * vec3(gn.y, 0.0, gn.z) * smoothstep(-0.1, 0.05, dK)
                  + 0.09 * vec3(gj.y, 0.0, gj.z) * (1.0 - smoothstep(-0.1, 0.0, dK)));
  float NoV = max(dot(N, V), 0.0);
  vec4 rr = roughReflection(N, V, 0.1, uSeaTune2.x);
  rr *= edgeK;
  float Fr = rr.a;
  float fwd = 0.42 + 0.72 * pow(max(dot(-V, uSunDir), 0.0), 4.0);
  // (micro-bubbles and the fine sediment the breaker keeps in suspension also scatter: the thin
  // curtain reads milky, olive-tinged teal rather than clean cyan - clip: G > B by 7-10 over the barrel)
  vec3 glow = mix(uGlowTint * uOptics.z * fwd * 1.15, uMilkColor * (uSunColor * 0.25 + skyAmbient(N)) * 0.5, 0.3);
  glow *= vec3(1.1, 1.02, 0.97);
  // the curtain thins and thickens along the crest in streaks every 2-5 cm, only in the last instants
  // (the stretched sheet tearing), and short along the fall
  float str = 0.5 + 0.3 * gnoise(vec2(P.x * 38.0 + b.seed * 7.0, dK * 20.0 - fl * 4.0)) + 0.2 * gnoise(vec2(P.x * 90.0, dK * 30.0));
  float wv = w * mix(1.0, 0.85 + 0.3 * str, smoothstep(-0.02, 0.04, dK) * smoothstep(-0.035, 0.0, tn));
  // (the sheet thins and thickens in 10-20 cm lumps: lighter and darker patches of lit-through water)
  wv *= 1.0 + 0.3 * gj.x;
  float rootL = 0.25 * b.s * (1.0 - smoothstep(0.0, 0.35, sg / max(sk, 0.05)));   // near the root: the thick jet
  float path = wv / max(NoV, 0.2) + rootL;
  // (the thin curtain is translucent: the dark barrel shows through it, sigma ~15 /m; thicker sheets
  // - the jet near its root, a thick curtain seen close up - take the body colour)
  float fill = 1.0 - exp(-(5.0 + 10.0 * aer) * path);
  float Tm = exp(-dot(uWaterAtten, vec3(0.3333)) * path * 3.0);
  vec3 H = normalize(V + uSunDir);
  float NoL = max(dot(N, uSunDir), 0.0), NoH = max(dot(N, H), 0.0);
  // (a soft sheen only: a sharp sun lobe on the smooth analytic sheet reads as a bright bar)
  vec3 spec = uSunColor * D_GGX(NoH, 0.22) * V_SmithGGXCorrelated(max(NoV, 1e-3), NoL, 0.22) * NoL * fresnelWater(max(dot(V, H), 0.0));
  vec3 emitW = glow * fill * (1.0 - Fr) + rr.rgb + min(spec * 0.3, vec3(1.0));
  float opac = 1.0 - (1.0 - Fr) * (1.0 - fill) * Tm;
  // (the bubbly lip is lit through: bias its foam normal toward the sun-facing up direction)
  vec3 fc = shadeFoam(normalize(mix(N, vec3(0.0, 1.0, 0.0), 0.6)), aer);
  // (the lumps of the torn foam shade each other: thin parts greyer, clumps bright)
  fc *= mix(0.8, 1.04, smoothstep(0.35, 0.85, th));
  float cov = clamp(aer, 0.0, 1.0);
  float alpha = mix(opac, 1.0, cov);
  vec3 emit = emitW * (1.0 - cov) + fc * cov;
  if (row < 1.0) {
    // crest top: from the heightfield's back face (same shading) into the glowing jet root
    vec3 hf = shadeWater(s, gl_FragCoord.xy);
    float k = smoothstep(0.35, 1.0, row) * smoothstep(BRK_TB, BRK_TE, tn);
    emit = mix(hf, emit / max(alpha, 1e-3), k);
    alpha = smoothstep(0.0, 0.45, row);
    emit *= alpha;
    gone = 1.0 - smoothstep(0.1, 0.3, tn);
  }
  gl_FragColor = vec4(emit, alpha) * (born * gone * wet);
  // the spilling white carries straight on from the crest skin (no birth fade)
  vec3 fs = shadeFoam(normalize(mix(N, vec3(0.0, 1.0, 0.0), 0.6)), spillW) * swv.y;
  gl_FragColor = gl_FragColor * (1.0 - spillW) + vec4(fs * spillW, spillW) * wet;
  if (uLipDebug == 3) gl_FragColor = vec4(vec3(aer), 1.0);
  if (uLipDebug == 4) gl_FragColor = vec4(opac, Fr * 5.0, fill, 1.0);
}
void main() {
  lipMain();
  // (explore) fade out toward the ends of the ribbon (both passes are premultiplied over)
  // (and within ~0.3 m of the eye: no slab cutting through a wading viewer)
  float edge = lipEdge(vWorld.x) * smoothstep(0.12, 0.45, vViewZ);
  if (edge < 1.0) gl_FragColor *= edge;
  if (gl_FragColor.a < 0.003) discard;
}`;

const FRAG_BACK = /* glsl */ `
in vec3 vWorld;
in vec3 vNrm;
in float vViewZ;
in vec4 vLip;
in vec4 vLip2;
in vec4 vLip3;
in vec2 vKnee;
void main() {
  // only the free sheet (lip) is an exit surface into the air; the fillet and anything
  // inside the wave body must not make the water in front of it look thin
  if (vLip.x >= 4.0 || vLip.w < 0.002 || vLip.z < -0.001 || lipEdge(vWorld.x) < 0.5) discard;
  gl_FragColor = vec4(1.0 / vViewZ, 0.0, 0.0, 1.0);
}`;

export class LipRibbon {
  constructor(shared, slot, { xMin = -4.0, xMax = 4.0, cols = 340 } = {}) {
    const explore = CONFIG.explore;
    const geo = LipRibbon.geo || (LipRibbon.geo = buildGeometry(explore && !location.search.includes("lipclip") ? makeLipColumns(LIP_HALF) : columnsClip(xMin, xMax, cols)));
    this.geometry = geo.sheet;
    const prelude = WATER_PRELUDE() + SLOT;
    const uSlot = { value: slot }, uLipDebug = { value: +(new URLSearchParams(location.search).get('lipdbg') || 0) };
    // clip: absolute x, no end fade; explore: recentred on the focus (updated before each draw)
    const uLipX0 = { value: new THREE.Vector4(0, explore ? LIP_HALF : 1e4, 0, 0) };
    const uniforms = { ...shared, uSlot, uPass: { value: 0 }, uLipDebug, uLipX0 };
    this.uniforms = uniforms;
    // the schedule picks the events whose lips are up around the focus (jetSlots)
    if (shared.uFocus) Schedule.focusRef = shared.uFocus;
    const recentre = explore && shared.uFocus ? () => {
      const fx = shared.uFocus.value.x;
      uLipX0.value.set(Math.round(fx / LIP_SNAP) * LIP_SNAP, LIP_HALF, fx, 0);
    } : null;
    const common = { uniforms, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2 };
    const vs = prelude + SHEET + VERT, fs = prelude + WATER_SHADE + FRAG;
    // premultiplied blending over the water already drawn behind (the barrel wall)
    this.material = new THREE.ShaderMaterial({
      ...common, vertexShader: vs, fragmentShader: fs,
      side: THREE.FrontSide, transparent: true, depthWrite: false,
      blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
    });
    // barrel tint: the same program (identical sources: three.js shares it), premultiplied over
    // (the water behind pulled toward the pocket's in-scatter colour), drawn before the sheets
    // (renderOrder) over the barrel rows only
    this.barrelMaterial = new THREE.ShaderMaterial({
      ...common, uniforms: { ...uniforms, uPass: { value: 1 } }, vertexShader: vs, fragmentShader: fs,
      side: THREE.FrontSide, transparent: true, depthWrite: false,
      blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,
    });
    this.backMaterial = new THREE.ShaderMaterial({ ...common, vertexShader: prelude + SHEET + VERT_BACK, fragmentShader: prelude + FRAG_BACK, side: THREE.BackSide });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.barrelMesh = new THREE.Mesh(geo.barrel, this.barrelMaterial);
    this.barrelMesh.renderOrder = -1;
    this.mesh.add(this.barrelMesh);   // follows the sheet's visibility (it is in the same scene)
    this.backMesh = new THREE.Mesh(this.geometry, this.backMaterial);
    this.mesh.frustumCulled = this.backMesh.frustumCulled = this.barrelMesh.frustumCulled = false;
    if (recentre) this.mesh.onBeforeRender = this.barrelMesh.onBeforeRender = this.backMesh.onBeforeRender = recentre;
  }
  set slot(i) { this.uniforms.uSlot.value = i; }
  set visible(v) { this.mesh.visible = this.backMesh.visible = v; }
}
