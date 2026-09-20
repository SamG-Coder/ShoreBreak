import * as THREE from 'three';

// Explore-mode terrain around the player (beach, storm berm, back-beach, seabed).
//
// One static mesh of tensor-product patches whose columns are offsets from the snapped focus
// (uBeachFocus.x rounded to LAND.snap) and whose rows are world z. Column spacing grows in
// world-aligned levels (2.5 cm .. 0.8 m, every level spacing divides the snap step), so a vertex
// keeps its world position when the focus snaps and only the level boundaries move. Beyond the
// levels (|dx| > 64 m, and everywhere on the relief-free seabed) the terrain is invariant along x,
// so coarse columns that slide with the snap are exact. Near patches meet on flat,
// shared rows: overlapping different tessellations can fight in the depth buffer.
// Rows are dense (2.5 cm) over the swash face, where the render-only drape
// relief (bedRelief) and the millimetre water films need them, 10 cm on the dry beach.
//
//   swash   x in f +- 64 m, z in [-1.40, 5.00]   2.5 cm .. 0.8 m columns, 2.5 cm rows
//   dry     x in f +- 64 m, z in [5, 40]         0.2 .. 0.8 m columns, 10 / 25 cm rows
//   seabed  x in f +- 64 m, z in [-60, -1.4]     3.2 m columns, 4 cm .. 2.5 m rows
//   far     |x - f| in [64, 4096] m              stitched into coarse columns / rows
//
// Behind the crest (z ~ 19) the back-slope levels out into a flat back-beach at the foot of the
// promenade wall (Promenade.js, z = LAND.wallZ).

export const LAND = {
  snap: 0.8,          // focus snap (m): multiple of every aligned column spacing
  wallZ: 40.0,        // promenade wall face
  backY: 1.55,        // back-beach level (m above SWL)
  zMax: 40.0,
  reach: 4096,        // along-shore extent either side of the focus (m)
  bayR: 10000,        // the Baie des Anges: beyond |x| = bayX0 the shore curves seaward (radius, m)
  bayX0: 150,
};

// Each patch must finish retiring a column BEFORE the snapped window can
// remove it. The dry patch changes spacing at 8/16 m; the swash at 16/32 m.
export const LAND_MORPH = [
  [.05,.6,1.2,.6,1.2],
  [.10,2.,3.2,2.,3.2],
  [.20,5.,7.2,5.,7.2],
  [.40,11.,15.2,4.,7.2],
  [.80,23.,31.2,9.,15.2],
];
export const LAND_MORPH_GLSL = /* glsl */ `
float landMorph(float x,float stepSize,float start,float end){
  float target=floor(x/stepSize+.001)*stepSize;
  return mix(x,target,smoothstep(start,end,abs(x-uBeachFocus.x)));
}
float landColumnX(vec3 p){
  float ox=floor(uBeachFocus.x/${LAND.snap.toFixed(3)}+.5)*${LAND.snap.toFixed(3)};
  float x=p.x+ox;
  // Blend the schedules on the final swash rows, so the shared z=5 edge
  // uses identical positions on both patches without adding a seam.
  float dry=smoothstep(4.5,5.,p.z);
  ${LAND_MORPH.map(([s,a,b,c,d])=>`x=landMorph(x,${s.toFixed(3)},mix(${a.toFixed(3)},${c.toFixed(3)},dry),mix(${b.toFixed(3)},${d.toFixed(3)},dry));`).join('\n  ')}
  return x;
}
`;

// CPU mirror of the generated column transform for continuity regression tests.
export function morphTerrainColumn(offset,z,focus){
  let x=offset+Math.floor(focus/LAND.snap+.5)*LAND.snap;
  const dry=smooth(4.5,5,z);
  for(const [step,a,b,c,d] of LAND_MORPH){
    const w=smooth(a+(c-a)*dry,b+(d-b)*dry,Math.abs(x-focus));
    x+=(Math.floor(x/step+.001)*step-x)*w;
  }
  return x;
}

// Render-only bay curvature: every land layer (terrain, promenade, city, palms) is drawn shifted
// seaward by bayBend(x) (world x, so nothing moves while walking; zero within +-150 m, where the
// swash, the breakers and the player live). The far shore swings toward the sea at both ends
// (36 m at 1 km, 170 m at 2 km, 740 m at 4 km) instead of running dead straight to a vanishing
// point. bayBendSlope: dz/dx of the shift (rotates the horizontal normals of the far land).
export const BEND_GLSL = /* glsl */ `
float bayBend(float x) { float d = max(abs(x) - ${LAND.bayX0.toFixed(1)}, 0.0); return d * d / ${(2 * LAND.bayR).toFixed(1)}; }
float bayBendSlope(float x) { return sign(x) * max(abs(x) - ${LAND.bayX0.toFixed(1)}, 0.0) / ${LAND.bayR.toFixed(1)}; }
// a horizontal direction / normal of the unbent world, rotated with the bent shore at x
vec3 bayRotate(vec3 n, float x) {
  float s = bayBendSlope(x), c = inversesqrt(1.0 + s * s);
  return vec3(c * n.x + s * c * n.z, n.y, -s * c * n.x + c * n.z);   // (1,0,0) -> tangent (1,0,-s)
}
`;
// far along-shore columns (offsets from the snapped focus) shared by the terrain skirts and the
// promenade, so the curved bay bends both along the same chords (<= 64 m: sagitta < 6 cm)
export const FAR_X = [63.2, 72, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512];
for (let x = 576; x <= LAND.reach; x += 64) FAR_X.push(x);
export const bayBendJS = (x) => { const d = Math.max(Math.abs(x) - LAND.bayX0, 0); return (d * d) / (2 * LAND.bayR); };

// relief keep factors (by distance from the player, where the columns stay fine enough)
// and the terrain height used by the vertex shader. Requires NOISE, BED, MACRO_NOISE_GLSL, uBeachFocus.
export const TERRAIN_GLSL = /* glsl */ `
// x: high-frequency drape (cusps 0.43 m, lumps 0.25 m), y: low-frequency drape, z: upper-beach swells
vec3 terrainKeep(vec2 xz) {
  float d = abs(xz.x - uBeachFocus.x);
  // Wide, C2-continuous bands: relief resolves over several walking steps.
  vec3 t=clamp((vec3(d)-vec3(.8,3.,18.))/vec3(4.8,14.,40.),0.,1.);
  return 1.-t*t*t*(t*(t*6.-15.)+10.);
}
// behind the berm the back-slope levels out into the back-beach
float landProfile(float z) {
  float b = bedProfile(z);
  if (z < 20.0) return b;
  float flat_ = ${LAND.backY.toFixed(3)} + 0.1 * softplus_((b - ${LAND.backY.toFixed(3)}) / 0.1);
  return mix(b, flat_, smoothstep(20.0, 22.0, z));
}
// render-only swells of the dry upper beach: old run-up ridges and storm lobes, 3-8 m long,
// 2-4 cm high (zero at the swash patch boundary, z = 5)
float upperRelief(vec2 xz) {
  float w = smoothstep(5.0, 7.0, xz.y) * (1.0 - smoothstep(18.5, 21.0, xz.y) * 0.5);
  if (w <= 0.0) return 0.0;
  vec4 nA = macroNoiseLod(xz + vec2(17.0, 3.0), vec2(5.5, 3.2));
  vec4 nB = macroNoiseLod(xz - vec2(3.0, 9.0), vec2(2.0, 1.5));
  vec4 nX = macroNoiseLod(vec2(xz.x, 0.0), vec2(16.0));
  float ridge = sin(xz.y * 1.9 + 0.9 * nA.w + 0.35 * sin(xz.x * 0.31));
  float u = 0.022 * nA.x + 0.011 * nB.y
          + 0.006 * ridge * smoothstep(6.0, 9.0, xz.y) * (1.0 - smoothstep(14.0, 17.0, xz.y));
  // storm-berm scarps: the last storms cut small steps into the upper face (a 5-12 cm riser over
  // ~0.35 m, relaxing over the next metres); they meander and fade along the shore
  float zs1 = 7.2 + 0.8 * nX.x + 0.25 * sin(xz.x / 7.3);
  float a1 = 0.08 + 0.05 * nX.y;
  float zs2 = 10.4 + 1.0 * nX.z + 0.3 * sin(xz.x / 9.1 + 1.0);
  float a2 = max(0.05 + 0.07 * nX.w, 0.0);
  u += a1 * smoothstep(zs1 - 0.25, zs1 + 0.1, xz.y) * (1.0 - smoothstep(zs1 + 0.5, zs1 + 4.5, xz.y));
  u += a2 * smoothstep(zs2 - 0.25, zs2 + 0.1, xz.y) * (1.0 - smoothstep(zs2 + 0.5, zs2 + 5.0, xz.y));
  // storm cusps on the upper face below the berm crest: horns every ~21 m, 4-6 cm
  float cph = xz.x / 21.0 + 0.3 * sin(xz.x / 37.0 + 0.4) + 0.2 * nX.x;
  float zc = (xz.y - 16.3) / 2.2;
  u += (0.045 + 0.015 * nX.y) * cos(6.2831853 * cph) * exp(-zc * zc);
  return u * w;
}
// pebbles piled against the foot of the promenade wall (storm run-up), 0.3-0.6 m high over
// 0.8-1.8 m, meandering along the shore (JS mirror: wallToeJS)
float wallToe(vec2 xz) {
  if (xz.y < ${(LAND.wallZ - 1.9).toFixed(2)}) return 0.0;
  vec4 n = macroNoiseLod(vec2(xz.x, 7.0), vec2(4.5));
  float w = 1.3 + 0.5 * n.x, h = 0.45 + 0.15 * n.y;
  float u = clamp((xz.y - (${LAND.wallZ.toFixed(2)} - w)) / w, 0.0, 1.0);
  return h * u * u;
}
// out bias: how far the drawn surface is lowered (the relief it no longer carries), so the water
// films tested against the exact bed never disappear under it
float terrainHeight(vec2 xz, vec3 k, out float bias) {
  float z = xz.y;
  float y = landProfile(z);
  bias = 0.0;
  if (z > -1.25 && z < 5.05) {
    float face = smoothstep(-1.2, 0.2, z) * smoothstep(5.0, 2.5, z);
    float rel = bedRelief(xz);
    // low-frequency part of bedRelief (common.js BED); the rest is the high-frequency part
    float lf = (0.003 * sin(xz.x * 5.2 + 0.8 * sin(xz.x * 1.7)) + 0.010 * gnoise(xz * 1.3 + 3.1)) * face;
    y += k.y * lf + k.x * (rel - lf);
    bias = face * ((1.0 - k.x) * 0.008 + (1.0 - k.y) * 0.014);
  }
  if (z > 5.0) y += k.z * upperRelief(xz);
  // Far skirts no longer resolve the irregular wall-foot ridge. Preserve its
  // mean ramp, but fade the varying relief before coarse columns can recycle;
  // otherwise a tiny sand/wall silhouette tick survives the near-beach morph.
  float toeU = clamp((z - ${(LAND.wallZ - 1.3).toFixed(2)}) / 1.3, 0.0, 1.0);
  y += mix(0.45 * toeU * toeU, wallToe(xz), k.z);
  return y;
}
`;

// JS mirror of the rendered ground (landProfile + the upper-beach relief; the few-cm swash-face
// drape is left out) for the player's eye height: groundHeightJS(x, z, noiseTexture).
const smooth = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
function sampleNoise(tex, x, y, lx, ly) {
  const { data, width: N } = tex.image;
  const u = (x / (32 * lx)) * N - 0.5, v = (y / (32 * ly)) * N - 0.5;
  const i0 = Math.floor(u), j0 = Math.floor(v), fu = u - i0, fv = v - j0;
  const out = [0, 0, 0, 0];
  for (let c = 0; c < 4; c++) {
    const at = (i, j) => data[((((j % N) + N) % N) * N + (((i % N) + N) % N)) * 4 + c] / 255;
    const a = at(i0, j0), b = at(i0 + 1, j0), cc = at(i0, j0 + 1), d = at(i0 + 1, j0 + 1);
    out[c] = (a + (b - a) * fu + (cc - a) * fv + (a - b - cc + d) * fu * fv) * 2 - 1;
  }
  return out;
}
export function groundHeightJS(x, z, bedProfile, noiseTex) {
  let y = bedProfile(z);
  if (z >= 20) {
    const sp = (v) => (v > 20 ? v : Math.log1p(Math.exp(v)));
    const flat = LAND.backY + 0.1 * sp((y - LAND.backY) / 0.1);
    y += (flat - y) * smooth(20, 22, z);
  }
  if (z > 5 && noiseTex) {
    const w = smooth(5, 7, z) * (1 - smooth(18.5, 21, z) * 0.5);
    const nA = sampleNoise(noiseTex, x + 17, z + 3, 5.5, 3.2), nB = sampleNoise(noiseTex, x - 3, z - 9, 2, 1.5), nX = sampleNoise(noiseTex, x, 0, 16, 16);
    const ridge = Math.sin(z * 1.9 + 0.9 * nA[3] + 0.35 * Math.sin(x * 0.31));
    let u = 0.022 * nA[0] + 0.011 * nB[1] + 0.006 * ridge * smooth(6, 9, z) * (1 - smooth(14, 17, z));
    const zs1 = 7.2 + 0.8 * nX[0] + 0.25 * Math.sin(x / 7.3), a1 = 0.08 + 0.05 * nX[1];
    const zs2 = 10.4 + 1.0 * nX[2] + 0.3 * Math.sin(x / 9.1 + 1.0), a2 = Math.max(0.05 + 0.07 * nX[3], 0);
    u += a1 * smooth(zs1 - 0.25, zs1 + 0.1, z) * (1 - smooth(zs1 + 0.5, zs1 + 4.5, z));
    u += a2 * smooth(zs2 - 0.25, zs2 + 0.1, z) * (1 - smooth(zs2 + 0.5, zs2 + 5.0, z));
    const cph = x / 21 + 0.3 * Math.sin(x / 37 + 0.4) + 0.2 * nX[0], zc = (z - 16.3) / 2.2;
    u += (0.045 + 0.015 * nX[1]) * Math.cos(2 * Math.PI * cph) * Math.exp(-zc * zc);
    y += u * w;
  }
  if (z > LAND.wallZ - 1.9 && noiseTex) {
    const n = sampleNoise(noiseTex, x, 7, 4.5, 4.5);
    const w = 1.3 + 0.5 * n[0], h = 0.45 + 0.15 * n[1];
    const u = Math.min(Math.max((z - (LAND.wallZ - w)) / w, 0), 1);
    y += h * u * u;
  }
  return y;
}

const r6 = (v) => Math.round(v * 1e6) / 1e6;
// positive offsets 0 .. last boundary, spacing per level ([spacing, boundary] ascending)
function levelOffsets(levels) {
  const out = [0];
  let x = 0;
  for (const [s, b] of levels) {
    const n = Math.round((b - x) / s);
    for (let i = 1; i <= n; i++) out.push(r6(x + i * s));
    x = b;
  }
  return out;
}
const sym = (pos) => [...pos.slice(1).map((v) => -v).reverse(), ...pos];
function range(z0, z1, s) {
  const n = Math.max(1, Math.round((z1 - z0) / s));
  return Array.from({ length: n + 1 }, (_, i) => r6(z0 + ((z1 - z0) * i) / n));
}
function seabedRows(zTop) {
  // from zTop seaward: 4 cm over the step toe, growing over the terrace, 20 cm across the drop-off
  // (knees at -16.8 / -22.5), then out to -60 m
  const zs = [zTop];
  let z = zTop, s = 0.04;
  while (z > -60) {
    if (z > -2.7) s = 0.04;
    else if (z > -15.4) s = Math.min(s * 1.03, 0.25);
    else if (z > -24.5) s = 0.2;
    else s = Math.min(s * 1.15, 2.5);
    z = Math.max(z - s, -60);
    zs.push(r6(z));
  }
  return zs.reverse();
}

export function buildExploreGeometry() {
  const pos = [];
  const idx = [];
  function patch(xs, zs) {
    const base = pos.length / 3, nc = xs.length;
    for (const z of zs) for (const x of xs) pos.push(x, 0, z);
    for (let j = 0; j < zs.length - 1; j++) for (let i = 0; i < nc - 1; i++) {
      const a = base + j * nc + i, b = a + 1, c = a + nc, d = c + 1;
      // rows ascend in z: (a, c, b) is counter-clockwise seen from above
      idx.push(a, c, b, b, c, d);
    }
  }
  function stitchColumns(xa,za,xb,zb){
    const a=pos.length/3;for(const z of za)pos.push(xa,0,z);
    const b=pos.length/3;for(const z of zb)pos.push(xb,0,z);
    let i=0,j=0;
    while(i<za.length-1||j<zb.length-1){
      if(i<za.length-1&&(j===zb.length-1||za[i+1]<=zb[j+1])){idx.push(a+i,a+i+1,b+j);i++;}
      else {idx.push(a+i,b+j+1,b+j);j++;}
    }
  }
  const swashCols = sym(levelOffsets([[0.025, 2], [0.05, 4], [0.1, 8], [0.2, 16], [0.4, 32], [0.8, 64]]));
  const dryCols = sym(levelOffsets([[0.2, 8], [0.4, 16], [0.8, 64]]));
  const seaCols = sym(levelOffsets([[3.2, 64]]));
  const swashRows = range(-1.4, 5.0, 0.025);
  const dryRows = [...range(5.0, 26.0, 0.1), ...range(26.0, LAND.zMax, 0.25).slice(1)];
  const seaRows = seabedRows(-1.4);
  patch(swashCols, swashRows);
  patch(dryCols, dryRows);
  patch(seaCols, seaRows);
  // far skirts (x-invariant terrain): coarse rows everywhere
  const farRows = [...seabedRows(-2.7).filter((z) => z < -2.7).filter((_, i, a) => i % 3 === 0 || i === a.length - 1),
    ...range(-2.7, 5.0, 0.1), ...range(5.0, 26.0, 0.25).slice(1), ...range(26.0, LAND.zMax, 0.5).slice(1)];
  // Connect the complete near boundary to the coarser far rows with a zipper.
  // The previous 0.8 m overlapping skirts could expose a long triangle seam.
  const nearRows=[...seaRows,...swashRows.slice(1),...dryRows.slice(1)];
  const farX = FAR_X.slice(1);
  stitchColumns(64,nearRows,farX[0],farRows);
  stitchColumns(-farX[0],farRows,-64,nearRows);
  patch(farX, farRows);
  patch(farX.map((v) => -v).reverse(), farRows);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(idx), 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5);
  g.userData.stats = { vertices: pos.length / 3, triangles: idx.length / 3 };
  return g;
}
