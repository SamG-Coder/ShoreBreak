import * as THREE from 'three';
import { CONFIG, glslDefines, bedProfileJS } from '../config.js';
import { NOISE, BED, SKY, OPTICS, DEPTH } from '../glsl/common.js';
import { BREAKER } from '../glsl/breaker.js';
import { CHOP, CHOP_FRAG, SWE_SAMPLE, WATER_GEOM, WATER_SHADE, OCEAN } from '../glsl/water.js';
import { FOAM } from '../glsl/foam.js';
import { SWASH_FAR } from '../glsl/swashfar.js';
import { waterLookups, WATER_LOOKUP_GLSL } from './WaterLookups.js';
import { OceanFFT } from './OceanFFT.js';
import { ExploreWaterMesh, EX } from './ExploreMesh.js';
import { skyTexture } from '../core/look.js';
import { SKY_MODEL } from '../sky/skyModel.js';
import { WATERLINE,CAUSTICS,UNDERSIDE } from '../glsl/underwater.js';

// One continuous world-space water mesh from the swash zone to the horizon.
// Rows are dense where the waves break (1.6 cm) and grow geometrically offshore;
// columns fan out with distance so vertex density stays roughly constant on screen.
// (clip mode: the fixed phone camera's trapezoid below; explore mode: ExploreMesh.js, a
//  focus-following wave band + ocean fan that covers every view direction)

// Offshore the row spacing follows the on-screen footprint (≈1.8 px rows at 1080p): the wind
// chop there is shaded from the mipmapped FFT textures, and sub-pixel rows would only cost
// quad over-shading (hundreds of rows used to pile up within a few pixels of the horizon).
function buildRows(zNear, zFar) {
  const rows = [];
  const cz = CONFIG.camera.position[2], h = CONFIG.camera.position[1];
  const f1080 = 1920 / 2 / Math.tan((CONFIG.camera.vfov * Math.PI) / 360);
  let z = zNear;
  while (z > 1.0) { rows.push([z, 0.02]); z -= 0.02; }
  while (z > -1.2) { rows.push([z, 0.016]); z -= 0.016; }
  while (z > -3.8) { rows.push([z, 0.009]); z -= 0.009; }   // breaking faces: sharp crests need fine rows
  let s0 = 0.009;
  let s = s0;
  while (z > zFar) {
    rows.push([z, s]);
    z -= s;
    const D = cz - z;
    s = Math.min(s * 1.3, Math.max(s * 1.004, (1.8 * D * D) / (h * f1080)));
  }
  rows.push([zFar, s]);
  return rows;
}

export function halfWidthAt(z) {
  const cz = CONFIG.camera.position[2];
  return 0.55 + 0.43 * Math.max(cz - z, 0.0);
}

function buildGeometry(cols) {
  const rows = buildRows(2.9, -4500);
  const nr = rows.length, nc = cols;
  const pos = new Float32Array(nr * nc * 3);
  const lod = new Float32Array(nr * nc);
  let k = 0;
  for (let j = 0; j < nr; j++) {
    const [z, s] = rows[j];
    const hw = halfWidthAt(z);
    const dxs = (2 * hw) / (nc - 1);
    for (let i = 0; i < nc; i++) {
      pos[k * 3] = -hw + i * dxs;
      pos[k * 3 + 1] = 0;
      pos[k * 3 + 2] = z;
      lod[k] = Math.max(s, dxs);
      k++;
    }
  }
  const idx = new Uint32Array((nr - 1) * (nc - 1) * 6);
  let t = 0;
  for (let j = 0; j < nr - 1; j++) {
    for (let i = 0; i < nc - 1; i++) {
      const a = j * nc + i, b = a + 1, c = a + nc, d = c + 1;
      // rows run toward -z: (a, b, c) is CCW seen from above (+y)
      idx[t++] = a; idx[t++] = b; idx[t++] = c;
      idx[t++] = b; idx[t++] = d; idx[t++] = c;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('aLod', new THREE.BufferAttribute(lod, 1));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5);
  return g;
}

// (the far-field swash chunk is included once, guarded: other chunks that need it can test SWASH_FAR_INC)
const SWASH_FAR_ONCE = `
#ifndef SWASH_FAR_INC
#define SWASH_FAR_INC
${SWASH_FAR}
#endif
`;
// Bathymetry lookup table: the water shaders evaluate the bed 4-6 times per pixel (depth test,
// refracted path, bed lighting) and the analytic profile costs 16 exp/log each. A 4096-texel
// linear LUT reproduces it to < 0.1 mm (smooth knees >= 0.18 m wide); linear beyond its range.
const BED_LUT_Z = [-64.0, 32.0], BED_LUT_N = 4096;
const BED_LUT = () => {
  const [z0, z1] = BED_LUT_Z, n = BED_LUT_N;
  const f = (v) => v.toFixed(6);
  const s0 = (bedProfileJS(z0 + 0.5) - bedProfileJS(z0)) / 0.5, s1 = (bedProfileJS(z1) - bedProfileJS(z1 - 0.5)) / 0.5;
  return `
${WATER_LOOKUP_GLSL}
float bedProfileT(float z) {
  float u = (z - (${f(z0)})) / ${f(z1 - z0)};
  if (u < 0.0) return max(${f(bedProfileJS(z0))} + ${f(s0)} * (z - (${f(z0)})), ${f(CONFIG.beach.minY)});
  if (u > 1.0) return ${f(bedProfileJS(z1))} + ${f(s1)} * (z - (${f(z1)}));
  return textureLod(uWaterLookup, vec2(u * ${f((n - 1) / n)} + ${f(0.5 / n)}, 0.625), 0.0).r;
}
#define bedProfile bedProfileT
`;
};
export const WATER_PRELUDE = () => glslDefines() + BED_LUT() + NOISE + BED + SKY + OPTICS + DEPTH + BREAKER + CHOP + SWE_SAMPLE + SWASH_FAR_ONCE + WATER_GEOM;

const VERT = /* glsl */ `
#ifdef OPT_EXPLORE
// explore mesh (ExploreMesh.js): position = (column / ray table index, kind, row table index)
uniform sampler2D uExLookup;
uniform vec2 uExOffsets;
uniform vec4 uExFanP;    // fan apex (x, z), along-shore extent (x min, x max)
uniform vec2 uExDetail;
uniform vec2 uFocus;
vec4 exTab(float offset, float i) { int k = int(offset + i + 0.5); return texelFetch(uExLookup, ivec2(k % 1024, k / 1024), 0); }
#else
in float aLod;
#endif
out vec3 vWorld;
out vec3 vNrm;       // macro normal (breaker + shallow water), without the wind chop
out vec4 vInfo;
out float vViewZ;
out vec3 vXZ0;       // undisplaced xz (texture coordinates of the chop) + cat's-paw factor
out vec4 vCrest;     // (on a breaking / shoaling wave, near a breaker face, camera-facing breaker slope, breaker elevation)
void main() {
#ifdef OPT_EXPLORE
  vec2 xz; float aLod;
  {
    vec2 r = exTab(uExOffsets.x, position.z).xy;
    if (position.y < 0.5) {
      vec2 c = exTab(0.0, position.x).xy;
      float d = abs(c.x - uFocus.x);
      // Collapse fine rows before a column enters the next draw tier. The
      // zipper then joins identical profiles instead of suddenly losing rows.
      vec4 morph = smoothstep(vec4(5.,14.,32.,44.),vec4(7.8,19.6,43.,59.2),vec4(d));
      float z = r.x;
      z = mix(z,exTab(uExOffsets.x,floor(position.z/2.)*2.).x,morph.x);
      z = mix(z,exTab(uExOffsets.x,floor(position.z/4.)*4.).x,morph.y);
      z = mix(z,exTab(uExOffsets.x,floor(position.z/8.)*8.).x,morph.z);
      z = mix(z,exTab(uExOffsets.x,floor(position.z/16.)*16.).x,morph.w);
      xz = vec2(c.x,z);
      // Filter the spectrum with a continuous world-space footprint. Neighbour
      // gaps double when a row disappears; using them here made detail pop.
      float H=max(cameraPosition.y,.28),fz=uFocus.y;
      float q0=length(vec2(H,max(0.,max(fz-${EX.detailTop.toFixed(3)},${EX.zBot.toFixed(3)}-fz))));
      float q1=max(length(vec2(H,max(abs(fz-(${EX.zBot.toFixed(3)})),abs(fz-${EX.detailTop.toFixed(3)})))),q0*1.001);
      float q=clamp(d,q0,q1);
      float colFoot=uExDetail.x*(q+d*d/q);
      float base=.009+.007*smoothstep(-1.3,-1.1,z)+.004*smoothstep(.9,1.1,z)+.035*smoothstep(3.35,3.45,z)+.011*max(-3.8-z,0.);
      float rowScale=(1.+morph.x)*(1.+morph.y)*(1.+morph.z)*(1.+morph.w);
      aLod=max(colFoot,max(base*rowScale,uExDetail.y*abs(z-fz)));
    } else {
      vec2 f = exTab(uExOffsets.y, position.x).xy;
      float dz = uExFanP.y - r.x;
      xz = vec2(clamp(uExFanP.x + dz * f.x, uExFanP.z, uExFanP.w), r.x); aLod = max(r.y, dz * f.y);
    }
  }
#else
  vec2 xz = position.xz;
#endif
  vec2 gm, gc; vec3 ch; vec4 info; float paw;
  float y = waterParts(xz, uTime, aLod, gm, ch, gc, info, paw);
  vec2 gBrk = gm;
#ifdef OPT_EXPLORE
  if (info.z > 0.0) { vec4 far; gm += clampLen(sweGradient(xz, 1.5, far), uSweSlope.x); }
#else
  if (info.z > 0.0) {
    vec2 uv = sweUV(xz); vec2 e = uSweTexel * 1.5;
    float wE = textureLod(uSweView, uv + vec2(e.x, 0.0), 0.0).w, wW = textureLod(uSweView, uv - vec2(e.x, 0.0), 0.0).w;
    float wN = textureLod(uSweView, uv + vec2(0.0, e.y), 0.0).w, wS = textureLod(uSweView, uv - vec2(0.0, e.y), 0.0).w;
    gm += clampLen(info.z * vec2(wE - wW, wN - wS) / (2.0 * e * uSweDom.zw), uSweSlope.x);
  }
#endif
  vec3 P = vec3(xz.x + ch.x, y + ch.y, xz.y + ch.z);
  vWorld = P;
  vNrm = normalize(vec3(-gm.x, 1.0, -gm.y));
  vInfo = info;
  vXZ0 = vec3(xz, paw);
  // (x: on a breaking/shoaling wave, y: in the wave zone) for the per-pixel crest optics
  // y: on a breaker's front face or in the trough just ahead of one (the surface rises within
  // 0.7 m seaward): where the per-pixel crest optics are evaluated;
  // z: camera-facing slope of the breaker surface alone (not of the bore / swash)
  float zone = smoothstep(-18.5, -17.5, xz.y) * smoothstep(0.6, 0.3, xz.y);
  float inWave = smoothstep(0.02, 0.08, length(gm)) * zone;
  float nearFace = 0.0;
#ifdef MAIN_PASS
  if (zone > 0.0)
    nearFace = zone * max(max(smoothstep(0.015, 0.04, brkSurfaceOnly(xz - vec2(0.0, 0.7), uTime) - gBrkEta),
                              smoothstep(0.02, 0.06, -gBrk.y)), smoothstep(0.06, 0.12, gBrkEta));
#endif
  vCrest = vec4(inWave, nearFace, max(-gBrk.y, 0.0), gBrkEta);
  vec4 mv = viewMatrix * vec4(P, 1.0);
  vViewZ = -mv.z;
  gl_Position = projectionMatrix * mv;
}`;

const FRAG_COMMON = /* glsl */ `
in vec3 vWorld;
in vec3 vNrm;
in vec4 vInfo;
in float vViewZ;
in vec3 vXZ0;
in vec4 vCrest;

uniform vec4 uFilm2;   // x: radiance gain of the sun lines mirrored by swash-film ripple crests
uniform float uPxAng;  // angular size of a pixel (rad), explore mode
uniform vec4 uPerf;    // (profiling) x: flat output, y: no foam, z: no crest optics, w: no ripples
mat2 rot2(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

// multi-octave capillary / wind ripple slopes, faded by pixel footprint (swash zone)
vec2 rippleGrad(vec2 p, float t, float dist, float strength) {
  vec2 g = vec2(0.0);
#ifdef OPT_EXPLORE
  float px = max(dist * uPxAng, 0.5 * max(length(fwidth(p)), 1e-5));   // pixel footprint (major axis at grazing incidence)
#else
  float px = dist * 0.00065;
#endif
  float L = 1.6;
  for (int i = 0; i < 6 + min(uEvtCount, 0); i++) {   // dynamic bound: not unrolled by FXC
    float keep = smoothstep(px * 3.0, px * 9.0, L);
    if (keep > 0.0) {
      vec2 q = rot2(float(i) * 1.93) * p / L;
      float c = sqrt(G_ACC * L / 6.2831);
      q += rot2(float(i) * 1.93) * uWind * t * c / L * 0.7;
      vec3 n = gnoised(q + float(i) * 17.3);
      g += n.yz * 0.075 * keep;
    }
    L *= 0.48;
  }
  return g * strength;
}

// Capillary ripples on a thin swash film (swash analysis; film detail at t = 6.6 s): a network of
// sharp ripple crests in 5-15 cm cells, drawn out along-shore, between which the film is nearly
// flat. Crest-sharpened (ridged) gradient noise: the slope is concentrated on thin lines along the
// zero set of the noise, which catch the bright low sky (thin bright curvy lines), while the flat
// film between them keeps a low, near-normal Fresnel (the film reads grey-neutral, not blue).
// px: pixel footprint (m). Returns the surface gradient (xy) and the ridge proximity (z): a crest
// narrower than a pixel is a small cylindrical mirror whose unresolved range of slopes catches
// the bright low sky and the sun whatever its orientation (the thin bright curvy lines).
vec3 filmRipple(vec2 p, float t, float px, out vec2 ridgeDir) {
#ifdef OPT_EXPLORE
  // Sub-millimetre capillary/gravity waves on a draining sheet. Gentle,
  // intersecting travelling ripples replace a ridged cellular network whose
  // added light made clear shallow water look like luminous soap bubbles.
  vec2 d0=normalize(vec2(.18,1.)),d1=normalize(vec2(-.43,.90)),d2=normalize(vec2(.94,.34));
  float warp=.42*gnoise(p*1.7);
  float a=dot(p,d0)*44.9-t*14.8+warp;
  float b=dot(p,d1)*89.8-t*27.1+.6*gnoise(p*2.3+7.1);
  float c=dot(p,d2)*157.1-t*43.7-warp;
  vec2 g=d0*(.030*sin(a))*(1.-smoothstep(.006,.024,px))
    +d1*(.018*sin(b))*(1.-smoothstep(.003,.012,px))
    +d2*(.009*sin(c))*(1.-smoothstep(.0018,.007,px));
  ridgeDir=d0;
  // Reflection and sunlight are already evaluated from these normals by the
  // water BRDF. No additional bright line is painted onto every ripple crest.
  return vec3(g,0.);
#else
  vec2 g = vec2(0.0);
  float ridge = 0.0;
  ridgeDir = vec2(0.0, 1.0);
  float lam = 0.11, h = 0.0013;
  for (int i = 0; i < 2 + min(uEvtCount, 0); i++) {   // dynamic bound: not unrolled by FXC
    float keep = 1.0 - smoothstep(lam * 0.12, lam * 0.3, px);
    if (keep > 0.0) {
      vec2 q = vec2(p.x * 0.85, p.y) / lam + vec2(0.13, -0.21) * t + float(i) * 7.31;
      vec3 n = gnoised(q + 0.35 * vec2(sin(t * 0.9 + q.y * 0.7), cos(t * 0.7 + q.x * 0.6)));
      float a = 1.0 - min(abs(n.x), 1.0);
      g += keep * (-3.0 * a * a * sign(n.x)) * (n.yz * vec2(0.85, 1.0)) * (h / lam);
      float rk = keep * smoothstep(0.9, 0.985, a) * (i == 0 ? 1.0 : 0.6);
      if (rk > ridge) { ridge = rk; ridgeDir = normalize(n.yz * vec2(0.85, 1.0) + 1e-5); }
    }
    lam *= 0.52; h *= 0.5;
  }
  return vec3(g, ridge);
#endif
}

`;

const FRAG = /* glsl */ `
void main() {
  vec3 P = vWorld;
  float B = hydraulicBedHeight(P.xz);
  float depth = P.y - B;
#ifdef OPT_EXPLORE
  // (footprint-filtered: at grazing incidence along the beach the fine octave aliases into dots)
  float fpE = length(fwidth(P.xz));
  float edgeN = vnoise(P.xz * 41.0) * 0.0016 * (1.0 - smoothstep(0.012, 0.03, fpE)) + vnoise(P.xz * 7.0) * 0.0018 * (1.0 - smoothstep(0.07, 0.2, fpE));
#else
  float edgeN = vnoise(P.xz * 41.0) * 0.0016 + vnoise(P.xz * 7.0) * 0.0018;
#endif
#ifdef OPT_EXPLORE
  float edge=0.0009+edgeN;
  float edgeCoverage=smoothstep(max(.00015,edge-.0007),edge+max(.0025,fwidth(depth)),depth);
  if(edgeCoverage<.001)discard;
#else
  if (depth < 0.0009 + edgeN) discard;
#endif
  if (uPerf.x > 0.5) { gl_FragColor = vec4(0.1, 0.3, 0.3, 1.0); return; }

  float dist = vViewZ;
  vec3 V = normalize(cameraPosition - P);
  float pxFoot = max(length(fwidth(P.xz)), 1e-4);   // pixel footprint (m)
  vec2 flowUV = P.xz; float turb = 0.0;
  vec4 foam = vec4(0.0);
  // foam only exists inside the shallow-water domain or on breaking crests
  if ((vInfo.z > 0.0 || vInfo.x > 0.0) && uPerf.y < 0.5) foam = surfaceFoam(P, uTime, vInfo.x, flowUV, turb);
  turb = clamp(turb, 0.0, 2.0);   // (guard: the solver's turbulence channel can overshoot)

  // backlit wave faces: crest geometry per pixel over the wave zone (steep faces span few rows;
  // the trough ahead of a face is flat but lies in the crest's shadow)
  vec4 crest = vec4(0.0);
  vec3 faceNormal=vec3(0.0,1.0,0.0);
  float keep = 1.0, hAbove = 0.0, crestProx = 0.0;
  if (vCrest.y > 0.001 && uDbg.x > -0.5 && uPerf.z < 0.5) {
    vec3 aux;
    crest = crestGeometry(vXZ0.xy, uTime, vCrest.w, aux, faceNormal);
    float gate=smoothstep(.001,.20,vCrest.y);
    crest.w*=gate;crest.z*=gate;aux.x*=gate;aux.z*=gate;aux.y=mix(1.0,aux.y,gate);
    float faceSlope = max(vCrest.z, aux.z);
    hAbove = aux.x;
    // the crest line of a shoaling swell (short waves compressed and steepened there): glints
    crestProx = crest.w * exp(-crest.x * crest.x / 0.05) * smoothstep(0.03, 0.12, P.y);
    // camera-facing wave face weight: any breaker front face steeper than ~3-5 degrees (the
    // forward-scattered glow is what makes the approaching swell a luminous band)
    crest.w *= smoothstep(uFaceSlope.x, uFaceSlope.y, faceSlope);
    // a gentle swell face is lit nearly evenly from crest to base; the darkening toward the base
    // grows as the face steepens under its crest (t = 1.5 -> 2.0 -> 3.05 s in the clip)
    crest.x *= mix(0.55, 1.0, smoothstep(0.2, 0.8, faceSlope));
    // the trough just ahead of a face sits low, under the face (drawn-down, sediment-stirred
    // water, darker than the open water: the dark toe band) even before the crest overhangs
    // Only the sampled sun-ray occlusion darkens a trough; crest proximity
    // alone is not a physical shadow and caused the hard side-view band.
    // the face of a shoaling wave is stretched and renewed as it stands up: it is clean glassy
    // water even where it runs over the previous wave's lace (the shallow-water foam and milk
    // sampled at xz would also smear down a steep face into vertical streaks)
    keep = aux.y;
    float sweFoam = 1.0 - (1.0 - foam.x) / max(1.0 - foam.w, 1e-3);   // shallow-water part (without the crest foam)
    foam.x = 1.0 - (1.0 - clamp(sweFoam, 0.0, 1.0) * keep) * (1.0 - foam.w);
    foam.y *= keep;
    turb *= keep;
  }

  // fine-scale normals: per-fragment SWE gradient + wind chop (FFT, LEAN) + swash ripples
  vec3 N = normalize(vNrm);
  float faceBlend=smoothstep(.05,.18,vCrest.w)*smoothstep(.12,.7,crest.y)*smoothstep(.04,.20,abs(faceNormal.z));
  faceBlend*=smoothstep(.001,.20,vCrest.y);
  N=normalize(mix(N,faceNormal,faceBlend));
  float wIn = sweInside(P.xz);
  vec2 g = vec2(0.0);
#ifdef OPT_EXPLORE
  // shallow-water window + far-field swash beyond it (swashfar.js)
  float wFar = swashFarWeight(P.xz);
  vec4 farS = vec4(0.0);
  if (wIn > 0.0 || wFar > 0.0) g += clampLen(0.35 * sweGradient(P.xz, 1.0, farS), uSweSlope.y) * smoothstep(0.3, 0.7, vNrm.y);
#else
  if (wIn > 0.0) {
    vec2 uv = sweUV(P.xz); vec2 e = uSweTexel;
    float wE = textureLod(uSweView, uv + vec2(e.x, 0.0), 0.0).w, wW = textureLod(uSweView, uv - vec2(e.x, 0.0), 0.0).w;
    float wN = textureLod(uSweView, uv + vec2(0.0, e.y), 0.0).w, wS = textureLod(uSweView, uv - vec2(0.0, e.y), 0.0).w;
    vec2 gs = vec2(wE - wW, wN - wS) / (2.0 * e * uSweDom.zw);
    // (sampled at xz: on a steep breaker face it would smear into vertical streaks)
    // (clamped: over a thin film the solver's surface follows the pebble-scale bed and its slope is
    //  noise; unclamped it tilts the normals toward grazing and the film mirrors the blue sky)
    g += clampLen(wIn * 0.35 * gs, uSweSlope.y) * smoothstep(0.3, 0.7, vNrm.y);
  }
#endif
  float chopW = vInfo.w;
  float paw = vXZ0.z;   // cat's-paw factor (per vertex, band-limited by the vertex spacing)
  vec4 lean = oceanSlopeLEAN(vXZ0.xy, paw, chopW, uSeaTune.y);
  g += lean.xy;
  vec2 sig2 = lean.zw;
  // capillary ripples on the thin swash / shallow water, flow-advected
  float shallowCalm = smoothstep(0.02, 0.5, depth);
  float rw = 1.0 - chopW;
  // thin swash film: crest-sharpened ripple network instead of the turbulent ripple spectrum
  float swashCoverage=wIn;
#ifdef OPT_EXPLORE
  swashCoverage=max(wIn,wFar);
#endif
  float film = swashCoverage * (1.0 - smoothstep(0.04, 0.14, depth)) * (1.0 - smoothstep(0.3, 0.8, foam.y));
  float rippleStrength = (0.7 * mix(0.35, 1.0, shallowCalm) + turb * 0.6) * rw * (1.0 - film) * smoothstep(0.3, 0.7, vNrm.y);
  vec2 rp = mix(P.xz, flowUV, swashCoverage * (1.0 - shallowCalm * 0.5));
  if (rippleStrength > 0.0 && uPerf.w < 0.5) g += rippleGrad(rp, uTime, dist, rippleStrength);
  float ridge = 0.0; vec2 ridgeDir = vec2(0.0, 1.0);
  if (film > 0.0) { vec3 fr = filmRipple(rp, uTime, pxFoot, ridgeDir);
#ifdef OPT_EXPLORE
    fr*=smoothstep(.0005,.006,depth);
#endif
    g += film * uFilm.x * fr.xy; ridge = film * fr.z; }
  sig2 += vec2(0.0025 + turb * 0.01 * (1.0 - 0.8 * film)) * rw + uFilm.w * ridge;
  N = normalize(N - vec3(g.x, 0.0, g.y));

  WIn s;
  s.P = P; s.N = N; s.V = V; s.viewZ = dist; s.thickHint = -1.0;
  // (the film between its ripple crests is glassy: bed-shear turbulence roughens deeper water)
  s.foam = foam.x; s.foamDense = foam.y; s.turb = turb * (1.0 - 0.8 * film); s.film = film;
  s.crest = clamp(max(vInfo.x * 1.5 + smoothstep(0.1, 0.6, P.y) * 0.6, crestProx), 0.0, 1.0);
  s.rough = 0.05;
  vec4 fmS = wIn * textureLod(uSweFoam, sweUV(P.xz), 0.0);
#ifdef OPT_EXPLORE
  if (wFar > 0.0) fmS = mix(fmS, swashFarFoam(P.xz, uTime), wFar);
#endif
  // milk / aeration of the bore and backwash; none on a rising face, and the backwash dives
  // under the next wave: the trough just ahead of a face (a crest rising within ~1 m) is clear
  float clearW = keep;
  // bubbles only stay up in shallow water; in deeper water (and in the trough the backwash
  // dives into) the milk channel is suspended fines: murk
  float deepW = smoothstep(0.2, 0.5, depth);
  s.milk = fmS.a * clearW * (1.0 - deepW);
  // (in the break zone the trough ahead of a face - where the backwash meets the next wave - is
  // always stirred: dark olive-teal)
  s.murk = fmS.a * deepW * clearW;
  // silt: a fast backwash / bore over the pebbles keeps fine sediment in suspension (olive-grey,
  // hides the bed and its caustics); fades as the flow slows
  float spd = length(textureLod(uSweView, sweUV(P.xz), 0.0).yz) * wIn;
#ifdef OPT_EXPLORE
  if (wFar > 0.0) spd = mix(spd, length(farS.yz), wFar);
#endif
  s.silt = smoothstep(0.35, 1.3, spd) * smoothstep(0.015, 0.06, depth) * keep;
#ifdef OPT_EXPLORE
  s.swash = max(wIn, wFar) * (1.0 - smoothstep(0.06, 0.25, depth)) * clearW * (1.0 - crest.w);
#else
  s.swash = wIn * (1.0 - smoothstep(0.06, 0.25, depth)) * clearW * (1.0 - crest.w);
#endif
  // aeration of the churning water column: blanket whitewater + the strongest bore turbulence
  // (the bed-shear turbulence of a thin sheet or of the backwash does not entrain air)
  s.aer = (smoothstep(0.05, 0.8, fmS.r) * 0.7 + smoothstep(0.8, 2.0, fmS.b) * 0.35 * smoothstep(0.06, 0.2, depth)) * clearW;
  s.under = smoothstep(0.2, 0.9, fmS.r) * clearW;
#ifdef OPT_EXPLORE
  // A millimetre-thin sheet cannot hold the same suspended bubble mass as a
  // bore. Let the photographic wet sand show through as the film drains.
  float airDepth=smoothstep(.002,.035,depth);
  s.under*=airDepth;s.aer*=airDepth;
  s.foam*=smoothstep(.0015,.010,depth);
#endif
  // grazing-incidence roughness, patchy with the cat's paws (rougher patches read darker)
  float xr = uSeaTune.x * mix(1.0, clamp(paw, 0.3, 1.8), uSeaTune.z);
  // pixel-scale facet noise tilts the visible facet distribution (dark dashes / bright crests)
  // (only where it matters: grazing incidence, far out)
  // The distribution is skewed: ~15-20 % of the far-sea pixels are dominated by steep camera-
  // facing ripple faces (dark dashes), more inside cat's paws / on the left of the view.
  // The dark dashes are sparse (a few % of the far sea) and gather in the rough cat's-paw lanes.
  float ftilt = 0.0;
  if (V.y < uSeaTune2.z && P.z < -9.0) {
    float fnz = clamp(facetNoise(vXZ0.xy, uTime), -2.5, 2.5);
    float thr = uSeaTune4.x - 0.45 * clamp(paw - 1.0, -1.0, 1.0);
    ftilt = (uSeaTune.w * 0.6 * fnz + uSeaTune4.y * smoothstep(thr, thr + 0.4, fnz))*(1.0-smoothstep(-22.0,-9.0,P.z));
  }
  vec3 col = shadeWaterFull(s, gl_FragCoord.xy, sig2, xr, crest, ftilt);
  // a film ripple crest is a small cylindrical mirror: its normals sweep across the ridge, so it
  // mirrors the sun along a thin bright line wherever the mirror slope lies on that sweep
  if (ridge > 0.0) {
    vec3 Hs = normalize(V + uSunDir);
    vec2 hs = -Hs.xz / max(Hs.y, 0.2);                   // facet slope that mirrors the sun
    float along = dot(hs, ridgeDir), across = abs(hs.x * ridgeDir.y - hs.y * ridgeDir.x);
    // (the crests' own small-scale wiggles spread the mirror condition: every crest glints a bit)
    float on = (0.35 + 0.65 * exp(-across * across / 0.03)) * (1.0 - smoothstep(0.6, 1.0, abs(along)));
    col = mix(col, col + uSunColor * uFilm2.x, ridge * on * (1.0 - clamp(s.foam, 0.0, 1.0)));
#ifdef OPT_EXPLORE
    // looking down into the sun's mirror direction (the player can): the crests whose normal sweep
    // holds the mirror slope reflect the sun disc itself - dazzling broken lines (clipped), the
    // glitter of a rippled film; nothing of it survives at the clip camera's angles
    float mir = exp(-along * along / 0.02) * exp(-across * across / 0.012);
    float brk = smoothstep(0.45, 0.8, vnoise(rp * 55.0 + vec2(uTime * 2.3, -uTime * 1.7)));
    col += uSunColor * uFilm2.y * mir * brk * ridge * (1.0 - clamp(s.foam, 0.0, 1.0));
#endif
  }
  if (uDbg.x > 20.5) col = vec3((uDbg.x < 21.5 ? film : uDbg.x < 22.5 ? turb * 0.5 : uDbg.x < 23.5 ? depth * 5.0 : length(g)) * uDbg.y);
#ifdef OPT_EXPLORE
  // Opaque water draw, continuous subpixel shoreline coverage. Blend with the
  // already rendered beach rather than a noisy alpha test or a hard film edge.
  col=mix(textureLod(uOpaqueColor,gl_FragCoord.xy/uResolution, 0.0).rgb,col,edgeCoverage);
#endif
  gl_FragColor = vec4(col, 1.0);
}`;

const FRAG_BACK = /* glsl */ `
void main() {
  vec3 P = vWorld;
  float depth = P.y - hydraulicBedHeight(P.xz);
  if (depth < 0.0012) discard;
  gl_FragColor = vec4(1.0 / vViewZ, 0.0, 0.0, 1.0);
}`;

// Calibrated optical constants (pre-grade scene-linear, see glsl/water.js WATER_SHADE).
// Matched band by band to the clip (correct BT.709 decode) over t = 0.3 .. 7 s.
export const OPTICS_PARAMS = {
  deep: [0.005, 0.145, 0.168],     // upwelling radiance of optically deep water
  atten: [0.36, 0.065, 0.062],     // extinction along the view path, clear water (1/m)
  bed: [0.64, 0.74, 0.71],         // pale pebble / sand terrace, wet
  bedLR: 0.1,                      // ... paler on the left of the view, darker on the right
  turbid: [0.02, 0.30, 0.31],      // in-scatter radiance of the turbid break-zone water
  turbidAtten: [1.0, 0.55, 0.55],  // its extinction (1/m)
  crestShade: [0.55, 0.62, 0.58, 2.0], // radiance factor in the full shadow of a crest (dark toe band); face-water extra extinction (1/m)
  faceGlow: [0.20, 0.45, 0.45, 0.0],   // in-scatter under the crest top of a steep, peaked face about to throw
  faceThick: [0.07, 0.32, 0.34, 0.0],  // ... under the thick crest of a shoaling swell
  faceSig: [0.65, 0.52, 0.62, 0.12],   // k of the (1 - k s²) darkening crest -> base; occlusion depth of the full toe shadow (m)
  faceSlope: [0.04, 0.25],         // camera-facing slope where the face glow starts / is full
  aerate: [0.8, 0.7, 3.0, 0.0],    // aeration -> milkiness, SWE milk gain, extra extinction (1/m)
  murk: [0.06, 0.15, 0.15, 1.5],   // radiance of sediment-laden deeper water (trough ahead of a face); gain
  silt: [0.50, 0.52, 0.46, 0.75],  // albedo of silt-laden fast bore / backwash water (olive-grey); gain
  film: [1.3, 0.08, 0.35, 0.0],    // swash-film ripple gain; glint slope std of the glassy film; reflection cut; unresolved slope variance on ripple crests
  sweSlope: [0.6, 0.3],            // max shallow-water surface slope in the vertex / per-pixel shading normal
  swashRefl: [0.6, 0.35],          // shallow swash water: sky-reflection desaturation, dimming
  foamShade: [0.13, 0.17, 0.15, 0.75],   // turbid water under a foam blanket (radiance), strength
  toe: [0.15, 0.20, 0.19, 0.9],    // backwash toe water (radiance: dark grey-teal); turbulence where it starts
  filmLine: 0.028,                  // radiance gain (x sun) of the sun lines mirrored by film ripple crests
  filmGlare: 1.1,                  // (explore) ... where they mirror the sun disc itself
  glintPsf: [0.75, 8.0, 2.0, 0.15],  // glint PSF sigma (px at 1080), peak cap of one glint, extra glint slope std on breaker crests (rel), glint density on the glassy face below (rel)
  glow: [0.23, 0.50, 0.48],        // radiance of sunlit, forward-scattering thin crest / lip water
  milk: [0.66, 0.66, 0.62],        // albedo of aerated / sediment-laden water (neutral-warm: fines + bubbles)
  milkBase: 0.09,                  // baseline milkiness of thin swash films (fines)
  micro: 0.03,                     // unresolved capillary slope std (beyond the FFT)
  glint: 12.0,                     // glint count multiplier (each is clipped by the sensor anyway)
  crestGlow: 1.0,                  // gain of the backlit crest / lip glow
  skyTrans: 0.4,                   // sky seen through very thin lips
  // FFT cascade gains. The clip's far sea has no wind-sea peak but a broad continuum whose
  // energy-containing visible scales are 1.5-5 m at D = 35-60 m (far_sea §6): the long cascade is
  // boosted well above JONSWAP U10 = 3 m/s (whose 2 m peak alone reads as pixel-row noise).
  amp: [3.0, 1.3, 1.14],
  paw: 1.0,                        // cat's-paw modulation strength
  // far-sea reflection statistics
  grazeRough: 0.1,                 // extra view-plane slope std at grazing incidence
  slopeLodBias: 0.0,               // LEAN texture LOD bias (negative values alias into 1-px rows)
  pawRough: 1.0,                   // cat's-paw modulation of that roughness
  facetNoise: 0.075,                // pixel-scale facet tilt noise (bright / dark facets)
  // its cells, in pixels of a 1080-wide output: ~2.6 x 11 px streaks, matching the clip's far-sea texture at
  // 1080 (luma high-pass autocorrelation 0.4-0.5 at 1 px vertical, 0.5-0.7 at 3 px horizontal;
  // 1-px cells aliased into row-wise hatching)
  facetCell: [2.6, 11.0],
  dashes: [1.3, 0.25],             // dark dashes: threshold (sigma of the facet noise; sparse, 3-5 %) and tilt
  dispFade: [15.0, 35.0],          // distance range (m) over which the chop displacement fades out
  cascadeBias: [0.0, 1.15, 1.35],    // extra slope-texture LOD bias per cascade (shading)
  nearGain0: 0.4,                  // long-cascade gain inshore of D = 15 m relative to amp[0]
  horizonSkew: [0.2, 0.03],        // extra visible-facet skew right under the horizon; eye-ray sin(elevation) range (the last rows are dark)
  shadowSlope: 0.18,               // sea slope std shadowing low reflected rays (Smith)
  grazeRange: [0.03, 0.14],        // eye-ray sin(elevation) over which "grazing" fades out
  glintCap: 9.0,                  // clipped radiance of a glint pixel
  glintSlope: [0.17, 0.2, 0.12],   // min / max capillary facet slope std for glints; far-field std
  skew: 0.2,                       // visible-facet skew toward the eye at grazing incidence
};
export class WaterSurface {
  constructor(shared, { cols = 360, swell = null } = {}) {
    // Keep resolved wind ripples clearer and reduce the broad veil over clear
    // water. Foam coverage, turbulence and the footprint filters stay intact.
    const O = CONFIG.explore ? { ...OPTICS_PARAMS,
      micro: 0.022, grazeRough: 0.075, cascadeBias: [0.0, 0.75, 1.0],
      film: [1.3, 0.08, 0.20, 0.0], swashRefl: [0.42, 0.20],
    } : OPTICS_PARAMS;
    this.ocean = new OceanFFT();
    this.swell=swell;
    // optics uniforms: added to the shared set here, before the lip ribbons copy it
    Object.assign(shared, {
      uSurfaceProbe:{value:null},uProbeOrigin:{value:new THREE.Vector2()},uUnderwaterOn:{value:0},uSandPhoto:{value:null},
      uOceanD: { value: this.ocean.disp() },
      uOceanS: { value: this.ocean.slope() },
      uOceanL: { value: new THREE.Vector3(...OCEAN.L) },
      uOceanAmp: { value: new THREE.Vector4(...O.amp, O.paw) },
      uSkyTex: { value: skyTexture(THREE) },
      uBedAlbedo: { value: new THREE.Vector3(...O.bed) },
      uBedAlbedoLR: { value: O.bedLR },
      uTurbidColor: { value: new THREE.Vector4(...O.turbid, 1.0) },
      uTurbidAtten: { value: new THREE.Vector3(...O.turbidAtten) },
      uCrestShade: { value: new THREE.Vector4(...O.crestShade) },
      uFaceGlow: { value: new THREE.Vector4(...O.faceGlow) },
      uFaceThick: { value: new THREE.Vector4(...O.faceThick) },
      uFaceSig: { value: new THREE.Vector4(...O.faceSig) },
      uFaceSlope: { value: new THREE.Vector2(...O.faceSlope) },
      uAerate: { value: new THREE.Vector4(...O.aerate) },
      uMurk: { value: new THREE.Vector4(...O.murk) },
      uSilt: { value: new THREE.Vector4(...O.silt) },
      uFilm: { value: new THREE.Vector4(...O.film) },
      uFilm2: { value: new THREE.Vector4(O.filmLine, O.filmGlare, 0, 0) },
      uPxAng: { value: 1 / 974 },
      uSweSlope: { value: new THREE.Vector2(...O.sweSlope) },
      uSwashRefl: { value: new THREE.Vector2(...O.swashRefl) },
      uFoamShade: { value: new THREE.Vector4(...O.foamShade) },
      uToe: { value: new THREE.Vector4(...O.toe) },
      uWaterLookup: { value: waterLookups().texture },
      uPerf: { value: new THREE.Vector4(0, 0, 0, 0) },
      uGlint: { value: new THREE.Vector4(...O.glintPsf) },
      uDbg: { value: new THREE.Vector4(0, 1, 0, 0) },
      uMilkBase: { value: O.milkBase },
      uOptics: { value: new THREE.Vector4(O.micro, O.glint, O.crestGlow, O.skyTrans) },
      uSeaTune: { value: new THREE.Vector4(O.grazeRough, O.slopeLodBias, O.pawRough, O.facetNoise) },
      uSeaTune2: { value: new THREE.Vector4(O.shadowSlope, ...O.grazeRange, O.glintCap) },
      uSeaTune3: { value: new THREE.Vector4(O.glintSlope[0], O.glintSlope[1], O.skew, O.glintSlope[2]) },
      uSeaTune4: { value: new THREE.Vector4(...O.dashes, ...O.dispFade) },
      uSeaTune5: { value: new THREE.Vector4(...O.cascadeBias, O.nearGain0) },
      uSeaTune6: { value: new THREE.Vector4(...O.horizonSkew, ...O.facetCell) },   // (zw: scaled to the output each frame)
    });
    shared.uDeepColor.value.set(...O.deep);
    shared.uWaterAtten.value.set(...O.atten);
    shared.uGlowTint.value.set(...O.glow);
    shared.uMilkColor.value.set(...O.milk);

    if (CONFIG.explore) {
      this.ex = new ExploreWaterMesh();
      Object.assign(shared, this.ex.uniforms);
    } else this.geometry = buildGeometry(cols);
    const prelude = WATER_PRELUDE();
    this.material = new THREE.ShaderMaterial({
      defines: { MAIN_PASS: 1 },
      uniforms: shared,
      vertexShader: prelude + VERT,
      fragmentShader: prelude + CHOP_FRAG + WATER_SHADE + FOAM + FRAG_COMMON + FRAG,
      side: THREE.FrontSide,
    });
    this.backMaterial = new THREE.ShaderMaterial({
      uniforms: shared,
      vertexShader: prelude + VERT,
      fragmentShader: prelude + FOAM + FRAG_COMMON + FRAG_BACK,
      side: THREE.BackSide,
    });
    this.underMaterial=new THREE.ShaderMaterial({uniforms:shared,vertexShader:prelude+VERT,
      fragmentShader:'#define UNDERWATER_PASS 1\n'+prelude+CHOP_FRAG+SKY_MODEL+FOAM+FRAG_COMMON+WATERLINE+CAUSTICS+
        'uniform sampler2D uOpaqueColor,uOpaqueDepth;\n'+UNDERSIDE,side:THREE.BackSide});
    this.underMesh=new THREE.Group();
    if (CONFIG.explore) { this._buildExplore(shared); return; }
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.backMesh = new THREE.Mesh(this.geometry, this.backMaterial);
    this.backMesh.frustumCulled = false;
    // The back-face pass is the first water draw of every frame: evolve the FFT sea there
    // (nested render-to-texture, the same pattern as three's Reflector).
    this.backMesh.renderOrder = -100;
    this.backMesh.onBeforeRender = (renderer) => { this.ocean.update(renderer, shared.uTime.value); this._perFrame(shared); };
  }

  // per-frame optics constants that depend on the output size
  _perFrame(shared) {
    const r = shared.uResolution.value, sres = Math.min(r.x, r.y) / 1080;
    const fc = this.facetCell || OPTICS_PARAMS.facetCell;   // (overridable for tuning)
    shared.uSeaTune6.value.z = fc[0] * sres;
    shared.uSeaTune6.value.w = fc[1] * sres;
  }

  // Explore mode: the focus-following mesh (ExploreMesh.js) as a group of draws sharing one
  // vertex buffer; the back-face pass covers the wave band near the player only (thin crests).
  _buildExplore(shared) {
    const ex = this.ex;
    this.mesh = new THREE.Group();
    this.backMesh = new THREE.Group();
    const add = (geom, mat, grp) => { const m = new THREE.Mesh(geom, mat); m.frustumCulled = false; grp.add(m); return m; };
    ex.blockGeometries.forEach((g) => add(g, this.material, this.mesh));
    add(ex.fanGeometry, this.material, this.mesh);
    add(ex.zipGeometry, this.material, this.mesh);
    for(const child of this.mesh.children)add(child.geometry,this.underMaterial,this.underMesh);
    const backs = ex.backBlocks.map((b) => add(ex.blockGeometries[b], this.backMaterial, this.backMesh));
    backs.push(add(ex.backZipGeometry, this.backMaterial, this.backMesh));
    backs.forEach((m) => { m.renderOrder = -100; });
    // the first water draw of every frame: evolve the FFT sea and re-centre the mesh on the eye
    backs[0].renderOrder = -101;
    backs[0].onBeforeRender = (renderer, scene, camera) => {
      this.swell?.update(renderer,shared.uTime.value);
      this.ocean.update(renderer, shared.uTime.value);
      this._perFrame(shared);
      ex.update(camera.position.x, camera.position.z, camera.position.y);
      shared.uPxAng.value = 2 / (camera.projectionMatrix.elements[5] * Math.max(shared.uResolution.value.y, 1));
    };
  }
}

// Legacy: the Gerstner chop list (unused since the FFT sea; kept for API compatibility).
export function makeChop(n = 20) {
  return new Float32Array(n * 4);
}
