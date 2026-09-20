// Hash-without-Sine routines: Copyright (c) 2014 David Hoskins, MIT.
// See licenses/Hash-without-Sine-MIT.txt and THIRD_PARTY_NOTICES.md.
// Water surface: geometry evaluation (shared by the colour pass and the back-face
// thickness pass) and the optical shading model (shared by the heightfield surface and
// the plunging-lip ribbon).

// Wind sea from the GPU FFT cascades (water/OceanFFT.js). Displacement textures D_c =
// (dx, h, dz), slope textures S_c = (sx, sz, sx², sz²), all mipmapped: sampling them with
// the pixel / vertex footprint gives the resolved mean slope and the unresolved slope
// variance (LEAN mapping) so the sea never aliases toward the horizon.
import { SWELL_SAMPLE } from './swell.js';
import { CONFIG } from '../config.js';
import { SKY_MODEL } from '../sky/skyModel.js';

// Explore mode (?explore): the world must hold up around a moving camera, so everything that was
// anchored to the fixed phone camera is re-anchored to the world (or to the camera position only,
// never its orientation). Clip mode compiles the original code paths unchanged.
const XDEF = CONFIG.explore ? '#ifndef OPT_EXPLORE\n#define OPT_EXPLORE\n#endif\n' : '';

export const OCEAN = {
  N: 256,
  L: [41.0, 7.3, 1.37],         // cascade patch sizes (m), non-commensurate
  ROT: [17.0, -31.0, 43.0],     // texture-space rotation (deg): tiles never line up with image rows
};
const rot = (d) => { const a = (d * Math.PI) / 180; return [Math.cos(a), Math.sin(a)]; };
const [RC0, RC1, RC2] = OCEAN.ROT.map(rot);
const m2 = ([c, s]) => `mat2(${c.toFixed(7)}, ${s.toFixed(7)}, ${(-s).toFixed(7)}, ${c.toFixed(7)})`;

export const CHOP = XDEF + /* glsl */ `
precision highp sampler2DArray;
uniform sampler2DArray uOceanD, uOceanS;
uniform vec3 uOceanL;          // cascade patch sizes (m)
uniform vec4 uOceanAmp;        // amplitude gain per cascade (xyz), cat's-paw strength (w)
uniform vec2 uWind;            // unit wind direction (xz)
uniform vec4 uSeaTune;         // x: grazing-incidence slope std for sky reflection, y: slope LOD bias, z: cat's-paw roughness modulation, w: facet-noise strength
uniform vec4 uSeaTune2;        // x: sea slope std shadowing low reflected rays, y/z: grazing range (eye-ray sin elevation), w: glint radiance cap
uniform vec4 uSeaTune3;        // x/y: min/max slope std of the glint facets, z: visible-facet skew at grazing, w: far-field glint std
uniform vec4 uSeaTune5;        // xyz: extra slope-texture LOD bias per cascade (shading), w: long-cascade gain inshore (rel)
uniform vec4 uSeaTune6;        // x: extra visible-facet skew right under the horizon, y: over which eye-ray sin(elevation), z/w: facet-noise cell height / width (px)
uniform vec4 uSeaTune4;        // x: facet-noise threshold of the dark dashes (sigma), y: their tilt, z/w: distance range over which the chop displacement fades out (m)

// cascade texture frames: u_c = R_c xz / L_c; vectors come back with R_c^T
const mat2 OCR0 = ${m2(RC0)};
const mat2 OCR1 = ${m2(RC1)};
const mat2 OCR2 = ${m2(RC2)};
const vec3 OCC2 = vec3(${(RC0[0] ** 2).toFixed(7)}, ${(RC1[0] ** 2).toFixed(7)}, ${(RC2[0] ** 2).toFixed(7)});   // cos²
vec2 ocVar(vec2 v, float c2) { return vec2(c2 * v.x + (1.0 - c2) * v.y, (1.0 - c2) * v.x + c2 * v.y); }

// Cat's paws: slowly drifting, reforming patches of rougher short waves (far_sea.md §7):
// elongated along-shore, more on the left of the view, evolving over ~1 s.
// fp: sampling footprint (m) along z: octaves finer than the footprint are faded (they would
// alias into row-wise streaks on the coarse offshore mesh / far pixels).
float pawFbm(vec2 p, float fz, int oct) {
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 5; i++) {
    if (i >= oct) break;
    s += a * gnoise(p) * (1.0 - smoothstep(0.25, 0.5, fz));
    p = mat2(1.6, 1.2, -1.2, 1.6) * p; a *= 0.5; fz *= 2.0;
  }
  return s;
}
float catsPaw(vec2 xz, float t, float fp) {
  vec2 q = xz * vec2(0.055, 0.16);
  float a = pawFbm(q + vec2(0.021, -0.12) * t, fp * 0.16, 4);
  float b = pawFbm(q * 1.7 + vec2(-0.05, -0.2) * t + 11.3, fp * 0.27, 3);
  float f = 0.5 + 0.8 * (0.62 * a + 0.38 * b);
#ifdef OPT_EXPLORE
  // world-anchored: the clip camera's wind lanes (more paws on its left) inside its frame, a slow
  // along-shore variation of the lanes elsewhere
  float D = max(4.1 - xz.y, 1.0);
  float r = xz.x / (0.72 * D);
  float lane = clamp(0.45 * gnoise(vec2(xz.x * 0.013 + 0.37, xz.y * 0.009 - 1.3)), -0.5, 0.5);
  float u = mix(clamp(r, -0.5, 0.5), lane, smoothstep(0.6, 1.6, abs(r)));
#else
  float D = max(cameraPosition.z - xz.y, 1.0);
  float u = clamp(xz.x / (0.72 * D), -0.5, 0.5);      // ≈ image u - 0.5
#endif
  float lr = 1.0 - 1.1 * u;                            // 1.55 (left) .. 0.45 (right)
  return mix(1.0, clamp(f, 0.15, 1.6) * lr, uOceanAmp.w);
}

// The long-cascade boost belongs to the open sea (1.5-5 m waves resolved as facets far out);
// inshore of the drop-off the swell zone keeps the plain spectrum.
#ifdef OPT_EXPLORE
float oceanFarGain(vec2 xz) { return mix(uSeaTune5.w, 1.0, smoothstep(15.0, 40.0, 4.1 - xz.y)); }   // (world: the bathymetry)
// horizontal distance from the eye (clip mode: the fixed camera's depth, cameraPosition.z - z)
float viewDistH(vec2 xz) { return length(xz - cameraPosition.xz); }
#else
float oceanFarGain(vec2 xz) { return mix(uSeaTune5.w, 1.0, smoothstep(15.0, 40.0, cameraPosition.z - xz.y)); }
float viewDistH(vec2 xz) { return cameraPosition.z - xz.y; }
#endif

// Chop displacement at undisplaced position xz for a sampling footprint fp (m):
// returns (dx, h, dz); slope = mean (dh/dx, dh/dz) at that footprint.
vec3 oceanDisp(vec2 xz, float fp, float paw, out vec2 slope) {
  vec3 d = vec3(0.0); slope = vec2(0.0);
  float lf = log2(max(fp, 1e-4) * ${OCEAN.N}.0) + 0.5;
  float l0 = lf - log2(uOceanL.x), l1 = lf - log2(uOceanL.y), l2 = lf - log2(uOceanL.z);
  paw = max(paw, 0.0);
  vec3 a = uOceanAmp.xyz * vec3(oceanFarGain(xz), sqrt(paw), paw);
  vec4 D = textureLod(uOceanD, vec3(OCR0 * xz / uOceanL.x, 0.0), l0);
  vec4 S = textureLod(uOceanS, vec3(OCR0 * xz / uOceanL.x, 0.0), l0);
  d.y = D.y * a.x; d.xz = (D.xz * OCR0) * a.x;
  slope += (S.xy * OCR0) * a.x;
  if (l1 < 8.0) {
    D = textureLod(uOceanD, vec3(OCR1 * xz / uOceanL.y, 1.0), l1); S = textureLod(uOceanS, vec3(OCR1 * xz / uOceanL.y, 1.0), l1);
    float keep=1.0-smoothstep(6.0,8.0,l1);
    d.y += D.y * a.y * keep; d.xz += (D.xz * OCR1) * a.y * keep; slope += (S.xy * OCR1) * a.y * keep;
  }
  if (l2 < 8.0) {
    D = textureLod(uOceanD, vec3(OCR2 * xz / uOceanL.z, 2.0), l2); S = textureLod(uOceanS, vec3(OCR2 * xz / uOceanL.z, 2.0), l2);
    float keep=1.0-smoothstep(6.0,8.0,l2);
    d.y += D.y * a.z * keep; d.xz += (D.xz * OCR2) * a.z * keep; slope += (S.xy * OCR2) * a.z * keep;
  }
  return d;
}

// legacy entry point: (eta, d/dx, d/dz) of the chop heightfield
vec3 chopWaves(vec2 xz, float t, float lod) {
  vec2 s; vec3 d = oceanDisp(xz, lod, 1.0, s);
  return vec3(d.y, s);
}
`;

// Fragment-only part of the chop (texture LOD bias).
export const CHOP_FRAG = XDEF + /* glsl */ `
// Wind chop slope at the pixel footprint (LEAN): world mean slope (xy) and unresolved
// slope variance (zw). lodBias < 0 keeps some pixel-to-pixel facet noise (a camera pixel
// box-filters only a few facets).
// The pixel footprint on the far sea is extremely anisotropic (at 50 m a pixel covers ~3 cm
// along-shore but ~0.7 m cross-shore); beyond the hardware's 8:1 anisotropic filter the slope
// texture aliases into 1-px horizontal hatching. The footprint's minor axis is widened to keep
// the ratio <= 6 (the lost slope detail becomes LEAN variance: roughness, not noise).
vec4 oceanSlopeLEAN(vec2 xz, float paw, float w, float lodBias) {
  vec2 ex = dFdx(xz), ey = dFdy(xz);
  float lx = length(ex), ly = length(ey);
  float lmaj = max(max(lx, ly), 1e-6);
  float kb = exp2(lodBias);
  if (lx < lmaj / 6.0) ex = (lx > 1e-7 ? ex / lx : vec2(1.0, 0.0)) * (lmaj / 6.0);
  if (ly < lmaj / 6.0) ey = (ly > 1e-7 ? ey / ly : vec2(0.0, 1.0)) * (lmaj / 6.0);
  ex *= kb; ey *= kb;
  // per-cascade extra blur: at a distance the 0.3-2.5 m wind waves span only 2-4 pixel rows and
  // read as pixel noise in the clip's place (the phone resolves the longer waves as facets and
  // the shorter as roughness): their resolved slope is handed over to the LEAN variance
  vec3 kc = exp2(uSeaTune5.xyz);
  vec4 s0 = textureGrad(uOceanS, vec3(OCR0 * xz / uOceanL.x, 0.0), OCR0 * ex * kc.x / uOceanL.x, OCR0 * ey * kc.x / uOceanL.x);
  vec4 s1 = textureGrad(uOceanS, vec3(OCR1 * xz / uOceanL.y, 1.0), OCR1 * ex * kc.y / uOceanL.y, OCR1 * ey * kc.y / uOceanL.y);
  vec4 s2 = textureGrad(uOceanS, vec3(OCR2 * xz / uOceanL.z, 2.0), OCR2 * ex * kc.z / uOceanL.z, OCR2 * ey * kc.z / uOceanL.z);
  paw = max(paw, 0.0);
  vec3 a = uOceanAmp.xyz * vec3(oceanFarGain(xz), sqrt(paw), paw) * clamp(w, 0.0, 1.0);
  vec2 m = (s0.xy * OCR0) * a.x + (s1.xy * OCR1) * a.y + (s2.xy * OCR2) * a.z;
  vec2 v = ocVar(max(s0.zw - s0.xy * s0.xy, 0.0), OCC2.x) * a.x * a.x
         + ocVar(max(s1.zw - s1.xy * s1.xy, 0.0), OCC2.y) * a.y * a.y
         + ocVar(max(s2.zw - s2.xy * s2.xy, 0.0), OCC2.z) * a.z * a.z;
  return vec4(m, v);
}

// Pixel-scale facet noise. A camera pixel far out integrates only a handful of short-wave
// facets, so the fraction of steep camera-facing facets (dark) versus grazing crest facets
// (bright sky) fluctuates from pixel to pixel: the dark horizontal dashes and bright glints
// of the reference. World-anchored value noise whose cell follows the pixel footprint
// (≈3 px wide, 1 px tall -> 4:1 horizontal dashes), octave-blended (no popping), renormalised
// to unit variance, re-rolled every ~0.4 s as the short waves evolve. Returns ~N(0,1).
float facetNoise(vec2 xz, float t) {
#ifdef OPT_EXPLORE
  // anchored to the world and the eye position (never the view direction): image columns are
  // lines of constant azimuth, image rows lines of constant horizontal distance
  vec2 rel = xz - cameraPosition.xz;
  float D = max(length(rel), 0.5);
  vec2 q0 = vec2(atan(rel.x, -rel.y), -D - 0.9 * t);
  float pa = max(max(abs(dFdx(q0.x)), abs(dFdy(q0.x))), 1e-6);
  pa = min(pa, 0.01);                                              // (the azimuth wraps behind the eye)
  float pz = max(abs(dFdy(D)), abs(dFdx(D)));
#else
  // screen-aligned, world-anchored coordinates: image columns are lines of constant x / D
  // (the camera only rotates), image rows are lines of constant z
  float D = max(cameraPosition.z - xz.y, 0.5);
  vec2 q0 = vec2(xz.x / D, xz.y - 0.9 * t);
  float pa = max(max(abs(dFdx(q0.x)), abs(dFdy(q0.x))), 1e-6);   // angular pixel size (~constant)
  float pz = max(abs(dFdy(xz.y)), abs(dFdx(xz.y)));
#endif
  float lz = log2(max(pz * uSeaTune6.z, 1e-4));
  float l0 = floor(lz), f = lz - l0;
  vec2 s0 = vec2(pa * uSeaTune6.w, exp2(l0)), s1 = vec2(pa * uSeaTune6.w, exp2(l0 + 1.0));
  // (cell space sheared: the value-noise lattice would otherwise show as a mosaic of aligned
  //  rectangles; the cell boundaries now slant ~50 deg on screen while the dashes stay horizontal)
  const mat2 FSH = mat2(0.955, -0.296, 0.296, 0.955);
  float n0 = vnoise3(vec3(FSH * (q0 / s0), t * 2.5)) - 0.5;
  float n1 = vnoise3(vec3(transpose(FSH) * (q0 / s1) + vec2(17.0, 31.0), t * 2.5 + 7.3)) - 0.5;
  return (n0 * (1.0 - f) + n1 * f) / sqrt(max((1.0 - f) * (1.0 - f) + f * f, 1e-3)) * 5.5;
}
`;

// SWE view sampling helpers (domain in world xz)
export const SWE_SAMPLE = /* glsl */ `
uniform sampler2D uSweView;   // (h, u, v, w)
uniform sampler2D uSweFoam;   // (whitewater, lace, turbulence k)
uniform sampler2D uSweLace;   // advected lace offsets (A.xy, B.xy)
uniform sampler2D uSweWet;    // (film, damp)
uniform vec4 uSweDom;         // xMin, zMin, width, depth
uniform vec2 uSweTexel;
uniform vec2 uLacePhase;
vec2 sweUV(vec2 xz) { return (xz - uSweDom.xy) / uSweDom.zw; }
float sweInside(vec2 xz) {
  vec2 uv = sweUV(xz);
  vec2 e = min(uv, 1.0 - uv);
  // Keep the original 39.2 cm cross-shore fringe when the dry reserve grows.
  return smoothstep(0.0, 0.04, min(e.x, e.y * uSweDom.w / 9.8));
}
`;

export const WATER_GEOM = XDEF + /* glsl */ `
// Chop weight: short wind waves are damped in very shallow water and are gone in the swash.
float chopWeight(float depth0) { return smoothstep(0.05, 1.2, depth0); }

${SWELL_SAMPLE}
float gBrkEta;   // breaker-only elevation of the last waterParts() call

// Water surface parts at xz (undisplaced), excluding the lip ribbon.
// gMacro = gradient of breaker + SWE (no chop); chop = chop displacement (dx, h, dz) already
// weighted; gChop = chop slope; info = (crestFoam, sweDepth, sweWeight, chopWeight).
float waterParts(vec2 xz, float t, float lod, out vec2 gMacro, out vec3 chop, out vec2 gChop, out vec4 info, out float paw) {
  vec2 d = vec2(0.0); float crestFoam = 0.0;
  // anti-aliasing: the analytic swell humps are a few metres wide; where the mesh rows are
  // coarser than that (far offshore) they would alias into row-wise bands -> fade them out
  float aa = 1.0 - smoothstep(0.08, 0.35, lod);
  float eta = 0.0;
#ifdef OPT_EXPLORE
  aa=mix(aa,1.0-smoothstep(.7,3.0,lod),1.0-smoothstep(-11.0,-7.0,xz.y));
  if (aa > 0.0 && xz.y > -40.0) {   // fine shoaling profile; farther crests use the shared swell field
#endif
  eta = brkSurface(xz, t, d, crestFoam);
  float etaX = brkSurfaceOnly(xz + vec2(0.04, 0.0), t);
  d.x = (etaX - eta) / 0.04;
#ifdef OPT_EXPLORE
  }
#endif
  eta *= aa; d *= aa; crestFoam *= aa;
#ifdef OPT_EXPLORE
  float sw=1.0-smoothstep(-32.0,-22.0,xz.y);
  if(sw>0.0){
    vec3 farSwell=swellAt(xz)*(1.0-smoothstep(3.0,14.0,lod));
    float q=clamp((xz.y+32.0)/10.0,0.0,1.0);
    float slope=-.6*q*(1.0-q);
    d=mix(d,farSwell.yz,sw)+vec2(0.0,(farSwell.x-eta)*slope);
    eta=mix(eta,farSwell.x,sw);
  }
#endif
  gBrkEta = eta;
  float B = bedProfile(xz.y);
  float depth0 = max(-B, 0.0);
  float chopW = chopWeight(depth0);
  // the steep breaking face carries less of the wind chop (it is stretched and smoothed)
  chopW *= 1.0 / (1.0 + 1.5 * dot(d, d));
  paw = catsPaw(xz, t, lod);
  chop = vec3(0.0); gChop = vec2(0.0);
  // far out, a centimetre of height moves a mesh row by a pixel and wave slopes exceed the
  // grazing angle: displaced rows fold over each other (1-px horizontal hatching). There the
  // chop is carried by the shading normals alone (LEAN + visible-facet reflection model).
  float chopFar = 1.0 - smoothstep(uSeaTune4.z, uSeaTune4.w, viewDistH(xz));
#ifdef OPT_EXPLORE
  if (chopW > 0.0 && chopFar > 0.0) { chop = oceanDisp(xz, lod, paw, gChop) * chopW; gChop *= chopW; }
#else
  if (chopW > 0.0) { chop = oceanDisp(xz, lod, paw, gChop) * chopW; gChop *= chopW; }
#endif
  chop *= chopFar;
  // no horizontal chop jiggle on the sharp upper part of a breaking wave (it would saw-tooth the crest)
  chop.xz *= 1.0 - smoothstep(0.04, 0.2, eta);
  // shallow-water residual (bore, swash, backwash, set-up) from the GPU solver
  float wIn = sweInside(xz);
  vec4 sv = textureLod(uSweView, sweUV(xz), 0.0);
  float sweEta = mix(0.0, sv.w, wIn);
  float sweH = sv.x;
  // beyond the shallow-water window (explore): the analytic far-field swash (swashfar.js)
  float wF = swashFarWeight(xz);
  if (wF > 0.0) {
    vec4 sf = swashFar(xz, t);
    sweEta = mix(sweEta, sf.w, wF);
    sweH = mix(sweH, sf.x, wF);
  }
  gMacro = d;
  info = vec4(crestFoam, sweH, max(wIn, wF), chopW);
  return sweEta + eta;
}
uniform vec2 uSweSlope;   // max slope of the shallow-water surface in the vertex normal (x) / the per-pixel shading normal (y)
vec2 clampLen(vec2 g, float m) { float l = length(g); return l > m ? g * (m / l) : g; }
// Surface gradient of the shallow-water residual at xz (the SWE window and the far-field swash,
// blended as in waterParts); e0: finite-difference half step in SWE texels; far: the far-field
// state at xz (zero inside the window).
vec2 sweGradient(vec2 xz, float e0, out vec4 far) {
  vec2 g = vec2(0.0);
  far = vec4(0.0);
  float wIn = sweInside(xz);
  if (wIn > 0.0) {
    vec2 uv = sweUV(xz); vec2 e = uSweTexel * e0;
    float wE = textureLod(uSweView, uv + vec2(e.x, 0.0), 0.0).w, wW = textureLod(uSweView, uv - vec2(e.x, 0.0), 0.0).w;
    float wN = textureLod(uSweView, uv + vec2(0.0, e.y), 0.0).w, wS = textureLod(uSweView, uv - vec2(0.0, e.y), 0.0).w;
    g = wIn * vec2(wE - wW, wN - wS) / (2.0 * e * uSweDom.zw);
  }
  float wF = swashFarWeight(xz);
  if (wF > 0.0) {
    // (one call site in a loop with an opaque trip count: the far model is inlined once)
    float h = 0.03;
    vec3 w = vec3(0.0);
    for (int k = 0; k < 3 + min(uEvtCount, 0); k++) {
      vec2 p = xz + (k == 1 ? vec2(h, 0.0) : (k == 2 ? vec2(0.0, h) : vec2(0.0)));
      vec4 fk = swashFar(p, uTime);
      if (k == 0) { w.x = fk.w; far = fk; } else if (k == 1) w.y = fk.w; else w.z = fk.w;
    }
    g = mix(g, vec2(w.y - w.x, w.z - w.x) / h, wF);
  }
  return g;
}

// Water surface height and gradient at xz, excluding the lip ribbon.
// out: grad = (d/dx, d/dz); info = (crestFoam, sweDepth, sweWeight, chopWeight)
float waterHeight(vec2 xz, float t, float lod, out vec2 grad, out vec4 info) {
  vec2 gm, gc; vec3 ch; float paw;
  float y = waterParts(xz, t, lod, gm, ch, gc, info, paw);
  grad = gm + gc;
  return y + ch.y;
}

// Backlit wave geometry at xz (breaker heightfield of the dominant event):
//   x = position on the face: 0 at the crest, 1 at the trough level (face fraction below the
//       crest) - sunlight reaching the water behind the face has come down through the crest:
//       the upper face glows, the base is darker and greener,
//   y = stage of the crest: 0 thick shoaling swell (sunlight crosses a thick crest: saturated,
//       darker turquoise), 1 steep peaked crest about to throw (thin: bright pale turquoise),
//   z = sun occlusion (m): how far the sun ray from the point runs under the wave surface,
//       sampled seaward at 0.04 .. 1.1 m - the shadow of a steep crest on the trough just
//       ahead of the toe (dark toe band),
//   w = 1 when a breaker crest is within reach (else the neutral (0, 0, 0, 0)).
// aux.x: height of the crest above the point within the sampled 1.1 m (m);
// aux.y: how much of the shallow-water foam / milk survives on the face (1 = all): the face of a
//        shoaling wave is stretched and renewed as it stands up - same rule as brkFoamKeep()
//        (breaker.js), for the dominant event only.
// Evaluated per pixel over the wave zone: one event loop picks the dominant event, whose stage is
// then evaluated once; the 9 profile samples reuse it (they mirror brkProfile()).
// aux.z: camera-facing slope (-d eta / dz) of the dominant breaker's profile at this pixel (per
//        pixel: interpolated between the few rows that span a steep face, the vertex slope sags
//        into dark teeth under the crest line)
// etaR: the breaker elevation of the rendered surface at this pixel (interpolated from the vertices):
//       the face fraction, the height above and the sun occlusion are measured from it, so they
//       follow the geometry (a triangle spanning a steep face is a straight ramp; the analytic
//       profile at the interpolated z is a step there, which drew dark teeth under the crest)
vec4 crestGeometry(vec2 xz, float t, float etaR, out vec3 aux, out vec3 faceNormal) {
  aux = vec3(0.0, 1.0, 0.0); faceNormal=vec3(0.0,1.0,0.0);
  // dominant event at xz: the active wave whose crest is nearest (face, crest or trough)
  int best = -1; float bestD = 2.5;
  for (int i = 0; i < uEvtCount; i++) {  // dynamic bound: keeps D3D/FXC from unrolling
    Brk b = brkAt(i, xz.x);
    if (b.str <= 0.0 || t < b.tb - 9.0 || t > b.ti + 0.6) continue;
    float dC = abs(xz.y - brkCrestZ(b, t));
    if (dC < bestD) { bestD = dC; best = i; }
  }
  if (best < 0) return vec4(0.0);
  Brk b = brkAt(best, xz.x);
  float tn = brkTn(b, t);
  if (tn < -6.0 || tn > BRK_TEND) return vec4(0.0);    // (first knot of the stage table)
  BrkStage g = brkStage(tn);
  vec2 A = brkApex(b, g, tn);
  // brkProfile(): eta = brkShape(g, (z - A.z) / s).x * s * amp * fade(z)
  float amp = smoothstep(-23.0, -13.0, A.x) * (1.0 - smoothstep(0.6, BRK_TEND, tn)) * b.str;
  vec2 sh0 = brkShape(g, (xz.y - A.x) / b.s);
  float eta0 = sh0.x * b.s * amp * (1.0 - smoothstep(-1.0, 0.1, xz.y));
  aux.z = max(-sh0.y * amp * (1.0 - smoothstep(-1.0, 0.1, xz.y)), 0.0);
  // crest (apex) and trough heights of this event now
  float fd = 1.0 - smoothstep(-1.0, 0.1, xz.y);
  float ya = g.ya * b.s * amp * fd, yt = g.yt * b.s * amp * fd;
  eta0 = clamp(etaR, yt, ya);
  float sFace = clamp((ya - eta0) / max(ya - yt, 0.05), 0.0, 1.2);
  // A steep face can span a single mesh row. Interpolating vertex normals there
  // turns finite-difference spikes into rectangular dark streaks. Reconstruct
  // its normal at the rendered HEIGHT on the continuous front profile instead.
  if(g.st>.12){
  float xiFace=brkFrontXi(g,clamp(1.0-sFace,.001,.999));
#ifdef UNDERWATER_PASS
  float xiBack=-g.lb*acosh(sqrt(max(ya/max(eta0,.0001),1.00001)));
  // On the polygonal crest, the interpolated height can lie just below the
  // exact apex. A binary front/back choice then flips two nonzero slopes at
  // once. Resolve the narrow crest shoulder continuously in the pixel footprint.
  float shoulder=max(.012*b.s,min(.065*b.s,fwidth(xz.y)*1.5));
  float dzSlope=mix(brkShape(g,xiBack).y,brkShape(g,xiFace).y,
    smoothstep(-shoulder,shoulder,xz.y-A.x))*amp*fd;
#else
  if(xz.y<A.x) xiFace=-g.lb*acosh(sqrt(max(ya/max(eta0,.0001),1.00001)));
  float dzSlope=brkShape(g,xiFace).y*amp*fd;
#endif
  // The arrival clock has no 45 cm closing staircase. A half-metre footprint
  // preserves the broad peel/crest direction without shading each closure as a bar.
  float zLeft=0.0,zRight=0.0;
  for(int n=0;n<2+min(uEvtCount,0);n++){
    Brk nb=brkAt(best,xz.x+(float(n)-.5)*.5);
    float zc=nb.zI-b.c*(nb.ti-nb.D-b.ti);
    if(n==0)zLeft=zc;else zRight=zc;
  }
  float crestTilt=clamp((zRight-zLeft)/.5,-.6,.6);
  faceNormal=normalize(vec3(crestTilt*dzSlope,1.0,-dzSlope));
  }
  float hMax = eta0, occ = 0.0;
  float rise = uSunDir.y / max(-uSunDir.z, 0.05);      // sun-ray rise per metre seaward (≈2.9)
  for (int k = 1; k <= 8 + min(uEvtCount, 0); k++) {   // dynamic bound: not unrolled by FXC
    float dz = 0.04 * pow(float(k), 1.6);
    float zk = xz.y - dz;
    float y = brkShape(g, (zk - A.x) / b.s).x * b.s * amp * (1.0 - smoothstep(-1.0, 0.1, zk));
    hMax = max(hMax, y);
    occ = max(occ, y - eta0 - rise * dz);
  }
  // Continuously retire the optical neighbourhood. A binary 2.5 m event
  // selection drew a broad diagonal on rear/side views.
  float support=1.0-smoothstep(1.45,2.48,bestD);
  aux.x = (hMax - eta0)*support;
  aux.z *= support;
  // foam keep (as brkFoamKeep, from the crest's back 0.3 m down the whole front face) - and the
  // drawdown ahead of an approaching face pulls the trough's surface (and its lace) down into
  // the face: the last ~0.5 m ahead of the toe is clean too, until the lip lands there
  float xi = (xz.y - A.x) / b.s;
  if (xz.y > -9.0 && xz.y < 0.5 && tn > -1.6 && tn < 0.25 && xi > -0.3) {
    float ahead = (xi - g.lf) * b.s;                    // m ahead of the toe (< 0 on the face)
    float fa = smoothstep(-0.1, 0.1, ahead);
    float live = mix(1.0 - smoothstep(-0.05, 0.25, tn), 0.85 * (1.0 - smoothstep(-0.45, -0.15, tn)), fa);
    aux.y = 1.0 - smoothstep(-0.3, 0.0, xi) * smoothstep(0.55, 0.1, ahead) * smoothstep(-1.6, -0.9, tn) * live;
  }
  aux.y=mix(1.0,aux.y,support);
  return vec4(sFace, g.st, occ*support, support);
}
`;

// Optical shading of water. Inputs are gathered in a struct so the heightfield and the lip
// ribbon produce identical-looking water. All colours are pre-grade scene-linear (the post
// applies the phone's x1.35 saturation and soft shoulder). Physical model:
//   reflection: sky integrated over the *visible* facet-slope distribution in the view plane
//               (resolved mean normal + unresolved LEAN variance + micro roughness), so the
//               far sea gets its low effective Fresnel and the up-sky colour of camera-facing
//               facets, and nothing aliases;
//   body:       the refracted view ray is traced to the bed (analytic bathymetry). Near the
//               swash the pebbles are read back from the opaque pass at the refracted point;
//               elsewhere a pale pebble bottom lit through the water column is used.
//               Beer-Lambert along the true refracted path + in-scatter toward the deep
//               water colour. Over the pale terrace (≈2 m) this is the luminous turquoise
//               band; beyond the drop-off it falls to the dark teal of deep water.
//   faces:      the high sun is behind the waves. Behind a face, the in-scatter is lit through the
//               crest (path ~ crest height above): thin crest tops glow pale turquoise, bases and
//               the trough band in front stay dark teal (crest geometry from the breaker model).
//   lips:       where the ray leaves through the back of a thin sheet (back-face depth pass /
//               ribbon thickness) the water glows with forward-scattered sunlight.
//   glints:     stochastic, energy-conserving sun glints from ~1 cm² capillary facets.
export const WATER_SHADE = XDEF + (CONFIG.explore ? SKY_MODEL : '') + /* glsl */ `
uniform sampler2D uOpaqueColor;   // HDR scene without water (sky, beach, seabed)
uniform sampler2D uOpaqueDepth;   // its depth buffer
uniform sampler2D uBackDepth;     // 1 / view depth of the nearest back-facing water surface (0 = none)
uniform vec2 uResolution;
uniform vec3 uDeepColor;          // upwelling radiance of optically deep water (in-scatter limit)
uniform vec3 uWaterAtten;         // extinction along the view path (1/m)
uniform vec3 uMilkColor;          // albedo of bubble/sediment-laden water
uniform vec3 uGlowTint;           // colour of sunlight forward-scattered through thin crests / lips
uniform vec3 uBedAlbedo;          // pale pebble bottom beyond the swash (wet)
uniform float uBedAlbedoLR;       // its relative left (+) / right (-) variation across the view
uniform vec4 uTurbidColor;        // rgb: in-scatter radiance of the turbid break-zone water, w: strength
uniform vec3 uTurbidAtten;        // its extinction (1/m): fine sediment kept in suspension by the breakers
uniform vec4 uCrestShade;         // xyz: radiance factor in the full shadow of a crest (toe band), w: extra extinction of face water (1/m)
uniform vec4 uFaceGlow;           // rgb: in-scatter right under the crest top of a steep, peaked face about to throw, w: unused
uniform vec4 uFaceThick;          // rgb: the same under the thick crest of a shoaling swell, w: unused
uniform vec4 uFaceSig;            // rgb: k of the (1 - k s²) darkening from crest to base, w: sun-occlusion depth of the full toe shadow (m)
uniform vec4 uAerate;             // x: aeration -> milkiness, y: SWE milk channel gain, z: extra extinction per unit aeration (1/m), w: unused
uniform vec4 uMurk;               // rgb: radiance of sediment-laden (murky) deeper water, w: gain
uniform vec4 uSilt;               // rgb: albedo of silt-laden bore / backwash water, w: gain
uniform vec4 uDbg;                // debug views (x selects; 0 = off)
uniform vec2 uFaceSlope;          // camera-facing slope where the face glow starts / is full
uniform vec4 uGlint;              // x: PSF sigma (px), y: peak radiance cap of one glint, z: extra glint slope std on breaker crests (rel), w: glint slope std on the glassy face below (rel)
uniform vec4 uFilm;               // x: swash-film ripple gain, y: glint slope std of the glassy film, z: film reflection cut, w: unused
uniform float uMilkBase;          // baseline milkiness of thin swash films
uniform vec4 uOptics;             // x: micro-roughness (slope std), y: glint gain, z: crest glow gain, w: sky-transmission gain
uniform vec2 uSwashRefl;          // shallow swash water: sky-reflection desaturation, dimming
uniform vec4 uFoamShade;          // rgb: radiance of the turbid water under a foam blanket, w: strength
uniform vec4 uToe;                // rgb: radiance of the backwash toe water, w: turbulence where it starts
uniform sampler2D uSkyTex;        // sky radiance vs elevation (look.js), for fast reflections
uniform mat4 projectionMatrix;

struct WIn {
  vec3 P;          // world position
  vec3 N;          // shading normal
  vec3 V;          // unit vector toward the eye
  float viewZ;     // positive view-space depth of the fragment
  float thickHint; // analytic thickness if known (lip), else -1
  float foam;      // foam coverage 0..1
  float foamDense; // dense whitewater 0..1
  float turb;      // turbulence (roughness)
  float crest;     // crest proximity (for the backlit glow)
  float rough;     // GGX roughness / sub-pixel slope spread
  float milk;      // suspended bubbles / sediment 0..1 (SWE milk channel)
  float aer;       // aeration of the churning water column (impact zone, bore) 0..1
  float murk;      // suspended fines in deeper water (after the bubbles have risen) 0..1
  float film;      // thin swash film 0..1 (glassy between resolved ripple crests)
  float silt;      // fine sediment stirred up by a fast bore / backwash 0..1
  float swash;     // shallow swash / bore / backwash water over the pebbles 0..1
  float under;     // water under a whitewater blanket (shadowed by it, stirred): seen through its holes 0..1
};

vec3 shadeFoam(vec3 N, float dense) {
  // foam is a dense scattering medium: very soft wrapped lighting and strong ambient
  float wrap = clamp((dot(N, uSunDir) + 0.8) / 1.8, 0.0, 1.0);
  // (clip blanket (211, 216, 218): a touch greener and less blue than a neutral albedo under the blue sky)
  vec3 albedo = vec3(0.91, 0.93, 0.905);
  vec3 c = albedo * (uSunColor * wrap / 3.14159 + skyAmbient(N) * 1.0);
  return c * mix(1.0, 0.93, dense);
}

// Sky radiance for reflections: elevation LUT texture + measured left/right gradient + a
// faint sun aureole. Below the horizon (rays reflected into other waves): sea colour.
#ifdef OPT_EXPLORE
float gCloudAmount = 0.0;
vec3 gSkyGain = vec3(1.0);   // skyAzimuthGain of the mean reflected ray (set per pixel in roughReflection)
#endif
vec3 skyRefl(vec3 d) {
  // LUT indexed by sqrt(sin(elevation)) (fine near the horizon); measured left/right gradient
  float sy = max(d.y, 0.0);
  vec3 c = textureLod(uSkyTex, vec2(sqrt(sy) * (255.0 / 256.0) + 0.5 / 256.0, 0.5), 0.0).rgb;   // no gradients: called inside loops
  float azs = clamp(d.x / max(length(d.xz), 1e-3), -0.985, 0.985);   // sin(azimuth)
  float fe = clamp(sy * 2.63, 0.0, 1.0);
  // (scalar gradient + the per-channel tint of the sky dome: redder toward the sun side)
  // (the far and mid sea mirror the low sky, whose tint is weak: the sea's right side is as
  //  R-clipped turquoise as its left in the clip)
#ifdef OPT_EXPLORE
  c = mix(c * gSkyGain, vec3(0.68, 0.73, 0.75), gCloudAmount);   // (skyModel.js: every azimuth; the clip's gradient at azimuth 0)
#else
  c *= (1.0 + mix(0.0029, 0.0043, fe) * 57.3 * azs) * (1.0 + vec3(0.005, 0.0011, -0.0005) * fe * fe * 57.3 * azs);
#endif
  return mix(uDeepColor * 1.1 + vec3(0.02, 0.05, 0.06), c, smoothstep(-0.07, 0.009, d.y));
}
// Reflection integrated over the visible slope distribution in the view plane.
// sigV: std of the facet slope along the view direction. Returns reflected radiance (rgb)
// and the mean Fresnel (a) — the body is weighted by 1 - a.
// Smith shadowing of a reflected ray leaving at elevation angle with tangent tE over a sea of
// slope std sigS (Beckmann, Walter et al. 2007): low reflected rays hit the next wave.
float smithG1(float tE, float sigS) {
  if (tE <= 0.0) return 0.0;
  float a = tE / (1.41421 * max(sigS, 1e-3));
  if (a >= 1.6) return 1.0;
  float L = (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
  return 1.0 / (1.0 + L);
}

// sigS: slope std of the surrounding sea, for the shadowing of reflected rays.
vec4 roughReflection(vec3 N, vec3 V, float sigV, float sigS) {
#ifdef OPT_EXPLORE
  gCloudAmount = highCloudAmount(reflect(-V, N));
  gSkyGain = skyAzimuthGain(reflect(-V, N));   // (once per pixel: the taps differ mostly in elevation)
#endif
  float mu0 = clamp(dot(N, V), -1.0, 1.0);
  vec3 e = V - N * mu0;
  float le = length(e);
  e = le > 1e-5 ? e / le : vec3(0.0, 0.0, 1.0);
  float d0 = asin(clamp(mu0, -1.0, 1.0));             // grazing angle over the mean plane
  float sLo = max(-3.0 * sigV, tan(-d0 + 0.002));
  float sHi = 3.2 * sigV;
  if (sHi <= sLo) { sLo = tan(-d0 + 0.002); sHi = sLo + 0.1; }
  vec3 acc = vec3(0.0); float fAcc = 0.0, wAcc = 0.0;
  const int K = 5;
  // Gaussian relative to its value at the first node (no underflow when every visible
  // facet lies in the far tail, e.g. back-tilted mean normals at grazing angles)
  float s0 = max(sLo, 0.0);
  float iv = 0.5 / max(sigV * sigV, 1e-6);
  for (int i = 0; i < K + min(uEvtCount, 0); i++) {   // dynamic bound: not unrolled by FXC
    float s = mix(sLo, sHi, (float(i) + 0.5) / float(K));
    vec3 n = normalize(N + s * e);
    float mu = dot(n, V);
    if (mu <= 0.0) continue;
    float g = exp(-min((s * s - s0 * s0) * iv, 60.0));
    float w = g * mu / dot(n, N);
    vec3 r = reflect(-V, n);
    float F = fresnelWater(mu);
    // rays reflected low over the sea are intercepted by the neighbouring waves: they see
    // the back of the next wave (sea colour) instead of the bright horizon haze
    float G = smithG1(r.y / max(length(r.xz), 1e-4), sigS);
    acc += w * F * mix(uDeepColor * 1.1 + vec3(0.02, 0.05, 0.06), skyRefl(r), G);
    fAcc += w * F;
    wAcc += w;
  }
  vec4 result = vec4(0.0);
  if (wAcc < 1e-12) result = vec4(skyRefl(reflect(-V, N)), 1.0);
  else result = vec4(acc / wAcc, fAcc / wAcc);
  return result;
}

// Pale pebble bottom beyond the swash, lit by sun and sky through the water column above it.
vec3 bedRadiance(vec3 Pb, float hCol, vec3 sigT) {
  float cs = sqrt(1.0 - (1.0 - uSunDir.y * uSunDir.y) / (1.333 * 1.333));  // refracted sun cosine
  float e = 0.08;
  float bz = (bedProfile(Pb.z + e) - bedProfile(Pb.z - e)) / (2.0 * e);
  vec3 Nb = normalize(vec3(0.0, 1.0, -bz));
  float ndl = max(dot(Nb, uSunDir), 0.0);
  // patchy bottom: darker stones / weed in 1-5 m blotches
  float patchy = 0.7 + 0.25 * vnoise(Pb.xz * vec2(0.35, 0.5) + 3.7);
  // the terrace is paler on the left of the view (clip: luminous band (36,181,188) on the left,
  // R-clipped (10,174,182) on the right)
#ifdef OPT_EXPLORE
  float uL = clamp(-Pb.x / (0.36 * max(4.1 - Pb.z, 1.0)), -1.0, 1.0);   // (world-anchored)
#else
  float uL = clamp(-Pb.x / (0.36 * max(cameraPosition.z - Pb.z, 1.0)), -1.0, 1.0);
#endif
  vec3 alb = uBedAlbedo * patchy * (1.0 + uBedAlbedoLR * uL);
  // inshore of the pale terrace the bottom turns to darker grey-brown gravel
  alb *= mix(vec3(0.48, 0.72, 0.72), vec3(1.0), smoothstep(-5.5, -9.5, Pb.z));
  vec3 Tsun = exp(-sigT * hCol / cs);
  vec3 Tsky = exp(-sigT * hCol * 1.25);
  return alb * (uSunColor * ndl * Tsun / 3.14159 + uSkyAmb * Tsky * 0.85);
}

// Opaque-pass radiance at a world point if it is visible on screen (else rgb = -1).
// (refracted points just off-screen read the edge pixel: rejecting them made the frame's bottom
// rows and left columns fall back to the pale analytic bed - a bright cyan border)
vec3 screenBed(vec3 Pb) {
  vec4 c = projectionMatrix * (viewMatrix * vec4(Pb, 1.0));
  if (c.w <= 0.0) return vec3(-1.0);
  vec2 uv = c.xy / c.w * 0.5 + 0.5;
#ifdef OPT_EXPLORE
  // (off-screen: the caller takes the bed straight behind - clamping to the edge row smeared it
  //  into streaks along the frame edge when standing in the water)
  if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return vec3(-1.0);
#else
  if (any(lessThan(uv, vec2(-0.03))) || any(greaterThan(uv, vec2(1.03)))) return vec3(-1.0);
#endif
  uv = clamp(uv, 0.5 / uResolution, 1.0 - 0.5 / uResolution);
  float dS = linearizeDepth(textureLod(uOpaqueDepth, uv, 0.0).x);
  if (dS < c.w - 0.05 - 0.03 * c.w) return vec3(-1.0);   // something in front of the bed point
  return textureLod(uOpaqueColor, uv, 0.0).rgb;
}

float hash13g(vec3 p3) { p3 = fract(p3 * 0.1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }

// Full model. sig2: unresolved slope variance (x, z) in world axes (LEAN), without micro.
// rough: extra view-plane slope std at grazing incidence (see uSeaTune).
// crest: (horizontal thickness, crest height above, sun occlusion, face weight); (1, 0, 0, 0) = none.
vec3 shadeWaterFull(WIn s, vec2 fragXY, vec2 sig2, float rough, vec4 crest, float facetTilt) {

  vec3 N = s.N, V = s.V, L = uSunDir;
  float NoV = max(dot(N, V), 1e-4);
  float micro = uOptics.x + clamp(s.turb, 0.0, 2.0) * 0.05;
  vec2 var2 = max(sig2, vec2(0.0)) + micro * micro;
  // ---------------- reflection (sky + sun)
  vec3 e = normalize(V - N * dot(N, V) + vec3(0.0, 1e-5, 0.0));
  float sigV = sqrt(var2.x * e.x * e.x + var2.y * e.z * e.z + micro * micro * e.y * e.y);
  // At grazing incidence only the crest fronts of the short waves are visible (masking is
  // correlated with slope, which Smith's model ignores): effectively steeper visible facets.
  float graze = 1.0 - smoothstep(uSeaTune2.y, uSeaTune2.z, V.y);
  float xr = rough * graze;
  // ...and they are the steep downwind (camera-facing) fronts of the short waves: the visible
  // slope distribution is skewed toward the eye (Cox-Munk skewness) -> tilt its centre.
  // (within the last ~2° under the horizon only the steepest camera-facing fronts are seen: the
  //  far sea darkens toward the horizon line - G 144 at 45 m -> 122 at 270 m in the clip)
  float skewH = uSeaTune6.x * (1.0 - smoothstep(0.0, uSeaTune6.y, V.y));
  vec3 Nr = normalize(N + e * ((uSeaTune3.z + facetTilt) * graze + skewH));
  vec4 rr = roughReflection(Nr, V, sqrt(sigV * sigV + xr * xr), uSeaTune2.x);
  vec3 refl = rr.rgb;
  float F = rr.a;
  // sun: anisotropic Beckmann lobe with the same slope statistics
  vec3 H = normalize(V + L);
  float NoL = max(dot(N, L), 0.0);
  float cH = max(dot(H, N), 0.05);
  vec3 Ht = H / cH - N;                                 // facet slope that mirrors the sun
  // Project onto the actual tangent plane. XZ projections lose the vertical
  // component on steep faces and incorrectly turn them into broad sun mirrors.
  vec3 tangentX=normalize(vec3(1.0,0.0,0.0)-N*N.x+vec3(0.0,0.00001,0.0));
  vec3 tangentZ=normalize(cross(tangentX,N));
  vec2 sl = vec2(dot(Ht,tangentX),dot(Ht,tangentZ));
  // capillary facets steeper than the rendered normals carry the glints (glint slope std)
  // far out (grazing) the steep 35-40° facets needed to mirror the sun are rarer: narrower lobe;
  // the short waves riding a breaker crest are compressed and steepened: wider lobe there
  float sGmin = mix(uSeaTune3.x, uSeaTune3.w, graze), sGmax = mix(uSeaTune3.y, uSeaTune3.w, graze);
  float crestG = 1.0 + uGlint.z * clamp(s.crest, 0.0, 1.0);
  sGmin *= crestG; sGmax *= crestG;
  // a glassy swash film between its resolved ripple crests: narrow lobe (the crests mirror the
  // sun as thin continuous lines)
  sGmin = mix(sGmin, uFilm.y, s.film); sGmax = mix(sGmax, 0.4, s.film);
  vec2 varG = clamp(var2, vec2(sGmin * sGmin), vec2(sGmax * sGmax));
  float ex = min(0.5 * (sl.x * sl.x / varG.x + sl.y * sl.y / varG.y), 80.0);
  float Dsun = exp(-ex) / (6.2831853 * sqrt(varG.x * varG.y) * cH * cH * cH * cH);
  float Fs = fresnelWater(max(dot(V, H), 0.0));
  // Smith masking-shadowing (grazing views / low sun on tilted facets)
  float sG = sqrt(max(varG.x, varG.y));
  float Gs = smithG1(NoV / sqrt(max(1.0 - NoV * NoV, 1e-4)), sG) * smithG1(NoL / sqrt(max(1.0 - NoL * NoL, 1e-4)), sG);
  vec3 sunMean = uSunColor * min(Dsun * Fs * Gs * 0.25 / max(NoV, 0.05), 2000.0);
  // The sun lobe is carried by small mirror-like capillary facets. Explore keeps
  // their modulation continuous in world space, with a subtle highlight gain.
  // The reference-clip branch below instead samples Poisson-distributed glints
  // with the statistical mean of the original sun lobe.
  vec3 sunSpec = sunMean;
#ifdef OPT_EXPLORE
  // World-anchored capillary highlights. Continuous wave-time modulation and
  // footprint filtering replace per-frame screen-pixel dice rolls in explore.
  {
    float footprint=max(length(dFdx(s.P)),length(dFdy(s.P)));
    float keep=1.0-smoothstep(.010,.042,footprint);
    vec3 q=s.P*31.0+vec3(.11,-.32,-.24)*uTime;
    // Volumetric coordinates remain isotropic on a curling face: no stretched
    // rectangles from projecting a ground-plane noise lattice up the wave.
    float n=.68*vnoise3(q)+.32*vnoise3(q.yzx*1.73+vec3(13.2,7.1,21.9));
    float sparkle=pow(max(n,0.0),6.0)*16.0;
    // A small sun-only lift with stronger rare cores; the same angular lobe,
    // world-space motion and footprint fade keep glints attached to the water.
    float pattern=mix(1.0,.50+sparkle,keep);
    float glintGain=mix(1.12,1.04,s.film);
    sunSpec=min(sunMean*glintGain*mix(pattern,1.0,s.film),vec3(uSeaTune2.w));
  }
#else
  {
    const vec3 LUM = vec3(0.2126, 0.7152, 0.0722);
    vec3 dPx = dFdx(s.P), dPy = dFdy(s.P);
    float Apx = max(length(cross(dPx, dPy)), 1e-9);      // world area of this pixel (m²)
    const float Af = 1.0e-4;
    float lumG = dot(uSunColor, LUM) * (Fs / 6.8e-5);    // radiance of one lit facet (sun disc 0.53°)
    float frac = dot(sunMean, LUM) / max(lumG, 1e-6);    // area fraction of lit facets
    float gain = max(uOptics.y, 1e-3);
    float pxW = sqrt(Apx);                                // pixel size (m)
    // a glint on screen is the optics + sensor PSF (~1.3 px at 1080, scaled with the output) or,
    // close up, the projected facet itself: a soft round 3-5 px blob. Screen cells 4 sigma wide
    // hold at most one glint position each (count k), jittered inside the cell, gathered from the
    // 2x2 nearest cells with a compact smooth falloff (no truncated squares / crosses)
    float sres = min(uResolution.x, uResolution.y) / 1080.0;
    float sig = max(uGlint.x * sres, 0.4 * sqrt(Af) / pxW);   // PSF sigma (px)
    float C = 4.0 * sig;                                  // cell size (px)
    float mu = frac * C * C * Apx / Af * gain;           // expected glints per cell
    // the stretched, renewed face below a breaker crest is glassy: its sparkles are rare (they
    // cluster on the crest line, clip at 1.0-2.6 s)
    mu *= mix(1.0, uGlint.w, crest.w * (1.0 - clamp(s.crest, 0.0, 1.0)));
    // sparkles come in clusters (patches of steeper capillaries ~0.3-0.6 m, reforming in ~1 s)
    // (clip at 1.6-2.6 s: tight clumps of 5-10 small sparkles, 0.2-0.4 m, on the upper face)
    float cl = vnoise(s.P.xz * vec2(2.2, 3.4) + vec2(0.6, -0.9) * uTime);
    mu *= mix(0.03, 14.0, smoothstep(0.6, 0.9, cl));
#ifdef OPT_EXPLORE
    // close to the sun's mirror direction (the player can look down into it) nearly every capillary
    // ripple holds a point that mirrors the sun: the glitter is dense (the 1 cm² facet statistics
    // above describe the rare off-specular sparkles the clip camera sees)
    mu *= 1.0 + 30.0 * exp(-dot(sl, sl) / (2.0 * 0.15 * 0.15));
#endif
    // peak radiance of one glint: one facet's energy (Af Lg / gain) spread over the PSF (the
    // sensor clips it anyway)
    float sigM = sig * pxW;
    float peak = min(Af / (6.2831853 * sigM * sigM) / gain * lumG, uGlint.y);
    float fr = floor(uTime * 60.0 + 0.5);
    float dense = smoothstep(0.8, 3.0, mu);               // several glints per cell: mean + sparkle
    float acc = 0.0;
    if (mu > 1e-4 && dense < 1.0 && uDbg.x > -1.5) {
      vec2 v = fragXY / C;
      vec2 c0 = floor(v - 0.5);
      for (int n = 0; n < 4 + min(uEvtCount, 0); n++) {   // dynamic bound: not unrolled by FXC
        vec2 cid = c0 + vec2(float(n - (n / 2) * 2), float(n / 2));
        vec3 h = hash33(vec3(cid + 7919.0, fr));
        float k = floor(mu + h.x);
        vec2 dv = (v - cid - 0.2 - 0.6 * h.yz) * C;         // px
        // (a sparkle is a ripple crest facet: drawn out along the crest, ~1.4:1 horizontal)
        acc += k * exp(-0.5 * dot(dv * vec2(0.85, 1.2), dv * vec2(0.85, 1.2)) / (sig * sig)) * smoothstep(0.7 * C, 0.4 * C, length(dv));
      }
    }
    // (glint halos take the water's cyan: phone chroma processing smears the clipped cores' chroma)
    vec3 tint = normalize(uSunColor) * 1.732 * vec3(0.65, 1.0, 1.05);
    vec3 sunGl = min(acc * peak * tint, vec3(uSeaTune2.w));
    if (dense > 0.0) {
      float spk = -log(max(hash13g(vec3(fragXY, fr)), 1e-3));   // exponential, mean 1
      sunGl = mix(sunGl, min(sunMean * spk, vec3(uSeaTune2.w)), dense);
    }
    // a glassy film mirrors the sun along its resolved ripple crests: the smooth mean
    sunSpec = mix(sunGl, min(sunMean, vec3(uSeaTune2.w)), s.film);
  }

#endif

  // ---------------- transmission
  vec2 suv = fragXY / uResolution;
  float dScene = linearizeDepth(textureLod(uOpaqueDepth, suv, 0.0).x);
  float ib = textureLod(uBackDepth, suv, 0.0).x;
  float dBack = ib > 0.0 ? 1.0 / ib : 1.0e6;
  bool exitsBack = dBack > s.viewZ + 0.003 && dBack < dScene;
  float thick = max((exitsBack ? dBack : dScene) - s.viewZ, 0.0);
  if (s.thickHint > 0.0) { thick = min(thick, s.thickHint); exitsBack = true; }
  // water mass: clear terrace / offshore water vs the turbid, saturated-turquoise water of the
  // break zone (fine sediment and micro-bubbles kept in suspension by the breakers): only the
  // last ~1.5 m before the plunge line
  float turbid = smoothstep(-5.0, -3.2, s.P.z) * uTurbidColor.w;
  vec3 sigT = mix(uWaterAtten, uTurbidAtten, turbid);
  vec3 Lin = mix(uDeepColor, uTurbidColor.rgb, turbid);
  // aeration (bubbles + stirred-up fines) of the impact zone, the bore and the early backwash:
  // an olive-grey, strongly scattering water column that hides the bed and its caustics
  float aer = clamp(s.aer, 0.0, 1.0);
  float silt = clamp(s.silt * uSilt.w, 0.0, 1.0);
  sigT += uAerate.z * aer + 12.0 * silt;
  vec3 body;
  vec3 Tr = refract(-V, N, 1.0 / 1.333);
  {
    // trace the refracted ray to the bed
    float hC = max(s.P.y - hydraulicBedHeight(s.P.xz), 0.0);
    float ty = max(-Tr.y, 0.12);
    vec3 Pb = s.P + Tr * (hC / ty);
    float hB = max(s.P.y - hydraulicBedHeight(Pb.xz), 0.0);
    Pb = s.P + Tr * (hB / ty);
    float sPath = length(Pb - s.P);
    vec3 Lbed = vec3(-1.0);
    float nearW = smoothstep(-6.0, -4.5, s.P.z);
    if (nearW > 0.0) {
      // the capillary ripples of the sheet bend the rays: the pebbles are seen slightly
      // blurred / doubled (contrast through a swash film drops to ~60-75 %)
      float rb = clamp(hB * 0.22, 0.0015, 0.03);
      vec3 b0 = screenBed(Pb);
      if (b0.x >= 0.0) {
        vec3 b1 = screenBed(Pb + vec3(rb, 0.0, 0.5 * rb));
        vec3 b2 = screenBed(Pb - vec3(0.5 * rb, 0.0, rb));
        vec3 acc = b0; float n = 1.0;
        if (b1.x >= 0.0) { acc += b1; n += 1.0; }
        if (b2.x >= 0.0) { acc += b2; n += 1.0; }
        Lbed = acc / n;
      } else if (dScene > s.viewZ) {
        // refracted point hidden (by a pebble crown, the swash edge...): the bed straight behind
        Lbed = textureLod(uOpaqueColor, suv, 0.0).rgb;
      }
    }
    // (the water drawn up into a steep wave face carries fine sediment: the bed hides behind it;
    //  through a gentle hump over the pale terrace the bottom still shows - the luminous band)
    vec3 T = exp(-(sigT + uCrestShade.w * crest.w * crest.w * crest.w) * sPath);
    vec3 Lan = T.g > 0.004 ? bedRadiance(Pb, s.P.y - Pb.y, sigT) : vec3(0.0);
    if (Lbed.x < 0.0) Lbed = Lan; else Lbed = mix(Lan, Lbed, nearW);
    // In-scatter behind a camera-facing wave face. The sun is high behind the wave. The view ray
    // enters a face flatter than it enters level water, so the light it collects has been
    // scattered less far from the sun's direction (forward-peaked phase function) and has come
    // down through the crest: the face glows brighter than level water. Under a thick shoaling
    // crest the sunlight crosses more water (red absorbed first: saturated, darker turquoise);
    // under the steep peaked crest about to throw it crosses little (bright, pale turquoise).
    // Down the face the path through the crest grows: measured profile ~(1 - k s²) from the
    // crest (s = 0) to the base (s = 1), the base turning greener (colour analysis §4.1).
    // crest = (face fraction s, crest stage, sun occlusion, face weight).
    float fw = crest.w;
    vec3 top = mix(uFaceThick.rgb, uFaceGlow.rgb, smoothstep(0.1, 0.9, crest.y));
    vec3 faceIn = top * max(1.0 - uFaceSig.rgb * crest.x * crest.x, vec3(0.15));
    vec3 LinF = mix(Lin, faceIn, fw);
    // the light scattered within the first few cm of water has not yet lost its red: the
    // in-scatter only takes the water colour over ~0.5 m of path (a swash film scatters grey)
    LinF = mix(vec3(dot(LinF, vec3(0.2126, 0.7152, 0.0722))), LinF, 1.0 - exp(-sPath * 2.5));
    // the crest shadows the trough just ahead of the toe (its bed and water): dark toe band
    vec3 shade = mix(vec3(1.0), uCrestShade.xyz, (1.0 - exp(-crest.z / uFaceSig.w)) * (1.0 - fw));
    body = (Lbed * T + LinF * (1.0 - T)) * shade;
  }
  // Thin sheets transmit the actual background with Beer-Lambert extinction.
  // Back-depth coverage is a confidence-weighted correction, never a binary
  // switch to an emissive cyan material. Thick wave bodies keep the same optics.
  if(exitsBack){
    float path=thick/max(NoV*.55+.45,.25);
    vec3 T=exp(-sigT*path);
    float phase=.18+.82*pow(max(dot(-V,L),0.0),5.0);
    vec3 scatter=mix(Lin,uGlowTint*uOptics.z*.30*phase,.55);
    vec3 behind=textureLod(uOpaqueColor,suv, 0.0).rgb;
    vec3 sheet=behind*T+scatter*(1.0-T);
    float confidence=smoothstep(.004,.055,thick)*(1.0-smoothstep(.12,.65,path));
    confidence*=smoothstep(.01,.16,dScene-dBack);
    if(s.thickHint>0.0) confidence=1.0;
    body=mix(body,sheet,confidence);
  }
  // milky (aerated / sediment-laden) water in the bore and early backwash. A bubble cloud
  // scatters the whole sky dome and the sun about equally: its ambient is the sky's luminance
  // (grey), not the blue of the sky overhead - the swash water never looks turquoise.
  float skyLum = dot(skyAmbient(N), vec3(0.2126, 0.7152, 0.0722));
  vec3 lit = uSunColor * 0.25 + vec3(skyLum);
  // swash films carry a little suspended fine sediment (grey veil over the pebbles) early on;
  // the SWE milk channel carries the bubbles of the bore and backwash (clears with tau ~1.25 s)
  float dW = s.P.y - hydraulicBedHeight(s.P.xz);
  float swFilm = smoothstep(-1.8, -0.6, s.P.z) * (1.0 - smoothstep(0.02, 0.25, dW));
  float milkAmt = max(clamp(s.milk * uAerate.y, 0.0, 1.0), uAerate.x * aer);
  milkAmt = max(milkAmt, uMilkBase * swFilm);
  // a rising wave face is renewed clear water drawn up from the trough: no milk on it
  milkAmt *= 1.0 - crest.w;
  // the backwash toe (where the backwash runs into the next bore: the most turbulent shallow water)
  // is sediment-laden, dark grey-teal - not a milky white line (clip 6.9 s, v 0.70-0.74)
  float toe = smoothstep(uToe.w, uToe.w + 0.9, s.turb) * smoothstep(0.04, 0.1, dW) * (1.0 - crest.w);
  milkAmt *= 1.0 - 0.6 * toe;
  // deeper water stirred by the backwash (the trough ahead of the next face) keeps the fines in
  // suspension after the bubbles have risen: murky, darker olive-teal rather than milky
  body = mix(body, uMurk.rgb, clamp(s.murk * uMurk.w, 0.0, 1.0) * (1.0 - crest.w));
  // silt-laden bore / backwash water: olive-grey, lit by sun and sky luminance
  body = mix(body, uSilt.rgb * lit * 0.5, silt * (1.0 - exp(-thick * 25.0)));
  // (a 1-3 cm late film holds too few bubbles to veil the pebbles: 60-75 % contrast remains)
  body = mix(body, uMilkColor * lit * 0.5, milkAmt * (1.0 - exp(-thick * 7.0)));
  body = mix(body, uToe.rgb, 0.7 * toe);
  // under a whitewater blanket the water column is shaded by the foam above and full of stirred
  // fines: through the blanket's holes it reads dark, turbid olive (clip (95-135, 110-150, 100-145))
  body = mix(body, uFoamShade.rgb, clamp(s.under, 0.0, 1.0) * uFoamShade.w);

  // a swash film is glassy between its ripple crests: its mean Fresnel stays near the flat value
  // (~0.06 at 30-40° depression) - the high blue sky it mirrors is a faint tint, the film reads
  // grey-neutral (the ripple and turbulence normals above overstate the mean for thin films)
  refl *= 1.0 - uFilm.z * s.film;
  // shallow swash, bore and backwash water is bubbly and rough at every scale: it mirrors the sky
  // dome as a broad, weak, desaturated sheen, not the deep blue of the sky overhead (the clip's
  // swash reads grey-olive, B <= R, at every time)
  float swS = clamp(s.swash, 0.0, 1.0) * uSwashRefl.x;
  refl = mix(refl, vec3(dot(refl, vec3(0.2126, 0.7152, 0.0722))) * vec3(1.04, 1.0, 0.92), swS) * (1.0 - uSwashRefl.y * swS);
  if (uDbg.z > 0.5) refl *= uDbg.w;   // (debug: reflection scale)
  vec3 col = body * (1.0 - F) + refl + sunSpec;
  // ---------------- foam
  // thick blanket foam is the brightest white (~0.9); thin lace and tearing scraps are
  // translucent: dimmer, with a little of the water beneath showing through (~0.7)
  vec3 fc = shadeFoam(normalize(mix(N, vec3(0.0, 1.0, 0.0), 0.3)), s.foamDense);
  float dns = clamp(s.foamDense, 0.0, 1.0);
  fc = mix(mix(fc, body, 0.12), fc, dns) * mix(0.82, 1.1, dns);
  col = mix(col, fc, clamp(s.foam, 0.0, 1.0));
  if (uDbg.x > 0.5) {   // debug views (grey = quantity x uDbg.y)
    float q = uDbg.x < 1.5 ? crest.x : uDbg.x < 2.5 ? crest.y : uDbg.x < 3.5 ? crest.z : uDbg.x < 4.5 ? crest.w
            : uDbg.x < 5.5 ? turbid : uDbg.x < 6.5 ? milkAmt : uDbg.x < 7.5 ? aer : uDbg.x < 8.5 ? s.murk : uDbg.x < 9.5 ? F : s.foam;
    col = uDbg.x < 10.5 ? vec3(q * uDbg.y) : uDbg.x < 11.5 ? body : refl;
  }
  return col;
}

// Interface used by the lip ribbon: isotropic unresolved variance from s.rough.
vec3 shadeWater(WIn s, vec2 fragXY) {
  // a thin sheet is seen from whichever side faces the eye: shade that side
  if (dot(s.N, s.V) < 0.0) s.N = -s.N;
  float r = max(s.rough - uOptics.x, 0.0);
  s.aer = 0.0; s.murk = 0.0; s.film = 0.0; s.silt = 0.0; s.swash = 0.0; s.under = 0.0;   // (the ribbon shades its own aeration)
  // the jet / curtain hangs in front of the face under the peaked crest: where it faces the
  // camera it glows like the upper face it continues (steep-stage face in-scatter)
  float fwL = smoothstep(0.05, 0.5, s.N.z);
  return shadeWaterFull(s, fragXY, vec2(r * r), uSeaTune.x, vec4(0.2, 1.0, 0.0, fwL), 0.0);
}
`;
