// Hash-without-Sine routines: Copyright (c) 2014 David Hoskins, MIT.
// See licenses/Hash-without-Sine-MIT.txt and THIRD_PARTY_NOTICES.md.
import * as THREE from 'three';
import { FullscreenPass, makeShader, floatRT } from '../core/gpu.js';
import { CONFIG, sunDirection } from '../config.js';
import { Exposure } from './Exposure.js';
import { WATERLINE } from '../glsl/underwater.js';
import { FXAAShader } from 'three/addons/shaders/FXAAShader.js';
import { bandLumaRef, BAND_LUMA_MEAN, bandCVRef, BAND_CV_MEAN, ROW_DV0, ROW_DV1 } from './farsea.js';

// HDR post chain tuned to look like the reference phone video (colour analysis §1, §8):
//   composite whitewater
//   -> small round bloom: thresholded half-res core (σ ≈ 1.5 px) + faint quarter-res tail
//   -> sun probe (explore, sun near the frame): visible fraction of the sun disc
//   -> GRADE: sun lens glare (veil, star, ghosts) + metering pull-down, exposure (main.js AE
//      with its excursion scaled to the clip's, see params.aeGain), vivid saturation, soft
//      shoulder, per-channel gamut clip, sRGB OETF
//   -> ISP/codec (display space, Y'CbCr): luma-only unsharp mask, chroma low-pass (4:2:0 +
//      encoder: glints take their neighbours' chroma), far-sea "maroon dash" chroma artefact,
//      ±0.5/255 dither.

const LUMA = /* glsl */ `const vec3 LUMA709 = vec3(0.2126, 0.7152, 0.0722);`;

const COMPOSITE = WATERLINE+/* glsl */ `
uniform sampler2D uScene;
uniform sampler2D uParticles;
uniform sampler2D uSceneDepth;
uniform sampler2D uCrashVolume;
uniform float uCrashOn;
uniform mat4 uEyeInverseVP;
uniform vec3 uEye;
in vec2 vUv;
void main() {
  vec3 s = textureLod(uScene, vUv, 0.0).rgb;
  vec4 p = texture(uParticles, vUv);
  if(uUnderwaterOn>.5){
    vec4 nearPoint=uEyeInverseVP*vec4(vUv*2.-1.,-1.,1.);
    vec3 np=nearPoint.xyz/nearPoint.w;
    float wet=submergedAt(np);
    float z=textureLod(uSceneDepth,vUv, 0.0).r;
    vec4 point=uEyeInverseVP*vec4(vUv*2.-1.,z*2.-1.,1.);
    float path=min(length(point.xyz/point.w-uEye),60.);
    s=mix(s,waterTravel(s,path),wet);
    if(uCrashOn>.5){vec4 crash=texture(uCrashVolume,vUv);s=s*(1.-crash.a)+crash.rgb;}
    // Above-water spray must not form a white screen below the surface.
    // Submerged bubbles are depth-tested in the main scene instead.
    p*=1.-wet;
  }
  gl_FragColor = vec4(s * (1.0 - p.a) + p.rgb, 1.0);
}`;

// 2x downsample: 4 bilinear taps on the diagonals + centre (≈ 4x4 px, centre-weighted);
// the first level applies a soft-knee threshold so only speculars bloom.
const DOWN = /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform float uThreshold;
uniform float uFirst;
in vec2 vUv;
vec3 pre(vec3 c) {
  if (uFirst < 0.5) return c;
  float br = max(c.r, max(c.g, c.b));
  float knee = uThreshold * 0.6;
  float soft = clamp(br - uThreshold + knee, 0.0, 2.0 * knee);
  soft = soft * soft / (4.0 * knee + 1e-4);
  float w = max(soft, br - uThreshold) / max(br, 1e-4);
  return c * w;
}
void main() {
  vec2 t = uTexel;
  vec3 a = pre(texture(uSrc, vUv + t * vec2(-1.0, -1.0)).rgb);
  vec3 b = pre(texture(uSrc, vUv + t * vec2(1.0, -1.0)).rgb);
  vec3 c = pre(texture(uSrc, vUv + t * vec2(-1.0, 1.0)).rgb);
  vec3 d = pre(texture(uSrc, vUv + t * vec2(1.0, 1.0)).rgb);
  vec3 e = pre(texture(uSrc, vUv).rgb);
  gl_FragColor = vec4((a + b + c + d) * 0.125 + e * 0.5, 1.0);
}`;

// separable 5-tap Gaussian (isotropic after both passes, so glint halos are round, not square)
const BLUR = /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uStep;      // one texel along the blur axis
uniform vec3 uW;         // weights for offsets 0, 1, 2
in vec2 vUv;
void main() {
  vec3 s = texture(uSrc, vUv).rgb * uW.x
         + (texture(uSrc, vUv + uStep).rgb + texture(uSrc, vUv - uStep).rgb) * uW.y
         + (texture(uSrc, vUv + 2.0 * uStep).rgb + texture(uSrc, vUv - 2.0 * uStep).rgb) * uW.z;
  gl_FragColor = vec4(s, 1.0);
}`;

// Sun probe (one fragment): visible fraction of the sun disc in the HDR composite (scene +
// whitewater), from 13 taps inside the disc compared with the unoccluded limb-darkened disc
// radiance. Spray, the berm or anything else in front of the sun lowers it; taps that fall
// outside the frame count as visible (nothing in this scene rises that high).
const SUNPROBE = /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uSunUv;     // disc centre (uv)
uniform vec2 uRadUv;     // disc radius (uv, x and y)
uniform float uLref;     // luminance of the unoccluded disc centre
in vec2 vUv;
${LUMA}
void main() {
  float s = 0.0;
  for (int i = 0; i < 13; i++) {
    float ring = i == 0 ? 0.0 : (i < 7 ? 0.45 : 0.8);
    float a = float(i) * 1.0471976 + (i < 7 ? 0.0 : 0.5235988);
    vec2 uv = uSunUv + ring * vec2(cos(a), sin(a)) * uRadUv;
    float v = 1.0;
    if (uv.x >= 0.0 && uv.y >= 0.0 && uv.x <= 1.0 && uv.y <= 1.0) {
      float limb = 0.45 + 0.55 * sqrt(max(1.0 - ring * ring, 0.0));
      v = smoothstep(0.08, 0.6, dot(textureLod(uSrc, uv, 0.0).rgb, LUMA709) / (uLref * limb));
    }
    s += v;
  }
  gl_FragColor = vec4(s / 13.0, 0.0, 0.0, 1.0);
}`;

const GRADE = /* glsl */ `
uniform sampler2D uHDR;
uniform sampler2D uBloom;
uniform sampler2D uBloomLow;
uniform float uExposure;
uniform sampler2D uMeteredExposure;
uniform float uMeterOn;
uniform float uBloomStrength;
uniform float uBloomTail;
uniform float uSaturation;
uniform float uShoulder;
// camera glare of the sun (explore: looking up toward it)
uniform float uSunOn;        // 0 = sun far outside the frame: no glare terms at all
uniform sampler2D uSunVis;   // 1x1 sun probe: visible fraction of the disc
uniform vec3 uSunV;          // direction toward the sun, view space
uniform vec4 uSunPx;         // sun image position (px, may be off-frame), optical centre (px)
uniform float uFocal;        // focal length (px)
uniform vec3 uSunTint;       // sun colour (unit luminance)
uniform vec4 uGlare;         // veil peak, veil core angle (rad), frame veil, star amplitude
uniform vec4 uStar;          // star ray half-length (rad), ray width (rad), rotation (rad), ghost amplitude
uniform float uSunEV;        // metering pull-down (EV) with the whole disc visible
in vec2 vUv;
${LUMA}
// Lens glare of the sun, scene-linear (added before exposure: it is light on the sensor).
//  - veiling glare: scatter in the lens stack / cover glass, PSF ~ 1 / (1 + (θ/θ0)^2), fading
//    out by ~20°, plus a faint uniform veil over the frame
//  - diffraction star: 6 thin rays (dispersed: red rays slightly wider)
//  - ghosts: inter-reflections of the lens surfaces, on the line from the sun through the
//    optical centre (the phone's small green ghost mirrored across the frame, fainter large ones)
float ghostDisc(vec2 p, vec2 c, float rad) {
  float d = length(p - c) / uFocal;
  return smoothstep(rad, rad * 0.55, d) * (0.7 + 0.3 * smoothstep(rad * 0.4, rad * 0.95, d));
}
// periodic 1D value noise over the angle around the sun's image (n cells per turn)
float hashA(float i) { return fract(sin(i * 12.9898 + 4.1) * 43758.5453); }
float angNoise(float phi, float n) {
  float x = (phi / 6.2831853 + 0.5) * n, i = floor(x), f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(hashA(mod(i, n)), hashA(mod(i + 1.0, n)), f);
}
vec3 sunGlare(float vis) {
  vec2 p = gl_FragCoord.xy;
  vec3 rd = normalize(vec3((p - uSunPx.zw) / uFocal, -1.0));
  float th = 2.0 * asin(min(length(rd - uSunV) * 0.5, 1.0));     // angle to the sun (rad)
  vec2 d = (p - uSunPx.xy) / uFocal;                              // image space around the sun
  float phi = atan(d.y, d.x);
  // the veil's inner part is streaky (scratches / dust on the cover glass): fine radial streaks
  float streak = 1.0 + 0.3 * (angNoise(phi, 163.0) + 0.6 * angNoise(phi, 347.0) - 0.8) * smoothstep(0.004, 0.025, th) * (1.0 - smoothstep(0.05, 0.14, th));
  float k2 = uGlare.y * uGlare.y;
  vec3 g = uSunTint * (uGlare.x / (1.0 + th * th / k2) * (1.0 - smoothstep(0.17, 0.35, th)) * streak + uGlare.z);
  vec3 star = vec3(0.0);
  for (int i = 0; i < 3; i++) {
    float a = uStar.z + float(i) * 1.0471976;
    vec2 ax = vec2(cos(a), sin(a));
    float al = dot(d, ax) / uStar.x, ac = d.x * ax.y - d.y * ax.x;
    vec3 w = uStar.y * vec3(1.2, 1.0, 0.85);
    star += (i == 0 ? 1.0 : (i == 1 ? 0.75 : 0.55)) * exp(-0.5 * ac * ac / (w * w)) / (1.0 + al * al * sqrt(abs(al)));
  }
  g += uSunTint * star * uGlare.w;
  // ghosts: g = centre + (sun - centre) * k
  vec2 c = uSunPx.zw, ax = uSunPx.xy - c;
  vec3 gh = vec3(0.25, 1.0, 0.45) * 1.4 * ghostDisc(p, c - ax * 0.62, 0.016)
          + vec3(0.85, 0.45, 1.0) * 0.30 * ghostDisc(p, c - ax * 1.10, 0.060)
          + vec3(1.0, 0.72, 0.36) * 0.45 * ghostDisc(p, c + ax * 0.38, 0.026);
  g += gh * uStar.w;
  return g * vis * uSunOn;
}
// Phone look fitted to the clip (colour analysis §8, option A): linear mid-tones,
// vivid saturation, a soft shoulder that rolls diffuse white to ~0.8 and lets only
// speculars clip, per-channel gamut clamp (saturated cyan gets R = 0 like the reference).
vec3 phoneLook(vec3 c, float exposure) {
  c *= exposure;
  float Y = dot(c, LUMA709);
  c = Y + (c - Y) * uSaturation;
  float pk = max(c.r, max(c.g, c.b));
  float S = uShoulder, D = 1.0 - uShoulder;
  if (pk > S) {
    float np = 1.0 - D * D / (pk + D - S);
    c *= np / pk;
    float g = 1.0 - 1.0 / (0.15 * (pk - np) + 1.0);
    c = mix(c, vec3(np), g);
  }
  return clamp(c, 0.0, 1.0);
}
vec3 toSRGB(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
void main() {
  vec3 bloom = texture(uBloom, vUv).rgb + texture(uBloomLow, vUv).rgb * uBloomTail;
  vec3 hdr = texture(uHDR, vUv).rgb + bloom * uBloomStrength;
  float exposure = uExposure * mix(1.0,texelFetch(uMeteredExposure,ivec2(0),0).g,uMeterOn);
  if (uSunOn > 0.0) {
    float vis = texelFetch(uSunVis, ivec2(0), 0).r;
    hdr += sunGlare(vis);
    exposure *= exp2(-uSunEV * vis);   // the metering sees the sun's glare: exposure drops
  }
  gl_FragColor = vec4(toSRGB(phoneLook(hdr, exposure)), 1.0);
}`;

// Row profile of the far sea: mean display luma of ROWS bins of dv below the horizon
// (u 0.05-0.95), one fragment per bin into a ROWS x 1 target, so the ISP pass can compare
// this frame's sea brightness with the reference's at the same image position.
const BAND = /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uHorizon;   // horizon line in top-left image coords: v = x + y * u
uniform vec2 uRowDV;     // dv range covered by the bins
uniform int uBins;
uniform int uNU;
uniform int uNV;
in vec2 vUv;
${LUMA}
void main() {
  float bins = float(uBins);
  float i0 = floor(gl_FragCoord.x);
  vec2 sz = vec2(textureSize(uSrc, 0));
  float s = 0.0, s2 = 0.0;
  for (int j = 0; j < uNV; j++) {
    float dv = uRowDV.x + (uRowDV.y - uRowDV.x) * (i0 + (float(j) + 0.5) / float(uNV)) / bins;
    for (int i = 0; i < uNU; i++) {
      float u = 0.05 + 0.9 * (float(i) + 0.5) / float(uNU);
      float v = uHorizon.x + uHorizon.y * u + dv;
      // exact texels (no bilinear averaging, which would hide the trough contrast)
      ivec2 ip = clamp(ivec2(vec2(u, 1.0 - v) * sz), ivec2(0), ivec2(sz) - 1);
      float y = dot(texelFetch(uSrc, ip, 0).rgb, LUMA709);
      s += y; s2 += y * y;
    }
  }
  float n = float(uNU * uNV);
  gl_FragColor = vec4(s / n, s2 / n, 0.0, 1.0);
}`;

const ISP = /* glsl */ `
uniform sampler2D uSrc;      // graded, sRGB-encoded
uniform sampler2D uBand;     // ROWS x 1: far-sea row luma profile of this frame
uniform vec2 uTexel;
uniform float uSharpen;
uniform float uChromaOff;    // bilinear tap offset (texels) of the chroma low-pass
uniform float uTime;
// far-sea maroon artefact
uniform float uMaroon;       // 0 = off, 1 = measured strength
uniform mat4 uInvViewProj;
uniform vec3 uCamPos;
uniform vec2 uHorizon;       // horizon line (top-left image coords): v = x + y * u
uniform vec2 uRowDV;
uniform float uGust;         // clip band brightness now / its mean (1 off the clip timeline)
uniform float uContrast;     // extra trough-contrast gain (tail shape)
uniform float uCVTarget;     // clip's trough contrast (row luma SD / mean) now
uniform vec4 uMar;           // threshold (x row mean), blob amplitude, patch amplitude, left-right offset
uniform vec2 uMarW;          // transition half-width, overall weight
in vec2 vUv;
${LUMA}
vec3 toYCC(vec3 c) { float y = dot(c, LUMA709); return vec3(y, (c.b - y) / 1.8556, (c.r - y) / 1.5748); }
vec3 fromYCC(vec3 q) {
  float r = q.x + 1.5748 * q.z, b = q.x + 1.8556 * q.y;
  return vec3(r, (q.x - 0.2126 * r - 0.0722 * b) / 0.7152, b);
}
float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float hash13(vec3 p) { p = fract(p * vec3(0.1031, 0.1030, 0.0973)); p += dot(p, p.yzx + 33.33); return fract((p.x + p.y) * p.z); }
float vnoise(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  vec2 e = vec2(1.0, 0.0);
  float a = mix(mix(hash13(i), hash13(i + e.xyy), f.x), mix(hash13(i + e.yxy), hash13(i + e.xxy), f.x), f.y);
  float b = mix(mix(hash13(i + e.yyx), hash13(i + e.xyx), f.x), mix(hash13(i + e.yxx), hash13(i + e.xxx), f.x), f.y);
  return mix(a, b, f.z) * 2.0 - 1.0;
}
// Far-sea codec artefact ("maroon dashes"): dark ripple troughs lose their cyan chroma.
// Fitted on the clip (post/farsea.js): with z = luma / row mean x (clip band brightness)^0.7,
// P(maroon | z) = 0.68 Phi((0.823 - z) / 0.052), i.e. troughs ~18 % darker than their image
// row flip. Here z uses this frame's row means (so the water's own level does not matter),
// its trough contrast scaled toward the clip's; the threshold is jittered by world-anchored
// noise: dash scale (which troughs flip, and the third that never do) plus cat's-paw patches.
// Returns false when the pixel cannot flip; otherwise the threshold T, the weight, 1 / row mean
// and the target chroma (Cb, Cr) of this dash.
bool maroonSetup(vec3 ycc, float ymin, out vec4 th, out vec2 target) {
  th = vec4(0.0); target = vec2(0.0);
  // hue gate: saturated cyan/teal water only (sky, foam, spray, gravel are untouched)
  float gH = smoothstep(-0.10, -0.16, ycc.z) * step(0.0, ycc.y);
  if (gH <= 0.0) return false;
  vec4 p = uInvViewProj * vec4(vUv * 2.0 - 1.0, 1.0, 1.0);
  vec3 dir = p.xyz / p.w - uCamPos;
  float b = -dir.y / max(length(dir.xz), 1e-6);   // tan(depression below the horizon)
  if (b <= 1e-5) return false;
  float D = uCamPos.y / b;                        // horizontal distance of the sea point (m)
  // distance profile (far-sea analysis §7): full 25-95 m, ~1/3 at 120-190 m, gone toward the
  // horizon; fades in over 18.5-25 m (1/4 of the plateau at 19-23 m, none in the bright shallows)
  float gD = (1.0 - 0.7 * smoothstep(95.0, 190.0, D)) * (1.0 - smoothstep(190.0, 420.0, D));
  if (gD <= 0.0 || D < 18.5) return false;
  float u = vUv.x;
  float dv = (1.0 - vUv.y) - (uHorizon.x + uHorizon.y * u);
  float x = clamp((dv - uRowDV.x) / (uRowDV.y - uRowDV.x) * 16.0 - 0.5, 0.0, 15.0);
  int i0 = int(x);   // row mean, mean square (texelFetch: float targets need not be filterable)
  vec2 rs = mix(texelFetch(uBand, ivec2(i0, 0), 0).rg, texelFetch(uBand, ivec2(min(i0 + 1, 15), 0), 0).rg, x - float(i0));
  float rowInv = 1.0 / max(rs.x, 0.02);
  // trough contrast of this row scaled to the clip's at this time (and output size)
  float cg = clamp(uContrast * uCVTarget / max(sqrt(max(rs.y - rs.x * rs.x, 0.0)) * rowInv, 1e-3), 0.7, 1.8);
  // left third ~4x the right third (brighter sky reflection toward the sun + a wind lane)
  float T = uMar.x + uMar.w * (0.5 - u) * 2.0 - 0.15 * (1.0 - smoothstep(18.5, 25.0, D));
  float zmin = (1.0 + (ymin * rowInv - 1.0) * cg) * uGust;
  if (zmin > T + 0.8 * (uMar.y + uMar.z) + uMarW.x) return false;   // bright water: nothing to do
  float az = atan(dir.x, -dir.z);                 // world-anchored angular coordinates (rad)
  float t = uTime;
  // cat's-paw patches: ~0.05 rad (75 px) along-shore, banded in depression, re-forming in ~1 s
  float paws = 0.65 * vnoise(vec3(az / 0.05, b / 0.012, t * 0.7)) + 0.35 * vnoise(vec3(az / 0.021, b / 0.005, t * 1.1 + 17.0));
  // dash scale (~6 x 1.5 px at 1080p, lifetime ~0.5 s)
  vec3 pd = vec3(az / 0.0040, b / 0.0011, t * 2.0);
  float blob = vnoise(pd + vec3(0.0, 0.0, 5.0));
  float elig = smoothstep(-0.32, -0.12, vnoise(pd * vec3(0.8, 0.9, 1.0) + vec3(0.0, 0.0, 41.0)));
  th = vec4(T + uMar.y * blob + uMar.z * paws, gD * gH * elig * uMarW.y, rowInv, cg);
  // dash colour varies like the clip's: Cb 7-23, Cr -24..+1 (8-bit), mauve to maroon
  target = vec2(15.0 + 14.0 * vnoise(pd * 0.7 + vec3(0.0, 0.0, 73.0)), -11.0 + 22.0 * vnoise(pd * 0.7 + vec3(0.0, 0.0, 97.0))) / 255.0;
  return true;
}
// collapse weight of one chroma tap with luma y
float maroonW(float y, vec4 th) {
  float z = (1.0 + (y * th.z - 1.0) * th.w) * uGust;
  return smoothstep(-uMarW.x, uMarW.x, th.x - z) * th.y;
}
void main() {
  vec2 t = uTexel;
  vec3 c = texture(uSrc, vUv).rgb;
  // luma: mild unsharp mask in linear light (phone ISP; radius ~0.7 px, luma only)
  float yc = dot(c, LUMA709);
  float yn = pow(dot(texture(uSrc, vUv + vec2(0.0, t.y)).rgb, LUMA709), 2.2)
           + pow(dot(texture(uSrc, vUv - vec2(0.0, t.y)).rgb, LUMA709), 2.2)
           + pow(dot(texture(uSrc, vUv + vec2(t.x, 0.0)).rgb, LUMA709), 2.2)
           + pow(dot(texture(uSrc, vUv - vec2(t.x, 0.0)).rgb, LUMA709), 2.2);
  float yl = pow(yc, 2.2);
  float ys = pow(max(yl + (yl - yn * 0.25) * uSharpen, 0.0), 1.0 / 2.2);
  // chroma: separable [p, 1-2p, p]^2 low-pass from 4 bilinear taps (4:2:0 + encoder smoothing)
  float o = uChromaOff;
  vec3 q0 = toYCC(texture(uSrc, vUv + t * vec2(o, o)).rgb), q1 = toYCC(texture(uSrc, vUv + t * vec2(-o, o)).rgb);
  vec3 q2 = toYCC(texture(uSrc, vUv + t * vec2(o, -o)).rgb), q3 = toYCC(texture(uSrc, vUv + t * vec2(-o, -o)).rgb);
  vec3 q = (q0 + q1 + q2 + q3) * 0.25;
  // far-sea codec artefact, applied per chroma tap so the dashes get the same chroma softness
  vec4 th; vec2 target;
  if (uMaroon > 0.0 && maroonSetup(q, min(min(q0.x, q1.x), min(q2.x, q3.x)), th, target)) {
    th.y *= uMaroon;
    q.yz = (mix(q0.yz, target, clamp(maroonW(q0.x, th), 0.0, 1.0)) + mix(q1.yz, target, clamp(maroonW(q1.x, th), 0.0, 1.0))
          + mix(q2.yz, target, clamp(maroonW(q2.x, th), 0.0, 1.0)) + mix(q3.yz, target, clamp(maroonW(q3.x, th), 0.0, 1.0))) * 0.25;
  }
  q.x = ys;
  vec3 s = clamp(fromYCC(q), 0.0, 1.0);
  // +-0.5/255 dither only: the reference has no visible sensor grain (heavy ISP denoise)
  s += (hash12(gl_FragCoord.xy + fract(uTime * 7.13) * 91.0) - 0.5) / 255.0;
  gl_FragColor = vec4(s, 1.0);
}`;

function gauss3(sigma) {
  const w1 = Math.exp(-1 / (2 * sigma * sigma)), w2 = Math.exp(-4 / (2 * sigma * sigma));
  const n = 1 + 2 * w1 + 2 * w2;
  return new THREE.Vector3(1 / n, w1 / n, w2 / n);
}
const smooth01 = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };

export class Post {
  constructor(renderer, w, h, shared={}) {
    this.renderer = renderer;
    this.params = {
      exposure: 1.0, bloom: 0.11, bloomThreshold: 3.0, bloomTail: 0.7, bloomSigma: 0.8, bloomTailSigma: 1.5, saturation: 1.07, shoulder: 0.58, sharpen: 0.035,
      chroma: 0.16,  // chroma low-pass at 1080p: bilinear tap offset 2p of a [p, 1-2p, p] kernel (σ ≈ 0.67 px)
      // far-sea codec artefact strength (clip fidelity only: in explore mode it would show as pink specks at grazing angles)
      maroon: 0.0,
      // its trigger (relative units of the row mean luma; post/farsea.js): threshold, dash noise,
      // cat's-paw noise, left-right offset, transition half-width, trough contrast gain
      marT: 0.879, marBlob: 0.1, marPaws: 0.13, marLR: 0.012, marW: 0.012, marContrast: 1.0, marGust: 0.6,
      // Auto-exposure gain on the clip camera. main.js sets `exposure` from the measured sky-G
      // curve (inverted with a plain 2.2 gamma) or its schedule-driven model; through this grade's
      // shoulder the sky then swings only ~55 % of the reference (G/B slope 0.56 over 0-8.5 s,
      // far sea 0.54). Scaling the EV excursion about the clip's mean (EV 0) by 1.77 matches it
      // (fit on sky G and B; the lag / anti-correlation come with the replayed curve).
      aeGain: 0.65, aePivot: 0.0,
      // sun glare (lens veil, star, ghosts) and the metering pull-down when the sun is in view
      sunGlare: 1.0, sunVeil: 38.0, sunVeilCore: 0.0087, sunFrameVeil: 0.004, sunStar: 1.1,
      sunStarLen: 0.03, sunStarWidth: 0.0011, sunStarRot: 0.26, sunGhost: 0.035, sunEV: 0.6,
    };
    const qs = typeof location !== 'undefined' ? new URLSearchParams(location.search) : new URLSearchParams();
    // the far-sea artefact replays the reference's gusts only on the clip's camera and timeline
    this.clipReplay = !CONFIG.explore;
    this.aeOff = qs.get('ae') === '0';
    this.realtime = !qs.has('capture');   // captures: no temporal adaptation (deterministic frames)
    this._sunEV = 0; this._wall = 0;
    this.meter = new Exposure();
    this.frameDt = 0;
    this.camera = null;
    this.waterDepth=-1;
    this.pComposite = new FullscreenPass(makeShader(COMPOSITE, {
      uCrashVolume:{value:null},uCrashOn:{value:0},uScene:{value:null},uParticles:{value:null},uSceneDepth:{value:null},uEyeInverseVP:{value:new THREE.Matrix4()},uEye:{value:new THREE.Vector3()},
      uSurfaceProbe:shared.uSurfaceProbe||{value:null},uProbeOrigin:shared.uProbeOrigin||{value:new THREE.Vector2()},uUnderwaterOn:shared.uUnderwaterOn||{value:0}
    }));
    this.pDown = new FullscreenPass(makeShader(DOWN, { uSrc: { value: null }, uTexel: { value: new THREE.Vector2() }, uThreshold: { value: 1 }, uFirst: { value: 0 } }));
    this.pBlur = new FullscreenPass(makeShader(BLUR, { uSrc: { value: null }, uStep: { value: new THREE.Vector2() }, uW: { value: new THREE.Vector3(1, 0, 0) } }));
    this.pGrade = new FullscreenPass(makeShader(GRADE, {
      uHDR: { value: null }, uBloom: { value: null }, uBloomLow: { value: null },
      uExposure: { value: 1 }, uBloomStrength: { value: 0 }, uBloomTail: { value: 0 }, uSaturation: { value: 1 }, uShoulder: { value: 0.6 },
      uMeteredExposure:{value:this.meter.history.read.texture},uMeterOn:{value:0},
      uSunOn: { value: 0 }, uSunVis: { value: null }, uSunV: { value: new THREE.Vector3(0, 0, -1) }, uSunPx: { value: new THREE.Vector4() },
      uFocal: { value: 1 }, uSunTint: { value: new THREE.Vector3(1, 1, 1) }, uGlare: { value: new THREE.Vector4() }, uStar: { value: new THREE.Vector4(1, 1, 0, 0) },
      uSunEV: { value: 0 },
    }));
    this.pSun = new FullscreenPass(makeShader(SUNPROBE, {
      uSrc: { value: null }, uSunUv: { value: new THREE.Vector2() }, uRadUv: { value: new THREE.Vector2() }, uLref: { value: 800 },
    }));
    this.sunProbe = floatRT(1, 1, { type: THREE.HalfFloatType });
    this.pBand = new FullscreenPass(makeShader(BAND, {
      uSrc: { value: null }, uHorizon: { value: new THREE.Vector2(0.35, 0) }, uRowDV: { value: new THREE.Vector2(ROW_DV0, ROW_DV1) },
      uBins: { value: 16 }, uNU: { value: 96 }, uNV: { value: 4 },
    }));
    this.pFinal = new FullscreenPass(makeShader(ISP, {
      uSrc: { value: null }, uBand: { value: null }, uTexel: { value: new THREE.Vector2() },
      uSharpen: { value: 0 }, uChromaOff: { value: 0 }, uTime: { value: 0 },
      uMaroon: { value: 0 }, uInvViewProj: { value: new THREE.Matrix4() }, uCamPos: { value: new THREE.Vector3() },
      uHorizon: { value: new THREE.Vector2(0.35, 0) }, uRowDV: { value: new THREE.Vector2(ROW_DV0, ROW_DV1) },
      uGust: { value: 1 }, uContrast: { value: 1 }, uCVTarget: { value: 0.095 },
      uMar: { value: new THREE.Vector4() }, uMarW: { value: new THREE.Vector2() },
    }));
    this.band = floatRT(16, 1, { type: THREE.FloatType });
    this.pAA=new FullscreenPass(makeShader(FXAAShader.fragmentShader.replace('varying vec2 vUv;', 'in vec2 vUv;').replace('float _SubpixelBlending = 1.0;', 'float _SubpixelBlending = 0.65;'),THREE.UniformsUtils.clone(FXAAShader.uniforms)));
    this._m = new THREE.Matrix4();
    globalThis.__post = this; // debug handle (capture tool --eval)
    this._v = new THREE.Vector3();
    this.resize(w, h);
  }

  resize(w, h) {
    this.w = w; this.h = h;
    const hf = { type: THREE.HalfFloatType, filter: THREE.LinearFilter };
    [this.comp, this.graded, this.display, this.b0, this.b0t, this.b1, this.b1t].forEach((r) => r?.dispose());
    this.comp = floatRT(w, h, hf);
    this.graded = floatRT(w, h, hf);
    this.display=floatRT(w,h,{type:THREE.UnsignedByteType,filter:THREE.LinearFilter});
    this.pAA.material.uniforms.resolution.value.set(1/w,1/h);
    const w0 = Math.max(1, w >> 1), h0 = Math.max(1, h >> 1), w1 = Math.max(1, w0 >> 1), h1 = Math.max(1, h0 >> 1);
    this.b0 = floatRT(w0, h0, hf); this.b0t = floatRT(w0, h0, hf);
    this.b1 = floatRT(w1, h1, hf); this.b1t = floatRT(w1, h1, hf);
  }

  /** debug: this frame's far-sea row profile (8-bit luma per dv bin), gust factor, horizon line. */
  debugRows() {
    const buf = new Float32Array(16 * 4);
    this.renderer.readRenderTargetPixels(this.band, 0, 0, 16, 1, buf);
    const rows = [], cv = [], fu = this.pFinal.material.uniforms;
    for (let i = 0; i < 16; i++) {
      const m = buf[i * 4], m2 = buf[i * 4 + 1];
      rows.push(+(m * 255).toFixed(1)); cv.push(+(Math.sqrt(Math.max(m2 - m * m, 0)) / m).toFixed(4));
    }
    return { rows, cv, cvTarget: fu.uCVTarget.value, gust: fu.uGust.value, horizon: fu.uHorizon.value.toArray() };
  }

  // horizon line of the sea plane (y = 0 vanishing line) in top-left image coords: v = a + b u
  horizonLine(cam) {
    const p = this._v, e = cam.matrixWorld.elements;
    // horizontal forward of the camera
    let fx = -e[8], fz = -e[10];
    const n = Math.hypot(fx, fz) || 1; fx /= n; fz /= n;
    const vs = [];
    for (const s of [-0.3, 0.3]) {
      const dx = fx - s * fz, dz = fz + s * fx;
      p.set(cam.position.x + dx * 1e5, cam.position.y, cam.position.z + dz * 1e5).project(cam);
      vs.push([p.x * 0.5 + 0.5, 0.5 - p.y * 0.5]);
    }
    const [[u0, v0], [u1, v1]] = vs;
    const slope = Math.abs(u1 - u0) > 1e-6 ? (v1 - v0) / (u1 - u0) : 0;
    return [v0 - slope * u0, slope];
  }

  /** Exposure of this frame: main.js's AE value with its EV excursion scaled on the clip camera. */
  exposure() {
    const P = this.params, e = Math.max(P.exposure, 1e-6);
    if (this.aeOff || !this.clipReplay || P.aeGain === 1) return e;
    return 2 ** (P.aePivot + (Math.log2(e) - P.aePivot) * P.aeGain);
  }

  /**
   * Sun glare set-up: projects the sun into the frame, fades the glare in as the sun comes within
   * ~20° of the frame, runs the 1-fragment sun probe (visible disc fraction, read by the grade on
   * the GPU: no read-back) and eases the metering pull-down toward the sun's centre weighting.
   */
  sunSetup(cam, gu) {
    const P = this.params;
    gu.uSunVis.value = this.sunProbe.texture;
    gu.uSunOn.value = 0; gu.uSunEV.value = 0;
    let on = 0, target = 0;
    const v = this._sv || (this._sv = new THREE.Vector3());
    if (cam && P.sunGlare > 0 && this.waterDepth<.04) {
      const sh = globalThis.__params?.shared;
      v.copy(sh?.uSunDir?.value || this._sunDefault || (this._sunDefault = new THREE.Vector3(...sunDirection())));
      cam.updateMatrixWorld();
      v.transformDirection(cam.matrixWorldInverse);
      const pe = cam.projectionMatrix.elements;
      let tx = 0, ty = 0;
      if (v.z < -1e-3) {
        tx = v.x / -v.z; ty = v.y / -v.z;
        // angle of the sun outside the frame (0 inside)
        const ex = Math.max(Math.atan(Math.abs(tx)) - Math.atan(1 / pe[0]), 0);
        const ey = Math.max(Math.atan(Math.abs(ty)) - Math.atan(1 / pe[5]), 0);
        on = 1 - smooth01(0.09, 0.40, Math.hypot(ex, ey));
        // centre-weighted metering: full pull-down within ~6° of the centre, none beyond ~29°
        target = P.sunEV * on * (1 - smooth01(0.10, 0.50, Math.acos(Math.min(-v.z, 1))));
      }
      if (on > 0) {
        const xn = pe[0] * tx, yn = pe[5] * ty;
        const focal = pe[5] * this.h * 0.5;
        const sc = sh?.uSunColor?.value;
        const lum = sc ? 0.2126 * sc.x + 0.7152 * sc.y + 0.0722 * sc.z : 2.0;
        gu.uSunOn.value = on * P.sunGlare;
        gu.uSunV.value.copy(v);
        gu.uSunPx.value.set((xn * 0.5 + 0.5) * this.w, (yn * 0.5 + 0.5) * this.h, this.w * 0.5, this.h * 0.5);
        gu.uFocal.value = focal;
        if (sc) gu.uSunTint.value.set(sc.x / lum, sc.y / lum, sc.z / lum);
        gu.uGlare.value.set(P.sunVeil, P.sunVeilCore, P.sunFrameVeil, P.sunStar);
        gu.uStar.value.set(P.sunStarLen, Math.max(P.sunStarWidth, 0.75 / focal), P.sunStarRot, P.sunGhost);
        // probe the disc (0.265° radius, Sky.js: 400 x sun colour at the centre, limb-darkened)
        const su = this.pSun.material.uniforms, rpx = 0.00465 * focal;
        su.uSrc.value = this.comp.texture;
        su.uSunUv.value.set(xn * 0.5 + 0.5, yn * 0.5 + 0.5);
        su.uRadUv.value.set(rpx / this.w, rpx / this.h);
        su.uLref.value = 400 * lum;
        this.pSun.render(this.renderer, this.sunProbe);
      }
    }
    // the phone's metering eases in over ~0.3 s (interactive only; captures take the settled value)
    const now = typeof performance !== 'undefined' ? performance.now() * 1e-3 : 0;
    const dt = now - this._wall;
    this._wall = now;
    this._sunEV = (this.realtime && dt > 0 && dt < 0.5) ? this._sunEV + (target - this._sunEV) * (1 - Math.exp(-dt / 0.3)) : target;
    if (on > 0) gu.uSunEV.value = this.clipReplay ? this._sunEV : 0;
  }

  render(sceneTex, particleTex, time, target = null, camera = null, depthTex=null) {
    const r = this.renderer, P = this.params;
    const res = this.h / 1920;   // output size relative to the 1080x1920 the look was measured at
    const cam = camera || this.camera || globalThis.__scene?.camera || null;
    const cu = this.pComposite.material.uniforms;
    cu.uScene.value = sceneTex; cu.uParticles.value = particleTex;
    cu.uSceneDepth.value=depthTex;
    if(cam){cu.uEye.value.copy(cam.position);cu.uEyeInverseVP.value.multiplyMatrices(cam.matrixWorld,cam.projectionMatrixInverse);}
    cu.uCrashVolume.value=this.plumeTexture||sceneTex;cu.uCrashOn.value=this.plumeTexture?1:0;
    this.pComposite.render(r, this.comp);
    const meterTexture=this.meter.render(r,this.comp.texture,this.frameDt,!this.realtime);

    // bloom: half-res core + quarter-res tail, each Gaussian-blurred (round halos)
    const du = this.pDown.material.uniforms, bu = this.pBlur.material.uniforms;
    du.uSrc.value = this.comp.texture; du.uTexel.value.set(1 / this.w, 1 / this.h);
    du.uThreshold.value = P.bloomThreshold; du.uFirst.value = 1;
    this.pDown.render(r, this.b0);
    du.uSrc.value = this.b0.texture; du.uTexel.value.set(1 / this.b0.width, 1 / this.b0.height); du.uFirst.value = 0;
    this.pDown.render(r, this.b1);
    const blur = (rt, tmp, sigma) => {
      bu.uW.value.copy(gauss3(sigma));
      bu.uSrc.value = rt.texture; bu.uStep.value.set(1 / rt.width, 0); this.pBlur.render(r, tmp);
      bu.uSrc.value = tmp.texture; bu.uStep.value.set(0, 1 / rt.height); this.pBlur.render(r, rt);
    };
    // radii and strength are set for 1080x1920 (fitted to the clip's glint halos) and scale with the
    // output size (a glint pixel carries ~1/res more energy at smaller sizes)
    blur(this.b0, this.b0t, Math.max(P.bloomSigma * res, 0.35));
    blur(this.b1, this.b1t, Math.max(P.bloomTailSigma * res, 0.35));

    const gu = this.pGrade.material.uniforms;
    this.sunSetup(cam, gu);
    const airLens=1-smooth01(-.04,.04,this.waterDepth);
    gu.uSunOn.value*=airLens;gu.uSunEV.value*=airLens;
    gu.uHDR.value = this.comp.texture; gu.uBloom.value = this.b0.texture; gu.uBloomLow.value = this.b1.texture;
    gu.uExposure.value = this.exposure(); gu.uBloomStrength.value = P.bloom * res; gu.uBloomTail.value = P.bloomTail;
    gu.uMeteredExposure.value=meterTexture;gu.uMeterOn.value=!this.clipReplay&&!this.aeOff?1:0;
    gu.uSaturation.value = P.saturation; gu.uShoulder.value = P.shoulder;
    this.pGrade.render(r, this.graded);

    const fu = this.pFinal.material.uniforms;
    let maroon = cam ? P.maroon : 0;
    if (maroon > 0) {
      cam.updateMatrixWorld();
      const [a, b] = this.horizonLine(cam);
      const nu = this.pBand.material.uniforms;
      nu.uSrc.value = this.graded.texture; nu.uHorizon.value.set(a, b); fu.uHorizon.value.set(a, b);
      this.pBand.render(r, this.band);
      this._m.multiplyMatrices(cam.matrixWorld, cam.projectionMatrixInverse);
      fu.uInvViewProj.value.copy(this._m);
      fu.uCamPos.value.copy(cam.position);
      const w = this.clipReplay ? smooth01(-0.6, 0, time) * (1 - smooth01(8.5, 9.1, time)) : 0;
      fu.uGust.value = Math.pow(1 + (bandLumaRef(time) / BAND_LUMA_MEAN - 1) * w, P.marGust);
      fu.uContrast.value = P.marContrast;
      const cv = BAND_CV_MEAN + (bandCVRef(time) - BAND_CV_MEAN) * w;
      fu.uCVTarget.value = cv * (1 + 0.09 * Math.log2(Math.min(res, 1)));   // clip at 540p: 0.91x
      // the clip's dashes are ~4 x 1 px at 1080p; at lower output sizes they average away like the clip's do
      fu.uMar.value.set(P.marT + 0.037 * Math.log2(Math.min(this.h / 1920, 1)), P.marBlob, P.marPaws, P.marLR);
      fu.uMarW.value.set(P.marW, 1);
    }
    fu.uBand.value = this.band.texture;
    fu.uSrc.value = this.graded.texture;
    fu.uTexel.value.set(1 / this.w, 1 / this.h);
    fu.uSharpen.value = P.sharpen; fu.uChromaOff.value = Math.min(P.chroma * res, 0.6); fu.uTime.value = time;
    fu.uMaroon.value = maroon;
    this.pFinal.render(r, this.display);
    this.pAA.material.uniforms.tDiffuse.value=this.display.texture;
    this.pAA.render(r,target);
  }
}
