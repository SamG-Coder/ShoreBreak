// Hash-without-Sine routines: Copyright (c) 2014 David Hoskins, MIT.
// See licenses/Hash-without-Sine-MIT.txt and THIRD_PARTY_NOTICES.md.
// Shared GLSL chunks. Every shader in the project composes from these so the
// bathymetry, sky model and optics are identical everywhere.

export const NOISE = /* glsl */ `
// ---------------------------------------------------------------- hashing
float hash11(float p) { p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec2  hash22(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }
vec3  hash32(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yxz + 33.33); return fract((p3.xxy + p3.yzz) * p3.zyx); }
vec3  hash33(vec3 p3) { p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yxz + 33.33); return fract((p3.xxy + p3.yxx) * p3.zyx); }

// ---------------------------------------------------------------- value / gradient noise
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float a = hash12(i), b = hash12(i + vec2(1, 0)), c = hash12(i + vec2(0, 1)), d = hash12(i + vec2(1, 1));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
// gradient noise with analytic derivative: returns (value, d/dx, d/dy), value in ~[-1,1]
vec3 gnoised(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  vec2 ga = hash22(i) * 2.0 - 1.0, gb = hash22(i + vec2(1, 0)) * 2.0 - 1.0;
  vec2 gc = hash22(i + vec2(0, 1)) * 2.0 - 1.0, gd = hash22(i + vec2(1, 1)) * 2.0 - 1.0;
  float va = dot(ga, f), vb = dot(gb, f - vec2(1, 0)), vc = dot(gc, f - vec2(0, 1)), vd = dot(gd, f - vec2(1, 1));
  float v = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd);
  vec2 d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
         + du * (u.yx * (va - vb - vc + vd) + vec2(vb, vc) - va);
  return vec3(v * 1.6, d * 1.6);
}
float gnoise(vec2 p) { return gnoised(p).x; }
float fbm(vec2 p, int oct) {
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 8; i++) { if (i >= oct) break; s += a * gnoise(p); p = mat2(1.6, 1.2, -1.2, 1.6) * p; a *= 0.5; }
  return s;
}
// 3D value noise (for time-evolving fields)
float vnoise3(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);
  float n000 = hash33(i).x, n100 = hash33(i + vec3(1,0,0)).x, n010 = hash33(i + vec3(0,1,0)).x, n110 = hash33(i + vec3(1,1,0)).x;
  float n001 = hash33(i + vec3(0,0,1)).x, n101 = hash33(i + vec3(1,0,1)).x, n011 = hash33(i + vec3(0,1,1)).x, n111 = hash33(i + vec3(1,1,1)).x;
  return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y), mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}
// Worley / cellular: returns (F1, F2, cell id hash)
vec3 worley(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  float f1 = 8.0, f2 = 8.0, id = 0.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    vec2 g = vec2(x, y);
    vec2 o = hash22(i + g);
    vec2 r = g + o - f;
    float d = dot(r, r);
    if (d < f1) { f2 = f1; f1 = d; id = hash12(i + g); } else if (d < f2) { f2 = d; }
  }
  return vec3(sqrt(f1), sqrt(f2), id);
}
float sat(float x) { return clamp(x, 0.0, 1.0); }
float sech2(float x) { float e = exp(-2.0 * abs(x)); float s = 2.0 * exp(-abs(x)) / (1.0 + e); return s * s; }
float remap(float v, float a, float b) { return clamp((v - a) / (b - a), 0.0, 1.0); }
`;

// Bathymetry shared by beach geometry, SWE bed, water depth tests and caustics.
export const BED = /* glsl */ `
// bedProfile(z) comes from glslDefines() (config.js). Small along-shore relief on the
// swash face: beach cusps (~1.2 m) and uneven pebble ridges make the swash front lobed.
float bedRelief(vec2 xz) {
  float face = smoothstep(-1.2, 0.2, xz.y) * smoothstep(5.0, 2.5, xz.y);
  float cusp = 0.003 * sin(xz.x * 5.2 + 0.8 * sin(xz.x * 1.7)) + 0.0025 * sin(xz.x * 14.5 + 1.3);   // the SWE has its own bed; this is render drape only
  float lumps = 0.010 * gnoise(xz * 1.3 + 3.1) + 0.005 * gnoise(xz * 4.1 - 1.3);
  return (cusp + lumps) * face;
}
float bedHeight(vec2 xz) { return bedProfile(xz.y) + bedRelief(xz); }
uniform sampler2D uRockBed;
uniform vec4 uRockDomain;
float rockBedHeight(vec2 xz) {
  if(uRockDomain.z <= 0.0) return -100.0;
  vec2 uv=(xz-uRockDomain.xy)/uRockDomain.zw;
  if(min(uv.x,uv.y)<0.0 || max(uv.x,uv.y)>1.0) return -100.0;
  return textureLod(uRockBed,uv,0.0).r;
}
float hydraulicBedHeight(vec2 xz) {return max(bedHeight(xz),rockBedHeight(xz));}
float localRockShadow(vec3 P,vec3 L){
  if(abs(P.x)>65.0 || abs(P.z)>3.3 || P.y>.65) return 1.0;
  float shade=1.0;
  for(int i=0;i<4;i++){
    float h=.045+.042*float((i+1)*(i+1));
    vec3 q=P+L*(h/max(L.y,.15));
    shade=min(shade,smoothstep(-.018,.035,q.y-rockBedHeight(q.xz)));
  }
  return shade;
}
`;

// Sky: elevation LUT measured from the clip (time-median over 86 frames), authored in
// pre-grade scene-linear (the post applies the phone's x1.35 saturation), plus the
// measured left-right brightness gradient and an extrapolation above the frame top
// (needed for reflections on steep faces).
export const SKY = /* glsl */ `
#define SKY_N 18
uniform vec4 uSkyLut[SKY_N];  // (elevation deg, r, g, b)
uniform vec3 uSunDir;         // normalized, toward sun
uniform vec3 uSunColor;       // sun "irradiance" (Lambert: albedo * uSunColor * NoL / PI)
uniform vec3 uSkyAmb;         // sky ambient radiance for a white upward-facing Lambertian surface

vec3 skyRadiance(vec3 d) {
  float el = degrees(asin(clamp(d.y, -1.0, 1.0)));
  float e = max(el, 0.0);
  vec3 c = uSkyLut[0].yzw;
  for (int i = 1; i < SKY_N; i++) {
    vec4 a = uSkyLut[i - 1], b = uSkyLut[i];
    if (e >= a.x) c = mix(a.yzw, b.yzw, clamp((e - a.x) / (b.x - a.x), 0.0, 1.0));
  }
  // brighter toward +x (sun side), measured +0.29 %/deg at the horizon, +0.43 %/deg at 22 deg
  float az = degrees(atan(d.x, -d.z));
  float k = mix(0.0029, 0.0043, clamp(e / 22.0, 0.0, 1.0));
  c *= 1.0 + k * clamp(az, -80.0, 80.0);
  // faint Mie glow around the (off-screen) sun, for reflections
  float mu = max(dot(d, uSunDir), 0.0);
  c += uSunColor * (0.004 * pow(mu, 12.0) + 0.02 * pow(mu, 200.0));
  // below the horizon: only reachable by reflections off steep facets -> hazy sea tone
  c = mix(vec3(0.05, 0.20, 0.26), c, smoothstep(-6.0, 0.0, el));
  return c;
}
vec3 sunDisk(vec3 d) {
  float mu = dot(d, uSunDir);
  return uSunColor * 400.0 * smoothstep(0.99996, 0.99999, mu);
}
// diffuse sky irradiance / PI for normal N (upper hemisphere dominated, slight ground bounce)
vec3 skyAmbient(vec3 N) {
  vec3 ground = vec3(0.13, 0.12, 0.11);
  return mix(ground, uSkyAmb, 0.62 + 0.38 * N.y);
}
`;

// Water optics helpers.
export const OPTICS = /* glsl */ `
float fresnelSchlick(float cosT, float f0) { return f0 + (1.0 - f0) * pow(1.0 - clamp(cosT, 0.0, 1.0), 5.0); }
// Exact unpolarized Fresnel for air->water (n=1.333), better at grazing angles than Schlick
float fresnelWater(float cosI) {
  cosI = clamp(cosI, 0.0, 1.0);
  float eta = 1.0 / 1.333;
  float cosT = sqrt(max(1.0 - eta * eta * (1.0 - cosI * cosI), 0.0));
  float rs = (cosI - 1.333 * cosT) / (cosI + 1.333 * cosT);
  float rp = (1.333 * cosI - cosT) / (1.333 * cosI + cosT);
  return clamp(0.5 * (rs * rs + rp * rp), 0.0, 1.0);
}
float D_GGX(float NoH, float a) { float a2 = a * a; float d = NoH * NoH * (a2 - 1.0) + 1.0; return a2 / (3.14159265 * d * d); }
float V_SmithGGXCorrelated(float NoV, float NoL, float a) {
  float a2 = a * a;
  float gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
  float gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-5);
}
// Beer-Lambert extinction of clear, slightly green-tinted Mediterranean water (1/m)
const vec3 WATER_SIGMA_A = vec3(0.46, 0.075, 0.062);
const vec3 WATER_SIGMA_S = vec3(0.020, 0.030, 0.032);
`;

// Camera helpers for screen-space sampling (linearizing perspective depth)
export const DEPTH = /* glsl */ `
uniform float uNear;
uniform float uFar;
float linearizeDepth(float d) {
  float z = d * 2.0 - 1.0;
  return (2.0 * uNear * uFar) / (uFar + uNear - z * (uFar - uNear));
}
`;
