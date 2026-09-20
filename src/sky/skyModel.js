// Explore-mode sky model (all azimuths), shared by the sky dome (Sky.js) and the water's sky
// reflections (glsl/water.js skyRefl). Requires the SKY chunk (common.js: uSkyLut, uSunDir, uSunColor).
//
// The clip measured the sky toward the sea only: an elevation profile (uSkyLut, az ~ 0) and a
// left/right gradient there (+0.29 %/deg at the horizon, +0.43 %/deg at 22 deg; redder toward the
// sun side). Around the whole horizon the sky is extended with a smooth function of the angle to
// the sun (aerosol forward scattering): h(g) = 1 + A exp(-g / g0), normalised so that it is exactly
// 1 at azimuth 0 (the measured profile is kept) and fitted so that its slope there reproduces the
// measured gradients (A = 4.05, g0 = 0.7 rad): the sun side is brighter and whiter, the anti-sun
// side (behind the beach) deeper blue and ~25-30 % darker. Per-channel exponents (R steeper, B
// flatter) reproduce the measured per-channel tint. No azimuth wrap anywhere: no seam behind.
export const SKY_MODEL = /* glsl */ `
// Optically thin high cirrus. Direction-space coordinates are seamless, shared by
// the sky dome and water reflection, with a low-contrast veil instead of cloud blobs.
float highCloudAmount(vec3 d) {
  float h = max(d.y, 0.09);
  vec2 q = d.xz / h;
  q = mat2(0.92, 0.39, -0.39, 0.92) * q;
  float broad = vnoise(q * vec2(0.38, 0.9) + 31.6);
  float strands = vnoise(q * vec2(0.7, 7.0) + vec2(4.1, broad * 2.0));
  float fine = vnoise(q * vec2(2.1, 21.0) + broad * 3.0);
  float density = smoothstep(0.57, 0.82, broad) * smoothstep(0.24, 0.76, strands * 0.75 + fine * 0.25);
  return density * 0.095 * smoothstep(0.08, 0.22, d.y);
}
vec3 highCloud(vec3 d, vec3 sky) {
  float a = highCloudAmount(d);
  vec3 cloud = vec3(0.68, 0.73, 0.75) + uSunColor * 0.045 * pow(max(dot(d, uSunDir), 0.0), 6.0);
  return mix(sky, cloud, a);
}

float skySunLobe(float g) { return 1.0 + 4.05 * exp(-g / 0.7); }
// Relative sky radiance (rgb) of direction d with respect to the measured profile at azimuth 0.
vec3 skyAzimuthGain(vec3 d) {
  float sy = clamp(d.y, 0.0, 1.0);
  float ch = sqrt(max(1.0 - sy * sy, 0.0));
  vec2 hz = d.xz / max(length(d.xz), 1e-5);
  vec3 dd = vec3(hz.x * ch, sy, hz.y * ch);
  float g = acos(clamp(dot(dd, uSunDir), -1.0, 1.0));
  float g0 = acos(clamp(sy * uSunDir.y - ch * uSunDir.z, -1.0, 1.0));   // same elevation, azimuth 0 (-z)
  float F = skySunLobe(g) / skySunLobe(g0);
  float fe = clamp(degrees(asin(sy)) / 22.0, 0.0, 1.0);
  return pow(vec3(F), mix(vec3(2.03, 1.0, 1.0), vec3(2.16, 1.26, 0.88), fe));
}
// Measured elevation profile (piecewise linear in elevation, as skyRadiance in common.js).
vec3 skyProfile(float elDeg) {
  float e = max(elDeg, 0.0);
  vec3 c = uSkyLut[0].yzw;
  for (int i = 1; i < SKY_N; i++) {
    vec4 a = uSkyLut[i - 1], b = uSkyLut[i];
    if (e >= a.x) c = mix(a.yzw, b.yzw, clamp((e - a.x) / (b.x - a.x), 0.0, 1.0));
  }
  return c;
}
// Sunlight scattered by the aerosols in the forward peak (aureole, whitish): within ~10 deg of the
// sun the sky brightens steeply and clips on the phone (the inner term: the ~2 deg blown-out glare
// of a small sensor pointed near the sun).
vec3 skyAureole(float g) {
  return uSunColor * (1.6 * exp(-g / 0.011) + 0.30 * exp(-g / 0.035) + 0.05 * exp(-g / 0.12) + 0.012 * exp(-g / 0.35));
}
`;
