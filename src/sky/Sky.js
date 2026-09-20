import * as THREE from 'three';
import { CONFIG, glslDefines } from '../config.js';
import { NOISE, SKY } from '../glsl/common.js';
import { SKY_MODEL } from './skyModel.js';

// Sky dome drawn at infinity: analytic gradient fitted to the clip, sun glow, the faint wispy
// contrail and the small airliner of the reference (world directions measured on the clip's
// frames through the camera model: sky analysis / critic sky_hp).
const VERT = /* glsl */ `
out vec3 vDir;
void main() {
  vDir = position;
  vec4 p = projectionMatrix * mat4(mat3(viewMatrix)) * vec4(position, 1.0);
  gl_Position = p.xyww;
}`;

const FRAG = /* glsl */ `
uniform float uTime;
uniform vec3 uDeepColor;
in vec3 vDir;
// extra per-channel left/right gradient on top of skyRadiance's scalar one (the sky is redder
// and brighter toward the sun side, deeper blue on the left: measured at 1080, pre-grade)
vec3 skyAzimuthTint(vec3 d) {
  float el = degrees(asin(clamp(d.y, -1.0, 1.0)));
  float az = clamp(degrees(atan(d.x, -d.z)), -80.0, 80.0);
  float f = clamp(el / 22.0, 0.0, 1.0);
  return 1.0 + vec3(mix(0.003, 0.005, f), mix(0.0, 0.0011, f), mix(0.0, -0.0005, f)) * az;
}
void main() {
  vec3 d = normalize(vDir);
#ifdef OPT_EXPLORE
  // every direction (skyModel.js): the measured elevation profile, its smooth dependence on the
  // angle to the sun (brighter / whiter on the sun side, deeper blue behind the beach), the
  // aureole and the sun itself
  float el = degrees(asin(clamp(d.y, -1.0, 1.0)));
  float g = acos(clamp(dot(d, uSunDir), -1.0, 1.0));
  vec3 c = highCloud(d, skyProfile(el) * skyAzimuthGain(d)) + skyAureole(g);
  // sun disc (0.53 deg, limb-darkened): far beyond the sensor's range, it clips and the post blooms it
  const float RS = 0.00465;
  float pxA = max(fwidth(g), 1e-6);
  float disc = 1.0 - smoothstep(RS - pxA, RS + pxA, g);
  float mu = sqrt(max(1.0 - (g / RS) * (g / RS), 0.0));
  c += uSunColor * 400.0 * disc * (0.45 + 0.55 * mu);
  // below the horizon (seen only where no geometry covers it): toward the sea, past the far edge
  // of the sea mesh (a fraction of a pixel), the sea's own grazing tone, so the horizon line stays
  // crisp at every azimuth; toward the land (along and behind the beach, beyond the modelled
  // ground) distant pebbles fading into the horizon haze
  vec3 hzGain = skyAzimuthGain(vec3(d.x, 0.0, d.z));
  vec3 seaHz = (uDeepColor * 1.1 + vec3(0.02, 0.05, 0.06)) * mix(vec3(1.0), hzGain, 0.5);
  vec3 landHz = mix(vec3(0.30, 0.27, 0.23), skyProfile(0.0) * hzGain, exp(min(el, 0.0) / 1.5));   // (min: exp overflows up the dome)
  vec2 hdir = d.xz / max(length(d.xz), 1e-5);
  vec3 below = mix(seaHz, landHz, smoothstep(-0.03, 0.06, hdir.y));
  c = mix(below, c, smoothstep(-0.3, 0.0, el));
#else
  vec3 c = (skyRadiance(d) + sunDisk(d)) * skyAzimuthTint(d);
#endif
  vec2 q = vec2(atan(d.x, -d.z), asin(clamp(d.y, -1.0, 1.0)));   // (azimuth, elevation) rad
  float px = max(fwidth(q.y), 1e-5);                                // angular pixel size
  // contrail: a thin ragged wisp from az 0.13 (el 0.083) up to a diffuse brighter head at
  // az ~0.30 (el 0.117), 3-6 codes above the sky
  float along = clamp((q.x - 0.13) / 0.18, 0.0, 1.0);
  float wob = 0.0012 * gnoise(vec2(q.x * 90.0, 3.1));
  float off = q.y - (0.083 + 0.2 * (q.x - 0.13)) - wob;
  float wid = max(mix(0.0009, 0.0026, along * along), px * 0.7);
  float streak = exp(-0.5 * off * off / (wid * wid)) * smoothstep(0.115, 0.15, q.x) * smoothstep(0.325, 0.30, q.x);
  streak *= clamp(0.62 + 0.4 * gnoise(vec2(q.x * 140.0, 7.0)) + 0.25 * gnoise(vec2(q.x * 420.0, 1.3)), 0.0, 1.2);
  vec2 hd = (q - vec2(0.300, 0.1175)) / vec2(0.011, 0.0042);
  streak = streak * mix(0.55, 1.0, along) + 0.9 * exp(-dot(hd, hd)) * (0.8 + 0.2 * gnoise(q * 600.0));
  c += vec3(0.013, 0.014, 0.015) * streak;
  // the airliner: a dark ~6 x 2.5 px (1080) silhouette crossing right to left
  vec2 ap = vec2(0.3228 - 0.01594 * (uTime - 0.5), 0.2111 - 0.00053 * (uTime - 0.5));
  vec2 a = (q - ap) / vec2(0.0021, 0.0008);
  float body = 1.0 - smoothstep(0.7, 1.0 + px / 0.0008, length(a));
  vec2 wg = (q - ap - vec2(0.0003, 0.0)) / vec2(0.0007, 0.0016);   // wings
  body = max(body, 1.0 - smoothstep(0.6, 1.0 + px / 0.0007, length(wg)));
  c = mix(c, vec3(0.05, 0.085, 0.12), body * 0.9 * step(-2.0, uTime) * (1.0 - step(40.0, uTime)));
  // subtle large-scale mottling so the gradient isn't synthetic-flat
#ifdef OPT_EXPLORE
  c *= 1.0 + 0.012 * (vnoise3(d * vec3(10.0, 14.0, 10.0) + 5.0) - 0.5);   // (seamless on the sphere)
#else
  c *= 1.0 + 0.012 * (vnoise(q * vec2(9.0, 14.0)) - 0.5);
#endif
  gl_FragColor = vec4(c, 1.0);
}`;

export class Sky {
  constructor(shared) {
    const g = new THREE.SphereGeometry(1, 64, 32);
    this.material = new THREE.ShaderMaterial({
      uniforms: shared,
      vertexShader: VERT,
      fragmentShader: glslDefines() + (CONFIG.explore ? '#define OPT_EXPLORE\n' : '') + NOISE + SKY + (CONFIG.explore ? SKY_MODEL : '') + FRAG,
      side: THREE.BackSide,
      depthWrite: false,
    });
    this.mesh = new THREE.Mesh(g, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = -1000;
  }
}
