import { bedProfileJS } from '../config.js';

// Finite-depth dispersion and ray travel, baked once. The long swell is separate
// from the short FFT wind sea. Units: metres, seconds, radians.
export const SWELL_PERIOD = 4.4;
export const TRAVEL_N = 4096;
export const TRAVEL_FAR = -900;
export const TRAVEL_NEAR = -1;
export function waveSpeed(depth, period = SWELL_PERIOD) {
  const d = Math.max(depth, .08), w = 2 * Math.PI / period;
  let k = Math.max(w * w / 9.81, w / Math.sqrt(9.81 * d));
  for (let i = 0; i < 10; i++) {
    const kd = k * d, th = Math.tanh(kd);
    k -= (9.81 * k * th - w * w) / (9.81 * (th + kd * (1 - th * th)));
  }
  const c = w / k, kd = k * d;
  return { k, c, cg: .5 * c * (1 + (kd < 20 ? 2 * kd / Math.sinh(2 * kd) : 0)) };
}
export function buildTravel() {
  const forward = new Float32Array(TRAVEL_N * 4), inverse = new Float32Array(TRAVEL_N * 4);
  const dz = (TRAVEL_NEAR - TRAVEL_FAR) / (TRAVEL_N - 1);
  let time = 0, previous;
  for (let i = TRAVEL_N - 1; i >= 0; i--) {
    const z = TRAVEL_FAR + i * dz, p = waveSpeed(-bedProfileJS(z));
    if (previous) time -= dz * .5 * (1 / p.c + 1 / previous.c);
    forward.set([time, p.c, p.cg, z], i * 4); previous = p;
  }
  const minTime = forward[0]; let j = 0;
  for (let i = 0; i < TRAVEL_N; i++) {
    const t = minTime * (1 - i / (TRAVEL_N - 1));
    while (j < TRAVEL_N - 2 && forward[(j + 1) * 4] < t) j++;
    const a = forward[j * 4], b = forward[(j + 1) * 4];
    const f = Math.max(0, Math.min(1, (t - a) / (b - a)));
    for (let k = 0; k < 4; k++) inverse[i * 4 + k] = forward[j * 4 + k] * (1 - f) + forward[(j + 1) * 4 + k] * f;
  }
  return { forward, inverse, minTime };
}
