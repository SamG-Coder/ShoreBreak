import { CONFIG } from '../config.js';
import { SKY_MODEL } from '../sky/skyModel.js';

// Aerial perspective of the land (beach far along the shore, promenade, city, hills): the land
// fades into the sky's own horizon colour at its azimuth (explore: the all-azimuth sky model of
// the dome, sky/skyModel.js; clip: skyRadiance). Requires the SKY chunk (common.js).
//   landHazeColor(dir)  horizon sky colour toward the horizontal direction of dir
//   landHaze(d)         haze share at distance d (m): clear within 150 m, e-fold 9 km
export const HAZE_GLSL = (CONFIG.explore ? SKY_MODEL : '') + /* glsl */ `
vec3 landHazeColor(vec3 dir) {
  vec3 h = normalize(vec3(dir.x, 0.0, dir.z) + vec3(0.0, 1e-4, 0.0));
${CONFIG.explore ? '  return skyProfile(0.2) * skyAzimuthGain(vec3(h.x, 0.0035, h.z));' : '  return skyRadiance(h);'}
}
float landHaze(float d) { return 1.0 - exp(-max(d - 60.0, 0.0) / 7400.0 - 0.055 * (1.0 - exp(-max(d, 0.0) / 480.0))); }
`;
