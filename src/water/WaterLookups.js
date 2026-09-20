import * as THREE from 'three';
import { bedProfileJS } from '../config.js';
import { buildTravel, TRAVEL_N } from './swellTravel.js';

// One sampler for three independent, unchanged 4096-entry float tables.
// Four rows give exact binary row centres, avoiding cross-table interpolation.
let cached;
export function waterLookups() {
  if (cached) return cached;
  const travel = buildTravel(), data = new Float32Array(TRAVEL_N * 4 * 4);
  data.set(travel.forward, 0);
  data.set(travel.inverse, TRAVEL_N * 4);
  for (let i = 0; i < TRAVEL_N; i++) data[(2 * TRAVEL_N + i) * 4] = bedProfileJS(-64 + 96 * i / (TRAVEL_N - 1));
  const texture = new THREE.DataTexture(data, TRAVEL_N, 4, THREE.RGBAFormat, THREE.FloatType);
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.colorSpace = THREE.NoColorSpace;
  texture.needsUpdate = true;
  cached = { texture, minTime: travel.minTime };
  return cached;
}

export const WATER_LOOKUP_GLSL = /* glsl */ `
#ifndef WATER_LOOKUP_DECLARED
#define WATER_LOOKUP_DECLARED
uniform sampler2D uWaterLookup;
#endif
`;
