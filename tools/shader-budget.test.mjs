import test from 'node:test';
import assert from 'node:assert/strict';
import { checkShaderBudget } from '../src/core/shaderBudget.js';
import { waterLookups } from '../src/water/WaterLookups.js';
import { buildTravel, TRAVEL_N } from '../src/water/swellTravel.js';
import { bedProfileJS } from '../src/config.js';
import { OceanFFT } from '../src/water/OceanFFT.js';
import { ExploreWaterMesh } from '../src/water/ExploreMesh.js';

const mockGL = uniforms => ({
  MAX_TEXTURE_IMAGE_UNITS: 1, MAX_VERTEX_TEXTURE_IMAGE_UNITS: 2, MAX_COMBINED_TEXTURE_IMAGE_UNITS: 3,
  LINK_STATUS: 4, ACTIVE_UNIFORMS: 5,
  getParameter: n => n === 3 ? 32 : 16,
  getProgramParameter: (_, name) => name === 4 ? true : uniforms.length,
  getActiveUniform: (_, i) => uniforms[i],
});
test('startup counts sampler arrays and rejects a program over the portable budget', () => {
  const u = [{ name: 'ocean', type: 36289, size: 2 }, { name: 'other', type: 35678, size: 14 }];
  assert.equal(checkShaderBudget(mockGL(u), [{ program: 1 }]).maximum, 16);
  u[1].size = 15;
  assert.throws(() => checkShaderBudget(mockGL(u), [{ program: 1 }]), /17\/16/);
});
test('packed water lookup rows preserve every original float and use exact row centres', () => {
  const lookup = waterLookups(), travel = buildTravel(), data = lookup.texture.image.data;
  assert.equal(lookup.texture.image.width, 4096); assert.equal(lookup.texture.image.height, 4);
  assert.deepEqual(data.slice(0, TRAVEL_N * 4), travel.forward);
  assert.deepEqual(data.slice(TRAVEL_N * 4, TRAVEL_N * 8), travel.inverse);
  for (let i = 0; i < TRAVEL_N; i++) assert.equal(data[(TRAVEL_N * 2 + i) * 4], Math.fround(bedProfileJS(-64 + 96 * i / (TRAVEL_N - 1))));
  assert.equal(lookup, waterLookups());
});
test('FFT draws every layer once and generates mipmaps after the final layer', () => {
  const ocean = new OceanFFT(), layers = []; let target = null, layer = 0, draws = 0;
  const renderer = {
    getRenderTarget: () => target, getActiveCubeFace: () => layer, getActiveMipmapLevel: () => 0,
    initRenderTarget: rt => { assert.ok(rt.textures.every(t => t.generateMipmaps && t.isDataArrayTexture && t.image.depth === 3)); },
    setRenderTarget: (rt, l = 0) => { target = rt; layer = l; },
    render: () => { draws++; if (target === ocean.cascades) layers.push([layer, ...target.textures.map(t => t.generateMipmaps)]); },
  };
  ocean.update(renderer, 1);
  assert.deepEqual(layers, [[0, false, false], [1, false, false], [2, true, true]]);
  assert.equal(draws, 20); assert.equal(ocean.cascades.textures.length, 2);
  ocean.update(renderer, 1); assert.equal(draws, 20);
});
test('mesh lookup packing preserves column, row and fan values during movement', () => {
  const mesh = new ExploreWaterMesh();
  for (const [x, z, y] of [[-34, 8, 2.7], [12.7, -4, .6], [59, 36, 5.2]]) {
    mesh.update(x, z, y, true); const data = mesh.lookupTex.image.data;
    for (const [texture, offset] of [[mesh.colTex, 0], [mesh.rowTex, mesh.lookupRowOffset], [mesh.fanTex, mesh.lookupFanOffset]]) {
      assert.deepEqual(data.slice(offset * 4, offset * 4 + texture.image.data.length), texture.image.data);
    }
  }
});
