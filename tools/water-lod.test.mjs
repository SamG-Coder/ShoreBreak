import test from 'node:test';
import assert from 'node:assert/strict';
import { ExploreWaterMesh } from '../src/water/ExploreMesh.js';

test('rendered ocean columns stay continuous through walking and changing eye height', () => {
  const mesh = new ExploreWaterMesh();
  const sample = (x, z, y) => {
    mesh.update(x, z, y, true);
    // Read the actual uploaded float table, including columns that cross tiers.
    return mesh.blocks.flatMap(B => Array.from({ length: B.n }, (_, c) => mesh.colTex.image.data[(B.col0 + c) * 4])).sort((a, b) => a - b);
  };
  const distance = (a, b, focus) => {
    let j = 0, worst = 0;
    for (const x of a) {
      if (Math.abs(x - focus) > 70) continue;
      while (j + 1 < b.length && Math.abs(b[j + 1] - x) <= Math.abs(b[j] - x)) j++;
      worst = Math.max(worst, Math.abs(b[j] - x));
    }
    return worst;
  };
  for (const [z, y] of [[8, 2.7], [2.4, 1.8], [-4, .65], [-7, -.52], [20, 4]]) {
    for (let i = 0; i < 500; i++) {
      const x = -36 + i * .008;
      const a = sample(x - .000001, z, y), b = sample(x + .000001, z, y);
      assert.ok(Math.max(distance(a, b, x), distance(b, a, x)) < .00003, `Column jump near ${x}, ${z}`);
    }
  }
  // Cross fine/coarse density changes while crouching and surfacing, without
  // removing points that still contribute to the rendered surface.
  for (let i = 0; i < 600; i++) {
    const y = .28 + i * .004;
    const a = sample(-34, 2.4, y - .000001), b = sample(-34, 2.4, y + .000001);
    assert.ok(Math.max(distance(a, b, -34), distance(b, a, -34)) < .00003, `Height transition at ${y}`);
  }
});
