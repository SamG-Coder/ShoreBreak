import test from 'node:test';
import assert from 'node:assert/strict';
import { makeLipColumns, morphLipColumn, LIP_SNAP } from '../src/water/lipLod.js';

const columns = Array.from(Float32Array.from(makeLipColumns(40)));
function distance(a, b) {
  let j = 0, worst = 0;
  for (const x of a) {
    while (j + 1 < b.length && Math.abs(b[j + 1] - x) <= Math.abs(b[j] - x)) j++;
    worst = Math.max(worst, Math.abs(b[j] - x));
  }
  return worst;
}

test('lip columns merge before every recenter, on both sides of the player', () => {
  let worst = 0, oldWorst = 0;
  for (let n = -800; n < 800; n++) {
    const focus = (n + .5) * LIP_SNAP;
    const pair = [-1, 1].map(sign => columns.map(x => morphLipColumn(x, focus + sign * .000001))
      .filter(x => Math.abs(x - focus) < 32).sort((a, b) => a - b));
    worst = Math.max(worst, distance(pair[0], pair[1]), distance(pair[1], pair[0]));
    const old = [-1, 1].map(sign => columns.map(x => x + Math.round((focus + sign * .000001) / LIP_SNAP) * LIP_SNAP)
      .filter(x => Math.abs(x - focus) < 32).sort((a, b) => a - b));
    oldWorst = Math.max(oldWorst, distance(old[0], old[1]), distance(old[1], old[0]));
  }
  assert.ok(oldWorst > .15, 'Regression must detect the original column pop');
  assert.ok(worst < .00001, `Merged column discontinuity ${worst} m`);
});

test('lip morph preserves ordered columns and the original fine crest near the player', () => {
  for (let i = 0; i < 320; i++) {
    const focus = -34.24 + i * .001;
    const xs = columns.map(x => morphLipColumn(x, focus));
    for (let j = 1; j < xs.length; j++) assert.ok(xs[j] >= xs[j - 1] - 1e-5, 'No folded strip');
    for (let j = 0; j < xs.length; j++) {
      const fixed = columns[j] + Math.round(focus / LIP_SNAP) * LIP_SNAP;
      if (Math.abs(fixed - focus) < 1.2) assert.equal(xs[j], fixed);
    }
  }
});
