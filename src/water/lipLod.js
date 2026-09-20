// World-aligned columns stay stationary while the ribbon follows the player.
// Each fine band must merge into its parent before a snapped window can remove it.
export const LIP_SNAP = 0.32;
export const LIP_BANDS = [[0.02, 2.56], [0.04, 5.12], [0.08, 10.24], [0.16, 20.48]];
const MORPH = [[0.04, 1.2, 2.24], [0.08, 3.2, 4.8], [0.16, 6.8, 9.92], [0.32, 13.6, 20.16]];

export function makeLipColumns(halfWidth) {
  const xs = [0];
  let x = 0;
  for (const [step, radius] of [...LIP_BANDS, [LIP_SNAP, halfWidth]]) {
    while (x < radius - 1e-6) {
      x = Math.round((x + step) / step) * step;
      xs.push(x);
    }
  }
  return [...xs.slice(1).reverse().map(x => -x), ...xs];
}

export function morphLipColumn(localX, focusX) {
  let x = localX + Math.round(focusX / LIP_SNAP) * LIP_SNAP;
  const d = Math.abs(x - focusX);
  for (const [step, start, end] of MORPH) {
    const t = Math.max(0, Math.min(1, (d - start) / (end - start)));
    x += (Math.floor(x / step + 0.001) * step - x) * t * t * (3 - 2 * t);
  }
  return x;
}

// Shared by the front sheet, barrel tint and back-depth pass. Evaluate the whole
// cross-section at the morphed x so its position, normal and optics agree.
export const LIP_MORPH_GLSL = /* glsl */ `
float lipColumn(float x) {
  if (uLipX0.y > 1000.0) return x; // fixed clip geometry
  float d = abs(x - uLipX0.z);
  ${MORPH.map(([step, start, end]) => `x = mix(x, floor(x / ${step.toFixed(4)} + 0.001) * ${step.toFixed(4)}, smoothstep(${start.toFixed(4)}, ${end.toFixed(4)}, d));`).join('\n  ')}
  return x;
}
`;
