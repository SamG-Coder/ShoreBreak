// Preserve the camera's aspect ratio while using 75% of the browser's area.
export const SCENE_AREA = 0.75;
export const SCENE_SCALE = Math.sqrt(SCENE_AREA);
export const PIXEL_BUDGET_SCALE = 0.90;
export const RESOLUTION_SCALES = Object.freeze([1, 1.5, 2]);

export function normalizeResolutionScale(value) {
  const scale = Number(value);
  return RESOLUTION_SCALES.includes(scale) ? scale : 1;
}

export function usesAdaptiveResolution(quality, resolutionScale) {
  // A deliberate supersampling choice must not be silently undone by Auto.
  return quality === 'auto' && normalizeResolutionScale(resolutionScale) === 1;
}

export function renderPixelRatio(width, height, { dpr, maxDpr, pixels, budgetScale = 1, dynamicScale = 1, resolutionScale = 1, maxDimension = Infinity, exact = false }) {
  if (exact) return 1;
  // Reserve 10% of the former cap; the smaller frame can spend the remaining
  // budget on crisper pixels. Device DPR and Auto's dynamic scale still apply.
  const base = Math.min(dpr, maxDpr, Math.sqrt(pixels * budgetScale / (width * height)));
  const scale = normalizeResolutionScale(resolutionScale);
  const ratio = Math.max(0.4, base * (scale > 1 ? 1 : dynamicScale)) * scale;
  // Apply the boost after the baseline DPR/budget caps so it actually raises
  // resolution on DPR=1 displays and capped presets. Preserve aspect ratio
  // if a device cannot allocate the requested texture/renderbuffer dimensions.
  return Math.min(ratio, maxDimension / Math.max(width, height));
}
