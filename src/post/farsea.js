// Measured far-sea data for the codec/ISP "maroon dash" artefact (post/Post.js, ISP pass).
//
// In the reference clip the dark troughs of far-sea wind ripples turn mauve/maroon
// (sRGB ≈ 82,89,109 against 4,129,147 water). This is a conversion artefact, not a physical colour:
// the phone's HDR→SDR/encoder chain collapses the (very negative) Cr of dark saturated cyan
// toward neutral. Measured on correctly decoded BT.709 1080p frames, far-sea band
// dv 0.015–0.075 below the horizon:
//   luma Y' (8-bit)   median Cr (8-bit)   P(Cr > −26, i.e. "maroon")
//      64–72              −14                 70 %
//      72–80              −18                 63 %
//      80–88              −30                 40–45 %
//      88–96              −50                 19 %
//      96–104             −62                 3 %
//     104–112             −67                 0.2 %
//      ≥112               −70                 0 %
// So it is a per-pixel function of display luma (with a ~10-code random spread that forms
// 4×1 px horizontal dashes at 1080p), clustered in cat's-paw patches, 2–8× denser on the left.
// Its coverage over time is driven by how dark the far-sea band is (r = −0.94 against band
// brightness): wind gusts darken the band at t ≈ 1.2–2.6 s and 5.4–6.4 s.

/** Reference far-sea band mean luma Y' (8-bit, BT.709 decode), dv 0.015–0.075, u 0.05–0.95, every 0.1 s from t = 0. */
export const BAND_LUMA_REF = [
  109.9, 109.8, 109.2, 109.3, 109.1, 109.4, 109.2, 109.6, 109.5, 109.3,
  108.4, 107.7, 107.0, 105.9, 104.7, 104.2, 103.3, 102.9, 102.6, 102.4,
  102.0, 101.9, 101.8, 102.8, 103.5, 104.7, 105.7, 107.5, 108.5, 110.0,
  110.9, 111.9, 111.8, 112.4, 112.4, 112.4, 111.9, 111.7, 111.0, 110.1,
  109.8, 109.6, 109.7, 109.6, 110.0, 110.7, 111.1, 111.2, 111.1, 110.9,
  110.4, 110.0, 109.4, 108.7, 107.7, 106.7, 105.4, 104.8, 104.3, 104.5,
  104.8, 105.9, 107.5, 108.9, 109.9, 110.8, 111.6, 112.8, 113.9, 114.6,
  115.4, 115.8, 116.6, 117.0, 117.8, 118.2, 118.7, 118.8, 119.0, 118.7,
  118.1, 117.1, 116.5, 115.8, 115.5, 114.7,
];

/**
 * Reference row profile below the horizon: mean luma of 16 bins of dv (0–0.12, step 0.0075, u 0.05–0.95)
 * divided by the band mean (dv 0.015–0.075); time-mean over 86 frames (SD ≤ 0.02 for dv < 0.08).
 */
export const ROW_PROFILE_REF = [0.8823, 0.8879, 0.9276, 0.9595, 0.9880, 0.9982, 1.0222, 1.0332, 1.0286, 1.0428, 1.0568, 1.1154, 1.2137, 1.3029, 1.3440, 1.3622];
export const ROW_DV0 = 0.0, ROW_DV1 = 0.12;

export const BAND_LUMA_MEAN = BAND_LUMA_REF.reduce((a, b) => a + b, 0) / BAND_LUMA_REF.length;

/**
 * Reference trough contrast: mean over the band's row bins of (luma SD / row mean), 1080p,
 * every 0.1 s (median-5 + box-3 smoothed; first-frame I-frame spike damped). At 540p the
 * clip's value is 0.91x this. Rises to ~0.10 after t = 5.8 s (rougher water).
 */
export const BAND_CV_REF = [
  0.1000, 0.0970, 0.0955, 0.0945, 0.0927, 0.0910, 0.0915, 0.0923, 0.0928, 0.0928,
  0.0925, 0.0923, 0.0915, 0.0904, 0.0893, 0.0892, 0.0898, 0.0904, 0.0905, 0.0907,
  0.0910, 0.0912, 0.0911, 0.0910, 0.0909, 0.0911, 0.0908, 0.0909, 0.0909, 0.0914,
  0.0915, 0.0915, 0.0915, 0.0915, 0.0915, 0.0915, 0.0913, 0.0910, 0.0908, 0.0908,
  0.0908, 0.0909, 0.0909, 0.0917, 0.0924, 0.0932, 0.0932, 0.0932, 0.0927, 0.0920,
  0.0913, 0.0909, 0.0900, 0.0891, 0.0883, 0.0898, 0.0919, 0.0944, 0.0954, 0.0970,
  0.0984, 0.0999, 0.1000, 0.1001, 0.1001, 0.1006, 0.1012, 0.1023, 0.1024, 0.1024,
  0.1015, 0.1012, 0.1008, 0.1009, 0.1011, 0.1012, 0.1012, 0.1012, 0.1008, 0.1004,
  0.0991, 0.0983, 0.0974, 0.0991, 0.1009, 0.1026,
];
export const BAND_CV_MEAN = BAND_CV_REF.reduce((a, b) => a + b, 0) / BAND_CV_REF.length;

/** Linear interpolation in a 0.1 s table, clamped to the clip. */
export function clipTable(table, t) {
  const n = table.length;
  const x = Math.min(Math.max(t * 10, 0), n - 1);
  const i = Math.min(Math.floor(x), n - 2);
  const f = x - i;
  return table[i] * (1 - f) + table[i + 1] * f;
}

/** Reference band luma at time t (s), linear interpolation, clamped to the clip. */
export const bandLumaRef = (t) => clipTable(BAND_LUMA_REF, t);
/** Reference trough contrast at time t (s), 1080p. */
export const bandCVRef = (t) => clipTable(BAND_CV_REF, t);
