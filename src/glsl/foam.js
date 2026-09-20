// Surface foam appearance on the water (swash analysis §3.3, §4; colour analysis §4.5, §5):
//   blanket  - opaque aerated foam behind the bore: soft 10-40 cm lumps and folds (shading), with
//              soft grey pockets where it will tear first (aerated water showing through, never clear);
//   tearing  - as the blanket thins it first goes translucent in soft grey patches, then holes open
//              and grow (each nucleates at its own time: 3-5 cm pinholes growing to 10-30 cm), drawn
//              out along an onshore flow (aspect up to ~1.6) and bent by 0.6-1 m swirls: soft-edged
//              foam bands and strands, not combs;
//   lace     - what survives: a reticulated net of froth strands (8 cm cells, curved, broken, gathered
//              in 0.3-1 m patches, isotropic in the backwash), broad and bright while much surface
//              foam is left, thinning into fine grey filaments (alpha ~0.45) as it decays; an
//              unresolved filament keeps its mean cover (a faint veil), it is never just dimmed away;
//   grain    - torn foam and lace are made of 1-3 mm bubbles: a mean-preserving luminance grain that
//              converges to its mean below a pixel (no discrete dots), and 1-3 cm bubble clumps that
//              rag the edges;
//   crest    - aerated crest of the breaking wave (from the analytic breaker), on the crest top.
// The structure lives in two advected "material" coordinate sets (SwashSim FOAM pass), cross-faded
// (fields before thresholding, contrast kept), each with its own seed that changes only while the set
// is invisible, so nothing re-rolls on screen. It is filtered by the pixel footprint and, where the
// advected frame is strongly stretched or the surface is steep, the foam there thins out instead of
// showing a blurred mean.
// Beyond the scrolling solver window (explore mode) the inputs come from the analytic far field
// (glsl/swashfar.js, included here include-guarded), blended over swashFarWeight.
// Owned by the swash/foam module; used by the water surface fragment shader.
// Requires: glslDefines, NOISE, BED, SWE_SAMPLE (uSweView, uSweFoam, uSweLace, uLacePhase), BREAKER (uEvtCount).
import { SWASH_FAR } from './swashfar.js';

export const FOAM = SWASH_FAR + /* glsl */ `
uniform vec2 uLaceSeeds;  // pattern seed of each advected coordinate set (SwashSim)
uniform vec4 uFoamLook;   // x: lace cells per metre, y: filament width (cell units), z: lace opacity, w: blanket opacity
uniform vec4 uFoamLook2;  // x: torn-foam hole cells per metre, y: max hole elongation along the flow, z: bubble grain, w: swirl warp (m)

// Torn foam: holes nucleate at random points, each at its own time, and grow (an additively weighted
// Voronoi distance, smooth-min so merging holes leave soft walls, not creases). Distances are
// measured in a local metric stretched along the flow (unit t, k = 1 - 1/an^2): only the offsets to
// the nuclei are stretched, so a direction that varies over the surface never shears the pattern.
float foamHoles(vec2 q, vec2 t, float k) {
  vec2 i = floor(q), f = fract(q);
  float acc = 0.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    vec2 g = vec2(float(x), float(y));
    vec3 hh = hash32(i + g);                   // (nucleus position, nucleation time)
    vec2 r = g + hh.xy - f;
    float a = dot(r, t);
    float d = sqrt(max(dot(r, r) - k * a * a, 0.0)) + 0.9 * hh.z;
    acc += exp(-d * 12.5);
  }
  return -0.08 * log(max(acc, 1e-30));
}
// Lace net: Voronoi cells (same local metric); returns ~ the distance to the nearest cell wall (cell units)
float laceWall(vec2 q, vec2 t, float k) {
  vec2 i = floor(q), f = fract(q);
  float f1 = 8.0, f2 = 8.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    vec2 g = vec2(float(x), float(y));
    vec2 r = g + hash22(i + g) - f;
    float a = dot(r, t);
    float d = dot(r, r) - k * a * a;
    if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) { f2 = d; }
  }
  return 0.5 * (sqrt(f2) - sqrt(f1));
}

// structure of one advected coordinate set
//   tear: torn-foam field (grown holes, uFoamLook2.x cells per metre), ~uniform on [0,1] (calibrated
//         offline; holes open from its low values)
//   net:  lace filament cover: a warped, broken, patchy Voronoi net (uFoamLook.x cells per metre),
//         footprint-filtered: an unresolved strand keeps its mean cover (a faint veil)
//   grain: bubble grain (-1..1, zero mean, faded out below a pixel)
//   wall: the hole field alone (no patches), ~uniform: at a low cover a net of froth strands meeting
//         in junctions (the torn blanket's remnant, ~16 cm cells)
struct FoamSet { float tear; float net; float grain; float clump; float wall; };
FoamSet foamSet(vec2 p, float seed, float fp, vec2 t, float an, float anL, bool tearOn, bool netOn, float wmul) {
  FoamSet s = FoamSet(0.5, 0.0, 0.0, 0.0, 0.5);
  // swirls: a smooth 0.6-1 m warp bends the holes and strands into curved bands
  vec2 pw = p + uFoamLook2.w * vec2(gnoise(p * 1.3 + vec2(seed, 1.7)), gnoise(p * 1.3 + vec2(4.1 - seed, -2.3)));
  if (tearOn) {
    vec2 q = pw * uFoamLook2.x;
    q += 0.6 * vec2(gnoise(q * 0.7 + vec2(seed * 1.3, 0.0)), gnoise(q * 0.7 + vec2(5.2 - seed, 0.0)));   // irregular hole shapes
    float h = foamHoles(q + seed * 7.3, t, 1.0 - 1.0 / (an * an)) * sqrt(an);
    // (0.5-1 m patches that tear sooner or later than their surroundings)
    float raw = h + 0.25 * gnoise(p * 1.7 + vec2(seed * 2.3, -seed)) - (0.78 + 0.11 * (an - 1.0));
    float ca = 0.57 + 0.092 * (an - 1.0);
    s.tear = smoothstep(-ca, ca, raw);
    float cw = 0.545 + 0.1 * (an - 1.0);
    s.wall = smoothstep(-cw, cw, h - (0.781 + 0.112 * (an - 1.0)));
  }
  if (netOn) {
    vec2 q = pw * uFoamLook.x;
    q += 0.55 * vec2(gnoise(q * 0.55 + vec2(seed, 0.0)), gnoise(q * 0.55 + vec2(5.2 - seed, 0.0)))
       + 0.2 * vec2(gnoise(q * 1.9 - vec2(seed * 1.3, 0.0)), gnoise(q * 1.9 + vec2(seed + 1.7, 0.0)));
    float lf = laceWall(q + seed * 13.1, t, 1.0 - 1.0 / (anL * anL));
    // soft filaments (a dense bubble core with a sparse halo; fresh lace - much surface foam left -
    // is made of broader froth strands); an unresolved one keeps its mean cover
    float w0 = uFoamLook.y * wmul;
    float we = max(w0, 0.7 * fp * uFoamLook.x);
    float core = exp(-(lf * lf) / (we * we)) * (w0 / we);
    // broken into segments (gaps where the chains have drained), strands varying in density, and
    // gathered in 0.3-1 m patches (a dense net in places, a few loose strands elsewhere)
    float seg = smoothstep(0.34, 0.62, 0.5 + 0.5 * gnoise(p * 13.0 + seed * 2.9));
    float pch = smoothstep(-0.25, 0.65, gnoise(p * 2.1 + vec2(seed * 1.7, 3.1)) + 0.5 * gnoise(p * 4.7 - vec2(seed, 0.0)));
    s.net = core * seg * (0.5 + 0.5 * smoothstep(-0.6, 0.6, gnoise(p * 27.0 - seed * 4.1))) * mix(0.02, 1.55, pch*pch);
  }
  // bubble grain: 2-3 mm, two octaves, faded out as it drops below a pixel; and 1-3 cm bubble clumps
  // (the mottled, ragged texture of torn foam: fuzzy margins, not smooth bands)
  float ga = 1.0 - smoothstep(0.0015, 0.004, fp);
  if (ga > 0.0) s.grain = ga * (1.4 * (vnoise(p * 380.0 + seed * 17.0) - 0.5) + 0.9 * (vnoise(p * 760.0 - seed * 9.0) - 0.5) * (1.0 - smoothstep(0.0008, 0.002, fp)));
  // (gradient noise, not value noise: thresholded value noise reads as round sequins)
  float gc = 1.0 - smoothstep(0.006, 0.014, fp);
  if (gc > 0.0) s.clump = gc * (0.45 * gnoise(p * 19.0 - seed * 5.3) + 0.25 * gnoise(p * 43.0 + seed * 3.1));
  return s;
}
// Soft 10-40 cm lumps of the churning blanket and the narrow self-shadowed folds between them
// (ridged noise). The blanket surface is renewed by the turbulence all the time, so this texture
// must not accumulate the shear of the long-lived advected frames (it would wind up into marble
// swirls): it rides the local flow over a short 0.5 s cycle instead (two phase-offset layers),
// with a smoothed, capped displacement (no compression into stripes at a bore front).
float lumpField(vec2 p, float seed, float t) {
  float n1 = gnoise(p * 4.5 + seed), n2 = gnoise(p * 10.0 + vec2(0.9 * t, -0.6 * t) - seed);
  float fold = 1.0 - abs(gnoise(p * vec2(6.0, 4.0) + vec2(-0.5 * t, 0.7 * t) + seed * 0.3));
  return 0.5 + 0.5 * (0.6 * n1 + 0.4 * n2) - 0.35 * fold * fold * fold * fold;
}
float blanketLumps(vec2 xz, vec2 suv, float t, float wf, vec2 farVel) {
  const float T = 0.5;
  vec2 e = 3.0 * uSweTexel;
  vec2 vel = 0.4 * textureLod(uSweView, suv, 0.0).yz + 0.15 * (textureLod(uSweView, suv + vec2(e.x, 0.0), 0.0).yz + textureLod(uSweView, suv - vec2(e.x, 0.0), 0.0).yz
           + textureLod(uSweView, suv + vec2(0.0, e.y), 0.0).yz + textureLod(uSweView, suv - vec2(0.0, e.y), 0.0).yz);
  vel = mix(vel, farVel, wf);
  vec2 d = vel * T;
  d *= min(1.0, 0.4 / max(length(d), 1e-4));
  float ca = t / T, cb = ca + 0.5;
  float fa = fract(ca), fb = fract(cb);
  float w = 1.0 - abs(2.0 * fa - 1.0);
  float la = lumpField(xz - d * fa, floor(ca) * 1.618, t), lb = lumpField(xz - d * fb, floor(cb) * 2.414 + 0.5, t);
  return 0.5 + (w * (la - 0.5) + (1.0 - w) * (lb - 0.5)) / sqrt(w * w + (1.0 - w) * (1.0 - w));
}

// Returns (coverage, foam density 0..1 for the optics' foam shading, lace alpha, crest foam).
vec4 surfaceFoam(vec3 P, float t, float crestFoam, out vec2 flowUV, out float turb) {
  vec2 suv = sweUV(P.xz);
  float wIn = sweInside(P.xz);
  vec4 fm = textureLod(uSweFoam, suv, 0.0) * wIn;   // (blanket R, lace G, turbulence K, milk M), face-masked
  vec4 lo = textureLod(uSweLace, suv, 0.0) * wIn;
  vec4 handoffDelta = vec4(0.0);
  vec4 svw = textureLod(uSweView, suv, 0.0);
  // (depth across the flow over ~12 cm: a film running off in rivulets - dry gaps between them -
  // carries no foam: combed into streaks down the face otherwise)
  float hE = textureLod(uSweView, suv + vec2(2.0 * uSweTexel.x, 0.0), 0.0).x, hW = textureLod(uSweView, suv - vec2(2.0 * uSweTexel.x, 0.0), 0.0).x;
  float hw = (2.0 * svw.x + hE + hW) * 0.25;
  float hMin = min(svw.x, min(hE, hW));
  vec2 fvel = svw.yz;
  // beyond the scrolling solver window (explore mode) the analytic far field takes over
  float wf = swashFarWeight(P.xz);
  vec2 farVel = vec2(0.0);
#ifdef SWASH_FAR_ON
  if (wf > 0.0) {
    // (three plain fetches: this shader is large and FXC compile time grows fast with its size)
    vec2 fuv = swashFarUV(P.xz);
    vec4 fV = textureLod(uSwashFarView, fuv, 0.0);
    float fD = textureLod(uSwashFarWetT, fuv, 0.0).z;
    fm = mix(fm, textureLod(uSwashFarFoamT, fuv, 0.0), wf);
    handoffDelta = vec4(0.0, -fD, 0.0, -fD) - lo;
    lo += wf * handoffDelta;
    hw = mix(hw, fV.x, wf);
    farVel = fV.yz;
    fvel = mix(fvel, farVel, wf);
  }
#endif
  turb = fm.b;
  // cross-fade of the two advected coordinate sets: each is shown alone for 28 % of its life, 0.29 s cross-fades
  float phA = uLacePhase.x;
  float wa = smoothstep(0.0, 0.22, phA) * (1.0 - smoothstep(0.5, 0.72, phA));
  float wb = 1.0 - wa;
  vec2 pa = P.xz + lo.xy, pb = P.xz + lo.zw;
  // capillary ripples drift with the sheet (a continuous blend of the two frames at a third of
  // their displacement: no jump when a frame is reset, only a slow slide during a cross-fade)
  flowUV = P.xz + 0.3 * (wa * lo.xy + wb * lo.zw);
  // derivatives before any divergent branch: pixel footprint (3D: a steep face has a large one),
  // steepness of the rendered surface, stretch of each advected frame
  vec3 dPx = dFdx(P), dPy = dFdy(P);
  float fp = max(max(length(dPx), length(dPy)), 1.0e-4);
  float ny = abs(normalize(cross(dPx, dPy)).y);
  float fpXZ = max(length(dPx.xz) + length(dPy.xz), 1.0e-5);
  // Different coordinate origins across the near/far handoff are not fluid
  // stretching. Remove the derivative of the blend weight itself; otherwise
  // a several-metre origin difference wipes out the foam in a moving band.
  vec4 frameDx = dFdx(lo) - handoffDelta * dFdx(wf);
  vec4 frameDy = dFdy(lo) - handoffDelta * dFdy(wf);
  float strA = (length(frameDx.xy) + length(frameDy.xy)) / fpXZ;
  float strB = (length(frameDx.zw) + length(frameDy.zw)) / fpXZ;
  // aerated crest of the breaking wave: only on the crest top (on the steep face it would streak)
  float cf = crestFoam * smoothstep(0.35, 0.75, ny)
           * smoothstep(0.2, 0.8, 0.5 + 0.5 * gnoise(P.xz * vec2(7.0, 11.0) + vec2(0.0, -t * 3.0)) + crestFoam * 0.5);
  if (fm.r < 0.01 && fm.g < 0.01) return vec4(cf, cf > 0.0 ? 0.85 : 0.0, 0.0, cf);

  float R = fm.r, G = clamp(fm.g, 0.0, 1.5);
  // the local flow draws the holes out along it (aspect up to ~1.7 in a fast sheet, measured 1.7-3);
  // a slow film's lace is nearly isotropic (0.95-1.05)
  float spd = length(fvel);
  vec2 dir = spd > 1.0e-3 ? fvel / spd : vec2(0.0, 1.0);
  // (only the uprush tears the blanket into flow-aligned holes; lace in the backwash is isotropic)
  float fast = smoothstep(0.3, 1.2, spd) * smoothstep(-0.2, 0.4, fvel.y / max(spd, 0.1));
  // (the lace of a fast sheet - a backwash running down the face - is drawn into streaks along it)
  float an = mix(1.0, uFoamLook2.y, fast), anL = mix(1.0, 1.65, smoothstep(.25,1.4,spd));
  float amtG = 1.0 - exp(-G / 0.9);
  // structure is shown only where the surface is flat enough and the advected frame is not
  // stretched (strands / drips / wood-grain otherwise): there the foam thins out
  float flatW = smoothstep(0.5, 0.8, ny);
  // (a frame stretched by more than ~1.5x already combs the holes and strands into drips)
  float keepA = flatW * (1.0 - smoothstep(0.35, 1.0, strA));
  float keepB = flatW * (1.0 - smoothstep(0.35, 1.0, strB));
  // a film thinner than the pebbles carries only a thin veil of foam, fading out toward its edge
  // (a draining film a few mm thick runs off in rivulets between the pebbles: its foam would be drawn
  // into streaks down the face; the clip's thin backwash films carry none)
  float edgeFade = smoothstep(0.002, 0.009, hw) * smoothstep(0.0005, 0.004, hMin + wf);
  float ribVis = 1.0 - smoothstep(0.022, 0.12, fp);         // broad, continuous detail transition

  float netVis = 1.0 - smoothstep(0.16, 0.60, fp * uFoamLook.x);
  // Fade the field itself: rib/wall also feed opacity and lighting outside strVis.
  // Skipping the expensive field is safe only once every contribution is zero.
  float tearVis = ribVis * (1.0 - smoothstep(1.05, 1.4, R));
  bool torn = tearVis > 0.0;
  bool laced = G > 0.02 && R < 0.7;
  bool netOn = laced && R < 0.45 && netVis > 0.0;
  float thin = 1.0 - smoothstep(0.3, 1.0, R);                 // torn foam shows its bubbles
  // both coordinate sets through one call site (dynamic trip count: not unrolled by FXC)
  float tearAcc = 0.0, netAcc = 0.0, grainAcc = 0.0, clumpAcc = 0.0, wallAcc = 0.0;
  for (int k = 0; k < 2 + min(uEvtCount, 0); k++) {
    float wk = k == 0 ? wa : wb;
    if (wk <= 0.0) continue;
    FoamSet S = foamSet(k == 0 ? pa : pb, k == 0 ? uLaceSeeds.x : uLaceSeeds.y + 3.3, fp, dir, an, anL, torn, netOn, 1.0 + 2.5 * amtG * amtG);
    float kk = wk * (k == 0 ? keepA : keepB);
    tearAcc += kk * (S.tear - 0.5);
    wallAcc += kk * (S.wall - 0.5);
    netAcc += kk * S.net;
    // (the bubble texture rides the frame too: where the frame is stretched it would be combed
    // into streaks, so it fades with the structure)
    grainAcc += kk * S.grain;
    clumpAcc += kk * S.clump;
  }
  float ka = wa * keepA, kb = wb * keepB;
  float nrm = 1.0 / max(sqrt(wa * wa + wb * wb), 1e-3);
  float clump = clumpAcc * nrm;
  // torn-foam field cross-faded before thresholding (contrast kept; ragged at the 1-3 cm clump
  // scale), the net after
  float rib = 0.5 + tearAcc * nrm * tearVis + 0.09 * clump;
  float wall = 0.5 + wallAcc * nrm * tearVis + 0.08 * clump;
  float net = netAcc;
  float grain = grainAcc * nrm * uFoamLook2.z;
  float lump = R > 0.3 ? mix(0.5, blanketLumps(P.xz, suv, t, wf, farVel), flatW*smoothstep(.3,.55,R)) : 0.5;
  float sup = ka + kb;                                        // structure allowed (flat, unstretched)
  float strVis = sup * ribVis;                                // how much of the torn structure shows

  // ---- blanket and tearing: the blanket first thins in soft translucent patches, then holes open
  // from the low parts of the torn-foam field as R decays
  // (mean cover cB: ~0.95 for a fresh blanket, 0.7 at R 0.5, 0.4 at 0.25, 0.13 at 0.1)
  float cB = (1.0 - exp(-pow(R / 0.42, 1.4))) * smoothstep(0.02, 0.1, R);
  // soft, diffuse edges: a ramp around the threshold (softer while mostly intact); when little is
  // left (cover < ew) the threshold is placed so the mean stays cB and the last strands get fainter
  float ew = mix(0.16, 0.36, smoothstep(0.3, 0.8, cB));
  float th = cB >= ew ? 1.0 - cB : 1.0 + ew - 2.0 * sqrt(ew * cB);
  // a mostly intact blanket (cover > ~0.6) thins in soft, dim patches (mean-preserving linear map)
  // rather than punching crisp holes; a torn one is a set of soft-edged bands and strands
  float rbHi = clamp(1.0 - 2.0 * (1.0 - cB) * (1.0 - rib), 0.0, 1.0);
  float rbLo = clamp((rib - th) / (2.0 * ew) + 0.5, 0.0, 1.0);
  rbLo = rbLo * rbLo * (3.0 - 2.0 * rbLo);
  // (where the structure is suppressed - a steep or strongly stretched surface - the foam thins out
  // instead of showing its blurred mean; far away, where it is merely unresolved, the mean is right)
  float rb = mix(cB * (1.0 - 0.65 * ribVis * (1.0 - sup)), mix(rbLo, rbHi, smoothstep(0.6, 0.8, cB)), strVis);
  // a dense blanket is lumpy, and soft grey pockets of aerated water show in it (where the torn-foam
  // field will open its first holes: 5-20 cm, translucent, never clear)
  float pocket = (1.0 - smoothstep(0.0, 0.3, rib)) * smoothstep(0.5, 1.2, R) * strVis;
  // torn foam is a translucent sparkle of bubbles over a faint veil of fine bubbles (the dark
  // windows between the strands are grey-green, not clear); the thick blanket is nearly opaque
  float aR = mix(0.7, 0.94, smoothstep(0.3, 1.2, R)) * (1.0 - 0.45 * pocket);
  float haze = 0.12 * smoothstep(0.02, 0.5, R + 0.12 * G) * thin;   // (the veil is the torn blanket's, not the lace's)
  // (torn bands have dense cores and translucent, bubbly, mottled margins)
  float blanket = rb * aR * mix(mix(0.62, 0.85, smoothstep(0.5, 0.9, cB)), 1.0, smoothstep(th, th + 0.45, rib))
                * (1.0 + (0.8 * grain + 0.3 * clump) * thin);
  blanket = clamp(blanket + haze * (1.0 - blanket), 0.0, 1.0) * uFoamLook.w;

  // ---- lace: the last strands of the torn foam, then a thin broken net once the blanket has gone
  // (swash §4: lace forms 1.2-1.7 s after the bore, peaks at 22-29 % cover, alpha 0.5 -> 0.38)
  float lace = 0.0, laceD = 0.25;
  if (laced) {
    // (lace amount: ~0.75 at the peak, fading as the surface foam decays and the film drains)
    float amt = amtG;
    // the torn blanket's last remnant scraps (a few, while much surface foam is left); the lace
    // itself is the net: broad froth strands while fresh, thinning into the fine broken net
    float cL = 0.07 * amt * amt;
    float ewL = 0.2;
    float thL = cL >= ewL ? 1.0 - cL : 1.0 + ewL - 2.0 * sqrt(ewL * cL);
    float rl = clamp((wall - thL) / (2.0 * ewL) + 0.5, 0.0, 1.0);
    float ribL = mix(cL * (1.0 - 0.65 * ribVis * (1.0 - sup)), rl * rl * (3.0 - 2.0 * rl), strVis);
    // the net (where its structure is suppressed it thins out; unresolved, the faint mean veil)
    float netMean = 0.09 * (1.0 + 2.5 * amt * amt);
    float netL = mix(netMean * (1.0 - 0.65 * netVis * (1.0 - sup)), net, sup * netVis)
               * min(1.3 * amt, 1.0) * smoothstep(0.45, 0.1, R);
    // (fresh froth is dense and bright, the late net thin and grey)
    float sL = 0.8 * clamp(ribL * (1.0 + grain + 0.35 * clump), 0.0, 1.0);
    float nL = mix(uFoamLook.z, 0.85, smoothstep(0.45, 0.8, amt)) * clamp(netL * (1.0 + grain + 0.3 * clump), 0.0, 1.0);
    lace = max(sL, nL) * smoothstep(0.02, 0.12, G) * (1.0 - smoothstep(0.3, 0.7, R));
    laceD = mix(mix(0.25, 0.5, smoothstep(0.45, 0.8, amt)), 0.6, sL / max(sL + nL, 1e-3));
  }
  // sparse 1-2 cm bubble rings on the early, milky backwash (swash §5), close to the camera:
  // big bubbles whose domes have burst leave a ring of small ones for ~1 s
  float milkyBack = smoothstep(0.25, 0.55, fm.a) * smoothstep(-0.15, -0.5, fvel.y) * (1.0 - smoothstep(0.2, 0.5, R)) * (ka + kb);
  milkyBack *= smoothstep(0.0, 0.025, milkyBack);
  if (milkyBack > 0.0 && fp < 0.008) {
    vec2 rq = (P.xz + (wa * lo.xy + wb * lo.zw)) / 0.07;
    vec3 wr = worley(rq + vec2(uLaceSeeds.x * 3.1, 0.0));
    float rr = mix(0.1, 0.17, fract(wr.z * 7.31));             // ring radius 0.7-1.2 cm
    float rw = max(0.035, 0.6 * fp / 0.07);                     // ring width (cell units), >= 1/2 px
    float ring = exp(-pow((wr.x - rr) / rw, 2.0)) * step(wr.z, 0.22) * min(1.0, 0.035 / rw);
    lace = max(lace, 0.55 * ring * milkyBack * (1.0 - smoothstep(0.005, 0.008, fp)));
  }

  // foam density for the optics' composition (water.js: 0 = thin translucent foam, dimmer, a
  // little of the water showing through; 1 = thick blanket, the brightest white, scene-linear
  // ~0.85-1.0 in the sun): the thick blanket is dense, its sunlit lump tops denser still and the
  // self-shadowed folds and pockets between them less so (brightness), bubbles give it a fine
  // grain; torn foam and lace are thin
  float dBl = mix(0.6, 0.95, smoothstep(0.3, 1.2, R)) + 1.1 * (lump - 0.5) * (1.0 - thin) + (0.35 * grain + 0.1 * clump) * (1.0 - thin)
            - 0.3 * (1.0 - rb) * (1.0 - thin) - 0.35 * pocket;
  dBl = mix(clamp(dBl, 0.0, 1.0), mix(0.3, 0.6, rb) * (1.0 + 0.5 * grain + 0.3 * clump), thin);
  float wBl = blanket * edgeFade, wLa = lace * edgeFade * (1.0 - wBl);
  float dense = clamp((dBl * wBl + laceD * wLa + 0.85 * cf) / max(wBl + wLa + cf, 1e-3), 0.0, 1.0);

  float cov = 1.0 - (1.0 - blanket * edgeFade) * (1.0 - lace * edgeFade) * (1.0 - cf);
  return vec4(clamp(cov, 0.0, 1.0), dense, lace * edgeFade, cf);
}
`;
