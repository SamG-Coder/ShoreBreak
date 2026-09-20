import * as THREE from 'three';
import { glslDefines } from '../config.js';
import { NOISE, SKY } from '../glsl/common.js';
import { BEND_GLSL } from './terrain.js';
import { HAZE_GLSL } from './haze.js';
import { Seafront } from './Seafront.js';

// Explore mode: what lies behind the promenade wall and around the bay, all cheap and procedural.
//  - the seafront: two rows of 6-12 storey buildings (Nice: Belle Epoque hotels and 1960s-70s
//    apartment blocks, rendered in muted cream / ochre / salmon with shutters, balconies,
//    cornices, set-back attics) on world-anchored slots along +-4 km of the curved bay:
//    instanced boxes with a procedural facade shader (window reveals and balcony shadows from
//    the real sun direction, band-limited to the pixel footprint);
//  - the Canary date palms of the promenade (instanced trunks + crossed alpha-tested crown cards);
//  - a far panorama (8.6-10.5 km strips whose silhouettes are built on the CPU, no discard): the
//    hills of Nice and the Prealpes behind the city, Mont Boron and the Chateau headland
//    closing the bay to the east with Cap Ferrat beyond, the low line of Cap d'Antibes and the
//    faint Esterel on the western sea horizon, all fading into the sky's horizon haze.
// x: along the shore (+x = west, the sun's side), z: +z inland.

const Y_PROM = 5.56;                    // promenade / street level (Promenade.js Y_CAP)
const CITY = [
  // slot (m), instances, front z, z jitter, depth, storeys min / max, tall share, street gap share
  { slot: 21, n: 400, z: 80, zj: 2.5, depth: 18, fmin: 5, fmax: 9, tall: 0.12, gap: 0.10 },
  { slot: 27, n: 310, z: 112, zj: 14, depth: 24, fmin: 6, fmax: 11, tall: 0.18, gap: 0.0 },
];

function boxGeometry() {
  const P = [], N = [], I = [];
  const face = (o, u, v, n) => {
    const b = P.length / 3;
    for (const [a, c] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      P.push(o[0] + a * u[0] + c * v[0], o[1] + a * u[1] + c * v[1], o[2] + a * u[2] + c * v[2]);
      N.push(...n);
    }
    I.push(b, b + 1, b + 2, b, b + 2, b + 3);
  };
  face([0, 0.5, 1], [1, 0, 0], [0, 0.5, 0], [0, 0, 1]);
  face([0, 0.5, -1], [-1, 0, 0], [0, 0.5, 0], [0, 0, -1]);
  face([1, 0.5, 0], [0, 0, -1], [0, 0.5, 0], [1, 0, 0]);
  face([-1, 0.5, 0], [0, 0, 1], [0, 0.5, 0], [-1, 0, 0]);
  face([0, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, 0]);
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(N, 3));
  g.setIndex(I);
  return g;
}

// ------------------------------------------------------------------ the seafront buildings
// Each slot's building is a hash of its world slot index; the attic pass draws a set-back top
// storey on some of them (same slots, same hashes).
const CITY_VERT = /* glsl */ `
uniform vec2 uBeachFocus;
uniform vec4 uCity0;    // slot, instances, front z, z jitter
uniform vec4 uCity1;    // depth, storeys min, max, tall share
uniform vec2 uGapAttic; // street gap share, attic pass (0 / 1)
out vec3 vW;            // world position (straight-beach frame)
out vec3 vN;
out vec3 vL;            // facade coords: u along the face (m), height above the street (m), face id
flat out vec4 vB;       // building: height, width, seed, style
out float vViewZ;
float edgeX(float k) { return (k + 0.32 * (hash12(vec2(k, uCity0.x)) - 0.5)) * uCity0.x; }
void main() {
  // Kilometres of fixed scenery cover the bounded Explore area. Recycling these
  // slots with the camera could make distant roof silhouettes appear in one frame.
  float s = float(gl_InstanceID) - floor(uCity0.y * 0.5);
  vec3 r = hash32(vec2(s, uCity0.x * 1.7));
  vec3 r2 = hash32(vec2(s + 71.3, uCity0.z));
  // The local frontage is modeled in Seafront.js; retain this cheap city at distance.
  if (uCity0.z < 90.0 && s >= -18.0 && s <= 18.0) { gl_Position=vec4(2.0,2.0,2.0,1.0); return; }
  float xa = edgeX(s), xb = edgeX(s + 1.0);
  float storeys = floor(mix(uCity1.y, uCity1.z + 0.99, r.y * r.y));
  if (r2.x < uCity1.w) storeys += 2.0 + floor(4.0 * r2.y);           // hotels / taller blocks
  float H = 4.4 + storeys * 3.05 + 0.9;                               // ground floor, storeys, cornice
  float gap = step(r2.z, uGapAttic.x);                                // a side street
  float zf = uCity0.z + uCity0.w * (r.z - 0.5) + 1.5 * step(0.7, r2.y);
  float depth = uCity1.x * (0.8 + 0.4 * r.x);
  float style = floor(r.x * 4.0);
  float y0 = 0.0;
  float has = step(0.45, hash12(vec2(s, 12.9))) * (1.0 - gap);
  if (uGapAttic.y > 0.5) {
    // set-back attic storey (penthouse / mansard) on ~half of the buildings
    float sb = 1.2 + 2.0 * hash12(vec2(s, 4.4));
    if (uGapAttic.y > 1.5) {
      // roof clutter: a lift / stair housing on the (attic) roof
      vec3 rc = hash32(vec2(s, 33.1));
      float w = 2.5 + 2.0 * rc.x, cx = mix(xa + 3.0, xb - 3.0 - w, rc.y);
      y0 = H + has * 3.1; H = (2.2 + 1.0 * rc.z) * (1.0 - gap) * step(0.25, rc.x + rc.z * 0.5);
      xa = cx; xb = cx + w; zf += sb + 2.0 + 3.0 * rc.z; depth = 3.0 + 2.0 * rc.y;
    } else {
      xa += 0.8; xb -= 0.8; zf += sb; depth -= sb + 1.0;
      y0 = H; H = mix(0.0, 3.1, has);
    }
  }
  vec3 p = position;
  vec3 P = vec3(mix(xa, xb, p.x * 0.5 + 0.5), ${Y_PROM.toFixed(2)} + y0 + p.y * H * (1.0 - gap), zf + (p.z * 0.5 + 0.5) * depth);
  vW = P;
  vN = bayRotate(normal, P.x);
  float face = normal.z < -0.5 ? 0.0 : (normal.y > 0.5 ? 2.0 : 1.0);
  vL = vec3(normal.z < -0.5 ? P.x - xa : (normal.x > 0.5 ? P.z - zf : zf + depth - P.z), P.y - ${Y_PROM.toFixed(2)} - y0, face);
  vB = vec4(H, xb - xa, hash12(vec2(s, 3.3)) + uCity0.z * 0.01, uGapAttic.y > 1.5 ? 8.0 : (uGapAttic.y > 0.5 ? 4.0 + style : style + 10.0 * step(0.75, hash12(vec2(s, 6.1)))));
  vec4 mv = viewMatrix * vec4(P - vec3(0.0, 0.0, bayBend(P.x)), 1.0);
  vViewZ = -mv.z;
  gl_Position = projectionMatrix * mv;
}`;

const CITY_FRAG = /* glsl */ `
in vec3 vW;
in vec3 vN;
in vec3 vL;
flat in vec4 vB;
in float vViewZ;
#define PI 3.14159265
// 1 inside [a, b] with pixel-footprint antialiasing (w: footprint in the same units)
float band(float x, float a, float b, float w) { return clamp(min(x - a, b - x) / max(w, 1e-4) + 0.5, 0.0, 1.0); }
void main() {
  vec3 P = vW;
  vec3 V = normalize(cameraPosition - P + vec3(0.0, 0.0, bayBend(P.x)));
  vec3 N = normalize(vN);
  float H = vB.x, Wd = vB.y, seed = vB.z;
  bool clutter = vB.w > 7.5 && vB.w < 9.0;    // roof housings
  bool attic = vB.w > 3.5 && vB.w < 7.5;
  bool sideWin = vB.w > 9.5;                  // a corner building: windows on its side walls too
  float style = mod(vB.w, 4.0);               // 0 classic, 1 classic + balconies, 2 1960s balconies, 3 plain modern
  vec3 rs = hash32(vec2(seed * 91.7, 4.1));
  // muted render colours of the Nice seafront (pre-grade: the post adds x1.35 saturation)
  vec3 base = rs.x < 0.24 ? vec3(0.60, 0.55, 0.45) : rs.x < 0.44 ? vec3(0.62, 0.50, 0.37) : rs.x < 0.62 ? vec3(0.64, 0.50, 0.44)
            : rs.x < 0.76 ? vec3(0.64, 0.58, 0.42) : rs.x < 0.84 ? vec3(0.58, 0.44, 0.36) : rs.x < 0.94 ? vec3(0.66, 0.64, 0.60) : vec3(0.52, 0.52, 0.51);
  if (style > 2.5) base = mix(base, vec3(0.60, 0.59, 0.57), 0.45);
  base *= 0.9 + 0.16 * rs.y;
  base = mix(vec3(dot(base, vec3(0.2126,0.7152,0.0722))), base, 0.82);
  vec3 shut = rs.z < 0.45 ? vec3(0.20, 0.27, 0.22) : rs.z < 0.75 ? vec3(0.32, 0.36, 0.37) : vec3(0.44, 0.34, 0.27);
  float u = vL.x, y = vL.y;
  vec2 fw = max(fwidth(vL.xy), vec2(1e-4));
  vec3 alb = base;
  float glass = 0.0, shade = 1.0;
  vec3 L = uSunDir;
  if (clutter) {
    alb = vec3(0.50, 0.49, 0.47) * (0.85 + 0.2 * rs.y) * (vL.z > 1.5 ? 0.85 : 1.0);
  } else if (vL.z > 1.5) {
    // roof: gravel / membrane, terracotta on the classic buildings, grey zinc on the attics
    alb = attic ? vec3(0.38, 0.39, 0.40) : style < 0.5 ? vec3(0.47, 0.30, 0.22) : vec3(0.40, 0.38, 0.35);
    alb *= 0.9 + 0.2 * vnoise(P.xz * 0.3 + seed * 40.0);
  } else if (attic) {
    // attic storey: rendered or zinc-clad, French windows between piers, some bays blind, a
    // terrace railing along its foot
    float bays = mix(2.6, 3.6, rs.z);
    float bi = floor(u / bays), bf = fract(u / bays);
    float blind = step(0.72, hash12(vec2(bi, seed * 7.0)));
    float win = band(bf, 0.28, 0.72, fw.x / bays) * band(y, 0.25, 2.35, fw.y) * (1.0 - blind);
    vec3 wall = rs.y < 0.5 ? base * 0.95 : vec3(0.42, 0.43, 0.44);
    alb = mix(wall, mix(vec3(0.05, 0.06, 0.07), vec3(0.16, 0.18, 0.2), hash12(vec2(bi, 3.0 + seed))), win);
    glass = win;
    alb *= 1.0 - 0.3 * band(y, 0.0, 0.9, fw.y) * (0.5 + 0.5 * step(0.5, fract(u * 8.0)));   // railing
    float detailA = 1.0 - smoothstep(0.35, 0.8, max(fw.x / bays, fw.y / 3.0) * 3.0);
    alb = mix(mix(wall, vec3(0.08), 0.25), alb, detailA);
    glass *= detailA;
  } else {
    // render weathering: soft blotches, darker streaks under the cornice, grimier low storeys
    alb *= 1.0 + 0.08 * (vnoise(vec2(u, y) * 0.35 + seed * 17.0) - 0.5);
    alb *= 0.94 + 0.08 * smoothstep(0.0, H, y);
    alb *= 1.0 - 0.08 * smoothstep(0.55, 0.9, vnoise(vec2(u * 1.3, y * 0.15) + seed * 5.0)) * smoothstep(H * 0.4, H, y);
    float gf = 4.4, fh = 3.05;
    float fl = (y - gf) / fh, fy = fract(fl), fi = floor(fl);
    float bw0 = mix(3.3, 4.2, fract(seed * 7.3));
    float nb = max(floor(Wd / bw0), 1.0);
    float bays = Wd / nb;
    float bx = u / bays, bi = floor(bx), bf = fract(bx);
    // fade the facade grid into its mean tone once a storey spans < 3 px
    float detail = 1.0 - smoothstep(0.35, 0.8, max(fw.x / bays, fw.y / fh) * 3.0);
    float upper = step(gf, y) * step(y, H - 1.0);
    bool side = vL.z > 0.5;
    vec3 wr = hash32(vec2(bi, fi) + seed * 13.0);
    // French windows (classic: tall, 1.1 m; modern: wide, 1.8 m)
    float ww = (style > 1.5 ? 1.8 : 1.15) / bays;
    float wy0 = style > 1.5 ? 0.12 : 0.18, wy1 = 0.84;
    float win = band(bf, 0.5 - ww * 0.5, 0.5 + ww * 0.5, fw.x / bays) * band(fy, wy0, wy1, fw.y / fh) * upper;
    if (side) win *= sideWin ? step(0.35, hash12(vec2(bi, fi) + seed)) : 0.0;   // (blank party walls)
    // reveals: the lintel and the sun-side jamb shade the top / right of each 18 cm deep recess
    float rev = 0.18;
    float sTop = rev * L.y / max(dot(L, vec3(0.0, 0.0, -1.0)), 0.12) / fh;
    float sSide = rev * L.x / max(dot(L, vec3(0.0, 0.0, -1.0)), 0.12) / bays;
    float inShadow = max(band(fy, wy1 - sTop, wy1, fw.y / fh), band(bf, 0.5 + ww * 0.5 - sSide, 0.5 + ww * 0.5, fw.x / bays));
    // shutters (classic): open beside the window, a third closed over it; louvres read as a tone
    float closed = step(wr.x, 0.3) * step(style, 1.5);
    float sh = (band(bf, 0.5 - ww * 1.12, 0.5 - ww * 0.5, fw.x / bays) + band(bf, 0.5 + ww * 0.5, 0.5 + ww * 1.12, fw.x / bays))
             * band(fy, wy0, wy1, fw.y / fh) * upper * step(style, 1.5) * (1.0 - closed) * (side ? 0.0 : 1.0);
    vec3 winC = mix(vec3(0.035, 0.04, 0.05), vec3(0.16, 0.18, 0.20), wr.y * wr.y);   // interiors, curtains
    winC = mix(winC, vec3(0.50, 0.48, 0.44), step(0.93, wr.z));                      // drawn blinds
    vec3 shutC = shut * (0.8 + 0.3 * wr.z) * (0.9 + 0.1 * sin(y * 60.0) * detail);
    // moulded stone surrounds around the classic windows
    float fb = 0.09 / bays, fyb = 0.09 / fh;
    float surround = band(bf, 0.5 - ww * 0.5 - fb, 0.5 + ww * 0.5 + fb, fw.x / bays) * band(fy, wy0 - fyb, wy1 + fyb * 2.0, fw.y / fh)
                   * upper * step(style, 1.5) * (side ? 0.0 : 1.0);
    alb = mix(alb, base * 1.13, surround * 0.8);
    alb = mix(alb, shutC, sh);
    alb = mix(alb, mix(winC, shutC, closed), win);
    glass = win * (1.0 - closed);
    shade *= 1.0 - 0.45 * inShadow * win;
    alb = mix(alb, base * 1.1, band(fy, wy0 - 0.03, wy0, fw.y / fh) * band(bf, 0.5 - ww * 0.6, 0.5 + ww * 0.6, fw.x / bays) * upper);   // sills
    // balconies: classic ones on some storeys and bays, continuous slabs on the 1960s blocks
    float hasBal = style > 1.5 ? 1.0 : (style > 0.5 ? step(0.5, hash12(vec2(fi, seed * 3.0))) : step(0.85, hash12(vec2(fi, seed))));
    float balW = style > 1.5 ? 1.0 : band(bf, 0.5 - ww * 0.9, 0.5 + ww * 0.9, fw.x / bays);
    float bal = hasBal * balW * upper * step(1.0, fi) * (side ? 0.0 : 1.0);
    if (bal > 0.0) {
      float depthB = style > 1.5 ? 1.3 : 0.7;
      float sB = depthB * L.y / max(dot(L, vec3(0.0, 0.0, -1.0)), 0.12) / fh;   // slab shadow (storeys)
      float slab = band(fy, 0.0, 0.05, fw.y / fh);
      float rail = band(fy, 0.05, 0.38, fw.y / fh);
      float railC = style > 1.5 ? 0.55 + 0.45 * step(0.5, fract(u * 8.0)) * detail : 0.6 + 0.4 * step(0.5, fract(u * 10.0)) * detail;
      alb = mix(alb, alb * 0.6 * railC + base * 0.25 * (1.0 - railC), rail * bal * 0.6);
      alb = mix(alb, base * 1.12, slab * bal);
      shade *= 1.0 - 0.5 * bal * band(fy, 1.0 - min(sB, 0.9), 1.0, fw.y / fh) * (1.0 - slab);
    }
    // string courses (classic), cornice and its shadow, pilasters every few bays
    alb *= 1.0 - 0.1 * band(fy, 0.0, 0.035, fw.y / fh) * upper * step(style, 1.5);
    alb *= 1.0 + 0.05 * step(style, 1.5) * band(bf, 0.0, 0.06, fw.x / bays) * step(0.5, fract(bi * 0.5));
    alb = mix(alb, base * 1.12, band(y, H - 0.9, H - 0.5, fw.y));
    shade *= 1.0 - 0.4 * band(y, H - 0.9 - 0.4 * L.y / 0.31, H - 0.9, fw.y);
    alb = mix(alb, mix(vec3(0.10, 0.10, 0.11), vec3(0.40, 0.22, 0.18), step(0.6, rs.z)), band(y, 0.3, gf - 0.3, fw.y) * 0.8);   // shops
    // far away: the facade's mean tone (windows ~25 % of the area, balcony shadows)
    vec3 mean = mix(base, vec3(0.08), side && !sideWin ? 0.0 : (style > 1.5 ? 0.38 : 0.26));
    alb = mix(mean, alb, detail);
    shade = mix(1.0, shade, detail);
    glass *= detail;
    if (side) {
      // party walls: a plainer, patchier render with long vertical stains
      alb *= 0.93 * (1.0 - 0.1 * smoothstep(0.5, 0.9, vnoise(vec2(u * 0.7, y * 0.08) + seed * 3.0)));
    }
  }
  float NoL = max(dot(N, L), 0.0);
  vec3 col = alb * (uSunColor * (NoL * shade / PI) + skyAmbient(N) * (0.75 + 0.25 * shade));
  // glass reflects the sky (Fresnel ~0.06-0.1 at these angles)
  vec3 R = reflect(-V, N);
  col += glass * 0.08 * landHazeColor(R) * (0.7 + 0.3 * clamp(R.y * 4.0 + 0.5, 0.0, 1.0));
  // aerial perspective (+ a little veiling over the first few hundred metres: humid sea air)
  float haze = max(landHaze(vViewZ), 0.045 * (1.0 - exp(-vViewZ / 300.0)));
  col = mix(col, landHazeColor(-V), haze);
  gl_FragColor = vec4(col, 1.0);
}`;

// ------------------------------------------------------------------ palm trees of the promenade
// Canary Island date palms: stout trunks with a leaf-base 'pineapple' below a dense round crown
// of ~40 arching fronds (4 crossed cards x 10), the lowest ones drooping and brown.
const PALM_STEP = 11.0, N_PALMS = 280, PALM_Z = 60.0, CARDS = 4;
const PALM_VERT = /* glsl */ `
uniform float uTime;
uniform vec2 uBeachFocus;
uniform float uCrown;   // 0: trunks (box), 1: crown cards (quads)
out vec3 vW;
out vec3 vN;
out vec2 vQ;
flat out vec3 vP;       // palm: height, seed, crown radius
out float vViewZ;
void main() {
  int id = gl_InstanceID;
  int palm = uCrown > 0.5 ? id / ${CARDS} : id, card = id - palm * ${CARDS};
  float s = float(palm - ${N_PALMS / 2});
  vec3 r = hash32(vec2(s, 17.3));
  float x0 = s * ${PALM_STEP.toFixed(1)} + 5.0 * (r.x - 0.5);
  if (abs(x0) < 440.0) { gl_Position=vec4(2.0,2.0,2.0,1.0); return; }
  float z0 = ${PALM_Z.toFixed(1)} + 1.2 * (r.y - 0.5) + (hash12(vec2(s, 2.0)) < 0.5 ? 0.0 : -12.0);
  float H = 6.0 + 5.0 * r.z;
  float cr = 3.2 + 1.2 * hash12(vec2(s, 5.5));
  vec2 lean = (hash22(vec2(s, 9.1)) - 0.5) * 0.10;
  vec3 P;
  vec3 n = normal;
  if (uCrown < 0.5) {
    float t = position.y;
    float wdt = mix(0.36, 0.30, t) + 0.12 * smoothstep(0.82, 1.0, t);      // (leaf bases flare below the crown)
    P = vec3(x0, ${Y_PROM.toFixed(2)}, z0) + vec3(position.x * wdt + lean.x * H * t * t, t * H, position.z * wdt + lean.y * H * t * t);
    vQ = vec2(position.x, t);
  } else {
    float a = float(card) * ${(Math.PI / CARDS).toFixed(4)} + r.x * 3.0;
    vec3 ax = vec3(cos(a), 0.0, sin(a));
    vec3 top = vec3(x0 + lean.x * H, ${Y_PROM.toFixed(2)} + H + 0.3, z0 + lean.y * H);
    P = top + ax * position.x * cr + vec3(0.0, position.y * cr * 0.8, 0.0);
    // Gentle onshore breeze: frond tips respond more than the woody crown.
    float flex = smoothstep(0.12, 0.95, abs(position.x));
    float breeze = sin(uTime * 1.15 + x0 * 0.08 + r.z * 6.0) + 0.35 * sin(uTime * 2.7 + r.x * 11.0);
    P.x += 0.06 * flex * breeze;
    P.z += 0.045 * flex * breeze;
    n = vec3(-ax.z, 0.0, ax.x);
    vQ = position.xy;
  }
  vP = vec3(H, hash12(vec2(s, 8.8)) + float(card) * 0.37, cr);
  vW = P;
  vN = bayRotate(n, P.x);
  vec4 mv = viewMatrix * vec4(P - vec3(0.0, 0.0, bayBend(P.x)), 1.0);
  vViewZ = -mv.z;
  gl_Position = projectionMatrix * mv;
}`;

const PALM_FRAG = /* glsl */ `
uniform float uCrown;
in vec3 vW;
in vec3 vN;
in vec2 vQ;
flat in vec3 vP;
in float vViewZ;
#define PI 3.14159265
void main() {
  vec3 P = vW;
  vec3 V = normalize(cameraPosition - P + vec3(0.0, 0.0, bayBend(P.x)));
  vec3 L = uSunDir;
  vec3 col;
  if (uCrown < 0.5) {
    // trunk: grey-brown diamond leaf scars, the rough 'pineapple' of old leaf bases at the top
    vec3 N = normalize(vN);
    // (irregular rings of old leaf scars, fibrous vertical texture; band-limited)
    float fwy = max(fwidth(P.y) * 4.0, 1e-3);
    float ring = vnoise(vec2(P.y * 4.5, vQ.x * 2.0 + vP.y * 30.0));
    float fib = vnoise(vec2(vQ.x * 9.0 + vP.y * 50.0, P.y * 0.8));
    vec3 alb = vec3(0.34, 0.29, 0.23) * (0.86 + 0.22 * (ring - 0.5) * (1.0 - smoothstep(0.08, 0.2, fwy)) + 0.1 * (fib - 0.5));
    alb = mix(alb, vec3(0.30, 0.22, 0.13), smoothstep(0.84, 0.95, vQ.y));
    col = alb * (uSunColor * (max(dot(N, L), 0.0) / PI) + skyAmbient(N));
  } else {
    vec2 q = vQ;
    q.y /= 0.8;
    float a = 0.0, tipS = 0.0, dead = 0.0, depthK = 0.0;
    float sd = vP.y * 17.0;
    for (int k = 0; k < 10 + min(int(uCrown) - 1, 0); k++) {
      float fk = float(k);
      float h1 = hash11(fk * 1.7 + sd), h2 = hash11(fk * 3.1 + sd);
      float phi = mix(-0.75, 3.89, (fk + 0.5) / 10.0) + 0.3 * (h1 - 0.5);
      vec2 d = vec2(cos(phi), sin(phi));
      float droop = mix(0.75, 0.2, clamp(d.y * 0.5 + 0.5, 0.0, 1.0)) * (0.8 + 0.4 * h2);
      float Lf = 0.8 + 0.2 * h2;
      float s = clamp(dot(q, d) / Lf, 0.0, 1.0);
      vec2 c = d * s * Lf - vec2(0.0, droop * s * s);
      vec2 e = q - c;
      float side = dot(e, vec2(-d.y, d.x));
      // leaflets: a comb of stiff pinnae along the rachis, the frond narrowing to its tip
      float w = 0.13 * sin(3.14159 * pow(s, 0.55)) * (1.0 - 0.25 * s);
      float comb = 0.45 + 0.55 * abs(sin(s * 46.0 + side * 25.0 + h1 * 6.0));
      float m = step(length(e), w * comb) * step(0.06, s);
      if (m > a) { a = m; tipS = s; dead = step(d.y, -0.35) * step(0.5, h1); depthK = h2; }
    }
    if (a < 0.5) discard;
    vec3 alb = mix(vec3(0.075, 0.105, 0.045), vec3(0.13, 0.15, 0.07), tipS * 0.7 + 0.3 * depthK);
    alb = mix(alb, vec3(0.27, 0.21, 0.12), dead);
    // a leafy volume: lit from above and the sun side, darker inside and below
    float up = clamp(q.y * 0.55 + 0.5, 0.0, 1.0);
    float sunSide = clamp(0.5 + 0.5 * dot(normalize(vec3(q.x, 0.0, 0.5)), normalize(vec3(L.x, 0.0, 0.3))), 0.0, 1.0);
    col = alb * (uSunColor * (0.2 + 0.5 * up * sunSide) / PI + skyAmbient(vec3(0.0, 1.0, 0.0)) * (0.4 + 0.5 * up));
  }
  float haze = max(landHaze(vViewZ), 0.045 * (1.0 - exp(-vViewZ / 300.0)));
  col = mix(col, landHazeColor(-V), haze);
  gl_FragColor = vec4(col, 1.0);
}`;

// ------------------------------------------------------------------ far panorama
// Silhouettes are built on the CPU (elevation angle vs azimuth phi: 0 inland +z, +90 west +x,
// -90 east) as vertical strips at a few km; the shader only shades them (no discard: early-z).
function hash1(n) { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); }
function vnoise1(x, seed) {
  const i = Math.floor(x), f = x - i, u = f * f * (3 - 2 * f);
  return hash1(i + seed * 57.3) * (1 - u) + hash1(i + 1 + seed * 57.3) * u;
}
function fbm1(x, seed, oct = 4) { let s = 0, a = 0.5, n = 0; for (let k = 0; k < oct; k++) { s += a * (vnoise1(x, seed + k) * 2 - 1); n += a; x *= 2.07; a *= 0.5; } return s / n; }
const sstep = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
const span = (p, a, b, e) => sstep(a - e, a + e, p) * (1 - sstep(b - e, b + e, p));
const ridge = (x, s) => 1 - Math.abs(fbm1(x, s, 3));
// [radius (m), virtual distance for the haze (m), elevation(phi) in degrees, albedo]
const LAYERS = [
  // Nice's own hills behind the city, Mont Boron / the Chateau headland (east), Cagnes (west)
  [8600, 3600, (p) => Math.max(
    (0.9 + 0.8 * fbm1(p * 0.09, 1)) * span(p, -70, 62, 10),
    span(p, -112, -64, 5) * (2.0 + 0.7 * fbm1(p * 0.12, 5)) * sstep(-121, -103, p),
    0.5 * span(p, 58, 96, 6) * (1 + 0.8 * fbm1(p * 0.2, 8))), [0.11, 0.125, 0.085]],
  // the Prealpes behind Nice
  [9400, 13000, (p) => (3.0 + 1.2 * fbm1(p * 0.05, 3) + 0.9 * ridge(p * 0.11, 2)) * span(p, -95, 80, 14), [0.14, 0.15, 0.13]],
  // Cap Ferrat (east) and Cap d'Antibes (west): low lines on the sea horizon
  [10000, 16000, (p) => Math.max(0.33 * span(p, -124, -114, 1.5) * (0.8 + 0.3 * fbm1(p * 0.8, 9)),
    0.2 * span(p, 108, 131, 2.5) * (0.75 + 0.5 * fbm1(p * 0.6, 4))), [0.12, 0.13, 0.11]],
  // the Esterel, far beyond Antibes
  [10500, 30000, (p) => (0.4 + 0.35 * ridge(p * 0.35, 6)) * span(p, 112, 127, 3), [0.13, 0.13, 0.13]],
];

function panoramaGeometry() {
  const seg = 1440, pos = [], attr = [], idx = [];
  LAYERS.forEach(([R, dist, prof], li) => {
    const base = pos.length / 3;
    for (let i = 0; i <= seg; i++) {
      const phi = -180 + (360 * i) / seg;
      const a = (phi * Math.PI) / 180;
      const el = Math.max(prof(phi), 0);
      const x = Math.sin(a) * R, z = Math.cos(a) * R;
      const top = el > 0.002 ? R * Math.tan((el * Math.PI) / 180) + 2.6 : 0;
      pos.push(x, 0, z, x, top, z);
      attr.push(li, dist, 0, li, dist, 1);
    }
    for (let i = 0; i < seg; i++) { const a = base + i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('aLayer', new THREE.Float32BufferAttribute(attr, 3));
  g.setIndex(idx);
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5);
  return g;
}

const PANO_VERT = /* glsl */ `
in vec3 aLayer;          // layer, virtual distance (m), 0 bottom / 1 top of the strip
out vec3 vW;
out vec3 vA;
void main() {
  vW = position;
  vA = aLayer;
  gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
}`;

const PANO_FRAG = /* glsl */ `
uniform vec3 uLayerAlb[${LAYERS.length}];
in vec3 vW;
in vec3 vA;
#define PI 3.14159265
void main() {
  vec3 d = normalize(vW - cameraPosition);
  int li = int(vA.x + 0.5);
  vec3 alb = uLayerAlb[0];
  for (int i = 1; i < ${LAYERS.length}; i++) if (i == li) alb = uLayerAlb[i];
  vec2 q = vec2(atan(d.x, d.z) * 180.0 / PI, asin(clamp(d.y, -1.0, 1.0)) * 180.0 / PI);
  if (li == 0) {
    // scrub, pines and rock with villas (pale specks) on the flanks of the near hills
    float t = vnoise(q * vec2(3.0, 9.0)) * 0.6 + vnoise(q * vec2(11.0, 33.0)) * 0.4;
    alb = mix(alb, vec3(0.22, 0.21, 0.17), smoothstep(0.45, 0.85, t));
    alb = mix(alb, vec3(0.55, 0.50, 0.42), step(0.82, vnoise(q * vec2(45.0, 110.0))) * 0.6);
  } else {
    alb *= 0.85 + 0.3 * vnoise(q * vec2(2.0, 6.0));
  }
  vec3 col = alb * (uSunColor * 0.33 / PI + skyAmbient(vec3(0.0, 1.0, 0.0)));
  // aerial perspective of the far land (e-fold 9 km), a touch denser toward the base of each layer
  float h = 1.0 - exp(-vA.y / 9000.0);
  h = mix(h, 1.0, 0.25 * (1.0 - clamp(vA.z * 3.0, 0.0, 1.0)) * h);
  col = mix(col, landHazeColor(d), h);
  gl_FragColor = vec4(col, 1.0);
}`;

export class Backdrop {
  constructor(shared) {
    const base = {
      uTime: shared.uTime, uBeachFocus: shared.uFocus, uSunDir: shared.uSunDir, uSunColor: shared.uSunColor,
      uSkyAmb: shared.uSkyAmb, uSkyLut: shared.uSkyLut,
    };
    const pre = glslDefines() + NOISE + SKY + HAZE_GLSL + BEND_GLSL;
    this.group = new THREE.Group();
    this.seafront = new Seafront(shared);
    this.group.add(this.seafront.group);
    const box = boxGeometry();
    for (const c of CITY) {
      for (const attic of [0, 1, 2]) {   // facades, attics, roof housings
        const g = box.clone();
        g.instanceCount = c.n;
        const m = new THREE.ShaderMaterial({
          uniforms: {
            ...base,
            uCity0: { value: new THREE.Vector4(c.slot, c.n, c.z, c.zj) },
            uCity1: { value: new THREE.Vector4(c.depth, c.fmin, c.fmax, c.tall) },
            uGapAttic: { value: new THREE.Vector2(c.gap, attic) },
          },
          vertexShader: pre + CITY_VERT, fragmentShader: pre + CITY_FRAG,
        });
        const mesh = new THREE.Mesh(g, m);
        mesh.frustumCulled = false;
        this.group.add(mesh);
      }
    }
    // palms: trunks and crown cards
    const trunk = box.clone();
    trunk.instanceCount = N_PALMS;
    const quad = new THREE.InstancedBufferGeometry();
    quad.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
    quad.setAttribute('normal', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], 3));
    quad.setIndex([0, 1, 2, 0, 2, 3]);
    quad.instanceCount = N_PALMS * CARDS;
    for (const [geo, crown] of [[trunk, 0], [quad, 1]]) {
      const m = new THREE.ShaderMaterial({
        uniforms: { ...base, uCrown: { value: crown } },
        vertexShader: pre + PALM_VERT, fragmentShader: pre + PALM_FRAG,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geo, m);
      mesh.frustumCulled = false;
      this.group.add(mesh);
    }
    // far panorama
    this.pano = new THREE.Mesh(panoramaGeometry(), new THREE.ShaderMaterial({
      uniforms: { ...base, uLayerAlb: { value: LAYERS.map((l) => new THREE.Vector3(...l[3])) } },
      vertexShader: pre + PANO_VERT, fragmentShader: pre + PANO_FRAG, side: THREE.DoubleSide,
    }));
    this.pano.frustumCulled = false;
    this.group.add(this.pano);
  }
}
