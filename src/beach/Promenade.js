import * as THREE from 'three';
import { glslDefines } from '../config.js';
import { NOISE, SKY } from '../glsl/common.js';
import { LAND, BEND_GLSL, FAR_X } from './terrain.js';
import { HAZE_GLSL } from './haze.js';
import { MACRO_NOISE_GLSL } from './macroNoise.js';
import { COAST_SHADOW, coastUniforms } from './CoastMaterial.js';

// Explore mode: the promenade seawall that closes the beach behind the storm berm (Nice: the
// Promenade des Anglais retaining wall, ~4 m above the back-beach). A pale limestone ashlar face
// with grime at the foot and runoff streaks, a projecting cast-stone coping (blocks, joints,
// weathered arris, lichen) that shades the top of the face, a low parapet with a slim painted
// railing, and the promenade's lamp posts behind it. Extruded along the shore to +-4 km
// in fixed world coordinates and curved with the bay (terrain.js bayBend). The playable
// coast is +-60 m: this already covers every viewpoint without shifting the wall's
// triangulation. Posts scroll on world slots and fade before their window retires them.

const Z = LAND.wallZ, Y0 = LAND.backY - 0.35;
const Y_FACE = 5.30, CAP = 0.25, Y_CAP = 5.56, SET = 0.12, Y_PAR = 6.40, PAR_T = 0.27;
// railing on the parapet: 4 cm tube 0.42 m above it, posts every 1.6 m
const R_Y = Y_PAR + 0.42, R_T = 0.045, R_Z = Z - SET + PAR_T * 0.5;
// profile (z, y) polyline and part id of each segment
const PROFILE = [
  [[Z, Y0], [Z, Y_FACE], 0],                            // wall face
  [[Z, Y_FACE], [Z - CAP, Y_FACE], 1],                  // coping soffit
  [[Z - CAP, Y_FACE], [Z - CAP, Y_CAP], 2],             // coping front
  [[Z - CAP, Y_CAP], [Z - SET, Y_CAP], 3],              // coping top
  [[Z - SET, Y_CAP], [Z - SET, Y_PAR], 4],              // parapet face
  [[Z - SET, Y_PAR], [Z - SET + PAR_T, Y_PAR], 5],      // parapet top
  // railing tube (square section)
  [[R_Z - R_T / 2, R_Y - R_T / 2], [R_Z - R_T / 2, R_Y + R_T / 2], 6],
  [[R_Z - R_T / 2, R_Y + R_T / 2], [R_Z + R_T / 2, R_Y + R_T / 2], 6],
  [[R_Z + R_T / 2, R_Y - R_T / 2], [R_Z - R_T / 2, R_Y - R_T / 2], 6],
];

function buildGeometry() {
  const near = [];
  for (let x = -64; x <= 64 + 1e-6; x += 2) near.push(x);
  const far = FAR_X.filter((v) => v > 64.5);
  const xs = [...far.map((v) => -v).reverse(), ...near, ...far];
  const pos = [], part = [], idx = [];
  for (const [[z0, y0], [z1, y1], id] of PROFILE) {
    const base = pos.length / 3;
    for (const x of xs) { pos.push(x, y0, z0, x, y1, z1); part.push(id, id); }
    for (let i = 0; i < xs.length - 1; i++) {
      const a = base + i * 2, b = a + 1, c = a + 2, d = a + 3;
      idx.push(a, b, c, b, d, c);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
  g.setAttribute('aPart', new THREE.BufferAttribute(new Float32Array(part), 1));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(idx), 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5);
  return g;
}

// box (5 faces, no bottom) around the origin with half sizes (1, 1, 1), y from 0 to 1
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

const VERT = /* glsl */ `
uniform vec2 uBeachFocus;
in float aPart;
out vec3 vW;
out float vPart;
out float vViewZ;
void main() {
  vec3 P = position;
  vW = P;
  vPart = aPart;
  vec4 mv = viewMatrix * vec4(P - vec3(0.0, 0.0, bayBend(P.x)), 1.0);
  vViewZ = -mv.z;
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = /* glsl */ `
uniform float uTime;
uniform sampler2D uStone, uStoneNormal, uStoneRough;
in vec3 vW;
in float vPart;
in float vViewZ;
#define PI 3.14159265
float lumW(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float keepW(float lam, float foot) { return 1.0 - smoothstep(0.3, 0.6, 2.0 * foot / lam); }
// pebbles piled at the wall foot (terrain.js wallToe, same noise)
float toeAt(float x) {
  vec4 n = macroNoiseLod(vec2(x, 7.0), vec2(4.5));
  return 0.45 + 0.15 * n.y;
}
void main() {
  vec3 P = vW;
  vec3 Pb = P - vec3(0.0, 0.0, bayBend(P.x));
  vec3 V = normalize(cameraPosition - Pb);
  int part = int(vPart + 0.5);
  vec3 N = part == 1 ? vec3(0.0, -1.0, 0.0) : (part == 3 || part == 5) ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, -1.0);
  if (part == 6) N = normalize(vec3(0.0, sign(P.y - ${R_Y.toFixed(3)}) * step(${(R_T * 0.45).toFixed(4)}, abs(P.y - ${R_Y.toFixed(3)})), -1.0));
  N = bayRotate(N, P.x);
  float foot = 0.5 * max(length(fwidth(Pb)), 1e-4);
  vec3 alb;
  float surfaceRough=0.8;
  if (part == 0) {
    // ---- limestone ashlar: 0.62 m courses, 1.1-2.3 m blocks in running bond, over a darker plinth
    float yb = P.y - ${Y0.toFixed(3)};
    float CH = 0.62;
    float course = floor(yb / CH);
    float cy = fract(yb / CH);
    float bl = mix(1.1, 2.3, hash12(vec2(course, 7.0)));
    float bx = P.x / bl + hash12(vec2(course, 3.1)) * 3.0;
    float blk = floor(bx);
    float cx = fract(bx);
    vec3 rb = hash32(vec2(blk, course) + 11.0);
    alb = vec3(0.52, 0.47, 0.39) * (0.88 + 0.2 * rb.x) * (1.0 + 0.07 * (rb.y - 0.5) * vec3(1.0, 0.25, -0.9));
    // Ashore's photographic stone, independently offset on each cut block.
    vec2 stoneUV=P.xy/2.8+rb.xy*5.0;
    vec3 stone=textureGrad(uStone,stoneUV,dFdx(P.xy)/2.8,dFdy(P.xy)/2.8).rgb;
    alb*=mix(vec3(1.0),clamp(stone/vec3(.42,.38,.32),vec3(.45),vec3(1.55)),.62);
    vec3 stoneN=textureGrad(uStoneNormal,stoneUV,dFdx(P.xy)/2.8,dFdy(P.xy)/2.8).xyz*2.0-1.0;
    N=normalize(N*max(.25,stoneN.z)+bayRotate(vec3(stoneN.x*.26,stoneN.y*.26,0.0),P.x));
    surfaceRough=textureGrad(uStoneRough,stoneUV,dFdx(P.xy)/2.8,dFdy(P.xy)/2.8).r;
    // stone texture: fossil-shell pitting, softer weathering blotches, sub-block tone drift
    alb *= 1.0 + 0.12 * (vnoise(P.xy * vec2(2.2, 3.5) + rb.xy * 50.0) - 0.5) * keepW(0.45, foot)
               + 0.08 * (vnoise(P.xy * 19.0) - 0.5) * keepW(0.06, foot)
               + 0.06 * (vnoise(P.xy * 0.6 + 4.0) - 0.5);
    // mortar joints (1.2 cm, recessed: darker), band-limited to the pixel footprint
    float jw = 0.012;
    float jy = min(cy, 1.0 - cy) * CH, jx = min(cx, 1.0 - cx) * bl;
    float joint = max(1.0 - smoothstep(jw * 0.5, jw * 0.5 + foot, jy), 1.0 - smoothstep(jw * 0.5, jw * 0.5 + foot, jx));
    alb = mix(alb, alb * 0.74, joint * keepW(0.25, foot));
    // plinth course (grey concrete)
    float plinth = 1.0 - smoothstep(-0.01, 0.01, P.y - ${(LAND.backY + 0.55).toFixed(3)});
    alb = mix(alb, vec3(0.36, 0.35, 0.33) * (0.93 + 0.14 * vnoise(P.xy * 4.0)), plinth);
    // weep holes: irregular spacing (4-8.5 m) and size (+-40 %), a dark rust / algae streak below each
    float ws = floor(P.x / 6.1);
    vec3 wr = hash32(vec2(ws, 5.3));
    float wx = (ws + 0.2 + 0.6 * wr.x) * 6.1, wsz = mix(0.6, 1.4, wr.y);
    vec2 wh = vec2(P.x - wx, P.y - ${(LAND.backY + 0.95).toFixed(3)} - 0.25 * (wr.z - 0.5));
    float weep = (1.0 - smoothstep(0.035 * wsz, 0.035 * wsz + foot, abs(wh.x))) * (1.0 - smoothstep(0.05 * wsz, 0.05 * wsz + foot, abs(wh.y)));
    alb *= 1.0 - 0.8 * weep * step(0.12, wr.z);
    alb *= 1.0 - 0.25 * step(0.12, wr.z) * (1.0 - smoothstep(0.03, 0.08 * wsz, abs(wh.x))) * smoothstep(0.0, -0.05, wh.y) * smoothstep(-0.9 * wsz, -0.05, wh.y);
    // runoff streaks from the coping, grime and salt at the foot (spray, damp from the pebbles)
    float streak = vnoise(vec2(P.x * 2.6, P.y * 0.22)) * vnoise(vec2(P.x * 0.7 + 3.0, 1.0));
    float top = smoothstep(${(Y_FACE - 2.5).toFixed(2)}, ${Y_FACE.toFixed(2)}, P.y);
    alb *= 1.0 - 0.28 * smoothstep(0.25, 0.7, streak) * (0.3 + 0.7 * top) * keepW(0.4, foot);
    // the pebble toe against the foot: damp contact line and a darker grimy band above it
    float gy = ${LAND.backY.toFixed(3)} + toeAt(P.x);
    float foot0 = 1.0 - smoothstep(0.0, 1.1, P.y - gy + 0.25 * vnoise(vec2(P.x * 1.3, 0.0)));
    alb = mix(alb, alb * vec3(0.66, 0.66, 0.63), foot0);
    alb *= 1.0 - 0.35 * (1.0 - smoothstep(0.0, 0.12 + 0.06 * vnoise(vec2(P.x * 3.0, 2.0)), P.y - gy));
  } else if (part <= 3) {
    // ---- coping: cast-stone blocks 0.9-1.3 m with 1 cm joints, weathered: darker drip line on
    // the soffit edge, rounded worn arris, grey-green lichen and salt blotches, rust spots
    float cb = P.x / 1.1 + 0.3;
    float cid = floor(cb);
    float cl = 1.1 * mix(0.85, 1.15, hash12(vec2(cid, 2.7)));
    float cxx = fract(cb);
    vec3 cr = hash32(vec2(cid, 9.1));
    alb = vec3(0.60, 0.57, 0.51) * (0.9 + 0.14 * cr.x) * (1.0 + 0.05 * (cr.y - 0.5) * vec3(1.0, 0.3, -0.8));
    float jc = min(cxx, 1.0 - cxx) * cl;
    alb *= 1.0 - 0.3 * (1.0 - smoothstep(0.005, 0.005 + foot, jc)) * keepW(0.2, foot);
    vec2 uvc = part == 3 ? P.xz : P.xy;
    float blot = vnoise(uvc * vec2(1.7, 3.0) + cr.xy * 30.0) * 0.6 + vnoise(uvc * 6.0 + 1.3) * 0.4;
    alb *= 1.0 - 0.16 * smoothstep(0.45, 0.8, blot);                                 // weathering
    float lich = smoothstep(0.62, 0.8, vnoise(uvc * 4.3 + 7.7)) * keepW(0.15, foot);
    alb = mix(alb, vec3(0.44, 0.47, 0.40), 0.35 * lich);                                // lichen
    alb *= 1.0 - 0.3 * step(0.93, vnoise(uvc * 23.0 + cr.z * 9.0)) * keepW(0.05, foot); // pits / rust
    if (part == 2) alb *= 0.93 - 0.08 * smoothstep(${(Y_FACE + 0.06).toFixed(3)}, ${Y_FACE.toFixed(3)}, P.y);   // drip edge
    if (part == 1) alb *= 0.82;
    if (part == 3) alb *= 1.0 + 0.08 * (1.0 - smoothstep(0.0, 0.05, abs(P.z - ${(Z - CAP).toFixed(3)})));      // worn arris
  } else if (part <= 5) {
    // ---- parapet: cast-stone blocks (1.6 m, joints under the rail posts), weathered, grime from the posts
    float pid = floor(P.x / 1.6 + 0.5);
    vec3 pr = hash32(vec2(pid, 21.7));
    alb = vec3(0.60, 0.58, 0.53) * (0.93 + 0.1 * pr.x) * (0.95 + 0.1 * vnoise(P.xy * 1.3 + pr.yz * 20.0));
    alb *= 1.0 - 0.1 * smoothstep(0.55, 0.85, vnoise(P.xy * vec2(2.5, 3.5) + 9.0)) * keepW(0.3, foot);
    float pj = abs(fract(P.x / 1.6 + 0.5) - 0.5) * 1.6;
    alb *= 1.0 - 0.25 * (1.0 - smoothstep(0.004, 0.004 + foot, pj)) * keepW(0.2, foot);
    float pp = pj;                                                                       // below each post
    alb *= 1.0 - 0.18 * (1.0 - smoothstep(0.02, 0.09, pp)) * smoothstep(${(Y_PAR - 0.5).toFixed(2)}, ${Y_PAR.toFixed(2)}, P.y) * keepW(0.1, foot);
  } else {
    // ---- railing tube: painted steel (weathered off-white)
    alb = vec3(0.66, 0.67, 0.66) * (0.9 + 0.1 * vnoise(vec2(P.x * 3.0, 0.0)));
  }
  // ---- lighting: sun (the coping shades the top of the face), sky ambient, occlusion
  vec3 L = uSunDir;
  float NoL = max(dot(N, L), 0.0);
  float sh = 1.0;
  if (part == 0) sh = 1.0 - smoothstep(-0.02, 0.02, P.y - (${Y_FACE.toFixed(3)} - ${CAP.toFixed(3)} * L.y / max(-L.z, 0.05)));
  sh *= coastShadow(Pb, N);
  float ao = 1.0;
  if (part == 0) ao = (0.72 + 0.28 * smoothstep(0.0, 0.8, P.y - ${LAND.backY.toFixed(3)})) * (0.82 + 0.18 * smoothstep(0.0, 0.5, ${Y_FACE.toFixed(3)} - P.y));
  if (part == 1) ao = 0.7;
  if (part == 3) ao = 0.9 + 0.1 * smoothstep(${(Z - SET - 0.08).toFixed(3)}, ${(Z - CAP).toFixed(3)}, P.z);   // (in the parapet's corner)
  vec3 col = alb * (uSunColor * (NoL * sh / PI) + skyAmbient(N) * ao);
  // weak sheen on the smoothed cast stone / paint
  vec3 H = normalize(V + L);
  col += uSunColor * (part == 6 ? 0.06 : 0.02) * pow(max(dot(N, H), 0.0), part == 6 ? 40.0 : mix(34.0,15.0,surfaceRough)) * NoL * sh;
  // aerial perspective far along the shore
  float haze = landHaze(vViewZ);
  if (haze > 0.0) col = mix(col, landHazeColor(-V), haze);
  gl_FragColor = vec4(col, 1.0);
}`;

// ---- instanced street furniture: railing posts (world slots of 1.6 m around the focus) and the
// promenade lamp posts (every 27 m: tapered pole, swan-neck arm, lantern)
const N_POSTS = 96, POST_STEP = 1.6;
const N_LAMPS = 160, LAMP_STEP = 27.0;   // +-2.2 km
const LAMP_PARTS = [
  // [dx, y0, dz, half x, height, half z, tint]  (box in lamp space: x along the shore, z toward land)
  [0, 0.0, 0, 0.16, 0.55, 0.16, 0],     // plinth
  [0, 0.55, 0, 0.075, 5.2, 0.075, 0],   // pole
  [0, 5.75, 0, 0.055, 1.6, 0.055, 0],   // upper pole
  [0, 7.25, -0.42, 0.04, 0.06, 0.46, 0], // arm toward the sea
  [0, 6.72, -0.84, 0.14, 0.5, 0.14, 1],  // lantern
  [0, 7.22, -0.84, 0.19, 0.08, 0.19, 0], // lantern cap
];

const FURN_VERT = /* glsl */ `
uniform vec2 uBeachFocus;
uniform float uKind;          // 0 railing posts, 1 lamp posts
uniform vec4 uLampParts[${LAMP_PARTS.length * 2}];
out vec3 vW;
out vec3 vN;
out float vTint;
out float vViewZ;
void main() {
  int id = gl_InstanceID;
  vec3 p = position, n = normal;
  float x0, z0, y0;
  vTint = 0.0;
  if (uKind < 0.5) {
    float s = floor(uBeachFocus.x / ${POST_STEP.toFixed(2)}) + float(id - ${N_POSTS / 2});
    x0 = s * ${POST_STEP.toFixed(2)}; z0 = ${R_Z.toFixed(3)}; y0 = ${Y_PAR.toFixed(3)};
    p *= vec3(0.018, ${(R_Y - Y_PAR).toFixed(3)}, 0.018);
    // Subpixel posts narrow smoothly to zero before either end can be recycled.
    p.xz *= 1.0 - smoothstep(48.0, 72.0, abs(x0 - uBeachFocus.x));
  } else {
    int lamp = id / ${LAMP_PARTS.length}, k = id - lamp * ${LAMP_PARTS.length};
    float s = float(lamp - ${N_LAMPS / 2});
    x0 = s * ${LAMP_STEP.toFixed(1)};
    z0 = ${(Z + 1.1).toFixed(2)}; y0 = ${Y_CAP.toFixed(3)};
    vec4 a = uLampParts[k * 2], b = uLampParts[k * 2 + 1];
    p = p * vec3(b.x, b.y, b.z) + a.xyz;
    vTint = b.w;
  }
  vec3 P = p + vec3(x0, y0, z0);
  vW = P;
  vN = bayRotate(n, P.x);
  vec4 mv = viewMatrix * vec4(P - vec3(0.0, 0.0, bayBend(P.x)), 1.0);
  vViewZ = -mv.z;
  gl_Position = projectionMatrix * mv;
}`;

const FURN_FRAG = /* glsl */ `
uniform float uKind;
in vec3 vW;
in vec3 vN;
in float vTint;
in float vViewZ;
#define PI 3.14159265
void main() {
  vec3 P = vW;
  vec3 V = normalize(cameraPosition - P + vec3(0.0, 0.0, bayBend(P.x)));
  vec3 N = normalize(vN);
  // painted steel: off-white railing, dark green-grey lamp posts; the lantern's frosted glass
  vec3 alb = uKind < 0.5 ? vec3(0.64, 0.65, 0.64) : vec3(0.10, 0.12, 0.11);
  if (vTint > 0.5) alb = vec3(0.55, 0.56, 0.52);
  vec3 L = uSunDir;
  float NoL = max(dot(N, L), 0.0);
  vec3 col = alb * (uSunColor * (NoL / PI) + skyAmbient(N));
  vec3 H = normalize(V + L);
  col += uSunColor * 0.08 * pow(max(dot(N, H), 0.0), 30.0) * NoL;
  float haze = landHaze(vViewZ);
  if (haze > 0.0) col = mix(col, landHazeColor(-V), haze);
  gl_FragColor = vec4(col, 1.0);
}`;

export class Promenade {
  constructor(shared, macroNoise) {
    const uniforms = {
      ...coastUniforms(shared),
      uBeachFocus: shared.uFocus, uTime: shared.uTime, uSunDir: shared.uSunDir, uSunColor: shared.uSunColor,
      uSkyAmb: shared.uSkyAmb, uSkyLut: shared.uSkyLut, uMacroNoise: { value: macroNoise },
      uStone:{value:null},uStoneNormal:{value:null},uStoneRough:{value:null},
    };
    const pre = glslDefines() + NOISE + SKY + HAZE_GLSL + BEND_GLSL + MACRO_NOISE_GLSL + COAST_SHADOW;
    this.material = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: pre + VERT,
      fragmentShader: pre + FRAG,
      side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(buildGeometry(), this.material);
    this.mesh.frustumCulled = false;

    // instanced furniture: one material, two draws (posts, lamps)
    const box = boxGeometry();
    const lampParts = [];
    for (const [dx, y0, dz, hx, h, hz, tint] of LAMP_PARTS) lampParts.push(new THREE.Vector4(dx, y0, dz, 0), new THREE.Vector4(hx, h, hz, tint));
    const mk = (kind, count) => {
      const g = box.clone();
      g.instanceCount = count;
      const m = new THREE.ShaderMaterial({
        uniforms: { ...uniforms, uKind: { value: kind }, uLampParts: { value: lampParts } },
        vertexShader: pre + FURN_VERT,
        fragmentShader: pre + FURN_FRAG,
      });
      const mesh = new THREE.Mesh(g, m);
      mesh.frustumCulled = false;
      return mesh;
    };
    this.posts = mk(0, N_POSTS);
    this.lamps = mk(1, N_LAMPS * LAMP_PARTS.length);
    this.mesh.add(this.posts, this.lamps);
  }
  async loadAssets(renderer){
    const loader=new THREE.TextureLoader();
    await Promise.all([['diff','uStone'],['normal','uStoneNormal'],['rough','uStoneRough']].map(async([kind,key])=>{
      const t=await loader.loadAsync(`/assets/stone-r6/stone-${kind}.webp`);
      t.wrapS=t.wrapT=THREE.RepeatWrapping;t.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());
      if(kind==='diff')t.colorSpace=THREE.SRGBColorSpace;
      renderer.initTexture(t);this.material.uniforms[key].value=t;
    }));
  }
}
