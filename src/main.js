import { checkShaderBudget } from './core/shaderBudget.js';
import { SCENE_AREA, SCENE_SCALE, PIXEL_BUDGET_SCALE, normalizeResolutionScale, usesAdaptiveResolution, renderPixelRatio } from './core/viewport.js';
import * as THREE from 'three';
import { FixedClock } from './core/clock.js';
import { CONFIG, sunDirection, deg } from './config.js';
import { Schedule } from './core/schedule.js';
import { skyLut, LIGHT, cameraSway } from './core/look.js';
import { ExploreControls } from './core/explore.js';
import { bedProfileJS } from './config.js';
import { FullscreenPass, makeShader } from './core/gpu.js';
import { SwashSim } from './swash/SwashSim.js';
import { Swell } from './water/Swell.js';
import { UnderwaterPlume } from './water/UnderwaterPlume.js';
import { UnderwaterBubbles } from './water/UnderwaterBubbles.js';
import { SurfaceProbe } from './water/SurfaceProbe.js';
import { WaterSurface, makeChop } from './water/WaterSurface.js';
import { LipRibbon } from './water/LipRibbon.js';
import { Beach } from './beach/Beach.js';
import { initCoastalBed } from './beach/CoastalBed.js';
import { loadPalms } from './beach/Palms.js';
import { bakeCoastShadows } from './beach/CoastMaterial.js';
import { Sky } from './sky/Sky.js';
import { Whitewater } from './whitewater/Whitewater.js';
import { Post } from './post/Post.js';
import { ContactLight, CONTACT_SAMPLE } from './post/ContactLight.js';

const __tStart = performance.now();
const __marks = [];
const mark = (n) => __marks.push([n, Math.round(performance.now() - __tStart)]);
window.__marks = __marks;
const qs = new URLSearchParams(location.search);
const num = (k, d) => { const n = Number.parseFloat(qs.get(k)); return Number.isFinite(n) ? n : d; };
const CAPTURE = qs.has('capture');
const START_T = num('t', 0);
let speed = Math.min(2, Math.max(0.05, num('speed', 1)));
let paused = qs.has('paused') || CAPTURE;
const LOOP = qs.has('loop') ? num('loop', CONFIG.sim.videoDuration) : 0;
const DPR = Math.min(num('dpr', window.devicePixelRatio || 1), 2);
const FIXED_W = qs.has('w') ? parseInt(qs.get('w'), 10) : 0;
const FIXED_H = qs.has('h') ? parseInt(qs.get('h'), 10) : 0;
const SHAKE = !qs.has('noshake') && !matchMedia('(prefers-reduced-motion: reduce)').matches;
// Explore uses a centered frame with 75% of the browser's area.
const EXPLORE = CONFIG.explore;
const FRAMED = EXPLORE && !CAPTURE;
document.documentElement.style.setProperty('--scene-scale', String(FRAMED ? SCENE_SCALE : 1));
document.body.classList.toggle('capture', CAPTURE);
const walking = EXPLORE;
const loading = window.__loading || (() => {});
const breathe = () => new Promise(r => setTimeout(r, 0));
const nextPaint = () => new Promise(r => requestAnimationFrame(r));
window.__ready = false;
// The loading overlay blocks pointer input; inert also blocks keyboard focus
// and settings changes while their render targets are being prepared.
const chrome = [...document.querySelectorAll('.chrome')];
chrome.forEach(el => { el.inert = true; });
loading('Shaping the seabed', 8);
await breathe();
if (EXPLORE) document.body.classList.add('explore');

// ------------------------------------------------------------------ renderer
const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance', alpha: false, preserveDrawingBuffer: CAPTURE });
renderer.autoClear = false;
renderer.debug.onShaderError = (context, program, vertexShader, fragmentShader) => {
  const logs = [context.getProgramInfoLog(program),
    vertexShader && context.getShaderInfoLog(vertexShader),
    fragmentShader && context.getShaderInfoLog(fragmentShader)].filter(Boolean);
  console.error('Shader compile details', ...logs);
  throw new Error('A graphics program could not compile: ' + logs.join('\n'));
};
renderer.setPixelRatio(1);
const stage = document.getElementById('stage');
stage.appendChild(renderer.domElement);
const gl = renderer.getContext();
if (!renderer.capabilities.isWebGL2 && !(gl instanceof WebGL2RenderingContext)) throw new Error('WebGL2 required');
if (!renderer.extensions.get('EXT_color_buffer_float')) throw new Error('Floating-point rendering is required.');
renderer.domElement.addEventListener('webglcontextlost', (event) => { event.preventDefault(); window.__loadError(new Error('Graphics context lost')); });
renderer.domElement.addEventListener('webglcontextrestored', () => location.reload());
renderer.extensions.get('OES_texture_float_linear');
const maxRenderDimension = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE),
  gl.getParameter(gl.MAX_RENDERBUFFER_SIZE), ...gl.getParameter(gl.MAX_VIEWPORT_DIMS));

// ------------------------------------------------------------------ shared uniforms
const V3 = (a) => new THREE.Vector3(...a);
const shared = {
  uFocus: { value: new THREE.Vector2(0, 4.1) },
  uTime: { value: 0 },
  uEvtA: { value: new Float32Array(24) },
  uEvtB: { value: new Float32Array(24) },
  uEvtC: { value: new Float32Array(24) },
  uEvtD: { value: new Float32Array(24) },
  uEvtE: { value: new Float32Array(24) },
  uEvtF: { value: new Float32Array(24) },
  uEvtG: { value: new Float32Array(24) },
  uEvtCount: { value: 0 },
  uSunDir: { value: V3(sunDirection()) },
  uSunColor: { value: V3(LIGHT.sun) },
  uSkyAmb: { value: V3(LIGHT.skyAmb) },
  uSkyLut: { value: skyLut() },
  uNear: { value: CONFIG.camera.near },
  uFar: { value: CONFIG.camera.far },
  uInjMass: { value: num('inj', 1.7) },
  uInjSpeed: { value: num('injv', 0.95) },
  uChop: { value: makeChop(20) },
  uWind: { value: new THREE.Vector2(Math.sin(deg(8)), Math.cos(deg(8))) },
  uResolution: { value: new THREE.Vector2(1, 1) },
  uOpaqueColor: { value: null },
  uOpaqueDepth: { value: null },
  uBackDepth: { value: null },
  uDeepColor: { value: V3([0.017, 0.16, 0.22]) },
  uWaterAtten: { value: V3([0.55, 0.085, 0.10]) },
  uMilkColor: { value: V3([0.62, 0.72, 0.66]) },
  uGlowTint: { value: V3([0.30, 1.0, 0.85]) },
  uLaceSeed: { value: new THREE.Vector2(0.3, 0.7) },
};

// ------------------------------------------------------------------ world
const schedule = new Schedule({ seed: num('seed', 7) });
mark('init');
const swell=EXPLORE?new Swell(shared,schedule):null;
if(EXPLORE) initCoastalBed(shared);
const swe = new SwashSim(renderer, shared);
mark('swe');
loading('Preparing the water', 14);
await breathe();
mark('pre-water');
const water = new WaterSurface(shared, { cols: Math.round(num('cols', 360)), swell });
const lips = [new LipRibbon(shared, 0), new LipRibbon(shared, 1)];
mark('lips');
loading('Shaping the sandy shoreline', 21);
await breathe();
const beach = new Beach(renderer, shared);
mark('beach');
if (EXPLORE) {
  loading('Planting the promenade', 25);
  await breathe();
  [beach.palms] = await Promise.all([loadPalms(renderer, shared), beach.promenade.loadAssets(renderer),beach.loadAssets(renderer)]);
  beach.mesh.add(beach.palms);
  shared.uSandPhoto.value=beach.material.uniforms.uPhotoColor.value;
  loading('Preparing the coastal light', 27);
  beach.coastShadow = bakeCoastShadows(renderer, [beach.backdrop.seafront.group, beach.palms, beach.rocks.mesh], shared);
  mark('coast-assets');
}
const sky = new Sky(shared);
mark('sky');
loading('Preparing foam and spray', 28);
await breathe();
const whitewater = new Whitewater(renderer, shared, schedule);
const bubbles=EXPLORE?new UnderwaterBubbles(shared,schedule):null;
const plume=EXPLORE?new UnderwaterPlume(shared,bubbles.material.uniforms):null;
const underScene=new THREE.Scene();underScene.add(water.underMesh);
mark('whitewater');

const DEBUG_HIDE = new Set((qs.get('hide') || '').split(',').filter(Boolean)); // water,lip,beach,sky,particles
if (DEBUG_HIDE.has('water')) { water.mesh.visible = false; water.backMesh.visible = false; }
if (DEBUG_HIDE.has('beach')) beach.mesh.visible = false;
if (DEBUG_HIDE.has('sky')) sky.mesh.visible = false;
const opaqueScene = new THREE.Scene();
opaqueScene.add(sky.mesh, beach.mesh);
const backScene = new THREE.Scene();
backScene.add(water.backMesh, ...lips.map((l) => l.backMesh));
const waterScene = new THREE.Scene();
waterScene.add(water.mesh, ...lips.map((l) => l.mesh));

// ------------------------------------------------------------------ camera
const C = CONFIG.camera;
const camera = new THREE.PerspectiveCamera(C.vfov, C.aspect, C.near, C.far);
camera.rotation.order = 'YXZ';
const surfaceProbe=EXPLORE?new SurfaceProbe(shared):null;
if(surfaceProbe)renderer.initRenderTarget(surfaceProbe.target);
const controls = EXPLORE ? new ExploreControls(renderer.domElement, camera, {
  ground: (x, z) => (beach.groundAt ? beach.groundAt(x, z) : bedProfileJS(z)),
  water: (x,z,t)=>surfaceProbe.sample(x,z,t),
  bounds: { xMin: -60, xMax: 60, zMin: -14.0, zMax: 36 },
  start: { x: -34, z: 6.8, yaw: deg(-94), pitch: deg(-4.2) },
  reducedMotion: !SHAKE,
}) : null;
if (controls) controls.enabled = walking;
let frameDt = 0;
function updateCamera(t) {
  if (controls) { controls.update(frameDt,t); return; }
  camera.position.fromArray(C.position);
  if (SHAKE) {
    const [rx, ry, rz] = cameraSway(t);
    camera.rotation.set(rx, ry, rz);
  } else camera.rotation.set(deg(C.pitchDeg), deg(C.yawDeg), deg(C.rollDeg));
  camera.updateMatrixWorld();
}

// ------------------------------------------------------------------ render targets
let W = 0, H = 0;
let targetRevision = 0;
let rtOpaque, rtBack, rtMain, rtParticles, post;
const contactLight = new ContactLight();
function makeTargets(w, h) {
  targetRevision++;
  [rtOpaque, rtBack, rtMain, rtParticles].forEach((r) => r && r.dispose());
  const hdr = { type: THREE.HalfFloatType, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: false };
  const msaa = 2;
  const opaqueSamples=quality==='high'||quality==='ultra'?4:2;
  rtOpaque = new THREE.WebGLRenderTarget(w, h, { ...hdr, samples: opaqueSamples, depthTexture: new THREE.DepthTexture(w, h, THREE.UnsignedIntType) });
  rtMain = new THREE.WebGLRenderTarget(w, h, { ...hdr, samples: msaa, depthTexture: new THREE.DepthTexture(w, h, THREE.UnsignedIntType) });
  rtBack = new THREE.WebGLRenderTarget(w, h, { type: THREE.FloatType, format: THREE.RedFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: true });
  rtParticles = new THREE.WebGLRenderTarget(w, h, { ...hdr });
  for (const r of [rtOpaque, rtMain, rtBack, rtParticles]) r.texture.colorSpace = THREE.NoColorSpace;
  if (post) post.resize(w, h); else post = new Post(renderer, w, h, shared);
  contactLight.resize(w,h);
  plume?.resize(w,h);
  shared.uResolution.value.set(w, h);
  shared.uOpaqueColor.value = rtOpaque.texture;
  shared.uOpaqueDepth.value = rtOpaque.depthTexture;
  shared.uBackDepth.value = rtBack.texture;
}

// Internal resolution = CSS size x DPR x dynamic scale (adaptive resolution keeps
// weaker GPUs near 60 fps; captures always render at the exact requested size).
const QUALITY = { auto: { pixels: 1800000, dpr: 1.5, foam: 0.6 }, low: { pixels: 560000, dpr: 1, foam: 0.5 }, medium: { pixels: 1050000, dpr: 1.25, foam: 0.55 }, high: { pixels: 2200000, dpr: 1.5, foam: 0.7 }, ultra: { pixels: 4200000, dpr: 2, foam: 0.85 } };
let quality = 'auto';
try { const saved = localStorage.getItem('shorebreak-quality'); if (saved in QUALITY) quality = saved; } catch {}
let resolutionScale = 1;
try { resolutionScale = normalizeResolutionScale(localStorage.getItem('shorebreak-resolution')); } catch {}
let dynScale = 1;
const renderSize = new THREE.Vector2();
function resize() {
  let cw, ch;
  if (FIXED_W && FIXED_H) {
    cw = FIXED_W; ch = FIXED_H;
    if (EXPLORE) { camera.aspect = cw / ch; camera.fov = 58; camera.updateProjectionMatrix(); }
  } else {
    const ww = window.innerWidth, wh = window.innerHeight;
    if (EXPLORE) {
      const rect = stage.getBoundingClientRect();
      cw = Math.max(1, rect.width); ch = Math.max(1, rect.height);
      camera.aspect = cw / ch;
      camera.fov = 58;
      camera.updateProjectionMatrix();
    } else if (ww / wh > C.aspect) { ch = wh; cw = Math.round(wh * C.aspect); } else { cw = ww; ch = Math.round(ww / C.aspect); }
  }
  const q = QUALITY[quality];
  if(water.ex){
    const colPx=quality==='high'||quality==='ultra'?7:quality==='low'?10:9;
    if(water.ex.params.colPx!==colPx){water.ex.params.colPx=colPx;water.ex.last=null;}
  }
  whitewater.params.bufS = q.foam;
  const boost = CAPTURE ? 1 : resolutionScale;
  whitewater.params.bufH = (quality === 'ultra' ? 1440 : 1080) * boost;
  const k = renderPixelRatio(cw, ch, { dpr: DPR, maxDpr: q.dpr, pixels: q.pixels,
    budgetScale: FRAMED ? PIXEL_BUDGET_SCALE : 1, dynamicScale: dynScale,
    resolutionScale: boost, maxDimension: maxRenderDimension, exact: !!(FIXED_W && FIXED_H) });
  renderer.setPixelRatio(k);
  renderer.setSize(cw, ch, true);
  renderer.getDrawingBufferSize(renderSize);
  const w = renderSize.x, h = renderSize.y;
  const samples=quality==='high'||quality==='ultra'?4:2;
  if (w !== W || h !== H || rtOpaque?.samples!==samples) { W = w; H = h; makeTargets(W, H); }
  window.__viewportInfo = { cssWidth: cw, cssHeight: ch, width: w, height: h,
    pixelRatio: k, screenArea: FRAMED ? SCENE_AREA : 1, resolutionScale: boost,
    pixelCap: q.pixels * (FRAMED ? PIXEL_BUDGET_SCALE : 1) * boost * boost,
    adaptiveResolution: usesAdaptiveResolution(quality, boost), maxRenderDimension };
}
const dyn = { acc: 0, n: 0, cooldown: 2 };
function adaptResolution(dt) {
  if (!usesAdaptiveResolution(quality, resolutionScale) || CAPTURE || FIXED_W || qs.has('dpr') || paused || document.hidden) return;
  dyn.acc += dt; dyn.n++;
  if (dyn.acc < 1.0) return;
  const ms = (dyn.acc / dyn.n) * 1000;
  dyn.acc = 0; dyn.n = 0;
  if (dyn.cooldown > 0) { dyn.cooldown--; return; }
  let next = dynScale;
  if (ms > 23) next = Math.max(0.5, dynScale * Math.max(0.82, Math.sqrt(18 / ms)));
  else if (ms < 15 && dynScale < 1) next = Math.min(1, dynScale * 1.1);
  if (Math.abs(next - dynScale) > 0.04) { dynScale = next; resize(); dyn.cooldown = 2; }
}
let resizeTimer;
const queueResize = () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { resizeTimer = null; resize(); }, 120);
};
window.addEventListener('resize', queueResize);
document.addEventListener('fullscreenchange', queueResize);
new ResizeObserver(queueResize).observe(stage);
resize();

// copy opaque colour + depth into the main target
const copyPass = new FullscreenPass(makeShader(/* glsl */ `
uniform sampler2D uColor;
uniform sampler2D uDepth;
${CONTACT_SAMPLE}
in vec2 vUv;
void main() { float d=texture(uDepth,vUv).x;float ao=contactAt(vUv,d);gl_FragColor = vec4(texture(uColor, vUv).rgb*ao, 1.0); gl_FragDepth = d; }`,
{ uColor: { value: null }, uDepth: { value: null },uContact:{value:null},uContactTexel:{value:new THREE.Vector2()},uCameraRange:{value:new THREE.Vector2(C.near,C.far)} }, { depthTest: true, depthWrite: true, depthFunc: THREE.AlwaysDepth }));

// ------------------------------------------------------------------ simulation clock
const DT = CONFIG.sim.dt;
const clock = new FixedClock(DT);
let simT = 0;
// the world recentres its meshes / simulation window on the player (explore) or the clip camera
function updateFocus() {
  if (controls) shared.uFocus.value.set(controls.pos.x, controls.pos.z);
  else shared.uFocus.value.set(C.position[0], C.position[2]);
}
function stepSim() {
  updateFocus();
  schedule.pack(simT);
  shared.uEvtA.value.set(schedule.A); shared.uEvtB.value.set(schedule.B); shared.uEvtC.value.set(schedule.C);
  shared.uEvtD.value.set(schedule.D); shared.uEvtE.value.set(schedule.E); shared.uEvtF.value.set(schedule.F); shared.uEvtG.value.set(schedule.G);
  shared.uEvtCount.value = schedule.count;
  shared.uTime.value = simT;
  swe.step(simT, DT);
  whitewater.step(simT, DT);
  simT += DT;
}
const __px = new Uint8Array(4);
const gpuSync = () => { renderer.setRenderTarget(null); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, __px); };
function seek(t) {
  clock.reset();
  surfaceProbe?.reset();
  if (controls) { frameDt = 0; controls.update(0); }
  updateFocus();
  swe.reset();
  whitewater.reset();
  simT = t - CONFIG.sim.warmup;
  if (!window.__seeded) {
    // first run: time the first-use compiles of each subsystem
    schedule.pack(simT);
    shared.uTime.value = simT;
    swe.step(simT, DT); gpuSync(); mark('swe-step');
    whitewater.step(simT, DT); gpuSync(); mark('ww-step');
    swe.reset(); whitewater.reset();
    window.__seeded = true;
  }
  while (simT < t - 1e-9) stepSim();
}

// ------------------------------------------------------------------ auto-exposure
// The phone's metering pumps ±0.15 EV: it darkens ~0.25 s after whitewater fills the
// lower frame and recovers before the next break (far-sea analysis §2.2). Inside the
// clip we replay the measured curve (sky G channel, BT.709); outside it we drive the same
// behaviour from the wave schedule, so it stays deterministic for captures.
const AE_T = [0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.25, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5];
const AE_G = [197, 195, 191, 192, 198, 203, 204, 198, 197, 197, 202, 205, 205, 205, 204, 201];
const AE_REF = 199.6;
const smooth01 = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
function aeMeasured(t) {
  let i = 0;
  while (i < AE_T.length - 2 && t > AE_T[i + 1]) i++;
  const f = Math.min(Math.max((t - AE_T[i]) / (AE_T[i + 1] - AE_T[i]), 0), 1);
  const g = AE_G[i] + (AE_G[i + 1] - AE_G[i]) * f;
  return Math.pow(g / AE_REF, 2.2);
}
function aeModel(t) {
  let ev = 0.05;
  for (const e of schedule.events) {
    const tau = t - (e.t0 + 0.12);
    if (tau < -1 || tau > 5) continue;
    ev -= 0.2 * (e.H / 0.5) ** 2 * smooth01(0.3, 1.6, tau) * (1 - smooth01(2.4, 3.7, tau));
  }
  return Math.pow(2, ev);
}
function exposureAt(t) {
  if (qs.get('ae') === '0') return 1;
  if (EXPLORE) return 1; // GPU metering uses the current view, independently of wave timing.
  const w = smooth01(-0.6, 0, t) * (1 - smooth01(8.5, 9.1, t));
  return aeModel(t) * (1 - w) + aeMeasured(Math.min(Math.max(t, 0), 8.5)) * w;
}

// ------------------------------------------------------------------ frame
function render(warmBothLips = false) {
  schedule.pack(simT);
  shared.uEvtA.value.set(schedule.A); shared.uEvtB.value.set(schedule.B); shared.uEvtC.value.set(schedule.C);
  shared.uEvtD.value.set(schedule.D); shared.uEvtE.value.set(schedule.E); shared.uEvtF.value.set(schedule.F); shared.uEvtG.value.set(schedule.G);
  shared.uEvtCount.value = schedule.count;
  shared.uTime.value = simT;
  updateCamera(simT);
  // world focus (player position) for modules that recentre their meshes / simulation
  shared.uFocus.value.set(camera.position.x, camera.position.z);
  // assign lip ribbons to the events whose lip can be airborne now
  const slots = schedule.jetSlots(simT, camera.position.x);
  lips.forEach((l, i) => {
    l.visible = (i < slots.length || (warmBothLips && schedule.count > 0)) && !DEBUG_HIDE.has('lip');
    if (l.mesh.visible) l.slot = slots[i] ?? Math.min(i, schedule.count - 1);
  });

  renderer.setClearColor(0x000000, 1);
  // 1) opaque: sky + beach/seabed
  renderer.setRenderTarget(rtOpaque);
  renderer.clear(true, true, false);
  renderer.render(opaqueScene, camera);
  contactLight.render(renderer,rtOpaque.depthTexture,camera,W,H);
  // 2) water back faces -> 1/viewZ of the exit surface (0 = none)
  renderer.setRenderTarget(rtBack);
  renderer.setClearColor(0x000000, 0);
  renderer.clear(true, true, false);
  renderer.render(backScene, camera);
  // Query after the FFT update: optics use this frame's GPU surface.
  if(surfaceProbe)surfaceProbe.request(renderer,camera,simT,camera.position.x,camera.position.z);
  const waterDepth=surfaceProbe?surfaceProbe.sample(camera.position.x,camera.position.z,simT).height-camera.position.y:-100;
  const underwater=EXPLORE&&camera.position.z<1.5&&(waterDepth>-.3||camera.position.y<.55);
  shared.uUnderwaterOn.value=underwater?1:0;
  post.waterDepth=waterDepth;
  // Once fully immersed, only the underside can be seen. Reuse its buffers
  // without submitting the full front surface as well.
  water.mesh.visible=!DEBUG_HIDE.has('water')&&!(waterDepth>.10&&camera.position.y<=-.10);
  // 3) main: opaque copy + water
  renderer.setRenderTarget(rtMain);
  renderer.setClearColor(0x000000, 1);
  renderer.clear(true, true, false);
  copyPass.material.uniforms.uColor.value = rtOpaque.texture;
  copyPass.material.uniforms.uDepth.value = rtOpaque.depthTexture;
  copyPass.material.uniforms.uContact.value=contactLight.target.texture;
  copyPass.material.uniforms.uContactTexel.value.set(1/contactLight.target.width,1/contactLight.target.height);
  renderer.render(copyPass.scene, copyPass.camera);
  renderer.render(waterScene, camera);
  if(underwater){bubbles.update(simT);renderer.render(underScene,camera);renderer.render(bubbles.scene,camera);}
  // 4) whitewater (soft particles against the full scene depth)
  if (DEBUG_HIDE.has('particles')) { renderer.setRenderTarget(rtParticles); renderer.setClearColor(0x000000, 0); renderer.clear(true, false, false); }
  else whitewater.render(camera, rtParticles, rtMain.depthTexture, W, H);
  // 5) post to screen
  post.plumeTexture=underwater?plume.render(renderer,camera,rtMain.depthTexture):null;
  post.params.exposure = exposureAt(simT);
  post.frameDt=frameDt;
  post.render(rtMain.texture, rtParticles.texture, simT, null, camera,rtMain.depthTexture);
}

// ------------------------------------------------------------------ loop / API
const hud = document.getElementById('hud');
let last = performance.now(), fpsAcc = 0, fpsN = 0, fps = 0;
function frame(now) {
  const real = Math.min(Math.max((now - last) / 1000, 0), 0.1);
  last = now;
  frameDt = real;
  if (document.hidden || restarting) { if (!CAPTURE) requestAnimationFrame(frame); return; }
  if (!paused && !restarting) {
    clock.advance(real, speed, stepSim);
    if (LOOP && simT > LOOP) void restart(0);
  }
  render();
  adaptResolution(real);
  fpsAcc += real; fpsN++;
  if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; }
  if (qs.has('stats') && !qs.has('nohud')) hud.textContent = `t ${simT.toFixed(2)}s  ${fps.toFixed(0)} fps  ${W}x${H}${paused ? '  [paused]' : ''}`;
  if (!CAPTURE) requestAnimationFrame(frame);
}

function updatePlayback() {
  const btn = document.getElementById('btn-pause');
  btn.setAttribute('aria-label', paused ? 'Resume waves' : 'Pause waves');
  btn.title = paused ? 'Resume waves (P)' : 'Pause waves (P)';
  btn.innerHTML = paused ? '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6 3l11 7-11 7Z"/></svg>' : '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 4v12M13 4v12"/></svg>';
  document.getElementById('scene-state').textContent = paused ? 'A MOMENT, HELD' : speed < 1 ? 'SLOW THE MOMENT' : 'THE LIVING COASTLINE';
}
function togglePause() { paused = !paused; clock.reset(); updatePlayback(); }
function toggleHelp(force) {
  const help = document.getElementById('help');
  help.hidden = force === undefined ? !help.hidden : !force;
  document.getElementById('btn-help').setAttribute('aria-expanded', String(!help.hidden));
  if (!help.hidden && document.pointerLockElement) document.exitPointerLock();
}
function setQuality(value) {
  if (!(value in QUALITY)) return;
  quality = value; dynScale = 1; dyn.cooldown = 3; dyn.acc = dyn.n = 0;
  document.getElementById('quality').value = value;
  try { localStorage.setItem('shorebreak-quality', value); } catch {}
  resize();
}
function setResolution(value) {
  resolutionScale = normalizeResolutionScale(value);
  dynScale = 1; dyn.cooldown = 3; dyn.acc = dyn.n = 0;
  document.getElementById('resolution').value = String(resolutionScale);
  try { localStorage.setItem('shorebreak-resolution', String(resolutionScale)); } catch {}
  resize();
}
function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen?.();
  else { setQuality('auto'); document.documentElement.requestFullscreen?.().catch(() => {}); }
}
document.getElementById('btn-fs').onclick = toggleFullscreen;
document.getElementById('btn-pause').onclick = togglePause;
document.getElementById('btn-help').onclick = () => toggleHelp();
document.getElementById('close-help').onclick = () => toggleHelp(false);
document.getElementById('speed').onchange = e => { speed = Number(e.target.value); clock.reset(); updatePlayback(); };
document.getElementById('quality').onchange = e => setQuality(e.target.value);
document.getElementById('quality').value = quality;
document.getElementById('resolution').onchange = e => setResolution(e.target.value);
document.getElementById('resolution').value = String(resolutionScale);
document.getElementById('speed').value = String(speed);
hud.hidden = !qs.has('stats');
document.body.classList.toggle('walking', walking);
document.getElementById('explore-hint').hidden = !walking;
document.getElementById('crosshair').hidden = true;
updatePlayback();
if (qs.has('nohud')) document.body.classList.add('ui-hidden');
window.addEventListener('keydown', e => {
  if (e.repeat || e.target?.closest('input,select,textarea,button') || !window.__ready) return;
  if (e.code === 'KeyF') toggleFullscreen();
  if (e.code === 'KeyP') { e.preventDefault(); togglePause(); }
  if (e.code === 'KeyH') toggleHelp();
  if (e.code === 'KeyU') document.body.classList.toggle('ui-hidden');
  if (e.code === 'Escape') { toggleHelp(false); document.body.classList.remove('ui-hidden'); }
  if (e.code === 'KeyR') void restart(START_T);
  if (e.code === 'ArrowRight' && !walking) { e.preventDefault(); for (let i = 0; i < 12; i++) stepSim(); }
  const rates = { Digit1: 1, Digit2: 0.25, Digit3: 0.1 };
  if (e.code in rates) { speed = rates[e.code]; clock.reset(); document.getElementById('speed').value = String(speed); updatePlayback(); }
});
document.addEventListener('visibilitychange', () => { last = performance.now(); clock.reset(); dyn.acc = dyn.n = 0; });

// deterministic capture API (tools/capture.mjs)
window.__seek = (t) => { seek(t); render(); gl.finish(); return simT; };
window.__advance = (sec) => { const end = simT + sec - 1e-9; while (simT < end) stepSim(); render(); gl.finish(); return simT; };
window.__diag = () => renderer.info.programs.map((p) => ({ name: p.name, type: p.type, diag: p.diagnostics ? { runnable: p.diagnostics.runnable, log: p.diagnostics.programLog, frag: p.diagnostics.fragmentShader?.log?.slice(0, 600), vert: p.diagnostics.vertexShader?.log?.slice(0, 600) } : null }));
window.__draw = async () => { frameDt = 0; render(); await drainGPU(); };
window.__grab = () => renderer.domElement.toDataURL('image/png');
// debug: read back the shallow-water state -> stats + false-colour image (h | foam | speed)
window.__sweDebug = () => {
  const { nx, nz } = swe;
  const st = new Float32Array(nx * nz * 4), bd = new Float32Array(nx * nz * 4);
  renderer.readRenderTargetPixels(swe.state.read, 0, 0, nx, nz, st);
  renderer.readRenderTargetPixels(swe.bed, 0, 0, nx, nz, bd);
  const cv = document.createElement('canvas'); cv.width = nx * 3; cv.height = nz;
  const cx = cv.getContext('2d'); const img = cx.createImageData(nx * 3, nz);
  let maxH = 0, maxV = 0, wetBeach = 0, nan = 0, mass = 0;
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    const k = (j * nx + i) * 4;
    const w = st[k], hu = st[k + 1], hv = st[k + 2], B = bd[k];
    if (!Number.isFinite(w)) { nan++; continue; }
    const h = Math.max(w - B, 0), v = h > 1e-3 ? Math.hypot(hu, hv) / h : 0;
    maxH = Math.max(maxH, h); maxV = Math.max(maxV, v); mass += h;
    if (B > 0.05 && h > 0.003) wetBeach++;
    const row = (nz - 1 - j);
    const put = (off, r, g, b) => { const p = (row * nx * 3 + off + i) * 4; img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255; };
    const hc = Math.min(h / 0.6, 1) * 255;
    put(0, B > 0 ? 120 : 0, hc, B > 0 ? 60 : hc * 0.5 + 60);
    put(nx * 2, Math.min(v / 4, 1) * 255, 0, 0);
  }
  cx.putImageData(img, 0, 0);
  // depth profile along z at x = 0 (every ~25 cm)
  const prof = [];
  const i0 = Math.floor(nx / 2);
  for (let j = 0; j < nz; j += 8) {
    const k = (j * nx + i0) * 4;
    const z = CONFIG.swe.zMin + (j + 0.5) * swe.dz;
    prof.push([+z.toFixed(2), +bd[k].toFixed(3), +Math.max(st[k] - bd[k], 0).toFixed(4), +(st[k + 2] / Math.max(st[k] - bd[k], 1e-3)).toFixed(2)]);
  }
  return { maxH, maxV, wetBeach, nan, mass: mass * swe.dx * swe.dz, prof, img: cv.toDataURL() };
};
// GPU timing of n rendered frames (each with the usual 2 sim steps)
// GPU timing with EXT_disjoint_timer_query_webgl2 (wall-clock is meaningless when several
// processes share the GPU). Returns ms of GPU time per rendered frame / per sim step.
window.__bench = async (n = 60) => {
  const ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
  const px = new Uint8Array(4);
  const sync = () => { renderer.setRenderTarget(null); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); };
  const measure = async (fn) => {
    if (!ext) { sync(); const t0 = performance.now(); for (let i = 0; i < n; i++) fn(); sync(); return (performance.now() - t0) / n; }
    const qs = [];
    for (let i = 0; i < n; i++) { const q = gl.createQuery(); gl.beginQuery(ext.TIME_ELAPSED_EXT, q); fn(); gl.endQuery(ext.TIME_ELAPSED_EXT); qs.push(q); }
    sync();
    let total = 0, got = 0;
    for (const q of qs) {
      for (let k = 0; k < 200 && !gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE); k++) await new Promise((r) => setTimeout(r, 5));
      if (gl.getParameter(ext.GPU_DISJOINT_EXT)) continue;
      total += gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6; got++;
      gl.deleteQuery(q);
    }
    return got ? total / got : NaN;
  };
  const msRender = await measure(() => render());
  const msSim = await measure(() => stepSim());
  const msFrame = msRender + 2 * msSim;
  return { gpuMsPerFrame: +msFrame.toFixed(2), gpuMsRender: +msRender.toFixed(2), gpuMsPerSimStep: +msSim.toFixed(2), fps: +(1000 / msFrame).toFixed(1), timer: !!ext, W, H };
};
if (controls) {
  const crouchButton=document.getElementById('btn-crouch');
  crouchButton.onclick=()=>{if(window.__ready)controls.toggleCrouch();};
  const movementUI=()=>{
    const swim=controls.swimming||controls.diving;
    crouchButton.setAttribute('aria-pressed',String(swim?controls.diving:controls.crouched));
    crouchButton.textContent=swim?(controls.diving?'Surface · C':'Dive · C'):(controls.crouched?'Stand · C':'Crouch · C');
    crouchButton.title=swim?'Dip underwater / return to the surface (C)':'Crouch / stand (C)';
    document.getElementById('movement-state').textContent=controls.diving?'UNDERWATER':swim?'SWIMMING':controls.mode==='wade'?'WADING':'';
  };
  controls.onCrouchChange=movementUI;controls.onDiveChange=movementUI;controls.onModeChange=movementUI;
  const hint = document.getElementById('explore-hint');
  controls.onLockChange = (locked) => { hint?.classList.toggle('hide', locked); document.getElementById('crosshair').hidden = !locked; };
  document.getElementById('steady-camera').checked = controls.motion === 0;
  document.getElementById('steady-camera').onchange = e => { controls.motion = e.target.checked ? 0 : 1; };
  const knob = document.getElementById('stick');
  controls.onStick = (on, x, y, dx, dy) => {
    if (!knob) return;
    knob.style.display = on ? 'block' : 'none';
    if (on) { knob.style.left = `${x - 40}px`; knob.style.top = `${y - 40}px`; knob.firstElementChild.style.transform = `translate(${dx * 28}px, ${dy * 28}px)`; }
  };
}
window.__explore = controls;
window.__scene = { THREE, beach, water, sky, lips, whitewater, camera, renderer, swe, surfaceProbe };
window.__params = { post: () => post.params, shared, schedule, CONFIG };

// ------------------------------------------------------------------ start
const overlay = document.getElementById('loading');
// Compile every program up front and in parallel (KHR_parallel_shader_compile lets the GPU
// process compile on worker threads; under ANGLE/D3D these big shaders are slow to compile
// one after another). Programs are keyed like the real draws: offscreen passes with a render
// target bound, the final post pass to the canvas.
async function precompile() {
  const jobs = [];
  const finalPass = post.pAA;
  const lipVis = lips.map((l) => l.mesh.visible);
  lips.forEach((l) => { l.visible = true; });
  renderer.setRenderTarget(rtMain);
  const scenes = [[opaqueScene, camera], [backScene, camera], [waterScene, camera],[underScene,camera]];
  if(bubbles)scenes.push([bubbles.scene,camera]);
  if(swell)scenes.push([swell.scene,swell.camera]);
  for (const k of Object.keys(whitewater)) {
    const v = whitewater[k];
    if (v && v.isScene) scenes.push([v, camera]);
  }
  for (const [sc, cam] of scenes) jobs.push(renderer.compileAsync(sc, cam));
  for (const p of FullscreenPass.all) if (p !== finalPass) jobs.push(renderer.compileAsync(p.scene, p.camera));
  renderer.setRenderTarget(null);
  jobs.push(renderer.compileAsync(finalPass.scene, finalPass.camera));
  await Promise.all(jobs);
  window.__graphicsInfo = checkShaderBudget(gl, renderer.info.programs);
  lips.forEach((l, i) => { l.visible = lipVis[i]; });
  mark('precompiled');
}
// Wait for GPU work without a main-thread readPixels/finish stall. This also bounds
// command-queue growth during warmup, so the loader remains responsive on slow GPUs.
async function drainGPU() {
  const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
  if (!fence) throw new Error('Graphics initialization interrupted');
  gl.flush();
  try {
    while (true) {
      const result = gl.clientWaitSync(fence, 0, 0);
      if (result === gl.ALREADY_SIGNALED || result === gl.CONDITION_SATISFIED) break;
      if (result === gl.WAIT_FAILED || gl.isContextLost()) throw new Error('Graphics initialization interrupted');
      await new Promise(r => setTimeout(r, 8));
    }
  } finally { gl.deleteSync(fence); }
}
async function warmViews() {
  // Compilation alone does not upload vertex/index buffers or create every
  // VAO. Draw the complete opaque scene once, including off-camera instances.
  const culled = [];
  opaqueScene.traverse(object => {
    if (object.isMesh) { culled.push([object, object.frustumCulled]); object.frustumCulled = false; }
  });
  try { render(true); await drainGPU(); }
  finally { for (const [object, value] of culled) object.frustumCulled = value; }

  if (!controls) return;
  const saved = { x: controls.pos.x, z: controls.pos.z, eyeY: controls.eyeY,
    yaw: controls.yaw, pitch: controls.pitch };
  try {
    // Exercise moving water-mesh tables, both above/below-water paths, grazing
    // waterline views, and scenery turns before any of them can be seen.
    const views = [
      [saved.x + .12, saved.z, saved.eyeY, saved.yaw, saved.pitch],
      [saved.x - .12, saved.z, saved.eyeY - .3, 0, -.4],
      [saved.x, saved.z, saved.eyeY, Math.PI, -.06],
      [saved.x, saved.z, saved.eyeY, -Math.PI / 2, -.06],
      [saved.x, saved.z, saved.eyeY, Math.PI / 2, -.06],
      [saved.x, -6, .25, 0, -.15],
      [saved.x, -6, -.4, 0, .85],
      [saved.x, -6, -.4, Math.PI, -.35],
    ];
    for (const [x, z, eyeY, yaw, pitch] of views) {
      controls.pos.x = x; controls.pos.z = z; controls.eyeY = eyeY;
      controls.yaw = yaw; controls.pitch = pitch;
      render(true);
      await drainGPU();
      if (surfaceProbe?.pending) await surfaceProbe.ready;
      await nextPaint();
    }
  } finally {
    controls.pos.x = saved.x; controls.pos.z = saved.z; controls.eyeY = saved.eyeY;
    controls.yaw = saved.yaw; controls.pitch = saved.pitch;
    surfaceProbe?.reset();
    shared.uUnderwaterOn.value = 0;
    updateCamera(simT); updateFocus();
  }
}
async function warmAnimatedFrames() {
  // Rehearse the real fixed-step + render combination, not just isolated stills.
  // This covers alternating simulation/exposure buffers, changing FFT fields,
  // texture updates and first-use driver work. The final seek below restores the
  // requested wave time, so startup preparation never skips the opening waves.
  for (let i = 0; i < 30; i++) {
    await nextPaint();
    clock.advance(1 / 60, 1, stepSim);
    render(true);
    await drainGPU();
    loading('Preparing the first moments', 48 + 4 * (i + 1) / 30);
  }
  clock.reset();
  mark('animated-frames-warmed');
}
async function warmSimulation(t, report = true) {
  surfaceProbe?.reset();
  clock.reset(); frameDt = 0; updateCamera(t); updateFocus();
  swe.reset(); whitewater.reset();
  simT = t - CONFIG.sim.warmup;
  const total = Math.ceil(CONFIG.sim.warmup / DT);
  let done = 0;
  while (simT < t - 1e-9) {
    // Small batches keep the GPU and browser's presentation queue responsive.
    for (let n = 0; n < 12 && simT < t - 1e-9; n++) { stepSim(); done++; }
    if (report) loading('Letting the sea settle', 52 + 35 * Math.min(done / total, 1));
    await drainGPU(); await breathe();
  }
  window.__seeded = true;
}
let restarting = false;
async function restart(t) {
  if (restarting) return;
  restarting = true;
  overlay.classList.remove('hide');
  loading('Returning to the shore', 100);
  try { await warmSimulation(t, false); render(); await drainGPU(); }
  catch (error) { window.__loadError(error); return; }
  finally { restarting = false; last = performance.now(); }
  overlay.classList.add('hide');
}
async function start() {
  loading('Preparing light and reflections', 34);
  // Allocate particle attachments before their first visible use.
  whitewater._targets(W, H);
  await precompile();
  loading('Preparing the breaking waves', 48);
  // Draw the actual render chain, including both lip slots and every post pass.
  // compileAsync alone cannot warm texture uploads, FFT passes or framebuffer state.
  frameDt = 0;
  for (const t of [2.85, 3.15, 3.55]) { simT = t; render(); await drainGPU(); await breathe(); }
  if(surfaceProbe){
    surfaceProbe.request(renderer,camera,simT,controls.pos.x,-10,true);
    await surfaceProbe.ready;
  }
  await warmViews();
  await warmAnimatedFrames();
  mark('render-passes-warmed');
  await warmSimulation(START_T);
  loading('Finishing the light', 91);
  post.meter.reset();
  // Present repeated final-size frames while still covered. Keep simulation
  // time fixed and wait for both the GPU and the browser's paint cadence.
  // If a resize landed during startup, prepare its new attachments here too.
  let settled = 0;
  while (settled < 8) {
    await nextPaint();
    const revision = targetRevision, programs = renderer.info.programs.length;
    if (resizeTimer) { clearTimeout(resizeTimer); resizeTimer = null; resize(); }
    render(); await drainGPU();
    if (surfaceProbe?.pending) await surfaceProbe.ready;
    settled = !resizeTimer && revision === targetRevision && programs === renderer.info.programs.length ? settled + 1 : 0;
    loading('Finishing the light', 91 + settled);
  }
  clock.reset(); frameDt = 0; dyn.acc = dyn.n = 0;
  window.__graphicsInfo = checkShaderBudget(gl, renderer.info.programs);
  mark('ready');
  loading('Welcome to the shore', 100);
  window.__ready = true;
  chrome.forEach(el => { el.inert = false; });
  overlay.classList.add('hide');
  last = performance.now();
  if (!CAPTURE) requestAnimationFrame(frame);
}
start().catch(window.__loadError);

if (qs.has('gui')) {
  import('lil-gui').then(({ default: GUI }) => {
    const gui = new GUI({ title: 'Shorebreak' });
    const ctl = { get time() { return +simT.toFixed(2); }, speed, paused, restart: () => seek(START_T) };
    gui.add(ctl, 'speed', 0, 2, 0.05).onChange((v) => { speed = v; });
    gui.add(ctl, 'paused').onChange((v) => { paused = v; });
    gui.add(ctl, 'restart');
    const P = post.params;
    const pf = gui.addFolder('Look');
    pf.add(P, 'saturation', 0.8, 1.8, 0.01);
    pf.add(P, 'shoulder', 0.3, 0.95, 0.01);
    pf.add(P, 'sharpen', 0, 1, 0.01);
    pf.add(P, 'bloom', 0, 0.3, 0.005);
    const sf = gui.addFolder('Swash');
    sf.add(shared.uInjMass, 'value', 0, 4, 0.01).name('bore mass');
    sf.add(shared.uInjSpeed, 'value', 0, 2, 0.01).name('bore speed');
  });
}
