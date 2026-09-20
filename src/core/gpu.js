import * as THREE from 'three';

// Single oversized triangle covering the viewport.
const triGeom = new THREE.BufferGeometry();
triGeom.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
triGeom.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 2, 0, 0, 2], 2));

export const FULLSCREEN_VERT = /* glsl */ `
out vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

export class FullscreenPass {
  static all = [];   // registry, so the app can precompile every pass in parallel at startup
  constructor(material) {
    FullscreenPass.all.push(this);
    this.material = material;
    this.mesh = new THREE.Mesh(triGeom, material);
    this.mesh.frustumCulled = false;
    this.scene = new THREE.Scene();
    this.scene.add(this.mesh);
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }
  render(renderer, target, layer = 0) {
    renderer.setRenderTarget(target, layer);
    renderer.render(this.scene, this.camera);
  }
}

export function makeShader(fragmentShader, uniforms = {}, extra = {}) {
  return new THREE.ShaderMaterial({
    
    vertexShader: FULLSCREEN_VERT,
    fragmentShader,
    uniforms,
    depthTest: false,
    depthWrite: false,
    ...extra,
  });
}

export function floatRT(w, h, { type = THREE.FloatType, filter = THREE.NearestFilter, format = THREE.RGBAFormat, depth = false, wrap = THREE.ClampToEdgeWrapping } = {}) {
  const rt = new THREE.WebGLRenderTarget(w, h, {
    type, format, minFilter: filter, magFilter: filter, depthBuffer: depth, stencilBuffer: false,
    wrapS: wrap, wrapT: wrap, generateMipmaps: false,
  });
  rt.texture.colorSpace = THREE.NoColorSpace;
  return rt;
}

export class PingPong {
  constructor(w, h, opts) { this.a = floatRT(w, h, opts); this.b = floatRT(w, h, opts); }
  get read() { return this.a; }
  get write() { return this.b; }
  swap() { const t = this.a; this.a = this.b; this.b = t; }
  dispose() { this.a.dispose(); this.b.dispose(); }
}
