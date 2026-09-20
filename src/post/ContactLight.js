import * as THREE from 'three';
import { FullscreenPass, makeShader, floatRT } from '../core/gpu.js';

// Deterministic horizon-style contact occlusion, at most 960x540. No rotating
// screen-space noise or temporal history. Reconstruction rejects depth edges.
export const CONTACT = /* glsl */ `
uniform sampler2D uDepth;
uniform mat4 uInvProjection;
uniform vec2 uTexel,uFocal;
in vec2 vUv;
vec3 pointAt(vec2 uv) {
  // The half-resolution pass lands between full-resolution depth texels.
  // Reconstruct at the texel we actually read; mixing a nearest depth with
  // an unsnapped ray creates false horizontal/vertical ridges on smooth sand.
  ivec2 size=textureSize(uDepth,0);
  ivec2 pixel=clamp(ivec2(floor(uv*vec2(size))),ivec2(0),size-1);
  vec2 centre=(vec2(pixel)+.5)/vec2(size);
  vec4 p=uInvProjection*vec4(centre*2.0-1.0,texelFetch(uDepth,pixel,0).r*2.0-1.0,1.0);
  return p.xyz/p.w;
}
void main() {
  vec3 P=pointAt(vUv);
  if(-P.z>220.0){gl_FragColor=vec4(1,0,0,1);return;}
  vec3 px=pointAt(vUv+vec2(uTexel.x,0))-P, mx=P-pointAt(vUv-vec2(uTexel.x,0));
  vec3 py=pointAt(vUv+vec2(0,uTexel.y))-P, my=P-pointAt(vUv-vec2(0,uTexel.y));
  vec3 N=normalize(cross(abs(px.z)<abs(mx.z)?px:mx,abs(py.z)<abs(my.z)?py:my));
  if(dot(N,-P)<0.0)N=-N;
  float radius=mix(.12,1.6,smoothstep(4.0,75.0,-P.z));
  vec2 ruv=min(vec2(.08),uFocal*radius/max(-P.z,.1));
  float occ=0.0;
  for(int i=0;i<12;i++) {
    float a=float(i%6)*1.04719755+.31;
    float ring=i<6?.38:1.0;
    vec2 uv=vUv+vec2(cos(a),sin(a))*ruv*ring;
    if(min(uv.x,uv.y)<0.0||max(uv.x,uv.y)>1.0)continue;
    vec3 D=pointAt(uv)-P;float d=length(D);
    float h=max(dot(N,D)/max(d,.001)-.12,0.0);
    occ+=h*(1.0-smoothstep(radius*.35,radius*1.8,d));
  }
  float ao=clamp(1.0-occ*.13,.70,1.0);
  gl_FragColor=vec4(ao,-P.z,0,1);
}`;

// Depth-aware upsample prevents dark halos around thin leaves and silhouettes.
export const CONTACT_SAMPLE = /* glsl */ `
uniform sampler2D uContact;
uniform vec2 uContactTexel;
uniform vec2 uCameraRange;
float contactAt(vec2 uv,float depth) {
  float n=uCameraRange.x,f=uCameraRange.y;
  float z=n*f/(f-depth*(f-n));
  float sum=0.0,wSum=0.0;
  for(int i=0;i<4;i++) {
    vec2 o=(vec2(float(i&1),float(i>>1))-.5)*uContactTexel;
    vec2 a=textureLod(uContact,uv+o, 0.0).rg;
    float w=exp(-abs(a.y-z)/max(.06,z*.015));
    sum+=a.x*w;wSum+=w;
  }
  return wSum>.001?sum/wSum:1.0;
}`;
export class ContactLight {
  constructor(){this.pass=new FullscreenPass(makeShader(CONTACT,{uDepth:{value:null},uInvProjection:{value:new THREE.Matrix4()},uTexel:{value:new THREE.Vector2()},uFocal:{value:new THREE.Vector2()}}));}
  resize(w,h){
    const k=Math.min(.5,960/w,540/h);
    this.target?.dispose();this.target=floatRT(Math.max(1,Math.round(w*k)),Math.max(1,Math.round(h*k)),{type:THREE.HalfFloatType,filter:THREE.LinearFilter});
  }
  render(renderer,depth,camera,w,h){
    const u=this.pass.material.uniforms;u.uDepth.value=depth;u.uInvProjection.value.copy(camera.projectionMatrixInverse);
    u.uTexel.value.set(1/w,1/h);u.uFocal.value.set(camera.projectionMatrix.elements[0]*.5,camera.projectionMatrix.elements[5]*.5);
    this.pass.render(renderer,this.target);
  }
}
