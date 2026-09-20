import * as THREE from 'three';
import { FullscreenPass, makeShader, floatRT, PingPong } from '../core/gpu.js';

// A small, fixed-cost, centre-weighted log-luminance meter. It never reads the
// framebuffer back to the CPU. Clipped sun/glints cannot dominate the exposure.
export const METER = /* glsl */ `
uniform sampler2D uScene;
in vec2 vUv;
void main() {
  vec2 e = vec2(0.25 / 16.0);
  float lum = 0.0;
  for (int i=0;i<4;i++) {
    vec2 q = vUv + (vec2(float(i&1),float(i>>1)) * 2.0 - 1.0)*e;
    lum += log2(clamp(dot(textureLod(uScene,q, 0.0).rgb,vec3(.2126,.7152,.0722)),.025,4.0));
  }
  vec2 p=(vUv-.5)*vec2(1.0,1.15);
  float w=.3+.7*exp(-dot(p,p)*6.0);
  gl_FragColor=vec4(lum*.25*w,w,0,1);
}`;
export const ADAPT = /* glsl */ `
uniform sampler2D uMeter,uPrevious;
uniform float uDt,uReset;
in vec2 vUv;
void main() {
  vec2 sum=vec2(0);
  for(int y=0;y<16;y++)for(int x=0;x<16;x++)sum+=texelFetch(uMeter,ivec2(x,y),0).rg;
  float target=clamp((log2(.38)-sum.x/max(sum.y,.001))*.38,-.50,.40);
  float old=texelFetch(uPrevious,ivec2(0),0).r;
  float tau=target<old?.65:1.6;
  float ev=mix(old,target,uReset>.5?1.0:1.0-exp(-uDt/tau));
  gl_FragColor=vec4(ev,exp2(ev),target,1);
}`;
export class Exposure {
  constructor() {
    this.meter=floatRT(16,16,{type:THREE.HalfFloatType});
    // Full precision costs only 32 bytes for both texels and avoids accumulated
    // half-float rounding making slow adaptation depend on refresh rate.
    this.history=new PingPong(1,1,{type:THREE.FloatType});
    this.pMeter=new FullscreenPass(makeShader(METER,{uScene:{value:null}}));
    this.pAdapt=new FullscreenPass(makeShader(ADAPT,{uMeter:{value:this.meter.texture},uPrevious:{value:null},uDt:{value:0},uReset:{value:1}}));
    this.initialized=false;
  }
  reset(){this.initialized=false;}
  render(renderer,scene,dt,settle=false) {
    this.pMeter.material.uniforms.uScene.value=scene;
    this.pMeter.render(renderer,this.meter);
    const u=this.pAdapt.material.uniforms;
    u.uPrevious.value=this.history.read.texture;
    u.uDt.value=Math.min(Math.max(dt,0),.1);
    u.uReset.value=settle||!this.initialized?1:0;
    this.pAdapt.render(renderer,this.history.write);this.history.swap();
    this.initialized=true;
    return this.history.read.texture;
  }
}
