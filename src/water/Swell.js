import * as THREE from 'three';
import { glslDefines } from '../config.js';
import { NOISE } from '../glsl/common.js';
import { BREAKER } from '../glsl/breaker.js';
import { waterLookups } from './WaterLookups.js';

const CAPACITY = 32, SEGMENTS = 256;
const VERT = glslDefines() + NOISE + BREAKER.replace('#define MAX_EVENTS 6', '#define MAX_EVENTS 32') + /* glsl */ `
in float aEvent;
out vec4 vShape0, vShape1;
void main() {
  Brk b=brkAt(int(aEvent+.5),position.x);
  float tn=brkTn(b,uTime); BrkStage g; vec2 A;
  brkTransport(b,min(tn,-.8),g,A);
  float alive=(1.0-smoothstep(-1.0,-.8,tn))*b.str;
  float z=mix(A.x-7.0*g.lb*b.s,A.x+(g.lf+6.0)*b.s,position.y);
  float v=log(1.0+max(-z-6.0,0.0)/8.0)/log(81.5);
  gl_Position=vec4(position.x/512.0,2.0*v-1.0,0,1);
  vShape0=vec4(A.x,b.s,g.ya,g.yt);
  vShape1=vec4(g.lf,g.lb,alive,0);
}`;
const FRAG = /* glsl */ `
in vec4 vShape0,vShape1;
void main() {
  float x=(gl_FragCoord.x/256.0)*1024.0-512.0;
  float z=-6.0-8.0*(exp((gl_FragCoord.y/512.0)*log(81.5))-1.0);
  float xi=(z-vShape0.x)/vShape0.y;
  float ya=vShape0.z,yt=vShape0.w,lf=vShape1.x,lb=vShape1.y;
  float h,dz;
  if(xi<0.0){float a=xi/lb;float e=exp(-abs(a));float sec=2.0*e/(1.0+e*e);h=ya*sec*sec;dz=h*(-2.0/lb)*tanh(a);}
  else if(xi>lf){float a=(xi-lf)/1.5;h=yt*exp(-a*a);dz=h*(-2.0*a/1.5);}
  else {float a=3.14159265*xi/lf;h=yt+(ya-yt)*(.5+.5*cos(a));dz=-(ya-yt)*1.5707963*sin(a)/lf;}
  h*=vShape0.y*vShape1.z;dz*=vShape1.z;
  gl_FragColor=vec4(h,dFdx(h)/4.0,dz,0);
}`;

export class Swell {
  constructor(shared,schedule) {
    this.schedule=schedule; this.time=NaN;
    const travel=waterLookups();
    this.target=new THREE.WebGLRenderTarget(256,512,{type:THREE.HalfFloatType,minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,depthBuffer:false,stencilBuffer:false});
    Object.assign(shared,{uWaterLookup:{value:travel.texture},uSwellMinTime:{value:travel.minTime},uSwellField:{value:this.target.texture}});
    this.uniforms={...shared,uTime:{value:0},uEvtCount:{value:0}};
    for(const name of 'ABCDEFG')this.uniforms['uEvt'+name]={value:new Float32Array(CAPACITY*4)};
    const g=new THREE.InstancedBufferGeometry(),pos=[],index=[];
    for(let i=0;i<=SEGMENTS;i++){const x=-512+i*1024/SEGMENTS;pos.push(x,0,0,x,1,0);if(i<SEGMENTS){const a=2*i;index.push(a,a+1,a+2,a+1,a+3,a+2);}}
    g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));g.setIndex(index);
    g.setAttribute('aEvent',new THREE.InstancedBufferAttribute(Float32Array.from({length:CAPACITY},(_,i)=>i),1));g.instanceCount=0;
    this.geometry=g;
    this.material=new THREE.ShaderMaterial({uniforms:this.uniforms,vertexShader:VERT,fragmentShader:FRAG,side:THREE.DoubleSide,depthTest:false,depthWrite:false,transparent:true,blending:THREE.CustomBlending,blendSrc:THREE.OneFactor,blendDst:THREE.OneFactor,blendEquation:THREE.AddEquation});
    this.mesh=new THREE.Mesh(g,this.material);this.mesh.frustumCulled=false;
    this.scene=new THREE.Scene();this.scene.add(this.mesh);this.camera=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
    schedule.extendTo(110); this.pack(0);
  }
  pack(t) {
    this.schedule.extendTo(t+110);
    const events=this.schedule.events.filter(e=>e.t0>t-6&&e.t0<t+114).slice(0,CAPACITY);
    this.geometry.instanceCount=events.length;this.uniforms.uEvtCount.value=events.length;
    for(let i=0;i<events.length;i++){
      const e=events[i],o=i*4,u=this.uniforms;
      u.uEvtA.value.set([e.t0,e.H,e.zI,e.seed],o);u.uEvtB.value.set([e.c,e.x0,e.wobble,e.strength],o);
      u.uEvtC.value.set([e.kappa,e.waviness,e.hvar,e.style],o);u.uEvtD.value.set([e.aR1,e.aR2,e.aL1,e.aL2],o);
      u.uEvtE.value.set([e.hPeel||0,e.stand||0,e.spill,e.tilt],o);u.uEvtF.value.set(e.F,o);u.uEvtG.value.set([e.bore,e.splash,e.far,e.farPhase],o);
    }
  }
  update(renderer,t) {
    if(t===this.time)return;this.time=t;this.pack(t);this.uniforms.uTime.value=t;
    const target=renderer.getRenderTarget(),color=renderer.getClearColor(new THREE.Color()),alpha=renderer.getClearAlpha();
    renderer.setRenderTarget(this.target);renderer.setClearColor(0,0);renderer.clear();renderer.render(this.scene,this.camera);
    renderer.setClearColor(color,alpha);renderer.setRenderTarget(target);
  }
}
