import * as THREE from 'three';
import { FullscreenPass, makeShader, floatRT } from '../core/gpu.js';
import { CONFIG, glslDefines } from '../config.js';
import { NOISE, BED, SKY } from '../glsl/common.js';
import { BREAKER } from '../glsl/breaker.js';
import { WATERLINE } from '../glsl/underwater.js';
import { ENTRAINED_AIR } from '../glsl/entrainedAir.js';
import { WATER_PRELUDE } from './WaterSurface.js';

// Precompute impact timing and air motion, rather than evaluating noise,
// bathymetry and exponentials for every event at every volume sample.
const FIELD=glslDefines()+NOISE+BED+BREAKER+ENTRAINED_AIR+/* glsl */`
uniform float uFieldLeft;
in vec2 vUv;
void main(){
  int row=int(gl_FragCoord.y),slot=row/3;
  if(slot>=uEvtCount){gl_FragColor=vec4(0.);return;}
  float x=uFieldLeft+vUv.x*48.;Brk b=brkAt(slot,x);
  float age=uTime-b.ti;
  AirPocket p=airPocket(x,age,b.zI,b.rs,b.seed);
  if(row%3==0)gl_FragColor=vec4(age,b.zI,b.rs,b.str);
  else if(row%3==1)gl_FragColor=vec4(p.depth,p.z,p.ry,p.rz);
  else gl_FragColor=vec4(p.life,p.finger,b.seed,0.);
}`;

// The cloud follows the actual FFT + swell + shallow-water surface, including
// horizontal chop inversion. Bake the already-advected swash state into the
// same atlas: surface height, entrained air, suspended fines, moving patchwork.
const SURFACE=WATER_PRELUDE()+/* glsl */`
uniform float uFieldLeft;
uniform vec2 uCrashZ;
in vec2 vUv;
void main(){
  vec2 world=vec2(uFieldLeft+vUv.x*48.,mix(uCrashZ.x,uCrashZ.y,vUv.y));
  float coverage;
  vec4 flow=sweBlendView(world,uTime,coverage);
  float bed=hydraulicBedHeight(world);
  // The upper reserve is usually dry. Do not evaluate two full ocean/crest
  // profiles for texels that cannot contain an underwater plume.
  if(flow.x<.001&&bed>.02){gl_FragColor=vec4(bed,0.,0.,.5);return;}
  vec2 xz=world,gm,gc;vec3 ch;vec4 info;float paw,y;
  y=waterParts(xz,uTime,.06,gm,ch,gc,info,paw);
  xz=world-ch.xz;
  y=waterParts(xz,uTime,.06,gm,ch,gc,info,paw);
  vec4 foam=sweBlendFoam(world,uTime);
  float h=max(min(flow.x,y+ch.y-bed),0.);
  // Thin water vents its air quickly. Bed shear can stir sand without creating
  // a fresh white bubble cloud, especially in the returning flow.
  float wet=smoothstep(.012,.09,h)*coverage;
  float air=(.72*smoothstep(.035,.9,foam.r)+.28*foam.a)
    *(.3+.7*smoothstep(.12,1.8,foam.b))*wet;
  float fines=(.22*foam.a+.16*smoothstep(.35,1.4,length(flow.yz))
    *smoothstep(.25,1.6,foam.b))*smoothstep(.015,.075,h)*coverage;
  vec4 offset=sweBlendLace(world,uTime);
  float wa=smoothstep(0.,.22,uLacePhase.x)*(1.-smoothstep(.5,.72,uLacePhase.x));
  vec2 scale=vec2(1.8,2.4);
  float patchWeight=.5+.5*mix(gnoise((world+offset.zw)*scale+7.1),gnoise((world+offset.xy)*scale+7.1),wa);
  gl_FragColor=vec4(y+ch.y,air,fines,clamp(patchWeight,0.,1.));
}`;

const VOLUME=glslDefines()+NOISE+BED+SKY+WATERLINE+/* glsl */`
uniform sampler2D uPlumeField,uPlumeSurface,uSceneDepth;
precision highp sampler3D;
uniform sampler3D uCloudNoise;
uniform mat4 uInverseVP;
uniform vec3 uEye;
uniform vec2 uCrashZ;
uniform float uFieldLeft,uTime,uShoreChurn;
uniform int uPlumeCount;
in vec2 vUv;
float cloudNoise(vec3 p){return textureLod(uCloudNoise,(p+.5)/64.,0.).r;}
vec4 coastAt(vec2 p){return textureLod(uPlumeSurface,vec2((p.x-uFieldLeft)/48.,(p.y-uCrashZ.x)/(uCrashZ.y-uCrashZ.x)),0.);}
float surfaceAt(vec2 p){return coastAt(p).r;}

// Return extinction, depth below the illuminated cloud top and sand tint. The cloud
// plunges first, rolls shoreward, then loses air as bubbles rise. Along-crest
// fingers and the rotating cross-section break up a uniform horizontal tube.
vec3 cloud(vec3 p,float footprint){
  float ux=(p.x-uFieldLeft)/48.;
  if(ux<0.||ux>1.)return vec3(0.);
  vec4 coast=coastAt(p.xz);
  float ceiling=coast.r;
  float bed=hydraulicBedHeight(p.xz);
  float depth=max(ceiling-bed,0.);
  float waterMask=(1.-smoothstep(ceiling-.035,ceiling+.015,p.y))
    *smoothstep(.003,min(.085,max(.008,depth*.25)),p.y-bed)
    *smoothstep(.008,.04,depth);
  if(waterMask<.001)return vec3(0.);
  float density=0.,shade=0.,sand=0.;
  for(int i=0;i<6;i++){
    if(i>=uPlumeCount)break;
    float row=float(i)*3.;
    vec4 f=textureLod(uPlumeField,vec2(ux,(row+.5)/18.),0.);
    float a=f.x,rs=f.z;
    if(a<=0.||a>=4.8||f.w<.001)continue;
    vec4 motion=textureLod(uPlumeField,vec2(ux,(row+1.5)/18.),0.);
    vec4 shape=textureLod(uPlumeField,vec2(ux,(row+2.5)/18.),0.);
    float life=shape.x*f.w,finger=shape.y,seed=shape.z;
    float cz=motion.y,cy=surfaceAt(vec2(p.x,cz))-motion.x;
    vec2 radii=motion.zw;
    vec2 q=vec2(p.y-cy,p.z-cz);
    float theta=.40*sin(p.x*1.7+a*1.3+seed*17.);
    q=mat2(cos(theta),-sin(theta),sin(theta),cos(theta))*q;
    float r=length(q/radii);
    if(r>1.8)continue;
    vec3 npos=vec3(p.x*3.8+seed*41.,(p.y-cy)*6.,(p.z-cz)*6.);
    float spin=(1.-exp(-a*.75))*2.8;
    npos.yz=mat2(cos(spin),-sin(spin),sin(spin),cos(spin))*npos.yz;
    float detail=1.-smoothstep(.055,.17,footprint);
    float n=cloudNoise(npos)+.36*mix(.5,cloudNoise(npos*2.31+7.3),detail);
    float boundary=r+(.69-n)*1.75;
    float d=(1.-smoothstep(.28,1.08,boundary))*life*(.22+.78*finger);
    // Voids within the roller and soft late remnants avoid a row of opaque
    // identical puffs. Fine structure is filtered to the integration footprint.
    float fine=mix(.5,cloudNoise(npos*4.7-a*.16),1.-smoothstep(.025,.085,footprint));
    d*=.22+1.45*fine*fine;
    float top=clamp((cy+radii.x-p.y)/(radii.x*2.),0.,1.);
    float lightNoise=cloudNoise(npos+vec3(uSunDir.x,uSunDir.y,uSunDir.z)*1.4);
    float relief=clamp((lightNoise-n/1.36)*2.4,-.5,.5);
    density+=d;shade+=d*clamp(top+relief,0.,1.);
    // Cheap suspended-sand colour proxy: strongest in fresh near-bed churn,
    // fading as the buoyant air rises and the disturbed sand settles out.
    sand+=d*smoothstep(.10,.45,a)*(1.-smoothstep(1.5,4.2,a))
      *exp(-max(p.y-bed,0.)*2.4)*smoothstep(.65,1.35,rs);
  }
  // The bore's existing foam/turbulence fields travel with the solved flow.
  // Their 3D continuation joins the plunge to the uprush, compressing into the
  // available water column and fading smoothly as the film drains. This is
  // unresolved microbubble/sediment scattering, not another particle system.
  if(uShoreChurn>0.&&max(coast.g,coast.b)>.0001){
  float aboveBed=max(p.y-bed,0.),belowSurface=max(ceiling-p.y,0.);
  float airReach=min(depth*.8,.12+.45*coast.g);
  float airProfile=exp(-belowSurface/max(airReach,.025));
  float sandProfile=exp(-aboveBed/max(.025,min(depth*.42,.19)));
  float detail=1.-smoothstep(.05,.20,footprint);
  float n=mix(.5,cloudNoise(vec3(p.x*3.1,p.y*7.-uTime*.18,p.z*3.1)+coast.a*2.2),detail);
  float patchWeight=smoothstep(.20,.72,coast.a*.72+n*.28);
  float air=uShoreChurn*coast.g*airProfile*(.025+.19*patchWeight);
  float sediment=uShoreChurn*coast.b*sandProfile*(.06+.20*patchWeight);
  // Preserve the turbulent impact core: the continuation fills its thinning
  // fringes rather than doubling extinction in already-dense impact pockets.
  float tail=(air+sediment)*exp(-density*2.5);
  shade+=tail*clamp(belowSurface/max(depth,.025),0.,1.);
  sand+=tail*clamp((sediment+.18*air)/max(air+sediment,.00001),0.,1.);
  density+=tail;
  }
  return vec3(density*24.*waterMask,shade/max(density,.00001),sand/max(density,.00001));
}
void main(){
  gl_FragColor=vec4(0.);
  vec4 np=uInverseVP*vec4(vUv*2.-1.,-1.,1.);
  vec3 ray=normalize(np.xyz/np.w-uEye);
  float wet=submergedAt(np.xyz/np.w);
  if(wet<.002||uPlumeCount==0)return;
  vec4 endp=uInverseVP*vec4(vUv*2.-1.,textureLod(uSceneDepth,vUv, 0.0).r*2.-1.,1.);
  float endD=min(length(endp.xyz/endp.w-uEye),18.);
  vec3 inv=sign(ray+vec3(1e-8))/max(abs(ray),vec3(1e-5));
  vec3 t0=(vec3(uFieldLeft,-1.7,uCrashZ.x)-uEye)*inv;
  vec3 t1=(vec3(uFieldLeft+48.,1.8,uCrashZ.y)-uEye)*inv;
  vec3 lo=min(t0,t1),hi=max(t0,t1);
  float enter=max(max(max(lo.x,lo.y),lo.z),0.);
  float leave=min(min(min(hi.x,hi.y),hi.z),endD);
  if(leave<=enter)return;
  float stepM=(leave-enter)/36.;
  float tr=1.;vec3 color=vec3(0.);
  // Stable stratification breaks up visible integration planes. Its pattern
  // never changes with time; no temporal accumulation or ghost trails.
  float offset=.04+.92*hash12(gl_FragCoord.xy);
  for(int j=0;j<36;j++){
    float dist=enter+(float(j)+offset)*stepM;
    vec3 p=uEye+ray*dist;
    vec3 d=cloud(p,stepM);d.x*=1.-smoothstep(14.,18.,dist);
    if(d.x<.005)continue;
    float a=1.-exp(-d.x*stepM);
    float direct=exp(-d.y*2.7);
    float forward=.5+.5*pow(max(dot(ray,uSunDir),0.),6.);
    vec3 light=uSkyAmb*(.16+.46*direct)+uSunColor*(.012+.38*direct)*forward;
    light*=mix(vec3(.50,.73,.76),vec3(.95,.98,.97),direct);
    light*=mix(vec3(1.),vec3(1.06,.90,.67),clamp(d.z*.7,0.,.7));
    color+=tr*a*waterTravel(light,dist);
    tr*=1.-a;
    if(tr<.025)break;
  }
  gl_FragColor=vec4(color*wet,(1.-tr)*wet);
}`;

export class UnderwaterPlume {
  constructor(shared,eventUniforms){
    this.field=floatRT(192,18,{filter:THREE.LinearFilter});
    this.surface=floatRT(192,128,{filter:THREE.LinearFilter,type:THREE.HalfFloatType});
    // 256 KiB, generated once before warmup. Hardware trilinear filtering
    // replaces dozens of procedural hashes at every ray sample.
    const data=new Uint8Array(64*64*64);
    for(let i=0;i<data.length;i++){
      let h=Math.imul(i^0x6d2b79f5,0x45d9f3b);h=Math.imul(h^(h>>>16),0x45d9f3b);
      data[i]=(h^(h>>>16))&255;
    }
    this.noise=new THREE.Data3DTexture(data,64,64,64);
    Object.assign(this.noise,{format:THREE.RedFormat,type:THREE.UnsignedByteType,
      minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,
      wrapS:THREE.RepeatWrapping,wrapT:THREE.RepeatWrapping,wrapR:THREE.RepeatWrapping,unpackAlignment:1});
    this.noise.needsUpdate=true;
    this.uniforms={...shared,...eventUniforms,uFieldLeft:{value:0},uPlumeField:{value:this.field.texture},
      uPlumeSurface:{value:this.surface.texture},uCloudNoise:{value:this.noise},uCrashZ:{value:new THREE.Vector2(-7,CONFIG.swe.zMax)},uShoreChurn:{value:1},uPlumeCount:eventUniforms.uEvtCount,uSceneDepth:{value:null},uInverseVP:{value:new THREE.Matrix4()},uEye:{value:new THREE.Vector3()}};
    this.pField=new FullscreenPass(makeShader(FIELD,this.uniforms));
    // Retained air events must never replace the active surface event uniforms.
    this.pSurface=new FullscreenPass(makeShader(SURFACE,{...shared,uFieldLeft:this.uniforms.uFieldLeft,uCrashZ:this.uniforms.uCrashZ}));
    this.pVolume=new FullscreenPass(makeShader(VOLUME,this.uniforms));
    this.resize(1,1);
  }
  resize(w,h){
    // Half resolution, capped at 960 x 540, <= 4 MiB of extra HDR storage.
    const scale=Math.min(.5,960/w,540/h);
    const x=Math.max(1,Math.ceil(w*scale)),y=Math.max(1,Math.ceil(h*scale));
    if(this.target?.width===x&&this.target?.height===y)return;
    this.target?.dispose();
    this.target=floatRT(x,y,{type:THREE.HalfFloatType,filter:THREE.LinearFilter});
  }
  render(renderer,camera,depth){
    const u=this.uniforms;
    if(!u.uPlumeCount.value)return null;
    let z0=CONFIG.swe.zMin,z1=CONFIG.swe.zMax;
    for(let i=0;i<u.uPlumeCount.value;i++){
      const z=u.uEvtA.value[i*4+2];z0=Math.min(z0,z-1.05);z1=Math.max(z1,z+2.1);
    }
    u.uCrashZ.value.set(z0,z1);
    // Quantised coverage with a 24 m margin; density is keyed to world position.
    u.uFieldLeft.value=Math.floor(camera.position.x/4)*4-24;
    u.uEye.value.copy(camera.position);
    u.uInverseVP.value.multiplyMatrices(camera.matrixWorld,camera.projectionMatrixInverse);
    u.uSceneDepth.value=depth;
    this.pField.render(renderer,this.field);
    this.pSurface.camera.position.copy(camera.position);
    this.pSurface.render(renderer,this.surface);
    this.pVolume.render(renderer,this.target);
    return this.target.texture;
  }
}
