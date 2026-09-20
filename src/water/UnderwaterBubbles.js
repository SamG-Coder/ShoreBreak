import * as THREE from 'three';
import { glslDefines } from '../config.js';
import { NOISE,BED,SKY } from '../glsl/common.js';
import { BREAKER } from '../glsl/breaker.js';
import { WATERLINE } from '../glsl/underwater.js';
import { ENTRAINED_AIR } from '../glsl/entrainedAir.js';

const DEFAULT_F=[0,0,1,0];
function put4(a,o,x,y,z,w){a[o]=x;a[o+1]=y;a[o+2]=z;a[o+3]=w;}

// Fixed world-anchored cells, one bounded draw. Event slots may change, but each
// bubble's seed is keyed to its event and cell, so the surviving cloud never jumps.
export class UnderwaterBubbles {
  constructor(shared,schedule){
    this.schedule=schedule;
    const uniforms={...shared,uEvtCount:{value:0}};
    for(const key of 'ABCDEFG')uniforms['uEvt'+key]={value:new Float32Array(24)};
    const base=new THREE.PlaneGeometry(2,2),g=new THREE.InstancedBufferGeometry();
    g.index=base.index;g.attributes=base.attributes;
    const count=6*64*24,seeds=new Float32Array(count*3);
    for(let i=0;i<count;i++)seeds.set([Math.floor(i/(64*24)),Math.floor(i/24)%64,i%24],i*3);
    g.setAttribute('aBubble',new THREE.InstancedBufferAttribute(seeds,3));g.instanceCount=count;
    const pre=glslDefines()+NOISE+BED+SKY+BREAKER+ENTRAINED_AIR;
    this.material=new THREE.ShaderMaterial({uniforms,transparent:true,depthWrite:false,
      vertexShader:pre+/* glsl */`
      uniform vec2 uFocus,uResolution;in vec3 aBubble;out vec2 vUv;out vec3 vP;out float vLife,vSeed;
      float rand(float n){return hash11(n);}
      void main(){
        int slot=int(aBubble.x);float cell=mod(aBubble.y-floor(uFocus.x/.5),64.)+floor(uFocus.x/.5)-32.;
        float seed=cell*3.71+aBubble.z*7.19+uEvtA[slot].w*173.;
        float r=rand(seed),r1=rand(seed+13.),r2=rand(seed+41.);
        float x=cell*.5+r*.5;Brk b=brkAt(slot,x);
        float age=uTime-b.ti-r1*.18,life=2.1+r2*2.3;
        float a=max(age,0.);
        float radius=mix(.0012,.012,r*r*r)*clamp(b.rs,.6,1.5);
        AirPocket pocket=airPocket(x,age,b.zI,b.rs,b.seed);
        float theta=seed+a*(2.5+r)*exp(-a*.3);
        float spread=(1.-exp(-a*3.))*exp(-a*.20);
        float z=pocket.z+(r2-.5)*pocket.rz*2.+cos(theta)*.07*spread;
        float y=brkSurfaceOnly(vec2(x,pocket.z),uTime)-pocket.depth+radius*7.*a;
        y+=sin(theta)*pocket.ry*.75*spread;
        x+=sin(a*1.8+seed)*.07*spread;
        float surface=brkSurfaceOnly(vec2(x,z),uTime);
        vLife=smoothstep(0.,.10,age)*(1.-smoothstep(life*.65,life,age))*(1.-smoothstep(surface-.045,surface+.015,y));
        vLife*=smoothstep(.005,.10,y-hydraulicBedHeight(vec2(x,z)))*step(float(slot)+.5,float(uEvtCount))*b.str;
        vLife*=.25+.75*pocket.finger;
        vLife*=1.-smoothstep(12.,16.,abs(x-uFocus.x));
        vec3 P=vec3(x,y,z);vP=P;vUv=uv;vSeed=seed;
        vec4 mv=viewMatrix*vec4(P,1.);
        float pixelRadius=radius*projectionMatrix[1][1]*uResolution.y/max(-mv.z,.05)*.5;
        vLife*=smoothstep(.15,.8,pixelRadius)*smoothstep(.035,.10,-mv.z);
        mv.xy+=position.xy*radius*vec2(1.+.14*sin(seed+a*8.),1.);
        gl_Position=projectionMatrix*mv;
        if(vLife<.001)gl_Position=vec4(2.,2.,2.,1.);
      }`,
      fragmentShader:WATERLINE+/* glsl */`
      in vec2 vUv;in vec3 vP;in float vLife,vSeed;
      void main(){
        vec2 q=vUv*2.-1.;float r2=dot(q,q),r=sqrt(r2),aa=max(fwidth(r),.03);
        float edge=1.-smoothstep(1.-aa,1.+aa,r);
        if(edge<.001)discard;
        float nz=sqrt(max(1.-r2,0.));
        float fresnel=.0204+.9796*pow(1.-nz,5.);
        float crescent=pow(max(dot(normalize(vec3(q,nz)),normalize(vec3(-.4,.75,.5))),0.),22.);
        float glint=exp(-dot(q-vec2(-.27,.48),q-vec2(-.27,.48))*65.);
        float wet=submergedAt(cameraPosition+normalize(vP-cameraPosition)*.04);
        float alpha=(.035+.35*fresnel+.38*crescent+.36*glint)*edge*vLife*wet;
        vec3 col=mix(vec3(.10,.24,.25),vec3(.79,.93,.91),clamp(crescent+glint+fresnel*.45,0.,1.));
        gl_FragColor=vec4(col,alpha);
      }`});
    this.mesh=new THREE.Mesh(g,this.material);this.mesh.frustumCulled=false;this.mesh.renderOrder=40;this.mesh.name='Crash entrained bubbles';
    this.scene=new THREE.Scene();this.scene.add(this.mesh);
  }
  update(t){
    const u=this.material.uniforms;let i=0;
    // Keep an impact for its complete bubble lifetime, independently of the
    // shorter optical event window used by the surface mesh.
    for(const e of this.schedule.events){
      if(t<e.t0-2.5||t>e._window[1]+5.8||i===6)continue;
      const o=i*4;
      put4(u.uEvtA.value,o,e.t0,e.H,e.zI,e.seed);
      put4(u.uEvtB.value,o,e.c,e.x0,e.wobble,e.strength);
      put4(u.uEvtC.value,o,e.kappa,e.waviness,e.hvar,e.style);
      put4(u.uEvtD.value,o,e.aR1,e.aR2,e.aL1,e.aL2);
      put4(u.uEvtE.value,o,e.hPeel||0,e.stand||0,e.spill,e.tilt);
      u.uEvtF.value.set(e.F||DEFAULT_F,o);
      put4(u.uEvtG.value,o,e.bore??1,e.splash??1,e.far||0,e.farPhase||0);i++;
    }
    u.uEvtCount.value=i;
  }
}
