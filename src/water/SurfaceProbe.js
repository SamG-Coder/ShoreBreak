import * as THREE from 'three';
import { FullscreenPass, makeShader, floatRT } from '../core/gpu.js';
import { WATER_PRELUDE } from './WaterSurface.js';

const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
// Five water samples, using the same breaker, SWE and FFT displacement as the
// scene. Async GPU readback never blocks the render loop; only one is in flight.
export class SurfaceProbe {
  constructor(shared) {
    this.target=floatRT(5,1);
    if(shared.uSurfaceProbe)shared.uSurfaceProbe.value=this.target.texture;
    this.shared=shared;
    this.data=new Float32Array(20);
    this.uniforms={...shared,uProbe:{value:new THREE.Vector2()}};
    this.pass=new FullscreenPass(makeShader(WATER_PRELUDE()+/* glsl */`
      uniform vec2 uProbe;
      void main(){
        int i=int(gl_FragCoord.x);
        vec2 world=uProbe+vec2(i==1?-.32:(i==2?.32:0.),i==3?-.32:(i==4?.32:0.));
        vec2 xz=world,gm,gc;vec3 ch;vec4 info;float paw,y;
        // Invert horizontal chop once so the query is at the swimmer's actual
        // world position, rather than at an undisplaced mesh coordinate.
        y=waterParts(xz,uTime,.025,gm,ch,gc,info,paw);
        xz=world-ch.xz;
        y=waterParts(xz,uTime,.025,gm,ch,gc,info,paw);
        vec4 flow=textureLod(uSweView,sweUV(world), 0.0);
        gl_FragColor=vec4(y+ch.y,flow.yz*sweInside(world),1.);
      }`,this.uniforms));
    this.pending=false;this.latest=null;this.lastRequest=-Infinity;this.failures=0;this.generation=0;
    this.result={height:0,slopeX:0,slopeZ:0,flowX:0,flowZ:0};
  }

  reset(){
    this.generation++;this.latest=null;this.lastRequest=-Infinity;
    Object.assign(this.result,{height:0,slopeX:0,slopeZ:0,flowX:0,flowZ:0});
  }

  request(renderer,camera,t,x,z,force=false) {
    const now=performance.now();
    if(!force&&z>1.5)return;
    const generation=this.generation;
    const previous=renderer.getRenderTarget();
    this.uniforms.uProbe.value.set(x,z);
    this.pass.camera.position.copy(camera.position);
    try {
      this.pass.render(renderer,this.target);
      this.shared.uProbeOrigin?.value.set(x,z);
      renderer.setRenderTarget(previous);
      if(this.pending||(!force&&now-this.lastRequest<32)||this.failures>=3)return;
      this.lastRequest=now;this.pending=true;
      this.ready=renderer.readRenderTargetPixelsAsync(this.target,0,0,5,1,this.data).then(()=>{
        if(generation!==this.generation)return;
        const a=this.data;
        if(!a.every(Number.isFinite)||a[3]<.5)throw Error('Invalid water surface sample');
        const slopeX=clamp((a[8]-a[4])/.64,-1.2,1.2),slopeZ=clamp((a[16]-a[12])/.64,-1.2,1.2);
        const old=this.latest,dt=old?t-old.t:0;
        let rate=0;
        if(dt>.001&&dt<.3)rate=clamp((a[0]-old.height-slopeX*(x-old.x)-slopeZ*(z-old.z))/dt,-2.5,2.5);
        this.latest={height:a[0],slopeX,slopeZ,flowX:a[1],flowZ:a[2],x,z,t,rate:old?old.rate*.5+rate*.5:0};
        this.failures=0;
      }).catch(error=>{
        this.failures++;
        if(this.failures===3)console.warn('Wave buoyancy readback unavailable',error);
      }).finally(()=>{this.pending=false;});
    } catch(error) {
      renderer.setRenderTarget(previous);this.pending=false;this.failures++;
      if(this.failures===3)console.warn('Wave buoyancy readback unavailable',error);
    }
  }

  sample(x,z,t) {
    const p=this.latest,out=this.result;
    if(!p)return out;
    // Short prediction removes readback latency; never extrapolate a stalled
    // sample indefinitely or turn camera movement into an apparent wave.
    out.height=p.height+p.slopeX*clamp(x-p.x,-.45,.45)+p.slopeZ*clamp(z-p.z,-.45,.45)+p.rate*clamp(t-p.t,0,.10);
    out.slopeX=p.slopeX;out.slopeZ=p.slopeZ;out.flowX=p.flowX;out.flowZ=p.flowZ;
    return out;
  }
}
