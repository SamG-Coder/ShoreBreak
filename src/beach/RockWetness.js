import * as THREE from 'three';
import { FAR_MAP } from '../glsl/swashfar.js';
import { FullscreenPass, makeShader, PingPong } from '../core/gpu.js';
import { ROCKS, ROCK_SEGMENTS, ROCK_RINGS, rockVertex } from './CoastalBed.js';

// A fixed sample for every existing mesh vertex, independent of the scrolling
// sand simulation. RG = draining surface film / absorbed moisture. Float32 is
// intentional: half floats round long moisture decays back to 1 at small dt.
export const ROCK_WET_WIDTH=ROCK_SEGMENTS+1;
export const ROCK_WET_HEIGHT=(ROCK_RINGS+1)*ROCKS.length;
export const ROCK_DRYING_GLSL=/* glsl */ `
vec2 rockDrying(vec2 previous,float contact,float retention,float dt){
  vec2 tau=vec2(mix(2.8,7.5,retention),mix(110.,260.,retention));
  vec2 drained=previous*exp(-dt/tau);
  vec2 soak=vec2(1.)-exp(-contact*dt*vec2(18.,10.));
  return clamp(mix(drained,vec2(1.),soak),0.,1.);
}
// Evaluate each field's contact separately: interpolating water levels across
// a dry raised solver cell and the far field can falsely wet a rock crown.
float rockWaterContact(vec4 water,float y){
  return smoothstep(.001,.008,water.x)*smoothstep(-.014,.014,water.w-y);
}
`;
export const ROCK_WET_UPDATE=FAR_MAP+ROCK_DRYING_GLSL+/* glsl */ `
uniform sampler2D uRockSamples,uRockWetPrevious,uSweView,uSwashFarView;
uniform vec4 uSweDom;
uniform float uRockWetDt,uRockWetReset;
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec4 samplePoint=texelFetch(uRockSamples,p,0);
  vec3 P=samplePoint.xyz;
  if(uRockWetReset>.5){
    // Only the still-water fringe starts damp; the normal loader warm-up
    // records wave contact on exposed crowns before the first visible frame.
    float damp=(1.-smoothstep(-.035,.07,P.y))*.8;
    gl_FragColor=vec4(0.,damp,0.,1.);return;
  }
  vec2 uv=(P.xz-uSweDom.xy)/uSweDom.zw;
  float inside=step(0.,uv.x)*step(uv.x,1.)*step(0.,uv.y)*step(uv.y,1.);
  float nearWeight=inside*(1.-swashFarWeight(P.xz));
  float contact=0.;
  if(nearWeight>0.)contact=nearWeight*rockWaterContact(textureLod(uSweView,uv,0.),P.y);
  if(nearWeight<1.){
    vec2 farUV=swashFarUV(P.xz);
    float valid=step(0.,farUV.x)*step(farUV.x,1.)*step(0.,farUV.y)*step(farUV.y,1.);
    contact+=(1.-nearWeight)*valid*rockWaterContact(textureLod(uSwashFarView,farUV,0.),P.y);
  }
  vec2 wet=rockDrying(texelFetch(uRockWetPrevious,p,0).xy,contact,samplePoint.w,uRockWetDt);
  gl_FragColor=vec4(wet,0.,1.);
}
`;

export class RockWetness {
  constructor(renderer,shared){
    this.renderer=renderer;this.shared=shared;this.elapsed=0;
    const data=new Float32Array(ROCK_WET_WIDTH*ROCK_WET_HEIGHT*4);
    ROCKS.forEach((rock,k)=>{
      for(let j=0;j<=ROCK_RINGS;j++)for(let i=0;i<=ROCK_SEGMENTS;i++){
        const [x,y,z]=rockVertex(rock,i,j);
        // Sheltered lower faces retain moisture longer; gentle, fixed patches
        // avoid a uniform drying line. Duplicate seam/centre vertices agree.
        const lower=1-THREE.MathUtils.clamp((y-rock.base)/rock.h,0,1);
        const patch=.5+.5*Math.sin(x*6.7+z*4.9)*Math.sin(z*8.1-x*3.2);
        const retention=THREE.MathUtils.clamp(.15+.6*lower+.25*patch,0,1);
        data.set([x,y,z,retention],((k*(ROCK_RINGS+1)+j)*ROCK_WET_WIDTH+i)*4);
      }
    });
    this.samples=new THREE.DataTexture(data,ROCK_WET_WIDTH,ROCK_WET_HEIGHT,THREE.RGBAFormat,THREE.FloatType);
    this.samples.minFilter=this.samples.magFilter=THREE.NearestFilter;this.samples.generateMipmaps=false;this.samples.needsUpdate=true;
    this.state=new PingPong(ROCK_WET_WIDTH,ROCK_WET_HEIGHT);
    shared.uRockWetness={value:this.state.read.texture};
    this.pass=new FullscreenPass(makeShader(ROCK_WET_UPDATE,{
      ...shared,uRockSamples:{value:this.samples},uRockWetPrevious:{value:this.state.read.texture},
      uRockWetDt:{value:0},uRockWetReset:{value:0},
    }));
  }
  reset(){
    this.elapsed=0;const u=this.pass.material.uniforms;u.uRockWetReset.value=1;
    // Bind the opposite attachment even in the reset branch: WebGL rejects
    // sampling an attached render target regardless of dynamic control flow.
    u.uRockWetPrevious.value=this.state.write.texture;this.pass.render(this.renderer,this.state.read);
    u.uRockWetPrevious.value=this.state.read.texture;this.pass.render(this.renderer,this.state.write);
    u.uRockWetReset.value=0;this.shared.uRockWetness.value=this.state.read.texture;
  }
  step(dt){
    if(!(dt>0))return;
    this.elapsed+=dt;
    if(this.elapsed<1/60-1e-7)return;
    const u=this.pass.material.uniforms;
    u.uRockWetDt.value=this.elapsed;u.uRockWetPrevious.value=this.state.read.texture;
    this.pass.render(this.renderer,this.state.write);this.state.swap();this.elapsed=0;
    this.shared.uRockWetness.value=this.state.read.texture;
  }
}
