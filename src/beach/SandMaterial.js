import * as THREE from 'three';
import {assetUrl} from '../core/assets.js';
import { CAUSTICS } from '../glsl/underwater.js';

export async function loadCoastTextures(renderer,kind,uniforms){
  await Promise.all(['diff','normal-rough'].map(async suffix=>{
    const texture=await new THREE.TextureLoader().loadAsync(assetUrl(`assets/coast-r8/${kind}-${suffix}.webp`));
    texture.wrapS=texture.wrapT=THREE.RepeatWrapping;
    texture.minFilter=THREE.LinearMipmapLinearFilter;
    texture.magFilter=THREE.LinearFilter;
    texture.colorSpace=suffix==='diff'?THREE.SRGBColorSpace:THREE.NoColorSpace;
    texture.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());
    uniforms['uPhoto'+({diff:'Color','normal-rough':'Normal'}[suffix])].value=texture;
    renderer.initTexture(texture);
  }));
}
export const photoUniforms=()=>Object.fromEntries(['Color','Normal'].map(x=>['uPhoto'+x,{value:null}]));
export const PHOTO_GLSL=/* glsl */`
uniform sampler2D uPhotoColor,uPhotoNormal;
`;
export const SAND_FRAG=PHOTO_GLSL+CAUSTICS+/* glsl */`
in vec3 vWorld;in vec3 vNb;in float vViewZ;
vec2 sandMoisture(vec2 p){
  float w=sweInside(p)*(1.-swashFarWeight(p));
  vec2 wet=textureLod(uSweWet,sweUV(p), 0.0).xy*w;
  if(w<1.)wet+=(1.-w)*swashFarWet(p,uTime);
  return wet;
}
void main(){
  vec3 P=vWorld,V=normalize(cameraPosition-P+vec3(0,0,bayBend(P.x))),Nb=normalize(vNb);
  vec2 uv=P.xz*.5;
  // Two rotated photographic layers and broad mineral variation conceal tiling.
  mat2 rot=mat2(.6,.8,-.8,.6);vec2 uv2=rot*uv*.73+vec2(.317,.671);
  float blend=smoothstep(-.55,.55,gnoise(P.xz*.14+17.3));
  vec3 photo=mix(texture(uPhotoColor,uv).rgb,texture(uPhotoColor,uv2).rgb,blend);
  vec4 nr1=texture(uPhotoNormal,uv),nr2=texture(uPhotoNormal,uv2);
  vec3 n1=nr1.xyz*2.-1.,n2=nr2.xyz*2.-1.;
  float footprint=max(length(dFdx(P.xz)),length(dFdy(P.xz)));
  // Millimetre grains arrive gradually as they become resolvable. Keep the
  // broad photographic colour while feathering micro-normal/contrast detail.
  float fine=1.-smoothstep(.0015,.014,footprint);
  vec2 slope=mix(n1.xy,transpose(rot)*n2.xy,blend)*.36*mix(.35,1.,fine);
  float rippleZone=smoothstep(-.22,-.7,P.y)*(1.-smoothstep(.008,.025,footprint));
  float ripplePhase=P.z*48.+2.1*gnoise(P.xz*.85)+.7*gnoise(P.xz*2.2);
  slope.y+=sin(ripplePhase)*.095*rippleZone;
  vec3 Tx=normalize(vec3(Nb.y,-Nb.x,0)),Tz=cross(Tx,Nb);
  vec3 N=normalize(Nb+Tx*slope.x+Tz*slope.y);
  float w=sweInside(P.xz)*(1.-swashFarWeight(P.xz));vec2 su=sweUV(P.xz);
  vec4 sv=textureLod(uSweView,su, 0.0);vec2 wet=sandMoisture(P.xz);
  // Capillary moisture spreads through sand over centimetres. A world-space
  // feather also keeps the coarser far-field wetness texels from reading as
  // an angular hard edge or appearing in chunks as the window moves.
  if(P.y>-.02&&P.y<.8){
    vec2 blur=(sandMoisture(P.xz+vec2(.16,0))+sandMoisture(P.xz-vec2(.16,0))+
               sandMoisture(P.xz+vec2(0,.16))+sandMoisture(P.xz-vec2(0,.16)))*.25;
    float feather=smoothstep(-.02,.05,P.y)*(1.-smoothstep(.65,.8,P.y));
    wet=mix(wet,mix(wet,blur,vec2(.45,.8)),feather);
  }
  vec4 far=vec4(0);if(w<1.)far=swashFar(P.xz,uTime);
  float wave=P.z<.5&&P.z> -9.?brkSurfaceOnly(P.xz,uTime):0.;
  // Dry fluid cells store their BED elevation in w. Comparing that directly
  // with a coarser rendered triangle created false water films and hard dark
  // edges as the terrain morphed. Only an actual water column can wet sand.
  float column=max(mix(far.x,sv.x,w),0.);
  float h=max(mix(max(far.w,0.),sv.w,w)+wave-P.y,0.)*smoothstep(.0003,.002,column);
  float sub=smoothstep(.001,.008,h);
  float damp=max(wet.y,1.-smoothstep(.04,.40,P.y+.018*gnoise(P.xz*1.1)));
  float skin=max(sub,pow(clamp(wet.x,0.,1.),3.8));
  // Ivory carbonate sand with a warm mineral component; preserve the photographed
  // grain variation without carrying the original brown exposure into the palette.
  float grain=clamp(dot(photo,vec3(.2126,.7152,.0722))/.165,.35,2.2);
  vec3 alb=vec3(.63,.555,.423)*pow(grain,mix(.23,.32,fine));
  alb*=1.+.025*cos(ripplePhase)*rippleZone;
  alb*=1.+.045*gnoise(P.xz*.27)+.035*gnoise(P.xz*1.7);
  alb=mix(alb,alb*vec3(.51,.48,.43),clamp(damp*.70+skin*.30,0.,1.));
  float rough=mix(clamp(mix(nr1.a,nr2.a,blend),.68,.94),.23,skin);
  vec3 L=uSunDir,H=normalize(L+V);float nl=max(dot(N,L),0.),nv=max(dot(N,V),.001);
  float sh=coastShadow(P,N)*localRockShadow(P+N*.014,L);
  vec3 Ts=exp(-WATER_SIGMA_A*h/max(L.y,.2));
  vec3 col=alb*(uSunColor*nl*sh/3.14159265*Ts+skyAmbient(N)*exp(-WATER_SIGMA_A*h*.7));
  float a=rough*rough,F=fresnelSchlick(max(dot(V,H),0.),mix(.035,.0204,skin));
  col+=uSunColor*min(D_GGX(max(dot(N,H),0.),a)*V_SmithGGXCorrelated(nv,nl,a)*F*nl,.7)*sh;
  // Low-energy moving focus on the bed, separate from the advected surface foam.
  col*=mix(1.,seabedCaustic(P,h),nl*sh);
  col=mix(col,landHazeColor(-V),landHaze(length(cameraPosition-P)));
  gl_FragColor=vec4(col,1.);
}`;
