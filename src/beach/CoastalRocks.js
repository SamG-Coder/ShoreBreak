import * as THREE from 'three';
import { glslDefines } from '../config.js';
import { NOISE,BED,SKY,OPTICS } from '../glsl/common.js';
import { BREAKER } from '../glsl/breaker.js';
import { SWE_SAMPLE,CHOP } from '../glsl/water.js';
import { CAUSTICS } from '../glsl/underwater.js';
import { SWASH_FAR } from '../glsl/swashfar.js';
import { COAST_SHADOW,coastUniforms } from './CoastMaterial.js';
import { HAZE_GLSL } from './haze.js';
import { ROCKS,ROCK_SEGMENTS,ROCK_RINGS,rockVertex } from './CoastalBed.js';
import { ROCK_WET_WIDTH,ROCK_WET_HEIGHT,ROCK_DRYING_GLSL } from './RockWetness.js';
import { photoUniforms,loadCoastTextures,PHOTO_GLSL } from './SandMaterial.js';

export class CoastalRocks {
 constructor(shared){
  const positions=[],indices=[],coords=[],wetUV=[];
  const N=ROCK_SEGMENTS,R=ROCK_RINGS;
  for(const [k,r] of ROCKS.entries()){
   const base=positions.length/3;
   for(let j=0;j<=R;j++)for(let i=0;i<=N;i++){
    const [x,y,z]=rockVertex(r,i,j);
    positions.push(x,y,z);coords.push(x+r.seed*2.3,y+r.seed*.6,z-r.seed);
    wetUV.push((i+.5)/ROCK_WET_WIDTH,(k*(R+1)+j+.5)/ROCK_WET_HEIGHT);
   }
   for(let j=0;j<R;j++)for(let i=0;i<N;i++){
    const a=base+j*(N+1)+i,b=a+1,d=a+N+1,e=d+1;indices.push(a,b,d,b,e,d);
   }
  }
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.setAttribute('aPhoto',new THREE.Float32BufferAttribute(coords,3));g.setAttribute('aWetUV',new THREE.Float32BufferAttribute(wetUV,2));g.setIndex(indices);g.computeVertexNormals();
  // The wrapped seam and coincident crown must share one normal; otherwise
  // specular highlights reveal the radial topology at low underwater angles.
  const normals=g.getAttribute('normal');
  for(let k=0;k<ROCKS.length;k++){
   const base=k*(R+1)*(N+1);let cx=0,cy=0,cz=0;
   for(let i=0;i<N;i++){cx+=normals.getX(base+i);cy+=normals.getY(base+i);cz+=normals.getZ(base+i);}
   const cl=Math.hypot(cx,cy,cz)||1;
   for(let i=0;i<=N;i++)normals.setXYZ(base+i,cx/cl,cy/cl,cz/cl);
   for(let j=1;j<=R;j++){
    const a=base+j*(N+1),b=a+N;
    const nx=normals.getX(a)+normals.getX(b),ny=normals.getY(a)+normals.getY(b),nz=normals.getZ(a)+normals.getZ(b),l=Math.hypot(nx,ny,nz)||1;
    normals.setXYZ(a,nx/l,ny/l,nz/l);normals.setXYZ(b,nx/l,ny/l,nz/l);
   }
  }
  g.computeBoundingSphere();
  const pre=glslDefines()+NOISE+BED+SKY+OPTICS+BREAKER+SWE_SAMPLE+SWASH_FAR+COAST_SHADOW+HAZE_GLSL;
  this.material=new THREE.ShaderMaterial({uniforms:{...shared,...coastUniforms(shared),...photoUniforms()},vertexShader:/* glsl */`
   uniform sampler2D uRockWetness;in vec2 aWetUV;in vec3 aPhoto;out vec3 vW,vN,vPhoto;out vec2 vRockWet;
   void main(){vW=position;vN=normal;vPhoto=aPhoto;vRockWet=textureLod(uRockWetness,aWetUV,0.).xy;gl_Position=projectionMatrix*viewMatrix*vec4(position,1.);}`,
   fragmentShader:pre+CHOP+CAUSTICS+PHOTO_GLSL+ROCK_DRYING_GLSL+/* glsl */`
   in vec3 vW,vN,vPhoto;in vec2 vRockWet;
   vec3 triColor(sampler2D map,vec3 p,vec3 w){return texture(map,p.yz).rgb*w.x+texture(map,p.xz).rgb*w.y+texture(map,p.xy).rgb*w.z;}
   void main(){
    vec3 N=normalize(vN),V=normalize(cameraPosition-vW),L=uSunDir;
    vec3 w=pow(abs(N),vec3(5.));w/=max(dot(w,vec3(1.)),.001);vec3 p=vPhoto/1.8;
    vec3 photo=triColor(uPhotoColor,p,w);
    vec4 nrx=texture(uPhotoNormal,p.yz),nry=texture(uPhotoNormal,p.xz),nrz=texture(uPhotoNormal,p.xy);
    vec3 nx=nrx.xyz*2.-1.,ny=nry.xyz*2.-1.,nz=nrz.xyz*2.-1.;
    vec3 detail=vec3(0,nx.x,nx.y)*w.x+vec3(ny.x,0,ny.y)*w.y+vec3(nz.x,nz.y,0)*w.z;
    N=normalize(N+detail*.55);
    vec2 su=sweUV(vW.xz);float wi=sweInside(vW.xz)*(1.-swashFarWeight(vW.xz));
    vec4 sv=textureLod(uSweView,su, 0.0);
    vec4 far=vec4(0.);if(wi<1.)far=swashFar(vW.xz,uTime);
    float level=mix(far.w,sv.w,wi);
    // Dry solver cells store the obstacle elevation. Only an actual water
    // column wets a crown; adjacent columns wet the sides as the surge rises.
    float sub=mix(rockWaterContact(far,vW.y),rockWaterContact(sv,vW.y),wi);
    float dryRough=clamp(dot(vec3(nrx.a,nry.a,nrz.a),w),.58,.94);
    // Existing photographed grain/cracks retain a little more moisture. This
    // variation is anchored to the rock and adds no texture or noise lookup.
    float crevice=clamp((.48-dot(photo,vec3(.2126,.7152,.0722)))*2.+(dryRough-.58),0.,1.);
    float damp=max(sub,pow(clamp(vRockWet.y,0.,1.),mix(1.,.65,crevice)));
    float gloss=max(sub,clamp(vRockWet.x,0.,1.));
    vec3 alb=photo*vec3(1.08,1.075,1.045);
    alb=mix(alb,alb*vec3(.48,.50,.51),damp);
    float rough=mix(dryRough-.07*damp,.18,gloss);
    float nl=max(dot(N,L),0.),nv=max(dot(N,V),.001),sh=coastShadow(vW,N)*localRockShadow(vW+N*.025,L);vec3 H=normalize(V+L);
    float h=max(level-vW.y,0.)*sub;
    float ao=mix(.77,1.,smoothstep(-.07,.22,vW.y-bedHeight(vW.xz)));
    vec3 col=alb*(uSunColor*nl*sh/3.14159265+skyAmbient(N)*ao)*exp(-WATER_SIGMA_A*h/max(L.y,.2));
    float a=rough*rough,F=fresnelSchlick(max(dot(V,H),0.),mix(.04,.0204,gloss));
    col+=uSunColor*min(D_GGX(max(dot(N,H),0.),a)*V_SmithGGXCorrelated(nv,nl,a)*F*nl,1.2)*sh;
    col+=skyAmbient(reflect(-V,N))*.11*gloss*fresnelSchlick(nv,.0204);
    col*=mix(1.,seabedCaustic(vW,h),nl*sh*sub);
    col=mix(col,landHazeColor(-V),landHaze(length(cameraPosition-vW)));
    gl_FragColor=vec4(col,1.);
   }`});
  this.mesh=new THREE.Mesh(g,this.material);this.mesh.name='Coastal limestone rocks';
 }
 loadAssets(renderer){return loadCoastTextures(renderer,'rock',this.material.uniforms);}
}
