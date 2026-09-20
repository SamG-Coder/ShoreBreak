import * as THREE from 'three';
import { glslDefines } from '../config.js';
import { NOISE, BED, SKY, OPTICS } from '../glsl/common.js';
import { SWE_SAMPLE } from '../glsl/water.js';
import { MACRO_NOISE_GLSL } from './macroNoise.js';
import { TERRAIN_GLSL } from './terrain.js';

const VERT=/* glsl */ `
uniform vec2 uTile,uBeachFocus;
in vec2 aCell;
out vec3 vP,vN,vLocal;flat out vec3 vSeed;out float vBed;
${TERRAIN_GLSL}
void main(){
  vec2 cell=uTile+aCell;
  vec3 seed=hash32(cell+17.31);vSeed=seed;
  vec2 xz=(cell+vec2(.25)+seed.xy*.5)*.2;
  float dist=length(xz-uBeachFocus);
  float keep=1.0-smoothstep(2.5,3.8,dist);
  float size=mix(.018,.037,seed.z)*mix(1.0,1.8,smoothstep(2.0,10.0,xz.y));
  float bias;float bed=terrainHeight(xz,terrainKeep(xz),bias)-bias;
  vBed=bed;
  vec3 scale=vec3(size,size*mix(.42,.7,seed.y),size*mix(.6,1.05,seed.x));
  float a=seed.z*31.7;mat2 rot=mat2(cos(a),-sin(a),sin(a),cos(a));
  vec3 p=position*scale*keep;
  p.xz=rot*p.xz;
  p.y-=scale.y*.26; // embedded in the gravel, never floating above it
  if(xz.y<-.7||xz.y>37.0) p.y-=1.0;
  vP=vec3(xz.x,bed,xz.y)+p;
  vec3 n=normal/scale;n.xz=rot*n.xz;vN=normalize(n);
  vLocal=position;
  gl_Position=projectionMatrix*viewMatrix*vec4(vP,1.0);
}`;
const FRAG=/* glsl */ `
in vec3 vP,vN,vLocal;flat in vec3 vSeed;in float vBed;
uniform float uSweFilmTau;
void main(){
  vec3 N=normalize(vN),V=normalize(cameraPosition-vP),L=uSunDir;
  vec2 uv=sweUV(vP.xz);float inside=sweInside(vP.xz);
  vec4 sw=textureLod(uSweView,uv, 0.0)*inside;
  vec2 wet=textureLod(uSweWet,uv, 0.0).xy*inside;
  float sub=smoothstep(.0002,.002,sw.x-max(vP.y-vBed,0.0));
  float damp=max(wet.y,.8*(1.0-smoothstep(.3,.75,vBed)));
  float retention=mix(.65,1.6,vSeed.x);
  float skin=max(sub,pow(clamp(wet.x,0.0,1.0),max(uSweFilmTau,.1)/(.55*retention)));
  float moisture=max(skin,damp*.7);
  vec3 warm=vec3(.235,.198,.168),cool=vec3(.22,.225,.221);
  vec3 alb=mix(warm,cool,smoothstep(.42,.8,vSeed.z));
  alb*=mix(.50,1.55,vSeed.x);
  if(vSeed.z>.93)alb=vec3(.52,.48,.41);
  // Mineral seams and worn irregularities; derivative filtering at oblique angles.
  float phase=(vLocal.x+vLocal.y*.7+vLocal.z*.3)*12.0+vSeed.y*8.0;
  float vein=1.0-smoothstep(.05,.12+fwidth(phase),abs(sin(phase)));
  alb=mix(alb,alb*1.22,vein*.35);
  alb*=mix(1.25,.65,moisture);
  float ao=mix(.54,1.0,smoothstep(-.25,.65,vLocal.y));
  float ndl=max(dot(N,L),0.0),ndv=max(dot(N,V),.001);
  vec3 bounce=vec3(.09,.075,.056)*(1.0-N.y)*.5;
  vec3 col=alb*(uSunColor*ndl/3.14159265+skyAmbient(N)*ao+bounce);
  vec3 H=normalize(V+L);float rough=mix(.68,.12,skin);
  float variance=.5*(dot(dFdx(N),dFdx(N))+dot(dFdy(N),dFdy(N)));
  float a=clamp(sqrt(pow(rough,4.0)+min(variance,.25)),.03,1.0);
  float F=mix(.04,.02,skin)+(1.0-.04)*pow(1.0-max(dot(V,H),0.0),5.0);
  col+=uSunColor*D_GGX(max(dot(N,H),0.0),a)*V_SmithGGXCorrelated(ndv,ndl,a)*F*ndl*(1.0-sub);
  vec3 R=reflect(-V,N);R.y=max(R.y,.025);
  col+=skyRadiance(normalize(R))*fresnelSchlick(ndv,.02)*skin*.3*(1.0-sub);
  gl_FragColor=vec4(col,1);
}`;

// Nine fixed 4 m tiles, 400 stones each. Only integer tile origins change when
// walking; world-coordinate hashes preserve every stone's size/colour/position.
export class NearPebbles {
  constructor(shared,macroNoise){
    this.group=new THREE.Group();this.group.name='Foreground rounded pebbles';
    const g0=new THREE.SphereGeometry(1,8,6),g=new THREE.InstancedBufferGeometry();
    g.index=g0.index;g.attributes={...g0.attributes};
    // Interleaved order distributes reduced-quality subsets across each tile.
    const cells=[];for(let i=0;i<400;i++){const j=(i*137)%400;cells.push(j%20,Math.floor(j/20));}
    g.setAttribute('aCell',new THREE.InstancedBufferAttribute(new Float32Array(cells),2));g.instanceCount=400;
    const pre=glslDefines()+NOISE+BED+SKY+OPTICS+SWE_SAMPLE+MACRO_NOISE_GLSL;
    for(let i=0;i<9;i++){
      const m=new THREE.ShaderMaterial({uniforms:{...shared,uTile:{value:new THREE.Vector2()},uBeachFocus:shared.uFocus,uMacroNoise:{value:macroNoise}},vertexShader:pre+VERT,fragmentShader:pre+FRAG});
      const mesh=new THREE.Mesh(g,m);mesh.frustumCulled=false;this.group.add(mesh);
    }
    this.group.userData.triangles=g.index.count/3*400*9;
  }
  setQuality(quality){
    const count={low:0,medium:100,auto:200,high:400,ultra:400}[quality]??200;
    this.group.visible=count>0;
    this.group.children[0].geometry.instanceCount=count;
  }
  update(camera){
    const x=Math.floor(camera.position.x/4),z=Math.floor(camera.position.z/4);
    for(let i=0;i<9;i++)this.group.children[i].material.uniforms.uTile.value.set((x+i%3-1)*20,(z+Math.floor(i/3)-1)*20);
  }
}
