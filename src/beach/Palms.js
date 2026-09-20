import * as THREE from 'three';
import { bayBendJS } from './terrain.js';
import { COAST_PRE, coastUniforms } from './CoastMaterial.js';

const ROOT='/assets/palms-r6/';
const TYPES={Float32Array,Uint16Array,Uint32Array,Int16Array,Uint8Array};
const VERT=/* glsl */`
uniform float uTime;
uniform float uLeaf;
in vec4 frondContact;
out vec3 vW;out vec3 vN;out vec3 vColor;out vec2 vUv;
void main(){
  vec3 p=position,n=normal;
  vec3 root=instanceMatrix[3].xyz;
  float phase=dot(root.xz,vec2(.07,.11));
  float gust=.7+.3*sin(uTime*.29+phase);
  // Rotation about each frond's attachment preserves connected leaflets.
  float a=uLeaf*frondContact.w*frondContact.w*(.008*sin(uTime*.83+phase-frondContact.w*1.7)) * gust;
  vec3 q=p-frondContact.xyz;
  p=frondContact.xyz+vec3(q.x,cos(a)*q.y-sin(a)*q.z,sin(a)*q.y+cos(a)*q.z);
  n=vec3(n.x,cos(a)*n.y-sin(a)*n.z,sin(a)*n.y+cos(a)*n.z);
  float bend=.00018*(.8+sin(uTime*.48+phase)*.2);
  p.x+=p.y*p.y*bend;n.y-=2.0*p.y*bend*n.x;
  vec4 world=modelMatrix*instanceMatrix*vec4(p,1.0);
  vW=world.xyz;vN=normalize(mat3(modelMatrix*instanceMatrix)*n);
  vColor=color;vUv=uv;
  gl_Position=projectionMatrix*viewMatrix*world;
}`;
const FRAG=/* glsl */`
uniform float uLeaf;
uniform sampler2D uBark,uBarkNormal,uBarkRough;
in vec3 vW;in vec3 vN;in vec3 vColor;in vec2 vUv;
void main(){
 vec3 N=normalize(vN);if(!gl_FrontFacing)N=-N;
 vec3 alb=vColor,V=normalize(cameraPosition-vW),L=uSunDir;
 float rough=.75;
 if(uLeaf<.5){
   alb=texture(uBark,vUv).rgb*(.72+vColor*.65);
   rough=texture(uBarkRough,vUv).r;
   vec3 T=normalize(dFdx(vW)*dFdy(vUv.y)-dFdy(vW)*dFdx(vUv.y));
   vec3 B=normalize(-dFdx(vW)*dFdy(vUv.x)+dFdy(vW)*dFdx(vUv.x));
   vec3 bump=texture(uBarkNormal,vUv).xyz*2.0-1.0;
   N=normalize(N*bump.z+(T*bump.x+B*bump.y)*.48);
 } else {
   float rib=exp(-pow((vUv.x-.5)/max(.13,fwidth(vUv.x)),2.0));
   alb*=.94+.10*rib;
   rough=.56;
 }
 float sh=coastShadow(vW,N),ndl=max(dot(N,L),0.0);
 float skyVis=mix(.64,.94,smoothstep(-.7,.6,N.y));
 vec3 col=alb*(uSunColor*ndl*sh/3.14159265+skyAmbient(N)*skyVis+vec3(.035,.03,.018)*(1.0-N.y)*.5);
 if(uLeaf>.5){
   float through=pow(max(dot(-N,L),0.0),1.7);
   col+=alb*vec3(1.12,1.23,.75)*uSunColor*through*.19*sh;
 }
 float normalVariance=min(.5*(dot(dFdx(N),dFdx(N))+dot(dFdy(N),dFdy(N))),1.0);
 col+=uSunColor*pow(max(dot(N,normalize(V+L)),0.0),mix(mix(45.0,12.0,rough),6.0,normalVariance))*.012*ndl*sh;
 col=mix(col,landHazeColor(-V),landHaze(length(cameraPosition-vW)));
 gl_FragColor=vec4(col,1.0);
}`;

async function bytes(url){
  const compressed=typeof DecompressionStream!=='undefined';
  const response=await fetch(url+(compressed?'.gz':''));
  if(!response.ok)throw new Error('Coast asset unavailable: '+url);
  const buffer=await response.arrayBuffer(),head=new Uint8Array(buffer,0,Math.min(buffer.byteLength,2));
  // Some CDNs decode .gz in transit. Inspect the bytes so either response works.
  if(compressed && head[0]===31 && head[1]===139)
    return new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  return buffer;
}
export async function loadPalms(renderer,shared){
  const response=await fetch(ROOT+'manifest.json');if(!response.ok)throw new Error('Palm manifest unavailable');
  const manifest=await response.json(),loader=new THREE.TextureLoader();
  const [maps,geometries]=await Promise.all([
    Promise.all(['diff','normal','rough'].map(async k=>{
      const t=await loader.loadAsync(ROOT+`bark-${k}.webp`);t.wrapS=t.wrapT=THREE.RepeatWrapping;
      t.anisotropy=Math.min(4,renderer.capabilities.getMaxAnisotropy());if(k==='diff')t.colorSpace=THREE.SRGBColorSpace;
      renderer.initTexture(t);return t;
    })),
    Promise.all(manifest.map(async entry=>{
      const buffer=await bytes(ROOT+entry.file),g=new THREE.BufferGeometry();
      for(const [name,a] of Object.entries(entry.attributes))g.setAttribute(name,new THREE.BufferAttribute(new TYPES[a.type](buffer,a.offset,a.length),a.itemSize,a.normalized));
      const ix=entry.index;g.setIndex(new THREE.BufferAttribute(new TYPES[ix.type](buffer,ix.offset,ix.length),1));
      if(!g.attributes.frondContact)g.setAttribute('frondContact',new THREE.Float32BufferAttribute(new Float32Array(g.attributes.position.count*4),4));
      g.computeBoundingSphere();return {entry,g};
    }))
  ]);
  const materials=[0,1].map(leaf=>new THREE.ShaderMaterial({
    vertexColors:true,side:leaf?THREE.DoubleSide:THREE.FrontSide,
    uniforms:{...coastUniforms(shared),uLeaf:{value:leaf},uBark:{value:maps[0]},uBarkNormal:{value:maps[1]},uBarkRough:{value:maps[2]}},
    vertexShader:COAST_PRE+VERT,fragmentShader:COAST_PRE+FRAG,
  }));
  const group=new THREE.Group();group.name='Ashore / Jellys date palms';
  const dummy=new THREE.Object3D();
  for(const {entry,g} of geometries)for(const side of [-1,1]){
    const members=[];
    for(let i=-30;i<=30;i++){
      const x=i*14.2+Math.sin(i*7.31)*1.4;
      const tier=Math.abs(x)<85?0:Math.abs(x)<205?1:2;
      if(tier!==entry.tier || (i%2===0?44:288)!==entry.seed || (x<0?-1:1)!==side)continue;
      members.push({x,z:Math.abs(i%3)===1?49.5:59.8,scale:.79+.12*(.5+.5*Math.sin(i*11.7)),angle:i*2.4});
    }
    if(!members.length)continue;
    const mesh=new THREE.InstancedMesh(g,materials[entry.part==='leaves'?1:0],members.length);
    members.forEach((p,i)=>{dummy.position.set(p.x,5.56,p.z-bayBendJS(p.x));dummy.rotation.set(0,p.angle,0);dummy.scale.setScalar(p.scale);dummy.updateMatrix();mesh.setMatrixAt(i,dummy.matrix);});
    mesh.instanceMatrix.needsUpdate=true;mesh.computeBoundingSphere();mesh.boundingSphere.radius+=1;
    group.add(mesh);
  }
  group.userData.triangles=group.children.reduce((n,m)=>n+m.geometry.index.count/3*m.count,0);
  return group;
}
