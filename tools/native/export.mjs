import fs from 'node:fs';
import path from 'node:path';
import * as THREE from '../../node_modules/three/build/three.module.js';
globalThis.location={search:process.env.QA_MODE==='clip'?'?clip&capture':'?explore&capture'};
fs.mkdirSync('.qa',{recursive:true});
const root=path.resolve('public');
globalThis.fetch=async url=>new Response(fs.readFileSync(path.join(root,url)));
THREE.TextureLoader.prototype.loadAsync=async function(url){const t=new THREE.Texture();t.userData.file=path.join(root,url);return t;};
const {CONFIG,sunDirection}=await import('../../src/config.js');
const {skyLut,LIGHT}=await import('../../src/core/look.js');
const {Swell}=await import('../../src/water/Swell.js');
const {UnderwaterPlume}=await import('../../src/water/UnderwaterPlume.js');
const {UnderwaterBubbles}=await import('../../src/water/UnderwaterBubbles.js');
const {SurfaceProbe}=await import('../../src/water/SurfaceProbe.js');
const {Schedule}=await import('../../src/core/schedule.js');
const {SwashSim}=await import('../../src/swash/SwashSim.js');
const {WaterSurface,makeChop}=await import('../../src/water/WaterSurface.js');
const {LipRibbon}=await import('../../src/water/LipRibbon.js');
const {Beach}=await import('../../src/beach/Beach.js');
const {initCoastalBed}=await import('../../src/beach/CoastalBed.js');
const {loadPalms}=await import('../../src/beach/Palms.js');
const {bakeCoastShadows}=await import('../../src/beach/CoastMaterial.js');
const {Sky}=await import('../../src/sky/Sky.js');
const {Whitewater}=await import('../../src/whitewater/Whitewater.js');
const {Post}=await import('../../src/post/Post.js');
const {ContactLight,CONTACT_SAMPLE}=await import('../../src/post/ContactLight.js');
const {FullscreenPass,makeShader}=await import('../../src/core/gpu.js');
const W=+(process.env.QA_WIDTH||960),H=+(process.env.QA_HEIGHT||Math.round(W*9/16));
const D={textures:{},targets:{},geometries:{},materials:{},uniforms:[],draws:[],commands:[],width:W,height:H};
const b64=a=>Buffer.from(a.buffer,a.byteOffset,a.byteLength).toString('base64');
const matMap=new Map(),uniformMap=new Map(),texVersions=new Map();
function tex(t){
 const id=t.uuid;
 if(!D.textures[id])D.textures[id]={width:t.image?.width||1,height:t.image?.height||1,volume:t.isData3DTexture?t.image.depth:0,array:t.isDataArrayTexture?t.image.depth:0,format:t.format,type:t.type,depth:!!t.isDepthTexture,repeat:t.wrapS===THREE.RepeatWrapping,linear:t.magFilter===THREE.LinearFilter,mips:t.generateMipmaps||t.minFilter===THREE.LinearMipmapLinearFilter,flip:t.flipY,srgb:t.colorSpace===THREE.SRGBColorSpace,file:t.userData.file};
 if(t.image?.data&&texVersions.get(id)!==t.version){
  const data={data:b64(t.image.data),type:t.image.data.constructor.name};
  if(texVersions.has(id))D.commands.push({upload:id,...data});else Object.assign(D.textures[id],data);
  texVersions.set(id,t.version);
 }
 return {texture:id};
}
function uniform(v){if(v==null)return null;if(v.isTexture)return tex(v);if(v.elements)return v.elements.slice();if(v.toArray)return v.toArray();if(ArrayBuffer.isView(v))return Array.from(v);if(Array.isArray(v))return v.flatMap(x=>x?.toArray?x.toArray():x);return v;}
function intern(o){const s=JSON.stringify(o);let id=uniformMap.get(s);if(id===undefined){id=D.uniforms.length;D.uniforms.push(o);uniformMap.set(s,id);}return id;}
class Recorder {
 target=null;layer=0;alpha=1;
 targetKey(){return this.target?this.target.texture.uuid+(this.target.isWebGLArrayRenderTarget?':layer:'+this.layer:''):'screen';}clearColor=new THREE.Color();capabilities={getMaxAnisotropy:()=>4};extensions={get:()=>null};
 getRenderTarget(){return this.target;}getActiveCubeFace(){return this.layer;}getActiveMipmapLevel(){return 0;}initTexture(t){tex(t);}
 getClearAlpha(){return this.alpha;}setClearAlpha(a){this.alpha=a;}getClearColor(c){return c.copy(this.clearColor);}
 setClearColor(c,a=1){this.clearColor.set(c);this.alpha=a;}
 setRenderTarget(rt,layer=0){this.target=rt;this.layer=layer;const key=this.targetKey();if(rt&&!D.targets[key]){D.targets[key]={width:rt.width,height:rt.height,colors:rt.textures.map(t=>tex(t).texture),layer:rt.isWebGLArrayRenderTarget?layer:null,depth:rt.depthTexture?tex(rt.depthTexture).texture:rt.depthBuffer};}}
 clear(color=true,depth=true){D.commands.push({clear:this.targetKey(),color:color?[...this.clearColor.toArray(),this.alpha]:null,depth});}
 render(scene,camera){
  scene.updateMatrixWorld(true);camera.updateMatrixWorld();
  const clip=new THREE.Frustum().setFromProjectionMatrix(new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse));
  const meshes=[];scene.traverseVisible(m=>{if(m.isMesh&&(m.frustumCulled===false||clip.intersectsObject(m)))meshes.push(m);});
  meshes.sort((a,b)=>a.renderOrder-b.renderOrder||Number(a.material.transparent)-Number(b.material.transparent));
  for(const m of meshes){
   m.onBeforeRender(this,scene,camera,m.geometry,m.material,null);
   const g=m.geometry,mat=scene.overrideMaterial||m.material;
   const gid=g.uuid+':'+(g.index?.version||0);
   const drawStart=g.drawRange.start,drawCount=Number.isFinite(g.drawRange.count)?g.drawRange.count:(g.index?.count||g.attributes.position.count);
   if(!D.geometries[gid]){
    const attributes={};for(const [n,a]of Object.entries(g.attributes)){
     const f=new Float32Array(a.count*a.itemSize);for(let i=0;i<a.count;i++)for(let j=0;j<a.itemSize;j++)f[i*a.itemSize+j]=a.getComponent(i,j);
     if(!f.every(Number.isFinite))throw Error('Nonfinite '+n);
     attributes[n]={data:b64(f),size:a.itemSize,instanced:!!a.isInstancedBufferAttribute};
    }
    D.geometries[gid]={attributes,index:g.index?b64(Uint32Array.from(g.index.array)):null};
   }
   const depth=!!mat.isMeshDepthMaterial;
   const key=JSON.stringify([depth?'depth':mat.vertexShader,depth?'depth':mat.fragmentShader,mat.defines,!!m.instanceMatrix,!!m.instanceColor]);
   let mid=matMap.get(key);if(mid===undefined){mid=matMap.size;matMap.set(key,mid);D.materials[mid]={vertex:mat.vertexShader,fragment:mat.fragmentShader,defines:mat.defines,depth,instanced:!!m.instanceMatrix,instanceColor:!!m.instanceColor};}
   const uniforms=Object.fromEntries(Object.entries(mat.uniforms||{}).map(([k,v])=>[k,uniform(v.value)]));
   const mv=new THREE.Matrix4().multiplyMatrices(camera.matrixWorldInverse,m.matrixWorld);
   Object.assign(uniforms,{modelMatrix:m.matrixWorld.elements.slice(),viewMatrix:camera.matrixWorldInverse.elements.slice(),modelViewMatrix:mv.elements,normalMatrix:new THREE.Matrix3().getNormalMatrix(mv).elements,projectionMatrix:camera.projectionMatrix.elements.slice(),cameraPosition:camera.position.toArray()});
   const draw={material:mid,geometry:gid,drawStart,drawCount,uniforms:intern(uniforms),count:m.isInstancedMesh?m.count:(Number.isFinite(g.instanceCount)?g.instanceCount:(m.count??1)),side:mat.side,depthTest:mat.depthTest,depthWrite:mat.depthWrite,depthFunc:mat.depthFunc,blend:mat.transparent?mat.blending:0,equation:mat.blendEquation,src:mat.blendSrc,dst:mat.blendDst,instance:m.instanceMatrix?b64(m.instanceMatrix.array):null,instanceColor:m.instanceColor?b64(m.instanceColor.array):null,name:m.name||'pass'};
   D.commands.push({target:this.targetKey(),draw:D.draws.length});D.draws.push(draw);
  }
 }
}
const r=new Recorder(),V=a=>new THREE.Vector3(...a);
const shared={uFocus:{value:new THREE.Vector2(-34,6.8)},uTime:{value:0},uEvtCount:{value:0},uSunDir:{value:V(sunDirection())},uSunColor:{value:V(LIGHT.sun)},uSkyAmb:{value:V(LIGHT.skyAmb)},uSkyLut:{value:skyLut()},uNear:{value:CONFIG.camera.near},uFar:{value:CONFIG.camera.far},uInjMass:{value:1.7},uInjSpeed:{value:.95},uChop:{value:makeChop(20)},uWind:{value:new THREE.Vector2(Math.sin(8*Math.PI/180),Math.cos(8*Math.PI/180))},uResolution:{value:new THREE.Vector2(W,H)},uOpaqueColor:{value:null},uOpaqueDepth:{value:null},uBackDepth:{value:null},uDeepColor:{value:V([.017,.16,.22])},uWaterAtten:{value:V([.55,.085,.10])},uMilkColor:{value:V([.62,.72,.66])},uGlowTint:{value:V([.30,1,.85])},uLaceSeed:{value:new THREE.Vector2(.3,.7)}};
for(const x of 'ABCDEFG')shared['uEvt'+x]={value:new Float32Array(24)};
initCoastalBed(shared);
const schedule=new Schedule({seed:7}),swell=new Swell(shared,schedule),swe=new SwashSim(r,shared),water=new WaterSurface(shared,{cols:360,swell}),lips=[new LipRibbon(shared,0),new LipRibbon(shared,1)];
const beach=new Beach(r,shared);
if(CONFIG.explore){[beach.palms]=await Promise.all([loadPalms(r,shared),beach.promenade.loadAssets(r),beach.loadAssets(r)]);beach.mesh.add(beach.palms);shared.uSandPhoto.value=beach.material.uniforms.uPhotoColor.value;
bakeCoastShadows(r,[beach.backdrop.seafront.group,beach.palms,beach.rocks.mesh],shared);}
const sky=new Sky(shared),ww=new Whitewater(r,shared,schedule),post=new Post(r,W,H,shared),contact=new ContactLight();contact.resize(W,H);
ww.params.bufS=.7;ww.params.bufH=1080;
const hdr={type:THREE.HalfFloatType,minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter};
const opaque=new THREE.WebGLRenderTarget(W,H,{...hdr,depthTexture:new THREE.DepthTexture(W,H,THREE.UnsignedIntType)}),main=new THREE.WebGLRenderTarget(W,H,{...hdr,depthTexture:new THREE.DepthTexture(W,H,THREE.UnsignedIntType)}),back=new THREE.WebGLRenderTarget(W,H,{type:THREE.FloatType,format:THREE.RedFormat,minFilter:THREE.NearestFilter,magFilter:THREE.NearestFilter,depthBuffer:true}),particles=new THREE.WebGLRenderTarget(W,H,hdr);
shared.uOpaqueColor.value=opaque.texture;shared.uOpaqueDepth.value=opaque.depthTexture;shared.uBackDepth.value=back.texture;
const os=new THREE.Scene();os.add(sky.mesh,beach.mesh);const bs=new THREE.Scene();bs.add(water.backMesh,...lips.map(l=>l.backMesh));const ws=new THREE.Scene();ws.add(water.mesh,...lips.map(l=>l.mesh));
const copy=new FullscreenPass(makeShader(`uniform sampler2D uColor,uDepth;${CONTACT_SAMPLE}\nin vec2 vUv;void main(){float d=texture(uDepth,vUv).x;gl_FragColor=vec4(texture(uColor,vUv).rgb*contactAt(vUv,d),1);gl_FragDepth=d;}`,{uColor:{value:opaque.texture},uDepth:{value:opaque.depthTexture},uContact:{value:contact.target.texture},uContactTexel:{value:new THREE.Vector2(1/contact.target.width,1/contact.target.height)},uCameraRange:{value:new THREE.Vector2(CONFIG.camera.near,CONFIG.camera.far)}},{depthTest:true,depthWrite:true,depthFunc:THREE.AlwaysDepth}));
const cam=new THREE.PerspectiveCamera(58,W/H,CONFIG.camera.near,CONFIG.camera.far);cam.rotation.order='YXZ';
const probe=new SurfaceProbe(shared),bubbles=new UnderwaterBubbles(shared,schedule),us=new THREE.Scene();us.add(water.underMesh);
const plume=new UnderwaterPlume(shared,bubbles.material.uniforms);plume.resize(W,H);
if(process.env.QA_PROGRAMS){
 for(const scene of [os,bs,ws,us,bubbles.scene,swell.scene,...Object.values(ww).filter(v=>v?.isScene)])scene.traverse(o=>{if(o.isMesh)o.frustumCulled=false;});
 for(const pass of [...FullscreenPass.all])r.render(pass.scene,pass.camera);
 for(const [scene,camera]of [[os,cam],[bs,cam],[ws,cam],[us,cam],[bubbles.scene,cam],[swell.scene,swell.camera]])r.render(scene,camera);
 for(const value of Object.values(ww))if(value?.isScene)r.render(value,cam);
 fs.writeFileSync(process.env.QA_PROGRAMS,JSON.stringify({materials:D.materials}));
 console.log('Exported production shader programs',Object.keys(D.materials).length);process.exit(0);
}
if(process.env.QA_SHADER_OUTPUT){
 const dir=process.env.QA_SHADER_OUTPUT;fs.mkdirSync(dir,{recursive:true});
 const source=(m,stage)=>Object.entries(m.defines||{}).map(([k,v])=>`#define ${k} ${v}\n`).join('')+m[stage+'Shader'];
 for(const [name,m,stage]of [['under',water.underMaterial,'fragment'],['cloud',plume.pVolume.material,'fragment'],['coast',plume.pSurface.material,'fragment'],['water',water.material,'fragment'],['sand',beach.material,'vertex'],['flux',swe.pFlux.material,'fragment']])
  fs.writeFileSync(path.join(dir,name+(stage==='vertex'?'.vert':'.frag')),source(m,stage));
 // Unique production vertex sources for validating stationary background geometry.
 const scenery = new Map();
 beach.promenade.mesh.traverse(o=>{if(o.material?.vertexShader)scenery.set(o.material.vertexShader,o.material);});
 beach.backdrop.group.traverse(o=>{if(o.material?.vertexShader)scenery.set(o.material.vertexShader,o.material);});
 let n=0;for(const material of scenery.values())fs.writeFileSync(path.join(dir,'scenery-'+n+++'.vert'),source(material,'vertex'));
 process.exit(0);
}
function pack(t){schedule.pack(t);for(const x of 'ABCDEFG')shared['uEvt'+x].value.set(schedule[x]);shared.uEvtCount.value=schedule.count;shared.uTime.value=t;}
let t=+(process.env.QA_START||2.85)-12;
// Identical fixed timestep and twelve-second warmup to production.
for(let i=0;i<1440;i++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
D.commands.push({checkpoint:'warmed'});
function frame(name,x,z,yaw,pitch,eye=1.64,focusX=x){
 if(process.env.QA_R23)D.commands.push({frameStart:name});
 cam.position.set(x,beach.groundAt(x,z)+eye,z);cam.rotation.set(pitch*Math.PI/180,yaw*Math.PI/180,0);cam.updateMatrixWorld();shared.uFocus.value.set(focusX,z);pack(t);
 const slots=schedule.jetSlots(t,x);lips.forEach((l,i)=>{l.visible=i<slots.length;if(i<slots.length)l.slot=slots[i];});
 r.setClearColor(0,1);r.setRenderTarget(opaque);r.clear();r.render(os,cam);contact.render(r,opaque.depthTexture,cam,W,H);
 r.setRenderTarget(back);r.setClearColor(0,0);r.clear();r.render(bs,cam);
 probe.uniforms.uProbe.value.set(x,z);probe.pass.camera.position.copy(cam.position);probe.pass.render(r,probe.target);shared.uProbeOrigin.value.set(x,z);
 const submerged=!!(process.env.QA_R23||process.env.QA_ROCK_FIX||process.env.QA_R10||process.env.QA_R11||process.env.QA_R12||process.env.QA_RUNUP||process.env.QA_R14||process.env.QA_R15||process.env.QA_R16)&&z<1.5&&cam.position.y<.65;
 shared.uUnderwaterOn.value=submerged?1:0;post.waterDepth=submerged?-cam.position.y:-10;
 water.mesh.visible=!submerged||cam.position.y>-.10;
 r.setRenderTarget(main);r.setClearColor(0,1);r.clear();r.render(copy.scene,copy.camera);r.render(ws,cam);if(submerged){bubbles.update(t);r.render(us,cam);r.render(bubbles.scene,cam);}
 ww.render(cam,particles,main.depthTexture,W,H);post.plumeTexture=submerged?plume.render(r,cam,main.depthTexture):null;post.render(main.texture,particles.texture,t,null,cam,main.depthTexture);
 if(process.env.QA_R10||process.env.QA_R11)D.commands.push({probe:probe.target.texture.uuid,time:t,x,z});
 if(process.env.QA_RUNUP)D.commands.push({runup:shared.uSweView.value.uuid,foam:shared.uSweFoam.value.uuid,time:t,dom:swe.dom.toArray(),size:[swe.nx,swe.nz],focus:x});
 if(process.env.QA_R14&&submerged)D.commands.push({coast:plume.surface.texture.uuid,time:t,range:plume.uniforms.uCrashZ.value.toArray(),left:plume.uniforms.uFieldLeft.value,name});
 D.commands.push({save:name,time:t,camera:cam.position.toArray()});
}
if(process.env.QA_R23){
 const uw=(name,x,z,yaw,pitch,y=-.52)=>frame(name,x,z,yaw,pitch,y-beach.groundAt(x,z));
 frame('opening',-34,6.8,-94,-4.2);
 uw('sky-window',-34,-7,0,70);
 uw('belly-shore',-34,-4.2,180,14);
 uw('belly-side',-33,-3.1,112,15);
 uw('belly-close',-34,-2.6,180,27,-.42);
 uw('belly-seaward',-34,-1.9,0,22,-.16);
 uw('belly-seaward-side',-34,-2.1,-48,18,-.16);
 uw('belly-high',-34,-2.4,0,32,-.08);
 // Paired captures may disable only uCrashOn through QA_UNIFORM_OVERRIDES.
 for(let k=0;k<4;k++){
  uw('belly-motion-'+k,-34,-3.5,160,18,-.50);
  if(k<3)for(let i=0;i<18;i++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
 }
}else if(process.env.QA_ROCK_FIX){
 frame('under-rock-front',-35.2,-3,180,2,-.40-beach.groundAt(-35.2,-3));
 frame('under-rock-side',-37,-1.4,-90,3,-.30-beach.groundAt(-37,-1.4));
 frame('under-rock-low',-35.2,-2.2,180,0,-.50-beach.groundAt(-35.2,-2.2));
}else if(process.env.QA_R20){
 frame('opening',-34,6.8,-94,-4.2);
 for(let k=0;k<7;k++){
  frame('rocks-'+k,-34.8,.9,15,-36,.85);
  D.commands.push({rockWet:swe.rockWetness.state.read.texture.uuid,time:t,name:'rocks-'+k});
  if(k<6)for(let j=0;j<80;j++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
 }
 // Stationary-camera checks while the solver moves away and returns. The
 // persistent rock atlas must not be reset or attenuated by camera distance.
 for(const fx of [-27,-20,-34.8]){
  shared.uFocus.value.set(fx,.9);swe._follow();
  swe.pView.material.uniforms.uState.value=swe.state.read.texture;swe.pView.render(r,swe.view);
  swe.pHeight.material.uniforms.uState.value=swe.state.read.texture;swe.pHeight.render(r,swe.height);swe._publish();
  frame('rock-focus-'+fx,-34.8,.9,15,-36,.85,fx);
  D.commands.push({rockWet:swe.rockWetness.state.read.texture.uuid,time:t,name:'focus-'+fx});
 }
 frame('under-rocks',-34.8,-.9,150,7,.03);
}else if(process.env.QA_R19){
 // Freeze time and camera across actual solver-cell scrolls. A changing
 // picture here is a handoff artifact, not ordinary evolving whitewater.
 swe._bakeFar(t);
 function publishShift(){
  swe.pView.material.uniforms.uState.value=swe.state.read.texture;swe.pView.render(r,swe.view);
  swe.pHeight.material.uniforms.uState.value=swe.state.read.texture;swe.pHeight.render(r,swe.height);swe._publish();
 }
 for(const sign of [1,-1]){
  const boundary=swe.cx+sign,tag=sign>0?'forward':'reverse';
  for(const side of [-1,1]){
   const fx=boundary+sign*side*.00002;shared.uFocus.value.set(fx,1.8);
   if(swe._follow())publishShift();
   for(const [name,z,eye]of [['stand',1.8,1.64],['crouch',1.4,.82]])
    frame(tag+'-'+name+'-'+(side<0?'a':'b'),boundary,z,sign>0?-90:90,-9,eye,fx);
  }
 }
 // Short running sequence crosses repeated allocation boundaries with real
 // wave/foam evolution. It is intentionally separate from the frozen pairs.
 for(let k=0;k<10;k++){
  const x=-34+k*.45;shared.uFocus.value.set(x,1.8);
  for(let j=0;j<12;j++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
  frame('foam-run-'+String(k).padStart(2,'0'),x,1.8,-90,-9);
 }
}else if(process.env.QA_R16){
 frame('opening',-34,6.8,-94,-4.2);
 frame('front',-34,2.4,0,-7,.82);
 frame('rear',-34,-4,180,-10);
 frame('side',-34,-4,-110,-5);
 frame('shallows',-34,1.3,-90,-35,.82);
 frame('sand',-34,8,-90,-20);
 frame('window',-34,-7,0,65,-.52-beach.groundAt(-34,-7));
 frame('under-shore',-34,-4.2,180,5,-.52-beach.groundAt(-34,-4.2));
 for(let i=0;i<468;i++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
 frame('under-crash',-34,-4.2,180,5,-.52-beach.groundAt(-34,-4.2));
 frame('under-shallow',-34,-.5,180,1,.045);
}else if(process.env.QA_R15){
 frame('opening',-34,6.8,-94,-4.2);
 frame('window',-34,-7,0,65,-.52-beach.groundAt(-34,-7));
 // Fixed time/camera isolates LOD recentering from animated wave motion.
 for(const [name,z,yaw,pitch,eye,fx]of [
  ['lip-front',2.4,0,-7,.82,-34.08],
  ['lip-rear',-4,180,-10,1.64,-34.08],
  ['lip-side',-4,-110,-5,1.64,-34.08],
  ['sand',8,-90,-20,1.64,-34],
  ['shallows',1.3,-90,-35,.82,-34.08],
 ])for(const sign of [-1,1])frame(name+'-'+(sign<0?'a':'b'),-34,z,yaw,pitch,eye,fx+sign*.00002);
 for(let i=0;i<7;i++)frame('walk-'+i,-34.64+i*.12,8,-90,-18);
 for(let i=0;i<7;i++)frame('run-turn-'+i,-34.64+i*.25,2.4,-110+i*5,-12,.82);
}else if(process.env.QA_R14){
 frame('opening',-34,6.8,-94,-4.2);
 const uw=(name,z,yaw,pitch,y=-.52)=>frame(name,-34,z,yaw,pitch,y-beach.groundAt(-34,z));
 uw('window',-7,0,65);
 for(let k=0;k<13;k++){
  uw('shore-'+String(k).padStart(2,'0'),-4.2,180,5,-.52);
  if([4,6,8,10].includes(k)){
   uw('side-'+k,-2.1,100,4,beach.groundAt(-34,-2.1)+.13);
   uw('shallow-'+k,-.5,180,1,beach.groundAt(-34,-.5)+.045);
  }
  if(k===6||k===8){
   plume.uniforms.uShoreChurn.value=0;
   uw('tail-off-'+k,-4.2,180,5,-.52);
   plume.uniforms.uShoreChurn.value=1;
  }
  for(let i=0;i<78;i++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
 }
}else if(process.env.QA_RUNUP){
 frame('opening',-34,6.8,-94,-4.2);
 for(let k=0;k<(process.env.QA_RUNUP_REVIEW?10:18);k++){
  if(!process.env.QA_RUNUP_REVIEW||[0,6,9].includes(k))frame('runup-'+String(k).padStart(2,'0'),-34,6.8,0,-32);
  if((process.env.QA_RUNUP_REVIEW?[6]:[4,8,12]).includes(k))frame('runup-side-'+k,-34,4.6,-90,-24);
  if(k===4){frame('under-crash',-34,-4.2,180,12,-.62-beach.groundAt(-34,-4.2));frame('under-up',-34,-7,0,65,-.52-beach.groundAt(-34,-7));}
  for(let i=0;i<78;i++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
 }
}else if(process.env.QA_R13){
 frame('opening',-34,6.8,-94,-4.2);
 // Hold camera/time fixed and cross only the terrain-window boundary. Any
 // image change in these pairs is a recentering artifact, not camera motion.
 for(const z of [2.4,5.,8.,12.,20.])for(const side of [-1,1])
  frame('snap-z'+z+'-'+(side<0?'a':'b'),-34,z,-90,-20,1.64,-34+side*.00002);
 for(const z of [8.,12.])for(let i=0;i<11;i++)frame('walk-z'+z+'-'+String(i).padStart(2,'0'),-34.5+i*.1,z,-90,-18);
 if(!process.env.QA_QUICK){
  for(let i=0;i<11;i++)frame('run-'+String(i).padStart(2,'0'),-35.2+i*.22,8,-90,-12);
  for(let i=0;i<7;i++)frame('reverse-'+i,-33.8-i*.2,8,90,-20);
 }
 // Look straight down and along the film edge during an uprush/backwash cycle.
 for(let k=0;k<9;k++){
  frame('shallows-'+k,-34,1.3,0,-65,.82);
  if(k===3||k===6)frame('film-side-'+k,-34,1.7,-90,-35,.82);
  for(let i=0;i<54;i++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
 }
} else if(process.env.QA_SAND){
 for(const z of (process.env.QA_QUICK?[4.8,5.2]:[2.4,3.8,4.8,5.2,6.8,8,10]))frame('sand-z'+z,-34,z,90,-45);
 for(let i=0;i<9;i++){if(process.env.QA_QUICK&&![0,5,8].includes(i))continue;frame('walk-'+i,-34+i*.2,4.8,-90,-22);}
} else if(process.env.QA_R12){
 frame('opening',-34,6.8,-94,-4.2);
 const uw=(name,x,z,yaw,pitch,y=-.52)=>frame(name,x,z,yaw,pitch,y-beach.groundAt(x,z));
 uw('window',-34,-7,0,65);
 uw('bed',-34,-7,0,-38);
 for(let k=0;k<13;k++){
  uw('far-'+String(k).padStart(2,'0'),-34,-8,180,0,-.62);
  if(k===4||k===8){
   uw('close-'+k,-34,-4.2,180,12,-.62);
   uw('side-'+k,-35,-3.8,105,12,-.55);
   uw('up-'+k,-34,-3.7,180,66,-.65);
  }
  if(k===6)uw('inside',-34,-2.6,180,10,beach.groundAt(-34,-2.6)+.22);
  for(let i=0;i<54;i++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
 }
 for(const y of [.55,.1,-.1,-.35])uw('line-'+y,-34,-9,0,0,y);
} else if(process.env.QA_R11){
 frame('opening',-34,6.8,-94,-4.2);
 const uw=(name,x,z,yaw,pitch,y=-.52)=>frame(name,x,z,yaw,pitch,y-beach.groundAt(x,z));
 uw('window',-34,-7,0,65);
 uw('bed',-34,-7,0,-38);
 for(let k=0;k<13;k++){
  uw('crash-'+String(k).padStart(2,'0'),-34,-4.2,180,12,-.62);
  if(k===4){uw('crash-side',-35,-3.8,105,12,-.55);uw('crash-up',-34,-3.7,180,66,-.65);}
  if(k===6){uw('inside',-34,-2.6,180,10,beach.groundAt(-34,-2.6)+.22);}
  for(let i=0;i<54;i++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
 }
 for(const y of [.55,.1,-.1,-.35])uw('line-'+y,-34,-9,0,0,y);
} else if(process.env.QA_R10){
 frame('opening',-34,6.8,-94,-4.2);
 const underwater=(name,z,yaw,pitch,y=-.52)=>frame(name,-34,z,yaw,pitch,y-beach.groundAt(-34,z));
 underwater('bed',-7,0,-38);
 underwater('window',-7,0,65);
 underwater('ceiling',-7,90,18);
 underwater('shore',-5,180,6,-.35);
 for(let k=0;k<7;k++){
  underwater('crash-'+k,-4.2,180,12,-.62);
  for(let i=0;i<54;i++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
 }
 for(const y of [.65,.55,.51,.49,.47,.43,.35,0,-.35])underwater('line-'+y,-9,0,0,y);
} else if(process.env.QA_R9){
 frame('opening',-34,6.8,-94,-4.2);
 // Freeze time and cross an actual terrain-window snap in 2.5 mm steps.
 for(let i=0;i<=12;i++)frame('sand-'+String(i).padStart(2,'0'),-34.015+i*.0025,8,90,-42);
 frame('rear',-34,-4,180,-10);
 const trace=process.env.QA_SWIM_TRACE?JSON.parse(fs.readFileSync(process.env.QA_SWIM_TRACE,'utf8')):null;
 for(let k=0;k<120;k++){
  shared.uFocus.value.set(-34,-10);pack(t);
  swell.update(r,t);water.ocean.update(r,t);
  cam.position.set(-34,.34,-10);
  probe.uniforms.uProbe.value.set(-34,-10);probe.pass.camera.position.copy(cam.position);
  probe.pass.render(r,probe.target);
  D.commands.push({probe:probe.target.texture.uuid,time:t,x:-34,z:-10});
  if(trace&&k%30===0){const y=trace[k].cameraY??trace[k].samples[0]+.34;frame('swim-'+k,-34,-10,k===60?180:0,-3,y-beach.groundAt(-34,-10));}
  for(let i=0;i<4;i++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
 }
} else if(process.env.QA_R8){
 frame('opening',-34,6.8,-94,-4.2);
 frame('rocks',-37.3,1.1,-50,-24,.82);
 frame('rocks-close',-35.8,.75,-22,-37,.82);
 frame('rear',-34,-4,180,-10);
 frame('side',-34,-4,-110,-5);
 frame('side-low',-34,-4,-110,-3,1.20);
 frame('crash-0',-34,4.2,0,-5,.82);
 for(let k=1;k<=4;k++){
  for(let i=0;i<48;i++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
  frame('crash-'+k,-34,4.2,0,-5,.82);
 }
 frame('rocks-wash',-37.3,1.1,-50,-24,.82);
 frame('side-later',-34,-4,-110,-5);
} else if(process.env.QA_REAR){
 frame('rear',-34,-4,180,-10);
 frame('side',-34,-4,-110,-5);
 frame('side-low',-34,-4,-110,-3,1.20);
 frame('front',-34,2.4,0,-7,.82);
 for(let i=0;i<96;i++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
 frame('side-later',-34,-4,-110,-5);
} else if(process.env.QA_REVIEW_LIP){
 frame('lip-oblique',-34,.8,-35,-14);
 frame('lip-low',-34,.8,-35,-7,.82);
 frame('lip-front',-34,4.2,0,-5,.82);
} else if(process.env.QA_MOVIE){
 const frames=+(process.env.QA_FRAMES||72);
 post.frameDt=1/24;
 for(let i=0;i<frames;i++){
  frame('crash-'+String(i).padStart(3,'0'),-34,4.2,0,-5,.82);
  for(let j=0;j<5;j++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
 }
} else {
 frame('opening',-34,6.8,-94,-4.2);
 frame('promenade',-34,29,-165,6);
 frame('standing',-34,1.6,0,-1);
 frame('crouch',-34,1.6,0,-1,.82);
 for(let i=0;i<60;i++){pack(t);swe.step(t,1/120);ww.step(t,1/120);t+=1/120;}
 frame('breaking',-34,1.6,0,-1,.82);
 frame('wading',-34,-3.8,0,-2);
 frame('wading-low',-34,-3.8,0,-2,Math.max(.82,.32-beach.groundAt(-34,-3.8)));
}
// Stream the recording: camera sweeps can exceed V8's single-string limit.
const fd=fs.openSync(process.env.QA_OUTPUT||'.qa/world.json','w');
fs.writeSync(fd,'{');let first=true;
for(const [key,value] of Object.entries(D)){
 if(!first)fs.writeSync(fd,',');first=false;fs.writeSync(fd,JSON.stringify(key)+':');
 if(value&&typeof value==='object'){
  const array=Array.isArray(value);fs.writeSync(fd,array?'[':'{');let itemFirst=true;
  for(const [k,v]of Object.entries(value)){if(!itemFirst)fs.writeSync(fd,',');itemFirst=false;fs.writeSync(fd,(array?'':JSON.stringify(k)+':')+JSON.stringify(v));}
  fs.writeSync(fd,array?']':'}');
 }else fs.writeSync(fd,JSON.stringify(value));
}
fs.writeSync(fd,'}');fs.closeSync(fd);
console.log({width:W,height:H,programs:matMap.size,commands:D.commands.length,draws:D.draws.length,uniforms:D.uniforms.length,architecture:beach.backdrop.seafront.group.userData.instances,rockTriangles:beach.rocks.mesh.geometry.index.count/3});
