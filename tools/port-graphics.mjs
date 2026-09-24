import {createHash} from 'node:crypto';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {graphicsPort} from './graphics-port.mjs';
import {graphicsSupport,swizzleSupport} from './graphics-support.mjs';
import {compile,serializableArtifact} from '../vendor/cuda-webshader/compiler/compiler.js';
import {shaderKey} from '../src/webcuda/graphics-layout.js';
const base=(await readFile('ShoreBreak.cu','utf8')).split('// ORIGINAL GRAPHICS STAGES')[0].replace(/^#ifndef SB_GRAPHICS\s*\n/,'').replace(/\n#endif\s*$/,'');
const materials=JSON.parse(await readFile('.qa/programs.json','utf8')).materials;
const materialKey=mat=>shaderKey({...mat,isMeshDepthMaterial:mat.depth,vertexShader:mat.vertex,fragmentShader:mat.fragment},{instanceMatrix:mat.instanced,instanceColor:mat.instanceColor});
const known=new Set(Object.values(materials).map(materialKey));
for(const [id,mat]of Object.entries(JSON.parse(await readFile('.qa/programs-clip.json','utf8')).materials))if(!known.has(materialKey(mat))){materials['clip'+id]=mat;known.add(materialKey(mat));}
const main=await readFile('src/main.js','utf8'),{CONTACT_SAMPLE}=await import('../src/post/ContactLight.js');
materials.copy={vertex:materials[0].vertex,fragment:main.match(/const copyPass = new FullscreenPass\(makeShader\(\/\* glsl \*\/ `([\s\S]*?)`,/)[1].replace('${CONTACT_SAMPLE}',CONTACT_SAMPLE),defines:{}};
materials.mip={vertex:'out vec2 vUv;void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));vUv=p;gl_Position=vec4(p*2.0-1.0,0.0,1.0);}',fragment:'uniform sampler2D uSource;in vec2 vUv;void main(){gl_FragColor=textureLod(uSource,vUv,0.0);}',defines:{}};
const manifest={},blocks=[];await mkdir('.qa/graphics',{recursive:true});
const index={};await mkdir('public/faithful/graphics',{recursive:true});
const support=graphicsSupport(base);
let cached={};try{cached=JSON.parse(await readFile('.qa/graphics/port-cache.json','utf8'));}catch{}

for(const [id,mat]of Object.entries(materials)){
 index[shaderKey({...mat,isMeshDepthMaterial:mat.depth,vertexShader:mat.vertex,fragmentShader:mat.fragment},{instanceMatrix:mat.instanced,instanceColor:mat.instanceColor})]=id;
 if(mat.depth){mat.vertex='void main(){vec4 p=vec4(position,1.0);\n#ifdef USE_INSTANCING\np=instanceMatrix*p;\n#endif\ngl_Position=projectionMatrix*modelViewMatrix*p;}';mat.fragment='void main(){gl_FragColor=vec4(1.0);}';}
 if(!mat.vertex)continue;
 manifest[id]={};
 for(const stage of ['vertex','fragment']){
  const key='SB_GRAPHICS_'+id+'_'+stage;
  try{
   const p=graphicsPort(mat[stage],stage,{...mat.defines,...(mat.instanced?{USE_INSTANCING:'1'}:{}),...(mat.instanceColor?{USE_INSTANCING_COLOR:'1'}:{})}),s=support+swizzleSupport(p.cuda)+p.cuda;
   await writeFile('.qa/graphics/current.txt',s);
   const hash=createHash('sha256').update(s).digest('hex');let a;
   if(cached[key]===hash){const old=JSON.parse(await readFile('.qa/graphics/'+id+'-'+stage+'.json','utf8'));a=old;}else a=serializableArtifact(compile(s,{entry:'graphicsStage',workgroupSize:[1,1,1]}));
   const {cuda,...metadata}=p;await writeFile('.qa/graphics/'+id+'-'+stage+'.json',JSON.stringify({...a,...metadata}));
   await writeFile('public/faithful/graphics/'+id+'-'+stage+'.json',JSON.stringify({...a,...metadata}));
   cached[key]=hash;await writeFile('.qa/graphics/port-cache.json',JSON.stringify(cached));
   manifest[id][stage]=metadata;blocks.push(`#ifdef ${key}\n${s}\n#endif\n`);console.log(id,stage,a.wgsl.length);
  }catch(e){console.error(id,stage,e.message);throw e;}
 }
}
await writeFile('ShoreBreak.cu','#ifndef SB_GRAPHICS\n'+base+'\n#endif\n// ORIGINAL GRAPHICS STAGES\n'+blocks.join('\n').replace(/[ \t]+$/gm,''));
await writeFile('.qa/graphics/manifest.json',JSON.stringify(manifest));
await writeFile('public/faithful/graphics/index.json',JSON.stringify(index));
await writeFile('tools/graphics-metadata.json',JSON.stringify({stages:manifest,index}));
