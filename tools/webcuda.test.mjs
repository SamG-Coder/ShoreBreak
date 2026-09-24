import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve,dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url));

test('active browser module graph has no Three.js, WebGL, or handwritten shaders',async()=>{
 const seen=new Set();
 async function visit(path){
  if(seen.has(path))return;seen.add(path);const text=await readFile(path,'utf8');
  assert(!/from\s+['"]three|WebGLRenderer|createRenderPipeline|createShaderModule\s*\(/.test(text),path);
  for(const m of text.matchAll(/(?:from\s*|import\s*\()(['"])(\.[^'"]+\.js)\1/g)){
   const next=resolve(dirname(path),m[2]);
   // WebCuda's runtime is the only permitted shader-module creation boundary.
   if(next.includes('vendor'))continue;await visit(next);
  }
 }
 await visit(resolve(root,'src/boot.js'));assert(seen.has(resolve(root,'src/webcuda/engine.js')));
});

test('all generated CUDA artifacts match the manifest and portable GPU budgets',async()=>{
 const manifest=JSON.parse(await readFile(resolve(root,'public/generated/manifest.json'),'utf8'));
 assert.equal(manifest.kernels.length,14);
 for(const kernel of manifest.kernels){
  const text=await readFile(resolve(root,`public/generated/${kernel.entry}.json`),'utf8'),artifact=JSON.parse(text);
  assert.equal(createHash('sha256').update(text).digest('hex'),kernel.sha256);
  assert(artifact.wgsl.includes('@compute'));assert(artifact.metadata.bindings.length<=8);
  assert(artifact.metadata.workgroupStorageBytes<=16384);assert.deepEqual(artifact.metadata.workgroupSize,kernel.workgroupSize);
 }
});

test('converted original geometry has bounded, forward-only BVH links and valid triangles',async()=>{
 const bytes=await readFile(resolve(root,'public/assets/webcuda/scene.bin'));
 const data=new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
 assert(data.every(Number.isFinite));const triBase=data[3],nodeCount=(triBase-1)/3,triCount=(data.length/4-triBase)/4;
 assert(Number.isInteger(nodeCount)&&Number.isInteger(triCount));assert(triCount>300000);
 for(let i=0;i<nodeCount;i++){
  const o=(1+i*3)*4,escape=data[o+3],start=data[o+7],count=data[o+8];
  assert(escape>i&&escape<=nodeCount);assert(count>=0&&count<=6);assert(start+count<=triCount);
  for(let a=0;a<3;a++)assert(data[o+a]<=data[o+4+a]);
 }
});
