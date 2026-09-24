import test from 'node:test';import assert from 'node:assert/strict';import {readFile,readdir} from 'node:fs/promises';import {createHash} from 'node:crypto';import {graphicsStage} from '../src/webcuda/graphics-stage.js';import {contextLayout,packContext} from '../src/webcuda/graphics-layout.js';
test('the canonical CUDA translation unit contains compute and original graphics stages',async()=>{
 const source=await readFile('ShoreBreak.cu','utf8');assert(source.includes('__global__ void originalFlux'));assert(source.includes('#ifdef SB_GRAPHICS_55_fragment'));assert(source.includes('sheetRaw'));
 const root=(await readdir('.')).filter(x=>x.endsWith('.cu'));assert.deepEqual(root,['ShoreBreak.cu']);
 const metadata=JSON.parse(await readFile('tools/graphics-metadata.json','utf8'));for(const id of Object.keys(metadata.stages))for(const stage of ['vertex','fragment'])assert(source.includes('#ifdef SB_GRAPHICS_'+id+'_'+stage));
});
test('graphics adapters retain native derivatives and per-invocation state',async()=>{
 const a=JSON.parse(await readFile('public/faithful/graphics/55-fragment.json','utf8')),v=JSON.parse(await readFile('public/faithful/graphics/55-vertex.json','utf8')),stage=graphicsStage(a,'fragment',v.outputs);
 assert(stage.wgsl.includes('@fragment fn fragmentMain'));assert(stage.wgsl.includes('dpdx('));assert(stage.wgsl.includes('dpdy('));assert(stage.wgsl.includes('var<private> sb_context:'));assert(!stage.wgsl.includes('@compute'));assert(!stage.wgsl.includes('var<private> b_input'));assert(stage.textureBindings.length<=16);
});
test('host uniforms use the compiled CUDA record layout, including matrices and arrays',()=>{
 const wgsl='struct M {\n cw_field_c0: vec3<f32>,\n cw_field_c1: vec3<f32>,\n cw_field_c2: vec3<f32>,\n}\nstruct cw_struct_GraphicsContext {\n cw_field_n: i32,\n cw_field_x: vec3<f32>,\n cw_field_m: M,\n cw_field_events: array<vec4<f32>, 2>,\n}';
 const layout=contextLayout(wgsl),view=new DataView(packContext(layout,{n:7,x:[1,2,3],m:[1,2,3,4,5,6,7,8,9],events:[10,11,12,13,14,15,16,17]}));assert.equal(layout.size,112);assert.equal(view.getInt32(0,true),7);assert.equal(view.getFloat32(16,true),1);assert.equal(view.getFloat32(48,true),4);assert.equal(view.getFloat32(108,true),17);
});
test('published graphics artifacts are built from the current CUDA source',async()=>{
 const build=JSON.parse(await readFile('public/faithful/graphics/build.json','utf8')),source=await readFile('ShoreBreak.cu','utf8');assert.equal(build.sha256,createHash('sha256').update(source).digest('hex'));
 for(const record of build.records){const contents=await readFile(`public/faithful/graphics/${record.id}-${record.stage}.json`,'utf8');assert.equal(record.sha256,createHash('sha256').update(contents).digest('hex'));}
});
test('original vector-object uniform arrays pack identically to flat numeric arrays',()=>{
 const layout=contextLayout('struct cw_struct_GraphicsContext {\n cw_field_layers: array<vec4<f32>, 2>,\n}');
 const original=[{toArray:()=>[.0042,11,1,.95]},{toArray:()=>[.009,23,2,.97]}];
 assert.deepEqual(new Uint8Array(packContext(layout,{layers:original})),new Uint8Array(packContext(layout,{layers:[.0042,11,1,.95,.009,23,2,.97]})));
});
