// Export unmodified reference shaders and parameter layouts. Does not write CUDA.
import {writeFile,mkdir} from 'node:fs/promises';
import {originalShaders} from './original-shaders.mjs';
globalThis.location={search:'?explore'};
const {glslDefines}=await import('../src/config.js'),{NOISE}=await import('../src/glsl/common.js'),{BREAKER}=await import('../src/glsl/breaker.js');
const {SHEET}=await originalShaders('src/water/LipRibbon.js',['SHEET']);
let sheet=SHEET.slice(0,SHEET.indexOf('vec3 sheetResidual'));const start=sheet.indexOf('bool lipCulled');let end=sheet.indexOf('{',start)+1,depth=1;for(;depth;end++){if(sheet[end]==='{')depth++;if(sheet[end]==='}')depth--;}sheet=sheet.slice(0,start)+sheet.slice(end);
const glsl=glslDefines()+NOISE+BREAKER+sheet;
const uniforms=s=>[...s.matchAll(/uniform\s+(\w+)\s+(\w+)\s*;/g)].map(m=>({type:m[1],name:m[2]}));
const fields=uniforms(glsl).filter(f=>f.type!=='sampler2D');
await mkdir('.qa/faithful',{recursive:true});await writeFile('.qa/faithful/original-waves.json',JSON.stringify({glsl,fields}));
const ocean=await originalShaders('src/water/OceanFFT.js',['SPECTRUM','FFT','COMPOSE']),swash=await originalShaders('src/swash/SwashSim.js',['FLUX','BED_INIT','STATE_INIT','SOURCES','FOAM','FOAM_VIEW','WET','WET_INIT','HEIGHT','VIEW','SHIFT','SHIFT_FOAM','FAR_PARAM','FAR_MODEL']);
await writeFile('.qa/faithful/original-passes.json',JSON.stringify([
 {entry:'originalSpectrum',width:768,height:256,original:ocean.SPECTRUM},
 {entry:'originalFFT',width:768,height:256,original:ocean.FFT},
 {entry:'originalCompose',width:256,height:256,original:ocean.COMPOSE},
 {entry:'originalFlux',original:swash.FLUX},
]));
const meta=[];for(const key of ['BED_INIT','STATE_INIT','SOURCES','FOAM','FOAM_VIEW','WET','WET_INIT','HEIGHT','VIEW','SHIFT','SHIFT_FOAM','FAR_PARAM','FAR_MODEL']){
 const original=swash[key],s=original.replace(glslDefines(),'').replace(NOISE,'').replace(BREAKER,'');
 const fs=uniforms(s);meta.push({entry:'original'+key.toLowerCase().split('_').map(x=>x[0].toUpperCase()+x.slice(1)).join(''),fields:fs.filter(f=>f.type!=='sampler2D'&&!fields.some(w=>w.name===f.name)),textures:fs.filter(f=>f.type==='sampler2D').map(f=>f.name),outputs:1+[...s.matchAll(/layout\(location\s*=\s*\d+\)\s*out/g)].length,original});
}await writeFile('.qa/faithful/swash-passes.json',JSON.stringify(meta));
