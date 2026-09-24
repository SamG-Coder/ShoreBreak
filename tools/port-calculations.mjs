// Mechanical GLSL -> CUDA source port. No numerical formula rewriting.
// This tool authors the single ShoreBreak.cu file; the build only compiles that file.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {originalShaders} from './original-shaders.mjs';
globalThis.location={search:'?explore'};
const {glslDefines}=await import('../src/config.js');
const {NOISE}=await import('../src/glsl/common.js');
const {BREAKER}=await import('../src/glsl/breaker.js');
const strip=s=>s.replace(/\/\*[\s\S]*?\*\//g,'').replace(/\/\/[^\n]*/g,'');
let prelude=`// ShoreBreak calculations ported to CUDA for SamG-Coder's WebCuda.\n// Original calculations: Christopher Canavan, MIT; see LICENSE.\n// This is the ONLY CUDA source file for the faithful port.\n// GLSL vector operations are expressed in the CUDA subset supported by WebCuda.\n\n`;
const components='xyzw';
for(const n of [2,3,4]){
 const t=`float${n}`,names=[...components.slice(0,n)];
 prelude+=`__device__ ${t} sb_vec${n}(${names.map(x=>'float '+x).join(',')}) { return make_float${n}(${names.join(',')}); }\n`;
 prelude+=`__device__ ${t} sb_vec${n}(float x) { return make_float${n}(${names.map(()=>'x').join(',')}); }\n`;
 prelude+=`__device__ ${t} sb_vec${n}(${t} x) { return x; }\n`;
}
prelude+=`__device__ float3 sb_vec3(float2 a,float b){return make_float3(a.x,a.y,b);}
__device__ float3 sb_vec3(float a,float2 b){return make_float3(a,b.x,b.y);}
__device__ float4 sb_vec4(float2 a,float2 b){return make_float4(a.x,a.y,b.x,b.y);}
__device__ float4 sb_vec4(float3 a,float b){return make_float4(a.x,a.y,a.z,b);}
__device__ float4 sb_vec4(float2 a,float b,float c){return make_float4(a.x,a.y,b,c);}
__device__ float4 sb_vec4(float a,float b,float2 c){return make_float4(a,b,c.x,c.y);}
__device__ float4 sb_vec4(float a,float2 b,float c){return make_float4(a,b.x,b.y,c);}
__device__ float2 sb_xy(float4 a){return make_float2(a.x,a.y);}
`;
const unary={sin:'sinf(x)',cos:'cosf(x)',exp:'expf(x)',log:'logf(x)',sqrt:'sqrtf(x)',abs:'fabsf(x)',floor:'floorf(x)',fract:'x-floorf(x)',sign:'x>0.0f?1.0f:(x<0.0f?-1.0f:0.0f)',tanh:'(x<0.0f?-1.0f:1.0f)*(1.0f-expf(-2.0f*fabsf(x)))/(1.0f+expf(-2.0f*fabsf(x)))'};
for(const [name,expr] of Object.entries(unary)){
 prelude+=`__device__ float sb_${name}(float x){return ${expr};}\n`;
 for(const n of [2,3,4])prelude+=`__device__ float${n} sb_${name}(float${n} x){return make_float${n}(${[...components.slice(0,n)].map(c=>`sb_${name}(x.${c})`).join(',')});}\n`;
}
prelude+=`__device__ int sb_min(int a,int b){return min(a,b);}
__device__ int sb_max(int a,int b){return max(a,b);}
__device__ float sb_min(float a,float b){return fminf(a,b);}
__device__ float sb_max(float a,float b){return fmaxf(a,b);}
__device__ float sb_clamp(float x,float a,float b){return fminf(fmaxf(x,a),b);}
__device__ float sb_mix(float a,float b,float t){return a*(1.0f-t)+b*t;}
__device__ float sb_step(float edge,float x){return x<edge?0.0f:1.0f;}
__device__ float sb_smoothstep(float a,float b,float x){float t=sb_clamp((x-a)/(b-a),0.0f,1.0f);return t*t*(3.0f-2.0f*t);}
__device__ float sb_mod(float x,float y){return x-y*floorf(x/y);}
`;
for(const n of [2,3,4])for(const name of ['min','max','clamp','mix','step','smoothstep','mod']){
 const arity=['clamp','mix','smoothstep'].includes(name)?3:2,vars=['a','b','c'].slice(0,arity);
 const signatures=[vars.map(()=>`float${n}`)];
 if(name==='mix')signatures.push([`float${n}`,`float${n}`,'float']);
 if(name==='clamp')signatures.push([`float${n}`,'float','float']);
 if(name==='smoothstep')signatures.push(['float','float',`float${n}`]);
 if(arity===2)signatures.push([`float${n}`,'float'],['float',`float${n}`]);
 for(const types of signatures)prelude+=`__device__ float${n} sb_${name}(${vars.map((v,i)=>types[i]+' '+v).join(',')}) {return make_float${n}(${[...components.slice(0,n)].map(c=>`sb_${name}(${vars.map((v,i)=>v+(types[i]==='float'?'':'.'+c)).join(',')})`).join(',')});}\n`;
}
prelude+=`__device__ float4 sb_lookup(const float4* lookup,float2 uv,float lod){
 float px=sb_clamp(uv.x*4096.0f-0.5f,0.0f,4095.0f),py=sb_clamp(uv.y*4.0f-0.5f,0.0f,3.0f);
 int x=(int)floorf(px),y=(int)floorf(py),x1=min(x+1,4095),y1=min(y+1,3);
 float fx=px-(float)x,fy=py-(float)y;
 return sb_mix(sb_mix(lookup[y*4096+x],lookup[y*4096+x1],fx),sb_mix(lookup[y1*4096+x],lookup[y1*4096+x1],fx),fy);
}
`;
const {SHEET}=await originalShaders('src/water/LipRibbon.js',['SHEET']);
let sheet=SHEET.slice(0,SHEET.indexOf('vec3 sheetResidual'));
const culled=sheet.indexOf('bool lipCulled');let brace=sheet.indexOf('{',culled)+1,depth=1;
for(;depth;brace++){if(sheet[brace]==='{')depth++;if(sheet[brace]==='}')depth--;}
// View-frustum culling belongs to the renderer; preserve every cross-section equation.
sheet=sheet.slice(0,culled)+sheet.slice(brace);
const original=glslDefines()+NOISE+BREAKER+sheet;
let code=strip(original);
code=code.replace(/^#define\s+(\w+)[ \t]*$/gm,'#define $1 1');
code=code.replace(/struct\s+\w+\s*\{[^}]*\}/g,s=>s.replace(/\b(float|vec[234]|int)\s+([\w,\s]+);/g,(_,t,n)=>n.split(',').map(x=>`${t} ${x.trim()};`).join(' ')));
code=code.replace(/\b(\w+\.\w+)\s*([+*\/-])=\s*([^;]+);/g,'$1 = $1 $2 ($3);');
let swizzleId=0;
code=code.replace(/\b(\w+)\.([xyzw]{2,4})\s*=\s*([^;]+);/g,(_,v,sw,rhs)=>{const t='lipSwizzle'+swizzleId++;return `{vec${sw.length} ${t}=${rhs};${[...sw].map((c,i)=>`${v}.${c}=${t}.${'xyzw'[i]};`).join('')}}`;});
// GLSL's column-major 2x2 multiply, unchanged arithmetic.
code=code.replace('mat2(1.6, 1.2, -1.2, 1.6) * p','vec2(1.6*p.x-1.2*p.y,1.2*p.x+1.6*p.y)');
const fields=[];
code=code.replace(/uniform\s+(\w+)\s+(\w+)(\[[^\]]+\])?\s*;/g,(_,type,name,array)=>{
 if(name==='uWaterLookup'||name.startsWith('uEvt')&&array)return '';
 fields.push({type,name});return '';
});
let structs='struct SBUniforms {\n'+fields.map(f=>`  ${f.type.replace('vec','float')} ${f.name};`).join('\n')+'\n};\n';
for(const {name} of fields)code=code.replace(new RegExp(`\\b${name}\\b`,'g'),`sb.${name}`);
for(const [i,letter] of [...'ABCDEFG'].entries())code=code.replace(new RegExp(`\\buEvt${letter}\\[([^\\]]+)\\]`,'g'),`events[${i}*MAX_EVENTS+($1)]`);
code=code.replace(/(events\[[^\]]+\])\.xy\b/g,'sb_xy($1)');
code=code.replace(/textureLod\(uWaterLookup\s*,/g,'sb_lookup(lookup,');
const functions=[...code.matchAll(/\b(?:float|int|void|vec[234]|Brk\w*)\s+(\w+)\s*\([^)]*\)\s*\{/g)].map(m=>m[1]);
for(const name of functions)code=code.replace(new RegExp(`\\b${name}\\s*\\(`,'g'),`${name}(events,lookup,sb,`);
code=code.replace(/\b(float|int|void|vec[234]|Brk\w*)\s+(\w+)\(events,lookup,sb,/g,'__device__ $1 $2(const float4* events,const float4* lookup,SBUniforms sb,');
code=code.replace(/,\s*\)/g,')').replace(/\b(out|inout)\s+(\w+)\s+(\w+)/g,'$2& $3');
const functionsMap=Object.fromEntries([...Object.keys(unary), 'min','max','clamp','mix','step','smoothstep','mod'].map(x=>[x,'sb_'+x]));
functionsMap.pow='powf';
for(const [from,to] of Object.entries(functionsMap))code=code.replace(new RegExp(`\\b${from}\\s*\\(`,'g'),to+'(');
code=code.replace(/\bvec([234])\s*\(/g,'sb_vec$1(').replace(/\bvec([234])\b/g,'float$1');
code=code.replace(/(?<![\w.])(\d+\.\d*|\.\d+|\d+[eE][+-]?\d+)([eE][+-]?\d+)?(?![\w.])/g,'$1$2f');
const kernel=`
// Batched original-wave evaluation. queries=(x,z,time,sigma), 12 float4 outputs/query.
__global__ void evaluateOriginalWaves(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* queries,float4* results,unsigned int count,int eventIndex){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 SBUniforms sb=uniforms[0];float4 q=queries[i];float2 xz=make_float2(q.x,q.y),d;float foam;
 Brk b=brkAt(events,lookup,sb,eventIndex,q.x);float tn=brkTn(events,lookup,sb,b,q.z);BrkStage stage=brkStage(events,lookup,sb,tn);
 BrkLip lip=brkLip(events,lookup,sb,b,q.z);float4 curve=brkLipCurve(events,lookup,sb,lip,q.w);
 float3 profile=brkProfile(events,lookup,sb,b,q.y,q.z);float eta=brkSurface(events,lookup,sb,xz,q.z,d,foam);
 float3 injection=brkInjection(events,lookup,sb,xz,q.z);
 results[i*12]=make_float4(b.ti,b.D,b.H,b.zI);
 results[i*12+1]=make_float4(tn,stage.za,stage.ya,stage.yt);
 results[i*12+2]=make_float4(stage.lf,stage.uw,stage.hw,stage.lb);
 results[i*12+3]=make_float4(profile.x,profile.y,profile.z,eta);
 results[i*12+4]=make_float4(d.x,d.y,foam,brkWhitecap(events,lookup,sb,b,q.z));
 results[i*12+5]=make_float4(lip.R.x,lip.R.y,lip.T.x,lip.T.y);
 results[i*12+6]=make_float4(lip.P1.x,lip.P1.y,lip.P2.x,lip.P2.y);
 results[i*12+7]=curve;
 results[i*12+8]=make_float4(injection.x,injection.y,injection.z,brkFoam(events,lookup,sb,xz,q.z));
 results[i*12+9]=make_float4(brkImpactSpeed(events,lookup,sb,b),brkLipWidth(events,lookup,sb,lip,q.w),brkSweFoamKeep(events,lookup,sb,xz,q.z),brkCrestSpeed(events,lookup,sb,b,q.z));
 results[i*12+10]=swellTravel(events,lookup,sb,q.y);
 results[i*12+11]=make_float4(stage.m,stage.p,stage.st,brkCrestWidth(events,lookup,sb,xz,profile.x*.75f,q.z));
}
__global__ void evaluateOriginalLip(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* queries,float4* results,unsigned int count,int eventIndex){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 SBUniforms sb=uniforms[0];float4 q=queries[i];Brk b=brkAt(events,lookup,sb,eventIndex,q.x);float4 meta,meta2;float2 knee;
 float2 point=sheetRaw(events,lookup,sb,b,q.w,q.z,meta,meta2,knee);
 results[i*3]=make_float4(point.x,point.y,knee.x,knee.y);results[i*3+1]=meta;results[i*3+2]=meta2;
}
`;
await writeFile(new URL('../ShoreBreak.cu',import.meta.url),prelude+structs+'\n// Direct port of glslDefines + NOISE + BREAKER (including SWELL_TRANSPORT).\n'+code+kernel);
await mkdir(new URL('../.qa/faithful/',import.meta.url),{recursive:true});
await writeFile(new URL('../.qa/faithful/original-waves.json',import.meta.url),JSON.stringify({glsl:original,fields,functions,sha256:createHash('sha256').update(original).digest('hex')}));
console.log(`Ported ${functions.length} original functions to ShoreBreak.cu`);
