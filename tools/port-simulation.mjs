// Mechanical lowering of the upstream FFT and KP flux passes into ShoreBreak.cu.
import {readFile,writeFile} from 'node:fs/promises';
import {originalShaders} from './original-shaders.mjs';
globalThis.location={search:'?explore'};
const ocean=await originalShaders('src/water/OceanFFT.js',['SPECTRUM','FFT','COMPOSE']);
const swash=await originalShaders('src/swash/SwashSim.js',['FLUX']);
const base=(await readFile('ShoreBreak.cu','utf8')).split('__device__ float2 sb_cmul')[0].split('// ORIGINAL SIMULATION PASSES')[0];
export function lower(s){
 s=s.replace(/\/\*[\s\S]*?\*\//g,'').replace(/\/\/[^\n]*/g,'').replace(/precision[^;]+;/g,'');
 s=s.replace(/struct\s+\w+\s*\{[^}]*\}/g,s=>s.replace(/\bfloat\s+([\w,\s]+);/g,(_,n)=>n.split(',').map(x=>`float ${x.trim()};`).join(' ')));
 s=s.replace(/\b(vec[234]|ivec2)\s*\(/g,(_,t)=>t==='ivec2'?'sb_vec2(':'sb_'+t+'(').replace(/\bivec2\b/g,'float2').replace(/\bvec([234])\b/g,'float$1');
 s=s.replace(/\b(out|inout)\s+(\w+)\s+(\w+)/g,'$2& $3');
 for(const n of ['sin','cos','exp','log','sqrt','abs','floor','fract','sign','tanh','min','max','clamp','mix','step','smoothstep','mod'])s=s.replace(new RegExp(`\\b${n}\\s*\\(`,'g'),`sb_${n}(`);
 s=s.replace(/\bpow\(/g,'powf(').replace(/(?<![\w.])(\d+\.\d*|\.\d+|\d+[eE][+-]?\d+)([eE][+-]?\d+)?(?![\w.])/g,'$1$2f');
 return s;
}
let extra=`\n// ORIGINAL SIMULATION PASSES — src/water/OceanFFT.js and src/swash/SwashSim.js\n`;
extra+=`__device__ float2 sb_zw(float4 a){return make_float2(a.z,a.w);}
__device__ float3 sb_xyz(float4 a){return make_float3(a.x,a.y,a.z);}
__device__ float4 sb_fetch(const float4* data,float2 p,int width){return data[(int)p.y*width+(int)p.x];}
`;
const declarations=[];
for(const [entry,shader,params,width,height] of [
 ['originalSpectrum',ocean.SPECTRUM,'const float4* uH0,float4* output0,float4* output1,float uT,float L0,float L1,float L2',768,256],
 ['originalFFT',ocean.FFT,'const float4* uIn0,const float4* uIn1,float4* output0,float4* output1,int uSub,int uHoriz',768,256],
 ['originalCompose',ocean.COMPOSE,'const float4* uIn0,const float4* uIn1,float4* output0,float4* output1,int uCascade,float uChopK',256,256],
]){
 let s=shader.replace(/uniform[^;]+;/g,'').replace(/layout\([^)]*\)\s*out[^;]+;/g,'');
 s=s.replace(/vec2 cmul[^\n]+\n/,'').replace(/\bcmul\(/g,'sb_cmul(');
 s=s.replace('void main() {',`__global__ void ${entry}(${params}) {\nint px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=${width}||py>=${height})return;int index=py*${width}+px;`);
 s=s.replace('ivec2 p = ivec2(gl_FragCoord.xy);','').replaceAll('p.x','px').replaceAll('p.y','py');
 s=s.replace(/texelFetch\((\w+),\s*p,\s*0\)/g,'$1[index]').replace(/texelFetch\((\w+),\s*(\w+),\s*0\)/g,'sb_fetch($1,$2,768)');
 s=s.replaceAll('uL.x','L0').replaceAll('uL.y','L1').replaceAll('uL.z','L2');
 s=s.replace(/\bgl_FragColor\b/g,'output0[index]').replace(/\b(out1|outS)\b/g,'output1[index]');
 s=lower(s);
 s=s.replace(/\b(\w+)\.(xy|zw)\b/g,'sb_$2($1)');
 extra+=s+'\n';declarations.push({entry,block:[8,8,1],width,height,original:shader});
}
extra=extra.replace('// ORIGINAL SIMULATION PASSES',`__device__ float2 sb_cmul(float2 a,float2 b){return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}\n// ORIGINAL SIMULATION PASSES`);
// KP uses integer grid coordinates; float2 stores those exact integers for WebCuda's vector arithmetic.
let flux=swash.FLUX.replace(/uniform[^;]+;/g,'');
const ctx='const float4* uState,const float4* uBed,float4 uDom,float2 uN,float uDt';
const args='uState,uBed,uDom,uN,uDt';
const names=[...flux.matchAll(/\b(?:float|ivec2|vec[234]|void|Face)\s+(\w+)\s*\([^)]*\)\s*\{/g)].map(m=>m[1]).filter(n=>n!=='main');
for(const name of names)flux=flux.replace(new RegExp(`\\b${name}\\s*\\(`,'g'),`kp_${name}(${args},`);
flux=flux.replace(/\b(float|ivec2|vec[234]|void|Face)\s+(kp_\w+)\([^,]+,[^,]+,[^,]+,[^,]+,[^,]+,/g,`__device__ $1 $2(${ctx},`);
// Balanced texture-call substitution keeps nested ghost-coordinate expressions intact.
flux=flux.replace(/texelFetch\((uState|uBed),\s*(kp_ghost\([^;]*?\)),\s*0\)\.xyz/g,'sb_xyz(sb_fetch($1,$2,(int)uN.x))');
flux=flux.replace('ivec2 p = ivec2(gl_FragCoord.xy);','ivec2 p = ivec2(px,py);');
flux=flux.replace('if (sp > VMAX) U.yz *= VMAX / sp;','if (sp > VMAX) { U.y *= VMAX / sp; U.z *= VMAX / sp; }');
flux=flux.replaceAll('U.yz','vec2(U.y,U.z)');
flux=flux.replace('void main() {',`__global__ void originalFlux(const float4* uState,const float4* uBed,float4* output0,float domX,float domZ,float dx,float dz,int nx,int nz,float uDt){\nint px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=nx||py>=nz)return;int index=py*nx+px;float4 uDom=make_float4(domX,domZ,dx,dz);float2 uN=make_float2(nx,nz);`);
flux=flux.replaceAll('gl_FragColor','output0[index]').replace(/\bconst float (THETA|EPS4|VMAX)/g,'__constant__ float $1');
extra+=lower(flux);
declarations.push({entry:'originalFlux',block:[8,8,1],original:swash.FLUX});
await writeFile('ShoreBreak.cu',base+extra);
await writeFile('.qa/faithful/original-passes.json',JSON.stringify(declarations));
console.log('Appended original spectrum, Stockham FFT, compose and KP flux passes');
