import {readFile,writeFile} from 'node:fs/promises';
import {portPasses} from './glsl-pass-port.mjs';
globalThis.location={search:'?explore'};
const {glslDefines}=await import('../src/config.js'),{NOISE}=await import('../src/glsl/common.js'),{BREAKER}=await import('../src/glsl/breaker.js'),{CHOP,SWE_SAMPLE,WATER_GEOM}=await import('../src/glsl/water.js'),{SWASH_FAR_CORE}=await import('../src/glsl/swashfar.js');
let base=(await readFile('ShoreBreak.cu','utf8')).split('// ORIGINAL WATER GEOMETRY')[0];
const wave=JSON.parse(await readFile('.qa/faithful/original-waves.json','utf8'));wave.functions=[...base.matchAll(/__device__\s+\w+\s+(\w+)\(const float4\* events,const float4\* lookup,SBUniforms sb/g)].map(m=>m[1]).filter(n=>!n.startsWith('original'));
let shader=CHOP+SWE_SAMPLE+SWASH_FAR_CORE+WATER_GEOM.slice(0,WATER_GEOM.indexOf('vec4 crestGeometry'));
shader=shader.replace(/^#define\s+(\w+)[ \t]*$/gm,'#define $1 1');
shader=shader.replace(/precision highp sampler2DArray;/g,'').replaceAll('sampler2DArray','sampler2D');
shader=shader.replace(/uniform\s+(\w+)\s+([\w, ]+);/g,(_,t,n)=>n.split(',').map(v=>`uniform ${t} ${v.trim()};`).join('\n'));
shader=shader.replace(/const vec3 OCC2[^;]+;/,'');
for(let i=0;i<3;i++){
 const m=shader.match(new RegExp(`const mat2 OCR${i} = mat2\\(([^)]+)\\);`)),[a,b,c,d]=m[1].split(',');
 shader=shader.replace(m[0],`vec2 rot${i}(vec2 v){return vec2(${a}*v.x+${c}*v.y,${b}*v.x+${d}*v.y);}vec2 rot${i}T(vec2 v){return vec2(${a}*v.x+${b}*v.y,${c}*v.x+${d}*v.y);}`);
 shader=shader.replace(new RegExp(`OCR${i} \\* (\\w+)`,'g'),`rot${i}($1)`).replace(new RegExp(`(\\w+\\.\\w+) \\* OCR${i}`,'g'),`rot${i}T($1)`);
}
shader=shader.replace('mat2(1.6, 1.2, -1.2, 1.6) * p','vec2(1.6*p.x-1.2*p.y,1.2*p.x+1.6*p.y)').replace('float gBrkEta;','uniform float gBrkEta;');
shader='uniform vec3 cameraPosition;uniform vec4 uFieldDom;uniform vec2 uFieldSize;'+shader+`
layout(location=1)out vec4 outSlope;
void main() {
 vec2 xz=uFieldDom.xy+(gl_FragCoord.xy-.5)/(uFieldSize-1.0)*uFieldDom.zw;
 vec2 gm,gc;vec3 ch;vec4 info;float paw;
 float y=waterParts(xz,uTime,.016,gm,ch,gc,info,paw);
 vec4 far;gm+=clampLen(sweGradient(xz,1.5,far),uSweSlope.x);
 gl_FragColor=vec4(y+ch.y,info.x,info.y,gBrkEta);
 outSlope=vec4(gm+gc,ch.x,ch.z);
}`;
const result=portPasses('',{WATER_FIELD:shader},['WATER_FIELD'],wave,glslDefines,NOISE,BREAKER);let text=result.source;
text=text.replace(/,originalWaterFieldParams sbuParams/g,',originalWaterFieldParams& sbuParams').replace(/\blog2\(/g,'log2f(');
for(const name of ['uOceanD','uOceanS']){const i=result.metadata[0].textures.indexOf(name);text=text.replaceAll(`sb_sample(data,descriptors,${i},`,`sb_sampleArray(data,descriptors,${i},`);}
const used=new Set([...text.matchAll(/sb_sw_(\w+)\(/g)].map(m=>m[1]));let helpers='';for(const sw of used)if(!base.includes(` sb_sw_${sw}(`))for(let n=Math.max(2,...[...sw].map(c=>'xyzw'.indexOf(c)+1));n<=4;n++)helpers+=`__device__ float${sw.length} sb_sw_${sw}(float${n} a){return make_float${sw.length}(${[...sw].map(c=>'a.'+c).join(',')});}\n`;
helpers+=`__device__ float4 sb_arrayMip(const float4* data,int base,float2 uv,int level){
 int side=256,offset=base;for(int j=0;j<level;j++){offset+=side*side;side/=2;}
 float x=(uv.x-floorf(uv.x))*(float)side-.5f,y=(uv.y-floorf(uv.y))*(float)side-.5f;int ix=(int)floorf(x),iy=(int)floorf(y),mask=side-1;
 return sb_mix(sb_mix(data[offset+(iy&mask)*side+(ix&mask)],data[offset+(iy&mask)*side+((ix+1)&mask)],x-floorf(x)),sb_mix(data[offset+((iy+1)&mask)*side+(ix&mask)],data[offset+((iy+1)&mask)*side+((ix+1)&mask)],x-floorf(x)),y-floorf(y));}
__device__ float4 sb_sampleArray(const float4* data,const float4* descriptors,int slot,float3 uv,float lod){
 float4 d=descriptors[slot];int base=(int)d.x+(int)uv.z*87381;float l=sb_clamp(lod,0.0f,8.0f);int a=(int)floorf(l);return sb_mix(sb_arrayMip(data,base,make_float2(uv.x,uv.y),a),sb_arrayMip(data,base,make_float2(uv.x,uv.y),min(a+1,8)),l-(float)a);}
`;
await writeFile('ShoreBreak.cu',base+'\n// ORIGINAL WATER GEOMETRY\n'+helpers+text.replace(/[ \t]+$/gm,''));
await writeFile('public/faithful/water-pass.json',JSON.stringify(result.metadata[0]));
