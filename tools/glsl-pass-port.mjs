import {lower} from './glsl-lower.mjs';
function calls(s,name,transform){let pos=0;for(;;){const start=s.indexOf(name+'(',pos);if(start<0)break;let end=start+name.length+1,level=1;for(;level;end++){if(s[end]==='(')level++;if(s[end]===')')level--;if(end>=s.length)throw Error('Unbalanced '+name);}
 const args=[];let current='',depth=0;for(const c of s.slice(start+name.length+1,end-1)){if(c===','&&depth===0){args.push(current.trim());current='';continue;}if(c==='('||c==='[')depth++;if(c===')'||c===']')depth--;current+=c;}args.push(current.trim());const next=transform(args);s=s.slice(0,start)+next+s.slice(end);pos=start+next.length;}return s;}

export function portPasses(source, originals, entries, wave, glslDefines, NOISE, BREAKER){
const metadata=[];
for(const key of entries){
 const entry='original'+key.toLowerCase().split('_').map(x=>x[0].toUpperCase()+x.slice(1)).join('');
 let s=originals[key].replace(glslDefines(),'').replace(NOISE,'').replace(BREAKER,'');
 s=s.replace(/\/\*[\s\S]*?\*\//g,'').replace(/\/\/[^\n]*/g,'');
 for(const macro of new Set([...s.matchAll(/^\s*#define\s+(\w+)/gm)].map(m=>m[1])))s=s.replace(new RegExp(`\\b${macro}\\b`,'g'),entry+'_'+macro);
 const fields=[],textures=[];
 s=s.replace(/uniform\s+(\w+)\s+(\w+)\s*;/g,(_,type,name)=>{if(type==='sampler2D')textures.push(name);else if(!wave.fields.some(f=>f.name===name))fields.push({type,name});return '';});
 s=s.replace(/in vec2 vUv;/g,'');
 const outputs=['gl_FragColor'];s=s.replace(/layout\(location\s*=\s*(\d+)\)\s*out vec4 (\w+);/g,(_,n,name)=>{outputs[+n]=name;return '';});
 let struct=`struct ${entry}Params {\n`+fields.map(({type,name})=>`${type==='ivec2'?'float2':type.replace('vec','float')} ${name};`).join('\n')+'\n};\n';if(!fields.length)struct=`struct ${entry}Params {float unused;};\n`;
 for(const {name} of fields)s=s.replace(new RegExp(`\\b${name}\\b`,'g'),'sbuParams.'+name);
 for(const {name} of wave.fields)s=s.replace(new RegExp(`\\b${name}\\b`,'g'),'sb.'+name);
 for(const [i,l] of [...'ABCDEFG'].entries())s=s.replace(new RegExp(`\\buEvt${l}\\[([^\\]]+)\\]`,'g'),`events[${i}*MAX_EVENTS+($1)]`);
 s=calls(s,'texelFetch',([texture,p])=>`sb_texel(data,descriptors,${textures.indexOf(texture)},${p})`);
 s=calls(s,'textureLod',([texture,uv,lod])=>`sb_sample(data,descriptors,${textures.indexOf(texture)},${uv},${lod})`);
 const args='events,lookup,sb,data,descriptors,sbuParams',ctx=`const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,${entry}Params sbuParams`;
 for(const name of wave.functions)s=s.replace(new RegExp(`\\b${name}\\s*\\(`,'g'),`${name}(events,lookup,sb,`);
 const names=[...s.matchAll(/\b(?:float|int|bool|void|vec[234]|ivec2)\s+(\w+)\s*\([^)]*\)\s*\{/g)].map(m=>m[1]).filter(n=>n!=='main');
 for(const name of names)s=s.replace(new RegExp(`\\b${name}\\s*\\(`,'g'),`${entry}_${name}(${args},`);
 s=s.replace(/\b(float|int|bool|void|vec[234]|ivec2)\s+(original\w+)\(events,lookup,sb,data,descriptors,sbuParams,/g,`__device__ $1 $2(${ctx},`);
 s=s.replace('void main() {',`__global__ void ${entry}(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const ${entry}Params* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];${entry}Params sbuParams=params[0];vec2 vUv=(vec2(px,py)+0.5)/vec2(width,height);`);
 s=s.replaceAll('gl_FragCoord.xy','vec2(px,py)');
 for(const [i,name] of outputs.entries())s=s.replace(new RegExp(`\\b${name}\\b`,'g'),`output[${i}*width*height+index]`);
 // Decompose multi-component writes, evaluating each RHS once.
 let temp=0;s=s.replace(/\b(\w+)\.([xyzwrgba]{2,4})\s*([+*/-]?)=\s*([^;]+);/g,(_,v,sw,op,rhs)=>{const t='swizzle'+temp++;return `{vec${sw.length} ${t}=${op?`${v}.${sw}${op}(${rhs})`:rhs};${[...sw].map((c,i)=>`${v}.${c}=${t}.${'xyzw'[i]};`).join('')}}`;});
 s=lower(s);
 s=s.replace(/\b(asinh|sinh)\(/g,'sb_$1(');
 s=s.replace(/\.([rgba])\b/g,(_,c)=>'.'+'xyzw'['rgba'.indexOf(c)]);
 s=s.replace(/any\(isnan\((\w+)\)\)/g,'sb_anynan($1)');
 s=s.replace(/,\s*\)/g,')');
 // WebCuda requires explicitly constructed vectors for record and buffer swizzles.
 const used=new Set();s=s.replace(/\b((?:sbuParams\.)?\w+)\.([xyzwrgba]{2,4})\b/g,(_,v,sw)=>{const canonical=sw.replace(/[rgba]/g,c=>'xyzw'['rgba'.indexOf(c)]);used.add(canonical);return `sb_sw_${canonical}(${v})`;});
 // Function-result swizzles from texture sampling.
 for(const name of ['sb_texel','sb_sample']){s=calls(s,name,args=>`${name}(${args.join(',')})`);s=s.replace(/\)\.([rgba])\b/g,(_,c)=>').'+('xyzw'['rgba'.indexOf(c)]));}
 // Result swizzles need balanced-call handling rather than a greedy expression regex.
 for(const name of ['sb_texel','sb_sample']){let cursor=0;for(;;){const at=s.indexOf(name+'(',cursor);if(at<0)break;let e=at+name.length+1,d=1;for(;d;e++){if(s[e]==='(')d++;if(s[e]===')')d--;}
 const m=s.slice(e).match(/^\.([xyzwrgba]{2,4})\b/);if(m){const sw=m[1].replace(/[rgba]/g,c=>'xyzw'['rgba'.indexOf(c)]);used.add(sw);const call=s.slice(at,e),replacement=`sb_sw_${sw}(${call})`;s=s.slice(0,at)+replacement+s.slice(e+m[0].length);cursor=at+replacement.length;}else cursor=e;}}
 source+=struct+s+'\n';metadata.push({entry,fields,textures,outputs:outputs.length,original:originals[key]});
}

return {source,metadata};
}
