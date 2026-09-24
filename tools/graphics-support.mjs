export function graphicsSupport(base){
 let s=base.slice(base.indexOf('__device__ float2 sb_vec2'),base.indexOf('struct SBUniforms'));
 for(let n=2;n<=4;n++)s+='__device__ float'+n+' sb_vec'+n+'(uint'+n+' a){return make_float'+n+'('+[...'xyzw'.slice(0,n)].map(c=>'float(a.'+c+')').join(',')+');}\n';
 for(let n=2;n<=4;n++)for(const [name,op]of [['xor','^'],['and','&'],['or','|'],['shl','<<'],['shr','>>']])for(const vector of [false,true])s+=`__device__ uint${n} sb_uint_${name}(uint${n} a,${vector?'uint'+n:'uint'} b){return make_uint${n}(${[...'xyzw'.slice(0,n)].map(c=>`a.${c}${op}b${vector?'.'+c:''}`).join(',')});}\n`;
 // CUDA helper functions retain scalar operation order used by upstream GLSL.
 for(const [name,fn]of [['round','roundf'],['pow','powf'],['exp2','exp2f'],['log2','log2f'],['ceil','ceilf'],['inversesqrt','rsqrtf'],['atan','atanf']]){
  const binary=name==='pow';s+=`__device__ float sb_${name}(float a${binary?',float b':''}){return ${fn}(a${binary?',b':''});}\n`;
  for(let n=2;n<=4;n++)for(const pair of binary?[[n,n],[n,1]]:[[n,0]])s+=`__device__ float${n} sb_${name}(float${n} a${binary?',float'+(pair[1]===1?'':n)+' b':''}){return make_float${n}(${[...'xyzw'.slice(0,n)].map(c=>`sb_${name}(a.${c}${binary?',b'+(pair[1]===1?'':'.'+c):''})`).join(',')});}\n`;
 }
 s+='__device__ float sb_atan(float a,float b){return atan2f(a,b);}\n';
 s+='__device__ bool sb_anynan(float4 a){return !isfinite(a.x)||!isfinite(a.y)||!isfinite(a.z)||!isfinite(a.w);}\n';
 s+='__device__ float3 cross(float3 a,float3 b){return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}\n';
 s+='__device__ float3 reflect(float3 i,float3 n){return i-2.0f*dot(n,i)*n;}\n__device__ float3 refract(float3 i,float3 n,float eta){float d=dot(n,i),k=1.0f-eta*eta*(1.0f-d*d);if(k<0.0f)return make_float3(0.0f,0.0f,0.0f);return eta*i-(eta*d+sqrtf(k))*n;}\n';
 s+='__device__ float2 sb_vec2(float3 a){return make_float2(a.x,a.y);}\n__device__ float2 sb_vec2(float4 a){return make_float2(a.x,a.y);}\n__device__ float3 sb_vec3(float4 a){return make_float3(a.x,a.y,a.z);}\n';
 for(let n=2;n<=4;n++){
  s+=`__device__ float${n} sb_ivec${n}(float${n} a){return make_float${n}(${[...'xyzw'.slice(0,n)].map(c=>'truncf(a.'+c+')').join(',')});}\n`;
  s+=`__device__ float${n} sb_ivec${n}(float a){return sb_ivec${n}(sb_vec${n}(a));}\n`;
  s+=`__device__ float${n} sb_ivec${n}(${[...'xyzw'.slice(0,n)].map(c=>'float '+c).join(',')}){return sb_ivec${n}(sb_vec${n}(${'xyzw'.slice(0,n).split('').join(',')}));}\n`;
 }
 for(let n=2;n<=4;n++){
  s+=`struct SBMat${n}{${Array.from({length:n},(_,i)=>`float${n} c${i};`).join('')}};\n`;
  s+=`__device__ SBMat${n} sb_mat${n}(${Array.from({length:n*n},(_,i)=>`float a${i}`).join(',')}){SBMat${n} m;${Array.from({length:n},(_,i)=>`m.c${i}=make_float${n}(${Array.from({length:n},(_,j)=>'a'+(i*n+j)).join(',')});`).join('')}return m;}\n`;
  s+=`__device__ SBMat${n} sb_mat${n}(${Array.from({length:n},(_,i)=>`float${n} c${i}`).join(',')}){SBMat${n} m;${Array.from({length:n},(_,i)=>`m.c${i}=c${i};`).join('')}return m;}\n`;
  s+=`__device__ SBMat${n} sb_mat${n}(float a){return sb_mat${n}(${Array.from({length:n*n},(_,i)=>i%(n+1)===0?'a':'0.0f').join(',')});}\n`;
  s+=`__device__ float${n} operator*(SBMat${n} m,float${n} v){return ${Array.from({length:n},(_,i)=>`m.c${i}*v.${'xyzw'[i]}`).join('+')};}\n`;
  s+=`__device__ float${n} operator*(float${n} v,SBMat${n} m){return make_float${n}(${Array.from({length:n},(_,i)=>`dot(v,m.c${i})`).join(',')});}\n`;
  s+=`__device__ SBMat${n} operator*(SBMat${n} a,SBMat${n} b){return sb_mat${n}(${Array.from({length:n},(_,i)=>`a*b.c${i}`).join(',')});}\n`;
  s+=`__device__ SBMat${n} transpose(SBMat${n} a){return sb_mat${n}(${Array.from({length:n*n},(_,i)=>`a.c${i%n}.${'xyzw'[Math.floor(i/n)]}`).join(',')});}\n`;
 }
 s+='__device__ SBMat3 sb_mat3(SBMat4 a){return sb_mat3(sb_vec3(a.c0),sb_vec3(a.c1),sb_vec3(a.c2));}\n';
 s+='__device__ SBMat4 sb_mat4(SBMat3 a){return sb_mat4(sb_vec4(a.c0,0.0f),sb_vec4(a.c1,0.0f),sb_vec4(a.c2,0.0f),sb_vec4(0.0f,0.0f,0.0f,1.0f));}\n';
 for(let n=2;n<=4;n++){
  for(const [name,op]of [['lessThan','<'],['greaterThan','>'],['lessThanEqual','<='],['greaterThanEqual','>='],['equal','=='],['notEqual','!=']])s+=`__device__ float${n} ${name}(float${n} a,float${n} b){return make_float${n}(${[...'xyzw'.slice(0,n)].map(c=>`a.${c}${op}b.${c}?1.0f:0.0f`).join(',')});}\n`;
  for(const [name,op]of [['all','&&'],['any','||']])s+=`__device__ bool ${name}(float${n} a){return ${[...'xyzw'.slice(0,n)].map(c=>`a.${c}!=0.0f`).join(op)};}\n`;
 }
 return s;
}
export function swizzleSupport(source){
 let s='';for(const sw of new Set([...source.matchAll(/sb_sw_([xyzw]{2,4})\(/g)].map(m=>m[1])))for(let n=Math.max(2,...[...sw].map(c=>'xyzw'.indexOf(c)+1));n<=4;n++)s+=`__device__ float${sw.length} sb_sw_${sw}(float${n} a){return make_float${sw.length}(${[...sw].map(c=>'a.'+c).join(',')});}\n`;return s;
}
