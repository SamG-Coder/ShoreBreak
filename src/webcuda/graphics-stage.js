// WebCuda graphics stage adapter. All application expressions come from the
// compiled CUDA source. This layer supplies native rasterization and IO only.
const type=t=>({float:'f32',int:'i32',bool:'i32',vec2:'vec2<f32>',ivec2:'vec2<f32>',vec3:'vec3<f32>',vec4:'vec4<f32>'}[t]);
function invocationContext(source){
 // GLSL globals are per invocation. CUDA's explicit context reference denotes
 // the same object at every call; represent it directly to avoid HLSL inout
 // copies of the complete event table at every helper call.
 // Original GLSL uses the graphics device's float division. The compute
 // compiler's compensated CUDA divide adds a branch at every divide and makes
 // native graphics compilation explode; retain the original graphics operation.
 return source.replace(/^fn cw_divide_f32\([^\n]+$/m,'fn cw_divide_f32(a:f32,b:f32)->f32{return a/b;}')
  .replace(/v_ctx:\s*ptr<function,\s*cw_struct_GraphicsContext>,\s*/g,'')
  .replace(/\(\*v_ctx\)/g,'sb_context').replace(/\(v_ctx,\s*/g,'(')
  .replace(/\(&ctx,\s*/g,'(').replace('var ctx=sb_uniforms[0];','sb_context=sb_uniforms[0];')
  .replace(/\bctx\./g,'sb_context.')+'\nvar<private> sb_context:cw_struct_GraphicsContext;\n';
}
export function graphicsStage(a,stage,varyings=a.outputs,{screen=false,height=1,depthTextures=new Set()}={}){
 const group=stage==='vertex'?0:1;
 let s=a.wgsl.replace('@group(0) @binding(0) var<storage, read> b_input: array<cw_struct_GraphicsContext>;',`@group(${group}) @binding(0) var<storage, read> sb_uniforms: array<cw_struct_GraphicsContext>;\nvar<private> b_input: array<cw_struct_GraphicsContext,1>;`)
 .replace('@group(0) @binding(1) var<storage, read_write> b_output: array<cw_struct_GraphicsContext>;','var<private> b_output: array<cw_struct_GraphicsContext,1>;')
 .replace(/@compute @workgroup_size\([^)]*\)\s*fn main\(/,'fn sb_run(').replace(/@builtin\((?:local_invocation_id|workgroup_id|num_workgroups)\) /g,'');
 // Keep the invocation record in function space. Copying it through private
 // arrays is unnecessary and trips a native D3D shader-compiler bug.
 s=s.replace(/var<private> b_(?:input|output):[^;]+;/g,'');
 const kernel=s.indexOf('fn sb_run('),body=s.slice(kernel);
 let run=body.replace('fn sb_run(','fn sb_run(v_ctx:ptr<function,cw_struct_GraphicsContext>,')
  .replace(/\s*let cw_value_copy_\d+ = b_input\[0i\];\s*var v_ctx: cw_struct_GraphicsContext = [^;]+;/,'')
  .replace(/\s*let cw_value_copy_\d+ = v_ctx;\s*b_output\[0i\] = [\s\S]*?;\s*}\s*$/,'\n}')
  .replace(/&v_ctx\b/g,'v_ctx').replace(/\bv_ctx\./g,'(*v_ctx).');
 s=s.slice(0,kernel)+run;
 const textureBindings=[],usedSlots=new Set();
 for(const m of s.matchAll(/\bsb_((?:texture\w*|texelFetch\w*)_\d)\(([^,]+),/g)){
  if(/^\s*\d+i\s*$/.test(m[2]))usedSlots.add(parseInt(m[2]));
  else a.textures.forEach((t,i)=>{if(m[1].includes('Array')?t.type==='sampler2DArray':m[1].includes('3D')?t.type==='sampler3D':t.type==='sampler2D')usedSlots.add(i);});
 }
 s=s.replace(/\b(sb_(?:texture\w*|texelFetch\w*)_\d)\(\s*(\d+)i,/g,'$1_s$2($2i,');
 for(const [i,t]of a.textures.entries()){
  if(!usedSlots.has(i))continue;
  const dim=t.type==='sampler2DArray'?'2d_array':t.type==='sampler3D'?'3d':'2d';
  s+=`\n@group(${group}) @binding(${2+i*2}) var sb_tex${i}:${depthTextures.has(t.name)?'texture_depth_2d':'texture_'+dim+'<f32>'};\n@group(${group}) @binding(${3+i*2}) var sb_samp${i}:sampler;\n`;
  textureBindings.push({...t,textureBinding:2+i*2,samplerBinding:3+i*2,dimension:dim.replace('_','-')});
 }
 const intrinsics=new Set([...s.matchAll(/\b(sb_(?:texture\w*|texelFetch\w*)_\d(?:_s\d+)?)\(/g)].map(m=>m[1]));
 for(const name of intrinsics){
  const m=name.match(/^sb_(.*)_(\d)(?:_s(\d+))?$/),op=m[1].replace(/Array|3D$/g,''),array=m[1].includes('Array'),volume=m[1].includes('3D'),dim=array||volume?3:2,arity=+m[2],fixed=m[3]===undefined?null:+m[3];
  const size=op==='textureSize',fetch=op==='texelFetch';
  const args=size?'slot:i32,level:i32':`slot:i32,uv:vec${dim}<f32>`+(op==='textureGrad'?`,dx:vec${volume?3:2}<f32>,dy:vec${volume?3:2}<f32>`:arity>=3?`,${fetch?'level:i32':'level:f32'}`:'');
  const result=size?`vec${dim}<f32>`:'vec4<f32>';
  let body=`fn ${name}(${args})->${result}{${fixed===null?'switch slot{':''}\n`;
  for(const [i,t]of a.textures.entries()){
   if(!usedSlots.has(i))continue;
   if(fixed!==null&&i!==fixed)continue;
   if((t.type==='sampler2DArray')!==array||(t.type==='sampler3D')!==volume)continue;
   const tx='sb_tex'+i,samp='sb_samp'+i;
   let expression;
   if(size)expression=array?`vec3<f32>(vec2<f32>(textureDimensions(${tx},level)),f32(textureNumLayers(${tx})))`:`vec${dim}<f32>(textureDimensions(${tx},level))`;
   else if(fetch)expression=`textureLoad(${tx},vec${volume?3:2}<i32>(uv${array?'.xy':''})${array?',i32(uv.z)':''},level)`;
   else{
    const coords=`${tx},${samp},uv${array?'.xy':''}${array?',i32(uv.z)':''}`;
    expression=op==='textureGrad'?`textureSampleGrad(${coords},dx,dy)`:op==='textureLod'||stage==='vertex'?`textureSampleLevel(${coords},${arity>=3?'level':'0.0'})`:arity===3?`textureSampleBias(${coords},level)`:`textureSample(${coords})`;
   }
   if(!size&&depthTextures.has(t.name)){
    if(op==='textureLod'||stage==='vertex')expression=`textureSampleLevel(${tx},${samp},uv,i32(${arity>=3?'level':'0'}))`;
    if(op==='textureGrad'||op==='texture'&&arity===3)expression=`textureSampleLevel(${tx},${samp},uv,0)`;
    expression=`vec4<f32>(${expression},0.0,0.0,1.0)`;
   }
   body+=fixed===null?`case ${i}:{return ${expression};}\n`:`return ${expression};\n`;
  }
  s+=body+(fixed===null?`default:{return ${result}(0);}}}\n`:'}\n');
 }
 const vary=varyings.map((f,i)=>`@location(${i}) ${f.flat?'@interpolate(flat) ':''}${f.name}:${type(f.type)},`).join('\n');
 let wrapper;
 if(stage==='vertex'){
  const attributes=[];let loc=0;
  for(const f of a.inputs){if(f.type==='mat4')for(let i=0;i<4;i++)attributes.push({name:f.name+'_'+i,type:'vec4',location:loc++,matrix:f.name,column:i});else attributes.push({...f,location:loc++});}
  wrapper=`struct SBVertexOutput{@builtin(position) position:vec4<f32>,${vary}}\n@vertex fn vertexMain(@builtin(vertex_index) vid:u32,@builtin(instance_index) iid:u32,${attributes.map(f=>`@location(${f.location}) ${f.name}:${type(f.type)}`).join(',')})->SBVertexOutput{\nb_input[0]=sb_uniforms[0];\n`;
  for(const f of attributes)wrapper+=`b_input[0].cw_field_${f.matrix||f.name}${f.matrix?'.cw_field_c'+f.column:''}=${f.name};\n`;
  wrapper+='b_input[0].cw_field_gl_VertexID=i32(vid);b_input[0].cw_field_gl_InstanceID=i32(iid);sb_run(vec3<u32>(vid,iid,0),vec3<u32>(0),vec3<u32>(1));var r:SBVertexOutput;r.position=b_output[0].cw_field_gl_Position;';
  wrapper+=`r.position.y*= ${screen?'1.0':'-1.0'};r.position.z=(r.position.z+r.position.w)*0.5;`;
  for(const f of varyings)wrapper+=`r.${f.name}=b_output[0].cw_field_${f.name};`;
  wrapper+='return r;}';
  wrapper=wrapper.replace('b_input[0]=sb_uniforms[0];','var ctx=sb_uniforms[0];').replaceAll('b_input[0]','ctx').replaceAll('b_output[0]','ctx').replace('sb_run(','sb_run(&ctx,');
  return {wgsl:invocationContext('diagnostic(off, derivative_uniformity);\n'+s+wrapper),textureBindings,attributes};
 }
 const colors=[...(a.outputs.some(f=>f.location===0)?[]:[{name:'gl_FragColor',type:'vec4',location:0}]),...a.outputs].sort((a,b)=>a.location-b.location),depth=/cw_field_gl_FragDepth\s*=/.test(s);
 wrapper=`struct SBFragmentInput{@builtin(position) position:vec4<f32>,@builtin(front_facing) front:bool,${vary}}\nstruct SBFragmentOutput{${colors.map((f,i)=>`@location(${i}) c${i}:vec4<f32>,`).join('')}${depth?'@builtin(frag_depth) depth:f32,':''}}\n@fragment fn fragmentMain(v:SBFragmentInput)->SBFragmentOutput{b_input[0]=sb_uniforms[0];\n`;
 for(const f of a.inputs)if(varyings.some(v=>v.name===f.name))wrapper+=`b_input[0].cw_field_${f.name}=v.${f.name};\n`;
 wrapper+=`b_input[0].cw_field_gl_FragCoord=vec4<f32>(v.position.x,${screen?height+'.0-v.position.y':'v.position.y'},v.position.z,v.position.w);b_input[0].cw_field_gl_FrontFacing=select(0,1,v.front);b_input[0].cw_field_discarded=0;sb_run(vec3<u32>(0),vec3<u32>(0),vec3<u32>(1));if(b_output[0].cw_field_discarded!=0){discard;}var r:SBFragmentOutput;`;
 colors.forEach((f,i)=>wrapper+=`r.c${i}=b_output[0].cw_field_${f.name};`);if(depth)wrapper+='r.depth=b_output[0].cw_field_gl_FragDepth;';wrapper+='return r;}';
 wrapper=wrapper.replace('b_input[0]=sb_uniforms[0];','var ctx=sb_uniforms[0];').replaceAll('b_input[0]','ctx').replaceAll('b_output[0]','ctx').replace('sb_run(','sb_run(&ctx,');
 return {wgsl:invocationContext('diagnostic(off, derivative_uniformity);\n'+s+wrapper),textureBindings};
}
