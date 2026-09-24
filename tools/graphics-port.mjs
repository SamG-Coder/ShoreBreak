// Translate upstream GLSL syntax to CUDA. The expressions and control flow are
// retained; this is an authoring tool, never a browser shader fallback.
import {parse,generate} from '@shaderfrog/glsl-parser/index.js';
import {preprocess} from '@shaderfrog/glsl-parser/preprocessor/index.js';
const spelling=t=>({uvec2:'uint2',uvec3:'uint3',uvec4:'uint4',vec2:'float2',vec3:'float3',vec4:'float4',ivec2:'float2',ivec3:'float3',ivec4:'float4',mat2:'SBMat2',mat3:'SBMat3',mat4:'SBMat4',sampler2D:'int',sampler2DArray:'int',sampler3D:'int'}[t]||t);
const token=n=>n?.token??n?.identifier??n?.literal;
const typename=n=>token(n?.specifier?.specifier??n?.specifier??n);
function walk(n,fn){if(!n||typeof n!=='object')return;if(Array.isArray(n)){n.forEach(v=>walk(v,fn));return;}fn(n);for(const [k,v]of Object.entries(n))if(k!=='scopes')walk(v,fn);}
export function graphicsPort(source,stage,defines={}){
 const prefix=stage==='vertex'?'in vec3 position; in vec3 normal; in vec2 uv; uniform mat4 modelMatrix; uniform mat4 modelViewMatrix; uniform mat4 viewMatrix; uniform mat4 projectionMatrix; uniform mat3 normalMatrix; uniform vec3 cameraPosition;':'uniform mat4 viewMatrix; uniform mat4 projectionMatrix; uniform vec3 cameraPosition;';
 // Three injects these declarations only if they are not present in the source.
 const injected=prefix.split(';').filter(s=>s.trim()&&!new RegExp('\\b(?:uniform|in)\\s+\\w+\\s+'+s.trim().split(/\s+/).at(-1)+'\\b').test(source)).join(';')+';';
 const instance=stage==='vertex'?(defines.USE_INSTANCING?'in mat4 instanceMatrix;':'')+(defines.USE_INSTANCING_COLOR?'in vec3 instanceColor;':'')+(!/\bin\s+vec3\s+color\b/.test(source)&&/\bcolor\b/.test(source)?'in vec3 color;':''):'';
 const ast=parse(preprocess(injected+'\n'+instance+'\n'+source,{defines}),{stage,quiet:true});
 walk(ast,n=>{while(n.type==='postfix'&&n.postfix?.type==='postfix'){const p=n.postfix;n.expression={type:'postfix',expression:n.expression,postfix:p.expression};n.postfix=p.postfix;}});
 const fields=[],textures=[],inputs=[],outputs=[],locals=[],constants=[],structs=[],functions=[];
 for(const n of ast.program){
  if(n.type==='function'){functions.push(n);continue;}
  if(n.type==='precision'||n.type==='preprocessor')continue;
  const d=n.declaration;if(d?.type!=='declarator_list')continue;
  const spec=d.specified_type,st=spec.specifier.specifier;
  if(st.type==='struct'){structs.push(n);continue;}
  const type=typename(spec),qual=(spec.qualifiers||[]).map(token);
  for(const v of d.declarations){const f={type,name:v.identifier.identifier,array:v.quantifier?.map(q=>generate(q.expression)).join('')||null,flat:qual.includes('flat'),location:Number(generate(spec).match(/location\s*=\s*(\d+)/)?.[1]||0)};
   if(qual.includes('uniform'))(type.startsWith('sampler')?textures:fields).push(f);
   else if(qual.includes('in')||qual.includes('attribute'))inputs.push(f);
   else if(qual.includes('out')||qual.includes('varying'))outputs.push(f);
   else if(qual.includes('const'))constants.push({f,initializer:v.initializer});
   else locals.push({f,initializer:v.initializer});
  }
 }
 const builtins=stage==='vertex'?[{type:'vec4',name:'gl_Position'},{type:'int',name:'gl_VertexID'},{type:'int',name:'gl_InstanceID'}]:[{type:'vec4',name:'gl_FragColor'},{type:'vec4',name:'gl_FragCoord'},{type:'bool',name:'gl_FrontFacing'},{type:'float',name:'gl_FragDepth'},{type:'bool',name:'discarded'}];
 const globals=[...fields,...inputs,...outputs,...locals.map(l=>l.f),...builtins];
 for(const f of globals)for(const ref of ast.scopes[0].bindings[f.name]?.references||[]){if(ref!==ast.scopes[0].bindings[f.name].declaration)ref.identifier='ctx.'+f.name;}
 // Built-ins don't always have a binding when a stage never reads them.
 walk(ast,n=>{if(n.type==='identifier'&&builtins.some(f=>f.name===n.identifier))n.identifier='ctx.'+n.identifier;});
 for(const [i,f]of textures.entries())for(const ref of ast.scopes[0].bindings[f.name]?.references||[]){if(ref!==ast.scopes[0].bindings[f.name].declaration)ref.identifier=String(i);}
 const names=new Set(functions.map(f=>f.prototype.header.name.identifier));
 const variableTypes=new Map(globals.map(f=>['ctx.'+f.name,f.type]));
 const structTypes=new Map();
 for(const n of structs){const st=n.declaration.specified_type.specifier.specifier;const m=new Map();for(const d of st.declarations)for(const v of d.declaration.declarations)m.set(v.identifier.identifier,typename(d.declaration.specified_type));structTypes.set(st.typeName.identifier,m);}
 walk(ast,n=>{if(n.type==='declarator_list')for(const d of n.declarations)variableTypes.set(d.identifier.identifier,typename(n.specified_type));if(n.type==='parameter_declaration'&&n.identifier)variableTypes.set(n.identifier.identifier,typename(n.specifier));});
 function infer(n){if(n?.type==='identifier')return variableTypes.get(n.identifier);if(n?.type==='postfix'){const t=infer(n.expression);return n.postfix.type==='field_selection'?structTypes.get(t)?.get(n.postfix.selection.identifier):/^mat[234]$/.test(t||'')?'vec'+t.slice(-1):t;}if(n?.type==='function_call'){const name=token(n.identifier?.specifier??n.identifier);return functions.find(f=>f.prototype.header.name.identifier===name)?typename(functions.find(f=>f.prototype.header.name.identifier===name).prototype.header.returnType):name;}return null;}
 const replaceNames=new Map([['uvec2','make_uint2'],['uvec3','make_uint3'],['uvec4','make_uint4'],['vec2','sb_vec2'],['ivec2','sb_vec2'],['vec3','sb_vec3'],['ivec3','sb_vec3'],['vec4','sb_vec4'],['ivec4','sb_vec4'],['mat2','sb_mat2'],['mat3','sb_mat3'],['mat4','sb_mat4'],...['sin','cos','exp','log','sqrt','abs','floor','fract','sign','tanh','min','max','clamp','mix','step','smoothstep','mod'].map(n=>[n,'sb_'+n]),['round','sb_round'],['pow','sb_pow'],['dFdx','__sb_dFdx'],['dFdy','__sb_dFdy'],['fwidth','__sb_fwidth'],['exp2','sb_exp2'],['log2','sb_log2'],['ceil','sb_ceil'],['inversesqrt','sb_inversesqrt'],['atan','sb_atan']]);
 let temp=0;
 function lower(n){
  if(!n)return '';
  const uintOp=op=>({'^':'xor','&':'and','|':'or','<<':'shl','>>':'shr'}[op]);
  if(n.type==='binary'&&infer(n.left)?.startsWith('uvec')&&uintOp(token(n.operator)))return 'sb_uint_'+uintOp(token(n.operator))+'('+lower(n.left)+','+lower(n.right)+')';
  if(n.type==='function_call'){
   const name=token(n.identifier?.specifier??n.identifier),args=n.args.filter(a=>a.literal!==',').map(lower);
   if(/^[i]?vec[234]$/.test(name))n.args.filter(a=>a.literal!==',').forEach((a,i)=>{if(a.type==='int_constant'||infer(a)==='int')args[i]='float('+args[i]+')';});
   if(['texture','textureLod','texelFetch','textureSize','textureGrad'].includes(name)){
    let dim=textures[Number(args[0])]?.type||'sampler2D';
    return '__sb_'+name+(dim==='sampler2DArray'?'Array':dim==='sampler3D'?'3D':'')+'('+args.join(',')+(name==='texture'&&stage==='vertex'?',0.0f':'')+')';
   }
   return (structTypes.has(name)?'sb_new_'+name:/^ivec[234]$/.test(name)?'sb_'+name:replaceNames.get(name)||name)+'('+(names.has(name)?'ctx'+(args.length?',':''):'')+args.join(',')+')';
  }
  if(n.type==='float_constant')return n.token.replace(/[fF]$/,'')+'f';
  if(n.type==='unary'&&token(n.operator)==='-')return '(0-('+lower(n.expression)+'))';
  if(n.type==='postfix'&&n.postfix.type==='quantifier'&&/^mat[234]$/.test(infer(n.expression)||'')){const i=lower(n.postfix.expression);if(!/^[0-3]$/.test(i))throw Error('Dynamic matrix column needs an explicit CUDA accessor');return lower(n.expression)+'.c'+i;}
  if(n.type==='postfix'&&n.postfix.type==='field_selection'){
   const field=n.postfix.selection.identifier,base=lower(n.expression);
   if(!structTypes.has(infer(n.expression))&&/^[xyzwrgba]{2,4}$/.test(field))return `sb_sw_${field.replace(/[rgba]/g,c=>'xyzw'['rgba'.indexOf(c)])}(${base})`;
   const component=base+'.'+(!structTypes.has(infer(n.expression))&&/^[rgba]$/.test(field)?'xyzw'['rgba'.indexOf(field)]:field);
   return infer(n.expression)?.startsWith('ivec')?'int('+component+')':component;
  }
  if(n.type==='expression_statement'&&n.expression?.type==='assignment'){
   const a=n.expression,l=a.left;
   if(a.right.type==='assignment'&&token(a.operator)==='='){
    const chain=[];let current=a;while(current.type==='assignment'&&token(current.operator)==='='){chain.push(lower(current.left));current=current.right;}
    return '{'+chain.reverse().map((lhs,i)=>lhs+'='+(i?chain[i-1]:lower(current))+';').join('')+'}';
   }
   if(l.type==='postfix'&&!structTypes.has(infer(l.expression))&&l.postfix.type==='field_selection'&&l.postfix.selection.identifier.length>1&&/^[xyzwrgba]+$/.test(l.postfix.selection.identifier)){
    const sw=l.postfix.selection.identifier.replace(/[rgba]/g,c=>'xyzw'['rgba'.indexOf(c)]),id='sb_write'+temp++,op=token(a.operator),base=lower(l.expression);
    return `{float${sw.length} ${id}=${op==='='?lower(a.right):lower(l)+op.slice(0,-1)+'('+lower(a.right)+')'};${[...sw].map((c,i)=>`${base}.${c}=${id}.${'xyzw'[i]};`).join('')}}`;
   }
  }
  if(n.type==='discard_statement')return '{ctx.discarded=true;return;}';
  if(n.type==='assignment'){const lhs=lower(n.left).replace(/^int\((.*)\)$/,'$1'),op=token(n.operator);return lhs+'='+(op==='='?lower(n.right):infer(n.left)?.startsWith('uvec')&&uintOp(op.slice(0,-1))?'sb_uint_'+uintOp(op.slice(0,-1))+'('+lhs+','+lower(n.right)+')':lhs+op.slice(0,-1)+'('+lower(n.right)+')');}
  if(n.type==='keyword')return spelling(n.token)+(n.whitespace||'');
  if(n.type==='parameter_declaration'){
   const type=typename(n.specifier),ref=(n.qualifier||[]).some(q=>['out','inout'].includes(token(q)));
   return spelling(type)+(ref?'& ':' ')+n.identifier.identifier+(n.quantifier?generate(n.quantifier):'');
  }
  // Let the canonical GLSL generator retain all remaining syntax while
  // replacing child nodes with already lowered source fragments.
  const copy={...n};for(const [k,v]of Object.entries(copy))if(v&&typeof v==='object')copy[k]=Array.isArray(v)?v.map(x=>typeof x==='object'?{type:'identifier',identifier:lower(x)}:x):{type:'identifier',identifier:lower(v)};
  return generate(copy);
 }
 const decl=f=>`${spelling(f.type==='bool'?'int':f.type)} ${f.name}${f.array?'['+f.array+']':''};`;
 let cuda=structs.map(lower).join('\n')+'\nstruct GraphicsContext {\n'+globals.map(decl).join('\n')+'\n};\n';
 for(const [name,members]of structTypes)cuda+=`__device__ ${name} sb_new_${name}(${[...members].map(([n,t])=>spelling(t)+' p_'+n).join(',')}){${name} a;${[...members].map(([n])=>'a.'+n+'=p_'+n+';').join('')}return a;}\n`;
 for(const c of constants)cuda+='// Constant '+c.f.name+' is initialized in each invocation.\n';
 // Constants are fields to support matrix/record constructors and CUDA's
 // restrictions on file-scope initialization, preserving evaluation order.
 if(constants.length){cuda=cuda.replace('struct GraphicsContext {','struct GraphicsContext {\n'+constants.map(c=>decl(c.f)).join('\n'));for(const c of constants)for(const ref of ast.scopes[0].bindings[c.f.name]?.references||[])if(ref!==ast.scopes[0].bindings[c.f.name].declaration)ref.identifier='ctx.'+c.f.name;}
 for(const f of functions){const p=f.prototype;variableTypes.clear();for(const g of globals)variableTypes.set('ctx.'+g.name,g.type);walk(f,n=>{if(n.type==='declarator_list')for(const d of n.declarations)variableTypes.set(d.identifier.identifier,typename(n.specified_type));if(n.type==='parameter_declaration'&&n.identifier)variableTypes.set(n.identifier.identifier,typename(n.specifier));});cuda+=`__device__ ${spelling(typename(p.header.returnType))} ${p.header.name.identifier}(GraphicsContext& ctx${p.parameters?.length?','+p.parameters.map(lower).join(','):''})${lower(f.body)}\n`;}
 const init=[...constants,...locals].filter(c=>c.initializer).map(c=>c.f.array&&c.initializer.type==='function_call'?c.initializer.args.filter(a=>a.literal!==',').map((a,i)=>`ctx.${c.f.name}[${i}]=${lower(a)};`).join('\n'):`ctx.${c.f.name}=${lower(c.initializer)};`).join('\n');
 cuda+=`__global__ void graphicsStage(const GraphicsContext* input,GraphicsContext* output){GraphicsContext ctx=input[0];${init}\nmain(ctx);output[0]=ctx;}\n`;
 cuda=cuda.replace(/struct\s+\w+\s*\{[^}]*\}/g,s=>s.replace(/\b(\w+)\s+([\w,\s]+);/g,(_,type,names)=>names.split(',').map(n=>type+' '+n.trim()+';').join(' '))).replace(/any\(isnan\((\w+)\)\)/g,'sb_anynan($1)');
 return {cuda,fields,textures,inputs,outputs,builtins,globals:[...constants.map(c=>c.f),...globals]};
}
