const align=(n,a)=>Math.ceil(n/a)*a;
export function contextLayout(wgsl){
 const structs=new Map([...wgsl.matchAll(/struct (\w+)\s*\{([^}]+)\}/g)].map(m=>[m[1],m[2]])),cache=new Map();
 function layout(t){
  t=t.trim();if(cache.has(t))return cache.get(t);
  let r,m;if(['f32','i32','u32'].includes(t))r={type:t,size:4,align:4};
  else if(m=t.match(/^vec([234])<(.+)>$/)){const n=+m[1];r={type:m[2],count:n,size:n*4,align:n===2?8:16};}
  else if(m=t.match(/^array<(.+),\s*(\d+)>$/)){const item=layout(m[1]),stride=align(item.size,item.align);r={item,count:+m[2],stride,size:stride*+m[2],align:item.align};}
  else{
   const source=structs.get(t);if(!source)throw Error('Unknown CUDA record '+t);
   let offset=0,max=1;const fields=[];
   for(const line of source.split('\n')){const m=line.trim().match(/^(\w+):\s*(.+),$/);if(!m)continue;const child=layout(m[2]);offset=align(offset,child.align);fields.push({name:m[1].replace(/^cw_field_/,''),offset,...child});offset+=child.size;max=Math.max(max,child.align);}
   r={fields,size:align(offset,max),align:max};
  }cache.set(t,r);return r;
 }
 return layout('cw_struct_GraphicsContext');
}
export function packContext(layout,values){
 const buffer=new ArrayBuffer(layout.size),view=new DataView(buffer);
 function write(l,value,offset){
  if(value==null)return;
  value=value.elements??(value.toArray?value.toArray():value);
  if(l.fields){if(Array.isArray(value)||ArrayBuffer.isView(value)){let i=0;for(const f of l.fields){const n=f.count||1;write(f,value.slice(i,i+n),offset+f.offset);i+=n;}}else for(const f of l.fields)write(f,value[f.name],offset+f.offset);}
  else if(l.item){
   const scalarCount=t=>t.fields?t.fields.reduce((n,f)=>n+scalarCount(f),0):t.item?t.count*scalarCount(t.item):t.count||1;
   const nested=value[0]!==null&&typeof value[0]==='object',n=scalarCount(l.item);
   for(let i=0;i<l.count;i++)write(l.item,nested?value[i]:n>1?value.slice(i*n,(i+1)*n):value[i],offset+l.stride*i);
  }
  else for(let i=0;i<(l.count||1);i++){const v=l.count?value[i]:Array.isArray(value)||ArrayBuffer.isView(value)?value[0]:value;view[l.type==='i32'?'setInt32':l.type==='u32'?'setUint32':'setFloat32'](offset+4*i,Number(v)||0,true);}
 }write(layout,values,0);return buffer;
}
export function shaderKey(mat,mesh){
 const key=JSON.stringify([mat.isMeshDepthMaterial?'depth':mat.vertexShader,mat.isMeshDepthMaterial?'depth':mat.fragmentShader,mat.defines||{},!!mesh.instanceMatrix,!!mesh.instanceColor]);let h=2166136261;for(let i=0;i<key.length;i++)h=Math.imul(h^key.charCodeAt(i),16777619);return (h>>>0).toString(16);
}
