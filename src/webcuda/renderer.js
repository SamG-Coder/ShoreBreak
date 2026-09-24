import {Color,Vector2,Matrix3,Matrix4,Frustum} from 'three/src/Three.Core.js';
import {graphicsStage} from './graphics-stage.js';
import {assetUrl} from '../core/assets.js';
import {contextLayout,packContext,shaderKey} from './graphics-layout.js';
const U=GPUTextureUsage,B=GPUBufferUsage;
const factors={200:'zero',201:'one',202:'src',203:'one-minus-src',204:'src-alpha',205:'one-minus-src-alpha',206:'dst-alpha',207:'one-minus-dst-alpha',208:'dst',209:'one-minus-dst'};
const depthCompare={0:'never',1:'always',2:'less',3:'less-equal',4:'equal',5:'greater-equal',6:'greater',7:'not-equal'};
export class WebCudaRenderer{
 static async create(options={}){
  const gpu=navigator.gpu;if(!gpu)throw Error('WebCuda requires WebGPU on localhost or HTTPS');const adapter=await gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('WebCuda requires a WebGPU adapter');
  for(const feature of ['float32-filterable','float32-blendable'])if(!adapter.features.has(feature))throw Error('WebCuda requires GPU feature '+feature);
  const features=['shader-f16','float32-filterable','float32-blendable'].filter(f=>adapter.features.has(f));
  const requiredLimits={};for(const k of ['maxSampledTexturesPerShaderStage','maxSamplersPerShaderStage','maxStorageBuffersPerShaderStage','maxInterStageShaderVariables','maxColorAttachmentBytesPerSample'])requiredLimits[k]=adapter.limits[k];
  const device=await adapter.requestDevice({requiredFeatures:features,requiredLimits});
  const manifest=await fetch(assetUrl('faithful/graphics/index.json')).then(r=>r.json()),artifacts={};
  await Promise.all(Object.entries(manifest).map(async([key,id])=>{artifacts[key]=await Promise.all(['vertex','fragment'].map(stage=>fetch(assetUrl(`faithful/graphics/${id}-${stage}.json`)).then(r=>r.json())));artifacts[key].id=id;}));
  const renderer=new WebCudaRenderer(device,artifacts,options);renderer.gpu=gpu;renderer.adapter=adapter;renderer.mipKey=Object.keys(manifest).find(k=>manifest[k]==='mip');return renderer;
 }
 constructor(device,artifacts,options){
  this.device=device;this.artifacts=artifacts;this.domElement=options.canvas||document.createElement('canvas');this.context=this.domElement.getContext('webgpu');this.format=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device,format:this.format,alphaMode:'opaque'});
  this.target=null;this.layer=0;this.level=0;this.clearColor=new Color();this.alpha=1;this.ratio=1;this.width=1;this.height=1;this.debug={};this.capabilities={isWebGPU:true,getMaxAnisotropy:()=>16};this.extensions={get:()=>null};this.info={programs:[],render:{calls:0,triangles:0}};
  this.textures=new WeakMap();this.targets=new WeakMap();this.buffers=new WeakMap();this.pipelines=new Map();this.uniformPool=[];this.uniformIndex=0;this.pending=0;this.encoder=null;this.modules=[];this.errors=[];this.zeroBuffers=new Map();this.emptyTextures=new Map();
  this.mv=new Matrix4();this.nm=new Matrix3();this.frustum=new Frustum();this.clip=new Matrix4();
  window.__cudaRenderer=this;device.addEventListener('uncapturederror',e=>{this.errors.push(e.error.message);console.error(e.error.message);window.__loadError?.(e.error);});device.lost.then(info=>{if(info.reason!=='destroyed')window.__loadError?.(new Error('WebCuda device lost ('+info.reason+'): '+info.message));});
 }
 setPixelRatio(r){this.ratio=r;}getPixelRatio(){return this.ratio;}
 setSize(w,h,css=true){this.flush();this.width=Math.max(1,Math.floor(w*this.ratio));this.height=Math.max(1,Math.floor(h*this.ratio));this.domElement.width=this.width;this.domElement.height=this.height;if(css){this.domElement.style.width=w+'px';this.domElement.style.height=h+'px';}this.screenDepth?.destroy();this.screenDepth=this.device.createTexture({size:[this.width,this.height],format:'depth32float',usage:U.RENDER_ATTACHMENT});}
 getDrawingBufferSize(v=new Vector2()){return v.set(this.width,this.height);}getSize(v=new Vector2()){return v.set(this.width/this.ratio,this.height/this.ratio);}
 getRenderTarget(){return this.target;}getActiveCubeFace(){return this.layer;}getActiveMipmapLevel(){return this.level;}
 getClearAlpha(){return this.alpha;}setClearAlpha(a){this.alpha=a;}getClearColor(c){return c.copy(this.clearColor);}setClearColor(c,a=1){this.clearColor.set(c);this.alpha=a;}
 setRenderTarget(target,layer=0,level=0){if(this.target&&this.target!==target)this.finishTarget(this.target);this.target=target;this.layer=layer;this.level=level;if(target)this.initRenderTarget(target);}
 command(){return this.encoder??=this.device.createCommandEncoder();}
 flush(){if(this.encoder){this.device.queue.submit([this.encoder.finish()]);this.encoder=null;}this.uniformIndex=0;this.pending=0;}
 async idle(){this.flush();await this.device.queue.onSubmittedWorkDone();if(this.errors.length)throw Error(this.errors[0]);}
 textureFormat(t){if(t.isDepthTexture)return 'depth32float';const prefix=t.format===1028?'r':t.format===1030?'rg':'rgba';return t.type===1015?prefix+'32float':t.type===1016?prefix+'16float':prefix+'8unorm'+(t.colorSpace==='srgb'&&prefix==='rgba'?'-srgb':'');}
 initRenderTarget(rt){
  let record=this.targets.get(rt);if(record&&record.width===rt.width&&record.height===rt.height)return record;
  if(record){this.flush();for(const t of record.colors)t.gpu.destroy();record.depth?.destroy();}
  const layers=rt.isWebGLArrayRenderTarget?rt.depth:1,colors=rt.textures.map(t=>this.allocateTexture(t,rt.width,rt.height,layers,true));
  const depth=rt.depthBuffer?this.device.createTexture({size:[rt.width,rt.height],format:'depth32float',usage:U.RENDER_ATTACHMENT|U.TEXTURE_BINDING|U.COPY_SRC}):null;
  if(rt.depthTexture)this.textures.set(rt.depthTexture,{gpu:depth,format:'depth32float',width:rt.width,height:rt.height,layers:1,mips:1,version:rt.depthTexture.version,sampler:this.device.createSampler({minFilter:'nearest',magFilter:'nearest'})});
  record={colors,depth,width:rt.width,height:rt.height};this.targets.set(rt,record);return record;
 }
 allocateTexture(t,width,height,layers=1,target=false){
  const format=this.textureFormat(t),mips=t.generateMipmaps?1+Math.floor(Math.log2(Math.max(width,height))):1,dimension=t.isData3DTexture?'3d':'2d';
  const gpu=this.device.createTexture({label:t.name||'ShoreBreak texture',size:[width,height,layers],format,dimension,mipLevelCount:mips,usage:U.TEXTURE_BINDING|U.COPY_DST|U.COPY_SRC|(dimension==='3d'?0:U.RENDER_ATTACHMENT)});
  const linear=t.magFilter!==1003,sampler=this.device.createSampler({addressModeU:t.wrapS===1000?'repeat':'clamp-to-edge',addressModeV:t.wrapT===1000?'repeat':'clamp-to-edge',addressModeW:t.wrapR===1000?'repeat':'clamp-to-edge',magFilter:linear?'linear':'nearest',minFilter:t.minFilter===1003?'nearest':'linear',mipmapFilter:t.minFilter===1008?'linear':'nearest',maxAnisotropy:linear&&t.minFilter===1008?Math.min(t.anisotropy||1,16):1});
  const record={gpu,format,width,height,layers,mips,version:-1,sampler,dimension,target,dirty:false};this.textures.set(t,record);return record;
 }
 initTexture(t){return this.texture(t);}
 texture(t,hint='2d',depth=false){
  if(!t){const key=hint+depth;if(this.emptyTextures.has(key))return this.emptyTextures.get(key);const gpu=this.device.createTexture({size:[1,1,hint==='2d-array'?3:1],dimension:hint==='3d'?'3d':'2d',format:depth?'depth32float':'rgba8unorm',usage:U.TEXTURE_BINDING|U.COPY_DST|(hint==='3d'?0:U.RENDER_ATTACHMENT)}),r={gpu,format:depth?'depth32float':'rgba8unorm',sampler:this.device.createSampler({minFilter:'nearest',magFilter:'nearest'}),mips:1};this.emptyTextures.set(key,r);return r;}
  let r=this.textures.get(t);if(r?.target||t.isDepthTexture&&r)return r;
  const image=t.image||{},width=image.width||1,height=image.height||1,layers=image.depth||1;
  if(!r||r.width!==width||r.height!==height){r?.gpu.destroy();r=this.allocateTexture(t,width,height,layers);}
  if(r.version!==t.version){
   if(image.data){const data=image.data,components=t.format===1028?1:t.format===1030?2:4,bytes=data.BYTES_PER_ELEMENT;this.flush();this.device.queue.writeTexture({texture:r.gpu},data,{bytesPerRow:width*components*bytes,rowsPerImage:height},[width,height,layers]);}
   else if(width>1&&(image instanceof HTMLImageElement||image instanceof HTMLCanvasElement||image instanceof ImageBitmap)){this.flush();this.device.queue.copyExternalImageToTexture({source:image,flipY:!!t.flipY},{texture:r.gpu},[width,height]);}
   r.version=t.version;if(r.mips>1){r.dirty=true;this.generateMips(r);}
  }return r;
 }
 finishTarget(rt){if(this.preparing)return;const r=this.targets.get(rt);if(r)for(const c of r.colors)if(c.dirty&&c.mips>1)this.generateMips(c);}
 attachment(clear=false,color=true,depth=true){
  const r=this.target?this.initRenderTarget(this.target):null,colors=r?r.colors:[{gpu:this.context.getCurrentTexture(),format:this.format,mips:1}],z=r?r.depth:this.screenDepth;
  const desc={colorAttachments:colors.map(c=>({view:c.gpu.createView({dimension:'2d',baseArrayLayer:this.layer,arrayLayerCount:1,baseMipLevel:this.level,mipLevelCount:1}),loadOp:clear&&color?'clear':'load',storeOp:'store',clearValue:{r:this.clearColor.r,g:this.clearColor.g,b:this.clearColor.b,a:this.alpha}}))};
  if(z)desc.depthStencilAttachment={view:z.createView(),depthLoadOp:clear&&depth?'clear':'load',depthStoreOp:'store',depthClearValue:1};
  if(r)for(const c of colors)c.dirty=true;
  return {desc,formats:colors.map(c=>c.format),depth:!!z};
 }
 clear(color=true,depth=true){if(this.preparing)return;const {desc}=this.attachment(true,color,depth);this.command().beginRenderPass(desc).end();}
 buffer(attribute,index=false){let r=this.buffers.get(attribute);if(!r||r.version!==attribute.version){let data;if(index)data=Uint32Array.from(attribute.array);else{data=new Float32Array(attribute.count*attribute.itemSize);for(let i=0;i<attribute.count;i++)for(let j=0;j<attribute.itemSize;j++)data[i*attribute.itemSize+j]=attribute.getComponent(i,j);}if(!r||r.size<data.byteLength){r?.gpu.destroy();r={gpu:this.device.createBuffer({size:Math.max(4,data.byteLength),usage:(index?B.INDEX:B.VERTEX)|B.COPY_DST}),size:data.byteLength};}this.flush();this.device.queue.writeBuffer(r.gpu,0,data);r.version=attribute.version;this.buffers.set(attribute,r);}return r.gpu;}
 uniforms(layout,values){const data=packContext(layout,values);let b=this.uniformPool[this.uniformIndex];if(!b||b.size<data.byteLength){b?.destroy();b=this.device.createBuffer({size:data.byteLength,usage:B.STORAGE|B.COPY_DST});this.uniformPool[this.uniformIndex]=b;}this.uniformIndex++;this.device.queue.writeBuffer(b,0,data);return b;}
 pipeline(mat,mesh,attachment,asynchronous=false){
  const key=shaderKey(mat,mesh),art=this.artifacts[key];if(!art){window.__missingShader={key,vertex:mat.vertexShader,fragment:mat.fragmentShader,defines:mat.defines,instanced:!!mesh.instanceMatrix,instanceColor:!!mesh.instanceColor};throw Error('Missing CUDA graphics program '+key+' '+(mat.name||mesh.name||'unnamed'));}
  const depthTextures=new Set(Object.entries(mat.uniforms||{}).filter(([,u])=>u.value?.isDepthTexture).map(([k])=>k));
  const attrs=art[0].inputs.map(f=>[f.name,!!mesh.geometry.attributes[f.name]?.isInstancedBufferAttribute]);
  const cacheKey=JSON.stringify([key,attachment.formats,attachment.depth,!this.target,mat.side,mat.depthTest,mat.depthWrite,mat.depthFunc,mat.blending,mat.transparent,mat.blendEquation,mat.blendSrc,mat.blendDst,mat.blendSrcAlpha,mat.blendDstAlpha,mat.polygonOffset,attrs,[...depthTextures]]);
  let result=this.pipelines.get(cacheKey);if(result)return result;
  const vs=graphicsStage(art[0],'vertex',art[0].outputs,{screen:!this.target,depthTextures}),fs=graphicsStage(art[1],'fragment',art[0].outputs,{screen:!this.target,height:this.height,depthTextures});
  const attributeGroups=['vertex','instance'].map(stepMode=>({stepMode,fields:vs.attributes.filter(f=>(!!f.matrix||f.name==='instanceColor'||!!mesh.geometry.attributes[f.name]?.isInstancedBufferAttribute) ===(stepMode==='instance'))})).filter(g=>g.fields.length);
  const buffers=attributeGroups.map(g=>{let stride=0;const attributes=g.fields.map(f=>{const n=Number(f.type.slice(-1))||1,a={shaderLocation:f.location,offset:stride,format:n===1?'float32':'float32x'+n};f.components=n;f.offset=stride/4;stride+=n*4;return a;});g.stride=stride;return {arrayStride:stride,stepMode:g.stepMode,attributes};});
  const blend=mat.transparent&&mat.blending!==0?(mat.blending===2?{color:{srcFactor:'src-alpha',dstFactor:'one',operation:'add'},alpha:{srcFactor:'one',dstFactor:'one',operation:'add'}}:mat.blending===5?{color:{srcFactor:factors[mat.blendSrc]||'one',dstFactor:factors[mat.blendDst]||'one-minus-src-alpha',operation:mat.blendEquation===104?'max':mat.blendEquation===103?'min':'add'},alpha:{srcFactor:factors[mat.blendSrcAlpha??mat.blendSrc]||'one',dstFactor:factors[mat.blendDstAlpha??mat.blendDst]||'one-minus-src-alpha',operation:mat.blendEquation===104?'max':'add'}}:{color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha',operation:'add'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha',operation:'add'}}):undefined;
  const vertex=this.device.createShaderModule({code:vs.wgsl}),fragment=this.device.createShaderModule({code:fs.wgsl});
  const groupLayouts=[vs,fs].map((stage,i)=>this.device.createBindGroupLayout({entries:[{binding:0,visibility:i===0?GPUShaderStage.VERTEX:GPUShaderStage.FRAGMENT,buffer:{type:'read-only-storage'}},...stage.textureBindings.flatMap(t=>[{binding:t.textureBinding,visibility:i===0?GPUShaderStage.VERTEX:GPUShaderStage.FRAGMENT,texture:{sampleType:depthTextures.has(t.name)?'depth':'float',viewDimension:t.dimension}},{binding:t.samplerBinding,visibility:i===0?GPUShaderStage.VERTEX:GPUShaderStage.FRAGMENT,sampler:{type:depthTextures.has(t.name)?'non-filtering':'filtering'}}])]}));
  const descriptor={label:'ShoreBreak CUDA '+key,layout:this.device.createPipelineLayout({bindGroupLayouts:groupLayouts}),vertex:{module:vertex,entryPoint:'vertexMain',buffers},fragment:{module:fragment,entryPoint:'fragmentMain',targets:attachment.formats.map(format=>({format,blend}))},primitive:{topology:'triangle-list',frontFace:this.target?'cw':'ccw',cullMode:mat.side===2?'none':mat.side===1?'front':'back'}};
  if(attachment.depth)descriptor.depthStencil={format:'depth32float',depthWriteEnabled:!!mat.depthWrite,depthCompare:mat.depthTest?depthCompare[mat.depthFunc]||'less-equal':'always',depthBias:mat.polygonOffset?mat.polygonOffsetUnits:0,depthBiasSlopeScale:mat.polygonOffset?mat.polygonOffsetFactor:0};
  result={pipeline:null,vs,fs,art,attributeGroups,layouts:art.map(a=>contextLayout(a.wgsl)),depthTextures};
  if(asynchronous)result.promise=this.device.createRenderPipelineAsync(descriptor).then(p=>{result.pipeline=p;});else result.pipeline=this.device.createRenderPipeline(descriptor);
  this.pipelines.set(cacheKey,result);this.info.programs.push({name:key,type:'WebCuda',diagnostics:{runnable:true}});return result;
 }
 render(scene,camera){
  scene.updateMatrixWorld(true);camera.updateMatrixWorld();this.frustum.setFromProjectionMatrix(this.clip.multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse));const meshes=[];
  scene.traverseVisible(m=>{if(m.isMesh&&(!m.frustumCulled||this.frustum.intersectsObject(m)))meshes.push(m);});meshes.sort((a,b)=>a.renderOrder-b.renderOrder||Number(a.material.transparent)-Number(b.material.transparent));
  for(const mesh of meshes){mesh.onBeforeRender(this,scene,camera,mesh.geometry,mesh.material,null);this.draw(mesh,scene.overrideMaterial||mesh.material,camera);}
  if(!this.target)this.flush();
 }
 draw(mesh,mat,camera){
  if(this.preparing){this.pipeline(mat,mesh,this.attachment(),true);return;}
  const attachment=this.attachment(),p=this.pipeline(mat,mesh,attachment),g=mesh.geometry,values=Object.fromEntries(Object.entries(mat.uniforms||{}).map(([k,v])=>[k,v.value]));
  this.mv.multiplyMatrices(camera.matrixWorldInverse,mesh.matrixWorld);this.nm.getNormalMatrix(this.mv);Object.assign(values,{modelMatrix:mesh.matrixWorld,viewMatrix:camera.matrixWorldInverse,modelViewMatrix:this.mv,normalMatrix:this.nm,projectionMatrix:camera.projectionMatrix,cameraPosition:camera.position});
  mesh.userData.webcudaVertexBuffers??=new Map();
  const vertexBuffers=p.attributeGroups.map(group=>{
   const key=group.fields.map(f=>f.name).join(','),sources=group.fields.map(f=>f.matrix?mesh.instanceMatrix:f.name==='instanceColor'?mesh.instanceColor:g.attributes[f.name]),version=sources.map(a=>a?.version??-1).join(','),count=group.stepMode==='vertex'?(g.attributes.position?.count||Object.values(g.attributes).find(a=>!a.isInstancedBufferAttribute)?.count||3):(mesh.isInstancedMesh?mesh.count:Number.isFinite(g.instanceCount)?g.instanceCount:1);let record=mesh.userData.webcudaVertexBuffers.get(key);
   if(!record||record.version!==version||record.count!==count){const data=new Float32Array(count*group.stride/4);group.fields.forEach((f,fi)=>{const a=sources[fi];if(!a)return;for(let i=0;i<count;i++)for(let j=0;j<f.components;j++)data[i*group.stride/4+f.offset+j]=a.getComponent(Math.min(i,a.count-1),j+(f.matrix?(f.column||0)*4:0));});if(!record||record.gpu.size<data.byteLength){record?.gpu.destroy();record={gpu:this.device.createBuffer({size:Math.max(4,data.byteLength),usage:B.VERTEX|B.COPY_DST})};}this.flush();this.device.queue.writeBuffer(record.gpu,0,data);record.version=version;record.count=count;mesh.userData.webcudaVertexBuffers.set(key,record);}return record.gpu;
  });
  const index=g.index?this.buffer(g.index,true):null,bindGroups=[],sampled=[p.vs,p.fs].map(a=>a.textureBindings.map(t=>this.texture(values[t.name],t.dimension,p.depthTextures.has(t.name))));
  for(const [stage,a]of [[0,p.vs],[1,p.fs]]){
   const entries=[{binding:0,resource:{buffer:this.uniforms(p.layouts[stage],values)}}];
   for(const [i,t]of a.textureBindings.entries()){const texture=sampled[stage][i];entries.push({binding:t.textureBinding,resource:texture.gpu.createView({dimension:t.dimension})},{binding:t.samplerBinding,resource:texture.sampler});}
   // Auto layouts omit resources unused by the selected shader entry.
   const usedBindings=new Set([...a.wgsl.matchAll(new RegExp('@group\\('+stage+'\\) @binding\\((\\d+)\\)','g'))].map(m=>+m[1]));
   bindGroups.push(this.device.createBindGroup({layout:p.pipeline.getBindGroupLayout(stage),entries:entries.filter(e=>usedBindings.has(e.binding))}));
  }
  const pass=this.command().beginRenderPass(attachment.desc);pass.setPipeline(p.pipeline);pass.setBindGroup(0,bindGroups[0]);pass.setBindGroup(1,bindGroups[1]);vertexBuffers.forEach((b,i)=>pass.setVertexBuffer(i,b));
  const start=g.drawRange.start,count=Math.min(Number.isFinite(g.drawRange.count)?g.drawRange.count:Infinity,(g.index?.count||(g.attributes.position?.count||Object.values(g.attributes).find(a=>!a.isInstancedBufferAttribute)?.count||3))-start),instances=mesh.isInstancedMesh?mesh.count:Number.isFinite(g.instanceCount)?g.instanceCount:1;
  if(index){pass.setIndexBuffer(index,'uint32');pass.drawIndexed(count,instances,start);}else pass.draw(count,instances,start);pass.end();this.info.render.calls++;this.info.render.triangles+=count/3*instances;if(++this.pending>=128)this.flush();
 }
 async compileAsync(scene,camera){const jobs=[];scene.traverse(o=>{if(!o.isMesh)return;const mat=scene.overrideMaterial||o.material,art=this.artifacts[shaderKey(mat,o)];if(!art)return;const colors=art[1].outputs.length+Number(!art[1].outputs.some(f=>f.location===0));const attachment=this.target?{formats:Array.from({length:colors},(_,i)=>this.textureFormat(this.target.textures[Math.min(i,this.target.textures.length-1)])),depth:!!this.target.depthBuffer}:{formats:[this.format],depth:!!this.screenDepth};jobs.push(this.pipeline(mat,o,attachment,true).promise);});await Promise.all(jobs);}
 beginCompile(){this.preparing=true;}
 async endCompile(){this.preparing=false;await Promise.all([...this.pipelines.values()].map(p=>p.promise));}
 async readRenderTargetPixelsAsync(rt,x,y,w,h,out,textureIndex=0){
  if(this.preparing)return out.fill(0);
  const r=this.initRenderTarget(rt),tex=r.colors[textureIndex],bytes=tex.format.includes('32')?4:tex.format.includes('16')?2:1,components=tex.format.startsWith('rgba')?4:tex.format.startsWith('rg')?2:1,pitch=Math.ceil(w*components*bytes/256)*256,b=this.device.createBuffer({size:pitch*h,usage:B.COPY_DST|B.MAP_READ});
  this.command().copyTextureToBuffer({texture:tex.gpu,origin:[x,y,0]},{buffer:b,bytesPerRow:pitch},[w,h]);this.flush();await b.mapAsync(GPUMapMode.READ);
  const data=new DataView(b.getMappedRange());for(let row=0;row<h;row++)for(let i=0;i<w*components;i++){
   const offset=row*pitch+i*bytes;let value;
   if(bytes===4)value=data.getFloat32(offset,true);
   else if(bytes===1)value=data.getUint8(offset);
   else{const bits=data.getUint16(offset,true),sign=bits&32768?-1:1,e=(bits>>10)&31,m=bits&1023;value=sign*(e===0?m*2**-24:e===31?m?NaN:Infinity:(1+m/1024)*2**(e-15));}
   out[row*w*components+i]=value;
  }b.unmap();b.destroy();return out;
 }
 generateMips(r){
  if(!r.dirty||r.mips<=1)return;
  this.mipPipelines??=new Map();let p=this.mipPipelines.get(r.format);
  if(!p){const art=this.artifacts[this.mipKey],v=graphicsStage(art[0],'vertex'),f=graphicsStage(art[1],'fragment',art[0].outputs),pipeline=this.device.createRenderPipeline({layout:'auto',vertex:{module:this.device.createShaderModule({code:v.wgsl}),entryPoint:'vertexMain',buffers:v.attributes.map(a=>({arrayStride:Number(a.type.slice(-1))*4,attributes:[{shaderLocation:a.location,offset:0,format:'float32x'+a.type.slice(-1)}]}))},fragment:{module:this.device.createShaderModule({code:f.wgsl}),entryPoint:'fragmentMain',targets:[{format:r.format}]},primitive:{topology:'triangle-list',cullMode:'none'}}),uniforms=art.map(a=>this.device.createBuffer({size:contextLayout(a.wgsl).size,usage:B.STORAGE})),zero=this.device.createBuffer({size:64,usage:B.VERTEX}),sampler=this.device.createSampler({minFilter:'linear',magFilter:'linear'});p={pipeline,uniforms,zero,sampler,v};this.mipPipelines.set(r.format,p);}
  for(let layer=0;layer<r.layers;layer++)for(let level=1;level<r.mips;level++){
   const src=r.gpu.createView({dimension:'2d',baseArrayLayer:layer,arrayLayerCount:1,baseMipLevel:level-1,mipLevelCount:1}),dst=r.gpu.createView({dimension:'2d',baseArrayLayer:layer,arrayLayerCount:1,baseMipLevel:level,mipLevelCount:1});
   const g0=this.device.createBindGroup({layout:p.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:p.uniforms[0]}}]}),g1=this.device.createBindGroup({layout:p.pipeline.getBindGroupLayout(1),entries:[{binding:0,resource:{buffer:p.uniforms[1]}},{binding:2,resource:src},{binding:3,resource:p.sampler}]});
   const pass=this.command().beginRenderPass({colorAttachments:[{view:dst,loadOp:'clear',storeOp:'store',clearValue:[0,0,0,0]}]});pass.setPipeline(p.pipeline);pass.setBindGroup(0,g0);pass.setBindGroup(1,g1);p.v.attributes.forEach((_,i)=>pass.setVertexBuffer(i,p.zero));pass.draw(3);pass.end();
  }r.dirty=false;
 }
 getContext(){return {finish:()=>this.flush(),flush:()=>this.flush(),readPixels:()=>this.flush(),getExtension:()=>null,isContextLost:()=>false};}
}
