import { GpuRuntime } from '../../vendor/cuda-webshader/runtime/runtime.js';

export const SIZE = Object.freeze({ocean:128,swashX:512,swashZ:192,waterX:1024,waterZ:512});
const groups = (w,h) => [Math.ceil(w/8),Math.ceil(h/8),1];
const base = import.meta.env.BASE_URL;
async function json(path) {
  const response = await fetch(base+path);
  if (!response.ok) throw new Error(`Cannot load ${path}: HTTP ${response.status}`);
  return response.json();
}
async function loadMaterials() {
  const result = new Uint32Array(349525*2);
  const paths = ['assets/coast-r8/sand-diff.webp','assets/coast-r8/rock-diff.webp'];
  for (let i=0;i<paths.length;i++) {
    const response = await fetch(base+paths[i]);
    if (!response.ok) throw new Error(`Cannot load ${paths[i]}`);
    const bitmap = await createImageBitmap(await response.blob());
    const canvas = new OffscreenCanvas(512,512),ctx=canvas.getContext('2d',{willReadFrequently:true});
    ctx.drawImage(bitmap,0,0,512,512);bitmap.close();
    const rgba=ctx.getImageData(0,0,512,512).data;
    result.set(new Uint32Array(rgba.buffer),i*349525);
  }
  return result;
}

export class ShoreBreak {
  static async create(canvas,progress=()=>{}) {
    progress('Opening WebCuda',8);
    const engine=new ShoreBreak(canvas);
    try { await engine.initialize(progress);return engine; }
    catch(error) {engine.dispose();throw error;}
  }
  constructor(canvas) {this.canvas=canvas;this.time=0;this.step=1/120;this.accumulator=0;this.frameCount=0;this.errors=[];this.bindings=new Map();this.disposed=false;}
  async initialize(progress) {
    this.runtime=await GpuRuntime.create({onError:error=>{this.errors.push(String(error));this.onError?.(error);}});
    this.device=this.runtime.device;this.context=this.canvas.getContext('webgpu');
    if(!this.context)throw new Error('A WebGPU canvas is required.');
    this.format=navigator.gpu.getPreferredCanvasFormat();
    this.manifest=await json('generated/manifest.json');this.kernels={};
    const artifacts=await Promise.all(this.manifest.kernels.map(async k=>[k.entry,await json(`generated/${k.entry}.json`)]));
    for(const [i,[name,artifact]] of artifacts.entries()){
      progress(`Preparing ${name}`,12+i/artifacts.length*55);
      this.kernels[name]=await this.runtime.kernel(artifact);
    }
    const alloc=(n,label)=>this.runtime.createBuffer(n,{label});
    this.h0=alloc(128*128*3*16,'spectrum');this.fftA=[alloc(128*128*3*16,'FFT A0'),alloc(128*128*3*16,'FFT A1')];
    this.fftB=[alloc(128*128*3*16,'FFT B0'),alloc(128*128*3*16,'FFT B1')];this.ocean=alloc(128*128*3*16,'ocean');
    this.bed=alloc(512*192*16,'bathymetry');this.state=[alloc(512*192*16,'swash 0'),alloc(512*192*16,'swash 1')];
    this.wetness=alloc(512*192*16,'wetness');this.water=alloc(1024*512*16,'surface');this.camera=alloc(48,'camera');
    this.particles=alloc(32768*16,'spray and bubbles');
    progress('Loading original coastal materials',70);
    this.textures=this.runtime.createBuffer(await loadMaterials(),{label:'original CC0 coastal materials'});
    const mipBatch=this.runtime.batch();let previous=0,offset=512*512;
    for(let side=256;side>=1;side/=2){this.invoke(mipBatch,'materialMip',{textures:this.textures},{side,offset,previous},groups(side,side));previous=offset;offset+=side*side;}
    mipBatch.submit();
    const sceneResponse=await fetch(base+'assets/webcuda/scene.bin');
    if(!sceneResponse.ok)throw new Error('Original scenery asset could not load.');
    this.geometry=this.runtime.createBuffer(new Float32Array(await sceneResponse.arrayBuffer()),{label:'Original ShoreBreak scenery BVH'});
    this.resize(800,600);await this.reset();progress('Coastline ready',100);
  }
  invoke(batch,name,buffers,scalars,grid,key=name) {
    let invocation=this.bindings.get(key);
    if(!invocation){invocation=this.kernels[name].bind(buffers,scalars);this.bindings.set(key,invocation);}
    else invocation.setScalars(scalars);
    batch.dispatch(invocation,grid);
  }
  resize(width,height) {
    if(this.disposed)return;
    width=Math.max(64,Math.min(Math.round(width),this.device.limits.maxTextureDimension2D));
    height=Math.max(64,Math.min(Math.round(height),this.device.limits.maxTextureDimension2D));
    if(this.width===width&&this.height===height)return;
    this.canvas.width=width;this.canvas.height=height;this.width=width;this.height=height;this.stride=Math.ceil(width/64)*64;
    if(this.hdr)this.runtime.destroyBuffer(this.hdr);if(this.pixels)this.runtime.destroyBuffer(this.pixels);
    if(this.spray)this.runtime.destroyBuffer(this.spray);
    this.hdr=this.runtime.createBuffer(width*height*16,{label:'CUDA HDR pixels'});
    this.pixels=this.runtime.createBuffer(this.stride*height*4,{label:'CUDA display pixels'});
    this.spray=this.runtime.createBuffer(width*height*4,{label:'spray coverage'});
    this.bindings.delete('renderCoast');this.bindings.delete('finishFrame');
    this.bindings.delete('projectSpray');
    this.context.configure({device:this.device,format:this.format,alphaMode:'opaque',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});
  }
  updateOcean(batch) {
    const grid=groups(384,128);
    this.invoke(batch,'evolveSpectrum',{h0:this.h0,fftA:this.fftA[0],fftB:this.fftB[0]},{time:this.time},grid);
    let source=0;
    for(const horizontal of [1,0])for(let sub=2;sub<=128;sub*=2){
      const target=1-source;
      this.invoke(batch,'fftStage',{inA:this.fftA[source],inB:this.fftB[source],outA:this.fftA[target],outB:this.fftB[target]},{sub,horizontal},grid,`fft-${source}`);
      source=target;
    }
    this.invoke(batch,'composeOcean',{inA:this.fftA[source],inB:this.fftB[source],ocean:this.ocean},{},grid);
  }
  updateSurface(batch) {
    this.invoke(batch,'buildSurface',{ocean:this.ocean,state:this.state[this.front],bed:this.bed,water:this.water},{time:this.time},groups(1024,512),`surface-${this.front}`);
  }
  advanceSteps(batch,count,forcing=1) {
    for(let i=0;i<count;i++){
      const next=1-this.front;
      this.invoke(batch,'stepSwash',{input:this.state[this.front],bed:this.bed,output:this.state[next]},{time:this.time,dt:this.step,forcing},groups(512,192),`swash-${this.front}`);
      this.front=next;this.time+=this.step;
    }
    if(count)this.invoke(batch,'updateWetness',{state:this.state[this.front],wetness:this.wetness},{dt:count*this.step},groups(512,192),`wet-${this.front}`);
  }
  async reset() {
    this.time=0;this.front=0;this.accumulator=0;
    const batch=this.runtime.batch();
    this.invoke(batch,'initializeSpectrum',{h0:this.h0},{},groups(384,128));
    this.invoke(batch,'initializeCoast',{bed:this.bed,state:this.state[0],wetness:this.wetness},{},groups(512,192));
    this.updateOcean(batch);this.updateSurface(batch);
    this.invoke(batch,'moveCamera',{camera:this.camera,water:this.water},{dt:0,lookX:0,lookY:0,keys:0,reset:1,steady:1,time:0},[1,1,1]);
    batch.submit();await this.runtime.idle();this.draw();await this.runtime.idle();
  }
  frame(dt,input,paused=false,speed=1) {
    this.accumulator+=paused?0:Math.min(dt,.1)*speed;
    const steps=Math.min(12,Math.floor((this.accumulator+1e-9)/this.step));this.accumulator-=steps*this.step;
    const batch=this.runtime.batch();
    if(steps){this.advanceSteps(batch,steps);this.updateOcean(batch);this.updateSurface(batch);}
    this.invoke(batch,'moveCamera',{camera:this.camera,water:this.water},{dt:Math.min(dt,.05),lookX:input.lookX,lookY:input.lookY,keys:input.keys,reset:0,steady:input.steady?1:0,time:this.time},[1,1,1]);
    batch.submit();this.draw();
  }
  draw() {
    const batch=this.runtime.batch(),grid=groups(this.width,this.height);
    this.invoke(batch,'renderCoast',{camera:this.camera,water:this.water,ocean:this.ocean,wetness:this.wetness,textures:this.textures,geometry:this.geometry,hdr:this.hdr},{width:this.width,height:this.height,time:this.time},grid);
    this.invoke(batch,'animateSpray',{particles:this.particles},{time:this.time},[256,1,1]);
    batch.clear(this.spray);
    this.invoke(batch,'projectSpray',{particles:this.particles,camera:this.camera,hdr:this.hdr,spray:this.spray},{width:this.width,height:this.height},[256,1,1]);
    this.invoke(batch,'finishFrame',{hdr:this.hdr,spray:this.spray,pixels:this.pixels},{width:this.width,height:this.height,stride:this.stride,bgra:this.format==='bgra8unorm'?1:0},grid);
    batch.submit();
    // The only presentation operation is a GPU-to-GPU copy of CUDA-produced pixels.
    const encoder=this.device.createCommandEncoder({label:'Present CUDA pixels'});
    encoder.copyBufferToTexture({buffer:this.pixels.gpuBuffer,bytesPerRow:this.stride*4,rowsPerImage:this.height},{texture:this.context.getCurrentTexture()},[this.width,this.height]);
    this.device.queue.submit([encoder.finish()]);this.frameCount++;
  }
  async seek(time) {
    if(!Number.isFinite(time)||time<0||time>3600)throw new RangeError('Seek time must be between 0 and 3600 seconds.');
    await this.reset();
    const count=Math.round(time/this.step);
    for(let i=0;i<count;i+=60){const batch=this.runtime.batch();this.advanceSteps(batch,Math.min(60,count-i));batch.submit();await this.runtime.idle();}
    const batch=this.runtime.batch();this.updateOcean(batch);this.updateSurface(batch);batch.submit();this.draw();await this.runtime.idle();return this.time;
  }
  async diagnostics() {
    await this.runtime.idle();
    return {backend:'WebCuda',renderer:'CUDA compute',time:this.time,frames:this.frameCount,width:this.width,height:this.height,device:this.runtime.describe(),stats:{...this.runtime.stats},errors:[...this.errors],camera:Array.from(await this.runtime.read(this.camera))};
  }
  dispose() {if(this.disposed)return;this.disposed=true;this.context?.unconfigure();this.runtime?.dispose();}
}
