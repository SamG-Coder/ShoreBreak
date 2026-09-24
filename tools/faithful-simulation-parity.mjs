import {readFile,writeFile} from 'node:fs/promises';
import {chromium} from 'playwright-core';
import {createServer} from 'vite';
import {originalShaders} from './original-shaders.mjs';
globalThis.location={search:'?explore'};
const ocean=await originalShaders('src/water/OceanFFT.js',['buildH0']);
const passes=JSON.parse(await readFile('.qa/faithful/original-passes.json','utf8'));
const artifacts={};for(const p of passes)artifacts[p.entry]=JSON.parse(await readFile(`.qa/faithful/${p.entry}.json`,'utf8'));
const server=await createServer({logLevel:'error',server:{port:0}});await server.listen();
const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:['--enable-unsafe-webgpu']});
try{
 const page=await browser.newPage();page.on('console',m=>console.log(m.text()));await page.goto(`http://localhost:${server.httpServer.address().port}/.qa/faithful/original-passes.json`);
 const report=await page.evaluate(async({passes,artifacts,h0,L,chop})=>{
  const {GpuRuntime}=await import('/vendor/cuda-webshader/runtime/runtime.js');const {OriginalShader,compare}=await import('/tools/faithful-gpu-harness.js');
  const rt=await GpuRuntime.create(),kernels={},oracles={},report=[];
  for(const p of passes){kernels[p.entry]=await rt.kernel(artifacts[p.entry]);oracles[p.entry]=new OriginalShader(p.original,p.width||96,p.height||64,p.entry==='originalFlux'?1:2);}
  const buf=a=>rt.createBuffer(a),field=a=>({data:a,width:768,height:256});
  const h=buf(new Float32Array(h0)),a=[buf(768*256*16),buf(768*256*16)],b=[buf(768*256*16),buf(768*256*16)];
  async function dispatch(entry,buffers,scalars,w=768,h=256){rt.batch().dispatch(kernels[entry].bind(buffers,scalars),[Math.ceil(w/8),Math.ceil(h/8),1]).submit();await rt.idle();}
  await dispatch('originalSpectrum',{uH0:h,output0:a[0],output1:a[1]},{uT:3.1,L0:L[0],L1:L[1],L2:L[2]});
  let expected=oracles.originalSpectrum.run({uH0:field(new Float32Array(h0))},{uT:{type:'uniform1f',value:3.1},uL:{type:'uniform3fv',value:L}});
  for(let j=0;j<2;j++)report.push({pass:'spectrum'+j,...compare(await rt.read(a[j]),expected[j],2e-7,2e-4)});
  let input=a,output=b;
  const isolated=[buf(768*256*16),buf(768*256*16),buf(768*256*16),buf(768*256*16)];
  for(const horiz of [1,0])for(let sub=2;sub<=256;sub*=2){
   rt.write(isolated[0],expected[0]);rt.write(isolated[1],expected[1]);
   await dispatch('originalFFT',{uIn0:isolated[0],uIn1:isolated[1],output0:isolated[2],output1:isolated[3]},{uSub:sub,uHoriz:horiz});
   await dispatch('originalFFT',{uIn0:input[0],uIn1:input[1],output0:output[0],output1:output[1]},{uSub:sub,uHoriz:horiz});
   expected=oracles.originalFFT.run({uIn0:field(expected[0]),uIn1:field(expected[1])},{uSub:{type:'uniform1i',value:sub},uHoriz:{type:'uniform1i',value:horiz}});
   for(let j=0;j<2;j++){
    report.push({pass:`isolated fft ${horiz}/${sub}/${j}`,...compare(await rt.read(isolated[j+2]),expected[j],2e-7,2e-6)});
    // Spectrum phase rounding accumulates through 16 FFT passes. The slope bound is
    // 8 microradians; displacement is checked separately below at 2 micrometres.
    report.push({pass:`fft ${horiz}/${sub}/${j}`,...compare(await rt.read(output[j]),expected[j],8e-6,2e-4)});
   }
   [input,output]=[output,input];
  }
  const out=[buf(256*256*16),buf(256*256*16)];
  for(let c=0;c<3;c++){
   await dispatch('originalCompose',{uIn0:input[0],uIn1:input[1],output0:out[0],output1:out[1]},{uCascade:c,uChopK:chop[c]},256,256);
   const ex=oracles.originalCompose.run({uIn0:field(expected[0]),uIn1:field(expected[1])},{uCascade:{type:'uniform1i',value:c},uChopK:{type:'uniform1f',value:chop[c]}});
   for(let j=0;j<2;j++)report.push({pass:`compose ${c}/${j}`,...compare(await rt.read(out[j]),ex[j],j===0?2e-6:8e-6,2e-4)});
  }
  console.log('FFT comparisons completed');
  const nx=96,nz=64,dx=.03,dz=.03,bed=new Float32Array(nx*nz*4),state=new Float32Array(bed.length);
  const B=(x,z)=>-.18+.004*z+.014*Math.sin(x*.37)*Math.cos(z*.29);
  for(let z=0;z<nz;z++)for(let x=0;x<nx;x++){const i=(z*nx+x)*4,v=B(x,z);bed.set([v,Math.max(v,B(Math.min(x+1,nx-1),z)),Math.max(v,B(x,Math.min(z+1,nz-1))),0],i);const h=Math.max(0,.04*Math.exp(-((x-45)**2+(z-23)**2)/60)-v);state.set([v+h,h*.6*Math.sin(x*.13),h*.5*Math.cos(z*.21),0],i);}
  const bedBuf=buf(bed);let st=buf(state),next=buf(state.byteLength),ex=state;
  for(let n=0;n<36;n++){
   await dispatch('originalFlux',{uState:st,uBed:bedBuf,output0:next},{domX:0,domZ:-1.5,dx,dz,nx,nz,uDt:1/720},nx,nz);
   ex=oracles.originalFlux.run({uState:{data:ex,width:nx,height:nz},uBed:{data:bed,width:nx,height:nz}},{uDom:{type:'uniform4fv',value:[0,-1.5,dx,dz]},uN:{type:'uniform2iv',value:[nx,nz]},uDt:{type:'uniform1f',value:1/720}})[0];
   if(n===0||n===5||n===35)report.push({pass:`KP substep ${n+1}`,...compare(await rt.read(next),ex,2e-6,2e-4)});[st,next]=[next,st];
  }
  return report;
 },{passes,artifacts,h0:Array.from(ocean.buildH0().data),L:ocean.OCEAN_L,chop:ocean.OCEAN_CHOP});
 await writeFile('.qa/faithful/simulation-parity.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(report.some(r=>r.failures))process.exitCode=1;
}finally{await browser.close();await server.close();}
