import {chromium} from 'playwright-core';
import {createServer} from 'vite';
import {writeFile} from 'node:fs/promises';
const server=await createServer({logLevel:'error',server:{port:0}});await server.listen();
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:['--enable-unsafe-webgpu']});
try{
 const page=await browser.newPage();await page.goto(`http://localhost:${server.httpServer.address().port}/faithful/graphics/index.json`);
 const report=await page.evaluate(async()=>{
  const T=await import('/node_modules/three/src/Three.Core.js'),{WebCudaRenderer}=await import('/src/webcuda/renderer.js'),{FullscreenPass,makeShader}=await import('/src/core/gpu.js'),{BAKE_STONES,bakeUniforms}=await import('/src/beach/pebbleBake.js'),{OriginalShader,compare}=await import('/tools/faithful-gpu-harness.js');
  const renderer=await WebCudaRenderer.create(),w=128,h=128,uniforms={...bakeUniforms(),uZero:{value:0}},oracle=new OriginalShader(BAKE_STONES,w,h,2),values={};
  for(const [key,u]of Object.entries(uniforms)){const array=Array.isArray(u.value)?u.value:null;values[key+(array&&typeof array[0]==='object'?'[0]':'')]={type:key==='uZero'?'uniform1i':array&&typeof array[0]==='number'?'uniform1fv':'uniform4fv',value:array?array.flatMap(x=>x.toArray?x.toArray():x):u.value.toArray?u.value.toArray():u.value};}
  const expected=oracle.run({},values),target=new T.WebGLRenderTarget(w,h,{count:2,type:T.FloatType,depthBuffer:false}),pass=new FullscreenPass(makeShader(BAKE_STONES,uniforms));pass.render(renderer,target);await renderer.idle();
  const report=[];for(let i=0;i<2;i++){const actual=new Float32Array(w*h*4);await renderer.readRenderTargetPixelsAsync(target,0,0,w,h,actual,i);report.push({pass:'original pebble bake',output:i,...compare(actual,expected[i],2e-4,2e-4)});}return report;
 });
 await writeFile('.qa/faithful/graphics-material-parity.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(report.some(r=>r.failures))process.exitCode=1;
}finally{await browser.close();await server.close();}
