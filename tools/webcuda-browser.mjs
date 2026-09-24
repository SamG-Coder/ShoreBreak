import { chromium } from 'playwright-core';
import { createServer, preview } from 'vite';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';

const executablePath=process.env.CHROME_PATH||['C:/Program Files/Google/Chrome/Application/chrome.exe','C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe','/usr/bin/google-chrome','/usr/bin/chromium'].find(existsSync);
if(!executablePath)throw new Error('Set CHROME_PATH to a WebGPU browser.');
const production=process.argv.includes('--dist');
const server=production?await preview({logLevel:'error',preview:{port:0}}):await createServer({logLevel:'error',server:{port:0}});
if(!production)await server.listen();
const httpServer=production?server.httpServer:server.httpServer;
const browser=await chromium.launch({executablePath,headless:true,args:['--enable-unsafe-webgpu']});
const page=await browser.newPage({viewport:{width:960,height:640}}),errors=[];
page.on('pageerror',e=>{errors.push(e.message);console.error(e.message);});
page.on('console',m=>{if(m.type()==='error'){errors.push(m.text());console.error(m.text());}});
await mkdir('captures/webcuda',{recursive:true});
try{
 await page.goto(`http://localhost:${httpServer.address().port}/?capture&w=800&h=600&nohud`);
 await page.waitForFunction(()=>window.__ready||document.getElementById('loading').classList.contains('failed'),{},{timeout:120000});
 assert.equal(await page.evaluate(()=>window.__ready),true,errors.join('\n'));
 await page.locator('#loading').waitFor({state:'hidden'});
 console.log('GPU pipelines ready');
 await page.evaluate(()=>window.__seek(3.3));
 await writeFile('captures/webcuda/opening.png',Buffer.from((await page.evaluate(()=>window.__grab())).split(',')[1],'base64'));
 const diag=await page.evaluate(()=>window.__diag());console.log(JSON.stringify(diag,null,2));
 assert.deepEqual(diag.errors,[]);assert.equal(diag.backend,'WebCuda');
 const numeric=await page.evaluate(async()=>{
  const e=window.__webcuda,s=await e.runtime.read(e.state[e.front]),o=await e.runtime.read(e.ocean);
  let min=Infinity,max=0,foam=0;for(let i=0;i<s.length;i+=4){min=Math.min(min,s[i]);max=Math.max(max,s[i]);foam=Math.max(foam,s[i+3]);}
  return{finite:s.every(Number.isFinite)&&o.every(Number.isFinite),min,max,foam,amplitude:Math.max(...o.filter((_,i)=>i%4===0))};
 });console.log('Numerics',numeric);assert(numeric.finite);assert(numeric.min>=0);assert(numeric.max<5);assert(numeric.amplitude>0&&numeric.amplitude<.5);
 await page.evaluate(()=>window.__setCamera([0,-.5,-9],[0,-.15]));
 await writeFile('captures/webcuda/underwater.png',Buffer.from((await page.evaluate(()=>window.__grab())).split(',')[1],'base64'));
 await page.evaluate(()=>window.__setCamera([0,3.2,12],[Math.PI,.12]));
 await writeFile('captures/webcuda/inland.png',Buffer.from((await page.evaluate(()=>window.__grab())).split(',')[1],'base64'));
 const performanceResult=await page.evaluate(async()=>{
  const e=window.__webcuda;e.runtime.write(e.camera,new Float32Array([-34,2.594875,6.8,0,1.64061,-.0733,0,0]));
  const samples=[];for(let i=0;i<12;i++){const start=performance.now();e.frame(1/60,{lookX:0,lookY:0,keys:0,steady:true});await e.runtime.idle();samples.push(performance.now()-start);}samples.sort((a,b)=>a-b);return{medianMs:samples[6],p95Ms:samples[11]};
 });console.log('800x600 frame timing',performanceResult);
 const movement=await page.evaluate(async()=>{
  const e=window.__webcuda;await e.reset();const before=Array.from(await e.runtime.read(e.camera));
  for(let i=0;i<60;i++){const b=e.runtime.batch();e.invoke(b,'moveCamera',{camera:e.camera,water:e.water},{dt:1/60,lookX:0,lookY:0,keys:1,reset:0,steady:1,time:e.time},[1,1,1]);b.submit();}
  await e.runtime.idle();const after=Array.from(await e.runtime.read(e.camera));
  e.resize(641,359);e.draw();await e.runtime.idle();const pixels=await e.runtime.read(e.pixels,Uint32Array);let alpha=true;for(let y=0;y<359;y++)for(let x=0;x<641;x++)if(pixels[y*e.stride+x]>>>24!==255)alpha=false;
  e.resize(800,600);return{distance:Math.hypot(after[0]-before[0],after[2]-before[2]),finite:after.every(Number.isFinite),alpha};
 });assert(movement.distance>1.9&&movement.distance<2.5);assert(movement.finite&&movement.alpha);console.log('Movement and unaligned resize',movement);
 const lake=await page.evaluate(async()=>{
  const e=window.__webcuda;await e.reset();
  // Disable boundary forcing and infiltration is zero below still-water level.
  for(let i=0;i<240;i+=60){const b=e.runtime.batch();e.advanceSteps(b,60,0);b.submit();await e.runtime.idle();}
  const s=await e.runtime.read(e.state[e.front]),bed=await e.runtime.read(e.bed);let momentum=0,heightError=0;
  for(let i=0;i<s.length;i+=4){momentum=Math.max(momentum,Math.abs(s[i+1]),Math.abs(s[i+2]));heightError=Math.max(heightError,Math.abs(s[i]-Math.max(-bed[i],0)));}
  return{momentum,heightError,finite:s.every(Number.isFinite)};
 });console.log('Lake at rest',lake);assert(lake.finite);assert(lake.momentum<1e-4);assert(lake.heightError<1e-4);
 await page.evaluate(()=>window.__seek(30));
 const longRun=await page.evaluate(async()=>{const e=window.__webcuda,s=await e.runtime.read(e.state[e.front]);let min=Infinity,max=0;for(let i=0;i<s.length;i+=4){min=Math.min(min,s[i]);max=Math.max(max,s[i]);}return{finite:s.every(Number.isFinite),min,max,errors:[...e.errors]};});
 console.log('30-second stability',longRun);assert(longRun.finite&&longRun.min>=0&&longRun.max<5);assert.deepEqual(longRun.errors,[]);
 await page.evaluate(()=>window.__setCamera([0,1.6,2],[0,-.15]));
 await writeFile('captures/webcuda/breakers.png',Buffer.from((await page.evaluate(()=>window.__grab())).split(',')[1],'base64'));
 await page.goto(`http://localhost:${httpServer.address().port}/`);
 await page.waitForFunction(()=>window.__ready,{},{timeout:120000});
 await page.waitForFunction(()=>window.__webcuda.frameCount>12);
 await page.keyboard.press('p');
 const pausedAt=await page.evaluate(()=>({time:__webcuda.time,frames:__webcuda.frameCount}));
 await page.waitForFunction(f=>__webcuda.frameCount>f+8,pausedAt.frames);
 assert.equal(await page.evaluate(()=>__webcuda.time),pausedAt.time);
 await page.keyboard.press('r');await page.keyboard.press('r');
 await page.waitForFunction(()=>__webcuda.time===0&&__webcuda.frameCount>10);
 await page.keyboard.press('c');assert.equal(await page.locator('#btn-crouch').getAttribute('aria-pressed'),'true');
 await page.keyboard.press('c');
 await page.keyboard.down('w');const startPosition=await page.evaluate(async()=>Array.from(await __webcuda.runtime.read(__webcuda.camera)));
 const f=await page.evaluate(()=>__webcuda.frameCount);await page.waitForFunction(f=>__webcuda.frameCount>f+25,f);await page.keyboard.up('w');
 const endPosition=await page.evaluate(async()=>Array.from(await __webcuda.runtime.read(__webcuda.camera)));
 assert(Math.hypot(endPosition[0]-startPosition[0],endPosition[2]-startPosition[2])>.1);
 await page.selectOption('#quality','low');await page.selectOption('#resolution','1.5');
 await page.waitForFunction(()=>__viewportInfo.quality==='low'&&__viewportInfo.resolution===1.5);
 assert.deepEqual(await page.evaluate(()=>__webcuda.errors),[]);
 const ui={pause:true,queuedReset:true,crouch:true,walking:true,qualityAndResolution:true};
 console.log('Live controls',ui);
 await writeFile('captures/webcuda/diagnostics.json',JSON.stringify({diag,numeric,performanceResult,movement,lake,longRun,ui,errors},null,2));
 assert.deepEqual(errors,[]);console.log('PASS WebCuda browser smoke and numeric checks');
}finally{await browser.close();if(production)await new Promise(resolve=>server.httpServer.close(resolve));else await server.close();}
