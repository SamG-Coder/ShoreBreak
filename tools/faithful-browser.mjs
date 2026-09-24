import {chromium} from 'playwright-core';
import {createServer,preview} from 'vite';
import {writeFile,mkdir} from 'node:fs/promises';
const clip=process.argv.includes('--clip'),dist=process.argv.includes('--dist'),label=clip?'clip':'browser';
await mkdir('.qa/faithful',{recursive:true});
const server=dist?await preview({logLevel:'error',preview:{port:0}}):await createServer({logLevel:'error',server:{port:0}});if(!dist)await server.listen();
const browser=await chromium.launchPersistentContext('.qa/faithful-'+label+'-profile',{executablePath:process.env.CHROME_PATH||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,viewport:{width:960,height:640},args:['--enable-unsafe-webgpu']});
try{
 const page=await browser.newPage(),errors=[];
 page.on('pageerror',e=>{console.error(e.stack);errors.push(String(e));});page.on('console',m=>{if(m.type()==='error'){errors.push(m.text());console.error(m.text());}});
 await page.addInitScript(()=>{window.__webglCalls=[];const original=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(type,...args){if(type.includes('webgl'))window.__webglCalls.push(type);return original.call(this,type,...args);};});
 await page.goto(`http://localhost:${server.httpServer.address().port}/?capture&w=480&h=320${clip?'&clip':''}`);
 const progress=setInterval(async()=>{try{console.log(await page.evaluate(()=>({loading:document.getElementById('load-status')?.textContent,percent:document.getElementById('load-percent')?.textContent,marks:window.__marks?.at(-1)})));}catch{}},30000);
 try{await page.waitForFunction(()=>window.__ready||document.getElementById('loading')?.classList.contains('failed'),null,{timeout:600000});}finally{clearInterval(progress);}
 await page.waitForTimeout(1000);
 const state=await page.evaluate(()=>({ready:window.__ready,status:document.getElementById('load-status')?.textContent,programs:window.__scene?.renderer.info.programs.length,errors:window.__scene?.renderer.errors,marks:window.__marks,webglCalls:window.__webglCalls}));
 console.log(JSON.stringify(state));
 if(!state.ready)throw Error('Application did not become ready');
 const save=name=>page.locator('#stage canvas').screenshot({path:`.qa/faithful/${label}${name}.png`});
 await save('');
 await page.evaluate(()=>window.__advance(3.15));await save('-wave');
 const stats=await page.evaluate(()=>window.__sweDebug());delete stats.img;
 await writeFile(`.qa/faithful/${label}-state.json`,JSON.stringify(stats,null,2));if(stats.nan)throw Error('Nonfinite shallow water');
 if(!clip){
  for(const [name,pos]of [['front',[0,2.5,1.6,0,-.12]],['underwater',[0,-6,-.4,0,.85]]]){
   await page.evaluate(async([x,z,eyeY,yaw,pitch])=>{const c=window.__explore;Object.assign(c.pos,{x,z});Object.assign(c,{eyeY,yaw,pitch});await window.__draw();},pos);await save('-'+name);
  }
 }
 const timings=await page.evaluate(()=>window.__bench(5));
 await page.evaluate(async()=>{window.__scene.renderer.setSize(641,359);await window.__draw();});
 await page.evaluate(()=>window.__scene.renderer.idle());
 const gpuErrors=await page.evaluate(()=>window.__scene.renderer.errors);
 await writeFile(`.qa/faithful/${label}.json`,JSON.stringify({state,errors,gpuErrors,timings,dist},null,2));
 if(errors.length||gpuErrors.length||state.webglCalls.length)throw Error('Browser GPU validation failed');
}finally{await browser.close();await server.close();}
