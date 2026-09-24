import {chromium} from 'playwright-core';
import {createServer} from 'vite';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
await mkdir('.qa/graphics',{recursive:true});
const manifest=JSON.parse(await readFile('tools/graphics-metadata.json','utf8')).stages;
const server=await createServer({logLevel:'error',server:{port:0}});await server.listen();
const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:['--enable-unsafe-webgpu']});
try{
 const page=await browser.newPage();page.on('console',m=>console.log(m.text()));await page.goto(`http://localhost:${server.httpServer.address().port}/faithful/graphics/index.json`);
 const report=await page.evaluate(async manifest=>{
  const {graphicsStage}=await import('/src/webcuda/graphics-stage.js');const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'}),device=await adapter.requestDevice({requiredFeatures:['shader-f16','float32-filterable']});
  const errors=[];
  for(const id of Object.keys(manifest)){
   const [v,f]=await Promise.all(['vertex','fragment'].map(s=>fetch('/faithful/graphics/'+id+'-'+s+'.json').then(r=>r.json())));
   for(const [stage,a]of [['vertex',v],['fragment',f]]){
    const result=graphicsStage(a,stage,v.outputs),module=device.createShaderModule({code:result.wgsl});
    const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>({message:m.message,line:m.lineNum,source:result.wgsl.split('\n').slice(Math.max(0,m.lineNum-2),m.lineNum+1).join('\n')}));
    if(messages.length){errors.push({id,stage,messages});console.log(id,stage,JSON.stringify(messages));}
   }
  }
  device.destroy();return {programs:Object.keys(manifest).length,errors};
 },manifest);
 await writeFile('.qa/graphics/validation.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(report.errors.length)process.exitCode=1;
}finally{await browser.close();await server.close();}
