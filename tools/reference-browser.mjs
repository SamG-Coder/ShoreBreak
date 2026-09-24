// Render the pinned upstream revision without changing the working application.
import {execFileSync} from 'node:child_process';
import {chromium} from 'playwright-core';
import {createServer} from 'vite';
import {writeFile} from 'node:fs/promises';
const clip=process.argv.includes('--clip'),label=clip?'reference-clip':'reference';
const original=Object.fromEntries(['boot','main'].map(name=>[name,execFileSync('git',['show',`11c8c05:src/${name}.js`],{encoding:'utf8',maxBuffer:8e6})]));
const server=await createServer({logLevel:'error',plugins:[{name:'original-reference',enforce:'pre',transform(code,id){const name=id.replaceAll('\\','/').match(/\/src\/(boot|main)\.js$/)?.[1];if(name)return original[name];}}],server:{port:0}});await server.listen();
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
try{
 const page=await browser.newPage({viewport:{width:960,height:640}}),errors=[];
 page.on('pageerror',e=>{errors.push(String(e));console.error(e);});
 await page.goto(`http://localhost:${server.httpServer.address().port}/?capture&w=480&h=320${clip?'&clip':''}`);
 await page.waitForFunction(()=>window.__ready,null,{timeout:600000});
 await page.waitForTimeout(1000);
 const save=name=>page.locator('#stage canvas').screenshot({path:'.qa/faithful/'+label+name+'.png'});
 await save('');await page.evaluate(()=>window.__advance(3.15));await save('-wave');
 if(!clip)for(const [name,pos]of [['front',[0,2.5,1.6,0,-.12]],['underwater',[0,-6,-.4,0,.85]]]){
  await page.evaluate(async([x,z,eyeY,yaw,pitch])=>{const c=window.__explore;Object.assign(c.pos,{x,z});Object.assign(c,{eyeY,yaw,pitch});await window.__draw();},pos);await save('-'+name);
 }
 await writeFile('.qa/faithful/'+label+'.json',JSON.stringify({errors,marks:await page.evaluate(()=>window.__marks)},null,2));
 console.log('Original reference ready');
}finally{await browser.close();await server.close();}
