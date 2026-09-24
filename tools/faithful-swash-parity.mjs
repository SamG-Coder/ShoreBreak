import {readFile,writeFile} from 'node:fs/promises';
import {chromium} from 'playwright-core';
import {createServer} from 'vite';
import {originalShaders} from './original-shaders.mjs';
globalThis.location={search:'?explore'};
const {Schedule}=await import('../src/core/schedule.js'),{waterLookups}=await import('../src/water/WaterLookups.js');
const {INJ,SWE,FOAM_SIM,FAR}=await originalShaders('src/swash/SwashSim.js',['INJ','SWE','FOAM_SIM','FAR']);
const wave=JSON.parse(await readFile('.qa/faithful/original-waves.json','utf8')),passes=JSON.parse(await readFile('.qa/faithful/swash-passes.json','utf8'));
const artifacts={};for(const p of passes)artifacts[p.entry]=JSON.parse(await readFile(`.qa/faithful/${p.entry}.json`,'utf8'));
const travel=waterLookups(),schedule=new Schedule().pack(3.1),events=[...'ABCDEFG'].flatMap(c=>Array.from(schedule[c]));
const values={uSwellMinTime:travel.minTime,uEvtCount:schedule.count,uTime:3.1,uInjMass:INJ.mass,uInjSpeed:INJ.speed,uInjJ:INJ.jet,uInjR:INJ.roller,uInjL:INJ.lobes,uFoamSrc:[6,.43,2,1],uFoamSpill:24,uFoamSpillK:[1.5,1.8],
 uRockDomain:[0,0,0,0],uDom:[-1.44,-1.4,.03,.03],uN:[96,64],uRelief:SWE.relief,uDtS:1/120,uManning:SWE.manning,uFricH:SWE.fricH,uInfil:SWE.infiltration,uInfZ:SWE.infZ,uInfLow:SWE.infLow,uThinN:SWE.thinN,uRetain:SWE.retain,uEdgeNudge:0,
 uFoamK:FOAM_SIM.k,uFoamS:FOAM_SIM.s,uFoamM:FOAM_SIM.m,uFoamK2:FOAM_SIM.k2,uFoamS2:FOAM_SIM.s2,uFoamRMax:FOAM_SIM.rMax,uFoamCap:FOAM_SIM.cap,uFoamPatch:FOAM_SIM.patch,uResetA:0,uResetB:0,uFilmTau:2.6,
 uSwashFarMap:[0,FAR.L,FAR.L*Math.asinh(FAR.range/FAR.L),1],uSwashFarMapZ:[FAR.z0,FAR.dz,64,96],uSwashFarWin:[0,5.8,8.8,1],uSwashFocus:[0,0],uSwashFarK:FAR.k,uSwashFarK2:[FAR.k2[0],FAR.k2[1],2.6,FAR.k2[3]],uFarSlots:schedule.count,uShift:5,uMode:0};
for(const p of passes)for(const f of p.fields)if(!(f.name in values))throw Error('Missing original parameter '+f.name);
const server=await createServer({logLevel:'error',server:{port:0}});await server.listen();const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:['--enable-unsafe-webgpu']});
try{
 const page=await browser.newPage();page.on('console',m=>console.log(m.text()));await page.goto(`http://localhost:${server.httpServer.address().port}/.qa/faithful/swash-passes.json`);
 const report=await page.evaluate(async({passes,artifacts,values,events,wave,lookup})=>{
  const {GpuRuntime}=await import('/vendor/cuda-webshader/runtime/runtime.js'),{OriginalShader,compare}=await import('/tools/faithful-gpu-harness.js');const rt=await GpuRuntime.create();
  function pack(fields){let offset=0,maxAlign=4;const layout=fields.map(({type,name})=>{const n=type.includes('vec')?+type.slice(-1):1,a=n===1?4:n===2?8:16;maxAlign=Math.max(maxAlign,a);offset=Math.ceil(offset/a)*a;const f={type,name,offset,n};offset+=n*4;return f;});const bytes=new Uint8Array(Math.max(4,Math.ceil(offset/maxAlign)*maxAlign)),dv=new DataView(bytes.buffer);for(const f of layout){const vs=Array.isArray(values[f.name])?values[f.name]:[values[f.name]];vs.forEach((v,j)=>dv[f.type==='int'?'setInt32':'setFloat32'](f.offset+j*4,v,true));}return bytes;}
  const eventsBuf=rt.createBuffer(new Float32Array(events)),lookupBuf=rt.createBuffer(new Float32Array(lookup)),uniforms=rt.createBuffer(pack(wave.fields));
  const n=96*64,empty=new Float32Array(n*4),textures={};for(const name of new Set(passes.flatMap(p=>p.textures)))textures[name]={data:empty,width:96,height:64,linear:true};
  textures.uWaterLookup={data:new Float32Array(lookup),width:4096,height:4,linear:true};
  const report=[];
  for(const p of passes){
   console.log('Checking '+p.entry);const w=96,h=p.entry==='originalFarParam'?12:64;
   const oracle=new OriginalShader(p.original,w,h,p.outputs),kernel=await rt.kernel(artifacts[p.entry]);
   const info=[],parts=[];let offset=0;const samplers={};for(const name of p.textures){const t=textures[name];info.push(offset/4,t.width,t.height,0);parts.push(t.data);offset+=t.data.length;samplers[name]=t;}
   const data=new Float32Array(Math.max(4,offset));offset=0;for(const part of parts){data.set(part,offset);offset+=part.length;}
   const allocated=[rt.createBuffer(data),rt.createBuffer(new Float32Array(info.length?info:[0,1,1,0])),rt.createBuffer(pack(p.fields)),rt.createBuffer(w*h*16*p.outputs)];
   rt.batch().dispatch(kernel.bind({events:eventsBuf,lookup:lookupBuf,uniforms,data:allocated[0],descriptors:allocated[1],params:allocated[2],output:allocated[3]},{width:w,height:h}),[Math.ceil(w/8),Math.ceil(h/8),1]).submit();const actual=await rt.read(allocated[3]);
   const u={};for(const {name,type} of [...wave.fields,...p.fields])u[name]={type:type==='int'?'uniform1i':type==='float'?'uniform1f':`uniform${type.slice(-1)}${type.startsWith('i')?'iv':'fv'}`,value:values[name]};for(let j=0;j<7;j++)u['uEvt'+'ABCDEFG'[j]+'[0]']={type:'uniform4fv',value:events.slice(j*24,(j+1)*24)};
   samplers.uWaterLookup=textures.uWaterLookup;const expected=oracle.run(samplers,u);
   for(let i=0;i<p.outputs;i++)report.push({pass:p.entry,output:i,...compare(actual.subarray(i*w*h*4,(i+1)*w*h*4),expected[i],2e-4,2e-4)});
   const tex=(data)=>({data,width:w,height:h,linear:true});
   if(p.entry==='originalBedInit')textures.uBed=tex(expected[0]);
   if(p.entry==='originalStateInit'){const state=expected[0].slice();for(let z=0;z<h;z++)for(let x=0;x<w;x++){const i=(z*w+x)*4,depth=Math.max(0,state[i]-textures.uBed.data[i]+.04*Math.exp(-((x-40)**2+(z-20)**2)/150));state[i]=textures.uBed.data[i]+depth;state[i+1]=depth*.6*Math.sin(x*.2);state[i+2]=depth*.5*Math.cos(z*.13);}textures.uState=tex(state);textures.uSrc=tex(state);}
   if(p.entry==='originalFoam'){textures.uFoamT=tex(expected[0]);textures.uFoamPrev=tex(expected[0]);textures.uUVPrev=tex(expected[1]);textures.uSrcA=tex(expected[0]);textures.uSrcB=tex(expected[1]);}
   if(p.entry==='originalWet')textures.uWetPrev=tex(expected[0]);
   if(p.entry==='originalFarParam')textures.uSwashFarParam=tex(expected[0]);
   for(const b of allocated)rt.destroyBuffer(b);
  }
  return report;
 },{passes,artifacts,values,events,wave,lookup:Array.from(travel.texture.image.data)});
 await writeFile('.qa/faithful/swash-parity.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(report.some(r=>r.failures))process.exitCode=1;
}finally{await browser.close();await server.close();}
