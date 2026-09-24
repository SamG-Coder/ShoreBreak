// Independent GPU oracle: original GLSL against the single-file CUDA port.
import {readFile,writeFile} from 'node:fs/promises';
import {chromium} from 'playwright-core';
import {createServer} from 'vite';
globalThis.location={search:'?explore'};
const {Schedule}=await import('../src/core/schedule.js');
const {waterLookups}=await import('../src/water/WaterLookups.js');
const original=JSON.parse(await readFile('.qa/faithful/original-waves.json','utf8'));
const lip=process.argv.includes('--lip'),entry=lip?'evaluateOriginalLip':'evaluateOriginalWaves',vectors=lip?3:12,eventIndex=1;
const artifact=JSON.parse(await readFile(`.qa/faithful/${entry}.json`,'utf8'));
const cu=await readFile('ShoreBreak.cu','utf8');
const kernelSource=cu.slice(cu.indexOf('__global__ void '+entry));
let body=kernelSource.slice(kernelSource.indexOf(' SBUniforms sb=uniforms[0];')).split('\n}')[0];
body=body.replace('SBUniforms sb=uniforms[0];','').replace('queries[i]','texelFetch(uQueries,ivec2(i,0),0)').replaceAll('events,lookup,sb,','').replace(/make_float([234])/g,'vec$1').replace(/\bfloat([234])\b/g,'vec$1').replace(/(\d)f\b/g,'$1').replace(new RegExp(`results\\[i\\*${vectors}(?:\\+(\\d+))?\\]`,'g'),(_,n)=>`r[${n||0}]`);
const fragment='#version 300 es\nprecision highp float;precision highp int;\n'+original.glsl+`\nuniform sampler2D uQueries;uniform int eventIndex;out vec4 outputColor;void main(){int i=int(gl_FragCoord.x);vec4 r[${vectors}];`+body+'outputColor=r[int(gl_FragCoord.y)];}';
const travel=waterLookups();
const schedule=new Schedule().pack(3.1);
const events=Array.from('ABCDEFG').flatMap(c=>Array.from(schedule[c]));
const values={uSwellMinTime:travel.minTime,uEvtCount:schedule.count,uTime:3.1,uInjMass:2.4,uInjSpeed:1,uInjJ:[.17,.15,2.875,3.55],uInjR:[.55,.3,1.2,2.2],uInjL:[.05,.025,.012,.15],uFoamSrc:[6,.43,2,1],uFoamSpill:24,uFoamSpillK:[1.5,1.8]};
const packed=new ArrayBuffer(112),dv=new DataView(packed);let offset=0;
for(const {name,type} of original.fields){const n=type.startsWith('vec')?+type.slice(3):1,alignment=n===1?4:n===2?8:16;offset=Math.ceil(offset/alignment)*alignment;for(const [j,v] of (Array.isArray(values[name])?values[name]:[values[name]]).entries())dv[type==='int'?'setInt32':'setFloat32'](offset+j*4,v,true);offset+=n*4;}
const queries=[];for(let i=0;i<4096;i++)queries.push(i%7===0?-64+(i%32)*128/31:-9+(i%127)*18/126,-15+(i%59)*19/58,schedule.active[eventIndex].t0-1.6+(i%97)*.045,(i%113)/112*(lip?5.999:1));
const server=await createServer({logLevel:'error',server:{port:0}});await server.listen();
const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:['--enable-unsafe-webgpu']});
try{
 const page=await browser.newPage();page.on('console',m=>console.log(m.text()));page.setDefaultTimeout(60000);await page.goto(`http://localhost:${server.httpServer.address().port}/.qa/faithful/original-waves.json`);
 const report=await page.evaluate(async({fragment,artifact,events,values,fields,packed,queries,lookup,vectors,eventIndex})=>{
  const {GpuRuntime}=await import('/vendor/cuda-webshader/runtime/runtime.js');
  console.log('Requesting WebGPU');const runtime=await GpuRuntime.create();console.log('Compiling CUDA pipeline');const kernel=await runtime.kernel(artifact);console.log('CUDA pipeline ready');const count=queries.length/4;
  const buffers={events:runtime.createBuffer(new Float32Array(events)),lookup:runtime.createBuffer(new Float32Array(lookup)),uniforms:runtime.createBuffer(new Uint8Array(packed)),queries:runtime.createBuffer(new Float32Array(queries)),results:runtime.createBuffer(count*vectors*16)};
  runtime.batch().dispatch(kernel.bind(buffers,{count,eventIndex}),[Math.ceil(count/64),1,1]).submit();
  const actual=await runtime.read(buffers.results);
  const canvas=document.createElement('canvas');const gl=canvas.getContext('webgl2');if(!gl.getExtension('EXT_color_buffer_float')||!gl.getExtension('OES_texture_float_linear'))throw Error('Float rendering/sampling unavailable');
  function shader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
  const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,'#version 300 es\nvoid main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.0-1.0,0,1);}'));gl.attachShader(p,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));gl.useProgram(p);
  function texture(w,h,data,unit,linear=false){gl.activeTexture(gl.TEXTURE0+unit);const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,w,h,0,gl.RGBA,gl.FLOAT,data);for(const k of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,k,linear?gl.LINEAR:gl.NEAREST);for(const k of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,k,gl.CLAMP_TO_EDGE);return t;}
  const target=texture(count,vectors,null,2);const fb=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,fb);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,target,0);if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('Framebuffer incomplete');
  texture(4096,4,new Float32Array(lookup),0,true);gl.uniform1i(gl.getUniformLocation(p,'uWaterLookup'),0);texture(count,1,new Float32Array(queries),1);gl.uniform1i(gl.getUniformLocation(p,'uQueries'),1);gl.uniform1i(gl.getUniformLocation(p,'eventIndex'),eventIndex);
  for(const {name,type} of fields){const loc=gl.getUniformLocation(p,name),v=values[name];if(type==='int')gl.uniform1i(loc,v);else if(type==='float')gl.uniform1f(loc,v);else gl[`uniform${type.slice(3)}fv`](loc,v);}
  for(let j=0;j<7;j++)gl.uniform4fv(gl.getUniformLocation(p,'uEvt'+'ABCDEFG'[j]+'[0]'),events.slice(j*24,(j+1)*24));
  gl.viewport(0,0,count,vectors);gl.drawArrays(gl.TRIANGLES,0,3);const expected=new Float32Array(count*vectors*4);gl.readPixels(0,0,count,vectors,gl.RGBA,gl.FLOAT,expected);if(gl.getError())throw Error('WebGL failure');
  const maxErrors=Array(vectors*4).fill(0),worst=[];let failures=0;
  for(let i=0;i<count;i++)for(let k=0;k<vectors*4;k++){
   const a=actual[i*vectors*4+k],b=expected[(Math.floor(k/4)*count+i)*4+k%4],error=Math.abs(a-b);maxErrors[k]=Math.max(maxErrors[k],error);
   let tolerance=2e-4+Math.abs(b)*2e-4;
   if(vectors===12&&k>=40&&k<44){
    // Direct lookup diagnostics: the original texture unit quantizes the linear
    // interpolation weight. Bound it by half an 8-bit step plus two float32
    // coordinate ULPs at column 4095; keep the CUDA interpolation full precision.
    const column=Math.min(4095,Math.max(0,(queries[i*4+1]+900)/899*4095)),ix=Math.floor(column),c=k-40;
    const slope=Math.abs(lookup[Math.min(ix+1,4095)*4+c]-lookup[ix*4+c]);
    tolerance=slope*(1/512+1/2048)+2e-7*Math.max(1,Math.abs(b));
   }
   if(!Number.isFinite(a)||!Number.isFinite(b)||error>tolerance){failures++;if(worst.length<15)worst.push({query:queries.slice(i*4,i*4+4),channel:k,actual:a,expected:b,error,tolerance});}
  }
  return{count,failures,maxErrors,worst};
 },{fragment,artifact,events,values,fields:original.fields,packed:Array.from(new Uint8Array(packed)),queries,lookup:Array.from(travel.texture.image.data),vectors,eventIndex});
 await writeFile(`.qa/faithful/${lip?'lip-parity':'parity'}.json`,JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(report.failures)process.exitCode=1;
}finally{await browser.close();await server.close();}
