import { ShoreBreak } from './engine.js';

const $=id=>document.getElementById(id),query=new URLSearchParams(location.search);
const capture=query.has('capture');document.body.classList.add('explore');
document.documentElement.style.setProperty('--scene-scale',capture?'1':String(Math.sqrt(.75)));
if(capture)document.body.classList.add('capture');
const canvas=document.createElement('canvas');canvas.setAttribute('aria-label','ShoreBreak WebCuda coastline');canvas.tabIndex=0;$('stage').append(canvas);
canvas.style.width='100%';canvas.style.height='100%';
const engine=await ShoreBreak.create(canvas,window.__loading);window.__webcuda=engine;
let running=true,paused=false,speed=1,busy=false,last=performance.now(),fps=0,frameMs=16,quality='auto',resolution=1,crouch=false;
let frameDone=Promise.resolve(),settleFrame=()=>{},operation=Promise.resolve(),pendingOperations=0;
const held=new Set();let lookX=0,lookY=0;
function fail(error){running=false;window.__loadError(error);}
engine.onError=fail;
try{quality=localStorage.getItem('shorebreak.webcuda.quality')||'auto';resolution=Number(localStorage.getItem('shorebreak.webcuda.resolution'))||1;}catch{}
if(!['auto','low','medium','high','ultra'].includes(quality))quality='auto';if(![1,1.5,2].includes(resolution))resolution=1;
$('quality').value=quality;$('resolution').value=String(resolution);
let adaptive=1;
function resize(){
  const bounds=$('stage').getBoundingClientRect();
  const area={auto:750000,low:350000,medium:650000,high:1100000,ultra:1800000}[quality];
  const ratio=Math.min(devicePixelRatio||1,Math.sqrt(area/(bounds.width*bounds.height)))*resolution*(quality==='auto'&&resolution===1?adaptive:1);
  const w=capture?Number(query.get('w')||960):Math.round(bounds.width*ratio),h=capture?Number(query.get('h')||640):Math.round(bounds.height*ratio);
  engine.resize(w,h);window.__viewportInfo={width:engine.width,height:engine.height,quality,resolution};
}
resize();addEventListener('resize',resize);
for(const id of ['quality','resolution'])$(id).onchange=()=>{quality=$('quality').value;resolution=Number($('resolution').value);adaptive=1;try{localStorage.setItem('shorebreak.webcuda.quality',quality);localStorage.setItem('shorebreak.webcuda.resolution',String(resolution));}catch{}resize();};
function pause(){paused=!paused;$('btn-pause').setAttribute('aria-label',paused?'Resume waves':'Pause waves');$('btn-pause').style.opacity=paused?'.55':'1';}
function help(){const hidden=!$('help').hidden;$('help').hidden=hidden;$('btn-help').setAttribute('aria-expanded',String(!hidden));}
function fullscreen(){(document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()).catch(fail);}
function toggleCrouch(){crouch=!crouch;$('btn-crouch').setAttribute('aria-pressed',String(crouch));}
$('btn-pause').onclick=pause;$('speed').onchange=e=>speed=Number(e.target.value);$('btn-help').onclick=help;$('close-help').onclick=help;$('btn-fs').onclick=fullscreen;$('btn-crouch').onclick=toggleCrouch;
$('steady-camera').checked=matchMedia('(prefers-reduced-motion: reduce)').matches;
canvas.onclick=()=>{if(!matchMedia('(hover:none)').matches)canvas.requestPointerLock()?.catch?.(()=>{});};
addEventListener('pointerlockchange',()=>{$('explore-hint').hidden=!!document.pointerLockElement;$('crosshair').hidden=!document.pointerLockElement;});
addEventListener('mousemove',e=>{if(document.pointerLockElement===canvas){lookX+=e.movementX*.002;lookY-=e.movementY*.002;}});
const bits={KeyW:1,ArrowUp:1,KeyS:2,ArrowDown:2,KeyA:4,ArrowLeft:4,KeyD:8,ArrowRight:8,ShiftLeft:16,ShiftRight:16,Space:64};
addEventListener('keydown',e=>{
 if(e.target.closest('input,select,textarea,button'))return;
 if(bits[e.code]){held.add(e.code);e.preventDefault();}
 if(e.repeat)return;
 if(e.code==='KeyP')pause();if(e.code==='KeyH')help();if(e.code==='KeyF')fullscreen();if(e.code==='KeyC')toggleCrouch();
 if(e.code==='KeyU')document.body.classList.toggle('ui-hidden');
 if(e.code==='KeyR')exclusive(()=>engine.reset()).catch(fail);
 if(['Digit1','Digit2','Digit3'].includes(e.code)){speed={Digit1:1,Digit2:.25,Digit3:.1}[e.code];$('speed').value=String(speed);}
});
addEventListener('keyup',e=>held.delete(e.code));addEventListener('blur',()=>held.clear());
addEventListener('visibilitychange',()=>{held.clear();last=performance.now();});
let touchLook=null,touchWalk=null,touchKeys=0;
canvas.addEventListener('pointerdown',e=>{if(e.pointerType==='mouse')return;canvas.setPointerCapture(e.pointerId);const p={id:e.pointerId,x:e.clientX,y:e.clientY};if(e.clientX<innerWidth*.4)touchWalk=p;else touchLook=p;});
canvas.addEventListener('pointermove',e=>{
 if(touchLook?.id===e.pointerId){lookX+=(e.clientX-touchLook.x)*.004;lookY-=(e.clientY-touchLook.y)*.004;touchLook.x=e.clientX;touchLook.y=e.clientY;}
 if(touchWalk?.id===e.pointerId){const dx=e.clientX-touchWalk.x,dy=e.clientY-touchWalk.y;touchKeys=(dy< -15?1:0)|(dy>15?2:0)|(dx< -15?4:0)|(dx>15?8:0)|(Math.hypot(dx,dy)>75?16:0);}
});
for(const name of ['pointerup','pointercancel'])canvas.addEventListener(name,e=>{if(touchLook?.id===e.pointerId)touchLook=null;if(touchWalk?.id===e.pointerId){touchWalk=null;touchKeys=0;}});
function exclusive(fn){
 pendingOperations++;
 const next=operation.catch(()=>{}).then(async()=>{await frameDone;busy=true;try{await engine.runtime.idle();return await fn();}finally{last=performance.now();busy=false;pendingOperations--;}});
 operation=next;return next;
}
window.__seek=t=>exclusive(()=>engine.seek(t));
window.__advance=seconds=>exclusive(async()=>{const count=Math.round(seconds/engine.step);for(let i=0;i<count;i+=60){const b=engine.runtime.batch();engine.advanceSteps(b,Math.min(60,count-i));b.submit();await engine.runtime.idle();}const b=engine.runtime.batch();engine.updateOcean(b);engine.updateSurface(b);b.submit();engine.draw();await engine.runtime.idle();return engine.time;});
window.__draw=()=>exclusive(async()=>{engine.draw();await engine.runtime.idle();});
window.__diag=()=>engine.diagnostics();window.__grab=()=>canvas.toDataURL('image/png');
window.__setCamera=async(position,angles=[0,-.18942])=>exclusive(async()=>{if(position.length!==3||angles.length!==2||![...position,...angles].every(Number.isFinite))throw new Error('Invalid camera');engine.runtime.write(engine.camera,new Float32Array([...position,0,...angles,0,0]));engine.draw();await engine.runtime.idle();});
window.__ready=true;$('loading').classList.add('hide');$('explore-hint').hidden=capture;
$('scene-state').textContent='THE LIVING COASTLINE · WEBCUDA';$('hud').hidden=capture;$('hud').style.cssText='position:fixed;top:90px;left:30px;color:#daeee8;font-size:12px;z-index:3';
if(query.has('nohud'))document.body.classList.add('ui-hidden');
async function frame(now){
 if(!running)return;requestAnimationFrame(frame);
 if(busy||pendingOperations||capture||document.hidden){last=now;return;}
 busy=true;
 frameDone=new Promise(resolve=>{settleFrame=resolve;});
 try{
  const dt=Math.min((now-last)/1000,.1);last=now;
  let keys=touchKeys|(crouch?32:0);for(const key of held)keys|=bits[key]||0;
  engine.frame(dt,{lookX,lookY,keys,steady:$('steady-camera').checked},paused,speed);lookX=0;lookY=0;
  await engine.runtime.idle();frameMs=frameMs*.92+(performance.now()-now)*.08;fps++;
  if(fps%30===0){$('hud').textContent=`WebCuda · ${engine.width} × ${engine.height} · ${Math.round(1000/Math.max(frameMs,1))} GPU frames/s`;
   if(quality==='auto'&&resolution===1){const next=Math.max(.55,Math.min(1,adaptive*(frameMs>32?.9:frameMs<18?1.04:1)));if(Math.abs(next-adaptive)>.015){adaptive=next;resize();}}
  }
 }catch(error){fail(error);}finally{busy=false;settleFrame();}
}
requestAnimationFrame(frame);addEventListener('pagehide',()=>{running=false;engine.dispose();},{once:true});
