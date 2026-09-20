import test from 'node:test';
import assert from 'node:assert/strict';
globalThis.location={search:'?explore'};
const {Schedule,brkTiming}=await import('../src/core/schedule.js');
const {waveSpeed,buildTravel,TRAVEL_N}=await import('../src/water/swellTravel.js');

test('swell satisfies finite-depth dispersion and slows in the shallows',()=>{
 let previous=0;
 for(const depth of [.2,.5,1,2,5,10,30]){
  const {k,c,cg}=waveSpeed(depth);
  assert.ok(Math.abs(9.81*k*Math.tanh(k*depth)-(2*Math.PI/4.4)**2)<1e-10);
  assert.ok(c>previous);assert.ok(cg>=c*.5&&cg<=c);previous=c;
 }
});
test('swell travel and inverse remain monotonic and continuous across the shelf',()=>{
 const {forward,inverse}=buildTravel();
 for(let i=1;i<TRAVEL_N;i++){
  assert.ok(forward[4*i]>forward[4*(i-1)]);
  assert.ok(inverse[4*i+3]>inverse[4*(i-1)+3]);
 }
 assert.ok(forward.every(Number.isFinite));assert.ok(inverse.every(Number.isFinite));
 for(let i=0;i<TRAVEL_N;i+=11){
  const z=inverse[i*4+3],j=(z+900)/899*(TRAVEL_N-1),lo=Math.min(Math.floor(j),TRAVEL_N-2),f=j-lo;
  const t=forward[lo*4]*(1-f)+forward[(lo+1)*4]*f;
  assert.ok(Math.abs(t-inverse[i*4])<.0001);
 }
});
test('larger waves are seeded, grouped and separated by quieter intervals',()=>{
 const a=new Schedule({seed:7}),b=new Schedule({seed:7});a.extendTo(1200);b.extendTo(1200);
 assert.deepEqual(a.events.map(e=>[e.t0,e.H]),b.events.map(e=>[e.t0,e.H]));
 const big=a.events.filter(e=>e.large);assert.ok(big.length>12);
 for(let i=0;i<big.length;i++){
  const e=big[i],index=a.events.indexOf(e);assert.ok(e.H>=.8&&e.H<=.88);
  assert.ok(a.events[index-1].H<e.H&&a.events[index-2].H<a.events[index-1].H);
  assert.ok(a.events[index+1].H<e.H);
  if(i)assert.ok(e.t0-big[i-1].t0>40&&e.t0-big[i-1].t0<80);
  const local=brkTiming(e,-34);assert.ok(local.zI<-2.3);assert.ok(local.splash>1&&local.bore>1);
 }
});
test('large sets fit the existing six-event solver budget with no dropped breakers',()=>{
 const s=new Schedule();s.extendTo(600);
 for(let t=0;t<600;t+=.2){
  const active=s.events.filter(e=>t>=e._window[0]&&t<=e._window[1]);
  assert.ok(active.length<=6);
  const visible=s.events.filter(e=>e.t0>t-6&&e.t0<t+114);assert.ok(visible.length<=32);
 }
});
