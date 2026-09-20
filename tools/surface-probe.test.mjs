import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {SurfaceProbe} from '../src/water/SurfaceProbe.js';

test('surface queries share one async readback and restore the render target',async()=>{
 const p=new SurfaceProbe({}),camera={position:new THREE.Vector3(0,.34,-10)};
 const original={name:'scene'};let current=original,reads=0,release;
 const renderer={getRenderTarget:()=>current,setRenderTarget:t=>{current=t;},render(){},readRenderTargetPixelsAsync:async(rt,x,y,w,h,data)=>{
  reads++;await new Promise(resolve=>{release=resolve;});
  for(let i=0;i<5;i++)data.set([.25+(i===1?-.032:i===2?.032:0),.1,.2,1],i*4);
 }};
 p.request(renderer,camera,10,0,-10,true);p.request(renderer,camera,10,0,-10,true);
 assert.equal(reads,1);assert.equal(current,original);assert.equal(p.pending,true);
 release();await p.ready;
 assert.equal(p.pending,false);const result={...p.sample(0,-10,10)};
 assert.ok(Math.abs(result.height-.25)<1e-6);assert.ok(Math.abs(result.slopeX-.1)<1e-6);
 assert.ok(Math.abs(result.flowZ-.2)<1e-6);
 const predicted=p.sample(100,-10,100);assert.ok(Number.isFinite(predicted.height)&&predicted.height<.31);
});

test('invalid water readbacks preserve the last finite surface and release the queue',async()=>{
 const p=new SurfaceProbe({});p.latest={height:.2,slopeX:0,slopeZ:0,flowX:0,flowZ:0,x:0,z:-10,t:0,rate:0};
 const camera={position:new THREE.Vector3()};let target=null;
 const renderer={getRenderTarget:()=>target,setRenderTarget:t=>{target=t;},render(){},readRenderTargetPixelsAsync:async(rt,x,y,w,h,data)=>{data.fill(NaN);}};
 p.request(renderer,camera,1,0,-10,true);await p.ready;
 assert.equal(p.pending,false);assert.equal(target,null);assert.equal(p.sample(0,-10,1).height,.2);assert.equal(p.failures,1);
});

test('the GPU waterline refreshes while the one CPU readback remains pending',async()=>{
 const shared={uSurfaceProbe:{value:null},uProbeOrigin:{value:new THREE.Vector2()}};
 const p=new SurfaceProbe(shared),camera={position:new THREE.Vector3(0,-.4,-8)};
 let current=null,draws=0,reads=0,release;
 const renderer={getRenderTarget:()=>current,setRenderTarget:t=>{current=t;},render(){draws++;},readRenderTargetPixelsAsync:async(rt,x,y,w,h,data)=>{
  reads++;await new Promise(resolve=>{release=resolve;});for(let i=0;i<5;i++)data.set([.15,0,0,1],i*4);
 }};
 p.request(renderer,camera,1,0,-8,true);p.request(renderer,camera,1.01,.02,-8,true);
 assert.equal(draws,2);assert.equal(reads,1);assert.equal(shared.uSurfaceProbe.value,p.target.texture);assert.equal(shared.uProbeOrigin.value.x,.02);assert.equal(current,null);
 release();await p.ready;assert.equal(p.pending,false);assert.equal(p.latest.x,0);
});
