import test from 'node:test';
import assert from 'node:assert/strict';
import { bedProfileJS } from '../src/config.js';
import { ROCKS,ROCK_SEGMENTS,ROCK_RINGS,rockVertex,rockHeight } from '../src/beach/CoastalBed.js';
import { CoastalRocks } from '../src/beach/CoastalRocks.js';

test('every rock rim remains buried across the sloping seabed',()=>{
 for(const r of ROCKS)for(let i=0;i<ROCK_SEGMENTS;i++){
  const [x,y,z]=rockVertex(r,i,ROCK_RINGS);
  assert.ok(y<bedProfileJS(z)-.02,`Exposed rim at ${x},${z}`);
 }
});
test('hydraulic rock heights and rendered vertices agree',()=>{
 for(const r of ROCKS)for(let j=0;j<=ROCK_RINGS;j++)for(let i=0;i<=ROCK_SEGMENTS;i++){
  const [x,y,z]=rockVertex(r,i,j);
  assert.ok(Number.isFinite(y));
  assert.ok(Math.abs(y-rockHeight(r,x,z))<1e-8);
 }
});
test('rock wrap seams and duplicated crowns have continuous shading normals',()=>{
 const rocks=new CoastalRocks({});
 const n=rocks.mesh.geometry.getAttribute('normal'),stride=ROCK_SEGMENTS+1;
 for(let k=0;k<ROCKS.length;k++){
  const base=k*stride*(ROCK_RINGS+1);
  for(let j=0;j<=ROCK_RINGS;j++){
   const a=base+j*stride,b=a+ROCK_SEGMENTS;
   for(let c=0;c<3;c++)assert.equal(n.getComponent(a,c),n.getComponent(b,c));
  }
  for(let i=1;i<=ROCK_SEGMENTS;i++)for(let c=0;c<3;c++)assert.equal(n.getComponent(base,c),n.getComponent(base+i,c));
 }
 rocks.mesh.geometry.dispose();rocks.material.dispose();
});
