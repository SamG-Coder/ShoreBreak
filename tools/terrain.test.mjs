import test from 'node:test';
import assert from 'node:assert/strict';
import {buildExploreGeometry,LAND,morphTerrainColumn} from '../src/beach/terrain.js';

test('beach patches cover the coast once, without overlapping seams or reversed triangles',()=>{
 const g=buildExploreGeometry(),p=g.attributes.position,idx=g.index.array;let area=0;
 for(let i=0;i<idx.length;i+=3){
  const a=idx[i],b=idx[i+1],c=idx[i+2];
  const signed=(p.getZ(b)-p.getZ(a))*(p.getX(c)-p.getX(a))-(p.getX(b)-p.getX(a))*(p.getZ(c)-p.getZ(a));
  assert.ok(signed>0,'Every terrain triangle must face upward and have positive area');area+=signed*.5;
 }
 assert.ok(Math.abs(area-2*LAND.reach*(LAND.zMax+60))<.01,`Unexpected terrain coverage ${area}`);
 g.dispose();
});

test('wet and dry beach columns stay continuous across every player-window snap',()=>{
 const g=buildExploreGeometry(),p=g.attributes.position;
 const rows=new Map([3,4.5,4.975,5,8,20,39].map(z=>[z,new Set()]));
 for(let i=0;i<p.count;i++)for(const [z,xs] of rows)if(Math.abs(p.getZ(i)-z)<1e-5)xs.add(p.getX(i));
 let worst=0;
 const distance=(a,b)=>{let j=0,m=0;for(const x of a){while(j+1<b.length&&Math.abs(b[j+1]-x)<=Math.abs(b[j]-x))j++;m=Math.max(m,Math.abs(b[j]-x));}return m;};
 for(const [z,set] of rows){
  assert.ok(set.size>20,`Missing terrain test row ${z}`);
  const xs=[...set];
  for(let n=-75;n<75;n++){
   const focus=(n+.5)*LAND.snap;
   const pair=[-1,1].map(sign=>xs.map(x=>morphTerrainColumn(x,z,focus+sign*1e-6)).filter(x=>Math.abs(x-focus)<40).sort((a,b)=>a-b));
   const jump=Math.max(distance(pair[0],pair[1]),distance(pair[1],pair[0]));
   worst=Math.max(worst,jump);
   assert.ok(jump<.00001,`Terrain row z=${z}, focus=${focus}: column jump ${jump} m`);
  }
 }
 g.dispose();
});
