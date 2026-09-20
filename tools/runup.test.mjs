import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';

globalThis.location={search:'?explore'};
const {CONFIG}=await import('../src/config.js');
const {ExploreWaterMesh,EX}=await import('../src/water/ExploreMesh.js');
const {SwashSim}=await import('../src/swash/SwashSim.js');

test('large run-up has continuous mesh, water and foam coverage beyond the former cutoff',()=>{
 const shared={uFocus:{value:new THREE.Vector2(-34,6.8)},uTime:{value:0},uInjMass:{value:1},uInjSpeed:{value:1},uEvtCount:{value:0}};
 for(const k of 'ABCDEFG')shared['uEvt'+k]={value:new Float32Array(24)};
 const noop=()=>{},renderer={setRenderTarget:noop,render:noop,clear:noop,setClearColor:noop,setClearAlpha:noop,getClearAlpha:()=>1};
 const sim=new SwashSim(renderer,shared),mesh=new ExploreWaterMesh();
 assert.ok(EX.zTop>=6,'Water must cover the full high-run-up reserve');
 assert.ok(CONFIG.swe.zMax>EX.zTop+2*sim.dz,'Fluid coverage must contain the rendered upper edge');
 assert.ok(Math.abs(sim.dz-9.8/328)<1e-15,'Extending the grid must preserve the original physics cell spacing');
 assert.ok(mesh.zb.some(z=>z>3.4)&&mesh.zb.some(z=>z<3.4),'The former cutoff must be internal to the water mesh');
 const map=shared.uSwashFarMapZ.value;
 assert.ok(map.x+(map.z-1)*map.y>=EX.zTop,'Far water, foam and wetness must also cover the reserve');
 for(const rt of [sim.view,sim.foamView,sim.wet.read])assert.equal(rt.height,sim.nz,'Water, foam and wetness must share the extended rows');
});
