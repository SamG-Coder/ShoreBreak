import test from 'node:test';
import assert from 'node:assert/strict';
import { ExploreControls } from '../src/core/explore.js';
const target={addEventListener(){}};
globalThis.window={...target,__ready:true};globalThis.document={...target};
function make(opts={}){
 const camera={position:{set(x,y,z){Object.assign(this,{x,y,z});}},rotation:{set(x,y,z){Object.assign(this,{x,y,z});}},updateMatrixWorld(){}};
 const c=new ExploreControls(target,camera,{ground:()=>1,bounds:{xMin:-1000,xMax:1000,zMin:-1000,zMax:1000},start:{x:0,z:0,yaw:0,pitch:0},...opts});return c;
}
test('touch movement starts in the left 35% of the inset canvas',()=>{
 const events={};const dom={addEventListener(type,fn){events[type]=fn;},getBoundingClientRect(){return {left:300,width:600};}};
 const camera={position:{set(){}},rotation:{set(){}},updateMatrixWorld(){}};
 const c=new ExploreControls(dom,camera,{ground:()=>0,bounds:{xMin:-60,xMax:60,zMin:-14,zMax:36}});
 events.touchstart({changedTouches:[{identifier:1,clientX:505,clientY:200},{identifier:2,clientX:515,clientY:200}],preventDefault(){}});
 assert.equal(c.stick.id,1);assert.equal(c.look.id,2);
});
for(const run of [false,true]){
 test(`${run?'running':'walking'} distance is stable from 30 to 240 Hz`,()=>{
  const distances=[30,60,90,144,240].map(hz=>{const c=make();c.keys.add('KeyW');if(run)c.keys.add('ShiftLeft');for(let i=0;i<hz*5;i++)c.update(1/hz);return -c.pos.z;});
  assert.ok(Math.max(...distances)-Math.min(...distances)<1e-9);
 });
 test(`${run?'running':'walking'} camera has bounded smooth motion`,()=>{
  const c=make();c.keys.add('KeyW');if(run)c.keys.add('ShiftLeft');let min=1e9,max=-1e9,last=2.64,maxDelta=0;
  for(let i=0;i<1200;i++){c.update(1/120);const y=c.camera.position.y;min=Math.min(y,min);max=Math.max(y,max);maxDelta=Math.max(maxDelta,Math.abs(y-last));last=y;}
  assert.ok(max-min<(run?.023:.013));assert.ok(maxDelta<.002);
 });
}
test('stopping settles without residual bounce',()=>{const c=make();c.keys.add('KeyW');c.keys.add('ShiftLeft');for(let i=0;i<240;i++)c.update(1/120);c.keys.clear();for(let i=0;i<240;i++)c.update(1/120);assert.ok(Math.abs(c.camera.position.y-2.64)<1e-5);});
test('blocked movement does not produce footfalls',()=>{const c=make({bounds:{xMin:0,xMax:0,zMin:0,zMax:0}});c.keys.add('KeyW');c.keys.add('ShiftLeft');for(let i=0;i<240;i++)c.update(1/120);assert.equal(c.bobAmp,0);assert.equal(c.bobPhase,0);});
test('reduced motion keeps a steady horizon',()=>{const c=make({reducedMotion:true});c.keys.add('KeyW');c.keys.add('ShiftLeft');for(let i=0;i<240;i++)c.update(1/120);assert.ok(Math.abs(c.camera.position.y-2.64)<1e-12);assert.ok(Math.abs(c.camera.rotation.z)<1e-12);});
test('crouching settles at a low eye height and returns to standing',()=>{const c=make();c.toggleCrouch();for(let i=0;i<240;i++)c.update(1/120);assert.ok(Math.abs(c.camera.position.y-1.82)<1e-6);c.toggleCrouch();for(let i=0;i<240;i++)c.update(1/120);assert.ok(Math.abs(c.camera.position.y-2.64)<1e-6);});
test('crouching limits speed even with Shift held',()=>{const c=make();c.toggleCrouch();c.keys.add('KeyW');c.keys.add('ShiftLeft');for(let i=0;i<600;i++)c.update(1/120);assert.equal(c.running,false);assert.ok(Math.abs(c.vel.z+.72)<1e-8);});
test('crouch transition is monotonic and frame-rate independent',()=>{const heights=[30,60,144,240].map(hz=>{const c=make();c.toggleCrouch();let previous=2.64;for(let i=0;i<hz;i++){c.update(1/hz);assert.ok(c.camera.position.y<=previous);assert.ok(c.camera.position.y>=1.82);previous=c.camera.position.y;}return c.camera.position.y;});assert.ok(Math.max(...heights)-Math.min(...heights)<1e-12);});
test('crouch key toggles once per press and ignores repeat',()=>{const events={};globalThis.window={addEventListener(type,fn){events[type]=fn;},__ready:true};const c=make();const event={code:'KeyC',target:{closest(){return null;}},preventDefault(){},repeat:false};events.keydown(event);assert.equal(c.crouched,true);events.keydown({...event,repeat:true});assert.equal(c.crouched,true);events.keydown(event);assert.equal(c.crouched,false);globalThis.window={...target,__ready:true};});

test('deeper wading progressively slows travel and prevents running',()=>{
 const distances=[0,-.3,-.85].map(depth=>{const c=make({ground:()=>depth});c.keys.add('KeyW');for(let i=0;i<240;i++)c.update(1/120);c.keys.add('ShiftLeft');if(depth<-.28)assert.equal(c.running,false);return -c.pos.z;});
 assert.ok(distances[0]>distances[1]&&distances[1]>distances[2]);assert.ok(distances[2]>.7&&distances[2]<1.2);
});
test('crouched wading keeps the eye above mean sea level',()=>{
 const c=make({ground:()=>-.88});c.toggleCrouch();for(let i=0;i<240;i++)c.update(1/120);
 assert.ok(Math.abs(c.camera.position.y-.32)<1e-7);
 c.keys.add('KeyW');for(let i=0;i<240;i++)c.update(1/120);assert.ok(c.camera.position.y>.315);
});

test('walk, wade, swim and walk ashore transition without a camera jump',()=>{
 const c=make({ground:(x,z)=>z*.16,start:{x:0,z:2,yaw:0,pitch:0}});
 c.keys.add('KeyW');c.keys.add('ShiftLeft');const modes=new Set();let previous=c.eyeY,maxJump=0;
 for(let i=0;i<60*60;i++){c.update(1/60);modes.add(c.mode);maxJump=Math.max(maxJump,Math.abs(c.eyeY-previous));previous=c.eyeY;}
 assert.ok(modes.has('walk')&&modes.has('wade')&&modes.has('swim'));assert.ok(c.swimBlend>.99);assert.ok(maxJump<.045);
 c.yaw=Math.PI;
 for(let i=0;i<90*60;i++)c.update(1/60);
 assert.equal(c.mode,'walk');assert.ok(c.swimBlend<1e-6);assert.ok(c.pos.z>1);
});
test('a resting swimmer follows the wave height with a bounded breathing viewpoint',()=>{
 const water=(x,z,t)=>({height:.32*Math.sin(t*1.5),slopeX:.12*Math.cos(t*1.5),slopeZ:.2*Math.cos(t*1.5),flowX:0,flowZ:0});
 const c=make({ground:()=>-2,water});let lo=10,hi=-10,minClearance=10;
 for(let i=0;i<1200;i++){c.update(1/120);if(i>240){lo=Math.min(lo,c.eyeY);hi=Math.max(hi,c.eyeY);minClearance=Math.min(minClearance,c.eyeY-water(0,0,c.waterTime).height);}}
 assert.equal(c.mode,'swim');assert.ok(hi-lo>.55&&hi-lo<.70);assert.ok(minClearance>=.13);assert.ok(c.bobAmp<1e-8);
 assert.ok(Math.abs(c.camera.rotation.z)<.025);
});
test('swimming is consistent at 30, 60 and 144 Hz and Shift increases swimming speed',()=>{
 const distances=[];
 for(const hz of [30,60,144]){const c=make({ground:()=>-2});for(let i=0;i<hz*4;i++)c.update(1/hz);c.keys.add('KeyW');for(let i=0;i<hz*8;i++)c.update(1/hz);distances.push(-c.pos.z);assert.equal(c.running,false);const slow=c.vel.z;c.keys.add('ShiftLeft');for(let i=0;i<hz*4;i++)c.update(1/hz);assert.ok(-c.vel.z>-slow*1.4);}
 assert.ok(Math.max(...distances)-Math.min(...distances)<.01);
});
test('swim hysteresis prevents mode chatter and crouch is cleared on entering deep water',()=>{
 let g=-.9;const c=make({ground:()=>g});c.toggleCrouch();g=-1.6;for(let i=0;i<300;i++)c.update(1/60);
 assert.equal(c.swimming,true);assert.equal(c.crouched,false);c.toggleCrouch();assert.equal(c.crouched,false);
 for(let i=0;i<600;i++){g=-1.2+.06*Math.sin(i*.1);c.update(1/60);assert.equal(c.swimming,true);}
 g=-.8;for(let i=0;i<240;i++)c.update(1/60);assert.equal(c.swimming,false);
});
test('swimming reaches the extended offshore limit smoothly and can return',()=>{
 const c=make({ground:()=>-2,bounds:{xMin:-60,xMax:60,zMin:-14,zMax:36},start:{x:0,z:-4,yaw:0,pitch:0}});
 c.keys.add('KeyW');c.keys.add('ShiftLeft');for(let i=0;i<60*40;i++)c.update(1/60);
 assert.ok(c.pos.z>=-14&&c.pos.z<-13.8);assert.ok(Math.abs(c.vel.z)<.025);
 c.yaw=Math.PI;for(let i=0;i<60*10;i++)c.update(1/60);assert.ok(c.pos.z>-3);
});
test('steady camera removes swim roll without removing buoyancy',()=>{
 const c=make({ground:()=>-2,reducedMotion:true,water:(x,z,t)=>({height:.3*Math.sin(t),slopeX:.5,slopeZ:.5,flowX:0,flowZ:0})});
 c.keys.add('KeyW');for(let i=0;i<600;i++)c.update(1/60);
 assert.equal(c.camera.rotation.z,0);assert.equal(c.camera.rotation.x,c.pitch);assert.ok(Math.abs(c.eyeY-.34)>.05);
});

test('C dips smoothly below the waves, respects the seabed, and surfaces again',()=>{
 const c=make({ground:()=>-1.5});for(let i=0;i<240;i++)c.update(1/60);
 c.toggleCrouch();assert.equal(c.diving,true);let maxStep=0,last=c.camera.position.y;
 for(let i=0;i<240;i++){c.update(1/60);maxStep=Math.max(maxStep,Math.abs(c.camera.position.y-last));last=c.camera.position.y;assert.ok(last>=-1.26);}
 assert.ok(c.eyeY<-.65&&c.eyeY>-.70);assert.ok(maxStep<.055);assert.equal(c.mode,'dive');
 c.toggleCrouch();for(let i=0;i<240;i++)c.update(1/60);assert.ok(c.eyeY>.33);assert.equal(c.mode,'swim');
});
test('diving ends automatically when returning to water too shallow to swim',()=>{
 let floor=-2;const c=make({ground:()=>floor});for(let i=0;i<240;i++)c.update(1/60);
 c.toggleCrouch();for(let i=0;i<180;i++)c.update(1/60);
 for(let i=0;i<600;i++){floor=Math.min(.1,floor+.0035);c.update(1/60);assert.ok(c.eyeY>floor-.01);}
 assert.equal(c.diving,false);assert.equal(c.swimming,false);assert.ok(c.eyeY>1.5);
});
test('a rapidly rising crest can wash over a surface swimmer',()=>{
 let height=0;const c=make({ground:()=>-2,water:()=>({height})});for(let i=0;i<240;i++)c.update(1/60);
 height=.8;c.update(1/60);assert.ok(c.eyeY<height);
 for(let i=0;i<240;i++)c.update(1/60);assert.ok(c.eyeY>height+.3);
});
test('Space produces one gravity-driven jump, cannot air-jump, and lands cleanly',()=>{
 const heights=[];
 for(const hz of [30,60,144,240]){
  const c=make();assert.equal(c.jump(),true);let max=0;
  for(let i=0;i<hz*2;i++){c.update(1/hz);max=Math.max(max,c.jumpY);if(i===2)assert.equal(c.jump(),false);}
  heights.push(max);assert.equal(c.jumpY,0);assert.equal(c.jumpVelocity,0);assert.ok(Math.abs(c.camera.position.y-2.64)<1e-6);
 }
 assert.ok(Math.max(...heights)-Math.min(...heights)<.002);assert.ok(heights[0]>.53&&heights[0]<.55);
});
test('jump is unavailable while swimming or diving',()=>{
 const c=make({ground:()=>-2});for(let i=0;i<240;i++)c.update(1/60);assert.equal(c.jump(),false);
 c.toggleCrouch();assert.equal(c.jump(),false);assert.equal(c.jumpVelocity,0);
});
test('Space handler jumps once per press and does not become a held movement key',()=>{
 const events={};globalThis.window={addEventListener(type,fn){events[type]=fn;},__ready:true};const c=make();let jumps=0;c.jump=()=>{jumps++;};
 const e={code:'Space',target:{closest(){return null;}},preventDefault(){},repeat:false};
 events.keydown(e);events.keydown({...e,repeat:true});assert.equal(jumps,1);assert.equal(c.keys.has('Space'),false);globalThis.window={...target,__ready:true};
});
