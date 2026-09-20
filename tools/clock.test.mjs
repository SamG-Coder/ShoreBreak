import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FixedClock } from '../src/core/clock.js';
for (const hz of [30,60,75,90,120,144,165,240]) for (const speed of [1,0.25,0.1]) {
  test(`${hz} Hz at ${speed}x preserves elapsed simulation time`,()=>{
    const clock=new FixedClock(1/120);let steps=0;
    for(let i=0;i<hz*20;i++)clock.advance(1/hz,speed,()=>steps++);
    assert.ok(Math.abs(steps/120-20*speed)<1/120+1e-8);
  });
}
test('irregular frames retain substep remainder',()=>{
 const clock=new FixedClock(1/120);let elapsed=0,steps=0;
 for(let i=0;i<1000;i++){const d=.003+(i%17)*.001;elapsed+=d;clock.advance(d,1,()=>steps++);}
 assert.ok(Math.abs(steps/120-elapsed)<1/120+1e-8);
});
test('a suspended tab cannot enqueue an unbounded simulation backlog',()=>{
 const clock=new FixedClock(1/120);let steps=0;
 clock.advance(120,1,()=>steps++);assert.equal(steps,12);
 clock.reset();assert.equal(clock.advance(0,1,()=>steps++),0);
});
