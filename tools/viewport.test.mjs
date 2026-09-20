import test from 'node:test';
import assert from 'node:assert/strict';
import { SCENE_AREA, SCENE_SCALE, PIXEL_BUDGET_SCALE, normalizeResolutionScale, usesAdaptiveResolution, renderPixelRatio } from '../src/core/viewport.js';

test('framed view preserves aspect and occupies 75% of landscape and portrait screens', () => {
  for (const [w, h] of [[1920,1080],[3840,2160],[390,844],[844,390]]) {
    const cw=w*SCENE_SCALE, ch=h*SCENE_SCALE;
    assert.ok(Math.abs(cw/ch-w/h)<1e-12);
    assert.ok(Math.abs(cw*ch/(w*h)-.75)<1e-12);
  }
  assert.equal(SCENE_AREA,.75);
});

test('capped modes retain 90% of the pixel budget and allow denser pixels in the smaller frame', () => {
  for (const pixels of [560000,1050000,1800000,2200000,4200000]) {
    const w=3840,h=2160,cw=w*SCENE_SCALE,ch=h*SCENE_SCALE;
    const opts={dpr:2,maxDpr:2,pixels};
    const before=renderPixelRatio(w,h,opts);
    const after=renderPixelRatio(cw,ch,{...opts,budgetScale:PIXEL_BUDGET_SCALE});
    // The existing 0.4 minimum ratio can bind on the lowest budget.
    if(before>.4){
      assert.ok(Math.abs(cw*ch*after**2/(w*h*before**2)-.9)<1e-12);
      assert.ok(Math.abs(after/before-Math.sqrt(.9/.75))<1e-12);
    }
  }
  assert.equal(PIXEL_BUDGET_SCALE,.9);
});

test('DPR limits, Auto downscaling and exact-size captures remain effective', () => {
  const options={dpr:1,maxDpr:1.5,pixels:1800000,budgetScale:.9};
  assert.equal(renderPixelRatio(390*SCENE_SCALE,844*SCENE_SCALE,options),1);
  assert.equal(renderPixelRatio(1920,1080,{...options,dynamicScale:.5}),
    renderPixelRatio(1920,1080,options)*.5);
  assert.equal(renderPixelRatio(3840,2160,{...options,exact:true}),1);
});

test('resolution boosts both dimensions above DPR and quality caps', () => {
  for (const [w,h,dpr,pixels] of [[1280,720,1,1800000],[3840,2160,2,2200000],[7680,4320,2,560000]]) {
    const options={dpr,maxDpr:1.5,pixels,budgetScale:.9};
    const baseline=renderPixelRatio(w,h,options);
    for (const scale of [1,1.5,2]) {
      const boosted=renderPixelRatio(w,h,{...options,resolutionScale:scale});
      assert.equal(boosted,baseline*scale);
      assert.ok(Math.abs((w*boosted*h*boosted)/(w*baseline*h*baseline)-scale**2)<1e-12);
    }
  }
});

test('manual supersampling is not cancelled by adaptive resolution', () => {
  const options={dpr:1,maxDpr:1.5,pixels:1800000,budgetScale:.9};
  assert.equal(usesAdaptiveResolution('auto',1),true);
  for(const scale of [1.5,2]) {
    assert.equal(usesAdaptiveResolution('auto',scale),false);
    assert.equal(renderPixelRatio(1920,1080,{...options,resolutionScale:scale,dynamicScale:.5}),
      renderPixelRatio(1920,1080,{...options,resolutionScale:scale}));
  }
  for(const quality of ['low','medium','high','ultra'])assert.equal(usesAdaptiveResolution(quality,1),false);
});

test('supersampling respects hardware dimensions without stretching or changing captures', () => {
  const options={dpr:2,maxDpr:2,pixels:4200000,resolutionScale:2,maxDimension:2048};
  for(const [w,h]of [[3840,2160],[2160,3840]]) {
    const ratio=renderPixelRatio(w,h,options);
    assert.equal(Math.max(w*ratio,h*ratio),2048);
    assert.ok(Math.abs((w*ratio)/(h*ratio)-w/h)<1e-12);
    assert.equal(renderPixelRatio(w,h,{...options,exact:true}),1);
  }
});

test('stored resolution choices validate and default safely', () => {
  assert.equal(normalizeResolutionScale('1.5'),1.5);
  assert.equal(normalizeResolutionScale('2'),2);
  for(const value of [null,undefined,'',0,-1,3,150,'bad',Infinity,NaN])assert.equal(normalizeResolutionScale(value),1);
});
