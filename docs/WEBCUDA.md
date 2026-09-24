# WebCuda migration

## Runtime boundary

`src/webcuda/main.js` owns the DOM, keyboard/pointer/touch events, presentation size and
the animation clock. `engine.js` loads assets, invokes the vendored WebCuda runtime,
queues fixed-step dispatches and copies finished pixels to the canvas.
There is no CPU surface sampling or simulation readback in the normal render loop.
Readback is exposed only to the diagnostics/test API.

All executable GPU programs originate in `kernels/*.cu`. `tools/compile-webcuda.mjs`
generates the original bathymetry and breaker table declarations, compiles fourteen
entry points, and writes precompiled WGSL/ABI JSON artifacts. Every entry must compile
before any artifact is replaced. The manifest records source and artifact SHA-256 hashes.

Presentation does not need a vertex/fragment pipeline: `finishFrame` writes RGBA/BGRA
uint pixels, with rows padded to 256 bytes; `copyBufferToTexture` transfers them to the
WebGPU swapchain. WebCuda owns all persistent GPU allocations. Resizing invalidates
the affected cached bindings; device errors stop the frame loop and show the retry UI.

## Implemented systems

| Source | GPU work |
| --- | --- |
| `common.cu` | Bathymetry, original rock caps, seeded noise, scheduled breakers and original Hermite stage table |
| `ocean.cu` | JONSWAP initialization, spectral evolution, Stockham inverse FFT and composition of three cascades |
| `coast.cu` | Hydrostatic Rusanov shallow-water fluxes, wet/dry handling, foam transport, wetness, surface field and player motion |
| `geometry.cu` | Stackless BVH traversal, triangle intersections and original palm instance transforms |
| `render.cu` | Terrain/water ray intersection, curling sheet, Fresnel reflection, depth absorption, bottom caustics, sky, surface foam, fog and display filtering |
| `spray.cu` | Event-driven ballistic spray / buoyant air markers and depth-tested projection |
| `materials.cu` | Linear-light mip generation for original sand and rock textures |

FFT cascades have lengths 41 m, 7.3 m and 1.37 m, each with 128² complex modes.
The shallow-water domain is x=[−64,64], z=[−6,12], sampled at 512×192, advanced at
1/120 second. The solver uses reconstructed water levels over the hydraulic rock bed,
Manning friction and infiltration above still-water level. Momentum is stored as h*u/h*v.
Surface and camera use the same GPU-generated field. Motion includes wading drag,
swimming hysteresis, running, crouching, single-press jumping, and optional gait bob.

The visible water field spans x=[−160,160], z=[−120,20] at 1024×512. There are 32,768
potential spray/bubble markers, evaluated deterministically from event age on the GPU.
Simulation time does not depend on render size. Camera movement remains active when
water is paused. Rendering backpressure prevents an unbounded queue of GPU frames.

## Original geometry

The offline scenery converter reads upstream `Seafront.js` and the supplied palm binaries.
It preserves the seafront's triangle positions and colors, converts the original tier-1
palm meshes, and adds the promenade wall cross-section. The resulting BVH contains
321,606 triangles and 147,461 nodes in a 26.4 MiB asset. Palm BLAS geometry is shared
across the original instance placements. The original assets remain bundled with their
notices; the conversion is covered by the original MIT project license.

Three.js is used by this offline asset converter and the inherited upstream tests.
It is not reachable from the production browser module graph. The active scene performs
its geometry intersections, transforms and lighting in CUDA.

## Differences and limits

This is not feature-for-feature numerical or visual parity with the upstream WebGL renderer:

- Original FFT resolution, wave-front interpolation, event continuation, run-up grid,
  stochastic foam and optical calibration have changed. The opening event times and
  shape table are retained, but the full upstream peel and transport formulas are not.
- The CUDA height-field renderer and separate analytic curling sheet replace mesh
  tessellation and the original ballistic lip ribbons. Very grazing rays can lose detail.
- The original seafront and palm meshes are retained; the far 4 km city extensions and
  mountain panorama are not included in the converted BVH. Navigation is bounded to
  x=[−145,145], z=[−105,30]. Water outside the simulated region is an analytic continuation.
- Underwater absorption, underside reflection, caustics and air particles are implemented;
  the original volumetric plume, contact-light pass, temporal exposure adaptation and
  material-specific shadow maps are not reproduced.
- GPU texture filtering and edge filtering reduce aliasing but do not provide the original
  renderer's multisample coverage or temporal stability. Very thin palm leaflets can shimmer.
- Physics is a graphics approximation, not an engineering fluid model. Performance and
  device coverage beyond the recorded local browser test are not certified.

The upstream source remains in place to make further parity work traceable. Its tests
exercise that reference implementation; the separate WebCuda browser suite is the
evidence for the new GPU path.

## Verification

`npm test` includes source-graph isolation, manifest hashes, binding/workgroup budgets,
BVH integrity, and the inherited 79 regression tests. `npm run test:webcuda:dist` uses a
real WebGPU browser against the production build and saves a machine-readable report.
It checks compiler/device validation, shallow-water finiteness/positivity, the resting-lake
invariant, movement, unaligned row pitch, and 30 simulated seconds of driven waves.

Optional test API: `__seek(seconds)`, `__advance(seconds)`, `__draw()`, `__setCamera(position,angles)`,
`__diag()` and `__grab()`. Seek/reset operations are serialized with frame submission.
Angles use yaw=0 toward −z, positive yaw toward +x, and positive pitch upward.
