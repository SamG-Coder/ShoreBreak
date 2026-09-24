# Rendering architecture

> This document describes the retained upstream WebGL implementation. The active fork
> uses WebCuda; see [the WebCuda architecture and migration limits](WEBCUDA.md).

ShoreBreak uses Three.js r186's WebGL renderer and a Vite frontend. Coordinates
are in metres: **x** follows the shore, **y** points up, and positive **z** points
inland. The nominal shoreline is near z = 0. The entry point is `src/boot.js`,
which paints the loading UI before importing `src/main.js`.

## Water and wave breaking

| System | Main files | Role |
| --- | --- | --- |
| Wave timing | `src/core/schedule.js` | Deterministic wave events and occasional stronger sets |
| Swell | `src/water/Swell.js`, `src/glsl/swell.js` | Incoming swell and propagation over changing depth |
| Wind sea | `src/water/OceanFFT.js` | Three FFT cascades for displacement and filtered slope detail |
| Breaker shape | `src/glsl/breaker.js` | Shoaling and breaking profile shared by rendering and effects |
| Plunging lip | `src/water/LipRibbon.js` | Separate mesh for the overturning sheet |
| Shallow water | `src/swash/SwashSim.js` | GPU shallow-water height and flow, wetting and drying |
| Surface optics | `src/glsl/water.js` | Reflection, refraction, absorption, microdetail, foam and glints |
| Whitewater | `src/whitewater/` | Impact-driven spray and foam |
| Underwater | `src/glsl/underwater.js`, `src/water/UnderwaterPlume.js` | Underside optics, caustics, entrained air and turbidity |

The FFT is a spectral wind-sea model. Larger breaking waves use scheduled
profiles and ballistic lip motion. The shallow-water solver transports flow
over the beach and around the hydraulic rock bed. These systems share timing,
bathymetry, and surface fields to keep effects aligned.

The surface is mesh-based; it is not a ray-marched implicit ocean. The underwater
plume uses a bounded volume-marching effect for suspended air. These are distinct
parts of the renderer. The project does not solve a complete 3D Navier–Stokes
fluid volume, and its scheduled breakers are not arbitrary emergent wave breaking.

## Land and exploration

`src/beach/CoastalBed.js` defines the rock layout and shared hydraulic/visible
heights. `CoastalRocks.js` builds the visible meshes; their bases follow the
sloping seabed. `RockWetness.js` maintains wetness history independent of the
camera, with separate fast surface-film and slower damp-rock behavior.

`src/beach/terrain.js` and `src/water/ExploreMesh.js` use moving detail windows
and smooth geometric transitions. `Palms.js` loads precomputed geometry from
`public/assets/palms-r6`; generators live in `tools/vendor/` and run offline.

`src/core/explore.js` handles walking, running, crouching, wading, swimming and
jumping. A small asynchronous GPU probe samples actual surface height for the
swimming viewpoint. The CPU also uses an approximate surface model for movement.

## Frame cost and loading

The fixed-step fluid simulation, FFT cascades, and precomputed lookup textures
are reused across water and effects. Existing programs stay within a conservative
16 combined-sampler budget. The shader budget check is in
`src/core/shaderBudget.js`; native tools can inspect linked ES 3.00 programs.

The loader prepares assets and shaders, warms real render paths, and performs
twelve seconds of simulated settling. Runtime rendering pixel budgets are
controlled by `src/core/viewport.js`. Changes in viewport size or driver behavior
can still affect latency after startup.

## Useful customization points

- `src/config.js`: beach profile, camera and scene configuration.
- `src/core/look.js`: lighting and visual calibration.
- `src/core/schedule.js`: wave events and deterministic timing.
- `src/core/viewport.js`: quality and resolution budget behavior.
- `src/beach/CoastalBed.js`: rock layout and bathymetry.
- `src/style.css` and `index.html`: interface and loading shell.

Tune one subsystem at a time and compare a fixed seed, time, and camera. Changes
to rock or bed geometry must agree with the hydraulic representation. New render
paths must participate in startup warmup and remain within texture-unit limits.

## Limits

This is an artistic and physically motivated coastal renderer. It is not a
coastal hazard model, geographic survey, or validated engineering simulation.
Native Mesa diagnostics validate shader and numerical behavior, but do not
certify all browser/driver combinations or establish hardware FPS.
