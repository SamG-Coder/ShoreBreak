# Faithful calculation port

The corrected calculation source is **`ShoreBreak.cu`**, at the repository root.
The WebCuda build reads this one file and discovers its kernel entry points. It
does not assemble the rejected renderer's `kernels/*.cu` files.

This work is on `faithful-cuda-calculations`. The existing default application
still runs the earlier, rejected approximation. The corrected calculation kernels
have not yet been connected to a replacement renderer. **This is not a completed
application migration or a claim that the currently displayed waves are fixed.**

## Ported equations

- The complete original breaker stage, transport, profile, derivative, ballistic
  lip, injection and foam functions from `src/glsl/breaker.js` and its dependencies.
- The original `sheetRaw` cross-section from `src/water/LipRibbon.js`: crest skin,
  shoulder, Bezier jet, hanging curtain, cap, underside, fillet and barrel wall.
  View-frustum culling and the residual full-surface displacement remain renderer work.
- Three 256 × 256 ocean cascades with the original spectrum evolution, all 16
  Stockham FFT passes and displacement/slope composition from `OceanFFT.js`.
- The original Kurganov–Petrova flux, including minmod reconstruction,
  desingularized velocities, wet/dry fallback, hydrostatic balance and boundaries.
- Bed/state initialization, source injection and friction, foam and lace transport,
  wetness, surface height/view, scrolling and far-field passes from `SwashSim.js`.

The calculation port has 19 kernel entry points in one translation unit. It uses
WebCuda's CUDA subset, including vector arithmetic and reference output parameters.
This file targets WebCuda; compilation with NVIDIA nvcc has not been established.

## Build and comparison

```sh
npm run compile:faithful
npm run test:faithful
```

`compile:faithful` compiles the existing `.cu` file directly. Test fixtures and
upstream shader metadata are exported separately, without regenerating that file.
The comparison tests require Microsoft Edge with a working WebGPU adapter on Windows.

The test oracle executes the **original GLSL** in a WebGL2 RGBA32F framebuffer.
The CUDA kernels run through the vendored WebCuda compiler/runtime on WebGPU, with
the same float32 inputs, original event packing and original lookup tables.
Results are read back and compared numerically. WebGL and Three.js in these tests
are reference infrastructure, not a proposed production rendering dependency.

The FFT test checks every pass both independently, with identical inputs, and in
the complete evolving-spectrum chain. The flux test includes wet/dry cells and
compares 1, 6 and 36 substeps. The swash test exercises all 13 auxiliary passes.
Wave and lip tests sample 4,096 positions/times/rows each. Reports are written to
`.qa/faithful/` and include worst differences and failing sample counts.

Different GPU math and texture units do not promise bit-for-bit equality. The
lookup sampler performs linear interpolation in float32. The original NVIDIA
texture unit quantizes its interpolation weights, so direct lookup diagnostics
use a local, table-slope-based bound for half an eight-bit filter-weight step plus
two coordinate ULPs. Other GPU vendors need their own comparison run. The FFT slope tolerance allows
the accumulated spectrum-phase rounding (8e-6 absolute plus 2e-4 relative); the
isolated FFT tolerance is 2e-7 absolute plus 2e-6 relative. Displacement is checked
at 2e-6 absolute plus 2e-4 relative. The wave/swash diagnostic tolerance is 2e-4
absolute plus 2e-4 relative; individual measured maxima remain in the reports.

## What remains

The application must still dispatch these kernels at the original resolutions,
step order and texture precisions, retain the original schedule and initialization,
and port the full water geometry/shading, whitewater, underwater and post passes.
Spectrum initialization and event scheduling currently come from the original
JavaScript in the tests. The tests have not established long-duration coupled
simulation stability or final-image parity. The earlier approximate renderer and
its screenshots must not be used as evidence for the corrected port.

The `tools/port-*.mjs` scripts record the mechanical source translation used to
create the file. They are authoring aids, not part of its normal compilation.
Original upstream source files remain unchanged as the reference.
