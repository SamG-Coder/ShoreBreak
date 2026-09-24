# WebCuda integration

The application starts at `src/boot.js`, then `src/main.js`. The original host code retains the scene, controls, deterministic schedule, initialization, texture precision, simulation resolutions, pass order and camera. `WebCudaRenderer` in `src/webcuda/renderer.js` replaces Three.js rendering with native WebGPU rendering of programs compiled from the single root `ShoreBreak.cu`.

The previous ray renderer and simplified wave equations have been removed. The original three 256 x 256 FFT cascades, Kurganov–Petrova flux, six substeps per 1/120-second step, scrolling shallow-water window, wave lookup tables, full lip geometry, whitewater, underwater effects and post-processing are retained.

## Source and build

`ShoreBreak.cu` is the canonical application GPU source. Each graphics stage is a conditional section in that file. The compute entry points provide separate numerical diagnostics of the same original calculations.

`npm run compile:cuda` reads the existing CUDA file directly and compiles each graphics entry with the vendored WebCuda compiler. It does not regenerate the CUDA from JavaScript or GLSL. `tools/graphics-metadata.json` describes stage IO and maps the original material identities to the corresponding CUDA stages. `public/faithful/graphics/build.json` records SHA-256 hashes of the CUDA source, compiler, metadata and generated artifacts. An unchanged build reuses verified artifacts.

The optional `tools/port-*.mjs` files are authoring aids used for the mechanical translation. Original GLSL remains in the original host modules and in the test oracle. The production renderer uses it only to identify materials; it never compiles or executes it.

## GPU boundary

`graphics-stage.js` supplies native stage IO, texture operations, derivatives, clip-space conversion and rasterization bindings to WebCuda's emitted WGSL. Application shading and simulation expressions come from CUDA. Matrix and vector helpers are CUDA functions in the same source file. A per-invocation context retains the semantics of the original shader globals.

The renderer manages native textures, depth buffers, MRT attachments, mipmaps, geometry uploads, uniforms, blending, culling and asynchronous readback. It uses the original mesh geometry and instance transforms. GPU work is submitted in order, including before host updates of buffers used by queued draws.

Three.js CPU scene, geometry, texture descriptors and math classes remain in the host. No Three.js WebGLRenderer or WebGPURenderer is instantiated. Browser event handling, scheduling and CPU scene construction remain JavaScript.

## Platform and validation

This targets the WebCuda CUDA subset, including graphics texture and derivative intrinsics added to the vendored compiler. It is not a verified nvcc application. The vendored base is CUDA WebShader revision `f0f3699b498cfe6fe5419e072a4f4e2faa63b781`; local graphics extensions are included.

Use a WebGPU browser on localhost or HTTPS, with hardware acceleration. The graphics path needs float32 texture filtering and blending. Native driver shader compilation is substantial and can take several minutes during startup; the loading screen stays visible while compilation and the original warmup complete.

Numerical GPU comparisons run original GLSL and CUDA on identical inputs and read back results. They cover full FFT chains, repeated flux steps, auxiliary swash passes, breaker profiles and lip geometry. Full browser tests exercise the actual integrated renderer. See `FAITHFUL-CUDA.md` for test commands and tolerances. Passing these checks does not assert bitwise image equality across different GPU drivers.
