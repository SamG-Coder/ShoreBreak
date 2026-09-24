# ShoreBreak — WebCuda

The original ShoreBreak calculations and graphics stages are ported into one application GPU source file: **[ShoreBreak.cu](ShoreBreak.cu)**. The default app uses that source through [WebCuda](https://github.com/SamG-Coder/cuda-webshader). The earlier approximate waves and ray renderer have been replaced.

Forked from [Christopher Canavan's ShoreBreak](https://github.com/cryptomanavan/ShoreBreak), created for [awakewithai.com](https://awakewithai.com). Original authorship, MIT licensing, photographic material attribution, palm credits and source history are preserved.

![WebCuda rendering of the original breaking wave](docs/screenshots/webcuda-wave.png)

## Run

Use Node.js 24 and a hardware-accelerated WebGPU browser:

```sh
npm ci
npm run dev
```

Open the localhost address printed by Vite. No CUDA Toolkit, separate compiler checkout or API key is required. Remote hosting requires HTTPS. The WebCuda compiler is vendored. Native graphics compilation can take several minutes at startup; the loading screen includes the original simulation warmup.

## What is preserved

- Original wave schedule, breaker profiles, ballistic lips and residual surface displacement.
- Three 256 x 256 FFT ocean cascades and the complete original shallow-water solver.
- Original foam, wetness, scrolling, far-field simulation and initialization.
- Original beach, scenery, instancing, water materials, whitewater, underwater effects and post-processing.
- Walking, swimming, crouching, camera controls and fixed-camera clip mode.

The GPU source is WebCuda CUDA, compiled to WebGPU programs. The original JavaScript host retains scheduling, input handling and CPU scene construction. Three.js CPU data and math classes remain; its renderer is replaced by `WebCudaRenderer`. No WebGL context is created by the production application.

## Build and tests

```sh
npm run build             # Compile the existing single CUDA file and build dist/
npm test                  # CPU regression and CUDA artifact/ABI checks
npm run test:faithful     # Original GLSL versus CUDA compute comparisons
npm run test:graphics     # Original GLSL versus actual CUDA graphics passes
npm run test:webcuda      # Complete browser startup, waves and state readback
npm run licenses:check    # Bundled asset attribution and checksums
npm run preview
```

GPU tests use Microsoft Edge on Windows; browser integration accepts `CHROME_PATH` for another Chromium executable. Test reports and captures are written under `.qa/faithful/`. Comparisons use floating-point tolerances; they do not promise bitwise equality between GPU drivers.

See [the integration details](docs/WEBCUDA.md), [calculation validation](docs/FAITHFUL-CUDA.md), and [asset provenance](docs/asset-provenance.json).

## Controls

Click to look; WASD or arrows to walk/swim; Shift to move faster; C to crouch or dive; Space to jump; P to pause; 1/2/3 for playback speed; R to reset; F for fullscreen; H for help; U to hide the interface; Escape to release the pointer. Touch controls use the left side to move and the right side to look.

## License

[MIT](LICENSE) for the original project and this port. CUDA WebShader retains its [MIT license](vendor/cuda-webshader/LICENSE). Included photographic textures remain CC0; see [asset credits](ASSETS.md), [third-party notices](THIRD_PARTY_NOTICES.md), and [the original README](docs/UPSTREAM-README.md).
