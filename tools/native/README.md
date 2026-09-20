# Native scene verification

These tools record the actual Three.js geometry, shader sources, textures,
uniforms, fixed-step simulation and draws, then replay them with ModernGL.
They are a software-rendering diagnostic, not a browser integration or FPS test.
MSAA is not reproduced. Cloud-browser WebGL was disabled during revision 08 work.

Run from the project root after `npm ci`. Python requires `moderngl`, `numpy`,
`Pillow` and a working EGL/OpenGL implementation (Mesa can run in software).

```sh
node tools/native/export.mjs
python3 tools/native/render.py
```

Intermediate recordings, checkpoints and PNGs are written to `.qa/` and can be
large (hundreds of MB). Exclude this directory from production and source archives.
The exporter defaults to 960×540, simulation time 2.85 s, with a full twelve-second
warmup. Set `QA_WIDTH=1280` for larger stills. For the larger-wave sequence:

```sh
QA_MOVIE=1 QA_START=40.4 QA_OUTPUT=.qa/crash.json node tools/native/export.mjs
QA_SOURCE=.qa/crash.json QA_CACHE=.qa/crash-cache.npz python3 tools/native/render.py
```

The movie recording has 72 frames at 24 Hz, with five 1/120-second simulation steps
between frames. `QA_REUSE=1` is only valid when the checkpoint matches the same
simulation, asset order and start time. It skips warmup replay; it is not a way to
validate changed physics. Shader-only diagnostics can use `QA_WATER_FRAG`.
Rendering code handles indexed draw ranges, instance counts, typed integer arrays
and framebuffer depth-write state explicitly.

Revision 08 review views and production-shader hydraulic tests:

```sh
QA_R8=1 QA_START=40.4 node tools/native/export.mjs
python3 tools/native/render.py
node tools/native/hydraulics.mjs
python3 tools/native/hydraulics.py
```

The hydraulic test covers equilibrium with partially dry rocks, moving-water
deflection relative to a sand-only control, positivity and wetness retention.
Material textures include roughness in normal-map alpha; the replay preserves it.

Revision 09 beach-window continuity and actual GPU water-surface samples:

```sh
QA_R9=1 QA_WIDTH=640 QA_START=40.4 QA_OUTPUT=.qa/r9.json node tools/native/export.mjs
QA_SOURCE=.qa/r9.json QA_CACHE=.qa/r9-cache.npz QA_PREFIX=r9- python3 tools/native/render.py
```

This freezes time across a terrain-window boundary and samples four seconds of
actual swimmer-height water at 30 Hz. The resulting `.qa/r9-probe.json` can be
passed as `QA_SWIM_TRACE=.qa/r9-probe.json` to a subsequent R9 export to render
swim views at the sampled waterline. Do not reuse caches across different target
sets or changed simulation/asset ordering.

Revision 10 underwater seabed, sky window, grazing ceiling, breaking-wave and
waterline views (21 frames):

```sh
QA_R10=1 QA_WIDTH=640 QA_START=40.4 QA_OUTPUT=.qa/r10.json node tools/native/export.mjs
QA_SOURCE=.qa/r10.json QA_CACHE=.qa/r10-cache.npz QA_PREFIX=r10- python3 tools/native/render.py
```

This renders the GPU surface probe before underwater optics, matching production.

Wide beach angles and walking across terrain detail windows:

```sh
QA_SAND=1 QA_WIDTH=900 QA_HEIGHT=323 QA_START=40.4 QA_OUTPUT=.qa/sand.json node tools/native/export.mjs
QA_SOURCE=.qa/sand.json QA_PREFIX=sand- python3 tools/native/render.py
```

Revision 11 submerged crash review, including side, upward and inside-plume views
and the full dispersal interval (23 views):

```sh
QA_R11=1 QA_WIDTH=640 QA_START=40.4 QA_OUTPUT=.qa/r11.json node tools/native/export.mjs
QA_SOURCE=.qa/r11.json QA_CACHE=.qa/r11-cache.npz QA_PREFIX=r11- python3 tools/native/render.py
```

The recorder/replay also preserves 3D texture data, dimensions and repeat state.

Revision 12 adds a shore-facing camera 8 m offshore, the impact/dispersal cycle,
close, side, upward, inside-plume and waterline views (27 total):

```sh
QA_R11=1 QA_R12=1 QA_WIDTH=640 QA_START=40.4 QA_OUTPUT=.qa/r12.json node tools/native/export.mjs
QA_SOURCE=.qa/r12.json QA_CACHE=.qa/r12-cache.npz QA_PREFIX=r12- python3 tools/native/render.py
```

Keep generated recordings and caches outside the project before deploying;
they are large, reproducible QA intermediates, not website assets.

Revision 13 adds fixed-camera terrain-window pairs at five cross-shore positions,
walk/run positions with water on the left, reverse views, and a shallow uprush /
backwash sequence (62 views):

```sh
QA_R13=1 QA_WIDTH=720 QA_START=44 QA_OUTPUT=.qa/r13.json node tools/native/export.mjs
QA_SOURCE=.qa/r13.json QA_CACHE=.qa/r13-cache.npz QA_PREFIX=r13- python3 tools/native/render.py
```

The terrain pairs hold time and camera fixed while the focus crosses a snap by
±20 micrometres, enough to remain distinct at float32 precision. Movement views
freeze the simulation to isolate spatial detail transitions; they do not measure
runtime camera frame pacing. `QA_SNAP_CROSS=1` upgrades the earlier ±1 micrometre
recording for replay. `QA_STOP_NAME=snap-z20-b` ends after all five pairs.

`QA_SHADER_OUTPUT=.qa/current-shaders node tools/native/export.mjs` extracts the
actual material source and defines without recording a new simulation. Fragment
and vertex overrides allow matched shader-only comparisons using the same cache:
`QA_UNDER_FRAG`, `QA_CLOUD_FRAG`, `QA_WATER_FRAG`, `QA_SAND_VERT`. Re-record when
geometry, asset/target order or simulation changes.

High run-up coverage and actual GPU flow/foam measurements:

```sh
QA_RUNUP=1 QA_WIDTH=720 QA_START=40.4 QA_OUTPUT=.qa/runup.json node tools/native/export.mjs
QA_SOURCE=.qa/runup.json QA_CACHE=.qa/runup-cache.npz QA_PREFIX=runup- python3 tools/native/render.py
```

This follows 11 seconds around the large set, measures depth and foam past z=3.4,
checks returning flow, and reports the depth in the outermost 16 reserve rows.
`QA_RUNUP_REVIEW=1` keeps seven representative views for geometry-only iteration.
The full run-up report also checks far-field water and foam. The hydraulic test
compares the dry-cell shortcut against the full production flux calculation.
A changed simulation grid requires a new warm checkpoint.


Revision 14 checks the shoreward underwater continuation, shallow side/near-bed
views and two matched pairs with its contribution disabled (25 views):

```sh
QA_R14=1 QA_WIDTH=640 QA_START=40.4 QA_OUTPUT=.qa/r14.json node tools/native/export.mjs
QA_SOURCE=.qa/r14.json QA_CACHE=.qa/r14-cache.npz QA_PREFIX=r14- python3 tools/native/render.py
```

GPU atlas readbacks produce `r14-coast.json` and false-color atlas PNGs for
coverage, finite-value and dry-reserve checks. `QA_SHADER_OUTPUT` also extracts
`coast.frag`, the complete production atlas-bake shader.
