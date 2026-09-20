# Developer guide

## Setup

Use Node.js 24.x. With nvm, run `nvm install` followed by `nvm use`; Windows users
can install Node.js 24 directly or select it in their version manager.

```sh
npm ci
npm run dev
```

`npm ci` uses the checked-in lockfile. The source release pins the versions
already used by revision 23; no dependency upgrade was made for this release.
`private: true` in `package.json` prevents accidental publication to the npm
registry. It does **not** restrict the public GitHub repository or MIT license.

## Verification

```sh
npm test
npm run licenses:check
npm run build
npm run preview
```

Node tests cover timing, exploration, surface probes, terrain and wave LOD,
resolution controls, rock grounding, run-up boundaries, and shader-budget logic.
They do not render a full browser scene. The CI workflow runs these same checks
on Windows and Linux; inspect the actual workflow results after publishing.

For a visual change, inspect both above and below water, the largest run-up,
walking parallel to shore, and the relevant LOD transitions. Keep camera, time,
seed, viewport, and quality fixed in before/after comparisons.

## Asset verification and regeneration

`npm run licenses:check` verifies SHA-256 hashes and complete coverage of runtime
assets against `docs/asset-provenance.json`. If an asset changes intentionally,
update its provenance and hash after reviewing its source and license.

```sh
npm run bake:palms
npm run licenses:check
```

The included palm generators produce two seeds and three geometry tiers.
Textures are already bundled and need no download at build time. The source
hashes in the provenance manifest identify their original Poly Haven files.

## Browser captures (optional)

Install Chrome or Chromium with working WebGL 2. The tool detects common
installation paths; set `CHROME_PATH` to the executable if it cannot find one.
`playwright-core` does not download a browser. Capture tools respect the browser's
normal GPU availability and do not bypass its GPU blocklist or sandbox.

```sh
npm run capture -- --times 0.5,3.3,5 --w 960 --h 540
```

The tool starts a local Vite server and writes to the ignored `captures/` folder.
Use `--dist` after building to capture the production output. `--ref` optionally
accepts your own reference-frame directory; no reference footage is included.
FFmpeg must be installed separately for video sequences or image comparisons.

If the browser has no WebGL context, use a supported local browser/GPU. A build
success alone does not imply the browser can execute the renderer.

## Native diagnostics (optional, Linux/EGL)

The tools under `tools/native/` use Python, ModernGL, NumPy and Pillow, plus a
working EGL/OpenGL driver. Install them in your own virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install moderngl numpy pillow
```

For a fast production-shader check, run from the project root:

```sh
QA_PROGRAMS=.qa/programs.json node tools/native/export.mjs
python tools/native/audit-programs.py .qa/programs.json .qa/audit.json --es --enforce
```

Full replay uses much more memory and can take several minutes with software
rendering. See [the native guide](../tools/native/README.md). Keep recordings,
warmup caches, and frame dumps out of Git. Native replay does not reproduce every
browser integration detail or MSAA behavior.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Node engine or install error | Select Node.js 24.x and rerun `npm ci` |
| Blank page when opening a file | Use `npm run dev`; do not open `index.html` directly |
| Long initial loading | Allow shader compilation and simulation warmup to finish |
| Poor frame rate | Start with Auto and 100%; turn off resolution boosts |
| WebGL or shader error | Check hardware acceleration, driver/browser support, and console logs |
| Missing asset after deploy | Build with npm and publish the complete `dist/` directory |
| Asset verification failure | Restore original assets or document/review intentional replacements |

The renderer contains absolute `/assets/` lookups. Deploy at the root of a domain
or subdomain. A host that serves under a repository subpath requires updating
those asset URLs as well as Vite's base setting.
