# ShoreBreak

**Real-time breaking waves in your browser.** Incoming swell, curling crests,
crash foam, shallow-water swash, backwash, and underwater optics, built with
Three.js r186 and WebGL 2.

Created by **Christopher Canavan / [awakewithai.com](https://awakewithai.com)**.

**[Explore the live demo](https://shorebreak-living-coast.netlify.app/)** ·
[How it works](docs/ARCHITECTURE.md) · [Asset credits](ASSETS.md) ·
[Contributing](CONTRIBUTING.md) · [MIT license](LICENSE)

## Run locally

Install **Node.js 24.x**, which includes npm. Clone the repository and start
the development server:

```sh
git clone https://github.com/cryptomanavan/ShoreBreak.git
cd ShoreBreak
npm ci
npm run dev
```

If you downloaded the source ZIP, extract it and run `npm ci` and `npm run dev`
in the folder containing `package.json`.

Open the local URL printed by Vite (normally `http://localhost:5173`). Let the
loader finish, then click the scene to look around. No API keys, accounts,
environment variables, paid assets, or separate asset downloads are needed.
All runtime assets are included. The first `npm ci` needs internet access.

Do not double-click `index.html`: the application needs an HTTP server.

## Requirements

- A browser with WebGL 2, hardware acceleration, and floating-point render
  targets. A desktop computer with a capable GPU is recommended.
- Keyboard and mouse provide the full experience; touch controls are included.
- The project targets Node.js 24.x for development and builds. `.nvmrc` and
  `.node-version` specify this version. Python is optional for native GPU diagnostics.

## Controls

| Input | Action |
| --- | --- |
| Click / mouse | Capture the pointer / look around |
| WASD or arrow keys | Walk, wade, or swim |
| Shift | Run or swim faster |
| C | Crouch on land; dip underwater while swimming |
| Space | Jump on land |
| P | Pause the waves |
| 1 / 2 / 3 | Real time / quarter speed / tenth speed |
| F | Fullscreen |
| H | Show controls |
| U | Hide the interface |
| R | Restart the wave sequence |
| Esc | Release the pointer |

Swimming starts automatically in deeper water. The **Steady camera** setting
reduces gait motion; reduced-motion preferences are also supported.

## Quality and performance

Start with **Auto quality and 100% resolution**. Auto adjusts the rendering
pixel budget while keeping the fluid simulation resolution fixed. Low, Medium,
High, and Ultra select other pixel budgets. The scene occupies 75% of the
browser's area, preserving its aspect ratio.

Resolution boosts multiply each render dimension: 150% uses approximately
2.25 times the pixels, and 200% uses four times the pixels, subject to hardware
limits. Boosts hold the selected resolution; return to 100% for Auto's adaptive
resolution. These settings persist in local storage.

The loader fetches local assets, compiles shaders, uploads off-camera scenery,
rehearses moving shoreline and underwater views, and runs animated simulation
and render frames before enabling controls. It then restores the opening wave
time and waits for the final render buffers to settle. Loading time and frame rate
depend on the browser, GPU, viewport, and selected settings. There is no fixed
FPS or zero-stutter guarantee.

## Build and test

```sh
npm test
npm run licenses:check
npm run build
npm run preview
```

The build is written to `dist/`; preview serves that build locally.
`npm run check` runs the Node tests followed by a clean production build.
The GitHub Actions workflow runs tests, asset-integrity checks, and builds on
Linux and Windows. GPU visual inspection remains a separate check.

Optional developer tools:

- `npm run bake:palms`: regenerate the included procedural palm geometry.
- `npm run capture -- --times 3.3`: capture deterministic frames using an
  installed Chrome/Chromium browser. See [development](docs/DEVELOPMENT.md).
- [Native GPU diagnostics](tools/native/README.md): optional Python/Mesa tools
  for shader compilation, sampler budgets, and deterministic scene replay.

## What is being simulated?

The visible ocean uses **meshes**: a three-cascade FFT wind sea, a scheduled
breaker profile, and separate plunging-lip geometry. A GPU shallow-water solver
handles swash and backwash. Foam, wetness, particles, and underwater effects are
coupled to those fields. Underwater churn includes a bounded volume-marching
pass; that does not make the main ocean surface ray marched.

This is a physically motivated real-time graphics approximation, not a full
three-dimensional Navier–Stokes fluid simulation. The Mediterranean scenery
and pale sandy beach are an artistic interpretation, not a surveyed recreation
of Nice. [Read the architecture guide](docs/ARCHITECTURE.md).

## Deploy or publish the source

The site is static. Netlify can build it with `npm run build` and publish `dist/`;
the included `netlify.toml` sets these defaults. No server functions are required.
See [publishing instructions](docs/PUBLISHING.md) for GitHub, Netlify, and ZIP releases.

## License and credits

Original project code, procedural geometry, and documentation are released under
the [MIT license](LICENSE). Photographic materials retain their **CC0-1.0**
license. Third-party components retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The public package includes all runtime assets, their origins, and a checksum
manifest in [ASSETS.md](ASSETS.md) and [asset provenance](docs/asset-provenance.json).
Reference videos and reference photographs are not bundled.
