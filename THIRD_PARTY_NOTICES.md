# Third-party notices

## WebCuda fork addition

CUDA WebShader compiler/runtime, Copyright (c) 2026 SamG-Coder and CUDA WebShader
contributors, MIT. Source: https://github.com/SamG-Coder/cuda-webshader at
f0f3699b498cfe6fe5419e072a4f4e2faa63b781. Full license: `vendor/cuda-webshader/LICENSE`.
The original project and asset notices below remain applicable.

The MIT license in the repository root covers original ShoreBreak contributions.
It does not replace the licenses of the components listed here. Keep their
required notices when redistributing source or a built application.

## Runtime components

| Component | License | Included notice / source |
| --- | --- | --- |
| Three.js 0.186.0 and add-ons | MIT | [License](licenses/three-MIT.txt), [upstream](https://github.com/mrdoob/three.js) |
| lil-gui 0.21.0 | MIT | [License](licenses/lil-gui-MIT.txt), [upstream](https://github.com/georgealways/lil-gui) |
| David Hoskins, Hash without Sine | MIT | [License](licenses/Hash-without-Sine-MIT.txt), [original](https://www.shadertoy.com/view/4djSRW) |
| Mulberry32 JavaScript implementation by bryc; original generator by Tommy Ettinger | Public domain | [Notice](licenses/Mulberry32-Public-Domain.txt), [implementation and dedication](https://github.com/bryc/code/blob/master/jshash/PRNGs.md) |
| Poly Haven photographic textures | CC0-1.0 | [Asset inventory](ASSETS.md), [CC0 text](licenses/CC0-1.0.txt), [provider policy](https://polyhaven.com/license) |

The Hash-without-Sine routines and adaptations occur in `src/glsl/common.js`,
`src/glsl/water.js`, `src/post/Post.js`, and `src/whitewater/emitters.js`.
Mulberry32 variants are used in the deterministic wave schedule, FFT initialization,
beach noise, and foam noise. Attribution comments identify these routines.

The FXAA shader is imported from Three.js's bundled add-ons. Its upstream header
credits the NVIDIA algorithm, Jasper Flick's C# implementation, and Dave Hoskins's
GLSL port. ShoreBreak includes it through the Three.js dependency rather than
relicensing or independently vendoring that add-on.

The complete runtime notices also live in
`public/THIRD-PARTY-NOTICES.txt`, which Vite copies into production output. Keep
that file with any distributed build.

## Development dependencies

Vite 8.3.0 is MIT licensed and Playwright Core 1.63.0 is Apache-2.0 licensed.
Their transitive dependencies have their own licenses, including MIT, ISC,
BSD-3-Clause, Apache-2.0 and MPL-2.0. The lockfile-based inventory is included in
`docs/dependency-licenses.json`. Installed packages contain their full upstream
license notices.

Development dependency source and `node_modules/` are not bundled in the source
ZIP or static site. `npm ci` retrieves the versions and integrity hashes in
`package-lock.json`. Installing those tools does not relicense the original
ShoreBreak application.

Optional native diagnostics require separately installed Python packages and
system graphics drivers, which retain their own licenses. No copies are shipped
in this repository. Research papers and reference websites are not licensed or
redistributed by this project.
