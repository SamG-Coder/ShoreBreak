# Contributing to ShoreBreak

ShoreBreak explores realistic coastal water within a practical browser GPU
budget. Improvements to stability, visual continuity, physical plausibility,
accessibility, and measured performance are welcome.

## Development

1. Fork and clone the repository, then create a branch for your change.
2. Use Node.js 24.x and run `npm ci`.
3. Run `npm run dev` and make a focused change.
4. Run `npm test`, `npm run licenses:check`, and `npm run build`.
5. Open a pull request describing the problem, the change, and validation.

For rendering changes, include before/after captures using the same seed,
simulation time, camera, quality, and resolution. Check the surface, underwater
view, shallow run-up, and moving camera when they are affected. For performance
claims, include the GPU, browser, viewport, pixel ratio, and settings; do not
compare runs with different pixel budgets.

Preserve the existing startup preload and shader warmup behavior. Watch sampler
counts when adding textures: the project enforces a conservative 16-sampler
combined budget. Avoid introducing hard transitions in foam, wetness, or LOD.

## Assets and attribution

Submit only work you have the right to contribute. New external assets need an
explicit redistribution license, source URL, author credit, and provenance
entry. Preserve third-party notices. Do not commit reference videos, secrets,
`node_modules/`, `dist/`, or local GPU recordings.

Contributions to original project code are accepted under the project's MIT
license. External components retain their own clearly documented licenses.
Discuss substantial dependency, architecture, or license changes before opening
a large pull request. Keep discussion constructive and specific.
