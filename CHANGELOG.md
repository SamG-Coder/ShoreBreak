# Changelog

## Startup preparation — 2026-09-20

- Draw off-camera scenery and both breaker slots behind the loader so their
  geometry and render state are prepared before the first turn.
- Warm moving water-mesh tables, waterline views, and underwater views facing
  both toward the sea and toward the beach.
- Rehearse 30 combined simulation/render frames, then restore the opening wave
  time and camera. Wait for eight stable final-size frames and pending GPU work.
- Keep scene controls out of keyboard focus until preparation finishes, and
  keep the loader visible if GPU synchronization cannot be established.
- Preserve all quality budgets, scene assets, shaders, and simulation settings.

## 0.1.0 — 2026-09-20

First public source package, based on the ShoreBreak revision 23 live build.

- MIT license, asset provenance, third-party notices, and contributor documentation.
- Pinned existing dependency versions, Node.js 24 setup, and Linux/Windows CI.
- Included procedural palm generators and all runtime assets.
- Asset checksum verification and portable developer capture paths.
- Browser capture tools use the browser's normal GPU support decisions.

This release preserves revision 23 scene behavior. Public-source preparation
adds documentation, legal notices, and development tooling, not a visual update.

### Included visual milestones

- Rippled underwater wave undersides with existing filtered FFT samples.
- Coastal rocks grounded into the seabed with continuous shading normals.
- Independent 100%, 150%, and 200% resolution control.
- Persistent rock wetness and continuous near/far foam handoff.
- Glossier water, sun glints, swimming, diving, crouching, and jump controls.
- Shallow-water swash, larger occasional wave sets, underwater churn and bubbles.
- Photographic sand and stone, procedural palms, and the Mediterranean promenade.
