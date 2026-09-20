# Public source release validation

## Startup preparation update — 2026-09-20

- All 79 existing Node tests pass; all 40 bundled asset checksums still match.
- Production build passes with the expanded startup sequence.
- A recording-renderer lifecycle check runs the actual `src/main.js` startup
  with real scene constructors and delayed GPU-fence/readback substitutes.
  It confirms scenery and both lip slots are submitted, underwater geometry is
  drawn, pending readbacks complete, input unlocks only at readiness, and the
  camera and opening simulation time are restored.
- A failed-fence check confirms startup retains the loader and input lock.
- The cloud browser cannot create WebGL in this environment. These checks do
  not measure browser GPU frame timing or establish a zero-stutter guarantee.

## Original 0.1.0 package

Date: 2026-09-20. Source release: 0.1.0, based on ShoreBreak revision 23.

| Check | Result |
| --- | --- |
| Clean dependency installation | `npm ci` passed on Node.js 24.19.0 / npm 11.9.0, Linux |
| Existing Node tests | 79 passed; zero failures or skipped tests |
| Production build | Vite 8.3.0 build passed |
| Scene behavior | Main runtime bundle `main-D61zY6Lt.js` is byte-identical to revision 23 |
| Runtime source changes | Attribution comments only |
| Inherited texture origins | Six source JPEGs match Poly Haven's published MD5 and byte sizes |
| Bundled asset inventory | All 40 public files verified by SHA-256, size, and provenance entry |
| Palm source reproducibility | Regenerated all palm tiers; original asset checksums still pass |
| Relative documentation links | No missing linked files |
| Release hygiene scan | No recognized secret tokens, private keys, credential URLs, or internal workspace identifiers found |

The checks above ran locally in the release workspace. The first GitHub Actions
run also passed all 79 tests on Linux and Windows, plus the Linux production
build. The Windows asset check identified checkout line-ending conversion in
`public/favicon.svg`; `.gitattributes` now explicitly preserves its LF endings.
See the [Actions page](https://github.com/cryptomanavan/ShoreBreak/actions) for
the latest complete results on both operating systems.
No new hardware FPS measurements or browser visual tests were performed for
this packaging-only release. Earlier revision 23 native render validation is
not a substitute for testing on a contributor's own GPU and browser.

License notices were added for the project, photographic materials, and identified
third-party runtime code. Dependencies retain their existing licenses. The
public release does not bundle `node_modules`, compiled output, private references,
deployment account metadata, or local render recordings.
