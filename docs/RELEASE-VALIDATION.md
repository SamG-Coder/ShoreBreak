# Public source release validation

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

The included GitHub Actions workflow is configured for Linux and Windows. It
has not run on GitHub yet. The checks above ran locally in the release workspace.
No new hardware FPS measurements or browser visual tests were performed for
this packaging-only release. Earlier revision 23 native render validation is
not a substitute for testing on a contributor's own GPU and browser.

License notices were added for the project, photographic materials, and identified
third-party runtime code. Dependencies retain their existing licenses. The
public release does not bundle `node_modules`, compiled output, private references,
deployment account metadata, or local render recordings.
