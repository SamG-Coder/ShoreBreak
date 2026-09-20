// Swash kinematics probe: runs the scene headless, reads the shallow-water state back every
// --dt seconds and measures, in IMAGE space (mean camera pose = the "stabilized v" of the
// timeline analysis), the landward swash edge (wet/dry line) and the landward edge of dense
// foam per column u, plus the whitewater area fraction of the rendered frame. Prints them
// next to the reference curves (timeline analysis §9 CSV) so timing can be tuned numerically.
//
//   node tools/swashprobe.mjs --ref-csv path/to/swash_foam_0p1s.csv [--t0 0 --t1 8.5 --dt 0.1]
//        [--params 'inj=2'] [--eval 'js'] [--thr 0.002] [--foam 0.45] [--prof 0.2] [--json out.json]
//   --tune spec.json   coordinate-descent search of solver/injection parameters. spec.json is
//        [{ "k": "name", "set": "s.uInjJ.value.x = $", "v": 0.3, "step": 0.1, "min": 0, "max": 1 }, ...]
//        (in the setter, s = shared uniforms, w = SwashSim, m = its source-pass uniforms, f = foam-pass uniforms)
import { createServer } from 'vite';
import { chromium } from 'playwright-core';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import os from 'node:os';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const W = parseInt(opt('w', '540'), 10), H = parseInt(opt('h', '960'), 10);
const t0 = parseFloat(opt('t0', '0')), t1 = parseFloat(opt('t1', '8.5')), dt = parseFloat(opt('dt', '0.1'));
const thr = parseFloat(opt('thr', '0.002')), foamThr = parseFloat(opt('foam', '0.45'));
const extra = opt('params', '');
const US = [0.1, 0.25, 0.5, 0.75, 0.9];

function findChrome() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  for (const p of ['C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'])
    if (fs.existsSync(p)) return p;
  const base = path.join(os.homedir(), 'AppData', 'Local', 'ms-playwright');
  const dirs = fs.existsSync(base) ? fs.readdirSync(base).filter((d) => /^chromium-\d+$/.test(d)).sort().reverse() : [];
  for (const d of dirs) for (const sub of ['chrome-win64', 'chrome-win', 'chrome-linux']) {
    const exe = path.join(base, d, sub, process.platform === 'win32' ? 'chrome.exe' : 'chrome');
    if (fs.existsSync(exe)) return exe;
  }
  throw new Error('No Chromium found; set CHROME_PATH');
}

// reference curves
let ref = null;
const refCsv = opt('ref-csv', process.env.SHOREBREAK_REF_CSV || '');
if (refCsv && fs.existsSync(refCsv)) {
  const lines = fs.readFileSync(refCsv, 'utf8').trim().split(/\r?\n/);
  const head = lines[0].split(',');
  ref = lines.slice(1).map((l) => { const c = l.split(','); const o = {}; head.forEach((h, i) => { o[h] = c[i]; }); return o; });
}
const refAt = (t, key) => {
  if (!ref) return null;
  const r = ref.find((o) => Math.abs(parseFloat(o.t) - t) < 0.051);
  if (!r) return null;
  const v = parseFloat(r[key]);
  return Number.isFinite(v) ? v : (r[key] === '' || r[key] === 'OFF' || r[key] === 'null' ? 'OFF' : null);
};

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, logLevel: 'error', server: { port: 0, strictPort: false, watch: { ignored: ['**/*'] } } });
await server.listen();
const port = server.httpServer.address().port;
const browser = await chromium.launch({
  executablePath: findChrome(), headless: true,
  args: ['--headless=new'],
});
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text().slice(0, 800)}`));
page.on('pageerror', (e) => logs.push('[pageerror] ' + e.message));
await page.goto(`http://localhost:${port}/?capture&nohud&w=${W}&h=${H}${extra ? '&' + extra : ''}`);
await page.waitForFunction(() => window.__ready === true, null, { timeout: 180000 });
if (opt('eval')) console.log('eval:', JSON.stringify(await page.evaluate(opt('eval'))));

const refBands = opt('bands') && fs.existsSync(opt('bands')) ? JSON.parse(fs.readFileSync(opt('bands'), 'utf8')) : null;
const BANDS = refBands ? refBands.BANDS : [];
const shifts = {};
if (ref) for (const r of ref) shifts[(+r.t).toFixed(2)] = parseFloat(r.horizon_v) - 0.3468;
const xprofZ = opt('xprof') ? parseFloat(opt('xprof')) : null;
const measure = () => page.evaluate(async ({ t0, t1, dt, US, thr, foamThr, BANDS, shifts, xprofZ }) => {
  const { THREE, swe, renderer } = window.__scene;
  const { CONFIG, bedProfileJS, deg } = await import('/src/config.js');
  const C = CONFIG.camera, S = CONFIG.swe;
  const cam = new THREE.PerspectiveCamera(C.vfov, C.aspect, C.near, C.far);
  cam.position.fromArray(C.position); cam.rotation.order = 'YXZ';
  cam.rotation.set(deg(C.pitchDeg), deg(C.yawDeg), deg(C.rollDeg)); cam.updateMatrixWorld(); cam.updateProjectionMatrix();
  const vs = []; for (let v = 0.60; v <= 1.0001; v += 0.002) vs.push(v);
  const dx = (S.xMax - S.xMin) / S.nx, dz = (S.zMax - S.zMin) / S.nz;
  const ground = US.map((u) => vs.map((v) => {
    const d = new THREE.Vector3(2 * u - 1, 1 - 2 * v, 0.5).unproject(cam).sub(cam.position).normalize();
    let a = 0, b = 40;
    for (let k = 0; k < 50; k++) { const m = (a + b) / 2; const p = cam.position.clone().addScaledVector(d, m); if (p.y > Math.max(bedProfileJS(p.z), 0)) a = m; else b = m; }
    const p = cam.position.clone().addScaledVector(d, a);
    const i = Math.min(S.nx - 1, Math.max(0, Math.floor((p.x - S.xMin) / dx)));
    const j = Math.min(S.nz - 1, Math.max(0, Math.floor((p.z - S.zMin) / dz)));
    return { z: p.z, x: p.x, k: (j * S.nx + i) * 4 };
  }));
  const cv = document.createElement('canvas'); cv.width = renderer.domElement.width; cv.height = renderer.domElement.height;
  const cx = cv.getContext('2d', { willReadFrequently: true });
  const rows = [];
  window.__seek(t0);
  for (let t = t0; t <= t1 + 1e-6; t += dt) {
    if (t > t0 + 1e-6) window.__advance(dt);
    const rb = swe.readback();
    const s = rb.s, f = rb.f;
    const edge = [], bore = [], edgeZ = [];
    for (let c = 0; c < US.length; c++) {
      const g = ground[c];
      const wet = g.map((q) => s[q.k] > thr);
      const foam = g.map((q) => f[q.k] > foamThr);
      let e = null;
      for (let k = 0; k < g.length - 1; k++) {
        const n = Math.min(8, g.length - k); let dry = 0;
        for (let m = 0; m < n; m++) if (!wet[k + m]) dry++;
        if (dry / n > 0.55 && g[k].z > -0.8) { e = k; break; }
      }
      edge.push(e === null ? 'OFF' : +vs[e].toFixed(3)); edgeZ.push(e === null ? null : +g[e].z.toFixed(2));
      let b = null, seen = false;
      for (let k = 0; k < g.length - 1; k++) {
        if (foam[k]) seen = true;
        else if (seen) { let any = false; for (let m = 1; m <= 6 && k + m < g.length; m++) if (foam[k + m]) any = true; if (!any) { b = k; break; } }
      }
      bore.push(b === null ? (seen ? 'OFF' : null) : +vs[b].toFixed(3));
    }
    // whitewater fraction of the rendered frame below the horizon (timeline analysis classifier)
    cx.drawImage(renderer.domElement, 0, 0);
    const img = cx.getImageData(0, 0, cv.width, cv.height).data;
    const y0 = Math.round(0.347 * cv.height);
    let white = 0, tot = 0;
    for (let y = y0; y < cv.height; y += 2) for (let x = 0; x < cv.width; x += 2) {
      const p = (y * cv.width + x) * 4; const mx = Math.max(img[p], img[p + 1], img[p + 2]), mn = Math.min(img[p], img[p + 1], img[p + 2]);
      tot++; if (mx / 255 > 0.70 && (mx - mn) / Math.max(mx, 1) < 0.15) white++;
    }
    const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
    const sh = shifts[t.toFixed(2)] || 0;
    const bands = BANDS.map(([a, b]) => {
      const ya = Math.max(0, Math.round((a + sh) * cv.height)), yb = Math.min(cv.height, Math.round((b + sh) * cv.height));
      let r = 0, g = 0, bl = 0, n = 0, wh = 0;
      for (let y = ya; y < yb; y += 2) for (let x = Math.round(0.05 * cv.width); x < Math.round(0.95 * cv.width); x += 2) {
        const p = (y * cv.width + x) * 4; const R = img[p], G = img[p + 1], B = img[p + 2];
        const mx = Math.max(R, G, B), mn = Math.min(R, G, B);
        if (mx / 255 > 0.70 && (mx - mn) / Math.max(mx, 1) < 0.15) wh++;
        r += lin(R); g += lin(G); bl += lin(B); n++;
      }
      return { rgb: [r / n, g / n, bl / n].map((v) => +v.toFixed(3)), white: +(wh / Math.max(n, 1)).toFixed(3) };
    });
    let mass = 0, maxV = 0, nan = 0;
    for (let k = 0; k < s.length; k += 4) { if (!Number.isFinite(s[k])) { nan++; continue; } if (s[k] > 1e-4) { mass += s[k]; maxV = Math.max(maxV, Math.hypot(s[k + 1], s[k + 2])); } }
    // cross-shore profile at x = 0: h (cm), v (m/s), foam R
    const prof = [];
    const i0 = Math.floor((0 - S.xMin) / dx);
    for (let z = -2.4; z <= 2.61; z += 0.2) {
      const j = Math.floor((z - S.zMin) / dz), k = (j * S.nx + i0) * 4;
      prof.push([+(s[k] * 100).toFixed(1), +s[k + 2].toFixed(2), +f[k].toFixed(2), +f[k + 1].toFixed(2), +f[k + 3].toFixed(2), +f[k + 2].toFixed(2)]);
    }
    // along-shore depth profile (mm) at z = xprofZ, x = -1.2 .. 1.2 every 6 cm
    const xprof = [];
    if (xprofZ !== null) {
      const j = Math.floor((xprofZ - S.zMin) / dz);
      for (let x = -1.2; x <= 1.2001; x += 0.06) { const i = Math.floor((x - S.xMin) / dx); xprof.push(Math.round(s[(j * S.nx + i) * 4] * 1000)); }
    }
    rows.push({ t: +t.toFixed(2), edge, edgeZ, bore, white: +(white / tot).toFixed(3), mass: +(mass * dx * dz).toFixed(3), maxV: +maxV.toFixed(2), nan, prof, bands, xprof });
  }
  return rows;
}, { t0, t1, dt, US, thr, foamThr, BANDS, shifts, xprofZ });

// error vs the reference: swash edge (OFF = 1.0) on all 5 columns, dense-foam edge where both exist
function score(res) {
  let e = 0, n = 0, eb = 0, nb = 0, nan = 0;
  for (const r of res) {
    nan += r.nan;
    US.forEach((u, i) => {
      const q = refAt(r.t, `swash_u${u}`); const v = r.edge[i];
      const a = v === 'OFF' ? 1.0 : v, b = q === 'OFF' ? 1.0 : q;
      if (typeof a === 'number' && typeof b === 'number') { e += (a - b) ** 2; n++; }
    });
    [0.25, 0.5, 0.75].forEach((u, i) => {
      const q = refAt(r.t, `bore_u${u}`); const v = r.bore[i + 1];
      if (typeof q === 'number') { const a = typeof v === 'number' ? v : (v === 'OFF' ? 1.0 : 0.6); eb += (a - q) ** 2; nb++; }
    });
  }
  const edge = Math.sqrt(e / Math.max(n, 1)), bore = Math.sqrt(eb / Math.max(nb, 1));
  // image bands (swash zone): mean luminance and white-foam fraction vs the reference
  let ey = 0, ew = 0, nbd = 0;
  const Y = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  if (refBands) for (const r of res) {
    const fr = refBands.frames.find((o) => Math.abs(o.t - r.t) < 0.051); if (!fr) continue;
    for (let i = 2; i < BANDS.length; i++) {
      ey += (Y(r.bands[i].rgb) - Y(fr.bands[i].rgb)) ** 2; ew += (r.bands[i].white - fr.bands[i].white) ** 2; nbd++;
    }
  }
  const bandY = Math.sqrt(ey / Math.max(nbd, 1)), bandW = Math.sqrt(ew / Math.max(nbd, 1));
  const wBand = parseFloat(opt('wband', '0'));
  return { edge, bore, bandY, bandW, total: edge + 0.35 * bore + wBand * (0.3 * bandY + bandW) + (nan ? 1 : 0) };
}

if (opt('tune')) {
  const spec = JSON.parse(fs.readFileSync(opt('tune'), 'utf8'));
  const setAll = (vals) => page.evaluate((code) => { const s = window.__params.shared, w = window.__scene.swe, m = w.pSrc.material.uniforms, f = w.pFoam.material.uniforms; eval(code); return 1; },
    spec.map((p, i) => p.set.replace(/\$/g, String(vals[i]))).join(';'));
  const evalAt = async (vals) => { await setAll(vals); return score(await measure()); };
  let cur = spec.map((p) => p.v), steps = spec.map((p) => p.step);
  let best = await evalAt(cur);
  console.log('start', JSON.stringify(best), cur.join(','));
  const iters = parseInt(opt('iters', '6'), 10);
  for (let it = 0; it < iters; it++) {
    let improved = false;
    for (let i = 0; i < spec.length; i++) {
      for (const dir of [1, -1]) {
        const v = Math.min(spec[i].max, Math.max(spec[i].min, +(cur[i] + dir * steps[i]).toFixed(5)));
        if (v === cur[i]) continue;
        const trial = cur.slice(); trial[i] = v;
        const sc = await evalAt(trial);
        if (sc.total < best.total - 1e-4) { best = sc; cur = trial; improved = true; console.log(`  ${spec[i].k}=${v}  ->  ${sc.total.toFixed(4)} (edge ${sc.edge.toFixed(4)} bore ${sc.bore.toFixed(4)})`); break; }
      }
    }
    if (!improved) steps = steps.map((s) => s / 2);
    console.log(`iter ${it}: ${best.total.toFixed(4)}  ` + spec.map((p, i) => `${p.k}=${cur[i]}`).join(' '));
  }
  console.log('BEST', JSON.stringify(best), '\n' + spec.map((p, i) => p.set.replace(/\$/g, String(cur[i]))).join('; '));
  await browser.close(); await server.close();
  process.exit(0);
}

const res = await measure();
const fmt = (v) => (v === null || v === undefined ? '  -  ' : typeof v === 'number' ? v.toFixed(3) : String(v).padEnd(5));
console.log('   t  | edge sim u.1 .25 .5 .75 .9          | edge ref                             | z.5 sim | bore sim .25 .5 .75   | bore ref .25 .5 .75   | white sim/ref | mass   maxV');
for (const r of res) {
  const er = US.map((u) => refAt(r.t, `swash_u${u}`));
  const br = [0.25, 0.5, 0.75].map((u) => refAt(r.t, `bore_u${u}`));
  const wr = refAt(r.t, 'whiteArea');
  console.log(`${r.t.toFixed(2).padStart(5)} | ${r.edge.map(fmt).join(' ')} | ${er.map(fmt).join(' ')} | ${String(r.edgeZ[2]).padStart(6)} | ${[r.bore[1], r.bore[2], r.bore[3]].map(fmt).join(' ')} | ${br.map(fmt).join(' ')} | ${fmt(r.white)} ${fmt(wr)} | ${r.mass.toFixed(2)} ${r.maxV.toFixed(2)}${r.nan ? ' NaN ' + r.nan : ''}`);
}
if (opt('prof')) {
  const every = parseFloat(opt('prof'));
  console.log('profile at x=0, z = -2.4 .. 2.6 step 0.2:  h[cm]/v[m/s]');
  let hdr = '   t  |'; for (let z = -2.4; z <= 2.61; z += 0.2) hdr += (z.toFixed(1)).padStart(9); console.log(hdr);
  for (const r of res) {
    const k = r.t / every; if (Math.abs(k - Math.round(k)) > 1e-3) continue;
    console.log(`${r.t.toFixed(2).padStart(5)} |` + r.prof.map(([h, v, R, G, M, K]) => (h > 0.05 ? (opt('pf') ? `${R.toFixed(1)}/${G.toFixed(1)}/${(opt('pf') === 'k' ? K : M).toFixed(1)}` : `${h.toFixed(1)}/${v.toFixed(1)}`) : '.').padStart(opt('pf') ? 12 : 9)).join(''));
  }
}
if (refBands) {
  console.log('bands (stabilized v): ' + BANDS.map(([a, b]) => `${a.toFixed(2)}-${b.toFixed(2)}`.padEnd(16)).join(''));
  console.log('   t  | per band: Ysim/Yref  whiteSim/whiteRef');
  for (const r of res) {
    const k = r.t / 0.2; if (Math.abs(k - Math.round(k)) > 1e-3) continue;
    const fr = refBands.frames.find((o) => Math.abs(o.t - r.t) < 0.051); if (!fr) continue;
    const Y = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    console.log(`${r.t.toFixed(2).padStart(5)} | ` + r.bands.map((b, i) => `${Y(b.rgb).toFixed(2)}/${Y(fr.bands[i].rgb).toFixed(2)} ${b.white.toFixed(2)}/${fr.bands[i].white.toFixed(2)}`.padEnd(24)).join(''));
  }
}
if (xprofZ !== null) for (const r of res) console.log(`xprof z=${xprofZ} t=${r.t.toFixed(2)}: ` + r.xprof.join(' '));
const sc = score(res);
console.log(`swash-edge rms error (v units, OFF = 1.0): ${sc.edge.toFixed(4)}   foam-edge rms: ${sc.bore.toFixed(4)}   band Y rms ${sc.bandY.toFixed(4)}  band white rms ${sc.bandW.toFixed(4)}   total ${sc.total.toFixed(4)}`);
if (opt('json')) fs.writeFileSync(opt('json'), JSON.stringify(res));
const errs = logs.filter((l) => /error|warn/i.test(l) && !/404/.test(l));
if (errs.length) console.log('--- console ---\n' + errs.slice(0, 20).join('\n'));
await browser.close();
await server.close();
