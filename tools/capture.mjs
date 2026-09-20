// Deterministic frame capture + side-by-side comparison with the reference clip.
//
//   node tools/capture.mjs --times 0.5,3.3,5 [--w 540 --h 960] [--out dir] [--ref dir]
//   node tools/capture.mjs --seq 0:8.5:0.1 --out dir         (frame sequence + mp4)
//   --ref DIR        reference frames NNNN.jpg (60 fps numbering) -> compare_t*.png side by side
//   --crop x0,y0,x1,y1  normalized crop applied to both images in comparisons (zoom on the break)
//   --sheet          with --seq and --ref: contact sheet (ref row over render row) of the sequence
//   --params 'a=1&b=2'  extra URL params (e.g. hide=particles,lip)   --eval 'js'  run JS before capturing
//   --swe / --diag / --perf  debug dumps
//
// Starts its own Vite dev server, drives a GPU-backed headless Chromium (ANGLE/D3D11 on
// Windows) and calls window.__seek(t) / window.__advance(dt) exposed by src/main.js.
import { createServer } from 'vite';
import { chromium } from 'playwright-core';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import os from 'node:os';
import { execFileSync } from 'node:child_process';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const flag = (k) => args.includes('--' + k);
const W = parseInt(opt('w', '540'), 10), H = parseInt(opt('h', '960'), 10);
const out = path.resolve(opt('out', 'captures'));
const refDir = opt('ref', process.env.SHOREBREAK_REF || '');  // Optional local reference frames; not part of the source distribution.
const extra = opt('params', '');
fs.mkdirSync(out, { recursive: true });

function findChrome() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  for (const p of [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ]) if (fs.existsSync(p)) return p;
  const base = path.join(os.homedir(), 'AppData', 'Local', 'ms-playwright');
  const dirs = fs.existsSync(base) ? fs.readdirSync(base).filter((d) => /^chromium-\d+$/.test(d)).sort().reverse() : [];
  for (const d of dirs) {
    for (const sub of ['chrome-win64', 'chrome-win', 'chrome-linux', 'chrome-mac']) {
      const exe = path.join(base, d, sub, process.platform === 'win32' ? 'chrome.exe' : 'chrome');
      if (fs.existsSync(exe)) return exe;
    }
  }
  throw new Error('No Chromium found; set CHROME_PATH');
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
// --dist: test the production build (vite build -> dist/) through vite's static preview server
let server, port;
const BASE_URL = opt('url');   // --url https://...: capture a deployed site instead of a local server
if (BASE_URL) {
  server = { close: async () => {} };
} else if (flag('dist')) {
  const { preview } = await import('vite');
  server = await preview({ root, logLevel: 'error', preview: { port: 0, strictPort: false } });
  port = server.httpServer.address().port;
} else {
  server = await createServer({ root, logLevel: 'error', server: { port: 0, strictPort: false } });
  await server.listen();
  port = server.httpServer.address().port;
}
const browser = await chromium.launch({
  executablePath: findChrome(),
  headless: true,
  args: ['--headless=new'],
});
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
const logs = [];
page.on('console', (m) => { const t = m.text(); logs.push(`[${m.type()}] ${t.slice(0, 1500)}`); });
page.on('pageerror', (e) => logs.push('[pageerror] ' + e.message));
const url = `${BASE_URL ? BASE_URL.replace(/\/$/, '') : 'http://localhost:' + port}/?capture&nohud&w=${W}&h=${H}${extra ? '&' + extra : ''}`;
await page.goto(url, { waitUntil: 'commit', timeout: 180000 });
try {
  await page.waitForFunction(() => window.__ready === true, null, { timeout: 180000 });
} catch (e) {
  console.error(logs.join('\n'));
  throw e;
}
const gpu = await page.evaluate(() => {
  const c = document.createElement('canvas').getContext('webgl2');
  const e = c && c.getExtension('WEBGL_debug_renderer_info');
  return e ? c.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown';
});
console.log('GPU:', gpu);

async function save(file) {
  const data = await page.evaluate(() => window.__grab());
  fs.writeFileSync(file, Buffer.from(data.split(',')[1], 'base64'));
}
function refFrame(t) {
  if (!refDir) return null;
  const n = Math.round(t * 60) + 1;
  const f = path.join(refDir, String(n).padStart(4, '0') + '.jpg');
  return fs.existsSync(f) ? f : null;
}
const CROP = opt('crop') ? opt('crop').split(',').map(Number) : null;
function cropFilter() {
  if (!CROP) return '';
  const [x0, y0, x1, y1] = CROP;
  const cw = Math.round(W * (x1 - x0)), ch = Math.round(H * (y1 - y0));
  const k = Math.max(1, Math.min(4, Math.floor(900 / Math.max(cw, ch * 0.6))));
  return `,crop=${cw}:${ch}:${Math.round(W * x0)}:${Math.round(H * y0)},scale=${cw * k}:${ch * k}:flags=lanczos`;
}
function compose(render, t, dstName) {
  const ref = refFrame(t);
  if (!ref) return null;
  const dst = path.join(out, dstName || `compare_t${t.toFixed(2)}.png`);
  const cf = cropFilter();
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', ref, '-i', render, '-filter_complex',
    `[0]scale=${W}:${H}${cf}[a];[1]scale=${W}:${H}${cf}[b];[a][b]hstack`, dst]);
  return dst;
}

if (opt('eval')) console.log('eval:', JSON.stringify(await page.evaluate(opt('eval'))));
const t0 = Date.now();
if (opt('seq')) {
  const [a, b, s] = opt('seq').split(':').map(Number);
  await page.evaluate((t) => window.__seek(t), a);
  let i = 0;
  const tiles = [];
  for (let t = a; t <= b + 1e-6; t += s, i++) {
    if (i > 0) await page.evaluate((d) => window.__advance(d), s);
    const f = path.join(out, `seq_${String(i).padStart(4, '0')}.png`);
    await save(f);
    if (flag('sheet')) { const c = compose(f, a + i * s, `cmp_${String(i).padStart(4, '0')}.png`); if (c) tiles.push(c); }
  }
  if (flag('sheet') && tiles.length) {
    // grid of comparison tiles (each = ref | render); the tile filter pads the last row
    const per = Math.min(Number(opt('per', '4')), tiles.length);
    const rows = Math.ceil(tiles.length / per);
    const tw = Number(opt('tw', '720'));
    execFileSync('ffmpeg', ['-v', 'error', '-y', '-framerate', '1', '-i', path.join(out, 'cmp_%04d.png'),
      '-vf', `scale=${tw}:-2,tile=${per}x${rows}:padding=4:color=black`, '-frames:v', '1', path.join(out, 'sheet.png')]);
    console.log('wrote', path.join(out, 'sheet.png'));
  }
  const fps = Math.round(1 / s);
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-framerate', String(fps), '-i', path.join(out, 'seq_%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', path.join(out, 'render.mp4')]);
  console.log('wrote', path.join(out, 'render.mp4'));
} else {
  const times = opt('times', '0.5,3.3,3.6,5.0').split(',').map(Number);
  for (const t of times) {
    await page.evaluate((tt) => window.__seek(tt), t);
    const f = path.join(out, `render_t${t.toFixed(2)}.png`);
    await save(f);
    compose(f, t);
    if (flag('swe')) {
      const d = await page.evaluate(() => window.__sweDebug());
      fs.writeFileSync(path.join(out, `swe_t${t.toFixed(2)}.png`), Buffer.from(d.img.split(',')[1], 'base64'));
      delete d.img;
      console.log('swe', t, JSON.stringify(d));
    }
    console.log('captured', t);
  }
}
if (flag('diag')) {
  const d = await page.evaluate(() => window.__diag());
  console.log(JSON.stringify(d.filter((x) => x.diag), null, 1).slice(0, 6000));
  console.log('programs:', d.length);
}
if (flag('perf')) {
  const r = await page.evaluate(() => window.__bench?.(60));
  console.log('perf', JSON.stringify(r));
}
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
if (logs.length) console.log('--- console ---\n' + logs.slice(0, 40).join('\n'));
await browser.close();
await (server.close ? server.close() : server.httpServer.close());
