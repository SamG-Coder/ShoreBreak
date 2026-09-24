import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { compile, serializableArtifact } from '../vendor/cuda-webshader/compiler/compiler.js';
import { CONFIG } from '../src/config.js';
import { BK_TABLE } from '../src/glsl/breaker.js';

export const entries = {
  initializeSpectrum: [8, 8, 1], evolveSpectrum: [8, 8, 1], fftStage: [8, 8, 1],
  composeOcean: [8, 8, 1], initializeCoast: [8, 8, 1], stepSwash: [8, 8, 1],
  updateWetness: [8, 8, 1], buildSurface: [8, 8, 1], moveCamera: [1, 1, 1],
  renderCoast: [8, 8, 1], finishFrame: [8, 8, 1],
  animateSpray: [128, 1, 1], projectSpray: [128, 1, 1],
  materialMip: [8, 8, 1],
};
const f = x => Number.isInteger(x) ? `${x}.0f` : `${x}f`;
const raw = z => {
  let y = CONFIG.beach.knees[0][1] * z;
  for (const [k,a,b,w] of CONFIG.beach.knees) y += (b-a)*w*Math.log1p(Math.exp((z-k)/w));
  return y;
};
let header = `// Generated from ShoreBreak's original bathymetry and breaker stage table.\n`;
header += `__device__ float softplus(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }\n`;
header += `__device__ float bedProfile(float z) { float y = ${f(CONFIG.beach.knees[0][1])}*z;\n`;
for (const [z,a,b,w] of CONFIG.beach.knees) header += `y += ${f(b-a)}*${f(w)}*softplus((z-(${f(z)}))/${f(w)});\n`;
header += `return fmaxf(y + (${f(-raw(0))}), ${f(CONFIG.beach.minY)}); }\n`;
for (const [key, values] of Object.entries(BK_TABLE)) header += `__constant__ float BK_${key}[${values.length}] = {${values.map(f).join(',')}};\n`;
const units = await Promise.all(['common','ocean','coast','geometry','render','spray','materials'].map(n => readFile(new URL(`../kernels/${n}.cu`, import.meta.url),'utf8')));
const source = header + units.join('\n');
await mkdir(new URL('../public/generated/',import.meta.url), {recursive:true});
await writeFile(new URL('../kernels/shorebreak.generated.cu',import.meta.url),source);
const manifest = [], outputs=[];
for (const [entry, workgroupSize] of Object.entries(entries)) {
  const artifact = serializableArtifact(compile(source,{entry,workgroupSize,optimize:'dependencies'}));
  const text=JSON.stringify(artifact);
  outputs.push([entry,text]);
  manifest.push({entry,workgroupSize,bytes:artifact.wgsl.length,sha256:createHash('sha256').update(text).digest('hex'),bindings:artifact.metadata.bindings.map(x=>x.name)});
  console.log(`${entry}: ${artifact.wgsl.length} WGSL bytes`);
}
// Do not replace any published shader until every kernel compiles successfully.
for(const [entry,text] of outputs)await writeFile(new URL(`../public/generated/${entry}.json`,import.meta.url),text);
await writeFile(new URL('../public/generated/manifest.json',import.meta.url),JSON.stringify({sourceHash:createHash('sha256').update(source).digest('hex'),compilerCommit:'f0f3699b498cfe6fe5419e072a4f4e2faa63b781',kernels:manifest},null,2));
