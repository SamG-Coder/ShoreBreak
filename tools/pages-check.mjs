// Check the actual production server at a repository subpath without requiring
// a GPU on CI. Byte comparisons catch Vite's HTML fallback masking missing files.
import {readFile,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {preview} from 'vite';
import assert from 'node:assert/strict';
import {gunzipSync} from 'node:zlib';

const base=process.env.VITE_BASE_PATH||'/ShoreBreak/';
assert(base.startsWith('/')&&base.endsWith('/'),'Use an absolute project base path');
const server=await preview({base,logLevel:'error',preview:{host:'127.0.0.1',port:0}});
const origin=`http://127.0.0.1:${server.httpServer.address().port}`;
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
async function files(dir){return (await Promise.all((await readdir(dir,{withFileTypes:true})).map(async e=>e.isDirectory()?files(`${dir}/${e.name}`):[`${dir}/${e.name}`]))).flat();}
try{
 const html=await readFile('dist/index.html','utf8');
 for(const [,url]of html.matchAll(/(?:src|href)="([^"]+)"/g))if(!/^https?:/.test(url))assert(new URL(url,origin+base).pathname.startsWith(base),`HTML URL outside project path: ${url}`);
 const paths=await files('dist');
 const bundles=await Promise.all(paths.filter(p=>p.endsWith('.js')).map(p=>readFile(p,'utf8')));
 assert(bundles.some(s=>s.includes(base)),'Deployment base is missing from the JavaScript bundles');
 for(const s of bundles)assert(!/["'`]\/(?:assets|faithful)\//.test(s),'A runtime asset URL still points to the domain root');
 let next=0;
 await Promise.all(Array.from({length:8},async()=>{while(next<paths.length){const p=paths[next++],response=await fetch(origin+base+p.slice('dist/'.length));assert.equal(response.status,200,p);let expected=await readFile(p);if(response.headers.get('content-encoding')==='gzip'&&p.endsWith('.gz'))expected=gunzipSync(expected);assert.equal(hash(new Uint8Array(await response.arrayBuffer())),hash(expected),`Wrong content served for ${p}`);}}));
 console.log(`Verified ${paths.length} production files under ${base}, including CUDA artifacts and scene assets.`);
}finally{await server.close();}
