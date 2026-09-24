import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {compile,serializableArtifact} from '../vendor/cuda-webshader/compiler/compiler.js';
const source=await readFile(new URL('../ShoreBreak.cu',import.meta.url),'utf8');
await mkdir(new URL('../.qa/faithful/',import.meta.url),{recursive:true});
const entries=[...source.matchAll(/__global__\s+void\s+(\w+)\s*\(/g)].map(m=>m[1]);
for(const entry of entries){
 const a=serializableArtifact(compile(source,{entry,workgroupSize:entry.startsWith('evaluate')?[64,1,1]:[8,8,1]}));
 await writeFile(new URL(`../.qa/faithful/${entry}.json`,import.meta.url),JSON.stringify(a));
 console.log(entry,a.wgsl.length);
}
