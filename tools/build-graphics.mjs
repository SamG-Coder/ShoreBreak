import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {compile,serializableArtifact} from '../vendor/cuda-webshader/compiler/compiler.js';
const source=await readFile('ShoreBreak.cu','utf8'),metadata=JSON.parse(await readFile('tools/graphics-metadata.json','utf8'));
const sha256=s=>createHash('sha256').update(s).digest('hex');
const inputs={sha256:sha256(source),compiler:sha256(await readFile('vendor/cuda-webshader/compiler/compiler.js')),metadata:sha256(JSON.stringify(metadata))};
await mkdir('public/faithful/graphics',{recursive:true});
let cached=false;
try{const previous=JSON.parse(await readFile('public/faithful/graphics/build.json','utf8'));cached=Object.entries(inputs).every(([k,v])=>previous[k]===v)&&previous.records.length===Object.keys(metadata.stages).length*2;
 if(cached)for(const r of previous.records)if(sha256(await readFile(`public/faithful/graphics/${r.id}-${r.stage}.json`))!==r.sha256){cached=false;break;}
 if(cached&&await readFile('public/faithful/graphics/index.json','utf8')!==JSON.stringify(metadata.index))cached=false;
}catch{}
if(cached){console.log('CUDA graphics artifacts match ShoreBreak.cu and the compiler.');process.exit(0);}
const records=[];
for(const [id,stages]of Object.entries(metadata.stages))for(const stage of ['vertex','fragment']){
 const key='SB_GRAPHICS_'+id+'_'+stage,start=source.indexOf('#ifdef '+key+'\n');if(start<0)throw Error('Missing CUDA stage '+key);
 // Each section is already preprocessed GLSL-to-CUDA and self-contained. Select
 // its translation unit from the canonical file without regenerating any math.
 const body=source.slice(start+('#ifdef '+key+'\n').length).split('\n#endif')[0];
 const artifact=serializableArtifact(compile(body,{entry:'graphicsStage',workgroupSize:[1,1,1]}));
 const text=JSON.stringify({...artifact,...stages[stage]});await writeFile(`public/faithful/graphics/${id}-${stage}.json`,text);
 records.push({id,stage,sha256:createHash('sha256').update(text).digest('hex')});console.log('CUDA graphics',id,stage);
}
await writeFile('public/faithful/graphics/index.json',JSON.stringify(metadata.index));
await writeFile('public/faithful/graphics/build.json',JSON.stringify({source:'ShoreBreak.cu',...inputs,records},null,2));
