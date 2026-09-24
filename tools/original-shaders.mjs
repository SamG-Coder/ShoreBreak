// Read private upstream shader constants without modifying the reference implementation.
import {readFile} from 'node:fs/promises';
export async function originalShaders(relative,names){
 const url=new URL(relative,new URL('../',import.meta.url));
 const source=(await readFile(url,'utf8')).replace(/from\s+(['"])([^'"]+)\1/g,(_,q,s)=>`from '${s.startsWith('.')?new URL(s,url).href:import.meta.resolve(s)}'`);
 return import('data:text/javascript;base64,'+Buffer.from(source+'\nexport {'+names.join(',')+'};').toString('base64'));
}
