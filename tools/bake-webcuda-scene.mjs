// Asset conversion only: preserve upstream scenery in a static ray-query format.
// Three.js is an offline build dependency; it is never imported by the browser entry.
import * as THREE from 'three';
import { Seafront } from '../src/beach/Seafront.js';
import { readFile,writeFile,mkdir } from 'node:fs/promises';
const triangles=[],nodes=[];
const vec=new THREE.Vector3(),matrix=new THREE.Matrix4(),color=new THREE.Color();
function addMesh(geometry,transform,tint=new THREE.Color(1,1,1),leaf=false){
 const position=geometry.getAttribute('position'),colors=geometry.getAttribute('color'),index=geometry.index;
 for(let i=0;i<index.count;i+=3){
  const vertices=[],c=[0,0,0];
  for(let k=0;k<3;k++){
   const j=index.getX(i+k);vec.fromBufferAttribute(position,j).applyMatrix4(transform);vertices.push(vec.x,vec.y,vec.z);
   if(colors){c[0]+=colors.getX(j)*tint.r/3;c[1]+=colors.getY(j)*tint.g/3;c[2]+=colors.getZ(j)*tint.b/3;}else{c[0]+=tint.r/3;c[1]+=tint.g/3;c[2]+=tint.b/3;}
  }
  triangles.push({v:vertices,c,leaf,center:[0,1,2].map(a=>(vertices[a]+vertices[3+a]+vertices[6+a])/3)});
 }
}
function tree(items){
 const index=nodes.length,node={lo:[Infinity,Infinity,Infinity],hi:[-Infinity,-Infinity,-Infinity],start:0,count:0,escape:0};nodes.push(node);
 for(const t of items)for(let a=0;a<3;a++)for(let k=0;k<3;k++){node.lo[a]=Math.min(node.lo[a],t.v[k*3+a]);node.hi[a]=Math.max(node.hi[a],t.v[k*3+a]);}
 if(items.length<=6){node.start=ordered.length;node.count=items.length;ordered.push(...items);}
 else{const spans=node.hi.map((v,i)=>v-node.lo[i]),axis=spans.indexOf(Math.max(...spans));items.sort((a,b)=>a.center[axis]-b.center[axis]);const mid=items.length>>1;tree(items.slice(0,mid));tree(items.slice(mid));}
 node.escape=nodes.length;return index;
}
const ordered=[];
const city=new Seafront({});city.group.updateMatrixWorld(true);
city.group.traverse(mesh=>{
 if(!mesh.isMesh)return;
 if(mesh.isInstancedMesh)for(let i=0;i<mesh.count;i++){mesh.getMatrixAt(i,matrix);matrix.premultiply(mesh.matrixWorld);mesh.getColorAt(i,color);addMesh(mesh.geometry,matrix,color);}
 else addMesh(mesh.geometry,mesh.matrixWorld);
});
// Original promenade cross-section. Keep a narrow face and coping in the BVH.
for(const [y,z,h,d] of [[3.45,40.15,3.7,.30],[5.43,40.0,.26,.55],[5.98,40.04,.84,.27]]){
 const g=new THREE.BoxGeometry(8000,h,d);addMesh(g,new THREE.Matrix4().makeTranslation(0,y,z),new THREE.Color(.42,.41,.37),2);
}
const cityRoot=tree(triangles.splice(0));console.log('City converted:',ordered.length,'triangles');
const manifest=JSON.parse(await readFile(new URL('../public/assets/palms-r6/manifest.json',import.meta.url),'utf8'));
const palmRoots=[];
for(const seed of [44,288]){
 for(const entry of manifest.filter(e=>e.seed===seed&&e.tier===1)){
  const bytes=await readFile(new URL(`../public/assets/palms-r6/${entry.file}`,import.meta.url));
  const buffer=bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),g=new THREE.BufferGeometry();
  const types={Float32Array,Uint32Array,Uint16Array,Int16Array,Uint8Array};
  for(const [name,a] of Object.entries(entry.attributes))g.setAttribute(name,new THREE.BufferAttribute(new types[a.type](buffer,a.offset,a.length),a.itemSize,a.normalized));
  const a=entry.index;g.setIndex(new THREE.BufferAttribute(new types[a.type](buffer,a.offset,a.length),1));
  addMesh(g,new THREE.Matrix4(),new THREE.Color(1,1,1),entry.part==='leaves');
 }
 palmRoots.push(tree(triangles.splice(0)));
}
const triBase=1+nodes.length*3,out=new Float32Array((triBase+ordered.length*4)*4);
out.set([cityRoot,...palmRoots,triBase]);
nodes.forEach((n,i)=>{const b=(1+i*3)*4;out.set([...n.lo,n.escape,...n.hi,n.start,n.count,0,0,0],b);});
ordered.forEach((t,i)=>{const b=(triBase+i*4)*4;out.set([...t.v.slice(0,3),0,...t.v.slice(3,6),0,...t.v.slice(6,9),0,...t.c,Number(t.leaf)],b);});
await mkdir(new URL('../public/assets/webcuda/',import.meta.url),{recursive:true});
await writeFile(new URL('../public/assets/webcuda/scene.bin',import.meta.url),new Uint8Array(out.buffer));
await writeFile(new URL('../public/assets/webcuda/CREDITS.txt',import.meta.url),'Derived from ShoreBreak Seafront.js and the bundled Ashore / Jellys date palms.\nOriginal code and geometry: Christopher Canavan and credited contributors.\nMIT; see LICENSE, THIRD_PARTY_NOTICES.md and public/assets/palms-r6/CREDITS.txt.\nThis file is an offline BVH conversion, not a new third-party asset.\n');
console.log(`${nodes.length} nodes; ${ordered.length} triangles; ${(out.byteLength/1048576).toFixed(1)} MiB`);
