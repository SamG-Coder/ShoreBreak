import * as THREE from 'three';
import { bedProfileJS } from '../config.js';

// Partly buried, wave-worn limestone-like coastal rocks. One deterministic surface
// supplies the visible mesh, the hydraulic bed and walking height queries.
const clusters=[[-35.2,-.65,.88],[-32.1,-1.25,.70],[-29.8,.42,.58],[-40.7,-.9,.74],[-21.8,-.55,.81],[-14.3,.15,.54],[2.6,-.9,.78],[13.4,-.25,.72],[26.7,-1.12,.92],[41.5,.35,.66],[56.2,-.65,.85],[-56.8,-.38,.71]];
export const ROCKS=[];
export const ROCK_SEGMENTS=48, ROCK_RINGS=14;
for(let i=0;i<clusters.length;i++){
  const [x,z,s]=clusters[i];
  for(let j=0;j<(i%3===0?3:2);j++){
    const seed=i*2.371+j*5.17, scale=s*(j===0?1:.37+.11*Math.sin(seed));
    const X=x+(j?Math.cos(seed)*s*1.32:0),Z=z+(j?Math.sin(seed)*s*.82:0);
    ROCKS.push({x:X,z:Z,rx:scale,rz:scale*(.59+.12*Math.sin(seed+3)),h:scale*(.53+.07*Math.cos(seed)),a:seed,seed,base:bedProfileJS(Z)-scale*.18});
  }
}
function local(r,x,z){const c=Math.cos(r.a),s=Math.sin(r.a),dx=x-r.x,dz=z-r.z;return [(c*dx+s*dz)/r.rx,(-s*dx+c*dz)/r.rz];}
export function rockRadius(r,a){return 1+.055*Math.sin(3*a+r.seed)+.035*Math.cos(5*a-r.seed*2)+.025*Math.sin(7*a+1.7);}
export function rockHeight(r,x,z){
  const [u,v]=local(r,x,z),rho=Math.hypot(u,v)/rockRadius(r,Math.atan2(v,u));
  if(rho>=1)return -100;
  const cap=Math.pow(1-rho*rho,.47);
  const shape=1+.07*Math.sin(u*4.1+r.seed)*Math.cos(v*3.8-1.3)+.025*Math.sin(u*12+v*7+r.seed);
  // Follow the local sloping seabed across the footprint. A flat base left
  // the seaward rim hanging above the sand when viewed from underwater.
  const substrate=bedProfileJS(z)-bedProfileJS(r.z);
  return r.base+substrate+r.h*cap*shape;
}
// Identical samples for the visible rock and its persistent wetness atlas.
export function rockVertex(r,i,j){
  const a=(i%ROCK_SEGMENTS)/ROCK_SEGMENTS*Math.PI*2,rho=j/ROCK_RINGS*.9998;
  const rr=rockRadius(r,a),u=Math.cos(a)*rho*rr,v=Math.sin(a)*rho*rr,c=Math.cos(r.a),s=Math.sin(r.a);
  const x=r.x+c*u*r.rx-s*v*r.rz,z=r.z+s*u*r.rx+c*v*r.rz;
  return [x,rockHeight(r,x,z),z];
}
export function rockTopAt(x,z){let y=-100;for(const r of ROCKS)if(Math.abs(x-r.x)<r.rx+r.rz&&Math.abs(z-r.z)<r.rx+r.rz)y=Math.max(y,rockHeight(r,x,z));return y;}
export function initCoastalBed(shared){
  if(shared.uRockBed)return;
  const width=2048,height=128,domain=[-64,-3,128,6],data=new Float32Array(width*height);
  // Outside a rock use the local sand height: filtering never mixes a sentinel
  // into a rock edge and creates a trench. The solver samples this only at bed setup.
  for(let j=0;j<height;j++)for(let i=0;i<width;i++){
    const x=domain[0]+(i+.5)*domain[2]/width,z=domain[1]+(j+.5)*domain[3]/height;
    data[j*width+i]=Math.max(bedProfileJS(z)-.03,rockTopAt(x,z));
  }
  const tex=new THREE.DataTexture(data,width,height,THREE.RedFormat,THREE.FloatType);
  tex.minFilter=tex.magFilter=THREE.LinearFilter;tex.generateMipmaps=false;tex.needsUpdate=true;
  shared.uRockBed={value:tex};shared.uRockDomain={value:new THREE.Vector4(...domain)};
}
