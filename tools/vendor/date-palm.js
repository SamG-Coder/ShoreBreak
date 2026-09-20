// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christopher Canavan
// Procedural geometry shared by the author's Ashore, Jellys and ShoreBreak projects.
import * as THREE from 'three';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';
import {leafRibbon} from './ashore-leaf.js';
function random(seed){return()=>{seed=(seed*1664525+1013904223)>>>0;return seed/4294967296;};}

// Mature date palms: nearly upright, stout trunks wearing retained leaf bases,
// and broad, layered pinnate crowns. Every leaf and boot has real geometry.
function join(parts){const g=mergeGeometries(parts);for(const p of parts)p.dispose();g.computeBoundingSphere();return g;}
function tinted(g,color){const c=new Float32Array(g.attributes.position.count*3);for(let i=0;i<c.length;i+=3)c.set(color.toArray(),i);g.setAttribute('color',new THREE.BufferAttribute(c,3));return g;}
function rachis(points,radius,color,far){
 const curve=new THREE.CatmullRomCurve3(points),g=new THREE.TubeGeometry(curve,far?8:20,radius,far?3:5,false),p=g.attributes.position;
 const sides=far?3:5,segments=far?8:20;
 for(let i=0;i<p.count;i++){const t=Math.floor(i/(sides+1))/segments,c=curve.getPointAt(t),scale=1-t*.94;p.setXYZ(i,c.x+(p.getX(i)-c.x)*scale,c.y+(p.getY(i)-c.y)*scale,c.z+(p.getZ(i)-c.z)*scale);}
 g.computeVertexNormals();tinted(g,color);
 g.setAttribute('windRoot',new THREE.Float32BufferAttribute(new Float32Array(p.count*3),3));g.setAttribute('windWeight',new THREE.Float32BufferAttribute(new Float32Array(p.count),1));return g;
}
export function palmGeometry(seed,tier=0){
 const far=true, spacing=[3,6,12][tier], widthGain=[1.75,3.5,6][tier];
 const rng=random(seed),height=seed===44?15.6:11.9,lean=seed===44?.30:-.19,center=t=>new THREE.Vector3(lean*t*t,height*t,.14*Math.sin(t*Math.PI));
 const radius=t=>1.32*(.50+.19*Math.exp(-t*13)-.075*t+.014*Math.sin(t*19)+.008*Math.sin(t*47));
 const path=Array.from({length:43},(_,i)=>center(i/42)),curve=new THREE.CatmullRomCurve3(path),core=new THREE.TubeGeometry(curve,far?21:42,1,far?10:16,false),p=core.attributes.position,uv=core.attributes.uv,segments=far?21:42,sides=far?10:16;
 for(let i=0;i<p.count;i++){const t=Math.floor(i/(sides+1))/segments,c=curve.getPointAt(t),r=radius(t);p.setXYZ(i,c.x+(p.getX(i)-c.x)*r,c.y+(p.getY(i)-c.y)*r,c.z+(p.getZ(i)-c.z)*r);uv.setXY(i,uv.getY(i)*2.8,t*height/1.25);}
 core.computeVertexNormals();tinted(core,new THREE.Color(.56,.47,.35));
 const bootP=[],bootUV=[],bootC=[],bootI=[],rows=Math.ceil(height/.33),around=10;
 const fiberP=[],fiberUV=[],fiberC=[],fiberI=[],fiberRng=random(seed+41821);
 for(let row=0;row<rows;row++)for(let col=0;col<around;col++){
  const t=(row+.25+(rng()-.5)*.44+.12*Math.sin(col*1.7+row*.41))/rows,c=center(t),angle=(col+(row%2)*.5)/around*Math.PI*2+.055*Math.sin(row*.83)+.030*(rng()-.5),r=radius(t);
  const width=(Math.PI*2*r/around)*(.91+rng()*.18),h=.37+rng()*.14,projection=.085+rng()*.085;
  // Curled upper shoulders overlap the next row; a chipped lower point
  // projects over the dark fibrous base of the preceding scar.
  const chip=rng(),curl=rng(),skew=(rng()-.5)*.12;
  const shape=[[-.50,.19,.25],[-.38,.46+curl*.13,.60],[.16+skew,.55,.78],[.49,.17,.30],[.31,-.25-chip*.12,.75],[skew,-.48-chip*.18,.86+curl*.19],[-.30,-.27,.64],[skew,.06,.82]];
  const base=bootP.length/3,tint=.72+rng()*.32,ux=rng()*1.8,uy=rng()*4;
  for(const [x,y,relief] of shape){const a=angle+x*width/r,rr=r+projection*relief;bootP.push(c.x+Math.sin(a)*rr,c.y+y*h,c.z+Math.cos(a)*rr);bootUV.push(ux+x*.26,uy+y*.16);const shade=tint*(y<-.2?.56:1);bootC.push(shade,shade*.88,shade*.72);}
  for(let i=0;i<7;i++)bootI.push(base+7,base+(i+1)%7,base+i);
  if(!far){
   // A recessed perimeter gives each retained leaf base thickness. The
   // sloped sides catch sunlight and cast their own fine overlap shadows.
   for(const [x,y,relief] of shape.slice(0,7)){
    const a=angle+x*width*1.06/r,rr=r+projection*relief*.16;
    bootP.push(c.x+Math.sin(a)*rr,c.y+y*h*1.04,c.z+Math.cos(a)*rr);
    bootUV.push(ux+x*.26,uy+y*.16);bootC.push(tint*.49,tint*.41,tint*.31);
   }
   for(let i=0;i<7;i++){const n=(i+1)%7;bootI.push(base+i,base+n,base+8+n,base+i,base+8+n,base+8+i);}
   if((row+col)%2===0){
    const attach=base+5,x=bootP[attach*3],y=bootP[attach*3+1],z=bootP[attach*3+2];
    const reach=.055+fiberRng()*.14,width=.003+fiberRng()*.004,start=fiberP.length/3,twist=(fiberRng()-.5)*.6;
    for(let ring=0;ring<=3;ring++)for(let side=0;side<3;side++){
     const u=ring/3,a=side/3*Math.PI*2,rad=width*(1-u*.96),curl=u*u*.035;
     fiberP.push(x+Math.sin(angle)*curl+Math.cos(angle)*Math.sin(u*2.1)*twist*.055+Math.cos(a)*rad,y-reach*u,z+Math.cos(angle)*curl+Math.sin(a)*rad);
     fiberUV.push(ux+side*.01,uy+u*.08);fiberC.push(tint*.69,tint*.56,tint*.39);
     if(ring<3){const n=start+ring*3+side,next=start+ring*3+(side+1)%3;fiberI.push(n,next,n+3,next,next+3,n+3);}
    }
   }
  }
 }
 const boots=new THREE.BufferGeometry();boots.setAttribute('position',new THREE.Float32BufferAttribute(bootP,3));boots.setAttribute('uv',new THREE.Float32BufferAttribute(bootUV,2));boots.setAttribute('color',new THREE.Float32BufferAttribute(bootC,3));boots.setIndex(bootI);boots.computeVertexNormals();
 const trunkParts=[core,boots];
 if(fiberP.length){const fibers=new THREE.BufferGeometry();fibers.setAttribute('position',new THREE.Float32BufferAttribute(fiberP,3));fibers.setAttribute('uv',new THREE.Float32BufferAttribute(fiberUV,2));fibers.setAttribute('color',new THREE.Float32BufferAttribute(fiberC,3));fibers.setIndex(fiberI);fibers.computeVertexNormals();trunkParts.push(fibers);}
 const trunk=join(trunkParts);trunk.userData.centerline=path;trunk.userData.radius=1.06;trunk.userData.bootCount=rows*around;
 const top=center(1),leaves=[],frondProfiles=[],frondCount=tier===2?64:(seed===44?100:96),leafletPairs=102,crownRng=random(seed+93017);
 for(let f=0;f<frondCount;f++){
  const age=f/(frondCount-1),a=f*2.399963+(crownRng()-.5)*.22,dx=Math.cos(a),dz=Math.sin(a);
  const skirt=THREE.MathUtils.smoothstep(age,.78,1),length=5.25+.90*Math.sin(age*Math.PI)+crownRng()*.42+skirt*.25;
  const base=top.clone().add(new THREE.Vector3(dx*.20,-age*.88,dz*.20));
  // The supplied photo has a tall spear center, an open shoulder and long
  // hanging outer leaves. Integrating the rachis tangent makes each leaf an
  // actual curved frond, with its tip turning down under its own weight.
  const inclination=.035+age*1.80,curvature=.16+age*.88,curl=(crownRng()-.5)*.44;
  const radial=t=>length*(Math.cos(inclination)-Math.cos(inclination+curvature*t))/curvature;
  const at=t=>base.clone().add(new THREE.Vector3(dx*radial(t)-dz*curl*t*t,length*(Math.sin(inclination+curvature*t)-Math.sin(inclination))/curvature,dz*radial(t)+dx*curl*t*t));
  const reach=radial(1),start=leaves.length,spine=Array.from({length:18},(_,i)=>at(i/17));
  const color=new THREE.Color(age<.20?'#586f3d':age>.82?'#74834f':'#788748');leaves.push(rachis(spine,.054,color,far));
  frondProfiles.push({age,base:base.toArray(),tip:at(1).toArray(),length});
  for(let j=0;j<leafletPairs;j++)for(const side of [-1,1]){
   const t=.27+(j+(side>0?.40:0)+(crownRng()-.5)*.25)/leafletPairs*.72,root=at(t),envelope=Math.pow(Math.max(0,Math.sin((t-.25)/.75*Math.PI)),.65);
   const span=(.14+1.07*envelope)*(.88+crownRng()*.22)*(1-skirt*.22),sweep=.36+crownRng()*.24;
   const insertion=.15*Math.sin(j*.91+f*.4)+.18-age*.12,theta=inclination+curvature*t;
   const along=new THREE.Vector3(dx*Math.sin(theta),Math.cos(theta),dz*Math.sin(theta)),upper=new THREE.Vector3(-dx*Math.cos(theta),Math.sin(theta),-dz*Math.cos(theta));
   const out=new THREE.Vector3(-dz*side,0,dx*side).addScaledVector(along,sweep).addScaledVector(upper,insertion).normalize();
   const halfWidth=(.012+.023*envelope)*(.91+crownRng()*.15),twist=(crownRng()-.5)*.10,ps=[],ws=[];
   for(let k=0;k<=4;k++){const q=k/4,v=root.clone().addScaledVector(out,q*span);v.y+=Math.sin(q*Math.PI)*(.065+.050*age)-q*q*(.065+age*.26);v.addScaledVector(new THREE.Vector3(dx,0,dz),twist*q*q);ps.push(v);ws.push(k===0||k===4?0:halfWidth*Math.sin(q*Math.PI)**.72);}
   const leafColor=new THREE.Color().setHSL(.205+crownRng()*.024,.40+crownRng()*.16,.225+crownRng()*.065+age*.015,THREE.SRGBColorSpace);if(age>.80)leafColor.lerp(new THREE.Color('#819260'),.18+crownRng()*.22);
   if(j%spacing===0)leaves.push(far?leafRibbon([ps[0],ps[2],ps[4]],[0,halfWidth*widthGain,0],leafColor):leafRibbon(ps,ws,leafColor));
  }
  // Phoenix petioles are bare at the sheath, then armed with short basal
  // spines before the first long pinnae. They are not feathery to the trunk.
  for(let k=0;k<7;k++)for(const side of [-1,1]){
   const t=.12+k*.019,root=at(t),span=.065+k*.022,out=new THREE.Vector3(-dz*side+dx*.7,.1,dx*side+dz*.7).normalize();
   if(!far||k%3===0)leaves.push(leafRibbon([root,root.clone().addScaledVector(out,span*.55),root.clone().addScaledVector(out,span)],[0,.0035,0],color));
  }
  for(let part=start;part<leaves.length;part++){
   const g=leaves[part],pos=g.attributes.position,anchors=g.attributes.windRoot,data=new Float32Array(pos.count*4),probe=new THREE.Vector3();
   for(let i=0;i<pos.count;i++){probe.fromBufferAttribute(part===start?pos:anchors,i);const t=THREE.MathUtils.clamp(((probe.x-base.x)*dx+(probe.z-base.z)*dz)/reach,0,1);data.set([base.x,base.y,base.z,t],i*4);}
   g.setAttribute('frondContact',new THREE.BufferAttribute(data,4));
  }
 }
 const foliage=join(leaves);foliage.userData.frondProfiles=frondProfiles;foliage.userData.frondCount=frondCount;foliage.userData.leafletsPerFrond=leafletPairs*2;
 return {trunk,leaves:foliage,top};
}
