import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { bayBendJS } from './terrain.js';
import { coastMaterial } from './CoastMaterial.js';

const STREET=5.56;
const hash=n=>{const v=Math.sin(n*127.1+311.7)*43758.5453;return v-Math.floor(v);};
const color=c=>new THREE.Color(c);

// Real silhouette geometry for the walkable beach's foreground. One instanced
// box batch per 84 m district; roofs and domes are merged per district. This
// gives balconies depth while culling entire streets behind the camera.
export class Seafront {
  constructor(shared){
    this.group=new THREE.Group();this.group.name='Promenade des Anglais seafront';
    const material=coastMaterial(shared),districts=new Map();
    const district=x=>{const id=Math.floor(x/84);if(!districts.has(id))districts.set(id,{boxes:[],roofs:[]});return districts.get(id);};
    const box=(x,y,z,w,h,d,c,kind=0,rot=0)=>district(x).boxes.push({x,y,z,w,h,d,c,kind,rot});
    const trim='#ddd4c1',rail='#414c4b',blue='#376b86';
    const colors=['#dfd8c8','#d8c7a8','#e3d9c6','#dbbca8','#cbd1ca','#e0d6bf','#d1c5b3'];
    const meshPart=(x,y,z,g,c,kind=0,rotation=0)=>{
      g.rotateY(rotation);g.translate(x,y,z-bayBendJS(x));
      const count=g.attributes.position.count,cs=new Float32Array(count*3),cc=color(c);
      for(let i=0;i<count;i++)cs.set(cc.toArray(),i*3);
      g.setAttribute('color',new THREE.Float32BufferAttribute(cs,3));
      g.setAttribute('aKind',new THREE.Float32BufferAttribute(new Float32Array(count).fill(kind),1));
      district(x).roofs.push(g);
    };
    for(let slot=-18;slot<=18;slot++){
      const widths=Array.from({length:37},(_,i)=>14+hash(i+51)*17);
      const total=widths.reduce((a,b)=>a+b,0),index=slot+18;
      const width=widths[index]*777/total-.35;
      const x=-378+widths.slice(0,index).reduce((a,b)=>a+b,0)*777/total+(width+.35)/2;
      if((x+width/2>62&&x-width/2<126)||(x+width/2>176&&x-width/2<240))continue;
      const r=hash(slot+8),style=Math.floor(r*4),classic=style<3;
      const floors=classic?5+Math.floor(hash(slot+21)*3):6+Math.floor(hash(slot+21)*4);
      const height=4.4+floors*3.05,front=79+hash(slot+17)*2.4,depth=17+hash(slot+19)*4;
      const tint=colors[Math.floor(hash(slot+11)*colors.length)];
      box(x,STREET+height/2,front+depth/2,width,height,depth,tint);
      // Rusticated ground-floor plinth; pronounced roof cornice catches the sun.
      box(x,STREET+2.15,front-.11,width,4.3,.25,classic?'#c4bfae':'#c6c8bf');
      for(const [dy,extension,thickness] of [[-.05,.6,.24],[-.45,.38,.22],[-.8,.18,.24]])
        box(x,STREET+height+dy,front+depth/2-.15,width+extension,thickness,depth+.6+extension,trim);
      if(classic){
        for(let f=0;f<=floors;f++)box(x,STREET+4.4+f*3.05,front-.20,width+.1,.12,.4,trim);
        // Zinc mansard slopes are actual four-sided roofs, with a flat ridge.
        const shape=new THREE.CylinderGeometry(.70,1,2.65,4,1,false);
        shape.rotateY(Math.PI/4);shape.scale(width/Math.SQRT2,1,depth/Math.SQRT2);
        meshPart(x,STREET+height+1.2,front+depth/2,shape,slot%3===0?'#a68576':'#758082',3);
      }else{
        box(x,STREET+height+1.35,front+depth/2+1,width-2,2.5,depth-3,'#c2c5bd');
        box(x+4,STREET+height+3.1,front+depth*.66,3.2,1.2,3.5,'#8b9592');
      }
      const bays=Math.max(3,Math.round(width/(style===1?3.9:4.5))),bw=width/bays,ww=classic?1.10+hash(slot+14)*.36:2.22;
      for(let f=0;f<floors;f++)for(let j=0;j<bays;j++){
        const wx=x-width/2+bw*(j+.5),wy=STREET+4.4+f*3.05+1.62;
        const shutter=classic&&hash(slot*70+f*5+j)>.69;
        const winColor=shutter?['#66786f','#7d8874','#799097','#a49b86'][style]:hash(slot*93+f*8+j)>.7?'#aca18b':'#293940';
        // Frame projects around a recessed dark reveal and reflective glass.
        box(wx,wy,front-.19,ww+.28,2.22,.30,'#8f9186');
        box(wx,wy,front-.37,ww,2.0,.08,winColor,shutter?0:1);
        for(const sx of [-1,1])box(wx+sx*(ww/2+.10),wy,front-.43,.13,2.27,.23,trim);
        box(wx,wy+1.09,front-.45,ww+.37,.16,.28,trim);
        box(wx,wy-1.09,front-.50,ww+.40,.14,.43,trim);
        box(wx,wy,front-.44,.045,2,.1,'#bcc1b8');
        box(wx,wy+.15,front-.44,ww,.045,.1,'#bcc1b8');
        if(classic&&!shutter){
          for(const sx of [-1,1]){
            box(wx+sx*(ww/2+.48),wy,front-.22,.54,1.98,.12,'#66786f');
            // Four relief louvres read as shutter depth at a grazing angle.
            for(let l=0;l<4;l++)box(wx+sx*(ww/2+.48),wy-.7+l*.45,front-.30,.50,.08,.1,'#78877b');
          }
        }
        if(f>0&&(!classic||f===1||f===floors-1||((j+slot)%3===0))){
          const widthB=classic?ww+1.1:bw-.10,reach=classic?.90:1.25,y=wy-1.13;
          box(wx,y,front-reach/2,widthB,.16,reach,trim);
          box(wx,y+1,front-reach,widthB,.048,.046,rail,2);
          box(wx,y+.14,front-reach,widthB,.035,.040,rail,2);
          for(let bar=0;bar<=6;bar++)box(wx-widthB/2+widthB*bar/6,y+.57,front-reach,.027,.86,.035,rail,2);
          for(const side of [-1,1])box(wx+side*widthB/2,y+1,front-reach/2,.038,.048,reach,rail,2);
        }
      }
      // Glazed shopfronts and occasional faded canvas awnings.
      const shops=Math.max(2,Math.round(width/5)),shopW=width/shops;
      for(let j=0;j<shops;j++){
        const sx=x-width/2+(j+.5)*shopW;box(sx,STREET+1.8,front-.21,shopW-.8,3.2,.14,'#34464a',1);
        box(sx,STREET+3.65,front-.45,4.15,.28,.40,'#7f8274');
        if(hash(slot*10+j)>.48)box(sx,STREET+3.55,front-1.05,4.1,.16,1.5,j%2?'#cfbfaa':'#788b83');
      }
      // Alternating pediments, rustication, drainpipes and recessed shutters distinguish façades.
      if(style===1)for(let j=0;j<bays;j++){
        const px=x-width/2+(j+.5)*bw;
        const ped=new THREE.CylinderGeometry(1,1,.16,3);ped.rotateX(Math.PI/2);ped.rotateZ(Math.PI);ped.scale(.95,.36,1);
        meshPart(px,STREET+7.8,front-.55,ped,trim);
      }
      if(style===2)for(let f=0;f<floors;f++)box(x,STREET+5.5+f*3.05,front-.25,width,.065,.11,'#b0aaa0');
      box(x+width/2-.22,STREET+height/2,front-.46,.085,height,.085,'#8b8c7c',2);
      if(classic)for(const sx of [-1,1])box(x+sx*(width/2-.35),STREET+height/2,front-.18,.46,height,.4,trim);
    }
    // Reference-informed silhouettes, compressed into the playable bay: a
    // Belle Époque corner rotunda and the Palais' open Art Deco colonnade.
    // These are interpretations, not a surveyed digital reconstruction.
    const hx=208,front=78,height=26;
    box(hx,STREET+height/2,front+12,59,height,24,'#e6ddd1');
    for(const y of [4.3,10.5,20,25.5,26])box(hx,STREET+y,front+11.5,60,.24,25,'#e8dfce');
    const roof=new THREE.CylinderGeometry(.76,1,3.4,4);roof.rotateY(Math.PI/4);roof.scale(59/Math.SQRT2,1,24/Math.SQRT2);
    meshPart(hx,STREET+27.6,front+12,roof,'#a17976',3);
    const cx=hx+23,cz=front+3;
    meshPart(cx,STREET+25,cz,new THREE.CylinderGeometry(4.35,4.5,5,32),'#e2d9c8');
    const dome=new THREE.SphereGeometry(4.65,32,16,0,Math.PI*2,0,Math.PI/2);dome.scale(1,1.14,1);
    meshPart(cx,STREET+27.5,cz,dome,'#bd8a85',3);
    meshPart(cx,STREET+33.6,cz,new THREE.CylinderGeometry(.27,.47,1.6,12),'#6e8076',2);
    box(cx,STREET+35,cz,.065,2,.065,'#616b67',2);
    for(let i=0;i<3;i++)box(cx+.22+i*.25,STREET+35.3,cz,.25,.47,.025,['#405d85','#e4e4dc','#b65a58'][i]);
    for(let f=0;f<6;f++)for(let j=0;j<16;j++){
      const x=hx-27+j*3.6,y=STREET+3.7+f*3.7;
      box(x,y,front-.16,1.28,2.22,.13,'#34484b',1);
      box(x,y+1.19,front-.30,1.80,.21,.45,'#ece3d4');
      box(x,y-1.17,front-.50,1.90,.17,.95,'#ddd4c4');
      if(f>0){
        box(x,y-.45,front-1,1.8,.055,.05,rail,2);
        for(let k=0;k<5;k++)box(x-.75+k*.375,y-.83,front-1,.031,.72,.03,rail,2);
      }
      if(j%3===0)box(x+1.68,y,front-.27,.28,3.65,.42,'#ede4d2');
    }
    // Palais de la Méditerranée: three monumental openings, paired piers,
    // deep shaded terrace, stepped entablature and a recessed upper hotel.
    const ax=94,az=79;
    box(ax,STREET+16.5,az+16,61,33,19,'#d8d5c9');
    box(ax,STREET+5,az+3,61,10,9,'#e2dece');
    box(ax,STREET+15.5,az+5.5,59,12,1,'#555e58');
    for(const dx of [-29,-20,-10,0,10,20,29]){
      box(ax+dx,STREET+16,az,1.55,13,1.9,'#e7e0cf');
      box(ax+dx,STREET+22.7,az-.2,2.1,.5,2.5,'#ede5d3');
    }
    for(const [y,w,h] of [[10.2,62,.65],[22.3,63,.7],[23.3,64,.9],[24.2,62,.55]])box(ax,STREET+y,az,w,h,2.6,'#e7e0cf');
    for(let j=0;j<15;j++){
      const x=ax-27+j*3.9;
      box(x,STREET+4.4,az-1.6,2.7,5.8,.14,'#344548',1);
      for(let f=0;f<3;f++)box(x,STREET+25.5+f*2.5,az+6.3,1.4,1.8,.12,'#485454',1);
      box(x,STREET+8.3,az-1.5,3.5,.32,1.5,'#cbb997');
    }
    // Broad paved promenade, cycle strip and road. Segments follow bay curvature.
    for(let x=-441;x<442;x+=21){
      box(x,STREET-.15,49.1,21,.3,18.2,'#bfbcb1',4);
      box(x,STREET-.14,60.5,21,.3,4.6,'#718a8c');
      box(x,STREET-.18,70.5,21,.3,15.1,'#727777');
      for(let lane=0;lane<3;lane++)box(x-7+lane*7,STREET-.018,69.8,3,.012,.12,'#d4d0b6');
      // Nice's blue chairs: individual slats, armrests and thin steel frames.
      for(const dx of [-3,0,3]){
        const cx=x+dx,cz=42.4;
        for(let slat=0;slat<5;slat++){
          box(cx,STREET+.48,cz-.22+slat*.105,.62,.034,.085,blue,2);
          box(cx,STREET+.68+slat*.077,cz+.27,.62,.052,.045,blue,2);
        }
        for(const sx of [-1,1]){
          box(cx+sx*.27,STREET+.25,cz-.17,.025,.5,.035,blue,2);
          box(cx+sx*.27,STREET+.49,cz+.23,.025,.98,.035,blue,2);
          box(cx+sx*.32,STREET+.71,cz,.026,.025,.61,blue,2);
        }
      }
    }
    const dummy=new THREE.Object3D();
    for(const [id,d] of districts){
      const g=new THREE.BoxGeometry(1,1,1);g.setAttribute('color',new THREE.Float32BufferAttribute(new Float32Array(g.attributes.position.count*3).fill(1),3));
      g.setAttribute('aKind',new THREE.InstancedBufferAttribute(Float32Array.from(d.boxes,b=>b.kind),1));
      const mesh=new THREE.InstancedMesh(g,material,d.boxes.length);mesh.name=`Seafront district ${id}`;
      d.boxes.forEach((b,i)=>{dummy.position.set(b.x,b.y,b.z-bayBendJS(b.x));dummy.rotation.set(0,b.rot+Math.atan(Math.sign(b.x)*Math.max(0,Math.abs(b.x)-150)/10000),0);dummy.scale.set(b.w,b.h,b.d);dummy.updateMatrix();mesh.setMatrixAt(i,dummy.matrix);mesh.setColorAt(i,color(b.c));});
      mesh.instanceMatrix.needsUpdate=true;mesh.instanceColor.needsUpdate=true;mesh.computeBoundingSphere();this.group.add(mesh);
      if(d.roofs.length){const merged=mergeGeometries(d.roofs);d.roofs.forEach(g=>g.dispose());const roofs=new THREE.Mesh(merged,material);roofs.name=`Seafront roofs ${id}`;this.group.add(roofs);}
    }
    this.group.userData.instances=[...districts.values()].reduce((n,d)=>n+d.boxes.length,0);
  }
}
