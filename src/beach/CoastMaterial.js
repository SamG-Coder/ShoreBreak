import * as THREE from 'three';
import { glslDefines } from '../config.js';
import { NOISE, SKY } from '../glsl/common.js';
import { HAZE_GLSL } from './haze.js';

export const COAST_SHADOW = /* glsl */ `
uniform sampler2D uCoastShadow;
uniform mat4 uCoastShadowMatrix;
float coastShadow(vec3 P, vec3 N) {
  vec4 q = uCoastShadowMatrix * vec4(P + N * 0.07, 1.0);
  vec3 p = q.xyz / q.w;
  if (min(min(p.x,p.y),p.z) < 0.0 || max(max(p.x,p.y),p.z) > 1.0) return 1.0;
  float s = 0.0;
  for (int i=0; i<4; i++) {
    vec2 o = vec2(float(i&1)-0.5,float(i>>1)-0.5) / vec2(2048.0,1024.0);
    s += step(p.z - 0.00035, textureLod(uCoastShadow,p.xy+o, 0.0).r);
  }
  return mix(1.0, s*0.25, smoothstep(0.0,0.055,min(min(p.x,1.0-p.x),min(p.y,1.0-p.y))));
}
`;
export function coastUniforms(shared) {
  if (!shared.uCoastShadow) {
    const tex = new THREE.DataTexture(new Uint8Array([255,255,255,255]),1,1);
    tex.needsUpdate=true;
    shared.uCoastShadow={value:tex};
    shared.uCoastShadowMatrix={value:new THREE.Matrix4().makeTranslation(10,10,10)};
  }
  return {uSunDir:shared.uSunDir,uSunColor:shared.uSunColor,uSkyAmb:shared.uSkyAmb,uSkyLut:shared.uSkyLut,uTime:shared.uTime,uCoastShadow:shared.uCoastShadow,uCoastShadowMatrix:shared.uCoastShadowMatrix};
}
export const COAST_PRE = glslDefines()+NOISE+SKY+HAZE_GLSL+COAST_SHADOW;
export function coastMaterial(shared) {
  return new THREE.ShaderMaterial({
    uniforms:coastUniforms(shared), vertexColors:true,
    vertexShader:COAST_PRE+/* glsl */`
      in float aKind;
      out vec3 vW; out vec3 vN; out vec3 vColor; out vec2 vUv; flat out float vKind;
      void main(){
        vec4 p=vec4(position,1.0);vec3 n=normal;
        #ifdef USE_INSTANCING
          p=instanceMatrix*p;n=mat3(instanceMatrix)*n;
        #endif
        p=modelMatrix*p;vW=p.xyz;vN=normalize(mat3(modelMatrix)*n);
        vColor=color;
        #ifdef USE_INSTANCING_COLOR
          vColor*=instanceColor;
        #endif
        vUv=uv;vKind=aKind;
        gl_Position=projectionMatrix*viewMatrix*p;
      }`,
    fragmentShader:COAST_PRE+/* glsl */`
      in vec3 vW;in vec3 vN;in vec3 vColor;in vec2 vUv;flat in float vKind;
      void main(){
        vec3 N=normalize(vN),V=normalize(cameraPosition-vW),L=uSunDir;
        vec3 alb=vColor;float rough=0.78;
        float grain=vnoise(vW.xy*7.0+vW.z*0.17)-0.5;
        alb*=1.0+grain*0.045;
        if(vKind<0.5){
          // Mineral plaster, broad rain weathering, fine roughness with no pixel sparkle.
          alb*=0.98+0.045*vnoise(vW.xy*vec2(.32,.13));
        } else if(vKind<1.5){
          // Window panes reflect the shared sky with dielectric Fresnel.
          float f=0.045+0.955*pow(1.0-max(dot(N,V),0.0),5.0);
          vec3 R=reflect(-V,N);
          #ifdef OPT_EXPLORE
            vec3 sky=highCloud(R,skyProfile(degrees(asin(clamp(R.y,-1.0,1.0))))*skyAzimuthGain(R));
          #else
            vec3 sky=skyRadiance(R);
          #endif
          // View-dependent room depth and partially drawn linen curtains. The
          // pane's local coordinates keep the detail attached while walking.
          vec2 room=vUv + vec2(V.x,V.y)/max(abs(V.z),.3)*.10;
          vec2 edge=min(room,1.0-room);
          float reveal=smoothstep(0.0,.11,min(edge.x,edge.y));
          float curtainWidth=mix(.08,.36,fract(alb.r*73.1+alb.g*31.7));
          float curtain=1.0-smoothstep(curtainWidth,curtainWidth+.025,min(room.x,1.0-room.x));
          float pleat=1.0+.09*sin(room.x*92.0)*(1.0-smoothstep(.08,.22,fwidth(room.x)*15.0));
          vec3 interior=mix(alb*.38,vec3(.38,.35,.29)*pleat,curtain)*mix(.48,1.0,reveal);
          vec3 col=mix(interior,sky,min(.86,f+.10));
          col=mix(col,landHazeColor(-V),landHaze(length(cameraPosition-vW)));
          gl_FragColor=vec4(col,1.0);return;
        } else if(vKind<2.5){rough=.42;} else if(vKind<3.5){
          // Zinc seams and rounded clay-tile courses.
          float phase=vUv.x*1.7;float keep=1.0-smoothstep(.25,.8,fwidth(phase));
          alb*=1.0+sin(phase*6.28318)*.06*keep;
        } else if(vKind<4.5){
          vec2 q=vW.xz/vec2(1.2,.8);vec2 f=abs(fract(q)-.5);
          float joint=smoothstep(.485-max(fwidth(q.x),fwidth(q.y)),.5,max(f.x,f.y));
          alb*=1.0-.18*joint;
        }
        float sh=coastShadow(vW,N),ndl=max(dot(N,L),0.0);
        float ao=mix(.74,1.0,smoothstep(5.5,7.2,vW.y));
        // Sky above, warm paving/limestone bounce below. Undersides no longer
        // receive the same flat ambient light as an exposed roof.
        float skyVisibility=mix(.58,1.0,smoothstep(-.65,.55,N.y));
        vec3 bounce=vec3(.075,.065,.047)*(1.0-N.y)*.5*ao;
        vec3 col=alb*(uSunColor*(ndl*sh/3.14159265)+skyAmbient(N)*ao*skyVisibility+bounce);
        vec3 H=normalize(V+L);
        col+=uSunColor*.018*pow(max(dot(N,H),0.0),mix(52.0,9.0,rough))*ndl*sh;
        col=mix(col,landHazeColor(-V),landHaze(length(cameraPosition-vW)));
        gl_FragColor=vec4(col,1.0);
      }`,
  });
}

// Architecture and vegetation are static in space. Bake their shadows once at
// startup; the foliage's centimetre-scale breeze does not require a shadow pass.
export function bakeCoastShadows(renderer, objects, shared) {
  const rt=new THREE.WebGLRenderTarget(2048,1024,{depthTexture:new THREE.DepthTexture(2048,1024,THREE.UnsignedIntType),minFilter:THREE.NearestFilter,magFilter:THREE.NearestFilter});
  const scene=new THREE.Scene();
  for(const object of objects) scene.add(object.clone(true));
  const depth=new THREE.MeshDepthMaterial({side:THREE.DoubleSide});scene.overrideMaterial=depth;
  const camera=new THREE.OrthographicCamera(-270,270,110,-110,1,700);
  const target=new THREE.Vector3(0,10,67);
  camera.position.copy(target).addScaledVector(shared.uSunDir.value,300);camera.lookAt(target);camera.updateMatrixWorld();
  const bias=new THREE.Matrix4().set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1);
  shared.uCoastShadowMatrix.value.copy(bias).multiply(camera.projectionMatrix).multiply(camera.matrixWorldInverse);
  const old=renderer.getRenderTarget();renderer.setRenderTarget(rt);renderer.clear();renderer.render(scene,camera);renderer.setRenderTarget(old);
  shared.uCoastShadow.value=rt.depthTexture;depth.dispose();
  return rt;
}
