// The five-pixel GPU query is also the waterline source. The CPU copy is only
// for movement; optics never wait for a readback to cross the moving surface.
export const WATERLINE = /* glsl */`
uniform sampler2D uSurfaceProbe;
uniform vec2 uProbeOrigin;
uniform float uUnderwaterOn;
float submergedAt(vec3 p){
  if(uUnderwaterOn<.5)return 0.;
  float h=texelFetch(uSurfaceProbe,ivec2(0,0),0).r;
  vec2 slope=vec2(texelFetch(uSurfaceProbe,ivec2(2,0),0).r-texelFetch(uSurfaceProbe,ivec2(1,0),0).r,
                  texelFetch(uSurfaceProbe,ivec2(4,0),0).r-texelFetch(uSurfaceProbe,ivec2(3,0),0).r)/.64;
  return smoothstep(-.018,.018,h+dot(slope,p.xz-uProbeOrigin)-p.y);
}
const vec3 UW_EXTINCTION=vec3(.245,.066,.046);
const vec3 UW_SCATTER=vec3(.022,.155,.185);
vec3 waterTravel(vec3 radiance,float distanceM){
  vec3 tr=exp(-UW_EXTINCTION*max(distanceM,0.));
  return radiance*tr+UW_SCATTER*(1.-tr);
}
`;

// A bounded approximation of refracted sunlight density: derive the light-ray
// Jacobian from the same animated FFT slope field used by the water. Filtering
// grows with depth and the pixel footprint, so light never becomes pixel noise.
// Requires CHOP declarations; no additional texture or simulation pass.
export const CAUSTICS = /* glsl */`
vec2 causticSlope(vec2 p,float lod){
  vec2 a=textureLod(uOceanS, vec3(OCR0*p/uOceanL.x, 0.0), lod).xy*OCR0;
  vec2 b=textureLod(uOceanS, vec3(OCR1*p/uOceanL.y, 1.0), max(lod-1.,0.)).xy*OCR1;
  vec2 c=textureLod(uOceanS, vec3(OCR2*p/uOceanL.z, 2.0), max(lod+log2(uOceanL.x/uOceanL.z),0.)).xy*OCR2;
  return a*uOceanAmp.x*oceanFarGain(p)+b*uOceanAmp.y*.7+c*uOceanAmp.z*.55;
}
float seabedCaustic(vec3 p,float depth){
  if(depth<.025||depth>12.)return 1.;
  float footprint=max(length(dFdx(p.xz)),length(dFdy(p.xz)));
  float e=max(.025+depth*.012,footprint*1.5);
  float lod=log2(e*256./uOceanL.x);
  vec2 q=p.xz-uSunDir.xz*depth/max(uSunDir.y, .4)/1.333;
  q-=causticSlope(q,lod)*depth*.25;
  vec2 dx=(causticSlope(q+vec2(e,0),lod)-causticSlope(q-vec2(e,0),lod))/(2.*e);
  vec2 dz=(causticSlope(q+vec2(0,e),lod)-causticSlope(q-vec2(0,e),lod))/(2.*e);
  float k=min(depth,5.)*.25;
  float jac=(1.-k*dx.x)*(1.-k*dz.y)-k*k*dx.y*dz.x;
  float width=.22+depth*.015+footprint*3.;
  float focus=.97+.72*exp(-jac*jac/(width*width));
  float strength=smoothstep(.025,.16,depth)*exp(-depth*.13)*(1.-smoothstep(.08,.35,footprint));
  return mix(1.,focus,strength*.72);
}
`;

export const UNDERSIDE = /* glsl */`
uniform sampler2D uSandPhoto;
uniform mat4 projectionMatrix;
float undersideFresnel(float ci){
  ci=clamp(ci,.0001,1.);
  float st2=1.333*1.333*(1.-ci*ci);
  float ct=sqrt(max(1.-st2,0.));
  float rs=(1.333*ci-ct)/max(1.333*ci+ct,.0001);
  float rp=(ci-1.333*ct)/max(ci+1.333*ct,.0001);
  return st2>=1.?1.:.5*(rs*rs+rp*rp);
}
vec3 reflectedBed(vec3 p,vec3 ray){
  float travel=clamp((bedHeight(p.xz)-p.y)/min(ray.y,-.04),0.,35.);
  for(int i=0;i<3;i++){
    vec3 q=p+ray*travel;
    travel=clamp((bedHeight(q.xz)-p.y)/min(ray.y,-.04),0.,35.);
  }
  vec3 q=p+ray*travel;
  vec3 photo=texture(uSandPhoto,q.xz*.5).rgb;
  float grain=clamp(dot(photo,vec3(.2126,.7152,.0722))/.165,.35,2.2);
  vec3 alb=vec3(.63,.555,.423)*vec3(.51,.48,.43)*pow(grain,.25);
  float h=max(-q.y,0.);
  vec3 bed=alb*(uSunColor*max(uSunDir.y,0.)/3.14159265*exp(-UW_EXTINCTION*h/max(uSunDir.y,.3))*seabedCaustic(q,h)+uSkyAmb*.75);
  return waterTravel(bed,travel);
}
void main(){
  vec3 P=vWorld,V=normalize(cameraPosition-P),ray=-V;
  float mask=submergedAt(cameraPosition+ray*uNear);
  float waterColumn=P.y-hydraulicBedHeight(P.xz);
  if(mask<.02||waterColumn<.003)discard;
  vec3 macro=normalize(vNrm);
  // A steep face can cross one mesh row. Reconstruct its normal at the
  // rendered HEIGHT, as above water: sampling the analytic profile at the
  // interpolated z instead produces a step and triangular Snell-window tears.
  if(vCrest.x>.001){
    vec3 aux,faceNormal;
    vec4 crest=crestGeometry(vXZ0.xy,uTime,vCrest.w,aux,faceNormal);
    float face=smoothstep(.05,.18,vCrest.w)*smoothstep(.12,.7,crest.y)
      *smoothstep(.04,.20,abs(faceNormal.z))*crest.w;
    macro=normalize(mix(macro,faceNormal,face));
  }
  // Geometry damps horizontal wind chop as the breaker stands up. That
  // stabilizes the lip, but must not erase small ripples from its inner face.
  // Carry the same filtered FFT detail around the curved surface instead of
  // adding it to the enormous height-field slope of a nearly vertical wall.
  float faceRipple=1.-smoothstep(.38,.90,macro.y);
  // Shoaling geometry also damps chop using the still-water bed depth. A
  // thick, moving wave above that bed still needs the small surface ripples.
  float shoaling=smoothstep(-10.,-5.,vXZ0.y)*(1.-smoothstep(-.5,1.,vXZ0.y));
  float rippleKeep=max(faceRipple,shoaling);
  float rippleWeight=mix(vInfo.w,max(vInfo.w,.82*smoothstep(.025,.28,waterColumn)),rippleKeep);
  // Unfold height into the cross-shore coordinate on a steep face. Pure xz
  // projection stretched a few texels into a perfectly clear vertical sheet.
  vec2 rippleUV=vXZ0.xy-vec2(0.,P.y*macro.z*faceRipple);
  vec4 lean=oceanSlopeLEAN(rippleUV,vXZ0.z,rippleWeight,0.);
  // Orthonormal frame from the up vector to macro (stable at vertical faces).
  float invUp=1./max(1.+macro.y,.001);
  vec3 tangentX=vec3(1.-macro.x*macro.x*invUp,-macro.x,-macro.x*macro.z*invUp);
  vec3 tangentZ=vec3(-macro.x*macro.z*invUp,-macro.z,1.-macro.z*macro.z*invUp);
  vec3 flatNormal=normalize(vec3(-macro.x/max(macro.y,.012)+lean.x,-1.,-macro.z/max(macro.y,.012)+lean.y));
  vec3 faceNormal=normalize(-macro+tangentX*lean.x+tangentZ*lean.y);
  vec3 N=normalize(mix(flatNormal,faceNormal,faceRipple));
  if(dot(N,V)<0.)N=-N;
  float ci=clamp(dot(N,V),.0001,1.);
  // Integrate the steep critical-angle change over the pixel / unresolved
  // slope footprint. The sky window stays crisp without a brittle one-pixel
  // reflection switch. Three scalar Fresnel evaluations; one scene lookup.
  float spread=clamp(.65*fwidth(ci)+.2*sqrt(max(lean.z+lean.w,0.)),.001,.035);
  float F=.25*undersideFresnel(ci-spread)+.5*undersideFresnel(ci)+.25*undersideFresnel(ci+spread);
  vec3 transmitted=vec3(0.);
  if(F<.99999){
    // The transmissive subfacets remain just inside the critical angle even
    // when the mean normal is totally reflecting; never refract a zero vector.
    float critical=sqrt(1.-1./(1.333*1.333));
    float ct=max(ci,critical+max(.0001,spread*.3));
    vec3 tangent=N-V*ci;
    tangent/=max(length(tangent),.00001);
    vec3 Nt=ct*V+tangent*sqrt(max(1.-ct*ct,0.));
    vec3 d=normalize(refract(ray,Nt,1.333));
    float el=degrees(asin(clamp(d.y,-1.,1.)));
    float gamma=acos(clamp(dot(d,uSunDir),-1.,1.));
    transmitted=highCloud(d,skyProfile(el)*skyAzimuthGain(d))+skyAureole(gamma);
    float disc=1.-smoothstep(.00465-max(fwidth(gamma),.001),.00465+max(fwidth(gamma),.001),gamma);
    transmitted+=uSunColor*180.*disc;
    // Follow the refracted ray to the coast instead of sampling an arbitrary
    // point a few metres away. That short lookup stretched building/seabed
    // silhouettes into hard notches. Reject disoccluded or inconsistent hits.
    if(d.y>0.&&d.y<.25){
      float travel=40.,z=1.,viewDepth=0.;vec2 uv=vec2(0.);
      vec3 hit=vec3(0.);float clipW=0.;
      for(int i=0;i<3;i++){
        vec3 point=P+d*travel;
        vec4 clip=projectionMatrix*viewMatrix*vec4(point,1.);
        clipW=clip.w;uv=clip.xy/max(clip.w,.001)*.5+.5;
        z=textureLod(uOpaqueDepth,clamp(uv,.001,.999),0.).r;
        viewDepth=uNear*uFar/max(uFar-z*(uFar-uNear),.0001);
        hit=cameraPosition+(point-cameraPosition)*(viewDepth/max(clip.w,.001));
        travel=clamp(dot(hit-P,d),4.,140.);
      }
      float valid=smoothstep(.008,.035,uv.x)*(1.-smoothstep(.965,.992,uv.x))
        *smoothstep(.008,.035,uv.y)*(1.-smoothstep(.965,.992,uv.y))*step(.01,clipW);
      float along=dot(hit-P,d),miss=length(hit-P-d*along);
      float consistent=1.-smoothstep(.10,.65,miss);
      float aboveWater=smoothstep(.08,.35,hit.y);
      float weight=valid*consistent*aboveWater*step(.5,along)*step(z,.99999)
        *smoothstep(0.,.018,d.y)*(1.-smoothstep(.16,.25,d.y));
      transmitted=mix(transmitted,textureLod(uOpaqueColor,clamp(uv,.001,.999),0.).rgb,weight);
    }
  }
  vec3 R=reflect(ray,N);
  // Grazing reflected paths become long and turbid continuously. A binary
  // upward/downward branch produced flat turquoise slivers on the wave faces.
  vec3 reflected=reflectedBed(P,normalize(vec3(R.x,min(R.y,-.001),R.z)));
  vec3 bounced=waterTravel(uSkyAmb*.65,12.);
  reflected=mix(reflected,bounced,smoothstep(-.06,.08,R.y));
  vec3 col=mix(transmitted,reflected,F);
  vec2 flowUV;float turb;
  float foam=surfaceFoam(P,uTime,vInfo.x,flowUV,turb).x;
  foam=clamp(foam*.9,0.,.96);
  vec3 whitewater=(uSkyAmb*.42+uSunColor*.075)*vec3(.76,.88,.83);
  col=mix(col,whitewater,foam);
  gl_FragColor=vec4(col,1.);
}
`;
