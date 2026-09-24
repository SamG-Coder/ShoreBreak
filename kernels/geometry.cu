__device__ float safeInverse(float x){return 1.0f/(fabsf(x)>1e-8f?x:(x<0.0f?-1e-8f:1e-8f));}
__device__ float3 xyz(float4 v){return v3(v.x,v.y,v.z);}
__device__ bool boundsHit(float3 origin,float3 inv,float3 lo,float3 hi,float maxT){
 float3 a=(lo-origin)*inv,b=(hi-origin)*inv;
 float nearT=fmaxf(fmaxf(fminf(a.x,b.x),fminf(a.y,b.y)),fminf(a.z,b.z));
 float farT=fminf(fminf(fmaxf(a.x,b.x),fmaxf(a.y,b.y)),fmaxf(a.z,b.z));
 return farT>=fmaxf(nearT,.01f)&&nearT<maxT;
}
__device__ float4 traceMesh(const float4* geometry,int root,float3 origin,float3 ray,float3 sun,float maxT){
 float3 inv=v3(safeInverse(ray.x),safeInverse(ray.y),safeInverse(ray.z));
 int node=root,end=(int)geometry[1+root*3].w,triBase=(int)geometry[0].w;
 float4 result=make_float4(-1.0f,0.0f,0.0f,maxT);
 // Stackless traversal: escape links jump past the complete missed subtree.
 while(node<end){
  float4 lo=geometry[1+node*3],hi=geometry[2+node*3],meta=geometry[3+node*3];
  if(!boundsHit(origin,inv,xyz(lo),xyz(hi),result.w)){node=(int)lo.w;continue;}
  int count=(int)meta.x;
  if(count==0){node++;continue;}
  int start=(int)hi.w;
  for(int j=0;j<count;j++){
   int index=triBase+(start+j)*4;float3 a=xyz(geometry[index]),b=xyz(geometry[index+1]),c=xyz(geometry[index+2]);
   float3 e1=b-a,e2=c-a,p=cross3(ray,e2);float det=dot3(e1,p);if(fabsf(det)<1e-8f)continue;
   float invDet=1.0f/det;float3 s=origin-a;float u=dot3(s,p)*invDet;if(u<0.0f||u>1.0f)continue;
   float3 q=cross3(s,e1);float v=dot3(ray,q)*invDet;if(v<0.0f||u+v>1.0f)continue;
   float t=dot3(e2,q)*invDet;if(t<.01f||t>=result.w)continue;
   float3 n=unit(cross3(e1,e2));if(dot3(n,ray)>0.0f)n=n*-1.0f;
   float4 material=geometry[index+3];float diffuse=fmaxf(dot3(n,sun),0.0f);
   float3 color=xyz(material)*(.28f+.95f*diffuse);
   if(material.w>.5f && material.w<1.5f)color+=xyz(material)*fmaxf(-dot3(n,sun),0.0f)*.28f;
   if(material.w>1.5f){float3 pos=origin+ray*t;float row=floorf(pos.y/.3f),u=fractf(pos.x/1.1f+row*.5f),v=fractf(pos.y/.3f);float mortar=smooth(.0f,.025f,fminf(u,fminf(1.0f-u,fminf(v,1.0f-v))));color*=mixf(.78f,1.0f,mixf(mortar,1.0f,smooth(20.0f,100.0f,t)));color*=.9f+.12f*noise(pos.x*.7f,pos.y*.5f);}
   result=make_float4(color.x,color.y,color.z,t);
  }
  node=(int)lo.w;
 }
 return result;
}
__device__ float3 rotateY(float3 p,float a){return v3(cosf(a)*p.x+sinf(a)*p.z,p.y,-sinf(a)*p.x+cosf(a)*p.z);}
__device__ float4 traceScenery(const float4* geometry,float3 origin,float3 ray,float maxT){
 float3 sun=unit(v3(.26f,.91f,-.31f));
 float4 result=traceMesh(geometry,(int)geometry[0].x,origin,ray,sun,maxT);
 // Each original tree retains its instanced transform and seed. BLAS is shared.
 for(int i=-30;i<=30;i++){
  float x=(float)i*14.2f+sinf((float)i*7.31f)*1.4f,z=abs(i%3)==1?49.5f:59.8f;
  float bend=powf(fmaxf(fabsf(x)-150.0f,0.0f),2.0f)/20000.0f;z-=bend;
  float3 root=v3(x,5.56f,z);float scale=.79f+.12f*(.5f+.5f*sinf((float)i*11.7f));
  float3 lo=root+v3(-5.5f,0.0f,-5.5f),hi=root+v3(5.5f,16.0f,5.5f);
  float3 inv=v3(safeInverse(ray.x),safeInverse(ray.y),safeInverse(ray.z));
  if(!boundsHit(origin,inv,lo,hi,result.w))continue;
  float a=-(float)i*2.4f;float3 local=rotateY(origin-root,a)/scale,dir=rotateY(ray,a);
  int tree=(int)(i%2==0?geometry[0].y:geometry[0].z);
  float4 hit=traceMesh(geometry,tree,local,dir,rotateY(sun,a),result.w/scale);
  if(hit.x>=0.0f){hit.w*=scale;result=hit;}
 }
 return result;
}
