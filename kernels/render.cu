// Every displayed pixel is authored in CUDA. No vertex/fragment/WGSL source is handwritten.
__device__ float3 sky(float3 d) {
 float3 sun=unit(v3(.26f,.91f,-.31f));float mu=fmaxf(dot3(d,sun),0.0f),up=sat(d.y);
 float3 c=mix3(v3(.30f,.43f,.55f),v3(.035f,.15f,.32f),powf(up,.38f));
 c+=v3(1.0f,.82f,.57f)*(powf(mu,18.0f)*.13f+powf(mu,6500.0f)*7.0f);
 float cloud=noise(d.x/fmaxf(d.y,.08f)*.9f,d.z/fmaxf(d.y,.08f)*.9f);
 cloud=smooth(.72f,.9f,cloud)*(1.0f-smooth(.35f,.7f,up))*.12f;c=mix3(c,v3(.60f,.68f,.73f),cloud);
 return c;
}
__device__ float3 unpack(unsigned int c) {return v3((float)(c&255),(float)((c>>8)&255),(float)((c>>16)&255))/255.0f;}
__device__ float3 photo(const unsigned int* textures,float x,float z,int tile,float distance) {
 int level=clampi((int)floorf(log2f(fmaxf(distance*.45f,1.0f))),0,9),side=512,offset=0;
 for(int j=0;j<level;j++){offset+=side*side;side/=2;}
 float u=fractf(x)*(float)side,v=fractf(z)*(float)side;int ix=(int)floorf(u),iz=(int)floorf(v),b=tile*349525+offset,mask=side-1;
 float3 a=unpack(textures[b+(iz&mask)*side+(ix&mask)]),c=unpack(textures[b+(iz&mask)*side+((ix+1)&mask)]);
 float3 d=unpack(textures[b+((iz+1)&mask)*side+(ix&mask)]),e=unpack(textures[b+((iz+1)&mask)*side+((ix+1)&mask)]);
 float3 color=mix3(mix3(a,c,fractf(u)),mix3(d,e,fractf(u)),fractf(v));return color*color;
}
__device__ float3 groundColor(const unsigned int* textures,const float4* wetness,float3 p,float distance,float time) {
 float rock=rockHeight(p.x,p.z),isRock=rock>bedProfile(p.z)+.012f?1.0f:0.0f;
 float3 c=photo(textures,p.x*.33f,p.z*.33f,isRock>.5f?1:0,distance);
 if(isRock<.5f)c=mix3(c,v3(.43f,.40f,.32f),.70f);
 float grains=noise(p.x*90.0f,p.z*90.0f),macro=noise(p.x*.15f,p.z*.15f);
 c*=mixf(.78f,1.15f,macro);c*=mixf(1.0f,.80f+grains*.35f,1.0f-smooth(4.0f,18.0f,distance));
 float damp=sat(-p.z*.5f+.4f),film=0.0f;
 if(fabsf(p.x)<64.0f && p.z>-6.0f && p.z<12.0f){float4 w=swashAt(wetness,p.x,p.z);damp=w.x;film=w.y;}
 c*=mixf(1.0f,.50f,damp);
 float e=.025f;float3 n=unit(v3(land(p.x-e,p.z)-land(p.x+e,p.z),2.0f*e,land(p.x,p.z-e)-land(p.x,p.z+e)));
 float light=.35f+.8f*fmaxf(dot3(n,unit(v3(.26f,.91f,-.31f))),0.0f);c*=light;
 if(p.z<0.0f){float caustic=powf(fabsf(sinf(p.x*5.3f+time*1.1f+sinf(p.z*7.0f))*sinf(p.z*5.8f-time*.8f+sinf(p.x*6.2f))),14.0f);c+=v3(.2f,.3f,.21f)*caustic;}
 if(p.z>24.0f){float seam=smooth(.015f,.03f,fminf(fractf(p.x*.5f),fractf(p.z*.5f)));c=v3(.38f,.36f,.32f)*mixf(.6f,1.0f,seam);}
 return c;
}
__device__ float seaHeight(const float4* water,const float4* ocean,float x,float z,float distance) {
 float h=surfaceAt(water,x,z).x;
 if(z<-120.0f)h=0.0f;
 float atten=(1.0f-smooth(-4.0f,.0f,z));
 h+=(oceanSample(ocean,x,z,1).x*(1.0f-smooth(10.0f,65.0f,distance))+oceanSample(ocean,x,z,2).x*(1.0f-smooth(2.0f,20.0f,distance)))*atten;
 return h;
}
__global__ void renderCoast(const float4* camera,const float4* water,const float4* ocean,const float4* wetness,const unsigned int* textures,const float4* geometry,float4* hdr,int width,int height,float time) {
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=width||y>=height)return;
 float4 cp=camera[0],cv=camera[1];float3 origin=v3(cp.x,cp.y+camera[2].y,cp.z);
 float3 forward=v3(sinf(cv.x)*cosf(cv.y),sinf(cv.y),-cosf(cv.x)*cosf(cv.y));
 float3 right=v3(cosf(cv.x),0.0f,sinf(cv.x)),up=cross3(right,forward);
 float px=((float)x+.5f)/(float)width*2.0f-1.0f,py=1.0f-((float)y+.5f)/(float)height*2.0f;
 float3 ray=unit(forward+right*(px*(float)width/(float)height*.624869f)+up*(py*.624869f));
 float3 color=sky(ray);float hitT=1500.0f,kind=0.0f;
 bool underwater=origin.y<seaHeight(water,ocean,origin.x,origin.z,0.0f)-.015f;
 // Intersect the terrain and surface together with adaptive small steps around breakers.
 float t=.07f,prevT=t,prevSign=0.0f;bool hit=false;
 for(int step=0;step<220;step++) {
  float3 p=origin+ray*t;
  if(t>1500.0f || p.z>50.0f || p.y>8.0f && ray.y>0.0f)break;
  float ground=land(p.x,p.z),sea=seaHeight(water,ocean,p.x,p.z,t);
  bool wet=sea>ground+.003f && p.z<12.0f;
  float surface=wet?sea:ground,sd=underwater?(wet?sea-p.y:100.0f):p.y-surface;
  float groundD=p.y-ground;
  if(groundD<.003f || sd<.002f) {
   float lo=prevT,hi=t;
   for(int j=0;j<7;j++){float mid=(lo+hi)*.5f;float3 q=origin+ray*mid;float g=land(q.x,q.z),s=seaHeight(water,ocean,q.x,q.z,mid);float v=underwater?fminf(q.y-g,s-q.y):q.y-fmaxf(g,s);if(v>0.0f)lo=mid;else hi=mid;}
   t=(lo+hi)*.5f;hit=true;kind=groundD<.003f?1.0f:(wet?2.0f:1.0f);break;
  }
  prevT=t;
  float dist=fminf(sd,groundD);float rate=fabsf(ray.y)+mixf(.65f,.045f,smooth(20.0f,80.0f,t));
  float stride=fmaxf(.018f,dist*.58f/rate);
  t+=fminf(stride,12.0f+t*.025f);
 }
 if(hit){
  hitT=t;float3 p=origin+ray*t;
  if(kind<1.5f)color=groundColor(textures,wetness,p,t,time);
  else{
   float eps=mixf(.015f,.15f,sat(t/100.0f));
   float hx=seaHeight(water,ocean,p.x+eps,p.z,t)-seaHeight(water,ocean,p.x-eps,p.z,t);
   float hz=seaHeight(water,ocean,p.x,p.z+eps,t)-seaHeight(water,ocean,p.x,p.z-eps,t);
   float3 n=unit(v3(-hx,2.0f*eps,-hz));if(underwater)n=n*-1.0f;
   float facing=sat(-dot3(ray,n)),fresnel=.0204f+.9796f*powf(1.0f-facing,5.0f);
   float3 reflected=sky(reflect3(ray,n)),sun=unit(v3(.26f,.91f,-.31f)),halfway=unit(sun-ray);
   float glint=powf(fmaxf(dot3(n,halfway),0.0f),mixf(360.0f,70.0f,smooth(8.0f,65.0f,t)))*mixf(2.0f,.16f,smooth(8.0f,65.0f,t));
   float depth=fmaxf(p.y-land(p.x,p.z),.0f),path=depth/fmaxf(facing,.25f);
   float3 substrate=groundColor(textures,wetness,v3(p.x+ray.x*path*.65f,land(p.x,p.z),p.z+ray.z*path*.65f),t,time);
   float3 trans=v3(expf(-path*.40f),expf(-path*.12f),expf(-path*.075f));
   float3 body=substrate*trans+v3(.012f,.26f,.25f)*(v3(1.0f,1.0f,1.0f)-trans);
   color=mix3(body,reflected,fresnel)+v3(1.0f,.92f,.77f)*glint;
   float4 fieldValue=surfaceAt(water,p.x,p.z);float foam=fieldValue.y;
   float cells=noise(p.x*22.0f+time*.2f,p.z*22.0f-time*.5f),lace=noise(p.x*4.0f,p.z*4.0f-time*.3f);
   foam=smooth(.12f,.60f,foam)*smooth(.16f,.58f,cells+lace*.45f);
   color=mix3(color,v3(.86f,.91f,.87f),foam);
   if(underwater){color=mix3(v3(.025f,.24f,.25f),color,facing);}
  }
  color=mix3(color,sky(ray),1.0f-expf(-t*.0012f));
 }
 // Thin curling sheet over the height field: actual overhang, shared event timing.
 if(!underwater && ray.z<-.02f && origin.z>-6.0f) {
  float start=fmaxf(.03f,(-1.0f-origin.z)/ray.z),end=fminf(hitT,(-4.0f-origin.z)/ray.z);
  float lt=start;
  for(int j=0;j<65;j++){
   if(lt>end)break;float3 p=origin+ray*lt;float4 br=breaker(p.x,p.z,time);
   if(br.z>-50.0f){float dz=p.z-br.z,dy=p.y-br.w,r=.16f;float circle=sqrtf(dz*dz+dy*dy);float a=atan2f(dy,dz);
    float d=fabsf(circle-r)-.018f;
    if(a>-.65f && a<2.6f && d<.006f && p.y>land(p.x,p.z)+.04f){float3 n=unit(v3(.08f,dy,dz));color=mix3(v3(.018f,.29f,.26f),sky(reflect3(ray,n)),.30f);color+=v3(.45f,.54f,.48f)*smooth(.0f,.02f,fabsf(d));hitT=lt;break;}
    lt+=fmaxf(.02f,d*.65f);
   }else break;
  }
 }
 float4 city=traceScenery(geometry,origin,ray,hitT);
 if(city.x>=0.0f)color=mix3(xyz(city),sky(ray),1.0f-expf(-city.w*.0012f));
 if(underwater){float distance=hit?hitT:35.0f;float tr=expf(-distance*.14f);color=mix3(v3(.012f,.17f,.20f),color,tr);float speck=powf(noise(px*150.0f+time*.7f,py*150.0f-time*.4f),24.0f);color+=v3(.13f,.23f,.21f)*speck;}
 hdr[y*width+x]=make_float4(color.x,color.y,color.z,hitT);
}
__device__ float tone(float c) { c=fmaxf(c,0.0f)*1.18f;return sat((c*(2.51f*c+.03f))/(c*(2.43f*c+.59f)+.14f)); }
__global__ void finishFrame(const float4* hdr,const unsigned int* spray,unsigned int* pixels,int width,int height,int stride,int bgra) {
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=width||y>=height)return;
 float4 c=hdr[y*width+x];float d=(hash((float)(y*width+x))-.5f)/255.0f;
 float4 l=hdr[y*width+max(x-1,0)],r0=hdr[y*width+min(x+1,width-1)],u=hdr[max(y-1,0)*width+x],v=hdr[min(y+1,height-1)*width+x];
 float contrast=fmaxf(fabsf(l.x-r0.x),fabsf(u.x-v.x));
 if(contrast>.12f)c=lerp4(c,(l+r0+u+v)*.25f,.35f);
 float foam=(float)spray[y*width+x]/65535.0f;
 c=lerp4(c,make_float4(.9f,.95f,.92f,0.0f),sat(foam));
 unsigned int r=(unsigned int)(sat(powf(tone(c.x),1.0f/2.2f)+d)*255.0f),g=(unsigned int)(sat(powf(tone(c.y),1.0f/2.2f)+d)*255.0f),b=(unsigned int)(sat(powf(tone(c.z),1.0f/2.2f)+d)*255.0f);
 pixels[y*stride+x]=bgra==1?(b|(g<<8)|(r<<16)|4278190080u):(r|(g<<8)|(b<<16)|4278190080u);
}
