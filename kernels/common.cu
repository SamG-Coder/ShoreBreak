// ShoreBreak WebCuda. SI units, +z inland; original MIT attribution in LICENSE.
#define PI 3.141592653589793f
#define TAU 6.283185307179586f
#define ON 128
#define SX 512
#define SZ 192
#define WX 1024
#define WZ 512
__device__ float sat(float x) { return fminf(fmaxf(x,0.0f),1.0f); }
__device__ float mixf(float a,float b,float t) { return a+(b-a)*t; }
__device__ float smooth(float a,float b,float x) { float t=sat((x-a)/(b-a)); return t*t*(3.0f-2.0f*t); }
__device__ float fractf(float x) { return x-floorf(x); }
__device__ float hash(float x) { return fractf(sinf(x*127.1f+311.7f)*43758.5453f); }
__device__ float noise(float x,float y) {
 float ix=floorf(x), iy=floorf(y), u=fractf(x), v=fractf(y); u=u*u*(3.0f-2.0f*u); v=v*v*(3.0f-2.0f*v);
 return mixf(mixf(hash(ix+iy*57.0f),hash(ix+1.0f+iy*57.0f),u),mixf(hash(ix+(iy+1.0f)*57.0f),hash(ix+1.0f+(iy+1.0f)*57.0f),u),v);
}
__device__ float3 v3(float x,float y,float z) { return make_float3(x,y,z); }
__device__ float dot3(float3 a,float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ float3 unit(float3 a) { return a*rsqrtf(fmaxf(dot3(a,a),1e-12f)); }
__device__ float3 cross3(float3 a,float3 b) { return v3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
__device__ float3 mix3(float3 a,float3 b,float f) { return a+(b-a)*f; }
__device__ float3 reflect3(float3 a,float3 n) { return a-n*(2.0f*dot3(a,n)); }
__device__ float2 cmul(float2 a,float2 b) { return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
__device__ float4 lerp4(float4 a,float4 b,float t) { return a+(b-a)*t; }
__device__ int clampi(int x,int a,int b) { return min(max(x,a),b); }
__device__ float4 field(const float4* data,float x,float z,int w,int h) {
 x=fminf(fmaxf(x,0.0f),(float)(w-1)); z=fminf(fmaxf(z,0.0f),(float)(h-1));
 int ix=(int)floorf(x), iz=(int)floorf(z), jx=min(ix+1,w-1), jz=min(iz+1,h-1);
 return lerp4(lerp4(data[iz*w+ix],data[iz*w+jx],fractf(x)),lerp4(data[jz*w+ix],data[jz*w+jx],fractf(x)),fractf(z));
}
__device__ float4 oceanSample(const float4* data,float x,float z,int c) {
 float L=c==0?41.0f:(c==1?7.3f:1.37f);
 float a=c==0?0.0f:(c==1?0.37f:-0.61f);
 float u=(cosf(a)*x-sinf(a)*z)/L*ON, v=(sinf(a)*x+cosf(a)*z)/L*ON;
 int ix=(int)floorf(u),iz=(int)floorf(v), base=c*ON*ON;
 int i00=base+(iz&(ON-1))*ON+(ix&(ON-1));
 int i10=base+(iz&(ON-1))*ON+((ix+1)&(ON-1));
 int i01=base+((iz+1)&(ON-1))*ON+(ix&(ON-1));
 int i11=base+((iz+1)&(ON-1))*ON+((ix+1)&(ON-1));
 return lerp4(lerp4(data[i00],data[i10],fractf(u)),lerp4(data[i01],data[i11],fractf(u)),fractf(v));
}
// Exact original cluster positions; the same cap is queried by hydraulics and rays.
__constant__ float RX[12]={-35.2f,-32.1f,-29.8f,-40.7f,-21.8f,-14.3f,2.6f,13.4f,26.7f,41.5f,56.2f,-56.8f};
__constant__ float RZ[12]={-.65f,-1.25f,.42f,-.9f,-.55f,.15f,-.9f,-.25f,-1.12f,.35f,-.65f,-.38f};
__constant__ float RS[12]={.88f,.70f,.58f,.74f,.81f,.54f,.78f,.72f,.92f,.66f,.85f,.71f};
__device__ float rockHeight(float x,float z) {
 float top=-100.0f;
 if(fabsf(z)>3.0f) return top;
 for(int i=0;i<12;i++) {
  if(fabsf(x-RX[i])>2.0f) continue;
  for(int j=0;j<3;j++) {
   if(j==2 && i%3!=0) continue;
   float seed=(float)i*2.371f+(float)j*5.17f, s=RS[i]*(j==0?1.0f:.37f+.11f*sinf(seed));
   float X=RX[i]+(j==0?0.0f:cosf(seed)*RS[i]*1.32f), Z=RZ[i]+(j==0?0.0f:sinf(seed)*RS[i]*.82f);
   float rz=s*(.59f+.12f*sinf(seed+3.0f)), hh=s*(.53f+.07f*cosf(seed));
   float dx=x-X,dz=z-Z,u=(cosf(seed)*dx+sinf(seed)*dz)/s,v=(-sinf(seed)*dx+cosf(seed)*dz)/rz;
   float a=atan2f(v,u),r=1.0f+.055f*sinf(3.0f*a+seed)+.035f*cosf(5.0f*a-seed*2.0f)+.025f*sinf(7.0f*a+1.7f);
   float rho=sqrtf(u*u+v*v)/r;
   if(rho<1.0f) { float shape=1.0f+.07f*sinf(u*4.1f+seed)*cosf(v*3.8f-1.3f)+.025f*sinf(u*12.0f+v*7.0f+seed); top=fmaxf(top,bedProfile(z)-s*.18f+hh*powf(1.0f-rho*rho,.47f)*shape); }
  }
 }
 return top;
}
__device__ float land(float x,float z) {
 float bed=bedProfile(z);
 if(z>20.0f) bed=mixf(bed,1.55f,smooth(20.0f,32.0f,z));
 if(z>40.0f) bed=5.56f;
 return fmaxf(bed,rockHeight(x,z));
}
// Original nonuniform Hermite stage table, evaluated on the GPU.
__device__ float cubicStage(float tn,int column) {
 int i=0; for(int k=1;k<17;k++) if(tn>=BK_T[k]) i=k;
 int j=min(i+1,17),lo=max(i-1,0),hi=min(i+2,17);
 float a=column==0?BK_ZA[i]:BK_YA[i], b=column==0?BK_ZA[j]:BK_YA[j];
 float l=column==0?BK_ZA[lo]:BK_YA[lo],h=column==0?BK_ZA[hi]:BK_YA[hi];
 float dt=BK_T[j]-BK_T[i],f=sat((tn-BK_T[i])/dt);
 float m1=(b-l)/(BK_T[j]-BK_T[lo])*dt,m2=(h-a)/(BK_T[hi]-BK_T[i])*dt;
 return a+f*(m1+f*(-3.0f*a-2.0f*m1+3.0f*b-m2+f*(2.0f*a+m1-2.0f*b+m2)));
}
__device__ float stageWidth(float tn,int col) {
 int i=0; for(int k=1;k<17;k++) if(tn>=BK_T[k]) i=k;
 float f=sat((tn-BK_T[i])/(BK_T[i+1]-BK_T[i]));
 if(col==0) return mixf(BK_LF[i],BK_LF[i+1],f);
 return mixf(BK_LB[i],BK_LB[i+1],f);
}
__device__ float waveArrival(int id) {
 if(id==0)return -4.84f; if(id==1)return -1.04f; if(id==2)return .85f;
 if(id==3)return 2.84f; if(id==4)return 6.722f; if(id==5)return 10.37f;
 return 10.37f+(float)(id-5)*3.85f+.22f*sinf((float)id*2.31f);
}
__device__ float waveHeight(int id) {
 if(id==0)return .42f; if(id==1)return .50f; if(id==2)return .16f;
 if(id==3)return .45f; if(id==4)return .30f; if(id==5)return .47f;
 return .36f+.12f*hash((float)id*5.91f);
}
__device__ float arrivalAt(int id,float x) {
 float peel=.65f/(1.0f+expf(-(x-.75f)/1.35f));
 float far=.55f*sinf(x*.185f+(float)id*1.71f)+.30f*sinf(x*.33f+(float)id*2.9f)+.15f*sinf(x*.57f+(float)id*4.7f);
 return waveArrival(id)+mixf(peel,far,id>5?1.0f:smooth(5.0f,13.0f,fabsf(x)));
}
__device__ float4 breaker(float x,float z,float t) {
 float height=0.0f,foam=0.0f,lpZ=-100.0f,lpY=-100.0f;
 int first=max(0,(int)floorf((t-10.37f)/3.85f)+3);
 for(int k=0;k<9;k++) {
  int id=first+k; float scale=waveHeight(id)/.45f, tn=(t-arrivalAt(id,x))/sqrtf(scale);
  if(tn < -6.0f || tn > 3.0f)continue;
  float ta=fminf(tn,.85f),za=-1.9f+cubicStage(ta,0)*scale;
  float ya=cubicStage(ta,1)*scale,xi=(z-za)/scale;
  float lf=stageWidth(ta,0),lb=stageWidth(ta,1),h=0.0f;
  if(tn<.85f) {
   if(xi<0.0f){float e=expf(-fabsf(xi/lb));float sec=2.0f*e/(1.0f+e*e);h=ya*sec*sec;}
   else if(xi<lf){float front=powf(sat(xi/lf),mixf(1.0f,.36f,smooth(-1.0f,-.33f,tn)));h=ya*(.5f+.5f*cosf(PI*front));}
   height+=h;
  }
  float boreZ=-1.9f+fmaxf(tn,0.0f)*1.55f;
  float bore=expf(-powf((z-boreZ)/(.38f+fmaxf(tn,0.0f)*.4f),2.0f));
  foam=fmaxf(foam,bore*smooth(-.1f,.12f,tn)*(1.0f-smooth(1.2f,3.0f,tn)));
  if(tn>-.42f && tn<.14f && id!=2) { lpZ=za+.21f*scale;lpY=ya*.70f; }
 }
 return make_float4(height,foam,lpZ,lpY);
}
