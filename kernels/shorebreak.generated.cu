// Generated from ShoreBreak's original bathymetry and breaker stage table.
__device__ float softplus(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }
__device__ float bedProfile(float z) { float y = 0.1f*z;
y += 0.1f*5.0f*softplus((z-(-60.0f))/5.0f);
y += 0.8f*1.2f*softplus((z-(-22.5f))/1.2f);
y += -0.9f*0.5f*softplus((z-(-16.8f))/0.5f);
y += 0.01999999999999999f*2.0f*softplus((z-(-12.0f))/2.0f);
y += 0.22999999999999998f*0.22f*softplus((z-(-2.3f))/0.22f);
y += -0.21499999999999997f*0.18f*softplus((z-(-0.6f))/0.18f);
y += -0.05f*1.5f*softplus((z-(11.0f))/1.5f);
y += -0.135f*1.2f*softplus((z-(19.0f))/1.2f);
return fmaxf(y + (-9.51869802609202f), -30.0f); }
__constant__ float BK_T[18] = {-6.0f,-4.0f,-3.0f,-2.0f,-1.4f,-1.0f,-0.8f,-0.55f,-0.4f,-0.33f,-0.22f,-0.12f,-0.04f,0.0f,0.12f,0.3f,0.6f,0.85f};
__constant__ float BK_ZA[18] = {-23.0f,-15.5f,-11.3f,-7.0f,-4.3f,-2.55f,-1.85f,-1.08f,-0.625f,-0.511f,-0.461f,-0.472f,-0.477f,-0.481f,-0.475f,-0.46f,-0.44f,-0.43f};
__constant__ float BK_YA[18] = {0.2f,0.24f,0.28f,0.33f,0.41f,0.46f,0.5f,0.548f,0.56f,0.553f,0.484f,0.392f,0.346f,0.3f,0.2f,0.08f,0.01f,0.0f};
__constant__ float BK_YT[18] = {-0.1f,-0.1f,-0.1f,-0.1f,-0.09f,-0.08f,-0.068f,-0.068f,-0.068f,-0.068f,-0.068f,-0.068f,-0.068f,-0.068f,-0.06f,-0.04f,-0.02f,0.0f};
__constant__ float BK_LF[18] = {8.0f,7.0f,6.0f,4.0f,2.2f,1.4f,1.04f,0.6f,0.46f,0.42f,0.42f,0.4f,0.38f,0.36f,0.4f,0.6f,1.0f,1.2f};
__constant__ float BK_UW[18] = {0.3f,0.3f,0.3f,0.3f,0.3f,0.3f,0.3f,0.26f,0.12f,0.04f,0.02f,0.02f,0.02f,0.02f,0.05f,0.2f,0.3f,0.3f};
__constant__ float BK_HW[18] = {0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.62f,0.66f,0.54f,0.56f,0.53f,0.47f,0.41f,0.4f,0.5f,0.5f,0.5f};
__constant__ float BK_M[18] = {1.6f,1.6f,1.6f,1.6f,1.6f,1.8f,2.0f,2.6f,3.2f,3.6f,4.5f,5.0f,5.5f,6.0f,4.0f,2.0f,2.0f,2.0f};
__constant__ float BK_P[18] = {2.0f,2.0f,2.0f,2.0f,2.0f,2.0f,2.0f,2.0f,3.0f,3.0f,3.0f,3.0f,3.0f,3.0f,2.0f,2.0f,2.0f,2.0f};
__constant__ float BK_ST[18] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.55f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,0.8f,0.3f,0.0f,0.0f};
__constant__ float BK_LB[18] = {4.0f,3.8f,3.6f,3.2f,2.9f,2.7f,2.6f,2.5f,2.4f,2.4f,2.4f,2.4f,2.4f,2.4f,2.4f,2.6f,3.0f,3.0f};
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

__device__ float spectrumPower(float kx,float kz,float L,int cascade) {
 float k=sqrtf(kx*kx+kz*kz);
 float lo=cascade==0?0.0f:(cascade==1?TAU/2.5f:TAU/.3f),hi=cascade==0?TAU/2.5f:(cascade==1?TAU/.3f:100000.0f);
 if(k<.00001f || k<lo || k>=hi)return 0.0f;
 float w=sqrtf(9.81f*k*(1.0f+k*k/(370.0f*370.0f))),wp=22.0f*powf(9.81f*9.81f/(3.0f*2000.0f),1.0f/3.0f);
 float alpha=.076f*powf(9.0f/(2000.0f*9.81f),.22f),sig=w<=wp?.07f:.09f;
 float r=expf(-powf(w-wp,2.0f)/(2.0f*sig*sig*wp*wp));
 float sw=alpha*9.81f*9.81f/powf(w,5.0f)*expf(-1.25f*powf(wp/w,4.0f))*powf(3.3f,r);
 float cs=kz/k,spread=powf(fabsf(cs),cascade==0?1.2f:(cascade==1?.6f:.5f))*(cs>0.0f?1.0f:.2f)/3.2f;
 return sw*9.81f*(1.0f+3.0f*k*k/(370.0f*370.0f))/(2.0f*w*k)*spread*expf(-powf(k/520.0f,2.0f))*powf(TAU/L,2.0f);
}
__device__ float2 initialH(int x,int y,int c) {
 float L=c==0?41.0f:(c==1?7.3f:1.37f);
 float kx=TAU/L*(float)(x<ON/2?x:x-ON),kz=TAU/L*(float)(y<ON/2?y:y-ON);
 float seed=(float)(y*ON+x+c*ON*ON)+1234.0f;
 float r=sqrtf(-2.0f*logf(fmaxf(hash(seed),1e-7f)))*sqrtf(spectrumPower(kx,kz,L,c))*.5f;
 float a=TAU*hash(seed+917.31f); return make_float2(r*cosf(a),r*sinf(a));
}
__global__ void initializeSpectrum(float4* h0) {
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=ON*3 || y>=ON)return;int c=x/ON,n=x%ON,i=c*ON*ON+y*ON+n;
 float2 a=initialH(n,y,c),b=initialH((ON-n)%ON,(ON-y)%ON,c);h0[i]=make_float4(a.x,a.y,b.x,b.y);
}
__global__ void evolveSpectrum(const float4* h0,float4* fftA,float4* fftB,float time) {
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=ON*3 || y>=ON)return;int c=x/ON,n=x%ON,i=c*ON*ON+y*ON+n;
 float L=c==0?41.0f:(c==1?7.3f:1.37f),kx=TAU/L*(float)(n<ON/2?n:n-ON),kz=TAU/L*(float)(y<ON/2?y:y-ON);
 float kl=sqrtf(kx*kx+kz*kz),w=sqrtf(9.81f*kl*(1.0f+kl*kl/136900.0f)),a=fractf(w*time/TAU)*TAU;
 float4 s=h0[i];float2 e=make_float2(cosf(a),sinf(a));
 float2 h=cmul(make_float2(s.x,s.y),make_float2(e.x,-e.y))+cmul(make_float2(s.z,-s.w),e);
 float2 ih=make_float2(-h.y,h.x),sx=ih*kx,sz=ih*kz,dx=ih*(-kx/fmaxf(kl,1e-6f)),dz=ih*(-kz/fmaxf(kl,1e-6f));
 fftA[i]=make_float4(h.x-sx.y,h.y+sx.x,sz.x-dx.y,sz.y+dx.x);fftB[i]=make_float4(dz.x,dz.y,0.0f,0.0f);
}
__global__ void fftStage(const float4* inA,const float4* inB,float4* outA,float4* outB,int sub,int horizontal) {
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=ON*3 || y>=ON)return;int c=x/ON,n=x%ON,idx=horizontal==1?n:y,hs=sub/2;
 int ev=(idx/sub)*hs+(idx&(hs-1)),od=ev+ON/2,base=c*ON*ON;
 int ie=base+(horizontal==1?y*ON+ev:ev*ON+n),io=base+(horizontal==1?y*ON+od:od*ON+n);
 float angle=TAU*(float)(idx&(sub-1))/(float)sub;float2 w=make_float2(cosf(angle),sinf(angle));
 float4 a=inA[ie],b=inA[io],d=inB[ie],e=inB[io];
 float2 p=cmul(w,make_float2(b.x,b.y)),q=cmul(w,make_float2(b.z,b.w)),r=cmul(w,make_float2(e.x,e.y));
 int i=base+y*ON+n;outA[i]=a+make_float4(p.x,p.y,q.x,q.y);outB[i]=d+make_float4(r.x,r.y,0.0f,0.0f);
}
__global__ void composeOcean(const float4* inA,const float4* inB,float4* ocean) {
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=ON*3 || y>=ON)return;int c=x/ON,n=x%ON,i=c*ON*ON+y*ON+n;
 float4 a=inA[i];ocean[i]=make_float4(a.x,a.y,a.z,inB[i].x);
}

__device__ float4 swashAt(const float4* state,float x,float z) {return field(state,(x+64.0f)/128.0f*SX-.5f,(z+6.0f)/18.0f*SZ-.5f,SX,SZ);}
__device__ float4 surfaceAt(const float4* water,float x,float z) {return field(water,(x+160.0f)/320.0f*(WX-1),(z+120.0f)/140.0f*(WZ-1),WX,WZ);}
__global__ void initializeCoast(float4* bed,float4* state,float4* wetness) {
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=SX||z>=SZ)return;int i=z*SX+x;
 float X=-64.0f+((float)x+.5f)*128.0f/SX,Z=-6.0f+((float)z+.5f)*18.0f/SZ,b=land(X,Z);
 bed[i]=make_float4(b,X,Z,0.0f);state[i]=make_float4(fmaxf(-b,0.0f),0.0f,0.0f,0.0f);wetness[i]=make_float4(b<0.0f?1.0f:0.0f,0.0f,0.0f,0.0f);
}
// Hydrostatic reconstruction + Rusanov flux. Momentum is h*u, h*v.
// The bed pressure correction preserves a resting lake over irregular rocks.
__device__ float4 flux(float4 a,float4 b,float ba,float bb,int axis) {
 float top=fmaxf(ba,bb),ha=fmaxf(a.x+ba-top,0.0f),hb=fmaxf(b.x+bb-top,0.0f);
 float ua=a.x>.0001f?a.y/a.x:0.0f,va=a.x>.0001f?a.z/a.x:0.0f,ub=b.x>.0001f?b.y/b.x:0.0f,vb=b.x>.0001f?b.z/b.x:0.0f;
 float ca=axis==0?ua:va,cb=axis==0?ub:vb,sa=fabsf(ca)+sqrtf(9.81f*ha),sb=fabsf(cb)+sqrtf(9.81f*hb),s=fmaxf(sa,sb);
 float4 qa=make_float4(ha,ha*ua,ha*va,ha*a.w),qb=make_float4(hb,hb*ub,hb*vb,0.0f);
 qb.w=hb*b.w;
 float4 fa=qa*ca,fb=qb*cb;
 if(axis==0){fa.y+=4.905f*ha*ha;fb.y+=4.905f*hb*hb;}else{fa.z+=4.905f*ha*ha;fb.z+=4.905f*hb*hb;}
 float4 out=(fa+fb-(qb-qa)*s)*.5f;
 float pressure=4.905f*(a.x*a.x-ha*ha);if(axis==0)out.y+=pressure;else out.z+=pressure;
 return out;
}
__global__ void stepSwash(const float4* input,const float4* bed,float4* output,float time,float dt,int forcing) {
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;if(x>=SX||z>=SZ)return;int i=z*SX+x;
 float4 a=input[i],b=bed[i];
 if(z==0) {
  float4 br=breaker(b.y,b.z,time);float eta=forcing==1?br.x:0.0f,h=fmaxf(eta-b.x,0.0f);
  output[i]=make_float4(h,0.0f,forcing==1?eta*sqrtf(9.81f*fmaxf(h,.01f)):0.0f,br.y);return;
 }
 int il=z*SX+max(x-1,0),ir=z*SX+min(x+1,SX-1),ib=max(z-1,0)*SX+x,it=min(z+1,SZ-1)*SX+x;
 float4 left=input[il],right=input[ir],back=input[ib],front=input[it];
 if(x==0)left.y=-a.y;if(x==SX-1)right.y=-a.y;if(z==SZ-1)front.z=-a.z;
 float4 fR=flux(a,right,b.x,bed[ir].x,0),fL=flux(a,left,b.x,bed[il].x,0);
 // Reverse the interface orientation for left/back: negate velocities, then invert mass and transverse flux.
 float4 ar=a,al=left;ar.y=-ar.y;al.y=-al.y;fL=flux(ar,al,b.x,bed[il].x,0);fL.x=-fL.x;fL.z=-fL.z;fL.w=-fL.w;
 float4 fT=flux(a,front,b.x,bed[it].x,1),ab=a,bb=back;ab.z=-ab.z;bb.z=-bb.z;
 float4 fB=flux(ab,bb,b.x,bed[ib].x,1);fB.x=-fB.x;fB.y=-fB.y;fB.w=-fB.w;
 float oldFoamMass=a.w*a.x;float4 q=a;q.w=oldFoamMass;
 q-=(fR-fL)*(dt*SX/128.0f)+(fT-fB)*(dt*SZ/18.0f);
 q.x=fmaxf(q.x-dt*.004f*smooth(0.0f,.05f,b.x),0.0f);
 if(q.x<.0002f){q.y=0.0f;q.z=0.0f;q.w=0.0f;}else{
  float speed=sqrtf(q.y*q.y+q.z*q.z)/q.x;
  float friction=1.0f+dt*9.81f*.022f*.022f*speed/powf(fmaxf(q.x,.003f),1.333333f);
  q.y/=friction;q.z/=friction;
  // Limit the characteristic velocity to the fixed-step CFL envelope.
  q.y=fminf(fmaxf(q.y,-q.x*3.0f),q.x*3.0f);q.z=fminf(fmaxf(q.z,-q.x*3.0f),q.x*3.0f);
  float injected=forcing==1?breaker(b.y,b.z,time).y:0.0f;
  q.w=sat(q.w/q.x*expf(-dt*.45f)+dt*(injected*3.0f+smooth(.7f,2.0f,speed)*.3f));
 }
 output[i]=q;
}
__global__ void updateWetness(const float4* state,float4* wetness,float dt) {
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;if(x>=SX||z>=SZ)return;int i=z*SX+x;
 float4 w=wetness[i],q=state[i];w.x=q.x>.001f?1.0f:fmaxf(w.x-dt*.04f,0.0f);w.y=q.x>.001f?1.0f:fmaxf(w.y-dt*2.2f,0.0f);wetness[i]=w;
}
__global__ void buildSurface(const float4* ocean,const float4* state,const float4* bed,float4* water,float time) {
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;if(x>=WX||z>=WZ)return;int i=z*WX+x;
 float X=-160.0f+(float)x/(WX-1)*320.0f,Z=-120.0f+(float)z/(WZ-1)*140.0f;
 float4 br=breaker(X,Z,time);float h=br.x,foam=br.y;
 h+=oceanSample(ocean,X,Z,0).x*(1.0f-smooth(-5.0f,0.0f,Z));
 if(Z<-24.0f) h+=.08f*sinf(Z*.16f+time*1.65f+sinf(X*.017f))*(1.0f-smooth(-28.0f,-24.0f,Z));
 if(Z>-5.8f && fabsf(X)<63.8f && Z<11.9f) {
  float4 q=swashAt(state,X,Z);float b=field(bed,(X+64.0f)/128.0f*SX-.5f,(Z+6.0f)/18.0f*SZ-.5f,SX,SZ).x;
  float blend=smooth(-5.8f,-3.0f,Z)*(1.0f-smooth(60.0f,63.8f,fabsf(X)));
  h=mixf(h,b+q.x,blend);foam=fmaxf(foam,q.w*blend);
 }else if(Z>-.5f) { h=fmaxf(h,bedProfile(Z)); }
 water[i]=make_float4(h,foam,br.z,br.w);
}
// GPU-owned camera: [position + vertical speed, yaw/pitch/crouch/swimming].
__global__ void moveCamera(float4* camera,const float4* water,float dt,float lookX,float lookY,int keys,int reset,int steady,float time) {
 if(reset==1){camera[0]=make_float4(-34.0f,land(-34.0f,6.8f)+1.68f,6.8f,0.0f);camera[1]=make_float4(1.6406095f,-.0733038f,0.0f,0.0f);camera[2]=make_float4(0.0f,0.0f,0.0f,1.68f);return;}
 float4 p=camera[0],v=camera[1],motion=camera[2];v.x+=lookX;v.y=fminf(fmaxf(v.y+lookY,-1.48f),1.48f);
 float f=((keys&1)!=0?1.0f:0.0f)-((keys&2)!=0?1.0f:0.0f),r=((keys&8)!=0?1.0f:0.0f)-((keys&4)!=0?1.0f:0.0f);
 float norm=rsqrtf(fmaxf(f*f+r*r,1.0f));f*=norm;r*=norm;
 float ground=land(p.x,p.z),sea=surfaceAt(water,p.x,p.z).x;bool swim=sea-ground>(v.w>.5f?1.0f:1.15f);bool crouch=(keys&32)!=0;
 float speed=swim?((keys&16)!=0?2.5f:1.7f):((keys&16)!=0?4.5f:2.2f);if(crouch)speed*=.6f;
 if(!swim)speed*=mixf(1.0f,.60f,smooth(.15f,1.1f,sea-ground));
 float nx=p.x+(sinf(v.x)*f+cosf(v.x)*r)*speed*dt,nz=p.z+(-cosf(v.x)*f+sinf(v.x)*r)*speed*dt;
 nx=fminf(fmaxf(nx,-145.0f),145.0f);nz=fminf(fmaxf(nz,-105.0f),30.0f);
 if(land(nx,nz)-ground<.38f){p.x=nx;p.z=nz;}
 ground=land(p.x,p.z);sea=surfaceAt(water,p.x,p.z).x;
 motion.w=mixf(motion.w,crouch?.83f:1.68f,1.0f-expf(-dt*10.0f));float target=ground+motion.w;
 if(swim){target=fmaxf(ground+.6f,sea+(crouch?-.4f:.18f));p.y=mixf(p.y,target,1.0f-expf(-dt*5.0f));p.w=0.0f;}
 else {if((keys&64)!=0 && motion.z<.5f && p.y<=target+.025f)p.w=3.2f;p.w-=9.81f*dt;p.y+=p.w*dt;if(p.y<target){p.y=target;p.w=0.0f;}}
 motion.x+=dt*speed*4.0f;motion.z=(keys&64)!=0?1.0f:0.0f;
 float bob=steady==1||swim?0.0f:sinf(motion.x)*.012f*fminf(f*f+r*r,1.0f);
 motion.y=mixf(motion.y,bob,1.0f-expf(-dt*12.0f));camera[2]=motion;
 v.z=crouch?1.0f:0.0f;v.w=swim?1.0f:0.0f;camera[0]=p;camera[1]=v;
}

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

#define PARTICLES 32768
// Deterministic ballistic droplets and entrained rising air. CUDA evaluates
// event ages directly, so seeking and different frame rates reproduce the cloud.
__global__ void animateSpray(float4* particles,float time){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=PARTICLES)return;
 float seed=(float)i,x=(hash(seed*1.91f)-.5f)*130.0f;
 int candidate=max(0,(int)floorf((time-10.37f)/3.85f)+4);float age=-10.0f;int event=0;
 for(int k=0;k<7;k++){int id=candidate+k;float a=time-arrivalAt(id,x)-hash(seed+17.0f)*.18f;if(a>=0.0f&&a<1.3f){age=a;event=id;}}
 if(age<0.0f){particles[i]=make_float4(0.0f,-100.0f,0.0f,0.0f);return;}
 float scale=waveHeight(event)/.45f;
 float vx=(hash(seed+81.0f)-.5f)*.8f,vz=.4f+hash(seed+73.0f)*1.8f,vy=.6f+hash(seed+19.0f)*2.3f;
 float y=.04f+vy*age-4.905f*age*age,z=-1.9f+vz*age;
 float radius=.005f+hash(seed+21.0f)*.018f;
 if(i%3==0){y=-.08f-hash(seed+29.0f)*.4f+age*.17f;radius*=.6f;z-=age*.3f;}
 particles[i]=make_float4(x+vx*age,y*scale,z,radius*(1.0f-smooth(.7f,1.3f,age)));
}
__global__ void projectSpray(const float4* particles,const float4* camera,const float4* hdr,unsigned int* spray,int width,int height){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=PARTICLES)return;float4 p=particles[i];if(p.w<=0.0f)return;
 float4 cp=camera[0],cv=camera[1];float3 origin=v3(cp.x,cp.y+camera[2].y,cp.z),pos=v3(p.x,p.y,p.z)-origin;
 float3 f=v3(sinf(cv.x)*cosf(cv.y),sinf(cv.y),-cosf(cv.x)*cosf(cv.y)),r=v3(cosf(cv.x),0.0f,sinf(cv.x)),u=cross3(r,f);
 float depth=dot3(pos,f);if(depth<.05f)return;
 float cx=(dot3(pos,r)/depth/.624869f*(float)height/(float)width*.5f+.5f)*(float)width;
 float cy=(.5f-dot3(pos,u)/depth/.624869f*.5f)*(float)height;
 float radius=fminf(fmaxf(p.w/depth*(float)height/.624869f,.65f),3.0f),distance=sqrtf(dot3(pos,pos));
 int ix=(int)floorf(cx),iy=(int)floorf(cy);
 for(int y=-3;y<=3;y++)for(int x=-3;x<=3;x++){
  int px=ix+x,py=iy+y;if(px<0||py<0||px>=width||py>=height)continue;
  float dx=(float)px+.5f-cx,dy=(float)py+.5f-cy,a=sat(1.0f-sqrtf(dx*dx+dy*dy)/radius)*.8f;
  if(a<=0.0f||distance>hdr[py*width+px].w+.04f)continue;
  atomicMax(&spray[py*width+px],(unsigned int)(a*65535.0f));
 }
}

// Linear-light box filtering of the original image assets; all mips made on GPU.
__global__ void materialMip(unsigned int* textures,int side,int offset,int previous){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=side||y>=side)return;
 for(int tile=0;tile<2;tile++){
  float3 sum=v3(0.0f,0.0f,0.0f);
  for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++){float3 c=unpack(textures[tile*349525+previous+(y*2+dy)*side*2+x*2+dx]);sum+=c*c;}
  sum*=.25f;unsigned int r=(unsigned int)(sqrtf(sum.x)*255.0f),g=(unsigned int)(sqrtf(sum.y)*255.0f),b=(unsigned int)(sqrtf(sum.z)*255.0f);
  textures[tile*349525+offset+y*side+x]=r|(g<<8)|(b<<16)|4278190080u;
 }
}
