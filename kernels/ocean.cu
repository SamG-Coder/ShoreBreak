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
