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
