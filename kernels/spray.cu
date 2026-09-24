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
