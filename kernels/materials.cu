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
