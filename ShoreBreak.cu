// ShoreBreak calculations ported to CUDA for SamG-Coder's WebCuda.
// Original calculations: Christopher Canavan, MIT; see LICENSE.
// This is the ONLY CUDA source file for the faithful port.
// GLSL vector operations are expressed in the CUDA subset supported by WebCuda.

__device__ float2 sb_vec2(float x,float y) { return make_float2(x,y); }
__device__ float2 sb_vec2(float x) { return make_float2(x,x); }
__device__ float2 sb_vec2(float2 x) { return x; }
__device__ float3 sb_vec3(float x,float y,float z) { return make_float3(x,y,z); }
__device__ float3 sb_vec3(float x) { return make_float3(x,x,x); }
__device__ float3 sb_vec3(float3 x) { return x; }
__device__ float4 sb_vec4(float x,float y,float z,float w) { return make_float4(x,y,z,w); }
__device__ float4 sb_vec4(float x) { return make_float4(x,x,x,x); }
__device__ float4 sb_vec4(float4 x) { return x; }
__device__ float3 sb_vec3(float2 a,float b){return make_float3(a.x,a.y,b);}
__device__ float3 sb_vec3(float a,float2 b){return make_float3(a,b.x,b.y);}
__device__ float4 sb_vec4(float2 a,float2 b){return make_float4(a.x,a.y,b.x,b.y);}
__device__ float4 sb_vec4(float3 a,float b){return make_float4(a.x,a.y,a.z,b);}
__device__ float4 sb_vec4(float2 a,float b,float c){return make_float4(a.x,a.y,b,c);}
__device__ float4 sb_vec4(float a,float b,float2 c){return make_float4(a,b,c.x,c.y);}
__device__ float4 sb_vec4(float a,float2 b,float c){return make_float4(a,b.x,b.y,c);}
__device__ float2 sb_xy(float4 a){return make_float2(a.x,a.y);}
__device__ float sb_sin(float x){return sinf(x);}
__device__ float2 sb_sin(float2 x){return make_float2(sb_sin(x.x),sb_sin(x.y));}
__device__ float3 sb_sin(float3 x){return make_float3(sb_sin(x.x),sb_sin(x.y),sb_sin(x.z));}
__device__ float4 sb_sin(float4 x){return make_float4(sb_sin(x.x),sb_sin(x.y),sb_sin(x.z),sb_sin(x.w));}
__device__ float sb_cos(float x){return cosf(x);}
__device__ float2 sb_cos(float2 x){return make_float2(sb_cos(x.x),sb_cos(x.y));}
__device__ float3 sb_cos(float3 x){return make_float3(sb_cos(x.x),sb_cos(x.y),sb_cos(x.z));}
__device__ float4 sb_cos(float4 x){return make_float4(sb_cos(x.x),sb_cos(x.y),sb_cos(x.z),sb_cos(x.w));}
__device__ float sb_exp(float x){return expf(x);}
__device__ float2 sb_exp(float2 x){return make_float2(sb_exp(x.x),sb_exp(x.y));}
__device__ float3 sb_exp(float3 x){return make_float3(sb_exp(x.x),sb_exp(x.y),sb_exp(x.z));}
__device__ float4 sb_exp(float4 x){return make_float4(sb_exp(x.x),sb_exp(x.y),sb_exp(x.z),sb_exp(x.w));}
__device__ float sb_log(float x){return logf(x);}
__device__ float2 sb_log(float2 x){return make_float2(sb_log(x.x),sb_log(x.y));}
__device__ float3 sb_log(float3 x){return make_float3(sb_log(x.x),sb_log(x.y),sb_log(x.z));}
__device__ float4 sb_log(float4 x){return make_float4(sb_log(x.x),sb_log(x.y),sb_log(x.z),sb_log(x.w));}
__device__ float sb_sqrt(float x){return sqrtf(x);}
__device__ float2 sb_sqrt(float2 x){return make_float2(sb_sqrt(x.x),sb_sqrt(x.y));}
__device__ float3 sb_sqrt(float3 x){return make_float3(sb_sqrt(x.x),sb_sqrt(x.y),sb_sqrt(x.z));}
__device__ float4 sb_sqrt(float4 x){return make_float4(sb_sqrt(x.x),sb_sqrt(x.y),sb_sqrt(x.z),sb_sqrt(x.w));}
__device__ float sb_abs(float x){return fabsf(x);}
__device__ float2 sb_abs(float2 x){return make_float2(sb_abs(x.x),sb_abs(x.y));}
__device__ float3 sb_abs(float3 x){return make_float3(sb_abs(x.x),sb_abs(x.y),sb_abs(x.z));}
__device__ float4 sb_abs(float4 x){return make_float4(sb_abs(x.x),sb_abs(x.y),sb_abs(x.z),sb_abs(x.w));}
__device__ float sb_floor(float x){return floorf(x);}
__device__ float2 sb_floor(float2 x){return make_float2(sb_floor(x.x),sb_floor(x.y));}
__device__ float3 sb_floor(float3 x){return make_float3(sb_floor(x.x),sb_floor(x.y),sb_floor(x.z));}
__device__ float4 sb_floor(float4 x){return make_float4(sb_floor(x.x),sb_floor(x.y),sb_floor(x.z),sb_floor(x.w));}
__device__ float sb_fract(float x){return x-floorf(x);}
__device__ float2 sb_fract(float2 x){return make_float2(sb_fract(x.x),sb_fract(x.y));}
__device__ float3 sb_fract(float3 x){return make_float3(sb_fract(x.x),sb_fract(x.y),sb_fract(x.z));}
__device__ float4 sb_fract(float4 x){return make_float4(sb_fract(x.x),sb_fract(x.y),sb_fract(x.z),sb_fract(x.w));}
__device__ float sb_sign(float x){return x>0.0f?1.0f:(x<0.0f?-1.0f:0.0f);}
__device__ float2 sb_sign(float2 x){return make_float2(sb_sign(x.x),sb_sign(x.y));}
__device__ float3 sb_sign(float3 x){return make_float3(sb_sign(x.x),sb_sign(x.y),sb_sign(x.z));}
__device__ float4 sb_sign(float4 x){return make_float4(sb_sign(x.x),sb_sign(x.y),sb_sign(x.z),sb_sign(x.w));}
__device__ float sb_tanh(float x){return (x<0.0f?-1.0f:1.0f)*(1.0f-expf(-2.0f*fabsf(x)))/(1.0f+expf(-2.0f*fabsf(x)));}
__device__ float2 sb_tanh(float2 x){return make_float2(sb_tanh(x.x),sb_tanh(x.y));}
__device__ float3 sb_tanh(float3 x){return make_float3(sb_tanh(x.x),sb_tanh(x.y),sb_tanh(x.z));}
__device__ float4 sb_tanh(float4 x){return make_float4(sb_tanh(x.x),sb_tanh(x.y),sb_tanh(x.z),sb_tanh(x.w));}
__device__ int sb_min(int a,int b){return min(a,b);}
__device__ int sb_max(int a,int b){return max(a,b);}
__device__ float sb_min(float a,float b){return fminf(a,b);}
__device__ float sb_max(float a,float b){return fmaxf(a,b);}
__device__ float sb_clamp(float x,float a,float b){return fminf(fmaxf(x,a),b);}
__device__ float sb_mix(float a,float b,float t){return a*(1.0f-t)+b*t;}
__device__ float sb_step(float edge,float x){return x<edge?0.0f:1.0f;}
__device__ float sb_smoothstep(float a,float b,float x){float t=sb_clamp((x-a)/(b-a),0.0f,1.0f);return t*t*(3.0f-2.0f*t);}
__device__ float sb_mod(float x,float y){return x-y*floorf(x/y);}
__device__ float2 sb_min(float2 a,float2 b) {return make_float2(sb_min(a.x,b.x),sb_min(a.y,b.y));}
__device__ float2 sb_min(float2 a,float b) {return make_float2(sb_min(a.x,b),sb_min(a.y,b));}
__device__ float2 sb_min(float a,float2 b) {return make_float2(sb_min(a,b.x),sb_min(a,b.y));}
__device__ float2 sb_max(float2 a,float2 b) {return make_float2(sb_max(a.x,b.x),sb_max(a.y,b.y));}
__device__ float2 sb_max(float2 a,float b) {return make_float2(sb_max(a.x,b),sb_max(a.y,b));}
__device__ float2 sb_max(float a,float2 b) {return make_float2(sb_max(a,b.x),sb_max(a,b.y));}
__device__ float2 sb_clamp(float2 a,float2 b,float2 c) {return make_float2(sb_clamp(a.x,b.x,c.x),sb_clamp(a.y,b.y,c.y));}
__device__ float2 sb_clamp(float2 a,float b,float c) {return make_float2(sb_clamp(a.x,b,c),sb_clamp(a.y,b,c));}
__device__ float2 sb_mix(float2 a,float2 b,float2 c) {return make_float2(sb_mix(a.x,b.x,c.x),sb_mix(a.y,b.y,c.y));}
__device__ float2 sb_mix(float2 a,float2 b,float c) {return make_float2(sb_mix(a.x,b.x,c),sb_mix(a.y,b.y,c));}
__device__ float2 sb_step(float2 a,float2 b) {return make_float2(sb_step(a.x,b.x),sb_step(a.y,b.y));}
__device__ float2 sb_step(float2 a,float b) {return make_float2(sb_step(a.x,b),sb_step(a.y,b));}
__device__ float2 sb_step(float a,float2 b) {return make_float2(sb_step(a,b.x),sb_step(a,b.y));}
__device__ float2 sb_smoothstep(float2 a,float2 b,float2 c) {return make_float2(sb_smoothstep(a.x,b.x,c.x),sb_smoothstep(a.y,b.y,c.y));}
__device__ float2 sb_smoothstep(float a,float b,float2 c) {return make_float2(sb_smoothstep(a,b,c.x),sb_smoothstep(a,b,c.y));}
__device__ float2 sb_mod(float2 a,float2 b) {return make_float2(sb_mod(a.x,b.x),sb_mod(a.y,b.y));}
__device__ float2 sb_mod(float2 a,float b) {return make_float2(sb_mod(a.x,b),sb_mod(a.y,b));}
__device__ float2 sb_mod(float a,float2 b) {return make_float2(sb_mod(a,b.x),sb_mod(a,b.y));}
__device__ float3 sb_min(float3 a,float3 b) {return make_float3(sb_min(a.x,b.x),sb_min(a.y,b.y),sb_min(a.z,b.z));}
__device__ float3 sb_min(float3 a,float b) {return make_float3(sb_min(a.x,b),sb_min(a.y,b),sb_min(a.z,b));}
__device__ float3 sb_min(float a,float3 b) {return make_float3(sb_min(a,b.x),sb_min(a,b.y),sb_min(a,b.z));}
__device__ float3 sb_max(float3 a,float3 b) {return make_float3(sb_max(a.x,b.x),sb_max(a.y,b.y),sb_max(a.z,b.z));}
__device__ float3 sb_max(float3 a,float b) {return make_float3(sb_max(a.x,b),sb_max(a.y,b),sb_max(a.z,b));}
__device__ float3 sb_max(float a,float3 b) {return make_float3(sb_max(a,b.x),sb_max(a,b.y),sb_max(a,b.z));}
__device__ float3 sb_clamp(float3 a,float3 b,float3 c) {return make_float3(sb_clamp(a.x,b.x,c.x),sb_clamp(a.y,b.y,c.y),sb_clamp(a.z,b.z,c.z));}
__device__ float3 sb_clamp(float3 a,float b,float c) {return make_float3(sb_clamp(a.x,b,c),sb_clamp(a.y,b,c),sb_clamp(a.z,b,c));}
__device__ float3 sb_mix(float3 a,float3 b,float3 c) {return make_float3(sb_mix(a.x,b.x,c.x),sb_mix(a.y,b.y,c.y),sb_mix(a.z,b.z,c.z));}
__device__ float3 sb_mix(float3 a,float3 b,float c) {return make_float3(sb_mix(a.x,b.x,c),sb_mix(a.y,b.y,c),sb_mix(a.z,b.z,c));}
__device__ float3 sb_step(float3 a,float3 b) {return make_float3(sb_step(a.x,b.x),sb_step(a.y,b.y),sb_step(a.z,b.z));}
__device__ float3 sb_step(float3 a,float b) {return make_float3(sb_step(a.x,b),sb_step(a.y,b),sb_step(a.z,b));}
__device__ float3 sb_step(float a,float3 b) {return make_float3(sb_step(a,b.x),sb_step(a,b.y),sb_step(a,b.z));}
__device__ float3 sb_smoothstep(float3 a,float3 b,float3 c) {return make_float3(sb_smoothstep(a.x,b.x,c.x),sb_smoothstep(a.y,b.y,c.y),sb_smoothstep(a.z,b.z,c.z));}
__device__ float3 sb_smoothstep(float a,float b,float3 c) {return make_float3(sb_smoothstep(a,b,c.x),sb_smoothstep(a,b,c.y),sb_smoothstep(a,b,c.z));}
__device__ float3 sb_mod(float3 a,float3 b) {return make_float3(sb_mod(a.x,b.x),sb_mod(a.y,b.y),sb_mod(a.z,b.z));}
__device__ float3 sb_mod(float3 a,float b) {return make_float3(sb_mod(a.x,b),sb_mod(a.y,b),sb_mod(a.z,b));}
__device__ float3 sb_mod(float a,float3 b) {return make_float3(sb_mod(a,b.x),sb_mod(a,b.y),sb_mod(a,b.z));}
__device__ float4 sb_min(float4 a,float4 b) {return make_float4(sb_min(a.x,b.x),sb_min(a.y,b.y),sb_min(a.z,b.z),sb_min(a.w,b.w));}
__device__ float4 sb_min(float4 a,float b) {return make_float4(sb_min(a.x,b),sb_min(a.y,b),sb_min(a.z,b),sb_min(a.w,b));}
__device__ float4 sb_min(float a,float4 b) {return make_float4(sb_min(a,b.x),sb_min(a,b.y),sb_min(a,b.z),sb_min(a,b.w));}
__device__ float4 sb_max(float4 a,float4 b) {return make_float4(sb_max(a.x,b.x),sb_max(a.y,b.y),sb_max(a.z,b.z),sb_max(a.w,b.w));}
__device__ float4 sb_max(float4 a,float b) {return make_float4(sb_max(a.x,b),sb_max(a.y,b),sb_max(a.z,b),sb_max(a.w,b));}
__device__ float4 sb_max(float a,float4 b) {return make_float4(sb_max(a,b.x),sb_max(a,b.y),sb_max(a,b.z),sb_max(a,b.w));}
__device__ float4 sb_clamp(float4 a,float4 b,float4 c) {return make_float4(sb_clamp(a.x,b.x,c.x),sb_clamp(a.y,b.y,c.y),sb_clamp(a.z,b.z,c.z),sb_clamp(a.w,b.w,c.w));}
__device__ float4 sb_clamp(float4 a,float b,float c) {return make_float4(sb_clamp(a.x,b,c),sb_clamp(a.y,b,c),sb_clamp(a.z,b,c),sb_clamp(a.w,b,c));}
__device__ float4 sb_mix(float4 a,float4 b,float4 c) {return make_float4(sb_mix(a.x,b.x,c.x),sb_mix(a.y,b.y,c.y),sb_mix(a.z,b.z,c.z),sb_mix(a.w,b.w,c.w));}
__device__ float4 sb_mix(float4 a,float4 b,float c) {return make_float4(sb_mix(a.x,b.x,c),sb_mix(a.y,b.y,c),sb_mix(a.z,b.z,c),sb_mix(a.w,b.w,c));}
__device__ float4 sb_step(float4 a,float4 b) {return make_float4(sb_step(a.x,b.x),sb_step(a.y,b.y),sb_step(a.z,b.z),sb_step(a.w,b.w));}
__device__ float4 sb_step(float4 a,float b) {return make_float4(sb_step(a.x,b),sb_step(a.y,b),sb_step(a.z,b),sb_step(a.w,b));}
__device__ float4 sb_step(float a,float4 b) {return make_float4(sb_step(a,b.x),sb_step(a,b.y),sb_step(a,b.z),sb_step(a,b.w));}
__device__ float4 sb_smoothstep(float4 a,float4 b,float4 c) {return make_float4(sb_smoothstep(a.x,b.x,c.x),sb_smoothstep(a.y,b.y,c.y),sb_smoothstep(a.z,b.z,c.z),sb_smoothstep(a.w,b.w,c.w));}
__device__ float4 sb_smoothstep(float a,float b,float4 c) {return make_float4(sb_smoothstep(a,b,c.x),sb_smoothstep(a,b,c.y),sb_smoothstep(a,b,c.z),sb_smoothstep(a,b,c.w));}
__device__ float4 sb_mod(float4 a,float4 b) {return make_float4(sb_mod(a.x,b.x),sb_mod(a.y,b.y),sb_mod(a.z,b.z),sb_mod(a.w,b.w));}
__device__ float4 sb_mod(float4 a,float b) {return make_float4(sb_mod(a.x,b),sb_mod(a.y,b),sb_mod(a.z,b),sb_mod(a.w,b));}
__device__ float4 sb_mod(float a,float4 b) {return make_float4(sb_mod(a,b.x),sb_mod(a,b.y),sb_mod(a,b.z),sb_mod(a,b.w));}
__device__ float4 sb_lookup(const float4* lookup,float2 uv,float lod){
 float px=sb_clamp(uv.x*4096.0f-0.5f,0.0f,4095.0f),py=sb_clamp(uv.y*4.0f-0.5f,0.0f,3.0f);
 int x=(int)floorf(px),y=(int)floorf(py),x1=min(x+1,4095),y1=min(y+1,3);
 float fx=px-(float)x,fy=py-(float)y;
 return sb_mix(sb_mix(lookup[y*4096+x],lookup[y*4096+x1],fx),sb_mix(lookup[y1*4096+x],lookup[y1*4096+x1],fx),fy);
}
struct SBUniforms {
  float uSwellMinTime;
  int uEvtCount;
  float uTime;
  float uInjMass;
  float uInjSpeed;
  float4 uInjJ;
  float4 uInjR;
  float4 uInjL;
  float4 uFoamSrc;
  float uFoamSpill;
  float2 uFoamSpillK;
};

// Direct port of glslDefines + NOISE + BREAKER (including SWELL_TRANSPORT).

#define G_ACC 9.81f
#define SWE_XMIN -4.8f
#define SWE_XMAX 4.8f
#define SWE_ZMIN -6.0f
#define SWE_ZMAX 12.404878048780489f
__device__ float softplus_(const float4* events,const float4* lookup,SBUniforms sb,float x) { return x > 20.0f ? x : sb_log(1.0f + sb_exp(x)); }
__device__ float bedProfile(const float4* events,const float4* lookup,SBUniforms sb,float z) {
  float y = 0.1f * z;
  y += 0.1f * 5.0f * softplus_(events,lookup,sb,(z - (-60.0f)) / 5.0f);
  y += 0.8f * 1.2f * softplus_(events,lookup,sb,(z - (-22.5f)) / 1.2f);
  y += -0.9f * 0.5f * softplus_(events,lookup,sb,(z - (-16.8f)) / 0.5f);
  y += 0.01999999999999999f * 2.0f * softplus_(events,lookup,sb,(z - (-12.0f)) / 2.0f);
  y += 0.22999999999999998f * 0.22f * softplus_(events,lookup,sb,(z - (-2.3f)) / 0.22f);
  y += -0.21499999999999997f * 0.18f * softplus_(events,lookup,sb,(z - (-0.6f)) / 0.18f);
  y += -0.05f * 1.5f * softplus_(events,lookup,sb,(z - (11.0f)) / 1.5f);
  y += -0.135f * 1.2f * softplus_(events,lookup,sb,(z - (19.0f)) / 1.2f);
  return sb_max(y + (-9.51869802609202f), -30.0f);
}

__device__ float hash11(const float4* events,const float4* lookup,SBUniforms sb,float p) { p = sb_fract(p * 0.1031f); p *= p + 33.33f; p *= p + p; return sb_fract(p); }
__device__ float hash12(const float4* events,const float4* lookup,SBUniforms sb,float2 p) { float3 p3 = sb_fract(sb_vec3(sb_sw_xyx(p)) * 0.1031f); p3 += dot(p3, sb_sw_yzx(p3) + 33.33f); return sb_fract((p3.x + p3.y) * p3.z); }
__device__ float2 hash22(const float4* events,const float4* lookup,SBUniforms sb,float2 p) { float3 p3 = sb_fract(sb_vec3(sb_sw_xyx(p)) * sb_vec3(0.1031f, 0.1030f, 0.0973f)); p3 += dot(p3, sb_sw_yzx(p3) + 33.33f); return sb_fract((sb_sw_xx(p3) + sb_sw_yz(p3)) * sb_sw_zy(p3)); }
__device__ float3 hash32(const float4* events,const float4* lookup,SBUniforms sb,float2 p) { float3 p3 = sb_fract(sb_vec3(sb_sw_xyx(p)) * sb_vec3(0.1031f, 0.1030f, 0.0973f)); p3 += dot(p3, sb_sw_yxz(p3) + 33.33f); return sb_fract((sb_sw_xxy(p3) + sb_sw_yzz(p3)) * sb_sw_zyx(p3)); }
__device__ float3 hash33(const float4* events,const float4* lookup,SBUniforms sb,float3 p3) { p3 = sb_fract(p3 * sb_vec3(0.1031f, 0.1030f, 0.0973f)); p3 += dot(p3, sb_sw_yxz(p3) + 33.33f); return sb_fract((sb_sw_xxy(p3) + sb_sw_yxx(p3)) * sb_sw_zyx(p3)); }

__device__ float vnoise(const float4* events,const float4* lookup,SBUniforms sb,float2 p) {
  float2 i = sb_floor(p), f = sb_fract(p);
  float2 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
  float a = hash12(events,lookup,sb,i), b = hash12(events,lookup,sb,i + sb_vec2(1, 0)), c = hash12(events,lookup,sb,i + sb_vec2(0, 1)), d = hash12(events,lookup,sb,i + sb_vec2(1, 1));
  return sb_mix(sb_mix(a, b, u.x), sb_mix(c, d, u.x), u.y);
}

__device__ float3 gnoised(const float4* events,const float4* lookup,SBUniforms sb,float2 p) {
  float2 i = sb_floor(p), f = sb_fract(p);
  float2 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
  float2 du = 30.0f * f * f * (f * (f - 2.0f) + 1.0f);
  float2 ga = hash22(events,lookup,sb,i) * 2.0f - 1.0f, gb = hash22(events,lookup,sb,i + sb_vec2(1, 0)) * 2.0f - 1.0f;
  float2 gc = hash22(events,lookup,sb,i + sb_vec2(0, 1)) * 2.0f - 1.0f, gd = hash22(events,lookup,sb,i + sb_vec2(1, 1)) * 2.0f - 1.0f;
  float va = dot(ga, f), vb = dot(gb, f - sb_vec2(1, 0)), vc = dot(gc, f - sb_vec2(0, 1)), vd = dot(gd, f - sb_vec2(1, 1));
  float v = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd);
  float2 d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
         + du * (sb_sw_yx(u) * (va - vb - vc + vd) + sb_vec2(vb, vc) - va);
  return sb_vec3(v * 1.6f, d * 1.6f);
}
__device__ float gnoise(const float4* events,const float4* lookup,SBUniforms sb,float2 p) { return gnoised(events,lookup,sb,p).x; }
__device__ float fbm(const float4* events,const float4* lookup,SBUniforms sb,float2 p, int oct) {
  float s = 0.0f, a = 0.5f;
  for (int i = 0; i < 8; i++) { if (i >= oct) break; s += a * gnoise(events,lookup,sb,p); p = sb_vec2(1.6f*p.x-1.2f*p.y,1.2f*p.x+1.6f*p.y); a *= 0.5f; }
  return s;
}

__device__ float vnoise3(const float4* events,const float4* lookup,SBUniforms sb,float3 p) {
  float3 i = sb_floor(p), f = sb_fract(p);
  float3 u = f * f * (3.0f - 2.0f * f);
  float n000 = hash33(events,lookup,sb,i).x, n100 = hash33(events,lookup,sb,i + sb_vec3(1,0,0)).x, n010 = hash33(events,lookup,sb,i + sb_vec3(0,1,0)).x, n110 = hash33(events,lookup,sb,i + sb_vec3(1,1,0)).x;
  float n001 = hash33(events,lookup,sb,i + sb_vec3(0,0,1)).x, n101 = hash33(events,lookup,sb,i + sb_vec3(1,0,1)).x, n011 = hash33(events,lookup,sb,i + sb_vec3(0,1,1)).x, n111 = hash33(events,lookup,sb,i + sb_vec3(1,1,1)).x;
  return sb_mix(sb_mix(sb_mix(n000, n100, u.x), sb_mix(n010, n110, u.x), u.y), sb_mix(sb_mix(n001, n101, u.x), sb_mix(n011, n111, u.x), u.y), u.z);
}

__device__ float3 worley(const float4* events,const float4* lookup,SBUniforms sb,float2 p) {
  float2 i = sb_floor(p), f = sb_fract(p);
  float f1 = 8.0f, f2 = 8.0f, id = 0.0f;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    float2 g = sb_vec2(x, y);
    float2 o = hash22(events,lookup,sb,i + g);
    float2 r = g + o - f;
    float d = dot(r, r);
    if (d < f1) { f2 = f1; f1 = d; id = hash12(events,lookup,sb,i + g); } else if (d < f2) { f2 = d; }
  }
  return sb_vec3(sb_sqrt(f1), sb_sqrt(f2), id);
}
__device__ float sat(const float4* events,const float4* lookup,SBUniforms sb,float x) { return sb_clamp(x, 0.0f, 1.0f); }
__device__ float sech2(const float4* events,const float4* lookup,SBUniforms sb,float x) { float e = sb_exp(-2.0f * sb_abs(x)); float s = 2.0f * sb_exp(-sb_abs(x)) / (1.0f + e); return s * s; }
__device__ float remap(const float4* events,const float4* lookup,SBUniforms sb,float v, float a, float b) { return sb_clamp((v - a) / (b - a), 0.0f, 1.0f); }
#ifndef OPT_EXPLORE
#define OPT_EXPLORE 1
#endif

#define MAX_EVENTS 6

#ifdef OPT_EXPLORE

#ifndef WATER_LOOKUP_DECLARED
#define WATER_LOOKUP_DECLARED 1

#endif

__device__ float4 swellTravel(const float4* events,const float4* lookup,SBUniforms sb,float z) {
  float u=sb_clamp((z+900.0f)/899.0f,0.0f,1.0f);
  return sb_lookup(lookup,sb_vec2((u*4095.0f+.5f)/4096.0f,.125f),0.0f);
}
__device__ float4 swellPosition(const float4* events,const float4* lookup,SBUniforms sb,float time) {
  float u=sb_clamp((time-sb.uSwellMinTime)/(-sb.uSwellMinTime),0.0f,1.0f);
  return sb_lookup(lookup,sb_vec2((u*4095.0f+.5f)/4096.0f,.375f),0.0f);
}
#endif

#define BRK_HREF 0.45f
#define BRK_TB (-0.42f)
#define BRK_TE (-0.33f)
#define BRK_TEND 0.85f

struct Brk {
  float ti;
  float tb;
  float Tj;
  float H;
  float c;
  float cFar;
  float tauS;
  float zI;
  float zB;
  float kap;
  float vy0;
  float Tr;
  float style;
  float spill;
  float str;
  float seed;
  float s;
  float rs;
  float x;
  float D;
  float hold;
};

struct BrkStage { float za; float ya; float yt; float lf; float uw; float hw; float m; float p; float st; float lb; };
__device__ BrkStage brkStage(const float4* events,const float4* lookup,SBUniforms sb,float tn) {
  float t0, iw;
  float4 cz, cy, l0, l1, q0, q1;
  if (tn < -4.000000f) { t0 = -6.000000f; iw = 0.5000000f; cz = sb_vec4(-23.00000f, 7.500000f, -0.3000000f, 0.3000000f); cy = sb_vec4(0.2000000f, 0.04000000f, -0.01333333f, 0.01333333f);
    l0 = sb_vec4(-0.1000000f, 8.000000f, 0.3000000f, 0.5000000f); l1 = sb_vec4(0.000000f, -1.000000f, 0.000000f, 0.000000f); q0 = sb_vec4(1.600000f, 2.000000f, 0.000000f, 4.000000f); q1 = sb_vec4(0.000000f, 0.000000f, 0.000000f, -0.2000000f); }
  else if (tn < -3.000000f) { t0 = -4.000000f; iw = 1.000000f; cz = sb_vec4(-15.50000f, 3.900000f, 0.5500000f, -0.2500000f); cy = sb_vec4(0.2400000f, 0.02666667f, 0.02166667f, -0.008333333f);
    l0 = sb_vec4(-0.1000000f, 7.000000f, 0.3000000f, 0.5000000f); l1 = sb_vec4(0.000000f, -1.000000f, 0.000000f, 0.000000f); q0 = sb_vec4(1.600000f, 2.000000f, 0.000000f, 3.800000f); q1 = sb_vec4(0.000000f, 0.000000f, 0.000000f, -0.2000000f); }
  else if (tn < -2.000000f) { t0 = -3.000000f; iw = 1.000000f; cz = sb_vec4(-11.30000f, 4.250000f, 0.02500000f, 0.02500000f); cy = sb_vec4(0.2800000f, 0.04500000f, -0.02125000f, 0.02625000f);
    l0 = sb_vec4(-0.1000000f, 6.000000f, 0.3000000f, 0.5000000f); l1 = sb_vec4(0.000000f, -2.000000f, 0.000000f, 0.000000f); q0 = sb_vec4(1.600000f, 2.000000f, 0.000000f, 3.600000f); q1 = sb_vec4(0.000000f, 0.000000f, 0.000000f, -0.4000000f); }
  else if (tn < -1.400000f) { t0 = -2.000000f; iw = 1.666667f; cz = sb_vec4(-7.000000f, 2.625000f, 0.1800000f, -0.1050000f); cy = sb_vec4(0.3300000f, 0.04875000f, 0.06450000f, -0.03325000f);
    l0 = sb_vec4(-0.1000000f, 4.000000f, 0.3000000f, 0.5000000f); l1 = sb_vec4(0.01000000f, -1.800000f, 0.000000f, 0.000000f); q0 = sb_vec4(1.600000f, 2.000000f, 0.000000f, 3.200000f); q1 = sb_vec4(0.000000f, 0.000000f, 0.000000f, -0.3000000f); }
  else if (tn < -1.000000f) { t0 = -1.400000f; iw = 2.500000f; cz = sb_vec4(-4.300000f, 1.780000f, 0.05666667f, -0.08666667f); cy = sb_vec4(0.4100000f, 0.05200000f, -0.01400000f, 0.01200000f);
    l0 = sb_vec4(-0.09000000f, 2.200000f, 0.3000000f, 0.5000000f); l1 = sb_vec4(0.01000000f, -0.8000000f, 0.000000f, 0.000000f); q0 = sb_vec4(1.600000f, 2.000000f, 0.000000f, 2.900000f); q1 = sb_vec4(0.2000000f, 0.000000f, 0.000000f, -0.2000000f); }
  else if (tn < -0.8000000f) { t0 = -1.000000f; iw = 5.000000f; cz = sb_vec4(-2.550000f, 0.8166667f, -0.1866667f, 0.07000000f); cy = sb_vec4(0.4600000f, 0.03000000f, 0.02088889f, -0.01088889f);
    l0 = sb_vec4(-0.08000000f, 1.400000f, 0.3000000f, 0.5000000f); l1 = sb_vec4(0.01200000f, -0.3600000f, 0.000000f, 0.000000f); q0 = sb_vec4(1.800000f, 2.000000f, 0.000000f, 2.700000f); q1 = sb_vec4(0.2000000f, 0.000000f, 0.000000f, -0.1000000f); }
  else if (tn < -0.5500000f) { t0 = -0.8000000f; iw = 4.000000f; cz = sb_vec4(-1.850000f, 0.8166667f, -0.08895833f, 0.04229167f); cy = sb_vec4(0.5000000f, 0.04888889f, 0.008722222f, -0.009611111f);
    l0 = sb_vec4(-0.06800000f, 1.040000f, 0.3000000f, 0.5000000f); l1 = sb_vec4(0.000000f, -0.4400000f, -0.04000000f, 0.1200000f); q0 = sb_vec4(2.000000f, 2.000000f, 0.000000f, 2.600000f); q1 = sb_vec4(0.6000000f, 0.000000f, 0.5500000f, -0.1000000f); }
  else if (tn < -0.4000000f) { t0 = -0.5500000f; iw = 6.666667f; cz = sb_vec4(-1.080000f, 0.4593750f, 0.05829545f, -0.06267045f); cy = sb_vec4(0.5480000f, 0.02250000f, -0.01240909f, 0.001909091f);
    l0 = sb_vec4(-0.06800000f, 0.6000000f, 0.2600000f, 0.6200000f); l1 = sb_vec4(0.000000f, -0.1400000f, -0.1400000f, 0.04000000f); q0 = sb_vec4(2.600000f, 2.000000f, 0.5500000f, 2.500000f); q1 = sb_vec4(0.6000000f, 1.000000f, 0.4500000f, -0.1000000f); }
  else if (tn < -0.3300000f) { t0 = -0.4000000f; iw = 14.28571f; cz = sb_vec4(-0.6250000f, 0.1810455f, -0.08386869f, 0.01682323f); cy = sb_vec4(0.5600000f, 0.001590909f, 0.005373737f, -0.01396465f);
    l0 = sb_vec4(-0.06800000f, 0.4600000f, 0.1200000f, 0.6600000f); l1 = sb_vec4(0.000000f, -0.04000000f, -0.08000000f, -0.1200000f); q0 = sb_vec4(3.200000f, 3.000000f, 1.000000f, 2.400000f); q1 = sb_vec4(0.4000000f, 0.000000f, 0.000000f, 0.000000f); }
  else if (tn < -0.2200000f) { t0 = -0.3300000f; iw = 9.090909f; cz = sb_vec4(-0.5110000f, 0.1002222f, -0.07087302f, 0.02065079f); cy = sb_vec4(0.5530000f, -0.04644444f, -0.02977778f, 0.007222222f);
    l0 = sb_vec4(-0.06800000f, 0.4200000f, 0.04000000f, 0.5400000f); l1 = sb_vec4(0.000000f, 0.000000f, -0.02000000f, 0.02000000f); q0 = sb_vec4(3.600000f, 3.000000f, 1.000000f, 2.400000f); q1 = sb_vec4(0.9000000f, 0.000000f, 0.000000f, 0.000000f); }
  else if (tn < -0.1200000f) { t0 = -0.2200000f; iw = 10.00000f; cz = sb_vec4(-0.4610000f, 0.01857143f, -0.06125397f, 0.03168254f); cy = sb_vec4(0.4840000f, -0.07666667f, -0.04600000f, 0.03066667f);
    l0 = sb_vec4(-0.06800000f, 0.4200000f, 0.02000000f, 0.5600000f); l1 = sb_vec4(0.000000f, -0.02000000f, 0.000000f, -0.03000000f); q0 = sb_vec4(4.500000f, 3.000000f, 1.000000f, 2.400000f); q1 = sb_vec4(0.5000000f, 0.000000f, 0.000000f, 0.000000f); }
  else if (tn < -0.04000000f) { t0 = -0.1200000f; iw = 12.50000f; cz = sb_vec4(-0.4720000f, -0.007111111f, 0.005222222f, -0.003111111f); cy = sb_vec4(0.3920000f, -0.06133333f, 0.04600000f, -0.03066667f);
    l0 = sb_vec4(-0.06800000f, 0.4000000f, 0.02000000f, 0.5300000f); l1 = sb_vec4(0.000000f, -0.02000000f, 0.000000f, -0.06000000f); q0 = sb_vec4(5.000000f, 3.000000f, 1.000000f, 2.400000f); q1 = sb_vec4(0.5000000f, 0.000000f, 0.000000f, 0.000000f); }
  else if (tn < 0.000000f) { t0 = -0.04000000f; iw = 25.00000f; cz = sb_vec4(-0.4770000f, -0.003000000f, -0.006500000f, 0.005500000f); cy = sb_vec4(0.3460000f, -0.03066667f, -0.04016667f, 0.02483333f);
    l0 = sb_vec4(-0.06800000f, 0.3800000f, 0.02000000f, 0.4700000f); l1 = sb_vec4(0.000000f, -0.02000000f, 0.000000f, -0.06000000f); q0 = sb_vec4(5.500000f, 3.000000f, 1.000000f, 2.400000f); q1 = sb_vec4(0.5000000f, 0.000000f, 0.000000f, 0.000000f); }
  else if (tn < 0.1200000f) { t0 = 0.000000f; iw = 8.333333f; cz = sb_vec4(-0.4810000f, 0.001500000f, 0.006600000f, -0.002100000f); cy = sb_vec4(0.3000000f, -0.1095000f, 0.007000000f, 0.002500000f);
    l0 = sb_vec4(-0.06800000f, 0.3600000f, 0.02000000f, 0.4100000f); l1 = sb_vec4(0.008000000f, 0.04000000f, 0.03000000f, -0.01000000f); q0 = sb_vec4(6.000000f, 3.000000f, 1.000000f, 2.400000f); q1 = sb_vec4(-2.000000f, -1.000000f, -0.2000000f, 0.000000f); }
  else if (tn < 0.3000000f) { t0 = 0.1200000f; iw = 5.555556f; cz = sb_vec4(-0.4750000f, 0.01260000f, 0.006675000f, -0.004275000f); cy = sb_vec4(0.2000000f, -0.1320000f, -0.02475000f, 0.03675000f);
    l0 = sb_vec4(-0.06000000f, 0.4000000f, 0.05000000f, 0.4000000f); l1 = sb_vec4(0.02000000f, 0.2000000f, 0.1500000f, 0.1000000f); q0 = sb_vec4(4.000000f, 2.000000f, 0.8000000f, 2.400000f); q1 = sb_vec4(-2.000000f, 0.000000f, -0.5000000f, 0.2000000f); }
  else if (tn < 0.6000000f) { t0 = 0.3000000f; iw = 3.333333f; cz = sb_vec4(-0.4600000f, 0.02187500f, -0.0001136364f, -0.001761364f); cy = sb_vec4(0.08000000f, -0.1187500f, 0.07113636f, -0.02238636f);
    l0 = sb_vec4(-0.04000000f, 0.6000000f, 0.2000000f, 0.5000000f); l1 = sb_vec4(0.02000000f, 0.4000000f, 0.1000000f, 0.000000f); q0 = sb_vec4(2.000000f, 2.000000f, 0.3000000f, 2.600000f); q1 = sb_vec4(0.000000f, 0.000000f, -0.3000000f, 0.4000000f); }
  else { t0 = 0.6000000f; iw = 4.000000f; cz = sb_vec4(-0.4400000f, 0.01363636f, -0.007272727f, 0.003636364f); cy = sb_vec4(0.01000000f, -0.03636364f, 0.05272727f, -0.02636364f);
    l0 = sb_vec4(-0.02000000f, 1.000000f, 0.3000000f, 0.5000000f); l1 = sb_vec4(0.02000000f, 0.2000000f, 0.000000f, 0.000000f); q0 = sb_vec4(2.000000f, 2.000000f, 0.000000f, 3.000000f); q1 = sb_vec4(0.000000f, 0.000000f, 0.000000f, 0.000000f); }
  float fr = sb_clamp((tn - t0) * iw, 0.0f, 1.0f);
  float4 fp = sb_vec4(1.0f, fr, fr * fr, fr * fr * fr);
  float4 l = l0 + l1 * fr, q = q0 + q1 * fr;
  BrkStage g;
  g.za = dot(cz, fp); g.ya = dot(cy, fp);
  g.yt = l.x; g.lf = l.y; g.uw = l.z; g.hw = l.w;
  g.m = q.x; g.p = q.y; g.st = q.z; g.lb = q.w;
  return g;
}

__device__ float brkEtaC(const float4* events,const float4* lookup,SBUniforms sb,float H) { return 1.229f * H; }
__device__ float brkEtaT(const float4* events,const float4* lookup,SBUniforms sb,float H) { return -0.151f * H; }

#define BRK_STAIR_L 0.45f
#define BRK_STAIR_K 0.25f
__device__ float brkStairX(const float4* events,const float4* lookup,SBUniforms sb,float x, float seed) {
  float s = x / BRK_STAIR_L + sb_fract(seed * 3.0f + 0.485f);
  return x + BRK_STAIR_K * BRK_STAIR_L * (sb_floor(s) + sb_smoothstep(0.45f, 1.0f, sb_fract(s)) - s + 0.175f);
}

__device__ float brkPeel(const float4* events,const float4* lookup,SBUniforms sb,float4 D, float4 F, float x0, float x) {
  float d = x - x0;
  float ds = sb_sqrt(d * d + 0.25f) - 0.5f;
  float a1 = d >= 0.0f ? D.x : D.z, a2 = d >= 0.0f ? D.y : D.w;
  if (a2 < 0.0f) ds = sb_min(ds, -a1 / (2.0f * a2));
  return a1 * ds + a2 * ds * ds + F.x / (1.0f + sb_exp(-(x - F.y) / sb_max(F.z, 1e-3f)));
}

#define BRK_XCLAMP 5.0f
#define BRK_XSAT 3.0f
__device__ float brkClampX(const float4* events,const float4* lookup,SBUniforms sb,float x) {
  float e = sb_max(sb_abs(x) - BRK_XCLAMP, 0.0f);
  return x - sb_sign(x) * (e - BRK_XSAT * e / (e + BRK_XSAT));
}

__device__ float2 brkFarSD(const float4* events,const float4* lookup,SBUniforms sb,float x, float ph) {
  float3 a = sb_vec3(0.185f, 0.33f, 0.57f) * x + sb_vec3(6.2831853f * ph, 17.1f * ph + 1.3f, 41.7f * ph + 2.9f);
  return sb_vec2(dot(sb_vec3(0.55f, 0.30f, 0.15f), sb_sin(a)), dot(sb_vec3(0.10175f, 0.099f, 0.0855f), sb_cos(a)));
}

__device__ float brkFarW(const float4* events,const float4* lookup,SBUniforms sb,float far, float x) { return sb_mix(sb_smoothstep(5.0f, 13.0f, sb_abs(x)), 1.0f, sb_step(1e-9f, far)) * sb_step(1e-9f, sb_abs(far)); }
#define BRK_FAR_STAND 0.25f
#define BRK_FAR_H 0.12f
#define BRK_FAR_Z 0.25f
#define BRK_TILT_L 12.0f
#define BRK_HOLD 0.5f
__device__ Brk brkAt(const float4* events,const float4* lookup,SBUniforms sb,int i, float x) {
  float4 A = events[0*MAX_EVENTS+(i)], B = events[1*MAX_EVENTS+(i)], C = events[2*MAX_EVENTS+(i)], D = events[3*MAX_EVENTS+(i)], E = events[4*MAX_EVENTS+(i)], F = events[5*MAX_EVENTS+(i)], G = events[6*MAX_EVENTS+(i)];
  Brk b;
  float sd = A.w;
  b.x = x;
  float xs = brkStairX(events,lookup,sb,x, sd);

  float useL = 1.0f - sb_step(1e-9f, G.z);
  float xc = brkClampX(events,lookup,sb,x);
  float peelS = useL * brkPeel(events,lookup,sb,D, F, B.y, xc);
  float peel = useL * brkPeel(events,lookup,sb,D, F, B.y, brkClampX(events,lookup,sb,xs));
  float lg = useL * F.x / (1.0f + sb_exp(-(xc - F.y) / sb_max(F.z, 1e-3f)));
  float stand = F.w * peelS;

  float Af = sb_abs(G.z);
  float wS = brkFarW(events,lookup,sb,G.z, x);
  float2 fm = brkFarSD(events,lookup,sb,x, G.w);
  peelS = sb_mix(peelS, Af * fm.x, wS);
  peel = sb_mix(peel, Af * (fm.x + fm.y * (xs - x)), wS);
  stand = sb_mix(stand, BRK_FAR_STAND * Af * (fm.x + 1.0f), wS);
  lg *= 1.0f - wS;
  float fS = fm.x * wS;

  b.D = sb_max(sb_max(E.y, 0.05f) + stand + (peel - peelS), 0.0f);
  float wob = 0.62f * sb_sin(x * 2.3f + sd * 6.1f) + 0.38f * sb_sin(x * 5.1f + sd * 11.7f);
  b.ti = A.x + peel + B.z * wob;
  b.H = A.y * (1.0f + C.z * (0.65f * sb_sin(x * 1.1f + sd * 3.3f) + 0.35f * sb_sin(x * 2.9f + sd * 8.9f)));

  b.H = b.H * ((1.0f + E.x * (sb_abs(F.x) > 1e-4f ? lg / F.x : 0.0f)) * (1.0f - BRK_FAR_H * fS));
  b.s = b.H / BRK_HREF;
  b.rs = sb_sqrt(b.s);
  b.c = B.x;
  b.cFar = E.x;
  b.tauS = E.y;
  b.spill = E.z;
  b.kap = C.x;
  b.style = C.w;
  b.str = B.w;
  b.seed = sd;
  b.Tj = -BRK_TE * b.rs;
  b.tb = b.ti - b.Tj;
  b.Tr = 0.3f * b.rs;
  b.vy0 = -0.10f * b.rs;
  b.zI = A.z + E.w * BRK_TILT_L * sb_sin(x / BRK_TILT_L) + BRK_FAR_Z * fS + C.y * (0.6f * sb_sin(x * 1.37f + sd * 4.7f) + 0.4f * sb_sin(x * 3.1f + sd * 2.3f));
  b.zB = b.zI + -0.511f * b.s;
  b.hold = BRK_HOLD * (1.0f - sb_smoothstep(0.3f, 0.5f, b.style));
  return b;
}

__device__ float brkTnLip(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float t) {
  float Dn = b.D / b.rs;
  float a = (t - b.ti) / b.rs + Dn;
  float Ts = -0.47f - 0.6f * Dn;
  return a - Dn * sb_smoothstep(Ts, Ts + sb_max(1.6f * Dn, 1e-3f), a);
}

#define BRK_HOLD_R 0.15f

__device__ float brkTn(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float t) {
  float tn = brkTnLip(events,lookup,sb,b, t);
  float a = 0.6f * b.hold, w = sb_max(0.8f * b.hold, 1e-4f);
  float u = sb_clamp((tn - a) / w, 0.0f, 1.0f);

  float I = w * u * u * u * (1.0f - 0.5f * u) + sb_max(tn - a - w, 0.0f);
  return sb_mix(tn, BRK_HOLD_R * tn + (1.0f - BRK_HOLD_R) * I, sb_step(0.0f, tn) * sb_step(1e-9f, b.hold));
}

__device__ float2 brkApex(const float4* events,const float4* lookup,SBUniforms sb,Brk b, BrkStage g, float tn) {
  float k = sb_mix(b.s, b.rs, sb_smoothstep(-0.8f, -2.0f, tn));
  return sb_vec2(b.zI + g.za * k, g.ya * b.s);
}

__device__ void brkTransport(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float tn, BrkStage& g, float2& A) {
  g=brkStage(events,lookup,sb,sb_max(tn,-3.0f)); A=brkApex(events,lookup,sb,b,g,sb_max(tn,-3.0f));
#ifdef OPT_EXPLORE
  if(tn < -3.0f) {
    float4 anchor=swellTravel(events,lookup,sb,A.x);
    float4 offshore=swellPosition(events,lookup,sb,anchor.x+(tn+3.0f)*b.rs);
    float stretch=sb_clamp(offshore.y/anchor.y,.75f,2.4f);
    float shoal=sb_clamp(sb_sqrt(anchor.z/sb_max(offshore.z,.1f)),.62f,1.25f);
    A.x=offshore.w; g.lf = g.lf * (stretch); g.lb = g.lb * (stretch);
    g.ya = g.ya * (shoal); g.yt = g.yt * (shoal);
  }
#endif
}

__device__ float brkCrestZ(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float t) {
  float tn = brkTn(events,lookup,sb,b, t);
  return brkApex(events,lookup,sb,b, brkStage(events,lookup,sb,tn), tn).x;
}
__device__ float brkCrestSpeed(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float t) { return (brkCrestZ(events,lookup,sb,b, t + 0.01f) - brkCrestZ(events,lookup,sb,b, t - 0.01f)) / 0.02f; }

__device__ float brkHeight(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float t) {
  BrkStage g = brkStage(events,lookup,sb,brkTn(events,lookup,sb,b, t));
  return (g.ya - g.yt) * b.s * (BRK_HREF / 0.621f);
}
__device__ float brkSteep(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float t) { return brkStage(events,lookup,sb,brkTn(events,lookup,sb,b, t)).st; }

__device__ float2 brkFcos(const float4* events,const float4* lookup,SBUniforms sb,float u) { u = sb_clamp(u, 0.0f, 1.0f); return sb_vec2(0.5f + 0.5f * sb_cos(3.14159265f * u), -1.5707963f * sb_sin(3.14159265f * u)); }
__device__ float2 brkFsteep(const float4* events,const float4* lookup,SBUniforms sb,float u, float uw, float hw, float m, float p) {
  u = sb_clamp(u, 0.0f, 1.0f);
  if (u < uw) {
    float r = u / uw;
    float rp = powf(r, p);
    float q = sb_max(1.0f - rp, 1e-5f);
    float v = hw + (1.0f - hw) * powf(q, 1.0f / p);
    float dv = -(1.0f - hw) * powf(r, p - 1.0f) * powf(q, 1.0f / p - 1.0f) / uw;
    return sb_vec2(v, sb_max(dv, -60.0f));
  }
  float w = sb_max((1.0f - u) / (1.0f - uw), 0.0f);
  return sb_vec2(hw * powf(w, m), -hw * m * powf(w, sb_max(m - 1.0f, 0.0f)) / (1.0f - uw));
}

__device__ float2 brkShape(const float4* events,const float4* lookup,SBUniforms sb,BrkStage g, float xi) {
  if (xi < 0.0f) {
    float a = xi / g.lb;
    float s2 = sech2(events,lookup,sb,a);
    return sb_vec2(g.ya * s2, g.ya * (-2.0f / g.lb) * sb_tanh(a) * s2);
  }
  if (xi > g.lf) {
    const float La = 1.5f;
    float e = (xi - g.lf) / La;
    float ex = sb_exp(-e * e);
    return sb_vec2(g.yt * ex, g.yt * ex * (-2.0f * e / La));
  }
  float u = xi / g.lf;
  float2 fc = brkFcos(events,lookup,sb,u), fs = brkFsteep(events,lookup,sb,u, g.uw, g.hw, g.m, g.p);
  float2 F = sb_mix(fc, fs, g.st);
  return sb_vec2(g.yt + (g.ya - g.yt) * F.x, (g.ya - g.yt) * F.y / g.lf);
}

__device__ float brkFrontXi(const float4* events,const float4* lookup,SBUniforms sb,BrkStage g, float h) {
  float lo = 0.0f, hi = 1.0f;
  int nIt = 12 + sb_min(sb.uEvtCount, 0);
  for (int k = 0; k < nIt; k++) {
    float u = 0.5f * (lo + hi);
    float F = sb_mix(brkFcos(events,lookup,sb,u).x, brkFsteep(events,lookup,sb,u, g.uw, g.hw, g.m, g.p).x, g.st);
    if (F > h) lo = u; else hi = u;
  }
  return 0.5f * (lo + hi) * g.lf;
}

__device__ float3 brkProfileS(const float4* events,const float4* lookup,SBUniforms sb,Brk b, BrkStage g, float2 A, float tn, float z) {
  if (b.str <= 0.0f || tn < -6.0f || tn > BRK_TEND) return sb_vec3(0.0f);
  float2 e = brkShape(events,lookup,sb,g, (z - A.x) / b.s);

  float birth=sb_smoothstep(-18.0f,-11.0f,A.x);
#ifdef OPT_EXPLORE
  birth=1.0f;
#endif
  float amp = birth * (1.0f - sb_smoothstep(0.6f, BRK_TEND, tn)) * b.str;
  float fade = 1.0f - sb_smoothstep(-1.0f, 0.1f, z);
  float k = amp * fade;

  float xi = (z - A.x) / b.s;

  float crestW = xi < 0.0f ? sb_exp(-powf(xi / 0.12f, 2.0f))
               : sb_exp(-powf((g.ya - e.x) / (0.025f + 0.1f * b.spill), 2.0f)) * (1.0f - sb_smoothstep(BRK_TB - 0.06f, BRK_TB, tn) * (1.0f - sb_smoothstep(0.0f, 0.1f, tn)));
  return sb_vec3(e.x * b.s * k, e.y * k, crestW * k);
}
__device__ float3 brkProfile(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float z, float t) {
  float tn = brkTn(events,lookup,sb,b, t);
  float3 result = sb_vec3(0.0f);
#ifdef OPT_EXPLORE
  if(tn < -3.0f) {
    BrkStage g = brkStage(events,lookup,sb,-3.0f); float2 A = sb_vec2(0.0f); brkTransport(events,lookup,sb,b,tn,g,A);
    float2 e=brkShape(events,lookup,sb,g,(z-A.x)/b.s);
    result = sb_vec3(e.x*b.s*b.str,e.y*b.str,0.0f);
  } else
#endif
  {
  BrkStage g = brkStage(events,lookup,sb,tn);
  result = brkProfileS(events,lookup,sb,b, g, brkApex(events,lookup,sb,b, g, tn), tn, z);
  }
  return result;
}

__device__ float brkArrival(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float t) { return (t - b.ti + b.D) / b.rs; }

__device__ float brkWhitecapOnset(const float4* events,const float4* lookup,SBUniforms sb,Brk b) { return sb_mix(-0.85f, -0.30f, sb_smoothstep(0.3f, 0.9f, b.spill)) - 0.45f * (1.0f - sb_smoothstep(0.35f, 0.6f, b.s)); }
__device__ float brkWhitecap(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float t) {
  float patchy = sb_smoothstep(0.15f, 0.75f, 0.5f + 0.5f * sb_sin(b.x * 3.3f + b.seed * 17.0f) * sb_sin(b.x * 1.7f + b.seed * 5.0f + 1.0f));
  float spill = sb_smoothstep(0.3f, 0.9f, b.spill);
  float a = brkArrival(events,lookup,sb,b, t) - brkWhitecapOnset(events,lookup,sb,b);

  return sb_mix(sb_smoothstep(0.0f, 0.4f, a) * 0.45f * patchy, sb_smoothstep(0.0f, 0.12f, a), spill);
}

__device__ float brkCrestFoam(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float t, float crestW) {
  float tn = brkTn(events,lookup,sb,b, t);

  float post = 0.08f * sb_smoothstep(0.02f, 0.12f, tn) * (1.0f - sb_smoothstep(0.15f, 0.35f, tn));
  return crestW * post;
}

__device__ float brkSurface(const float4* events,const float4* lookup,SBUniforms sb,float2 xz, float t, float2& d, float& crestFoam) {
  float eta = 0.0f; d = sb_vec2(0.0f); crestFoam = 0.0f;
  for (int i = 0; i < sb.uEvtCount; i++) {
    Brk b = brkAt(events,lookup,sb,i, xz.x);
    float3 p = brkProfile(events,lookup,sb,b, xz.y, t);
    eta += p.x;
    d.y = d.y + (p.y);
    if (p.z > 0.01f) crestFoam = sb_max(crestFoam, brkCrestFoam(events,lookup,sb,b, t, p.z));
  }
  return eta;
}

__device__ float brkCrestWidth(const float4* events,const float4* lookup,SBUniforms sb,float2 xz, float y, float t) {
  float best = -1.0f;
  for (int i = 0; i < sb.uEvtCount; i++) {
    Brk b = brkAt(events,lookup,sb,i, xz.x);
    if (b.str <= 0.0f) continue;
    float tn = brkTn(events,lookup,sb,b, t);
    if (tn < -2.0f || tn > 0.3f) continue;
    BrkStage g = brkStage(events,lookup,sb,tn);
    float2 A = brkApex(events,lookup,sb,b, g, tn);
    float amp = sb_smoothstep(-18.0f, -11.0f, A.x) * b.str;
    float ya = g.ya * b.s * amp, yt = g.yt * b.s * amp;
    float h = (y - yt) / sb_max(ya - yt, 1e-3f);
    if (h < 0.35f || h > 1.02f) continue;
    float xi = (xz.y - A.x) / b.s;
    if (xi < -1.5f || xi > g.lf) continue;

    float r = sb_sqrt(sb_max(ya, 1e-3f) / sb_max(y, 1e-3f));
    float xb = y > 0.0f ? -g.lb * sb_log(r + sb_sqrt(sb_max(r * r - 1.0f, 0.0f))) : -3.0f;
    float xf = brkFrontXi(events,lookup,sb,g, sb_clamp(h, 0.0f, 1.0f));
    float w = sb_max(xf - xb, 0.0f) * b.s;
    best = best < 0.0f ? w : sb_min(best, w);
  }
  return best;
}
__device__ float brkSurfaceOnly(const float4* events,const float4* lookup,SBUniforms sb,float2 xz, float t) {
  float eta = 0.0f;
  for (int i = 0; i < sb.uEvtCount; i++) {
    eta += brkProfile(events,lookup,sb,brkAt(events,lookup,sb,i, xz.x), xz.y, t).x;
  }
  return eta;
}

struct BrkLip {
  float2 A;
  float2 R;
  float2 T;
  float2 P1; float2 P2;
  float2 vT;
  float wR;
  float wT;
  float fl;
  float grow;
  float tn;
  BrkStage g;
};
__device__ float2 brkLipV0(const float4* events,const float4* lookup,SBUniforms sb,Brk b) { return sb_vec2(1.05f * (1.0f + b.kap) * sb_mix(0.55f, 1.0f, b.style), -0.10f) * b.rs; }
__device__ BrkLip brkLip(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float t) {
  BrkLip L;
  float tn = sb_max(brkTnLip(events,lookup,sb,b, t), BRK_TB);
  L.tn = tn;
  BrkStage g = brkStage(events,lookup,sb,tn);
  L.g = g;
  L.A = brkApex(events,lookup,sb,b, g, tn);
  L.grow = sb_smoothstep(BRK_TB, BRK_TB + 0.06f, tn);
  L.wR = 0.075f * b.s;

  L.R = L.A + sb_vec2(sb_mix(0.02f, 0.10f, sb_smoothstep(BRK_TB, BRK_TE + 0.05f, tn)) * b.s, -0.5f * L.wR);
  float2 rel = sb_vec2(0.17f, -0.043f) * b.s;
  if (tn < BRK_TE) {
    float f = sb_smoothstep(BRK_TB, BRK_TE, tn);
    L.T = L.A + sb_mix(sb_vec2(0.03f, -0.06f) * b.s, rel, f);
    L.vT = sb_vec2(1.5f, -0.3f) * b.rs;
    L.fl = 0.0f;
  } else {
    BrkStage gE = brkStage(events,lookup,sb,BRK_TE);
    float2 Te = brkApex(events,lookup,sb,b, gE, BRK_TE) + rel;
    float2 V0 = brkLipV0(events,lookup,sb,b);
    L.fl = (tn - BRK_TE) * b.rs;
    float fb = sb_min(L.fl, b.Tj);
    L.T = Te + V0 * fb + sb_vec2(0.0f, -0.5f * G_ACC * fb * fb);
    L.vT = V0 + sb_vec2(0.0f, -G_ACC * fb);
    if (L.fl > b.Tj) {

      float a = L.fl - b.Tj;
      L.T = L.T + (sb_vec2(V0.x * a * 0.6f, -0.25f * b.s * sb_smoothstep(0.0f, 0.25f * b.rs, a)));
    }
  }
  float2 ch = L.T - L.R;
  float len = sb_max(length(ch), 1e-4f);
  float2 dT = normalize(ch / len + normalize(L.vT));
  L.P1 = L.R + sb_vec2(0.45f * sb_max(ch.x, 0.02f * b.s) + 0.1f * len, 0.0f);
  L.P2 = L.T - dT * 0.35f * len;
  L.wT = 0.035f * b.s * (1.0f - 0.5f * sb_min(L.fl / 0.3f, 1.0f)) * L.grow;
  return L;
}

__device__ float4 brkLipCurve(const float4* events,const float4* lookup,SBUniforms sb,BrkLip L, float sg) {
  float u = 1.0f - sg;
  float2 p = u * u * u * L.R + 3.0f * u * u * sg * L.P1 + 3.0f * u * sg * sg * L.P2 + sg * sg * sg * L.T;
  float2 dp = 3.0f * u * u * (L.P1 - L.R) + 6.0f * u * sg * (L.P2 - L.P1) + 3.0f * sg * sg * (L.T - L.P2);
  float l = length(dp);
  return sb_vec4(p, l > 1e-6f ? dp / l : sb_vec2(1.0f, 0.0f));
}
__device__ float brkLipWidth(const float4* events,const float4* lookup,SBUniforms sb,BrkLip L, float sg) { return L.wT + (L.wR - L.wT) * powf(1.0f - sb_clamp(sg, 0.0f, 1.0f), 0.8f); }

__device__ float3 brkJet(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float sigma, float t) {
  BrkLip L = brkLip(events,lookup,sb,b, t);
  float sg = 1.0f - sb_clamp(sigma, 0.0f, 1.0f);
  float4 c = brkLipCurve(events,lookup,sb,L, sg);
  return sb_vec3(sb_sw_xy(c), brkLipWidth(events,lookup,sb,L, sg));
}

__device__ float brkImpactSpeed(const float4* events,const float4* lookup,SBUniforms sb,Brk b) {
  float2 v = brkLipV0(events,lookup,sb,b) + sb_vec2(0.0f, -G_ACC * b.Tj);
  return length(v);
}

__device__ float brkGamma(const float4* events,const float4* lookup,SBUniforms sb,float tau, float T) { return tau > 0.0f ? tau / (T * T) * sb_exp(-tau / T) : 0.0f; }

__device__ float brkThrow(const float4* events,const float4* lookup,SBUniforms sb,Brk b) { return sb_mix(0.3f, 1.0f, sb_smoothstep(0.4f, 0.75f, b.style)); }

__device__ float brkLobes(const float4* events,const float4* lookup,SBUniforms sb,Brk b) {
  float x = b.x;
  float far = sb_smoothstep(4.0f, 8.0f, sb_abs(x));
  float ph = far * 2.5f * sb_sin(0.37f * x + b.seed * 5.1f + 1.7f * sb_sin(0.13f * x));
  float am = sb_mix(1.0f, 0.45f + 0.9f * (0.5f + 0.5f * sb_sin(0.29f * x + b.seed * 3.3f) * sb_sin(0.71f * x + 1.1f)), far);

  float wm = 1.1f * sb_sin(1.9f * x + b.seed * 13.7f) + 0.6f * sb_sin(4.4f * x + b.seed * 29.3f);
  float fine = 0.6f * sb_sin(24.1f * x + b.seed * 53.3f + 2.1f * sb_sin(3.7f * x + b.seed * 9.0f)) + 0.4f * sb_sin(37.9f * x + b.seed * 71.9f + 1.7f * sb_sin(5.9f * x + b.seed * 4.0f));
  return am * (sb.uInjL.x * sb_sin(5.236f * x + b.seed * 37.1f + ph) + sb.uInjL.y * sb_sin(14.96f * x + b.seed * 91.7f + wm + 1.9f * ph)
       + sb.uInjL.z * fine);
}

__device__ float brkRunupMod(const float4* events,const float4* lookup,SBUniforms sb,float x, float seed) {
  float far = sb_smoothstep(4.0f, 8.0f, sb_abs(x));
  float und = 0.6f * sb_sin(0.21f * x + seed * 7.3f + 0.5f * sb_sin(0.09f * x + seed * 3.1f)) + 0.4f * sb_sin(0.47f * x + seed * 13.1f);
  float cph = x / 17.0f + 0.25f * sb_sin(x / 23.0f + 1.1f);
  float horn = 1.0f - sb_abs(sb_sin(3.14159f * cph));
  return far * (0.14f * und + 0.13f * (1.0f - 2.0f * horn * horn));
}
__device__ float3 brkInjection(const float4* events,const float4* lookup,SBUniforms sb,float2 xz, float t) {
  float3 src = sb_vec3(0.0f);
  for (int i = 0; i < sb.uEvtCount; i++) {
    Brk b = brkAt(events,lookup,sb,i, xz.x);
    if (b.str <= 0.0f) continue;
    float tau = t - b.ti;
    if (tau < -0.1f || tau > 3.0f) continue;
    float hs = sb_sqrt(b.H / 0.45f);

    float M = sb.uInjMass * 0.2025f * powf(b.H / 0.45f, 1.5f) * b.str * sb_max(events[6*MAX_EVENTS+(i)].x, 0.0f);

    float tJ = tau + 0.04f;
    float zJ = b.zI + 0.10f + sb.uInjJ.z * brkThrow(events,lookup,sb,b) * tJ;
    float sJ = 0.22f * hs + 0.20f * tJ;
    float qJ = M * sb.uInjJ.x * brkGamma(events,lookup,sb,tJ, sb.uInjJ.y) * sb_exp(-0.5f * powf((xz.y - zJ) / sJ, 2.0f)) / (2.5066f * sJ);

    float tR = tau - sb.uInjR.x;
    float zR = sb_mix(brkCrestZ(events,lookup,sb,b, b.ti), b.zI, 0.5f) + sb.uInjR.z * sb_max(tR, 0.0f);
    float sR = 0.30f * hs + 0.15f * sb_max(tR, 0.0f);
    float qR = M * (1.0f - sb.uInjJ.x) * brkGamma(events,lookup,sb,tR, sb.uInjR.y) * sb_exp(-0.5f * powf((xz.y - zR) / sR, 2.0f)) / (2.5066f * sR);

    float thr = brkThrow(events,lookup,sb,b);
    float mz = sb.uInjSpeed * (qJ * sb.uInjJ.w * sb_mix(0.5f, 1.0f, thr) + qR * sb.uInjR.w * sb_mix(0.6f, 1.0f, thr))
             * (1.0f + brkLobes(events,lookup,sb,b)) * (1.0f + 0.5f * brkRunupMod(events,lookup,sb,b.x, b.seed));
    src += sb_vec3(qJ + qR, mz, mz * sb.uInjL.w);
  }
  return src;
}

__device__ float brkFoam(const float4* events,const float4* lookup,SBUniforms sb,float2 xz, float t) {
  float f = 0.0f;
  for (int i = 0; i < sb.uEvtCount; i++) {
    Brk b = brkAt(events,lookup,sb,i, xz.x);
    if (b.str <= 0.0f) continue;
    float tau = t - b.ti;

    float spl = sb_smoothstep(0.5f, 0.9f, b.spill) * (1.0f - sb_smoothstep(0.1f, 0.3f, b.style)) * b.str;

    if (spl > 0.0f && tau > -0.3f && tau < 2.4f) {

      float on = sb_smoothstep(-0.3f, 0.0f, tau) * (1.0f - sb_smoothstep(sb.uFoamSpillK.y - 1.1f, sb.uFoamSpillK.y, tau));
      float zc = b.zI + 0.1f + sb.uFoamSpillK.x * sb_max(tau, 0.0f);
      f += sb.uFoamSpill * spl * (b.H / 0.45f) * on * sb_exp(-0.5f * powf((xz.y - zc) / (0.3f + 0.2f * sb_max(tau, 0.0f)), 2.0f));
    }
    if (tau < -0.1f || tau > 3.0f) continue;
    float E = (b.H * b.H) / (0.45f * 0.45f) * b.str * sb_mix(0.6f, 1.0f, b.style);
    float2 gain = sb_max(sb_xy(events[6*MAX_EVENTS+(i)]), sb_vec2(0.0f));
    float tJ = tau + 0.04f;
    float zJ = b.zI + 0.10f + sb.uInjJ.z * brkThrow(events,lookup,sb,b) * tJ;
    float sJ = sb.uFoamSrc.w * (0.25f + 0.30f * sb_max(tJ, 0.0f));
    f += E * gain.y * sb.uFoamSrc.x * brkGamma(events,lookup,sb,tJ, sb.uFoamSrc.y) * sb_exp(-0.5f * powf((xz.y - zJ) / sJ, 2.0f));

    float tR = tau - 0.21f;
    float zR = sb_mix(brkCrestZ(events,lookup,sb,b, b.ti), b.zI, 0.5f) + 0.55f * sb_max(tR, 0.0f);
    float sR = 0.35f + 0.2f * sb_max(tR, 0.0f);
    f += E * gain.x * sb.uFoamSrc.z * brkGamma(events,lookup,sb,tR, 0.4f) * sb_exp(-0.5f * powf((xz.y - zR) / sR, 2.0f));
  }
  return f;
}

__device__ float brkSweFoamKeep(const float4* events,const float4* lookup,SBUniforms sb,float2 xz, float t) {
  float keep = 1.0f;
  if (xz.y < -9.0f || xz.y > 0.6f) return keep;
  float fade = 1.0f - sb_smoothstep(-1.0f, 0.1f, xz.y);
  for (int i = 0; i < sb.uEvtCount; i++) {
    Brk b = brkAt(events,lookup,sb,i, xz.x);
    if (b.str <= 0.0f) continue;
    float tn = brkTn(events,lookup,sb,b, t);
    if (tn < -2.4f || tn > 0.5f) continue;
    BrkStage g = brkStage(events,lookup,sb,tn);
    float2 A = brkApex(events,lookup,sb,b, g, tn);
    float xi = (xz.y - A.x) / b.s;
    if (xi < -0.8f || xi > g.lf) continue;
    float amp = sb_smoothstep(-23.0f, -13.0f, A.x) * (1.0f - sb_smoothstep(0.6f, BRK_TEND, tn)) * b.str * fade;
    float eta = brkShape(events,lookup,sb,g, xi).x * b.s * amp;
    float raised = sb_smoothstep(0.012f, 0.06f, eta - 0.5f * g.yt * b.s * amp);
    float front = sb_smoothstep(-0.8f, -0.2f, xi);
    float grown = sb_smoothstep(-2.4f, -1.5f, tn);
    float live = 1.0f - sb_smoothstep(0.1f, 0.5f, tn);
    keep = sb_min(keep, 1.0f - front * raised * grown * live);
  }
  return keep;
}

__device__ float lipFinger(const float4* events,const float4* lookup,SBUniforms sb,float x, float seed) {
  return 0.6f * vnoise(events,lookup,sb,sb_vec2(x * 6.3f + seed * 31.0f, 0.5f)) + 0.4f * vnoise(events,lookup,sb,sb_vec2(x * 15.7f - seed * 7.0f, 1.5f));
}

__device__ float lipDrape(const float4* events,const float4* lookup,SBUniforms sb,Brk b, BrkLip L) { return 0.14f * b.s * sb_smoothstep(0.0f, 0.15f, L.fl) * L.grow; }
#define LIP_TGONE 0.32f

__device__ float2 sheetRaw(const float4* events,const float4* lookup,SBUniforms sb,Brk b, float r, float t, float4& meta, float4& meta2, float2& knee) {
  float tn = brkTnLip(events,lookup,sb,b, t);

  float tnH = brkTn(events,lookup,sb,b, t);
  BrkStage gH = brkStage(events,lookup,sb,tnH);
  float2 AH = brkApex(events,lookup,sb,b, gH, tnH);
  meta = sb_vec4(0.0f, 0.0f, tn, 0.0f);
  meta2 = sb_vec4(1.0f, 0.0f, 0.0f, 0.0f);
  knee = sb_vec2(0.0f);
  float seg = sb_floor(r), f = r - seg;
  float2 q = AH;
  if (tn < BRK_TB - 0.01f && brkArrival(events,lookup,sb,b, t) > brkWhitecapOnset(events,lookup,sb,b) - 0.02f && seg < 4.0f && b.str > 0.0f) {

    BrkStage g = gH;
    float2 A = AH;

    float ri = seg < 1.0f ? r * 8.0f : (seg < 2.0f ? 8.0f + f * 32.0f : (seg < 3.0f ? 40.0f + f * 4.0f : 44.0f + f * 16.0f));
    float pp = ri / 60.0f;
    float xi = pp < 0.3f ? sb_mix(-0.15f, 0.0f, pp / 0.3f)
                        : brkFrontXi(events,lookup,sb,g, 1.0f - sb_mix(0.0f, sb_mix(0.16f, 0.32f, sb_smoothstep(0.4f, 0.9f, b.spill)), (pp - 0.3f) / 0.7f) / sb_max((g.ya - g.yt) * b.s, 0.05f));
    float z = A.x + xi * b.s;
    float3 pr = brkProfileS(events,lookup,sb,b, gH, AH, tnH, z);
    float2 nrm = normalize(sb_vec2(-pr.y, 1.0f));
    {float2 lipSwizzle0=sb_vec2(pp, -1.0f);meta.x=lipSwizzle0.x;meta.y=lipSwizzle0.y;}
    q = sb_vec2(z, pr.x) + sb_vec2(nrm.x, nrm.y) * 0.004f;
  } else

  if (tn < BRK_TB - 0.01f || tn > LIP_TGONE || b.str <= 0.0f) {

    meta.y = -2.0f;
    q = AH + sb_vec2(-0.1f, -0.25f) * b.s;
  } else {
  BrkLip L = brkLip(events,lookup,sb,b, t);
  meta.w = L.fl;

  float2 R0 = L.R, t0 = normalize(L.P1 - L.R), n0 = sb_vec2(-t0.y, t0.x);

  float4 cK = brkLipCurve(events,lookup,sb,L, 1.0f);
  float Ld = lipDrape(events,lookup,sb,b, L);
  float2 Fp = sb_sw_xy(cK) + Ld * sb_vec2(0.216f, -0.976f);
  Fp.y = Fp.y - (lipFinger(events,lookup,sb,b.x, b.seed) * 0.035f * b.s * sb_smoothstep(0.06f, 0.2f, L.fl));
  float lenJ = 0.5f * (length(L.T - L.R) + length(L.P1 - L.R) + length(L.P2 - L.P1) + length(L.T - L.P2));
  float lenD = length(Fp - sb_sw_xy(cK));
  float sk = lenJ / sb_max(lenJ + lenD, 1e-4f);
  meta2 = sb_vec4(sk, lenJ + lenD, lenD, 0.0f);
  knee = sb_sw_xy(cK);
  if (seg < 1.0f) {

    float zb = L.A.x - 0.16f * b.s;
    float3 bp = brkProfileS(events,lookup,sb,b, gH, AH, tnH, zb);
    float2 Bk = sb_vec2(zb, bp.x);
    float2 top = R0 + n0 * (0.5f * L.wR + 0.006f * b.s);

    float f2=f*f, f3=f2*f, span=top.x-zb;
    q=sb_vec2(sb_mix(zb,top.x,f), (2.0f*f3-3.0f*f2+1.0f)*Bk.y
      +(f3-2.0f*f2+f)*span*bp.y+(-2.0f*f3+3.0f*f2)*top.y);
    {float2 lipSwizzle1=sb_vec2(0.0f, L.wR);meta.x=lipSwizzle1.x;meta.y=lipSwizzle1.y;}
  } else if (seg < 4.0f) {

    float sg = seg < 2.0f ? f : (seg < 3.0f ? 1.0f : 1.0f - f);
    float4 c = sb_vec4(0.0f); float w = 0.0f;
    if (sg <= sk || lenD < 1e-4f) {
      float sj = sb_min(sg / sb_max(sk, 1e-4f), 1.0f);
      c = brkLipCurve(events,lookup,sb,L, sj);
      w = brkLipWidth(events,lookup,sb,L, sj);
    } else {
      float u = (sg - sk) / sb_max(1.0f - sk, 1e-4f);
      float2 m0 = sb_sw_zw(cK) * lenD * 0.8f, m1 = sb_vec2(0.1f, -0.995f) * lenD * 0.8f;
      float u2 = u * u, u3 = u2 * u;
      float2 p = (2.0f * u3 - 3.0f * u2 + 1.0f) * sb_sw_xy(cK) + (u3 - 2.0f * u2 + u) * m0 + (-2.0f * u3 + 3.0f * u2) * Fp + (u3 - u2) * m1;
      float2 dp = (6.0f * u2 - 6.0f * u) * sb_sw_xy(cK) + (3.0f * u2 - 4.0f * u + 1.0f) * m0 + (6.0f * u - 6.0f * u2) * Fp + (3.0f * u2 - 2.0f * u) * m1;
      float ld = length(dp);
      c = sb_vec4(p, ld > 1e-6f ? dp / ld : sb_vec2(0.0f, -1.0f));
      w = L.wT * sb_mix(1.0f, 0.55f, u);
    }
    float2 nrm = sb_vec2(-c.w, c.z);
    if (seg < 2.0f) q = sb_sw_xy(c) + nrm * (0.5f * w + 0.006f * b.s);
    else if (seg < 3.0f) { float th = 3.14159265f * f; q = sb_sw_xy(c) + (nrm * sb_cos(th) + sb_sw_zw(c) * sb_sin(th)) * 0.5f * w; }
    else q = sb_sw_xy(c) - nrm * 0.5f * w;
    {float2 lipSwizzle2=sb_vec2(sg, w);meta.x=lipSwizzle2.x;meta.y=lipSwizzle2.y;}
  } else if (seg >= 5.0f) {

    BrkStage g = L.g;
    float hw = sb_max(g.hw, 0.05f);
    float h = sb_mix(hw, 0.02f, sb_min(f / 0.65f, 1.0f));
    float u = 1.0f - (1.0f - g.uw) * powf(h / hw, 1.0f / sb_max(g.m, 1.0f));
    float z = L.A.x + u * g.lf * b.s;
    z = sb_mix(z, sb_max(b.zI + 0.12f * b.s, z + 0.05f), sb_clamp((f - 0.65f) / 0.35f, 0.0f, 1.0f));
    q = sb_vec2(z, brkProfileS(events,lookup,sb,b, gH, AH, tnH, z).x);
    {float2 lipSwizzle3=sb_vec2(-1.0f - f, L.wR);meta.x=lipSwizzle3.x;meta.y=lipSwizzle3.y;}
  } else {

    float2 u0 = R0 - n0 * 0.5f * L.wR;

    BrkStage g = L.g;
    float zW = L.A.x + g.uw * g.lf * b.s * 0.999f;
    float3 pw = brkProfileS(events,lookup,sb,b, gH, AH, tnH, zW);
    float2 W = sb_vec2(zW - 0.01f * b.s, pw.x);
    float2 dW = sb_vec2(0.0f, -1.0f);
    float k = 0.9f * length(W - u0);
    float f2 = f * f, f3 = f2 * f;
    q = (2.0f * f3 - 3.0f * f2 + 1.0f) * u0 + (f3 - 2.0f * f2 + f) * sb_vec2(-k, 0.0f) + (-2.0f * f3 + 3.0f * f2) * W + (f3 - f2) * dW * k;
    {float2 lipSwizzle4=sb_vec2(-f, L.wR);meta.x=lipSwizzle4.x;meta.y=lipSwizzle4.y;}
  }
  if (seg < 4.0f) {

    q.y = q.y - (0.012f * b.s * (1.0f - L.grow) * (1.0f - sb_smoothstep(0.4f, 0.9f, b.spill)));

    q.y = q.y - (0.5f * b.s * sb_smoothstep(0.06f, 0.5f, tn));
  } else if (tn > 0.18f) {

    meta.y = -2.0f;
    q = AH + sb_vec2(-0.1f, -0.25f) * b.s;
  }
  }
  return q;
}

// Batched original-wave evaluation. queries=(x,z,time,sigma), 12 float4 outputs/query.
__global__ void evaluateOriginalWaves(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* queries,float4* results,unsigned int count,int eventIndex){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 SBUniforms sb=uniforms[0];float4 q=queries[i];float2 xz=make_float2(q.x,q.y),d;float foam;
 Brk b=brkAt(events,lookup,sb,eventIndex,q.x);float tn=brkTn(events,lookup,sb,b,q.z);BrkStage stage=brkStage(events,lookup,sb,tn);
 BrkLip lip=brkLip(events,lookup,sb,b,q.z);float4 curve=brkLipCurve(events,lookup,sb,lip,q.w);
 float3 profile=brkProfile(events,lookup,sb,b,q.y,q.z);float eta=brkSurface(events,lookup,sb,xz,q.z,d,foam);
 float3 injection=brkInjection(events,lookup,sb,xz,q.z);
 results[i*12]=make_float4(b.ti,b.D,b.H,b.zI);
 results[i*12+1]=make_float4(tn,stage.za,stage.ya,stage.yt);
 results[i*12+2]=make_float4(stage.lf,stage.uw,stage.hw,stage.lb);
 results[i*12+3]=make_float4(profile.x,profile.y,profile.z,eta);
 results[i*12+4]=make_float4(d.x,d.y,foam,brkWhitecap(events,lookup,sb,b,q.z));
 results[i*12+5]=make_float4(lip.R.x,lip.R.y,lip.T.x,lip.T.y);
 results[i*12+6]=make_float4(lip.P1.x,lip.P1.y,lip.P2.x,lip.P2.y);
 results[i*12+7]=curve;
 results[i*12+8]=make_float4(injection.x,injection.y,injection.z,brkFoam(events,lookup,sb,xz,q.z));
 results[i*12+9]=make_float4(brkImpactSpeed(events,lookup,sb,b),brkLipWidth(events,lookup,sb,lip,q.w),brkSweFoamKeep(events,lookup,sb,xz,q.z),brkCrestSpeed(events,lookup,sb,b,q.z));
 results[i*12+10]=swellTravel(events,lookup,sb,q.y);
 results[i*12+11]=make_float4(stage.m,stage.p,stage.st,brkCrestWidth(events,lookup,sb,xz,profile.x*.75f,q.z));
}
__global__ void evaluateOriginalLip(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* queries,float4* results,unsigned int count,int eventIndex){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 SBUniforms sb=uniforms[0];float4 q=queries[i];Brk b=brkAt(events,lookup,sb,eventIndex,q.x);float4 meta,meta2;float2 knee;
 float2 point=sheetRaw(events,lookup,sb,b,q.w,q.z,meta,meta2,knee);
 results[i*3]=make_float4(point.x,point.y,knee.x,knee.y);results[i*3+1]=meta;results[i*3+2]=meta2;
}

__device__ float2 sb_cmul(float2 a,float2 b){return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
// ORIGINAL SIMULATION PASSES — src/water/OceanFFT.js and src/swash/SwashSim.js
__device__ float2 sb_zw(float4 a){return make_float2(a.z,a.w);}
__device__ float3 sb_xyz(float4 a){return make_float3(a.x,a.y,a.z);}
__device__ float4 sb_fetch(const float4* data,float2 p,int width){return data[(int)p.y*width+(int)p.x];}

__global__ void originalSpectrum(const float4* uH0,float4* output0,float4* output1,float uT,float L0,float L1,float L2) {
int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=768||py>=256)return;int index=py*768+px;

  int c = px / 256;
  int n = px - c * 256;
  int m = py;
  float L = c == 0 ? L0 : (c == 1 ? L1 : L2);
  float dk = 6.283185307f / L;
  float2 k = dk * sb_vec2(n < 128 ? n : n - 256, m < 128 ? m : m - 256);
  float kl = length(k);
  float4 h0 = uH0[index];
  float w = sb_sqrt(9.81f * kl * (1.0f + kl * kl / 136900.0f));

  float ph = sb_mod(w * uT, 6.283185307f);
  float2 e = sb_vec2(sb_cos(ph), sb_sin(ph));

  float2 h = sb_cmul(sb_xy(h0), sb_vec2(e.x, -e.y)) + sb_cmul(sb_vec2(h0.z, -h0.w), e);
  float2 ih = sb_vec2(-h.y, h.x);
  float2 kn = kl > 1e-6f ? k / kl : sb_vec2(0.0f);

  float2 sx = k.x * ih, sz = k.y * ih;
  float2 dx = -kn.x * ih, dz = -kn.y * ih;
  float2 A = h + sb_vec2(-sx.y, sx.x);
  float2 B = sz + sb_vec2(-dx.y, dx.x);
  output0[index] = sb_vec4(A, B);
  output1[index] = sb_vec4(dz, 0.0f, 0.0f);
}

__global__ void originalFFT(const float4* uIn0,const float4* uIn1,float4* output0,float4* output1,int uSub,int uHoriz) {
int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=768||py>=256)return;int index=py*768+px;

  int idx = uHoriz == 1 ? (px & 255) : py;
  int hs = uSub >> 1;
  int ev = (idx / uSub) * hs + (idx & (hs - 1));
  int od = ev + 128;
  float2 pe = uHoriz == 1 ? sb_vec2(px - idx + ev, py) : sb_vec2(px, ev);
  float2 po = uHoriz == 1 ? sb_vec2(px - idx + od, py) : sb_vec2(px, od);
  float ang = 6.283185307f * float(idx & (uSub - 1)) / float(uSub);
  float2 w = sb_vec2(sb_cos(ang), sb_sin(ang));
  float4 e0 = sb_fetch(uIn0,pe,768), o0 = sb_fetch(uIn0,po,768);
  float4 e1 = sb_fetch(uIn1,pe,768), o1 = sb_fetch(uIn1,po,768);
  output0[index] = sb_vec4(sb_xy(e0) + sb_cmul(w, sb_xy(o0)), sb_zw(e0) + sb_cmul(w, sb_zw(o0)));
  output1[index] = sb_vec4(sb_xy(e1) + sb_cmul(w, sb_xy(o1)), sb_zw(e1) + sb_cmul(w, sb_zw(o1)));
}

__global__ void originalCompose(const float4* uIn0,const float4* uIn1,float4* output0,float4* output1,int uCascade,float uChopK) {
int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=256||py>=256)return;int index=py*256+px;

  float2 q = sb_vec2(px + uCascade * 256, py);
  float4 a = sb_fetch(uIn0,q,768);
  float4 b = sb_fetch(uIn1,q,768);
  float h = a.x, sx = a.y, sz = a.z, dx = a.w, dz = b.x;
  output0[index] = sb_vec4(dx * uChopK, h, dz * uChopK, 0.0f);
  output1[index] = sb_vec4(sx, sz, sx * sx, sz * sz);
}
#define SWE_EXPLORE 1
#define G_ACC 9.81f

__device__ float2 kp_cellXZ(const float4* uState,const float4* uBed,float4 uDom,float2 uN,float uDt,float2 ij) { return sb_vec2(uDom.x + (float(ij.x) + 0.5f) * uDom.z, uDom.y + (float(ij.y) + 0.5f) * uDom.w); }

__constant__ float THETA = 1.3f;
__constant__ float EPS4 = 1.0e-8f;
__constant__ float VMAX = 7.5f;

__device__ float2 kp_ghost(const float4* uState,const float4* uBed,float4 uDom,float2 uN,float uDt,float2 p) {
  float2 q = p;
#ifndef SWE_EXPLORE
  if (q.x < 0) q.x = -q.x - 1; else if (q.x >= uN.x) q.x = 2 * uN.x - q.x - 1;
#endif
  if (q.y >= uN.y) q.y = 2 * uN.y - q.y - 1;
  return sb_clamp(q, sb_vec2(0), uN - 1);
}
__device__ float3 kp_St(const float4* uState,const float4* uBed,float4 uDom,float2 uN,float uDt,float2 p) {
  float3 U = sb_xyz(sb_fetch(uState,kp_ghost(uState,uBed,uDom,uN,uDt,p),(int)uN.x));
#ifndef SWE_EXPLORE
  if (p.x < 0 || p.x >= uN.x) U.y = -U.y;
#endif
  if (p.y >= uN.y) U.z = -U.z;
  return U;
}
__device__ float3 kp_Bd(const float4* uState,const float4* uBed,float4 uDom,float2 uN,float uDt,float2 p) { return sb_xyz(sb_fetch(uBed,kp_ghost(uState,uBed,uDom,uN,uDt,p),(int)uN.x)); }

__device__ float kp_mm3(const float4* uState,const float4* uBed,float4 uDom,float2 uN,float uDt,float a, float b, float c) {
  if (a > 0.0f && b > 0.0f && c > 0.0f) return sb_min(a, sb_min(b, c));
  if (a < 0.0f && b < 0.0f && c < 0.0f) return sb_max(a, sb_max(b, c));
  return 0.0f;
}
__device__ float3 kp_mmv(const float4* uState,const float4* uBed,float4 uDom,float2 uN,float uDt,float3 l, float3 c, float3 r) {
  float3 a = THETA * (c - l), b = 0.5f * (r - l), d = THETA * (r - c);
  return sb_vec3(kp_mm3(uState,uBed,uDom,uN,uDt,a.x, b.x, d.x), kp_mm3(uState,uBed,uDom,uN,uDt,a.y, b.y, d.y), kp_mm3(uState,uBed,uDom,uN,uDt,a.z, b.z, d.z));
}

__device__ void kp_recon(const float4* uState,const float4* uBed,float4 uDom,float2 uN,float uDt,float3 Ul, float3 Uc, float3 Ur, float Bm, float Bp, float3& Um, float3& Up) {
  float3 s = kp_mmv(uState,uBed,uDom,uN,uDt,Ul, Uc, Ur);
  Up = Uc + 0.5f * s; Um = Uc - 0.5f * s;

  if (Up.x < Bp || Um.x < Bm) {
    Up=Uc;Um=Uc;
  }
}
struct Face { float w; float h; float u; float v; };
__device__ Face kp_mkFace(const float4* uState,const float4* uBed,float4 uDom,float2 uN,float uDt,float3 U, float B) {
  Face f; f.h = sb_max(U.x - B, 0.0f); f.w = B + f.h;
  float h4 = f.h * f.h * f.h * f.h;
  float den = sb_sqrt(h4 + sb_max(h4, EPS4));
  f.u = sb_clamp(1.41421356f * f.h * U.y / den, -VMAX, VMAX);
  f.v = sb_clamp(1.41421356f * f.h * U.z / den, -VMAX, VMAX);
  return f;
}

__device__ float3 kp_kpFlux(const float4* uState,const float4* uBed,float4 uDom,float2 uN,float uDt,Face L, Face R, int dir) {
  float unL = dir == 0 ? L.u : L.v, unR = dir == 0 ? R.u : R.v;
  float cL = sb_sqrt(G_ACC * L.h), cR = sb_sqrt(G_ACC * R.h);
  float ap = sb_max(sb_max(unL + cL, unR + cR), 0.0f);
  float am = sb_min(sb_min(unL - cL, unR - cR), 0.0f);
  float d = ap - am;
  if (d < 1.0e-7f) return sb_vec3(0.0f);
  float pL = 0.5f * G_ACC * L.h * L.h, pR = 0.5f * G_ACC * R.h * R.h;
  float3 FL, FR;
  if (dir == 0) {
    FL = sb_vec3(L.h * L.u, L.h * L.u * L.u + pL, L.h * L.u * L.v);
    FR = sb_vec3(R.h * R.u, R.h * R.u * R.u + pR, R.h * R.u * R.v);
  } else {
    FL = sb_vec3(L.h * L.v, L.h * L.u * L.v, L.h * L.v * L.v + pL);
    FR = sb_vec3(R.h * R.v, R.h * R.u * R.v, R.h * R.v * R.v + pR);
  }
  float3 UL = sb_vec3(L.w, L.h * L.u, L.h * L.v), UR = sb_vec3(R.w, R.h * R.u, R.h * R.v);
  return (ap * FL - am * FR) / d + (ap * am / d) * (UR - UL);
}

__global__ void originalFlux(const float4* uState,const float4* uBed,float4* output0,float domX,float domZ,float dx,float dz,int nx,int nz,float uDt){
int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=nx||py>=nz)return;int index=py*nx+px;float4 uDom=make_float4(domX,domZ,dx,dz);float2 uN=make_float2(nx,nz);
  float2 p = sb_vec2(px,py);
  float2 ex = sb_vec2(1, 0), ez = sb_vec2(0, 1);
  float3 C = kp_St(uState,uBed,uDom,uN,uDt,p), E = kp_St(uState,uBed,uDom,uN,uDt,p + ex), W = kp_St(uState,uBed,uDom,uN,uDt,p - ex), N = kp_St(uState,uBed,uDom,uN,uDt,p + ez), S = kp_St(uState,uBed,uDom,uN,uDt,p - ez);
  float3 EE = kp_St(uState,uBed,uDom,uN,uDt,p + 2 * ex), WW = kp_St(uState,uBed,uDom,uN,uDt,p - 2 * ex), NN = kp_St(uState,uBed,uDom,uN,uDt,p + 2 * ez), SS = kp_St(uState,uBed,uDom,uN,uDt,p - 2 * ez);
  float3 bC = kp_Bd(uState,uBed,uDom,uN,uDt,p), bE = kp_Bd(uState,uBed,uDom,uN,uDt,p + ex), bW = kp_Bd(uState,uBed,uDom,uN,uDt,p - ex), bN = kp_Bd(uState,uBed,uDom,uN,uDt,p + ez), bS = kp_Bd(uState,uBed,uDom,uN,uDt,p - ez);
  float3 bWW = kp_Bd(uState,uBed,uDom,uN,uDt,p - 2 * ex), bSS = kp_Bd(uState,uBed,uDom,uN,uDt,p - 2 * ez);

#ifdef SWE_EXPLORE

  if(C.x<=bC.x+1e-7f&&E.x<=bE.x+1e-7f&&W.x<=bW.x+1e-7f&&N.x<=bN.x+1e-7f&&S.x<=bS.x+1e-7f
    &&WW.x<=bWW.x+1e-7f&&SS.x<=bSS.x+1e-7f&&EE.x<=kp_Bd(uState,uBed,uDom,uN,uDt,p+2*ex).x+1e-7f&&NN.x<=kp_Bd(uState,uBed,uDom,uN,uDt,p+2*ez).x+1e-7f){
    output0[index]=sb_vec4(bC.x,0.f,0.f,0.f);return;
  }
#endif

  float BeC=bC.y,BwC=bW.y,BnC=bC.z,BsC=bS.z;

  float3 UwC, UeC, UsC, UnC, UwE, UeE, UwW, UeW, UsN, UnN, UsS, UnS;
  kp_recon(uState,uBed,uDom,uN,uDt,W, C, E, BwC, BeC, UwC, UeC);
  kp_recon(uState,uBed,uDom,uN,uDt,C, E, EE, BeC, bE.y, UwE, UeE);
  kp_recon(uState,uBed,uDom,uN,uDt,WW, W, C, bWW.y, BwC, UwW, UeW);
  kp_recon(uState,uBed,uDom,uN,uDt,S, C, N, BsC, BnC, UsC, UnC);
  kp_recon(uState,uBed,uDom,uN,uDt,C, N, NN, BnC, bN.z, UsN, UnN);
  kp_recon(uState,uBed,uDom,uN,uDt,SS, S, C, bSS.z, BsC, UsS, UnS);

  float3 HE = kp_kpFlux(uState,uBed,uDom,uN,uDt,kp_mkFace(uState,uBed,uDom,uN,uDt,UeC, BeC), kp_mkFace(uState,uBed,uDom,uN,uDt,UwE, BeC), 0);
  float3 HW = kp_kpFlux(uState,uBed,uDom,uN,uDt,kp_mkFace(uState,uBed,uDom,uN,uDt,UeW, BwC), kp_mkFace(uState,uBed,uDom,uN,uDt,UwC, BwC), 0);
  float3 GN = kp_kpFlux(uState,uBed,uDom,uN,uDt,kp_mkFace(uState,uBed,uDom,uN,uDt,UnC, BnC), kp_mkFace(uState,uBed,uDom,uN,uDt,UsN, BnC), 1);
  float3 GS = kp_kpFlux(uState,uBed,uDom,uN,uDt,kp_mkFace(uState,uBed,uDom,uN,uDt,UnS, BsC), kp_mkFace(uState,uBed,uDom,uN,uDt,UsC, BsC), 1);

  float4 h0=sb_max(sb_vec4(C.x)-sb_vec4(BeC,BwC,BnC,BsC),sb_vec4(0.0f));
  float Sx=.5f*G_ACC*(h0.x*h0.x-h0.y*h0.y)/uDom.z;
  float Sz=.5f*G_ACC*(h0.z*h0.z-h0.w*h0.w)/uDom.w;

  float3 U = C - uDt / uDom.z * (HE - HW) - uDt / uDom.w * (GN - GS) + uDt * sb_vec3(0.0f, Sx, Sz);
  float h = U.x - bC.x;
  if (h < 1.0e-5f) { U = sb_vec3(bC.x, 0.0f, 0.0f); }
  else {

    float sp = length(sb_vec2(U.y,U.z)) / h;
    if (sp > VMAX) { U.y *= VMAX / sp; U.z *= VMAX / sp; }
  }
  output0[index] = sb_vec4(U, 0.0f);
}
__device__ float3 sb_sw_xyx(float2 a){return make_float3(a.x,a.y,a.x);}
__device__ float3 sb_sw_xyx(float3 a){return make_float3(a.x,a.y,a.x);}
__device__ float3 sb_sw_xyx(float4 a){return make_float3(a.x,a.y,a.x);}
__device__ float3 sb_sw_yzx(float3 a){return make_float3(a.y,a.z,a.x);}
__device__ float3 sb_sw_yzx(float4 a){return make_float3(a.y,a.z,a.x);}
__device__ float2 sb_sw_xx(float2 a){return make_float2(a.x,a.x);}
__device__ float2 sb_sw_xx(float3 a){return make_float2(a.x,a.x);}
__device__ float2 sb_sw_xx(float4 a){return make_float2(a.x,a.x);}
__device__ float2 sb_sw_yz(float3 a){return make_float2(a.y,a.z);}
__device__ float2 sb_sw_yz(float4 a){return make_float2(a.y,a.z);}
__device__ float2 sb_sw_zy(float3 a){return make_float2(a.z,a.y);}
__device__ float2 sb_sw_zy(float4 a){return make_float2(a.z,a.y);}
__device__ float3 sb_sw_yxz(float3 a){return make_float3(a.y,a.x,a.z);}
__device__ float3 sb_sw_yxz(float4 a){return make_float3(a.y,a.x,a.z);}
__device__ float3 sb_sw_xxy(float2 a){return make_float3(a.x,a.x,a.y);}
__device__ float3 sb_sw_xxy(float3 a){return make_float3(a.x,a.x,a.y);}
__device__ float3 sb_sw_xxy(float4 a){return make_float3(a.x,a.x,a.y);}
__device__ float3 sb_sw_yzz(float3 a){return make_float3(a.y,a.z,a.z);}
__device__ float3 sb_sw_yzz(float4 a){return make_float3(a.y,a.z,a.z);}
__device__ float3 sb_sw_zyx(float3 a){return make_float3(a.z,a.y,a.x);}
__device__ float3 sb_sw_zyx(float4 a){return make_float3(a.z,a.y,a.x);}
__device__ float3 sb_sw_yxx(float2 a){return make_float3(a.y,a.x,a.x);}
__device__ float3 sb_sw_yxx(float3 a){return make_float3(a.y,a.x,a.x);}
__device__ float3 sb_sw_yxx(float4 a){return make_float3(a.y,a.x,a.x);}
__device__ float2 sb_sw_yx(float2 a){return make_float2(a.y,a.x);}
__device__ float2 sb_sw_yx(float3 a){return make_float2(a.y,a.x);}
__device__ float2 sb_sw_yx(float4 a){return make_float2(a.y,a.x);}
__device__ float2 sb_sw_xy(float2 a){return make_float2(a.x,a.y);}
__device__ float2 sb_sw_xy(float3 a){return make_float2(a.x,a.y);}
__device__ float2 sb_sw_xy(float4 a){return make_float2(a.x,a.y);}
__device__ float2 sb_sw_zw(float4 a){return make_float2(a.z,a.w);}
__device__ float2 sb_sw_xz(float3 a){return make_float2(a.x,a.z);}
__device__ float2 sb_sw_xz(float4 a){return make_float2(a.x,a.z);}
__device__ float3 sb_sw_xyz(float3 a){return make_float3(a.x,a.y,a.z);}
__device__ float3 sb_sw_xyz(float4 a){return make_float3(a.x,a.y,a.z);}
// ORIGINAL SWASH PASSES — unaltered upstream equations, explicit texture accesses.
__device__ float sb_asinh(float x){return sb_sign(x)*logf(fabsf(x)+sqrtf(x*x+1.0f));}
__device__ float sb_sinh(float x){return (expf(x)-expf(-x))*0.5f;}
__device__ bool sb_anynan(float4 a){return a.x!=a.x||a.y!=a.y||a.z!=a.z||a.w!=a.w;}
__device__ float4 sb_texel(const float4* data,const float4* descriptors,int slot,float2 p){
 float4 d=descriptors[slot];int x=(int)p.x,y=(int)p.y;return data[(int)d.x+y*(int)d.y+x];}
__device__ float4 sb_sample(const float4* data,const float4* descriptors,int slot,float2 uv,float lod){
 float4 d=descriptors[slot];float x=sb_clamp(uv.x*d.y-0.5f,0.0f,d.y-1.0f),y=sb_clamp(uv.y*d.z-0.5f,0.0f,d.z-1.0f);
 int ix=(int)floorf(x),iy=(int)floorf(y),jx=min(ix+1,(int)d.y-1),jy=min(iy+1,(int)d.z-1),o=(int)d.x,w=(int)d.y;
 return sb_mix(sb_mix(data[o+iy*w+ix],data[o+iy*w+jx],x-(float)ix),sb_mix(data[o+jy*w+ix],data[o+jy*w+jx],x-(float)ix),y-(float)iy);}
struct originalBedInitParams {
float4 uRockDomain;
float4 uDom;
float2 uN;
float uRelief;
};

__device__ float originalBedInit_bedRelief(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalBedInitParams sbuParams,float2 xz) {
  float face = sb_smoothstep(-1.2f, 0.2f, xz.y) * sb_smoothstep(5.0f, 2.5f, xz.y);
  float cusp = 0.003f * sb_sin(xz.x * 5.2f + 0.8f * sb_sin(xz.x * 1.7f)) + 0.0025f * sb_sin(xz.x * 14.5f + 1.3f);
  float lumps = 0.010f * gnoise(events,lookup,sb,xz * 1.3f + 3.1f) + 0.005f * gnoise(events,lookup,sb,xz * 4.1f - 1.3f);
  return (cusp + lumps) * face;
}
__device__ float originalBedInit_bedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalBedInitParams sbuParams,float2 xz) { return bedProfile(events,lookup,sb,xz.y) + originalBedInit_bedRelief(events,lookup,sb,data,descriptors,sbuParams,xz); }

__device__ float originalBedInit_rockBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalBedInitParams sbuParams,float2 xz) {
  if(sbuParams.uRockDomain.z <= 0.0f) return -100.0f;
  float2 uv=(xz-sb_sw_xy(sbuParams.uRockDomain))/sb_sw_zw(sbuParams.uRockDomain);
  if(sb_min(uv.x,uv.y)<0.0f || sb_max(uv.x,uv.y)>1.0f) return -100.0f;
  return sb_sample(data,descriptors,0,uv,0.0f).x;
}
__device__ float originalBedInit_hydraulicBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalBedInitParams sbuParams,float2 xz) {return sb_max(originalBedInit_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),originalBedInit_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,xz));}
__device__ float originalBedInit_localRockShadow(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalBedInitParams sbuParams,float3 P,float3 L){
  if(sb_abs(P.x)>65.0f || sb_abs(P.z)>3.3f || P.y>.65f) return 1.0f;
  float shade=1.0f;
  for(int i=0;i<4;i++){
    float h=.045f+.042f*float((i+1)*(i+1));
    float3 q=P+L*(h/sb_max(L.y,.15f));
    shade=sb_min(shade,sb_smoothstep(-.018f,.035f,q.y-originalBedInit_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,sb_sw_xz(q))));
  }
  return shade;
}

__device__ float2 originalBedInit_cellXZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalBedInitParams sbuParams,float2 ij) { return sb_vec2(sbuParams.uDom.x + (float(ij.x) + 0.5f) * sbuParams.uDom.z, sbuParams.uDom.y + (float(ij.y) + 0.5f) * sbuParams.uDom.w); }

__device__ float originalBedInit_sweBed(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalBedInitParams sbuParams,float2 xz) {
  float face = sb_smoothstep(-1.2f, 0.2f, xz.y) * sb_smoothstep(5.0f, 2.5f, xz.y);
  float lumps = 0.010f * gnoise(events,lookup,sb,xz * 1.3f + 3.1f) + 0.005f * gnoise(events,lookup,sb,xz * 4.1f - 1.3f);
  float cusp = 0.012f * sb_sin(xz.x * 5.2f + 0.8f * sb_sin(xz.x * 1.7f)) + 0.010f * sb_sin(xz.x * 14.5f + 1.3f);
  float knee = sb_smoothstep(-1.9f, -1.1f, xz.y) * (1.0f - sb_smoothstep(-0.2f, 0.4f, xz.y));
  float shift = knee * 0.16f * (0.5f * sb_sin(xz.x * 2.3f + 1.0f) + 0.3f * sb_sin(xz.x * 5.1f + 2.0f) + 0.2f * sb_sin(xz.x * 9.7f + 0.4f));
  return sb_max(bedProfile(events,lookup,sb,xz.y - shift) + sbuParams.uRelief * (lumps + 0.15f * cusp) * face,originalBedInit_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,xz));
}
__global__ void originalBedInit(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const originalBedInitParams* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];originalBedInitParams sbuParams=params[0];float2 vUv=(sb_vec2(px,py)+0.5f)/sb_vec2(width,height);
  float2 ij = sb_vec2(sb_vec2(px,py));
  float2 c = originalBedInit_cellXZ(events,lookup,sb,data,descriptors,sbuParams,ij);
  float2 h = 0.5f * sb_sw_zw(sbuParams.uDom);
  float bNE = originalBedInit_sweBed(events,lookup,sb,data,descriptors,sbuParams,c + sb_vec2(h.x, h.y)), bNW = originalBedInit_sweBed(events,lookup,sb,data,descriptors,sbuParams,c + sb_vec2(-h.x, h.y));
  float bSE = originalBedInit_sweBed(events,lookup,sb,data,descriptors,sbuParams,c + sb_vec2(h.x, -h.y)), bSW = originalBedInit_sweBed(events,lookup,sb,data,descriptors,sbuParams,c + sb_vec2(-h.x, -h.y));
  float Bc = 0.25f * (bNE + bNW + bSE + bSW);

  float Bce=.25f*(bNE+bSE+originalBedInit_sweBed(events,lookup,sb,data,descriptors,sbuParams,c+sb_vec2(3.0f*h.x,h.y))+originalBedInit_sweBed(events,lookup,sb,data,descriptors,sbuParams,c+sb_vec2(3.0f*h.x,-h.y)));
  float Bcn=.25f*(bNE+bNW+originalBedInit_sweBed(events,lookup,sb,data,descriptors,sbuParams,c+sb_vec2(h.x,3.0f*h.y))+originalBedInit_sweBed(events,lookup,sb,data,descriptors,sbuParams,c+sb_vec2(-h.x,3.0f*h.y)));
  output[0*width*height+index] = sb_vec4(Bc,sb_max(Bc,Bce),sb_max(Bc,Bcn),0.0f);
}
struct originalStateInitParams {
float4 uDom;
float2 uN;
};

__device__ float2 originalStateInit_cellXZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalStateInitParams sbuParams,float2 ij) { return sb_vec2(sbuParams.uDom.x + (float(ij.x) + 0.5f) * sbuParams.uDom.z, sbuParams.uDom.y + (float(ij.y) + 0.5f) * sbuParams.uDom.w); }

__global__ void originalStateInit(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const originalStateInitParams* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];originalStateInitParams sbuParams=params[0];float2 vUv=(sb_vec2(px,py)+0.5f)/sb_vec2(width,height);
  float2 ij = sb_vec2(sb_vec2(px,py));
  float B = sb_texel(data,descriptors,0,ij).x;
  output[0*width*height+index] = sb_vec4(sb_max(B, 0.0f), 0.0f, 0.0f, 0.0f);
}
struct originalSourcesParams {
float4 uRockDomain;
float4 uDom;
float2 uN;
float4 uSwashFarMap;
float4 uSwashFarMapZ;
float4 uSwashFarWin;
float2 uSwashFocus;
float uDtS;
float uManning;
float uFricH;
float uInfil;
float2 uInfZ;
float uInfLow;
float uThinN;
float4 uRetain;
float uEdgeNudge;
};
#define originalSources_SWE_EXPLORE 1

__device__ float originalSources_bedRelief(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 xz) {
  float face = sb_smoothstep(-1.2f, 0.2f, xz.y) * sb_smoothstep(5.0f, 2.5f, xz.y);
  float cusp = 0.003f * sb_sin(xz.x * 5.2f + 0.8f * sb_sin(xz.x * 1.7f)) + 0.0025f * sb_sin(xz.x * 14.5f + 1.3f);
  float lumps = 0.010f * gnoise(events,lookup,sb,xz * 1.3f + 3.1f) + 0.005f * gnoise(events,lookup,sb,xz * 4.1f - 1.3f);
  return (cusp + lumps) * face;
}
__device__ float originalSources_bedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 xz) { return bedProfile(events,lookup,sb,xz.y) + originalSources_bedRelief(events,lookup,sb,data,descriptors,sbuParams,xz); }

__device__ float originalSources_rockBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 xz) {
  if(sbuParams.uRockDomain.z <= 0.0f) return -100.0f;
  float2 uv=(xz-sb_sw_xy(sbuParams.uRockDomain))/sb_sw_zw(sbuParams.uRockDomain);
  if(sb_min(uv.x,uv.y)<0.0f || sb_max(uv.x,uv.y)>1.0f) return -100.0f;
  return sb_sample(data,descriptors,0,uv,0.0f).x;
}
__device__ float originalSources_hydraulicBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 xz) {return sb_max(originalSources_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),originalSources_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,xz));}
__device__ float originalSources_localRockShadow(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float3 P,float3 L){
  if(sb_abs(P.x)>65.0f || sb_abs(P.z)>3.3f || P.y>.65f) return 1.0f;
  float shade=1.0f;
  for(int i=0;i<4;i++){
    float h=.045f+.042f*float((i+1)*(i+1));
    float3 q=P+L*(h/sb_max(L.y,.15f));
    shade=sb_min(shade,sb_smoothstep(-.018f,.035f,q.y-originalSources_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,sb_sw_xz(q))));
  }
  return shade;
}

__device__ float2 originalSources_cellXZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 ij) { return sb_vec2(sbuParams.uDom.x + (float(ij.x) + 0.5f) * sbuParams.uDom.z, sbuParams.uDom.y + (float(ij.y) + 0.5f) * sbuParams.uDom.w); }

#ifndef originalSources_SWASH_FAR_CORE_DEF
#define originalSources_SWASH_FAR_CORE_DEF 1
#define originalSources_SWASH_FAR_ON 1

__device__ float originalSources_swashFarU(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float x) {
  float g = sbuParams.uSwashFarMap.y * sb_asinh((x - sbuParams.uSwashFarMap.x) / sbuParams.uSwashFarMap.y);
  return 0.5f + 0.5f * g / sbuParams.uSwashFarMap.z;
}
__device__ float originalSources_swashFarXofColumn(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float i) {
  float g = ((i + 0.5f) / sbuParams.uSwashFarMapZ.w - 0.5f) * 2.0f * sbuParams.uSwashFarMap.z;
  return sbuParams.uSwashFarMap.x + sbuParams.uSwashFarMap.y * sb_sinh(g / sbuParams.uSwashFarMap.y);
}
__device__ float2 originalSources_swashFarUV(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 xz) { return sb_vec2(originalSources_swashFarU(events,lookup,sb,data,descriptors,sbuParams,xz.x), (xz.y - sbuParams.uSwashFarMapZ.x) / (sbuParams.uSwashFarMapZ.y * sbuParams.uSwashFarMapZ.z) + 0.5f / sbuParams.uSwashFarMapZ.z); }
__device__ float originalSources_swashVisualFocusX(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams) {

  return sb_clamp(sbuParams.uSwashFocus.x, sbuParams.uSwashFarWin.x - 1.1f, sbuParams.uSwashFarWin.x + 1.1f);
}
__device__ float originalSources_swashFarWeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 xz) {

  float visible = sb_smoothstep(sbuParams.uSwashFarWin.y, sbuParams.uSwashFarWin.z, sb_abs(xz.x - originalSources_swashVisualFocusX(events,lookup,sb,data,descriptors,sbuParams)));

  float bounds = sb_smoothstep(9.95f, 10.14f, sb_abs(xz.x - sbuParams.uSwashFarWin.x));
  return sbuParams.uSwashFarWin.w * sb_max(visible, bounds);
}

__device__ float originalSources_swashFarInZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 uv) { return sb_step(0.0f, uv.y) * sb_step(uv.y, 1.0f); }
__device__ float4 originalSources_swashFar(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 xz, float t) {
  float2 uv = originalSources_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  float4 result = sb_vec4(0.0f);
  if (originalSources_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv) < 0.5f) { float B = bedProfile(events,lookup,sb,xz.y); result = sb_vec4(sb_max(-B, 0.0f), 0.0f, 0.0f, sb_max(B, 0.0f)); }
  else result = sb_sample(data,descriptors,1,uv,0.0f);
  return result;
}
__device__ float4 originalSources_swashFarFoam(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 xz, float t) {
  float2 uv = originalSources_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  return sb_sample(data,descriptors,2,uv,0.0f) * originalSources_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv);
}
__device__ float2 originalSources_swashFarWet(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 xz, float t) {
  float2 uv = originalSources_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  float2 result = sb_vec2(0.0f);
  if (originalSources_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv) < 0.5f) result = sb_vec2(0.0f, 1.0f - sb_smoothstep(0.35f, 0.75f, bedProfile(events,lookup,sb,xz.y)));
  else result = sb_sw_xy(sb_sample(data,descriptors,3,uv,0.0f));
  return result;
}
__device__ float4 originalSources_swashFarLace(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 xz, float t) {
  float2 uv = originalSources_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  float d = sb_sample(data,descriptors,3,uv,0.0f).z * originalSources_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv);
  return sb_vec4(0.0f, -d, 0.0f, -d);
}
__device__ float2 originalSources_swashFarGrad(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 xz, float t) {
  float ex = 0.05f, ez = sbuParams.uSwashFarMapZ.y;
  float wE = originalSources_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz + sb_vec2(ex, 0.0f), t).w, wW = originalSources_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz - sb_vec2(ex, 0.0f), t).w;
  float wN = originalSources_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz + sb_vec2(0.0f, ez), t).w, wS = originalSources_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz - sb_vec2(0.0f, ez), t).w;
  return sb_vec2((wE - wW) / (2.0f * ex), (wN - wS) / (2.0f * ez));
}
__device__ void originalSources_swashFarAll(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalSourcesParams sbuParams,float2 xz, float t, float4& view, float4& foam, float2& wet, float4& lace) {
  float2 uv = originalSources_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  float inZ = originalSources_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv);
  float B = bedProfile(events,lookup,sb,xz.y);
  view = inZ > 0.5f ? sb_sample(data,descriptors,1,uv,0.0f) : sb_vec4(sb_max(-B, 0.0f), 0.0f, 0.0f, sb_max(B, 0.0f));
  foam = sb_sample(data,descriptors,2,uv,0.0f) * inZ;
  float4 w = sb_sample(data,descriptors,3,uv,0.0f);
  wet = inZ > 0.5f ? sb_sw_xy(w) : sb_vec2(0.0f, 1.0f - sb_smoothstep(0.35f, 0.75f, B));
  lace = sb_vec4(0.0f, -w.z, 0.0f, -w.z) * inZ;
}
#endif

__global__ void originalSources(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const originalSourcesParams* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];originalSourcesParams sbuParams=params[0];float2 vUv=(sb_vec2(px,py)+0.5f)/sb_vec2(width,height);
  float2 p = sb_vec2(sb_vec2(px,py));
  float3 U = sb_sw_xyz(sb_texel(data,descriptors,4,p));
  float B = sb_texel(data,descriptors,5,p).x;
  float2 xz = originalSources_cellXZ(events,lookup,sb,data,descriptors,sbuParams,p);

  float3 inj = brkInjection(events,lookup,sb,xz, sb.uTime);
  U.x += inj.x * sbuParams.uDtS;
  U.y += inj.z * sbuParams.uDtS;
  U.z += inj.y * sbuParams.uDtS;
  float h = U.x - B;
  if (h > 1.0e-4f) {
    float2 vel = sb_sw_yz(U) / h;
    float sp = length(vel);

    float n = sbuParams.uManning * sb_mix(sbuParams.uThinN, 1.0f, sb_smoothstep(0.003f, 0.012f, h));
    float cf = G_ACC * n * n * sp / powf(sb_max(h, sbuParams.uFricH), 1.3333f);
    {float2 swizzle0=sb_sw_yz(U)/(1.0f + sbuParams.uDtS * cf);U.y=swizzle0.x;U.z=swizzle0.y;}

    float dry = sb_max(sb_smoothstep(sbuParams.uInfZ.x, sbuParams.uInfZ.y, B), sbuParams.uInfLow * sb_smoothstep(0.006f, 0.0f, h) * sb_step(0.0f, B));

    float hRet = sbuParams.uRetain.x * (1.0f - sb_smoothstep(sbuParams.uRetain.y, sbuParams.uRetain.z, B)) * sb_step(0.0f, B);
    float dh = sb_min(sb_max(h - hRet, 0.0f), sbuParams.uInfil * sbuParams.uDtS * dry * (0.4f + 0.6f * sb_smoothstep(0.05f, 0.0f, h)))
             + sb_min(h, hRet) * sbuParams.uDtS / sbuParams.uRetain.w;
    U.x -= dh;
    h -= dh;
    if (h < 1.0e-4f) {float2 swizzle1=sb_vec2(0.0f);U.y=swizzle1.x;U.z=swizzle1.y;}
  } else {
    U = sb_vec3(sb_max(U.x, B), 0.0f, 0.0f);
  }
#ifdef originalSources_SWE_EXPLORE

  float sp = 1.0f - sb_smoothstep(SWE_ZMIN, SWE_ZMIN + 0.9f, xz.y);
  if (B < 0.0f) U.x = sb_mix(U.x, 0.0f, sp * 0.10f);
  {float2 swizzle2=sb_sw_yz(U)*(1.0f - sp * 0.10f);U.y=swizzle2.x;U.z=swizzle2.y;}
  float xR = sbuParams.uDom.x + float(sbuParams.uN.x) * sbuParams.uDom.z;
  float sx = 1.0f - sb_smoothstep(0.0f, 1.0f, sb_min(xz.x - sbuParams.uDom.x, xR - xz.x));
  if (sx * sbuParams.uEdgeNudge > 0.0f) {
    float4 fv = originalSources_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz, sb.uTime);
    float wt = sb_mix(B + fv.x, sb_max(bedProfile(events,lookup,sb,xz.y) + fv.x, B), sb_smoothstep(0.03f, 0.1f, fv.x));
    U = sb_mix(U, sb_vec3(wt, fv.x * fv.y, fv.x * fv.z), sx * sbuParams.uEdgeNudge);
  }
#else

  float sp = 1.0f - sb_smoothstep(SWE_ZMIN, SWE_ZMIN + 0.9f, xz.y);
  float sx = sb_smoothstep(SWE_XMAX - 0.8f, SWE_XMAX, sb_abs(xz.x));
  float relax = sb_max(sp * 0.10f, sx * 0.03f);
  if (B < 0.0f) U.x = sb_mix(U.x, 0.0f, relax);
  {float2 swizzle3=sb_sw_yz(U)*(1.0f - relax);U.y=swizzle3.x;U.z=swizzle3.y;}
#endif
  output[0*width*height+index] = sb_vec4(U, 0.0f);
}
struct originalFoamParams {
float4 uRockDomain;
float4 uDom;
float2 uN;
float uDtS;
float uResetA;
float uResetB;
float4 uFoamK;
float4 uFoamS;
float4 uFoamM;
float4 uFoamK2;
float uFoamRMax;
float uFoamCap;
float uFoamPatch;
float4 uFoamS2;
};

__device__ float originalFoam_bedRelief(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamParams sbuParams,float2 xz) {
  float face = sb_smoothstep(-1.2f, 0.2f, xz.y) * sb_smoothstep(5.0f, 2.5f, xz.y);
  float cusp = 0.003f * sb_sin(xz.x * 5.2f + 0.8f * sb_sin(xz.x * 1.7f)) + 0.0025f * sb_sin(xz.x * 14.5f + 1.3f);
  float lumps = 0.010f * gnoise(events,lookup,sb,xz * 1.3f + 3.1f) + 0.005f * gnoise(events,lookup,sb,xz * 4.1f - 1.3f);
  return (cusp + lumps) * face;
}
__device__ float originalFoam_bedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamParams sbuParams,float2 xz) { return bedProfile(events,lookup,sb,xz.y) + originalFoam_bedRelief(events,lookup,sb,data,descriptors,sbuParams,xz); }

__device__ float originalFoam_rockBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamParams sbuParams,float2 xz) {
  if(sbuParams.uRockDomain.z <= 0.0f) return -100.0f;
  float2 uv=(xz-sb_sw_xy(sbuParams.uRockDomain))/sb_sw_zw(sbuParams.uRockDomain);
  if(sb_min(uv.x,uv.y)<0.0f || sb_max(uv.x,uv.y)>1.0f) return -100.0f;
  return sb_sample(data,descriptors,0,uv,0.0f).x;
}
__device__ float originalFoam_hydraulicBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamParams sbuParams,float2 xz) {return sb_max(originalFoam_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),originalFoam_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,xz));}
__device__ float originalFoam_localRockShadow(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamParams sbuParams,float3 P,float3 L){
  if(sb_abs(P.x)>65.0f || sb_abs(P.z)>3.3f || P.y>.65f) return 1.0f;
  float shade=1.0f;
  for(int i=0;i<4;i++){
    float h=.045f+.042f*float((i+1)*(i+1));
    float3 q=P+L*(h/sb_max(L.y,.15f));
    shade=sb_min(shade,sb_smoothstep(-.018f,.035f,q.y-originalFoam_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,sb_sw_xz(q))));
  }
  return shade;
}

__device__ float2 originalFoam_cellXZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamParams sbuParams,float2 ij) { return sb_vec2(sbuParams.uDom.x + (float(ij.x) + 0.5f) * sbuParams.uDom.z, sbuParams.uDom.y + (float(ij.y) + 0.5f) * sbuParams.uDom.w); }

__device__ float3 originalFoam_st(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamParams sbuParams,float2 q) { return sb_sw_xyz(sb_texel(data,descriptors,1,sb_clamp(q, sb_vec2(0), sbuParams.uN - 1))); }
__device__ float originalFoam_bd(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamParams sbuParams,float2 q) { return sb_texel(data,descriptors,2,sb_clamp(q, sb_vec2(0), sbuParams.uN - 1)).x; }
__device__ float2 originalFoam_velAt(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamParams sbuParams,float2 q) {
  float3 U = originalFoam_st(events,lookup,sb,data,descriptors,sbuParams,q); float h = sb_max(U.x - originalFoam_bd(events,lookup,sb,data,descriptors,sbuParams,q), 0.0f);
  float h4 = h * h * h * h;
  return 1.41421356f * h * sb_sw_yz(U) / sb_sqrt(h4 + sb_max(h4, 1.0e-8f));
}
__device__ float originalFoam_hAt(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamParams sbuParams,float2 q) { return sb_max(originalFoam_st(events,lookup,sb,data,descriptors,sbuParams,q).x - originalFoam_bd(events,lookup,sb,data,descriptors,sbuParams,q), 0.0f); }
__global__ void originalFoam(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const originalFoamParams* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];originalFoamParams sbuParams=params[0];float2 vUv=(sb_vec2(px,py)+0.5f)/sb_vec2(width,height);
  float2 p = sb_vec2(sb_vec2(px,py));
  float2 xz = originalFoam_cellXZ(events,lookup,sb,data,descriptors,sbuParams,p);
  float3 U = originalFoam_st(events,lookup,sb,data,descriptors,sbuParams,p);
  float B = originalFoam_bd(events,lookup,sb,data,descriptors,sbuParams,p);
  float h = sb_max(U.x - B, 0.0f);
  float2 vel = originalFoam_velAt(events,lookup,sb,data,descriptors,sbuParams,p);
  float2 texel = 1.0f / sb_vec2(sbuParams.uN);
  float2 uv = (sb_vec2(p) + 0.5f) * texel;
  float2 back = uv - vel * sbuParams.uDtS / (sb_vec2(sbuParams.uN) * sb_sw_zw(sbuParams.uDom));
  float4 F = sb_sample(data,descriptors,3,back,0.0f);
  float4 L = sb_sample(data,descriptors,4,back,0.0f);

  float thinF = (1.0f - sb_smoothstep(0.02f, 0.05f, h)) * sb_step(1.0e-4f, h);
  if (thinF > 0.0f) {
    float2 t2 = sb_vec2(2.0f / float(sbuParams.uN.x), 0.0f);
    F = sb_mix(F, 0.5f * (sb_sample(data,descriptors,3,back + t2,0.0f) + sb_sample(data,descriptors,3,back - t2,0.0f)), 0.3f * thinF);
  }

  L -= sb_vec4(vel * sbuParams.uDtS, vel * sbuParams.uDtS);

  {
    float2 tx = sb_vec2(1.0f / float(sbuParams.uN.x), 0.0f), tz = sb_vec2(0.0f, 1.0f / float(sbuParams.uN.y));
    float4 LE = sb_sample(data,descriptors,4,back + tx,0.0f), LW = sb_sample(data,descriptors,4,back - tx,0.0f);
    float4 LN = sb_sample(data,descriptors,4,back + tz,0.0f), LS = sb_sample(data,descriptors,4,back - tz,0.0f);
    float4 gx = (LE - LW) / (2.0f * sbuParams.uDom.z), gz = (LN - LS) / (2.0f * sbuParams.uDom.w);
    float gA = length(sb_vec4(sb_sw_xy(gx), sb_sw_xy(gz))), gB = length(sb_vec4(sb_sw_zw(gx), sb_sw_zw(gz)));
    float4 avg = 0.25f * (LE + LW + LN + LS) - sb_vec4(vel * sbuParams.uDtS, vel * sbuParams.uDtS);
    {float2 swizzle0=sb_mix(sb_sw_xy(L), sb_sw_xy(avg), 0.5f * sb_smoothstep(0.8f, 2.5f, gA));L.x=swizzle0.x;L.y=swizzle0.y;}
    {float2 swizzle1=sb_mix(sb_sw_zw(L), sb_sw_zw(avg), 0.5f * sb_smoothstep(0.8f, 2.5f, gB));L.z=swizzle1.x;L.w=swizzle1.y;}
  }
  if (sbuParams.uResetA > 0.5f) {float2 swizzle2=sb_vec2(0.0f);L.x=swizzle2.x;L.y=swizzle2.y;}
  if (sbuParams.uResetB > 0.5f) {float2 swizzle3=sb_vec2(0.0f);L.z=swizzle3.x;L.w=swizzle3.y;}

  float2 ex = sb_vec2(1, 0), ez = sb_vec2(0, 1);
  float2 vE = originalFoam_velAt(events,lookup,sb,data,descriptors,sbuParams,p + ex), vW = originalFoam_velAt(events,lookup,sb,data,descriptors,sbuParams,p - ex), vN = originalFoam_velAt(events,lookup,sb,data,descriptors,sbuParams,p + ez), vS = originalFoam_velAt(events,lookup,sb,data,descriptors,sbuParams,p - ez);
  float hE = originalFoam_hAt(events,lookup,sb,data,descriptors,sbuParams,p + ex), hW = originalFoam_hAt(events,lookup,sb,data,descriptors,sbuParams,p - ex), hN = originalFoam_hAt(events,lookup,sb,data,descriptors,sbuParams,p + ez), hS = originalFoam_hAt(events,lookup,sb,data,descriptors,sbuParams,p - ez);
  float dhdt = -((hE * vE.x - hW * vW.x) / (2.0f * sbuParams.uDom.z) + (hN * vN.y - hS * vS.y) / (2.0f * sbuParams.uDom.w));
  float sp = length(vel);
  float front = sb_smoothstep(sbuParams.uFoamS.w * 0.25f, sbuParams.uFoamS.w, dhdt) * sb_smoothstep(0.012f, 0.045f, sb_max(h, sb_max(hS, hN))) * sb_max(sp, 0.6f);

  front *= sb_smoothstep(-0.1f, 0.5f, vel.y / sb_max(sp, 0.25f));

  float Fr = sp / sb_sqrt(G_ACC * sb_max(h, 0.003f));
  front += sbuParams.uFoamM.y * sb_smoothstep(0.4f, 1.2f, vel.y) * sb_smoothstep(1.1f, 2.0f, Fr) * sb_smoothstep(0.01f, 0.035f, h);

  float conv = -((vE.x - vW.x) / (2.0f * sbuParams.uDom.z) + (vN.y - vS.y) / (2.0f * sbuParams.uDom.w));
  float jump = sb_smoothstep(sbuParams.uFoamK2.w * 0.3f, sbuParams.uFoamK2.w, conv) * sb_smoothstep(0.008f, 0.03f, h);

  float2 bedSlope=sb_vec2(originalFoam_bd(events,lookup,sb,data,descriptors,sbuParams,p+ex)-originalFoam_bd(events,lookup,sb,data,descriptors,sbuParams,p-ex),originalFoam_bd(events,lookup,sb,data,descriptors,sbuParams,p+ez)-originalFoam_bd(events,lookup,sb,data,descriptors,sbuParams,p-ez))/(2.0f*sb_sw_zw(sbuParams.uDom));
  float rockRise=sb_max(originalFoam_bd(events,lookup,sb,data,descriptors,sbuParams,p)-originalFoam_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),0.0f);
  float contact=sb_clamp(dot(vel,bedSlope),0.0f,1.8f)*sb_smoothstep(.035f,.12f,rockRise)*sb_smoothstep(.12f,.70f,sp)*sb_smoothstep(.008f,.035f,h);
  front+=.065f*contact;
  float toe = sb_smoothstep(0.3f, 0.75f, 0.5f + 0.5f * gnoise(events,lookup,sb,sb_vec2(xz.x * 2.6f, sb.uTime * 0.9f + xz.y * 1.5f)));
  front += sbuParams.uFoamM.w * jump * sb_mix(0.2f + 1.2f * toe, 1.0f, sb_smoothstep(0.1f, 0.5f, vS.y));

  float edge = sb_smoothstep(0.002f, 0.006f, h) * (1.0f - sb_smoothstep(0.3f * h, 0.7f * h, hN)) * (1.0f - sb_smoothstep(0.9f, 1.5f, vel.y));
  front += sbuParams.uFoamK2.z * edge * sb_smoothstep(0.1f, 0.6f, vel.y) * sb_smoothstep(0.2f, 0.8f, F.z + 0.3f * F.x);
  float3 inj = brkInjection(events,lookup,sb,xz, sb.uTime);
  float splash = brkFoam(events,lookup,sb,xz, sb.uTime);

  float R = F.x, Gf = F.y, K = F.z, Mk = F.w;

  float entrain = sbuParams.uFoamS2.y * sb_smoothstep(0.35f, 1.3f, K) * sb_smoothstep(0.02f, 0.07f, h);

  float pn = gnoise(events,lookup,sb,xz * sb_vec2(1.1f, 1.5f) + sb_vec2(0.0f, sb.uTime * 0.35f)) + 0.5f * gnoise(events,lookup,sb,xz * sb_vec2(2.6f, 3.2f) - sb_vec2(sb.uTime * 0.5f, 1.7f));
  float prod = 1.0f + sbuParams.uFoamPatch * (2.0f * sb_smoothstep(-0.7f, 0.7f, pn) - 1.0f);
  R += sbuParams.uDtS * prod * (sbuParams.uFoamS.x * inj.x + sbuParams.uFoamS.y * front + splash + entrain);

  R = sb_min(R, sbuParams.uFoamCap * sb_max(h - 0.002f, 0.0f) + 0.02f);

  float persist = sb_smoothstep(0.05f, 0.13f, h) * (1.0f - sb_smoothstep(0.2f, 0.9f, xz.y)) * sb_smoothstep(0.15f, 0.7f, K)
                * sb_smoothstep(-0.9f, -0.3f, vel.y) * sb_smoothstep(-1.9f, -1.2f, xz.y);
  float tauR = sbuParams.uFoamK.x * sb_mix(1.0f, sbuParams.uFoamM.z, persist) * (1.0f + sbuParams.uFoamS2.x * sb_max(R - 0.5f, 0.0f))
             * sb_mix(sbuParams.uFoamS2.z, 1.0f, sb_max(sb_smoothstep(0.15f, 0.5f, sp), sb_smoothstep(0.03f, 0.08f, h)));
  float loss = R * (1.0f - sb_exp(-sbuParams.uDtS / tauR));
  R -= loss;
  float tauG = sb_mix(sbuParams.uFoamK.z, sbuParams.uFoamK.w, sb_smoothstep(0.12f, 0.45f, h));
  Gf += sbuParams.uFoamK.y * loss;
  Gf *= sb_exp(-sbuParams.uDtS / tauG);
  float drained = 1.0f - sb_smoothstep(0.0004f, 0.0022f, h);
  R *= sb_exp(-sbuParams.uDtS * 10.0f * drained);
  Gf *= sb_exp(-sbuParams.uDtS * 7.0f * drained);

  float shear = sb_smoothstep(0.1f, 0.9f, sp) * sb_smoothstep(0.002f, 0.012f, h) * (1.0f - sb_smoothstep(0.15f, 0.4f, h));

  K += sbuParams.uDtS * (6.0f * inj.x + 1.5f * front + sbuParams.uFoamK2.x * shear + 2.0f * jump * sb_mix(1.0f, 0.35f, sb_smoothstep(0.2f, 0.6f, -vN.y)));
  K *= sb_exp(-sbuParams.uDtS / sb_mix(0.7f, sbuParams.uFoamS2.w, sb_smoothstep(0.05f, 0.2f, h)));

  Mk += sbuParams.uFoamS.z * (loss + sbuParams.uDtS * inj.x) + sbuParams.uDtS * sbuParams.uFoamK2.y * shear * sb_smoothstep(0.02f, 0.045f, h) * (1.0f - sb_smoothstep(0.2f, 0.7f, -vel.y));
  Mk *= sb_exp(-sbuParams.uDtS / (sbuParams.uFoamM.x * sb_mix(0.5f, 1.0f, sb_smoothstep(0.01f, 0.04f, h)))) * sb_exp(-sbuParams.uDtS * (4.0f * drained + 2.5f * jump * sb_smoothstep(0.1f, 0.5f, -vN.y)));
  output[0*width*height+index] = sb_vec4(sb_clamp(R, 0.0f, sbuParams.uFoamRMax), sb_clamp(Gf, 0.0f, 2.0f), sb_clamp(K, 0.0f, 4.0f), sb_clamp(Mk, 0.0f, 1.0f));
  output[1*width*height+index] = L;
}
struct originalFoamViewParams {
float4 uRockDomain;
float4 uDom;
float2 uN;
};

__device__ float originalFoamView_bedRelief(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamViewParams sbuParams,float2 xz) {
  float face = sb_smoothstep(-1.2f, 0.2f, xz.y) * sb_smoothstep(5.0f, 2.5f, xz.y);
  float cusp = 0.003f * sb_sin(xz.x * 5.2f + 0.8f * sb_sin(xz.x * 1.7f)) + 0.0025f * sb_sin(xz.x * 14.5f + 1.3f);
  float lumps = 0.010f * gnoise(events,lookup,sb,xz * 1.3f + 3.1f) + 0.005f * gnoise(events,lookup,sb,xz * 4.1f - 1.3f);
  return (cusp + lumps) * face;
}
__device__ float originalFoamView_bedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamViewParams sbuParams,float2 xz) { return bedProfile(events,lookup,sb,xz.y) + originalFoamView_bedRelief(events,lookup,sb,data,descriptors,sbuParams,xz); }

__device__ float originalFoamView_rockBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamViewParams sbuParams,float2 xz) {
  if(sbuParams.uRockDomain.z <= 0.0f) return -100.0f;
  float2 uv=(xz-sb_sw_xy(sbuParams.uRockDomain))/sb_sw_zw(sbuParams.uRockDomain);
  if(sb_min(uv.x,uv.y)<0.0f || sb_max(uv.x,uv.y)>1.0f) return -100.0f;
  return sb_sample(data,descriptors,0,uv,0.0f).x;
}
__device__ float originalFoamView_hydraulicBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamViewParams sbuParams,float2 xz) {return sb_max(originalFoamView_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),originalFoamView_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,xz));}
__device__ float originalFoamView_localRockShadow(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamViewParams sbuParams,float3 P,float3 L){
  if(sb_abs(P.x)>65.0f || sb_abs(P.z)>3.3f || P.y>.65f) return 1.0f;
  float shade=1.0f;
  for(int i=0;i<4;i++){
    float h=.045f+.042f*float((i+1)*(i+1));
    float3 q=P+L*(h/sb_max(L.y,.15f));
    shade=sb_min(shade,sb_smoothstep(-.018f,.035f,q.y-originalFoamView_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,sb_sw_xz(q))));
  }
  return shade;
}

__device__ float2 originalFoamView_cellXZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFoamViewParams sbuParams,float2 ij) { return sb_vec2(sbuParams.uDom.x + (float(ij.x) + 0.5f) * sbuParams.uDom.z, sbuParams.uDom.y + (float(ij.y) + 0.5f) * sbuParams.uDom.w); }

__global__ void originalFoamView(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const originalFoamViewParams* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];originalFoamViewParams sbuParams=params[0];float2 vUv=(sb_vec2(px,py)+0.5f)/sb_vec2(width,height);
  float2 p = sb_vec2(sb_vec2(px,py));
  float4 F = sb_texel(data,descriptors,1,p);
  F = sb_anynan(F) ? sb_vec4(0.0f) : sb_clamp(F, sb_vec4(0.0f), sb_vec4(8.0f, 2.0f, 4.0f, 1.0f));
  output[0*width*height+index] = F * brkSweFoamKeep(events,lookup,sb,originalFoamView_cellXZ(events,lookup,sb,data,descriptors,sbuParams,p), sb.uTime);
}
struct originalWetParams {
float4 uRockDomain;
float4 uDom;
float2 uN;
float uDtS;
float uFilmTau;
};
#define originalWet_SWE_EXPLORE 1

__device__ float originalWet_bedRelief(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalWetParams sbuParams,float2 xz) {
  float face = sb_smoothstep(-1.2f, 0.2f, xz.y) * sb_smoothstep(5.0f, 2.5f, xz.y);
  float cusp = 0.003f * sb_sin(xz.x * 5.2f + 0.8f * sb_sin(xz.x * 1.7f)) + 0.0025f * sb_sin(xz.x * 14.5f + 1.3f);
  float lumps = 0.010f * gnoise(events,lookup,sb,xz * 1.3f + 3.1f) + 0.005f * gnoise(events,lookup,sb,xz * 4.1f - 1.3f);
  return (cusp + lumps) * face;
}
__device__ float originalWet_bedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalWetParams sbuParams,float2 xz) { return bedProfile(events,lookup,sb,xz.y) + originalWet_bedRelief(events,lookup,sb,data,descriptors,sbuParams,xz); }

__device__ float originalWet_rockBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalWetParams sbuParams,float2 xz) {
  if(sbuParams.uRockDomain.z <= 0.0f) return -100.0f;
  float2 uv=(xz-sb_sw_xy(sbuParams.uRockDomain))/sb_sw_zw(sbuParams.uRockDomain);
  if(sb_min(uv.x,uv.y)<0.0f || sb_max(uv.x,uv.y)>1.0f) return -100.0f;
  return sb_sample(data,descriptors,0,uv,0.0f).x;
}
__device__ float originalWet_hydraulicBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalWetParams sbuParams,float2 xz) {return sb_max(originalWet_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),originalWet_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,xz));}
__device__ float originalWet_localRockShadow(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalWetParams sbuParams,float3 P,float3 L){
  if(sb_abs(P.x)>65.0f || sb_abs(P.z)>3.3f || P.y>.65f) return 1.0f;
  float shade=1.0f;
  for(int i=0;i<4;i++){
    float h=.045f+.042f*float((i+1)*(i+1));
    float3 q=P+L*(h/sb_max(L.y,.15f));
    shade=sb_min(shade,sb_smoothstep(-.018f,.035f,q.y-originalWet_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,sb_sw_xz(q))));
  }
  return shade;
}

__device__ float2 originalWet_cellXZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalWetParams sbuParams,float2 ij) { return sb_vec2(sbuParams.uDom.x + (float(ij.x) + 0.5f) * sbuParams.uDom.z, sbuParams.uDom.y + (float(ij.y) + 0.5f) * sbuParams.uDom.w); }

__global__ void originalWet(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const originalWetParams* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];originalWetParams sbuParams=params[0];float2 vUv=(sb_vec2(px,py)+0.5f)/sb_vec2(width,height);
  float2 p = sb_vec2(sb_vec2(px,py));
  float h = sb_max(sb_texel(data,descriptors,1,p).x - sb_texel(data,descriptors,2,p).x, 0.0f);
  float2 W = sb_sw_xy(sb_texel(data,descriptors,3,p));
  float cover = sb_smoothstep(0.0008f, 0.004f, h);

  W.x = sb_max(cover, W.x * sb_exp(-sbuParams.uDtS / sbuParams.uFilmTau));
  W.y = sb_max(cover, W.y * sb_exp(-sbuParams.uDtS / 55.0f));
#ifdef originalWet_SWE_EXPLORE

  W.y = sb_max(W.y, 1.0f - sb_smoothstep(0.35f, 0.75f, originalWet_bedHeight(events,lookup,sb,data,descriptors,sbuParams,originalWet_cellXZ(events,lookup,sb,data,descriptors,sbuParams,p))));
#endif
  float oldLine=sb_texel(data,descriptors,3,p).z;
  float B=originalWet_bedHeight(events,lookup,sb,data,descriptors,sbuParams,originalWet_cellXZ(events,lookup,sb,data,descriptors,sbuParams,p));
  float line=sb_mix(B,oldLine,sb_exp(-sbuParams.uDtS/20.0f));
  if(h>.001f) line=sb_max(line,sb_texel(data,descriptors,1,p).x);
  output[0*width*height+index] = sb_vec4(W,line,0.0f);
}
struct originalWetInitParams {
float4 uRockDomain;
float4 uDom;
float2 uN;
};

__device__ float originalWetInit_bedRelief(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalWetInitParams sbuParams,float2 xz) {
  float face = sb_smoothstep(-1.2f, 0.2f, xz.y) * sb_smoothstep(5.0f, 2.5f, xz.y);
  float cusp = 0.003f * sb_sin(xz.x * 5.2f + 0.8f * sb_sin(xz.x * 1.7f)) + 0.0025f * sb_sin(xz.x * 14.5f + 1.3f);
  float lumps = 0.010f * gnoise(events,lookup,sb,xz * 1.3f + 3.1f) + 0.005f * gnoise(events,lookup,sb,xz * 4.1f - 1.3f);
  return (cusp + lumps) * face;
}
__device__ float originalWetInit_bedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalWetInitParams sbuParams,float2 xz) { return bedProfile(events,lookup,sb,xz.y) + originalWetInit_bedRelief(events,lookup,sb,data,descriptors,sbuParams,xz); }

__device__ float originalWetInit_rockBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalWetInitParams sbuParams,float2 xz) {
  if(sbuParams.uRockDomain.z <= 0.0f) return -100.0f;
  float2 uv=(xz-sb_sw_xy(sbuParams.uRockDomain))/sb_sw_zw(sbuParams.uRockDomain);
  if(sb_min(uv.x,uv.y)<0.0f || sb_max(uv.x,uv.y)>1.0f) return -100.0f;
  return sb_sample(data,descriptors,0,uv,0.0f).x;
}
__device__ float originalWetInit_hydraulicBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalWetInitParams sbuParams,float2 xz) {return sb_max(originalWetInit_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),originalWetInit_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,xz));}
__device__ float originalWetInit_localRockShadow(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalWetInitParams sbuParams,float3 P,float3 L){
  if(sb_abs(P.x)>65.0f || sb_abs(P.z)>3.3f || P.y>.65f) return 1.0f;
  float shade=1.0f;
  for(int i=0;i<4;i++){
    float h=.045f+.042f*float((i+1)*(i+1));
    float3 q=P+L*(h/sb_max(L.y,.15f));
    shade=sb_min(shade,sb_smoothstep(-.018f,.035f,q.y-originalWetInit_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,sb_sw_xz(q))));
  }
  return shade;
}

__device__ float2 originalWetInit_cellXZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalWetInitParams sbuParams,float2 ij) { return sb_vec2(sbuParams.uDom.x + (float(ij.x) + 0.5f) * sbuParams.uDom.z, sbuParams.uDom.y + (float(ij.y) + 0.5f) * sbuParams.uDom.w); }

__global__ void originalWetInit(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const originalWetInitParams* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];originalWetInitParams sbuParams=params[0];float2 vUv=(sb_vec2(px,py)+0.5f)/sb_vec2(width,height);
  float2 xz = originalWetInit_cellXZ(events,lookup,sb,data,descriptors,sbuParams,sb_vec2(sb_vec2(px,py)));
  float B = originalWetInit_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz);
  output[0*width*height+index] = sb_vec4(0.0f, 1.0f - sb_smoothstep(0.35f, 0.75f, B), B, 0.0f);
}
struct originalHeightParams {
float4 uRockDomain;
float4 uDom;
float2 uN;
};

__device__ float originalHeight_bedRelief(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalHeightParams sbuParams,float2 xz) {
  float face = sb_smoothstep(-1.2f, 0.2f, xz.y) * sb_smoothstep(5.0f, 2.5f, xz.y);
  float cusp = 0.003f * sb_sin(xz.x * 5.2f + 0.8f * sb_sin(xz.x * 1.7f)) + 0.0025f * sb_sin(xz.x * 14.5f + 1.3f);
  float lumps = 0.010f * gnoise(events,lookup,sb,xz * 1.3f + 3.1f) + 0.005f * gnoise(events,lookup,sb,xz * 4.1f - 1.3f);
  return (cusp + lumps) * face;
}
__device__ float originalHeight_bedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalHeightParams sbuParams,float2 xz) { return bedProfile(events,lookup,sb,xz.y) + originalHeight_bedRelief(events,lookup,sb,data,descriptors,sbuParams,xz); }

__device__ float originalHeight_rockBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalHeightParams sbuParams,float2 xz) {
  if(sbuParams.uRockDomain.z <= 0.0f) return -100.0f;
  float2 uv=(xz-sb_sw_xy(sbuParams.uRockDomain))/sb_sw_zw(sbuParams.uRockDomain);
  if(sb_min(uv.x,uv.y)<0.0f || sb_max(uv.x,uv.y)>1.0f) return -100.0f;
  return sb_sample(data,descriptors,0,uv,0.0f).x;
}
__device__ float originalHeight_hydraulicBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalHeightParams sbuParams,float2 xz) {return sb_max(originalHeight_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),originalHeight_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,xz));}
__device__ float originalHeight_localRockShadow(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalHeightParams sbuParams,float3 P,float3 L){
  if(sb_abs(P.x)>65.0f || sb_abs(P.z)>3.3f || P.y>.65f) return 1.0f;
  float shade=1.0f;
  for(int i=0;i<4;i++){
    float h=.045f+.042f*float((i+1)*(i+1));
    float3 q=P+L*(h/sb_max(L.y,.15f));
    shade=sb_min(shade,sb_smoothstep(-.018f,.035f,q.y-originalHeight_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,sb_sw_xz(q))));
  }
  return shade;
}

__device__ float2 originalHeight_cellXZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalHeightParams sbuParams,float2 ij) { return sb_vec2(sbuParams.uDom.x + (float(ij.x) + 0.5f) * sbuParams.uDom.z, sbuParams.uDom.y + (float(ij.y) + 0.5f) * sbuParams.uDom.w); }

__global__ void originalHeight(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const originalHeightParams* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];originalHeightParams sbuParams=params[0];float2 vUv=(sb_vec2(px,py)+0.5f)/sb_vec2(width,height);
  float2 p = sb_vec2(sb_vec2(px,py));
  float2 xz = originalHeight_cellXZ(events,lookup,sb,data,descriptors,sbuParams,p);
  float w = sb_texel(data,descriptors,1,p).x;
  float2 uv = (sb_vec2(p) + 0.5f) / sb_vec2(sbuParams.uN);
  float2 e = sb_min(uv, 1.0f - uv);
  float wIn = sb_smoothstep(0.0f, 0.04f, sb_min(e.x, e.y));
  output[0*width*height+index] = sb_vec4(sb_mix(0.0f, w, wIn) + brkSurfaceOnly(events,lookup,sb,xz, sb.uTime), 0.0f, 0.0f, 1.0f);
}
struct originalViewParams {
float4 uRockDomain;
float4 uDom;
float2 uN;
};

__device__ float originalView_bedRelief(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalViewParams sbuParams,float2 xz) {
  float face = sb_smoothstep(-1.2f, 0.2f, xz.y) * sb_smoothstep(5.0f, 2.5f, xz.y);
  float cusp = 0.003f * sb_sin(xz.x * 5.2f + 0.8f * sb_sin(xz.x * 1.7f)) + 0.0025f * sb_sin(xz.x * 14.5f + 1.3f);
  float lumps = 0.010f * gnoise(events,lookup,sb,xz * 1.3f + 3.1f) + 0.005f * gnoise(events,lookup,sb,xz * 4.1f - 1.3f);
  return (cusp + lumps) * face;
}
__device__ float originalView_bedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalViewParams sbuParams,float2 xz) { return bedProfile(events,lookup,sb,xz.y) + originalView_bedRelief(events,lookup,sb,data,descriptors,sbuParams,xz); }

__device__ float originalView_rockBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalViewParams sbuParams,float2 xz) {
  if(sbuParams.uRockDomain.z <= 0.0f) return -100.0f;
  float2 uv=(xz-sb_sw_xy(sbuParams.uRockDomain))/sb_sw_zw(sbuParams.uRockDomain);
  if(sb_min(uv.x,uv.y)<0.0f || sb_max(uv.x,uv.y)>1.0f) return -100.0f;
  return sb_sample(data,descriptors,0,uv,0.0f).x;
}
__device__ float originalView_hydraulicBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalViewParams sbuParams,float2 xz) {return sb_max(originalView_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),originalView_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,xz));}
__device__ float originalView_localRockShadow(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalViewParams sbuParams,float3 P,float3 L){
  if(sb_abs(P.x)>65.0f || sb_abs(P.z)>3.3f || P.y>.65f) return 1.0f;
  float shade=1.0f;
  for(int i=0;i<4;i++){
    float h=.045f+.042f*float((i+1)*(i+1));
    float3 q=P+L*(h/sb_max(L.y,.15f));
    shade=sb_min(shade,sb_smoothstep(-.018f,.035f,q.y-originalView_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,sb_sw_xz(q))));
  }
  return shade;
}

__device__ float2 originalView_cellXZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalViewParams sbuParams,float2 ij) { return sb_vec2(sbuParams.uDom.x + (float(ij.x) + 0.5f) * sbuParams.uDom.z, sbuParams.uDom.y + (float(ij.y) + 0.5f) * sbuParams.uDom.w); }

__global__ void originalView(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const originalViewParams* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];originalViewParams sbuParams=params[0];float2 vUv=(sb_vec2(px,py)+0.5f)/sb_vec2(width,height);
  float2 p = sb_vec2(sb_vec2(px,py));
  float3 U = sb_sw_xyz(sb_texel(data,descriptors,1,p));
  float B = sb_texel(data,descriptors,2,p).x;
  float h = sb_max(U.x - B, 0.0f);
  float h4 = h * h * h * h;
  float2 vel = 1.41421356f * h * sb_sw_yz(U) / sb_sqrt(h4 + sb_max(h4, 1.0e-8f));

  float Bfull = originalView_hydraulicBedHeight(events,lookup,sb,data,descriptors,sbuParams,originalView_cellXZ(events,lookup,sb,data,descriptors,sbuParams,p));
  float2 ex = sb_vec2(1, 0), ez = sb_vec2(0, 1), hi = sbuParams.uN - 1;
  float ws = 0.4f * U.x + 0.1f * (sb_texel(data,descriptors,1,sb_clamp(p + ex, sb_vec2(0), hi)).x + sb_texel(data,descriptors,1,sb_clamp(p - ex, sb_vec2(0), hi)).x
                              + sb_texel(data,descriptors,1,sb_clamp(p + ez, sb_vec2(0), hi)).x + sb_texel(data,descriptors,1,sb_clamp(p - ez, sb_vec2(0), hi)).x)
           + 0.05f * (sb_texel(data,descriptors,1,sb_clamp(p + 2 * ex, sb_vec2(0), hi)).x + sb_texel(data,descriptors,1,sb_clamp(p - 2 * ex, sb_vec2(0), hi)).x
                   + sb_texel(data,descriptors,1,sb_clamp(p + 2 * ez, sb_vec2(0), hi)).x + sb_texel(data,descriptors,1,sb_clamp(p - 2 * ez, sb_vec2(0), hi)).x);
  output[0*width*height+index] = sb_vec4(h, vel, sb_mix(Bfull + h, sb_max(ws, Bfull + h * 0.5f), sb_smoothstep(0.04f, 0.12f, h)));
}
struct originalShiftParams {
float4 uRockDomain;
float4 uDom;
float2 uN;
float4 uSwashFarMap;
float4 uSwashFarMapZ;
float4 uSwashFarWin;
float2 uSwashFocus;
int uShift;
int uMode;
};

__device__ float originalShift_bedRelief(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 xz) {
  float face = sb_smoothstep(-1.2f, 0.2f, xz.y) * sb_smoothstep(5.0f, 2.5f, xz.y);
  float cusp = 0.003f * sb_sin(xz.x * 5.2f + 0.8f * sb_sin(xz.x * 1.7f)) + 0.0025f * sb_sin(xz.x * 14.5f + 1.3f);
  float lumps = 0.010f * gnoise(events,lookup,sb,xz * 1.3f + 3.1f) + 0.005f * gnoise(events,lookup,sb,xz * 4.1f - 1.3f);
  return (cusp + lumps) * face;
}
__device__ float originalShift_bedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 xz) { return bedProfile(events,lookup,sb,xz.y) + originalShift_bedRelief(events,lookup,sb,data,descriptors,sbuParams,xz); }

__device__ float originalShift_rockBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 xz) {
  if(sbuParams.uRockDomain.z <= 0.0f) return -100.0f;
  float2 uv=(xz-sb_sw_xy(sbuParams.uRockDomain))/sb_sw_zw(sbuParams.uRockDomain);
  if(sb_min(uv.x,uv.y)<0.0f || sb_max(uv.x,uv.y)>1.0f) return -100.0f;
  return sb_sample(data,descriptors,0,uv,0.0f).x;
}
__device__ float originalShift_hydraulicBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 xz) {return sb_max(originalShift_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),originalShift_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,xz));}
__device__ float originalShift_localRockShadow(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float3 P,float3 L){
  if(sb_abs(P.x)>65.0f || sb_abs(P.z)>3.3f || P.y>.65f) return 1.0f;
  float shade=1.0f;
  for(int i=0;i<4;i++){
    float h=.045f+.042f*float((i+1)*(i+1));
    float3 q=P+L*(h/sb_max(L.y,.15f));
    shade=sb_min(shade,sb_smoothstep(-.018f,.035f,q.y-originalShift_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,sb_sw_xz(q))));
  }
  return shade;
}

__device__ float2 originalShift_cellXZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 ij) { return sb_vec2(sbuParams.uDom.x + (float(ij.x) + 0.5f) * sbuParams.uDom.z, sbuParams.uDom.y + (float(ij.y) + 0.5f) * sbuParams.uDom.w); }

#ifndef originalShift_SWASH_FAR_CORE_DEF
#define originalShift_SWASH_FAR_CORE_DEF 1
#define originalShift_SWASH_FAR_ON 1

__device__ float originalShift_swashFarU(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float x) {
  float g = sbuParams.uSwashFarMap.y * sb_asinh((x - sbuParams.uSwashFarMap.x) / sbuParams.uSwashFarMap.y);
  return 0.5f + 0.5f * g / sbuParams.uSwashFarMap.z;
}
__device__ float originalShift_swashFarXofColumn(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float i) {
  float g = ((i + 0.5f) / sbuParams.uSwashFarMapZ.w - 0.5f) * 2.0f * sbuParams.uSwashFarMap.z;
  return sbuParams.uSwashFarMap.x + sbuParams.uSwashFarMap.y * sb_sinh(g / sbuParams.uSwashFarMap.y);
}
__device__ float2 originalShift_swashFarUV(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 xz) { return sb_vec2(originalShift_swashFarU(events,lookup,sb,data,descriptors,sbuParams,xz.x), (xz.y - sbuParams.uSwashFarMapZ.x) / (sbuParams.uSwashFarMapZ.y * sbuParams.uSwashFarMapZ.z) + 0.5f / sbuParams.uSwashFarMapZ.z); }
__device__ float originalShift_swashVisualFocusX(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams) {

  return sb_clamp(sbuParams.uSwashFocus.x, sbuParams.uSwashFarWin.x - 1.1f, sbuParams.uSwashFarWin.x + 1.1f);
}
__device__ float originalShift_swashFarWeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 xz) {

  float visible = sb_smoothstep(sbuParams.uSwashFarWin.y, sbuParams.uSwashFarWin.z, sb_abs(xz.x - originalShift_swashVisualFocusX(events,lookup,sb,data,descriptors,sbuParams)));

  float bounds = sb_smoothstep(9.95f, 10.14f, sb_abs(xz.x - sbuParams.uSwashFarWin.x));
  return sbuParams.uSwashFarWin.w * sb_max(visible, bounds);
}

__device__ float originalShift_swashFarInZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 uv) { return sb_step(0.0f, uv.y) * sb_step(uv.y, 1.0f); }
__device__ float4 originalShift_swashFar(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 xz, float t) {
  float2 uv = originalShift_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  float4 result = sb_vec4(0.0f);
  if (originalShift_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv) < 0.5f) { float B = bedProfile(events,lookup,sb,xz.y); result = sb_vec4(sb_max(-B, 0.0f), 0.0f, 0.0f, sb_max(B, 0.0f)); }
  else result = sb_sample(data,descriptors,1,uv,0.0f);
  return result;
}
__device__ float4 originalShift_swashFarFoam(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 xz, float t) {
  float2 uv = originalShift_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  return sb_sample(data,descriptors,2,uv,0.0f) * originalShift_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv);
}
__device__ float2 originalShift_swashFarWet(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 xz, float t) {
  float2 uv = originalShift_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  float2 result = sb_vec2(0.0f);
  if (originalShift_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv) < 0.5f) result = sb_vec2(0.0f, 1.0f - sb_smoothstep(0.35f, 0.75f, bedProfile(events,lookup,sb,xz.y)));
  else result = sb_sw_xy(sb_sample(data,descriptors,3,uv,0.0f));
  return result;
}
__device__ float4 originalShift_swashFarLace(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 xz, float t) {
  float2 uv = originalShift_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  float d = sb_sample(data,descriptors,3,uv,0.0f).z * originalShift_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv);
  return sb_vec4(0.0f, -d, 0.0f, -d);
}
__device__ float2 originalShift_swashFarGrad(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 xz, float t) {
  float ex = 0.05f, ez = sbuParams.uSwashFarMapZ.y;
  float wE = originalShift_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz + sb_vec2(ex, 0.0f), t).w, wW = originalShift_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz - sb_vec2(ex, 0.0f), t).w;
  float wN = originalShift_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz + sb_vec2(0.0f, ez), t).w, wS = originalShift_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz - sb_vec2(0.0f, ez), t).w;
  return sb_vec2((wE - wW) / (2.0f * ex), (wN - wS) / (2.0f * ez));
}
__device__ void originalShift_swashFarAll(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftParams sbuParams,float2 xz, float t, float4& view, float4& foam, float2& wet, float4& lace) {
  float2 uv = originalShift_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  float inZ = originalShift_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv);
  float B = bedProfile(events,lookup,sb,xz.y);
  view = inZ > 0.5f ? sb_sample(data,descriptors,1,uv,0.0f) : sb_vec4(sb_max(-B, 0.0f), 0.0f, 0.0f, sb_max(B, 0.0f));
  foam = sb_sample(data,descriptors,2,uv,0.0f) * inZ;
  float4 w = sb_sample(data,descriptors,3,uv,0.0f);
  wet = inZ > 0.5f ? sb_sw_xy(w) : sb_vec2(0.0f, 1.0f - sb_smoothstep(0.35f, 0.75f, B));
  lace = sb_vec4(0.0f, -w.z, 0.0f, -w.z) * inZ;
}
#endif

__global__ void originalShift(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const originalShiftParams* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];originalShiftParams sbuParams=params[0];float2 vUv=(sb_vec2(px,py)+0.5f)/sb_vec2(width,height);
  float2 p = sb_vec2(sb_vec2(px,py));
  float2 q = p + sb_vec2(sbuParams.uShift, 0);
  if (q.x >= 0 && q.x < sbuParams.uN.x) { output[0*width*height+index] = sb_texel(data,descriptors,4,q); return; }
  float2 xz = originalShift_cellXZ(events,lookup,sb,data,descriptors,sbuParams,p);
  float2 qe = sb_vec2(sb_clamp(q.x, 0, sbuParams.uN.x - 1), p.y);
  float2 pe = sb_vec2(qe.x - sbuParams.uShift, p.y);
  bool edge = pe.x >= 0 && pe.x < sbuParams.uN.x;
  float2 xe = originalShift_cellXZ(events,lookup,sb,data,descriptors,sbuParams,pe);
  if (sbuParams.uMode == 0) {
    float B = sb_texel(data,descriptors,5,p).x;
    float4 fv = originalShift_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz, sb.uTime);
    if (!edge) {
      float w = sb_mix(B + fv.x, sb_max(bedProfile(events,lookup,sb,xz.y) + fv.x, B), sb_smoothstep(0.03f, 0.1f, fv.x));
      float h = sb_max(w - B, 0.0f);
      output[0*width*height+index] = sb_vec4(sb_max(w, B), h * fv.y, h * fv.z, 0.0f);
      return;
    }
    float3 Ue = sb_sw_xyz(sb_texel(data,descriptors,4,qe));
    float Be = sb_texel(data,descriptors,5,pe).x;
    float4 fe = originalShift_swashFar(events,lookup,sb,data,descriptors,sbuParams,xe, sb.uTime);
    float he = sb_max(Ue.x - Be, 0.0f);
    float hn = sb_max(he + fv.x - fe.x, 0.0f);
    float w = sb_mix(B + hn, sb_max(Ue.x + fv.w - fe.w, B), sb_smoothstep(0.05f, 0.12f, he));
    float h = sb_max(w - B, 0.0f);
    float2 vel = he > 1.0e-3f ? sb_sw_yz(Ue) / he : sb_vec2(0.0f);
    vel *= sb_min(1.0f, 3.0f / sb_max(length(vel), 1.0e-4f));
    output[0*width*height+index] = sb_vec4(sb_max(w, B), h * vel, 0.0f);
  } else {
    float2 wf = originalShift_swashFarWet(events,lookup,sb,data,descriptors,sbuParams,xz, sb.uTime);
    if (!edge) { output[0*width*height+index] = sb_vec4(wf, sb_max(originalShift_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),originalShift_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz,sb.uTime).w), 0.0f); return; }
    float2 We = sb_sw_xy(sb_texel(data,descriptors,4,qe));
    output[0*width*height+index] = sb_vec4(sb_clamp(We + wf - originalShift_swashFarWet(events,lookup,sb,data,descriptors,sbuParams,xe, sb.uTime), 0.0f, 1.0f), sb_max(originalShift_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),sb_texel(data,descriptors,4,qe).z), 0.0f);
  }
}
struct originalShiftFoamParams {
float4 uRockDomain;
float4 uDom;
float2 uN;
float4 uSwashFarMap;
float4 uSwashFarMapZ;
float4 uSwashFarWin;
float2 uSwashFocus;
int uShift;
};

__device__ float originalShiftFoam_bedRelief(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 xz) {
  float face = sb_smoothstep(-1.2f, 0.2f, xz.y) * sb_smoothstep(5.0f, 2.5f, xz.y);
  float cusp = 0.003f * sb_sin(xz.x * 5.2f + 0.8f * sb_sin(xz.x * 1.7f)) + 0.0025f * sb_sin(xz.x * 14.5f + 1.3f);
  float lumps = 0.010f * gnoise(events,lookup,sb,xz * 1.3f + 3.1f) + 0.005f * gnoise(events,lookup,sb,xz * 4.1f - 1.3f);
  return (cusp + lumps) * face;
}
__device__ float originalShiftFoam_bedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 xz) { return bedProfile(events,lookup,sb,xz.y) + originalShiftFoam_bedRelief(events,lookup,sb,data,descriptors,sbuParams,xz); }

__device__ float originalShiftFoam_rockBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 xz) {
  if(sbuParams.uRockDomain.z <= 0.0f) return -100.0f;
  float2 uv=(xz-sb_sw_xy(sbuParams.uRockDomain))/sb_sw_zw(sbuParams.uRockDomain);
  if(sb_min(uv.x,uv.y)<0.0f || sb_max(uv.x,uv.y)>1.0f) return -100.0f;
  return sb_sample(data,descriptors,0,uv,0.0f).x;
}
__device__ float originalShiftFoam_hydraulicBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 xz) {return sb_max(originalShiftFoam_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),originalShiftFoam_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,xz));}
__device__ float originalShiftFoam_localRockShadow(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float3 P,float3 L){
  if(sb_abs(P.x)>65.0f || sb_abs(P.z)>3.3f || P.y>.65f) return 1.0f;
  float shade=1.0f;
  for(int i=0;i<4;i++){
    float h=.045f+.042f*float((i+1)*(i+1));
    float3 q=P+L*(h/sb_max(L.y,.15f));
    shade=sb_min(shade,sb_smoothstep(-.018f,.035f,q.y-originalShiftFoam_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,sb_sw_xz(q))));
  }
  return shade;
}

__device__ float2 originalShiftFoam_cellXZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 ij) { return sb_vec2(sbuParams.uDom.x + (float(ij.x) + 0.5f) * sbuParams.uDom.z, sbuParams.uDom.y + (float(ij.y) + 0.5f) * sbuParams.uDom.w); }

#ifndef originalShiftFoam_SWASH_FAR_CORE_DEF
#define originalShiftFoam_SWASH_FAR_CORE_DEF 1
#define originalShiftFoam_SWASH_FAR_ON 1

__device__ float originalShiftFoam_swashFarU(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float x) {
  float g = sbuParams.uSwashFarMap.y * sb_asinh((x - sbuParams.uSwashFarMap.x) / sbuParams.uSwashFarMap.y);
  return 0.5f + 0.5f * g / sbuParams.uSwashFarMap.z;
}
__device__ float originalShiftFoam_swashFarXofColumn(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float i) {
  float g = ((i + 0.5f) / sbuParams.uSwashFarMapZ.w - 0.5f) * 2.0f * sbuParams.uSwashFarMap.z;
  return sbuParams.uSwashFarMap.x + sbuParams.uSwashFarMap.y * sb_sinh(g / sbuParams.uSwashFarMap.y);
}
__device__ float2 originalShiftFoam_swashFarUV(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 xz) { return sb_vec2(originalShiftFoam_swashFarU(events,lookup,sb,data,descriptors,sbuParams,xz.x), (xz.y - sbuParams.uSwashFarMapZ.x) / (sbuParams.uSwashFarMapZ.y * sbuParams.uSwashFarMapZ.z) + 0.5f / sbuParams.uSwashFarMapZ.z); }
__device__ float originalShiftFoam_swashVisualFocusX(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams) {

  return sb_clamp(sbuParams.uSwashFocus.x, sbuParams.uSwashFarWin.x - 1.1f, sbuParams.uSwashFarWin.x + 1.1f);
}
__device__ float originalShiftFoam_swashFarWeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 xz) {

  float visible = sb_smoothstep(sbuParams.uSwashFarWin.y, sbuParams.uSwashFarWin.z, sb_abs(xz.x - originalShiftFoam_swashVisualFocusX(events,lookup,sb,data,descriptors,sbuParams)));

  float bounds = sb_smoothstep(9.95f, 10.14f, sb_abs(xz.x - sbuParams.uSwashFarWin.x));
  return sbuParams.uSwashFarWin.w * sb_max(visible, bounds);
}

__device__ float originalShiftFoam_swashFarInZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 uv) { return sb_step(0.0f, uv.y) * sb_step(uv.y, 1.0f); }
__device__ float4 originalShiftFoam_swashFar(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 xz, float t) {
  float2 uv = originalShiftFoam_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  float4 result = sb_vec4(0.0f);
  if (originalShiftFoam_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv) < 0.5f) { float B = bedProfile(events,lookup,sb,xz.y); result = sb_vec4(sb_max(-B, 0.0f), 0.0f, 0.0f, sb_max(B, 0.0f)); }
  else result = sb_sample(data,descriptors,1,uv,0.0f);
  return result;
}
__device__ float4 originalShiftFoam_swashFarFoam(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 xz, float t) {
  float2 uv = originalShiftFoam_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  return sb_sample(data,descriptors,2,uv,0.0f) * originalShiftFoam_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv);
}
__device__ float2 originalShiftFoam_swashFarWet(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 xz, float t) {
  float2 uv = originalShiftFoam_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  float2 result = sb_vec2(0.0f);
  if (originalShiftFoam_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv) < 0.5f) result = sb_vec2(0.0f, 1.0f - sb_smoothstep(0.35f, 0.75f, bedProfile(events,lookup,sb,xz.y)));
  else result = sb_sw_xy(sb_sample(data,descriptors,3,uv,0.0f));
  return result;
}
__device__ float4 originalShiftFoam_swashFarLace(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 xz, float t) {
  float2 uv = originalShiftFoam_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  float d = sb_sample(data,descriptors,3,uv,0.0f).z * originalShiftFoam_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv);
  return sb_vec4(0.0f, -d, 0.0f, -d);
}
__device__ float2 originalShiftFoam_swashFarGrad(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 xz, float t) {
  float ex = 0.05f, ez = sbuParams.uSwashFarMapZ.y;
  float wE = originalShiftFoam_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz + sb_vec2(ex, 0.0f), t).w, wW = originalShiftFoam_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz - sb_vec2(ex, 0.0f), t).w;
  float wN = originalShiftFoam_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz + sb_vec2(0.0f, ez), t).w, wS = originalShiftFoam_swashFar(events,lookup,sb,data,descriptors,sbuParams,xz - sb_vec2(0.0f, ez), t).w;
  return sb_vec2((wE - wW) / (2.0f * ex), (wN - wS) / (2.0f * ez));
}
__device__ void originalShiftFoam_swashFarAll(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalShiftFoamParams sbuParams,float2 xz, float t, float4& view, float4& foam, float2& wet, float4& lace) {
  float2 uv = originalShiftFoam_swashFarUV(events,lookup,sb,data,descriptors,sbuParams,xz);
  float inZ = originalShiftFoam_swashFarInZ(events,lookup,sb,data,descriptors,sbuParams,uv);
  float B = bedProfile(events,lookup,sb,xz.y);
  view = inZ > 0.5f ? sb_sample(data,descriptors,1,uv,0.0f) : sb_vec4(sb_max(-B, 0.0f), 0.0f, 0.0f, sb_max(B, 0.0f));
  foam = sb_sample(data,descriptors,2,uv,0.0f) * inZ;
  float4 w = sb_sample(data,descriptors,3,uv,0.0f);
  wet = inZ > 0.5f ? sb_sw_xy(w) : sb_vec2(0.0f, 1.0f - sb_smoothstep(0.35f, 0.75f, B));
  lace = sb_vec4(0.0f, -w.z, 0.0f, -w.z) * inZ;
}
#endif

__global__ void originalShiftFoam(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const originalShiftFoamParams* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];originalShiftFoamParams sbuParams=params[0];float2 vUv=(sb_vec2(px,py)+0.5f)/sb_vec2(width,height);
  float2 p = sb_vec2(sb_vec2(px,py));
  float2 q = p + sb_vec2(sbuParams.uShift, 0);
  if (q.x >= 0 && q.x < sbuParams.uN.x) { output[0*width*height+index] = sb_texel(data,descriptors,4,q); output[1*width*height+index] = sb_texel(data,descriptors,5,q); return; }
  float2 xz = originalShiftFoam_cellXZ(events,lookup,sb,data,descriptors,sbuParams,p);
  float2 qe = sb_vec2(sb_clamp(q.x, 0, sbuParams.uN.x - 1), p.y);
  float2 pe = sb_vec2(qe.x - sbuParams.uShift, p.y);
  float4 ff = originalShiftFoam_swashFarFoam(events,lookup,sb,data,descriptors,sbuParams,xz, sb.uTime);
  if (pe.x < 0 || pe.x >= sbuParams.uN.x) { output[0*width*height+index] = ff; output[1*width*height+index] = sb_vec4(0.0f); return; }
  output[0*width*height+index] = sb_max(sb_texel(data,descriptors,4,qe) + ff - originalShiftFoam_swashFarFoam(events,lookup,sb,data,descriptors,sbuParams,originalShiftFoam_cellXZ(events,lookup,sb,data,descriptors,sbuParams,pe), sb.uTime), 0.0f);
  output[1*width*height+index] = sb_texel(data,descriptors,5,qe);
}
struct originalFarParamParams {
float4 uRockDomain;
float4 uSwashFarMap;
float4 uSwashFarMapZ;
float4 uSwashFarWin;
float2 uSwashFocus;
};

__device__ float originalFarParam_bedRelief(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarParamParams sbuParams,float2 xz) {
  float face = sb_smoothstep(-1.2f, 0.2f, xz.y) * sb_smoothstep(5.0f, 2.5f, xz.y);
  float cusp = 0.003f * sb_sin(xz.x * 5.2f + 0.8f * sb_sin(xz.x * 1.7f)) + 0.0025f * sb_sin(xz.x * 14.5f + 1.3f);
  float lumps = 0.010f * gnoise(events,lookup,sb,xz * 1.3f + 3.1f) + 0.005f * gnoise(events,lookup,sb,xz * 4.1f - 1.3f);
  return (cusp + lumps) * face;
}
__device__ float originalFarParam_bedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarParamParams sbuParams,float2 xz) { return bedProfile(events,lookup,sb,xz.y) + originalFarParam_bedRelief(events,lookup,sb,data,descriptors,sbuParams,xz); }

__device__ float originalFarParam_rockBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarParamParams sbuParams,float2 xz) {
  if(sbuParams.uRockDomain.z <= 0.0f) return -100.0f;
  float2 uv=(xz-sb_sw_xy(sbuParams.uRockDomain))/sb_sw_zw(sbuParams.uRockDomain);
  if(sb_min(uv.x,uv.y)<0.0f || sb_max(uv.x,uv.y)>1.0f) return -100.0f;
  return sb_sample(data,descriptors,0,uv,0.0f).x;
}
__device__ float originalFarParam_hydraulicBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarParamParams sbuParams,float2 xz) {return sb_max(originalFarParam_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),originalFarParam_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,xz));}
__device__ float originalFarParam_localRockShadow(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarParamParams sbuParams,float3 P,float3 L){
  if(sb_abs(P.x)>65.0f || sb_abs(P.z)>3.3f || P.y>.65f) return 1.0f;
  float shade=1.0f;
  for(int i=0;i<4;i++){
    float h=.045f+.042f*float((i+1)*(i+1));
    float3 q=P+L*(h/sb_max(L.y,.15f));
    shade=sb_min(shade,sb_smoothstep(-.018f,.035f,q.y-originalFarParam_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,sb_sw_xz(q))));
  }
  return shade;
}

__device__ float originalFarParam_swashFarU(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarParamParams sbuParams,float x) {
  float g = sbuParams.uSwashFarMap.y * sb_asinh((x - sbuParams.uSwashFarMap.x) / sbuParams.uSwashFarMap.y);
  return 0.5f + 0.5f * g / sbuParams.uSwashFarMap.z;
}
__device__ float originalFarParam_swashFarXofColumn(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarParamParams sbuParams,float i) {
  float g = ((i + 0.5f) / sbuParams.uSwashFarMapZ.w - 0.5f) * 2.0f * sbuParams.uSwashFarMap.z;
  return sbuParams.uSwashFarMap.x + sbuParams.uSwashFarMap.y * sb_sinh(g / sbuParams.uSwashFarMap.y);
}
__device__ float2 originalFarParam_swashFarUV(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarParamParams sbuParams,float2 xz) { return sb_vec2(originalFarParam_swashFarU(events,lookup,sb,data,descriptors,sbuParams,xz.x), (xz.y - sbuParams.uSwashFarMapZ.x) / (sbuParams.uSwashFarMapZ.y * sbuParams.uSwashFarMapZ.z) + 0.5f / sbuParams.uSwashFarMapZ.z); }
__device__ float originalFarParam_swashVisualFocusX(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarParamParams sbuParams) {

  return sb_clamp(sbuParams.uSwashFocus.x, sbuParams.uSwashFarWin.x - 1.1f, sbuParams.uSwashFarWin.x + 1.1f);
}
__device__ float originalFarParam_swashFarWeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarParamParams sbuParams,float2 xz) {

  float visible = sb_smoothstep(sbuParams.uSwashFarWin.y, sbuParams.uSwashFarWin.z, sb_abs(xz.x - originalFarParam_swashVisualFocusX(events,lookup,sb,data,descriptors,sbuParams)));

  float bounds = sb_smoothstep(9.95f, 10.14f, sb_abs(xz.x - sbuParams.uSwashFarWin.x));
  return sbuParams.uSwashFarWin.w * sb_max(visible, bounds);
}

__global__ void originalFarParam(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const originalFarParamParams* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];originalFarParamParams sbuParams=params[0];float2 vUv=(sb_vec2(px,py)+0.5f)/sb_vec2(width,height);
  float2 ij = sb_vec2(sb_vec2(px,py));
  float x = originalFarParam_swashFarXofColumn(events,lookup,sb,data,descriptors,sbuParams,float(ij.x));
  int k = ij.y / 2;
  if (k >= sb.uEvtCount) { output[0*width*height+index] = sb_vec4(-1.0e4f, 0.0f, -2.0f, 0.0f); return; }

  float dxc = originalFarParam_swashFarXofColumn(events,lookup,sb,data,descriptors,sbuParams,float(ij.x) + 0.5f) - originalFarParam_swashFarXofColumn(events,lookup,sb,data,descriptors,sbuParams,float(ij.x) - 0.5f);
  int ns = dxc > 0.05f ? 4 : 1;
  float ti = 0.0f, zI = 0.0f, H = 0.0f;
  for (int j = 0; j < 4 + sb_min(sb.uEvtCount, 0); j++) {
    if (j >= ns) break;
    Brk bj = brkAt(events,lookup,sb,k, x + dxc * ((float(j) + 0.5f) / float(ns) - 0.5f));
    ti += bj.ti; zI += bj.zI; H += bj.H;
  }
  ti /= float(ns); zI /= float(ns); H /= float(ns);
  float4 A = events[0*MAX_EVENTS+(k)], C = events[2*MAX_EVENTS+(k)], E4 = events[4*MAX_EVENTS+(k)];
  float sd = A.w, gB = sb_max(events[6*MAX_EVENTS+(k)].x, 0.0f), str = sb_max(events[1*MAX_EVENTS+(k)].w, 0.0f), style = C.w, spill = E4.z;

  if (ij.y - 2 * k == 1) { output[0*width*height+index] = sb_vec4(sd, style, H, spill); return; }
  float ph = 2.5f * sb_sin(0.37f * x + sd * 5.1f + 1.7f * sb_sin(0.13f * x));
  float lp1 = 1.0f - sb_smoothstep(0.2f, 0.45f, 5.236f * dxc / 6.2832f), lp2 = 1.0f - sb_smoothstep(0.2f, 0.45f, 14.96f * dxc / 6.2832f);
  float lob = 0.12f * sb_sin(x * 1.186f + sd * 41.3f + 0.4f * ph) + 0.07f * sb_sin(x * 2.732f + sd * 17.9f + 1.3f + 0.8f * ph)
            + (0.06f * lp1 * sb_sin(x * 5.236f + sd * 37.1f + ph) + 0.025f * lp2 * sb_sin(x * 14.96f + sd * 91.7f + 0.7f * sb_sin(2.1f * x) + 1.9f * ph))
              * (0.45f + 0.9f * (0.5f + 0.5f * sb_sin(0.29f * x + sd * 3.3f) * sb_sin(0.71f * x + 1.1f)));

  float Rm = (4.2f * H + 0.35f) * powf(gB, 0.7f) * (0.85f + 0.15f * sb_smoothstep(0.3f, 0.8f, style)) * str * (1.0f + lob) * (1.0f + brkRunupMod(events,lookup,sb,x, sd));
  float s = H / 0.45f;
  float E = s * s * str * sb_mix(0.6f, 1.0f, style) * gB;
  output[0*width*height+index] = sb_vec4(ti, Rm, zI, E);
}
struct originalFarModelParams {
float4 uRockDomain;
float4 uSwashFarMap;
float4 uSwashFarMapZ;
float4 uSwashFarWin;
float2 uSwashFocus;
float4 uSwashFarK;
float4 uSwashFarK2;
int uFarSlots;
};

__device__ float originalFarModel_bedRelief(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarModelParams sbuParams,float2 xz) {
  float face = sb_smoothstep(-1.2f, 0.2f, xz.y) * sb_smoothstep(5.0f, 2.5f, xz.y);
  float cusp = 0.003f * sb_sin(xz.x * 5.2f + 0.8f * sb_sin(xz.x * 1.7f)) + 0.0025f * sb_sin(xz.x * 14.5f + 1.3f);
  float lumps = 0.010f * gnoise(events,lookup,sb,xz * 1.3f + 3.1f) + 0.005f * gnoise(events,lookup,sb,xz * 4.1f - 1.3f);
  return (cusp + lumps) * face;
}
__device__ float originalFarModel_bedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarModelParams sbuParams,float2 xz) { return bedProfile(events,lookup,sb,xz.y) + originalFarModel_bedRelief(events,lookup,sb,data,descriptors,sbuParams,xz); }

__device__ float originalFarModel_rockBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarModelParams sbuParams,float2 xz) {
  if(sbuParams.uRockDomain.z <= 0.0f) return -100.0f;
  float2 uv=(xz-sb_sw_xy(sbuParams.uRockDomain))/sb_sw_zw(sbuParams.uRockDomain);
  if(sb_min(uv.x,uv.y)<0.0f || sb_max(uv.x,uv.y)>1.0f) return -100.0f;
  return sb_sample(data,descriptors,0,uv,0.0f).x;
}
__device__ float originalFarModel_hydraulicBedHeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarModelParams sbuParams,float2 xz) {return sb_max(originalFarModel_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz),originalFarModel_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,xz));}
__device__ float originalFarModel_localRockShadow(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarModelParams sbuParams,float3 P,float3 L){
  if(sb_abs(P.x)>65.0f || sb_abs(P.z)>3.3f || P.y>.65f) return 1.0f;
  float shade=1.0f;
  for(int i=0;i<4;i++){
    float h=.045f+.042f*float((i+1)*(i+1));
    float3 q=P+L*(h/sb_max(L.y,.15f));
    shade=sb_min(shade,sb_smoothstep(-.018f,.035f,q.y-originalFarModel_rockBedHeight(events,lookup,sb,data,descriptors,sbuParams,sb_sw_xz(q))));
  }
  return shade;
}

__device__ float originalFarModel_swashFarU(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarModelParams sbuParams,float x) {
  float g = sbuParams.uSwashFarMap.y * sb_asinh((x - sbuParams.uSwashFarMap.x) / sbuParams.uSwashFarMap.y);
  return 0.5f + 0.5f * g / sbuParams.uSwashFarMap.z;
}
__device__ float originalFarModel_swashFarXofColumn(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarModelParams sbuParams,float i) {
  float g = ((i + 0.5f) / sbuParams.uSwashFarMapZ.w - 0.5f) * 2.0f * sbuParams.uSwashFarMap.z;
  return sbuParams.uSwashFarMap.x + sbuParams.uSwashFarMap.y * sb_sinh(g / sbuParams.uSwashFarMap.y);
}
__device__ float2 originalFarModel_swashFarUV(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarModelParams sbuParams,float2 xz) { return sb_vec2(originalFarModel_swashFarU(events,lookup,sb,data,descriptors,sbuParams,xz.x), (xz.y - sbuParams.uSwashFarMapZ.x) / (sbuParams.uSwashFarMapZ.y * sbuParams.uSwashFarMapZ.z) + 0.5f / sbuParams.uSwashFarMapZ.z); }
__device__ float originalFarModel_swashVisualFocusX(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarModelParams sbuParams) {

  return sb_clamp(sbuParams.uSwashFocus.x, sbuParams.uSwashFarWin.x - 1.1f, sbuParams.uSwashFarWin.x + 1.1f);
}
__device__ float originalFarModel_swashFarWeight(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarModelParams sbuParams,float2 xz) {

  float visible = sb_smoothstep(sbuParams.uSwashFarWin.y, sbuParams.uSwashFarWin.z, sb_abs(xz.x - originalFarModel_swashVisualFocusX(events,lookup,sb,data,descriptors,sbuParams)));

  float bounds = sb_smoothstep(9.95f, 10.14f, sb_abs(xz.x - sbuParams.uSwashFarWin.x));
  return sbuParams.uSwashFarWin.w * sb_max(visible, bounds);
}

__device__ float originalFarModel_sfApexZ(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarModelParams sbuParams,float tn) { float d = sb_max(-0.33f - tn, 0.0f); return -0.48f - 3.3f * d * (1.0f + 0.12f * d); }
__device__ float originalFarModel_sfFaceL(const float4* events,const float4* lookup,SBUniforms sb,const float4* data,const float4* descriptors,originalFarModelParams sbuParams,float tn) { return sb_mix(0.4f, 4.0f, sb_smoothstep(-0.4f, -2.0f, tn)); }
__global__ void originalFarModel(const float4* events,const float4* lookup,const SBUniforms* uniforms,const float4* data,const float4* descriptors,const originalFarModelParams* params,float4* output,int width,int height){
 int px=blockIdx.x*blockDim.x+threadIdx.x,py=blockIdx.y*blockDim.y+threadIdx.y;if(px>=width||py>=height)return;int index=py*width+px;SBUniforms sb=uniforms[0];originalFarModelParams sbuParams=params[0];float2 vUv=(sb_vec2(px,py)+0.5f)/sb_vec2(width,height);
  float2 ij = sb_vec2(sb_vec2(px,py));
  float x = originalFarModel_swashFarXofColumn(events,lookup,sb,data,descriptors,sbuParams,float(ij.x));
  float z = sbuParams.uSwashFarMapZ.x + float(ij.y) * sbuParams.uSwashFarMapZ.y;
  float2 xz = sb_vec2(x, z);
  float t = sb.uTime;
  float B = bedProfile(events,lookup,sb,z);

  if (sb_abs(x - sbuParams.uSwashFarWin.x) < sbuParams.uSwashFarWin.y - 1.4f) {
    output[0*width*height+index] = sb_vec4(sb_max(-B, 0.0f), 0.0f, 0.0f, sb_max(B, 0.0f));
    output[1*width*height+index] = sb_vec4(0.0f); output[2*width*height+index] = sb_vec4(0.0f, 1.0f - sb_smoothstep(0.35f, 0.75f, B), 0.0f, 0.0f);
    return;
  }
  float a = sbuParams.uSwashFarK.x, zr0 = sbuParams.uSwashFarK.w, zt = sbuParams.uSwashFarK2.w;

  float dxc = originalFarModel_swashFarXofColumn(events,lookup,sb,data,descriptors,sbuParams,float(ij.x) + 0.5f) - originalFarModel_swashFarXofColumn(events,lookup,sb,data,descriptors,sbuParams,float(ij.x) - 0.5f);
  float rag = 0.022f * gnoise(events,lookup,sb,sb_vec2(x * 13.0f, t * 0.6f)) * (1.0f - sb_smoothstep(0.1f, 0.22f, dxc))
            + 0.012f * gnoise(events,lookup,sb,sb_vec2(x * 41.0f + 3.7f, t * 1.3f)) * (1.0f - sb_smoothstep(0.03f, 0.07f, dxc))
            + 0.05f * gnoise(events,lookup,sb,sb_vec2(x * 2.3f + 1.9f, t * 0.15f)) * (1.0f - sb_smoothstep(0.6f, 1.2f, dxc));
  float h = 0.0f, vS = 0.0f, wS = 1.0e-9f, cov = 0.0f, since = 1.0e3f, keepF = 1.0f, dS = 0.0f, dW = 0.0f;
  float4 F = sb_vec4(0.0f);

  for (int k = sbuParams.uFarSlots - 1; k >= 0; k--) {
    float4 P = sb_texel(data,descriptors,1,sb_vec2(ij.x, 2 * k));
    float tau = t - P.x, Rm = P.y, zI = P.z, E = P.w;
    if (E <= 1.0e-4f || Rm <= 0.01f) continue;
    float rs = sb_clamp(powf(E, 0.25f), 0.6f, 1.3f);
    if (tau < 0.0f) {

      float tn = tau / rs;
      if (tn > -2.4f) {
        float zc = zI + originalFarModel_sfApexZ(events,lookup,sb,data,descriptors,sbuParams,tn) * rs * rs, L = originalFarModel_sfFaceL(events,lookup,sb,data,descriptors,sbuParams,tn) * rs * rs;
        float face = sb_smoothstep(zc - 0.8f, zc - 0.2f, z) * (1.0f - sb_smoothstep(zc + 0.6f * L, zc + L, z));
        keepF = sb_min(keepF, 1.0f - face * sb_smoothstep(-2.4f, -1.5f, tn) * sb_step(z, 0.6f));
      }
      continue;
    }
    if (tau > 16.0f || (z > Rm + 0.3f && z > 0.3f)) continue;
    float U0 = sb_sqrt(2.0f * a * Rm);
    float tc = sbuParams.uSwashFarK.y * rs, tup = U0 / a, tm = tc + tup;
    float zr = sb_min(zr0, 0.8f * Rm);
    bool adv = tau < tm;
    float S;
    if (tau < tc) S = zI * powf(1.0f - tau / tc, 1.5f);
    else if (adv) { float s = tau - tc; S = U0 * s - 0.5f * a * s * s; }
    else S = zr + (Rm - zr) * sb_exp(-powf((tau - tm) / sbuParams.uSwashFarK.z, 1.5f));
    S += rag * sb_smoothstep(-0.3f, 0.3f, S);

    float hb = sbuParams.uSwashFarK2.x * rs * rs;
    float rise = sb_smoothstep(tc - 0.7f * rs, tc + 0.1f * rs, tau);
    float sheet = (0.035f + 0.19f * sb_exp(-sb_max(tau - tc - 0.3f, 0.0f) / 1.4f)) * rs * (adv ? rise : 0.4f + 0.6f * sb_exp(-(tau - tm) / 0.8f));
    float Sp = sb_max(S - zt, 1.0e-3f);
    float q = sb_clamp((z - zt) / Sp, 0.0f, 1.0f);
    float p = adv ? 0.6f + 0.15f * sb_smoothstep(0.0f, 0.5f, (tau - tc) / tup) : 0.9f;
    float lens = (z < S && z >= zt && S > zt) ? sheet * powf(1.0f - q, p) * (1.0f - powf(q, 6.0f)) : 0.0f;
    float eta;
    if (tau < tc) eta = hb * sb_smoothstep(S + 0.12f, S - 0.2f, z) * sb_exp(sb_min(z - S, 0.0f) / 0.9f);
    else {
      float lvl = hb * rise * sb_exp(-sb_max(tau - tc - 0.3f * rs, 0.0f) / 0.9f);
      eta = lvl * sb_exp(sb_min(z, 0.0f) / 1.1f) - (adv ? 0.0f : 0.03f * rs * sb_smoothstep(0.0f, 0.8f, tau - tm)) * sb_exp(sb_min(z, 0.0f) / 0.7f);
    }
    float he = sb_max(lens, z < 0.3f ? eta - B : 0.0f);

    float ve;
    if (tau < tc) ve = -zI * 1.5f / tc * sb_sqrt(sb_max(1.0f - tau / tc, 0.0f)) * sb_smoothstep(zI - 0.5f, S, z) * sb_step(z, S + 0.05f);
    else if (adv) {
      float s = tau - tc;
      float vf = 1.2f * rs * (1.0f - powf(s / tup, 3.0f)) + 0.4f * sb_max(U0 - a * s - 1.2f * rs, 0.0f);
      ve = vf * powf(sb_clamp((z - zt) / Sp, 0.0f, 1.0f), 0.8f) * sb_step(z, S + 0.05f) - 0.12f * sb_step(z, zt);
    } else {
      float Vb = 1.4f * rs * (1.0f - sb_exp(-(tau - tm) / 0.6f));
      ve = -Vb * powf(sb_clamp(1.0f - (z - zt) / Sp, 0.0f, 1.0f), 0.7f) * sb_step(zt - 0.2f, z) * sb_step(z, S) - 0.12f * sb_step(z, zt);
    }
    float keep = 1.0f - cov;
    h = sb_max(h, he);
    float w2 = he * he * (keep + 0.05f);
    vS += w2 * ve; wS += w2;
    float cover = adv ? sb_smoothstep(S + 0.02f, S - 0.25f, z) : sb_smoothstep(S + 0.02f, S - 0.1f, z);
    if (tau < tc) cover *= sb_step(zI - 0.8f, z);

    float Dz = adv ? 0.8f * (S - Rm) : S - Rm;
    dS += Dz * cover * keep; dW += cover * keep;

    float tArr = z < 0.0f ? tc * (1.0f - powf(sb_clamp(z / sb_min(zI, -0.1f), 0.0f, 1.0f), 0.6667f))
                         : tc + (U0 - sb_sqrt(sb_max(U0 * U0 - 2.0f * a * sb_clamp(z, 0.0f, 0.999f * Rm), 0.0f))) / a;
    float tLeft = z > zr ? tm + sbuParams.uSwashFarK.z * powf(-sb_log(sb_clamp((z - zr) / sb_max(Rm - zr, 1.0e-3f), 1.0e-6f, 1.0f)), 0.6667f) : 1.0e9f;
    bool reached = z < Rm && tau > tArr;
    if (reached) since = sb_min(since, sb_max(tau - tLeft, 0.0f));

    float sd = sb_texel(data,descriptors,1,sb_vec2(ij.x, 2 * k + 1)).x * 37.0f;
    float2 pq = sb_vec2(x / 2.2f, (z - Dz) / 0.8f) + sb_vec2(sd, -sd * 0.7f);
    float patchF = sb_smoothstep(-0.6f, 0.6f, gnoise(events,lookup,sb,pq) + 0.55f * gnoise(events,lookup,sb,pq * 2.4f + 5.1f));
    float patchP = sb_smoothstep(-0.7f, 0.7f, gnoise(events,lookup,sb,sb_vec2(x / 3.1f + sd * 1.3f, tau * 0.25f)));
    float ageF = sb_max(tau - tArr, 0.0f);
    float zp = zI + 0.15f + 0.1f * sb_min(tau, 2.0f);
    float dp = (z - zp) / (0.6f + 0.15f * tau);
    float Rp = 2.3f * sb_mix(0.45f, 1.2f, patchP) * sb_exp(-sb_max(tau - 1.2f, 0.0f) / 0.9f) * sb_smoothstep(0.1f, 0.9f, tau) * sb_exp(-0.5f * dp * dp);

    float Rb = 1.25f * sb_mix(0.3f, 1.3f, patchF) * sb_exp(-sb_max(tau - tc - 0.5f * rs, 0.0f) / 0.5f) * cover * sb_smoothstep(-0.95f, -0.65f, z) * sb_smoothstep(0.0f, 0.35f, tau);
    float df = (S - z) / 0.4f;
    float Rf = adv ? 0.8f * sb_mix(0.6f, 1.15f, patchF) * sb_exp(-df * df) * cover * sb_smoothstep(0.0f, 0.3f, tau - tc + 0.3f) * (1.0f - sb_smoothstep(0.6f, 1.0f, (tau - tc) / tup)) : 0.0f;

    float zj = zt - 0.12f + 0.08f * gnoise(events,lookup,sb,sb_vec2(x * 1.7f + sd, tau * 0.8f));
    float dj = (z - zj) / 0.1f;
    float Rt = adv ? 0.0f : 0.45f * sb_smoothstep(0.2f, 1.0f, (tau - tm) / 0.6f) * sb_smoothstep(0.35f, 0.8f, patchP + 0.3f) * sb_exp(-dj * dj);
    float Eg = sb_sqrt(sb_min(E, 1.5f));
    float lg = (z - zp) / 1.0f;
    float G = sb_max(1.3f * sb_mix(0.45f, 1.2f, patchF) * sb_smoothstep(0.0f, 0.5f, ageF) * sb_exp(-sb_max(tau - tc - 1.6f * rs, 0.0f) / 2.2f) * ((reached && tau < tLeft + 0.3f) ? 1.0f : 0.0f),
                  1.8f * sb_mix(0.5f, 1.1f, patchP) * sb_smoothstep(0.2f, 0.8f, tau) * sb_exp(-sb_max(tau - 1.5f, 0.0f) / 2.2f) * sb_exp(-lg * lg));
    float sea = sb_smoothstep(zI - 1.4f, zI - 0.2f, z);
    float Mk = (0.25f + 0.55f * sb_smoothstep(0.0f, 0.4f, ageF) * sb_exp(-sb_max(tau - tc - 1.2f * rs, 0.0f) / 1.25f)) * sb_max(reached ? 1.0f : 0.0f, sb_step(z, 0.0f) * sea);
    float Kp = 1.6f * sb_exp(-sb_max(tau - 1.3f, 0.0f) / 1.2f) * sb_smoothstep(0.1f, 0.9f, tau) * sb_exp(-0.5f * (z - zp) * (z - zp) / 0.36f);
    float Kl = 2.8f * (adv ? sb_exp(-sb_max(tau - tc - 1.4f, 0.0f) / 1.2f) : 0.55f + 0.45f * sb_exp(-(tau - tm))) * cover * sb_step(zt - 0.3f, z);
    Kl = sb_max(Kl, 2.0f * Rt);
    F = sb_max(F, sb_vec4(E * (Rp + Rb + Rf) + Eg * Rt, sb_min(G * Eg, 2.0f), sb_max(Kp, Kl) * Eg, Mk * sb_sqrt(sb_min(E, 1.2f))) * keep);
    cov = sb_max(cov, cover);
  }

  h = sb_max(h, sbuParams.uSwashFarK2.y * sb_smoothstep(zr0 + 0.1f, zr0 - 0.3f, z) * sb_step(-0.3f, z));
  float dr = sb_smoothstep(0.0005f, 0.002f, h);
  float film = h > 0.001f ? 1.0f : sb_exp(-since / sbuParams.uSwashFarK2.z);
  float damp = sb_max(h > 0.001f ? 1.0f : sb_exp(-since / 55.0f), 1.0f - sb_smoothstep(0.35f, 0.75f, B));

  float lowF = (1.0f - sb_smoothstep(0.002f, 0.008f, h)) * sb_smoothstep(12.0f, 18.0f, sb_abs(x - originalFarModel_swashVisualFocusX(events,lookup,sb,data,descriptors,sbuParams)));
  float w = sb_mix(originalFarModel_bedHeight(events,lookup,sb,data,descriptors,sbuParams,xz) + h, B + h, sb_smoothstep(0.04f, 0.12f, h)) - 0.03f * lowF;
  output[0*width*height+index] = sb_vec4(h, 0.1f * sb_smoothstep(0.003f, 0.02f, h), vS / wS, w);

  output[1*width*height+index] = sb_vec4(F.x * keepF, F.y * keepF, F.z, F.w * keepF) * dr * sb_smoothstep(sbuParams.uSwashFarMapZ.x, sbuParams.uSwashFarMapZ.x + 0.6f, z);
  output[2*width*height+index] = sb_vec4(film, damp, dS / sb_max(dW, 1.0e-3f) * sb_min(dW, 1.0f), 0.0f);
}
