// Both the distant swell and nearshore breaker use this same transport table.
import { WATER_LOOKUP_GLSL } from '../water/WaterLookups.js';
export const SWELL_TRANSPORT = /* glsl */ `
#ifdef OPT_EXPLORE
${WATER_LOOKUP_GLSL}
uniform float uSwellMinTime;
vec4 swellTravel(float z) {
  float u=clamp((z+900.0)/899.0,0.0,1.0);
  return textureLod(uWaterLookup,vec2((u*4095.0+.5)/4096.0,.125),0.0);
}
vec4 swellPosition(float time) {
  float u=clamp((time-uSwellMinTime)/(-uSwellMinTime),0.0,1.0);
  return textureLod(uWaterLookup,vec2((u*4095.0+.5)/4096.0,.375),0.0);
}
#endif
`;
export const SWELL_SAMPLE = /* glsl */ `
#ifdef OPT_EXPLORE
uniform sampler2D uSwellField;
vec3 swellAt(vec2 xz) {
  float v=log(1.0+max(-xz.y-6.0,0.0)/8.0)/log(81.5);
  vec2 uv=vec2((xz.x+512.0)/1024.0,v);
  float fade=(1.0-smoothstep(420.0,512.0,abs(xz.x)))*smoothstep(-650.0,-460.0,xz.y);
  return textureLod(uSwellField,uv,0.0).rgb*fade;
}
#endif
`;
