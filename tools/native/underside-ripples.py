"""Probe the production underside normal block across wave slopes on the GPU."""
from pathlib import Path
import json,sys,numpy as np,moderngl as gl
source=Path(sys.argv[1]).read_text()
a=source.index('  float faceRipple=');b=source.index('\n  if(dot(N,V)<0.)',a)
block=source[a:b]
# A controlled FFT slope isolates the normal mapping from texture animation.
block=block.replace('vec4 lean=oceanSlopeLEAN(rippleUV,vXZ0.z,rippleWeight,0.);','vec4 lean=vec4(vec2(.10,.08)*rippleWeight,0.,0.);')
ctx=gl.create_standalone_context(backend='egl')
vs='''#version 330
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0.,1.);}'''
fs='''#version 330
out vec4 result;
void main(){
float angle=(gl_FragCoord.x-.5)/90.*1.57079632679;
float az=(gl_FragCoord.y-.5)/32.*6.28318530718;
vec3 macro=vec3(sin(angle)*sin(az),cos(angle),sin(angle)*cos(az));
float slope=sin(angle)/max(cos(angle),.00001);
vec4 vInfo=vec4(0.,0.,0.,.7/(1.+1.5*slope*slope));
vec3 vXZ0=vec3(2.,-14.,1.),P=vec3(2.,.7,-14.);float waterColumn=1.;
'''+block+'''
vec2 oldLean=vec2(.10,.08)*vInfo.w;
vec3 oldN=normalize(vec3(-macro.x/max(macro.y,.012)+oldLean.x,-1.,-macro.z/max(macro.y,.012)+oldLean.y));
float newAngle=acos(clamp(dot(N,-macro),-1.,1.));
float oldAngle=acos(clamp(dot(oldN,-macro),-1.,1.));
result=vec4(newAngle,oldAngle,length(N-oldN),length(N));
}'''
p=ctx.program(vertex_shader=vs,fragment_shader=fs)
fbo=ctx.framebuffer([ctx.texture((91,33),4,dtype='f4')]);fbo.use();ctx.vertex_array(p,[]).render(vertices=3)
v=np.frombuffer(fbo.read(components=4,dtype='f4'),dtype='f4').reshape(33,91,4)
assert np.isfinite(v).all()
assert np.max(np.abs(v[:,:,3]-1))<1e-5
assert np.max(v[:,:25,2])<1e-6,'Offshore low-angle mapping changed'
assert np.min(v[:,70:89,0])>.07,'Steep face lost ripple amplitude'
# Old mapping suppresses perturbations as the geometric slope grows.
assert np.max(v[:,70:89,1])<.01,'Probe does not expose the previous flattening'
result={'finiteNormalSamples':int(v.shape[0]*v.shape[1]),'maxUnitLengthError':float(np.max(np.abs(v[:,:,3]-1))),'maximumOffshoreFlatNormalChange':float(v[:,:25,2].max()),'steepFaceRippleDegrees':[float(np.degrees(v[:,70:89,0].min())),float(np.degrees(v[:,70:89,0].max()))],'previousSteepFaceMaximumRippleDegrees':float(np.degrees(v[:,70:89,1].max()))}
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
