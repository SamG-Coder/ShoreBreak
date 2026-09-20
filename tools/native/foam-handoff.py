"""Exercise the actual exported GLSL handoff and foam-stretch expressions."""
import json, sys
from pathlib import Path
import moderngl as gl
import numpy as np

source = json.loads(Path(sys.argv[1]).read_text())
ctx = gl.create_standalone_context(backend='egl')
width = 4096
target = ctx.framebuffer([ctx.texture((width, 2), 4, dtype='f4')])
vertex = '''#version 330
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
gl_Position=vec4(p*2.-1.,0.,1.);}'''
prefix = '#version 330\nout vec4 result;\nuniform vec2 uRange;\n' + source['map']
point = 'float x=mix(uRange.x,uRange.y,gl_FragCoord.x/4096.);'
program = ctx.program(vertex_shader=vertex, fragment_shader=prefix + '''
void main(){''' + point + '''
float w=swashFarWeight(vec2(x,0.));
float legacy=smoothstep(7.9,9.9,abs(x-uSwashFarWin.x));
result=vec4(w,legacy,0.,1.);}''')
vao = ctx.vertex_array(program, [])

def draw(center, focus, enabled=1):
    target.use()
    program['uSwashFarWin'].value=(center,5.8,8.8,enabled)
    program['uSwashFocus'].value=(focus,0)
    program['uRange'].value=(-12,12)
    vao.render(vertices=3)
    return np.frombuffer(target.read(components=4,dtype='f4'),dtype='f4').reshape(2,width,4)

metrics = {}
for sign in [-1,1]:
    a=draw(0,sign*(1-0.00002)); b=draw(sign*1.02,sign*(1+0.00002))
    delta=np.max(np.abs(b-a),axis=(0,1))
    assert delta[0]<0.0001, delta
    assert delta[1]>.6, 'Probe must detect the legacy one-metre pop'
    metrics[str(sign)]={'continuousMaxDelta':float(delta[0]),'legacyMaxDelta':float(delta[1])}
xs=np.linspace(-12,12,width,endpoint=False)+12/width
for focus in [-20,-1.1,0,1.1,20]:
    values=draw(0,focus)[0,:,0]
    assert np.all(values[np.abs(xs)>=10.15]>.9999), 'Read from allocation-edge fringe'
    assert np.all(values[np.abs(xs)<4.4]<1e-6), 'Far data requested inside skipped bake region'
assert not draw(0,0,0)[:,: ,0].any(), 'Disabled handoff must remain zero'

# The two fields describe equal, unstretched motion but different origins.
# Use the production derivative correction verbatim, rather than a CPU copy.
foam=source['foam']; start=foam.index('  vec4 frameDx ='); end=foam.index('\n',foam.index('  float strB =',start))
expressions=foam[start:end]
probe=ctx.program(vertex_shader=vertex,fragment_shader=prefix+'''
void main(){'''+point+'''
float wf=swashFarWeight(vec2(x,0.));
vec4 handoffDelta=vec4(0.,-5.,0.,-5.);
vec4 lo=wf*handoffDelta;
float fpXZ=abs(dFdx(x))+abs(dFdy(x));
'''+expressions+'''
float legacy=(length(dFdx(lo.xy))+length(dFdy(lo.xy)))/fpXZ;
result=vec4(strA,legacy,1.-smoothstep(.35,1.,strA),wf);}''')
probe['uRange'].value=(-12,12);probe['uSwashFarWin'].value=(0,5.8,8.8,1);probe['uSwashFocus'].value=(0,0)
ctx.vertex_array(probe,[]).render(vertices=3)
v=np.frombuffer(target.read(components=4,dtype='f4'),dtype='f4').reshape(2,width,4)
assert np.isfinite(v).all() and v[:,:,0].max()<.001 and v[:,:,2].min()>.999
assert v[:,:,1].max()>2, 'Probe must expose false legacy stretching'
metrics['coordinateOrigins']={'correctedMaxStretch':float(v[:,:,0].max()),'legacyMaxFalseStretch':float(v[:,:,1].max()),'minimumCorrectedStructureKeep':float(v[:,:,2].min())}
Path(sys.argv[2]).write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics))
