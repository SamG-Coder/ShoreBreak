"""Exercise the production rock-history shader with controlled water contact.

Usage: python tools/native/rock-wetness.py programs.json report.json
The full scene replay separately tests actual wave/rock interactions.
"""
import json,sys
from pathlib import Path
import moderngl as gl
import numpy as np

source=json.loads(Path(sys.argv[1]).read_text())
matches=[m['fragment'] for m in source['materials'].values() if 'uRockWetReset' in (m.get('fragment') or '')]
assert len(matches)==1
ctx=gl.create_standalone_context(backend='egl')
vertex='#version 330\nvoid main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0.,1.);}'
fragment='#version 330\nout vec4 result;\n#define gl_FragColor result\n'+matches[0]
p=ctx.program(vertex_shader=vertex,fragment_shader=fragment)
vao=ctx.vertex_array(p,[])
# Sun-exposed crown, sheltered face, unreached crown, and equal far-field face.
samples=np.array([[0,.3,0,.2],[0,.3,0,.85],[0,.6,0,.2],[30,.3,0,.2]],dtype='f4')
points=ctx.texture((4,1),4,samples.tobytes(),dtype='f4')
near=ctx.texture((1,1),4,dtype='f4');far=ctx.texture((1,1),4,dtype='f4')
states=[ctx.texture((4,1),4,dtype='f4') for _ in range(2)]
targets=[ctx.framebuffer([t]) for t in states]
for t in [points,near,far,*states]:
    t.filter=(gl.NEAREST,gl.NEAREST);t.repeat_x=t.repeat_y=False
points.use(0);near.use(2);far.use(3)
for name,unit in [('uRockSamples',0),('uRockWetPrevious',1),('uSweView',2),('uSwashFarView',3)]:p[name].value=unit
p['uSweDom'].value=(-11,-4,22,10)
p['uSwashFarMap'].value=(0,12,12*np.arcsinh(128/12),1)
p['uSwashFarMapZ'].value=(-4,.1,100,4)
p['uSwashFarWin'].value=(0,5.8,8.8,1);p['uSwashFocus'].value=(0,0)
index=0

def water(depth,level):
    a=np.array([depth,0,0,level],dtype='f4').tobytes();near.write(a);far.write(a)

def step(dt,reset=False):
    global index
    p['uRockWetDt'].value=dt;p['uRockWetReset'].value=float(reset)
    states[index].use(1);index=1-index;targets[index].use();vao.render(vertices=3)

def read():
    a=np.frombuffer(states[index].read(),dtype='f4').reshape(4,4)[:,:2].copy()
    assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1
    return a

def soak(hz=60):
    water(.1,.4);step(0,True)
    for _ in range(round(hz*1.5)):step(1/hz)
    result=read();assert result[0].min()>.97 and result[2].max()==0
    assert np.max(np.abs(result[0]-result[3]))<1e-6
    return result

metrics={'wetAfter1_5Seconds':soak().tolist()}
# A high stored terrain level without water must never re-wet an exposed rock.
water(0,.8)
elapsed=0;previous=read();checkpoints={}
for seconds in [3,10,60,180,600]:
    for _ in range((seconds-elapsed)*60):step(1/60)
    a=read();assert np.all(a<=previous+1e-6),'Dry history grew without water'
    assert a[2].max()==0,'A never-covered crown became wet'
    assert np.max(np.abs(a[0]-a[3]))<1e-6,'Near/far history differs'
    checkpoints[str(seconds)]=a.tolist();previous=a;elapsed=seconds
assert checkpoints['10'][0][0]<.08 and checkpoints['10'][0][1]>.92
assert checkpoints['60'][1][1]>checkpoints['60'][0][1]>.6
assert checkpoints['600'][1][1]<.09,'Moisture froze due to precision loss'
metrics['dryingSeconds']=checkpoints
# A second wave refreshes the same samples without wetting an unreached crown.
water(.1,.4)
for _ in range(90):step(1/60)
rewet=read();assert rewet[0].min()>.97 and rewet[2].max()==0
metrics['rewet']=rewet.tolist()
# Moving the field or looking away must not erase history. At zero simulation
# time contact and drying are both exactly inert, even across the whole domain.
before=read()
for center in [-80,-10,0,20,80]:
    p['uSweDom'].value=(center-11,-4,22,10)
    p['uSwashFarWin'].value=(center,5.8,8.8,1);p['uSwashFocus'].value=(center,0)
    p['uSwashFarMap'].value=(center,12,12*np.arcsinh(128/12),1)
    step(0);assert np.array_equal(read(),before),'Camera motion altered history while paused'
metrics['pausedCameraShiftMaxDifference']=0
p['uSweDom'].value=(-11,-4,22,10)
p['uSwashFarWin'].value=(0,5.8,8.8,1);p['uSwashFocus'].value=(0,0)
p['uSwashFarMap'].value=(0,12,12*np.arcsinh(128/12),1)
rates={}
for hz in [30,60,120]:
    soak(hz);water(0,.8)
    for _ in range(10*hz):step(1/hz)
    rates[str(hz)]=read()
delta=max(float(np.max(np.abs(a-rates['60']))) for a in rates.values())
# The separate soak/drain integration has a small timestep truncation error;
# bound it below 0.13 of one 8-bit intensity step. Production runs at 60 Hz.
assert delta<.0005,delta
metrics['timestepMaxDifference']=delta
metrics['passed']=True
Path(sys.argv[2]).write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics))
