"""Exercise the production GPU flux and wetness shaders around an actual coastal rock."""
import json,base64,time,re
from pathlib import Path
import numpy as np
import moderngl as gl
root=Path(__file__).resolve().parents[2]
D=json.loads((root/'.qa/hydraulics.json').read_text());ctx=gl.create_standalone_context(backend='egl')
N=128;dom=(-36.8,-2.25,.025,.025)
vert='''#version 330
out vec2 vUv;
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);vUv=p;gl_Position=vec4(p*2.-1.,0.,1.);}'''
programs={}
D['shaders']['pFluxReference']=re.sub(r'// DRY_STENCIL_BEGIN.*?// DRY_STENCIL_END','',D['shaders']['pFlux'],flags=re.S)
for k,s in D['shaders'].items():
 p=ctx.program(vertex_shader=vert,fragment_shader='#version 330\nlayout(location=0)out vec4 fragColor;\n#define gl_FragColor fragColor\n'+s)
 programs[k]=(p,ctx.vertex_array(p,[]))
rock=ctx.texture((D['rock']['width'],D['rock']['height']),1,base64.b64decode(D['rock']['data']),dtype='f4');rock.filter=(gl.LINEAR,gl.LINEAR);rock.repeat_x=rock.repeat_y=False

def texture():return ctx.texture((N,N),4,dtype='f4')
def run(name,target,values={},maps={}):
 p,vao=programs[name]
 for k,v in {'uN':(N,N),'uDom':dom,**values}.items():
  if k in p:p[k].value=v
 for i,(k,t) in enumerate(maps.items()):
  if k in p:t.use(i);p[k].value=i
 target.use();ctx.viewport=(0,0,N,N);vao.render(gl.TRIANGLES,vertices=3)
def read(tex):return np.frombuffer(tex.read(),dtype='f4').reshape(N,N,4)

def bed(with_rocks):
 t=texture();f=ctx.framebuffer([t]);run('pBed',f,{'uRelief':D['relief'],'uRockDomain':tuple(D['rock']['domain']) if with_rocks else (0,0,0,0)},{'uRockBed':rock});return t,f

def flux(bt,velocity,steps,program='pFlux'):
 B=read(bt)[:,:,0];state=np.zeros((N,N,4),dtype='f4');state[:,:,0]=np.maximum(B,0.);
 state[:,:,2]=np.maximum(-B,0.)*velocity
 a,b=texture(),texture();a.write(state.tobytes());fa,fb=ctx.framebuffer([a]),ctx.framebuffer([b])
 for i in range(steps):run(program,fb,{'uDt':1/1200},{'uBed':bt,'uState':a});a,b=b,a;fa,fb=fb,fa
 U=read(a);h=U[:,:,0]-B;assert np.isfinite(U).all();assert h.min()>-2e-5,(h.min(),'negative depth')
 return a,U,h

start=time.monotonic();bt,bf=bed(True);flat,ff=bed(False)
a,U,h=flux(bt,0,720)
mask=h>.005;speed=np.linalg.norm(U[:,:,1:3],axis=2)/np.maximum(h,1e-5)
assert speed[mask].max()<.003,('lake at rest',speed[mask].max())
print('lake_at_rest_max_speed_m_s',float(speed[mask].max()))
a,U,h=flux(bt,.85,1200);ac,Uc,hc=flux(flat,.85,1200)
_,reference,_=flux(bt,.85,1200,'pFluxReference')
dry_error=float(np.max(np.abs(U-reference)))
assert dry_error<2e-6,('Dry-stencil shortcut changed wet/dry flow',dry_error)
print('dry_stencil_reference_max_error',dry_error)
xx=dom[0]+(np.arange(N)+.5)*dom[2];zz=dom[1]+(np.arange(N)+.5)*dom[3];X,Z=np.meshgrid(xx,zz)
region=(np.abs(X+35.2)<1.2)&(np.abs(Z+.65)<.8)&(h>.006)&(hc>.006)
lateral=np.mean(np.abs(U[:,:,1][region]/h[region]));control=np.mean(np.abs(Uc[:,:,1][region]/hc[region]))
assert lateral>max(.025,control*1.25),('no flow deflection',lateral,control)
print('rock_lateral_speed_m_s',float(lateral),'sand_control',float(control))
# The production wet pass must retain and then lower a recently submerged crown's waterline.
B=read(bt)[:,:,0];flood=np.zeros((N,N,4),dtype='f4');flood[:,:,0]=B+.08
state=texture();state.write(flood.tobytes());w0,w1=texture(),texture();wf0,wf1=ctx.framebuffer([w0]),ctx.framebuffer([w1])
run('pWetInit',wf0,{'uRockDomain':tuple(D['rock']['domain'])},{'uRockBed':rock})
run('pWet',wf1,{'uDtS':.1,'uFilmTau':D['filmTau']},{'uBed':bt,'uState':state,'uWetPrev':w0});peak=read(w1)
flood[:,:,0]=B;state.write(flood.tobytes())
run('pWet',wf0,{'uDtS':1.,'uFilmTau':D['filmTau']},{'uBed':bt,'uState':state,'uWetPrev':w1});drained=read(w0)
rockmask=B-read(flat)[:,:,0]>.08
assert np.all(drained[:,:,0][rockmask]<.75) and np.all(drained[:,:,0][rockmask]>.6)
assert np.all(drained[:,:,2][rockmask]<peak[:,:,2][rockmask])
assert np.all(drained[:,:,2][rockmask]>B[rockmask])
print('wet_waterline_retained_and_draining',int(rockmask.sum()),'cells')
report={'lakeRestMaxSpeed':float(speed[mask].max()),'rockLateralSpeed':float(lateral),'sandLateralSpeed':float(control),'wetCellsChecked':int(rockmask.sum()),'dryStencilReferenceMaxError':dry_error,'seconds':time.monotonic()-start,'renderer':ctx.info['GL_RENDERER']}
(root/'.qa/hydraulics-result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
