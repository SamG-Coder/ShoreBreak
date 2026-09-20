import moderngl as gl,json,base64,numpy as np,time,sys,os,ctypes
from PIL import Image
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[2])
D=json.loads(Path(os.getenv('QA_SOURCE','.qa/world.json')).read_text());ctx=gl.create_standalone_context(backend='egl');start=time.monotonic()
if os.getenv('QA_UNIFORM_OVERRIDES'):
 overrides=json.loads(Path(os.getenv('QA_UNIFORM_OVERRIDES')).read_text())
 for values in D['uniforms']:
  for key,value in overrides.items():
   if key in values:values[key]=value
def raw(x):return base64.b64decode(x)
def fbytes(x):return np.array(x,dtype='f4').tobytes()
textures={};targets={};programs={};geos={};vaos={}
for id,t in D['textures'].items():
 size=(t['width'],t['height']);components={1028:1,1030:2,1022:3,1023:4}.get(t['format'],4)
 if t['depth']:tex=ctx.depth_texture(size);tex.compare_func=''
 elif t.get('file'):
  im=Image.open(t['file']);channels=4 if 'A' in im.getbands() else 3;im=im.convert('RGBA' if channels==4 else 'RGB');im=im.transpose(Image.Transpose.FLIP_TOP_BOTTOM) if t['flip'] else im;a=np.asarray(im).astype('f4')/255
  if t['srgb']:a[:,:,:3]=np.where(a[:,:,:3]<=.04045,a[:,:,:3]/12.92,((a[:,:,:3]+.055)/1.055)**2.4)
  tex=ctx.texture(im.size,channels,a.astype('f4').tobytes(),dtype='f4')
 else:
  dtype='f1' if t['type'] in [1009,'Uint8Array'] else 'f2' if t['type'] in [1016,'Uint16Array'] else 'f4'
  tex=(ctx.texture_array((*size,t['array']),components,raw(t['data']) if t.get('data') else None,dtype=dtype) if t.get('array') else ctx.texture3d((*size,t['volume']),components,raw(t['data']) if t.get('data') else None,dtype=dtype) if t.get('volume') else ctx.texture(size,components,raw(t['data']) if t.get('data') else None,dtype=dtype))
 tex.repeat_x=tex.repeat_y=t['repeat'];tex.filter=(gl.LINEAR if t['linear'] else gl.NEAREST,gl.LINEAR if t['linear'] else gl.NEAREST)
 if t.get('volume'):tex.repeat_z=t['repeat']
 if t['mips']:tex.build_mipmaps();tex.filter=(gl.LINEAR_MIPMAP_LINEAR,gl.LINEAR)
 textures[id]=tex
libgl=ctypes.CDLL('libGL.so.1')
for id,t in D['targets'].items():
 depth=textures[t['depth']] if isinstance(t['depth'],str) else ctx.depth_renderbuffer((t['width'],t['height'])) if t['depth'] else None
 if t.get('layer') is not None:
  # ModernGL has no layer-view wrapper; attach the actual array layer to an
  # otherwise ordinary framebuffer with the OpenGL function Three uses.
  placeholder=[ctx.texture((t['width'],t['height']),textures[x].components,dtype=textures[x].dtype) for x in t['colors']]
  targets[id]=ctx.framebuffer(placeholder,depth);targets[id].use()
  for i,x in enumerate(t['colors']):libgl.glFramebufferTextureLayer(0x8D40,0x8CE0+i,textures[x].glo,0,t['layer'])
  assert libgl.glCheckFramebufferStatus(0x8D40)==0x8CD5,'Incomplete array-layer framebuffer'
 else:targets[id]=ctx.framebuffer([textures[x] for x in t['colors']],depth)
W,H=D['width'],D['height'];targets['screen']=ctx.framebuffer([ctx.texture((W,H),4,dtype='f4')],ctx.depth_renderbuffer((W,H)))
for id,g in D['geometries'].items():geos[id]=({n:ctx.buffer(raw(a['data'])) for n,a in g['attributes'].items()},ctx.buffer(raw(g['index'])) if g['index'] else None)
std='uniform mat4 modelMatrix,modelViewMatrix,projectionMatrix,viewMatrix;uniform mat3 normalMatrix;uniform vec3 cameraPosition;\n'
for id,m in D['materials'].items():
 flags=''.join(f'#define {k} {v}\n' for k,v in (m.get('defines') or {}).items())
 flags+=('#define USE_INSTANCING\nin mat4 instanceMatrix;\n' if m['instanced'] else '')+('#define USE_INSTANCING_COLOR\nin vec3 instanceColor;\n' if m['instanceColor'] else '')
 vs=m.get('vertex');frag=m.get('fragment')
 if os.getenv('QA_SHADER_REPLACEMENTS'):
  for replacement in json.loads(Path(os.getenv('QA_SHADER_REPLACEMENTS')).read_text()):
   if frag:frag=frag.replace(replacement['before'],replacement['after'])
   if vs:vs=vs.replace(replacement['before'],replacement['after'])
 if os.getenv('QA_LIP_HELPER') and 'vec2 sheetRaw(' in (vs or ''):
  a=vs.index('vec2 sheetRaw(');b=vs.index('// Everything the heightfield adds',a)
  vs=vs[:a]+Path(os.getenv('QA_LIP_HELPER')).read_text()+vs[b:]
 if os.getenv('QA_SCENERY_VERTS'):
  for marker in ['in float aPart;', 'uniform float uKind;', 'uniform vec4 uCity0;', 'uniform float uCrown;']:
   if marker in (vs or ''):
    choices=[p.read_text() for p in Path(os.getenv('QA_SCENERY_VERTS')).glob('scenery-*.vert') if marker in p.read_text()]
    assert len(choices)==1,(marker,len(choices))
    vs=choices[0]
 if os.getenv('QA_SAND_FLAT') and 'Millimetre grains arrive' in (frag or ''):frag=frag.replace('Nb=normalize(vNb)','Nb=normalize(vec3(0.,1.,-.135))')
 if os.getenv('QA_SAND_FRAG') and 'Millimetre grains arrive' in (frag or ''):frag=Path(os.getenv('QA_SAND_FRAG')).read_text()
 if os.getenv('QA_SAND_VERT') and 'Millimetre grains arrive' in (frag or ''):vs=Path(os.getenv('QA_SAND_VERT')).read_text()
 if os.getenv('QA_NO_CONTACT') and 'texture(uColor,vUv).rgb*contactAt' in (frag or ''):frag=frag.replace('texture(uColor,vUv).rgb*contactAt(vUv,d)','texture(uColor,vUv).rgb')
 if os.getenv('QA_UNDER_VERT') and 'vec3 reflectedBed(' in (frag or ''):vs=Path(os.getenv('QA_UNDER_VERT')).read_text()
 if os.getenv('QA_UNDER_FRAG') and 'vec3 reflectedBed(' in (frag or ''):frag=Path(os.getenv('QA_UNDER_FRAG')).read_text()
 if os.getenv('QA_CLOUD_FRAG') and any(s in (frag or '') for s in ['vec2 cloud(','vec3 cloud(']):frag=Path(os.getenv('QA_CLOUD_FRAG')).read_text()
 if os.getenv('QA_COAST_FRAG') and 'float air=(.72*' in (frag or ''):frag=Path(os.getenv('QA_COAST_FRAG')).read_text()
 if os.getenv('QA_WATER_FRAG') and 'crest = crestGeometry' in (frag or ''):frag=Path(os.getenv('QA_WATER_FRAG')).read_text()
 if os.getenv('QA_GEOM_FACE') and 'crest = crestGeometry' in (frag or ''):frag=frag.replace('vec3 N = normalize(vNrm);','vec3 N = normalize(vNrm); vec3 gn=normalize(cross(dFdx(P),dFdy(P)));if(dot(gn,V)<0.0)gn=-gn;N=normalize(mix(N,gn,crest.w*smoothstep(.5,.85,N.z)));')
 if os.getenv('QA_FLAT_FACE') and 'crest = crestGeometry' in (frag or ''):frag=frag.replace('vec3 N = normalize(vNrm);','vec3 N = normalize(vNrm); if(crest.w>.1)N=normalize(vec3(0.0,.22,1.0));')
 if m['depth']:
  vs='void main(){vec4 p=vec4(position,1);\n#ifdef USE_INSTANCING\np=instanceMatrix*p;\n#endif\ngl_Position=projectionMatrix*viewMatrix*modelMatrix*p;}'
  frag='void main(){gl_FragColor=vec4(1);}'
 try:p=ctx.program(vertex_shader='#version 330\n'+flags+std+'in vec3 position,normal,color;in vec2 uv;\n'+vs,fragment_shader='#version 330\n'+''.join(f'#define {k} {v}\n' for k,v in (m.get('defines') or {}).items())+'uniform mat4 viewMatrix;uniform vec3 cameraPosition;\n'+'layout(location=0)out vec4 fragColor;\n#define gl_FragColor fragColor\n'+frag)
 except Exception:
  print('COMPILE FAILED',id,flush=True);Path(f'.qa/failed-{id}.vert').write_text(vs);Path(f'.qa/failed-{id}.frag').write_text(frag);raise
 programs[int(id)]=p
 print('compiled',id,'elapsed',round(time.monotonic()-start,1),flush=True)
# Cache VAOs by geometry/program/instance content. Most simulation passes reuse one triangle.
reuse=os.getenv('QA_REUSE')
if reuse:
 cache=np.load(os.getenv('QA_CACHE','.qa/warm-cache.npz'))
 assert len(cache.files)==len(textures),(len(cache.files),len(textures))
 for tid,cid in zip(textures,cache.files):
  t=textures[tid];buf=cache[cid].tobytes()
  if len(buf)==t.width*t.height*(D['textures'][tid].get('volume') or D['textures'][tid].get('array') or 1)*t.components*{'f1':1,'f2':2,'f4':4}.get(t.dtype,4):
   t.write(buf)
   if D['textures'][tid]['mips']:t.build_mipmaps()
 print('RESTORED warm simulation',flush=True)
resume_at=next(i for i,c in enumerate(D['commands']) if 'checkpoint' in c)
last_target=None
probe_trace=[]
runup_trace=[]
coast_trace=[]
rock_trace=[]
far_uniforms=next((u for u in D['uniforms'] if isinstance(u.get('uSwashFarView'),dict) and 'uSwashFarMapZ' in u),None)
first_save=next(i for i,c in enumerate(D['commands']) if 'save' in c)
frame_start=next(i for i,c in enumerate(D['commands']) if 'draw' in c and D['draws'][c['draw']]['material']==17) if False else first_save-200
for i,c in enumerate(D['commands']):
 if reuse and i<=resume_at:continue
 if 'checkpoint' in c:
  np.savez(os.getenv('QA_CACHE','.qa/warm-cache.npz'),**{tid:np.frombuffer(tex.read(),dtype='u1') for tid,tex in textures.items()})
  print('CHECKPOINT',i,flush=True);continue
 if 'rockWet' in c:
  tex=textures[c['rockWet']];wet=np.frombuffer(tex.read(),dtype='f4').reshape(tex.height,tex.width,tex.components)
  assert np.isfinite(wet).all() and wet[:,:,:2].min()>=0 and wet[:,:,:2].max()<=1,'Invalid rock wetness'
  # The first rock uses rows 0..14, with its crown at row zero.
  item={'time':c['time'],'name':c['name'],'crownFilm':float(wet[0,:,0].mean()),'crownDamp':float(wet[0,:,1].mean()),'firstRockFilmMean':float(wet[:15,:,0].mean()),'firstRockDampMean':float(wet[:15,:,1].mean())}
  rock_trace.append(item)
  Path('.qa/'+os.getenv('QA_PREFIX','world-')+'rock-wetness.json').write_text(json.dumps(rock_trace,indent=2))
  print('ROCK WETNESS',json.dumps(item),flush=True);continue
 if 'coast' in c:
  tex=textures[c['coast']];a=np.frombuffer(tex.read(),dtype='f2').reshape(tex.height,tex.width,tex.components).astype('f4')
  assert np.isfinite(a).all(),'Nonfinite coast plume atlas'
  assert a[:,:,1:3].min()>=0 and a[:,:,1:3].max()<=1,'Unbounded shore churn'
  zz=c['range'][0]+(np.arange(tex.height)+.5)/tex.height*(c['range'][1]-c['range'][0]);xx=c['left']+(np.arange(tex.width)+.5)/tex.width*48
  central=np.abs(xx+34)<5.;air=a[:,central,1];sand=a[:,central,2];active=np.any(air>.015,axis=1)
  item={'time':c['time'],'view':c['name'],'airMax':float(air.max()),'sandMax':float(sand.max()),'airShoreExtent':float(zz[active].max()) if active.any() else None,'dryReserveAirMax':float(air[-12:].max()),'dryReserveSandMax':float(sand[-12:].max()),'bands':{}}
  for label,lo,hi in [('impact',-4,-2),('lowerShore',-2,0),('uprush',0,3.4),('highRunup',3.4,9)]:
   band=(zz>=lo)&(zz<hi);item['bands'][label]={'airMean':float(air[band].mean()),'airMax':float(air[band].max()),'sandMax':float(sand[band].max())}
  coast_trace.append(item);Path('.qa/'+os.getenv('QA_PREFIX','world-')+'coast.json').write_text(json.dumps(coast_trace,indent=2))
  Image.fromarray(np.uint8(np.clip(a[:,:,1:4],0,1)*255)[::-1]).save('.qa/'+os.getenv('QA_PREFIX','world-')+'atlas-'+c['name']+'.png')
  continue
 if os.getenv('QA_FAST') and 100<i<frame_start:continue
 if 'upload' in c:textures[c['upload']].write(raw(c['data']));continue
 if 'runup' in c:
  def field(tid):
   tex=textures[tid];return np.frombuffer(tex.read(),dtype='f2' if tex.dtype=='f2' else 'f4').reshape(tex.height,tex.width,tex.components).astype('f4')
  v=field(c['runup']);foam=field(c['foam']);assert np.isfinite(v).all() and np.isfinite(foam).all()
  x0,z0,dx,dz=c['dom'];nx,nz=c['size'];xx=x0+(np.arange(nx)+.5)*dx;zz=z0+(np.arange(nz)+.5)*dz
  central=np.abs(xx-c['focus'])<5.;h=v[:,central,0];vel=v[:,central,2];f=foam[:,central,0]
  rows=np.any(h>.001,axis=1);shore=np.abs(zz-3.4)<.09;active=shore[:,None]&(h>.005)
  runup_trace.append({'time':c['time'],'maxRunupZ':float(zz[rows].max()) if rows.any() else None,'reserveDepthMax':float(h[-16:].max()),'foamPastOldEdge':float(f[(zz>3.4)].max()),'oldEdgeDepthMax':float(h[shore].max()),'oldEdgeSpeedMin':float(vel[active].min()) if active.any() else 0.,'oldEdgeSpeedMax':float(vel[active].max()) if active.any() else 0.})
  if far_uniforms:
   fv=field(far_uniforms['uSwashFarView']['texture']);ff=field(far_uniforms['uSwashFarFoamT']['texture']);assert np.isfinite(fv).all() and np.isfinite(ff).all()
   _,L,half,_=far_uniforms['uSwashFarMap'];fz,fdz,fnz,fnx=far_uniforms['uSwashFarMapZ']
   fx=L*np.sinh(((np.arange(int(fnx))+.5)/fnx-.5)*2*half/L);fzz=fz+np.arange(int(fnz))*fdz
   outer=(np.abs(fx)>12)&(np.abs(fx)<70);fh=fv[:,outer,0];wet=np.any(fh>.001,axis=1)
   runup_trace[-1].update(farMaxRunupZ=float(fzz[wet].max()) if wet.any() else None,farReserveDepthMax=float(fh[-16:].max()),farFoamPastOldEdge=float(ff[fzz>3.4][:,outer,0].max()))
  Path('.qa/'+os.getenv('QA_PREFIX','world-')+'runup.json').write_text(json.dumps(runup_trace,indent=2))
  continue
 if 'probe' in c:
  values=np.frombuffer(targets[c['probe']].read(components=4,dtype='f4'),dtype='f4').tolist()
  assert np.isfinite(values).all(),'Nonfinite swimmer surface probe'
  probe_trace.append(dict(time=c['time'],x=c['x'],z=c['z'],samples=values))
  Path('.qa/'+os.getenv('QA_PREFIX','world-')+'probe.json').write_text(json.dumps(probe_trace))
  continue
 if 'save' in c:
  fb=targets['screen'];a=np.frombuffer(fb.read(components=3,dtype='f4'),dtype='f4').reshape(H,W,3)[::-1]
  finite=bool(np.isfinite(a).all())
  destination=Path('.qa/'+os.getenv('QA_PREFIX','world-')+c['save']+'.png')
  temporary=destination.with_suffix('.pending.png')
  Image.fromarray(np.uint8(np.clip(np.nan_to_num(a)*255,0,255))).save(temporary)
  temporary.replace(destination)
  if c['save'].endswith('000') or c['save']=='opening':
   for tid,t in D['targets'].items():
    if t['width']==256 and t['height']==512:
     arr=np.frombuffer(targets[tid].read(components=3,dtype='f4'),dtype='f4').reshape(512,256,3)
     print('SWELL field',np.nanmin(arr[:,:,0]),np.nanmax(arr[:,:,0]),'finite',np.isfinite(arr).all(),flush=True)
     np.save('.qa/swell-field.npy',arr)
  if c['save']=='opening':
   arrays=[]
   for tid,meta in D['textures'].items():
    if not meta.get('array'):continue
    texture=textures[tid]
    values=np.frombuffer(texture.read(),dtype='f2').reshape(meta['array'],meta['height'],meta['width'],4).astype('f4')
    assert np.isfinite(values).all(),'Nonfinite FFT array'
    peaks=np.max(np.abs(values),axis=(1,2,3));assert np.all(peaks>0),'Empty FFT cascade'
    arrays.append({'layers':meta['array'],'resolution':[meta['width'],meta['height']],'absoluteMaxPerLayer':peaks.tolist(),'finite':True})
   Path('.qa/'+os.getenv('QA_PREFIX','world-')+'arrays.json').write_text(json.dumps(arrays,indent=2))
  print('SAVED',c['save'],'finite',finite,'time',c['time'],'elapsed',round(time.monotonic()-start,1),flush=True)
  if os.getenv('QA_STOP_FIRST') or c['save']==os.getenv('QA_STOP_NAME'):break
  continue
 target=c.get('target',c.get('clear'));fb=targets[target]
 if target!=last_target:
  fb.use();last_target=target
 if 'clear' in c:
  fb.depth_mask=True
  if c['color'] is not None:fb.clear(*c['color'],depth=1)
  elif c['depth']:fb.clear(depth=1)
  continue
 d=D['draws'][c['draw']]
 if os.getenv('QA_NO_BACK') and D['targets'].get(target,{}).get('width')==W and len(D['targets'].get(target,{}).get('colors',[]))==1 and D['textures'][D['targets'][target]['colors'][0]]['format']==1028:continue
 if os.getenv('QA_NO_LIP_BACK') and 'only the free sheet (lip) is an exit' in D['materials'][str(d['material'])].get('fragment',''):continue
 if os.getenv('QA_NO_LIP_FRONT') and D['uniforms'][d['uniforms']].get('uPass')==0:continue
 if os.getenv('QA_NO_BARREL') and D['uniforms'][d['uniforms']].get('uPass')==1:continue
 p=programs[d['material']];g=D['geometries'][d['geometry']];m=D['materials'][str(d['material'])]
 key=(d['material'],d['geometry'],d['instance'],d['instanceColor'])
 if key not in vaos:
  buffers,index=geos[d['geometry']];content=[]
  for n,a in g['attributes'].items():
   if n in p:content.append((buffers[n],f"{a['size']}f"+(' /i' if a['instanced'] else ''),n))
  for n,key2,size in [('instanceMatrix','instance',16),('instanceColor','instanceColor',3)]:
   if n in p:content.append((ctx.buffer(raw(d[key2])),f'{size}f /i',n))
  vaos[key]=ctx.vertex_array(p,content,index,index_element_size=4)
 if d['depthTest']:ctx.enable(gl.DEPTH_TEST);ctx.depth_func='1' if d['depthFunc']==1 else '<='
 else:ctx.disable(gl.DEPTH_TEST)
 fb.depth_mask=d['depthWrite']
 if d['side']==2:ctx.disable(gl.CULL_FACE)
 else:ctx.enable(gl.CULL_FACE);ctx.cull_face='front' if d['side']==1 else 'back'
 if d['blend']:
  ctx.enable(gl.BLEND);ctx.blend_equation=gl.MAX if d['equation']==104 else gl.FUNC_ADD
  factor={200:gl.ZERO,201:gl.ONE,204:gl.SRC_ALPHA,205:gl.ONE_MINUS_SRC_ALPHA}
  ctx.blend_func=(factor.get(d['src'],gl.ONE),factor.get(d['dst'],gl.ONE))
 else:ctx.disable(gl.BLEND)
 unit=0
 for k,v in D['uniforms'][d['uniforms']].items():
  if k not in p:continue
  if k=='uPerf' and os.getenv('QA_NO_CREST'):v=[0,0,1,0]
  if k=='uSwashFarWin' and os.getenv('QA_SWASH_LEGACY'):v=[v[0],7.9,9.9,v[3]]
  if k=='uDbg' and os.getenv('QA_WATER_DEBUG'):v=[float(os.getenv('QA_WATER_DEBUG')),1,0,0]
  if k=='uPerf' and os.getenv('QA_NO_CREST'):v=[0,0,1,0]
  if k=='uLipDebug' and os.getenv('QA_LIP_DEBUG'):v=float(os.getenv('QA_LIP_DEBUG'))
  if os.getenv('QA_SNAP_CROSS') and k in ['uBeachFocus','uFocus'] and isinstance(v,list) and .0000005<abs(v[0]+34)<.0000015:v=[-34+(.00002 if v[0]>-34 else -.00002),*v[1:]]
  if isinstance(v,dict) and 'texture' in v:textures[v['texture']].use(unit);p[k].value=unit;unit+=1
  elif isinstance(v,list):
   if len(v)<=4 and p[k].array_length==1:p[k].value=tuple(int(x) for x in v) if p[k].gl_type in [35667,35668,35669] else tuple(v)
   else:p[k].write(np.array(v,dtype='i4' if p[k].gl_type in [5124,35667,35668,35669] else 'f4').tobytes())
  elif isinstance(v,(int,float)):p[k].value=int(v) if p[k].gl_type in [5124,5125,35670] else v
  elif v is None and p[k].gl_type==35678:
   # Unused/null samplers have Three's initialized black texture.
   if 'black' not in textures:textures['black']=ctx.texture((1,1),4,bytes([0,0,0,0]))
   textures['black'].use(unit);p[k].value=unit;unit+=1
 vaos[key].render(instances=d['count'],vertices=d.get('drawCount',-1),first=d.get('drawStart',0))
 for tid in D['targets'].get(target,{}).get('colors',[]):
  if D['textures'][tid]['mips']:textures[tid].build_mipmaps()
 if i%2000==0:print('rendered',i,'/',len(D['commands']),'elapsed',round(time.monotonic()-start,1),flush=True)
print(json.dumps({'programs':len(programs),'commands':len(D['commands']),'renderer':ctx.info['GL_RENDERER'],'seconds':time.monotonic()-start}))
