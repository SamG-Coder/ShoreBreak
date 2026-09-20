"""Link production shaders and inspect per-stage/combined active sampler budgets."""
import ctypes,json,os,sys
from pathlib import Path
import moderngl as gl
ctx=gl.create_standalone_context(backend='egl')
lib=ctypes.CDLL('libGL.so.1')
getindex=lib.glGetProgramResourceIndex;getindex.restype=ctypes.c_uint;getindex.argtypes=[ctypes.c_uint,ctypes.c_uint,ctypes.c_char_p]
getprops=lib.glGetProgramResourceiv;getprops.argtypes=[ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,ctypes.c_int,ctypes.POINTER(ctypes.c_uint),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)]
d=json.loads(Path(sys.argv[1]).read_text());rows=[]
version='#version 300 es\nprecision highp float;precision highp int;\n' if '--es' in sys.argv else '#version 330\n'
std='uniform mat4 modelMatrix,modelViewMatrix,projectionMatrix,viewMatrix;uniform mat3 normalMatrix;uniform vec3 cameraPosition;\n'
for key,m in d['materials'].items():
 flags=''.join(f'#define {k} {v}\n'for k,v in (m.get('defines')or{}).items())
 flags+=('#define USE_INSTANCING\nin mat4 instanceMatrix;\n'if m.get('instanced')else'')+('#define USE_INSTANCING_COLOR\nin vec3 instanceColor;\n'if m.get('instanceColor')else'')
 vs=m.get('vertex');fs=m.get('fragment')
 if m.get('depth'):
  vs='void main(){vec4 p=vec4(position,1);\n#ifdef USE_INSTANCING\np=instanceMatrix*p;\n#endif\ngl_Position=projectionMatrix*viewMatrix*modelMatrix*p;}'
  fs='void main(){gl_FragColor=vec4(vec3(gl_FragCoord.z),1.);}'
 vert=version+flags+std+'in vec3 position;in vec3 normal;in vec3 color;in vec2 uv;\n'+vs
 frag=version+flags.replace('in mat4 instanceMatrix;','').replace('in vec3 instanceColor;','')+'uniform mat4 viewMatrix;uniform vec3 cameraPosition;\n'+'layout(location=0)out vec4 _fragColor;\n#define gl_FragColor _fragColor\n'+fs
 try:p=ctx.program(vertex_shader=vert,fragment_shader=frag)
 except Exception:
  Path('/tmp/failed.vert').write_text(vert);Path('/tmp/failed.frag').write_text(frag);raise
 samplers=[]
 for name in p:
  u=p[name]
  if not hasattr(u,'gl_type')or u.gl_type not in [35678,35679,35680,35682,36289,36292,36293,36306,36307,36308]:continue
  index=getindex(p.glo,0x92E1,name.encode());props=(ctypes.c_uint*2)(0x9306,0x930A);vals=(ctypes.c_int*2)();length=ctypes.c_int()
  getprops(p.glo,0x92E1,index,2,props,2,ctypes.byref(length),vals)
  samplers.append({'name':name,'size':u.array_length,'vertex':bool(vals[0]),'fragment':bool(vals[1])})
 row={'id':key,'combined':sum(s['size']for s in samplers),'vertex':sum(s['size']for s in samplers if s['vertex']),'fragment':sum(s['size']for s in samplers if s['fragment']),'samplers':samplers};rows.append(row)
 if row['combined']>16:print(key,row['combined'],row['vertex'],row['fragment'],','.join(s['name']for s in samplers),flush=True)
 p.release()
Path(sys.argv[2]).write_text(json.dumps(rows,indent=2))
print('Programs',len(rows),'max combined/vertex/fragment',*[max(r[k]for r in rows)for k in ['combined','vertex','fragment']],flush=True)
if '--enforce' in sys.argv:assert all(r['combined']<=16 and r['vertex']<=16 and r['fragment']<=16 for r in rows),'Portable sampler budget exceeded'
