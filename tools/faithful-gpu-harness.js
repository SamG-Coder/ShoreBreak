// Test-only runner for untouched upstream fragment shaders. Never used by the application.
export class OriginalShader {
 constructor(source,width,height,outputs=1){
  this.width=width;this.height=height;this.outputs=outputs;
  const gl=this.gl=document.createElement('canvas').getContext('webgl2');
  if(!gl.getExtension('EXT_color_buffer_float'))throw Error('RGBA32F unavailable');
  gl.getExtension('OES_texture_float_linear');
  const compile=(type,s)=>{const a=gl.createShader(type);gl.shaderSource(a,s);gl.compileShader(a);if(!gl.getShaderParameter(a,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(a));return a;};
  this.program=gl.createProgram();gl.attachShader(this.program,compile(gl.VERTEX_SHADER,'#version 300 es\nvoid main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.0-1.0,0,1);}'));
  gl.attachShader(this.program,compile(gl.FRAGMENT_SHADER,'#version 300 es\nprecision highp float;precision highp int;\nlayout(location=0) out vec4 sbOutput;\n'+source.replaceAll('gl_FragColor','sbOutput')));gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));
  gl.useProgram(this.program);this.fb=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,this.fb);
  for(let i=0;i<outputs;i++){this.texture(width,height,null,i);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0+i,gl.TEXTURE_2D,this.lastTexture,0);}
  gl.drawBuffers(Array.from({length:outputs},(_,i)=>gl.COLOR_ATTACHMENT0+i));if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('Incomplete target');
 }
 texture(width,height,data,unit,linear=false){const gl=this.gl;gl.activeTexture(gl.TEXTURE0+unit);const t=this.lastTexture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,width,height,0,gl.RGBA,gl.FLOAT,data);for(const k of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,k,linear?gl.LINEAR:gl.NEAREST);for(const k of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,k,gl.CLAMP_TO_EDGE);return t;}
 run(textures,uniforms){const gl=this.gl,inputs=[];let unit=this.outputs;
  for(const [name,{data,width,height,linear}] of Object.entries(textures)){inputs.push(this.texture(width,height,data,unit,linear));gl.uniform1i(gl.getUniformLocation(this.program,name),unit++);}
  for(const [name,{type,value}] of Object.entries(uniforms)){const loc=gl.getUniformLocation(this.program,name);gl[type](loc,value);}
  gl.viewport(0,0,this.width,this.height);gl.drawArrays(gl.TRIANGLES,0,3);
  const out=Array.from({length:this.outputs},(_,i)=>{gl.readBuffer(gl.COLOR_ATTACHMENT0+i);const a=new Float32Array(this.width*this.height*4);gl.readPixels(0,0,this.width,this.height,gl.RGBA,gl.FLOAT,a);return a;});
  if(gl.getError())throw Error('Original shader GPU error');for(const t of inputs)gl.deleteTexture(t);return out;
 }
}
export function compare(actual,expected,absolute,relative){let maxAbsolute=0,failures=0,worst=null;
 for(let i=0;i<expected.length;i++){const d=Math.abs(actual[i]-expected[i]);if(d>maxAbsolute){maxAbsolute=d;worst={i,actual:actual[i],expected:expected[i]};}if(!Number.isFinite(actual[i])||!Number.isFinite(expected[i])||d>absolute+relative*Math.abs(expected[i]))failures++;}
 return{values:expected.length,maxAbsolute,failures,worst};
}
