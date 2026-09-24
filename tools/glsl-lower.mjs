export function lower(s){
 s=s.replace(/\/\*[\s\S]*?\*\//g,'').replace(/\/\/[^\n]*/g,'').replace(/precision[^;]+;/g,'');
 s=s.replace(/struct\s+\w+\s*\{[^}]*\}/g,s=>s.replace(/\bfloat\s+([\w,\s]+);/g,(_,n)=>n.split(',').map(x=>`float ${x.trim()};`).join(' ')));
 s=s.replace(/\b(vec[234]|ivec2)\s*\(/g,(_,t)=>t==='ivec2'?'sb_vec2(':'sb_'+t+'(').replace(/\bivec2\b/g,'float2').replace(/\bvec([234])\b/g,'float$1');
 s=s.replace(/\b(out|inout)\s+(\w+)\s+(\w+)/g,'$2& $3');
 for(const n of ['sin','cos','exp','log','sqrt','abs','floor','fract','sign','tanh','min','max','clamp','mix','step','smoothstep','mod'])s=s.replace(new RegExp(`\\b${n}\\s*\\(`,'g'),`sb_${n}(`);
 s=s.replace(/\bpow\(/g,'powf(').replace(/(?<![\w.])(\d+\.\d*|\.\d+|\d+[eE][+-]?\d+)([eE][+-]?\d+)?(?![\w.])/g,'$1$2f');
 return s;
}
