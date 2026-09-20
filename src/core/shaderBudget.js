// Three allocates one texture unit per active sampler across the linked program.
// Keep the union within the smaller stage limit as a conservative browser budget.
export function checkShaderBudget(gl, programs) {
  const limits = {
    fragment: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS),
    vertex: gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS),
    combined: gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS),
  };
  const budget = Math.min(16, limits.fragment, limits.vertex, limits.combined);
  const samplerTypes = new Set([35678, 35679, 35680, 35682, 36289, 36292, 36293, 36298, 36299, 36300, 36303, 36306, 36307, 36308, 36311]);
  const rows = [];
  for (const item of programs) {
    const program = item.program;
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const logs = [gl.getProgramInfoLog(program), ...(gl.getAttachedShaders(program) || []).map(shader => gl.getShaderInfoLog(shader))];
      throw new Error('Shader link failed: ' + logs.filter(Boolean).join('\n'));
    }
    const samplers = [];
    for (let i = 0; i < gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS); i++) {
      const u = gl.getActiveUniform(program, i);
      if (samplerTypes.has(u.type)) samplers.push({ name: u.name, size: u.size });
    }
    const count = samplers.reduce((sum, u) => sum + u.size, 0);
    if (count > budget) throw new Error(`Shader texture budget exceeded (${count}/${budget}): ${samplers.map(u => u.name).join(', ')}`);
    rows.push({ name: item.name || item.id, count, samplers });
  }
  return { limits, budget, maximum: Math.max(0, ...rows.map(row => row.count)), programs: rows };
}
