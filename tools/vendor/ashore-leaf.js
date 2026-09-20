// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christopher Canavan
// Procedural geometry shared by the author's Ashore, Jellys and ShoreBreak projects.
// Ported from ASHORE commit 9bed60253a348775807593fa23f85f10472328d3.
import * as THREE from "three";
function leafRibbon(points, widths, color) {
  const p = [], c = [], uv = [], idx = [];
  for (let j = 0; j < points.length; j++) {
    const tangent = points[Math.min(j + 1, points.length - 1)].clone().sub(points[Math.max(0, j - 1)]);
    const side = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize().multiplyScalar(widths[j]);
    for (const s of [-1, 0, 1]) {
      const q = points[j].clone().addScaledVector(side, s);
      q.y += s === 0 ? widths[j] * 0.26 : 0;
      p.push(q.x, q.y, q.z);
      const fac = s === 0 ? 1.09 : 0.86;
      c.push(color.r * fac, color.g * fac, color.b * fac);
      uv.push((s + 1) / 2, j / (points.length - 1));
    }
    if (j > 0) {
      const a = j * 3, b = a - 3;
      if (widths[j - 1] > 1e-8) idx.push(b, b + 1, a, b + 1, b + 2, a + 1);
      if (widths[j] > 1e-8) idx.push(b + 1, a + 1, a, b + 2, a + 2, a + 1);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(c, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  const normals = g.attributes.normal;
  for (let j = 0; j < points.length; j++) if (widths[j] <= 1e-8) {
    const n = new THREE.Vector3();
    for (let s = 0; s < 3; s++) n.add(new THREE.Vector3().fromBufferAttribute(normals, j * 3 + s));
    if (n.lengthSq() < 1e-12) n.set(0, 1, 0);
    else n.normalize();
    for (let s = 0; s < 3; s++) normals.setXYZ(j * 3 + s, n.x, n.y, n.z);
  }
  const root = points[0], anchors = [], weights = [];
  for (let i = 0; i < p.length / 3; i++) {
    anchors.push(root.x, root.y, root.z);
    weights.push(THREE.MathUtils.smoothstep(Math.floor(i / 3) / (points.length - 1), 0, 1));
  }
  g.setAttribute("windRoot", new THREE.Float32BufferAttribute(anchors, 3));
  g.setAttribute("windWeight", new THREE.Float32BufferAttribute(weights, 1));
  return g;
}
export {
  leafRibbon
};
