import * as THREE from 'three';

// Water mesh for explore mode: covers every view direction around a moving focus (the player).
//
//   wave band   z in [-16, 12.2] (swash, bore, breakers, shoaling swell) over x in focus.x +- 4 km:
//               a tensor grid of world-fixed rows x world-lattice columns, split into tiers by the
//               along-shore distance |x - focus.x| (all rows near the player, every 2nd / 4th / 16th
//               row further out), joined by static zipper strips (no T-junctions, no cracks);
//   ocean fan   z < -16 to the horizon (-4.5 km): rays of uniform angle from the player (constant
//               on-screen density at every azimuth) x geometric rows, joined to the band's seaward
//               row by a zipper rebuilt on the CPU when the focus moves.
//
// Vertex positions are not stored: every vertex holds (table index, kind, row index) and the vertex
// shader reads its x from a column (or ray) table and its z from a row table. The tables are small
// float textures refreshed on the CPU every frame the camera moves (a few kB), so the index buffers
// of the grids stay static.
//
// No texture swimming: columns sit on world lattices of power-of-two spacings (LOD by distance), rows on a
// fixed world list; LOD transitions geomorph (CDLOD): an odd column / row slides onto its even
// neighbour as the LOD target passes it, so nothing pops when the player walks or runs.

const TEXW = 1024;
export const EX = {
  zTop: 12.2, zBot: -16.0,        // dry reserve beyond the large-set run-up
  detailTop: 3.4,               // original dense surf band; upper reserve carries flatter film
  xExt: 4000,                  // along-shore extent either side of the focus (m)
  tiers: [8.0, 20.0, 60.0],    // |x - focus.x| bounds of the row tiers 0 | 1 | 2 | 3
  steps: [1, 2, 4, 16],        // row step of each tier
  colPx: 7.0,                  // High/Ultra capacity; runtime quality can coarsen this to 9–10 px
  f: 974,                      // focal length (px) the density is designed for (1920 x 1080, vfov 58)
  sBase: 0.01,                 // finest column lattice (m)
  rowK: 0.004,                 // row LOD: target row spacing grows with |z - focus.z| (m per m): the
                               // break zone keeps its 0.9 cm rows within ~3 m (the sheet / lip ribbon
                               // carries the crest detail further out; the 4x MSAA target makes
                               // sub-pixel rows expensive)
  fanPx: 12.0,                 // ocean fan: angular ray spacing (px at f)
  fanS0: 0.12, fanGrow: 1.035, fanEnd: -4500,
  fanRowPx: 1.5,               // far out the fan rows keep >= 1.5 px on screen (s = P dz^2 / (f h))
};

// ------------------------------------------------------------------ rows
function baseRows() {
  const z = [];
  // Keep every original surf-zone row in place. 160 extra upper-beach rows
  // are divisible by all tier steps, so existing row ranks also stay aligned.
  let y = 3.4;
  for(let j=160;j>0;j--)z.push(y+j*.055);
  while (y > 1.0 + 1e-6) { z.push(y); y -= 0.02; }
  while (y > -1.2 + 1e-6) { z.push(y); y -= 0.016; }
  while (y > -3.8 + 1e-6) { z.push(y); y -= 0.009; }
  // geometric to zBot; the count makes (N - 1) a multiple of the largest row step
  const M = EX.steps[EX.steps.length - 1];
  const start = z.length;
  let n = 192;
  while ((start + n - 1) % M !== 0) n++;
  const span = y - EX.zBot, s0 = 0.009;
  // s0 (r + r^2 + ... + r^n) = span  -> r by bisection (the last step lands on zBot)
  let lo = 1.0, hi = 1.2;
  for (let k = 0; k < 60; k++) { const r = 0.5 * (lo + hi); const S = (s0 * r * (Math.pow(r, n - 1) - 1)) / (r - 1); if (S > span) hi = r; else lo = r; }
  const r = 0.5 * (lo + hi);
  z.push(y);
  let s = s0;
  for (let k = 1; k < n; k++) { s *= r; y -= s; z.push(y); }
  z[z.length - 1] = EX.zBot;
  return Float64Array.from(z);
}
function fanRows(zStart) {
  const z = [];
  let y = zStart, s = EX.fanS0;
  // (the rays are radial: a fan cell is ~fanPx wide on screen and its rows are seen end-on; far
  //  out they grow with the square of the distance so they never pile up under the horizon)
  const k = EX.fanRowPx / (EX.f * 2.3);
  while (y > EX.fanEnd) { z.push(y); y -= s; const dz = 4.1 - y; s = Math.max(s * EX.fanGrow, k * dz * dz); }
  z.push(EX.fanEnd);
  return Float64Array.from(z);
}
const rankOf = (j, n) => {
  if (j === 0 || j === n - 1) return 99;
  let r = 0;
  while ((j & 1) === 0) { j >>= 1; r++; }
  return r;
};

// ------------------------------------------------------------------ columns
// Target column spacing at along-shore distance d from the focus: the on-screen spacing of an
// x-step at (d, dz) from an eye at height H is s f sqrt(H^2 + dz^2) / R^2; its worst case over the
// band's rows (dz in [dzMin, dzMax]) is s f / g(d), g = min over q of (q + d^2 / q).
function colTarget(q0, q1) {
  const k = EX.colPx / EX.f;
  return (d) => k * (d < q0 ? q0 + (d * d) / q0 : d > q1 ? q1 + (d * d) / q1 : 2 * d);
}
function colInverse(q0, q1) {   // d at which colTarget reaches s (0 if below the centre value)
  const k = EX.colPx / EX.f;
  return (s) => {
    const G = s / k;
    if (G <= q0) return 0;
    if (G <= 2 * q0) return Math.sqrt((G - q0) * q0);
    if (G <= 2 * q1) return G / 2;
    return Math.sqrt((G - q1) * q1);
  };
}

// scratch for genColumns (per-frame, allocation free)
const GC = { lx: new Float64Array(16384), ll: new Int8Array(16384), rx: new Float64Array(16384), rl: new Int8Array(16384) };

/** Sorted, geomorphed column positions around fx (world lattices of spacing sBase 2^l) written to
 *  out; returns their count. */
function genColumns(fx, fz, eyeH, out) {
  const H = Math.max(eyeH, 0.28);
  // Dry reserve must not raise column density across the entire sea. Its
  // gentle film uses the original surf-band column budget and 5.5 cm rows.
  const dzMin = Math.max(0, fz - EX.detailTop, EX.zBot - fz), dzMax = Math.max(Math.abs(fz - EX.zBot), Math.abs(fz - EX.detailTop));
  const q0 = Math.hypot(H, dzMin), q1 = Math.max(Math.hypot(H, dzMax), q0 * 1.001);
  const tgt = colTarget(q0, q1), inv = colInverse(q0, q1);
  const sB = EX.sBase, X = EX.xExt;
  const S = (l) => sB * 2 ** l;
  const lam = (d) => Math.max(Math.log2(tgt(d) / sB), 0);
  const lc = Math.floor(lam(0));
  // outer end of level l on one side, on the lattice of level l + 1
  const bound = (l, side) => {
    const e = Math.min(inv(S(l + 1)), X), s1 = S(l + 1);
    return side > 0 ? Math.ceil((fx + e) / s1 - 1e-9) * s1 : Math.floor((fx - e) / s1 + 1e-9) * s1;
  };
  // centre level (both sides) goes into the right list; the left list grows outward
  const { lx, ll, rx, rl } = GC;
  let nl = 0, nr = 0;
  let xl = bound(lc, -1), xr = bound(lc, 1);
  for (let k = Math.round(xl / S(lc)); k * S(lc) <= xr + 1e-9; k++) { rx[nr] = k * S(lc); rl[nr++] = lc; }
  let l = lc;
  while ((xr < fx + X || xl > fx - X) && l < 48) {
    l++;
    const sl = S(l);
    if (xr < fx + X) {
      const e = Math.max(bound(l, 1), xr);
      for (let k = Math.floor(xr / sl + 1e-9) + 1; k * sl <= e + 1e-9; k++) { rx[nr] = k * sl; rl[nr++] = l; }
      xr = rx[nr - 1];
    }
    if (xl > fx - X) {
      const e = Math.min(bound(l, -1), xl);
      for (let k = Math.ceil(xl / sl - 1e-9) - 1; k * sl >= e - 1e-9; k--) { lx[nl] = k * sl; ll[nl++] = l; }
      if (nl) xl = lx[nl - 1];
    }
  }
  // geomorph: a column that is not on the next coarser lattice slides onto its lower neighbour
  // as the LOD target passes through [l + 0.5, l + 1] (fully merged where its level ends)
  const morph = (x, lv) => {
    const s = S(lv);
    if (!(Math.round(x / s) & 1)) return x;
    let m = Math.min(Math.max((lam(Math.abs(x - fx)) - lv - 0.35) / 0.65, 0), 1);
    m = m * m * (3 - 2 * m);
    return x - m * s;
  };
  let n = 0;
  for (let i = nl - 1; i >= 0; i--) out[n++] = morph(lx[i], ll[i]);
  for (let i = 0; i < nr; i++) out[n++] = morph(rx[i], rl[i]);
  return n;
}

// ------------------------------------------------------------------ zippers
// Strip between two rails, merged by row index (vertical joints between tiers) or by x (the
// band / fan joint): each step emits one triangle with two vertices on the rail that advances.
// Winding: (left_a, right_b, left_a+1) and (right_b, right_b+1, left_a) for a left / right pair of
// vertical rails; (up_a, up_a+1, down_b) and (up_a, down_b+1, down_b) for horizontal ones - all
// counter-clockwise seen from above like the grid quads. Written in update() without allocation.

function dataTex(n) {
  const h = Math.max(1, Math.ceil(n / TEXW));
  const t = new THREE.DataTexture(new Float32Array(TEXW * h * 4), TEXW, h, THREE.RGBAFormat, THREE.FloatType);
  t.minFilter = t.magFilter = THREE.NearestFilter;
  t.colorSpace = THREE.NoColorSpace;
  t.needsUpdate = true;
  return t;
}

export class ExploreWaterMesh {
  constructor() {
    // rows: band rows then fan rows, one table
    this.zb = baseRows();
    this.nr = this.zb.length;
    this.sb = new Float64Array(this.nr);
    for (let j = 0; j < this.nr; j++) this.sb[j] = j < this.nr - 1 ? this.zb[j] - this.zb[j + 1] : this.zb[j - 1] - this.zb[j];
    this.rank = Int32Array.from({ length: this.nr }, (_, j) => rankOf(j, this.nr));
    this.byRank = [];
    for (let j = 0; j < this.nr; j++) { const r = Math.min(this.rank[j], 31); (this.byRank[r] ||= []).push(j); }
    this.zf = fanRows(EX.zBot - EX.fanS0);
    this.nf = this.zf.length;
    this.rowTex = dataTex(this.nr + this.nf);
    this.zc = new Float64Array(this.nr);

    // fan rays (uniform angle)
    const dth = EX.fanPx / EX.f;
    const thMax = Math.atan(EX.xExt / 8.0);
    this.nRay = 2 * Math.ceil(thMax / dth) + 1;
    this.fanTex = dataTex(this.nRay);
    this.tanTh = new Float64Array(this.nRay);
    {
      const d = this.fanTex.image.data;
      for (let k = 0; k < this.nRay; k++) {
        const th = -thMax + (2 * thMax * k) / (this.nRay - 1);
        const t = Math.tan(th);
        this.tanTh[k] = t;
        d[k * 4] = t; d[k * 4 + 1] = dth * (1 + t * t);
      }
      const r = this.rowTex.image.data;
      for (let j = 0; j < this.nf; j++) {
        const s = j < this.nf - 1 ? this.zf[j] - this.zf[j + 1] : this.zf[j - 1] - this.zf[j];
        r[(this.nr + j) * 4] = this.zf[j]; r[(this.nr + j) * 4 + 1] = s;
      }
    }

    // column blocks: maximum counts over focus positions (the lattice counts vary by a few)
    this.nTier = EX.tiers.length + 1;
    const maxc = new Int32Array(2 * this.nTier - 1);
    this.xs = new Float64Array(32768);
    this.bStart = new Int32Array(2 * this.nTier);
    for (let iz = 0; iz <= 12; iz++) {
      const fz = -14.0 + (50.0 * iz) / 12;
      for (let ix = 0; ix < 160; ix++) {
        const fx = ix * 0.0371;
        const n = genColumns(fx, fz, 0.28, this.xs);
        this._split(this.xs, n, fx);
        for (let b = 0; b < maxc.length; b++) maxc[b] = Math.max(maxc[b], this.bStart[b + 1] - this.bStart[b] + 2);
      }
    }
    // block order: T3L T2L T1L T0 T1R T2R T3R
    this.blockN = Array.from(maxc);
    this.blockStart = [];
    let acc = 0;
    for (const nb of this.blockN) { this.blockStart.push(acc); acc += nb; }
    this.nc = acc;
    this.colTex = dataTex(this.nc);
    this.colX = new Float64Array(this.nc);
    // Pack the existing CPU lookup tables, without changing their values or
    // nearest-texel addressing. The linked water program needs only one unit.
    this.lookupRowOffset = this.colTex.image.data.length / 4;
    this.lookupFanOffset = this.lookupRowOffset + this.rowTex.image.data.length / 4;
    this.lookupTex = dataTex(this.lookupFanOffset + this.fanTex.image.data.length / 4);
    this.uIdx = new Uint32Array(this.nc);
    this.uX = new Float64Array(this.nc);

    this._buildGeometry();
    this.uniforms = {
      uExLookup: { value: this.lookupTex },
      uExOffsets: { value: new THREE.Vector2(this.lookupRowOffset, this.lookupFanOffset) },
      uExFanP: { value: new THREE.Vector4(0, 4.1, -EX.xExt, EX.xExt) },
      uExDetail: { value: new THREE.Vector2(EX.colPx / EX.f, EX.rowK) },
    };
    this.last = null;
    this.params = EX;   // (tuning / debugging handle)
    this.update(0, 4.1, 2.3, true);
  }

  // split the n sorted columns into the 7 blocks by |x - fx|: bStart[b] .. bStart[b + 1]
  _split(xs, n, fx) {
    const T = EX.tiers, nT = T.length + 1, st = this.bStart;
    const blockOf = (x) => {
      const d = Math.abs(x - fx);
      let t = 0;
      while (t < T.length && d > T[t]) t++;
      return t === 0 ? nT - 1 : x < fx ? nT - 1 - t : nT - 1 + t;
    };
    let b = 0;
    st[0] = 0;
    for (let i = 0; i < n; i++) {
      const bi = blockOf(xs[i]);
      while (b < bi) st[++b] = i;
    }
    while (b < 2 * nT - 1) st[++b] = n;
  }

  _buildGeometry() {
    const nT = this.nTier, nb = 2 * nT - 1;
    const verts = [];
    this.blocks = [];
    for (let b = 0; b < nb; b++) {
      const t = Math.abs(b - (nT - 1));
      const step = EX.steps[t];
      const rows = [];
      for (let j = 0; j < this.nr; j += step) rows.push(j);
      const base = verts.length / 3;
      const nc = this.blockN[b];
      for (let c = 0; c < nc; c++) for (const j of rows) verts.push(this.blockStart[b] + c, 0, j);
      this.blocks.push({ base, nc, rows, col0: this.blockStart[b], tier: t, n: nc });
    }
    const fanBase = verts.length / 3;
    for (let k = 0; k < this.nRay; k++) for (let j = 0; j < this.nf; j++) verts.push(k, 1, this.nr + j);
    this.fanBase = fanBase;
    const pos = new THREE.Float32BufferAttribute(new Float32Array(verts), 3);
    const mk = (idx) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', pos);
      g.setIndex(new THREE.BufferAttribute(idx instanceof Uint32Array ? idx : new Uint32Array(idx), 1));
      g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e7);
      return g;
    };
    // one geometry per block: triangles ordered strip by strip, so the draw range can skip the
    // padding slots at the end (their vertices are then never shaded)
    this.V = (b, c, ri) => this.blocks[b].base + c * this.blocks[b].rows.length + ri;
    const V = this.V;
    this.blockGeometries = this.blocks.map((B, b) => {
      const idx = [], nr = B.rows.length;
      for (let c = 0; c < B.nc - 1; c++) for (let r = 0; r < nr - 1; r++) {
        const a = V(b, c, r), bb = V(b, c + 1, r), cc = V(b, c, r + 1), d = V(b, c + 1, r + 1);
        idx.push(a, bb, cc, bb, d, cc);
      }
      B.strip = (nr - 1) * 6;
      return mk(idx);
    });
    // ocean fan
    const fan = [];
    for (let k = 0; k < this.nRay - 1; k++) for (let j = 0; j < this.nf - 1; j++) {
      const a = fanBase + k * this.nf + j, bb = a + this.nf, cc = a + 1, d = bb + 1;
      fan.push(a, bb, cc, bb, d, cc);
    }
    this.fanGeometry = mk(fan);
    // dynamic zippers: tier joints (vertical) + band / fan joint (horizontal); back pass: tiers 0-1
    let maxZ = 3 * (this.nc + this.nRay + 8);
    for (let b = 0; b < nb - 1; b++) maxZ += 3 * (this.blocks[b].rows.length + this.blocks[b + 1].rows.length);
    this.zipIdx = new Uint32Array(maxZ);
    this.zipGeometry = mk(this.zipIdx);
    this.zipGeometry.index.setUsage(THREE.DynamicDrawUsage);
    this.backZipIdx = new Uint32Array(6 * (2 * this.nr + 8));
    this.backZipGeometry = mk(this.backZipIdx);
    this.backZipGeometry.index.setUsage(THREE.DynamicDrawUsage);
    this.backBlocks = this.blocks.map((B, b) => b).filter((b) => this.blocks[b].tier <= 1);
    this.vertexCount = verts.length / 3;
  }

  /** Vertices / triangles drawn with the current tables. */
  stats() {
    let v = this.nRay * this.nf, t = (this.nRay - 1) * (this.nf - 1) * 2 + this.zipGeometry.drawRange.count / 3;
    for (const B of this.blocks) { v += B.n * B.rows.length; t += (B.n - 1) * (B.rows.length - 1) * 2; }
    return { vertices: v, triangles: Math.round(t), columns: this.blocks.map((B) => B.n) };
  }

  /** Refresh the tables for a camera at (x, eyeY, z). Cheap; skips when nothing moved. */
  update(fx, fz, eyeY, force = false) {
    if (!force && this.last && Math.abs(fx - this.last[0]) < 1e-4 && Math.abs(fz - this.last[1]) < 1e-4 && Math.abs(eyeY - this.last[2]) < 1e-3) return;
    this.last = [fx, fz, eyeY];
    this.uniforms?.uExDetail.value.set(EX.colPx / EX.f, EX.rowK);
    // ---- rows: geomorphed LOD by distance from the player along z
    const zb = this.zb, sb = this.sb, zc = this.zc;
    for (let r = this.byRank.length - 1; r >= 0; r--) {
      const list = this.byRank[r];
      if (!list) continue;
      for (const j of list) {
        if (r >= 31) { zc[j] = zb[j]; continue; }
        const L = Math.log2(Math.max(1, (EX.rowK * Math.abs(zb[j] - fz)) / sb[j]));
        let m = Math.min(Math.max((L - r - 0.35) / 0.65, 0), 1);
        m = m * m * (3 - 2 * m);
        zc[j] = zb[j] + (zc[j - (1 << r)] - zb[j]) * m;
      }
    }
    const rd = this.rowTex.image.data;
    for (let j = 0; j < this.nr; j++) {
      const up = j > 0 ? zc[j - 1] - zc[j] : 0, dn = j < this.nr - 1 ? zc[j] - zc[j + 1] : 0;
      rd[j * 4] = zc[j]; rd[j * 4 + 1] = Math.max(up, dn, sb[j]);
    }
    this.rowTex.needsUpdate = true;

    // ---- columns: the real columns of a block fill its first slots (sorted by x)
    const H = Math.max(eyeY, 0.28);
    const xs = this.xs;
    const nx = genColumns(fx, fz, H, xs);
    this._split(xs, nx, fx);
    const cd = this.colTex.image.data, colX = this.colX;
    const nT = this.nTier, nb = 2 * nT - 1, st = this.bStart;
    for (let b = 0; b < nb; b++) {
      const N = this.blockN[b], o = this.blockStart[b];
      let i0 = st[b], i1 = st[b + 1];
      if (i1 - i0 > N) { if (b < nT - 1) i0 = i1 - N; else i1 = i0 + N; }   // (keep the inner end)
      const B = this.blocks[b];
      if (i1 <= i0) { B.n = 1; colX[o] = xs[b < nT - 1 ? 0 : nx - 1]; }
      else { B.n = i1 - i0; for (let c = 0; c < B.n; c++) colX[o + c] = xs[i0 + c]; }
      for (let c = B.n; c < N; c++) colX[o + c] = colX[o + B.n - 1];
      this.blockGeometries[b].setDrawRange(0, Math.max(B.n - 1, 0) * B.strip);
    }
    // spacing = the larger gap to a real neighbour (across block joints too); padding copies the last
    let prevX = NaN, prevI = -1;
    for (let b = 0; b < nb; b++) {
      const B = this.blocks[b], o = this.blockStart[b];
      for (let c = 0; c < B.n; c++) {
        const i = o + c, x = colX[i];
        cd[i * 4] = x;
        const up = prevI >= 0 ? x - prevX : 0;
        cd[i * 4 + 1] = Math.max(up, EX.sBase);
        if (prevI >= 0) cd[prevI * 4 + 1] = Math.max(cd[prevI * 4 + 1], up);
        prevX = x; prevI = i;
      }
    }
    for (let b = 0; b < nb; b++) {
      const B = this.blocks[b], o = this.blockStart[b], l = (o + B.n - 1) * 4;
      for (let c = B.n; c < this.blockN[b]; c++) { cd[(o + c) * 4] = cd[l]; cd[(o + c) * 4 + 1] = cd[l + 1]; }
    }
    this.colTex.needsUpdate = true;
    this.lookupTex.image.data.set(this.colTex.image.data, 0);
    this.lookupTex.image.data.set(this.rowTex.image.data, this.lookupRowOffset * 4);
    this.lookupTex.image.data.set(this.fanTex.image.data, this.lookupFanOffset * 4);
    this.lookupTex.needsUpdate = true;

    // ---- zippers (written straight into the index buffers)
    const Z = this.zipIdx, BZ = this.backZipIdx;
    let n = 0, nbz = 0;
    for (let b = 0; b < nb - 1; b++) {
      const A = this.blocks[b], C = this.blocks[b + 1];
      const la = A.rows.length, lb = C.rows.length, ca = A.n - 1;
      const va = (ri) => A.base + ca * la + ri, vb = (ri) => C.base + ri;
      const s0 = n;
      let a = 0, c = 0;
      while (a < la - 1 || c < lb - 1) {
        if (c === lb - 1 || (a < la - 1 && A.rows[a + 1] <= C.rows[c + 1])) { Z[n++] = va(a); Z[n++] = vb(c); Z[n++] = va(a + 1); a++; }
        else { Z[n++] = vb(c); Z[n++] = vb(c + 1); Z[n++] = va(a); c++; }
      }
      if (A.tier <= 1 && C.tier <= 1) for (let k = s0; k < n; k++) BZ[nbz++] = Z[k];
    }
    // band seaward row / fan first row (merge by x)
    const ax = fx, az = Math.max(fz, 1.0);
    const xmin = colX[this.blockStart[0]], xmax = colX[this.blockStart[nb - 1] + this.blocks[nb - 1].n - 1];
    if (this.uniforms) this.uniforms.uExFanP.value.set(ax, az, xmin, xmax);
    const z0 = this.zf[0], nf = this.nf;
    const fanX = (k) => Math.min(Math.max(ax + (az - z0) * this.tanTh[k], xmin), xmax);
    // upper rail: the band's seaward row over every real column, in x order
    const UI = this.uIdx, UX = this.uX;
    let nu = 0;
    for (let b = 0; b < nb; b++) {
      const B = this.blocks[b], r = B.rows.length;
      for (let c = 0; c < B.n; c++) { UI[nu] = B.base + c * r + r - 1; UX[nu++] = colX[this.blockStart[b] + c]; }
    }
    let a = 0, d = 0;
    while (a < nu - 1 || d < this.nRay - 1) {
      if (d === this.nRay - 1 || (a < nu - 1 && UX[a + 1] <= fanX(d + 1))) { Z[n++] = UI[a]; Z[n++] = UI[a + 1]; Z[n++] = this.fanBase + d * nf; a++; }
      else { Z[n++] = UI[a]; Z[n++] = this.fanBase + (d + 1) * nf; Z[n++] = this.fanBase + d * nf; d++; }
    }
    Z.fill(0, n);
    this.zipGeometry.setDrawRange(0, n);
    this.zipGeometry.index.needsUpdate = true;
    BZ.fill(0, nbz);
    this.backZipGeometry.setDrawRange(0, nbz);
    this.backZipGeometry.index.needsUpdate = true;
  }
}
