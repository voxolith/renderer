// Instance records as the GPU reads them, and a CPU sampler that mirrors the shader.
//
// Shared by the renderer (which packs them into the index buffer) and by headless checks (which
// compare what an instance draws with what baking or stamping writes), so there is one definition
// of the maths. The WGSL side is grid.wesl: toModel, instanceVoxel, partsVoxel, instanceNear.
//
// An instance maps a world point to its model through one world→model affine (3x4). A model
// added with parts (one part index per voxel, e.g. a skeleton's bones) can instead be drawn with
// one transform per part: a world point is taken back through each part's inverse in turn, and
// the first part whose own voxel is there wins, parents first. That is posing on the GPU from a
// single rest model shared by every instance, instead of re-baking the posed voxels each frame.
// A small grid of part masks per instance (one pair of words per 8³ brick of its box) says which
// parts can be in each brick, so a sample tests one or two parts, not all of them, and a ray
// skips the bricks of the box no part reaches.
//
// Posing by moving samples into a shared, static rest pose (rather than rebuilding geometry) is
// the idea of Gruen, Benthin, Kern and McAllister, "Ray Tracing Massive Amounts of Animated
// Geometry" (HPG 2026, doi:10.1145/3820014), and of Kao, Makowski, Fujieda and Harada,
// "Voxel Deformation-Aware Neural Intersection Function" (EG 2026 Short Papers,
// doi:10.2312/egs.20261026), which maps rays in deformed space back to rest space per voxel.
// Here the deformation is rigid per part, over the renderer's brick index.

/**
 * Words per instance record: world→model affine (0-11), model (12), palette base (13), flags (14),
 * part offset (15) and count (16), the world cells it can touch (17-22: lo x, y, z, hi x, y, z,
 * exclusive, as i32; the first test a sample meets), part mask grid offset (23, or NO_MASKS).
 */
export const INST_WORDS = 24;
/**
 * Words per part record: world→model affine (0-11), posed world box (12-17, f32), joint cell
 * (18-20, i32), parent (21, or NO_PARENT), has voxels (22), pad (23).
 */
export const PART_WORDS = 24;
/** Instance flag: the model is mirrored along x (applied after flooring, like a stamped orientation). */
export const INST_MIRROR = 1;
/** Instance flag: drawn with one transform per part. */
export const INST_PARTS = 2;
/** A part record's `parent` when it has none. */
export const NO_PARENT = 0xffffffff;
/** An instance with more parts than a mask word holds tests every part (no mask grid). */
export const NO_MASKS = 0xffffffff;
/** Bricks of a part mask grid, in voxels. */
export const MASK_B = 8;

type Vec3 = [number, number, number];

/** A 3x4 affine, row-major: [r00 r01 r02 tx, r10 r11 r12 ty, r20 r21 r22 tz]. */
export type Affine = ArrayLike<number>;

/** What packing needs to know about an instance's model. */
export interface PackModel {
  size: { x: number; y: number; z: number };
  /** Per part: min x, y, z, max x, y, z of its voxels (inclusive), min > max when empty. */
  partBoxes?: Int32Array;
  /** Per part: its parent part (-1 for none) and the joint where it meets it, in model voxels. */
  joints?: readonly { parent: number; at: readonly [number, number, number] }[];
}

/** The placement fields of an instance (the renderer's `Instance`). */
export interface PackInstance {
  x: number;
  y: number;
  z: number;
  anchor?: readonly [number, number, number];
  yaw?: number;
  /** A 3x3 rotation, row-major (world = R · model), instead of `yaw`. */
  rotation?: ArrayLike<number>;
  mirror?: boolean;
  base: number;
  /** Per-part model-space transforms (12 floats each), applied before the placement. */
  parts?: ArrayLike<number>;
}

/** out = a · b (apply b, then a), each at its offset. `out` must not alias `a` or `b`. */
export function mulAffine(a: Affine, b: Affine, out: Float64Array | Float32Array = new Float64Array(12), ao = 0, bo = 0, oo = 0): Float64Array | Float32Array {
  for (let r = 0; r < 3; r++) {
    const a0 = a[ao + r * 4], a1 = a[ao + r * 4 + 1], a2 = a[ao + r * 4 + 2], a3 = a[ao + r * 4 + 3];
    out[oo + r * 4] = a0 * b[bo] + a1 * b[bo + 4] + a2 * b[bo + 8];
    out[oo + r * 4 + 1] = a0 * b[bo + 1] + a1 * b[bo + 5] + a2 * b[bo + 9];
    out[oo + r * 4 + 2] = a0 * b[bo + 2] + a1 * b[bo + 6] + a2 * b[bo + 10];
    out[oo + r * 4 + 3] = a0 * b[bo + 3] + a1 * b[bo + 7] + a2 * b[bo + 11] + a3;
  }
  return out;
}

/** Inverse of an affine with an invertible 3x3 part, each at its offset. */
export function invertAffine(m: Affine, out: Float64Array | Float32Array = new Float64Array(12), mo = 0, oo = 0): Float64Array | Float32Array {
  const a = m[mo], b = m[mo + 1], c = m[mo + 2], tx = m[mo + 3];
  const d = m[mo + 4], e = m[mo + 5], f = m[mo + 6], ty = m[mo + 7];
  const g = m[mo + 8], h = m[mo + 9], i = m[mo + 10], tz = m[mo + 11];
  const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!det) throw new Error("invertAffine: singular transform");
  const k = 1 / det;
  const r00 = A * k, r01 = (c * h - b * i) * k, r02 = (b * f - c * e) * k;
  const r10 = B * k, r11 = (a * i - c * g) * k, r12 = (c * d - a * f) * k;
  const r20 = C * k, r21 = (b * g - a * h) * k, r22 = (a * e - b * d) * k;
  out[oo] = r00; out[oo + 1] = r01; out[oo + 2] = r02; out[oo + 3] = -(r00 * tx + r01 * ty + r02 * tz);
  out[oo + 4] = r10; out[oo + 5] = r11; out[oo + 6] = r12; out[oo + 7] = -(r10 * tx + r11 * ty + r12 * tz);
  out[oo + 8] = r20; out[oo + 9] = r21; out[oo + 10] = r22; out[oo + 11] = -(r20 * tx + r21 * ty + r22 * tz);
  return out;
}

/**
 * The model→world affine of an instance's placement. The model turns about its anchor voxel's
 * centre, so a quarter turn maps voxel centres onto voxel centres exactly (as a stamped
 * orientation does), and the anchor voxel's low corner lands on (x, y, z). For a mirrored instance
 * the anchor is the mirrored model's; the mirror itself is applied after flooring.
 */
export function placement(inst: PackInstance, size: { x: number; y: number; z: number }, fwd = new Float64Array(12)): { fwd: Float64Array; anchor: Vec3 } {
  const a0 = inst.anchor ?? [size.x / 2, 0, size.z / 2];
  const an: Vec3 = inst.mirror ? [size.x - 1 - a0[0], a0[1], a0[2]] : [a0[0], a0[1], a0[2]];
  let r00: number, r01: number, r02: number, r10: number, r11: number, r12: number, r20: number, r21: number, r22: number;
  if (inst.rotation) {
    const R = inst.rotation;
    r00 = R[0]; r01 = R[1]; r02 = R[2]; r10 = R[3]; r11 = R[4]; r12 = R[5]; r20 = R[6]; r21 = R[7]; r22 = R[8];
  } else {
    const yaw = inst.yaw ?? 0, c = Math.cos(yaw), s = Math.sin(yaw);
    r00 = c; r01 = 0; r02 = s; r10 = 0; r11 = 1; r12 = 0; r20 = -s; r21 = 0; r22 = c;
  }
  // world = R (m - pivot) + pos + 0.5, pivot = anchor + 0.5.
  const px = an[0] + 0.5, py = an[1] + 0.5, pz = an[2] + 0.5;
  fwd[0] = r00; fwd[1] = r01; fwd[2] = r02; fwd[3] = inst.x + 0.5 - (r00 * px + r01 * py + r02 * pz);
  fwd[4] = r10; fwd[5] = r11; fwd[6] = r12; fwd[7] = inst.y + 0.5 - (r10 * px + r11 * py + r12 * pz);
  fwd[8] = r20; fwd[9] = r21; fwd[10] = r22; fwd[11] = inst.z + 0.5 - (r20 * px + r21 * py + r22 * pz);
  return { fwd, anchor: an };
}

/** World box of the model-space box [lo, hi) under the affine at m[mo], into out[oo..oo+6]. */
function boxInto(m: Affine, mo: number, x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, out: Float64Array | Float32Array, oo: number): void {
  let bx0 = Infinity, by0 = Infinity, bz0 = Infinity, bx1 = -Infinity, by1 = -Infinity, bz1 = -Infinity;
  for (let k = 0; k < 8; k++) {
    const x = k & 1 ? x1 : x0, y = k & 2 ? y1 : y0, z = k & 4 ? z1 : z0;
    const wx = m[mo] * x + m[mo + 1] * y + m[mo + 2] * z + m[mo + 3];
    const wy = m[mo + 4] * x + m[mo + 5] * y + m[mo + 6] * z + m[mo + 7];
    const wz = m[mo + 8] * x + m[mo + 9] * y + m[mo + 10] * z + m[mo + 11];
    if (wx < bx0) bx0 = wx;
    if (wx > bx1) bx1 = wx;
    if (wy < by0) by0 = wy;
    if (wy > by1) by1 = wy;
    if (wz < bz0) bz0 = wz;
    if (wz > bz1) bz1 = wz;
  }
  out[oo] = bx0; out[oo + 1] = by0; out[oo + 2] = bz0; out[oo + 3] = bx1; out[oo + 4] = by1; out[oo + 5] = bz1;
}

const scratchFwd = new Float64Array(12), scratchInv = new Float64Array(12), scratchW = new Float64Array(12), scratchBox = new Float64Array(6);

/** Bricks of an instance's mask grid along each axis, from its world cells [lo, hi). */
function maskDims(lo: ArrayLike<number>, hi: ArrayLike<number>): [number, number, number] {
  return [Math.max(1, Math.ceil((hi[0] - lo[0]) / MASK_B)), Math.max(1, Math.ceil((hi[1] - lo[1]) / MASK_B)), Math.max(1, Math.ceil((hi[2] - lo[2]) / MASK_B))];
}

/**
 * Pack one instance into `words` at `o` (INST_WORDS words), and, for an instance with parts, its
 * part records (PART_WORDS each) followed by its mask grid into `parts` at `partOff` (size it with
 * {@link maxPartWords}). Returns the instance's world box, which decides the top cells that list
 * it, and the part words used.
 */
export function packInstance(
  inst: PackInstance,
  model: PackModel,
  modelId: number,
  words: Uint32Array,
  o: number,
  parts?: Uint32Array,
  partOff = 0,
): { box: number[]; partWords: number } {
  const f = new Float32Array(words.buffer, words.byteOffset, words.length);
  const wi = new Int32Array(words.buffer, words.byteOffset, words.length);
  const n = inst.parts && model.partBoxes ? model.partBoxes.length / 6 : 0;
  // Parts carry their own transforms; mirroring applies only to plain instances.
  const { fwd } = placement(n ? { ...inst, mirror: false } : inst, model.size, scratchFwd);
  invertAffine(fwd, scratchInv);
  for (let k = 0; k < 12; k++) f[o + k] = scratchInv[k];
  words[o + 12] = modelId;
  words[o + 13] = inst.base;
  words[o + 14] = (inst.mirror && !n ? INST_MIRROR : 0) | (n ? INST_PARTS : 0);
  words[o + 15] = partOff;
  words[o + 16] = n;
  words[o + 23] = NO_MASKS;
  const cells = (b: ArrayLike<number>) => {
    wi[o + 17] = Math.floor(b[0]); wi[o + 18] = Math.floor(b[1]); wi[o + 19] = Math.floor(b[2]);
    wi[o + 20] = Math.ceil(b[3]); wi[o + 21] = Math.ceil(b[4]); wi[o + 22] = Math.ceil(b[5]);
  };
  if (!n) {
    boxInto(fwd, 0, 0, 0, 0, model.size.x, model.size.y, model.size.z, scratchBox, 0);
    cells(scratchBox);
    return { box: Array.from(scratchBox), partWords: 0 };
  }

  // Parts: world_b = placement · part_b, sampled through its inverse.
  if (!parts) throw new Error("packInstance: an instance with parts needs a parts buffer");
  const pf = new Float32Array(parts.buffer, parts.byteOffset, parts.length);
  const pi = new Int32Array(parts.buffer, parts.byteOffset, parts.length);
  const box = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  const pb = model.partBoxes!, local = inst.parts!;
  for (let b = 0; b < n; b++) {
    const q = partOff + b * PART_WORDS;
    mulAffine(fwd, local, scratchW, 0, b * 12, 0);
    invertAffine(scratchW, pf, 0, q);
    const empty = pb[b * 6] > pb[b * 6 + 3];
    if (empty) {
      pf[q + 12] = pf[q + 13] = pf[q + 14] = 0;
      pf[q + 15] = pf[q + 16] = pf[q + 17] = -1;
    } else {
      boxInto(scratchW, 0, pb[b * 6], pb[b * 6 + 1], pb[b * 6 + 2], pb[b * 6 + 3] + 1, pb[b * 6 + 4] + 1, pb[b * 6 + 5] + 1, scratchBox, 0);
      for (let k = 0; k < 6; k++) pf[q + 12 + k] = scratchBox[k];
      for (let a = 0; a < 3; a++) {
        if (scratchBox[a] < box[a]) box[a] = scratchBox[a];
        if (scratchBox[a + 3] > box[a + 3]) box[a + 3] = scratchBox[a + 3];
      }
    }
    const j = model.joints?.[b];
    if (j && j.parent >= 0) {
      const x = j.at[0], y = j.at[1], z = j.at[2];
      pi[q + 18] = Math.floor(scratchW[0] * x + scratchW[1] * y + scratchW[2] * z + scratchW[3]);
      pi[q + 19] = Math.floor(scratchW[4] * x + scratchW[5] * y + scratchW[6] * z + scratchW[7]);
      pi[q + 20] = Math.floor(scratchW[8] * x + scratchW[9] * y + scratchW[10] * z + scratchW[11]);
      parts[q + 21] = j.parent;
    } else {
      pi[q + 18] = pi[q + 19] = pi[q + 20] = 0;
      parts[q + 21] = NO_PARENT;
    }
    parts[q + 22] = empty ? 0 : 1;
    parts[q + 23] = 0;
  }
  // A weld cell can sit one voxel outside every part's box.
  const out = isFinite(box[0]) ? [box[0] - 1, box[1] - 1, box[2] - 1, box[3] + 1, box[4] + 1, box[5] + 1] : [0, 0, 0, 0, 0, 0];
  cells(out);
  let used = n * PART_WORDS;
  if (n > 32) return { box: out, partWords: used };

  // Mask grid over the instance's cells, one brick per MASK_B³: bit b of the first word when part
  // b's posed box touches the brick, of the second when a cell of the brick is in part b's weld
  // block (the 3x3x3 cells around its joint).
  const lo = [wi[o + 17], wi[o + 18], wi[o + 19]], hi = [wi[o + 20], wi[o + 21], wi[o + 22]];
  const [dx, dy, dz] = maskDims(lo, hi);
  const m0 = partOff + used;
  if (parts.length < m0 + dx * dy * dz * 2) throw new Error("packInstance: parts buffer too small for the mask grid (see maxPartWords)");
  parts.fill(0, m0, m0 + dx * dy * dz * 2);
  const mark = (x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, bit: number, word: number) => {
    // Cells [x0, x1) etc. → bricks, clamped to the grid.
    const bx0 = Math.max(0, Math.floor((x0 - lo[0]) / MASK_B)), bx1 = Math.min(dx - 1, Math.floor((x1 - 1 - lo[0]) / MASK_B));
    const by0 = Math.max(0, Math.floor((y0 - lo[1]) / MASK_B)), by1 = Math.min(dy - 1, Math.floor((y1 - 1 - lo[1]) / MASK_B));
    const bz0 = Math.max(0, Math.floor((z0 - lo[2]) / MASK_B)), bz1 = Math.min(dz - 1, Math.floor((z1 - 1 - lo[2]) / MASK_B));
    for (let z = bz0; z <= bz1; z++) for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) parts[m0 + (x + y * dx + z * dx * dy) * 2 + word] |= 1 << bit;
  };
  for (let b = 0; b < n; b++) {
    const q = partOff + b * PART_WORDS;
    if (!parts[q + 22]) continue;
    mark(Math.floor(pf[q + 12]), Math.floor(pf[q + 13]), Math.floor(pf[q + 14]), Math.ceil(pf[q + 15]), Math.ceil(pf[q + 16]), Math.ceil(pf[q + 17]), b, 0);
    const par = parts[q + 21];
    if (par !== NO_PARENT && parts[partOff + par * PART_WORDS + 22]) {
      const jx = pi[q + 18], jy = pi[q + 19], jz = pi[q + 20];
      mark(jx - 1, jy - 1, jz - 1, jx + 2, jy + 2, jz + 2, b, 1);
    }
  }
  words[o + 23] = m0;
  used += dx * dy * dz * 2;
  return { box: out, partWords: used };
}

/**
 * Part words an instance of `model` with parts can need at most (its records plus the largest
 * mask grid its box can have), for sizing a parts buffer before packing.
 */
export function maxPartWords(model: PackModel): number {
  const n = model.partBoxes ? model.partBoxes.length / 6 : 0;
  if (!n) return 0;
  // Any pose stays within a cube of the model's diagonal about its joints, plus the weld margin.
  const d = 2 * Math.ceil(Math.hypot(model.size.x, model.size.y, model.size.z)) + 4;
  const b = Math.ceil(d / MASK_B) + 1;
  return n * PART_WORDS + (n > 32 ? 0 : b * b * b * 2);
}

/** Per-part voxel boxes of a model with parts (the `partBoxes` packing needs). */
export function partBoxes(size: { x: number; y: number; z: number }, data: Uint8Array, parts: Uint8Array, count: number): Int32Array {
  const box = new Int32Array(count * 6);
  for (let b = 0; b < count; b++) {
    box.fill(1 << 30, b * 6, b * 6 + 3);
    box.fill(-(1 << 30), b * 6 + 3, b * 6 + 6);
  }
  let i = 0;
  for (let z = 0; z < size.z; z++)
    for (let y = 0; y < size.y; y++)
      for (let x = 0; x < size.x; x++, i++) {
        if (!data[i] || parts[i] >= count) continue;
        const o = parts[i] * 6;
        if (x < box[o]) box[o] = x;
        if (y < box[o + 1]) box[o + 1] = y;
        if (z < box[o + 2]) box[o + 2] = z;
        if (x > box[o + 3]) box[o + 3] = x;
        if (y > box[o + 4]) box[o + 4] = y;
        if (z > box[o + 5]) box[o + 5] = z;
      }
  return box;
}

/** Lookups the CPU sampler needs: the model's voxel and part (+1, 0 = none) at a model cell. */
export interface SampleModel {
  size: { x: number; y: number; z: number };
  voxel(x: number, y: number, z: number): number;
  part(x: number, y: number, z: number): number;
}

/**
 * The value an instance draws at world cell (x, y, z), or 0: the CPU twin of grid.wesl's
 * instance sampling, reading the same packed words (as f32, like the GPU).
 */
export function sampleInstance(words: Uint32Array, o: number, parts: Uint32Array | undefined, model: SampleModel, x: number, y: number, z: number): number {
  const wi = new Int32Array(words.buffer, words.byteOffset, words.length);
  if (x < wi[o + 17] || y < wi[o + 18] || z < wi[o + 19] || x >= wi[o + 20] || y >= wi[o + 21] || z >= wi[o + 22]) return 0;
  const f = new Float32Array(words.buffer, words.byteOffset, words.length);
  const fr = Math.fround;
  const wx = x + 0.5, wy = y + 0.5, wz = z + 0.5;
  const { size } = model;
  const inside = (q: Vec3) => q[0] >= 0 && q[1] >= 0 && q[2] >= 0 && q[0] < size.x && q[1] < size.y && q[2] < size.z;
  const map = (m: Float32Array, at: number): Vec3 => [
    Math.floor(fr(fr(fr(fr(m[at] * wx) + fr(m[at + 1] * wy)) + fr(m[at + 2] * wz)) + m[at + 3])),
    Math.floor(fr(fr(fr(fr(m[at + 4] * wx) + fr(m[at + 5] * wy)) + fr(m[at + 6] * wz)) + m[at + 7])),
    Math.floor(fr(fr(fr(fr(m[at + 8] * wx) + fr(m[at + 9] * wy)) + fr(m[at + 10] * wz)) + m[at + 11])),
  ];
  const flags = words[o + 14];
  if (!(flags & INST_PARTS)) {
    const q = map(f, o);
    if (flags & INST_MIRROR) q[0] = size.x - 1 - q[0];
    if (!inside(q)) return 0;
    return model.voxel(q[0], q[1], q[2]);
  }
  const pf = new Float32Array(parts!.buffer, parts!.byteOffset, parts!.length);
  const pi = new Int32Array(parts!.buffer, parts!.byteOffset, parts!.length);
  const off = words[o + 15], n = words[o + 16];
  // Which parts to test: the brick's masks, or every part (by its box) when there is no mask grid.
  const m0 = words[o + 23];
  let hits = 0xffffffff, welds = 0xffffffff;
  if (m0 !== NO_MASKS) {
    const lo = [wi[o + 17], wi[o + 18], wi[o + 19]], hi = [wi[o + 20], wi[o + 21], wi[o + 22]];
    const [dx, dy] = maskDims(lo, hi);
    const bi = Math.floor((x - lo[0]) / MASK_B) + Math.floor((y - lo[1]) / MASK_B) * dx + Math.floor((z - lo[2]) / MASK_B) * dx * dy;
    hits = parts![m0 + bi * 2];
    welds = parts![m0 + bi * 2 + 1];
  }
  for (let b = 0; b < n; b++) {
    if (b < 32 && !((hits >>> b) & 1)) continue;
    const q0 = off + b * PART_WORDS;
    if (!parts![q0 + 22]) continue;
    // Without masks, the cells the part's posed box touches (floor(min) .. ceil(max) - 1). With
    // them no box test is needed: a cell whose centre maps into the part's rest box lies in it.
    if (m0 === NO_MASKS &&
      (x < Math.floor(pf[q0 + 12]) || y < Math.floor(pf[q0 + 13]) || z < Math.floor(pf[q0 + 14]) ||
        x >= Math.ceil(pf[q0 + 15]) || y >= Math.ceil(pf[q0 + 16]) || z >= Math.ceil(pf[q0 + 17]))) continue;
    const q = map(pf, q0);
    if (!inside(q)) continue;
    const v = model.voxel(q[0], q[1], q[2]);
    if (v && model.part(q[0], q[1], q[2]) === b + 1) return v;
  }
  // Weld: near a joint, either side's voxels may fill the cell, so a turned child stays on.
  for (let b = 0; b < n; b++) {
    if (b < 32 && !((welds >>> b) & 1)) continue;
    const q0 = off + b * PART_WORDS;
    const par = parts![q0 + 21];
    if (par === NO_PARENT || !parts![q0 + 22] || !parts![off + par * PART_WORDS + 22]) continue;
    if (Math.abs(x - pi[q0 + 18]) > 1 || Math.abs(y - pi[q0 + 19]) > 1 || Math.abs(z - pi[q0 + 20]) > 1) continue;
    for (const via of [b, par]) {
      const q = map(pf, off + via * PART_WORDS);
      if (!inside(q)) continue;
      const v = model.voxel(q[0], q[1], q[2]);
      const pid = model.part(q[0], q[1], q[2]) - 1;
      if (v && (pid === b || pid === par)) return v;
    }
  }
  return 0;
}
