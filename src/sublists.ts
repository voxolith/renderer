// The sub-cell tables of static instance cells: built on the CPU at setInstances, read by
// grid.wesl's instRange. (movingTable, below, adds a moving set's instances to the tables of the
// cells they are in, every frame.) Headless (no GPU), so the renderer's verify can check a build against
// brute-force sampling (sampleInstance).
//
// Layout, in the "subs" region: one word per world top cell (0 = no table, else the table's offset
// + 1), then the tables. A table has 64 entries, one per 16³ sub-cell of its 64³ cell: (offset
// from the table to the sub-cell's list << 16) | (mask of its 8 bricks the list reaches << 8) |
// count; each table is followed by its lists.
//
// What an instance reaches is tight: every occupied 2³ sub-cell of its model, turned into world
// space, marks the world cells whose voxel centre can sample it (with slack for the shader's f32
// rounding), not the model's 8³ brick boxes widened by a voxel. The idea of tight occupancy
// instead of loose boxes, and the analysis that led here, come from Atsushi Yoshimura and Takahiro
// Harada, "Subspace Culling for Ray–Box Intersection", Proc. ACM Comput. Graph. Interact. Tech.
// 6(1), I3D 2023, doi:10.1145/3585503. Their technique itself (a sub-cell mask per box and a
// ray-mask table) was implemented and measured here and not kept: on nightwood at 20/50/100 vox/m
// it added 2.5-8 points over these tables alone, for 4/44/245 MiB of masks, and world-brick masks
// made the world page 4-12% slower.

import { BRICK_B, TOP_B } from "./brick";
import { INST_WORDS, placement, scaledSize, type PackInstance } from "./instance";

/** One placed instance, as the build needs it. */
export interface SubListInstance {
  /** Its placement. */
  inst: PackInstance;
  /** The model's extent in its own voxels. */
  size: { x: number; y: number; z: number };
  /** The factor the model is drawn enlarged by (default 1). */
  scale?: number;
  /** The model's occupied 2³ sub-cells (x, y, z triples), brick by brick (neighbours reach the same bricks). */
  subs: ArrayLike<number>;
  /** Posed instances: the world box (min x, y, z, max x, y, z); anything in it may be drawn. */
  posedBox?: ArrayLike<number>;
}

/** The built region. */
export interface SubLists {
  /** The region's words (cells first, then tables). */
  data: Uint32Array;
  /** Its first `cells` words (the per-cell pointers), for restoring cells a moving set touched. */
  cellWords: Uint32Array;
  /** Tables built and (sub-cell, instance) pairs listed. */
  stats: { tables: number; pairs: number };
}

/**
 * The slack around a world box when finding the voxel centres in it: the shader's f32 transform
 * rounds, by a few ulps of the largest coordinate.
 */
export function sampleSlack(gridMax: number): number {
  return 1 / 64 + gridMax * 2 ** -20;
}

/**
 * Pieces per axis each occupied sub-cell of a model drawn at `scale` is marked as, under the
 * rotation `fwd` (a placement affine). An enlarged sub-cell is a cube 2 × scale voxels wide; its
 * world box is exact when the rotation keeps axes (quarter turns, mirrors), but grows by up to
 * 40% of its width under a yaw, so a turned one is split into pieces at most 8 voxels (a brick)
 * wide, each with its own tighter box. Measured in verify's scaled scene (k = 2-10, mostly turned):
 * 2811 sub-cell entries unsplit, 2661 with 8-voxel pieces, 2633 with 4-voxel ones at several times
 * the marking. Not exported from the package.
 */
export function subPieces(fwd: ArrayLike<number>, scale: number): number {
  if (scale === 1) return 1;
  const a = (i: number) => Math.abs(fwd[i]);
  const axial = [0, 4, 8].every((r) => Math.abs(a(r) + a(r + 1) + a(r + 2) - 1) < 1e-9);
  return axial ? 1 : Math.ceil(scale / 4);
}

/**
 * Build the sub-cell tables for the static cells in `perCell` (top cell index → instance indices,
 * at most `maxList` each; longer lists are not split) of a world with `brickDim` bricks and
 * `topDim` top cells, `gridMax` voxels on its longest side. `marked`, when given, is called after
 * each instance has been marked (with its index; the marking is nearly all of the build's time).
 */
export function buildSubLists(
  brickDim: readonly [number, number, number],
  topDim: readonly [number, number, number],
  perCell: Map<number, number[]>,
  placed: readonly SubListInstance[],
  maxList: number,
  gridMax: number,
  marked?: (k: number) => void,
): SubLists {
  const cells = topDim[0] * topDim[1] * topDim[2];
  const [tx, ty] = topDim;
  const table = new Int32Array(cells).fill(-1);
  let tables = 0;
  for (const [ci, l] of perCell) if (l.length <= maxList) table[ci] = tables++;
  const slots = tables * 64;
  // (slot, instance) pairs in instance order, each pair once, then a counting sort by slot.
  let pairSlot: Uint32Array<ArrayBuffer> = new Uint32Array(1 << 16), pairK: Uint32Array<ArrayBuffer> = new Uint32Array(1 << 16), n = 0;
  const counts = new Uint32Array(slots), last = new Int32Array(slots).fill(-1), masks = new Uint8Array(slots);
  const [bx, by, bz] = brickDim;
  const seenK = new Int32Array(256).fill(-1), seenB = new Int32Array(256);
  const markBrick = (k: number, gx: number, gy: number, gz: number) => {
    const bi = gx + (gy + gz * by) * bx, h = (gx & 7) | ((gy & 3) << 3) | ((gz & 7) << 5);
    if (seenK[h] === k && seenB[h] === bi) return;
    seenK[h] = k;
    seenB[h] = bi;
    const t = table[(gx >> 3) + (gy >> 3) * tx + (gz >> 3) * tx * ty];
    if (t < 0) return;
    const slot = t * 64 + ((gx >> 1) & 3) + ((gy >> 1) & 3) * 4 + ((gz >> 1) & 3) * 16;
    masks[slot] |= 1 << ((gx & 1) + (gy & 1) * 2 + (gz & 1) * 4);
    if (last[slot] === k) return;
    last[slot] = k;
    counts[slot]++;
    if (n === pairSlot.length) { pairSlot = grow(pairSlot, n * 2); pairK = grow(pairK, n * 2); }
    pairSlot[n] = slot;
    pairK[n++] = k;
  };
  const mark = (k: number, x0: number, y0: number, z0: number, x1: number, y1: number, z1: number) => {
    const gx0 = Math.max(0, Math.floor(x0 / BRICK_B)), gx1 = Math.min(bx - 1, Math.floor(x1 / BRICK_B));
    const gy0 = Math.max(0, Math.floor(y0 / BRICK_B)), gy1 = Math.min(by - 1, Math.floor(y1 / BRICK_B));
    const gz0 = Math.max(0, Math.floor(z0 / BRICK_B)), gz1 = Math.min(bz - 1, Math.floor(z1 / BRICK_B));
    for (let gz = gz0; gz <= gz1; gz++) for (let gy = gy0; gy <= gy1; gy++) for (let gx = gx0; gx <= gx1; gx++) markBrick(k, gx, gy, gz);
  };
  const slack = sampleSlack(gridMax);
  const maxX = bx * BRICK_B - 1, maxY = by * BRICK_B - 1, maxZ = bz * BRICK_B - 1;
  const fwd = new Float64Array(12);
  if (tables) {
    for (let k = 0; k < placed.length; k++) {
      const p = placed[k];
      if (p.posedBox) {
        // Posed: anywhere in its box.
        const b = p.posedBox;
        mark(k, b[0] - 1, b[1] - 1, b[2] - 1, b[3] + 1, b[4] + 1, b[5] + 1);
        if (marked) marked(k);
        continue;
      }
      // Every occupied 2³ sub-cell's world box: centre = fwd · (sub-cell centre), half-extent |R| · 1.
      // Mirroring flips the voxel index along x after the turn (see sampleInstance), so sub-cell x
      // covers voxels size.x - 2x - 2 .. size.x - 2x - 1 before it. At a scale k the instance's
      // space is the model enlarged k times: the sub-cell is the cube of side 2k about k times its
      // centre, marked whole or, when turned, as pieces (subPieces) of side 2k / n.
      const ks = p.scale ?? 1;
      placement(p.inst, scaledSize(p.size, ks), fwd);
      const np = subPieces(fwd, ks), half = ks / np;
      const ex = (Math.abs(fwd[0]) + Math.abs(fwd[1]) + Math.abs(fwd[2])) * half;
      const ey = (Math.abs(fwd[4]) + Math.abs(fwd[5]) + Math.abs(fwd[6])) * half;
      const ez = (Math.abs(fwd[8]) + Math.abs(fwd[9]) + Math.abs(fwd[10])) * half;
      const mir = !!p.inst.mirror, sxm = p.size.x;
      const subs = p.subs;
      const pieces = np * np * np;
      // One pass per (sub-cell i, piece j): j runs through the pieces before i moves on.
      for (let i = 0, j = 0; i < subs.length; j + 1 < pieces ? j++ : ((j = 0), (i += 3))) {
        let mx = mir ? sxm - 2 * subs[i] - 1 : 2 * subs[i] + 1, my = 2 * subs[i + 1] + 1, mz = 2 * subs[i + 2] + 1;
        if (ks !== 1) {
          // Piece (a, b, c) of n³: its centre, in the enlarged model's voxels.
          const a = j % np, b = Math.floor(j / np) % np, c = Math.floor(j / (np * np));
          mx = ks * (mx - 1) + half * (2 * a + 1);
          my = ks * (my - 1) + half * (2 * b + 1);
          mz = ks * (mz - 1) + half * (2 * c + 1);
        }
        const wx = fwd[0] * mx + fwd[1] * my + fwd[2] * mz + fwd[3];
        const wy = fwd[4] * mx + fwd[5] * my + fwd[6] * mz + fwd[7];
        const wz = fwd[8] * mx + fwd[9] * my + fwd[10] * mz + fwd[11];
        // The world cells whose centre (c + 0.5) lies in the box, give or take the slack.
        let cx0 = Math.ceil(wx - ex - 0.5 - slack), cx1 = Math.floor(wx + ex - 0.5 + slack);
        let cy0 = Math.ceil(wy - ey - 0.5 - slack), cy1 = Math.floor(wy + ey - 0.5 + slack);
        let cz0 = Math.ceil(wz - ez - 0.5 - slack), cz1 = Math.floor(wz + ez - 0.5 + slack);
        if (cx0 < 0) cx0 = 0;
        if (cy0 < 0) cy0 = 0;
        if (cz0 < 0) cz0 = 0;
        if (cx1 > maxX) cx1 = maxX;
        if (cy1 > maxY) cy1 = maxY;
        if (cz1 > maxZ) cz1 = maxZ;
        if (cx0 > cx1 || cy0 > cy1 || cz0 > cz1) continue;
        // Inline: most bricks were marked by this instance's previous sub-cells (the cache).
        for (let gz = cz0 >> 3; gz <= cz1 >> 3; gz++)
          for (let gy = cy0 >> 3; gy <= cy1 >> 3; gy++)
            for (let gx = cx0 >> 3; gx <= cx1 >> 3; gx++) {
              const bi = gx + (gy + gz * by) * bx, h = (gx & 7) | ((gy & 3) << 3) | ((gz & 7) << 5);
              if (seenK[h] !== k || seenB[h] !== bi) markBrick(k, gx, gy, gz);
            }
      }
      if (marked) marked(k);
    }
  }
  const data = new Uint32Array(cells + slots + n);
  const base = new Uint32Array(tables);
  let at = cells;
  for (let t = 0; t < tables; t++) {
    base[t] = at;
    at += 64;
    for (let sc = 0; sc < 64; sc++) {
      const slot = t * 64 + sc, c = counts[slot];
      if (at - base[t] > 0xffff) throw new Error("buildSubLists: a sub-cell table outgrew its 16-bit offsets");
      data[base[t] + sc] = (((at - base[t]) << 16) | (masks[slot] << 8) | c) >>> 0;
      counts[slot] = at;
      at += c;
    }
  }
  for (let i = 0; i < n; i++) data[counts[pairSlot[i]]++] = pairK[i];
  for (let ci = 0; ci < cells; ci++) if (table[ci] >= 0) data[ci] = base[table[ci]] + 1;
  return { data, cellWords: data.slice(0, cells), stats: { tables, pairs: n } };
}

/**
 * CPU reading of a built region, as grid.wesl's instRange does: the instances listed for world
 * cell (x, y, z)'s brick, or null when its top cell has no table.
 */
export function readSubLists(lists: SubLists, topDim: readonly [number, number, number], x: number, y: number, z: number): number[] | null {
  const [tx, ty] = topDim;
  const d = lists.data;
  const ptr = d[Math.floor(x / TOP_B) + Math.floor(y / TOP_B) * tx + Math.floor(z / TOP_B) * tx * ty];
  if (!ptr) return null;
  const tbl = ptr - 1;
  const e = d[tbl + ((x >> 4) & 3) + ((y >> 4) & 3) * 4 + ((z >> 4) & 3) * 16];
  const bi = ((x >> 3) & 1) + ((y >> 3) & 1) * 2 + ((z >> 3) & 1) * 4;
  if (!((e >> (8 + bi)) & 1)) return [];
  const first = tbl + (e >>> 16);
  return Array.from(d.subarray(first, first + (e & 0xff)));
}

/** The most words {@link movingTable} writes: 64 entries and 64 lists of at most 255. */
export const MOVING_TABLE_WORDS = 64 + 64 * 255;

const mtCount = new Uint16Array(64), mtMask = new Uint8Array(64), mtAt = new Uint32Array(64), mtLast = new Int32Array(64);

/**
 * The sub-cell table of top cell (cx, cy, cz) while moving instances are in it: its static table
 * (when it has one) with the moving instances added to the sub-cells and bricks their boxes reach,
 * so a sample there still tests a short list, not the cell's whole one. Written to `out` at `at`
 * in the tables' layout (see the top of this file), lists in the cell list's order: static first.
 *
 * A moving instance is marked by its world cell box, the one in its record (words 17-22) that the
 * shader tests before it samples the instance, so nothing it draws is outside what is marked. That
 * is looser than the static marking (every brick of the box counts as occupied), and costs nothing
 * to build, which matters every frame.
 *
 * @param out - Where the table goes; needs {@link MOVING_TABLE_WORDS} words from `at`.
 * @param at - Its first word in `out`.
 * @param stat - The words holding the cell's static table, at `statOff`; `statOff` < 0 without one
 *   (a cell with no static instances).
 * @param moving - The moving instances' indices, of which the first `n` are in the cell's list
 *   (the static sub-lists and `n` together stay within 255 a list, as the cell's list does).
 * @param inst - The instance records, as i32.
 * @returns The words written.
 */
export function movingTable(
  out: Uint32Array, at: number, cx: number, cy: number, cz: number,
  stat: Uint32Array, statOff: number, moving: ArrayLike<number>, n: number, inst: Int32Array,
): number {
  const x0 = cx * TOP_B, y0 = cy * TOP_B, z0 = cz * TOP_B;
  for (let sc = 0; sc < 64; sc++) {
    const e = statOff < 0 ? 0 : stat[statOff + sc];
    mtCount[sc] = e & 0xff;
    mtMask[sc] = (e >>> 8) & 0xff;
    mtLast[sc] = -1;
  }
  // Two passes over the moving instances' bricks in this cell (local brick coordinates 0..7): count
  // and mask, then fill.
  const each = (visit: (sc: number, bit: number, i: number) => void) => {
    for (let i = 0; i < n; i++) {
      const o = moving[i] * INST_WORDS;
      const bx0 = Math.max(0, (inst[o + 17] - x0) >> 3), bx1 = Math.min(7, (inst[o + 20] - 1 - x0) >> 3);
      const by0 = Math.max(0, (inst[o + 18] - y0) >> 3), by1 = Math.min(7, (inst[o + 21] - 1 - y0) >> 3);
      const bz0 = Math.max(0, (inst[o + 19] - z0) >> 3), bz1 = Math.min(7, (inst[o + 22] - 1 - z0) >> 3);
      for (let gz = bz0; gz <= bz1; gz++)
        for (let gy = by0; gy <= by1; gy++)
          for (let gx = bx0; gx <= bx1; gx++) visit((gx >> 1) + (gy >> 1) * 4 + (gz >> 1) * 16, (gx & 1) + (gy & 1) * 2 + (gz & 1) * 4, i);
    }
  };
  each((sc, bit, i) => {
    mtMask[sc] |= 1 << bit;
    if (mtLast[sc] !== i) { mtLast[sc] = i; mtCount[sc]++; }
  });
  let to = at + 64;
  for (let sc = 0; sc < 64; sc++) {
    const e = statOff < 0 ? 0 : stat[statOff + sc], sn = e & 0xff;
    out[at + sc] = (((to - at) << 16) | (mtMask[sc] << 8) | mtCount[sc]) >>> 0;
    const from = statOff + (e >>> 16);
    for (let j = 0; j < sn; j++) out[to + j] = stat[from + j];
    mtAt[sc] = to + sn;
    to += mtCount[sc];
    mtLast[sc] = -1;
  }
  each((sc, _bit, i) => {
    if (mtLast[sc] !== i) { mtLast[sc] = i; out[mtAt[sc]++] = moving[i]; }
  });
  return to - at;
}

function grow(a: Uint32Array, need: number): Uint32Array<ArrayBuffer> {
  const out = new Uint32Array(Math.max(need, a.length * 2));
  out.set(a);
  return out;
}
