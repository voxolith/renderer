// Sparse brick storage for the voxel grid.
//
// A dense grid pays for air: the examples forest is 10% solid, so 90% of a
// 275 MB texture is nothing. This splits the world into 8^3 bricks and keeps
// only the ones that contain something, which is the standard GPU voxel layout
// (the leaf size in both GVDB and NanoVDB).
//
// Each brick also carries its own small palette, so a voxel is stored as a
// 4-bit index into 15 local entries rather than an 8-bit global one. Measured
// over the 4x4 forest, no 8^3 brick anywhere contains more than 16 distinct
// values and 98% contain 8 or fewer, so 4 bits is almost always enough — and
// halving the voxel payload is worth more than the palette costs.
//
// Nibble 0 is reserved for "empty", which leaves 15 usable entries. Content
// that needs more (a detailed .vox model, a Minecraft region) falls back to a
// second tier storing 8 bits per voxel and no palette. Same idea as Minecraft's
// own chunk sections, which grow bits-per-block as a section's palette fills;
// this engine already decodes that format in formats/minecraft/chunk.ts.

import type { DirtyBox } from "./box";

/** Brick edge in voxels. */
export const BRICK_B = 8;
const BRICK_VOXELS = BRICK_B * BRICK_B * BRICK_B; // 512
/** u32 words per brick: 4 bits/voxel. */
export const BRICK_WORDS_4 = BRICK_VOXELS / 8; // 64
/** u32 words per brick: 8 bits/voxel. */
export const BRICK_WORDS_8 = BRICK_VOXELS / 4; // 128
/** u32 words of palette per 4-bit brick: 15 entries + 1 unused, packed 2 per word. */
export const PALETTE_WORDS = 8;
/** Usable palette entries in a 4-bit brick (nibble 0 means empty). */
export const PALETTE_ENTRIES = 15;

/** Tier flag in an index entry. */
const TIER_BIT = 0x8000_0000;
const SLOT_MASK = 0x7fff_ffff;

export interface BrickStats {
  /** Bricks holding something. */
  used: number;
  /** Of those, how many needed the 8-bit fallback. */
  wide: number;
  /** Bytes of brick payload (excludes the index volume). */
  payloadBytes: number;
  /** Bytes the same grid would take dense. */
  denseBytes: number;
}

/** Which slots changed, so a caller can upload only those. */
export interface BrickEdit {
  /** Index-volume region to re-upload, in brick coords. */
  index: DirtyBox;
  /** Slots whose 4-bit payload or palette changed. */
  slots4: number[];
  /** Slots whose 8-bit payload changed. */
  slots8: number[];
}

/**
 * Sparse brick view over a dense grid.
 *
 * The dense array stays the source of truth — callers already hold one and edit
 * it — and this maintains the sparse mirror the GPU reads. Rebuilding a brick is
 * cheap (512 voxel reads), so edits re-derive whole bricks rather than trying to
 * patch them in place.
 */
export class BrickGrid {
  /** Brick-space dimensions. */
  readonly dim: [number, number, number];
  /** One entry per brick: 0 = empty, else (slot + 1) with TIER_BIT set for 8-bit. */
  readonly index: Uint32Array;
  /** 4-bit brick payloads, BRICK_WORDS_4 per slot. */
  voxels4: Uint32Array;
  /** Per-brick palettes, PALETTE_WORDS per slot, two u16 global ids per word. */
  palettes: Uint32Array;
  /** 8-bit brick payloads, BRICK_WORDS_8 per slot. */
  voxels8: Uint32Array;

  private readonly size: { x: number; y: number; z: number };
  private slots4 = 0;
  private slots8 = 0;
  private free4: number[] = [];
  private free8: number[] = [];

  constructor(size: { x: number; y: number; z: number }, data: Uint8Array) {
    this.size = size;
    this.dim = [
      Math.ceil(size.x / BRICK_B),
      Math.ceil(size.y / BRICK_B),
      Math.ceil(size.z / BRICK_B),
    ];
    this.index = new Uint32Array(this.dim[0] * this.dim[1] * this.dim[2]);
    // Start small; the pools grow geometrically as bricks are claimed.
    this.voxels4 = new Uint32Array(BRICK_WORDS_4 * 64);
    this.palettes = new Uint32Array(PALETTE_WORDS * 64);
    this.voxels8 = new Uint32Array(BRICK_WORDS_8 * 4);
    this.rebuildAll(data);
  }

  get slotCount4(): number {
    return this.slots4;
  }
  get slotCount8(): number {
    return this.slots8;
  }

  stats(): BrickStats {
    let used = 0;
    let wide = 0;
    for (const e of this.index) {
      if (!e) continue;
      used++;
      if (e & TIER_BIT) wide++;
    }
    return {
      used,
      wide,
      payloadBytes:
        this.slots4 * (BRICK_WORDS_4 + PALETTE_WORDS) * 4 + this.slots8 * BRICK_WORDS_8 * 4,
      denseBytes: this.size.x * this.size.y * this.size.z,
    };
  }

  /** Full rebuild from the dense grid. */
  rebuildAll(data: Uint8Array): void {
    this.index.fill(0);
    this.slots4 = 0;
    this.slots8 = 0;
    this.free4.length = 0;
    this.free8.length = 0;
    const [bx, by, bz] = this.dim;
    for (let z = 0; z < bz; z++)
      for (let y = 0; y < by; y++) for (let x = 0; x < bx; x++) this.rebuildBrick(data, x, y, z);
  }

  /**
   * Re-derive every brick overlapping a fine-voxel dirty box. Returns the slots
   * that changed so the caller can upload just those.
   */
  rebuildBox(data: Uint8Array, box: DirtyBox): BrickEdit {
    const [bx, by, bz] = this.dim;
    const x0 = Math.max(0, (box.x0 / BRICK_B) | 0);
    const y0 = Math.max(0, (box.y0 / BRICK_B) | 0);
    const z0 = Math.max(0, (box.z0 / BRICK_B) | 0);
    const x1 = Math.min(bx - 1, (box.x1 / BRICK_B) | 0);
    const y1 = Math.min(by - 1, (box.y1 / BRICK_B) | 0);
    const z1 = Math.min(bz - 1, (box.z1 / BRICK_B) | 0);
    const edit: BrickEdit = { index: { x0, y0, z0, x1, y1, z1 }, slots4: [], slots8: [] };
    for (let z = z0; z <= z1; z++)
      for (let y = y0; y <= y1; y++)
        for (let x = x0; x <= x1; x++) {
          const e = this.rebuildBrick(data, x, y, z);
          if (!e) continue;
          const slot = (e & SLOT_MASK) - 1;
          (e & TIER_BIT ? edit.slots8 : edit.slots4).push(slot);
        }
    return edit;
  }

  /** Read back a voxel from the sparse form. For verification and CPU picking. */
  get(x: number, y: number, z: number): number {
    const { x: sx, y: sy, z: sz } = this.size;
    if (x < 0 || y < 0 || z < 0 || x >= sx || y >= sy || z >= sz) return 0;
    const [bx, by] = this.dim;
    const e = this.index[((x / BRICK_B) | 0) + ((y / BRICK_B) | 0) * bx + ((z / BRICK_B) | 0) * bx * by];
    if (!e) return 0;
    const li = (x % BRICK_B) + (y % BRICK_B) * BRICK_B + (z % BRICK_B) * BRICK_B * BRICK_B;
    const slot = (e & SLOT_MASK) - 1;
    if (e & TIER_BIT) {
      const w = this.voxels8[slot * BRICK_WORDS_8 + (li >> 2)];
      return (w >>> ((li & 3) * 8)) & 0xff;
    }
    const w = this.voxels4[slot * BRICK_WORDS_4 + (li >> 3)];
    const nib = (w >>> ((li & 7) * 4)) & 0xf;
    if (nib === 0) return 0;
    const p = this.palettes[slot * PALETTE_WORDS + ((nib - 1) >> 1)];
    return (p >>> (((nib - 1) & 1) * 16)) & 0xffff;
  }

  // --- internals ------------------------------------------------------------

  /** Rebuild one brick from the dense grid; returns its new index entry. */
  private rebuildBrick(data: Uint8Array, bxi: number, byi: number, bzi: number): number {
    const { x: sx, y: sy, z: sz } = this.size;
    const [bx, by] = this.dim;
    const ii = bxi + byi * bx + bzi * bx * by;
    const prev = this.index[ii];

    // Gather the brick's contents and its distinct values in one pass.
    const cells = new Uint8Array(BRICK_VOXELS);
    const seen = new Map<number, number>(); // value -> palette entry (1-based)
    let any = false;
    const ox = bxi * BRICK_B;
    const oy = byi * BRICK_B;
    const oz = bzi * BRICK_B;
    for (let lz = 0; lz < BRICK_B; lz++) {
      const wz = oz + lz;
      if (wz >= sz) break;
      for (let ly = 0; ly < BRICK_B; ly++) {
        const wy = oy + ly;
        if (wy >= sy) break;
        const row = wy * sx + wz * sx * sy;
        for (let lx = 0; lx < BRICK_B; lx++) {
          const wx = ox + lx;
          if (wx >= sx) break;
          const v = data[row + wx];
          if (v === 0) continue;
          cells[lx + ly * BRICK_B + lz * BRICK_B * BRICK_B] = v;
          any = true;
          if (!seen.has(v)) seen.set(v, seen.size + 1);
        }
      }
    }

    if (!any) {
      this.release(prev);
      this.index[ii] = 0;
      return 0;
    }

    const wide = seen.size > PALETTE_ENTRIES;
    // Reuse the existing slot when the tier is unchanged, so a repeated edit in
    // one place does not churn the free list.
    let slot: number;
    if (prev && !!(prev & TIER_BIT) === wide) {
      slot = (prev & SLOT_MASK) - 1;
    } else {
      this.release(prev);
      slot = wide ? this.claim8() : this.claim4();
    }

    if (wide) {
      const base = slot * BRICK_WORDS_8;
      this.voxels8.fill(0, base, base + BRICK_WORDS_8);
      for (let i = 0; i < BRICK_VOXELS; i++) {
        const v = cells[i];
        if (v) this.voxels8[base + (i >> 2)] |= v << ((i & 3) * 8);
      }
      this.index[ii] = (slot + 1) | TIER_BIT;
      return this.index[ii];
    }

    const vbase = slot * BRICK_WORDS_4;
    const pbase = slot * PALETTE_WORDS;
    this.voxels4.fill(0, vbase, vbase + BRICK_WORDS_4);
    this.palettes.fill(0, pbase, pbase + PALETTE_WORDS);
    for (const [value, entry] of seen) {
      const e = entry - 1;
      this.palettes[pbase + (e >> 1)] |= (value & 0xffff) << ((e & 1) * 16);
    }
    for (let i = 0; i < BRICK_VOXELS; i++) {
      const v = cells[i];
      if (!v) continue;
      this.voxels4[vbase + (i >> 3)] |= seen.get(v)! << ((i & 7) * 4);
    }
    this.index[ii] = slot + 1;
    return this.index[ii];
  }

  private release(entry: number): void {
    if (!entry) return;
    const slot = (entry & SLOT_MASK) - 1;
    if (entry & TIER_BIT) this.free8.push(slot);
    else this.free4.push(slot);
  }

  private claim4(): number {
    const reused = this.free4.pop();
    if (reused !== undefined) return reused;
    const slot = this.slots4++;
    if ((slot + 1) * BRICK_WORDS_4 > this.voxels4.length) {
      this.voxels4 = grow(this.voxels4, (slot + 1) * BRICK_WORDS_4);
      this.palettes = grow(this.palettes, (slot + 1) * PALETTE_WORDS);
    }
    return slot;
  }

  private claim8(): number {
    const reused = this.free8.pop();
    if (reused !== undefined) return reused;
    const slot = this.slots8++;
    if ((slot + 1) * BRICK_WORDS_8 > this.voxels8.length) {
      this.voxels8 = grow(this.voxels8, (slot + 1) * BRICK_WORDS_8);
    }
    return slot;
  }
}

function grow(a: Uint32Array, need: number): Uint32Array {
  let len = Math.max(a.length * 2, 1024);
  while (len < need) len *= 2;
  const next = new Uint32Array(len);
  next.set(a);
  return next;
}
