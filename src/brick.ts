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
//
// A brick holding one value throughout (solid ground, the inside of a wall) is
// stored in its index entry alone, with no payload.
//
// The index has two levels. A dense per-brick index costs 4 bytes per 512
// voxels whether they hold anything or not, which at 1 cm voxels is gigabytes
// for a valley. So bricks are grouped in 8^3 blocks (64^3 voxels): a top-level
// array holds one entry per block, and only blocks with something in them
// exist. A null top entry is also the renderer's big empty-space skip.
//
// Several grids can share one pool of bricks and blocks: the world, and every
// model drawn as an instance (see renderer.addModel), which is what lets one
// GPU buffer serve them all.

import type { DirtyBox } from "./box";

/** Brick edge in voxels. */
export const BRICK_B = 8;
const BRICK_VOXELS = BRICK_B * BRICK_B * BRICK_B; // 512
/** Top-level cell edge in voxels: 8^3 bricks. */
export const TOP_B = 64;
/** Brick entries per index block. */
export const BLOCK_ENTRIES = 512;
/** u32 words per brick: 4 bits/voxel. */
export const BRICK_WORDS_4 = BRICK_VOXELS / 8; // 64
/** u32 words per brick: 8 bits/voxel. */
export const BRICK_WORDS_8 = BRICK_VOXELS / 4; // 128
/** u32 words of palette per 4-bit brick: 15 entries + 1 unused, packed 2 per word. */
export const PALETTE_WORDS = 8;
/** Usable palette entries in a 4-bit brick (nibble 0 means empty). */
export const PALETTE_ENTRIES = 15;

/** Index entry flags. An entry is 0 (empty), a slot (+1) or a uniform value. */
export const TIER_BIT = 0x8000_0000;
/** The whole brick is the value in the low 16 bits; no payload. */
export const UNIFORM_BIT = 0x4000_0000;
/**
 * Empty, but next to a brick that is not (model grids only). A ray through a
 * rotated instance cannot trust "empty" at one brick, so it skips only where
 * the neighbourhood is empty too.
 */
export const NEAR_BIT = 0x2000_0000;
/** Index entry bits holding the payload slot + 1. */
export const SLOT_MASK = 0x1fff_ffff;

/** Memory use of one `BrickGrid`, from `stats()`. */
export interface BrickStats {
  /** Bricks holding something (uniform ones included). */
  used: number;
  /** Of those, how many needed the 8-bit fallback. */
  wide: number;
  /** Of those, how many are one value throughout and cost no payload. */
  uniform: number;
  /** Index blocks in use. */
  blocks: number;
  /** Bytes of brick payload (excludes the index). */
  payloadBytes: number;
  /** Bytes the same grid would take dense. */
  denseBytes: number;
}

/** What changed, so a caller can upload only that. */
export interface BrickEdit {
  /** Slots whose 4-bit payload or palette changed. */
  slots4: number[];
  /** Slots whose 8-bit payload changed. */
  slots8: number[];
  /** Index blocks whose entries changed. */
  blocks: number[];
  /** Top-level entries (indices into the grid's `top`) that changed. */
  tops: number[];
}

/** What `BrickPool.claimFor` claimed for a payload, to fill with `copyIn` or give back with `unclaim`. */
export interface PoolClaim {
  /** The pool slot of each local 4-bit slot. */
  slots4: Uint32Array;
  /** The pool slot of each local 8-bit slot. */
  slots8: Uint32Array;
  /** The pool block of each local index block. */
  blocks: Uint32Array;
  /** How many of `slots4` came off the free list (the first ones; the rest are new, consecutive). */
  reused4: number;
  /** How many of `slots8` came off the free list. */
  reused8: number;
  /** How many of `blocks` came off the free list. */
  reusedBlocks: number;
}

/** A `BrickEdit` with nothing in it, to accumulate several edits into. */
export const emptyEdit = (): BrickEdit => ({ slots4: [], slots8: [], blocks: [], tops: [] });

/** Brick payloads and index blocks, shared by every grid built on it. */
export class BrickPool {
  /** 4-bit brick payloads, BRICK_WORDS_4 per slot. */
  voxels4 = new Uint32Array(BRICK_WORDS_4 * 64);
  /** Per-brick palettes, PALETTE_WORDS per slot, two u16 global ids per word. */
  palettes = new Uint32Array(PALETTE_WORDS * 64);
  /** 8-bit brick payloads, BRICK_WORDS_8 per slot. */
  voxels8 = new Uint32Array(BRICK_WORDS_8 * 4);
  /** Index blocks, BLOCK_ENTRIES brick entries each. */
  blocks = new Uint32Array(BLOCK_ENTRIES * 16);
  private blockUsed = new Uint16Array(16);
  slots4 = 0;
  slots8 = 0;
  blockCount = 0;
  private free4: number[] = [];
  private free8: number[] = [];
  private freeBlocks: number[] = [];
  private readonly lut = new Uint8Array(256);
  private readonly used = new Uint8Array(256);
  private max4 = Infinity;
  private max8 = Infinity;
  private overflow?: (slots4: number, slots8: number) => Error;

  /**
   * Cap the slots of each tier, for a pool whose payloads are mirrored into GPU buffers of a
   * bounded size (the renderer passes its device's buffer limits). The CPU arrays then never grow
   * past what the GPU could hold, and a claim or reservation past the cap throws `overflow`'s
   * error before anything is claimed or allocated. Default: no cap.
   *
   * @param max4 - Most 4-bit slots (and palettes).
   * @param max8 - Most 8-bit slots.
   * @param overflow - Makes the error for slot counts that do not fit.
   */
  setSlotLimits(max4: number, max8: number, overflow: (slots4: number, slots8: number) => Error): void {
    this.max4 = max4;
    this.max8 = max8;
    this.overflow = overflow;
  }

  /** Make the 4-bit arrays hold `slots` slots (1.5x steps, capped at the limit); throws past it. */
  private fit4(slots: number): void {
    if (slots * BRICK_WORDS_4 <= this.voxels4.length) return;
    if (slots > this.max4) throw this.overflowError(slots, this.slots8);
    const n = grownSlots(this.voxels4.length / BRICK_WORDS_4, slots, this.max4);
    this.voxels4 = resized(this.voxels4, n * BRICK_WORDS_4);
    this.palettes = resized(this.palettes, n * PALETTE_WORDS);
  }

  /** Make the 8-bit array hold `slots` slots (1.5x steps, capped at the limit); throws past it. */
  private fit8(slots: number): void {
    if (slots * BRICK_WORDS_8 <= this.voxels8.length) return;
    if (slots > this.max8) throw this.overflowError(this.slots4, slots);
    this.voxels8 = resized(this.voxels8, grownSlots(this.voxels8.length / BRICK_WORDS_8, slots, this.max8) * BRICK_WORDS_8);
  }

  private overflowError(slots4: number, slots8: number): Error {
    return this.overflow?.(slots4, slots8) ?? new Error(`brick pool: ${slots4} 4-bit / ${slots8} 8-bit slots exceed the limit (${this.max4} / ${this.max8})`);
  }

  /** Take a zeroed index block (reused or new); returns its id. */
  claimBlock(): number {
    const reused = this.freeBlocks.pop();
    const b = reused ?? this.blockCount++;
    if ((b + 1) * BLOCK_ENTRIES > this.blocks.length) {
      this.blocks = grow(this.blocks, (b + 1) * BLOCK_ENTRIES);
      const u = new Uint16Array(this.blocks.length / BLOCK_ENTRIES);
      u.set(this.blockUsed);
      this.blockUsed = u;
    }
    this.blocks.fill(0, b * BLOCK_ENTRIES, (b + 1) * BLOCK_ENTRIES);
    this.blockUsed[b] = 0;
    return b;
  }

  /** Free an index block and every brick payload its entries hold. */
  releaseBlock(b: number): void {
    for (let i = b * BLOCK_ENTRIES, e = i + BLOCK_ENTRIES; i < e; i++) {
      this.release(this.blocks[i]);
      this.blocks[i] = 0;
    }
    this.blockUsed[b] = 0;
    this.freeBlocks.push(b);
  }

  /** Write one entry of a block; returns how many non-zero entries the block now has. */
  setBlockEntry(b: number, i: number, e: number): number {
    const at = b * BLOCK_ENTRIES + i;
    const was = this.blocks[at];
    if (!was && e) this.blockUsed[b]++;
    else if (was && !e) this.blockUsed[b]--;
    this.blocks[at] = e;
    return this.blockUsed[b];
  }

  /** Read a brick's 512 voxels into `out` (zero-filled when the entry holds none). */
  decode(e: number, out: Uint8Array): void {
    if (!(e & (SLOT_MASK | UNIFORM_BIT))) {
      out.fill(0);
      return;
    }
    if (e & UNIFORM_BIT) {
      out.fill(e & 0xffff);
      return;
    }
    const slot = (e & SLOT_MASK) - 1;
    if (e & TIER_BIT) {
      const base = slot * BRICK_WORDS_8;
      for (let i = 0; i < BRICK_VOXELS; i++) out[i] = (this.voxels8[base + (i >> 2)] >>> ((i & 3) * 8)) & 0xff;
      return;
    }
    const vbase = slot * BRICK_WORDS_4;
    const pbase = slot * PALETTE_WORDS;
    for (let i = 0; i < BRICK_VOXELS; i++) {
      const nib = (this.voxels4[vbase + (i >> 3)] >>> ((i & 7) * 4)) & 0xf;
      if (nib === 0) {
        out[i] = 0;
        continue;
      }
      const w = this.palettes[pbase + ((nib - 1) >> 1)];
      out[i] = (w >>> (((nib - 1) & 1) * 16)) & 0xffff;
    }
  }

  /** One voxel of a brick entry, local index `li`. */
  voxel(e: number, li: number): number {
    if (e & UNIFORM_BIT) return e & 0xffff;
    if (!(e & SLOT_MASK)) return 0;
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

  /**
   * Encode 512 voxels, replacing entry `prev` (whose slot is reused when the
   * tier matches, so a repeated edit in one place does not churn the free
   * list). Returns the new entry; the slot written goes into `edit`.
   */
  encode(prev: number, cells: Uint8Array, edit: BrickEdit): number {
    // Local palette via a reused lookup table (value -> 1-based entry) rather
    // than a Map: this runs for every brick a moving thing touches, every frame.
    const lut = this.lut, used = this.used;
    let count = 0, solid = 0;
    for (let i = 0; i < BRICK_VOXELS; i++) {
      const v = cells[i];
      if (v === 0) continue;
      solid++;
      if (lut[v]) continue;
      used[count++] = v;
      lut[v] = count;
    }
    const reset = () => { for (let k = 0; k < count; k++) lut[used[k]] = 0; };

    if (count === 0) {
      this.release(prev);
      return 0;
    }
    if (count === 1 && solid === BRICK_VOXELS) {
      reset();
      this.release(prev);
      return UNIFORM_BIT | used[0];
    }

    const wide = count > PALETTE_ENTRIES;
    let slot: number;
    const prevSlot = prev & SLOT_MASK && !(prev & UNIFORM_BIT);
    if (prevSlot && !!(prev & TIER_BIT) === wide) {
      slot = (prev & SLOT_MASK) - 1;
    } else {
      this.release(prev);
      slot = wide ? this.claim8() : this.claim4();
    }

    if (wide) {
      reset();
      const base = slot * BRICK_WORDS_8;
      this.voxels8.fill(0, base, base + BRICK_WORDS_8);
      for (let i = 0; i < BRICK_VOXELS; i++) {
        const v = cells[i];
        if (v) this.voxels8[base + (i >> 2)] |= v << ((i & 3) * 8);
      }
      edit.slots8.push(slot);
      return (slot + 1) | TIER_BIT;
    }

    const vbase = slot * BRICK_WORDS_4;
    const pbase = slot * PALETTE_WORDS;
    this.voxels4.fill(0, vbase, vbase + BRICK_WORDS_4);
    this.palettes.fill(0, pbase, pbase + PALETTE_WORDS);
    for (let e = 0; e < count; e++) this.palettes[pbase + (e >> 1)] |= (used[e] & 0xffff) << ((e & 1) * 16);
    for (let i = 0; i < BRICK_VOXELS; i++) {
      const v = cells[i];
      if (!v) continue;
      this.voxels4[vbase + (i >> 3)] |= lut[v] << ((i & 7) * 4);
    }
    reset();
    edit.slots4.push(slot);
    return slot + 1;
  }

  /** Return an entry's payload slot to the free list (uniform and empty entries hold none). */
  release(entry: number): void {
    if (!entry || entry & UNIFORM_BIT || !(entry & SLOT_MASK)) return;
    const slot = (entry & SLOT_MASK) - 1;
    if (entry & TIER_BIT) this.free8.push(slot);
    else this.free4.push(slot);
  }

  private claim4(): number {
    const reused = this.free4.pop();
    if (reused !== undefined) return reused;
    this.fit4(this.slots4 + 1);
    return this.slots4++;
  }

  private claim8(): number {
    const reused = this.free8.pop();
    if (reused !== undefined) return reused;
    this.fit8(this.slots8 + 1);
    return this.slots8++;
  }

  /**
   * Copy in bricks and index blocks encoded against a pool of their own (numbered from 0 in claim
   * order, as `encodeModel` writes them): claims one slot or block here per local one, in that
   * order, so the claims are the ones encoding straight into this pool would have made, and
   * rewrites the index entries to name the claimed slots.
   *
   * @param src - Local payloads (`voxels4` and `palettes` 64 and 8 words per slot, `voxels8` 128)
   *   and index blocks (512 entries each).
   * @returns The pool slot of each local slot of either tier, and the pool block of each local
   *   block: what was written, for the upload.
   */
  adopt(src: { blocks: Uint32Array; voxels4: Uint32Array; palettes: Uint32Array; voxels8: Uint32Array }): { slots4: Uint32Array; slots8: Uint32Array; blocks: Uint32Array } {
    const claim = this.claimFor(src);
    this.copyIn(src, claim, "4", 0, claim.slots4.length);
    this.copyIn(src, claim, "8", 0, claim.slots8.length);
    this.copyIn(src, claim, "blocks", 0, claim.blocks.length);
    return { slots4: claim.slots4, slots8: claim.slots8, blocks: claim.blocks };
  }

  /**
   * The first half of `adopt`, for a copy spread over several calls: claims the slots and (zeroed)
   * index blocks `adopt(src)` would claim, in the same order, and copies nothing. Fill them with
   * `copyIn`, in slices and in any order; give them back with `unclaim`. Other claims, edits and
   * growth may come in between: the claimed slots stay this claim's, and the arrays keep what was
   * copied in when they grow.
   *
   * @param src - Local payload and block counts, as for `adopt`.
   * @returns The claim: the pool slot of each local slot, the pool block of each local block, and
   *   what `unclaim` needs to undo it.
   */
  claimFor(src: { blocks: Uint32Array; voxels4: Uint32Array; voxels8: Uint32Array }): PoolClaim {
    const n4 = src.voxels4.length / BRICK_WORDS_4, n8 = src.voxels8.length / BRICK_WORDS_8, nb = src.blocks.length / BLOCK_ENTRIES;
    const reused4 = Math.min(n4, this.free4.length), reused8 = Math.min(n8, this.free8.length);
    const slots4 = this.claimMany(n4, false), slots8 = this.claimMany(n8, true), blocks = new Uint32Array(nb);
    const reusedBlocks = Math.min(nb, this.freeBlocks.length);
    for (let i = 0; i < nb; i++) blocks[i] = this.claimBlock();
    return { slots4, slots8, blocks, reused4, reused8, reusedBlocks };
  }

  /**
   * Copy local items `[from, to)` of one part of `src` into what `claimFor(src)` claimed: 4-bit
   * slots (payload and palette), 8-bit slots, or index blocks (entries rewritten to name the
   * claimed slots). Every item copied once, in any slices, leaves the pool as `adopt(src)` would.
   *
   * @param src - The payload `claimFor` was given.
   * @param claim - Its claim.
   * @param part - `"4"`, `"8"` or `"blocks"`.
   * @param from - First local item.
   * @param to - One past the last.
   */
  copyIn(src: { blocks: Uint32Array; voxels4: Uint32Array; palettes: Uint32Array; voxels8: Uint32Array }, claim: PoolClaim, part: "4" | "8" | "blocks", from: number, to: number): void {
    if (part === "4") {
      // Copy in runs of consecutive slots: slots claimed past the end are one run.
      copyRuns(claim.slots4, src.voxels4, this.voxels4, BRICK_WORDS_4, from, to);
      copyRuns(claim.slots4, src.palettes, this.palettes, PALETTE_WORDS, from, to);
      return;
    }
    if (part === "8") {
      copyRuns(claim.slots8, src.voxels8, this.voxels8, BRICK_WORDS_8, from, to);
      return;
    }
    const map4 = claim.slots4, map8 = claim.slots8, blocks = this.blocks;
    for (let i = from; i < to; i++) {
      const b = claim.blocks[i];
      let used = 0;
      for (let j = 0, s = i * BLOCK_ENTRIES, d = b * BLOCK_ENTRIES; j < BLOCK_ENTRIES; j++) {
        let e = src.blocks[s + j];
        if (!e) continue;
        used++;
        if (!(e & UNIFORM_BIT) && e & SLOT_MASK) {
          const local = (e & SLOT_MASK) - 1;
          e = (e & ~SLOT_MASK) | ((e & TIER_BIT ? map8[local] : map4[local]) + 1);
        }
        blocks[d + j] = e;
      }
      this.blockUsed[b] = used;
    }
  }

  /**
   * Give back what `claimFor` claimed, copied in or not. Its blocks are zeroed (as freed blocks
   * are) and its slots past the pool's end, when nothing was claimed after them, are zeroed and
   * unclaimed, so with nothing claimed or freed in between the pool is as it was before
   * `claimFor`: the same counts and free lists, the same words at every slot in use and every
   * block, zeros past the end. Otherwise the slots and blocks go onto the free lists. Either way
   * the free slots it had reused keep what was copied into them (a free slot's words are never
   * read).
   *
   * @param claim - A claim of this pool, not given back yet.
   * @returns The pool slot ranges that were zeroed, `[lo, hi)` per tier, for a caller mirroring
   *   the pool elsewhere.
   */
  unclaim(claim: PoolClaim): { zeroed4: [number, number]; zeroed8: [number, number] } {
    // Blocks first, last claimed first, so the free list gets its old order back.
    for (let i = claim.blocks.length - 1; i >= 0; i--) {
      const b = claim.blocks[i];
      this.blocks.fill(0, b * BLOCK_ENTRIES, (b + 1) * BLOCK_ENTRIES);
      this.blockUsed[b] = 0;
      if (i >= claim.reusedBlocks && b === this.blockCount - 1) this.blockCount--;
      else this.freeBlocks.push(b);
    }
    const back = (map: Uint32Array, reused: number, wide: boolean): [number, number] => {
      const free = wide ? this.free8 : this.free4;
      let zeroed: [number, number] = [0, 0];
      const rest = map.length - reused;
      if (rest > 0) {
        const first = map[reused], end = wide ? this.slots8 : this.slots4;
        if (first + rest === end) {
          if (wide) {
            this.slots8 = first;
            this.voxels8.fill(0, first * BRICK_WORDS_8, end * BRICK_WORDS_8);
          } else {
            this.slots4 = first;
            this.voxels4.fill(0, first * BRICK_WORDS_4, end * BRICK_WORDS_4);
            this.palettes.fill(0, first * PALETTE_WORDS, end * PALETTE_WORDS);
          }
          zeroed = [first, end];
        } else for (let i = map.length - 1; i >= reused; i--) free.push(map[i]);
      }
      // Popped from the top of the free list in order, so pushed back in reverse.
      for (let i = reused - 1; i >= 0; i--) free.push(map[i]);
      return zeroed;
    };
    const zeroed8 = back(claim.slots8, claim.reused8, true);
    const zeroed4 = back(claim.slots4, claim.reused4, false);
    return { zeroed4, zeroed8 };
  }

  /**
   * Make room in the CPU payload arrays for `slots4` and `slots8` slots in all (grown as claims
   * grow them, so only the spare capacity past the claimed slots differs). Claims nothing. Throws,
   * changing nothing, past the limits of `setSlotLimits`.
   *
   * @param slots4 - 4-bit slots to have room for, claimed ones included.
   * @param slots8 - 8-bit slots to have room for, claimed ones included.
   */
  reserve(slots4: number, slots8: number): void {
    if (slots4 > this.max4 || slots8 > this.max8) throw this.overflowError(slots4, slots8);
    this.fit4(slots4);
    this.fit8(slots8);
  }

  /** `n` slots of one tier, as `n` calls of claim4 / claim8 would give them (freed ones first, last freed first). */
  private claimMany(n: number, wide: boolean): Uint32Array {
    const map = new Uint32Array(n);
    const free = wide ? this.free8 : this.free4;
    let i = 0;
    for (; i < n && free.length; i++) map[i] = free.pop()!;
    if (i === n) return map;
    // The rest past the end: one growth, to the size the one-at-a-time claims would have grown to.
    // Grown (or refused) before anything is claimed, so a throw leaves the pool as it was.
    const first = wide ? this.slots8 : this.slots4, rest = n - i;
    if (wide) this.fit8(first + rest);
    else this.fit4(first + rest);
    for (let k = 0; k < rest; k++) map[i + k] = first + k;
    if (wide) this.slots8 += rest;
    else this.slots4 += rest;
    return map;
  }

  /** Bytes of payload claimed so far. */
  payloadBytes(): number {
    return this.slots4 * (BRICK_WORDS_4 + PALETTE_WORDS) * 4 + this.slots8 * BRICK_WORDS_8 * 4;
  }
}

/**
 * Sparse view of one grid (the world, or one model) over a pool.
 *
 * `fill` callbacks and edits work a brick at a time, so a world can be built
 * and later edited without a dense array of it ever existing. Headless: the
 * renderer keeps one internally, and tools use it to measure a scene or to
 * check what a GPU upload would hold.
 *
 * @example
 * ```ts
 * import { BrickGrid } from "@voxolith/renderer/core";
 *
 * // An empty 256×64×256 world, filled a brick at a time: ground below y = 10.
 * const grid = new BrickGrid({ x: 256, y: 64, z: 256 });
 * grid.editBox({ x0: 0, y0: 0, z0: 0, x1: 255, y1: 9, z1: 255 }, (cells, ox, oy, oz) => {
 *   for (let lz = 0; lz < 8; lz++)
 *     for (let ly = 0; ly < 8; ly++)
 *       for (let lx = 0; lx < 8; lx++) if (oy + ly < 10) cells[lx + ly * 8 + lz * 64] = 1;
 *   return true;
 * });
 * grid.get(3, 9, 3); // 1
 * grid.stats(); // { used, uniform, payloadBytes, denseBytes, ... }
 * ```
 */
export class BrickGrid {
  /** Brick-space dimensions. */
  readonly dim: [number, number, number];
  /** Top-level dimensions (64^3-voxel cells). */
  readonly topDim: [number, number, number];
  /** One entry per top cell: 0 = nothing there, else pool block id + 1. */
  readonly top: Uint32Array;
  /** Where the payloads and index blocks live; shared when one was passed in. */
  readonly pool: BrickPool;
  private readonly size: { x: number; y: number; z: number };
  /** Reused 512-voxel staging buffer; a brick is never big enough to justify allocating one per call. */
  private readonly scratch = new Uint8Array(BRICK_VOXELS);

  /**
   * Build from a dense grid, or omit `data` for an empty world that is filled
   * through `editBox`. The second form is what lets a large scene exist without
   * ever materialising a dense mirror of itself.
   */
  constructor(size: { x: number; y: number; z: number }, data?: Uint8Array, pool?: BrickPool) {
    this.size = size;
    this.pool = pool ?? new BrickPool();
    this.dim = [Math.ceil(size.x / BRICK_B), Math.ceil(size.y / BRICK_B), Math.ceil(size.z / BRICK_B)];
    this.topDim = [Math.ceil(this.dim[0] / 8), Math.ceil(this.dim[1] / 8), Math.ceil(this.dim[2] / 8)];
    this.top = new Uint32Array(this.topDim[0] * this.topDim[1] * this.topDim[2]);
    if (data) this.rebuildAll(data);
  }

  /** 4-bit payload slots claimed in the pool (by every grid sharing it). */
  get slotCount4(): number {
    return this.pool.slots4;
  }
  /** 8-bit payload slots claimed in the pool (by every grid sharing it). */
  get slotCount8(): number {
    return this.pool.slots8;
  }

  /** Index entry of brick (bx, by, bz); 0 when empty or out of range. */
  entry(bx: number, by: number, bz: number): number {
    const [dx, dy, dz] = this.dim;
    if (bx < 0 || by < 0 || bz < 0 || bx >= dx || by >= dy || bz >= dz) return 0;
    const [tx, ty] = this.topDim;
    const t = this.top[(bx >> 3) + (by >> 3) * tx + (bz >> 3) * tx * ty];
    if (!t) return 0;
    return this.pool.blocks[(t - 1) * BLOCK_ENTRIES + (bx & 7) + (by & 7) * 8 + (bz & 7) * 64];
  }

  /** Set brick (bx, by, bz)'s entry, claiming or freeing its block. The old entry's payload is not released. */
  setEntry(bx: number, by: number, bz: number, e: number, edit: BrickEdit): void {
    const [tx, ty] = this.topDim;
    const ti = (bx >> 3) + (by >> 3) * tx + (bz >> 3) * tx * ty;
    let t = this.top[ti];
    if (!t) {
      if (!e) return;
      t = this.pool.claimBlock() + 1;
      this.top[ti] = t;
      edit.tops.push(ti);
    }
    const li = (bx & 7) + (by & 7) * 8 + (bz & 7) * 64;
    const n = this.pool.setBlockEntry(t - 1, li, e);
    if (n === 0) {
      this.pool.releaseBlock(t - 1);
      this.top[ti] = 0;
      edit.tops.push(ti);
      return;
    }
    edit.blocks.push(t - 1);
  }

  /** Count this grid's bricks and blocks; `payloadBytes` is the whole pool's. Walks every block. */
  stats(): BrickStats {
    let used = 0, wide = 0, uniform = 0, blocks = 0;
    for (const t of this.top) {
      if (!t) continue;
      blocks++;
      for (let i = (t - 1) * BLOCK_ENTRIES, e = i + BLOCK_ENTRIES; i < e; i++) {
        const v = this.pool.blocks[i];
        if (!(v & (SLOT_MASK | UNIFORM_BIT))) continue;
        used++;
        if (v & UNIFORM_BIT) uniform++;
        else if (v & TIER_BIT) wide++;
      }
    }
    return {
      used,
      wide,
      uniform,
      blocks,
      payloadBytes: this.pool.payloadBytes(),
      denseBytes: this.size.x * this.size.y * this.size.z,
    };
  }

  /** Full rebuild from the dense grid. */
  rebuildAll(data: Uint8Array): BrickEdit {
    const edit = emptyEdit();
    for (let i = 0; i < this.top.length; i++) {
      if (!this.top[i]) continue;
      this.pool.releaseBlock(this.top[i] - 1);
      this.top[i] = 0;
      edit.tops.push(i);
    }
    const [bx, by, bz] = this.dim;
    for (let z = 0; z < bz; z++)
      for (let y = 0; y < by; y++) for (let x = 0; x < bx; x++) this.rebuildBrick(data, x, y, z, edit);
    return edit;
  }

  /**
   * Re-derive every brick overlapping a fine-voxel dirty box. Returns what
   * changed so the caller can upload just that.
   */
  rebuildBox(data: Uint8Array, box: DirtyBox): BrickEdit {
    const edit = emptyEdit();
    const r = this.brickRange(box);
    if (!r) return edit;
    for (let z = r[2]; z <= r[5]; z++)
      for (let y = r[1]; y <= r[4]; y++)
        for (let x = r[0]; x <= r[3]; x++) this.rebuildBrick(data, x, y, z, edit);
    return edit;
  }

  /**
   * Edit every brick overlapping `box` in place.
   *
   * `fill` receives the brick's current 512 voxels (local index
   * `lx + ly*8 + lz*64`) and its world-space origin, mutates what it wants, and
   * returns whether it changed anything. Bricks are the source of truth here —
   * nothing reads back from a dense array — so a world can be built, and later
   * edited, without one existing at all.
   */
  editBox(box: DirtyBox, fill: (cells: Uint8Array, ox: number, oy: number, oz: number) => boolean, edit = emptyEdit()): BrickEdit {
    const r = this.brickRange(box);
    if (!r) return edit;
    const cells = this.scratch;
    for (let z = r[2]; z <= r[5]; z++)
      for (let y = r[1]; y <= r[4]; y++)
        for (let x = r[0]; x <= r[3]; x++) {
          const prev = this.entry(x, y, z);
          this.pool.decode(prev, cells);
          if (!fill(cells, x * BRICK_B, y * BRICK_B, z * BRICK_B)) continue;
          const e = this.pool.encode(prev, cells, edit);
          if (e !== prev) this.setEntry(x, y, z, e, edit);
        }
    return edit;
  }

  /**
   * Drop every brick fully inside `box`, returning their slots to the pool.
   * Used when a chunk leaves the resident set: its bricks are the only place
   * its voxels existed, so freeing them frees the memory.
   *
   * Bricks only partly covered are edited rather than dropped, so a box that
   * does not land on brick boundaries cannot delete a neighbour's content.
   */
  clearBox(box: DirtyBox): BrickEdit {
    const edit = emptyEdit();
    const r = this.brickRange(box);
    if (!r) return edit;
    const cells = this.scratch;
    for (let z = r[2]; z <= r[5]; z++)
      for (let y = r[1]; y <= r[4]; y++)
        for (let x = r[0]; x <= r[3]; x++) {
          const prev = this.entry(x, y, z);
          if (!prev) continue;
          const ox = x * BRICK_B, oy = y * BRICK_B, oz = z * BRICK_B;
          const whole =
            ox >= box.x0 && oy >= box.y0 && oz >= box.z0 &&
            ox + BRICK_B - 1 <= box.x1 && oy + BRICK_B - 1 <= box.y1 && oz + BRICK_B - 1 <= box.z1;
          if (whole) {
            this.pool.release(prev);
            this.setEntry(x, y, z, 0, edit);
            continue;
          }
          this.pool.decode(prev, cells);
          for (let lz = 0; lz < BRICK_B; lz++)
            for (let ly = 0; ly < BRICK_B; ly++)
              for (let lx = 0; lx < BRICK_B; lx++) {
                const wx = ox + lx, wy = oy + ly, wz = oz + lz;
                if (wx < box.x0 || wx > box.x1 || wy < box.y0 || wy > box.y1 || wz < box.z0 || wz > box.z1) continue;
                cells[lx + ly * BRICK_B + lz * BRICK_B * BRICK_B] = 0;
              }
          const e = this.pool.encode(prev, cells, edit);
          if (e !== prev) this.setEntry(x, y, z, e, edit);
        }
    return edit;
  }

  /** Release everything this grid holds (a model being removed). */
  free(): void {
    for (let i = 0; i < this.top.length; i++) {
      if (!this.top[i]) continue;
      this.pool.releaseBlock(this.top[i] - 1);
      this.top[i] = 0;
    }
  }

  /**
   * Mark empty bricks next to occupied ones with NEAR_BIT (model grids).
   * Changes only index entries, never payloads.
   *
   * @param edit - Receives the blocks and top entries changed.
   * @param tick - Called after each z slab of bricks with the bricks it covered (progress).
   */
  markNear(edit: BrickEdit, tick?: (bricks: number) => void): void {
    const [dx, dy, dz] = this.dim;
    const [tx, ty] = this.topDim;
    const sx = dx, sxy = dx * dy;
    // Each brick's state from the blocks themselves: 1 occupied, 2 marked near already, 0 empty.
    const state = new Uint8Array(dx * dy * dz);
    for (let ti = 0; ti < this.top.length; ti++) {
      const t = this.top[ti];
      if (!t) continue;
      const X = (ti % tx) * 8, Y = (Math.floor(ti / tx) % ty) * 8, Z = Math.floor(ti / (tx * ty)) * 8;
      for (let li = 0, at = (t - 1) * BLOCK_ENTRIES; li < BLOCK_ENTRIES; li++) {
        const e = this.pool.blocks[at + li];
        if (!e) continue;
        const x = X + (li & 7), y = Y + ((li >> 3) & 7), z = Z + (li >> 6);
        if (x >= dx || y >= dy || z >= dz) continue;
        state[x + y * sx + z * sxy] = e & (SLOT_MASK | UNIFORM_BIT) ? 1 : e === NEAR_BIT ? 2 : 3;
      }
    }
    // Near = within the 3³ neighbourhood of an occupied brick: a box dilation, one axis at a time.
    const a = new Uint8Array(state.length), b = new Uint8Array(state.length);
    for (let i = 0; i < state.length; i++) a[i] = state[i] === 1 ? 1 : 0;
    for (let r = 0; r < dy * dz; r++) {
      const o = r * sx;
      for (let x = 0; x < dx; x++) b[o + x] = a[o + x] | (x > 0 ? a[o + x - 1] : 0) | (x < dx - 1 ? a[o + x + 1] : 0);
    }
    for (let z = 0; z < dz; z++)
      for (let y = 0; y < dy; y++) {
        const o = y * sx + z * sxy, lo = y > 0 ? o - sx : -1, hi = y < dy - 1 ? o + sx : -1;
        for (let x = 0; x < dx; x++) a[o + x] = b[o + x] | (lo >= 0 ? b[lo + x] : 0) | (hi >= 0 ? b[hi + x] : 0);
      }
    for (let z = 0; z < dz; z++) {
      const o = z * sxy, lo = z > 0 ? o - sxy : -1, hi = z < dz - 1 ? o + sxy : -1;
      for (let i = 0; i < sxy; i++) b[o + i] = a[o + i] | (lo >= 0 ? a[lo + i] : 0) | (hi >= 0 ? a[hi + i] : 0);
    }
    for (let z = 0; z < dz; z++) {
      for (let y = 0; y < dy; y++)
        for (let x = 0; x < dx; x++) {
          const i = x + y * sx + z * sxy, st = state[i];
          if (st === 1) continue;
          const want = b[i] ? NEAR_BIT : 0;
          // Same writes, in the same order, as comparing every entry: only what differs is set.
          const e = st === 0 ? 0 : st === 2 ? NEAR_BIT : this.entry(x, y, z);
          if (e !== want) this.setEntry(x, y, z, want, edit);
        }
      tick?.(sxy);
    }
  }

  /** Read back a voxel from the sparse form. For verification and CPU picking. */
  get(x: number, y: number, z: number): number {
    const { x: sx, y: sy, z: sz } = this.size;
    if (x < 0 || y < 0 || z < 0 || x >= sx || y >= sy || z >= sz) return 0;
    const e = this.entry((x / BRICK_B) | 0, (y / BRICK_B) | 0, (z / BRICK_B) | 0);
    if (!e) return 0;
    return this.pool.voxel(e, (x % BRICK_B) + (y % BRICK_B) * BRICK_B + (z % BRICK_B) * BRICK_B * BRICK_B);
  }

  // --- internals ------------------------------------------------------------

  private brickRange(box: DirtyBox): number[] | null {
    const [bx, by, bz] = this.dim;
    const x0 = Math.max(0, Math.floor(box.x0 / BRICK_B));
    const y0 = Math.max(0, Math.floor(box.y0 / BRICK_B));
    const z0 = Math.max(0, Math.floor(box.z0 / BRICK_B));
    const x1 = Math.min(bx - 1, Math.floor(box.x1 / BRICK_B));
    const y1 = Math.min(by - 1, Math.floor(box.y1 / BRICK_B));
    const z1 = Math.min(bz - 1, Math.floor(box.z1 / BRICK_B));
    if (x1 < x0 || y1 < y0 || z1 < z0) return null;
    return [x0, y0, z0, x1, y1, z1];
  }

  /** Rebuild one brick from the dense grid. */
  private rebuildBrick(data: Uint8Array, bxi: number, byi: number, bzi: number, edit: BrickEdit): void {
    const { x: sx, y: sy, z: sz } = this.size;
    const cells = this.scratch;
    cells.fill(0);
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
          cells[lx + ly * BRICK_B + lz * BRICK_B * BRICK_B] = data[row + wx];
        }
      }
    }
    const prev = this.entry(bxi, byi, bzi);
    const e = this.pool.encode(prev, cells, edit);
    if (e !== prev) this.setEntry(bxi, byi, bzi, e, edit);
  }
}

/** Copy local slot i's `words` words of `from` to slot `map[i]` of `to`, a run of consecutive slots at a time. */
function copyRuns(map: Uint32Array, from: Uint32Array, to: Uint32Array, words: number, lo = 0, hi = map.length): void {
  for (let i = lo; i < hi; ) {
    let j = i + 1;
    while (j < hi && map[j] === map[j - 1] + 1) j++;
    to.set(from.subarray(i * words, j * words), map[i] * words);
    i = j;
  }
}

/**
 * Slot capacity for payload arrays that must hold `need` slots, from `have`: 1.5x steps (at least
 * 64 slots), so at a few hundred MB the spare room and the copy stay a third of the array rather
 * than all of it, and never past `max`, since a mirror bigger than the GPU pool it feeds is
 * useless (at 1 GiB its doubling to 2 GiB failed to allocate in Chrome).
 */
function grownSlots(have: number, need: number, max: number): number {
  let n = Math.max(have, 64);
  while (n < need) n = Math.ceil(n * 1.5);
  return Math.min(n, max);
}

/** A copy of `a` with `len` words (zeros past its end). */
function resized(a: Uint32Array, len: number): Uint32Array<ArrayBuffer> {
  const next = new Uint32Array(len);
  next.set(a);
  return next;
}

function grow(a: Uint32Array, need: number): Uint32Array<ArrayBuffer> {
  let len = Math.max(a.length * 2, 1024);
  while (len < need) len *= 2;
  const next = new Uint32Array(len);
  next.set(a);
  return next;
}
