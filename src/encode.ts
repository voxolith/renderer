// Model encoding: everything `Renderer.addModel` computes on the CPU for a model (its 8³ brick
// payloads and index blocks, the part grid of a model with parts, the part boxes and the occupied
// 2³ sub-cells the placement bake reads), as one headless function on plain data. The renderer
// runs it on the main thread for the synchronous `addModel`; a host that wants no stall runs it in
// a worker (it imports nothing that needs a DOM, a GPU or a bundler) and hands the result to
// `Renderer.addEncodedModel`. Both paths run this same code, so they upload the same bytes.
//
// The encoding does not know the renderer's brick pool, so it is written against a pool of its
// own: payload slots and index blocks are numbered from 0 in the order they were claimed, and
// index entries and top entries name those local numbers. Applying it claims one real slot and
// block per local one, in that same order, and rewrites the references, which is exactly the
// order of claims `addModel` made when it encoded straight into the renderer's pool.

import { BrickGrid, BrickPool, BLOCK_ENTRIES, BRICK_B, BRICK_WORDS_4, BRICK_WORDS_8, PALETTE_WORDS, emptyEdit } from "./brick";
import { MAX_PARTS, partBoxes } from "./instance";
import { sparseDims, type SparseVoxels } from "./sparse";
import type { PlacementModel } from "./placement";

/**
 * The layout version of {@link EncodedModel}. It changes whenever the payload layout does (the
 * brick encoding, the index entry bits, the sub-cell list), so a host that caches encoded models
 * (in IndexedDB, say) keys them by it and never feeds a renderer stale bytes.
 * `Renderer.addEncodedModel` throws on any other version.
 */
export const ENCODED_MODEL_VERSION = 1;

/** A model drawn by instances; see `Renderer.addModel` and {@link encodeModel}. */
export interface ModelSource {
  /** Extent in voxels. */
  size: { x: number; y: number; z: number };
  /** Dense role values, `x + y*sx + z*sx*sy`; or `sparse`. */
  data?: Uint8Array;
  /** Sparse role values, in 8³ bricks; or `data`. */
  sparse?: SparseVoxels;
  /**
   * Part index per voxel (dense models only), laid out like `data`: e.g. a skeleton's bone per
   * voxel. A model with parts can be drawn with one transform per part (`Instance.parts`), posed
   * on the GPU from this single rest model. Parts must be ordered parents first; at most 32.
   */
  parts?: Uint8Array;
  /**
   * Per part, its parent (-1 for none) and the joint where it meets it, in model voxels. Cells
   * around a posed joint may be filled from either side, so a turned child stays attached.
   */
  joints?: readonly { parent: number; at: readonly [number, number, number] }[];
  /**
   * Per-part voxel boxes to use instead of this model's own (min x, y, z, max x, y, z each). Poses
   * are packed per set of boxes, so copies of a model that pass the same boxes object (a creature's
   * wounded copies, with the undamaged model's boxes, which contain theirs) share every pose.
   */
  partBoxes?: Int32Array;
}

/**
 * A model encoded for the GPU, not yet in any renderer: what {@link encodeModel} returns and
 * `Renderer.addEncodedModel` takes. Plain, structured-clone-friendly data; the big arrays each own
 * their buffer, so they transfer ({@link encodedTransferables}).
 *
 * Payload slots and index blocks are numbered locally, from 0 in the order they were claimed:
 * slot `i` of the 4-bit tier is words `[i * 64, i * 64 + 64)` of `voxels4` and
 * `[i * 8, i * 8 + 8)` of `palettes`; slot `i` of the 8-bit tier is words `[i * 128, i * 128 + 128)`
 * of `voxels8`; block `b` is words `[b * 512, b * 512 + 512)` of `blocks`, one index entry per brick
 * (the renderer's entry bits, with the slot + 1 naming a local slot). A top entry is a local block
 * + 1, or 0. The layout is versioned by {@link ENCODED_MODEL_VERSION}.
 *
 * It is a superset of what the placement bake reads about the model (`size`, `subs`, `partBoxes`,
 * `joints`: a `PlacementModel` without its keys), so a worker that encodes a model can register it
 * for placement too ({@link placementModelOf}) once the renderer has given it its keys.
 */
export interface EncodedModel {
  /** {@link ENCODED_MODEL_VERSION} at encoding time. */
  version: number;
  /** Extent in voxels. */
  size: { x: number; y: number; z: number };
  /** The role grid's top level: one entry per 64³ top cell, local block + 1 (0 = empty). */
  top: Uint32Array<ArrayBuffer>;
  /** The part grid's top level (part index + 1 per voxel), for a model with parts. */
  partTop?: Uint32Array<ArrayBuffer>;
  /** Index blocks, 512 entries each: the role grid's, then the part grid's. */
  blocks: Uint32Array<ArrayBuffer>;
  /** 4-bit brick payloads, 64 words per slot. */
  voxels4: Uint32Array<ArrayBuffer>;
  /** Palettes of the 4-bit bricks, 8 words per slot. */
  palettes: Uint32Array<ArrayBuffer>;
  /** 8-bit brick payloads, 128 words per slot. */
  voxels8: Uint32Array<ArrayBuffer>;
  /** The model's occupied 2³ sub-cells, as x, y, z triples (what the placement bake marks). */
  subs: Int32Array<ArrayBuffer>;
  /**
   * Per-part voxel boxes, for a model with parts (min x, y, z, max x, y, z each): the source's
   * `partBoxes` object itself when it gave one. Models added with the same boxes object share
   * packed poses, so a host that encodes copies of a model separately (their boxes arrive as
   * separate objects) can set one shared object here before `addEncodedModel` to keep that sharing.
   */
  partBoxes?: Int32Array;
  /** Per part, its parent (-1 for none) and the joint where it meets it (the source's `joints`). */
  joints?: ModelSource["joints"];
}

/** Options of {@link encodeModel}. */
export interface EncodeOptions {
  /**
   * Progress of the encode, for a loading screen. `done` and `total` count opaque work units,
   * scaled so that `done / total` grows roughly linearly with the encode's time; only that
   * fraction means anything. `total` is fixed for one encode. The first call is `(0, total)` and
   * the last is `(total, total)`, just before the encode returns; in between, calls come at most
   * every 50 ms or so, and `done` never decreases. Without it the encode does no progress work.
   */
  onProgress?: (done: number, total: number) => void;
}

/**
 * What a renderer gave a model when it added it, beyond its encoding: `PlacementModel.key`,
 * `poseKey` and `scale` (`Renderer.placementModel(id)` has all three). A host passing keys
 * between threads passes `scale` too, or a worker's bake of a scaled model is wrong.
 */
export type PlacementKeys = Pick<PlacementModel, "key" | "poseKey" | "scale">;

// Work units of the progress report. Measured on nightwood's models at 100 vox/m (sparse, 27k-230k
// stored bricks in grids of 1.5M-5.1M brick positions) and on dense test models: encoding a stored
// brick and listing its sub-cells took 2.4-2.7 µs, the near-brick pass 12-25 ns per brick position,
// and a pass over a dense grid (encoding its bricks, the part grid or the sub-cells) about 1 µs
// per brick position.
/** Units per stored sparse brick: encoding it and listing its occupied sub-cells. */
const SPARSE_UNITS = 100;
/** Units per brick position of a dense grid, per pass over it (roles, parts, sub-cells). */
const DENSE_UNITS = 50;
/** Units per brick position of the near-brick pass. */
const NEAR_UNITS = 1;
/** Least time between progress calls, ms. */
const PROGRESS_MS = 50;

/**
 * Encode a model for the GPU: its bricks, the part grid and part boxes of a model with parts, and
 * the occupied sub-cells the placement bake reads. Headless and pure: run it on the main thread
 * or in a worker; the result given to `Renderer.addEncodedModel` uploads exactly what
 * `Renderer.addModel(src)` would (which is `addEncodedModel(encodeModel(src))`).
 *
 * Throws on a model with parts but no dense `data`, and on one with more than 32 parts.
 *
 * @param src - Size plus dense `data` or `sparse` bricks, and optionally parts.
 * @param opts - Progress reporting ({@link EncodeOptions}).
 * @returns The encoded model; its arrays each own their buffer (except a given `partBoxes`).
 *
 * @example
 * ```ts
 * // encode.worker.ts
 * import { encodeModel, encodedTransferables } from "@voxolith/renderer/core";
 * self.onmessage = ({ data }) => {
 *   const encoded = encodeModel(data.model, { onProgress: (done, total) => self.postMessage({ id: data.id, done, total }) });
 *   self.postMessage({ id: data.id, encoded }, encodedTransferables(encoded));
 * };
 * // Main thread: only slots are claimed and bytes copied.
 * worker.onmessage = ({ data }) => { if (data.encoded) ids.set(data.id, renderer.addEncodedModel(data.encoded)); };
 * ```
 */
export function encodeModel(src: ModelSource, opts?: EncodeOptions): EncodedModel {
  if (src.parts && !src.data) throw new Error("encodeModel: parts need a dense model (data)");
  const size = { x: src.size.x, y: src.size.y, z: src.size.z };
  const [dx, dy, dz] = [Math.ceil(size.x / BRICK_B), Math.ceil(size.y / BRICK_B), Math.ceil(size.z / BRICK_B)];
  const positions = dx * dy * dz;

  const report = opts?.onProgress;
  let done = 0, total = 0, last = 0;
  if (report) {
    const passes = (src.sparse ? 0 : src.data ? 2 : 0) + (src.parts && src.data ? 1 : 0);
    total = (src.sparse ? src.sparse.bricks.size * SPARSE_UNITS : 0) + positions * (DENSE_UNITS * passes + NEAR_UNITS);
    report(0, total);
    last = performance.now();
  }
  const tick = (units: number) => {
    done += units;
    const now = performance.now();
    if (now - last >= PROGRESS_MS) {
      last = now;
      report!(done, total);
    }
  };

  // A pool of the model's own: fresh, so its slots and blocks are numbered in claim order.
  const pool = new BrickPool();
  const grid = new BrickGrid(size, undefined, pool);
  const subs = new Triples();
  // What changed is not needed (everything is new), so the edit lists are emptied as they fill.
  const scratch = emptyEdit();
  const clearEdit = () => { scratch.slots4.length = 0; scratch.slots8.length = 0; scratch.blocks.length = 0; scratch.tops.length = 0; };
  if (src.sparse) {
    const [sx, sy] = sparseDims(size);
    const here = new Uint8Array(64);
    for (const [key, cells] of src.sparse.bricks) {
      const bx = key % sx, by = Math.floor(key / sx) % sy, bz = Math.floor(key / (sx * sy));
      const e = pool.encode(0, cells, scratch);
      if (e) grid.setEntry(bx, by, bz, e, scratch);
      clearEdit();
      // A brick is exactly 4³ sub-cells, so each brick's own are distinct.
      const X = bx * 4, Y = by * 4, Z = bz * 4;
      here.fill(0);
      for (let i = 0; i < cells.length; i++) if (cells[i]) here[((i & 7) >> 1) + (((i >> 3) & 7) >> 1) * 4 + ((i >> 6) >> 1) * 16] = 1;
      for (let j = 0; j < 64; j++) if (here[j]) subs.push(X + (j & 3), Y + ((j >> 2) & 3), Z + (j >> 4));
      if (report) tick(SPARSE_UNITS);
    }
  } else if (src.data) {
    rebuildSlabs(grid, src.data, size, report && ((n) => tick(n * DENSE_UNITS)));
  }
  grid.markNear(scratch, report && ((n) => tick(n * NEAR_UNITS)));
  clearEdit();

  // Parts: a second grid of part index + 1 in the same pool, beside the roles (both are 8-bit).
  let partGrid: BrickGrid | undefined, boxes: Int32Array | undefined;
  if (src.parts && src.data) {
    const ids = new Uint8Array(src.data.length);
    let count = 0;
    for (let i = 0; i < ids.length; i++) if (src.data[i]) { ids[i] = src.parts[i] + 1; count = Math.max(count, src.parts[i] + 1); }
    const parts = Math.max(count, src.joints?.length ?? 0);
    // Each brick of a posed instance keeps a 32-bit mask of the parts in it (see instance.ts).
    if (parts > MAX_PARTS) throw new Error(`encodeModel: at most ${MAX_PARTS} parts can be posed on the GPU (this model has ${parts})`);
    partGrid = new BrickGrid(size, undefined, pool);
    rebuildSlabs(partGrid, ids, size, report && ((n) => tick(n * DENSE_UNITS)));
    // Given boxes (a superset, e.g. the undamaged model's) let damaged copies share poses.
    boxes = src.partBoxes ?? partBoxes(size, src.data, src.parts, parts);
  }
  if (src.data && !src.sparse) denseSubs(src.data, size, subs, report && ((n) => tick(n * DENSE_UNITS)));

  const encoded: EncodedModel = {
    version: ENCODED_MODEL_VERSION,
    size,
    top: grid.top.slice(),
    blocks: pool.blocks.slice(0, pool.blockCount * BLOCK_ENTRIES),
    voxels4: pool.voxels4.slice(0, pool.slots4 * BRICK_WORDS_4),
    palettes: pool.palettes.slice(0, pool.slots4 * PALETTE_WORDS),
    voxels8: pool.voxels8.slice(0, pool.slots8 * BRICK_WORDS_8),
    subs: subs.done(),
  };
  if (partGrid) encoded.partTop = partGrid.top.slice();
  if (boxes) encoded.partBoxes = boxes;
  if (src.joints) encoded.joints = src.joints;
  if (report) report(total, total);
  return encoded;
}

/**
 * The buffers of an encoded model, for `postMessage(encoded, encodedTransferables(encoded))`: the
 * arrays move to the receiver instead of being copied, and the sender's are empty afterwards.
 * `partBoxes` (small, and possibly the source's own object) is always copied.
 *
 * @param e - The encoded model.
 * @param opts - `keepPlacement: true` leaves `subs` out, so it is copied and the sender keeps it:
 *   what a worker that also bakes placement wants ({@link placementModelOf}).
 * @returns The buffers to transfer.
 */
export function encodedTransferables(e: EncodedModel, opts: { keepPlacement?: boolean } = {}): ArrayBuffer[] {
  const out = [e.top, e.blocks, e.voxels4, e.palettes, e.voxels8].map((a) => a.buffer);
  if (e.partTop) out.push(e.partTop.buffer);
  if (!opts.keepPlacement) out.push(e.subs.buffer);
  return out;
}

/**
 * The placement bake's view of an encoded model, once a renderer has added it: the model's own
 * `size`, `subs`, `partBoxes` and `joints` (shared, not copied) under the keys and scale the
 * renderer gave it (`Renderer.placementModel(id)` has them). What a worker that both encodes and bakes registers with
 * its `PlacementBaker`, so the model's sub-cells never travel back to it.
 *
 * @param e - The encoded model (the worker's copy: transfer it with `keepPlacement`).
 * @param keys - `key`, `poseKey` and `scale` from `Renderer.placementModel(id)` after `addEncodedModel`.
 * @returns A `PlacementModel` equal to the renderer's own.
 *
 * @example
 * ```ts
 * // Worker: encode, keep the sub-cells, and register once the main thread names the keys.
 * const encoded = encodeModel(src);
 * self.postMessage({ id, encoded }, encodedTransferables(encoded, { keepPlacement: true }));
 * pending.set(id, encoded);
 * // ...on { type: "keys", id, keys } from the main thread (after renderer.addEncodedModel):
 * baker.register(placementModelOf(pending.get(id)!, keys));
 * ```
 */
export function placementModelOf(e: EncodedModel, keys: PlacementKeys): PlacementModel {
  const m: PlacementModel = { key: keys.key, size: { ...e.size }, subs: e.subs };
  if (keys.scale !== undefined && keys.scale !== 1) m.scale = keys.scale;
  if (e.partBoxes) {
    m.partBoxes = e.partBoxes;
    m.poseKey = keys.poseKey;
  }
  if (e.joints) m.joints = e.joints;
  return m;
}

/** A fresh grid's full rebuild from dense data, one z slab of bricks at a time (the same order as `rebuildAll`). */
function rebuildSlabs(grid: BrickGrid, data: Uint8Array, size: { x: number; y: number; z: number }, tick: ((bricks: number) => void) | undefined): void {
  const [dx, dy, dz] = grid.dim;
  for (let bz = 0; bz < dz; bz++) {
    grid.rebuildBox(data, { x0: 0, y0: 0, z0: bz * BRICK_B, x1: size.x - 1, y1: size.y - 1, z1: bz * BRICK_B + BRICK_B - 1 });
    tick?.(dx * dy);
  }
}

/** The occupied 2³ sub-cells of a dense model, brick by brick (neighbours in the list are neighbours in space: the bake caches by brick). */
function denseSubs(d: Uint8Array, size: { x: number; y: number; z: number }, out: Triples, tick: ((bricks: number) => void) | undefined): void {
  const { x: sx, y: sy, z: sz } = size;
  const at = (x: number, y: number, z: number) => x < sx && y < sy && z < sz && d[x + y * sx + z * sx * sy] !== 0;
  const slab = Math.ceil(sx / 8) * Math.ceil(sy / 8);
  for (let z0 = 0; z0 < sz; z0 += 8) {
    for (let y0 = 0; y0 < sy; y0 += 8)
      for (let x0 = 0; x0 < sx; x0 += 8)
        for (let z = z0; z < z0 + 8 && z < sz; z += 2)
          for (let y = y0; y < y0 + 8 && y < sy; y += 2)
            for (let x = x0; x < x0 + 8 && x < sx; x += 2)
              if (at(x, y, z) || at(x + 1, y, z) || at(x, y + 1, z) || at(x + 1, y + 1, z) || at(x, y, z + 1) || at(x + 1, y, z + 1) || at(x, y + 1, z + 1) || at(x + 1, y + 1, z + 1))
                out.push(x >> 1, y >> 1, z >> 1);
    tick?.(slab);
  }
}

/** A growing list of int triples (a model's sub-cells run to millions of words at 100 vox/m). */
class Triples {
  private a = new Int32Array(3 * 1024);
  private n = 0;

  push(x: number, y: number, z: number): void {
    if (this.n + 3 > this.a.length) {
      const b = new Int32Array(this.a.length * 2);
      b.set(this.a);
      this.a = b;
    }
    this.a[this.n] = x;
    this.a[this.n + 1] = y;
    this.a[this.n + 2] = z;
    this.n += 3;
  }

  /** The triples, in an array that owns its buffer. */
  done(): Int32Array<ArrayBuffer> {
    return this.a.slice(0, this.n);
  }
}
