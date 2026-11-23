// The static placement bake: everything `Renderer.setInstances` computes on the CPU for a static
// set (instance records, packed poses, per-cell lists and the sub-cell tables of sublists.ts), as
// one headless function on plain data. The renderer runs it on the main thread for the synchronous
// `setInstances`; a host that wants no stall runs it in a worker (it imports nothing that needs a
// DOM, a GPU or a bundler) and hands the result to `Renderer.applyPlacement`. Both paths run this
// same code, so they upload the same bytes.
//
// Transport: the input names models by id and carries each one's registration key; the model data
// the bake reads (size, occupied sub-cells, part boxes, joints) is a `PlacementModel`, which a host
// sends to its worker once per model (`Renderer.placementModel`) rather than with every bake. The
// result is typed arrays that each own their buffer, so they transfer (`placementTransferables`).
//
// Several workers: the marking of the sub-cell tables, nearly all of a bake, splits exactly over
// contiguous ranges of the instances (see SubListMarks in sublists.ts for why). `splitPlacement`
// cuts an input into ranges of equal marking work, `bakePlacementShare` marks one range, and
// `mergePlacementShares` joins the shares into the bake `bakePlacement` makes, byte for byte.
// `bakePlacement` is the same three steps with one range, without the copies between them.

import { TOP_B } from "./brick";
import { INST_MODEL, INST_WORDS, maxPoseWords, packInstance, packPose, placement, scaledSize, type PackInstance } from "./instance";
import { fillSubLists, markSubLists, subListTables, subPieces, type SubListInstance } from "./sublists";

/** Instances per top cell list; more are dropped (with a console warning when applied). */
export const MAX_CELL_INSTANCES = 255;

/**
 * The version of {@link PlacementBake}'s layout and meaning: the instance record and pose
 * packing, the cell lists and the sub-cell tables, and what the bake computes from its input.
 * A host that stores bakes (the engine's scene cache) keys them by it, so a renderer that bakes
 * differently never applies a stored bake of an older one.
 *
 * Bump it whenever a bake of the same input and models would come out different: a changed
 * word, array or field, a new one, or the same words computed another way. The renderer's
 * verify holds a digest of a fixed scene's bake next to the version it was recorded under and
 * fails when the bake changes and the version does not, so both are updated together.
 */
export const PLACEMENT_BAKE_VERSION = 1;

/**
 * What the placement bake reads about one model: plain, structured-clone-friendly data. Get it
 * with `Renderer.placementModel(id)` after `addModel`, and send it to a worker once (register it
 * with a {@link PlacementBaker}); every later bake names the model by its `key`.
 */
export interface PlacementModel {
  /**
   * The model's registration key: unique on the page and never reused, so a worker's copy can
   * never be mistaken for a model that later took the same id. {@link PlacementInput.models} names
   * the key each model id stands for.
   */
  key: number;
  /** Extent in voxels. */
  size: { x: number; y: number; z: number };
  /**
   * The integer factor the model is drawn enlarged by (`ModelOptions.scale`); absent means 1, and
   * the renderer leaves it out then, so a scale-1 model's data is what it was before scales.
   */
  scale?: number;
  /** The model's occupied 2³ sub-cells, as x, y, z triples (the biggest array; send it once). */
  subs: Int32Array;
  /** Per-part voxel boxes, for models added with parts (min x, y, z, max x, y, z each). */
  partBoxes?: Int32Array;
  /**
   * Models with the same `poseKey` share packed poses, as copies of a model passing the same
   * `ModelSource.partBoxes` object do. Set when `partBoxes` is.
   */
  poseKey?: number;
  /** Per part, its parent (-1 for none) and the joint where it meets it, in model voxels. */
  joints?: readonly { parent: number; at: readonly [number, number, number] }[];
}

/** The world grid a bake is for: a renderer's, fixed for its lifetime. */
export interface PlacementGrid {
  /** The world's extent in 8³ bricks. */
  brickDim: [number, number, number];
  /** The world's extent in 64³ top cells. */
  topDim: [number, number, number];
  /** Voxels on the world's longest side (sets the sampling slack of the sub-cell tables). */
  gridMax: number;
}

/**
 * One static instance, as the bake takes it: the renderer's `Instance` fields only, so it survives
 * `postMessage`. `model` is a renderer model id. Instances sharing one `parts` object share a
 * packed pose, and structured cloning keeps that sharing within one message.
 */
export interface PlacementInstance extends PackInstance {
  /** Model id from `Renderer.addModel`. */
  model: number;
}

/** The input of {@link bakePlacement}, from `Renderer.placementInput`. */
export interface PlacementInput {
  /** The renderer's world grid. */
  grid: PlacementGrid;
  /**
   * The registration key of each model id the instances name (`models[id]`; 0 where no instance
   * names it). The bake looks each up in its registrations, and `applyPlacement` checks them
   * against the renderer's models, so a bake made before a model was removed or replaced throws
   * instead of drawing the wrong model.
   */
  models: number[];
  /** The static set, in order, with instances of removed models already left out. */
  instances: PlacementInstance[];
}

/**
 * A baked static set: the words `Renderer.applyPlacement` uploads, as typed arrays that each own
 * their buffer (transfer them with {@link placementTransferables}).
 */
export interface PlacementBake {
  /** The grid it was baked for. */
  grid: PlacementGrid;
  /** The model keys it was baked against (the input's `models`). */
  models: number[];
  /** Instances placed. */
  count: number;
  /** The instance records, `INST_WORDS` each. */
  inst: Uint32Array<ArrayBuffer>;
  /** Each instance's world box (min x, y, z, max x, y, z). */
  boxes: Float64Array<ArrayBuffer>;
  /** The packed poses the posed instances name. */
  parts: Uint32Array<ArrayBuffer>;
  /** One word per world top cell: (list offset << 8) | count, 0 for none. */
  cells: Uint32Array<ArrayBuffer>;
  /** The per-cell instance lists. */
  list: Uint32Array<ArrayBuffer>;
  /** The sub-cell region: a pointer word per top cell, then the tables and their lists. */
  subs: Uint32Array<ArrayBuffer>;
  /** A copy of the first `cells` words of `subs` (the pointers), restored when moving instances leave a cell. */
  subCells: Uint32Array<ArrayBuffer>;
  /** Cell entries dropped over the {@link MAX_CELL_INSTANCES}-per-cell limit. */
  dropped: number;
  /** Sub-cell tables built and (sub-cell, instance) pairs listed. */
  stats: { tables: number; pairs: number };
}

/** Options of {@link bakePlacement} and {@link PlacementBaker.bake}. */
export interface PlacementBakeOptions {
  /**
   * Progress of the bake, for a loading screen. `done` and `total` count opaque work units, scaled
   * so that `done / total` grows roughly linearly with the bake's time; only that fraction means
   * anything. `total` is fixed for one bake. The first call is `(0, total)`, made once the total is
   * known, and the last is `(total, total)`, just before the bake returns; in between, calls come at
   * most every 50 ms or so, and `done` never decreases. Without it the bake does no progress work.
   */
  onProgress?: (done: number, total: number) => void;
}

// Work units of the progress report: one unit is one occupied 2³ sub-cell of one instance marked
// into the sub-cell tables, about 25 ns, which is nearly all of a big bake. Measured on a
// nightwood-like scene through the mock device (2056 trees, shrubs and plants from the real
// generators): marking took 99.1-99.4% of the bake at 20, 50 and 100 vox/m (560 ms, 4.5 s,
// 18.3 s; 25-26 ns per sub-cell at each scale); packing the instances and filling the cell lists
// 0.2-0.6% (1.5-45 µs per instance, growing with the cells each one covers, so the per-instance
// weight below is a middle value), and writing the tables out 0.2-0.5% of the marking.
/** Progress units per instance for packing it and listing it in its cells. */
const PACK_UNITS = 256;
/** Progress units of writing the tables out, as a share of the marking units. */
const FILL_SHARE = 1 / 200;
/** Least time between progress calls, ms. */
const PROGRESS_MS = 50;

/** Looks up the registered model of a model id of `input`, checked; throws when there is none. */
function resolver(input: PlacementInput, model: (key: number) => PlacementModel | undefined): (id: number) => PlacementModel {
  const byId: PlacementModel[] = [];
  return (id) => {
    const known = byId[id];
    if (known) return known;
    const key = input.models[id];
    if (!key) throw new Error(`bakePlacement: an instance names model ${id}, which has no key in the input`);
    const m = model(key);
    if (!m) throw new Error(`bakePlacement: model key ${key} (id ${id}) is not registered; send Renderer.placementModel(${id}) first`);
    if (m.key !== key) throw new Error(`bakePlacement: the registration for key ${key} is model key ${m.key}`);
    return (byId[id] = m);
  };
}

/**
 * The marking units of instances `lo` to `hi` (exclusive), as markSubLists marks them: an
 * instance's sub-cells (times its pieces, when scaled and turned), or its box's bricks when posed.
 */
function markingUnits(instances: readonly PlacementInstance[], resolve: (id: number) => PlacementModel, lo: number, hi: number): { units: Float64Array; marking: number } {
  const units = new Float64Array(hi - lo);
  let marking = 0;
  const fwd = new Float64Array(12);
  for (let k = lo; k < hi; k++) {
    const p = instances[k], m = resolve(p.model);
    const { x, y, z } = m.size;
    // A scaled model's sub-cells are marked as pieces when turned (subPieces).
    const pieces = m.scale && m.scale > 1 ? subPieces(placement(p, scaledSize(m.size, m.scale), fwd).fwd, m.scale) ** 3 : 1;
    marking += units[k - lo] = p.parts && m.partBoxes ? (Math.ceil(x / 8) + 2) * (Math.ceil(y / 8) + 2) * (Math.ceil(z / 8) + 2) : (m.subs.length / 3) * pieces;
  }
  return { units, marking };
}

/** A throttled progress report: `tick(units)` adds work done, `end()` reports the total. */
function progress(report: (done: number, total: number) => void, total: number): { tick: (units: number) => void; end: () => void } {
  let done = 0;
  report(0, total);
  let last = performance.now();
  return {
    tick: (units) => {
      done += units;
      const now = performance.now();
      if (now - last >= PROGRESS_MS) {
        last = now;
        report(done, total);
      }
    },
    end: () => report(total, total),
  };
}

/**
 * The part of a bake before the marking: the instance records, boxes and poses, each top cell's
 * instances (in instance order) and the instances as markSubLists takes them.
 */
function packPlacement(input: PlacementInput, resolve: (id: number) => PlacementModel, tick?: (units: number) => void) {
  const { grid, instances } = input;
  const [tx, ty, tz] = grid.topDim;
  const count = instances.length;
  const inst = new Uint32Array(count * INST_WORDS);
  const boxes = new Float64Array(count * 6);
  // Poses: one packed pose per (parts object, part boxes), as the renderer's pose store keys them.
  let parts = new Uint32Array(1024);
  let partLen = 0;
  const poses = new Map<object, Map<number, { off: number; box: number[] }>>();
  const perCell = new Map<number, number[]>();
  const placed: SubListInstance[] = [];
  for (let k = 0; k < count; k++) {
    const p = instances[k];
    const m = resolve(p.model);
    const pm = { size: m.size, scale: m.scale, partBoxes: m.partBoxes, joints: m.joints };
    let pose: { off: number; box: number[] } | undefined;
    if (p.parts && m.partBoxes) {
      if ((m.scale ?? 1) !== 1) throw new Error(`bakePlacement: instance ${k} poses model ${p.model}, which is drawn at scale ${m.scale}; posing (Instance.parts) needs scale 1`);
      const key = p.parts as unknown as object;
      let byBoxes = poses.get(key);
      if (!byBoxes) poses.set(key, (byBoxes = new Map()));
      pose = byBoxes.get(m.poseKey!);
      if (!pose) {
        const most = maxPoseWords(pm);
        if (parts.length < partLen + most) parts = grow(parts, partLen + most);
        const { words, box } = packPose(p.parts, pm, parts, partLen);
        pose = { off: partLen, box };
        partLen += words;
        byBoxes.set(m.poseKey!, pose);
      }
    }
    const { box } = packInstance(p, pm, p.model, inst, k * INST_WORDS, pose);
    boxes.set(box, k * 6);
    placed.push({ inst: p, size: m.size, scale: m.scale, subs: m.subs, posedBox: pose ? boxes.subarray(k * 6, k * 6 + 6) : undefined });
    // Every world top cell the box touches (with a voxel of slack each side).
    const cx0 = Math.max(0, Math.floor((box[0] - 1) / TOP_B)), cx1 = Math.min(tx - 1, Math.floor((box[3] + 1) / TOP_B));
    const cy0 = Math.max(0, Math.floor((box[1] - 1) / TOP_B)), cy1 = Math.min(ty - 1, Math.floor((box[4] + 1) / TOP_B));
    const cz0 = Math.max(0, Math.floor((box[2] - 1) / TOP_B)), cz1 = Math.min(tz - 1, Math.floor((box[5] + 1) / TOP_B));
    for (let cz = cz0; cz <= cz1; cz++)
      for (let cy = cy0; cy <= cy1; cy++)
        for (let cx = cx0; cx <= cx1; cx++) {
          const ci = cx + cy * tx + cz * tx * ty;
          let l = perCell.get(ci);
          if (!l) perCell.set(ci, (l = []));
          l.push(k);
        }
    if (tick) tick(PACK_UNITS);
  }
  return { inst, boxes, parts: parts.slice(0, partLen), perCell, placed };
}

/** The per-cell words and lists of a bake, from each top cell's instances. */
function cellLists(perCell: Map<number, number[]>, cellCount: number) {
  const cells = new Uint32Array(cellCount);
  let entries = 0;
  for (const l of perCell.values()) entries += Math.min(MAX_CELL_INSTANCES, l.length);
  const list = new Uint32Array(entries);
  let at = 0, dropped = 0;
  for (const [ci, l] of perCell) {
    const n = Math.min(MAX_CELL_INSTANCES, l.length);
    dropped += l.length - n;
    cells[ci] = (at << 8) | n;
    for (let i = 0; i < n; i++) list[at++] = l[i];
  }
  return { cells, list, dropped };
}

/**
 * Bake a static instance set: the instance records, the poses of posed instances, the per-cell
 * lists (each 64³ top cell lists at most {@link MAX_CELL_INSTANCES}; the rest are dropped and
 * counted) and the sub-cell tables. Headless and pure: run it on the main thread or in a worker;
 * the result applied with `Renderer.applyPlacement` is exactly what `setInstances` would upload.
 *
 * To spread the work over several workers, see {@link splitPlacement}.
 *
 * @param input - From `Renderer.placementInput(list)` (structured-cloned is fine).
 * @param model - The registered model for a key, or undefined when it is not registered.
 * @param opts - Progress reporting ({@link PlacementBakeOptions}).
 * @returns The bake; its arrays each own their buffer.
 */
export function bakePlacement(input: PlacementInput, model: (key: number) => PlacementModel | undefined, opts?: PlacementBakeOptions): PlacementBake {
  const { grid, instances } = input;
  const cellCount = grid.topDim[0] * grid.topDim[1] * grid.topDim[2];
  const resolve = resolver(input, model);
  const count = instances.length;
  // Progress: each instance's marking units, summed up front so the total is fixed.
  let units: Float64Array | undefined, prog: ReturnType<typeof progress> | undefined;
  if (opts?.onProgress) {
    const u = markingUnits(instances, resolve, 0, count);
    units = u.units;
    prog = progress(opts.onProgress, count * PACK_UNITS + u.marking + Math.ceil(u.marking * FILL_SHARE));
  }
  const { inst, boxes, parts, perCell, placed } = packPlacement(input, resolve, prog?.tick);
  const { cells, list, dropped } = cellLists(perCell, cellCount);
  const { table, tables } = subListTables(cellCount, perCell, MAX_CELL_INSTANCES);
  const marks = markSubLists(grid.brickDim, grid.topDim, table, tables, placed, 0, count, grid.gridMax, units && ((k) => prog!.tick(units[k])));
  const built = fillSubLists(cellCount, table, tables, [marks]);
  const bake: PlacementBake = {
    grid: copyGrid(grid),
    models: [...input.models],
    count,
    inst,
    boxes,
    parts,
    cells,
    list,
    subs: built.data as Uint32Array<ArrayBuffer>,
    subCells: built.cellWords as Uint32Array<ArrayBuffer>,
    dropped,
    stats: built.stats,
  };
  prog?.end();
  return bake;
}

/**
 * A contiguous range of a static set's instances, for one {@link bakePlacementShare}: one of the
 * ranges {@link splitPlacement} cuts an input into.
 */
export interface PlacementRange {
  /** The first instance of the range (an index into `PlacementInput.instances`). */
  lo: number;
  /** One past its last instance; `lo === hi` is an empty range. */
  hi: number;
  /**
   * Whether this share also returns everything of the bake that is not marking (the instance
   * records, boxes, poses and cell lists: {@link PlacementShare.pack}). Exactly one share of a
   * split does; {@link splitPlacement} gives it to the first range.
   */
  pack: boolean;
}

/**
 * One share of a split bake, from {@link bakePlacementShare}: what marking the instances of its
 * range found, for {@link mergePlacementShares}. Plain data in typed arrays that each own their
 * buffer (transfer them with {@link placementShareTransferables}). A share is 4 bytes per
 * (sub-cell, instance) pair of its range, so the pairs of all shares together are the lists of
 * the bake's `subs`, plus 320 bytes per sub-cell table of the whole set in every share; the share
 * with the `pack` also carries the bake's other arrays.
 */
export interface PlacementShare {
  /** The first instance of its range. */
  lo: number;
  /** One past the last. */
  hi: number;
  /** Instances in the whole set (the input's), which every share of one bake agrees on. */
  count: number;
  /** Sub-cell tables in the whole set, which every share of one bake agrees on. */
  tables: number;
  /** The slot (table × 64 + sub-cell) of each (sub-cell, instance) pair, instance by instance. */
  slots: Uint32Array<ArrayBuffer>;
  /** Per instance of the range, the number of pairs up to and including its own. */
  ends: Uint32Array<ArrayBuffer>;
  /** Per slot, the instances of the range that reach it. */
  counts: Uint32Array<ArrayBuffer>;
  /** Per slot, the mask of its 8 bricks the range's instances reach. */
  masks: Uint8Array<ArrayBuffer>;
  /** The rest of the bake, from the share whose range has `pack` set. */
  pack?: {
    /** {@link PlacementBake.inst}. */
    inst: Uint32Array<ArrayBuffer>;
    /** {@link PlacementBake.boxes}. */
    boxes: Float64Array<ArrayBuffer>;
    /** {@link PlacementBake.parts}. */
    parts: Uint32Array<ArrayBuffer>;
    /** {@link PlacementBake.cells}. */
    cells: Uint32Array<ArrayBuffer>;
    /** {@link PlacementBake.list}. */
    list: Uint32Array<ArrayBuffer>;
    /** {@link PlacementBake.dropped}. */
    dropped: number;
    /** The top cell of each sub-cell table, in table order. */
    tableCells: Uint32Array<ArrayBuffer>;
  };
}

/**
 * Cut a static set into `n` contiguous ranges of instances for {@link bakePlacementShare}, so
 * that several workers bake it together:
 *
 * 1. `splitPlacement(input, model, n)`, anywhere the models are registered. It is deterministic
 *    and cheap (no marking), so each worker can also call it itself and take its own range.
 * 2. Each worker, with the same models registered, runs `bakePlacementShare(input, model, range)`
 *    on one range and posts the share back ({@link placementShareTransferables}).
 * 3. `mergePlacementShares(input, shares)` joins them, on any thread and without the models, into
 *    the {@link PlacementBake} that {@link bakePlacement} makes of the same input, byte for byte.
 *
 * The ranges are balanced by marking work (an instance's occupied sub-cells, or its box's bricks
 * when posed), which is nearly all of a bake, not by instance count: one tree can cost a
 * thousand tufts of grass. They are in order, cover every instance, and the first has `pack`
 * set. A range can be empty (more ranges than instances, or one instance outweighing a whole
 * share); an empty range without `pack` need not be baked.
 *
 * @param input - From `Renderer.placementInput(list)`.
 * @param model - The registered model for a key, as for {@link bakePlacement}.
 * @param n - How many ranges (at least 1; fractions round down).
 * @returns `n` ranges.
 */
export function splitPlacement(input: PlacementInput, model: (key: number) => PlacementModel | undefined, n: number): PlacementRange[] {
  n = Math.max(1, Math.floor(n) || 1);
  const count = input.instances.length;
  const { units, marking } = markingUnits(input.instances, resolver(input, model), 0, count);
  const ranges: PlacementRange[] = [];
  let lo = 0, before = 0;
  for (let i = 1; i <= n; i++) {
    let hi = lo;
    if (i === n) hi = count;
    else if (marking > 0) {
      // Up to the instance that takes the running total closest to i shares of the whole.
      const goal = (marking * i) / n;
      while (hi < count && Math.abs(before + units[hi] - goal) <= Math.abs(before - goal)) before += units[hi++];
    } else hi = Math.max(lo, Math.floor((count * i) / n));
    ranges.push({ lo, hi, pack: i === 1 });
    lo = hi;
  }
  return ranges;
}

/**
 * Bake one share of a static set: mark the instances of `range` into the sub-cell tables, which
 * is nearly all of a bake's time, and with `range.pack` also make the rest of the bake. See
 * {@link splitPlacement} for the whole procedure. Headless and pure, as {@link bakePlacement}.
 *
 * Every share packs all the instances first, whatever its range: the tables' numbering comes
 * from every instance's box, and each share has to arrive at the same one. That is well under
 * 1% of a single bake, and the reason a share needs every model of the input registered, not
 * only those of its range.
 *
 * Progress (`opts.onProgress`) is per share, in the same units as {@link bakePlacement}'s: add
 * the shares' `done` and `total` to get the bake's.
 *
 * @param input - The whole input, the same for every share.
 * @param model - The registered model for a key, as for {@link bakePlacement}.
 * @param range - One of {@link splitPlacement}'s ranges of this input.
 * @param opts - Progress reporting ({@link PlacementBakeOptions}).
 * @returns The share; its arrays each own their buffer.
 */
export function bakePlacementShare(input: PlacementInput, model: (key: number) => PlacementModel | undefined, range: PlacementRange, opts?: PlacementBakeOptions): PlacementShare {
  const { grid, instances } = input;
  const cellCount = grid.topDim[0] * grid.topDim[1] * grid.topDim[2];
  const count = instances.length;
  const { lo, hi } = range;
  if (!Number.isInteger(lo) || !Number.isInteger(hi) || lo < 0 || hi < lo || hi > count) throw new Error(`bakePlacementShare: the range ${lo}..${hi} is not within the input's ${count} instances`);
  const resolve = resolver(input, model);
  let units: Float64Array | undefined, prog: ReturnType<typeof progress> | undefined;
  if (opts?.onProgress) {
    const u = markingUnits(instances, resolve, lo, hi);
    units = u.units;
    prog = progress(opts.onProgress, count * PACK_UNITS + u.marking);
  }
  const { inst, boxes, parts, perCell, placed } = packPlacement(input, resolve, prog?.tick);
  const { table, tables } = subListTables(cellCount, perCell, MAX_CELL_INSTANCES);
  const marks = markSubLists(grid.brickDim, grid.topDim, table, tables, placed, lo, hi, grid.gridMax, units && ((k) => prog!.tick(units[k - lo])));
  const share: PlacementShare = { lo, hi, count, tables, slots: marks.slots.slice(0, marks.pairs), ends: marks.ends, counts: marks.counts, masks: marks.masks };
  if (range.pack) {
    const tableCells = new Uint32Array(tables);
    for (let ci = 0; ci < cellCount; ci++) if (table[ci] >= 0) tableCells[table[ci]] = ci;
    share.pack = { inst, boxes, parts, ...cellLists(perCell, cellCount), tableCells };
  }
  prog?.end();
  return share;
}

/**
 * The buffers of a share, for `postMessage(share, placementShareTransferables(share))`. The
 * sender's arrays are empty afterwards.
 */
export function placementShareTransferables(share: PlacementShare): ArrayBuffer[] {
  const p = share.pack;
  return [share.slots, share.ends, share.counts, share.masks, ...(p ? [p.inst, p.boxes, p.parts, p.cells, p.list, p.tableCells] : [])].map((a) => a.buffer);
}

/**
 * Join the shares of a split bake ({@link splitPlacement}, {@link bakePlacementShare}) into the
 * bake: byte for byte what {@link bakePlacement} makes of `input`, in every array, `dropped` and
 * `stats`. It needs no models and does none of the marking or packing, only the last step of a
 * bake (laying the tables out and sorting the shares' pairs into them, a fraction of a percent
 * of the marking's time), so it is fine on the main thread.
 *
 * The shares may come in any order. Together their ranges must cover every instance of the
 * input once (empty ones may be left out), and exactly one must carry the `pack`; it throws
 * otherwise, or when the shares are not of one bake of this input as far as it can tell (the
 * instance and table counts).
 *
 * @param input - The input the shares were baked from.
 * @param shares - Every share. The bake takes over the `pack`'s arrays (they are not copied).
 * @returns The bake; its arrays each own their buffer.
 */
export function mergePlacementShares(input: PlacementInput, shares: readonly PlacementShare[]): PlacementBake {
  const { grid } = input;
  const cellCount = grid.topDim[0] * grid.topDim[1] * grid.topDim[2];
  const count = input.instances.length;
  const packed = shares.filter((s) => s.pack);
  if (packed.length !== 1) throw new Error(`mergePlacementShares: ${packed.length} shares carry the pack; exactly one must (the range with pack set)`);
  const pack = packed[0].pack!;
  const tables = pack.tableCells.length;
  const order = [...shares].sort((a, b) => a.lo - b.lo || a.hi - b.hi);
  let at = 0;
  for (const s of order) {
    if (s.count !== count) throw new Error(`mergePlacementShares: a share is of a set of ${s.count} instances, the input has ${count}`);
    if (s.tables !== tables || s.counts.length !== tables * 64 || s.masks.length !== tables * 64) throw new Error(`mergePlacementShares: a share has ${s.tables} sub-cell tables, the pack ${tables}; not shares of one bake`);
    if (s.lo !== at) throw new Error(`mergePlacementShares: the shares ${s.lo > at ? "leave out" : "overlap at"} instances ${Math.min(s.lo, at)}..${Math.max(s.lo, at)}`);
    if (s.ends.length !== s.hi - s.lo) throw new Error(`mergePlacementShares: the share ${s.lo}..${s.hi} is damaged`);
    at = s.hi;
  }
  if (at !== count) throw new Error(`mergePlacementShares: the shares leave out instances ${at}..${count}`);
  if (pack.inst.length !== count * INST_WORDS || pack.cells.length !== cellCount) throw new Error("mergePlacementShares: the pack is of another input");
  const table = new Int32Array(cellCount).fill(-1);
  for (let t = 0; t < tables; t++) table[pack.tableCells[t]] = t;
  const built = fillSubLists(cellCount, table, tables, order.map((s) => ({ lo: s.lo, hi: s.hi, pairs: s.slots.length, slots: s.slots, ends: s.ends, counts: s.counts, masks: s.masks })));
  return {
    grid: copyGrid(grid),
    models: [...input.models],
    count,
    inst: pack.inst,
    boxes: pack.boxes,
    parts: pack.parts,
    cells: pack.cells,
    list: pack.list,
    subs: built.data as Uint32Array<ArrayBuffer>,
    subCells: built.cellWords as Uint32Array<ArrayBuffer>,
    dropped: pack.dropped,
    stats: built.stats,
  };
}

/**
 * The buffers of a bake, for `postMessage(bake, placementTransferables(bake))`: the arrays move to
 * the receiver instead of being copied. The sender's arrays are empty afterwards.
 */
export function placementTransferables(bake: PlacementBake): ArrayBuffer[] {
  return [bake.inst, bake.boxes, bake.parts, bake.cells, bake.list, bake.subs, bake.subCells].map((a) => a.buffer);
}

/**
 * A bake with the model ids taken out, for storing it beyond this page: every instance record
 * names its model by order of first appearance in the static set (the first model named is 0,
 * the next new one 1, and so on) instead of by renderer id, and `models` is empty. Nothing else
 * in a bake depends on the ids (the lists and tables name instances by index, poses are shared
 * by `parts` object and `poseKey` class, not by value), so this is the same for any ids the same
 * set was baked under. Turn it back into a bake for this page's ids with {@link bindPlacement}.
 * Normalising a normalised bake gives the same bake.
 *
 * @param bake - A bake from {@link bakePlacement} (or a normalised one).
 * @returns A new bake with its own `inst` and `grid`; the other arrays are the given bake's own
 *   (not copied), so transferring either transfers both.
 */
export function normalizePlacement(bake: PlacementBake): PlacementBake {
  const inst = bake.inst.slice();
  const canon = new Map<number, number>();
  for (let k = 0; k < bake.count; k++) {
    const o = k * INST_WORDS + INST_MODEL, id = inst[o];
    let c = canon.get(id);
    if (c === undefined) canon.set(id, (c = canon.size));
    inst[o] = c;
  }
  return { ...bake, grid: copyGrid(bake.grid), models: [], inst, stats: { ...bake.stats } };
}

/**
 * Bind a normalised bake ({@link normalizePlacement}) to the models of `input`: each record's
 * model word becomes the renderer id the input's instance names, and `models` becomes the
 * input's keys. The result equals, byte for byte, what {@link bakePlacement} makes of `input`,
 * provided the input is the set the bake was made from with only its model ids and keys
 * changed, and its models have the same placement data (size, sub-cells, part boxes, joints,
 * and which of them share poses). A host storing bakes makes sure of that with its key (the
 * engine's scene cache digests exactly those); this checks what it can without the models: the
 * grid, the instance count and that the input names its models in the same pattern, and throws
 * otherwise.
 *
 * @param bake - A normalised bake.
 * @param input - The input to bind it to (from `Renderer.placementInput` on this page).
 * @returns A new bake with its own `inst`, `grid` and `models`; the other arrays are the given
 *   bake's own (not copied), ready for `Renderer.applyPlacement`.
 */
export function bindPlacement(bake: PlacementBake, input: PlacementInput): PlacementBake {
  const g = bake.grid, h = input.grid;
  if (g.gridMax !== h.gridMax || [0, 1, 2].some((a) => g.brickDim[a] !== h.brickDim[a] || g.topDim[a] !== h.topDim[a])) {
    throw new Error("bindPlacement: the bake is for another grid");
  }
  const { instances } = input;
  if (instances.length !== bake.count) throw new Error(`bindPlacement: the bake has ${bake.count} instances, the input ${instances.length}`);
  const inst = bake.inst.slice();
  const canon = new Map<number, number>();
  for (let k = 0; k < bake.count; k++) {
    const id = instances[k].model;
    let c = canon.get(id);
    if (c === undefined) {
      if (!input.models[id]) throw new Error(`bindPlacement: an instance names model ${id}, which has no key in the input`);
      canon.set(id, (c = canon.size));
    }
    const o = k * INST_WORDS + INST_MODEL;
    if (inst[o] !== c) throw new Error(`bindPlacement: instance ${k} names model ${inst[o]} of the bake but the input's model ${c}; not a normalised bake of this set`);
    inst[o] = id;
  }
  return { ...bake, grid: copyGrid(g), models: [...input.models], inst, stats: { ...bake.stats } };
}

const copyGrid = (g: PlacementGrid): PlacementGrid => ({ brickDim: [...g.brickDim], topDim: [...g.topDim], gridMax: g.gridMax });

/**
 * The worker side of an off-main-thread placement: holds the models registered with it and bakes
 * static sets against them. Headless (import it from `@voxolith/renderer/core`).
 *
 * @example
 * ```ts
 * // placement.worker.ts
 * import { PlacementBaker, placementTransferables } from "@voxolith/renderer/core";
 * const baker = new PlacementBaker();
 * self.onmessage = ({ data }) => {
 *   if (data.type === "model") baker.register(data.model);
 *   else if (data.type === "drop") baker.unregister(data.key);
 *   else if (data.type === "bake") {
 *     const bake = baker.bake(data.input, { onProgress: (done, total) => self.postMessage({ id: data.id, done, total }) });
 *     self.postMessage({ id: data.id, bake }, placementTransferables(bake));
 *   }
 * };
 * ```
 */
export class PlacementBaker {
  private readonly models = new Map<number, PlacementModel>();

  /** Register a model (from `Renderer.placementModel`); a later registration of the same key replaces it. */
  register(model: PlacementModel): void {
    this.models.set(model.key, model);
  }

  /** Forget a model, after `Renderer.removeModel`. */
  unregister(key: number): void {
    this.models.delete(key);
  }

  /** Whether a model key is registered. */
  has(key: number): boolean {
    return this.models.has(key);
  }

  /**
   * {@link bakePlacement} against the registered models; throws when an instance names one not registered.
   *
   * @param input - From `Renderer.placementInput(list)`.
   * @param opts - Progress reporting ({@link PlacementBakeOptions}).
   */
  bake(input: PlacementInput, opts?: PlacementBakeOptions): PlacementBake {
    return bakePlacement(input, (key) => this.models.get(key), opts);
  }

  /**
   * {@link splitPlacement} against the registered models: `n` ranges of the input's instances,
   * balanced by marking work, for {@link PlacementBaker.bakeShare} on this and other bakers.
   *
   * @param input - From `Renderer.placementInput(list)`.
   * @param n - How many ranges.
   */
  split(input: PlacementInput, n: number): PlacementRange[] {
    return splitPlacement(input, (key) => this.models.get(key), n);
  }

  /**
   * {@link bakePlacementShare} against the registered models: one share of a bake split over
   * several bakers, each of which needs every model of the input registered. Join the shares
   * with {@link mergePlacementShares}.
   *
   * @param input - The whole input, the same for every share.
   * @param range - One of the ranges of {@link PlacementBaker.split} (or {@link splitPlacement}).
   * @param opts - Progress reporting ({@link PlacementBakeOptions}), for this share.
   */
  bakeShare(input: PlacementInput, range: PlacementRange, opts?: PlacementBakeOptions): PlacementShare {
    return bakePlacementShare(input, (key) => this.models.get(key), range, opts);
  }
}

function grow(a: Uint32Array, need: number): Uint32Array<ArrayBuffer> {
  let len = Math.max(a.length * 2, 64);
  while (len < need) len *= 2;
  const next = new Uint32Array(len);
  next.set(a);
  return next;
}
