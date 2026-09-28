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

import { TOP_B } from "./brick";
import { INST_WORDS, maxPoseWords, packInstance, packPose, type PackInstance } from "./instance";
import { buildSubLists, type SubListInstance } from "./sublists";

/** Instances per top cell list; more are dropped (with a console warning when applied). */
export const MAX_CELL_INSTANCES = 255;

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

/**
 * Bake a static instance set: the instance records, the poses of posed instances, the per-cell
 * lists (each 64³ top cell lists at most {@link MAX_CELL_INSTANCES}; the rest are dropped and
 * counted) and the sub-cell tables. Headless and pure: run it on the main thread or in a worker;
 * the result applied with `Renderer.applyPlacement` is exactly what `setInstances` would upload.
 *
 * @param input - From `Renderer.placementInput(list)` (structured-cloned is fine).
 * @param model - The registered model for a key, or undefined when it is not registered.
 * @param opts - Progress reporting ({@link PlacementBakeOptions}).
 * @returns The bake; its arrays each own their buffer.
 */
export function bakePlacement(input: PlacementInput, model: (key: number) => PlacementModel | undefined, opts?: PlacementBakeOptions): PlacementBake {
  const { grid, instances } = input;
  const [tx, ty, tz] = grid.topDim;
  const cellCount = tx * ty * tz;
  const byId: PlacementModel[] = [];
  const resolve = (id: number): PlacementModel => {
    const known = byId[id];
    if (known) return known;
    const key = input.models[id];
    if (!key) throw new Error(`bakePlacement: an instance names model ${id}, which has no key in the input`);
    const m = model(key);
    if (!m) throw new Error(`bakePlacement: model key ${key} (id ${id}) is not registered; send Renderer.placementModel(${id}) first`);
    if (m.key !== key) throw new Error(`bakePlacement: the registration for key ${key} is model key ${m.key}`);
    return (byId[id] = m);
  };

  const count = instances.length;
  // Progress: each instance's marking units (its sub-cells, or its box's bricks when posed, as
  // buildSubLists marks it), summed up front so the total is fixed.
  const report = opts?.onProgress;
  let done = 0, total = 0, last = 0;
  let units: Float64Array | undefined;
  if (report) {
    units = new Float64Array(count);
    let marking = 0;
    for (let k = 0; k < count; k++) {
      const p = instances[k], m = resolve(p.model);
      const { x, y, z } = m.size;
      marking += units[k] = p.parts && m.partBoxes ? (Math.ceil(x / 8) + 2) * (Math.ceil(y / 8) + 2) * (Math.ceil(z / 8) + 2) : m.subs.length / 3;
    }
    total = count * PACK_UNITS + marking + Math.ceil(marking * FILL_SHARE);
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
    const pm = { size: m.size, partBoxes: m.partBoxes, joints: m.joints };
    let pose: { off: number; box: number[] } | undefined;
    if (p.parts && m.partBoxes) {
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
    placed.push({ inst: p, size: m.size, subs: m.subs, posedBox: pose ? boxes.subarray(k * 6, k * 6 + 6) : undefined });
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
    if (report) tick(PACK_UNITS);
  }

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
  const built = buildSubLists(grid.brickDim, grid.topDim, perCell, placed, MAX_CELL_INSTANCES, grid.gridMax, units && ((k) => tick(units[k])));
  const bake: PlacementBake = {
    grid: { brickDim: [...grid.brickDim], topDim: [...grid.topDim], gridMax: grid.gridMax },
    models: [...input.models],
    count,
    inst,
    boxes,
    parts: parts.slice(0, partLen),
    cells,
    list,
    subs: built.data as Uint32Array<ArrayBuffer>,
    subCells: built.cellWords as Uint32Array<ArrayBuffer>,
    dropped,
    stats: built.stats,
  };
  if (report) report(total, total);
  return bake;
}

/**
 * The buffers of a bake, for `postMessage(bake, placementTransferables(bake))`: the arrays move to
 * the receiver instead of being copied. The sender's arrays are empty afterwards.
 */
export function placementTransferables(bake: PlacementBake): ArrayBuffer[] {
  return [bake.inst, bake.boxes, bake.parts, bake.cells, bake.list, bake.subs, bake.subCells].map((a) => a.buffer);
}

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
}

function grow(a: Uint32Array, need: number): Uint32Array<ArrayBuffer> {
  let len = Math.max(a.length * 2, 64);
  while (len < need) len *= 2;
  const next = new Uint32Array(len);
  next.set(a);
  return next;
}
