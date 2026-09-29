// Headless checks for the renderer's CPU-side data structures. No GPU.
//
//   bun run --cwd renderer verify

import { BrickGrid, BrickPool, BRICK_B, NEAR_BIT, PALETTE_ENTRIES, SLOT_MASK, UNIFORM_BIT, emptyEdit } from "../src/brick";
import { makeSparse, sparseCount, sparseFromDense, sparseGet, sparseSet, sparseToDense } from "../src/sparse";
import { OccupancyGrid } from "../src/occupancy";
import { seededRandom } from "../src/random";
import { makePerf } from "../src/perf";
import { buildSubLists, readSubLists, type SubListInstance } from "../src/sublists";
import { INST_SCALE_SHIFT, INST_WORDS, packInstance, sampleInstance } from "../src/instance";
import { PLACEMENT_BAKE_VERSION, PlacementBaker, bakePlacement, bindPlacement, normalizePlacement, placementTransferables, type PlacementBake, type PlacementInput, type PlacementInstance, type PlacementModel } from "../src/placement";
import { ENCODED_MODEL_VERSION, encodeModel, encodedTransferables, placementModelOf, type EncodedModel, type ModelSource } from "../src/encode";
import { installRawLoader, mockGpu } from "./mock-gpu";
import type { Instance } from "../src/renderer";

const warn0 = console.warn;

let failed = 0;
let checks = 0;

function ok(cond: boolean, what: string, detail = ""): void {
  checks++;
  if (cond) {
    console.log(`  ✓ ${what}`);
  } else {
    failed++;
    console.log(`  ✗ ${what}${detail ? ` — ${detail}` : ""}`);
  }
}

type Size = { x: number; y: number; z: number };
const at = (s: Size, x: number, y: number, z: number) => x + y * s.x + z * s.x * s.y;

/** Every voxel must read back identically through the sparse form. */
function roundTrip(name: string, size: Size, data: Uint8Array): BrickGrid {
  const bricks = new BrickGrid(size, data);
  let bad = 0;
  let firstBad = "";
  for (let z = 0; z < size.z; z++)
    for (let y = 0; y < size.y; y++)
      for (let x = 0; x < size.x; x++) {
        const want = data[at(size, x, y, z)];
        const got = bricks.get(x, y, z);
        if (got !== want) {
          if (!bad) firstBad = `at ${x},${y},${z} want ${want} got ${got}`;
          bad++;
        }
      }
  ok(bad === 0, `${name}: round-trips exactly`, `${bad} voxels differ, ${firstBad}`);
  return bricks;
}

console.log("brick storage:");

// A brick whose palette exactly fills, and one that overflows it by one. The
// 4-bit tier has 15 usable entries because nibble 0 means empty, so 15 must fit
// and 16 must fall back to the 8-bit tier rather than corrupt or drop values.
{
  const size: Size = { x: BRICK_B, y: BRICK_B, z: BRICK_B };
  for (const distinct of [1, PALETTE_ENTRIES, PALETTE_ENTRIES + 1, 255]) {
    const data = new Uint8Array(size.x * size.y * size.z);
    for (let i = 0; i < data.length; i++) data[i] = (i % distinct) + 1;
    const b = roundTrip(`${distinct} distinct values in one brick`, size, data);
    const wide = b.stats().wide;
    ok(
      distinct <= PALETTE_ENTRIES ? wide === 0 : wide === 1,
      `  ${distinct} distinct uses the ${distinct <= PALETTE_ENTRIES ? "4-bit" : "8-bit"} tier`,
      `wide=${wide}`,
    );
  }
}

// Grids whose dimensions are not multiples of the brick edge: the edge bricks
// are partial and must not read beyond the grid or pad with rubbish.
{
  const size: Size = { x: 13, y: 7, z: 19 };
  const data = new Uint8Array(size.x * size.y * size.z);
  const rng = seededRandom(7);
  for (let i = 0; i < data.length; i++) data[i] = rng() < 0.4 ? 1 + ((rng() * 9) | 0) : 0;
  roundTrip("ragged grid (13x7x19)", size, data);
}

// Empty and full extremes.
{
  const size: Size = { x: 16, y: 16, z: 16 };
  const empty = new Uint8Array(size.x * size.y * size.z);
  const b = roundTrip("all-empty grid", size, empty);
  ok(b.stats().used === 0, "  all-empty allocates no bricks", `used=${b.stats().used}`);

  const full = new Uint8Array(size.x * size.y * size.z).fill(42);
  const f = roundTrip("uniform solid grid", size, full);
  ok(f.stats().used === 8, "  uniform solid allocates every brick", `used=${f.stats().used}`);
}

// Incremental edits must match a full rebuild, including a brick that changes
// tier (palette overflow) and one that empties out and frees its slot.
{
  const size: Size = { x: 32, y: 16, z: 32 };
  const data = new Uint8Array(size.x * size.y * size.z);
  const rng = seededRandom(99);
  for (let i = 0; i < data.length; i++) data[i] = rng() < 0.25 ? 1 + ((rng() * 5) | 0) : 0;
  const bricks = new BrickGrid(size, data);

  // Push one brick over the palette limit.
  for (let i = 0; i < 40; i++) {
    const x = 8 + (i % 8);
    const y = 0 + ((i / 8) | 0);
    data[at(size, x, y, 8)] = 20 + i;
  }
  bricks.rebuildBox(data, { x0: 8, y0: 0, z0: 8, x1: 15, y1: 7, z1: 8 });

  // Clear a different brick entirely.
  for (let z = 16; z < 24; z++)
    for (let y = 0; y < 8; y++) for (let x = 16; x < 24; x++) data[at(size, x, y, z)] = 0;
  bricks.rebuildBox(data, { x0: 16, y0: 0, z0: 16, x1: 23, y1: 7, z1: 23 });

  let bad = 0;
  for (let z = 0; z < size.z; z++)
    for (let y = 0; y < size.y; y++)
      for (let x = 0; x < size.x; x++) if (bricks.get(x, y, z) !== data[at(size, x, y, z)]) bad++;
  ok(bad === 0, "incremental edits match the dense grid", `${bad} voxels differ`);

  const fresh = new BrickGrid(size, data);
  ok(
    fresh.stats().used === bricks.stats().used,
    "  edited grid has the same brick count as a fresh build",
    `${bricks.stats().used} vs ${fresh.stats().used}`,
  );
}

// Slot reuse: clearing and refilling the same brick must not leak slots.
{
  const size: Size = { x: BRICK_B, y: BRICK_B, z: BRICK_B };
  const data = new Uint8Array(size.x * size.y * size.z).fill(3);
  const bricks = new BrickGrid(size, data);
  const box = { x0: 0, y0: 0, z0: 0, x1: 7, y1: 7, z1: 7 };
  for (let i = 0; i < 20; i++) {
    data.fill(0);
    bricks.rebuildBox(data, box);
    data.fill(3 + (i % 4));
    data[0] = 9; // not uniform, so the brick needs a payload slot
    bricks.rebuildBox(data, box);
  }
  ok(bricks.slotCount4 === 1, "clearing and refilling reuses one slot", `slots=${bricks.slotCount4}`);
  ok(bricks.get(1, 2, 3) === 3 + 19 % 4, "  value after reuse is correct");
}

console.log("two-level index, uniform bricks:");
{
  // A tall, wide, empty world costs only its top level.
  const size: Size = { x: 12800, y: 2048, z: 12800 };
  const g = new BrickGrid(size);
  ok(g.top.length === 200 * 32 * 200 && g.stats().blocks === 0, `a 12800x2048x12800 world starts as a ${(g.top.length * 4 / 1048576).toFixed(1)} MB top level`);
  // Solid ground far apart: blocks appear only where things are.
  g.editBox({ x0: 0, y0: 0, z0: 0, x1: 63, y1: 15, z1: 63 }, (c) => { c.fill(5); return true; });
  g.editBox({ x0: 12000, y0: 100, z0: 9000, x1: 12003, y1: 101, z1: 9001 }, (c, ox, oy, oz) => {
    for (let i = 0; i < 512; i++) { const x = ox + (i & 7), y = oy + ((i >> 3) & 7), z = oz + (i >> 6); if (x >= 12000 && x <= 12003 && y >= 100 && y <= 101 && z >= 9000 && z <= 9001) c[i] = 7; }
    return true;
  });
  const st = g.stats();
  ok(st.blocks === 2 && st.uniform === 128 && st.used === 129, `two places, two blocks; the solid slab is ${st.uniform} uniform bricks with no payload (${st.used} used)`);
  ok(g.pool.slots4 === 1 && g.get(10, 3, 10) === 5 && g.get(12001, 100, 9000) === 7 && g.get(12005, 100, 9000) === 0, "  values read back through both levels");
  g.clearBox({ x0: 11968, y0: 64, z0: 8960, x1: 12031, y1: 127, z1: 9023 });
  ok(g.stats().blocks === 1 && g.get(12001, 100, 9000) === 0, "  clearing a region frees its block");
  // Uniform brick edited back to mixed gets a payload again.
  g.editBox({ x0: 0, y0: 0, z0: 0, x1: 0, y1: 0, z1: 0 }, (c) => { c[0] = 0; return true; });
  ok(g.get(0, 0, 0) === 0 && g.get(1, 0, 0) === 5 && g.stats().uniform === 127, "  a uniform brick edited becomes a payload brick");
}

console.log("model grids and sparse volumes:");
{
  const size: Size = { x: 40, y: 20, z: 30 };
  const s = makeSparse(size);
  for (let x = 5; x < 35; x++) sparseSet(s, x, 3, 12, 2);
  sparseSet(s, 39, 19, 29, 4);
  ok(sparseCount(s) === 31 && sparseGet(s, 20, 3, 12) === 2 && sparseGet(s, 39, 19, 29) === 4, "sparse set/get/count");
  const dense = sparseToDense(s);
  const back = sparseFromDense(size, dense);
  ok(back.bricks.size === s.bricks.size && sparseToDense(back).every((v, i) => v === dense[i]), "  dense round trip");
  const pool = new BrickPool();
  const world = new BrickGrid({ x: 64, y: 64, z: 64 }, undefined, pool);
  const model = new BrickGrid(size, dense, pool);
  const edit = { slots4: [], slots8: [], blocks: [], tops: [] };
  model.markNear(edit);
  const near = (bx: number, by: number, bz: number) => (model.entry(bx, by, bz) & NEAR_BIT) !== 0;
  ok(model.get(20, 3, 12) === 2 && near(0, 1, 1) && near(3, 0, 2) && !near(0, 2, 3), "  a model grid marks empty bricks next to content as near, and only those");
  ok(world.stats().blocks === 0 && pool.blockCount >= 1, "  grids share one pool of blocks and bricks");
}

// The sparse form must agree with OccupancyGrid about what is empty, since the
// brick index replaces the coarse occupancy test in the shader.
{
  const size: Size = { x: 64, y: 32, z: 64 };
  const data = new Uint8Array(size.x * size.y * size.z);
  const rng = seededRandom(1234);
  for (let z = 0; z < size.z; z++)
    for (let x = 0; x < size.x; x++) {
      const h = 4 + ((rng() * 8) | 0);
      for (let y = 0; y < h; y++) data[at(size, x, y, z)] = 1 + ((rng() * 3) | 0);
    }
  const bricks = new BrickGrid(size, data);
  const occ = new OccupancyGrid(size, data);
  let disagree = 0;
  for (let z = 0; z < size.z; z++)
    for (let y = 0; y < size.y; y++)
      for (let x = 0; x < size.x; x++) {
        const solid = data[at(size, x, y, z)] !== 0;
        if (solid && bricks.get(x, y, z) === 0) disagree++;
      }
  ok(disagree === 0, "brick index never hides a solid voxel", `${disagree} hidden`);
  ok(occ.data.length > 0, "  occupancy grid still builds alongside");

  const st = bricks.stats();
  console.log(
    `  terrain sample: ${st.used} bricks, ${st.wide} wide, ` +
      `${(st.payloadBytes / 1024).toFixed(0)} KB payload vs ${(st.denseBytes / 1024).toFixed(0)} KB dense`,
  );
}

console.log("adaptive render scale:");
{
  // A GPU whose frame cost grows with the pixel count, behind a 60 Hz vsync:
  // rAF only reports whole refresh intervals, so a 17 ms frame shows as 33 ms.
  // A controller that climbs whenever it sees 16.7 ms and drops when it sees
  // 33 ms hunts between two scales forever, and every change resamples the
  // image — a visible shimmer on a still scene.
  const simulate = (costAtFull: number, seconds: number, retryAfterMs?: number) => {
    const perf = makePerf({ enabled: false, scale: 0.5, minScale: 0.3, maxScale: 1, retryAfterMs });
    const vsync = 1000 / 60;
    let now = 0, changes = 0, lateChanges = 0, prev = perf.scale();
    while (now < seconds * 1000) {
      const cost = costAtFull * perf.scale() * perf.scale();
      now += Math.ceil(cost / vsync - 1e-9) * vsync;
      perf.frame(now);
      if (perf.scale() !== prev) {
        changes++;
        if (now > 20000 && now < 50000) lateChanges++;
        prev = perf.scale();
      }
    }
    return { scale: perf.scale(), changes, lateChanges };
  };
  const r = simulate(28, 60);
  ok(r.lateChanges <= 2, `settles instead of hunting: ${r.lateChanges} scale changes between 20 s and 50 s`, `ended at ${r.scale}`);
  ok(r.scale >= 0.6 && 28 * r.scale * r.scale <= 1000 / 60, `and settles near the best scale that fits the frame (${r.scale})`);
  const still = simulate(28, 60, Infinity);
  ok(still.lateChanges === 0, `with timed retries off, a still view never changes scale once settled (${still.lateChanges})`);
  const fast = simulate(8, 20);
  ok(fast.scale === 1, "a cheap scene still climbs to full resolution");
}


console.log("\nsub-cell instance tables:");

// For random models and placements (turned, tilted, mirrored, fractional), every world cell an
// instance draws (sampleInstance, the shader's CPU twin, at voxel centres) is listed by its
// sub-cell's table for its brick; and a build that misplaces instances (shifted, mirror flip dropped)
// is caught.
{
  const rng = seededRandom(777);
  const world = { x: 192, y: 96, z: 192 };
  const brickDim: [number, number, number] = [world.x / 8, world.y / 8, world.z / 8];
  const topDim: [number, number, number] = [Math.ceil(world.x / 64), Math.ceil(world.y / 64), Math.ceil(world.z / 64)];
  const models = Array.from({ length: 4 }, (_, mi) => {
    const size = { x: 5 + Math.floor(rng() * 20), y: 5 + Math.floor(rng() * 24), z: 5 + Math.floor(rng() * 20) };
    const data = new Uint8Array(size.x * size.y * size.z);
    // A few blobs and single voxels (sparse, like leaves).
    for (let i = 0; i < data.length; i++) if (rng() < (mi === 0 ? 0.02 : 0.08)) data[i] = 1 + Math.floor(rng() * 3);
    const subs = new Set<string>();
    let i = 0;
    for (let z = 0; z < size.z; z++) for (let y = 0; y < size.y; y++) for (let x = 0; x < size.x; x++, i++) if (data[i]) subs.add(`${x >> 1},${y >> 1},${z >> 1}`);
    const flat = (st: Set<string>) => Int32Array.from([...st].flatMap((k) => k.split(",").map(Number)));
    return { size, data, subs: flat(subs) };
  });
  const placed: SubListInstance[] = [];
  const words = new Uint32Array(INST_WORDS * 64);
  const boxes: number[][] = [];
  for (let k = 0; k < 60; k++) {
    const mi = k % models.length, m = models[mi];
    const kind = rng();
    const inst = {
      x: 30 + rng() * 130, y: 10 + rng() * 50, z: 30 + rng() * 130, base: 1,
      yaw: kind < 0.3 ? (Math.floor(rng() * 4) * Math.PI) / 2 : rng() * Math.PI * 2,
      mirror: rng() < 0.3,
      rotation: kind > 0.85 ? (() => { const a = rng() * 3, b = rng() * 3; const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b); return [cb, -sb * ca, sb * sa, sb, cb * ca, -cb * sa, 0, sa, ca]; })() : undefined,
    };
    const { box } = packInstance(inst, { size: m.size }, mi, words, k * INST_WORDS);
    boxes.push(box);
    placed.push({ inst, size: m.size, subs: m.subs });
  }
  const perCell = new Map<number, number[]>();
  placed.forEach((_, k) => {
    const b = boxes[k];
    for (let cz = Math.max(0, Math.floor((b[2] - 1) / 64)); cz <= Math.min(topDim[2] - 1, Math.floor((b[5] + 1) / 64)); cz++)
      for (let cy = Math.max(0, Math.floor((b[1] - 1) / 64)); cy <= Math.min(topDim[1] - 1, Math.floor((b[4] + 1) / 64)); cy++)
        for (let cx = Math.max(0, Math.floor((b[0] - 1) / 64)); cx <= Math.min(topDim[0] - 1, Math.floor((b[3] + 1) / 64)); cx++) {
          const ci = cx + cy * topDim[0] + cz * topDim[0] * topDim[1];
          let l = perCell.get(ci);
          if (!l) perCell.set(ci, (l = []));
          l.push(k);
        }
  });
  const sampleModel = (mi: number) => {
    const m = models[mi];
    return { size: m.size, voxel: (x: number, y: number, z: number) => m.data[x + y * m.size.x + z * m.size.x * m.size.y], part: () => 0 };
  };
  const check = (build: typeof buildSubLists) => {
    const built = build(brickDim, topDim, perCell, placed, 255, 192);
    let drawn = 0, unlisted = 0, first = "";
    placed.forEach((_, k) => {
      const b = boxes[k], sm = sampleModel(k % models.length);
      for (let z = Math.max(0, Math.floor(b[2]) - 1); z <= Math.min(world.z - 1, Math.ceil(b[5]) + 1); z++)
        for (let y = Math.max(0, Math.floor(b[1]) - 1); y <= Math.min(world.y - 1, Math.ceil(b[4]) + 1); y++)
          for (let x = Math.max(0, Math.floor(b[0]) - 1); x <= Math.min(world.x - 1, Math.ceil(b[3]) + 1); x++) {
            if (!sampleInstance(words, k * INST_WORDS, undefined, sm, x, y, z)) continue;
            drawn++;
            const l = readSubLists(built, topDim, x, y, z);
            if (!l || !l.includes(k)) { unlisted++; first ||= `instance ${k} at ${x},${y},${z}`; }
          }
    });
    return { drawn, unlisted, first, pairs: built.stats.pairs };
  };
  const r = check(buildSubLists);
  ok(r.drawn > 5000 && r.unlisted === 0, `every drawn cell is listed for its brick (${r.drawn} cells, ${r.pairs} sub-cell entries)`, `${r.unlisted} unlisted, first ${r.first}`);
  // The check itself: shift every instance a little, or drop the mirror flip, and it must fail.
  const shifted = check((d, t, pc, pl, m, g) => buildSubLists(d, t, pc, pl.map((p) => ({ ...p, inst: { ...p.inst, x: p.inst.x + 0.3 } })), m, g));
  const unflipped = check((d, t, pc, pl, m, g) => buildSubLists(d, t, pc, pl.map((p) => ({ ...p, inst: { ...p.inst, mirror: false } })), m, g));
  ok(shifted.unlisted > 0 && unflipped.unlisted > 0, `  and a build that misplaces them is caught (${shifted.unlisted} and ${unflipped.unlisted} cells unlisted)`);
}

console.log("\nplacement bake (the renderer against a mock device):");

// The static set baked off the main thread (placementInput → a worker's PlacementBaker, through
// structured cloning and transfer → applyPlacement) must upload exactly what setInstances does:
// every buffer byte for byte, on a scene with dropped over-limit cells, posed instances sharing
// poses, a removed model and a moving set on top.
installRawLoader();
const { Renderer } = await import("../src/renderer");
const placementScene = await import("./placement-scene");
{
  const make = () => {
    const mock = mockGpu();
    const r = new Renderer(mock.gpu, placementScene.world(), "", { pipeline: "compute" });
    const ids = placementScene.addModels(r);
    return { mock, r, ids };
  };
  const A = make(), B = make();
  const { staticSet, staticSet2, moving, moving2 } = placementScene.lists(A.ids);
  const baker = new PlacementBaker();
  const registered = new Set<number>();
  const viaWorker = (list: typeof staticSet) => {
    // What a host does: register models the worker lacks (once), post the input, apply the reply.
    for (const id of B.ids.all) {
      const m = B.r.placementModel(id);
      if (m && !registered.has(m.key)) { baker.register(structuredClone(m)); registered.add(m.key); }
    }
    const bake = baker.bake(structuredClone(B.r.placementInput(list)));
    return structuredClone(bake, { transfer: placementTransferables(bake) });
  };
  const same = (what: string) => {
    let diff = -1, where = "";
    if (A.mock.buffers.length !== B.mock.buffers.length) where = `${A.mock.buffers.length} vs ${B.mock.buffers.length} buffers`;
    else A.mock.buffers.forEach((a, i) => {
      const b = B.mock.buffers[i];
      if (a.size !== b.size) { where ||= `buffer ${i}: ${a.size} vs ${b.size} bytes`; return; }
      if (diff < 0) for (let j = 0; j < a.size; j++) if (a.bytes[j] !== b.bytes[j]) { diff = j; where = `buffer ${i} byte ${j}`; break; }
    });
    ok(!where, what, where);
  };
  A.r.setInstances(moving, { dynamic: true });
  B.r.setInstances(moving, { dynamic: true });
  const warn = console.warn;
  const warned: string[] = [];
  console.warn = (m: string) => warned.push(m);
  A.r.setInstances(staticSet);
  const bake1 = viaWorker(staticSet);
  B.r.applyPlacement(bake1);
  console.warn = warn;
  ok(bake1.dropped > 0 && warned.length === 2 && warned[0] === warned[1], `the scene drops over-limit cell entries (${bake1.dropped}), and both paths warn alike`);
  ok(bake1.parts.length > 0 && bake1.stats.tables > 0 && bake1.count === staticSet.length - placementScene.removedCount(staticSet, A.ids), `  it has posed instances (${bake1.parts.length} pose words), ${bake1.stats.tables} sub-cell tables and skips the removed model's instances`);
  same("  a static set applied from a worker bake uploads the same bytes as setInstances, under a moving set");
  A.r.setInstances(moving2, { dynamic: true });
  B.r.setInstances(moving2, { dynamic: true });
  same("  and a new moving set on top of it too");
  console.warn = () => {};
  A.r.setInstances(staticSet2);
  B.r.applyPlacement(viaWorker(staticSet2));
  same("  a second static set");
  A.r.setInstances(staticSet);
  B.r.applyPlacement(bake1);
  console.warn = warn;
  same("  re-applying the first bake after moving sets used it");
  A.r.setInstances([], { dynamic: true });
  B.r.setInstances([], { dynamic: true });
  same("  and emptying the moving set");
  // Progress: (0, total) first, (total, total) last, never decreasing, throttled by time; and the
  // same bytes as a bake without it. A clock that moves 10 ms a reading makes the throttle let
  // calls through on a scene this small (a real bake of it takes a few ms).
  {
    const input = structuredClone(B.r.placementInput(staticSet));
    const plain = baker.bake(input);
    const calls: [number, number][] = [];
    const now = performance.now;
    let clock = 0;
    performance.now = () => (clock += 10);
    let reported: typeof plain;
    try { reported = baker.bake(input, { onProgress: (done, total) => calls.push([done, total]) }); } finally { performance.now = now; }
    const total = calls[0]?.[1] ?? 0;
    const bytes = (b: typeof plain) => placementTransferables(b).map((a) => new Uint8Array(a));
    const same = bytes(plain).every((a, i) => { const b = bytes(reported)[i]; return a.length === b.length && a.every((v, j) => v === b[j]); });
    ok(
      calls.length > 2 && total > 0 && calls[0][0] === 0 && calls.at(-1)![0] === total && calls.every(([d, t], i) => t === total && (i === 0 || d >= calls[i - 1][0])),
      `  bake progress runs from 0 to its fixed total (${total} units, ${calls.length} calls) and never goes back`,
      JSON.stringify(calls.slice(0, 5)),
    );
    let quiet = 0;
    performance.now = () => 0;
    try { baker.bake(input, { onProgress: () => quiet++ }); } finally { performance.now = now; }
    ok(same && quiet === 2, "  and changes no byte of the bake; a bake quicker than the throttle reports only its first and last call", `same ${same}, ${quiet} calls`);
  }
  // A bake made before a model changed must not draw the wrong model.
  B.r.removeModel(B.ids.tree);
  let threw = "";
  try { B.r.applyPlacement(bake1); } catch (e) { threw = String(e); }
  ok(threw.includes("removed or replaced"), "  applying a bake after its model was removed throws", threw);
  let unregistered = "";
  try { new PlacementBaker().bake(structuredClone(A.r.placementInput(staticSet))); } catch (e) { unregistered = String(e); }
  ok(unregistered.includes("not registered"), "  baking against a model the worker lacks throws", unregistered);
}

console.log("\nplacement bakes across visits (normalizePlacement → bindPlacement):");

// A host that stores bakes (the engine's scene cache) keeps them normalised, since the next visit
// gives the same models other ids: the renderer numbers models by free slot. Bound to that
// visit's input, a stored bake must be exactly the bake that visit would make, on a scene with
// posed instances, poses shared across two models, and models removed and added again.
{
  const bakeOn = (r: InstanceType<typeof Renderer>, ids: number[], list: Parameters<typeof r.placementInput>[0]) => {
    const baker = new PlacementBaker();
    for (const id of ids) { const m = r.placementModel(id); if (m) baker.register(structuredClone(m)); }
    const input = structuredClone(r.placementInput(list));
    return { input, bake: baker.bake(input) };
  };
  const diffBakes = (a: PlacementBake, b: PlacementBake): string => {
    const head = (x: PlacementBake) => JSON.stringify({ grid: x.grid, models: x.models, count: x.count, dropped: x.dropped, stats: x.stats });
    if (head(a) !== head(b)) return `${head(a)} vs ${head(b)}`;
    for (const name of ["inst", "boxes", "parts", "cells", "list", "subs", "subCells"] as const) {
      const x = new Uint8Array(a[name].buffer, a[name].byteOffset, a[name].byteLength), y = new Uint8Array(b[name].buffer, b[name].byteOffset, b[name].byteLength);
      if (x.length !== y.length) return `${name}: ${x.length} vs ${y.length} bytes`;
      for (let j = 0; j < x.length; j++) if (x[j] !== y[j]) return `${name} byte ${j}`;
    }
    return "";
  };
  const warn = console.warn;
  console.warn = () => {};
  // Visit A: the scene's models in order. Visit B: two other models first, one of them removed
  // (a free slot below), the scene's models added, the tree and the rat pair removed and added
  // again (new keys; the rats' new part boxes object gives them a new shared poseKey), so every
  // id and key differs and slots are reused out of order.
  const rA = new Renderer(mockGpu().gpu, placementScene.world(), "", { pipeline: "compute" });
  const idsA = placementScene.addModels(rA);
  const rB = new Renderer(mockGpu().gpu, placementScene.world(), "", { pipeline: "compute" });
  const filler = { size: { x: 4, y: 4, z: 4 }, data: new Uint8Array(64).fill(1) };
  const f1 = rB.addModel(filler);
  rB.addModel(filler);
  const first = placementScene.addModels(rB);
  rB.removeModel(f1);
  for (const id of first.all) rB.removeModel(id);
  const again = placementScene.addModels(rB);
  const idsB = { ...again, gone: first.gone };
  const { staticSet: setA, staticSet2: set2A } = placementScene.lists(idsA);
  const { staticSet: setB, staticSet2: set2B } = placementScene.lists(idsB);
  const a = bakeOn(rA, idsA.all, setA), b = bakeOn(rB, idsB.all, setB);
  const moved = idsA.all.filter((id, i) => id !== idsB.all[i]).length;
  const keysA = idsA.all.map((id) => rA.placementModel(id)!.key), keysB = idsB.all.map((id) => rB.placementModel(id)!.key);
  ok(moved >= 3 && keysA.every((k) => !keysB.includes(k)) && a.bake.parts.length > 0, `  visit B names the models by other ids (${idsA.all} vs ${idsB.all}) and keys; the set has ${a.bake.parts.length} pose words`);
  const nA = normalizePlacement(a.bake), nB = normalizePlacement(b.bake);
  ok(!diffBakes(nA, nB), "  both visits' bakes normalise to the same bake", diffBakes(nA, nB));
  ok(!diffBakes(normalizePlacement(nA), nA), "  and normalising again changes nothing", diffBakes(normalizePlacement(nA), nA));
  const bound = bindPlacement(structuredClone(nA), b.input);
  ok(!diffBakes(bound, b.bake), "  visit A's bake bound to visit B's input equals B's own bake, byte for byte", diffBakes(bound, b.bake));
  const unbound = diffBakes({ ...nA, models: [...b.input.models] }, b.bake);
  ok(unbound.startsWith("inst"), "  (and without rebinding its instance records it does not)", unbound);
  const back = bindPlacement(nB, a.input);
  ok(!diffBakes(back, a.bake), "  and B's bound to A's input equals A's", diffBakes(back, a.bake));
  // A second set (other shared poses, a third of the instances) through the same path.
  const a2 = bakeOn(rA, idsA.all, set2A), b2 = bakeOn(rB, idsB.all, set2B);
  const bound2 = bindPlacement(normalizePlacement(a2.bake), b2.input);
  ok(!diffBakes(bound2, b2.bake), "  a second static set too", diffBakes(bound2, b2.bake));
  // Applying the bound bake draws what visit B's setInstances draws.
  const mB = mockGpu(), rC = new Renderer(mB.gpu, placementScene.world(), "", { pipeline: "compute" });
  const mD = mockGpu(), rD = new Renderer(mD.gpu, placementScene.world(), "", { pipeline: "compute" });
  for (const r of [rC, rD]) {
    const g1 = r.addModel(filler);
    r.addModel(filler);
    const f = placementScene.addModels(r);
    r.removeModel(g1);
    for (const id of f.all) r.removeModel(id);
    placementScene.addModels(r);
  }
  rC.setInstances(setB);
  rD.applyPlacement(bindPlacement(nA, structuredClone(rD.placementInput(setB))));
  const sameBytes = mB.buffers.length === mD.buffers.length && mB.buffers.every((x, i) => x.size === mD.buffers[i].size && x.bytes.every((v, j) => v === mD.buffers[i].bytes[j]));
  ok(sameBytes, "  applied, it uploads exactly what setInstances does");
  console.warn = warn;
  // Binding checks what it can without the models.
  const throws = (f: () => unknown) => { try { f(); return ""; } catch (e) { return String(e); } };
  const swapped = setB.map((i) => (i.model === idsB.tree ? { ...i, model: idsB.rock } : i.model === idsB.rock ? { ...i, model: idsB.tree } : i));
  const other = { ...b.input, grid: { ...b.input.grid, gridMax: b.input.grid.gridMax + 1 } };
  const errs = [
    throws(() => bindPlacement(nA, structuredClone(rB.placementInput(swapped.map((i) => (i.model === idsB.tree ? { ...i, model: idsB.rat } : i)))))),
    throws(() => bindPlacement(nA, { ...b.input, instances: b.input.instances.slice(1) })),
    throws(() => bindPlacement(nA, other)),
  ];
  ok(errs[0].includes("not a normalised bake") && errs[1].includes("instances") && errs[2].includes("another grid"), "  binding to an input with another pattern of models, count or grid throws", errs.join(" | "));
}

// PLACEMENT_BAKE_VERSION: a digest of a fixed bake, recorded with the version it was made under.
// The scene is hand-made placement data (no encoding, no trigonometry: rotations and poses are
// exact quarter turns and 3-4-5 rotations), so only the bake itself moves it. When this fails,
// the bake's bytes changed: bump PLACEMENT_BAKE_VERSION and record the new digest with it.
{
  const RECORDED = { version: 1, digest: "0c575c97ffd7fa08" };
  const rng = seededRandom(7);
  const subsOf = (size: Size, p: number) => {
    const out: number[] = [];
    for (let Z = 0; Z < size.z / 2; Z += 4)
      for (let Y = 0; Y < size.y / 2; Y += 4)
        for (let X = 0; X < size.x / 2; X += 4)
          for (let j = 0; j < 64; j++) {
            const x = X + (j & 3), y = Y + ((j >> 2) & 3), z = Z + (j >> 4);
            if (x * 2 < size.x && y * 2 < size.y && z * 2 < size.z && rng() < p) out.push(x, y, z);
          }
    return new Int32Array(out);
  };
  const boxesP = new Int32Array([0, 0, 1, 5, 5, 6, 6, 0, 1, 10, 5, 6, 11, 0, 1, 15, 5, 6]);
  const joints = [{ parent: -1, at: [3, 2, 4] as const }, { parent: 0, at: [6, 2, 4] as const }, { parent: 1, at: [11, 2, 4] as const }];
  const models: PlacementModel[] = [
    { key: 11, size: { x: 12, y: 20, z: 12 }, subs: subsOf({ x: 12, y: 20, z: 12 }, 0.3) },
    { key: 12, size: { x: 20, y: 10, z: 20 }, subs: subsOf({ x: 20, y: 10, z: 20 }, 0.5) },
    { key: 13, size: { x: 16, y: 8, z: 8 }, subs: subsOf({ x: 16, y: 8, z: 8 }, 0.8), partBoxes: boxesP, poseKey: 1, joints },
    { key: 14, size: { x: 16, y: 8, z: 8 }, subs: subsOf({ x: 16, y: 8, z: 8 }, 0.6), partBoxes: boxesP, poseKey: 1, joints },
  ];
  const quarter = [0, 0, 1, 0, 1, 0, -1, 0, 0], tilt = [0.8, -0.6, 0, 0.6, 0.8, 0, 0, 0, 1];
  // Part transforms: identity, a quarter turn of the tail about its joint, a 3-4-5 turn of the middle.
  const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
  const poseA = new Float32Array([...I, 0.8, -0.6, 0, 6 - 0.8 * 6 + 0.6 * 2, 0.6, 0.8, 0, 2 - 0.6 * 6 - 0.8 * 2, 0, 0, 1, 0, 0, -1, 0, 11 + 2, 1, 0, 0, 2 - 11, 0, 0, 1, 0]);
  const poseB = new Float32Array([...I, ...I, 0, 1, 0, 11 - 2, -1, 0, 0, 2 + 11, 0, 0, 1, 0]);
  const instances: PlacementInstance[] = [];
  for (let k = 0; k < 280; k++) instances.push({ model: 0, base: 300, x: 60 + (k % 7) * 5, y: 10 + (k % 5) * 6, z: 60 + ((k * 3) % 11) * 4, rotation: k % 3 ? quarter : undefined, mirror: k % 4 === 0 });
  for (let k = 0; k < 60; k++) instances.push({ model: k % 2 ? 1 : 0, base: 256 + k, x: 8 + k * 3.25, y: 5 + (k % 9) * 10, z: 240 - k * 3.5, rotation: k % 5 === 0 ? tilt : undefined, anchor: k % 7 === 0 ? [2, 1, 3] : undefined });
  for (let k = 0; k < 24; k++) instances.push({ model: k % 3 ? 2 : 3, base: 400, x: 20 + k * 9, y: 30 + (k % 4) * 8, z: 30 + k * 8, rotation: k % 2 ? quarter : undefined, mirror: k % 5 === 0, parts: k % 4 ? poseA : poseB });
  const input: PlacementInput = { grid: { brickDim: [32, 16, 32], topDim: [4, 2, 4], gridMax: 256 }, models: [11, 12, 13, 14], instances };
  const bake = normalizePlacement(bakePlacement(input, (key) => models.find((m) => m.key === key)));
  // FNV-1a over the header's JSON and every array's bytes, two lanes for 64 bits.
  let h1 = 0x811c9dc5, h2 = 0x01000193 ^ 0x5bd1e995;
  const eat = (bytes: Uint8Array) => { for (let i = 0; i < bytes.length; i++) { h1 = Math.imul(h1 ^ bytes[i], 0x01000193); h2 = Math.imul(h2 ^ bytes[i], 0x01000193) ^ (h2 >>> 15); } };
  eat(new TextEncoder().encode(JSON.stringify({ grid: bake.grid, models: bake.models, count: bake.count, dropped: bake.dropped, stats: bake.stats })));
  for (const a of [bake.inst, bake.boxes, bake.parts, bake.cells, bake.list, bake.subs, bake.subCells]) eat(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
  const digest = (h1 >>> 0).toString(16).padStart(8, "0") + (h2 >>> 0).toString(16).padStart(8, "0");
  ok(bake.dropped > 0 && bake.parts.length > 0 && bake.stats.tables > 0, `  the fixed scene has dropped cell entries, ${bake.parts.length} pose words and ${bake.stats.tables} sub-cell tables`);
  ok(
    PLACEMENT_BAKE_VERSION === RECORDED.version && digest === RECORDED.digest,
    `  its bake is the one recorded for PLACEMENT_BAKE_VERSION ${RECORDED.version} (digest ${digest})`,
    PLACEMENT_BAKE_VERSION !== RECORDED.version
      ? `the version is ${PLACEMENT_BAKE_VERSION}: record this digest, ${digest}, with it`
      : `the bake changed (digest ${digest}, recorded ${RECORDED.digest}): bump PLACEMENT_BAKE_VERSION and record the new digest with it`,
  );
}

console.log("\nmodel encoding (encodeModel → addEncodedModel, against a mock device):");

// markNear, the neighbourhood pass every model grid gets: the same entries, written in the same
// order (so the same blocks claimed), as the plain 27-neighbour test, on fresh grids, on grids
// marked already and on grids edited after marking (stale near bits).
{
  const reference = (g: BrickGrid, edit: ReturnType<typeof emptyEdit>) => {
    const [dx, dy, dz] = g.dim;
    const occ = new Uint8Array(dx * dy * dz);
    for (let z = 0; z < dz; z++) for (let y = 0; y < dy; y++) for (let x = 0; x < dx; x++) if (g.entry(x, y, z) & (SLOT_MASK | UNIFORM_BIT)) occ[x + y * dx + z * dx * dy] = 1;
    for (let z = 0; z < dz; z++)
      for (let y = 0; y < dy; y++)
        for (let x = 0; x < dx; x++) {
          if (occ[x + y * dx + z * dx * dy]) continue;
          let near = false;
          for (let k = -1; k <= 1; k++) for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
            const a = x + i, b = y + j, c = z + k;
            if (a >= 0 && b >= 0 && c >= 0 && a < dx && b < dy && c < dz && occ[a + b * dx + c * dx * dy]) near = true;
          }
          const want = near ? NEAR_BIT : 0;
          if (g.entry(x, y, z) !== want) g.setEntry(x, y, z, want, edit);
        }
  };
  let bad = "";
  const rng = seededRandom(77);
  for (let t = 0; t < 12 && !bad; t++) {
    const size = { x: 8 + Math.floor(rng() * 150), y: 8 + Math.floor(rng() * 90), z: 8 + Math.floor(rng() * 150) };
    const data = new Uint8Array(size.x * size.y * size.z);
    const blobs = 1 + Math.floor(rng() * 6);
    for (let n = 0; n < blobs; n++) {
      const cx = rng() * size.x, cy = rng() * size.y, cz = rng() * size.z, r = 2 + rng() * 12;
      for (let z = 0; z < size.z; z++) for (let y = 0; y < size.y; y++) for (let x = 0; x < size.x; x++) if ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < r * r) data[at(size, x, y, z)] = 1 + n;
    }
    const pa = new BrickPool(), pb = new BrickPool();
    const a = new BrickGrid(size, data, pa), b = new BrickGrid(size, data, pb);
    for (let round = 0; round < 3 && !bad; round++) {
      const ea = emptyEdit(), eb = emptyEdit();
      a.markNear(ea);
      reference(b, eb);
      if (JSON.stringify(ea) !== JSON.stringify(eb) || a.top.some((v, i) => v !== b.top[i]) || pa.blocks.some((v, i) => v !== pb.blocks[i])) bad = `grid ${t} round ${round}`;
      // Edit between rounds: empty some bricks and fill others, leaving stale near bits.
      const box = { x0: Math.floor(rng() * size.x), y0: 0, z0: Math.floor(rng() * size.z), x1: size.x - 1, y1: Math.floor(rng() * size.y), z1: size.z - 1 };
      const fill = round === 0 ? 0 : 3;
      for (const g of [a, b]) g.editBox(box, (c) => { c.fill(fill); return true; });
    }
  }
  ok(!bad, "markNear writes what the 27-neighbour test writes, in the same order, fresh, re-marked and after edits", bad);
}

// The worker path (encodeModel, structured-cloned with its buffers transferred, then
// addEncodedModel) must upload exactly what addModel does: every buffer byte for byte, for sparse,
// dense, 8-bit and posed models, models sharing part boxes, and a model added, removed and
// re-added, with a static set on top.
{
  const wide: ModelSource = (() => {
    const size = { x: 20, y: 17, z: 9 }, data = new Uint8Array(size.x * size.y * size.z);
    for (let i = 0; i < data.length; i++) data[i] = i % 5 === 0 ? 0 : 1 + (i % 40); // more than 15 values per brick
    return { size, data };
  })();
  // What a host does: encode on a worker, transfer, add. Copies of a model that passed one boxes
  // object arrive with one each, so the host restores the sharing (EncodedModel.partBoxes).
  const shared = new WeakMap<Int32Array, Int32Array>();
  const viaWorker = (r: InstanceType<typeof Renderer>) => (src: ModelSource): number => {
    const e = encodeModel(src);
    const moved = structuredClone(e, { transfer: encodedTransferables(e) }) as EncodedModel;
    if (src.partBoxes && moved.partBoxes) {
      const first = shared.get(src.partBoxes);
      if (first) moved.partBoxes = first;
      else shared.set(src.partBoxes, moved.partBoxes);
    }
    return r.addEncodedModel(moved);
  };
  const make = (worker: boolean) => {
    const mock = mockGpu();
    const r = new Renderer(mock.gpu, placementScene.world(), "", { pipeline: "compute" });
    const add = worker ? viaWorker(r) : (src: ModelSource) => r.addModel(src);
    const ids = placementScene.addModels({ addModel: add, removeModel: (id) => r.removeModel(id) });
    const w = add(wide);
    r.removeModel(ids.rock);
    const again = add(wide);
    r.removeModel(w);
    const rockAgain = placementScene.addModels({ addModel: add, removeModel: (id) => r.removeModel(id) });
    return { mock, r, ids, again, rockAgain };
  };
  const A = make(false), B = make(true);
  const sameBuffers = () => {
    if (A.mock.buffers.length !== B.mock.buffers.length) return `${A.mock.buffers.length} vs ${B.mock.buffers.length} buffers`;
    for (let i = 0; i < A.mock.buffers.length; i++) {
      const a = A.mock.buffers[i], b = B.mock.buffers[i];
      if (a.size !== b.size) return `buffer ${i}: ${a.size} vs ${b.size} bytes`;
      for (let j = 0; j < a.size; j++) if (a.bytes[j] !== b.bytes[j]) return `buffer ${i} byte ${j}`;
    }
    return "";
  };
  const ids = [...A.ids.all, A.again, ...A.rockAgain.all];
  const subsSame = ids.every((id) => {
    const a = A.r.placementModel(id), b = B.r.placementModel(id);
    return !!a && !!b && a.subs.length === b.subs.length && a.subs.every((v, i) => v === b.subs[i]) && !!a.poseKey === !!b.poseKey;
  });
  ok(!sameBuffers() && subsSame, "models added from a worker's transferred encoding upload the same bytes as addModel (sparse, dense, 8-bit, posed; removed and re-added)", sameBuffers());
  console.warn = () => {};
  const { staticSet } = placementScene.lists(A.rockAgain);
  A.r.setInstances(staticSet);
  B.r.setInstances(staticSet);
  console.warn = warn0;
  const shares = (r: typeof A.r) => r.placementModel(A.rockAgain.rat)!.poseKey === r.placementModel(A.rockAgain.ratCopy)!.poseKey;
  ok(!sameBuffers() && shares(A.r) && shares(B.r), "  and a static set over them too, with the copies' poses shared on both paths", sameBuffers());

  // Placement from the encoding: a worker that keeps its copy (keepPlacement) registers it under
  // the renderer's keys and bakes what a baker fed placementModel(id) bakes.
  {
    const r = B.r, src = [wide, { size: { x: 20, y: 10, z: 20 }, sparse: (() => { const s = makeSparse({ x: 20, y: 10, z: 20 }); for (let n = 0; n < 50; n++) sparseSet(s, n % 20, n % 10, (n * 7) % 20, 1); return s; })() }];
    const fromWorker = new PlacementBaker(), fromMain = new PlacementBaker();
    const models = src.map((m) => {
      const e = encodeModel(m);
      const moved = structuredClone(e, { transfer: encodedTransferables(e, { keepPlacement: true }) }) as EncodedModel;
      const id = r.addEncodedModel(moved);
      const keys = r.placementModel(id)!;
      fromWorker.register(placementModelOf(e, { key: keys.key, poseKey: keys.poseKey }));
      fromMain.register(structuredClone(keys));
      return { id, kept: e.subs.length > 0 && e.subs.length === keys.subs.length };
    });
    const list = models.flatMap(({ id }, k) => [{ model: id, base: 300, x: 40 + k * 30, y: 20, z: 40, yaw: 0.3 }, { model: id, base: 300, x: 90, y: 30 + k * 20, z: 120, mirror: true }]);
    const input = r.placementInput(list);
    const bytes = (b: ReturnType<PlacementBaker["bake"]>) => placementTransferables(b).map((a) => new Uint8Array(a));
    const x = bytes(fromWorker.bake(input)), y = bytes(fromMain.bake(input));
    ok(models.every((m) => m.kept) && x.every((a, i) => a.length === y[i].length && a.every((v, j) => v === y[i][j])), "  a worker registers its kept encoding with placementModelOf and bakes what placementModel(id) bakes");
  }

  // Readback: an encoding adopted into a pool of its own reads back every voxel (and part) of its source.
  {
    let bad = "";
    const rat = { size: { x: 16, y: 8, z: 8 }, data: new Uint8Array(16 * 64), parts: new Uint8Array(16 * 64) };
    for (let i = 0; i < rat.data.length; i++) if (i % 3) { rat.data[i] = 1 + (i % 4); rat.parts[i] = (i % 16) >> 2; }
    const s = makeSparse({ x: 70, y: 30, z: 41 });
    const rng = seededRandom(5);
    for (let n = 0; n < 4000; n++) sparseSet(s, Math.floor(rng() * 70), Math.floor(rng() * 30), Math.floor(rng() * 41), 1 + Math.floor(rng() * 30));
    for (const [name, src] of [["dense", wide], ["sparse", { size: s.size, sparse: s }], ["parts", rat]] as [string, ModelSource][]) {
      const e = encodeModel(src);
      // A pool in use, with freed slots and blocks: the second copy takes those first, in another order.
      const pool = new BrickPool();
      const grid = (top: Uint32Array, blocks: Uint32Array) => { const g = new BrickGrid(src.size, undefined, pool); top.forEach((t, i) => (g.top[i] = t ? blocks[t - 1] + 1 : 0)); return g; };
      new BrickGrid({ x: 64, y: 64, z: 64 }, new Uint8Array(64 ** 3).map((_, i) => i % 7), pool);
      const first = pool.adopt(e);
      grid(e.top, first.blocks).free();
      const claimed = pool.adopt(e);
      const g = grid(e.top, claimed.blocks), pg = e.partTop && grid(e.partTop, claimed.blocks);
      if (e.voxels4.length && claimed.slots4.every((v, i) => v === first.slots4[i])) bad = `${name}: the second copy did not move`;
      const { x: sx, y: sy, z: sz } = src.size;
      for (let z = 0; z < sz && !bad; z++)
        for (let y = 0; y < sy && !bad; y++)
          for (let x = 0; x < sx && !bad; x++) {
            const want = src.sparse ? sparseGet(src.sparse, x, y, z) : src.data![at(src.size, x, y, z)];
            if (g.get(x, y, z) !== want) bad = `${name} at ${x},${y},${z}`;
            if (pg && pg.get(x, y, z) !== (want ? src.parts![at(src.size, x, y, z)] + 1 : 0)) bad = `${name} part at ${x},${y},${z}`;
          }
    }
    ok(!bad, "  an encoding adopted into a pool already in use reads back every voxel and part of its source (dense, sparse, parts)", bad);
  }

  // Progress: (0, total) first, (total, total) last, never decreasing, and the same bytes.
  {
    const calls: [number, number][] = [];
    const now = performance.now;
    let clock = 0;
    performance.now = () => (clock += 10);
    let reported: EncodedModel, plain: EncodedModel;
    const src = { size: wide.size, data: wide.data, parts: wide.data!.map((v) => v % 3) };
    try { reported = encodeModel(src, { onProgress: (d, t) => calls.push([d, t]) }); } finally { performance.now = now; }
    plain = encodeModel(src);
    const total = calls[0]?.[1] ?? 0;
    const buf = (e: EncodedModel) => encodedTransferables(e).map((a) => new Uint8Array(a));
    const same = buf(plain).every((a, i) => { const b = buf(reported)[i]; return a.length === b.length && a.every((v, j) => v === b[j]); });
    ok(
      calls.length > 2 && total > 0 && calls[0][0] === 0 && calls.at(-1)![0] === total && calls.every(([d, t], i) => t === total && d <= total && (i === 0 || d >= calls[i - 1][0])) && same,
      `  encode progress runs from 0 to its fixed total (${total} units, ${calls.length} calls), never goes back, and changes no byte`,
      JSON.stringify(calls.slice(0, 5)),
    );
  }

  // reserveBricks: the same batch added after one reservation differs only in the pools' spare
  // capacity. Every other buffer the renderer holds is byte-identical, the brick pools (GPU and
  // CPU mirror) hold the same words at every claimed slot and zeros past them, and later adds
  // don't grow the pools again.
  {
    const batchSrc: ModelSource[] = [];
    const rng = seededRandom(31);
    for (let k = 0; k < 6; k++) {
      const size = { x: 30 + k * 7, y: 20 + k * 3, z: 25 }, data = new Uint8Array(size.x * size.y * size.z);
      for (let i = 0; i < data.length; i++) if (rng() < 0.3) data[i] = 1 + Math.floor(rng() * (k % 2 ? 40 : 6));
      batchSrc.push({ size, data });
    }
    const run = (reserve: boolean) => {
      const mock = mockGpu();
      const r = new Renderer(mock.gpu, placementScene.world(), "", { pipeline: "compute" });
      const ids0 = placementScene.addModels(r); // a pool in use, with a removed model's freed slots
      const batch = batchSrc.map((m) => encodeModel(m));
      if (reserve) r.reserveBricks(batch);
      const poolsOf = () => { const q = r as unknown as Record<string, unknown>; return [q.brickVox4, q.brickPal, q.brickVox8]; };
      const before = poolsOf();
      const ids = batch.map((e) => r.addEncodedModel(e));
      const grewAfter = poolsOf().filter((b, i) => b !== before[i]).length;
      console.warn = () => {};
      r.setInstances([...placementScene.lists(ids0).staticSet, ...ids.map((id, k) => ({ model: id, base: 300, x: 20 + k * 35, y: 60, z: 180, yaw: k }))]);
      console.warn = warn0;
      r.render(placementScene.frame());
      return { r, ids, grewAfter };
    };
    const A = run(false), B = run(true);
    type Buf = { size: number; bytes: Uint8Array; destroyed: boolean };
    const held = (r: object) => Object.entries(r).filter(([, v]) => v && typeof v === "object" && "bytes" in v && "destroyed" in v) as [string, Buf][];
    const pools = new Set(["brickVox4", "brickPal", "brickVox8"]);
    const pa = (A.r as unknown as { pool: BrickPool }).pool, pb = (B.r as unknown as { pool: BrickPool }).pool;
    const used: Record<string, number> = { brickVox4: pa.slots4 * 64 * 4, brickPal: pa.slots4 * 8 * 4, brickVox8: pa.slots8 * 128 * 4 };
    let diff = "", spare = "";
    const hb = new Map(held(B.r));
    for (const [name, a] of held(A.r)) {
      const b = hb.get(name)!;
      if (!b || a.destroyed || b.destroyed) { diff ||= `${name} missing or destroyed`; continue; }
      if (pools.has(name)) {
        const n = used[name];
        if (b.size !== a.size) spare ||= name;
        if (b.size < n) diff ||= `${name} is too small`;
        for (let j = 0; j < Math.max(a.size, b.size) && !diff; j++) if ((b.bytes[j] ?? 0) !== (j < n ? a.bytes[j] : 0) || (j >= n && (a.bytes[j] ?? 0) !== 0)) diff = `${name} byte ${j}`;
      } else if (a.size !== b.size || a.bytes.some((v, j) => v !== b.bytes[j])) diff ||= name;
    }
    for (const [k, words] of [["voxels4", 64], ["palettes", 8], ["voxels8", 128]] as const) {
      const n = (k === "voxels8" ? pa.slots8 : pa.slots4) * words, x = pa[k], y = pb[k];
      if (pa.slots4 !== pb.slots4 || pa.slots8 !== pb.slots8 || y.length < n || x.subarray(0, n).some((v, j) => v !== y[j]) || y.subarray(n).some((v) => v !== 0)) diff ||= `pool ${k}`;
    }
    if (pa.blockCount !== pb.blockCount || pa.blocks.some((v, j) => v !== pb.blocks[j])) diff ||= "pool blocks";
    ok(!diff && !!spare && A.ids.every((id, k) => id === B.ids[k]) && B.grewAfter === 0 && A.grewAfter > 0,
      `reserveBricks: only the pools' spare capacity differs (every other buffer byte-identical, pools equal at the claimed slots), the same ids, and the adds after it grow nothing (without it, the adds reallocate ${A.grewAfter} pools)`,
      diff || `spare ${spare}, grew after ${B.grewAfter}`);
  }

  // Growing the brick pools copies them GPU-side (copyBufferToBuffer) instead of re-uploading the
  // CPU mirror. A world streamed in brick column by brick column, with models added and removed
  // and old bricks rewritten and cleared between the batches, grows both pools many times. After
  // it every GPU pool must hold exactly what the re-upload would have written (the mirror, zeros
  // past it), each copy must be submitted before its source is destroyed (the mock refuses a copy
  // of a destroyed buffer, and a write into one), and a renderer that reserved the same counts up
  // front (reserveBricks({ bricks4, bricks8 })) must match it byte for byte outside the pools'
  // spare capacity: growth changes nothing but capacity, the index buffer included.
  {
    const brickFill = (seed: number) => (cells: Uint8Array, ox: number, oy: number, oz: number) => {
      const h = ((ox * 73856093) ^ (oy * 19349663) ^ (oz * 83492791) ^ seed) >>> 0;
      const rng = seededRandom(h);
      const kind = h % 11;
      if (kind === 0) return false; // left as it is
      if (kind === 1) { cells.fill(7); return true; } // uniform: no slot
      const values = kind === 2 || kind === 3 ? 60 : 6; // some need the 8-bit tier
      for (let i = 0; i < 512; i++) cells[i] = rng() < 0.5 ? 1 + Math.floor(rng() * values) : 0;
      return true;
    };
    const columns = (from: number, to: number, h: number) => {
      const boxes = [];
      for (let c = from; c < to; c++) { const bx = c % 32, bz = Math.floor(c / 32); boxes.push({ x0: bx * 8, y0: 0, z0: bz * 8, x1: bx * 8 + 7, y1: h - 1, z1: bz * 8 + 7 }); }
      return boxes;
    };
    const run = (reserve?: { bricks4: number; bricks8: number }) => {
      const mock = mockGpu();
      const r = new Renderer(mock.gpu, placementScene.world(), "", { pipeline: "compute" });
      const q = r as unknown as { poolGrowths: number; pool: BrickPool } & Record<string, Buf>;
      const grown0 = q.poolGrowths;
      if (reserve) r.reserveBricks(reserve);
      for (let batch = 0; batch < 8; batch++) {
        r.editMany(columns(batch * 96, (batch + 1) * 96, 32), brickFill(batch));
        if (batch === 2) placementScene.addModels(r);
        if (batch === 4) r.removeModel(1);
        // Rewrite and clear bricks claimed batches ago, in the same commit as fresh claims.
        if (batch >= 3) r.editMany([...columns((batch - 3) * 96, (batch - 3) * 96 + 40, 32), ...columns(800 + batch * 20, 820 + batch * 20, 40)], brickFill(100 + batch));
        if (batch === 5) r.clear({ x0: 0, y0: 0, z0: 0, x1: 255, y1: 15, z1: 23 });
      }
      r.render(placementScene.frame());
      return { mock, r, q, grown: q.poolGrowths - grown0, pool: q.pool };
    };
    type Buf = { size: number; usage: number; bytes: Uint8Array; destroyed: boolean };
    const A = run();
    const words = { brickVox4: [64, A.pool.slots4, "voxels4"], brickPal: [8, A.pool.slots4, "palettes"], brickVox8: [128, A.pool.slots8, "voxels8"] } as const;
    let asMirror = "";
    for (const [name, [w, , k]] of Object.entries(words)) {
      const b = A.q[name], mirror = new Uint8Array(A.pool[k].buffer, 0, Math.min(b.size, A.pool[k].byteLength));
      for (let j = 0; j < b.size && !asMirror; j++) if (b.bytes[j] !== (j < mirror.length ? mirror[j] : 0)) asMirror = `${name} byte ${j} (slot ${Math.floor(j / (w * 4))})`;
    }
    const current = new Set(Object.keys(words).map((n) => A.q[n] as unknown));
    const sources = A.mock.copies.map((c) => c.src);
    const replacedPools = A.mock.buffers.filter((b) => (b.usage & 4) && !current.has(b) && b.size % 32 === 0 && sources.includes(b));
    ok(A.grown >= 4 && !asMirror && sources.every((b) => b.destroyed) && A.mock.copies.length > 0 && replacedPools.length === new Set(sources).size,
      `pool growth: ${A.grown} reallocations while streaming (${A.pool.slots4} 4-bit and ${A.pool.slots8} 8-bit slots) leave every GPU pool equal to the CPU mirror, as the re-upload wrote it; ${A.mock.copies.length} GPU-side copies, each source destroyed after its submit`,
      asMirror || `grew ${A.grown}, copies ${A.mock.copies.length}`);

    const B = run({ bricks4: A.pool.slots4, bricks8: A.pool.slots8 });
    const held = (r: object) => Object.entries(r).filter(([, v]) => v && typeof v === "object" && "bytes" in v && "destroyed" in v) as [string, Buf][];
    const hb = new Map(held(B.r));
    let diff = "", spare = "";
    for (const [name, a] of held(A.r)) {
      const b = hb.get(name);
      if (!b || a.destroyed || b.destroyed) { diff ||= `${name} missing or destroyed`; continue; }
      if (name in words) {
        const [w, slots] = words[name as keyof typeof words], n = slots * w * 4;
        if (a.size !== b.size) spare ||= name;
        if (b.size < n) diff ||= `${name} is too small`;
        for (let j = 0; j < Math.max(a.size, b.size) && !diff; j++) if ((b.bytes[j] ?? 0) !== (j < n ? a.bytes[j] : 0) || (j >= n && (a.bytes[j] ?? 0) !== 0)) diff = `${name} byte ${j}`;
      } else if (a.size !== b.size || a.bytes.some((v, j) => v !== b.bytes[j])) diff ||= name;
    }
    ok(!diff && B.grown === 1 && B.pool.slots4 === A.pool.slots4 && B.pool.slots8 === A.pool.slots8,
      `  reserveBricks({ bricks4, bricks8 }) up front: one reallocation, none while streaming, and every buffer byte-identical to growing as it goes (pools equal at every claimed slot; ${spare ? "spare capacity differs" : "the same capacity"})`,
      diff || `grew ${B.grown}`);

    // The count form checks its arguments, and the limits: a reservation that fits the device's
    // storage binding exactly succeeds (no headroom past it), one slot more throws, and a
    // maxBufferSize under the binding limit is the one that counts.
    const mock = mockGpu();
    const r = new Renderer(mock.gpu, placementScene.world(), "", { pipeline: "compute" });
    const lim = mock.gpu.limits as { maxStorageBufferBindingSize: number; maxBufferSize: number };
    const bad = [{ bricks4: -1 }, { bricks8: 1.5 }, { bricks4: NaN }].filter((c) => { try { r.reserveBricks(c); return true; } catch { return false; } });
    lim.maxStorageBufferBindingSize = 256 * 4000; // 4000 4-bit slots, 2000 8-bit ones
    r.reserveBricks({ bricks4: 4000, bricks8: 2000 });
    const st = r.stats();
    let over = "", overBuf = "";
    try { r.reserveBricks({ bricks4: 4001 }); } catch (e) { over = String(e); }
    lim.maxStorageBufferBindingSize = 1 << 30;
    lim.maxBufferSize = 512 * 3000;
    try { r.reserveBricks({ bricks8: 3001 }); } catch (e) { overBuf = String(e); }
    r.reserveBricks({ bricks8: 3000 });
    const before = r.stats();
    r.reserveBricks({});
    r.reserveBricks({ bricks4: 10 });
    ok(!bad.length && st.cap4 === 4000 && st.cap8 === 2000 && over.includes("storage binding") && overBuf.includes("maxBufferSize") && before.cap8 === 3000 && r.stats().cap4 === before.cap4,
      "  reserveBricks({ bricks4, bricks8 }) refuses bad counts, fills the device's limit exactly, and throws past it (binding or buffer size, whichever is lower)",
      JSON.stringify({ bad, st, over, overBuf }));
  }

  // The CPU mirror grows by 1.5x and stops at what the device's GPU pools could hold: a renderer
  // on a device capping buffers at 16 MiB (65536 4-bit slots, 32768 8-bit ones; nightwood at
  // 100 vox/m hit the same wall at 1 GiB, where the mirror asked for 2 GiB) streams bricks in until
  // a claim no longer fits. The mirror then holds exactly the limit, and the claim past it throws
  // the named-limit error instead of allocating; a reservation past it throws and changes nothing.
  {
    const mock = mockGpu();
    const lim = mock.gpu.limits as { maxStorageBufferBindingSize: number; maxBufferSize: number };
    lim.maxStorageBufferBindingSize = 16 << 20;
    lim.maxBufferSize = 16 << 20;
    const r = new Renderer(mock.gpu, { size: { x: 512, y: 256, z: 512 }, palette: new Float32Array(256 * 4) }, "", { pipeline: "compute" });
    const pool = (r as unknown as { pool: BrickPool }).pool;
    const lengths: number[] = [];
    let err = "", batches = 0;
    const noisy = (cells: Uint8Array, ox: number, oy: number, oz: number) => {
      const rng = seededRandom(((ox * 7919) ^ (oy * 104729) ^ (oz * 1299709)) >>> 0);
      for (let i = 0; i < 512; i++) cells[i] = 1 + Math.floor(rng() * 6);
      return true;
    };
    try {
      for (let y = 0; y < 256 && !err; y += 8, batches++) {
        r.editMany([{ x0: 0, y0: y, z0: 0, x1: 511, y1: y + 7, z1: 511 }], noisy);
        if (lengths.at(-1) !== pool.voxels4.length) lengths.push(pool.voxels4.length);
      }
    } catch (e) { err = String(e); }
    const slots = lengths.map((n) => n / 64), steps = slots.slice(1).map((n, i) => n / slots[i]);
    const at = pool.voxels4.length, before = r.stats();
    let resErr = "";
    try { r.reserveBricks({ bricks4: 65536 - pool.slots4 + 1 }); } catch (e) { resErr = String(e); }
    ok(err.includes("storage binding") && Math.max(...slots) === 65536 && at === 65536 * 64 && pool.palettes.length === 65536 * 8 && pool.slots4 === 65536
        && steps.slice(0, -1).every((q) => q > 1.45 && q < 1.55) && resErr.includes("brick pool") && pool.voxels4.length === at && r.stats().cap4 === before.cap4,
      `mirror growth: 1.5x steps (${slots.length} sizes, up to ${Math.max(...slots)} slots), capped at the device's pool limit, and the claim past it throws the named-limit error instead of allocating more`,
      JSON.stringify({ err: err.slice(0, 120), slots, resErr: resErr.slice(0, 80), claimed: pool.slots4, batches }));
  }

  // beginEncodedModel: the add spread over steps. A big model (several slices of each part) added
  // in 1, 3 and N steps, with other adds (one of them pending too), edits, a removal, a static set
  // of other models and a pool growth between the steps, must leave every buffer byte-identical to
  // addEncodedModel called at the begin point followed by the same operations. Then cancel, at
  // several points, must give the pools back exactly, an instance naming a pending id must throw,
  // and progress must run from 0 to 1.
  {
    type Buf = { size: number; bytes: Uint8Array; destroyed: boolean };
    type Pending = ReturnType<InstanceType<typeof Renderer>["beginEncodedModel"]>;
    const SLICES = { s4: 3640, s8: 2048, blocks: 256 }; // renderer.ts SLICE_*
    // 18 x 1 x 17 top cells, each with bricks of both tiers: 306 index blocks, about 9000 4-bit
    // and 4600 8-bit slots.
    const bigSrc = (() => {
      const size = { x: 1150, y: 40, z: 1088 }, s = makeSparse(size);
      const rng = seededRandom(808);
      const [dx, dy] = [Math.ceil(size.x / 8), Math.ceil(size.y / 8)];
      for (let tz = 0; tz < 17; tz++)
        for (let tx = 0; tx < 18; tx++)
          for (let n = 0; n < 45; n++) {
            const bx = tx * 8 + Math.floor(rng() * 8), by = Math.floor(rng() * 5), bz = tz * 8 + Math.floor(rng() * 8);
            if (bx * 8 >= size.x || bz * 8 >= size.z) continue;
            const cells = new Uint8Array(512), values = n % 3 === 0 ? 50 : 9;
            for (let i = 0; i < 512; i++) if (rng() < 0.6) cells[i] = 1 + Math.floor(rng() * values);
            s.bricks.set(bx + by * dx + bz * dx * dy, cells);
          }
      return { size, sparse: s } as ModelSource;
    })();
    const X0 = encodeModel(bigSrc);
    const n4 = X0.voxels4.length / 64, n8 = X0.voxels8.length / 128, nb = X0.blocks.length / 512;
    const slices = Math.ceil(n4 / SLICES.s4) + Math.ceil(n8 / SLICES.s8) + Math.ceil(nb / SLICES.blocks);
    const copy = (e: EncodedModel) => structuredClone(e) as EncodedModel;
    const smallSrc = (k: number): ModelSource => {
      const size = { x: 40 + k * 9, y: 24, z: 30 }, data = new Uint8Array(size.x * size.y * size.z), rng = seededRandom(900 + k);
      for (let i = 0; i < data.length; i++) if (rng() < 0.3) data[i] = 1 + Math.floor(rng() * (k % 2 ? 40 : 7));
      return { size, data };
    };
    const S1 = encodeModel(smallSrc(1)), S2 = encodeModel(smallSrc(2));
    const fill = (seed: number) => (cells: Uint8Array, ox: number, oy: number, oz: number) => {
      const rng = seededRandom(((ox * 7919) ^ (oy * 104729) ^ (oz * 1299709) ^ seed) >>> 0);
      for (let i = 0; i < 512; i++) cells[i] = rng() < 0.4 ? 1 + Math.floor(rng() * (seed % 2 ? 30 : 5)) : 0;
      return true;
    };
    const setup = () => {
      const mock = mockGpu();
      const r = new Renderer(mock.gpu, placementScene.world(), "", { pipeline: "compute" });
      const ids0 = placementScene.addModels(r); // a pool in use, with a removed model's freed slots
      return { mock, r, ids0, q: r as unknown as { pool: BrickPool; poolGrowths: number } & Record<string, Buf> };
    };
    const held = (r: object) => Object.entries(r).filter(([, v]) => v && typeof v === "object" && "bytes" in v && "destroyed" in v) as [string, Buf][];
    const sameAll = (a: { mock: ReturnType<typeof mockGpu>; r: object }, b: { mock: ReturnType<typeof mockGpu>; r: object }) => {
      if (a.mock.buffers.length !== b.mock.buffers.length) return `${a.mock.buffers.length} vs ${b.mock.buffers.length} buffers`;
      for (let i = 0; i < a.mock.buffers.length; i++) {
        const x = a.mock.buffers[i], y = b.mock.buffers[i];
        if (x.size !== y.size || x.destroyed !== y.destroyed) return `buffer ${i}: size or state`;
        // A pool replaced by a growth holds what was written before it (the sync add wrote more).
        if (x.destroyed) continue;
        for (let j = 0; j < x.size; j++) if (x.bytes[j] !== y.bytes[j]) return `buffer ${i} byte ${j}`;
      }
      const pa = (a.r as { pool: BrickPool }).pool, pb = (b.r as { pool: BrickPool }).pool;
      for (const k of ["voxels4", "palettes", "voxels8", "blocks"] as const) if (pa[k].length !== pb[k].length || pa[k].some((v, j) => v !== pb[k][j])) return `pool ${k}`;
      if (pa.slots4 !== pb.slots4 || pa.slots8 !== pb.slots8 || pa.blockCount !== pb.blockCount) return "pool counts";
      return "";
    };
    // The GPU pools hold the CPU mirror at every claimed slot (what a re-upload would write).
    const asMirror = (q: ReturnType<typeof setup>["q"]) => {
      for (const [name, w, k, n] of [["brickVox4", 64, "voxels4", q.pool.slots4], ["brickPal", 8, "palettes", q.pool.slots4], ["brickVox8", 128, "voxels8", q.pool.slots8]] as const) {
        const b = q[name], m = new Uint8Array(q.pool[k].buffer, 0, n * w * 4);
        for (let j = 0; j < m.length; j++) if (b.bytes[j] !== m[j]) return `${name} byte ${j}`;
      }
      return "";
    };

    // --- interleaved steps, against the sync add at the begin point ---
    const run = (mode: "sync" | "1" | "3" | "N") => {
      const s = setup(), { r, ids0, q } = s;
      let small = -1, other: Pending | undefined, grewWhilePending = 0;
      const ops: (() => void)[] = [
        () => { small = r.addEncodedModel(copy(S1)); },
        () => r.editMany([{ x0: 0, y0: 0, z0: 0, x1: 127, y1: 15, z1: 127 }], fill(1)),
        () => { other = r.beginEncodedModel(copy(S2)); other.step(0); },
        () => { const g = q.poolGrowths; r.reserveBricks({ bricks4: r.stats().cap4, bricks8: r.stats().cap8 }); grewWhilePending += q.poolGrowths - g; },
        () => r.removeModel(ids0.tree),
        // (Not the removed model's instances: the big model took its id and is still pending.)
        () => { console.warn = () => {}; r.setInstances(placementScene.lists(ids0).staticSet.filter((i) => i.model !== ids0.gone)); console.warn = warn0; },
        () => { other!.step(Infinity); },
        () => r.editMany([{ x0: 64, y0: 8, z0: 64, x1: 200, y1: 23, z1: 180 }], fill(2)),
      ];
      let id: number, steps = 0, progress: number[] = [];
      if (mode === "sync") {
        id = r.addEncodedModel(copy(X0));
        for (const f of ops) f();
      } else {
        const p = r.beginEncodedModel(copy(X0));
        id = p.id;
        progress.push(p.progress);
        const budgets = mode === "1" ? [Infinity] : mode === "3" ? [0, 0, Infinity] : [];
        let k = 0;
        for (;;) {
          const last = mode === "N" ? false : steps === budgets.length - 1;
          // Before step i, its share of the operations (all of them before the only step of "1").
          const upto = mode === "N" ? steps + 1 : Math.round(((steps + 1) * ops.length) / budgets.length);
          while (k < Math.min(upto, ops.length)) ops[k++]();
          const done = p.step(mode === "N" ? 0 : budgets[steps]);
          steps++;
          progress.push(p.progress);
          if (done) break;
          if (last) throw new Error("not done after its last step");
        }
        while (k < ops.length) ops[k++]();
      }
      console.warn = () => {};
      r.setInstances([...placementScene.lists(ids0).staticSet, { model: id, base: 300, x: 20, y: 40, z: 20, yaw: 0.4 }, { model: small, base: 300, x: 150, y: 60, z: 150 }]);
      console.warn = warn0;
      r.render(placementScene.frame());
      return { ...s, id, small, steps, progress, grewWhilePending };
    };
    const ref = run("sync");
    const results = (["1", "3", "N"] as const).map((m) => ({ m, x: run(m) }));
    let bad = "";
    for (const { m, x } of results) {
      const d = sameAll(ref, x);
      if (d) bad ||= `${m} steps: ${d}`;
      if (x.id !== ref.id || x.small !== ref.small) bad ||= `${m} steps: ids ${x.id}/${x.small} vs ${ref.id}/${ref.small}`;
      if (x.progress[0] !== 0 || x.progress.at(-1) !== 1 || x.progress.some((v, i) => i && v < x.progress[i - 1])) bad ||= `${m} steps: progress ${x.progress.slice(0, 4)}`;
    }
    const N = results[2].x;
    ok(!bad && n4 > 2 * SLICES.s4 && n8 > 2 * SLICES.s8 && nb > SLICES.blocks && results[1].x.steps === 3 && N.steps === slices && N.grewWhilePending > 0 && ref.grewWhilePending > 0 && !asMirror(N.q),
      `beginEncodedModel in 1, 3 and ${N.steps} steps, with adds (one pending too), edits, a removal, a static set and a pool growth between them, leaves every buffer and the pools byte-identical to addEncodedModel at the begin point (${n4} 4-bit, ${n8} 8-bit slots, ${nb} blocks; progress 0 to 1, never back)`,
      bad || JSON.stringify({ n4, n8, nb, steps3: results[1].x.steps, stepsN: N.steps, slices, grew: N.grewWhilePending, mirror: asMirror(N.q) }));

    // --- cancel ---
    const snapshot = (s: ReturnType<typeof setup>) => {
      const p = s.q.pool as unknown as Record<string, unknown> & BrickPool, rr = s.r as unknown as Record<string, unknown>;
      // Every slot in use: free slots' words are never read (a cancel leaves in them what it copied).
      const free4 = new Set(p.free4 as number[]), free8 = new Set(p.free8 as number[]);
      const used = (a: Uint32Array, n: number, w = 0, free?: Set<number>) => Array.from(a.subarray(0, n)).map((v, j) => (free?.has(Math.floor(j / w)) ? 0 : v));
      return JSON.stringify({
        slots4: p.slots4, slots8: p.slots8, blockCount: p.blockCount, free4: p.free4, free8: p.free8, freeBlocks: p.freeBlocks,
        v4: used(p.voxels4, p.slots4 * 64, 64, free4).join(), pal: used(p.palettes, p.slots4 * 8, 8, free4).join(), v8: used(p.voxels8, p.slots8 * 128, 128, free8).join(), blocks: used(p.blocks, p.blockCount * 512).join(),
        zeros: [p.voxels4.subarray(p.slots4 * 64), p.palettes.subarray(p.slots4 * 8), p.voxels8.subarray(p.slots8 * 128), p.blocks.subarray(p.blockCount * 512)].every((a) => a.every((v) => v === 0)),
        topFree: rr.topFree, topEnd: rr.topEnd, models: (rr.models as unknown[]).length,
      });
    };
    const refAdd = setup();
    const refId = refAdd.r.addEncodedModel(copy(X0));
    refAdd.r.render(placementScene.frame());
    bad = "";
    for (const at of [0, 1, 3, Math.ceil(n4 / SLICES.s4) + 1, slices - 1]) {
      const s = setup(), before = snapshot(s);
      const p = s.r.beginEncodedModel(copy(X0));
      for (let i = 0; i < at; i++) s.r.editMany([], fill(0)), p.step(0);
      const progressAt = p.progress;
      p.cancel();
      const after = snapshot(s);
      // Then the same add, sync: exactly what a renderer that never began one ends up with.
      const id = s.r.addEncodedModel(copy(X0));
      s.r.render(placementScene.frame());
      const d = sameAll(refAdd, s);
      if (after !== before) bad ||= `cancel after ${at} slices: the pools differ from before`;
      else if (d || id !== refId) bad ||= `cancel after ${at} slices, then addEncodedModel: ${d || `id ${id} vs ${refId}`}`;
      else if (!(at === 0 ? progressAt === 0 : progressAt > 0 && progressAt < 1)) bad ||= `cancel after ${at} slices: progress ${progressAt}`;
    }
    ok(!bad, `  cancel after 0, 1, 3 slices, in the 8-bit slots and before the last of ${slices} slices gives the pools back exactly (every slot in use, blocks, free lists, counts, zeros past the end), and the same add after it uploads what one never cancelled does`, bad);

    // Cancel with a claim after it: its slots and blocks go onto the free lists, the GPU pools
    // still hold the mirror, and the next adds take them.
    {
      const s = setup(), pool = s.q.pool as unknown as { free4: number[]; free8: number[]; freeBlocks: number[] } & BrickPool;
      const p = s.r.beginEncodedModel(copy(X0));
      p.step(0), p.step(0);
      const a = s.r.addEncodedModel(copy(S1));
      const f4 = pool.free4.length, fb = pool.freeBlocks.length;
      p.cancel();
      const freed = pool.free4.length - f4 === n4 && pool.freeBlocks.length - fb === nb;
      const again = s.r.addEncodedModel(copy(X0));
      console.warn = () => {};
      s.r.setInstances([{ model: again, base: 300, x: 10, y: 10, z: 10 }, { model: a, base: 300, x: 100, y: 10, z: 100 }]);
      console.warn = warn0;
      const reused = pool.free4.length === f4 && pool.freeBlocks.length === fb;
      ok(freed && reused && again === p.id && a !== p.id && !asMirror(s.q) && s.r.placementModel(again)!.subs.length > 0,
        "  a cancel after later claims frees its slots and blocks onto the free lists, the GPU pools still match the mirror, and the id and slots are taken again",
        JSON.stringify({ freed, reused, again, a, id: p.id, mirror: asMirror(s.q) }));
    }

    // --- a pending id ---
    {
      const s = setup(), r = s.r;
      const models0 = r.instanceStats().models;
      const p = r.beginEncodedModel(copy(S2));
      const other = r.addEncodedModel(copy(S1));
      const threw = (f: () => void) => { try { f(); return ""; } catch (e) { return String(e); } };
      const inst = { model: p.id, base: 300, x: 10, y: 10, z: 10 };
      const t1 = threw(() => r.setInstances([inst]));
      const t2 = threw(() => r.setInstances([inst], { dynamic: true }));
      r.setInstances([], { dynamic: true });
      const t3 = threw(() => r.placementInput([inst]));
      r.removeModel(p.id); // not a model yet: a no-op
      const hidden = r.placementModel(p.id) === null && r.instanceStats().models === models0 + 1;
      const done = p.step(Infinity), again = p.step(0);
      const t4 = threw(() => r.setInstances([inst]));
      const c = r.beginEncodedModel(copy(S2));
      c.cancel(); c.cancel(); p.cancel();
      const t5 = threw(() => c.step(1));
      ok([t1, t2, t3].every((t) => t.includes("still being added")) && !t4 && t5.includes("cancelled") && hidden && done && again && p.done && other !== p.id && r.placementModel(p.id) !== null,
        "  an instance naming a pending id throws (static, dynamic, placementInput); the id is invisible and skipped by other adds until done; cancel twice or after done is a no-op; step after cancel throws",
        JSON.stringify({ t1, t2, t3, t4, t5, hidden, other, id: p.id, models: r.instanceStats().models, models0 }));
    }
  }

  // A stale encoding (another layout version) is refused, and nothing is claimed.
  {
    const e = encodeModel(wide);
    const before = A.r.instanceStats().models;
    let threw = "";
    try { A.r.addEncodedModel({ ...e, version: ENCODED_MODEL_VERSION + 1 }); } catch (err) { threw = String(err); }
    ok(threw.includes("version") && A.r.instanceStats().models === before, "  an encoding of another version throws and adds nothing", threw);
  }
}

console.log("\nupscaled instances (ModelOptions.scale):");

// A model drawn at scale k must draw exactly what the model upsampled k times by nearest
// neighbour draws at scale 1: the same instance record, and the same value at every world cell,
// for k = 2, 3, 5, 10, at quarter and arbitrary yaws, tilted, mirrored, at fractional positions
// and with given anchors. Then the placement bake at k must list every cell such an instance
// draws, and the renderer's sync and worker paths must upload the same bytes.
{
  const rng = seededRandom(5150);
  const upsample = (size: Size, data: Uint8Array, k: number) => {
    const up = { x: size.x * k, y: size.y * k, z: size.z * k }, d = new Uint8Array(up.x * up.y * up.z);
    let i = 0;
    for (let z = 0; z < up.z; z++) for (let y = 0; y < up.y; y++) for (let x = 0; x < up.x; x++, i++) d[i] = data[at(size, Math.floor(x / k), Math.floor(y / k), Math.floor(z / k))];
    return { size: up, data: d };
  };
  const subsOf = (size: Size, data: Uint8Array) => {
    const s = new Set<number>(), out: number[] = [];
    let i = 0;
    for (let z = 0; z < size.z; z++) for (let y = 0; y < size.y; y++) for (let x = 0; x < size.x; x++, i++) {
      const key = (x >> 1) + (y >> 1) * 4096 + (z >> 1) * 4096 * 4096;
      if (data[i] && !s.has(key)) { s.add(key); out.push(x >> 1, y >> 1, z >> 1); }
    }
    return Int32Array.from(out);
  };
  const modelAt = (k: number) => {
    // Coarse models about 40 world voxels wide at every k, with odd sizes (partial bricks).
    const n = () => Math.max(3, Math.round((30 + rng() * 20) / k)) | 1;
    const size = { x: n(), y: n() + 1, z: n() };
    const data = new Uint8Array(size.x * size.y * size.z);
    for (let i = 0; i < data.length; i++) if (rng() < 0.18) data[i] = 1 + Math.floor(rng() * 5);
    return { size, data };
  };
  const tilt = () => { const a = rng() * 3, b = rng() * 3; const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b); return [cb, -sb * ca, sb * sa, sb, cb * ca, -cb * sa, 0, sa, ca]; };
  const placementsFor = (k: number, size: Size, count: number) => Array.from({ length: count }, (_, j) => {
    const kind = j % 5;
    return {
      x: 60 + rng() * 120 + (kind === 4 ? 0 : 0.37), y: 20 + rng() * 40, z: 60 + rng() * 120, base: 1,
      yaw: kind === 0 ? (Math.floor(rng() * 4) * Math.PI) / 2 : kind === 1 ? 0 : rng() * Math.PI * 2,
      rotation: kind === 3 ? tilt() : undefined,
      mirror: j % 3 === 0,
      anchor: j % 4 === 1 ? ([Math.floor(rng() * size.x * k), Math.floor(rng() * 3), Math.floor(rng() * size.z * k)] as [number, number, number]) : undefined,
    };
  });

  // 1. The CPU twin at k against the upsampled model at 1.
  let cells = 0, drawn = 0, diff = 0, wordsDiffer = 0, first = "";
  const t0 = performance.now();
  for (const k of [2, 3, 5, 10]) {
    for (let mm = 0; mm < 2; mm++) {
      const m = modelAt(k), up = upsample(m.size, m.data, k);
      const coarse = { size: m.size, voxel: (x: number, y: number, z: number) => m.data[at(m.size, x, y, z)], part: () => 0 };
      const fine = { size: up.size, voxel: (x: number, y: number, z: number) => up.data[at(up.size, x, y, z)], part: () => 0 };
      for (const inst of placementsFor(k, m.size, 5)) {
        const wa = new Uint32Array(INST_WORDS), wb = new Uint32Array(INST_WORDS);
        const { box } = packInstance(inst, { size: m.size, scale: k }, 0, wa, 0);
        packInstance(inst, { size: up.size }, 0, wb, 0);
        // The same record but for the scale, which the flags word carries.
        if (wa.some((v, i) => v !== (i === 14 ? (wb[i] | ((k - 1) << INST_SCALE_SHIFT)) >>> 0 : wb[i]))) wordsDiffer++;
        for (let z = Math.floor(box[2]) - 1; z <= Math.ceil(box[5]) + 1; z++)
          for (let y = Math.floor(box[1]) - 1; y <= Math.ceil(box[4]) + 1; y++)
            for (let x = Math.floor(box[0]) - 1; x <= Math.ceil(box[3]) + 1; x++) {
              cells++;
              const a = sampleInstance(wa, 0, undefined, coarse, x, y, z), b = sampleInstance(wb, 0, undefined, fine, x, y, z);
              if (b) drawn++;
              if (a !== b) { diff++; first ||= `k ${k} at ${x},${y},${z}: ${a} vs ${b}`; }
            }
      }
    }
  }
  ok(wordsDiffer === 0, "an instance of a model at scale k packs the record of the model upsampled k times at scale 1, plus k - 1 in its flags (k = 2, 3, 5, 10)", `${wordsDiffer} records differ`);
  ok(diff === 0 && drawn > 100000, `  and samples the same value at every world cell (${cells} cells, ${drawn} drawn, ${Math.round(performance.now() - t0)} ms)`, `${diff} differ, first ${first}`);

  // 2. The placement bake at k lists every cell the instance draws, in its sub-cell table or, for
  // a cell without one, in its cell list. Checked against the same set with the scale left out
  // of the placement data (it must miss cells), and compared with baking the upsampled models.
  {
    const grid = { brickDim: [40, 16, 40] as [number, number, number], topDim: [5, 2, 5] as [number, number, number], gridMax: 320 };
    const models: PlacementModel[] = [], upModels: PlacementModel[] = [], samplers: SampleModelLike[] = [];
    const instances: PlacementInstance[] = [];
    let key = 900;
    for (const k of [2, 3, 5, 10]) {
      for (let mm = 0; mm < 2; mm++) {
        const m = modelAt(k), up = upsample(m.size, m.data, k), id = models.length;
        models.push({ key: ++key, size: m.size, scale: k, subs: subsOf(m.size, m.data) });
        upModels.push({ key, size: up.size, subs: subsOf(up.size, up.data) });
        samplers.push({ size: m.size, voxel: (x, y, z) => m.data[at(m.size, x, y, z)], part: () => 0 });
        for (const p of placementsFor(k, m.size, 5)) instances.push({ ...p, model: id, x: p.x + 20, z: p.z + 20 });
      }
    }
    const input: PlacementInput = { grid, models: models.map((m) => m.key), instances };
    const bakeWith = (ms: PlacementModel[]) => { const t = performance.now(); const b = bakePlacement(input, (kk) => ms.find((m) => m.key === kk)); return { b, ms: performance.now() - t }; };
    const scaled = bakeWith(models), unscaled = bakeWith(models.map(({ scale: _, ...m }) => m)), upsampled = bakeWith(upModels);
    const unlisted = (bake: PlacementBake) => {
      let n = 0, drawnHere = 0, firstMiss = "";
      const lists = { data: bake.subs, cellWords: bake.subCells, stats: bake.stats };
      instances.forEach((p, k) => {
        const b = scaled.b.boxes.subarray(k * 6, k * 6 + 6), sm = samplers[p.model];
        for (let z = Math.max(0, Math.floor(b[2]) - 1); z <= Math.min(127, Math.ceil(b[5]) + 1); z++)
          for (let y = Math.max(0, Math.floor(b[1]) - 1); y <= Math.min(127, Math.ceil(b[4]) + 1); y++)
            for (let x = Math.max(0, Math.floor(b[0]) - 1); x <= Math.min(319, Math.ceil(b[3]) + 1); x++) {
              if (!sampleInstance(scaled.b.inst, k * INST_WORDS, undefined, sm, x, y, z)) continue;
              drawnHere++;
              const ci = (x >> 6) + (y >> 6) * 5 + (z >> 6) * 10, c = bake.cells[ci];
              const l = readSubLists(lists, grid.topDim, x, y, z) ?? Array.from(bake.list.subarray(c >>> 8, (c >>> 8) + (c & 0xff)));
              if (!l.includes(k)) { n++; firstMiss ||= `instance ${k} at ${x},${y},${z}`; }
            }
      });
      return { n, drawnHere, firstMiss };
    };
    const s = unlisted(scaled.b), u = unlisted(unscaled.b);
    ok(s.n === 0 && s.drawnHere > 100000 && scaled.b.stats.tables > 0, `the placement bake at k lists every cell its instances draw (${s.drawnHere} cells, ${scaled.b.stats.tables} tables)`, `${s.n} unlisted, first ${s.firstMiss}`);
    ok(u.n > 0, `  and the same set baked with the scale left out misses cells (${u.n})`);
    ok(scaled.b.inst.every((v, i) => (i % INST_WORDS === 14 ? v & ((1 << INST_SCALE_SHIFT) - 1) : v) === upsampled.b.inst[i]), `  its instance records are the upsampled models', but for the scale in the flags (${scaled.b.stats.pairs} sub-cell entries in ${scaled.ms.toFixed(0)} ms vs ${upsampled.b.stats.pairs} in ${upsampled.ms.toFixed(0)} ms for the upsampled models)`);
    // Progress: the scaled bake's units still run 0 → total.
    const calls: number[][] = [];
    bakePlacement(input, (kk) => models.find((m) => m.key === kk), { onProgress: (d, t) => calls.push([d, t]) });
    ok(calls.length >= 2 && calls[0][0] === 0 && calls.at(-1)![0] === calls[0][1], "  its progress runs from 0 to its total");
    // Posing needs scale 1.
    let posed = "";
    try {
      bakePlacement({ grid, models: [7], instances: [{ model: 0, base: 1, x: 50, y: 20, z: 50, parts: new Float32Array(12) }] }, () => ({ key: 7, size: { x: 4, y: 4, z: 4 }, scale: 2, subs: new Int32Array(3), partBoxes: new Int32Array([0, 0, 0, 3, 3, 3]), poseKey: 1 }));
    } catch (e) { posed = String(e); }
    ok(posed.includes("needs scale 1"), "  a posed instance of a scaled model throws", posed);
  }

  // 3. The renderer: addModel(src, { scale }) + setInstances against the worker path
  // (encodeModel, transferred, addEncodedModel(e, { scale }), a baker fed placementModelOf with
  // the renderer's keys, applyPlacement), byte for byte, with a moving scaled set on top; and
  // scale 1 given explicitly changes no byte.
  {
    const srcs = [2, 5, 1].map((k) => {
      const m = modelAt(Math.max(2, k));
      return { src: { size: m.size, data: m.data } as ModelSource, k };
    });
    const list = (ids: number[]): Instance[] => ids.flatMap((id, j) => [
      { model: id, base: 300, x: 30 + j * 60, y: 20, z: 40, yaw: 0.4 },
      { model: id, base: 300, x: 50 + j * 50, y: 30, z: 150.5, mirror: true, yaw: Math.PI / 2 },
      { model: id, base: 300, x: 140, y: 10 + j * 30, z: 90, rotation: [0.8, -0.6, 0, 0.6, 0.8, 0, 0, 0, 1] },
    ]);
    const make = () => { const mock = mockGpu(); return { mock, r: new Renderer(mock.gpu, placementScene.world(), "", { pipeline: "compute" }) }; };
    const A = make(), B = make(), C = make(), D = make();
    const idsA = srcs.map(({ src, k }) => A.r.addModel(src, { scale: k }));
    const baker = new PlacementBaker();
    const idsB = srcs.map(({ src, k }) => {
      const e = encodeModel(src);
      const moved = structuredClone(e, { transfer: encodedTransferables(e, { keepPlacement: true }) }) as EncodedModel;
      const id = B.r.addEncodedModel(moved, { scale: k });
      const keys = B.r.placementModel(id)!;
      baker.register(placementModelOf(e, { key: keys.key, poseKey: keys.poseKey, scale: keys.scale }));
      return id;
    });
    const moving = list(idsA).slice(0, 4).map((p) => ({ ...p, x: p.x + 3.3, yaw: 1.1 }));
    A.r.setInstances(list(idsA));
    B.r.applyPlacement(baker.bake(structuredClone(B.r.placementInput(list(idsB)))));
    A.r.setInstances(moving, { dynamic: true });
    B.r.setInstances(moving, { dynamic: true });
    const sameBytes = (x: typeof A, y: typeof A) => {
      if (x.mock.buffers.length !== y.mock.buffers.length) return "buffer count";
      for (let i = 0; i < x.mock.buffers.length; i++) {
        const a = x.mock.buffers[i], b = y.mock.buffers[i];
        if (a.size !== b.size) return `buffer ${i}: ${a.size} vs ${b.size} bytes`;
        for (let j = 0; j < a.size; j++) if (a.bytes[j] !== b.bytes[j]) return `buffer ${i} byte ${j}`;
      }
      return "";
    };
    ok(A.r.placementModel(idsA[1])!.scale === 5 && A.r.placementModel(idsA[2])!.scale === undefined && !sameBytes(A, B), "scaled models added from a worker's encoding, baked on a worker, upload the same bytes as addModel + setInstances, under a moving set", sameBytes(A, B));
    const plain = srcs.map(({ src }) => C.r.addModel(src)), ones = srcs.map(({ src }) => D.r.addModel(src, { scale: 1 }));
    C.r.setInstances(list(plain));
    D.r.setInstances(list(ones));
    ok(!sameBytes(C, D), "  scale 1 given explicitly uploads the same bytes as no scale", sameBytes(C, D));
    const errs: string[] = [];
    for (const bad of [0, 1.5, -2]) { try { A.r.addModel(srcs[0].src, { scale: bad }); } catch (e) { errs.push(String(e)); } }
    const rigged = A.r.addModel({ size: { x: 4, y: 4, z: 4 }, data: new Uint8Array(64).fill(1), parts: new Uint8Array(64) }, { scale: 3 });
    try { A.r.setInstances([{ model: rigged, base: 1, x: 40, y: 40, z: 40, parts: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]) }], { dynamic: true }); } catch (e) { errs.push(String(e)); }
    A.r.setInstances([{ model: rigged, base: 1, x: 40, y: 40, z: 40 }], { dynamic: true });
    ok(errs.length === 4 && errs.slice(0, 3).every((e) => e.includes("integer >= 1")) && errs[3].includes("needs scale 1"), "  a scale that is not an integer from 1 throws, and so does posing a scaled model (drawn plain, it is fine)", errs.join(" | "));
  }
}
type SampleModelLike = { size: Size; voxel(x: number, y: number, z: number): number; part(x: number, y: number, z: number): number };

console.log("\npipeline warm-up (prepare):");

// prepare() must compile, asynchronously, every variant render() then picks, so frames after it
// make no pipeline synchronously; whatever the pipeline mode, canvas kind, instances, poses,
// scaled models, temporal accumulation and offscreen targets. Only a scene with a scaled model
// draws with SCALED compiled in.
{
  const frame = placementScene.frame();
  let bad = "";
  let cases = 0;
  for (const mode of ["fragment", "compute", "auto"] as const)
    for (const canvasStorage of [false, true])
      for (const temporal of [false, true])
      for (const deferPipelines of [false, true])
      for (const scaled of [false, true]) {
        cases++;
        const mock = mockGpu({ canvasStorage });
        const r = new Renderer(mock.gpu, placementScene.world(), "", { pipeline: mode, deferPipelines });
        if (deferPipelines && mock.pipelines.length && !bad) bad = `${mode}: deferPipelines still made ${mock.pipelines.length} in the constructor`;
        const ids = placementScene.addModels(r);
        const { staticSet, moving } = placementScene.lists(ids);
        if (scaled) {
          const big = r.addModel({ size: { x: 4, y: 6, z: 4 }, data: new Uint8Array(96).fill(1) }, { scale: 3 });
          staticSet.push({ model: big, base: 300, x: 100, y: 20, z: 100, yaw: 0.5 });
        }
        console.warn = () => {};
        r.setInstances(staticSet);
        console.warn = warn0;
        r.setInstances(moving, { dynamic: true });
        r.setQuality({ temporal });
        const p = r.prepare({ target: true });
        await mock.flush();
        await p;
        const made = mock.pipelines.length;
        const target = { view: {} as GPUTextureView, width: 32, height: 24 };
        // Posed and not, temporal and not (debug mode 1 turns it off), canvas and target.
        for (const posed of [true, false]) {
          r.setInstances(posed ? moving : moving.filter((m) => !m.parts), { dynamic: true });
          for (const debug of [0, 1]) {
            r.setDebug(debug);
            r.render(frame);
            r.render(frame, target);
          }
        }
        const scaledDraws = mock.drawn.filter((q) => q.constants?.[4] === 1).length;
        if ((scaled ? !scaledDraws : scaledDraws) && !bad) bad = `${mode}, scaled ${scaled}: ${scaledDraws} passes drawn with SCALED`;
        r.setInstances([], { dynamic: true });
        r.setInstances([]);
        r.render(frame);
        r.render(frame, target);
        const sync = mock.pipelines.slice(made);
        if (sync.length && !bad) bad = `${mode}, canvasStorage ${canvasStorage}, temporal ${temporal}, defer ${deferPipelines}, scaled ${scaled}: render made ${sync.map((q) => `${q.entry}${JSON.stringify(q.constants)}`).join(", ")}`;
        const again = mock.pipelines.length;
        await r.prepare({ target: true });
        if (mock.pipelines.length !== again && !bad) bad = `${mode}: a second prepare compiled again`;
      }
  ok(!bad, `after prepare, render makes no pipeline in ${cases} mode, canvas, temporal, deferPipelines and scaled combinations, and only scaled scenes draw with SCALED`, bad);

  // deferPipelines with a frame before prepare: the base pipelines are made synchronously, as today.
  {
    let fallback = "";
    for (const mode of ["fragment", "compute", "auto"] as const)
      for (const canvasStorage of [false, true]) {
        const mock = mockGpu({ canvasStorage });
        const r = new Renderer(mock.gpu, placementScene.world(), "", { pipeline: mode, deferPipelines: true });
        r.render(frame);
        r.render(frame, { view: {} as GPUTextureView, width: 32, height: 24 });
        const want = mode === "fragment" ? 1 : canvasStorage ? 3 : 2; // plain [+ canvas kernel] + frame kernel + present
        if ((mock.pipelines.length !== want || mock.pipelines.some((q) => q.async)) && !fallback) fallback = `${mode}, canvasStorage ${canvasStorage}: ${mock.pipelines.length} made`;
        const p = r.prepare({ target: true });
        await mock.flush();
        await p;
        if (mock.pipelines.length !== want && !fallback) fallback = `${mode}: prepare recompiled what render made`;
      }
    ok(!fallback, "  with deferPipelines, a frame before prepare makes the base pipelines synchronously, and prepare then skips them", fallback);
  }

  // Concurrent calls share compiles; a frame during a compile makes its own, and keeps it.
  const mock = mockGpu();
  const events: string[] = [];
  const r = new Renderer(mock.gpu, placementScene.world(), "", { pipeline: "compute", onLoad: (ph, kind, label) => { if (ph === "pipelines") events.push(`${kind} ${label}`); } });
  const ids = placementScene.addModels(r);
  console.warn = () => {};
  r.setInstances(placementScene.lists(ids).staticSet.filter((i) => !i.parts));
  console.warn = warn0;
  events.length = 0;
  const base = mock.pipelines.length;
  const p1 = r.prepare(), p2 = r.prepare();
  r.render(frame);
  const syncMade = mock.pipelines.length - base;
  await mock.flush();
  await Promise.all([p1, p2]);
  const asyncMade = mock.pipelines.length - base - syncMade;
  r.render(frame);
  const used = mock.drawn[mock.drawn.length - 1];
  ok(syncMade === 1 && asyncMade === 1 && !used.async, `two concurrent prepares compile once; a frame meanwhile compiles synchronously and keeps its pipeline (${syncMade} sync, ${asyncMade} async)`);
  ok(events.join(" | ") === "start compute+instances | start compute+instances | end compute+instances | end compute+instances", "  onLoad reports the async compile from call to ready, beside the synchronous one", events.join(" | "));
}

console.log(`\n${failed ? `${failed} of ${checks} checks FAILED` : `ALL ${checks} CHECKS PASSED`}`);
process.exit(failed ? 1 : 0);
