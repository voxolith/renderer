// Headless checks for the renderer's CPU-side data structures. No GPU.
//
//   bun run --cwd renderer verify

import { BrickGrid, BrickPool, BRICK_B, NEAR_BIT, PALETTE_ENTRIES } from "../src/brick";
import { makeSparse, sparseCount, sparseFromDense, sparseGet, sparseSet, sparseToDense } from "../src/sparse";
import { OccupancyGrid } from "../src/occupancy";
import { seededRandom } from "../src/random";
import { makePerf } from "../src/perf";
import { buildSubLists, readSubLists, type SubListInstance } from "../src/sublists";
import { INST_WORDS, packInstance, sampleInstance } from "../src/instance";
import { PlacementBaker, placementTransferables } from "../src/placement";
import { installRawLoader, mockGpu } from "./mock-gpu";

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

console.log("\npipeline warm-up (prepare):");

// prepare() must compile, asynchronously, every variant render() then picks, so frames after it
// make no pipeline synchronously; whatever the pipeline mode, canvas kind, instances, poses,
// temporal accumulation and offscreen targets.
{
  const frame = placementScene.frame();
  let bad = "";
  let cases = 0;
  for (const mode of ["fragment", "compute", "auto"] as const)
    for (const canvasStorage of [false, true])
      for (const temporal of [false, true])
      for (const deferPipelines of [false, true]) {
        cases++;
        const mock = mockGpu({ canvasStorage });
        const r = new Renderer(mock.gpu, placementScene.world(), "", { pipeline: mode, deferPipelines });
        if (deferPipelines && mock.pipelines.length && !bad) bad = `${mode}: deferPipelines still made ${mock.pipelines.length} in the constructor`;
        const ids = placementScene.addModels(r);
        const { staticSet, moving } = placementScene.lists(ids);
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
        r.setInstances([], { dynamic: true });
        r.setInstances([]);
        r.render(frame);
        r.render(frame, target);
        const sync = mock.pipelines.slice(made);
        if (sync.length && !bad) bad = `${mode}, canvasStorage ${canvasStorage}, temporal ${temporal}, defer ${deferPipelines}: render made ${sync.map((q) => `${q.entry}${JSON.stringify(q.constants)}`).join(", ")}`;
        const again = mock.pipelines.length;
        await r.prepare({ target: true });
        if (mock.pipelines.length !== again && !bad) bad = `${mode}: a second prepare compiled again`;
      }
  ok(!bad, `after prepare, render makes no pipeline in ${cases} mode, canvas, temporal and deferPipelines combinations`, bad);

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
