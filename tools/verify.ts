// Headless checks for the renderer's CPU-side data structures. No GPU.
//
//   bun run --cwd renderer verify

import { BrickGrid, BrickPool, BRICK_B, NEAR_BIT, PALETTE_ENTRIES } from "../src/brick";
import { makeSparse, sparseCount, sparseFromDense, sparseGet, sparseSet, sparseToDense } from "../src/sparse";
import { OccupancyGrid } from "../src/occupancy";
import { seededRandom } from "../src/random";
import { makePerf } from "../src/perf";

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

console.log(`\n${failed ? `${failed} of ${checks} checks FAILED` : `ALL ${checks} CHECKS PASSED`}`);
process.exit(failed ? 1 : 0);
