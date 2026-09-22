// Headless checks for the renderer's CPU-side data structures. No GPU.
//
//   bun run --cwd renderer verify

import { BrickGrid, BRICK_B, PALETTE_ENTRIES } from "../src/brick";
import { OccupancyGrid } from "../src/occupancy";
import { seededRandom } from "../src/random";

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
    bricks.rebuildBox(data, box);
  }
  ok(bricks.slotCount4 === 1, "clearing and refilling reuses one slot", `slots=${bricks.slotCount4}`);
  ok(bricks.get(1, 2, 3) === 3 + 19 % 4, "  value after reuse is correct");
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

console.log(`\n${failed ? `${failed} of ${checks} checks FAILED` : `ALL ${checks} CHECKS PASSED`}`);
process.exit(failed ? 1 : 0);
