// Voxel sampling through the sparse brick index.
//
// A voxel read is: index volume → brick slot → 4-bit local index → the brick's
// own palette → a global palette slot. The extra indirection costs two storage
// reads and buys a ~6x smaller world; see src/brick.ts for the layout and why
// the palette is per brick.
//
// The index volume also replaces the old coarse occupancy texture: a null entry
// means the whole brick is empty, which is exactly the empty-space-skip test.

import package::uniforms::{U, indexTex, brickVox4, brickPal, brickVox8};

/** Brick edge in voxels. Must match BRICK_B in brick.ts. */
const COARSE_B : i32 = 8;

const TIER_BIT : u32 = 0x80000000u;
const SLOT_MASK : u32 = 0x7fffffffu;

/** Raw index entry for the brick containing fine cell `p` (assumes p in range). */
fn brickEntry(p : vec3<i32>) -> u32 {
  return textureLoad(indexTex, p / COARSE_B, 0).r;
}

fn sampleVoxel(p : vec3<i32>) -> u32 {
  if (any(p < vec3<i32>(0)) || any(p >= vec3<i32>(U.gridSize))) {
    return 0u;
  }
  let e = brickEntry(p);
  if (e == 0u) {
    return 0u;
  }
  let slot = (e & SLOT_MASK) - 1u;
  let l = p % COARSE_B;
  let li = u32(l.x) + u32(l.y) * 8u + u32(l.z) * 64u;

  if ((e & TIER_BIT) != 0u) {
    let w = brickVox8[slot * 128u + (li >> 2u)];
    return (w >> ((li & 3u) * 8u)) & 0xffu;
  }

  let w = brickVox4[slot * 64u + (li >> 3u)];
  let nib = (w >> ((li & 7u) * 4u)) & 0xfu;
  if (nib == 0u) {
    return 0u;  // nibble 0 is reserved for empty, leaving 15 palette entries
  }
  let pw = brickPal[slot * 8u + ((nib - 1u) >> 1u)];
  return (pw >> (((nib - 1u) & 1u) * 16u)) & 0xffffu;
}

fn isSolid(p : vec3<i32>) -> bool {
  return sampleVoxel(p) != 0u;
}

// Does the brick containing fine cell `p` hold anything? Out-of-range → treat as
// solid (don't fast-forward; let the fine loop's bounds check exit). `p` may be
// one cell outside the grid, so this floors rather than truncating: integer
// division rounds toward zero and would map -1 into brick 0.
fn coarseSolid(p : vec3<i32>) -> bool {
  let c = vec3<i32>(floor(vec3<f32>(p) / f32(COARSE_B)));
  if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(U.coarseDim))) {
    return true;
  }
  return textureLoad(indexTex, c, 0).r != 0u;
}
