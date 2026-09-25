// Voxel sampling through the sparse two-level index, plus instanced models.
//
// A voxel read is: top level → index block → brick entry → 4-bit local index →
// the brick's own palette → a global palette slot. A uniform brick answers at
// the entry. See src/brick.ts for the layout and why the palette is per brick.
//
// Instances are models drawn by reference (renderer.addModel/setInstances).
// Each world top cell lists the instances whose box touches it; a voxel the
// world leaves empty is looked up in each of them, the world point turned
// into the model's space. That is resampling at world voxel centres, so a
// turned instance still reads as axis-aligned cubes, like a baked pose.
//
// Null entries are the empty-space skip at two sizes: a null top cell with no
// instances skips 64 voxels, a null brick 8 (for instances, only where the
// model's brick neighbourhood is empty too: NEAR_BIT).

import package::uniforms::{U, idx, brickVox4, brickPal, brickVox8, topTex};

/** Brick edge in voxels. Must match BRICK_B in brick.ts. */
const COARSE_B : i32 = 8;
/** Top-level cell edge in voxels. Must match TOP_B in brick.ts. */
const TOP_B : i32 = 64;

const TIER_BIT : u32 = 0x80000000u;
const UNIFORM_BIT : u32 = 0x40000000u;
const SLOT_MASK : u32 = 0x1fffffffu;
const INST_WORDS : u32 = 12u;

// Pipeline constant: whether this pipeline samples instances at all. The
// renderer builds the instanced variant only once a scene places one, so a
// scene without instances runs a shader with that code compiled away (it
// costs even when never taken: several times the frame on SwiftShader).
@id(0) override INSTANCES : bool = false;
const MODEL_WORDS : u32 = 8u;

// Index arithmetic is all unsigned: coordinates are converted once, after
// the bounds tests. Signed shifts and masks (b >> 3, b & 7 on vec3<i32>)
// read wrong on Adreno 7xx, where every voxel came back empty.

// Signed <-> unsigned vector conversions are done a component at a time:
// on Adreno 7xx, vec3<u32>(v) of a computed vec3<i32> read wrong (brick
// lookups whose coordinates came from the ray walk all came back empty),
// while scalar conversions were fine.
fn toU(v : vec3<i32>) -> vec3<u32> {
  return vec3<u32>(u32(v.x), u32(v.y), u32(v.z));
}
fn toI(v : vec3<u32>) -> vec3<i32> {
  return vec3<i32>(i32(v.x), i32(v.y), i32(v.z));
}

/** Brick entry of a grid whose top level is at `topOff` with dims `td`; `b` in range. */
fn gridEntry(topOff : u32, td : vec3<u32>, bi : vec3<i32>) -> u32 {
  let b = toU(bi);
  let t = b / 8u;
  let blk = idx[U.topsOff + topOff + t.x + t.y * td.x + t.z * td.x * td.y];
  if (blk == 0u) {
    return 0u;
  }
  let l = b % 8u;
  return idx[U.blocksOff + (blk - 1u) * 512u + l.x + l.y * 8u + l.z * 64u];
}

/** World brick entry for brick `b` (in range), through the top-level texture. */
fn worldEntry(bi : vec3<i32>) -> u32 {
  let b = toU(bi);
  let blk = textureLoad(topTex, b / 8u, 0).r;
  if (blk == 0u) {
    return 0u;
  }
  let l = b % 8u;
  return idx[U.blocksOff + (blk - 1u) * 512u + l.x + l.y * 8u + l.z * 64u];
}

/** Voxel at local cell `l` (0..7) of a brick with entry `e`. */
fn brickVoxel(e : u32, li3 : vec3<i32>) -> u32 {
  let l = toU(li3);
  if ((e & UNIFORM_BIT) != 0u) {
    return e & 0xffffu;
  }
  if ((e & SLOT_MASK) == 0u) {
    return 0u;
  }
  let slot = (e & SLOT_MASK) - 1u;
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

/** The instance list of the world top cell containing `p` (in range): count in the low 8 bits. */
fn cellList(p : vec3<i32>) -> u32 {
  let t = toU(p) / u32(TOP_B);
  return idx[U.cellOff + t.x + t.y * U.topDim.x + t.z * U.topDim.x * U.topDim.y];
}

/** A world point in instance `k`'s model space (voxels). */
fn toModel(k : u32, wp : vec3<f32>) -> vec3<f32> {
  let o = U.instOff + k * INST_WORDS;
  let pos = vec3<f32>(bitcast<f32>(idx[o]), bitcast<f32>(idx[o + 1u]), bitcast<f32>(idx[o + 2u]));
  let c = bitcast<f32>(idx[o + 3u]);
  let anchor = vec3<f32>(bitcast<f32>(idx[o + 4u]), bitcast<f32>(idx[o + 5u]), bitcast<f32>(idx[o + 6u]));
  let s = bitcast<f32>(idx[o + 7u]);
  // The turn pivots on the anchor voxel's centre, so a quarter turn maps
  // voxel centres onto voxel centres exactly (as a stamped orientation does).
  let d = wp - pos - vec3<f32>(0.5, 0.0, 0.5);
  // Inverse of world = R (m - anchor) + pos, R = [c 0 s; 0 1 0; -s 0 c].
  return vec3<f32>(c * d.x - s * d.z, d.y, s * d.x + c * d.z) + anchor + vec3<f32>(0.5, 0.0, 0.5);
}

/** Palette slot of the instanced voxel at world cell `p`, or 0. */
fn instanceVoxel(p : vec3<i32>) -> u32 {
  if (!INSTANCES) {
    return 0u;
  }
  let cl = cellList(p);
  let n = cl & 0xffu;
  if (n == 0u) {
    return 0u;
  }
  let start = cl >> 8u;
  let wp = vec3<f32>(p) + vec3<f32>(0.5);
  for (var i = 0u; i < n; i = i + 1u) {
    let k = idx[U.listOff + start + i];
    let o = U.instOff + k * INST_WORDS;
    let mi = U.modelsOff + idx[o + 8u] * MODEL_WORDS;
    let size = vec3<i32>(i32(idx[mi + 4u]), i32(idx[mi + 5u]), i32(idx[mi + 6u]));
    var q = vec3<i32>(floor(toModel(k, wp)));
    // Mirrored instances flip the voxel index along x, after the turn.
    if ((idx[o + 10u] & 1u) != 0u) {
      q.x = size.x - 1 - q.x;
    }
    if (any(q < vec3<i32>(0)) || any(q >= size)) {
      continue;
    }
    let qu = toU(q);
    let e = gridEntry(idx[mi], vec3<u32>(idx[mi + 1u], idx[mi + 2u], idx[mi + 3u]), toI(qu / 8u));
    let v = brickVoxel(e, toI(qu % 8u));
    if (v != 0u) {
      return idx[o + 9u] + v - 1u;
    }
  }
  return 0u;
}

fn sampleVoxel(p : vec3<i32>) -> u32 {
  if (any(p < vec3<i32>(0)) || any(p >= vec3<i32>(U.gridSize))) {
    return 0u;
  }
  let pu = toU(p);
  let e = worldEntry(toI(pu / 8u));
  if (e != 0u) {
    let v = brickVoxel(e, toI(pu % 8u));
    if (v != 0u || (!INSTANCES || U.instanceCount == 0u)) {
      return v;
    }
  }
  if ((!INSTANCES || U.instanceCount == 0u)) {
    return 0u;
  }
  return instanceVoxel(p);
}

fn isSolid(p : vec3<i32>) -> bool {
  return sampleVoxel(p) != 0u;
}

/** May an instance have voxels in the world brick `c` (brick coords, in range)? */
fn instanceNear(c : vec3<i32>) -> bool {
  if (!INSTANCES) {
    return false;
  }
  let cl = cellList(c * COARSE_B);
  let n = cl & 0xffu;
  if (n == 0u) {
    return false;
  }
  let start = cl >> 8u;
  let wc = vec3<f32>(c * COARSE_B) + vec3<f32>(4.0);
  for (var i = 0u; i < n; i = i + 1u) {
    let k = idx[U.listOff + start + i];
    let o = U.instOff + k * INST_WORDS;
    let mi = U.modelsOff + idx[o + 8u] * MODEL_WORDS;
    let size = vec3<i32>(i32(idx[mi + 4u]), i32(idx[mi + 5u]), i32(idx[mi + 6u]));
    let bdim = (size + vec3<i32>(COARSE_B - 1)) / COARSE_B;
    var qm = toModel(k, wc);
    if ((idx[o + 10u] & 1u) != 0u) {
      qm.x = f32(size.x) - qm.x;
    }
    let qb = vec3<i32>(floor(qm / f32(COARSE_B)));
    // One brick outside the model can still overlap its edge bricks, whose
    // NEAR flags cover their own neighbourhood: clamp onto them.
    if (any(qb < vec3<i32>(-1)) || any(qb > bdim)) {
      continue;
    }
    let cb = clamp(qb, vec3<i32>(0), bdim - vec3<i32>(1));
    if (gridEntry(idx[mi], vec3<u32>(idx[mi + 1u], idx[mi + 2u], idx[mi + 3u]), cb) != 0u) {
      return true;
    }
  }
  return false;
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
  if (worldEntry(c) != 0u) {
    return true;
  }
  return (INSTANCES && U.instanceCount != 0u) && instanceNear(c);
}

// How far an empty brick at `p` lets a ray skip: its whole 64^3 top cell when
// that holds nothing at all (no world block, no instances), else the brick.
// Call only where coarseSolid(p) is false, so `p` is in range.
fn skipSize(p : vec3<i32>) -> i32 {
  if (textureLoad(topTex, toU(p) / u32(TOP_B), 0).r != 0u) {
    return COARSE_B;
  }
  if ((INSTANCES && U.instanceCount != 0u) && (cellList(p) & 0xffu) != 0u) {
    return COARSE_B;
  }
  return TOP_B;
}

// --- per-brick queries for ray walks --------------------------------------------
//
// A ray takes several fine steps inside each brick. Looking the brick up
// once when it enters (world entry, and whether an instance may be there)
// and reusing that for every step inside is what keeps the two-level index
// as cheap to walk as a flat one.

struct BrickQuery {
  /** World brick entry (0 when empty or out of range). */
  entry : u32,
  /** False when a ray may skip this brick: nothing of the world or of an instance there. */
  solid : bool,
  /** An instance may have voxels here. */
  near  : bool,
  /** The brick lies inside the grid. */
  inGrid : bool,
};

/** Brick coordinates of a fine cell, flooring (cells may be one outside the grid). */
fn brickOf(p : vec3<i32>) -> vec3<i32> {
  return vec3<i32>(floor(vec3<f32>(p) / f32(COARSE_B)));
}

fn queryBrick(c : vec3<i32>) -> BrickQuery {
  var q : BrickQuery;
  q.inGrid = all(c >= vec3<i32>(0)) && all(c < vec3<i32>(U.coarseDim));
  if (!q.inGrid) {
    // Out of range: never skip; the walk's own bounds test ends the ray.
    q.entry = 0u;
    q.near = false;
    q.solid = true;
    return q;
  }
  q.entry = worldEntry(c);
  q.near = q.entry == 0u && (INSTANCES && U.instanceCount != 0u) && instanceNear(c);
  q.solid = q.entry != 0u || q.near;
  return q;
}

/** The voxel at fine cell `p`, which lies in the brick `q` describes. */
fn voxelIn(q : BrickQuery, p : vec3<i32>) -> u32 {
  if (!q.inGrid || any(p < vec3<i32>(0)) || any(p >= vec3<i32>(U.gridSize))) {
    return 0u;
  }
  if (q.entry != 0u) {
    let v = brickVoxel(q.entry, toI(toU(p) % 8u));
    if (v != 0u || (!INSTANCES || U.instanceCount == 0u)) {
      return v;
    }
    return instanceVoxel(p);
  }
  if (q.near) {
    return instanceVoxel(p);
  }
  return 0u;
}
