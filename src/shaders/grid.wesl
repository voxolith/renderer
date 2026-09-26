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
/** Words per instance record. Must match INST_WORDS in instance.ts. */
const INST_WORDS : u32 = 24u;
/** Words per part record, and of a pose's header. Must match PART_WORDS and POSE_HEADER in instance.ts. */
const PART_WORDS : u32 = 24u;
const POSE_HEADER : u32 = 8u;
const INST_MIRROR : u32 = 1u;
const INST_PARTS : u32 = 2u;
const NO_PARENT : u32 = 0xffffffffu;
/** Brick edge of an instance's part mask grid. Must match MASK_B in instance.ts. */
const MASK_B : i32 = 8;

// Pipeline constant: whether this pipeline samples instances at all. The
// renderer builds the instanced variant only once a scene places one, so a
// scene without instances runs a shader with that code compiled away (it
// costs even when never taken: several times the frame on SwiftShader).
@id(0) override INSTANCES : bool = false;
// Pipeline constant: whether any instance is drawn with parts (posed on the GPU). Its own variant,
// for the same reason: a scene of plain instances runs without the part sampling compiled in.
@id(1) override PARTS : bool = false;
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

/**
 * Brick entry, in a grid whose top level is at `topOff` with dims `td`, of
 * the brick containing voxel `v` (in range). Every lookup takes a voxel and
 * divides it here, unsigned: on Adreno 7xx, lookups handed brick coordinates
 * computed elsewhere (a floor, a signed divide) came back empty.
 */
fn gridEntry(topOff : u32, td : vec3<u32>, v : vec3<i32>) -> u32 {
  let b = toU(v) / 8u;
  let t = b / 8u;
  let blk = idx[U.topsOff + topOff + t.x + t.y * td.x + t.z * td.x * td.y];
  if (blk == 0u) {
    return 0u;
  }
  let l = b % 8u;
  return idx[U.blocksOff + (blk - 1u) * 512u + l.x + l.y * 8u + l.z * 64u];
}

/** World brick entry of the brick containing voxel `v` (in range), through the top-level texture. */
fn worldEntry(v : vec3<i32>) -> u32 {
  let b = toU(v) / 8u;
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

/** Apply the 3x4 affine stored at idx[o] (row-major, as f32 bits) to a point. */
fn affineAt(o : u32, p : vec3<f32>) -> vec3<f32> {
  let r0 = vec4<f32>(bitcast<f32>(idx[o]), bitcast<f32>(idx[o + 1u]), bitcast<f32>(idx[o + 2u]), bitcast<f32>(idx[o + 3u]));
  let r1 = vec4<f32>(bitcast<f32>(idx[o + 4u]), bitcast<f32>(idx[o + 5u]), bitcast<f32>(idx[o + 6u]), bitcast<f32>(idx[o + 7u]));
  let r2 = vec4<f32>(bitcast<f32>(idx[o + 8u]), bitcast<f32>(idx[o + 9u]), bitcast<f32>(idx[o + 10u]), bitcast<f32>(idx[o + 11u]));
  let h = vec4<f32>(p, 1.0);
  return vec3<f32>(dot(r0, h), dot(r1, h), dot(r2, h));
}

/**
 * A world point in instance `k`'s model space (voxels): its world→model affine, which turns about
 * the anchor voxel's centre, so a quarter turn maps voxel centres onto voxel centres exactly (as a
 * stamped orientation does). See placement() in instance.ts.
 */
fn toModel(k : u32, wp : vec3<f32>) -> vec3<f32> {
  return affineAt(U.instOff + k * INST_WORDS, wp);
}

/** Value at model cell `q` (in range) of the grid whose record starts at `mi`, `part` selecting its part grid. */
fn modelValue(mi : u32, q : vec3<i32>, part : bool) -> u32 {
  var top = idx[mi];
  if (part) {
    top = idx[mi + 7u] - 1u;
  }
  let e = gridEntry(top, vec3<u32>(idx[mi + 1u], idx[mi + 2u], idx[mi + 3u]), q);
  return brickVoxel(e, toI(toU(q) % 8u));
}

/**
 * Role value an instance drawn with parts has at world cell `p`, or 0. Each part maps the cell
 * centre back to the shared rest model through its own inverse; the first part whose own voxel is
 * there wins (parts are ordered parents first). Near a joint, either side's voxels may fill the
 * cell, so a turned child stays on its parent.
 *
 * Posing by moving samples into a shared, static rest pose follows Gruen, Benthin, Kern and
 * McAllister, "Ray Tracing Massive Amounts of Animated Geometry" (HPG 2026,
 * doi:10.1145/3820014), and Kao, Makowski, Fujieda and Harada, "Voxel Deformation-Aware Neural
 * Intersection Function" (EG 2026, doi:10.2312/egs.20261026); here the deformation is rigid per
 * part, over the brick index. sampleInstance() in instance.ts is the CPU twin.
 */
fn partsVoxel(o : u32, mi : u32, wp : vec3<f32>) -> u32 {
  let size = vec3<i32>(i32(idx[mi + 4u]), i32(idx[mi + 5u]), i32(idx[mi + 6u]));
  // Into the posed model's space: the pose (part inverses, joints, masks) is packed there once and
  // shared by every instance showing it.
  let u = affineAt(o, wp);
  let uc = vec3<i32>(floor(u));
  let off = U.partsOff + idx[o + 15u];
  let lo = vec3<i32>(bitcast<i32>(idx[off]), bitcast<i32>(idx[off + 1u]), bitcast<i32>(idx[off + 2u]));
  let dims = vec3<i32>(i32(idx[off + 3u]), i32(idx[off + 4u]), i32(idx[off + 5u]));
  let n = idx[o + 16u];
  // Which parts can be here: the brick's pair of masks (one bit per part, at most 32). The first
  // says whose posed box touches the brick, the second whose joint weld does; a bit is set only for
  // parts that have voxels (and, for welds, whose parent has too).
  let d = uc - lo;
  if (any(d < vec3<i32>(0)) || any(d >= dims * MASK_B)) {
    return 0u;
  }
  let bc = toU(d) / u32(MASK_B);
  let pd = toU(dims);
  let P = off + POSE_HEADER;
  let mb = P + n * PART_WORDS + (bc.x + bc.y * pd.x + bc.z * pd.x * pd.y) * 2u;
  var hits = idx[mb];
  var welds = idx[mb + 1u];
  // One loop over the candidates, so the lookups are compiled once: first each part whose box is
  // here, through its own inverse, taking only its own voxels (parents first, as baking does);
  // then each joint weld here, through the child's inverse and then the parent's, taking either's.
  var pendB = 0u;
  var pendPar = NO_PARENT;
  loop {
    var b = 0u;
    var via = 0u;
    var par = NO_PARENT;
    if (hits != 0u) {
      b = firstTrailingBit(hits);
      hits = hits & (hits - 1u);
      via = b;
    } else if (pendPar != NO_PARENT) {
      b = pendB;
      par = pendPar;
      via = par;
      pendPar = NO_PARENT;
    } else {
      if (welds == 0u) {
        break;
      }
      b = firstTrailingBit(welds);
      welds = welds & (welds - 1u);
      let q0 = P + b * PART_WORDS;
      let j = vec3<i32>(bitcast<i32>(idx[q0 + 18u]), bitcast<i32>(idx[q0 + 19u]), bitcast<i32>(idx[q0 + 20u]));
      if (any(abs(uc - j) > vec3<i32>(1))) {
        continue;
      }
      par = idx[q0 + 21u];
      via = b;
      pendB = b;
      pendPar = par;
    }
    let q = vec3<i32>(floor(affineAt(P + via * PART_WORDS, u)));
    if (any(q < vec3<i32>(0)) || any(q >= size)) {
      continue;
    }
    let v = modelValue(mi, q, false);
    if (v == 0u) {
      continue;
    }
    let pid = modelValue(mi, q, true);
    if (pid == b + 1u || (par != NO_PARENT && pid == par + 1u)) {
      return v;
    }
  }
  return 0u;
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
    // The world cells the instance can touch: most samples in a listed cell miss it here.
    let lo = vec3<i32>(bitcast<i32>(idx[o + 17u]), bitcast<i32>(idx[o + 18u]), bitcast<i32>(idx[o + 19u]));
    let hi = vec3<i32>(bitcast<i32>(idx[o + 20u]), bitcast<i32>(idx[o + 21u]), bitcast<i32>(idx[o + 22u]));
    if (any(p < lo) || any(p >= hi)) {
      continue;
    }
    let mi = U.modelsOff + idx[o + 12u] * MODEL_WORDS;
    let flags = idx[o + 14u];
    if (PARTS && (flags & INST_PARTS) != 0u) {
      let v = partsVoxel(o, mi, wp);
      if (v != 0u) {
        return idx[o + 13u] + v - 1u;
      }
      continue;
    }
    let size = vec3<i32>(i32(idx[mi + 4u]), i32(idx[mi + 5u]), i32(idx[mi + 6u]));
    var q = vec3<i32>(floor(toModel(k, wp)));
    // Mirrored instances flip the voxel index along x, after the turn.
    if ((flags & INST_MIRROR) != 0u) {
      q.x = size.x - 1 - q.x;
    }
    if (any(q < vec3<i32>(0)) || any(q >= size)) {
      continue;
    }
    let v = modelValue(mi, q, false);
    if (v != 0u) {
      return idx[o + 13u] + v - 1u;
    }
  }
  return 0u;
}

fn sampleVoxel(p : vec3<i32>) -> u32 {
  if (any(p < vec3<i32>(0)) || any(p >= vec3<i32>(U.gridSize))) {
    return 0u;
  }
  let pu = toU(p);
  let e = worldEntry(p);
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

/** The world's own voxel at `p`, ignoring instances (the device self-test). */
fn worldVoxel(p : vec3<i32>) -> u32 {
  if (any(p < vec3<i32>(0)) || any(p >= vec3<i32>(U.gridSize))) {
    return 0u;
  }
  let e = worldEntry(p);
  if (e == 0u) {
    return 0u;
  }
  return brickVoxel(e, toI(toU(p) % 8u));
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
  let blo = c * COARSE_B;
  let bhi = blo + vec3<i32>(COARSE_B);
  for (var i = 0u; i < n; i = i + 1u) {
    let k = idx[U.listOff + start + i];
    let o = U.instOff + k * INST_WORDS;
    let ilo = vec3<i32>(bitcast<i32>(idx[o + 17u]), bitcast<i32>(idx[o + 18u]), bitcast<i32>(idx[o + 19u]));
    let ihi = vec3<i32>(bitcast<i32>(idx[o + 20u]), bitcast<i32>(idx[o + 21u]), bitcast<i32>(idx[o + 22u]));
    // A brick more than one brick away from the instance's cells cannot hold its voxels (the
    // NEAR clamp below reaches one brick past a model's edge).
    if (any(blo >= ihi + vec3<i32>(COARSE_B)) || any(bhi <= ilo - vec3<i32>(COARSE_B))) {
      continue;
    }
    let mi = U.modelsOff + idx[o + 12u] * MODEL_WORDS;
    let flags = idx[o + 14u];
    if (PARTS && (flags & INST_PARTS) != 0u) {
      // Posed: the world brick, taken into the posed model's space, may hold voxels where a mask
      // brick it reaches has a part or a weld in it.
      let off = U.partsOff + idx[o + 15u];
      let lo = vec3<i32>(bitcast<i32>(idx[off]), bitcast<i32>(idx[off + 1u]), bitcast<i32>(idx[off + 2u]));
      let dims = vec3<i32>(i32(idx[off + 3u]), i32(idx[off + 4u]), i32(idx[off + 5u]));
      let uc = toModel(k, wc);
      // Half the brick's extent along each posed axis: 4 × the row's absolute sum (4 for an axis the
      // turn leaves alone, up to 5.7 under a yaw), so the reach is exact for any rotation.
      let io = U.instOff + k * INST_WORDS;
      let ext = 4.0 * vec3<f32>(
        abs(bitcast<f32>(idx[io])) + abs(bitcast<f32>(idx[io + 1u])) + abs(bitcast<f32>(idx[io + 2u])),
        abs(bitcast<f32>(idx[io + 4u])) + abs(bitcast<f32>(idx[io + 5u])) + abs(bitcast<f32>(idx[io + 6u])),
        abs(bitcast<f32>(idx[io + 8u])) + abs(bitcast<f32>(idx[io + 9u])) + abs(bitcast<f32>(idx[io + 10u])),
      );
      let r0 = vec3<i32>(floor(uc - ext)) - lo;
      let r1 = vec3<i32>(floor(uc + ext)) - lo;
      if (any(r1 < vec3<i32>(0)) || any(r0 >= dims * MASK_B)) {
        continue;
      }
      let c0 = clamp(r0 / MASK_B, vec3<i32>(0), dims - vec3<i32>(1));
      let c1 = clamp(r1 / MASK_B, vec3<i32>(0), dims - vec3<i32>(1));
      let mbase = off + POSE_HEADER + idx[o + 16u] * PART_WORDS;
      for (var z = c0.z; z <= c1.z; z = z + 1) {
        for (var y = c0.y; y <= c1.y; y = y + 1) {
          for (var x = c0.x; x <= c1.x; x = x + 1) {
            let bi = mbase + u32(x + y * dims.x + z * dims.x * dims.y) * 2u;
            if ((idx[bi] | idx[bi + 1u]) != 0u) {
              return true;
            }
          }
        }
      }
      continue;
    }
    let size = vec3<i32>(i32(idx[mi + 4u]), i32(idx[mi + 5u]), i32(idx[mi + 6u]));
    let bdim = (size + vec3<i32>(COARSE_B - 1)) / COARSE_B;
    var qm = toModel(k, wc);
    if ((flags & INST_MIRROR) != 0u) {
      qm.x = f32(size.x) - qm.x;
    }
    let qb = vec3<i32>(floor(qm / f32(COARSE_B)));
    // One brick outside the model can still overlap its edge bricks, whose
    // NEAR flags cover their own neighbourhood: clamp onto them.
    if (any(qb < vec3<i32>(-1)) || any(qb > bdim)) {
      continue;
    }
    let cb = clamp(qb, vec3<i32>(0), bdim - vec3<i32>(1));
    if (gridEntry(idx[mi], vec3<u32>(idx[mi + 1u], idx[mi + 2u], idx[mi + 3u]), cb * COARSE_B) != 0u) {
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
  if (worldEntry(p) != 0u) {
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

/**
 * Look up the brick containing fine cell `p`. Takes the cell, not the brick:
 * on Adreno 7xx a lookup fed with brick coordinates computed first (a floor
 * or a signed divide) came back empty, while one derived from the voxel's
 * own coordinates, as sampleVoxel does, works.
 */
fn queryBrick(p : vec3<i32>) -> BrickQuery {
  let c = brickOf(p);
  var q : BrickQuery;
  q.inGrid = all(c >= vec3<i32>(0)) && all(c < vec3<i32>(U.coarseDim));
  if (!q.inGrid) {
    // Out of range: never skip; the walk's own bounds test ends the ray.
    q.entry = 0u;
    q.near = false;
    q.solid = true;
    return q;
  }
  q.entry = worldEntry(p);
  q.near = q.entry == 0u && (INSTANCES && U.instanceCount != 0u) && instanceNear(c);
  q.solid = q.entry != 0u || q.near;
  return q;
}

/** The voxel at fine cell `p`, which lies in the brick `q` describes. */
fn voxelIn(q : BrickQuery, p : vec3<i32>) -> u32 {
  if (!q.inGrid || any(p < vec3<i32>(0)) || any(p >= vec3<i32>(U.gridSize))) {
    return 0u;
  }
  // One call site for the instance lookup (it is large): a world voxel wins, else instances where
  // the world is empty but the brick is solid or near one.
  var inst = q.near;
  if (q.entry != 0u) {
    let v = brickVoxel(q.entry, toI(toU(p) % 8u));
    if (v != 0u || (!INSTANCES || U.instanceCount == 0u)) {
      return v;
    }
    inst = true;
  }
  if (inst) {
    return instanceVoxel(p);
  }
  return 0u;
}
