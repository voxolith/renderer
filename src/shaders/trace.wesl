// DDA voxel traversal (Amanatides & Woo) with coarse empty-space skipping. Walks
// from the box entry point to the first solid voxel, returning the hit cell,
// face normal, palette colour, and material index.

import package::uniforms::{U, palette};
import package::grid::{skipSize, brickOf, queryBrick, voxelIn, BrickQuery};
import package::dda::{stepMask, seedSideDist, skipBlock};
import package::materials::matOf;

struct Hit {
  hit    : bool,
  cell   : vec3<i32>,
  normal : vec3<f32>,
  color  : vec3<f32>,
  mat    : u32,
  /**
   * Distance along the ray to where it enters the hit cell (from the ray's origin), exact for the
   * cell's box. Shading still measures fog and glow to the face centre; this is for reprojection.
   */
  t      : f32,
  /** Loop iterations the walk took (fine steps plus skips), for the step heatmap. */
  steps  : i32,
};

// Hard ceiling for the primary-ray DDA. The effective cap is U.maxSteps (set via
// Renderer.setQuality), clamped to this; with coarse empty-space skipping on,
// typical rays use far fewer steps and the cap is only a backstop.
//
// It has to clear the worst case for the largest grid a consumer might hand us,
// because a ray that runs out of steps returns "no hit" and shades as sky — a
// hole in solid geometry. One axis is stepped at a time, so the worst case is
// the grid's Manhattan extent: 2736 for a 1280x176x1280 scene. Raising this
// costs nothing on its own; rays that terminate early still terminate early.
const MAX_STEPS : i32 = 4096;

fn traverse(ro : vec3<f32>, rd : vec3<f32>, tStart : f32) -> Hit {
  return walk(ro, rd, tStart, false);
}

// The same walk, but water voxels count as empty: what a ray that has entered
// water sees below the surface (the bed, a sunken rock, the far bank).
fn traverseUnderWater(ro : vec3<f32>, rd : vec3<f32>, tStart : f32) -> Hit {
  return walk(ro, rd, tStart, true);
}

// Where the ray enters `cell`: the last of its three slab entries (no earlier than tStart). An
// axis the ray runs parallel to never bounds the entry.
fn cellEntry(ro : vec3<f32>, rd : vec3<f32>, cell : vec3<i32>, tStart : f32) -> f32 {
  let plane = vec3<f32>(cell) + select(vec3<f32>(1.0), vec3<f32>(0.0), rd > vec3<f32>(0.0));
  let tp = select((plane - ro) / rd, vec3<f32>(-1e30), abs(rd) < vec3<f32>(1e-12));
  return max(max(max(tp.x, tp.y), tp.z), tStart);
}

fn walk(ro : vec3<f32>, rd : vec3<f32>, tStart : f32, skipWater : bool) -> Hit {
  var h : Hit;
  h.hit = false;

  let p = ro + rd * (tStart + 1e-3);
  var cell = vec3<i32>(floor(p));
  let stepDir = vec3<i32>(sign(rd));
  let deltaDist = abs(1.0 / rd);
  // t to the first voxel boundary on each axis.
  var sideDist = seedSideDist(cell, p, rd, deltaDist);
  var normal = vec3<f32>(0.0);

  let steps = min(i32(U.maxSteps), MAX_STEPS);
  // The current brick's lookup, reused while the ray stays in it.
  var qb = vec3<i32>(-2147483647);
  var q : BrickQuery;
  // The view distance (fog.distance), from the ray's origin: past it the fog is total. (Measured
  // from the cell, since a skip re-seeds sideDist from where it lands.)
  let reach = select(1e30, U.viewDistance, U.viewDistance > 0.0);
  for (var i = 0; i < steps; i = i + 1) {
    if (any(cell < vec3<i32>(-1)) || any(cell > vec3<i32>(U.gridSize))) {
      break;
    }
    if (dot(vec3<f32>(cell) + vec3<f32>(0.5) - ro, rd) > reach) {
      break;
    }
    let b = brickOf(cell);
    if (any(b != qb)) {
      qb = b;
      q = queryBrick(cell);
    }
    // Empty-space skip: if the brick (or its whole 64^3 block) is empty,
    // jump the ray to that block's far plane in one hop.
    if ((U.debug < 0.5 || U.debug > 9.5) && !q.solid) {
      let seed = skipBlock(ro, rd, cell, deltaDist, skipSize(cell));
      cell = seed.cell;
      sideDist = seed.sideDist;
      normal = seed.axis * -sign(rd);
      continue;
    }
    let c = voxelIn(q, cell);
    let water = skipWater && U.materialsEnabled > 0.5 && c != 0u && abs(matOf(c).kind - 4.0) < 0.5;
    if (c != 0u && !water) {
      h.hit = true;
      h.cell = cell;
      h.normal = normal;
      h.color = palette[c].rgb;
      h.mat = c;
      h.steps = i;
      h.t = cellEntry(ro, rd, cell, tStart);
      return h;
    }
    // Advance along the axis with the smallest accumulated distance.
    let mask = stepMask(sideDist);
    sideDist += mask * deltaDist;
    cell += vec3<i32>(mask) * stepDir;
    normal = mask * -sign(rd);
    h.steps = i;
  }
  return h;
}
