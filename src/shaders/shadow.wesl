// Coarse-accelerated shadow ray: true if any voxel blocks the straight path to
// the light. Skips provably-empty coarse blocks but stays fine-exact.

import package::uniforms::U;
import package::grid::{skipSize, brickOf, queryBrick, voxelIn, BrickQuery};
import package::dda::{stepMask, seedSideDist, skipBlock};

// Hard ceiling; the effective cap is U.shadowSteps (0 disables shadows upstream).
const SHADOW_STEPS : i32 = 512;

fn inShadow(origin : vec3<f32>, rd : vec3<f32>) -> bool {
  var cell = vec3<i32>(floor(origin));
  let stepDir = vec3<i32>(sign(rd));
  let deltaDist = abs(1.0 / rd);
  var sideDist = seedSideDist(cell, origin, rd, deltaDist);

  let steps = min(i32(U.shadowSteps), SHADOW_STEPS);
  var qb = vec3<i32>(-2147483647);
  var q : BrickQuery;
  for (var i = 0; i < steps; i = i + 1) {
    if (any(cell < vec3<i32>(0)) || any(cell >= vec3<i32>(U.gridSize))) {
      return false;
    }
    let b = brickOf(cell);
    if (any(b != qb)) { qb = b; q = queryBrick(cell); }
    if (U.debug < 0.5 && !q.solid) {
      let seed = skipBlock(origin, rd, cell, deltaDist, skipSize(cell));
      cell = seed.cell;
      sideDist = seed.sideDist;
      continue;
    }
    let mask = stepMask(sideDist);
    sideDist += mask * deltaDist;
    cell += vec3<i32>(mask) * stepDir;
    let nb = brickOf(cell);
    if (any(nb != qb)) { qb = nb; q = queryBrick(cell); }
    if (voxelIn(q, cell) != 0u) {
      return true;
    }
  }
  return false;
}

// Shadow ray towards a point light: true if a voxel blocks the path before
// `maxT` (the distance to the light). The same coarse-skipping walk as
// inShadow; it stops as soon as the cell it is in lies beyond the light, so a
// lamp above the ground is not shadowed by the ground far behind it.
fn occludedTo(origin : vec3<f32>, rd : vec3<f32>, maxT : f32) -> bool {
  var cell = vec3<i32>(floor(origin));
  let stepDir = vec3<i32>(sign(rd));
  let deltaDist = abs(1.0 / rd);
  var sideDist = seedSideDist(cell, origin, rd, deltaDist);
  let reach = maxT - 0.87; // stop half a voxel diagonal short of the light

  let steps = min(i32(U.shadowSteps), SHADOW_STEPS);
  var qb = vec3<i32>(-2147483647);
  var q : BrickQuery;
  for (var i = 0; i < steps; i = i + 1) {
    if (any(cell < vec3<i32>(0)) || any(cell >= vec3<i32>(U.gridSize))) {
      return false;
    }
    if (length(vec3<f32>(cell) + vec3<f32>(0.5) - origin) > reach) {
      return false;
    }
    let b = brickOf(cell);
    if (any(b != qb)) { qb = b; q = queryBrick(cell); }
    if (U.debug < 0.5 && !q.solid) {
      let seed = skipBlock(origin, rd, cell, deltaDist, skipSize(cell));
      cell = seed.cell;
      sideDist = seed.sideDist;
      continue;
    }
    let mask = stepMask(sideDist);
    sideDist += mask * deltaDist;
    cell += vec3<i32>(mask) * stepDir;
    if (length(vec3<f32>(cell) + vec3<f32>(0.5) - origin) > reach) {
      return false;
    }
    let nb = brickOf(cell);
    if (any(nb != qb)) { qb = nb; q = queryBrick(cell); }
    if (voxelIn(q, cell) != 0u) {
      return true;
    }
  }
  return false;
}
