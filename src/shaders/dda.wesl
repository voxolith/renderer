// Shared DDA stepping for the primary and shadow rays.
//
// Both routines exist to keep a ray from cutting a voxel corner. A ray that
// moves diagonally slips between two face-adjacent solid voxels and carries on
// through the scene, which renders as a thin streak of background across solid
// geometry — most visible when the camera yaw is near a multiple of 45 degrees,
// because then |rd.x| == |rd.z| for a whole line of pixels and every one of
// them ties on every step.

import package::grid::COARSE_B;

struct Seed {
  cell     : vec3<i32>,
  sideDist : vec3<f32>,
  // Mask of the single axis whose plane was crossed, for the face normal.
  axis     : vec3<f32>,
};

// Exactly one component is 1.0, so a step advances exactly one axis. Ties break
// x > y > z; the priority is arbitrary, only the "one axis" part matters.
fn stepMask(sideDist : vec3<f32>) -> vec3<f32> {
  let xFirst = sideDist.x <= sideDist.y && sideDist.x <= sideDist.z;
  let yFirst = !xFirst && sideDist.y <= sideDist.z;
  return vec3<f32>(
    select(0.0, 1.0, xFirst),
    select(0.0, 1.0, yFirst),
    select(0.0, 1.0, !xFirst && !yFirst),
  );
}

// Distance along the ray, per axis, from `p` to the far face of `cell`.
fn seedSideDist(cell : vec3<i32>, p : vec3<f32>, rd : vec3<f32>, deltaDist : vec3<f32>) -> vec3<f32> {
  return (sign(rd) * (vec3<f32>(cell) - p) + sign(rd) * 0.5 + 0.5) * deltaDist;
}

// Jump a ray across a provably-empty coarse block, landing on the first cell
// past the block's exit plane.
//
// Only the one axis that actually reaches the plane is stepped. Re-seeding from
// `ro + rd * (tExit + eps)` instead — a fixed nudge along the ray — also pushes
// any *other* axis that happens to be grazing its own boundary at that instant
// across it, so the cell diagonally past the corner is skipped entirely. When
// that cell is solid the ray passes straight through it.
fn skipBlock(ro : vec3<f32>, rd : vec3<f32>, cell : vec3<i32>, deltaDist : vec3<f32>) -> Seed {
  let cc = floor(vec3<f32>(cell) / f32(COARSE_B));
  let planeF = (cc + step(vec3<f32>(0.0), rd)) * f32(COARSE_B);
  let tP = (planeF - ro) / rd;
  let tExit = min(min(tP.x, tP.y), tP.z);
  // Land exactly on the plane, not past it.
  let pj = ro + rd * tExit;
  // Cell on the far side of the plane for the crossed axis: the plane index
  // itself going forward, one less going back.
  let crossed = vec3<i32>(planeF) - vec3<i32>(step(rd, vec3<f32>(0.0)));
  let m = stepMask(tP);
  var nc = select(vec3<i32>(floor(pj)), crossed, m > vec3<f32>(0.0));
  // Never let an axis fall back to the cell the ray just left. On a diagonal
  // ray the exit point sits exactly on a second axis's boundary too, and
  // flooring there returns the cell behind it; the next skip would then cross
  // that axis and floor the first one back, ping-ponging between two blocks
  // until the step budget runs out.
  nc = select(min(nc, cell), max(nc, cell), rd > vec3<f32>(0.0));
  var out : Seed;
  out.cell = nc;
  out.sideDist = seedSideDist(nc, pj, rd, deltaDist);
  out.axis = m;
  return out;
}
