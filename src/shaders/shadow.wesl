// Coarse-accelerated shadow ray: true if any voxel blocks the straight path to
// the light. Skips provably-empty coarse blocks but stays fine-exact.

import package::uniforms::U;
import package::grid::{isSolid, coarseSolid, COARSE_B};

const SHADOW_STEPS : i32 = 90;

fn inShadow(origin : vec3<f32>, rd : vec3<f32>) -> bool {
  var cell = vec3<i32>(floor(origin));
  let stepDir = vec3<i32>(sign(rd));
  let deltaDist = abs(1.0 / rd);
  var sideDist = (sign(rd) * (vec3<f32>(cell) - origin) + sign(rd) * 0.5 + 0.5) * deltaDist;

  for (var i = 0; i < SHADOW_STEPS; i = i + 1) {
    if (any(cell < vec3<i32>(0)) || any(cell >= vec3<i32>(U.gridSize))) {
      return false;
    }
    if (U.debug < 0.5 && !coarseSolid(cell)) {
      let cc = floor(vec3<f32>(cell) / f32(COARSE_B));
      let planeF = (cc + step(vec3<f32>(0.0), rd)) * f32(COARSE_B);
      let tP = (planeF - origin) / rd;
      let tExit = min(min(tP.x, tP.y), tP.z);
      let pj = origin + rd * (tExit + 1e-3);
      cell = vec3<i32>(floor(pj));
      sideDist = (sign(rd) * (vec3<f32>(cell) - pj) + sign(rd) * 0.5 + 0.5) * deltaDist;
      continue;
    }
    let mask = select(
      vec3<f32>(0.0),
      vec3<f32>(1.0),
      sideDist.xyz <= min(sideDist.yzx, sideDist.zxy),
    );
    sideDist += mask * deltaDist;
    cell += vec3<i32>(mask) * stepDir;
    if (isSolid(cell)) {
      return true;
    }
  }
  return false;
}
