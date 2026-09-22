// DDA voxel traversal (Amanatides & Woo) with coarse empty-space skipping. Walks
// from the box entry point to the first solid voxel, returning the hit cell,
// face normal, palette colour, and material index.

import package::uniforms::{U, palette};
import package::grid::{sampleVoxel, coarseSolid};
import package::dda::{stepMask, seedSideDist, skipBlock};

struct Hit {
  hit    : bool,
  cell   : vec3<i32>,
  normal : vec3<f32>,
  color  : vec3<f32>,
  mat    : u32,
};

// Hard ceiling for the primary-ray DDA. The effective cap is U.maxSteps (set via
// Renderer.setQuality), clamped to this; with coarse empty-space skipping on,
// typical rays use far fewer steps and the cap is only a backstop.
const MAX_STEPS : i32 = 1024;

fn traverse(ro : vec3<f32>, rd : vec3<f32>, tStart : f32) -> Hit {
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
  for (var i = 0; i < steps; i = i + 1) {
    if (any(cell < vec3<i32>(-1)) || any(cell > vec3<i32>(U.gridSize))) {
      break;
    }
    // Empty-space skip: if the coarse block is empty, jump the ray to that
    // block's far plane in one hop (re-seed exactly like the entry above).
    if (U.debug < 0.5 && !coarseSolid(cell)) {
      let seed = skipBlock(ro, rd, cell, deltaDist);
      cell = seed.cell;
      sideDist = seed.sideDist;
      normal = seed.axis * -sign(rd);
      continue;
    }
    let c = sampleVoxel(cell);
    if (c != 0u) {
      h.hit = true;
      h.cell = cell;
      h.normal = normal;
      h.color = palette[c].rgb;
      h.mat = c;
      return h;
    }
    // Advance along the axis with the smallest accumulated distance.
    let mask = stepMask(sideDist);
    sideDist += mask * deltaDist;
    cell += vec3<i32>(mask) * stepDir;
    normal = mask * -sign(rd);
  }
  return h;
}
