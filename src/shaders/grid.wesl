// Voxel grid sampling + the coarse occupancy test used for empty-space skipping.

import package::uniforms::{U, voxTex, coarseTex};

const COARSE_B : i32 = 4;  // matches COARSE_B in occupancy.ts

fn sampleVoxel(p : vec3<i32>) -> u32 {
  if (any(p < vec3<i32>(0)) || any(p >= vec3<i32>(U.gridSize))) {
    return 0u;
  }
  return textureLoad(voxTex, p, 0).r;
}

fn isSolid(p : vec3<i32>) -> bool {
  return sampleVoxel(p) != 0u;
}

// Is the coarse block containing fine cell `p` occupied? Out-of-range → treat as
// solid (don't fast-forward; let the fine loop's bounds check exit).
fn coarseSolid(p : vec3<i32>) -> bool {
  let c = vec3<i32>(floor(vec3<f32>(p) / f32(COARSE_B)));
  if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(U.coarseDim))) {
    return true;
  }
  return textureLoad(coarseTex, c, 0).r != 0u;
}
