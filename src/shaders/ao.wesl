// Classic voxel face AO: darken where the empty cell in front of the hit face has
// solid neighbours along the two tangent axes (plus corners).

import package::grid::isSolid;

fn faceAO(cell : vec3<i32>, n : vec3<f32>) -> f32 {
  let ni = vec3<i32>(n);
  let front = cell + ni;
  // Two tangent axes perpendicular to the normal.
  var t1 : vec3<i32>;
  var t2 : vec3<i32>;
  if (abs(n.y) > 0.5) {
    t1 = vec3<i32>(1, 0, 0);
    t2 = vec3<i32>(0, 0, 1);
  } else if (abs(n.x) > 0.5) {
    t1 = vec3<i32>(0, 1, 0);
    t2 = vec3<i32>(0, 0, 1);
  } else {
    t1 = vec3<i32>(1, 0, 0);
    t2 = vec3<i32>(0, 1, 0);
  }

  // Edges weigh 0.10, corners 0.05. One loop (not eight calls) so the voxel lookup, which with
  // instances is large, is compiled once.
  var occ = 0.0;
  for (var k = 0u; k < 8u; k = k + 1u) {
    var o = t1;
    switch k {
      case 0u: { o = t1; }
      case 1u: { o = -t1; }
      case 2u: { o = t2; }
      case 3u: { o = -t2; }
      case 4u: { o = t1 + t2; }
      case 5u: { o = t1 - t2; }
      case 6u: { o = -t1 + t2; }
      default: { o = -t1 - t2; }
    }
    if (isSolid(front + o)) {
      occ = occ + select(0.05, 0.10, k < 4u);
    }
  }
  return clamp(1.0 - occ, 0.45, 1.0);
}
