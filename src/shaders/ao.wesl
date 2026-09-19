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

  let s1 = select(0.0, 1.0, isSolid(front + t1));
  let s2 = select(0.0, 1.0, isSolid(front - t1));
  let s3 = select(0.0, 1.0, isSolid(front + t2));
  let s4 = select(0.0, 1.0, isSolid(front - t2));
  let c1 = select(0.0, 1.0, isSolid(front + t1 + t2));
  let c2 = select(0.0, 1.0, isSolid(front + t1 - t2));
  let c3 = select(0.0, 1.0, isSolid(front - t1 + t2));
  let c4 = select(0.0, 1.0, isSolid(front - t1 - t2));

  let occ = (s1 + s2 + s3 + s4) * 0.10 + (c1 + c2 + c3 + c4) * 0.05;
  return clamp(1.0 - occ, 0.45, 1.0);
}
