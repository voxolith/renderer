// Ray vs the occupied AABB (tighter than the full grid → sky/empty rays exit
// immediately). Returns vec2(tNear, tFar). tFar < tNear -> miss.

import package::uniforms::U;

fn intersectBox(ro : vec3<f32>, rd : vec3<f32>) -> vec2<f32> {
  let inv = 1.0 / rd;
  let t0 = (U.occMin - ro) * inv;
  let t1 = (U.occMax + vec3<f32>(1.0) - ro) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let tNear = max(max(tmin.x, tmin.y), tmin.z);
  let tFar = min(min(tmax.x, tmax.y), tmax.z);
  return vec2<f32>(tNear, tFar);
}
