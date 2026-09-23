// Point lights: diffuse with a range-limited shadow ray, and a visible glow.
//
// Each light has a hard range, past which it contributes exactly nothing, so
// the loop is cheap for every pixel the light cannot reach: the range test
// comes before any shadow ray. Inside the range the falloff is a smooth window
// times a soft inverse square, so a light fades to zero at its edge instead of
// ending in a ring.

import package::uniforms::{U, lights};
import package::shadow::occludedTo;

// Inverse-square softness: at 1/sqrt(K) voxels the light is at half strength
// (before the range window).
const FALLOFF_K : f32 = 0.0025;
// What an occluded light still contributes, so a stair-stepped voxel edge in
// its shadow does not read as a black hole.
const SHADOW_FLOOR : f32 = 0.08;

struct PointLight {
  pos     : vec3<f32>,
  range   : f32,
  color   : vec3<f32>,
  shadows : bool,
  glow    : f32,
};

fn pointLightAt(i : u32) -> PointLight {
  let a = lights[i * 3u];
  let b = lights[i * 3u + 1u];
  let c = lights[i * 3u + 2u];
  var l : PointLight;
  l.pos = a.xyz;
  l.range = a.w;
  l.color = b.xyz;
  l.shadows = c.x > 0.5;
  l.glow = c.y;
  return l;
}

// Summed point-light irradiance at `p` (just outside a surface) with normal `n`.
// `shadows` false skips occlusion, for cheap secondary bounces.
fn pointLighting(p : vec3<f32>, n : vec3<f32>, shadows : bool) -> vec3<f32> {
  var sum = vec3<f32>(0.0);
  let count = u32(U.lightCount);
  for (var i = 0u; i < count; i = i + 1u) {
    let l = pointLightAt(i);
    let d = l.pos - p;
    let dist = length(d);
    if (dist >= l.range) { continue; }
    let dir = d / max(dist, 1e-4);
    let ndl = dot(n, dir);
    if (ndl <= 0.0) { continue; }
    let x = dist / l.range;
    let window = (1.0 - x * x) * (1.0 - x * x);
    var vis = 1.0;
    if (shadows && l.shadows && U.shadowSteps > 0.5 && occludedTo(p, dir, dist)) {
      vis = SHADOW_FLOOR;
    }
    sum += l.color * (ndl * window * vis / (1.0 + dist * dist * FALLOFF_K));
  }
  return sum;
}

// Visible glow of the lights along a ray, up to `tMax` (the hit distance, or
// far away on a miss): a gaussian of the ray's closest approach to each light.
// A light behind the surface the ray hit shows nothing.
fn lightGlow(ro : vec3<f32>, rd : vec3<f32>, tMax : f32) -> vec3<f32> {
  var sum = vec3<f32>(0.0);
  let count = u32(U.lightCount);
  for (var i = 0u; i < count; i = i + 1u) {
    let l = pointLightAt(i);
    if (l.glow <= 0.0) { continue; }
    let t = dot(l.pos - ro, rd);
    if (t < 0.0 || t > tMax + l.glow) { continue; }
    let closest = ro + rd * min(t, tMax);
    let d = length(l.pos - closest) / l.glow;
    sum += l.color * exp(-d * d);
  }
  return sum;
}
