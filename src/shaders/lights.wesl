// Point lights: diffuse with a range-limited shadow ray, and a visible glow.
//
// Each light has a hard range, past which it contributes exactly nothing, so
// the loop is cheap for every pixel the light cannot reach: the range test
// comes before any shadow ray. Inside the range the falloff is a smooth window
// times a soft inverse square, so a light fades to zero at its edge instead of
// ending in a ring.

import package::uniforms::{U, lights};
import package::shadow::occludedTo;
import package::fog::fogAmount;

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
// far away on a miss): a gaussian of the ray's closest approach to each light,
// seen through the fog between the eye and that light (so add it after the
// fog). A light behind the surface the ray hit shows nothing.
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
    sum += l.color * exp(-d * d) * (1.0 - fogAmount(ro, rd, t));
  }
  return sum;
}

// Point lighting for temporal accumulation, split three ways at `p` with normal `n`: `free`, the
// lights that cast no shadows; `open`, the shadowed lights as if nothing blocked them; and `lit`,
// the same lights with their shadow rays (only when `trace`; else 0). The rays aim at a point
// `jitter` (voxels, in a ball of the light's size) off each light's centre, so accumulated
// samples give soft point shadows. See temporal.wesl for how `lit / open` is kept.
struct PointSplit {
  free : vec3<f32>,
  open : vec3<f32>,
  lit  : vec3<f32>,
};

fn pointLightingSplit(p : vec3<f32>, n : vec3<f32>, trace : bool, jitter : vec3<f32>) -> PointSplit {
  var s : PointSplit;
  s.free = vec3<f32>(0.0);
  s.open = vec3<f32>(0.0);
  s.lit = vec3<f32>(0.0);
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
    let e = l.color * (ndl * window / (1.0 + dist * dist * FALLOFF_K));
    if (!(l.shadows && U.shadowSteps > 0.5)) {
      s.free += e;
      continue;
    }
    s.open += e;
    if (trace) {
      let dj = d + jitter;
      let lj = length(dj);
      var vis = 1.0;
      if (occludedTo(p, dj / max(lj, 1e-4), lj)) {
        vis = SHADOW_FLOOR;
      }
      s.lit += e * vis;
    }
  }
  return s;
}
