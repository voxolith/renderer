// Fog: exponential extinction towards U.fogColor, optionally thinning with
// height. Applied to surface hits by their distance and to the sky at a fixed
// far distance, so the horizon dissolves into it.

import package::uniforms::U;

// What rays that miss everything count as having travelled.
const SKY_DISTANCE : f32 = 900.0;

fn fogAmount(ro : vec3<f32>, rd : vec3<f32>, t : f32) -> f32 {
  if (U.fogDensity <= 0.0) { return 0.0; }
  var density = U.fogDensity;
  if (U.fogFalloff > 0.0) {
    // Height fog: use the density at the segment's mean height.
    let y = max(ro.y + rd.y * t * 0.5, 0.0);
    density *= exp(-y * U.fogFalloff);
  }
  return 1.0 - exp(-density * t);
}

fn applyFog(color : vec3<f32>, ro : vec3<f32>, rd : vec3<f32>, t : f32) -> vec3<f32> {
  return mix(color, U.fogColor, fogAmount(ro, rd, t));
}

fn applySkyFog(color : vec3<f32>, ro : vec3<f32>, rd : vec3<f32>) -> vec3<f32> {
  return mix(color, U.fogColor, fogAmount(ro, rd, SKY_DISTANCE * max(U.effectScale, 1e-3)));
}
