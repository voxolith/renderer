// Ray-miss background: an optional infinite blocky ground plane for downward rays
// (so a floor is visible past the grid), else the host's background module
// (RendererOptions.background, linked as ./userbg.wesl). The floor is opt-in via
// setFloor() — games whose ground is real voxels leave it disabled.

import package::uniforms::U;
import package::userbg::background;

// `footprint` is the angular size of what the ray stands for (radians), handed to background().
fn missBackground(ro : vec3<f32>, rd : vec3<f32>, footprint : f32) -> vec3<f32> {
  if (U.floorParams.x > 0.5 && rd.y < -0.001) {
    let tg = (U.floorParams.y - ro.y) / rd.y;
    if (tg > 0.0) {
      let hp = ro + rd * tg;
      let tile = (floor(hp.x / 3.0) + floor(hp.z / 3.0));
      let base = select(U.floorColorB, U.floorColorA, (i32(tile) & 1) == 0);
      let lit = base * (U.ambientSky * 0.85 + max(U.lightDir.y, 0.0) * 0.6 * U.lightColor);
      let fog = smoothstep(140.0, 430.0, tg);
      return mix(lit, background(rd, footprint), fog);
    }
  }
  return background(rd, footprint);
}

// The angular size of one pixel at the centre of the view (radians).
fn pixelFootprint() -> f32 {
  return 2.0 * U.tanHalfFov / U.resolution.y;
}
