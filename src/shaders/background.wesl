// Ray-miss background: an optional infinite blocky ground plane for downward rays
// (so a floor is visible past the grid), else the sky. The floor is opt-in via
// setFloor() — games whose ground is real voxels leave it disabled.

import package::uniforms::U;
import package::sky::skyColor;

fn background(ro : vec3<f32>, rd : vec3<f32>, uv : vec2<f32>) -> vec3<f32> {
  if (U.floorParams.x > 0.5 && rd.y < -0.001) {
    let tg = (U.floorParams.y - ro.y) / rd.y;
    if (tg > 0.0) {
      let hp = ro + rd * tg;
      let tile = (floor(hp.x / 3.0) + floor(hp.z / 3.0));
      let base = select(U.floorColorB, U.floorColorA, (i32(tile) & 1) == 0);
      let lit = base * (U.ambientSky * 0.85 + max(U.lightDir.y, 0.0) * 0.6 * U.lightColor);
      let fog = smoothstep(140.0, 430.0, tg);
      return mix(lit, skyColor(rd, uv), fog);
    }
  }
  return skyColor(rd, uv);
}
