// Sky background: a vertical gradient with blocky night stars and world-positioned
// sun/moon discs (intensities/colours come from the environment uniforms).

import package::uniforms::U;

// A blocky (voxel-style) celestial disc in 3D: project the ray into a tangent
// basis around `dir`, quantise the local UV into cells, then radius-test → square
// edges. Being world-positioned, it's seen through the window from any angle.
fn celestialDisc(rd : vec3<f32>, dir : vec3<f32>, radius : f32, cells : f32) -> f32 {
  let d = normalize(dir);
  if (dot(rd, d) < 0.8) { return 0.0; }
  var up = vec3<f32>(0.0, 1.0, 0.0);
  if (abs(d.y) > 0.95) { up = vec3<f32>(1.0, 0.0, 0.0); }
  let t1 = normalize(cross(up, d));
  let t2 = cross(d, t1);
  let cell = radius / cells;
  let qu = (floor(dot(rd, t1) / cell) + 0.5) * cell;
  let qv = (floor(dot(rd, t2) / cell) + 0.5) * cell;
  return select(0.0, 1.0, qu * qu + qv * qv < radius * radius);
}

fn skyColor(rd : vec3<f32>, uv : vec2<f32>) -> vec3<f32> {
  let t = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
  var col = mix(U.skyHorizon, U.skyTop, smoothstep(0.0, 0.6, t));

  // Blocky stars at night (quantise the direction into cells, sparse hash).
  if (U.nightFactor > 0.05) {
    let q = floor(rd * 60.0);
    let h = fract(sin(dot(q, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
    let star = select(0.0, 1.0, h > 0.992);
    col += vec3<f32>(0.9, 0.92, 1.0) * star * U.nightFactor;
  }

  // Sun: glow + blocky disc (world-positioned, seen through the window).
  if (U.sunIntensity > 0.01) {
    col += U.sunColor * pow(max(dot(rd, normalize(U.sunDir)), 0.0), 40.0) * U.sunIntensity * 0.5;
    col = mix(col, U.sunColor, celestialDisc(rd, U.sunDir, 0.1, 4.0) * U.sunIntensity);
  }
  // Moon.
  if (U.moonIntensity > 0.01) {
    col += U.moonColor * pow(max(dot(rd, normalize(U.moonDir)), 0.0), 60.0) * U.moonIntensity * 0.3;
    col = mix(col, U.moonColor, celestialDisc(rd, U.moonDir, 0.08, 4.0) * U.moonIntensity);
  }

  return col;
}
