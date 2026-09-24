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

fn hash2(p : vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn valueNoise2(p : vec2<f32>) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash2(i);
  let b = hash2(i + vec2<f32>(1.0, 0.0));
  let c = hash2(i + vec2<f32>(0.0, 1.0));
  let d = hash2(i + vec2<f32>(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Cloud density along a sky direction: fbm on a plane above the world,
// drifting with the wind. 0 clear sky .. 1 solid cloud.
fn cloudAmount(rd : vec3<f32>) -> f32 {
  if (U.cloudCover <= 0.001) { return 0.0; }
  let h = max(rd.y, 0.04);
  var p = rd.xz / h * 1.6 + U.cloudDrift * U.time * 0.02;
  var n = 0.0;
  var amp = 0.55;
  for (var i = 0; i < 4; i = i + 1) {
    n += valueNoise2(p) * amp;
    p = p * 2.03 + vec2<f32>(17.1, 9.3);
    amp *= 0.5;
  }
  // Cover moves the threshold: at 1 the whole sky is cloud.
  let edge = 1.0 - U.cloudCover;
  let c = smoothstep(edge * 0.9 - 0.05, edge * 0.9 + 0.25, n) ;
  return max(c, smoothstep(0.75, 1.0, U.cloudCover));
}

fn skyColor(rd : vec3<f32>, uv : vec2<f32>) -> vec3<f32> {
  let t = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
  var col = mix(U.skyHorizon, U.skyTop, smoothstep(0.0, 0.6, t));
  let cloud = cloudAmount(rd);
  // Cloud hides the stars and the discs; overcast also dims their glow.
  let clear = 1.0 - max(cloud, U.cloudCover * 0.85);

  // Blocky stars at night (quantise the direction into cells, sparse hash).
  if (U.nightFactor > 0.05) {
    let q = floor(rd * 60.0);
    let h = fract(sin(dot(q, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
    let star = select(0.0, 1.0, h > 0.992);
    col += vec3<f32>(0.9, 0.92, 1.0) * star * U.nightFactor * clear;
  }

  // Sun: glow + blocky disc (world-positioned, seen through the window).
  if (U.sunIntensity > 0.01) {
    col += U.sunColor * pow(max(dot(rd, normalize(U.sunDir)), 0.0), 40.0) * U.sunIntensity * 0.5 * (0.35 + 0.65 * clear);
    col = mix(col, U.sunColor, celestialDisc(rd, U.sunDir, 0.1, 4.0) * U.sunIntensity * clear);
  }
  // Moon.
  if (U.moonIntensity > 0.01) {
    col += U.moonColor * pow(max(dot(rd, normalize(U.moonDir)), 0.0), 60.0) * U.moonIntensity * 0.3 * clear;
    col = mix(col, U.moonColor, celestialDisc(rd, U.moonDir, 0.08, 4.0) * U.moonIntensity * clear);
  }

  // Clouds over everything, lit a little from the sun side.
  if (cloud > 0.0) {
    let lit = 0.85 + 0.25 * max(dot(rd, normalize(U.sunDir)), 0.0) * U.sunIntensity;
    col = mix(col, U.cloudColor * lit, cloud * smoothstep(-0.02, 0.12, rd.y));
  }
  return col;
}
