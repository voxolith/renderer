// Rain and snow as analytic particles: nothing is stored or uploaded.
//
// The primary ray samples a handful of depth layers, near to far, stopping at
// whatever it hit, so geometry occludes the particles for free. At each
// sample, space is sheared along the fall direction (so streaks lean with the
// wind), stretched for rain, scrolled by fall × time, and cut into cells; a
// hash puts at most one particle in each cell and decides whether it exists.
// The world positions stay put as the camera moves, and the field covers any
// size of world.

import package::uniforms::U;

fn hash3(p : vec3<f32>) -> f32 {
  return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

// Particle colour and coverage (0..1) along a ray up to distance tMax.
fn precipitation(ro : vec3<f32>, rd : vec3<f32>, tMax : f32) -> vec4<f32> {
  if (U.precipKind < 0.5 || U.precipDensity <= 0.0) { return vec4<f32>(0.0); }
  let snow = U.precipKind > 1.5;
  let fall = U.precipFall;
  // Shear so that points on one falling particle's path share an (x, z).
  let shear = fall.xz / min(fall.y, -0.001);
  let stretch = select(7.0, 1.0, snow);
  var acc = 0.0;
  for (var k = 0; k < 8; k = k + 1) {
    let t = 2.5 * pow(1.55, f32(k));
    if (t > tMax) { break; }
    let p = ro + rd * t;
    // Where this point was when the particle field started falling.
    let q = p - fall * U.time;
    let qs = vec3<f32>(q.x - shear.x * q.y, q.y / stretch, q.z - shear.y * q.y);
    let size = select(1.3, 2.2, snow) * (1.0 + f32(k) * 0.35);
    let cell = floor(qs / size);
    let h = hash3(cell + vec3<f32>(f32(k) * 17.0));
    if (h > U.precipDensity * 0.7) { continue; }
    // Where the particle sits in its cell.
    let j = vec3<f32>(hash3(cell + 3.1), hash3(cell + 7.7), hash3(cell + 11.3)) * 0.6 + 0.2;
    // Work in world units so a particle can be kept at least a pixel wide:
    // real drops are a fraction of a voxel and would alias away past a few
    // voxels. A widened particle is made fainter to keep its brightness.
    let local = (qs / size - cell - j) * size;
    let pix = t * 2.0 * U.tanHalfFov / U.resolution.y;
    var a : f32;
    if (snow) {
      let r = max(0.28, pix * 1.2);
      a = smoothstep(r, r * 0.35, length(local)) * min(1.0, 0.28 / r * 1.5);
    } else {
      let r = max(0.07, pix * 1.1);
      let len = size * 0.35;
      a = smoothstep(r, r * 0.3, length(local.xz)) * smoothstep(len, len * 0.4, abs(local.y)) * min(1.0, 0.07 / r * 2.5);
    }
    // Far layers are fainter; very near ones too, so drops do not smear the lens.
    acc += a * (1.0 - f32(k) / 9.0) * smoothstep(3.0, 9.0, t);
  }
  let alpha = clamp(acc * select(0.95, 0.95, snow), 0.0, 0.85);
  return vec4<f32>(U.precipColor, alpha);
}
