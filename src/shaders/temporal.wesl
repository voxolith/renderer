// Temporal accumulation of the traced shading terms (RenderQuality.temporal): the key light's
// shadow, face AO and the point lights' shadows are kept per pixel across frames and reused
// wherever the pixel sees the same voxel face as before, so a still or slowly moving view traces
// them only until they have converged. Only the temporal compute kernels (raymarch.wesl
// trace_temporal*) reference anything here; every other pipeline is built from entry points that
// never touch these bindings, so it compiles none of it.
//
// History is rejected without motion vectors: each pixel stores the hash of the face it hit and
// the distance to it, and a pixel reuses the previous frame's value only where the reprojected
// previous sample hit the same face at a consistent distance. What changes the shading of a
// surface the pixel still sees (an edit, a moving instance, its shadow) is reported by the CPU as
// per-brick-column change stamps (`changed`), which make those pixels trace afresh.
//
// The shadowed share of the point lights is accumulated as a ratio of shadowed to unshadowed
// irradiance and applied to the unshadowed irradiance evaluated every frame, so lights that change
// colour or brightness (flicker, fireflies) need no history reset. This is the ratio estimator of
// Heitz, Hill and McGuire, "Combining Analytic Direct Illumination and Stochastic Shadows"
// (I3D 2018, doi:10.1145/3190834.3190852).

import package::uniforms::U;

struct Temporal {
  prevPos   : vec3<f32>, frame : u32,       // 0-3    the previous frame's camera; this frame's number
  prevRight : vec3<f32>, flags : u32,       // 4-7    FLAG_*
  prevUp    : vec3<f32>, prevTan : f32,     // 8-11
  prevFwd   : vec3<f32>, prevAspect : f32,  // 12-15
  prevRes   : vec2<f32>, cap : f32, rate : u32, // 16-19  samples a pixel averages; tiles take turns 1 in `rate` frames
  spread    : f32, lampRadius : f32, show : f32, _b : f32, // 20-23  key light cone half-angle (rad); point light radius (voxels); development view
};

// The previous frame's history is there to reuse (off after a reset).
const FLAG_HISTORY : u32 = 1u;
// Converged pixels stop sampling (off while the key light moves).
const FLAG_HOLD : u32 = 2u;
// The key light jumped: its shadow is traced afresh everywhere this frame.
const FLAG_FORCE_KEY : u32 = 4u;
// A shadowed point light moved: point shadows are traced afresh everywhere this frame.
const FLAG_FORCE_POINT : u32 = 8u;
// Some point light casts shadows (else there is no point term to accumulate).
const FLAG_POINT_SHADOWS : u32 = 16u;

@group(2) @binding(0) var<uniform> T : Temporal;
// Last frame's history, and this frame's: per pixel (face hash, distance bits, key|ao|count,
// point ratio rgb), packed by packHistory.
@group(2) @binding(1) var histIn : texture_2d<u32>;
@group(2) @binding(2) var histOut : texture_storage_2d<rgba32uint, write>;
// Per brick column (x / 8, z / 8): the number of the frame in which something there last changed.
@group(2) @binding(3) var changed : texture_2d<u32>;

struct History {
  valid : bool,
  /** Key light visibility, 0 (shadowed) .. 1 (lit). */
  key   : f32,
  ao    : f32,
  /** Shadowed over unshadowed point irradiance, per channel. */
  ratio : vec3<f32>,
  /** Samples averaged so far. */
  count : u32,
};

// A voxel face (cell plus the side the ray entered through) as a non-zero 32-bit hash; 0 means
// "no hit". Collisions are caught by the distance test in lookupHistory.
// Which of the six sides a normal is (0 for none): axis * 2 + positive, 2..7.
fn sideOf(n : vec3<f32>) -> u32 {
  return u32(dot(abs(n), vec3<f32>(1.0, 2.0, 3.0))) * 2u + select(0u, 1u, dot(n, vec3<f32>(1.0)) > 0.0);
}

fn faceHash(c : vec3<i32>, n : vec3<f32>) -> u32 {
  let side = sideOf(n);
  var h = bitcast<u32>(c.x) * 0x8da6b343u;
  h = h ^ (bitcast<u32>(c.y) * 0xd8163841u);
  h = h ^ (bitcast<u32>(c.z) * 0xcb1ab31fu);
  h = h ^ (side * 0x165667b1u);
  h = h ^ (h >> 15u);
  h = h * 0x2c1b3c6du;
  h = h ^ (h >> 12u);
  h = h * 0x297a2d39u;
  h = h ^ (h >> 15u);
  return max(h, 1u);
}

fn packHistory(hash : u32, t : f32, side : u32, s : History) -> vec4<u32> {
  let key = u32(round(clamp(s.key, 0.0, 1.0) * 16383.0));
  let ao = u32(round(clamp(s.ao, 0.0, 1.0) * 4095.0));
  let r = vec3<u32>(round(clamp(s.ratio, vec3<f32>(0.0), vec3<f32>(1.0)) * vec3<f32>(1023.0, 1023.0, 511.0)));
  return vec4<u32>(hash, bitcast<u32>(t), key | (ao << 14u) | (min(s.count, 63u) << 26u), r.x | (r.y << 10u) | (r.z << 20u) | (side << 29u));
}

fn unpackHistory(v : vec4<u32>) -> History {
  var s : History;
  s.valid = true;
  s.key = f32(v.z & 16383u) / 16383.0;
  s.ao = f32((v.z >> 14u) & 4095u) / 4095.0;
  s.count = v.z >> 26u;
  s.ratio = vec3<f32>(f32(v.w & 1023u) / 1023.0, f32((v.w >> 10u) & 1023u) / 1023.0, f32((v.w >> 20u) & 511u) / 511.0);
  return s;
}

// The previous frame's history texel for the face `hash` (normal `n`) at world point `p`: the
// texel `p` projects to in the previous view, else another of the four around it, provided it saw
// the same face at a consistent distance. Failing that, the nearest of them that saw a face of the
// same plane within a voxel and a half, else any of them that saw a surface, as estimates with
// their sample count cut (to 2, and to 0 so the first sample replaces it): a face narrower than a
// pixel is often missed by the previous frame's samples, and one pixel starting over costs its
// whole tile. All zero when the four saw nothing.
fn lookupHistory(p : vec3<f32>, hash : u32, n : vec3<f32>) -> vec4<u32> {
  if ((T.flags & FLAG_HISTORY) == 0u) { return vec4<u32>(0u); }
  let d = p - T.prevPos;
  let z = dot(d, T.prevFwd);
  if (z < 1e-3) { return vec4<u32>(0u); }
  let uv = vec2<f32>(dot(d, T.prevRight) / (z * T.prevAspect * T.prevTan), -dot(d, T.prevUp) / (z * T.prevTan));
  let f = (uv * 0.5 + vec2<f32>(0.5)) * T.prevRes;
  let near = vec2<i32>(floor(f));
  let base = vec2<i32>(floor(f - vec2<f32>(0.5)));
  let dims = vec2<i32>(T.prevRes);
  let dist = length(d);
  let side = sideOf(n);
  let an = abs(n);
  var best = vec4<u32>(0u);
  var bestD = 2.25;
  var other = vec4<u32>(0u);
  for (var k = 0; k < 5; k = k + 1) {
    var q = near;
    if (k > 0) {
      q = base + vec2<i32>((k - 1) & 1, (k - 1) >> 1u);
      if (all(q == near)) { continue; }
    }
    if (any(q < vec2<i32>(0)) || any(q >= dims)) { continue; }
    let v = textureLoad(histIn, q, 0);
    if (v.x == 0u) { continue; }
    let tq = bitcast<f32>(v.y);
    if (v.x == hash && abs(tq - dist) <= 1.5 + 0.01 * dist) {
      return v;
    }
    if (other.x == 0u) { other = v; }
    if ((v.w >> 29u) != side || side < 2u) { continue; }
    // Where that texel's ray hit: the same plane, near p?
    let uq = ((vec2<f32>(q) + vec2<f32>(0.5)) / T.prevRes) * 2.0 - vec2<f32>(1.0);
    let dq = normalize(T.prevFwd + T.prevRight * (uq.x * T.prevAspect * T.prevTan) - T.prevUp * (uq.y * T.prevTan));
    let e = T.prevPos + dq * tq - p;
    let off = dot(e, an);
    let lat = dot(e, e) - off * off;
    if (abs(off) < 0.05 && lat < bestD) {
      bestD = lat;
      best = v;
    }
  }
  // Cut the count (bits 26..31) of an estimate.
  if (best.x != 0u) {
    return vec4<u32>(best.x, best.y, (best.z & 0x03ffffffu) | (min(best.z >> 26u, 2u) << 26u), best.w);
  }
  return vec4<u32>(other.x, other.y, other.z & 0x03ffffffu, other.w);
}

// Whether something changed this frame in the brick column holding `cell`.
fn changedAt(cell : vec3<i32>) -> bool {
  let dims = vec2<i32>(textureDimensions(changed));
  let q = vec2<i32>(cell.x >> 3u, cell.z >> 3u);
  if (cell.x < 0 || cell.z < 0 || q.x >= dims.x || q.y >= dims.y) { return false; }
  return textureLoad(changed, q, 0).x >= T.frame;
}

// Sample `k` of a face's sequence: the R2 low-discrepancy sequence (Roberts 2018), rotated per face
// so neighbouring faces do not step in lockstep. Two numbers in [0, 1).
fn faceSample(hash : u32, k : u32) -> vec2<f32> {
  let rot = vec2<f32>(f32(hash & 0xffffu), f32(hash >> 16u)) / 65536.0;
  return fract(rot + f32(k) * vec2<f32>(0.7548776662, 0.5698402910));
}

// A direction within `spread` radians of the unit vector `l`, uniform over the disc (u in [0, 1)^2).
fn coneDir(l : vec3<f32>, u : vec2<f32>, spread : f32) -> vec3<f32> {
  let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(l.x) > 0.9);
  let t1 = normalize(cross(l, a));
  let t2 = cross(l, t1);
  let r = sqrt(u.x) * spread;
  let phi = 6.2831853 * u.y;
  return normalize(l + t1 * (r * cos(phi)) + t2 * (r * sin(phi)));
}
