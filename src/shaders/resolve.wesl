// The resolve pass of temporal upscaling (RenderQuality.upscale): turns the small, jittered frames
// the temporal kernel traces into a picture at the output's size.
//
// Each frame's primary rays go through a different point of their pixels (temporal.wesl
// pixelJitter), so over a few frames the samples cover the output's pixels too. For every output
// pixel this pass takes the frame's sample nearest to it, finds where the pixel's surface was in
// the previous accumulated picture (through the previous camera, with the hit distance the
// temporal history stores), clamps that history to the range of the frame's 3x3 samples around the
// pixel so what no longer belongs there (a disocclusion, a moved light) fades out, and blends the
// new sample in: more when it landed close to the pixel's centre, all of it where there is no
// history. This is temporal supersampling as in Brian Karis, "High Quality Temporal Supersampling"
// (SIGGRAPH 2014, Advances in Real-Time Rendering in Games course): jittered samples, reprojected
// history, neighbourhood clamping.
//
// A pass and a pipeline of its own, never part of the trace kernel: nothing here costs the kernel
// a register. Self-contained WGSL (no imports), so the renderer compiles it as it is.

struct Resolve {
  camPos   : vec3<f32>, _0 : f32,          // 0-3    this frame's camera, as in the main uniforms
  camRight : vec3<f32>, _1 : f32,          // 4-7
  camUp    : vec3<f32>, _2 : f32,          // 8-11
  camFwd   : vec3<f32>, tanHalfFov : f32,  // 12-15
  aspect   : f32, valid : f32, raw : f32, changedCount : f32, // 16-19  the output's aspect; 1 when the accumulated picture can be reused; development view (below); entries of `changed`
  // 20-275  The point lights that changed since the last frame (moved, recoloured, added, removed),
  // where they are and where they were: position and range. Surfaces in reach of one must not hold.
  changed  : array<vec4<f32>, 64>,
};

// The temporal block (temporal.wesl Temporal): the previous camera and this frame's jitter index.
struct Temporal {
  prevPos   : vec3<f32>, frame : u32,
  prevRight : vec3<f32>, flags : u32,
  prevUp    : vec3<f32>, prevTan : f32,
  prevFwd   : vec3<f32>, prevAspect : f32,
  prevRes   : vec2<f32>, cap : f32, rate : u32,
  spread    : f32, lampRadius : f32, show : f32, jitter : f32,
};

@group(0) @binding(0) var<uniform> R : Resolve;
@group(0) @binding(1) var<uniform> T : Temporal;
// This frame, at the traced size.
@group(0) @binding(2) var frame : texture_2d<f32>;
// This frame's temporal history, at the traced size: x the face hash (0 = no hit), y the hit distance's bits.
@group(0) @binding(3) var hist : texture_2d<u32>;
// The accumulated picture of the previous frame, and this frame's, at the output's size: colour,
// and in alpha the weight of the samples behind it (so the present pass must not copy alpha).
@group(0) @binding(4) var accPrev : texture_2d<f32>;
@group(0) @binding(5) var accSampler : sampler;
@group(0) @binding(6) var accOut : texture_storage_2d<rgba16float, write>;

// The clamp box (the range of the frame's 3x3 samples) is padded by this share of its range on each side.
const CLAMP_PAD : f32 = 0.25;
// A pixel counts as not having moved when its reprojection lands within this many output pixels of itself.
const STILL : f32 = 0.02;
// The weight at which a still pixel's history holds: about this many well-placed samples.
const STILL_WEIGHT : f32 = 24.0;
// The clamp box's padding for a still pixel, as CLAMP_PAD.
const STILL_PAD : f32 = 1.0;
// Frames running a held pixel must fall outside that box before it starts over.
const STILL_FAILS : f32 = 4.0;
// The most weight the history of a pixel being relit carries, and the share of a changed light's
// range within which a surface counts as relit.
const RELIT_WEIGHT : f32 = 12.0;
const RELIT_REACH : f32 = 0.7;
// The weight a moving pixel's history is given, for the running average to start from when it stops.
const MOVING_WEIGHT : f32 = 2.0;

// Halton radical inverse of `i` in base `b` (as temporal.wesl).
fn halton(i : u32, b : u32) -> f32 {
  var f = 1.0;
  var r = 0.0;
  var n = i;
  while (n > 0u) {
    f = f / f32(b);
    r = r + f * f32(n % b);
    n = n / b;
  }
  return r;
}

// Where in its pixel this frame's rays went (temporal.wesl pixelJitter; keep the two in step).
fn pixelJitter() -> vec2<f32> {
  if (T.jitter < 0.5) {
    return vec2<f32>(0.5);
  }
  let i = u32(T.jitter);
  return vec2<f32>(halton(i, 2u), halton(i, 3u));
}

// The accumulated picture at `uv`, through a Catmull-Rom filter: reprojecting through a bilinear
// one every frame is what blurs a picture in motion. Five bilinear taps stand in for the filter's
// sixteen texels (the four corner taps, whose weights are small, are left out and the rest
// renormalised).
fn historyAt(uv : vec2<f32>) -> vec3<f32> {
  let dim = vec2<f32>(textureDimensions(accPrev));
  let pos = uv * dim;
  let c = floor(pos - vec2<f32>(0.5)) + vec2<f32>(0.5);
  let f = pos - c;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let t0 = (c - vec2<f32>(1.0)) / dim;
  let t3 = (c + vec2<f32>(2.0)) / dim;
  let t12 = (c + w2 / w12) / dim;
  let a = w12.x * w0.y;
  let b = w0.x * w12.y;
  let m = w12.x * w12.y;
  let d = w3.x * w12.y;
  let e = w12.x * w3.y;
  let sum = textureSampleLevel(accPrev, accSampler, vec2<f32>(t12.x, t0.y), 0.0).rgb * a
    + textureSampleLevel(accPrev, accSampler, vec2<f32>(t0.x, t12.y), 0.0).rgb * b
    + textureSampleLevel(accPrev, accSampler, t12, 0.0).rgb * m
    + textureSampleLevel(accPrev, accSampler, vec2<f32>(t3.x, t12.y), 0.0).rgb * d
    + textureSampleLevel(accPrev, accSampler, vec2<f32>(t12.x, t3.y), 0.0).rgb * e;
  return sum / (a + b + m + d + e);
}

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) id : vec3<u32>) {
  let outDim = textureDimensions(accOut);
  if (id.x >= outDim.x || id.y >= outDim.y) {
    return;
  }
  let lowDim = vec2<i32>(textureDimensions(frame));
  let scale = vec2<f32>(lowDim) / vec2<f32>(outDim);
  let o = vec2<f32>(id.xy) + vec2<f32>(0.5);
  // The output pixel's centre in traced pixels, and the traced pixel whose sample is nearest to it.
  let c = o * scale;
  let jit = pixelJitter();
  let j = clamp(vec2<i32>(floor(c - jit + vec2<f32>(0.5))), vec2<i32>(0), lowDim - vec2<i32>(1));
  let cur = textureLoad(frame, j, 0).rgb;
  // Development view (globalThis.__voxolithTemporal.raw): the traced frame as it is, nearest
  // sample, no accumulation. A positive value shows only the frames traced at that jitter index
  // (the picture keeps the last one), so two captures compare the trace for the same rays; a
  // negative one shows every frame.
  if (R.raw != 0.0) {
    var shown = cur;
    if (R.raw > 0.0 && abs(T.jitter - R.raw) > 0.5) {
      shown = textureLoad(accPrev, vec2<i32>(id.xy), 0).rgb;
    }
    textureStore(accOut, id.xy, vec4<f32>(shown, 0.0));
    return;
  }
  var lo = cur;
  var hi = cur;
  for (var k = 0; k < 9; k = k + 1) {
    let s = textureLoad(frame, clamp(j + vec2<i32>(k % 3 - 1, k / 3 - 1), vec2<i32>(0), lowDim - vec2<i32>(1)), 0).rgb;
    lo = min(lo, s);
    hi = max(hi, s);
  }

  // How much this frame's sample counts for this pixel: by how far it landed from the pixel's
  // centre, in output pixels. A near one counts for more, which is what brings out detail finer
  // than the traced pixels.
  let off = (vec2<f32>(j) + jit - c) / scale;
  // The kernel is as wide as the samples are far apart: 3.0 at half scale, where a traced pixel
  // covers two output pixels each way.
  let k = max(exp(-12.0 * scale.x * scale.x * dot(off, off)), 1e-3);
  var color = cur;
  // The weight accumulated behind `color`, kept in the picture's alpha.
  var weight = k;
  if (R.valid > 0.5) {
    // Where the pixel's surface was in the previous view: the point its ray reaches at the stored
    // hit distance (far away for the sky).
    let h = textureLoad(hist, j, 0);
    let uv = (o / vec2<f32>(outDim)) * 2.0 - vec2<f32>(1.0);
    let dir = normalize(R.camFwd + R.camRight * (uv.x * R.aspect * R.tanHalfFov) + R.camUp * (-uv.y * R.tanHalfFov));
    let t = select(1e6, bitcast<f32>(h.y), h.x != 0u);
    let d = R.camPos + dir * t - T.prevPos;
    let z = dot(d, T.prevFwd);
    if (z > 1e-3) {
      let puv = vec2<f32>(dot(d, T.prevRight) / (z * R.aspect * T.prevTan), -dot(d, T.prevUp) / (z * T.prevTan));
      let st = puv * 0.5 + vec2<f32>(0.5);
      if (all(st >= vec2<f32>(0.0)) && all(st <= vec2<f32>(1.0))) {
        let pad = (hi - lo) * CLAMP_PAD;
        if (all(abs(st * vec2<f32>(outDim) - o) < vec2<f32>(STILL))) {
          // The pixel has not moved (a still camera): the same texel again, and a running average
          // of its samples by their weights, so the picture converges and then holds. Blending a
          // fixed share of each new sample, as below, never settles: every frame's jitter moves a
          // pixel that covers detail finer than the traced pixels.
          //
          // Once it carries STILL_WEIGHT it holds as it is: the jitter repeats every 16 frames, so
          // an average that kept moving would only ripple with it. What still has to get through
          // is a real change (a light, something moving, with the camera still): a held pixel is
          // tested against the frame's 3x3 range, padded, and starts over from the clamped value
          // once it has been outside it STILL_FAILS frames running. One frame is not enough: over
          // detail finer than the traced pixels a single jittered sample often lands on something
          // else, and restarting on it is a visible flash. The run is counted in the weight, above
          // STILL_WEIGHT.
          let p = textureLoad(accPrev, vec2<i32>(id.xy), 0);
          let wide = (hi - lo) * STILL_PAD + vec3<f32>(0.004);
          let inside = clamp(p.rgb, lo - wide, hi + wide);
          // A surface a light that changed this frame can matter to (within RELIT_REACH of its
          // range) is being relit: it keeps averaging, never clamped (the clamp is what puts the
          // jitter back), but with its weight capped at RELIT_WEIGHT, so it follows the light
          // within a few frames. Once the light is still it accumulates on towards the hold.
          var relit = false;
          if (h.x != 0u) {
            let hitAt = R.camPos + dir * t;
            let n = u32(R.changedCount);
            for (var i = 0u; i < n; i = i + 1u) {
              let l = R.changed[i];
              let e = hitAt - l.xyz;
              let reach = l.w * RELIT_REACH;
              if (dot(e, e) < reach * reach) { relit = true; }
            }
          }
          if (relit) {
            let had = min(p.a, RELIT_WEIGHT);
            color = (p.rgb * had + cur * k) / (had + k);
            weight = min(had + k, RELIT_WEIGHT);
          } else if (p.a < STILL_WEIGHT) {
            color = (p.rgb * p.a + cur * k) / (p.a + k);
            weight = min(p.a + k, STILL_WEIGHT);
          } else if (all(inside == p.rgb)) {
            color = p.rgb;
            weight = STILL_WEIGHT;
          } else if (p.a + 1.0 < STILL_WEIGHT + STILL_FAILS) {
            color = p.rgb;
            weight = p.a + 1.0;
          } else {
            color = mix(inside, cur, clamp(0.35 * k, 0.005, 1.0));
            weight = MOVING_WEIGHT;
          }
        } else {
          let prev = clamp(historyAt(st), lo - pad, hi + pad);
          color = mix(prev, cur, clamp(0.35 * k, 0.005, 1.0));
          weight = MOVING_WEIGHT;
        }
      }
    }
  }
  textureStore(accOut, id.xy, vec4<f32>(color, weight));
}
