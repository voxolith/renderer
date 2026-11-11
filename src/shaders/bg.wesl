// The background's resources (group 0, bindings 9-12), for the background module a host supplies
// (RendererOptions.background): `import package::bg::{BG, bgTex0, bgTex1, bgSampler};`.
// Sample the textures with textureSampleLevel: the background runs in compute kernels and inside
// non-uniform control flow, where textureSample (implicit derivatives) is not allowed.

struct Background {
  // FrameParams.background, 64 floats (zero past what was given).
  params : array<vec4<f32>, 16>,
  // FrameParams.time, seconds.
  time   : f32,
  _p0    : f32,
  _p1    : f32,
  _p2    : f32,
};

@group(0) @binding(9) var<uniform> BG : Background;
// RendererOptions.background.textures, or 1x1 black when not given.
@group(0) @binding(10) var bgTex0 : texture_2d<f32>;
@group(0) @binding(11) var bgTex1 : texture_2d<f32>;
// Linear filtering, repeating in u (longitude), clamped in v.
@group(0) @binding(12) var bgSampler : sampler;
