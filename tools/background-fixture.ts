// A non-trivial background module for the checks (tools/verify.ts): it reads every binding the
// background may import (BG.params, BG.time, both textures, the sampler) and its footprint, so a
// host's module that uses them all links. Not shipped: skies live in the engine
// (@voxolith/engine/atmosphere).

/** A background using BG.params, BG.time, bgTex0, bgTex1, bgSampler and the footprint. */
export const TEXTURED_BACKGROUND_WESL = /* wgsl */ `
import package::bg::{BG, bgTex0, bgTex1, bgSampler};

fn background(rd : vec3<f32>, footprint : f32) -> vec3<f32> {
  let uv = vec2<f32>(atan2(rd.z, rd.x) / 6.2831853 + 0.5 + BG.time * 0.01, acos(clamp(rd.y, -1.0, 1.0)) / 3.1415927);
  let a = textureSampleLevel(bgTex0, bgSampler, uv, 0.0).rgb;
  let b = textureSampleLevel(bgTex1, bgSampler, uv, 0.0).rgb;
  let glow = smoothstep(1.0 - footprint * 4.0, 1.0, dot(rd, normalize(BG.params[2].xyz)));
  return BG.params[0].xyz * a + BG.params[1].xyz * b + glow;
}
`;
