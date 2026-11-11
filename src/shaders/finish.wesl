// The present pass of temporal upscaling (RenderQuality.upscale): the resolved picture into the
// canvas or target, with the point lights' glow added here, per output pixel, from this frame's
// lights. The trace leaves the glow out while upscaling (U.glowLater), so it never enters the
// accumulated picture: there it would freeze on a still camera (a held pixel cannot tell a
// drifting firefly from jitter) and smear in motion.
//
// The renderer compiles this after the text of uniforms.wesl (for U and `lights`); no imports.
// fogAmount and glow repeat fog.wesl's fogAmount and lights.wesl's lightGlow: keep them in step.

// The resolved picture (alpha is the resolve's weights, not opacity) and this frame's temporal
// history at the traced size (x the face hash, 0 = no hit; y the hit distance's bits).
@group(1) @binding(0) var resolved : texture_2d<f32>;
@group(1) @binding(1) var traced : texture_2d<u32>;

fn fogAmount(ro : vec3<f32>, rd : vec3<f32>, t : f32) -> f32 {
  var wall = 0.0;
  if (U.viewDistance > 0.0) {
    wall = smoothstep(0.75 * U.viewDistance, U.viewDistance, t);
  }
  if (U.fogDensity <= 0.0) { return wall; }
  var density = U.fogDensity;
  if (U.fogFalloff > 0.0) {
    let y = max(ro.y + rd.y * t * 0.5, 0.0);
    density *= exp(-y * U.fogFalloff);
  }
  return max(1.0 - exp(-density * t), wall);
}

fn glow(ro : vec3<f32>, rd : vec3<f32>, tMax : f32) -> vec3<f32> {
  var sum = vec3<f32>(0.0);
  let count = u32(U.lightCount);
  for (var i = 0u; i < count; i = i + 1u) {
    let a = lights[i * 3u];
    let radius = lights[i * 3u + 2u].y;
    if (radius <= 0.0) { continue; }
    let t = dot(a.xyz - ro, rd);
    if (t < 0.0 || t > tMax + radius) { continue; }
    let closest = ro + rd * min(t, tMax);
    let d = length(a.xyz - closest) / radius;
    sum += lights[i * 3u + 1u].xyz * exp(-d * d) * (1.0 - fogAmount(ro, rd, t));
  }
  return sum;
}

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
  let c = textureLoad(resolved, vec2<i32>(pos.xy), 0).rgb;
  if (U.lightCount < 0.5) {
    return vec4<f32>(c, 1.0);
  }
  let outDim = vec2<f32>(textureDimensions(resolved));
  let lowDim = vec2<i32>(textureDimensions(traced));
  let uv = (pos.xy / outDim) * 2.0 - vec2<f32>(1.0);
  let dir = normalize(U.camFwd + U.camRight * (uv.x * U.aspect * U.tanHalfFov) + U.camUp * (-uv.y * U.tanHalfFov));
  let h = textureLoad(traced, clamp(vec2<i32>(pos.xy * vec2<f32>(lowDim) / outDim), vec2<i32>(0), lowDim - vec2<i32>(1)), 0);
  if (h.x == 0u) {
    // A miss: the trace adds the glow to the sky as it is.
    let far = select(1e6, U.viewDistance, U.viewDistance > 0.0);
    return vec4<f32>(c + glow(U.camPos, dir, far), 1.0);
  }
  // A hit: the trace adds the glow before its vignette and its final curve (raymarch.wesl
  // shadeHit), so undo the curve, add the glow under the same vignette, and apply it again.
  let v = 1.0 - 0.16 * dot(uv, uv);
  let g = glow(U.camPos, dir, bitcast<f32>(h.y));
  return vec4<f32>(pow(pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 0.92)) + g * v, vec3<f32>(0.92)), 1.0);
}
