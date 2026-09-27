// Root module of the raymarch pass: shadePixel ties together intersection,
// traversal, shadow, AO, highlight and background for one pixel, and two entry
// points run it: the fullscreen vertex + fragment pair, and a compute kernel
// (trace_main) that writes the frame into a storage texture for the present
// pass. Imports pull in only what's used (the linker drops the rest).

import package::uniforms::{U, palette};
import package::intersect::intersectBox;
import package::trace::{traverse, Hit};
import package::shadow::inShadow;
import package::ao::faceAO;
import package::background::background;
import package::highlight::highlightEdge;
import package::materials::matOf;
import package::lights::{pointLighting, pointLightingSplit, lightGlow};
import package::water::shadeWater;
import package::fog::{applyFog, applySkyFog};
import package::precip::precipitation;
import package::shadow::occludedTo;
import package::selftest::{selfTest, selfTest2, selfTest3};
import package::temporal::{T, History, histOut, faceHash, sideOf, packHistory, unpackHistory, lookupHistory, changedAt, faceSample, coneDir, FLAG_HOLD, FLAG_FORCE_KEY, FLAG_FORCE_POINT, FLAG_POINT_SHADOWS};

struct VsOut {
  @builtin(position) pos : vec4<f32>,
};

fn hash31(p : vec3<f32>) -> f32 {
  return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}

// One-bounce lit trace for reflections/refraction: hit + simple diffuse (+ any
// emissive) or the sky. Deliberately does NOT recurse (WGSL has no recursion).
fn traceLit(ro : vec3<f32>, rd : vec3<f32>) -> vec3<f32> {
  let box = intersectBox(ro, rd);
  let tN = max(box.x, 0.0);
  if (box.y < tN) { return background(ro, rd, vec2<f32>(0.0)); }
  let h = traverse(ro, rd, tN);
  if (!h.hit) { return background(ro, rd, vec2<f32>(0.0)); }
  let n = h.normal;
  let L = normalize(U.lightDir);
  let ndl = max(dot(n, L), 0.0);
  let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
  let ambient = mix(U.ambientGround, U.ambientSky, up) * 0.85;
  let p = vec3<f32>(h.cell) + vec3<f32>(0.5) + n * 0.5;
  var c = h.color * (ambient + ndl * 0.6 * U.lightColor + pointLighting(p, n, false));
  if (U.materialsEnabled > 0.5) { c += h.color * matOf(h.mat).emit; }
  return c;
}

// A ray that hits nothing: sky (or the optional floor), lamp glows, fog and
// the rain or snow in front of it.
fn missColor(ro : vec3<f32>, rd : vec3<f32>, uv : vec2<f32>) -> vec3<f32> {
  // With a view distance, lights past it are behind the fog wall.
  let far = select(1e6, U.viewDistance, U.viewDistance > 0.0);
  let c = applySkyFog(background(ro, rd, uv), ro, rd) + lightGlow(ro, rd, far);
  let pr = precipitation(ro, rd, 1e6);
  return mix(c, pr.rgb, pr.a);
}

// Fullscreen triangle.
@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VsOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  var out : VsOut;
  out.pos = vec4<f32>(p[vi], 0.0, 1.0);
  return out;
}

@fragment
fn fs(@builtin(position) fragCoord : vec4<f32>) -> @location(0) vec4<f32> {
  return shadePixel(fragCoord);
}

// The compute path's output: the frame at render resolution, presented (copied to
// the canvas or a RenderTarget) by the renderer's present pass.
@group(1) @binding(0) var frameOut : texture_storage_2d<rgba16float, write>;

// One thread per pixel, in WG_X x WG_Y tiles (pipeline constants, 8x8 by default;
// 8x8 measured best among 8x4 to 32x4 on a desktop Radeon).
// The pixel centre matches what the fragment stage's @builtin(position) gives, so
// both paths shade exactly the same rays.
@id(2) override WG_X : u32 = 8u;
@id(3) override WG_Y : u32 = 8u;
@compute @workgroup_size(WG_X, WG_Y)
fn trace_main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= u32(U.resolution.x) || id.y >= u32(U.resolution.y)) {
    return;
  }
  textureStore(frameOut, id.xy, shadePixel(vec4<f32>(vec2<f32>(id.xy) + vec2<f32>(0.5), 0.0, 1.0)));
}

// Straight into the canvas when it allows storage writes: no present pass. One
// kernel per canvas format (bgra8unorm needs the bgra8unorm-storage feature).
@group(1) @binding(1) var canvasBgra : texture_storage_2d<bgra8unorm, write>;
@group(1) @binding(2) var canvasRgba : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(WG_X, WG_Y)
fn trace_canvas_bgra(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= u32(U.resolution.x) || id.y >= u32(U.resolution.y)) {
    return;
  }
  textureStore(canvasBgra, id.xy, shadePixel(vec4<f32>(vec2<f32>(id.xy) + vec2<f32>(0.5), 0.0, 1.0)));
}

@compute @workgroup_size(WG_X, WG_Y)
fn trace_canvas_rgba(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= u32(U.resolution.x) || id.y >= u32(U.resolution.y)) {
    return;
  }
  textureStore(canvasRgba, id.xy, shadePixel(vec4<f32>(vec2<f32>(id.xy) + vec2<f32>(0.5), 0.0, 1.0)));
}

// The primary ray through a pixel centre: NDC `uv` in [-1, 1] (y up), origin and direction.
struct Primary {
  uv  : vec2<f32>,
  ro  : vec3<f32>,
  dir : vec3<f32>,
};

fn primaryRay(fragCoord : vec2<f32>) -> Primary {
  var r : Primary;
  r.uv = (fragCoord / U.resolution) * 2.0 - 1.0;
  r.dir = normalize(
    U.camFwd +
    U.camRight * (r.uv.x * U.aspect * U.tanHalfFov) +
    U.camUp * (-r.uv.y * U.tanHalfFov),
  );
  r.ro = U.camPos;
  return r;
}

// --- temporal accumulation (RenderQuality.temporal; see temporal.wesl) ---------------------------
//
// The same shading, with the key shadow, AO and point shadows taken from the pixel's history where
// it still sees the same face. A tile (one workgroup) traces a term when any of its pixels must
// (no history, or a change there) or, on its turn (1 frame in T.rate), when any has not yet
// converged; the rays cost the whole tile anyway (its lanes run in lockstep), so every pixel in a
// tracing tile takes the sample. Converged pixels hold their value, so a still image is stable.

// Bits of the terms some pixel of this tile needs traced now (1 key, 2 AO, 4 point), and those
// wanted on the tile's turn (the same, shifted by 3).
var<workgroup> tileNeeds : atomic<u32>;

fn temporalPixel(px : vec2<u32>, tile : vec2<u32>) -> vec4<f32> {
  // Only what the shading needs after the rays stays live across them (the hit's cell, normal,
  // material and distance, and the raw history texel): the walks, with instance sampling inside,
  // are where registers bind. The ray and the colour are recomputed afterwards.
  let inside = f32(px.x) < U.resolution.x && f32(px.y) < U.resolution.y;
  var hit = false;
  var cell = vec3<i32>(0);
  var nrm = vec3<f32>(0.0);
  var mat = 0u;
  var t = 0.0;
  var hv = vec4<u32>(0u);
  var hash = 0u;
  var terms = 0u;
  var fresh = false;
  if (inside) {
    let r = primaryRay(vec2<f32>(px) + vec2<f32>(0.5));
    let box = intersectBox(r.ro, r.dir);
    let tNear = max(box.x, 0.0);
    if (box.y >= tNear) {
      let h = traverse(r.ro, r.dir, tNear);
      hit = h.hit;
      cell = h.cell;
      nrm = h.normal;
      mat = h.mat;
      t = h.t;
    }
    if (hit) {
      hash = faceHash(cell, nrm);
      hv = lookupHistory(r.ro + r.dir * t, hash, nrm);
      fresh = hv.x == 0u || changedAt(cell);
      // The terms this face has: a key shadow if it faces the light, AO, point shadows.
      terms = select(0u, 1u, U.shadowSteps > 0.5 && dot(nrm, U.lightDir) > 0.0) | select(0u, 2u, U.aoEnabled > 0.5) |
        select(0u, 4u, U.lightCount > 0.5 && (T.flags & FLAG_POINT_SHADOWS) != 0u);
      var need = select(0u, terms, fresh);
      need |= terms & (select(0u, 1u, (T.flags & FLAG_FORCE_KEY) != 0u) | select(0u, 4u, (T.flags & FLAG_FORCE_POINT) != 0u));
      let open = !fresh && ((hv.z >> 26u) < u32(T.cap) || (T.flags & FLAG_HOLD) == 0u);
      let want = select(0u, terms, open);
      if ((need | (want << 3u)) != 0u) {
        atomicOr(&tileNeeds, need | (want << 3u));
      }
    }
  }
  workgroupBarrier();
  let bits = atomicLoad(&tileNeeds);
  if (!inside) {
    return vec4<f32>(0.0);
  }
  if (!hit) {
    textureStore(histOut, px, vec4<u32>(0u));
    let r = primaryRay(vec2<f32>(px) + vec2<f32>(0.5));
    return vec4<f32>(missColor(r.ro, r.dir, vec2<f32>(r.uv.x, -r.uv.y)), 1.0);
  }

  let turn = (((tile.x & 1u) * 2u + (tile.y & 1u) * 3u + T.frame) % max(T.rate, 1u)) == 0u;
  let traced = (bits | select(0u, bits >> 3u, turn)) & 7u & terms;
  let forceKey = (T.flags & FLAG_FORCE_KEY) != 0u;
  let forcePoint = (T.flags & FLAG_FORCE_POINT) != 0u;
  // A pixel starting over takes the unjittered sample (today's hard shadow, exactly); the soft
  // shadow builds up from the jittered ones after it. The face's next sample is its count; once
  // converged (a moving light keeps sampling), the frame number.
  let n0 = hv.z >> 26u;
  let k = select(n0, T.frame, n0 >= u32(T.cap));
  let hitCenter = vec3<f32>(cell) + vec3<f32>(0.5) + nrm * 0.5;

  // AO: the same every sample (it has no noise); history saves the lookups.
  var sAo = 1.0;
  if ((traced & 2u) != 0u) {
    sAo = faceAO(cell, nrm);
  }
  // Key light: visibility 0 (shadowed) .. 1.
  var sKey = 1.0;
  if ((traced & 1u) != 0u) {
    let L = normalize(U.lightDir);
    var l = L;
    if (!fresh && !forceKey) {
      let lj = coneDir(L, faceSample(hash, k), T.spread);
      if (dot(nrm, lj) > 0.0) { l = lj; }
    }
    sKey = select(1.0, 0.0, inShadow(hitCenter, l));
  }
  // Point lights: shadowed over unshadowed irradiance, applied to this frame's unshadowed.
  var jv = vec3<f32>(0.0);
  if (!fresh && !forcePoint && (traced & 4u) != 0u) {
    // A point on the light's sphere.
    let v = faceSample(hash ^ 0x9e3779b9u, k);
    let z = 2.0 * v.x - 1.0;
    let rxy = sqrt(max(1.0 - z * z, 0.0));
    jv = vec3<f32>(rxy * cos(6.2831853 * v.y), z, rxy * sin(6.2831853 * v.y)) * T.lampRadius;
  }
  let ps = pointLightingSplit(hitCenter, nrm, (traced & 4u) != 0u, jv);

  // Fold the samples into the history.
  var hist : History;
  hist.valid = false;
  hist.count = 0u;
  if (hv.x != 0u) { hist = unpackHistory(hv); }
  let blend = !fresh && (hist.count < u32(T.cap) || (T.flags & FLAG_HOLD) == 0u);
  let count = select(min(hist.count + 1u, u32(T.cap)), 1u, fresh);
  let w = 1.0 / f32(max(count, 1u));
  var next : History;
  next.valid = true;
  next.count = select(hist.count, count, fresh || (blend && traced != 0u));
  next.key = select(1.0, hist.key, hist.valid && (terms & 1u) != 0u);
  if ((traced & 1u) != 0u) {
    if (fresh || forceKey) { next.key = sKey; } else if (blend) { next.key = mix(hist.key, sKey, w); }
  }
  next.ao = select(1.0, hist.ao, hist.valid && (terms & 2u) != 0u);
  if ((traced & 2u) != 0u) {
    next.ao = sAo;
  }
  next.ratio = select(vec3<f32>(1.0), hist.ratio, hist.valid && (terms & 4u) != 0u);
  if ((traced & 4u) != 0u) {
    let s = select(vec3<f32>(1.0), ps.lit / max(ps.open, vec3<f32>(1e-6)), ps.open > vec3<f32>(1e-6));
    if (fresh || forcePoint) { next.ratio = s; } else if (blend) { next.ratio = mix(hist.ratio, s, w); }
  }
  textureStore(histOut, px, packHistory(hash, t, sideOf(nrm), next));

  var h : Hit;
  h.hit = true;
  h.cell = cell;
  h.normal = nrm;
  h.mat = mat;
  h.color = palette[mat].rgb;
  h.t = t;
  let c = shadeHit(primaryRay(vec2<f32>(px) + vec2<f32>(0.5)), h, mix(0.6, 1.0, next.key), next.ao, ps.free + ps.open * next.ratio);
  if (T.show > 1.5) {
    // Development view 2: samples averaged (black none .. white converged); red where no history
    // was found, blue where a change stamp forced a restart.
    if (hv.x == 0u) { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }
    if (fresh) { return vec4<f32>(0.0, 0.0, 1.0, 1.0); }
    return vec4<f32>(vec3<f32>(f32(next.count) / T.cap), 1.0);
  }
  if (T.show > 0.5) {
    // Development view 1: what this tile traced, red key, green AO, blue point; white where the
    // pixel started over.
    let tint = vec3<f32>(f32(traced & 1u), f32((traced >> 1u) & 1u), f32((traced >> 2u) & 1u));
    return vec4<f32>(mix(c.rgb * 0.5, select(tint, vec3<f32>(1.0), fresh), 0.5), 1.0);
  }
  return c;
}

// The temporal kernels, one per output as for the plain ones: the frame texture, or the canvas.
@compute @workgroup_size(WG_X, WG_Y)
fn trace_temporal(@builtin(global_invocation_id) id : vec3<u32>, @builtin(workgroup_id) wg : vec3<u32>) {
  let c = temporalPixel(id.xy, wg.xy);
  if (id.x < u32(U.resolution.x) && id.y < u32(U.resolution.y)) {
    textureStore(frameOut, id.xy, c);
  }
}

@compute @workgroup_size(WG_X, WG_Y)
fn trace_temporal_bgra(@builtin(global_invocation_id) id : vec3<u32>, @builtin(workgroup_id) wg : vec3<u32>) {
  let c = temporalPixel(id.xy, wg.xy);
  if (id.x < u32(U.resolution.x) && id.y < u32(U.resolution.y)) {
    textureStore(canvasBgra, id.xy, c);
  }
}

@compute @workgroup_size(WG_X, WG_Y)
fn trace_temporal_rgba(@builtin(global_invocation_id) id : vec3<u32>, @builtin(workgroup_id) wg : vec3<u32>) {
  let c = temporalPixel(id.xy, wg.xy);
  if (id.x < u32(U.resolution.x) && id.y < u32(U.resolution.y)) {
    textureStore(canvasRgba, id.xy, c);
  }
}

/** The colour of the pixel whose centre is `fragCoord.xy` (in pixels, as @builtin(position) gives it). */
fn shadePixel(fragCoord : vec4<f32>) -> vec4<f32> {
  if (U.debug > 6.5 && U.debug < 7.5) {
    return vec4<f32>(selfTest(fragCoord.xy), 1.0);
  }
  if (U.debug > 7.5 && U.debug < 8.5) {
    return vec4<f32>(selfTest2(fragCoord.xy), 1.0);
  }
  if (U.debug > 8.5 && U.debug < 9.5) {
    return vec4<f32>(selfTest3(fragCoord.xy), 1.0);
  }
  let r = primaryRay(fragCoord.xy);
  let uv = r.uv;
  let dir = r.dir;
  let ro = r.ro;

  let box = intersectBox(ro, dir);
  let tNear = max(box.x, 0.0);
  if (box.y < tNear) {
    return vec4<f32>(missColor(ro, dir, vec2<f32>(uv.x, -uv.y)), 1.0);
  }

  let h = traverse(ro, dir, tNear);
  if (U.debug > 9.5 && U.debug < 10.5) {
    // Step heatmap: blue (few) through green and yellow to red (the step cap), for finding what a
    // scene's primary rays spend their time on.
    let f = clamp(f32(h.steps) / max(U.maxSteps, 1.0), 0.0, 1.0);
    return vec4<f32>(clamp(vec3<f32>(2.0 * f - 0.5, 1.5 - abs(2.0 * f - 1.0) * 1.5, 1.0 - 2.0 * f), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
  }
  if (!h.hit) {
    return vec4<f32>(missColor(ro, dir, vec2<f32>(uv.x, -uv.y)), 1.0);
  }

  let L = normalize(U.lightDir);
  let n = h.normal;
  let ndl = max(dot(n, L), 0.0);

  // Shadow ray from the centre of the hit face toward the light.
  let hitCenter = vec3<f32>(h.cell) + vec3<f32>(0.5) + n * 0.5;
  var shadow = 1.0;
  if (U.shadowSteps > 0.5 && ndl > 0.0 && inShadow(hitCenter, L)) {
    shadow = 0.6;
  }

  var ao = 1.0;
  if (U.aoEnabled > 0.5) {
    ao = faceAO(h.cell, n);
  }

  // Point lights, each with its own range-limited shadow ray.
  var point = vec3<f32>(0.0);
  if (U.lightCount > 0.5) {
    point = pointLighting(hitCenter, n, true);
  }
  return shadeHit(r, h, shadow, ao, point);
}

/**
 * The colour of a primary hit, given its traced terms: `shadow` the key light's factor (0.6 in
 * shadow .. 1 lit), `ao` the face occlusion (1 open) and `point` the point lights' irradiance.
 * Everything else (ambient, materials, water, fog, weather) is evaluated here, every frame.
 */
fn shadeHit(r : Primary, h : Hit, shadow : f32, ao : f32, point : vec3<f32>) -> vec4<f32> {
  let uv = r.uv;
  let dir = r.dir;
  let ro = r.ro;
  let L = normalize(U.lightDir);
  let n = h.normal;
  let ndl = max(dot(n, L), 0.0);
  let hitCenter = vec3<f32>(h.cell) + vec3<f32>(0.5) + n * 0.5;

  // Hemispheric ambient (sky above, ground bounce below) — time-of-day tinted
  // via uniforms so the scene shifts warm/day → cool/night.
  let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
  let ambient = mix(U.ambientGround, U.ambientSky, up) * 0.85;

  // Soft fill from the camera side so the shadowed half doesn't go muddy.
  let fill = max(dot(n, -U.camFwd), 0.0) * 0.14;

  // Surface state: wet ground is darker; settled snow covers upward faces
  // that are open to the sky. Water is neither.
  let isWater = U.materialsEnabled > 0.5 && abs(matOf(h.mat).kind - 4.0) < 0.5;
  var albedo = h.color;
  var coverAmt = 0.0;
  if (!isWater) {
    if (U.cover > 0.0 && n.y > 0.5) {
      let jitter = hash31(vec3<f32>(f32(h.cell.x), 0.0, f32(h.cell.z))) - 0.5;
      coverAmt = smoothstep(0.1, 0.7, U.cover + jitter * 0.35);
      // Sheltered ground (under a canopy or a roof) stays mostly bare. The
      // upward check costs a short ray, so it follows the shadow budget.
      if (coverAmt > 0.0 && U.shadowSteps > 0.5 && occludedTo(hitCenter, vec3<f32>(0.0, 1.0, 0.0), 48.0 * max(U.effectScale, 1e-3))) {
        coverAmt *= 0.12;
      }
      albedo = mix(albedo, U.coverColor, coverAmt);
    }
    if (U.wet > 0.0) {
      albedo *= 1.0 - 0.3 * U.wet * (1.0 - coverAmt);
    }
  }

  // Key light tinted by the sun/moon colour.
  var color = albedo * (ambient + (ndl * 0.55 * shadow + fill) * U.lightColor) * ao;
  // Point lights (traced by the caller).
  if (U.lightCount > 0.5) {
    color += albedo * point * ao;
  }
  // Wet upward faces pick up a soft sheen from the key light.
  if (U.wet > 0.0 && !isWater && n.y > 0.5) {
    let sheen = pow(max(dot(reflect(dir, n), L), 0.0), 24.0);
    color += U.lightColor * sheen * 0.35 * U.wet * shadow * (1.0 - coverAmt);
  }

  // Material shading (emissive / metal reflection / glass transmission). Gated by
  // materialsEnabled so flat-palette scenes are byte-for-byte unchanged.
  if (U.materialsEnabled > 0.5) {
    let m = matOf(h.mat);
    // Water: its own shading replaces the diffuse one entirely.
    if (abs(m.kind - 4.0) < 0.5) {
      let w = shadeWater(ro, dir, h.cell, n, h.color, m.att);
      // The reflection is the one bounce this pass allows; skip it when it
      // would barely show, which is most of a surface seen from above.
      var reflCol = background(w.point, w.reflectDir, vec2<f32>(0.0));
      if (w.fresnel > 0.06) { reflCol = traceLit(w.point + w.normal * 0.6, w.reflectDir); }
      color = mix(w.under, reflCol, clamp(w.fresnel, 0.0, 1.0)) + w.glint * shadow;
    }
    // Emissive self-glow.
    color += h.color * m.emit;
    // Metal: a single (optionally roughened) reflection bounce.
    if (m.metal > 0.01) {
      var rdir = reflect(dir, n);
      if (m.rough > 0.01) {
        let j = m.rough * 0.5;
        rdir = normalize(rdir + vec3<f32>(
          (hash31(hitCenter) - 0.5) * j,
          (hash31(hitCenter + 3.1) - 0.5) * j,
          (hash31(hitCenter + 7.7) - 0.5) * j));
      }
      let rc = traceLit(hitCenter, rdir);
      // Tint the reflection by the metal's own colour (coloured metals).
      color = mix(color, rc * mix(vec3<f32>(1.0), h.color, 0.7), clamp(m.metal, 0.0, 1.0));
    }
    // Glass: transmit past the voxel (refracted), tinted, with a Fresnel reflection.
    if (m.kind > 1.5 && m.kind < 2.5) {
      let eta = 1.0 / max(1.0 + m.ior, 1.0);
      var tdir = refract(dir, n, eta);
      if (dot(tdir, tdir) < 1e-4) { tdir = dir; } // total internal reflection → straight
      let tro = vec3<f32>(h.cell) + vec3<f32>(0.5) + dir * 2.0; // start past the entry face
      let through = traceLit(tro, tdir) * mix(vec3<f32>(1.0), h.color, clamp(m.att + 0.35, 0.0, 1.0));
      let refl = traceLit(hitCenter, reflect(dir, n));
      let fres = pow(1.0 - max(dot(-dir, n), 0.0), 3.0) * 0.5;
      let glass = mix(through, refl, fres);
      color = mix(glass, color, clamp(m.alpha, 0.0, 1.0)); // alpha=0 clear → all glass
    }
  }

  // Highlight mode: spotlight the selection (full colour + bright rim), grey the rest.
  if (U.highlight.x > 0.5) {
    let isSel = h.mat >= u32(U.highlight.y) && h.mat <= u32(U.highlightMax);
    if (isSel) {
      let edge = highlightEdge(h.cell);
      color = mix(color, vec3<f32>(1.0, 0.96, 0.62), edge * 0.7);
    } else {
      let g = dot(color, vec3<f32>(0.299, 0.587, 0.114));
      color = mix(vec3<f32>(g), color, 0.12) * 0.5;
    }
  }

  // Atmosphere in front of the hit: fog, the lamps in front of it glowing through their own
  // share of it, then falling rain or snow.
  let tHit = length(hitCenter - ro);
  color = applyFog(color, ro, dir, tHit);
  if (U.lightCount > 0.5) {
    color += lightGlow(ro, dir, tHit);
  }
  let pr = precipitation(ro, dir, tHit);
  color = mix(color, pr.rgb, pr.a);

  // Subtle vignette.
  let v = 1.0 - 0.16 * dot(uv, uv);
  color *= v;

  return vec4<f32>(pow(color, vec3<f32>(0.92)), 1.0);
}
