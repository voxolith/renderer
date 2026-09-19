// Root module of the raymarch pass: the fullscreen vertex shader + the fragment
// shader that ties together intersection, traversal, shadow, AO, highlight, and
// background. Imports pull in only what's used (the linker drops the rest).

import package::uniforms::U;
import package::intersect::intersectBox;
import package::trace::traverse;
import package::shadow::inShadow;
import package::ao::faceAO;
import package::background::background;
import package::highlight::highlightEdge;
import package::materials::matOf;

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
  var c = h.color * (ambient + ndl * 0.6 * U.lightColor);
  if (U.materialsEnabled > 0.5) { c += h.color * matOf(h.mat).emit; }
  return c;
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
  // NDC in [-1, 1], y up.
  let uv = (fragCoord.xy / U.resolution) * 2.0 - 1.0;
  let dir = normalize(
    U.camFwd +
    U.camRight * (uv.x * U.aspect * U.tanHalfFov) +
    U.camUp * (-uv.y * U.tanHalfFov),
  );
  let ro = U.camPos;

  let box = intersectBox(ro, dir);
  let tNear = max(box.x, 0.0);
  if (box.y < tNear) {
    return vec4<f32>(background(ro, dir, vec2<f32>(uv.x, -uv.y)), 1.0);
  }

  let h = traverse(ro, dir, tNear);
  if (!h.hit) {
    return vec4<f32>(background(ro, dir, vec2<f32>(uv.x, -uv.y)), 1.0);
  }

  let L = normalize(U.lightDir);
  let n = h.normal;
  let ndl = max(dot(n, L), 0.0);

  // Soft shadow ray from just outside the hit face toward the light.
  let hitCenter = vec3<f32>(h.cell) + vec3<f32>(0.5) + n * 0.5;
  var shadow = 1.0;
  if (U.shadowSteps > 0.5 && ndl > 0.0 && inShadow(hitCenter, L)) {
    shadow = 0.6;
  }

  var ao = 1.0;
  if (U.aoEnabled > 0.5) {
    ao = faceAO(h.cell, n);
  }

  // Hemispheric ambient (sky above, ground bounce below) — time-of-day tinted
  // via uniforms so the scene shifts warm/day → cool/night.
  let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
  let ambient = mix(U.ambientGround, U.ambientSky, up) * 0.85;

  // Soft fill from the camera side so the shadowed half doesn't go muddy.
  let fill = max(dot(n, -U.camFwd), 0.0) * 0.14;

  // Key light tinted by the sun/moon colour.
  var color = h.color * (ambient + (ndl * 0.55 * shadow + fill) * U.lightColor) * ao;

  // Material shading (emissive / metal reflection / glass transmission). Gated by
  // materialsEnabled so flat-palette scenes are byte-for-byte unchanged.
  if (U.materialsEnabled > 0.5) {
    let m = matOf(h.mat);
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

  // Subtle vignette.
  let v = 1.0 - 0.16 * dot(uv, uv);
  color *= v;

  return vec4<f32>(pow(color, vec3<f32>(0.92)), 1.0);
}
