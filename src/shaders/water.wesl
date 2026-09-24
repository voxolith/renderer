// Water: an animated, transparent surface over the voxel ground.
//
// Water is ordinary voxels whose material kind is 4, so it streams, clips and
// edits like everything else; only its shading is special. Top faces get a
// rippled normal from a few travelling waves (animated by U.time, so nothing
// is re-uploaded to make it move). From that normal the surface reflects the
// scene or the sky, refracts down to the bed — found by a second walk that
// treats water as empty — and blends the two by Fresnel. The bed fades into
// the water's own colour with depth, catches moving caustics, and the sun and
// point lights leave glints on the ripples.

import package::uniforms::U;
import package::trace::traverseUnderWater;
import package::lights::pointLighting;

// Surface height offset of the ripples at a point: three travelling waves at
// different angles and speeds, small enough to live inside one voxel.
fn ripple(p : vec2<f32>, t : f32) -> f32 {
  return 0.06 * sin(dot(p, vec2<f32>(0.21, 0.13)) + t * 1.3)
       + 0.04 * sin(dot(p, vec2<f32>(-0.17, 0.29)) + t * 1.9)
       + 0.025 * sin(dot(p, vec2<f32>(0.47, -0.36)) + t * 2.7);
}

fn rippleNormal(p : vec2<f32>, t : f32) -> vec3<f32> {
  let e = 0.35;
  let dx = ripple(p + vec2<f32>(e, 0.0), t) - ripple(p - vec2<f32>(e, 0.0), t);
  let dz = ripple(p + vec2<f32>(0.0, e), t) - ripple(p - vec2<f32>(0.0, e), t);
  return normalize(vec3<f32>(-dx / (2.0 * e) * 4.0, 1.0, -dz / (2.0 * e) * 4.0));
}

// Bright moving lines on the bed where the ripples focus the light.
fn caustic(p : vec2<f32>, t : f32) -> f32 {
  let a = sin(p.x * 0.9 + t * 1.1 + sin(p.y * 0.7 + t * 0.6) * 1.7);
  let b = sin(p.y * 0.8 - t * 0.9 + sin(p.x * 0.6 - t * 0.5) * 1.9);
  return pow(clamp(1.0 - abs(a * b), 0.0, 1.0), 6.0);
}

fn lightDiffuse(p : vec3<f32>, n : vec3<f32>) -> vec3<f32> {
  let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
  let ambient = mix(U.ambientGround, U.ambientSky, up) * 0.85;
  let ndl = max(dot(n, normalize(U.lightDir)), 0.0);
  return ambient + ndl * 0.6 * U.lightColor + pointLighting(p, n, false);
}

/**
 * Shade a water voxel hit by the primary ray. `cell`/`n` are the hit cell and
 * face normal, `waterColor` the palette colour (the colour of deep water),
 * `att` how fast the bed fades with depth. The reflection itself is traced by
 * the caller, which owns the one-bounce tracer; this returns its direction.
 */
fn shadeWater(ro : vec3<f32>, rd : vec3<f32>, cell : vec3<i32>, n : vec3<f32>, waterColor : vec3<f32>, att : f32) -> WaterShade {
  var out : WaterShade;
  // Exact point on the face that was hit.
  let c = vec3<f32>(cell);
  var hp = c + vec3<f32>(0.5) + n * 0.5;
  var wn = n;
  if (n.y > 0.5) {
    let t = (c.y + 1.0 - ro.y) / rd.y;
    hp = ro + rd * t;
    wn = rippleNormal(hp.xz, U.time);
  }
  out.point = hp;
  out.normal = wn;

  // Down through the surface to whatever is under it.
  let tdir = refract(rd, wn, 0.75);
  let tr = select(rd, normalize(tdir), dot(tdir, tdir) > 1e-4);
  let bed = traverseUnderWater(hp - wn * 0.02, tr, 0.0);
  let body = waterColor * lightDiffuse(hp, vec3<f32>(0.0, 1.0, 0.0));
  var under = body;
  if (bed.hit) {
    let bc = vec3<f32>(bed.cell) + vec3<f32>(0.5) + bed.normal * 0.5;
    let depth = length(bc - hp);
    let lit = bed.color * lightDiffuse(bc, bed.normal);
    let sunUp = clamp(normalize(U.lightDir).y, 0.0, 1.0);
    let caus = caustic(bc.xz, U.time) * 0.55 * sunUp * exp(-depth * 0.25);
    let seen = lit * (1.0 + caus * U.lightColor);
    under = mix(body, seen, exp(-depth * max(att, 0.01)));
  }
  out.under = under;

  // Sun glint off the ripples.
  let refl = reflect(rd, wn);
  let L = normalize(U.lightDir);
  out.glint = pow(max(dot(refl, L), 0.0), 90.0) * 1.6 * U.lightColor;
  out.reflectDir = refl;
  // Schlick Fresnel for water (F0 about 0.02).
  out.fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(-rd, wn), 0.0), 5.0);
  return out;
}

struct WaterShade {
  point      : vec3<f32>,
  normal     : vec3<f32>,
  under      : vec3<f32>,
  glint      : vec3<f32>,
  reflectDir : vec3<f32>,
  fresnel    : f32,
};
