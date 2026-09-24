// Per-palette-slot material lookup. The host packs 256 × 2 vec4 into the
// `materials` storage buffer (uniforms.wesl binding 4); this decodes one slot.
//   a = (kind, rough, metal, emit)   kind: 0 diffuse, 1 metal, 2 glass, 3 emit, 4 water
//   b = (ior, alpha, att, spec)      alpha: opacity (glass < 1)

import package::uniforms::materials;

struct Mat {
  kind : f32,
  rough : f32,
  metal : f32,
  emit : f32,
  ior : f32,
  alpha : f32,
  att : f32,
  spec : f32,
};

fn matOf(slot : u32) -> Mat {
  let a = materials[slot * 2u];
  let b = materials[slot * 2u + 1u];
  return Mat(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
}
