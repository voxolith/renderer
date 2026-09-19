// Shared uniform block + resource bindings for the raymarch pass. Other modules
// import `U` / the textures / the palette from here so there's one definition.

struct Uniforms {
  camPos    : vec3<f32>,
  _p0       : f32,
  camRight  : vec3<f32>,
  _p1       : f32,
  camUp     : vec3<f32>,
  _p2       : f32,
  camFwd    : vec3<f32>,
  tanHalfFov: f32,
  gridSize  : vec3<f32>,
  time      : f32,
  lightDir  : vec3<f32>,
  aspect    : f32,
  resolution: vec2<f32>,  // 24,25
  highlight  : vec2<f32>, // 26,27  x = mode (0/1), y = highlight min slot
  highlightMax : f32,     // 28
  materialsEnabled : f32, // 29  (0 = flat palette; 1 = per-slot material shading)
  _pad1     : f32,        // 30
  _pad2     : f32,        // 31
  // --- environment/sky block (each vec3 on a 16-byte boundary) ---
  sunDir       : vec3<f32>, _s0 : f32,  // 32-35
  moonDir      : vec3<f32>, _s1 : f32,  // 36-39
  sunColor     : vec3<f32>, _s2 : f32,  // 40-43
  moonColor    : vec3<f32>, _s3 : f32,  // 44-47
  skyTop       : vec3<f32>, _s4 : f32,  // 48-51
  skyHorizon   : vec3<f32>, _s5 : f32,  // 52-55
  lightColor   : vec3<f32>, _s6 : f32,  // 56-59
  ambientSky   : vec3<f32>, _s7 : f32,  // 60-63
  ambientGround: vec3<f32>, _s8 : f32,  // 64-67
  nightFactor  : f32,                   // 68
  sunIntensity : f32,                   // 69
  moonIntensity: f32,                   // 70
  _s9          : f32,                   // 71
  _screen0     : vec2<f32>,             // 72,73  (reserved; was screen-space sun)
  _screen1     : vec2<f32>,             // 74,75
  occMin       : vec3<f32>, _o0 : f32,  // 76-79  occupied AABB (voxel coords, inclusive)
  occMax       : vec3<f32>, _o1 : f32,  // 80-83
  coarseDim    : vec3<f32>, _o2 : f32,  // 84-87  coarse occupancy grid dims
  debug        : f32,                   // 88     1 = force single-step DDA (no skip)
  _o3          : f32,                   // 89
  _o4          : f32,                   // 90
  _o5          : f32,                   // 91
  // --- optional ground-plane floor ---
  floorParams  : vec2<f32>,             // 92,93  x = enabled (0/1), y = floor Y
  _f0          : vec2<f32>,             // 94,95  pad
  floorColorA  : vec3<f32>, _f1 : f32,  // 96-99
  floorColorB  : vec3<f32>, _f2 : f32,  // 100-103
};

@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var voxTex : texture_3d<u32>;
@group(0) @binding(2) var<storage, read> palette : array<vec4<f32>, 256>;
@group(0) @binding(3) var coarseTex : texture_3d<u32>;
// Per-slot materials: 256 × 2 vec4. See materials.wesl for the field packing.
@group(0) @binding(4) var<storage, read> materials : array<vec4<f32>, 512>;
