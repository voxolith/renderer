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
  lightCount: f32,        // 30  point lights in use (see lights.wesl)
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
  maxSteps     : f32,                   // 89     primary-ray DDA step cap (quality)
  shadowSteps  : f32,                   // 90     shadow-ray step cap; 0 = shadows off
  aoEnabled    : f32,                   // 91     1 = face ambient occlusion
  // --- optional ground-plane floor ---
  floorParams  : vec2<f32>,             // 92,93  x = enabled (0/1), y = floor Y
  _f0          : vec2<f32>,             // 94,95  pad
  floorColorA  : vec3<f32>, _f1 : f32,  // 96-99
  floorColorB  : vec3<f32>, _f2 : f32,  // 100-103
  // --- atmosphere (all off at zero; see src/atmosphere.ts) ---
  fogColor     : vec3<f32>, fogDensity : f32,     // 104-107
  cloudColor   : vec3<f32>, cloudCover : f32,     // 108-111
  precipFall   : vec3<f32>, precipDensity : f32,  // 112-115  fall velocity, voxels/s
  precipColor  : vec3<f32>, precipKind : f32,     // 116-119  0 none, 1 rain, 2 snow
  coverColor   : vec3<f32>, cover : f32,          // 120-123
  wet          : f32,                             // 124
  fogFalloff   : f32,                             // 125
  cloudDrift   : vec2<f32>,                       // 126,127
  waterWind    : vec2<f32>,                       // 128,129
  effectScale  : f32,                             // 130  voxels per unit of the effects below (1 = native)
  _w0          : f32,                             // 131
  // --- index regions (u32 word offsets into idx; see renderer.ts) ---
  topDim       : vec3<u32>, instanceCount : u32,  // 132-135  world top-level dims
  blocksOff    : u32, cellOff : u32, listOff : u32, instOff : u32, // 136-139
  modelsOff    : u32, topsOff : u32, _i0 : u32, _i1 : u32,         // 140-143
};

@group(0) @binding(0) var<uniform> U : Uniforms;
// Every index table, in regions (offsets in U): the top levels of the world
// and of each model (one entry per 64^3 cell: 0 = nothing, else block + 1),
// index blocks (512 brick entries per block: 0 = empty, else slot + 1 with bit
// 31 for 8-bit bricks, or bit 30 and a value for a uniform brick), the
// instance lists per world top cell, instances and models. The null entries
// double as the empty-space-skip structure at both levels.
@group(0) @binding(1) var<storage, read> idx : array<u32>;
@group(0) @binding(2) var<storage, read> palette : array<vec4<f32>, 256>;
// 4-bit brick payloads: 64 u32 per slot, 8 voxels per word.
@group(0) @binding(3) var<storage, read> brickVox4 : array<u32>;
// Per-slot materials: 256 × 2 vec4. See materials.wesl for the field packing.
@group(0) @binding(4) var<storage, read> materials : array<vec4<f32>, 512>;
// Per-brick palettes: 8 u32 per slot, two u16 global indices per word.
@group(0) @binding(5) var<storage, read> brickPal : array<u32>;
// 8-bit fallback payloads for bricks with more than 15 distinct values:
// 128 u32 per slot, 4 voxels per word, indices are global and need no palette.
@group(0) @binding(6) var<storage, read> brickVox8 : array<u32>;
// Point lights: 3 vec4 each — (position, range), (colour × intensity, 0),
// (shadows 0/1, glow radius, 0, 0). U.lightCount of them are live.
@group(0) @binding(7) var<storage, read> lights : array<vec4<f32>>;
// The world's top level again (one texel per 64^3 block: 0 or block + 1):
// the lookup every ray step starts with, and texture reads cache better.
@group(0) @binding(8) var topTex : texture_3d<u32>;
