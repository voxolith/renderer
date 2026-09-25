// A self-test of the voxel lookup, one stage per vertical band, green when
// the stage works and red when it does not (renderer.setDebug(7)). For
// devices with no console: a screenshot shows which stage breaks.
//   1 uniforms: u32 fields read     2 index regions laid out
//   3 top-level texture has a block 4 index blocks have entries
//   5 brick payloads present        6 palette present
//   7 a column of the world has a voxel (sampleVoxel)
//   8 a ray straight down that column hits it (traverse)

import package::uniforms::{U, idx, topTex, brickVox4, palette};
import package::grid::worldEntry;
import package::grid::sampleVoxel;
import package::trace::traverse;

fn selfTest(frag : vec2<f32>) -> vec3<f32> {
  let band = i32(floor(frag.x / U.resolution.x * 8.0));
  var ok = false;
  let c = vec3<i32>(U.gridSize) / 2;
  switch band {
    case 0: { ok = U.topDim.x > 0u && U.topDim.y > 0u; }
    case 1: { ok = U.blocksOff > 0u && U.blocksOff < arrayLength(&idx); }
    case 2: {
      for (var z = 0u; z < U.topDim.z && !ok; z++) {
        for (var y = 0u; y < U.topDim.y && !ok; y++) {
          for (var x = 0u; x < U.topDim.x && !ok; x++) { ok = textureLoad(topTex, vec3<u32>(x, y, z), 0).r != 0u; }
        }
      }
    }
    case 3: { for (var i = 0u; i < 512u && !ok; i++) { ok = idx[U.blocksOff + i] != 0u; } }
    case 4: { for (var i = 0u; i < 4096u && !ok; i++) { ok = brickVox4[i] != 0u; } }
    case 5: { for (var i = 1u; i < 256u && !ok; i++) { ok = dot(palette[i].rgb, vec3<f32>(1.0)) > 0.0; } }
    case 6: { for (var y = 0; y < i32(U.gridSize.y) && !ok; y++) { ok = sampleVoxel(vec3<i32>(c.x, y, c.z)) != 0u; } }
    default: {
      let ro = vec3<f32>(f32(c.x) + 0.5, U.gridSize.y + 1.0, f32(c.z) + 0.5);
      ok = traverse(ro, vec3<f32>(0.0001, -1.0, 0.0002), 0.0).hit;
    }
  }
  let col = select(vec3<f32>(0.85, 0.12, 0.1), vec3<f32>(0.1, 0.75, 0.2), ok);
  // A dark line between bands.
  let edge = fract(frag.x / U.resolution.x * 8.0) < 0.02;
  return select(col, vec3<f32>(0.05), edge);
}

// setDebug(8): the same column read with each form of index arithmetic, to
// tell a shader-compiler fault apart from missing data.
//   1 top texel via signed shift (b >> 3)   2 via signed division (b / 8)
//   3 via unsigned division                4 signed mask (b & 7) == unsigned (b % 8)
//   5 worldEntry finds an entry            6 sampleVoxel finds a voxel
//   7 a ray down the column hits           8 (always green: reference)
fn selfTest2(frag : vec2<f32>) -> vec3<f32> {
  let band = i32(floor(frag.x / U.resolution.x * 8.0));
  var ok = false;
  let c = vec3<i32>(U.gridSize) / 2;
  let ny = i32(U.gridSize.y);
  switch band {
    case 0: { for (var y = 0; y < ny && !ok; y += 8) { let b = vec3<i32>(c.x, y, c.z) / 8; ok = textureLoad(topTex, b >> vec3<u32>(3u), 0).r != 0u; } }
    case 1: { for (var y = 0; y < ny && !ok; y += 8) { let b = vec3<i32>(c.x, y, c.z) / 8; ok = textureLoad(topTex, b / 8, 0).r != 0u; } }
    case 2: { for (var y = 0u; y < u32(ny) && !ok; y += 8u) { let b = vec3<u32>(u32(c.x), y, u32(c.z)) / 8u; ok = textureLoad(topTex, b / 8u, 0).r != 0u; } }
    case 3: {
      ok = true;
      for (var y = 0; y < ny; y++) { let b = vec3<i32>(c.x + y, y, c.z + 2 * y); ok = ok && all(vec3<u32>(b & vec3<i32>(7)) == vec3<u32>(b) % 8u); }
    }
    case 4: { for (var y = 0; y < ny && !ok; y += 8) { ok = worldEntry(vec3<i32>(c.x, y, c.z) / 8) != 0u; } }
    case 5: { for (var y = 0; y < ny && !ok; y++) { ok = sampleVoxel(vec3<i32>(c.x, y, c.z)) != 0u; } }
    case 6: {
      let ro = vec3<f32>(f32(c.x) + 0.5, U.gridSize.y + 1.0, f32(c.z) + 0.5);
      ok = traverse(ro, vec3<f32>(0.0001, -1.0, 0.0002), 0.0).hit;
    }
    default: { ok = true; }
  }
  let col = select(vec3<f32>(0.85, 0.12, 0.1), vec3<f32>(0.1, 0.75, 0.2), ok);
  let edge = fract(frag.x / U.resolution.x * 8.0) < 0.02;
  // A blue strip along the top marks this as the second test.
  return select(select(col, vec3<f32>(0.1, 0.3, 0.9), frag.y < 24.0), vec3<f32>(0.05), edge);
}
