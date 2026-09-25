// A self-test of the voxel lookup, one stage per vertical band, green when
// the stage works and red when it does not (renderer.setDebug(7)). For
// devices with no console: a screenshot shows which stage breaks.
//   1 uniforms: u32 fields read     2 index regions laid out
//   3 top-level texture has a block 4 index blocks have entries
//   5 brick payloads present        6 palette present
//   7 a column of the world has a voxel (sampleVoxel)
//   8 a ray straight down that column hits it (traverse)

import package::uniforms::{U, idx, topTex, brickVox4, palette};
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
