// Turns a parsed extended VoxScene into a playable animation: a stable Y-up grid
// (covering all frames so it never resizes mid-playback) plus frame(i) which
// composes that frame's model placements — rotate + translate each model into
// the grid. Used by hosts that render one grid via the raymarcher (e.g. VoxView
// re-uploads frame(i) each tick with renderer.updateVoxels).

import type { VoxScene, Mat3 } from "./vox";

const KIND: Record<string, number> = { diffuse: 0, metal: 1, glass: 2, emit: 3 };

/**
 * Pack a scene's materials into the renderer's material buffer (256 × 8 f32):
 * `[kind, rough, metal, emit]`, `[ior, alpha(opacity), att, spec]`. Returns
 * undefined when the scene has no materials (so the flat-palette path is kept).
 */
export function packMaterials(scene: VoxScene): Float32Array | undefined {
  if (!scene.materials.some(Boolean)) return undefined;
  const out = new Float32Array(256 * 8);
  for (let i = 0; i < 256; i++) {
    const m = scene.materials[i];
    if (!m) continue;
    const o = i * 8;
    out[o + 0] = KIND[m.type] ?? 0;
    out[o + 1] = m.rough;
    out[o + 2] = m.type === "metal" ? (m.metal > 0 ? m.metal : m.weight) : 0;
    out[o + 3] = m.type === "emit" ? Math.max(1, m.flux, m.emit) : 0;
    out[o + 4] = m.ior;
    out[o + 5] = m.type === "glass" ? (m.alpha > 0 ? m.alpha : 0.25) : 1;
    out[o + 6] = m.att;
    out[o + 7] = m.spec;
  }
  return out;
}

/** A `VoxScene` as one Y-up grid that can be filled for any animation frame. */
export interface VoxSceneAnimator {
  /** Engine grid (Y-up), stable across all frames. */
  size: { x: number; y: number; z: number };
  /** 256 × vec4 palette (0..1). */
  palette: Float32Array;
  /** Frames in the animation; `frame` takes 0..frameCount-1. */
  frameCount: number;
  /** Fill + return the grid for animation frame `i` (a cached buffer). */
  frame(i: number): Uint8Array;
}

function mat3vec(m: Mat3, x: number, y: number, z: number): [number, number, number] {
  return [
    m[0] * x + m[1] * y + m[2] * z,
    m[3] * x + m[4] * y + m[5] * z,
    m[6] * x + m[7] * y + m[8] * z,
  ];
}

/**
 * Prepare a parsed scene for playback in one renderer: a grid sized to the
 * scene's bounds over every frame (so it never resizes mid-playback), with
 * MagicaVoxel's Z-up turned into the renderer's Y-up. Each `frame(i)` call
 * clears the grid and places that frame's models into it; upload the result
 * with `renderer.updateVoxels`. Voxel values are the scene's palette indices.
 */
export function voxSceneAnimator(scene: VoxScene): VoxSceneAnimator {
  const b = scene.bounds(); // MagicaVoxel Z-up world AABB over all frames
  // Z-up (world) → Y-up (engine grid): X=x, Y=z, Z=y (matches toViewModel).
  const gx = Math.max(1, b.max[0] - b.min[0] + 1);
  const gy = Math.max(1, b.max[2] - b.min[2] + 1);
  const gz = Math.max(1, b.max[1] - b.min[1] + 1);
  const data = new Uint8Array(gx * gy * gz);

  const palette = new Float32Array(256 * 4);
  for (let i = 1; i < 256; i++) {
    if (scene.palette[i * 4 + 3] === 0) continue;
    palette[i * 4 + 0] = scene.palette[i * 4 + 0] / 255;
    palette[i * 4 + 1] = scene.palette[i * 4 + 1] / 255;
    palette[i * 4 + 2] = scene.palette[i * 4 + 2] / 255;
    palette[i * 4 + 3] = 1;
  }

  function frame(i: number): Uint8Array {
    data.fill(0);
    for (const p of scene.sample(i)) {
      const s = p.model.size;
      const px = Math.floor(s.x / 2), py = Math.floor(s.y / 2), pz = Math.floor(s.z / 2);
      for (const v of p.model.voxels) {
        // rotate about the model pivot, then translate — all in Z-up world.
        const w = mat3vec(p.rot, v.x - px, v.y - py, v.z - pz);
        const wx = Math.round(w[0] + p.trans[0]);
        const wy = Math.round(w[1] + p.trans[1]);
        const wz = Math.round(w[2] + p.trans[2]);
        // world → grid (Y-up): gx=wx-minX, gy=wz-minZ, gz=wy-minY.
        const cx = wx - b.min[0];
        const cy = wz - b.min[2];
        const cz = wy - b.min[1];
        if (cx < 0 || cx >= gx || cy < 0 || cy >= gy || cz < 0 || cz >= gz) continue;
        data[cx + cy * gx + cz * gx * gy] = v.c;
      }
    }
    return data;
  }

  return { size: { x: gx, y: gy, z: gz }, palette, frameCount: scene.frameCount, frame };
}
