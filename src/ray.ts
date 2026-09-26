/**
 * Ray helpers for picking. Build a world-space ray from a screen tap, test a
 * ray against an axis-aligned box, and walk a dense voxel grid on the CPU.
 * Shared by any game doing click selection or collision.
 *
 * @packageDocumentation
 */

import type { CameraFrame } from "./camera";

/** A point or direction in grid space: x, y (up), z, in voxels. */
export type Vec3 = [number, number, number];

/** Ray vs axis-aligned box; returns entry distance along `d` (or null on miss). */
export function rayAABB(o: Vec3, d: Vec3, min: Vec3, max: Vec3): number | null {
  let tmin = -Infinity;
  let tmax = Infinity;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(d[i]) < 1e-8) {
      if (o[i] < min[i] || o[i] > max[i]) return null;
    } else {
      let t1 = (min[i] - o[i]) / d[i];
      let t2 = (max[i] - o[i]) / d[i];
      if (t1 > t2) [t1, t2] = [t2, t1];
      tmin = Math.max(tmin, t1);
      tmax = Math.min(tmax, t2);
      if (tmin > tmax) return null;
    }
  }
  return tmax < 0 ? null : Math.max(tmin, 0);
}

/**
 * Reconstruct the world-space ray through a client (pixel) coordinate, matching
 * the shader's per-pixel ray basis. Returns the camera origin + a (non-unit) dir.
 * `clientX`/`clientY` are pointer-event coordinates; the canvas's CSS box and
 * backing-store aspect are read, so this needs a browser.
 */
export function makeRay(
  canvas: HTMLCanvasElement,
  frame: CameraFrame,
  clientX: number,
  clientY: number,
): { origin: Vec3; dir: Vec3 } {
  const rect = canvas.getBoundingClientRect();
  const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
  const ndcY = ((clientY - rect.top) / rect.height) * 2 - 1; // y-down (fragCoord)
  const aspect = canvas.width / canvas.height;
  const t = frame.tanHalfFov;
  const k = ndcX * aspect * t;
  const m = -ndcY * t;
  const dir: Vec3 = [
    frame.camFwd[0] + frame.camRight[0] * k + frame.camUp[0] * m,
    frame.camFwd[1] + frame.camRight[1] * k + frame.camUp[1] * m,
    frame.camFwd[2] + frame.camRight[2] * k + frame.camUp[2] * m,
  ];
  return { origin: [...frame.camPos] as Vec3, dir };
}

/** Where a `voxelRaycast` stopped. */
export interface VoxelHit {
  /** Distance along `dir` (in units of |dir|) to the hit face. */
  t: number;
  /** The solid cell that was hit. */
  cell: Vec3;
  /** Outward face normal of the hit (axis-aligned unit vector). */
  normal: Vec3;
}

/**
 * CPU voxel raycast (Amanatides-Woo DDA), the same walk `shaders/trace.wesl`
 * does on the GPU minus the coarse skip. `dir` need not be normalised; `t` is
 * in units of `dir` (so `origin + dir * t` is the hit point). Cells count as
 * solid when `data[i] !== 0` unless a `solid` predicate is given. Returns null
 * when nothing is hit within `maxT`.
 *
 * Headless and cheap (at most one step per cell crossed), so it suits
 * per-frame collision tests as well as picking. It walks one dense grid; a
 * scene held only as bricks has no such array to pass.
 *
 * @param size - Grid extent in voxels.
 * @param data - Dense grid, `x + y*sx + z*sx*sy`.
 * @param origin - Ray start in grid space; it may lie outside the grid.
 * @param dir - Ray direction; its length sets the unit of `t` and `maxT`.
 * @param maxT - Give up beyond `origin + dir * maxT`.
 * @param solid - Narrows which non-zero cells stop the ray (water, glass, the
 *   shooter's own voxels).
 * @returns The first solid cell and its entry face, or null.
 *
 * @example
 * ```ts
 * import { voxelRaycast, type Vec3 } from "@voxolith/renderer/core";
 *
 * // Will a ball at `p` moving by `move` this step hit anything?
 * const hit = voxelRaycast(size, data, p, move, 1);
 * if (hit) {
 *   const at: Vec3 = [p[0] + move[0] * hit.t, p[1] + move[1] * hit.t, p[2] + move[2] * hit.t];
 *   bounce(at, hit.normal);
 * }
 * ```
 */
export function voxelRaycast(
  size: { x: number; y: number; z: number },
  data: Uint8Array,
  origin: Vec3,
  dir: Vec3,
  maxT: number,
  solid: (value: number, x: number, y: number, z: number) => boolean = (v) => v !== 0,
): VoxelHit | null {
  const { x: sx, y: sy, z: sz } = size;
  // Enter the grid first if we start outside it.
  let t = 0;
  const inside =
    origin[0] >= 0 && origin[0] < sx && origin[1] >= 0 && origin[1] < sy && origin[2] >= 0 && origin[2] < sz;
  if (!inside) {
    const tEnter = rayAABB(origin, dir, [0, 0, 0], [sx, sy, sz]);
    if (tEnter === null || tEnter > maxT) return null;
    t = tEnter + 1e-6;
  }
  const px = origin[0] + dir[0] * t, py = origin[1] + dir[1] * t, pz = origin[2] + dir[2] * t;
  let cx = Math.floor(px), cy = Math.floor(py), cz = Math.floor(pz);
  const stepX = dir[0] > 0 ? 1 : dir[0] < 0 ? -1 : 0;
  const stepY = dir[1] > 0 ? 1 : dir[1] < 0 ? -1 : 0;
  const stepZ = dir[2] > 0 ? 1 : dir[2] < 0 ? -1 : 0;
  const ddx = stepX ? Math.abs(1 / dir[0]) : Infinity;
  const ddy = stepY ? Math.abs(1 / dir[1]) : Infinity;
  const ddz = stepZ ? Math.abs(1 / dir[2]) : Infinity;
  let tx = stepX ? t + ((stepX > 0 ? cx + 1 - px : px - cx) * ddx) : Infinity;
  let ty = stepY ? t + ((stepY > 0 ? cy + 1 - py : py - cy) * ddy) : Infinity;
  let tz = stepZ ? t + ((stepZ > 0 ? cz + 1 - pz : pz - cz) * ddz) : Infinity;
  let nx = 0, ny = 0, nz = 0;
  // Safety cap: no ray crosses more cells than the grid's Manhattan extent.
  const cap = sx + sy + sz + 3;
  for (let i = 0; i < cap; i++) {
    if (cx < 0 || cy < 0 || cz < 0 || cx >= sx || cy >= sy || cz >= sz) return null;
    const v = data[cx + cy * sx + cz * sx * sy];
    if (v !== 0 && solid(v, cx, cy, cz)) {
      return { t, cell: [cx, cy, cz], normal: [nx, ny, nz] };
    }
    if (tx <= ty && tx <= tz) {
      t = tx; tx += ddx; cx += stepX; nx = -stepX; ny = 0; nz = 0;
    } else if (ty <= tz) {
      t = ty; ty += ddy; cy += stepY; nx = 0; ny = -stepY; nz = 0;
    } else {
      t = tz; tz += ddz; cz += stepZ; nx = 0; ny = 0; nz = -stepZ;
    }
    if (t > maxT) return null;
  }
  return null;
}
