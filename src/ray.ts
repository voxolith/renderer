// Ray helpers for picking: build a world-space ray from a screen tap, and a
// ray vs axis-aligned box test. Shared by any game doing click selection.

import type { CameraFrame } from "./camera";

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
