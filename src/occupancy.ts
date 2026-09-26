// Coarse occupancy acceleration grid for empty-space skipping. Each coarse cell
// covers a COARSE_B³ block of fine voxels and is 1 if ANY voxel in the block is
// non-empty. The shader fast-forwards its DDA across empty coarse cells, so rays
// through the big air volumes (room interior, yard, sky) skip in B-voxel jumps.
//
// The GPU no longer reads this grid: the renderer's brick index is its own
// empty-space structure (a null brick or top-level block is the skip), and
// Renderer.updateCoarse is a no-op kept for callers that still pass one. What
// remains is the CPU grid. The shader's COARSE_B in grid.wesl is the brick
// edge (8), not this.

import type { DirtyBox } from "./box";

/** Edge of an `OccupancyGrid` cell, in fine voxels. */
export const COARSE_B = 4;

/**
 * A coarse occupancy grid over a dense voxel grid: one byte per 4^3 block
 * (`COARSE_B`), 1 when any voxel in the block is non-empty. Headless.
 *
 * The renderer no longer needs one (its brick index skips empty space, and
 * `Renderer.updateCoarse` ignores what it is given); it remains for CPU tests
 * such as a quick "anything here?" before a finer scan, and for a scrolling
 * world that keeps it up to date with `shiftZ`.
 *
 * @example
 * ```ts
 * import { COARSE_B, OccupancyGrid } from "@voxolith/renderer/core";
 *
 * const occ = new OccupancyGrid(size, data);
 * // Anything in the block holding voxel (x, y, z)?
 * const i = ((x / COARSE_B) | 0) + ((y / COARSE_B) | 0) * occ.cx + ((z / COARSE_B) | 0) * occ.cx * occ.cy;
 * if (occ.data[i]) scanBlock(x, y, z);
 * // After a paint stroke inside `box`:
 * occ.updateBox(data, box);
 * ```
 */
export class OccupancyGrid {
  /** One byte per coarse cell (i, j, k) at `i + j*cx + k*cx*cy`: 1 occupied, 0 empty. */
  readonly data: Uint8Array;
  /** Coarse cells along x (`ceil(size.x / COARSE_B)`). */
  readonly cx: number;
  /** Coarse cells along y. */
  readonly cy: number;
  /** Coarse cells along z. */
  readonly cz: number;
  private readonly sx: number;
  private readonly sy: number;
  private readonly sz: number;

  /** Build from a dense grid (`x + y*sx + z*sx*sy`); a full scan. */
  constructor(size: { x: number; y: number; z: number }, fine: Uint8Array) {
    this.sx = size.x;
    this.sy = size.y;
    this.sz = size.z;
    this.cx = Math.ceil(size.x / COARSE_B);
    this.cy = Math.ceil(size.y / COARSE_B);
    this.cz = Math.ceil(size.z / COARSE_B);
    this.data = new Uint8Array(this.cx * this.cy * this.cz);
    this.rebuildAll(fine);
  }

  /** Full rebuild from the fine grid (startup + rare decorate/carpet/chair changes). */
  rebuildAll(fine: Uint8Array): void {
    const { sx, sy, sz, cx, cy } = this;
    this.data.fill(0);
    for (let z = 0; z < sz; z++)
      for (let y = 0; y < sy; y++) {
        const base = y * sx + z * sx * sy;
        for (let x = 0; x < sx; x++) {
          if (fine[base + x] === 0) continue;
          const ci =
            ((x / COARSE_B) | 0) +
            ((y / COARSE_B) | 0) * cx +
            ((z / COARSE_B) | 0) * cx * cy;
          this.data[ci] = 1;
        }
      }
  }

  /**
   * Recompute the coarse cells overlapping a fine-voxel dirty box (the cat's
   * footprint). Returns the changed region in coarse-cell coordinates.
   */
  updateBox(fine: Uint8Array, box: DirtyBox): DirtyBox {
    const { cx, cy } = this;
    const cx0 = (box.x0 / COARSE_B) | 0;
    const cy0 = (box.y0 / COARSE_B) | 0;
    const cz0 = (box.z0 / COARSE_B) | 0;
    const cx1 = (box.x1 / COARSE_B) | 0;
    const cy1 = (box.y1 / COARSE_B) | 0;
    const cz1 = (box.z1 / COARSE_B) | 0;

    for (let czi = cz0; czi <= cz1; czi++)
      for (let cyi = cy0; cyi <= cy1; cyi++)
        for (let cxi = cx0; cxi <= cx1; cxi++) {
          this.data[cxi + cyi * cx + czi * cx * cy] = this.blockOccupied(
            fine, cxi, cyi, czi,
          );
        }
    return { x0: cx0, y0: cy0, z0: cz0, x1: cx1, y1: cy1, z1: cz1 };
  }

  /**
   * Shift the whole coarse grid toward −Z by `nFine` fine voxels (a scrolling
   * world's "recenter": the fine grid was just moved the same way with
   * copyWithin), then recompute the freed front rows from `fine`. `nFine` must be
   * a multiple of COARSE_B. Far cheaper than rebuildAll for a large grid — the
   * cost is O(freed rows), not O(whole grid).
   */
  shiftZ(fine: Uint8Array, nFine: number): void {
    if (nFine <= 0 || nFine % COARSE_B !== 0) throw new Error("shiftZ: nFine must be a positive multiple of COARSE_B");
    const { cx, cy, cz } = this;
    const nc = nFine / COARSE_B; // coarse rows to shift
    if (nc >= cz) { this.rebuildAll(fine); return; }
    const layer = cx * cy;
    this.data.copyWithin(0, nc * layer, cz * layer); // rows [nc, cz) → [0, cz-nc)
    for (let czi = cz - nc; czi < cz; czi++)
      for (let cyi = 0; cyi < cy; cyi++)
        for (let cxi = 0; cxi < cx; cxi++)
          this.data[cxi + cyi * cx + czi * layer] = this.blockOccupied(fine, cxi, cyi, czi);
  }

  private blockOccupied(fine: Uint8Array, cxi: number, cyi: number, czi: number): number {
    const { sx, sy, sz } = this;
    const x0 = cxi * COARSE_B;
    const y0 = cyi * COARSE_B;
    const z0 = czi * COARSE_B;
    const x1 = Math.min(x0 + COARSE_B, sx);
    const y1 = Math.min(y0 + COARSE_B, sy);
    const z1 = Math.min(z0 + COARSE_B, sz);
    for (let z = z0; z < z1; z++)
      for (let y = y0; y < y1; y++) {
        const base = y * sx + z * sx * sy;
        for (let x = x0; x < x1; x++) if (fine[base + x] !== 0) return 1;
      }
    return 0;
  }
}
