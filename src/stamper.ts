// Generic dirty-box compositor over a static base grid. Keeps a live copy of the
// base; each stamp() restores the previous stamped region from the base (so it
// never punches holes in static geometry), writes the given world voxels, and
// returns the union dirty box (old ∪ new) for a partial re-upload. Batch all
// moving objects (e.g. RTS units) into one stamp() call per frame.

import type { DirtyBox } from "./renderer";

export interface StampVoxel {
  x: number;
  y: number;
  z: number;
  /** Palette slot to write. */
  c: number;
}

export class GridStamper {
  readonly liveData: Uint8Array;
  private prev: DirtyBox | null = null;
  private readonly sx: number;
  private readonly sy: number;
  private readonly sz: number;

  constructor(
    private baseData: Uint8Array,
    private readonly size: { x: number; y: number; z: number },
  ) {
    this.liveData = baseData.slice();
    this.sx = size.x;
    this.sy = size.y;
    this.sz = size.z;
  }

  /** Adopt a new clean plate; mirror it into liveData and drop the stale stamp. */
  setBase(base: Uint8Array): void {
    this.baseData = base;
    this.liveData.set(base);
    this.prev = null;
  }

  /** Read the base plate (static geometry) at a cell; 0 outside the grid. */
  baseAt(x: number, y: number, z: number): number {
    if (x < 0 || x >= this.sx || y < 0 || y >= this.sy || z < 0 || z >= this.sz) return 0;
    return this.baseData[x + y * this.sx + z * this.sx * this.sy];
  }

  /**
   * Permanently edit the base plate (carve with c = 0, or add rubble) and mirror
   * it into liveData, without the O(grid) copy of setBase(). Cells inside the
   * current stamp region are re-covered by the next stamp() anyway. Returns the
   * box to upload, or null if nothing was in range.
   */
  writeBase(voxels: StampVoxel[]): DirtyBox | null {
    let x0 = Infinity, y0 = Infinity, z0 = Infinity;
    let x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    for (const v of voxels) {
      if (v.x < 0 || v.x >= this.sx || v.y < 0 || v.y >= this.sy || v.z < 0 || v.z >= this.sz)
        continue;
      const i = v.x + v.y * this.sx + v.z * this.sx * this.sy;
      this.baseData[i] = v.c;
      this.liveData[i] = v.c;
      if (v.x < x0) x0 = v.x;
      if (v.x > x1) x1 = v.x;
      if (v.y < y0) y0 = v.y;
      if (v.y > y1) y1 = v.y;
      if (v.z < z0) z0 = v.z;
      if (v.z > z1) z1 = v.z;
    }
    return x1 >= x0 ? { x0, y0, z0, x1, y1, z1 } : null;
  }

  /** Restore the last stamped region from the base and forget it. */
  clear(): DirtyBox | null {
    if (!this.prev) return null;
    this.restore(this.prev);
    const b = clamp(this.prev, this.size);
    this.prev = null;
    return b;
  }

  /** Restore the previous region, stamp `voxels`, return the union dirty box. */
  stamp(voxels: StampVoxel[]): DirtyBox | null {
    if (this.prev) this.restore(this.prev);

    let x0 = Infinity, y0 = Infinity, z0 = Infinity;
    let x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    for (const v of voxels) {
      if (v.x < 0 || v.x >= this.sx || v.y < 0 || v.y >= this.sy || v.z < 0 || v.z >= this.sz)
        continue;
      this.liveData[v.x + v.y * this.sx + v.z * this.sx * this.sy] = v.c;
      if (v.x < x0) x0 = v.x;
      if (v.x > x1) x1 = v.x;
      if (v.y < y0) y0 = v.y;
      if (v.y > y1) y1 = v.y;
      if (v.z < z0) z0 = v.z;
      if (v.z > z1) z1 = v.z;
    }

    const stamped = x1 >= x0 ? { x0, y0, z0, x1, y1, z1 } : null;
    const dirty = mergeBoxes(stamped, this.prev);
    this.prev = stamped;
    return dirty ? clamp(dirty, this.size) : null;
  }

  private restore(b: DirtyBox): void {
    const { sx, sy } = this;
    for (let z = b.z0; z <= b.z1; z++)
      for (let y = b.y0; y <= b.y1; y++) {
        const base = y * sx + z * sx * sy;
        for (let x = b.x0; x <= b.x1; x++) {
          const i = base + x;
          this.liveData[i] = this.baseData[i];
        }
      }
  }
}

function mergeBoxes(a: DirtyBox | null, b: DirtyBox | null): DirtyBox | null {
  if (!a) return b;
  if (!b) return a;
  return {
    x0: Math.min(a.x0, b.x0),
    y0: Math.min(a.y0, b.y0),
    z0: Math.min(a.z0, b.z0),
    x1: Math.max(a.x1, b.x1),
    y1: Math.max(a.y1, b.y1),
    z1: Math.max(a.z1, b.z1),
  };
}

function clamp(b: DirtyBox, size: { x: number; y: number; z: number }): DirtyBox {
  return {
    x0: Math.max(0, b.x0),
    y0: Math.max(0, b.y0),
    z0: Math.max(0, b.z0),
    x1: Math.min(size.x - 1, b.x1),
    y1: Math.min(size.y - 1, b.y1),
    z1: Math.min(size.z - 1, b.z1),
  };
}
