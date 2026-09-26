// Sparse voxel volumes: a map of 8^3 bricks.
//
// What a model too large to hold densely looks like on the CPU (a 1 cm-voxel
// tree is over a gigabyte dense and a few percent of that as bricks). The
// renderer accepts it directly (`addModel`), and the engine's EntityModel can
// carry one in place of its dense `data`.

/** Brick edge of `SparseVoxels`, in voxels. */
export const SPARSE_B = 8;
const SV = SPARSE_B * SPARSE_B * SPARSE_B;

/**
 * A voxel volume stored as a map of 8^3 bricks, for models too large to hold
 * densely. Only bricks with something in them exist. Pass it to
 * `Renderer.addModel` as `sparse`, or build it with `sparseSet` or
 * `sparseFromDense`; read it with `sparseGet`.
 */
export interface SparseVoxels {
  /** Extent in voxels. */
  size: { x: number; y: number; z: number };
  /**
   * Bricks by `bx + by * bdx + bz * bdx * bdy` (bdx = ceil(size.x / 8), ...),
   * each 512 values indexed `lx + ly * 8 + lz * 64`. Missing bricks are empty.
   */
  bricks: Map<number, Uint8Array>;
}

/** Bricks along each axis for a volume of `size`: `ceil(size / 8)`. */
export function sparseDims(size: { x: number; y: number; z: number }): [number, number, number] {
  return [Math.ceil(size.x / SPARSE_B), Math.ceil(size.y / SPARSE_B), Math.ceil(size.z / SPARSE_B)];
}

/** An empty sparse volume of `size` (the size is copied). */
export function makeSparse(size: { x: number; y: number; z: number }): SparseVoxels {
  return { size: { ...size }, bricks: new Map() };
}

/** Read a voxel; 0 outside the volume or in a missing brick. */
export function sparseGet(s: SparseVoxels, x: number, y: number, z: number): number {
  if (x < 0 || y < 0 || z < 0 || x >= s.size.x || y >= s.size.y || z >= s.size.z) return 0;
  const [dx, dy] = sparseDims(s.size);
  const b = s.bricks.get((x >> 3) + (y >> 3) * dx + (z >> 3) * dx * dy);
  return b ? b[(x & 7) + (y & 7) * 8 + (z & 7) * 64] : 0;
}

/** Set a voxel, creating its brick on demand. Out of range is ignored. */
export function sparseSet(s: SparseVoxels, x: number, y: number, z: number, v: number): void {
  if (x < 0 || y < 0 || z < 0 || x >= s.size.x || y >= s.size.y || z >= s.size.z) return;
  const [dx, dy] = sparseDims(s.size);
  const key = (x >> 3) + (y >> 3) * dx + (z >> 3) * dx * dy;
  let b = s.bricks.get(key);
  if (!b) {
    if (!v) return;
    b = new Uint8Array(SV);
    s.bricks.set(key, b);
  }
  b[(x & 7) + (y & 7) * 8 + (z & 7) * 64] = v;
}

/**
 * Convert a dense grid (`x + y*sx + z*sx*sy`) to bricks, keeping only bricks
 * with a non-zero voxel. Costs one pass over the dense array.
 */
export function sparseFromDense(size: { x: number; y: number; z: number }, data: Uint8Array): SparseVoxels {
  const s = makeSparse(size);
  const { x: sx, y: sy, z: sz } = size;
  const [dx, dy, dz] = sparseDims(size);
  for (let bz = 0; bz < dz; bz++)
    for (let by = 0; by < dy; by++)
      for (let bx = 0; bx < dx; bx++) {
        let brick: Uint8Array | null = null;
        for (let lz = 0; lz < 8; lz++) {
          const z = bz * 8 + lz;
          if (z >= sz) break;
          for (let ly = 0; ly < 8; ly++) {
            const y = by * 8 + ly;
            if (y >= sy) break;
            const row = y * sx + z * sx * sy;
            for (let lx = 0; lx < 8; lx++) {
              const x = bx * 8 + lx;
              if (x >= sx) break;
              const v = data[row + x];
              if (!v) continue;
              if (!brick) brick = new Uint8Array(SV);
              brick[lx + ly * 8 + lz * 64] = v;
            }
          }
        }
        if (brick) s.bricks.set(bx + by * dx + bz * dx * dy, brick);
      }
  return s;
}

/** Dense copy; for small models and tests only. */
export function sparseToDense(s: SparseVoxels): Uint8Array {
  const { x: sx, y: sy, z: sz } = s.size;
  const out = new Uint8Array(sx * sy * sz);
  const [dx, dy] = sparseDims(s.size);
  for (const [key, b] of s.bricks) {
    const bx = key % dx, by = Math.floor(key / dx) % dy, bz = Math.floor(key / (dx * dy));
    for (let lz = 0; lz < 8; lz++) {
      const z = bz * 8 + lz;
      if (z >= sz) break;
      for (let ly = 0; ly < 8; ly++) {
        const y = by * 8 + ly;
        if (y >= sy) break;
        for (let lx = 0; lx < 8; lx++) {
          const x = bx * 8 + lx;
          if (x >= sx) break;
          const v = b[lx + ly * 8 + lz * 64];
          if (v) out[x + y * sx + z * sx * sy] = v;
        }
      }
    }
  }
  return out;
}

/** Solid voxel count. */
export function sparseCount(s: SparseVoxels): number {
  let n = 0;
  for (const b of s.bricks.values()) for (let i = 0; i < SV; i++) if (b[i]) n++;
  return n;
}
