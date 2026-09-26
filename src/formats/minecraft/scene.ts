// Build a renderable voxel grid from a Minecraft region: crop a bounded box at
// 1:1 blocks (or downsampled), map blocks → an on-the-fly ≤255-colour palette,
// and produce the { size, data, palette, materials? } the raymarch renderer wants.
// Minecraft is Y-up, so no axis swap. Static (single frame), no transforms.

import { readRegion } from "./region";
import { readChunkBlocks, type ChunkBlocks } from "./chunk";
import { blockInfo, type BlockInfo } from "./blocks";

/** Which part of a region `buildMinecraftRegion` crops, and at what resolution. */
export interface MinecraftBuildOpts {
  /** Region-local block crop (0..511). Default: centred 256×256. */
  x0?: number; x1?: number; z0?: number; z1?: number;
  /** World-Y band (inclusive). Default: the occupied section range. */
  yMin?: number; yMax?: number;
  /** Cap per axis; larger crops are trimmed (top-anchored in Y). Default 320. */
  maxDim?: number;
  /** Sample every Nth block (nearest). Default 1. */
  downsample?: number;
}

/**
 * A cropped Minecraft region as a dense grid, shaped to pass straight to
 * `createRenderer` (`size`, `data`, `palette`, `materials`).
 */
export interface MinecraftScene {
  /** Grid extent in voxels (Y-up, one voxel per sampled block). */
  size: { x: number; y: number; z: number };
  /** Dense palette slots, `x + y*sx + z*sx*sy`; 0 is air. */
  data: Uint8Array;
  /** 256 RGBA floats 0..1; slots are allocated per distinct block colour. */
  palette: Float32Array;
  /** Per-slot materials (256×8 f32) when any block is emissive, metal or glass. */
  materials?: Float32Array;
  /** What was built, for a status line. */
  meta: {
    /** Chunks decoded. */
    chunks: number;
    /** Palette slots used (at most 255). */
    colours: number;
    /** Solid voxels in the grid. */
    voxels: number;
    /** The crop actually used, region-local block coordinates, inclusive. */
    crop: { x0: number; x1: number; z0: number; z1: number; yMin: number; yMax: number };
    downsample: number;
    /** True when `maxDim` trimmed the requested crop. */
    clamped: boolean;
  };
}

/**
 * Decode a Minecraft Anvil region file (`.mca`, 32×32 chunks, 512×512 blocks)
 * and crop it into a renderable grid. Blocks map to a palette of at most 255
 * colours built on the fly (a region with more distinct colours reuses the
 * last slot); air-like blocks stay empty and unknown ones get a fallback
 * colour. Minecraft is Y-up, so no axis swap is needed. Headless (it needs only
 * `DecompressionStream`); chunks decompress in parallel, hence async.
 *
 * @param bytes - The whole `.mca` file.
 * @param opts - Crop box, Y band, per-axis cap and downsampling.
 * @returns The grid, its palette and materials, and what was built.
 *
 * @example
 * ```ts
 * import { buildMinecraftRegion, createRenderer } from "@voxolith/renderer";
 *
 * const bytes = new Uint8Array(await file.arrayBuffer());
 * // A 256×256 block crop from the middle of the region, every 2nd block.
 * const scene = await buildMinecraftRegion(bytes, { x0: 128, x1: 383, z0: 128, z1: 383, downsample: 2 });
 * const renderer = await createRenderer(gpu, scene);
 * info.textContent = `${scene.meta.chunks} chunks, ${scene.meta.voxels} voxels`;
 * ```
 */
export async function buildMinecraftRegion(
  bytes: Uint8Array,
  opts: MinecraftBuildOpts = {},
): Promise<MinecraftScene> {
  const region = readRegion(bytes);
  const d = Math.max(1, Math.floor(opts.downsample ?? 1));
  const maxDim = opts.maxDim ?? 320;

  let x0 = opts.x0 ?? 128, x1 = opts.x1 ?? 383;
  let z0 = opts.z0 ?? 128, z1 = opts.z1 ?? 383;
  x0 = Math.max(0, x0); z0 = Math.max(0, z0);
  x1 = Math.min(511, x1); z1 = Math.min(511, z1);
  let clamped = false;
  // Trim footprint to the per-axis cap (× downsample).
  if ((x1 - x0 + 1) / d > maxDim) { x1 = x0 + maxDim * d - 1; clamped = true; }
  if ((z1 - z0 + 1) / d > maxDim) { z1 = z0 + maxDim * d - 1; clamped = true; }

  // Decode every chunk overlapping the crop (in parallel).
  const cx0 = x0 >> 4, cx1 = x1 >> 4, cz0 = z0 >> 4, cz1 = z1 >> 4;
  const chunks = new Map<number, ChunkBlocks>();
  const jobs: Promise<void>[] = [];
  for (let cz = cz0; cz <= cz1; cz++)
    for (let cx = cx0; cx <= cx1; cx++) {
      if (!region.present(cx, cz)) continue;
      jobs.push(
        region.chunkNbt(cx, cz).then((nbt) => {
          if (nbt) chunks.set(cx + cz * 32, readChunkBlocks(nbt));
        }),
      );
    }
  await Promise.all(jobs);

  // Y band: default to the occupied section range across decoded chunks.
  let yMin = opts.yMin, yMax = opts.yMax;
  if (yMin === undefined || yMax === undefined) {
    let lo = Infinity, hi = -Infinity;
    for (const cb of chunks.values()) { if (cb.minY < lo) lo = cb.minY; if (cb.maxY > hi) hi = cb.maxY; }
    if (!isFinite(lo)) { lo = 0; hi = 0; }
    yMin = yMin ?? lo;
    yMax = yMax ?? hi;
  }
  if ((yMax - yMin + 1) / d > maxDim) { yMin = yMax - maxDim * d + 1; clamped = true; } // keep the top

  const sx = Math.max(1, Math.ceil((x1 - x0 + 1) / d));
  const sy = Math.max(1, Math.ceil((yMax - yMin + 1) / d));
  const sz = Math.max(1, Math.ceil((z1 - z0 + 1) / d));
  const data = new Uint8Array(sx * sy * sz);

  // Colour → palette-slot cache (slot 0 = empty).
  const palette = new Float32Array(256 * 4);
  const matOfSlot: (BlockInfo["mat"] | undefined)[] = [];
  const slotOf = new Map<string, number>();
  let nextSlot = 1;
  const infoCache = new Map<number | string, BlockInfo | null>();
  const slotFor = (info: BlockInfo): number => {
    const key = `${info.rgb[0]},${info.rgb[1]},${info.rgb[2]}|${info.mat ?? ""}`;
    let s = slotOf.get(key);
    if (s === undefined) {
      if (nextSlot > 255) return 255; // out of slots → reuse last (rare with curated colours)
      s = nextSlot++;
      slotOf.set(key, s);
      palette[s * 4] = info.rgb[0] / 255;
      palette[s * 4 + 1] = info.rgb[1] / 255;
      palette[s * 4 + 2] = info.rgb[2] / 255;
      palette[s * 4 + 3] = 1;
      matOfSlot[s] = info.mat;
    }
    return s;
  };

  for (let oz = 0; oz < sz; oz++) {
    const wz = z0 + oz * d, cz = wz >> 4, lz = wz & 15;
    for (let ox = 0; ox < sx; ox++) {
      const wx = x0 + ox * d, cx = wx >> 4, lx = wx & 15;
      const cb = chunks.get(cx + cz * 32);
      if (!cb) continue;
      for (let oy = 0; oy < sy; oy++) {
        const wy = yMin + oy * d;
        const bk = cb.blockAt(lx, wy, lz);
        if (bk === 0) continue;
        let info = infoCache.get(bk);
        if (info === undefined) { info = blockInfo(bk); infoCache.set(bk, info); }
        if (!info) continue;
        data[ox + oy * sx + oz * sx * sy] = slotFor(info);
      }
    }
  }

  // Materials for slots that carry one (packed like voxScene.packMaterials).
  let hasMat = false;
  const materials = new Float32Array(256 * 8);
  for (let s = 1; s < nextSlot; s++) {
    const m = matOfSlot[s];
    if (!m) continue;
    hasMat = true;
    const o = s * 8;
    if (m === "emit") { materials[o] = 3; materials[o + 3] = 2; }
    else if (m === "metal") { materials[o] = 1; materials[o + 1] = 0.25; materials[o + 2] = 0.85; materials[o + 7] = 0.5; }
    else if (m === "glass") { materials[o] = 2; materials[o + 4] = 0.4; materials[o + 5] = 0.35; }
  }

  let voxels = 0;
  for (const v of data) if (v) voxels++;

  return {
    size: { x: sx, y: sy, z: sz },
    data,
    palette,
    materials: hasMat ? materials : undefined,
    meta: { chunks: chunks.size, colours: nextSlot - 1, voxels, crop: { x0, x1, z0, z1, yMin, yMax }, downsample: d, clamped },
  };
}
