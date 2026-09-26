/**
 * The headless core: everything that does not touch the GPU.
 *
 * Node/bun tools (generators, offline verifiers, CPU renderers) import from
 * `@voxolith/renderer/core` so they never pull in renderer.ts, whose `?raw`
 * shader imports only a bundler (Vite) can resolve. Browser code imports the
 * full barrel from `@voxolith/renderer` instead, which re-exports all of this.
 *
 * A few exports here still need a browser at run time (`makePerf` draws a DOM
 * overlay, `makeRay` reads a canvas); they import fine headless but need one
 * when called.
 *
 * @packageDocumentation
 */

export { parseVox, writeVox, parseVoxScene, decodeVoxRotation } from "./vox";
export type { Voxel, VoxModel, VoxScene, VoxMaterial, Placement, Mat3 } from "./vox";

export { voxSceneAnimator, packMaterials } from "./voxScene";
export type { VoxSceneAnimator } from "./voxScene";

// Minecraft .mca region format (peer to .vox; kept under src/formats/).
export { buildMinecraftRegion } from "./formats/minecraft/scene";
export type { MinecraftScene, MinecraftBuildOpts } from "./formats/minecraft/scene";

export { OccupancyGrid, COARSE_B } from "./occupancy";
export { BrickGrid, BrickPool, BRICK_B, TOP_B, PALETTE_ENTRIES, UNIFORM_BIT, NEAR_BIT } from "./brick";
export { SPARSE_B, makeSparse, sparseGet, sparseSet, sparseDims, sparseFromDense, sparseToDense, sparseCount } from "./sparse";
export type { SparseVoxels } from "./sparse";
export type { BrickStats, BrickEdit } from "./brick";
export type { DirtyBox } from "./box";

export { makeCamera, firstPersonFrame, chaseFrame } from "./camera";
export type { CameraFrame, CameraConfig, ChaseOptions } from "./camera";

export { MAX_LIGHTS, packLights } from "./lights";
export type { PointLight } from "./lights";

export type { AtmosphereParams, CloudParams, FogParams, PrecipitationParams, SurfaceParams } from "./atmosphere";

export { makePerf } from "./perf";
export type { Perf, PerfOptions } from "./perf";

export { rayAABB, makeRay, voxelRaycast } from "./ray";
export type { Vec3, VoxelHit } from "./ray";

export { seededRandom, hashSeed } from "./random";

export { GridStamper } from "./stamper";
export type { StampVoxel } from "./stamper";

export { makeExplosion, makeMuzzleFlash } from "./effects";
export type {
  VoxEffect,
  ExplosionOpts,
  ExplosionSize,
  MuzzleFlashOpts,
  MuzzleType,
} from "./effects";
