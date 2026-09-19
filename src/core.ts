// @voxolith/renderer — headless core. Everything that does not touch the GPU.
//
// Node/bun tools (generators, offline verifiers, CPU renderers) import from
// "@voxolith/renderer/core" so they never pull in renderer.ts, whose `?raw`
// shader imports only a bundler (Vite) can resolve. Browser code imports the
// full barrel from "@voxolith/renderer" instead.

export { parseVox, writeVox, parseVoxScene, decodeVoxRotation } from "./vox";
export type { Voxel, VoxModel, VoxScene, VoxMaterial, Placement, Mat3 } from "./vox";

export { voxSceneAnimator, packMaterials } from "./voxScene";
export type { VoxSceneAnimator } from "./voxScene";

// Minecraft .mca region format (peer to .vox; kept under src/formats/).
export { buildMinecraftRegion } from "./formats/minecraft/scene";
export type { MinecraftScene, MinecraftBuildOpts } from "./formats/minecraft/scene";

export { OccupancyGrid, COARSE_B } from "./occupancy";
export type { DirtyBox } from "./renderer";

export { makeCamera, firstPersonFrame, chaseFrame } from "./camera";
export type { CameraFrame, CameraConfig, ChaseOptions } from "./camera";

export { makeOrbitControl } from "./orbitControl";
export type { OrbitControl } from "./orbitControl";

export { makePanControl } from "./panControl";
export type { PanControl, PanOptions } from "./panControl";

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
