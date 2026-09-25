// @voxolith/renderer — a WebGPU voxel raymarching engine. Public API barrel.
//
// Browser apps import everything from "@voxolith/renderer". Node/bun tools (and
// any module a tool transitively imports) must import from the headless entry
// "@voxolith/renderer/core" (or the narrower "@voxolith/renderer/vox" and
// "@voxolith/renderer/ray") instead, to avoid pulling in the renderer's `?raw`
// shader imports, which only a bundler can resolve.

export {
  initGpu,
  resizeToDisplay,
  setRenderScale,
  showUnsupportedScreen,
  WebGPUUnsupportedError,
} from "./device";
export type { GpuContext, GpuOptions, AdapterInfo, UnsupportedOpts } from "./device";

export { Renderer, createRenderer, raymarchShaderCode, QUALITY_PRESETS, WORLD_SLOTS } from "./renderer";
export type {
  RenderScene,
  DirtyBox,
  RenderTarget,
  FrameParams,
  FloorParams,
  RenderQuality,
  QualityPreset,
  ModelSource,
  Instance,
} from "./renderer";

// Render-on-demand loop (browser only; not part of ./core).
export { makeFrameLoop, observeResize } from "./frameLoop";
export type { FrameLoop, FrameLoopOptions } from "./frameLoop";

export * from "./core";
