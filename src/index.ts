// @voxolith/render — a WebGPU voxel raymarching engine. Public API barrel.
//
// Browser apps import everything from "@voxolith/render". Node/bun tools (and
// any module a tool transitively imports) must import from the headless entry
// "@voxolith/render/core" (or the narrower "@voxolith/render/vox" and
// "@voxolith/render/ray") instead, to avoid pulling in the renderer's `?raw`
// shader imports, which only a bundler can resolve.

export {
  initGpu,
  resizeToDisplay,
  showUnsupportedScreen,
  WebGPUUnsupportedError,
} from "./device";
export type { GpuContext, UnsupportedOpts } from "./device";

export { Renderer, createRenderer, raymarchShaderCode } from "./renderer";
export type {
  RenderScene,
  DirtyBox,
  RenderTarget,
  FrameParams,
  FloorParams,
} from "./renderer";

export * from "./core";
