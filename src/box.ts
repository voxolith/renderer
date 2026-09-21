// Inclusive voxel-space boxes, shared by the renderer, the occupancy grid and
// the stamper.
//
// This lives in its own leaf module so that the headless entry points
// (`@voxolith/renderer/core`, `/vox`, `/ray`) never pull the GPU renderer into
// their type graph. Node and bun consumers can then typecheck without the
// WebGPU or bundler ambient types.

/** Inclusive voxel-space bounding box for a partial grid update. */
export interface DirtyBox {
  x0: number;
  y0: number;
  z0: number;
  x1: number;
  y1: number;
  z1: number;
}
