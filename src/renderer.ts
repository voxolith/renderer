// Owns the WebGPU pipeline and resources for the fullscreen voxel raymarch pass.

import { link } from "wesl";
import uniformsWesl from "./shaders/uniforms.wesl?raw";
import gridWesl from "./shaders/grid.wesl?raw";
import highlightWesl from "./shaders/highlight.wesl?raw";
import intersectWesl from "./shaders/intersect.wesl?raw";
import skyWesl from "./shaders/sky.wesl?raw";
import traceWesl from "./shaders/trace.wesl?raw";
import shadowWesl from "./shaders/shadow.wesl?raw";
import aoWesl from "./shaders/ao.wesl?raw";
import backgroundWesl from "./shaders/background.wesl?raw";
import materialsWesl from "./shaders/materials.wesl?raw";
import raymarchWesl from "./shaders/raymarch.wesl?raw";
import type { GpuContext } from "./device";
import { COARSE_B } from "./occupancy";

// WESL modules of the raymarch pass, linked once into the final WGSL. Keys are
// the modules' relative paths (./foo.wesl → import path `package::foo`).
const WESL_SRC: Record<string, string> = {
  "./uniforms.wesl": uniformsWesl,
  "./grid.wesl": gridWesl,
  "./highlight.wesl": highlightWesl,
  "./intersect.wesl": intersectWesl,
  "./sky.wesl": skyWesl,
  "./trace.wesl": traceWesl,
  "./shadow.wesl": shadowWesl,
  "./ao.wesl": aoWesl,
  "./background.wesl": backgroundWesl,
  "./materials.wesl": materialsWesl,
  "./raymarch.wesl": raymarchWesl,
};

// Material storage buffer: 256 slots × 2 vec4 (8 f32 each). See materials.wesl.
const MATERIAL_FLOATS = 256 * 8;

let shaderCodePromise: Promise<string> | null = null;
/** Link the WESL modules into WGSL (once; cached for all renderers). */
export function raymarchShaderCode(): Promise<string> {
  if (!shaderCodePromise) {
    shaderCodePromise = link({ weslSrc: WESL_SRC, rootModuleName: "./raymarch.wesl" })
      .then((linked) => linked.dest);
  }
  return shaderCodePromise;
}

// Uniform buffer layout: 104 f32 (416 bytes). See struct Uniforms in the shader
// for the exact float map (camera 0..31, environment/sky 32..71, occ/coarse
// 76..91, optional ground-plane floor 92..103).
const UNIFORM_FLOATS = 104;

/** Minimal scene data the renderer needs to build/upload the voxel grid. */
export interface RenderScene {
  size: { x: number; y: number; z: number };
  data: Uint8Array;
  palette: Float32Array;
  /** Optional per-slot materials (256×8 f32); enables material shading. */
  materials?: Float32Array;
}

/** Inclusive voxel-space bounding box for a partial grid update. */
export interface DirtyBox {
  x0: number;
  y0: number;
  z0: number;
  x1: number;
  y1: number;
  z1: number;
}

/** Optional offscreen render destination (defaults to the canvas swapchain). */
export interface RenderTarget {
  view: GPUTextureView;
  width: number;
  height: number;
}

type Vec3 = [number, number, number];

export interface FrameParams {
  // Camera (from camera.ts).
  camPos: Vec3;
  camRight: Vec3;
  camUp: Vec3;
  camFwd: Vec3;
  tanHalfFov: number;
  // Environment lighting + sky. Generic: the caller supplies the values (a game
  // may drive them from a day/night clock, or pass static constants).
  lightDir: Vec3;
  lightColor: Vec3;
  ambientSky: Vec3;
  ambientGround: Vec3;
  sunDir: Vec3;
  moonDir: Vec3;
  sunColor: Vec3;
  moonColor: Vec3;
  skyTop: Vec3;
  skyHorizon: Vec3;
  nightFactor: number;
  sunIntensity: number;
  moonIntensity: number;
}

/** Optional infinite ground-plane drawn on ray-miss below `y` (off by default). */
/** Per-frame cost knobs. All are runtime uniforms; changing them is free. */
export interface RenderQuality {
  /** Primary-ray DDA step cap (16..1024). Lower is cheaper; too low clips far geometry. */
  maxSteps: number;
  /** Shadow-ray step cap (0..128). 0 disables shadows entirely. */
  shadowSteps: number;
  /** Face ambient occlusion (8 neighbour lookups per hit). */
  ao: boolean;
}

export type QualityPreset = "low" | "medium" | "high";

/** Presets consumers can offer in a UI; `high` matches the engine's original look. */
export const QUALITY_PRESETS: Record<QualityPreset, RenderQuality> = {
  low: { maxSteps: 256, shadowSteps: 0, ao: false },
  medium: { maxSteps: 512, shadowSteps: 48, ao: true },
  high: { maxSteps: 768, shadowSteps: 90, ao: true },
};

export interface FloorParams {
  enabled: boolean;
  y: number;
  colorA: Vec3;
  colorB: Vec3;
}

export class Renderer {
  private readonly gpu: GpuContext;
  private readonly pipeline: GPURenderPipeline;
  private readonly uniformBuffer: GPUBuffer;
  private readonly uniformData = new Float32Array(UNIFORM_FLOATS);
  private readonly bindGroup: GPUBindGroup;
  private readonly gridSize: [number, number, number];
  private readonly voxTexture: GPUTexture;
  private readonly paletteBuffer: GPUBuffer;
  private occMin: Vec3;
  private occMax: Vec3;
  private readonly coarseTexture: GPUTexture;
  private readonly coarseDim: Vec3;
  private readonly materialBuffer: GPUBuffer;
  private materialsEnabled = 0;
  // Generic selection highlight: when mode=1, voxels with a palette slot in
  // [slotMin,slotMax] keep colour + edge glow while the rest greys out.
  private highlight = { mode: 0, slotMin: 0, slotMax: 0 };
  private floor: FloorParams = {
    enabled: false,
    y: 0,
    colorA: [0, 0, 0],
    colorB: [0, 0, 0],
  };
  private debugMode = 0;
  private quality: RenderQuality = { ...QUALITY_PRESETS.high };

  constructor(gpu: GpuContext, scene: RenderScene, shaderCode: string) {
    this.gpu = gpu;
    const { device } = gpu;
    this.gridSize = [scene.size.x, scene.size.y, scene.size.z];
    [this.occMin, this.occMax] = occupiedBounds(scene.data, scene.size);

    const module = device.createShaderModule({ code: shaderCode });

    const voxTexture = device.createTexture({
      size: [scene.size.x, scene.size.y, scene.size.z],
      dimension: "3d",
      format: "r8uint",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.voxTexture = voxTexture;
    device.queue.writeTexture(
      { texture: voxTexture },
      scene.data,
      {
        bytesPerRow: scene.size.x,
        rowsPerImage: scene.size.y,
      },
      { width: scene.size.x, height: scene.size.y, depthOrArrayLayers: scene.size.z },
    );

    const paletteBuffer = device.createBuffer({
      size: scene.palette.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paletteBuffer, 0, scene.palette);
    this.paletteBuffer = paletteBuffer;

    // Coarse occupancy texture (empty-space skipping). Filled via updateCoarse().
    const cdx = Math.ceil(scene.size.x / COARSE_B);
    const cdy = Math.ceil(scene.size.y / COARSE_B);
    const cdz = Math.ceil(scene.size.z / COARSE_B);
    this.coarseDim = [cdx, cdy, cdz];
    const coarseTexture = device.createTexture({
      size: [cdx, cdy, cdz],
      dimension: "3d",
      format: "r8uint",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.coarseTexture = coarseTexture;
    device.queue.writeTexture(
      { texture: coarseTexture },
      new Uint8Array(cdx * cdy * cdz),
      { bytesPerRow: cdx, rowsPerImage: cdy },
      { width: cdx, height: cdy, depthOrArrayLayers: cdz },
    );

    // Material buffer (always bound). Populated from scene.materials, else zero
    // (materialsEnabled=0 → the shader keeps the flat-palette path unchanged).
    this.materialBuffer = device.createBuffer({
      size: MATERIAL_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (scene.materials) {
      device.queue.writeBuffer(this.materialBuffer, 0, scene.materials);
      this.materialsEnabled = 1;
    } else {
      device.queue.writeBuffer(this.materialBuffer, 0, new Float32Array(MATERIAL_FLOATS));
    }

    this.uniformBuffer = device.createBuffer({
      size: UNIFORM_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const layout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "uint", viewDimension: "3d" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "uint", viewDimension: "3d" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
      ],
    });

    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: { module, entryPoint: "vs" },
      fragment: {
        module,
        entryPoint: "fs",
        targets: [{ format: gpu.format }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.bindGroup = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: voxTexture.createView() },
        { binding: 2, resource: { buffer: paletteBuffer } },
        { binding: 3, resource: coarseTexture.createView() },
        { binding: 4, resource: { buffer: this.materialBuffer } },
      ],
    });
  }

  /** Re-upload the 256-entry colour palette (e.g. after a carpet swap). */
  updatePalette(palette: Float32Array): void {
    this.gpu.device.queue.writeBuffer(this.paletteBuffer, 0, palette);
  }

  /** Upload per-slot materials (256×8 f32) and enable material shading. */
  updateMaterials(materials: Float32Array): void {
    this.gpu.device.queue.writeBuffer(this.materialBuffer, 0, materials);
    this.materialsEnabled = 1;
  }

  /** Toggle/parameterise selection-highlight shading (greyout + edge highlight). */
  setHighlight(state: { mode: number; slotMin: number; slotMax: number }): void {
    this.highlight = state;
  }

  /** Configure the optional infinite ground-plane drawn on ray-miss. */
  setFloor(state: FloorParams): void {
    this.floor = state;
  }

  /** 1 = force single-step DDA (ignore coarse skip) — for the ?diff exactness check. */
  /** Set per-frame cost knobs (partial updates allowed). Takes effect next frame. */
  setQuality(q: Partial<RenderQuality> | QualityPreset): void {
    const src = typeof q === "string" ? QUALITY_PRESETS[q] : q;
    this.quality = {
      maxSteps: Math.max(16, Math.min(1024, Math.round(src.maxSteps ?? this.quality.maxSteps))),
      shadowSteps: Math.max(0, Math.min(128, Math.round(src.shadowSteps ?? this.quality.shadowSteps))),
      ao: src.ao ?? this.quality.ao,
    };
  }

  getQuality(): RenderQuality {
    return { ...this.quality };
  }

  setDebug(v: number): void {
    this.debugMode = v;
  }

  /**
   * Override the ray-clip AABB (inclusive voxel coords). Defaults to the occupied
   * bounds scanned at construction; dynamic scenes whose occupied region grows or
   * starts empty (e.g. a VFX stage) should widen it to the full grid.
   */
  setClipBounds(min: Vec3, max: Vec3): void {
    this.occMin = min;
    this.occMax = max;
  }

  /** Re-upload coarse occupancy (full, or a coarse-voxel sub-box mirroring updateVoxels). */
  updateCoarse(data: Uint8Array, box?: DirtyBox): void {
    const [cx, cy, cz] = this.coarseDim;
    const { device } = this.gpu;
    if (!box) {
      device.queue.writeTexture(
        { texture: this.coarseTexture },
        data,
        { bytesPerRow: cx, rowsPerImage: cy },
        { width: cx, height: cy, depthOrArrayLayers: cz },
      );
      return;
    }
    const bw = box.x1 - box.x0 + 1;
    const bh = box.y1 - box.y0 + 1;
    const bd = box.z1 - box.z0 + 1;
    device.queue.writeTexture(
      { texture: this.coarseTexture, origin: { x: box.x0, y: box.y0, z: box.z0 } },
      data,
      { offset: box.x0 + box.y0 * cx + box.z0 * cx * cy, bytesPerRow: cx, rowsPerImage: cy },
      { width: bw, height: bh, depthOrArrayLayers: bd },
    );
  }

  /**
   * Re-upload voxel data. With a `box`, only that sub-region is sent (a strided
   * copy straight out of the full-grid `data`, no scratch buffer); otherwise the
   * whole grid is uploaded.
   */
  updateVoxels(data: Uint8Array, box?: DirtyBox): void {
    const [sx, sy] = this.gridSize;
    if (!box) {
      this.gpu.device.queue.writeTexture(
        { texture: this.voxTexture },
        data,
        { bytesPerRow: sx, rowsPerImage: sy },
        { width: sx, height: sy, depthOrArrayLayers: this.gridSize[2] },
      );
      return;
    }
    const bw = box.x1 - box.x0 + 1;
    const bh = box.y1 - box.y0 + 1;
    const bd = box.z1 - box.z0 + 1;
    this.gpu.device.queue.writeTexture(
      { texture: this.voxTexture, origin: { x: box.x0, y: box.y0, z: box.z0 } },
      data,
      {
        offset: box.x0 + box.y0 * sx + box.z0 * sx * sy,
        bytesPerRow: sx,
        rowsPerImage: sy,
      },
      { width: bw, height: bh, depthOrArrayLayers: bd },
    );
  }

  /**
   * Draw the scene. Renders to the canvas swapchain by default, or to a
   * provided offscreen target (used for headless capture / future post passes).
   */
  render(p: FrameParams, target?: RenderTarget): void {
    const width = target?.width ?? this.gpu.width;
    const height = target?.height ?? this.gpu.height;

    const u = this.uniformData;
    u[0] = p.camPos[0]; u[1] = p.camPos[1]; u[2] = p.camPos[2];
    u[4] = p.camRight[0]; u[5] = p.camRight[1]; u[6] = p.camRight[2];
    u[8] = p.camUp[0]; u[9] = p.camUp[1]; u[10] = p.camUp[2];
    u[12] = p.camFwd[0]; u[13] = p.camFwd[1]; u[14] = p.camFwd[2];
    u[15] = p.tanHalfFov;
    u[16] = this.gridSize[0]; u[17] = this.gridSize[1]; u[18] = this.gridSize[2];
    u[19] = 0;
    u[20] = p.lightDir[0]; u[21] = p.lightDir[1]; u[22] = p.lightDir[2];
    u[23] = width / height;
    u[24] = width; u[25] = height;
    u[26] = this.highlight.mode;
    u[27] = this.highlight.slotMin;
    u[28] = this.highlight.slotMax;
    u[29] = this.materialsEnabled;
    // Environment/sky block (each vec3 on a 16-byte boundary; mirrors WGSL).
    const w3 = (o: number, v: Vec3) => { u[o] = v[0]; u[o + 1] = v[1]; u[o + 2] = v[2]; };
    w3(32, p.sunDir); w3(36, p.moonDir);
    w3(40, p.sunColor); w3(44, p.moonColor);
    w3(48, p.skyTop); w3(52, p.skyHorizon);
    w3(56, p.lightColor); w3(60, p.ambientSky); w3(64, p.ambientGround);
    u[68] = p.nightFactor;
    u[69] = p.sunIntensity;
    u[70] = p.moonIntensity;
    // 72..75 reserved/padding (was screen-space sun/moon, now unused).
    w3(76, this.occMin); w3(80, this.occMax);
    w3(84, this.coarseDim);
    u[88] = this.debugMode;
    u[89] = this.quality.maxSteps;
    u[90] = this.quality.shadowSteps;
    u[91] = this.quality.ao ? 1 : 0;
    // Optional ground-plane floor (92..103).
    u[92] = this.floor.enabled ? 1 : 0;
    u[93] = this.floor.y;
    w3(96, this.floor.colorA);
    w3(100, this.floor.colorB);

    const { device } = this.gpu;
    device.queue.writeBuffer(this.uniformBuffer, 0, u);

    const view = target?.view ?? this.gpu.context.getCurrentTexture().createView();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
}

/**
 * Build a Renderer, linking the WESL shader first (cached after the first call).
 * Async because linking is async; the per-call cost after warm-up is just the
 * pipeline/texture setup. Prefer this over `new Renderer(...)`.
 */
export async function createRenderer(gpu: GpuContext, scene: RenderScene): Promise<Renderer> {
  return new Renderer(gpu, scene, await raymarchShaderCode());
}

/** Inclusive min/max voxel coords that contain any non-empty voxel (one-time scan). */
function occupiedBounds(
  data: Uint8Array,
  size: { x: number; y: number; z: number },
): [Vec3, Vec3] {
  const { x: sx, y: sy, z: sz } = size;
  let x0 = sx, y0 = sy, z0 = sz, x1 = 0, y1 = 0, z1 = 0;
  for (let z = 0; z < sz; z++)
    for (let y = 0; y < sy; y++) {
      const base = y * sx + z * sx * sy;
      for (let x = 0; x < sx; x++) {
        if (data[base + x] === 0) continue;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
        if (z < z0) z0 = z;
        if (z > z1) z1 = z;
      }
    }
  if (x1 < x0) return [[0, 0, 0], [sx - 1, sy - 1, sz - 1]]; // empty grid fallback
  return [[x0, y0, z0], [x1, y1, z1]];
}
