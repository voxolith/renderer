// Owns the WebGPU pipeline and resources for the fullscreen voxel raymarch pass.

import { link } from "wesl";
import uniformsWesl from "./shaders/uniforms.wesl?raw";
import gridWesl from "./shaders/grid.wesl?raw";
import ddaWesl from "./shaders/dda.wesl?raw";
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
import type { DirtyBox } from "./box";

export type { DirtyBox };
import { BrickGrid, BRICK_B, BRICK_WORDS_4, BRICK_WORDS_8, PALETTE_WORDS } from "./brick";

// WESL modules of the raymarch pass, linked once into the final WGSL. Keys are
// the modules' relative paths (./foo.wesl → import path `package::foo`).
const WESL_SRC: Record<string, string> = {
  "./uniforms.wesl": uniformsWesl,
  "./grid.wesl": gridWesl,
  "./dda.wesl": ddaWesl,
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
  /** Primary-ray DDA step cap (16..4096). Lower is cheaper; too low clips far geometry. */
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
  private bindGroup: GPUBindGroup;
  private readonly gridSize: [number, number, number];
  private readonly paletteBuffer: GPUBuffer;
  private occMin: Vec3;
  private occMax: Vec3;
  /** Sparse mirror of the caller's dense grid; see src/brick.ts. */
  private readonly bricks: BrickGrid;
  private readonly indexTexture: GPUTexture;
  /** Brick-space dims, also the bounds test for the empty-space skip. */
  private readonly coarseDim: Vec3;
  private brickVox4: GPUBuffer;
  private brickPal: GPUBuffer;
  private brickVox8: GPUBuffer;
  private slotCap4 = 0;
  private slotCap8 = 0;
  private readonly bindLayout: GPUBindGroupLayout;
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

    // The only 3D texture is the brick index, one texel per BRICK_B^3 voxels, so
    // the per-axis ceiling is maxTextureDimension3D * BRICK_B voxels — eight
    // times what a dense grid could reach on the same adapter.
    const maxDim = gpu.limits?.maxTextureDimension3D ?? 2048;
    const maxVoxels = maxDim * BRICK_B;
    const biggest = Math.max(scene.size.x, scene.size.y, scene.size.z);
    if (biggest > maxVoxels) {
      throw new Error(
        `Scene is ${scene.size.x}x${scene.size.y}x${scene.size.z}, but this adapter's ` +
          `maxTextureDimension3D of ${maxDim} caps a grid at ${maxVoxels} voxels per axis. ` +
          `Reduce the grid, or raise the limit via initGpu({ limits: { maxTextureDimension3D } }) ` +
          `if the adapter supports more.`,
      );
    }

    // Sparse brick form of the grid. The caller's dense array stays the source
    // of truth; this is the mirror the GPU reads.
    this.bricks = new BrickGrid(scene.size, scene.data);
    this.coarseDim = [...this.bricks.dim] as Vec3;
    this.indexTexture = device.createTexture({
      size: this.bricks.dim,
      dimension: "3d",
      format: "r32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    const paletteBuffer = device.createBuffer({
      size: scene.palette.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paletteBuffer, 0, scene.palette);
    this.paletteBuffer = paletteBuffer;

    // Brick pools, sized with headroom: entities stamped in after construction
    // claim more bricks, and growPools() reallocates when they run out. The
    // contents are uploaded once the bind group exists, at the end of the ctor.
    this.slotCap4 = Math.max(256, Math.ceil(this.bricks.slotCount4 * 1.5));
    this.slotCap8 = Math.max(16, Math.ceil(this.bricks.slotCount8 * 1.5));
    this.checkPoolFits(this.slotCap4, this.slotCap8);
    this.brickVox4 = this.makePool(BRICK_WORDS_4, this.slotCap4);
    this.brickPal = this.makePool(PALETTE_WORDS, this.slotCap4);
    this.brickVox8 = this.makePool(BRICK_WORDS_8, this.slotCap8);

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
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 6,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
      ],
    });
    this.bindLayout = layout;

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

    this.bindGroup = this.makeBindGroup();
    this.uploadIndex();
    this.uploadSlots(null, null);
  }

  /**
   * Resident voxel memory, for benchmarks and budgeting. `dense` is what the
   * same grid would have cost as one 3D texture.
   */
  stats(): { bricks: number; wide: number; bytes: number; dense: number } {
    const st = this.bricks.stats();
    const [bx, by, bz] = this.coarseDim;
    return {
      bricks: st.used,
      wide: st.wide,
      bytes:
        this.slotCap4 * (BRICK_WORDS_4 + PALETTE_WORDS) * 4 +
        this.slotCap8 * BRICK_WORDS_8 * 4 +
        bx * by * bz * 4,
      dense: st.denseBytes,
    };
  }

  /** Bind group is rebuilt whenever a brick pool is reallocated. */
  private makeBindGroup(): GPUBindGroup {
    return this.gpu.device.createBindGroup({
      layout: this.bindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.indexTexture.createView() },
        { binding: 2, resource: { buffer: this.paletteBuffer } },
        { binding: 3, resource: { buffer: this.brickVox4 } },
        { binding: 4, resource: { buffer: this.materialBuffer } },
        { binding: 5, resource: { buffer: this.brickPal } },
        { binding: 6, resource: { buffer: this.brickVox8 } },
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
      maxSteps: Math.max(16, Math.min(4096, Math.round(src.maxSteps ?? this.quality.maxSteps))),
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
   * Release every GPU resource this renderer owns. Call it before dropping a
   * renderer — an app that builds a fresh one per load (the viewer does, because
   * the grid size changes) otherwise leaks a full-size 3D texture each time,
   * which at forest scale is hundreds of megabytes per reload. The instance is
   * unusable afterwards.
   */
  destroy(): void {
    this.indexTexture.destroy();
    this.brickVox4.destroy();
    this.brickPal.destroy();
    this.brickVox8.destroy();
    this.paletteBuffer.destroy();
    this.materialBuffer.destroy();
    this.uniformBuffer.destroy();
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

  /**
   * No longer needed: the brick index *is* the empty-space structure, so a null
   * brick means "skip". Kept as a no-op because consumers pair it with
   * updateVoxels, which now maintains both.
   */
  updateCoarse(_data: Uint8Array, _box?: DirtyBox): void {
    /* intentionally empty */
  }

  /**
   * Re-derive the sparse form from the caller's dense grid. With a `box`, only
   * the bricks overlapping it are rebuilt and uploaded; otherwise everything is.
   *
   * `data` must always be the full-grid array in `x + y*sx + z*sx*sy` order —
   * the box selects a region of it, it is not a standalone sub-grid.
   */
  updateVoxels(data: Uint8Array, box?: DirtyBox): void {
    if (!box) {
      this.bricks.rebuildAll(data);
      this.growPools();
      this.uploadIndex();
      this.uploadSlots(null, null);
      return;
    }
    const edit = this.bricks.rebuildBox(data, box);
    // Growing reallocates and re-uploads everything, so there is nothing left
    // to send afterwards.
    if (this.growPools()) {
      this.uploadIndex();
      return;
    }
    this.uploadIndex(edit.index);
    this.uploadSlots(edit.slots4, edit.slots8);
  }

  // --- brick pool plumbing ---------------------------------------------------

  /**
   * A binding larger than the device allows does not fail at creation — it
   * makes the bind group invalid, and rendering stops with a validation error
   * per frame. Catch it here, where the message can say what to do.
   */
  private checkPoolFits(cap4: number, cap8: number): void {
    const limit = this.gpu.limits?.maxStorageBufferBindingSize ?? 134217728;
    const biggest = Math.max(cap4 * BRICK_WORDS_4, cap8 * BRICK_WORDS_8) * 4;
    if (biggest <= limit) return;
    const mb = (n: number) => `${(n / 1048576).toFixed(0)} MiB`;
    throw new Error(
      `Scene needs a ${mb(biggest)} brick pool but this device caps a storage binding at ` +
        `${mb(limit)}. Raise it via initGpu({ limits: { maxStorageBufferBindingSize } }) if the ` +
        `adapter supports more, or use a smaller world.`,
    );
  }

  private makePool(wordsPerSlot: number, slots: number): GPUBuffer {
    // WebGPU rejects a zero-sized buffer, so empty pools still get one slot.
    const size = Math.max(1, slots) * wordsPerSlot * 4;
    return this.gpu.device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Make sure the GPU pools can hold every claimed slot. Entities stamped in
   * after construction claim more bricks, and a storage buffer cannot be
   * resized, so this reallocates with headroom and rebinds. Returns true when
   * it reallocated, in which case the caller must re-upload everything.
   */
  private growPools(): boolean {
    const need4 = this.bricks.slotCount4;
    const need8 = this.bricks.slotCount8;
    if (need4 <= this.slotCap4 && need8 <= this.slotCap8) return false;
    // Geometric growth with a floor, so planting a thousand entities does not
    // reallocate a thousand times. 1.5x rather than 2x: at forest scale the
    // headroom is hundreds of megabytes of otherwise idle GPU memory, and a
    // storage binding has a hard ceiling that doubling walks straight into.
    const cap = (need: number, have: number, floor: number) =>
      Math.max(floor, Math.ceil(need * 1.5), Math.ceil(have * 1.5));
    this.slotCap4 = cap(need4, this.slotCap4, 256);
    this.slotCap8 = cap(need8, this.slotCap8, 16);
    this.checkPoolFits(this.slotCap4, this.slotCap8);
    this.brickVox4.destroy();
    this.brickPal.destroy();
    this.brickVox8.destroy();
    this.brickVox4 = this.makePool(BRICK_WORDS_4, this.slotCap4);
    this.brickPal = this.makePool(PALETTE_WORDS, this.slotCap4);
    this.brickVox8 = this.makePool(BRICK_WORDS_8, this.slotCap8);
    this.bindGroup = this.makeBindGroup();
    this.uploadSlots(null, null);
    return true;
  }

  /** Upload the brick index volume, whole or a brick-space sub-box. */
  private uploadIndex(box?: DirtyBox): void {
    const [bx, by, bz] = this.coarseDim;
    const { device } = this.gpu;
    const data = this.bricks.index;
    if (!box) {
      device.queue.writeTexture(
        { texture: this.indexTexture },
        data,
        { bytesPerRow: bx * 4, rowsPerImage: by },
        { width: bx, height: by, depthOrArrayLayers: bz },
      );
      return;
    }
    const w = box.x1 - box.x0 + 1;
    const h = box.y1 - box.y0 + 1;
    const d = box.z1 - box.z0 + 1;
    if (w <= 0 || h <= 0 || d <= 0) return;
    device.queue.writeTexture(
      { texture: this.indexTexture, origin: { x: box.x0, y: box.y0, z: box.z0 } },
      data,
      {
        offset: (box.x0 + box.y0 * bx + box.z0 * bx * by) * 4,
        bytesPerRow: bx * 4,
        rowsPerImage: by,
      },
      { width: w, height: h, depthOrArrayLayers: d },
    );
  }

  /**
   * Upload brick payloads. `null` means every slot. A list is sorted and
   * coalesced into contiguous runs first: bricks claimed together get
   * consecutive slots, so stamping one entity usually collapses to a handful of
   * writes rather than one per brick.
   */
  private uploadSlots(slots4: number[] | null, slots8: number[] | null): void {
    const { device } = this.gpu;
    if (slots4 === null) {
      // The CPU pools are sized to the slots actually claimed, which is less
      // than the GPU capacity; only send what exists.
      const nv = Math.min(this.slotCap4 * BRICK_WORDS_4, this.bricks.voxels4.length);
      const np = Math.min(this.slotCap4 * PALETTE_WORDS, this.bricks.palettes.length);
      device.queue.writeBuffer(this.brickVox4, 0, this.bricks.voxels4, 0, nv);
      device.queue.writeBuffer(this.brickPal, 0, this.bricks.palettes, 0, np);
    } else {
      for (const [lo, hi] of runs(slots4)) {
        const n = hi - lo + 1;
        device.queue.writeBuffer(
          this.brickVox4, lo * BRICK_WORDS_4 * 4,
          this.bricks.voxels4, lo * BRICK_WORDS_4, n * BRICK_WORDS_4,
        );
        device.queue.writeBuffer(
          this.brickPal, lo * PALETTE_WORDS * 4,
          this.bricks.palettes, lo * PALETTE_WORDS, n * PALETTE_WORDS,
        );
      }
    }
    if (slots8 === null) {
      const n = Math.min(this.slotCap8 * BRICK_WORDS_8, this.bricks.voxels8.length);
      device.queue.writeBuffer(this.brickVox8, 0, this.bricks.voxels8, 0, n);
    } else {
      for (const [lo, hi] of runs(slots8)) {
        const n = hi - lo + 1;
        device.queue.writeBuffer(
          this.brickVox8, lo * BRICK_WORDS_8 * 4,
          this.bricks.voxels8, lo * BRICK_WORDS_8, n * BRICK_WORDS_8,
        );
      }
    }
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

/** Sort and coalesce slot indices into inclusive contiguous [lo, hi] runs. */
function runs(slots: number[]): [number, number][] {
  if (slots.length === 0) return [];
  const sorted = [...new Set(slots)].sort((a, b) => a - b);
  const out: [number, number][] = [];
  let lo = sorted[0];
  let prev = lo;
  for (let i = 1; i < sorted.length; i++) {
    const v = sorted[i];
    if (v === prev + 1) { prev = v; continue; }
    out.push([lo, prev]);
    lo = v;
    prev = v;
  }
  out.push([lo, prev]);
  return out;
}
