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
import lightsWesl from "./shaders/lights.wesl?raw";
import waterWesl from "./shaders/water.wesl?raw";
import fogWesl from "./shaders/fog.wesl?raw";
import precipWesl from "./shaders/precip.wesl?raw";
import raymarchWesl from "./shaders/raymarch.wesl?raw";
import type { GpuContext } from "./device";
import type { DirtyBox } from "./box";

export type { DirtyBox };
import { BrickGrid, BrickPool, BLOCK_ENTRIES, BRICK_WORDS_4, BRICK_WORDS_8, PALETTE_WORDS, TOP_B, emptyEdit, type BrickEdit } from "./brick";
import { sparseDims, type SparseVoxels } from "./sparse";
import { LIGHT_FLOATS, MAX_LIGHTS, packLights, type PointLight } from "./lights";
import type { AtmosphereParams } from "./atmosphere";

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
  "./lights.wesl": lightsWesl,
  "./water.wesl": waterWesl,
  "./fog.wesl": fogWesl,
  "./precip.wesl": precipWesl,
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

// Uniform buffer layout: 144 words (576 bytes). See struct Uniforms in the shader
// for the exact map (camera 0..31, environment/sky 32..71, occ/coarse 76..91,
// optional ground-plane floor 92..103, atmosphere 104..131, index regions
// 132..143 as u32).
const UNIFORM_FLOATS = 144;

// The index buffer (binding 1) holds every u32 table the traversal reads, in
// regions: top levels of the world and of each model, index blocks, the
// per-top-cell instance lists, instances and models. One buffer keeps the pass
// within the default limit of 8 storage buffers per stage.
type RegionName = "tops" | "blocks" | "cells" | "list" | "inst" | "models";
const REGIONS: RegionName[] = ["tops", "blocks", "cells", "list", "inst", "models"];
/** u32 words per instance: position, cos, anchor, sin, model, palette base, 2 pad. */
const INST_WORDS = 12;
/** u32 words per model: top offset, top dims, size, pad. */
const MODEL_WORDS = 8;
/** Instances per top cell list; more are dropped (with a console warning). */
const MAX_CELL_INSTANCES = 255;

/** A model drawn by instances; see Renderer.addModel. */
export interface ModelSource {
  size: { x: number; y: number; z: number };
  /** Dense role values, `x + y*sx + z*sx*sy`; or `sparse`. */
  data?: Uint8Array;
  sparse?: SparseVoxels;
}

/** One placement of a model: its anchor at a world position, turned about y. */
export interface Instance {
  model: number;
  /** World position of the model's anchor (voxels, fractional allowed). */
  x: number;
  y: number;
  z: number;
  /** Model anchor, in model voxels (default the base centre: size.x/2, 0, size.z/2). */
  anchor?: Vec3;
  /** Radians about +y; 0 leaves the model as authored. */
  yaw?: number;
  /** Palette slot of role 1: a voxel of role r draws slot base + r - 1. */
  base: number;
}

interface GpuModel {
  grid: BrickGrid;
  topOff: number;
  size: { x: number; y: number; z: number };
}

/** Minimal scene data the renderer needs to build/upload the voxel grid. */
export interface RenderScene {
  size: { x: number; y: number; z: number };
  /**
   * Dense voxels, `x + y*sx + z*sx*sy`. Optional: omit it for an empty world
   * and fill it through `Renderer.edit`, which is how a large scene avoids ever
   * holding a dense copy of itself in memory.
   */
  data?: Uint8Array;
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

export interface FrameParams extends AtmosphereParams {
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
  /** Seconds, for animated materials (water ripples). Leave it constant and nothing moves. */
  time?: number;
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
  /** Bricks and index blocks for the world and every model. */
  private readonly pool: BrickPool;
  private idxBuffer: GPUBuffer;
  private layout: Record<RegionName, { off: number; cap: number }>;
  private models: (GpuModel | null)[] = [];
  private modelData = new Uint32Array(MODEL_WORDS * 16);
  /** Free ranges of the tops region after the world's own top level: [offset, length]. */
  private topFree: [number, number][] = [];
  private topEnd = 0;
  private instData = new Uint32Array(INST_WORDS * 64);
  private instCount = 0;
  private cellData: Uint32Array;
  private cellTouched: number[] = [];
  private listData = new Uint32Array(1024);
  private listLen = 0;
  /** Brick-space dims, also the bounds test for the empty-space skip. */
  private readonly coarseDim: Vec3;
  private brickVox4: GPUBuffer;
  private brickPal: GPUBuffer;
  private brickVox8: GPUBuffer;
  private slotCap4 = 0;
  private slotCap8 = 0;
  private readonly bindLayout: GPUBindGroupLayout;
  private readonly materialBuffer: GPUBuffer;
  private readonly lightBuffer: GPUBuffer;
  private readonly lightData = new Float32Array(MAX_LIGHTS * LIGHT_FLOATS);
  private lightCount = 0;
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
    [this.occMin, this.occMax] = scene.data
      ? occupiedBounds(scene.data, scene.size)
      : // Nothing to scan yet; a world filled through edit() sets its own bounds
        // via setClipBounds, and the full grid is the safe default until then.
        [[0, 0, 0], [scene.size.x - 1, scene.size.y - 1, scene.size.z - 1]];

    const module = device.createShaderModule({ code: shaderCode });

    // Sparse brick form of the grid. The caller's dense array stays the source
    // of truth; this is the mirror the GPU reads.
    this.pool = new BrickPool();
    this.bricks = new BrickGrid(scene.size, scene.data, this.pool);
    this.coarseDim = [...this.bricks.dim] as Vec3;
    this.topEnd = this.bricks.top.length;
    this.cellData = new Uint32Array(this.bricks.top.length);
    this.layout = this.planLayout();
    this.idxBuffer = this.makeIdxBuffer();

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

    // Point lights (always bound; lightCount 0 means none are read).
    this.lightBuffer = device.createBuffer({
      size: MAX_LIGHTS * LIGHT_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.lightBuffer, 0, this.lightData);

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
          buffer: { type: "read-only-storage" },
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
        {
          binding: 7,
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
    this.uploadIndexAll();
    this.uploadSlots(null, null);
  }

  /**
   * Resident voxel memory, for benchmarks and budgeting. `dense` is what the
   * same grid would have cost as one 3D texture.
   */
  stats(): { bricks: number; wide: number; bytes: number; dense: number } {
    const st = this.bricks.stats();
    return {
      bricks: st.used,
      wide: st.wide,
      bytes:
        this.slotCap4 * (BRICK_WORDS_4 + PALETTE_WORDS) * 4 +
        this.slotCap8 * BRICK_WORDS_8 * 4 +
        this.idxBuffer.size,
      dense: st.denseBytes,
    };
  }

  /** Bind group is rebuilt whenever a brick pool is reallocated. */
  private makeBindGroup(): GPUBindGroup {
    return this.gpu.device.createBindGroup({
      layout: this.bindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.idxBuffer } },
        { binding: 2, resource: { buffer: this.paletteBuffer } },
        { binding: 3, resource: { buffer: this.brickVox4 } },
        { binding: 4, resource: { buffer: this.materialBuffer } },
        { binding: 5, resource: { buffer: this.brickPal } },
        { binding: 6, resource: { buffer: this.brickVox8 } },
        { binding: 7, resource: { buffer: this.lightBuffer } },
      ],
    });
  }

  /** Re-upload the 256-entry colour palette (e.g. after a carpet swap). */
  updatePalette(palette: Float32Array): void {
    this.gpu.device.queue.writeBuffer(this.paletteBuffer, 0, palette);
  }

  /**
   * Replace the point lights (up to MAX_LIGHTS; extras are ignored). Cheap: it
   * rewrites a 1.5 KB buffer, so moving a lamp every frame is fine. Invalidate
   * the frame loop afterwards.
   */
  setLights(lights: readonly PointLight[]): void {
    const { count } = packLights(lights, this.lightData);
    this.lightCount = count;
    this.gpu.device.queue.writeBuffer(this.lightBuffer, 0, this.lightData);
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
    this.idxBuffer.destroy();
    this.brickVox4.destroy();
    this.brickPal.destroy();
    this.brickVox8.destroy();
    this.paletteBuffer.destroy();
    this.materialBuffer.destroy();
    this.lightBuffer.destroy();
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
    this.commit(box ? this.bricks.rebuildBox(data, box) : this.bricks.rebuildAll(data));
  }

  /**
   * Edit the world a brick at a time, with no dense array involved.
   *
   * `fill` gets each overlapping brick's current 512 voxels and its world
   * origin; see BrickGrid.editBox. Touched bricks are uploaded before this
   * returns.
   */
  edit(box: DirtyBox, fill: (cells: Uint8Array, ox: number, oy: number, oz: number) => boolean): void {
    this.commit(this.bricks.editBox(box, fill));
  }

  /**
   * Edit many separate boxes (typically single bricks) with one upload at the
   * end, instead of one per box. What moving things need: a crowd touches a
   * few thousand scattered bricks a frame, and a box around all of them would
   * visit most of the world.
   */
  editMany(boxes: readonly DirtyBox[], fill: (cells: Uint8Array, ox: number, oy: number, oz: number) => boolean): void {
    if (!boxes.length) return;
    const edit = emptyEdit();
    for (const box of boxes) this.bricks.editBox(box, fill, edit);
    this.commit(edit);
  }

  /** Free every brick in `box` — see BrickGrid.clearBox. */
  clear(box: DirtyBox): void {
    this.commit(this.bricks.clearBox(box));
  }

  // --- models and instances ----------------------------------------------------

  /**
   * Upload a model once, to be drawn any number of times by `setInstances`.
   * Its voxels are role values (1..), mapped to palette slots per instance.
   * Returns the model id.
   */
  addModel(src: ModelSource): number {
    const grid = new BrickGrid(src.size, undefined, this.pool);
    const edit = emptyEdit();
    if (src.sparse) {
      const [dx, dy] = sparseDims(src.size);
      for (const [key, cells] of src.sparse.bricks) {
        const bx = key % dx, by = Math.floor(key / dx) % dy, bz = Math.floor(key / (dx * dy));
        const e = this.pool.encode(0, cells, edit);
        if (e) grid.setEntry(bx, by, bz, e, edit);
      }
    } else if (src.data) {
      const g = grid.rebuildAll(src.data);
      for (const k of ["slots4", "slots8", "blocks", "tops"] as const) for (const v of g[k]) edit[k].push(v);
    }
    grid.markNear(edit);
    const topOff = this.claimTops(grid.top.length);
    let id = this.models.indexOf(null);
    if (id < 0) id = this.models.push(null) - 1;
    this.models[id] = { grid, topOff, size: { ...src.size } };
    if (this.modelData.length < (id + 1) * MODEL_WORDS) this.modelData = growU32(this.modelData, (id + 1) * MODEL_WORDS);
    const m = id * MODEL_WORDS;
    this.modelData.set([topOff, grid.topDim[0], grid.topDim[1], grid.topDim[2], src.size.x, src.size.y, src.size.z, 0], m);
    const grewPools = this.growPools();
    if (this.ensureLayout() || grewPools) {
      this.uploadIndexAll();
      return id;
    }
    this.uploadSlots(edit.slots4, edit.slots8);
    this.uploadBlocks(edit.blocks);
    this.writeRegion("tops", topOff, grid.top, 0, grid.top.length);
    this.writeRegion("models", m, this.modelData, m, MODEL_WORDS);
    return id;
  }

  /** Free a model's bricks. Instances still naming it must be replaced first. */
  removeModel(id: number): void {
    const m = this.models[id];
    if (!m) return;
    m.grid.free();
    this.topFree.push([m.topOff, m.grid.top.length]);
    this.models[id] = null;
  }

  /**
   * Replace every instance. Cheap enough to call per frame for a few thousand:
   * it rewrites the instance table and the lists of the top cells they cover.
   */
  setInstances(list: readonly Instance[]): void {
    const [tx, ty, tz] = this.bricks.topDim;
    if (this.instData.length < list.length * INST_WORDS) this.instData = growU32(this.instData, list.length * INST_WORDS);
    const f = new Float32Array(this.instData.buffer);
    const perCell = new Map<number, number[]>();
    let dropped = 0;
    this.instCount = 0;
    for (const inst of list) {
      const m = this.models[inst.model];
      if (!m) continue;
      const k = this.instCount++;
      const o = k * INST_WORDS;
      const yaw = inst.yaw ?? 0, c = Math.cos(yaw), sn = Math.sin(yaw);
      const an = inst.anchor ?? [m.size.x / 2, 0, m.size.z / 2];
      f[o] = inst.x; f[o + 1] = inst.y; f[o + 2] = inst.z; f[o + 3] = c;
      f[o + 4] = an[0]; f[o + 5] = an[1]; f[o + 6] = an[2]; f[o + 7] = sn;
      this.instData[o + 8] = inst.model;
      this.instData[o + 9] = inst.base;
      this.instData[o + 10] = 0; this.instData[o + 11] = 0;
      // World box of the turned model: world = R (m - anchor) + pos, with
      // R = [c 0 s; 0 1 0; -s 0 c] (the same turn as bakePose).
      let x0 = Infinity, z0 = Infinity, x1 = -Infinity, z1 = -Infinity;
      for (const mx of [0, m.size.x]) for (const mz of [0, m.size.z]) {
        const dx = mx - an[0], dz = mz - an[2];
        const wx = c * dx + sn * dz + inst.x, wz = -sn * dx + c * dz + inst.z;
        x0 = Math.min(x0, wx); x1 = Math.max(x1, wx); z0 = Math.min(z0, wz); z1 = Math.max(z1, wz);
      }
      const y0 = inst.y - an[1], y1 = y0 + m.size.y;
      const cx0 = Math.max(0, Math.floor((x0 - 1) / TOP_B)), cx1 = Math.min(tx - 1, Math.floor((x1 + 1) / TOP_B));
      const cy0 = Math.max(0, Math.floor((y0 - 1) / TOP_B)), cy1 = Math.min(ty - 1, Math.floor((y1 + 1) / TOP_B));
      const cz0 = Math.max(0, Math.floor((z0 - 1) / TOP_B)), cz1 = Math.min(tz - 1, Math.floor((z1 + 1) / TOP_B));
      for (let cz = cz0; cz <= cz1; cz++)
        for (let cy = cy0; cy <= cy1; cy++)
          for (let cx = cx0; cx <= cx1; cx++) {
            const ci = cx + cy * tx + cz * tx * ty;
            let l = perCell.get(ci);
            if (!l) perCell.set(ci, (l = []));
            if (l.length < MAX_CELL_INSTANCES) l.push(k);
            else dropped++;
          }
    }
    if (dropped) console.warn(`setInstances: ${dropped} cell entries over the ${MAX_CELL_INSTANCES}-per-cell limit were dropped`);
    // Lists, and the cell entries pointing into them.
    let total = 0;
    for (const l of perCell.values()) total += l.length;
    if (this.listData.length < total) this.listData = growU32(this.listData, total);
    const touched: number[] = [];
    for (const ci of this.cellTouched) this.cellData[ci] = 0;
    let at = 0;
    for (const [ci, l] of perCell) {
      this.cellData[ci] = (at << 8) | l.length;
      for (const k of l) this.listData[at++] = k;
      touched.push(ci);
    }
    const dirty = [...this.cellTouched, ...touched];
    this.cellTouched = touched;
    this.listLen = total;
    if (this.ensureLayout()) {
      this.uploadIndexAll();
      return;
    }
    this.writeRegion("inst", 0, this.instData, 0, this.instCount * INST_WORDS);
    this.writeRegion("list", 0, this.listData, 0, this.listLen);
    for (const [lo, hi] of runs(dirty, 64)) this.writeRegion("cells", lo, this.cellData, lo, hi - lo + 1);
  }

  /** Models uploaded and instances placed, for overlays and budgets. */
  instanceStats(): { models: number; instances: number; blocks: number; bytes: number } {
    return {
      models: this.models.filter(Boolean).length,
      instances: this.instCount,
      blocks: this.pool.blockCount,
      bytes: this.slotCap4 * (BRICK_WORDS_4 + PALETTE_WORDS) * 4 + this.slotCap8 * BRICK_WORDS_8 * 4 + this.idxBuffer.size,
    };
  }

  private claimTops(n: number): number {
    const i = this.topFree.findIndex(([, len]) => len >= n);
    if (i >= 0) {
      const [off, len] = this.topFree[i];
      if (len === n) this.topFree.splice(i, 1);
      else this.topFree[i] = [off + n, len - n];
      return off;
    }
    const off = this.topEnd;
    this.topEnd += n;
    return off;
  }

  /** Upload what an edit of the world grid changed. */
  private commit(edit: BrickEdit): void {
    const grewPools = this.growPools();
    if (this.ensureLayout() || grewPools) {
      this.uploadIndexAll();
      return;
    }
    this.uploadSlots(edit.slots4, edit.slots8);
    this.uploadBlocks(edit.blocks);
    for (const [lo, hi] of runs(edit.tops, 16)) this.writeRegion("tops", lo, this.bricks.top, lo, hi - lo + 1);
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

  /** Region sizes wanted now, in words. */
  private needs(): Record<RegionName, number> {
    return {
      tops: this.topEnd,
      blocks: this.pool.blockCount * BLOCK_ENTRIES,
      cells: this.cellData.length,
      list: this.listLen,
      inst: this.instCount * INST_WORDS,
      models: this.models.length * MODEL_WORDS,
    };
  }

  /** Offsets with headroom; regions are laid out in REGIONS order. */
  private planLayout(): Record<RegionName, { off: number; cap: number }> {
    const need = this.needs();
    const floor: Record<RegionName, number> = { tops: 0, blocks: BLOCK_ENTRIES * 16, cells: 0, list: 256, inst: INST_WORDS * 16, models: MODEL_WORDS * 8 };
    const out = {} as Record<RegionName, { off: number; cap: number }>;
    let off = 0;
    for (const r of REGIONS) {
      // The world's cells never grow; everything else gets 1.5x headroom.
      const cap = r === "cells" ? need.cells : Math.max(floor[r], Math.ceil(need[r] * 1.5));
      out[r] = { off, cap };
      off += cap;
    }
    return out;
  }

  private makeIdxBuffer(): GPUBuffer {
    const words = REGIONS.reduce((n, r) => n + this.layout[r].cap, 0);
    const limit = this.gpu.limits?.maxStorageBufferBindingSize ?? 134217728;
    if (words * 4 > limit) {
      throw new Error(
        `Scene needs a ${(words * 4 / 1048576).toFixed(0)} MiB index buffer but this device caps a storage binding at ` +
          `${(limit / 1048576).toFixed(0)} MiB. Raise it via initGpu({ limits: { maxStorageBufferBindingSize } }).`,
      );
    }
    return this.gpu.device.createBuffer({ size: Math.max(4, words * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }

  /** Reallocate the index buffer when a region outgrows it. Returns true when it did (re-upload everything). */
  private ensureLayout(): boolean {
    const need = this.needs();
    if (REGIONS.every((r) => need[r] <= this.layout[r].cap)) return false;
    this.layout = this.planLayout();
    this.idxBuffer.destroy();
    this.idxBuffer = this.makeIdxBuffer();
    this.bindGroup = this.makeBindGroup();
    return true;
  }

  private writeRegion(r: RegionName, at: number, src: Uint32Array, from: number, n: number): void {
    if (n <= 0) return;
    this.gpu.device.queue.writeBuffer(this.idxBuffer, (this.layout[r].off + at) * 4, src, from, n);
  }

  private uploadBlocks(ids: number[]): void {
    for (const [lo, hi] of runs(ids)) this.writeRegion("blocks", lo * BLOCK_ENTRIES, this.pool.blocks, lo * BLOCK_ENTRIES, (hi - lo + 1) * BLOCK_ENTRIES);
  }

  /** Upload every index region. */
  private uploadIndexAll(): void {
    this.writeRegion("tops", 0, this.bricks.top, 0, this.bricks.top.length);
    for (const m of this.models) if (m) this.writeRegion("tops", m.topOff, m.grid.top, 0, m.grid.top.length);
    this.writeRegion("blocks", 0, this.pool.blocks, 0, this.pool.blockCount * BLOCK_ENTRIES);
    this.writeRegion("cells", 0, this.cellData, 0, this.cellData.length);
    this.writeRegion("list", 0, this.listData, 0, this.listLen);
    this.writeRegion("inst", 0, this.instData, 0, this.instCount * INST_WORDS);
    this.writeRegion("models", 0, this.modelData, 0, this.models.length * MODEL_WORDS);
    this.uploadSlots(null, null);
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
      const nv = Math.min(this.slotCap4 * BRICK_WORDS_4, this.pool.voxels4.length);
      const np = Math.min(this.slotCap4 * PALETTE_WORDS, this.pool.palettes.length);
      device.queue.writeBuffer(this.brickVox4, 0, this.pool.voxels4, 0, nv);
      device.queue.writeBuffer(this.brickPal, 0, this.pool.palettes, 0, np);
    } else {
      for (const [lo, hi] of runs(slots4)) {
        const n = hi - lo + 1;
        device.queue.writeBuffer(
          this.brickVox4, lo * BRICK_WORDS_4 * 4,
          this.pool.voxels4, lo * BRICK_WORDS_4, n * BRICK_WORDS_4,
        );
        device.queue.writeBuffer(
          this.brickPal, lo * PALETTE_WORDS * 4,
          this.pool.palettes, lo * PALETTE_WORDS, n * PALETTE_WORDS,
        );
      }
    }
    if (slots8 === null) {
      const n = Math.min(this.slotCap8 * BRICK_WORDS_8, this.pool.voxels8.length);
      device.queue.writeBuffer(this.brickVox8, 0, this.pool.voxels8, 0, n);
    } else {
      for (const [lo, hi] of runs(slots8)) {
        const n = hi - lo + 1;
        device.queue.writeBuffer(
          this.brickVox8, lo * BRICK_WORDS_8 * 4,
          this.pool.voxels8, lo * BRICK_WORDS_8, n * BRICK_WORDS_8,
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
    u[19] = p.time ?? 0;
    u[20] = p.lightDir[0]; u[21] = p.lightDir[1]; u[22] = p.lightDir[2];
    u[23] = width / height;
    u[24] = width; u[25] = height;
    u[26] = this.highlight.mode;
    u[27] = this.highlight.slotMin;
    u[28] = this.highlight.slotMax;
    u[29] = this.materialsEnabled;
    u[30] = this.lightCount;
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
    // Atmosphere (104..131); every effect is off at zero.
    const fog = p.fog, cl = p.clouds, pr = p.precipitation, sf = p.surface;
    w3(104, fog?.color ?? [0, 0, 0]); u[107] = fog ? Math.max(0, fog.density) : 0;
    w3(108, cl?.color ?? [0, 0, 0]); u[111] = cl ? Math.max(0, Math.min(1, cl.cover)) : 0;
    w3(112, pr?.fall ?? [0, -1, 0]); u[115] = pr ? Math.max(0, Math.min(1, pr.density)) : 0;
    w3(116, pr?.color ?? [0.8, 0.82, 0.86]); u[119] = pr ? (pr.kind === "snow" ? 2 : 1) : 0;
    w3(120, sf?.coverColor ?? [0.92, 0.94, 0.97]); u[123] = sf ? Math.max(0, Math.min(1, sf.cover)) : 0;
    u[124] = sf ? Math.max(0, Math.min(1, sf.wet)) : 0;
    u[125] = fog?.heightFalloff ?? 0;
    u[126] = cl?.drift[0] ?? 0; u[127] = cl?.drift[1] ?? 0;
    u[128] = p.waterWind?.[0] ?? 0; u[129] = p.waterWind?.[1] ?? 0;
    // Index regions (132..143), as u32.
    const w = new Uint32Array(u.buffer);
    const [tx, ty, tz] = this.bricks.topDim;
    w[132] = tx; w[133] = ty; w[134] = tz; w[135] = this.instCount;
    w[136] = this.layout.blocks.off; w[137] = this.layout.cells.off; w[138] = this.layout.list.off; w[139] = this.layout.inst.off;
    w[140] = this.layout.models.off; w[141] = this.layout.tops.off; w[142] = 0; w[143] = 0;

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
function runs(slots: number[], gap = 1): [number, number][] {
  if (slots.length === 0) return [];
  const sorted = [...new Set(slots)].sort((a, b) => a - b);
  const out: [number, number][] = [];
  let lo = sorted[0];
  let prev = lo;
  for (let i = 1; i < sorted.length; i++) {
    const v = sorted[i];
    // Indices within `gap` of the run join it: one larger write beats many tiny ones.
    if (v <= prev + gap) { prev = v; continue; }
    out.push([lo, prev]);
    lo = v;
    prev = v;
  }
  out.push([lo, prev]);
  return out;
}

function growU32(a: Uint32Array, need: number): Uint32Array<ArrayBuffer> {
  let len = Math.max(a.length * 2, 64);
  while (len < need) len *= 2;
  const next = new Uint32Array(len);
  next.set(a);
  return next;
}
