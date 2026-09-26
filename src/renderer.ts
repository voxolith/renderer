// Owns the WebGPU pipeline and resources for the fullscreen voxel raymarch pass.
// Browser only: the shaders are imported with Vite's `?raw`, so only a bundler
// can load this module.

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
import selftestWesl from "./shaders/selftest.wesl?raw";
import raymarchWesl from "./shaders/raymarch.wesl?raw";
import type { GpuContext } from "./device";
import type { DirtyBox } from "./box";
import { INST_WORDS, MAX_PARTS, maxPoseWords, packInstance, packPose, partBoxes } from "./instance";

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
  "./selftest.wesl": selftestWesl,
  "./raymarch.wesl": raymarchWesl,
};

/**
 * The world's palette: voxels in the world are 8-bit, so their colours are
 * slots 0..255. Instances draw from palettes of their own (addPalette),
 * stored after these in the same buffer, as many as a scene needs.
 */
export const WORLD_SLOTS = 256;
/** f32 per material slot: 2 vec4. See materials.wesl. */
const MATERIAL_WORDS = 8;

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
// per-top-cell instance lists, instances, models and the part transforms of
// instances drawn with parts. One buffer keeps the pass within the default
// limit of 8 storage buffers per stage. Instance and part records are laid out
// in instance.ts (INST_WORDS, PART_WORDS), shared with the CPU sampler.
type RegionName = "tops" | "blocks" | "cells" | "list" | "inst" | "models" | "parts";
const REGIONS: RegionName[] = ["tops", "blocks", "cells", "list", "inst", "models", "parts"];
/** u32 words per model: top offset, top dims, size, part grid top offset + 1 (0 = no parts). */
const MODEL_WORDS = 8;
/** Instances per top cell list; more are dropped (with a console warning). */
const MAX_CELL_INSTANCES = 255;

/** A model drawn by instances; see Renderer.addModel. */
export interface ModelSource {
  /** Extent in voxels. */
  size: { x: number; y: number; z: number };
  /** Dense role values, `x + y*sx + z*sx*sy`; or `sparse`. */
  data?: Uint8Array;
  sparse?: SparseVoxels;
  /**
   * Part index per voxel (dense models only), laid out like `data`: e.g. a skeleton's bone per
   * voxel. A model with parts can be drawn with one transform per part (`Instance.parts`), posed
   * on the GPU from this single rest model. Parts must be ordered parents first; at most 32.
   */
  parts?: Uint8Array;
  /**
   * Per part, its parent (-1 for none) and the joint where it meets it, in model voxels. Cells
   * around a posed joint may be filled from either side, so a turned child stays attached.
   */
  joints?: readonly { parent: number; at: readonly [number, number, number] }[];
  /**
   * Per-part voxel boxes to use instead of this model's own (min x, y, z, max x, y, z each). Poses
   * are packed per set of boxes, so copies of a model that pass the same boxes object (a creature's
   * wounded copies, with the undamaged model's boxes, which contain theirs) share every pose.
   */
  partBoxes?: Int32Array;
}

/** One placement of a model: its anchor at a world position, turned about y. */
export interface Instance {
  /** Model id from `Renderer.addModel`. */
  model: number;
  /** World position of the model's anchor (voxels, fractional allowed). */
  x: number;
  /** See `x`. */
  y: number;
  /** See `x`. */
  z: number;
  /** Model anchor, in model voxels (default the base centre: size.x/2, 0, size.z/2). */
  anchor?: Vec3;
  /** Radians about +y; 0 leaves the model as authored. Turns pivot on the anchor voxel's centre. */
  yaw?: number;
  /**
   * A full rotation instead of `yaw`: a 3x3 row-major matrix, world = R · model, pivoting on the
   * anchor voxel's centre (tumbling debris, a tilted placement). Sampling at world voxel centres,
   * so any rotation still draws as axis-aligned cubes.
   */
  rotation?: ArrayLike<number>;
  /**
   * Mirror the model along its x before turning (the engine's orientation
   * bit 2): with yaw = -(o & 3) * PI / 2 this draws exactly what stamping
   * with orientation o would. `anchor` stays the unmirrored model's.
   */
  mirror?: boolean;
  /** Palette slot of role 1 (from addPalette, or the world's): a voxel of role r draws slot base + r - 1. */
  base: number;
  /**
   * One model-space transform per part of a model added with `parts` (a 3x4 row-major affine, 12
   * floats each, e.g. `poseMatrices` output), applied before this placement: the model is posed
   * on the GPU, no baking. Each world cell is taken back through every part's inverse and drawn
   * from the first part whose own voxel is there. `mirror` does not apply.
   *
   * Based on the rest-space animation of Gruen, Benthin, Kern and McAllister, "Ray Tracing Massive
   * Amounts of Animated Geometry" (HPG 2026, doi:10.1145/3820014), and Kao, Makowski, Fujieda and
   * Harada, "Voxel Deformation-Aware Neural Intersection Function" (EG 2026,
   * doi:10.2312/egs.20261026).
   */
  parts?: ArrayLike<number>;
}

interface GpuModel {
  grid: BrickGrid;
  topOff: number;
  size: { x: number; y: number; z: number };
  /** Part index + 1 per voxel, for models added with parts. */
  partGrid?: BrickGrid;
  partTopOff?: number;
  partBoxes?: Int32Array;
  joints?: ModelSource["joints"];
}

/** Minimal scene data the renderer needs to build/upload the voxel grid. */
export interface RenderScene {
  /** World grid extent in voxels; fixed for the renderer's lifetime. */
  size: { x: number; y: number; z: number };
  /**
   * Dense voxels, `x + y*sx + z*sx*sy`. Optional: omit it for an empty world
   * and fill it through `Renderer.edit`, which is how a large scene avoids ever
   * holding a dense copy of itself in memory.
   */
  data?: Uint8Array;
  /** The world's 256 colours, RGBA floats 0..1 per slot (1024 floats); slot 0 is empty. */
  palette: Float32Array;
  /** Optional per-slot materials (256×8 f32); enables material shading. */
  materials?: Float32Array;
}

/** Optional offscreen render destination (defaults to the canvas swapchain). */
export interface RenderTarget {
  /** A view of a texture in `gpu.format` with RENDER_ATTACHMENT usage. */
  view: GPUTextureView;
  /** Pixels; sets the aspect ratio and ray basis. */
  width: number;
  /** Pixels. */
  height: number;
}

type Vec3 = [number, number, number];

/**
 * Everything one `Renderer.render` call draws with: a camera (spread a
 * `CameraFrame` from `makeCamera`, `firstPersonFrame` or `chaseFrame`), the
 * key light and sky, and the optional atmosphere. Colours are linear RGB;
 * directions are unit vectors in grid space. The renderer knows nothing about
 * time of day: `@voxolith/engine/atmosphere` computes these from a clock, or
 * pass constants.
 */
export interface FrameParams extends AtmosphereParams {
  // Camera (from camera.ts).
  /** Eye position, grid space (voxels). */
  camPos: Vec3;
  /** Unit screen-right. */
  camRight: Vec3;
  /** Unit screen-up. */
  camUp: Vec3;
  /** Unit view direction. */
  camFwd: Vec3;
  /** tan of half the vertical field of view. */
  tanHalfFov: number;
  // Environment lighting + sky. Generic: the caller supplies the values (a game
  // may drive them from a day/night clock, or pass static constants).
  /** Towards the key light (sun or moon); it lights and casts the shadows. */
  lightDir: Vec3;
  /** Key light colour and strength. */
  lightColor: Vec3;
  /** Ambient light from above (hemisphere, sky side). */
  ambientSky: Vec3;
  /** Ambient light from below (hemisphere, ground side). */
  ambientGround: Vec3;
  /** Where the sun disc is drawn in the sky. */
  sunDir: Vec3;
  /** Where the moon disc is drawn. The moon lights nothing; `lightDir` does. */
  moonDir: Vec3;
  /** Colour of the sun disc and its glow. */
  sunColor: Vec3;
  /** Colour of the moon disc. */
  moonColor: Vec3;
  /** Sky colour at the zenith. */
  skyTop: Vec3;
  /** Sky colour at the horizon. */
  skyHorizon: Vec3;
  /** 0 day .. 1 night: blends in the night sky (stars). */
  nightFactor: number;
  /** Sun disc brightness; 0 hides it. */
  sunIntensity: number;
  /** Moon disc brightness; 0 hides it. */
  moonIntensity: number;
  /** Seconds, for animated materials (water ripples). Leave it constant and nothing moves. */
  time?: number;
}

/** Per-frame cost knobs. All are runtime uniforms; changing them is free. */
export interface RenderQuality {
  /** Primary-ray DDA step cap (16..4096). Lower is cheaper; too low clips far geometry. */
  maxSteps: number;
  /** Shadow-ray step cap (0..512). 0 disables shadows entirely. */
  shadowSteps: number;
  /** Face ambient occlusion (8 neighbour lookups per hit). */
  ao: boolean;
}

/** Names of the `QUALITY_PRESETS`; "low" has no shadows or AO. */
export type QualityPreset = "low" | "medium" | "high";

/** Presets consumers can offer in a UI; `high` matches the engine's original look. */
export const QUALITY_PRESETS: Record<QualityPreset, RenderQuality> = {
  low: { maxSteps: 256, shadowSteps: 0, ao: false },
  medium: { maxSteps: 512, shadowSteps: 48, ao: true },
  high: { maxSteps: 768, shadowSteps: 90, ao: true },
};

/** Optional infinite ground-plane drawn on ray-miss below `y` (off by default). */
export interface FloorParams {
  enabled: boolean;
  /** Height of the plane, grid space (voxels). */
  y: number;
  /** One colour of the 3-voxel checker, linear RGB. */
  colorA: Vec3;
  /** The other colour of the checker, linear RGB. */
  colorB: Vec3;
}

/**
 * The fullscreen voxel raymarcher: one world grid (8-bit palette slots, held
 * as sparse bricks on the GPU), models drawn by instance, point lights and
 * atmosphere. Build it with `createRenderer`, edit the world with `edit`,
 * `editMany` or `updateVoxels`, and call `render` for each frame; nothing
 * draws between calls. Browser only.
 *
 * World coordinates are voxels, y up, with the grid from (0, 0, 0) to `size`;
 * a voxel value is a palette slot and 0 is empty. GPU buffers grow as needed.
 * Call `destroy` before dropping a renderer.
 */
export class Renderer {
  private readonly gpu: GpuContext;
  private readonly pipeline: GPURenderPipeline;
  /** The same pass with instance sampling compiled in; made when a scene first places one. */
  private instancedPipeline: GPURenderPipeline | null = null;
  private partsPipeline: GPURenderPipeline | null = null;
  private readonly makePipeline: (instances: boolean, parts?: boolean) => GPURenderPipeline;
  private readonly uniformBuffer: GPUBuffer;
  private readonly uniformData = new Float32Array(UNIFORM_FLOATS);
  private bindGroup: GPUBindGroup;
  private readonly gridSize: [number, number, number];
  private paletteBuffer: GPUBuffer;
  /** CPU mirror of every palette slot: the world's, then the instance palettes'. */
  private paletteData = new Float32Array(WORLD_SLOTS * 2 * 4);
  private materialData = new Float32Array(WORLD_SLOTS * 2 * MATERIAL_WORDS);
  /** Slots claimed so far (the world's 256 always). */
  private paletteEnd = WORLD_SLOTS;
  /** Freed instance palettes: [first slot, length]. */
  private paletteFree: [number, number][] = [];
  private occMin: Vec3;
  private occMax: Vec3;
  /** Sparse mirror of the caller's dense grid; see src/brick.ts. */
  private readonly bricks: BrickGrid;
  /** Bricks and index blocks for the world and every model. */
  private readonly pool: BrickPool;
  private idxBuffer: GPUBuffer;
  /** The world's top level again, as a 3D texture: the lookup every ray step starts with, and texture reads cache better. */
  private readonly topTexture: GPUTexture;
  private layout: Record<RegionName, { off: number; cap: number }>;
  private models: (GpuModel | null)[] = [];
  private modelData = new Uint32Array(MODEL_WORDS * 16);
  /** Free ranges of the tops region after the world's own top level: [offset, length]. */
  private topFree: [number, number][] = [];
  private topEnd = 0;
  private instData = new Uint32Array(INST_WORDS * 64);
  private instCount = 0;
  /** Part records of instances drawn with parts: the static set's, then the moving set's. */
  /**
   * Pose store: the packed poses posed instances name (instance.ts packPose), each packed once and
   * shared by every instance showing it, keyed by the part transforms' identity and the model's part
   * boxes. The static set's poses come first; the moving set's after them, and that area is cleared
   * and refilled when it holds far more than the moving set uses.
   */
  private partData = new Uint32Array(4096);
  private partLen = 0;
  private staticPartLen = 0;
  private staticPoses = new WeakMap<object, Map<Int32Array, { off: number; box: number[] }>>();
  private movingPoses = new WeakMap<object, Map<Int32Array, { off: number; box: number[] }>>();
  /** Words of the moving area the poses shown in the last moving set use. */
  private movingPoseWords = 0;
  /** Newly packed pose ranges [from, to) waiting to be uploaded. */
  private poseDirty: [number, number][] = [];
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
  private materialBuffer: GPUBuffer;
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

  /**
   * Prefer `createRenderer`, which links the shader for you. `shaderCode` is
   * the WGSL from `raymarchShaderCode()`. Building scans a dense `scene.data`
   * once for its occupied bounds and uploads it as bricks.
   */
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
    this.staticCells = new Uint32Array(this.bricks.top.length);
    this.layout = this.planLayout();
    this.idxBuffer = this.makeIdxBuffer();
    this.topTexture = device.createTexture({
      size: this.bricks.topDim,
      dimension: "3d",
      format: "r32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.paletteData.set(scene.palette.subarray(0, Math.min(scene.palette.length, WORLD_SLOTS * 4)));
    this.paletteBuffer = this.makePaletteBuffer(4);

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
    this.materialBuffer = this.makePaletteBuffer(MATERIAL_WORDS);
    if (scene.materials) {
      this.materialData.set(scene.materials.subarray(0, Math.min(scene.materials.length, WORLD_SLOTS * MATERIAL_WORDS)));
      this.materialsEnabled = 1;
    }
    device.queue.writeBuffer(this.paletteBuffer, 0, this.paletteData);
    device.queue.writeBuffer(this.materialBuffer, 0, this.materialData);

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
        {
          binding: 8,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "uint", viewDimension: "3d" },
        },
      ],
    });
    this.bindLayout = layout;

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    // Instance sampling is a pipeline constant (grid.wesl INSTANCES), and so is
    // sampling instances drawn with parts (PARTS), so a scene never pays for
    // code it does not use: compiled-in code costs even when never taken.
    this.makePipeline = (instances, parts = false) =>
      device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: { module, entryPoint: "vs" },
        fragment: {
          module,
          entryPoint: "fs",
          targets: [{ format: gpu.format }],
          constants: { 0: instances ? 1 : 0, 1: parts ? 1 : 0 },
        },
        primitive: { topology: "triangle-list" },
      });
    this.pipeline = this.makePipeline(false);

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
        { binding: 8, resource: this.topTexture.createView() },
      ],
    });
  }

  /** Re-upload the world's 256-entry colour palette (e.g. after a carpet swap). */
  updatePalette(palette: Float32Array): void {
    const n = Math.min(palette.length, WORLD_SLOTS * 4);
    this.paletteData.set(palette.subarray(0, n));
    this.gpu.device.queue.writeBuffer(this.paletteBuffer, 0, this.paletteData, 0, n);
  }

  /**
   * Replace the point lights (up to MAX_LIGHTS; extras are ignored). Cheap: it
   * rewrites a 1.5 KB buffer, so moving a lamp every frame is fine. Invalidate
   * the frame loop afterwards.
   *
   * @param lights - The whole set; pass `[]` for none.
   *
   * @example
   * ```ts
   * renderer.setLights([
   *   { position: lamp, color: [1.0, 0.7, 0.38], intensity: 2.6, range: 90, glow: 3.5 },
   *   // A soft unshadowed spill, so light still reaches just round a trunk.
   *   { position: lamp, color: [1.0, 0.75, 0.45], intensity: 0.25, range: 50, shadows: false },
   * ]);
   * loop.invalidate();
   * ```
   */
  setLights(lights: readonly PointLight[]): void {
    const { count } = packLights(lights, this.lightData);
    this.lightCount = count;
    this.gpu.device.queue.writeBuffer(this.lightBuffer, 0, this.lightData);
  }

  /** Upload per-slot materials (256×8 f32) and enable material shading. */
  updateMaterials(materials: Float32Array): void {
    const n = Math.min(materials.length, WORLD_SLOTS * MATERIAL_WORDS);
    this.materialData.set(materials.subarray(0, n));
    this.gpu.device.queue.writeBuffer(this.materialBuffer, 0, this.materialData, 0, n);
    this.materialsEnabled = 1;
  }

  // --- instance palettes -------------------------------------------------------

  /**
   * A palette of its own for instances: `colors` is RGBA per entry (a role's
   * colour at index r - 1), `materials` optionally 8 floats per entry (see
   * PaletteAllocator.buildMaterials). Returns the slot of entry 0, the
   * `base` an instance uses. There is no budget beyond GPU memory: a
   * species, or a single placement, can have its own.
   */
  addPalette(colors: Float32Array, materials?: Float32Array): number {
    const n = Math.max(1, Math.floor(colors.length / 4));
    const i = this.paletteFree.findIndex(([, len]) => len >= n);
    let base: number;
    if (i >= 0) {
      const [at, len] = this.paletteFree[i];
      base = at;
      if (len === n) this.paletteFree.splice(i, 1);
      else this.paletteFree[i] = [at + n, len - n];
    } else {
      base = this.paletteEnd;
      this.paletteEnd += n;
    }
    this.growPalette();
    this.writePalette(base, colors, materials);
    return base;
  }

  /** Recolour a palette in place (restyle every instance that uses it). */
  setPaletteColors(base: number, colors: Float32Array, materials?: Float32Array): void {
    if (base < WORLD_SLOTS) throw new Error("setPaletteColors is for instance palettes; use updatePalette for the world's");
    this.writePalette(base, colors, materials);
  }

  /** Free a palette from addPalette. Instances still naming it draw garbage colours. */
  removePalette(base: number, entries: number): void {
    if (base < WORLD_SLOTS) return;
    this.paletteFree.push([base, entries]);
  }

  private writePalette(base: number, colors: Float32Array, materials?: Float32Array): void {
    const n = Math.floor(colors.length / 4);
    this.paletteData.set(colors.subarray(0, n * 4), base * 4);
    this.gpu.device.queue.writeBuffer(this.paletteBuffer, base * 16, this.paletteData, base * 4, n * 4);
    const m = this.materialData;
    if (materials) {
      m.set(materials.subarray(0, n * MATERIAL_WORDS), base * MATERIAL_WORDS);
      this.materialsEnabled = 1;
    } else m.fill(0, base * MATERIAL_WORDS, (base + n) * MATERIAL_WORDS);
    this.gpu.device.queue.writeBuffer(this.materialBuffer, base * MATERIAL_WORDS * 4, m, base * MATERIAL_WORDS, n * MATERIAL_WORDS);
  }

  private makePaletteBuffer(wordsPerSlot: number): GPUBuffer {
    const slots = this.paletteData.length / 4;
    return this.gpu.device.createBuffer({ size: slots * wordsPerSlot * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }

  /** Make room for every claimed slot: reallocate both buffers with headroom and rebind. */
  private growPalette(): void {
    if (this.paletteEnd * 4 <= this.paletteData.length) return;
    const slots = Math.ceil(this.paletteEnd * 1.5);
    const p = new Float32Array(slots * 4); p.set(this.paletteData); this.paletteData = p;
    const m = new Float32Array(slots * MATERIAL_WORDS); m.set(this.materialData); this.materialData = m;
    this.paletteBuffer.destroy();
    this.materialBuffer.destroy();
    this.paletteBuffer = this.makePaletteBuffer(4);
    this.materialBuffer = this.makePaletteBuffer(MATERIAL_WORDS);
    this.gpu.device.queue.writeBuffer(this.paletteBuffer, 0, this.paletteData);
    this.gpu.device.queue.writeBuffer(this.materialBuffer, 0, this.materialData);
    this.bindGroup = this.makeBindGroup();
  }

  /** Toggle/parameterise selection-highlight shading (greyout + edge highlight). */
  setHighlight(state: { mode: number; slotMin: number; slotMax: number }): void {
    this.highlight = state;
  }

  /** Configure the optional infinite ground-plane drawn on ray-miss. */
  setFloor(state: FloorParams): void {
    this.floor = state;
  }

  /**
   * Set per-frame cost knobs (partial updates allowed). Takes effect next frame.
   * Values are clamped to the ranges `RenderQuality` states.
   *
   * @param q - A preset name, or the fields to change.
   *
   * @example
   * ```ts
   * renderer.setQuality(gpu.software ? "low" : "medium");
   * renderer.setQuality({ shadowSteps: 0 }); // keep the rest
   * loop.invalidate();
   * ```
   */
  setQuality(q: Partial<RenderQuality> | QualityPreset): void {
    const src = typeof q === "string" ? QUALITY_PRESETS[q] : q;
    this.quality = {
      maxSteps: Math.max(16, Math.min(4096, Math.round(src.maxSteps ?? this.quality.maxSteps))),
      shadowSteps: Math.max(0, Math.min(512, Math.round(src.shadowSteps ?? this.quality.shadowSteps))),
      ao: src.ao ?? this.quality.ao,
    };
  }

  /** The knobs in effect (a copy). */
  getQuality(): RenderQuality {
    return { ...this.quality };
  }

  /**
   * Diagnostic modes. 0 is normal; 1 forces single-step DDA (no empty-space
   * skip), for exactness checks and for grids that change every frame; 7, 8
   * and 9 replace the picture with self-test bands of the voxel lookup, green
   * where a stage works (see shaders/selftest.wesl).
   */
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
    this.topTexture.destroy();
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
   *
   * @param box - Inclusive voxel box; every brick it overlaps is visited whole.
   * @param fill - Gets the brick's 512 voxels (`lx + ly*8 + lz*64`) and its
   *   world origin; mutate the cells and return true if anything changed.
   *   Cells outside `box` are the caller's to leave alone.
   *
   * @example
   * ```ts
   * // Carve a 9×9×9 hole centred on (cx, cy, cz).
   * const box = { x0: cx - 4, y0: cy - 4, z0: cz - 4, x1: cx + 4, y1: cy + 4, z1: cz + 4 };
   * renderer.edit(box, (cells, ox, oy, oz) => {
   *   let changed = false;
   *   for (let i = 0; i < 512; i++) {
   *     const x = ox + (i & 7), y = oy + ((i >> 3) & 7), z = oz + (i >> 6);
   *     if (x < box.x0 || x > box.x1 || y < box.y0 || y > box.y1 || z < box.z0 || z > box.z1) continue;
   *     if (cells[i]) { cells[i] = 0; changed = true; }
   *   }
   *   return changed;
   * });
   * loop.invalidate();
   * ```
   */
  edit(box: DirtyBox, fill: (cells: Uint8Array, ox: number, oy: number, oz: number) => boolean): void {
    this.commit(this.bricks.editBox(box, fill));
  }

  /**
   * Edit many separate boxes (typically single bricks) with one upload at the
   * end, instead of one per box. What moving things need: a crowd touches a
   * few thousand scattered bricks a frame, and a box around all of them would
   * visit most of the world.
   *
   * @param boxes - Inclusive voxel boxes. Boxes sharing a brick visit it once
   *   per box, so keep them disjoint.
   * @param fill - As for `edit`, called once per brick per box.
   *
   * @example
   * ```ts
   * // Stream in the ground for the brick columns that became visible.
   * const boxes = columns.map(([bx, bz]) => ({
   *   x0: bx * 8, y0: 0, z0: bz * 8,
   *   x1: bx * 8 + 7, y1: size.y - 1, z1: bz * 8 + 7,
   * }));
   * renderer.editMany(boxes, (cells, ox, oy, oz) => ground.fillBrick(cells, ox, oy, oz));
   * ```
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
   *
   * The model's bricks share the world's GPU pool, so a model costs its
   * occupied bricks once however many instances draw it. Ids of removed
   * models are reused.
   *
   * @param src - Size plus dense `data` or `sparse` bricks.
   * @returns The id an `Instance` names in `model`.
   *
   * @example
   * ```ts
   * const tree = renderer.addModel({ size: model.size, data: model.data });
   * // Role r of this tree draws colour r - 1 of its own palette.
   * const bark = renderer.addPalette(new Float32Array([0.4, 0.28, 0.18, 1, 0.25, 0.5, 0.2, 1]));
   * renderer.setInstances([
   *   { model: tree, base: bark, x: 40, y: 12, z: 60 },
   *   { model: tree, base: bark, x: 90.5, y: 14, z: 31, yaw: Math.PI / 3, mirror: true },
   * ]);
   * ```
   */
  addModel(src: ModelSource): number {
    if (src.parts && !src.data) throw new Error("addModel: parts need a dense model (data)");
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
    // Parts: a second grid of part index + 1 in the same pool, beside the roles (both are 8-bit).
    let partGrid: BrickGrid | undefined, partTopOff: number | undefined, boxes: Int32Array | undefined;
    if (src.parts && src.data) {
      const ids = new Uint8Array(src.data.length);
      let count = 0;
      for (let i = 0; i < ids.length; i++) if (src.data[i]) { ids[i] = src.parts[i] + 1; count = Math.max(count, src.parts[i] + 1); }
      partGrid = new BrickGrid(src.size, undefined, this.pool);
      const g = partGrid.rebuildAll(ids);
      for (const k of ["slots4", "slots8", "blocks", "tops"] as const) for (const v of g[k]) edit[k].push(v);
      partTopOff = this.claimTops(partGrid.top.length);
      const parts = Math.max(count, src.joints?.length ?? 0);
      // Each brick of a posed instance keeps a 32-bit mask of the parts in it (see instance.ts).
      if (parts > MAX_PARTS) throw new Error(`addModel: at most ${MAX_PARTS} parts can be posed on the GPU (this model has ${parts})`);
      // Given boxes (a superset, e.g. the undamaged model's) let damaged copies share poses.
      boxes = src.partBoxes ?? partBoxes(src.size, src.data, src.parts, parts);
    }
    let id = this.models.indexOf(null);
    if (id < 0) id = this.models.push(null) - 1;
    this.models[id] = { grid, topOff, size: { ...src.size }, partGrid, partTopOff, partBoxes: boxes, joints: src.joints };
    if (this.modelData.length < (id + 1) * MODEL_WORDS) this.modelData = growU32(this.modelData, (id + 1) * MODEL_WORDS);
    const m = id * MODEL_WORDS;
    this.modelData.set([topOff, grid.topDim[0], grid.topDim[1], grid.topDim[2], src.size.x, src.size.y, src.size.z, partGrid ? partTopOff! + 1 : 0], m);
    const grewPools = this.growPools();
    if (this.ensureLayout() || grewPools) {
      this.uploadIndexAll();
      return id;
    }
    this.uploadSlots(edit.slots4, edit.slots8);
    this.uploadBlocks(edit.blocks);
    this.writeRegion("tops", topOff, grid.top, 0, grid.top.length);
    if (partGrid) this.writeRegion("tops", partTopOff!, partGrid.top, 0, partGrid.top.length);
    this.writeRegion("models", m, this.modelData, m, MODEL_WORDS);
    return id;
  }

  /** Free a model's bricks. Instances still naming it must be replaced first. */
  removeModel(id: number): void {
    const m = this.models[id];
    if (!m) return;
    m.grid.free();
    this.topFree.push([m.topOff, m.grid.top.length]);
    if (m.partGrid) {
      m.partGrid.free();
      this.topFree.push([m.partTopOff!, m.partGrid.top.length]);
    }
    this.models[id] = null;
  }

  /**
   * Place instances. Static ones (the default: scenery) replace the static
   * set and build its per-cell lists once. `{ dynamic: true }` replaces only
   * the moving set, which is what a crowd calls every frame: its cost is the
   * moving instances and the cells they touch, however much scenery there is.
   *
   * Each 64^3 top cell draws at most 255 instances; more are dropped with a
   * console warning. Instances naming a removed model are skipped.
   *
   * @param list - The whole static set, or the whole moving set.
   * @param opts - `dynamic: true` for the moving set.
   *
   * @example
   * ```ts
   * renderer.setInstances(trees); // once: scenery
   * // Every frame, for what moves:
   * const moving = rats.map((r) => ({ model: ratModel, base: ratPalette, x: r.x, y: r.y, z: r.z, yaw: r.yaw }));
   * renderer.setInstances(moving, { dynamic: true });
   * ```
   */
  setInstances(list: readonly Instance[], opts: { dynamic?: boolean } = {}): void {
    if (opts.dynamic) {
      this.dynamicList = list;
      this.rebuildDynamic();
      return;
    }
    // Static: instances [0, n), lists [0, total), and the cell entries they give.
    const perCell = new Map<number, number[]>();
    this.staticCount = 0;
    // A new static set repacks the store from the start, so both areas start over.
    this.partLen = 0;
    this.staticPoses = new WeakMap();
    this.movingPoses = new WeakMap();
    this.movingPoseWords = 0;
    this.poseDirty = [];
    for (const inst of list) {
      const k = this.writeInstance(inst, this.staticCount);
      if (k < 0) continue;
      this.cellsOf(k, (ci) => {
        let l = perCell.get(ci);
        if (!l) perCell.set(ci, (l = []));
        l.push(k);
      });
      this.staticCount++;
    }
    this.staticPartLen = this.partLen;
    for (const ci of this.staticTouched) this.staticCells[ci] = 0;
    this.staticTouched = [];
    let total = 0;
    for (const l of perCell.values()) total += Math.min(MAX_CELL_INSTANCES, l.length);
    if (this.listData.length < total) this.listData = growU32(this.listData, total);
    let at = 0, dropped = 0;
    for (const [ci, l] of perCell) {
      const n = Math.min(MAX_CELL_INSTANCES, l.length);
      dropped += l.length - n;
      this.staticCells[ci] = (at << 8) | n;
      for (let i = 0; i < n; i++) this.listData[at++] = l[i];
      this.staticTouched.push(ci);
    }
    if (dropped) console.warn(`setInstances: ${dropped} cell entries over the ${MAX_CELL_INSTANCES}-per-cell limit were dropped`);
    this.staticListLen = total;
    this.cellData.set(this.staticCells);
    this.cellTouched = [];
    this.staticDirty = true;
    this.rebuildDynamic();
  }

  private dynamicList: readonly Instance[] = [];
  private staticCount = 0;
  private staticListLen = 0;
  private staticCells = new Uint32Array(0);
  private staticTouched: number[] = [];
  private staticDirty = false;

  /** The packed pose for these part transforms of model `m`, packing it into the store the first time. */
  private poseOf(parts: ArrayLike<number>, m: GpuModel, moving: boolean): { off: number; box: number[] } {
    const cache = moving ? this.movingPoses : this.staticPoses;
    const key = parts as unknown as object;
    let byBoxes = cache.get(key);
    if (!byBoxes) cache.set(key, (byBoxes = new Map()));
    let pose = byBoxes.get(m.partBoxes!);
    if (!pose) {
      const pm = { size: m.size, partBoxes: m.partBoxes, joints: m.joints };
      const most = maxPoseWords(pm);
      if (this.partData.length < this.partLen + most) this.partData = growU32(this.partData, this.partLen + most);
      const { words, box } = packPose(parts, pm, this.partData, this.partLen);
      pose = { off: this.partLen, box };
      this.poseDirty.push([this.partLen, this.partLen + words]);
      this.partLen += words;
      byBoxes.set(m.partBoxes!, pose);
    }
    return pose;
  }

  /** Write instance `k`'s words; returns k, or -1 when its model is gone. The packing is instance.ts's, shared with the CPU sampler. */
  private writeInstance(inst: Instance, k: number, moving = false): number {
    const m = this.models[inst.model];
    if (!m) return -1;
    if (this.instData.length < (k + 1) * INST_WORDS) this.instData = growU32(this.instData, (k + 1) * INST_WORDS);
    const pose = inst.parts && m.partBoxes ? this.poseOf(inst.parts, m, moving) : undefined;
    const { box } = packInstance(inst, { size: m.size, partBoxes: m.partBoxes, joints: m.joints }, inst.model, this.instData, k * INST_WORDS, pose);
    if (this.instBox.length < (k + 1) * 6) { const nb = new Float64Array(Math.max(64, (k + 1) * 12)); nb.set(this.instBox); this.instBox = nb; }
    this.instBox.set(box, k * 6);
    return k;
  }
  private instBox = new Float64Array(64 * 6);

  /** Every world top cell instance `k`'s box touches. */
  private cellsOf(k: number, visit: (ci: number) => void): void {
    const [tx, ty, tz] = this.bricks.topDim;
    const b = this.instBox, o = k * 6;
    const cx0 = Math.max(0, Math.floor((b[o] - 1) / TOP_B)), cx1 = Math.min(tx - 1, Math.floor((b[o + 3] + 1) / TOP_B));
    const cy0 = Math.max(0, Math.floor((b[o + 1] - 1) / TOP_B)), cy1 = Math.min(ty - 1, Math.floor((b[o + 4] + 1) / TOP_B));
    const cz0 = Math.max(0, Math.floor((b[o + 2] - 1) / TOP_B)), cz1 = Math.min(tz - 1, Math.floor((b[o + 5] + 1) / TOP_B));
    for (let cz = cz0; cz <= cz1; cz++)
      for (let cy = cy0; cy <= cy1; cy++)
        for (let cx = cx0; cx <= cx1; cx++) visit(cx + cy * tx + cz * tx * ty);
  }

  /**
   * Moving instances go after the static ones. Each cell they touch gets a
   * fresh list (its static instances, then the moving ones) after the static
   * lists; cells they left get their static entry back.
   */
  private rebuildDynamic(): void {
    let n = this.staticCount;
    // Clear the moving area when it holds far more than the last moving set used (poses that are no
    // longer shown); this frame's poses are then packed afresh.
    if (this.partLen - this.staticPartLen > Math.max(1 << 16, 4 * this.movingPoseWords)) {
      this.partLen = this.staticPartLen;
      this.movingPoses = new WeakMap();
    }
    const seen = new Set<object>();
    let used = 0;
    const perCell = new Map<number, number[]>();
    for (const inst of this.dynamicList) {
      const k = this.writeInstance(inst, n, true);
      if (k < 0) continue;
      if (inst.parts && !seen.has(inst.parts as unknown as object)) {
        seen.add(inst.parts as unknown as object);
        const m = this.models[inst.model]!;
        if (m.partBoxes) used += maxPoseWords({ size: m.size, partBoxes: m.partBoxes });
      }
      this.cellsOf(k, (ci) => {
        let l = perCell.get(ci);
        if (!l) perCell.set(ci, (l = []));
        l.push(k);
      });
      n++;
    }
    const dynCount = n - this.staticCount;
    this.movingPoseWords = used;
    let at = this.staticListLen;
    const touched: number[] = [];
    for (const ci of this.cellTouched) this.cellData[ci] = this.staticCells[ci];
    for (const [ci, l] of perCell) {
      const st = this.staticCells[ci];
      const sn = st & 0xff, ss = st >>> 8;
      const count = Math.min(MAX_CELL_INSTANCES, sn + l.length);
      if (this.listData.length < at + count) this.listData = growU32(this.listData, at + count);
      this.cellData[ci] = (at << 8) | count;
      for (let i = 0; i < sn && i < count; i++) this.listData[at + i] = this.listData[ss + i];
      for (let i = sn; i < count; i++) this.listData[at + i] = l[i - sn];
      at += count;
      touched.push(ci);
    }
    const dirty = [...this.cellTouched, ...touched];
    this.cellTouched = touched;
    this.instCount = n;
    this.listLen = at;
    if (this.ensureLayout()) {
      this.staticDirty = false;
      this.uploadIndexAll();
      return;
    }
    if (this.staticDirty) {
      this.writeRegion("inst", 0, this.instData, 0, this.instCount * INST_WORDS);
      this.writeRegion("parts", 0, this.partData, 0, this.partLen);
      this.poseDirty = [];
      this.writeRegion("list", 0, this.listData, 0, this.listLen);
      this.writeRegion("cells", 0, this.cellData, 0, this.cellData.length);
      this.staticDirty = false;
      return;
    }
    this.writeRegion("inst", this.staticCount * INST_WORDS, this.instData, this.staticCount * INST_WORDS, dynCount * INST_WORDS);
    // Only poses packed since the last upload: a pose already on the GPU costs nothing more.
    for (const [lo, hi] of this.poseDirty) this.writeRegion("parts", lo, this.partData, lo, hi - lo);
    this.poseDirty = [];
    this.writeRegion("list", this.staticListLen, this.listData, this.staticListLen, this.listLen - this.staticListLen);
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
    this.uploadTopTexture(edit.tops);
  }

  /** Mirror world top-level entries into the texture: the changed ones, or all. */
  private uploadTopTexture(changed?: number[]): void {
    const [tx, ty, tz] = this.bricks.topDim;
    const top = this.bricks.top;
    if (!changed || changed.length > 2048) {
      this.gpu.device.queue.writeTexture({ texture: this.topTexture }, top, { bytesPerRow: tx * 4, rowsPerImage: ty }, { width: tx, height: ty, depthOrArrayLayers: tz });
      return;
    }
    for (const i of new Set(changed)) {
      const x = i % tx, y = Math.floor(i / tx) % ty, z = Math.floor(i / (tx * ty));
      this.gpu.device.queue.writeTexture({ texture: this.topTexture, origin: { x, y, z } }, top, { offset: i * 4, bytesPerRow: 4, rowsPerImage: 1 }, { width: 1, height: 1, depthOrArrayLayers: 1 });
    }
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
      parts: this.partLen,
    };
  }

  /** Offsets with headroom; regions are laid out in REGIONS order. */
  private planLayout(): Record<RegionName, { off: number; cap: number }> {
    const need = this.needs();
    const floor: Record<RegionName, number> = { tops: 0, blocks: BLOCK_ENTRIES * 16, cells: 0, list: 256, inst: INST_WORDS * 16, models: MODEL_WORDS * 8, parts: 1024 };
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
    this.uploadTopTexture();
    for (const m of this.models) {
      if (!m) continue;
      this.writeRegion("tops", m.topOff, m.grid.top, 0, m.grid.top.length);
      if (m.partGrid) this.writeRegion("tops", m.partTopOff!, m.partGrid.top, 0, m.partGrid.top.length);
    }
    this.writeRegion("blocks", 0, this.pool.blocks, 0, this.pool.blockCount * BLOCK_ENTRIES);
    this.writeRegion("cells", 0, this.cellData, 0, this.cellData.length);
    this.writeRegion("list", 0, this.listData, 0, this.listLen);
    this.writeRegion("inst", 0, this.instData, 0, this.instCount * INST_WORDS);
    this.writeRegion("models", 0, this.modelData, 0, this.models.length * MODEL_WORDS);
    this.writeRegion("parts", 0, this.partData, 0, this.partLen);
    this.poseDirty = [];
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
   *
   * One fullscreen pass, submitted before this returns. The swapchain size is
   * `gpu.width` × `gpu.height`, so call `resizeToDisplay(gpu)` first when the
   * canvas may have changed size.
   *
   * @param p - Camera, lighting, sky and atmosphere for this frame.
   * @param target - Draw here instead of the canvas.
   *
   * @example
   * ```ts
   * const camera = makeCamera({ target: [48, 8, 48], distance: 200, pitchDeg: 32, fovDeg: 35 });
   * const sun: [number, number, number] = [0.48, 0.81, 0.33];
   * resizeToDisplay(gpu);
   * renderer.render({
   *   ...camera(35),
   *   lightDir: sun, lightColor: [1, 0.98, 0.94],
   *   ambientSky: [0.5, 0.53, 0.6], ambientGround: [0.3, 0.29, 0.27],
   *   sunDir: sun, moonDir: [0, -1, 0], sunColor: [1, 0.96, 0.85], moonColor: [0, 0, 0],
   *   skyTop: [0.35, 0.55, 0.9], skyHorizon: [0.75, 0.82, 0.92],
   *   nightFactor: 0, sunIntensity: 1, moonIntensity: 0,
   *   time: performance.now() / 1000,
   * });
   * ```
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
    u[130] = p.effectScale ?? 1;
    // Index regions (132..143), as u32.
    const w = new Uint32Array(u.buffer);
    const [tx, ty, tz] = this.bricks.topDim;
    w[132] = tx; w[133] = ty; w[134] = tz; w[135] = this.instCount;
    w[136] = this.layout.blocks.off; w[137] = this.layout.cells.off; w[138] = this.layout.list.off; w[139] = this.layout.inst.off;
    w[140] = this.layout.models.off; w[141] = this.layout.tops.off; w[142] = this.layout.parts.off; w[143] = 0;

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
    let pipeline = this.pipeline;
    if (this.instCount > 0 && this.partLen > 0) pipeline = this.partsPipeline ??= this.makePipeline(true, true);
    else if (this.instCount > 0) pipeline = this.instancedPipeline ??= this.makePipeline(true);
    pass.setPipeline(pipeline);
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
 *
 * @param gpu - From `initGpu`.
 * @param scene - Grid size, optional dense voxels, the world palette and
 *   optional materials. The renderer keeps its own copy as bricks; `data` can
 *   be dropped afterwards.
 * @returns A renderer at the high quality preset, floor off, no lights.
 *
 * @example
 * ```ts
 * import { createRenderer, initGpu } from "@voxolith/renderer";
 *
 * const gpu = await initGpu(canvas);
 * const size = { x: 96, y: 32, z: 96 };
 * const data = new Uint8Array(size.x * size.y * size.z);
 * for (let z = 0; z < size.z; z++)
 *   for (let x = 0; x < size.x; x++) data[x + z * size.x * size.y] = 1; // grass at y = 0
 * const palette = new Float32Array(256 * 4);
 * palette.set([0.36, 0.62, 0.28, 1], 1 * 4);
 * const renderer = await createRenderer(gpu, { size, data, palette });
 * ```
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
