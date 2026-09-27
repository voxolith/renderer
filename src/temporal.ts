// Temporal accumulation state for the renderer's compute path (RenderQuality.temporal): the two
// history textures it ping-pongs between, the previous frame's camera, and the per-brick-column
// change stamps that make the shader trace afresh where the world or a moving instance changed.
// The shader side is shaders/temporal.wesl, whose `Temporal` uniform block this fills.

import { LIGHT_FLOATS } from "./lights";

type Vec3 = [number, number, number];

/** Floats in the `Temporal` uniform block (temporal.wesl). */
const TEMPORAL_FLOATS = 24;
const FLAG_HISTORY = 1;
const FLAG_HOLD = 2;
const FLAG_FORCE_KEY = 4;
const FLAG_FORCE_POINT = 8;
const FLAG_POINT_SHADOWS = 16;
/** Voxels per change-stamp column (a brick). */
const COLUMN = 8;
/** Voxels round a changed box that are traced afresh besides its shadow (AO reaches one). */
const MARGIN = 2;
/** How far a change's shadow is assumed to reach, at most, in voxels. */
const MAX_REACH = 160;
/**
 * Above this share of the world's columns changed for one frame (a crowd over the whole map), the
 * frame is rendered without temporal accumulation: nearly every pixel would trace anyway, and the
 * temporal kernel costs more than the plain one. It comes back below `CROWDED_OFF`.
 */
const CROWDED_ON = 0.3;
const CROWDED_OFF = 0.2;
/** A camera move above this many voxels in one frame counts as a cut (the history restarts). */
const JUMP = 32;
/** A key light turn above this (radians, about 2°) retraces its shadow everywhere at once. */
const KEY_JUMP = 0.035;

/**
 * Samples a pixel averages (`cap`); 1 in how many frames a tile with unconverged pixels samples
 * them while the view is still (`rate`) and while it moves (`moving`); and the development views
 * (`show`: 1 what each tile traced, 2 samples per pixel); `nohold` keeps converged pixels sampling
 * (to measure the cost of tracing everything). Tuning hook: globalThis.__voxolithTemporal.
 */
function settings(): { cap: number; rate: number; moving: number; show: number; nohold: boolean } {
  const g = (globalThis as { __voxolithTemporal?: { cap?: number; rate?: number; moving?: number; show?: number; nohold?: boolean } }).__voxolithTemporal;
  const r = (v: number | undefined, d: number) => Math.max(1, Math.min(8, Math.round(v ?? d)));
  return { cap: Math.max(1, Math.min(63, Math.round(g?.cap ?? 32))), rate: r(g?.rate, 1), moving: r(g?.moving, 4), show: g?.show ?? 0, nohold: !!g?.nohold };
}

/** The history and change stamps behind `RenderQuality.temporal`; owned by a Renderer. */
export class TemporalHistory {
  /** Bind group layout of group 2 (temporal.wesl). */
  readonly layout: GPUBindGroupLayout;
  private readonly device: GPUDevice;
  private readonly uniform: GPUBuffer;
  private readonly data = new Float32Array(TEMPORAL_FLOATS);
  private readonly changed: GPUTexture;
  private readonly stamps: Uint32Array;
  private readonly cols: [number, number];
  /** Rows of `stamps` changed since the last upload: [first, last], or first > last for none. */
  private dirty: [number, number] = [1, 0];
  private hist: { tex: [GPUTexture, GPUTexture]; groups: [GPUBindGroup, GPUBindGroup]; w: number; h: number } | null = null;
  /** Number of the next frame to render (stamps of changes before it equal it). */
  private frame = 1;
  /** Whether the last frame's history may be reused. */
  private valid = false;
  /** Frames left before the image has settled. */
  private settle = 0;
  private readonly prevCam = new Float32Array(16);
  private prevRes: [number, number] = [0, 0];
  private keyDir: Vec3 = [0, 1, 0];
  /** Columns stamped for the coming frame. */
  private marked = 0;
  /** Whether frames are being rendered without accumulation because too much changes (CROWDED_ON). */
  private crowd = false;
  /** Position and range of every shadowed point light, as last rendered. */
  private shadowed = new Float32Array(0);

  constructor(device: GPUDevice, gridSize: readonly number[]) {
    this.device = device;
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32uint" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint" } },
      ],
    });
    this.uniform = device.createBuffer({ size: TEMPORAL_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.cols = [Math.max(1, Math.ceil(gridSize[0] / COLUMN)), Math.max(1, Math.ceil(gridSize[2] / COLUMN))];
    this.stamps = new Uint32Array(this.cols[0] * this.cols[1]);
    this.changed = device.createTexture({
      size: { width: this.cols[0], height: this.cols[1] },
      format: "r32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: this.changed }, this.stamps, { bytesPerRow: this.cols[0] * 4 }, { width: this.cols[0], height: this.cols[1] });
  }

  /** Forget the history: the next frame traces everything afresh. */
  reset(): void {
    this.valid = false;
    this.settle = this.settleFrames();
  }

  /** Whether the image is still converging (the caller should render another frame). */
  converging(): boolean {
    return this.settle > 0;
  }

  /**
   * Record that the voxels in an inclusive box changed (an edit, or a moving instance's old or new
   * box). The columns it covers, and those its shadow may fall on (away from the key light as far
   * as its height can throw one, and away from each shadowed point light in range), are traced
   * afresh next frame.
   */
  mark(x0: number, y0: number, z0: number, x1: number, y1: number, z1: number): void {
    const [cw, ch] = this.cols;
    const st = this.stamps, f = this.frame;
    // The box's columns, with MARGIN voxels all round (AO reaches one), moved by (dx, dz).
    const rect = (dx: number, dz: number) => {
      const c0 = Math.max(0, Math.floor((x0 - MARGIN + dx) / COLUMN)), c1 = Math.min(cw - 1, Math.floor((x1 + MARGIN + dx) / COLUMN));
      const r0 = Math.max(0, Math.floor((z0 - MARGIN + dz) / COLUMN)), r1 = Math.min(ch - 1, Math.floor((z1 + MARGIN + dz) / COLUMN));
      if (c0 > c1 || r0 > r1) return;
      for (let r = r0; r <= r1; r++) {
        for (let i = r * cw + c0, e = r * cw + c1; i <= e; i++) {
          if (st[i] !== f) { st[i] = f; this.marked++; }
        }
      }
      if (r0 < this.dirty[0]) this.dirty[0] = r0;
      if (r1 > this.dirty[1]) this.dirty[1] = r1;
    };
    // Its shadow: the box swept along (dx, dz), a brick at a time.
    const sweep = (dx: number, dz: number) => {
      let len = Math.hypot(dx, dz);
      if (len > MAX_REACH) { dx *= MAX_REACH / len; dz *= MAX_REACH / len; len = MAX_REACH; }
      const n = Math.ceil(len / COLUMN);
      for (let i = 1; i <= n; i++) rect((dx * i) / n, (dz * i) / n);
    };
    rect(0, 0);
    // The key light's shadow of the box's top, on ground down to a brick below its bottom.
    const L = this.keyDir;
    if (L[1] > 0.01) {
      const h = (y1 - y0 + 1 + COLUMN) / Math.max(L[1], 0.05);
      sweep(-L[0] * h, -L[2] * h);
    } else sweep(-L[0] * MAX_REACH, -L[2] * MAX_REACH);
    // Point lights: away from each one in range, to the end of its range.
    const s = this.shadowed;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, cz = (z0 + z1) / 2;
    const half = Math.hypot(x1 - x0, y1 - y0, z1 - z0) / 2;
    for (let i = 0; i < s.length; i += 4) {
      const dx = cx - s[i], dy = cy - s[i + 1], dz = cz - s[i + 2], range = s[i + 3];
      const d = Math.hypot(dx, dy, dz);
      if (d - half >= range) continue;
      const k = (range - Math.max(0, d - half)) / Math.max(d, 1);
      sweep(dx * k, dz * k);
    }
    this.settle = this.settleFrames();
  }

  /**
   * Set up this frame: history textures for its size, what changed since the last one, and the
   * uniform block. `u` is the frame's main uniform block (camera in words 0..15). Returns the
   * bind group (group 2) to dispatch with.
   */
  prepare(u: Float32Array, lightDir: Vec3, lightAngle: number, effectScale: number, lights: Float32Array, lightCount: number, width: number, height: number): GPUBindGroup {
    const { device } = this;
    if (!this.hist || this.hist.w !== width || this.hist.h !== height) {
      this.hist?.tex.forEach((t) => t.destroy());
      const make = () => device.createTexture({ size: { width, height }, format: "rgba32uint", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
      const tex: [GPUTexture, GPUTexture] = [make(), make()];
      const group = (i: number) => device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: tex[1 - i].createView() },
          { binding: 2, resource: tex[i].createView() },
          { binding: 3, resource: this.changed.createView() },
        ],
      });
      this.hist = { tex, groups: [group(0), group(1)], w: width, h: height };
      this.reset();
    }
    const { cap, rate: still, moving, show, nohold } = settings();
    let flags = this.valid ? FLAG_HISTORY | (nohold ? 0 : FLAG_HOLD) : 0;
    let capNow = cap;
    // The key light: a jump retraces its shadow at once; a slow turn keeps sampling.
    const ll = Math.hypot(lightDir[0], lightDir[1], lightDir[2]) || 1;
    const L: Vec3 = [lightDir[0] / ll, lightDir[1] / ll, lightDir[2] / ll];
    const k = this.keyDir;
    const turn = Math.acos(Math.max(-1, Math.min(1, L[0] * k[0] + L[1] * k[1] + L[2] * k[2])));
    if (turn > KEY_JUMP) flags |= FLAG_FORCE_KEY;
    if (turn > 1e-5) {
      flags &= ~FLAG_HOLD;
      capNow = Math.min(cap, 8);
      this.settle = this.settleFrames();
    }
    this.keyDir = L;
    // Shadowed point lights: moving one retraces point shadows (colour and brightness need nothing).
    const sh: number[] = [];
    for (let i = 0; i < lightCount; i++) {
      const o = i * LIGHT_FLOATS;
      if (lights[o + 8] > 0.5) sh.push(lights[o], lights[o + 1], lights[o + 2], lights[o + 3]);
    }
    if (sh.length !== this.shadowed.length || sh.some((v, i) => v !== this.shadowed[i])) {
      flags |= FLAG_FORCE_POINT;
      this.shadowed = Float32Array.from(sh);
      this.settle = this.settleFrames();
    }
    if (sh.length) flags |= FLAG_POINT_SHADOWS;
    // A camera that moved leaves pixels to converge; while it moves, tiles take turns to sample.
    let moved = u[15] !== this.prevCam[15];
    for (let i = 0; i < 15 && !moved; i++) moved = i % 4 !== 3 && u[i] !== this.prevCam[i];
    if (moved) this.settle = this.settleFrames();
    // A cut (a turn over about 10°, or a jump of more than 32 voxels) starts over: estimates from
    // what the previous view saw at a pixel would be wrong for a few frames.
    const pc = this.prevCam;
    const fwdDot = u[12] * pc[12] + u[13] * pc[13] + u[14] * pc[14];
    if (moved && (fwdDot < 0.985 || Math.hypot(u[0] - pc[0], u[1] - pc[1], u[2] - pc[2]) > JUMP)) flags &= ~FLAG_HISTORY;
    const rate = moved ? moving : still;

    const t = this.data, w = new Uint32Array(t.buffer);
    t.set(this.prevCam.subarray(0, 3), 0); w[3] = this.frame;
    t.set(this.prevCam.subarray(4, 7), 4); w[7] = flags;
    t.set(this.prevCam.subarray(8, 11), 8); t[11] = this.prevCam[15];
    t.set(this.prevCam.subarray(12, 15), 12); t[15] = this.prevRes[0] / Math.max(1, this.prevRes[1]);
    t[16] = this.prevRes[0]; t[17] = this.prevRes[1]; t[18] = capNow; w[19] = rate;
    t[20] = Math.max(0, lightAngle); t[21] = Math.max(0, effectScale); t[22] = show;
    device.queue.writeBuffer(this.uniform, 0, t);
    if (this.dirty[0] <= this.dirty[1]) {
      const [cw] = this.cols;
      const [r0, r1] = this.dirty;
      device.queue.writeTexture({ texture: this.changed, origin: { x: 0, y: r0 } }, this.stamps, { offset: r0 * cw * 4, bytesPerRow: cw * 4 }, { width: cw, height: r1 - r0 + 1 });
      this.dirty = [1, 0];
    }
    this.prevCam.set(u.subarray(0, 16));
    this.prevRes = [width, height];
    return this.hist.groups[this.frame & 1];
  }

  /**
   * Whether this frame should skip accumulation because most of the world changed for it (see
   * CROWDED_ON). A skipped frame must be followed by `skip()` instead of `prepare`/`finish`.
   */
  crowded(): boolean {
    const share = this.marked / this.stamps.length;
    this.crowd = share > (this.crowd ? CROWDED_OFF : CROWDED_ON);
    return this.crowd;
  }

  /** A frame rendered without accumulation: the history does not carry over it. */
  skip(): void {
    this.frame++;
    this.marked = 0;
    this.valid = false;
  }

  /** After a frame rendered with the group from `prepare`. */
  finish(): void {
    this.marked = 0;
    this.frame++;
    this.valid = true;
    if (this.settle > 0) this.settle--;
  }

  /** Frames until a pixel that starts over now has converged. */
  private settleFrames(): number {
    const { cap, rate } = settings();
    return cap * rate + 2;
  }

  /** Release the GPU resources. */
  destroy(): void {
    this.hist?.tex.forEach((t) => t.destroy());
    this.changed.destroy();
    this.uniform.destroy();
  }
}
