// A stand-in WebGPU device for headless checks of the Renderer class: buffers keep what is written
// to them (so two renderers' index buffers can be compared byte for byte), pipelines record how
// they were made, and passes record the pipeline they were drawn with. Nothing is drawn.
//
// Loading renderer.ts under bun also needs its `?raw` shader imports served as text:
// `installRawLoader()` registers a bun plugin for that; import the renderer dynamically after it.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import type { GpuContext } from "../src/device";

/** Register the `?raw` loader (idempotent). */
export function installRawLoader(): void {
  const g = globalThis as { __voxRawLoader?: boolean };
  if (g.__voxRawLoader) return;
  g.__voxRawLoader = true;
  Bun.plugin({
    name: "raw",
    setup(build) {
      build.onResolve({ filter: /\?raw$/ }, (args) => ({ path: resolve(dirname(args.importer), args.path.replace(/\?raw$/, "")), namespace: "raw" }));
      build.onLoad({ filter: /.*/, namespace: "raw" }, (args) => ({ contents: `export default ${JSON.stringify(readFileSync(args.path, "utf8"))};`, loader: "js" }));
    },
  });
}

/** A buffer that keeps its contents. */
export interface MockBuffer {
  size: number;
  bytes: Uint8Array;
  destroyed: boolean;
  destroy(): void;
}

/** How a pipeline was made. */
export interface MockPipeline {
  kind: "render" | "compute";
  async: boolean;
  entry: string;
  constants: Record<string, number>;
}

/** The mock and what it recorded. */
export interface MockGpu {
  gpu: GpuContext;
  buffers: MockBuffer[];
  pipelines: MockPipeline[];
  /** The pipeline of every pass begun, in order. */
  drawn: MockPipeline[];
  /** Resolve pending async pipeline creations (they wait for this, to model an in-flight compile). */
  flush(): Promise<void>;
}

/** A fake GpuContext. `canvasStorage` chooses whether compute kernels write the canvas directly. */
export function mockGpu(opts: { canvasStorage?: boolean; format?: GPUTextureFormat } = {}): MockGpu {
  const gl = globalThis as Record<string, unknown>;
  gl.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
  gl.GPUTextureUsage ??= { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
  gl.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
  const buffers: MockBuffer[] = [];
  const pipelines: MockPipeline[] = [];
  const drawn: MockPipeline[] = [];
  let waiting: (() => void)[] = [];
  const texture = () => ({ createView: () => ({}), destroy() {} });
  const pipeline = (kind: "render" | "compute", async: boolean, d: GPURenderPipelineDescriptor | GPUComputePipelineDescriptor): MockPipeline => {
    const stage = kind === "render" ? (d as GPURenderPipelineDescriptor).fragment! : (d as GPUComputePipelineDescriptor).compute;
    const p = { kind, async, entry: stage.entryPoint ?? "", constants: { ...(stage.constants as Record<string, number>) } };
    pipelines.push(p);
    return p;
  };
  const later = <T>(make: () => T): Promise<T> => new Promise((res) => waiting.push(() => res(make())));
  const pass = () => {
    const rec = { setPipeline: (p: MockPipeline) => drawn.push(p), setBindGroup() {}, draw() {}, dispatchWorkgroups() {}, end() {} };
    return rec;
  };
  const device = {
    createShaderModule: () => ({}),
    createBuffer: ({ size }: { size: number }) => {
      const b: MockBuffer = { size, bytes: new Uint8Array(size), destroyed: false, destroy() { b.destroyed = true; } };
      buffers.push(b);
      return b;
    },
    createTexture: texture,
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createBindGroup: () => ({}),
    createRenderPipeline: (d: GPURenderPipelineDescriptor) => pipeline("render", false, d),
    createComputePipeline: (d: GPUComputePipelineDescriptor) => pipeline("compute", false, d),
    createRenderPipelineAsync: (d: GPURenderPipelineDescriptor) => later(() => pipeline("render", true, d)),
    createComputePipelineAsync: (d: GPUComputePipelineDescriptor) => later(() => pipeline("compute", true, d)),
    createCommandEncoder: () => ({ beginRenderPass: pass, beginComputePass: pass, finish: () => ({}) }),
    queue: {
      writeBuffer(buf: MockBuffer, offset: number, data: ArrayBufferView, dataOffset = 0, size?: number) {
        const el = (data as unknown as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
        const n = size ?? (data.byteLength / el - dataOffset);
        const src = new Uint8Array(data.buffer, data.byteOffset + dataOffset * el, n * el);
        if (offset + src.length > buf.size) throw new Error(`writeBuffer past the end (${offset + src.length} > ${buf.size})`);
        buf.bytes.set(src, offset);
      },
      writeTexture() {},
      submit() {},
    },
  };
  const gpu = {
    device: device as unknown as GPUDevice,
    context: { getCurrentTexture: texture } as unknown as GPUCanvasContext,
    canvas: {} as HTMLCanvasElement,
    format: opts.format ?? "bgra8unorm",
    width: 64,
    height: 48,
    pixelRatio: 1,
    renderScale: 1,
    adapterInfo: { vendor: "", architecture: "", device: "", description: "" },
    software: false,
    limits: { maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 30, maxTextureDimension3D: 2048 },
    features: new Set<string>(),
    canvasStorage: !!opts.canvasStorage,
  } as unknown as GpuContext;
  return {
    gpu, buffers, pipelines, drawn,
    async flush() {
      while (waiting.length) {
        const w = waiting;
        waiting = [];
        for (const f of w) f();
        await Promise.resolve();
      }
      await new Promise((r) => setTimeout(r, 0));
    },
  };
}
