// WebGPU device + canvas context setup, with a graceful fallback when the
// platform doesn't support WebGPU.

/**
 * The WebGPU device and canvas an app renders with, from `initGpu`. One per
 * canvas; pass it to `createRenderer`. `width`, `height` and `renderScale` are
 * live: `resizeToDisplay` and `setRenderScale` update them in place.
 */
export interface GpuContext {
  device: GPUDevice;
  /** Configured for `device` and `format`, premultiplied alpha. */
  context: GPUCanvasContext;
  canvas: HTMLCanvasElement;
  /** The browser's preferred canvas format. */
  format: GPUTextureFormat;
  /** Current backing-store width in physical pixels. */
  width: number;
  /** Current backing-store height in physical pixels. */
  height: number;
  /** Capped device pixel ratio. */
  pixelRatio: number;
  /** Quality scale on top of DPR (lowers the raymarch resolution). Adaptive. */
  renderScale: number;
  /** What the browser reported for the adapter (fields may be empty strings). */
  adapterInfo: AdapterInfo;
  /**
   * True when the adapter looks like a CPU implementation (SwiftShader, llvmpipe,
   * lavapipe, or a browser "fallback" adapter). Expect single-digit fps.
   */
  software: boolean;
  /**
   * The limits actually granted on the device, after negotiating against the
   * adapter. Consumers should size scenes against these rather than assume the
   * WebGPU defaults (`maxTextureDimension3D` 2048, `maxBufferSize` 256 MiB),
   * which are well below what most adapters offer.
   */
  limits: GrantedLimits;
}

/**
 * The device limits `initGpu` negotiates, as granted. The WebGPU defaults are
 * 2048, 256 MiB and 128 MiB; most adapters grant far more when asked.
 */
export interface GrantedLimits {
  /** Largest 3D texture edge, in texels. */
  maxTextureDimension3D: number;
  /** Largest single buffer, in bytes. */
  maxBufferSize: number;
  /**
   * Caps a single storage-buffer binding. The default is only 128 MiB, well
   * under what a large sparse world's brick pool needs, and exceeding it makes
   * the bind group invalid — which surfaces as rendering silently stopping
   * rather than as an allocation failure.
   */
  maxStorageBufferBindingSize: number;
}

/** What the browser reports about the GPU adapter; any string may be empty. */
export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  /** True for a browser "fallback" (usually CPU) adapter. */
  isFallbackAdapter: boolean;
}

/** Options for `initGpu`. */
export interface GpuOptions {
  /** Cap on window.devicePixelRatio (default 2). Use 1 on weak GPUs. */
  maxPixelRatio?: number;
  /** Initial render scale on top of DPR (default 0.8). */
  renderScale?: number;
  /** Passed to requestAdapter (default "high-performance"). */
  powerPreference?: GPUPowerPreference;
  /** Log the chosen adapter to the console (default true). */
  log?: boolean;
  /**
   * Ceilings to ask the device for, each clamped to what the adapter reports.
   * Defaults are generous: a large voxel grid needs far more than the WebGPU
   * defaults allow, and asking costs nothing when the adapter can supply it.
   */
  limits?: Partial<GrantedLimits>;
}

/** Asked for by default; each is clamped to `adapter.limits` before requesting. */
const WANT_LIMITS: GrantedLimits = {
  maxTextureDimension3D: 4096,
  maxBufferSize: 1 << 30, // 1 GiB
  maxStorageBufferBindingSize: 1 << 30, // 1 GiB
};

/**
 * Thrown by `initGpu` when the page cannot use WebGPU at all: no
 * `navigator.gpu` (or not a secure context), no adapter, or no canvas context.
 * Its message is written for the user; show it with `showUnsupportedScreen`.
 */
export class WebGPUUnsupportedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WebGPUUnsupportedError";
  }
}

/** Cap the internal resolution so phones don't render at 3x retina for free. */
const MAX_PIXEL_RATIO = 2;

/**
 * Request a WebGPU adapter and device and configure `canvas` for it. Browser
 * only; WebGPU needs a secure context (HTTPS or localhost).
 *
 * Device limits are raised towards `opts.limits` (by default 4096 3D texture
 * edge, 1 GiB buffers and storage bindings), each clamped to what the adapter
 * offers; read what was granted from `limits`. If the driver refuses the raised
 * set, the device falls back to the WebGPU defaults with a console warning.
 * The canvas is sized once before this returns.
 *
 * @param canvas - The canvas to draw into.
 * @param opts - Pixel ratio cap, initial render scale, power preference, limits.
 * @returns The context to hand to `createRenderer`.
 * @throws WebGPUUnsupportedError when WebGPU is missing or unusable.
 *
 * @example
 * ```ts
 * import { initGpu, showUnsupportedScreen, WebGPUUnsupportedError } from "@voxolith/renderer";
 *
 * const canvas = document.querySelector("canvas")!;
 * try {
 *   const gpu = await initGpu(canvas, { maxPixelRatio: 2 });
 *   if (gpu.software) console.warn("CPU WebGPU adapter: start on the low preset");
 *   // ...createRenderer(gpu, scene)
 * } catch (err) {
 *   if (err instanceof WebGPUUnsupportedError) showUnsupportedScreen(err.message, { appName: "Viewer" });
 *   else throw err;
 * }
 * ```
 */
export async function initGpu(canvas: HTMLCanvasElement, opts: GpuOptions = {}): Promise<GpuContext> {
  if (!("gpu" in navigator) || !navigator.gpu) {
    // WebGPU is only exposed in a secure context. Over plain http on a LAN IP
    // (e.g. a phone hitting the dev server) `navigator.gpu` is hidden even though
    // the browser supports it — the real fix is to serve over HTTPS.
    if (!window.isSecureContext) {
      throw new WebGPUUnsupportedError(
        "This page isn't a secure context, so WebGPU is hidden. Open it over " +
          "HTTPS (or localhost) — on your phone use the https:// dev URL and " +
          "accept the certificate.",
      );
    }
    throw new WebGPUUnsupportedError("WebGPU is not available in this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: opts.powerPreference ?? "high-performance",
  });
  if (!adapter) {
    throw new WebGPUUnsupportedError("No suitable GPU adapter was found.");
  }

  // Requesting a limit the adapter cannot meet rejects the promise, so every
  // value is clamped to what the adapter reports first. Without this the device
  // runs on WebGPU defaults, and a single writeTexture of a large grid blows the
  // 256 MiB default maxBufferSize via the driver's staging buffer.
  const want = { ...WANT_LIMITS, ...opts.limits };
  const requiredLimits: Record<string, number> = {};
  const limits: GrantedLimits = { ...WANT_LIMITS };
  for (const key of Object.keys(WANT_LIMITS) as (keyof GrantedLimits)[]) {
    const supported = Number(adapter.limits[key] ?? 0);
    const asked = Math.min(want[key], supported);
    // Never ask for less than the default: that would *lower* the limit.
    if (asked > 0) requiredLimits[key] = asked;
    limits[key] = asked || supported;
  }

  const device = await adapter.requestDevice({ requiredLimits }).catch(async (err) => {
    // A driver that refuses the negotiated set is still better served than not
    // running at all; fall back to defaults and let callers size accordingly.
    console.warn("[voxolith] requestDevice with raised limits failed, using defaults:", err);
    limits.maxTextureDimension3D = 2048;
    limits.maxBufferSize = 268435456;
    limits.maxStorageBufferBindingSize = 134217728;
    return adapter.requestDevice();
  });
  for (const key of Object.keys(limits) as (keyof GrantedLimits)[]) {
    limits[key] = Number(device.limits[key] ?? limits[key]);
  }
  device.lost.then((info) => {
    // Surfaced to the console; a production build would attempt re-init here.
    console.error("WebGPU device lost:", info.message, info.reason);
  });

  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new WebGPUUnsupportedError("Could not create a WebGPU canvas context.");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  const pixelRatio = Math.min(window.devicePixelRatio || 1, opts.maxPixelRatio ?? MAX_PIXEL_RATIO);
  const adapterInfo = readAdapterInfo(adapter);
  const software = looksSoftware(adapterInfo);
  if (opts.log !== false) {
    const d = [adapterInfo.vendor, adapterInfo.architecture, adapterInfo.device, adapterInfo.description]
      .filter(Boolean)
      .join(" · ");
    console.info(`[voxolith] WebGPU adapter: ${d || "unknown"}${software ? " (SOFTWARE — expect low fps)" : ""}`);
    console.info(
      `[voxolith] limits: max 3D texture ${limits.maxTextureDimension3D}, ` +
        `max buffer ${(limits.maxBufferSize / 1048576).toFixed(0)} MiB, ` +
        `max storage binding ${(limits.maxStorageBufferBindingSize / 1048576).toFixed(0)} MiB`,
    );
  }

  context.configure({
    device,
    format,
    alphaMode: "premultiplied",
  });

  const gpu: GpuContext = {
    device,
    context,
    canvas,
    format,
    width: 1,
    height: 1,
    pixelRatio,
    renderScale: opts.renderScale ?? 0.8, // Balanced default; consumers tune it via makePerf/setRenderScale.
    adapterInfo,
    software,
    limits,
  };

  resizeToDisplay(gpu);
  return gpu;
}

function readAdapterInfo(adapter: GPUAdapter): AdapterInfo {
  // `info` is the standard; older Chromes only had the deprecated isFallbackAdapter.
  const info = (adapter as { info?: Partial<AdapterInfo> }).info ?? {};
  const legacyFallback = (adapter as { isFallbackAdapter?: boolean }).isFallbackAdapter ?? false;
  return {
    vendor: info.vendor ?? "",
    architecture: info.architecture ?? "",
    device: info.device ?? "",
    description: info.description ?? "",
    isFallbackAdapter: info.isFallbackAdapter ?? legacyFallback,
  };
}

function looksSoftware(i: AdapterInfo): boolean {
  if (i.isFallbackAdapter) return true;
  const s = `${i.vendor} ${i.architecture} ${i.device} ${i.description}`.toLowerCase();
  return /swiftshader|llvmpipe|lavapipe|softpipe|software|cpu/.test(s);
}

/** Set the render scale (0.1..1). Call resizeToDisplay afterwards, or let the frame loop do it. */
export function setRenderScale(gpu: GpuContext, scale: number): void {
  gpu.renderScale = Math.max(0.1, Math.min(1, scale));
}

/**
 * Resize the canvas backing store to match its CSS size × pixelRatio.
 * Returns true if the size actually changed.
 */
export function resizeToDisplay(gpu: GpuContext): boolean {
  const rect = gpu.canvas.getBoundingClientRect();
  const s = gpu.pixelRatio * gpu.renderScale;
  const width = Math.max(1, Math.round(rect.width * s));
  const height = Math.max(1, Math.round(rect.height * s));
  if (width === gpu.width && height === gpu.height) return false;
  gpu.canvas.width = width;
  gpu.canvas.height = height;
  gpu.width = width;
  gpu.height = height;
  return true;
}

/** Options for `showUnsupportedScreen`. */
export interface UnsupportedOpts {
  /** App name shown in the card (default "This app"). */
  appName?: string;
  /** Emoji shown at the top of the card (default "🧊"). Ignored when iconHtml is set. */
  emoji?: string;
  /** Inline HTML (typically an <svg> or <img>) shown instead of the emoji. */
  iconHtml?: string;
}

/**
 * Render a friendly "unsupported" card into the #app root instead of a blank
 * screen / crash. Used when WebGPU init throws. The card uses `.unsupported*`
 * class names; the consuming app styles them.
 */
export function showUnsupportedScreen(message: string, opts: UnsupportedOpts = {}): void {
  const { appName = "This app", emoji = "🧊", iconHtml } = opts;
  const app = document.getElementById("app");
  if (!app) return;
  const icon = iconHtml
    ? `<div class="unsupported-icon">${iconHtml}</div>`
    : `<div class="unsupported-emoji">${emoji}</div>`;
  app.innerHTML = `
    <div class="unsupported">
      <div class="unsupported-card">
        ${icon}
        <h1>Almost there!</h1>
        <p>${appName} needs <strong>WebGPU</strong> to run.</p>
        <p class="unsupported-hint">Try the latest Chrome, Edge, Firefox, or
        Safari (iOS 26+). ${message}</p>
      </div>
    </div>`;
}
