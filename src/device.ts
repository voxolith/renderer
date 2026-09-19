// WebGPU device + canvas context setup, with a graceful fallback when the
// platform doesn't support WebGPU.

export interface GpuContext {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvas: HTMLCanvasElement;
  format: GPUTextureFormat;
  /** Current backing-store size in physical pixels. */
  width: number;
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
}

export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isFallbackAdapter: boolean;
}

export interface GpuOptions {
  /** Cap on window.devicePixelRatio (default 2). Use 1 on weak GPUs. */
  maxPixelRatio?: number;
  /** Initial render scale on top of DPR (default 0.8). */
  renderScale?: number;
  /** Passed to requestAdapter (default "high-performance"). */
  powerPreference?: GPUPowerPreference;
  /** Log the chosen adapter to the console (default true). */
  log?: boolean;
}

export class WebGPUUnsupportedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WebGPUUnsupportedError";
  }
}

/** Cap the internal resolution so phones don't render at 3x retina for free. */
const MAX_PIXEL_RATIO = 2;

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

  const device = await adapter.requestDevice();
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
