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
}

export class WebGPUUnsupportedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WebGPUUnsupportedError";
  }
}

/** Cap the internal resolution so phones don't render at 3x retina for free. */
const MAX_PIXEL_RATIO = 2;

export async function initGpu(canvas: HTMLCanvasElement): Promise<GpuContext> {
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
    powerPreference: "high-performance",
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
  const pixelRatio = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO);

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
    renderScale: 0.8, // Balanced default; adaptively tuned in [0.6, 1.0].
  };

  resizeToDisplay(gpu);
  return gpu;
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
  /** Emoji shown at the top of the card (default "🧊"). */
  emoji?: string;
}

/**
 * Render a friendly "unsupported" card into the #app root instead of a blank
 * screen / crash. Used when WebGPU init throws. The card uses `.unsupported*`
 * class names; the consuming app styles them.
 */
export function showUnsupportedScreen(message: string, opts: UnsupportedOpts = {}): void {
  const { appName = "This app", emoji = "🧊" } = opts;
  const app = document.getElementById("app");
  if (!app) return;
  app.innerHTML = `
    <div class="unsupported">
      <div class="unsupported-card">
        <div class="unsupported-emoji">${emoji}</div>
        <h1>Almost there!</h1>
        <p>${appName} needs <strong>WebGPU</strong> to run.</p>
        <p class="unsupported-hint">Try the latest Chrome, Edge, Firefox, or
        Safari (iOS 26+). ${message}</p>
      </div>
    </div>`;
}
