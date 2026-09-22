// Lightweight perf overlay + adaptive render-scale controller. Reads frame
// cadence (CPU performance.now deltas on RENDERED frames) into an EMA, shows it
// in a corner div, and suggests a renderScale that climbs toward maxScale when
// we comfortably hit the target and backs off quickly when frames stretch.
// Overlay enabled via ?perf=1 (consumer decides).

export interface Perf {
  /** Call once per rendered frame with performance.now(). */
  frame(now: number): void;
  /** Current adaptive render scale in [minScale, maxScale]. */
  scale(): number;
  /** Pin the scale (adaptation continues from here). */
  setScale(s: number): void;
  /** Smoothed frame time in ms. */
  frameMs(): number;
  /** Change the overlay's extra label (e.g. adapter name). */
  setLabel(label: string): void;
}

export interface PerfOptions {
  enabled: boolean;
  /** Starting scale (usually gpu.renderScale). */
  scale: number;
  /** Lowest scale the controller will go to (default 0.35). */
  minScale?: number;
  /** Highest scale (default 1). */
  maxScale?: number;
  /** Frame-time target in ms (default 16.7 = 60 fps). */
  targetMs?: number;
  /** How often the controller may change the scale (default 500 ms). */
  adaptEveryMs?: number;
  /** Fixed scale: disables adaptation when true. */
  locked?: boolean;
  /**
   * Frame gaps longer than this are treated as idle (tab hidden, render-on-
   * demand waiting) and ignored. Default 250 ms. A scene that renders
   * continuously and may legitimately be slower than that must raise it, or
   * every frame is discarded: the readout then sits at its initial value,
   * reporting 60 fps, and the scale controller never sees a sample.
   */
  maxSampleMs?: number;
  /** Extra text shown in the overlay (adapter name, quality preset, ...). */
  label?: string;
}

export function makePerf(opts: PerfOptions): Perf {
  const minScale = opts.minScale ?? 0.35;
  const maxScale = opts.maxScale ?? 1;
  const targetMs = opts.targetMs ?? 1000 / 60;
  const adaptEvery = opts.adaptEveryMs ?? 500;
  const maxSample = opts.maxSampleMs ?? 250;
  let scale = Math.max(minScale, Math.min(maxScale, opts.scale));
  let label = opts.label ?? "";
  let emaMs = targetMs;
  let last = performance.now();
  let lastAdapt = last;
  let lastShow = last;
  let samples = 0;

  let el: HTMLDivElement | null = null;
  if (opts.enabled) {
    el = document.createElement("div");
    el.className = "perf";
    el.style.cssText =
      "position:fixed;right:8px;bottom:8px;z-index:9;font:11px/1.4 ui-monospace,monospace;" +
      "padding:4px 8px;border-radius:6px;background:rgba(0,0,0,.6);color:#fff;pointer-events:none;white-space:pre";
    document.body.appendChild(el);
  }

  const round = (v: number) => Math.round(v * 100) / 100;

  function frame(now: number): void {
    const dt = now - last;
    last = now;
    // Ignore gaps (tab hidden, on-demand idle): they are not render cost.
    if (dt > 0 && dt < maxSample) {
      emaMs += (dt - emaMs) * 0.2;
      samples++;
    }

    if (!opts.locked && samples >= 8 && now - lastAdapt > adaptEvery) {
      lastAdapt = now;
      if (emaMs > targetMs * 1.25 && scale > minScale) {
        // Over budget: drop proportionally (resolution cost is ~quadratic in scale).
        const want = scale * Math.sqrt(targetMs / emaMs);
        scale = Math.max(minScale, round(Math.min(scale - 0.05, want)));
      } else if (emaMs < targetMs * 1.05 && scale < maxScale) {
        scale = Math.min(maxScale, round(scale + 0.05));
      }
    }

    if (el && now - lastShow > 250) {
      lastShow = now;
      const fps = emaMs > 0 ? 1000 / emaMs : 0;
      el.textContent = `${emaMs.toFixed(1)} ms · ${fps.toFixed(0)} fps · scale ${scale.toFixed(2)}${label ? "\n" + label : ""}`;
    }
  }

  return {
    frame,
    scale: () => scale,
    setScale: (s) => {
      scale = Math.max(minScale, Math.min(maxScale, s));
    },
    frameMs: () => emaMs,
    setLabel: (l) => {
      label = l;
    },
  };
}
