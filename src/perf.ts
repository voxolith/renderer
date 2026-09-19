// Lightweight perf overlay + adaptive render-scale controller. Reads frame
// cadence (CPU performance.now deltas on RENDERED frames) into an EMA, shows it
// in a corner div, and suggests a renderScale that climbs toward 1.0 when we
// hold ~60fps and backs off when frames start dropping. Enabled via ?perf=1.

export interface Perf {
  /** Call once per rendered frame with performance.now(). */
  frame(now: number): void;
  /** Current adaptive render scale in [min, 1]. */
  scale(): number;
}

export function makePerf(opts: {
  enabled: boolean;
  scale: number;
  minScale?: number;
}): Perf {
  const minScale = opts.minScale ?? 0.6;
  let scale = opts.scale;
  let emaMs = 1000 / 60;
  let last = performance.now();
  let lastAdapt = last;
  let lastShow = last;

  let el: HTMLDivElement | null = null;
  if (opts.enabled) {
    el = document.createElement("div");
    el.className = "perf";
    document.body.appendChild(el);
  }

  function frame(now: number): void {
    const dt = now - last;
    last = now;
    if (dt > 0 && dt < 1000) emaMs += (dt - emaMs) * 0.1;

    // Adapt at most ~1×/sec: climb toward 1.0 while we comfortably hold 60,
    // drop when frames are clearly stretching past ~48fps.
    if (now - lastAdapt > 1000) {
      lastAdapt = now;
      if (emaMs > 21 && scale > minScale) {
        scale = Math.max(minScale, Math.round((scale - 0.1) * 100) / 100);
      } else if (emaMs < 18 && scale < 1.0) {
        scale = Math.min(1.0, Math.round((scale + 0.05) * 100) / 100);
      }
    }

    if (el && now - lastShow > 250) {
      lastShow = now;
      const fps = emaMs > 0 ? 1000 / emaMs : 0;
      el.textContent = `${emaMs.toFixed(1)} ms · ${fps.toFixed(0)} fps · scale ${scale.toFixed(2)}`;
    }
  }

  return { frame, scale: () => scale };
}
