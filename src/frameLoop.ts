// Render-on-demand frame loop. A raymarcher redraws the whole screen every
// frame, so a static scene should not render at all: consumers call
// invalidate() when the camera, scene or viewport changed, and setContinuous()
// while something animates. requestAnimationFrame is only scheduled while
// there is work, so an idle viewer costs nothing.

export interface FrameLoop {
  /** Request one frame (coalesced). */
  invalidate(): void;
  /** Keep rendering every frame while on (animation, particles, held keys). */
  setContinuous(on: boolean): void;
  continuous(): boolean;
  /** Stop scheduling frames (invalidate() restarts). */
  stop(): void;
  /** Dispose observers registered with observeResize. */
  dispose(): void;
}

export interface FrameLoopOptions {
  /** Called for each rendered frame with performance.now() and dt in seconds (capped). */
  render: (now: number, dt: number) => void;
  /** Frame-rate cap for continuous mode (default 60). */
  maxFps?: number;
  /** Start in continuous mode (default false). */
  continuous?: boolean;
}

export function makeFrameLoop(opts: FrameLoopOptions): FrameLoop {
  const minDt = 1000 / (opts.maxFps ?? 60) - 1;
  let cont = opts.continuous ?? false;
  let dirty = true;
  let handle = 0;
  let last = performance.now();
  const disposers: (() => void)[] = [];

  function schedule() {
    if (handle) return;
    handle = requestAnimationFrame(tick);
  }
  function tick(now: number) {
    handle = 0;
    if (cont) schedule();
    if (cont && now - last < minDt) return; // wait for the next slot
    if (!cont && !dirty) return;
    const dt = Math.min(0.05, Math.max(0, (now - last) / 1000));
    last = now;
    dirty = false;
    opts.render(now, dt);
  }

  const loop: FrameLoop = {
    invalidate() {
      dirty = true;
      schedule();
    },
    setContinuous(on) {
      if (cont === on) return;
      cont = on;
      if (on) {
        last = performance.now();
        schedule();
      }
    },
    continuous: () => cont,
    stop() {
      if (handle) cancelAnimationFrame(handle);
      handle = 0;
      dirty = false;
    },
    dispose() {
      loop.stop();
      for (const d of disposers) d();
      disposers.length = 0;
    },
  };
  // Expose a way for observeResize to tie its lifetime to the loop.
  (loop as FrameLoop & { _own: (d: () => void) => void })._own = (d) => disposers.push(d);
  if (cont) schedule();
  return loop;
}

/**
 * Invalidate the loop whenever the element's CSS size changes (window resize,
 * split view, orientation). Returns a disposer; also disposed with the loop.
 */
export function observeResize(el: Element, loop: FrameLoop): () => void {
  const ro = new ResizeObserver(() => loop.invalidate());
  ro.observe(el);
  const dispose = () => ro.disconnect();
  (loop as FrameLoop & { _own?: (d: () => void) => void })._own?.(dispose);
  return dispose;
}
