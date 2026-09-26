// Render-on-demand frame loop. A raymarcher redraws the whole screen every
// frame, so a static scene should not render at all: consumers call
// invalidate() when the camera, scene or viewport changed, and setContinuous()
// while something animates. requestAnimationFrame is only scheduled while
// there is work, so an idle viewer costs nothing.

/** A render-on-demand loop from `makeFrameLoop`. */
export interface FrameLoop {
  /** Request one frame (coalesced). */
  invalidate(): void;
  /** Keep rendering every frame while on (animation, particles, held keys). */
  setContinuous(on: boolean): void;
  /** Whether continuous mode is on. */
  continuous(): boolean;
  /** Stop scheduling frames (invalidate() restarts). */
  stop(): void;
  /** Dispose observers registered with observeResize. */
  dispose(): void;
}

/** Options for `makeFrameLoop`. */
export interface FrameLoopOptions {
  /** Called for each rendered frame with performance.now() and dt in seconds (capped). */
  render: (now: number, dt: number) => void;
  /** Frame-rate cap for continuous mode (default 60). */
  maxFps?: number;
  /** Start in continuous mode (default false). */
  continuous?: boolean;
}

/**
 * Start a render-on-demand loop over `requestAnimationFrame`. Nothing is drawn
 * until something asks: `invalidate()` requests one frame (calls within a frame
 * coalesce), `setContinuous(true)` renders every frame up to `maxFps` while
 * something animates. A loop that starts on demand draws nothing until the
 * first `invalidate()`. `dt` is capped at 50 ms, so a long pause does not
 * arrive as one huge step. Browser only.
 *
 * @param opts - The render callback, fps cap and starting mode.
 * @returns The loop; pass it to `observeResize` and to input that should redraw.
 *
 * @example
 * ```ts
 * import { makeFrameLoop, observeResize, resizeToDisplay } from "@voxolith/renderer";
 *
 * const loop = makeFrameLoop({
 *   render: () => {
 *     resizeToDisplay(gpu);
 *     renderer.render({ ...camera(yaw), ...sky });
 *   },
 * });
 * observeResize(canvas, loop);
 * loop.invalidate(); // first frame
 * slider.oninput = () => { yaw = slider.valueAsNumber; loop.invalidate(); };
 * ```
 */
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
    // The fps cap only throttles the *animation*; a pending invalidate (and the
    // initial dirty flag) always gets its frame. Without that exception a
    // continuous loop can stall forever: `last` is seeded from performance.now()
    // while `now` is the rAF frame timestamp, which may already be in the past,
    // and a tick that returns here never updates `last` to resynchronise them.
    if (!dirty && cont && now - last < minDt) return; // wait for the next slot
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
