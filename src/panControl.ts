// RTS-style camera control: drag to pan the look-at target across the ground
// (XZ plane), wheel/pinch to zoom (orbit distance). Yaw is fixed. Pointer events
// cover mouse and touch. The consumer reads target()/distance() each frame and
// feeds them to makeCamera's frame(yaw, distance, target).

type Vec3 = [number, number, number];

export interface PanControl {
  /** Current look-at target on the ground. */
  target(): Vec3;
  /** Current orbit distance (zoom). */
  distance(): number;
  /** True while the pointer is dragging (so a click vs drag can be told apart). */
  dragged(): boolean;
  setEnabled(on: boolean): void;
}

export interface PanOptions {
  start: Vec3;
  distance: number;
  minDistance: number;
  maxDistance: number;
  /** World units panned per CSS pixel of drag (default 0.6). */
  panSpeed?: number;
  /** Clamp the target to these XZ bounds (optional). */
  bounds?: { minX: number; maxX: number; minZ: number; maxZ: number };
  /** Called whenever target or distance changes (render-on-demand hook). */
  onChange?: () => void;
}

export function makePanControl(el: HTMLElement, opts: PanOptions): PanControl {
  const panSpeed = opts.panSpeed ?? 0.6;
  let [tx, ty, tz] = opts.start;
  let dist = opts.distance;
  let dragging = false;
  let moved = false;
  let enabled = true;
  let lastX = 0;
  let lastY = 0;

  const clampTarget = () => {
    const b = opts.bounds;
    if (!b) return;
    tx = Math.max(b.minX, Math.min(b.maxX, tx));
    tz = Math.max(b.minZ, Math.min(b.maxZ, tz));
  };

  el.addEventListener("pointerdown", (e) => {
    if (!enabled) return;
    dragging = true;
    moved = false;
    lastX = e.clientX;
    lastY = e.clientY;
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    // Drag right/down → world slides with the cursor (target moves opposite).
    tx -= dx * panSpeed * (dist / opts.distance);
    tz -= dy * panSpeed * (dist / opts.distance);
    clampTarget();
    opts.onChange?.();
  });
  const end = () => {
    dragging = false;
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
  el.addEventListener(
    "wheel",
    (e) => {
      if (!enabled) return;
      e.preventDefault();
      dist = Math.max(
        opts.minDistance,
        Math.min(opts.maxDistance, dist * (1 + Math.sign(e.deltaY) * 0.1)),
      );
      opts.onChange?.();
    },
    { passive: false },
  );

  return {
    target: () => [tx, ty, tz],
    distance: () => dist,
    dragged: () => moved,
    setEnabled: (on: boolean) => {
      enabled = on;
      if (!on) dragging = false;
    },
  };
}
