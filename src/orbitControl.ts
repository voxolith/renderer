// Horizontal drag/swipe → camera yaw, clamped to a range so the view always
// stays inside the room (both back walls visible). Pointer events cover mouse
// and touch.

export interface OrbitControl {
  yaw(): number;
  /** Disable while in decorate mode so taps reach picking and don't orbit. */
  setEnabled(on: boolean): void;
}

export function makeOrbitControl(
  el: HTMLElement,
  opts: { start: number; min: number; max: number; sensitivity?: number; onChange?: () => void },
): OrbitControl {
  const sensitivity = opts.sensitivity ?? 0.35; // degrees per pixel
  let yaw = opts.start;
  let dragging = false;
  let enabled = true;
  let lastX = 0;

  const clamp = (v: number) => Math.max(opts.min, Math.min(opts.max, v));

  el.addEventListener("pointerdown", (e) => {
    if (!enabled) return; // let taps through to picking
    dragging = true;
    lastX = e.clientX;
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    lastX = e.clientX;
    // Drag right → scene rotates so we look further left (yaw decreases).
    const next = clamp(yaw - dx * sensitivity);
    if (next !== yaw) {
      yaw = next;
      opts.onChange?.();
    }
  });
  const end = () => {
    dragging = false;
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);

  return {
    yaw: () => yaw,
    setEnabled: (on: boolean) => {
      enabled = on;
      if (!on) dragging = false;
    },
  };
}
