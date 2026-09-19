// A target-orbiting camera: given a yaw (and optional zoom distance + look-at
// target override) it produces the ray basis the renderer needs. Games drive
// the yaw/distance/target however they like (fixed iso orbit, RTS pan/zoom...).

type Vec3 = [number, number, number];

/** Camera basis the renderer needs; environment fields are merged in by the app. */
export interface CameraFrame {
  camPos: Vec3;
  camRight: Vec3;
  camUp: Vec3;
  camFwd: Vec3;
  tanHalfFov: number;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

export interface CameraConfig {
  /** Default look-at target (a `frame()` call may override it). */
  target: Vec3;
  /** Default orbit distance (a `frame()` call may override it for zoom). */
  distance: number;
  pitchDeg: number;
  fovDeg: number;
}

export function makeCamera(cfg: CameraConfig) {
  const defaultPitch = (cfg.pitchDeg * Math.PI) / 180;
  const tanHalfFov = Math.tan((cfg.fovDeg * Math.PI) / 360);

  return function frame(
    yawDeg: number,
    distance: number = cfg.distance,
    target: Vec3 = cfg.target,
    pitchDeg?: number,
  ): CameraFrame {
    const yaw = (yawDeg * Math.PI) / 180;
    const pitch = pitchDeg === undefined ? defaultPitch : (pitchDeg * Math.PI) / 180;
    const cp = Math.cos(pitch);
    const camPos: Vec3 = [
      target[0] + distance * cp * Math.sin(yaw),
      target[1] + distance * Math.sin(pitch),
      target[2] + distance * cp * Math.cos(yaw),
    ];

    const fwd = norm(sub(target, camPos));
    const right = norm(cross(fwd, [0, 1, 0]));
    const up = cross(right, fwd);

    return { camPos, camFwd: fwd, camRight: right, camUp: up, tanHalfFov };
  };
}

/**
 * First-person camera frame: look from `eye` along a yaw/pitch direction (for a
 * walkthrough where the camera sits inside the grid). yaw 0 faces +Z; pitch +
 * looks up. Clamp pitch to ~±89° upstream to avoid the up-vector degenerating.
 */
export function firstPersonFrame(eye: Vec3, yawDeg: number, pitchDeg: number, fovDeg: number): CameraFrame {
  const yaw = (yawDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;
  const cp = Math.cos(pitch);
  const fwd: Vec3 = [cp * Math.sin(yaw), Math.sin(pitch), cp * Math.cos(yaw)];
  const right = norm(cross(fwd, [0, 1, 0]));
  const up = cross(right, fwd);
  return { camPos: eye, camFwd: fwd, camRight: right, camUp: up, tanHalfFov: Math.tan((fovDeg * Math.PI) / 360) };
}

export interface ChaseOptions {
  /** Heading the target is facing, degrees; yaw 0 faces +Z (same convention as firstPersonFrame). */
  yawDeg: number;
  /** How far behind the target to place the eye (along −heading). */
  distance: number;
  /** How high above the target the eye sits. */
  height: number;
  /** Look at a point this far ahead of the target along the heading (framing leads the car). Default 0. */
  lookAhead?: number;
  fovDeg: number;
}

/**
 * Chase camera: eye sits behind + above `target` along its heading and looks
 * forward past it (arcade-driver / Outrun framing). yaw 0 faces +Z; increasing
 * yaw turns like firstPersonFrame. A cockpit/hood view is just `distance≈0` with
 * a small `height` and `target` set to the car's cabin. Renderer-agnostic — the
 * raymarcher handles a camera inside or above the grid.
 */
export function chaseFrame(target: Vec3, opts: ChaseOptions): CameraFrame {
  const yaw = (opts.yawDeg * Math.PI) / 180;
  const heading: Vec3 = [Math.sin(yaw), 0, Math.cos(yaw)]; // horizontal facing
  const camPos: Vec3 = [
    target[0] - heading[0] * opts.distance,
    target[1] + opts.height,
    target[2] - heading[2] * opts.distance,
  ];
  const ahead = opts.lookAhead ?? 0;
  const lookAt: Vec3 = [
    target[0] + heading[0] * ahead,
    target[1],
    target[2] + heading[2] * ahead,
  ];
  const fwd = norm(sub(lookAt, camPos));
  const right = norm(cross(fwd, [0, 1, 0]));
  const up = cross(right, fwd);
  return { camPos, camFwd: fwd, camRight: right, camUp: up, tanHalfFov: Math.tan((opts.fovDeg * Math.PI) / 360) };
}
