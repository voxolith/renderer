// Day and night: every lighting field of FrameParams from one number.
//
// `phase` runs 0..1 over a day: 0 midnight, 0.25 sunrise, 0.5 noon, 0.75
// sunset. The sun rises in +X, arcs overhead and sets in -X, tilted towards +Z;
// the moon is opposite. By day the key light follows the sun (never lower than
// a flattering elevation); at night it becomes a dim blue moonlight, and the
// ambient drops low enough for point lights to carry the scene. Around dusk
// both keys fade out together, so the hand-over from sun to moon is invisible.
//
// Animate a transition by animating `phase`; the sky passes through sunset on
// the way.

type Vec3 = [number, number, number];

/** The lighting half of FrameParams (everything but the camera). */
export interface Lighting {
  lightDir: Vec3;
  lightColor: Vec3;
  ambientSky: Vec3;
  ambientGround: Vec3;
  sunDir: Vec3;
  moonDir: Vec3;
  sunColor: Vec3;
  moonColor: Vec3;
  skyTop: Vec3;
  skyHorizon: Vec3;
  nightFactor: number;
  sunIntensity: number;
  moonIntensity: number;
}

export interface DayNightOptions {
  /** How dark the night is, 0 (pitch black ambient) .. 1 (as bright as day). Default 0.12. */
  nightAmbient?: number;
  /** Moonlight strength relative to sunlight. Default 0.35. */
  moonlight?: number;
}

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const smooth = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};
const mix = (a: Vec3, b: Vec3, t: number): Vec3 => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const scale = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const norm = (v: Vec3): Vec3 => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};
/** Keep a light direction at least `minY` above the horizon. */
const lift = (v: Vec3, minY: number): Vec3 => norm([v[0], Math.max(v[1], minY), v[2]]);

export function dayNight(phase: number, opts: DayNightOptions = {}): Lighting {
  const p = ((phase % 1) + 1) % 1;
  const a = (p - 0.25) * Math.PI * 2;
  const elev = Math.sin(a);
  const sunDir = norm([Math.cos(a), elev, 0.35]);
  const moonDir = norm([-Math.cos(a), -elev, -0.35]);

  const dayness = smooth(-0.12, 0.2, elev); // 0 night → 1 day
  const night = 1 - dayness;
  const high = clamp(elev * 2, 0, 1); // how far into the day: warm → white

  // Sky: night → dusk → noon (the ramps catagochi settled on).
  const skyTop = mix([0.02, 0.03, 0.09], mix([0.28, 0.3, 0.6], [0.35, 0.55, 0.9], high), dayness);
  const skyHorizon = mix([0.05, 0.07, 0.16], mix([0.98, 0.56, 0.34], [0.75, 0.82, 0.92], high), dayness);
  const sunColor = mix([1.0, 0.58, 0.28], [1.0, 0.96, 0.85], high);
  const moonColor: Vec3 = [0.86, 0.9, 1.0];

  // Keys: the sun by day, the moon by night, both faded out around dusk.
  const moonK = opts.moonlight ?? 0.35;
  const sunLight = scale(mix([1.0, 0.62, 0.38], [1.0, 0.98, 0.94], high), smooth(0.35, 1, dayness));
  const moonLight = scale([0.55, 0.66, 1.0], moonK * smooth(0.35, 1, night));
  const lightDir = dayness >= 0.5 ? lift(sunDir, 0.25) : lift(moonDir, 0.35);
  const lightColor = add(sunLight, moonLight);

  // Ambient: DAYLIGHT's values by day, a blue dark by night.
  const na = opts.nightAmbient ?? 0.12;
  const ambientSky = mix(scale([0.42, 0.5, 0.85], na), [0.5, 0.53, 0.6], dayness);
  const ambientGround = mix(scale([0.25, 0.27, 0.42], na), [0.3, 0.29, 0.27], dayness);

  return {
    lightDir,
    lightColor,
    ambientSky,
    ambientGround,
    sunDir,
    moonDir,
    sunColor,
    moonColor,
    skyTop,
    skyHorizon,
    nightFactor: night,
    sunIntensity: dayness,
    moonIntensity: night * 0.7,
  };
}
