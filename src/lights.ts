// Point lights: a lamp, a lantern, a lit window that lights what is around it.
//
// An explicit list the app owns, uploaded with `renderer.setLights()`. Each
// light has a hard range, so pixels it cannot reach cost almost nothing; inside
// it, a light can trace its own shadow ray (bounded at the light, capped by the
// quality preset's `shadowSteps`), and can draw a visible glow so the lamp
// itself shows.

type Vec3 = [number, number, number];

/**
 * A point light for `Renderer.setLights`. Emissive voxels light only
 * themselves, so a lamp that should light its surroundings needs one of these
 * at it. Cost scales with the pixels inside `range`.
 */
export interface PointLight {
  /** Grid-space position (voxel units). */
  position: Vec3;
  /** Linear RGB. */
  color: Vec3;
  /** Multiplies `color`. Default 1. */
  intensity?: number;
  /** Voxels. The light contributes exactly nothing beyond this. */
  range: number;
  /** Trace a shadow ray towards the light. Default true. */
  shadows?: boolean;
  /** Radius of the visible halo, in voxels. 0 (default) draws none. */
  glow?: number;
}

/** Lights the renderer uploads per frame; extra ones are ignored. */
export const MAX_LIGHTS = 32;

/** Floats per light in the GPU buffer: 3 vec4. Mirrors lights.wesl. */
export const LIGHT_FLOATS = 12;

/** Pack lights into the GPU layout. Returns the buffer and how many were packed. */
export function packLights(lights: readonly PointLight[], out = new Float32Array(MAX_LIGHTS * LIGHT_FLOATS)): { data: Float32Array; count: number } {
  const count = Math.min(lights.length, MAX_LIGHTS);
  out.fill(0);
  for (let i = 0; i < count; i++) {
    const l = lights[i];
    const o = i * LIGHT_FLOATS;
    const k = l.intensity ?? 1;
    out[o] = l.position[0]; out[o + 1] = l.position[1]; out[o + 2] = l.position[2];
    out[o + 3] = Math.max(0, l.range);
    out[o + 4] = l.color[0] * k; out[o + 5] = l.color[1] * k; out[o + 6] = l.color[2] * k;
    out[o + 8] = l.shadows === false ? 0 : 1;
    out[o + 9] = Math.max(0, l.glow ?? 0);
  }
  return { data: out, count };
}
