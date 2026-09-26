// Atmospheric capabilities: the raw settings the renderer can draw.
//
// These say nothing about weather. Fog, a cloud layer, falling particles, wet
// or covered ground and wind on the water are independent effects, each off
// until its amount is above zero, so a scene that sets none of them renders
// exactly as before. What they *mean* — rain, a blizzard, a foggy morning — is
// decided above the renderer (see @voxolith/engine/atmosphere).

type Vec3 = [number, number, number];

/** Distance fog, optionally thinning with height. */
export interface FogParams {
  /** Extinction per voxel of distance; 0.002 is a light haze, 0.02 thick fog. */
  density: number;
  /** Linear RGB the fog fades to. */
  color: Vec3;
  /** Fog thins with height at this rate per voxel above y = 0. 0 = uniform. */
  heightFalloff?: number;
}

/** A cloud layer drawn in the sky. */
export interface CloudParams {
  /** 0 clear .. 1 fully overcast. */
  cover: number;
  /** Linear RGB of the clouds, already lit for the time of day. */
  color: Vec3;
  /** Drift of the cloud layer, sky units per second on x and z. */
  drift: [number, number];
}

/** Falling rain or snow: analytic particles fixed in the world, hidden by geometry, never stored. */
export interface PrecipitationParams {
  /** Streaks or flakes. */
  kind: "rain" | "snow";
  /** 0 .. 1: how many particles. */
  density: number;
  /** Fall velocity in voxels per second (wind tilts it). */
  fall: Vec3;
  /** Particle colour, already lit for the scene. */
  color?: Vec3;
}

/** Wet or snow-covered ground: a tint of upward-facing surfaces. */
export interface SurfaceParams {
  /** 0 .. 1: darker ground and a sheen on upward faces. */
  wet: number;
  /** 0 .. 1: upward faces blend to `coverColor` (settled snow). */
  cover: number;
  /** Linear RGB of the cover. */
  coverColor: Vec3;
}

/** The optional atmosphere half of FrameParams. */
export interface AtmosphereParams {
  fog?: FogParams;
  clouds?: CloudParams;
  precipitation?: PrecipitationParams;
  surface?: SurfaceParams;
  /** Wind on water: ripple drift (x, z); its length also raises the ripples. */
  waterWind?: [number, number];
  /**
   * How many voxels the effects' own sizes are measured in (default 1): water
   * wavelengths and bed fade, rain and snow particle size and layering, the
   * sky's fog distance and the snow-shelter test. A world at k times the
   * resolution passes k so they keep their size in it. Fog density and fall
   * speeds are already per voxel and are scaled by whoever sets them.
   */
  effectScale?: number;
}
