// Atmospheric capabilities: the raw settings the renderer can draw.
//
// These say nothing about weather. Fog, a cloud layer, falling particles, wet
// or covered ground and wind on the water are independent effects, each off
// until its amount is above zero, so a scene that sets none of them renders
// exactly as before. What they *mean* — rain, a blizzard, a foggy morning — is
// decided above the renderer (see @voxolith/engine/atmosphere).

type Vec3 = [number, number, number];

export interface FogParams {
  /** Extinction per voxel of distance; 0.002 is a light haze, 0.02 thick fog. */
  density: number;
  color: Vec3;
  /** Fog thins with height at this rate per voxel above y = 0. 0 = uniform. */
  heightFalloff?: number;
}

export interface CloudParams {
  /** 0 clear .. 1 fully overcast. */
  cover: number;
  color: Vec3;
  /** Drift of the cloud layer, sky units per second on x and z. */
  drift: [number, number];
}

export interface PrecipitationParams {
  kind: "rain" | "snow";
  /** 0 .. 1: how many particles. */
  density: number;
  /** Fall velocity in voxels per second (wind tilts it). */
  fall: Vec3;
  /** Particle colour, already lit for the scene. */
  color?: Vec3;
}

export interface SurfaceParams {
  /** 0 .. 1: darker ground and a sheen on upward faces. */
  wet: number;
  /** 0 .. 1: upward faces blend to `coverColor` (settled snow). */
  cover: number;
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
}
