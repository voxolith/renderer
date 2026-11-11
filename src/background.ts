// Backgrounds: what a ray that leaves the world shows (the sky, in most scenes).
//
// The renderer draws no sky of its own. A background is a WESL module the host hands to
// `createRenderer` (`RendererOptions.background`), linked into the raymarch shader in place of
// `package::userbg`. It exports
//
//   fn background(rd : vec3<f32>, footprint : f32) -> vec3<f32>
//
// and may import `package::bg::{BG, bgTex0, bgTex1, bgSampler}`: 64 floats of per-frame
// parameters (`FrameParams.background`), the frame time and two optional textures. Headless and
// worker-safe: nothing here touches the GPU, so bun tools can import it from `/core`.

type Vec3 = [number, number, number];

/** How many floats `FrameParams.background` carries (`BG.params`, 16 vec4). */
export const BACKGROUND_FLOATS = 64;

/**
 * A background for `RendererOptions.background`: the WESL module that colours rays leaving
 * the world, and the textures it samples.
 */
export interface BackgroundOptions {
  /**
   * A WESL module exporting `fn background(rd: vec3<f32>, footprint: f32) -> vec3<f32>`: the
   * linear RGB seen along the unit direction `rd` (grid space, y up). `footprint` is the angular
   * size of what the ray stands for, in radians: one pixel at the centre of the view for primary
   * rays (`2 · tanHalfFov / height`, at the render resolution), twice that for reflections (water,
   * metal, glass), which are rarely flat mirrors. Use it to anti-alias small things (stars, discs).
   *
   * The module may `import package::bg::{BG, bgTex0, bgTex1, bgSampler};`:
   * - `BG.params: array<vec4<f32>, 16>` is `FrameParams.background` (zero past what was given);
   * - `BG.time: f32` is `FrameParams.time`;
   * - `bgTex0`, `bgTex1: texture_2d<f32>` are {@link BackgroundOptions.textures} (1x1 black when
   *   not given);
   * - `bgSampler` filters linearly, repeating in u and clamped in v (for longitude-latitude maps).
   *
   * Sample with `textureSampleLevel`: the background runs in compute kernels and inside
   * non-uniform control flow, where `textureSample` is not allowed. It is called from a few places
   * in the shader (primary misses, reflections, the floor's fade), so keep it small: SwiftShader's
   * compile time grows with every inlined copy.
   */
  source: string;
  /** Up to two filterable 2D float texture views, bound as `bgTex0` and `bgTex1`. */
  textures?: [GPUTextureView?, GPUTextureView?];
}

/**
 * The renderer's default background: a plain two-colour sky, `params[0].rgb` at the horizon (and
 * everywhere below it), blending smoothly to `params[1].rgb` from the horizon up to 0.6 of the way
 * to the zenith (`smoothstep(0, 0.6, rd.y)`). Nothing else: no sun, stars or clouds. Small, so it
 * costs SwiftShader little to compile. Pack its parameters with {@link gradientBackgroundParams};
 * without `FrameParams.background` it draws {@link GRADIENT_BACKGROUND_DEFAULTS}.
 */
export const GRADIENT_BACKGROUND_WESL = /* wgsl */ `
import package::bg::BG;

fn background(rd : vec3<f32>, footprint : f32) -> vec3<f32> {
  return mix(BG.params[0].xyz, BG.params[1].xyz, smoothstep(0.0, 0.6, max(rd.y, 0.0)));
}
`;

/**
 * The colours the default background ({@link GRADIENT_BACKGROUND_WESL}) draws when a frame gives
 * no `FrameParams.background`: a neutral, slightly hazy daylight blue, linear RGB.
 */
export const GRADIENT_BACKGROUND_DEFAULTS: { readonly horizon: Readonly<Vec3>; readonly zenith: Readonly<Vec3> } = {
  horizon: [0.75, 0.82, 0.92],
  zenith: [0.35, 0.55, 0.9],
};

/**
 * Pack the parameters {@link GRADIENT_BACKGROUND_WESL} reads (`FrameParams.background`): the
 * horizon colour, then the zenith colour, linear RGB (defaults {@link GRADIENT_BACKGROUND_DEFAULTS}).
 * Writes into `out` when given (at least {@link BACKGROUND_FLOATS} long, zeroed past the colours),
 * so a per-frame caller allocates nothing.
 *
 * @example
 * ```ts
 * renderer.render({ ...camera(35), ...lighting, background: gradientBackgroundParams([0.9, 0.7, 0.5], [0.2, 0.3, 0.6]) });
 * ```
 */
export function gradientBackgroundParams(
  horizon: Readonly<Vec3> = GRADIENT_BACKGROUND_DEFAULTS.horizon,
  zenith: Readonly<Vec3> = GRADIENT_BACKGROUND_DEFAULTS.zenith,
  out = new Float32Array(BACKGROUND_FLOATS),
): Float32Array {
  out.fill(0);
  out[0] = horizon[0]; out[1] = horizon[1]; out[2] = horizon[2];
  out[4] = zenith[0]; out[5] = zenith[1]; out[6] = zenith[2];
  return out;
}
