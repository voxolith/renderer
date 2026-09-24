<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/voxolith/.github/main/profile/lockup-dark.svg">
    <img alt="Voxolith — WebGPU voxel engine" src="https://raw.githubusercontent.com/voxolith/.github/main/profile/lockup.svg" width="420">
  </picture>
</p>

# @voxolith/renderer

A WebGPU voxel raymarching engine for the browser. Formerly `voxray`, the engine behind
[Catagochi](https://github.com/voxolith/games).

- Fullscreen DDA raymarch over a dense `Uint8Array` voxel grid (palette-indexed, 256 slots)
- Coarse occupancy grid (`COARSE_B = 4`) for empty-space skipping
- Soft shadows, ambient occlusion, sky/sun/moon lighting, emissive / glass / metal materials
- Dirty-box partial re-uploads so dynamic scenes stay cheap
- Full MagicaVoxel `.vox` reader and writer (scene graph, rotations, materials, animation frames)
- Minecraft Anvil `.mca` region reader (NBT, all three chunk encodings) to a colour-mapped grid
- Orbit, pan, first-person and chase cameras; explosion and muzzle-flash effects
- Shaders are WGSL modules (`.wesl`) linked at runtime by [`wesl`](https://wesl-lang.dev)

The engine is **WebGPU only**. It needs a browser with `navigator.gpu` and a secure context
(HTTPS or `localhost`). There is no WebGL fallback; `showUnsupportedScreen` renders a friendly
message when WebGPU is missing.

## Install

```sh
bun add @voxolith/renderer
```

The package ships raw TypeScript. Your bundler compiles it together with your app, so you need:

- Vite (or another bundler that understands `import x from "./file.wesl?raw"`)
- `typescript`, `@webgpu/types` and `vite` as dev dependencies, with
  `"types": ["@webgpu/types", "vite/client"]` in your `tsconfig.json`
- In `vite.config.ts`: `optimizeDeps: { exclude: ["@voxolith/renderer"] }`

## Entry points

| import | use from | contents |
|---|---|---|
| `@voxolith/renderer` | browser code | everything, including `initGpu`, `createRenderer` |
| `@voxolith/renderer/core` | Node/bun tools | everything that does not touch the GPU |
| `@voxolith/renderer/vox` | Node/bun tools | `.vox` parse/write only |
| `@voxolith/renderer/ray` | Node/bun tools | ray/AABB helpers only |

Never import the main barrel from a script that runs outside a bundler: it pulls in the
renderer, whose `?raw` shader imports only Vite can resolve.

## Quick start

```ts
import {
  initGpu, resizeToDisplay, showUnsupportedScreen, WebGPUUnsupportedError,
  createRenderer, OccupancyGrid, makeCamera, parseVox,
} from "@voxolith/renderer";

const canvas = document.querySelector("canvas")!;
let gpu;
try {
  gpu = await initGpu(canvas);
} catch (e) {
  if (e instanceof WebGPUUnsupportedError) { showUnsupportedScreen(document.body); throw e; }
  throw e;
}

// A 32³ grid with one palette slot filled.
const size = { x: 32, y: 32, z: 32 };
const data = new Uint8Array(size.x * size.y * size.z).fill(1);
const palette = new Float32Array(256 * 4);
palette.set([0.9, 0.5, 0.2, 1], 4); // slot 1

const renderer = createRenderer(gpu, { size, data, palette });
const occ = new OccupancyGrid(size, data);
renderer.updateCoarse(occ.data);

const camera = makeCamera({ target: [16, 16, 16], distance: 80, pitchDeg: 30, fovDeg: 35 });

function frame(t: number) {
  resizeToDisplay(gpu);
  renderer.render({ ...camera(t / 40), /* lighting fields, see FrameParams */ });
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

See [voxolith/examples](https://github.com/voxolith/examples) for complete, runnable pages and
[voxolith/viewer](https://github.com/voxolith/viewer) for a full `.vox` / `.mca` viewer.

## API overview

Grouped as in `src/index.ts`:

- **Device**: `initGpu(canvas, GpuOptions?)`, `resizeToDisplay`, `setRenderScale`, `showUnsupportedScreen` (options `appName`, `emoji`, `iconHtml`), `WebGPUUnsupportedError`, `GpuContext` (includes `adapterInfo` and a `software` flag)
- **Renderer**: `createRenderer`, `Renderer` (`render`, `updateVoxels`, `updateCoarse`, `setFloor`,
  `setClipBounds`, `setQuality`, `getQuality`, `setDebug`), `QUALITY_PRESETS`, `RenderQuality`, `RenderScene`, `FrameParams`, `FloorParams`, `DirtyBox`, `raymarchShaderCode`
- **Frame loop**: `makeFrameLoop` (render on demand), `observeResize`
- **Formats**: `parseVox`, `writeVox`, `parseVoxScene`, `decodeVoxRotation`, `voxSceneAnimator`,
  `packMaterials`, `buildMinecraftRegion`
- **Acceleration**: `OccupancyGrid`, `COARSE_B`, `GridStamper` (`stamp` for movers, `writeBase` for permanent edits such as carving or rubble)
- **Cameras**: `makeCamera`, `firstPersonFrame`, `chaseFrame`. Input (orbit and first-person
  controllers, gestures, keys, gamepad, touch controls) lives in
  [`@voxolith/engine/input`](https://github.com/voxolith/engine#input); the renderer only draws.
- **Lighting**: `renderer.setLights()` (point lights with range-limited shadows and a visible
  glow, up to `MAX_LIGHTS`)
- **Atmosphere**: optional `FrameParams` fields `fog`, `clouds`, `precipitation`, `surface`,
  `waterWind` (types `AtmosphereParams` and friends)
- **Effects**: `makeExplosion`, `makeMuzzleFlash`
- **Utilities**: `rayAABB`, `makeRay`, `voxelRaycast` (CPU DDA for hit tests), `seededRandom` / `hashSeed`, `makePerf` (adaptive render scale + overlay)

`COARSE_B` is duplicated as a WGSL constant in `src/shaders/raymarch.wesl`. Keep them in sync.

## Performance controls

Everything below is a runtime knob; nothing needs a rebuild.

- **Quality** (`renderer.setQuality`): `maxSteps` caps the primary-ray DDA walk, `shadowSteps`
  caps the shadow ray (0 turns shadows off), `ao` toggles face ambient occlusion. Use a preset
  (`renderer.setQuality("low")`) or pass a partial object. `QUALITY_PRESETS.high` is the
  original look; `low` is roughly 2 to 3 times cheaper per pixel.
- **Resolution** (`gpu.renderScale`, `gpu.pixelRatio`): rays per frame scale with the square of
  these. `initGpu(canvas, { maxPixelRatio: 1 })` caps HiDPI; `makePerf({ minScale, targetMs })`
  adapts the scale to hit a frame-time target, or pin it with `perf.setScale()` or `setRenderScale()`.
- **Render on demand** (`makeFrameLoop`): a raymarcher redraws the whole screen every frame, so
  only render when something changed. Call `loop.invalidate()` from your controls (the engine's
  input takes the loop and does it for you), `observeResize(canvas, loop)` for viewport
  changes, and `loop.setContinuous(true)` only while something animates.
- **Adapter check**: `gpu.adapterInfo` is what the browser reported and `gpu.software` is true for
  CPU implementations (SwiftShader, llvmpipe, lavapipe, fallback adapters). `initGpu` logs the
  adapter to the console. If a capable GPU shows as software, the browser is not using it.
- **Slow on Linux Chrome despite a hardware adapter**: when `chrome://gpu` reports
  `Disabled Features: webgpu_on_vk_via_gl_interop`, the compositor runs on OpenGL and each WebGPU
  frame is copied through the CPU. Enabling `chrome://flags/#enable-vulkan` fixes it.

```ts
const gpu = await initGpu(canvas, { maxPixelRatio: gpu.software ? 1 : 2 });
const renderer = await createRenderer(gpu, scene);
renderer.setQuality(gpu.software ? "low" : "high");

const perf = makePerf({ enabled: false, scale: gpu.renderScale, minScale: 0.35 });
const loop = makeFrameLoop({
  render(now) {
    perf.frame(now);
    gpu.renderScale = perf.scale();
    resizeToDisplay(gpu);
    renderer.render({ ...camera(orbit.yaw(), orbit.distance(), undefined, orbit.pitch()), ...ENV });
  },
});
observeResize(canvas, loop);
// From @voxolith/engine/input: every input event invalidates the loop.
const orbit = makeOrbitController(createInput(canvas, { loop }), { distance: 120, distanceLimits: [20, 600] });
loop.invalidate();
```

## Lights and time of day

The scene has one directional key light (`lightDir`, `lightColor` in `FrameParams`) plus up to
`MAX_LIGHTS` (32) point lights:

```ts
renderer.setLights([
  { position: [120, 30, 80], color: [1, 0.7, 0.4], intensity: 2.5, range: 90, glow: 3 },
  { position: [60, 20, 40], color: [0.4, 0.6, 1], range: 40, shadows: false },
]);
loop.invalidate();
```

- `range` is hard: past it a light contributes exactly nothing, and the range test comes first,
  so a pixel no light reaches costs a loop over the list and nothing else.
- Inside the range the light fades smoothly to zero at the edge, with a soft inverse square.
- `shadows` (default on) traces one shadow ray per pixel towards the light, stopping at the light
  and capped by the quality preset's `shadowSteps`. So `low` keeps the lights but drops their
  shadows, and the cost is one short ray per shadowed light in range per pixel.
- `glow` draws a visible halo, so the lamp itself shows, including against the sky.
- Metal and glass reflections pick up the lights, unshadowed.
- `setLights` rewrites a 1.5 KB buffer, so moving a lamp every frame is cheap.

**Water** is a material kind, not a separate system: voxels whose material is `water` (kind 4)
ripple over time (pass `time` in seconds in `FrameParams`; a constant time freezes them), reflect
the scene by Fresnel, refract down to the bed (found by a second walk that treats water as empty),
fade into the water's colour with depth (`att`) and catch moving caustics. The water stays voxels,
so it streams, clips and edits like everything else.

The sun and moon discs only paint the sky; the key light, its colour and the ambient do the
lighting. Turning a time of day into those values is not the renderer's job: see `timeOfDay` in
[`@voxolith/engine/atmosphere`](https://github.com/voxolith/engine#atmosphere).

## Atmosphere

Raw effects, each off until its amount is above zero, so a scene that sets none renders as
before. The renderer knows nothing about weather; `@voxolith/engine/atmosphere` turns "rain" or
"a blizzard" into these.

- `fog`: exponential extinction towards a colour, optionally thinning with height; applied to
  hits by distance and to the sky, so the horizon dissolves.
- `clouds`: a drifting noise layer over the sky gradient; `cover` 1 is overcast and hides the sun,
  moon and stars.
- `precipitation`: rain streaks or snowflakes as analytic particles along each primary ray,
  stopped by whatever the ray hit (so roofs and trees occlude them), tilted by `fall`, never
  uploaded and unlimited in extent. Particles are kept at least a pixel wide at distance.
- `surface`: `wet` darkens the ground and adds a sheen to upward faces; `cover` blends upward
  faces towards a snow colour. On medium and high quality a short upward ray keeps it off ground
  under canopies and roofs; `low` skips that ray.
- `waterWind`: drifts and raises the ripples on water.

Cost: fog and clouds are a few arithmetic ops per pixel; precipitation is 8 hash samples; snow
shelter is one short ray per upward pixel while `cover > 0`.

## Development

```sh
bun install
bun run typecheck
```

For local development against the viewer, editor, examples and games, clone the sibling repos
next to this one and use a bun workspace root that lists them; consumers then resolve
`@voxolith/renderer` through a symlink to this checkout.

## Releasing

Publishing happens in GitHub Actions when a version tag is pushed:

```sh
# bump "version" in package.json, commit, then
git tag v0.1.0
git push origin main --tags
```

`.github/workflows/publish.yml` typechecks, verifies the tag matches `package.json`, runs
`bun publish --access public` and creates a GitHub release. It needs an `NPM_TOKEN` repository
secret holding an npm automation token with publish rights on the `@voxolith` scope.

## License

MIT
