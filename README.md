<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/voxolith/.github/main/profile/lockup-dark.svg">
    <img alt="Voxolith — WebGPU voxel engine" src="https://raw.githubusercontent.com/voxolith/.github/main/profile/lockup.svg" width="420">
  </picture>
</p>

# @voxolith/renderer

A WebGPU voxel raymarcher for the browser, formerly `voxray`, the engine behind
[Catagochi](https://github.com/voxolith/games). One fullscreen DDA pass in WGSL (`.wesl` modules
linked at runtime by [`wesl`](https://wesl-lang.dev)) walks a world stored as sparse 8³ bricks
under a two-level index, draws instanced models at any yaw with palettes of their own, and shades
them with soft shadows, ambient occlusion, up to 32 point lights, emissive, glass, metal and
animated water materials, fog, clouds, rain and snow. It reads and writes MagicaVoxel `.vox` and
reads Minecraft `.mca` regions. It is **WebGPU only**: it needs `navigator.gpu` and a secure
context (HTTPS or `localhost`), and there is no WebGL fallback.

## Install

The package is **not on npm yet**. Until it is, clone this repo next to your app and link it from
a bun workspace (`"@voxolith/renderer": "workspace:*"`); the
[installation guide](https://voxolith.github.io/docs/getting-started/installation/) has the
layout. Once published:

```sh
bun add @voxolith/renderer
```

It ships raw TypeScript that your bundler compiles with your app: use Vite (or a bundler that
understands `?raw` imports), add `typescript`, `@webgpu/types` and `vite` as dev dependencies with
`"types": ["@webgpu/types", "vite/client"]`, and set
`optimizeDeps: { exclude: ["@voxolith/renderer"] }` in `vite.config.ts`.

## Entry points

| import | use from | contents |
|---|---|---|
| `@voxolith/renderer` | browser code | everything, including `initGpu`, `createRenderer`, `makeFrameLoop` |
| `@voxolith/renderer/core` | Node/bun tools | everything that does not touch the GPU |
| `@voxolith/renderer/vox` | Node/bun tools | `.vox` parse/write only |
| `@voxolith/renderer/ray` | Node/bun tools | ray/AABB helpers only |

Never import the main barrel from a script that runs outside a bundler: it pulls in the
renderer, whose `?raw` shader imports only Vite can resolve.

## Quick start

```ts
import { initGpu, createRenderer, makeCamera, makeFrameLoop, observeResize, resizeToDisplay,
  showUnsupportedScreen, WebGPUUnsupportedError } from "@voxolith/renderer";

const canvas = document.querySelector("canvas")!;
const gpu = await initGpu(canvas).catch((e) => {
  if (e instanceof WebGPUUnsupportedError) showUnsupportedScreen(e.message, { appName: "My app" });
  throw e;
});
const size = { x: 32, y: 32, z: 32 };
const data = new Uint8Array(size.x * size.y * size.z).fill(1); // palette slot 1 everywhere
const palette = new Float32Array(256 * 4);
palette.set([0.9, 0.5, 0.2, 1], 4); // slot 1, linear RGBA
const renderer = await createRenderer(gpu, { size, data, palette }); // stored as 8³ bricks
const camera = makeCamera({ target: [16, 16, 16], distance: 80, pitchDeg: 30, fovDeg: 35 });
const loop = makeFrameLoop({ render() { resizeToDisplay(gpu); renderer.render({ ...camera(35), ...LIGHTING }); } });
observeResize(canvas, loop);
loop.invalidate();
```

`LIGHTING` is the key light and sky fields of `FrameParams`; `timeOfDay(phase)` from
`@voxolith/engine/atmosphere` produces them from one number. See
[the examples](https://voxolith.github.io/examples/) for complete pages and
[the viewer](https://voxolith.github.io/viewer/) for a full `.vox` / `.mca` app.

## Documentation

The long-form material lives on the documentation site, in the
[renderer section](https://voxolith.github.io/docs/renderer/):

- [Entry points](https://voxolith.github.io/docs/renderer/entry-points/): bundler setup, starting the GPU, a first frame, cameras, picking, effects
- [Storage](https://voxolith.github.io/docs/renderer/storage/): bricks, per-brick palettes, the two-level index, editing a brick at a time
- [Instances](https://voxolith.github.io/docs/renderer/instances/): `addModel` / `setInstances`, static and dynamic sets, instance palettes
- [Lighting and materials](https://voxolith.github.io/docs/renderer/lighting/) and [Atmosphere](https://voxolith.github.io/docs/renderer/atmosphere/): point lights, water, fog, clouds, precipitation
- [Quality and performance](https://voxolith.github.io/docs/renderer/quality-and-performance/): presets, render scale, the frame loop, software adapters
- [Formats](https://voxolith.github.io/docs/renderer/formats/): `.vox` and `.mca`
- [API reference](https://voxolith.github.io/docs/renderer/api/): every export, generated from the source

## Development

```sh
bun install
bun run typecheck
bun run verify
```

For local development against the viewer, editor, examples and games, clone the sibling repos
next to this one and use a bun workspace root that lists them; consumers then resolve
`@voxolith/renderer` through a symlink to this checkout.

`BRICK_B` and `TOP_B` in `src/brick.ts` are duplicated as WGSL constants (`COARSE_B`, `TOP_B`) in
`src/shaders/grid.wesl`. Keep them in sync.

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

## References

Instances with parts (renderer) and rigged crowds (engine) pose animated voxel models on the GPU
from one shared rest model, following the rest-space animation of:

- Holger Gruen, Carsten Benthin, Michael Kern, David McAllister. *Ray Tracing Massive Amounts of
  Animated Geometry.* Proc. ACM Comput. Graph. Interact. Tech. 9(4), Article 49 (HPG 2026).
  [doi:10.1145/3820014](https://doi.org/10.1145/3820014)
- Chih-Chen Kao, Grzegorz Makowski, Shin Fujieda, Takahiro Harada. *Voxel Deformation-Aware Neural
  Intersection Function.* Eurographics 2026 Short Papers.
  [doi:10.2312/egs.20261026](https://doi.org/10.2312/egs.20261026)

What was taken and what is Voxolith's own: [Research and credits](https://voxolith.github.io/docs/credits/).

## License

MIT
