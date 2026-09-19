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

- **Device**: `initGpu`, `resizeToDisplay`, `showUnsupportedScreen` (options `appName`, `emoji`, `iconHtml`), `WebGPUUnsupportedError`, `GpuContext`
- **Renderer**: `createRenderer`, `Renderer` (`render`, `updateVoxels`, `updateCoarse`, `setFloor`,
  `setClipBounds`, `setDebug`), `RenderScene`, `FrameParams`, `FloorParams`, `DirtyBox`, `raymarchShaderCode`
- **Formats**: `parseVox`, `writeVox`, `parseVoxScene`, `decodeVoxRotation`, `voxSceneAnimator`,
  `packMaterials`, `buildMinecraftRegion`
- **Acceleration**: `OccupancyGrid`, `COARSE_B`, `GridStamper`
- **Cameras and input**: `makeCamera`, `firstPersonFrame`, `chaseFrame`, `makeOrbitControl`, `makePanControl`
- **Effects**: `makeExplosion`, `makeMuzzleFlash`
- **Utilities**: `rayAABB`, `makeRay`, `makePerf`

`COARSE_B` is duplicated as a WGSL constant in `src/shaders/raymarch.wesl`. Keep them in sync.

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
