# AGENTS.md: renderer

`@voxolith/renderer` is the bottom layer: a WebGPU software raymarcher (DDA over a two-level
brick index) with shadows, AO, point lights, water and atmosphere, plus `.vox` and `.mca`
I/O. It is published to npm on `v*` tags (`publish.yml`, which needs `NPM_TOKEN`). It depends
on nothing else in the org; the engine and every app build on it.

## Commands

```sh
bun run --cwd renderer typecheck   # tsc --noEmit
bun run --cwd renderer verify      # headless CPU checks (tools/verify.ts), no GPU
```

CI (`ci.yml`, job `typecheck`) runs both. Neither renders. After a shader change, open an app that
exercises it (examples: orbit, world, instances, rigged, nightwood) in a WebGPU browser.

## Map

- `src/renderer.ts`: `createRenderer` and the `Renderer` class, covering:
  - uniforms (144 words; the layout is in `shaders/uniforms.wesl`);
  - the index buffer regions (`REGIONS`: tops, blocks, cells, list, inst, models, parts, subs);
  - models and instances (`addModel`, `setInstances`, sub-cell tables for crowded cells);
  - the fragment and compute pipelines (`pipeline: "auto"`), quality, debug modes and GPU timings.
- `src/brick.ts`: `BrickGrid` / `BrickPool` (8³ bricks, 64³ top cells, 4- and 8-bit payloads,
  `NEAR_BIT` on model grids). `src/sparse.ts` holds sparse models (`modelAt`).
- `src/instance.ts`: instance and pose packing (`INST_WORDS`, `PART_WORDS`, `MASK_B`,
  `POSE_HEADER`, `MAX_PARTS`), and `sampleInstance`, the CPU twin of the shader's instance
  sampling. The engine's checks use it.
- `src/stamper.ts` (`GridStamper`, movers against a dense world copy), `src/device.ts`
  (`initGpu`, features, `gpu.software`), `src/timer.ts` (timestamp queries),
  `src/perf.ts` (`makePerf`, adaptive render scale), `src/frameLoop.ts` (`makeFrameLoop`),
  `src/camera.ts`, `src/lights.ts`, `src/atmosphere.ts` (raw effect parameters).
- `src/shaders/*.wesl`: WGSL modules linked at runtime by `wesl`:
  - `raymarch.wesl`: the entry points and `shadePixel`, shared by the fragment and compute paths;
  - `trace.wesl`: the primary walk;
  - `grid.wesl`: the index lookups and instance sampling;
  - `shadow.wesl`, `ao.wesl`, `lights.wesl`, `fog.wesl`, `water.wesl`, `precip.wesl`, `sky.wesl`.
- Entry points: `src/index.ts` (the browser barrel), and `src/core.ts`, `src/vox.ts`,
  `src/ray.ts` (safe in bun and node).

## Invariants

- **Raw TypeScript.** The package ships sources. `renderer.ts` imports shaders with Vite's `?raw`,
  so:
  - browser code imports `@voxolith/renderer`;
  - bun and node scripts import `/core`, `/vox` or `/ray`, never the barrel;
  - every consumer's `vite.config.ts` has `optimizeDeps: { exclude: ["@voxolith/renderer"] }`.
- **Constants duplicated in WGSL**, kept in sync by hand:
  - `INST_WORDS`, `PART_WORDS`, `MASK_B` and the instance flag bits (`instance.ts`) match
    `grid.wesl`;
  - `BRICK_B` (8) and `TOP_B` (64) (`brick.ts`) match `COARSE_B` and `TOP_B` in `grid.wesl`;
  - `MODEL_WORDS` matches in `renderer.ts` and `grid.wesl`;
  - the uniform word map matches in `renderer.ts` (`render`) and `uniforms.wesl`.
- **Keep the shader small.** SwiftShader's compile time explodes with inlined copies of large
  functions. Keep one call site per big lookup (AO is one loop with one `isSolid`; instance
  sampling is one function). Optional features sit behind pipeline override constants
  (`INSTANCES`, `PARTS`), so scenes without them compile them away.
- **Adreno 7xx quirks** (see the comments in `grid.wesl`):
  - index arithmetic is unsigned, and lookups take voxel coordinates, not brick coordinates
    computed elsewhere;
  - signed and unsigned vectors are converted one component at a time.
- **Skipping must stay conservative.** A brick or sub-cell may be skipped only if nothing can be
  there. A hole in the image is worse than a slow frame. Compare against `setDebug(1)` (no skipping)
  when changing the walk. Step caps make capped rays differ legitimately, so compare with shadows
  off and a `fog.distance` that ends rays before `maxSteps`.
- **Parity.** The fragment and compute paths produce the same image (byte-identical on a desktop
  GPU, within 1 LSB on SwiftShader). Posed instances match `bakePose` cell for cell (the engine's
  `verify-animation`).
- **World voxels are 8-bit palette slots** (256). Instances use palettes of their own after
  them (`addPalette`).
- Performance claims need measurements: `renderer.gpuTimings()` (`timestamp-query`) gives GPU
  milliseconds per pass. Measure before and after on a real GPU, not SwiftShader.

## Working in the Voxolith repos

- **Layout.** Every Voxolith repo is checked out side by side under one bun workspace root, and
  depends on its siblings as `"workspace:*"`. Run `bun install` from that root, never inside a
  repo. [CONTRIBUTING](https://github.com/voxolith/.github/blob/main/CONTRIBUTING.md) lists
  which siblings each repo needs.
- **Toolchain: bun only.** There is no npm or node step anywhere. It is TypeScript 7 and Vite 8;
  scripts run `tsc`, `vite` and `bun tools/x.ts`. Use current dependency versions.
- **`tsconfig.base.json` is byte-identical in every repo**, because consumers compile the
  renderer's and engine's sources under their own flags. Change it everywhere or nowhere.
- **WebGPU, not WebGL.** Dev servers are HTTPS (`@vitejs/plugin-basic-ssl`), because WebGPU needs a
  secure context. Checks cannot see pixels: anything that changes what is drawn must be looked
  at in a WebGPU browser, with a before/after screenshot in the pull request.
- **Docs live on the site** ([voxolith.github.io](https://voxolith.github.io/docs/), repo
  `voxolith.github.io`). READMEs stay short and link there. The API reference is generated from
  the sources, so doc comments are published content: every exported symbol has a `/** */`, and
  entry files open with `@packageDocumentation`.
- **Credit research.** When an idea comes from a paper, cite it (authors, title, venue, DOI) in
  the code comment, in the docs (the page's References and `/docs/credits/`) and in the commit
  body. Check the citation against the paper or DataCite; don't cite from memory.
- **Prose.** British spelling in prose and comments (`colour`, `normalise`); identifiers follow the
  web platform (`lightColor`). "Voxolith" is capitalised in prose; lowercase is only for the
  wordmark.
- **Commits.** History is linear and read as prose:
  - The subject says what is now true, in plain words: no `feat:` prefixes, no trailing full
    stop, about 70 characters at most.
  - The body says why, what it costs and what it deliberately does not do, wrapped at about 72
    columns.
  - One change per commit. AI-assisted commits keep their `Co-Authored-By` trailer.
  - Pull requests are squash-merged or rebased; there are no merge commits.
  - Don't push, tag or publish unless asked.
- **Community files** (CONTRIBUTING with the AI policy, CODE_OF_CONDUCT, SECURITY, templates) live
  once in `voxolith/.github` and apply org-wide; don't copy them in here.
- **CI's job names are required checks** on `main` (rulesets). Renaming a job breaks merging.
