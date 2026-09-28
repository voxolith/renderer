// A small but non-trivial instance scene for the renderer's headless checks (verify.ts): dense,
// sparse and posed models, two models sharing part boxes (so poses are shared between them), a
// removed model still named by instances, a top cell over the per-cell limit, and moving sets that
// overlap the static one.

import { seededRandom } from "../src/random";
import { makeSparse, sparseSet } from "../src/sparse";
import { partBoxes } from "../src/instance";
import type { FrameParams, Instance, ModelSource, RenderScene } from "../src/renderer";

/** The world: 256 x 128 x 256, empty (instances only). */
export function world(): RenderScene {
  return { size: { x: 256, y: 128, z: 256 }, palette: new Float32Array(256 * 4) };
}

interface Target {
  addModel(src: ModelSource): number;
  removeModel(id: number): void;
}

/** The model ids of {@link addModels}. */
export interface SceneIds {
  tree: number;
  rock: number;
  rat: number;
  ratCopy: number;
  gone: number;
  all: number[];
}

const RAT = { x: 16, y: 8, z: 8 };

/** Upload the scene's models (the same ones, in the same order, every call). */
export function addModels(r: Target): SceneIds {
  const rng = seededRandom(4242);
  const treeSize = { x: 12, y: 20, z: 12 };
  const tree = new Uint8Array(treeSize.x * treeSize.y * treeSize.z);
  for (let i = 0; i < tree.length; i++) if (rng() < 0.06) tree[i] = 1 + Math.floor(rng() * 3);
  const rockSize = { x: 20, y: 10, z: 20 };
  const rock = makeSparse(rockSize);
  for (let n = 0; n < 300; n++) sparseSet(rock, Math.floor(rng() * 20), Math.floor(rng() * 10), Math.floor(rng() * 20), 1 + Math.floor(rng() * 2));
  // A three-part body along x: parts 0, 1, 2 from left to right, each a child of the one before.
  const data = new Uint8Array(RAT.x * RAT.y * RAT.z), parts = new Uint8Array(data.length);
  let i = 0;
  for (let z = 0; z < RAT.z; z++)
    for (let y = 0; y < RAT.y; y++)
      for (let x = 0; x < RAT.x; x++, i++) {
        if (y > 5 || z < 1 || z > 6) continue;
        data[i] = 1 + (x % 3);
        parts[i] = x < 6 ? 0 : x < 11 ? 1 : 2;
      }
  const joints = [{ parent: -1, at: [3, 2, 4] as const }, { parent: 0, at: [6, 2, 4] as const }, { parent: 1, at: [11, 2, 4] as const }];
  const boxes = partBoxes(RAT, data, parts, 3);
  const ids = {
    tree: r.addModel({ size: treeSize, data: tree }),
    rock: r.addModel({ size: rockSize, sparse: rock }),
    rat: r.addModel({ size: RAT, data, parts, joints, partBoxes: boxes }),
    // A copy passing the same boxes object: its poses are shared with the rat's.
    ratCopy: r.addModel({ size: RAT, data: data.map((v, k) => (k % 7 ? v : 0)), parts, joints, partBoxes: boxes }),
    gone: r.addModel({ size: treeSize, data: tree }),
  };
  r.removeModel(ids.gone);
  return { ...ids, all: [ids.tree, ids.rock, ids.rat, ids.ratCopy] };
}

/** A pose: each part turned about its joint by an angle about z. */
function pose(angles: number[]): Float32Array {
  const out = new Float32Array(angles.length * 12);
  const at = [[3, 2], [6, 2], [11, 2]];
  angles.forEach((a, b) => {
    const c = Math.cos(a), s = Math.sin(a), [px, py] = at[b];
    out.set([c, -s, 0, px - c * px + s * py, s, c, 0, py - s * px - c * py, 0, 0, 1, 0], b * 12);
  });
  return out;
}

/** The static sets and moving sets. */
export function lists(ids: SceneIds): { staticSet: Instance[]; staticSet2: Instance[]; moving: Instance[]; moving2: Instance[] } {
  const rng = seededRandom(99);
  const poseA = pose([0.2, -0.3, 0.5]), poseB = pose([-0.4, 0.1, 0]), poseM = pose([0.7, 0.2, -0.6]);
  const staticSet: Instance[] = [];
  // More than 255 in one top cell: the rest are dropped.
  for (let k = 0; k < 300; k++) staticSet.push({ model: ids.tree, base: 300, x: 70 + rng() * 40, y: 10 + rng() * 30, z: 70 + rng() * 40, yaw: rng() * 6.3 });
  for (let k = 0; k < 200; k++) {
    const kind = rng();
    staticSet.push({
      model: kind < 0.5 ? ids.tree : ids.rock, base: 256 + Math.floor(rng() * 40),
      x: 10 + rng() * 230, y: 5 + rng() * 100, z: 10 + rng() * 230,
      yaw: kind < 0.3 ? (Math.floor(rng() * 4) * Math.PI) / 2 : rng() * 6.3,
      mirror: rng() < 0.3,
      rotation: kind > 0.9 ? [Math.cos(1), -Math.sin(1), 0, Math.sin(1), Math.cos(1), 0, 0, 0, 1] : undefined,
      anchor: kind > 0.8 ? [2, 1, 3] : undefined,
    });
  }
  for (let k = 0; k < 20; k++) staticSet.push({ model: ids.rat, base: 400, x: 20 + rng() * 200, y: 30, z: 20 + rng() * 200, yaw: rng() * 6.3, parts: poseA });
  for (let k = 0; k < 5; k++) staticSet.push({ model: ids.rat, base: 400, x: 20 + rng() * 200, y: 40, z: 20 + rng() * 200, parts: pose([rng(), rng(), rng()]) });
  for (let k = 0; k < 5; k++) staticSet.push({ model: ids.ratCopy, base: 400, x: 20 + rng() * 200, y: 50, z: 20 + rng() * 200, parts: poseA, mirror: true });
  for (let k = 0; k < 5; k++) staticSet.push({ model: ids.gone, base: 300, x: 100, y: 10, z: 100 });
  const staticSet2 = staticSet.filter((_, k) => k % 3 === 0).map((i) => (i.parts ? { ...i, parts: poseB } : i));
  const moving: Instance[] = [];
  for (let k = 0; k < 30; k++) moving.push({ model: k % 3 === 0 ? ids.rat : ids.tree, base: 400, x: 60 + rng() * 80, y: 20, z: 60 + rng() * 80, yaw: rng() * 6.3, parts: k % 3 === 0 ? poseM : undefined });
  const moving2 = moving.map((m, k) => ({ ...m, x: m.x + 40, parts: m.parts && k % 2 ? poseA : m.parts }));
  return { staticSet, staticSet2, moving, moving2 };
}

/** Instances of a list naming the removed model. */
export function removedCount(list: readonly Instance[], ids: SceneIds): number {
  return list.filter((i) => i.model === ids.gone).length;
}

/** A frame's parameters (nothing is drawn by the mock; render only needs them to be complete). */
export function frame(): FrameParams {
  const z: [number, number, number] = [0, 0, 0];
  return {
    camPos: [128, 60, -40], camRight: [1, 0, 0], camUp: [0, 1, 0], camFwd: [0, 0, 1], tanHalfFov: 0.4,
    lightDir: [0, 1, 0], lightColor: [1, 1, 1], ambientSky: z, ambientGround: z, sunDir: [0, 1, 0], moonDir: [0, -1, 0],
    sunColor: z, moonColor: z, skyTop: z, skyHorizon: z, nightFactor: 0, sunIntensity: 1, moonIntensity: 0,
  };
}
