// Minimal MagicaVoxel ".vox" parser.
//
// The format is a RIFF-like tree of chunks:
//   "VOX " <version:i32>
//   chunk := <id:4 bytes> <contentSize:i32> <childrenSize:i32> <content> <children>
// The root "MAIN" chunk holds everything else as children.
//
// We read the first model only (SIZE + XYZI) plus the optional RGBA palette.
// Voxel colour indices are 1-based: XYZI index `c` -> palette entry `c`
// (file RGBA entry `i` maps to palette index `i + 1`).
//
// Reference: https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt

export interface Voxel {
  x: number;
  y: number;
  z: number;
  /** 1-based palette index. */
  c: number;
}

export interface VoxModel {
  size: { x: number; y: number; z: number };
  voxels: Voxel[];
  /** 256 RGBA entries (1024 bytes). Index by Voxel.c; entry 0 is unused. */
  palette: Uint8Array;
}

export function parseVox(buffer: ArrayBuffer): VoxModel {
  const view = new DataView(buffer);
  const magic = readTag(view, 0);
  if (magic !== "VOX ") {
    throw new Error(`Not a .vox file (bad magic "${magic}")`);
  }

  let size: VoxModel["size"] | null = null;
  const voxels: Voxel[] = [];
  let palette = defaultPalette();

  // Walk top-level chunks. We descend into MAIN's children inline because every
  // payload chunk we care about lives directly under MAIN.
  let offset = 8; // skip "VOX " + version
  while (offset + 12 <= view.byteLength) {
    const id = readTag(view, offset);
    const contentSize = view.getInt32(offset + 4, true);
    const childrenSize = view.getInt32(offset + 8, true);
    const content = offset + 12;

    switch (id) {
      case "MAIN":
        // Content is empty; children follow immediately — keep walking.
        offset = content;
        continue;
      case "SIZE":
        if (!size) {
          size = {
            x: view.getInt32(content, true),
            y: view.getInt32(content + 4, true),
            z: view.getInt32(content + 8, true),
          };
        }
        break;
      case "XYZI": {
        const count = view.getInt32(content, true);
        if (voxels.length === 0) {
          for (let i = 0; i < count; i++) {
            const p = content + 4 + i * 4;
            voxels.push({
              x: view.getUint8(p),
              y: view.getUint8(p + 1),
              z: view.getUint8(p + 2),
              c: view.getUint8(p + 3),
            });
          }
        }
        break;
      }
      case "RGBA": {
        // 256 entries; file entry i -> palette index i+1.
        const next = new Uint8Array(1024);
        for (let i = 0; i < 255; i++) {
          const src = content + i * 4;
          const dst = (i + 1) * 4;
          next[dst] = view.getUint8(src);
          next[dst + 1] = view.getUint8(src + 1);
          next[dst + 2] = view.getUint8(src + 2);
          next[dst + 3] = view.getUint8(src + 3);
        }
        palette = next;
        break;
      }
      default:
        break;
    }

    offset = content + contentSize + childrenSize;
  }

  if (!size) throw new Error("No SIZE chunk found in .vox file");
  return { size, voxels, palette };
}

// ---- extended format: scene graph + keyframe animation + materials ---------
//
// parseVox() above reads only the first model (base format, kept for its many
// callers). parseVoxScene() below reads the FULL extended format: every model,
// the nTRN/nGRP/nSHP scene graph, MATL materials, LAYR layers, and keyframes —
// then evaluates the scene at any animation frame.

/** Row-major 3×3 (a signed permutation for .vox rotations). */
export type Mat3 = [number, number, number, number, number, number, number, number, number];
const IDENTITY3: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

export interface VoxMaterial {
  type: "diffuse" | "metal" | "glass" | "emit";
  weight: number;
  rough: number;
  spec: number;
  ior: number;
  att: number;
  flux: number;
  emit: number;
  alpha: number;
  metal: number;
}

/** A model placed in the world at a given animation frame (MagicaVoxel Z-up). */
export interface Placement {
  model: VoxModel;
  rot: Mat3;
  trans: [number, number, number];
  layerId: number;
}

export interface VoxScene {
  models: VoxModel[];
  palette: Uint8Array;
  /** Indexed by palette index (0..255); null = plain diffuse. */
  materials: (VoxMaterial | null)[];
  layers: { id: number; name: string; hidden: boolean }[];
  /** Number of animation frames (max keyframe index + 1; ≥ 1). */
  frameCount: number;
  /** Model placements for a given frame. */
  sample(frame: number): Placement[];
  /** World AABB (inclusive) over ALL frames — for sizing a stable playback grid. */
  bounds(): { min: [number, number, number]; max: [number, number, number] };
}

/** Decode a MagicaVoxel `_r` rotation byte to a signed-permutation matrix (identity = 4). */
export function decodeVoxRotation(r: number): Mat3 {
  const i0 = r & 0b11;
  const i1 = (r >> 2) & 0b11;
  const i2 = 3 - i0 - i1;
  const s0 = (r >> 4) & 1 ? -1 : 1;
  const s1 = (r >> 5) & 1 ? -1 : 1;
  const s2 = (r >> 6) & 1 ? -1 : 1;
  const m: Mat3 = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  m[i0] = s0;
  m[3 + i1] = s1;
  m[6 + i2] = s2;
  return m;
}

function mat3mul(a: Mat3, b: Mat3): Mat3 {
  const o: Mat3 = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  return o;
}
function mat3vec(m: Mat3, v: [number, number, number]): [number, number, number] {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

interface TrnNode { kind: 0; child: number; layer: number; frames: { t: [number, number, number]; r: number; f: number }[] }
interface GrpNode { kind: 1; children: number[] }
interface ShpNode { kind: 2; models: { id: number; f: number }[] }
type SceneNode = TrnNode | GrpNode | ShpNode;

export function parseVoxScene(buffer: ArrayBuffer): VoxScene {
  const view = new DataView(buffer);
  if (readTag(view, 0) !== "VOX ") throw new Error("Not a .vox file");
  const i32 = (o: number) => view.getInt32(o, true);

  const readStr = (o: number): [string, number] => {
    const len = i32(o);
    let s = "";
    for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(o + 4 + i));
    return [s, o + 4 + len];
  };
  const readDict = (o: number): [Map<string, string>, number] => {
    const n = i32(o);
    o += 4;
    const d = new Map<string, string>();
    for (let i = 0; i < n; i++) {
      const [k, o1] = readStr(o);
      const [v, o2] = readStr(o1);
      d.set(k, v);
      o = o2;
    }
    return [d, o];
  };

  const models: VoxModel[] = [];
  let palette = defaultPalette();
  const materials: (VoxMaterial | null)[] = new Array(256).fill(null);
  const layers: { id: number; name: string; hidden: boolean }[] = [];
  const nodes = new Map<number, SceneNode>();
  let pendingSize: VoxModel["size"] | null = null;
  let maxFrame = 0;

  let offset = 8;
  while (offset + 12 <= view.byteLength) {
    const id = readTag(view, offset);
    const contentSize = i32(offset + 4);
    const childrenSize = i32(offset + 8);
    const c = offset + 12;

    switch (id) {
      case "MAIN":
        offset = c;
        continue;
      case "SIZE":
        pendingSize = { x: i32(c), y: i32(c + 4), z: i32(c + 8) };
        break;
      case "XYZI": {
        const count = i32(c);
        const voxels: Voxel[] = [];
        for (let i = 0; i < count; i++) {
          const p = c + 4 + i * 4;
          voxels.push({ x: view.getUint8(p), y: view.getUint8(p + 1), z: view.getUint8(p + 2), c: view.getUint8(p + 3) });
        }
        models.push({ size: pendingSize ?? { x: 0, y: 0, z: 0 }, voxels, palette });
        pendingSize = null;
        break;
      }
      case "RGBA": {
        const next = new Uint8Array(1024);
        for (let i = 0; i < 255; i++) {
          const src = c + i * 4, dst = (i + 1) * 4;
          next[dst] = view.getUint8(src);
          next[dst + 1] = view.getUint8(src + 1);
          next[dst + 2] = view.getUint8(src + 2);
          next[dst + 3] = view.getUint8(src + 3);
        }
        palette = next;
        break;
      }
      case "nTRN": {
        let o = c;
        const nodeId = i32(o); o += 4;
        [, o] = readDict(o); // node attributes
        const child = i32(o); o += 4;
        o += 4; // reserved (-1)
        const layer = i32(o); o += 4;
        const numFrames = i32(o); o += 4;
        const frames: TrnNode["frames"] = [];
        for (let fi = 0; fi < numFrames; fi++) {
          const [fd, o2] = readDict(o); o = o2;
          const t = fd.has("_t") ? (fd.get("_t")!.split(" ").map(Number) as [number, number, number]) : [0, 0, 0] as [number, number, number];
          const r = fd.has("_r") ? Number(fd.get("_r")) : 4;
          const f = fd.has("_f") ? Number(fd.get("_f")) : fi;
          maxFrame = Math.max(maxFrame, f);
          frames.push({ t, r, f });
        }
        nodes.set(nodeId, { kind: 0, child, layer, frames });
        break;
      }
      case "nGRP": {
        let o = c;
        const nodeId = i32(o); o += 4;
        [, o] = readDict(o);
        const n = i32(o); o += 4;
        const children: number[] = [];
        for (let i = 0; i < n; i++) { children.push(i32(o)); o += 4; }
        nodes.set(nodeId, { kind: 1, children });
        break;
      }
      case "nSHP": {
        let o = c;
        const nodeId = i32(o); o += 4;
        [, o] = readDict(o);
        const n = i32(o); o += 4;
        const ms: ShpNode["models"] = [];
        for (let i = 0; i < n; i++) {
          const mid = i32(o); o += 4;
          const [md, o2] = readDict(o); o = o2;
          const f = md.has("_f") ? Number(md.get("_f")) : 0;
          maxFrame = Math.max(maxFrame, f);
          ms.push({ id: mid, f });
        }
        nodes.set(nodeId, { kind: 2, models: ms });
        break;
      }
      case "MATL": {
        const matId = i32(c);
        const [d] = readDict(c + 4);
        const num = (k: string, def: number) => (d.has(k) ? parseFloat(d.get(k)!) : def);
        const tRaw = (d.get("_type") ?? "_diffuse").replace(/^_/, "");
        const type = (["diffuse", "metal", "glass", "emit"].includes(tRaw) ? tRaw : "diffuse") as VoxMaterial["type"];
        materials[matId & 255] = {
          type,
          weight: num("_weight", 1),
          rough: num("_rough", 0.1),
          spec: num("_spec", 0.5),
          ior: num("_ior", 0.3),
          att: num("_att", 0),
          flux: num("_flux", 0),
          emit: num("_emit", 0),
          alpha: num("_alpha", num("_media", 0)),
          metal: num("_metal", 0),
        };
        break;
      }
      case "LAYR": {
        const layerId = i32(c);
        const [d] = readDict(c + 4);
        layers.push({ id: layerId, name: d.get("_name") ?? "", hidden: d.get("_hidden") === "1" });
        break;
      }
      default:
        break;
    }
    offset = c + contentSize + childrenSize;
  }

  for (const m of models) m.palette = palette; // RGBA may arrive after the models
  const frameCount = Math.max(1, maxFrame + 1);

  const pickFrame = (frames: TrnNode["frames"], f: number) => {
    let best = frames[0];
    for (const fr of frames) if (fr.f <= f && fr.f >= best.f) best = fr;
    return best;
  };
  const pickModel = (ms: ShpNode["models"], f: number) => {
    let best = ms[0];
    for (const m of ms) if (m.f <= f && m.f >= best.f) best = m;
    return best;
  };

  function sample(frame: number): Placement[] {
    const out: Placement[] = [];
    const walk = (nodeId: number, rot: Mat3, trans: [number, number, number], layer: number) => {
      const node = nodes.get(nodeId);
      if (!node) return;
      if (node.kind === 0) {
        const fr = pickFrame(node.frames, frame);
        const rot2 = mat3mul(rot, decodeVoxRotation(fr.r));
        const tv = mat3vec(rot, fr.t);
        const trans2: [number, number, number] = [trans[0] + tv[0], trans[1] + tv[1], trans[2] + tv[2]];
        walk(node.child, rot2, trans2, node.layer);
      } else if (node.kind === 1) {
        for (const cId of node.children) walk(cId, rot, trans, layer);
      } else {
        const m = pickModel(node.models, frame);
        if (models[m.id]) out.push({ model: models[m.id], rot, trans, layerId: layer });
      }
    };
    if (nodes.has(0)) walk(0, IDENTITY3, [0, 0, 0], 0);
    else if (models[0]) out.push({ model: models[0], rot: IDENTITY3, trans: [0, 0, 0], layerId: 0 }); // base file, no graph
    return out;
  }

  function bounds() {
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let f = 0; f < frameCount; f++) {
      for (const p of sample(f)) {
        const s = p.model.size;
        const pivot: [number, number, number] = [Math.floor(s.x / 2), Math.floor(s.y / 2), Math.floor(s.z / 2)];
        for (let cxi = 0; cxi <= 1; cxi++)
          for (let cyi = 0; cyi <= 1; cyi++)
            for (let czi = 0; czi <= 1; czi++) {
              const local: [number, number, number] = [
                (cxi ? s.x - 1 : 0) - pivot[0],
                (cyi ? s.y - 1 : 0) - pivot[1],
                (czi ? s.z - 1 : 0) - pivot[2],
              ];
              const w = mat3vec(p.rot, local);
              for (let a = 0; a < 3; a++) {
                const c = w[a] + p.trans[a];
                if (c < min[a]) min[a] = c;
                if (c > max[a]) max[a] = c;
              }
            }
      }
    }
    if (!isFinite(min[0])) return { min: [0, 0, 0] as [number, number, number], max: [0, 0, 0] as [number, number, number] };
    return {
      min: [Math.floor(min[0]), Math.floor(min[1]), Math.floor(min[2])] as [number, number, number],
      max: [Math.ceil(max[0]), Math.ceil(max[1]), Math.ceil(max[2])] as [number, number, number],
    };
  }

  return { models, palette, materials, layers, frameCount, sample, bounds };
}

/**
 * Serialize one model to a MagicaVoxel ".vox" ArrayBuffer (SIZE + XYZI + RGBA).
 * `voxels` are `[x, y, z, colorIndex]` with a 1-based colour index; `colorFor`
 * maps a 1-based index (1..255) to an RGB triple or null (transparent/unused).
 * Pure (no fs) — callers write the returned bytes to disk themselves.
 */
export function writeVox(
  size: { x: number; y: number; z: number },
  voxels: ReadonlyArray<[number, number, number, number]>,
  colorFor: (index1: number) => [number, number, number] | null,
): ArrayBuffer {
  const n = voxels.length;
  const childrenSize = 24 + (12 + 4 + n * 4) + (12 + 1024);
  const total = 8 + 12 + childrenSize;

  const buf = new ArrayBuffer(total);
  const view = new DataView(buf);
  let o = 0;
  const tag = (s: string) => {
    for (let i = 0; i < 4; i++) view.setUint8(o++, s.charCodeAt(i));
  };
  const i32 = (v: number) => {
    view.setInt32(o, v, true);
    o += 4;
  };
  const u8 = (v: number) => view.setUint8(o++, v);

  tag("VOX ");
  i32(150);
  tag("MAIN");
  i32(0);
  i32(childrenSize);
  tag("SIZE");
  i32(12);
  i32(0);
  i32(size.x);
  i32(size.y);
  i32(size.z);
  tag("XYZI");
  i32(4 + n * 4);
  i32(0);
  i32(n);
  for (const [x, y, z, c] of voxels) {
    u8(x);
    u8(y);
    u8(z);
    u8(c);
  }
  tag("RGBA");
  i32(1024);
  i32(0);
  for (let i = 0; i < 255; i++) {
    const rgb = colorFor(i + 1); // file entry i -> palette index i+1
    if (rgb) {
      u8(rgb[0]);
      u8(rgb[1]);
      u8(rgb[2]);
      u8(255);
    } else {
      u8(0);
      u8(0);
      u8(0);
      u8(0);
    }
  }
  return buf;
}

function readTag(view: DataView, offset: number): string {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );
}

/**
 * The canonical MagicaVoxel default palette, used when a file omits an RGBA
 * chunk. Stored as 256 RGBA entries; entry 0 is the (unused) empty slot.
 * Values come from the format's default 0xAABBGGRR table, here pre-unpacked.
 */
function defaultPalette(): Uint8Array {
  const argbBgr = DEFAULT_PALETTE_ABGR;
  const out = new Uint8Array(1024);
  // File table index i (0..254) maps to palette index i+1.
  for (let i = 0; i < 255; i++) {
    const packed = argbBgr[i];
    const dst = (i + 1) * 4;
    out[dst] = packed & 0xff; // R
    out[dst + 1] = (packed >>> 8) & 0xff; // G
    out[dst + 2] = (packed >>> 16) & 0xff; // B
    out[dst + 3] = (packed >>> 24) & 0xff; // A
  }
  return out;
}

// Default palette packed as 0xAABBGGRR (MagicaVoxel's native layout).
const DEFAULT_PALETTE_ABGR: number[] = [
  0xffffffff, 0xffccffff, 0xff99ffff, 0xff66ffff, 0xff33ffff, 0xff00ffff,
  0xffffccff, 0xffccccff, 0xff99ccff, 0xff66ccff, 0xff33ccff, 0xff00ccff,
  0xffff99ff, 0xffcc99ff, 0xff9999ff, 0xff6699ff, 0xff3399ff, 0xff0099ff,
  0xffff66ff, 0xffcc66ff, 0xff9966ff, 0xff6666ff, 0xff3366ff, 0xff0066ff,
  0xffff33ff, 0xffcc33ff, 0xff9933ff, 0xff6633ff, 0xff3333ff, 0xff0033ff,
  0xffff00ff, 0xffcc00ff, 0xff9900ff, 0xff6600ff, 0xff3300ff, 0xff0000ff,
  0xffffffcc, 0xffccffcc, 0xff99ffcc, 0xff66ffcc, 0xff33ffcc, 0xff00ffcc,
  0xffffcccc, 0xffcccccc, 0xff99cccc, 0xff66cccc, 0xff33cccc, 0xff00cccc,
  0xffff99cc, 0xffcc99cc, 0xff9999cc, 0xff6699cc, 0xff3399cc, 0xff0099cc,
  0xffff66cc, 0xffcc66cc, 0xff9966cc, 0xff6666cc, 0xff3366cc, 0xff0066cc,
  0xffff33cc, 0xffcc33cc, 0xff9933cc, 0xff6633cc, 0xff3333cc, 0xff0033cc,
  0xffff00cc, 0xffcc00cc, 0xff9900cc, 0xff6600cc, 0xff3300cc, 0xff0000cc,
  0xffffff99, 0xffccff99, 0xff99ff99, 0xff66ff99, 0xff33ff99, 0xff00ff99,
  0xffffcc99, 0xffcccc99, 0xff99cc99, 0xff66cc99, 0xff33cc99, 0xff00cc99,
  0xffff9999, 0xffcc9999, 0xff999999, 0xff669999, 0xff339999, 0xff009999,
  0xffff6699, 0xffcc6699, 0xff996699, 0xff666699, 0xff336699, 0xff006699,
  0xffff3399, 0xffcc3399, 0xff993399, 0xff663399, 0xff333399, 0xff003399,
  0xffff0099, 0xffcc0099, 0xff990099, 0xff660099, 0xff330099, 0xff000099,
  0xffffff66, 0xffccff66, 0xff99ff66, 0xff66ff66, 0xff33ff66, 0xff00ff66,
  0xffffcc66, 0xffcccc66, 0xff99cc66, 0xff66cc66, 0xff33cc66, 0xff00cc66,
  0xffff9966, 0xffcc9966, 0xff999966, 0xff669966, 0xff339966, 0xff009966,
  0xffff6666, 0xffcc6666, 0xff996666, 0xff666666, 0xff336666, 0xff006666,
  0xffff3366, 0xffcc3366, 0xff993366, 0xff663366, 0xff333366, 0xff003366,
  0xffff0066, 0xffcc0066, 0xff990066, 0xff660066, 0xff330066, 0xff000066,
  0xffffff33, 0xffccff33, 0xff99ff33, 0xff66ff33, 0xff33ff33, 0xff00ff33,
  0xffffcc33, 0xffcccc33, 0xff99cc33, 0xff66cc33, 0xff33cc33, 0xff00cc33,
  0xffff9933, 0xffcc9933, 0xff999933, 0xff669933, 0xff339933, 0xff009933,
  0xffff6633, 0xffcc6633, 0xff996633, 0xff666633, 0xff336633, 0xff006633,
  0xffff3333, 0xffcc3333, 0xff993333, 0xff663333, 0xff333333, 0xff003333,
  0xffff0033, 0xffcc0033, 0xff990033, 0xff660033, 0xff330033, 0xff000033,
  0xffffff00, 0xffccff00, 0xff99ff00, 0xff66ff00, 0xff33ff00, 0xff00ff00,
  0xffffcc00, 0xffcccc00, 0xff99cc00, 0xff66cc00, 0xff33cc00, 0xff00cc00,
  0xffff9900, 0xffcc9900, 0xff999900, 0xff669900, 0xff339900, 0xff009900,
  0xffff6600, 0xffcc6600, 0xff996600, 0xff666600, 0xff336600, 0xff006600,
  0xffff3300, 0xffcc3300, 0xff993300, 0xff663300, 0xff333300, 0xff003300,
  0xffff0000, 0xffcc0000, 0xff990000, 0xff660000, 0xff330000, 0xff0000ee,
  0xff0000dd, 0xff0000bb, 0xff0000aa, 0xff000088, 0xff000077, 0xff000055,
  0xff000044, 0xff000022, 0xff000011, 0xff00ee00, 0xff00dd00, 0xff00bb00,
  0xff00aa00, 0xff008800, 0xff007700, 0xff005500, 0xff004400, 0xff002200,
  0xff001100, 0xffee0000, 0xffdd0000, 0xffbb0000, 0xffaa0000, 0xff880000,
  0xff770000, 0xff550000, 0xff440000, 0xff220000, 0xff110000, 0xffeeeeee,
  0xffdddddd, 0xffbbbbbb, 0xffaaaaaa, 0xff888888, 0xff777777, 0xff555555,
  0xff444444, 0xff222222, 0xff111111,
];
