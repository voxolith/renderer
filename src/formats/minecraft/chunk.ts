// Decode a chunk's block volume across all three era encodings, dispatched
// structurally + by DataVersion. Blocks are indexed YZX (i = y*256 + z*16 + x)
// within a 16×16×16 section. Minecraft is Y-up.
//
//  - Legacy  (< 1.13): Level.Sections[] with `Blocks` (+ `Add`/`Data` nibbles),
//    numeric block IDs → BlockKey = number (id*16 + data).
//  - Modern  (1.13–1.17): Level.Sections[] with `Palette` + `BlockStates`
//    (packed LongArray; indices span longs when DataVersion < 2529).
//  - 1.18+:  top-level `sections[]` with `block_states = { palette, data }`,
//    negative section Y. → BlockKey = palette entry name (string).

import type { NbtCompound, NbtValue } from "../nbt";

export type BlockKey = number | string; // 0 = air; number = legacy id*16+data; string = modern name

export interface ChunkBlocks {
  /** Inclusive world-Y range covered by present sections. */
  minY: number;
  maxY: number;
  /** Block key at chunk-local x/z (0..15) and world y. 0 = empty/air. */
  blockAt(lx: number, wy: number, lz: number): BlockKey;
}

const nibble = (arr: Uint8Array, i: number) => (i & 1 ? arr[i >> 1] >> 4 : arr[i >> 1] & 15);

/** Unpack `count` fixed-width indices from a packed LongArray. Exported for tests. */
export function unpackIndices(longs: BigInt64Array, bits: number, count: number, spanning: boolean): Uint16Array {
  const out = new Uint16Array(count);
  const mask = (1n << BigInt(bits)) - 1n;
  if (!spanning) {
    const per = Math.floor(64 / bits);
    for (let i = 0; i < count; i++) {
      const li = (i / per) | 0;
      const off = BigInt((i % per) * bits);
      out[i] = Number((BigInt.asUintN(64, longs[li]) >> off) & mask);
    }
  } else {
    for (let i = 0; i < count; i++) {
      const bitPos = i * bits;
      const li = bitPos >> 6;
      const off = bitPos & 63;
      let v = BigInt.asUintN(64, longs[li]) >> BigInt(off);
      if (off + bits > 64) v |= BigInt.asUintN(64, longs[li + 1]) << BigInt(64 - off);
      out[i] = Number(v & mask);
    }
  }
  return out;
}

interface Section {
  get(i: number): BlockKey; // i in 0..4095
}

function paletteNames(palette: NbtValue[]): string[] {
  return palette.map((e) => String((e as NbtCompound).Name ?? ""));
}

// A modern section: pre-decode its indices once so blockAt is a plain lookup.
function modernSection(palette: NbtValue[], states: BigInt64Array | undefined, dataVersion: number): Section {
  const names = paletteNames(palette);
  if (!states || palette.length <= 1) return { get: () => names[0] ?? "minecraft:air" };
  const bits = Math.max(4, Math.ceil(Math.log2(palette.length)));
  const spanning = dataVersion > 0 && dataVersion < 2529; // 1.16 made indices non-spanning
  const idx = unpackIndices(states, bits, 4096, spanning);
  return { get: (i) => names[idx[i]] ?? "minecraft:air" };
}

export function readChunkBlocks(nbt: NbtCompound): ChunkBlocks {
  const dataVersion = typeof nbt.DataVersion === "number" ? nbt.DataVersion : 0;
  const sections = new Map<number, Section>();

  const addLevelSection = (s: NbtCompound) => {
    const y = Number(s.Y);
    if (s.Blocks) {
      // Legacy numeric IDs.
      const blocks = s.Blocks as Uint8Array;
      const add = s.Add as Uint8Array | undefined;
      const data = s.Data as Uint8Array | undefined;
      sections.set(y, {
        get: (i) => {
          let id = blocks[i];
          if (add) id |= nibble(add, i) << 8;
          if (id === 0) return 0;
          return id * 16 + (data ? nibble(data, i) : 0);
        },
      });
    } else if (s.Palette) {
      sections.set(y, modernSection(s.Palette as NbtValue[], s.BlockStates as BigInt64Array | undefined, dataVersion));
    }
  };

  if (Array.isArray(nbt.sections)) {
    // 1.18+ : top-level sections with block_states.
    for (const raw of nbt.sections as NbtValue[]) {
      const s = raw as NbtCompound;
      const bs = s.block_states as NbtCompound | undefined;
      if (!bs || !Array.isArray(bs.palette)) continue;
      sections.set(Number(s.Y), modernSection(bs.palette as NbtValue[], bs.data as BigInt64Array | undefined, dataVersion));
    }
  } else {
    const level = nbt.Level as NbtCompound | undefined;
    const list = level && Array.isArray(level.Sections) ? (level.Sections as NbtValue[]) : [];
    for (const raw of list) addLevelSection(raw as NbtCompound);
  }

  let minSec = Infinity, maxSec = -Infinity;
  for (const y of sections.keys()) { if (y < minSec) minSec = y; if (y > maxSec) maxSec = y; }
  if (!isFinite(minSec)) { minSec = 0; maxSec = -1; }

  return {
    minY: minSec * 16,
    maxY: maxSec * 16 + 15,
    blockAt(lx, wy, lz) {
      const secY = wy >> 4;
      const sec = sections.get(secY);
      if (!sec) return 0;
      const ly = wy - secY * 16;
      return sec.get(ly * 256 + lz * 16 + lx);
    },
  };
}
