// Minecraft Anvil region (.mca) reader: an 8 KiB header (a 1024-entry location
// table + timestamps) followed by zlib/gzip-compressed NBT chunk payloads packed
// into 4 KiB sectors. Reference: https://minecraft.wiki/w/Region_file_format
//
// Decompression uses the Web `DecompressionStream` (present in browsers and in
// Node 18+), so the same code runs in VoxView and in offline tests.

import { parseNbt, type NbtCompound } from "../nbt";

/** Inflate a chunk payload. comp: 1=gzip, 2=zlib(deflate/RFC1950), 3=uncompressed. */
export async function inflate(data: Uint8Array, comp: number): Promise<Uint8Array> {
  if (comp === 3) return data;
  const format = comp === 1 ? "gzip" : "deflate"; // 2 = zlib → "deflate"
  // TS 7 lib types require an ArrayBuffer-backed view for BlobPart; chunk payloads always are.
  const stream = new Blob([data as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream(format));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export interface Region {
  /** True if the chunk (region-local 0..31) is present. */
  present(cx: number, cz: number): boolean;
  /** Parse a chunk's NBT (region-local coords). Null if absent/external. */
  chunkNbt(cx: number, cz: number): Promise<NbtCompound | null>;
  /** Number of present chunks (0..1024). */
  count(): number;
}

export function readRegion(bytes: Uint8Array): Region {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const entry = (cx: number, cz: number) => {
    const i = (cx & 31) + (cz & 31) * 32;
    const b0 = view.getUint8(i * 4), b1 = view.getUint8(i * 4 + 1), b2 = view.getUint8(i * 4 + 2);
    return { off: (b0 << 16) | (b1 << 8) | b2, sectors: view.getUint8(i * 4 + 3) };
  };

  return {
    present: (cx, cz) => entry(cx, cz).off !== 0,
    count() {
      let n = 0;
      for (let i = 0; i < 1024; i++) {
        const b0 = view.getUint8(i * 4), b1 = view.getUint8(i * 4 + 1), b2 = view.getUint8(i * 4 + 2);
        if (((b0 << 16) | (b1 << 8) | b2) !== 0) n++;
      }
      return n;
    },
    async chunkNbt(cx, cz) {
      const { off } = entry(cx, cz);
      if (off === 0) return null;
      const base = off * 4096;
      const len = view.getUint32(base, false); // big-endian
      const comp = view.getUint8(base + 4);
      if (comp >= 128) return null; // ≥128 → payload in external .mcc (unsupported)
      const payload = bytes.subarray(base + 5, base + 4 + len);
      return parseNbt(await inflate(payload, comp));
    },
  };
}
