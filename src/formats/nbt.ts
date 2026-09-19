// Minimal NBT (Named Binary Tag) reader — the tree format Minecraft chunk data
// is stored in. Big-endian throughout; the root is a single named Compound.
// Reference: https://minecraft.wiki/w/NBT_format
//
// Tag payloads become plain JS: numbers for Byte/Short/Int/Float/Double, bigint
// for Long, Uint8Array for Byte_Array, string for String, arrays for List, plain
// objects for Compound, Int32Array for Int_Array, BigInt64Array for Long_Array.

export type NbtValue =
  | number
  | bigint
  | string
  | Uint8Array
  | Int32Array
  | BigInt64Array
  | NbtValue[]
  | NbtCompound;

export interface NbtCompound {
  [key: string]: NbtValue;
}

const TAG_END = 0;

export function parseNbt(bytes: Uint8Array): NbtCompound {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let o = 0;

  const u8 = () => view.getUint8(o++);
  const i16 = () => { const v = view.getInt16(o, false); o += 2; return v; };
  const u16 = () => { const v = view.getUint16(o, false); o += 2; return v; };
  const i32 = () => { const v = view.getInt32(o, false); o += 4; return v; };
  const i64 = () => { const v = view.getBigInt64(o, false); o += 8; return v; };
  const f32 = () => { const v = view.getFloat32(o, false); o += 4; return v; };
  const f64 = () => { const v = view.getFloat64(o, false); o += 8; return v; };
  const str = () => {
    const len = u16();
    let s = "";
    for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(o + i));
    o += len;
    return s;
  };

  function payload(type: number): NbtValue {
    switch (type) {
      case 1: return view.getInt8(o++);
      case 2: return i16();
      case 3: return i32();
      case 4: return i64();
      case 5: return f32();
      case 6: return f64();
      case 7: { const n = i32(); const a = bytes.subarray(o, o + n); o += n; return a; }
      case 8: return str();
      case 9: {
        const elem = u8();
        const n = i32();
        const arr: NbtValue[] = [];
        for (let i = 0; i < n; i++) arr.push(elem === TAG_END ? {} : payload(elem));
        return arr;
      }
      case 10: {
        const c: NbtCompound = {};
        for (;;) {
          const t = u8();
          if (t === TAG_END) break;
          const name = str();
          c[name] = payload(t);
        }
        return c;
      }
      case 11: { const n = i32(); const a = new Int32Array(n); for (let i = 0; i < n; i++) a[i] = i32(); return a; }
      case 12: { const n = i32(); const a = new BigInt64Array(n); for (let i = 0; i < n; i++) a[i] = i64(); return a; }
      default: throw new Error(`Unknown NBT tag type ${type} at offset ${o}`);
    }
  }

  const rootType = u8();
  if (rootType !== 10) throw new Error(`NBT root is not a Compound (type ${rootType})`);
  str(); // root name (usually empty)
  return payload(10) as NbtCompound;
}
