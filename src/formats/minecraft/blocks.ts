// Block → colour (+ optional material) tables. Two curated maps cover the two
// key spaces produced by chunk.ts: legacy numeric IDs (BlockKey = id*16 + data)
// and modern namespaced names ("minecraft:stone"). Flat per-block colours — no
// textures. A few blocks map to engine materials (emit / glass / metal).

import type { BlockKey } from "./chunk";

export type BlockMat = "emit" | "glass" | "metal";
export interface BlockInfo {
  rgb: [number, number, number];
  mat?: BlockMat;
}

const FALLBACK: BlockInfo = { rgb: [150, 150, 152] };

// Minecraft wool / clay dye order (data value 0..15).
const DYE: [number, number, number][] = [
  [233, 236, 236], [240, 118, 19], [189, 68, 179], [58, 175, 217],
  [248, 198, 39], [112, 185, 25], [237, 141, 172], [62, 68, 71],
  [142, 142, 134], [21, 137, 145], [121, 42, 172], [53, 57, 157],
  [114, 71, 40], [84, 109, 27], [161, 39, 34], [20, 21, 25],
];

// Legacy numeric block IDs (pre-1.13). Keyed by base id; a few use data.
const LEGACY: Record<number, BlockInfo> = {
  1: { rgb: [128, 128, 130] },   // stone
  2: { rgb: [95, 159, 53] },     // grass
  3: { rgb: [134, 96, 67] },     // dirt
  4: { rgb: [122, 122, 124] },   // cobblestone
  5: { rgb: [186, 150, 97] },    // planks
  7: { rgb: [40, 40, 43] },      // bedrock
  8: { rgb: [60, 110, 200] },    // water (flowing)
  9: { rgb: [60, 110, 200] },    // water
  10: { rgb: [231, 118, 30], mat: "emit" }, // lava (flowing)
  11: { rgb: [231, 118, 30], mat: "emit" }, // lava
  12: { rgb: [222, 210, 160] },  // sand
  13: { rgb: [136, 130, 127] },  // gravel
  14: { rgb: [216, 190, 60] },   // gold ore
  15: { rgb: [190, 160, 130] },  // iron ore
  16: { rgb: [60, 60, 64] },     // coal ore
  17: { rgb: [104, 78, 47] },    // log
  18: { rgb: [58, 130, 52] },    // leaves
  20: { rgb: [200, 220, 230], mat: "glass" }, // glass
  24: { rgb: [222, 210, 160] },  // sandstone
  35: { rgb: [233, 236, 236] },  // wool (recoloured by data below)
  41: { rgb: [246, 208, 62], mat: "metal" }, // gold block
  42: { rgb: [220, 222, 224], mat: "metal" }, // iron block
  43: { rgb: [160, 160, 162] },  // double stone slab
  45: { rgb: [150, 92, 76] },    // bricks
  46: { rgb: [150, 60, 40] },    // tnt
  48: { rgb: [96, 118, 92] },    // mossy cobblestone
  49: { rgb: [24, 20, 34] },     // obsidian
  56: { rgb: [110, 190, 200] },  // diamond ore
  57: { rgb: [96, 216, 208], mat: "metal" }, // diamond block
  79: { rgb: [160, 200, 240], mat: "glass" }, // ice
  80: { rgb: [245, 250, 252] },  // snow block
  82: { rgb: [160, 166, 178] },  // clay
  87: { rgb: [110, 52, 52] },    // netherrack
  89: { rgb: [245, 220, 120], mat: "emit" }, // glowstone
  95: { rgb: [200, 220, 230], mat: "glass" }, // stained glass (recoloured by data)
  98: { rgb: [122, 122, 124] },  // stone bricks
  155: { rgb: [236, 232, 224] }, // quartz block
  159: { rgb: [180, 130, 100] }, // stained clay (approx)
};

// Modern namespaced ids (1.13+). Superset of the common blocks.
const MODERN: Record<string, BlockInfo> = {
  "minecraft:stone": { rgb: [128, 128, 130] },
  "minecraft:granite": { rgb: [154, 108, 92] },
  "minecraft:diorite": { rgb: [200, 200, 202] },
  "minecraft:andesite": { rgb: [138, 138, 140] },
  "minecraft:deepslate": { rgb: [80, 80, 84] },
  "minecraft:cobblestone": { rgb: [122, 122, 124] },
  "minecraft:grass_block": { rgb: [95, 159, 53] },
  "minecraft:dirt": { rgb: [134, 96, 67] },
  "minecraft:coarse_dirt": { rgb: [120, 86, 60] },
  "minecraft:sand": { rgb: [222, 210, 160] },
  "minecraft:red_sand": { rgb: [190, 102, 50] },
  "minecraft:gravel": { rgb: [136, 130, 127] },
  "minecraft:sandstone": { rgb: [222, 210, 160] },
  "minecraft:water": { rgb: [60, 110, 200] },
  "minecraft:lava": { rgb: [231, 118, 30], mat: "emit" },
  "minecraft:bedrock": { rgb: [40, 40, 43] },
  "minecraft:oak_log": { rgb: [104, 78, 47] },
  "minecraft:spruce_log": { rgb: [78, 58, 34] },
  "minecraft:birch_log": { rgb: [200, 194, 176] },
  "minecraft:oak_planks": { rgb: [186, 150, 97] },
  "minecraft:oak_leaves": { rgb: [58, 130, 52] },
  "minecraft:spruce_leaves": { rgb: [48, 96, 60] },
  "minecraft:birch_leaves": { rgb: [110, 150, 70] },
  "minecraft:glass": { rgb: [200, 220, 230], mat: "glass" },
  "minecraft:ice": { rgb: [160, 200, 240], mat: "glass" },
  "minecraft:packed_ice": { rgb: [150, 190, 235], mat: "glass" },
  "minecraft:snow_block": { rgb: [245, 250, 252] },
  "minecraft:snow": { rgb: [245, 250, 252] },
  "minecraft:clay": { rgb: [160, 166, 178] },
  "minecraft:bricks": { rgb: [150, 92, 76] },
  "minecraft:obsidian": { rgb: [24, 20, 34] },
  "minecraft:netherrack": { rgb: [110, 52, 52] },
  "minecraft:glowstone": { rgb: [245, 220, 120], mat: "emit" },
  "minecraft:sea_lantern": { rgb: [210, 236, 226], mat: "emit" },
  "minecraft:magma_block": { rgb: [180, 70, 30], mat: "emit" },
  "minecraft:iron_block": { rgb: [220, 222, 224], mat: "metal" },
  "minecraft:gold_block": { rgb: [246, 208, 62], mat: "metal" },
  "minecraft:diamond_block": { rgb: [96, 216, 208], mat: "metal" },
  "minecraft:quartz_block": { rgb: [236, 232, 224] },
  "minecraft:stone_bricks": { rgb: [122, 122, 124] },
  "minecraft:mossy_cobblestone": { rgb: [96, 118, 92] },
  "minecraft:grass": { rgb: [95, 159, 53] },
  "minecraft:tall_grass": { rgb: [95, 159, 53] },
};

const AIR = new Set(["minecraft:air", "minecraft:cave_air", "minecraft:void_air"]);

/** Resolve a block key to colour + optional material, or null for air/empty. */
export function blockInfo(key: BlockKey): BlockInfo | null {
  if (key === 0) return null;
  if (typeof key === "string") {
    if (AIR.has(key)) return null;
    const m = MODERN[key];
    if (m) return m;
    // Recolour wool/concrete/terracotta/stained glass by their colour word.
    if (key.includes("wool") || key.includes("concrete") || key.includes("terracotta")) {
      const c = DYE_BY_NAME(key);
      if (c) return { rgb: c };
    }
    if (key.includes("stained_glass")) {
      const c = DYE_BY_NAME(key);
      if (c) return { rgb: c, mat: "glass" };
    }
    if (key.includes("leaves")) return { rgb: [58, 130, 52] };
    if (key.includes("log") || key.includes("planks") || key.includes("wood")) return { rgb: [140, 104, 62] };
    return FALLBACK;
  }
  const id = Math.floor(key / 16);
  const data = key % 16;
  if (id === 35) return { rgb: DYE[data] }; // wool
  if (id === 95) return { rgb: DYE[data], mat: "glass" }; // stained glass
  if (id === 159) return { rgb: DYE[data] }; // stained clay
  return LEGACY[id] ?? FALLBACK;
}

const DYE_NAMES = [
  "white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
  "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black",
];
// Check longest names first so "light_gray"/"light_blue" win over "gray"/"blue".
const DYE_ORDER = DYE_NAMES.map((_, i) => i).sort((a, b) => DYE_NAMES[b].length - DYE_NAMES[a].length);
function DYE_BY_NAME(name: string): [number, number, number] | null {
  for (const i of DYE_ORDER) if (name.includes(DYE_NAMES[i])) return DYE[i];
  return null;
}
