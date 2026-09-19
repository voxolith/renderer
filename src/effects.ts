// Reusable voxel VFX for the raymarch engine. An effect owns a small "stage"
// grid + its own palette and mutates its `data` each `tick(dt)`; the host builds
// a Renderer from {size, data, palette} and re-uploads `data` every frame.
//
// Effects: `makeExplosion` (3 size presets — a ground-anchored blast: a billowing
// fire dome sitting on the floor with debris streaks thrown up-and-out) and
// `makeMuzzleFlash` (3 types — a brief directional gun flash that loops with a gap).

import { seededRandom } from "./random";

export type Vec3 = [number, number, number];

export interface VoxEffect {
  /** Stage grid the effect wants a Renderer built for. */
  readonly size: { x: number; y: number; z: number };
  /** The effect's own colours (fire ramp) in reserved slots; index by voxel value. */
  readonly palette: Float32Array;
  /** Current stage voxels (palette indices, 0 = empty); mutated by tick(). */
  readonly data: Uint8Array;
  /** Advance the simulation by `dt` seconds (loops on its own). */
  tick(dt: number): void;
  /** Restart from the beginning. */
  reset(): void;
}

// Fire → smoke temperature ramp (reserved palette slots 1..7).
const FIRE_COLORS: [number, number, number][] = [
  [255, 244, 214], // 1 white-hot core
  [255, 214, 92], // 2 yellow
  [255, 150, 44], // 3 orange
  [232, 80, 26], // 4 red-orange
  [150, 36, 14], // 5 deep red
  [96, 90, 88], // 6 smoke light
  [46, 43, 44], // 7 smoke dark
];
const CORE = 1, YELLOW = 2, ORANGE = 3, RED = 4, DEEP = 5, SMOKE_L = 6, SMOKE_D = 7;

function firePalette(): Float32Array {
  const palette = new Float32Array(256 * 4);
  for (let i = 0; i < FIRE_COLORS.length; i++) {
    const [r, g, b] = FIRE_COLORS[i];
    const p = (i + 1) * 4; // slots 1..7
    palette[p] = r / 255;
    palette[p + 1] = g / 255;
    palette[p + 2] = b / 255;
    palette[p + 3] = 1;
  }
  return palette;
}

function hash3(x: number, y: number, z: number, s: number): number {
  const h = Math.sin(x * 127.1 + y * 311.7 + z * 74.7 + s * 13.33) * 43758.5453;
  return h - Math.floor(h);
}
const easeOut = (u: number) => 1 - (1 - u) ** 3;

// ---- explosions ------------------------------------------------------------

// Smooth value noise (trilinear over a hashed integer lattice) → big lumps for
// the billowing cauliflower surface; animate by drifting a coordinate.
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const fade = (t: number) => t * t * (3 - 2 * t);
function hashi(ix: number, iy: number, iz: number, s: number): number {
  const h = Math.sin(ix * 127.1 + iy * 311.7 + iz * 74.7 + s * 13.33) * 43758.5453;
  return h - Math.floor(h);
}
function vnoise(x: number, y: number, z: number, s: number): number {
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
  const fx = fade(x - ix), fy = fade(y - iy), fz = fade(z - iz);
  const c = (dx: number, dy: number, dz: number) => hashi(ix + dx, iy + dy, iz + dz, s);
  const x00 = lerp(c(0, 0, 0), c(1, 0, 0), fx), x10 = lerp(c(0, 1, 0), c(1, 1, 0), fx);
  const x01 = lerp(c(0, 0, 1), c(1, 0, 1), fx), x11 = lerp(c(0, 1, 1), c(1, 1, 1), fx);
  return lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz);
}
function fbm2(x: number, y: number, z: number): number {
  return (vnoise(x, y, z, 7) * 0.65 + vnoise(x * 2.1, y * 2.1, z * 2.1, 19) * 0.35);
}
const rng = seededRandom;

const tempSlot = (h: number, n: number): number => {
  if (h > 0.72) return CORE;
  if (h > 0.5) return YELLOW;
  if (h > 0.34) return ORANGE;
  if (h > 0.2) return RED;
  if (h > 0.08) return DEEP;
  return n > 0.5 ? SMOKE_L : SMOKE_D;
};

export type ExplosionSize = "small" | "medium" | "large";

export interface ExplosionOpts {
  size?: ExplosionSize;
}

interface ExplosionParams {
  sx: number; sy: number; sz: number; // stage grid (wider than tall)
  Rh: number; // horizontal fire-dome radius (spreads along the ground)
  Ry: number; // vertical dome radius (dome is squashed: Ry < Rh)
  lifetime: number;
  ejecta: number; // debris streak count
}

const EXPLOSION_PARAMS: Record<ExplosionSize, ExplosionParams> = {
  small: { sx: 64, sy: 56, sz: 64, Rh: 20, Ry: 13, lifetime: 1.0, ejecta: 24 },
  medium: { sx: 96, sy: 88, sz: 96, Rh: 30, Ry: 19, lifetime: 1.7, ejecta: 42 },
  large: { sx: 128, sy: 112, sz: 128, Rh: 42, Ry: 26, lifetime: 2.6, ejecta: 66 },
};

interface Ejecta {
  az: number; // azimuth
  vh: number; // outward (horizontal) speed
  vy0: number; // launch vertical speed
  delay: number;
  parity: number;
}

export function makeExplosion(opts: ExplosionOpts = {}): VoxEffect {
  const P = EXPLOSION_PARAMS[opts.size ?? "medium"];
  const size = { x: P.sx, y: P.sy, z: P.sz };
  const data = new Uint8Array(P.sx * P.sy * P.sz);
  const palette = firePalette();

  const cx = Math.floor(P.sx / 2);
  const cz = Math.floor(P.sz / 2);
  const life = P.lifetime;

  // Debris ballistics: reach ~0.6×stage height at apex ~0.42×life, so streaks
  // arc up then fall back within the lifetime.
  const apex = P.sy * 0.6;
  const tApex = life * 0.42;
  const G = (2 * apex) / (tApex * tApex);
  const vyApex = G * tApex;

  const rand = rng(1337);
  const ej: Ejecta[] = [];
  for (let i = 0; i < P.ejecta; i++) {
    ej.push({
      az: rand() * Math.PI * 2,
      vh: (P.Rh * (0.6 + rand() * 2.2)) / life,
      vy0: vyApex * (0.4 + rand() * 0.6), // some tall spikes, some low
      delay: rand() * 0.05,
      parity: (rand() * 2) | 0,
    });
  }

  const put = (x: number, y: number, z: number, c: number) => {
    x = Math.round(x); y = Math.round(y); z = Math.round(z);
    if (x < 0 || x >= P.sx || y < 0 || y >= P.sy || z < 0 || z >= P.sz) return;
    data[x + y * P.sx + z * P.sx * P.sy] = c;
  };

  let age = 0;

  // Scorched debris scatter on the ground around the blast.
  function drawGround(t: number) {
    const keep = t > 0.72 ? 1 - (t - 0.72) / 0.28 : 1;
    if (keep <= 0) return;
    const R = P.Rh * (0.65 + 0.7 * easeOut(Math.min(1, t / 0.34)));
    const r = Math.ceil(R);
    for (let z = cz - r; z <= cz + r; z++)
      for (let x = cx - r; x <= cx + r; x++) {
        const rh = Math.hypot(x - cx, z - cz);
        if (rh > R || rh < R * 0.25) continue;
        const n = hashi(x, 0, z, 11);
        if (n > 0.55 * keep) continue; // sparse
        put(x, 0, z, rh > R * 0.7 ? SMOKE_D : DEEP);
      }
  }

  // The billowing fire dome anchored on the ground (flat bottom, lumpy top).
  function drawDome(t: number) {
    const grow = easeOut(Math.min(1, t / 0.34));
    const Rh = Math.max(2, P.Rh * grow);
    const Ry = Math.max(2, (P.Ry * (1 + t * 0.8)) * grow); // rises/elongates over life
    const cool = 0.55;
    const drift = age * 0.8; // smooth churn (drifts the noise field)
    const flick = Math.floor(age * 8);
    const thin = t > 0.6 ? (t - 0.6) / 0.4 : 0;

    const rx = Math.ceil(Rh * 1.55);
    const x0 = Math.max(0, cx - rx), x1 = Math.min(P.sx - 1, cx + rx);
    const z0 = Math.max(0, cz - rx), z1 = Math.min(P.sz - 1, cz + rx);
    const y1 = Math.min(P.sy - 1, Math.ceil(Ry * 1.7));

    for (let z = z0; z <= z1; z++) {
      const nz = (z - cz) / Rh;
      for (let y = 0; y <= y1; y++) {
        const ny = y / Ry;
        const rowBase = y * P.sx + z * P.sx * P.sy;
        for (let x = x0; x <= x1; x++) {
          const nx = (x - cx) / Rh;
          const q = nx * nx + ny * ny + nz * nz;
          if (q > 1.6) continue; // definitely outside → skip the noise
          let surf = 1;
          if (q > 0.4) {
            const wob = (fbm2(nx * 2 + drift, ny * 2, nz * 2 - drift) - 0.5) * 1.3;
            surf = 1 + wob * 0.6;
            if (q > surf) continue;
          }
          if (thin > 0 && hashi(x, y, z, 3) < thin * 0.85) continue;
          const nn = hashi(x, y, z, flick);
          const depth = 1 - q / surf; // 1 core → 0 surface
          // Hot deep + low-centre; cooler toward the top → smoke plume rising off it.
          const heat = depth * (1 - t * cool) - ny * 0.4 + (nn - 0.5) * 0.22 + 0.08;
          data[rowBase + x] = tempSlot(heat, nn);
        }
      }
    }
  }

  // Debris streaks: ballistic particles with a dark trail (fiery near the base).
  function drawEjecta() {
    for (const e of ej) {
      const a = age - e.delay;
      if (a <= 0) continue;
      const y = e.vy0 * a - 0.5 * G * a * a;
      if (y < 0) continue; // launched and already fallen back
      const hx = cx + Math.cos(e.az) * e.vh * a;
      const hz = cz + Math.sin(e.az) * e.vh * a;
      const vy = e.vy0 - G * a;
      let vx = Math.cos(e.az) * e.vh, vz = Math.sin(e.az) * e.vh;
      const vlen = Math.hypot(vx, vy, vz) || 1;
      vx /= vlen; const vyn = vy / vlen; vz /= vlen;
      const trail = Math.round(4 + Math.hypot(e.vh, e.vy0) * 0.05);
      const hyFrac = y / P.sy;
      for (let k = 0; k < trail; k++) {
        let c: number;
        if (k === 0 && hyFrac < 0.4) c = YELLOW;
        else if (k <= 1 && hyFrac < 0.55) c = ORANGE;
        else if (hyFrac < 0.28) c = RED;
        else c = (k + e.parity) % 2 === 0 ? SMOKE_D : SMOKE_L;
        put(hx - vx * k, y - vyn * k, hz - vz * k, c);
      }
    }
  }

  function rebuild() {
    data.fill(0);
    const t = age / life;
    drawGround(t);
    drawDome(t);
    drawEjecta();
  }

  rebuild();

  return {
    size,
    palette,
    data,
    tick(dt: number) {
      age += dt;
      if (age >= life) age -= life; // loop
      rebuild();
    },
    reset() {
      age = 0;
      rebuild();
    },
  };
  // Terrain-destruction hook (later): the sim is pure + grid-agnostic. A future
  // step can read the blast centre [cx, 0, cz] + current dome radius to carve a
  // target grid within the radius and reuse the ejecta as thrown terrain chunks.
}

// ---- muzzle flashes --------------------------------------------------------

export type MuzzleType = "star" | "cone" | "bloom";

export interface MuzzleFlashOpts {
  type?: MuzzleType;
  size?: number;
}

export function makeMuzzleFlash(opts: MuzzleFlashOpts = {}): VoxEffect {
  const S = opts.size ?? 48;
  const type = opts.type ?? "star";

  const size = { x: S, y: S, z: S };
  const data = new Uint8Array(S * S * S);
  const palette = firePalette();

  // Muzzle sits left-of-centre and fires toward +X so the flash reads in profile.
  const mx = Math.floor(S * 0.34);
  const my = Math.floor(S / 2);
  const mz = Math.floor(S / 2);

  const period = 0.42; // flash-then-gap rhythm (repeated fire)
  const flashDur = 0.13;
  let age = 0;

  const put = (x: number, y: number, z: number, c: number) => {
    x = Math.round(x); y = Math.round(y); z = Math.round(z);
    if (x < 0 || x >= S || y < 0 || y >= S || z < 0 || z >= S) return;
    data[x + y * S + z * S * S] = c;
  };

  function rebuild() {
    data.fill(0);
    if (age > flashDur) return; // dark gap between flashes
    const u = age / flashDur; // 0..1 within the flash
    const env = 1 - u; // brightness/scale envelope (sharp attack, quick decay)
    const churn = age * 40;

    // Bright core at the muzzle (all types).
    for (let dz = -1; dz <= 1; dz++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) put(mx + dx, my + dy, mz + dz, CORE);

    if (type === "star") {
      // Flat radial star in the YZ plane (a thin slab in X), spiky.
      const R = S * 0.34 * env;
      const spikes = 7;
      for (let s = 0; s < spikes; s++) {
        const ang = (s / spikes) * Math.PI * 2 + 0.3;
        const len = R * (0.55 + hash3(s, 0, 0, 1) * 0.6);
        const cyd = Math.cos(ang), czd = Math.sin(ang);
        for (let r = 1; r <= len; r++) {
          const rt = r / len;
          const c = rt < 0.4 ? CORE : rt < 0.75 ? YELLOW : ORANGE;
          const y = my + cyd * r;
          const z = mz + czd * r;
          put(mx, y, z, c);
          if (rt < 0.6) { put(mx - 1, y, z, c); put(mx + 1, y, z, c); } // thicken near core
        }
      }
    } else if (type === "cone") {
      // A forward flame jet along +X: narrow at the muzzle, fat mid, tapering.
      const len = Math.round(S * 0.5 * (0.55 + 0.45 * env));
      const maxW = S * 0.14 * env + 1;
      for (let i = 0; i <= len; i++) {
        const tt = i / len;
        const ringR = maxW * Math.sin(Math.PI * Math.min(1, tt * 1.1)) + 0.4;
        const x = mx + i;
        const ri = Math.ceil(ringR);
        for (let dz = -ri; dz <= ri; dz++)
          for (let dy = -ri; dy <= ri; dy++) {
            const rr = Math.hypot(dy, dz);
            if (rr > ringR) continue;
            const heat = (1 - tt) * (1 - rr / (ringR + 0.001)) + (hash3(x, my + dy, mz + dz, churn) - 0.5) * 0.2;
            const c = heat > 0.6 ? CORE : heat > 0.35 ? YELLOW : heat > 0.15 ? ORANGE : RED;
            put(x, my + dy, mz + dz, c);
          }
      }
    } else {
      // "bloom": a quick round pop, nudged forward — hot, no smoke.
      const R = S * 0.2 * easeOut(Math.min(1, u * 1.7)) + 1;
      const bcx = mx + R * 0.4;
      const reach = Math.ceil(R) + 1;
      for (let dz = -reach; dz <= reach; dz++)
        for (let dy = -reach; dy <= reach; dy++)
          for (let dx = -reach; dx <= reach; dx++) {
            const d = Math.hypot(dx + (bcx - mx - Math.round(bcx - mx)), dy, dz);
            const n = hash3(mx + dx, my + dy, mz + dz, churn);
            if (d > R * (1 + (n - 0.5) * 0.4)) continue;
            if (u > 0.5 && n < (u - 0.5) * 1.3) continue; // thin out as it fades
            const heat = 1 - d / R;
            const c = heat > 0.55 ? CORE : heat > 0.3 ? YELLOW : ORANGE;
            put(bcx + dx, my + dy, mz + dz, c);
          }
    }
  }

  rebuild();

  return {
    size,
    palette,
    data,
    tick(dt: number) {
      age += dt;
      if (age >= period) age -= period; // loop: flash → gap → flash
      rebuild();
    },
    reset() {
      age = 0;
      rebuild();
    },
  };
}
