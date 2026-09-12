import { GAME } from "../config";

const DEG = Math.PI / 180;

export type Mode = "strafe" | "static" | "duo" | "calib";

const LIFETIME_MS = 6000; // a female that isn't shot flies off and another lands
const CALIB_MS = 1200; // calibration mode: teleport this often

export interface Target {
  az: number; // world azimuth, radians
  el: number; // world elevation, radians
  alive: boolean;
  vel: number; // rad/s of azimuth drift
  velEl: number; // rad/s of elevation drift
  nextTurn: number; // brain ms
  spawnedAt: number;
  respawnAt: number;
  flash: number;
  variant: number; // which photo
  facing: number; // +1 facing right, -1 facing left
}

export interface Stats {
  shots: number;
  hits: number;
  kills: number;
  score: number;
  ttkSum: number;
  onTargetMs: number;
  playMs: number;
}

/** One line of a Call of Duty style score popup. */
export interface Award {
  points: number;
  label: string;
}

const KILL_POINTS = 100;
const QUICK_KILL_MS = 2000; // shot within this long of her landing
const CHAIN_MS = 3000; // kills closer together than this chain into a multi kill
const CHAIN_NAMES = ["", "", "Double Kill", "Triple Kill"];

const emptyStats = (): Stats => ({ shots: 0, hits: 0, kills: 0, score: 0, ttkSum: 0, onTargetMs: 0, playMs: 0 });

const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
let random = Math.random;
/** Makes target spawns and movement reproducible (?seed=N), so settings can be compared on the same targets. */
export function seedRandom(seed: number) {
  let s = seed >>> 0;
  random = () => {
    // mulberry32
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = (lo: number, hi: number) => lo + random() * (hi - lo);

/** Aim-trainer rules. All times are brain time, so a slow GPU slows the game, not the fly. */
export class Game {
  heading = 0; // radians, + = right
  pitch = 0; // radians, + = up
  time = 0;
  mode: Mode = "strafe";
  radiusDeg = GAME.targetRadiusDeg;
  targets: Target[] = [];
  stats: Stats = emptyStats();
  lastShot = -1e9;
  private spawns = 0;
  private lastKill = -1e9;
  private chain = 0;
  onHit: ((t: Target, awards: Award[]) => void) | null = null;

  constructor() {
    this.reset();
  }

  reset() {
    this.stats = emptyStats();
    this.chain = 0;
    this.lastKill = -1e9;
    const count = this.mode === "duo" ? 2 : 1;
    this.targets = Array.from({ length: count }, () => this.spawn({} as Target));
  }

  private spawn(t: Target): Target {
    const side = random() < 0.5 ? -1 : 1;
    if (this.mode === "calib") {
      t.az = wrap(this.heading + rand(-40, 40) * DEG);
      t.el = rand(-1, 1) * this.maxEl();
    } else {
      t.az = wrap(this.heading + side * rand(35, 100) * DEG);
      t.el = rand(-0.7, 0.7) * this.maxEl();
    }
    t.alive = true;
    const moving = this.mode === "strafe" || this.mode === "duo";
    t.facing = random() < 0.5 ? -1 : 1;
    t.vel = moving ? t.facing * rand(10, 28) * DEG : 0;
    t.velEl = moving ? (random() < 0.5 ? -1 : 1) * rand(3, 10) * DEG : 0;
    t.nextTurn = this.time + rand(900, 2500);
    t.spawnedAt = this.time;
    t.respawnAt = 0;
    t.flash = 0;
    t.variant = this.spawns++;
    return t;
  }

  /** Highest |elevation| a target centre may reach: the whole fly stays within the neck's pitch range. */
  maxEl(): number {
    return Math.max(0, GAME.maxPitchDeg - this.radiusDeg) * DEG;
  }

  /** Unit vector the crosshair points along. */
  private forward(): [number, number, number] {
    const cp = Math.cos(this.pitch);
    return [Math.sin(this.heading) * cp, Math.sin(this.pitch), -Math.cos(this.heading) * cp];
  }

  /** Angle between the crosshair and a target's centre, radians. */
  offset(t: Target): number {
    const [fx, fy, fz] = this.forward();
    const ce = Math.cos(t.el);
    const dot = fx * Math.sin(t.az) * ce + fy * Math.sin(t.el) - fz * Math.cos(t.az) * ce;
    return Math.acos(Math.max(-1, Math.min(1, dot)));
  }

  relAz(t: Target): number {
    return wrap(t.az - this.heading);
  }

  relEl(t: Target): number {
    return t.el - this.pitch;
  }

  update(brainMs: number, yawRateDeg: number, pitchRateDeg: number) {
    const dt = brainMs / 1000;
    this.time += brainMs;
    this.stats.playMs += brainMs;
    this.heading = wrap(this.heading + yawRateDeg * DEG * dt);
    this.pitch = Math.max(-GAME.maxPitchDeg * DEG, Math.min(GAME.maxPitchDeg * DEG, this.pitch + pitchRateDeg * DEG * dt));
    let onTarget = false;
    for (const t of this.targets) {
      t.flash = Math.max(0, t.flash - dt * 4);
      if (!t.alive) {
        if (this.time >= t.respawnAt) this.spawn(t);
        continue;
      }
      if (this.time - t.spawnedAt > (this.mode === "calib" ? CALIB_MS : LIFETIME_MS)) {
        this.spawn(t);
        continue;
      }
      if (this.mode === "strafe" || this.mode === "duo") {
        if (this.time >= t.nextTurn) {
          t.vel = -t.vel * rand(0.7, 1.3);
          t.facing = Math.sign(t.vel) || t.facing;
          t.velEl = (random() < 0.5 ? -1 : 1) * rand(3, 10) * DEG;
          t.nextTurn = this.time + rand(900, 2500);
        }
        t.az = wrap(t.az + t.vel * dt);
        t.el += t.velEl * dt;
        if (Math.abs(t.el) > this.maxEl()) {
          t.el = Math.sign(t.el) * this.maxEl();
          t.velEl = -t.velEl;
        }
      }
      if (this.offset(t) < this.radiusDeg * DEG) onTarget = true;
    }
    if (onTarget) this.stats.onTargetMs += brainMs;
  }

  /** The fly sang a song bout: fire if off cooldown. Returns the shot (and what it hit), or null. */
  trigger(): { hit: Target | null } | null {
    if (this.time - this.lastShot < GAME.fireCooldown) return null;
    this.lastShot = this.time;
    this.stats.shots++;
    const hitbox = this.radiusDeg * DEG * 0.7; // roughly the body, not the wing tips
    const hit = this.targets.filter((t) => t.alive && this.offset(t) < hitbox).sort((a, b) => this.offset(a) - this.offset(b))[0];
    if (hit) {
      this.stats.hits++;
      this.stats.kills++;
      this.stats.ttkSum += this.time - hit.spawnedAt;
      hit.alive = false;
      hit.flash = 1;
      hit.respawnAt = this.time + 450;
      const awards: Award[] = [{ points: KILL_POINTS, label: "Kill" }];
      if (this.time - hit.spawnedAt < QUICK_KILL_MS) awards.push({ points: 50, label: "Quick Kill" });
      this.chain = this.time - this.lastKill < CHAIN_MS ? this.chain + 1 : 1;
      this.lastKill = this.time;
      if (this.chain >= 2) awards.push({ points: 50 * (this.chain - 1), label: CHAIN_NAMES[this.chain] ?? "Multi Kill" });
      this.stats.score += awards.reduce((s, a) => s + a.points, 0);
      this.onHit?.(hit, awards);
    }
    return { hit: hit ?? null };
  }
}
