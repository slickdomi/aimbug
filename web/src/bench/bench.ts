// Benchmark and analysis tools, loaded only with ?bench=1 (web/scripts/smoke.mjs adds it), so they stay out
// of the game bundle: a window.aimbug handle for the headless smoke test, recorded samples, song vs. aim
// statistics, activity snapshots and surveys, and knobs that only make sense when comparing runs.

import { GAME, MODEL } from "../config";
import type { BrainInfo } from "../data/types";
import { setRandom, type Game } from "../game/game";
import type { Brain, ProbeGroup } from "../sim/brain";

const DEG = Math.PI / 180;

interface Rates {
  get(name: string, side: number): number;
}

/** What the tools need from the running game. */
export interface Session {
  game: Game;
  rates: Rates;
  data: BrainInfo;
  probes: ProbeGroup[];
  gpuBrain: Brain | null; // null on the CPU fallback: no activity snapshots or surveys
  seizures: () => number;
  songThreshold: () => number;
  paused: () => boolean;
}

const num = (query: URLSearchParams, key: string, fallback: number) => (query.has(key) ? parseFloat(query.get(key)!) : fallback);
const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

/** mulberry32: a small seeded random source, so runs can be compared on the same target sequence. */
function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Before loading: model and readout constants without a slider, ?adapt= ?rateTau= ?pitchTau= ?spring=
 * ?pitchRef= and ?sightRadius= (the sight probe group is built while loading).
 */
export function configure(query: URLSearchParams) {
  MODEL.adapt = num(query, "adapt", MODEL.adapt);
  GAME.rateTau = num(query, "rateTau", GAME.rateTau);
  GAME.pitchTau = num(query, "pitchTau", GAME.pitchTau);
  GAME.neckSpring = num(query, "spring", GAME.neckSpring);
  GAME.pitchRef = num(query, "pitchRef", GAME.pitchRef);
  GAME.sightRadiusDeg = num(query, "sightRadius", GAME.sightRadiusDeg);
}

/**
 * Once the game is set up. ?seed=N fixes the target sequence; ?mode=calib holds each female still and moves
 * her every 1.2 s to within ?calibSpread= degrees (default 40) of the crosshair; ?record=1 keeps a sample per
 * brain readback. Returns the hooks the game loop calls.
 */
export function attach(s: Session, query: URLSearchParams) {
  const { game, rates, data, gpuBrain } = s;
  const random = query.has("seed") ? seeded(num(query, "seed", 1)) : Math.random;
  setRandom(random);
  if (query.get("mode") === "calib") {
    const spread = num(query, "calibSpread", 40) * DEG;
    game.mode = "static";
    game.lifetimeMs = 1200;
    game.place = (t) => {
      t.az = wrap(game.heading + (random() * 2 - 1) * spread);
      t.el = (random() * 2 - 1) * Math.min(game.maxEl(), spread);
    };
  }
  game.reset();
  const sight = s.probes.find((g) => g.name === "sight");
  console.log(`sight: ${sight?.count ?? 0} LC10 cells within ${GAME.sightRadiusDeg}° of straight ahead`);

  // Most active cell types (spike trace ~ Hz * actTau), for chasing runaway activity.
  const topActive = gpuBrain
    ? async (k = 20) => {
        const act = await gpuBrain.readActivity();
        const byType = new Map<string, { sum: number; n: number; sc: string }>();
        for (let i = data.ng; i < data.n; i++) {
          const key = data.types[data.typeId[i]] || "(untyped)";
          const e = byType.get(key) ?? { sum: 0, n: 0, sc: data.meta.superclasses[data.superclass[i]] };
          e.sum += act[i];
          e.n++;
          byType.set(key, e);
        }
        const hz = 1000 / MODEL.actTau;
        return [...byType.entries()]
          .sort((a, b) => b[1].sum - a[1].sum)
          .slice(0, k)
          .map(([t, e]) => `${t}(${e.sc}) n=${e.n} total ${(e.sum * hz).toFixed(0)} Hz, ${((e.sum * hz) / e.n).toFixed(0)} Hz/cell`);
      }
    : undefined;

  // Survey (smoke.mjs SURVEY=1): mean spike trace per spiking cell type and side, plus every descending
  // neuron and LC10/LC4/LPLC2/LC9/LC11 cell on its own, for finding cells tuned to where the target is.
  const surveyCols: { label: string; idx: number[] }[] = [];
  if (gpuBrain) {
    const byTypeSide = new Map<string, number[]>();
    const dnClass = data.meta.superclasses.indexOf("descending_neuron");
    const single = /^(LC10|LC4$|LPLC2$|LC9$|LC11$)/;
    for (let i = data.ng; i < data.n; i++) {
      const t = data.types[data.typeId[i]] || "(untyped)";
      const key = `${t}:${"?LR"[data.side[i]]}`;
      let cells = byTypeSide.get(key);
      if (!cells) byTypeSide.set(key, (cells = []));
      cells.push(i);
      if (data.superclass[i] === dnClass || single.test(t)) surveyCols.push({ label: `#${i} ${key}`, idx: [i] });
    }
    for (const [key, idx] of byTypeSide) surveyCols.push({ label: key, idx });
  }
  const surveySample = gpuBrain
    ? async () => {
        const act = await gpuBrain.readActivity();
        return surveyCols.map((c) => {
          let sum = 0;
          for (const i of c.idx) sum += act[i];
          return Math.round((sum / c.idx.length) * 1000) / 1000;
        });
      }
    : undefined;

  // Song vs. aim statistics: pIP10 rate binned by crosshair offset.
  const songBins = [10, 20, 40, 80, 180].map((maxDeg) => ({ maxDeg, samples: 0, rateSum: 0, overThreshold: 0 }));
  // One row per readback; columns are listed in pipeline/analyze_samples.py.
  const samples: number[][] | null = query.has("record") ? [] : null;

  (window as unknown as { aimbug: object }).aimbug = {
    game,
    rates,
    MODEL,
    GAME,
    backend: gpuBrain ? "webgpu" : "cpu",
    seizures: s.seizures,
    songBins,
    samples,
    topActive,
    surveyLabels: () => surveyCols.map((c) => c.label),
    surveySample,
  };

  return {
    /** After every brain readback (rates already updated). */
    onReadback() {
      const alive = game.targets.filter((t) => t.alive);
      if (!alive.length || s.paused()) return;
      const songRate = rates.get("pIP10", 1) + rates.get("pIP10", 2);
      const offDeg = Math.min(...alive.map((t) => game.offset(t))) / DEG;
      const bin = songBins.find((b) => offDeg < b.maxDeg)!;
      bin.samples++;
      bin.rateSum += songRate;
      bin.overThreshold += +(songRate >= s.songThreshold());
      if (samples && samples.length < 50000) {
        const t = alive[0];
        samples.push([
          game.relAz(t) / DEG, game.relEl(t) / DEG, game.time - t.spawnedAt,
          rates.get("DNp53", 1), rates.get("DNp53", 2), rates.get("DNp01", 1), rates.get("DNp01", 2),
          rates.get("LC4", 1) + rates.get("LPLC2", 1), rates.get("LC4", 2) + rates.get("LPLC2", 2),
          rates.get("DNa02", 1), rates.get("DNa02", 2), songRate,
          rates.get("DNa01", 1), rates.get("DNa01", 2), rates.get("LC10a", 1), rates.get("LC10a", 2),
          rates.get("AOTU019", 1), rates.get("AOTU019", 2), game.time, rates.get("sight", 0),
        ]);
      }
    },
    /** A seizure was detected. Call before the brain reset: the activity copy is submitted first. */
    onSeizure(n: number) {
      topActive?.(8).then((top) => console.warn(`seizure ${n} cells: ${top.join(" | ")}`));
    },
  };
}
