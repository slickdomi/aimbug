// CPU fallback: loads the connectome and runs the brain simulation in this worker.

import type { ModelParams } from "../config";
import { loadBrainData } from "../data/build";
import type { BrainInfo, WorkerIn } from "../data/types";
import { CpuSim, type EyeAtlas, type SceneState } from "./cpuSim";

export type CpuIn =
  | { kind: "init"; req: WorkerIn; model: ModelParams; spriteUrl: string }
  | { kind: "probes"; idx: Uint32Array }
  | { kind: "params"; model: ModelParams }
  | { kind: "reset"; includeGraded: boolean }
  | { kind: "step"; brainMs: number; scene: SceneState; wantViz: boolean };

export type CpuOut =
  | { kind: "progress"; label: string; done: number; total: number }
  | { kind: "ready"; info: BrainInfo }
  | { kind: "result"; counts: Uint32Array; brainTime: number; spikes: number; frameMs: number; wallMs: number; act?: Float32Array; graded?: Float32Array }
  | { kind: "error"; message: string };

const post = (msg: CpuOut, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);

async function loadEyeAtlas(spriteUrl: string): Promise<EyeAtlas> {
  const meta: { cells: { u0: number; v0: number; u1: number; v1: number }[] } = await (await fetch(`${spriteUrl}/females.json`)).json();
  const bitmap = await createImageBitmap(await (await fetch(`${spriteUrl}/females.png`)).blob(), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
  // a quarter of the resolution: still finer than the fly's ~4.8 degree ommatidia at game distances
  const width = Math.max(4, bitmap.width >> 2);
  const height = Math.max(4, bitmap.height >> 2);
  const ctx = new OffscreenCanvas(width, height).getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, width, height);
  return {
    width,
    height,
    rgba: ctx.getImageData(0, 0, width, height).data,
    rects: meta.cells.map((c) => ({ cu: (c.u0 + c.u1) / 2, cv: (c.v0 + c.v1) / 2, hu: (c.u1 - c.u0) / 2, hv: (c.v1 - c.v0) / 2 })),
  };
}

let sim: CpuSim | null = null;

self.onmessage = async (ev: MessageEvent<CpuIn>) => {
  const msg = ev.data;
  try {
    if (msg.kind === "init") {
      const [data, atlas] = await Promise.all([
        loadBrainData(msg.req, (label, done, total) => post({ kind: "progress", label, done, total })),
        loadEyeAtlas(msg.spriteUrl),
      ]);
      sim = new CpuSim(data, msg.model, atlas);
      const { meta, types, n, ng, typeId, side, superclass, pos, rf, visUnits, visCount, stats } = data;
      post({ kind: "ready", info: { meta, types, n, ng, typeId: typeId.slice(), side: side.slice(), superclass: superclass.slice(), pos: pos.slice(), rf: rf.slice(), visUnits: visUnits.slice(0), visCount, stats } });
      return;
    }
    if (!sim) return;
    if (msg.kind === "probes") sim.setProbes(msg.idx);
    else if (msg.kind === "params") sim.setParams(msg.model);
    else if (msg.kind === "reset") sim.reset(msg.includeGraded);
    else if (msg.kind === "step") {
      const t0 = performance.now();
      const r = sim.step(msg.brainMs, msg.scene);
      const out: CpuOut = { kind: "result", ...r, brainTime: sim.brainTime, wallMs: performance.now() - t0 };
      const transfer: Transferable[] = [r.counts.buffer];
      if (msg.wantViz) {
        out.act = sim.act.slice();
        out.graded = sim.graded().slice();
        transfer.push(out.act.buffer, out.graded.buffer);
      }
      post(out, transfer);
    }
  } catch (err) {
    post({ kind: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
