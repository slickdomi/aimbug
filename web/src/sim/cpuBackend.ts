import type { ModelParams } from "../config";
import type { BrainInfo, WorkerIn } from "../data/types";
import type { Readback } from "./brain";
import type { SceneState } from "./cpuSim";
import type { CpuIn, CpuOut } from "./cpuWorker";

/** Main-thread handle for the CPU fallback brain running in a worker. */
export class CpuBackend {
  onReadback: ((r: Readback) => void) | null = null;
  onViz: ((act: Float32Array, graded: Float32Array) => void) | null = null;
  busy = false;
  lastWallMs = 0;
  private worker = new Worker(new URL("./cpuWorker.ts", import.meta.url), { type: "module" });

  constructor(private model: ModelParams) {}

  private send(msg: CpuIn, transfer: Transferable[] = []) {
    this.worker.postMessage(msg, transfer);
  }

  init(req: WorkerIn, spriteUrl: string, onProgress: (label: string, frac: number) => void): Promise<BrainInfo> {
    return new Promise((resolve, reject) => {
      this.worker.onerror = (e) => reject(new Error(e.message));
      this.worker.onmessage = (ev: MessageEvent<CpuOut>) => {
        const msg = ev.data;
        if (msg.kind === "progress") onProgress(msg.label, msg.total > 0 ? Math.min(1, msg.done / msg.total) : 0);
        else if (msg.kind === "ready") resolve(msg.info);
        else if (msg.kind === "error") reject(new Error(msg.message));
        else if (msg.kind === "result") {
          this.busy = false;
          this.lastWallMs = msg.wallMs;
          this.onReadback?.({ counts: msg.counts, brainTime: msg.brainTime, spikes: msg.spikes, frameMs: msg.frameMs });
          if (msg.act && msg.graded) this.onViz?.(msg.act, msg.graded);
        }
      };
      this.send({ kind: "init", req, model: { ...this.model }, spriteUrl });
    });
  }

  setProbes(idx: Uint32Array) {
    this.send({ kind: "probes", idx });
  }

  writeParams() {
    this.send({ kind: "params", model: { ...this.model } });
  }

  setArousal(mV: number) {
    this.model.arousal = mV;
    this.writeParams();
  }

  reset(includeGraded = true) {
    this.send({ kind: "reset", includeGraded });
  }

  step(brainMs: number, scene: SceneState, wantViz: boolean) {
    this.busy = true;
    this.send({ kind: "step", brainMs, scene, wantViz });
  }
}
