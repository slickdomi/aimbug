import { DATA_URL, MODEL } from "../config";
import type { BrainData, WorkerIn, WorkerOut } from "./types";

export function loadBrainData(onProgress: (label: string, frac: number) => void): Promise<BrainData> {
  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  return new Promise((resolve, reject) => {
    worker.onmessage = (ev: MessageEvent<WorkerOut>) => {
      const msg = ev.data;
      // done can exceed total when the server inflates gzip on the wire
      if (msg.kind === "progress") onProgress(msg.label, msg.total > 0 ? Math.min(1, msg.done / msg.total) : 0);
      else if (msg.kind === "done") {
        worker.terminate();
        resolve(msg.data);
      } else {
        worker.terminate();
        reject(new Error(msg.message));
      }
    };
    worker.onerror = (e) => reject(new Error(e.message));
    const req: WorkerIn = { url: new URL(DATA_URL, location.href).href, prune: MODEL.prune, laminaWeight: MODEL.laminaWeight, wSyn: MODEL.wSyn, tauS: MODEL.tauS };
    worker.postMessage(req);
  });
}
