// WebGPU path: decodes the connectome off the main thread and hands the matrices over.

import { loadBrainData } from "./build";
import type { WorkerIn, WorkerOut } from "./types";

const post = (msg: WorkerOut, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);

self.onmessage = async (ev: MessageEvent<WorkerIn>) => {
  try {
    const d = await loadBrainData(ev.data, (label, done, total) => post({ kind: "progress", label, done, total }));
    post({ kind: "done", data: d }, [
      d.typeId.buffer, d.side.buffer, d.superclass.buffer, d.pos.buffer, d.rf.buffer,
      d.spikeOffsets.buffer, d.spikeEdges.buffer, d.gradedOffsets.buffer, d.gradedEntries,
      d.ifaceOffsets.buffer, d.ifaceEntries, d.visUnits,
    ]);
  } catch (err) {
    post({ kind: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
