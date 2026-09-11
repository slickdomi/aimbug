export interface Shard {
  file: string;
  rowStart: number;
  rowEnd: number;
  edges: number;
  bytes: number;
}

export interface Meta {
  dataset: string;
  source: string;
  license: string;
  neurons: number;
  edges: number;
  synapses: number;
  superclasses: string[];
  transmitters: string[];
  shards: Shard[];
  graded: { superclasses: string[]; count: number };
  eye: { count: number };
}

/** Everything the GPU brain needs, produced by the loader worker. */
export interface BrainData {
  meta: Meta;
  types: string[];
  n: number;
  ng: number; // neurons [0, ng) are graded optic-lobe units, [ng, n) spike
  typeId: Uint16Array;
  side: Uint8Array; // 0 unknown, 1 left, 2 right
  superclass: Uint8Array;
  pos: Float32Array; // xyz micrometres, NaN if unknown
  // spiking -> spiking CSR by presynaptic neuron (global indices, n+1 offsets);
  // packed = target (18 bits) | signed synapse count (14 bits) << 18
  spikeOffsets: Uint32Array;
  spikeEdges: Uint32Array;
  // graded -> graded CSR by postsynaptic graded neuron: interleaved {u32 col, f32 signed input fraction}
  gradedOffsets: Uint32Array;
  gradedEntries: ArrayBuffer;
  // graded -> spiking CSR by postsynaptic spiking neuron (row r = neuron ng + r):
  // interleaved {u32 graded col, f32 mV per unit graded activity (before coupling scale)}
  ifaceOffsets: Uint32Array;
  ifaceEntries: ArrayBuffer;
  // visual units: interleaved {dx, dy, dz, weight: f32; graded index, side, mode, pad: u32}
  // mode 1 = photoreceptor (activity clamped to contrast), 2 = lamina (adds weight * contrast)
  visUnits: ArrayBuffer;
  visCount: number;
  stats: { spikeEdges: number; gradedEdges: number; ifaceEdges: number };
}

/** The parts of BrainData the main thread needs when the brain itself runs in a CPU worker. */
export type BrainInfo = Pick<BrainData, "meta" | "types" | "n" | "ng" | "typeId" | "side" | "superclass" | "pos" | "visUnits" | "visCount" | "stats">;

export type WorkerIn ={ url: string; prune: number; laminaWeight: number; wSyn: number; tauS: number };
export type WorkerOut =
  | { kind: "progress"; label: string; done: number; total: number }
  | { kind: "done"; data: BrainData }
  | { kind: "error"; message: string };
