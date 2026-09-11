// Downloads the packed MaleCNS connectome and splits it into the three sparse
// matrices the hybrid model runs on (see pipeline/sim_hybrid.py for the reference).
// Shared by the WebGPU loader worker and the CPU fallback worker.

import type { BrainData, Meta, WorkerIn } from "./types";

export type Progress = (label: string, done: number, total: number) => void;

async function fetchBytes(url: string, label: string, expected: number, gz: boolean, onProgress: Progress, base = 0): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
  const chunks: Uint8Array[] = [];
  let got = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.byteLength;
    onProgress(label, base + got, expected);
  }
  const blob = new Blob(chunks as BlobPart[]);
  // Some servers (e.g. Vite's dev server) send *.gz with Content-Encoding: gzip, so the
  // browser has already inflated the body. Only decompress if the gzip magic is still there.
  const head = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
  if (!gz || head[0] !== 0x1f || head[1] !== 0x8b) return new Uint8Array(await blob.arrayBuffer());
  const stream = blob.stream().pipeThrough(new DecompressionStream("gzip") as unknown as ReadableWritablePair<Uint8Array, Uint8Array>);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** LEB128 varint reader over a byte array. */
class Varints {
  pos: number;
  constructor(private buf: Uint8Array, start: number) {
    this.pos = start;
  }
  next(): number {
    let v = 0;
    let shift = 0;
    for (;;) {
      const b = this.buf[this.pos++];
      v += (b & 0x7f) * 2 ** shift;
      if (b < 0x80) return v;
      shift += 7;
    }
  }
}

export async function loadBrainData(req: WorkerIn, onProgress: Progress): Promise<BrainData> {
  const meta: Meta = await (await fetch(`${req.url}/meta.json`)).json();
  const types: string[] = await (await fetch(`${req.url}/types.json`)).json();
  const n = meta.neurons;
  const ng = meta.graded.count;
  const ne = meta.eye.count;

  const nb = await fetchBytes(`${req.url}/neurons.bin.gz`, "neurons", 0, true, onProgress);
  const nbuf = nb.buffer;
  const typeId = new Uint16Array(nbuf.slice(8 * n, 10 * n));
  const superclass = new Uint8Array(nbuf.slice(10 * n, 11 * n));
  const side = new Uint8Array(nbuf.slice(11 * n, 12 * n));
  const sign = new Int8Array(nbuf.slice(12 * n, 13 * n));
  const pos = new Float32Array(nbuf.slice(16 * n, 28 * n));
  const col = new Float32Array(nbuf.slice(40 * n, 48 * n));
  const nt = new Uint8Array(nbuf.slice(48 * n, 49 * n));

  // Decode all shards into (offsets by pre, post, count).
  const offsets = new Uint32Array(n + 1);
  const postIdx = new Uint32Array(meta.edges);
  const count = new Uint16Array(meta.edges);
  const totalBytes = meta.shards.reduce((s, x) => s + x.bytes, 0);
  let downloaded = 0;
  let e = 0;
  for (const sh of meta.shards) {
    const buf = await fetchBytes(`${req.url}/${sh.file}`, "downloading synapses", totalBytes, true, onProgress, downloaded);
    downloaded += sh.bytes;
    const head = new Uint32Array(buf.buffer, buf.byteOffset, 5);
    const [rowStart, rowEnd, edges, deltaBytes] = head;
    const deg = new Varints(buf, 20);
    const degrees = new Uint32Array(rowEnd - rowStart);
    for (let r = 0; r < degrees.length; r++) degrees[r] = deg.next();
    const dlt = new Varints(buf, deg.pos);
    const wts = new Varints(buf, deg.pos + deltaBytes);
    for (let r = 0; r < degrees.length; r++) {
      offsets[rowStart + r] = e;
      let tgt = 0;
      for (let k = 0; k < degrees[r]; k++) {
        tgt = k === 0 ? dlt.next() : tgt + dlt.next();
        postIdx[e] = tgt;
        count[e] = wts.next();
        e++;
      }
    }
    if (e !== offsets[rowStart] + edges) throw new Error(`shard ${sh.file}: edge count mismatch`);
  }
  offsets[n] = e;

  onProgress("wiring the optic lobe", 0, 1);
  const totalIn = new Float64Array(n);
  for (let i = 0; i < e; i++) totalIn[postIdx[i]] += count[i];

  const modulatory = new Uint8Array(meta.transmitters.length);
  meta.transmitters.forEach((t, i) => (modulatory[i] = +["dopamine", "serotonin", "octopamine"].includes(t)));
  const r16 = types.indexOf("R1-R6");
  const lamTypes = new Set(["L1", "L2", "L3"].map((t) => types.indexOf(t)));
  const isLam = new Uint8Array(n);
  for (let i = 0; i < ng; i++) if (lamTypes.has(typeId[i]) && !Number.isNaN(col[2 * i])) isLam[i] = 1;

  // Pass 1: count entries per matrix.
  const gCount = new Uint32Array(ng);
  const iCount = new Uint32Array(n - ng);
  let nSpike = 0;
  for (let pre = 0; pre < n; pre++) {
    const preGraded = pre < ng;
    for (let k = offsets[pre]; k < offsets[pre + 1]; k++) {
      const q = postIdx[k];
      if (preGraded) {
        if (q < ng) {
          if (typeId[pre] === r16 && isLam[q]) continue;
          if (count[k] / totalIn[q] < req.prune) continue;
          gCount[q]++;
        } else iCount[q - ng]++;
      } else if (q >= ng && !modulatory[nt[pre]]) nSpike++;
    }
  }
  const cumsum = (c: Uint32Array) => {
    const o = new Uint32Array(c.length + 1);
    for (let i = 0; i < c.length; i++) o[i + 1] = o[i] + c[i];
    return o;
  };
  const gradedOffsets = cumsum(gCount);
  const ifaceOffsets = cumsum(iCount);
  const gradedEntries = new ArrayBuffer(8 * gradedOffsets[ng]);
  const ifaceEntries = new ArrayBuffer(8 * ifaceOffsets[n - ng]);
  const gU = new Uint32Array(gradedEntries);
  const gF = new Float32Array(gradedEntries);
  const iU = new Uint32Array(ifaceEntries);
  const iF = new Float32Array(ifaceEntries);
  const spikeOffsets = new Uint32Array(n + 1);
  const spikeEdges = new Uint32Array(nSpike);
  const gFill = gradedOffsets.slice(0, ng);
  const iFill = ifaceOffsets.slice(0, n - ng);
  const mvPerSyn = (req.wSyn * req.tauS) / 1000;

  // Pass 2: fill.
  let s = 0;
  for (let pre = 0; pre < n; pre++) {
    spikeOffsets[pre] = s;
    const preGraded = pre < ng;
    const sg = sign[pre];
    for (let k = offsets[pre]; k < offsets[pre + 1]; k++) {
      const q = postIdx[k];
      const c = count[k];
      if (preGraded) {
        if (q < ng) {
          if (typeId[pre] === r16 && isLam[q]) continue;
          const frac = c / totalIn[q];
          if (frac < req.prune) continue;
          const j = gFill[q]++;
          gU[2 * j] = pre;
          gF[2 * j + 1] = frac * sg;
        } else {
          const j = iFill[q - ng]++;
          iU[2 * j] = pre;
          iF[2 * j + 1] = c * sg * mvPerSyn;
        }
      } else if (q >= ng && !modulatory[nt[pre]]) {
        spikeEdges[s++] = (q | (((c * sg) & 0x3fff) << 18)) >>> 0;
      }
    }
  }
  spikeOffsets[n] = s;

  // Visual units: photoreceptors clamp to contrast, lamina cells add -w * contrast.
  const units: number[] = [];
  for (let i = 0; i < ng; i++) if (i < ne || isLam[i]) units.push(i);
  const visUnits = new ArrayBuffer(32 * units.length);
  const vF = new Float32Array(visUnits);
  const vU = new Uint32Array(visUnits);
  units.forEach((i, u) => {
    const az = (col[2 * i] * Math.PI) / 180;
    const el = (col[2 * i + 1] * Math.PI) / 180;
    // Fly frame: +x right, +y up, -z forward.
    vF[8 * u] = Math.sin(az) * Math.cos(el);
    vF[8 * u + 1] = Math.sin(el);
    vF[8 * u + 2] = -Math.cos(az) * Math.cos(el);
    const eye = i < ne;
    vF[8 * u + 3] = eye ? 1 : -req.laminaWeight;
    vU[8 * u + 4] = i;
    vU[8 * u + 5] = side[i] || (col[2 * i] >= 0 ? 2 : 1); // which eye's mean luminance to normalise by
    vU[8 * u + 6] = eye ? 1 : 2;
  });

  return {
    meta,
    types,
    n,
    ng,
    typeId,
    side,
    superclass,
    pos,
    spikeOffsets,
    spikeEdges,
    gradedOffsets,
    gradedEntries,
    ifaceOffsets,
    ifaceEntries,
    visUnits,
    visCount: units.length,
    stats: { spikeEdges: s, gradedEdges: gradedOffsets[ng], ifaceEdges: ifaceOffsets[n - ng] },
  };
}
