import commonSrc from "../shaders/common.wgsl?raw";
import coupleSrc from "../shaders/couple.wgsl?raw";
import eyeSrc from "../shaders/eye.wgsl?raw";
import gradedSrc from "../shaders/graded.wgsl?raw";
import neuronSrc from "../shaders/neuron.wgsl?raw";
import probeSrc from "../shaders/probe.wgsl?raw";
import sceneSrc from "../shaders/scene.wgsl?raw";
import scatterSrc from "../shaders/scatter.wgsl?raw";
import { GAME, MODEL } from "../config";
import type { BrainData } from "../data/types";
import type { SpriteAtlas } from "../game/sprites";

type Slot = "u" | "ud" | "r" | "rw" | "tex" | "samp";

export interface ProbeGroup {
  name: string;
  side: number;
  start: number;
  count: number;
}

export interface Readback {
  counts: Uint32Array; // cumulative spike counts per probe neuron
  brainTime: number; // ms of simulated time when the counts were copied
  spikes: number; // spikes emitted during the frame that produced this readback
  frameMs: number; // brain ms simulated in that frame
}

const align4 = (x: number) => Math.max(8, Math.ceil(x / 4) * 4);

export class Brain {
  readonly n: number;
  readonly ng: number;
  readonly neurons: GPUBuffer;
  readonly gradedA: GPUBuffer;
  readonly gradedB: GPUBuffer;
  readonly units: GPUBuffer;
  /** Which of gradedA/gradedB holds the latest graded activity. */
  gradedCurrent: 0 | 1 = 0;
  brainTime = 0;
  onReadback: ((r: Readback) => void) | null = null;

  private device: GPUDevice;
  private params: GPUBuffer;
  private paramsData = new ArrayBuffer(80);
  private stepBuf: GPUBuffer;
  private stepAlign: number;
  private baseDrive: GPUBuffer;
  private spikeCount: GPUBuffer;
  private probeOut: GPUBuffer;
  private probeCount: number;
  private stages: GPUBuffer[] = [];
  private pipes: Record<"neuron" | "scatter" | "graded" | "couple" | "eyeMean" | "eyeContrast" | "probe", GPUComputePipeline>;
  private groups: {
    neuron: GPUBindGroup;
    scatter: GPUBindGroup;
    graded: [GPUBindGroup, GPUBindGroup];
    couple: [GPUBindGroup, GPUBindGroup];
    eyeMean: GPUBindGroup;
    eyeContrast: GPUBindGroup;
    probe: GPUBindGroup;
  };
  private ringSlots: number;
  private delaySteps: number;
  private t = 0;
  private nextGraded = 0;
  private visCount: number;
  private pC1: number[];
  private pendingStage: (() => Promise<void>) | null = null;
  private ring: GPUBuffer;

  constructor(
    device: GPUDevice,
    d: BrainData,
    sceneBuffer: GPUBuffer,
    sprites: SpriteAtlas,
    readonly probes: ProbeGroup[],
    probeIdx: Uint32Array,
  ) {
    this.device = device;
    this.n = d.n;
    this.ng = d.ng;
    this.visCount = d.visCount;
    const S = GPUBufferUsage.STORAGE;
    const CD = GPUBufferUsage.COPY_DST;
    const CS = GPUBufferUsage.COPY_SRC;

    const make = (size: number, usage: number, data?: ArrayBufferView | ArrayBuffer, label?: string) => {
      const b = device.createBuffer({ size: align4(size), usage, mappedAtCreation: !!data, label });
      if (data) {
        const src = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        new Uint8Array(b.getMappedRange()).set(src);
        b.unmap();
      }
      return b;
    };

    this.delaySteps = Math.max(1, Math.round(MODEL.tDelay / MODEL.dt));
    this.ringSlots = this.delaySteps + 1;
    this.stepAlign = device.limits.minUniformBufferOffsetAlignment;

    this.params = make(80, GPUBufferUsage.UNIFORM | CD, undefined, "params");
    this.stepBuf = make(this.stepAlign * GAME.maxStepsPerFrame, GPUBufferUsage.UNIFORM | CD, undefined, "steps");
    this.neurons = make(24 * d.n, S | CD | CS, undefined, "neurons");
    this.baseDrive = make(4 * d.n, S | CD, undefined, "baseDrive");
    const gDrive = make(4 * d.n, S, undefined, "gDrive");
    const ring = make(4 * this.ringSlots * d.n, S | CD, undefined, "ring");
    this.ring = ring;
    const spikes = make(4 * GAME.maxSpikesPerStep, S, undefined, "spikes");
    this.spikeCount = make(4 * GAME.maxStepsPerFrame, S | CD | CS, undefined, "spikeCount");
    const spikeOffsets = make(d.spikeOffsets.byteLength, S, d.spikeOffsets, "spikeOffsets");
    const spikeEdges = make(d.spikeEdges.byteLength, S, d.spikeEdges, "spikeEdges");
    const gOffsets = make(d.gradedOffsets.byteLength, S, d.gradedOffsets, "gradedOffsets");
    const gEntries = make(d.gradedEntries.byteLength, S, d.gradedEntries, "gradedEntries");
    const iOffsets = make(d.ifaceOffsets.byteLength, S, d.ifaceOffsets, "ifaceOffsets");
    const iEntries = make(d.ifaceEntries.byteLength, S, d.ifaceEntries, "ifaceEntries");
    this.gradedA = make(4 * d.ng, S, undefined, "gradedA");
    this.gradedB = make(4 * d.ng, S, undefined, "gradedB");
    const ext = make(4 * d.ng, S, undefined, "ext");
    const modeData = new Uint32Array(d.ng);
    const unitU = new Uint32Array(d.visUnits);
    for (let u = 0; u < d.visCount; u++) modeData[unitU[8 * u + 4]] = unitU[8 * u + 6];
    const mode = make(4 * d.ng, S, modeData, "mode");
    this.units = make(d.visUnits.byteLength, S, d.visUnits, "visUnits");
    const means = make(16, S, new Float32Array([1, 1, 1, 1]), "eyeMeans");
    const eyeData = new ArrayBuffer(16);
    new Uint32Array(eyeData)[0] = d.visCount;
    new Float32Array(eyeData)[1] = (2.4 * Math.PI) / 180; // half acceptance angle
    const eyeParams = make(16, GPUBufferUsage.UNIFORM, eyeData, "eyeParams");
    this.probeCount = probeIdx.length;
    const probeIdxBuf = make(probeIdx.byteLength, S, probeIdx, "probeIdx");
    this.probeOut = make(4 * probeIdx.length, S | CS, undefined, "probeOut");
    for (let k = 0; k < 3; k++) {
      this.stages.push(device.createBuffer({ size: align4(4 * (this.probeCount + GAME.maxStepsPerFrame)), usage: GPUBufferUsage.MAP_READ | CD }));
    }

    const layout = (slots: Slot[]) =>
      device.createBindGroupLayout({
        entries: slots.map((s, i): GPUBindGroupLayoutEntry => {
          if (s === "tex") return { binding: i, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } };
          if (s === "samp") return { binding: i, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } };
          return {
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: s === "u" || s === "ud" ? "uniform" : s === "r" ? "read-only-storage" : "storage",
              hasDynamicOffset: s === "ud",
            },
          };
        }),
      });
    const pipeline = (code: string, slots: Slot[], entryPoint = "main") => {
      const bgl = layout(slots);
      const p = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module: device.createShaderModule({ code }), entryPoint },
      });
      return { p, bgl };
    };
    const group = (bgl: GPUBindGroupLayout, res: (GPUBuffer | [GPUBuffer, number] | GPUTextureView | GPUSampler)[]) =>
      device.createBindGroup({
        layout: bgl,
        entries: res.map((b, i) =>
          Array.isArray(b)
            ? { binding: i, resource: { buffer: b[0], size: b[1] } }
            : b instanceof GPUBuffer
              ? { binding: i, resource: { buffer: b } }
              : { binding: i, resource: b },
        ),
      });

    const neuron = pipeline(commonSrc + neuronSrc, ["u", "ud", "rw", "r", "r", "rw", "rw", "rw"]);
    const scatter = pipeline(commonSrc + scatterSrc, ["u", "ud", "r", "rw", "r", "r", "rw"]);
    const graded = pipeline(commonSrc + gradedSrc, ["u", "r", "r", "r", "rw", "r", "r"]);
    const couple = pipeline(commonSrc + coupleSrc, ["u", "r", "r", "r", "rw"]);
    const eyeMean = pipeline(sceneSrc + eyeSrc, ["u", "u", "r", "rw", "rw", "tex", "samp"], "meanPass");
    const eyeContrast = pipeline(sceneSrc + eyeSrc, ["u", "u", "r", "rw", "rw", "tex", "samp"], "contrastPass");
    const probe = pipeline(commonSrc + probeSrc, ["r", "r", "rw"]);
    this.pipes = {
      neuron: neuron.p, scatter: scatter.p, graded: graded.p, couple: couple.p,
      eyeMean: eyeMean.p, eyeContrast: eyeContrast.p, probe: probe.p,
    };
    const step: [GPUBuffer, number] = [this.stepBuf, 16];
    this.groups = {
      neuron: group(neuron.bgl, [this.params, step, this.neurons, this.baseDrive, gDrive, ring, spikes, this.spikeCount]),
      scatter: group(scatter.bgl, [this.params, step, spikes, this.spikeCount, spikeOffsets, spikeEdges, ring]),
      graded: [
        group(graded.bgl, [this.params, gOffsets, gEntries, this.gradedA, this.gradedB, ext, mode]),
        group(graded.bgl, [this.params, gOffsets, gEntries, this.gradedB, this.gradedA, ext, mode]),
      ],
      couple: [
        group(couple.bgl, [this.params, iOffsets, iEntries, this.gradedA, gDrive]),
        group(couple.bgl, [this.params, iOffsets, iEntries, this.gradedB, gDrive]),
      ],
      eyeMean: group(eyeMean.bgl, [sceneBuffer, eyeParams, this.units, ext, means, sprites.texture.createView(), sprites.sampler]),
      eyeContrast: group(eyeContrast.bgl, [sceneBuffer, eyeParams, this.units, ext, means, sprites.texture.createView(), sprites.sampler]),
      probe: group(probe.bgl, [probeIdxBuf, this.neurons, this.probeOut]),
    };

    this.pC1 = [];
    d.typeId.forEach((t, i) => {
      if (i >= d.ng && d.types[t].startsWith("pC1_")) this.pC1.push(i);
    });
    this.setArousal(MODEL.arousal);
    this.writeParams();
  }

  writeParams() {
    const m = MODEL;
    const u = new Uint32Array(this.paramsData);
    const f = new Float32Array(this.paramsData);
    u[0] = this.n;
    u[1] = this.ng;
    u[2] = this.ringSlots;
    u[3] = GAME.maxSpikesPerStep;
    const em = Math.exp(-m.dt / m.tauM);
    const es = Math.exp(-m.dt / m.tauS);
    f[4] = em;
    f[5] = es;
    f[6] = (m.tauS / (m.tauS - m.tauM)) * (es - em);
    f[7] = Math.exp(-m.dt / m.tauAdapt);
    f[8] = m.vThreshold;
    f[9] = m.vReset;
    f[10] = m.wSyn;
    f[11] = m.adapt;
    f[12] = m.tRefractory;
    f[13] = m.dt;
    f[14] = Math.exp(-m.dt / m.actTau);
    f[15] = m.coupling;
    f[16] = m.gradedGain;
    f[17] = Math.exp(-m.gradedDt / m.gradedTau);
    f[18] = m.gradedMin;
    f[19] = m.gradedMax;
    this.device.queue.writeBuffer(this.params, 0, this.paramsData);
  }

  setArousal(mV: number) {
    const drive = new Float32Array(this.n);
    for (const i of this.pC1) drive[i] = mV;
    this.device.queue.writeBuffer(this.baseDrive, 0, drive);
  }

  /** Clears all neural state (voltages, synaptic input, graded activity). */
  /** Clears spiking state (voltages, synaptic input, adaptation); optionally the graded optic lobe too. */
  reset(includeGraded = true) {
    const q = this.device.queue;
    q.writeBuffer(this.neurons, 0, new Uint8Array(24 * this.n));
    q.writeBuffer(this.ring, 0, new Uint8Array(4 * this.ringSlots * this.n));
    if (includeGraded) {
      q.writeBuffer(this.gradedA, 0, new Float32Array(this.ng));
      q.writeBuffer(this.gradedB, 0, new Float32Array(this.ng));
    }
  }

  /** Simulates `brainMs` of brain time; the eye samples the scene once at the start. */
  encode(encoder: GPUCommandEncoder, brainMs: number): number {
    const m = MODEL;
    const steps = Math.min(GAME.maxStepsPerFrame, Math.max(0, Math.round(brainMs / m.dt)));
    const stepData = new ArrayBuffer(this.stepAlign * Math.max(1, steps));
    const sv = new Uint32Array(stepData);
    for (let s = 0; s < steps; s++) {
      const o = (this.stepAlign / 4) * s;
      const t = this.t + s;
      sv[o] = s;
      sv[o + 1] = t % this.ringSlots;
      sv[o + 2] = (t + this.delaySteps) % this.ringSlots;
    }
    const q = this.device.queue;
    if (steps > 0) q.writeBuffer(this.stepBuf, 0, stepData);
    q.writeBuffer(this.spikeCount, 0, new Uint32Array(GAME.maxStepsPerFrame));

    const pass = encoder.beginComputePass({ label: "brain" });
    pass.setPipeline(this.pipes.eyeMean);
    pass.setBindGroup(0, this.groups.eyeMean);
    pass.dispatchWorkgroups(2);
    pass.setPipeline(this.pipes.eyeContrast);
    pass.setBindGroup(0, this.groups.eyeContrast);
    pass.dispatchWorkgroups(Math.ceil(this.visCount / 128));

    const ns = this.n - this.ng;
    for (let s = 0; s < steps; s++) {
      if (this.brainTime >= this.nextGraded) {
        pass.setPipeline(this.pipes.graded);
        pass.setBindGroup(0, this.groups.graded[this.gradedCurrent]);
        pass.dispatchWorkgroups(Math.ceil(this.ng / 256));
        this.gradedCurrent = this.gradedCurrent === 0 ? 1 : 0;
        pass.setPipeline(this.pipes.couple);
        pass.setBindGroup(0, this.groups.couple[this.gradedCurrent]);
        pass.dispatchWorkgroups(Math.ceil(ns / 256));
        this.nextGraded += m.gradedDt;
      }
      const off = [this.stepAlign * s];
      pass.setPipeline(this.pipes.neuron);
      pass.setBindGroup(0, this.groups.neuron, off);
      pass.dispatchWorkgroups(Math.ceil(ns / 256));
      pass.setPipeline(this.pipes.scatter);
      pass.setBindGroup(0, this.groups.scatter, off);
      pass.dispatchWorkgroups(Math.ceil(GAME.maxSpikesPerStep / 64));
      this.brainTime += m.dt;
    }
    this.t += steps;

    pass.setPipeline(this.pipes.probe);
    pass.setBindGroup(0, this.groups.probe);
    pass.dispatchWorkgroups(Math.ceil(this.probeCount / 64));
    pass.end();

    const stage = this.stages.pop();
    if (stage) {
      encoder.copyBufferToBuffer(this.probeOut, 0, stage, 0, 4 * this.probeCount);
      encoder.copyBufferToBuffer(this.spikeCount, 0, stage, 4 * this.probeCount, 4 * GAME.maxStepsPerFrame);
      const brainTime = this.brainTime;
      const frameMs = steps * m.dt;
      // Map after the caller submits the encoder.
      this.pendingStage = () =>
        stage.mapAsync(GPUMapMode.READ).then(() => {
          const all = new Uint32Array(stage.getMappedRange().slice(0));
          stage.unmap();
          this.stages.push(stage);
          let spikes = 0;
          for (let s = 0; s < steps; s++) spikes += all[this.probeCount + s];
          this.onReadback?.({ counts: all.subarray(0, this.probeCount), brainTime, spikes, frameMs });
        });
    }
    return steps;
  }

  /** Call right after queue.submit() of the encoder passed to encode(). */
  afterSubmit() {
    const p = this.pendingStage;
    this.pendingStage = null;
    p?.();
  }

  /** Debug: copies every neuron's spike trace (act) back to the CPU. */
  async readActivity(): Promise<Float32Array> {
    const stage = this.device.createBuffer({ size: 24 * this.n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.neurons, 0, stage, 0, 24 * this.n);
    this.device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(stage.getMappedRange().slice(0));
    stage.destroy();
    const act = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) act[i] = f[6 * i + 4];
    return act;
  }

  currentGraded(): GPUBuffer {
    return this.gradedCurrent === 0 ? this.gradedA : this.gradedB;
  }
}
