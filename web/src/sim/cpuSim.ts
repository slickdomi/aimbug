// CPU (plain JavaScript) port of the WebGPU brain, for browsers without WebGPU.
// Same model, same update order as web/src/shaders/{eye,graded,couple,neuron,scatter}.wgsl;
// runs inside a worker and is simply slower (the game runs on brain time, so it slows down too).

import type { ModelParams } from "../config";
import type { BrainData } from "../data/types";

export interface SceneTarget {
  az: number;
  el: number;
  visible: boolean;
  flash: number;
  flip: number;
  variant: number;
}

export interface SceneState {
  heading: number;
  pitch: number;
  tanR: number;
  targets: SceneTarget[];
}

export interface SpriteRect {
  cu: number;
  cv: number;
  hu: number;
  hv: number;
}

/** Low-resolution RGBA atlas used by the fly's eye (the acceptance angle blurs it anyway). */
export interface EyeAtlas {
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
  rects: SpriteRect[];
}

const TARGET_DIST = 9;
const FLOOR_Y = -1.6;

const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const fract = (x: number) => x - Math.floor(x);

export class CpuSim {
  brainTime = 0;
  private n: number;
  private ng: number;
  private v: Float32Array;
  private g: Float32Array;
  private refr: Float32Array;
  private theta: Float32Array;
  readonly act: Float32Array;
  private cnt: Uint32Array;
  private baseDrive: Float32Array;
  private gDrive: Float32Array;
  private ring: Int32Array;
  private ringSlots: number;
  private delaySteps: number;
  private aCur: Float32Array;
  private aNext: Float32Array;
  private ext: Float32Array;
  private mode: Uint8Array;
  private spikeList: Int32Array;
  private t = 0;
  private nextGraded = 0;
  private pC1: number[] = [];
  private probeIdx: Uint32Array<ArrayBufferLike> = new Uint32Array(0);
  // matrices (views over the same memory layout the GPU uses)
  private spikeOffsets: Uint32Array;
  private spikeEdges: Uint32Array;
  private gOff: Uint32Array;
  private gU: Uint32Array;
  private gF: Float32Array;
  private iOff: Uint32Array;
  private iU: Uint32Array;
  private iF: Float32Array;
  private unitsF: Float32Array;
  private unitsU: Uint32Array;
  private visCount: number;

  constructor(d: BrainData, private m: ModelParams, private atlas: EyeAtlas) {
    this.n = d.n;
    this.ng = d.ng;
    const n = d.n;
    this.v = new Float32Array(n);
    this.g = new Float32Array(n);
    this.refr = new Float32Array(n);
    this.theta = new Float32Array(n);
    this.act = new Float32Array(n);
    this.cnt = new Uint32Array(n);
    this.baseDrive = new Float32Array(n);
    this.gDrive = new Float32Array(n);
    this.delaySteps = Math.max(1, Math.round(m.tDelay / m.dt));
    this.ringSlots = this.delaySteps + 1;
    this.ring = new Int32Array(this.ringSlots * n);
    this.aCur = new Float32Array(d.ng);
    this.aNext = new Float32Array(d.ng);
    this.ext = new Float32Array(d.ng);
    this.mode = new Uint8Array(d.ng);
    this.spikeList = new Int32Array(n);
    this.spikeOffsets = d.spikeOffsets;
    this.spikeEdges = d.spikeEdges;
    this.gOff = d.gradedOffsets;
    this.gU = new Uint32Array(d.gradedEntries);
    this.gF = new Float32Array(d.gradedEntries);
    this.iOff = d.ifaceOffsets;
    this.iU = new Uint32Array(d.ifaceEntries);
    this.iF = new Float32Array(d.ifaceEntries);
    this.unitsF = new Float32Array(d.visUnits);
    this.unitsU = new Uint32Array(d.visUnits);
    this.visCount = d.visCount;
    for (let u = 0; u < d.visCount; u++) this.mode[this.unitsU[8 * u + 4]] = this.unitsU[8 * u + 6];
    d.typeId.forEach((t, i) => {
      if (i >= d.ng && d.types[t].startsWith("pC1_")) this.pC1.push(i);
    });
    this.setArousal(m.arousal);
  }

  setParams(m: ModelParams) {
    this.m = m;
    this.setArousal(m.arousal);
  }

  setArousal(mV: number) {
    this.baseDrive.fill(0);
    for (const i of this.pC1) this.baseDrive[i] = mV;
  }

  setProbes(idx: Uint32Array) {
    this.probeIdx = idx;
  }

  reset(includeGraded: boolean) {
    this.v.fill(0);
    this.g.fill(0);
    this.refr.fill(0);
    this.theta.fill(0);
    this.act.fill(0);
    this.cnt.fill(0);
    this.ring.fill(0);
    if (includeGraded) {
      this.aCur.fill(0);
      this.aNext.fill(0);
    }
  }

  graded(): Float32Array {
    return this.aCur;
  }

  /** Simulates `brainMs`; returns cumulative probe counts and spikes emitted. */
  step(brainMs: number, scene: SceneState): { counts: Uint32Array; spikes: number; frameMs: number } {
    const m = this.m;
    const steps = Math.max(0, Math.round(brainMs / m.dt));
    this.eye(scene);
    const em = Math.exp(-m.dt / m.tauM);
    const es = Math.exp(-m.dt / m.tauS);
    const kg = (m.tauS / (m.tauS - m.tauM)) * (es - em);
    const adaptDecay = Math.exp(-m.dt / m.tauAdapt);
    const actDecay = Math.exp(-m.dt / m.actTau);
    const { n, ng, v, g, refr, theta, act, cnt, baseDrive, gDrive, ring, spikeList, spikeOffsets, spikeEdges } = this;
    let spikes = 0;
    for (let s = 0; s < steps; s++) {
      if (this.brainTime >= this.nextGraded) {
        this.gradedStep();
        this.couple();
        this.nextGraded += m.gradedDt;
      }
      const readBase = (this.t % this.ringSlots) * n;
      const writeBase = ((this.t + this.delaySteps) % this.ringSlots) * n;
      let nsp = 0;
      for (let i = ng; i < n; i++) {
        const inp = ring[readBase + i];
        ring[readBase + i] = 0;
        theta[i] *= adaptDecay;
        act[i] *= actDecay;
        if (refr[i] > 0) {
          refr[i] -= m.dt;
          v[i] = m.vReset;
          g[i] += m.wSyn * inp;
          continue;
        }
        const drive = baseDrive[i] + gDrive[i];
        const vi = drive + (v[i] - drive) * em + g[i] * kg;
        g[i] = g[i] * es + m.wSyn * inp;
        if (vi > m.vThreshold + theta[i]) {
          v[i] = m.vReset;
          g[i] = 0;
          refr[i] = m.tRefractory;
          theta[i] += m.adapt;
          act[i] += 1;
          cnt[i]++;
          spikeList[nsp++] = i;
        } else {
          v[i] = vi;
        }
      }
      for (let k = 0; k < nsp; k++) {
        const pre = spikeList[k];
        for (let e = spikeOffsets[pre], end = spikeOffsets[pre + 1]; e < end; e++) {
          const packed = spikeEdges[e];
          ring[writeBase + (packed & 0x3ffff)] += (packed | 0) >> 18;
        }
      }
      spikes += nsp;
      this.brainTime += m.dt;
      this.t++;
    }
    const counts = new Uint32Array(this.probeIdx.length);
    for (let k = 0; k < counts.length; k++) counts[k] = cnt[this.probeIdx[k]];
    return { counts, spikes, frameMs: steps * m.dt };
  }

  private gradedStep() {
    const m = this.m;
    const decay = Math.exp(-m.gradedDt / m.gradedTau);
    const { ng, gOff, gU, gF, aCur, aNext, ext, mode } = this;
    const lo = m.gradedMin;
    const hi = m.gradedMax;
    for (let i = 0; i < ng; i++) {
      if (mode[i] === 1) {
        aNext[i] = ext[i];
        continue;
      }
      let sum = 0;
      for (let e = gOff[i], end = gOff[i + 1]; e < end; e++) {
        const a = aCur[gU[2 * e]];
        sum += gF[2 * e + 1] * (a < lo ? lo : a > hi ? hi : a);
      }
      let goal = m.gradedGain * sum;
      if (mode[i] === 2) goal += ext[i];
      aNext[i] = goal + (aCur[i] - goal) * decay;
    }
    this.aCur = aNext;
    this.aNext = aCur;
  }

  private couple() {
    const m = this.m;
    const { n, ng, iOff, iU, iF, aCur, gDrive } = this;
    const lo = m.gradedMin;
    const hi = m.gradedMax;
    for (let r = 0; r < n - ng; r++) {
      let sum = 0;
      for (let e = iOff[r], end = iOff[r + 1]; e < end; e++) {
        const a = aCur[iU[2 * e]];
        sum += iF[2 * e + 1] * (a < lo ? lo : a > hi ? hi : a);
      }
      gDrive[r + ng] = m.coupling * sum;
    }
  }

  /** Samples the arena for every visual unit and writes contrast into `ext` (see eye.wgsl). */
  private eye(scene: SceneState) {
    const { unitsF, unitsU, visCount } = this;
    const lum = new Float32Array(visCount);
    const sums = [0, 0, 0];
    const cnts = [0, 0, 0];
    const cp = Math.cos(scene.pitch);
    const sp = Math.sin(scene.pitch);
    const ch = Math.cos(scene.heading);
    const sh = Math.sin(scene.heading);
    for (let u = 0; u < visCount; u++) {
      const dx = unitsF[8 * u];
      const dy = unitsF[8 * u + 1];
      const dz = unitsF[8 * u + 2];
      // toWorld: pitch about x, then heading about y
      const py = dy * cp - dz * sp;
      const pz = dy * sp + dz * cp;
      const l = this.sceneLum(dx * ch - pz * sh, py, dx * sh + pz * ch, scene);
      lum[u] = l;
      if (unitsU[8 * u + 6] === 2) {
        const side = unitsU[8 * u + 5];
        sums[side] += l;
        cnts[side]++;
      }
    }
    const means = sums.map((s, i) => Math.max(s / Math.max(cnts[i], 1), 1e-3));
    for (let u = 0; u < visCount; u++) {
      const mean = means[unitsU[8 * u + 5]];
      const c = Math.min(3, Math.max(-1, (lum[u] - mean) / mean));
      this.ext[unitsU[8 * u + 4]] = unitsF[8 * u + 3] * c;
    }
  }

  /** Luminance of the analytic arena along a unit ray from the fly's head (scene.wgsl, ro = 0). */
  private sceneLum(rx: number, ry: number, rz: number, scene: SceneState): number {
    const skyT = smoothstep(-0.1, 0.7, ry);
    let r = 0.9 + 0.03 * skyT;
    let g = 0.92 + 0.02 * skyT;
    let b = 0.95 + 0.02 * skyT;
    if (ry < 0) {
      const t = Math.min((FLOOR_Y - 0) / ry, 400);
      const fx = Math.abs(fract(rx * t * 0.5 + 0.5) - 0.5);
      const fz = Math.abs(fract(rz * t * 0.5 + 0.5) - 0.5);
      const line = 1 - smoothstep(0, 0.012 + 0.004 * t, Math.min(fx, fz));
      const k = line * Math.exp(-t * 0.05);
      const fr = 0.9 - 0.06 * k;
      const fg = 0.91 - 0.05 * k;
      const fb = 0.94 - 0.04 * k;
      const mix = smoothstep(0, 0.06, -ry);
      r += (fr - r) * mix;
      g += (fg - g) * mix;
      b += (fb - b) * mix;
    }
    const at = this.atlas;
    const cells = Math.max(at.rects.length, 1);
    for (const tg of scene.targets) {
      if (!tg.visible) continue;
      const ce = Math.cos(tg.el);
      const cx = Math.sin(tg.az) * ce;
      const cy = Math.sin(tg.el);
      const cz = -Math.cos(tg.az) * ce;
      const denom = rx * cx + ry * cy + rz * cz;
      if (denom <= 0.02) continue;
      const t = TARGET_DIST / denom;
      const lx = rx * t - cx * TARGET_DIST;
      const ly = ry * t - cy * TARGET_DIST;
      const lz = rz * t - cz * TARGET_DIST;
      const rtx = Math.cos(tg.az);
      const rtz = Math.sin(tg.az);
      // up = cross(rt, c) with rt = (rtx, 0, rtz)
      const upx = -rtz * cy;
      const upy = rtz * cx - rtx * cz;
      const upz = rtx * cy;
      const halfLen = TARGET_DIST * scene.tanR;
      const qx = (lx * rtx + lz * rtz) / halfLen;
      const qy = (lx * upx + ly * upy + lz * upz) / halfLen;
      const s = at.rects[tg.variant % cells];
      const cu = s.cu + qx * s.hu * tg.flip;
      const cv = s.cv - qy * s.hu;
      if (Math.abs(cu - s.cu) > s.hu || Math.abs(cv - s.cv) > s.hv) continue;
      // premultiplied bilinear sample of the low-res atlas
      const px = ((tg.variant + cu) / cells) * at.width - 0.5;
      const py = cv * at.height - 0.5;
      const x0 = Math.max(0, Math.min(at.width - 2, Math.floor(px)));
      const y0 = Math.max(0, Math.min(at.height - 2, Math.floor(py)));
      const wx = Math.min(1, Math.max(0, px - x0));
      const wy = Math.min(1, Math.max(0, py - y0));
      const px8 = at.rgba;
      const o00 = 4 * (y0 * at.width + x0);
      const o10 = o00 + 4;
      const o01 = o00 + 4 * at.width;
      const o11 = o01 + 4;
      // weights times straight alpha -> premultiplied contributions
      const a00 = (1 - wx) * (1 - wy) * px8[o00 + 3];
      const a10 = wx * (1 - wy) * px8[o10 + 3];
      const a01 = (1 - wx) * wy * px8[o01 + 3];
      const a11 = wx * wy * px8[o11 + 3];
      const pa = (a00 + a10 + a01 + a11) / 255;
      const inv = 1 / 65025;
      const pr = (a00 * px8[o00] + a10 * px8[o10] + a01 * px8[o01] + a11 * px8[o11]) * inv;
      const pg = (a00 * px8[o00 + 1] + a10 * px8[o10 + 1] + a01 * px8[o01 + 1] + a11 * px8[o11 + 1]) * inv;
      const pb = (a00 * px8[o00 + 2] + a10 * px8[o10 + 2] + a01 * px8[o01 + 2] + a11 * px8[o11 + 2]) * inv;
      r = r * (1 - pa) + pr;
      g = g * (1 - pa) + pg;
      b = b * (1 - pa) + pb;
      if (tg.flash > 0) {
        const f = tg.flash * pa;
        r += (1 - r) * f;
        g += (0.95 - g) * f;
        b += (0.55 - b) * f;
      }
    }
    return 0.299 * r + 0.587 * g + 0.114 * b;
  }
}
