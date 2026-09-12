// Canvas 2D fallback for the brain point cloud and the compound-eye mosaic (no WebGPU).
// Draws from activity snapshots sent by the CPU worker a few times per second.

import type { BrainInfo } from "../data/types";
import { sparseBoost } from "./density";

export class BrainView2D {
  angle = 0.6;
  tilt = -0.25;
  private act: Float32Array | null = null;
  private graded: Float32Array | null = null;
  private lastInteraction = -Infinity;
  private brainCtx: CanvasRenderingContext2D;
  private eyeCtx: CanvasRenderingContext2D;
  private center: [number, number, number];
  private scale: number;
  private eyeList: number[] = [];
  private unitsF: Float32Array;
  private unitsU: Uint32Array;
  private boost: Float32Array;

  constructor(private brainCanvas: HTMLCanvasElement, private eyeCanvas: HTMLCanvasElement, private d: BrainInfo) {
    this.boost = sparseBoost(d.pos, d.n);
    this.brainCtx = brainCanvas.getContext("2d")!;
    this.eyeCtx = eyeCanvas.getContext("2d")!;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < d.n; i++) {
      if (Number.isNaN(d.pos[3 * i])) continue;
      for (let k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k], d.pos[3 * i + k]);
        hi[k] = Math.max(hi[k], d.pos[3 * i + k]);
      }
    }
    this.center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    this.scale = 1.8 / Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    this.unitsF = new Float32Array(d.visUnits);
    this.unitsU = new Uint32Array(d.visUnits);
    const l2 = d.types.indexOf("L2");
    for (let u = 0; u < d.visCount; u++) if (this.unitsU[8 * u + 6] === 2 && d.typeId[this.unitsU[8 * u + 4]] === l2) this.eyeList.push(u);

    // drag to rotate, same feel as the WebGPU view
    let drag: { x: number; y: number } | null = null;
    brainCanvas.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY };
      brainCanvas.setPointerCapture(e.pointerId);
    });
    brainCanvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.angle -= (e.clientX - drag.x) * 0.01;
      this.tilt = Math.max(-1.5, Math.min(1.5, this.tilt + (e.clientY - drag.y) * 0.01));
      drag = { x: e.clientX, y: e.clientY };
      this.lastInteraction = performance.now();
    });
    const end = () => {
      drag = null;
      this.lastInteraction = performance.now();
    };
    brainCanvas.addEventListener("pointerup", end);
    brainCanvas.addEventListener("pointercancel", end);
  }

  setActivity(act: Float32Array, graded: Float32Array) {
    this.act = act;
    this.graded = graded;
  }

  private fit(canvas: HTMLCanvasElement) {
    const w = Math.max(1, Math.round(canvas.clientWidth));
    const h = Math.max(1, Math.round(canvas.clientHeight));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    return [w, h];
  }

  render(realDtMs: number) {
    if (performance.now() - this.lastInteraction > 4000) this.angle += realDtMs * 0.00012;
    // skip views whose panel section is collapsed
    if (this.brainCanvas.clientWidth > 0) this.renderBrain();
    if (this.eyeCanvas.clientWidth > 0) this.renderEye();
  }

  private renderBrain() {
    const { d } = this;
    const [w, h] = this.fit(this.brainCanvas);
    const ctx = this.brainCtx;
    const img = ctx.createImageData(w, h);
    const px = img.data;
    for (let k = 0; k < px.length; k += 4) {
      px[k] = 5;
      px[k + 1] = 6;
      px[k + 2] = 10;
      px[k + 3] = 255;
    }
    const ca = Math.cos(this.angle);
    const sa = Math.sin(this.angle);
    const ct = Math.cos(this.tilt);
    const st = Math.sin(this.tilt);
    const half = Math.min(w, h) / 2;
    const add = (x: number, y: number, r: number, g: number, b: number) => {
      if (x < 0 || y < 0 || x >= w || y >= h) return;
      const o = 4 * (y * w + x);
      px[o] = Math.min(255, px[o] + r);
      px[o + 1] = Math.min(255, px[o + 1] + g);
      px[o + 2] = Math.min(255, px[o + 2] + b);
    };
    for (let i = 0; i < d.n; i++) {
      const x0 = d.pos[3 * i];
      if (Number.isNaN(x0)) continue;
      const x = (x0 - this.center[0]) * this.scale;
      const y = (d.pos[3 * i + 1] - this.center[1]) * this.scale;
      const z = (d.pos[3 * i + 2] - this.center[2]) * this.scale;
      const rx = ca * x + sa * z;
      const rz = -sa * x + ca * z;
      const ry = ct * y - st * rz;
      const sx = Math.round(w / 2 + rx * half);
      const sy = Math.round(h / 2 - ry * half);
      // faint grey structure for every cell, stronger where cells are sparse (additive, so keep
      // the base low or the packed optic lobes saturate to white)
      const k0 = this.boost[i];
      add(sx, sy, 3 * k0, 3.3 * k0, 3.7 * k0);
      let intensity = 0;
      let r = 0;
      let g = 0;
      let b = 0;
      if (i < d.ng) {
        const a = this.graded ? this.graded[i] : 0;
        intensity = Math.min(1, Math.abs(a) * 1.5);
        if (a > 0) [r, g, b] = [50, 215, 255];
        else [r, g, b] = [240, 65, 205];
      } else if (this.act) {
        intensity = 1 - Math.exp(-this.act[i] * 0.8);
        [r, g, b] = [255, 115 + 125 * intensity, 25 + 180 * intensity];
      }
      if (intensity > 0.04) {
        const k = intensity;
        add(sx, sy, r * k, g * k, b * k);
        if (k > 0.3) {
          add(sx + 1, sy, r * k * 0.5, g * k * 0.5, b * k * 0.5);
          add(sx, sy + 1, r * k * 0.5, g * k * 0.5, b * k * 0.5);
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  private renderEye() {
    const [w, h] = this.fit(this.eyeCanvas);
    const ctx = this.eyeCtx;
    ctx.fillStyle = "rgb(5,6,10)";
    ctx.fillRect(0, 0, w, h);
    const size = Math.max(2, h * 0.022);
    for (const u of this.eyeList) {
      const dx = this.unitsF[8 * u];
      const dy = this.unitsF[8 * u + 1];
      const dz = this.unitsF[8 * u + 2];
      const az = Math.atan2(dx, -dz);
      const el = Math.asin(Math.max(-1, Math.min(1, dy)));
      const x = w / 2 + (az / 2.45) * (w / 2);
      const y = h / 2 - (el / 1.45) * (h / 2);
      const a = this.graded ? this.graded[this.unitsU[8 * u + 4]] : 0;
      const on = Math.max(0, Math.min(1, a * 1.4));
      const off = Math.max(0, Math.min(1, -a * 3)) * 0.6;
      const r = 26 + 255 * on + 64 * off;
      const g = 31 + 184 * on + 115 * off;
      const b = 41 + 51 * on + 255 * off;
      ctx.fillStyle = `rgb(${Math.min(255, r)},${Math.min(255, g)},${Math.min(255, b)})`;
      ctx.fillRect(x - size / 2, y - size / 2, size, size);
    }
  }
}
