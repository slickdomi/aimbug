// Canvas 2D fallback for the player view (no WebGPU): same arena as scene.wgsl, drawn
// with a projected floor grid and the female photos as sprites. First person only.

import type { Orbit, ViewMode } from "./arena";
import { firstPersonCamera, TARGET_DIST, type V3 } from "./camera";
import type { Game, Target } from "./game";
import type { SpriteCell } from "./sprites";

const FLOOR_Y = -1.6;

export class Arena2D {
  mode: ViewMode = "first";
  orbit: Orbit = { yaw: 0, pitch: 0.3, dist: 3.4 };
  flash = 0;
  panic = 0;
  song = 0;
  private ctx: CanvasRenderingContext2D;

  constructor(private canvas: HTMLCanvasElement, private image: ImageBitmap, private cells: SpriteCell[]) {
    this.ctx = canvas.getContext("2d")!;
  }

  writeScene(_game: Game) {}

  addTracer(_game: Game, _hit: Target | null) {}

  render(_time: number, game: Game) {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    const ctx = this.ctx;
    const cam = firstPersonCamera(game.heading, game.pitch, w / h, 100);
    const toCam = (p: V3): V3 => [
      p[0] * cam.right[0] + p[1] * cam.right[1] + p[2] * cam.right[2],
      p[0] * cam.up[0] + p[1] * cam.up[1] + p[2] * cam.up[2],
      p[0] * cam.fwd[0] + p[1] * cam.fwd[1] + p[2] * cam.fwd[2],
    ];
    const screen = (c: V3): [number, number] => [w / 2 + (c[0] / c[2] / cam.tanX) * (w / 2), h / 2 - (c[1] / c[2] / cam.tanY) * (h / 2)];

    // sky and floor (the horizon is flat: the fly never rolls)
    const horizon = screen(toCam([Math.sin(game.heading) * 1e4, 0, -Math.cos(game.heading) * 1e4]))[1];
    const sky = ctx.createLinearGradient(0, horizon - h, 0, horizon);
    sky.addColorStop(0, "rgb(237,240,247)");
    sky.addColorStop(1, "rgb(222,227,236)");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h);
    if (horizon < h) {
      ctx.fillStyle = "rgb(230,232,240)";
      ctx.fillRect(0, Math.max(0, horizon), w, h);
      ctx.strokeStyle = "rgba(150,160,180,0.35)";
      ctx.lineWidth = Math.max(1, dpr);
      ctx.beginPath();
      const near = 0.05;
      const line = (a: V3, b: V3) => {
        let ca = toCam(a);
        let cb = toCam(b);
        if (ca[2] < near && cb[2] < near) return;
        if (ca[2] < near || cb[2] < near) {
          const t = (near - ca[2]) / (cb[2] - ca[2]);
          const p: V3 = [ca[0] + (cb[0] - ca[0]) * t, ca[1] + (cb[1] - ca[1]) * t, near];
          if (ca[2] < near) ca = p;
          else cb = p;
        }
        const [ax, ay] = screen(ca);
        const [bx, by] = screen(cb);
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
      };
      for (let k = -30; k <= 30; k++) {
        line([k * 2, FLOOR_Y, -60], [k * 2, FLOOR_Y, 60]);
        line([-60, FLOOR_Y, k * 2], [60, FLOOR_Y, k * 2]);
      }
      ctx.stroke();
      // haze towards the horizon, like the exponential fade in scene.wgsl
      const haze = ctx.createLinearGradient(0, horizon, 0, horizon + h * 0.25);
      haze.addColorStop(0, "rgba(230,232,240,1)");
      haze.addColorStop(1, "rgba(230,232,240,0)");
      ctx.fillStyle = haze;
      ctx.fillRect(0, horizon, w, h * 0.25);
    }

    // female photos as billboards at TARGET_DIST
    const tanR = Math.tan((game.radiusDeg * Math.PI) / 180);
    const cellPx = this.image.height;
    for (const t of game.targets) {
      if (!t.alive && t.flash <= 0) continue;
      const ce = Math.cos(t.el);
      const c = toCam([Math.sin(t.az) * ce * TARGET_DIST, Math.sin(t.el) * TARGET_DIST, -Math.cos(t.az) * ce * TARGET_DIST]);
      if (c[2] < 0.5) continue;
      const [sx, sy] = screen(c);
      const cell = this.cells[t.variant % this.cells.length];
      const halfW = ((TARGET_DIST * tanR) / c[2] / cam.tanX) * (w / 2);
      const aspect = (cell.v1 - cell.v0) / (cell.u1 - cell.u0);
      const variant = t.variant % this.cells.length;
      ctx.save();
      ctx.translate(sx, sy);
      ctx.scale(t.facing, 1);
      ctx.drawImage(
        this.image,
        (variant + cell.u0) * cellPx, cell.v0 * cellPx, (cell.u1 - cell.u0) * cellPx, (cell.v1 - cell.v0) * cellPx,
        -halfW, -halfW * aspect, 2 * halfW, 2 * halfW * aspect,
      );
      if (t.flash > 0) {
        ctx.globalCompositeOperation = "lighter";
        ctx.fillStyle = `rgba(255,240,140,${0.6 * t.flash})`;
        ctx.beginPath();
        ctx.ellipse(0, 0, halfW * 0.7, halfW * aspect * 0.7, 0, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    if (this.flash > 0) {
      ctx.fillStyle = `rgba(255,247,204,${this.flash * 0.25})`;
      ctx.fillRect(0, 0, w, h);
    }
    const vignette = ctx.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.35, w / 2, h / 2, Math.max(w, h) * 0.75);
    vignette.addColorStop(0, "rgba(0,0,0,0)");
    vignette.addColorStop(1, `rgba(${this.panic > 0 ? "80,10,5" : "0,0,0"},${0.22 + this.panic * 0.15})`);
    ctx.fillStyle = vignette;
    ctx.fillRect(0, 0, w, h);
  }
}
