import brainSrc from "../shaders/brainview.wgsl?raw";
import eyeSrc from "../shaders/eyeview.wgsl?raw";
import type { BrainData } from "../data/types";
import type { Brain } from "../sim/brain";
import { sparseBoost } from "./density";

function setup(canvas: HTMLCanvasElement, device: GPUDevice, format: GPUTextureFormat) {
  const ctx = canvas.getContext("webgpu") as GPUCanvasContext;
  ctx.configure({ device, format, alphaMode: "opaque" });
  return ctx;
}

function fit(canvas: HTMLCanvasElement) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return w / h;
}

/** Live brain point cloud + compound-eye mosaic, both reading GPU state directly. */
export class BrainView {
  private brainCtx: GPUCanvasContext;
  private eyeCtx: GPUCanvasContext;
  private basePipe: GPURenderPipeline;
  private glowPipe: GPURenderPipeline;
  private eyePipe: GPURenderPipeline;
  private baseCam: GPUBuffer;
  private glowCam: GPUBuffer;
  private eyeCam: GPUBuffer;
  private baseGroups: [GPUBindGroup, GPUBindGroup];
  private glowGroups: [GPUBindGroup, GPUBindGroup];
  private eyeGroups: [GPUBindGroup, GPUBindGroup];
  private eyeCount: number;
  private n: number;
  private ng: number;
  private center: [number, number, number];
  private scale: number;
  angle = 0.6;
  tilt = -0.25;
  private lastInteraction = -Infinity;

  constructor(
    private device: GPUDevice,
    private brainCanvas: HTMLCanvasElement,
    private eyeCanvas: HTMLCanvasElement,
    format: GPUTextureFormat,
    brain: Brain,
    d: BrainData,
  ) {
    this.n = d.n;
    this.ng = d.ng;
    this.brainCtx = setup(brainCanvas, device, format);
    this.eyeCtx = setup(eyeCanvas, device, format);

    // Soma positions -> vec4 (w = weight of the grey anatomy colour, 0 = no position). Frame the brain + ventral nerve cord.
    const boost = sparseBoost(d.pos, d.n);
    const pos = new Float32Array(4 * d.n);
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < d.n; i++) {
      const x = d.pos[3 * i];
      if (Number.isNaN(x)) continue;
      for (let k = 0; k < 3; k++) {
        const v = d.pos[3 * i + k];
        pos[4 * i + k] = v;
        lo[k] = Math.min(lo[k], v);
        hi[k] = Math.max(hi[k], v);
      }
      pos[4 * i + 3] = boost[i];
    }
    this.center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    this.scale = 1.8 / Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    const posBuf = device.createBuffer({ size: pos.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Float32Array(posBuf.getMappedRange()).set(pos);
    posBuf.unmap();

    const additive: GPUBlendState = {
      color: { srcFactor: "one", dstFactor: "one", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
    };
    const translucent: GPUBlendState = {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    };
    const bm = device.createShaderModule({ code: brainSrc });
    const bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });
    const pointPipe = (blend: GPUBlendState) =>
      device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        vertex: { module: bm, entryPoint: "vs" },
        fragment: { module: bm, entryPoint: "fs", targets: [{ format, blend }] },
        primitive: { topology: "triangle-strip" },
      });
    this.basePipe = pointPipe(translucent);
    this.glowPipe = pointPipe(additive);
    this.baseCam = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.glowCam = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bg = (cam: GPUBuffer, graded: GPUBuffer) =>
      device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: cam } },
          { binding: 1, resource: { buffer: posBuf } },
          { binding: 2, resource: { buffer: brain.neurons } },
          { binding: 3, resource: { buffer: graded } },
        ],
      });
    this.baseGroups = [bg(this.baseCam, brain.gradedA), bg(this.baseCam, brain.gradedB)];
    this.glowGroups = [bg(this.glowCam, brain.gradedA), bg(this.glowCam, brain.gradedB)];

    // Click and drag to rotate (horizontal) and tilt (vertical); auto-spin resumes after a pause.
    let drag: { x: number; y: number } | null = null;
    brainCanvas.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY };
      brainCanvas.setPointerCapture(e.pointerId);
    });
    brainCanvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.angle -= (e.clientX - drag.x) * 0.01; // drag right spins the brain right
      this.tilt = Math.max(-1.5, Math.min(1.5, this.tilt + (e.clientY - drag.y) * 0.01));
      drag = { x: e.clientX, y: e.clientY };
      this.lastInteraction = performance.now();
    });
    const end = (e: PointerEvent) => {
      drag = null;
      this.lastInteraction = performance.now();
      if (brainCanvas.hasPointerCapture(e.pointerId)) brainCanvas.releasePointerCapture(e.pointerId);
    };
    brainCanvas.addEventListener("pointerup", end);
    brainCanvas.addEventListener("pointercancel", end);

    // Eye mosaic: L2 lamina units only (one per column).
    const l2 = d.types.indexOf("L2");
    const unitU = new Uint32Array(d.visUnits);
    const list: number[] = [];
    for (let u = 0; u < d.visCount; u++) if (unitU[8 * u + 6] === 2 && d.typeId[unitU[8 * u + 4]] === l2) list.push(u);
    this.eyeCount = list.length;
    const listBuf = device.createBuffer({ size: Math.max(8, 4 * list.length), usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Uint32Array(listBuf.getMappedRange()).set(list);
    listBuf.unmap();
    const em = device.createShaderModule({ code: eyeSrc });
    this.eyePipe = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: em, entryPoint: "vs" },
      fragment: { module: em, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-strip" },
    });
    this.eyeCam = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const eg = (graded: GPUBuffer) =>
      device.createBindGroup({
        layout: this.eyePipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.eyeCam } },
          { binding: 1, resource: { buffer: brain.units } },
          { binding: 2, resource: { buffer: listBuf } },
          { binding: 3, resource: { buffer: graded } },
        ],
      });
    this.eyeGroups = [eg(brain.gradedA), eg(brain.gradedB)];
  }

  render(encoder: GPUCommandEncoder, brain: Brain, realDtMs: number) {
    if (performance.now() - this.lastInteraction > 4000) this.angle += realDtMs * 0.00012;
    // skip views whose panel section is collapsed (zero-size canvas): saves GPU time for the brain
    if (this.brainCanvas.clientWidth > 0) this.renderBrain(encoder, brain);
    if (this.eyeCanvas.clientWidth > 0) this.renderEye(encoder, brain);
  }

  private renderBrain(encoder: GPUCommandEncoder, brain: Brain) {
    const aspect = fit(this.brainCanvas);
    for (const [buf, glow] of [[this.baseCam, 0], [this.glowCam, 1]] as const) {
      const camData = new ArrayBuffer(48);
      new Float32Array(camData).set([
        ...this.center, this.scale, Math.cos(this.angle), Math.sin(this.angle), Math.cos(this.tilt), Math.sin(this.tilt), aspect, 0.0035,
      ]);
      new Uint32Array(camData).set([this.ng, glow], 10);
      this.device.queue.writeBuffer(buf, 0, camData);
    }
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.brainCtx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0.02, 0.025, 0.04, 1] }],
    });
    pass.setPipeline(this.basePipe);
    pass.setBindGroup(0, this.baseGroups[brain.gradedCurrent]);
    pass.draw(4, this.n);
    pass.setPipeline(this.glowPipe);
    pass.setBindGroup(0, this.glowGroups[brain.gradedCurrent]);
    pass.draw(4, this.n);
    pass.end();
  }

  private renderEye(encoder: GPUCommandEncoder, brain: Brain) {
    const eyeAspect = fit(this.eyeCanvas);
    this.device.queue.writeBuffer(this.eyeCam, 0, new Float32Array([eyeAspect, 0.022, 0, 0]));
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.eyeCtx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0.02, 0.025, 0.04, 1] }],
    });
    pass.setPipeline(this.eyePipe);
    pass.setBindGroup(0, this.eyeGroups[brain.gradedCurrent]);
    pass.draw(4, this.eyeCount);
    pass.end();
  }
}
