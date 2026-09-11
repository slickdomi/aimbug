import beamSrc from "../shaders/beam.wgsl?raw";
import meshSrc from "../shaders/flymesh.wgsl?raw";
import sceneSrc from "../shaders/scene.wgsl?raw";
import viewSrc from "../shaders/view.wgsl?raw";
import {
  addV, firstPersonCamera, identity, mul, pitch, roll, scaleV, TARGET_DIST, thirdPersonCamera, toWorld, transformPoint, translation, yaw,
  type CameraFrame, type M4, type V3,
} from "./camera";
import { AIM_PIVOT, buildFly, FLOATS_PER_VERTEX, MUZZLE, WING_ROOT_L, WING_ROOT_R } from "./flymodel";
import type { Game, Target } from "./game";
import type { SpriteAtlas } from "./sprites";

const MAX_TARGETS = 8;
const MAX_SPRITES = 4;
const MAX_BEAMS = 8;
const BEAM_FLOATS = 11;
const DEPTH_FORMAT: GPUTextureFormat = "depth24plus";
export const SCENE_BYTES = 480;

export type ViewMode = "first" | "third";

const ORBIT_TARGET: V3 = [0, -0.05, 0];

export interface Orbit {
  yaw: number; // world azimuth of the view direction (0 = looking along -z)
  pitch: number; // > 0 looks down on the fly
  dist: number;
}

interface Tracer {
  from: V3;
  to: V3;
  born: number; // performance.now()
}

/** Owns the scene uniform (read by both the eye compute pass and the player view) and draws the view. */
export class Arena {
  readonly sceneBuffer: GPUBuffer;
  private sceneData = new ArrayBuffer(SCENE_BYTES);
  private viewBuffer: GPUBuffer;
  private meshBuffer: GPUBuffer;
  private beamUniform: GPUBuffer;
  private beamInstances: GPUBuffer;
  private rayPipe: GPURenderPipeline;
  private meshPipe: GPURenderPipeline;
  private wingPipe: GPURenderPipeline;
  private beamPipe: GPURenderPipeline;
  private rayGroup: GPUBindGroup;
  private meshGroup: GPUBindGroup;
  private beamGroup: GPUBindGroup;
  private vertexBuffer: GPUBuffer;
  private opaqueIndex: GPUBuffer;
  private wingIndex: GPUBuffer;
  private opaqueCount: number;
  private wingCount: number;
  private depth: GPUTexture | null = null;
  private context: GPUCanvasContext;
  private tracers: Tracer[] = [];
  private lastShot = -1e9;
  mode: ViewMode = "first";
  orbit: Orbit = { yaw: -0.7, pitch: 0.3, dist: 3.4 }; // over the right (gun) shoulder
  flash = 0;
  panic = 0;
  song = 0; // 0..1, extends and buzzes the wings while pIP10 sings

  constructor(private device: GPUDevice, private canvas: HTMLCanvasElement, format: GPUTextureFormat, private sprites: SpriteAtlas) {
    const U = GPUBufferUsage;
    this.sceneBuffer = device.createBuffer({ size: SCENE_BYTES, usage: U.UNIFORM | U.COPY_DST });
    this.viewBuffer = device.createBuffer({ size: 144, usage: U.UNIFORM | U.COPY_DST });
    this.meshBuffer = device.createBuffer({ size: 416, usage: U.UNIFORM | U.COPY_DST });
    this.beamUniform = device.createBuffer({ size: 80, usage: U.UNIFORM | U.COPY_DST });
    this.beamInstances = device.createBuffer({ size: 4 * BEAM_FLOATS * MAX_BEAMS, usage: U.VERTEX | U.COPY_DST });
    this.context = canvas.getContext("webgpu") as GPUCanvasContext;
    this.context.configure({ device, format, alphaMode: "opaque" });

    // Full-screen ray cast (writes depth).
    const rayModule = device.createShaderModule({ code: sceneSrc + viewSrc });
    this.rayPipe = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: rayModule, entryPoint: "vs" },
      fragment: { module: rayModule, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "always" },
    });
    this.rayGroup = device.createBindGroup({
      layout: this.rayPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneBuffer } },
        { binding: 1, resource: { buffer: this.viewBuffer } },
        { binding: 2, resource: sprites.texture.createView() },
        { binding: 3, resource: sprites.sampler },
      ],
    });

    // Fly model.
    const fly = buildFly();
    const makeBuffer = (data: Float32Array | Uint32Array, usage: number) => {
      const b = device.createBuffer({ size: Math.max(8, Math.ceil(data.byteLength / 4) * 4), usage, mappedAtCreation: true });
      new (data instanceof Float32Array ? Float32Array : Uint32Array)(b.getMappedRange()).set(data);
      b.unmap();
      return b;
    };
    this.vertexBuffer = makeBuffer(fly.vertices, U.VERTEX);
    this.opaqueIndex = makeBuffer(fly.opaque, U.INDEX);
    this.wingIndex = makeBuffer(fly.translucent, U.INDEX);
    this.opaqueCount = fly.opaque.length;
    this.wingCount = fly.translucent.length;
    const meshModule = device.createShaderModule({ code: meshSrc });
    const meshLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
    });
    const f = (offset: number, format: GPUVertexFormat, shaderLocation: number): GPUVertexAttribute => ({ offset, format, shaderLocation });
    const vertexLayout: GPUVertexBufferLayout = {
      arrayStride: 4 * FLOATS_PER_VERTEX,
      attributes: [f(0, "float32x3", 0), f(12, "float32x3", 1), f(24, "float32x3", 2), f(36, "float32", 3), f(40, "float32", 4), f(44, "float32", 5)],
    };
    const meshPipeline = (blend: GPUBlendState | undefined, depthWriteEnabled: boolean) =>
      device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [meshLayout] }),
        vertex: { module: meshModule, entryPoint: "vs", buffers: [vertexLayout] },
        fragment: { module: meshModule, entryPoint: "fs", targets: [{ format, blend }] },
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled, depthCompare: "less" },
      });
    this.meshPipe = meshPipeline(undefined, true);
    this.wingPipe = meshPipeline(
      { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } },
      false,
    );
    this.meshGroup = device.createBindGroup({ layout: meshLayout, entries: [{ binding: 0, resource: { buffer: this.meshBuffer } }] });

    // Beams (laser sight, tracers, muzzle flash).
    const beamModule = device.createShaderModule({ code: beamSrc });
    this.beamPipe = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: beamModule,
        entryPoint: "vs",
        buffers: [{ arrayStride: 4 * BEAM_FLOATS, stepMode: "instance", attributes: [f(0, "float32x3", 0), f(12, "float32x3", 1), f(24, "float32x4", 2), f(40, "float32", 3)] }],
      },
      fragment: {
        module: beamModule,
        entryPoint: "fs",
        targets: [{ format, blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } } }],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: "less" },
    });
    this.beamGroup = device.createBindGroup({ layout: this.beamPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.beamUniform } }] });
  }

  writeScene(game: Game) {
    const f = new Float32Array(this.sceneData);
    const u = new Uint32Array(this.sceneData);
    f.fill(0);
    f[0] = game.heading;
    u[1] = Math.min(MAX_TARGETS, game.targets.length);
    f[2] = game.time / 1000;
    f[3] = game.pitch;
    const tanR = Math.tan((game.radiusDeg * Math.PI) / 180);
    game.targets.slice(0, MAX_TARGETS).forEach((t, i) => {
      const o = 4 + 12 * i;
      // world direction: azimuth measured from -z towards +x
      f[o] = Math.sin(t.az) * Math.cos(t.el);
      f[o + 1] = Math.sin(t.el);
      f[o + 2] = -Math.cos(t.az) * Math.cos(t.el);
      f[o + 3] = tanR;
      f[o + 4] = Math.cos(t.az);
      f[o + 5] = 0;
      f[o + 6] = Math.sin(t.az);
      f[o + 7] = t.alive || t.flash > 0 ? 1 : 0;
      f[o + 8] = t.flash;
      f[o + 9] = t.facing;
      f[o + 10] = t.variant % Math.max(1, this.sprites.cells.length);
    });
    const cells = this.sprites.cells.slice(0, MAX_SPRITES);
    cells.forEach((c, i) => {
      const o = 100 + 4 * i; // byte 400
      f[o] = (c.u0 + c.u1) / 2;
      f[o + 1] = (c.v0 + c.v1) / 2;
      f[o + 2] = (c.u1 - c.u0) / 2;
      f[o + 3] = (c.v1 - c.v0) / 2;
    });
    u[116] = cells.length; // byte 464
    this.device.queue.writeBuffer(this.sceneBuffer, 0, this.sceneData);
  }

  private aimMatrix(game: Game): M4 {
    return mul(yaw(game.heading), mul(translation(AIM_PIVOT), mul(pitch(game.pitch), translation(scaleV(AIM_PIVOT, -1)))));
  }

  /** A shot was fired: draw a tracer from the muzzle to the target (or off into the distance). */
  addTracer(game: Game, hit: Target | null) {
    const from = transformPoint(this.aimMatrix(game), MUZZLE);
    const dir = toWorld([0, 0, -1], game.heading, game.pitch);
    const to: V3 = hit
      ? scaleV([Math.sin(hit.az) * Math.cos(hit.el), Math.sin(hit.el), -Math.cos(hit.az) * Math.cos(hit.el)], TARGET_DIST)
      : addV(from, scaleV(dir, 40));
    this.lastShot = performance.now();
    this.tracers.push({ from, to, born: this.lastShot });
    if (this.tracers.length > 4) this.tracers.shift();
  }

  private camera(game: Game, aspect: number): CameraFrame {
    if (this.mode === "first") return firstPersonCamera(game.heading, game.pitch, aspect, 100);
    // Fixed pivot above the pedestal centre (the fly turns about its head at the origin),
    // so the camera stays put while the fly spins.
    return thirdPersonCamera(ORBIT_TARGET, this.orbit.yaw, this.orbit.pitch, this.orbit.dist, aspect, 80);
  }

  render(encoder: GPUCommandEncoder, time: number, game: Game) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h || !this.depth) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.depth?.destroy();
      this.depth = this.device.createTexture({ size: [w, h], format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    const cam = this.camera(game, w / h);
    const q = this.device.queue;

    const view = new Float32Array(36);
    view.set([...cam.eye, cam.tanX, ...cam.right, cam.tanY, ...cam.up, (2 * cam.tanX) / w, ...cam.fwd, this.flash], 0);
    view.set(cam.viewProj, 16);
    view.set([time, this.panic, this.mode === "first" ? this.panic * 0.004 : 0, 0], 32);
    q.writeBuffer(this.viewBuffer, 0, view);

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }],
      depthStencilAttachment: { view: this.depth.createView(), depthLoadOp: "clear", depthClearValue: 1, depthStoreOp: "store" },
    });
    pass.setPipeline(this.rayPipe);
    pass.setBindGroup(0, this.rayGroup);
    pass.draw(3);

    if (this.mode === "third") {
      const now = performance.now();
      const flash = Math.max(0, 1 - (now - this.lastShot) / 90);
      const body = yaw(game.heading);
      // both wings swing out and buzz with the song (mirror images), plus a faint idle flutter
      const spread = this.song * 1.1;
      const buzz = (this.song * 0.12 + 0.015) * Math.sin(time * 190);
      const wingL = mul(body, mul(translation(WING_ROOT_L), mul(yaw(0.05 + spread), roll(0.04 - buzz))));
      const wingR = mul(body, mul(translation(WING_ROOT_R), mul(yaw(-0.05 - spread), roll(-0.04 + buzz))));
      const mesh = new Float32Array(104);
      [cam.viewProj, body, this.aimMatrix(game), wingL, wingR, identity()].forEach((m, i) => mesh.set(m, 16 * i));
      mesh.set([...cam.eye, time, flash, 0, 0, 0], 96);
      q.writeBuffer(this.meshBuffer, 0, mesh);

      pass.setBindGroup(0, this.meshGroup);
      pass.setVertexBuffer(0, this.vertexBuffer);
      pass.setPipeline(this.meshPipe);
      pass.setIndexBuffer(this.opaqueIndex, "uint32");
      pass.drawIndexed(this.opaqueCount);
      pass.setPipeline(this.wingPipe);
      pass.setIndexBuffer(this.wingIndex, "uint32");
      pass.drawIndexed(this.wingCount);

      // beams: laser sight, muzzle flash, fading tracers
      const aim = this.aimMatrix(game);
      const muzzle = transformPoint(aim, MUZZLE);
      const dir = toWorld([0, 0, -1], game.heading, game.pitch);
      const beams: number[] = [];
      const beam = (a: V3, b: V3, rgba: number[], width: number) => beams.push(...a, ...b, ...rgba, width);
      beam(muzzle, addV(muzzle, scaleV(dir, 30)), [0.3, 0.95, 1.0, 0.22], 0.007);
      if (flash > 0) beam(muzzle, addV(muzzle, scaleV(dir, 0.35)), [0.6, 1.0, 1.0, flash], 0.07);
      this.tracers = this.tracers.filter((t) => now - t.born < 180);
      for (const t of this.tracers) beam(t.from, t.to, [0.45, 1.0, 1.0, 1 - (now - t.born) / 180], 0.03);
      const count = beams.length / BEAM_FLOATS;
      q.writeBuffer(this.beamInstances, 0, new Float32Array(beams));
      const bu = new Float32Array(20);
      bu.set(cam.viewProj, 0);
      bu.set(cam.eye, 16);
      q.writeBuffer(this.beamUniform, 0, bu);
      pass.setPipeline(this.beamPipe);
      pass.setBindGroup(0, this.beamGroup);
      pass.setVertexBuffer(0, this.beamInstances);
      pass.draw(6, count);
    }
    pass.end();
  }
}
