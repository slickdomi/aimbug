// Procedural low-poly male Drosophila with a blaster, for the third-person view.
// Fly frame: +x right, +y up, -z forward; the head (the fly's viewpoint) sits at the origin.

export const PART = { body: 0, aim: 1, wingL: 2, wingR: 3, pedestal: 4 } as const;
export const FLOATS_PER_VERTEX = 12; // position 3, normal 3, colour 3, alpha, part, emissive

/** Where the aim part (head, gun arm, gun) pivots with pitch, and where shots leave the barrel. */
export const AIM_PIVOT: [number, number, number] = [0, -0.02, 0.12];
export const MUZZLE: [number, number, number] = [0.26, 0.04, -1.06];
export const WING_ROOT_L: [number, number, number] = [-0.1, 0.19, 0.24];
export const WING_ROOT_R: [number, number, number] = [0.1, 0.19, 0.24];

type V3 = [number, number, number];

const BODY = [0.6, 0.62, 0.66] as V3;
const BODY_DARK = [0.34, 0.35, 0.39] as V3;
const EYE = [0.72, 0.1, 0.07] as V3;
const LEG = [0.56, 0.56, 0.6] as V3;
const GUN = [0.06, 0.065, 0.075] as V3;
const GUN_METAL = [0.2, 0.21, 0.24] as V3;
const NEON = [0.25, 0.95, 1.0] as V3;
const WING = [0.82, 0.86, 0.92] as V3;
const PEDESTAL = [0.13, 0.14, 0.16] as V3;

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

class MeshBuilder {
  v: number[] = [];
  opaque: number[] = [];
  translucent: number[] = [];

  private vertex(p: V3, n: V3, c: V3, alpha: number, part: number, emissive: number) {
    this.v.push(...p, ...n, ...c, alpha, part, emissive);
    return this.v.length / FLOATS_PER_VERTEX - 1;
  }

  private tri(a: number, b: number, c: number, translucent: boolean) {
    (translucent ? this.translucent : this.opaque).push(a, b, c);
  }

  ellipsoid(center: V3, r: V3, color: V3 | ((p: V3) => V3), part: number, lat = 12, lon = 18, emissive = 0) {
    const base = this.v.length / FLOATS_PER_VERTEX;
    for (let i = 0; i <= lat; i++) {
      const th = (Math.PI * i) / lat;
      for (let j = 0; j <= lon; j++) {
        const ph = (2 * Math.PI * j) / lon;
        const u: V3 = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
        const p: V3 = [center[0] + u[0] * r[0], center[1] + u[1] * r[1], center[2] + u[2] * r[2]];
        const c = typeof color === "function" ? color(p) : color;
        this.vertex(p, norm([u[0] / r[0], u[1] / r[1], u[2] / r[2]]), c, 1, part, emissive);
      }
    }
    for (let i = 0; i < lat; i++) {
      for (let j = 0; j < lon; j++) {
        const a = base + i * (lon + 1) + j;
        const b = a + lon + 1;
        this.tri(a, b, a + 1, false);
        this.tri(a + 1, b, b + 1, false);
      }
    }
  }

  /** Cylinder from a to b with rounded joints. */
  limb(a: V3, b: V3, radius: number, color: V3, part: number, sides = 8, joints = true) {
    const axis = sub(b, a);
    const d = norm(axis);
    const helper: V3 = Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const u = norm(cross(d, helper));
    const w = cross(d, u);
    const base = this.v.length / FLOATS_PER_VERTEX;
    for (let k = 0; k <= sides; k++) {
      const ang = (2 * Math.PI * k) / sides;
      const n = add(scale(u, Math.cos(ang)), scale(w, Math.sin(ang)));
      this.vertex(add(a, scale(n, radius)), n, color, 1, part, 0);
      this.vertex(add(b, scale(n, radius * 0.85)), n, color, 1, part, 0);
    }
    for (let k = 0; k < sides; k++) {
      const i = base + 2 * k;
      this.tri(i, i + 2, i + 1, false);
      this.tri(i + 1, i + 2, i + 3, false);
    }
    if (joints) {
      this.ellipsoid(a, [radius, radius, radius], color, part, 5, 8);
      this.ellipsoid(b, [radius * 0.85, radius * 0.85, radius * 0.85], color, part, 5, 8);
    }
  }

  box(center: V3, size: V3, color: V3, part: number, emissive = 0) {
    const [hx, hy, hz] = [size[0] / 2, size[1] / 2, size[2] / 2];
    const faces: [V3, V3, V3][] = [
      [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
      [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
      [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
      [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
      [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    ];
    // half extent along an axis-aligned unit vector
    const ext = (vec: V3) => Math.abs(vec[0]) * hx + Math.abs(vec[1]) * hy + Math.abs(vec[2]) * hz;
    for (const [n, s, t] of faces) {
      const base = this.v.length / FLOATS_PER_VERTEX;
      const c0 = add(center, scale(n, ext(n)));
      for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        const p = add(add(c0, scale(s, a * ext(s))), scale(t, b * ext(t)));
        this.vertex(p, n, color, 1, part, emissive);
      }
      this.tri(base, base + 1, base + 2, false);
      this.tri(base, base + 2, base + 3, false);
    }
  }

  /** Flat translucent wing in the xz plane, root at the origin, extending along `dir` (x sign). */
  wing(dirX: number, part: number) {
    const ring = 20;
    const len = 0.78;
    const wid = 0.24;
    const vein = (t: number) => (Math.abs(Math.sin(t * 9)) < 0.12 ? 0.75 : 1);
    for (const side of [1, -1]) {
      const n: V3 = [0, side, 0];
      const center = this.vertex([dirX * 0.1, 0, len * 0.5], n, WING, 0.32, part, 0);
      const first = this.v.length / FLOATS_PER_VERTEX;
      for (let k = 0; k <= ring; k++) {
        const a = (2 * Math.PI * k) / ring;
        const x = Math.cos(a) * wid;
        const z = Math.sin(a) * len * 0.5 + len * 0.5;
        // rotate the ellipse outward a little so the wings splay like a resting fly's
        const splay = 0.35 * dirX;
        const px = x * Math.cos(splay) + z * Math.sin(splay) + dirX * 0.1;
        const pz = -x * Math.sin(splay) + z * Math.cos(splay);
        this.vertex([px, 0, pz], n, scale(WING, vein(a)), 0.32, part, 0);
      }
      for (let k = 0; k < ring; k++) {
        if (side > 0) this.tri(center, first + k, first + k + 1, true);
        else this.tri(center, first + k + 1, first + k, true);
      }
    }
  }
}

export interface FlyMesh {
  vertices: Float32Array;
  opaque: Uint32Array;
  translucent: Uint32Array;
}

export function buildFly(): FlyMesh {
  const m = new MeshBuilder();
  const A = PART.aim;
  const B = PART.body;

  // head, compound eyes, antennae, proboscis (move with pitch)
  m.ellipsoid([0, 0.02, -0.02], [0.18, 0.16, 0.13], BODY, A);
  for (const s of [-1, 1]) {
    m.ellipsoid([s * 0.13, 0.04, -0.05], [0.09, 0.125, 0.1], EYE, A, 10, 14);
    m.limb([s * 0.035, 0.09, -0.14], [s * 0.06, 0.2, -0.22], 0.012, BODY_DARK, A, 5);
    m.limb([s * 0.06, 0.2, -0.22], [s * 0.1, 0.27, -0.2], 0.006, BODY_DARK, A, 4, false);
  }
  m.limb([0, -0.1, -0.1], [0, -0.2, -0.14], 0.025, BODY_DARK, A, 6);

  // thorax with bristles, striped abdomen
  m.ellipsoid([0, 0.02, 0.27], [0.21, 0.2, 0.25], BODY, B);
  for (let k = 0; k < 6; k++) {
    const x = (k - 2.5) * 0.05;
    m.limb([x, 0.2, 0.2 + (k % 2) * 0.08], [x * 1.3, 0.3, 0.3 + (k % 2) * 0.08], 0.006, BODY_DARK, B, 4, false);
  }
  const stripes = (p: V3): V3 => (p[2] > 0.58 && Math.sin((p[2] - 0.58) * 26) > 0.35 && p[1] > -0.05 ? BODY_DARK : BODY);
  m.ellipsoid([0, -0.03, 0.72], [0.2, 0.18, 0.33], stripes, B, 14, 20);

  // legs: short, knees low and splayed, body close to the pedestal (feet at y = -0.44)
  const leg = (hip: V3, knee: V3, foot: V3, part: number) => {
    m.limb(hip, knee, 0.032, LEG, part, 7);
    m.limb(knee, foot, 0.024, LEG, part, 7);
    m.limb(foot, add(foot, [0, -0.02, -0.06]), 0.015, BODY_DARK, part, 5);
  };
  leg([-0.11, -0.14, 0.16], [-0.26, -0.16, 0.0], [-0.3, -0.44, -0.1], B);
  for (const s of [-1, 1]) {
    leg([s * 0.13, -0.15, 0.3], [s * 0.34, -0.14, 0.3], [s * 0.44, -0.44, 0.32], B);
    leg([s * 0.12, -0.14, 0.42], [s * 0.3, -0.16, 0.6], [s * 0.36, -0.44, 0.78], B);
  }

  // right foreleg raised forward, holding the blaster (moves with pitch)
  m.limb([0.12, -0.12, 0.14], [0.33, -0.14, -0.12], 0.032, LEG, A, 7);
  m.limb([0.33, -0.14, -0.12], [0.26, -0.1, -0.4], 0.024, LEG, A, 7);

  // blaster, barrel along -z
  m.box([0.26, -0.13, -0.42], [0.07, 0.2, 0.08], GUN, A); // grip
  m.box([0.26, 0.02, -0.58], [0.11, 0.13, 0.46], GUN_METAL, A); // body
  m.box([0.26, -0.12, -0.62], [0.06, 0.17, 0.07], GUN, A); // magazine
  m.box([0.26, 0.09, -0.58], [0.116, 0.02, 0.38], NEON, A, 1); // neon strip
  m.box([0.26, 0.12, -0.5], [0.04, 0.05, 0.14], GUN, A); // sight
  m.limb([0.26, 0.04, -0.8], [0.26, 0.04, -1.04], 0.034, GUN, A, 10);
  m.ellipsoid([0.26, 0.04, -1.045], [0.042, 0.042, 0.016], NEON, A, 4, 10, 1); // muzzle ring

  // wings (translucent), relative to their roots
  m.wing(-1, PART.wingL);
  m.wing(1, PART.wingR);

  // pedestal from the floor (y = -1.6) up to the feet
  // (world-fixed and centred on the head, so the turning fly stays on top of it)
  m.box([0, -1.02, 0], [2.6, 1.16, 2.6], PEDESTAL, PART.pedestal);

  return { vertices: new Float32Array(m.v), opaque: new Uint32Array(m.opaque), translucent: new Uint32Array(m.translucent) };
}
