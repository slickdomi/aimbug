// Small column-major matrix helpers and the first/third-person cameras.
// World frame: +y up, heading 0 looks along -z, positive heading turns right (+x).

export type V3 = [number, number, number];
export type M4 = Float32Array;

/** Distance at which the arena places targets (only matters for the third-person view:
 *  the fly itself sees angles, which do not depend on it). */
export const TARGET_DIST = 9;

export const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const addV = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scaleV = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const normalize = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

export function mul(a: M4, b: M4): M4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  return o;
}

export const identity = (): M4 => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
export const translation = (v: V3): M4 => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, v[0], v[1], v[2], 1]);

/** Heading rotation about +y (same convention as toWorld in scene.wgsl). */
export function yaw(h: number): M4 {
  const c = Math.cos(h);
  const s = Math.sin(h);
  return new Float32Array([c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1]);
}

/** Pitch about +x; positive tips -z (forward) upwards. */
export function pitch(p: number): M4 {
  const c = Math.cos(p);
  const s = Math.sin(p);
  return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
}

/** Roll about +z (used for wing flaps). */
export function roll(a: number): M4 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return new Float32Array([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

export function transformPoint(m: M4, p: V3): V3 {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
  ];
}

/** Fly frame (+x right, +y up, -z forward) to world, after pitching then turning. */
export function toWorld(d: V3, heading: number, pitchRad: number): V3 {
  const cp = Math.cos(pitchRad);
  const sp = Math.sin(pitchRad);
  const p: V3 = [d[0], d[1] * cp - d[2] * sp, d[1] * sp + d[2] * cp];
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  return [p[0] * c - p[2] * s, p[1], p[0] * s + p[2] * c];
}

export interface CameraFrame {
  eye: V3;
  right: V3;
  up: V3;
  fwd: V3;
  tanX: number;
  tanY: number;
  viewProj: M4;
}

function frame(eye: V3, fwd: V3, upHint: V3, aspect: number, fovXDeg: number): CameraFrame {
  const right = normalize(cross(fwd, upHint));
  const up = cross(right, fwd);
  const tanX = Math.tan((fovXDeg * Math.PI) / 360);
  const tanY = tanX / aspect;
  const near = 0.05;
  const far = 400;
  const view = new Float32Array([
    right[0], up[0], -fwd[0], 0,
    right[1], up[1], -fwd[1], 0,
    right[2], up[2], -fwd[2], 0,
    -dot(right, eye), -dot(up, eye), dot(fwd, eye), 1,
  ]);
  const proj = new Float32Array([
    1 / tanX, 0, 0, 0,
    0, 1 / tanY, 0, 0,
    0, 0, far / (near - far), -1,
    0, 0, (near * far) / (near - far), 0,
  ]);
  return { eye, right, up, fwd, tanX, tanY, viewProj: mul(proj, view) };
}

export function firstPersonCamera(heading: number, pitchRad: number, aspect: number, fovXDeg: number): CameraFrame {
  return frame([0, 0, 0], toWorld([0, 0, -1], heading, pitchRad), toWorld([0, 1, 0], heading, pitchRad), aspect, fovXDeg);
}

/** World-fixed orbit camera around `target` (does not follow the fly's heading); orbitPitch > 0 looks down. */
export function thirdPersonCamera(target: V3, orbitYaw: number, orbitPitch: number, dist: number, aspect: number, fovXDeg: number): CameraFrame {
  const fwd: V3 = [Math.sin(orbitYaw) * Math.cos(orbitPitch), -Math.sin(orbitPitch), -Math.cos(orbitYaw) * Math.cos(orbitPitch)];
  const eye = addV(target, scaleV(fwd, -dist));
  return frame(eye, fwd, [0, 1, 0], aspect, fovXDeg);
}
