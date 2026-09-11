// The fly's view: one hexagon per L2 lamina column at its viewing direction,
// coloured by L2's graded activity (L2 depolarises when its column darkens).

struct Unit {
  dx: f32, dy: f32, dz: f32, weight: f32,
  g: u32, side: u32, mode: u32, pad: u32,
};

struct EyeCam {
  aspect: f32, size: f32, pad0: f32, pad1: f32,
};

@group(0) @binding(0) var<uniform> cam: EyeCam;
@group(0) @binding(1) var<storage, read> units: array<Unit>;
@group(0) @binding(2) var<storage, read> list: array<u32>;
@group(0) @binding(3) var<storage, read> graded: array<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec3f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let u = units[list[ii]];
  let az = atan2(u.dx, -u.dz);
  let el = asin(clamp(u.dy, -1.0, 1.0));
  let c = vec2f(az / 2.45, el / 1.45); // +-140 deg azimuth, +-83 deg elevation
  let corner = vec2f(f32(vi & 1u), f32((vi >> 1u) & 1u)) * 2.0 - 1.0;
  let a = graded[u.g];
  var o: VOut;
  o.pos = vec4f(c.x + corner.x * cam.size / cam.aspect, c.y + corner.y * cam.size, 0.5, 1.0);
  o.uv = corner;
  let on = clamp(a * 1.4, 0.0, 1.0);
  let off = clamp(-a * 3.0, 0.0, 1.0);
  o.color = vec3f(0.10, 0.12, 0.16) + vec3f(1.0, 0.72, 0.2) * on + vec3f(0.25, 0.45, 1.0) * off * 0.6;
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  // flat-topped hexagon
  let p = abs(in.uv);
  if (max(p.x * 0.866 + p.y * 0.5, p.y) > 0.92) {
    discard;
  }
  return vec4f(in.color, 1.0);
}
