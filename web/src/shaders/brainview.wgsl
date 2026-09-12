// Point cloud of every neuron at its soma position. Drawn twice: a faint grey,
// alpha-blended pass for every cell (so the anatomy is visible when quiet; pos.w boosts
// cells in sparse regions, see ui/density.ts), then an
// additive glow pass for active cells only: spiking cells glow warm with their spike
// trace, graded optic-lobe cells cyan (depolarised) or magenta (hyperpolarised).

struct Neuron {
  v: f32, g: f32, refr: f32, theta: f32, act: f32, cnt: u32,
};

struct Cam {
  cx: f32, cy: f32, cz: f32, scale: f32,
  cosA: f32, sinA: f32, cosT: f32, sinT: f32,
  aspect: f32, size: f32, ng: u32, glow: u32, // glow 0 = grey base pass, 1 = activity pass
};

@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> neurons: array<Neuron>;
@group(0) @binding(3) var<storage, read> graded: array<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec4f,
};

const OFFSCREEN = vec4f(2.0, 2.0, 2.0, 1.0);

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let p4 = pos[ii];
  if (p4.w == 0.0) {
    o.pos = OFFSCREEN;
    return o;
  }
  var p = (p4.xyz - vec3f(cam.cx, cam.cy, cam.cz)) * cam.scale;
  p = vec3f(cam.cosA * p.x + cam.sinA * p.z, p.y, -cam.sinA * p.x + cam.cosA * p.z);
  p = vec3f(p.x, cam.cosT * p.y - cam.sinT * p.z, cam.sinT * p.y + cam.cosT * p.z);

  var intensity = 0.0;
  var color = vec3f(0.0);
  if (ii < cam.ng) {
    let a = graded[ii];
    intensity = clamp(abs(a) * 1.5, 0.0, 1.0);
    color = select(vec3f(0.95, 0.25, 0.8), vec3f(0.2, 0.85, 1.0), a > 0.0);
  } else {
    intensity = 1.0 - exp(-neurons[ii].act * 0.8);
    color = mix(vec3f(1.0, 0.45, 0.1), vec3f(1.0, 0.95, 0.8), intensity);
  }

  var s = cam.size;
  if (cam.glow == 1u) {
    if (intensity < 0.04) {
      o.pos = OFFSCREEN;
      return o;
    }
    o.color = vec4f(color * intensity, 1.0);
    s = cam.size * (1.0 + 1.5 * intensity);
  } else {
    o.color = vec4f(0.62, 0.66, 0.72, 0.09 * p4.w);
  }

  let corner = vec2f(f32(vi & 1u), f32((vi >> 1u) & 1u)) * 2.0 - 1.0;
  o.pos = vec4f(p.x / cam.aspect + corner.x * s / cam.aspect, -p.y + corner.y * s, 0.5, 1.0);
  o.uv = corner;
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let r = length(in.uv);
  if (r > 1.0) {
    discard;
  }
  let fall = 1.0 - r * r;
  if (cam.glow == 1u) {
    return vec4f(in.color.rgb * fall, 1.0);
  }
  return vec4f(in.color.rgb, in.color.a * fall);
}
