// Samples the arena for every visual unit (photoreceptor or lamina column) and
// writes its contrast input into the graded network. Prepended with scene.wgsl.

struct Unit {
  dx: f32, dy: f32, dz: f32, weight: f32, // fly-frame view direction; weight 1 (photoreceptor) or -w (lamina)
  g: u32, side: u32, mode: u32, pad: u32,
};

struct EyeParams {
  count: u32,
  blur: f32, // radians, half acceptance angle
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<uniform> eye: EyeParams;
@group(0) @binding(2) var<storage, read> units: array<Unit>;
@group(0) @binding(3) var<storage, read_write> ext: array<f32>;
@group(0) @binding(4) var<storage, read_write> means: array<f32>; // [1] left eye, [2] right eye
@group(0) @binding(5) var flyTex: texture_2d<f32>;
@group(0) @binding(6) var flySamp: sampler;

fn lum(d: vec3f) -> f32 {
  return luminance(sceneColor(vec3f(0.0), toWorld(d, scene.heading, scene.pitch), eye.blur));
}

fn sampleUnit(d: vec3f) -> f32 {
  var a = cross(d, vec3f(0.0, 1.0, 0.0));
  if (length(a) < 1e-3) {
    a = vec3f(1.0, 0.0, 0.0);
  }
  a = normalize(a) * eye.blur;
  let b = cross(d, a);
  return (2.0 * lum(d) + lum(normalize(d + a)) + lum(normalize(d - a)) + lum(normalize(d + b)) + lum(normalize(d - b))) / 6.0;
}

@compute @workgroup_size(1)
fn meanPass(@builtin(global_invocation_id) id: vec3u) {
  let side = id.x + 1u;
  var sum = 0.0;
  var cnt = 0.0;
  // Every 3rd unit is plenty for a mean luminance and keeps this serial loop cheap.
  for (var k = 0u; k < eye.count; k += 3u) {
    let u = units[k];
    if (u.side != side || u.mode != 2u) {
      continue;
    }
    sum += lum(vec3f(u.dx, u.dy, u.dz));
    cnt += 1.0;
  }
  means[side] = max(sum / max(cnt, 1.0), 1e-3);
}

@compute @workgroup_size(128)
fn contrastPass(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= eye.count) {
    return;
  }
  let u = units[id.x];
  let m = means[u.side];
  let c = clamp((sampleUnit(vec3f(u.dx, u.dy, u.dz)) - m) / m, -1.0, 3.0);
  ext[u.g] = u.weight * c;
}
