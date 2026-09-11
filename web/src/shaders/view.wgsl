// Player view: full-screen ray cast of the analytic arena from any camera (the fly's
// head in first person, an orbit camera in third person). Writes depth so the fly
// model drawn afterwards is occluded correctly. Prepended with scene.wgsl.

struct View {
  eye: vec3f,
  tanX: f32,
  right: vec3f,
  tanY: f32,
  up: vec3f,
  pixelAngle: f32,
  fwd: vec3f,
  flash: f32, // hit flash
  viewProj: mat4x4f,
  time: f32,
  panic: f32,
  shake: f32, // giant-fibre panic, radians of jitter
  pad: f32,
};

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var flyTex: texture_2d<f32>;
@group(0) @binding(3) var flySamp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) ndc: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  var o: VOut;
  o.pos = vec4f(p, 0.0, 1.0);
  o.ndc = p;
  return o;
}

struct FOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(in: VOut) -> FOut {
  let jitter = view.shake * vec2f(sin(view.time * 91.0), cos(view.time * 77.0));
  let r = normalize(view.fwd + (in.ndc.x * view.tanX + jitter.x) * view.right + (in.ndc.y * view.tanY + jitter.y) * view.up);
  let hit = sceneHit(view.eye, r, view.pixelAngle);
  var col = hit.rgb;
  col = mix(col, vec3f(1.0, 0.97, 0.8), view.flash * 0.25);
  col = mix(col, vec3f(1.0, 0.2, 0.15), view.panic * 0.18 * smoothstep(0.3, 1.2, length(in.ndc)));
  let vig = 1.0 - 0.22 * smoothstep(0.6, 1.5, length(in.ndc));
  var o: FOut;
  o.color = vec4f(col * vig, 1.0);
  o.depth = 1.0;
  if (hit.a < NO_HIT) {
    let clip = view.viewProj * vec4f(view.eye + r * hit.a, 1.0);
    o.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  }
  return o;
}
