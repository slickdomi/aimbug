// Camera-facing glowing line segments: laser sight, tracers, muzzle flash.

struct BeamU {
  viewProj: mat4x4f,
  camPos: vec3f,
  pad: f32,
};

@group(0) @binding(0) var<uniform> U: BeamU;

struct Inst {
  @location(0) a: vec3f,
  @location(1) b: vec3f,
  @location(2) color: vec4f,
  @location(3) width: f32,
};

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) color: vec4f,
  @location(1) across: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, inst: Inst) -> VOut {
  // two triangles: (0,1,2) (2,1,3); x = along 0/1, y = side -1/+1
  let along = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0)[vi];
  let side = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0)[vi];
  let p = mix(inst.a, inst.b, along);
  let axis = inst.b - inst.a;
  var s = cross(axis, U.camPos - p);
  if (length(s) < 1e-6) {
    s = vec3f(0.0, 1.0, 0.0);
  }
  let offset = normalize(s) * inst.width * side;
  var o: VOut;
  o.clip = U.viewProj * vec4f(p + offset, 1.0);
  o.color = inst.color;
  o.across = side;
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let fall = 1.0 - in.across * in.across;
  return vec4f(in.color.rgb * in.color.a * fall, 0.0);
}
