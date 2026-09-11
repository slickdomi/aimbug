// Third-person fly model: per-part rigid transforms, simple key + rim lighting.

struct MeshU {
  viewProj: mat4x4f,
  body: mat4x4f,
  aim: mat4x4f,
  wingL: mat4x4f,
  wingR: mat4x4f,
  pedestal: mat4x4f,
  camPos: vec3f,
  time: f32,
  flash: f32, // muzzle flash 0..1, brightens the neon parts
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

@group(0) @binding(0) var<uniform> U: MeshU;

struct VIn {
  @location(0) pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) color: vec3f,
  @location(3) alpha: f32,
  @location(4) part: f32,
  @location(5) emissive: f32,
};

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) color: vec3f,
  @location(3) alpha: f32,
  @location(4) emissive: f32,
};

fn partMatrix(part: f32) -> mat4x4f {
  let p = u32(part + 0.5);
  if (p == 1u) {
    return U.aim;
  }
  if (p == 2u) {
    return U.wingL;
  }
  if (p == 3u) {
    return U.wingR;
  }
  if (p == 4u) {
    return U.pedestal;
  }
  return U.body;
}

@vertex
fn vs(v: VIn) -> VOut {
  let m = partMatrix(v.part);
  let world = m * vec4f(v.pos, 1.0);
  var o: VOut;
  o.clip = U.viewProj * world;
  o.world = world.xyz;
  o.normal = normalize((m * vec4f(v.normal, 0.0)).xyz);
  o.color = v.color;
  o.alpha = v.alpha;
  o.emissive = v.emissive;
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let viewDir = normalize(U.camPos - in.world);
  // two-sided: always light the side facing the camera (independent of triangle winding)
  var n = normalize(in.normal);
  if (dot(n, viewDir) < 0.0 && in.alpha < 1.0) {
    n = -n;
  }
  let key = normalize(vec3f(0.45, 0.85, 0.35));
  let diffuse = max(dot(n, key), 0.0) * 0.8 + 0.3 * (0.5 + 0.5 * n.y);
  let rim = pow(1.0 - abs(dot(n, viewDir)), 2.5) * 0.45;
  let spec = pow(max(dot(reflect(-key, n), viewDir), 0.0), 24.0) * 0.25;
  var col = in.color * diffuse + vec3f(0.55, 0.7, 0.9) * rim + spec;
  if (in.emissive > 0.0) {
    col = in.color * (1.3 + 1.5 * U.flash);
  }
  if (in.alpha < 1.0) {
    // wings: thin film, brighter at grazing angles
    let a = in.alpha + rim * 0.6;
    return vec4f(col, clamp(a, 0.0, 0.85));
  }
  return vec4f(col, 1.0);
}
