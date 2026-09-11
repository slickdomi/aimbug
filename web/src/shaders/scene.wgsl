// Analytic arena shared by the player view (fragment shader) and the fly's eye
// (compute shader): what the fly sees is exactly what you see.
// Requires module-scope `scene: Scene`, `flyTex: texture_2d<f32>` and
// `flySamp: sampler` bindings in the including shader.

struct Target {
  cx: f32, cy: f32, cz: f32, tanR: f32, // world unit direction of centre, tan(half body length)
  rx: f32, ry: f32, rz: f32, alive: f32, // horizontal right vector
  flash: f32, flip: f32, variant: f32, pad1: f32, // flip = +1 facing right, -1 facing left
};

struct SpriteRect {
  cu: f32, cv: f32, hu: f32, hv: f32, // centre and half size inside its atlas cell
};

struct Scene {
  heading: f32, // radians, positive = turned right
  count: u32,
  time: f32,
  pitch: f32, // radians, positive = looking up
  targets: array<Target, 8>,
  sprites: array<SpriteRect, 4>,
  spriteCount: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

// Fly frame (+x right, +y up, -z forward) -> world: pitch about x, then heading about y.
fn toWorld(d: vec3f, heading: f32, pitch: f32) -> vec3f {
  let cp = cos(pitch);
  let sp = sin(pitch);
  let p = vec3f(d.x, d.y * cp - d.z * sp, d.y * sp + d.z * cp);
  let c = cos(heading);
  let s = sin(heading);
  return vec3f(p.x * c - p.z * s, p.y, p.x * s + p.z * c);
}

const FLOOR_Y = -1.6; // floor sits 1.6 units below the fly's eye
const TARGET_DIST = 9.0; // targets are billboards this far from the fly (keep in sync with camera.ts)
const NO_HIT = 1e6;

// Nearly uniform luminance on purpose: the fly's optic lobe responds to contrast
// everywhere, so a dark floor or strong grid drives looming detectors on both sides
// and drowns the steering signal. Only the targets are high contrast.
// Returns colour and hit distance (NO_HIT for sky).
fn background(ro: vec3f, r: vec3f) -> vec4f {
  let sky = mix(vec3f(0.90, 0.92, 0.95), vec3f(0.93, 0.94, 0.97), smoothstep(-0.1, 0.7, r.y));
  if (r.y >= 0.0 || ro.y <= FLOOR_Y) {
    return vec4f(sky, NO_HIT);
  }
  let t = min((FLOOR_Y - ro.y) / r.y, 400.0);
  let p = (ro.xz + r.xz * t) * 0.5;
  let fx = abs(fract(p.x + 0.5) - 0.5);
  let fz = abs(fract(p.y + 0.5) - 0.5);
  let line = 1.0 - smoothstep(0.0, 0.012 + 0.004 * t, min(fx, fz));
  let fade = exp(-t * 0.05);
  let floorCol = mix(vec3f(0.90, 0.91, 0.94), vec3f(0.84, 0.86, 0.90), line * fade);
  return vec4f(mix(sky, floorCol, smoothstep(0.0, 0.06, -r.y)), t);
}

// Ray from `ro` along unit `r`. `aa` = angular size of one sample (radians), for mip selection.
// Returns colour and distance to the nearest opaque hit (NO_HIT if none).
fn sceneHit(ro: vec3f, r: vec3f, aa: f32) -> vec4f {
  let bg = background(ro, r);
  var col = bg.rgb;
  var depth = bg.a;
  let atlasSize = vec2f(textureDimensions(flyTex, 0));
  let cells = f32(max(scene.spriteCount, 1u));
  for (var i = 0u; i < scene.count; i++) {
    let tg = scene.targets[i];
    if (tg.alive <= 0.0) {
      continue;
    }
    // Billboard in the plane at TARGET_DIST that faces the fly. For rays from the fly
    // (ro = 0) this is exactly the gnomonic projection the eye model was tuned with.
    let c = vec3f(tg.cx, tg.cy, tg.cz);
    let denom = dot(r, c);
    if (denom <= 0.02) {
      continue;
    }
    let t = dot(c * TARGET_DIST - ro, c) / denom;
    if (t <= 0.0) {
      continue;
    }
    let rt = vec3f(tg.rx, tg.ry, tg.rz);
    let up = cross(rt, c);
    let local = ro + r * t - c * TARGET_DIST;
    let halfLen = TARGET_DIST * tg.tanR;
    // sprite coords: x in [-1, 1] spans the fly's length
    let q = vec2f(dot(local, rt), dot(local, up)) / halfLen;
    let s = scene.sprites[u32(tg.variant) % max(scene.spriteCount, 1u)];
    let cellUv = vec2f(s.cu + q.x * s.hu * tg.flip, s.cv - q.y * s.hu);
    if (abs(cellUv.x - s.cu) > s.hu || abs(cellUv.y - s.cv) > s.hv) {
      continue;
    }
    let uv = vec2f((tg.variant + cellUv.x) / cells, cellUv.y);
    let texelsPerSample = aa * t / halfLen * s.hu * atlasSize.y;
    let lod = log2(max(texelsPerSample, 1.0));
    let tex = textureSampleLevel(flyTex, flySamp, uv, lod);
    col = mix(col, tex.rgb, tex.a);
    col = mix(col, vec3f(1.0, 0.95, 0.55), tg.flash * tex.a);
    if (tex.a > 0.4) {
      depth = min(depth, t);
    }
  }
  return vec4f(col, depth);
}

fn sceneColor(ro: vec3f, r: vec3f, aa: f32) -> vec3f {
  return sceneHit(ro, r, aa).rgb;
}

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.299, 0.587, 0.114));
}
