// Graded -> spiking coupling: deviations of graded release from rest become a tonic
// drive (mV) on spiking cells (visual projection neurons and beyond).

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> entries: array<Entry>;
@group(0) @binding(3) var<storage, read> a: array<f32>;
@group(0) @binding(4) var<storage, read_write> gDrive: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let r = id.x;
  if (r >= P.n - P.ng) {
    return;
  }
  var sum = 0.0;
  for (var e = offsets[r]; e < offsets[r + 1u]; e++) {
    let en = entries[e];
    sum += en.val * clamp(a[en.col], P.gMin, P.gMax);
  }
  gDrive[r + P.ng] = P.coupling * sum;
}
