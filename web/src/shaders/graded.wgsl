// Graded (non-spiking) optic lobe: rate units around a resting operating point.
//   tau da/dt = -a + gain * sum_j frac_ij * clamp(a_j) + lamina input
// integrated with exponential Euler. Photoreceptors are clamped to contrast.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> entries: array<Entry>;
@group(0) @binding(3) var<storage, read> aIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> aOut: array<f32>;
@group(0) @binding(5) var<storage, read> ext: array<f32>;
@group(0) @binding(6) var<storage, read> mode: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= P.ng) {
    return;
  }
  if (mode[i] == 1u) {
    aOut[i] = ext[i];
    return;
  }
  var sum = 0.0;
  for (var e = offsets[i]; e < offsets[i + 1u]; e++) {
    let en = entries[e];
    sum += en.val * clamp(aIn[en.col], P.gMin, P.gMax);
  }
  var goal = P.gGain * sum;
  if (mode[i] == 2u) {
    goal += ext[i];
  }
  aOut[i] = goal + (aIn[i] - goal) * P.gDecay;
}
