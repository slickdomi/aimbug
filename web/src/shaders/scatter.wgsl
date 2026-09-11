// Event-driven spike propagation: each neuron that spiked this step adds its signed
// synapse counts to its targets' input slot `tDelay` in the future.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> S: Step;
@group(0) @binding(2) var<storage, read> spikes: array<u32>;
@group(0) @binding(3) var<storage, read_write> spikeCount: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<storage, read> edges: array<u32>;
@group(0) @binding(6) var<storage, read_write> ring: array<atomic<i32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let count = min(atomicLoad(&spikeCount[S.stepIdx]), P.maxSpikes);
  if (id.x >= count) {
    return;
  }
  let pre = spikes[id.x];
  let base = S.writeSlot * P.n;
  for (var e = offsets[pre]; e < offsets[pre + 1u]; e++) {
    let packed = edges[e];
    // target in the low 18 bits, signed synapse count in the high 14
    atomicAdd(&ring[base + (packed & 0x3FFFFu)], bitcast<i32>(packed) >> 18u);
  }
}
