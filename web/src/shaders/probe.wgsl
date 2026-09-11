// Copies the cumulative spike counters of selected neurons for CPU readback.

@group(0) @binding(0) var<storage, read> probeIdx: array<u32>;
@group(0) @binding(1) var<storage, read> neurons: array<Neuron>;
@group(0) @binding(2) var<storage, read_write> probeOut: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&probeIdx)) {
    return;
  }
  probeOut[id.x] = neurons[probeIdx[id.x]].cnt;
}
