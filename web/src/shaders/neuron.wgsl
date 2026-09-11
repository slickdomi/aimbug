// Spiking neuron update: Shiu et al. 2024 LIF with exact integration, plus
// spike-frequency adaptation. Voltages are relative to rest (-52 mV).

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> S: Step;
@group(0) @binding(2) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(3) var<storage, read> baseDrive: array<f32>;
@group(0) @binding(4) var<storage, read> gDrive: array<f32>;
@group(0) @binding(5) var<storage, read_write> ring: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> spikes: array<u32>;
@group(0) @binding(7) var<storage, read_write> spikeCount: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x + P.ng;
  if (i >= P.n) {
    return;
  }
  var nr = neurons[i];
  let slot = S.readSlot * P.n + i;
  let inp = f32(atomicLoad(&ring[slot]));
  atomicStore(&ring[slot], 0);
  nr.theta *= P.adaptDecay;
  nr.act *= P.actDecay;
  if (nr.refr > 0.0) {
    nr.refr -= P.dt;
    nr.v = P.vreset;
    nr.g += P.wsyn * inp;
  } else {
    let drive = baseDrive[i] + gDrive[i];
    nr.v = drive + (nr.v - drive) * P.em + nr.g * P.kg;
    nr.g = nr.g * P.es + P.wsyn * inp;
    if (nr.v > P.vth + nr.theta) {
      nr.v = P.vreset;
      nr.g = 0.0;
      nr.refr = P.refr;
      nr.theta += P.adapt;
      nr.act += 1.0;
      nr.cnt += 1u;
      let k = atomicAdd(&spikeCount[S.stepIdx], 1u);
      if (k < P.maxSpikes) {
        spikes[k] = i;
      }
    }
  }
  neurons[i] = nr;
}
