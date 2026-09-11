// Shared declarations for the hybrid brain kernels (reference: pipeline/sim_hybrid.py).

struct Params {
  n: u32, ng: u32, ringSlots: u32, maxSpikes: u32,
  em: f32, es: f32, kg: f32, adaptDecay: f32,
  vth: f32, vreset: f32, wsyn: f32, adapt: f32,
  refr: f32, dt: f32, actDecay: f32, coupling: f32,
  gGain: f32, gDecay: f32, gMin: f32, gMax: f32,
};

struct Step {
  stepIdx: u32, readSlot: u32, writeSlot: u32, pad: u32,
};

struct Neuron {
  v: f32, g: f32, refr: f32, theta: f32, act: f32, cnt: u32,
};

struct Entry {
  col: u32, val: f32,
};
