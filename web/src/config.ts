// Model constants. These mirror the defaults of pipeline/sim_hybrid.py, where they were validated
// (closed-loop aiming from photoreceptors to DNa02); change both together.

export const DATA_URL = `${import.meta.env.BASE_URL}data/malecns-v1`;

export const MODEL = {
  // Spiking part: Shiu et al. 2024 leaky integrate-and-fire (mV, ms)
  dt: 0.5,
  vThreshold: 7, // above rest (-45 vs -52 mV)
  vReset: 0,
  tauM: 20,
  tauS: 5,
  tRefractory: 2.2,
  tDelay: 1.8,
  wSyn: 0.275,
  // Spike-frequency adaptation: without it recurrent cliques (e.g. cholinergic
  // antennal-lobe LNs) lock at max rate after any perturbation. 1 mV (with songThreshold 12) let the
  // steering pathway fire harder and acquire targets faster, but the accuracy and kill gains were within
  // noise (45 vs 41 %, 16.4 vs 15.5 kills/min in 5-minute GPU trials) and seizures doubled.
  adapt: 2.0,
  tauAdapt: 200,
  // Graded (non-spiking) optic lobe
  gradedDt: 8.33,
  gradedTau: 20,
  // 2.3, not 2.5: lamina feedback loops (L1/L2/C2/C3/T1) have eigenvalues ~0.42, so gains
  // above ~2.39 make them unstable (pipeline/graded_stability.py; graded_longrun.py: flat at 2.3,
  // creeping up at 2.5). A small LA_ME/Tm29/Tm5c/Tm31 loop (0.57) is linearly unstable from 1.75
  // but held by the gradedMin/gradedMax clamps.
  gradedGain: 2.3,
  gradedMin: -1,
  gradedMax: 4,
  prune: 0.005, // drop graded edges below this input fraction
  laminaWeight: 0.5, // synthetic R1-6 input to L1-3 (lamina is mostly outside the reconstruction)
  // graded -> spiking release scale (Hz-equivalent per unit activity). Tuned in 2-minute GPU trials
  // (13 deg targets): 1800 -> ~6 % accuracy / ~10 % on target, 3000 -> ~15 % / ~19 %, 3600 overshoots.
  // Retuned with 19 deg targets together with GAME.neckSpring (5-minute GPU trials, 4 runs each):
  // 3000 + spring 0.6 -> 37 % accuracy, 14 kills/min; 3400 + spring 0.3 -> 45 %, 17 kills/min.
  // 2600 is clearly worse. The extra drive costs a few more seizures (~1 per 3 min instead of 5).
  coupling: 3400,
  arousal: 6, // tonic mV into P1/pC1 (male courtship arousal)
  actTau: 120, // ms, spike trace for visualisation
};

export const GAME = {
  yawGain: 10, // deg/s of turning per Hz of DNa02 R-L difference (best of 3/6/10/15 in smoke trials)
  rateTau: 50, // ms, DNa02 rate filter
  // Pitch: DNp53 (both hemispheres) tracks target elevation more strongly than any other descending
  // neuron (pipeline/pitch_scan.py; in game ~0.3-0.5 Hz below, 3.5-6.5 Hz above). Nothing reliably
  // prefers targets below, so it is balanced against LC4+LPLC2, which report "a target is visible"
  // at any elevation: pitch rate = gain * (DNp53 - pitchRef * LC) - spring * pitch.
  pitchGain: 8, // deg/s per Hz
  pitchRef: 0.1, // DNp53 Hz per LC4+LPLC2 Hz that means "level"
  pitchTau: 250, // ms, slower filter: DNp53 fires only a few spikes per second
  // 1/s pull back towards the horizon. Weaker lets DNp53 hold the target: share of time inside the
  // hitbox for spring 1.2 / 0.6 / 0.3 / 0.15 / 0 was 35 / 42 / 45 / 41 / 41 % (non-lethal trials).
  neckSpring: 0.3,
  maxPitchDeg: 50,
  // Trigger: one shot per pIP10 song bout (filtered rate crossing the threshold), not per spike.
  // Song rate does not depend on aim (analyze_samples.py), so this only sets how trigger-happy it is.
  songThreshold: 15, // Hz, pIP10 left + right, slow-filtered (~0.6 shots/s in smoke trials)
  fireCooldown: 1000, // ms minimum between shots
  // Seizure: the cholinergic lLN1_bc clique in the antennal lobe (~4 mV of recurrent drive per Hz)
  // occasionally ignites after minutes; when the whole brain runs away, the spiking state is reset.
  seizureSpikesPerSec: 90000,
  seizureMs: 300,
  // Half body length as seen by the fly (her distance). The eye samples every 4.8 deg, so below
  // ~13 deg the female covers too few columns (11 deg: ~1 % on target). With coupling 3000 and
  // spring 0.6: 16 deg -> ~21 % accuracy, 19 deg -> ~37 % accuracy.
  targetRadiusDeg: 19,
  maxBrainMsPerFrame: 25,
  maxStepsPerFrame: 64,
  maxSpikesPerStep: 8192,
};

export type ModelParams = typeof MODEL;
export type GameParams = typeof GAME;
