/** Tiny synth: pIP10 spikes become courtship-song pulses, shots go pew, hits go splat. */
export class Sound {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  enabled = true;

  start() {
    if (this.ctx) return;
    this.ctx = new AudioContext();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.35;
    this.master.connect(this.ctx.destination);
  }

  private env(t: number, attack: number, decay: number, peak: number) {
    const g = this.ctx!.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(peak, t + attack);
    g.gain.exponentialRampToValueAtTime(1e-4, t + attack + decay);
    g.connect(this.master!);
    return g;
  }

  /** One pulse of Drosophila pulse song (~200 Hz carrier, a few cycles). */
  songPulse(delay = 0) {
    if (!this.ctx || !this.enabled) return;
    const t = this.ctx.currentTime + delay;
    const o = this.ctx.createOscillator();
    o.frequency.value = 190 + Math.random() * 30;
    o.connect(this.env(t, 0.002, 0.018, 0.5));
    o.start(t);
    o.stop(t + 0.03);
  }

  pew() {
    if (!this.ctx || !this.enabled) return;
    const t = this.ctx.currentTime;
    const o = this.ctx.createOscillator();
    o.type = "square";
    o.frequency.setValueAtTime(1400, t);
    o.frequency.exponentialRampToValueAtTime(180, t + 0.09);
    o.connect(this.env(t, 0.001, 0.09, 0.12));
    o.start(t);
    o.stop(t + 0.12);
  }

  splat() {
    if (!this.ctx || !this.enabled) return;
    const t = this.ctx.currentTime;
    const o = this.ctx.createOscillator();
    o.type = "triangle";
    o.frequency.setValueAtTime(420, t);
    o.frequency.exponentialRampToValueAtTime(60, t + 0.18);
    o.connect(this.env(t, 0.002, 0.2, 0.5));
    o.start(t);
    o.stop(t + 0.25);
  }
}
