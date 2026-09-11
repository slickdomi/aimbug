import { Sound } from "./audio";
import { GAME, MODEL } from "./config";
import { loadBrainData } from "./data/loader";
import type { BrainData } from "./data/types";
import { Arena, type ViewMode } from "./game/arena";
import { Game, type Mode } from "./game/game";
import { loadSprites } from "./game/sprites";
import { Brain, type ProbeGroup, type Readback } from "./sim/brain";
import { BrainView } from "./ui/brainview";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const LOADING_QUIPS = [
  "anaesthetising the fly",
  "slicing into 8 nm sections",
  "tracing 25,582,938 connections",
  "guessing neurotransmitters",
  "explaining FPS games to a fly",
];

function probesFor(d: BrainData): { groups: ProbeGroup[]; idx: Uint32Array } {
  const names = ["DNa02", "DNa01", "DNp53", "pIP10", "DNp01", "LC4", "LPLC2", "LC10a", "AOTU019"];
  const groups: ProbeGroup[] = [];
  const idx: number[] = [];
  for (const name of names) {
    const t = d.types.indexOf(name);
    for (const side of [1, 2]) {
      const start = idx.length;
      for (let i = 0; i < d.n; i++) if (d.typeId[i] === t && d.side[i] === side) idx.push(i);
      groups.push({ name, side, start, count: idx.length - start });
    }
  }
  return { groups, idx: new Uint32Array(idx) };
}

/** Per-group firing rates (Hz per neuron) from cumulative probe counters. */
class Rates {
  rate = new Map<string, number>();
  slowRate = new Map<string, number>();
  delta = new Map<string, number>();
  private prev: Uint32Array | null = null;
  private prevTime = 0;

  constructor(private groups: ProbeGroup[]) {}

  update(r: Readback) {
    const dtMs = r.brainTime - this.prevTime;
    if (this.prev && dtMs > 0) {
      const k = 1 - Math.exp(-dtMs / GAME.rateTau);
      const ks = 1 - Math.exp(-dtMs / GAME.pitchTau);
      for (const g of this.groups) {
        let spikes = 0;
        // counters restart from zero after a brain reset
        for (let i = g.start; i < g.start + g.count; i++) spikes += r.counts[i] >= this.prev[i] ? r.counts[i] - this.prev[i] : r.counts[i];
        const key = `${g.name}:${g.side}`;
        const inst = (spikes * 1000) / dtMs / Math.max(1, g.count);
        this.rate.set(key, (this.rate.get(key) ?? 0) + (inst - (this.rate.get(key) ?? 0)) * k);
        this.slowRate.set(key, (this.slowRate.get(key) ?? 0) + (inst - (this.slowRate.get(key) ?? 0)) * ks);
        this.delta.set(key, spikes);
      }
    }
    this.prev = r.counts.slice();
    this.prevTime = r.brainTime;
  }

  get(name: string, side: number) {
    return this.rate.get(`${name}:${side}`) ?? 0;
  }

  /** Both hemispheres, slow filter (GAME.pitchTau). */
  slowBoth(name: string) {
    return (this.slowRate.get(`${name}:1`) ?? 0) + (this.slowRate.get(`${name}:2`) ?? 0);
  }

  spikes(name: string) {
    return (this.delta.get(`${name}:1`) ?? 0) + (this.delta.get(`${name}:2`) ?? 0);
  }
}

async function main() {
  const loadLabel = $("loadLabel");
  const loadBar = $("loadBar");
  const fail = (msg: string) => {
    const el = $("loadError");
    el.hidden = false;
    el.textContent = msg;
    loadLabel.textContent = "the fly died";
  };

  if (!navigator.gpu) {
    fail("WebGPU is not available in this browser. Try a recent Chrome, Edge, or Firefox Nightly with WebGPU enabled.");
    return;
  }
  const adapter = (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" })) ?? (await navigator.gpu.requestAdapter());
  if (!adapter) {
    fail("No WebGPU adapter found.");
    return;
  }
  const want = (name: keyof GPUSupportedLimits, value: number) => Math.min(value, adapter.limits[name] as number);
  console.log("WebGPU adapter:", adapter.info?.vendor, adapter.info?.architecture, adapter.info?.description);
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: want("maxStorageBufferBindingSize", 512 * 2 ** 20),
      maxBufferSize: want("maxBufferSize", 512 * 2 ** 20),
      maxStorageBuffersPerShaderStage: want("maxStorageBuffersPerShaderStage", 8),
    },
  });
  device.lost.then((info) => fail(`GPU device lost: ${info.message}`));
  device.addEventListener("uncapturederror", (ev) => console.error("WebGPU:", (ev as GPUUncapturedErrorEvent).error.message));

  let quip = 0;
  const data = await loadBrainData((label, frac) => {
    loadLabel.textContent = `${label} · ${LOADING_QUIPS[Math.floor(quip++ / 40) % LOADING_QUIPS.length]}`;
    loadBar.style.width = `${Math.round(frac * 100)}%`;
  }).catch((e: Error) => {
    fail(e.message);
    throw e;
  });
  loadLabel.textContent = `${data.n.toLocaleString()} neurons · ${data.stats.spikeEdges.toLocaleString()} spiking + ${data.stats.gradedEdges.toLocaleString()} graded + ${data.stats.ifaceEdges.toLocaleString()} coupling synapses`;
  loadBar.style.width = "100%";

  const format = navigator.gpu.getPreferredCanvasFormat();
  const game = new Game();
  const sprites = await loadSprites(device);
  $("photoCredits").innerHTML = sprites.cells
    .map((c) => `<a href="${c.url}" target="_blank" rel="noopener">${c.credit}</a> (<a href="${c.licenseUrl}" target="_blank" rel="noopener">${c.license}</a>, background removed)`)
    .join("; ");
  const arena = new Arena(device, $<HTMLCanvasElement>("view"), format, sprites);
  const probes = probesFor(data);
  const brain = new Brain(device, data, arena.sceneBuffer, sprites, probes.groups, probes.idx);
  const brainView = new BrainView(device, $<HTMLCanvasElement>("brain"), $<HTMLCanvasElement>("eye"), format, brain, data);
  const rates = new Rates(probes.groups);
  const sound = new Sound();
  // Debug handle for the headless smoke test and the devtools console.
  const debug = { game, rates, brain, MODEL, GAME, seizures: () => seizures } as Record<string, unknown>;
  // Debug: most active cell types (spike trace ~ Hz * actTau), for chasing runaway activity.
  debug.topActive = async (k = 20) => {
    const act = await brain.readActivity();
    const byType = new Map<string, { sum: number; n: number; sc: string }>();
    for (let i = data.ng; i < data.n; i++) {
      const key = data.types[data.typeId[i]] || "(untyped)";
      const e = byType.get(key) ?? { sum: 0, n: 0, sc: data.meta.superclasses[data.superclass[i]] };
      e.sum += act[i];
      e.n++;
      byType.set(key, e);
    }
    const hz = 1000 / MODEL.actTau;
    return [...byType.entries()]
      .sort((a, b) => b[1].sum - a[1].sum)
      .slice(0, k)
      .map(([t, e]) => `${t}(${e.sc}) n=${e.n} total ${(e.sum * hz).toFixed(0)} Hz, ${((e.sum * hz) / e.n).toFixed(0)} Hz/cell`);
  };
  (window as unknown as { aimbug: object }).aimbug = debug;

  // ---- controls -------------------------------------------------------------
  // URL overrides, e.g. ?mode=static&yaw=0 (yaw may be negative: mirrored steering).
  const query = new URLSearchParams(location.search);
  const num = (key: string, fallback: number) => (query.has(key) ? parseFloat(query.get(key)!) : fallback);
  let yawGain = num("yaw", GAME.yawGain);
  let pitchGain = num("pitch", GAME.pitchGain);
  let songThreshold = num("trigger", GAME.songThreshold);
  let paused = false;
  const slider = (id: string, value: number, fmt: (v: number) => string, apply: (v: number) => void) => {
    const input = $<HTMLInputElement>(id);
    const out = $(`${id}V`);
    input.value = String(value);
    const sync = (v: number) => {
      out.textContent = fmt(v);
      apply(v);
    };
    input.addEventListener("input", () => sync(parseFloat(input.value)));
    sync(value);
  };
  const modeParam = query.get("mode");
  if (modeParam === "static" || modeParam === "strafe" || modeParam === "duo" || modeParam === "calib") {
    game.mode = modeParam;
    if (modeParam !== "calib") $<HTMLSelectElement>("mode").value = modeParam;
    game.reset();
  }
  MODEL.arousal = num("arousal", MODEL.arousal);
  MODEL.gradedGain = num("ggain", MODEL.gradedGain);
  MODEL.coupling = num("couple", MODEL.coupling);
  game.radiusDeg = num("size", game.radiusDeg);
  slider("arousal", MODEL.arousal, (v) => `${v.toFixed(1)} mV`, (v) => brain.setArousal(v));
  slider("yaw", yawGain, (v) => `${v.toFixed(1)}°/s/Hz`, (v) => (yawGain = v));
  slider("pitch", pitchGain, (v) => `${v.toFixed(1)}°/s/Hz`, (v) => (pitchGain = v));
  slider("trigger", songThreshold, (v) => `${v.toFixed(0)} Hz`, (v) => (songThreshold = v));
  slider("ggain", MODEL.gradedGain, (v) => v.toFixed(2), (v) => {
    MODEL.gradedGain = v;
    brain.writeParams();
  });
  slider("couple", MODEL.coupling, (v) => `${v}`, (v) => {
    MODEL.coupling = v;
    brain.writeParams();
  });
  slider("size", game.radiusDeg, (v) => `${v}°`, (v) => (game.radiusDeg = v));
  $<HTMLSelectElement>("mode").addEventListener("change", (e) => {
    game.mode = (e.target as HTMLSelectElement).value as Mode;
    game.reset();
  });
  for (const btn of document.querySelectorAll<HTMLButtonElement>(".knob .info")) {
    btn.addEventListener("click", () => {
      const text = btn.closest(".knob")!.querySelector<HTMLElement>(".explain")!;
      text.hidden = !text.hidden;
      btn.setAttribute("aria-expanded", String(!text.hidden));
    });
  }
  const pauseBtn = $<HTMLButtonElement>("pause");
  const togglePause = () => {
    paused = !paused;
    pauseBtn.textContent = paused ? "Resume" : "Pause";
  };
  pauseBtn.onclick = togglePause;
  $<HTMLButtonElement>("resetBrain").onclick = () => brain.reset();
  $<HTMLButtonElement>("resetScore").onclick = () => game.reset();
  const muteBtn = $<HTMLButtonElement>("mute");
  muteBtn.onclick = () => {
    sound.enabled = !sound.enabled;
    muteBtn.textContent = sound.enabled ? "Sound on" : "Sound off";
  };
  // First / third person. In third person, drag on the arena orbits around the fly and the wheel zooms.
  const viewBtn = $<HTMLButtonElement>("viewToggle");
  const setView = (mode: ViewMode) => {
    arena.mode = mode;
    viewBtn.textContent = mode === "first" ? "🎥 Third person (V)" : "👁 First person (V)";
    $("crosshair").hidden = mode === "third";
    $("view").classList.toggle("orbit", mode === "third");
  };
  viewBtn.onclick = () => setView(arena.mode === "first" ? "third" : "first");
  setView(query.get("view") === "3" ? "third" : "first");
  const viewCanvas = $<HTMLCanvasElement>("view");
  let orbitDrag: { x: number; y: number } | null = null;
  viewCanvas.addEventListener("pointerdown", (e) => {
    if (arena.mode !== "third") return;
    orbitDrag = { x: e.clientX, y: e.clientY };
    viewCanvas.setPointerCapture(e.pointerId);
  });
  viewCanvas.addEventListener("pointermove", (e) => {
    if (!orbitDrag) return;
    arena.orbit.yaw += (e.clientX - orbitDrag.x) * 0.008; // drag right turns the view right
    arena.orbit.pitch = Math.max(-0.12, Math.min(1.35, arena.orbit.pitch + (e.clientY - orbitDrag.y) * 0.006));
    orbitDrag = { x: e.clientX, y: e.clientY };
  });
  const endOrbit = () => (orbitDrag = null);
  viewCanvas.addEventListener("pointerup", endOrbit);
  viewCanvas.addEventListener("pointercancel", endOrbit);
  viewCanvas.addEventListener(
    "wheel",
    (e) => {
      if (arena.mode !== "third") return;
      e.preventDefault();
      arena.orbit.dist = Math.max(1.6, Math.min(9, arena.orbit.dist * Math.exp(e.deltaY * 0.001)));
    },
    { passive: false },
  );

  window.addEventListener("keydown", (e) => {
    if (e.key === "p" || e.key === " ") togglePause();
    if (e.key === "v" || e.key === "V") setView(arena.mode === "first" ? "third" : "first");
  });

  // ---- brain -> game ----------------------------------------------------------
  /** DNp53 (looks up) against LC4+LPLC2 ("something is there"), in Hz; > 0 means pitch up. */
  const pitchDrive = () => rates.slowBoth("DNp53") / 2 - GAME.pitchRef * (rates.slowBoth("LC4") + rates.slowBoth("LPLC2"));
  let seizures = 0;
  let triggerArmed = true;
  let runawayMs = 0;
  const seizureEl = $("seizure");
  const crosshair = $("crosshair");
  const hitmarker = $("hitmarker");
  const panicEl = $("panic");
  let spikesPerSec = 0;
  const scorePop = $("scorePop");
  game.onHit = (_target, awards) => {
    sound.splat();
    arena.flash = 1;
    hitmarker.classList.add("show");
    requestAnimationFrame(() => requestAnimationFrame(() => hitmarker.classList.remove("show")));
    // Call of Duty style: big total, one line per reason
    const total = awards.reduce((s, a) => s + a.points, 0);
    scorePop.replaceChildren();
    const big = document.createElement("b");
    big.textContent = `+${total}`;
    scorePop.append(big, ...awards.map((a) => Object.assign(document.createElement("span"), { textContent: a.label })));
    scorePop.classList.remove("show");
    void scorePop.offsetWidth; // restart the animation
    scorePop.classList.add("show");
  };
  // Song vs. aim statistics (read by the smoke test): pIP10 rate binned by crosshair offset.
  const songBins = [10, 20, 40, 80, 180].map((maxDeg) => ({ maxDeg, samples: 0, rateSum: 0, overThreshold: 0 }));
  debug.songBins = songBins;
  // ?record=1: keep per-readback samples for offline calibration (smoke test dumps them).
  const samples: number[][] = [];
  if (query.has("record")) debug.samples = samples;
  brain.onReadback = (r) => {
    rates.update(r);
    if (r.frameMs > 0) {
      const inst = (r.spikes * 1000) / r.frameMs;
      spikesPerSec += (inst - spikesPerSec) * 0.1;
      runawayMs = inst > GAME.seizureSpikesPerSec ? runawayMs + r.frameMs : 0;
      if (runawayMs > GAME.seizureMs) {
        seizures++;
        runawayMs = 0;
        spikesPerSec = 0;
        brain.reset(false);
        seizureEl.textContent = `SEIZURE #${seizures} · rebooting fly`;
        seizureEl.classList.add("show");
        setTimeout(() => seizureEl.classList.remove("show"), 1500);
        console.warn(`seizure ${seizures} at brain ${(r.brainTime / 1000).toFixed(1)}s`);
      }
    }
    const songRate = rates.get("pIP10", 1) + rates.get("pIP10", 2);
    const alive = game.targets.filter((t) => t.alive);
    if (alive.length && !paused) {
      const offDeg = (Math.min(...alive.map((t) => game.offset(t))) * 180) / Math.PI;
      const bin = songBins.find((b) => offDeg < b.maxDeg)!;
      bin.samples++;
      bin.rateSum += songRate;
      bin.overThreshold += +(songRate >= songThreshold);
      if (debug.samples && samples.length < 50000) {
        const t = alive[0];
        const deg = 180 / Math.PI;
        samples.push([
          game.relAz(t) * deg, game.relEl(t) * deg, game.time - t.spawnedAt,
          rates.get("DNp53", 1), rates.get("DNp53", 2), rates.get("DNp01", 1), rates.get("DNp01", 2),
          rates.get("LC4", 1) + rates.get("LPLC2", 1), rates.get("LC4", 2) + rates.get("LPLC2", 2),
          rates.get("DNa02", 1), rates.get("DNa02", 2), songRate,
        ]);
      }
    }
    for (let k = 0; k < Math.min(rates.spikes("pIP10"), 2); k++) sound.songPulse(k * 0.035);
    // One shot per song bout: fire when the pIP10 rate crosses the threshold, re-arm once it
    // has dropped below half of it. A male serenading continuously only fires once.
    const bout = rates.slowBoth("pIP10"); // slow filter so flicker around the threshold is one bout
    if (bout < songThreshold / 2) triggerArmed = true;
    arena.song = Math.min(1, bout / (2 * songThreshold));
    const shot = triggerArmed && bout >= songThreshold && !paused ? game.trigger() : null;
    if (shot) {
      triggerArmed = false;
      sound.pew();
      arena.addTracer(game, shot.hit);
      crosshair.classList.add("fire");
      setTimeout(() => crosshair.classList.remove("fire"), 80);
    }
    // The giant fiber idles at ~50-90 Hz whenever a big dark object is in view; only a real burst is panic.
    if (rates.get("DNp01", 1) + rates.get("DNp01", 2) > 260) {
      arena.panic = 1;
      panicEl.classList.add("show");
      requestAnimationFrame(() => requestAnimationFrame(() => panicEl.classList.remove("show")));
    }
  };

  // ---- HUD --------------------------------------------------------------------
  const setLR = (l: HTMLElement, r: HTMLElement, left: number, right: number, full: number) => {
    l.style.width = `${Math.min(50, (50 * left) / full)}%`;
    r.style.width = `${Math.min(50, (50 * right) / full)}%`;
  };
  let lastHud = 0;
  let fps = 60;
  let brainSpeed = 1;
  let budget: number = GAME.maxBrainMsPerFrame;
  const updateHud = (now: number) => {
    setLR($("dnaL"), $("dnaR"), rates.get("DNa02", 1), rates.get("DNa02", 2), 40);
    setLR($("lcL"), $("lcR"), rates.get("LC4", 1) + rates.get("LPLC2", 1), rates.get("LC4", 2) + rates.get("LPLC2", 2), 40);
    setLR($("pitchD"), $("pitchU"), Math.max(0, -pitchDrive()), Math.max(0, pitchDrive()), 3);
    $("song").style.width = `${Math.min(100, ((rates.get("pIP10", 1) + rates.get("pIP10", 2)) / (2 * songThreshold)) * 100)}%`;
    $("gf").style.width = `${Math.min(100, (rates.get("DNp01", 1) + rates.get("DNp01", 2)) * 0.8)}%`;
    if (now - lastHud < 200) return;
    lastHud = now;
    const s = game.stats;
    $("scoreTotal").textContent = s.score.toLocaleString();
    $("kills").textContent = String(s.kills);
    $("acc").textContent = s.shots ? `${Math.round((100 * s.hits) / s.shots)}%` : "–";
    $("ttk").textContent = s.kills ? `${(s.ttkSum / s.kills / 1000).toFixed(1)}s` : "–";
    $("ontarget").textContent = s.playMs > 0 ? `${Math.round((100 * s.onTargetMs) / s.playMs)}%` : "–";
    $("brainstats").textContent =
      `${Math.round(spikesPerSec).toLocaleString()} spikes/s · brain time ${brainSpeed.toFixed(2)}× · ${Math.round(fps)} fps` +
      `${seizures ? ` · ${seizures} seizure${seizures > 1 ? "s" : ""}` : ""}\n` +
      `DNa02 L ${rates.get("DNa02", 1).toFixed(0)} / R ${rates.get("DNa02", 2).toFixed(0)} Hz · DNp53 ${(rates.slowBoth("DNp53") / 2).toFixed(1)} Hz · ` +
      `heading ${((game.heading * 180) / Math.PI).toFixed(0)}° pitch ${((game.pitch * 180) / Math.PI).toFixed(0)}°`;
  };

  // ---- loop -------------------------------------------------------------------
  let last = performance.now();
  const frame = (now: number) => {
    const elapsed = now - last;
    const realDt = Math.min(100, elapsed);
    last = now;
    fps += (1000 / Math.max(1, elapsed) - fps) * 0.05;
    // Keep the frame rate up by simulating less brain time per frame on slow GPUs.
    if (elapsed > 24) budget = Math.max(4, budget * 0.95);
    else if (elapsed < 19) budget = Math.min(GAME.maxBrainMsPerFrame, budget * 1.02);
    const brainMs = paused ? 0 : Math.min(realDt, budget);
    brainSpeed += (brainMs / Math.max(1, elapsed) - brainSpeed) * 0.05;

    const yawRate = yawGain * (rates.get("DNa02", 2) - rates.get("DNa02", 1));
    const pitchRate = pitchGain * pitchDrive() - GAME.neckSpring * ((game.pitch * 180) / Math.PI);
    if (!paused) game.update(brainMs, yawRate, pitchRate);
    arena.flash = Math.max(0, arena.flash - realDt / 250);
    arena.panic = Math.max(0, arena.panic - realDt / 400);
    arena.writeScene(game);

    const encoder = device.createCommandEncoder();
    brain.encode(encoder, brainMs);
    arena.render(encoder, now / 1000, game);
    brainView.render(encoder, brain, realDt);
    device.queue.submit([encoder.finish()]);
    brain.afterSubmit();
    updateHud(now);
    requestAnimationFrame(frame);
  };

  const start = $<HTMLButtonElement>("start");
  start.hidden = false;
  start.onclick = () => {
    sound.start();
    $("loading").remove();
    last = performance.now();
    requestAnimationFrame(frame);
  };
}

main();
