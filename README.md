# Aimbug

An FPS aim trainer played by the connectome of a male fruit fly, running live in the browser on WebGPU.

**Play:** https://slickdomi.github.io/aimbug/ (best with WebGPU; falls back to the CPU) · by SlickDomi · [Support on Ko-fi](https://ko-fi.com/domi_zip)

All 166,700 neurons of the [MaleCNS v1.0](https://male-cns.janelia.org/) connectome are simulated in real time, over 19.2 M of its 25.6 M connections (see [The model](#the-model) for what is left out). The arena is rendered into the fly's compound eyes.

- **Turning:** the DNa02 steering neurons.
- **Looking up and down:** DNp53.
- **Firing:** one shot per bout of courtship song from pIP10, the song command neuron.
- **Targets:** photos of female flies.

There is no trained readout and no aim logic, and the game code never tells the fly where the targets are. Aiming comes out of the wiring:

```
photoreceptors → lamina → medulla (Tm1/2/9/20, T5 …) → LC10 (mostly LC10a) → AOTU025 / AOTU019 … → DNa02 → heading
                                                     ↘ … → DNp53 (vs. LC4 + LPLC2) → pitch
```

## Does it actually aim?

These are 2-minute trials on an RX 9070, measured by `web/scripts/smoke.mjs`. Female photos strafe across ±35° of elevation. Time on target is the main measure; kill counts are noisy because the fly only fires about every 2 s.

| condition | time on target | kills / min | accuracy |
|---|---|---|---|
| default (yaw gain 10, pitch gain 8) | **17.2 %** | 4.0 | 12.7 % |
| pitch locked level | 12.1 % | 5.1 | 17.2 % |
| no yaw steering (yaw gain 0) | 0 % | 0 | 0 % |
| mirrored yaw (yaw gain −10) | 0.8 % | 0.5 | 2 % |

In an earlier 90 s run, mirroring pitch (gain −8) dropped time on target to 9.3 %.

The table above used a 16° target radius and the old coupling of 1800. With the current defaults (coupling 3400, neck spring 0.3, 19° targets), four 5-minute runs averaged **45 % accuracy (44–47 %), 43 % time on target, and 17 kills/min**. The previous defaults (coupling 3000, spring 0.6) averaged 37 % accuracy (35–39 %), 45 % time on target, and 14 kills/min in the same trials.

Weaker spike adaptation (1 mV instead of 2, with a 12 Hz trigger to keep the shot rate) turns onto a new female faster (median 550–870 ms vs 730–1190 ms), but its other gains are within run-to-run noise: 45 % accuracy and 16.4 kills/min over four runs (`?seed=1` to `4`), against 41 % and 15.5 kills/min for three default runs on the same seeds. It also doubles the seizure rate, so it is not the default. Target elevation is capped so the whole female stays within the ±50° neck pitch range.

Size is a hard limit of the model's eye, which samples every 4.8°:

| target radius | time on target |
|---|---|
| 14° | 12 % |
| 13° | 9 % |
| 11° | ~1 % (the fly is effectively blind to her) |

The fly fires about 0.6 times per second. pIP10 song rate turned out to be independent of aim: in a 5-minute trial the crosshair was on average 22° from the female while the song was above 16 Hz, and 24° overall (`pipeline/analyze_samples.py`). So the trigger threshold only sets how trigger-happy it is. It is a fly. The song does fade as she climbs, from about 12 Hz for females 15–35° below the crosshair to about 5 Hz for females 15–25° above it.

## Running it

All code runs in Docker. 

```sh
# 1. data (≈1.1 GB download, produces web/public/data/malecns-v1, ≈37 MB)
sh pipeline/fetch.sh
docker build -t aimbug-pipeline -f pipeline/Dockerfile pipeline
docker run --rm -v "$PWD/data:/work/data" -v "$PWD/web/public/data:/work/web/public/data" \
  -v "$PWD/pipeline:/work/pipeline:ro" aimbug-pipeline python pipeline/build_connectome.py
# target photos → web/public/sprites (background removed)
sh pipeline/fetch_sprites.sh
docker run --rm -v "$PWD/data:/work/data" -v "$PWD/web/public:/work/web/public" \
  -v "$PWD/pipeline:/work/pipeline:ro" -w /work aimbug-pipeline python pipeline/build_sprites.py

# 2. web app (dev server on http://localhost:5173)
cd web
docker run --rm -v "$PWD:/app" -w /app node:22-alpine npm ci
docker run --rm -p 5173:5173 -v "$PWD:/app" -w /app node:22-alpine npx vite --host 0.0.0.0

# production build → web/dist (static files, any host; data shards are < 20 MB each)
docker run --rm -v "$PWD:/app" -w /app node:22-alpine npm run build
```

WebGPU (current Chrome, Edge, Safari, or Firefox with WebGPU enabled) runs the brain in real time and enables third person. Without it, or with `?cpu=1`, the same model runs in a web worker on the CPU. That fallback draws the arena and brain view with Canvas 2D, is first person only, and runs slower than real time (about 0.3× on a desktop CPU). The game is timed in brain time, so it plays in slow motion rather than aiming worse.

URL knobs:
- `?mode=static|strafe|duo`
- `?yaw=10` and `?pitch=8` (negative values mirror the steering)
- `?trigger=15` (song Hz needed to fire)
- `?arousal=6`
- `?size=16`
- `?record=1` (keeps calibration samples for the smoke test)
- `?seed=1` (the same target sequence every run, for comparing settings)
- `?adapt=2`, `?rateTau=50`, `?pitchTau=250`, `?spring=0.3`, `?pitchRef=0.1` (constants without a slider)
- `?view=3` (start in third person)

**Score:** a kill is worth +100. A kill within 2 s of her landing adds +50 (Quick Kill). Kills less than 3 s apart add +50 per chained kill (Double Kill, Triple Kill, Multi Kill). Each hit shows a Call of Duty style popup next to the crosshair.

**Side panel:** click a section title to collapse it. The pin keeps one section (the brain, say, or what the fly sees) at the top while the rest scrolls. The layout is remembered in the browser, and collapsed brain or eye views stop rendering, which frees GPU time.

**Third person:** press `V` or use the button at the top to switch. Drag to orbit around the fly and scroll to zoom.
- The fly is a procedural 3D model standing on a pedestal in the centre of the arena, holding a blaster in its right foreleg.
- Its body turns with DNa02, and its head and gun arm tilt with the DNp53 pitch signal.
- Its wings extend and buzz with the pIP10 song rate, which is also what pulls the trigger.
- Targets sit 9 units out.
- The fly's eyes still render from its head and never see its own body or the pedestal, so aiming is identical in both views.

### Headless GPU test

```sh
docker build -t aimbug-gputest -f web/scripts/Dockerfile.gputest web/scripts
cd web && docker run --rm --device /dev/dri/renderD128 --ipc=host -v "$PWD:/app:ro" \
  -v "$PWD/../.cache/smoke:/out" -e ADAPTER=hardware -e SECONDS=60 -e QUERY='?yaw=6' \
  aimbug-gputest sh -c 'cp /app/scripts/smoke.mjs /runner/ && node /runner/smoke.mjs'

# or test a running dev server: start it as a named container, share its network
docker run -d --name aimbug-vite -v "$PWD:/app" -w /app node:22-alpine npx vite --host 0.0.0.0
docker run --rm --network container:aimbug-vite --device /dev/dri/renderD128 --ipc=host \
  -v "$PWD:/app:ro" -v "$PWD/../.cache/smoke:/out" -e ADAPTER=hardware -e APP_URL=http://localhost:5173/ \
  aimbug-gputest sh -c 'cp /app/scripts/smoke.mjs /runner/ && node /runner/smoke.mjs'
```

`ADAPTER=swiftshader` (the default) runs on the CPU. It works without a GPU but is roughly 100× slower than real time.

## The model

The model is validated in `pipeline/sim_hybrid.py`, a NumPy reference that uses the same data files. The browser model in `web/src/shaders/*.wgsl` mirrors it, and its constants live in `web/src/config.ts`.

**Spiking neurons** cover the central brain, the ventral nerve cord, and the visual projection neurons. They use the leaky integrate-and-fire model of Shiu et al. 2024 with its published parameters (0.275 mV per synapse, τm 20 ms, τs 5 ms, 1.8 ms delay, 2.2 ms refractory period). Synapse signs come from the predicted transmitter: ACh is excitatory, and GABA, glutamate, and histamine are inhibitory. The 2,070 neurons without a confident prediction (0.8 % of synapses) count as excitatory. There are three additions:
- **Spike-frequency adaptation** (+2 mV threshold per spike, τ 200 ms). Without it, recurrent cholinergic cliques such as the lLN1_bc antennal-lobe local neurons lock at maximum rate after any perturbation, and the brain never goes quiet. Halving it lets the steering pathway fire harder but doubles the seizure rate, with no clear gain in aim (see [Does it actually aim?](#does-it-actually-aim)).
- **Neuromodulators have no fast synaptic effect** on spiking cells. Dopamine, serotonin, and octopamine act through GPCRs. This is not applied inside the graded optic lobe, where 16 serotonergic and octopaminergic cells still act as fast synapses.
- **P1 arousal:** a constant 6 mV drive to the 148 male-specific P1 cells (the `pC1_*` types in MaleCNS; the dimorphic `pC1x` types get none). A visible female drives pIP10 even without it, but only to 10.5 Hz (left + right). The drive lifts that to 17 Hz, above the 15 Hz trigger (`python pipeline/sim_hybrid.py --az 0 --seconds 2`, with and without `--arousal 0`).

**Graded neurons** are the optic lobe intrinsic and sensory cells, 95,501 of them. Most of these cells are non-spiking in the fly. As plain LIF units they drown retinotopic signals in noise, so here they are rate units around a resting point:

`τ da/dt = −a + 2.3 · Σ frac_ij · sign_j · clamp(a_j, −1, 4)`, with τ = 20 ms and exponential Euler at 8.33 ms.

`frac_ij` is the connectome input fraction. Edges below 0.5 % are pruned, which keeps 3.9 M of 8.9 M with no change in behaviour. Deviations of graded activity from rest drive spiking cells with a coupling of 3400 Hz-equivalent release.

**Connections that run:** 13.2 M spiking, 3.9 M graded and 2.1 M graded → spiking, 19.2 M in total (the loading screen shows the exact counts). Left out of the 25.6 M: the 5.0 M pruned graded edges, all synapses from spiking cells back onto the optic lobe, synapses from dopamine, serotonin and octopamine cells onto spiking cells, and the R1-6 → L1-3 synapses that the lamina gap fill replaces.

That coupling was tuned in 2-minute GPU trials with 13° targets (yaw gain 10, pitch gain 8):

| coupling | accuracy | time on target |
|---|---|---|
| 1800 | ~6 % | ~10 % |
| 3000 | 13–18 % | 17–22 % |
| 3600 | 13 % | 11 % |

With 19° targets, the coupling and the neck spring were retuned together in 5-minute GPU trials (four runs each). A coupling of 3400 with a spring of 0.3 beats 3000 with 0.6: 45 % vs 37 % accuracy, 17 vs 14 kills/min. A weaker coupling (2600) or a stiffer spring (1.2) is clearly worse. The stronger drive costs a few more seizures, about one every 3 minutes instead of every 5.

Lower arousal or a stricter trigger mostly stops the fly from singing, so it barely fires.

The gain is 2.3 rather than 2.5 on purpose. The lamina feedback loops (L1/L2 ↔ C2/C3/T1) have eigenvalues around 0.42, so a gain above about 2.39 makes them unstable (`pipeline/graded_stability.py`). In 150 s runs of the optic lobe alone (`pipeline/graded_longrun.py`), activity at 2.3 stays flat and at 2.5 it keeps creeping up. Strictly, the linear model is already unstable above 1.75: a small loop of LA_ME, Tm29, Tm5c and Tm31 cells has eigenvalue 0.57. The activity clamps hold that loop in the running model.

**Steering:** LC4 and LPLC2 make no synapses onto AOTU019 or AOTU025. LC10 cells supply 75 % of AOTU025's input and 44 % of AOTU019's, with LC10a the largest single type (32 % and 23 %). AOTU025 (ACh) excites the DNa02 on its own side, and AOTU019 (GABA) inhibits the DNa02 on the other side, so both push the fly toward the target. Silencing cells in the model (`pipeline/steer_ablation.py`) confirms that LC10 does the steering:

| silenced | yaw command, target 30° right / left | closed loop from +50°: within 10° after |
|---|---|---|
| nothing | +320 / −220 °/s | 0.10 s |
| LC4 + LPLC2 | +370* / −210 | 0.10 s |
| LC10a | +290 / −110 | 0.20 s |
| all LC10 | +50* / −110 | never (drifts to +66°) |
| AOTU019 + AOTU025 | +190 / −130 | 0.25 s |

\* the run went above 90 k spikes/s (a seizure). AOTU019 and AOTU025 carry about half of the turn signal; the rest reaches DNa02 through other optic tubercle cells (AOTU001, AOTU012, AOTU015). LC4 and LPLC2 still matter for pitch, below.

**Pitch:** DNp53 (both hemispheres) tracks target elevation more strongly and consistently than any other descending neuron: about 5, 17 and 29 Hz for a female 25° below, level and 25° above (`pipeline/pitch_scan.py`). In play it averages 0.3–0.5 Hz for females below the crosshair and 3.6–6.5 Hz for females above. Several other types, such as DNg82 and DNp27, also rise with elevation, but less. The few that prefer targets below (DNpe025, DNpe034 and DNg105, left side only) are weak and inconsistent across azimuths. So DNp53 is balanced against LC4 + LPLC2, which report that a target is visible at any elevation:

`pitch rate = 8 · (DNp53 − 0.1 · LC) − 0.3 · pitch`

**Seizures:** the cholinergic lLN1_bc clique in the antennal lobe (acetylcholine is ground truth for this type in MaleCNS) gets about 4 mV of recurrent drive per Hz of its own activity, from itself, other cholinergic local neurons and projection neurons. Adaptation pushes back by only 0.4 mV per Hz, synaptic saturation can't hold it either, and synaptic depression strong enough to would silence the visual pathway. So every few minutes some input tips it over: lLN1_bc and lLN2P fire at 200–260 Hz, the Kenyon cells follow, and the brain jumps to over 90 k spikes/s. When that happens the spiking state is reset and the HUD counts a seizure. Seizures cost little aim: runs with 4 and 11 seizures (at 1 mV adaptation) tracked equally well. Two suspected artefacts are not the cause: inputs from neurons without a known transmitter (under 2 % of the clique's excitation), and central synapses driving olfactory receptor neuron terminals (blocking them left the seizure rate unchanged). A real fix probably needs antennal-lobe physiology the model lacks, such as electrical coupling and slow GABA-B inhibition.

**Eyes:** each column's viewing direction comes from the hex coordinates, oriented by a fit to lamina soma positions. The hex axes are 120° apart. The 4.8° spacing between columns (close to a real fly's) and the 12° overlap across the midline are assumed, not measured from the data. Each eye then covers about −12° to +131° azimuth (mirrored for the left eye) and ±78° elevation. Photoreceptor activity is clamped to local luminance contrast, normalised per eye.

**Lamina gap:** the MaleCNS lamina lies mostly outside the reconstructed volume. The median L1 cell gets 0 % of its input from R1-6. So each L1-3 cell receives its own column's contrast directly (weight 0.5), in place of the missing R1-6 synapses.

### How we got here

1. The raw LIF model on the full connectome never aimed. Visual signals died after the lamina, or the central brain ignited.
2. Direct stimulation showed the central half works as-is. With Shiu's parameters, LC10a-left drives DNa02-left at 80 Hz and DNa02-right stays at 0.
3. Signal loss traced back to the missing lamina synapses. Filling that gap, plus a graded optic lobe, produced hemifield-selective LC4/LPLC2 responses.
4. Adaptation stopped the central-brain ignition. After that, a closed-loop simulation turned toward targets from ±60° within about 0.5 s.

## Layout

```
pipeline/   data download, packing (build_connectome.py), reference sims, analyses
web/src/    worker (decode + matrix split), WebGPU brain, arena, UI
web/public/data/malecns-v1/   generated connectome files
```

## Releasing on GitHub Pages

The generated connectome files (about 37 MB, every file under 20 MB) and the sprite atlas are committed. CI therefore only has to build the web app, and never downloads the 1.1 GB raw release.

1. **Push:** push to `main` of https://github.com/slickdomi/aimbug.
2. **Enable Pages:** in Settings → Pages → Build and deployment, set Source to **GitHub Actions**.
3. **Deploy:** `.github/workflows/pages.yml` builds `web/` with the locked dependencies and deploys it to https://slickdomi.github.io/aimbug/. Its actions are pinned to commit SHAs of releases older than two weeks.

The build uses a relative base (`web/vite.config.ts`), so it works at `https://<user>.github.io/<repo>/` without configuration.

## Licenses

| What | License |
|---|---|
| Source code (everything not listed below) | [MIT](LICENSE), © 2026 SlickDomi |
| `web/public/data/malecns-v1/`, adapted from the Male CNS connectome v1.0 | [CC BY 4.0](web/public/data/malecns-v1/LICENSE.md) |
| `web/public/sprites/`, adapted Wikimedia Commons photos | [CC BY-SA 4.0](web/public/sprites/LICENSE.md) |

Redistributing the connectome is allowed under CC BY 4.0 with attribution and a description of changes; both are in the data directory's `LICENSE.md`. No code from Shiu et al. or DOOMFLY is included. The LIF parameters come from the published paper and are credited below.

## Credits

- Connectome: [Male CNS v1.0](https://male-cns.janelia.org/), a collaboration of FlyEM (HHMI Janelia), the University of Cambridge, the MRC Laboratory of Molecular Biology, and Google Research (CC BY 4.0). Berg, S. et al., *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, [Cell 2026](https://doi.org/10.1016/j.cell.2026.08.015); preprint: *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*, [bioRxiv 2025](https://doi.org/10.1101/2025.10.09.680999)
- LIF model parameters: Shiu, P. K. et al., "A Drosophila computational brain model reveals sensorimotor processing", Nature 2024
- Target photos: Rolf Dietrich Brecher, ["Drosophila melanogaster ♀"](https://commons.wikimedia.org/wiki/File:Drosophila_melanogaster_%E2%99%80_(38978426500).jpg) (CC BY 2.0), and Hannah Davis, ["Standing female Drosophila melanogaster"](https://commons.wikimedia.org/wiki/File:Standing_female_Drosophila_melanogaster.jpg) (CC BY-SA 4.0). Both are cropped, resized, and have the background removed; the derived `web/public/sprites/females.png` is shared under the same licenses.
- Inspired by [DOOMFLY](https://github.com/nftechie/doomfly)

This is a toy: reconstructed wiring with approximate dynamics, not a validated emulation of a fly. Some of the cells it relies on (DNa02, DNp53, lLN1_bc) are still marked preliminary in MaleCNS v1.0, so their wiring may change in later releases.
