# Aimbug

An FPS aim trainer played by the connectome of a male fruit fly, running live in the browser on WebGPU.

**Play:** https://slickdomi.github.io/aimbug/ (best with WebGPU; falls back to the CPU) · by SlickDomi · [Support on Ko-fi](https://ko-fi.com/domi_zip)

All 166,700 neurons and 25.6 M connections of the [MaleCNS v1.0](https://male-cns.janelia.org/) connectome are simulated in real time. The arena is rendered into the fly's compound eyes.

- **Turning:** the DNa02 steering neurons.
- **Looking up and down:** DNp53.
- **Firing:** one shot per bout of courtship song from pIP10, the song command neuron.
- **Targets:** photos of female flies.

There is no trained readout and no aim logic, and the game code never tells the fly where the targets are. Aiming comes out of the wiring:

```
photoreceptors → lamina → medulla (Tm1/2/9/20, T5 …) → LC4 / LPLC2 → AOTU019 → DNa02 → heading
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

The table above used a 16° target radius and the old coupling of 1800. With the current defaults (coupling 3000, 19° targets), a 2.5-minute run reached **51 % accuracy, 44 % time on target, and 18 kills/min**. Target elevation is capped so the whole female stays within the ±50° neck pitch range.

Size is a hard limit of the fly's eye, which samples every 4.8°:

| target radius | time on target |
|---|---|
| 14° | 12 % |
| 13° | 9 % |
| 11° | ~1 % (the fly is effectively blind to her) |

The fly fires about 0.5 times per second. pIP10 song rate turned out to be independent of aim (`pipeline/analyze_samples.py`), so the trigger threshold only sets how trigger-happy it is. It is a fly.

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
- `?view=3` (start in third person)

**Score:** a kill is worth +100. A kill within 2 s of her landing adds +50 (Quick Kill). Kills less than 3 s apart add +50 per chained kill (Double Kill, Triple Kill, Multi Kill). Each hit shows a Call of Duty style popup next to the crosshair.

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

**Spiking neurons** cover the central brain, the ventral nerve cord, and the visual projection neurons. They use the leaky integrate-and-fire model of Shiu et al. 2024 with its published parameters (0.275 mV per synapse, τm 20 ms, τs 5 ms, 1.8 ms delay, 2.2 ms refractory period). Synapse signs come from the predicted transmitter: ACh is excitatory, and GABA, glutamate, and histamine are inhibitory. There are three additions:
- **Spike-frequency adaptation** (+2 mV threshold per spike, τ 200 ms). Without it, recurrent cholinergic cliques such as the lLN1_bc antennal-lobe local neurons lock at maximum rate after any perturbation, and the brain never goes quiet.
- **Neuromodulators have no fast synaptic effect.** Dopamine, serotonin, and octopamine act through GPCRs.
- **P1/pC1 arousal:** a constant 6 mV drive to pC1 cells. This is what lets pIP10 sing when a female is visible.

**Graded neurons** are the optic lobe intrinsic and sensory cells, 95,501 of them. Most of these cells are non-spiking in the fly. As plain LIF units they drown retinotopic signals in noise, so here they are rate units around a resting point:

`τ da/dt = −a + 2.3 · Σ frac_ij · sign_j · clamp(a_j, −1, 4)`, with τ = 20 ms and exponential Euler at 8.33 ms.

`frac_ij` is the connectome input fraction. Edges below 0.5 % are pruned, which keeps 3.9 M of 8.9 M with no change in behaviour. Deviations of graded activity from rest drive spiking cells with a coupling of 3000 Hz-equivalent release.

That coupling was tuned in 2-minute GPU trials with 13° targets (yaw gain 10, pitch gain 8):

| coupling | accuracy | time on target |
|---|---|---|
| 1800 | ~6 % | ~10 % |
| 3000 (default) | 13–18 % | 17–22 % |
| 3600 | 13 % | 11 % |

Lower arousal or a stricter trigger mostly stops the fly from singing, so it barely fires.

The gain is 2.3 rather than 2.5 on purpose. The lamina feedback loops (L1/L2 ↔ C2/C3/T1) have eigenvalues around 0.42, so any gain above about 2.39 makes them slowly unstable (`pipeline/graded_stability.py`).

**Pitch:** DNp53 (both hemispheres) is the only descending neuron that tracks target elevation (`pipeline/pitch_scan.py`). No descending neuron prefers targets below the fly. So DNp53 is balanced against LC4 + LPLC2, which report that a target is visible at any elevation:

`pitch rate = 8 · (DNp53 − 0.1 · LC) − 0.6 · pitch`

**Seizures:** the cholinergic lLN1_bc clique in the antennal lobe gets about 4 mV of recurrent drive per Hz of its own activity. Adaptation and synaptic saturation can't hold it, and synaptic depression strong enough to would silence the visual pathway. Every few minutes it ignites the olfactory system: the brain jumps to over 90 k spikes/s. When that happens the spiking state is reset and the HUD counts a seizure.

**Eyes:** each column's viewing direction comes from the hex coordinates, oriented by a fit to lamina soma positions. The hex axes are 120° apart and the spacing is 4.8°. Both eyes cover about −12° to +131° azimuth and ±78° elevation. Photoreceptor activity is clamped to local luminance contrast, normalised per eye.

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

- Connectome: [Male CNS v1.0](https://male-cns.janelia.org/), a collaboration of FlyEM (HHMI Janelia), the University of Cambridge, the MRC Laboratory of Molecular Biology, and Google Research (CC BY 4.0). Berg, S. et al., *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*, [bioRxiv 2025](https://doi.org/10.1101/2025.10.09.680999) / [Cell 2026](https://doi.org/10.1016/j.cell.2026.08.015)
- LIF model parameters: Shiu, P. K. et al., "A Drosophila computational brain model reveals sensorimotor processing", Nature 2024
- Target photos: Rolf Dietrich Brecher, ["Drosophila melanogaster ♀"](https://commons.wikimedia.org/wiki/File:Drosophila_melanogaster_%E2%99%80_(38978426500).jpg) (CC BY 2.0), and Hannah Davis, ["Standing female Drosophila melanogaster"](https://commons.wikimedia.org/wiki/File:Standing_female_Drosophila_melanogaster.jpg) (CC BY-SA 4.0). Both are cropped, resized, and have the background removed; the derived `web/public/sprites/females.png` is shared under the same licenses.
- Inspired by [DOOMFLY](https://github.com/nftechie/doomfly)

This is a toy: reconstructed wiring with approximate dynamics, not a validated emulation of a fly.
