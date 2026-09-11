"""Why does the spiking (Shiu LIF) network keep firing after its input stops?

Kicks a group for `--kick` seconds, then removes all input and reports what is
still active. Variants remove autapses or silence the VNC.
"""

import argparse

import numpy as np

from sim_reference import T_DLY, T_RFC, TAU_M, TAU_S, V0, VRST, VTH, W_SYN, Connectome

POI_JUMP = 0.275 * 250


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kick-group", default="LC10a")
    ap.add_argument("--kick", type=float, default=0.1)
    ap.add_argument("--after", type=float, default=0.4)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--no-autapses", action="store_true")
    ap.add_argument("--no-vnc", action="store_true", help="silence vnc_* and ascending neurons")
    ap.add_argument("--mod-zero", action="store_true", help="dopamine/serotonin/octopamine have no fast effect")
    ap.add_argument("--adapt", type=float, default=0.0, help="threshold increment per spike (mV)")
    ap.add_argument("--tau-adapt", type=float, default=200.0)
    args = ap.parse_args()

    c = Connectome()
    n = c.n
    deg = np.diff(c.offsets)
    src = np.repeat(np.arange(n), deg)
    w = c.w_signed.copy()
    if args.no_autapses:
        w[src == c.tg] = 0
    if args.mod_zero:
        # Neuromodulators act through GPCRs, not fast ionotropic transmission.
        modulatory = np.isin(c.nt, ["dopamine", "serotonin", "octopamine"])
        w[modulatory[src]] = 0
    sc = np.array(c.meta["superclasses"])[c.sc_id]
    silent = np.zeros(n, dtype=bool)
    if args.no_vnc:
        silent |= np.char.startswith(sc.astype(str), "vnc_") | np.isin(sc, ["ascending_neuron", "sensory_ascending", "efferent_ascending"])

    dt = args.dt
    D = max(1, round(T_DLY / dt))
    em, es = np.exp(-dt / TAU_M), np.exp(-dt / TAU_S)
    kg = TAU_S / (TAU_S - TAU_M) * (es - em)
    x = np.zeros(n)
    g = np.zeros(n)
    refr = np.zeros(n)
    ring = np.zeros((D + 1, n))
    rate = np.zeros(n)
    rate[c.group(args.kick_group)] = 100.0
    rng = np.random.default_rng(0)
    kick_steps = int(args.kick * 1000 / dt)
    total = kick_steps + int(args.after * 1000 / dt)
    counts = np.zeros(n)
    theta = np.zeros(n)
    ea = np.exp(-dt / args.tau_adapt)
    timeline = []
    for s in range(total):
        if s == kick_steps:
            rate[:] = 0
        theta *= ea
        inp = ring[s % (D + 1)]
        active = refr > 0
        g_new = g * es
        x_new = x * em + g * kg
        x = np.where(active, VRST - V0, x_new)
        g = np.where(active, g, g_new) + W_SYN * inp
        inp[:] = 0
        refr = np.maximum(refr - dt, 0)
        x = x + POI_JUMP * ((rng.random(n) < rate * dt / 1000) & ~active)
        spk = np.flatnonzero((x > VTH - V0 + theta) & ~active & ~silent)
        if len(spk):
            x[spk] = VRST - V0
            g[spk] = 0
            refr[spk] = T_RFC
            theta[spk] += args.adapt
            if s >= kick_steps:
                counts[spk] += 1
            starts, ends = c.offsets[spk], c.offsets[spk + 1]
            lens = ends - starts
            if lens.sum():
                idx = np.repeat(ends - lens.cumsum(), lens) + np.arange(lens.sum())
                ring[(s + D) % (D + 1)] += np.bincount(c.tg[idx], weights=w[idx], minlength=n)
        if s % int(50 / dt) == 0:
            timeline.append(len(spk))
    rates = counts / args.after
    print(vars(args))
    print("spikes/step every 50ms:", timeline)
    print(f"after kick: {rates.sum():.0f} spikes/s, active {np.mean(rates > 0):.2%}")
    by_sc = {}
    for s_ in np.unique(sc):
        m = sc == s_
        if rates[m].sum() > 0:
            by_sc[s_] = f"{rates[m].sum():.0f}"
    print("spikes/s by superclass:", by_sc)
    tn = c.types[c.type_id]
    order = np.argsort(-rates)[:25]
    print("top cells:", ", ".join(f"{tn[i]}({'?LR'[c.side[i]]},{rates[i]:.0f}Hz,{sc[i][:6]})" for i in order))
    autapse = np.zeros(n)
    np.add.at(autapse, src[src == c.tg], c.w_signed[src == c.tg])
    print("autapse weight of top cells:", [int(autapse[i]) for i in order])


if __name__ == "__main__":
    main()
