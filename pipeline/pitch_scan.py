"""Which spiking cells tell 'target above' from 'target below'?

Runs the hybrid model (same settings as the browser) with a static target above,
level with and below the horizon at a few azimuths, and ranks cell types (side
resolved) whose rate changes monotonically with elevation at every azimuth.
Also prints each candidate's connectome-derived receptive-field elevation.

  python pipeline/pitch_scan.py --el 30 --az 0 --az 30 --az -30
"""

import argparse

import numpy as np

from sim_hybrid import Hybrid, run
from sim_reference import Connectome, type_side_rates

MODEL_ARGS = dict(
    gain=2.5, r0=1200, ifscale=1.0, exp_euler=True, dt_g=8.33, tau_g=20.0, prune=0.005, dt=0.5, amax=4.0,
    adapt=2.0, tau_adapt=200.0, mod_zero=1, w_lam=0.5, cgain=1.0, bg=1.0, fg=0.05, arousal=6.0,
    stim=None, closed_loop=False, yaw_gain=0.0, tau_rate=50.0, seed=1, input="contrast",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--el", type=float, default=30.0)
    ap.add_argument("--az", type=float, action="append")
    ap.add_argument("--radius", type=float, default=16.0)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.2)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--pool", choices=["dn", "spiking"], default="dn")
    args = ap.parse_args()
    ns = argparse.Namespace(**MODEL_ARGS, radius=args.radius, seconds=args.seconds, warmup=args.warmup)

    c = Connectome()
    h = Hybrid(c, ns)
    azs = args.az or [0.0, 30.0, -30.0]
    els = [args.el, 0.0, -args.el]
    sc = np.array(c.meta["superclasses"])[c.sc_id]
    pool = (sc == "descending_neuron") if args.pool == "dn" else ~h.graded
    res = np.zeros((len(azs), len(els), len(c.types) * 3))
    for i, az in enumerate(azs):
        for j, el in enumerate(els):
            rates, _, wall = run(h, ns, az, el)
            sums, cnt = type_side_rates(c, rates)
            res[i, j] = sums / np.maximum(cnt, 1)
            print(f"az {az:+.0f} el {el:+.0f}: {wall:.1f}s wall, {rates.sum():.0f} spikes/s, pool spikes/s {rates[pool].sum():.0f}", flush=True)

    up_step = res[:, 0] - res[:, 1]  # above minus level
    down_step = res[:, 1] - res[:, 2]  # level minus below
    pool_keys = sorted({t * 3 + s for t, s in zip(c.type_id[pool].tolist(), c.side[pool].tolist())})
    idx_all = np.arange(c.n)
    for label, sgn in (("UP (rate rises with elevation)", 1), ("DOWN (rate rises as target drops)", -1)):
        rows = []
        for k in pool_keys:
            steps = np.concatenate([up_step[:, k], down_step[:, k]]) * sgn
            if np.all(steps > 0):
                rows.append((steps.min(), k))
        rows.sort(reverse=True)
        print(f"\n{label}: worst-case Hz step between adjacent elevations")
        for score, k in rows[: args.top]:
            t, side = divmod(k, 3)
            cells = idx_all[(c.type_id == t) & (c.side == side) & pool]
            print(f"  {c.types[t]:14s} {'?LR'[side]} n={len(cells):3d}  step {score:5.1f}  "
                  f"rates above/level/below (mean over az) {np.round(res[:, :, k].mean(0), 1)}  "
                  f"RF el {np.average(c.rf_el[cells], weights=np.maximum(c.rf_str[cells], 1e-6)):+.0f}")
        if not rows:
            print("  (none)")


if __name__ == "__main__":
    main()
