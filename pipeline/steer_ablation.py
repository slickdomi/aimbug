"""Which cells does the hybrid model steer with? Silence candidates and compare DNa02.

Browser settings (web/src/config.ts). Open loop: static target left/right, DNa02 L/R rates and the
resulting yaw command. Closed loop: target starts at +50 deg; prints rel az (DNa02 L/R Hz) every 50 ms.
Silenced cells keep spiking but have no outgoing synapses.

  python pipeline/steer_ablation.py
"""

import argparse

import numpy as np
import scipy.sparse as sp

from sim_hybrid import Hybrid, run
from sim_reference import Connectome

MODEL_ARGS = dict(
    gain=2.3, r0=3400.0, ifscale=1.0, exp_euler=True, dt_g=8.33, tau_g=20.0, prune=0.005, dt=0.5, amax=4.0,
    adapt=2.0, tau_adapt=200.0, mod_zero=1, w_lam=0.5, cgain=1.0, bg=1.0, fg=0.05, arousal=6.0,
    stim=None, closed_loop=False, yaw_gain=10.0, tau_rate=50.0, seed=1, input="contrast",
    radius=19.0, seconds=1.0, warmup=0.2,
)
SHOW = ["LC10", "LC4", "LPLC2", "AOTU019", "AOTU025", "DNa02"]
CONDITIONS = [
    ("intact", []),
    ("LC4+LPLC2 silenced", ["LC4", "LPLC2"]),
    ("LC10a silenced", ["LC10a"]),
    ("all LC10 silenced", ["LC10"]),
    ("AOTU019+AOTU025 silenced", ["AOTU019", "AOTU025"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--az", type=float, action="append")
    args = ap.parse_args()
    azs = args.az or [30.0, -30.0, 40.0, -40.0]

    c = Connectome()
    ns = argparse.Namespace(**MODEL_ARGS)
    h = Hybrid(c, ns)
    B0, w0 = h.B.copy(), h.sp_w.copy()
    tix = {t: i for i, t in enumerate(c.types)}

    def cells(name):
        if name in c.meta["groups"]:
            return c.group(name)
        return np.flatnonzero(c.type_id == tix[name])

    # Wiring: share of each cell type's input synapses that come from each candidate.
    src = np.repeat(np.arange(c.n), np.diff(c.offsets))
    total_in = np.bincount(c.tg, weights=c.w_count, minlength=c.n)
    print("share of input synapses:")
    for post in ["AOTU019", "AOTU025", "DNa02"]:
        into = c.type_id[c.tg] == tix[post]
        tot = total_in[cells(post)].sum()
        shares = [f"{pre} {c.w_count[into & np.isin(src, cells(pre))].sum() / tot:.1%}" for pre in ["LC10a", "LC10", "LC4", "LPLC2", "AOTU019", "AOTU025"]]
        print(f"  onto {post:8s} ({c.nt[cells(post)[0]]}): " + ", ".join(shares))
    for pre in ["AOTU019", "AOTU025"]:
        m = np.isin(src, cells(pre)) & (c.type_id[c.tg] == tix["DNa02"])
        same = c.side[src[m]] == c.side[c.tg[m]]
        print(f"  {pre} -> DNa02: {c.w_count[m][same].sum():.0f} synapses same side, {c.w_count[m][~same].sum():.0f} opposite side")

    def lr(rates, name):
        i = cells(name)
        return rates[i[c.side[i] == 1]].mean(), rates[i[c.side[i] == 2]].mean()

    for label, names in CONDITIONS:
        idx = np.concatenate([cells(t) for t in names]) if names else np.array([], dtype=np.int64)
        assert not h.graded[idx].any()
        mask = np.ones(c.n)
        mask[idx] = 0
        h.B = sp.diags(mask) @ B0
        h.sp_w = w0.copy()
        for i in idx:
            h.sp_w[h.sp_off[i]:h.sp_off[i + 1]] = 0

        print(f"\n=== {label} ===", flush=True)
        ns.closed_loop = False
        for az in azs:
            rates, _, _ = run(h, ns, az)
            dl, dr = lr(rates, "DNa02")
            cols = "  ".join(f"{t} {l:5.1f}/{r:<5.1f}" for t in SHOW for l, r in [lr(rates, t)])
            print(f"target {az:+.0f}: {cols}  yaw {ns.yaw_gain * (dr - dl):+6.0f} deg/s  {rates.sum():7.0f} spikes/s", flush=True)
        ns.closed_loop = True
        run(h, ns, 50.0)


if __name__ == "__main__":
    main()
