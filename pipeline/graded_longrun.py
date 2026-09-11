"""Long run of the graded optic lobe alone, to catch slow instabilities.

Integrates the browser's graded update (exponential Euler, clamp [-1, amax]) with a
static target for minutes of simulated time and prints activity statistics every
few seconds, plus which cell types sit at the clamp.

  python pipeline/graded_longrun.py --gain 2.3 --seconds 150
"""

import argparse

import numpy as np

from sim_hybrid import Hybrid, column_contrast
from sim_reference import Connectome, eye_rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, default=2.3)
    ap.add_argument("--dt", type=float, default=8.33)
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--seconds", type=float, default=150.0)
    ap.add_argument("--amax", type=float, default=4.0)
    ap.add_argument("--az", type=float, default=30.0)
    ap.add_argument("--el", type=float, default=10.0)
    ap.add_argument("--noise", type=float, default=0.0, help="per-step gaussian input noise (e.g. 1e-6)")
    ap.add_argument("--report", type=float, default=5.0)
    args = ap.parse_args()
    ns = argparse.Namespace(prune=0.005, ifscale=1.0, mod_zero=1, radius=16.0, bg=1.0, fg=0.05, cgain=1.0, r0=20.0,
                            input="contrast", w_lam=0.5)
    c = Connectome()
    h = Hybrid(c, ns)
    ng = h.ng
    a = np.zeros(ng, dtype=np.float32)
    contrast = (eye_rates(c, ns, args.az, args.el) / ns.r0 - 1.0).astype(np.float32)
    lam_in = np.zeros(ng, dtype=np.float32)
    lam_in[h.lam_g] = -ns.w_lam * column_contrast(c, h.lam, ns, args.az, args.el)
    A = h.A.astype(np.float32)
    decay = np.float32(np.exp(-args.dt / args.tau))
    tn = c.types[c.type_id[np.flatnonzero(h.graded)]]
    rng = np.random.default_rng(0)
    steps = int(args.seconds * 1000 / args.dt)
    every = int(args.report * 1000 / args.dt)
    for s in range(steps):
        a[h.eye_g] = contrast
        out = np.clip(a, -1.0, args.amax)
        goal = args.gain * (A @ out) + lam_in
        if args.noise:
            goal += rng.normal(0, args.noise, ng).astype(np.float32)
        a = goal + (a - goal) * decay
        a[h.eye_g] = contrast
        if s % every == 0 or s == steps - 1:
            sat = np.flatnonzero(a >= args.amax * 0.99)
            types, counts = np.unique(tn[sat], return_counts=True)
            top = ", ".join(f"{t}:{n}" for t, n in sorted(zip(types, counts), key=lambda x: -x[1])[:6])
            print(f"t={s * args.dt / 1000:6.1f}s  mean|a| {np.abs(a).mean():.4f}  max {a.max():+.3f}  "
                  f"at clamp {len(sat):5d}  [{top}]", flush=True)


if __name__ == "__main__":
    main()
