"""Hybrid model: graded optic lobe + spiking (Shiu LIF) visual projection neurons and central brain.

Optic lobe intrinsic/sensory neurons are non-spiking in the fly. Here they are
rate units around a tonic operating point:
    tau_g da/dt = -a + gain * sum_j frac_ij * sign_j * clamp(a_j, -1, amax)
where frac_ij is the input fraction (synapses from j / all synapses onto i).
Photoreceptor activity is clamped to local contrast. Deviations of graded activity
from rest drive spiking neurons like a release rate of r0 * a (r0 = MODEL.coupling).
The defaults match the browser model (web/src/config.ts).

  python pipeline/sim_hybrid.py --az 40 --az -40
"""

import argparse
import time

import numpy as np
import scipy.sparse as sp

from sim_reference import (PATH_TYPES, T_DLY, T_RFC, TAU_M, TAU_S, V0, VRST, VTH, W_SYN, Connectome, angular_dist,
                           eye_rates, type_side_rates)

GRADED_SC = ("ol_sensory", "ol_intrinsic")
SUMMARY_TYPES = ["LC10a", "LC4", "LPLC2", "AOTU019", "DNa01", "DNa02", "DNp01", "pIP10"]
POI_JUMP = W_SYN * 250


class Hybrid:
    def __init__(self, c, args):
        self.c = c
        n = c.n
        sc = np.array(c.meta["superclasses"])[c.sc_id]
        self.graded = np.isin(sc, GRADED_SC)
        ng = int(self.graded.sum())
        gi = -np.ones(n, dtype=np.int64)
        gi[self.graded] = np.arange(ng)
        self.gi, self.ng = gi, ng

        deg = np.diff(c.offsets)
        src = np.repeat(np.arange(n), deg)
        dst = c.tg.astype(np.int64)
        sign = np.sign(c.w_signed)
        total_in = np.bincount(dst, weights=c.w_count, minlength=n)
        frac = c.w_count / np.maximum(total_in[dst], 1) * sign

        # Lamina: the reconstruction misses most R1-6 -> L1-3 synapses, so those edges are
        # replaced by a per-column synthetic photoreceptor input (see run()).
        tnames = c.types[c.type_id]
        self.lam = np.flatnonzero(np.isin(tnames, ["L1", "L2", "L3"]) & ~np.isnan(c.col[:, 0]))
        is_r16 = tnames == "R1-R6"
        is_lam = np.zeros(n, dtype=bool)
        is_lam[self.lam] = True
        m = self.graded[src] & self.graded[dst] & ~(is_r16[src] & is_lam[dst]) & (np.abs(frac) >= args.prune)
        self.A = sp.csr_matrix((frac[m], (gi[dst[m]], gi[src[m]])), shape=(ng, ng))
        self.lam_g = gi[self.lam]
        m2 = self.graded[src] & ~self.graded[dst]
        self.B = sp.csr_matrix((c.w_signed[m2] * args.ifscale * W_SYN * TAU_S / 1000, (dst[m2], gi[src[m2]])), shape=(n, ng))
        # Event-driven spiking edges only from spiking neurons to spiking neurons.
        keep = ~self.graded[src] & ~self.graded[dst]
        self.sp_tg = dst[keep]
        self.sp_w = c.w_signed[keep].copy()
        if args.mod_zero:
            # Dopamine/serotonin/octopamine act through GPCRs: no fast synaptic effect.
            self.sp_w[np.isin(c.nt, ["dopamine", "serotonin", "octopamine"])[src[keep]]] = 0
        self.sp_off = np.concatenate([[0], np.cumsum(np.bincount(src[keep], minlength=n))])
        eye_g = gi[: c.meta["eye"]["count"]]
        assert (eye_g >= 0).all()
        self.eye_g = eye_g
        print(f"graded {ng} neurons, {self.A.nnz} graded edges, {self.B.nnz} graded->spiking, {len(self.sp_tg)} spiking edges")


def column_contrast(c, idx, args, target_az, target_el=0.0):
    """Luminance contrast (vs. that eye's mean) seen by the eye column of each neuron in idx."""
    lum = np.full(len(idx), args.bg)
    if target_az is not None:
        d = angular_dist(c.col[idx, 0], c.col[idx, 1], target_az, target_el)
        lum[d < args.radius] = args.fg
    out = np.zeros(len(idx))
    for s in (1, 2):
        m = c.side[idx] == s
        if m.any():
            mean = max(lum[m].mean(), 1e-3)
            out[m] = np.clip(args.cgain * (lum[m] - mean) / mean, -1, 3)
    return out


def run(h, args, target_az, target_el=0.0):
    c = h.c
    n, ng = c.n, h.ng
    rng = np.random.default_rng(args.seed)
    dt = args.dt
    D = max(1, round(T_DLY / dt))
    em, es = np.exp(-dt / TAU_M), np.exp(-dt / TAU_S)
    kg = TAU_S / (TAU_S - TAU_M) * (es - em)
    gsub = max(1, round(args.dt_g / dt))

    x = np.zeros(n)
    g = np.zeros(n)
    refr = np.zeros(n)
    theta = np.zeros(n)
    ea = np.exp(-dt / args.tau_adapt)
    ring = np.zeros((D + 1, n))
    a = np.zeros(ng)
    rate = np.zeros(n)
    for spec in (args.stim or []):
        grp, sd, hz = spec.split(":")
        idx = c.group(grp)
        if sd != "B":
            idx = idx[c.side[idx] == {"L": 1, "R": 2}[sd]]
        rate[idx] = float(hz)
    base_drive = np.zeros(n)
    base_drive[c.group("P1_pC1")] += args.arousal
    spiking = ~h.graded

    ne = c.meta["eye"]["count"]
    contrast = eye_rates(c, args, target_az, target_el) / args.r0 - 1.0  # eye_rates returns r0*(1+cgain*C)
    a[h.eye_g] = contrast
    lam_in = np.zeros(ng)
    lam_in[h.lam_g] = -args.w_lam * column_contrast(c, h.lam, args, target_az, target_el)

    steps = int(args.seconds * 1000 / dt)
    warm = int(args.warmup * 1000 / dt)
    counts = np.zeros(n, dtype=np.int64)
    a_acc = np.zeros(ng)
    drive = base_drive.copy()
    # Closed loop: heading turns by yaw_gain * (DNa02_R - DNa02_L) (deg/s per Hz).
    dna02 = c.group("DNa02")
    dna02_l, dna02_r = dna02[c.side[dna02] == 1], dna02[c.side[dna02] == 2]
    rate_l = rate_r = 0.0
    heading = 0.0
    trace = []
    t0 = time.time()
    for s in range(steps + warm):
        if args.closed_loop and s % gsub == 0 and target_az is not None:
            rel = ((target_az - heading + 180) % 360) - 180
            contrast = eye_rates(c, args, rel) / args.r0 - 1.0
            a[h.eye_g] = contrast
            lam_in[h.lam_g] = -args.w_lam * column_contrast(c, h.lam, args, rel)
            if s >= warm:
                heading += args.yaw_gain * (rate_r - rate_l) * (gsub * dt) / 1000
        if args.closed_loop and target_az is not None and s % int(50 / dt) == 0:
            trace.append(f"{((target_az - heading + 180) % 360) - 180:+.0f}({rate_l:.0f}/{rate_r:.0f})")
        if s % gsub == 0:
            out = np.clip(a, -1.0, args.amax)
            target = args.gain * (h.A @ out) + lam_in
            if args.exp_euler:
                a = target + (a - target) * np.exp(-args.dt_g / args.tau_g)
            else:
                a += (args.dt_g / args.tau_g) * (target - a)
            a[h.eye_g] = contrast
            # Only deviations from resting release reach spiking cells (resting release is
            # absorbed into their resting potential), so visual projection neurons idle at rest.
            drive = base_drive + args.r0 * (h.B @ np.clip(a, -1.0, args.amax))
            if s >= warm:
                a_acc += a
        inp = ring[s % (D + 1)]
        active = refr > 0
        g_new = g * es
        x_new = drive + (x - drive) * em + g * kg
        x = np.where(active, VRST - V0, x_new)
        g = np.where(active, g, g_new) + W_SYN * inp
        inp[:] = 0
        refr = np.maximum(refr - dt, 0)
        theta *= ea
        if args.stim:
            x = x + POI_JUMP * ((rng.random(n) < rate * dt / 1000) & ~active)
        spk = np.flatnonzero((x > VTH - V0 + theta) & ~active & spiking)
        decay = np.exp(-dt / args.tau_rate)
        rate_l *= decay
        rate_r *= decay
        if len(spk):
            rate_l += np.isin(spk, dna02_l).sum() * 1000 / args.tau_rate / len(dna02_l)
            rate_r += np.isin(spk, dna02_r).sum() * 1000 / args.tau_rate / len(dna02_r)
            x[spk] = VRST - V0
            g[spk] = 0
            refr[spk] = T_RFC
            theta[spk] += args.adapt
            if s >= warm:
                counts[spk] += 1
            starts, ends = h.sp_off[spk], h.sp_off[spk + 1]
            lens = ends - starts
            if lens.sum():
                idx = np.repeat(ends - lens.cumsum(), lens) + np.arange(lens.sum())
                ring[(s + D) % (D + 1)] += np.bincount(h.sp_tg[idx], weights=h.sp_w[idx], minlength=n)
    a_mean = a_acc / max(1, (steps // gsub))
    if args.closed_loop:
        print(f"closed loop target={target_az:+.0f}: rel az (DNa02 L/R Hz) every 50ms: " + " ".join(trace))
    return counts / args.seconds, a_mean, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--az", type=float, action="append")
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--warmup", type=float, default=0.2)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--dt-g", type=float, default=8.33)
    ap.add_argument("--tau-g", type=float, default=20.0)
    ap.add_argument("--gain", type=float, default=2.3)
    ap.add_argument("--amax", type=float, default=4.0)
    ap.add_argument("--r0", type=float, default=3400.0, help="graded -> spiking release scale (Hz-equivalent), MODEL.coupling")
    ap.add_argument("--ifscale", type=float, default=1.0, help="scale of graded->spiking synapses")
    ap.add_argument("--radius", type=float, default=19.0)
    ap.add_argument("--bg", type=float, default=1.0)
    ap.add_argument("--fg", type=float, default=0.05)
    ap.add_argument("--cgain", type=float, default=1.0)
    ap.add_argument("--w-lam", type=float, default=0.5, help="weight of synthetic photoreceptor input to L1-3")
    ap.add_argument("--adapt", type=float, default=2.0, help="spike threshold increment (mV)")
    ap.add_argument("--tau-adapt", type=float, default=200.0)
    ap.add_argument("--mod-zero", type=int, default=1)
    ap.add_argument("--stim", action="append", help="GROUP:SIDE:RATE extra Poisson drive")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--exp-euler", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prune", type=float, default=0.005, help="drop graded edges with input fraction below this")
    ap.add_argument("--closed-loop", action="store_true")
    ap.add_argument("--yaw-gain", type=float, default=10.0, help="deg/s of turning per Hz of DNa02 R-L difference")
    ap.add_argument("--tau-rate", type=float, default=50.0, help="ms, DNa02 rate filter")
    ap.add_argument("--arousal", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    args.input = "contrast"
    print(vars(args))

    c = Connectome()
    h = Hybrid(c, args)
    azs = args.az or [40.0, -40.0]
    tix = {t: i for i, t in enumerate(c.types)}
    results = []
    for az in azs:
        rates, a_mean, elapsed = run(h, args, az)
        sums, cnt = type_side_rates(c, rates)
        results.append((az, sums / np.maximum(cnt, 1), rates, a_mean))
        print(f"az={az:+.0f}: {elapsed:.1f}s wall, {rates.sum():.0f} spikes/s, spiking active {np.mean(rates[~h.graded] > 0):.1%}, "
              f"graded a: mean {a_mean.mean():+.3f} min {a_mean.min():+.2f} max {a_mean.max():+.2f}")

    print(f"\n{'az':>5s} " + " ".join(f"{t:>13s}" for t in SUMMARY_TYPES))
    for az, mean, _, _ in results:
        cells = []
        for t in SUMMARY_TYPES:
            i = tix[t]
            cells.append(f"{mean[i * 3 + 1]:5.1f}/{mean[i * 3 + 2]:<5.1f}")
        print(f"{az:+5.0f} " + " ".join(f"{x:>13s}" for x in cells))
    print("(L/R rates in Hz)")
    if len(results) < 2 or not args.detail:
        return

    # Retinotopy check on graded types: mean activity of cells whose connectome RF is near the target
    # minus cells whose RF is in the mirrored position.
    print("\ngraded retinotopy (a near target - a at mirror position), per type:")
    gidx = np.flatnonzero(h.graded)
    pos_az = np.where(np.isnan(c.col[gidx, 0]), c.rf_az[gidx], c.col[gidx, 0])
    pos_el = np.where(np.isnan(c.col[gidx, 1]), c.rf_el[gidx], c.col[gidx, 1])
    row = []
    for t in ["R1-R6", "L1", "L2", "L3", "Mi1", "Tm1", "Tm2", "Tm3", "Tm9", "Tm20", "T2", "T3", "T4a", "T5a", "Tm5Y", "TmY21", "Tm5a", "Li22"]:
        if t not in tix:
            continue
        m = c.type_id[gidx] == tix[t]
        vals = []
        for az, _, _, a_mean in results[:2]:
            near = m & (angular_dist(pos_az, pos_el, az, 0) < 20)
            mir = m & (angular_dist(pos_az, pos_el, -az, 0) < 20)
            vals.append(a_mean[near].mean() - a_mean[mir].mean() if near.any() and mir.any() else np.nan)
        row.append(f"{t}({vals[0]:+.3f})")
    print("  " + ", ".join(row))

    a, b = results[0][1], results[1][1]
    fs = 2 if azs[0] > 0 else 1
    os_ = 3 - fs
    print(f"\n{'type':10s} {'ipsi(c1)':>9s} {'ipsi(c2)':>9s} {'contra(c1)':>10s} {'contra(c2)':>10s} {'LI':>7s}")
    for t in ["LC10a", "LC10d", "LC4", "LPLC2", "LC9", "LC11", "AOTU019", "AOTU012", "DNa01", "DNa02", "pIP10", "DNp01"]:
        if t not in tix:
            continue
        i = tix[t]
        li = (a[i * 3 + fs] - b[i * 3 + fs]) - (a[i * 3 + os_] - b[i * 3 + os_])
        print(f"{t:10s} {a[i*3+fs]:9.1f} {b[i*3+fs]:9.1f} {a[i*3+os_]:10.1f} {b[i*3+os_]:10.1f} {li:+7.1f}")


if __name__ == "__main__":
    main()
