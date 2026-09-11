"""CPU reference of the browser LIF model (same data files, same update rule).

Used to sanity-check the packed format and tune input/gain parameters before
touching WebGPU. Example:
  python pipeline/sim_reference.py --az 40 --az -40 --seconds 1
Condition 1 vs 2 lateralisation per cell type is printed: LI > 0 means the
neurons on the side of the first target got relatively more active.
"""

import argparse
import gzip
import json
import os
import time

import numpy as np

DATA = os.environ.get("OUT", "web/public/data/malecns-v1")

# Shiu et al. 2024 parameters (mV, ms)
V0, VRST, VTH = -52.0, -52.0, -45.0
TAU_M, TAU_S, T_RFC, T_DLY = 20.0, 5.0, 2.2, 1.8
W_SYN, POI_JUMP = 0.275, 0.275 * 250

PATH_TYPES = ["R1-R6", "L1", "L2", "L3", "Mi1", "Tm1", "Tm2", "Tm9", "Tm20", "T2", "T3", "T4a", "T5a",
              "LC10a", "LC10b", "LC10d", "LC4", "LPLC2", "LC9", "LC11", "DNa01", "DNa02", "pIP10", "DNp01"]


def decode_varints(buf, count):
    b = np.frombuffer(buf, dtype=np.uint8)
    ends = np.flatnonzero(b < 0x80)[:count]
    last = ends[-1] + 1
    b = b[:last]
    group = np.zeros(last, dtype=np.int64)
    group[ends[:-1] + 1] = 1
    group = np.cumsum(group)
    starts = np.concatenate([[0], ends[:-1] + 1])
    shift = (np.arange(last) - starts[group]) * 7
    vals = np.zeros(count, dtype=np.int64)
    np.add.at(vals, group, (b & 0x7F).astype(np.int64) << shift)
    return vals, last


class Connectome:
    def __init__(self):
        meta = json.load(open(f"{DATA}/meta.json"))
        n = meta["neurons"]
        nb = gzip.open(f"{DATA}/neurons.bin.gz").read()
        self.meta, self.n = meta, n
        self.type_id = np.frombuffer(nb, dtype="<u2", count=n, offset=8 * n)
        self.sc_id = np.frombuffer(nb, dtype=np.uint8, count=n, offset=10 * n)
        self.side = np.frombuffer(nb, dtype=np.uint8, count=n, offset=11 * n)
        sign = np.frombuffer(nb, dtype=np.int8, count=n, offset=12 * n)
        self.types = np.array(json.load(open(f"{DATA}/types.json")))
        rf = np.frombuffer(nb, dtype="<f4", count=3 * n, offset=28 * n).reshape(n, 3)
        self.rf_az, self.rf_el, self.rf_str = rf[:, 0], rf[:, 1], rf[:, 2]
        self.col = np.frombuffer(nb, dtype="<f4", count=2 * n, offset=40 * n).reshape(n, 2)
        self.nt = np.array(meta["transmitters"])[np.frombuffer(nb, dtype=np.uint8, count=n, offset=48 * n)]

        degrees, targets, weights = [], [], []
        for sh in meta["shards"]:
            buf = gzip.open(f"{DATA}/{sh['file']}").read()
            a, b, ne, dlen, wlen = np.frombuffer(buf[:20], dtype="<u4")
            deg, used = decode_varints(buf[20:], b - a)
            p = 20 + used
            dlt, _ = decode_varints(buf[p:p + dlen], ne)
            wt, _ = decode_varints(buf[p + dlen:p + dlen + wlen], ne)
            off = np.concatenate([[0], np.cumsum(deg)])
            cs = np.cumsum(dlt)  # targets are delta-coded within each row
            row_base = np.where(off[:-1] > 0, cs[np.maximum(off[:-1] - 1, 0)], 0)
            targets.append((cs - np.repeat(row_base, deg)).astype(np.int32))
            degrees.append(deg)
            weights.append(wt)
        deg = np.concatenate(degrees)
        self.offsets = np.concatenate([[0], np.cumsum(deg)])
        self.tg = np.concatenate(targets)
        src = np.repeat(np.arange(n), deg)
        self.w_count = np.concatenate(weights).astype(np.float64)
        self.w_signed = self.w_count * sign[src]

        er = gzip.open(f"{DATA}/eye.bin.gz").read()
        ne = meta["eye"]["count"]
        self.eye_dir = np.frombuffer(er, dtype="<f4", count=2 * ne).reshape(ne, 2)
        self.eye_side = self.side[:ne].copy()

    def group(self, name):
        return np.array(self.meta["groups"][name]["indices"], dtype=np.int64)

    def weights(self, norm, wscale):
        w = self.w_signed
        if norm > 0:
            total_in = np.bincount(self.tg, weights=self.w_count, minlength=self.n)
            w = w * norm / np.maximum(total_in[self.tg], 1.0)
        return w * wscale


def angular_dist(az1, el1, az2, el2):
    a1, e1, a2, e2 = map(np.radians, (az1, el1, az2, el2))
    c = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def eye_rates(c, args, target_az, target_el=0.0):
    ne = c.meta["eye"]["count"]
    lum = np.full(ne, args.bg)
    if target_az is not None:
        d = angular_dist(c.eye_dir[:, 0], c.eye_dir[:, 1], target_az, target_el)
        lum[d < args.radius] = args.fg
    if args.input == "lum":
        return args.r0 * lum
    # Contrast coding against each eye's mean luminance (fast adaptation stand-in).
    out = np.zeros(ne)
    for s in (1, 2):
        m = c.eye_side == s
        mean = max(lum[m].mean(), 1e-3)
        out[m] = args.r0 * np.clip(1 + args.cgain * (lum[m] - mean) / mean, 0, 4)
    return out


def run(c, w, args, target_az):
    n = c.n
    rng = np.random.default_rng(args.seed)
    dt = args.dt
    D = max(1, round(T_DLY / dt))
    em, es = np.exp(-dt / TAU_M), np.exp(-dt / TAU_S)
    kg = TAU_S / (TAU_S - TAU_M) * (es - em)
    ea = np.exp(-dt / args.tau_adapt)

    x = np.zeros(n)  # v - V0
    g = np.zeros(n)
    theta = np.zeros(n)  # adaptive threshold offset
    refr = np.zeros(n)
    ring = np.zeros((D + 1, n))
    drive = np.full(n, args.drive_all)
    for name in ["L1", "L2", "L3"]:
        drive[c.group(name)] = args.lamina_drive
    drive[c.group("P1_pC1")] += args.arousal

    rate = np.zeros(n)
    for spec in (args.stim or []):
        # GROUP:SIDE:RATE, SIDE in L/R/B; "{side}" in a condition label is substituted by main()
        grp, sd, hz = spec.split(":")
        idx = c.group(grp)
        if sd != "B":
            idx = idx[c.side[idx] == {"L": 1, "R": 2}[sd]]
        rate[idx] = float(hz)
    ne = c.meta["eye"]["count"]
    if target_az is None and args.no_eye:
        r_eye = np.zeros(ne)
    else:
        r_eye = eye_rates(c, args, target_az)
    if args.graded_eye:
        # Photoreceptors are non-spiking: replace their Poisson spikes by the mean
        # synaptic potential they would cause (rate * weight * tau_s) as tonic drive.
        e_end = c.offsets[ne]
        src = np.repeat(np.arange(ne), np.diff(c.offsets[: ne + 1]))
        drive += np.bincount(c.tg[:e_end], weights=w[:e_end] * r_eye[src] * W_SYN * TAU_S / 1000, minlength=n)
    else:
        rate[:ne] = np.maximum(rate[:ne], r_eye)

    steps = int(args.seconds * 1000 / dt)
    warm = int(args.warmup * 1000 / dt)
    counts = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for s in range(steps + warm):
        inp = ring[s % (D + 1)]
        active = refr > 0
        g_new = g * es
        x_new = drive + (x - drive) * em + g * kg
        x = np.where(active, VRST - V0, x_new)
        g = np.where(active, g, g_new) + W_SYN * inp
        inp[:] = 0
        refr = np.maximum(refr - dt, 0)
        theta *= ea
        poisson = (rng.random(n) < rate * dt / 1000) & ~active
        x = x + POI_JUMP * poisson
        spk = np.flatnonzero((x > VTH - V0 + theta) & ~active)
        if len(spk):
            x[spk] = VRST - V0
            g[spk] = 0
            refr[spk] = T_RFC
            theta[spk] += args.adapt
            if s >= warm:
                counts[spk] += 1
            starts, ends = c.offsets[spk], c.offsets[spk + 1]
            lens = ends - starts
            if lens.sum():
                idx = np.repeat(ends - lens.cumsum(), lens) + np.arange(lens.sum())
                ring[(s + D) % (D + 1)] += np.bincount(c.tg[idx], weights=w[idx], minlength=n)
    return counts / args.seconds, time.time() - t0


def type_side_rates(c, rates):
    key =c.type_id.astype(np.int64) * 3 + c.side
    sums = np.bincount(key, weights=rates, minlength=len(c.types) * 3)
    cnt = np.bincount(key, minlength=len(c.types) * 3)
    return sums, cnt


def decode_test(c, w, args):
    """Population-vector readout of target azimuth from connectome receptive fields."""
    azs = args.az or [-60.0, -30.0, 0.0, 30.0, 60.0]
    base, _ = run(c, w, args, None)
    sc = np.array(c.meta["superclasses"])[c.sc_id]
    tnames = c.types[c.type_id]
    pops = {
        "LC10a": tnames == "LC10a",
        "LC*": np.char.startswith(tnames.astype(str), "LC"),
        "visual_projection": sc == "visual_projection",
        "ol_intrinsic": sc == "ol_intrinsic",
    }
    ux = np.cos(np.radians(c.rf_el)) * np.sin(np.radians(c.rf_az))
    uz = np.cos(np.radians(c.rf_el)) * np.cos(np.radians(c.rf_az))
    print(f"baseline {base.sum():.0f} spikes/s")
    print(f"{'target':>7s} " + " ".join(f"{p:>22s}" for p in pops))
    for az in azs:
        rates, _ = run(c, w, args, az)
        d = rates - base
        cells = []
        for m in pops.values():
            for sgn_label, val in [("+", d), ("|", np.abs(d))]:
                wv = val[m] * c.rf_str[m]
                pred = np.degrees(np.arctan2((wv * ux[m]).sum(), (wv * uz[m]).sum()))
                cells.append(f"{sgn_label}{pred:+5.0f}")
            cells[-2:] = [f"{cells[-2]} {cells[-1]} n={int((np.abs(d[m]) > 0).sum()):5d}"]
        print(f"{az:+7.0f} " + " ".join(f"{x:>22s}" for x in cells))


def side_test(c, w, args):
    """Drive a group on the left vs on the right and report lateralised responses."""
    import copy

    res = {}
    for s in ["L", "R"]:
        a2 = copy.copy(args)
        a2.stim = [spec.replace(":X:", f":{s}:") for spec in args.stim]
        rates, elapsed = run(c, w, a2, None)
        sums, cnt = type_side_rates(c, rates)
        res[s] = sums / np.maximum(cnt, 1)
        print(f"stim {s}: {elapsed:.1f}s wall, {rates.sum():.0f} spikes/s, active {np.mean(rates > 0):.2%}")
    tix = {t: i for i, t in enumerate(c.types)}
    print(f"{'type':10s} {'L|stimL':>8s} {'R|stimL':>8s} {'L|stimR':>8s} {'R|stimR':>8s} {'ipsiLI':>7s}")
    for t in ["LC10a", "AOTU008", "AOTU019", "AOTU012", "DNa01", "DNa02", "DNa03", "DNa04", "pIP10", "DNp01", "DNb01", "pC1_1a"]:
        if t not in tix:
            continue
        i = tix[t]
        lL, rL, lR, rR = res["L"][i * 3 + 1], res["L"][i * 3 + 2], res["R"][i * 3 + 1], res["R"][i * 3 + 2]
        print(f"{t:10s} {lL:8.1f} {rL:8.1f} {lR:8.1f} {rR:8.1f} {(lL - lR) + (rR - rL):+7.1f}")
    lis = np.array([(res["L"][i * 3 + 1] - res["R"][i * 3 + 1]) + (res["R"][i * 3 + 2] - res["L"][i * 3 + 2]) for i in range(len(c.types))])
    act = np.array([max(res["L"][i * 3 + 1], res["L"][i * 3 + 2], res["R"][i * 3 + 1], res["R"][i * 3 + 2]) for i in range(len(c.types))])
    dn = set(c.types[c.type_id[c.sc_id == c.meta["superclasses"].index("descending_neuron")]])
    order = [i for i in np.argsort(-np.abs(lis)) if c.types[i] in dn][:15]
    print("most lateralised DNs (ipsiLI, max rate): " + ", ".join(f"{c.types[i]}({lis[i]:+.0f},{act[i]:.0f}Hz)" for i in order))
    order = [i for i in np.argsort(-act) if c.types[i] in dn][:10]
    print("most active DNs: " + ", ".join(f"{c.types[i]}({act[i]:.0f}Hz, LI {lis[i]:+.0f})" for i in order))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--az", type=float, action="append", help="target azimuth (deg, + = right)")
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--radius", type=float, default=15.0)
    ap.add_argument("--bg", type=float, default=1.0)
    ap.add_argument("--fg", type=float, default=0.05)
    ap.add_argument("--input", choices=["lum", "contrast"], default="contrast")
    ap.add_argument("--r0", type=float, default=20.0, help="photoreceptor rate at mean luminance (Hz)")
    ap.add_argument("--cgain", type=float, default=1.0)
    ap.add_argument("--lamina-drive", type=float, default=9.0)
    ap.add_argument("--arousal", type=float, default=0.0)
    ap.add_argument("--drive-all", type=float, default=0.0, help="tonic mV added to every neuron")
    ap.add_argument("--norm", type=float, default=0.0, help="normalise each neuron's total input to this many synapses")
    ap.add_argument("--wscale", type=float, default=1.0)
    ap.add_argument("--adapt", type=float, default=0.0, help="threshold increment per spike (mV)")
    ap.add_argument("--tau-adapt", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--decode", action="store_true", help="population-vector azimuth decoding test")
    ap.add_argument("--graded-eye", action="store_true", help="photoreceptors as graded (mean-field) input")
    ap.add_argument("--stim", action="append", help="GROUP:SIDE:RATE Poisson drive; SIDE L/R/B or X (= side under test)")
    ap.add_argument("--no-eye", action="store_true", help="no photoreceptor input at all")
    ap.add_argument("--sidetest", action="store_true", help="compare --stim with X=L vs X=R instead of target azimuths")
    args = ap.parse_args()
    print(vars(args))

    c = Connectome()
    w = c.weights(args.norm, args.wscale)
    if args.decode:
        decode_test(c, w, args)
        return
    if args.sidetest:
        side_test(c, w, args)
        return
    azs = args.az or [40.0, -40.0]
    res = []
    for az in azs:
        rates, elapsed = run(c, w, args, az)
        sums, cnt = type_side_rates(c, rates)
        mean = sums / np.maximum(cnt, 1)
        res.append(mean)
        by_sc = {c.meta["superclasses"][i]: round(float(rates[c.sc_id == i].mean()), 1) for i in np.unique(c.sc_id)
                 if c.meta["superclasses"][i] in ("ol_intrinsic", "visual_projection", "cb_intrinsic", "descending_neuron", "vnc_intrinsic")}
        print(f"az={az:+.0f}: {elapsed:.1f}s wall, {rates.sum():.0f} spikes/s, active {np.mean(rates > 0):.1%}, {by_sc}")

    a, b = res[0], res[1]
    first_side = 2 if azs[0] > 0 else 1
    other_side = 3 - first_side
    tix = {t: i for i, t in enumerate(c.types)}

    def li(t):
        i = tix[t]
        same = a[i * 3 + first_side] - b[i * 3 + first_side]
        opp = a[i * 3 + other_side] - b[i * 3 + other_side]
        return same - opp

    print(f"\n{'type':10s} {'ipsi(c1)':>9s} {'ipsi(c2)':>9s} {'contra(c1)':>10s} {'contra(c2)':>10s} {'LI':>7s}")
    for t in PATH_TYPES:
        if t not in tix:
            continue
        i = tix[t]
        print(f"{t:10s} {a[i*3+first_side]:9.1f} {b[i*3+first_side]:9.1f} {a[i*3+other_side]:10.1f} {b[i*3+other_side]:10.1f} {li(t):+7.1f}")

    lis = np.array([li(t) for t in c.types])
    dn_types = set(c.types[c.type_id[c.sc_id == c.meta["superclasses"].index("descending_neuron")]])
    vpn_types = set(c.types[c.type_id[c.sc_id == c.meta["superclasses"].index("visual_projection")]])
    for label, pool in [("visual projection", vpn_types), ("descending", dn_types)]:
        idx = [i for i, t in enumerate(c.types) if t in pool]
        idx = sorted(idx, key=lambda i: -abs(lis[i]))[: args.top // 2]
        print(f"\nmost lateralised {label} types: " + ", ".join(f"{c.types[i]}({lis[i]:+.1f})" for i in idx))


if __name__ == "__main__":
    main()
