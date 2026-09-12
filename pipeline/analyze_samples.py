"""Bin in-game samples (web ?bench=1&record=1, dumped by scripts/smoke.mjs) by target position.

Columns: relAz, relEl, msSinceSpawn, DNp53 L, DNp53 R, DNp01 L, DNp01 R, LC4+LPLC2 L, LC4+LPLC2 R,
DNa02 L, DNa02 R, pIP10 L+R, then (newer recordings) DNa01 L, DNa01 R, LC10a L, LC10a R, AOTU019 L,
AOTU019 R, brain time (ms), sight (Hz per LC10 sight cell).

  python pipeline/analyze_samples.py .cache/smoke/samples.json --keep 0.75
"""

import argparse
import json

import numpy as np

COLS = ["az", "el", "since", "p53L", "p53R", "gfL", "gfR", "lcL", "lcR", "dnaL", "dnaR", "song"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--keep", type=float, default=1.0, help="fraction of samples to keep from the start")
    ap.add_argument("--settle", type=float, default=300.0, help="ms after spawn to ignore")
    ap.add_argument("--sight", type=float, default=3.0, help="sight threshold (Hz per cell) for the gate statistics")
    args = ap.parse_args()
    s = np.array(json.load(open(args.path)), dtype=np.float64)
    s = s[: int(len(s) * args.keep)]
    d = {k: s[:, i] for i, k in enumerate(COLS)}
    m = d["since"] > args.settle
    print(f"{m.sum()} settled samples of {len(s)}")

    p53 = (d["p53L"] + d["p53R"]) / 2
    gf = d["gfL"] + d["gfR"]
    lc = d["lcL"] + d["lcR"]
    front = m & (np.abs(d["az"]) < 25)
    print("\nby elevation (|az| < 25):")
    print(f"{'el bin':>12s} {'n':>5s} {'DNp53':>7s} {'GF':>7s} {'LC4+LPLC2':>10s} {'song':>6s} {'p53/GF':>7s} {'p53/LC':>7s}")
    for lo in range(-35, 35, 10):
        b = front & (d["el"] >= lo) & (d["el"] < lo + 10)
        if b.sum() < 5:
            continue
        print(f"{lo:+4d}..{lo + 10:+4d}   {b.sum():5d} {p53[b].mean():7.2f} {gf[b].mean():7.1f} {lc[b].mean():10.1f} "
              f"{d['song'][b].mean():6.1f} {p53[b].mean() / max(gf[b].mean(), 1e-3):7.3f} {p53[b].mean() / max(lc[b].mean(), 1e-3):7.3f}")

    print("\nby azimuth (|el| < 15):")
    level = m & (np.abs(d["el"]) < 15)
    for lo in range(-40, 40, 10):
        b = level & (d["az"] >= lo) & (d["az"] < lo + 10)
        if b.sum() < 5:
            continue
        print(f"{lo:+4d}..{lo + 10:+4d}   n {b.sum():4d}  DNa02 L/R {d['dnaL'][b].mean():5.1f}/{d['dnaR'][b].mean():5.1f}  "
              f"LC L/R {d['lcL'][b].mean():5.1f}/{d['lcR'][b].mean():5.1f}  DNp53 {p53[b].mean():5.2f}  song {d['song'][b].mean():5.1f}")

    song = d["song"][m]
    print("\nsong rate distribution (pIP10 L+R, filtered):", {q: round(float(np.percentile(song, q)), 1) for q in (50, 75, 90, 95, 99)})
    off = np.degrees(np.arccos(np.clip(np.cos(np.radians(d["el"])) * np.cos(np.radians(d["az"])), -1, 1)))
    for thr in (8, 12, 16, 20, 25, 30):
        over = m & (d["song"] >= thr)
        print(f"  threshold {thr:2d} Hz: over {100 * over.sum() / m.sum():5.1f}% of time, mean offset when over {off[over].mean() if over.any() else float('nan'):5.1f} deg (all: {off[m].mean():.1f})")

    if s.shape[1] > 19:
        sight = s[:, 19]
        print(f"\nLC10 sight by crosshair offset (gate at {args.sight} Hz):")
        for lo, hi in [(0, 7), (7, 13.3), (13.3, 20), (20, 30), (30, 60), (60, 180)]:
            b = m & (off >= lo) & (off < hi)
            if b.any():
                print(f"  {lo:5.1f}-{hi:5.1f} deg  n {b.sum():5d}  mean {sight[b].mean():5.1f} Hz  open {np.mean(sight[b] >= args.sight):6.1%}")
        on, far = m & (off < 13.3), m & (off > 20)
        print(f"  open in the kill box (< 13.3 deg) {np.mean(sight[on] >= args.sight):.1%}, more than 20 deg off {np.mean(sight[far] >= args.sight):.1%}")


if __name__ == "__main__":
    main()
