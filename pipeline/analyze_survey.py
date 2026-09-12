"""Which cells tell a target in the crosshair from one off to the side? For choosing the trigger's sight.

Reads a survey from web/scripts/smoke.mjs (SURVEY=1): calibration mode with the fly held still, targets
teleporting around the crosshair, and the spike trace of every spiking cell type (per side) plus every
descending neuron and LC10/LC4/LPLC2/LC9/LC11 cell every 0.5 s. Compares samples with the target inside the
kill box (< 13.3 deg) against samples more than 20 deg away.

  QUERY='?mode=calib&yaw=0&pitch=0&trigger=1000&calibSpread=25&seed=1' SURVEY=1 SECONDS=300 ... smoke.mjs
  python pipeline/analyze_survey.py .cache/survey
"""

import argparse
import gzip
import json

import numpy as np

DATA = "web/public/data/malecns-v1"
HZ = 1 / 0.12  # spike trace time constant (MODEL.actTau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("survey")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    sv = json.load(open(f"{args.survey}/survey.json"))
    labels, meta = sv["labels"], sv["meta"]
    X = np.fromfile(f"{args.survey}/survey.bin", dtype="<f4").reshape(len(meta), len(labels)) * HZ
    off = np.hypot([m["az"] for m in meta], [m["el"] for m in meta])
    ok = np.array([m["alive"] and m["age"] > 450 for m in meta])
    on, far = ok & (off < 13.3), ok & (off > 20)
    col = {label: i for i, label in enumerate(labels)}
    print(f"{ok.sum()} usable samples: {on.sum()} in the kill box, {far.sum()} more than 20 deg off")

    def score(sig):
        """AUC (kill box vs > 20 deg) and the share of off-centre samples above the threshold that catches 80 % on target."""
        a, b = sig[on], sig[far]
        auc = (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()
        return auc, (b >= np.percentile(a, 20)).mean(), a.mean(), b.mean()

    rows = []
    for t in sorted({label.rsplit(":", 1)[0] for label in labels if not label.startswith("#")}):
        sig = X[:, [col[f"{t}:{s}"] for s in "LR?" if f"{t}:{s}" in col]].sum(1)
        if sig[on].mean() >= 3:
            rows.append((f"type {t} (both sides)", *score(sig)))

    # Groups picked by anatomy: LC cells whose connectome receptive field points near straight ahead.
    n = json.load(open(f"{DATA}/meta.json"))["neurons"]
    rf = np.frombuffer(gzip.open(f"{DATA}/neurons.bin.gz").read(), dtype="<f4", count=3 * n, offset=28 * n).reshape(n, 3)
    single = [(i, int(label.split(" ")[0][1:]), label.split(" ")[1].rsplit(":", 1)[0]) for i, label in enumerate(labels) if label.startswith("#")]
    for name, pick in {"LC10": lambda t: t.startswith("LC10"), "LC4+LPLC2": lambda t: t in ("LC4", "LPLC2"),
                       "LC10+LC4+LPLC2+LC11": lambda t: t.startswith("LC10") or t in ("LC4", "LPLC2", "LC11")}.items():
        for radius in (10, 15, 20, 25):
            cs = [c for c, cell, t in single if pick(t) and rf[cell, 2] > 0 and np.hypot(*rf[cell, :2]) < radius]
            if cs:
                rows.append((f"{name}, receptive field < {radius} deg ({len(cs)} cells)", *score(X[:, cs].sum(1))))
    rows.append(("pIP10 song (both sides)", *score(X[:, [col["pIP10:L"], col["pIP10:R"]]].sum(1))))

    print(f"\n{'signal':52s} {'AUC':>5s} {'false@80%':>9s} {'on Hz':>7s} {'off Hz':>7s}")
    for r in sorted(rows, key=lambda r: (r[2], -r[1]))[: args.top]:
        print(f"{r[0][:52]:52s} {r[1]:5.2f} {r[2]:9.1%} {r[3]:7.1f} {r[4]:7.1f}")
    r = rows[-1]
    print(f"{r[0][:52]:52s} {r[1]:5.2f} {r[2]:9.1%} {r[3]:7.1f} {r[4]:7.1f}")


if __name__ == "__main__":
    main()
