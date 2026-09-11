"""Which descending neurons does each source group reach, and how lateralised is it?

Propagates a signed "input fraction" (w / total input of postsynaptic cell) from a
source group for a few hops and ranks descending neurons by accumulated influence.
"""

import sys

import numpy as np
import pandas as pd
import pyarrow.feather as pf

HOPS = 3
SOURCES = sys.argv[1:] or ["LC10a", "LC4", "LPLC2", "R1-R6"]
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}

ann = pd.read_feather("data/raw/annotations.feather")
ann = ann[ann["superclass"].notna() & (ann["status"] != "Glia")].reset_index(drop=True)
nt = pd.read_feather("data/raw/neurotransmitters.feather", columns=["body", "consensus_nt", "celltype_predicted_nt"])
best = nt["consensus_nt"].astype("string")
best = best.where(best != "unclear", nt["celltype_predicted_nt"].astype("string"))
sign_by_body = dict(zip(nt["body"], best.map(NT_SIGN).fillna(1)))

body = ann["bodyId"].to_numpy()
order = np.argsort(body)
e = pf.read_table("data/raw/edges.feather").to_pandas()
e = e[np.isin(e["body_pre"], body) & np.isin(e["body_post"], body)]
pre = order[np.searchsorted(body, e["body_pre"].to_numpy(), sorter=order)]
post = order[np.searchsorted(body, e["body_post"].to_numpy(), sorter=order)]
w = e["weight"].to_numpy().astype(np.float64)
n = len(body)
total_in = np.bincount(post, weights=w, minlength=n)
sign = pd.Series(body).map(sign_by_body).fillna(1).to_numpy()
frac = w / np.maximum(total_in[post], 1) * sign[pre]

side = ann["somaSide"].astype("string").where(ann["somaSide"].isin(["L", "R"]), ann["rootSide"].astype("string")).fillna("?").to_numpy()
types = ann["type"].astype("string").fillna("").to_numpy()
is_dn = ann["superclass"].astype(str).str.startswith("descending_neuron").to_numpy()

for src in SOURCES:
    res = {}
    for s in ["L", "R"]:
        x = ((types == src) & (side == s)).astype(np.float64)
        if x.sum() == 0:
            continue
        x /= x.sum()
        acc = np.zeros(n)
        for _ in range(HOPS):
            x = np.bincount(post, weights=frac * x[pre], minlength=n)
            acc += x
        res[s] = acc
    if len(res) < 2:
        print(src, "missing a side")
        continue
    df = pd.DataFrame({"type": types, "side": side, "L": res["L"], "R": res["R"]})[is_dn]
    df["total"] = df["L"].abs() + df["R"].abs()
    df["lat"] = (df["L"] - df["R"]) / df["total"].replace(0, np.nan)  # +1 = driven by left source only
    print(f"\n== {src}: top descending neurons ({HOPS} hops, signed input fraction) ==")
    print(df.sort_values("total", ascending=False).head(25).to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    for t in ["DNa01", "DNa02", "pIP10", "DNp01", "DNp20", "DNpe017"]:
        print(df[df["type"] == t].to_string(index=False, header=False, float_format=lambda v: f"{v:.2e}"))
