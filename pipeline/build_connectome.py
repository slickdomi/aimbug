"""Pack the MaleCNS v1.0 flat connectome into browser-loadable binaries.

Inputs (data/raw/, fetched by pipeline/fetch.sh):
  annotations.feather, neurotransmitters.feather, edges.feather

Outputs (web/public/data/malecns-v1/):
  meta.json          counts, shard table, named neuron groups, eye layout, provenance
  neurons.bin.gz     per-neuron attributes (see NEURON_LAYOUT in meta.json)
  types.json         cell type string table
  edges-NN.bin.gz    CSR shards: [header u32*5][degree varints][target-delta varints][weight varints]

Neuron order: photoreceptors with an assigned eye column come first (so the eye
shader writes a contiguous range), then everything else grouped by
superclass/type/side so CSR target deltas stay small.
"""

import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

RAW = "data/raw"
OUT = os.environ.get("OUT", "web/public/data/malecns-v1")
SHARD_TARGET_BYTES = 18 * 1024 * 1024  # compressed; keeps every file under common static-host limits
MIN_WEIGHT = int(os.environ.get("MIN_WEIGHT", "1"))

# Sign by (presynaptic) transmitter, as in Shiu et al. 2024: ACh excites,
# GABA/glutamate/histamine inhibit. Modulators have no fast sign; treat as excitatory.
NT_SIGN = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "dopamine": 1,
    "serotonin": 1,
    "octopamine": 1,
}

# Named groups the browser needs by index. Values are regexes on `type`.
GROUPS = {
    "DNa01": r"^DNa01$",
    "DNa02": r"^DNa02$",
    "DNp01_GF": r"^DNp01$",
    "DNp20": r"^DNp20$",
    "DNpe017": r"^DNpe017$",
    "pIP10": r"^pIP10$",
    "vPR6": r"^vPR6$",
    "P1_pC1": r"^pC1_",
    "LC10a": r"^LC10a$",
    "LC10": r"^LC10",
    "LC4": r"^LC4$",
    "LPLC2": r"^LPLC2$",
    "MDN": r"^MDN$",
    "KC": r"^KC",
    "MBON11": r"^MBON11$",
    "PPL101": r"^PPL101$",
    "PAM": r"^PAM",
    "L1": r"^L1$",
    "L2": r"^L2$",
    "L3": r"^L3$",
    "HS": r"^HS[NES]$",
    "T4": r"^T4[a-d]$",
    "T5": r"^T5[a-d]$",
}
SIDE_CODE = {"L": 1, "R": 2}
# Optic-lobe cells that are modelled as graded (non-spiking) units in the browser.
GRADED_SUPERCLASSES = ["ol_sensory", "ol_intrinsic"]


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def load_neurons():
    ann = pd.read_feather(f"{RAW}/annotations.feather")
    ann = ann[ann["superclass"].notna() & (ann["status"] != "Glia")].copy()
    ann["type"] = ann["type"].astype("string").fillna("")
    ann["superclass"] = ann["superclass"].astype(str)
    log("neurons", len(ann))

    nt = pd.read_feather(
        f"{RAW}/neurotransmitters.feather",
        columns=["body", "consensus_nt", "celltype_predicted_nt", "predicted_nt"],
    )
    nt = nt[nt["body"].isin(ann["bodyId"])]
    best = nt["consensus_nt"].astype("string")
    for col in ["celltype_predicted_nt", "predicted_nt"]:
        best = best.where(best.notna() & (best != "unclear"), nt[col].astype("string"))
    nt_map = dict(zip(nt["body"], best))
    ann["nt"] = ann["bodyId"].map(nt_map).fillna("unclear")
    ann["sign"] = ann["nt"].map(NT_SIGN).fillna(1).astype(np.int8)
    log("nt", ann["nt"].value_counts().to_dict())

    # Side: prefer soma side; photoreceptors only have rootSide.
    side = ann["somaSide"].astype("string")
    side = side.where(side.isin(["L", "R"]), ann["rootSide"].astype("string"))
    ann["side"] = side.map(SIDE_CODE).fillna(0).astype(np.uint8)
    return ann


def load_edges(body_ids):
    tbl = pf.read_table(f"{RAW}/edges.feather")
    ids = pa.array(body_ids)
    mask = pc.and_(pc.is_in(tbl["body_pre"], value_set=ids), pc.is_in(tbl["body_post"], value_set=ids))
    if MIN_WEIGHT > 1:
        mask = pc.and_(mask, pc.greater_equal(tbl["weight"], MIN_WEIGHT))
    tbl = tbl.filter(mask)
    log("edges", tbl.num_rows)
    return (
        tbl["body_pre"].to_numpy(),
        tbl["body_post"].to_numpy(),
        tbl["weight"].to_numpy().astype(np.int32),
    )


def assign_eye_columns(ann, pre, post, w):
    """Each photoreceptor inherits the hex column of its strongest column-assigned partner."""
    is_r = ann["type"].str.match(r"^R[1-8]").fillna(False).to_numpy()
    r_ids = ann.loc[is_r, "bodyId"].to_numpy()
    hexed = ann[ann["assignedOlHex1"].notna()][["bodyId", "assignedOlHex1", "assignedOlHex2", "side"]]

    m = np.isin(pre, r_ids)
    df = pd.DataFrame({"pre": pre[m], "post": post[m], "w": w[m]})
    df = df.merge(hexed, left_on="post", right_on="bodyId")
    df = df.groupby(["pre", "assignedOlHex1", "assignedOlHex2", "side"], as_index=False)["w"].sum()
    df = df.sort_values("w", ascending=False).drop_duplicates("pre")
    cols = df.set_index("pre")[["assignedOlHex1", "assignedOlHex2", "side"]]

    ann = ann.join(cols.rename(columns={"assignedOlHex1": "eyeHex1", "assignedOlHex2": "eyeHex2", "side": "eyeSide"}), on="bodyId")
    ann["isEye"] = ann["eyeHex1"].notna()
    ann["rKind"] = 0  # 0 none, 1 R1-6, 2 R7, 3 R8
    t = ann["type"]
    ann.loc[ann["isEye"] & t.str.startswith("R1"), "rKind"] = 1
    ann.loc[ann["isEye"] & t.str.startswith("R7"), "rKind"] = 2
    ann.loc[ann["isEye"] & t.str.startswith("R8"), "rKind"] = 3
    log("eye inputs", int(ann["isEye"].sum()), ann.loc[ann["isEye"], "rKind"].value_counts().to_dict())
    return ann


def column_directions(ann):
    """Viewing direction (azimuth, elevation in degrees) of the eye column of every
    column-assigned neuron (photoreceptors use their inferred column); NaN otherwise.

    Lamina cartridges keep retinal order (the first chiasm is after the lamina), so a
    linear fit of L1/L2 soma positions on hex coordinates gives each hex axis's
    anterior/dorsal direction. Brain frame: +x = fly left, +y = ventral, +z = posterior.
    Azimuth: 0 = straight ahead, positive = fly's right. Elevation positive = up.
    """
    t = ann["type"]
    out = np.full((len(ann), 2), np.nan, dtype=np.float32)
    col_side = np.where(ann["isEye"].to_numpy(), ann["eyeSide"].to_numpy(dtype=np.float64), ann["side"].to_numpy(dtype=np.float64))
    fits = {}
    for side_code, side_sign in [(1, -1.0), (2, 1.0)]:
        lam = ann[t.isin(["L1", "L2"]) & (ann["side"] == side_code) & ann["somaLocation"].notna() & ann["assignedOlHex1"].notna()]
        X = np.c_[lam["assignedOlHex1"], lam["assignedOlHex2"], np.ones(len(lam))]
        Y = np.stack(lam["somaLocation"].to_numpy()).astype(np.float64)
        coef, *_ = np.linalg.lstsq(X, np.c_[-Y[:, 2], -Y[:, 1]], rcond=None)  # (anterior, dorsal)
        # Soma positions sit on a curved sheet, so only trust the orientation. Neighbour
        # distances show hex axes 120 deg apart ((1,1) is a nearest neighbour, (1,-1) is
        # sqrt(3) away): hex1+hex2 runs along one lattice axis, hex1-hex2 across it.
        d_hat = coef[0] + coef[1]
        d_hat /= np.linalg.norm(d_hat)
        a_hat = (coef[0] - coef[1]) - np.dot(coef[0] - coef[1], d_hat) * d_hat
        a_hat /= np.linalg.norm(a_hat)
        step = 4.8  # degrees between neighbouring columns
        e1 = step * (d_hat + np.sqrt(3) * a_hat) / 2
        e2 = step * (d_hat - np.sqrt(3) * a_hat) / 2

        cols = ann[(ann["side"] == side_code) & ann["assignedOlHex1"].notna()]
        p_all = np.c_[cols["assignedOlHex1"], cols["assignedOlHex2"]] @ np.stack([e1, e2])
        rows = np.flatnonzero((col_side == side_code) & (ann["hex1"].to_numpy() >= 0))
        h = ann.iloc[rows][["hex1", "hex2"]].to_numpy(dtype=np.float64)
        p = h @ np.stack([e1, e2])
        # Most anterior column looks ~12 deg across the midline (binocular overlap);
        # elevation centred on the column population.
        lateral = (p_all[:, 0].max() - 12.0) - p[:, 0]
        out[rows, 0] = side_sign * lateral
        out[rows, 1] = p[:, 1] - p_all[:, 1].mean()
        fits[int(side_code)] = {
            "hex1": [round(float(v), 3) for v in e1],
            "hex2": [round(float(v), 3) for v in e2],
            "azRange": [float(side_sign * ((p_all[:, 0].max() - 12.0) - p_all[:, 0]).min()), float(side_sign * ((p_all[:, 0].max() - 12.0) - p_all[:, 0]).max())],
            "elRange": [float((p_all[:, 1] - p_all[:, 1].mean()).min()), float((p_all[:, 1] - p_all[:, 1].mean()).max())],
        }
        log("eye fit side", side_code, fits[int(side_code)])
    return out, fits


def receptive_fields(n, src, dst, w, col_dir, hops=8):
    """Connectome-derived receptive field centre for every neuron.

    Seeds every column-assigned neuron with the unit vector of its column's viewing
    direction (the lamina is only partly reconstructed, so photoreceptors alone leave
    most columns unseeded) and repeatedly averages the rest over presynaptic partners
    weighted by input fraction (sign ignored: RF position does not depend on
    transmitter). Returns (azimuth, elevation, strength); strength ~ how visual and
    retinotopically coherent a cell's input is (1 = single direction, 0 = none/wide-field).
    """
    has = ~np.isnan(col_dir[:, 0])
    az, el = np.radians(col_dir[has, 0]), np.radians(col_dir[has, 1])
    seed = np.zeros((n, 3))
    seed[has] = np.c_[np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)]
    total_in = np.bincount(dst, weights=w, minlength=n)
    frac = w / np.maximum(total_in[dst], 1)
    v = seed.copy()
    for _ in range(hops):
        nv = np.stack([np.bincount(dst, weights=frac * v[src, k], minlength=n) for k in range(3)], axis=1)
        nv[has] = seed[has]
        v = nv
    strength = np.linalg.norm(v, axis=1)
    u = v / np.maximum(strength[:, None], 1e-9)
    rf_az = np.degrees(np.arctan2(u[:, 0], u[:, 2]))
    rf_el = np.degrees(np.arcsin(np.clip(u[:, 1], -1, 1)))
    return rf_az.astype(np.float32), rf_el.astype(np.float32), strength.astype(np.float32)


def order_neurons(ann):
    """Eye inputs first, then the rest of the graded (non-spiking) optic lobe, then spiking cells."""
    ann["hex1"] = ann["assignedOlHex1"].fillna(ann["eyeHex1"]).fillna(-1)
    ann["hex2"] = ann["assignedOlHex2"].fillna(ann["eyeHex2"]).fillna(-1)
    ann["graded"] = ann["superclass"].isin(GRADED_SUPERCLASSES)
    assert ann.loc[ann["isEye"], "graded"].all()
    ann["block"] = np.where(ann["isEye"], 0, np.where(ann["graded"], 1, 2))
    ann = ann.sort_values(["block", "superclass", "type", "side", "hex1", "hex2", "bodyId"]).reset_index(drop=True)
    return ann


def varint(values):
    """LEB128-encode a non-negative int64 array."""
    v = values.astype(np.uint64)
    nbytes = np.ones(len(v), dtype=np.int64)
    for k in range(1, 5):
        nbytes += v >= (np.uint64(1) << np.uint64(7 * k))
    out = np.zeros(int(nbytes.sum()), dtype=np.uint8)
    starts = np.concatenate([[0], np.cumsum(nbytes)[:-1]])
    for k in range(5):
        sel = nbytes > k
        byte = ((v[sel] >> np.uint64(7 * k)) & np.uint64(0x7F)).astype(np.uint8)
        byte |= (nbytes[sel] > k + 1).astype(np.uint8) << 7
        out[starts[sel] + k] = byte
    return out


def write_shards(n, src, dst, w):
    order = np.lexsort((dst, src))
    src, dst, w = src[order], dst[order], w[order]
    degree = np.bincount(src, minlength=n).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(degree)])

    row_start = np.searchsorted(src, np.arange(n), side="left")
    prev = np.empty_like(dst)
    prev[1:] = dst[:-1]
    first = np.zeros(len(dst), dtype=bool)
    first[row_start[degree > 0]] = True
    delta = np.where(first, dst, dst - prev).astype(np.int64)

    # Pick shard boundaries by estimated raw size (~3 bytes/edge before gzip, ~0.7 after).
    est_per_edge = 2.2
    bounds = [0]
    acc = 0.0
    for i in range(n):
        acc += degree[i] * est_per_edge + 1
        if acc >= SHARD_TARGET_BYTES:
            bounds.append(i + 1)
            acc = 0.0
    if bounds[-1] != n:
        bounds.append(n)

    shards = []
    for s in range(len(bounds) - 1):
        a, b = bounds[s], bounds[s + 1]
        ea, eb = offsets[a], offsets[b]
        deg_b = varint(degree[a:b])
        dlt_b = varint(delta[ea:eb])
        wt_b = varint(w[ea:eb])
        header = np.array([a, b, eb - ea, len(dlt_b), len(wt_b)], dtype="<u4").tobytes()
        name = f"edges-{s:02d}.bin.gz"
        with open(f"{OUT}/{name}", "wb") as f:
            f.write(gzip.compress(header + deg_b.tobytes() + dlt_b.tobytes() + wt_b.tobytes(), 9))
        size = os.path.getsize(f"{OUT}/{name}")
        shards.append({"file": name, "rowStart": int(a), "rowEnd": int(b), "edges": int(eb - ea), "bytes": size})
        log(name, a, b, eb - ea, f"{size / 1e6:.1f}MB")
    return shards


def main():
    os.makedirs(OUT, exist_ok=True)
    ann = load_neurons()
    pre, post, w = load_edges(ann["bodyId"].to_numpy())
    ann = assign_eye_columns(ann, pre, post, w)
    ann = order_neurons(ann)
    n = len(ann)

    body = ann["bodyId"].to_numpy()
    by_body = np.argsort(body)
    src = by_body[np.searchsorted(body, pre, sorter=by_body)]
    dst = by_body[np.searchsorted(body, post, sorter=by_body)]
    shards = write_shards(n, src, dst, w)

    eye_count = int(ann["isEye"].sum())
    assert ann["isEye"].to_numpy()[:eye_count].all()
    col_dir, eye_fits = column_directions(ann)
    eye_dir = col_dir[:eye_count]
    assert not np.isnan(eye_dir).any()
    eye_blob = eye_dir.astype("<f4").tobytes() + ann["rKind"].to_numpy()[:eye_count].astype(np.uint8).tobytes()
    with open(f"{OUT}/eye.bin.gz", "wb") as f:
        f.write(gzip.compress(eye_blob, 9))
    rf_az, rf_el, rf_str = receptive_fields(n, src, dst, w.astype(np.float64), col_dir)
    for t in ["L1", "Tm1", "T5a", "LC10a", "LC4", "LPLC2", "DNa02", "pIP10", "KCg-m"]:
        m = (ann["type"] == t).to_numpy()
        log(f"rf {t:6s} strength median {np.median(rf_str[m]):.2f}  az range {np.percentile(rf_az[m], 5):.0f}..{np.percentile(rf_az[m], 95):.0f}")

    # Per-neuron attributes. Positions: MaleCNS voxels are 8 nm -> micrometres.
    types = sorted(ann["type"].unique().tolist())
    type_id = ann["type"].map({t: i for i, t in enumerate(types)}).to_numpy().astype(np.uint16)
    superclasses = sorted(ann["superclass"].unique().tolist())
    sc_id = ann["superclass"].map({t: i for i, t in enumerate(superclasses)}).to_numpy().astype(np.uint8)
    pos = np.full((n, 3), np.nan, dtype=np.float32)
    has = ann["somaLocation"].notna().to_numpy()
    pos[has] = np.stack(ann.loc[has, "somaLocation"].to_numpy()).astype(np.float32) * 0.008
    layout = [
        ["bodyId", "u64"], ["type", "u16"], ["superclass", "u8"], ["side", "u8"], ["sign", "i8"],
        ["rKind", "u8"], ["hex1", "i8"], ["hex2", "i8"], ["pos", "f32x3"], ["rf", "f32x3 (azDeg, elDeg, strength)"],
        ["col", "f32x2 (azDeg, elDeg) of eye column, NaN if none"],
        ["nt", "u8 index into meta.transmitters"],
    ]
    transmitters = sorted(ann["nt"].unique().tolist())
    nt_id = ann["nt"].map({t: i for i, t in enumerate(transmitters)}).to_numpy().astype(np.uint8)
    blob = b"".join([
        ann["bodyId"].to_numpy().astype("<u8").tobytes(),
        type_id.astype("<u2").tobytes(),
        sc_id.tobytes(),
        ann["side"].to_numpy().astype(np.uint8).tobytes(),
        ann["sign"].to_numpy().astype(np.int8).tobytes(),
        ann["rKind"].to_numpy().astype(np.uint8).tobytes(),
        ann["hex1"].to_numpy().astype(np.int8).tobytes(),
        ann["hex2"].to_numpy().astype(np.int8).tobytes(),
        pos.astype("<f4").tobytes(),
        np.stack([rf_az, rf_el, rf_str], axis=1).astype("<f4").tobytes(),
        col_dir.astype("<f4").tobytes(),
        nt_id.tobytes(),
    ])
    with open(f"{OUT}/neurons.bin.gz", "wb") as f:
        f.write(gzip.compress(blob, 9))
    with open(f"{OUT}/types.json", "w") as f:
        json.dump(types, f, separators=(",", ":"))

    groups = {}
    for name, pat in GROUPS.items():
        idx = np.flatnonzero(ann["type"].str.match(pat).fillna(False).to_numpy())
        groups[name] = {"indices": idx.tolist(), "sides": ann["side"].to_numpy()[idx].tolist()}
        log("group", name, len(idx))
    dn = np.flatnonzero(ann["superclass"].str.startswith("descending_neuron").to_numpy())
    groups["descending"] = {"indices": dn.tolist(), "sides": ann["side"].to_numpy()[dn].tolist()}

    col = ann[ann["assignedOlHex1"].notna()]
    meta = {
        "dataset": "MaleCNS v1.0 (minconf 0.5)",
        "source": "https://male-cns.janelia.org/download/",
        "license": "CC-BY 4.0, Janelia FlyEM / Google MaleCNS team",
        "neurons": n,
        "edges": int(len(w)),
        "synapses": int(w.sum()),
        "minWeight": MIN_WEIGHT,
        "neuronLayout": layout,
        "superclasses": superclasses,
        "transmitters": transmitters,
        "shards": shards,
        "graded": {"superclasses": GRADED_SUPERCLASSES, "count": int(ann["graded"].sum())},
        "eye": {
            "count": eye_count,
            "hex1Range": [int(col["assignedOlHex1"].min()), int(col["assignedOlHex1"].max())],
            "hex2Range": [int(col["assignedOlHex2"].min()), int(col["assignedOlHex2"].max())],
            "layout": "f32 [azimuthDeg, elevationDeg] * count, then u8 rKind * count (1=R1-6, 2=R7, 3=R8)",
            "fits": eye_fits,
        },
        "groups": groups,
    }
    with open(f"{OUT}/meta.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    total = sum(s["bytes"] for s in shards)
    log(f"done: {n} neurons, {len(w)} edges, {total / 1e6:.1f}MB edge shards")


if __name__ == "__main__":
    main()
