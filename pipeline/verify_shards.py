"""Decode edge shards and compare a sample of rows against the raw feather table."""

import gzip
import json
import os

import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as pf

OUT = os.environ.get("OUT", "web/public/data/malecns-v1")


def read_varints(buf, count, pos):
    vals = np.empty(count, dtype=np.int64)
    for i in range(count):
        v = 0
        shift = 0
        while True:
            b = buf[pos]
            pos += 1
            v |= (b & 0x7F) << shift
            if b < 0x80:
                break
            shift += 7
        vals[i] = v
    return vals, pos


meta = json.load(open(f"{OUT}/meta.json"))
n = meta["neurons"]
body = np.frombuffer(gzip.open(f"{OUT}/neurons.bin.gz").read()[: 8 * n], dtype="<u8")

rng = np.random.default_rng(0)
shard = meta["shards"][1]
raw = gzip.open(f"{OUT}/{shard['file']}").read()
a, b, ne, dlen, wlen = np.frombuffer(raw[:20], dtype="<u4")
deg, pos = read_varints(raw, b - a, 20)
assert deg.sum() == ne
off = np.concatenate([[0], np.cumsum(deg)])
# Decode only the first ~200k edges' worth of rows for speed.
rows = int(np.searchsorted(off, 200_000))
dlt, _ = read_varints(raw, int(off[rows]), pos)
wts, _ = read_varints(raw, int(off[rows]), pos + int(dlen))

tbl = pf.read_table("data/raw/edges.feather")
for r in rng.choice(rows, 25, replace=False):
    tg = np.cumsum(dlt[off[r]:off[r + 1]])
    got = dict(zip(body[tg].tolist(), wts[off[r]:off[r + 1]].tolist()))
    pre_body = int(body[a + r])
    sub = tbl.filter(pc.equal(tbl["body_pre"], pre_body)).to_pandas()
    sub = sub[sub["body_post"].isin(body)]
    want = dict(zip(sub["body_post"].tolist(), sub["weight"].tolist()))
    assert got == want, (r, pre_body, len(got), len(want))
print("ok: 25 sampled rows match")
