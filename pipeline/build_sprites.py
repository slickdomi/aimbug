"""Cut female Drosophila photos out of their light backgrounds into a sprite atlas.

Inputs: data/raw/sprites/*.jpg (see SOURCES; fetched by pipeline/fetch_sprites.sh)
Output: web/public/sprites/females.png (RGBA, square cells) + females.json (cells, credits)

Matting: the background (a light, smoothly lit surface) is fitted as a low-order
polynomial to pixels that look like background; alpha comes from how far a pixel's
colour and saturation are from it (so wings stay translucent), then only the connected
fly is kept, colours are un-premultiplied against the background and bled into the
transparent area so mipmaps don't get dark fringes.
"""

import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage

RAW = "data/raw/sprites"
OUT = "web/public/sprites"
CELL = 512

SOURCES = [
    {
        "file": "brecher_female_side.jpg",
        "crop": (0.02, 0.03, 0.98, 0.97),  # drop the photo's frame
        "credit": "Rolf Dietrich Brecher, \"Drosophila melanogaster ♀\"",
        "license": "CC BY 2.0",
        "licenseUrl": "https://creativecommons.org/licenses/by/2.0/",
        "url": "https://commons.wikimedia.org/wiki/File:Drosophila_melanogaster_%E2%99%80_(38978426500).jpg",
    },
    {
        "file": "davis_female_standing.jpg",
        "crop": (0.0, 0.0, 1.0, 1.0),
        "credit": "Hannah Davis, \"Standing female Drosophila melanogaster\"",
        "license": "CC BY-SA 4.0",
        "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/",
        "url": "https://commons.wikimedia.org/wiki/File:Standing_female_Drosophila_melanogaster.jpg",
    },
]


def fit_background(rgb, iterations=4):
    """Per-channel quadratic surface fitted to the pixels closest to it (robust re-weighting)."""
    h, w, _ = rgb.shape
    step = 8
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    sample = rgb[::step, ::step].reshape(-1, 3)
    u = (xs.ravel() / w) * 2 - 1
    v = (ys.ravel() / h) * 2 - 1
    basis = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, v ** 3, u * u * v, u * v * v], 1)
    sat = sample.max(1) - sample.min(1)
    good = (sat < np.percentile(sat, 60)) & (sample.mean(1) > np.percentile(sample.mean(1), 30))
    for _ in range(iterations):
        coef, *_ = np.linalg.lstsq(basis[good], sample[good], rcond=None)
        resid = np.abs(sample - basis @ coef).max(1)
        good = resid < max(np.percentile(resid[good], 80), 0.02)
    yy, xx = np.mgrid[0:h, 0:w]
    U = (xx / w) * 2 - 1
    V = (yy / h) * 2 - 1
    full = np.stack([np.ones_like(U), U, V, U * U, U * V, V * V, U ** 3, V ** 3, U * U * V, U * V * V], -1)
    return np.clip(full @ coef, 0, 1).astype(np.float32)


def bleed(fg, alpha, passes=24):
    """Fill colours under transparent pixels with nearby opaque colours."""
    w = (alpha > 0.2).astype(np.float32)
    col = fg * w[..., None]
    for _ in range(passes):
        wb = ndimage.uniform_filter(w, 5)
        cb = np.stack([ndimage.uniform_filter(col[..., k], 5) for k in range(3)], -1)
        fill = (w == 0) & (wb > 1e-4)
        col[fill] = cb[fill] / wb[fill, None]
        w = np.where(fill, 1.0, w)
        col = fg * (alpha > 0.2)[..., None] + col * (alpha <= 0.2)[..., None]
    return np.where((alpha > 0.2)[..., None], fg, col)


def matte(img):
    rgb = img.astype(np.float32) / 255.0
    bg = fit_background(rgb)

    sat = rgb.max(-1) - rgb.min(-1)
    bg_sat = bg.max(-1) - bg.min(-1)
    dlum = np.abs(rgb.mean(-1) - bg.mean(-1))
    dsat = np.maximum(sat - bg_sat, 0)
    score = np.maximum(dsat / 0.18, dlum / 0.35)
    alpha = np.clip((score - 0.25) / 0.75, 0, 1)
    alpha = ndimage.gaussian_filter(alpha, 1.0)

    # Keep the fly: the largest solid blob plus anything touching its generous surroundings
    # (legs, wings); drops dust, frame remnants and stray shadows.
    solid = alpha > 0.5
    labels, n = ndimage.label(solid)
    sizes = ndimage.sum(solid, labels, range(1, n + 1))
    body = labels == (np.argmax(sizes) + 1)
    reach = ndimage.binary_dilation(body, iterations=max(8, rgb.shape[1] // 60))
    weak_labels, _ = ndimage.label(alpha > 0.15)
    keep_ids = np.unique(weak_labels[reach & (weak_labels > 0)])
    keep = np.isin(weak_labels, keep_ids) & reach
    keep = ndimage.binary_dilation(keep, iterations=2)
    alpha = alpha * keep

    a = np.maximum(alpha[..., None], 0.08)
    fg = np.clip((rgb - bg * (1 - alpha[..., None])) / a, 0, 1)
    return fg, alpha, bg


def main():
    os.makedirs(OUT, exist_ok=True)
    atlas = Image.new("RGBA", (CELL * len(SOURCES), CELL), (0, 0, 0, 0))
    cells = []
    for i, src in enumerate(SOURCES):
        im = Image.open(f"{RAW}/{src['file']}").convert("RGB")
        w, h = im.size
        x0, y0, x1, y1 = src["crop"]
        im = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        # Matte at reduced resolution: the photos are far larger than a 512 px sprite.
        im.thumbnail((1600, 1600), Image.LANCZOS)
        fg, alpha, bg = matte(np.array(im))
        ys, xs = np.nonzero(alpha > 0.1)
        pad = 12
        bx0, bx1 = max(xs.min() - pad, 0), min(xs.max() + pad, alpha.shape[1])
        by0, by1 = max(ys.min() - pad, 0), min(ys.max() + pad, alpha.shape[0])
        fg, alpha = fg[by0:by1, bx0:bx1], alpha[by0:by1, bx0:bx1]
        rgba = np.dstack([bleed(fg, alpha), alpha])
        sprite = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")
        scale = (CELL - 8) / max(sprite.size)
        sprite = sprite.resize((max(1, round(sprite.size[0] * scale)), max(1, round(sprite.size[1] * scale))), Image.LANCZOS)
        ox = i * CELL + (CELL - sprite.size[0]) // 2
        oy = (CELL - sprite.size[1]) // 2
        atlas.alpha_composite(sprite, (ox, oy))
        cells.append({
            # sprite rect inside its cell, in cell-uv units
            "x": i, "u0": (ox - i * CELL) / CELL, "v0": oy / CELL,
            "u1": (ox - i * CELL + sprite.size[0]) / CELL, "v1": (oy + sprite.size[1]) / CELL,
            **{k: src[k] for k in ("credit", "license", "licenseUrl", "url")},
            "modifications": "background removed, cropped, resized",
        })
        print(src["file"], "sprite", sprite.size, "coverage", float((alpha > 0.5).mean()))
    atlas.save(f"{OUT}/females.png", optimize=True)
    with open(f"{OUT}/females.json", "w") as f:
        json.dump({"cellSize": CELL, "cells": cells}, f, indent=1, ensure_ascii=False)
    # Preview on a dark and a light background for eyeballing the matte.
    prev = Image.new("RGBA", (atlas.size[0], CELL * 2), (30, 34, 44, 255))
    prev.paste(Image.new("RGBA", (atlas.size[0], CELL), (232, 236, 244, 255)), (0, CELL))
    prev.alpha_composite(atlas, (0, 0))
    prev.alpha_composite(atlas, (0, CELL))
    prev.convert("RGB").save(f"{RAW}/atlas_preview.jpg", quality=90)


if __name__ == "__main__":
    main()
